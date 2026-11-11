// nn::os (y su API en C, nnos*) sobre pthreads. Cada objeto del juego se asocia por su
// dirección a un objeto nuestro; si el juego usa uno sin inicializarlo (inicializador
// estático), se crea al vuelo con valores seguros (mutex recursivo, evento manual).

#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/os.h"
#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

using namespace nn;
using namespace nn::os;

namespace {

int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

timespec deadline_after(int64_t ns) {
    int64_t t = now_ns() + (ns > 0 ? ns : 0);
    return {static_cast<time_t>(t / 1'000'000'000), static_cast<long>(t % 1'000'000'000)};
}

// Objeto del juego (por dirección) -> objeto nuestro. 64 trozos para no serializar todo.
template <typename T>
class Registry {
public:
    T& get(const void* key) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        auto& slot = s.map[key];
        if (!slot) slot = std::make_unique<T>();
        return *slot;
    }
    template <typename... Args>
    T& reset(const void* key, Args&&... args) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        auto& slot = s.map[key];
        slot = std::make_unique<T>(std::forward<Args>(args)...);
        return *slot;
    }
    void erase(const void* key) {
        Shard& s = shard(key);
        std::lock_guard lock(s.m);
        s.map.erase(key);
    }
    template <typename Fn>
    void for_each(Fn fn) {
        for (Shard& s : shards_) {
            std::lock_guard lock(s.m);
            for (auto& [key, value] : s.map) fn(*value);
        }
    }

private:
    struct Shard {
        std::mutex m;
        std::unordered_map<const void*, std::unique_ptr<T>> map;
    };
    Shard& shard(const void* key) { return shards_[(reinterpret_cast<uintptr_t>(key) >> 4) % 64]; }
    Shard shards_[64];
};

struct HostMutex {
    pthread_mutex_t m;
    explicit HostMutex(bool recursive = true) {
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_settype(&a, recursive ? PTHREAD_MUTEX_RECURSIVE : PTHREAD_MUTEX_NORMAL);
        pthread_mutex_init(&m, &a);
    }
};

struct HostCond {
    pthread_cond_t c;
    HostCond() {
        pthread_condattr_t a;
        pthread_condattr_init(&a);
        pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
        pthread_cond_init(&c, &a);
    }
    // false si venció el plazo.
    bool wait(pthread_mutex_t* m, int64_t timeout_ns = -1) {
        if (timeout_ns < 0) return pthread_cond_wait(&c, m), true;
        timespec ts = deadline_after(timeout_ns);
        return pthread_cond_timedwait(&c, m, &ts) != ETIMEDOUT;
    }
};

// Cualquier evento señalado despierta a quien espere en WaitAny. Con el reloj monotónico, como los plazos de
// deadline_after: con PTHREAD_COND_INITIALIZER (tiempo real) el plazo quedaba en el pasado, cada espera
// vencía al momento y el hilo del vídeo gastaba un núcleo entero.
HostCond g_any_cond;
pthread_mutex_t g_any_pm = PTHREAD_MUTEX_INITIALIZER;
std::atomic<uint64_t> g_any_gen{0};

void notify_any() {
    pthread_mutex_lock(&g_any_pm);
    g_any_gen++;
    pthread_cond_broadcast(&g_any_cond.c);
    pthread_mutex_unlock(&g_any_pm);
}

struct HostEvent {
    HostMutex lock{false};
    HostCond cond;
    bool signaled = false;
    bool auto_clear = false;

    void signal() {
        pthread_mutex_lock(&lock.m);
        signaled = true;
        auto_clear ? pthread_cond_signal(&cond.c) : pthread_cond_broadcast(&cond.c);
        pthread_mutex_unlock(&lock.m);
        notify_any();
    }
    bool wait(int64_t timeout_ns = -1) {
        int64_t end = timeout_ns < 0 ? -1 : now_ns() + timeout_ns;
        pthread_mutex_lock(&lock.m);
        while (!signaled) {
            if (end < 0) {
                cond.wait(&lock.m);
                continue;
            }
            // Ojo: con plazo vencido el resto sale negativo, y para HostCond negativo es "sin plazo".
            int64_t left = end - now_ns();
            if (left <= 0 || !cond.wait(&lock.m, left)) break;
        }
        bool got = signaled;
        if (got && auto_clear) signaled = false;
        pthread_mutex_unlock(&lock.m);
        return got;
    }
    bool try_take() { return wait(0); }
    void clear() {
        pthread_mutex_lock(&lock.m);
        signaled = false;
        pthread_mutex_unlock(&lock.m);
    }
};

struct HostSemaphore {
    HostMutex lock{false};
    HostCond cond;
    int count = 0, max = 1;
};

struct HostThread {
    ThreadType* type = nullptr;
    ThreadFunction fn = nullptr;
    void* arg = nullptr;
    size_t stack_size = 0;
    int priority = 16, ideal_core = 0;
    uint64_t core_mask = 0;
    const char* name = "";
    pthread_t pthread{};
    pid_t tid = 0;
    bool started = false, joined = false;
};

struct HostMultiWait {
    std::mutex m;
    std::vector<MultiWaitHolderType*> holders;
};

struct HostHolder {
    EventType* event = nullptr;
    MultiWaitType* owner = nullptr;
};

Registry<HostMutex> g_mutexes;
Registry<HostCond> g_conds;
Registry<HostEvent> g_events;
Registry<HostSemaphore> g_semaphores;
Registry<HostThread> g_threads;
Registry<HostMultiWait> g_multiwaits;
Registry<HostHolder> g_holders;

std::mutex g_regions_m;
std::unordered_map<uintptr_t, size_t> g_regions;

thread_local ThreadType* t_current = nullptr;

// Byte de estado dentro del ThreadType del juego. Hay código que lo lee en línea (cabeceras del
// SDK) en vez de preguntar a nn::os: el device de E/S de Wwise guarda su ThreadType en +0x10 y,
// tras StartThread, da el hilo por válido si el byte +0x50 del device (ThreadType+0x40) no es 0.
constexpr size_t kThreadStateOffset = 0x40;
enum ThreadState : uint8_t { NotInitialized = 0, Initialized = 1, Started = 3, Terminated = 4 };

void set_state(ThreadType* type, ThreadState state) {
    reinterpret_cast<volatile uint8_t*>(type)[kThreadStateOffset] = state;
}

// --- Afinidad de los hilos críticos ---
// El frame es una cadena: el hilo principal del juego (envía y presenta), MainDraw (graba los comandos) y
// NvnSubmit (los reproduce en Vulkan). Sin afinidad, el planificador los podía dejar en los núcleos pequeños:
// se limitan a los grandes (los de frecuencia máxima por encima de la mínima; en el SD865, cpu4-7) con una
// máscara y no un núcleo fijo, porque core_ctl puede aislar alguno. Los demás hilos no se tocan, salvo que un
// hilo nuevo hereda la máscara de quien lo crea: los de nn::os que no son críticos vuelven a la completa al
// arrancar, y los demás en la revisión de cada segundo. Android rehace la afinidad al mover el proceso de
// cpuset (segundo plano y vuelta): refresh_critical_threads la vuelve a poner. Interruptor sin_afinidad:
// máscara completa para los hilos críticos; lo mira refresh_critical_threads como mucho una vez por segundo,
// desde el hilo que presenta, y lo aplica a todos (sched_setaffinity admite el tid de otro hilo del proceso).
struct CpuMasks {
    cpu_set_t big, all;
    bool usable = false;  // hay núcleos de al menos dos frecuencias máximas
};

uint64_t mask_bits(const cpu_set_t& set) {  // los 64 primeros, para el log
    uint64_t bits = 0;
    for (int cpu = 0; cpu < 64; cpu++)
        if (CPU_ISSET(cpu, &set)) bits |= uint64_t{1} << cpu;
    return bits;
}

const CpuMasks& cpu_masks() {
    static const CpuMasks masks = [] {
        CpuMasks m;
        CPU_ZERO(&m.big);
        CPU_ZERO(&m.all);
        const long n = std::min<long>(sysconf(_SC_NPROCESSORS_CONF), 64);
        std::vector<unsigned long> freq(n > 0 ? n : 0, 0);
        unsigned long lowest = ~0ul;
        for (int cpu = 0; cpu < n; cpu++) {
            CPU_SET(cpu, &m.all);
            char path[80];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
            if (FILE* f = fopen(path, "r")) {
                if (fscanf(f, "%lu", &freq[cpu]) != 1) freq[cpu] = 0;
                fclose(f);
            }
            if (freq[cpu] && freq[cpu] < lowest) lowest = freq[cpu];
        }
        for (int cpu = 0; cpu < n; cpu++)
            if (freq[cpu] && freq[cpu] > lowest) CPU_SET(cpu, &m.big);
        m.usable = CPU_COUNT(&m.big) > 0;
        printf("[nnsdk] afinidad: núcleos grandes %#llx de %#llx%s\n", static_cast<unsigned long long>(mask_bits(m.big)),
               static_cast<unsigned long long>(mask_bits(m.all)), m.usable ? "" : " (todos iguales o sin datos: no se usa)");
        return m;
    }();
    return masks;
}

bool affinity_off() { return NVN_FLAG("sin_afinidad"); }

struct CriticalThread {
    pid_t tid;
    char name[16];
};
std::mutex g_critical_m;
std::vector<CriticalThread> g_critical;
std::atomic<bool> g_any_critical{false};
thread_local bool t_critical = false;

// Pone la máscara si no la tiene ya; false si el hilo ya no existe. Si el sistema no deja, se queda como está.
bool apply_mask(const CriticalThread& t, const cpu_set_t& want, bool refreshing) {
    cpu_set_t now;
    CPU_ZERO(&now);
    if (sched_getaffinity(t.tid, sizeof now, &now) == 0 && CPU_EQUAL(&now, &want)) return true;
    if (sched_setaffinity(t.tid, sizeof want, &want) != 0) {
        if (errno == ESRCH) return false;
        static std::atomic<bool> failed_logged{false};
        if (!failed_logged.exchange(true))
            printf("[nnsdk] afinidad: no se pudo poner la de '%s' (%s); se deja como está\n", t.name, strerror(errno));
        return true;
    }
    static std::atomic<int> lost_logged{0};  // DEPURACIÓN: ¿quién nos la quita?
    if (refreshing && lost_logged.fetch_add(1) < 5)
        printf("[nnsdk] afinidad de '%s' perdida (era %#llx): se vuelve a poner\n", t.name,
               static_cast<unsigned long long>(mask_bits(now)));
    return true;
}

void pin_current_thread() {
    const CpuMasks& m = cpu_masks();
    if (!m.usable || t_critical) return;
    t_critical = true;
    CriticalThread t{gettid(), {}};
    pthread_getname_np(pthread_self(), t.name, sizeof t.name);
    const bool off = affinity_off();
    std::lock_guard lock(g_critical_m);
    apply_mask(t, off ? m.all : m.big, false);
    g_critical.push_back(t);
    g_any_critical = true;
    printf("[nnsdk] hilo crítico '%s' (tid %d): %s\n", t.name, t.tid, off ? "sin afinidad (sin_afinidad)" : "núcleos grandes");
}

void unpin_current_thread() {  // el hilo termina: su tid se puede reutilizar
    if (!t_critical) return;
    t_critical = false;
    const pid_t tid = gettid();
    std::lock_guard lock(g_critical_m);
    std::erase_if(g_critical, [&](const CriticalThread& t) { return t.tid == tid; });
}

// Un hilo nuevo hereda la máscara de quien lo crea: si fue un hilo crítico, se le devuelve la completa.
void release_inherited_affinity() {
    if (!g_any_critical) return;  // nadie tiene máscara: nada que heredar
    const CpuMasks& m = cpu_masks();
    if (!m.usable) return;
    cpu_set_t now;
    CPU_ZERO(&now);
    if (sched_getaffinity(0, sizeof now, &now) == 0 && !CPU_EQUAL(&now, &m.all)) sched_setaffinity(0, sizeof m.all, &m.all);
}

// Lo mismo para los hilos que no son de nn::os (del driver de Vulkan, de AAudio, el que guarda la caché de
// pipelines) si los crea un hilo crítico: una vez por segundo, los que no son críticos y tienen justo la máscara
// de los grandes (la heredada) vuelven a la completa. Con g_critical_m.
void release_inherited_masks(const CpuMasks& m) {
    DIR* dir = opendir("/proc/self/task");
    if (!dir) return;
    while (const dirent* e = readdir(dir)) {
        const pid_t tid = static_cast<pid_t>(atoi(e->d_name));
        if (tid <= 0 || std::any_of(g_critical.begin(), g_critical.end(), [&](const CriticalThread& t) { return t.tid == tid; }))
            continue;
        cpu_set_t now;
        CPU_ZERO(&now);
        if (sched_getaffinity(tid, sizeof now, &now) == 0 && CPU_EQUAL(&now, &m.big))
            sched_setaffinity(tid, sizeof m.all, &m.all);
    }
    closedir(dir);
}

bool critical_name(const char* name) { return strcmp(name, "MainDraw") == 0; }

void* thread_entry(void* p) {
    auto* t = static_cast<HostThread*>(p);
    t_current = t->type;
    t->tid = gettid();
    char name[16];  // Linux admite 15 caracteres: con más, el nombre no se pone (ManaSwitchDecoderWatchThread)
    snprintf(name, sizeof name, "%s", t->name[0] ? t->name : "nn::os");
    pthread_setname_np(pthread_self(), name);
    if (critical_name(t->name))
        pin_current_thread();
    else
        release_inherited_affinity();
    nnsdk::on_thread_start();
    t->fn(t->arg);
    unpin_current_thread();
    set_state(t->type, Terminated);
    return nullptr;
}

// Hilos que no creó nn::os (el principal): se les da un ThreadType propio.
ThreadType* current_thread() {
    if (!t_current) {
        auto* fake = static_cast<ThreadType*>(calloc(1, 0x400));
        HostThread& t = g_threads.reset(fake);
        t.type = fake;
        t.pthread = pthread_self();
        t.tid = gettid();
        t.started = true;
        t.name = "MainThread";
        set_state(fake, Started);
        t_current = fake;
    }
    return t_current;
}

Result create_thread(ThreadType* type, ThreadFunction fn, void* arg, size_t stack_size, int priority, int core) {
    HostThread& t = g_threads.reset(type);
    t.type = type;
    t.fn = fn;
    t.arg = arg;
    t.stack_size = stack_size;
    t.priority = priority;
    t.ideal_core = core;
    set_state(type, Initialized);
    return ResultSuccess;
}

}  // namespace

namespace nn::os {

// --- Mutex, variables de condición, RW lock ---

void InitializeMutex(MutexType* m, bool recursive, int) { g_mutexes.reset(m, recursive); }
void FinalizeMutex(MutexType* m) { g_mutexes.erase(m); }
void LockMutex(MutexType* m) { pthread_mutex_lock(&g_mutexes.get(m).m); }
bool TryLockMutex(MutexType* m) { return pthread_mutex_trylock(&g_mutexes.get(m).m) == 0; }
void UnlockMutex(MutexType* m) { pthread_mutex_unlock(&g_mutexes.get(m).m); }

void InitializeConditionVariable(ConditionVariableType* c) { g_conds.reset(c); }
void FinalizeConditionVariable(ConditionVariableType* c) { g_conds.erase(c); }
void SignalConditionVariable(ConditionVariableType* c) { pthread_cond_signal(&g_conds.get(c).c); }
void BroadcastConditionVariable(ConditionVariableType* c) { pthread_cond_broadcast(&g_conds.get(c).c); }
void WaitConditionVariable(ConditionVariableType* c, MutexType* m) { g_conds.get(c).wait(&g_mutexes.get(m).m); }
ConditionVariableStatus TimedWaitConditionVariable(ConditionVariableType* c, MutexType* m, TimeSpan t) {
    return g_conds.get(c).wait(&g_mutexes.get(m).m, t.ns) ? ConditionVariableStatus_NoTimeout
                                                          : ConditionVariableStatus_Timeout;
}

// Solo se usa el lado de escritura; el dueño puede volver a entrar.
void InitializeReaderWriterLock(ReaderWriterLockType* l) { g_mutexes.reset(l); }
void FinalizeReaderWriterLock(ReaderWriterLockType* l) { g_mutexes.erase(l); }
void AcquireWriteLock(ReaderWriterLockType* l) { pthread_mutex_lock(&g_mutexes.get(l).m); }
void ReleaseWriteLock(ReaderWriterLockType* l) { pthread_mutex_unlock(&g_mutexes.get(l).m); }

// --- Eventos, eventos de sistema, semáforos, MultiWait ---

void InitializeEvent(EventType* e, bool signaled, EventClearMode mode) {
    HostEvent& ev = g_events.reset(e);
    ev.signaled = signaled;
    ev.auto_clear = mode == EventClearMode_AutoClear;
}
void FinalizeEvent(EventType* e) { g_events.erase(e); }
void SignalEvent(EventType* e) { g_events.get(e).signal(); }
void WaitEvent(EventType* e) { g_events.get(e).wait(); }
bool TimedWaitEvent(EventType* e, TimeSpan t) { return g_events.get(e).wait(t.ns); }
void ClearEvent(EventType* e) { g_events.get(e).clear(); }

void DestroySystemEvent(SystemEventType* e) { g_events.erase(e); }
void WaitSystemEvent(SystemEventType* e) { g_events.get(e).wait(); }
bool TryWaitSystemEvent(SystemEventType* e) { return g_events.get(e).try_take(); }
bool TimedWaitSystemEvent(SystemEventType* e, TimeSpan t) { return g_events.get(e).wait(t.ns); }

void InitializeSemaphore(SemaphoreType* s, int initial, int max) {
    HostSemaphore& sem = g_semaphores.reset(s);
    sem.count = initial;
    sem.max = max;
}
void FinalizeSemaphore(SemaphoreType* s) { g_semaphores.erase(s); }
void AcquireSemaphore(SemaphoreType* s) {
    HostSemaphore& sem = g_semaphores.get(s);
    pthread_mutex_lock(&sem.lock.m);
    while (sem.count == 0) sem.cond.wait(&sem.lock.m);
    sem.count--;
    pthread_mutex_unlock(&sem.lock.m);
}
void ReleaseSemaphore(SemaphoreType* s) {
    HostSemaphore& sem = g_semaphores.get(s);
    pthread_mutex_lock(&sem.lock.m);
    if (sem.count < sem.max) sem.count++;
    pthread_cond_signal(&sem.cond.c);
    pthread_mutex_unlock(&sem.lock.m);
}

void InitializeMultiWait(MultiWaitType* w) { g_multiwaits.reset(w); }
void FinalizeMultiWait(MultiWaitType* w) { g_multiwaits.erase(w); }
void InitializeMultiWaitHolder(MultiWaitHolderType* h, EventType* e) { g_holders.reset(h).event = e; }
void FinalizeMultiWaitHolder(MultiWaitHolderType* h) { g_holders.erase(h); }
void LinkMultiWaitHolder(MultiWaitType* w, MultiWaitHolderType* h) {
    HostMultiWait& mw = g_multiwaits.get(w);
    std::lock_guard lock(mw.m);
    mw.holders.push_back(h);
    g_holders.get(h).owner = w;
}
void UnlinkMultiWaitHolder(MultiWaitHolderType* h) {
    HostHolder& holder = g_holders.get(h);
    if (!holder.owner) return;
    HostMultiWait& mw = g_multiwaits.get(holder.owner);
    std::lock_guard lock(mw.m);
    std::erase(mw.holders, h);
    holder.owner = nullptr;
}
MultiWaitHolderType* WaitAny(MultiWaitType* w) {
    HostMultiWait& mw = g_multiwaits.get(w);
    for (;;) {
        uint64_t gen = g_any_gen.load();
        {
            std::lock_guard lock(mw.m);
            for (MultiWaitHolderType* h : mw.holders)
                if (g_events.get(g_holders.get(h).event).try_take()) return h;
        }
        pthread_mutex_lock(&g_any_pm);
        if (g_any_gen.load() == gen) {
            timespec ts = deadline_after(10'000'000);
            pthread_cond_timedwait(&g_any_cond.c, &g_any_pm, &ts);
        }
        pthread_mutex_unlock(&g_any_pm);
    }
}

// --- Hilos ---

Result CreateThread(ThreadType* t, ThreadFunction fn, void* arg, void*, size_t stack_size, int priority) {
    return create_thread(t, fn, arg, stack_size, priority, 0);
}
Result CreateThread(ThreadType* t, ThreadFunction fn, void* arg, void*, size_t stack_size, int priority,
                    int core) {
    return create_thread(t, fn, arg, stack_size, priority, core);
}

// La pila que da el juego no se usa: bionic reserva la suya del mismo tamaño.
void StartThread(ThreadType* type) {
    HostThread& t = g_threads.get(type);
    // Ya corre: el vídeo reintenta Configure + StartThread si su Start falla, y un segundo pthread_create
    // sobre el mismo HostThread dejaba el primero sin join (su pila perdida) y dos bucles en un decoder.
    if (t.started && !t.joined) return;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, t.stack_size < 256 * 1024 ? 256 * 1024 : t.stack_size);
    set_state(type, Started);
    t.started = pthread_create(&t.pthread, &attr, thread_entry, &t) == 0;
    pthread_attr_destroy(&attr);
    printf("[nnsdk] hilo '%s' fn=%p arg=%p pila=%zu KB%s\n", t.name, reinterpret_cast<void*>(t.fn), t.arg,
           t.stack_size / 1024, t.started ? "" : " NO SE PUDO CREAR");
}

void WaitThread(ThreadType* type) {
    HostThread& t = g_threads.get(type);
    if (t.started && !t.joined) {
        pthread_join(t.pthread, nullptr);
        t.joined = true;
    }
}

void DestroyThread(ThreadType* type) {
    WaitThread(type);
    g_threads.erase(type);
    set_state(type, NotInitialized);
}

ThreadType* GetCurrentThread() { return current_thread(); }
void YieldThread() { sched_yield(); }
void SleepThread(TimeSpan t) {
    timespec ts{static_cast<time_t>(t.ns / 1'000'000'000), static_cast<long>(t.ns % 1'000'000'000)};
    nanosleep(&ts, nullptr);
}

int ChangeThreadPriority(ThreadType* type, int priority) {
    HostThread& t = g_threads.get(type);
    int old = t.priority;
    t.priority = priority;
    return old;
}
void SetThreadCoreMask(ThreadType* type, int ideal_core, uint64_t mask) {
    HostThread& t = g_threads.get(type);
    t.ideal_core = ideal_core;
    t.core_mask = mask;
}
uint64_t GetThreadAvailableCoreMask() { return 0b111; }  // las aplicaciones de Switch usan los núcleos 0-2
uint64_t GetThreadId(const ThreadType* type) { return g_threads.get(type).tid; }
void SetThreadName(ThreadType* type, const char* name) {
    HostThread& t = g_threads.get(type);
    t.name = name ? name : "";
}
const char* GetThreadNamePointer(const ThreadType* type) { return g_threads.get(type).name; }

// --- TLS ---

Result AllocateTlsSlot(TlsSlot* slot, TlsDestructor destructor) {
    pthread_key_t key;
    if (pthread_key_create(&key, reinterpret_cast<void (*)(void*)>(destructor)) != 0) return Result{0x1C01};
    slot->inner = static_cast<uint32_t>(key);
    return ResultSuccess;
}
void FreeTlsSlot(TlsSlot slot) { pthread_key_delete(static_cast<pthread_key_t>(slot.inner)); }
uintptr_t GetTlsValue(TlsSlot slot) {
    return reinterpret_cast<uintptr_t>(pthread_getspecific(static_cast<pthread_key_t>(slot.inner)));
}
void SetTlsValue(TlsSlot slot, uintptr_t value) {
    pthread_setspecific(static_cast<pthread_key_t>(slot.inner), reinterpret_cast<void*>(value));
}

// --- Tiempo: el contador virtual de ARM, igual que en la Switch (19.2 MHz en ambas) ---

Tick GetSystemTick() {
    uint64_t v;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return Tick{static_cast<int64_t>(v)};
}
int64_t GetSystemTickFrequency() {
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return static_cast<int64_t>(f);
}
TimeSpan ConvertToTimeSpan(Tick tick) {
    __int128 ns = static_cast<__int128>(tick.value) * 1'000'000'000 / GetSystemTickFrequency();
    return TimeSpan{static_cast<int64_t>(ns)};
}

// --- Memoria virtual ---

Result AllocateAddressRegion(uintptr_t* out, size_t size) {
    void* p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return Result{0x1801};
    std::lock_guard lock(g_regions_m);
    g_regions[reinterpret_cast<uintptr_t>(p)] = size;
    *out = reinterpret_cast<uintptr_t>(p);
    printf("[nnsdk] AllocateAddressRegion %p +%zu KB\n", p, size / 1024);
    return ResultSuccess;
}
Result FreeAddressRegion(uintptr_t address) {
    std::lock_guard lock(g_regions_m);
    auto it = g_regions.find(address);
    if (it == g_regions.end()) return Result{0x1801};
    munmap(reinterpret_cast<void*>(address), it->second);
    g_regions.erase(it);
    return ResultSuccess;
}
Result AllocateMemoryPages(uintptr_t address, size_t size) {
    printf("[nnsdk] AllocateMemoryPages 0x%lx +%zu KB\n", address, size / 1024);
    return mprotect(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE) == 0 ? ResultSuccess
                                                                                         : Result{0x1801};
}
Result FreeMemoryPages(uintptr_t address, size_t size) {
    printf("[nnsdk] FreeMemoryPages 0x%lx +%zu KB\n", address, size / 1024);
    madvise(reinterpret_cast<void*>(address), size, MADV_DONTNEED);
    mprotect(reinterpret_cast<void*>(address), size, PROT_NONE);
    return ResultSuccess;
}

void GenerateRandomBytes(void* out, size_t size) { arc4random_buf(out, size); }

}  // namespace nn::os

void nnsdk::signal_system_event(SystemEventType* event) { g_events.get(event).signal(); }

void nnsdk::create_system_event(SystemEventType* event, bool auto_clear, bool signaled) {
    HostEvent& ev = g_events.reset(event);
    ev.auto_clear = auto_clear;
    ev.signaled = signaled;
}

// Declaradas en nvn/commands.h: los hilos críticos que no crea nn::os (NvnSubmit, el principal) se apuntan solos.
namespace nvn {

void pin_critical_thread() { pin_current_thread(); }

void refresh_critical_threads() {
    static std::atomic<int64_t> next{0};
    const int64_t now = now_ns();
    if (now < next.load(std::memory_order_relaxed)) return;
    next.store(now + 1'000'000'000, std::memory_order_relaxed);
    const CpuMasks& m = cpu_masks();
    if (!m.usable || !g_any_critical) return;
    const bool off = affinity_off();
    static bool applied_off = false;  // con g_critical_m
    std::lock_guard lock(g_critical_m);
    const bool flag_changed = off != applied_off;  // entonces no es que se haya perdido
    applied_off = off;
    std::erase_if(g_critical, [&](const CriticalThread& t) { return !apply_mask(t, off ? m.all : m.big, !flag_changed); });
    release_inherited_masks(m);
}

}  // namespace nvn

// --- API en C (nnos*): mismos objetos, misma implementación ---

extern "C" {

// Para el watchdog del loader: los hilos del juego vivos.
int nnsdk_list_threads(pthread_t* out, int max) {
    int n = 0;
    g_threads.for_each([&](HostThread& t) {
        if (t.started && !t.joined && n < max && !pthread_equal(t.pthread, pthread_self())) out[n++] = t.pthread;
    });
    return n;
}

uint32_t nnosCreateThread(ThreadType* t, ThreadFunction fn, void* arg, void* stack, size_t size, int prio) {
    return CreateThread(t, fn, arg, stack, size, prio).value;
}
void nnosDestroyThread(ThreadType* t) { DestroyThread(t); }
void nnosStartThread(ThreadType* t) { StartThread(t); }
void nnosWaitThread(ThreadType* t) { WaitThread(t); }
ThreadType* nnosGetCurrentThread() { return GetCurrentThread(); }
int nnosGetThreadPriority(const ThreadType* t) { return g_threads.get(t).priority; }
void nnosSetThreadNamePointer(ThreadType* t, const char* name) { SetThreadName(t, name); }
void nnosGetThreadCoreMask(int* ideal_core, uint64_t* mask, const ThreadType* t) {
    HostThread& ht = g_threads.get(t);
    if (ideal_core) *ideal_core = ht.ideal_core;
    if (mask) *mask = ht.core_mask;
}
void nnosSetThreadCoreMask(ThreadType* t, int ideal_core, uint64_t mask) { SetThreadCoreMask(t, ideal_core, mask); }
void nnosSleepThread(int64_t ns) { SleepThread(TimeSpan{ns}); }
int64_t nnosGetSystemTick() { return GetSystemTick().value; }
int64_t nnosGetSystemTickFrequency() { return GetSystemTickFrequency(); }

void nnosInitializeMutex(MutexType* m, bool recursive, int level) { InitializeMutex(m, recursive, level); }
void nnosFinalizeMutex(MutexType* m) { FinalizeMutex(m); }
void nnosLockMutex(MutexType* m) { LockMutex(m); }
void nnosUnlockMutex(MutexType* m) { UnlockMutex(m); }

void nnosInitializeEvent(EventType* e, bool signaled, EventClearMode mode) { InitializeEvent(e, signaled, mode); }
void nnosFinalizeEvent(EventType* e) { FinalizeEvent(e); }
void nnosSignalEvent(EventType* e) { SignalEvent(e); }
void nnosWaitEvent(EventType* e) { WaitEvent(e); }

}  // extern "C"
