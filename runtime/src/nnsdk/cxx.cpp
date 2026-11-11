// Las 16 funciones de la libc++ de Nintendo (std::__1) que importa el juego. El layout es
// el de libc++ ABI v1, igual que en el NDK, pero se implementan a mano: la memoria de los
// strings tiene que salir del operator new del juego, que es quien la libera después.
// Se definen con su nombre mangleado porque std::__1 no existe en el NDK (usa std::__ndk1).

#include <time.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nnsdk/os.h"
#include "nnsdk/runtime.h"

using namespace nn;

namespace {

// basic_string<char>: largo = {capacidad reservada | 1, tamaño, datos}; corto = {tamaño << 1, 23 bytes}.
struct String {
    union {
        struct {
            size_t cap;
            size_t size;
            char* data;
        } l;
        struct {
            unsigned char size;
            char data[23];
        } s;
    };
    bool is_long() const { return s.size & 1; }
    size_t length() const { return is_long() ? l.size : s.size >> 1; }
    const char* chars() const { return is_long() ? l.data : s.data; }
};
static_assert(sizeof(String) == 24);
constexpr size_t kShortMax = 22;

// Escribe [p, p+n) en str. Si str ya tenía contenido (fresh = false) libera su buffer al final,
// por si p apunta dentro de él.
void set_chars(String* str, const char* p, size_t n, bool fresh) {
    char* old = !fresh && str->is_long() ? str->l.data : nullptr;
    if (old && (str->l.cap & ~size_t{1}) > n) {
        memmove(old, p, n);
        old[n] = 0;
        str->l.size = n;
        return;
    }
    if (n <= kShortMax) {
        char tmp[kShortMax + 1];
        memcpy(tmp, p, n);
        str->s.size = static_cast<unsigned char>(n << 1);
        memcpy(str->s.data, tmp, n);
        str->s.data[n] = 0;
    } else {
        size_t alloc = (n + 16) & ~size_t{15};
        auto* d = static_cast<char*>(nnsdk::host().game_new(alloc));
        memcpy(d, p, n);
        d[n] = 0;
        str->l.cap = alloc | 1;
        str->l.size = n;
        str->l.data = d;
    }
    if (old) nnsdk::host().game_delete(old);
}

[[noreturn]] void fail(const char* what) {
    printf("[libc++] %s\n", what);
    fflush(stdout);
    abort();
}

int64_t clock_ns(clockid_t clock) {
    timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
}

// unique_lock<mutex> = {mutex*, bool}: el mutex es el primer campo.
os::MutexType* lock_mutex(void* unique_lock) { return *static_cast<os::MutexType**>(unique_lock); }

}  // namespace

extern "C" {

// basic_string(const basic_string&)
void _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_(String* self, const String* other) {
    set_chars(self, other->chars(), other->length(), true);
}

// basic_string& assign(const char*)
String* _ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6assignEPKc(String* self, const char* s) {
    set_chars(self, s, strlen(s), false);
    return self;
}

// int compare(size_type pos, size_type n1, const char* s, size_type n2) const
int _ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEmmPKcm(const String* self, size_t pos,
                                                                                    size_t n1, const char* s,
                                                                                    size_t n2) {
    size_t size = self->length();
    if (pos > size) fail("basic_string::compare: out_of_range");
    size_t len = n1 < size - pos ? n1 : size - pos;
    int r = memcmp(self->chars() + pos, s, len < n2 ? len : n2);
    if (r) return r;
    return len < n2 ? -1 : len > n2 ? 1 : 0;
}

void _ZNKSt3__121__basic_string_commonILb1EE20__throw_length_errorEv() { fail("basic_string: length_error"); }
void _ZNKSt3__120__vector_base_commonILb1EE20__throw_length_errorEv() { fail("vector: length_error"); }
void _ZNKSt3__120__vector_base_commonILb1EE20__throw_out_of_rangeEv() { fail("vector: out_of_range"); }

// Primo >= n, para el número de cubetas de unordered_map.
size_t _ZNSt3__112__next_primeEm(size_t n) {
    if (n <= 2) return 2;
    for (n |= 1;; n += 2) {
        bool prime = true;
        for (size_t d = 3; d * d <= n && prime; d += 2) prime = n % d != 0;
        if (prime) return n;
    }
}

// chrono: steady_clock en nanosegundos, system_clock en microsegundos (libc++).
int64_t _ZNSt3__16chrono12steady_clock3nowEv() { return clock_ns(CLOCK_MONOTONIC); }
int64_t _ZNSt3__16chrono12system_clock3nowEv() { return clock_ns(CLOCK_REALTIME) / 1000; }

// std::mutex y std::condition_variable: por dirección, sobre los de nn::os.
void _ZNSt3__15mutex4lockEv(os::MutexType* m) { os::LockMutex(m); }
void _ZNSt3__15mutex6unlockEv(os::MutexType* m) { os::UnlockMutex(m); }
void _ZNSt3__15mutexD1Ev(os::MutexType* m) { os::FinalizeMutex(m); }

void _ZNSt3__118condition_variable10notify_oneEv(os::ConditionVariableType* c) { os::SignalConditionVariable(c); }
void _ZNSt3__118condition_variableD1Ev(os::ConditionVariableType* c) { os::FinalizeConditionVariable(c); }
void _ZNSt3__118condition_variable4waitERNS_11unique_lockINS_5mutexEEE(os::ConditionVariableType* c, void* lock) {
    os::WaitConditionVariable(c, lock_mutex(lock));
}
// __do_timed_wait(unique_lock&, time_point<system_clock, nanoseconds>): plazo absoluto en ns.
void _ZNSt3__118condition_variable15__do_timed_waitERNS_11unique_lockINS_5mutexEEENS_6chrono10time_pointINS5_12system_clockENS5_8durationIxNS_5ratioILl1ELl1000000000EEEEEEE(
    os::ConditionVariableType* c, void* lock, int64_t deadline_ns) {
    int64_t left = deadline_ns - clock_ns(CLOCK_REALTIME);
    os::TimedWaitConditionVariable(c, lock_mutex(lock), TimeSpan{left > 0 ? left : 0});
}

}  // extern "C"
