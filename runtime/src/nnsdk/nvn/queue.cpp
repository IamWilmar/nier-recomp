// nvnQueue* y nvnSync*. Enviar comandos los graba y envía a Vulkan al momento (executor.cpp), sin
// esperar a la GPU; un NVNsync recuerda el último envío al hacer FenceSync y esperar el sync es esperar
// a ese envío. Adquirir textura de la ventana es rotar.

#include <cstring>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/nvn.h"

namespace {

// NVNqueueBuilder es opaco; solo guardamos lo que necesitamos.
struct QueueBuilder {
    void* device;
    int flags;
    int command_memory, compute_memory, control_memory;
};

struct Queue {};
struct Sync {
    uint64_t submit = 0;  // envío que lo señala (0: señalado desde el principio)
};

// NVNsyncWaitResult.
constexpr int kSyncAlreadySignaled = 0, kSyncConditionSatisfied = 1, kSyncTimeoutExpired = 2;
constexpr int kWindowAcquireSuccess = 0;

nvn::Objects<Queue> g_queues;
nvn::Objects<Sync> g_syncs;

}  // namespace

extern "C" {

void nvnQueueBuilderSetDefaults(QueueBuilder* b) { memset(b, 0, sizeof *b); }
void nvnQueueBuilderSetDevice(QueueBuilder* b, void* device) { b->device = device; }
void nvnQueueBuilderSetFlags(QueueBuilder* b, int flags) { b->flags = flags; }
void nvnQueueBuilderSetCommandMemorySize(QueueBuilder* b, size_t size) { b->command_memory = size; }
void nvnQueueBuilderSetComputeMemorySize(QueueBuilder* b, size_t size) { b->compute_memory = size; }
void nvnQueueBuilderSetControlMemorySize(QueueBuilder* b, size_t size) { b->control_memory = size; }
void nvnQueueBuilderSetCommandFlushThreshold(QueueBuilder*, size_t) {}
size_t nvnQueueBuilderGetQueueMemorySize(const QueueBuilder*) { return 4096; }
void nvnQueueBuilderSetQueueMemory(QueueBuilder*, void*, size_t) {}

nvn::Boolean nvnQueueInitialize(void* queue, const QueueBuilder*) {
    g_queues.create(queue);
    return 1;
}
void nvnQueueFinalize(void* queue) { g_queues.destroy(queue); }

void nvnQueueSubmitCommands(void* queue, int count, const uint64_t* handles) {
    if (!g_queues.get(queue)) return;
    std::vector<const nvn::CommandList*> lists;
    for (int k = 0; k < count; k++)
        if (handles[k]) lists.push_back(reinterpret_cast<const nvn::CommandList*>(handles[k]));
    nvn::execute(lists);
}
void nvnQueueFlush(void*) {}
void nvnQueueFinish(void*) { nvn::wait_idle(); }
void nvnQueueFenceSync(void*, void* sync, int, int) { nvn::sync_signal_at(sync, nvn::last_submit()); }
// Espera de la GPU a un sync: con una sola cola el orden de envío ya lo garantiza.
nvn::Boolean nvnQueueWaitSync(void*, const void*) { return 1; }

void nvnQueuePresentTexture(void*, void* window, int index) {
    // El hilo principal del juego lo crea el loader, no nn::os, y no tiene nombre del juego: se le reconoce por
    // ser el que presenta (y envía los comandos). Es crítico como MainDraw y NvnSubmit.
    thread_local bool pinned = false;
    if (!pinned) {
        pinned = true;
        nvn::pin_critical_thread();
    }
    nvn::refresh_critical_threads();
    if (nvn::Window* w = nvn::g_windows.get(window)) nvn::queue_present(*w, index);
}

nvn::Boolean nvnSyncInitialize(void* sync, void*) {
    g_syncs.create(sync);
    return 1;
}
void nvnSyncFinalize(void* sync) { g_syncs.destroy(sync); }
int nvnSyncWait(const void* sync, uint64_t timeout_ns) {
    Sync* s = g_syncs.get(sync);
    if (!s || !s->submit) return kSyncAlreadySignaled;
    if (nvn::wait_submit(s->submit, 0)) return kSyncAlreadySignaled;
    return nvn::wait_submit(s->submit, timeout_ns) ? kSyncConditionSatisfied : kSyncTimeoutExpired;
}

int nvnWindowAcquireTexture(void* window, void*, int* index) {
    nvn::Window* w = nvn::g_windows.get(window);
    if (!w || w->textures.empty()) return 1;
    *index = w->next_texture;
    w->next_texture = (w->next_texture + 1) % w->textures.size();
    return kWindowAcquireSuccess;
}

}  // extern "C"

void nvn::sync_signal_at(void* sync, uint64_t submit) {
    Sync* s = g_syncs.get(sync);
    if (!s) s = &g_syncs.create(sync);
    s->submit = submit;
}

NVN_PROC(nvnQueueBuilderSetDefaults);
NVN_PROC(nvnQueueBuilderSetDevice);
NVN_PROC(nvnQueueBuilderSetFlags);
NVN_PROC(nvnQueueBuilderSetCommandMemorySize);
NVN_PROC(nvnQueueBuilderSetComputeMemorySize);
NVN_PROC(nvnQueueBuilderSetControlMemorySize);
NVN_PROC(nvnQueueBuilderSetCommandFlushThreshold);
NVN_PROC(nvnQueueBuilderGetQueueMemorySize);
NVN_PROC(nvnQueueBuilderSetQueueMemory);
NVN_PROC(nvnQueueInitialize);
NVN_PROC(nvnQueueFinalize);
NVN_PROC(nvnQueueSubmitCommands);
NVN_PROC(nvnQueueFlush);
NVN_PROC(nvnQueueFinish);
NVN_PROC(nvnQueueFenceSync);
NVN_PROC(nvnQueueWaitSync);
NVN_PROC(nvnQueuePresentTexture);
NVN_PROC(nvnSyncInitialize);
NVN_PROC(nvnSyncFinalize);
NVN_PROC(nvnSyncWait);
NVN_PROC(nvnWindowAcquireTexture);
