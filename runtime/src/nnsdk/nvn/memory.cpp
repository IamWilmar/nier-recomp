// nvnMemoryPool*: la memoria la pone el juego (SetStorage). Como dirección de GPU usamos la
// misma dirección de CPU: así cualquier dirección que el juego pase en un comando se lee directo.

#include <cstring>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

namespace {

// NVNmemoryPoolBuilder es opaco; guardamos aquí dentro solo estos campos (cabe sobrado).
struct MemoryPoolBuilder {
    void* device;
    void* storage;
    size_t size;
    int flags;
};

}  // namespace

nvn::Objects<nvn::MemoryPool> nvn::g_memory_pools;

extern "C" {

void nvnMemoryPoolBuilderSetDefaults(MemoryPoolBuilder* b) { memset(b, 0, sizeof *b); }
void nvnMemoryPoolBuilderSetDevice(MemoryPoolBuilder* b, void* device) { b->device = device; }
void nvnMemoryPoolBuilderSetStorage(MemoryPoolBuilder* b, void* storage, size_t size) {
    b->storage = storage;
    b->size = size;
}
void nvnMemoryPoolBuilderSetFlags(MemoryPoolBuilder* b, int flags) { b->flags = flags; }
void* nvnMemoryPoolBuilderGetMemory(const MemoryPoolBuilder* b) { return b->storage; }
size_t nvnMemoryPoolBuilderGetSize(const MemoryPoolBuilder* b) { return b->size; }
int nvnMemoryPoolBuilderGetFlags(const MemoryPoolBuilder* b) { return b->flags; }

nvn::Boolean nvnMemoryPoolInitialize(void* pool, const MemoryPoolBuilder* b) {
    nvn::MemoryPool& p = nvn::g_memory_pools.create(pool);
    p.storage = static_cast<uint8_t*>(b->storage);
    p.size = b->size;
    p.flags = b->flags;
    return 1;
}
void nvnMemoryPoolFinalize(void* pool) { nvn::g_memory_pools.destroy(pool); }

void* nvnMemoryPoolMap(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->storage : nullptr;
}
uint64_t nvnMemoryPoolGetBufferAddress(const void* pool) { return reinterpret_cast<uint64_t>(nvnMemoryPoolMap(pool)); }
size_t nvnMemoryPoolGetSize(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->size : 0;
}
int nvnMemoryPoolGetFlags(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->flags : 0;
}
// La GPU lee la memoria del juego al ejecutar los comandos: no hay cachés que sincronizar.
void nvnMemoryPoolFlushMappedRange(const void*, int64_t, size_t) {}
void nvnMemoryPoolInvalidateMappedRange(const void*, int64_t, size_t) {}

}  // extern "C"

NVN_PROC(nvnMemoryPoolBuilderSetDefaults);
NVN_PROC(nvnMemoryPoolBuilderSetDevice);
NVN_PROC(nvnMemoryPoolBuilderSetStorage);
NVN_PROC(nvnMemoryPoolBuilderSetFlags);
NVN_PROC(nvnMemoryPoolBuilderGetMemory);
NVN_PROC(nvnMemoryPoolBuilderGetSize);
NVN_PROC(nvnMemoryPoolBuilderGetFlags);
NVN_PROC(nvnMemoryPoolInitialize);
NVN_PROC(nvnMemoryPoolFinalize);
NVN_PROC(nvnMemoryPoolMap);
NVN_PROC(nvnMemoryPoolGetBufferAddress);
NVN_PROC(nvnMemoryPoolGetSize);
NVN_PROC(nvnMemoryPoolGetFlags);
NVN_PROC(nvnMemoryPoolFlushMappedRange);
NVN_PROC(nvnMemoryPoolInvalidateMappedRange);
