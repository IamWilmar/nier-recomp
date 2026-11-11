// nvnBuffer*: una ventana (offset, tamaño) sobre un memory pool. Dirección de GPU = de CPU.

#include <cstring>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

namespace {

// NVNbufferBuilder es opaco; guardamos aquí dentro solo estos campos.
struct BufferBuilder {
    void* device;
    void* pool;
    int64_t offset;
    size_t size;
};

struct Buffer {
    uint8_t* data = nullptr;
    size_t size = 0;
};

nvn::Objects<Buffer> g_buffers;

}  // namespace

extern "C" {

void nvnBufferBuilderSetDefaults(BufferBuilder* b) { memset(b, 0, sizeof *b); }
void nvnBufferBuilderSetDevice(BufferBuilder* b, void* device) { b->device = device; }
void nvnBufferBuilderSetStorage(BufferBuilder* b, void* pool, int64_t offset, size_t size) {
    b->pool = pool;
    b->offset = offset;
    b->size = size;
}

nvn::Boolean nvnBufferInitialize(void* buffer, const BufferBuilder* b) {
    nvn::MemoryPool* pool = nvn::g_memory_pools.get(b->pool);
    if (!pool) {
        NVN_LOG("buffer %p sobre un memory pool desconocido %p\n", buffer, b->pool);
        return 0;
    }
    Buffer& buf = g_buffers.create(buffer);
    buf.data = pool->storage + b->offset;
    buf.size = b->size;
    return 1;
}
void nvnBufferFinalize(void* buffer) { g_buffers.destroy(buffer); }

void* nvnBufferMap(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->data : nullptr;
}
uint64_t nvnBufferGetAddress(const void* buffer) { return reinterpret_cast<uint64_t>(nvnBufferMap(buffer)); }
size_t nvnBufferGetSize(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->size : 0;
}
void nvnBufferFlushMappedRange(const void*, int64_t, size_t) {}
void nvnBufferInvalidateMappedRange(const void*, int64_t, size_t) {}

}  // extern "C"

NVN_PROC(nvnBufferBuilderSetDefaults);
NVN_PROC(nvnBufferBuilderSetDevice);
NVN_PROC(nvnBufferBuilderSetStorage);
NVN_PROC(nvnBufferInitialize);
NVN_PROC(nvnBufferFinalize);
NVN_PROC(nvnBufferMap);
NVN_PROC(nvnBufferGetAddress);
NVN_PROC(nvnBufferGetSize);
NVN_PROC(nvnBufferFlushMappedRange);
NVN_PROC(nvnBufferInvalidateMappedRange);
