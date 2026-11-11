// nvnTextureBuilder*, nvnTexture*, nvnTextureView*, nvnTexturePool*, nvnSamplerBuilder*, nvnSampler*,
// nvnSamplerPool* y los handles del dispositivo. Las imágenes de Vulkan las crea el ejecutor cuando
// hacen falta; aquí solo se guarda la descripción y dónde están los texels en memoria del juego.

#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/formats.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

using nvn::SamplerDesc;
using nvn::TextureDesc;
using nvn::TextureViewDesc;

namespace {

constexpr size_t kStorageAlignment = 512;

nvn::Objects<nvn::Sampler> g_samplers;

nvn::TextureLayout layout_of(const TextureDesc& d) {
    const nvn::FormatInfo* f = nvn::format_info(d.format);
    uint32_t bpb = f && f->bytes ? f->bytes : 4, block_w = f ? f->block : 1, block_h = f ? f->block_height() : 1;
    bool array = d.target == nvn::kTarget1DArray || d.target == nvn::kTarget2DArray ||
                 d.target == nvn::kTarget2DMultisampleArray || d.target == nvn::kTargetCubeArray;
    bool cube = d.target == nvn::kTargetCube || d.target == nvn::kTargetCubeArray;
    bool is_3d = d.target == nvn::kTarget3D;
    // Un cubemap tiene 6 caras aunque el juego deje depth en 1 (NieR lo hace); un array de cubos, depth
    // caras (múltiplo de 6). Con 1 sola capa, GetStorageSize salía de una cara y los datos de las otras
    // 5 se escribían sobre lo que hubiera detrás (el mar del prólogo salía magenta: ASTC inválido).
    uint32_t layers = d.target == nvn::kTargetCube ? 6 : array || cube ? std::max(d.depth, 1) : 1;
    uint32_t depth = is_3d ? std::max(d.depth, 1) : 1;
    uint32_t height = d.target == nvn::kTarget1D || d.target == nvn::kTarget1DArray ? 1 : d.height;
    uint32_t samples = std::max(d.samples, 1);
    return nvn::compute_layout(d.width * (samples > 1 ? 2 : 1), height * (samples > 2 ? 2 : 1), depth, layers,
                               d.levels, bpb * (samples > 4 ? 2 : 1), block_w, block_h, d.flags & nvn::kTextureFlagLinear,
                               d.stride, is_3d);
}

}  // namespace

nvn::Objects<nvn::Texture> nvn::g_textures;
std::atomic<uint64_t> nvn::g_texture_gpu_bytes{0}, nvn::g_texture_gpu_images{0};

uint64_t nvn::next_texture_serial() {
    static std::atomic<uint64_t> serial{1};
    return serial++;
}

// Destruirla ya podía pillar a la GPU leyéndola en un envío pendiente: la retira el ejecutor.
void nvn::release_texture_gpu(Texture& t) { nvn::retire_texture_gpu(t); }
nvn::Objects<nvn::DescriptorPool> nvn::g_texture_pools, nvn::g_sampler_pools;

nvn::Sampler* nvn::find_sampler(const void* sampler) { return g_samplers.get(sampler); }

extern "C" {

// --- Builder de texturas ---

void nvnTextureBuilderSetDefaults(TextureDesc* b) {
    memset(b, 0, sizeof *b);
    b->target = nvn::kTarget2D;
    b->width = b->height = b->depth = b->levels = 1;
}
void nvnTextureBuilderSetDevice(TextureDesc* b, void* device) { b->device = device; }
void nvnTextureBuilderSetFlags(TextureDesc* b, int flags) { b->flags = flags; }
void nvnTextureBuilderSetTarget(TextureDesc* b, int target) { b->target = target; }
void nvnTextureBuilderSetFormat(TextureDesc* b, int format) { b->format = format; }
void nvnTextureBuilderSetLevels(TextureDesc* b, int levels) { b->levels = levels; }
void nvnTextureBuilderSetSamples(TextureDesc* b, int samples) { b->samples = samples; }
void nvnTextureBuilderSetStride(TextureDesc* b, int64_t stride) { b->stride = static_cast<int>(stride); }
void nvnTextureBuilderSetWidth(TextureDesc* b, int w) { b->width = w; }
void nvnTextureBuilderSetHeight(TextureDesc* b, int h) { b->height = h; }
void nvnTextureBuilderSetDepth(TextureDesc* b, int d) { b->depth = d; }
void nvnTextureBuilderSetSize1D(TextureDesc* b, int w) { b->width = w, b->height = b->depth = 1; }
void nvnTextureBuilderSetSize2D(TextureDesc* b, int w, int h) { b->width = w, b->height = h, b->depth = 1; }
void nvnTextureBuilderSetSize3D(TextureDesc* b, int w, int h, int d) { b->width = w, b->height = h, b->depth = d; }
void nvnTextureBuilderSetStorage(TextureDesc* b, void* pool, int64_t offset) {
    b->pool = pool;
    b->offset = offset;
}
// El juego reserva esto y escribe ahí los texels en disposición Tegra: tiene que ser el tamaño real.
size_t nvnTextureBuilderGetStorageSize(const TextureDesc* b) {
    size_t size = (layout_of(*b).total_size() + kStorageAlignment - 1) & ~(kStorageAlignment - 1);
    if (size > (64u << 20))
        NVN_LOG("tamaño de textura enorme: %zu MB para %dx%dx%d formato 0x%x target %d niveles %d muestras %d flags 0x%x\n",
                size >> 20, b->width, b->height, b->depth, b->format, b->target, b->levels, b->samples, b->flags);
    return size;
}
size_t nvnTextureBuilderGetStorageAlignment(const TextureDesc*) { return kStorageAlignment; }

// --- Texturas ---

nvn::Boolean nvnTextureInitialize(void* texture, const TextureDesc* b) {
    // Reinicializar sin Finalize: el mismo objeto (los comandos grabados y los pools de texturas guardan su
    // dirección), sin la imagen anterior.
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    nvn::Texture* existing = nvn::g_textures.get(texture);
    if (existing) {
        lock.lock();  // el ejecutor puede estar leyendo este objeto
        nvn::release_texture_gpu(*existing);
        *existing = nvn::Texture{};
    }
    nvn::Texture& t = existing ? *existing : nvn::g_textures.create(texture);
    t.serial = nvn::next_texture_serial();
    t.desc = *b;
    if (!nvn::format_info(b->format) || !nvn::format_info(b->format)->bytes)
        NVN_LOG("textura con formato 0x%x sin mapear: %dx%dx%d target %d niveles %d flags 0x%x\n", b->format, b->width,
                b->height, b->depth, b->target, b->levels, b->flags);
    t.layout = layout_of(*b);
    if (nvn::MemoryPool* pool = nvn::g_memory_pools.get(b->pool)) t.storage = pool->storage + b->offset;
    if ((b->width == 1600 && b->height == 904) || (b->width == 800 && b->height == 452))  // DEPURACIÓN: vídeo
        NVN_LOG("textura de vídeo %p: %dx%d formato 0x%x flags 0x%x stride %d pool %p +0x%llx -> %p\n", texture, b->width,
                b->height, b->format, b->flags, b->stride, b->pool, static_cast<unsigned long long>(b->offset),
                static_cast<void*>(t.storage));
    return 1;
}
void nvnTextureFinalize(void* texture) {
    // La imagen se libera, pero el objeto queda vivo (sin imagen: se muestrea la de relleno) por si algún pool
    // de texturas o lista de comandos aún lo nombra.
    std::lock_guard executor_lock(nvn::executor_mutex());
    if (std::unique_ptr<nvn::Texture> t = nvn::g_textures.take(texture)) {
        nvn::release_texture_gpu(*t);
        t->serial = nvn::next_texture_serial();
        t->storage = nullptr;
        t->vk_failed = true;
        static std::mutex m;
        static std::vector<std::unique_ptr<nvn::Texture>> finalized;
        std::lock_guard lock(m);
        finalized.push_back(std::move(t));
    }
}

#define TEXTURE_GETTER(name, field)                          \
    int nvnTextureGet##name(const void* t) {                \
        nvn::Texture* tex = nvn::g_textures.get(t);         \
        return tex ? tex->desc.field : 0;                   \
    }
TEXTURE_GETTER(Width, width)
TEXTURE_GETTER(Height, height)
TEXTURE_GETTER(Depth, depth)
TEXTURE_GETTER(Format, format)
TEXTURE_GETTER(Target, target)
TEXTURE_GETTER(Levels, levels)
TEXTURE_GETTER(Flags, flags)
#undef TEXTURE_GETTER
// El juego escribe por CPU en texturas lineales (los frames de vídeo) con este stride.
int64_t nvnTextureGetStride(const void* t) {
    nvn::Texture* tex = nvn::g_textures.get(t);
    return tex && tex->layout.linear ? tex->layout.stride : 0;
}
size_t nvnTextureGetStorageSize(const void* t) {
    nvn::Texture* tex = nvn::g_textures.get(t);
    return tex ? tex->layout.total_size() : 0;
}

// --- Vistas ---

void nvnTextureViewSetDefaults(TextureViewDesc* v) { memset(v, 0, sizeof *v); }
void nvnTextureViewSetLevels(TextureViewDesc* v, int base, int count) {
    v->flags |= nvn::kViewLevels;
    v->base_level = base, v->levels = count;
}
void nvnTextureViewSetLayers(TextureViewDesc* v, int base, int count) {
    v->flags |= nvn::kViewLayers;
    v->base_layer = base, v->layers = count;
}
void nvnTextureViewSetFormat(TextureViewDesc* v, int format) {
    v->flags |= nvn::kViewFormat;
    v->format = format;
}
void nvnTextureViewSetTarget(TextureViewDesc* v, int target) {
    v->flags |= nvn::kViewTarget;
    v->target = target;
}
void nvnTextureViewSetDepthStencilMode(TextureViewDesc* v, int mode) {
    v->flags |= nvn::kViewDepthStencil;
    v->depth_stencil_mode = mode;
}
void nvnTextureViewSetSwizzle(TextureViewDesc* v, int r, int g, int b, int a) {
    static int logged = 0;  // DEPURACIÓN: qué combinaciones usa el juego
    static int seen[16][4];
    bool fresh = true;
    for (int k = 0; k < std::min(logged, 16); k++)
        if (seen[k][0] == r && seen[k][1] == g && seen[k][2] == b && seen[k][3] == a) fresh = false;
    if (fresh && logged < 16) {
        seen[logged][0] = r, seen[logged][1] = g, seen[logged][2] = b, seen[logged][3] = a;
        logged++;
        NVN_LOG("swizzle de vista: %d %d %d %d\n", r, g, b, a);
    }
    v->flags |= nvn::kViewSwizzle;
    v->swizzle[0] = r, v->swizzle[1] = g, v->swizzle[2] = b, v->swizzle[3] = a;
}

// --- Pools de descriptores ---

nvn::Boolean nvnTexturePoolInitialize(void* pool, void* memory_pool, int64_t offset, int count) {
    nvn::DescriptorPool& p = nvn::g_texture_pools.create(pool);
    p.memory_pool = memory_pool;
    p.offset = offset;
    p.count = count;
    p.textures.resize(count);
    return 1;
}
void nvnTexturePoolFinalize(void* pool) { nvn::g_texture_pools.destroy(pool); }
void nvnTexturePoolRegisterTexture(void* pool, int id, const void* texture, const TextureViewDesc* view) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    if (!p || id < 0 || id >= p->count) return;
    p->textures[id] = {nvn::g_textures.get(texture), view ? *view : TextureViewDesc{}};
}
void nvnTexturePoolRegisterImage(void* pool, int id, const void* texture, const TextureViewDesc* view) {
    nvnTexturePoolRegisterTexture(pool, id, texture, view);
}

nvn::Boolean nvnSamplerPoolInitialize(void* pool, void* memory_pool, int64_t offset, int count) {
    nvn::DescriptorPool& p = nvn::g_sampler_pools.create(pool);
    p.memory_pool = memory_pool;
    p.offset = offset;
    p.count = count;
    p.samplers.resize(count);
    return 1;
}
void nvnSamplerPoolFinalize(void* pool) { nvn::g_sampler_pools.destroy(pool); }
void nvnSamplerPoolRegisterSampler(void* pool, int id, const void* sampler) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    if (!p || id < 0 || id >= p->count) return;
    p->samplers[id] = g_samplers.get(sampler);
}

// --- Handles: separados = id; combinado = sampler << 32 | textura (los resuelve el ejecutor) ---

uint64_t nvnDeviceGetSeparateTextureHandle(const void*, int texture_id) { return static_cast<uint32_t>(texture_id); }
uint64_t nvnDeviceGetSeparateSamplerHandle(const void*, int sampler_id) { return static_cast<uint32_t>(sampler_id); }
uint64_t nvnDeviceGetTextureHandle(const void*, int texture_id, int sampler_id) {
    return uint64_t{static_cast<uint32_t>(sampler_id)} << 32 | static_cast<uint32_t>(texture_id);
}
uint64_t nvnDeviceGetImageHandle(const void*, int image_id) { return static_cast<uint32_t>(image_id); }
uint64_t nvnDeviceGetTexelFetchHandle(const void*, int texture_id) { return static_cast<uint32_t>(texture_id); }

// --- Samplers ---

void nvnSamplerBuilderSetDefaults(SamplerDesc* b) {
    memset(b, 0, sizeof *b);
    b->min_filter = 4;  // NEAREST_MIPMAP_LINEAR (el de OpenGL)
    b->mag_filter = 1;  // LINEAR
    b->wrap[0] = b->wrap[1] = b->wrap[2] = 1;  // REPEAT
    b->max_anisotropy = 1.0f;
    b->lod_min = -1000.0f;
    b->lod_max = 1000.0f;
    b->compare_func = 8;  // ALWAYS
}
void nvnSamplerBuilderSetDevice(SamplerDesc* b, void* device) { b->device = device; }
void nvnSamplerBuilderSetMinMagFilter(SamplerDesc* b, int min, int mag) { b->min_filter = min, b->mag_filter = mag; }
void nvnSamplerBuilderSetWrapMode(SamplerDesc* b, int s, int t, int r) { b->wrap[0] = s, b->wrap[1] = t, b->wrap[2] = r; }
void nvnSamplerBuilderSetLodClamp(SamplerDesc* b, float min, float max) { b->lod_min = min, b->lod_max = max; }
void nvnSamplerBuilderSetLodBias(SamplerDesc* b, float bias) { b->lod_bias = bias; }
void nvnSamplerBuilderSetCompare(SamplerDesc* b, int mode, int func) { b->compare_mode = mode, b->compare_func = func; }
void nvnSamplerBuilderSetBorderColor(SamplerDesc* b, const float* color) { memcpy(b->border, color, sizeof b->border); }
void nvnSamplerBuilderSetMaxAnisotropy(SamplerDesc* b, float a) { b->max_anisotropy = a; }
void nvnSamplerBuilderSetReductionFilter(SamplerDesc* b, int filter) { b->reduction = filter; }

// Como las texturas, el objeto no se borra nunca: los pools de samplers guardan su dirección (borrarlo
// dejaba al ejecutor escribiendo en memoria liberada). Su VkSampler se retira hasta que la GPU acaba.
nvn::Boolean nvnSamplerInitialize(void* sampler, const SamplerDesc* b) {
    std::lock_guard lock(nvn::executor_mutex());
    nvn::Sampler* s = g_samplers.get(sampler);
    if (!s) s = &g_samplers.create(sampler);
    if (s->vk) nvn::retire_sampler(s->vk);
    *s = nvn::Sampler{};
    s->desc = *b;
    return 1;
}
void nvnSamplerFinalize(void* sampler) {
    std::lock_guard lock(nvn::executor_mutex());
    if (nvn::Sampler* s = g_samplers.get(sampler)) {
        if (s->vk) nvn::retire_sampler(s->vk);
        s->vk = VK_NULL_HANDLE;
    }
}

}  // extern "C"

NVN_PROC(nvnTextureBuilderSetDefaults);
NVN_PROC(nvnTextureBuilderSetDevice);
NVN_PROC(nvnTextureBuilderSetFlags);
NVN_PROC(nvnTextureBuilderSetTarget);
NVN_PROC(nvnTextureBuilderSetFormat);
NVN_PROC(nvnTextureBuilderSetLevels);
NVN_PROC(nvnTextureBuilderSetSamples);
NVN_PROC(nvnTextureBuilderSetStride);
NVN_PROC(nvnTextureBuilderSetWidth);
NVN_PROC(nvnTextureBuilderSetHeight);
NVN_PROC(nvnTextureBuilderSetDepth);
NVN_PROC(nvnTextureBuilderSetSize1D);
NVN_PROC(nvnTextureBuilderSetSize2D);
NVN_PROC(nvnTextureBuilderSetSize3D);
NVN_PROC(nvnTextureBuilderSetStorage);
NVN_PROC(nvnTextureBuilderGetStorageSize);
NVN_PROC(nvnTextureBuilderGetStorageAlignment);
NVN_PROC(nvnTextureInitialize);
NVN_PROC(nvnTextureFinalize);
NVN_PROC(nvnTextureGetWidth);
NVN_PROC(nvnTextureGetHeight);
NVN_PROC(nvnTextureGetDepth);
NVN_PROC(nvnTextureGetFormat);
NVN_PROC(nvnTextureGetTarget);
NVN_PROC(nvnTextureGetLevels);
NVN_PROC(nvnTextureGetFlags);
NVN_PROC(nvnTextureGetStorageSize);
NVN_PROC(nvnTextureGetStride);
NVN_PROC(nvnTextureViewSetDefaults);
NVN_PROC(nvnTextureViewSetLevels);
NVN_PROC(nvnTextureViewSetLayers);
NVN_PROC(nvnTextureViewSetFormat);
NVN_PROC(nvnTextureViewSetTarget);
NVN_PROC(nvnTextureViewSetDepthStencilMode);
NVN_PROC(nvnTextureViewSetSwizzle);
NVN_PROC(nvnTexturePoolInitialize);
NVN_PROC(nvnTexturePoolFinalize);
NVN_PROC(nvnTexturePoolRegisterTexture);
NVN_PROC(nvnTexturePoolRegisterImage);
NVN_PROC(nvnSamplerPoolInitialize);
NVN_PROC(nvnSamplerPoolFinalize);
NVN_PROC(nvnSamplerPoolRegisterSampler);
NVN_PROC(nvnDeviceGetSeparateTextureHandle);
NVN_PROC(nvnDeviceGetSeparateSamplerHandle);
NVN_PROC(nvnDeviceGetTextureHandle);
NVN_PROC(nvnDeviceGetImageHandle);
NVN_PROC(nvnDeviceGetTexelFetchHandle);
NVN_PROC(nvnSamplerBuilderSetDefaults);
NVN_PROC(nvnSamplerBuilderSetDevice);
NVN_PROC(nvnSamplerBuilderSetMinMagFilter);
NVN_PROC(nvnSamplerBuilderSetWrapMode);
NVN_PROC(nvnSamplerBuilderSetLodClamp);
NVN_PROC(nvnSamplerBuilderSetLodBias);
NVN_PROC(nvnSamplerBuilderSetCompare);
NVN_PROC(nvnSamplerBuilderSetBorderColor);
NVN_PROC(nvnSamplerBuilderSetMaxAnisotropy);
NVN_PROC(nvnSamplerBuilderSetReductionFilter);
NVN_PROC(nvnSamplerInitialize);
NVN_PROC(nvnSamplerFinalize);
