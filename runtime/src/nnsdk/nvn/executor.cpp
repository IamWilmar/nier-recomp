// SPDX-License-Identifier: GPL-3.0-or-later
// Ejecutor: reproduce en Vulkan los comandos que el juego envía a la cola NVN. El estado (render
// targets, programas, buffers, texturas…) es de la cola y persiste entre listas, como en NVN.
// De momento todo es síncrono: cada envío graba un VkCommandBuffer, lo manda y espera, así que los
// syncs de NVN ya están señalados al volver.
//
// Memoria: el driver no tiene VK_EXT_external_memory_host, así que los uniform buffers, vértices e
// índices se copian de la memoria del juego a un staging al dibujar. Las imágenes viven siempre en
// VK_IMAGE_LAYOUT_GENERAL (correcto y simple; las dependencias de los render passes sincronizan).

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <set>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_set>
#include <cstring>
#include <mutex>
#include <string>
#include <fstream>
#include <unordered_map>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/fsr.h"
#include "nnsdk/nvn/lsfg/framegen.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/runtime.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/formats.h"

namespace nvn {

namespace {

using nvnshader::Resource;
using nvnshader::Stage;

// Dónde espera el compilador de NVN cada cosa (deducido de los shaders traducidos):
constexpr uint32_t kTextureHandleBank = 2;      // handles de textura: c2[0x168 + 8·i]
constexpr uint32_t kTextureHandleBase = 0x168;
constexpr uint32_t kSamplerHandleBase = 0x568;  // samplers separados (BindSeparateSampler), en c2 tras los de textura
constexpr uint32_t kUniformBankBase = 3;        // uniform buffer i del juego = banco 3 + i
constexpr uint32_t kShaderConstantsBank = 1;   // c1: constantes que el compilador dejó tras el código
constexpr uint32_t kComputeDriverBank = 0;      // compute: c0 del driver, con los handles de textura en
                                                // 0x168 + 8·i y la tabla de storage buffers en 0x310 + 16·i
constexpr uint32_t kComputeStorageBase = 0x310;
constexpr uint32_t kComputeDriverSize = 0x400;
// Gráficos: un c0 común con una tabla de storage buffers por etapa en 0x110 + 0x100·etapa (orden de
// nvnshader::Stage; el pixel shader los busca en 0x510).
constexpr uint32_t kGraphicsStorageBase = 0x110, kGraphicsStorageStride = 0x100, kGraphicsDriverSize = 0x700;
constexpr int kStages = 6;
constexpr int kMaxUniforms = 18, kMaxTextures = 32, kMaxStorage = 16, kMaxVertexBuffers = 16;
constexpr VkDeviceSize kStagingChunk = 16 << 20;  // también lo más grande que cabe en una reserva
constexpr int kLoggedDraws = 12;

std::recursive_mutex g_mutex;  // el estado del ejecutor (y lo que lee de las texturas y samplers del juego)
// La VkQueue no admite accesos concurrentes: cada vkQueueSubmit/vkQueuePresentKHR, y nada más. Aparte de g_mutex
// para que presentar (y sus esperas al fence y al swapchain) no espere a que acabe una reproducción.
std::mutex g_queue_mutex;

// NVNshaderStage: VERTEX 0, FRAGMENT 1, GEOMETRY 2, TESS_CONTROL 3, TESS_EVALUATION 4, COMPUTE 5.
Stage from_nvn_stage(int s) {
    switch (s) {
        case 0: return Stage::Vertex;
        case 1: return Stage::Fragment;
        case 2: return Stage::Geometry;
        case 3: return Stage::TessControl;
        case 4: return Stage::TessEval;
        default: return Stage::Compute;
    }
}

VkShaderStageFlagBits vk_stage(Stage s) {
    switch (s) {
        case Stage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
        case Stage::TessControl: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case Stage::TessEval: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case Stage::Geometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
        case Stage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
        default: return VK_SHADER_STAGE_COMPUTE_BIT;
    }
}

bool is_depth_format(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT;
}
bool has_stencil(VkFormat f) {
    return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT;
}

// Posición de un tipo en el variant Command (para recorrer listas con un switch, sin std::visit).
template <typename T, typename V>
struct VariantIndex;
template <typename T, typename... Ts>
struct VariantIndex<T, std::variant<Ts...>> {
    static constexpr size_t value = [] {
        constexpr bool same[] = {std::is_same_v<T, Ts>...};
        size_t i = 0;
        while (i < sizeof...(Ts) && !same[i]) i++;
        return i;
    }();
};
template <typename T>
constexpr size_t kCommandIndex = VariantIndex<T, Command>::value;
static_assert(kCommandIndex<cmd::SetRenderTargets> == 0 && kCommandIndex<cmd::FenceSync> == std::variant_size_v<Command> - 1);

// Mismos targets con las mismas vistas (lo que hay más allá de count no cuenta).
bool same_targets(const cmd::SetRenderTargets& a, const cmd::SetRenderTargets& b) {
    if (a.count != b.count || a.depth != b.depth) return false;
    if (a.depth && memcmp(&a.depth_view, &b.depth_view, sizeof a.depth_view) != 0) return false;
    for (int k = 0; k < a.count; k++) {
        if (a.colors[k] != b.colors[k]) return false;
        if (a.colors[k] && memcmp(&a.color_views[k], &b.color_views[k], sizeof a.color_views[k]) != 0) return false;
    }
    return true;
}

// Carga y guardado de cada adjunto de un render pass (por defecto LOAD y STORE: lo de siempre). No cuentan para la
// compatibilidad de render passes: los pipelines y framebuffers se crean con el de LOAD/STORE y valen para todos.
struct PassOps {
    VkAttachmentLoadOp color_load[8];
    VkAttachmentStoreOp color_store[8];
    VkAttachmentLoadOp depth_load, stencil_load;
    VkAttachmentStoreOp depth_store;  // también el del stencil: se guardan o no los dos
};
PassOps default_pass_ops() {
    PassOps o;
    for (int k = 0; k < 8; k++) o.color_load[k] = VK_ATTACHMENT_LOAD_OP_LOAD, o.color_store[k] = VK_ATTACHMENT_STORE_OP_STORE;
    o.depth_load = o.stencil_load = VK_ATTACHMENT_LOAD_OP_LOAD;
    o.depth_store = VK_ATTACHMENT_STORE_OP_STORE;
    return o;
}

// Intersección de dos rectángulos (vacía: tamaño 0 en el origen de b).
VkRect2D intersect(const VkRect2D& a, const VkRect2D& b) {
    const int64_t x0 = std::max<int64_t>(a.offset.x, b.offset.x), y0 = std::max<int64_t>(a.offset.y, b.offset.y);
    const int64_t x1 = std::min<int64_t>(int64_t{a.offset.x} + a.extent.width, int64_t{b.offset.x} + b.extent.width);
    const int64_t y1 = std::min<int64_t>(int64_t{a.offset.y} + a.extent.height, int64_t{b.offset.y} + b.extent.height);
    if (x1 <= x0 || y1 <= y0) return {b.offset, {0, 0}};
    return {{static_cast<int32_t>(x0), static_cast<int32_t>(y0)}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
}

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

// Huella de la memoria de una textura: entera si es pequeña; si no, 64 bytes de cada página de 4 KB
// (los frames de vídeo cambian enteros; las texturas estáticas no cambian).
// Cuatro cadenas independientes, como range_hash: los mismos bytes, sin esperar cada multiplicación.
uint64_t content_hash(const Texture& t) {
    constexpr uint64_t kP1 = 0x9E3779B185EBCA87ULL, kP2 = 0xC2B2AE3D27D4EB4FULL;
    const size_t size = t.layout.total_size();
    const uint8_t* p = t.storage;
    uint64_t v[4] = {size ^ kP1, size + kP2, size, size - kP1};
    auto round = [](uint64_t acc, uint64_t w) {
        acc += w * kP2;
        acc = (acc << 31) | (acc >> 33);
        return acc * kP1;
    };
    auto mix = [&](const uint8_t* q, size_t n) {
        size_t k = 0;
        for (; k + 32 <= n; k += 32) {
            uint64_t w[4];
            memcpy(w, q + k, 32);
            v[0] = round(v[0], w[0]), v[1] = round(v[1], w[1]), v[2] = round(v[2], w[2]), v[3] = round(v[3], w[3]);
        }
        for (; k + 8 <= n; k += 8) {
            uint64_t w;
            memcpy(&w, q + k, 8);
            v[0] = round(v[0], w);
        }
    };
    if (size <= 64 * 1024) {
        mix(p, size);
    } else {
        for (size_t off = 0; off + 64 <= size; off += 4096) mix(p + off, 64);
        mix(p + size - 64, 64);
    }
    uint64_t h = ((v[0] << 1) | (v[0] >> 63)) + ((v[1] << 7) | (v[1] >> 57)) + ((v[2] << 12) | (v[2] >> 52)) +
                 ((v[3] << 18) | (v[3] >> 46));
    h ^= h >> 33;
    h *= kP2;
    return h ^ (h >> 29);
}

// Claves de las cachés (pipelines, vistas, layouts): 8 bytes por paso y mezcla final. Byte a byte, la
// clave de pipeline (casi 400 bytes) costaba ~0,3 ms por frame en la fábrica.
uint64_t hash_bytes(const void* data, size_t size, uint64_t h = 1469598103934665603ULL) {
    auto* p = static_cast<const uint8_t*>(data);
    size_t k = 0;
    for (; k + 8 <= size; k += 8) {
        uint64_t w;
        memcpy(&w, p + k, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
    }
    for (; k < size; k++) h = (h ^ p[k]) * 1099511628211ULL;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    return h ^ (h >> 33);
}

template <typename T>
std::string key_of(const T& v) {
    return std::string(reinterpret_cast<const char*>(&v), sizeof v);
}

// Claves de caché de tamaño fijo (sin construir strings en cada draw): se comparan y se hashean como
// bytes, así que se crean a cero antes de rellenarlas (el relleno entre campos también cuenta).
template <typename T>
struct BytesKey {
    T value;
    BytesKey() { memset(&value, 0, sizeof value); }
    bool operator==(const BytesKey& o) const { return memcmp(&value, &o.value, sizeof value) == 0; }
};
template <typename T>
struct BytesKeyHash {
    size_t operator()(const BytesKey<T>& k) const { return static_cast<size_t>(hash_bytes(&k.value, sizeof k.value)); }
};
struct ViewKeyData {
    uint64_t texture;  // Texture::serial
    TextureViewDesc view;
    bool attachment;
};
struct LayoutKeyData {
    const void* stages[6];
    uint64_t generation[6];  // un programa al que el juego cambia los shaders reutiliza sus ProgramStage
};

// --- Staging: trozos host-visible que se reciclan cuando la GPU termina ---

class Staging {
public:
    struct Alloc {
        VkBuffer buffer;
        VkDeviceSize offset;
        uint8_t* data;
    };
    // Trozos de `chunk` bytes (o más grandes, para una reserva que no cabe en uno).
    explicit Staging(VkDeviceSize chunk = kStagingChunk) : chunk_(chunk) {}
    VkDeviceSize bytes() const {  // perfil
        VkDeviceSize total = 0;
        for (const Chunk& c : chunks_) total += c.size;
        return total;
    }
    size_t used_chunks() const {  // los que tienen algo desde el último reset()
        size_t n = 0;
        for (const Chunk& c : chunks_) n += c.used > 0;
        return n;
    }
    // Más de kStagingChunk no cabe: nullptr (antes se recortaba y quien copiaba el tamaño pedido escribía fuera).
    Alloc alloc(VkDeviceSize size, VkDeviceSize alignment) {
        if (size > kStagingChunk) return {VK_NULL_HANDLE, 0, nullptr};
        for (;; current_++) {
            if (current_ == chunks_.size() && (failed_ || !grow(size))) return {VK_NULL_HANDLE, 0, nullptr};
            Chunk& c = chunks_[current_];
            VkDeviceSize offset = (c.used + alignment - 1) / alignment * alignment;
            if (offset + size <= c.size) {
                c.used = offset + size;
                return {c.buffer, offset, c.mapped + offset};
            }
        }
    }
    void reset() {
        for (Chunk& c : chunks_) c.used = 0;
        current_ = 0;
        failed_ = false;
    }
    // Libera los trozos de más (tras un pico, p. ej. subir texturas al cargar): con varios envíos en
    // vuelo, cada uno guardando sus picos, la memoria se disparaba hasta que Android mataba el juego.
    // Deja como mucho los `keep` primeros, y menos si pasan de `max_bytes` (sin bajar de uno).
    void trim(size_t keep, VkDeviceSize max_bytes = ~VkDeviceSize{0}) {
        VkContext* c = vk();
        while (chunks_.size() > keep || (chunks_.size() > 1 && bytes() > max_bytes)) {
            Chunk& ch = chunks_.back();
            vkDestroyBuffer(c->device, ch.buffer, nullptr);
            vkFreeMemory(c->device, ch.memory, nullptr);
            chunks_.pop_back();
        }
        current_ = std::min(current_, chunks_.empty() ? size_t{0} : chunks_.size() - 1);
    }

private:
    struct Chunk {
        VkBuffer buffer;
        VkDeviceMemory memory;
        uint8_t* mapped;
        VkDeviceSize used, size;
    };
    bool grow(VkDeviceSize need) {
        VkContext* c = vk();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = (std::max(need, chunk_) + chunk_ - 1) / chunk_ * chunk_;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        Chunk chunk{};
        if (vkCreateBuffer(c->device, &bi, nullptr, &chunk.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(c->device, chunk.buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        // Sin RAM, cada alloc siguiente volvía a intentarlo y dejaba un VkBuffer perdido: se desiste hasta reset().
        if (vkAllocateMemory(c->device, &ai, nullptr, &chunk.memory) != VK_SUCCESS) {
            vkDestroyBuffer(c->device, chunk.buffer, nullptr);
            failed_ = true;
            return false;
        }
        vkBindBufferMemory(c->device, chunk.buffer, chunk.memory, 0);
        if (vkMapMemory(c->device, chunk.memory, 0, bi.size, 0, reinterpret_cast<void**>(&chunk.mapped)) != VK_SUCCESS) {
            vkDestroyBuffer(c->device, chunk.buffer, nullptr);
            vkFreeMemory(c->device, chunk.memory, nullptr);
            failed_ = true;
            return false;
        }
        chunk.size = bi.size;
        chunks_.push_back(chunk);
        return true;
    }
    VkDeviceSize chunk_;
    std::vector<Chunk> chunks_;
    size_t current_ = 0;
    bool failed_ = false;  // grow() falló desde el último reset()

public:
    bool failed() const { return failed_; }
};

// --- Caché de vértices e índices ---
// Copiar cada vertex buffer y recorrer todos los índices en cada draw costaba ~25 ms por frame en el
// prólogo. La mayoría son mallas estáticas: cada rango se copia una vez a un arena host-visible y se
// reutiliza mientras su huella no cambie (se comprueba una vez por envío: el juego no puede tocar
// memoria que usa un envío pendiente). Lo que cambia a menudo (UI, partículas) se marca dinámico y
// vuelve al camino de siempre (staging por draw). Las entradas no se reescriben: un cambio ocupa sitio
// nuevo, así un draw anterior del mismo envío sigue leyendo lo suyo. Si el arena se llena, se empieza
// otra generación (otro arena) y lo de la vieja se vuelve a copiar al usarse; la vieja se recicla cuando
// la GPU acaba su último envío. Con <datos>/sin_generaciones, como antes: se vacía al terminar un envío,
// con la GPU parada.

// Cuatro cadenas independientes (como xxh64): la CPU las solapa en vez de esperar cada multiplicación. Mira los
// mismos bytes que antes; se calcula cada envío para cada malla cacheada (~1 ms por frame en la fábrica).
uint64_t range_hash(const uint8_t* p, size_t size) {
    constexpr uint64_t kP1 = 0x9E3779B185EBCA87ULL, kP2 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t v[4] = {size ^ kP1, size + kP2, size, size - kP1};
    auto round = [](uint64_t acc, uint64_t w) {
        acc += w * kP2;
        acc = (acc << 31) | (acc >> 33);
        return acc * kP1;
    };
    uint64_t tail = 1469598103934665603ULL;
    auto mix = [&](const uint8_t* q, size_t n) {
        size_t k = 0;
        for (; k + 32 <= n; k += 32) {
            uint64_t w[4];
            memcpy(w, q + k, 32);
            v[0] = round(v[0], w[0]), v[1] = round(v[1], w[1]), v[2] = round(v[2], w[2]), v[3] = round(v[3], w[3]);
        }
        for (; k + 8 <= n; k += 8) {
            uint64_t w;
            memcpy(&w, q + k, 8);
            v[0] = round(v[0], w);
        }
        for (; k < n; k++) tail = (tail ^ q[k]) * 1099511628211ULL;  // la cola (rangos de 4 bytes: contadores)
    };
    if (size <= 16 * 1024) {
        mix(p, size);
    } else {
        for (size_t off = 0; off + 64 <= size; off += 2048) mix(p + off, 64);
        mix(p + size - 256, 256);
    }
    uint64_t h = ((v[0] << 1) | (v[0] >> 63)) + ((v[1] << 7) | (v[1] >> 57)) + ((v[2] << 12) | (v[2] >> 52)) +
                 ((v[3] << 18) | (v[3] >> 46));
    h ^= tail;
    h ^= h >> 33;
    h *= kP2;
    return h ^ (h >> 29);
}

class BufferCache {
public:
    struct Entry {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        uint64_t hash = 0, checked = 0;
        uint32_t changes = 0, min_index = 0, max_index = 0;  // índices: el menor y el mayor (sin el de reinicio)
        bool dynamic = false;
    };
    // RAM: el Adreno no tiene memoria propia. Tiene que caber lo que se dibuja en un frame: con 128 MB, en la
    // fábrica se vaciaba decenas de veces por frame (todo se volvía a copiar: 1 fps).
    static constexpr VkDeviceSize kArenaLimit = 384ull << 20;  // sin generaciones
    // Con generaciones hay dos arenas que se turnan, de 256 MB: como mucho 512 MB (antes 384 fijos) y sin
    // reservar ni soltar memoria en cada cambio (cada arena guarda sus trozos para su siguiente turno). El viejo
    // solo espera a la GPU unos pocos envíos (nada nuevo lo enlaza), así que dos bastan.
    static constexpr VkDeviceSize kGenerationLimit = 256ull << 20;
    // Lleno, lo nuevo se copia en cada draw (al staging) hasta el cambio de generación (o el vaciado): como mucho
    // uno cada tantos frames, para que lo que no cabe no tire en cada frame lo que sí cabe. Sin generaciones el
    // vaciado solo puede hacerse con la GPU parada: el envío en curso se parte (al cargar la fábrica un solo envío
    // llegó a 3,4 GB).
    static constexpr uint64_t kResetEvery = 30;
    static constexpr uint32_t kDynamicAfter = 3;  // cambios vistos antes de dejar de cachear

    // Devuelve la entrada vigente para [address, address+size), o nullptr si el rango es dinámico o
    // no cabe. index_type >= 0: índices (los de 8 bits se amplían a 16) y se guarda el índice máximo.
    const Entry* get(uint64_t address, size_t size, int index_type, uint64_t submit) {
        if (!address || !size || size > kStagingChunk) return nullptr;
        Entry& e = entries_[{address, (uint64_t{static_cast<uint32_t>(index_type + 1)} << 56) | size}];
        if (e.dynamic) return nullptr;
        if (e.buffer && e.checked == submit) return &e;
        const auto* src = reinterpret_cast<const uint8_t*>(address);
        uint64_t h = range_hash(src, size);
        e.checked = submit;
        if (e.buffer && h == e.hash) return &e;
        if (e.buffer && ++e.changes >= kDynamicAfter) {
            e.dynamic = true;
            return nullptr;
        }
        const size_t out = index_type == 0 ? size * 2 : size;
        // Sin sitio (hasta el siguiente vaciado copia cada draw): la copia vieja ya no vale para este envío.
        Generation& g = gens_[active_];
        Staging::Alloc a = g.used + out <= limit_ ? g.arena.alloc(out, 16) : Staging::Alloc{};
        if (!a.data) {
            full_ = true;
            e.buffer = VK_NULL_HANDLE;
            return nullptr;
        }
        g.used += out;
        if (index_type == 0)
            for (size_t k = 0; k < size; k++) reinterpret_cast<uint16_t*>(a.data)[k] = src[k];
        else
            memcpy(a.data, src, size);
        if (index_type >= 0) index_range(src, size, index_type, &e.min_index, &e.max_index);
        e.buffer = a.buffer;
        e.offset = a.offset;
        e.hash = h;
        return &e;
    }
    bool needs_reset(uint64_t frame) const { return full_ && frame >= reset_frame_ + kResetEvery; }
    VkDeviceSize bytes() const { return gens_[0].arena.bytes() + gens_[1].arena.bytes(); }  // perfil
    void set_generations(bool on) { limit_ = on ? kGenerationLimit : kArenaLimit; }
    // Con generaciones, entre draws (nadie guarda punteros a entradas): si toca, lo de ahora se queda para la GPU
    // hasta el envío `last_submit` (el que se está grabando) y se sigue en la otra generación, vacía. Las entradas
    // se olvidan (también lo dinámico, como al vaciar) y se vuelven a copiar al usarse. false si la otra
    // generación aún está en vuelo.
    bool next_generation(uint64_t frame, uint64_t last_submit) {
        if (!needs_reset(frame)) return false;
        Generation& next = gens_[active_ ^ 1];
        if (next.in_flight) return false;
        Generation& old = gens_[active_];
        static int logged = 0;
        if (logged++ < 20 || logged % 100 == 0)
            NVN_LOG("caché de vértices: generación nueva (%zu entradas, %llu MB, %llu frames desde la anterior)\n",
                    entries_.size(), static_cast<unsigned long long>(old.used >> 20),
                    static_cast<unsigned long long>(frame - reset_frame_));
        old.in_flight = true;
        old.last_submit = last_submit;
        active_ ^= 1;
        next.arena.reset();
        next.used = 0;
        entries_.clear();
        full_ = false;
        reset_frame_ = frame;
        return true;
    }
    // La generación vieja vuelve a estar libre cuando la GPU ha terminado su último envío (completed: el último
    // envío terminado). Guarda sus trozos para su siguiente turno; sin generaciones, los suelta.
    void reap(uint64_t completed, bool generations) {
        Generation& other = gens_[active_ ^ 1];
        if (other.in_flight && other.last_submit <= completed) {
            other.in_flight = false;
            other.arena.reset();
            other.arena.trim(~size_t{0}, limit_ + 2 * kStagingChunk);  // lo de más, de cuando no había generaciones
            other.used = 0;
        }
        if (!generations && !other.in_flight && other.arena.bytes()) other.arena.trim(0);
    }
    // Sin generaciones, con la GPU parada: si toca, se empieza de cero (y se olvida lo dinámico).
    void maybe_reset(uint64_t frame) {
        if (!needs_reset(frame)) return;
        static int logged = 0;
        if (logged++ < 20 || logged % 100 == 0)
            NVN_LOG("caché de vértices vaciada (%zu entradas, %llu frames desde la anterior)\n", entries_.size(),
                    static_cast<unsigned long long>(frame - reset_frame_));
        gens_[active_].arena.reset();
        gens_[active_].used = 0;
        Generation& other = gens_[active_ ^ 1];  // de cuando había generaciones: la GPU tampoco la lee ya
        other.in_flight = false;
        other.arena.trim(0);
        other.used = 0;
        entries_.clear();
        full_ = false;
        reset_frame_ = frame;
    }
    // Menor y mayor índice (sin el de reinicio de primitiva); sin ninguno, lo > hi.
    static void index_range(const uint8_t* p, size_t size, int type, uint32_t* lo, uint32_t* hi) {
        uint32_t a = ~0u, b = 0;
        if (type == 0) {
            for (size_t k = 0; k < size; k++) a = std::min<uint32_t>(a, p[k]), b = std::max<uint32_t>(b, p[k]);
        } else if (type == 1) {
            const auto* q = reinterpret_cast<const uint16_t*>(p);
            for (size_t k = 0; k < size / 2; k++)
                if (q[k] != 0xFFFF) a = std::min<uint32_t>(a, q[k]), b = std::max<uint32_t>(b, q[k]);
        } else {
            const auto* q = reinterpret_cast<const uint32_t*>(p);
            for (size_t k = 0; k < size / 4; k++)
                if (q[k] != 0xFFFFFFFF) a = std::min(a, q[k]), b = std::max(b, q[k]);
        }
        *lo = a;
        *hi = b;
    }

private:
    struct KeyHash {
        size_t operator()(const std::pair<uint64_t, uint64_t>& k) const { return k.first * 31 ^ k.second; }
    };
    std::unordered_map<std::pair<uint64_t, uint64_t>, Entry, KeyHash> entries_;
    struct Generation {
        Staging arena;
        VkDeviceSize used = 0;
        uint64_t last_submit = 0;  // en vuelo: el último envío que puede leerla
        bool in_flight = false;    // ya no es la activa y la GPU aún puede leerla
    };
    Generation gens_[2];
    int active_ = 0;
    VkDeviceSize limit_ = kGenerationLimit;
    bool full_ = false;  // se rechazó una copia por falta de sitio
    uint64_t reset_frame_ = 0;
};

struct StageBindings {
    BufferRange uniforms[kMaxUniforms]{};
    BufferRange storage[kMaxStorage]{};
    uint64_t textures[kMaxTextures]{};
    bool combined[kMaxTextures]{};
    uint64_t samplers[kMaxTextures]{};
};

struct QueueState {
    cmd::SetRenderTargets rt{};
    bool rt_dirty = true;
    VkViewport viewport{0, 0, 1280, 720, 0, 1};
    VkRect2D scissor{};
    bool scissor_set = false;
    const ProgramStage* stages[kStages] = {};  // por nvnshader::Stage
    uint64_t stage_generation[kStages] = {};
    BlendState blend[8]{};
    ColorState color{};
    ChannelMaskState masks{0xFFFFFFFF};
    DepthStencilState depth_stencil{};
    PolygonState polygon{};
    MultisampleState multisample{};
    std::vector<VertexAttribState> attribs;
    std::vector<VertexStreamState> streams;
    BufferRange vertex_buffers[kMaxVertexBuffers]{};
    StageBindings bindings[kStages];
    DescriptorPool* texture_pool = nullptr;
    DescriptorPool* sampler_pool = nullptr;
    uint32_t stencil_ref[2] = {}, stencil_mask[2] = {0xFF, 0xFF}, stencil_value_mask[2] = {0xFF, 0xFF};
    float blend_color[4] = {};
    bool render_enabled = true;
    ProgramStage* compute = nullptr;
};

struct PipelineKey {
    const ProgramStage* stages[kStages];
    uint64_t generation[kStages];
    VkRenderPass render_pass;
    uint32_t topology;
    PolygonState polygon;
    DepthStencilState depth_stencil;
    ColorState color;
    ChannelMaskState masks;
    BlendState blend[8];
    uint32_t color_count, samples;
    uint8_t alpha_to_coverage;  // estaba en el pipeline pero no en la clave
    uint8_t skip_gs;            // pre-pase sin su geometry shader de compactación
    VertexAttribState attribs[16];
    uint32_t strides[16], divisors[16];
};

struct Layout {
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline = VK_NULL_HANDLE;
    // Gráficos: UBO/SSBO dinámicos (el desplazamiento de cada draw va en el bind), así un draw igual que el
    // anterior salvo por dónde están sus datos reutiliza el set (en la fábrica, ~85 % de los draws).
    bool dynamic = false;
    uint64_t memo_epoch = 0;
    VkDescriptorSet memo_set = VK_NULL_HANDLE;
    std::vector<uint64_t> memo_key;
};

class Executor {
public:
    void run(const std::vector<const CommandList*>& lists);
    uint64_t last_submit() const { return submitted_.load(std::memory_order_acquire); }
    // true si el envío ya terminó (o termina antes del plazo). Sin g_mutex: no espera a la reproducción en curso.
    bool wait_submit(uint64_t submit, uint64_t timeout_ns);
    void wait_idle() {
        std::lock_guard lock(g_mutex);
        wait_all();
        reap_retired();
    }
    void retire_texture(Texture& t);
    void retire_sampler(VkSampler s);
    bool blit(Texture& t, VkImage dst, VkFormat format, VkExtent2D extent, VkCommandBuffer cmd);
    void dump_frame(Texture* presented);
    void profile_frame();

private:
    bool init();
    void begin();
    void maybe_split();
    static constexpr VkDeviceSize kStagingSoftCap = 128ull << 20;
    // Varios envíos por frame: tras tantos draws, el siguiente límite de render pass manda lo grabado a la GPU, que
    // empieza a trabajar mientras se reproduce el resto (con un envío al final, la GPU esperaba a toda la
    // reproducción). Con <datos>/un_envio, uno por envío del juego, como antes.
    static constexpr uint64_t kSplitDraws = 300;
    uint64_t submit_draws_ = 0;  // draws_ al empezar el envío de Vulkan en curso
    bool split_ = true, generations_ = true;  // interruptores, leídos una vez por envío del juego
    void finish();
    void apply(const Command& c);
    void draw(const cmd::Draw& d);
    void clear_color(const cmd::ClearColor& c);
    void clear_depth_stencil(const cmd::ClearDepthStencil& c);
    void copy_texture(const cmd::CopyTexture& c);
    void barrier();
    void full_barrier();
    void nvn_barrier(int bits);
    // full_area: renderArea entero aunque la mirada adelante diga otra cosa (borrados dentro del pase).
    bool ensure_render_pass(bool full_area = false);
    void end_render_pass();
    bool ensure_image(Texture& t, VkCommandBuffer cmd);
    void upload(Texture& t);
    VkImageView view(Texture& t, const TextureViewDesc& v, bool attachment);
    VkSampler sampler(Sampler* s);
    VkRenderPass render_pass(const VkFormat* colors, uint32_t count, VkFormat depth, VkSampleCountFlagBits samples,
                             const PassOps& ops = default_pass_ops());
    Layout& layout(const ProgramStage* const* stages);
    VkPipeline pipeline(const cmd::Draw& d, VkRenderPass rp, Layout& l, uint32_t color_count, VkSampleCountFlagBits samples);
    bool resolve_texture(Stage s, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp);
    VkDescriptorSet descriptor_set(VkDescriptorSetLayout layout);
    void log_draw(const cmd::Draw& d);
    bool traced_in_list(const char* var) const;
    static constexpr int kCounterSamplesPassed = 1;  // NVNcounterType
    static constexpr int kCounterTimestamp = 0;
    uint64_t samples_passed_ = 0;
    uint64_t watch_addr_ = 0;
    bool full_trace_ = false;  // DEPURACIÓN: la traza pedida vuelca también los render targets
    struct PrepassArgs {
        uint64_t indices;
        uint32_t count;
        int type;
    };
    std::unordered_map<uint64_t, PrepassArgs> prepass_args_;  // DEPURACIÓN: args de la compactación -> índices originales
    struct CulledDraw {
        cmd::Draw draw;  // el draw directo del pre-pase (malla entera)
        uint64_t frame;
    };
    std::unordered_map<uint64_t, CulledDraw> culled_draws_;  // args que escribe su geometry shader -> ese draw
    struct TracedGpuDraw {
        uint64_t args_address;
        const uint8_t* args;
        const uint8_t* indices;
        int type;
    };
    std::vector<TracedGpuDraw> traced_gpu_draws_;
    // DEPURACIÓN: con <datos>/profundidad_lequal, los draws con depth EQUAL usan LEQUAL (prueba en caliente).
    bool depth_lequal_ = false;
    static constexpr uint32_t kDepthEqual = 3, kDepthLequal = 4;  // NVNdepthFunc  // DEPURACIÓN: al acabar el frame trazado, quién lee el primer contador condicional
    std::vector<std::pair<Texture*, uint32_t>> tex_dumps_;  // DEPURACIÓN: texturas por volcar (NIER_VOLCAR_TEX)

    bool ready_ = false;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    // Envíos en vuelo: cada uno con su command buffer, fence, staging y pools de descriptores. El juego
    // sigue mientras la GPU trabaja; solo se espera al reutilizar un contexto, cuando la CPU necesita un
    // resultado (write-back de compute, volcados) o cuando el juego espera un NVNsync.
    // pending, submit y waiters los cambia solo quien tiene g_mutex, y además con fence_mutex_: los hilos del juego
    // que esperan un envío (wait_submit) los leen con fence_mutex_, sin g_mutex.
    // Con varios envíos por frame (~5 en el mundo abierto) 12 cubren unos 2 frames en vuelo; más adelantada, la CPU
    // espera en begin(). Staging en trozos de 4 MB (un envío partido usa 1-2 MB) y cada contexto guarda lo que usó
    // la última vez hasta 16 MB, como antes un trozo: ~48 MB fijos (eran 6 x 16 MB).
    static constexpr VkDeviceSize kSlotStagingChunk = 4 << 20;
    static constexpr int kInFlight = 12;
    struct InFlight {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        Staging staging{kSlotStagingChunk};
        std::vector<VkDescriptorPool> pools;
        size_t pool = 0;
        bool pending = false;
        uint64_t submit = 0;
        int waiters = 0;  // hilos esperando su fence fuera de los locks: hasta que salgan no se reinicia
        // Informes TIMESTAMP de este envío: su valor lo escribe la GPU (vkCmdWriteTimestamp) y se copia a la memoria
        // del juego cuando el envío termina (note_completed), antes de que el juego lo vea terminado.
        VkQueryPool queries = VK_NULL_HANDLE;
        struct Stamp {
            uint32_t query;
            uint64_t address;
            int64_t extra;  // ticks de 614,4 MHz de más (el coste de LSFG, en el informe del final del frame)
            bool end;       // el del final del frame (16 bytes detrás del del comienzo)
        };
        std::vector<Stamp> stamps;
    };
    static constexpr uint32_t kStampsPerSlot = 64;
    // Tiempos de GPU para la resolución dinámica del juego (ver ReportCounter): un tick de la GPU en ticks de 614,4 MHz
    // (las unidades de NVN), el comienzo del último frame y lo que tardó.
    double stamp_scale_ = 32.0;
    uint64_t stamp_mask_ = ~0ull, frame_start_ticks_ = 0;
    std::atomic<int64_t> frame_gpu_ns_{0};
    bool gpu_stamps_ = true;     // interruptor ts_cpu: los de antes (reloj de la CPU al reproducir)
    bool skip_compaction_gs_ = true;  // interruptor sin_salto_gs
    int64_t stamp_extra_ = 0;    // lo que se suma al final del frame en este envío del juego
    int run_stamps_ = 0;         // informes TIMESTAMP vistos en este envío del juego
    uint64_t last_stamp_addr_ = 0;  // el del comienzo: el del final va 16 bytes detrás (main+0x9bbb08 lee +0x8 y +0x18)
    int scene_vp_w_ = 0, scene_vp_h_ = 0;  // perfil: el viewport de la escena más grande desde el último
    int scene_vp_min_w_ = 1 << 30;         // ...y el ancho más pequeño
    void write_stamps(InFlight& f);
    // DEPURACIÓN: perfil de GPU por pase (<datos>/perfil_gpu, se borra solo): un timestamp antes de cada render pass y
    // de cada dispatch de un frame, y el log con lo que tardó cada grupo de pases.
    static constexpr uint32_t kProfMarks = 1024;
    struct ProfMark {
        std::string label;
        uint64_t draws = 0;
    };
    VkQueryPool prof_pool_ = VK_NULL_HANDLE;
    std::vector<ProfMark> prof_marks_;
    uint64_t prof_frame_ = ~0ull, prof_done_submit_ = 0, prof_mark_draws_ = 0;
    bool prof_closed_ = false;
    void prof_gpu_mark(const char* kind);
    void prof_gpu_report();
    int64_t reserve_margin_ns();
    int64_t reserve_margin_ = 2000000;
    std::chrono::steady_clock::time_point reserve_next_{};
    InFlight inflight_[kInFlight];
    std::mutex fence_mutex_;
    std::condition_variable fence_cv_;  // waiters volvió a 0
    // Recursos que el juego soltó, pendientes de que la GPU termine el último envío que pudo usarlos.
    struct Retired {
        uint64_t serial = 0;  // textura: sus vistas (y los framebuffers que las usan) caen con ella
        VkImage image = VK_NULL_HANDLE;
        GpuAllocation memory;
        uint64_t bytes = 0;
        VkSampler sampler = VK_NULL_HANDLE;
        uint64_t submit = 0;
    };
    std::vector<Retired> retired_;
    void reap_retired();
    // Caché de pipelines del driver, guardada en la SD: sin ella cada sesión recompila todos los pipelines la
    // primera vez que se ven (tirones al entrar en cada zona).
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    VkPipeline bound_graphics_ = VK_NULL_HANDLE;  // el enlazado en cmd_
    struct DynamicState {
        VkViewport viewport;
        VkRect2D scissor;
        float blend[4];
        uint32_t ref[2], write_mask[2], compare_mask[2];
    };
    DynamicState bound_dynamic_{};  // el puesto en cmd_ (válido si dynamic_valid_)
    VkBuffer bound_vb_buf_[16] = {};  // vertex buffers enlazados en cmd_
    VkDeviceSize bound_vb_off_[16] = {};
    bool dynamic_valid_ = false;
    uint64_t descriptor_epoch_ = 1;  // cambia en cada begin(): los sets recordados por los layouts caducan
    uint64_t pipelines_created_ = 0, pipelines_saved_ = 0, last_pipeline_frame_ = 0;
    std::atomic<bool> cache_saving_{false};
    void load_pipeline_cache();
    void save_pipeline_cache();
    // Cada pool (4096 sets) cuesta unos MB de memoria de la GPU: tope por envío y lo que se guarda tras un pico.
    static constexpr size_t kMaxDescriptorPools = 32, kKeptDescriptorPools = 2;
    // Memoria que el driver tiene reservada para el proceso (incluye la interna, que no pasa por mis contadores).
    long kgsl_logged_mb_ = 0;
    std::chrono::steady_clock::time_point kgsl_checked_{};
    void watch_kgsl();
    int cur_ = 0;
    // Envíos a la cola y el último terminado (todos los anteriores también: la señal de un fence cubre lo enviado
    // antes a la cola). completed_ lo adelanta también quien espera un envío desde otro hilo.
    std::atomic<uint64_t> submitted_{0}, completed_{0};
    void note_completed(uint64_t submit) {  // con fence_mutex_
        InFlight* done[kInFlight];
        int n = 0;
        for (InFlight& f : inflight_)
            if (f.pending && f.submit <= submit && !f.stamps.empty()) done[n++] = &f;
        std::sort(done, done + n, [](const InFlight* a, const InFlight* b) { return a->submit < b->submit; });
        for (int k = 0; k < n; k++) write_stamps(*done[k]);
        uint64_t c = completed_.load(std::memory_order_relaxed);
        while (c < submit && !completed_.compare_exchange_weak(c, submit, std::memory_order_acq_rel)) {
        }
    }
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool recording_ = false;
    Staging* staging_ = nullptr;
    void wait_slot(InFlight& f);
    void wait_all();
    void finish(bool wait);
    QueueState st_;
    bool in_render_pass_ = false;
    VkExtent2D rp_extent_{};
    VkRenderPass rp_ = VK_NULL_HANDLE;  // el de LOAD/STORE (compatible con el empezado): pipelines y framebuffers
    uint32_t rp_color_count_ = 0;
    VkSampleCountFlagBits rp_samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkRect2D rp_area_{};          // renderArea del pase abierto
    bool rp_area_full_ = true;    // ...y si es el framebuffer entero
    Texture* rp_targets_[9] = {};  // texturas adjuntas al pase abierto
    uint32_t rp_target_count_ = 0;

    // --- Optimizaciones del tiler (Adreno) ---
    // Cada una con su interruptor en vivo (flags.h): por defecto activas, <datos>/sin_<nombre> la apaga. Se leen
    // una vez por envío (read_flags, al empezar run()); todas pueden cambiar entre envíos: nada de su estado pasa
    // de un envío al siguiente, salvo sin_ubwc, que solo cuenta para las imágenes que se creen después.
    struct TilerOptions {
        bool clear_diferido = true;    // borrado completo -> loadOp CLEAR del siguiente pase con ese target
        bool rt_igual = true;          // volver a enlazar los mismos targets no corta el pase
        bool barrera_perezosa = true;  // una barrera dentro del pase solo lo corta si hace falta
        bool store_ops = true;         // STORE_OP_NONE para lo que el pase no escribe
        bool area = true;              // renderArea = lo que dibujan los draws del pase
        bool ubwc = true;              // profundidad/stencil sin MUTABLE_FORMAT (compresión UBWC)
    };
    TilerOptions opt_;
    void read_flags();
    // Lo que se está reproduciendo en run() (cada run() con su número) y el comando en curso: la mirada adelante de
    // los pases recorre lo que queda.
    const std::vector<const CommandList*>* run_lists_ = nullptr;
    const Command* cur_command_ = nullptr;
    size_t cur_list_ = 0, cur_index_ = 0;  // posición en run(): la misma lista puede ir dos veces en un envío
    uint64_t run_serial_ = 0;
    // Borrados diferidos: un ClearColor/ClearDepthStencil de todo el target antes de que empiece su pase no abre
    // un pase (LOAD + borrar + STORE) sino que espera: el siguiente pase que lo use como adjunto (misma vista) lo
    // hace con loadOp CLEAR, sin leer nada. Si antes se muestrea, se copia, se presenta o acaba el envío, se hace
    // con vkCmdClearColorImage/DepthStencilImage fuera del pase. Como mucho uno por textura (una sola vista).
    struct PendingClear {
        Texture* texture;
        TextureViewDesc view;
        VkImageAspectFlags aspects;
        VkClearValue value;
    };
    std::vector<PendingClear> pending_clears_;
    bool defer_clear(Texture* t, const TextureViewDesc& v, VkImageAspectFlags aspects, const VkClearValue& value);
    void flush_clears(const Texture* only);  // los de esa textura (todos si es nula), ya
    void touch_targets();
    VkExtent2D target_extent() const;
    // Barrera de NVN aplazada al final del pase abierto (nvn_barrier), y si en el pase ya se escribió storage
    // desde la última barrera (entonces no se aplaza).
    bool pending_barrier_ = false, storage_since_barrier_ = false;
    bool splitting_ = false;  // finish() de un envío partido a mitad (maybe_split): sin vaciar overlays
    bool is_pass_target(const Texture* t) const {
        for (uint32_t k = 0; k < rp_target_count_; k++)
            if (rp_targets_[k] == t) return true;
        return false;
    }
    // Mirada adelante desde el comando en curso hasta el siguiente SetRenderTargets con otros targets (el pase no
    // puede seguir más allá): qué adjuntos escriben sus draws y borrados, si hay borrados y dónde pueden dibujar.
    // Un pase con esos targets que empiece antes de `end` en el mismo envío solo ve una parte de eso: le vale
    // (un pase que se corta por una subida o una barrera reutiliza la del primero).
    struct PassScan {
        uint64_t run = 0;  // run_serial_ del envío (0: ninguna)
        cmd::SetRenderTargets rt{};
        size_t end_list = 0, end_cmd = 0;
        uint32_t colors_written = 0;  // bit por render target
        bool depth_written = false, cleared = false, drawn = false;
        int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // unión de lo que pueden tocar sus draws (sin recortar)
    };
    PassScan scan_;
    const PassScan& scan_pass();
    // Perfil del tiler (por 300 frames).
    uint64_t prof_clear_loads_ = 0, prof_clear_flushes_ = 0, prof_store_none_ = 0, prof_area_cut_ = 0;
    uint64_t prof_barriers_deferred_ = 0, prof_barrier_splits_ = 0, prof_rt_kept_ = 0;
    std::unordered_map<std::string, VkRenderPass> render_passes_;
    std::unordered_map<std::string, VkFramebuffer> framebuffers_;
    std::unordered_map<BytesKey<ViewKeyData>, VkImageView, BytesKeyHash<ViewKeyData>> views_;
    std::unordered_map<BytesKey<LayoutKeyData>, Layout, BytesKeyHash<LayoutKeyData>> layouts_;
    std::unordered_map<uint64_t, VkPipeline> pipelines_;
    Texture dummy_;
    VkSampler default_sampler_ = VK_NULL_HANDLE;
    uint32_t allocations_ = 0;
    BufferCache buffer_cache_;
    // Uniform/storage buffers ya copiados en el envío en curso: el juego no puede cambiar esa memoria
    // hasta que la GPU lo ejecute, así que varios draws que enlazan el mismo rango comparten la copia.
    // Se vacía al empezar cada envío y con CopyBuffer (que sí escribe memoria a mitad de envío).
    // Tabla plana (sin un nodo por copia: miles de malloc/free por frame) que se vacía cambiando de época.
    // Llena, no se apunta más: esos rangos se vuelven a copiar.
    class CopiedTable {
    public:
        const std::pair<VkBuffer, VkDeviceSize>* find(uint64_t address, uint64_t size) const {
            for (size_t i = slot(address, size), n = 0; n < kProbe; i = (i + 1) & (kSize - 1), n++) {
                const Slot& e = slots_[i];
                if (e.epoch != epoch_) return nullptr;
                if (e.address == address && e.size == size) return &e.value;
            }
            return nullptr;
        }
        void insert(uint64_t address, uint64_t size, VkBuffer buffer, VkDeviceSize offset) {
            for (size_t i = slot(address, size), n = 0; n < kProbe; i = (i + 1) & (kSize - 1), n++) {
                Slot& e = slots_[i];
                if (e.epoch != epoch_ || (e.address == address && e.size == size)) {
                    e = {address, size, epoch_, {buffer, offset}};
                    return;
                }
            }
        }
        void clear() { epoch_++; }

    private:
        static constexpr size_t kSize = 8192, kProbe = 32;
        struct Slot {
            uint64_t address = 0, size = 0, epoch = 0;
            std::pair<VkBuffer, VkDeviceSize> value{};
        };
        static size_t slot(uint64_t address, uint64_t size) {
            uint64_t h = (address ^ (size * 0x9E3779B97F4A7C15ULL)) * 0xff51afd7ed558ccdULL;
            return static_cast<size_t>(h >> 51) & (kSize - 1);
        }
        std::vector<Slot> slots_ = std::vector<Slot>(kSize);
        uint64_t epoch_ = 1;
    };
    CopiedTable copied_;
    uint64_t prof_copied_bytes_ = 0;
    // Lo que escriben los compute en storage buffers se queda en la GPU: cada rango escrito tiene una
    // copia persistente (overlay) que se enlaza tal cual en los SSBO/UBO que lo lean después, en vez de
    // esperar a la GPU y copiarlo de vuelta a la memoria del juego (5 vaciados de la GPU por frame en el
    // prólogo). La copia se crea con lo que hay en la memoria del juego y se descarta si la CPU la
    // reescribe (su huella deja de coincidir). Se alinea como la dirección original (módulo 256) para
    // que un buffer enlazado en una dirección alineada lo siga estando.
    struct Overlay {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;  // dónde está la dirección de inicio dentro de buffer
        uint8_t* data = nullptr;  // la misma memoria, mapeada (DEPURACIÓN: leer lo que escribió la GPU)
        uint64_t size = 0, cpu_hash = 0, checked = 0;
    };
    std::map<uint64_t, Overlay> overlays_;
    // Cuántos overlays tocan cada bloque de 64 KB (por hash: una colisión solo hace mirar el mapa). Casi todas las
    // búsquedas fallan, y así se descartan sin recorrer el mapa. Todo cambio de overlays_ pasa por aquí.
    uint32_t overlay_filter_[1 << 16] = {};
    static size_t overlay_slot(uint64_t granule) { return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> 48); }
    void filter_overlay(uint64_t address, uint64_t size, int delta) {
        for (uint64_t g = address >> 16, end = (address + std::max<uint64_t>(size, 1) - 1) >> 16; g <= end; g++)
            overlay_filter_[overlay_slot(g)] += delta;
    }
    std::map<uint64_t, Overlay>::iterator erase_overlay(std::map<uint64_t, Overlay>::iterator it) {
        filter_overlay(it->first, it->second.size, -1);
        return overlays_.erase(it);
    }
    void compact_overlays();
    void clear_overlays() {
        overlays_.clear();
        memset(overlay_filter_, 0, sizeof overlay_filter_);
    }
    Staging overlay_arena_;
    VkDeviceSize overlay_used_ = 0;
    // Overlay que contiene [address, address+size), con el desplazamiento de address en su buffer.
    const Overlay* find_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset);
    const Overlay* make_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset);
    // CopyBuffer cuyo origen está en un overlay: copia en la GPU a un overlay del destino.
    bool copy_from_overlay(uint64_t src, uint64_t dst, uint64_t size);
    // [address, address+size) con lo que haya de la GPU encima de la memoria del juego, en un buffer temporal
    // (argumentos indirectos escritos solo en parte por la GPU). false si ningún overlay lo toca.
    bool gather_overlays(uint64_t address, uint64_t size, VkBuffer* buffer, VkDeviceSize* offset);
    uint64_t submit_index_ = 0;  // envíos ejecutados: la caché de vértices revisa una vez por envío
    std::vector<Texture*> frame_targets_, frame_uploads_;  // para los volcados de depuración
    void dump(Texture& t, const std::string& name, uint32_t layer = 0, int forced_mip = -1);
    void dump_raw(Texture& t, const std::string& name);
    void dump_depth(Texture& t, const std::string& name);
    void trace_draw(const cmd::Draw& d);  // DEPURACIÓN: estado completo de cada draw del frame trazado
    void dispatch(const cmd::Dispatch& v);
    VkPipeline compute_pipeline(const ProgramStage& cs, Layout& l);
    std::map<std::pair<const ProgramStage*, uint64_t>, VkPipeline> compute_pipelines_;  // (etapa, generación)
    // Frame trazado: el que sigue a la aparición de <datos>/trazar (se borra al verlo); ninguno si no.
    uint64_t trace_frame_ = UINT64_MAX;
    int traced_draws_ = 0;
    // Perfil: tiempo total en run() (incluye la espera a la GPU) y en las esperas a la GPU.
    uint64_t prof_run_us_ = 0, prof_gpu_wait_us_ = 0, prof_submits_ = 0, prof_draws_ = 0;
    // Los tramos de cada draw (~12 relojes por draw) solo se miden en uno de cada kProfileEvery frames; el log
    // los da por frame medido, con el mismo sentido que antes.
    static constexpr uint64_t kProfileEvery = 8;
    bool prof_timing_ = false;                   // el frame en curso se mide
    uint64_t prof_timed_frames_ = 0, prof_timed_frame_ = UINT64_MAX;
    std::chrono::steady_clock::time_point prof_now() const {
        return prof_timing_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    }
    uint64_t prof_section_us_[6] = {};  // en ns (en µs, cada tramo perdía ~0,5 µs por draw)
    uint64_t prof_sub_us_[4] = {};  // descriptores: buffers, texturas, actualizar+enlazar, reservar set
    uint64_t prof_dispatch_us_ = 0, prof_draw_total_us_ = 0;
    void prof_mark_sub(int k, std::chrono::steady_clock::time_point& t) {
        if (!prof_timing_) return;
        auto now = std::chrono::steady_clock::now();
        prof_sub_us_[k] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
        t = now;
    }
    uint64_t prof_render_passes_ = 0, prof_writebacks_ = 0, prof_dispatches_ = 0;
    std::map<uint64_t, uint64_t> gpu_written_ranges_;  // DEPURACIÓN: [inicio, fin) escritos por compute
    uint64_t prof_reads_[6] = {};                       // lecturas de eso: vértices, índices, ubo, ssbo, indirecto, cs-ssbo
    void note_read(int kind, uint64_t a, uint64_t size) {
        auto it = gpu_written_ranges_.upper_bound(a);
        if (it == gpu_written_ranges_.begin()) return;
        --it;
        if (a < it->second && a + size > it->first) {
            prof_reads_[kind]++;
            if (a < it->first || a + size > it->second) prof_partial_++;
        }
    }
    uint64_t prof_partial_ = 0;
    uint64_t prof_texture_bytes_ = 0;
    uint64_t prof_disabled_draws_ = 0;  // DEPURACIÓN: draws descartados por SetRenderEnable(false) en la traza
    void prof_mark(int section, std::chrono::steady_clock::time_point& t) {
        if (!prof_timing_) return;
        auto now = std::chrono::steady_clock::now();
        prof_section_us_[section] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
        t = now;
    }
    uint32_t trace_counts_[std::variant_size_v<Command>] = {};

public:
    uint64_t frame_ = 0;

private:
    uint64_t draws_ = 0, skipped_draws_ = 0, dispatches_ = 0;
};

bool Executor::init() {
    VkContext* c = vk();
    if (!c) return false;
    load_pipeline_cache();
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->queue_family;
    vkCreateCommandPool(c->device, &pci, nullptr, &pool_);
    for (InFlight& f : inflight_) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(c->device, &ai, &f.cmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(c->device, &fi, nullptr, &f.fence);
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kStampsPerSlot;
        if (vkCreateQueryPool(c->device, &qi, nullptr, &f.queries) != VK_SUCCESS) f.queries = VK_NULL_HANDLE;
    }
    {
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c->gpu, &n, nullptr);
        std::vector<VkQueueFamilyProperties> families(n);
        vkGetPhysicalDeviceQueueFamilyProperties(c->gpu, &n, families.data());
        const uint32_t bits = c->queue_family < n ? families[c->queue_family].timestampValidBits : 0;
        stamp_mask_ = bits >= 64 ? ~0ull : (1ull << bits) - 1;
        stamp_scale_ = c->props.limits.timestampPeriod * 0.6144;  // ns por tick x ticks de 614,4 MHz por ns
        if (!bits) for (InFlight& f : inflight_) f.queries = VK_NULL_HANDLE;
        VkQueryPoolCreateInfo pq{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        pq.queryType = VK_QUERY_TYPE_TIMESTAMP;
        pq.queryCount = kProfMarks;
        if (!bits || vkCreateQueryPool(c->device, &pq, nullptr, &prof_pool_) != VK_SUCCESS) prof_pool_ = VK_NULL_HANDLE;
        NVN_LOG("timestamps de la GPU: %u bits, %.3f ns por tick (x%.2f a ticks de NVN)\n", bits,
                c->props.limits.timestampPeriod, stamp_scale_);
    }
    staging_ = &inflight_[0].staging;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.maxLod = VK_LOD_CLAMP_NONE;
    vkCreateSampler(c->device, &si, nullptr, &default_sampler_);

    // Textura de reserva (1x1 negra) para lo que el shader pide y no está enlazado.
    dummy_.desc.format = 0x25;  // RGBA8
    dummy_.desc.width = dummy_.desc.height = dummy_.desc.depth = dummy_.desc.levels = 1;
    dummy_.desc.target = kTarget2D;
    dummy_.layout = compute_layout(1, 1, 1, 1, 1, 4, 1, 1, true, 4, false);
    dummy_.gpu_written = true;
    ready_ = true;
    return true;
}

void Executor::wait_slot(InFlight& f) {
    if (!f.pending) return;
    auto t0 = std::chrono::steady_clock::now();
    vkWaitForFences(vk()->device, 1, &f.fence, VK_TRUE, UINT64_MAX);
    prof_gpu_wait_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard lock(fence_mutex_);
    note_completed(f.submit);  // antes de dejar de estar en vuelo: quien no lo encuentre ya lo ve terminado
    f.pending = false;
}

void Executor::wait_all() {
    for (int k = 1; k <= kInFlight; k++) wait_slot(inflight_[(cur_ + k) % kInFlight]);  // del más antiguo
}

void Executor::begin() {
    if (recording_) return;
    InFlight& f = inflight_[cur_];
    wait_slot(f);  // su staging y sus descriptores ya no los lee la GPU
    const size_t used_chunks = f.staging.used_chunks();
    f.staging.reset();
    f.staging.trim(std::max<size_t>(used_chunks, 1), kStagingChunk);
    // El tope de staging es por envío: los demás contextos guardaban su pico hasta reutilizarse (6 x 128 MB).
    // Los que la GPU ya terminó lo devuelven ahora (nadie en la CPU lee su staging tras un begin()).
    // Se miran también los demás envíos terminados, del más antiguo al más nuevo (acaban en orden): completed_
    // avanza y lo retirado se libera antes.
    bool running = false;
    for (int k = 1; k < kInFlight; k++) {
        InFlight& o = inflight_[(cur_ + k) % kInFlight];
        if (o.pending && !running) {
            if (vkGetFenceStatus(vk()->device, o.fence) == VK_SUCCESS)
                wait_slot(o);  // no bloquea: ya acabó
            else
                running = true;  // los posteriores tampoco
        }
        if (!o.pending && o.staging.bytes() > kStagingChunk) {
            o.staging.reset();
            o.staging.trim(1);
        }
    }
    buffer_cache_.reap(completed_.load(std::memory_order_acquire), generations_);
    reap_retired();
    while (f.pools.size() > kKeptDescriptorPools) {  // como el staging: no guardar el pico de un envío enorme
        vkDestroyDescriptorPool(vk()->device, f.pools.back(), nullptr);
        f.pools.pop_back();
    }
    copied_.clear();
    for (VkDescriptorPool p : f.pools) vkResetDescriptorPool(vk()->device, p, 0);
    f.pool = 0;
    cmd_ = f.cmd;
    submit_draws_ = draws_;
    bound_graphics_ = VK_NULL_HANDLE;  // un command buffer nuevo no tiene nada enlazado
    dynamic_valid_ = false;
    memset(bound_vb_buf_, 0, sizeof bound_vb_buf_);
    descriptor_epoch_++;
    staging_ = &f.staging;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd_, 0);
    vkBeginCommandBuffer(cmd_, &bi);
    recording_ = true;
    f.stamps.clear();  // ya escritos (wait_slot) o de un envío que no se mandó
    if (f.queries) vkCmdResetQueryPool(cmd_, f.queries, 0, kStampsPerSlot);
    // Lo de este envío va después de todo lo anterior (subidas que pisan texturas que otro envío aún
    // lee, render targets que se muestrean…): antes lo garantizaba esperar a la GPU en cada envío.
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
    if (!dummy_.image && ensure_image(dummy_, cmd_)) {
        VkClearColorValue black{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, dummy_.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
    }
}

void Executor::finish() { finish(false); }

void Executor::finish(bool wait) {
    if (!recording_) return;
    end_render_pass();
    flush_clears(nullptr);  // borrados diferidos: presentar (otro command buffer) o la CPU pueden leer la textura
    vkEndCommandBuffer(cmd_);
    InFlight& f = inflight_[cur_];
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VkContext* c = vk();
    {
        // Reiniciar un fence sí necesita que nadie lo use a la vez (esperarlo, no). Ya está señalado (begin()
        // esperó este contexto) y no está en vuelo, así que no entra nadie nuevo y quien quede sale enseguida.
        std::unique_lock lock(fence_mutex_);
        fence_cv_.wait(lock, [&] { return f.waiters == 0; });
        vkResetFences(c->device, 1, &f.fence);
    }
    VkResult r;
    {
        std::lock_guard qlock(vk_queue_mutex());
        r = vkQueueSubmit(c->queue, 1, &si, f.fence);
    }
    recording_ = false;
    if (r != VK_SUCCESS) {
        NVN_LOG("vkQueueSubmit: %s\n", vk_result_name(r));
    } else {
        {
            std::lock_guard lock(fence_mutex_);
            f.submit = submitted_.load(std::memory_order_relaxed) + 1;
            f.pending = true;
        }
        submitted_.store(f.submit, std::memory_order_release);
        prof_submits_++;
        if (wait) wait_slot(f);
    }
    cur_ = (cur_ + 1) % kInFlight;
    if (NVN_FLAG("registro")) watch_kgsl();  // DEPURACIÓN: lee /sys de la GPU cada medio segundo
    // Sin sitio para overlays nuevos (grow() falló) también se vacía: overlay_used_ ya no subiría y los overlays
    // quedarían desactivados hasta reiniciar.
    // Solo al acabar el envío del juego (no al partirlo a mitad de frame): lo que la GPU escribió antes en este
    // frame vive en los overlays, y vaciarlos entre ese draw y el que lo lee le daba datos viejos (índices y
    // argumentos fuera de rango: la GPU se colgaba al cargar las Ruinas).
    const bool overlays_full = !splitting_ && (overlay_used_ > (64ull << 20) || overlay_arena_.failed());
    const bool cache_reset = !generations_ && buffer_cache_.needs_reset(frame_);  // con generaciones no se espera
    if (cache_reset || overlays_full) {
        wait_all();
        if (!generations_) buffer_cache_.maybe_reset(frame_);
        if (overlays_full) compact_overlays();
    }
}

// Arena de overlays llena (con la GPU parada). Antes se tiraban todos y se rehacían desde la memoria del juego, que la
// GPU nunca actualiza: lo que la GPU guarda de un frame al siguiente (la exposición automática, partículas…) volvía a
// un valor viejo y la pantalla bajaba de brillo de golpe al mover la cámara (al moverla aparecen rangos nuevos y la
// arena se llena antes). Ahora los usados hace poco se copian a la arena vacía; los demás se rehacen si vuelven.
void Executor::compact_overlays() {
    constexpr uint64_t kKeepSubmits = 8;       // usados en los últimos envíos del juego
    constexpr uint64_t kKeepBytes = 32ull << 20;  // y como mucho esto: lo que pase se rehace como antes
    struct Kept {
        uint64_t address, cpu_hash, checked;
        std::vector<uint8_t> bytes;
    };
    std::vector<Kept> kept;
    std::vector<std::pair<uint64_t, uint64_t>> recent;  // (último uso, dirección), los más recientes primero
    for (const auto& [address, o] : overlays_)
        if (submit_index_ - o.checked <= kKeepSubmits && o.data) recent.push_back({o.checked, address});
    std::sort(recent.begin(), recent.end(), std::greater<>());
    uint64_t bytes = 0;
    for (const auto& [checked, address] : recent) {
        const Overlay& o = overlays_[address];
        if (bytes + o.size > kKeepBytes) continue;
        bytes += o.size;
        kept.push_back({address, o.cpu_hash, o.checked, std::vector<uint8_t>(o.data, o.data + o.size)});  // memoria coherente
    }
    const size_t before = overlays_.size();
    clear_overlays();
    overlay_arena_.reset();
    overlay_used_ = 0;
    size_t restored = 0;
    for (const Kept& k : kept) {
        Staging::Alloc a = overlay_arena_.alloc(k.bytes.size() + 256, 256);
        if (!a.data) break;
        overlay_used_ += k.bytes.size() + 256;
        const VkDeviceSize skew = k.address & 255;
        memcpy(a.data + skew, k.bytes.data(), k.bytes.size());
        Overlay& o = overlays_[k.address];
        o.buffer = a.buffer;
        o.offset = a.offset + skew;
        o.data = a.data + skew;
        o.size = k.bytes.size();
        o.cpu_hash = k.cpu_hash;
        o.checked = k.checked;
        filter_overlay(k.address, o.size, +1);
        restored++;
    }
    static int logged = 0;
    if (logged++ < 10 || logged % 100 == 0)
        NVN_LOG("overlays: arena llena (%zu); se conservan %zu usados hace poco (%.1f MB)\n", before, restored, bytes / 1048576.0);
}

// DEPURACIÓN: marca de tiempo antes de un render pass o un dispatch (fuera de un pase) en el frame que se perfila.
void Executor::prof_gpu_mark(const char* kind) {
    if (!prof_pool_) return;
    if (prof_frame_ == ~0ull) {  // ¿empezar? se mira al empezar el primer pase de cada frame, no en cada pase
        static uint64_t checked = ~0ull;
        if (checked == frame_) return;
        checked = frame_;
        const char* data = getenv("NIER_DATA");
        const std::string file = std::string(data ? data : ".") + "/perfil_gpu";
        if (access(file.c_str(), F_OK) != 0) return;
        unlink(file.c_str());
        prof_frame_ = frame_;
        prof_marks_.clear();
        prof_closed_ = false;
        vkCmdResetQueryPool(cmd_, prof_pool_, 0, kProfMarks);
    }
    if (prof_closed_) return;
    if (!prof_marks_.empty()) prof_marks_.back().draws = draws_ - prof_mark_draws_;
    prof_mark_draws_ = draws_;
    const bool end = kind[0] == 'f' || frame_ != prof_frame_ || prof_marks_.size() + 1 >= kProfMarks;
    char label[96];
    if (end) {
        snprintf(label, sizeof label, "fin");
    } else if (kind[0] == 'c') {
        snprintf(label, sizeof label, "compute");
    } else {
        const Texture* c0 = st_.rt.count > 0 ? st_.rt.colors[0] : nullptr;
        const Texture* d = st_.rt.depth;
        const Texture* any = c0 ? c0 : d;
        snprintf(label, sizeof label, "pase %dx%d %s%s", any ? any->desc.width : 0, any ? any->desc.height : 0,
                 c0 ? (format_info(c0->desc.format) ? format_info(c0->desc.format)->name : "?") : "-", d ? " +z" : "");
    }
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, prof_pool_, static_cast<uint32_t>(prof_marks_.size()));
    prof_marks_.push_back({label, 0});
    if (end) {
        prof_closed_ = true;
        prof_done_submit_ = submitted_.load(std::memory_order_relaxed) + 1;  // el envío que se está grabando
    }
}

// DEPURACIÓN: con el frame perfilado ya terminado en la GPU, el log: cada grupo (tipo de pase o compute), su tiempo y
// sus draws, de más a menos.
void Executor::prof_gpu_report() {
    if (!prof_closed_ || prof_frame_ == ~0ull || completed_.load(std::memory_order_acquire) < prof_done_submit_) return;
    const uint32_t n = static_cast<uint32_t>(prof_marks_.size());
    std::vector<uint64_t> t(n);
    const VkResult r = vkGetQueryPoolResults(vk()->device, prof_pool_, 0, n, n * sizeof(uint64_t), t.data(), sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT);
    prof_frame_ = ~0ull;
    if (r != VK_SUCCESS || n < 2) return;
    struct Group {
        double ms = 0;
        uint64_t draws = 0, count = 0;
    };
    std::map<std::string, Group> groups;
    double total = 0;
    for (uint32_t k = 0; k + 1 < n; k++) {
        const double ms = static_cast<double>((t[k + 1] - t[k]) & stamp_mask_) * vk()->props.limits.timestampPeriod / 1e6;
        Group& g = groups[prof_marks_[k].label];
        g.ms += ms, g.draws += prof_marks_[k].draws, g.count++;
        total += ms;
    }
    std::vector<std::pair<std::string, Group>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    NVN_LOG("perfil GPU del frame %llu: %.2f ms en %u marcas\n", static_cast<unsigned long long>(frame_), total, n);
    for (size_t k = 0; k < sorted.size() && k < 24; k++)
        NVN_LOG("perfil GPU:  %6.2f ms  %4llu x  %5llu draws  %s\n", sorted[k].second.ms,
                static_cast<unsigned long long>(sorted[k].second.count), static_cast<unsigned long long>(sorted[k].second.draws),
                sorted[k].first.c_str());
}

// Margen que se reserva además del coste de LSFG (ns): <datos>/lsfg_reserva con milisegundos, leído como mucho una vez
// por segundo; sin fichero, 2 ms.
int64_t Executor::reserve_margin_ns() {
    const auto now = std::chrono::steady_clock::now();
    if (now >= reserve_next_) {
        reserve_next_ = now + std::chrono::seconds(1);
        const char* data = getenv("NIER_DATA");
        double ms = 2.0;
        if (FILE* f = fopen((std::string(data ? data : ".") + "/lsfg_reserva").c_str(), "r")) {
            if (fscanf(f, "%lf", &ms) != 1) ms = 2.0;
            fclose(f);
        }
        const int64_t v = static_cast<int64_t>(ms * 1e6);
        if (v != reserve_margin_) NVN_LOG("resolución dinámica: margen para LSFG %.1f ms\n", ms);
        reserve_margin_ = v;
    }
    return reserve_margin_;
}

// El envío de f terminó (con fence_mutex_): copia sus timestamps de la GPU a los informes del juego.
void Executor::write_stamps(InFlight& f) {
    const uint32_t n = static_cast<uint32_t>(f.stamps.size());
    uint64_t ticks[kStampsPerSlot];
    if (vkGetQueryPoolResults(vk()->device, f.queries, 0, n, sizeof ticks, ticks, sizeof ticks[0], VK_QUERY_RESULT_64_BIT) !=
        VK_SUCCESS) {
        f.stamps.clear();
        return;
    }
    for (uint32_t k = 0; k < n; k++) {
        const InFlight::Stamp& s = f.stamps[k];
        const uint64_t t = ticks[s.query] & stamp_mask_;
        const uint64_t report[2] = {0, static_cast<uint64_t>(static_cast<double>(t) * stamp_scale_) + s.extra};
        memcpy(reinterpret_cast<void*>(s.address), report, sizeof report);
        if (!s.end) {
            frame_start_ticks_ = t;
        } else if (frame_start_ticks_ && t > frame_start_ticks_) {
            frame_gpu_ns_.store(static_cast<int64_t>((t - frame_start_ticks_) * vk()->props.limits.timestampPeriod),
                                std::memory_order_relaxed);
            frame_start_ticks_ = 0;
        }
    }
    f.stamps.clear();
}

void Executor::retire_texture(Texture& t) {
    std::lock_guard lock(g_mutex);
    // Fuera de run() no queda ninguno (finish los hace); por si acaso, ninguno con una imagen que ya no está.
    pending_clears_.erase(std::remove_if(pending_clears_.begin(), pending_clears_.end(),
                                         [&](const PendingClear& p) { return p.texture == &t; }),
                          pending_clears_.end());
    if (t.image || t.memory.memory) {
        Retired r;
        r.serial = t.serial;
        r.image = t.image;
        r.memory = t.memory;
        r.bytes = t.memory.memory ? t.gpu_bytes : 0;
        // El siguiente envío, no el último: el blit de presentar (window.cpp) va a la misma cola con su propio
        // fence, fuera de mi cuenta, y una señal de fence cubre todo lo enviado antes en la cola. (Presentar graba
        // el blit y lo envía sin soltar g_mutex: no puede quedar un blit de esta imagen por enviar.)
        r.submit = submitted_.load(std::memory_order_relaxed) + 1;
        retired_.push_back(r);
    }
    t.image = VK_NULL_HANDLE;
    t.memory = {};
    t.gpu_bytes = 0;
}

void Executor::retire_sampler(VkSampler s) {
    std::lock_guard lock(g_mutex);
    Retired r;
    r.sampler = s;
    r.submit = submitted_.load(std::memory_order_relaxed) + 1;  // como las texturas
    retired_.push_back(r);
}

// Destruye lo retirado cuyo último envío ya terminó. Las vistas de una textura van por su serial; los
// framebuffers, por los handles de las vistas que llevan en la clave (el pase de render, las vistas y el
// tamaño): un handle de vista destruido puede volver a salir en una vista nueva.
void Executor::reap_retired() {
    if (retired_.empty()) return;
    VkContext* c = vk();
    std::vector<uint64_t> serials;
    size_t keep = 0;
    const uint64_t completed = completed_.load(std::memory_order_acquire);
    for (const Retired& r : retired_) {
        if (r.submit > completed) {
            retired_[keep++] = r;
            continue;
        }
        if (r.image) vkDestroyImage(c->device, r.image, nullptr);
        if (r.memory.memory) {
            GpuAllocation memory = r.memory;
            gpu_free(memory);
            g_texture_gpu_bytes -= r.bytes;
            g_texture_gpu_images--;
        }
        if (r.sampler) vkDestroySampler(c->device, r.sampler, nullptr);
        if (r.serial) serials.push_back(r.serial);
    }
    retired_.resize(keep);
    if (serials.empty()) return;
    std::sort(serials.begin(), serials.end());
    std::unordered_set<VkImageView> dead;
    for (auto it = views_.begin(); it != views_.end();) {
        if (!std::binary_search(serials.begin(), serials.end(), it->first.value.texture)) {
            ++it;
            continue;
        }
        dead.insert(it->second);
        vkDestroyImageView(c->device, it->second, nullptr);
        it = views_.erase(it);
    }
    if (dead.empty()) return;
    for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
        const std::string& key = it->first;
        bool uses = false;
        for (size_t o = sizeof(VkRenderPass); !uses && o + sizeof(VkImageView) + 2 * sizeof(uint32_t) <= key.size();
             o += sizeof(VkImageView)) {
            VkImageView v;
            memcpy(&v, key.data() + o, sizeof v);
            uses = dead.count(v) != 0;
        }
        if (!uses) {
            ++it;
            continue;
        }
        vkDestroyFramebuffer(c->device, it->second, nullptr);
        it = framebuffers_.erase(it);
    }
}

// Sin g_mutex: con él, un hilo del juego que esperaba la GPU (aunque fuera un envío viejo) se quedaba detrás de la
// reproducción en curso, que lo tiene 10-30 ms. Se busca el fence con fence_mutex_ y se espera fuera de todo lock;
// mientras alguien lo espera, el ejecutor no lo reinicia (finish()). Si ya no está en vuelo, terminó.
bool Executor::wait_submit(uint64_t submit, uint64_t timeout_ns) {
    if (submit <= completed_.load(std::memory_order_acquire)) return true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(std::min<uint64_t>(timeout_ns, 1ull << 62));
    // Con <datos>/sin_espera_libre, como antes (para comparar): se mira el fence con g_mutex cada 200 µs.
    if (NVN_FLAG("sin_espera_libre")) {
        for (;;) {
            {
                std::lock_guard lock(g_mutex);
                std::lock_guard fence_lock(fence_mutex_);
                if (submit <= completed_.load(std::memory_order_acquire)) return true;
                InFlight* slot = nullptr;
                for (InFlight& f : inflight_)
                    if (f.pending && f.submit == submit) slot = &f;
                if (!slot) return true;  // ya no está en vuelo: terminó
                if (vkGetFenceStatus(vk()->device, slot->fence) == VK_SUCCESS) {
                    note_completed(submit);
                    return true;
                }
            }
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    InFlight* slot = nullptr;
    VkFence fence = VK_NULL_HANDLE;
    {
        std::lock_guard lock(fence_mutex_);
        if (submit <= completed_.load(std::memory_order_acquire)) return true;
        for (InFlight& f : inflight_)
            if (f.pending && f.submit == submit) slot = &f;
        if (!slot) return true;  // ya no está en vuelo: terminó
        slot->waiters++;
        fence = slot->fence;
    }
    const auto now = std::chrono::steady_clock::now();
    const uint64_t left = now >= deadline ? 0 : std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
    const VkResult r = left ? vkWaitForFences(vk()->device, 1, &fence, VK_TRUE, left) : vkGetFenceStatus(vk()->device, fence);
    {
        std::lock_guard lock(fence_mutex_);
        if (r == VK_SUCCESS) note_completed(submit);
        if (--slot->waiters == 0) fence_cv_.notify_all();
    }
    return r == VK_SUCCESS;
}

void Executor::run(const std::vector<const CommandList*>& lists) {
    std::lock_guard lock(g_mutex);
    if (!ready_ && !init()) return;
    auto t0 = std::chrono::steady_clock::now();
    submit_index_++;
    // Interruptores, una vez por envío del juego. sin_generaciones cambia la caché en el siguiente punto seguro
    // (entre draws o al terminar un envío de Vulkan).
    split_ = !NVN_FLAG("un_envio");
    gpu_stamps_ = !NVN_FLAG("ts_cpu");
    skip_compaction_gs_ = !NVN_FLAG("sin_salto_gs");
    prof_gpu_report();
    run_stamps_ = 0;
    // A la resolución dinámica se le dice que el frame cuesta lo suyo más LSFG más 2 ms de margen (con todo justo a 32 ms
    // los frames reales llegaban tarde y la guarda de LSFG saltaba). ts_sin_lsfg: sin sumar nada.
    {
        const int64_t lsfg = NVN_FLAG("ts_sin_lsfg") ? 0 : lsfg::reserved_ns();
        stamp_extra_ = lsfg ? static_cast<int64_t>((lsfg + reserve_margin_ns()) * 0.6144) : 0;
    }
    generations_ = !NVN_FLAG("sin_generaciones");
    buffer_cache_.set_generations(generations_);
    prof_timing_ = frame_ % kProfileEvery == 0 && NVN_FLAG("registro");  // los tiempos por draw, solo depurando
    if (prof_timing_ && prof_timed_frame_ != frame_) prof_timed_frame_ = frame_, prof_timed_frames_++;
    read_flags();
    begin();
    run_lists_ = &lists;  // la mirada adelante de los render passes (scan_pass) recorre lo que queda
    run_serial_++;
    for (cur_list_ = 0; cur_list_ < lists.size(); cur_list_++)
        for (cur_index_ = 0; cur_index_ < lists[cur_list_]->commands.size(); cur_index_++)
            apply(lists[cur_list_]->commands[cur_index_]);
    run_lists_ = nullptr;
    finish();
    prof_run_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

void Executor::read_flags() {
    opt_.clear_diferido = !NVN_FLAG("sin_clear_diferido");
    opt_.rt_igual = !NVN_FLAG("sin_rt_igual");
    opt_.barrera_perezosa = !NVN_FLAG("sin_barrera_perezosa");
    opt_.store_ops = !NVN_FLAG("sin_store_ops");
    opt_.area = !NVN_FLAG("sin_area");
    opt_.ubwc = !NVN_FLAG("sin_ubwc");  // solo para las imágenes que se creen a partir de ahora
}

// --- Estado ---

void Executor::apply(const Command& c) {
    cur_command_ = &c;
    if (frame_ == trace_frame_) trace_counts_[c.index()]++;
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, cmd::SetRenderTargets>) {
                // Los mismos targets con las mismas vistas: el pase abierto sigue (NieR vuelve a enlazar los suyos).
                const bool same = opt_.rt_igual && same_targets(v, st_.rt);
                if (same && in_render_pass_ && !st_.rt_dirty) prof_rt_kept_++;
                st_.rt = v;
                if (!same) st_.rt_dirty = true;
            } else if constexpr (std::is_same_v<T, cmd::ClearColor>) {
                clear_color(v);
            } else if constexpr (std::is_same_v<T, cmd::ClearDepthStencil>) {
                clear_depth_stencil(v);
            } else if constexpr (std::is_same_v<T, cmd::Viewport>) {
                if (v.index == 0) st_.viewport = {v.x, v.y, v.w, v.h, st_.viewport.minDepth, st_.viewport.maxDepth};
            } else if constexpr (std::is_same_v<T, cmd::Scissor>) {
                if (v.index == 0) {
                    st_.scissor = {{std::max(v.x, 0), std::max(v.y, 0)},
                                   {static_cast<uint32_t>(std::max(v.w, 0)), static_cast<uint32_t>(std::max(v.h, 0))}};
                    st_.scissor_set = true;
                }
            } else if constexpr (std::is_same_v<T, cmd::DepthRange>) {
                st_.viewport.minDepth = v.n;
                st_.viewport.maxDepth = v.f;
            } else if constexpr (std::is_same_v<T, cmd::BindProgram>) {
                // Bits de etapa: VERTEX 1, FRAGMENT 2, GEOMETRY 4, TESS_CONTROL 8, TESS_EVALUATION 16, COMPUTE 32.
                static const Stage kBits[] = {Stage::Vertex, Stage::Fragment, Stage::Geometry,
                                              Stage::TessControl, Stage::TessEval, Stage::Compute};
                for (int b = 0; b < 6; b++)
                    if (v.stages & (1 << b)) st_.stages[static_cast<int>(kBits[b])] = nullptr;
                if (v.stages & 32) st_.compute = nullptr;
                // El compute va aparte: st_.stages solo lleva las etapas del pipeline gráfico.
                if (v.program)
                    for (ProgramStage& ps : v.program->stages)
                        if (ensure_translated(ps)) {
                            if (ps.shader.stage == Stage::Compute) {
                                st_.compute = &ps;
                                st_.stage_generation[static_cast<int>(Stage::Compute)] = v.program->generation;
                                continue;
                            }
                            st_.stages[static_cast<int>(ps.shader.stage)] = &ps;
                            st_.stage_generation[static_cast<int>(ps.shader.stage)] = v.program->generation;
                        }
            } else if constexpr (std::is_same_v<T, cmd::BindVertexBuffers>) {
                for (size_t k = 0; k < v.ranges.size() && v.first + k < kMaxVertexBuffers; k++)
                    st_.vertex_buffers[v.first + k] = v.ranges[k];
            } else if constexpr (std::is_same_v<T, cmd::BindUniformBuffers>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                for (size_t k = 0; k < v.ranges.size() && v.first + k < kMaxUniforms; k++)
                    b.uniforms[v.first + k] = v.ranges[k];
            } else if constexpr (std::is_same_v<T, cmd::BindStorageBuffer>) {
                if (v.index < kMaxStorage) st_.bindings[static_cast<int>(from_nvn_stage(v.stage))].storage[v.index] = v.range;
            } else if constexpr (std::is_same_v<T, cmd::BindTexture>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                if (v.index < kMaxTextures) b.textures[v.index] = v.handle, b.combined[v.index] = v.combined;
            } else if constexpr (std::is_same_v<T, cmd::BindSampler>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                if (v.index < kMaxTextures) b.samplers[v.index] = v.handle;
            } else if constexpr (std::is_same_v<T, BlendState>) {
                st_.blend[v.target & 7] = v;
            } else if constexpr (std::is_same_v<T, ColorState>) {
                st_.color = v;
            } else if constexpr (std::is_same_v<T, ChannelMaskState>) {
                st_.masks = v;
            } else if constexpr (std::is_same_v<T, DepthStencilState>) {
                st_.depth_stencil = v;
            } else if constexpr (std::is_same_v<T, PolygonState>) {
                st_.polygon = v;
            } else if constexpr (std::is_same_v<T, MultisampleState>) {
                st_.multisample = v;
            } else if constexpr (std::is_same_v<T, cmd::BindVertexAttribs>) {
                st_.attribs = v.attribs;
            } else if constexpr (std::is_same_v<T, cmd::BindVertexStreams>) {
                st_.streams = v.streams;
            } else if constexpr (std::is_same_v<T, cmd::SetPools>) {
                if (v.textures) st_.texture_pool = v.textures;
                if (v.samplers) st_.sampler_pool = v.samplers;
            } else if constexpr (std::is_same_v<T, cmd::Stencil>) {
                uint32_t* target = v.what == cmd::Stencil::Ref ? st_.stencil_ref
                                   : v.what == cmd::Stencil::Mask ? st_.stencil_mask
                                                                   : st_.stencil_value_mask;
                if (v.faces & 1) target[0] = v.value;
                if (v.faces & 2) target[1] = v.value;
            } else if constexpr (std::is_same_v<T, cmd::Draw>) {
                const auto t0 = prof_now();
                draw(v);
                if (prof_timing_)
                    prof_draw_total_us_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            } else if constexpr (std::is_same_v<T, cmd::Dispatch>) {
                const auto t0 = prof_now();
                dispatch(v);
                if (prof_timing_)
                    prof_dispatch_us_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            } else if constexpr (std::is_same_v<T, cmd::BlendColor>) {
                memcpy(st_.blend_color, v.color, sizeof v.color);
            } else if constexpr (std::is_same_v<T, cmd::RenderEnable>) {
                if (frame_ == trace_frame_ && v.mode >= 0 && !watch_addr_) watch_addr_ = v.counters;
                if (frame_ == trace_frame_ && v.mode >= 0) {
                    const auto* c = reinterpret_cast<const uint64_t*>(v.counters);
                    NVN_LOG("F render enable condicional modo %d, contadores %#llx: %llx %llx\n", v.mode,
                            static_cast<unsigned long long>(v.counters), static_cast<unsigned long long>(c ? c[0] : 0),
                            static_cast<unsigned long long>(c ? c[1] : 0));
                } else if (frame_ == trace_frame_) {
                    NVN_LOG("F render enable %d\n", v.enable);
                }
                st_.render_enabled = v.enable;
            } else if constexpr (std::is_same_v<T, cmd::ClearBuffer>) {
                // Como CopyBuffer: memoria del juego, se rellena al ejecutar. Lo que hubiera en la GPU
                // (overlays de compute) para ese rango deja de valer: los lectores vuelven a la memoria.
                auto* p = reinterpret_cast<uint32_t*>(v.address);
                for (uint64_t k = 0; k < v.size / 4; k++) p[k] = v.value;
                for (auto it = overlays_.begin(); it != overlays_.end();) {
                    bool overlap = it->first < v.address + v.size && v.address < it->first + it->second.size;
                    it = overlap ? erase_overlay(it) : std::next(it);
                }
                copied_.clear();
            } else if constexpr (std::is_same_v<T, cmd::FenceSync>) {
                // Lo pone la cola de envíos al encolar (el ticket del envío): con el envío asíncrono, el juego
                // puede esperar el sync antes de que esto se reproduzca.
            } else if constexpr (std::is_same_v<T, cmd::ReportValue>) {
                // Ejecutamos en orden y de forma síncrona: el trabajo anterior ya está hecho.
                if (frame_ == trace_frame_ && v.address)
                    NVN_LOG("F ReportValue %u -> %#llx (había %08x)\n", v.value, static_cast<unsigned long long>(v.address),
                            *reinterpret_cast<const uint32_t*>(v.address));
                if (v.address) *reinterpret_cast<volatile uint32_t*>(v.address) = v.value;
            } else if constexpr (std::is_same_v<T, cmd::CopyBuffer>) {
                if (frame_ == trace_frame_)
                    NVN_LOG("F CopyBuffer %#llx -> %#llx (%llu bytes)\n", (unsigned long long)v.src, (unsigned long long)v.dst,
                            (unsigned long long)v.size);
                // Si el origen lo escribió la GPU (overlay), la copia se hace en la GPU: p. ej. NieR copia el número
                // de balas que cuenta un compute al instanceCount de un draw indirecto.
                if (copy_from_overlay(v.src, v.dst, v.size)) {
                    copied_.clear();
                    return;
                }
                // Los buffers son memoria del juego: la copia de la GPU es una copia de memoria aquí.
                memmove(reinterpret_cast<void*>(v.dst), reinterpret_cast<const void*>(v.src), v.size);
                for (auto it = overlays_.begin(); it != overlays_.end();) {  // como ClearBuffer
                    bool overlap = it->first < v.dst + v.size && v.dst < it->first + it->second.size;
                    it = overlap ? erase_overlay(it) : std::next(it);
                }
                copied_.clear();
            } else if constexpr (std::is_same_v<T, cmd::CopyTexture>) {
                copy_texture(v);
            } else if constexpr (std::is_same_v<T, cmd::CopyBufferToTexture>) {
                if (v.dst) v.dst->uploaded = false;  // se resube desde su memoria al usarla
            } else if constexpr (std::is_same_v<T, cmd::Barrier>) {
                nvn_barrier(v.bits);
            } else if constexpr (std::is_same_v<T, cmd::ReportCounter>) {
                // Informe de 16 bytes: {valor, timestamp}. Ejecutamos al momento: el trabajo previo ya está.
                // SAMPLES_PASSED (1): NieR lo pide por objeto en el pre-pase de profundidad y solo dibuja en color
                // lo que vio pasar muestras. Sin consultas de oclusión de verdad, todo cuenta como visible (con un
                // contador creciente, por si compara dos informes).
                uint64_t report[2] = {0, 0};
                if (v.type == kCounterSamplesPassed) report[0] = samples_passed_ += uint64_t{1} << 20;
                if (frame_ == trace_frame_) {
                    const auto* old = reinterpret_cast<const uint32_t*>(v.address);
                    NVN_LOG("F ReportCounter había %08x %08x %08x %08x | %08x %08x %08x %08x\n", old[0], old[1], old[2], old[3],
                            old[4], old[5], old[6], old[7]);
                }
                // TIMESTAMP (0): NieR escribe uno al empezar y otro al acabar cada frame, espera a su sync y con la resta
                // (en ticks de 614,4 MHz, x625/384 a ns en main+0x9bbb08) decide la resolución dinámica: más de 32 ms
                // la baja, menos de 30 la sube. Con el reloj de la CPU al reproducir veía 32 veces menos y nunca bajaba.
                // Ahora lo escribe la GPU, y al del final se le suma lo que cuesta el frame generado de LSFG, para que la
                // resolución deje sitio a los dos (ver write_stamps).
                InFlight& slot = inflight_[cur_];
                if (v.type == kCounterTimestamp && gpu_stamps_ && slot.queries && recording_ &&
                    slot.stamps.size() < kStampsPerSlot) {
                    const uint32_t q = static_cast<uint32_t>(slot.stamps.size());
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.queries, q);
                    const bool end = last_stamp_addr_ && v.address == last_stamp_addr_ + 16;
                    if (!end) last_stamp_addr_ = v.address;
                    if (end && frame_ == prof_frame_ && !prof_closed_ && !in_render_pass_) prof_gpu_mark("fin");
                    run_stamps_++;
                    slot.stamps.push_back({q, v.address, end ? stamp_extra_ : 0, end});
                } else {
                    asm volatile("mrs %0, cntvct_el0" : "=r"(report[1]));
                    memcpy(reinterpret_cast<void*>(v.address), report, sizeof report);
                }
                if (frame_ == trace_frame_)
                    NVN_LOG("F ReportCounter tipo %d -> %#llx\n", v.type, static_cast<unsigned long long>(v.address));
            }
        },
        c);
}

// --- Imágenes ---

bool Executor::ensure_image(Texture& t, VkCommandBuffer cmd) {
    if (t.image) return true;
    if (t.vk_failed || frame_ < t.retry_frame) return false;
    VkContext* c = vk();
    const FormatInfo* f = format_info(t.desc.format);
    if (!f || f->vk == VK_FORMAT_UNDEFINED) {
        t.vk_failed = true;
        return false;
    }
    VkFormat format = f->vk;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c->gpu, format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
        !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        NVN_LOG("formato %s sin soporte de imagen en la GPU\n", f->name);
        t.vk_failed = true;
        return false;
    }
    bool depth = is_depth_format(format);
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    int target = t.desc.target;
    bool cube = target == kTargetCube || target == kTargetCubeArray;
    ii.imageType = target == kTarget3D ? VK_IMAGE_TYPE_3D
                   : target == kTarget1D || target == kTarget1DArray ? VK_IMAGE_TYPE_1D
                                                                       : VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = {static_cast<uint32_t>(std::max(t.desc.width, 1)), static_cast<uint32_t>(std::max(t.desc.height, 1)),
                 target == kTarget3D ? static_cast<uint32_t>(std::max(t.desc.depth, 1)) : 1};
    t.layers = target == kTarget1DArray || target == kTarget2DArray || target == kTarget2DMultisampleArray || cube
                   ? std::max(t.desc.depth, cube ? 6 : 1)
                   : 1;
    ii.mipLevels = std::max(t.desc.levels, 1);
    ii.arrayLayers = t.layers;
    ii.samples = static_cast<VkSampleCountFlagBits>(std::max(t.desc.samples, 1));
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (depth && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        ii.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!depth && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
        ii.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((t.desc.flags & kTextureFlagImage) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
        ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (cube) ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // MUTABLE_FORMAT (vistas con otro formato) probablemente le quita al Adreno la compresión UBWC. En Vulkan cada
    // formato de profundidad/stencil es su propia clase de compatibilidad: sin MUTABLE no se pierde ninguna vista
    // válida. Las de color sí se reinterpretan (NVN lo permite y no sabemos con qué formatos), y las comprimidas no
    // llevan UBWC: esas se quedan como estaban.
    t.mutable_format = !(depth && opt_.ubwc);
    if (t.mutable_format) ii.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (VkResult r = vkCreateImage(c->device, &ii, nullptr, &t.image); r != VK_SUCCESS) {
        NVN_LOG("vkCreateImage %dx%d %s: %s\n", t.desc.width, t.desc.height, f->name, vk_result_name(r));
        t.vk_failed = true;
        return false;
    }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, t.image, &req);
    if (!gpu_alloc(req, find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), t.memory)) {
        NVN_LOG("sin memoria para una textura %dx%d (%u creadas; vivas %llu, %llu MB)\n", t.desc.width, t.desc.height,
                allocations_, (unsigned long long)g_texture_gpu_images.load(), (unsigned long long)(g_texture_gpu_bytes.load() >> 20));
        vkDestroyImage(c->device, t.image, nullptr);
        t.image = VK_NULL_HANDLE;
        t.retry_frame = frame_ + 30;  // sin memoria ahora: se reintenta en un segundo, no se da por perdida
        return false;
    }
    t.gpu_bytes = req.size;
    g_texture_gpu_bytes += req.size;
    g_texture_gpu_images++;
    if (++allocations_ % 500 == 0)
        NVN_LOG("%u imágenes creadas; vivas %llu, %llu MB\n", allocations_, (unsigned long long)g_texture_gpu_images.load(),
                (unsigned long long)(g_texture_gpu_bytes.load() >> 20));
    vkBindImageMemory(c->device, t.image, t.memory.memory, t.memory.offset);
    t.vk_format = format;
    t.aspect = depth ? (VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0))
                     : VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_S8_UINT) t.aspect = VK_IMAGE_ASPECT_STENCIL_BIT;

    if (cmd == cmd_ && in_render_pass_) {  // una barrera de imagen dentro de un render pass no es válida
        end_render_pass();
        st_.rt_dirty = true;
    }
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = {t.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    return true;
}

// Sube los texels que el juego dejó en su memoria (disposición Tegra) a la imagen. Fuera de render pass.
void Executor::upload(Texture& t) {
    if (t.gpu_written || !t.storage || !t.image) return;
    if (std::find(frame_uploads_.begin(), frame_uploads_.end(), &t) == frame_uploads_.end()) frame_uploads_.push_back(&t);
    if (!t.uploaded) prof_texture_bytes_ += t.layout.total_size();  // DEPURACIÓN: memoria de texturas subidas
    t.uploaded = true;
    t.content_hash = content_hash(t);
    t.checked_frame = frame_;
    const TextureLayout& L = t.layout;
    const uint32_t bpb = L.bytes_per_block;
    // Cada nivel puede caer en un trozo de staging distinto: una copia por trozo. (Antes se cortaba al
    // cambiar de trozo y las mips restantes se quedaban sin subir: basura en lo lejano, según lo lleno
    // que estuviera el staging.)
    std::vector<std::pair<VkBuffer, VkBufferImageCopy>> regions;
    for (uint32_t layer = 0; layer < std::min(L.layers, t.layers); layer++) {
        for (uint32_t level = 0; level < L.levels.size() && level < static_cast<uint32_t>(std::max(t.desc.levels, 1)); level++) {
            const LevelLayout& lv = L.levels[level];
            const size_t linear_size = size_t{lv.width_blocks} * bpb * lv.height_blocks * lv.depth;
            if (linear_size > kStagingChunk) {
                NVN_LOG("nivel de textura de %zu bytes: no cabe en el staging\n", linear_size);
                continue;
            }
            Staging::Alloc a = staging_->alloc(linear_size, 16);
            if (!a.data) return;
            const uint8_t* src = t.storage + layer * L.layer_size + lv.offset;
            if (L.linear) {
                for (uint32_t y = 0; y < lv.height_blocks; y++)
                    memcpy(a.data + size_t{y} * lv.width_blocks * bpb, src + size_t{y} * L.stride, lv.width_blocks * bpb);
            } else {
                deswizzle_level(lv, bpb, src, a.data);
            }
            VkBufferImageCopy r{};
            r.bufferOffset = a.offset;
            r.bufferRowLength = lv.width_blocks * L.block_w;
            r.bufferImageHeight = lv.height_blocks * L.block_h;
            r.imageSubresource = {t.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT, level, layer, 1};
            if (!r.imageSubresource.aspectMask) r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            r.imageExtent = {lv.width, lv.height, std::max(lv.depth, 1u)};
            regions.emplace_back(a.buffer, r);
        }
    }
    for (size_t k = 0; k < regions.size();) {
        size_t end = k;
        std::vector<VkBufferImageCopy> batch;
        while (end < regions.size() && regions[end].first == regions[k].first) batch.push_back(regions[end++].second);
        vkCmdCopyBufferToImage(cmd_, regions[k].first, t.image, VK_IMAGE_LAYOUT_GENERAL,
                               static_cast<uint32_t>(batch.size()), batch.data());
        k = end;
    }
}

VkImageView Executor::view(Texture& t, const TextureViewDesc& v, bool attachment) {
    // La última vista muestreada de cada textura, sin pasar por views_ (casi siempre es la misma). Va por el
    // serial: si la textura se reinicia o se retira, deja de coincidir.
    if (!attachment && t.view_memo && t.view_memo_serial == t.serial && memcmp(&t.view_memo_desc, &v, sizeof v) == 0)
        return t.view_memo;
    auto remember = [&](VkImageView out) {
        if (!attachment && out) t.view_memo = out, t.view_memo_serial = t.serial, t.view_memo_desc = v;
        return out;
    };
    BytesKey<ViewKeyData> key;
    key.value.texture = t.serial;
    memcpy(&key.value.view, &v, sizeof v);
    key.value.attachment = attachment;
    if (auto it = views_.find(key); it != views_.end()) return remember(it->second);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.image;
    int target = (v.flags & kViewTarget) ? v.target : t.desc.target;
    vi.viewType = target == kTarget3D ? VK_IMAGE_VIEW_TYPE_3D
                  : target == kTarget1D ? VK_IMAGE_VIEW_TYPE_1D
                  : target == kTarget1DArray ? VK_IMAGE_VIEW_TYPE_1D_ARRAY
                  : target == kTarget2DArray || target == kTarget2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                  : target == kTargetCube ? VK_IMAGE_VIEW_TYPE_CUBE
                  : target == kTargetCubeArray ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                                               : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = t.vk_format;
    if (v.flags & kViewFormat)
        if (const FormatInfo* f = format_info(v.format); f && f->vk != VK_FORMAT_UNDEFINED && f->vk != vi.format) {
            // Qué reinterpretaciones hace el juego (una línea por par de formatos): sin MUTABLE (profundidad con
            // ubwc) una vista con otro formato no es válida y se queda con el de la imagen.
            static std::set<std::pair<int, int>> seen;
            if (seen.insert({t.desc.format, v.format}).second)
                NVN_LOG("vista con otro formato: textura 0x%x vista 0x%x (%dx%d)%s\n", t.desc.format, v.format, t.desc.width,
                        t.desc.height, t.mutable_format ? "" : ", sin MUTABLE: con el formato de la textura");
            if (t.mutable_format) vi.format = f->vk;
        }
    uint32_t base_level = (v.flags & kViewLevels) ? v.base_level : 0;
    uint32_t levels = (v.flags & kViewLevels) && v.levels > 0 ? v.levels : std::max(t.desc.levels, 1) - base_level;
    uint32_t base_layer = (v.flags & kViewLayers) ? v.base_layer : 0;
    uint32_t layers = (v.flags & kViewLayers) && v.layers > 0 ? v.layers : t.layers - base_layer;
    if (attachment) levels = 1, layers = 1;
    if (vi.viewType == VK_IMAGE_VIEW_TYPE_2D || vi.viewType == VK_IMAGE_VIEW_TYPE_1D) layers = 1;
    if (vi.viewType == VK_IMAGE_VIEW_TYPE_CUBE) layers = 6;
    VkImageAspectFlags aspect = t.aspect;
    // Para muestrear profundidad+stencil hay que elegir uno: el modo de la vista (DEPTH 0, STENCIL 1).
    if (!attachment && (aspect & VK_IMAGE_ASPECT_DEPTH_BIT))
        aspect = (v.flags & kViewDepthStencil) && v.depth_stencil_mode == 1 ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                            : VK_IMAGE_ASPECT_DEPTH_BIT;
    vi.subresourceRange = {aspect, base_level, levels, base_layer, layers};
    // Swizzle de la vista (NVNtextureSwizzle: ZERO 0, ONE 1, R 2, G 3, B 4, A 5). Máscaras de un canal leídas
    // como RRR1 o 000R, normales de dos canales…: sin él esos materiales leían ceros y salían negros.
    if ((v.flags & kViewSwizzle) && !attachment) {
        auto map = [](uint8_t s) {
            switch (s) {
                case 0: return VK_COMPONENT_SWIZZLE_ZERO;
                case 1: return VK_COMPONENT_SWIZZLE_ONE;
                case 2: return VK_COMPONENT_SWIZZLE_R;
                case 3: return VK_COMPONENT_SWIZZLE_G;
                case 4: return VK_COMPONENT_SWIZZLE_B;
                case 5: return VK_COMPONENT_SWIZZLE_A;
                default: return VK_COMPONENT_SWIZZLE_IDENTITY;
            }
        };
        vi.components = {map(v.swizzle[0]), map(v.swizzle[1]), map(v.swizzle[2]), map(v.swizzle[3])};
    }
    VkImageView out = VK_NULL_HANDLE;
    if (VkResult r = vkCreateImageView(vk()->device, &vi, nullptr, &out); r != VK_SUCCESS)
        NVN_LOG("vkCreateImageView: %s\n", vk_result_name(r));
    views_[key] = out;
    return remember(out);
}

// Filtros de NVN: min NEAREST 0, LINEAR 1, NEAREST_MIPMAP_NEAREST 2, LINEAR_MIPMAP_NEAREST 3,
// NEAREST_MIPMAP_LINEAR 4, LINEAR_MIPMAP_LINEAR 5; mag NEAREST 0, LINEAR 1. Wrap: CLAMP 0, REPEAT 1,
// MIRROR_CLAMP 2, MIRROR_CLAMP_TO_EDGE 3, MIRROR_CLAMP_TO_BORDER 4, CLAMP_TO_BORDER 5, MIRRORED_REPEAT 6,
// CLAMP_TO_EDGE 7.
VkSampler Executor::sampler(Sampler* s) {
    if (!s) return default_sampler_;
    if (s->vk) return s->vk;
    const SamplerDesc& d = s->desc;
    auto wrap = [](int w) {
        switch (w) {
            case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case 2:
            case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
            case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
    };
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d.mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.minFilter = (d.min_filter == 1 || d.min_filter == 3 || d.min_filter == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = (d.min_filter == 4 || d.min_filter == 5) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = wrap(d.wrap[0]);
    si.addressModeV = wrap(d.wrap[1]);
    si.addressModeW = wrap(d.wrap[2]);
    si.mipLodBias = d.lod_bias;
    // Anisotrópico: el que pide el juego (antes se ignoraba y el suelo y las paredes se emborronaban en ángulo), y como
    // mínimo 4x en las texturas con mipmaps y filtro lineal. sin_af al crear el sampler: como antes.
    {
        const bool mip_linear = si.minFilter == VK_FILTER_LINEAR && d.min_filter >= 2;
        float af = d.max_anisotropy;
        if (mip_linear) af = std::max(af, 4.0f);
        af = std::min(af, vk()->props.limits.maxSamplerAnisotropy);
        si.anisotropyEnable = af > 1.0f && !NVN_FLAG("sin_af") ? VK_TRUE : VK_FALSE;
        si.maxAnisotropy = si.anisotropyEnable ? af : 1.0f;
    }
    si.compareEnable = d.compare_mode != 0;
    si.compareOp = vk_compare_op(d.compare_func);
    si.minLod = d.min_filter >= 2 ? std::max(d.lod_min, 0.0f) : 0.0f;
    si.maxLod = d.min_filter >= 2 ? std::min(d.lod_max, 16.0f) : 0.25f;
    si.borderColor = d.border[3] > 0.5f ? (d.border[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                                                : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK)
                                        : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (vkCreateSampler(vk()->device, &si, nullptr, &s->vk) != VK_SUCCESS) return default_sampler_;
    return s->vk;
}

// --- Render passes ---
// En el Adreno cada pase pasa sus adjuntos por la memoria del tile (GMEM): carga al empezar (LOAD), guarda al acabar
// (STORE), en cada bin. Lo que sigue intenta no mover lo que no hace falta: borrados como loadOp CLEAR, nada de
// guardar lo que el pase no escribe, área de render ajustada a lo que se dibuja y menos cortes de pase.

VkRenderPass Executor::render_pass(const VkFormat* colors, uint32_t count, VkFormat depth, VkSampleCountFlagBits samples,
                                   const PassOps& ops) {
    std::string key = std::string(reinterpret_cast<const char*>(colors), count * sizeof(VkFormat)) + key_of(depth) +
                      key_of(samples) + key_of(ops);
    if (auto it = render_passes_.find(key); it != render_passes_.end()) return it->second;
    std::vector<VkAttachmentDescription> atts;
    std::vector<VkAttachmentReference> color_refs;
    for (uint32_t k = 0; k < count; k++) {
        if (colors[k] == VK_FORMAT_UNDEFINED) {
            color_refs.push_back({VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL});
            continue;
        }
        color_refs.push_back({static_cast<uint32_t>(atts.size()), VK_IMAGE_LAYOUT_GENERAL});
        atts.push_back({0, colors[k], samples, ops.color_load[k], ops.color_store[k], VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
    }
    VkAttachmentReference depth_ref{VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL};
    if (depth != VK_FORMAT_UNDEFINED) {
        depth_ref.attachment = static_cast<uint32_t>(atts.size());
        atts.push_back({0, depth, samples, ops.depth_load, ops.depth_store, ops.stencil_load, ops.depth_store,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
    }
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = count;
    sub.pColorAttachments = color_refs.data();
    sub.pDepthStencilAttachment = depth != VK_FORMAT_UNDEFINED ? &depth_ref : nullptr;
    // Dependencias completas con lo de antes y lo de después: las imágenes no cambian de layout.
    VkSubpassDependency deps[2] = {
        {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
         VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0},
        {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0}};
    VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ri.attachmentCount = static_cast<uint32_t>(atts.size());
    ri.pAttachments = atts.data();
    ri.subpassCount = 1;
    ri.pSubpasses = &sub;
    ri.dependencyCount = 2;
    ri.pDependencies = deps;
    VkRenderPass rp = VK_NULL_HANDLE;
    if (VkResult r = vkCreateRenderPass(vk()->device, &ri, nullptr, &rp); r != VK_SUCCESS)
        NVN_LOG("vkCreateRenderPass: %s\n", vk_result_name(r));
    render_passes_[key] = rp;
    return rp;
}

// Mirada adelante (ver PassScan): recorre los comandos desde el que pide el pase con una copia del estado que importa.
const Executor::PassScan& Executor::scan_pass() {
    // Dónde está el comando en curso (lista, índice): lo lleva run(), no la dirección (una lista puede repetirse).
    const std::vector<const CommandList*>& lists = *run_lists_;
    size_t l = cur_list_, k = cur_index_;
    const bool inside = l < scan_.end_list || (l == scan_.end_list && k < scan_.end_cmd);  // antes de donde acabó
    if (scan_.run == run_serial_ && inside && same_targets(scan_.rt, st_.rt))
        return scan_;
    PassScan s;
    s.run = run_serial_;
    s.rt = st_.rt;
    if (l == lists.size()) {  // no está en las listas (no debería pasar): todo escrito, área entera
        s.colors_written = ~0u;
        s.depth_written = s.cleared = true;
        scan_ = s;
        scan_.run = 0;
        return scan_;
    }
    s.x0 = s.y0 = INT64_MAX;
    s.x1 = s.y1 = INT64_MIN;
    uint32_t masks = st_.masks.masks;
    DepthStencilState ds = st_.depth_stencil;
    uint8_t polygon_mode = st_.polygon.polygon_mode;
    VkViewport vp = st_.viewport;
    VkRect2D sc = st_.scissor;
    bool sc_set = st_.scissor_set;
    // Etapas que pueden convertir triángulos en puntos o líneas (geometry 4, teselación 8 y 16: bits de NVN).
    uint32_t geo = (st_.stages[static_cast<int>(Stage::Geometry)] ? 4u : 0u) |
                   (st_.stages[static_cast<int>(Stage::TessControl)] ? 8u : 0u) |
                   (st_.stages[static_cast<int>(Stage::TessEval)] ? 16u : 0u);
    while (l < lists.size()) {
        const std::vector<Command>& cmds = lists[l]->commands;
        if (k >= cmds.size()) {
            l++, k = 0;
            continue;
        }
        const Command& c = cmds[k];
        bool stop = false;
        switch (c.index()) {
            case kCommandIndex<cmd::SetRenderTargets>:
                stop = !same_targets(*std::get_if<cmd::SetRenderTargets>(&c), s.rt);
                break;
            case kCommandIndex<cmd::ClearColor>: {
                const cmd::ClearColor& v = *std::get_if<cmd::ClearColor>(&c);
                if (v.index >= 0 && v.index < s.rt.count && s.rt.colors[v.index]) s.colors_written |= 1u << v.index, s.cleared = true;
                break;
            }
            case kCommandIndex<cmd::ClearDepthStencil>:
                if (s.rt.depth) s.depth_written = s.cleared = true;
                break;
            case kCommandIndex<cmd::Viewport>: {
                const cmd::Viewport& v = *std::get_if<cmd::Viewport>(&c);
                if (v.index == 0) vp.x = v.x, vp.y = v.y, vp.width = v.w, vp.height = v.h;
                break;
            }
            case kCommandIndex<cmd::Scissor>: {
                const cmd::Scissor& v = *std::get_if<cmd::Scissor>(&c);
                if (v.index == 0) {  // como apply()
                    sc = {{std::max(v.x, 0), std::max(v.y, 0)},
                          {static_cast<uint32_t>(std::max(v.w, 0)), static_cast<uint32_t>(std::max(v.h, 0))}};
                    sc_set = true;
                }
                break;
            }
            case kCommandIndex<ChannelMaskState>: masks = std::get_if<ChannelMaskState>(&c)->masks; break;
            case kCommandIndex<DepthStencilState>: ds = *std::get_if<DepthStencilState>(&c); break;
            case kCommandIndex<PolygonState>: polygon_mode = std::get_if<PolygonState>(&c)->polygon_mode; break;
            case kCommandIndex<cmd::BindProgram>: {
                const cmd::BindProgram& v = *std::get_if<cmd::BindProgram>(&c);
                geo &= ~(static_cast<uint32_t>(v.stages) & 28u);
                if (v.program)
                    for (const ProgramStage& ps : v.program->stages) {
                        if (!ps.tried) geo |= 28u;  // sin traducir todavía: no se sabe qué etapa es
                        else if (ps.module && ps.shader.stage == Stage::Geometry) geo |= 4u;
                        else if (ps.module && ps.shader.stage == Stage::TessControl) geo |= 8u;
                        else if (ps.module && ps.shader.stage == Stage::TessEval) geo |= 16u;
                    }
                break;
            }
            case kCommandIndex<cmd::Draw>: {
                const int p = std::get_if<cmd::Draw>(&c)->primitive;
                for (int t = 0; t < s.rt.count; t++)
                    if ((masks >> (4 * t)) & 0xF) s.colors_written |= 1u << t;
                if (s.rt.depth && (ds.depth_write || ds.stencil_test)) s.depth_written = true;  // de más: sin mirar ops
                // Dónde puede dibujar: dentro del scissor (sin él, en todo el framebuffer); los triángulos rellenos,
                // además, dentro del viewport (Vulkan los recorta a él). Los puntos (tamaño del shader) y las líneas
                // pueden salirse del viewport, y un geometry shader o la teselación pueden convertir en eso lo que sea.
                int64_t x0 = 0, y0 = 0, x1 = INT32_MAX, y1 = INT32_MAX;
                if (sc_set)
                    x0 = sc.offset.x, y0 = sc.offset.y, x1 = x0 + sc.extent.width, y1 = y0 + sc.extent.height;
                const bool points_or_lines = p == 0 || p == 1 || p == 3 || p == 10 || p == 11 || p == 14;
                if (!points_or_lines && !geo && polygon_mode >= 2) {  // NVN: POINT 0, LINE 1, FILL 2
                    x0 = std::max<int64_t>(x0, static_cast<int64_t>(std::floor(std::min(vp.x, vp.x + vp.width))));
                    x1 = std::min<int64_t>(x1, static_cast<int64_t>(std::ceil(std::max(vp.x, vp.x + vp.width))));
                    y0 = std::max<int64_t>(y0, static_cast<int64_t>(std::floor(std::min(vp.y, vp.y + vp.height))));
                    y1 = std::min<int64_t>(y1, static_cast<int64_t>(std::ceil(std::max(vp.y, vp.y + vp.height))));
                }
                if (x1 > x0 && y1 > y0) {
                    s.drawn = true;
                    s.x0 = std::min(s.x0, x0), s.y0 = std::min(s.y0, y0), s.x1 = std::max(s.x1, x1), s.y1 = std::max(s.y1, y1);
                }
                break;
            }
            default: break;
        }
        if (stop) break;
        k++;
    }
    s.end_list = l;
    s.end_cmd = k;
    scan_ = s;
    return scan_;
}

bool Executor::ensure_render_pass(bool full_area) {
    if (in_render_pass_ && !st_.rt_dirty) return true;
    end_render_pass();
    st_.rt_dirty = false;
    const cmd::SetRenderTargets& rt = st_.rt;
    VkFormat formats[8] = {};
    std::vector<VkImageView> views;
    // Adjunto i (en el orden de views, que es el del render pass): textura, vista y render target (-1: profundidad).
    Texture* targets[9] = {};
    const TextureViewDesc* target_views[9] = {};
    int slots[9] = {};
    int slot = 0;
    uint32_t width = UINT32_MAX, height = UINT32_MAX;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    auto add = [&](Texture* t, const TextureViewDesc& v) -> VkFormat {
        if (!t || !ensure_image(*t, cmd_)) return VK_FORMAT_UNDEFINED;
        t->gpu_written = true;
        VkImageView iv = view(*t, v, true);
        if (!iv) return VK_FORMAT_UNDEFINED;
        targets[views.size()] = t, target_views[views.size()] = &v, slots[views.size()] = slot;
        views.push_back(iv);
        uint32_t level = (v.flags & kViewLevels) ? v.base_level : 0;
        width = std::min<uint32_t>(width, std::max(t->desc.width >> level, 1));
        height = std::min<uint32_t>(height, std::max(t->desc.height >> level, 1));
        samples = static_cast<VkSampleCountFlagBits>(std::max(t->desc.samples, 1));
        return t->vk_format;
    };
    for (int k = 0; k < rt.count; k++) {
        slot = k;
        formats[k] = add(rt.colors[k], rt.color_views[k]);
        if (rt.colors[k] && std::find(frame_targets_.begin(), frame_targets_.end(), rt.colors[k]) == frame_targets_.end())
            frame_targets_.push_back(rt.colors[k]);
    }
    slot = -1;
    VkFormat depth = add(rt.depth, rt.depth_view);
    if (rt.depth && std::find(frame_targets_.begin(), frame_targets_.end(), rt.depth) == frame_targets_.end())
        frame_targets_.push_back(rt.depth);
    if (views.empty()) return false;
    // El de LOAD/STORE: con él se crean los framebuffers y los pipelines (las cargas y guardados no cuentan para la
    // compatibilidad, así que valen para el que se empiece de verdad).
    VkRenderPass rp = render_pass(formats, static_cast<uint32_t>(rt.count), depth, samples);
    if (!rp) return false;
    std::string fb_key = key_of(rp) + std::string(reinterpret_cast<const char*>(views.data()), views.size() * sizeof(VkImageView)) +
                         key_of(width) + key_of(height);
    VkFramebuffer fb;
    if (auto it = framebuffers_.find(fb_key); it != framebuffers_.end()) {
        fb = it->second;
    } else {
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = rp;
        fi.attachmentCount = static_cast<uint32_t>(views.size());
        fi.pAttachments = views.data();
        fi.width = width;
        fi.height = height;
        fi.layers = 1;
        if (VkResult r = vkCreateFramebuffer(vk()->device, &fi, nullptr, &fb); r != VK_SUCCESS) {
            NVN_LOG("vkCreateFramebuffer: %s\n", vk_result_name(r));
            return false;
        }
        framebuffers_[fb_key] = fb;
    }
    const uint32_t n = static_cast<uint32_t>(views.size());
    // Cargas: el borrado diferido de un adjunto se hace con loadOp CLEAR si es de esa misma vista y el pase cubre el
    // adjunto entero (el área será todo el framebuffer); si no, ahora, antes de empezar.
    PassOps ops = default_pass_ops();
    VkClearValue clears[9] = {};
    VkImageAspectFlags clear_aspects[9] = {};
    bool load_clear = false;
    for (uint32_t i = 0; i < n && !pending_clears_.empty(); i++) {
        Texture* t = targets[i];
        auto it = std::find_if(pending_clears_.begin(), pending_clears_.end(), [&](const PendingClear& p) { return p.texture == t; });
        if (it == pending_clears_.end()) continue;
        const TextureViewDesc& v = *target_views[i];
        const int level = (v.flags & kViewLevels) ? v.base_level : 0;
        bool usable = memcmp(&it->view, &v, sizeof v) == 0 && static_cast<uint32_t>(std::max(t->desc.width >> level, 1)) == width &&
                      static_cast<uint32_t>(std::max(t->desc.height >> level, 1)) == height;
        for (uint32_t j = 0; j < n; j++) usable = usable && (j == i || targets[j] != t);  // dos veces en el pase: aparte
        if (!usable) {
            flush_clears(t);
            continue;
        }
        if (slots[i] >= 0) {
            ops.color_load[slots[i]] = VK_ATTACHMENT_LOAD_OP_CLEAR;
        } else {
            if (it->aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ops.depth_load = VK_ATTACHMENT_LOAD_OP_CLEAR;
            if (it->aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ops.stencil_load = VK_ATTACHMENT_LOAD_OP_CLEAR;
        }
        clears[i] = it->value;
        clear_aspects[i] = it->aspects;
        pending_clears_.erase(it);
        load_clear = true;
        prof_clear_loads_++;
    }
    // Guardados y área, de la mirada adelante. Lo que ningún draw ni borrado del pase puede escribir no se guarda: con
    // STORE_OP_NONE la memoria se queda como estaba (nunca DONT_CARE: se puede leer después). El área es lo que pueden
    // tocar sus draws; con borrados, todo (loadOp CLEAR y vkCmdClearAttachments no pasan del área).
    const PassScan* scan = run_lists_ && (opt_.store_ops || opt_.area) ? &scan_pass() : nullptr;
    if (scan && opt_.store_ops && vk()->store_op_none)
        for (uint32_t i = 0; i < n; i++) {
            if (slots[i] >= 0 && !((scan->colors_written >> slots[i]) & 1) && ops.color_load[slots[i]] != VK_ATTACHMENT_LOAD_OP_CLEAR) {
                ops.color_store[slots[i]] = VK_ATTACHMENT_STORE_OP_NONE_QCOM;
                prof_store_none_++;
            } else if (slots[i] < 0 && !scan->depth_written && ops.depth_load != VK_ATTACHMENT_LOAD_OP_CLEAR &&
                       ops.stencil_load != VK_ATTACHMENT_LOAD_OP_CLEAR) {
                ops.depth_store = VK_ATTACHMENT_STORE_OP_NONE_QCOM;
                prof_store_none_++;
            }
        }
    const VkRect2D full{{0, 0}, {width, height}};
    VkRect2D area = full;
    if (scan && opt_.area && !full_area && !load_clear && !scan->cleared && scan->drawn) {
        const int64_t x0 = std::clamp<int64_t>(scan->x0, 0, width), x1 = std::clamp<int64_t>(scan->x1, 0, width);
        const int64_t y0 = std::clamp<int64_t>(scan->y0, 0, height), y1 = std::clamp<int64_t>(scan->y1, 0, height);
        if (x1 > x0 && y1 > y0)
            area = {{static_cast<int32_t>(x0), static_cast<int32_t>(y0)}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
    }
    const PassOps defaults = default_pass_ops();
    VkRenderPass begun = memcmp(&ops, &defaults, sizeof ops) == 0 ? rp : render_pass(formats, static_cast<uint32_t>(rt.count), depth, samples, ops);
    const bool fallback = !begun;  // sin esa variante: el de siempre, y los borrados dentro del pase
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = fallback ? rp : begun;
    bi.framebuffer = fb;
    bi.renderArea = area;
    bi.clearValueCount = load_clear && !fallback ? n : 0;
    bi.pClearValues = load_clear && !fallback ? clears : nullptr;
    prof_gpu_mark("pase");
    vkCmdBeginRenderPass(cmd_, &bi, VK_SUBPASS_CONTENTS_INLINE);
    if (fallback && load_clear)
        for (uint32_t i = 0; i < n; i++)
            if (clear_aspects[i]) {
                VkClearAttachment a{clear_aspects[i], slots[i] >= 0 ? static_cast<uint32_t>(slots[i]) : 0u, clears[i]};
                VkClearRect r{full, 0, 1};
                vkCmdClearAttachments(cmd_, 1, &a, 1, &r);
            }
    prof_render_passes_++;
    in_render_pass_ = true;
    rp_ = rp;
    rp_extent_ = {width, height};
    rp_area_ = area;
    rp_area_full_ = area.offset.x == 0 && area.offset.y == 0 && area.extent.width == width && area.extent.height == height;
    if (!rp_area_full_) prof_area_cut_++;
    rp_color_count_ = static_cast<uint32_t>(rt.count);
    rp_samples_ = samples;
    rp_target_count_ = n;
    std::copy(targets, targets + n, rp_targets_);
    storage_since_barrier_ = false;  // lo de antes del pase ya va ordenado por su dependencia externa
    if (frame_ == trace_frame_) {  // DEPURACIÓN: cargas (L/C) y guardados (s/n) de cada adjunto
        std::string a;
        for (uint32_t i = 0; i < n; i++) {
            const bool clear = slots[i] >= 0 ? ops.color_load[slots[i]] == VK_ATTACHMENT_LOAD_OP_CLEAR
                                             : ops.depth_load == VK_ATTACHMENT_LOAD_OP_CLEAR || ops.stencil_load == VK_ATTACHMENT_LOAD_OP_CLEAR;
            const bool stored = (slots[i] >= 0 ? ops.color_store[slots[i]] : ops.depth_store) == VK_ATTACHMENT_STORE_OP_STORE;
            a += std::string(" ") + (slots[i] >= 0 ? std::to_string(slots[i]) : std::string("d")) + ":" + (clear ? "C" : "L") +
                 (stored ? "s" : "n");
        }
        NVN_LOG("F pase %ux%u área %d,%d %ux%u adjuntos%s\n", width, height, area.offset.x, area.offset.y, area.extent.width,
                area.extent.height, a.c_str());
    }
    return true;
}

void Executor::end_render_pass() {
    if (!in_render_pass_) return;
    vkCmdEndRenderPass(cmd_);
    in_render_pass_ = false;
    if (pending_barrier_) {  // la barrera de NVN aplazada (nvn_barrier)
        pending_barrier_ = false;
        full_barrier();
    }
}

void Executor::full_barrier() {
    storage_since_barrier_ = false;
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}

void Executor::barrier() {
    pending_barrier_ = false;  // esta la cubre
    end_render_pass();
    st_.rt_dirty = true;
    full_barrier();
}

// Barrera de NVN. Todo lo de antes del pase abierto ya va ordenado con él por su dependencia externa: dentro del pase
// solo importan sus propios draws. Si en el pase ya se escribieron storage buffers o imágenes desde la última barrera,
// lo que venga puede leerlos: se corta como siempre. Si no, se aplaza al final del pase (end_render_pass); draw() lo
// corta antes si un draw muestrea un adjunto del pase o escribe storage.
void Executor::nvn_barrier(int bits) {
    const bool lazy = opt_.barrera_perezosa && in_render_pass_ && !st_.rt_dirty && !storage_since_barrier_;
    if (frame_ == trace_frame_) NVN_LOG("F barrera 0x%x%s\n", bits, lazy ? " aplazada" : "");
    if (!lazy) {
        barrier();
        return;
    }
    pending_barrier_ = true;
    prof_barriers_deferred_++;
}

// --- Borrados y copias ---

// Lo que haría ensure_render_pass con los targets, sin empezar el pase: imágenes creadas y marcadas como escritas por
// la GPU (ya no se suben de la memoria del juego) y apuntadas para los volcados.
void Executor::touch_targets() {
    auto touch = [&](Texture* t) {
        if (!t) return;
        if (ensure_image(*t, cmd_)) t->gpu_written = true;
        if (std::find(frame_targets_.begin(), frame_targets_.end(), t) == frame_targets_.end()) frame_targets_.push_back(t);
    };
    for (int k = 0; k < st_.rt.count; k++) touch(st_.rt.colors[k]);
    touch(st_.rt.depth);
}

// Tamaño del framebuffer de los targets enlazados (como en ensure_render_pass: el menor de todos).
VkExtent2D Executor::target_extent() const {
    uint32_t width = UINT32_MAX, height = UINT32_MAX;
    auto add = [&](const Texture* t, const TextureViewDesc& v) {
        if (!t || !t->image) return;
        uint32_t level = (v.flags & kViewLevels) ? v.base_level : 0;
        width = std::min<uint32_t>(width, std::max(t->desc.width >> level, 1));
        height = std::min<uint32_t>(height, std::max(t->desc.height >> level, 1));
    };
    for (int k = 0; k < st_.rt.count; k++) add(st_.rt.colors[k], st_.rt.color_views[k]);
    add(st_.rt.depth, st_.rt.depth_view);
    return {width, height};
}

// Un borrado de todo el adjunto antes de que empiece el pase de estos targets: se apunta para el siguiente pase que lo
// use (loadOp CLEAR) en vez de abrir uno solo para borrar. false si no se puede (se borra como siempre): textura sin
// imagen, multisample, vista con otro formato o de otro tipo, el adjunto no es todo el framebuffer, scissor parcial, o
// ya hay otro pendiente de esa textura con otra vista.
bool Executor::defer_clear(Texture* t, const TextureViewDesc& v, VkImageAspectFlags aspects, const VkClearValue& value) {
    end_render_pass();  // el de antes, de otros targets: se iba a cerrar en el siguiente draw
    touch_targets();
    if (!t->image || t->desc.samples > 1) return false;
    const int target = (v.flags & kViewTarget) ? v.target : t->desc.target;
    if (target != kTarget2D && target != kTarget2DArray && target != kTargetRectangle) return false;
    if (v.flags & kViewFormat)
        if (const FormatInfo* f = format_info(v.format); !f || f->vk != t->vk_format) return false;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk()->gpu, t->vk_format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) return false;  // para vkCmdClear*Image
    const uint32_t level = (v.flags & kViewLevels) ? v.base_level : 0;
    const uint32_t layer = (v.flags & kViewLayers) ? v.base_layer : 0;
    if (level >= static_cast<uint32_t>(std::max(t->desc.levels, 1)) || layer >= t->layers) return false;
    const VkExtent2D fb = target_extent();
    if (static_cast<uint32_t>(std::max(t->desc.width >> level, 1)) != fb.width ||
        static_cast<uint32_t>(std::max(t->desc.height >> level, 1)) != fb.height)
        return false;
    if (st_.scissor_set && (st_.scissor.offset.x > 0 || st_.scissor.offset.y > 0 ||
                            int64_t{st_.scissor.offset.x} + st_.scissor.extent.width < fb.width ||
                            int64_t{st_.scissor.offset.y} + st_.scissor.extent.height < fb.height))
        return false;
    for (PendingClear& p : pending_clears_) {
        if (p.texture != t) continue;
        if (memcmp(&p.view, &v, sizeof v) != 0) return false;
        // Otro sobre lo mismo: manda el último (de profundidad+stencil, por aspecto).
        if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) p.value.color = value.color;
        if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) p.value.depthStencil.depth = value.depthStencil.depth;
        if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) p.value.depthStencil.stencil = value.depthStencil.stencil;
        p.aspects |= aspects;
        return true;
    }
    pending_clears_.push_back({t, v, aspects, value});
    return true;
}

// Hace ya los borrados diferidos de esa textura (de todas si es nula), fuera del render pass: la van a leer sin
// empezar un pase con ella (muestrearla, copiarla, presentarla, volcarla) o se acaba el envío.
void Executor::flush_clears(const Texture* only) {
    if (pending_clears_.empty()) return;
    bool any = false;
    size_t keep = 0;
    for (size_t i = 0; i < pending_clears_.size(); i++) {
        const PendingClear p = pending_clears_[i];
        if ((only && p.texture != only) || !p.texture->image) {
            if (only && p.texture != only) pending_clears_[keep++] = p;
            continue;
        }
        if (!any) {
            if (in_render_pass_) {  // el pase abierto (de otros targets) no puede seguir tras esto
                end_render_pass();
                st_.rt_dirty = true;
            }
            // Lo anterior que escribe la imagen (una subida no lleva barrera detrás), antes del borrado.
            VkMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr,
                                 0, nullptr);
            any = true;
        }
        const TextureViewDesc& v = p.view;
        const VkImageSubresourceRange range{p.aspects, (v.flags & kViewLevels) ? static_cast<uint32_t>(v.base_level) : 0u, 1,
                                            (v.flags & kViewLayers) ? static_cast<uint32_t>(v.base_layer) : 0u, 1};
        if (p.aspects & VK_IMAGE_ASPECT_COLOR_BIT)
            vkCmdClearColorImage(cmd_, p.texture->image, VK_IMAGE_LAYOUT_GENERAL, &p.value.color, 1, &range);
        else
            vkCmdClearDepthStencilImage(cmd_, p.texture->image, VK_IMAGE_LAYOUT_GENERAL, &p.value.depthStencil, 1, &range);
        prof_clear_flushes_++;
    }
    pending_clears_.resize(keep);
    if (!any) return;
    VkMemoryBarrier b{};  // el borrado, antes de lo que venga (render passes, copias, compute, presentar)
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0,
                         nullptr);
}

void Executor::clear_color(const cmd::ClearColor& c) {
    if (frame_ == trace_frame_)
        NVN_LOG("F clear rt%d (%p) %.3f %.3f %.3f %.3f mask 0x%x\n", c.index,
                c.index < st_.rt.count ? static_cast<void*>(st_.rt.colors[c.index]) : nullptr, c.color[0], c.color[1],
                c.color[2], c.color[3], c.mask);
    if (c.index >= st_.rt.count || !st_.rt.colors[c.index]) return;
    if (Texture* t = st_.rt.colors[c.index]) {
        memcpy(t->clear_color, c.color, sizeof c.color);
        t->cleared = true;
    }
    VkClearValue value{};
    memcpy(value.color.float32, c.color, sizeof c.color);
    // Todo el target (máscara completa) antes de que empiece su pase: diferido (ver PendingClear).
    if (opt_.clear_diferido && (!in_render_pass_ || st_.rt_dirty) && (c.mask & 0xF) == 0xF &&
        defer_clear(st_.rt.colors[c.index], st_.rt.color_views[c.index], VK_IMAGE_ASPECT_COLOR_BIT, value))
        return;
    if (!ensure_render_pass()) return;
    if (!rp_area_full_) {  // la mirada adelante no lo vio (no debería pasar): un pase con todo el área
        end_render_pass();
        st_.rt_dirty = true;
        if (!ensure_render_pass(true)) return;
    }
    VkClearAttachment a{VK_IMAGE_ASPECT_COLOR_BIT, static_cast<uint32_t>(c.index), value};
    VkClearRect r{{{0, 0}, rp_extent_}, 0, 1};
    vkCmdClearAttachments(cmd_, 1, &a, 1, &r);
}

void Executor::clear_depth_stencil(const cmd::ClearDepthStencil& c) {
    Texture* t = st_.rt.depth;
    if (!t) return;
    const bool deferrable = opt_.clear_diferido && (!in_render_pass_ || st_.rt_dirty);
    if (deferrable) {
        end_render_pass();  // el de antes, de otros targets
        touch_targets();    // para t->aspect (y lo que haría el pase)
        if (!t->image) return;
    } else if (!ensure_render_pass() || !t->image) {
        return;
    }
    VkImageAspectFlags aspect = 0;
    if (c.depth_mask) aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (c.stencil_mask && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!aspect) return;
    VkClearValue value{};
    value.depthStencil = {c.depth, static_cast<uint32_t>(c.stencil)};
    if (deferrable && (aspect & t->aspect) && defer_clear(t, st_.rt.depth_view, aspect & t->aspect, value)) return;
    if (!ensure_render_pass() || !t->image) return;
    if (!rp_area_full_) {  // como en clear_color
        end_render_pass();
        st_.rt_dirty = true;
        if (!ensure_render_pass(true)) return;
    }
    VkClearAttachment a{aspect, 0, value};
    VkClearRect r{{{0, 0}, rp_extent_}, 0, 1};
    vkCmdClearAttachments(cmd_, 1, &a, 1, &r);
}

void Executor::copy_texture(const cmd::CopyTexture& c) {
    if (!c.src || !c.dst) return;
    end_render_pass();
    st_.rt_dirty = true;
    if (!ensure_image(*c.src, cmd_) || !ensure_image(*c.dst, cmd_)) return;
    flush_clears(c.src);
    flush_clears(c.dst);
    upload(*c.src);
    c.dst->gpu_written = true;
    auto level = [](const TextureViewDesc& v) { return (v.flags & kViewLevels) ? static_cast<uint32_t>(v.base_level) : 0u; };
    auto layer = [](const TextureViewDesc& v) { return (v.flags & kViewLayers) ? static_cast<uint32_t>(v.base_layer) : 0u; };
    VkImageBlit b{};
    b.srcSubresource = {c.src->aspect, level(c.src_view), layer(c.src_view), 1};
    b.srcOffsets[0] = {c.src_region.x, c.src_region.y, c.src_region.z};
    b.srcOffsets[1] = {c.src_region.x + c.src_region.width, c.src_region.y + c.src_region.height,
                       c.src_region.z + std::max(c.src_region.depth, 1)};
    b.dstSubresource = {c.dst->aspect, level(c.dst_view), layer(c.dst_view), 1};
    b.dstOffsets[0] = {c.dst_region.x, c.dst_region.y, c.dst_region.z};
    b.dstOffsets[1] = {c.dst_region.x + c.dst_region.width, c.dst_region.y + c.dst_region.height,
                       c.dst_region.z + std::max(c.dst_region.depth, 1)};
    bool same_size = c.src_region.width == c.dst_region.width && c.src_region.height == c.dst_region.height;
    if (same_size && c.src->vk_format == c.dst->vk_format) {
        VkImageCopy ic{b.srcSubresource, b.srcOffsets[0], b.dstSubresource, b.dstOffsets[0],
                       {static_cast<uint32_t>(c.src_region.width), static_cast<uint32_t>(c.src_region.height),
                        static_cast<uint32_t>(std::max(c.src_region.depth, 1))}};
        vkCmdCopyImage(cmd_, c.src->image, VK_IMAGE_LAYOUT_GENERAL, c.dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    } else {
        bool depth = c.src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT;
        vkCmdBlitImage(cmd_, c.src->image, VK_IMAGE_LAYOUT_GENERAL, c.dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                       depth ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    }
    barrier();
}

// --- Draws ---

bool Executor::resolve_texture(Stage s, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp) {
    tex = nullptr;
    smp = nullptr;
    v = {};
    // Tabla de handles: c0 en compute, c2 en gráficos (también la de los TEX con índice inmediato,
    // ver texture_bound_buffer).
    const uint32_t bank = s == Stage::Compute ? kComputeDriverBank : kTextureHandleBank;
    if (r.cbuf_index != bank || r.cbuf_offset < kTextureHandleBase) return false;
    static int logged31 = 0;  // DEPURACIÓN: ranuras altas
    if ((r.cbuf_offset - kTextureHandleBase) / 8 >= 24 && logged31++ < 10)
        NVN_LOG("textura en c%u[0x%x] (%s), ranura %u: %s\n", r.cbuf_index, r.cbuf_offset, nvnshader::stage_name(s),
                (r.cbuf_offset - kTextureHandleBase) / 8,
                st_.bindings[static_cast<int>(s)].textures[std::min<uint32_t>((r.cbuf_offset - kTextureHandleBase) / 8, kMaxTextures - 1)] ? "enlazada" : "vacía");
    uint32_t index = (r.cbuf_offset - kTextureHandleBase) / 8;
    if (index >= kMaxTextures) return false;
    const StageBindings& b = st_.bindings[static_cast<int>(s)];
    uint64_t handle = b.textures[index];
    uint32_t texture_id = handle_texture_id(handle);
    // DEPURACIÓN: NIER_TEX_SUST="a:b,c:d" usa la textura b donde se pide a. Se lee una vez: con la variable
    // vacía, cada textura de cada draw construía dos strings.
    static const std::vector<std::pair<uint32_t, uint32_t>> substitutions = [] {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        const char* list = getenv("NIER_TEX_SUST");
        for (const char* q = list; q && *q;) {
            unsigned from, to;
            if (sscanf(q, "%u:%u", &from, &to) == 2) out.emplace_back(from, to);
            q = strchr(q, ',');
            if (q) q++;
        }
        return out;
    }();
    for (auto [from, to] : substitutions)
        if (from == texture_id) {
            texture_id = to;
            break;
        }
    // Textura y sampler separados: el sampler va por su propia ranura (c2[0x568 + 8·j]), que no tiene por
    // qué coincidir con la de la textura.
    uint32_t sampler_slot = index;
    if (r.has_secondary && r.secondary_cbuf_offset >= kSamplerHandleBase)
        sampler_slot = std::min<uint32_t>((r.secondary_cbuf_offset - kSamplerHandleBase) / 8, kMaxTextures - 1);
    uint32_t sampler_id = b.combined[index] ? handle_sampler_id(handle) : static_cast<uint32_t>(b.samplers[sampler_slot]);
    if (st_.texture_pool && texture_id < st_.texture_pool->textures.size()) {
        const TexturePoolEntry& e = st_.texture_pool->textures[texture_id];
        tex = e.texture;
        v = e.view;
    }
    if (st_.sampler_pool && sampler_id < st_.sampler_pool->samplers.size()) smp = st_.sampler_pool->samplers[sampler_id];
    // DEPURACIÓN: qué texturas muestrean los primeros draws.
    static int logged = 0;
    if (logged < 16 || (frame_ == trace_frame_ && traced_draws_ < 1500)) {
        logged++;
        std::string hex;
        char buf[4];
        if (tex && tex->storage)
            for (int k = 0; k < 32; k++) snprintf(buf, sizeof buf, "%02x", tex->storage[k]), hex += buf;
        if (tex && tex->storage && frame_ == trace_frame_) {  // estadística de toda la memoria de la textura
            size_t n = size_t(tex->desc.width) * tex->desc.height, nz = 0, sum = 0;
            uint8_t mx = 0;
            for (size_t k = 0; k < n; k++) nz += tex->storage[k] != 0, sum += tex->storage[k], mx = std::max(mx, tex->storage[k]);
            hex += " | fin " + std::to_string(reinterpret_cast<uintptr_t>(tex->storage + n)) + " no-cero " + std::to_string(nz) + "/" + std::to_string(n) + " media " + std::to_string(n ? sum / n : 0) +
                   " max " + std::to_string(mx);
        }
        if (tex && frame_ == trace_frame_ && getenv("NIER_VOLCAR_TEX") &&  // DEPURACIÓN: volcar esas texturas
            (std::string(",") + getenv("NIER_VOLCAR_TEX") + ",").find("," + std::to_string(texture_id) + ",") != std::string::npos &&
            std::find(tex_dumps_.begin(), tex_dumps_.end(), std::make_pair(tex, texture_id)) == tex_dumps_.end())
            tex_dumps_.emplace_back(tex, texture_id);
        NVN_LOG("  textura id %u (handle 0x%llx, sampler %u): %s %dx%dx%d target %d niveles %d capas %u formato 0x%x flags 0x%x storage %p subida %d gpu %d %s\n",
                texture_id, static_cast<unsigned long long>(handle), sampler_id, tex ? "ok" : "NO HAY",
                tex ? tex->desc.width : 0, tex ? tex->desc.height : 0, tex ? tex->desc.depth : 0, tex ? tex->desc.target : 0,
                tex ? tex->desc.levels : 0, tex ? tex->layers : 0, tex ? tex->desc.format : 0,
                tex ? tex->desc.flags : 0, tex ? static_cast<void*>(tex->storage) : nullptr, tex ? tex->uploaded : 0,
                tex ? tex->gpu_written : 0, hex.c_str());
    }
    return tex != nullptr;
}

Layout& Executor::layout(const ProgramStage* const* stages) {
    BytesKey<LayoutKeyData> key;
    for (int s = 0; s < kStages; s++) {
        key.value.stages[s] = stages[s];
        key.value.generation[s] = stages[s] ? st_.stage_generation[s] : 0;
    }
    if (auto it = layouts_.find(key); it != layouts_.end()) return it->second;
    // Dinámicos solo en gráficos y si caben en los límites del dispositivo (si no, el camino de siempre).
    uint32_t ubos = 0, ssbos = 0;
    for (int s = 0; s < kStages; s++)
        if (stages[s])
            for (const Resource& r : stages[s]->shader.resources)
                ubos += r.kind == Resource::UniformBuffer, ssbos += r.kind == Resource::StorageBuffer;
    const VkPhysicalDeviceLimits& limits = vk()->props.limits;
    const bool dynamic = !stages[static_cast<int>(Stage::Compute)] && ubos <= limits.maxDescriptorSetUniformBuffersDynamic &&
                         ssbos <= limits.maxDescriptorSetStorageBuffersDynamic;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (int s = 0; s < kStages; s++) {
        if (!stages[s]) continue;
        for (const Resource& r : stages[s]->shader.resources) {
            VkDescriptorType type;
            switch (r.kind) {
                case Resource::UniformBuffer:
                    type = dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    break;
                case Resource::StorageBuffer:
                    type = dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    break;
                case Resource::TextureBuffer: type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; break;
                case Resource::ImageBuffer: type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; break;
                case Resource::Texture: type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; break;
                default: type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; break;
            }
            bindings.push_back({r.binding, type, 1, static_cast<VkShaderStageFlags>(vk_stage(static_cast<Stage>(s))), nullptr});
        }
    }
    Layout l;
    l.dynamic = dynamic;
    static bool logged = false;
    if (!logged) {
        logged = true;
        NVN_LOG("descriptores dinámicos: hasta %u UBO y %u SSBO por set\n", limits.maxDescriptorSetUniformBuffersDynamic,
                limits.maxDescriptorSetStorageBuffersDynamic);
    }
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = static_cast<uint32_t>(bindings.size());
    si.pBindings = bindings.data();
    if (VkResult r = vkCreateDescriptorSetLayout(vk()->device, &si, nullptr, &l.set); r != VK_SUCCESS) {
        l.set = VK_NULL_HANDLE;
        NVN_LOG("vkCreateDescriptorSetLayout (%zu bindings): %s\n", bindings.size(), vk_result_name(r));
    }
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &l.set;
    vkCreatePipelineLayout(vk()->device, &pi, nullptr, &l.pipeline);
    return layouts_[key] = l;
}

VkPipeline Executor::pipeline(const cmd::Draw& d, VkRenderPass rp, Layout& l, uint32_t color_count,
                              VkSampleCountFlagBits samples) {
    PipelineKey key;
    memset(&key, 0, sizeof key);
    memcpy(key.stages, st_.stages, sizeof key.stages);
    memcpy(key.generation, st_.stage_generation, sizeof key.generation);
    key.render_pass = rp;
    key.topology = static_cast<uint32_t>(d.primitive);
    key.polygon = st_.polygon;
    key.depth_stencil = st_.depth_stencil;
    key.color = st_.color;
    key.masks = st_.masks;
    memcpy(key.blend, st_.blend, sizeof key.blend);
    key.color_count = color_count;
    key.samples = samples;
    // El geometry shader de la compactación cuesta mucho en el Adreno (traducido por Eden como passthrough emulado: el
    // pre-pase de profundidad se llevaba ~5,5 ms de GPU en las Ruinas) y lo que produce ya no se usa: sin él, el vertex
    // shader va directo al fragment (mismas posiciones, la misma profundidad). sin_salto_gs: con él, como antes.
    const ProgramStage* gs = st_.stages[static_cast<int>(Stage::Geometry)];
    const ProgramStage* vs_stage = st_.stages[static_cast<int>(Stage::Vertex)];
    const ProgramStage* fs_stage = st_.stages[static_cast<int>(Stage::Fragment)];
    key.skip_gs = gs && gs->compaction_gs && skip_compaction_gs_ && vs_stage &&
                  (!fs_stage || (fs_stage->shader.input_generics & ~vs_stage->shader.output_generics) == 0);
    key.alpha_to_coverage = st_.multisample.alpha_to_coverage;
    const ProgramStage* vs = st_.stages[static_cast<int>(Stage::Vertex)];
    uint64_t inputs = vs ? vs->shader.input_generics : 0;
    for (int k = 0; k < 16; k++) {
        if (!(inputs & (1ull << k))) continue;
        VertexAttribState a = k < static_cast<int>(st_.attribs.size()) ? st_.attribs[k] : VertexAttribState{};
        key.attribs[k] = a;
        if (a.stream < st_.streams.size()) {
            key.strides[k] = st_.streams[a.stream].stride;
            key.divisors[k] = st_.streams[a.stream].divisor;
        }
    }
    uint64_t h = hash_bytes(&key, sizeof key);
    if (auto it = pipelines_.find(h); it != pipelines_.end()) return it->second;

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s] || s == static_cast<int>(Stage::Compute)) continue;
        if (key.skip_gs && s == static_cast<int>(Stage::Geometry)) continue;
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = vk_stage(static_cast<Stage>(s));
        si.module = st_.stages[s]->module;
        const ProgramStage* vs = st_.stages[static_cast<int>(Stage::Vertex)];
        if (s == static_cast<int>(Stage::Geometry) && vs) {
            // Topología de entrada según la primitiva de NVN (orden de OpenGL).
            int topology = d.primitive == 0                       ? 0
                           : d.primitive >= 1 && d.primitive <= 3  ? 1
                           : d.primitive == 10 || d.primitive == 11 ? 2
                           : d.primitive == 12 || d.primitive == 13 ? 4
                                                                    : 3;
            si.module = geometry_variant(*const_cast<ProgramStage*>(st_.stages[s]), vs->shader.output_generics, topology);
            if (!si.module) return pipelines_[h] = VK_NULL_HANDLE;
        }
        si.pName = "main";
        stages.push_back(si);
    }

    // Atributos: cada genérico usado por el vertex shader en su location; un binding por atributo
    // (cada uno con su propio buffer en el staging, ver draw()).
    std::vector<VkVertexInputAttributeDescription> attrs;
    std::vector<VkVertexInputBindingDescription> binds;
    for (uint32_t k = 0; k < 16; k++) {
        if (!(inputs & (1ull << k))) continue;
        const FormatInfo* f = format_info(key.attribs[k].format);
        VkFormat format = f && f->vk != VK_FORMAT_UNDEFINED ? f->vk : VK_FORMAT_R32G32B32A32_SFLOAT;
        attrs.push_back({k, k, format, 0});
        binds.push_back({k, key.strides[k], key.divisors[k] ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = static_cast<uint32_t>(binds.size());
    vi.pVertexBindingDescriptions = binds.data();
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = d.primitive == 7 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : vk_topology(d.primitive);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = vk_polygon_mode(st_.polygon.polygon_mode);
    rs.cullMode = vk_cull_mode(st_.polygon.cull_face);
    // Con el viewport invertido (ver draw) el giro coincide con el de NVN: CW es CW.
    rs.frontFace = st_.polygon.front_face ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = samples;
    ms.alphaToCoverageEnable = st_.multisample.alpha_to_coverage;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    const DepthStencilState& z = st_.depth_stencil;
    ds.depthTestEnable = z.depth_test;
    ds.depthWriteEnable = z.depth_write;
    ds.depthCompareOp = vk_compare_op(z.depth_func);
    ds.stencilTestEnable = z.stencil_test;
    ds.front = {vk_stencil_op(z.front_fail), vk_stencil_op(z.front_zpass), vk_stencil_op(z.front_zfail),
                vk_compare_op(z.front_func), 0xFF, 0xFF, 0};
    ds.back = {vk_stencil_op(z.back_fail), vk_stencil_op(z.back_zpass), vk_stencil_op(z.back_zfail),
               vk_compare_op(z.back_func), 0xFF, 0xFF, 0};
    std::vector<VkPipelineColorBlendAttachmentState> blends(color_count);
    for (uint32_t k = 0; k < color_count; k++) {
        const BlendState& b = st_.blend[k];
        VkPipelineColorBlendAttachmentState& a = blends[k];
        a.blendEnable = (st_.color.blend_enable >> k) & 1;
        a.srcColorBlendFactor = vk_blend_factor(b.src_rgb);
        a.dstColorBlendFactor = vk_blend_factor(b.dst_rgb);
        a.colorBlendOp = vk_blend_op(b.eq_rgb);
        a.srcAlphaBlendFactor = vk_blend_factor(b.src_alpha);
        a.dstAlphaBlendFactor = vk_blend_factor(b.dst_alpha);
        a.alphaBlendOp = vk_blend_op(b.eq_alpha);
        a.colorWriteMask = (st_.masks.masks >> (4 * k)) & 0xF;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = color_count;
    cb.pAttachments = blends.data();
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                            VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 6;
    dy.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pi.stageCount = static_cast<uint32_t>(stages.size());
    pi.pStages = stages.data();
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dy;
    pi.layout = l.pipeline;
    pi.renderPass = rp;
    VkPipeline p = VK_NULL_HANDLE;
    if (VkResult r = vkCreateGraphicsPipelines(vk()->device, pipeline_cache_, 1, &pi, nullptr, &p); r != VK_SUCCESS)
    {
        std::string stages;
        for (int k = 0; k < kStages; k++)
            if (st_.stages[k])
                stages += std::string(" ") + nvnshader::stage_name(static_cast<Stage>(k)) + "(entradas 0x" +
                          [&] { char b[20]; snprintf(b, sizeof b, "%llx", (unsigned long long)st_.stages[k]->shader.input_generics); return std::string(b); }() + ")";
        NVN_LOG("vkCreateGraphicsPipelines: %s (%d), primitiva %d, etapas:%s%s\n", vk_result_name(r), static_cast<int>(r),
                d.primitive, stages.c_str(), key.skip_gs ? " (sin el geometry shader: se vuelve a usar)" : "");
        if (key.skip_gs) const_cast<ProgramStage*>(gs)->compaction_gs = false;  // el siguiente draw, con él
    }
    if (pipelines_.size() % 100 == 99) NVN_LOG("%zu pipelines\n", pipelines_.size() + 1);
    pipelines_created_++;
    last_pipeline_frame_ = frame_;
    return pipelines_[h] = p;
}

// Solo un pool lleno justifica abrir otro. Antes cualquier fallo (layout inválido, sin memoria…) abría un
// pool nuevo en cada vuelta, unos MB de memoria de la GPU cada uno, y al cargar la fábrica se tragó 2,5 GB
// en segundos hasta que el sistema se quedó sin RAM.
VkDescriptorSet Executor::descriptor_set(VkDescriptorSetLayout layout) {
    VkContext* c = vk();
    InFlight& f = inflight_[cur_];
    if (!layout) return VK_NULL_HANDLE;
    for (;; f.pool++) {
        bool fresh = false;
        if (f.pool == f.pools.size()) {
            if (f.pools.size() >= kMaxDescriptorPools) {
                static int logged = 0;
                if (logged++ < 5) NVN_LOG("%zu pools de descriptores en un envío: draw sin dibujar\n", f.pools.size());
                f.pool = f.pools.size() - 1;
                return VK_NULL_HANDLE;
            }
            fresh = true;
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16384},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4096},
                                            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
                                            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 4096;
            pi.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
            pi.pPoolSizes = sizes;
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(c->device, &pi, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
            f.pools.push_back(pool);
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = f.pools[f.pool];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(c->device, &ai, &set);
        if (r == VK_SUCCESS) return set;
        if (fresh || (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)) {
            static int logged = 0;
            if (logged++ < 8) NVN_LOG("vkAllocateDescriptorSets: %s (pool %zu de %zu)\n", vk_result_name(r), f.pool, f.pools.size());
            return VK_NULL_HANDLE;
        }
    }
}

void Executor::log_draw(const cmd::Draw& d) {
    NVN_LOG("draw %llu: primitiva %d, %d %s\n", static_cast<unsigned long long>(draws_), d.primitive, d.count,
            d.index_type >= 0 ? "índices" : "vértices");
    for (int s = 0; s < kStages; s++) {
        const ProgramStage* ps = st_.stages[s];
        if (!ps) continue;
        std::string want, bound;
        for (const Resource& r : ps->shader.resources)
            want += r.kind == Resource::UniformBuffer
                        ? " c" + std::to_string(r.cbuf_index) + "(" + std::to_string(ps->shader.constant_buffer_used[r.cbuf_index]) + ")"
                        : " tex@" + std::to_string(r.cbuf_offset);
        const StageBindings& b = st_.bindings[s];
        for (int k = 0; k < kMaxUniforms; k++)
            if (b.uniforms[k].size) bound += " ubo" + std::to_string(k) + "(" + std::to_string(b.uniforms[k].size) + ")";
        for (int k = 0; k < kMaxTextures; k++)
            if (b.textures[k] || b.samplers[k]) bound += " tex" + std::to_string(k);
        NVN_LOG("  %s pide:%s | enlazado:%s\n", nvnshader::stage_name(static_cast<Stage>(s)), want.c_str(), bound.c_str());
    }
}

void Executor::trace_draw(const cmd::Draw& d) {
    std::string rts;
    for (int k = 0; k < st_.rt.count; k++)
        if (Texture* t = st_.rt.colors[k])
            rts += " " + std::to_string(k) + ":" + std::to_string(t->desc.width) + "x" + std::to_string(t->desc.height) +
                   "/0x" + [&] { char b[32]; snprintf(b, sizeof b, "%x@%llx", t->desc.format, (unsigned long long)(uintptr_t)t->storage); return std::string(b); }();
    if (st_.rt.depth)
        rts += " d:" + [&] { char b[48]; snprintf(b, sizeof b, "%dx%d/0x%x@%llx", st_.rt.depth->desc.width, st_.rt.depth->desc.height,
                                                  st_.rt.depth->desc.format, (unsigned long long)(uintptr_t)st_.rt.depth->storage); return std::string(b); }();
    const auto& v = st_.viewport;
    const auto& ds = st_.depth_stencil;
    NVN_LOG("F draw prim %d n %d inst %d idx %d | rt[%d]%s depth %p | vp %.0f,%.0f %.0fx%.0f z %.2f-%.2f | sc %s %d,%d %ux%u\n",
            d.primitive, d.count, d.instances, d.index_type, st_.rt.count, rts.c_str(), static_cast<void*>(st_.rt.depth),
            v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth, st_.scissor_set ? "on" : "off", st_.scissor.offset.x,
            st_.scissor.offset.y, st_.scissor.extent.width, st_.scissor.extent.height);
    NVN_LOG("F   masks 0x%08x blend_en 0x%x b0 %d/%d/%d/%d eq %d | depth t%d w%d f%d sten %d | cull %d front %d mode %d | attribs %zu streams %zu\n",
            st_.masks.masks, st_.color.blend_enable, st_.blend[0].src_rgb, st_.blend[0].dst_rgb, st_.blend[0].src_alpha,
            st_.blend[0].dst_alpha, st_.blend[0].eq_rgb, int(ds.depth_test), int(ds.depth_write), int(ds.depth_func),
            int(ds.stencil_test), st_.polygon.cull_face, st_.polygon.front_face, st_.polygon.polygon_mode,
            st_.attribs.size(), st_.streams.size());
    for (size_t k = 0; k < st_.attribs.size(); k++)
        NVN_LOG("F   attrib %zu: formato 0x%x stream %d offset %d\n", k, st_.attribs[k].format, st_.attribs[k].stream,
                st_.attribs[k].offset);
    for (size_t k = 0; k < st_.streams.size(); k++) {
        const BufferRange& vb = st_.vertex_buffers[k];
        std::string vals;
        char b[24];
        if (vb.address)
            for (size_t i = 0; i < std::min<size_t>(vb.size / 4, 16); i++)
                snprintf(b, sizeof b, " %.3g", reinterpret_cast<const float*>(vb.address)[i]), vals += b;
        NVN_LOG("F   stream %zu: stride %u divisor %u | vb %#llx (%llu bytes):%s\n", k, st_.streams[k].stride,
                st_.streams[k].divisor, static_cast<unsigned long long>(vb.address),
                static_cast<unsigned long long>(vb.size), vals.c_str());
    }
    if (d.index_type >= 0 && d.index_address) {
        std::string vals;
        for (int i = 0; i < std::min(d.count, 8); i++) {
            uint32_t v = d.index_type == 0   ? reinterpret_cast<const uint8_t*>(d.index_address)[i]
                         : d.index_type == 1 ? reinterpret_cast<const uint16_t*>(d.index_address)[i]
                                             : reinterpret_cast<const uint32_t*>(d.index_address)[i];
            vals += " " + std::to_string(v);
        }
        NVN_LOG("F   índices (%#llx):%s first %d base_vertex %d\n", static_cast<unsigned long long>(d.index_address),
                vals.c_str(), d.first, d.base_vertex);
    }
}

// DEPURACIÓN: el valor de una variable de traza. Un fichero <datos>/<VARIABLE> con el valor la pone en vivo, sin
// reiniciar (se lee una vez por frame trazado); si no hay fichero, la del entorno.
const char* debug_var(const char* var, uint64_t frame) {
    struct Cached {
        uint64_t frame = ~0ull;
        bool from_file = false;
        std::string value;
    };
    static std::unordered_map<std::string, Cached> cache;
    Cached& c = cache[var];
    if (c.frame != frame) {
        c.frame = frame;
        const char* data = getenv("NIER_DATA");
        std::ifstream in(std::string(data ? data : ".") + "/" + var);
        c.from_file = static_cast<bool>(in);
        if (c.from_file) {
            std::getline(in, c.value);
            while (!c.value.empty() && (c.value.back() == '\r' || c.value.back() == ' ')) c.value.pop_back();
        }
    }
    return c.from_file ? c.value.c_str() : getenv(var);
}

// DEPURACIÓN: ¿el draw trazado actual está en la lista "a,b,c" de esa variable de entorno?
bool Executor::traced_in_list(const char* var) const {
    const char* v = debug_var(var, trace_frame_);
    if (!v) return false;
    std::string list = std::string(",") + v + ",";
    return list.find("," + std::to_string(traced_draws_) + ",") != std::string::npos;
}

// Al empezar un draw o un dispatch (nadie guarda entradas de la caché ni datos del envío a medias).
void Executor::maybe_split() {
    if (!recording_ || !staging_) return;
    // Caché de vértices llena: con generaciones se sigue en otra, sin partir el envío ni esperar a la GPU.
    if (generations_) buffer_cache_.next_generation(frame_, submitted_.load(std::memory_order_relaxed) + 1);
    // Tope de RAM por envío: si el staging del envío actual pasa de esto, se manda a la GPU y se sigue en otro,
    // en vez de seguir creciendo (con mucho que copiar en un frame llegó a agotar la RAM del sistema). Lo
    // mismo con la caché de vértices sin generaciones: solo se vacía entre envíos.
    const bool staging_full = staging_->bytes() >= kStagingSoftCap;
    const bool cache_full = !generations_ && buffer_cache_.needs_reset(frame_);
    // Bastantes draws desde el último envío y el render pass cerrado (o a punto de cerrarse: cambian los render
    // targets): se manda lo grabado y la GPU empieza. Partir con el pass abierto obligaría a guardarlo y cargarlo.
    const bool many_draws = split_ && draws_ - submit_draws_ >= kSplitDraws && (!in_render_pass_ || st_.rt_dirty);
    if (!staging_full && !cache_full && !many_draws) return;
    static int logged = 0;
    if ((staging_full || cache_full) && logged++ < 5)
        NVN_LOG("%s en un envío: se parte\n", staging_full ? "staging lleno" : "caché de vértices llena");
    end_render_pass();
    st_.rt_dirty = true;
    splitting_ = true;
    finish(false);
    splitting_ = false;
    begin();
}

void Executor::draw(const cmd::Draw& in) {
    if (!st_.render_enabled) {
        if (frame_ == trace_frame_) prof_disabled_draws_++;
        return;
    }
    maybe_split();
    cmd::Draw d = in;
    if (d.indirect) note_read(4, d.indirect, 20);
    // Argumentos escritos por la GPU (p. ej. el geometry shader del pre-pase de NieR compacta los triángulos
    // visibles y suma su número aquí): el draw se hace con vkCmdDraw*Indirect desde ese buffer.
    // Descarte de triángulos de NieR: el geometry shader del pre-pase compacta los triángulos visibles con
    // operaciones de warp (ballot, shuffle) que el Adreno no tiene en geometría; la emulación de Eden (cada
    // hilo como si estuviera solo) no sirve para un algoritmo cooperativo y la lista salía con triángulos
    // repetidos, índices fuera de rango y huecos (piezas negras). Se dibuja la malla original del pre-pase:
    // la rasterización descarta lo mismo (caras traseras, fuera de pantalla, sin píxeles).
    if (d.indirect && d.index_type >= 0)
        if (auto it = culled_draws_.find(d.indirect); it != culled_draws_.end() && frame_ - it->second.frame <= 1) {
            const cmd::Draw& o = it->second.draw;
            d.index_address = o.index_address;
            d.index_type = o.index_type;
            d.count = o.count;
            d.first = o.first;
            d.base_vertex = o.base_vertex;
            d.base_instance = 0;
            d.instances = 1;
            d.indirect = 0;
        }
    VkBuffer gpu_args = VK_NULL_HANDLE;
    VkDeviceSize gpu_args_offset = 0;
    if (d.indirect) {
        const uint64_t args_size = d.index_type >= 0 ? 20 : 16;
        if (const Overlay* o = find_overlay(d.indirect, args_size, &gpu_args_offset))
            gpu_args = o->buffer;
        else if (gather_overlays(d.indirect, args_size, &gpu_args, &gpu_args_offset))
            st_.rt_dirty = true;  // las copias cerraron el render pass
    }
    if (gpu_args) {
        if (frame_ == trace_frame_) {
            NVN_LOG("F indirecto en la GPU %#llx (índices %d en %#llx)\n", static_cast<unsigned long long>(d.indirect),
                    d.index_type, static_cast<unsigned long long>(d.index_address));
            VkDeviceSize io = 0, ao = 0;
            const Overlay* ia = find_overlay(d.index_address, 4, &io);
            const Overlay* aa = find_overlay(d.indirect, 20, &ao);
            if (ia && aa && ia->data && aa->data)
                traced_gpu_draws_.push_back({d.indirect, aa->data + (ao - aa->offset), ia->data + (io - ia->offset), d.index_type});
        }
        d.count = 0;  // no se sabe en la CPU
        d.indirect = 0;
    } else if (d.indirect) {  // {count, instances, first(índice), base_vertex, base_instance} o {count, instances, first, base_instance}
        const auto* a = reinterpret_cast<const uint32_t*>(d.indirect);
        if (frame_ == trace_frame_)  // DEPURACIÓN
            NVN_LOG("F indirecto %#llx: %u %u %u %u %u (índices %d en %#llx)\n", static_cast<unsigned long long>(d.indirect), a[0],
                    a[1], a[2], a[3], a[4], d.index_type, static_cast<unsigned long long>(d.index_address));
        d.count = static_cast<int>(a[0]);
        d.instances = static_cast<int>(a[1]);
        if (d.index_type >= 0) {
            d.index_address += uint64_t{a[2]} * index_size(d.index_type);
            d.base_vertex = static_cast<int32_t>(a[3]);
            d.base_instance = static_cast<int>(a[4]);
        } else {
            d.first = static_cast<int>(a[2]);
            d.base_instance = static_cast<int>(a[3]);
        }
        d.indirect = 0;
        if (!d.count || !d.instances) return;
    }
    draws_++;
    if (draws_ <= kLoggedDraws) log_draw(d);
    // DEPURACIÓN: los draws con argumentos de la GPU (balas, partículas) se trazan siempre, con sus shaders.
    const bool gpu_counted = gpu_args != VK_NULL_HANDLE;
    if (frame_ == trace_frame_ && (traced_draws_++ < 1500 || gpu_counted)) {
        trace_draw(d);
        log_draw(d);
        const bool half_res = st_.rt.count > 0 && st_.rt.colors[0] && st_.rt.colors[0]->desc.width == 640;
        const char* from = debug_var("NIER_TRAZA_DESDE", trace_frame_);
        const bool dump_code = gpu_counted || traced_draws_ <= 40 || half_res ||
                               (from && *from && traced_draws_ >= atoi(from) && traced_draws_ < atoi(from) + 40) ||
                               traced_in_list("NIER_TRAZA_SHADERS");  // DEPURACIÓN: "a,b,c": los shaders de esos draws
        // <datos>/traza_completa: esta vez con shaders y render targets aunque NIER_TRAZA_LIGERA esté puesta.
        const char* light = NVN_FLAG("traza_completa") ? nullptr : getenv("NIER_TRAZA_LIGERA");
        for (int st = 0; st < kStages && dump_code && (gpu_counted || !(light && *light)); st++)  // escribir en la SD es lento: solo unos pocos
            if (st_.stages[st])
                dump_stage_code(*st_.stages[st], "traza" + std::to_string(traced_draws_) + "_" +
                                                     nvnshader::stage_name(static_cast<Stage>(st)));
        // DEPURACIÓN: NIER_TRAZA_OBJ=<zancada>: el registro del objeto que lee el vertex shader (SSBO 0, índice en
        // el primer u32 de su uniform buffer 1), p. ej. las esferas de las balas (160 bytes por objeto).
        if (const char* obj = getenv("NIER_TRAZA_OBJ"); obj && *obj) {
            const StageBindings& vb = st_.bindings[static_cast<int>(Stage::Vertex)];
            const uint32_t stride = static_cast<uint32_t>(atoi(obj));
            if (vb.uniforms[1].address && vb.storage[0].address && stride) {
                const uint32_t index = *reinterpret_cast<const uint32_t*>(vb.uniforms[1].address);
                const uint64_t at = uint64_t{index} * stride;
                if (at + stride <= vb.storage[0].size) {
                    VkDeviceSize ov_offset = 0;
                    const Overlay* ov = find_overlay(vb.storage[0].address + at, stride, &ov_offset);
                    const auto* f = reinterpret_cast<const float*>(ov && ov->data ? ov->data + (ov_offset - ov->offset)
                                                                                   : reinterpret_cast<const uint8_t*>(vb.storage[0].address + at));
                    std::string vals;
                    char b[24];
                    for (uint32_t i = 0; i < stride / 4 && i < 40; i++) snprintf(b, sizeof b, " %.4g", f[i]), vals += b;
                    NVN_LOG("F   objeto %u (%s):%s\n", index, ov ? "GPU" : "memoria", vals.c_str());
                }
            }
        }
        for (int st = 0; st < kStages; st++) {  // primeros floats de cada uniform buffer enlazado
            if (!st_.stages[st]) continue;
            for (int k = 0; k < kMaxUniforms; k++) {
                const BufferRange& u = st_.bindings[st].uniforms[k];
                if (!u.size) continue;
                const float* f = reinterpret_cast<const float*>(u.address);
                std::string vals;
                char b[24];
                for (size_t i = 0; i < std::min<size_t>(u.size / 4, 16); i++) snprintf(b, sizeof b, " %.3g", f[i]), vals += b;
                NVN_LOG("F   %s ubo%d (%llu bytes):%s\n", nvnshader::stage_name(static_cast<Stage>(st)), k,
                        static_cast<unsigned long long>(u.size), vals.c_str());
            }
        }
    }
    const ProgramStage* vs = st_.stages[static_cast<int>(Stage::Vertex)];
    if (!vs) {
        skipped_draws_++;
        return;
    }
    auto prof_t = prof_now();  // perfil por secciones (solo en los frames medidos)
    // 1. Texturas: crear y subir lo que falte (las subidas van fuera del render pass).
    struct Bound {
        const Resource* r;
        Stage s;
        Texture* tex;
        TextureViewDesc view;
        Sampler* smp;
    };
    static std::vector<Bound> textures;  // se reutiliza entre draws (draw() no se anida): sin reservas por draw
    textures.clear();
    // Para la barrera aplazada (nvn_barrier): ¿escribe storage este draw? ¿muestrea un adjunto del pase abierto?
    bool writes_storage = false, samples_target = false;
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s]) continue;
        for (const Resource& r : st_.stages[s]->shader.resources) {
            if (r.kind == Resource::Image || r.kind == Resource::ImageBuffer || (r.kind == Resource::StorageBuffer && r.is_written))
                writes_storage = true;
            if (r.kind != Resource::Texture) continue;
            Bound b{&r, static_cast<Stage>(s), nullptr, {}, nullptr};
            resolve_texture(b.s, r, b.tex, b.view, b.smp);
            if (b.tex && !pending_clears_.empty()) flush_clears(b.tex);  // su borrado diferido, antes de leerla
            // Muestrear un adjunto del pase abierto: con una barrera aplazada o con el pase que sigue tras volver a
            // enlazar los mismos targets (rt_igual), el pase se corta, como cortaba antes el SetRenderTargets.
            if (b.tex && in_render_pass_ && (pending_barrier_ || opt_.rt_igual))
                samples_target = samples_target || is_pass_target(b.tex);
            if (b.tex && ensure_image(*b.tex, cmd_) && !b.tex->gpu_written && b.tex->storage) {
                bool dirty = !b.tex->uploaded;
                if (!dirty && b.tex->checked_frame != frame_) {
                    b.tex->checked_frame = frame_;
                    dirty = content_hash(*b.tex) != b.tex->content_hash;
                }
                if (dirty) {
                    end_render_pass();
                    st_.rt_dirty = true;
                    upload(*b.tex);
                }
            }
            textures.push_back(b);
        }
    }
    prof_mark(0, prof_t);
    for (auto& [tex, id] : tex_dumps_)  // DEPURACIÓN (NIER_VOLCAR_TEX), ya subidas
        if (tex) {
            end_render_pass();
            st_.rt_dirty = true;
            if (tex->layers > 1) {  // cubemaps y arrays: cada capa, en el mip 0 y en uno intermedio
                for (uint32_t l = 0; l < std::min<uint32_t>(tex->layers, 6); l++) {
                    dump(*tex, "tex" + std::to_string(id) + "_c" + std::to_string(l) + "_m0", l, 0);
                    dump(*tex, "tex" + std::to_string(id) + "_c" + std::to_string(l) + "_m" + std::to_string(tex->desc.levels / 2), l,
                         tex->desc.levels / 2);
                }
            } else {
                dump(*tex, "tex" + std::to_string(id));
            }
            begin();
            tex = nullptr;
        }
    // 2. Render pass, pipeline. Con una barrera aplazada, el pase se corta aquí si este draw la necesita.
    if (in_render_pass_ && ((pending_barrier_ && writes_storage) || samples_target)) {
        end_render_pass();
        st_.rt_dirty = true;
        prof_barrier_splits_++;
    }
    if (!ensure_render_pass()) {
        skipped_draws_++;
        return;
    }
    if (writes_storage) storage_since_barrier_ = true;
    // Resolución dinámica (perfil): el viewport de la escena, la que tiene profundidad de 1280x720.
    if (st_.rt.depth && st_.rt.depth->desc.width == 1280 && st_.rt.depth->desc.height == 720) {
        scene_vp_w_ = std::max(scene_vp_w_, static_cast<int>(st_.viewport.width));
        scene_vp_h_ = std::max(scene_vp_h_, static_cast<int>(st_.viewport.height));
        if (st_.viewport.width >= 320) scene_vp_min_w_ = std::min(scene_vp_min_w_, static_cast<int>(st_.viewport.width));
    }
    Layout& l = layout(st_.stages);
    // DEPURACIÓN: NIER_SIN_BLEND="a,b": esos draws del frame trazado se dibujan sin mezcla (se ve su color tal cual).
    // NIER_SIN_DEPTH: esos, sin depth test.
    const auto saved_blend = st_.color.blend_enable;
    const auto saved_depth = st_.depth_stencil;
    if (frame_ == trace_frame_ && traced_in_list("NIER_SIN_BLEND")) st_.color.blend_enable = 0;
    if (frame_ == trace_frame_ && traced_in_list("NIER_SIN_DEPTH")) st_.depth_stencil.depth_test = false;
    if (depth_lequal_ && st_.depth_stencil.depth_func == kDepthEqual) st_.depth_stencil.depth_func = kDepthLequal;
    VkPipeline p = pipeline(d, rp_, l, rp_color_count_, rp_samples_);
    st_.color.blend_enable = saved_blend;
    st_.depth_stencil = saved_depth;
    if (!p) {
        skipped_draws_++;
        return;
    }
    if (p != bound_graphics_) {  // seguidos con el mismo estado: sin volver a enlazarlo
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        bound_graphics_ = p;
    }

    prof_mark(1, prof_t);
    // 3. Descriptores: uniform buffers copiados al staging y texturas resueltas por los pools.
    auto prof_sub = prof_now();
    // Se reutilizan entre draws. Las writes apuntan dentro de buffers e images: reservados para el máximo de
    // recursos de todas las etapas, nunca se realojan a mitad (con 64 podían quedar colgando).
    static std::vector<VkWriteDescriptorSet> writes;
    static std::vector<VkDescriptorBufferInfo> buffers;
    static std::vector<VkDescriptorImageInfo> images;
    static std::vector<uint32_t> dynamic_offsets;
    static std::vector<uint64_t> contents;  // lo que define el set: buffers y tamaños, vistas y samplers
    writes.clear();
    buffers.clear();
    images.clear();
    dynamic_offsets.clear();
    contents.clear();
    const bool dynamic = l.dynamic;
    auto push_buffer = [&](VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
        if (dynamic) {
            buffers.push_back({buffer, 0, size});
            dynamic_offsets.push_back(static_cast<uint32_t>(offset));
            contents.push_back(reinterpret_cast<uint64_t>(buffer));
            contents.push_back(size);
        } else {
            buffers.push_back({buffer, offset, size});
        }
    };
    buffers.reserve(kStages * (kMaxUniforms + kMaxStorage + 2));
    images.reserve(kStages * kMaxTextures + kStages * 8);
    writes.reserve(buffers.capacity() + images.capacity());
    const VkDeviceSize ubo_align = std::max<VkDeviceSize>(vk()->props.limits.minUniformBufferOffsetAlignment, 16);
    const VkDeviceSize ssbo_align = std::max<VkDeviceSize>(vk()->props.limits.minStorageBufferOffsetAlignment, 16);
    uint8_t graphics_driver[kGraphicsDriverSize];
    bool graphics_driver_ready = false;
    uint8_t handles_cbuf[0x700];
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s]) continue;
        for (const Resource& r : st_.stages[s]->shader.resources) {
            if (r.kind != Resource::UniformBuffer && r.kind != Resource::StorageBuffer) continue;
            BufferRange range{};
            const void* src = nullptr;
            if (r.kind == Resource::StorageBuffer) {
                const uint32_t base = kGraphicsStorageBase + kGraphicsStorageStride * s;
                if (r.cbuf_index == 0 && r.cbuf_offset >= base && r.cbuf_offset < base + 16 * kMaxStorage)
                    range = st_.bindings[s].storage[(r.cbuf_offset - base) / 16];
                if (r.is_written && range.address) {
                    static int logged = 0;
                    if (frame_ == trace_frame_ && range.size == 20 && d.index_type >= 0)  // DEPURACIÓN: args de la compactación
                        prepass_args_[range.address] = {d.index_address, static_cast<uint32_t>(d.count), d.index_type};
                    if (s == static_cast<int>(Stage::Geometry) && range.size == 20 && d.index_type >= 0 && !d.indirect) {
                        culled_draws_[range.address] = {d, frame_};
                        const_cast<ProgramStage*>(st_.stages[s])->compaction_gs = true;
                    }
                    if (frame_ == trace_frame_)
                        NVN_LOG("F SSBO escrito %#llx+%llu\n", static_cast<unsigned long long>(range.address),
                                static_cast<unsigned long long>(range.size));
                    if (logged++ < 4)
                        NVN_LOG("SSBO escrito desde un shader %s: %#llx+%llu, se queda en la GPU\n",
                                nvnshader::stage_name(static_cast<Stage>(s)), static_cast<unsigned long long>(range.address),
                                static_cast<unsigned long long>(range.size));
                }
            }
            if (r.kind == Resource::UniformBuffer) {
                int index = static_cast<int>(r.cbuf_index) - static_cast<int>(kUniformBankBase);
                if (index >= 0 && index < kMaxUniforms) range = st_.bindings[s].uniforms[index];
                if (r.cbuf_index == kShaderConstantsBank)
                    range = {st_.stages[s]->constants, st_.stages[s]->constants_size};
                if (r.cbuf_index == kTextureHandleBank) {
                    // c2 leído como datos: algunos shaders comprueban o combinan los handles (textura en
                    // 0x168 + 8·i, sampler separado en 0x568 + 8·i, y los combinan con OR). Con ceros
                    // tomaban el camino "sin textura" y el material salía negro.
                    memset(handles_cbuf, 0, sizeof handles_cbuf);
                    const StageBindings& sbind = st_.bindings[s];
                    for (int k = 0; k < kMaxTextures; k++) {
                        const uint32_t tex = static_cast<uint32_t>(sbind.textures[k]);
                        const uint32_t smp = static_cast<uint32_t>(sbind.samplers[k]);
                        if (kTextureHandleBase + 8 * k + 4 <= sizeof handles_cbuf)
                            memcpy(handles_cbuf + kTextureHandleBase + 8 * k, &tex, 4);
                        if (kSamplerHandleBase + 8 * k + 4 <= sizeof handles_cbuf)
                            memcpy(handles_cbuf + kSamplerHandleBase + 8 * k, &smp, 4);
                    }
                    src = handles_cbuf;
                    range.size = sizeof handles_cbuf;
                }
                if (r.cbuf_index == 0) {  // c0 del driver: tamaños de los SSBO (direcciones a 0, ver dispatch)
                    if (!graphics_driver_ready) {
                        memset(graphics_driver, 0, sizeof graphics_driver);
                        // Direcciones reales: Eden calcula cada acceso como dirección − c0 (alineación 1) y
                        // el buffer enlazado empieza en la dirección, así que da igual; y los shaders que
                        // comprueban "puntero nulo" ven el buffer.
                        for (int st = 0; st < kStages; st++)
                            for (int k = 0; k < kMaxStorage; k++) {
                                const BufferRange& sr = st_.bindings[st].storage[k];
                                uint32_t size = static_cast<uint32_t>(sr.size);
                                uint32_t at = kGraphicsStorageBase + kGraphicsStorageStride * st + 16 * k;
                                if (at + 12 > sizeof graphics_driver) continue;
                                memcpy(graphics_driver + at, &sr.address, 8);
                                memcpy(graphics_driver + at + 8, &size, 4);
                            }
                        graphics_driver_ready = true;
                    }
                    src = graphics_driver;
                    range.size = sizeof graphics_driver;
                }
            }
            if (!src && range.address) src = reinterpret_cast<const void*>(range.address);
            if (range.address) note_read(r.kind == Resource::UniformBuffer ? 2 : 3, range.address, range.size);
            VkDeviceSize size = std::min<VkDeviceSize>(range.size ? range.size : 256,
                                                       r.kind == Resource::UniformBuffer ? 65536 : kStagingChunk);
            if (r.kind == Resource::UniformBuffer && r.cbuf_index < 18) {  // solo lo que lee el shader
                uint32_t used = st_.stages[s]->shader.constant_buffer_used[r.cbuf_index];
                if (used) size = std::min<VkDeviceSize>(size, (used + 15) & ~15u);
            }
            VkDeviceSize ov_offset = 0;
            const bool from_game = src == reinterpret_cast<const void*>(range.address) && range.size;
            const Overlay* ov = !from_game ? nullptr
                                : r.kind == Resource::StorageBuffer && r.is_written ? make_overlay(range.address, size, &ov_offset)
                                                                                    : find_overlay(range.address, size, &ov_offset);
            if (frame_ == trace_frame_ && r.kind == Resource::StorageBuffer && !r.is_written) {  // DEPURACIÓN
                std::string vals;
                char b[24];
                if (ov && ov->data)
                    for (size_t i = 0; i < std::min<size_t>(size / 4, 8); i++)
                        snprintf(b, sizeof b, " %.4g", reinterpret_cast<const float*>(ov->data + (ov_offset - ov->offset))[i]), vals += b;
                NVN_LOG("F SSBO leído %#llx+%llu: %s%s\n", static_cast<unsigned long long>(range.address),
                        static_cast<unsigned long long>(size), ov ? "de la GPU, último valor:" : "de la memoria", vals.c_str());
            }
            if (ov) {
                push_buffer(ov->buffer, ov_offset, size);
            } else if (const auto* hit = from_game ? copied_.find(range.address, size) : nullptr) {
                push_buffer(hit->first, hit->second, size);  // ya copiado en este envío
            } else {
                Staging::Alloc a = staging_->alloc(size, r.kind == Resource::UniformBuffer ? ubo_align : ssbo_align);
                if (!a.data) return;
                if (src && range.size)
                    memcpy(a.data, src, size);
                else
                    memset(a.data, 0, size);
                prof_copied_bytes_ += size;
                if (from_game) copied_.insert(range.address, size, a.buffer, a.offset);
                push_buffer(a.buffer, a.offset, size);
            }
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstBinding = r.binding;
            w.descriptorCount = 1;
            w.descriptorType = r.kind == Resource::UniformBuffer
                                   ? (dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                                   : (dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            w.pBufferInfo = &buffers.back();
            writes.push_back(w);
        }
    }
    prof_mark_sub(0, prof_sub);
    for (const Bound& b : textures) {
        Texture* t = b.tex && b.tex->image ? b.tex : &dummy_;
        if (t == &dummy_ && frame_ == trace_frame_)  // DEPURACIÓN
            NVN_LOG("F textura sin resolver: c%u[0x%x] (%s)\n", b.r->cbuf_index, b.r->cbuf_offset,
                    b.tex ? "sin imagen" : "no está en el pool");
        VkImageView iv = view(*t, t == b.tex ? b.view : TextureViewDesc{}, false);
        images.push_back({sampler(b.smp), iv, VK_IMAGE_LAYOUT_GENERAL});
        if (dynamic) {
            contents.push_back(reinterpret_cast<uint64_t>(iv));
            contents.push_back(reinterpret_cast<uint64_t>(images.back().sampler));
        }
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = b.r->binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &images.back();
        writes.push_back(w);
    }
    prof_mark_sub(1, prof_sub);
    // El set del draw anterior con este layout sirve si apunta a lo mismo (solo cambian los desplazamientos,
    // que van en el bind). Vale hasta el siguiente begin(): ahí se vacían los pools y se liberan buffers y vistas.
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (dynamic && l.memo_set && l.memo_epoch == descriptor_epoch_ && l.memo_key == contents) {
        set = l.memo_set;
    } else {
        set = descriptor_set(l.set);
        prof_mark_sub(3, prof_sub);
        if (!set) {
            skipped_draws_++;
            return;
        }
        for (VkWriteDescriptorSet& w : writes) w.dstSet = set;
        vkUpdateDescriptorSets(vk()->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        if (dynamic) {
            l.memo_set = set;
            l.memo_epoch = descriptor_epoch_;
            l.memo_key = contents;
        }
    }
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, l.pipeline, 0, 1, &set,
                            static_cast<uint32_t>(dynamic_offsets.size()), dynamic_offsets.data());
    prof_mark_sub(2, prof_sub);

    prof_mark(2, prof_t);
    // 4. Estado dinámico.
    // NVN mapea la Y de NDC como OpenGL (+1 arriba) aunque la ventana tenga el origen arriba a la
    // izquierda (viewport, scissor y gl_FragCoord desde arriba); Vulkan pone -1 arriba. Viewport con
    // altura negativa (Vulkan 1.1): misma caja, Y invertida.
    // Persiste en el command buffer (todos los pipelines declaran los mismos estados dinámicos): solo lo que cambia.
    DynamicState ds{};
    ds.viewport = st_.viewport;
    ds.viewport.y += ds.viewport.height;
    ds.viewport.height = -ds.viewport.height;
    ds.scissor = st_.scissor_set ? st_.scissor : VkRect2D{{0, 0}, rp_extent_};
    if (!rp_area_full_) ds.scissor = intersect(ds.scissor, rp_area_);  // nada fuera del renderArea, pase lo que pase
    memcpy(ds.blend, st_.blend_color, sizeof ds.blend);
    for (int f = 0; f < 2; f++)
        ds.ref[f] = st_.stencil_ref[f], ds.write_mask[f] = st_.stencil_mask[f], ds.compare_mask[f] = st_.stencil_value_mask[f];
    const bool all = !dynamic_valid_;
    if (all || memcmp(&ds.viewport, &bound_dynamic_.viewport, sizeof ds.viewport)) vkCmdSetViewport(cmd_, 0, 1, &ds.viewport);
    if (all || memcmp(&ds.scissor, &bound_dynamic_.scissor, sizeof ds.scissor)) vkCmdSetScissor(cmd_, 0, 1, &ds.scissor);
    if (all || memcmp(ds.blend, bound_dynamic_.blend, sizeof ds.blend)) vkCmdSetBlendConstants(cmd_, ds.blend);
    constexpr VkStencilFaceFlags kFace[2] = {VK_STENCIL_FACE_FRONT_BIT, VK_STENCIL_FACE_BACK_BIT};
    for (int f = 0; f < 2; f++) {
        if (all || ds.ref[f] != bound_dynamic_.ref[f]) vkCmdSetStencilReference(cmd_, kFace[f], ds.ref[f]);
        if (all || ds.write_mask[f] != bound_dynamic_.write_mask[f]) vkCmdSetStencilWriteMask(cmd_, kFace[f], ds.write_mask[f]);
        if (all || ds.compare_mask[f] != bound_dynamic_.compare_mask[f])
            vkCmdSetStencilCompareMask(cmd_, kFace[f], ds.compare_mask[f]);
    }
    bound_dynamic_ = ds;
    dynamic_valid_ = true;

    prof_mark(3, prof_t);
    // 5. Vértices: cada atributo usado, desde la caché si su stream no cambia; si no, copiado al staging.
    const uint8_t* indices = d.index_type >= 0 ? reinterpret_cast<const uint8_t*>(d.index_address) : nullptr;
    const uint32_t isize = indices ? index_size(d.index_type) : 0;
    VkBuffer gpu_index = VK_NULL_HANDLE;  // índices que escribió la GPU
    VkDeviceSize gpu_index_offset = 0;
    if (indices)
        if (const Overlay* o = find_overlay(d.index_address, VkDeviceSize{isize} * std::max(d.count, 1), &gpu_index_offset))
            gpu_index = o->buffer;
    if ((gpu_index || gpu_args) && (d.primitive == 7 || d.index_type == 0)) {
        static int logged = 0;
        if (logged++ < 8)
            NVN_LOG("draw con datos de la GPU no soportado (primitiva %d, índices %d, args %d, índices en la GPU %d)\n",
                    d.primitive, d.index_type, gpu_args != VK_NULL_HANDLE, gpu_index != VK_NULL_HANDLE);
        skipped_draws_++;
        return;
    }
    if (indices && !gpu_index) note_read(1, d.index_address, uint64_t{isize} * d.count);
    // Cuántos índices leer: los del draw; con el número en la GPU y los índices en la CPU (p. ej. quads de
    // balas o partículas, que la GPU cuenta) no se sabe: un rango generoso, hasta el final del memory pool.
    size_t index_bytes = size_t{isize} * d.count;
    if (gpu_args && indices && !gpu_index) {
        size_t avail = 0;
        g_memory_pools.find_if([&](const void*, const MemoryPool& pool) {
            auto start = reinterpret_cast<uint64_t>(pool.storage);
            if (d.index_address < start || d.index_address >= start + pool.size) return false;
            avail = start + pool.size - d.index_address;
            return true;
        });
        index_bytes = std::min<size_t>(avail, size_t{1} << 20) / isize * isize;
        if (frame_ == trace_frame_)
            NVN_LOG("F indirecto en la GPU con índices de la CPU: %zu bytes desde %#llx\n", index_bytes,
                    static_cast<unsigned long long>(d.index_address));
    }
    const BufferCache::Entry* cached_indices =
        indices && !gpu_index && d.primitive != 7 && index_bytes
            ? buffer_cache_.get(d.index_address, index_bytes, d.index_type, submit_index_)
            : nullptr;
    VkBuffer host_index = VK_NULL_HANDLE;  // ese rango, si cambia demasiado para la caché: copia directa
    VkDeviceSize host_index_offset = 0;
    if (gpu_args && indices && !gpu_index && !cached_indices && index_bytes) {
        Staging::Alloc al = staging_->alloc(index_bytes, 16);
        if (!al.data) return;
        memcpy(al.data, indices, index_bytes);
        host_index = al.buffer;
        host_index_offset = al.offset;
    }
    if (gpu_args && indices && !gpu_index && !cached_indices && !host_index) {
        skipped_draws_++;
        return;
    }
    // Con índices escritos por la GPU no se sabe cuántos vértices se leen: los streams se copian enteros. Con
    // argumentos de la GPU pero índices de la CPU, el máximo del rango de índices acota los vértices (copiar los
    // streams enteros en cada draw de partículas agotaba la RAM en segundos).
    const bool unknown_vertices = gpu_index != VK_NULL_HANDLE || (gpu_args && !indices);
    // Ventana de vértices: los streams por vértice se copian (y se enlazan) desde first_vertex, vertex_count
    // vértices. En un draw directo empieza en el primer vértice que usa y el draw se desplaza para compensar:
    // las mallas de la fábrica comparten un buffer de 4 MB con índices absolutos (149724…) y cada draw copiaba
    // desde el vértice 0 hasta su máximo, 4 MB por draw y sin compartirse (cada uno con su máximo).
    const bool direct = !gpu_args && !gpu_index;
    uint32_t first_vertex = 0, vertex_count = 1u << 24;
    int32_t vertex_offset = d.base_vertex;  // vertexOffset de los draws indexados
    if (!unknown_vertices && !indices) {
        first_vertex = direct ? static_cast<uint32_t>(d.first) : 0;
        vertex_count = static_cast<uint32_t>(d.first + d.count) - first_vertex;
    } else if (!unknown_vertices) {
        uint32_t lo, hi;
        if (cached_indices)
            lo = cached_indices->min_index, hi = cached_indices->max_index;
        else
            BufferCache::index_range(indices, index_bytes, d.index_type, &lo, &hi);
        if (lo > hi) lo = hi = 0;  // sin índices válidos
        if (direct && d.base_vertex >= 0) {
            first_vertex = lo + static_cast<uint32_t>(d.base_vertex);
            vertex_count = hi - lo + 1;
            vertex_offset = -static_cast<int32_t>(lo);
        } else {
            vertex_count = hi + 1 + static_cast<uint32_t>(std::max(d.base_vertex, 0));
        }
    }
    // Mayor desplazamiento de atributo de cada stream: todos sus atributos comparten una entrada de la caché
    // (con el desplazamiento de cada uno, un stream de 5 atributos se copiaba 5 veces).
    uint32_t stream_offset[kMaxVertexBuffers] = {};
    // De dónde sale cada stream (overlay, caché o copia): se resuelve con su primer atributo y los demás solo le
    // suman su desplazamiento. Los binds se juntan en una llamada, solo con lo que cambia respecto a cmd_.
    std::pair<VkBuffer, VkDeviceSize> stream_src[kMaxVertexBuffers] = {};
    VkBuffer bind_buf[16] = {};
    VkDeviceSize bind_off[16] = {};
    uint32_t bind_used = 0;
    auto bind = [&](uint32_t k, VkBuffer buffer, VkDeviceSize offset) {
        bind_buf[k] = buffer;
        bind_off[k] = offset;
        bind_used |= 1u << k;
    };
    for (uint32_t k = 0; k < 16 && k < st_.attribs.size(); k++)
        if ((vs->shader.input_generics & (1ull << k)) && st_.attribs[k].stream < kMaxVertexBuffers)
            stream_offset[st_.attribs[k].stream] = std::max<uint32_t>(stream_offset[st_.attribs[k].stream], st_.attribs[k].offset);
    for (uint32_t k = 0; k < 16; k++) {
        if (!(vs->shader.input_generics & (1ull << k))) continue;
        VertexAttribState a = k < st_.attribs.size() ? st_.attribs[k] : VertexAttribState{};
        uint32_t stride = a.stream < st_.streams.size() ? st_.streams[a.stream].stride : 0;
        BufferRange range = a.stream < kMaxVertexBuffers ? st_.vertex_buffers[a.stream] : BufferRange{};
        uint32_t divisor = a.stream < st_.streams.size() ? st_.streams[a.stream].divisor : 0;
        if (range.address) note_read(0, range.address, range.size);
        // Los streams por instancia no se desplazan (vertexOffset no les afecta).
        const VkDeviceSize skip = divisor ? 0 : VkDeviceSize{stride} * first_vertex;
        const bool in_range = range.address && range.size > a.offset;
        if (in_range && a.stream < kMaxVertexBuffers && stream_src[a.stream].first) {
            bind(k, stream_src[a.stream].first, stream_src[a.stream].second + a.offset);
            continue;
        }
        auto remember = [&](VkBuffer buffer, VkDeviceSize base) {
            if (a.stream < kMaxVertexBuffers) stream_src[a.stream] = {buffer, base};
            bind(k, buffer, base + a.offset);
        };
        if (in_range) {
            VkDeviceSize ov_offset = 0;  // vértices escritos por la GPU (p. ej. instancias de un compute)
            if (const Overlay* o = find_overlay(range.address, range.size, &ov_offset)) {
                remember(o->buffer, ov_offset + skip);
                continue;
            }
        }
        // El rango desde el primer vértice de la ventana.
        if (skip < range.size) {
            range.address += skip;
            range.size -= skip;
        } else if (skip) {
            range = BufferRange{};  // fuera de lo enlazado: ceros
        }
        uint32_t elements = divisor ? static_cast<uint32_t>(d.base_instance + d.instances) : vertex_count;
        VkDeviceSize need = stride ? VkDeviceSize{stride} * elements : 64;
        // A la caché va solo lo que lee el draw: con el rango enlazado entero (mallas que enlazan desde su
        // vértice hasta el final de un buffer compartido) cada malla copiaba MB que no usa.
        const VkDeviceSize read = (a.stream < kMaxVertexBuffers ? stream_offset[a.stream] : a.offset) + need;
        if (range.address && range.size > a.offset)
            if (const BufferCache::Entry* e = buffer_cache_.get(range.address, std::min<VkDeviceSize>(range.size, read), -1,
                                                                 submit_index_)) {
                remember(e->buffer, e->offset);
                continue;
            }
        const VkDeviceSize window = std::min<VkDeviceSize>(read, range.size);
        if (range.address && range.size > a.offset && window <= kStagingChunk && a.stream < kMaxVertexBuffers) {  // el stream, una vez
            const VkDeviceSize size = std::max<VkDeviceSize>(window, 64);
            Staging::Alloc al = staging_->alloc(size, 16);
            if (!al.data) return;
            memcpy(al.data, reinterpret_cast<const void*>(range.address), std::min<VkDeviceSize>(size, range.size));
            remember(al.buffer, al.offset);
            continue;
        }
        VkDeviceSize avail = range.size > a.offset ? range.size - a.offset : 0;
        VkDeviceSize size = std::min<VkDeviceSize>(std::max<VkDeviceSize>(std::min(need, avail), 64), kStagingChunk);
        Staging::Alloc al = staging_->alloc(size, 16);
        if (!al.data) return;
        if (range.address && avail)
            memcpy(al.data, reinterpret_cast<const void*>(range.address + a.offset), std::min(size, avail));
        else
            memset(al.data, 0, size);
        bind(k, al.buffer, al.offset);
    }
    if (bind_used) {
        uint32_t lo = 16, hi = 0;
        for (uint32_t k = 0; k < 16; k++)
            if ((bind_used >> k & 1) && (bind_buf[k] != bound_vb_buf_[k] || bind_off[k] != bound_vb_off_[k]))
                lo = std::min(lo, k), hi = k;
        if (lo <= hi) {
            const uint32_t first_used = static_cast<uint32_t>(__builtin_ctz(bind_used));
            for (uint32_t k = lo; k <= hi; k++)
                if (!(bind_used >> k & 1)) {  // hueco: lo que ya había, o cualquier buffer válido
                    bind_buf[k] = bound_vb_buf_[k] ? bound_vb_buf_[k] : bind_buf[first_used];
                    bind_off[k] = bound_vb_buf_[k] ? bound_vb_off_[k] : 0;
                }
            vkCmdBindVertexBuffers(cmd_, lo, hi - lo + 1, bind_buf + lo, bind_off + lo);
            for (uint32_t k = lo; k <= hi; k++) bound_vb_buf_[k] = bind_buf[k], bound_vb_off_[k] = bind_off[k];
        }
    }

    prof_mark(4, prof_t);
    // 6. Dibujar. Los quads (NVN 7) se convierten a triángulos.
    if (d.primitive == 7) {
        uint32_t quads = static_cast<uint32_t>(d.count) / 4;
        Staging::Alloc al = staging_->alloc(quads * 6 * sizeof(uint32_t), 16);
        if (!al.data) return;
        auto* out = reinterpret_cast<uint32_t*>(al.data);
        for (uint32_t q = 0; q < quads; q++) {
            uint32_t v[4];
            for (int j = 0; j < 4; j++) {
                uint32_t k = q * 4 + j;
                v[j] = !indices ? d.first + k - first_vertex : d.index_type == 0 ? indices[k] : d.index_type == 1 ? reinterpret_cast<const uint16_t*>(indices)[k]
                                                                                                     : reinterpret_cast<const uint32_t*>(indices)[k];
            }
            uint32_t tri[6] = {v[0], v[1], v[2], v[0], v[2], v[3]};
            memcpy(out + q * 6, tri, sizeof tri);
        }
        vkCmdBindIndexBuffer(cmd_, al.buffer, al.offset, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd_, quads * 6, std::max(d.instances, 1), 0, indices ? vertex_offset : 0, d.base_instance);
    } else if (gpu_args || gpu_index) {
        const VkIndexType type = d.index_type == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
        if (gpu_index)
            vkCmdBindIndexBuffer(cmd_, gpu_index, gpu_index_offset, type);
        else if (cached_indices)
            vkCmdBindIndexBuffer(cmd_, cached_indices->buffer, cached_indices->offset, type);
        else if (host_index)
            vkCmdBindIndexBuffer(cmd_, host_index, host_index_offset, type);
        if (gpu_args && indices)
            vkCmdDrawIndexedIndirect(cmd_, gpu_args, gpu_args_offset, 1, 20);
        else if (gpu_args)
            vkCmdDrawIndirect(cmd_, gpu_args, gpu_args_offset, 1, 16);
        else
            vkCmdDrawIndexed(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), 0, vertex_offset, d.base_instance);
    } else if (cached_indices) {
        vkCmdBindIndexBuffer(cmd_, cached_indices->buffer, cached_indices->offset,
                             d.index_type == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), 0, vertex_offset, d.base_instance);
    } else if (indices) {
        // Vulkan 1.1 sin VK_EXT_index_type_uint8 activo: los índices de 8 bits se amplían a 16.
        uint32_t out_size = d.index_type == 0 ? 2 : isize;
        Staging::Alloc al = staging_->alloc(VkDeviceSize{out_size} * d.count, 16);
        if (!al.data) return;
        if (d.index_type == 0)
            for (int k = 0; k < d.count; k++) reinterpret_cast<uint16_t*>(al.data)[k] = indices[k];
        else
            memcpy(al.data, indices, VkDeviceSize{isize} * d.count);
        vkCmdBindIndexBuffer(cmd_, al.buffer, al.offset, d.index_type == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), 0, vertex_offset, d.base_instance);
    } else {
        vkCmdDraw(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), static_cast<uint32_t>(d.first) - first_vertex,
                  static_cast<uint32_t>(d.base_instance));
    }
    prof_mark(5, prof_t);
    // DEPURACIÓN: NIER_VOLCAR_TRAS="a,b,c": vuelca el render target 0 tras esos draws del frame trazado.
    if (frame_ == trace_frame_ && st_.rt.count > 0 && st_.rt.colors[0]) {
        const Texture* rt0 = st_.rt.colors[0];
        const char* by_size = debug_var("NIER_VOLCAR_TAM", trace_frame_);  // DEPURACIÓN: "640x360": tras cada draw a un RT de ese tamaño
        const bool size_match = by_size && *by_size &&
                                std::to_string(rt0->desc.width) + "x" + std::to_string(rt0->desc.height) == by_size;
        if (traced_in_list("NIER_VOLCAR_TRAS") || size_match) {
            end_render_pass();
            st_.rt_dirty = true;
            dump(*st_.rt.colors[0], "tras" + std::to_string(traced_draws_));
            begin();  // dump() envió lo grabado hasta aquí: se sigue en un command buffer nuevo
        }
    }
    if (draws_ % 5000 == 0 && NVN_FLAG("registro"))
        NVN_LOG("%llu draws (%llu sin dibujar), %zu pipelines\n", static_cast<unsigned long long>(draws_),
                static_cast<unsigned long long>(skipped_draws_), pipelines_.size());
}

VkPipeline Executor::compute_pipeline(const ProgramStage& cs, Layout& l) {
    const std::pair<const ProgramStage*, uint64_t> key{&cs, st_.stage_generation[static_cast<int>(Stage::Compute)]};
    if (auto it = compute_pipelines_.find(key); it != compute_pipelines_.end()) return it->second;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = cs.module;
    ci.stage.pName = "main";
    ci.layout = l.pipeline;
    VkPipeline p = VK_NULL_HANDLE;
    if (VkResult r = vkCreateComputePipelines(vk()->device, pipeline_cache_, 1, &ci, nullptr, &p); r != VK_SUCCESS)
        NVN_LOG("vkCreateComputePipelines: %s\n", vk_result_name(r));
    pipelines_created_++;
    last_pipeline_frame_ = frame_;
    return compute_pipelines_[key] = p;
}

const Executor::Overlay* Executor::find_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset) {
    if (overlays_.empty() || !address || !overlay_filter_[overlay_slot(address >> 16)]) return nullptr;
    auto it = overlays_.upper_bound(address);
    if (it == overlays_.begin()) return nullptr;
    --it;
    Overlay& o = it->second;
    if (address + size > it->first + o.size) return nullptr;
    if (o.checked != submit_index_) {  // ¿la CPU ha reescrito el rango? entonces manda la memoria del juego
        o.checked = submit_index_;
        if (range_hash(reinterpret_cast<const uint8_t*>(it->first), o.size) != o.cpu_hash) {
            erase_overlay(it);
            return nullptr;
        }
    }
    *offset = o.offset + (address - it->first);
    return &o;
}

bool Executor::copy_from_overlay(uint64_t src, uint64_t dst, uint64_t size) {
    VkDeviceSize src_offset = 0;
    const Overlay* so = find_overlay(src, size, &src_offset);
    if (!so) return false;
    const VkBuffer src_buffer = so->buffer;
    VkDeviceSize dst_offset = 0;
    const Overlay* d = make_overlay(dst, size, &dst_offset);  // puede invalidar `so`: ya no se usa
    if (!d) return false;
    barrier();
    VkBufferCopy region{src_offset, dst_offset, size};
    vkCmdCopyBuffer(cmd_, src_buffer, d->buffer, 1, &region);
    barrier();
    return true;
}

bool Executor::gather_overlays(uint64_t address, uint64_t size, VkBuffer* buffer, VkDeviceSize* offset) {
    auto it = overlays_.upper_bound(address);
    if (it != overlays_.begin()) --it;
    std::vector<std::pair<const Overlay*, uint64_t>> parts;  // overlay e inicio de su rango
    while (it != overlays_.end() && it->first < address + size) {
        Overlay& o = it->second;
        if (it->first + o.size <= address) {
            ++it;
            continue;
        }
        // Como find_overlay: si la CPU reescribió el rango desde que lo escribió la GPU, manda la CPU.
        if (o.checked != submit_index_) {
            o.checked = submit_index_;
            if (range_hash(reinterpret_cast<const uint8_t*>(it->first), o.size) != o.cpu_hash) {
                it = erase_overlay(it);
                continue;
            }
        }
        parts.push_back({&o, it->first});
        ++it;
    }
    if (parts.empty()) return false;
    Staging::Alloc al = staging_->alloc(size, 16);
    if (!al.data) return false;
    memcpy(al.data, reinterpret_cast<const void*>(address), size);  // lo de la CPU, y encima lo de la GPU
    barrier();
    for (auto [o, start] : parts) {
        const uint64_t from = std::max(address, start), to = std::min(address + size, start + o->size);
        VkBufferCopy region{o->offset + (from - start), al.offset + (from - address), to - from};
        vkCmdCopyBuffer(cmd_, o->buffer, al.buffer, 1, &region);
    }
    barrier();
    *buffer = al.buffer;
    *offset = al.offset;
    return true;
}

const Executor::Overlay* Executor::make_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset) {
    if (const Overlay* o = find_overlay(address, size, offset)) return o;
    if (size + 256 > kStagingChunk) return nullptr;
    // Un rango nuevo que solape otros los sustituye.
    for (auto it = overlays_.lower_bound(address); it != overlays_.end() && it->first < address + size;) it = erase_overlay(it);
    if (auto it = overlays_.lower_bound(address); it != overlays_.begin() && std::prev(it)->first + std::prev(it)->second.size > address)
        erase_overlay(std::prev(it));
    Staging::Alloc a = overlay_arena_.alloc(size + 256, 256);
    if (!a.data) return nullptr;
    overlay_used_ += size + 256;
    const VkDeviceSize skew = address & 255;
    memcpy(a.data + skew, reinterpret_cast<const void*>(address), size);
    Overlay& o = overlays_[address];
    o.buffer = a.buffer;
    o.offset = a.offset + skew;
    o.data = a.data + skew;
    o.size = size;
    filter_overlay(address, size, +1);
    o.cpu_hash = range_hash(reinterpret_cast<const uint8_t*>(address), size);
    o.checked = submit_index_;
    *offset = o.offset;
    return &o;
}

// Compute: mismo esquema que un draw, con un c0 del driver sintético. Lo que escribe en storage buffers
// es memoria del juego (vértices, argumentos de draws indirectos…): se envía en el acto y se copia de
// vuelta antes de seguir, para que lo que venga después lo lea ya escrito.
void Executor::dispatch(const cmd::Dispatch& v) {
    ProgramStage* cs = st_.compute;
    uint32_t x = v.x, y = v.y, z = v.z;
    if (v.indirect) {
        const auto* a = reinterpret_cast<const uint32_t*>(v.indirect);
        x = a[0], y = a[1], z = a[2];
    }
    dispatches_++;
    maybe_split();
    if (frame_ == trace_frame_) {  // DEPURACIÓN
        std::string ssbos;
        char b[96];
        for (int k = 0; k < kMaxStorage; k++)
            if (st_.bindings[static_cast<int>(Stage::Compute)].storage[k].size)
                snprintf(b, sizeof b, " s%d %#llx+%llu", k,
                         (unsigned long long)st_.bindings[static_cast<int>(Stage::Compute)].storage[k].address,
                         (unsigned long long)st_.bindings[static_cast<int>(Stage::Compute)].storage[k].size), ssbos += b;
        NVN_LOG("F dispatch %u %u %u%s |%s\n", x, y, z, cs && cs->module ? "" : " (sin shader)", ssbos.c_str());
        if (cs) dump_stage_code(*cs, "traza_cs" + std::to_string(dispatches_));
    }
    if (!cs || !cs->module || !x || !y || !z) return;
    end_render_pass();
    st_.rt_dirty = true;
    const int S = static_cast<int>(Stage::Compute);
    const StageBindings& sb = st_.bindings[S];

    struct Bound {
        const Resource* r;
        Texture* tex;
        TextureViewDesc view;
        Sampler* smp;
    };
    std::vector<Bound> textures;
    for (const Resource& r : cs->shader.resources) {
        if (r.kind != Resource::Texture && r.kind != Resource::Image) continue;
        Bound b{&r, nullptr, {}, nullptr};
        resolve_texture(Stage::Compute, r, b.tex, b.view, b.smp);
        if (b.tex) flush_clears(b.tex);  // su borrado diferido, antes de leerla o escribirla
        if (b.tex && ensure_image(*b.tex, cmd_) && !b.tex->gpu_written && b.tex->storage) {
            bool dirty = !b.tex->uploaded;
            if (!dirty && b.tex->checked_frame != frame_) {
                b.tex->checked_frame = frame_;
                dirty = content_hash(*b.tex) != b.tex->content_hash;
            }
            if (dirty) upload(*b.tex);
        }
        if (b.tex && r.kind == Resource::Image) b.tex->gpu_written = true;
        textures.push_back(b);
    }

    const ProgramStage* stages[kStages] = {};
    stages[S] = cs;
    Layout& l = layout(stages);
    VkPipeline p = compute_pipeline(*cs, l);
    if (!p) return;
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p);

    // c0 del driver: handles de textura y tabla de storage buffers (dirección real y tamaño; el shader
    // calcula cada acceso como dirección − base y el buffer enlazado empieza en esa dirección).
    uint8_t driver[kComputeDriverSize] = {};
    for (int k = 0; k < kMaxTextures && kTextureHandleBase + 8 * k + 8 <= kComputeDriverSize; k++)
        memcpy(driver + kTextureHandleBase + 8 * k, &sb.textures[k], 8);
    for (int k = 0; k < kMaxStorage && kComputeStorageBase + 16 * k + 16 <= kComputeDriverSize; k++) {
        uint32_t size = static_cast<uint32_t>(sb.storage[k].size);
        memcpy(driver + kComputeStorageBase + 16 * k, &sb.storage[k].address, 8);
        memcpy(driver + kComputeStorageBase + 16 * k + 8, &size, 4);
    }

    struct Writeback {
        uint8_t* staging;
        uint64_t address;
        size_t size;
    };
    std::vector<Writeback> writebacks;
    VkDescriptorSet set = descriptor_set(l.set);
    if (!set) return;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    buffers.reserve(cs->shader.resources.size());
    images.reserve(cs->shader.resources.size());
    const VkPhysicalDeviceLimits& limits = vk()->props.limits;
    const VkDeviceSize ubo_align = std::max<VkDeviceSize>(limits.minUniformBufferOffsetAlignment, 16);
    const VkDeviceSize ssbo_align = std::max<VkDeviceSize>(limits.minStorageBufferOffsetAlignment, 16);
    for (const Resource& r : cs->shader.resources) {
        if (r.kind != Resource::UniformBuffer && r.kind != Resource::StorageBuffer) continue;
        BufferRange range{};
        const void* src = nullptr;
        if (r.kind == Resource::UniformBuffer) {
            if (r.cbuf_index == kComputeDriverBank) {
                src = driver;
                range.size = sizeof driver;
            } else if (r.cbuf_index == kShaderConstantsBank) {
                range = {cs->constants, cs->constants_size};
            } else {
                int index = static_cast<int>(r.cbuf_index) - static_cast<int>(kUniformBankBase);
                if (index >= 0 && index < kMaxUniforms) range = sb.uniforms[index];
            }
        } else if (r.cbuf_index == kComputeDriverBank && r.cbuf_offset >= kComputeStorageBase) {
            int index = static_cast<int>((r.cbuf_offset - kComputeStorageBase) / 16);
            if (index < kMaxStorage) range = sb.storage[index];
            if (range.address && !r.is_written) note_read(5, range.address, range.size);
        }
        if (!src && range.address) src = reinterpret_cast<const void*>(range.address);
        VkDeviceSize size = std::min<VkDeviceSize>(range.size ? range.size : 256,
                                                   r.kind == Resource::UniformBuffer ? 65536 : kStagingChunk);
        // Lo que escribe el compute va a su overlay; lo que lee, del overlay si otro compute lo escribió.
        VkDeviceSize ov_offset = 0;
        const Overlay* ov = nullptr;
        if (src == reinterpret_cast<const void*>(range.address) && range.address && range.size) {
            ov = r.kind == Resource::StorageBuffer && r.is_written ? make_overlay(range.address, size, &ov_offset)
                                                                    : find_overlay(range.address, size, &ov_offset);
        }
        if (frame_ == trace_frame_ && r.kind == Resource::StorageBuffer) {  // DEPURACIÓN
            std::string vals;
            char b[24];
            if (ov && ov->data)
                for (size_t i = 0; i < std::min<size_t>(size / 4, 64); i++)
                    snprintf(b, sizeof b, " %x", reinterpret_cast<const uint32_t*>(ov->data + (ov_offset - ov->offset))[i]), vals += b;
            NVN_LOG("F compute SSBO %s %#llx+%llu: %s%s\n", r.is_written ? "escrito" : "leído",
                    static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(size),
                    ov ? "en la GPU:" : "de la memoria", vals.c_str());
        }
        if (ov) {
            buffers.push_back({ov->buffer, ov_offset, size});
        } else {
            Staging::Alloc a = staging_->alloc(size, r.kind == Resource::UniformBuffer ? ubo_align : ssbo_align);
            if (!a.data) return;
            if (src && range.size)
                memcpy(a.data, src, size);
            else
                memset(a.data, 0, size);
            if (r.kind == Resource::StorageBuffer && r.is_written && range.address && range.size)
                writebacks.push_back({a.data, range.address, static_cast<size_t>(size)});
            buffers.push_back({a.buffer, a.offset, size});
        }
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = r.binding;
        w.descriptorCount = 1;
        w.descriptorType = r.kind == Resource::UniformBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                             : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &buffers.back();
        writes.push_back(w);
    }
    for (const Bound& b : textures) {
        Texture* t = b.tex && b.tex->image ? b.tex : &dummy_;
        VkImageView iv = view(*t, t == b.tex ? b.view : TextureViewDesc{}, false);
        const bool image = b.r->kind == Resource::Image;
        images.push_back({image ? VK_NULL_HANDLE : sampler(b.smp), iv, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = b.r->binding;
        w.descriptorCount = 1;
        w.descriptorType = image ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &images.back();
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(vk()->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, l.pipeline, 0, 1, &set, 0, nullptr);
    barrier();  // lo que leyeron draws anteriores de un overlay que este compute va a pisar
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, l.pipeline, 0, 1, &set, 0, nullptr);
    prof_gpu_mark("compute");
    vkCmdDispatch(cmd_, x, y, z);
    barrier();
    if (!writebacks.empty()) {
        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // las escrituras, visibles para la CPU
        hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0,
                             nullptr, 0, nullptr);
        prof_writebacks_++;
        finish(true);  // espera a la GPU; los datos siguen en el staging hasta que se reutilice el contexto
        for (const Writeback& w : writebacks) memcpy(reinterpret_cast<void*>(w.address), w.staging, w.size);
        begin();
    }
}

bool Executor::blit(Texture& t, VkImage dst, VkFormat format, VkExtent2D extent, VkCommandBuffer cmd) {
    std::lock_guard lock(g_mutex);
    if (!ready_ && !init()) return false;
    if (!t.image) return false;
    // FSR 1: más nítido que el blit lineal, por ~1 ms (sin_fsr: el blit).
    if (fsr::record(cmd, view(t, TextureViewDesc{}, false), {static_cast<uint32_t>(t.desc.width), static_cast<uint32_t>(t.desc.height)},
                    dst, format, extent))
        return true;
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[1] = {t.desc.width, t.desc.height, 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    vkCmdBlitImage(cmd, t.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                   VK_FILTER_LINEAR);
    return true;
}

// DEPURACIÓN: copia una textura de color a RGBA8 en la CPU y la guarda como PPM en <datos>/volcados.
// DEPURACIÓN: profundidad de nivel 0, capa 0, a gris (D32F tal cual; D24: los 24 bits bajos).
void Executor::dump_depth(Texture& t, const std::string& name) {
    const uint32_t w = t.desc.width, h = t.desc.height;
    begin();
    flush_clears(&t);
    Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * 4, 16);
    if (!a.data) return;
    barrier();
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, a.buffer, 1, &r);
    uint8_t* src = a.data;
    finish(true);
    std::vector<uint8_t> rgb(size_t{w} * h * 3);
    double sum = 0, mn = 1e9, mx = -1e9;
    for (size_t k = 0; k < size_t{w} * h; k++) {
        float v;
        if (t.vk_format == VK_FORMAT_D32_SFLOAT || t.vk_format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
            memcpy(&v, src + k * 4, 4);
        } else {
            uint32_t u;
            memcpy(&u, src + k * 4, 4);
            v = (u & 0xFFFFFF) / 16777215.0f;
        }
        sum += v, mn = std::min<double>(mn, v), mx = std::max<double>(mx, v);
        rgb[k * 3] = rgb[k * 3 + 1] = rgb[k * 3 + 2] = static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255);
    }
    std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    if (FILE* f = fopen((dir + "/" + name + ".ppm").c_str(), "wb")) {
        fprintf(f, "P6 %u %u 255\n", w, h);
        fwrite(rgb.data(), 1, rgb.size(), f);
        fclose(f);
    }
    NVN_LOG("volcado %s: %ux%u profundidad media %.4f mín %.4f máx %.4f\n", name.c_str(), w, h, sum / (double(w) * h), mn, mx);
}

// DEPURACIÓN: los bits tal cual del mip 0 (para ver NaN/infinitos de un R11G11B10F: el volcado a RGBA8 los pierde).
void Executor::dump_raw(Texture& t, const std::string& name) {
    const uint32_t w = t.desc.width, h = t.desc.height;
    begin();
    flush_clears(&t);
    Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * 4, 16);
    if (!a.data) return;
    barrier();
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, a.buffer, 1, &r);
    uint8_t* src = a.data;
    finish(true);
    std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    if (FILE* f = fopen((dir + "/" + name + ".raw").c_str(), "wb")) {
        uint32_t hdr[3] = {w, h, static_cast<uint32_t>(t.vk_format)};
        fwrite(hdr, sizeof hdr, 1, f);
        fwrite(src, 1, size_t{w} * h * 4, f);
        fclose(f);
    }
}

void Executor::dump(Texture& t, const std::string& name, uint32_t layer, int forced_mip) {
    if (t.image && (t.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && t.desc.width > 0 && t.desc.height > 0) return dump_depth(t, name);
    if (t.image && t.vk_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && layer == 0 && forced_mip < 0) dump_raw(t, name);
    if (!t.image || (t.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || t.desc.width <= 0 || t.desc.height <= 0) return;
    VkContext* c = vk();
    // DEPURACIÓN: las texturas grandes se vuelcan desde un mip más pequeño (que quepa en el staging).
    uint32_t mip = 0;
    while (mip + 1 < static_cast<uint32_t>(std::max(t.desc.levels, 1)) && (uint64_t{uint32_t(t.desc.width) >> mip} * (uint32_t(t.desc.height) >> mip)) > 2048ull * 2048)
        mip++;
    if (forced_mip >= 0) mip = std::min<uint32_t>(forced_mip, std::max(t.desc.levels, 1) - 1);
    if (layer >= t.layers) layer = 0;
    const uint32_t w = std::max(uint32_t(t.desc.width) >> mip, 1u), h = std::max(uint32_t(t.desc.height) >> mip, 1u);
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {w, h, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage tmp;
    if (vkCreateImage(c->device, &ii, nullptr, &tmp) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, tmp, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory tmp_mem;
    vkAllocateMemory(c->device, &ai, nullptr, &tmp_mem);
    vkBindImageMemory(c->device, tmp, tmp_mem, 0);
    begin();
    flush_clears(&t);
    Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * 4, 16);
    if (!a.data) return;
    auto to = [&](VkImageLayout from, VkImageLayout layout) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout = from;
        b.newLayout = layout;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = tmp;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
    };
    to(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    {  // gris de control: si sobrevive, el blit no escribió
        VkClearColorValue gray{{0.5f, 0.5f, 0.5f, 1.0f}};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &gray, 1, &range);
        to(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
    const bool control = name == "control";
    VkImageBlit bl{};
    bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
    bl.srcOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
    bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.dstOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
    if (!control)
        vkCmdBlitImage(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl,
                       VK_FILTER_NEAREST);
    to(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd_, tmp, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a.buffer, 1, &r);
    std::vector<uint8_t> rgba(size_t{w} * h * 4);
    uint8_t* src = a.data;
    finish(true);  // espera a la GPU; los datos siguen en el staging hasta que se reutilice el contexto
    memcpy(rgba.data(), src, rgba.size());
    vkDestroyImage(c->device, tmp, nullptr);
    vkFreeMemory(c->device, tmp_mem, nullptr);

    // En la memoria interna: escribir megas en la SD (FUSE) a veces se bloquea segundos.
    std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    if (FILE* f = fopen((dir + "/" + name + ".ppm").c_str(), "wb")) {
        fprintf(f, "P6 %u %u 255\n", w, h);
        std::vector<uint8_t> rgb(size_t{w} * h * 3);
        for (size_t k = 0; k < size_t{w} * h; k++) memcpy(&rgb[k * 3], &rgba[k * 4], 3);
        fwrite(rgb.data(), 1, rgb.size(), f);
        fclose(f);
    }
    if (FILE* f = fopen((dir + "/" + name + "_a.pgm").c_str(), "wb")) {  // el alfa, aparte
        fprintf(f, "P5 %u %u 255\n", w, h);
        std::vector<uint8_t> alpha(size_t{w} * h);
        for (size_t k = 0; k < alpha.size(); k++) alpha[k] = rgba[k * 4 + 3];
        fwrite(alpha.data(), 1, alpha.size(), f);
        fclose(f);
    }
    NVN_LOG("volcado %s: %ux%u formato %s\n", name.c_str(), w, h, format_info(t.desc.format) ? format_info(t.desc.format)->name : "?");
}

// Cada 300 frames: ms por frame dentro de run() (CPU del executor + espera a la GPU), de espera a la
// GPU, envíos y draws por frame.
namespace {
// MB que KGSL tiene asignados a este proceso (-1 si el kernel no lo expone).
long kgsl_process_mb() {
    char path[64];
    snprintf(path, sizeof path, "/sys/class/kgsl/kgsl/proc/%d/kernel", getpid());
    long long bytes = -1;
    if (FILE* f = fopen(path, "r")) {
        if (fscanf(f, "%lld", &bytes) != 1) bytes = -1;
        fclose(f);
    }
    return bytes < 0 ? -1 : static_cast<long>(bytes >> 20);
}
}  // namespace

// Un salto grande de memoria de la GPU deja constancia en el acto, con lo que cuento yo al lado.
void Executor::watch_kgsl() {
    auto now = std::chrono::steady_clock::now();
    if (now - kgsl_checked_ < std::chrono::milliseconds(500)) return;
    kgsl_checked_ = now;
    long mb = kgsl_process_mb();
    if (mb < 0 || mb < kgsl_logged_mb_ + 256) {
        if (mb >= 0 && mb + 256 < kgsl_logged_mb_) kgsl_logged_mb_ = mb;  // bajó: el siguiente salto se mide desde aquí
        return;
    }
    size_t pools = 0;
    VkDeviceSize staging = 0;
    for (const InFlight& f : inflight_) pools += f.pools.size(), staging += f.staging.bytes();
    size_t blocks = 0, dedicated = 0;
    VkDeviceSize block_bytes = 0;
    gpu_memory_stats(&blocks, &dedicated, &block_bytes);
    NVN_LOG("memoria de la GPU: %ld MB (antes %ld); míos: texturas %llu MB (%zu reservas propias), staging %llu MB, caché de "
            "vértices %llu MB, overlays %llu MB, %zu pools de descriptores, %zu pipelines, %zu vistas\n",
            mb, kgsl_logged_mb_, (unsigned long long)(g_texture_gpu_bytes.load() >> 20), dedicated,
            (unsigned long long)(staging >> 20), (unsigned long long)(buffer_cache_.bytes() >> 20),
            (unsigned long long)(overlay_arena_.bytes() >> 20), pools, pipelines_.size(), views_.size());
    kgsl_logged_mb_ = mb;
}

namespace {
std::string pipeline_cache_path() {
    const char* data = getenv("NIER_DATA");
    return std::string(data ? data : "/sdcard/NieR") + "/pipelines.bin";
}
}  // namespace

// La cabecera se comprueba antes de dársela al driver (un fichero de otro driver o roto se descarta).
void Executor::load_pipeline_cache() {
    VkContext* c = vk();
    std::vector<uint8_t> data;
    if (FILE* f = fopen(pipeline_cache_path().c_str(), "rb")) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size > 32 && size < (128l << 20)) {
            data.resize(static_cast<size_t>(size));
            if (fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        }
        fclose(f);
    }
    if (!data.empty()) {
        uint32_t header[4];
        memcpy(header, data.data(), sizeof header);
        const bool ok = header[0] >= 32 && header[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                        header[2] == c->props.vendorID && header[3] == c->props.deviceID &&
                        memcmp(data.data() + 16, c->props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        if (!ok) data.clear();
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(c->device, &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        if (vkCreatePipelineCache(c->device, &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) pipeline_cache_ = VK_NULL_HANDLE;
    }
    NVN_LOG("caché de pipelines: %zu KB leídos de la SD\n", data.size() >> 10);
}

// En otro hilo (escribir en la SD es lento) y solo cuando lleva un rato sin crearse ninguno. La caché la
// sincroniza el driver: se puede leer mientras el ejecutor crea pipelines.
void Executor::save_pipeline_cache() {
    if (!pipeline_cache_ || pipelines_created_ == pipelines_saved_ || frame_ < last_pipeline_frame_ + 60) return;
    if (cache_saving_.exchange(true)) return;
    pipelines_saved_ = pipelines_created_;
    const VkDevice device = vk()->device;
    const VkPipelineCache cache = pipeline_cache_;
    std::thread([this, device, cache] {
        size_t size = 0;
        std::vector<uint8_t> data;
        if (vkGetPipelineCacheData(device, cache, &size, nullptr) == VK_SUCCESS && size) {
            data.resize(size);
            if (vkGetPipelineCacheData(device, cache, &size, data.data()) != VK_SUCCESS) data.clear();
            data.resize(std::min(size, data.size()));
        }
        if (!data.empty()) {
            const std::string path = pipeline_cache_path(), tmp = path + ".tmp";
            if (FILE* f = fopen(tmp.c_str(), "wb")) {
                const bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
                fflush(f);
                fsync(fileno(f));
                fclose(f);
                if (ok) rename(tmp.c_str(), path.c_str());
            }
        }
        cache_saving_ = false;
    }).detach();
}

void Executor::profile_frame() {
    if (frame_ % 300) return;
    std::lock_guard lock(g_mutex);
    // Sin <datos>/registro, solo lo que hace falta (guardar la caché de pipelines): el perfil lee /proc, recorre la
    // memoria y escribe una docena de líneas en la SD cada 10 s.
    if (!NVN_FLAG("registro")) {
        save_pipeline_cache();
        return;
    }
    NVN_LOG("resolución dinámica: GPU del frame %.1f ms (%s), a LSFG se le reservan %.1f ms; escena %dx%d (ancho mín %d)\n",
            frame_gpu_ns_.load(std::memory_order_relaxed) / 1e6, gpu_stamps_ ? "timestamps de la GPU" : "reloj de la CPU",
            stamp_extra_ / 0.6144 / 1e6, scene_vp_w_, scene_vp_h_, scene_vp_min_w_ == 1 << 30 ? 0 : scene_vp_min_w_);
    scene_vp_w_ = scene_vp_h_ = 0;
    scene_vp_min_w_ = 1 << 30;
    save_pipeline_cache();
    size_t blocks = 0, dedicated = 0;
    VkDeviceSize block_bytes = 0;
    gpu_memory_stats(&blocks, &dedicated, &block_bytes);
    {  // RAM: la "memoria de GPU" del Adreno es RAM del sistema
        VkDeviceSize staging = 0;
        for (const InFlight& f : inflight_) staging += f.staging.bytes();
        long pages = 0, resident = 0;
        if (FILE* f = fopen("/proc/self/statm", "r")) {
            if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
            fclose(f);
        }
        long available_kb = 0;
        if (FILE* f = fopen("/proc/meminfo", "r")) {
            char line[128];
            while (fgets(line, sizeof line, f))
                if (sscanf(line, "MemAvailable: %ld kB", &available_kb) == 1) break;
            fclose(f);
        }
        NVN_LOG("perfil RAM: proceso %ld MB, GPU %ld MB, disponible en el sistema %ld MB; míos: staging %llu MB, caché de vértices %llu MB, "
                "overlays %llu MB, texturas %llu MB en bloques\n",
                resident * 4096 / (1 << 20), kgsl_process_mb(), available_kb / 1024, (unsigned long long)(staging >> 20),
                (unsigned long long)(buffer_cache_.bytes() >> 20), (unsigned long long)(overlay_arena_.bytes() >> 20),
                (unsigned long long)(block_bytes >> 20));
    }
    NVN_LOG("perfil memoria: %llu texturas en la GPU (%llu MB; %zu bloques de %llu MB y %zu reservas propias), %zu vistas, "
            "%zu framebuffers, %zu overlays\n",
            (unsigned long long)g_texture_gpu_images.load(), (unsigned long long)(g_texture_gpu_bytes.load() >> 20), blocks,
            (unsigned long long)(block_bytes >> 20), dedicated, views_.size(), framebuffers_.size(), overlays_.size());
    NVN_LOG("perfil: %.1f ms/frame en el executor (%.1f esperando a la GPU), %.1f envíos, %.0f draws y %.1f render passes por frame\n",
            prof_run_us_ / 300.0 / 1000.0, prof_gpu_wait_us_ / 300.0 / 1000.0, prof_submits_ / 300.0,
            (draws_ - prof_draws_) / 300.0, prof_render_passes_ / 300.0);
    NVN_LOG("perfil compute: %.1f dispatches y %.1f esperas por write-back por frame; lecturas de lo escrito: vértices %llu, "
            "índices %llu, ubo %llu, ssbo %llu, indirecto %llu, ssbo de compute %llu; %zu rangos\n",
            (dispatches_ - prof_dispatches_) / 300.0, prof_writebacks_ / 300.0, (unsigned long long)prof_reads_[0],
            (unsigned long long)prof_reads_[1], (unsigned long long)prof_reads_[2], (unsigned long long)prof_reads_[3],
            (unsigned long long)prof_reads_[4], (unsigned long long)prof_reads_[5], gpu_written_ranges_.size());
    NVN_LOG("perfil compute: %llu lecturas solo parcialmente dentro; texturas subidas: %llu MB\n", (unsigned long long)prof_partial_,
            (unsigned long long)(prof_texture_bytes_ >> 20));
    prof_partial_ = 0;
    memset(prof_reads_, 0, sizeof prof_reads_);
    prof_dispatches_ = dispatches_;
    prof_render_passes_ = prof_writebacks_ = 0;
    // Los tramos por draw se midieron en uno de cada kProfileEvery frames: ms por frame medido.
    const double timed = std::max<uint64_t>(prof_timed_frames_, 1) * 1e6;
    prof_timed_frames_ = 0;
    NVN_LOG("perfil descriptores (ms/frame): buffers %.1f, texturas %.1f, actualizar %.1f, reservar %.1f; dispatch %.1f; draws en total %.1f\n",
            prof_sub_us_[0] / timed, prof_sub_us_[1] / timed, prof_sub_us_[2] / timed, prof_sub_us_[3] / timed,
            prof_dispatch_us_ / timed, prof_draw_total_us_ / timed);
    NVN_LOG("perfil: %.2f MB de uniform/storage buffers copiados por frame\n", prof_copied_bytes_ / 300.0 / 1048576.0);
    prof_copied_bytes_ = 0;
    memset(prof_sub_us_, 0, sizeof prof_sub_us_);
    prof_dispatch_us_ = prof_draw_total_us_ = 0;
    NVN_LOG("perfil draw (ms/frame): texturas %.1f, pipeline %.1f, descriptores %.1f, estado %.1f, vértices %.1f, dibujo %.1f\n",
            prof_section_us_[0] / timed, prof_section_us_[1] / timed, prof_section_us_[2] / timed,
            prof_section_us_[3] / timed, prof_section_us_[4] / timed, prof_section_us_[5] / timed);
    memset(prof_section_us_, 0, sizeof prof_section_us_);
    NVN_LOG("perfil tiler (por frame): %.1f borrados como loadOp CLEAR y %.1f aparte, %.1f adjuntos sin guardar, %.1f pases "
            "con área recortada, %.1f barreras aplazadas (%.1f cortaron luego), %.1f re-enlaces sin cortar\n",
            prof_clear_loads_ / 300.0, prof_clear_flushes_ / 300.0, prof_store_none_ / 300.0, prof_area_cut_ / 300.0,
            prof_barriers_deferred_ / 300.0, prof_barrier_splits_ / 300.0, prof_rt_kept_ / 300.0);
    prof_clear_loads_ = prof_clear_flushes_ = prof_store_none_ = prof_area_cut_ = 0;
    prof_barriers_deferred_ = prof_barrier_splits_ = prof_rt_kept_ = 0;
    prof_run_us_ = prof_gpu_wait_us_ = prof_submits_ = 0;
    prof_draws_ = draws_;
}

void Executor::dump_frame(Texture* presented) {
    std::lock_guard lock(g_mutex);
    if (frame_ == trace_frame_ && ready_) {
        std::string f = std::to_string(frame_);
        if (presented) dump(*presented, "f" + f + "_ventana");
        finish(true);  // DEPURACIÓN: la GPU ha escrito ya las listas compactadas
        for (size_t k = 0; k < traced_gpu_draws_.size() && k < 120; k++) {
            const TracedGpuDraw& g = traced_gpu_draws_[k];
            const auto* args = reinterpret_cast<const uint32_t*>(g.args);
            const uint32_t count = args[0];
            auto idx = [&](const uint8_t* p, uint32_t i) {
                return g.type == 2 ? reinterpret_cast<const uint32_t*>(p)[i] : reinterpret_cast<const uint16_t*>(p)[i];
            };
            uint32_t mx = 0, outside = 0;
            auto it = prepass_args_.find(g.args_address);
            uint32_t orig_max = 0;
            if (it != prepass_args_.end())
                for (uint32_t i = 0; i < it->second.count; i++)
                    orig_max = std::max(orig_max, idx(reinterpret_cast<const uint8_t*>(it->second.indices), i));
            for (uint32_t i = 0; i < std::min<uint32_t>(count, 1 << 20); i++) {
                const uint32_t v = idx(g.indices, i);
                mx = std::max(mx, v);
                if (it != prepass_args_.end() && v > orig_max) outside++;
            }
            std::string first;
            for (uint32_t i = 0; i < std::min<uint32_t>(count, 9); i++) first += " " + std::to_string(idx(g.indices, i));
            NVN_LOG("F compactado %zu: %u índices (de %u originales, máx original %u), máx %u, fuera de rango %u:%s\n", k, count,
                    it != prepass_args_.end() ? it->second.count : 0, orig_max, mx, outside, first.c_str());
        }
        traced_gpu_draws_.clear();
        prepass_args_.clear();
        // DEPURACIÓN: solo la imagen final y el log, salvo con <datos>/traza_completa
        const char* light = NVN_FLAG("traza_completa") ? nullptr : getenv("NIER_TRAZA_LIGERA");
        for (size_t k = 0; k < frame_targets_.size() && k < 32 && (full_trace_ || !(light && *light)); k++)
            dump(*frame_targets_[k], "f" + f + "_rt" + std::to_string(k));
    }
    if (frame_ % 30 == 0) {  // ¿piden una traza?
        const char* data = getenv("NIER_DATA");
        std::string flag = std::string(data ? data : "/sdcard/NieR") + "/trazar";
        const bool lequal = access((std::string(data ? data : "/sdcard/NieR") + "/profundidad_lequal").c_str(), F_OK) == 0;
        if (lequal != depth_lequal_) NVN_LOG("depth EQUAL como LEQUAL: %s\n", lequal ? "sí" : "no");
        depth_lequal_ = lequal;
        if (access(flag.c_str(), F_OK) == 0) {
            // "completa" dentro del archivo: también vuelca los render targets del frame (si no, solo log e imagen).
            char mode[16] = {};
            if (FILE* f = fopen(flag.c_str(), "r")) {
                fread(mode, 1, sizeof mode - 1, f);
                fclose(f);
            }
            full_trace_ = strstr(mode, "completa") != nullptr;
            unlink(flag.c_str());
            trace_frame_ = frame_ + 1;
            traced_draws_ = 0;
            memset(trace_counts_, 0, sizeof trace_counts_);
            NVN_LOG("traza del frame %llu\n", static_cast<unsigned long long>(trace_frame_));
        }
    }
    if (frame_ == trace_frame_ + 1 && watch_addr_ && nnsdk::host().watch_writes && getenv("NIER_VIGILAR")) {
        nnsdk::host().watch_writes(watch_addr_);
        watch_addr_ = 0;
    }
    if (frame_ == trace_frame_) {
        std::string counts;
        for (size_t k = 0; k < std::size(trace_counts_); k++)
            if (trace_counts_[k]) counts += " " + std::to_string(k) + ":" + std::to_string(trace_counts_[k]);
        NVN_LOG("F comandos del frame (índice del variant:cuenta):%s; draws con render desactivado %llu\n", counts.c_str(),
                static_cast<unsigned long long>(prof_disabled_draws_));
    }
    frame_targets_.clear();
    frame_uploads_.clear();
    if (frame_ % 300 == 0)  // los args de frames viejos ya no sirven
        for (auto it = culled_draws_.begin(); it != culled_draws_.end();)
            it = frame_ - it->second.frame > 2 ? culled_draws_.erase(it) : std::next(it);
}

Executor g_executor;

}  // namespace

namespace {

// --- Envío asíncrono ---
// nvnQueueSubmitCommands solo encola: un hilo reproduce los comandos en orden mientras el juego graba el frame
// siguiente (antes el hilo del juego que envía pasaba ~20 ms por frame dentro de la llamada, y MainDraw
// esperaba). Cada envío del juego tiene un ticket: un NVNsync recuerda uno, y esperarlo es esperar a que se
// haya reproducido y a que la GPU haya terminado su último envío de Vulkan. Presentar también pasa por la cola
// (en orden), con como mucho un frame por delante. DEPURACIÓN: con <datos>/sincrono, como antes (sin hilo).
class SubmitQueue {
public:
    uint64_t submit(const std::vector<const CommandList*>& lists) {
        std::unique_lock lock(m_);
        const uint64_t ticket = ++queued_;
        for (const CommandList* list : lists)  // los NVNsync de dentro de las listas: este envío
            for (const Command& c : list->commands)
                if (const auto* f = std::get_if<cmd::FenceSync>(&c)) sync_signal_at(f->sync, ticket);
        if (!async()) {
            drain_locked(lock);
            lock.unlock();
            run(lists, ticket);
            return ticket;
        }
        start_locked();
        jobs_.push_back({lists, nullptr, 0, ticket});
        work_cv_.notify_one();
        return ticket;
    }
    void present(Window& w, int index) {
        std::unique_lock lock(m_);
        if (!async()) {
            drain_locked(lock);
            lock.unlock();
            nvn::present(w, index);
            return;
        }
        start_locked();
        done_cv_.wait(lock, [&] { return presents_pending_ < 2; });  // dos frames por delante como mucho
        presents_pending_++;
        jobs_.push_back({{}, &w, index, 0});
        work_cv_.notify_one();
    }
    uint64_t last_ticket() {
        std::lock_guard lock(m_);
        return queued_;
    }
    // true si el envío ya se reprodujo y la GPU lo terminó (o termina antes del plazo).
    bool wait(uint64_t ticket, uint64_t timeout_ns) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(std::min<uint64_t>(timeout_ns, 1ull << 62));
        uint64_t gpu = 0;
        {
            std::unique_lock lock(m_);
            if (!done_cv_.wait_until(lock, deadline, [&] { return done_ >= ticket; })) return false;
            auto it = gpu_submit_.lower_bound(ticket);  // los viejos se olvidan: uno posterior también vale
            gpu = it != gpu_submit_.end() ? it->second : gpu_submit_.empty() ? 0 : gpu_submit_.rbegin()->second;
        }
        if (!gpu) return true;
        const auto now = std::chrono::steady_clock::now();
        const uint64_t left = now >= deadline ? 0 : std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
        return g_executor.wait_submit(gpu, left);
    }
    void drain() {
        std::unique_lock lock(m_);
        drain_locked(lock);
    }
    // Una lista que el juego recicla puede estar aún en la cola: se borra cuando se haya reproducido.
    void release(CommandList* list) {
        std::lock_guard lock(m_);
        if (done_ >= queued_ && !busy_) {
            recycle_command_list(list);
            return;
        }
        garbage_.push_back({queued_, list});
    }

private:
    struct Job {
        std::vector<const CommandList*> lists;
        Window* window;
        int index;
        uint64_t ticket;
    };
    bool async() {
        if (++checks_ % 64 == 1) {  // mirar el fichero de vez en cuando, no en cada envío
            const char* data = getenv("NIER_DATA");
            sync_mode_ = access((std::string(data ? data : ".") + "/sincrono").c_str(), F_OK) == 0;
        }
        return !sync_mode_;
    }
    void start_locked() {
        if (started_) return;
        started_ = true;
        std::thread([this] { loop(); }).detach();  // vive lo que el proceso: sin join (al salir no hay terminate)
    }
    void drain_locked(std::unique_lock<std::mutex>& lock) {
        done_cv_.wait(lock, [&] { return jobs_.empty() && !busy_; });
    }
    void run(const std::vector<const CommandList*>& lists, uint64_t ticket) {
        g_executor.run(lists);
        const uint64_t gpu = g_executor.last_submit();
        std::lock_guard lock(m_);
        gpu_submit_[ticket] = gpu;
        while (gpu_submit_.size() > 256) gpu_submit_.erase(gpu_submit_.begin());
        done_ = std::max(done_, ticket);
        collect_garbage_locked();
        done_cv_.notify_all();
    }
    void collect_garbage_locked() {
        while (!garbage_.empty() && garbage_.front().first <= done_) {
            recycle_command_list(garbage_.front().second);  // al pool de cmdbuf.cpp: no se vacía ni libera aquí
            garbage_.pop_front();
        }
    }
    void loop() {
        pthread_setname_np(pthread_self(), "NvnSubmit");
        pin_critical_thread();  // núcleos grandes (os.cpp)
        std::unique_lock lock(m_);
        for (;;) {
            work_cv_.wait(lock, [&] { return !jobs_.empty(); });
            Job job = std::move(jobs_.front());
            jobs_.pop_front();
            busy_ = true;
            lock.unlock();
            if (job.window) {
                nvn::present(*job.window, job.index);
                lock.lock();
                presents_pending_--;
            } else {
                run(job.lists, job.ticket);
                lock.lock();
            }
            busy_ = false;
            collect_garbage_locked();
            done_cv_.notify_all();
        }
    }
    std::mutex m_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<Job> jobs_;
    std::deque<std::pair<uint64_t, CommandList*>> garbage_;  // (último ticket encolado al soltarla, lista)
    std::map<uint64_t, uint64_t> gpu_submit_;                 // ticket -> envío de Vulkan
    uint64_t queued_ = 0, done_ = 0, checks_ = 0;
    int presents_pending_ = 0;
    bool busy_ = false, sync_mode_ = false, started_ = false;
};
SubmitQueue g_submit_queue;

}  // namespace

void execute(const std::vector<const CommandList*>& lists) { g_submit_queue.submit(lists); }
void queue_present(Window& w, int index) { g_submit_queue.present(w, index); }
void release_command_list(CommandList* list) { g_submit_queue.release(list); }

void next_frame() {
    g_executor.frame_++;
    g_executor.profile_frame();
}

void debug_dump_frame(Texture* presented) { g_executor.dump_frame(presented); }

uint64_t last_submit() { return g_submit_queue.last_ticket(); }
std::recursive_mutex& executor_mutex() { return g_mutex; }
void retire_texture_gpu(Texture& t) { g_executor.retire_texture(t); }
void retire_sampler(VkSampler sampler) { g_executor.retire_sampler(sampler); }
bool wait_submit(uint64_t submit, uint64_t timeout_ns) { return g_submit_queue.wait(submit, timeout_ns); }
void wait_idle() {
    g_submit_queue.drain();
    g_executor.wait_idle();
}

bool blit_to_swapchain(Texture& texture, VkImage image, VkFormat format, VkExtent2D extent, VkCommandBuffer cmd) {
    return g_executor.blit(texture, image, format, extent, cmd);
}

std::mutex& vk_queue_mutex() { return g_queue_mutex; }

}  // namespace nvn
