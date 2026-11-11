#pragma once
#include <atomic>
// Objetos NVN inicializados que comparten varios archivos (pools, texturas, samplers, programas,
// ventana). Los objetos del juego son opacos: se asocian por dirección a estos.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nnsdk/nvn/gpu_memory.h"
#include "nnsdk/nvn/layout.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/vk.h"
#include "nvn_shader.h"

namespace nvn {

struct MemoryPool {
    uint8_t* storage = nullptr;  // memoria del juego; también es la "dirección de GPU"
    size_t size = 0;
    int flags = 0;
};

// Estado de un NVNtextureBuilder; vive dentro del builder del juego (≤ 96 bytes).
struct TextureDesc {
    void* device;
    int target, flags, format;
    int width, height, depth, levels, samples;
    int stride;
    void* pool;
    int64_t offset;
};
static_assert(sizeof(TextureDesc) <= 96);

constexpr int kTextureFlagDisplay = 0x1, kTextureFlagImage = 0x4, kTextureFlagLinear = 0x10;
constexpr int kTarget1D = 0, kTarget2D = 1, kTarget3D = 2, kTarget1DArray = 3, kTarget2DArray = 4,
              kTarget2DMultisample = 5, kTarget2DMultisampleArray = 6, kTargetRectangle = 7, kTargetCube = 8,
              kTargetCubeArray = 9, kTargetBuffer = 10;

// NVNtextureView: vive dentro del struct del juego (≤ 40 bytes). flags dice qué campos valen.
struct TextureViewDesc {
    uint16_t flags;
    uint8_t depth_stencil_mode, pad;
    int16_t base_level, levels;
    int32_t base_layer, layers;
    int32_t format, target;
    uint8_t swizzle[4];
};
static_assert(sizeof(TextureViewDesc) <= 40);
constexpr uint16_t kViewLevels = 1, kViewLayers = 2, kViewFormat = 4, kViewTarget = 8, kViewDepthStencil = 16,
                   kViewSwizzle = 32;

struct Texture {
    // Único en toda la ejecución (nunca se reutiliza, al contrario que la dirección del objeto): clave de
    // las cachés de vistas. Cambia al reinicializar o finalizar la textura.
    uint64_t serial = 0;
    TextureDesc desc{};
    TextureLayout layout;
    uint8_t* storage = nullptr;  // memoria del juego donde están sus texels (si los sube la CPU)
    // Vulkan, creado cuando hace falta:
    VkImage image = VK_NULL_HANDLE;
    GpuAllocation memory;  // dentro de un bloque compartido o propia (gpu_memory.h)
    uint64_t gpu_bytes = 0;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    uint32_t layers = 1;
    bool vk_failed = false;    // no se pudo crear en Vulkan (formato sin soporte…): no reintentar
    bool mutable_format = true;  // imagen con MUTABLE_FORMAT: admite vistas con otro formato (ejecutor)
    uint64_t retry_frame = 0;  // sin memoria al crearla: no se reintenta antes de este frame
    bool gpu_written = false;  // ha sido render target o destino de una copia: no se sube de la CPU
    bool uploaded = false;
    uint64_t content_hash = 0;   // de su memoria al subirla: si cambia, el juego la ha reescrito
    uint64_t checked_frame = ~uint64_t{0};
    float clear_color[4] = {};  // último ClearColor (para presentar antes de tener render real)
    bool cleared = false;
    // Ejecutor: la última vista muestreada (válida si view_memo_serial == serial).
    VkImageView view_memo = VK_NULL_HANDLE;
    uint64_t view_memo_serial = 0;
    TextureViewDesc view_memo_desc{};
};

// Libera la imagen de Vulkan de una textura (texture.cpp); el objeto sigue siendo válido.
void release_texture_gpu(Texture& t);
uint64_t next_texture_serial();
// Memoria de GPU de las texturas vivas (DEPURACIÓN/perfil).
extern std::atomic<uint64_t> g_texture_gpu_bytes, g_texture_gpu_images;

struct SamplerDesc {  // NVNsamplerBuilder (≤ 96 bytes)
    void* device;
    uint8_t min_filter, mag_filter, wrap[3], compare_mode, compare_func, reduction;
    float max_anisotropy, lod_min, lod_max, lod_bias;
    float border[4];
};
static_assert(sizeof(SamplerDesc) <= 96);

struct Sampler {
    SamplerDesc desc{};
    VkSampler vk = VK_NULL_HANDLE;
};

struct TexturePoolEntry {
    Texture* texture = nullptr;
    TextureViewDesc view{};
};

struct DescriptorPool {
    void* memory_pool = nullptr;
    int64_t offset = 0;
    int count = 0;
    std::vector<TexturePoolEntry> textures;  // pool de texturas: id -> textura + vista
    std::vector<Sampler*> samplers;          // pool de samplers: id -> sampler
};

struct ProgramStage {
    // Geometry shader de la compactación de NieR (pre-pase de profundidad): escribe la lista de triángulos visibles y sus
    // argumentos indirectos, que ya no usamos (el pase de color dibuja la malla entera, ver culled_draws_).
    bool compaction_gs = false;
    uint64_t code = 0;  // dirección del código (NVN solo lo lee al dibujar: se traduce en el executor)
    const void* control = nullptr;  // sección de control (memoria de CPU del juego)
    // Constantes del propio shader (su c1): el compilador las deja tras el código, en
    // code + control[0x700] hasta code + control[0x704].
    uint64_t constants = 0;
    uint32_t constants_size = 0;
    bool tried = false;
    nvnshader::Translated shader;
    VkShaderModule module = VK_NULL_HANDLE;
    // Geometry: módulos por (genéricos que escribe el vertex shader, topología de entrada).
    std::unordered_map<uint64_t, VkShaderModule> variants;
};

struct Program {
    std::vector<ProgramStage> stages;
    uint64_t generation = 0;
};

struct Window {
    void* native_window = nullptr;
    int present_interval = 1;
    std::vector<void*> textures;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;  // el del swapchain (el HUD lo necesita)
    std::vector<VkImage> images;
    int next_texture = 0;
    uint64_t presented = 0;
};

extern Objects<MemoryPool> g_memory_pools;
extern Objects<Texture> g_textures;
extern Objects<Window> g_windows;
extern Objects<DescriptorPool> g_texture_pools, g_sampler_pools;

Program* find_program(const void* program);
// Traduce la etapa la primera vez que se usa (program.cpp); false si no se pudo.
bool ensure_translated(ProgramStage& stage);
// Geometry: el módulo para un vertex shader que escribe `previous_stores` y una topología de entrada de
// Eden; VK_NULL_HANDLE si no se puede traducir.
VkShaderModule geometry_variant(ProgramStage& stage, uint64_t previous_stores, int input_topology);
// DEPURACIÓN: guarda el código de la etapa en <datos>/shaders/<name>.code.
void dump_stage_code(const ProgramStage& stage, const std::string& name);
Sampler* find_sampler(const void* sampler);

// Handles de NVN (nuestro formato): separados = id; combinado = sampler << 32 | textura.
inline uint32_t handle_texture_id(uint64_t h) { return static_cast<uint32_t>(h & 0xFFFFFFFF); }
inline uint32_t handle_sampler_id(uint64_t h) { return static_cast<uint32_t>(h >> 32); }

// Presenta en pantalla la textura index de la ventana (window.cpp).
void present(Window& window, int index);

}  // namespace nvn
