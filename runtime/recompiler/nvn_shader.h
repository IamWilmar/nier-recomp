// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Traducción de una etapa NVN (código en memoria: cabecera NVN de 0x30 bytes, SPH de Maxwell e
// instrucciones) a SPIR-V con el recompilador de Eden. Lo usan libnnsdk.so y tools/shadertool.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nvnshader {

enum class Stage { Vertex, TessControl, TessEval, Geometry, Fragment, Compute };

// Un recurso del shader y su binding en el set 0 (bindings unificados, en el orden de Eden).
struct Resource {
    enum Kind { UniformBuffer, StorageBuffer, TextureBuffer, ImageBuffer, Texture, Image } kind;
    uint32_t binding;
    uint32_t cbuf_index;   // UBO: banco Maxwell (c0..c17). Resto: banco donde está el handle
    uint32_t cbuf_offset;  // offset del handle (texturas/imágenes/SSBO)
    uint32_t texture_type;  // Shader::TextureType para texturas e imágenes
    bool is_depth = false;
    bool is_written = false;
    // Texturas con handle partido (textura | sampler leídos de dos sitios y combinados con OR).
    bool has_secondary = false;
    uint32_t secondary_cbuf_index = 0, secondary_cbuf_offset = 0;
    uint32_t shift_left = 0, secondary_shift_left = 0;
};

struct Translated {
    Stage stage = Stage::Vertex;
    std::vector<uint32_t> spirv;
    std::vector<Resource> resources;
    uint32_t constant_buffer_mask = 0;
    uint32_t constant_buffer_used[18] = {};  // bytes leídos de cada banco (0 si no se usa)
    uint64_t input_generics = 0;   // atributos genéricos que lee (vertex: entradas)
    uint64_t output_generics = 0;  // atributos genéricos que escribe (para la etapa siguiente)
    uint32_t color_outputs = 0;    // fragment: salidas de color escritas (bit por RT)
    bool has_brx = false;          // tiene un switch (BRX): Eden leyó su tabla de c1
    std::string error;             // vacío si salió bien
};

struct Options {
    uint32_t spirv_version = 0x00010300;  // Vulkan 1.1
    uint32_t texture_bound_buffer = 2;  // TEX con índice inmediato: handle en c2[índice·4] (tabla de NVN)
    bool convert_depth_mode = false;
    std::string* ir = nullptr;  // si no es nulo, recibe el IR de Eden en texto (depuración)
    // Compute (de la sección de control del shader): inicio del código, workgroup, memoria compartida.
    uint32_t compute_code_offset = 0x100;
    uint32_t workgroup[3] = {1, 1, 1};
    uint32_t shared_memory = 0, local_memory = 0;
    // Geometry: genéricos que escribe la etapa anterior (solo esos se declaran como entrada: el Adreno
    // admite 128 componentes y declarar los 32 + la posición se pasa) y topología de entrada
    // (InputTopology de Eden: puntos 0, líneas 1, líneas con adyacencia 2, triángulos 3, con adyacencia 4).
    uint64_t previous_stores = ~0ull;
    int input_topology = 0;
    // Subgrupos del dispositivo (VkPhysicalDeviceSubgroupProperties): tamaño, etapas (bits de
    // VkShaderStageFlagBits) y operaciones (VkSubgroupFeatureFlagBits). Sin nada, no se usan.
    uint32_t subgroup_size = 32, subgroup_stages = 0, subgroup_operations = 0;  // NVN NEAR_IS_MINUS_W: z de [-w,w] a [0,w] (Vulkan)
    // c1: las constantes que el compilador dejó tras el código (sección de control 0x700..0x704). Son parte
    // del programa, así que se conocen al traducir: Eden lee de ahí las tablas de salto de BRX/JMX
    // (un switch del shader). Sin ellas todas las entradas valían 0 y los casos no se traducían.
    const uint8_t* constants = nullptr;
    uint32_t constants_size = 0;
};

constexpr uint32_t kNvnHeaderSize = 0x30;
constexpr uint32_t kNvnCodeMagic = 0x12345678;         // gráficos: cabecera + SPH en 0x30, código en 0x80
constexpr uint32_t kNvnComputeCodeMagic = 0x12345679;  // compute: sin SPH, código donde diga el control

// code apunta a la cabecera NVN; max_size acota las lecturas (hasta el final del memory pool).
bool translate(const uint8_t* code, size_t max_size, const Options& options, Translated& out);

const char* stage_name(Stage s);

// Cada etapa numera sus recursos desde su propia base (todas comparten el set 0 del pipeline).
constexpr uint32_t kBindingsPerStage = 64;
uint32_t binding_base(Stage s);

}  // namespace nvnshader
