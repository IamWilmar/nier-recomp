#pragma once
// Disposición de texturas en memoria de la GPU Tegra X1 (la que asume el juego al escribir sus
// texturas ya "swizzleadas", tal como vienen en sus archivos): block-linear con GOBs de 64 bytes x 8
// filas y bloques de 1..32 GOBs de alto, o lineal con stride (flag LINEAR).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nvn {

struct LevelLayout {
    size_t offset;               // desde el inicio de la capa
    size_t size;
    uint32_t width, height, depth;          // en texels
    uint32_t width_blocks, height_blocks;   // en bloques del formato (4x4 en BCn, 6x6 en ASTC 6x6…)
    uint32_t block_height_gobs;             // alto del bloque block-linear, en GOBs
    uint32_t block_depth;
};

struct TextureLayout {
    bool linear = false;
    uint32_t stride = 0;  // solo lineal
    uint32_t bytes_per_block = 4, block_w = 1, block_h = 1;
    std::vector<LevelLayout> levels;
    size_t layer_size = 0;  // tamaño de una capa (todas sus mips), alineado al bloque
    uint32_t layers = 1;
    size_t total_size() const { return layer_size * layers; }
};

TextureLayout compute_layout(uint32_t width, uint32_t height, uint32_t depth, uint32_t layers, uint32_t levels,
                             uint32_t bytes_per_block, uint32_t block_w, uint32_t block_h, bool linear, uint32_t stride,
                             bool is_3d);

// Copia un nivel de una capa de block-linear (src) a lineal compacto (dst, filas de width_blocks).
void deswizzle_level(const LevelLayout& level, uint32_t bytes_per_block, const uint8_t* src, uint8_t* dst);

}  // namespace nvn
