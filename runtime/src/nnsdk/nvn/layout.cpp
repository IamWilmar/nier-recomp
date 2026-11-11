#include "nnsdk/nvn/layout.h"

#include <algorithm>
#include <cstring>

namespace nvn {

namespace {

constexpr uint32_t kGobWidth = 64, kGobHeight = 8, kGobSize = kGobWidth * kGobHeight;

uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
size_t align(size_t v, size_t a) { return (v + a - 1) / a * a; }

uint32_t pow2_ceil(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Alto del bloque del nivel 0: lo justo para cubrir el alto en GOBs, hasta 16; las mips lo reducen
// a la mitad mientras el nivel quepa en medio bloque.
uint32_t base_block_height(uint32_t height_blocks) {
    return std::min<uint32_t>(16, pow2_ceil(div_ceil(height_blocks, kGobHeight)));
}

uint32_t level_block_height(uint32_t base, uint32_t height_blocks) {
    uint32_t bh = base;
    while (bh > 1 && height_blocks <= (bh * kGobHeight) / 2) bh /= 2;
    return bh;
}

uint32_t level_block_depth(uint32_t base, uint32_t depth) {
    uint32_t bd = base;
    while (bd > 1 && depth <= bd / 2) bd /= 2;
    return bd;
}

}  // namespace

TextureLayout compute_layout(uint32_t width, uint32_t height, uint32_t depth, uint32_t layers, uint32_t levels,
                             uint32_t bytes_per_block, uint32_t block_w, uint32_t block_h, bool linear, uint32_t stride,
                             bool is_3d) {
    TextureLayout t;
    t.linear = linear;
    t.bytes_per_block = bytes_per_block;
    t.block_w = block_w;
    t.block_h = block_h;
    t.layers = std::max(layers, 1u);
    levels = std::max(levels, 1u);
    width = std::max(width, 1u), height = std::max(height, 1u), depth = std::max(depth, 1u);

    if (linear) {
        uint32_t w_blocks = div_ceil(width, block_w), h_blocks = div_ceil(height, block_h);
        t.stride = stride ? stride : static_cast<uint32_t>(align(w_blocks * bytes_per_block, 128));
        t.levels.push_back({0, size_t{t.stride} * h_blocks, width, height, 1, w_blocks, h_blocks, 1, 1});
        t.layer_size = t.levels[0].size;
        return t;
    }

    uint32_t base_bh = base_block_height(div_ceil(height, block_h));
    uint32_t base_bd = is_3d ? std::min<uint32_t>(32, pow2_ceil(depth)) : 1;
    size_t offset = 0;
    for (uint32_t l = 0; l < levels; l++) {
        uint32_t w = std::max(width >> l, 1u), h = std::max(height >> l, 1u), d = is_3d ? std::max(depth >> l, 1u) : 1;
        uint32_t w_blocks = div_ceil(w, block_w), h_blocks = div_ceil(h, block_h);
        uint32_t bh = level_block_height(base_bh, h_blocks);
        uint32_t bd = is_3d ? level_block_depth(base_bd, d) : 1;
        size_t row_bytes = align(size_t{w_blocks} * bytes_per_block, kGobWidth);
        size_t rows = align(h_blocks, kGobHeight * bh);
        size_t size = row_bytes * rows * align(d, bd);
        t.levels.push_back({offset, size, w, h, d, w_blocks, h_blocks, bh, bd});
        offset += size;
    }
    // Cada capa empieza alineada al bloque del nivel 0.
    t.layer_size = t.layers > 1 ? align(offset, size_t{kGobSize} * base_bh * base_bd) : offset;
    return t;
}

// Dirección block-linear (Tegra): dentro de un GOB de 64x8 bytes los 512 bytes se ordenan en
// sectores de 16x2; los GOBs se apilan en vertical dentro del bloque y los bloques van por filas.
void deswizzle_level(const LevelLayout& level, uint32_t bpb, const uint8_t* src, uint8_t* dst) {
    const uint32_t row_bytes = level.width_blocks * bpb;
    const uint32_t gobs_per_row = div_ceil(row_bytes, kGobWidth);
    const uint32_t bh = level.block_height_gobs;
    const size_t block_size = size_t{kGobSize} * bh;
    const size_t block_row_size = block_size * gobs_per_row;
    for (uint32_t y = 0; y < level.height_blocks; y++) {
        const size_t block_row = y / (kGobHeight * bh);
        const uint32_t gob_y = (y / kGobHeight) % bh;
        const uint32_t in_y = y % kGobHeight;
        uint8_t* out = dst + size_t{y} * row_bytes;
        for (uint32_t x = 0; x < row_bytes; x += 16) {
            const uint32_t gob_x = x / kGobWidth, in_x = x % kGobWidth;
            const size_t gob = block_row * block_row_size + size_t{gob_x} * block_size + size_t{gob_y} * kGobSize;
            // Offset dentro del GOB (fórmula de la Tegra para el byte (in_x, in_y)).
            const uint32_t in_gob = ((in_x % 64) / 32) * 256 + ((in_y % 8) / 2) * 64 + ((in_x % 32) / 16) * 32 +
                                    (in_y % 2) * 16;
            std::memcpy(out + x, src + gob + in_gob, std::min<uint32_t>(16, row_bytes - x));
        }
    }
}

}  // namespace nvn
