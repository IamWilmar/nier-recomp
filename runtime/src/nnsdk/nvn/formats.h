#pragma once
// NVNformat -> VkFormat y bytes por bloque. El orden del enum es el de NVN (estilo OpenGL);
// lo confirman los dos formatos que NieR registra para borrado rápido: 0x25 (RGBA8, el de
// la ventana de 1280x720) y 0x3f (R11G11B10F, el del buffer HDR).

#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn {

struct FormatInfo {
    VkFormat vk;
    uint8_t bytes;   // por píxel, o por bloque si block > 1
    uint8_t block;   // ancho del bloque comprimido (1 = sin comprimir)
    const char* name;
    uint8_t block_h = 0;  // alto del bloque si no es cuadrado (ASTC 5x4, 8x6…); 0 = igual al ancho
    uint32_t block_height() const { return block_h ? block_h : block; }
};

// nullptr si no lo conocemos todavía (se registra en el log).
const FormatInfo* format_info(int nvn_format);

}  // namespace nvn
