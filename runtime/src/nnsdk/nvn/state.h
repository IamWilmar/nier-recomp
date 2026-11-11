#pragma once
// Objetos de estado de NVN (blend, color, máscara de canales, profundidad/stencil, polígono,
// multisample, atributos y streams de vértice). Viven dentro de structs del juego cuyo tamaño sale de
// las direcciones de la traza (p. ej. tres BlendState seguidos a 8 bytes): aquí se codifican en esos
// bytes con nuestro propio formato. Los valores de los enums siguen el orden estilo OpenGL de NVN; los
// que el juego usa al arrancar lo confirman (ONE/ZERO, LEQUAL, ALWAYS, BACK, FILL, REPEAT…).

#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn {

struct BlendState {  // 8 bytes
    uint8_t target;
    uint8_t src_rgb, dst_rgb, src_alpha, dst_alpha;  // NVNblendFunc (ZERO = 1, ONE = 2…)
    uint8_t eq_rgb, eq_alpha;                        // NVNblendEquation (ADD = 1…)
    uint8_t pad;
};
struct ColorState {  // 4 bytes
    uint8_t blend_enable;  // bit por render target
    uint8_t logic_op;
    uint8_t alpha_test;    // NVNalphaFunc (NEVER = 1 … ALWAYS = 8)
    uint8_t pad;
};
struct ChannelMaskState {  // 4 bytes: 4 bits (RGBA) por render target
    uint32_t masks;
};
struct DepthStencilState {  // 8 bytes; funciones NVNdepthFunc/NVNstencilFunc (NEVER = 1 … ALWAYS = 8)
    uint64_t depth_test : 1, depth_write : 1, depth_func : 4, stencil_test : 1;
    uint64_t front_func : 4, back_func : 4;
    uint64_t front_fail : 4, front_zfail : 4, front_zpass : 4;  // NVNstencilOp (KEEP = 1 …)
    uint64_t back_fail : 4, back_zfail : 4, back_zpass : 4;
};
struct PolygonState {  // 4 bytes
    uint8_t cull_face;     // NVNface: NONE 0, FRONT 1, BACK 2, FRONT_AND_BACK 3
    uint8_t front_face;    // CW 0, CCW 1
    uint8_t polygon_mode;  // POINT 0, LINE 1, FILL 2
    uint8_t offset_enables;
};
struct MultisampleState {  // 24 bytes en NVN; usamos 4
    uint8_t enable, samples, alpha_to_coverage, pad;
};
struct VertexAttribState {  // 4 bytes
    uint8_t format;   // NVNformat
    uint8_t stream;
    uint16_t offset;
};
struct VertexStreamState {  // 8 bytes
    uint32_t stride;
    uint32_t divisor;
};

static_assert(sizeof(BlendState) == 8 && sizeof(ColorState) == 4 && sizeof(ChannelMaskState) == 4 &&
              sizeof(DepthStencilState) == 8 && sizeof(PolygonState) == 4 && sizeof(VertexAttribState) == 4 &&
              sizeof(VertexStreamState) == 8);

VkBlendFactor vk_blend_factor(int nvn);
VkBlendOp vk_blend_op(int nvn);
VkCompareOp vk_compare_op(int nvn);
VkStencilOp vk_stencil_op(int nvn);
VkCullModeFlags vk_cull_mode(int nvn_face);
VkPolygonMode vk_polygon_mode(int nvn);
VkPrimitiveTopology vk_topology(int nvn_primitive);
VkIndexType vk_index_type(int nvn);
uint32_t index_size(int nvn);

}  // namespace nvn
