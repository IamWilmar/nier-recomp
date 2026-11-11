// Funciones de los objetos de estado de NVN y su traducción a Vulkan (ver state.h).

#include "nnsdk/nvn/state.h"

#include <cstring>

#include "nnsdk/nvn/nvn.h"

namespace nvn {

VkBlendFactor vk_blend_factor(int f) {
    switch (f) {
        case 1: return VK_BLEND_FACTOR_ZERO;
        case 2: return VK_BLEND_FACTOR_ONE;
        case 3: return VK_BLEND_FACTOR_SRC_COLOR;
        case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 7: return VK_BLEND_FACTOR_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR;
        case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 16: return VK_BLEND_FACTOR_SRC1_COLOR;
        case 17: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case 18: return VK_BLEND_FACTOR_SRC1_ALPHA;
        case 19: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp vk_blend_op(int e) {
    switch (e) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

VkCompareOp vk_compare_op(int f) {
    switch (f) {
        case 1: return VK_COMPARE_OP_NEVER;
        case 2: return VK_COMPARE_OP_LESS;
        case 3: return VK_COMPARE_OP_EQUAL;
        case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case 5: return VK_COMPARE_OP_GREATER;
        case 6: return VK_COMPARE_OP_NOT_EQUAL;
        case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        default: return VK_COMPARE_OP_ALWAYS;
    }
}

VkStencilOp vk_stencil_op(int o) {
    switch (o) {
        case 2: return VK_STENCIL_OP_ZERO;
        case 3: return VK_STENCIL_OP_REPLACE;
        case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 6: return VK_STENCIL_OP_INVERT;
        case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}

VkCullModeFlags vk_cull_mode(int face) {
    switch (face) {
        case 1: return VK_CULL_MODE_FRONT_BIT;
        case 2: return VK_CULL_MODE_BACK_BIT;
        case 3: return VK_CULL_MODE_FRONT_AND_BACK;
        default: return VK_CULL_MODE_NONE;
    }
}

VkPolygonMode vk_polygon_mode(int m) {
    return m == 0 ? VK_POLYGON_MODE_POINT : m == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
}

// NVNdrawPrimitive: orden de OpenGL. Los quads y polígonos no existen en Vulkan (se avisa al dibujar).
VkPrimitiveTopology vk_topology(int p) {
    switch (p) {
        case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        case 10: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
        case 11: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
        case 12: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
        case 13: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
        case 14: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

// NVNindexType: UNSIGNED_BYTE 0, UNSIGNED_SHORT 1, UNSIGNED_INT 2.
VkIndexType vk_index_type(int t) { return t == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16; }
uint32_t index_size(int t) { return t == 0 ? 1 : t == 1 ? 2 : 4; }

}  // namespace nvn

using namespace nvn;

extern "C" {

// --- Blend ---
void nvnBlendStateSetDefaults(BlendState* s) {
    *s = {};
    s->src_rgb = s->src_alpha = 2;  // ONE
    s->dst_rgb = s->dst_alpha = 1;  // ZERO
    s->eq_rgb = s->eq_alpha = 1;    // ADD
}
void nvnBlendStateSetBlendTarget(BlendState* s, int target) { s->target = static_cast<uint8_t>(target); }
void nvnBlendStateSetBlendFunc(BlendState* s, int src_rgb, int dst_rgb, int src_alpha, int dst_alpha) {
    s->src_rgb = src_rgb, s->dst_rgb = dst_rgb, s->src_alpha = src_alpha, s->dst_alpha = dst_alpha;
}
void nvnBlendStateSetBlendEquation(BlendState* s, int rgb, int alpha) { s->eq_rgb = rgb, s->eq_alpha = alpha; }

// --- Color ---
void nvnColorStateSetDefaults(ColorState* s) {
    *s = {};
    s->alpha_test = 8;  // ALWAYS
}
void nvnColorStateSetBlendEnable(ColorState* s, int index, bool enable) {
    if (enable)
        s->blend_enable |= 1u << index;
    else
        s->blend_enable &= ~(1u << index);
}
void nvnColorStateSetLogicOp(ColorState* s, int op) { s->logic_op = op; }
void nvnColorStateSetAlphaTest(ColorState* s, int func) { s->alpha_test = func; }

// --- Máscara de canales ---
void nvnChannelMaskStateSetDefaults(ChannelMaskState* s) { s->masks = 0xFFFFFFFF; }
void nvnChannelMaskStateSetChannelMask(ChannelMaskState* s, int index, bool r, bool g, bool b, bool a) {
    uint32_t bits = (r ? 1 : 0) | (g ? 2 : 0) | (b ? 4 : 0) | (a ? 8 : 0);
    s->masks = (s->masks & ~(0xFu << (4 * index))) | bits << (4 * index);
}

// --- Profundidad y stencil ---
void nvnDepthStencilStateSetDefaults(DepthStencilState* s) {
    *s = {};
    s->depth_func = 2;                   // LESS
    s->front_func = s->back_func = 8;    // ALWAYS
    s->front_fail = s->front_zfail = s->front_zpass = 1;  // KEEP
    s->back_fail = s->back_zfail = s->back_zpass = 1;
}
void nvnDepthStencilStateSetDepthTestEnable(DepthStencilState* s, bool e) { s->depth_test = e; }
void nvnDepthStencilStateSetDepthWriteEnable(DepthStencilState* s, bool e) { s->depth_write = e; }
void nvnDepthStencilStateSetDepthFunc(DepthStencilState* s, int f) { s->depth_func = f; }
void nvnDepthStencilStateSetStencilTestEnable(DepthStencilState* s, bool e) { s->stencil_test = e; }
void nvnDepthStencilStateSetStencilFunc(DepthStencilState* s, int faces, int f) {
    if (faces & 1) s->front_func = f;
    if (faces & 2) s->back_func = f;
}
void nvnDepthStencilStateSetStencilOp(DepthStencilState* s, int faces, int fail, int zfail, int zpass) {
    if (faces & 1) s->front_fail = fail, s->front_zfail = zfail, s->front_zpass = zpass;
    if (faces & 2) s->back_fail = fail, s->back_zfail = zfail, s->back_zpass = zpass;
}

// --- Polígono ---
void nvnPolygonStateSetDefaults(PolygonState* s) {
    *s = {};
    s->polygon_mode = 2;  // FILL
    s->front_face = 1;    // CCW
}
void nvnPolygonStateSetCullFace(PolygonState* s, int face) { s->cull_face = face; }
void nvnPolygonStateSetFrontFace(PolygonState* s, int face) { s->front_face = face; }
void nvnPolygonStateSetPolygonMode(PolygonState* s, int mode) { s->polygon_mode = mode; }
void nvnPolygonStateSetPolygonOffsetEnables(PolygonState* s, int enables) { s->offset_enables = enables; }

// --- Multisample ---
void nvnMultisampleStateSetDefaults(MultisampleState* s) {
    *s = {};
    s->samples = 1;
}
void nvnMultisampleStateSetMultisampleEnable(MultisampleState* s, bool e) { s->enable = e; }
void nvnMultisampleStateSetSamples(MultisampleState* s, int samples) { s->samples = samples; }
void nvnMultisampleStateSetAlphaToCoverageEnable(MultisampleState* s, bool e) { s->alpha_to_coverage = e; }

// --- Vértices ---
void nvnVertexAttribStateSetDefaults(VertexAttribState* s) { *s = {}; }
void nvnVertexAttribStateSetFormat(VertexAttribState* s, int format, ptrdiff_t offset) {
    s->format = format;
    s->offset = static_cast<uint16_t>(offset);
}
void nvnVertexAttribStateSetStreamIndex(VertexAttribState* s, int stream) { s->stream = stream; }
void nvnVertexStreamStateSetDefaults(VertexStreamState* s) { *s = {}; }
void nvnVertexStreamStateSetStride(VertexStreamState* s, ptrdiff_t stride) { s->stride = static_cast<uint32_t>(stride); }
void nvnVertexStreamStateSetDivisor(VertexStreamState* s, int divisor) { s->divisor = divisor; }

}  // extern "C"

NVN_PROC(nvnBlendStateSetDefaults);
NVN_PROC(nvnBlendStateSetBlendTarget);
NVN_PROC(nvnBlendStateSetBlendFunc);
NVN_PROC(nvnBlendStateSetBlendEquation);
NVN_PROC(nvnColorStateSetDefaults);
NVN_PROC(nvnColorStateSetBlendEnable);
NVN_PROC(nvnColorStateSetLogicOp);
NVN_PROC(nvnColorStateSetAlphaTest);
NVN_PROC(nvnChannelMaskStateSetDefaults);
NVN_PROC(nvnChannelMaskStateSetChannelMask);
NVN_PROC(nvnDepthStencilStateSetDefaults);
NVN_PROC(nvnDepthStencilStateSetDepthTestEnable);
NVN_PROC(nvnDepthStencilStateSetDepthWriteEnable);
NVN_PROC(nvnDepthStencilStateSetDepthFunc);
NVN_PROC(nvnDepthStencilStateSetStencilTestEnable);
NVN_PROC(nvnDepthStencilStateSetStencilFunc);
NVN_PROC(nvnDepthStencilStateSetStencilOp);
NVN_PROC(nvnPolygonStateSetDefaults);
NVN_PROC(nvnPolygonStateSetCullFace);
NVN_PROC(nvnPolygonStateSetFrontFace);
NVN_PROC(nvnPolygonStateSetPolygonMode);
NVN_PROC(nvnPolygonStateSetPolygonOffsetEnables);
NVN_PROC(nvnMultisampleStateSetDefaults);
NVN_PROC(nvnMultisampleStateSetMultisampleEnable);
NVN_PROC(nvnMultisampleStateSetSamples);
NVN_PROC(nvnMultisampleStateSetAlphaToCoverageEnable);
NVN_PROC(nvnVertexAttribStateSetDefaults);
NVN_PROC(nvnVertexAttribStateSetFormat);
NVN_PROC(nvnVertexAttribStateSetStreamIndex);
NVN_PROC(nvnVertexStreamStateSetDefaults);
NVN_PROC(nvnVertexStreamStateSetStride);
NVN_PROC(nvnVertexStreamStateSetDivisor);
