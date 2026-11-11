// SPDX-License-Identifier: GPL-3.0-or-later
#include "nvn_shader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace nvnshader {

namespace {

class NvnEnvironment final : public Shader::Environment {
public:
    NvnEnvironment(const uint8_t* code, size_t max_size, const Options& options, bool compute)
        // En compute NVN deja toda la tabla del driver en c0 (handles en 0x168 + 8·i, SSBO en 0x310 + 16·k):
        // los TEX con índice inmediato leen ahí, no en c2 como en gráficos.
        : code_(code), max_size_(max_size), texture_bound_(compute ? 0 : options.texture_bound_buffer), options_(options) {
        if (compute) {
            stage = Shader::Stage::Compute;
            start_address = options.compute_code_offset;
            return;
        }
        start_address = kNvnHeaderSize;
        std::memcpy(&sph, code_ + kNvnHeaderSize, sizeof(sph));
        switch (sph.common0.shader_type) {
            case 1: stage = Shader::Stage::VertexB; break;
            case 2: stage = Shader::Stage::TessellationControl; break;
            case 3: stage = Shader::Stage::TessellationEval; break;
            case 4: stage = Shader::Stage::Geometry; break;
            case 5: stage = Shader::Stage::Fragment; break;
            default: stage = Shader::Stage::Compute; break;
        }
    }

    u64 ReadInstruction(u32 address) override {
        u64 v = 0;
        if (address + sizeof v <= max_size_) std::memcpy(&v, code_ + address, sizeof v);
        return v;
    }
    // Los valores de los constant buffers solo existen al dibujar, salvo c1 (constantes del propio programa):
    // ahí están las tablas de salto de los BRX, que Eden necesita para encontrar los destinos del switch.
    u32 ReadCbufValue(u32 index, u32 offset) override {
        if (index == 1) brx_ = true;  // Eden solo lee c1 al traducir para las tablas de un BRX (switch)
        u32 v = 0;
        if (index == 1 && options_.constants && offset + sizeof v <= options_.constants_size)
            std::memcpy(&v, options_.constants + offset, sizeof v);
        return v;
    }
    bool HasBrx() const { return brx_; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bound_; }
    // Gráficos: lo dice la cabecera (SPH), como en Eden con un GPU de verdad (más la pila CRS). Compute no
    // tiene SPH: lo que diga la sección de control o, si no, un margen (solo cuenta si el shader la usa).
    u32 LocalMemorySize() const override {
        if (stage != Shader::Stage::Compute)
            return static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
        return options_.local_memory ? options_.local_memory : 1024;
    }
    u32 SharedMemorySize() const override { return options_.shared_memory; }
    std::array<u32, 3> WorkgroupSize() const override {
        return {options_.workgroup[0], options_.workgroup[1], options_.workgroup[2]};
    }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}

private:
    const uint8_t* code_;
    bool brx_ = false;
    size_t max_size_;
    uint32_t texture_bound_;
    const Options& options_;
};

Stage to_stage(Shader::Stage s) {
    switch (s) {
        case Shader::Stage::VertexB: return Stage::Vertex;
        case Shader::Stage::TessellationControl: return Stage::TessControl;
        case Shader::Stage::TessellationEval: return Stage::TessEval;
        case Shader::Stage::Geometry: return Stage::Geometry;
        case Shader::Stage::Fragment: return Stage::Fragment;
        default: return Stage::Compute;
    }
}

// Mismo orden y mismo contador que EmitContext con bindings unificados y sin aliasing de
// descriptores: UBOs y SSBOs avanzan `count`, el resto uno por descriptor.
void list_resources(const Shader::Info& info, uint32_t binding, Translated& out) {
    for (const auto& d : info.constant_buffer_descriptors) {
        for (uint32_t k = 0; k < d.count; k++)
            out.resources.push_back({Resource::UniformBuffer, binding++, d.index + k, 0, 0});
    }
    for (const auto& d : info.storage_buffers_descriptors) {
        for (uint32_t k = 0; k < d.count; k++) {
            Resource r{Resource::StorageBuffer, binding++, d.cbuf_index, d.cbuf_offset + k * 16, 0};
            r.is_written = d.is_written;
            out.resources.push_back(r);
        }
    }
    for (const auto& d : info.texture_buffer_descriptors)
        out.resources.push_back({Resource::TextureBuffer, binding++, d.cbuf_index, d.cbuf_offset, 0});
    for (const auto& d : info.image_buffer_descriptors)
        out.resources.push_back({Resource::ImageBuffer, binding++, d.cbuf_index, d.cbuf_offset, 0});
    for (const auto& d : info.texture_descriptors) {
        Resource r{Resource::Texture, binding++, d.cbuf_index, d.cbuf_offset, static_cast<uint32_t>(d.type)};
        r.is_depth = d.is_depth;
        r.has_secondary = d.has_secondary;
        r.secondary_cbuf_index = d.secondary_cbuf_index;
        r.secondary_cbuf_offset = d.secondary_cbuf_offset;
        r.shift_left = d.shift_left;
        r.secondary_shift_left = d.secondary_shift_left;
        out.resources.push_back(r);
    }
    for (const auto& d : info.image_descriptors) {
        Resource r{Resource::Image, binding++, d.cbuf_index, d.cbuf_offset, static_cast<uint32_t>(d.type)};
        r.is_written = d.is_written;
        out.resources.push_back(r);
    }
}

}  // namespace

uint32_t binding_base(Stage s) { return s == Stage::Compute ? 0 : static_cast<uint32_t>(s) * kBindingsPerStage; }

const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Vertex: return "vertex";
        case Stage::TessControl: return "tess-control";
        case Stage::TessEval: return "tess-eval";
        case Stage::Geometry: return "geometry";
        case Stage::Fragment: return "pixel";
        case Stage::Compute: return "compute";
    }
    return "?";
}

bool translate(const uint8_t* code, size_t max_size, const Options& options, Translated& out) {
    out = {};
    uint32_t magic = 0;
    if (max_size >= 4) std::memcpy(&magic, code, 4);
    const bool compute = magic == kNvnComputeCodeMagic;
    if ((magic != kNvnCodeMagic && !compute) || max_size < kNvnHeaderSize + sizeof(Shader::ProgramHeader)) {
        out.error = "no es código NVN";
        return false;
    }
    try {
        NvnEnvironment env(code, max_size, options, compute);
        out.stage = to_stage(env.ShaderStage());
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
        Shader::ObjectPool<Shader::IR::Inst> insts;
        Shader::ObjectPool<Shader::IR::Block> blocks;
        Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress() + (compute ? 0 : sizeof(Shader::ProgramHeader)));
        Shader::HostTranslateInfo host{};
        Shader::IR::Program program = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, host);
        if (options.ir) {
            *options.ir = Shader::IR::DumpProgram(program);
            // Qué textura (índice de descriptor) usa cada instrucción de imagen: el volcado de Eden no lo dice.
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (Shader::IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                        char line[96];
                        snprintf(line, sizeof line, "[%016llx] descriptor %u\n",
                                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&inst)),
                                 static_cast<unsigned>(inst.Flags<Shader::IR::TextureInstInfo>().descriptor_index));
                        *options.ir += line;
                    }
        }

        // DEPURACIÓN: NIER_SIN_DESCARTE: los pixel shaders sin sus discard (con NIER_SIN_BLEND, el alfa que calculan
        // queda escrito tal cual en el render target).
        if (const char* keep = getenv("NIER_SIN_DESCARTE"); keep && *keep && out.stage == Stage::Fragment)
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (inst.GetOpcode() == Shader::IR::Opcode::DemoteToHelperInvocation)
                        inst.ReplaceOpcode(Shader::IR::Opcode::Void);

        out.has_brx = env.HasBrx();

        if (out.stage == Stage::Geometry) {
            // Lecturas que no existen en un geometry shader (la cabecera las marca en algunos de NieR) y
            // que Eden declararía como entradas inválidas: se leerán como 0.
            using A = Shader::IR::Attribute;
            for (A a : {A::InstanceId, A::VertexId, A::FrontFace, A::PointSpriteS, A::PointSpriteT,
                        A::TessellationEvaluationPointU, A::TessellationEvaluationPointV, A::Layer, A::ViewportIndex})
                program.info.loads.Set(a, false), program.info.passthrough.Set(a, false);
        }
        const Shader::Info& info = program.info;
        out.constant_buffer_mask = info.constant_buffer_mask;
        for (size_t k = 0; k < 18 && k < info.constant_buffer_used_sizes.size(); k++)
            out.constant_buffer_used[k] = info.constant_buffer_used_sizes[k];
        for (size_t k = 0; k < 32; k++) {
            if (info.loads.Generic(k)) out.input_generics |= uint64_t{1} << k;
            if (info.stores.Generic(k)) out.output_generics |= uint64_t{1} << k;
        }
        for (size_t k = 0; k < info.stores_frag_color.size(); k++)
            if (info.stores_frag_color[k]) out.color_outputs |= 1u << k;
        const uint32_t base = binding_base(out.stage);
        list_resources(info, base, out);

        Shader::Profile profile{};
        profile.supported_spirv = options.spirv_version;
        profile.unified_descriptor_binding = true;
        profile.support_multi_viewport = true;  // el Adreno lo tiene (y el dispositivo activa todas sus features)
        // Subgrupos: votos y ballot solo si el dispositivo los tiene, y en las etapas que los admite; si el
        // subgrupo es mayor que el warp de 32 de Maxwell, Eden emula los votos con ballot.
        constexpr uint32_t kVote = 0x2, kBallot = 0x8, kShuffle = 0x10, kShuffleRelative = 0x20, kQuad = 0x80;
        const uint32_t ops = options.subgroup_operations;
        const bool usable = (ops & kVote) && (ops & kBallot) && (ops & kShuffle);
        profile.support_vote = usable;
        profile.support_shuffle_relative = (ops & kShuffleRelative) != 0;
        profile.support_quad_shuffles = (ops & kQuad) != 0;
        profile.warp_size_potentially_larger_than_guest = options.subgroup_size > 32;
        profile.max_subgroup_size = options.subgroup_size;
        profile.supported_subgroup_stages = 0;
        if (usable) {
            static const uint32_t kVkStage[] = {0x1, 0x2, 0x4, 0x8, 0x10, 0x20};  // VertexB … Compute
            for (uint32_t k = 0; k < 6; k++)
                if (options.subgroup_stages & kVkStage[k]) profile.supported_subgroup_stages |= 1u << k;
        }
        Shader::RuntimeInfo runtime{};
        // NVN enlaza las etapas en el momento del draw (un programa por etapa): se asume que la etapa
        // anterior escribe todos los atributos genéricos, o Eden no declararía las entradas.
        // Solo genéricos y posición: con todo marcado, un geometry shader arrastraba como entradas valores
        // de sistema que no existen en su etapa (gl_InstanceIndex, gl_TessCoord…) y el driver lo rechazaba.
        for (size_t k = 0; k < 32; k++)
            if (options.previous_stores >> k & 1)
                for (size_t c = 0; c < 4; c++)
                    runtime.previous_stage_stores.Set(Shader::IR::Attribute::Generic0X + k * 4 + c);
        for (size_t c = 0; c < 4; c++) runtime.previous_stage_stores.Set(Shader::IR::Attribute::PositionX + c);
        runtime.input_topology = static_cast<Shader::InputTopology>(options.input_topology);
        runtime.convert_depth_mode = options.convert_depth_mode;
        Shader::Backend::Bindings bindings{};
        bindings.unified = base;
        out.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
        return true;
    } catch (const std::exception& e) {
        out.error = e.what();
        return false;
    }
}

}  // namespace nvnshader
