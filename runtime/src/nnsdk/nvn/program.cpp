// SPDX-License-Identifier: GPL-3.0-or-later
// nvnProgram*: el motor usa un programa por etapa. Cada etapa trae su código (dirección de GPU =
// de CPU, dentro de un memory pool) y su sección de control. Al fijar los shaders se traducen a
// SPIR-V con el recompilador de Eden (runtime/recompiler) y se crea el VkShaderModule.
// Los que fallan se vuelcan a <datos>/shaders/ para estudiarlos con tools/shadertool.

#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/objects.h"
#include "nvn_shader.h"

namespace {

struct ShaderData {
    uint64_t code;        // dirección del código
    const void* control;  // sección de control
};

nvn::Objects<nvn::Program> g_programs;
std::atomic<int> g_translated{0}, g_failed{0};
std::atomic<int64_t> g_translate_us{0};

// Copia hasta size bytes sin caerse si la memoria se acaba antes (devuelve lo copiado).
size_t safe_copy(void* dst, const void* src, size_t size) {
    iovec local{dst, size}, remote{const_cast<void*>(src), size};
    ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return n > 0 ? static_cast<size_t>(n) : 0;
}

void dump_code(uint64_t code, const std::string& name) {
    const char* data = getenv("NIER_DATA");
    std::string dir = std::string(data ? data : "/sdcard/NieR") + "/shaders";
    mkdir(dir.c_str(), 0775);
    std::vector<uint8_t> buf(64 * 1024);
    size_t n = safe_copy(buf.data(), reinterpret_cast<const void*>(code), buf.size());
    if (FILE* f = fopen((dir + "/" + name + ".code").c_str(), "wb")) {
        fwrite(buf.data(), 1, n, f);
        fclose(f);
    }
}

// Hasta dónde se puede leer código: el final del memory pool que lo contiene.
size_t code_bound(uint64_t address) {
    size_t bound = 0;
    nvn::g_memory_pools.find_if([&](const void*, const nvn::MemoryPool& p) {
        auto start = reinterpret_cast<uint64_t>(p.storage);
        if (address < start || address >= start + p.size) return false;
        bound = start + p.size - address;
        return true;
    });
    return bound;
}

// Huella del código de un shader (FNV-1a hasta donde empiezan sus constantes): identifica el programa entre partidas.
uint64_t code_print(const nvn::ProgramStage& stage) {
    if (!stage.constants) return 0;
    const auto* p = reinterpret_cast<const uint8_t*>(stage.code);
    const size_t n = static_cast<size_t>(stage.constants - stage.code);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < n; k++) h = (h ^ p[k]) * 0x100000001b3ULL;
    return h;
}

// Compute con switch que se traducen con sus tablas de c1 a pesar de todo: los de <datos>/con_c1 (una huella
// hexadecimal por línea, o "*": todos). Se lee una vez.
bool c1_allowed(uint64_t print) {
    static const std::vector<uint64_t> list = [] {
        std::vector<uint64_t> l;
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/con_c1").c_str(), "r")) {
            char line[64];
            while (fgets(line, sizeof line, f)) l.push_back(line[0] == '*' ? ~0ULL : strtoull(line, nullptr, 16));
            fclose(f);
        }
        return l;
    }();
    for (uint64_t x : list)
        if (x == print || x == ~0ULL) return true;
    return false;
}

bool translate_stage(nvn::ProgramStage& out, uint64_t previous_stores = ~0ull, int input_topology = 0) {
    size_t bound = code_bound(out.code);
    if (!bound) {
        NVN_LOG("shader en 0x%llx fuera de cualquier memory pool\n", static_cast<unsigned long long>(out.code));
        return false;
    }
    auto start = std::chrono::steady_clock::now();
    nvnshader::Options options;
    options.convert_depth_mode = nvn::g_depth_mode == 0;
    if (out.control) {  // compute: inicio del código (+0x708), workgroup (+0x748) y memoria compartida (+0x754)
        const auto* control = static_cast<const uint8_t*>(out.control);
        memcpy(&options.compute_code_offset, control + 0x708, 4);
        memcpy(options.workgroup, control + 0x748, 12);
        memcpy(&options.shared_memory, control + 0x754, 4);
        for (uint32_t& w : options.workgroup) w = std::max(w, 1u);
    }
    // c1 (constantes tras el código): Eden lee de ahí las tablas de salto de los BRX (switch del shader). En los shaders
    // gráficos sí: sin ellas los casos no se traducen y el valor del switch queda con basura (la piel mojada de 2B y 9S,
    // con el azul a 13000-64512 y un halo de bloom enorme). En compute no: el único con switch que hemos visto (huella
    // 3e672e96e2eb85c5, un evaluador de curvas de 12 casos) con sus casos colgaba la GPU (en la demo del título, al
    // cargar las Ruinas, en combate) y sus resultados salían magenta (menú, salpicaduras, objetos resaltados). Sin
    // tablas cae siempre en el mismo camino, como hasta ahora. <datos>/con_c1 añade huellas (o "*": todos).
    uint32_t magic = 0;
    memcpy(&magic, reinterpret_cast<const void*>(out.code), 4);
    const bool compute = magic == nvnshader::kNvnComputeCodeMagic;
    const uint64_t print = code_print(out);
    if (out.constants && (!compute || c1_allowed(print))) {
        options.constants = reinterpret_cast<const uint8_t*>(out.constants);
        options.constants_size = out.constants_size;
    }
    options.previous_stores = previous_stores;
    options.input_topology = input_topology;
    if (nvn::VkContext* c = nvn::vk()) {
        options.subgroup_size = c->subgroup.subgroupSize;
        options.subgroup_stages = c->subgroup.supportedStages;
        options.subgroup_operations = c->subgroup.supportedOperations;
    }
    bool ok = nvnshader::translate(reinterpret_cast<const uint8_t*>(out.code), bound, options, out.shader);
    g_translate_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    if (!ok) {
        int n = g_failed++;
        NVN_LOG("shader %d no se pudo traducir: %s\n", n, out.shader.error.c_str());
        if (n < 32) {
            dump_code(out.code, "fallo" + std::to_string(n));
            if (out.control) dump_code(reinterpret_cast<uint64_t>(out.control), "fallo" + std::to_string(n) + "_ctrl");
        }
        return false;
    }
    if (out.shader.has_brx) {  // para ampliar la lista de c1_allowed sin adivinar
        NVN_LOG("shader %016llx (%s) con switch: tablas de c1 %s\n", static_cast<unsigned long long>(print),
                nvnshader::stage_name(out.shader.stage), options.constants ? "sí" : "no");
        if (NVN_FLAG("volcar_brx")) {  // DEPURACIÓN: su código y su sección de control, para estudiarlo con shadertool
            char name[40];
            snprintf(name, sizeof name, "brx_%016llx", static_cast<unsigned long long>(print));
            dump_code(out.code, name);
            if (out.control) dump_code(reinterpret_cast<uint64_t>(out.control), std::string(name) + "_ctrl");
        }
    }
    nvn::VkContext* c = nvn::vk();
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = out.shader.spirv.size() * sizeof(uint32_t);
    info.pCode = out.shader.spirv.data();
    if (VkResult r = vkCreateShaderModule(c->device, &info, nullptr, &out.module); r != VK_SUCCESS) {
        NVN_LOG("vkCreateShaderModule: %s\n", nvn::vk_result_name(r));
        return false;
    }
    int n = ++g_translated;
    if (n % 100 == 0)
        NVN_LOG("%d shaders traducidos (%lld ms en total), %d fallidos\n", n,
                static_cast<long long>(g_translate_us / 1000), g_failed.load());
    return true;
}

}  // namespace

nvn::Program* nvn::find_program(const void* program) { return g_programs.get(program); }

bool nvn::ensure_translated(ProgramStage& stage) {
    if (!stage.tried) {
        stage.tried = true;
        if (stage.control) {
            const auto* control = static_cast<const uint8_t*>(stage.control);
            uint32_t begin, end;
            memcpy(&begin, control + 0x700, 4);
            memcpy(&end, control + 0x704, 4);
            if (end > begin && end - begin <= 0x10000) {
                stage.constants = stage.code + begin;
                stage.constants_size = end - begin;
            }
        }
        translate_stage(stage);
    }
    return stage.module != VK_NULL_HANDLE;
}

VkShaderModule nvn::geometry_variant(ProgramStage& stage, uint64_t previous_stores, int input_topology) {
    const uint64_t key = previous_stores << 3 ^ static_cast<uint64_t>(input_topology);
    if (auto it = stage.variants.find(key); it != stage.variants.end()) return it->second;
    ProgramStage variant;
    variant.code = stage.code;
    variant.control = stage.control;
    variant.constants = stage.constants;
    variant.constants_size = stage.constants_size;
    VkShaderModule module = translate_stage(variant, previous_stores, input_topology) ? variant.module : VK_NULL_HANDLE;
    return stage.variants[key] = module;
}

void nvn::dump_stage_code(const ProgramStage& stage, const std::string& name) {
    dump_code(stage.code, name);
    if (stage.control) dump_code(reinterpret_cast<uint64_t>(stage.control), name + "_ctrl");
}

extern "C" {

nvn::Boolean nvnProgramInitialize(void* program, void*) {
    g_programs.create(program);
    return 1;
}
void nvnProgramFinalize(void* program) {
    if (nvn::Program* p = g_programs.get(program))
        for (nvn::ProgramStage& s : p->stages)
            if (s.module) vkDestroyShaderModule(nvn::vk()->device, s.module, nullptr);
    g_programs.destroy(program);
}

nvn::Boolean nvnProgramSetShaders(void* program, int count, const ShaderData* stages) {
    nvn::Program* p = g_programs.get(program);
    if (!p) return 0;
    for (nvn::ProgramStage& s : p->stages)
        if (s.module) vkDestroyShaderModule(nvn::vk()->device, s.module, nullptr);
    p->stages.clear();
    p->stages.resize(count);
    // Solo se apunta la dirección: el juego puede terminar de escribir el código después (NVN lo lee
    // al dibujar), así que se traduce en el executor la primera vez que se enlaza.
    for (int k = 0; k < count; k++) p->stages[k].code = stages[k].code, p->stages[k].control = stages[k].control;
    // Única en toda la ejecución: un Program nuevo puede reutilizar la memoria de otro ya finalizado, y las
    // cachés del ejecutor (pipelines, layouts, sets) van por (ProgramStage*, generación).
    static std::atomic<uint64_t> serial{0};
    p->generation = ++serial;
    return 1;
}

}  // extern "C"

NVN_PROC(nvnProgramInitialize);
NVN_PROC(nvnProgramFinalize);
NVN_PROC(nvnProgramSetShaders);
