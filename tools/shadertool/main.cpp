// SPDX-License-Identifier: GPL-3.0-or-later
// shadertool: traduce en el Mac un shader NVN volcado por nnsdk/nvn/program.cpp a SPIR-V con el
// mismo puente que usa la Flip (runtime/recompiler/nvn_shader) y lista sus recursos con su binding.
// Uso: shadertool <shader.code> [salida.spv]

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iterator>
#include <vector>

#include "nvn_shader.h"

static void on_abort(int) {
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    _exit(134);
}

int main(int argc, char** argv) {
    signal(SIGABRT, on_abort);
    if (argc < 2) {
        std::fprintf(stderr, "uso: %s <shader.code> [salida.spv]\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> code((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    nvnshader::Translated t;
    nvnshader::Options options;
    std::string ir;
    if (getenv("IR")) options.ir = &ir;  // IR=1 shadertool …: imprime el IR de Eden
    std::vector<uint8_t> ctrl;
    if (const char* ctrl_path = getenv("CTRL")) {  // CTRL=<sección de control>: datos de compute como en la Flip
        std::ifstream cf(ctrl_path, std::ios::binary);
        ctrl.assign((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        if (ctrl.size() >= 0x708 && !getenv("SIN_C1")) {  // c1 como en program.cpp (SIN_C1=1: como antes)
            uint32_t begin, end;
            std::memcpy(&begin, &ctrl[0x700], 4);
            std::memcpy(&end, &ctrl[0x704], 4);
            if (end > begin && end - begin <= 0x10000 && end <= code.size()) {
                options.constants = code.data() + begin;
                options.constants_size = end - begin;
            }
        }
        if (ctrl.size() >= 0x758) {
            std::memcpy(&options.compute_code_offset, &ctrl[0x708], 4);
            std::memcpy(options.workgroup, &ctrl[0x748], 12);
            std::memcpy(&options.shared_memory, &ctrl[0x754], 4);
            for (uint32_t& w : options.workgroup) w = w ? w : 1;
        }
    }
    if (!nvnshader::translate(code.data(), code.size(), options, t)) {
        std::fprintf(stderr, "%s: %s\n", argv[1], t.error.c_str());
        return 1;
    }
    static const char* kKinds[] = {"ubo", "ssbo", "texbuf", "imgbuf", "textura", "imagen"};
    std::printf("%s: %s, %zu palabras de SPIR-V, entradas 0x%llx, colores 0x%x\n", argv[1],
                nvnshader::stage_name(t.stage), t.spirv.size(), static_cast<unsigned long long>(t.input_generics),
                t.color_outputs);
    for (const auto& r : t.resources) {
        if (r.kind == nvnshader::Resource::UniformBuffer)
            std::printf("  binding %u: ubo c%u\n", r.binding, r.cbuf_index);
        else
            std::printf("  binding %u: %s, handle en c%u[0x%x]%s\n", r.binding, kKinds[r.kind], r.cbuf_index,
                        r.cbuf_offset,
                        r.has_secondary ? (" | c" + std::to_string(r.secondary_cbuf_index) + "[0x" +
                                           [&] { char b[16]; snprintf(b, sizeof b, "%x", r.secondary_cbuf_offset); return std::string(b); }() +
                                           "] desplazamientos " + std::to_string(r.shift_left) + "/" + std::to_string(r.secondary_shift_left)).c_str()
                                        : "");
    }
    if (!ir.empty()) std::printf("%s\n", ir.c_str());
    if (argc > 2) {
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(t.spirv.data()), t.spirv.size() * sizeof(uint32_t));
    }
    return 0;
}
