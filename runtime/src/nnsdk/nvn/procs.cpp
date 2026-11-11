// Punto de entrada de NVN. El juego pide cada función por nombre (nvnBootstrapLoader y luego
// nvnDeviceGetProcAddress). Lo implementado está registrado con NVN_PROC; el resto es un stub
// con nombre que registra la llamada, para ver la secuencia real de NieR.

#include <cstring>
#include <string>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/runtime.h"

namespace {

std::mutex& procs_mutex() {
    static std::mutex m;
    return m;
}

// Funciones de función-static: se llenan desde constructores estáticos de otros archivos.
std::unordered_map<std::string, void*>& implemented() {
    static std::unordered_map<std::string, void*> map;
    return map;
}

std::unordered_map<std::string, void*>& resolved() {
    static std::unordered_map<std::string, void*> map;
    return map;
}

// Mientras no estén hechas: los *Initialize devuelven NVN_TRUE para que el juego siga adelante.
uint64_t stub_result(const char* name) {
    size_t n = strlen(name);
    return n > 10 && strcmp(name + n - 10, "Initialize") == 0 ? 1 : 0;
}

}  // namespace

void nvn::register_proc(const char* name, void* fn) {
    std::lock_guard lock(procs_mutex());
    implemented()[name] = fn;
}

void* nvn::find_proc(const char* name) {
    std::lock_guard lock(procs_mutex());
    auto& cache = resolved();
    if (auto it = cache.find(name); it != cache.end()) return it->second;
    auto it = implemented().find(name);
    void* fn = it != implemented().end() ? it->second : nnsdk::host().make_stub(name, stub_result(name));
    cache.emplace(name, fn);
    return fn;
}

extern "C" void* nvnDeviceGetProcAddress(const void*, const char* name) { return name ? nvn::find_proc(name) : nullptr; }
NVN_PROC(nvnDeviceGetProcAddress);

extern "C" void* nvnBootstrapLoader(const char* name) {
    NVN_LOG("bootstrap: %s\n", name);
    return nvn::find_proc(name);
}
