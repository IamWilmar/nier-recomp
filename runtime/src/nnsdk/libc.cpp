// Partes de la libc de Nintendo (musl) cuya ABI no coincide con la de bionic.

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nnsdk/runtime.h"

namespace {
NnsdkHost g_host{};
}

namespace nnsdk {
const NnsdkHost& host() { return g_host; }
void on_thread_start() {
    if (g_host.thread_start) g_host.thread_start();
}
}  // namespace nnsdk

extern "C" void nnsdk_attach(const NnsdkHost* host) { g_host = *host; }

// El crt0 de cada módulo la llama desde DT_INIT con sus tablas; si no devuelve 1, el
// propio crt0 recorre el .init_array. Con 0 dejamos que lo haga él.
extern "C" int __nnmusl_init_dso() { return 0; }

extern "C" int* __errno_location() { return &errno; }

extern "C" [[noreturn]] void __assert_fail(const char* expr, const char* file, int line, const char* func) {
    fprintf(stdout, "assert del juego: %s (%s:%d, %s)\n", expr, file, line, func);
    fflush(stdout);
    abort();
}

// El juego libera con su propio free: hay que reservar con su malloc.
extern "C" char* strdup(const char* s) {
    size_t n = strlen(s) + 1;
    auto* p = static_cast<char*>(g_host.game_malloc(n));
    return p ? static_cast<char*>(memcpy(p, s, n)) : nullptr;
}

// Valores de musl: FP_NAN 0, FP_INFINITE 1, FP_ZERO 2, FP_SUBNORMAL 3, FP_NORMAL 4. Bionic usa otros.
extern "C" int __fpclassifyl(long double x) {
    switch (std::fpclassify(x)) {
        case FP_NAN: return 0;
        case FP_INFINITE: return 1;
        case FP_ZERO: return 2;
        case FP_SUBNORMAL: return 3;
        default: return 4;
    }
}

extern "C" int __signbitl(long double x) { return std::signbit(x); }
