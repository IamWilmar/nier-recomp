#include "loader/run.h"

#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <string>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "loader/crash.h"
#include "loader/imports.h"
#include "loader/nso.h"
#include "nnsdk/host.h"

namespace {

Module g_main, g_sdk;
std::atomic<bool> g_done{false};

[[noreturn]] void die(const std::string& msg) {
    fprintf(stdout, "error: %s\n", msg.c_str());
    fflush(stdout);
    exit(1);
}

// MainThreadStackSize del main.npdm (META, offset 0x1C).
uint32_t npdm_stack_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    char buf[0x20] = {};
    if (!f.read(buf, sizeof buf) || memcmp(buf, "META", 4) != 0) return 0;
    uint32_t v;
    memcpy(&v, buf + 0x1C, sizeof v);
    return v;
}

template <typename Fn>
Fn game_fn(const Module& m, const char* name) {
    const ElfSym* s = m.find_export(name);
    return s ? reinterpret_cast<Fn>(m.addr(s->value)) : nullptr;
}

// Orden de arranque del SDK: nninitStartup (monta el heap) antes de los constructores, luego nnMain.
void* game_thread(void*) {
    install_thread_altstack();
    using VoidFn = void (*)();
    if (auto startup = game_fn<VoidFn>(g_main, "nninitStartup")) {
        printf("→ nninitStartup\n");
        startup();
    }
    if (g_main.init) {
        // El DT_INIT del crt0 de Nintendo recorre él mismo el .init_array
        // (cuando __nnmusl_init_dso devuelve 0): no hay que correrlo otra vez.
        printf("→ DT_INIT (constructores)\n");
        reinterpret_cast<VoidFn>(g_main.addr(g_main.init))();
    } else {
        auto* ctors = reinterpret_cast<uint64_t*>(g_main.addr(g_main.init_array));
        size_t count = g_main.init_array_size / sizeof(uint64_t);
        printf("→ %zu constructores\n", count);
        for (size_t k = 0; k < count; k++)
            if (ctors[k] && ctors[k] != ~uint64_t{0}) reinterpret_cast<VoidFn>(ctors[k])();
    }
    printf("→ nnMain\n");
    game_fn<VoidFn>(g_main, "nnMain")();
    printf("nnMain volvió\n");
    g_done = true;
    return nullptr;
}

}  // namespace

int nier_run(const RunConfig& cfg) {
    install_crash_handler();
    std::string err;
    if (!init_imports(cfg.nnsdk, err)) die(err);
    if (!load_nso(cfg.exefs + "/main", g_main, err)) die(err);
    bool have_sdk = load_nso(cfg.exefs + "/sdk", g_sdk, err);
    if (!have_sdk) printf("[aviso] sin sdk para las constantes: %s\n", err.c_str());
    register_module(&g_main);
    printf("main en %p (%.1f MB), %zu símbolos, %zu relocaciones\n", g_main.base, g_main.size / 1048576.0,
           g_main.symbols.size(), g_main.relocs.size());

    auto resolve = [&](const ElfSym& s) { return resolve_import(s, have_sdk ? &g_sdk : nullptr); };
    if (!relocate(g_main, resolve, err) || !protect(g_main, err)) die(err);
    report_imports(cfg.list_stubs);

    NnsdkHost host{};
    host.game_malloc = game_fn<void* (*)(size_t)>(g_main, "malloc");
    host.game_free = game_fn<void (*)(void*)>(g_main, "free");
    host.game_new = game_fn<void* (*)(size_t)>(g_main, "_Znwm");
    host.game_delete = game_fn<void (*)(void*)>(g_main, "_ZdlPv");
    host.thread_start = install_thread_altstack;
    host.make_stub = make_named_stub;
    host.watch_writes = watch_writes;
    host.native_window = cfg.native_window;
    host.input = cfg.input;
    host.window_state = cfg.window_state;
    if (!host.game_malloc || !host.game_free || !host.game_new || !host.game_delete)
        die("main no exporta malloc/free/new/delete");
    reinterpret_cast<void (*)(const NnsdkHost*)>(nnsdk_symbol("nnsdk_attach"))(&host);

    uint32_t stack = std::max<uint32_t>(npdm_stack_size(cfg.exefs + "/main.npdm"), 1 << 20);
    printf("hilo principal con %u KB de pila\n", stack / 1024);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stack);
    pthread_t thread;
    if (pthread_create(&thread, &attr, game_thread, nullptr) != 0) die("pthread_create");

    // Watchdog: cada watchdog_seconds vuelca dónde está el hilo principal, sin pararlo. En la
    // segunda instantánea vuelca además todos los hilos del juego, uno detrás de otro.
    auto list_threads = reinterpret_cast<int (*)(pthread_t*, int)>(nnsdk_symbol("nnsdk_list_threads"));
    // DEPURACIÓN: con el fichero <datos>/hilos se vuelcan todos los hilos del juego en ese momento (p. ej. uno que
    // gasta un núcleo entero) y se borra el fichero.
    const char* data = getenv("NIER_DATA");
    const std::string threads_request = std::string(data ? data : ".") + "/hilos";
    for (int tenths = 1, snapshots = 0; !g_done; tenths++) {
        usleep(100'000);
        if (tenths % 10 == 0 && list_threads && access(threads_request.c_str(), F_OK) == 0) {
            unlink(threads_request.c_str());
            pthread_t others[64];
            int n = list_threads(others, 64);
            for (int k = 0; k < n; k++) {
                usleep(50'000);
                pthread_kill(others[k], SIGUSR2);
            }
        }
        if (cfg.watchdog_seconds <= 0 || tenths % (cfg.watchdog_seconds * 10) != 0) continue;
        pthread_kill(thread, SIGUSR2);
        if (++snapshots != 2 || !list_threads) continue;
        pthread_t others[64];
        int n = list_threads(others, 64);
        for (int k = 0; k < n; k++) {
            usleep(50'000);
            pthread_kill(others[k], SIGUSR2);
        }
    }
    pthread_join(thread, nullptr);
    return 0;
}
