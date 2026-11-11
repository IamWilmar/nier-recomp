#pragma once
// Interruptores en vivo: un fichero vacío en la carpeta de datos (NIER_DATA) cambia un comportamiento sin
// reinstalar ni reiniciar el juego (`adb shell touch <datos>/<nombre>`, `rm` para volver). Sirven para
// comparar una optimización con y sin ella en la misma escena, y para apagarla si algo sale mal.
// Cada interruptor mira su fichero como mucho una vez por segundo y anota los cambios en el log.

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

#include "nnsdk/nvn/nvn.h"

namespace nvn {

class LiveFlag {
public:
    explicit LiveFlag(const char* name) : name_(name) {}
    bool get() {
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        if (now >= next_.load(std::memory_order_relaxed)) {
            next_.store(now + 1000, std::memory_order_relaxed);
            const char* data = getenv("NIER_DATA");
            const bool v = access((std::string(data ? data : ".") + "/" + name_).c_str(), F_OK) == 0;
            if (v != value_.exchange(v, std::memory_order_relaxed) || !logged_.exchange(true))
                NVN_LOG("interruptor %s: %s\n", name_, v ? "sí" : "no");
        }
        return value_.load(std::memory_order_relaxed);
    }

private:
    const char* name_;
    std::atomic<int64_t> next_{0};
    std::atomic<bool> value_{false}, logged_{false};
};

}  // namespace nvn

// true si existe <NIER_DATA>/<nombre>; `nombre` es un literal. Cuesta un steady_clock::now(): en caminos
// por draw, mejor leerlo una vez por frame o por envío.
#define NVN_FLAG(nombre)                     \
    ([]() -> bool {                          \
        static ::nvn::LiveFlag flag(nombre); \
        return flag.get();                   \
    }())
