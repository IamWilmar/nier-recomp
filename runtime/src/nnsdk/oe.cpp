// nn::oe: estado de la aplicación en el sistema. Lo que el juego usa como reloj tiene que avanzar:
// CRI Mana (los vídeos) mide la reproducción con GetProgramTotalActiveTime (main+0xa5154c), y con 0
// fijo el vídeo se queda en los primeros frames.

#include <time.h>

#include "nnsdk/types.h"

namespace nn::oe {

namespace {

int64_t monotonic_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec;
}

const int64_t g_start_ns = monotonic_ns();

}  // namespace

// Tiempo activo desde que arrancó el programa (en la consola no cuenta el tiempo en suspensión).
TimeSpan GetProgramTotalActiveTime() { return TimeSpan{monotonic_ns() - g_start_ns}; }

}  // namespace nn::oe
