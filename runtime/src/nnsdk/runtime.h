#pragma once
// Servicios internos de libnnsdk.so compartidos entre módulos (os, libc…).

#include "nnsdk/host.h"

namespace nnsdk {

const NnsdkHost& host();

// Cada hilo nuevo del juego avisa al loader (pila alternativa para el crash handler).
void on_thread_start();

}  // namespace nnsdk
