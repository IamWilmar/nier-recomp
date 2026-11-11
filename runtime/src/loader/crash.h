#pragma once
#include <stdint.h>
// Si el juego se cae: registros, dirección traducida a main+offset, últimos stubs y backtrace.

void install_crash_handler();

// sigaltstack es por hilo: cada hilo del juego lo llama al arrancar.
void install_thread_altstack();

// DEPURACIÓN: protege la página de addr; el primer acceso (lectura o escritura) se registra y se deja pasar.
void watch_writes(uint64_t addr);
