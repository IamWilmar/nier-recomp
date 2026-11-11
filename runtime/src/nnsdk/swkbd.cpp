// nn::swkbd: el teclado de software del sistema. NieR lo abre para el nombre de usuario de la partida
// (main+0x984cb4): pasa un búfer UTF-16 de 512 bytes como texto inicial y, si ShowKeyboard tiene éxito,
// lee de ese mismo búfer el resultado (UTF-16 terminado en 0). No hay applet: se devuelve el texto
// inicial si ya tiene nombre, o el de <datos>/nombre.txt (UTF-8, una línea), o "Flip".

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnsdk/types.h"

namespace nn::swkbd {

struct KeyboardConfig;
struct ShowKeyboardArg;
struct String {
    void* ptr;
    size_t size;  // capacidad del búfer en bytes
};

namespace {

const char16_t* g_initial = nullptr;

std::string configured_name() {
    std::string name = "Flip";
    const char* data = getenv("NIER_DATA");
    if (FILE* f = fopen((std::string(data ? data : "/sdcard/NieR") + "/nombre.txt").c_str(), "r")) {
        char line[64] = {};
        if (fgets(line, sizeof line, f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
            if (s.size() >= 2) name = s;
        }
        fclose(f);
    }
    return name;
}

size_t length16(const char16_t* s) {
    size_t n = 0;
    while (s && s[n]) n++;
    return n;
}

}  // namespace

void MakePreset(KeyboardConfig*, int) {}
size_t GetRequiredWorkBufferSize(bool) { return 0x1000; }
void SetGuideTextUtf8(KeyboardConfig*, const char*) {}
void SetInitialText(ShowKeyboardArg*, const char16_t* text) { g_initial = text; }

Result ShowKeyboard(String* out, const ShowKeyboardArg&) {
    auto* dst = static_cast<char16_t*>(out->ptr);
    const size_t capacity = out->size / sizeof(char16_t);
    if (!dst || capacity < 2) return ResultSuccess;
    if (g_initial && length16(g_initial) >= 2) {
        if (g_initial != dst) memmove(dst, g_initial, std::min(length16(g_initial) + 1, capacity) * sizeof(char16_t));
        dst[capacity - 1] = 0;
        return ResultSuccess;
    }
    // Nombre ASCII/UTF-8 básico a UTF-16 (los nombres de usuario de prueba no necesitan más).
    std::string name = configured_name();
    size_t n = 0;
    for (unsigned char c : name) {
        if (n + 1 >= capacity) break;
        if (c < 0x80) dst[n++] = c;
    }
    dst[n] = 0;
    printf("[swkbd] nombre: %s\n", name.c_str());
    return ResultSuccess;
}

}  // namespace nn::swkbd
