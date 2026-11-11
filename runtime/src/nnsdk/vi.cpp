// nn::vi (pantalla y capa) y nv:: (arranque del driver). En Android solo hay una ventana:
// la de la APK, que es lo que devuelve GetNativeWindow y luego recibe NVN.

#include <cstddef>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace nn::vi {

struct Display;
struct Layer;

namespace {
char g_display, g_layer;
}

void Initialize() {}

Result OpenDefaultDisplay(Display** out) {
    *out = reinterpret_cast<Display*>(&g_display);
    return ResultSuccess;
}

Result CreateLayer(Layer** out, Display*) {
    *out = reinterpret_cast<Layer*>(&g_layer);
    return ResultSuccess;
}

// Sin ventana (terminal) se devuelve error: el juego no tiene dónde presentar.
Result GetNativeWindow(void** out, Layer*) {
    *out = nnsdk::host().native_window;
    return *out ? ResultSuccess : Result{0x272};
}

}  // namespace nn::vi

namespace nv {

void InitializeGraphics(void*, size_t) {}
void SetGraphicsAllocator(void* (*)(size_t, size_t, void*), void (*)(void*, void*), void* (*)(void*, size_t, void*),
                          void*) {}

}  // namespace nv
