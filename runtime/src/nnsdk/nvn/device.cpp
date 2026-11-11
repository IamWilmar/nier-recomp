// nvnDevice*: el NVNdevice del juego es la puerta al contexto Vulkan (vk.cpp).

#include <cstring>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/vk.h"

namespace {

// NVNdeviceBuilder: opaco y reservado por el juego; solo usamos sus primeros bytes.
struct DeviceBuilder {
    int flags;
};

struct Device {
    int flags = 0;
    int window_origin_mode = 0;
    int depth_mode = 0;
};

nvn::Objects<Device> g_devices;

// nvnDeviceGetInteger. NieR pregunta 17 valores (main+0x9d53c0) y los guarda en globales;
// el significado sale de cómo usa cada uno:
int device_info(int id) {
    switch (id) {
        case 0x04: return 256;      // alineación del código de shaders (+0x4e de relleno, tope 7 MB; main+0x9c8834)
        case 0x0a: return 256;      // alineación (la comprueba potencia de 2) de un pool interno (main+0x9d0894)
        case 0x0f: return 32;       // tamaño de descriptor de textura: × (reservados + 10240) (main+0x9ca0d4)
        case 0x10: return 32;       // tamaño de descriptor de sampler: × (reservados + 2048)
        case 0x11: return 256;      // descriptores de textura reservados al principio del pool
        case 0x12: return 256;      // descriptores de sampler reservados
        case 0x14: return 4096;     // alineación con la que redondea un bloque de 512 KB (main+0x9c5c94)
        case 0x38: return 32;       // stride mínimo de texturas lineales: el juego usa max(ancho·bpp, esto) (main+0x9c6014)
        case 0x4e: return 1024;     // relleno detrás del código de shaders
        default: return 0;          // 0x0d 0x13 0x19 0x1a 0x24 0x25 0x52 0x56: los guarda y no los usa
    }
}

}  // namespace

namespace nvn {
int g_window_origin_mode = 0;
int g_depth_mode = 0;
}  // namespace nvn
using nvn::g_depth_mode;
using nvn::g_window_origin_mode;

extern "C" {

void nvnDeviceBuilderSetDefaults(DeviceBuilder* b) { b->flags = 0; }
void nvnDeviceBuilderSetFlags(DeviceBuilder* b, int flags) { b->flags = flags; }

nvn::Boolean nvnDeviceInitialize(void* device, const DeviceBuilder* b) {
    if (!nvn::vk()) return 0;
    g_devices.create(device).flags = b->flags;
    NVN_LOG("dispositivo %p listo (flags 0x%x)\n", device, b->flags);
    return 1;
}

void nvnDeviceFinalize(void* device) { g_devices.destroy(device); }

void nvnDeviceGetInteger(const void*, int id, int* value) { *value = device_info(id); }

void nvnDeviceSetWindowOriginMode(void* device, int mode) {
    NVN_LOG("WindowOriginMode %d\n", mode);
    g_window_origin_mode = mode;
    if (Device* d = g_devices.get(device)) d->window_origin_mode = mode;
}

void nvnDeviceSetDepthMode(void* device, int mode) {
    NVN_LOG("DepthMode %d\n", mode);
    g_depth_mode = mode;
    if (Device* d = g_devices.get(device)) d->depth_mode = mode;
}

// Los colores de borrado rápido son una optimización del hardware de la Switch: en Vulkan no aplican.
nvn::Boolean nvnDeviceRegisterFastClearColor(void*, const float*, int) { return 1; }

}  // extern "C"

NVN_PROC(nvnDeviceBuilderSetDefaults);
NVN_PROC(nvnDeviceBuilderSetFlags);
NVN_PROC(nvnDeviceInitialize);
NVN_PROC(nvnDeviceFinalize);
NVN_PROC(nvnDeviceGetInteger);
NVN_PROC(nvnDeviceSetWindowOriginMode);
NVN_PROC(nvnDeviceSetDepthMode);
NVN_PROC(nvnDeviceRegisterFastClearColor);
