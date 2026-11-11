// nn::hid: la Flip es una consola portátil, así que el juego ve un mando en modo portátil
// (NpadIdType Handheld = 0x20, estilo Handheld) con los botones que traduce el loader (host.h).
// Sin conexión, NieR muestra el applet de "conecta un mando" una y otra vez.
//
// Se definen con su nombre mangleado: los tipos BitFlagSet<32, NpadStyleTag> se pasan por valor como
// un uint32 y así no hay que reproducir las plantillas del SDK.

#include <time.h>

#include <cstdint>
#include <cstring>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

namespace {

constexpr uint32_t kNpadHandheld = 0x20;
constexpr uint32_t kStyleHandheld = 1u << 1;
// Atributos del estado: conectado, con cable, mitad izquierda y derecha conectadas y con cable.
constexpr uint32_t kHandheldAttributes = 0x3F;

// Estado común de todos los estilos (NpadFullKeyState, NpadHandheldState, NpadJoyDualState): 40 bytes.
struct NpadState {
    int64_t sampling_number;
    uint64_t buttons;
    int32_t lx, ly, rx, ry;
    uint32_t attributes;
    uint32_t reserved;
};
static_assert(sizeof(NpadState) == 0x28);

std::atomic<int64_t> g_sampling{0};

void fill_state(NpadState* out, uint32_t id) {
    memset(out, 0, sizeof *out);
    out->sampling_number = ++g_sampling;
    NnsdkInput* in = nnsdk::host().input;
    if (id != kNpadHandheld || !in) return;
    uint64_t buttons = in->buttons.load();
    if (uint64_t latched = in->latched.load()) {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        if (int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec < in->latched_until_ns.load())
            buttons |= latched;
        else
            in->latched.fetch_and(~latched);
    }
    out->buttons = buttons;
    static uint64_t last = 0;  // DEPURACIÓN: qué ve el juego cuando cambian los botones
    if (buttons != last) {
        static int logged = 0;
        if (logged++ < 60) printf("[hid] botones 0x%llx\n", static_cast<unsigned long long>(buttons));
        last = buttons;
    }
    out->lx = in->sticks[0], out->ly = in->sticks[1], out->rx = in->sticks[2], out->ry = in->sticks[3];
    out->attributes = kHandheldAttributes;
}

}  // namespace

extern "C" {

void _ZN2nn3hid14InitializeNpadEv() {}
void _ZN2nn3hid21InitializeTouchScreenEv() {}
void _ZN2nn3hid22SetSupportedNpadIdTypeEPKjm(const uint32_t*, size_t) {}
void _ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE(uint32_t) {}
void _ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE(int) {}
void _ZN2nn3hid24ShowControllerStrapGuideEv() {}

// Solo hay mando en el modo portátil.
uint32_t _ZN2nn3hid15GetNpadStyleSetERKj(const uint32_t& id) {
    return id == kNpadHandheld && nnsdk::host().input ? kStyleHandheld : 0;
}

void _ZN2nn3hid12GetNpadStateEPNS0_16NpadFullKeyStateERKj(NpadState* out, const uint32_t& id) { fill_state(out, id); }
void _ZN2nn3hid12GetNpadStateEPNS0_17NpadHandheldStateERKj(NpadState* out, const uint32_t& id) { fill_state(out, id); }
void _ZN2nn3hid12GetNpadStateEPNS0_16NpadJoyDualStateERKj(NpadState* out, const uint32_t& id) { fill_state(out, id); }

// Applet de selección de mando: el portátil ya está conectado como jugador 1.
struct ControllerSupportResultInfo {
    int8_t player_count;
    char pad[3];
    uint32_t selected_id;
};
nn::Result _ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE(
    ControllerSupportResultInfo* out, const void*) {
    if (out) {
        out->player_count = 1;
        out->selected_id = kNpadHandheld;
    }
    return nn::ResultSuccess;
}
void _ZN2nn3hid20ControllerSupportArg10SetDefaultEv(void*) {}

// Táctil: sin toques (cabecera de TouchScreenState<1>: muestra y número de toques).
void _ZN2nn3hid19GetTouchScreenStateILm1EEEvPNS0_16TouchScreenStateIXT_EEE(int64_t* out) {
    out[0] = g_sampling.load();
    out[1] = 0;
}

// Vibración y sensores de movimiento: presentes pero inertes de momento.
int _ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE(
    uint32_t* handles, int count, const uint32_t& id, uint32_t) {
    int n = count < 2 ? count : 2;
    for (int k = 0; k < n; k++) handles[k] = (id & 0xFF) << 8 | k;
    return n;
}
void _ZN2nn3hid22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE(uint32_t* info,
                                                                                                   const uint32_t& handle) {
    info[0] = 1;  // actuador lineal
    info[1] = handle & 1;  // izquierda / derecha
}
void _ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE(const uint32_t&) {}
void _ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE(const uint32_t&, const void*) {}

int _ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE(
    uint32_t* handles, int count, const uint32_t& id, uint32_t) {
    int n = count < 2 ? count : 2;
    for (int k = 0; k < n; k++) handles[k] = (id & 0xFF) << 8 | k;
    return n;
}
void _ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE(const uint32_t&) {}
// Solo la cabecera (delta y número de muestra) y la aceleración: el resto del struct se deja como está.
void _ZN2nn3hid21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE(uint8_t* out, const uint32_t&) {
    memset(out, 0, 28);
    int64_t sampling = g_sampling.load();
    memcpy(out + 8, &sampling, sizeof sampling);
}
bool _ZN2nn3hid21IsSixAxisSensorAtRestERKNS0_19SixAxisSensorHandleE(const uint32_t&) { return true; }

}  // extern "C"
