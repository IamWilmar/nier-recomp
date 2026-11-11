// Entrada de la APK (NativeActivity): espera la ventana, manda stdout a <datos>/log.txt y
// arranca el juego en su propio hilo. El hilo de android_main sigue atendiendo a Android y traduce
// el mando de la Flip a botones de la Switch.

#include <android/input.h>
#include <android/keycodes.h>
#include <android/log.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "loader/run.h"
#include "nnsdk/host.h"
#include "nnsdk/rutas.h"

namespace {

// <datos>: la carpeta del juego que eligió el usuario en la pantalla de inicio (SetupActivity, files/carpeta.txt).
std::string g_data;
RunConfig g_cfg;
NnsdkInput g_input;
uint64_t g_key_buttons = 0, g_axis_buttons = 0;

// Bits de nn::hid::NpadButton.
enum : uint64_t {
    kA = 1 << 0, kB = 1 << 1, kX = 1 << 2, kY = 1 << 3, kStickL = 1 << 4, kStickR = 1 << 5, kL = 1 << 6, kR = 1 << 7,
    kZL = 1 << 8, kZR = 1 << 9, kPlus = 1 << 10, kMinus = 1 << 11, kLeft = 1 << 12, kUp = 1 << 13, kRight = 1 << 14,
    kDown = 1 << 15, kStickLLeft = 1 << 16, kStickLUp = 1 << 17, kStickLRight = 1 << 18, kStickLDown = 1 << 19,
    kStickRLeft = 1 << 20, kStickRUp = 1 << 21, kStickRRight = 1 << 22, kStickRDown = 1 << 23,
};

// Por posición, con la disposición de Nintendo: el de abajo es B, el de la derecha A.
uint64_t button_for(int32_t keycode) {
    switch (keycode) {
        case AKEYCODE_BUTTON_A: return kB;
        case AKEYCODE_BUTTON_B: return kA;
        case AKEYCODE_BUTTON_X: return kY;
        case AKEYCODE_BUTTON_Y: return kX;
        case AKEYCODE_BUTTON_L1: return kL;
        case AKEYCODE_BUTTON_R1: return kR;
        case AKEYCODE_BUTTON_L2: return kZL;
        case AKEYCODE_BUTTON_R2: return kZR;
        case AKEYCODE_BUTTON_THUMBL: return kStickL;
        case AKEYCODE_BUTTON_THUMBR: return kStickR;
        case AKEYCODE_BUTTON_START: return kPlus;
        case AKEYCODE_BUTTON_SELECT:
        case AKEYCODE_BACK: return kMinus;
        case AKEYCODE_DPAD_UP: return kUp;
        case AKEYCODE_DPAD_DOWN: return kDown;
        case AKEYCODE_DPAD_LEFT: return kLeft;
        case AKEYCODE_DPAD_RIGHT: return kRight;
        default: return 0;
    }
}

int32_t to_stick(float v) { return static_cast<int32_t>(std::lround(std::fmax(-1.0f, std::fmin(1.0f, v)) * 32767.0f)); }

uint64_t g_auto_buttons = 0;  // DEPURACIÓN: autojuego (NIER_AUTOJUEGO)
// El jugador tocó el mando físico: el autojuego se retira para siempre (las teclas que inyecta adb vienen del
// dispositivo virtual, id < 0, y no cuentan).
std::atomic<bool> g_player_took_over{false};
void take_over(AInputEvent* event) {
    if (AInputEvent_getDeviceId(event) <= 0) return;
    // Solo un botón pulsado o un stick movido de verdad: el mando de la Flip manda eventos de ejes en reposo, y eso
    // paraba el paseo automático nada más empezar.
    const bool key = AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY && AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_DOWN;
    const bool stick = AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION &&
                       (std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_X, 0)) > 0.5f ||
                        std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_Y, 0)) > 0.5f ||
                        std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_Z, 0)) > 0.5f ||
                        std::fabs(AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_RZ, 0)) > 0.5f);
    if ((!key && !stick) || g_player_took_over.exchange(true)) return;
    g_auto_buttons = 0;
    printf("[entrada] el jugador toma el mando: autojuego desactivado\n");
}

void publish() { g_input.buttons = g_key_buttons | g_axis_buttons | g_auto_buttons; }

// DEPURACIÓN: con NIER_AUTOJUEGO=<segundo>, desde ese segundo mantiene R (disparo de la unidad de vuelo) y
// mueve el stick izquierdo en zigzag, para que las pruebas lleguen lejos sin que el jugador muera.
void* autoplay(void*) {
    const double from = atof(getenv("NIER_AUTOJUEGO"));
    timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        usleep(50'000);
        timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        const double t = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (t < from) continue;
        if (g_player_took_over) return nullptr;
        g_auto_buttons = 1 << 7;  // R
        g_input.sticks[0] = to_stick(0.7f * static_cast<float>(std::sin(t * 1.6)));
        g_input.sticks[1] = to_stick(0.4f * static_cast<float>(std::sin(t * 0.7)));
        publish();
    }
    return nullptr;
}

// DEPURACIÓN: mientras exista <datos>/paseo, el personaje camina en zigzag y la cámara gira despacio (para medir el
// mundo abierto sin el jugador). Al quitar el fichero se sueltan los sticks; tocar el mando físico lo para del todo.
void* walk(void*) {
    const std::string file = g_data + "/paseo";
    bool active = false;
    double t = 0;
    for (;;) {
        usleep(50'000);
        if (g_player_took_over) return nullptr;
        const bool want = access(file.c_str(), F_OK) == 0;
        if (want != active) {
            active = want;
            printf("[entrada] paseo automático: %s\n", active ? "sí" : "no");
            if (!active) {
                for (auto& s : g_input.sticks) s = 0;
                publish();
            }
        }
        if (!active) continue;
        t += 0.05;
        g_input.sticks[0] = to_stick(0.6f * static_cast<float>(std::sin(t * 0.5)));
        g_input.sticks[1] = to_stick(0.9f);                                          // adelante
        g_input.sticks[2] = to_stick(0.35f * static_cast<float>(std::sin(t * 0.2)));  // la cámara barre
        publish();
    }
    return nullptr;
}


int32_t on_input(android_app*, AInputEvent* event) {
    take_over(event);
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY) {
        uint64_t bit = button_for(AKeyEvent_getKeyCode(event));
        if (!bit) return 0;
        if (AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_DOWN) {
            g_key_buttons |= bit;
            timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            g_input.latched |= bit;
            g_input.latched_until_ns = int64_t{ts.tv_sec} * 1'000'000'000 + ts.tv_nsec + 60'000'000;  // ~4 frames
        }
        else if (AKeyEvent_getAction(event) == AKEY_EVENT_ACTION_UP)
            g_key_buttons &= ~bit;
        publish();
        return 1;  // si no, Android convierte B en "atrás" y cierra la app
    }
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION &&
        (AInputEvent_getSource(event) & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        auto axis = [&](int32_t a) { return AMotionEvent_getAxisValue(event, a, 0); };
        float lx = axis(AMOTION_EVENT_AXIS_X), ly = axis(AMOTION_EVENT_AXIS_Y);
        float rx = axis(AMOTION_EVENT_AXIS_Z), ry = axis(AMOTION_EVENT_AXIS_RZ);
        g_input.sticks[0] = to_stick(lx);
        g_input.sticks[1] = to_stick(-ly);  // en Android abajo es positivo
        g_input.sticks[2] = to_stick(rx);
        g_input.sticks[3] = to_stick(-ry);
        float hx = axis(AMOTION_EVENT_AXIS_HAT_X), hy = axis(AMOTION_EVENT_AXIS_HAT_Y);
        float lt = std::fmax(axis(AMOTION_EVENT_AXIS_LTRIGGER), axis(AMOTION_EVENT_AXIS_BRAKE));
        float rt = std::fmax(axis(AMOTION_EVENT_AXIS_RTRIGGER), axis(AMOTION_EVENT_AXIS_GAS));
        uint64_t b = 0;
        if (hx < -0.5f) b |= kLeft;
        if (hx > 0.5f) b |= kRight;
        if (hy < -0.5f) b |= kUp;
        if (hy > 0.5f) b |= kDown;
        if (lt > 0.5f) b |= kZL;
        if (rt > 0.5f) b |= kZR;
        if (lx < -0.5f) b |= kStickLLeft;
        if (lx > 0.5f) b |= kStickLRight;
        if (ly < -0.5f) b |= kStickLUp;
        if (ly > 0.5f) b |= kStickLDown;
        if (rx < -0.5f) b |= kStickRLeft;
        if (rx > 0.5f) b |= kStickRRight;
        if (ry < -0.5f) b |= kStickRUp;
        if (ry > 0.5f) b |= kStickRDown;
        g_axis_buttons = b;
        publish();
        return 1;
    }
    return 0;
}

void* game_main(void*) {
    int code = nier_run(g_cfg);
    __android_log_print(ANDROID_LOG_INFO, "NieR", "nier_run terminó: %d", code);
    return nullptr;
}

NnsdkWindow g_window_state;

void on_cmd(android_app* app, int32_t cmd) {
    if (cmd == APP_CMD_INIT_WINDOW && app->window) {
        g_window_state.window = app->window;
        if (!g_cfg.native_window) {  // la primera: arranca el juego
            g_cfg.native_window = app->window;
            g_window_state.serial = g_window_state.done = 1;
            pthread_t thread;
            pthread_create(&thread, nullptr, game_main, nullptr);
            pthread_detach(thread);
        } else {
            g_window_state.serial++;  // al volver de segundo plano: el presentador crea el swapchain en la nueva
        }
    } else if (cmd == APP_CMD_TERM_WINDOW && g_cfg.native_window) {
        // Android destruye la ventana en cuanto volvemos de aquí: el presentador tiene que soltarla antes (como mucho
        // 1,5 s, para no bloquear la app; presenta cada 33 ms).
        g_window_state.window = nullptr;
        const uint32_t serial = ++g_window_state.serial;
        for (int k = 0; k < 150 && g_window_state.done.load() != serial; k++) usleep(10'000);
    }
}

// Atiende a Android hasta que cierra la actividad.
[[noreturn]] void pump(android_app* app) {
    while (!app->destroyRequested) {
        android_poll_source* source = nullptr;
        if (ALooper_pollOnce(-1, nullptr, nullptr, reinterpret_cast<void**>(&source)) >= 0 && source)
            source->process(app, source);
    }
    exit(0);
}

}  // namespace

void android_main(android_app* app) {
    if (FILE* f = fopen((std::string(app->activity->internalDataPath) + "/carpeta.txt").c_str(), "r")) {
        char line[4096] = {};
        if (fgets(line, sizeof line, f)) g_data.assign(line, strcspn(line, "\r\n"));
        fclose(f);
    }
    if (g_data.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "NieR", "sin carpeta del juego: abre la app desde su icono");
        ANativeActivity_finish(app->activity);
        pump(app);
    }
    setenv("NIER_DATA", g_data.c_str(), 1);
    setenv("NIER_TRAZA_DESDE", "94", 1);  // DEPURACIÓN: también vuelca los shaders de los draws trazados desde aquí
    setenv("NIER_VOLCAR_TRAS", "", 1);  // DEPURACIÓN: RT tras esos draws
    setenv("NIER_VOLCAR_TAM", "", 1);  // DEPURACIÓN: RT tras los draws a ese tamaño
    setenv("NIER_AUTOJUEGO", "", 1);  // DEPURACIÓN: "150": disparo y zigzag automáticos desde el segundo 150
    setenv("NIER_TEX_SUST", "", 1);  // DEPURACIÓN
    setenv("NIER_VOLCAR_TEX", "", 1);  // DEPURACIÓN
    setenv("NIER_TRAZA_LIGERA", "1", 1);
    setenv("NIER_SONDA_MEMORIA", "", 1);  // DEPURACIÓN: "1": mide los límites del driver al arrancar  // DEPURACIÓN: trazas sin volcar render targets ni shaders
    std::string log = g_data + "/log.txt";
    if (freopen(log.c_str(), "w", stdout)) {
        // Por bloques: línea a línea, cada printf era una escritura en la SD. <datos>/registro: línea a línea (para leerlo
        // en vivo, y lo último no se pierde si el proceso se cae).
        static char buffer[1 << 16];
        const bool verbose = access((g_data + "/registro").c_str(), F_OK) == 0;
        setvbuf(stdout, verbose ? nullptr : buffer, verbose ? _IOLBF : _IOFBF, verbose ? 0 : sizeof buffer);
        dup2(fileno(stdout), STDERR_FILENO);
        __android_log_print(ANDROID_LOG_INFO, "NieR", "log en %s", log.c_str());
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "NieR", "no puedo escribir %s: ¿falta el permiso de archivos?",
                            log.c_str());
    }
    g_cfg.exefs = rutas::get().exefs;
    g_cfg.nnsdk = "libnnsdk.so";
    // DEPURACIÓN: el vigilante interrumpe el hilo del juego cada 15 s para volcar su pila (un tirón y más de 100 líneas en
    // la SD cada vez): solo con <datos>/registro.
    g_cfg.watchdog_seconds = access((g_data + "/registro").c_str(), F_OK) == 0 ? 15 : 0;
    g_cfg.input = &g_input;
    g_cfg.window_state = &g_window_state;
    if (const char* a = getenv("NIER_AUTOJUEGO"); a && *a) {
        pthread_t thread;
        pthread_create(&thread, nullptr, autoplay, nullptr);
        pthread_detach(thread);
    }
    {
        pthread_t thread;
        pthread_create(&thread, nullptr, walk, nullptr);
        pthread_detach(thread);
    }

    // Como cualquier juego: la pantalla no se apaga mientras corre.
    ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON, 0);
    app->onAppCmd = on_cmd;
    app->onInputEvent = on_input;
    pump(app);
}
