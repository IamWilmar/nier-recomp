// nn::swkbd: el teclado de software del sistema. NieR lo abre para el nombre de usuario de la partida
// (main+0x984cb4): pasa un búfer UTF-16 de 512 bytes como texto inicial y, si ShowKeyboard tiene éxito,
// lee de ese mismo búfer el resultado (UTF-16 terminado en 0).
// Si se puede, el jugador escribe el nombre en un cuadro de texto de Android con el teclado del sistema
// (teclado_android.cpp): ShowKeyboard espera su respuesta, como el applet, y si lo cancela devuelve la cancelación del
// applet, que NieR distingue (main+0x984dfc: guarda "cancelado" y su menú de nombre vuelve atrás, main+0x681328). Sin
// actividad de Android, con <datos>/nombre.txt (las pruebas automáticas), con <datos>/sin_teclado, con el autojuego
// (NIER_AUTOJUEGO) o si el cuadro no se pudo mostrar, como siempre: se devuelve el texto inicial si ya tiene nombre, o
// el de <datos>/nombre.txt (UTF-8, una línea), o "Flip". Como el applet, el cuadro no acepta un nombre vacío ni los caracteres de invalid_char_flag
// (NieR prohíbe espacio, @, %, / y \): se vuelve a abrir con el motivo.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnsdk/teclado_android.h"
#include "nnsdk/types.h"

namespace nn::swkbd {

struct String {
    void* ptr;
    size_t size;  // capacidad del búfer en bytes
};

// El config del SDK (el mismo que en MH Rise). NieR pone a cero los 0x500 bytes del ShowKeyboardArg, llama a
// MakePreset(UserName) y escribe a mano (main+0x984d10) el modo de teclado, invalid_char_flag, text_max_length (lo
// que pide su menú, 9 o más), input_form_mode, el fondo borroso y el work buffer; SetGuideTextUtf8 con nullptr. No
// pone cabecera, subtexto, texto del botón ni text check.
struct KeyboardConfig {
    uint32_t keyboard_mode;
    char16_t ok_text[9];
    char16_t left_optional_symbol_key;
    char16_t right_optional_symbol_key;
    bool is_prediction_enabled;
    uint32_t invalid_char_flag;
    int32_t initial_cursor_pos;
    char16_t header_text[65];
    char16_t sub_text[129];
    char16_t guide_text[257];
    int32_t text_max_length;  // 0x3ac
    int32_t text_min_length;
    uint32_t password_mode;
    uint32_t input_form_mode;
    bool is_use_new_line;
    bool is_use_utf8;  // 0x3bd: el resultado en UTF-8 y no en UTF-16 (NieR no lo pide)
    bool is_use_blur_background;
    int32_t initial_string_offset;  // 0x3c0
    int32_t initial_string_length;
    int32_t user_dictionary_offset;
    int32_t user_dictionary_num;
    bool is_use_text_check;  // 0x3d0
    void* text_check_callback;
    int32_t separate_text_pos[8];  // 0x3e0
    uint32_t customized_dictionaries[24][2];
    uint8_t customized_dictionary_count;
    uint8_t reserved[0xf];
};
static_assert(sizeof(KeyboardConfig) == 0x4d0);
static_assert(offsetof(KeyboardConfig, guide_text) == 0x1a8);
static_assert(offsetof(KeyboardConfig, text_max_length) == 0x3ac);
static_assert(offsetof(KeyboardConfig, text_check_callback) == 0x3d8);

struct ShowKeyboardArg {
    KeyboardConfig keyboard_config;
    void* work_buf;  // 0x4d0
    size_t work_buf_size;
    void* text_check_work_buf;
    size_t text_check_work_buf_size;
    void* customized_dictionary_buf;
    size_t customized_dictionary_buf_size;
};
static_assert(sizeof(ShowKeyboardArg) == 0x500);

namespace {

// Lo que devuelve el applet cuando el jugador cancela (módulo 159, descripción 1): NieR compara con 0x29f.
constexpr Result kResultCanceled{159 | 1 << 9};

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

size_t length16(const char16_t* s, size_t max = SIZE_MAX) {
    size_t n = 0;
    while (s && n < max && s[n]) n++;
    return n;
}

std::u16string utf8_to_utf16(const std::string& in) {
    std::u16string out;
    for (size_t i = 0; i < in.size();) {
        const unsigned char c = in[i];
        const int extra = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
        if (extra < 0 || i + extra >= in.size()) {  // byte suelto o secuencia cortada: se salta
            i++;
            continue;
        }
        uint32_t cp = extra ? c & (0x3F >> extra) : c;
        for (int k = 1; k <= extra; k++) cp = cp << 6 | (in[i + k] & 0x3F);
        i += 1 + extra;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out += char16_t(0xD800 | cp >> 10);
            out += char16_t(0xDC00 | (cp & 0x3FF));
        } else {
            out += char16_t(cp);
        }
    }
    return out;
}

std::string utf16_to_utf8(const char16_t* in, size_t n) {
    std::string out;
    for (size_t i = 0; i < n && in[i]; i++) {
        uint32_t cp = in[i];
        if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < n && in[i + 1] >= 0xDC00 && in[i + 1] < 0xE000)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (in[++i] - 0xDC00);
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | cp >> 6);
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | cp >> 12);
            out += char(0x80 | (cp >> 6 & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | cp >> 18);
            out += char(0x80 | (cp >> 12 & 0x3F));
            out += char(0x80 | (cp >> 6 & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

std::u16string too_short_message(int min) {
    const bool es = android_ui::en_espanol();
    if (min <= 1) return utf8_to_utf16(es ? "Escribe un nombre." : "Enter a name.");
    char text[96];
    snprintf(text, sizeof text, es ? "Escribe al menos %d caracteres." : "Enter at least %d characters.", min);
    return utf8_to_utf16(text);
}

// Los caracteres que el applet no deja escribir según invalid_char_flag (NieR pone 0x3e: espacio, @, %, / y \).
std::u16string invalid_chars(uint32_t flags) {
    std::u16string chars;
    if (flags & (1u << 1)) chars += u' ';
    if (flags & (1u << 2)) chars += u'@';
    if (flags & (1u << 3)) chars += u'%';
    if (flags & (1u << 4)) chars += u'/';
    if (flags & (1u << 5)) chars += u'\\';
    if (flags & (1u << 6)) chars += u"0123456789";
    return chars;
}

std::u16string invalid_chars_message(const std::u16string& chars) {
    const bool es = android_ui::en_espanol();
    std::string list;
    for (char16_t c : chars) {
        if (c >= u'0' && c <= u'9') {
            if (c == u'0') list += es ? " números" : " numbers";
            continue;
        }
        list += ' ';
        list += c == u' ' ? (es ? "espacios" : "spaces") : std::string(1, char(c));
    }
    return utf8_to_utf16((es ? "No se admiten:" : "Not allowed:") + list);
}

}  // namespace

// Con el tipo Preset del SDK, para que el import del juego (…MakePreset…NS0_6PresetE) llegue aquí y no a un stub. No hace
// nada: NieR ya puso el config a cero y escribe encima lo que usa.
enum Preset : int { Preset_Default = 0, Preset_Password = 1, Preset_UserName = 2, Preset_DownloadCode = 3 };
void MakePreset(KeyboardConfig*, Preset) {}
size_t GetRequiredWorkBufferSize(bool) { return 0x1000; }
// Como el SDK: el texto en UTF-16 en guide_text (hasta 256 caracteres); con nullptr lo deja vacío.
void SetGuideTextUtf8(KeyboardConfig* config, const char* text) {
    const std::u16string guide = text ? utf8_to_utf16(text) : std::u16string();
    const size_t n = std::min<size_t>(guide.size(), 256);
    if (n) memcpy(config->guide_text, guide.data(), n * sizeof(char16_t));
    config->guide_text[n] = 0;
}
void SetInitialText(ShowKeyboardArg*, const char16_t* text) { g_initial = text; }

Result ShowKeyboard(String* out, const ShowKeyboardArg& arg) {
    auto* dst = static_cast<char16_t*>(out->ptr);
    const size_t capacity = out->size / sizeof(char16_t);
    if (!dst || capacity < 2) return ResultSuccess;

    // El jugador escribe en el cuadro de Android hasta que acepta un nombre o cancela. Con el autojuego (bot de
    // pruebas) no: nadie lo contestaría y el hilo del juego se quedaría esperando.
    const char* autojuego = getenv("NIER_AUTOJUEGO");
    if (android_ui::disponible() && !(autojuego && *autojuego)) {
        const KeyboardConfig& config = arg.keyboard_config;
        android_ui::Pedido p;
        const std::u16string header(config.header_text, length16(config.header_text, 65));
        const std::u16string guide(config.guide_text, length16(config.guide_text, 257));
        // NieR no trae ni cabecera ni guía: el título dice qué se pide (es el preset UserName).
        p.titulo = !header.empty() ? header
                   : !guide.empty() ? guide
                                    : utf8_to_utf16(android_ui::en_espanol() ? "Nombre" : "Name");
        p.guia = header.empty() ? std::u16string() : guide;
        p.mensaje.assign(config.sub_text, length16(config.sub_text, 129));
        if (g_initial) p.inicial.assign(g_initial, length16(g_initial, 256));  // su búfer es de 512 bytes
        p.ok.assign(config.ok_text, length16(config.ok_text, 9));
        // El máximo del config (el que pide su menú) sin pasar de lo que cabe en el String con el 0.
        p.max = int(capacity - 1);
        if (config.text_max_length > 0) p.max = std::min(p.max, int(config.text_max_length));
        p.clave = config.password_mode != 0;
        // Un nombre vacío no: como mínimo un carácter, aunque el config diga 0.
        const int min = std::max(int(config.text_min_length), 1);
        printf("[swkbd] teclado de Android: de %d a %d caracteres\n", min, p.max);
        for (;;) {
            std::u16string text;
            const android_ui::Estado e = android_ui::preguntar(p, &text);
            if (e == android_ui::Fallo) break;  // no se pudo mostrar: como siempre
            if (e == android_ui::Cancelado) {
                dst[0] = 0;
                printf("[swkbd] cancelado por el jugador\n");
                return kResultCanceled;
            }
            p.inicial = text;
            if (text.size() < size_t(min) || text.find_first_not_of(u' ') == std::u16string::npos) {
                p.mensaje = too_short_message(min);
                continue;
            }
            if (const std::u16string prohibidos = invalid_chars(config.invalid_char_flag);
                text.find_first_of(prohibidos) != std::u16string::npos) {
                p.mensaje = invalid_chars_message(prohibidos);
                continue;
            }
            if (text.size() > size_t(p.max)) text.resize(size_t(p.max));
            memcpy(dst, text.data(), text.size() * sizeof(char16_t));
            dst[text.size()] = 0;
            printf("[swkbd] nombre: %s\n", utf16_to_utf8(text.data(), text.size()).c_str());
            return ResultSuccess;
        }
    }

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
