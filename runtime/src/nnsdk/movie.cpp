// movie:: (la librería de vídeo de Nintendo, subsdk0) sobre AMediaCodec. Es un envoltorio fino del
// MediaCodec de Android (Stagefright): mismas claves de formato ("mime", "width", "height",
// "max-input-size"), mismos índices de buffer y el mismo flag de fin de stream (4). El juego solo
// importa 17 funciones; el resto del decoder lo llama por vtable, y el orden de los métodos virtuales
// es el del vtable que exporta subsdk0 (tools/vtable.py), así el compilador genera uno idéntico.

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>

#include "nnsdk/os.h"

namespace movie {

enum DecoderType : int {};
enum DecoderOutputFormat : int {};
enum DecoderMode : int {};
using Status = int;
constexpr Status kOk = 0, kTryAgain = 1, kError = 2;

// Layout de subsdk0 (64 bytes): offset/tamaño en +0, "es dueño" en +0x10, capacidad en +0x18,
// datos en +0x28. El juego reserva el objeto; Base()/Capacity()/SetRange se leen de aquí.
class Buffer {
public:
    Buffer();
    ~Buffer();
    void SetDataAndCapacity(void* data, size_t capacity);
    void SetRange(int offset, int size);
    void* Base() const;
    size_t Capacity() const;

    int offset_ = 0, size_ = 0;
    uint64_t reserved0_ = 0;
    bool owned_ = false;
    size_t capacity_ = 0;
    uint64_t reserved1_ = 0;
    void* base_ = nullptr;
    uint64_t reserved2_[2] = {};
};
static_assert(sizeof(Buffer) == 0x40 && offsetof(Buffer, capacity_) == 0x18 && offsetof(Buffer, base_) == 0x28);

Buffer::Buffer() {}
Buffer::~Buffer() {}
void Buffer::SetDataAndCapacity(void* data, size_t capacity) {
    base_ = data;
    capacity_ = capacity;
}
void Buffer::SetRange(int offset, int size) {
    offset_ = offset;
    size_ = size;
}
void* Buffer::Base() const { return base_; }
size_t Buffer::Capacity() const { return capacity_; }

// En subsdk0 es un puntero a su almacén interno; aquí, a un mapa de claves.
class MediaData {
public:
    MediaData();
    ~MediaData();
    void Clear();
    bool FindInt32(const char* key, int* out) const;
    bool FindRect(const char* key, int* left, int* top, int* right, int* bottom) const;

    struct Value {
        bool is_string = false;
        int32_t i = 0;
        std::string s;
        int rect[4] = {};
        bool is_rect = false;
    };
    std::map<std::string, Value>* values_;
};

MediaData::MediaData() : values_(new std::map<std::string, Value>) {}
MediaData::~MediaData() { delete values_; }
void MediaData::Clear() { values_->clear(); }
bool MediaData::FindInt32(const char* key, int* out) const {
    auto it = values_->find(key);
    if (it == values_->end() || it->second.is_string || it->second.is_rect) return false;
    *out = it->second.i;
    return true;
}
bool MediaData::FindRect(const char* key, int* left, int* top, int* right, int* bottom) const {
    auto it = values_->find(key);
    if (it == values_->end() || !it->second.is_rect) return false;
    *left = it->second.rect[0], *top = it->second.rect[1], *right = it->second.rect[2], *bottom = it->second.rect[3];
    return true;
}

namespace {

void set_int(MediaData* d, const char* key, int32_t v) {
    MediaData::Value& value = (*d->values_)[key];
    value = {};
    value.i = v;
}
void set_string(MediaData* d, const char* key, const char* v) {
    MediaData::Value& value = (*d->values_)[key];
    value = {};
    value.is_string = true;
    value.s = v;
}

// Lo que el juego lee del formato de salida (main+0xa689cc): enteros, el rectángulo "crop" y las
// claves nv12-* que añade la librería de Nintendo (posición y stride de los planos Y y UV dentro del
// buffer; sin ellas el juego copia el plano UV desde un puntero basura).
void fill_from(MediaData* d, AMediaFormat* f) {
    d->Clear();
    for (const char* key : {"width", "height", "stride", "slice-height", "color-format"}) {
        int32_t v;
        if (AMediaFormat_getInt32(f, key, &v)) set_int(d, key, v);
    }
    int32_t width = 0, height = 0, stride = 0, slice = 0;
    AMediaFormat_getInt32(f, "width", &width);
    AMediaFormat_getInt32(f, "height", &height);
    if (!AMediaFormat_getInt32(f, "stride", &stride) || stride <= 0) stride = width;
    if (!AMediaFormat_getInt32(f, "slice-height", &slice) || slice <= 0) slice = height;
    set_int(d, "nv12-y-offset", 0);
    set_int(d, "nv12-y-stride", stride);
    set_int(d, "nv12-uv-offset", stride * slice);
    set_int(d, "nv12-uv-stride", stride);
    int32_t l, t, r, b;
    if (AMediaFormat_getRect(f, "crop", &l, &t, &r, &b)) {
        MediaData::Value& value = (*d->values_)["crop"];
        value = {};
        value.is_rect = true;
        value.rect[0] = l, value.rect[1] = t, value.rect[2] = r, value.rect[3] = b;
    }
}

}  // namespace

bool CreateVp9DecoderConfigurationData(int width, int height, int, MediaData* out) {
    if (!out) return false;
    set_string(out, "mime", "video/x-vnd.on2.vp9");
    set_int(out, "width", width);
    set_int(out, "height", height);
    set_int(out, "max-input-size", width * height);
    return true;
}

void SetAllocator(void* (*)(size_t, size_t, void*), void (*)(void*, void*), void* (*)(void*, size_t, void*), void*) {}
void SetCoreMask(uint64_t) {}
int GetThreadPriority() { return 16; }
void SetThreadPriority(int) {}

// Eventos que el juego espera con WaitAny (main+0xa68a54). El orden sale de qué rama atiende cada
// holder en su hilo de vídeo (main+0xa68944): el 1º lleva a AcquireInputBufferIndex, el 2º a
// AcquireOutputBufferIndex, el 3º a GetOutputFormat; el 4º es el de error.
struct DecoderEvents {
    nn::os::EventType* input_available;
    nn::os::EventType* output_available;
    nn::os::EventType* format_changed;
    nn::os::EventType* error;
};

class Decoder {
public:
    Decoder(DecoderType, DecoderOutputFormat, DecoderMode, int);
    virtual ~Decoder();
    virtual Status Configure(MediaData* format);
    virtual Status RegisterClientEvents(DecoderEvents* events);
    virtual Status GetOutputFormat(MediaData* out);
    virtual Status Start();
    virtual Status Stop();
    virtual Status Reset();
    virtual Status Flush();
    virtual Status AcquireInputBufferIndex(int* index, int* remaining);
    virtual Status GetInputBuffer(int index, Buffer* out);
    virtual Status SendInputBufferForDecode(int index, int offset, int size, long time_us, unsigned flags);
    virtual Status AcquireOutputBufferIndex(int* index, long* time_us, unsigned* flags, int* remaining);
    virtual Status GetOutputBuffer(int index, Buffer* out);
    virtual Status GetOutputBufferSize(int* a, int* b);
    virtual Status ReleaseOutputBufferIndex(int index);
    virtual Status GetLastError();
    virtual Status GetOutputBuffer(int index, Buffer* out, MediaData* format);
    virtual Status GetOutputBufferFormat(int index, MediaData* format);
    // La memoria del objeto la reserva el juego: el destructor con delete no la libera.
    static void operator delete(void*) {}

    struct Impl;  // público: lo usan los callbacks de MediaCodec

private:
    Impl* impl_;
};

struct Decoder::Impl {
    std::mutex m;
    AMediaCodec* codec = nullptr;
    DecoderEvents events{};
    std::deque<int> inputs;
    std::deque<std::pair<int, AMediaCodecBufferInfo>> outputs;
    std::map<int, AMediaCodecBufferInfo> given;  // índices de salida entregados y no liberados
    media_status_t error = AMEDIA_OK;
    bool logged_first_frame = false;
    int decoded = 0, sent = 0, released = 0;  // DEPURACIÓN

    void signal(nn::os::EventType* e) {
        if (e) nn::os::SignalEvent(e);
    }
    void clear_queues() {
        std::lock_guard lock(m);
        inputs.clear();
        outputs.clear();
        given.clear();
    }
};

namespace {

using Impl = Decoder::Impl;

void on_input(AMediaCodec*, void* user, int32_t index) {
    auto* d = static_cast<Impl*>(user);
    {
        std::lock_guard lock(d->m);
        d->inputs.push_back(index);
        static int n = 0;
        if (++n <= 40) printf("[movie] on_input %d (índice %d, en cola %zu)\n", n, index, d->inputs.size());
    }
    d->signal(d->events.input_available);
}

void on_output(AMediaCodec*, void* user, int32_t index, AMediaCodecBufferInfo* info) {
    auto* d = static_cast<Impl*>(user);
    {
        std::lock_guard lock(d->m);
        d->outputs.emplace_back(index, *info);
        static int n = 0;
        if (++n <= 40) printf("[movie] on_output %d (índice %d, t=%lld, flags %u)\n", n, index, static_cast<long long>(info->presentationTimeUs), info->flags);
        if (++d->decoded % 60 == 0)
            printf("[movie] %d frames decodificados (t=%lld us), %d entradas enviadas, %d liberados\n", d->decoded,
                   static_cast<long long>(info->presentationTimeUs), d->sent, d->released);
        if (!d->logged_first_frame) {
            d->logged_first_frame = true;
            printf("[movie] primer frame decodificado (%d bytes, t=%lld us)\n", info->size,
                   static_cast<long long>(info->presentationTimeUs));
        }
    }
    d->signal(d->events.output_available);
}

void on_format(AMediaCodec*, void* user, AMediaFormat*) { static_cast<Impl*>(user)->signal(static_cast<Impl*>(user)->events.format_changed); }

void on_error(AMediaCodec*, void* user, media_status_t error, int32_t, const char* detail) {
    auto* d = static_cast<Impl*>(user);
    printf("[movie] error de MediaCodec %d: %s\n", error, detail ? detail : "");
    d->error = error;
    d->signal(d->events.error);
}

}  // namespace

static void trace(const char* what, int n) {  // DEPURACIÓN
    if (n <= 6 || n % 100 == 0) printf("[movie] %s #%d\n", what, n);
}

Decoder::Decoder(DecoderType, DecoderOutputFormat, DecoderMode, int) : impl_(new Impl) {}

Decoder::~Decoder() {
    if (impl_->codec) {
        AMediaCodec_stop(impl_->codec);
        AMediaCodec_delete(impl_->codec);
    }
    delete impl_;
}

Status Decoder::RegisterClientEvents(DecoderEvents* events) {
    impl_->events = *events;
    return kOk;
}

Status Decoder::Configure(MediaData* format) {
    auto& values = *format->values_;
    auto mime = values.find("mime");
    if (mime == values.end() || !mime->second.is_string) return kError;
    // Reconfigurar (el juego lo repite si Start falla, y al reutilizar el decoder tras Stop) no puede dejar el
    // codec anterior vivo: es una sesión de vídeo por hardware con sus buffers, y sus callbacks seguían
    // metiendo índices viejos en las colas.
    if (impl_->codec) {
        AMediaCodec_stop(impl_->codec);
        AMediaCodec_delete(impl_->codec);
        impl_->codec = nullptr;
        impl_->clear_queues();
    }
    static int failures = 0;
    AMediaCodec* codec = AMediaCodec_createDecoderByType(mime->second.s.c_str());
    if (!codec) {
        if (failures++ < 8) printf("[movie] no hay decoder para %s\n", mime->second.s.c_str());
        return kError;
    }
    AMediaFormat* f = AMediaFormat_new();
    for (auto& [key, value] : values) {
        if (value.is_string)
            AMediaFormat_setString(f, key.c_str(), value.s.c_str());
        else if (!value.is_rect)
            AMediaFormat_setInt32(f, key.c_str(), value.i);
    }
    media_status_t r = AMediaCodec_configure(codec, f, nullptr, nullptr, 0);
    AMediaFormat_delete(f);
    if (r != AMEDIA_OK) {
        if (failures++ < 8) printf("[movie] AMediaCodec_configure: %d\n", r);
        AMediaCodec_delete(codec);
        return kError;
    }
    AMediaCodecOnAsyncNotifyCallback callbacks{on_input, on_output, on_format, on_error};
    AMediaCodec_setAsyncNotifyCallback(codec, callbacks, impl_);
    impl_->codec = codec;
    char* name = nullptr;
    AMediaCodec_getName(codec, &name);
    printf("[movie] decoder %s listo (%s)\n", mime->second.s.c_str(), name ? name : "?");
    if (name) AMediaCodec_releaseName(codec, name);
    return kOk;
}

Status Decoder::GetOutputFormat(MediaData* out) {
    static int calls = 0;
    trace("GetOutputFormat", ++calls);
    if (!impl_->codec) return kError;
    AMediaFormat* f = AMediaCodec_getOutputFormat(impl_->codec);
    fill_from(out, f);
    AMediaFormat_delete(f);
    return kOk;
}

Status Decoder::Start() {
    media_status_t r = impl_->codec ? AMediaCodec_start(impl_->codec) : AMEDIA_ERROR_UNKNOWN;
    return r == AMEDIA_OK ? kOk : kError;
}

Status Decoder::Stop() {
    if (impl_->codec) AMediaCodec_stop(impl_->codec);
    impl_->clear_queues();
    return kOk;
}

Status Decoder::Reset() { return Stop(); }

// En modo asíncrono, tras flush hay que volver a llamar a start.
Status Decoder::Flush() {
    if (!impl_->codec) return kError;
    AMediaCodec_flush(impl_->codec);
    impl_->clear_queues();
    return AMediaCodec_start(impl_->codec) == AMEDIA_OK ? kOk : kError;
}

Status Decoder::AcquireInputBufferIndex(int* index, int* remaining) {
    std::lock_guard lock(impl_->m);
    static int calls = 0, empty = 0;
    trace(impl_->inputs.empty() ? "AcquireInput vacío" : "AcquireInput", impl_->inputs.empty() ? ++empty : ++calls);
    if (impl_->inputs.empty()) return kTryAgain;
    *index = impl_->inputs.front();
    impl_->inputs.pop_front();
    *remaining = static_cast<int>(impl_->inputs.size());
    return kOk;
}

Status Decoder::GetInputBuffer(int index, Buffer* out) {
    size_t capacity = 0;
    uint8_t* data = impl_->codec ? AMediaCodec_getInputBuffer(impl_->codec, index, &capacity) : nullptr;
    if (!data) return kError;
    out->SetDataAndCapacity(data, capacity);
    out->SetRange(0, 0);
    return kOk;
}

Status Decoder::SendInputBufferForDecode(int index, int offset, int size, long time_us, unsigned flags) {
    if (!impl_->codec) return kError;
    if (impl_->sent++ < 40 || (flags & 4))
        printf("[movie] entrada %d: %d bytes t=%ld flags 0x%x\n", impl_->sent, size, time_us, flags);
    return AMediaCodec_queueInputBuffer(impl_->codec, index, offset, size, time_us, flags) == AMEDIA_OK ? kOk : kError;
}

Status Decoder::AcquireOutputBufferIndex(int* index, long* time_us, unsigned* flags, int* remaining) {
    std::lock_guard lock(impl_->m);
    static int calls = 0, empty = 0;
    trace(impl_->outputs.empty() ? "AcquireOutput vacío" : "AcquireOutput", impl_->outputs.empty() ? ++empty : ++calls);
    if (impl_->outputs.empty()) return kTryAgain;
    auto [i, info] = impl_->outputs.front();
    impl_->outputs.pop_front();
    impl_->given[i] = info;
    *index = i;
    *time_us = info.presentationTimeUs;
    *flags = info.flags;
    *remaining = static_cast<int>(impl_->outputs.size());
    return kOk;
}

// Como en la API de Nintendo, el frame se copia en la memoria del juego: antes pregunta el tamaño con
// GetOutputBufferSize, reserva ese buffer (main+0xa676a0) y lo pasa ya preparado con SetDataAndCapacity;
// luego lee los planos de su memoria con los offsets nv12-* del formato de salida.
Status Decoder::GetOutputBuffer(int index, Buffer* out) {
    size_t capacity = 0;
    uint8_t* data = impl_->codec ? AMediaCodec_getOutputBuffer(impl_->codec, index, &capacity) : nullptr;
    if (!data) return kError;
    AMediaCodecBufferInfo info;
    {
        std::lock_guard lock(impl_->m);
        info = impl_->given[index];
    }
    if (!out->Base() || !out->Capacity()) {  // sin memoria del juego: se le da la del decoder
        out->SetDataAndCapacity(data + info.offset, info.size);
        out->SetRange(0, info.size);
        return kOk;
    }
    size_t n = std::min(out->Capacity(), static_cast<size_t>(info.size));
    memcpy(out->Base(), data + info.offset, n);
    out->SetRange(0, static_cast<int>(n));
    return kOk;
}

// Tamaño del buffer que el juego debe reservar para un frame: NV12 con el stride y el alto de corte
// del decoder, redondeado a 4 KB como lo entrega el de Qualcomm (1600x904 -> 2170880).
Status Decoder::GetOutputBufferSize(int* size, int* count) {
    int32_t width = 0, height = 0, stride = 0, slice = 0;
    if (impl_->codec) {
        AMediaFormat* f = AMediaCodec_getOutputFormat(impl_->codec);
        AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_WIDTH, &width);
        AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_HEIGHT, &height);
        AMediaFormat_getInt32(f, AMEDIAFORMAT_KEY_STRIDE, &stride);
        AMediaFormat_getInt32(f, "slice-height", &slice);
        AMediaFormat_delete(f);
    }
    size_t bytes = size_t(std::max(stride, width)) * std::max(slice, height) * 3 / 2;
    {
        std::lock_guard lock(impl_->m);
        for (const auto& [i, info] : impl_->given) bytes = std::max(bytes, static_cast<size_t>(info.size));
        for (const auto& [i, info] : impl_->outputs) bytes = std::max(bytes, static_cast<size_t>(info.size));
    }
    *size = static_cast<int>((bytes + 4095) & ~size_t{4095});
    *count = 1;
    printf("[movie] tamaño de frame %d bytes (%dx%d, stride %d, corte %d)\n", *size, width, height, stride, slice);
    return kOk;
}

Status Decoder::ReleaseOutputBufferIndex(int index) {
    static int calls = 0;
    trace("ReleaseOutput", ++calls);
    {
        std::lock_guard lock(impl_->m);
        impl_->given.erase(index);
        impl_->released++;
    }
    return impl_->codec && AMediaCodec_releaseOutputBuffer(impl_->codec, index, false) == AMEDIA_OK ? kOk : kError;
}

Status Decoder::GetLastError() { return impl_->error == AMEDIA_OK ? kOk : kError; }

Status Decoder::GetOutputBuffer(int index, Buffer* out, MediaData* format) {
    Status s = GetOutputBuffer(index, out);
    return s == kOk ? GetOutputFormat(format) : s;
}

Status Decoder::GetOutputBufferFormat(int, MediaData* format) { return GetOutputFormat(format); }

}  // namespace movie
