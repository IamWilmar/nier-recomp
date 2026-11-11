// nn::codec::HardwareOpusDecoder. El juego solo importa el constructor: el resto va por vtable, y
// el orden de los métodos virtuales es el del vtable que exporta el sdk (tools/vtable.py), así el
// compilador genera uno idéntico. El juego reserva 64 bytes para el objeto (main+0xd8d734).
//
// De momento decodifica silencio de la duración correcta: cada paquete trae la cabecera de Nintendo
// (tamaño y rango final, big-endian) y la duración sale del byte TOC de Opus. Así el reloj de audio
// de Wwise avanza igual que con sonido. El sonido real llegará con libopus.

#include <cstdint>
#include <cstring>

#include "nnsdk/types.h"

namespace {

uint32_t be32(const uint8_t* p) { return uint32_t{p[0]} << 24 | uint32_t{p[1]} << 16 | uint32_t{p[2]} << 8 | p[3]; }

// Muestras por canal de un paquete Opus (RFC 6716, 3.1): duración del frame por la config del TOC
// y número de frames por el código del TOC.
int opus_samples(const uint8_t* packet, size_t size, int sample_rate) {
    if (size == 0) return 0;
    int config = packet[0] >> 3;
    int tenths_ms;  // duración de un frame en décimas de ms
    if (config < 12)
        tenths_ms = (const int[]){100, 200, 400, 600}[config & 3];
    else if (config < 16)
        tenths_ms = (config & 1) ? 200 : 100;
    else
        tenths_ms = (const int[]){25, 50, 100, 200}[config & 3];
    int code = packet[0] & 3;
    int frames = code == 0 ? 1 : code < 3 ? 2 : (size > 1 ? packet[1] & 0x3F : 0);
    return frames * tenths_ms * sample_rate / 10000;
}

}  // namespace

namespace nn::codec {

class HardwareOpusDecoder {
public:
    explicit HardwareOpusDecoder(int);
    virtual ~HardwareOpusDecoder();
    virtual size_t GetWorkBufferSize(int sample_rate, int channels) const;
    virtual bool IsInitialized() const;
    virtual Result Initialize(int sample_rate, int channels, void* work, size_t work_size);
    virtual Result Finalize();
    virtual int GetSampleRate() const;
    virtual int GetChannelCount() const;
    virtual Result Reset();
    virtual Result DecodeInterleaved(size_t* consumed, int* samples, int16_t* out, size_t out_size, const void* in,
                                     size_t in_size);
    // La memoria del objeto la reserva el juego: el destructor con delete no la libera.
    static void operator delete(void*) {}

private:
    int sample_rate_ = 0, channels_ = 0;
    bool initialized_ = false;
};

HardwareOpusDecoder::HardwareOpusDecoder(int) {}
HardwareOpusDecoder::~HardwareOpusDecoder() {}
size_t HardwareOpusDecoder::GetWorkBufferSize(int, int) const { return 64 * 1024; }
bool HardwareOpusDecoder::IsInitialized() const { return initialized_; }

Result HardwareOpusDecoder::Initialize(int sample_rate, int channels, void*, size_t) {
    sample_rate_ = sample_rate;
    channels_ = channels;
    initialized_ = true;
    return ResultSuccess;
}

Result HardwareOpusDecoder::Finalize() {
    initialized_ = false;
    return ResultSuccess;
}

int HardwareOpusDecoder::GetSampleRate() const { return sample_rate_; }
int HardwareOpusDecoder::GetChannelCount() const { return channels_; }
Result HardwareOpusDecoder::Reset() { return ResultSuccess; }

Result HardwareOpusDecoder::DecodeInterleaved(size_t* consumed, int* samples, int16_t* out, size_t out_size,
                                              const void* in, size_t in_size) {
    auto* data = static_cast<const uint8_t*>(in);
    size_t packet_size = in_size >= 8 ? be32(data) : 0;
    const uint8_t* packet = data + 8;
    size_t used = 8 + packet_size;
    if (in_size < 8 || used > in_size) {  // sin cabecera de Nintendo: el paquete entero
        packet = data;
        packet_size = in_size;
        used = in_size;
    }
    int n = opus_samples(packet, packet_size, sample_rate_);
    size_t bytes = static_cast<size_t>(n) * channels_ * sizeof(int16_t);
    if (bytes > out_size) {
        n = static_cast<int>(out_size / (channels_ * sizeof(int16_t)));
        bytes = out_size;
    }
    memset(out, 0, bytes);
    *consumed = used;
    *samples = n;
    return ResultSuccess;
}

}  // namespace nn::codec
