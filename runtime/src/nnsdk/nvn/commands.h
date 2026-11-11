#pragma once
// Comandos que graba un NVNcommandBuffer. Se guardan tal cual (con copias de los objetos de estado
// y de las vistas, que el juego puede reutilizar); el ejecutor los reproduce al enviar a la cola, con
// el estado de la cola persistiendo entre listas, como en NVN.

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <variant>
#include <vector>

#include "nnsdk/nvn/objects.h"
#include "nnsdk/nvn/state.h"

namespace nvn {

struct BufferRange {
    uint64_t address;
    uint64_t size;
};
struct CopyRegion {
    int x, y, z, width, height, depth;
};

// Lista corta de T guardada dentro del propio comando (antes un std::vector: ~13k reservas de memoria por
// frame, sobre todo BindUniformBuffers). Si no caben N, los elementos van a memoria de la lista de comandos
// (CommandList::spill), que vive lo mismo que el comando. Sin destructor ni reservas propias: Command es
// trivial de copiar y de destruir, y vaciar una lista no tiene que recorrerla. Copiar una SmallVector copia el
// puntero a esa memoria: la copia no debe vivir más que su lista (el ejecutor no guarda comandos de estos).
template <typename T, uint32_t N>
class SmallVector {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    SmallVector() = default;
    // spill(bytes) da memoria que vive lo mismo que el comando; solo se pide si n > N.
    template <typename Spill>
    SmallVector(const T* items, size_t n, Spill&& spill) : size_(static_cast<uint32_t>(n)) {
        T* out = inline_;
        if (n > N) out = external_ = static_cast<T*>(spill(n * sizeof(T)));
        if (n) memcpy(out, items, n * sizeof(T));
    }
    static constexpr size_t inline_capacity() { return N; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T* data() const { return size_ <= N ? inline_ : external_; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + size_; }
    const T& operator[](size_t k) const { return data()[k]; }
    // El estado del ejecutor guarda std::vector (`st_.attribs = v.attribs`): se copia a un vector del hilo y la
    // asignación reutiliza la capacidad del destino, sin reservar memoria. La referencia vale hasta la siguiente
    // conversión de este mismo tipo en el mismo hilo: no se guarda.
    operator const std::vector<T>&() const {
        thread_local std::vector<T> scratch;
        scratch.assign(begin(), end());
        return scratch;
    }

private:
    uint32_t size_ = 0;
    union {
        T inline_[N];
        T* external_;
    };
};

namespace cmd {

struct SetRenderTargets {
    int count;
    Texture* colors[8];
    TextureViewDesc color_views[8];
    Texture* depth;
    TextureViewDesc depth_view;
};
struct ClearColor {
    int index;
    float color[4];
    int mask;
};
struct ClearDepthStencil {
    float depth;
    bool depth_mask;
    int stencil, stencil_mask;
};
struct Viewport {
    int index;
    float x, y, w, h;
};
struct Scissor {
    int index;
    int x, y, w, h;
};
struct DepthRange {
    float n, f;
};
struct BindProgram {
    Program* program;
    int stages;
};
// Capacidades dentro del comando: cada struct de estos queda en ≤ 128 bytes (lo que ocupa CopyTexture), para
// que no sean ellos los que fijen el tamaño de Command si SetRenderTargets deja algún día de hacerlo.
struct BindVertexBuffers {
    int first;
    SmallVector<BufferRange, 6> ranges;
};
struct BindUniformBuffers {
    int stage, first;
    SmallVector<BufferRange, 6> ranges;
};
struct BindStorageBuffer {
    int stage, index;
    BufferRange range;
};
struct BindTexture {  // separada o combinada (sampler en los 32 bits altos)
    int stage, index;
    uint64_t handle;
    bool combined;
};
struct BindSampler {
    int stage, index;
    uint64_t handle;
};
struct BindVertexAttribs {
    SmallVector<VertexAttribState, 16> attribs;
};
struct BindVertexStreams {
    SmallVector<VertexStreamState, 8> streams;
};
struct SetPools {
    DescriptorPool* textures;
    DescriptorPool* samplers;
};
struct Stencil {
    enum What { Ref, Mask, ValueMask } what;
    int faces, value;
};
struct Draw {
    int primitive;
    int index_type;  // -1: sin índices
    int count, first;
    uint64_t index_address;
    int base_vertex, base_instance, instances;
    uint64_t indirect = 0;  // si no es 0, los parámetros están en esta dirección (se leen al ejecutar)
};
struct Dispatch {
    int x, y, z;
    uint64_t indirect = 0;  // {x, y, z} en esta dirección
};
struct BlendColor {
    float color[4];
};
struct RenderEnable {
    bool enable;
    int mode = -1;             // condicional: NVNconditionalRenderMode (-1: no lo es)
    uint64_t counters = 0;     // condicional: dirección de los dos contadores que compara
};
struct ReportValue {
    uint32_t value;
    uint64_t address;
};
struct ClearBuffer {
    uint64_t address, size;
    uint32_t value;
};
struct FenceSync {
    void* sync;
};
struct CopyBuffer {
    uint64_t src, dst, size;
};
struct CopyTexture {
    Texture* src;
    TextureViewDesc src_view;
    CopyRegion src_region;
    Texture* dst;
    TextureViewDesc dst_view;
    CopyRegion dst_region;
};
struct CopyBufferToTexture {
    uint64_t src;
    Texture* dst;
    TextureViewDesc dst_view;
    CopyRegion region;
};
struct Barrier {
    int bits;
};
struct ReportCounter {
    int type;
    uint64_t address;
};

}  // namespace cmd

using Command = std::variant<cmd::SetRenderTargets, cmd::ClearColor, cmd::ClearDepthStencil, cmd::Viewport,
                             cmd::Scissor, cmd::DepthRange, cmd::BindProgram, cmd::BindVertexBuffers,
                             cmd::BindUniformBuffers, cmd::BindStorageBuffer, cmd::BindTexture, cmd::BindSampler,
                             BlendState, ColorState, ChannelMaskState, DepthStencilState, PolygonState,
                             MultisampleState, cmd::BindVertexAttribs, cmd::BindVertexStreams, cmd::SetPools,
                             cmd::Stencil, cmd::Draw, cmd::Dispatch, cmd::CopyBuffer, cmd::CopyTexture,
                             cmd::CopyBufferToTexture, cmd::Barrier, cmd::ReportCounter, cmd::BlendColor,
                             cmd::RenderEnable, cmd::ReportValue, cmd::ClearBuffer, cmd::FenceSync>;

// Las listas se reciclan (cmdbuf.cpp) y vaciarlas no recorre sus ~35k comandos: que siga siendo así.
static_assert(std::is_trivially_destructible_v<Command> && std::is_trivially_copyable_v<Command>);
// Las listas cortas no hacen crecer Command (lo fija SetRenderTargets).
static_assert(sizeof(cmd::BindVertexBuffers) <= sizeof(cmd::SetRenderTargets) &&
              sizeof(cmd::BindUniformBuffers) <= sizeof(cmd::SetRenderTargets) &&
              sizeof(cmd::BindVertexAttribs) <= sizeof(cmd::SetRenderTargets) &&
              sizeof(cmd::BindVertexStreams) <= sizeof(cmd::SetRenderTargets));

struct CommandList {
    std::vector<Command> commands;
    // Los elementos de las SmallVector que no caben en su comando: viven lo mismo que la lista.
    std::vector<std::unique_ptr<uint64_t[]>> spilled;
    void* spill(size_t bytes) { return spilled.emplace_back(new uint64_t[(bytes + 7) / 8]).get(); }
};

// Ejecuta en Vulkan las listas que el juego envía a la cola (executor.cpp).
void execute(const std::vector<const CommandList*>& lists);
// Presentar en orden con los envíos (la cola de envíos asíncrona) y soltar una lista que el juego recicla.
void queue_present(Window& window, int index);
void release_command_list(CommandList* list);
// Una lista que ya no se va a reproducir (la cola ya la soltó): vuelve al pool de cmdbuf.cpp para que la
// próxima grabación reutilice su memoria. Barato: se puede llamar con el mutex de la cola tomado (no vacía ni
// libera nada ahí; con el interruptor sin_reciclar la borra, como antes).
void recycle_command_list(CommandList* list);
// Afinidad (os.cpp): el hilo que llama pasa a los núcleos grandes junto con los demás hilos críticos (el
// principal del juego, MainDraw y NvnSubmit). refresh_critical_threads, una vez por frame desde el hilo que
// presenta: como mucho una vez por segundo mira el interruptor sin_afinidad, rehace las máscaras que se hayan
// perdido o cambiado y devuelve la completa a los hilos no críticos que heredaron la de los grandes.
void pin_critical_thread();
void refresh_critical_threads();
// Copia la textura de la ventana a la imagen del swapchain (la llama window.cpp al presentar).
// Con FSR 1 (fsr.cpp) si el formato lo permite; si no, un blit lineal.
bool blit_to_swapchain(Texture& texture, VkImage swapchain_image, VkFormat format, VkExtent2D extent, VkCommandBuffer cmd);
// Se presentó un frame (window.cpp): las texturas que escribe la CPU se revisan una vez por frame.
void next_frame();
// DEPURACIÓN: en ciertos frames vuelca a la SD la textura presentada y los render targets del frame.
void debug_dump_frame(Texture* presented);
// Envíos asíncronos (executor.cpp): número del último envío, espera a que termine uno (false si vence
// el plazo; no espera a que acabe la reproducción de otro) y espera a que terminen todos.
uint64_t last_submit();
// Un NVNsync queda señalado al terminar ese envío (queue.cpp; FenceSync de la cola y del command buffer).
void sync_signal_at(void* sync, uint64_t submit);
bool wait_submit(uint64_t submit, uint64_t timeout_ns);
void wait_idle();
// La VkQueue no admite accesos concurrentes: todo vkQueueSubmit/vkQueuePresentKHR pasa por este mutex (solo
// la llamada: no se espera nada con él). Va después de executor_mutex() si se toman los dos.
std::mutex& vk_queue_mutex();
// Lo que el juego suelta (la imagen de una textura, un sampler) puede seguir en un envío pendiente: se
// destruye cuando la GPU termina el último envío hecho hasta ahora; con la imagen caen sus vistas y
// framebuffers. Quien toca un objeto que el ejecutor puede estar leyendo toma executor_mutex() (el mismo
// que serializa la ejecución de los comandos).
std::recursive_mutex& executor_mutex();
void retire_texture_gpu(Texture& t);  // deja la textura sin imagen
void retire_sampler(VkSampler sampler);

}  // namespace nvn
