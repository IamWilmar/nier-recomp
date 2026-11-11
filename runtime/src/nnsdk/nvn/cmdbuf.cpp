// nvnCommandBuffer*: graba los comandos en una CommandList propia (commands.h). La memoria de
// comandos y de control que aporta el juego no se usa; el handle que recibe es el puntero a la lista.

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"

using namespace nvn;

namespace {

// Las listas grabadas viven hasta que el juego recicla la memoria de control en la que (en NVN) estaría
// su handle: al volver a dar esa memoria con AddControlMemory, sus handles dejan de ser válidos y se
// liberan. Antes no se liberaban nunca: ~4 MB/s de fuga y Android acababa matando el juego.
struct Recorded {
    const uint8_t* control;  // memoria de control vigente al grabarla
    CommandList* list;
};
struct CommandBuffer {
    CommandList* recording = nullptr;
    const uint8_t* control = nullptr;
    size_t control_size = 0;
    std::vector<Recorded> lists;
    size_t last_size = 0;  // comandos de su última grabación: qué lista del pool le viene bien
};
std::atomic<int64_t> g_live_lists{0};

Objects<CommandBuffer> g_command_buffers;

// --- Búsqueda del command buffer ---
// record() se llama 33-40k veces por frame y cada vez buscaba el command buffer (mutex + hash). Cada hilo
// recuerda el último que usó; g_generation cambia al inicializar o finalizar cualquiera, así que una dirección
// reutilizada nunca encuentra el objeto viejo. Interruptor sin_cache_cb (se mira en BeginRecording): buscarlo
// cada vez, como antes.
std::atomic<uint64_t> g_generation{1};
std::atomic<bool> g_cache_enabled{true};
struct LastBuffer {
    const void* key = nullptr;
    CommandBuffer* cb = nullptr;
    uint64_t generation = 0;
};
thread_local LastBuffer t_last;

CommandBuffer* find(const void* cmd) {
    if (!g_cache_enabled.load(std::memory_order_relaxed)) return g_command_buffers.get(cmd);
    const uint64_t generation = g_generation.load(std::memory_order_acquire);
    if (t_last.key == cmd && t_last.generation == generation) return t_last.cb;
    CommandBuffer* cb = g_command_buffers.get(cmd);
    t_last = {cmd, cb, generation};
    return cb;
}

CommandList* recording(void* cmd) {
    CommandBuffer* cb = find(cmd);
    return cb ? cb->recording : nullptr;
}

// --- Listas recicladas ---
// Una lista grande (~35k comandos, 11-14 MB) crecía desde vacía en cada BeginRecording: realloc por
// duplicación, fallos de página y munmap al borrarla, ~5 % de la CPU. Las que la cola ya reprodujo vuelven
// aquí (recycle_command_list) y BeginRecording reutiliza su memoria: clear() conserva la capacidad y, con
// Command trivial, no recorre los comandos. Cada command buffer pide una lista del tamaño de su última
// grabación; si aun así se llena, se pasa a una mayor del pool (grow) en vez de realojar. Se guardan como mucho
// kPoolLists listas y kPoolBytes; lo que sobra se borra al grabar, fuera de cualquier mutex de la cola.
// Interruptor sin_reciclar (se mira en BeginRecording): una lista nueva en cada grabación, que crece sola, y
// delete al soltarla, como antes.
constexpr size_t kPoolLists = 64, kPoolBytes = size_t{64} << 20;
std::mutex g_pool_mutex;
std::vector<CommandList*> g_pool;
size_t g_pool_bytes = 0;  // capacidad de las listas del pool
std::atomic<bool> g_recycle{true};

size_t capacity_bytes(const CommandList& list) { return list.commands.capacity() * sizeof(Command); }

// La lista del pool que mejor le viene a una grabación de unos `hint` comandos: la más pequeña en la que
// caben o, si no cabe en ninguna, la más grande. nullptr si el pool está vacío.
CommandList* take_list(size_t hint) {
    // ¿Le viene mejor una lista de capacidad a que una de capacidad b?
    auto better = [hint](size_t a, size_t b) {
        if ((a >= hint) != (b >= hint)) return a >= hint;  // gana la que tiene sitio
        return a >= hint ? a < b : a > b;                 // la más justa; sin sitio, la más grande
    };
    std::vector<CommandList*> excess;
    CommandList* best = nullptr;
    {
        std::lock_guard lock(g_pool_mutex);
        if (!g_pool.empty()) {
            size_t pick = 0;
            for (size_t k = 1; k < g_pool.size(); k++)
                if (better(g_pool[k]->commands.capacity(), g_pool[pick]->commands.capacity())) pick = k;
            best = g_pool[pick];
            g_pool[pick] = g_pool.back();
            g_pool.pop_back();
            g_pool_bytes -= capacity_bytes(*best);
        }
        // Lo que pase del tope: primero las más pequeñas (por número), luego las más grandes (por memoria).
        while (g_pool.size() > kPoolLists || g_pool_bytes > kPoolBytes) {
            const bool by_count = g_pool.size() > kPoolLists;
            size_t victim = 0;
            for (size_t k = 1; k < g_pool.size(); k++) {
                const size_t cap = g_pool[k]->commands.capacity(), victim_cap = g_pool[victim]->commands.capacity();
                if (by_count ? cap < victim_cap : cap > victim_cap) victim = k;
            }
            excess.push_back(g_pool[victim]);
            g_pool_bytes -= capacity_bytes(*g_pool[victim]);
            g_pool[victim] = g_pool.back();
            g_pool.pop_back();
        }
    }
    for (CommandList* list : excess) delete list;
    return best;
}

void drain_pool() {
    std::vector<CommandList*> lists;
    {
        std::lock_guard lock(g_pool_mutex);
        lists.swap(g_pool);
        g_pool_bytes = 0;
    }
    for (CommandList* list : lists) delete list;
}

CommandList* new_list(size_t hint) {
    if (!g_recycle.load(std::memory_order_relaxed)) {
        drain_pool();  // lo que volviera mientras se apagaba
        return new CommandList;
    }
    if (CommandList* list = take_list(hint)) {
        list->commands.clear();
        list->spilled.clear();
        return list;
    }
    return new CommandList;  // pool vacío (al arrancar): crece sola
}

// La lista que se graba está llena: se pasa a una mayor del pool (su memoria ya está en uso: sin fallos de
// página ni munmap) copiando lo grabado, que es trivial de copiar. Si no hay ninguna mayor, el vector crece
// como siempre. La lista (el handle) es la misma: solo cambia su memoria.
void grow(CommandList& list) {
    if (!g_recycle.load(std::memory_order_relaxed)) return;
    const size_t size = list.commands.size();
    CommandList* bigger = take_list(2 * size + 64);
    if (!bigger) return;
    if (bigger->commands.capacity() > size) {
        bigger->commands.clear();
        bigger->commands.insert(bigger->commands.end(), list.commands.begin(), list.commands.end());
        std::swap(list.commands, bigger->commands);  // `bigger` se queda la memoria pequeña
    }
    bigger->spilled.clear();  // lo de su grabación anterior (la de `list` sigue en `list`)
    recycle_command_list(bigger);
}

// Se construye directamente en la lista (sin un Command temporal de 344 bytes que luego se mueve).
template <typename T>
void add(CommandList& list, T&& c) {
    if (list.commands.size() == list.commands.capacity()) [[unlikely]]
        grow(list);
    list.commands.emplace_back(std::in_place_type<std::decay_t<T>>, std::forward<T>(c));
}
template <typename T>
void record(void* cmd, T&& c) {
    if (CommandList* list = recording(cmd)) add(*list, std::forward<T>(c));
}

// Copia count elementos a la SmallVector del comando (lo que no quepa, a memoria de la lista).
using RangeList = decltype(cmd::BindUniformBuffers::ranges);
static_assert(std::is_same_v<RangeList, decltype(cmd::BindVertexBuffers::ranges)>);
using AttribList = decltype(cmd::BindVertexAttribs::attribs);
using StreamList = decltype(cmd::BindVertexStreams::streams);
template <typename V, typename T>
V items(CommandList& list, const T* p, int count) {
    const size_t n = count > 0 && p ? static_cast<size_t>(count) : 0;
    static int logged = 0;  // DEPURACIÓN: si salen a menudo, la capacidad de dentro se queda corta
    if (n > V::inline_capacity() && logged < 5) {
        logged++;
        NVN_LOG("lista de %zu elementos de %zu bytes fuera del comando (caben %zu)\n", n, sizeof(T),
                V::inline_capacity());
    }
    return V(p, n, [&](size_t bytes) { return list.spill(bytes); });
}

TextureViewDesc view_of(const void* view) {
    return view ? *static_cast<const TextureViewDesc*>(view) : TextureViewDesc{};
}

}  // namespace

void nvn::recycle_command_list(CommandList* list) {
    if (!list) return;
    if (!g_recycle.load(std::memory_order_relaxed) || list->commands.capacity() == 0) {  // sin memoria: no sirve
        delete list;
        return;
    }
    std::lock_guard lock(g_pool_mutex);
    g_pool.push_back(list);
    g_pool_bytes += capacity_bytes(*list);
}

extern "C" {

nvn::Boolean nvnCommandBufferInitialize(void* cmd, void*) {
    g_command_buffers.create(cmd);
    g_generation.fetch_add(1, std::memory_order_acq_rel);  // tras crearlo: lo que se recordara de esta dirección ya no vale
    return 1;
}
void nvnCommandBufferFinalize(void* cmd) {
    if (CommandBuffer* cb = g_command_buffers.get(cmd)) {
        for (Recorded& r : cb->lists) release_command_list(r.list);
        g_live_lists -= static_cast<int64_t>(cb->lists.size());
        recycle_command_list(cb->recording);  // a medio grabar: nunca llegó a la cola
        cb->recording = nullptr;
    }
    g_command_buffers.destroy(cmd);
    g_generation.fetch_add(1, std::memory_order_acq_rel);  // tras destruirlo: ninguna búsqueda posterior lo recuerda
}
void nvnCommandBufferAddCommandMemory(void*, void*, int64_t, size_t) {}
void nvnCommandBufferAddControlMemory(void* cmd, void* memory, size_t size) {
    CommandBuffer* cb = g_command_buffers.get(cmd);
    if (!cb) return;
    const auto* begin = static_cast<const uint8_t*>(memory);
    size_t freed = 0;
    for (size_t k = 0; k < cb->lists.size();) {
        const uint8_t* c = cb->lists[k].control;
        if (c && c >= begin && c < begin + size) {
            release_command_list(cb->lists[k].list);
            cb->lists[k] = cb->lists.back();
            cb->lists.pop_back();
            freed++;
        } else {
            k++;
        }
    }
    g_live_lists -= static_cast<int64_t>(freed);
    cb->control = begin;
    cb->control_size = size;
}
size_t nvnCommandBufferGetCommandMemoryFree(const void*) { return 1 << 20; }
size_t nvnCommandBufferGetControlMemoryFree(const void*) { return 1 << 20; }
size_t nvnCommandBufferGetCommandMemoryUsed(const void*) { return 0; }
size_t nvnCommandBufferGetControlMemoryUsed(const void*) { return 0; }
void nvnCommandBufferSetMemoryCallback(void*, void*) {}
void nvnCommandBufferSetMemoryCallbackData(void*, void*) {}

void nvnCommandBufferBeginRecording(void* cmd) {
    // Interruptores: una vez por grabación, no por comando.
    const bool recycle = !NVN_FLAG("sin_reciclar");
    g_recycle.store(recycle, std::memory_order_relaxed);
    g_cache_enabled.store(!NVN_FLAG("sin_cache_cb"), std::memory_order_relaxed);
    if (CommandBuffer* cb = find(cmd)) {
        recycle_command_list(cb->recording);  // una grabación sin EndRecording: nunca llegó a la cola
        cb->recording = new_list(cb->last_size);
    }
}
uint64_t nvnCommandBufferEndRecording(void* cmd) {
    CommandBuffer* cb = find(cmd);
    if (!cb || !cb->recording) return 0;
    CommandList* list = cb->recording;
    cb->recording = nullptr;
    cb->last_size = list->commands.size();
    cb->lists.push_back({cb->control, list});
    int64_t live = ++g_live_lists;
    static int64_t next_report = 1000;
    if (live >= next_report) {  // DEPURACIÓN: si esto crece sin parar, hay listas que no se liberan
        NVN_LOG("%lld listas de comandos vivas\n", static_cast<long long>(live));
        next_report *= 2;
    }
    return reinterpret_cast<uint64_t>(list);
}

// --- Render targets y borrados ---

void nvnCommandBufferSetRenderTargets(void* cmd, int count, void* const* colors, const void* const* color_views,
                                      void* depth, const void* depth_view) {
    cmd::SetRenderTargets c{};
    c.count = count > 8 ? 8 : count;
    for (int k = 0; k < c.count; k++) {
        c.colors[k] = colors && colors[k] ? g_textures.get(colors[k]) : nullptr;
        c.color_views[k] = view_of(color_views ? color_views[k] : nullptr);
    }
    c.depth = depth ? g_textures.get(depth) : nullptr;
    c.depth_view = view_of(depth_view);
    static int logged = 0;  // DEPURACIÓN: ¿se pierde algún target de un MRT?
    if (count > 1 && logged < 20) {
        logged++;
        std::string s;
        for (int k = 0; k < count && k < 8; k++) {
            char b[64];
            snprintf(b, sizeof b, " %p->%s", colors ? colors[k] : nullptr, c.colors[k] ? "ok" : "NULO");
            s += b;
        }
        NVN_LOG("SetRenderTargets %d:%s depth %p->%s\n", count, s.c_str(), depth, c.depth ? "ok" : "NULO");
    }
    record(cmd, c);
}
void nvnCommandBufferClearColor(void* cmd, int index, const float* color, int mask) {
    cmd::ClearColor c{index, {}, mask};
    memcpy(c.color, color, sizeof c.color);
    record(cmd, c);
}
void nvnCommandBufferClearDepthStencil(void* cmd, float depth, nvn::Boolean depth_mask, int stencil, int mask) {
    record(cmd, cmd::ClearDepthStencil{depth, depth_mask != 0, stencil, mask});
}

// --- Viewport, scissor, profundidad, stencil ---

void nvnCommandBufferSetViewport(void* cmd, int x, int y, int w, int h) {
    record(cmd, cmd::Viewport{0, float(x), float(y), float(w), float(h)});
}
void nvnCommandBufferSetViewports(void* cmd, int first, int count, const float* rects) {
    for (int k = 0; k < count; k++)
        record(cmd, cmd::Viewport{first + k, rects[4 * k], rects[4 * k + 1], rects[4 * k + 2], rects[4 * k + 3]});
}
void nvnCommandBufferSetScissor(void* cmd, int x, int y, int w, int h) { record(cmd, cmd::Scissor{0, x, y, w, h}); }
void nvnCommandBufferSetScissors(void* cmd, int first, int count, const int* rects) {
    for (int k = 0; k < count; k++)
        record(cmd, cmd::Scissor{first + k, rects[4 * k], rects[4 * k + 1], rects[4 * k + 2], rects[4 * k + 3]});
}
void nvnCommandBufferSetDepthRange(void* cmd, float n, float f) { record(cmd, cmd::DepthRange{n, f}); }
// Sin equivalente directo o sin efecto visible todavía: se ignoran.
void nvnCommandBufferSetDepthBounds(void*, nvn::Boolean, float, float) {}
void nvnCommandBufferSetDepthClamp(void*, nvn::Boolean) {}
void nvnCommandBufferSetPolygonOffsetClamp(void*, float, float, float) {}
void nvnCommandBufferSetShaderScratchMemory(void*, const void*, int64_t, size_t) {}
void nvnCommandBufferSaveZCullData(void*, uint64_t, size_t) {}
void nvnCommandBufferRestoreZCullData(void*, uint64_t, size_t) {}
void nvnCommandBufferSetStencilRef(void* cmd, int faces, int ref) {
    record(cmd, cmd::Stencil{cmd::Stencil::Ref, faces, ref});
}
void nvnCommandBufferSetStencilMask(void* cmd, int faces, int mask) {
    record(cmd, cmd::Stencil{cmd::Stencil::Mask, faces, mask});
}
void nvnCommandBufferSetStencilValueMask(void* cmd, int faces, int mask) {
    record(cmd, cmd::Stencil{cmd::Stencil::ValueMask, faces, mask});
}

// --- Objetos de estado (se copian: el juego puede reutilizar el struct) ---

void nvnCommandBufferBindBlendState(void* cmd, const BlendState* s) { record(cmd, *s); }
void nvnCommandBufferBindColorState(void* cmd, const ColorState* s) { record(cmd, *s); }
void nvnCommandBufferBindChannelMaskState(void* cmd, const ChannelMaskState* s) { record(cmd, *s); }
void nvnCommandBufferBindDepthStencilState(void* cmd, const DepthStencilState* s) { record(cmd, *s); }
void nvnCommandBufferBindPolygonState(void* cmd, const PolygonState* s) { record(cmd, *s); }
void nvnCommandBufferBindMultisampleState(void* cmd, const MultisampleState* s) { record(cmd, *s); }
void nvnCommandBufferBindVertexAttribState(void* cmd, int count, const VertexAttribState* attribs) {
    if (CommandList* list = recording(cmd)) add(*list, cmd::BindVertexAttribs{items<AttribList>(*list, attribs, count)});
}
void nvnCommandBufferBindVertexStreamState(void* cmd, int count, const VertexStreamState* streams) {
    if (CommandList* list = recording(cmd)) add(*list, cmd::BindVertexStreams{items<StreamList>(*list, streams, count)});
}

// --- Programas, buffers, texturas ---

void nvnCommandBufferBindProgram(void* cmd, const void* program, int stages) {
    record(cmd, cmd::BindProgram{program ? find_program(program) : nullptr, stages});
}
void nvnCommandBufferBindVertexBuffers(void* cmd, int first, int count, const BufferRange* ranges) {
    if (CommandList* list = recording(cmd))
        add(*list, cmd::BindVertexBuffers{first, items<RangeList>(*list, ranges, count)});
}
void nvnCommandBufferBindVertexBuffer(void* cmd, int index, uint64_t address, size_t size) {
    const BufferRange range{address, size};
    if (CommandList* list = recording(cmd)) add(*list, cmd::BindVertexBuffers{index, items<RangeList>(*list, &range, 1)});
}
void nvnCommandBufferBindUniformBuffers(void* cmd, int stage, int first, int count, const BufferRange* ranges) {
    if (CommandList* list = recording(cmd))
        add(*list, cmd::BindUniformBuffers{stage, first, items<RangeList>(*list, ranges, count)});
}
void nvnCommandBufferBindUniformBuffer(void* cmd, int stage, int index, uint64_t address, size_t size) {
    const BufferRange range{address, size};
    if (CommandList* list = recording(cmd))
        add(*list, cmd::BindUniformBuffers{stage, index, items<RangeList>(*list, &range, 1)});
}
void nvnCommandBufferBindStorageBuffer(void* cmd, int stage, int index, uint64_t address, size_t size) {
    record(cmd, cmd::BindStorageBuffer{stage, index, {address, size}});
}
void nvnCommandBufferBindStorageBuffers(void* cmd, int stage, int first, int count, const BufferRange* ranges) {
    for (int k = 0; k < count; k++) record(cmd, cmd::BindStorageBuffer{stage, first + k, ranges[k]});
}
void nvnCommandBufferBindSeparateTexture(void* cmd, int stage, int index, uint64_t handle) {
    record(cmd, cmd::BindTexture{stage, index, handle, false});
}
void nvnCommandBufferBindTexture(void* cmd, int stage, int index, uint64_t handle) {
    static int logged = 0;  // DEPURACIÓN: ¿índices fuera de 0..31?
    if (index >= 32 && logged++ < 20) NVN_LOG("BindTexture etapa %d índice %d\n", stage, index);
    record(cmd, cmd::BindTexture{stage, index, handle, true});
}
void nvnCommandBufferBindSeparateSampler(void* cmd, int stage, int index, uint64_t handle) {
    record(cmd, cmd::BindSampler{stage, index, handle});
}
void nvnCommandBufferSetTexturePool(void* cmd, const void* pool) {
    record(cmd, cmd::SetPools{g_texture_pools.get(pool), nullptr});
}
void nvnCommandBufferSetSamplerPool(void* cmd, const void* pool) {
    record(cmd, cmd::SetPools{nullptr, g_sampler_pools.get(pool)});
}

// --- Draws y compute ---

void nvnCommandBufferDrawArrays(void* cmd, int primitive, int first, int count) {
    record(cmd, cmd::Draw{primitive, -1, count, first, 0, 0, 0, 1});
}
void nvnCommandBufferDrawElements(void* cmd, int primitive, int type, int count, uint64_t indices) {
    record(cmd, cmd::Draw{primitive, type, count, 0, indices, 0, 0, 1});
}
void nvnCommandBufferDrawElementsBaseVertex(void* cmd, int primitive, int type, int count, uint64_t indices,
                                            int base_vertex) {
    record(cmd, cmd::Draw{primitive, type, count, 0, indices, base_vertex, 0, 1});
}
void nvnCommandBufferDrawArraysInstanced(void* cmd, int primitive, int first, int count, int base_instance,
                                         int instances) {
    record(cmd, cmd::Draw{primitive, -1, count, first, 0, 0, base_instance, instances});
}
void nvnCommandBufferDrawElementsInstanced(void* cmd, int primitive, int type, int count, uint64_t indices,
                                           int base_vertex, int base_instance, int instances) {
    record(cmd, cmd::Draw{primitive, type, count, 0, indices, base_vertex, base_instance, instances});
}
void nvnCommandBufferDispatchCompute(void* cmd, int x, int y, int z) { record(cmd, cmd::Dispatch{x, y, z}); }
// Indirectos: los parámetros los escribe a menudo la propia GPU (un compute anterior), así que se leen
// de memoria al ejecutar, no al grabar.
void nvnCommandBufferDispatchComputeIndirect(void* cmd, uint64_t indirect) {
    record(cmd, cmd::Dispatch{0, 0, 0, indirect});
}
void nvnCommandBufferDrawArraysIndirect(void* cmd, int primitive, uint64_t indirect) {
    record(cmd, cmd::Draw{primitive, -1, 0, 0, 0, 0, 0, 1, indirect});
}
void nvnCommandBufferDrawElementsIndirect(void* cmd, int primitive, int type, uint64_t indices, uint64_t indirect) {
    record(cmd, cmd::Draw{primitive, type, 0, 0, indices, 0, 0, 1, indirect});
}

// --- Copias, barreras, contadores ---

void nvnCommandBufferCopyBufferToBuffer(void* cmd, uint64_t src, uint64_t dst, size_t size, int) {
    record(cmd, cmd::CopyBuffer{src, dst, size});
}
void nvnCommandBufferCopyTextureToTexture(void* cmd, const void* src, const void* src_view, const CopyRegion* src_region,
                                          const void* dst, const void* dst_view, const CopyRegion* dst_region, int) {
    record(cmd, cmd::CopyTexture{g_textures.get(src), view_of(src_view), *src_region, g_textures.get(dst),
                                 view_of(dst_view), *dst_region});
}
void nvnCommandBufferCopyBufferToTexture(void* cmd, uint64_t src, const void* dst, const void* dst_view,
                                         const CopyRegion* region, int) {
    record(cmd, cmd::CopyBufferToTexture{src, g_textures.get(dst), view_of(dst_view), *region});
}
void nvnCommandBufferBarrier(void* cmd, int bits) { record(cmd, cmd::Barrier{bits}); }
void nvnCommandBufferSetBlendColor(void* cmd, const float* color) {
    cmd::BlendColor c;
    memcpy(c.color, color, sizeof c.color);
    record(cmd, c);
}
void nvnCommandBufferSetRenderEnable(void* cmd, nvn::Boolean enable) { record(cmd, cmd::RenderEnable{enable != 0}); }
// Condicional a un informe de contador (oclusión): nuestros contadores no miden muestras, así que se
// dibuja siempre.
void nvnCommandBufferSetRenderEnableConditional(void* cmd, int mode, uint64_t counters) {
    record(cmd, cmd::RenderEnable{true, mode, counters});
}
void nvnCommandBufferReportValue(void* cmd, uint32_t value, uint64_t address) {
    record(cmd, cmd::ReportValue{value, address});
}
void nvnCommandBufferResetCounter(void*, int) {}
void nvnCommandBufferClearBuffer(void* cmd, uint64_t address, size_t size, uint32_t value) {
    record(cmd, cmd::ClearBuffer{address, size, value});
}
void nvnCommandBufferFenceSync(void* cmd, void* sync, int, int) { record(cmd, cmd::FenceSync{sync}); }
void nvnCommandBufferReportCounter(void* cmd, int type, uint64_t address) {
    record(cmd, cmd::ReportCounter{type, address});
}

}  // extern "C"

NVN_PROC(nvnCommandBufferInitialize);
NVN_PROC(nvnCommandBufferFinalize);
NVN_PROC(nvnCommandBufferAddCommandMemory);
NVN_PROC(nvnCommandBufferAddControlMemory);
NVN_PROC(nvnCommandBufferGetCommandMemoryFree);
NVN_PROC(nvnCommandBufferGetControlMemoryFree);
NVN_PROC(nvnCommandBufferGetCommandMemoryUsed);
NVN_PROC(nvnCommandBufferGetControlMemoryUsed);
NVN_PROC(nvnCommandBufferSetMemoryCallback);
NVN_PROC(nvnCommandBufferSetMemoryCallbackData);
NVN_PROC(nvnCommandBufferBeginRecording);
NVN_PROC(nvnCommandBufferEndRecording);
NVN_PROC(nvnCommandBufferSetRenderTargets);
NVN_PROC(nvnCommandBufferClearColor);
NVN_PROC(nvnCommandBufferClearDepthStencil);
NVN_PROC(nvnCommandBufferSetViewport);
NVN_PROC(nvnCommandBufferSetViewports);
NVN_PROC(nvnCommandBufferSetScissor);
NVN_PROC(nvnCommandBufferSetScissors);
NVN_PROC(nvnCommandBufferSetDepthRange);
NVN_PROC(nvnCommandBufferSetDepthBounds);
NVN_PROC(nvnCommandBufferSetDepthClamp);
NVN_PROC(nvnCommandBufferSetPolygonOffsetClamp);
NVN_PROC(nvnCommandBufferSetShaderScratchMemory);
NVN_PROC(nvnCommandBufferSaveZCullData);
NVN_PROC(nvnCommandBufferRestoreZCullData);
NVN_PROC(nvnCommandBufferSetStencilRef);
NVN_PROC(nvnCommandBufferSetStencilMask);
NVN_PROC(nvnCommandBufferSetStencilValueMask);
NVN_PROC(nvnCommandBufferBindBlendState);
NVN_PROC(nvnCommandBufferBindColorState);
NVN_PROC(nvnCommandBufferBindChannelMaskState);
NVN_PROC(nvnCommandBufferBindDepthStencilState);
NVN_PROC(nvnCommandBufferBindPolygonState);
NVN_PROC(nvnCommandBufferBindMultisampleState);
NVN_PROC(nvnCommandBufferBindVertexAttribState);
NVN_PROC(nvnCommandBufferBindVertexStreamState);
NVN_PROC(nvnCommandBufferBindProgram);
NVN_PROC(nvnCommandBufferBindVertexBuffers);
NVN_PROC(nvnCommandBufferBindVertexBuffer);
NVN_PROC(nvnCommandBufferBindUniformBuffers);
NVN_PROC(nvnCommandBufferBindUniformBuffer);
NVN_PROC(nvnCommandBufferBindStorageBuffer);
NVN_PROC(nvnCommandBufferBindStorageBuffers);
NVN_PROC(nvnCommandBufferBindSeparateTexture);
NVN_PROC(nvnCommandBufferBindTexture);
NVN_PROC(nvnCommandBufferBindSeparateSampler);
NVN_PROC(nvnCommandBufferSetTexturePool);
NVN_PROC(nvnCommandBufferSetSamplerPool);
NVN_PROC(nvnCommandBufferDrawArrays);
NVN_PROC(nvnCommandBufferDrawElements);
NVN_PROC(nvnCommandBufferDrawElementsBaseVertex);
NVN_PROC(nvnCommandBufferDrawArraysInstanced);
NVN_PROC(nvnCommandBufferDrawElementsInstanced);
NVN_PROC(nvnCommandBufferDispatchCompute);
NVN_PROC(nvnCommandBufferDispatchComputeIndirect);
NVN_PROC(nvnCommandBufferDrawArraysIndirect);
NVN_PROC(nvnCommandBufferDrawElementsIndirect);
NVN_PROC(nvnCommandBufferSetBlendColor);
NVN_PROC(nvnCommandBufferSetRenderEnable);
NVN_PROC(nvnCommandBufferSetRenderEnableConditional);
NVN_PROC(nvnCommandBufferReportValue);
NVN_PROC(nvnCommandBufferResetCounter);
NVN_PROC(nvnCommandBufferClearBuffer);
NVN_PROC(nvnCommandBufferFenceSync);
NVN_PROC(nvnCommandBufferCopyBufferToBuffer);
NVN_PROC(nvnCommandBufferCopyTextureToTexture);
NVN_PROC(nvnCommandBufferCopyBufferToTexture);
NVN_PROC(nvnCommandBufferBarrier);
NVN_PROC(nvnCommandBufferReportCounter);
