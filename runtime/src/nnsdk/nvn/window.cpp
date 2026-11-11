// nvnWindow*: la ventana NVN es un swapchain de Vulkan sobre la ventana de la APK. Las texturas
// de la ventana del juego (1280x720) son nuestras; al presentar se copian a la imagen del swapchain.
// Primer paso: se presenta el último ClearColor que el juego hizo sobre la textura.

#include <android/native_window.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/hud.h"
#include "nnsdk/nvn/lsfg/framegen.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"
#include "nnsdk/runtime.h"

namespace {

// NVNwindowBuilder es opaco; solo guardamos lo que necesitamos.
struct WindowBuilder {
    void* device;
    void* native_window;
    int present_interval;
    int texture_count;
    void* const* textures;
};

// Varios juegos de command buffer, fence y semáforos: con uno, presentar el frame N esperaba a que la GPU
// acabara el N-1 (CPU y GPU por turnos: la GPU al 66 % y 22 fps en la fábrica).
constexpr int kPresentSlots = 3;
struct PresentSlot {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
};
struct Presenter {
    VkCommandPool pool = VK_NULL_HANDLE;
    PresentSlot slots[kPresentSlots];
    int next = 0;
};

Presenter g_presenter;
// Un present cada vez (g_presenter, su pool y el ritmo). No es el del ejecutor: esperar al fence de hace
// kPresentSlots frames o a una imagen del swapchain no para a los hilos del juego ni a una reproducción.
std::mutex g_present_mutex;

bool create_swapchain(nvn::Window& w) {
    nvn::VkContext* c = nvn::vk();
    if (!c) return false;
    VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    sci.window = static_cast<ANativeWindow*>(w.native_window);
    if (VkResult r = vkCreateAndroidSurfaceKHR(c->instance, &sci, nullptr, &w.surface); r != VK_SUCCESS) {
        NVN_LOG("vkCreateAndroidSurfaceKHR: %s\n", nvn::vk_result_name(r));
        return false;
    }
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->gpu, w.surface, &caps);
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->gpu, w.surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->gpu, w.surface, &count, formats.data());
    VkSurfaceFormatKHR format = formats[0];
    for (const VkSurfaceFormatKHR& f : formats)
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) format = f;

    // La pantalla de la Flip es vertical de fábrica (1080x1920) y Android la usa girada: el surface
    // da el tamaño ya en horizontal (1920x1080) con una rotación de 90°. Pretransformar obligaría a
    // dibujar girado (el blit no puede), así que se pide sin transformar y la gira el compositor.
    w.extent = caps.currentExtent;
    VkSurfaceTransformFlagBitsKHR transform = caps.currentTransform;
    if (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    NVN_LOG("swapchain %ux%u (transformación actual 0x%x, usada 0x%x)\n", w.extent.width, w.extent.height,
            caps.currentTransform, transform);
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = w.surface;
    info.minImageCount = std::max(nvn::lsfg::swapchain_images(caps), caps.minImageCount);  // 4 con LSFG posible
    info.imageFormat = format.format;
    w.format = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = w.extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.preTransform = transform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & info.compositeAlpha)) info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    if (VkResult r = vkCreateSwapchainKHR(c->device, &info, nullptr, &w.swapchain); r != VK_SUCCESS) {
        NVN_LOG("vkCreateSwapchainKHR: %s\n", nvn::vk_result_name(r));
        return false;
    }
    vkGetSwapchainImagesKHR(c->device, w.swapchain, &count, nullptr);
    w.images.resize(count);
    vkGetSwapchainImagesKHR(c->device, w.swapchain, &count, w.images.data());

    if (!g_presenter.pool) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = c->queue_family;
        vkCreateCommandPool(c->device, &pci, nullptr, &g_presenter.pool);
        for (PresentSlot& slot : g_presenter.slots) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = g_presenter.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(c->device, &ai, &slot.cmd);
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            vkCreateSemaphore(c->device, &si, nullptr, &slot.acquired);
            vkCreateSemaphore(c->device, &si, nullptr, &slot.rendered);
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(c->device, &fi, nullptr, &slot.done);
        }
    }
    NVN_LOG("swapchain %ux%u, %u imágenes, formato %d\n", w.extent.width, w.extent.height, count, format.format);
    return true;
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
}

}  // namespace

nvn::Objects<nvn::Window> nvn::g_windows;

namespace {
// DEPURACIÓN: <datos>/lsfg_margen con unos milisegundos (p. ej. "4"): margen fijo entre el generado y el real, en vez
// del que se ajusta solo. Se lee como mucho una vez por segundo; -1 si no hay fichero.
int64_t live_margin_ns() {
    static int64_t value = -1;
    static std::chrono::steady_clock::time_point next{};
    const auto now = std::chrono::steady_clock::now();
    if (now >= next) {
        next = now + std::chrono::seconds(1);
        const char* data = getenv("NIER_DATA");
        double ms = -1;
        if (FILE* f = fopen((std::string(data ? data : ".") + "/lsfg_margen").c_str(), "r")) {
            if (fscanf(f, "%lf", &ms) != 1) ms = -1;
            fclose(f);
        }
        const int64_t v = ms < 0 ? -1 : static_cast<int64_t>(ms * 1e6);
        if (v != value) {
            if (v < 0) NVN_LOG("lsfg: margen del real tras el generado: automático\n");
            else NVN_LOG("lsfg: margen del real tras el generado fijo en %.1f ms\n", ms);
        }
        value = v;
    }
    return value;
}
// El real tiene que salir un refresco después de que salga el generado, y el generado sale cuando la GPU lo acaba
// (detrás del frame del juego): desde que se manda hasta que se ve pasan L ms, que dependen de la carga. Se mide L con
// los tiempos de presentación pasados (VK_GOOGLE_display_timing) y el real se pide para "mandado + 16,7 + L - 1"; si
// un generado no llega a verse (el compositor lo tiró), se sube L.
class LsfgPacer {
public:
    int64_t margin_ns(VkDevice device, VkSwapchainKHR swapchain) {
        if (const int64_t fixed = live_margin_ns(); fixed >= 0) return fixed;
        if (!get_past_) get_past_ = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                            vkGetDeviceProcAddr(device, "vkGetPastPresentationTimingGOOGLE"));
        if (get_past_) {
            uint32_t n = 0;
            get_past_(device, swapchain, &n, nullptr);
            n = std::min<uint32_t>(n, 16);
            VkPastPresentationTimingGOOGLE past[16];
            if (n && get_past_(device, swapchain, &n, past) >= 0)
                for (uint32_t k = 0; k < n; k++) note(past[k]);
        }
        return std::max<int64_t>(latency_ns_ - 1000000, 2000000);
    }
    void queued_generated(uint32_t id, int64_t ns) { sent_[id % kRing] = {id, ns}; }

private:
    static constexpr uint32_t kRing = 64;
    struct Sent {
        uint32_t id = 0;
        int64_t ns = 0;
    };
    void note(const VkPastPresentationTimingGOOGLE& t) {
        if (!(t.presentID & 1)) {  // un real: ¿se vio el generado de delante?
            const uint32_t g = t.presentID - 1;
            if (g > last_seen_g_ && g > 1) {
                latency_ns_ = std::min<int64_t>(latency_ns_ + 4000000, 45000000);  // no salió: más margen
                if (dropped_++ < 5) NVN_LOG("lsfg: el generado %u no se vio; margen %.1f ms\n", g, latency_ns_ / 1e6);
            }
            return;
        }
        const Sent& s = sent_[t.presentID % kRing];
        if (s.id != t.presentID || !t.actualPresentTime) return;
        last_seen_g_ = t.presentID;
        const int64_t l = static_cast<int64_t>(t.actualPresentTime) - s.ns;
        if (l > 0 && l < 100000000) latency_ns_ += (l - latency_ns_) / 8;  // media móvil
    }
    PFN_vkGetPastPresentationTimingGOOGLE get_past_ = nullptr;
    Sent sent_[kRing];
    uint32_t last_seen_g_ = 0, dropped_ = 0;
    int64_t latency_ns_ = 12000000;
};
LsfgPacer g_lsfg_pacer;
}  // namespace

// La ventana de Android cambió (la app se fue a segundo plano o volvió; ver NnsdkWindow): se suelta el swapchain y la
// superficie de la vieja y, si hay una nueva, se crean en ella. Antes, al volver del escritorio la pantalla se quedaba
// en negro (se seguía presentando en la ventana destruida).
void follow_native_window(nvn::Window& w) {
    NnsdkWindow* state = nnsdk::host().window_state;
    if (!state) return;
    const uint32_t serial = state->serial.load(std::memory_order_acquire);
    if (serial == state->done.load(std::memory_order_relaxed)) return;
    void* now = state->window.load(std::memory_order_acquire);
    nvn::VkContext* c = nvn::vk();
    if (now != w.native_window || !w.swapchain) {
        {
            std::lock_guard queue_lock(nvn::vk_queue_mutex());
            vkQueueWaitIdle(c->queue);  // lo presentado en la vieja, terminado antes de soltarla
        }
        if (w.swapchain) vkDestroySwapchainKHR(c->device, w.swapchain, nullptr);
        if (w.surface) vkDestroySurfaceKHR(c->instance, w.surface, nullptr);
        w.swapchain = VK_NULL_HANDLE;
        w.surface = VK_NULL_HANDLE;
        w.images.clear();
        w.native_window = now;
        if (now) {
            const bool ok = create_swapchain(w);
            NVN_LOG("ventana nueva: swapchain %s\n", ok ? "creado" : "sin crear");
        } else {
            NVN_LOG("sin ventana (segundo plano): no se presenta\n");
        }
    }
    state->done.store(serial, std::memory_order_release);
}

void nvn::present(Window& w, int index) {
    VkContext* c = vk();
    if (!c || index < 0 || index >= static_cast<int>(w.textures.size())) return;
    std::lock_guard present_lock(g_present_mutex);
    follow_native_window(w);
    Texture* t = g_textures.get(w.textures[index]);
    // Intervalo de presentación de NVN (NieR pide 2: 30 fps, como en la Switch). FIFO solo garantiza
    // uno por vsync, así que el resto lo pone este ritmo. El reloj del juego mide tiempo real (pasos de 1/60 s,
    // entre 1 y 2 por frame): a 60 va a su velocidad, y por debajo de 30 a cámara lenta. Con <datos>/60fps se
    // fuerza el intervalo 1. Sin el lock del ejecutor: dormido con él, los hilos del juego que esperaban un sync
    // se quedaban parados hasta 33 ms.
    {
        using clock = std::chrono::steady_clock;
        static clock::time_point next = clock::now();
        static int logged_interval = 0;
        const bool fps60 = NVN_FLAG("60fps");
        const int interval = fps60 ? 1 : std::max(w.present_interval, 1);
        if (interval != logged_interval) {
            NVN_LOG("ritmo de presentación: intervalo %d%s\n", interval, fps60 ? " (forzado por <datos>/60fps)" : "");
            logged_interval = interval;
        }
        const auto period = std::chrono::microseconds(16667) * interval;
        // A intervalo 2 se despierta 2 ms antes (FIFO pone el resto). A 1, en `next`: un frame de menos de 16,7 ms
        // le cuenta al juego como uno entero y lo aceleraría un poco.
        const auto early = std::chrono::milliseconds(interval == 1 ? 0 : 2);
        auto now = clock::now();
        if (now < next - early) std::this_thread::sleep_until(next - early);
        next = std::max(next + period, clock::now() + period / 2);
    }
    if (!w.swapchain) {  // sin ventana (en segundo plano): el juego sigue a su ritmo, sin presentar
        std::lock_guard executor_lock(executor_mutex());
        next_frame();
        return;
    }
    hud::real_frame(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    // LSFG 2x encendido: antes del real, el frame intermedio (lsfg/framegen.cpp).
    uint32_t generated_id = 0;
    const int64_t generated_ns = lsfg::present_generated(w, t, &generated_id);
    if (generated_ns) g_lsfg_pacer.queued_generated(generated_id, generated_ns);
    PresentSlot& slot = g_presenter.slots[g_presenter.next];
    g_presenter.next = (g_presenter.next + 1) % kPresentSlots;

    vkWaitForFences(c->device, 1, &slot.done, VK_TRUE, UINT64_MAX);  // el de hace kPresentSlots frames
    uint32_t image = 0;
    VkResult r = vkAcquireNextImageKHR(c->device, w.swapchain, UINT64_MAX, slot.acquired, VK_NULL_HANDLE, &image);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        NVN_LOG("vkAcquireNextImageKHR: %s\n", vk_result_name(r));
        return;  // el fence sigue señalado (se reinicia después): este contexto no se queda esperando para siempre
    }
    vkResetFences(c->device, 1, &slot.done);

    // El blit lee la imagen de la textura: desde grabarlo hasta enviarlo, con el lock del ejecutor. Así, si el
    // juego la retira, se libera tras un envío del ejecutor posterior a este (va detrás en la cola).
    std::unique_lock executor_lock(executor_mutex());
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(slot.cmd, 0);
    vkBeginCommandBuffer(slot.cmd, &bi);
    {  // lo que pintaron los envíos anteriores (en vuelo, misma cola), visible para el blit
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                             &mb, 0, nullptr, 0, nullptr);
    }
    barrier(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue color{};
    if (t && t->cleared) memcpy(color.float32, t->clear_color, sizeof color.float32);
    // Lo que el juego dibujó en la textura de la ventana, escalado a la pantalla; si aún no hay
    // imagen (no se ha dibujado nada), su último color de borrado.
    if (!t || !blit_to_swapchain(*t, w.images[image], w.format, w.extent, slot.cmd)) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    hud::record(slot.cmd, w.images[image], w.extent, w.format);  // FPS y gráfica, encima del frame
    barrier(slot.cmd, w.images[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkEndCommandBuffer(slot.cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &slot.acquired;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &slot.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &slot.rendered;
    {
        std::lock_guard queue_lock(vk_queue_mutex());
        vkQueueSubmit(c->queue, 1, &si, slot.done);
    }
    executor_lock.unlock();

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &slot.rendered;
    pi.swapchainCount = 1;
    pi.pSwapchains = &w.swapchain;
    pi.pImageIndices = &image;
    // Con un generado delante, el real no antes de un refresco después: con los dos listos en el mismo refresco,
    // el compositor de Android se queda con el más nuevo y tira el otro (en pantalla seguían saliendo 30).
    // <datos>/lsfg_margen (ms) añade margen en vivo por si el generado aún no está en pantalla a esa hora.
    VkPresentTimeGOOGLE when{};
    VkPresentTimesInfoGOOGLE times{VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
    if (generated_ns && c->display_timing) {
        when.presentID = generated_id + 1;  // par: el real detrás del generado generated_id
        when.desiredPresentTime =
            static_cast<uint64_t>(generated_ns + 16666667 + g_lsfg_pacer.margin_ns(c->device, w.swapchain));
        times.swapchainCount = 1;
        times.pTimes = &when;
        pi.pNext = &times;
    }
    {
        std::lock_guard queue_lock(vk_queue_mutex());
        vkQueuePresentKHR(c->queue, &pi);
    }
    executor_lock.lock();  // el frame del ejecutor y sus volcados
    debug_dump_frame(t);
    next_frame();
    executor_lock.unlock();

    w.presented++;
    // Rendimiento: frames por segundo cada ~5 s.
    static auto window_start = std::chrono::steady_clock::now();
    static uint64_t window_frames = 0;
    window_frames++;
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - window_start).count();
    if (secs >= 5.0) {
        if (NVN_FLAG("registro")) NVN_LOG("%.1f fps (frame %llu)\n", window_frames / secs, static_cast<unsigned long long>(w.presented));
        window_start = now;
        window_frames = 0;
    }
}

extern "C" {

void nvnWindowBuilderSetDefaults(WindowBuilder* b) {
    memset(b, 0, sizeof *b);
    b->present_interval = 1;
}
void nvnWindowBuilderSetDevice(WindowBuilder* b, void* device) { b->device = device; }
void nvnWindowBuilderSetNativeWindow(WindowBuilder* b, void* native_window) { b->native_window = native_window; }
void nvnWindowBuilderSetPresentInterval(WindowBuilder* b, int interval) { b->present_interval = interval; }
void nvnWindowBuilderSetTextures(WindowBuilder* b, int count, void* const* textures) {
    b->texture_count = count;
    b->textures = textures;
}

nvn::Boolean nvnWindowInitialize(void* window, const WindowBuilder* b) {
    nvn::Window& w = nvn::g_windows.create(window);
    w.native_window = b->native_window;
    w.present_interval = b->present_interval;
    NVN_LOG("ventana con intervalo de presentación %d\n", w.present_interval);
    w.textures.assign(b->textures, b->textures + b->texture_count);
    return create_swapchain(w) ? 1 : 0;
}
void nvnWindowFinalize(void* window) { nvn::g_windows.destroy(window); }
int nvnWindowGetPresentInterval(const void* window) {
    nvn::Window* w = nvn::g_windows.get(window);
    return w ? w->present_interval : 1;
}
void nvnWindowSetPresentInterval(void* window, int interval) {
    static int last = -1;
    if (interval != last) NVN_LOG("intervalo de presentación %d\n", interval), last = interval;
    if (nvn::Window* w = nvn::g_windows.get(window)) w->present_interval = interval;
}
int nvnWindowGetNumTextures(const void* window) {
    nvn::Window* w = nvn::g_windows.get(window);
    return w ? static_cast<int>(w->textures.size()) : 0;
}
void nvnWindowSetCrop(void*, int, int, int, int) {}

}  // extern "C"

NVN_PROC(nvnWindowBuilderSetDefaults);
NVN_PROC(nvnWindowBuilderSetDevice);
NVN_PROC(nvnWindowBuilderSetNativeWindow);
NVN_PROC(nvnWindowBuilderSetPresentInterval);
NVN_PROC(nvnWindowBuilderSetTextures);
NVN_PROC(nvnWindowInitialize);
NVN_PROC(nvnWindowFinalize);
NVN_PROC(nvnWindowGetPresentInterval);
NVN_PROC(nvnWindowSetPresentInterval);
NVN_PROC(nvnWindowGetNumTextures);
NVN_PROC(nvnWindowSetCrop);
