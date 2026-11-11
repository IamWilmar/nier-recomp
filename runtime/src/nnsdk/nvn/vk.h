#pragma once
// Contexto Vulkan único detrás de NVN: instancia, GPU, dispositivo y cola gráfica.

#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

namespace nvn {

struct VkContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceMemoryProperties memory{};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    // VK_QCOM_render_pass_store_ops activa: los render passes pueden usar VK_ATTACHMENT_STORE_OP_NONE.
    bool store_op_none = false;
    // VK_GOOGLE_display_timing activa: se puede pedir a qué hora se muestra cada imagen (LSFG: el real 16,7 ms
    // después del generado, para que el compositor no tire uno de los dos).
    bool display_timing = false;
    bool push_descriptor = false;  // VK_KHR_push_descriptor activa (el escalado FSR, fsr.cpp)
    bool shader_float16 = false;  // VK_KHR_shader_float16_int8 encendida: los shaders FP16 de LSFG (lsfg/)
};

// Se crea en la primera llamada; nullptr si Vulkan no arranca.
VkContext* vk();

const char* vk_result_name(VkResult r);

}  // namespace nvn
