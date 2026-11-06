// PrismRT.cpp — zero-dependency Win32 host for the spectral prism ray tracer.
//
// Drives the PrismRT shaders on VK_KHR_ray_tracing_pipeline: hardware-traced
// spectral dispersion with single-scattering fog, so the white beam and the
// dispersed rainbow fan are visible in air (a beam in vacuum is invisible —
// the fog is what makes the classic prism picture exist in 3D).
//
// Zero dependencies: vulkan-1.dll is loaded at runtime (LoadLibrary +
// vkGetInstanceProcAddr), so nothing links against the Vulkan SDK. The SDK is
// only needed at build time for glslc (compile_shaders.bat). The numbers the
// host and the shaders share (bindings, grid sizes, instance ids) live in
// layout.h, which both include.
//
// Single frame in flight, TLAS rebuilt per frame (three instances — trivial),
// progressive XYZ accumulation reset on any camera/scene change, resolve
// compute pass -> blit to swapchain.
//
//   LMB drag   orbit camera        wheel      zoom
//   RMB drag   rotate prism (dx: twist Y, dy: incidence Z)
//   G          SF10 <-> BK7        F          cycle fog density
//   + / -      exposure            R          reset accumulation
//   Esc        quit
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan.h>

#include "layout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0
#endif

namespace {

// ============================ tiny math ================================

struct V3 {
    float x, y, z;
    constexpr V3 operator+(V3 o) const { return { x + o.x, y + o.y, z + o.z }; }
    constexpr V3 operator-(V3 o) const { return { x - o.x, y - o.y, z - o.z }; }
    constexpr V3 operator*(float s) const { return { x * s, y * s, z * s }; }
    constexpr V3 operator-() const { return { -x, -y, -z }; }
};
constexpr float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr V3 cross(V3 a, V3 b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
V3 normalize(V3 a) { return a * (1.0f / std::sqrt(dot(a, a))); }

// column-major 4x4, matching GLSL std140 mat4: m[column * 4 + row]
struct M4 {
    std::array<float, 16> m{};
    float& at(int row, int col) { return m[static_cast<size_t>(col * 4 + row)]; }
    float at(int row, int col) const { return m[static_cast<size_t>(col * 4 + row)]; }
};
M4 mul(const M4& a, const M4& b) {
    M4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.at(row, k) * b.at(k, c);
            r.at(row, c) = s;
        }
    return r;
}

// Inverse of the view matrix of a camera at `eye` looking at `at`: the
// camera basis as columns, the eye as translation. Rays are generated in
// the shader from clip space, so only the inverse is ever needed.
M4 inverseLookAt(V3 eye, V3 at, V3 up) {
    V3 f = normalize(at - eye), s = normalize(cross(f, up)), u = cross(s, f);
    M4 r;
    r.at(0, 0) = s.x;  r.at(1, 0) = s.y;  r.at(2, 0) = s.z;
    r.at(0, 1) = u.x;  r.at(1, 1) = u.y;  r.at(2, 1) = u.z;
    r.at(0, 2) = -f.x; r.at(1, 2) = -f.y; r.at(2, 2) = -f.z;
    r.at(0, 3) = eye.x; r.at(1, 3) = eye.y; r.at(2, 3) = eye.z; r.at(3, 3) = 1.0f;
    return r;
}

// Inverse of the Vulkan perspective (depth 0..1, y flipped):
//   P = [f/a 0 0 0; 0 -f 0 0; 0 0 A B; 0 0 -1 0],  A = zf/(zn-zf), B = zn*zf/(zn-zf)
M4 inversePerspective(float fovyDeg, float aspect, float zn, float zf) {
    const float f = 1.0f / std::tan(fovyDeg * std::numbers::pi_v<float> / 360.0f);
    const float A = zf / (zn - zf), B = zn * zf / (zn - zf);
    M4 r;
    r.at(0, 0) = aspect / f;
    r.at(1, 1) = -1.0f / f;
    r.at(2, 3) = -1.0f;
    r.at(3, 2) = 1.0f / B;
    r.at(3, 3) = A / B;
    return r;
}

constexpr float degToRad(float d) { return d * std::numbers::pi_v<float> / 180.0f; }

// ===================== Vulkan dynamic loader ============================

PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
PFN_vkGetDeviceProcAddr   vkGetDeviceProcAddr;

#define VK_GLOBAL_FNS(X) \
    X(vkCreateInstance)
#define VK_INSTANCE_FNS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) \
    X(vkCreateWin32SurfaceKHR) \
    X(vkDestroySurfaceKHR) \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfaceSupportKHR)
#define VK_DEVICE_FNS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkDeviceWaitIdle) \
    X(vkQueueWaitIdle) \
    X(vkCreateSwapchainKHR) \
    X(vkDestroySwapchainKHR) \
    X(vkGetSwapchainImagesKHR) \
    X(vkAcquireNextImageKHR) \
    X(vkQueuePresentKHR) \
    X(vkQueueSubmit) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkResetCommandBuffer) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkCreateSemaphore) \
    X(vkDestroySemaphore) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
    X(vkCreateImage) \
    X(vkDestroyImage) \
    X(vkGetImageMemoryRequirements) \
    X(vkBindImageMemory) \
    X(vkCreateImageView) \
    X(vkDestroyImageView) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdDispatch) \
    X(vkCmdPushConstants) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdBlitImage) \
    X(vkCreateRayTracingPipelinesKHR) \
    X(vkGetRayTracingShaderGroupHandlesKHR) \
    X(vkCmdTraceRaysKHR) \
    X(vkCmdFillBuffer) \
    X(vkCreateAccelerationStructureKHR) \
    X(vkDestroyAccelerationStructureKHR) \
    X(vkGetAccelerationStructureBuildSizesKHR) \
    X(vkCmdBuildAccelerationStructuresKHR) \
    X(vkGetAccelerationStructureDeviceAddressKHR) \
    X(vkGetBufferDeviceAddress)

#define VKFN_DECL(n) PFN_##n n;
VK_GLOBAL_FNS(VKFN_DECL)
VK_INSTANCE_FNS(VKFN_DECL)
VK_DEVICE_FNS(VKFN_DECL)

[[noreturn]] void fatal(const std::string& msg) {
    MessageBoxA(nullptr, msg.c_str(), "PrismRT", MB_ICONERROR);
    ExitProcess(1);
}
void vkCheck(VkResult r, std::string_view what) {
    if (r != VK_SUCCESS) fatal(std::format("{}\nfailed with VkResult {}", what, static_cast<int>(r)));
}
#define VK_CHECK(call) vkCheck((call), #call)

// ========================= app / scene state ============================

struct SceneUBO {                    // std140 mirror of common.glsl SceneUBO
    float    invViewProj[16];
    float    camPos[4];
    float    lightDir[4];            // xyz toward light, w = cos(half-angle)
    float    srcPosRadius[4];        // xyz aperture centre, w radius
    uint32_t dimsFrameSpp[4];        // w, h, frameIndex, spp
    uint32_t misc[4];                // glassSF10, maxDepth, bits(intensity), bits(fogSigma)
};
static_assert(sizeof(SceneUBO) == 144, "UBO layout");

constexpr uint32_t kMaxDepth = 10;
constexpr float kBeamIntensity = 420.0f;
constexpr float kSourceHalfAngleDeg = 0.35f;
constexpr std::array<float, 4> kFog{ 0.0f, 0.15f, 0.35f, 0.7f };

struct App {
    // orbit camera
    double yaw = 0.35, pitch = 0.22, dist = 3.0;
    // prism rotation
    double prismYawY = 0.0, prismTiltZ = 0.0;
    bool   sf10 = true;
    float  exposure = 0.7f;   // ACES lifts midtones vs Reinhard; compensate
    int    fogIdx = 2;        // index into kFog
    bool   reset = true;
    bool   quit = false;
    bool   sizeDirty = false;
    int    winW = 1280, winH = 720;
    bool   dragL = false, dragR = false;
    POINT  lastMouse{};
};
App g;

LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
    case WM_SIZE:
        g.winW = LOWORD(lp); g.winH = HIWORD(lp);
        if (g.winW > 0 && g.winH > 0) { g.sizeDirty = true; g.reset = true; }
        return 0;
    case WM_LBUTTONDOWN: g.dragL = true;  SetCapture(h); GetCursorPos(&g.lastMouse); return 0;
    case WM_RBUTTONDOWN: g.dragR = true;  SetCapture(h); GetCursorPos(&g.lastMouse); return 0;
    case WM_LBUTTONUP:   g.dragL = false; if (!g.dragR) ReleaseCapture(); return 0;
    case WM_RBUTTONUP:   g.dragR = false; if (!g.dragL) ReleaseCapture(); return 0;
    case WM_MOUSEMOVE:
        if (g.dragL || g.dragR) {
            POINT p; GetCursorPos(&p);
            const double dx = p.x - g.lastMouse.x, dy = p.y - g.lastMouse.y;
            g.lastMouse = p;
            if (g.dragL) {
                g.yaw += dx * 0.008;
                g.pitch = std::clamp(g.pitch + dy * 0.008, -1.35, 1.35);
            }
            if (g.dragR) {
                g.prismYawY += dx * 0.008;
                g.prismTiltZ += dy * 0.008;
            }
            g.reset = true;
        }
        return 0;
    case WM_MOUSEWHEEL:
        g.dist = std::clamp(g.dist * (GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 0.9 : 1.111), 0.6, 12.0);
        g.reset = true;
        return 0;
    case WM_KEYDOWN:
        switch (wp) {
        case VK_ESCAPE: g.quit = true; break;
        case 'G': g.sf10 = !g.sf10; g.reset = true; break;
        case 'F': g.fogIdx = (g.fogIdx + 1) % static_cast<int>(kFog.size()); g.reset = true; break;
        case 'R': g.reset = true; break;
        case VK_OEM_PLUS:  case VK_ADD:      g.exposure *= 1.25f; break;
        case VK_OEM_MINUS: case VK_SUBTRACT: g.exposure /= 1.25f; break;
        }
        return 0;
    case WM_DPICHANGED: {   // moved to a monitor with another scale: take the suggested rect
        const RECT& s = *reinterpret_cast<const RECT*>(lp);
        SetWindowPos(h, nullptr, s.left, s.top, s.right - s.left, s.bottom - s.top,
                     SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_CLOSE: g.quit = true; return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProc(h, m, wp, lp);
}

// Per-monitor DPI awareness: the swapchain gets the monitor's real pixels
// instead of a bitmap-scaled window. The user32 entry points are resolved at
// runtime so the exe still starts on Windows versions without them.
template <typename Fn>
Fn user32Fn(const char* name) {
    const HMODULE user32 = GetModuleHandleW(L"user32.dll");
    return user32 ? reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(user32, name))) : nullptr;
}

void enableDpiAwareness() {
    if (auto fn = user32Fn<BOOL(WINAPI*)(HANDLE)>("SetProcessDpiAwarenessContext"))
        fn(reinterpret_cast<HANDLE>(static_cast<intptr_t>(-4)));  // ..._PER_MONITOR_AWARE_V2
}

// 96 on an unscaled display, 192 at 200 %.
UINT systemDpi() {
    if (auto fn = user32Fn<UINT(WINAPI*)()>("GetDpiForSystem")) return fn();
    return USER_DEFAULT_SCREEN_DPI;
}

// Window rect for a client area of w x h pixels, with the frame sized for `dpi`.
RECT windowRectFor(int w, int h, UINT dpi) {
    RECT r{ 0, 0, w, h };
    if (auto fn = user32Fn<BOOL(WINAPI*)(LPRECT, DWORD, BOOL, DWORD, UINT)>("AdjustWindowRectExForDpi"))
        fn(&r, WS_OVERLAPPEDWINDOW, FALSE, 0, dpi);
    else
        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    return r;
}

// ====================== Vulkan context + helpers ========================

struct Vk {
    VkInstance instance{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice phys{};
    VkPhysicalDeviceMemoryProperties memProps{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    uint32_t queueFamily = 0;
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool cmdPool{};
    VkCommandBuffer cmd{};
    VkFence fence{};
    VkSemaphore semAcquire{}, semRender{};

    VkSwapchainKHR swapchain{};
    VkExtent2D extent{};
    std::vector<VkImage> swapImages;
};
Vk vk;

struct Buffer {
    VkBuffer buf{};
    VkDeviceMemory mem{};
    void* mapped = nullptr;
    VkDeviceAddress addr = 0;
};

constexpr VkMemoryPropertyFlags kHostMem =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kDeviceMem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < vk.memProps.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (vk.memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    fatal("no suitable memory type");
}

Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags,
                    bool wantAddr) {
    Buffer b;
    const VkBufferCreateInfo bi{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | (wantAddr ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &b.buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk.device, b.buf, &req);
    const VkMemoryAllocateFlagsInfo flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    const VkMemoryAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = wantAddr ? &flags : nullptr,
        .allocationSize = req.size,
        .memoryTypeIndex = findMemType(req.memoryTypeBits, memFlags),
    };
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(vk.device, b.buf, b.mem, 0));

    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(vk.device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    if (wantAddr) {
        const VkBufferDeviceAddressInfo dai{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b.buf };
        b.addr = vkGetBufferDeviceAddress(vk.device, &dai);
    }
    return b;
}

// Host-visible buffer filled with `data` (geometry, instances, SBTs).
template <typename T>
Buffer createFilledBuffer(std::span<const T> data, VkBufferUsageFlags usage, bool wantAddr) {
    Buffer b = createBuffer(data.size_bytes(), usage, kHostMem, wantAddr);
    std::memcpy(b.mapped, data.data(), data.size_bytes());
    return b;
}

void destroyBuffer(Buffer& b) {
    if (b.buf) vkDestroyBuffer(vk.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(vk.device, b.mem, nullptr);
    b = {};
}

constexpr VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); }

struct Image {
    VkImage img{};
    VkDeviceMemory mem{};
    VkImageView view{};
};

Image createStorageImage(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    Image im;
    const VkImageCreateInfo ii{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt,
        .extent = { w, h, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &im.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(vk.device, im.img, &req);
    const VkMemoryAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = findMemType(req.memoryTypeBits, kDeviceMem),
    };
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &im.mem));
    VK_CHECK(vkBindImageMemory(vk.device, im.img, im.mem, 0));
    const VkImageViewCreateInfo vi{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = im.img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    VK_CHECK(vkCreateImageView(vk.device, &vi, nullptr, &im.view));
    return im;
}

void destroyImage(Image& im) {
    if (im.view) vkDestroyImageView(vk.device, im.view, nullptr);
    if (im.img)  vkDestroyImage(vk.device, im.img, nullptr);
    if (im.mem)  vkFreeMemory(vk.device, im.mem, nullptr);
    im = {};
}

void barrierImage(VkCommandBuffer cmd, VkImage img, VkImageLayout oldL, VkImageLayout newL,
                  VkAccessFlags srcA, VkAccessFlags dstA,
                  VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
    const VkImageMemoryBarrier b{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = srcA,
        .dstAccessMask = dstA,
        .oldLayout = oldL,
        .newLayout = newL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void barrierMemory(VkCommandBuffer cmd, VkAccessFlags srcA, VkAccessFlags dstA,
                   VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
    const VkMemoryBarrier b{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = srcA, .dstAccessMask = dstA };
    vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// Runs the recorded commands on the queue and waits (setup-time builds).
template <typename Record>
void submitAndWait(Record&& record) {
    const VkCommandBufferBeginInfo cbi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(vk.cmd, &cbi));
    record(vk.cmd);
    VK_CHECK(vkEndCommandBuffer(vk.cmd));
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &vk.cmd };
    VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(vk.queue));
}

// SPIR-V from the spv/ folder next to the sources; a few locations are
// probed so the exe works from the build directory or the repo root.
std::vector<uint32_t> loadSpirv(std::string_view name) {
    constexpr std::array<std::string_view, 5> prefixes{
        "", "spv/", "PrismRT/spv/", "../../PrismRT/spv/", "..\\..\\PrismRT\\spv\\" };
    for (std::string_view prefix : prefixes) {
        std::ifstream file(std::string(prefix) + std::string(name), std::ios::binary | std::ios::ate);
        if (!file) continue;
        const std::streamsize size = file.tellg();
        if (size <= 0 || size % 4 != 0) fatal(std::format("{} is not a SPIR-V module", name));
        std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(words.data()), size);
        if (words[0] != 0x07230203u) fatal(std::format("{} is not a SPIR-V module", name));
        return words;
    }
    fatal(std::format("cannot find {}\nRun compile_shaders.bat first (needs glslc from the Vulkan SDK).",
                      name));
}

VkShaderModule loadShader(std::string_view name) {
    const std::vector<uint32_t> code = loadSpirv(name);
    const VkShaderModuleCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode = code.data(),
    };
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(vk.device, &ci, nullptr, &m));
    return m;
}

// ============================ init ======================================

constexpr std::array<const char*, 4> kDeviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

bool supportsExtensions(VkPhysicalDevice d) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &n, exts.data());
    return std::ranges::all_of(kDeviceExtensions, [&](const char* need) {
        return std::ranges::any_of(exts, [&](const VkExtensionProperties& e) {
            return std::strcmp(e.extensionName, need) == 0; });
    });
}

// A graphics + compute queue family that can present to the surface, or -1.
int findQueueFamily(VkPhysicalDevice d) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &n, families.data());
    constexpr VkQueueFlags needed = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < n; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(d, i, vk.surface, &present);
        if ((families[i].queueFlags & needed) == needed && present) return static_cast<int>(i);
    }
    return -1;
}

void initVulkan(HWND hWnd, HINSTANCE hInst) {
    HMODULE dll = LoadLibraryA("vulkan-1.dll");
    if (!dll) fatal("vulkan-1.dll not found — install a Vulkan driver.");
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(dll, "vkGetInstanceProcAddr")));
    if (!vkGetInstanceProcAddr) fatal("vkGetInstanceProcAddr not found");
#define VKFN_GLOBAL(n) n = reinterpret_cast<PFN_##n>(vkGetInstanceProcAddr(nullptr, #n));
    VK_GLOBAL_FNS(VKFN_GLOBAL)

    const VkApplicationInfo app{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "PrismRT",
        .apiVersion = VK_API_VERSION_1_2,
    };
    constexpr std::array<const char*, 2> instExts{
        VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
    const VkInstanceCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledExtensionCount = static_cast<uint32_t>(instExts.size()),
        .ppEnabledExtensionNames = instExts.data(),
    };
    VK_CHECK(vkCreateInstance(&ici, nullptr, &vk.instance));
#define VKFN_INST(n) n = reinterpret_cast<PFN_##n>(vkGetInstanceProcAddr(vk.instance, #n));
    VK_INSTANCE_FNS(VKFN_INST)
    vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        vkGetInstanceProcAddr(vk.instance, "vkGetDeviceProcAddr"));

    const VkWin32SurfaceCreateInfoKHR sci{
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, .hinstance = hInst, .hwnd = hWnd };
    VK_CHECK(vkCreateWin32SurfaceKHR(vk.instance, &sci, nullptr, &vk.surface));

    // pick a device with the RT extensions and a graphics/present queue;
    // a discrete GPU wins over an integrated one that also qualifies
    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(vk.instance, &ndev, nullptr));
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(vk.instance, &ndev, devs.data()));
    int bestScore = -1;
    for (VkPhysicalDevice d : devs) {
        if (!supportsExtensions(d)) continue;
        const int family = findQueueFamily(d);
        if (family < 0) continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
        if (score > bestScore) {
            bestScore = score;
            vk.phys = d;
            vk.queueFamily = static_cast<uint32_t>(family);
        }
    }
    if (!vk.phys)
        fatal("No GPU with VK_KHR_ray_tracing_pipeline found.\n"
              "Hardware ray tracing support is required.");

    vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.memProps);
    vk.rtProps = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
    VkPhysicalDeviceProperties2 p2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &vk.rtProps };
    vkGetPhysicalDeviceProperties2(vk.phys, &p2);

    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo qci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk.queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .rayTracingPipeline = VK_TRUE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &rtFeatures,
        .accelerationStructure = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features v12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &asFeatures,
        .scalarBlockLayout = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    const VkDeviceCreateInfo dci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &v12,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size()),
        .ppEnabledExtensionNames = kDeviceExtensions.data(),
    };
    VK_CHECK(vkCreateDevice(vk.phys, &dci, nullptr, &vk.device));
#define VKFN_DEV(n) n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(vk.device, #n));
    VK_DEVICE_FNS(VKFN_DEV)
    vkGetDeviceQueue(vk.device, vk.queueFamily, 0, &vk.queue);

    const VkCommandPoolCreateInfo cpi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = vk.queueFamily,
    };
    VK_CHECK(vkCreateCommandPool(vk.device, &cpi, nullptr, &vk.cmdPool));
    const VkCommandBufferAllocateInfo cai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk.cmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(vk.device, &cai, &vk.cmd));

    const VkFenceCreateInfo fci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
    VK_CHECK(vkCreateFence(vk.device, &fci, nullptr, &vk.fence));
    const VkSemaphoreCreateInfo semci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VK_CHECK(vkCreateSemaphore(vk.device, &semci, nullptr, &vk.semAcquire));
    VK_CHECK(vkCreateSemaphore(vk.device, &semci, nullptr, &vk.semRender));
}

void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.phys, vk.surface, &caps));
    vk.extent = caps.currentExtent;
    if (vk.extent.width == 0xFFFFFFFFu)
        vk.extent = { static_cast<uint32_t>(g.winW), static_cast<uint32_t>(g.winH) };

    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &nf, formats.data());
    auto preferred = std::ranges::find(formats, VK_FORMAT_B8G8R8A8_UNORM, &VkSurfaceFormatKHR::format);
    const VkSurfaceFormatKHR pick = preferred != formats.end() ? *preferred : formats.at(0);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount) imageCount = std::min(imageCount, caps.maxImageCount);

    const VkSwapchainCreateInfoKHR sci{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = vk.surface,
        .minImageCount = imageCount,
        .imageFormat = pick.format,
        .imageColorSpace = pick.colorSpace,
        .imageExtent = vk.extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,   // vsync, always available
        .clipped = VK_TRUE,
        .oldSwapchain = vk.swapchain,
    };
    VkSwapchainKHR newSwap;
    VK_CHECK(vkCreateSwapchainKHR(vk.device, &sci, nullptr, &newSwap));
    if (vk.swapchain) vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr);
    vk.swapchain = newSwap;

    uint32_t ni = 0;
    vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &ni, nullptr);
    vk.swapImages.resize(ni);
    vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &ni, vk.swapImages.data());
}

// ================= geometry, BLAS/TLAS, pipelines, SBT ==================

// Prism mesh: triangle in XY (apex up), extruded along Z. Same cross-section
// as the 2D renderers. Winding is irrelevant — the closest-hit derives the
// geometric normal and orients it against the ray.
constexpr std::array<float, 18> kPrismPos{
     0.000f,  0.620f,  0.550f,
    -0.520f, -0.280f,  0.550f,
     0.520f, -0.280f,  0.550f,
     0.000f,  0.620f, -0.550f,
    -0.520f, -0.280f, -0.550f,
     0.520f, -0.280f, -0.550f,
};
constexpr std::array<uint32_t, 24> kPrismIdx{
    0,1,2,  3,5,4,          // caps
    0,3,4,  0,4,1,          // A-B side
    1,4,5,  1,5,2,          // B-C side
    2,5,3,  2,3,0,          // C-A side
};

// floor: large quad at y=0 in local space; the TLAS instance drops it to the
// prism base. The closest-hit never fetches its geometry (normal is +Y).
// Big enough that its far edge is fully swallowed by fog attenuation.
constexpr std::array<float, 12> kFloorPos{
    -120.0f, 0.0f, -120.0f,
     120.0f, 0.0f, -120.0f,
     120.0f, 0.0f,  120.0f,
    -120.0f, 0.0f,  120.0f,
};
constexpr std::array<uint32_t, 6> kQuadIdx{ 0,1,2,  0,2,3 };
constexpr float kFloorY = static_cast<float>(FLOOR_PLANE_Y);   // just below the prism base (y = -0.28)

// collimated beam, same geometry as the 2D renderers (in the XY plane).
// The SOURCE IS GEOMETRY: an emissive quad at the aperture (TLAS instance
// INST_SOURCE) so light connections hit it before the floor behind it —
// evaluating the aperture in the miss shader never fires when geometry lies
// past the source.
constexpr float kBeamDeg = 24.0f;
constexpr float kBeamRadius = 0.085f;
constexpr float kQuadHalf = 0.11f;   // quad slightly over the soft disk edge
V3 beamDir() { return { std::cos(degToRad(kBeamDeg)), std::sin(degToRad(kBeamDeg)), 0.0f }; }
V3 beamSourcePos() { return V3{ -0.24f, 0.20f, 0.0f } - beamDir() * 1.15f; }

constexpr VkDeviceSize kGridBytes = VkDeviceSize{ GRID_X } * GRID_Y * GRID_Z * 3 * sizeof(uint32_t);
constexpr VkDeviceSize kCausBytes = VkDeviceSize{ CAUS_X } * CAUS_Z * 3 * sizeof(uint32_t);

struct Blas {
    Buffer positions, indices, storage;
    VkAccelerationStructureKHR as{};
    VkDeviceAddress addr = 0;
};

struct RT {
    Blas prism, floor, source;
    Buffer instBuf, tlasBuf, tlasScratch;
    VkAccelerationStructureKHR tlas{};

    Buffer uboBuf;
    Buffer gridBuf, causBuf;                 // photon fog grid + floor caustic
    Buffer cameraSbt, photonSbt;

    VkDescriptorSetLayout rtSetLayout{}, resSetLayout{};
    VkDescriptorPool descPool{};
    VkDescriptorSet rtSet{}, resSet{};
    VkPipelineLayout rtLayout{}, resLayout{};
    VkPipeline cameraPipe{}, photonPipe{}, resPipe{};

    struct SbtRegions { VkStridedDeviceAddressRegionKHR rgen{}, miss{}, hit{}, call{}; };
    SbtRegions cameraRegions, photonRegions;

    Image accum, outImg;

    // persistent TLAS build description (rebuilt each frame)
    VkAccelerationStructureGeometryKHR tlasGeom{};
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
};
RT rt;

Blas buildBlas(std::span<const float> positions, std::span<const uint32_t> indices) {
    constexpr VkBufferUsageFlags geoUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    Blas out;
    out.positions = createFilledBuffer(positions, geoUsage, true);
    out.indices = createFilledBuffer(indices, geoUsage, true);
    const uint32_t primCount = static_cast<uint32_t>(indices.size() / 3);

    const VkAccelerationStructureGeometryKHR geom{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry = { .triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData = { .deviceAddress = out.positions.addr },
            .vertexStride = 3 * sizeof(float),
            .maxVertex = static_cast<uint32_t>(positions.size() / 3 - 1),
            .indexType = VK_INDEX_TYPE_UINT32,
            .indexData = { .deviceAddress = out.indices.addr },
        } },
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    VkAccelerationStructureBuildGeometryInfoKHR build{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries = &geom,
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    vkGetAccelerationStructureBuildSizesKHR(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &build, &primCount, &sizes);

    out.storage = createBuffer(sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, kDeviceMem, true);
    Buffer scratch = createBuffer(sizes.buildScratchSize + 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDeviceMem, true);

    const VkAccelerationStructureCreateInfoKHR aci{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = out.storage.buf,
        .size = sizes.accelerationStructureSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    };
    VK_CHECK(vkCreateAccelerationStructureKHR(vk.device, &aci, nullptr, &out.as));
    build.dstAccelerationStructure = out.as;
    build.scratchData.deviceAddress = alignUp(scratch.addr, 256);

    const VkAccelerationStructureBuildRangeInfoKHR range{ .primitiveCount = primCount };
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    submitAndWait([&](VkCommandBuffer cmd) {
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &pRange);
    });
    destroyBuffer(scratch);   // a BLAS is built once; the scratch is not needed again

    const VkAccelerationStructureDeviceAddressInfoKHR dai{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = out.as,
    };
    out.addr = vkGetAccelerationStructureDeviceAddressKHR(vk.device, &dai);
    return out;
}

void buildBlases() {
    rt.prism = buildBlas(kPrismPos, kPrismIdx);
    rt.floor = buildBlas(kFloorPos, kQuadIdx);

    // emissive source quad: world-space verts around the aperture centre,
    // e1 in the beam plane, e2 = world Z (identity TLAS instance)
    const V3 src = beamSourcePos();
    const V3 toLight = -beamDir();
    const V3 e1 = normalize(cross(toLight, V3{ 0, 0, 1 })) * kQuadHalf;
    const V3 e2 = V3{ 0, 0, 1 } * kQuadHalf;
    const std::array<V3, 4> corners{ src - e1 - e2, src + e1 - e2, src + e1 + e2, src - e1 + e2 };
    std::array<float, 12> quadPos;
    for (size_t i = 0; i < corners.size(); ++i) {
        quadPos[i * 3 + 0] = corners[i].x;
        quadPos[i * 3 + 1] = corners[i].y;
        quadPos[i * 3 + 2] = corners[i].z;
    }
    rt.source = buildBlas(quadPos, kQuadIdx);
}

void createTlas() {
    constexpr uint32_t instanceCount = 3;
    rt.instBuf = createBuffer(instanceCount * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, kHostMem, true);

    rt.tlasGeom = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = { .instances = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .arrayOfPointers = VK_FALSE,
            .data = { .deviceAddress = rt.instBuf.addr },
        } },
    };
    rt.tlasBuild = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries = &rt.tlasGeom,
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    uint32_t prims = instanceCount;
    vkGetAccelerationStructureBuildSizesKHR(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &rt.tlasBuild, &prims, &sizes);

    rt.tlasBuf = createBuffer(sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, kDeviceMem, true);
    rt.tlasScratch = createBuffer(sizes.buildScratchSize + 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kDeviceMem, true);

    const VkAccelerationStructureCreateInfoKHR aci{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = rt.tlasBuf.buf,
        .size = sizes.accelerationStructureSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    };
    VK_CHECK(vkCreateAccelerationStructureKHR(vk.device, &aci, nullptr, &rt.tlas));
    rt.tlasBuild.dstAccelerationStructure = rt.tlas;
    rt.tlasBuild.scratchData.deviceAddress = alignUp(rt.tlasScratch.addr, 256);
}

VkAccelerationStructureInstanceKHR makeInstance(const float (&rows)[3][4], uint32_t customIndex,
                                                VkDeviceAddress blas) {
    VkAccelerationStructureInstanceKHR inst{
        .instanceCustomIndex = customIndex,
        .mask = 0xFF,
        .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
        .accelerationStructureReference = blas,
    };
    std::memcpy(&inst.transform, rows, sizeof(inst.transform));
    return inst;
}

// TLAS instances: the prism with Ry(yaw)*Rz(tilt), the static floor at the
// prism base, the source quad in world space. customIndex is what the
// closest-hit reports (INST_* in layout.h).
void writeInstances() {
    const auto cy = static_cast<float>(std::cos(g.prismYawY));
    const auto sy = static_cast<float>(std::sin(g.prismYawY));
    const auto cz = static_cast<float>(std::cos(g.prismTiltZ));
    const auto sz = static_cast<float>(std::sin(g.prismTiltZ));
    const float prismXf[3][4] = {
        {  cy * cz, -cy * sz,  sy,   0.0f },
        {  sz,       cz,       0.0f, 0.0f },
        { -sy * cz,  sy * sz,  cy,   0.0f },
    };
    const float floorXf[3][4] = {
        { 1, 0, 0, 0.0f },
        { 0, 1, 0, kFloorY },
        { 0, 0, 1, 0.0f },
    };
    const float identityXf[3][4] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };

    const std::array<VkAccelerationStructureInstanceKHR, 3> instances{
        makeInstance(prismXf, INST_PRISM, rt.prism.addr),
        makeInstance(floorXf, INST_FLOOR, rt.floor.addr),
        makeInstance(identityXf, INST_SOURCE, rt.source.addr),
    };
    std::memcpy(rt.instBuf.mapped, instances.data(), sizeof(instances));
}

// One SBT: a base-aligned record per group (raygen, miss, hit).
RT::SbtRegions createSbt(VkPipeline pipeline, Buffer& buffer) {
    constexpr uint32_t groupCount = 3;
    const uint32_t handleSize = vk.rtProps.shaderGroupHandleSize;
    const VkDeviceSize baseAlign = vk.rtProps.shaderGroupBaseAlignment;
    const VkDeviceSize record = alignUp(handleSize, baseAlign);

    std::vector<uint8_t> handles(groupCount * handleSize);
    VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(vk.device, pipeline, 0, groupCount,
                                                  handles.size(), handles.data()));
    // the buffer's own address need not be base-aligned: pad and align inside
    buffer = createBuffer(record * groupCount + baseAlign,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, kHostMem, true);
    const VkDeviceAddress base = alignUp(buffer.addr, baseAlign);
    uint8_t* dst = static_cast<uint8_t*>(buffer.mapped) + (base - buffer.addr);
    std::memset(dst, 0, static_cast<size_t>(record * groupCount));
    for (uint32_t i = 0; i < groupCount; ++i)
        std::memcpy(dst + i * record, handles.data() + i * handleSize, handleSize);

    return {
        .rgen = { base, record, record },
        .miss = { base + record, record, record },
        .hit = { base + record * 2, record, record },
        .call = {},
    };
}

void createPipelines() {
    // ---- RT descriptor set layout (bindings from layout.h) ----
    constexpr VkShaderStageFlags rgen = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    constexpr VkShaderStageFlags chit = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    constexpr VkShaderStageFlags all = rgen | chit | VK_SHADER_STAGE_MISS_BIT_KHR;
    const std::array<VkDescriptorSetLayoutBinding, 7> rtBindings{ {
        { BIND_TLAS,      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, rgen, nullptr },
        { BIND_ACCUM,     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr },
        { BIND_POSITIONS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, chit, nullptr },
        { BIND_INDICES,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, chit, nullptr },
        { BIND_SCENE,     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, all, nullptr },
        { BIND_FOG_GRID,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr },
        { BIND_CAUSTIC,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr },
    } };
    VkDescriptorSetLayoutCreateInfo dli{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(rtBindings.size()),
        .pBindings = rtBindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &dli, nullptr, &rt.rtSetLayout));

    // ---- resolve descriptor set layout ----
    const std::array<VkDescriptorSetLayoutBinding, 2> resBindings{ {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    } };
    dli.bindingCount = static_cast<uint32_t>(resBindings.size());
    dli.pBindings = resBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &dli, nullptr, &rt.resSetLayout));

    // ---- descriptor pool + sets ----
    const std::array<VkDescriptorPoolSize, 4> poolSizes{ {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
    } };
    const VkDescriptorPoolCreateInfo dpi{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(vk.device, &dpi, nullptr, &rt.descPool));

    VkDescriptorSetAllocateInfo dsa{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = rt.descPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &rt.rtSetLayout,
    };
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &dsa, &rt.rtSet));
    dsa.pSetLayouts = &rt.resSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &dsa, &rt.resSet));

    // ---- pipeline layouts ----
    VkPipelineLayoutCreateInfo pli{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &rt.rtSetLayout,
    };
    VK_CHECK(vkCreatePipelineLayout(vk.device, &pli, nullptr, &rt.rtLayout));

    const VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };  // ResolvePC
    pli.pSetLayouts = &rt.resSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &pli, nullptr, &rt.resLayout));

    // ---- the two RT pipelines: own raygen each, shared thin miss / chit ----
    const VkShaderModule mCamera = loadShader("prism.rgen.spv");
    const VkShaderModule mPhoton = loadShader("photon.rgen.spv");
    const VkShaderModule mMiss = loadShader("prism.rmiss.spv");
    const VkShaderModule mChit = loadShader("prism.rchit.spv");

    std::array<VkPipelineShaderStageCreateInfo, 3> stages{ {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR, .module = mCamera, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_MISS_BIT_KHR, .module = mMiss, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, .module = mChit, .pName = "main" },
    } };
    constexpr auto general = [](uint32_t shader) {
        return VkRayTracingShaderGroupCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = shader,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
    };
    const std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{ {
        general(0),
        general(1),
        { .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
          .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
          .generalShader = VK_SHADER_UNUSED_KHR,
          .closestHitShader = 2,
          .anyHitShader = VK_SHADER_UNUSED_KHR,
          .intersectionShader = VK_SHADER_UNUSED_KHR },
    } };
    VkRayTracingPipelineCreateInfoKHR rpi{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .groupCount = static_cast<uint32_t>(groups.size()),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = 1,   // bounce loops live in rgen
        .layout = rt.rtLayout,
    };
    VK_CHECK(vkCreateRayTracingPipelinesKHR(vk.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            1, &rpi, nullptr, &rt.cameraPipe));
    stages[0].module = mPhoton;
    VK_CHECK(vkCreateRayTracingPipelinesKHR(vk.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            1, &rpi, nullptr, &rt.photonPipe));
    for (VkShaderModule m : { mCamera, mPhoton, mMiss, mChit })
        vkDestroyShaderModule(vk.device, m, nullptr);

    rt.cameraRegions = createSbt(rt.cameraPipe, rt.cameraSbt);
    rt.photonRegions = createSbt(rt.photonPipe, rt.photonSbt);

    // ---- resolve compute pipeline ----
    const VkShaderModule mRes = loadShader("resolve.comp.spv");
    const VkComputePipelineCreateInfo cpi{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mRes, .pName = "main" },
        .layout = rt.resLayout,
    };
    VK_CHECK(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &rt.resPipe));
    vkDestroyShaderModule(vk.device, mRes, nullptr);

    // ---- UBO + photon storage (fog grid, floor caustic; device local) ----
    rt.uboBuf = createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHostMem, false);
    constexpr VkBufferUsageFlags photonUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    rt.gridBuf = createBuffer(kGridBytes, photonUsage, kDeviceMem, false);
    rt.causBuf = createBuffer(kCausBytes, photonUsage, kDeviceMem, false);
}

// (re)create render targets and point the descriptor sets at them
void createTargets() {
    destroyImage(rt.accum);
    destroyImage(rt.outImg);
    rt.accum = createStorageImage(VK_FORMAT_R32G32B32A32_SFLOAT,
        vk.extent.width, vk.extent.height, VK_IMAGE_USAGE_STORAGE_BIT);
    rt.outImg = createStorageImage(VK_FORMAT_R8G8B8A8_UNORM,
        vk.extent.width, vk.extent.height,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // both live in GENERAL layout from now on
    submitAndWait([&](VkCommandBuffer cmd) {
        for (VkImage img : { rt.accum.img, rt.outImg.img })
            barrierImage(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, 0,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    });

    const VkWriteDescriptorSetAccelerationStructureKHR asWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &rt.tlas,
    };
    const VkDescriptorImageInfo accumInfo{ VK_NULL_HANDLE, rt.accum.view, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, rt.outImg.view, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorBufferInfo posInfo{ rt.prism.positions.buf, 0, VK_WHOLE_SIZE };
    const VkDescriptorBufferInfo idxInfo{ rt.prism.indices.buf, 0, VK_WHOLE_SIZE };
    const VkDescriptorBufferInfo uboInfo{ rt.uboBuf.buf, 0, sizeof(SceneUBO) };
    const VkDescriptorBufferInfo gridInfo{ rt.gridBuf.buf, 0, VK_WHOLE_SIZE };
    const VkDescriptorBufferInfo causInfo{ rt.causBuf.buf, 0, VK_WHOLE_SIZE };

    const auto write = [](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                          const VkDescriptorImageInfo* image, const VkDescriptorBufferInfo* buffer,
                          const void* next = nullptr) {
        return VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = next,
            .dstSet = set,
            .dstBinding = binding,
            .descriptorCount = 1,
            .descriptorType = type,
            .pImageInfo = image,
            .pBufferInfo = buffer,
        };
    };
    const std::array<VkWriteDescriptorSet, 9> writes{ {
        write(rt.rtSet, BIND_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, &asWrite),
        write(rt.rtSet, BIND_ACCUM, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumInfo, nullptr),
        write(rt.rtSet, BIND_POSITIONS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &posInfo),
        write(rt.rtSet, BIND_INDICES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &idxInfo),
        write(rt.rtSet, BIND_SCENE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo),
        write(rt.rtSet, BIND_FOG_GRID, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gridInfo),
        write(rt.rtSet, BIND_CAUSTIC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &causInfo),
        write(rt.resSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumInfo, nullptr),
        write(rt.resSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr),
    } };
    vkUpdateDescriptorSets(vk.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// =========================== frame loop =================================

void updateUbo(uint32_t frameIndex) {
    // orbit camera around the prism
    const V3 target{ 0.0f, 0.10f, 0.0f };
    const float cy = static_cast<float>(std::cos(g.yaw)), sy = static_cast<float>(std::sin(g.yaw));
    const float cp = static_cast<float>(std::cos(g.pitch)), sp = static_cast<float>(std::sin(g.pitch));
    const V3 eye = target + V3{ sy * cp, sp, cy * cp } * static_cast<float>(g.dist);
    const float aspect = static_cast<float>(vk.extent.width) / static_cast<float>(vk.extent.height);
    const M4 invViewProj = mul(inverseLookAt(eye, target, { 0, 1, 0 }),
                               inversePerspective(40.0f, aspect, 0.05f, 100.0f));

    const V3 toLight = -beamDir();
    const V3 src = beamSourcePos();
    SceneUBO u{
        .camPos = { eye.x, eye.y, eye.z, 0.0f },
        .lightDir = { toLight.x, toLight.y, toLight.z, std::cos(degToRad(kSourceHalfAngleDeg)) },
        .srcPosRadius = { src.x, src.y, src.z, kBeamRadius },
        .dimsFrameSpp = { vk.extent.width, vk.extent.height, frameIndex, 1 },
        .misc = { g.sf10 ? 1u : 0u, kMaxDepth, std::bit_cast<uint32_t>(kBeamIntensity),
                  std::bit_cast<uint32_t>(kFog[static_cast<size_t>(g.fogIdx)]) },
    };
    std::ranges::copy(invViewProj.m, u.invViewProj);
    std::memcpy(rt.uboBuf.mapped, &u, sizeof(u));
}

void recordFrame(uint32_t swapIdx, uint32_t frameIndex) {
    VK_CHECK(vkResetCommandBuffer(vk.cmd, 0));
    const VkCommandBufferBeginInfo cbi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(vk.cmd, &cbi));
    constexpr VkPipelineStageFlags rtStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

    // 1) rebuild the TLAS with the current prism transform (3 instances)
    const VkAccelerationStructureBuildRangeInfoKHR range{ .primitiveCount = 3 };
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    vkCmdBuildAccelerationStructuresKHR(vk.cmd, 1, &rt.tlasBuild, &pRange);
    barrierMemory(vk.cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, rtStage);

    // 2) photon pass: forward beam transport into the fog grid + caustic map.
    // On accumulation reset, zero both first.
    if (frameIndex == 0) {
        vkCmdFillBuffer(vk.cmd, rt.gridBuf.buf, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(vk.cmd, rt.causBuf.buf, 0, VK_WHOLE_SIZE, 0);
        barrierMemory(vk.cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, rtStage);
    }
    vkCmdBindPipeline(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt.photonPipe);
    vkCmdBindDescriptorSets(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            rt.rtLayout, 0, 1, &rt.rtSet, 0, nullptr);
    const RT::SbtRegions& ph = rt.photonRegions;
    vkCmdTraceRaysKHR(vk.cmd, &ph.rgen, &ph.miss, &ph.hit, &ph.call, PHOTON_APERTURE, PHOTON_LAMBDA, 1);
    barrierMemory(vk.cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, rtStage, rtStage);

    // 3) camera pass: accumulate spectral XYZ
    vkCmdBindPipeline(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt.cameraPipe);
    vkCmdBindDescriptorSets(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            rt.rtLayout, 0, 1, &rt.rtSet, 0, nullptr);
    const RT::SbtRegions& cam = rt.cameraRegions;
    vkCmdTraceRaysKHR(vk.cmd, &cam.rgen, &cam.miss, &cam.hit, &cam.call,
                      vk.extent.width, vk.extent.height, 1);
    barrierMemory(vk.cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  rtStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // 4) resolve: average, tonemap, gamma -> outImg
    const struct { uint32_t frameCount; float exposure; } pc{ frameIndex + 1, g.exposure };
    vkCmdBindPipeline(vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rt.resPipe);
    vkCmdBindDescriptorSets(vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            rt.resLayout, 0, 1, &rt.resSet, 0, nullptr);
    vkCmdPushConstants(vk.cmd, rt.resLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(vk.cmd, (vk.extent.width + 7) / 8, (vk.extent.height + 7) / 8, 1);

    // 5) blit outImg -> swapchain image
    barrierImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrierImage(vk.cmd, vk.swapImages[swapIdx],
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    const VkOffset3D size{ static_cast<int32_t>(vk.extent.width), static_cast<int32_t>(vk.extent.height), 1 };
    const VkImageBlit blit{
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .srcOffsets = { {}, size },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstOffsets = { {}, size },
    };
    vkCmdBlitImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vk.swapImages[swapIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

    barrierImage(vk.cmd, vk.swapImages[swapIdx],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    barrierImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    VK_CHECK(vkEndCommandBuffer(vk.cmd));
}

} // namespace

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow) {
    enableDpiAwareness();

    const WNDCLASSW wc{
        .lpfnWndProc = WndProc,
        .hInstance = hInst,
        .hCursor = LoadCursor(nullptr, IDC_ARROW),
        .lpszClassName = L"PrismRTWnd",
    };
    RegisterClassW(&wc);

    // the default 1280x720 is in 96-dpi units; scale it so the window opens
    // at the same apparent size on a HiDPI display (and renders every pixel)
    const UINT dpi = systemDpi();
    g.winW = MulDiv(g.winW, static_cast<int>(dpi), USER_DEFAULT_SCREEN_DPI);
    g.winH = MulDiv(g.winH, static_cast<int>(dpi), USER_DEFAULT_SCREEN_DPI);
    const RECT r = windowRectFor(g.winW, g.winH, dpi);
    HWND hWnd = CreateWindowW(L"PrismRTWnd", L"PrismRT", WS_OVERLAPPEDWINDOW,
                              CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                              nullptr, nullptr, hInst, nullptr);
    if (!hWnd) return 1;

    initVulkan(hWnd, hInst);
    createSwapchain();
    buildBlases();
    createTlas();
    createPipelines();
    createTargets();
    ShowWindow(hWnd, nCmdShow);

    uint32_t frameIndex = 0;
    ULONGLONG lastTitle = 0;
    LARGE_INTEGER clockFreq, lastTick;
    QueryPerformanceFrequency(&clockFreq);
    QueryPerformanceCounter(&lastTick);
    double frameMs = 0.0;

    while (!g.quit) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) g.quit = true;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (g.quit) break;
        if (g.winW <= 0 || g.winH <= 0) { Sleep(30); continue; }    // minimised

        VK_CHECK(vkWaitForFences(vk.device, 1, &vk.fence, VK_TRUE, UINT64_MAX));

        if (g.sizeDirty) {
            VK_CHECK(vkDeviceWaitIdle(vk.device));
            createSwapchain();
            createTargets();
            g.sizeDirty = false;
            g.reset = true;
        }
        if (g.reset) { frameIndex = 0; g.reset = false; }

        writeInstances();
        updateUbo(frameIndex);

        uint32_t swapIdx = 0;
        const VkResult acquired = vkAcquireNextImageKHR(vk.device, vk.swapchain, UINT64_MAX,
                                                        vk.semAcquire, VK_NULL_HANDLE, &swapIdx);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {   // nothing acquired, semaphore untouched
            g.sizeDirty = true;
            continue;
        }
        // VK_SUBOPTIMAL_KHR did acquire an image (and will signal the semaphore):
        // render this frame on it and rebuild the swapchain afterwards.
        if (acquired == VK_SUBOPTIMAL_KHR) g.sizeDirty = true;
        else VK_CHECK(acquired);
        VK_CHECK(vkResetFences(vk.device, 1, &vk.fence));

        recordFrame(swapIdx, frameIndex);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkSubmitInfo si{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &vk.semAcquire,
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &vk.cmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &vk.semRender,
        };
        VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, vk.fence));

        const VkPresentInfoKHR pi{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &vk.semRender,
            .swapchainCount = 1,
            .pSwapchains = &vk.swapchain,
            .pImageIndices = &swapIdx,
        };
        const VkResult presented = vkQueuePresentKHR(vk.queue, &pi);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) g.sizeDirty = true;
        else VK_CHECK(presented);

        ++frameIndex;

        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        const double ms = 1000.0 * static_cast<double>(now.QuadPart - lastTick.QuadPart)
                         / static_cast<double>(clockFreq.QuadPart);
        lastTick = now;
        frameMs = frameMs == 0.0 ? ms : frameMs * 0.9 + ms * 0.1;

        const ULONGLONG tick = GetTickCount64();
        if (tick - lastTitle > 250) {
            const std::wstring title = std::format(
                L"PrismRT - {}   fog {:.2f}   {} frames   {:.1f} ms   "
                L"[LMB] orbit  [RMB] prism  [wheel] zoom  [G]lass  [F]og  [+/-] exposure  [Esc]",
                g.sf10 ? L"SF10" : L"BK7", kFog[static_cast<size_t>(g.fogIdx)], frameIndex, frameMs);
            SetWindowTextW(hWnd, title.c_str());
            lastTitle = tick;
        }
    }

    vkDeviceWaitIdle(vk.device);
    // OS reclaims everything on exit; explicit teardown omitted deliberately
    DestroyWindow(hWnd);
    return 0;
}
