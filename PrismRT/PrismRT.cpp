// PrismRT.cpp — zero-dependency Win32 host for the spectral prism ray tracer.
//
// Drives the PrismRT shaders on VK_KHR_ray_tracing_pipeline: hardware-traced
// spectral dispersion with single-scattering fog, so the white beam and the
// dispersed rainbow fan are visible in air (a beam in vacuum is invisible —
// the fog is what makes the classic prism picture exist in 3D).
//
// Zero dependencies: vulkan-1.dll is loaded at runtime (LoadLibrary +
// vkGetInstanceProcAddr), so nothing links against the Vulkan SDK. The SDK is
// only needed at build time for glslc (compile_shaders.bat), same as before.
//
// Single frame in flight, TLAS rebuilt per frame (one instance — trivial),
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ============================ tiny math ================================
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
struct V3 { float x, y, z; };
static V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 operator*(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 norm(V3 a) { float l = std::sqrt(dot(a, a)); return a * (1.0f / l); }

// column-major 4x4, matching GLSL std140 mat4
struct M4 { float m[16]; };
static M4 mul(const M4& A, const M4& B) {
    M4 R{};
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += A.m[k * 4 + r] * B.m[c * 4 + k];
            R.m[c * 4 + r] = s;
        }
    return R;
}
static M4 lookAt(V3 eye, V3 at, V3 up) {
    V3 f = norm(at - eye), s = norm(cross(f, up)), u = cross(s, f);
    M4 R{};
    R.m[0] = s.x; R.m[4] = s.y; R.m[8]  = s.z; R.m[12] = -dot(s, eye);
    R.m[1] = u.x; R.m[5] = u.y; R.m[9]  = u.z; R.m[13] = -dot(u, eye);
    R.m[2] = -f.x; R.m[6] = -f.y; R.m[10] = -f.z; R.m[14] = dot(f, eye);
    R.m[3] = 0;   R.m[7] = 0;   R.m[11] = 0;   R.m[15] = 1;
    return R;
}
// Vulkan perspective: depth 0..1, y flipped
static M4 perspective(float fovyDeg, float aspect, float zn, float zf) {
    float f = 1.0f / std::tan(fovyDeg * (float)M_PI / 360.0f);
    M4 R{};
    R.m[0] = f / aspect;
    R.m[5] = -f;
    R.m[10] = zf / (zn - zf); R.m[11] = -1.0f;
    R.m[14] = zn * zf / (zn - zf);
    return R;
}
// general 4x4 inverse (adjugate) — the classic gluInvertMatrix routine
static M4 inverse(const M4& s) {
    const float* m = s.m;
    M4 o{}; float* inv = o.m;
    inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
    float det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) inv[i] *= det;
    return o;
}

// ===================== Vulkan dynamic loader ============================
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
static PFN_vkGetDeviceProcAddr   vkGetDeviceProcAddr;

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

#define VKFN_DECL(n) static PFN_##n n;
VK_GLOBAL_FNS(VKFN_DECL)
VK_INSTANCE_FNS(VKFN_DECL)
VK_DEVICE_FNS(VKFN_DECL)

static void fatal(const char* msg) {
    MessageBoxA(nullptr, msg, "PrismRT", MB_ICONERROR);
    ExitProcess(1);
}
#define VK_CHECK(call) do { VkResult _r = (call); if (_r != VK_SUCCESS) { \
    char _b[256]; snprintf(_b, 256, "%s\nfailed with VkResult %d", #call, (int)_r); \
    fatal(_b); } } while (0)

// ========================= app / scene state ============================
struct SceneUBO {                    // std140 mirror of common.glsl SceneUBO
    float    invViewProj[16];
    float    camPos[4];
    float    lightDir[4];            // xyz toward light, w = cos(half-angle)
    float    srcPosRadius[4];        // xyz aperture centre, w radius
    uint32_t dimsFrameSpp[4];
    uint32_t misc[4];                // glassSF10, maxDepth, bits(intensity), bits(fogSigma)
};
static_assert(sizeof(SceneUBO) == 144, "UBO layout");

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct AppState {
    // orbit camera
    double yaw = 0.35, pitch = 0.22, dist = 3.0;
    // prism rotation
    double prismYawY = 0.0, prismTiltZ = 0.0;
    bool   sf10 = true;
    float  exposure = 0.7f;   // ACES lifts midtones vs Reinhard; compensate
    int    fogIdx = 2;               // index into kFog
    bool   reset = true;
    bool   quit = false;
    bool   sizeDirty = false;
    int    winW = 1280, winH = 720;
    bool   dragL = false, dragR = false;
    POINT  lastMouse{};
};
static const float kFog[4] = { 0.0f, 0.15f, 0.35f, 0.7f };
static AppState g;

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
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
            double dx = p.x - g.lastMouse.x, dy = p.y - g.lastMouse.y;
            g.lastMouse = p;
            if (g.dragL) {
                g.yaw   += dx * 0.008;
                g.pitch = std::clamp(g.pitch + dy * 0.008, -1.35, 1.35);
            }
            if (g.dragR) {
                g.prismYawY  += dx * 0.008;
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
        case 'F': g.fogIdx = (g.fogIdx + 1) % 4; g.reset = true; break;
        case 'R': g.reset = true; break;
        case VK_OEM_PLUS:  case VK_ADD:      g.exposure *= 1.25f; break;
        case VK_OEM_MINUS: case VK_SUBTRACT: g.exposure /= 1.25f; break;
        }
        return 0;
    case WM_CLOSE: g.quit = true; return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProc(h, m, wp, lp);
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
    VkFormat swapFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{};
    std::vector<VkImage> swapImages;
};
static Vk vk;

struct Buffer {
    VkBuffer buf{};
    VkDeviceMemory mem{};
    void* mapped = nullptr;
    VkDeviceAddress addr = 0;
};

static uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < vk.memProps.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) &&
            (vk.memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    fatal("no suitable memory type");
    return 0;
}

static Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags memFlags, bool wantAddr) {
    Buffer b;
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage | (wantAddr ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(vk.device, &bi, nullptr, &b.buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk.device, b.buf, &req);
    VkMemoryAllocateFlagsInfo fi{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = wantAddr ? &fi : nullptr;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, memFlags);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(vk.device, b.buf, b.mem, 0));

    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(vk.device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    if (wantAddr) {
        VkBufferDeviceAddressInfo dai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        dai.buffer = b.buf;
        b.addr = vkGetBufferDeviceAddress(vk.device, &dai);
    }
    return b;
}

struct ImageRT {
    VkImage img{};
    VkDeviceMemory mem{};
    VkImageView view{};
};
static ImageRT createStorageImage(VkFormat fmt, uint32_t w, uint32_t h,
                                  VkImageUsageFlags usage) {
    ImageRT im;
    VkImageCreateInfo ii{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = fmt;
    ii.extent = { w, h, 1 };
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(vk.device, &ii, nullptr, &im.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(vk.device, im.img, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &im.mem));
    VK_CHECK(vkBindImageMemory(vk.device, im.img, im.mem, 0));
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(vk.device, &vi, nullptr, &im.view));
    return im;
}
static void destroyImage(ImageRT& im) {
    if (im.view) vkDestroyImageView(vk.device, im.view, nullptr);
    if (im.img)  vkDestroyImage(vk.device, im.img, nullptr);
    if (im.mem)  vkFreeMemory(vk.device, im.mem, nullptr);
    im = {};
}

static void barrierImage(VkCommandBuffer cmd, VkImage img,
                         VkImageLayout oldL, VkImageLayout newL,
                         VkAccessFlags srcA, VkAccessFlags dstA,
                         VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.oldLayout = oldL; b.newLayout = newL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    b.srcAccessMask = srcA; b.dstAccessMask = dstA;
    vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static std::vector<char> loadFile(const char* rel) {
    // probe a few locations so the exe works from the build dir or repo root
    const char* prefixes[] = { "", "spv/", "PrismRT/spv/", "../../PrismRT/spv/", "..\\..\\PrismRT\\spv\\" };
    for (const char* p : prefixes) {
        std::string path = std::string(p) + rel;
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") == 0 && f) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            std::vector<char> data((size_t)n);
            fread(data.data(), 1, (size_t)n, f);
            fclose(f);
            return data;
        }
    }
    std::string msg = std::string("cannot find ") + rel +
        "\nRun compile_shaders.bat first (needs glslc from the Vulkan SDK).";
    fatal(msg.c_str());
    return {};
}
static VkShaderModule loadShader(const char* rel) {
    std::vector<char> code = loadFile(rel);
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = code.size();
    ci.pCode = (const uint32_t*)code.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(vk.device, &ci, nullptr, &m));
    return m;
}

// ============================ init ======================================
static void initVulkan(HWND hWnd, HINSTANCE hInst) {
    HMODULE dll = LoadLibraryA("vulkan-1.dll");
    if (!dll) fatal("vulkan-1.dll not found — install a Vulkan driver.");
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(dll, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) fatal("vkGetInstanceProcAddr not found");
#define VKFN_GLOBAL(n) n = (PFN_##n)vkGetInstanceProcAddr(nullptr, #n);
    VK_GLOBAL_FNS(VKFN_GLOBAL)

    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "PrismRT";
    app.apiVersion = VK_API_VERSION_1_2;
    const char* instExts[] = { VK_KHR_SURFACE_EXTENSION_NAME,
                               VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = instExts;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &vk.instance));
#define VKFN_INST(n) n = (PFN_##n)vkGetInstanceProcAddr(vk.instance, #n);
    VK_INSTANCE_FNS(VKFN_INST)
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(vk.instance, "vkGetDeviceProcAddr");

    VkWin32SurfaceCreateInfoKHR sci{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    sci.hinstance = hInst;
    sci.hwnd = hWnd;
    VK_CHECK(vkCreateWin32SurfaceKHR(vk.instance, &sci, nullptr, &vk.surface));

    // pick a physical device with the RT extensions + a graphics/present queue
    const char* devExts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(vk.instance, &ndev, nullptr));
    if (!ndev) fatal("no Vulkan devices");
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(vk.instance, &ndev, devs.data()));
    for (VkPhysicalDevice d : devs) {
        uint32_t nx = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &nx, nullptr);
        std::vector<VkExtensionProperties> exts(nx);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &nx, exts.data());
        int found = 0;
        for (const char* need : devExts)
            for (auto& e : exts)
                if (!strcmp(e.extensionName, need)) { ++found; break; }
        if (found != 4) continue;
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, vk.surface, &present);
            if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
                vk.phys = d; vk.queueFamily = i;
                break;
            }
        }
        if (vk.phys) break;
    }
    if (!vk.phys)
        fatal("No GPU with VK_KHR_ray_tracing_pipeline found.\n"
              "Hardware ray tracing support is required.");

    vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.memProps);
    vk.rtProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
    VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    p2.pNext = &vk.rtProps;
    vkGetPhysicalDeviceProperties2(vk.phys, &p2);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = vk.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
    rtf.rayTracingPipeline = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    asf.accelerationStructure = VK_TRUE;
    asf.pNext = &rtf;
    VkPhysicalDeviceVulkan12Features v12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    v12.bufferDeviceAddress = VK_TRUE;
    v12.scalarBlockLayout = VK_TRUE;
    v12.pNext = &asf;

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &v12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 4;
    dci.ppEnabledExtensionNames = devExts;
    VK_CHECK(vkCreateDevice(vk.phys, &dci, nullptr, &vk.device));
#define VKFN_DEV(n) n = (PFN_##n)vkGetDeviceProcAddr(vk.device, #n);
    VK_DEVICE_FNS(VKFN_DEV)
    vkGetDeviceQueue(vk.device, vk.queueFamily, 0, &vk.queue);

    VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = vk.queueFamily;
    VK_CHECK(vkCreateCommandPool(vk.device, &cpi, nullptr, &vk.cmdPool));
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = vk.cmdPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(vk.device, &cai, &vk.cmd));

    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(vk.device, &fci, nullptr, &vk.fence));
    VkSemaphoreCreateInfo semci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VK_CHECK(vkCreateSemaphore(vk.device, &semci, nullptr, &vk.semAcquire));
    VK_CHECK(vkCreateSemaphore(vk.device, &semci, nullptr, &vk.semRender));
}

static void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.phys, vk.surface, &caps));
    vk.extent = caps.currentExtent;
    if (vk.extent.width == 0xFFFFFFFFu) {
        vk.extent = { (uint32_t)g.winW, (uint32_t)g.winH };
    }

    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &nf, fmts.data());
    VkSurfaceFormatKHR pick = fmts[0];
    for (auto& f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { pick = f; break; }
    vk.swapFormat = pick.format;

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sci.surface = vk.surface;
    sci.minImageCount = imgCount;
    sci.imageFormat = pick.format;
    sci.imageColorSpace = pick.colorSpace;
    sci.imageExtent = vk.extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;   // vsync, always available
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = vk.swapchain;
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
static const float kPrismPos[] = {
     0.000f,  0.620f,  0.550f,
    -0.520f, -0.280f,  0.550f,
     0.520f, -0.280f,  0.550f,
     0.000f,  0.620f, -0.550f,
    -0.520f, -0.280f, -0.550f,
     0.520f, -0.280f, -0.550f,
};
static const uint32_t kPrismIdx[] = {
    0,1,2,  3,5,4,          // caps
    0,3,4,  0,4,1,          // A-B side
    1,4,5,  1,5,2,          // B-C side
    2,5,3,  2,3,0,          // C-A side
};
static const uint32_t kPrimCount = 8;

struct RT {
    Buffer posBuf, idxBuf, uboBuf, instBuf, sbtBuf;
    Buffer floorPosBuf, floorIdxBuf, quadPosBuf, quadIdxBuf;
    Buffer prismBlasBuf, prismScratch, floorBlasBuf, floorScratch;
    Buffer quadBlasBuf, quadScratch;
    Buffer tlasBuf, tlasScratch;
    VkAccelerationStructureKHR prismBlas{}, floorBlas{}, quadBlas{}, tlas{};
    VkDeviceAddress prismBlasAddr = 0, floorBlasAddr = 0, quadBlasAddr = 0;

    Buffer gridBuf, causBuf;                 // photon fog grid + floor caustic
    Buffer photonSbtBuf;

    VkDescriptorSetLayout rtSetLayout{}, resSetLayout{};
    VkDescriptorPool descPool{};
    VkDescriptorSet rtSet{}, resSet{};
    VkPipelineLayout rtLayout{}, resLayout{};
    VkPipeline rtPipe{}, photonPipe{}, resPipe{};

    VkStridedDeviceAddressRegionKHR sbtRgen{}, sbtMiss{}, sbtHit{}, sbtCall{};
    VkStridedDeviceAddressRegionKHR phRgen{}, phMiss{}, phHit{}, phCall{};

    ImageRT accum, outImg;

    // persistent TLAS build description (rebuilt each frame, one instance)
    VkAccelerationStructureGeometryKHR tlasGeom{};
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
};
static RT rt;

static VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); }

// photon storage dimensions — must match common.glsl
static const uint32_t kGridX = 148, kGridY = 52, kGridZ = 40;
static const uint32_t kCausX = 384, kCausZ = 384;
static const VkDeviceSize kGridBytes = (VkDeviceSize)kGridX * kGridY * kGridZ * 3 * 4;
static const VkDeviceSize kCausBytes = (VkDeviceSize)kCausX * kCausZ * 3 * 4;

// floor: large quad at y=0 in local space; the TLAS instance drops it to the
// prism base. The closest-hit never fetches its geometry (normal is +Y).
// Big enough that its far edge is fully swallowed by fog attenuation.
static const float kFloorPos[] = {
    -120.0f, 0.0f, -120.0f,
     120.0f, 0.0f, -120.0f,
     120.0f, 0.0f,  120.0f,
    -120.0f, 0.0f,  120.0f,
};
static const uint32_t kFloorIdx[] = { 0,1,2,  0,2,3 };

// collimated beam, same geometry as the 2D renderers (in the XY plane).
// The SOURCE IS GEOMETRY: an emissive quad at the aperture (TLAS instance 2)
// so light connections hit it before the floor behind it — evaluating the
// aperture in the miss shader never fires when geometry lies past the source.
static const float kBeamDeg = 24.0f;
static V3 beamD() {
    return { std::cos(kBeamDeg * (float)M_PI / 180.0f),
             std::sin(kBeamDeg * (float)M_PI / 180.0f), 0.0f };
}
static V3 beamSrcPos() { V3 D = beamD(); return V3{ -0.24f, 0.20f, 0.0f } - D * 1.15f; }
static const float kBeamRadius = 0.085f;
static const float kQuadHalf = 0.11f;   // quad slightly over the soft disk edge

struct BlasOut {
    VkAccelerationStructureKHR as;
    VkDeviceAddress addr;
};
static BlasOut buildOneBLAS(const Buffer& posBuf, uint32_t maxVertex,
                            const Buffer& idxBuf, uint32_t primCount,
                            Buffer& asBuf, Buffer& scratch) {
    VkAccelerationStructureGeometryKHR geom{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& tri = geom.geometry.triangles;
    tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.vertexData.deviceAddress = posBuf.addr;
    tri.vertexStride = 12;
    tri.maxVertex = maxVertex;
    tri.indexType = VK_INDEX_TYPE_UINT32;
    tri.indexData.deviceAddress = idxBuf.addr;

    VkAccelerationStructureBuildGeometryInfoKHR bi{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sz{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    uint32_t prims = primCount;
    vkGetAccelerationStructureBuildSizesKHR(vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sz);

    asBuf = createBuffer(sz.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
    scratch = createBuffer(sz.buildScratchSize + 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    VkAccelerationStructureCreateInfoKHR aci{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    aci.buffer = asBuf.buf;
    aci.size = sz.accelerationStructureSize;
    aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    BlasOut out{};
    VK_CHECK(vkCreateAccelerationStructureKHR(vk.device, &aci, nullptr, &out.as));

    bi.dstAccelerationStructure = out.as;
    bi.scratchData.deviceAddress = alignUp(scratch.addr, 256);

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(vk.cmd, &cbi));
    vkCmdBuildAccelerationStructuresKHR(vk.cmd, 1, &bi, &pRange);
    VK_CHECK(vkEndCommandBuffer(vk.cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk.cmd;
    VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(vk.queue));

    VkAccelerationStructureDeviceAddressInfoKHR dai{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    dai.accelerationStructure = out.as;
    out.addr = vkGetAccelerationStructureDeviceAddressKHR(vk.device, &dai);
    return out;
}

static void buildBLAS() {
    const VkBufferUsageFlags geoUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkMemoryPropertyFlags hostMem =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    rt.posBuf = createBuffer(sizeof(kPrismPos), geoUsage, hostMem, true);
    rt.idxBuf = createBuffer(sizeof(kPrismIdx), geoUsage, hostMem, true);
    std::memcpy(rt.posBuf.mapped, kPrismPos, sizeof(kPrismPos));
    std::memcpy(rt.idxBuf.mapped, kPrismIdx, sizeof(kPrismIdx));
    rt.floorPosBuf = createBuffer(sizeof(kFloorPos), geoUsage, hostMem, true);
    rt.floorIdxBuf = createBuffer(sizeof(kFloorIdx), geoUsage, hostMem, true);
    std::memcpy(rt.floorPosBuf.mapped, kFloorPos, sizeof(kFloorPos));
    std::memcpy(rt.floorIdxBuf.mapped, kFloorIdx, sizeof(kFloorIdx));

    // emissive source quad: world-space verts around the aperture centre,
    // e1 in the beam plane, e2 = world Z (identity TLAS instance)
    V3 srcPos = beamSrcPos();
    V3 L = beamD() * -1.0f;                 // toward the light
    V3 e1 = norm(cross(L, V3{ 0, 0, 1 })) * kQuadHalf;
    V3 e2 = V3{ 0, 0, 1 } * kQuadHalf;
    V3 q0 = srcPos - e1 - e2, q1 = srcPos + e1 - e2;
    V3 q2 = srcPos + e1 + e2, q3 = srcPos - e1 + e2;
    const float quadPos[12] = { q0.x,q0.y,q0.z, q1.x,q1.y,q1.z,
                                q2.x,q2.y,q2.z, q3.x,q3.y,q3.z };
    static const uint32_t quadIdx[6] = { 0,1,2, 0,2,3 };
    rt.quadPosBuf = createBuffer(sizeof(quadPos), geoUsage, hostMem, true);
    rt.quadIdxBuf = createBuffer(sizeof(quadIdx), geoUsage, hostMem, true);
    std::memcpy(rt.quadPosBuf.mapped, quadPos, sizeof(quadPos));
    std::memcpy(rt.quadIdxBuf.mapped, quadIdx, sizeof(quadIdx));

    BlasOut p = buildOneBLAS(rt.posBuf, 5, rt.idxBuf, kPrimCount,
                             rt.prismBlasBuf, rt.prismScratch);
    rt.prismBlas = p.as; rt.prismBlasAddr = p.addr;
    BlasOut f = buildOneBLAS(rt.floorPosBuf, 3, rt.floorIdxBuf, 2,
                             rt.floorBlasBuf, rt.floorScratch);
    rt.floorBlas = f.as; rt.floorBlasAddr = f.addr;
    BlasOut q = buildOneBLAS(rt.quadPosBuf, 3, rt.quadIdxBuf, 2,
                             rt.quadBlasBuf, rt.quadScratch);
    rt.quadBlas = q.as; rt.quadBlasAddr = q.addr;
}

static void createTLAS() {
    rt.instBuf = createBuffer(3 * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);

    rt.tlasGeom = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    rt.tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    auto& inst = rt.tlasGeom.geometry.instances;
    inst.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst.arrayOfPointers = VK_FALSE;
    inst.data.deviceAddress = rt.instBuf.addr;

    rt.tlasBuild = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    rt.tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    rt.tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    rt.tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    rt.tlasBuild.geometryCount = 1;
    rt.tlasBuild.pGeometries = &rt.tlasGeom;

    VkAccelerationStructureBuildSizesInfoKHR sz{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    uint32_t prims = 3;
    vkGetAccelerationStructureBuildSizesKHR(vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &rt.tlasBuild, &prims, &sz);

    rt.tlasBuf = createBuffer(sz.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
    rt.tlasScratch = createBuffer(sz.buildScratchSize + 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    VkAccelerationStructureCreateInfoKHR aci{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    aci.buffer = rt.tlasBuf.buf;
    aci.size = sz.accelerationStructureSize;
    aci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VK_CHECK(vkCreateAccelerationStructureKHR(vk.device, &aci, nullptr, &rt.tlas));

    rt.tlasBuild.dstAccelerationStructure = rt.tlas;
    rt.tlasBuild.scratchData.deviceAddress = alignUp(rt.tlasScratch.addr, 256);
}

// TLAS instances: [0] prism with Ry(yaw)*Rz(tilt), [1] static floor at the
// prism base. customIndex selects the material in the closest-hit shader.
static void writeInstanceTransform() {
    float cy = (float)std::cos(g.prismYawY), sy = (float)std::sin(g.prismYawY);
    float cz = (float)std::cos(g.prismTiltZ), sz = (float)std::sin(g.prismTiltZ);
    float R[3][4] = {
        {  cy * cz, -cy * sz,  sy, 0.0f },
        {  sz,       cz,       0.0f, 0.0f },
        { -sy * cz,  sy * sz,  cy, 0.0f },
    };
    VkAccelerationStructureInstanceKHR inst[3]{};
    std::memcpy(&inst[0].transform, R, sizeof(R));
    inst[0].instanceCustomIndex = 0;
    inst[0].mask = 0xFF;
    inst[0].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst[0].accelerationStructureReference = rt.prismBlasAddr;

    float F[3][4] = {
        { 1, 0, 0, 0.0f },
        { 0, 1, 0, -0.281f },      // just below the prism base (y = -0.28)
        { 0, 0, 1, 0.0f },
    };
    std::memcpy(&inst[1].transform, F, sizeof(F));
    inst[1].instanceCustomIndex = 1;
    inst[1].mask = 0xFF;
    inst[1].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst[1].accelerationStructureReference = rt.floorBlasAddr;

    float Q[3][4] = {                       // identity: quad verts are world-space
        { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 },
    };
    std::memcpy(&inst[2].transform, Q, sizeof(Q));
    inst[2].instanceCustomIndex = 2;
    inst[2].mask = 0xFF;
    inst[2].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst[2].accelerationStructureReference = rt.quadBlasAddr;

    std::memcpy(rt.instBuf.mapped, inst, sizeof(inst));
}

static void createPipelines() {
    // ---- RT descriptor set layout (matches common.glsl bindings) ----
    VkDescriptorSetLayoutBinding rtBind[7]{};
    rtBind[0] = { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                  VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
    rtBind[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                  VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
    rtBind[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
    rtBind[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
    rtBind[4] = { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                  VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                  VK_SHADER_STAGE_MISS_BIT_KHR, nullptr };
    rtBind[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                  VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };   // photon fog grid
    rtBind[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                  VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };   // floor caustic map
    VkDescriptorSetLayoutCreateInfo dli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = 7;
    dli.pBindings = rtBind;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &dli, nullptr, &rt.rtSetLayout));

    // ---- resolve descriptor set layout ----
    VkDescriptorSetLayoutBinding resBind[2]{};
    resBind[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    resBind[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    dli.bindingCount = 2;
    dli.pBindings = resBind;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device, &dli, nullptr, &rt.resSetLayout));

    // ---- descriptor pool + sets ----
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
    };
    VkDescriptorPoolCreateInfo dpi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 2;
    dpi.poolSizeCount = 4;
    dpi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(vk.device, &dpi, nullptr, &rt.descPool));

    VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsa.descriptorPool = rt.descPool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &rt.rtSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &dsa, &rt.rtSet));
    dsa.pSetLayouts = &rt.resSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(vk.device, &dsa, &rt.resSet));

    // ---- pipeline layouts ----
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &rt.rtSetLayout;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &pli, nullptr, &rt.rtLayout));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };  // ResolvePC
    pli.pSetLayouts = &rt.resSetLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(vk.device, &pli, nullptr, &rt.resLayout));

    // ---- RT pipeline (rgen / miss / chit) + SBT ----
    VkShaderModule mRgen = loadShader("prism.rgen.spv");
    VkShaderModule mMiss = loadShader("prism.rmiss.spv");
    VkShaderModule mChit = loadShader("prism.rchit.spv");

    VkPipelineShaderStageCreateInfo stages[3]{};
    for (auto& s : stages) s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;      stages[0].module = mRgen; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;        stages[1].module = mMiss; stages[1].pName = "main";
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; stages[2].module = mChit; stages[2].pName = "main";

    VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
    for (auto& gr : groups) {
        gr.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        gr.generalShader = VK_SHADER_UNUSED_KHR;
        gr.closestHitShader = VK_SHADER_UNUSED_KHR;
        gr.anyHitShader = VK_SHADER_UNUSED_KHR;
        gr.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].closestHitShader = 2;

    VkRayTracingPipelineCreateInfoKHR rpi{
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
    rpi.stageCount = 3;
    rpi.pStages = stages;
    rpi.groupCount = 3;
    rpi.pGroups = groups;
    rpi.maxPipelineRayRecursionDepth = 1;   // bounce loops live in rgen
    rpi.layout = rt.rtLayout;
    VK_CHECK(vkCreateRayTracingPipelinesKHR(vk.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            1, &rpi, nullptr, &rt.rtPipe));

    // photon pipeline: photon.rgen + the same thin miss/chit modules
    VkShaderModule mPhot = loadShader("photon.rgen.spv");
    stages[0].module = mPhot;
    VK_CHECK(vkCreateRayTracingPipelinesKHR(vk.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            1, &rpi, nullptr, &rt.photonPipe));
    vkDestroyShaderModule(vk.device, mPhot, nullptr);
    vkDestroyShaderModule(vk.device, mRgen, nullptr);
    vkDestroyShaderModule(vk.device, mMiss, nullptr);
    vkDestroyShaderModule(vk.device, mChit, nullptr);

    // SBT: one aligned record per group
    const uint32_t handleSize = vk.rtProps.shaderGroupHandleSize;
    const uint32_t baseAlign  = vk.rtProps.shaderGroupBaseAlignment;
    const VkDeviceSize record = alignUp(handleSize, baseAlign);
    std::vector<uint8_t> handles(3 * handleSize);
    VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(vk.device, rt.rtPipe, 0, 3,
                                                  handles.size(), handles.data()));
    rt.sbtBuf = createBuffer(record * 3,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    uint8_t* dst = (uint8_t*)rt.sbtBuf.mapped;
    std::memset(dst, 0, (size_t)(record * 3));
    for (int i = 0; i < 3; ++i)
        std::memcpy(dst + i * record, handles.data() + i * handleSize, handleSize);
    rt.sbtRgen = { rt.sbtBuf.addr,              record, record };
    rt.sbtMiss = { rt.sbtBuf.addr + record,     record, record };
    rt.sbtHit  = { rt.sbtBuf.addr + record * 2, record, record };
    rt.sbtCall = { 0, 0, 0 };

    // photon SBT (separate pipeline -> separate handles)
    std::vector<uint8_t> phHandles(3 * handleSize);
    VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(vk.device, rt.photonPipe, 0, 3,
                                                  phHandles.size(), phHandles.data()));
    rt.photonSbtBuf = createBuffer(record * 3,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    uint8_t* pdst = (uint8_t*)rt.photonSbtBuf.mapped;
    std::memset(pdst, 0, (size_t)(record * 3));
    for (int i = 0; i < 3; ++i)
        std::memcpy(pdst + i * record, phHandles.data() + i * handleSize, handleSize);
    rt.phRgen = { rt.photonSbtBuf.addr,              record, record };
    rt.phMiss = { rt.photonSbtBuf.addr + record,     record, record };
    rt.phHit  = { rt.photonSbtBuf.addr + record * 2, record, record };
    rt.phCall = { 0, 0, 0 };

    // ---- resolve compute pipeline ----
    VkShaderModule mRes = loadShader("resolve.comp.spv");
    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mRes;
    cpi.stage.pName = "main";
    cpi.layout = rt.resLayout;
    VK_CHECK(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &rt.resPipe));
    vkDestroyShaderModule(vk.device, mRes, nullptr);

    // ---- UBO ----
    rt.uboBuf = createBuffer(sizeof(SceneUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);

    // ---- photon storage: fog grid + floor caustic (device local) ----
    rt.gridBuf = createBuffer(kGridBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    rt.causBuf = createBuffer(kCausBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
}

// (re)create render targets and point the descriptor sets at them
static void createTargets() {
    destroyImage(rt.accum);
    destroyImage(rt.outImg);
    rt.accum = createStorageImage(VK_FORMAT_R32G32B32A32_SFLOAT,
        vk.extent.width, vk.extent.height, VK_IMAGE_USAGE_STORAGE_BIT);
    rt.outImg = createStorageImage(VK_FORMAT_R8G8B8A8_UNORM,
        vk.extent.width, vk.extent.height,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // transition both to GENERAL once
    VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(vk.cmd, &cbi));
    barrierImage(vk.cmd, rt.accum.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                 0, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    barrierImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                 0, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    VK_CHECK(vkEndCommandBuffer(vk.cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk.cmd;
    VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(vk.queue));

    // descriptor writes
    VkWriteDescriptorSetAccelerationStructureKHR asw{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    asw.accelerationStructureCount = 1;
    asw.pAccelerationStructures = &rt.tlas;
    VkDescriptorImageInfo accumInfo{ VK_NULL_HANDLE, rt.accum.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, rt.outImg.view, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo posInfo{ rt.posBuf.buf, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo idxInfo{ rt.idxBuf.buf, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo uboInfo{ rt.uboBuf.buf, 0, sizeof(SceneUBO) };
    VkDescriptorBufferInfo gridInfo{ rt.gridBuf.buf, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo causInfo{ rt.causBuf.buf, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet w[9]{};
    for (auto& x : w) x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = rt.rtSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[0].pNext = &asw;
    w[1].dstSet = rt.rtSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &accumInfo;
    w[2].dstSet = rt.rtSet; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &posInfo;
    w[3].dstSet = rt.rtSet; w[3].dstBinding = 3; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[3].pBufferInfo = &idxInfo;
    w[4].dstSet = rt.rtSet; w[4].dstBinding = 4; w[4].descriptorCount = 1;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[4].pBufferInfo = &uboInfo;
    w[5].dstSet = rt.rtSet; w[5].dstBinding = 5; w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[5].pBufferInfo = &gridInfo;
    w[6].dstSet = rt.rtSet; w[6].dstBinding = 6; w[6].descriptorCount = 1;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[6].pBufferInfo = &causInfo;
    w[7].dstSet = rt.resSet; w[7].dstBinding = 0; w[7].descriptorCount = 1;
    w[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[7].pImageInfo = &accumInfo;
    w[8].dstSet = rt.resSet; w[8].dstBinding = 1; w[8].descriptorCount = 1;
    w[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[8].pImageInfo = &outInfo;
    vkUpdateDescriptorSets(vk.device, 9, w, 0, nullptr);
}

// =========================== frame loop =================================
static void updateUBO(uint32_t frameIndex) {
    // orbit camera around the prism
    V3 target{ 0.0f, 0.10f, 0.0f };
    float cy = (float)std::cos(g.yaw), sy = (float)std::sin(g.yaw);
    float cp = (float)std::cos(g.pitch), sp = (float)std::sin(g.pitch);
    V3 eye = target + V3{ sy * cp, sp, cy * cp } * (float)g.dist;
    M4 V = lookAt(eye, target, { 0, 1, 0 });
    M4 P = perspective(40.0f, (float)vk.extent.width / (float)vk.extent.height,
                       0.05f, 100.0f);
    M4 inv = inverse(mul(P, V));

    V3 D = beamD();
    V3 srcPos = beamSrcPos();

    SceneUBO u{};
    std::memcpy(u.invViewProj, inv.m, sizeof(inv.m));
    u.camPos[0] = eye.x; u.camPos[1] = eye.y; u.camPos[2] = eye.z; u.camPos[3] = 0;
    u.lightDir[0] = -D.x; u.lightDir[1] = -D.y; u.lightDir[2] = -D.z;
    u.lightDir[3] = std::cos(0.35f * (float)M_PI / 180.0f);
    u.srcPosRadius[0] = srcPos.x; u.srcPosRadius[1] = srcPos.y;
    u.srcPosRadius[2] = srcPos.z; u.srcPosRadius[3] = kBeamRadius;
    u.dimsFrameSpp[0] = vk.extent.width;
    u.dimsFrameSpp[1] = vk.extent.height;
    u.dimsFrameSpp[2] = frameIndex;
    u.dimsFrameSpp[3] = 1;                                  // spp per frame
    u.misc[0] = g.sf10 ? 1u : 0u;
    u.misc[1] = 10u;                                        // maxDepth
    u.misc[2] = bits(420.0f);                                // beam intensity
    u.misc[3] = bits(kFog[g.fogIdx]);                       // fog sigma
    std::memcpy(rt.uboBuf.mapped, &u, sizeof(u));
}

static void recordFrame(uint32_t swapIdx, uint32_t frameIndex) {
    VK_CHECK(vkResetCommandBuffer(vk.cmd, 0));
    VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(vk.cmd, &cbi));

    // 1) rebuild the TLAS with the current prism transform (3 instances)
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = 3;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    vkCmdBuildAccelerationStructuresKHR(vk.cmd, 1, &rt.tlasBuild, &pRange);
    VkMemoryBarrier asb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    asb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    asb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(vk.cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 1, &asb, 0, nullptr, 0, nullptr);

    // 2) photon pass: forward beam transport into the fog grid + caustic map.
    // On accumulation reset, zero both first.
    if (frameIndex == 0) {
        vkCmdFillBuffer(vk.cmd, rt.gridBuf.buf, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(vk.cmd, rt.causBuf.buf, 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier fb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(vk.cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &fb, 0, nullptr, 0, nullptr);
    }
    vkCmdBindPipeline(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt.photonPipe);
    vkCmdBindDescriptorSets(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            rt.rtLayout, 0, 1, &rt.rtSet, 0, nullptr);
    vkCmdTraceRaysKHR(vk.cmd, &rt.phRgen, &rt.phMiss, &rt.phHit, &rt.phCall,
                      60, 12, 1);                       // PH_APERTURE x PH_LAMBDA

    VkMemoryBarrier pb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    pb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(vk.cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 1, &pb, 0, nullptr, 0, nullptr);

    // 3) camera pass: accumulate spectral XYZ
    vkCmdBindPipeline(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt.rtPipe);
    vkCmdBindDescriptorSets(vk.cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            rt.rtLayout, 0, 1, &rt.rtSet, 0, nullptr);
    vkCmdTraceRaysKHR(vk.cmd, &rt.sbtRgen, &rt.sbtMiss, &rt.sbtHit, &rt.sbtCall,
                      vk.extent.width, vk.extent.height, 1);

    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(vk.cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    // 4) resolve: average, tonemap, gamma -> outImg
    struct { uint32_t frameCount; float exposure; } pc{ frameIndex + 1, g.exposure };
    vkCmdBindPipeline(vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rt.resPipe);
    vkCmdBindDescriptorSets(vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            rt.resLayout, 0, 1, &rt.resSet, 0, nullptr);
    vkCmdPushConstants(vk.cmd, rt.resLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pc);
    vkCmdDispatch(vk.cmd, (vk.extent.width + 7) / 8, (vk.extent.height + 7) / 8, 1);

    // 4) blit outImg -> swapchain image
    barrierImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrierImage(vk.cmd, vk.swapImages[swapIdx], VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit blit{};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.srcOffsets[1] = { (int32_t)vk.extent.width, (int32_t)vk.extent.height, 1 };
    blit.dstOffsets[1] = { (int32_t)vk.extent.width, (int32_t)vk.extent.height, 1 };
    vkCmdBlitImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vk.swapImages[swapIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_NEAREST);

    barrierImage(vk.cmd, vk.swapImages[swapIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    barrierImage(vk.cmd, rt.outImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    VK_CHECK(vkEndCommandBuffer(vk.cmd));
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow) {
    WNDCLASSW wc = {};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"PrismRTWnd";
    RegisterClassW(&wc);

    RECT r = { 0, 0, g.winW, g.winH };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hWnd = CreateWindowW(L"PrismRTWnd", L"PrismRT", WS_OVERLAPPEDWINDOW,
                              CW_USEDEFAULT, CW_USEDEFAULT,
                              r.right - r.left, r.bottom - r.top,
                              nullptr, nullptr, hInst, nullptr);
    if (!hWnd) return 1;

    initVulkan(hWnd, hInst);
    createSwapchain();
    buildBLAS();
    createTLAS();
    createPipelines();
    createTargets();
    ShowWindow(hWnd, nCmdShow);

    uint32_t frameIndex = 0;
    ULONGLONG lastTitle = 0;

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

        writeInstanceTransform();
        updateUBO(frameIndex);

        uint32_t swapIdx = 0;
        VkResult acq = vkAcquireNextImageKHR(vk.device, vk.swapchain, UINT64_MAX,
                                             vk.semAcquire, VK_NULL_HANDLE, &swapIdx);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            g.sizeDirty = true;
            continue;
        }
        VK_CHECK(acq);
        VK_CHECK(vkResetFences(vk.device, 1, &vk.fence));

        recordFrame(swapIdx, frameIndex);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &vk.semAcquire;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &vk.cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &vk.semRender;
        VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, vk.fence));

        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &vk.semRender;
        pi.swapchainCount = 1;
        pi.pSwapchains = &vk.swapchain;
        pi.pImageIndices = &swapIdx;
        VkResult pres = vkQueuePresentKHR(vk.queue, &pi);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR)
            g.sizeDirty = true;
        else
            VK_CHECK(pres);

        ++frameIndex;

        ULONGLONG now = GetTickCount64();
        if (now - lastTitle > 250) {
            wchar_t t[200];
            swprintf(t, 200,
                L"PrismRT - %s   fog %.2f   %u frames   "
                L"[LMB] orbit  [RMB] prism  [wheel] zoom  [G]lass  [F]og  [Esc]",
                g.sf10 ? L"SF10" : L"BK7", (double)kFog[g.fogIdx], frameIndex);
            SetWindowTextW(hWnd, t);
            lastTitle = now;
        }
    }

    vkDeviceWaitIdle(vk.device);
    // OS reclaims everything on exit; explicit teardown omitted deliberately
    DestroyWindow(hWnd);
    return 0;
}
