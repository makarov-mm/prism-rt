// common.glsl — thin payload, scene UBO, photon-grid layout, shared scene
// constants. MUST match PrismRT.cpp (std140 UBO, grid dimensions, scale).
#ifndef COMMON_GLSL
#define COMMON_GLSL

// Thin payload: closest-hit only reports the hit. ALL material logic lives in
// the raygen shaders — deterministic Whitted-style shading, no RNG in payload.
struct Payload {
    float hitT;     // 1e30 on miss
    uint  inst;     // instanceCustomIndex; 0xFFFFFFFF on miss
    vec3  N;        // geometric normal, unoriented (valid for the prism)
};

layout(binding = 4, set = 0, std140) uniform SceneUBO {
    mat4  invViewProj;
    vec4  camPos;
    vec4  lightDir;       // xyz toward light, w = cos(half-angle) of the source
    vec4  srcPosRadius;   // xyz aperture centre, w disk radius
    uvec4 dimsFrameSpp;   // w, h, frameIndex, unused
    uvec4 misc;           // glassSF10, maxDepth, bits(intensity), bits(fogSigma)
} ubo;

#define U_DIMS       (ubo.dimsFrameSpp.xy)
#define U_FRAME      (ubo.dimsFrameSpp.z)
#define U_GLASS_SF10 (ubo.misc.x)
#define U_MAXDEPTH   (ubo.misc.y)
#define U_INTENSITY  (uintBitsToFloat(ubo.misc.z))
#define U_FOG_SIGMA  (uintBitsToFloat(ubo.misc.w))
#define U_COSHALF    (ubo.lightDir.w)

// ---- forward photon transport storage (fixed point, atomically added) ----
// 3D fog grid: XYZ energy density; 2D floor caustic map: XYZ irradiance.
// The photon pass fills them; the camera pass reads them. Cleared on reset;
// both accumulate over frames, normalized by (frameIndex + 1).
const int   GX = 148, GY = 52, GZ = 40;
const float GRID_H = 0.03;
const vec3  GRID_MIN = vec3(-1.85, -0.35, -0.60);
const int   CX = 384, CZ = 384;
const float CAUS_MIN = -3.0, CAUS_MAX = 3.0;
const float FP_SCALE = 32.0;          // fixed-point scale for atomic uints

layout(binding = 5, set = 0, std430) buffer FogGrid { uint fogGrid[]; };   // GX*GY*GZ*3
layout(binding = 6, set = 0, std430) buffer CausMap { uint causMap[]; };   // CX*CZ*3

// ---- shared scene constants ----
const vec3  SUN_DIR      = vec3(-0.4082483, 0.8164966, 0.4082483);
const float SUN_COS_IN   = 0.99925;
const float SUN_COS_OUT  = 0.99890;
const float SUN_RADIANCE = 60.0;
const float SUN_E        = 2.4;
const float FLOOR_Y      = -0.281;

// XYZ integrals of the achromatic material spectra (precomputed)
const vec3 XYZ_WHITE   = vec3(106.6534, 106.8575, 106.3342);
const vec3 XYZ_HORIZON = vec3( 88.5078,  89.3285,  94.9784);
const vec3 XYZ_ZENITH  = vec3( 46.3290,  49.3488,  76.4997);
const vec3 XYZ_BELOW   = vec3( 37.3287,  37.4001,  37.2170);
const vec3 XYZ_AMB     = vec3( 62.4670,  64.0195,  77.0980);
const vec3 R_TILE_A    = vec3(0.85);
const vec3 R_TILE_B    = vec3(0.50960, 0.52789, 0.69962);

float blueW(float lambda) { return clamp((620.0 - lambda) / 170.0, 0.0, 1.0); }

float glassN(float lambda) {
    return (U_GLASS_SF10 == 1u) ? sellmeierSF10(lambda) : sellmeierBK7(lambda);
}

// emissive disk profile at a source-quad hit (0 outside the disk)
float diskEmit(vec3 hitP, vec3 I) {
    float c = dot(I, normalize(ubo.lightDir.xyz));
    float ang = smoothstep(U_COSHALF - 0.0006, U_COSHALF + 0.0002, c);
    if (ang <= 0.0) return 0.0;
    float r = length(hitP - ubo.srcPosRadius.xyz);
    return ang * (1.0 - smoothstep(ubo.srcPosRadius.w * 0.92, ubo.srcPosRadius.w, r));
}

// one refraction event; returns false on TIR; oneMinusF = deterministic weight
bool refractAt(vec3 I, vec3 Ng, float nG, out vec3 outDir, out float oneMinusF) {
    vec3 N = Ng;
    float cosi = dot(I, Ng);
    float n1, n2;
    if (cosi > 0.0) { N = -Ng; n1 = nG;  n2 = 1.0; }
    else            {          n1 = 1.0; n2 = nG; cosi = -cosi; }
    float eta = n1 / n2;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (k < 0.0) { outDir = vec3(0.0); oneMinusF = 0.0; return false; }
    float cost = sqrt(k);
    oneMinusF = 1.0 - fresnelDielectric(n1, n2, cosi, cost);
    outDir = normalize(eta * I + (eta * cosi - cost) * N);
    return true;
}
#endif // COMMON_GLSL
