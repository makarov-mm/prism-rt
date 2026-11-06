#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#include "spectral.glsl"
#include "common.glsl"

// prism.rchit — thin hit reporter. All material logic lives in the raygen
// shaders (deterministic Whitted-style), so this just returns what was hit.

layout(location = 0) rayPayloadInEXT Payload pay;
hitAttributeEXT vec2 attribs;

layout(set = 0, binding = BIND_POSITIONS, scalar) readonly buffer Positions { vec3 pos[]; };
layout(set = 0, binding = BIND_INDICES,   scalar) readonly buffer Indices  { uint idx[]; };

void main() {
    pay.hitT = gl_HitTEXT;
    pay.inst = gl_InstanceCustomIndexEXT;
    if (pay.inst == uint(INST_PRISM)) {      // prism: geometric normal
        uint base = 3u * uint(gl_PrimitiveID);
        vec3 p0 = pos[idx[base + 0u]];
        vec3 p1 = pos[idx[base + 1u]];
        vec3 p2 = pos[idx[base + 2u]];
        vec3 nObj = normalize(cross(p1 - p0, p2 - p0));
        pay.N = normalize(nObj * mat3(gl_WorldToObjectEXT));
    } else if (pay.inst == uint(INST_FLOOR)) {
        pay.N = vec3(0.0, 1.0, 0.0);
    } else {
        pay.N = normalize(ubo.lightDir.xyz); // source quad (unused by shading)
    }
}
