#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "spectral.glsl"
#include "common.glsl"

// prism.rmiss — thin miss marker; environment is evaluated in raygen.
layout(location = 0) rayPayloadInEXT Payload pay;
void main() {
    pay.hitT = 1e30;
    pay.inst = 0xFFFFFFFFu;
    pay.N = vec3(0.0);
}
