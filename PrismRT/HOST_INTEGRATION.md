# PrismRT — shader/host contract

> **Superseded as an integration guide:** the repository now ships a complete
> host (`PrismRT.cpp`). This file stays as the authoritative description of
> the shader-side contract the host implements.

## Descriptor set 0

| binding | type            | stages            | content                          |
|--------:|-----------------|-------------------|----------------------------------|
| 0 | acceleration structure | rgen        | TLAS (one prism instance)        |
| 1 | storage image rgba32f  | rgen        | XYZ accumulation                 |
| 2 | storage buffer (scalar)| rchit       | positions, tight `vec3[6]`       |
| 3 | storage buffer (scalar)| rchit       | indices, `uint[24]`              |
| 4 | uniform buffer std140  | rgen/rchit/rmiss | `SceneUBO` (below)          |

## SceneUBO (std140, 144 bytes — mirrors `SceneUBO` in PrismRT.cpp)

```
mat4  invViewProj;   // clip -> world
vec4  camPos;        // xyz camera position
vec4  lightDir;      // xyz unit dir TOWARD light, w = cos(half-angle)
vec4  srcPosRadius;  // xyz beam aperture centre, w = aperture radius
uvec4 dimsFrameSpp;  // w, h, frameIndex, spp
uvec4 misc;          // glassSF10, maxDepth, floatBits(intensity), floatBits(fogSigma)
```

Push constants are used only by `resolve.comp`: `{ uint frameCount; float exposure; }`.

## Rendering architecture (v3, deterministic + photon transport)

Two ray-trace dispatches per frame share one descriptor set and the thin
hit-reporting closest-hit/miss shaders:

1. **photon.rgen** (60 x 12 threads): traces the beam FORWARD from the
   aperture — a golden-spiral set of aperture points times 12 spectral
   bundles, jittered per frame. Air segments deposit XYZ energy into a 3D
   fog grid (binding 5); photons landing on the floor deposit into a 2D
   caustic map (binding 6). Fixed-point atomics, accumulated across frames,
   cleared with `vkCmdFillBuffer` on accumulation reset.
2. **prism.rgen** (camera pass): deterministic Whitted-style shading — a
   fixed rotating wavelength set instead of a random lambda, Fresnel as
   weights on both branches instead of a coin flip, interleaved-gradient
   dither instead of white noise. The fog march samples the photon grid
   (zero rays per step). A single frame is already essentially converged.

Why forward transport: backward light connections from fog points through
the prism are physically impossible — the reversed path enters the exit
face near-normal and TIRs at the entry face (the true arriving direction
differs from the beam axis by the full ~50 deg deviation). This is the
classic caustic problem; tracing photons the way the light actually goes
solves it and gives a real rainbow caustic on the floor for free.

## Light model

The light is a collimated circular aperture and the source is **geometry**:
an emissive quad (TLAS instance, customIndex 2) centred at `srcPosRadius.xyz`
with the emitting disk of radius `srcPosRadius.w` (soft edge), collimated
along `-lightDir` within `cosHalf`. Two lessons encoded here: a purely
directional light would illuminate the entire fog volume uniformly (no
visible beam), and evaluating the aperture in the miss shader silently fails
whenever any geometry (the floor) lies beyond the source plane — the miss
shader never runs. Camera rays hitting the quad outside the disk pass
through; connections crossing the plane outside the disk terminate dark.

TLAS instances: 0 = prism (rotates), 1 = floor (static), 2 = source quad
(static).

## Fog (single scattering)

`prism.rgen` marches every **air** segment of the camera path
(`FOG_STEPS` jittered steps, capped at `FOG_MAXDIST`), and at each step traces
a connection toward the light through up to `CONNECT_MAX` deterministic
refractions (`mode = 1` in the payload; TIR blocks the connection). Eye-leg
transmittance is `exp(-sigma * distance)`; light-leg attenuation is
deliberately ignored (thin medium). Phase function is isotropic `1/4π`.

## Payload modes

`mode = 0` — camera path: stochastic Fresnel branch (reflect vs refract),
throughput stays 1 (importance sampled).
`mode = 1` — light connection: deterministic transmission weighted by
`(1-F)`; TIR terminates with zero radiance.

## Per-frame loop (as implemented by the host)

1. wait fence → update instance transform + UBO (reset `frameIndex` on any change)
2. rebuild TLAS (one instance) + AS barrier
3. `vkCmdTraceRaysKHR` (spp = 1 per frame, progressive)
4. barrier → resolve compute (8×8 groups) → barrier
5. blit `outImage` → swapchain image → present
