// layout.h — the numbers the host and the shaders must agree on, in one
// place. Plain #defines so both C++ (PrismRT.cpp) and GLSL (common.glsl)
// can include it; the shaders wrap them in typed constants.
#ifndef PRISMRT_LAYOUT_H
#define PRISMRT_LAYOUT_H

// descriptor set 0 bindings
#define BIND_TLAS      0
#define BIND_ACCUM     1
#define BIND_POSITIONS 2
#define BIND_INDICES   3
#define BIND_SCENE     4
#define BIND_FOG_GRID  5
#define BIND_CAUSTIC   6

// TLAS instance custom indices (what the closest-hit reports)
#define INST_PRISM  0
#define INST_FLOOR  1
#define INST_SOURCE 2

// photon pass launch size: aperture points x spectral bundles
#define PHOTON_APERTURE 60
#define PHOTON_LAMBDA   12

// 3D fog grid: XYZ energy density, 3 uints per cell, fixed point
#define GRID_X 148
#define GRID_Y 52
#define GRID_Z 40
#define GRID_CELL     0.03
#define GRID_ORIGIN_X -1.85
#define GRID_ORIGIN_Y -0.35
#define GRID_ORIGIN_Z -0.60

// 2D floor caustic map: XYZ irradiance over [CAUS_RANGE_MIN, CAUS_RANGE_MAX]^2 in x/z
#define CAUS_X 384
#define CAUS_Z 384
#define CAUS_RANGE_MIN -3.0
#define CAUS_RANGE_MAX  3.0

#define FIXED_POINT_SCALE 32.0   // scale of the atomic uints

#define FLOOR_PLANE_Y -0.281     // floor plane, just below the prism base

#endif // PRISMRT_LAYOUT_H
