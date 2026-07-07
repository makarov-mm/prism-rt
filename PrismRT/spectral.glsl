// spectral.glsl — portable spectral core, GLSL mirror of the CPU Spectral.h.
// Same math, same coefficients. Include with GL_GOOGLE_include_directive.
#ifndef SPECTRAL_GLSL
#define SPECTRAL_GLSL

// ---- dispersion, lambda in nm ----
float sellmeierSF10(float l) {
    float l2 = (l * 1e-3) * (l * 1e-3);
    const float B1 = 1.62153902, B2 = 0.256287842, B3 = 1.64447552;
    const float C1 = 0.0122241457, C2 = 0.0595736775, C3 = 147.468793;
    return sqrt(1.0 + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2) + B3 * l2 / (l2 - C3));
}
float sellmeierBK7(float l) {
    float l2 = (l * 1e-3) * (l * 1e-3);
    const float B1 = 1.03961212, B2 = 0.231792344, B3 = 1.01046945;
    const float C1 = 0.00600069867, C2 = 0.0200179144, C3 = 103.560653;
    return sqrt(1.0 + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2) + B3 * l2 / (l2 - C3));
}

// ---- exact unpolarised Fresnel reflectance ----
float fresnelDielectric(float n1, float n2, float cosi, float cost) {
    float Rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
    float Rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);
    return 0.5 * (Rs * Rs + Rp * Rp);
}

// ---- CIE 1931 XYZ from wavelength (Wyman/Sloan/Shirley, JCGT 2013) ----
float cie_g(float x, float m, float s1, float s2) {
    float t = (x - m) * (x < m ? s1 : s2);
    return exp(-0.5 * t * t);
}
vec3 cieXYZ(float l) {
    float X = 1.056 * cie_g(l, 599.8, 0.0264, 0.0323)
            + 0.362 * cie_g(l, 442.0, 0.0624, 0.0374)
            - 0.065 * cie_g(l, 501.1, 0.0490, 0.0382);
    float Y = 0.821 * cie_g(l, 568.8, 0.0213, 0.0247)
            + 0.286 * cie_g(l, 530.9, 0.0613, 0.0322);
    float Z = 1.217 * cie_g(l, 437.0, 0.0845, 0.0278)
            + 0.681 * cie_g(l, 459.0, 0.0385, 0.0725);
    return vec3(X, Y, Z);
}

// ---- XYZ (D65) -> linear sRGB ----
vec3 xyzToSRGB(vec3 c) {
    return vec3(
        dot(c, vec3( 3.2406, -1.5372, -0.4986)),
        dot(c, vec3(-0.9689,  1.8758,  0.0415)),
        dot(c, vec3( 0.0557, -0.2040,  1.0570)));
}

// integral of y-bar over [400,700] with the fit above; keeps flat spectra white
const float CIE_Y_INTEGRAL = 106.8813;
const float LAMBDA_MIN = 400.0;
const float LAMBDA_MAX = 700.0;

// ---- small PCG rng ----
uint pcgHash(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) { return float(pcgHash(s)) * (1.0 / 4294967296.0); }

#endif // SPECTRAL_GLSL
