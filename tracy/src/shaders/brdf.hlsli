#ifndef BRDF_HLSLI
#define BRDF_HLSLI

#include "utility.hlsli"

// alpha is the square of the material input roughness

float AlphaFromRoughness(float const roughness) {
  return max(roughness * roughness, 1e-3);
}


float DistributionTrowbridgeReitz(float const n_dot_h, float const alpha) {
  float const alpha2 = alpha * alpha;
  float const n_dot_h2 = n_dot_h * n_dot_h;
  float denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
  denom = kPi * denom * denom;
  return alpha2 / denom;
}


float SmithG1Ggx(float const n_dot_x, float const alpha) {
  float const alpha2 = alpha * alpha;
  float const n_dot_x2 = n_dot_x * n_dot_x;
  return 2.0 * n_dot_x / (n_dot_x + sqrt(alpha2 + (1.0 - alpha2) * n_dot_x2));
}


// Separable Smith shadowing-masking G2 = G1(v) * G1(l) for GGX.
float SmithG2SeparableGgx(float const n_dot_v, float const n_dot_l, float const alpha) {
  return SmithG1Ggx(n_dot_v, alpha) * SmithG1Ggx(n_dot_l, alpha);
}


float3 FresnelSchlick(float const v_dot_h, float3 const f0) {
  return f0 + (1.0 - f0) * pow(saturate(1.0 - v_dot_h), 5.0);
}


float3 CalcF0(float3 const base_color, float const metallic) {
  // calculate reflectance at normal incidence; if dielectric (like plastic) use F0 
  // of 0.04 and if it's a metal, use the base color as F0 (metallic workflow)
  return lerp((float3)0.04, base_color, metallic);
}


float3 CalcDiffuseAlbedo(float3 const base_color, float const metallic) {
  // metals have no diffuse reflection (metallic workflow)
  return base_color * (1.0 - metallic);
}


// Returns a sampled incoming L direction
float3 SampleCosineMalley(float const random1, float const random2) {
  float const r = sqrt(random1);
  float const theta = 2.0f * kPi * random2;

  float const x = r * cos(theta);
  float const y = r * sin(theta);
  float const z = sqrt(max(0.0f, 1.0f - random1));

  return float3(x, y, z);
}


// Eric Heitz VNDF sampling
// Returns a sampled normal
float3 SampleGgxVndf(float3 const view_ts, float alpha_x, float alpha_y, float random1, float random2) {
  float3 view_stretched = normalize(float3(alpha_x * view_ts.x, alpha_y * view_ts.y, view_ts.z));

  float lensq = view_stretched.x * view_stretched.x + view_stretched.y * view_stretched.y;
  float3 T1 = lensq > 0 ? float3(-view_stretched.y, view_stretched.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
  float3 T2 = cross(view_stretched, T1);

  float r = sqrt(random1);
  float phi = 2.0 * kPi * random2;
  float t1 = r * cos(phi);
  float t2 = r * sin(phi);
  float s = 0.5 * (1.0 + view_stretched.z);
  t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

  float3 normal_stretched = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * view_stretched;

  float3 normal_ts = normalize(float3(alpha_x * normal_stretched.x, alpha_y * normal_stretched.y,
    max(0.0, normal_stretched.z)));
  return normal_ts;
}


float MalleyPdf(float3 const l_ts) {
  return l_ts.z / kPi;
}


float GgxVndfPdf(float3 const view_ts, float3 const h_ts, float3 const l_ts, float const alpha) {
  if (view_ts.z <= 0.0 || h_ts.z <= 0.0 ||
      l_ts.z <= 0.0 || dot(view_ts, h_ts) <= 0.0) {
    return 0.0;
  }

  float D = DistributionTrowbridgeReitz(h_ts.z, alpha);
  float G1 = SmithG1Ggx(view_ts.z, alpha);

  return D * G1 / (4.0 * view_ts.z);
}

#endif
