#include "shader_interop.h"

ConstantBuffer<ResolveParams> g_params : register(b0);


float3 AgxDefaultContrastApprox(float3 const x) {
  float3 x2 = x * x;
  float3 x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}


float3 Expose(float3 const color, float const exposure) {
  return color * exposure;
}


float3 TonemapAgx(float3 color) {
  static float3x3 const agx_mtx =
  {
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
  };

  static float3x3 const agx_mtx_inv =
  {
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116
  };

  static float const min_ev = -12.47393;
  static float const max_ev = 4.026069;

  color = mul(color, agx_mtx);
  color = clamp(log2(max(color, 1e-10)), min_ev, max_ev);
  color = (color - min_ev) / (max_ev - min_ev);
  color = AgxDefaultContrastApprox(color);
  color = mul(color, agx_mtx_inv);
  return pow(max(color, 0), 2.2);
}


float3 GammaCorrect(float3 const color) {
  return pow(color, 1 / 2.2);
}


[numthreads(COMPUTE_SHADER_TILE_DIM, COMPUTE_SHADER_TILE_DIM, 1)]
void main(uint3 const dtid : SV_DispatchThreadID) {
  if (dtid.x >= g_params.width || dtid.y >= g_params.height) {
    return;
  }

  Texture2D<float3> const accum_tex = ResourceDescriptorHeap[g_params.accum_srv];

  float3 const accum = accum_tex[dtid.xy];
  float3 const resolved = accum / g_params.sample_count;
  float3 const exposed = Expose(resolved, g_params.exposure);
  float3 const tonemapped = TonemapAgx(exposed);
  float3 const gamma_corrected = GammaCorrect(tonemapped);

  RWTexture2D<float4> const target_tex = ResourceDescriptorHeap[g_params.target_uav];
  target_tex[dtid.xy] = float4(gamma_corrected, 1.0);
}
