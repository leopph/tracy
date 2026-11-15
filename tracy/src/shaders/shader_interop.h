#ifndef SHADER_INTEROP_H
#define SHADER_INTEROP_H

#ifdef __cplusplus
#include <array>
#include <cstddef>
#include <cstdint>

#define uint std::uint32_t
#define BOOL std::uint32_t
#define float3 std::array<float, 3>
#define float2 std::array<float, 2>
#define float4x4 std::array<std::array<float, 4>, 4>

#define PipelineParamIdx(StructureType, ParamName) offsetof(StructureType, member) / 4

#define row_major
#else
#define BOOL bool
#endif

struct ComputeDispatchParams {
  uint as_srv;
  uint target_uav;
  uint width;
  uint height;

  uint inst_table_srv;
  uint geom_table_srv;
  uint mtl_table_srv;
  uint sample_count;

  float3 sky_zenith_tint;
  float sky_radiance;

  float3 sky_horizon_tint;
  uint max_bounces;

  float3 dir_to_sun;
  float sun_irradiance;

  float3 sun_tint;
  float sun_radius_cos;

  uint rr_start_bounce;
  BOOL rr_enabled;
};


struct RayDispatchParams {
  uint tlas_idx;
  uint uav_idx;
};


struct ResolveParams {
  uint accum_srv;
  uint sample_count;
  uint target_uav;
  uint width;
  uint height;
  float exposure;
};


struct GpuInstance {
  uint geom_idx;
  uint mtl_idx;
};


struct GpuGeometry {
  uint vtx_buf_srv;
  uint idx_buf_srv;
};


struct GpuMaterial {
  float3 base_color;
  float3 emission;
  BOOL is_reflective;
};


// This should match ImDrawVert
struct ImGuiVertexData {
  float2 pos_os;
  float2 uv;
  uint col;
};


struct ImGuiDrawParams {
  row_major float4x4 proj_mtx;
  uint vb_idx;
  uint tex_idx;
  uint samp_idx;
};


#define COMPUTE_SHADER_TILE_DIM 8
#define INVALID_RES_IDX ((uint)-1)

#endif
