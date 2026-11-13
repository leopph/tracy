#ifndef SHADER_INTEROP_H
#define SHADER_INTEROP_H

#ifdef __cplusplus
#include <array>
#include <cstddef>
#include <cstdint>

#define uint std::uint32_t
#define BOOL std::uint32_t
#define float3 std::array<float, 3>

#define PipelineParamIdx(StructureType, ParamName) offsetof(StructureType, member) / 4
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
};


struct RayDispatchParams {
  uint tlas_idx;
  uint uav_idx;
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
  float3 color;
  BOOL is_reflective;
  BOOL is_shadowed;
};


#define COMPUTE_SHADER_TILE_DIM 8

#endif
