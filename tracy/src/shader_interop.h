#ifndef SHADER_INTEROP_H
#define SHADER_INTEROP_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

#define uint std::uint32_t

#define PipelineParamIdx(StructureType, ParamName) offsetof(StructureType, member) / 4
#endif

struct ComputeDispatchParams {
  uint as_srv;
  uint target_uav;
  uint width;
  uint height;
};


struct RayDispatchParams {
  uint tlas_idx;
  uint uav_idx;
};


#define COMPUTE_SHADER_TILE_DIM 8

#endif
