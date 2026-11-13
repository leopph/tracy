#include "shader_interop.h"

ConstantBuffer<ComputeDispatchParams> g_params : register(b0);

static float3 const camera = float3(0, 1.5, -7);
static float3 const light = float3(0, 200, 0);
static float3 const sky_top = float3(0.24, 0.44, 0.72);
static float3 const sky_bottom = float3(0.75, 0.86, 0.93);


[numthreads(COMPUTE_SHADER_TILE_DIM, COMPUTE_SHADER_TILE_DIM, 1)]
void main(uint3 const dtid : SV_DispatchThreadID) {
  if (dtid.x >= g_params.width || dtid.y >= g_params.height) {
    return;
  }

  uint2 const idx = dtid.xy;
  float2 const size = float2(g_params.width, g_params.height);
  float2 const uv = idx / size;
  float3 const target = float3(
    (uv.x * 2 - 1) * 1.8 * (size.x / size.y),
    (1 - uv.y) * 4 - 2 + camera.y,
    0
  );

  RayDesc cam_ray;
  cam_ray.Origin = camera;
  cam_ray.Direction = target - camera;
  cam_ray.TMin = 0.001;
  cam_ray.TMax = 1000;

  float3 color = float3(0, 0, 0);
  bool allow_reflection = true;

  RaytracingAccelerationStructure const as = ResourceDescriptorHeap[g_params.as_srv];

  StructuredBuffer<GpuInstance> const instances = ResourceDescriptorHeap[g_params.inst_table_srv];
  uint inst_count;
  uint inst_stride;
  instances.GetDimensions(inst_count, inst_stride);

  StructuredBuffer<GpuGeometry> const geometries = ResourceDescriptorHeap[g_params.geom_table_srv];
  uint geom_count;
  uint geom_stride;
  geometries.GetDimensions(geom_count, geom_stride);

  StructuredBuffer<GpuMaterial> const materials = ResourceDescriptorHeap[g_params.mtl_table_srv];
  uint mtl_count;
  uint mtl_stride;
  materials.GetDimensions(mtl_count, mtl_stride);

  RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
  q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, cam_ray);

  bool run_ray_loop = true;

  while (run_ray_loop) {
    q.Proceed(); // Forcing opaque so single call is enough

    switch (q.CommittedStatus()) {
      // Miss
      case COMMITTED_NOTHING: {
        float const slope = normalize(q.WorldRayDirection()).y;
        float const t = saturate(slope * 5 + 0.5);
        color = lerp(sky_bottom, sky_top, t);
        run_ray_loop = false;
        break;
      }

      case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        // This shouldn't be hit since we don't have procedural primitives
        color = float3(1, 0, 1); // display error color
        run_ray_loop = false;
        break;
      }

      case COMMITTED_TRIANGLE_HIT: {
        uint const inst_idx = q.CommittedInstanceID();

        if (inst_idx >= inst_count) {
          // This shouldn't happen
          color = float3(1, 0, 1); // display error color
          run_ray_loop = false;
          break;
        }

        GpuInstance const inst = instances[inst_idx];

        if (inst.geom_idx >= geom_count || inst.mtl_idx >= mtl_count) {
          // This shouldn't happen
          color = float3(1, 0, 1); // display error color
          run_ray_loop = false;
          break;
        }

        GpuGeometry const geom = geometries[inst.geom_idx];

        StructuredBuffer<uint> const indices = ResourceDescriptorHeap[geom.idx_buf_srv];
        uint index_count;
        uint index_stride;
        indices.GetDimensions(index_count, index_stride);

        uint3 const index_positions = uint3(0, 1, 2) + q.CommittedPrimitiveIndex() * 3;

        if (index_positions.z >= index_count) {
          // This shouldn't happpen
          color = float3(1, 0, 1); // display error color
          run_ray_loop = false;
          break;
        }

        uint3 const prim_indices = uint3(indices[index_positions.x], indices[index_positions.y],
          indices[index_positions.z]);

        StructuredBuffer<float3> const vertices = ResourceDescriptorHeap[geom.vtx_buf_srv];
        uint vertex_count;
        uint vertex_stride;
        vertices.GetDimensions(vertex_count, vertex_stride);

        if (prim_indices.x >= vertex_count || prim_indices.y >= vertex_count || prim_indices.z >= vertex_count) {
          // This shouldn't happen
          color = float3(1, 0, 1); // display error color
          run_ray_loop = false;
          break;
        }

        float3 const pos = q.WorldRayOrigin() + q.WorldRayDirection() * q.CommittedRayT();

        float3 const vtx0 = vertices[prim_indices.x];
        float3 const vtx1 = vertices[prim_indices.y];
        float3 const vtx2 = vertices[prim_indices.z];
        float3 const normal_os = normalize(cross(vtx2 - vtx0, vtx1 - vtx0));
        float3 const normal_ws = normalize(mul(normal_os, (float3x3)q.CommittedObjectToWorld4x3()));

        GpuMaterial const mtl = materials[inst.mtl_idx];
        color += mtl.color * (saturate(dot(normal_ws, normalize(light))) + 0.33);

        if (mtl.is_reflective && allow_reflection) {
          float3 const reflection_ray_dir = reflect(normalize(q.WorldRayDirection()), normal_ws);

          RayDesc mirror_ray;
          mirror_ray.Origin = pos;
          mirror_ray.Direction = reflection_ray_dir;
          mirror_ray.TMin = 0.001;
          mirror_ray.TMax = 1000;

          allow_reflection = false;
          q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, mirror_ray);
          break;
        }

        if (mtl.is_shadowed) {
          RayDesc shadow_ray;
          shadow_ray.Origin = pos;
          shadow_ray.Direction = light - pos;
          shadow_ray.TMin = 0.001;
          shadow_ray.TMax = 1;

          RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> sq;
          sq.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, shadow_ray);
          sq.Proceed();

          if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            color /= 2;
          }
        }

        run_ray_loop = false;
        break;
      }

      default: {
        // This shouldn't be hit since we've exhausted all possible commit status values
        color = float3(1, 1, 0); // display error color
        run_ray_loop = false;
        break;
      }
    }
  }

  RWTexture2D<float4> uav = ResourceDescriptorHeap[g_params.target_uav];
  uav[idx] += float4(color, 1);
}
