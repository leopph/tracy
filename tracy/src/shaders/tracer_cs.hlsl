#include "pcg4d.hlsli"
#include "shader_interop.h"
#include "utility.hlsli"

ConstantBuffer<ComputeDispatchParams> g_params : register(b0);

static float3 const cam_pos_ws = float3(0, 1.5, -7);
static float const PI = 3.14159265359f;


float3 SampleSkyRadiance(float3 const ray_dir) {
  // The solid angle of the sun disk in steradians (sr)
  float const sun_solid_angle = 2 * PI * (1 - g_params.sun_radius_cos);
  float const sun_cutoff = dot(ray_dir, g_params.dir_to_sun) > g_params.sun_radius_cos;
  float const sun_radiance = g_params.sun_irradiance / sun_solid_angle;
  float3 const sun = sun_radiance * g_params.sun_tint * sun_cutoff;
  float3 const sky_tint = lerp(g_params.sky_horizon_tint, g_params.sky_zenith_tint, saturate(ray_dir.y));
  float3 const atmos = g_params.sky_radiance * sky_tint;
  return sun + atmos;
}


float3 SampleCosineMalley(float const random1, float const random2) {
  float const r = sqrt(random1);
  float const theta = 2.0f * PI * random2;

  float const x = r * cos(theta);
  float const y = r * sin(theta);
  float const z = sqrt(max(0.0f, 1.0f - random1));

  return float3(x, y, z);
}


// normal must be normalized.
void OnbDuff(float3 const normal, out float3 tangent, out float3 bitangent) {
  float s = (normal.z >= 0.0f) ? 1.0f : -1.0f;
  float a = -1.0f / (s + normal.z);
  float b = normal.x * normal.y * a;

  tangent = float3(
    1.0f + s * normal.x * normal.x * a,
    s * b,
    -s * normal.x
  );

  bitangent = float3(
    b,
    s + normal.y * normal.y * a,
    -normal.y
  );
}


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
    (1 - uv.y) * 4 - 2 + cam_pos_ws.y,
    0
  );

  RayDesc cam_ray;
  cam_ray.Origin = cam_pos_ws;
  cam_ray.Direction = target - cam_pos_ws;
  cam_ray.TMin = 0.001;
  cam_ray.TMax = 1000;

  float3 radiance = float3(0, 0, 0);
  float3 throughput = float3(1, 1, 1);

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

  for (uint bounce = 0; run_ray_loop && bounce <= g_params.max_bounces; ++bounce) {
    q.Proceed(); // Forcing opaque so single call is enough

    switch (q.CommittedStatus()) {
      // Miss
      case COMMITTED_NOTHING: {
        float3 const emission = SampleSkyRadiance(normalize(q.WorldRayDirection()));
        radiance += throughput * emission;
        run_ray_loop = false;
        break;
      }

      case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
        // This shouldn't be hit since we don't have procedural primitives
        radiance = float3(1, 0, 1); // display error color
        run_ray_loop = false;
        break;
      }

      case COMMITTED_TRIANGLE_HIT: {
        uint const inst_idx = q.CommittedInstanceID();

        if (inst_idx >= inst_count) {
          // This shouldn't happen
          radiance = float3(1, 0, 1); // display error color
          run_ray_loop = false;
          break;
        }

        GpuInstance const inst = instances[inst_idx];

        if (inst.geom_idx >= geom_count || inst.mtl_idx >= mtl_count) {
          // This shouldn't happen
          radiance = float3(1, 0, 1); // display error color
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
          radiance = float3(1, 0, 1); // display error color
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
          radiance = float3(1, 0, 1); // display error color
          run_ray_loop = false;
          break;
        }

        float3 const pos = q.WorldRayOrigin() + q.WorldRayDirection() * q.CommittedRayT();

        float3 const vtx0 = vertices[prim_indices.x];
        float3 const vtx1 = vertices[prim_indices.y];
        float3 const vtx2 = vertices[prim_indices.z];
        float3 const normal_os = normalize(cross(vtx2 - vtx0, vtx1 - vtx0));
        float3 normal_ws = normalize(mul(normal_os, (float3x3)q.CommittedObjectToWorld4x3()));

        // Have the normal always be on the same side of the triangle the ray hits it
        if (dot(normal_ws, q.WorldRayDirection()) > 0) {
          normal_ws = -normal_ws;
        }

        GpuMaterial const mtl = materials[inst.mtl_idx];
        radiance += throughput * mtl.emission;
        throughput *= mtl.base_color;

        // Set up a continuation ray
        RayDesc next_ray;
        next_ray.Origin = pos;
        next_ray.TMin = 0.001;
        next_ray.TMax = 1000;

        if (mtl.is_reflective) {
          // If the material is reflective, we're tracing a specular bounce
          // is_reflective == (metallic == 1 && roughness == 0)

          float3 const refl_dir_ws = reflect(normalize(q.WorldRayDirection()), normal_ws);
          next_ray.Direction = refl_dir_ws;
        } else {
          // Otherwise we're tracing a diffuse bounce

          uint4 const pcg = pcg4d(uint4(dtid.xy, g_params.sample_count, bounce));
          float3 const refl_dir_ts = SampleCosineMalley(UintToFloat(pcg.x), UintToFloat(pcg.y));

          float3 tanget_ws;
          float3 bitangent_ws;
          OnbDuff(normal_ws, tanget_ws, bitangent_ws);

          float3x3 const tangent_to_world = float3x3(tanget_ws, bitangent_ws, normal_ws);
          float3 const refl_dir_ws = mul(refl_dir_ts, tangent_to_world);
          next_ray.Direction = refl_dir_ws;
        }

        q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, next_ray);
        break;
      }

      default: {
        // This shouldn't be hit since we've exhausted all possible commit status values
        radiance = float3(1, 1, 0); // display error color
        run_ray_loop = false;
        break;
      }
    }
  }

  RWTexture2D<float4> uav = ResourceDescriptorHeap[g_params.target_uav];
  uav[idx] += float4(radiance, 1);
}
