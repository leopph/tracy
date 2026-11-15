#include "brdf.hlsli"
#include "pcg4d.hlsli"
#include "shader_interop.h"
#include "utility.hlsli"

ConstantBuffer<ComputeDispatchParams> g_params : register(b0);

static float3 const cam_pos_ws = float3(0, 1.5, -7);


float3 SampleSkyRadiance(float3 const ray_dir) {
  // The solid angle of the sun disk in steradians (sr)
  float const sun_solid_angle = 2 * kPi * (1 - g_params.sun_radius_cos);
  float const sun_cutoff = dot(ray_dir, g_params.dir_to_sun) > g_params.sun_radius_cos;
  float const sun_radiance = g_params.sun_irradiance / sun_solid_angle;
  float3 const sun = sun_radiance * g_params.sun_tint * sun_cutoff;
  float3 const sky_tint = lerp(g_params.sky_horizon_tint, g_params.sky_zenith_tint, saturate(ray_dir.y));
  float3 const atmos = g_params.sky_radiance * sky_tint;
  return sun + atmos;
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
  cam_ray.Direction = normalize(target - cam_pos_ws);
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

        float3 const pos_ws = q.WorldRayOrigin() + q.WorldRayDirection() * q.CommittedRayT();

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

        // Extract material properties

        float3 const F0 = CalcF0(mtl.base_color, mtl.metallic);
        float3 const diffuse_albedo = CalcDiffuseAlbedo(mtl.base_color, mtl.metallic);
        float const alpha = AlphaFromRoughness(mtl.roughness);

        // Build orthonormal basis

        float3 tanget_ws;
        float3 bitangent_ws;
        OnbDuff(normal_ws, tanget_ws, bitangent_ws);
        float3x3 const tangent_to_world = float3x3(tanget_ws, bitangent_ws, normal_ws);

        float3 const view_ws = -q.WorldRayDirection();
        float3 const view_ts = mul(tangent_to_world, view_ws);
        float3 l_ts;

        // Random numbers
        uint4 const pcg = pcg4d(uint4(dtid.xy, g_params.sample_count, bounce));

        // 50 percent chance to trace a specular ray, 50 to trace a diffuse one
        float const p_spec = 0.5;

        if (UintToFloat(pcg.w) < p_spec) {
          // Specular
          float3 const h_ts = SampleGgxVndf(view_ts, alpha, alpha, UintToFloat(pcg.x), UintToFloat(pcg.y));
          l_ts = reflect(-view_ts, h_ts);
        } else {
          // Diffuse
          l_ts = SampleCosineMalley(UintToFloat(pcg.x), UintToFloat(pcg.y));
        }

        float3 const n_ts = float3(0, 0, 1);

        // If we sampled a reflection direction on the other side of the triangle, end the path
        if (dot(n_ts, l_ts) <= 0) {
          run_ray_loop = false;
          break;
        }

        float3 const h = normalize(view_ts + l_ts);
        float const n_dot_h = saturate(dot(n_ts, h));
        float const n_dot_v = saturate(dot(n_ts, view_ts));
        float const n_dot_l = saturate(dot(n_ts, l_ts));
        float const v_dot_h = saturate(dot(view_ts, h));

        float const D = DistributionTrowbridgeReitz(n_dot_h, alpha);
        float const G = SmithG2SeparableGgx(n_dot_v, n_dot_l, alpha);
        float3 const F = FresnelSchlick(v_dot_h, F0);

        // How strongly light arriving from l is specularly reflected toward v
        float3 const f_spec = (D * G * F) / (4.0 * n_dot_v * n_dot_l);
        // How strongly light arriving from l that enters the surface and scatters back out goes toward v
        float3 const f_diff = (1 - F) * diffuse_albedo / kPi;
        float3 const f = f_spec + f_diff;

        // Density with which the specular sampler picks l
        float const pdf_spec = GgxVndfPdf(view_ts, h, l_ts, alpha);
        // Density with which the diffuse sampler picks l
        float const pdf_diff = MalleyPdf(l_ts);
        // Density with which this bounce picks l overall
        float const pdf = p_spec * pdf_spec + (1 - p_spec) * pdf_diff;
        throughput *= f * n_dot_l / pdf;

        // Russian roulette: end the path with some probability based on throughput.
        // Skipped for the first bounces, which carry most of the energy.

        if (g_params.rr_enabled && bounce >= g_params.rr_start_bounce) {
          float const survival = max(max(max(throughput.r, throughput.g), throughput.b), 0.05);

          if (UintToFloat(pcg.z) >= survival) {
            run_ray_loop = false;
            break;
          }

          throughput /= survival;
        }

        // Set up a continuation ray
        RayDesc next_ray;
        next_ray.Origin = pos_ws;
        next_ray.Direction = mul(l_ts, tangent_to_world);
        next_ray.TMin = 0.001;
        next_ray.TMax = 1000;

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
