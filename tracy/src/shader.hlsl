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
        float2 const bary = q.CommittedTriangleBarycentrics();

        switch (q.CommittedInstanceID()) {
          // Cube
          case 0: {
            uint const tri = q.CommittedPrimitiveIndex() / 2;
            float3 const normal = (tri.xxx % 3 == uint3(0, 1, 2)) * (tri < 3 ? -1 : 1);
            float3 const world_normal = normalize(mul(normal, (float3x3)q.CommittedObjectToWorld4x3()));

            color = abs(normal) / 3 + 0.5;
            if (bary.x < 0.03 || bary.y < 0.03) {
              color = 0.25.rrr;
            }

            color *= saturate(dot(world_normal, normalize(light))) + 0.33;
            run_ray_loop = false;
            break;
          }

          // Mirror
          case 1: {
            if (!allow_reflection) {
              run_ray_loop = false;
              break;
            }

            float3 const pos = q.WorldRayOrigin() + q.WorldRayDirection() * q.CommittedRayT();
            float3 const normal = normalize(mul(float3(0, 1, 0), (float3x3)q.CommittedObjectToWorld4x3()));
            float3 const reflected = reflect(normalize(q.WorldRayDirection()), normal);

            RayDesc mirror_ray;
            mirror_ray.Origin = pos;
            mirror_ray.Direction = reflected;
            mirror_ray.TMin = 0.001;
            mirror_ray.TMax = 1000;

            allow_reflection = false;
            q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, mirror_ray);
            break;
          }

          // Floor
          case 2: {
            float3 const pos = q.WorldRayOrigin() + q.WorldRayDirection() * q.CommittedRayT();

            bool2 const pattern = frac(pos.xz) > 0.5;
            color = (pattern.x ^ pattern.y ? 0.6 : 0.4).rrr;

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
            
            run_ray_loop = false;
            break;
          }

          default: {
            // This shouldn't be hit since we covered all instance IDs
            color = float3(0, 1, 1); // display error color
            run_ray_loop = false;
            break;
          }
        }

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
  uav[idx] = float4(color, 1);
}
