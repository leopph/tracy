struct Payload {
  float3 color;
  bool allow_reflection;
  bool missed;
};

RaytracingAccelerationStructure scene : register(t0);
RWTexture2D<float4> uav : register(u0);

static float3 const camera = float3(0, 1.5, -7);
static float3 const light = float3(0, 200, 0);
static float3 const sky_top = float3(0.24, 0.44, 0.72);
static float3 const sky_bottom = float3(0.75, 0.86, 0.93);


[shader("raygeneration")]
void RayGeneration() {
  uint2 const idx = DispatchRaysIndex().xy;
  float2 const size = DispatchRaysDimensions().xy;
  float2 const uv = idx / size;
  float3 const target = float3((uv.x * 2 - 1) * 1.8 * (size.x / size.y),
                               (1 - uv.y) * 4 - 2 + camera.y,
                               0);

  RayDesc ray;
  ray.Origin = camera;
  ray.Direction = target - camera;
  ray.TMin = 0.001;
  ray.TMax = 1000;

  Payload payload;
  payload.allow_reflection = true;
  payload.missed = false;

  TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

  uav[idx] = float4(payload.color, 1);
}

[shader("miss")]
void Miss(inout Payload payload) {
  float const slope = normalize(WorldRayDirection()).y;
  float const t = saturate(slope * 5 + 0.5);
  payload.color = lerp(sky_bottom, sky_top, t);
  payload.missed = true;
}

void HitCube(inout Payload payload, float2 uv);
void HitMirror(inout Payload payload, float2 uv);
void HitFloor(inout Payload payload, float2 uv);

[shader("closesthit")]
void ClosestHit(inout Payload payload, BuiltInTriangleIntersectionAttributes attribs) {
  float2 const uv = attribs.barycentrics;

  switch (InstanceID()) {
  case 0: {
    HitCube(payload, uv);
    break;
  }

  case 1: {
    HitMirror(payload, uv);
    break;
  }

  case 2: {
    HitFloor(payload, uv);
    break;
  }

  default: {
    payload.color = float3(1, 0, 1);
    break;
  }
  }
}

void HitCube(inout Payload payload, float2 uv) {
  uint const tri = PrimitiveIndex() / 2;
  float3 const normal = (tri.xxx % 3 == uint3(0, 1, 2)) * (tri < 3 ? -1 : 1);
  float3 const world_normal = normalize(mul(normal, (float3x3)ObjectToWorld4x3()));

  float3 color = abs(normal) / 3 + 0.5;
  if (uv.x < 0.03 || uv.y < 0.03) {
    color = 0.25.rrr;
  }

  color *= saturate(dot(world_normal, normalize(light))) + 0.33;
  payload.color = color;
  // payload.color = float3(uv, 0);
}

void HitMirror(inout Payload payload, float2 const uv) {
  if (!payload.allow_reflection) {
    return;
  }

  float3 const pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
  float3 const normal = normalize(mul(float3(0, 1, 0), (float3x3)ObjectToWorld4x3()));
  float3 const reflected = reflect(normalize(WorldRayDirection()), normal);

  RayDesc mirror_ray;
  mirror_ray.Origin = pos;
  mirror_ray.Direction = reflected;
  mirror_ray.TMin = 0.001;
  mirror_ray.TMax = 1000;

  payload.allow_reflection = false;
  TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, mirror_ray, payload);
}

void HitFloor(inout Payload payload, float2 const uv) {
  float3 const pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

  bool2 const pattern = frac(pos.xz) > 0.5;
  payload.color = (pattern.x ^ pattern.y ? 0.6 : 0.4).rrr;

  RayDesc shadow_ray;
  shadow_ray.Origin = pos;
  shadow_ray.Direction = light - pos;
  shadow_ray.TMin = 0.001;
  shadow_ray.TMax = 1;

  Payload shadow;
  shadow.allow_reflection = false;
  shadow.missed = false;

  TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, shadow_ray, shadow);

  if (!shadow.missed) {
    payload.color /= 2;
  }
}
