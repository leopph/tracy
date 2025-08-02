//#include "common.hlsli"

// ReSharper disable CppEnforceCVQualifiersPlacement

struct PsIn {
  float4 pos_os : SV_Position;
  float2 uv : TEXCOORD;
};

float2 UvToNdc(const float2 uv) {
  return uv * float2(2, -2) + float2(-1, 1);
}

PsIn VsMain(const uint vertex_id : SV_VertexID) {
  PsIn ret;
  ret.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
  ret.pos_os = float4(UvToNdc(ret.uv), 1, 1);
  return ret;
}

float4 PsMain(const PsIn ps_in) : SV_Target {
  return float4(1, 0, 1, 1);
}
