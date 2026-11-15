#pragma once

#include <array>
#include <cstdint>


namespace tracy {
struct SunSettings {
  std::array<float, 3> tint;
  float irradiance;
  float radius;
  float elevation;
  float azimuth;
};


struct SkySettings {
  std::array<float, 3> zenith_tint;
  std::array<float, 3> horizon_tint;
  float radiance;
};


struct RenderSettings {
  SunSettings sun;
  SkySettings sky;
  std::uint32_t max_bounces;
};
}
