#pragma once

#include <cstdint>

#include "render_settings.hpp"


namespace tracy {
struct RenderStats {
  std::uint32_t sample_count;
  float accumulation_seconds;
};


struct UiDrawResult {
  bool reset_accumulation;
};


auto DrawSettingsUi(RenderSettings& settings, RenderStats const& stats) -> UiDrawResult;
}
