#pragma once

#include "render_settings.hpp"


namespace tracy {
struct UiDrawResult {
  bool reset_accumulation;
};


auto DrawSettingsUi(RenderSettings& settings) -> UiDrawResult;
}
