#include "ui.hpp"

#include <cstdint>

#include <imgui.h>


namespace tracy {
auto DrawSettingsUi(RenderSettings& settings) -> UiDrawResult {
  auto changed = false;

  ImGui::SetNextWindowSize({360, 0}, ImGuiCond_FirstUseEver);

  ImGui::Begin("Settings");

  if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::PushID("Sun");
    auto& sun = settings.sun;
    changed |= ImGui::ColorEdit3("Tint", sun.tint.data(), ImGuiColorEditFlags_Float);
    changed |= ImGui::DragFloat("Irradiance", &sun.irradiance, 0.1f, 0.0f, 1000.0f, "%.2f",
      ImGuiSliderFlags_AlwaysClamp);
    // SliderAngle stores radians but displays and edits degrees
    changed |= ImGui::SliderAngle("Angular radius", &sun.radius, 0.1f, 20.0f, "%.2f deg",
      ImGuiSliderFlags_AlwaysClamp);
    changed |= ImGui::SliderAngle("Elevation", &sun.elevation, -10.0f, 90.0f, "%.1f deg",
      ImGuiSliderFlags_AlwaysClamp);
    changed |= ImGui::SliderAngle("Azimuth", &sun.azimuth, -180.0f, 180.0f, "%.1f deg",
      ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopID();
  }

  if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::PushID("Sky");
    auto& sky = settings.sky;
    changed |= ImGui::ColorEdit3("Zenith tint", sky.zenith_tint.data(), ImGuiColorEditFlags_Float);
    changed |= ImGui::ColorEdit3("Horizon tint", sky.horizon_tint.data(), ImGuiColorEditFlags_Float);
    changed |= ImGui::DragFloat("Radiance", &sky.radiance, 0.01f, 0.0f, 100.0f, "%.2f",
      ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopID();
  }

  if (ImGui::CollapsingHeader("Path tracing", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto constexpr min_bounces = std::uint32_t{0};
    auto constexpr max_bounces = std::uint32_t{32};
    changed |= ImGui::SliderScalar("Max bounces", ImGuiDataType_U32, &settings.max_bounces, &min_bounces,
      &max_bounces, nullptr, ImGuiSliderFlags_AlwaysClamp);
  }

  ImGui::End();

  return UiDrawResult{.reset_accumulation = changed};
}
}
