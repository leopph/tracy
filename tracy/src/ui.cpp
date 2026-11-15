#include "ui.hpp"

#include <cstdint>

#include <imgui.h>


namespace tracy {
auto DrawSettingsUi(RenderSettings& settings, RenderStats const& stats) -> UiDrawResult {
  auto changed = false;

  ImGui::SetNextWindowSize({360, 0}, ImGuiCond_FirstUseEver);

  ImGui::Begin("Settings");

  // Stats. One sample per pixel is accumulated per frame, so the frame rate is also samples per second.
  auto const& io = ImGui::GetIO();
  ImGui::Text("Samples: %u", stats.sample_count);
  ImGui::Text("Time: %.1f s", stats.accumulation_seconds);
  ImGui::Text("Frame: %.2f ms (%.0f spp/s)", 1000.0f / io.Framerate, io.Framerate);
  ImGui::Separator();

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
    changed |= ImGui::Checkbox("Russian roulette", &settings.rr_enabled);
    ImGui::BeginDisabled(!settings.rr_enabled);
    changed |= ImGui::SliderScalar("RR start bounce", ImGuiDataType_U32, &settings.rr_start_bounce, &min_bounces,
      &max_bounces, nullptr, ImGuiSliderFlags_AlwaysClamp);
    ImGui::EndDisabled();
  }

  ImGui::End();

  return UiDrawResult{.reset_accumulation = changed};
}


auto DrawMaterialsUi(std::span<MaterialSettings> const materials) -> bool {
  auto changed = false;

  // New windows all open at the same spot, so put this one on the right instead of on top of Settings
  ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x - 380.0f, 20.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({360, 0}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Materials");

  for (auto i = 0uz; i < materials.size(); i++) {
    auto& mtl = materials[i];
    ImGui::PushID(static_cast<int>(i));

    if (ImGui::CollapsingHeader(mtl.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      changed |= ImGui::ColorEdit3("Base color", mtl.base_color.data(), ImGuiColorEditFlags_Float);
      changed |= ImGui::SliderFloat("Metallic", &mtl.metallic, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
      changed |= ImGui::SliderFloat("Roughness", &mtl.roughness, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
      changed |= ImGui::ColorEdit3("Emission color", mtl.emission_color.data(), ImGuiColorEditFlags_Float);
      changed |= ImGui::DragFloat("Emission strength", &mtl.emission_strength, 0.05f, 0.0f, 1000.0f, "%.2f",
        ImGuiSliderFlags_AlwaysClamp);
      changed |= ImGui::Checkbox("Reflective (temporary)", &mtl.is_reflective);
    }

    ImGui::PopID();
  }

  ImGui::End();

  return changed;
}
}
