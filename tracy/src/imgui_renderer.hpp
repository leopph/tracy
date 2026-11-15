#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <imgui.h>
#include <wand/wand.hpp>


class ImGuiRenderer {
public:
  ImGuiRenderer(wand::GraphicsDevice& device, wand::SwapChain const& swap_chain);
  ImGuiRenderer(ImGuiRenderer const& other) = delete;
  ImGuiRenderer(ImGuiRenderer&& other) noexcept = delete;

  ~ImGuiRenderer() = default;

  auto operator=(ImGuiRenderer const& other) -> ImGuiRenderer& = delete;
  auto operator=(ImGuiRenderer&& other) noexcept -> ImGuiRenderer& = delete;

  auto UpdateFonts() -> void;
  auto Record(wand::CommandList& cmd, std::uint32_t frame_idx) -> void;

private:
  struct CmdList {
    std::vector<ImDrawCmd> cmd_buffer;
    std::vector<ImDrawIdx> idx_buffer;
    std::vector<ImDrawVert> vtx_buffer;
    ImDrawListFlags flags;
  };


  struct DrawData {
    bool valid;
    int cmd_lists_count;
    int total_idx_count;
    int total_vtx_count;
    std::vector<CmdList> cmd_lists;
    ImVec2 display_pos;
    ImVec2 display_size;
    ImVec2 framebuffer_scale;
    wand::SharedDeviceHandle<wand::Texture> font_tex;
  };


  wand::GraphicsDevice* device_;
  wand::SwapChain const* swap_chain_;

  wand::SharedDeviceHandle<wand::PipelineState> pso_;
  wand::UniqueSamplerHandle samp_;
  wand::SharedDeviceHandle<wand::Texture> font_tex_;

  std::array<wand::SharedDeviceHandle<wand::Buffer>, 2> font_tex_upload_bufs_;

  std::array<wand::SharedDeviceHandle<wand::BufferView>, 2> vtx_buffers_;
  std::array<wand::SharedDeviceHandle<wand::Buffer>, 2> idx_buffers_;

  std::array<void*, 2> vb_ptrs_{};
  std::array<void*, 2> ib_ptrs_{};

  std::array<DrawData, 2> draw_data_{};

  bool fonts_dirty_{false};

  ImTextureID const static kFontTexId;
};
