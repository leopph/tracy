// Based on Dear ImGui's DX12 renderer backend
// Adapted to tracy using wand's GraphicsDevice

#include "imgui_renderer.hpp"

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>

#include <DirectXMath.h>

#include "shaders/shader_interop.h"

#ifndef NDEBUG
#include "shaders/generated/Debug/imgui_ps.h"
#include "shaders/generated/Debug/imgui_vs.h"
#else
#include "shaders/generated/Release/imgui_ps.h"
#include "shaders/generated/Release/imgui_vs.h"
#endif


ImGuiRenderer::ImGuiRenderer(wand::GraphicsDevice& device, wand::SwapChain const& swap_chain) :
  device_{&device},
  swap_chain_{&swap_chain} {
  auto& io = ImGui::GetIO();
  io.BackendRendererName = "Tracy ImGui Renderer";
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

  pso_ = device_->CreatePipelineState(wand::PipelineDesc{
    .vs = CD3DX12_SHADER_BYTECODE{g_imgui_vs_bin, ARRAYSIZE(g_imgui_vs_bin)},
    .ps = CD3DX12_SHADER_BYTECODE{g_imgui_ps_bin, ARRAYSIZE(g_imgui_ps_bin)},
    .blend_state = CD3DX12_BLEND_DESC{
      D3D12_BLEND_DESC{
        .AlphaToCoverageEnable = FALSE,
        .IndependentBlendEnable = false,
        .RenderTarget = {
          {
            .BlendEnable = TRUE,
            .LogicOpEnable = FALSE,
            .SrcBlend = D3D12_BLEND_SRC_ALPHA,
            .DestBlend = D3D12_BLEND_INV_SRC_ALPHA,
            .BlendOp = D3D12_BLEND_OP_ADD,
            .SrcBlendAlpha = D3D12_BLEND_ONE,
            .DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA,
            .BlendOpAlpha = D3D12_BLEND_OP_ADD,
            .LogicOp = D3D12_LOGIC_OP_NOOP,
            .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL
          }
        }
      }
    },
    .depth_stencil_state = CD3DX12_DEPTH_STENCIL_DESC1{
      FALSE, {}, {}, FALSE, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    },
    .rasterizer_state = CD3DX12_RASTERIZER_DESC{
      D3D12_FILL_MODE_SOLID, D3D12_CULL_MODE_NONE, FALSE, D3D12_DEFAULT_DEPTH_BIAS, D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
      D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS, TRUE, FALSE, FALSE, 0, D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
    },
    .rt_formats = CD3DX12_RT_FORMAT_ARRAY{D3D12_RT_FORMAT_ARRAY{{DXGI_FORMAT_R8G8B8A8_UNORM}, 1}},
  }, sizeof(ImGuiDrawParams) / 4);

  samp_ = device_->CreateSampler(D3D12_SAMPLER_DESC{
    D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    D3D12_TEXTURE_ADDRESS_MODE_WRAP, 0, 0, D3D12_COMPARISON_FUNC_NEVER, {0.0f, 0.0f, 0.0f, 0.0f}, 0,
    std::numeric_limits<float>::max(),
  });

  UpdateFonts();
}


auto ImGuiRenderer::UpdateFonts() -> void {
  auto& fonts = *ImGui::GetIO().Fonts;
  fonts.Build();
  fonts.SetTexID(kFontTexId);
  fonts_dirty_ = true;
}


auto ImGuiRenderer::Record(wand::CommandList& cmd, std::uint32_t const frame_idx) -> void {
  auto const& imgui_draw_data = *ImGui::GetDrawData();
  auto& draw_data = draw_data_[frame_idx];

  draw_data.valid = imgui_draw_data.Valid;
  draw_data.cmd_lists_count = imgui_draw_data.CmdListsCount;
  draw_data.total_idx_count = imgui_draw_data.TotalIdxCount;
  draw_data.total_vtx_count = imgui_draw_data.TotalVtxCount;
  draw_data.display_pos = imgui_draw_data.DisplayPos;
  draw_data.display_size = imgui_draw_data.DisplaySize;
  draw_data.framebuffer_scale = imgui_draw_data.FramebufferScale;

  draw_data.cmd_lists.clear();

  // We never shrink the CmdLists vector to keep the vectors inside the CmdLists alive and prevent unnecessary
  // allocations. TODO we could just flatten the whole thing into arrays in DrawData and store only indices in DrawLists
  if (draw_data.cmd_lists_count > draw_data.cmd_lists.size()) {
    draw_data.cmd_lists.resize(draw_data.cmd_lists_count);
  }

  for (auto i = 0; i < imgui_draw_data.CmdListsCount; ++i) {
    auto const& draw_list{*imgui_draw_data.CmdLists[i]};
    auto& cmd_list{draw_data.cmd_lists[i]};

    cmd_list.cmd_buffer.assign(draw_list.CmdBuffer.begin(), draw_list.CmdBuffer.end());
    cmd_list.idx_buffer.assign(draw_list.IdxBuffer.begin(), draw_list.IdxBuffer.end());
    cmd_list.vtx_buffer.assign(draw_list.VtxBuffer.begin(), draw_list.VtxBuffer.end());
    cmd_list.flags = draw_list.Flags;
  }

  if (fonts_dirty_) {
    unsigned char* fonts_tex_pixel_data;
    int fonts_tex_width;
    int fonts_tex_height;
    ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&fonts_tex_pixel_data, &fonts_tex_width, &fonts_tex_height);

    font_tex_ = device_->CreateTexture(wand::TextureDesc{
      .dimension = wand::TextureDimension::k2D, .width = static_cast<UINT>(fonts_tex_width),
      .height = static_cast<UINT>(fonts_tex_height), .depth_or_array_size = 1, .mip_levels = 1,
      .format = DXGI_FORMAT_R8G8B8A8_UNORM, .sample_count = 1,
      .component_mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .depth_stencil = false, .render_target = false,
      .shader_resource = true, .unordered_access = false
    }, wand::CpuAccess::kNone, nullptr);
    font_tex_->SetDebugName(L"UI Font Texture");

    auto footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT{};
    auto row_counts = UINT{};
    auto row_sizes = UINT64{};
    auto total_size = UINT64{};
    device_->GetCopyableFootprints(font_tex_->GetDesc(), 0, 1, 0, &footprint, &row_counts, &row_sizes, &total_size);

    auto& upload_buf = font_tex_upload_bufs_[frame_idx];

    if (!upload_buf || upload_buf->GetDesc().size < total_size) {
      upload_buf = device_->CreateBuffer(wand::BufferDesc{
        .size = total_size,
        .usage = wand::BufferUsage::kCopySource
      }, wand::CpuAccess::kWrite);
    }

    auto const src_row_pitch = static_cast<std::size_t>(fonts_tex_width) * 4;
    auto const upload_ptr = static_cast<std::byte*>(upload_buf->Map());

    for (auto row = UINT{0}; row < row_counts; ++row) {
      std::memcpy(upload_ptr + footprint.Offset + static_cast<std::size_t>(row) * footprint.Footprint.RowPitch,
        fonts_tex_pixel_data + row * src_row_pitch, row_sizes);
    }

    upload_buf->Unmap();

    cmd.CopyTextureRegion(*font_tex_, 0, 0, 0, 0, *upload_buf, footprint);

    fonts_dirty_ = false;
  }

  draw_data.font_tex = font_tex_;

  // Avoid rendering when minimized or when there is nothing to draw. The latter happens e.g. on the first frame,
  // because ImGui hides new windows for a frame while it measures them; creating 0-sized buffers would be an error.
  if (draw_data.display_size.x <= 0.0f || draw_data.display_size.y <= 0.0f || draw_data.total_vtx_count <= 0) {
    return;
  }

  auto& vb = vtx_buffers_[frame_idx];
  auto& vb_ptr = vb_ptrs_[frame_idx];

  if (auto const vtx_data_byte_size = draw_data.total_vtx_count * sizeof(ImDrawVert);
    !vb || vb->GetDesc().region.size < vtx_data_byte_size) {
    vb = CreateBufferWithView(*device_, wand::BufferDesc{
      .size = vtx_data_byte_size,
      .usage = wand::BufferUsage::kShaderResource
    }, wand::BufferViewDesc{
      .region = {.offset = 0, .size = vtx_data_byte_size, .stride = static_cast<UINT>(sizeof(ImDrawVert))},
      .usage = wand::BufferViewUsage::kShaderResource
    }, wand::CpuAccess::kWrite);
    vb->GetBuffer()->SetDebugName(L"UI Vertex Buffer");
    vb_ptr = vb->GetBuffer()->Map();
  }

  auto& ib = idx_buffers_[frame_idx];
  auto& ib_ptr = ib_ptrs_[frame_idx];

  if (auto const idx_data_byte_size = draw_data.total_idx_count * sizeof(ImDrawIdx);
    !ib || ib->GetDesc().size < idx_data_byte_size) {
    ib = device_->CreateBuffer(wand::BufferDesc{
      .size = idx_data_byte_size,
      .usage = wand::BufferUsage::kIndexBuffer
    }, wand::CpuAccess::kWrite);
    ib->SetDebugName(L"UI Index Buffer");
    ib_ptr = ib->Map();
  }

  auto constexpr idx_format = [] {
    if constexpr (sizeof(ImDrawIdx) == 2) {
      return DXGI_FORMAT_R16_UINT;
    } else {
      return DXGI_FORMAT_R32_UINT;
    }
  }();

  auto vtx_dst = static_cast<ImDrawVert*>(vb_ptr);
  auto idx_dst = static_cast<ImDrawIdx*>(ib_ptr);

  for (auto i = 0; i < draw_data.cmd_lists_count; ++i) {
    auto const imgui_cmd{draw_data.cmd_lists[i]};
    std::memcpy(vtx_dst, imgui_cmd.vtx_buffer.data(), imgui_cmd.vtx_buffer.size() * sizeof(ImDrawVert));
    std::memcpy(idx_dst, imgui_cmd.idx_buffer.data(), imgui_cmd.idx_buffer.size() * sizeof(ImDrawIdx));
    vtx_dst += imgui_cmd.vtx_buffer.size();
    idx_dst += imgui_cmd.idx_buffer.size();
  }

  auto const& rt = swap_chain_->GetCurrentTexture();

  auto proj_mtx = DirectX::XMFLOAT4X4{};
  DirectX::XMStoreFloat4x4(&proj_mtx, DirectX::XMMatrixOrthographicOffCenterLH(draw_data.display_pos.x,
    // ImGui's y axis points down, so bottom is pos.y + size.y and top is pos.y
    draw_data.display_pos.x + draw_data.display_size.x, draw_data.display_pos.y + draw_data.display_size.y,
    draw_data.display_pos.y, -1, 1));

  cmd.SetPipelineState(*pso_);

  cmd.SetBlendFactor(std::array{0.f, 0.f, 0.f, 0.f});
  cmd.SetIndexBuffer(*ib, idx_format);
  cmd.SetPipelineParameters(PIPELINE_PARAM_INDEX(ImGuiDrawParams, proj_mtx),
    std::span{std::start_lifetime_as_array<UINT const>(&proj_mtx.m, 16), 16});
  cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ImGuiDrawParams, samp_idx), samp_.Get());
  cmd.SetShaderResource(PIPELINE_PARAM_INDEX(ImGuiDrawParams, vb_idx), *vb);
  cmd.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  cmd.SetRenderTargets(std::span{std::array{(&rt)}.data(), 1}, nullptr);
  cmd.SetViewports(std::array<D3D12_VIEWPORT, 1>{
    CD3DX12_VIEWPORT{0.0f, 0.0f, draw_data.display_size.x, draw_data.display_size.y}
  });

  // Render command lists
  // (Because we merged all buffers into a single one, we maintain our own offset into them)
  auto global_vtx_offset = 0;
  auto global_idx_offset = 0;

  auto const& clip_off = draw_data.display_pos;
  auto const imgui_cmds = std::span{draw_data.cmd_lists.data(), static_cast<std::uint32_t>(draw_data.cmd_lists_count)};

  for (auto const& imgui_cmd : imgui_cmds) {
    for (auto const& draw_cmd : imgui_cmd.cmd_buffer) {
      if (draw_cmd.UserCallback != nullptr) {
        // TODO honor user callback
        // User callback, registered via ImDrawList::AddCallback()
        // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
        /*if (draw_cmd.UserCallback == ImDrawCallback_ResetRenderState) {
          // TODO reset render state - ImGui_ImplDX12_SetupRenderState(draw_data, ctx, fr);
        } else {
          draw_cmd.UserCallback(imgui_cmd, &draw_cmd);
        }*/
      } else {
        // Project scissor/clipping rectangles into framebuffer space
        auto const clip_min = std::array{draw_cmd.ClipRect.x - clip_off.x, draw_cmd.ClipRect.y - clip_off.y};
        auto const clip_max = std::array{draw_cmd.ClipRect.z - clip_off.x, draw_cmd.ClipRect.w - clip_off.y};

        if (clip_max[0] <= clip_min[0] || clip_max[1] <= clip_min[1]) {
          continue;
        }

        cmd.SetScissorRects(std::array{
          D3D12_RECT{
            .left = static_cast<LONG>(clip_min[0]),
            .top = static_cast<LONG>(clip_min[1]),
            .right = static_cast<LONG>(clip_max[0]),
            .bottom = static_cast<LONG>(clip_max[1])
          }
        });

        // Only the font atlas is supported for now
        cmd.SetShaderResource(PIPELINE_PARAM_INDEX(ImGuiDrawParams, tex_idx), *draw_data.font_tex);

        cmd.DrawIndexedInstanced(draw_cmd.ElemCount, 1, draw_cmd.IdxOffset + global_idx_offset,
          draw_cmd.VtxOffset + global_vtx_offset, 0);
      }
    }
    global_idx_offset += static_cast<int>(imgui_cmd.idx_buffer.size());
    global_vtx_offset += static_cast<int>(imgui_cmd.vtx_buffer.size());
  }
}


auto const ImGuiRenderer::kFontTexId{std::numeric_limits<ImTextureID>::max()};
