#include <bit>
#include <cstdint>

#include <DirectXMath.h>
#include <wand/wand.hpp>

#include "shader_interop.h"

#ifndef NDEBUG
#include "generated/Debug/resolve.h"
#include "generated/Debug/shader_lib.h"
#include "generated/Debug/tracer.h"
#else
#include "generated/Release/resolve.h"
#include "generated/Release/shader_lib.h"
#include "generated/Release/tracer.h"
#endif

import std;


extern "C" {
__declspec(dllexport) extern UINT const D3D12SDKVersion = D3D12_SDK_VERSION;
__declspec(dllexport) extern char const* D3D12SDKPath = ".\\D3D12\\";
}


template<unsigned N>
using Vector = std::array<float, N>;


constexpr UINT kMaxGpuQueuedFrames{1};
constexpr auto kNumFramesInFlight{kMaxGpuQueuedFrames + 1};
constexpr auto kUseRayQuery = true;
constexpr auto kAnimate = false;
constexpr auto kExposure = 1.0f;


struct RenderingContext {
  std::unique_ptr<wand::GraphicsDevice> device;
  wand::SharedDeviceHandle<wand::SwapChain> swap_chain;
  std::array<wand::SharedDeviceHandle<wand::CommandList>, kNumFramesInFlight> cmd_lists;
  wand::SharedDeviceHandle<wand::Texture> accum_tex;
  wand::SharedDeviceHandle<wand::Texture> resolve_tex;
  std::uint32_t sample_count{0};
  bool reset_requested{true};
};


constexpr UINT kNumInstances{3};


template<std::unsigned_integral T>
[[nodiscard]] constexpr auto SatSub(T const lhs, T const rhs) -> T {
  T ret{lhs - rhs};
  ret &= -(ret <= lhs);
  return ret;
}


auto ThrowIfFailed(HRESULT const hr) -> void {
  if (FAILED(hr)) {
    throw std::runtime_error{"HRESULT returned failure code."};
  }
}


struct HwndDeleter {
  auto operator()(HWND const hwnd) const -> void {
    DestroyWindow(hwnd);
  }
};


auto Resize(HWND const hwnd) -> void {
  auto* const ctx = reinterpret_cast<RenderingContext*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

  if (!ctx) [[unlikely]] {
    return;
  }

  if (!ctx->swap_chain) [[unlikely]] {
    return;
  }

  RECT rect;
  GetClientRect(hwnd, &rect);

  auto const width = std::max<UINT>(rect.right - rect.left, 1);
  auto const height = std::max<UINT>(rect.bottom - rect.top, 1);

  ctx->device->WaitIdle();
  ctx->device->ResizeSwapChain(*ctx->swap_chain, width, height);

  auto constexpr accum_tex_format = DXGI_FORMAT_R32G32B32A32_FLOAT;

  ctx->accum_tex = ctx->device->CreateTexture(
    wand::TextureDesc{
      .dimension = wand::TextureDimension::k2D, .width = width, .height = height, .depth_or_array_size = 1,
      .mip_levels = 1, .format = accum_tex_format, .sample_count = 1,
      .component_mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .depth_stencil = false,
      .render_target = true, .shader_resource = true, .unordered_access = true
    }, wand::CpuAccess::kNone, std::array{
      D3D12_CLEAR_VALUE{
        .Format = accum_tex_format,
        .Color = {0.0f, 0.0f, 0.0f, 1.0f}
      }
    }.data());

  ctx->resolve_tex = ctx->device->CreateTexture(
    wand::TextureDesc{
      .dimension = wand::TextureDimension::k2D, .width = width, .height = height, .depth_or_array_size = 1,
      .mip_levels = 1, .format = DXGI_FORMAT_R8G8B8A8_UNORM, .sample_count = 1,
      .component_mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, .depth_stencil = false,
      .render_target = false, .shader_resource = false, .unordered_access = true
    }, wand::CpuAccess::kNone, nullptr);

  ctx->reset_requested = true;
}


auto UpdateTransforms(D3D12_RAYTRACING_INSTANCE_DESC* const instance_data) -> void {
  using namespace DirectX;
  auto const set = [instance_data](int const idx, XMMATRIX const& mx) {
    auto* const ptr = reinterpret_cast<XMFLOAT3X4*>(&instance_data[idx].Transform);
    XMStoreFloat3x4(ptr, mx);
  };

  auto const time = kAnimate ? static_cast<float>(GetTickCount64()) / 1000.0f : 0.0f;

  auto cube = XMMatrixRotationRollPitchYaw(time / 2, time / 3, time / 5);
  cube *= XMMatrixTranslation(-1.5, 2, 2);
  set(0, cube);

  auto mirror = XMMatrixRotationX(-1.8f);
  mirror *= XMMatrixRotationY(XMScalarSinEst(time) / 8 + 1);
  mirror *= XMMatrixTranslation(2, 2, 2);
  set(1, mirror);

  auto floor = XMMatrixScaling(5, 5, 5);
  floor *= XMMatrixTranslation(0, 0, 2);
  set(2, floor);
}


auto WINAPI WndProc(HWND const hwnd, UINT const msg, WPARAM const wparam, LPARAM const lparam) -> LRESULT {
  switch (msg) {
    case WM_CLOSE: {
      PostQuitMessage(0);
      return 0;
    }
    case WM_SIZE: {
      Resize(hwnd);
      return 0;
    }
    default: {
      return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
  }
}


auto main() -> int {
  // Alternatively, DPI_AWARENESS_CONTEXT_UNAWARE
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  // Create window

  WNDCLASSW const wcw = {
    .lpfnWndProc = &WndProc, .hCursor = LoadCursor(nullptr, IDC_ARROW), .lpszClassName = L"TracyWindowClass"
  };

  RegisterClassW(&wcw);

  auto const hwnd = std::unique_ptr<std::remove_pointer_t<HWND>, HwndDeleter>{
    CreateWindowExW(0, wcw.lpszClassName, L"tracy", WS_VISIBLE | WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,
      CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, nullptr, nullptr)
  };

  // Create rendering context

  RenderingContext ctx;
  SetWindowLongPtrW(hwnd.get(), GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&ctx));

#ifndef NDEBUG
  constexpr auto enable_debug = true;
#else
  constexpr auto enable_debug = false;
#endif

  ctx.device = std::make_unique<wand::GraphicsDevice>(enable_debug, false);

  ctx.swap_chain = ctx.device->CreateSwapChain(
    wand::SwapChainDesc{
      .width = 0, .height = 0, .buffer_count = 2, .format = DXGI_FORMAT_R8G8B8A8_UNORM, .usage = 0,
      .scaling = DXGI_SCALING_NONE
    }, hwnd.get());

  for (auto i = 0u; i < kNumFramesInFlight; i++) {
    ctx.cmd_lists[i] = ctx.device->CreateCommandList();
  }

  // Force a resize to create render target
  Resize(hwnd.get());

  // Init meshes

  auto const create_buffer_for = [&ctx](auto const& data) {
    auto const view = wand::CreateBufferWithView(*ctx.device,
      wand::BufferDesc{
        .size = sizeof(data),
        .usage = wand::BufferUsage::kShaderResource
      }, wand::BufferViewDesc{
        .region = {.offset = 0, .size = wand::kRemainingSize, .stride = sizeof(data[0])},
        .usage = wand::BufferViewUsage::kShaderResource
      }, wand::CpuAccess::kWrite);

    auto* const data_ptr = view->GetBuffer()->Map();
    std::memcpy(data_ptr, &data, sizeof(data));
    view->GetBuffer()->Unmap();

    return view;
  };

  auto constexpr quad_vertices = std::array{
    Vector<3>{-1, 0, -1}, Vector<3>{-1, 0, 1}, Vector<3>{1, 0, 1},
    Vector<3>{-1, 0, -1}, Vector<3>{1, 0, -1}, Vector<3>{1, 0, 1}
  };

  auto constexpr quad_indices = std::array{
    0u, 1u, 2u, 5u, 4u, 3u
  };

  auto constexpr cube_vertices = std::array{
    Vector<3>{-1, -1, -1}, Vector<3>{1, -1, -1}, Vector<3>{-1, 1, -1}, Vector<3>{1, 1, -1},
    Vector<3>{-1, -1, 1}, Vector<3>{1, -1, 1}, Vector<3>{-1, 1, 1}, Vector<3>{1, 1, 1}
  };

  auto constexpr cube_indices = std::array{
    4u, 6u, 0u, 2u, 0u, 6u, 0u, 1u, 4u, 5u, 4u, 1u,
    0u, 2u, 1u, 3u, 1u, 2u, 1u, 3u, 5u, 7u, 5u, 3u,
    2u, 6u, 3u, 7u, 3u, 6u, 4u, 5u, 6u, 7u, 6u, 5u
  };

  auto const quad_vb = create_buffer_for(quad_vertices);
  auto const quad_ib = create_buffer_for(quad_indices);
  auto const cube_vb = create_buffer_for(cube_vertices);
  auto const cube_ib = create_buffer_for(cube_indices);

  // AS utilities

  auto const make_as = [&ctx](wand::BuildAccelerationStructureInputs const& inputs,
                              wand::AccelerationStructureType const type, UINT64* update_scratch_size = nullptr) {
    auto const prebuild_info = ctx.device->GetAccelerationStructurePrebuildInfo(inputs);

    if (update_scratch_size) {
      *update_scratch_size = prebuild_info.UpdateScratchDataSizeInBytes;
    }

    auto const scratch = ctx.device->CreateBuffer(
      wand::BufferDesc{
        .size = prebuild_info.ScratchDataSizeInBytes, .usage = wand::BufferUsage::kAccelerationScratch
      }, wand::CpuAccess::kNone);

    auto const as = wand::CreateBufferWithAs(*ctx.device, wand::BufferDesc{
      .size = prebuild_info.ResultDataMaxSizeInBytes, .usage = wand::BufferUsage::kAccelerationStructure
    }, wand::AccelerationStructureDesc{
      .region = {},
      .type = type
    });

    wand::BuildAccelerationStructureDesc const build_desc = {
      .dst_as = *as,
      .inputs = inputs,
      .scratch_buffer = wand::BufferRange{*scratch}
    };

    ctx.cmd_lists[0]->Begin(nullptr);
    ctx.cmd_lists[0]->BuildAccelerationStructure(build_desc);
    ctx.cmd_lists[0]->End();

    ctx.device->ExecuteCommandLists(std::span{ctx.cmd_lists[0].get(), 1});
    ctx.device->WaitIdle();

    return as;
  };

  auto const make_blas = [make_as]
  (wand::Buffer const& vertex_buffer, wand::Buffer const* const index_buffer = nullptr) {
    auto const geometry_desc = wand::TriangleGeometry{
      .vertices = wand::BufferRange{vertex_buffer, wand::BufferRegion{.stride = 12}},
      .vertex_format = DXGI_FORMAT_R32G32B32_FLOAT,
      .indices = index_buffer ? std::make_optional(wand::BufferRange{*index_buffer}) : std::nullopt,
      .index_format = index_buffer ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_UNKNOWN,
      .flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
    };

    auto const inputs = wand::BuildBottomLevelAsInputs{
      .geometries = std::span{&geometry_desc, 1},
      .flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
    };

    return make_as(inputs, wand::AccelerationStructureType::kBottomLevel);
  };

  // BLAS for meshes

  auto const quad_blas = make_blas(*quad_vb->GetBuffer(), quad_ib.get()->GetBuffer().get());
  auto const cube_blas = make_blas(*cube_vb->GetBuffer(), cube_ib.get()->GetBuffer().get());

  // Init scene

  wand::BufferDesc constexpr rt_instance_buf_desc{
    .size = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * kNumInstances, .usage = wand::BufferUsage::kNone
  };

  std::array<wand::SharedDeviceHandle<wand::Buffer>, kNumFramesInFlight> instance_bufs;
  std::array<D3D12_RAYTRACING_INSTANCE_DESC*, kNumFramesInFlight> instance_data;

  for (auto i = 0u; i < kNumFramesInFlight; i++) {
    instance_bufs[i] = ctx.device->CreateBuffer(rt_instance_buf_desc, wand::CpuAccess::kWrite);

    instance_data[i] = reinterpret_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(instance_bufs[i]->Map());

    for (UINT j = 0; j < kNumInstances; j++) {
      instance_data[i][j] = {
        .InstanceID = j,
        .InstanceMask = 1,
        .AccelerationStructure = (j ? quad_blas : cube_blas)->GetGpuAddress()
      };
    }
  }

  UpdateTransforms(instance_data[0]);

  // Create gpu materials

  auto const mtl_table_buf = wand::CreateBufferWithView(*ctx.device, wand::BufferDesc{
    .size = kNumInstances * sizeof(GpuMaterial),
    .usage = wand::BufferUsage::kShaderResource
  }, wand::BufferViewDesc{
    .region = {.offset = 0, .size = wand::kRemainingSize, .stride = sizeof(GpuMaterial)},
    .usage = wand::BufferViewUsage::kShaderResource
  }, wand::CpuAccess::kWrite);

  {
    auto constexpr materials = std::array{
      // Cube
      GpuMaterial{
        .base_color = {1.0f, 0.0f, 0.0f}, .emission = {}, .is_reflective = false
      },
      // Mirror
      GpuMaterial{
        .base_color = {1.0f, 1.0f, 1.0f}, .emission = {}, .is_reflective = true
      },
      // Floor
      GpuMaterial{
        .base_color = {0.5f, 0.5f, 0.5f}, .emission = {}, .is_reflective = false
      },
    };
    auto const mapped = mtl_table_buf->GetBuffer()->Map();
    std::memcpy(mapped, materials.data(), sizeof(materials));
    mtl_table_buf->GetBuffer()->Unmap();
  }

  // Create gpu geometries

  auto const geom_table_buf = wand::CreateBufferWithView(*ctx.device, wand::BufferDesc{
    .size = 2 * sizeof(GpuGeometry),
    .usage = wand::BufferUsage::kShaderResource
  }, wand::BufferViewDesc{
    .region = {.offset = 0, .size = wand::kRemainingSize, .stride = sizeof(GpuGeometry)},
    .usage = wand::BufferViewUsage::kShaderResource
  }, wand::CpuAccess::kWrite);

  {
    auto const geometries = std::array{
      // Cube
      GpuGeometry{
        .vtx_buf_srv = cube_vb->GetShaderResource(), .idx_buf_srv = cube_ib->GetShaderResource()
      },
      // Quad
      GpuGeometry{
        .vtx_buf_srv = quad_vb->GetShaderResource(), .idx_buf_srv = quad_ib->GetShaderResource()
      },
    };
    auto const mapped = geom_table_buf->GetBuffer()->Map();
    std::memcpy(mapped, geometries.data(), sizeof(geometries));
    geom_table_buf->GetBuffer()->Unmap();
  }

  // Create gpu instances

  auto const inst_table_buf = wand::CreateBufferWithView(*ctx.device, wand::BufferDesc{
    .size = kNumInstances * sizeof(GpuInstance),
    .usage = wand::BufferUsage::kShaderResource
  }, wand::BufferViewDesc{
    .region = {.offset = 0, .size = wand::kRemainingSize, .stride = sizeof(GpuInstance)},
    .usage = wand::BufferViewUsage::kShaderResource
  }, wand::CpuAccess::kWrite);

  {
    auto constexpr instances = std::array{
      // Cube
      GpuInstance{
        .geom_idx = 0, .mtl_idx = 0
      },
      // Mirror
      GpuInstance{
        .geom_idx = 1, .mtl_idx = 1
      },
      // Floor
      GpuInstance{
        .geom_idx = 1, .mtl_idx = 2
      }
    };
    auto const mapped = inst_table_buf->GetBuffer()->Map();
    std::memcpy(mapped, instances.data(), sizeof(instances));
    inst_table_buf->GetBuffer()->Unmap();
  }

  // TLAS for scene

  auto const inputs = wand::BuildTopLevelAsInputs{
    .instances = wand::BufferRange{
      *instance_bufs[0], wand::BufferRegion{
        .offset = 0,
        .size = kNumInstances * sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
        .stride = sizeof(D3D12_RAYTRACING_INSTANCE_DESC)
      }
    },
    .flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
  };

  UINT64 update_scratch_size;
  auto const tlas = make_as(inputs, wand::AccelerationStructureType::kTopLevel, &update_scratch_size);

  auto const tlas_update_scratch = ctx.device->CreateBuffer(
    wand::BufferDesc{
      /* WARP bug workaround: use 8 if the required size was reported as less */
      .size = std::max<UINT64>(update_scratch_size, 8ull),
      .usage = wand::BufferUsage::kAccelerationScratch
    }, wand::CpuAccess::kNone
  );

  // Create PSOs

  // Create RT PSO

  wand::RtStateObjectDesc pso_desc;

  auto& lib_desc = pso_desc.AddDxilLibrary();
  CD3DX12_SHADER_BYTECODE const shader_lib_code{g_shader_lib_bin, std::size(g_shader_lib_bin)};
  lib_desc.SetDXILLibrary(&shader_lib_code);

  auto& hit_group_desc = pso_desc.AddHitGroup();
  hit_group_desc.SetHitGroupExport(L"HitGroup");
  hit_group_desc.SetClosestHitShaderImport(L"ClosestHit");

  auto& shader_config_desc = pso_desc.AddShaderConfig();
  shader_config_desc.Config(20, 8); // sizeof(Payload), sizeof(attribs)

  auto& pipeline_config_desc = pso_desc.AddPipelineConfig();
  pipeline_config_desc.Config(3); // cam->mirror->floor->light

  auto const pso = ctx.device->CreateRtStateObject(pso_desc, 2);

  // Create shader table

  constexpr UINT64 kShaderTableSize{3}; // 1 raygen + 1 miss + 1 hitgroup

  auto const shader_table = ctx.device->CreateBuffer(
    wand::BufferDesc{
      .size = kShaderTableSize * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, .usage = wand::BufferUsage::kNone
    }, wand::CpuAccess::kWrite);

  {
    auto shader_table_data = shader_table->Map();

    auto const write_id = [&](wchar_t const* const name) {
      auto const* const id = pso->GetShaderIdentifier(name);
      std::memcpy(shader_table_data, id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
      shader_table_data = static_cast<char*>(shader_table_data) + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    };

    write_id(L"RayGeneration");
    write_id(L"Miss");
    write_id(L"HitGroup");

    shader_table->Unmap();
  }

  // Create CS PSO

  auto const cs_pso = ctx.device->CreatePipelineState(wand::PipelineDesc{
    .cs = CD3DX12_SHADER_BYTECODE{g_tracer_cs_bin, std::size(g_tracer_cs_bin)}
  }, sizeof(ComputeDispatchParams) / 4);

  // Create resolve PSO

  auto const resolve_pso = ctx.device->CreatePipelineState(wand::PipelineDesc{
    .cs = CD3DX12_SHADER_BYTECODE{g_resolve_bin, std::size(g_resolve_bin)}
  }, sizeof(ResolveParams) / 4);

  // Create fence for frame synchronization
  auto const frame_fence = ctx.device->CreateFence(0);

  // Main loop

  UINT frame_count{0};

  for (MSG msg;;) {
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        ctx.device->WaitIdle();
        SetWindowLongPtrW(hwnd.get(), GWLP_USERDATA, 0);
        return 0;
      }

      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }

    auto const frame_idx = frame_count % kNumFramesInFlight;

    auto& cmd = *ctx.cmd_lists[frame_idx];
    cmd.Begin(nullptr);

    // Handle accumulation reset

    if (ctx.reset_requested) {
      cmd.ClearRenderTarget(*ctx.accum_tex, std::array{0.0f, 0.0f, 0.0f, 1.0f}, {});
      ctx.sample_count = 0;
      ctx.reset_requested = false;
    }

    // Update scene transforms

    UpdateTransforms(instance_data[frame_idx]);

    // Update TLAS

    auto const tlas_update_desc = wand::BuildAccelerationStructureDesc{
      .dst_as = *tlas,
      .inputs = wand::BuildTopLevelAsInputs{
        .instances = wand::BufferRange{
          *instance_bufs[frame_idx], wand::BufferRegion{
            .offset = 0,
            .size = kNumInstances * sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
            .stride = sizeof(D3D12_RAYTRACING_INSTANCE_DESC)
          }
        },
        .flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE
      },
      .src_as = tlas.get(),
      .scratch_buffer = wand::BufferRange{*tlas_update_scratch}
    };

    cmd.BuildAccelerationStructure(tlas_update_desc);

    // Dispatch

    auto const rt_desc = ctx.accum_tex->GetDesc();

    if constexpr (kUseRayQuery) {
      cmd.SetPipelineState(*cs_pso);
      cmd.SetAccelerationStructure(PIPELINE_PARAM_INDEX(ComputeDispatchParams, as_srv), *tlas);
      cmd.SetUnorderedAccess(PIPELINE_PARAM_INDEX(ComputeDispatchParams, target_uav), *ctx.accum_tex);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ComputeDispatchParams, width), rt_desc.width);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ComputeDispatchParams, height), rt_desc.height);
      cmd.SetShaderResource(PIPELINE_PARAM_INDEX(ComputeDispatchParams, inst_table_srv), *inst_table_buf);
      cmd.SetShaderResource(PIPELINE_PARAM_INDEX(ComputeDispatchParams, geom_table_srv), *geom_table_buf);
      cmd.SetShaderResource(PIPELINE_PARAM_INDEX(ComputeDispatchParams, mtl_table_srv), *mtl_table_buf);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ComputeDispatchParams, sample_count), ctx.sample_count);

      auto const dispatch_x = (rt_desc.width + COMPUTE_SHADER_TILE_DIM - 1) / COMPUTE_SHADER_TILE_DIM;
      auto const dispatch_y = (rt_desc.height + COMPUTE_SHADER_TILE_DIM - 1) / COMPUTE_SHADER_TILE_DIM;
      cmd.Dispatch(dispatch_x, dispatch_y, 1);
    } else {
      cmd.SetRtState(*pso);
      cmd.SetAccelerationStructure(PIPELINE_PARAM_INDEX(RayDispatchParams, tlas_idx), *tlas);
      cmd.SetUnorderedAccess(PIPELINE_PARAM_INDEX(RayDispatchParams, uav_idx), *ctx.accum_tex);

      cmd.DispatchRays(wand::DispatchRaysDesc{
        .ray_generation_shader_record = wand::BufferRange{
          *shader_table, wand::BufferRegion{.size = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES}
        },
        .miss_shader_table = wand::BufferRange{
          *shader_table,
          wand::BufferRegion{
            .offset = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, .size = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
          }
        },
        .hit_group_table = wand::BufferRange{
          *shader_table, wand::BufferRegion{
            .offset = 2 * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, .size = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
          }
        },
        .width = rt_desc.width,
        .height = rt_desc.height
      });
    }


    // Increase sample count to account for this frame

    ++ctx.sample_count;

    // Resolve pass
    {
      cmd.SetPipelineState(*resolve_pso);
      cmd.SetShaderResource(PIPELINE_PARAM_INDEX(ResolveParams, accum_srv), *ctx.accum_tex);
      cmd.SetUnorderedAccess(PIPELINE_PARAM_INDEX(ResolveParams, target_uav), *ctx.resolve_tex);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ResolveParams, width), rt_desc.width);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ResolveParams, height), rt_desc.height);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ResolveParams, sample_count), ctx.sample_count);
      cmd.SetPipelineParameter(PIPELINE_PARAM_INDEX(ResolveParams, exposure), std::bit_cast<UINT>(kExposure));

      auto const dispatch_x = (rt_desc.width + COMPUTE_SHADER_TILE_DIM - 1) / COMPUTE_SHADER_TILE_DIM;
      auto const dispatch_y = (rt_desc.height + COMPUTE_SHADER_TILE_DIM - 1) / COMPUTE_SHADER_TILE_DIM;
      cmd.Dispatch(dispatch_x, dispatch_y, 1);

      cmd.CopyTexture(ctx.swap_chain->GetCurrentTexture(), *ctx.resolve_tex);
    }

    cmd.End();
    ctx.device->ExecuteCommandLists(std::span{&cmd, 1});

    ctx.device->Present(*ctx.swap_chain);

    // Wait for queued frames
    auto const frame_fence_val = frame_fence->GetNextValue();
    ctx.device->SignalFence(*frame_fence);
    frame_fence->Wait(SatSub<UINT64>(frame_fence_val, kMaxGpuQueuedFrames));

    ++frame_count;
  }
}
