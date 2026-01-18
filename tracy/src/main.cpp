#include <DirectXMath.h>
#include <wand/wand.hpp>

#ifndef NDEBUG
#include "generated/Debug/shader_lib.h"
#else
#include "generated/Release/shader_lib.h"
#endif

import std;

extern "C" {
__declspec(dllexport) extern UINT const D3D12SDKVersion = D3D12_SDK_VERSION;
__declspec(dllexport) extern char const* D3D12SDKPath = ".\\D3D12\\";
}


template<unsigned N>
using Vector = std::array<float, N>;


constexpr UINT kNumFramesInFlight{2};


struct RenderingContext {
  std::unique_ptr<wand::GraphicsDevice> device;
  wand::SharedDeviceChildHandle<wand::SwapChain> swap_chain;
  std::array<wand::SharedDeviceChildHandle<wand::CommandList>, kNumFramesInFlight> cmd_lists;
  wand::SharedDeviceChildHandle<wand::Texture> render_target;
};


constexpr UINT kNumInstances{3};
constexpr DXGI_SAMPLE_DESC kNoAaDesc = {
  .Count = 1,
  .Quality = 0
};
constexpr D3D12_HEAP_PROPERTIES kUploadHeapProps{
  .Type = D3D12_HEAP_TYPE_UPLOAD
};
constexpr D3D12_HEAP_PROPERTIES kDefaultHeapProps{
  .Type = D3D12_HEAP_TYPE_DEFAULT
};
constexpr D3D12_RESOURCE_DESC kBasicBufferDesc{
  .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
  .Width = 0,
  .Height = 1,
  .DepthOrArraySize = 1,
  .MipLevels = 1,
  .SampleDesc = kNoAaDesc,
  .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
};


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

  ctx->render_target = ctx->device->CreateTexture(
    wand::TextureDesc{
      .dimension = wand::TextureDimension::k2D, .width = width, .height = height, .depth_or_array_size = 1,
      .mip_levels = 1, .format = DXGI_FORMAT_R8G8B8A8_UNORM, .sample_count = 1, .depth_stencil = false,
      .render_target = false, .shader_resource = false, .unordered_access = true
    }, wand::CpuAccess::kNone, nullptr);
}


auto UpdateTransforms(D3D12_RAYTRACING_INSTANCE_DESC* const instance_data) -> void {
  using namespace DirectX;
  auto const set = [instance_data](int const idx, XMMATRIX const& mx) {
    auto* const ptr = reinterpret_cast<XMFLOAT3X4*>(&instance_data[idx].Transform);
    XMStoreFloat3x4(ptr, mx);
  };

  auto const time = static_cast<float>(GetTickCount64()) / 1000.0f;

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
    .lpfnWndProc = &WndProc, .hCursor = LoadCursor(nullptr, IDC_ARROW), .lpszClassName = L"DxrTutorialClass"
  };

  RegisterClassW(&wcw);

  auto const hwnd = std::unique_ptr<std::remove_pointer_t<HWND>, HwndDeleter>{
    CreateWindowExW(0, wcw.lpszClassName, L"DXR tutorial", WS_VISIBLE | WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,
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
    auto const buffer = ctx.device->CreateBuffer(
      wand::BufferDesc{
        .size = sizeof(data), .stride = 0, .constant_buffer = false,
        .shader_resource = false, .unordered_access = false
      }, wand::CpuAccess::kWrite);

    auto* const data_ptr = buffer->Map();
    std::memcpy(data_ptr, &data, sizeof(data));
    buffer->Unmap();

    return buffer;
  };

  constexpr std::array quad_vertices{
    Vector<3>{-1, 0, -1}, Vector<3>{-1, 0, 1}, Vector<3>{1, 0, 1},
    Vector<3>{-1, 0, -1}, Vector<3>{1, 0, -1}, Vector<3>{1, 0, 1}
  };

  constexpr std::array cube_vertices{
    Vector<3>{-1, -1, -1}, Vector<3>{1, -1, -1}, Vector<3>{-1, 1, -1}, Vector<3>{1, 1, -1},
    Vector<3>{-1, -1, 1}, Vector<3>{1, -1, 1}, Vector<3>{-1, 1, 1}, Vector<3>{1, 1, 1}
  };

  constexpr std::array cube_indices{
    4u, 6u, 0u, 2u, 0u, 6u, 0u, 1u, 4u, 5u, 4u, 1u,
    0u, 2u, 1u, 3u, 1u, 2u, 1u, 3u, 5u, 7u, 5u, 3u,
    2u, 6u, 3u, 7u, 3u, 6u, 4u, 5u, 6u, 7u, 6u, 5u
  };

  auto const quad_vb = create_buffer_for(quad_vertices);
  auto const cube_vb = create_buffer_for(cube_vertices);
  auto const cube_ib = create_buffer_for(cube_indices);

  // AS utilities

  auto const make_as = [&ctx](D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const& inputs,
                              UINT64* update_scratch_size = nullptr) {
    auto const make_buffer = [&ctx](UINT64 const size) {
      return ctx.device->CreateBuffer(
        wand::BufferDesc{
          .size = size, .stride = 0, .constant_buffer = false, .shader_resource = false,
          .unordered_access = true
        }, wand::CpuAccess::kWrite);
    };

    auto const prebuild_info = ctx.device->GetRtAccelerationStructurePrebuildInfo(inputs);

    if (update_scratch_size) {
      *update_scratch_size = prebuild_info.UpdateScratchDataSizeInBytes;
    }

    auto const scratch = make_buffer(prebuild_info.ScratchDataSizeInBytes);
    auto const as = make_buffer(prebuild_info.ResultDataMaxSizeInBytes);

    wand::BuildRaytracingAccelerationStructureDesc const build_desc = {
      .dst_as = as.get(),
      .inputs = inputs,
      .scratch_buffer = scratch.get()
    };

    ctx.cmd_lists[0]->Begin(nullptr);
    ctx.cmd_lists[0]->BuildRaytracingAccelerationStructure(std::span{&build_desc, 1});
    ctx.cmd_lists[0]->End();

    ctx.device->ExecuteCommandLists(std::span{ctx.cmd_lists[0].get(), 1});
    ctx.device->WaitIdle();

    return as;
  };

  auto const make_blas = [make_as](wand::Buffer const& vertex_buffer, UINT const vtx_count,
                                   wand::Buffer const* const index_buffer = nullptr,
                                   UINT const idx_count = 0) {
    D3D12_RAYTRACING_GEOMETRY_DESC const geometry_desc = {
      .Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
      .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
      .Triangles = {
        .Transform3x4 = 0,
        .IndexFormat = index_buffer ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_UNKNOWN,
        .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
        .IndexCount = idx_count,
        .VertexCount = vtx_count,
        .IndexBuffer = index_buffer ? index_buffer->GetInternalResource()->GetGPUVirtualAddress() : 0,
        .VertexBuffer = {
          .StartAddress = vertex_buffer.GetInternalResource()->GetGPUVirtualAddress(),
          .StrideInBytes = 3 * sizeof(float)
        }
      }
    };

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const inputs = {
      .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
      .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
      .NumDescs = 1,
      .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
      .pGeometryDescs = &geometry_desc
    };

    return make_as(inputs);
  };

  auto const make_tlas = [make_as](wand::Buffer const& instances, UINT const num_instances,
                                   UINT64* const update_scratch_size) {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const inputs = {
      .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
      .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE,
      .NumDescs = num_instances,
      .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
      .InstanceDescs = instances.GetInternalResource()->GetGPUVirtualAddress()
    };

    return make_as(inputs, update_scratch_size);
  };

  // BLAS for meshes

  auto const quad_blas = make_blas(*quad_vb, std::size(quad_vertices));
  auto const cube_blas = make_blas(*cube_vb, std::size(cube_vertices), cube_ib.get(), std::size(cube_indices));

  // Init scene

  wand::BufferDesc constexpr instance_buf_desc{
    .size = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * kNumInstances, .stride = sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
    .constant_buffer = false, .shader_resource = false, .unordered_access = false
  };

  std::array<wand::SharedDeviceChildHandle<wand::Buffer>, kNumFramesInFlight> instance_bufs;
  std::array<D3D12_RAYTRACING_INSTANCE_DESC*, kNumFramesInFlight> instance_data;

  for (auto i = 0u; i < kNumFramesInFlight; i++) {
    instance_bufs[i] = ctx.device->CreateBuffer(instance_buf_desc, wand::CpuAccess::kWrite);

    instance_data[i] = reinterpret_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(instance_bufs[i]->Map());

    for (UINT j = 0; j < kNumInstances; j++) {
      instance_data[i][j] = {
        .InstanceID = j,
        .InstanceMask = 1,
        .AccelerationStructure = (j ? quad_blas : cube_blas)->GetInternalResource()->GetGPUVirtualAddress()
      };
    }
  }

  UpdateTransforms(instance_data[0]);

  // TLAS for scene

  UINT64 update_scratch_size;
  auto const tlas = make_tlas(*instance_bufs[0], kNumInstances, &update_scratch_size);

  auto const tlas_update_scratch = ctx.device->CreateBuffer(
    wand::BufferDesc{
      /* WARP bug workaround: use 8 if the required size was reported as less */
      .size = std::max<UINT64>(update_scratch_size, 8ull),
      .stride = 0, .constant_buffer = false, .shader_resource = false, .unordered_access = true
    }, wand::CpuAccess::kNone
  );

  // Create PSO

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
      .size = kShaderTableSize * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, .stride = 0, .constant_buffer = false,
      .shader_resource = false, .unordered_access = false
    }, wand::CpuAccess::kWrite);

  {
    ComPtr<ID3D12StateObjectProperties> pso_props;
    ThrowIfFailed(pso->QueryInterface(IID_PPV_ARGS(&pso_props)));

    auto shader_table_data = shader_table->Map();

    auto const write_id = [&](wchar_t const* const name) {
      auto const* const id = pso_props->GetShaderIdentifier(name);
      std::memcpy(shader_table_data, id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
      shader_table_data = static_cast<char*>(shader_table_data) + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    };

    write_id(L"RayGeneration");
    write_id(L"Miss");
    write_id(L"HitGroup");

    shader_table->Unmap();
  }

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

    ctx.cmd_lists[frame_idx]->Begin(nullptr);

    // Update scene transforms

    UpdateTransforms(instance_data[frame_idx]);

    // Update TLAS

    wand::BuildRaytracingAccelerationStructureDesc const tlas_update_desc{
      .dst_as = tlas.get(),
      .inputs = {
        .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
        .NumDescs = kNumInstances,
        .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
        .InstanceDescs = instance_bufs[frame_idx]->GetInternalResource()->GetGPUVirtualAddress()
      },
      .src_as = tlas.get(),
      .scratch_buffer = tlas_update_scratch.get()
    };

    ctx.cmd_lists[frame_idx]->BuildRaytracingAccelerationStructure(std::span{&tlas_update_desc, 1});

    // Dispatch

    ctx.cmd_lists[frame_idx]->SetPipelineState(pso.Get());
    ctx.cmd_lists[frame_idx]->SetComputeRootSignature(root_signature.Get());
    ctx.cmd_lists[frame_idx]->SetDescriptorHeaps(1, ctx.uav_heap.GetAddressOf());
    auto const uav_table = ctx.uav_heap->GetGPUDescriptorHandleForHeapStart();
    ctx.cmd_lists[frame_idx]->SetComputeRootDescriptorTable(0, uav_table);
    ctx.cmd_lists[frame_idx]->SetComputeRootShaderResourceView(1, tlas->GetGPUVirtualAddress());

    auto const rt_desc = ctx.render_target->GetDesc();

    D3D12_DISPATCH_RAYS_DESC const dispatch_desc = {
      .RayGenerationShaderRecord = {
        .StartAddress = shader_table->GetGPUVirtualAddress(),
        .SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
      },
      .MissShaderTable = {
        .StartAddress = shader_table->GetGPUVirtualAddress() + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
        .SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
      },
      .HitGroupTable = {
        .StartAddress = shader_table->GetGPUVirtualAddress() + 2 * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
        .SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
      },
      .Width = static_cast<UINT>(rt_desc.Width),
      .Height = rt_desc.Height,
      .Depth = 1
    };

    ctx.cmd_lists[frame_idx]->DispatchRays(&dispatch_desc);

    {
      ComPtr<ID3D12Resource> back_buffer;
      ThrowIfFailed(ctx.swap_chain->GetBuffer(ctx.swap_chain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back_buffer)));

      ctx.cmd_lists[frame_idx]->CopyResource(back_buffer.Get(), ctx.render_target.Get());
    }

    ctx.cmd_lists[frame_idx]->End();
    ctx.device->ExecuteCommandLists(std::span{ctx.cmd_lists[frame_idx].get(), 1});

    ctx.device->Present(*ctx.swap_chain);

    ++frame_count;
  }
}
