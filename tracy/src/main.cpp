#define NOMINMAX
#define WIN32_LEAN_AND_MEAN 
#include <d3d12.h>
#include <d3dx12.h>
#include <DirectXMath.h>
#include <dxgi1_4.h>
#include <Windows.h>
#include <wrl/client.h>

#ifndef NDEBUG
#include "generated/Debug/shader_lib.h"
#else
#include "generated/Release/shader_lib.h"
#endif

import std;

using Microsoft::WRL::ComPtr;

extern "C" {
__declspec(dllexport) extern UINT const D3D12SDKVersion = D3D12_SDK_VERSION;
__declspec(dllexport) extern char const* D3D12SDKPath = ".\\D3D12\\";
}


constexpr UINT kNumFramesInFlight{2};


struct RenderingContext {
  ComPtr<ID3D12Device5> device;
  ComPtr<ID3D12CommandQueue> cmd_queue;
  ComPtr<ID3D12Fence> fence;
  ComPtr<ID3D12Resource> render_target;
  ComPtr<IDXGISwapChain3> swap_chain;
  ComPtr<ID3D12DescriptorHeap> uav_heap;
  std::array<ComPtr<ID3D12CommandAllocator>, kNumFramesInFlight> cmd_alloc;
  std::array<ComPtr<ID3D12GraphicsCommandList4>, kNumFramesInFlight> cmd_list;

  UINT64 fence_val = 1;
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


auto WaitGpuIdle(RenderingContext& ctx) -> void {
  ThrowIfFailed(ctx.cmd_queue->Signal(ctx.fence.Get(), ctx.fence_val));
  ThrowIfFailed(ctx.fence->SetEventOnCompletion(ctx.fence_val++, nullptr));
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

  WaitGpuIdle(*ctx);

  ThrowIfFailed(ctx->swap_chain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0));

  D3D12_RESOURCE_DESC const rt_desc = {
    .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    .Width = width,
    .Height = height,
    .DepthOrArraySize = 1,
    .MipLevels = 1,
    .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
    .SampleDesc = kNoAaDesc,
    .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
  };
  ThrowIfFailed(ctx->device->CreateCommittedResource(&kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                     nullptr, IID_PPV_ARGS(&ctx->render_target)));

  constexpr D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {
    .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
    .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D
  };
  ctx->device->CreateUnorderedAccessView(ctx->render_target.Get(), nullptr, &uav_desc,
                                         ctx->uav_heap->GetCPUDescriptorHandleForHeapStart());
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

  // Create D3D12 device

  {
    ComPtr<IDXGIFactory4> factory;

    if (FAILED(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)))) {
      ThrowIfFailed(CreateDXGIFactory2(00, IID_PPV_ARGS(&factory)));
    }

    if (ComPtr<ID3D12Debug> debug; SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
      debug->EnableDebugLayer();
    }

    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&ctx.device)));

    constexpr D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
    ThrowIfFailed(ctx.device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(&ctx.cmd_queue)));

    ThrowIfFailed(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fence)));

    // Create swapchain

    constexpr DXGI_SWAP_CHAIN_DESC1 sc_desc = {
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = kNoAaDesc,
      .BufferCount = 2,
      .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
    };

    ComPtr<IDXGISwapChain1> swap_chain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(ctx.cmd_queue.Get(), hwnd.get(), &sc_desc, nullptr, nullptr,
                                                  &swap_chain1));
    ThrowIfFailed(swap_chain1->QueryInterface(IID_PPV_ARGS(&ctx.swap_chain)));
  }

  // Create UAV heap

  constexpr D3D12_DESCRIPTOR_HEAP_DESC uav_heap_desc = {
    .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    .NumDescriptors = 1,
    .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
  };

  ThrowIfFailed(ctx.device->CreateDescriptorHeap(&uav_heap_desc, IID_PPV_ARGS(&ctx.uav_heap)));

  Resize(hwnd.get());

  // Queue and command list

  for (auto i = 0u; i < kNumFramesInFlight; i++) {
    ThrowIfFailed(ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx.cmd_alloc[i])));
    ThrowIfFailed(ctx.device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE,
                                                 IID_PPV_ARGS(&ctx.cmd_list[i])));
  }

  // Init meshes

  auto const create_buffer_for = [&ctx](auto& data) {
    auto desc = kBasicBufferDesc;
    desc.Width = sizeof(data);

    ComPtr<ID3D12Resource> res;
    ThrowIfFailed(ctx.device->CreateCommittedResource(&kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)));
    void* ptr;
    ThrowIfFailed(res->Map(0, nullptr, &ptr));
    std::memcpy(ptr, data, sizeof(data));
    res->Unmap(0, nullptr);

    return res;
  };

  constexpr float quad_vtx[] = {
    -1, 0, -1, -1, 0, 1, 1, 0, 1,
    -1, 0, -1, 1, 0, -1, 1, 0, 1
  };
  constexpr float cube_vtx[] = {
    -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
    -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1
  };
  constexpr short cube_idx[] = {
    4, 6, 0, 2, 0, 6, 0, 1, 4, 5, 4, 1,
    0, 2, 1, 3, 1, 2, 1, 3, 5, 7, 5, 3,
    2, 6, 3, 7, 3, 6, 4, 5, 6, 7, 6, 5
  };

  auto const quad_vb = create_buffer_for(quad_vtx);
  auto const cube_vb = create_buffer_for(cube_vtx);
  auto const cube_ib = create_buffer_for(cube_idx);

  // AS utilities

  auto const make_as = [&ctx](D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const& inputs,
                              UINT64* update_scratch_size = nullptr) {
    auto const make_buffer = [&ctx](UINT64 const size, auto const initial_state) {
      auto desc = kBasicBufferDesc;
      desc.Width = size;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

      ComPtr<ID3D12Resource> buffer;
      ctx.device->CreateCommittedResource(&kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc, initial_state, nullptr,
                                          IID_PPV_ARGS(&buffer));
      return buffer;
    };

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info;
    ctx.device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);

    if (update_scratch_size) {
      *update_scratch_size = prebuild_info.UpdateScratchDataSizeInBytes;
    }

    auto const scratch = make_buffer(prebuild_info.ScratchDataSizeInBytes, D3D12_RESOURCE_STATE_COMMON);
    auto const as = make_buffer(prebuild_info.ResultDataMaxSizeInBytes,
                                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC const build_desc = {
      .DestAccelerationStructureData = as->GetGPUVirtualAddress(),
      .Inputs = inputs,
      .ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress()
    };

    ThrowIfFailed(ctx.cmd_alloc[0]->Reset());
    ThrowIfFailed(ctx.cmd_list[0]->Reset(ctx.cmd_alloc[0].Get(), nullptr));

    ctx.cmd_list[0]->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    ThrowIfFailed(ctx.cmd_list[0]->Close());
    ctx.cmd_queue->ExecuteCommandLists(1, CommandListCast(ctx.cmd_list[0].GetAddressOf()));

    WaitGpuIdle(ctx);

    return as;
  };

  auto const make_blas = [make_as](ID3D12Resource* const vertex_buffer, UINT const vertex_floats,
                                   ID3D12Resource* const index_buffer = nullptr,
                                   UINT const indices = 0) {
    D3D12_RAYTRACING_GEOMETRY_DESC const geometry_desc = {
      .Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
      .Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
      .Triangles = {
        .Transform3x4 = 0,
        .IndexFormat = index_buffer ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_UNKNOWN,
        .VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT,
        .IndexCount = indices,
        .VertexCount = vertex_floats / 3,
        .IndexBuffer = index_buffer ? index_buffer->GetGPUVirtualAddress() : 0,
        .VertexBuffer = {
          .StartAddress = vertex_buffer->GetGPUVirtualAddress(),
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

  auto const make_tlas = [make_as](ID3D12Resource* const instances, UINT const num_instances,
                                   UINT64* const update_scratch_size) {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const inputs = {
      .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
      .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE,
      .NumDescs = num_instances,
      .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
      .InstanceDescs = instances->GetGPUVirtualAddress()
    };

    return make_as(inputs, update_scratch_size);
  };

  // BLAS for meshes

  auto const quad_blas = make_blas(quad_vb.Get(), std::size(quad_vtx));
  auto const cube_blas = make_blas(cube_vb.Get(), std::size(cube_vtx), cube_ib.Get(), std::size(cube_idx));

  // Init scene

  auto instance_buf_desc = kBasicBufferDesc;
  instance_buf_desc.Width = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * kNumInstances;

  std::array<ComPtr<ID3D12Resource>, kNumFramesInFlight> instance_bufs;
  std::array<D3D12_RAYTRACING_INSTANCE_DESC*, kNumFramesInFlight> instance_data;

  for (auto i = 0u; i < kNumFramesInFlight; i++) {
    ThrowIfFailed(ctx.device->CreateCommittedResource(&kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &instance_buf_desc,
                                                      D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                      IID_PPV_ARGS(&instance_bufs[i])));

    ThrowIfFailed(instance_bufs[i]->Map(0, nullptr, reinterpret_cast<void**>(&instance_data[i])));

    for (UINT j = 0; j < kNumInstances; j++) {
      instance_data[i][j] = {
        .InstanceID = j,
        .InstanceMask = 1,
        .AccelerationStructure = (j ? quad_blas : cube_blas)->GetGPUVirtualAddress()
      };
    }
  }

  UpdateTransforms(instance_data[0]);

  // TLAS for scene

  UINT64 update_scratch_size;
  auto const tlas = make_tlas(instance_bufs[0].Get(), kNumInstances, &update_scratch_size);

  auto tlas_update_scratch_desc = kBasicBufferDesc;
  // WARP bug workaround: use 8 if the required size was reported as less
  tlas_update_scratch_desc.Width = std::max<UINT64>(update_scratch_size, 8ull);
  tlas_update_scratch_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  ComPtr<ID3D12Resource> tlas_update_scratch;
  ThrowIfFailed(ctx.device->CreateCommittedResource(&kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &tlas_update_scratch_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&tlas_update_scratch)));

  // Create root signature

  CD3DX12_DESCRIPTOR_RANGE const uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0};

  std::array<CD3DX12_ROOT_PARAMETER, 2> root_params;
  root_params[0].InitAsDescriptorTable(1, &uav_range);
  root_params[1].InitAsShaderResourceView(0);

  CD3DX12_ROOT_SIGNATURE_DESC const root_sig_desc{static_cast<UINT>(root_params.size()), root_params.data()};

  ComPtr<ID3DBlob> root_sig_blob;
  ThrowIfFailed(D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &root_sig_blob, nullptr));

  ComPtr<ID3D12RootSignature> root_signature;
  ThrowIfFailed(ctx.device->CreateRootSignature(0, root_sig_blob->GetBufferPointer(), root_sig_blob->GetBufferSize(),
                                                IID_PPV_ARGS(&root_signature)));

  // Create PSO

  CD3DX12_STATE_OBJECT_DESC pso_desc{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE};

  auto* const lib_desc = pso_desc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
  CD3DX12_SHADER_BYTECODE const shader_lib_code{g_shader_lib_bin, std::size(g_shader_lib_bin)};
  lib_desc->SetDXILLibrary(&shader_lib_code);

  auto* const hit_group_desc = pso_desc.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
  hit_group_desc->SetHitGroupExport(L"HitGroup");
  hit_group_desc->SetClosestHitShaderImport(L"ClosestHit");

  auto* const shader_config_desc = pso_desc.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
  shader_config_desc->Config(20, 8); // sizeof(Payload), sizeof(attribs)

  auto* const global_root_sig_desc = pso_desc.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
  global_root_sig_desc->SetRootSignature(root_signature.Get());

  auto* const pipeline_config_desc = pso_desc.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
  pipeline_config_desc->Config(3); // cam->mirror->floor->light

  ComPtr<ID3D12StateObject> pso;
  ThrowIfFailed(ctx.device->CreateStateObject(pso_desc, IID_PPV_ARGS(&pso)));

  // Create shader table

  constexpr UINT64 kShaderTableSize{3}; // 1 raygen + 1 miss + 1 hitgroup
  auto shader_id_buf_desc = kBasicBufferDesc;
  shader_id_buf_desc.Width = kShaderTableSize * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;

  ComPtr<ID3D12Resource> shader_table;
  ThrowIfFailed(ctx.device->CreateCommittedResource(&kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &shader_id_buf_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&shader_table)));

  {
    ComPtr<ID3D12StateObjectProperties> pso_props;
    ThrowIfFailed(pso->QueryInterface(IID_PPV_ARGS(&pso_props)));

    void* shader_table_data;
    ThrowIfFailed(shader_table->Map(0, nullptr, &shader_table_data));

    auto const write_id = [&](wchar_t const* const name) {
      auto const* const id = pso_props->GetShaderIdentifier(name);
      std::memcpy(shader_table_data, id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
      shader_table_data = static_cast<char*>(shader_table_data) + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    };

    write_id(L"RayGeneration");
    write_id(L"Miss");
    write_id(L"HitGroup");

    shader_table->Unmap(0, nullptr);
  }

  // Main loop

  UINT frame_count{0};

  for (MSG msg;;) {
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        WaitGpuIdle(ctx);
        SetWindowLongPtrW(hwnd.get(), GWLP_USERDATA, 0);
        return 0;
      }

      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }

    auto const frame_idx = frame_count % kNumFramesInFlight;

    ThrowIfFailed(ctx.cmd_alloc[frame_idx]->Reset());
    ThrowIfFailed(ctx.cmd_list[frame_idx]->Reset(ctx.cmd_alloc[frame_idx].Get(), nullptr));

    // Update scene transforms

    UpdateTransforms(instance_data[frame_idx]);

    // Update TLAS

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC const tlas_update_desc = {
      .DestAccelerationStructureData = tlas->GetGPUVirtualAddress(),
      .Inputs = {
        .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
        .NumDescs = kNumInstances,
        .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
        .InstanceDescs = instance_bufs[frame_idx]->GetGPUVirtualAddress()
      },
      .SourceAccelerationStructureData = tlas->GetGPUVirtualAddress(),
      .ScratchAccelerationStructureData = tlas_update_scratch->GetGPUVirtualAddress()
    };

    ctx.cmd_list[frame_idx]->BuildRaytracingAccelerationStructure(&tlas_update_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER const tlas_uav_barrier{
      .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = tlas.Get()}
    };

    ctx.cmd_list[frame_idx]->ResourceBarrier(1, &tlas_uav_barrier);

    // Dispatch

    ctx.cmd_list[frame_idx]->SetPipelineState1(pso.Get());
    ctx.cmd_list[frame_idx]->SetComputeRootSignature(root_signature.Get());
    ctx.cmd_list[frame_idx]->SetDescriptorHeaps(1, ctx.uav_heap.GetAddressOf());
    auto const uav_table = ctx.uav_heap->GetGPUDescriptorHandleForHeapStart();
    ctx.cmd_list[frame_idx]->SetComputeRootDescriptorTable(0, uav_table);
    ctx.cmd_list[frame_idx]->SetComputeRootShaderResourceView(1, tlas->GetGPUVirtualAddress());

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

    ctx.cmd_list[frame_idx]->DispatchRays(&dispatch_desc);

    {
      ComPtr<ID3D12Resource> back_buffer;
      ThrowIfFailed(ctx.swap_chain->GetBuffer(ctx.swap_chain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back_buffer)));

      auto const rt_barrier = [&ctx, frame_idx](auto* const resource, auto const before, auto const after) {
        D3D12_RESOURCE_BARRIER const rb = {
          .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
          .Transition = {
            .pResource = resource,
            .StateBefore = before,
            .StateAfter = after
          }
        };
        ctx.cmd_list[frame_idx]->ResourceBarrier(1, &rb);
      };

      rt_barrier(ctx.render_target.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
      rt_barrier(back_buffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);

      ctx.cmd_list[frame_idx]->CopyResource(back_buffer.Get(), ctx.render_target.Get());

      rt_barrier(back_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
      rt_barrier(ctx.render_target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    ThrowIfFailed(ctx.cmd_list[frame_idx]->Close());
    ctx.cmd_queue->
        ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(ctx.cmd_list[frame_idx].GetAddressOf()));

    ThrowIfFailed(ctx.cmd_queue->Signal(ctx.fence.Get(), ctx.fence_val));
    ++ctx.fence_val;
    ThrowIfFailed(ctx.fence->SetEventOnCompletion(ctx.fence_val - kNumFramesInFlight, nullptr));

    ThrowIfFailed(ctx.swap_chain->Present(1, 0));

    ++frame_count;
  }
}
