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


extern "C" {
__declspec(dllexport) extern UINT const D3D12SDKVersion = D3D12_SDK_VERSION;
__declspec(dllexport) extern char const* D3D12SDKPath = ".\\D3D12\\";
}


using Microsoft::WRL::ComPtr;

constexpr DXGI_SAMPLE_DESC NO_AA = {.Count = 1, .Quality = 0};
constexpr D3D12_HEAP_PROPERTIES UPLOAD_HEAP = {.Type = D3D12_HEAP_TYPE_UPLOAD};
constexpr D3D12_HEAP_PROPERTIES DEFAULT_HEAP = {.Type = D3D12_HEAP_TYPE_DEFAULT};
constexpr D3D12_RESOURCE_DESC BASIC_BUFFER_DESC = {
  .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
  .Width = 0,
  .Height = 1,
  .DepthOrArraySize = 1,
  .MipLevels = 1,
  .SampleDesc = NO_AA,
  .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
};


ID3D12Device5* device;
ID3D12CommandQueue* cmd_queue;
ID3D12Fence* fence;

ID3D12Resource* render_target;

IDXGISwapChain3* swap_chain;
ID3D12DescriptorHeap* uav_heap;

ID3D12CommandAllocator* cmd_alloc;
ID3D12GraphicsCommandList4* cmd_list;

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


constexpr UINT NUM_INSTANCES = 3;
D3D12_RAYTRACING_INSTANCE_DESC* instance_data;

ID3D12RootSignature* root_signature;

ID3D12StateObject* pso;
constexpr UINT64 NUM_SHADER_IDS = 3;
ID3D12Resource* shader_ids;


auto Resize(HWND hwnd) -> void;
auto Flush() -> void;
auto UpdateTransforms() -> void;


auto ThrowIfFailed(HRESULT const hr) -> void {
  if (FAILED(hr)) {
    throw std::runtime_error{"HRESULT returned failure code."};
  }
}


auto WINAPI WndProc(HWND const hwnd, UINT const msg, WPARAM const wparam, LPARAM const lparam) -> LRESULT {
  switch (msg) {
  case WM_CLOSE:
  case WM_DESTROY: {
    PostQuitMessage(0);
    [[fallthrough]];
  }
  case WM_SIZING:
  case WM_SIZE: {
    Resize(hwnd);
    [[fallthrough]];
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

  auto const hwnd = CreateWindowExW(
    0, wcw.lpszClassName, L"DXR tutorial", WS_VISIBLE | WS_OVERLAPPEDWINDOW,
    CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
    nullptr, nullptr, nullptr, nullptr);

  // Create D3D12 device

  {
    ComPtr<IDXGIFactory4> factory;

    if (FAILED(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)))) {
      ThrowIfFailed(CreateDXGIFactory2(00, IID_PPV_ARGS(&factory)));
    }

    if (ComPtr<ID3D12Debug> debug; SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
      debug->EnableDebugLayer();
    }

    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));

    constexpr D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
    ThrowIfFailed(device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(&cmd_queue)));

    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    // Create swapchain

    constexpr DXGI_SWAP_CHAIN_DESC1 sc_desc = {
      .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
      .SampleDesc = NO_AA,
      .BufferCount = 2,
      .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
    };

    ComPtr<IDXGISwapChain1> swap_chain1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(cmd_queue, hwnd, &sc_desc, nullptr, nullptr, &swap_chain1));
    ThrowIfFailed(swap_chain1->QueryInterface(IID_PPV_ARGS(&swap_chain)));
  }

  // Create UAV heap

  constexpr D3D12_DESCRIPTOR_HEAP_DESC uav_heap_desc = {
    .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    .NumDescriptors = 1,
    .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
  };

  ThrowIfFailed(device->CreateDescriptorHeap(&uav_heap_desc, IID_PPV_ARGS(&uav_heap)));

  Resize(hwnd);

  // Queue and command list

  ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmd_alloc)));
  ThrowIfFailed(device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE,
                                           IID_PPV_ARGS(&cmd_list)));

  // Init meshes

  auto const create_buffer_for = [](auto& data) {
    auto desc = BASIC_BUFFER_DESC;
    desc.Width = sizeof(data);

    ComPtr<ID3D12Resource> res;
    ThrowIfFailed(device->CreateCommittedResource(&UPLOAD_HEAP, D3D12_HEAP_FLAG_NONE, &desc,
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)));
    void* ptr;
    ThrowIfFailed(res->Map(0, nullptr, &ptr));
    std::memcpy(ptr, data, sizeof(data));
    res->Unmap(0, nullptr);

    return res;
  };

  auto const quad_vb = create_buffer_for(quad_vtx);
  auto const cube_vb = create_buffer_for(cube_vtx);
  auto const cube_ib = create_buffer_for(cube_idx);

  // AS utilities

  auto const make_as = [](D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const& inputs,
                          UINT64* update_scratch_size = nullptr) {
    auto const make_buffer = [](UINT64 const size, auto const initial_state) {
      auto desc = BASIC_BUFFER_DESC;
      desc.Width = size;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

      ComPtr<ID3D12Resource> buffer;
      device->CreateCommittedResource(&DEFAULT_HEAP, D3D12_HEAP_FLAG_NONE, &desc, initial_state, nullptr,
                                      IID_PPV_ARGS(&buffer));
      return buffer;
    };

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info;
    device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);

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

    ThrowIfFailed(cmd_alloc->Reset());
    ThrowIfFailed(cmd_list->Reset(cmd_alloc, nullptr));

    cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

    ThrowIfFailed(cmd_list->Close());
    cmd_queue->ExecuteCommandLists(1, CommandListCast(&cmd_list));

    Flush();

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

  auto instance_buf_desc = BASIC_BUFFER_DESC;
  instance_buf_desc.Width = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * NUM_INSTANCES;

  ComPtr<ID3D12Resource> instance_buf;
  ThrowIfFailed(device->CreateCommittedResource(&UPLOAD_HEAP, D3D12_HEAP_FLAG_NONE, &instance_buf_desc,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&instance_buf)));

  ThrowIfFailed(instance_buf->Map(0, nullptr, reinterpret_cast<void**>(&instance_data)));

  for (UINT i = 0; i < NUM_INSTANCES; i++) {
    instance_data[i] = {
      .InstanceID = i,
      .InstanceMask = 1,
      .AccelerationStructure = (i ? quad_blas : cube_blas)->GetGPUVirtualAddress()
    };
  }

  UpdateTransforms();

  // TLAS for scene

  UINT64 update_scratch_size;
  auto const tlas = make_tlas(instance_buf.Get(), NUM_INSTANCES, &update_scratch_size);

  auto tlas_update_scratch_desc = BASIC_BUFFER_DESC;
  // WARP bug workaround: use 8 if the required size was reported as less
  tlas_update_scratch_desc.Width = std::max<UINT64>(update_scratch_size, 8ull);
  tlas_update_scratch_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  ComPtr<ID3D12Resource> tlas_update_scratch;
  ThrowIfFailed(device->CreateCommittedResource(&DEFAULT_HEAP, D3D12_HEAP_FLAG_NONE, &tlas_update_scratch_desc,
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

  ThrowIfFailed(device->CreateRootSignature(0, root_sig_blob->GetBufferPointer(), root_sig_blob->GetBufferSize(),
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
  global_root_sig_desc->SetRootSignature(root_signature);

  auto* const pipeline_config_desc = pso_desc.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
  pipeline_config_desc->Config(3); // cam->mirror->floor->light

  ThrowIfFailed(device->CreateStateObject(pso_desc, IID_PPV_ARGS(&pso)));

  auto id_desc = BASIC_BUFFER_DESC;
  id_desc.Width = NUM_SHADER_IDS * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
  ThrowIfFailed(device->CreateCommittedResource(&UPLOAD_HEAP, D3D12_HEAP_FLAG_NONE, &id_desc,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&shader_ids)));

  {
    ComPtr<ID3D12StateObjectProperties> props;
    ThrowIfFailed(pso->QueryInterface(IID_PPV_ARGS(&props)));

    void* data;

    auto const write_id = [&](wchar_t const* const name) {
      auto const* const id = props->GetShaderIdentifier(name);
      std::memcpy(data, id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
      data = static_cast<char*>(data) + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    };

    ThrowIfFailed(shader_ids->Map(0, nullptr, &data));
    write_id(L"RayGeneration");
    write_id(L"Miss");
    write_id(L"HitGroup");
    shader_ids->Unmap(0, nullptr);
  }

  // Main loop

  for (MSG msg;;) {
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        return 0;
      }

      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }

    ThrowIfFailed(cmd_alloc->Reset());
    ThrowIfFailed(cmd_list->Reset(cmd_alloc, nullptr));

    // Update scene transforms

    UpdateTransforms();

    // Update TLAS

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC const tlas_update_desc = {
      .DestAccelerationStructureData = tlas->GetGPUVirtualAddress(),
      .Inputs = {
        .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
        .NumDescs = NUM_INSTANCES,
        .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
        .InstanceDescs = instance_buf->GetGPUVirtualAddress()
      },
      .SourceAccelerationStructureData = tlas->GetGPUVirtualAddress(),
      .ScratchAccelerationStructureData = tlas_update_scratch->GetGPUVirtualAddress()
    };

    cmd_list->BuildRaytracingAccelerationStructure(&tlas_update_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER const tlas_uav_barrier{
      .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = tlas.Get()}
    };

    cmd_list->ResourceBarrier(1, &tlas_uav_barrier);

    // Dispatch

    cmd_list->SetPipelineState1(pso);
    cmd_list->SetComputeRootSignature(root_signature);
    cmd_list->SetDescriptorHeaps(1, &uav_heap);
    auto const uav_table = uav_heap->GetGPUDescriptorHandleForHeapStart();
    cmd_list->SetComputeRootDescriptorTable(0, uav_table);
    cmd_list->SetComputeRootShaderResourceView(1, tlas->GetGPUVirtualAddress());

    auto const rt_desc = render_target->GetDesc();

    D3D12_DISPATCH_RAYS_DESC const dispatch_desc = {
      .RayGenerationShaderRecord = {
        .StartAddress = shader_ids->GetGPUVirtualAddress(),
        .SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
      },
      .MissShaderTable = {
        .StartAddress = shader_ids->GetGPUVirtualAddress() + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
        .SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
      },
      .HitGroupTable = {
        .StartAddress = shader_ids->GetGPUVirtualAddress() + 2 * D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT,
        .SizeInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES
      },
      .Width = static_cast<UINT>(rt_desc.Width),
      .Height = rt_desc.Height,
      .Depth = 1
    };

    cmd_list->DispatchRays(&dispatch_desc);

    {
      ComPtr<ID3D12Resource> back_buffer;
      ThrowIfFailed(swap_chain->GetBuffer(swap_chain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back_buffer)));

      auto const rt_barrier = [](auto* const resource, auto const before, auto const after) {
        D3D12_RESOURCE_BARRIER const rb = {
          .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
          .Transition = {
            .pResource = resource,
            .StateBefore = before,
            .StateAfter = after
          }
        };
        cmd_list->ResourceBarrier(1, &rb);
      };

      rt_barrier(render_target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
      rt_barrier(back_buffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);

      cmd_list->CopyResource(back_buffer.Get(), render_target);

      rt_barrier(back_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
      rt_barrier(render_target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    ThrowIfFailed(cmd_list->Close());
    cmd_queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&cmd_list));

    Flush();

    ThrowIfFailed(swap_chain->Present(1, 0));
  }
}


auto Resize(HWND const hwnd) -> void {
  if (!swap_chain) [[unlikely]] {
    return;
  }

  RECT rect;
  GetClientRect(hwnd, &rect);

  auto const width = std::max<UINT>(rect.right - rect.left, 1);
  auto const height = std::max<UINT>(rect.bottom - rect.top, 1);

  Flush();

  swap_chain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);

  if (render_target) [[likely]] {
    render_target->Release();
  }

  D3D12_RESOURCE_DESC const rt_desc = {
    .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    .Width = width,
    .Height = height,
    .DepthOrArraySize = 1,
    .MipLevels = 1,
    .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
    .SampleDesc = NO_AA,
    .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
  };
  device->CreateCommittedResource(&DEFAULT_HEAP, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  nullptr, IID_PPV_ARGS(&render_target));

  constexpr D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {
    .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
    .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D
  };
  device->CreateUnorderedAccessView(render_target, nullptr, &uav_desc,
                                    uav_heap->GetCPUDescriptorHandleForHeapStart());
}


auto Flush() -> void {
  static UINT64 value = 1;
  cmd_queue->Signal(fence, value);
  fence->SetEventOnCompletion(value++, nullptr);
}


auto UpdateTransforms() -> void {
  using namespace DirectX;
  auto const set = [](int const idx, XMMATRIX const mx) {
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
