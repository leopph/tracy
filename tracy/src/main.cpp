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


auto ThrowIfFailed(HRESULT const hr) -> void {
  if (FAILED(hr)) {
    throw std::runtime_error{"HRESULT returned failure code."};
  }
}


auto Resize(HWND hwnd) -> void;


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


auto Init(HWND hwnd) -> void;
auto Render() -> void;


auto main() -> int {
  // Alternatively, DPI_AWARENESS_CONTEXT_UNAWARE
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  WNDCLASSW const wcw = {
    .lpfnWndProc = &WndProc, .hCursor = LoadCursor(nullptr, IDC_ARROW), .lpszClassName = L"DxrTutorialClass"
  };

  RegisterClassW(&wcw);

  auto const hwnd = CreateWindowExW(
    0, wcw.lpszClassName, L"DXR tutorial", WS_VISIBLE | WS_OVERLAPPEDWINDOW,
    CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
    nullptr, nullptr, nullptr, nullptr);

  Init(hwnd);

  for (MSG msg;;) {
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        return 0;
      }

      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }

    Render();
  }
}


auto Init(HWND const hwnd) -> void {
#define DECLARE_AND_CALL(fn) void fn(); fn()
  DECLARE_AND_CALL(InitDevice);
  auto InitSurfaces(HWND) -> void;
  InitSurfaces(hwnd);
  DECLARE_AND_CALL(InitCommand);
  DECLARE_AND_CALL(InitMeshes);
  DECLARE_AND_CALL(InitBottomLevel);
  DECLARE_AND_CALL(InitScene);
  DECLARE_AND_CALL(InitTopLevel);
  DECLARE_AND_CALL(InitRootSignature);
  DECLARE_AND_CALL(InitPipeline);
#undef DECLARE_AND_CALL
}


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


IDXGIFactory4* factory;
ID3D12Device5* device;
ID3D12CommandQueue* cmd_queue;
ID3D12Fence* fence;


auto InitDevice() -> void {
  if (FAILED(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)))) {
    CreateDXGIFactory2(00, IID_PPV_ARGS(&factory));
  }

  if (ID3D12Debug* debug;
    SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
    debug->EnableDebugLayer();
    debug->Release();
  }

  IDXGIAdapter* adapter = nullptr;
  // factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
  D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));

  constexpr D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
  device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(&cmd_queue));

  device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
}


auto Flush() -> void {
  static UINT64 value = 1;
  cmd_queue->Signal(fence, value);
  fence->SetEventOnCompletion(value++, nullptr);
}


IDXGISwapChain3* swap_chain;
ID3D12DescriptorHeap* uav_heap;


auto InitSurfaces(HWND hwnd) -> void {
  constexpr DXGI_SWAP_CHAIN_DESC1 sc_desc = {
    .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
    .SampleDesc = NO_AA,
    .BufferCount = 2,
    .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
  };
  IDXGISwapChain1* swap_chain1;
  factory->CreateSwapChainForHwnd(cmd_queue, hwnd, &sc_desc, nullptr, nullptr, &swap_chain1);
  swap_chain1->QueryInterface(IID_PPV_ARGS(&swap_chain));
  swap_chain1->Release();

  factory->Release();

  constexpr D3D12_DESCRIPTOR_HEAP_DESC uav_heap_desc = {
    .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    .NumDescriptors = 1,
    .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
  };
  device->CreateDescriptorHeap(&uav_heap_desc, IID_PPV_ARGS(&uav_heap));

  Resize(hwnd);
}


ID3D12Resource* render_target;


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
  device->CreateCommittedResource(&DEFAULT_HEAP, D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  nullptr, IID_PPV_ARGS(&render_target));

  constexpr D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {
    .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
    .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D
  };
  device->CreateUnorderedAccessView(render_target, nullptr, &uav_desc, uav_heap->GetCPUDescriptorHandleForHeapStart());
}


ID3D12CommandAllocator* cmd_alloc;
ID3D12GraphicsCommandList4* cmd_list;


auto InitCommand() -> void {
  device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmd_alloc));
  device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmd_list));
}


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


ID3D12Resource* quad_vb;
ID3D12Resource* cube_vb;
ID3D12Resource* cube_ib;


auto InitMeshes() -> void {
  auto const make_and_copy = [](auto& data) {
    auto desc = BASIC_BUFFER_DESC;
    desc.Width = sizeof(data);
    ID3D12Resource* res;
    device->CreateCommittedResource(&UPLOAD_HEAP, D3D12_HEAP_FLAG_NONE, &desc,
                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
    void* ptr;
    res->Map(0, nullptr, &ptr);
    std::memcpy(ptr, data, sizeof(data));
    res->Unmap(0, nullptr);
    return res;
  };

  quad_vb = make_and_copy(quad_vtx);
  cube_vb = make_and_copy(cube_vtx);
  cube_ib = make_and_copy(cube_idx);
}


auto MakeAccelerationStructure(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const& inputs,
                               UINT64* update_scratch_size = nullptr) -> ID3D12Resource* {
  auto const make_buffer = [](UINT64 const size, auto const initial_state) {
    auto desc = BASIC_BUFFER_DESC;
    desc.Width = size;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* buffer;
    device->CreateCommittedResource(&DEFAULT_HEAP, D3D12_HEAP_FLAG_NONE, &desc, initial_state, nullptr,
                                    IID_PPV_ARGS(&buffer));
    return buffer;
  };

  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info;
  device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild_info);

  if (update_scratch_size) {
    *update_scratch_size = prebuild_info.UpdateScratchDataSizeInBytes;
  }

  auto* scratch = make_buffer(prebuild_info.ScratchDataSizeInBytes, D3D12_RESOURCE_STATE_COMMON);
  auto* as = make_buffer(prebuild_info.ResultDataMaxSizeInBytes,
                         D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC const build_desc = {
    .DestAccelerationStructureData = as->GetGPUVirtualAddress(),
    .Inputs = inputs,
    .ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress()
  };

  cmd_alloc->Reset();
  cmd_list->Reset(cmd_alloc, nullptr);
  cmd_list->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);
  cmd_list->Close();
  cmd_queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&cmd_list));

  Flush();
  scratch->Release();
  return as;
}


auto MakeBlas(ID3D12Resource* const vertex_buffer, UINT const vertex_floats,
              ID3D12Resource* const index_buffer = nullptr, UINT const indices = 0) -> ID3D12Resource* {
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

  return MakeAccelerationStructure(inputs);
}


ID3D12Resource* quad_blas;
ID3D12Resource* cube_blas;


auto InitBottomLevel() -> void {
  quad_blas = MakeBlas(quad_vb, std::size(quad_vtx));
  cube_blas = MakeBlas(cube_vb, std::size(cube_vtx), cube_ib, std::size(cube_idx));
}


auto MakeTlas(ID3D12Resource* const instances, UINT const num_instances,
              UINT64* const update_scratch_size) -> ID3D12Resource* {
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS const inputs = {
    .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
    .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE,
    .NumDescs = num_instances,
    .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
    .InstanceDescs = instances->GetGPUVirtualAddress()
  };

  return MakeAccelerationStructure(inputs, update_scratch_size);
}


constexpr UINT NUM_INSTANCES = 3;
ID3D12Resource* instances;
D3D12_RAYTRACING_INSTANCE_DESC* instance_data;


auto UpdateTransforms() -> void;


auto InitScene() -> void {
  auto instance_desc = BASIC_BUFFER_DESC;
  instance_desc.Width = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * NUM_INSTANCES;
  device->CreateCommittedResource(&UPLOAD_HEAP, D3D12_HEAP_FLAG_NONE, &instance_desc, D3D12_RESOURCE_STATE_COMMON,
                                  nullptr, IID_PPV_ARGS(&instances));
  instances->Map(0, nullptr, reinterpret_cast<void**>(&instance_data));

  for (UINT i = 0; i < NUM_INSTANCES; i++) {
    instance_data[i] = {
      .InstanceID = i,
      .InstanceMask = 1,
      .AccelerationStructure = (i ? quad_blas : cube_blas)->GetGPUVirtualAddress()
    };
  }

  UpdateTransforms();
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


ID3D12Resource* tlas;
ID3D12Resource* tlas_update_scratch;


auto InitTopLevel() -> void {
  UINT64 update_scratch_size;
  tlas = MakeTlas(instances, NUM_INSTANCES, &update_scratch_size);

  auto desc = BASIC_BUFFER_DESC;
  // WARP bug workaround: use 8 if the required size was reported as less
  desc.Width = std::max<UINT64>(update_scratch_size, 8ull);
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  device->CreateCommittedResource(&DEFAULT_HEAP, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                  nullptr, IID_PPV_ARGS(&tlas_update_scratch));
}


ID3D12RootSignature* root_signature;


auto InitRootSignature() -> void {
  CD3DX12_DESCRIPTOR_RANGE const uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0};

  std::array<CD3DX12_ROOT_PARAMETER, 2> root_params;
  root_params[0].InitAsDescriptorTable(1, &uav_range);
  root_params[1].InitAsShaderResourceView(0);

  CD3DX12_ROOT_SIGNATURE_DESC const root_sig_desc{static_cast<UINT>(root_params.size()), root_params.data()};

  ComPtr<ID3DBlob> root_sig_blob;
  ThrowIfFailed(D3D12SerializeRootSignature(&root_sig_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &root_sig_blob, nullptr));

  ThrowIfFailed(device->CreateRootSignature(0, root_sig_blob->GetBufferPointer(), root_sig_blob->GetBufferSize(),
                                            IID_PPV_ARGS(&root_signature)));
}


ID3D12StateObject* pso;
constexpr UINT64 NUM_SHADER_IDS = 3;
ID3D12Resource* shader_ids;


auto InitPipeline() -> void {
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
  device->CreateCommittedResource(&UPLOAD_HEAP, D3D12_HEAP_FLAG_NONE, &id_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                  IID_PPV_ARGS(&shader_ids));

  ID3D12StateObjectProperties* props;
  pso->QueryInterface(IID_PPV_ARGS(&props));

  void* data;

  auto const write_id = [&](wchar_t const* const name) {
    void* const id = props->GetShaderIdentifier(name);
    std::memcpy(data, id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    data = static_cast<char*>(data) + D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
  };

  shader_ids->Map(0, nullptr, &data);
  write_id(L"RayGeneration");
  write_id(L"Miss");
  write_id(L"HitGroup");
  shader_ids->Unmap(0, nullptr);

  props->Release();
}


auto UpdateScene() -> void {
  UpdateTransforms();

  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC const desc = {
    .DestAccelerationStructureData = tlas->GetGPUVirtualAddress(),
    .Inputs = {
      .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
      .Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
      .NumDescs = NUM_INSTANCES,
      .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
      .InstanceDescs = instances->GetGPUVirtualAddress()
    },
    .SourceAccelerationStructureData = tlas->GetGPUVirtualAddress(),
    .ScratchAccelerationStructureData = tlas_update_scratch->GetGPUVirtualAddress()
  };
  cmd_list->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);

  D3D12_RESOURCE_BARRIER const barrier{.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV, .UAV = {.pResource = tlas}};
  cmd_list->ResourceBarrier(1, &barrier);
}


auto Render() -> void {
  cmd_alloc->Reset();
  cmd_list->Reset(cmd_alloc, nullptr);

  UpdateScene();

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

  ID3D12Resource* back_buffer;
  swap_chain->GetBuffer(swap_chain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back_buffer));

  auto const barrier = [](auto* const resource, auto const before, auto const after) {
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

  barrier(render_target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  barrier(back_buffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);

  cmd_list->CopyResource(back_buffer, render_target);

  barrier(back_buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
  barrier(render_target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  back_buffer->Release();

  cmd_list->Close();
  cmd_queue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&cmd_list));

  Flush();
  swap_chain->Present(1, 0);
}
