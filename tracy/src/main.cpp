// ReSharper disable once CppInconsistentNaming
#define _CRT_SECURE_NO_WARNINGS

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <Windows.h>
#include <wrl/client.h>

import std;

auto ThrowIfFailed(HRESULT const hr) -> void {
  if (FAILED(hr)) {
    throw std::runtime_error{"Error"};
  }
}

[[nodiscard]] auto CALLBACK WindowProc(HWND const hwnd, UINT const msg, WPARAM const wParam,
                                       LPARAM const lParam) -> LRESULT {
  switch (msg) {
  case WM_DESTROY: {
    PostQuitMessage(0);
    return 0;
  }
  default: {
    return DefWindowProcW(hwnd, msg, wParam, lParam);
  }
  }
}

auto main() -> int {
  WNDCLASSW const wnd_class{
    .style = 0,
    .lpfnWndProc = &WindowProc,
    .cbClsExtra = 0,
    .cbWndExtra = 0,
    .hInstance = GetModuleHandleW(nullptr),
    .hIcon = nullptr,
    .hCursor = LoadCursorW(nullptr, IDC_ARROW),
    .hbrBackground = nullptr,
    .lpszMenuName = nullptr,
    .lpszClassName = L"TracyWindowClass",
  };

  if (!RegisterClassW(&wnd_class)) {
    return -1;
  }

  class WindowDeleter {
  public:
    auto operator()(HWND const hwnd) const -> void {
      if (hwnd) {
        DestroyWindow(hwnd);
      }
    }
  };

  std::unique_ptr<std::remove_pointer_t<HWND>, WindowDeleter> hwnd{
    CreateWindowExW(0, wnd_class.lpszClassName, L"Tracy",WS_OVERLAPPEDWINDOW,CW_USEDEFAULT, CW_USEDEFAULT,
                    CW_USEDEFAULT, CW_USEDEFAULT, nullptr, nullptr, wnd_class.hInstance, nullptr)
  };

  if (!hwnd) {
    return -1;
  }

  using Microsoft::WRL::ComPtr;

  UINT dxgi_factory_flags{0};
#ifndef NDEBUG
  dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
#endif

  ComPtr<IDXGIFactory7> dxgi_factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&dxgi_factory)));

  ComPtr<IDXGIAdapter4> hp_adapter;
  ThrowIfFailed(
    dxgi_factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&hp_adapter)));

  UINT d3d_device_flags{0};
#ifndef NDEBUG
  d3d_device_flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

  ComPtr<ID3D11Device> dev;
  ComPtr<ID3D11DeviceContext> ctx;
  ThrowIfFailed(D3D11CreateDevice(hp_adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, d3d_device_flags,
                                  std::array{D3D_FEATURE_LEVEL_12_1}.data(), 1, D3D11_SDK_VERSION, &dev, nullptr,
                                  &ctx));

#ifndef NDEBUG
  ComPtr<ID3D11Debug> debug;
  ThrowIfFailed(dev.As<ID3D11Debug>(&debug));
  ComPtr<ID3D11InfoQueue> info_queue;
  ThrowIfFailed(debug.As<ID3D11InfoQueue>(&info_queue));
  ThrowIfFailed(info_queue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_CORRUPTION, TRUE));
  ThrowIfFailed(info_queue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_ERROR, TRUE));
#endif

  auto tearing_supported{FALSE};
  ThrowIfFailed(dxgi_factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing_supported,
                                                  sizeof tearing_supported));

  UINT swap_chain_flags{0};
  UINT present_flags{0};

  if (tearing_supported) {
    swap_chain_flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    present_flags |= DXGI_PRESENT_ALLOW_TEARING;
  }

  auto constexpr swap_chain_format{DXGI_FORMAT_R8G8B8A8_UNORM};
  auto constexpr swap_chain_buffer_count{2};

  DXGI_SWAP_CHAIN_DESC1 const swap_chain_desc{
    .Width = 0, .Height = 0,
    .Format = swap_chain_format, .Stereo = FALSE, .SampleDesc{.Count = 1, .Quality = 0},
    .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS,
    .BufferCount = swap_chain_buffer_count, .Scaling = DXGI_SCALING_STRETCH,
    .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD, .AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED, .Flags = swap_chain_flags
  };

  ComPtr<IDXGISwapChain1> tmp_swap_chain;
  ThrowIfFailed(dxgi_factory->CreateSwapChainForHwnd(dev.Get(), hwnd.get(), &swap_chain_desc, nullptr, nullptr,
                                                     &tmp_swap_chain));

  ComPtr<IDXGISwapChain2> swap_chain;
  ThrowIfFailed(tmp_swap_chain.As(&swap_chain));

  ComPtr<ID3D11Texture2D> swap_chain_tex;
  ThrowIfFailed(swap_chain->GetBuffer(0, IID_PPV_ARGS(&swap_chain_tex)));

  D3D11_RENDER_TARGET_VIEW_DESC constexpr swap_chain_rtv_desc{
    .Format = swap_chain_format, .ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D, .Texture2D = {.MipSlice = 0}
  };

  ComPtr<ID3D11RenderTargetView> swap_chain_rtv;
  ThrowIfFailed(dev->CreateRenderTargetView(swap_chain_tex.Get(), &swap_chain_rtv_desc, &swap_chain_rtv));

  ShowWindow(hwnd.get(), SW_SHOW);

  int ret;

  while (true) {
    MSG msg;

    if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) {
        ret = static_cast<int>(msg.wParam);
        break;
      }

      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }

    ctx->ClearRenderTargetView(swap_chain_rtv.Get(), std::array{1.0F, 0.0F, 1.0F, 1.0F}.data());

    ThrowIfFailed(swap_chain->Present(0, present_flags));
  }

  return ret;
}
