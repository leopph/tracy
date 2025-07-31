#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

import std;

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
  }

  return ret;
}
