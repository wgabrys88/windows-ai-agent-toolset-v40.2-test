"""
Franz region selector — visible topmost overlay to select CAPTURE_CROP.

Windows 11 + Python 3.13 + stdlib/ctypes only.

Controls:
- Left mouse drag: select region
- ESC or right click: cancel
- Ctrl+Shift+Q: emergency quit (global hotkey)
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as W
import sys


# ----- Robust typedefs -----
HCURSOR = getattr(W, "HCURSOR", W.HANDLE)
HICON = getattr(W, "HICON", W.HANDLE)
HMODULE = getattr(W, "HMODULE", W.HANDLE)

LRESULT = ctypes.c_ssize_t  # pointer-sized on Win64
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, W.HWND, W.UINT, W.WPARAM, W.LPARAM)

u32 = ctypes.WinDLL("user32", use_last_error=True)
g32 = ctypes.WinDLL("gdi32", use_last_error=True)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Best-effort DPI awareness
try:
    ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
except Exception:
    try:
        u32.SetProcessDPIAware()
    except Exception:
        pass


# ----- Constants -----
WS_EX_LAYERED = 0x00080000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080  # keep it out of Alt-Tab; remove if you prefer Alt-Tab visibility
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000

LWA_ALPHA = 0x00000002

WM_PAINT = 0x000F
WM_ERASEBKGND = 0x0014
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEMOVE = 0x0200
WM_RBUTTONDOWN = 0x0204
WM_KEYDOWN = 0x0100
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
WM_HOTKEY = 0x0312

VK_ESCAPE = 0x1B
IDC_CROSS = 32515

CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001

SM_CXSCREEN = 0
SM_CYSCREEN = 1

PS_SOLID = 0
PS_DASH = 1
TRANSPARENT = 1
NULL_BRUSH = 5  # stock object id

MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
HOTKEY_ID_QUIT = 1


# ----- Structures -----
class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", W.HDC),
        ("fErase", W.BOOL),
        ("rcPaint", W.RECT),
        ("fRestore", W.BOOL),
        ("fIncUpdate", W.BOOL),
        ("rgbReserved", W.BYTE * 32),
    ]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", W.UINT),
        ("style", W.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", W.HINSTANCE),
        ("hIcon", HICON),
        ("hCursor", HCURSOR),
        ("hbrBackground", W.HBRUSH),
        ("lpszMenuName", W.LPCWSTR),
        ("lpszClassName", W.LPCWSTR),
        ("hIconSm", HICON),
    ]


# ----- Prototypes -----
u32.GetSystemMetrics.argtypes = [ctypes.c_int]
u32.GetSystemMetrics.restype = ctypes.c_int

k32.GetModuleHandleW.argtypes = [W.LPCWSTR]
k32.GetModuleHandleW.restype = HMODULE

u32.LoadCursorW.argtypes = [W.HINSTANCE, W.LPCWSTR]
u32.LoadCursorW.restype = HCURSOR

u32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
u32.RegisterClassExW.restype = W.ATOM

u32.CreateWindowExW.argtypes = [
    W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    W.HWND, W.HMENU, W.HINSTANCE, W.LPVOID
]
u32.CreateWindowExW.restype = W.HWND

u32.SetLayeredWindowAttributes.argtypes = [W.HWND, W.DWORD, W.BYTE, W.DWORD]
u32.SetLayeredWindowAttributes.restype = W.BOOL

u32.DefWindowProcW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
u32.DefWindowProcW.restype = LRESULT

u32.BeginPaint.argtypes = [W.HWND, ctypes.POINTER(PAINTSTRUCT)]
u32.BeginPaint.restype = W.HDC
u32.EndPaint.argtypes = [W.HWND, ctypes.POINTER(PAINTSTRUCT)]
u32.EndPaint.restype = W.BOOL

u32.InvalidateRect.argtypes = [W.HWND, ctypes.c_void_p, W.BOOL]
u32.InvalidateRect.restype = W.BOOL

u32.DestroyWindow.argtypes = [W.HWND]
u32.DestroyWindow.restype = W.BOOL

u32.PostQuitMessage.argtypes = [ctypes.c_int]
u32.PostQuitMessage.restype = None

u32.GetMessageW.argtypes = [ctypes.POINTER(W.MSG), W.HWND, W.UINT, W.UINT]
u32.GetMessageW.restype = ctypes.c_int  # IMPORTANT: can be -1

u32.TranslateMessage.argtypes = [ctypes.POINTER(W.MSG)]
u32.TranslateMessage.restype = W.BOOL

u32.DispatchMessageW.argtypes = [ctypes.POINTER(W.MSG)]
u32.DispatchMessageW.restype = LRESULT

u32.SetCapture.argtypes = [W.HWND]
u32.SetCapture.restype = W.HWND
u32.ReleaseCapture.argtypes = []
u32.ReleaseCapture.restype = W.BOOL

u32.SetForegroundWindow.argtypes = [W.HWND]
u32.SetForegroundWindow.restype = W.BOOL
u32.SetFocus.argtypes = [W.HWND]
u32.SetFocus.restype = W.HWND

u32.RegisterHotKey.argtypes = [W.HWND, ctypes.c_int, W.UINT, W.UINT]
u32.RegisterHotKey.restype = W.BOOL
u32.UnregisterHotKey.argtypes = [W.HWND, ctypes.c_int]
u32.UnregisterHotKey.restype = W.BOOL

g32.CreateSolidBrush.argtypes = [W.DWORD]
g32.CreateSolidBrush.restype = W.HBRUSH
g32.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, W.DWORD]
g32.CreatePen.restype = W.HGDIOBJ
g32.SelectObject.argtypes = [W.HDC, W.HGDIOBJ]
g32.SelectObject.restype = W.HGDIOBJ
g32.DeleteObject.argtypes = [W.HGDIOBJ]
g32.DeleteObject.restype = W.BOOL
g32.Rectangle.argtypes = [W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
g32.Rectangle.restype = W.BOOL
g32.SetBkMode.argtypes = [W.HDC, ctypes.c_int]
g32.SetBkMode.restype = ctypes.c_int
g32.GetStockObject.argtypes = [ctypes.c_int]
g32.GetStockObject.restype = W.HGDIOBJ


# ----- Helpers / state -----
def _last_error_text() -> str:
    err = ctypes.get_last_error()
    try:
        msg = ctypes.WinError(err).strerror
    except Exception:
        msg = "unknown"
    return f"{err} ({msg})"


def _get_xy(lp: int) -> tuple[int, int]:
    x = lp & 0xFFFF
    y = (lp >> 16) & 0xFFFF
    if x > 32767:
        x -= 65536
    if y > 32767:
        y -= 65536
    return x, y


sw = u32.GetSystemMetrics(SM_CXSCREEN)
sh = u32.GetSystemMetrics(SM_CYSCREEN)
if sw <= 0 or sh <= 0:
    sw, sh = 1920, 1080

NULL_BRUSH_OBJ = g32.GetStockObject(NULL_BRUSH)

dragging = False
sx = sy = ex = ey = 0
result_rect: tuple[int, int, int, int] | None = None
_hwnd: W.HWND | None = None

# IMPORTANT: keep the callback object alive (prevents GC -> dead window)
_WNDPROC_REF = None  # set in run()


def wndproc(hwnd: W.HWND, msg: int, wp: int, lp: int) -> int:
    global dragging, sx, sy, ex, ey, result_rect, _hwnd

    if msg == WM_ERASEBKGND:
        return 1

    if msg == WM_HOTKEY:
        # Ctrl+Shift+Q = emergency quit
        if int(wp) == HOTKEY_ID_QUIT:
            result_rect = None
            u32.DestroyWindow(hwnd)
            return 0

    if msg == WM_KEYDOWN:
        if int(wp) == VK_ESCAPE:
            result_rect = None
            u32.DestroyWindow(hwnd)
            return 0

    if msg == WM_RBUTTONDOWN:
        result_rect = None
        u32.DestroyWindow(hwnd)
        return 0

    if msg == WM_CLOSE:
        result_rect = None
        u32.DestroyWindow(hwnd)
        return 0

    if msg == WM_LBUTTONDOWN:
        sx, sy = _get_xy(lp)
        ex, ey = sx, sy
        dragging = True
        u32.SetCapture(hwnd)
        u32.InvalidateRect(hwnd, None, True)
        return 0

    if msg == WM_MOUSEMOVE:
        if dragging:
            ex, ey = _get_xy(lp)
            u32.InvalidateRect(hwnd, None, True)
        return 0

    if msg == WM_LBUTTONUP:
        if dragging:
            ex, ey = _get_xy(lp)
            dragging = False
            u32.ReleaseCapture()
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                result_rect = (x1, y1, x2, y2)
                u32.DestroyWindow(hwnd)
            else:
                u32.InvalidateRect(hwnd, None, True)
        return 0

    if msg == WM_PAINT:
        ps = PAINTSTRUCT()
        hdc = u32.BeginPaint(hwnd, ctypes.byref(ps))

        # Dim overlay background (solid black; window alpha makes it translucent)
        brush_bg = g32.CreateSolidBrush(0x00000000)  # black
        rc = W.RECT(0, 0, sw, sh)
        u32.FillRect(hdc, ctypes.byref(rc), brush_bg)
        g32.DeleteObject(brush_bg)

        # Draw selection rectangle (hollow)
        if dragging or (ex != sx or ey != sy):
            x1, y1 = min(sx, ex), min(sy, ey)
            x2, y2 = max(sx, ex), max(sy, ey)

            pen_white = g32.CreatePen(PS_SOLID, 3, 0x00FFFFFF)
            pen_green = g32.CreatePen(PS_DASH, 1, 0x0000FF00)

            old_pen = g32.SelectObject(hdc, pen_white)
            old_brush = g32.SelectObject(hdc, NULL_BRUSH_OBJ)
            g32.SetBkMode(hdc, TRANSPARENT)

            g32.Rectangle(hdc, x1, y1, x2, y2)
            g32.SelectObject(hdc, pen_green)
            g32.Rectangle(hdc, x1 - 2, y1 - 2, x2 + 2, y2 + 2)

            g32.SelectObject(hdc, old_pen)
            g32.SelectObject(hdc, old_brush)

            g32.DeleteObject(pen_white)
            g32.DeleteObject(pen_green)

        u32.EndPaint(hwnd, ctypes.byref(ps))
        return 0

    if msg == WM_DESTROY:
        # best-effort cleanup
        try:
            u32.UnregisterHotKey(hwnd, HOTKEY_ID_QUIT)
        except Exception:
            pass
        u32.PostQuitMessage(0)
        return 0

    return int(u32.DefWindowProcW(hwnd, msg, wp, lp))


def run() -> int:
    global _WNDPROC_REF, _hwnd, result_rect

    print("Franz Region Selector")
    print("- Drag with LEFT mouse button to select a rectangle")
    print("- ESC or right click cancels")
    print("- Ctrl+Shift+Q emergency quit (global hotkey)")
    print()

    hinst = k32.GetModuleHandleW(None)
    cls_name = "FranzSelector"

    # Keep WNDPROC alive globally
    _WNDPROC_REF = WNDPROC(wndproc)

    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.style = CS_HREDRAW | CS_VREDRAW
    wc.lpfnWndProc = _WNDPROC_REF
    wc.cbClsExtra = 0
    wc.cbWndExtra = 0
    wc.hInstance = hinst
    wc.hIcon = 0
    wc.hCursor = u32.LoadCursorW(None, ctypes.cast(IDC_CROSS, W.LPCWSTR))
    wc.hbrBackground = 0
    wc.lpszMenuName = None
    wc.lpszClassName = cls_name
    wc.hIconSm = 0

    atom = u32.RegisterClassExW(ctypes.byref(wc))
    if not atom:
        err = ctypes.get_last_error()
        # ERROR_CLASS_ALREADY_EXISTS (1410) can happen if you run twice in same process
        if err != 1410:
            print(f"ERR: RegisterClassExW failed: {_last_error_text()}", file=sys.stderr)
            return 1

    ex_style = WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
    hwnd = u32.CreateWindowExW(
        ex_style,
        cls_name,
        "Franz Region Select",
        WS_POPUP | WS_VISIBLE,
        0, 0, sw, sh,
        None, None, hinst, None,
    )
    if not hwnd:
        print(f"ERR: CreateWindowExW failed: {_last_error_text()}", file=sys.stderr)
        return 1

    _hwnd = hwnd

    # Visible dim overlay: entire window alpha (0..255). Lower = more transparent.
    alpha = 90
    if not u32.SetLayeredWindowAttributes(hwnd, 0, alpha, LWA_ALPHA):
        print(f"ERR: SetLayeredWindowAttributes failed: {_last_error_text()}", file=sys.stderr)
        return 1

    # Make sure we can receive keyboard input (ESC)
    u32.SetForegroundWindow(hwnd)
    u32.SetFocus(hwnd)

    # Register global hotkey: Ctrl+Shift+Q
    if not u32.RegisterHotKey(hwnd, HOTKEY_ID_QUIT, MOD_CONTROL | MOD_SHIFT, ord("Q")):
        # Not fatal, but warn
        print(f"WARN: RegisterHotKey failed: {_last_error_text()}", file=sys.stderr)

    # Message loop
    msg = W.MSG()
    while True:
        r = u32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if r == -1:
            print(f"ERR: GetMessageW failed: {_last_error_text()}", file=sys.stderr)
            return 1
        if r == 0:
            break
        u32.TranslateMessage(ctypes.byref(msg))
        u32.DispatchMessageW(ctypes.byref(msg))

    if result_rect:
        px1, py1, px2, py2 = result_rect

        nx1 = max(0, min(1000, round(px1 * 1000 / sw)))
        ny1 = max(0, min(1000, round(py1 * 1000 / sh)))
        nx2 = max(0, min(1000, round(px2 * 1000 / sw)))
        ny2 = max(0, min(1000, round(py2 * 1000 / sh)))

        print()
        print(f'CAPTURE_CROP = {{"x1": {nx1}, "y1": {ny1}, "x2": {nx2}, "y2": {ny2}}}')
        return 0

    print("Selection cancelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())