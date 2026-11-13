"""A small, non-activating Windows window over the primary taskbar's free area.

This is an application-owned overlay, not an Explorer extension. No foreign
window is reparented or resized. Only text from the existing snapshot is drawn.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

from .i18n import tr


@dataclass
class PanelSettings:
    enabled: bool = True
    position: float = 1.0  # fraction of available horizontal space; 1 = next to tray

    @classmethod
    def load(cls, path: Path):
        try:
            with path.open(encoding="utf-8") as stream:
                data = json.loads(stream.read(4096))
            enabled, position = data.get("enabled"), data.get("position")
            if (type(enabled) is bool and type(position) in (int, float)
                    and math.isfinite(position) and 0 <= position <= 1):
                return cls(enabled, float(position))
        except (OSError, ValueError, AttributeError, RecursionError):
            pass
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix="taskbar-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"enabled": self.enabled, "position": self.position}, stream)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def settings_path() -> Path:
    root = Path(os.environ["LOCALAPPDATA"]) if os.environ.get("LOCALAPPDATA") else Path.home() / ".config"
    return root / "gputaskman" / "taskbar.json"


def panel_rect(bar: tuple[int, int, int, int], tray_left: int, text_width: int,
               scale: float = 1, position: float = 1) -> tuple[int, int, int, int] | None:
    """Keep the panel within a horizontal taskbar and outside its notification area.

    The Shell does not reserve space for this overlay. Users can drag it to a
    different free spot if application buttons occupy the default location.
    """
    left, top, right, bottom = bar
    if bottom <= top or right - left <= bottom - top:
        return None
    margin = max(2, round(6 * scale))
    end = min(right, tray_left) - margin
    start = left + margin
    room = end - start
    if room < round(160 * scale):
        return None
    # Reserve room for changing digits so the panel does not jump each time
    # utilization crosses 9/10/100% on an ordinary single-GPU system.
    width = min(max(round(360 * scale), text_width + round(20 * scale)), round(580 * scale), room)
    x = start + round(max(0, min(position, 1)) * (room - width))
    return x, top + margin, width, max(1, bottom - top - margin * 2)


def visible_lines(summary: str) -> list[str]:
    rows = summary.replace("\0", "").splitlines() or [tr("Waiting for GPU readings")]
    if len(rows) > 2:
        return [rows[0], rows[1] + tr(" … (+{0} GPU)").format(len(rows) - 2)]
    return rows


class TaskbarPanel:
    def __init__(self, user, kernel, instance, api, menu, restore, enabled: bool = True):
        self.u, self.k, self.instance = user, kernel, instance
        self.api, self.menu, self.restore = api, menu, restore
        self.path = settings_path()
        self.settings = PanelSettings.load(self.path)
        if not enabled:
            self.settings.enabled = False  # command-line override does not change the saved preference
        self.hwnd = None
        self.error = ""
        self.rows = [tr("Waiting for GPU readings")]
        self.scale = 1.0
        self.font = None
        self._drag = None
        self._rect = None
        self._bar = None
        self._tray_left = 0

    def start(self):
        u, api = self.u, self.api
        self.gdi = g = C.WinDLL("gdi32", use_last_error=True, winmode=0x800)
        api(u, "FindWindowW", W.HWND, W.LPCWSTR, W.LPCWSTR)
        api(u, "FindWindowExW", W.HWND, W.HWND, W.HWND, W.LPCWSTR, W.LPCWSTR)
        api(u, "GetWindowRect", W.BOOL, W.HWND, C.POINTER(W.RECT))
        api(u, "GetClientRect", W.BOOL, W.HWND, C.POINTER(W.RECT))
        api(u, "GetForegroundWindow", W.HWND)
        api(u, "GetClassNameW", C.c_int, W.HWND, W.LPWSTR, C.c_int)
        api(u, "SetWindowTextW", W.BOOL, W.HWND, W.LPCWSTR)
        api(u, "GetDpiForWindow", W.UINT, W.HWND)
        api(u, "SetWindowPos", W.BOOL, W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT)
        api(u, "SetTimer", C.c_size_t, W.HWND, C.c_size_t, W.UINT, C.c_void_p)
        api(u, "KillTimer", W.BOOL, W.HWND, C.c_size_t)
        api(u, "InvalidateRect", W.BOOL, W.HWND, C.c_void_p, W.BOOL)
        api(u, "GetDC", W.HDC, W.HWND)
        api(u, "ReleaseDC", C.c_int, W.HWND, W.HDC)
        api(u, "FillRect", C.c_int, W.HDC, C.POINTER(W.RECT), W.HBRUSH)
        api(u, "DrawTextW", C.c_int, W.HDC, W.LPCWSTR, C.c_int, C.POINTER(W.RECT), W.UINT)
        api(u, "SetCapture", W.HWND, W.HWND)
        api(u, "ReleaseCapture", W.BOOL)
        api(g, "CreateFontW", W.HFONT, *([C.c_int] * 5), *([W.DWORD] * 8), W.LPCWSTR)
        api(g, "SelectObject", W.HANDLE, W.HDC, W.HANDLE)
        api(g, "DeleteObject", W.BOOL, W.HANDLE)
        api(g, "CreateSolidBrush", W.HBRUSH, W.DWORD)
        api(g, "SetTextColor", W.DWORD, W.HDC, W.DWORD)
        api(g, "SetBkMode", C.c_int, W.HDC, C.c_int)
        api(g, "GetTextExtentPoint32W", W.BOOL, W.HDC, W.LPCWSTR, C.c_int, C.POINTER(W.SIZE))
        api(u, "MonitorFromWindow", W.HANDLE, W.HWND, W.DWORD)

        class MonitorInfo(C.Structure):
            _fields_ = [("size", W.DWORD), ("monitor", W.RECT), ("work", W.RECT), ("flags", W.DWORD)]
        self.MonitorInfo = MonitorInfo
        api(u, "GetMonitorInfoW", W.BOOL, W.HANDLE, C.POINTER(MonitorInfo))

        class Paint(C.Structure):
            _fields_ = [("dc", W.HDC), ("erase", W.BOOL), ("rect", W.RECT),
                        ("restore", W.BOOL), ("inc_update", W.BOOL), ("reserved", C.c_byte * 32)]
        self.Paint = Paint
        api(u, "BeginPaint", W.HDC, W.HWND, C.POINTER(Paint))
        api(u, "EndPaint", W.BOOL, W.HWND, C.POINTER(Paint))
        callback_type = C.WINFUNCTYPE(C.c_ssize_t, W.HWND, W.UINT, W.WPARAM, W.LPARAM)

        class WindowClass(C.Structure):
            _fields_ = [("style", W.UINT), ("proc", callback_type), ("class_extra", C.c_int),
                        ("window_extra", C.c_int), ("instance", W.HINSTANCE), ("icon", W.HICON),
                        ("cursor", W.HANDLE), ("background", W.HBRUSH),
                        ("menu", W.LPCWSTR), ("name", W.LPCWSTR)]
        # RegisterClassW is shared with the tray; both class layouts match.
        api(u, "RegisterClassW", W.ATOM, C.c_void_p)
        self._callback = callback_type(self._window_proc)
        self._class_name = f"gputaskman.taskbar.{os.getpid()}.{id(self)}"
        wc = WindowClass()
        wc.style, wc.proc, wc.instance, wc.name = 8, self._callback, self.instance, self._class_name
        if not u.RegisterClassW(C.byref(wc)):
            raise C.WinError(C.get_last_error())
        try:
            # TOPMOST | TOOLWINDOW | NOACTIVATE; POPUP. No extra taskbar button.
            self.hwnd = u.CreateWindowExW(0x08000088, self._class_name, "gputaskman — панель показателей",
                                         0x80000000, 0, 0, 1, 1, None, None, self.instance, None)
            if not self.hwnd:
                raise C.WinError(C.get_last_error())
            if not u.SetTimer(self.hwnd, 1, 1000, None):
                raise C.WinError(C.get_last_error())
            self.position()
        except Exception:
            self.close()
            raise

    def _read_rect(self, hwnd):
        rect = W.RECT()
        if hwnd and self.u.GetWindowRect(hwnd, C.byref(rect)):
            return rect.left, rect.top, rect.right, rect.bottom
        return None

    def _set_font(self, scale):
        if self.font and scale == self.scale:
            return
        font = self.gdi.CreateFontW(-round(13 * scale), 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")
        if not font:
            raise C.WinError(C.get_last_error())
        if self.font:
            self.gdi.DeleteObject(self.font)
        self.font, self.scale = font, scale

    def _text_width(self):
        dc = self.u.GetDC(self.hwnd)
        if not dc:
            return round(360 * self.scale)
        old = self.gdi.SelectObject(dc, self.font)
        try:
            widths = []
            for row in self.rows:
                size = W.SIZE()
                if self.gdi.GetTextExtentPoint32W(dc, row, len(row.encode("utf-16-le")) // 2, C.byref(size)):
                    widths.append(size.cx)
            return max(widths, default=round(360 * self.scale))
        finally:
            self.gdi.SelectObject(dc, old)
            self.u.ReleaseDC(self.hwnd, dc)

    def _fullscreen(self, bar_hwnd):
        info = self.MonitorInfo()
        info.size = C.sizeof(info)
        monitor = self.u.MonitorFromWindow(bar_hwnd, 2)
        if not self.u.GetMonitorInfoW(monitor, C.byref(info)):
            return False
        foreground = self.u.GetForegroundWindow()
        if foreground in (self.hwnd, bar_hwnd):
            return False
        bounds = self._read_rect(foreground)
        m = info.monitor
        # Desktop windows cover the monitor too; distinguish them by class.
        if bounds and bounds[0] <= m.left and bounds[1] <= m.top and bounds[2] >= m.right and bounds[3] >= m.bottom:
            name = C.create_unicode_buffer(128)
            self.u.GetClassNameW(foreground, name, len(name))
            return name.value not in ("Progman", "WorkerW", "Shell_TrayWnd")
        return False

    def position(self):
        if not self.hwnd:
            return
        u = self.u
        bar_hwnd = u.FindWindowW("Shell_TrayWnd", None)
        bar = self._read_rect(bar_hwnd)
        if not self.settings.enabled or not bar or not u.IsWindowVisible(bar_hwnd) or self._fullscreen(bar_hwnd):
            u.ShowWindow(self.hwnd, 0)
            return
        self._set_font((u.GetDpiForWindow(bar_hwnd) or 96) / 96)
        info = self.MonitorInfo()
        info.size = C.sizeof(info)
        if u.GetMonitorInfoW(u.MonitorFromWindow(bar_hwnd, 2), C.byref(info)):
            visible_height = min(bar[3], info.monitor.bottom) - max(bar[1], info.monitor.top)
            if visible_height < round(10 * self.scale):
                u.ShowWindow(self.hwnd, 0)
                return
        tray = self._read_rect(u.FindWindowExW(bar_hwnd, None, "TrayNotifyWnd", None))
        # The fallback leaves clock/language/system icons unobscured.
        tray_left = tray[0] if tray and bar[0] < tray[0] < bar[2] else bar[2] - round(260 * self.scale)
        rect = panel_rect(bar, tray_left, self._text_width(), self.scale, self.settings.position)
        if rect is None or bar[3] - bar[1] < round(20 * self.scale):
            self.error = tr("No room for readings on the primary horizontal taskbar")
            u.ShowWindow(self.hwnd, 0)
            return
        if self.error == tr("No room for readings on the primary horizontal taskbar"):
            self.error = ""
        self._bar, self._tray_left, self._rect = bar, tray_left, rect
        if not u.SetWindowPos(self.hwnd, W.HWND(-1), *rect, 0x0010 | 0x0040):
            raise C.WinError(C.get_last_error())

    def update(self, summary):
        self.rows = visible_lines(summary)
        self.u.SetWindowTextW(self.hwnd, "gputaskman — " + summary.replace("\n", " | "))
        self.position()
        self.u.InvalidateRect(self.hwnd, None, False)

    def toggle(self):
        self.settings.enabled = not self.settings.enabled
        self._save()
        self.position()

    def _save(self):
        try:
            self.settings.save(self.path)
        except OSError as error:
            self.error = tr("Could not save taskbar position: {0}").format(error)

    def _paint(self, hwnd):
        paint = self.Paint()
        dc = self.u.BeginPaint(hwnd, C.byref(paint))
        if not dc:
            return
        brush, old = None, None
        try:
            # Use the system's taskbar theme, without changing any Windows setting.
            import winreg
            light = False
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
                    light = bool(winreg.QueryValueEx(key, "SystemUsesLightTheme")[0])
            except OSError:
                pass
            rect = W.RECT()
            self.u.GetClientRect(hwnd, C.byref(rect))
            brush = self.gdi.CreateSolidBrush(0xF2F2F2 if light else 0x202020)
            self.u.FillRect(dc, C.byref(rect), brush)
            old = self.gdi.SelectObject(dc, self.font)
            self.gdi.SetBkMode(dc, 1)
            self.gdi.SetTextColor(dc, 0x202020 if light else 0xF5F5F5)
            row_height = max(1, (rect.bottom - rect.top) // len(self.rows))
            for i, row in enumerate(self.rows):
                line_rect = W.RECT(round(10 * self.scale), i * row_height,
                                   rect.right - round(10 * self.scale), (i + 1) * row_height)
                self.u.DrawTextW(dc, row, -1, C.byref(line_rect), 0x20 | 0x4 | 0x8000 | 0x800)
        finally:
            if old:
                self.gdi.SelectObject(dc, old)
            if brush:
                self.gdi.DeleteObject(brush)
            self.u.EndPaint(hwnd, C.byref(paint))

    def _window_proc(self, hwnd, message, wparam, lparam):
        try:
            if message == 0x000F:  # WM_PAINT
                self._paint(hwnd)
                return 0
            if message == 0x0014:  # WM_ERASEBKGND; paint fills the entire client
                return 1
            if message == 0x0021:  # WM_MOUSEACTIVATE
                return 3  # MA_NOACTIVATE
            if message in (0x0113, 0x007E, 0x001A, 0x02E0):  # timer/display/settings/DPI
                self.position()
                self.u.InvalidateRect(hwnd, None, False)
                return 0
            if message == 0x0201 and self._rect:  # left down: drag without activating
                # Use the event's client coordinate: the cursor may already
                # have moved again by the time this queued message is handled.
                self._drag = C.c_short(lparam & 0xffff).value
                self.u.SetCapture(hwnd)
                return 0
            if message == 0x0200 and self._drag is not None and self._bar:
                point = W.POINT()
                self.u.GetCursorPos(C.byref(point))
                margin = max(2, round(6 * self.scale))
                start = self._bar[0] + margin
                room = self._tray_left - margin - start - self._rect[2]
                self.settings.position = max(0, min(1, (point.x - self._drag - start) / max(1, room)))
                self.position()
                return 0
            if message in (0x0202, 0x0215) and self._drag is not None:  # up / capture lost
                self._drag = None
                self.u.ReleaseCapture()
                self._save()
                return 0
            if message == 0x0203:
                self.restore()
                return 0
            if message in (0x0205, 0x007B):
                self.menu(hwnd)
                return 0
            if message == 0x0010:
                self.settings.enabled = False
                self._save()
                self.position()
                return 0
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        return self.u.DefWindowProcW(hwnd, message, wparam, lparam)

    def close(self):
        if self.hwnd:
            self.u.KillTimer(self.hwnd, 1)
            self.u.DestroyWindow(self.hwnd)
            self.hwnd = None
        if self.font:
            self.gdi.DeleteObject(self.font)
            self.font = None
        if getattr(self, "_class_name", None):
            self.u.UnregisterClassW(self._class_name, self.instance)
