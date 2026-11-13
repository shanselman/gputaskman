"""Windows notification icon, using existing snapshots only; no GPU probes.

The message loop owns all native resources. The TUI posts updates and remains the
only collector. A pseudoconsole's invisible HWND is never shown or manipulated.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import os
import threading

from .i18n import language, tr
from .llm import gb
from .nvml import format_metric
from .report import intel_memory_note, vram_total_label


def compact_summary(snapshots, paused: bool = False, gpu_index: int | None = None) -> str:
    rows = []
    for i, snap in enumerate(snapshots):
        index = gpu_index if gpu_index is not None else i
        nv = snap.nv
        used = gb(snap.vram_used) if snap.vram_used is not None else "n/a"
        memory = "VRAM≈" if intel_memory_note(snap) else "VRAM"
        temp_kind = f" ({tr(nv.temp_kind)})" if nv and nv.temp_kind else ""
        rows.append(f"GPU {index}: {format_metric(nv.util_gpu if nv else None, unit='%')} · "
                    f"{memory} {used}/{vram_total_label(snap)} · "
                    f"{format_metric(nv.temp if nv else None, unit='°C')}{temp_kind}")
    prefix = tr("[PAUSED] ") if paused else ""
    return prefix + ("\n".join(rows) if rows else tr("Waiting for GPU readings"))


def tooltip_text(value: str) -> str:
    # NOTIFYICONDATAW.szTip holds 128 UTF-16 units, including NUL. Never split
    # a surrogate pair when a GPU name or a future translation contains emoji.
    encoded = value.replace("\0", "").encode("utf-16-le")
    if len(encoded) <= 254:
        return encoded.decode("utf-16-le")
    return encoded[:252].decode("utf-16-le", errors="ignore") + "…"


class WindowsTray:
    UPDATE = 0x8001
    CALLBACK = 0x8002

    def __init__(self, quit_callback, locale: str = "ru", taskbar: bool = True):
        self.quit_callback = quit_callback
        self.locale = locale
        self.error = ""
        self.hwnd = None
        self.panel = None
        self.panel_error = ""
        self.taskbar_enabled = taskbar
        self._thread = None
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._summary = "gputaskman"
        self._title = ""
        self._lock = threading.Lock()

    def start(self) -> bool:
        if os.name != "nt":
            self.error = "Windows only"
            return False
        self._thread = threading.Thread(target=self._run, name="gputaskman-tray", daemon=True)
        self._thread.start()
        if not self._ready.wait(3):
            self.error = "Windows notification area did not respond"
            self._closing.set()
            return False
        return not self.error

    def update(self, summary: str) -> None:
        with self._lock:
            self._summary = summary
        if self.hwnd:
            self.user.PostMessageW(self.hwnd, self.UPDATE, 0, 0)

    def minimize(self) -> bool:
        # GetConsoleWindow under Windows Terminal is a message-only window.
        if not self.hwnd:
            return False
        console = self.kernel.GetConsoleWindow()
        if console and self.user.IsWindowVisible(console):
            self.user.ShowWindow(console, 6)  # SW_MINIMIZE: remains in taskbar
            return True
        return False

    def restore(self) -> bool:
        console = self.kernel.GetConsoleWindow()
        if console and self.user.IsWindowVisible(console):
            self.user.ShowWindow(console, 9)  # SW_RESTORE
            self.user.SetForegroundWindow(console)
            return True
        return False

    def close(self) -> None:
        self._closing.set()
        if self.hwnd:
            self.user.PostMessageW(self.hwnd, 0x0010, 0, 0)  # WM_CLOSE
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        try:
            with language(self.locale):
                self._message_loop()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.hwnd = None
            self._ready.set()

    def _message_loop(self) -> None:
        self.user = u = C.WinDLL("user32", use_last_error=True, winmode=0x800)
        self.kernel = k = C.WinDLL("kernel32", use_last_error=True, winmode=0x800)
        shell = C.WinDLL("shell32", use_last_error=True, winmode=0x800)
        result_type = C.c_ssize_t
        callback_type = C.WINFUNCTYPE(result_type, W.HWND, W.UINT, W.WPARAM, W.LPARAM)

        class WindowClass(C.Structure):
            _fields_ = [("style", W.UINT), ("proc", callback_type), ("class_extra", C.c_int),
                        ("window_extra", C.c_int), ("instance", W.HINSTANCE), ("icon", W.HICON),
                        ("cursor", W.HANDLE), ("background", W.HBRUSH),
                        ("menu", W.LPCWSTR), ("name", W.LPCWSTR)]

        class Guid(C.Structure):
            _fields_ = [("a", W.DWORD), ("b", W.WORD), ("c", W.WORD), ("d", C.c_ubyte * 8)]

        class NotifyData(C.Structure):
            _fields_ = [("size", W.DWORD), ("hwnd", W.HWND), ("id", W.UINT), ("flags", W.UINT),
                        ("callback", W.UINT), ("icon", W.HICON), ("tip", W.WCHAR * 128),
                        ("state", W.DWORD), ("state_mask", W.DWORD), ("info", W.WCHAR * 256),
                        ("version", W.UINT), ("info_title", W.WCHAR * 64), ("info_flags", W.DWORD),
                        ("guid", Guid), ("balloon_icon", W.HICON)]

        # Explicit pointer-sized prototypes are required on Windows x64/ARM64.
        def api(dll, name, result, *args):
            function = getattr(dll, name)
            function.restype, function.argtypes = result, list(args)
            return function

        api(k, "GetModuleHandleW", W.HMODULE, W.LPCWSTR)
        api(k, "GetConsoleWindow", W.HWND)
        api(k, "GetConsoleTitleW", W.DWORD, W.LPWSTR, W.DWORD)
        api(k, "SetConsoleTitleW", W.BOOL, W.LPCWSTR)
        api(u, "RegisterClassW", W.ATOM, C.POINTER(WindowClass))
        api(u, "UnregisterClassW", W.BOOL, W.LPCWSTR, W.HINSTANCE)
        api(u, "RegisterWindowMessageW", W.UINT, W.LPCWSTR)
        api(u, "CreateWindowExW", W.HWND, W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD,
            C.c_int, C.c_int, C.c_int, C.c_int, W.HWND, W.HMENU, W.HINSTANCE, W.LPVOID)
        api(u, "DefWindowProcW", result_type, W.HWND, W.UINT, W.WPARAM, W.LPARAM)
        api(u, "PostMessageW", W.BOOL, W.HWND, W.UINT, W.WPARAM, W.LPARAM)
        api(u, "DestroyWindow", W.BOOL, W.HWND)
        api(u, "PostQuitMessage", None, C.c_int)
        api(u, "GetMessageW", W.BOOL, C.POINTER(W.MSG), W.HWND, W.UINT, W.UINT)
        api(u, "TranslateMessage", W.BOOL, C.POINTER(W.MSG))
        api(u, "DispatchMessageW", result_type, C.POINTER(W.MSG))
        api(u, "IsWindowVisible", W.BOOL, W.HWND)
        api(u, "ShowWindow", W.BOOL, W.HWND, C.c_int)
        api(u, "SetForegroundWindow", W.BOOL, W.HWND)
        api(u, "LoadIconW", W.HICON, W.HINSTANCE, C.c_void_p)
        api(u, "CreatePopupMenu", W.HMENU)
        api(u, "AppendMenuW", W.BOOL, W.HMENU, W.UINT, C.c_size_t, W.LPCWSTR)
        api(u, "TrackPopupMenu", W.UINT, W.HMENU, W.UINT, C.c_int, C.c_int, C.c_int, W.HWND, C.c_void_p)
        api(u, "DestroyMenu", W.BOOL, W.HMENU)
        api(u, "GetCursorPos", W.BOOL, C.POINTER(W.POINT))
        api(u, "SetThreadDpiAwarenessContext", W.HANDLE, W.HANDLE)
        api(shell, "Shell_NotifyIconW", W.BOOL, W.DWORD, C.POINTER(NotifyData))

        instance = k.GetModuleHandleW(None)
        name = f"gputaskman.tray.{os.getpid()}.{id(self)}"
        restart = u.RegisterWindowMessageW("TaskbarCreated")
        data = NotifyData()
        data.size, data.id = C.sizeof(data), 1
        data.flags, data.callback = 1 | 2 | 4 | 128, self.CALLBACK  # message/icon/tip/showtip
        data.icon = u.LoadIconW(None, 32512)  # shared system icon; must not be destroyed
        old_title = C.create_unicode_buffer(32768)
        k.GetConsoleTitleW(old_title, len(old_title))
        self._title = old_title.value

        def notify(operation):
            with self._lock:
                text = self._summary
            data.tip = tooltip_text(text)
            if not shell.Shell_NotifyIconW(operation, C.byref(data)):
                return False
            if operation == 0:
                data.version = 4
                shell.Shell_NotifyIconW(4, C.byref(data))  # NIM_SETVERSION
            return True

        def menu(hwnd):
            popup = u.CreatePopupMenu()
            if not popup:
                return
            try:
                console = k.GetConsoleWindow()
                enabled = bool(console and u.IsWindowVisible(console))
                u.AppendMenuW(popup, 0 if enabled else 1, 1, tr("Open monitor"))
                u.AppendMenuW(popup, 0 if enabled else 1, 2, tr("Minimize"))
                if self.panel:
                    u.AppendMenuW(popup, 8 if self.panel.settings.enabled else 0, 4,
                                  tr("Show readings on taskbar"))
                u.AppendMenuW(popup, 0x800, 0, None)
                u.AppendMenuW(popup, 0, 3, tr("Quit"))
                point = W.POINT()
                u.GetCursorPos(C.byref(point))
                u.SetForegroundWindow(hwnd)
                command = u.TrackPopupMenu(popup, 0x100 | 2, point.x, point.y, 0, hwnd, None)
                if command == 1:
                    self.restore()
                elif command == 2:
                    self.minimize()
                elif command == 3:
                    self.quit_callback()
                elif command == 4 and self.panel:
                    self.panel.toggle()
                u.PostMessageW(hwnd, 0, 0, 0)  # dismiss correctly on the next click
            finally:
                u.DestroyMenu(popup)

        @callback_type
        def window_proc(hwnd, message, wparam, lparam):
            # Exceptions must never escape a ctypes window procedure.
            try:
                if message == self.UPDATE:
                    notify(1)
                    with self._lock:
                        k.SetConsoleTitleW("gputaskman · " + self._summary.replace("\n", " | "))
                        summary = self._summary
                    if self.panel:
                        try:
                            self.panel.update(summary)
                        except Exception as error:
                            self.panel_error = f"{type(error).__name__}: {error}"
                    return 0
                if restart and message == restart:
                    notify(0)
                    if self.panel:
                        self.panel.position()
                    return 0
                if message == self.CALLBACK:
                    event = lparam & 0xffff
                    if event in (0x0203, 0x0400, 0x0401):  # double click / NIN_SELECT / KEYSELECT
                        self.restore()
                    elif event in (0x007B, 0x0205):  # context menu / right button up
                        menu(hwnd)
                    return 0
                if message == 0x0010:
                    u.DestroyWindow(hwnd)
                    return 0
                if message == 0x0002:
                    u.PostQuitMessage(0)
                    return 0
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
            return u.DefWindowProcW(hwnd, message, wparam, lparam)

        wc = WindowClass()
        wc.proc, wc.instance, wc.name = window_proc, instance, name
        if not u.RegisterClassW(C.byref(wc)):
            raise C.WinError(C.get_last_error())
        added = False
        previous_dpi = u.SetThreadDpiAwarenessContext(W.HANDLE(-4))
        try:
            self.hwnd = u.CreateWindowExW(0, name, "gputaskman", 0, 0, 0, 0, 0, None, None, instance, None)
            if not self.hwnd:
                raise C.WinError(C.get_last_error())
            data.hwnd = self.hwnd
            added = notify(0)
            if not added:
                raise OSError("Shell_NotifyIconW: notification area unavailable")
            try:
                from .taskbar import TaskbarPanel
                self.panel = TaskbarPanel(u, k, instance, api, menu, self.restore, self.taskbar_enabled)
                self.panel.start()
                self.panel_error = self.panel.error
            except Exception as error:
                self.panel_error = f"{type(error).__name__}: {error}"
                self.panel = None
            self._ready.set()
            if self._closing.is_set():
                u.PostMessageW(self.hwnd, 0x0010, 0, 0)
            msg = W.MSG()
            while True:
                result = u.GetMessageW(C.byref(msg), None, 0, 0)
                if result == -1:
                    raise C.WinError(C.get_last_error())
                if result == 0:
                    break
                u.TranslateMessage(C.byref(msg))
                u.DispatchMessageW(C.byref(msg))
        finally:
            if self.panel:
                self.panel.close()
                self.panel = None
            if added:
                shell.Shell_NotifyIconW(2, C.byref(data))
            if self.hwnd:
                u.DestroyWindow(self.hwnd)
            u.UnregisterClassW(name, instance)
            k.SetConsoleTitleW(self._title)
            if previous_dpi:
                u.SetThreadDpiAwarenessContext(previous_dpi)
