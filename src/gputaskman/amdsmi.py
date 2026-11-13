"""Optional read-only AMD SMI telemetry. WDDM collection works without this module."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib
import io
import math
import os
import re
import sys
import threading

from .dxgi import Adapter, VENDOR_AMD
from .telemetry import DeviceTelemetry

_ROCM_AMDSMI_PATHS = ("/opt/rocm-wsl/share/amd_smi", "/opt/rocm/share/amd_smi")
_WSL_SETUP = ("Check AMD's supported GPU/Windows-driver matrix, then install matching WSL AMD SMI "
              "(rocdxg-amd-smi-lib) and ROCDXG runtime (rocdxg-roct) packages yourself; "
              "native Linux ROCm/AMD SMI alone is not sufficient. "
              "See https://github.com/ROCm/librocdxg#readme. gputaskman never installs or changes them.")
_discovery_status = ""
_sessions: dict[int, int] = {}
_session_lock = threading.Lock()


def amdsmi_status() -> str:
    return _discovery_status


def set_discovery_status(reason: str) -> None:
    global _discovery_status
    _discovery_status = f"AMD SMI {reason}. {_WSL_SETUP}" if reason else ""


def _acquire(api) -> None:
    with _session_lock:
        key = id(api)
        if key not in _sessions:
            api.amdsmi_init()
            try:
                version = getattr(api, "__version__", "")
                get_version = getattr(api, "amdsmi_get_lib_version", None)
                library = get_version() if get_version is not None else None
                if (isinstance(version, str) and re.match(r"^\d+\.", version)
                        and isinstance(library, dict) and isinstance(library.get("major"), int)
                        and int(version.split(".")[0]) != library["major"]):
                    raise RuntimeError("AMD SMI Python wrapper and loaded library have different major versions "
                                       f"(wrapper {version}, library major {library['major']})")
            except Exception:
                api.amdsmi_shut_down()
                raise
        _sessions[key] = _sessions.get(key, 0) + 1


def _release(api) -> None:
    with _session_lock:
        key = id(api)
        count = _sessions[key]
        if count == 1:
            del _sessions[key]
            api.amdsmi_shut_down()
        else:
            _sessions[key] = count - 1


@contextmanager
def amdsmi_session(api):
    """Discovery and collectors share the library's process-wide initialization."""
    _acquire(api)
    try:
        yield
    finally:
        _release(api)


def _import_amdsmi():
    for p in reversed(_ROCM_AMDSMI_PATHS):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        # Some vendor wrappers print loader errors to stdout, which would corrupt --json.
        # Let the wrapper load its own library; never preload one from a different installation.
        with redirect_stdout(io.StringIO()):
            return importlib.import_module("amdsmi")
    except Exception as error:
        set_discovery_status(f"unavailable ({type(error).__name__}: {error})")
        return None


def _read_vram_bytes(api, handle, *, wsl: bool = False) -> tuple[int | None, int | None]:
    """Read (total_bytes, used_bytes). Prefer amdsmi_get_gpu_memory_total/usage (used by amd-smi CLI);
    amdsmi_get_gpu_vram_usage in rocm-wsl leaves vram_used uninitialized (e.g. >400 GB)."""
    total = used = None
    try:
        mtype = api.AmdSmiMemoryType.VRAM
        t = api.amdsmi_get_gpu_memory_total(handle, mtype)
        if _number(t) is not None and t > 0:
            total = int(t)
        u = api.amdsmi_get_gpu_memory_usage(handle, mtype)
        if _number(u) is not None and u >= 0 and (total is None or u <= total):
            used = int(u)
    except Exception:
        pass
    if not wsl and (total is None or used is None):
        try:
            vram = api.amdsmi_get_gpu_vram_usage(handle)
            if isinstance(vram, dict):
                vt = _number(vram.get("vram_total"))
                vu = _number(vram.get("vram_used"))
                if total is None and vt is not None and vt > 0:
                    total = int(vt * (1 << 20))
                if used is None and vu is not None and vu >= 0 and total is not None and vu * (1 << 20) <= total:
                    used = int(vu * (1 << 20))
        except Exception:
            pass
    if total is not None and used is not None and used > total:
        used = None
    return total, used


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _bdf(value):
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)", str(value))
    if not match:
        return None
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    # WDDM's ADAPTERADDRESS has no domain. Refuse nonzero domains rather than guess.
    return (bus, device, function) if domain == 0 else None


class AmdSmi:
    def __init__(self, adapter: Adapter | None, adapters: list[Adapter], api=None):
        self.api = None
        self.handle = None
        self.driver = ""
        self.bdf = ""
        self.name = adapter.name if adapter else "?"
        self.initialized = False
        self.wsl = os.path.exists("/dev/dxg")
        self.status = ""
        if not adapter or adapter.vendor_id != VENDOR_AMD or adapter.pci_address is None:
            return
        if sum(a.pci_address == adapter.pci_address for a in adapters if not a.software) != 1:
            return
        try:
            self.api = api or _import_amdsmi()
            if self.api is None:
                self.status = amdsmi_status()
                return
            _acquire(self.api)
            self.initialized = True
            matches = [h for h in self.api.amdsmi_get_processor_handles()
                       if _bdf(self._call("amdsmi_get_gpu_device_bdf", h)) == adapter.pci_address]
            if len(matches) == 1:
                self.handle = matches[0]
                self.bdf = str(self._call("amdsmi_get_gpu_device_bdf", self.handle) or "")
                info = self._call("amdsmi_get_gpu_driver_info", self.handle) or {}
                if isinstance(info, dict):
                    self.driver = str(info.get("driver_version") or "")
            else:
                self.status = "AMD SMI could not uniquely match the selected GPU's PCI address."
        except Exception as error:
            self.status = f"AMD SMI initialization failed ({type(error).__name__}: {error})."
            if self.wsl:
                self.status += " " + _WSL_SETUP
            self.close()

    def _call(self, name, *args):
        try:
            return getattr(self.api, name)(*args)
        except Exception:
            return None

    def sample(self) -> DeviceTelemetry | None:
        if self.handle is None:
            return None
        notes = []
        d = DeviceTelemetry(index=0, name=self.name, pci_bus=self.bdf, source="AMD SMI")
        d.mem_total, d.mem_used = _read_vram_bytes(self.api, self.handle, wsl=self.wsl)
        if d.mem_used is None:
            notes.append("AMD SMI VRAM usage is unavailable or outside the device total; shown as unknown.")
        activity = self._call("amdsmi_get_gpu_activity", self.handle)
        if isinstance(activity, dict):
            d.util_gpu = _number(activity.get("gfx_activity"))
            d.util_mem = _number(activity.get("umc_activity"))
        power = self._call("amdsmi_get_power_info", self.handle)
        if isinstance(power, dict):
            p = _number(power.get("socket_power"))
            if p is None:
                p = _number(power.get("current_socket_power"))
            if p is None:
                p = _number(power.get("average_socket_power"))
            d.power_w = p
            pl = _number(power.get("power_limit"))
            if pl is not None and pl > 0 and pl not in ((1 << 32) - 1, (1 << 64) - 1):
                d.power_limit_w = pl
        if d.power_w is None or d.util_gpu is None:
            metrics = self._call("amdsmi_get_gpu_metrics_info", self.handle)
            if isinstance(metrics, dict):
                if d.power_w is None:
                    p = _number(metrics.get("current_socket_power"))
                    if p is None:
                        p = _number(metrics.get("average_socket_power"))
                    d.power_w = p
                if d.util_gpu is None:
                    d.util_gpu = _number(metrics.get("average_gfx_activity"))
                if d.util_mem is None:
                    d.util_mem = _number(metrics.get("average_umc_activity"))
        if d.power_limit_w is None:
            cap = self._call("amdsmi_get_power_cap_info", self.handle, 0)
            if isinstance(cap, dict):
                pc = _number(cap.get("power_cap"))
                if pc is not None and pc > 0 and pc not in ((1 << 32) - 1, (1 << 64) - 1):
                    d.power_limit_w = pc if sys.platform == "win32" else pc / 1_000_000
        if d.power_limit_w is None:
            notes.append("AMD SMI power limit is unavailable.")
        d.power_kind = "socket" if d.power_w is not None else ""
        try:
            clk = self._call("amdsmi_get_clock_info", self.handle, self.api.AmdSmiClkType.GFX)
            if isinstance(clk, dict):
                c = _number(clk.get("clk"))
                m = _number(clk.get("max_clk"))
                if c is not None:
                    d.gfx_clock = int(c)
                if m is not None:
                    d.sm_clock_max = int(m)
            mclk = self._call("amdsmi_get_clock_info", self.handle, self.api.AmdSmiClkType.MEM)
            if isinstance(mclk, dict):
                mc = _number(mclk.get("clk"))
                if mc is not None:
                    d.mem_clock = int(mc)
            vclk = self._call("amdsmi_get_clock_info", self.handle, self.api.AmdSmiClkType.VCLK0)
            if isinstance(vclk, dict):
                vc = _number(vclk.get("clk"))
                if vc is not None:
                    d.video_clock = int(vc)
        except AttributeError:
            pass
        try:
            temp = self._call("amdsmi_get_temp_metric", self.handle,
                              self.api.AmdSmiTemperatureType.EDGE, self.api.AmdSmiTemperatureMetric.CURRENT)
            value = _number(temp)
            critical = getattr(self.api.AmdSmiTemperatureMetric, "CRITICAL", None)
            limit = (self._call("amdsmi_get_temp_metric", self.handle,
                                self.api.AmdSmiTemperatureType.EDGE, critical)
                     if critical is not None else None)
            # AMD's CLI treats an edge critical limit of zero as an unsupported sensor.
            d.temp = value if limit != 0 else None
            if limit == 0:
                notes.append("AMD SMI edge temperature is unsupported (critical limit is zero).")
        except AttributeError:
            pass
        self.status = " ".join(notes)
        return d

    def close(self):
        self.handle = None
        if self.initialized:
            self.initialized = False
            try:
                _release(self.api)
            except Exception as error:
                self.status = f"AMD SMI shutdown failed ({type(error).__name__}: {error})."
