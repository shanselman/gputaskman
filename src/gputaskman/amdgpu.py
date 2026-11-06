"""Read-only Linux amdgpu sysfs telemetry and DRM fdinfo clients (no ROCm dependency).

Device totals are measurements. Per-client DRM memory may include shared buffers;
it is deliberately not treated as additive per-process VRAM here.
"""

from __future__ import annotations

import os
import platform
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .dxgi import Adapter, VENDOR_AMD
from .pdh import AdapterGpu, ProcGpu
from .telemetry import DeviceTelemetry

DRM_ROOT = Path("/sys/class/drm")
PROC_ROOT = Path("/proc")
_PCI = re.compile(r"[0-9a-fA-F]+:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")


def _quantity(value: str, units: dict[str, int]) -> int | None:
    parts = value.split()
    if (not parts or len(parts) > 2 or len(parts[0]) > 20
            or not parts[0].isascii() or not parts[0].isdecimal()):
        return None
    scale = units.get(parts[1] if len(parts) == 2 else "")
    return int(parts[0]) * scale if scale is not None else None


@dataclass
class DrmClient:
    device: int
    client: int
    vram: int | None
    gtt: int | None
    engines: dict[str, int]
    capacities: dict[str, int]
    bad_engines: bool = False


def parse_fdinfo(text: str) -> DrmClient | None:
    from .linux import pci_key

    fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    bus, client = fields.get("drm-pdev", ""), fields.get("drm-client-id", "")
    if fields.get("drm-driver") != "amdgpu" or not _PCI.fullmatch(bus):
        return None
    if len(client) > 20 or not client.isascii() or not client.isdecimal():
        return None  # no safe deduplication without an adapter and client identifier
    units = {"": 1, "KiB": 1024, "MiB": 1 << 20}
    def resident(region):
        return _quantity(fields.get(f"drm-resident-{region}", fields.get(f"drm-memory-{region}", "")), units)
    engines, capacities, bad_engines = {}, {}, False
    for key, value in fields.items():
        if key.startswith("drm-engine-capacity-"):
            capacity = _quantity(value, {"": 1})
            if capacity is not None:
                capacities[key.removeprefix("drm-engine-capacity-")] = capacity
            else:
                bad_engines = True
        elif key.startswith("drm-engine-"):
            busy = _quantity(value, {"ns": 1})
            if busy is not None:
                engines[key.removeprefix("drm-engine-")] = busy
            else:
                bad_engines = True
    return DrmClient(pci_key(bus), int(client), resident("vram"), resident("gtt"), engines, capacities, bad_engines)


def _engine(name: str) -> str:
    for prefix, label in (("gfx", "3d"), ("compute", "compute"), ("dma", "copy"),
                          ("enc", "encode"), ("dec", "decode"), ("jpeg", "decode")):
        if name.startswith(prefix):
            return label
    return name


def _start_time(pid: int) -> str:
    # PID reuse must not turn a previous process's engine time into this one's rate.
    stat = _text(PROC_ROOT / str(pid) / "stat")
    try:
        return stat.rsplit(")", 1)[1].split()[19]
    except IndexError:
        return ""


class DrmProcesses:
    """Local PID handles, deduplicated by (PCI address, DRM client id).

    Resident counts can repeat shared buffers across distinct clients. Shared
    client handles seen in multiple PIDs have no unambiguous per-PID counters.
    """

    def __init__(self, device_keys):
        self.device_keys = set(device_keys)
        self.previous = {}
        self.known_engines = defaultdict(set)
        self.cursor = 0
        self.status = "not sampled"

    def sample(self, budget: float | None = 0.05) -> dict[int, dict[int, ProcGpu]]:
        if not self.device_keys:
            self.status = "no amdgpu devices"
            return {}
        deadline = None if budget is None else time.monotonic() + budget
        try:
            with os.scandir(PROC_ROOT) as entries:
                pids = sorted(int(p.name) for p in entries if p.name.isdecimal())
        except OSError:
            self.status = "unavailable (/proc is inaccessible)"
            self.previous = {}
            return {}
        pids = [p for p in pids if p >= self.cursor] + [p for p in pids if p < self.cursor]
        clients, owners, starts, timestamps = {}, defaultdict(set), {}, {}
        partial, denied, unreadable = False, 0, set()
        for pid in pids:
            if deadline is not None and time.monotonic() >= deadline:
                partial = True
                self.cursor = pid
                break
            pending = {}
            try:
                with os.scandir(PROC_ROOT / str(pid) / "fd") as fds:
                    for fd in fds:
                        if deadline is not None and time.monotonic() >= deadline:
                            partial = True
                            self.cursor = pid + 1
                            break
                        try:
                            if not os.readlink(fd.path).startswith("/dev/dri/"):
                                continue
                            try:
                                with (PROC_ROOT / str(pid) / "fdinfo" / fd.name).open(encoding="ascii") as stream:
                                    data = stream.read(65536)
                            except (OSError, UnicodeError):
                                unreadable.add(pid)
                                continue
                            if len(data) == 65536:
                                unreadable.add(pid)
                                continue
                            client = parse_fdinfo(data)
                        except (OSError, UnicodeError):
                            continue  # protected or closed between fd and fdinfo
                        if client is not None and client.device in self.device_keys:
                            pending.setdefault((client.device, client.client), client)
            except PermissionError:
                denied += 1
                continue
            except OSError:
                continue
            if partial:
                break  # do not publish incomplete per-PID totals
            if pending:
                starts[pid] = _start_time(pid)
                for key, client in pending.items():
                    clients.setdefault(key, client)
                    owners[key].add(pid)
                    timestamps.setdefault(key, time.monotonic_ns())
        if not partial:
            self.cursor = 0
        self.status = ("partial (scan budget reached; rows cover only this refresh's scanned PIDs)"
                       if partial else "complete (accessible local DRM handles)")
        if denied:
            self.status += f"; {denied} process fd directories inaccessible"
        if unreadable:
            self.status += f"; {len(unreadable)} processes have unreadable or oversized DRM fdinfo"
        rows, history = defaultdict(dict), {}
        per_pid = defaultdict(list)
        for key, client in clients.items():
            self.known_engines[client.device].update(client.engines)
            for pid in owners[key]:
                per_pid[(client.device, pid)].append(key)
        # amdgpu_show_fdinfo omits engines whose cumulative usage is zero.
        # Establish support from a real engine counter on this PCI device first;
        # never turn malformed counters or an unsupported device into zero.
        for client in clients.values():
            if not client.bad_engines:
                for engine in self.known_engines[client.device]:
                    client.engines.setdefault(engine, 0)
        for (device, pid), keys in per_pid.items():
            g = ProcGpu(attributed=False, available=set())
            rows[device][pid] = g
            if partial or pid in unreadable or any(len(owners[key]) != 1 for key in keys):
                continue
            if all(clients[key].vram is not None for key in keys):
                g.drm_vram = sum(clients[key].vram for key in keys)
            if all(clients[key].gtt is not None for key in keys):
                g.drm_gtt = sum(clients[key].gtt for key in keys)
            rates = defaultdict(float)
            valid = bool(starts[pid])
            for key in keys:
                c, now = clients[key], timestamps[key]
                prev = self.previous.get(key)
                counters = dict(c.engines)
                identity = (pid, starts[pid])
                if c.bad_engines or not c.engines or prev is None or prev[2] != identity:
                    valid = False
                else:
                    then, old, _ = prev
                    for engine, value in c.engines.items():
                        capacity = c.capacities.get(engine, 1)
                        if now <= then or capacity <= 0:
                            valid = False
                            continue
                        counters[engine] = max(old.get(engine, 0), value)
                        rates[_engine(engine)] += 100 * (counters[engine] - old.get(engine, 0)) / (now - then) / capacity
                if not c.bad_engines:
                    history[key] = (now, counters, identity)
            if valid:
                g.available.add("engine")
                g.engines.update({engine: min(100.0, value) for engine, value in rates.items()})
        self.previous = history
        return dict(rows)


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ""


def _number(path: Path, scale: int = 1) -> int | float | None:
    try:
        value = int(_text(path))
        return (value if scale == 1 else value / scale) if value >= 0 else None
    except ValueError:
        return None


def devices() -> dict[int, Path]:
    from .linux import pci_key

    out = {}
    for card in sorted(DRM_ROOT.glob("card*")):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        try:
            device = (card / "device").resolve()
            driver = (device / "driver").resolve().name
        except (OSError, RuntimeError):
            continue  # inaccessible or changing sysfs topology
        if _text(device / "vendor") != "0x1002":
            continue
        if driver != "amdgpu":
            continue
        if not _PCI.fullmatch(device.name):
            continue
        out[pci_key(device.name)] = device
    return out


def enumerate_adapters() -> list[Adapter]:
    return [Adapter(key, _text(path / "product_name") or f"AMD GPU {path.name}", VENDOR_AMD,
                    _number(path / "mem_info_vram_total"), 0, False)
            for key, path in devices().items()]


class AmdGpuCounters:
    def __init__(self) -> None:
        self.paths = devices()
        self.driver = f"amdgpu (Linux {platform.release()})"
        self.telemetry: dict[int, DeviceTelemetry] = {}
        self.process_reader = DrmProcesses(self.paths)
        self.scan_budget: float | None = 0.05

    def sample(self) -> dict[int, AdapterGpu]:
        out = {}
        self.telemetry = {}
        for key, path in self.paths.items():
            d = DeviceTelemetry(index=0, name=_text(path / "product_name") or f"AMD GPU {path.name}",
                                pci_bus=path.name, driver_model="amdgpu", source="AMD sysfs")
            d.mem_total = _number(path / "mem_info_vram_total")
            d.mem_used = _number(path / "mem_info_vram_used")
            d.util_gpu = _number(path / "gpu_busy_percent")
            d.util_mem = _number(path / "mem_busy_percent")
            hwmon = next((p for p in sorted((path / "hwmon").glob("hwmon*"))
                          if _text(p / "name") == "amdgpu"), None)
            if hwmon is not None:
                d.temp = _number(hwmon / "temp1_input", 1000)
                d.power_w = _number(hwmon / "power1_average", 1_000_000)
                if d.power_w is None:
                    d.power_w = _number(hwmon / "power1_input", 1_000_000)
                d.power_kind = "SoC (includes CPU on APUs)" if d.power_w is not None else ""
                d.power_limit_w = _number(hwmon / "power1_cap", 1_000_000)
                d.gfx_clock = _number(hwmon / "freq1_input", 1_000_000)
                d.mem_clock = _number(hwmon / "freq2_input", 1_000_000)
            self.telemetry[key] = d
            out[key] = AdapterGpu(dedicated=d.mem_used)
        return out
