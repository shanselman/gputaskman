"""macOS GPU backend: device stats and per-process GPU time from IOKit, GPU memory from footprint.

Apple Silicon and Intel integrated GPUs share system memory with the CPU. Discrete AMD GPUs in Intel Macs
expose device-wide VRAM, but not per-process VRAM through footprint. What macOS does expose without root:

- Device-wide: the IOAccelerator's PerformanceStatistics -- device utilization and GPU memory counters.
- Per process, for every user: Metal clients name their creator PID and keep cumulative GPU time. GPU% is
  the change between two samples.
- Per process, own processes only: footprint(1) reports the process's memory charged to graphics (Metal
  buffers and textures) -- the "(graphics)" categories. Other users' processes (WindowServer, daemons) need
  root, so their GPU memory is unknown, never zero.

Measured on an M4 Mac mini (macOS 26.3): a PyTorch MPS job with a 2 GiB buffer showed 2.01 GiB (2062 MiB) of graphics
footprint, GPU time matched Device Utilization (98% vs 99%), and Alloc system memory rose by the same amount.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import plistlib
import re
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

from .dxgi import Adapter
from .nvml import NvDevice
from .pdh import AdapterGpu, ProcGpu

VENDOR_APPLE = 0x106B
VENDOR_AMD = 0x1002
VENDOR_INTEL = 0x8086
VENDOR_NVIDIA = 0x10DE
VENDOR_UNKNOWN = 0
ADAPTER_KEY = 1  # legacy key when the only accelerator has no registry ID
_CREATOR = re.compile(r"pid (\d+)")
# Other users' processes show only GPU time; keep the idle ones out of the table (except the compositor).
ALWAYS_SHOWN = {"windowserver"}


def _ioreg(*args: str) -> list | None:
    try:
        out = subprocess.run(["ioreg", "-a", "-w", "0", "-r", *args], capture_output=True, timeout=5).stdout
        return plistlib.loads(out) if out.strip() else []
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException, ValueError):
        return None


@dataclass
class DeviceStats:
    model: str = "Apple GPU"
    cores: int | None = None
    utilization: float | None = None  # Device Utilization %
    renderer: float | None = None
    tiler: float | None = None
    allocated: int | None = None
    total: int | None = None
    vendor_id: int = VENDOR_APPLE
    process_memory_supported: bool = True


def _number(value) -> int | None:
    if isinstance(value, bytes):
        return int.from_bytes(value, "little") if len(value) in (4, 8) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 0xFFFFFFFFFFFFFFFF else None
    if isinstance(value, float):
        return int(value) if 0 <= value < 2**64 and value.is_integer() else None
    if isinstance(value, str) and len(value) <= 20 and value.isascii() and value.isdecimal():
        return _number(int(value))
    return None


def _first_number(*values) -> int | None:
    return next((number for value in values if (number := _number(value)) is not None), None)


def _metric(*values) -> float | None:
    for value in values:
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and 0 <= value <= 0xFFFFFFFFFFFFFFFF and math.isfinite(value)):
            return value
        number = _number(value)
        if number is not None:
            return number
    return None


def _model(value) -> str:
    if isinstance(value, bytes):
        return value.rstrip(b"\0").decode("utf-8", "replace")
    return value if isinstance(value, str) else ""


def _pci_parents(entries: list | None) -> dict[int, dict]:
    """Map accelerator registry IDs to their nearest PCI ancestor, never by model or list order."""
    parents: dict[int, dict] = {}
    ambiguous: set[int] = set()

    def visit(entry, pci=None):
        if not isinstance(entry, dict):
            return
        if entry.get("IOClass") == "IOPCIDevice":
            pci = entry
        if pci is not None and isinstance(entry.get("PerformanceStatistics"), dict):
            key = _number(entry.get("IORegistryEntryID"))
            if key is not None and key not in ambiguous:
                if key in parents:
                    parents.pop(key)
                    ambiguous.add(key)
                else:
                    parents[key] = {k: pci[k] for k in ("model", "VRAM,totalMB", "vram,totalMB", "ATY,memsize")
                                    if k in pci}
        for child in entry.get("IORegistryEntryChildren") or []:
            visit(child, pci)

    for entry in entries or []:
        visit(entry)
    return parents


def _read_pci_parents() -> dict[int, dict]:
    return _pci_parents(_ioreg("-l", "-p", "IOService", "-c", "IOPCIDevice"))


def _has_amd(entries: list | None) -> bool:
    return any(any(label in f"{_model(e.get('model'))} {e.get('IOClass', '')}".lower()
                   for label in ("amd", "radeon"))
               for e in entries or [] if isinstance(e, dict))


def parse_accelerator(entries: list | None, parents: dict[int, dict] | None = None) -> DeviceStats | None:
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        stats = e.get("PerformanceStatistics")
        if not isinstance(stats, dict):
            continue
        parent = (parents or {}).get(_number(e.get("IORegistryEntryID")), {})
        name = _model(e.get("model")) or _model(parent.get("model")) or _model(e.get("IOClass")) or "Mac GPU"
        kind = f"{name} {e.get('IOClass', '')}".lower()
        vendor_id = VENDOR_AMD if "amd" in kind or "radeon" in kind else (
            VENDOR_INTEL if "intel" in kind else
            VENDOR_NVIDIA if "nvidia" in kind or "nvaccelerator" in kind else
            VENDOR_APPLE if "apple" in kind or "agx" in kind else VENDOR_UNKNOWN)
        if vendor_id == VENDOR_AMD:
            allocated = _number(stats.get("inUseVidMemoryBytes"))
            total = None
            for key in ("VRAM,totalMB", "vram,totalMB", "ATY,memsize"):
                total = _first_number(parent.get(key), e.get(key), stats.get(key))
                if total is not None:
                    total = _number(total * (1 if key == "ATY,memsize" else 1 << 20))
                    break
        elif vendor_id == VENDOR_INTEL:
            allocated = _number(stats.get("gartUsedBytes"))
            total = _number(stats.get("gartSizeBytes")) if allocated is not None else None
            if allocated is None:
                allocated = _number(stats.get("Alloc system memory"))
        elif vendor_id == VENDOR_APPLE:
            allocated = _number(stats.get("Alloc system memory"))
            total = physical_memory()
        else:
            allocated = None
            total = None
        utilization = _metric(stats.get("Device Utilization %"), stats.get("GPU Activity(%)"))
        return DeviceStats(model=name, cores=_number(e.get("gpu-core-count")), utilization=utilization,
                           renderer=_metric(stats.get("Renderer Utilization %")),
                           tiler=_metric(stats.get("Tiler Utilization %")),
                           allocated=allocated, total=total,
                           vendor_id=vendor_id,
                           process_memory_supported=vendor_id in (VENDOR_APPLE, VENDOR_INTEL))
    return None


def parse_clients(entries: list | None) -> dict[int, dict[int, int]]:
    """{pid: {client id: accumulated GPU ns}}. Rates are taken per client: a closed client takes its
    accumulated time with it, so per-process sums would drop and hide real work."""
    out: dict[int, dict[int, int]] = defaultdict(dict)
    for i, e in enumerate(entries or []):
        m = _CREATOR.match(str(e.get("IOUserClientCreator", "")))
        if not m:
            continue
        cid = e.get("IORegistryEntryID", -1 - i)
        usage = e.get("AppUsage")
        if isinstance(usage, list):
            elapsed = sum(int(u.get("accumulatedGPUTime", 0))
                          for u in usage if isinstance(u, dict))
        else:
            elapsed = int(e.get("accumulatedGPUTime", 0))
        out[int(m[1])][cid] = elapsed
    return dict(out)


def graphics_bytes(process: dict) -> int:
    """footprint JSON: dirty bytes of the categories macOS charges to graphics."""
    return int(sum(v.get("dirty", 0) for k, v in (process.get("categories") or {}).items() if k.endswith("(graphics)")))


def physical_memory() -> int:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # no os.sysconf on Windows
        return 0


def macos_version() -> str:
    try:
        return subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def enumerate_adapters() -> list[Adapter]:
    entries = _ioreg("-d", "1", "-c", "IOAccelerator")
    parents = _read_pci_parents() if _has_amd(entries) else {}
    has_id = any(_number(e.get("IORegistryEntryID")) is not None
                 for e in entries or [] if isinstance(e, dict))
    return [Adapter(luid=key, name=dev.model, vendor_id=dev.vendor_id,
                    dedicated_vram=dev.total if dev.total is not None else 0,
                    shared_system=0, software=False)
            for key, dev in _devices(entries, parents).items() if has_id or dev.vendor_id == VENDOR_APPLE]


def _devices(entries: list | None, parents: dict[int, dict] | None = None) -> dict[int, DeviceStats]:
    candidates = [e for e in entries or [] if isinstance(e, dict)]
    ids = [_number(e.get("IORegistryEntryID")) for e in candidates]
    devices = {}
    for entry, key in zip(candidates, ids):
        if key is None:
            if len(candidates) != 1:
                continue
            key = ADAPTER_KEY
        elif ids.count(key) != 1:
            continue
        dev = parse_accelerator([entry], parents)
        if dev is not None:
            devices[key] = dev
    return devices


def _owner(pid: int) -> int | None:
    try:
        import psutil
        return psutil.Process(pid).uids().real
    except Exception:
        return None


def _identity(pid: int) -> float | None:
    """Process start time: tells a reused PID from the process that had it before."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


class FootprintPoller:
    """footprint(1) takes ~70 ms per process, too slow for the refresh loop, so a background thread measures
    the targets: busy or not-yet-measured processes every pass (~2 s), idle ones every 30 s. Results are tied
    to a process identity (PID + start time), so a measurement never lands on a reused PID."""

    CYCLE = 2.0  # seconds between passes
    IDLE_EVERY = 30.0  # re-measure idle processes this often
    FAIL_LIMIT = 3  # consecutive failures before backing off (one timeout under load is not "denied")
    FAIL_RETRY = 60.0

    def __init__(self, run=subprocess.run, clock=time.monotonic) -> None:
        self._run = run
        self._clock = clock
        self._targets: dict[int, tuple[object, bool]] = {}  # pid -> (identity, busy), busiest first
        self._sizes: dict[int, tuple[object, int]] = {}  # pid -> (identity, graphics bytes)
        self._due: dict[int, float] = {}
        self._fails: dict[int, int] = {}
        self._retry: dict[int, float] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._passed = threading.Event()  # set after each completed pass
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="macos-footprint", daemon=True)
            self._thread.start()

    def set_targets(self, targets: list[tuple[int, object, bool]]) -> None:
        """[(pid, identity, busy)], busiest first. Anything else, or a PID whose identity changed, is forgotten."""
        with self._lock:
            self._targets = {pid: (ident, busy) for pid, ident, busy in targets}
            for store in (self._sizes, self._due, self._fails, self._retry):
                for pid in list(store):
                    if pid not in self._targets:
                        del store[pid]
            for pid, (ident, _) in self._targets.items():
                if pid in self._sizes and self._sizes[pid][0] != ident:  # reused PID
                    for store in (self._sizes, self._due, self._fails, self._retry):
                        store.pop(pid, None)
        self._wake.set()

    def get(self, pid: int, identity: object = None) -> int | None:
        with self._lock:
            hit = self._sizes.get(pid)
        return hit[1] if hit and hit[0] == identity else None

    def measure(self, pid: int) -> int | None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fp.json")
            try:
                self._run(["footprint", "-f", "bytes", "-j", path, "-p", str(pid)],
                          capture_output=True, timeout=10)
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, subprocess.SubprocessError, ValueError):
                return None
        for p in data.get("processes") or []:
            if p.get("pid") == pid:
                return graphics_bytes(p)
        return None

    def poll_once(self) -> None:
        with self._lock:
            targets = list(self._targets.items())
        for pid, (ident, busy) in targets:
            if self._stop:
                return
            now = self._clock()
            with self._lock:
                if self._retry.get(pid, 0) > now:
                    continue
                if not busy and pid in self._sizes and self._due.get(pid, 0) > now:
                    continue
            size = self.measure(pid)
            with self._lock:
                if self._targets.get(pid, (None,))[0] != ident:
                    continue  # exited or reused while footprint ran: don't resurrect it
                if size is None:
                    self._fails[pid] = self._fails.get(pid, 0) + 1
                    if self._fails[pid] >= self.FAIL_LIMIT:  # other user's process, or footprint keeps failing
                        self._retry[pid] = now + self.FAIL_RETRY
                        self._sizes.pop(pid, None)
                else:
                    self._sizes[pid] = (ident, size)
                    self._fails.pop(pid, None)
                    self._due[pid] = now + self.IDLE_EVERY

    def wait_pass(self, timeout: float) -> bool:
        self._passed.clear()
        return self._passed.wait(timeout)

    def _loop(self) -> None:
        while not self._stop:
            start = self._clock()
            self.poll_once()
            self._passed.set()
            self._wake.wait(max(self.CYCLE - (self._clock() - start), 0.2))
            self._wake.clear()

    def close(self) -> None:
        self._stop = True
        self._wake.set()


class MacCounters:
    """Same interface as pdh.GpuCounters: sample() -> ({adapter key: {pid: ProcGpu}}, {adapter key: AdapterGpu}).
    `device` carries the device-wide stats in the NvDevice shape the header already renders."""

    def __init__(self, poller: FootprintPoller | None = None, uid: int | None = None, root: bool | None = None,
                 identity=_identity, owner=_owner) -> None:
        self.ok = True
        self.device: NvDevice | None = None
        self._uid = os.getuid() if uid is None else uid
        # As root footprint can read every process, so every process is measurable (not only uid 0's).
        self._root = (os.geteuid() == 0) if root is None else root
        self._identity, self._owner = identity, owner
        self._prev: dict[int, tuple[object, dict[int, int]]] = {}  # pid -> (identity, {client id: ns})
        self._last: float | None = None  # monotonic time of the previous sample
        self._owners: dict[int, int | None] = {}
        self._names: dict[int, str] = {}
        self._unknown_identity: set[int] = set()
        self._pci_parents: dict[int, dict] | None = None
        self._selected: Adapter | None = None
        self._selection_set = False
        self._multiple = False
        self._sampled_key: int | None = None
        self.status = ""
        self.process_memory_available = True
        self.poller = poller or FootprintPoller()
        self.poller.start()
        self.sample()  # baseline: GPU% is a rate, like the WDDM counters' first collection

    def select_adapter(self, adapter: Adapter | None, multiple: bool = False) -> None:
        keep_baseline = (adapter is not None and adapter.luid == self._sampled_key
                         and self.device is not None and self.device.name == adapter.name
                         and not multiple and not self._multiple)
        self._selected = adapter
        self._selection_set = True
        self._multiple = self._multiple or multiple
        if not keep_baseline:
            self._prev.clear()
            self._last = None
            self.poller.set_targets([])

    def warm_up(self, timeout: float) -> None:
        """One-shot output: wait for a full footprint pass so own processes have their GPU memory."""
        self.poller.wait_pass(timeout)

    def _name(self, pid: int) -> str:
        if pid not in self._names:
            try:
                import psutil
                self._names[pid] = psutil.Process(pid).name()
            except Exception:
                self._names[pid] = ""
        return self._names[pid]

    def sample(self, accel: list | None = None, clients: list | None = None,
               now: float | None = None) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        now = time.monotonic() if now is None else now  # a rate: immune to wall-clock adjustments
        accel = _ioreg("-d", "1", "-c", "IOAccelerator") if accel is None else accel
        if clients is None:
            clients = _ioreg("-l", "-c", "AGXDeviceUserClient")
            if clients == []:
                clients = _ioreg("-l", "-c", "IOAccelSubmitter2")
        procs: dict[int, dict[int, ProcGpu]] = defaultdict(dict)
        adapters: dict[int, AdapterGpu] = defaultdict(AdapterGpu)
        if self._pci_parents is None and _has_amd(accel):
            self._pci_parents = _read_pci_parents()
        candidates = [e for e in accel or [] if isinstance(e, dict)]
        if len(candidates) > 1:
            self._multiple = True
        devices = _devices(accel, self._pci_parents)
        key = self._selected.luid if self._selected else next(iter(devices), ADAPTER_KEY)
        dev = devices.get(key)
        if self._selection_set and self._selected is None:
            dev = None
        if (self._selected and self._selected.vendor_id != VENDOR_APPLE and len(candidates) == 1
                and _number(candidates[0].get("IORegistryEntryID")) is None):
            dev = None
        if dev and self._selected and (dev.vendor_id != self._selected.vendor_id or
                                       (key == ADAPTER_KEY and dev.model != self._selected.name)):
            dev = None
        self.status = ""
        self.process_memory_available = bool(dev and dev.process_memory_supported and not self._multiple)
        self._sampled_key = key if dev is not None else None
        if dev is None:
            self.device = None  # no reading this time: never keep showing the last one as current
            self.status = "Selected macOS GPU is unavailable or cannot be matched to its registry identity."
        else:
            if dev.allocated is not None:
                adapters[key].dedicated = float(dev.allocated)
            self.device = NvDevice(index=0, name=dev.model, driver_model="Metal",
                                   mem_total=dev.total or None,
                                   mem_used=dev.allocated, util_gpu=dev.utilization, source="IOKit")
            if dev.allocated is not None and not dev.total:
                self.status = "macOS GPU memory total is unavailable from the matching driver counter."
        if dev is None or self._multiple:
            if self._multiple and dev is not None:
                self.status = " ".join(filter(None, (self.status,
                    "Multiple macOS GPUs: per-process GPU data cannot be matched to a selected GPU.")))
            self._prev.clear()
            self._last = None
            self.poller.set_targets([])
            return procs, adapters
        if clients is None:
            # A failed client query is not an empty client list: preserve the last valid rate baseline and
            # footprint targets so the next successful sample spans the full elapsed interval.
            self.process_memory_available = False
            self.status = " ".join(filter(None, (self.status, "macOS GPU process counters are unavailable.")))
            return procs, adapters
        per_pid = parse_clients(clients)
        elapsed = now - self._last if self._last is not None else None
        targets: list[tuple[float, int, object]] = []
        current: dict[int, tuple[object, dict[int, int]]] = {}
        for pid, times in per_pid.items():
            ident = self._identity(pid)
            was_unknown = pid in self._unknown_identity
            prev = self._prev.get(pid)
            if ident is None:
                self._unknown_identity.add(pid)
                self._owners.pop(pid, None)
                self._names.pop(pid, None)
                prev_times = {}
            elif prev is None or prev[0] != ident or was_unknown:
                prev_times = {}
                self._owners.pop(pid, None)  # new or reused PID: re-read its owner and name
                self._names.pop(pid, None)
            else:
                prev_times = prev[1]
            if ident is not None:
                current[pid] = (ident, times)
                self._unknown_identity.discard(pid)
            pct = None if ident is None or was_unknown else 0.0
            if pct is not None and elapsed and elapsed > 0:
                # A client that appeared since the last sample did all of its work in this interval.
                busy_ns = sum(max(ns - prev_times.get(cid, 0), 0) for cid, ns in times.items())
                pct = min(busy_ns / 1e9 / elapsed * 100.0, 100.0)
            if pid not in self._owners:
                self._owners[pid] = self._owner(pid)
            mine = self._root or self._owners[pid] == self._uid
            if not mine and (pct is None or pct < 0.5) and self._name(pid).lower() not in ALWAYS_SHOWN:
                continue
            g = ProcGpu()
            if pct is None:
                g.available.discard("engine")
            elif pct:
                g.engines["gpu"] = pct
            if dev and not dev.process_memory_supported:
                g.attributed = False  # footprint doesn't prove VRAM ownership on this adapter
                g.available.discard("dedicated")
            elif mine:
                if ident is not None:
                    targets.append((pct or 0.0, pid, ident))
                    size = self.poller.get(pid, ident)
                else:
                    size = None
                if size is None:
                    g.attributed = False  # not measured yet (or footprint failed): unknown, not zero
                else:
                    g.dedicated = float(size)
            else:
                g.attributed = False  # footprint needs root for other users' processes
            procs[key][pid] = g
        self._prev, self._last = current, now
        self._unknown_identity.intersection_update(per_pid)
        self._owners = {p: o for p, o in self._owners.items() if p in per_pid}
        self._names = {p: n for p, n in self._names.items() if p in per_pid}
        targets.sort(key=lambda t: -t[0])
        self.poller.set_targets([(pid, ident, pct >= 0.5) for pct, pid, ident in targets])
        return procs, adapters

    def close(self) -> None:
        self.poller.close()


# ---------------------------------------------------------------- memory maps (psutil has none on macOS)

PROC_PIDREGIONINFO = 7


class _RegionInfo(ctypes.Structure):  # struct proc_regioninfo, <sys/proc_info.h>
    _fields_ = [(n, ctypes.c_uint32) for n in ("protection", "max_protection", "inheritance", "flags")] + [
        ("offset", ctypes.c_uint64)] + [(n, ctypes.c_uint32) for n in (
            "behavior", "user_wired_count", "user_tag", "pages_resident", "pages_shared_now_private",
            "pages_swapped_out", "pages_dirtied", "ref_count", "shadow_depth", "share_mode",
            "private_pages_resident", "shared_pages_resident", "obj_id", "depth")] + [
        ("address", ctypes.c_uint64), ("size", ctypes.c_uint64)]


@dataclass
class MappedFile:
    path: str
    rss: int


_libproc = None


def _load_libproc():
    global _libproc
    if _libproc is None:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        lib.proc_regionfilename.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32]
        _libproc = lib
    return _libproc


def memory_maps(pid: int, max_regions: int = 200_000) -> list[MappedFile]:
    """File-backed regions of a process, grouped by path (like psutil's memory_maps(grouped=True)). Walks the
    address space with proc_pidinfo(PROC_PIDREGIONINFO) -- ~4 ms for a desktop app. Raises PermissionError
    for processes this user can't inspect."""
    lib = _load_libproc()
    info, buf = _RegionInfo(), ctypes.create_string_buffer(4096)
    page = os.sysconf("SC_PAGE_SIZE")  # macOS only
    files: dict[str, int] = {}
    addr, n = 0, 0
    while n < max_regions:
        if lib.proc_pidinfo(pid, PROC_PIDREGIONINFO, addr, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
            break
        n += 1
        if lib.proc_regionfilename(pid, info.address, buf, len(buf)) > 0:
            path = buf.value.decode("utf-8", "replace")
            files[path] = files.get(path, 0) + info.pages_resident * page
        if info.size == 0:
            break
        addr = info.address + info.size
    if n == 0:
        raise PermissionError(f"cannot read the address space of pid {pid}")
    return [MappedFile(p, rss) for p, rss in files.items()]
