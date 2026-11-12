"""Read-only Xe device queries and PCI-matched engine performance counters.

The query layouts are the public drm/xe_drm.h ABI. No driver library, workload,
register writes, device setting changes or privilege escalation is involved.
"""
from __future__ import annotations

import ctypes
import errno
import os
import re
import struct
from pathlib import Path

QUERY = struct.Struct("=QIIQ2Q")
IOCTL_DEVICE_QUERY = 0xC0286440  # DRM_IOWR(DRM_COMMAND_BASE, struct drm_xe_device_query)
ENGINE = struct.Struct("=4H3Q")
REGION = struct.Struct("=HHI4Q6Q")
CLASS_NAMES = {0: "rcs", 1: "bcs", 2: "vcs", 3: "vecs", 4: "ccs"}


def perfmon_capable(status: str = "/proc/self/status") -> bool:
    """The memory query silently returns used=0 without either effective capability."""
    try:
        text = Path(status).read_text()
        bits = int(re.search(r"^CapEff:\s*([0-9a-fA-F]+)$", text, re.M)[1], 16)
        return bool(bits & ((1 << 38) | (1 << 21)))  # CAP_PERFMON / CAP_SYS_ADMIN
    except (OSError, ValueError, TypeError):
        return False


class XeDevice:
    def __init__(self, sysfs: str, pdev: str, *, dev_root: str = "/dev/dri",
                 pmu_root: str = "/sys/bus/event_source/devices") -> None:
        self.sysfs = Path(sysfs)
        self.pdev = pdev
        self.dev_root = Path(dev_root)
        self.pmu_root = Path(pmu_root) / ("xe_" + pdev.replace(":", "_"))
        self.fd: int | None = None
        self.query_error = ""
        self.pmu_error = ""
        self._open_attempted = False
        self._perf: dict[str, tuple[int, int]] = {}
        self._before: dict[str, tuple[int, int]] = {}
        self._pmu_attempted = False
        self._engine_list: list[tuple[int, int, int]] | None = None

    def query(self, kind: int) -> bytes | None:
        try:
            import fcntl  # import-safe on Windows and macOS

            if not self._open_attempted:
                self._open_attempted = True
                nodes = sorted(p.name for p in (self.sysfs / "drm").iterdir()
                               if re.fullmatch(r"renderD\d+", p.name))
                if len(nodes) != 1:
                    self.query_error = "no unique render node for this PCI device"
                    return None
                self.fd = os.open(self.dev_root / nodes[0], os.O_RDONLY | os.O_CLOEXEC)
            if self.fd is None:
                return None
            request = bytearray(QUERY.pack(0, kind, 0, 0, 0, 0))
            fcntl.ioctl(self.fd, IOCTL_DEVICE_QUERY, request, True)
            size = QUERY.unpack(request)[2]
            if not 8 <= size <= 1 << 20:
                self.query_error = "invalid device-query size"
                return None
            data = ctypes.create_string_buffer(size)
            QUERY.pack_into(request, 0, 0, kind, size, ctypes.addressof(data), 0, 0)
            fcntl.ioctl(self.fd, IOCTL_DEVICE_QUERY, request, True)
            actual = QUERY.unpack(request)[2]
            if not 8 <= actual <= size:
                self.query_error = "device-query size changed"
                return None
            self.query_error = ""
            return data.raw[:actual]
        except (OSError, ImportError) as error:
            self.query_error = ("render-node access denied" if getattr(error, "errno", None)
                                in (errno.EACCES, errno.EPERM) else "device query unavailable")
            return None

    def engines(self) -> list[tuple[int, int, int]] | None:
        if self._engine_list is not None:
            return self._engine_list
        data = self.query(0)
        if data is None:
            return None
        count = struct.unpack_from("=I", data)[0]
        if not 0 < count <= 4096 or len(data) != 8 + count * ENGINE.size:
            self.query_error = "invalid engine-query layout"
            return None
        engines = []
        for index in range(count):
            cls, instance, gt, _pad, *_reserved = ENGINE.unpack_from(data, 8 + index * ENGINE.size)
            if cls == 5:  # VM_BIND is a virtual queue, not a timed hardware engine
                continue
            if cls not in CLASS_NAMES or instance > 255 or gt > 15:
                self.query_error = "unsupported engine-query entry"
                return None
            engines.append((cls, instance, gt))
        if not engines or len(set(engines)) != len(engines):
            self.query_error = "empty or duplicate engine-query entries"
            return None
        self._engine_list = engines
        return engines

    def memory(self) -> tuple[int | None, int | None]:
        data = self.query(1)
        if data is None:
            return None, None
        count = struct.unpack_from("=I", data)[0]
        if not 0 < count <= 4096 or len(data) != 8 + count * REGION.size:
            self.query_error = "invalid memory-query layout"
            return None, None
        regions = []
        seen = set()
        for index in range(count):
            cls, instance, _page, total, used, _visible, _visible_used, *_reserved = REGION.unpack_from(
                data, 8 + index * REGION.size)
            if cls != 1:  # system RAM is not dedicated device VRAM
                continue
            if instance in seen or not total or used > total:
                self.query_error = "invalid VRAM region"
                return None, None
            seen.add(instance)
            regions.append((total, used))
        if not regions:
            return None, None
        total = sum(region[0] for region in regions)
        # Without the capability even an apparently valid zero is deliberately hidden.
        used = sum(region[1] for region in regions) if perfmon_capable() else None
        return total, used

    def _spec(self) -> tuple[int, int, dict[str, tuple[int, int]]] | None:
        try:
            typ = int((self.pmu_root / "type").read_text())
            cpu = int(re.split(r"[-,]", (self.pmu_root / "cpumask").read_text().strip())[0])
            fields = {}
            for name in ("event", "gt", "engine_class", "engine_instance"):
                text = (self.pmu_root / "format" / name).read_text().strip()
                match = re.fullmatch(r"config:(\d+)-(\d+)", text)
                if match is None:
                    return None
                lo, hi = map(int, match.groups())
                if not 0 <= lo <= hi <= 63:
                    return None
                fields[name] = (lo, hi)

            def encode(name, value):
                lo, hi = fields[name]
                if not 0 <= value < 1 << (hi - lo + 1):
                    raise ValueError("PMU field overflow")
                return value << lo

            events = []
            for name in ("engine-active-ticks", "engine-total-ticks"):
                text = (self.pmu_root / "events" / name).read_text().strip()
                key, value = text.split("=", 1)
                if key != "event":
                    return None
                events.append(encode("event", int(value, 0)))
            engines = self.engines()
            if engines is None:
                return None
            configs = {}
            for cls, instance, gt in engines:
                selector = encode("gt", gt) | encode("engine_class", cls) | encode("engine_instance", instance)
                configs[f"{CLASS_NAMES[cls]}{instance}@gt{gt}"] = tuple(selector | event for event in events)
            return typ, cpu, configs
        except (OSError, ValueError):
            return None

    def rates(self, open_counter, read_counter) -> tuple[float | None, float | None]:
        if not self._pmu_attempted:
            self._pmu_attempted = True
            spec = self._spec()
            if spec is None:
                self.pmu_error = self.query_error or "Xe engine performance counters unavailable"
                return None, None
            typ, cpu, configs = spec
            for name, pair in configs.items():
                opened = []
                for config in pair:
                    fd, error = open_counter(typ, config, cpu)
                    if fd is None:
                        for existing in opened:
                            os.close(existing)
                        self._close_perf()
                        self.pmu_error = "denied" if error == "denied" else "Xe performance counter unavailable"
                        return None, None
                    opened.append(fd)
                self._perf[name] = tuple(opened)
        if self.pmu_error or not self._perf:
            return None, None
        values = {}
        for name, (active_fd, total_fd) in self._perf.items():
            active, total = read_counter(active_fd), read_counter(total_fd)
            before = self._before.get(name)
            if active is None or total is None:
                self._close_perf()
                self._pmu_attempted = False  # a transient read failure can recover next refresh
                return None, None
            self._before[name] = (active, total)
            if before is not None and total > before[1] and active >= before[0]:
                values[name] = min(100.0, 100.0 * (active - before[0]) / (total - before[1]))
        video_names = [name for name in self._perf if name.startswith(("vcs", "vecs"))]
        video = max((values[name] for name in video_names), default=None) if all(
            name in values for name in video_names) else None
        return (max(values.values()) if len(values) == len(self._perf) else None), video

    def _close_perf(self):
        for pair in self._perf.values():
            for fd in pair:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._perf.clear()
        self._before.clear()

    def close(self):
        self._close_perf()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
