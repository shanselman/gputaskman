"""Read-only BMG G31 PMT counters, matched by PCI path and exact telemetry GUID.

Register layout reference (MIT, Copyright 2024-2026 Intel Corporation):
intel/compute-runtime @ 13f35c66bc22a281b812e328d79871ea43fb9e0d,
level_zero/sysman/source/shared/linux/product_helper/xe2_hpg_core/bmg/
sysman_product_helper_bmg.cpp. Only the G31 PUNIT rev 3 and OOBMSM rev 16
layouts are decoded. Other revisions must never inherit these offsets.
Reference attribution and MIT notice: licenses/intel-pmt.txt.

No mmap, PCI register writes, runtime initialization or privilege changes.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import struct
import time

PUNIT_GUID = 0x1E2F8302
OOB_GUID = 0x5E2F8311
SOURCE = "Intel PMT (BMG G31)"
MEMORY_RATES = ("mem_read_bps", "mem_write_bps", "mem_bandwidth_pct")
PCIE_RATES = ("pcie_rx_kbs", "pcie_tx_kbs")

# Channel stride is 80 bytes; MSU stride is 160. Entries are (lower-word
# offset in MSU0/channel0, request size, direction). Upper words follow +4.
REQUESTS = ((376, 32, "read"), (384, 64, "read"), (392, 32, "write"),
            (400, 64, "write"), (408, 32, "read"), (416, 64, "read"),
            (424, 32, "read"), (432, 64, "read"), (440, 32, "write"),
            (448, 64, "write"))


def request_layout(mask: int):
    """Layout includes rev 16's documented MSU6/MSU7 CH1 GT32-read swap."""
    for msu in range(8):
        if mask & (1 << msu):
            for channel in range(2):
                for base, size, direction in REQUESTS:
                    index = 13 - msu if channel == 1 and base == 376 and msu in (6, 7) else msu
                    yield base + 160 * index + 80 * channel, size, direction


class _ReadError(Exception):
    pass


class _Pending(_ReadError):
    pass


def _reason(error: OSError) -> str:
    if error.errno in (errno.EACCES, errno.EPERM):
        return "read access denied (host permission required)"
    return "unavailable" if error.errno in (errno.ENOENT, errno.ENODEV) else f"read failed ({error.errno})"


class _Node:
    def __init__(self, path: Path, guid: int, size: int):
        self.path, self.guid, self.size = path, guid, size
        self.fd: int | None = None
        self.retry_at = 0.0
        self.failure = ""

    def frames(self, now: float) -> tuple[bytes, bytes, bytes]:
        if self.fd is None and now < self.retry_at:
            raise _ReadError(self.failure)
        try:
            if self.fd is None:
                self.fd = os.open(self.path / "telem", os.O_RDONLY | os.O_CLOEXEC)
            # Three bounded snapshots give upper/lower/upper consistency without
            # hundreds of syscalls or treating a torn 64-bit counter as a spike.
            frames = tuple(os.pread(self.fd, self.size, 0) for _ in range(3))
            if any(len(frame) != self.size for frame in frames):
                raise _ReadError("short telemetry read")
            self.failure = ""
            return frames
        except OSError as error:
            self.close()
            self.failure = _reason(error)
            self.retry_at = now + 5
            raise _ReadError(self.failure) from error

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise _ReadError("telemetry layout exceeds the advertised region")
    return struct.unpack_from("<I", data, offset)[0]


def _counter(frames: tuple[bytes, bytes, bytes], offset: int) -> int:
    hi = _u32(frames[0], offset + 4)
    if hi != _u32(frames[2], offset + 4):
        raise _ReadError("counter changed during sampling; waiting for a stable reading")
    return (hi << 32) | _u32(frames[1], offset)


class PmtTelemetry:
    def __init__(self, sysfs: str, *, root: str = "/sys/class/intel_pmt", clock=time.monotonic_ns):
        self.device = Path(sysfs).resolve()
        self.root = Path(root)
        self.clock = clock
        self.nodes: dict[int, _Node] = {}
        self.discovery_error = ""
        self.next_discovery = 0.0
        self.previous: dict[str, tuple[tuple, int]] = {}
        self.sources: dict[str, str] = {}

    def _discover(self, now: float):
        if now < self.next_discovery:
            return
        self.next_discovery = now + 5
        candidates: dict[int, list[_Node]] = {}
        unknown = set()
        try:
            paths = sorted(self.root.glob("telem[0-9]*"))
            if len(paths) > 256:
                raise _ReadError("too many PMT endpoints to inspect safely")
            for path in paths:
                try:
                    target = path.resolve(strict=True)
                    if self.device not in target.parents:
                        continue
                    guid = int((path / "guid").read_text().strip(), 0)
                    size = int((path / "size").read_text().strip(), 0)
                    if guid not in (PUNIT_GUID, OOB_GUID):
                        unknown.add(hex(guid))
                        continue
                    if not 4096 <= size <= 16384:
                        continue
                    candidates.setdefault(guid, []).append(_Node(target, guid, size))
                except (OSError, ValueError):
                    continue
        except (OSError, _ReadError) as error:
            self.discovery_error = str(error)
            return
        found = {}
        for guid, nodes in candidates.items():
            if len(nodes) != 1:
                continue  # do not guess between two endpoints with the same GUID
            node = nodes[0]
            old = self.nodes.get(guid)
            found[guid] = old if old and (old.path, old.size) == (node.path, node.size) else node
        if {g: (n.path, n.size) for g, n in found.items()} != {g: (n.path, n.size) for g, n in self.nodes.items()}:
            self.previous.clear()
            self.sources.clear()
        for guid, node in self.nodes.items():
            if found.get(guid) is not node:
                node.close()
        self.nodes = found
        self.discovery_error = ("ambiguous PMT endpoints" if any(len(v) > 1 for v in candidates.values()) else
                                "unsupported PMT GUID(s): " + ", ".join(sorted(unknown)) if unknown else
                                "no matching BMG G31 PMT endpoints")

    def _rates(self, kind: str, values: tuple, stamp: int):
        previous = self.previous.get(kind)
        self.previous[kind] = values, stamp
        if previous is None:
            raise _Pending("waiting for a second counter sample")
        before, when = previous
        if stamp <= when or len(values) != len(before) or any(a < b for a, b in zip(values, before)):
            raise _Pending("counter reset or invalid interval; collecting a new baseline")
        return tuple((a - b) * 1e9 / (stamp - when) for a, b in zip(values, before))

    def sample(self) -> tuple[dict, dict, dict]:
        stamp = self.clock()
        now = stamp / 1e9
        self._discover(now)
        values, status = {}, {}
        maximum = None
        punit = self.nodes.get(PUNIT_GUID)
        punit_fields = ("mem_clock", "mem_bandwidth_max_bps")
        try:
            if punit is None:
                raise _ReadError(self.discovery_error)
            frames = punit.frames(now)
            word = _u32(frames[1], 56)
            values["mem_clock"] = word & 0xFFFF  # actual MHz; a measured zero remains zero
            maximum = (word >> 16) * 12_500_000  # 100 Mb/s -> bytes/s, decimal units
            if maximum:
                values["mem_bandwidth_max_bps"] = maximum
            else:
                status["mem_bandwidth_max_bps"] = "PMT did not report maximum bandwidth"
            self.sources.update({field: SOURCE for field in punit_fields})
        except _ReadError as error:
            status.update({field: f"Intel PMT: {error}" for field in punit_fields})
        oob = self.nodes.get(OOB_GUID)
        try:
            if oob is None:
                raise _ReadError(self.discovery_error)
            frames = oob.frames(now)
        except _ReadError as error:
            self.previous.clear()
            status.update({field: f"Intel PMT: {error}" for field in (*MEMORY_RATES, *PCIE_RATES)})
            return values, status, dict(self.sources)
        try:
            counters = tuple(_counter(frames, offset) for offset in (280, 288))
            self.sources.update({field: SOURCE for field in PCIE_RATES})
            rx, tx = self._rates("pcie", counters, stamp)
            values.update(pcie_rx_kbs=rx / 1024, pcie_tx_kbs=tx / 1024)
        except _ReadError as error:
            if not isinstance(error, _Pending):
                self.previous.pop("pcie", None)
            status.update({field: f"Intel PMT: {error}" for field in PCIE_RATES})
        try:
            mask = _u32(frames[1], 3688)
            if not mask or mask & ~0xFF or any(_u32(frame, 3688) != mask for frame in frames):
                raise _ReadError("invalid or changing memory-subsystem mask")
            layout = tuple(request_layout(mask))
            counters = tuple(_counter(frames, offset) for offset, _, _ in layout)
            # Include the mask in the baseline identity; equal-length masks can name different MSUs.
            kind = f"memory:{mask}"
            for old in list(self.previous):
                if old.startswith("memory:") and old != kind:
                    del self.previous[old]
            self.sources.update({field: SOURCE for field in MEMORY_RATES})
            rates = self._rates(kind, counters, stamp)
            read = sum(rate * size for rate, (_, size, direction) in zip(rates, layout) if direction == "read")
            written = sum(rate * size for rate, (_, size, direction) in zip(rates, layout) if direction == "write")
            values.update(mem_read_bps=read, mem_write_bps=written)
            if maximum:
                values["mem_bandwidth_pct"] = 100 * (read + written) / maximum
            else:
                status["mem_bandwidth_pct"] = "maximum bandwidth unavailable; measured bytes/sec remain available"
        except _ReadError as error:
            if not isinstance(error, _Pending):
                for kind in list(self.previous):
                    if kind.startswith("memory:"):
                        del self.previous[kind]
            status.update({field: f"Intel PMT: {error}" for field in MEMORY_RATES})
        return values, status, dict(self.sources)

    def close(self):
        for node in self.nodes.values():
            node.close()
        self.previous.clear()


def pci_link(sysfs: str) -> tuple[dict, str]:
    """PCIe capability Link Status, never sysfs defaults or advertised maximum speed."""
    try:
        with (Path(sysfs) / "config").open("rb", buffering=0) as stream:
            data = stream.read(256)
        if len(data) < 64:
            return {}, "PCI configuration header unavailable"
        if not (struct.unpack_from("<H", data, 6)[0] & 0x10):
            return {}, "PCI device reports no capability list"
        offset, seen = data[0x34], set()
        for _ in range(48):
            if offset == 0:
                return {}, "PCIe capability not present"
            if offset < 0x40 or offset & 3 or offset in seen or offset + 2 > 256:
                return {}, "invalid PCI capability chain"
            if offset + 2 > len(data):
                return {}, "PCI capability read access required (configuration is truncated)"
            seen.add(offset)
            kind, next_offset = data[offset:offset + 2]
            if kind == 0x10:
                if offset + 20 > len(data):
                    return {}, "PCIe Link Status read access required (configuration is truncated)"
                status = struct.unpack_from("<H", data, offset + 18)[0]
                gen, width = status & 0xF, (status >> 4) & 0x3F
                if gen not in range(1, 8) or width not in (1, 2, 4, 8, 12, 16, 32):
                    return {}, "PCIe Link Status is unknown or link is down"
                return {"pcie_gen": gen, "pcie_width": width}, ""
            offset = next_offset
        return {}, "PCI capability chain exceeds scan limit"
    except OSError as error:
        return {}, "PCI configuration: " + _reason(error)
