"""Optional, read-only Level Zero Sysman telemetry on native 64-bit Linux.

Uses zesInit (never legacy zeInit/ZES_ENABLE_SYSMAN), exact PCI BDF matching,
and only enumeration/property/state/counter functions. The installed driver
supplies libze_loader.so.1; no driver library is distributed with gputaskman.
ABI reference: oneapi-src/level-zero include/zes_api.h (stable C API).
"""
from __future__ import annotations

import ctypes as c
import math
import os
import re
import sys

U32, U64, I32, I64, PTR = c.c_uint32, c.c_uint64, c.c_int32, c.c_int64, c.c_void_p
P32, PH = c.POINTER(U32), c.POINTER(PTR)
SOURCE = "Intel Level Zero Sysman"
MEM = ("mem_read_bps", "mem_write_bps", "mem_bandwidth_max_bps", "mem_bandwidth_pct")
PCI = ("pcie_rx_kbs", "pcie_tx_kbs")


class Address(c.Structure):
    _fields_ = [(name, U32) for name in ("domain", "bus", "device", "function")]


class Speed(c.Structure):
    _fields_ = [("gen", I32), ("width", I32), ("maxBandwidth", I64)]


class PciProperties(c.Structure):
    _fields_ = [("stype", U32), ("pNext", PTR), ("address", Address), ("maxSpeed", Speed),
                ("haveBandwidthCounters", U32), ("havePacketCounters", U32), ("haveReplayCounters", U32)]


class PciState(c.Structure):
    _fields_ = [("stype", U32), ("pNext", PTR), ("status", U32), ("qualityIssues", U32),
                ("stabilityIssues", U32), ("speed", Speed)]


class PciStats(c.Structure):
    _fields_ = [(name, U64) for name in ("timestamp", "replayCounter", "packetCounter", "rxCounter", "txCounter")] + [("speed", Speed)]


class MemProperties(c.Structure):
    _fields_ = [("stype", U32), ("pNext", PTR), ("type", U32), ("onSubdevice", U32),
                ("subdeviceId", U32), ("location", U32), ("physicalSize", U64), ("busWidth", I32), ("numChannels", I32)]


class MemBandwidth(c.Structure):
    _fields_ = [(name, U64) for name in ("readCounter", "writeCounter", "maxBandwidth", "timestamp")]


class FreqProperties(c.Structure):
    _fields_ = [("stype", U32), ("pNext", PTR), ("type", U32), ("onSubdevice", U32), ("subdeviceId", U32),
                ("canControl", U32), ("isThrottleEventSupported", U32), ("min", c.c_double), ("max", c.c_double)]


class FreqState(c.Structure):
    _fields_ = [("stype", U32), ("pNext", PTR)] + [(name, c.c_double) for name in
                ("currentVoltage", "request", "tdp", "efficient", "actual")] + [("throttleReasons", U32)]


class FanProperties(c.Structure):
    _fields_ = [("stype", U32), ("pNext", PTR), ("onSubdevice", U32), ("subdeviceId", U32), ("canControl", U32),
                ("supportedModes", U32), ("supportedUnits", U32), ("maxRPM", I32), ("maxPoints", I32)]


SIGNATURES = {
    "zesInit": [U32], "zesDriverGet": [P32, PH], "zesDeviceGet": [PTR, P32, PH],
    "zesDevicePciGetProperties": [PTR, c.POINTER(PciProperties)],
    "zesDevicePciGetState": [PTR, c.POINTER(PciState)], "zesDevicePciGetStats": [PTR, c.POINTER(PciStats)],
    "zesDeviceEnumMemoryModules": [PTR, P32, PH], "zesMemoryGetProperties": [PTR, c.POINTER(MemProperties)],
    "zesMemoryGetBandwidth": [PTR, c.POINTER(MemBandwidth)],
    "zesDeviceEnumFrequencyDomains": [PTR, P32, PH], "zesFrequencyGetProperties": [PTR, c.POINTER(FreqProperties)],
    "zesFrequencyGetState": [PTR, c.POINTER(FreqState)],
    "zesDeviceEnumFans": [PTR, P32, PH], "zesFanGetProperties": [PTR, c.POINTER(FanProperties)],
    "zesFanGetState": [PTR, U32, c.POINTER(I32)],
}


class _Error(Exception):
    pass


class _Pending(_Error):
    pass


class _Skip(_Error):
    pass


def _number(value):
    return value if math.isfinite(value) and value >= 0 else None


class Sysman:
    def __init__(self, pdev: str, api=None):
        self.api, self.handle, self.error = api, None, ""
        self.previous = {}
        self.domains, self.domain_errors = {}, {}
        self.sources = {}
        self.pci_properties = None
        if not sys.platform.startswith("linux") or c.sizeof(PTR) != 8:
            self.error = "optional Sysman provider requires 64-bit native Linux"
            return
        try:
            if os.environ.get("ZES_ENABLE_SYSMAN") == "1":
                raise _Error("legacy ZES_ENABLE_SYSMAN=1 conflicts with zesInit; start the monitor without it")
            if self.api is None:
                try:
                    self.api = c.CDLL("libze_loader.so.1")
                except OSError as error:
                    raise _Error("optional Level Zero Sysman loader is not installed or cannot be loaded") from error
            match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", pdev)
            if match is None:
                raise _Error("GPU PCI address is invalid")
            wanted = tuple(int(value, 16) for value in match.groups())
            self._call("zesInit", 0)
            matches = []
            for driver in self._enum("zesDriverGet"):
                for device in self._enum("zesDeviceGet", driver):
                    prop = PciProperties(stype=2)
                    self._call("zesDevicePciGetProperties", device, c.byref(prop))
                    address = prop.address
                    if (address.domain, address.bus, address.device, address.function) == wanted:
                        matches.append((device, prop))
            if len(matches) != 1:
                raise _Error("no unique PCI-matched Sysman device")
            self.handle, self.pci_properties = matches[0]
            for kind, enum, get, structure, stype in (
                ("memory", "zesDeviceEnumMemoryModules", "zesMemoryGetProperties", MemProperties, 0xB),
                ("frequency", "zesDeviceEnumFrequencyDomains", "zesFrequencyGetProperties", FreqProperties, 9),
                ("fan", "zesDeviceEnumFans", "zesFanGetProperties", FanProperties, 7),
            ):
                try:
                    domains = []
                    for handle in self._enum(enum, self.handle):
                        prop = structure(stype=stype)
                        self._call(get, handle, c.byref(prop))
                        if kind == "memory" and prop.location != 1:
                            continue  # system RAM is never local-memory bandwidth
                        if kind == "frequency" and prop.type != 1:
                            continue  # only the MEMORY frequency domain
                        domains.append((handle, prop))
                    if not domains:
                        raise _Error(f"driver exposes no {kind} domains for this GPU")
                    if kind == "memory" and {bool(p.onSubdevice) for _, p in domains} == {True, False}:
                        raise _Error("mixed root and subdevice memory domains cannot be summed safely")
                    self.domains[kind] = domains
                except _Error as error:
                    self.domain_errors[kind] = str(error)
        except (OSError, _Error, AttributeError) as error:
            self.handle = None
            self.error = str(error)

    def _call(self, name, *args):
        function = getattr(self.api, name, None)
        if function is None:
            raise _Error(f"{name} is not exposed by the installed loader")
        if isinstance(function, c._CFuncPtr):
            function.argtypes = SIGNATURES[name]
            function.restype = U32
        result = function(*args)
        if result:
            reason = {0x70010000: "read access denied", 0x70010001: "temporarily unavailable",
                      0x70020000: "required driver dependency unavailable", 0x78000003: "unsupported by this driver",
                      0x70000001: "device lost"}.get(result, f"error 0x{result:08x}")
            if result == 0x70000001:
                self.error = "device lost; Sysman device discovery must be refreshed"
            raise _Error(f"{name}: {reason}")

    def _enum(self, name, *prefix):
        count = U32()
        self._call(name, *prefix, c.byref(count), None)
        if count.value == 0:
            return []
        if count.value > 64:
            raise _Error(f"{name}: enumeration exceeds safety bound")
        capacity = count.value
        handles = (PTR * capacity)()
        self._call(name, *prefix, c.byref(count), handles)
        if (count.value > capacity or any(not handle for handle in handles[:count.value])
                or len(set(handles[:count.value])) != count.value):
            raise _Error(f"{name}: invalid or changed enumeration")
        return list(handles[:count.value])

    def _rates(self, key, counters, timestamp):
        old = self.previous.get(key)
        self.previous[key] = counters, timestamp
        if old is None:
            raise _Pending("waiting for a second counter sample")
        before, when = old
        if timestamp <= when or any(value < previous for value, previous in zip(counters, before)):
            raise _Pending("counter reset or stalled timestamp; collecting a new baseline")
        return tuple((value - previous) * 1e6 / (timestamp - when) for value, previous in zip(counters, before))

    def sample(self, wanted: set[str] | None = None) -> tuple[dict, dict, dict]:
        values, status = {}, {}
        wanted = set((*MEM, *PCI, "pcie_gen", "pcie_width", "mem_clock", "fan", "fan_rpm")) if wanted is None else wanted
        if self.handle is None:
            return values, {"sysman": self.error}, {}
        try:
            if not wanted.intersection(MEM):
                self.previous = {k: v for k, v in self.previous.items() if k[0] != "memory"}
                raise _Skip()
            domains = self.domains.get("memory")
            if not domains:
                raise _Error(self.domain_errors.get("memory", "no local-memory domains"))
            samples = []
            for handle, prop in domains:
                sample = MemBandwidth()
                self._call("zesMemoryGetBandwidth", handle, c.byref(sample))
                samples.append((handle, sample))
            maximum = sum(sample.maxBandwidth for _, sample in samples) if all(s.maxBandwidth for _, s in samples) else None
            if maximum:
                values["mem_bandwidth_max_bps"] = maximum
            totals, pending = [], None
            for handle, sample in samples:
                try:
                    totals.append(self._rates(("memory", handle), (sample.readCounter, sample.writeCounter), sample.timestamp))
                except _Pending as error:
                    pending = error
            self.sources.update({field: SOURCE for field in MEM})
            if pending:
                raise pending
            read, written = (sum(pair[i] for pair in totals) for i in (0, 1))
            values.update(mem_read_bps=read, mem_write_bps=written)
            if maximum:
                values["mem_bandwidth_pct"] = 100 * (read + written) / maximum
            else:
                status["mem_bandwidth_pct"] = "maximum bandwidth unavailable; bytes/sec remain available"
        except _Skip:
            pass
        except _Error as error:
            if not isinstance(error, _Pending):
                self.previous = {k: v for k, v in self.previous.items() if k[0] != "memory"}
            status.update({field: f"Sysman: {error}" for field in MEM if field not in values})
        try:
            if not wanted.intersection(PCI):
                self.previous.pop(("pci", self.handle), None)
                raise _Skip()
            if not self.pci_properties.haveBandwidthCounters:
                raise _Error("driver reports no PCIe byte counters")
            sample = PciStats()
            self._call("zesDevicePciGetStats", self.handle, c.byref(sample))
            self.sources.update({field: SOURCE for field in PCI})
            rx, tx = self._rates(("pci", self.handle), (sample.rxCounter, sample.txCounter), sample.timestamp)
            values.update(pcie_rx_kbs=rx / 1024, pcie_tx_kbs=tx / 1024)
        except _Skip:
            pass
        except _Error as error:
            if not isinstance(error, _Pending):
                self.previous.pop(("pci", self.handle), None)
            status.update({field: f"Sysman: {error}" for field in PCI})
        try:
            if not wanted.intersection(("pcie_gen", "pcie_width")):
                raise _Skip()
            state = PciState(stype=0x17)
            self._call("zesDevicePciGetState", self.handle, c.byref(state))
            if state.speed.gen not in range(1, 8) or state.speed.width not in (1, 2, 4, 8, 12, 16, 32):
                raise _Error("driver did not report live PCIe link speed")
            values.update(pcie_gen=state.speed.gen, pcie_width=state.speed.width)
            self.sources.update({field: SOURCE + " (driver-reported link state)" for field in ("pcie_gen", "pcie_width")})
        except _Skip:
            pass
        except _Error as error:
            status.update({field: f"Sysman: {error}" for field in ("pcie_gen", "pcie_width")})
        try:
            if "mem_clock" not in wanted:
                raise _Skip()
            domains = self.domains.get("frequency")
            if not domains:
                raise _Error(self.domain_errors.get("frequency", "no memory frequency domain"))
            actual = []
            for handle, prop in domains:
                state = FreqState(stype=0x1B)
                state.actual = -1
                self._call("zesFrequencyGetState", handle, c.byref(state))
                value = _number(state.actual)
                if value is None:
                    raise _Error("driver did not report actual memory frequency")
                actual.append(value)
            if len(set(actual)) != 1:
                raise _Error("memory domains report different frequencies; no single clock value")
            values["mem_clock"] = actual[0]
            self.sources["mem_clock"] = SOURCE
        except _Skip:
            pass
        except _Error as error:
            status["mem_clock"] = f"Sysman: {error}"
        try:
            if not wanted.intersection(("fan", "fan_rpm")):
                raise _Skip()
            domains = self.domains.get("fan")
            if not domains:
                raise _Error(self.domain_errors.get("fan", "no fan domain"))
            if len(domains) != 1:
                raise _Error("multiple fans; a single fan reading is ambiguous")
            handle, prop = domains[0]
            rpm, percent = I32(-1), I32(-1)
            if prop.supportedUnits & 1:
                try:
                    self._call("zesFanGetState", handle, 0, c.byref(rpm))
                    if rpm.value >= 0:
                        values["fan_rpm"] = rpm.value
                        self.sources["fan_rpm"] = SOURCE
                except _Error as error:
                    rpm.value = -1
                    status["fan_rpm"] = f"Sysman: {error}"
            if prop.supportedUnits & 2:
                try:
                    self._call("zesFanGetState", handle, 1, c.byref(percent))
                    if 0 <= percent.value <= 100:
                        values["fan"] = percent.value
                        self.sources["fan"] = SOURCE
                except _Error:
                    pass  # RPM / reported maximum can still be available.
            if "fan" not in values and prop.maxRPM > 0 and rpm.value >= 0:
                if rpm.value <= prop.maxRPM:
                    values["fan"] = round(100 * rpm.value / prop.maxRPM)
                    self.sources["fan"] = SOURCE + " (RPM / reported maximum RPM)"
            if "fan" not in values:
                raise _Error("fan percentage and maximum RPM are not reported")
        except _Skip:
            pass
        except _Error as error:
            status["fan"] = f"Sysman: {error}"
        return values, status, dict(self.sources)

    def close(self):
        self.previous.clear()  # Sysman exposes no per-device shutdown API; no compute context was created.
