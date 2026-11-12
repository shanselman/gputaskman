"""NVIDIA GPUs in TCC mode: compute-only, so Windows lists no adapter and publishes no GPU counters for them."""

from __future__ import annotations

import re
import threading

from . import nvml as nvml_mod
from .dxgi import VENDOR_NVIDIA, Adapter
from .nvml import Nvml
from .pdh import AdapterGpu, ProcGpu

NVML_DRIVER_TCC = 1


class TccAdapter(Adapter):
    """A TCC-mode NVIDIA GPU, found through NVML because DXGI does not list it. It adds no fields, so --json
    looks the same for every adapter. Its luid is made up from the PCI address, so it is the same on every run
    (`diff` pairs adapters by it)."""

    def __init__(self, luid: int, name: str, vendor_id: int, dedicated_vram: int, shared_system: int, software: bool,
                 pci_address: tuple[int, int, int] | None = None, pci_domain: int = 0) -> None:
        super().__init__(luid, name, vendor_id, dedicated_vram, shared_system, software, pci_address)
        self._pci_domain = pci_domain

    @property
    def pci_domain(self) -> int:
        return self._pci_domain

    def matches_pci(self, value) -> bool:
        return _pci_identity(value) == (self.pci_domain, self.pci_address)


_lock = threading.Lock()
_found: list[TccAdapter] | None = None  # NVML is asked once per process
_discovery_status = ""


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _value(v) -> int | None:
    # NVML reports NVML_VALUE_NOT_AVAILABLE (unsigned -1) or None when a process's memory is hidden.
    if v is None or v >= (1 << 63):
        return None
    return int(v)


def _pci_identity(value) -> tuple[int, tuple[int, int, int]] | None:
    text = _text(value)
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)", text)
    if not match:
        return None
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    if domain > 0xFFFFFFFF or bus > 0xFF or device > 0x1F or function > 7:
        return None
    return domain, (bus, device, function)


def _nvml_call(N, name: str, *args):
    if nvml_mod._unsafe:
        raise RuntimeError("NVML is marked unsafe")
    return getattr(N, name)(*args)


def _read_device(N, index: int) -> TccAdapter | None:
    handle = _nvml_call(N, "nvmlDeviceGetHandleByIndex", index)
    if _nvml_call(N, "nvmlDeviceGetDriverModel", handle)[0] != NVML_DRIVER_TCC:
        return None
    identity = _pci_identity(_nvml_call(N, "nvmlDeviceGetPciInfo", handle).busId)
    if identity is None:
        raise ValueError("NVML returned an invalid PCI address")
    domain, bdf = identity
    packed_bdf = bdf[0] << 8 | bdf[1] << 3 | bdf[2]
    luid = (1 << 62) | domain << 16 | packed_bdf
    return TccAdapter(luid=luid, name=_text(_nvml_call(N, "nvmlDeviceGetName", handle)),
                      vendor_id=VENDOR_NVIDIA, dedicated_vram=_nvml_call(N, "nvmlDeviceGetMemoryInfo", handle).total,
                      shared_system=0, software=False, pci_address=bdf, pci_domain=domain)


def _discover() -> list[TccAdapter]:
    global _discovery_status
    N = nvml_mod.N
    if N is None:
        _discovery_status = "NVML is unavailable; TCC GPUs could not be checked."
        return []
    if nvml_mod._unsafe:
        _discovery_status = "NVML is marked unsafe; TCC GPUs were not checked."
        return []
    try:
        _nvml_call(N, "nvmlInit")
    except Exception:
        _discovery_status = "NVML initialization failed; TCC GPUs could not be checked."
        return []
    found, faulted, failed, shutdown_failed = [], False, False, False
    try:
        for i in range(_nvml_call(N, "nvmlDeviceGetCount")):
            if nvml_mod._unsafe:
                failed = True
                break
            try:
                adapter = _read_device(N, i)
            except Exception as error:  # one unreadable device must not hide a TCC peer
                failed = True
                faulted |= isinstance(error, OSError)  # an access violation inside nvml.dll
                continue
            if adapter is not None:
                found.append(adapter)
    except Exception as error:
        failed = True
        faulted |= isinstance(error, OSError)
    if faulted:
        nvml_mod.mark_unsafe()  # nvmlShutdown() can hang after a fault; process exit cleans up
    if not nvml_mod._unsafe:
        try:
            _nvml_call(N, "nvmlShutdown")
        except Exception:
            shutdown_failed = True
    if nvml_mod._unsafe:
        _discovery_status = "NVML encountered an unsafe driver fault; TCC discovery is incomplete."
    elif failed:
        _discovery_status = "NVML TCC discovery was incomplete; one or more devices could not be read."
    elif found:
        _discovery_status = f"NVML TCC discovery succeeded; found {len(found)} TCC GPU(s)."
    else:
        _discovery_status = "NVML TCC discovery succeeded; no TCC GPUs found."
    if shutdown_failed:
        _discovery_status += " NVML shutdown failed."
    return found


def tcc_discovery_status() -> str:
    """A safe, user-facing summary of the one-time TCC discovery attempt."""
    return _discovery_status


def tcc_adapters(listed: list[Adapter]) -> list[Adapter]:
    """TCC GPUs that DXGI (`listed`) does not show. NVML is asked once, at the first call: a TCC GPU added or
    removed while gputaskman runs needs a restart."""
    global _found
    with _lock:
        if _found is None:
            _found = _discover()
        found = list(_found)
    seen = {a.pci_address for a in listed if a.pci_address}
    return [a for a in found if a.pci_domain != 0 or a.pci_address not in seen]


class TccCounters:
    """pdh.GpuCounters' interface for one TCC GPU. Windows publishes no GPU counters for it, so adapter memory and
    per-process memory come from NVML. NVML documents per-process memory as never available under WDDM (Windows
    manages it); under TCC the NVIDIA driver does, so it is expected here, and an entry without it stays unknown.
    It uses the collector's NVML session and never opens or closes one."""

    def __init__(self, adapter: TccAdapter, nvml: Nvml) -> None:
        self.adapter = adapter
        self.nvml = nvml
        self._handles: list | None = None  # the list `_located` was resolved from
        self._located: tuple[object, str] | None = None  # this GPU's (NVML handle, PCI bus ID)

    def close(self) -> None:
        pass

    def _locate(self) -> tuple[object, str] | None:
        nvml = self.nvml
        if self._handles is not nvml.handles:  # Nvml.refresh() builds a new list whenever it reacquires handles
            self._handles, self._located = nvml.handles, None
        if self._located is None:
            for _index, handle in self._handles:
                pci = nvml._read(nvml_mod.N.nvmlDeviceGetPciInfo, handle)
                bus = _text(pci.busId) if pci is not None else ""
                if _pci_identity(bus) == (self.adapter.pci_domain, self.adapter.pci_address):
                    self._located = (handle, bus)
                    break
        return self._located

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        nvml = self.nvml
        located = self._locate() if nvml.ok else None
        if located is None or any(_pci_identity(bus) == (self.adapter.pci_domain, self.adapter.pci_address)
                                  for bus in nvml_mod.LOST):
            return {}, {}
        handle, bus = located
        N = nvml_mod.N
        nvml._faulted = False  # Nvml._read: an OSError is an access violation inside nvml.dll
        memory = (nvml._read(N.nvmlDeviceGetMemoryInfo, handle, N.nvmlMemory_v2)
                  or nvml._read(N.nvmlDeviceGetMemoryInfo, handle))
        running = nvml._read(N.nvmlDeviceGetComputeRunningProcesses, handle)
        if nvml._faulted:
            nvml_mod.LOST.add(bus)
            nvml_mod.mark_unsafe()
            return {}, {}
        if memory is None or running is None:  # unreadable is not empty: the collector says why
            return {}, {}
        procs: dict[int, ProcGpu] = {}
        for p in running:
            g = procs.setdefault(p.pid, ProcGpu(available={"dedicated"}))
            used = _value(getattr(p, "usedGpuMemory", None))
            if used is None:
                g.attributed, g.available = False, set()
            else:
                g.dedicated = max(g.dedicated, float(used))
        luid = self.adapter.luid
        return {luid: procs}, {luid: AdapterGpu(dedicated=float(memory.used))}
