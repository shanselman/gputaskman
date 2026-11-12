"""NVIDIA GPUs in TCC mode: found through NVML (DXGI does not list them) and sampled through NVML.

No TCC GPU was available when this was written. FakeNvml replays what NVIDIA documents: a TCC device answers
nvmlDeviceGetDriverModel with TCC and, unlike a WDDM device, reports each process's memory. The sizes are made up;
what is tested is discovery, ordering, and the rules that keep an unknown value from becoming a zero.
"""

import contextlib
import dataclasses
import io
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gputaskman.collector as collector_mod
from gputaskman import nvml as nvml_mod
from gputaskman import tcc as tcc_mod
from gputaskman.cli import main
from gputaskman.collector import Collector, Snapshot, ordered_adapters, select_adapter
from gputaskman.dxgi import VENDOR_AMD, VENDOR_NVIDIA, Adapter
from gputaskman.nvml import Nvml
from gputaskman.procinfo import ProcDetail
from gputaskman.recovery import LiveCollectors
from gputaskman.report import to_dict, to_text
from gputaskman.tcc import TccAdapter, TccCounters, tcc_adapters

GiB = 1 << 30
MEMORY_V2 = object()  # stands for N.nvmlMemory_v2
NOT_AVAILABLE = (1 << 64) - 1  # NVML_VALUE_NOT_AVAILABLE, as ctypes reports it


def listed_adapter(luid, vendor=VENDOR_NVIDIA, bus=1, total=16 * GiB):
    return Adapter(luid, f"GPU at {bus}", vendor, total, 0, False, (bus, 0, 0))


def tcc_adapter(bus=0x0A, name="Tesla P40", total=24 * GiB, domain=0):
    packed_bdf = bus << 8
    luid = (1 << 62) | domain << 16 | packed_bdf
    return TccAdapter(luid, name, VENDOR_NVIDIA, total, 0, False, (bus, 0, 0), domain)


def device(bus, *, model=1, pending=None, name="Tesla P40", total=24 * GiB, used=0, processes=(), bus_id=None):
    """One NVML device: model 0 is WDDM, 1 is TCC, 2 is MCDM; `processes` are (pid, usedGpuMemory) pairs."""
    return SimpleNamespace(handle=f"handle-{bus:02x}", model=model, pending=pending, name=name, total=total,
                           used=used, processes=list(processes), bus_id=bus_id or f"00000000:{bus:02X}:00.0")


class NvmlError(Exception):
    def __init__(self, value):
        super().__init__(value)
        self.value = value


class FakeNvml:
    """The pynvml calls gputaskman makes, over fixed devices. `log` records every explicit call in order."""

    nvmlMemory_v2 = MEMORY_V2

    def __init__(self):
        self.devices = []
        self.log = []
        self.memory_requests = []
        self.faults = {}  # (function, handle or index) -> the exception that call raises
        self.init_error = self.count_error = None
        self.init_delay = 0.0
        self.v2_rejected = False

    def __getattr__(self, name):  # everything gputaskman reads but these tests do not care about
        if name.startswith("__"):
            raise AttributeError(name)
        return 0 if name.startswith("NVML_") else (lambda *args: None)

    def _call(self, function, key=None):
        self.log.append((function, key))
        error = self.faults.get((function, key))
        if error is not None:
            raise error

    def _device(self, handle):
        for candidate in self.devices:
            if candidate.handle == handle:
                return candidate
        raise NvmlError(2)

    def rehandle(self, candidate, handle):
        """NVML hands out a new handle after it reacquires the device; the old one stops working."""
        candidate.handle = handle

    def nvmlInit(self):
        self._call("nvmlInit")
        if self.init_delay:
            time.sleep(self.init_delay)
        if self.init_error is not None:
            raise self.init_error

    def nvmlShutdown(self):
        self._call("nvmlShutdown")

    def nvmlDeviceGetCount(self):
        self._call("nvmlDeviceGetCount")
        if self.count_error is not None:
            raise self.count_error
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        self._call("nvmlDeviceGetHandleByIndex", index)
        return self.devices[index].handle

    def nvmlDeviceGetDriverModel(self, handle):
        self._call("nvmlDeviceGetDriverModel", handle)
        candidate = self._device(handle)
        return [candidate.model, candidate.model if candidate.pending is None else candidate.pending]

    def nvmlDeviceGetPciInfo(self, handle):
        self._call("nvmlDeviceGetPciInfo", handle)
        return SimpleNamespace(busId=self._device(handle).bus_id)

    def nvmlDeviceGetName(self, handle):
        self._call("nvmlDeviceGetName", handle)
        return self._device(handle).name

    def nvmlDeviceGetMemoryInfo(self, handle, version=None):
        self.memory_requests.append(version)
        self._call("nvmlDeviceGetMemoryInfo", handle)
        if version is not None and self.v2_rejected:
            raise NvmlError(3)
        candidate = self._device(handle)
        return SimpleNamespace(total=candidate.total, used=candidate.used, free=candidate.total - candidate.used,
                               reserved=0)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        self._call("nvmlDeviceGetComputeRunningProcesses", handle)
        return [SimpleNamespace(pid=pid, usedGpuMemory=used) for pid, used in self._device(handle).processes]

    def nvmlSystemGetDriverVersion(self):
        return "581.15"

    def nvmlSystemGetCudaDriverVersion_v2(self):
        return 13000


class TccTestCase(unittest.TestCase):
    def setUp(self):
        self.api = FakeNvml()
        for target, value in (("gputaskman.nvml.N", self.api), ("gputaskman.nvml._unsafe", False),
                              ("gputaskman.nvml.LOST", set()), ("gputaskman.tcc._found", None),
                              ("gputaskman.tcc._discovery_status", "")):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def names(self):
        return [name for name, _key in self.api.log]

    def counters(self, adapter=None):
        """A TccCounters over a real Nvml session on the fake, with the setup calls cleared from the log."""
        nvml = Nvml()
        self.api.log.clear()
        self.api.memory_requests.clear()
        return TccCounters(adapter or tcc_adapter(), nvml), nvml


class DiscoveryTests(TccTestCase):
    def test_finds_a_tcc_gpu_with_every_field(self):
        self.api.devices = [device(0x0A)]
        (found,) = tcc_adapters([])
        self.assertIsInstance(found, TccAdapter)
        self.assertEqual((found.name, found.vendor_id, found.vendor, found.dedicated_vram, found.shared_system,
                          found.software, found.pci_address),
                         ("Tesla P40", VENDOR_NVIDIA, "NVIDIA", 24 * GiB, 0, False, (0x0A, 0, 0)))

    def test_only_a_gpu_that_is_in_tcc_mode_now_counts(self):
        # WDDM, MCDM, an unknown model, a WDDM GPU that will switch to TCC at the next reboot, and the reverse.
        self.api.devices = [device(1, model=0), device(2, model=2), device(3, model=7),
                            device(4, model=0, pending=1), device(5, model=1, pending=0)]
        self.assertEqual([a.pci_address[0] for a in tcc_adapters([])], [5])

    def test_a_gpu_dxgi_lists_is_never_added_again(self):
        self.api.devices = [device(0x0A), device(0x0B)]
        listed = Adapter(7, "Tesla P40", VENDOR_NVIDIA, 24 * GiB, 0, False, (0x0A, 0, 0))
        self.assertEqual([a.pci_address for a in tcc_adapters([listed])], [(0x0B, 0, 0)])

    def test_listed_adapters_that_cannot_be_paired_hide_nothing(self):
        self.api.devices = [device(0x0A)]
        unpaired = Adapter(7, "Tesla P40", VENDOR_NVIDIA, 24 * GiB, 0, False)
        elsewhere = Adapter(8, "Tesla P40", VENDOR_NVIDIA, 24 * GiB, 0, False, (0x0B, 0, 0))
        self.assertEqual(len(tcc_adapters([unpaired, elsewhere])), 1)

    def test_a_nonzero_pci_domain_is_preserved_in_tcc_identity(self):
        self.api.devices = [device(1, bus_id="00010000:01:00.0"), device(2)]
        found = tcc_adapters([])
        self.assertEqual([(a.pci_domain, a.pci_address) for a in found], [(0x10000, (1, 0, 0)), (0, (2, 0, 0))])
        self.assertNotEqual(found[0].luid, found[1].luid)

    def test_identical_bdf_values_in_different_domains_have_distinct_ids(self):
        self.api.devices = [device(1, bus_id="00000000:01:00.0"), device(2, bus_id="00010000:01:00.0")]
        found = tcc_adapters([])
        self.assertEqual([(a.pci_domain, a.pci_address) for a in found],
                         [(0, (1, 0, 0)), (0x10000, (1, 0, 0))])
        self.assertEqual(len({a.luid for a in found}), 2)

    def test_only_domain_zero_can_be_deduplicated_against_dxgi(self):
        self.api.devices = [device(1, bus_id="00000000:01:00.0"), device(2, bus_id="00010000:01:00.0")]
        listed = Adapter(7, "Listed GPU", VENDOR_NVIDIA, 24 * GiB, 0, False, (1, 0, 0))
        found = tcc_adapters([listed])
        self.assertEqual([(a.pci_domain, a.pci_address) for a in found], [(0x10000, (1, 0, 0))])

    def test_bytes_and_lowercase_text_from_other_bindings(self):
        self.api.devices = [device(0x0A, bus_id=b"00000000:0a:00.0", name=b"Tesla P40")]
        (found,) = tcc_adapters([])
        self.assertEqual((found.name, found.pci_address), ("Tesla P40", (0x0A, 0, 0)))

    def test_never_raises_without_pynvml(self):
        with patch("gputaskman.nvml.N", None):
            self.assertEqual(tcc_adapters([]), [])
        self.assertEqual(self.api.log, [])

    def test_nvml_that_cannot_start_means_no_tcc_gpus_and_no_shutdown(self):
        self.api.devices = [device(0x0A)]
        self.api.init_error = NvmlError(9)
        self.assertEqual(tcc_adapters([]), [])
        self.assertEqual(self.api.log, [("nvmlInit", None)])

    def test_a_failing_device_count_is_survivable(self):
        self.api.devices = [device(0x0A)]
        for error, unsafe in ((NvmlError(999), False), (OSError("access violation"), True)):
            with self.subTest(error=type(error).__name__):
                self.api.log.clear()
                self.api.count_error = error
                tcc_mod._found = None
                nvml_mod._unsafe = False
                self.assertEqual(tcc_adapters([]), [])
                self.assertIs(nvml_mod._unsafe, unsafe)
                self.assertEqual(self.names().count("nvmlShutdown"), 0 if unsafe else 1)
                self.assertIn("unsafe driver fault" if unsafe else "incomplete", tcc_mod.tcc_discovery_status())

    def test_one_unreadable_device_does_not_hide_a_tcc_peer(self):
        self.api.devices = [device(1), device(2)]
        for key, error, unsafe in (
                (("nvmlDeviceGetHandleByIndex", 0), NvmlError(15), False),
                (("nvmlDeviceGetDriverModel", "handle-01"), NvmlError(3), False),
                (("nvmlDeviceGetMemoryInfo", "handle-01"), NvmlError(3), False),
                (("nvmlDeviceGetPciInfo", "handle-01"), OSError("access violation"), True)):
            with self.subTest(call=key[0]):
                self.api.faults = {key: error}
                self.api.log.clear()
                tcc_mod._found = None
                nvml_mod._unsafe = False
                self.assertEqual([a.pci_address for a in tcc_adapters([])], [(2, 0, 0)])
                self.assertIs(nvml_mod._unsafe, unsafe)  # an access violation inside nvml.dll can hang a shutdown
                self.assertEqual(self.names().count("nvmlShutdown"), 0 if unsafe else 1)
                self.assertIn("unsafe driver fault" if unsafe else "incomplete", tcc_mod.tcc_discovery_status())

    def test_nvml_is_asked_once_per_process_even_when_it_found_nothing(self):
        for devices in ([device(0x0A)], []):
            with self.subTest(devices=len(devices)):
                self.api.devices = devices
                self.api.log.clear()
                tcc_mod._found = None
                first = tcc_adapters([])
                self.assertEqual(tcc_adapters([]), first)
                self.assertEqual(self.names().count("nvmlInit"), 1)

    def test_init_comes_first_and_exactly_one_shutdown_comes_last(self):
        self.api.devices = [device(1), device(2, model=0), device(3)]
        tcc_adapters([])
        names = self.names()
        self.assertEqual((names[0], names[-1]), ("nvmlInit", "nvmlShutdown"))
        self.assertEqual((names.count("nvmlInit"), names.count("nvmlShutdown")), (1, 1))

    def test_no_shutdown_once_nvml_is_marked_unsafe(self):
        self.api.devices = [device(0x0A)]
        nvml_mod._unsafe = True
        self.assertEqual(tcc_adapters([]), [])
        self.assertEqual(self.api.log, [])
        self.assertIn("marked unsafe", tcc_mod.tcc_discovery_status())

    def test_nvml_sessions_make_no_dll_calls_after_the_unsafe_flag_is_set(self):
        nvml_mod._unsafe = True
        nvml = Nvml()
        nvml.refresh()
        self.assertEqual(nvml.sample(), [])
        nvml.close()
        self.assertEqual(self.api.log, [])

    def test_discovery_status_distinguishes_no_tcc_devices_from_a_failed_probe(self):
        self.assertEqual(tcc_adapters([]), [])
        self.assertIn("succeeded; no TCC GPUs found", tcc_mod.tcc_discovery_status())
        tcc_mod._found = None
        self.api.init_error = NvmlError("private driver detail")
        self.assertEqual(tcc_adapters([]), [])
        self.assertIn("initialization failed", tcc_mod.tcc_discovery_status())
        self.assertNotIn("private driver detail", tcc_mod.tcc_discovery_status())

    def test_a_failing_shutdown_is_reported_without_hiding_discovery(self):
        self.api.devices = [device(0x0A)]
        self.api.faults = {("nvmlShutdown", None): NvmlError(3)}
        self.assertEqual(len(tcc_adapters([])), 1)
        self.assertIn("shutdown failed", tcc_mod.tcc_discovery_status())

    def test_the_listed_adapters_filter_every_call_while_nvml_is_asked_once(self):
        self.api.devices = [device(0x0A)]
        listed = Adapter(7, "Tesla P40", VENDOR_NVIDIA, 24 * GiB, 0, False, (0x0A, 0, 0))
        self.assertEqual(len(tcc_adapters([])), 1)
        self.assertEqual(tcc_adapters([listed]), [])
        self.assertEqual(len(tcc_adapters([])), 1)
        self.assertEqual(self.names().count("nvmlInit"), 1)

    def test_concurrent_first_calls_ask_nvml_once(self):
        self.api.devices = [device(0x0A)]
        self.api.init_delay = 0.05
        barrier, results = threading.Barrier(6), []

        def ask():
            barrier.wait()
            results.append(len(tcc_adapters([])))

        threads = [threading.Thread(target=ask, daemon=True) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(results, [1] * 6)
        self.assertEqual(self.names().count("nvmlInit"), 1)

    def test_luids_are_made_from_the_pci_address_and_never_change(self):
        self.api.devices = [device(0x3B, bus_id="00000000:3B:02.1"), device(0x0A)]
        first = tcc_adapters([])
        tcc_mod._found = None
        again = tcc_adapters([])
        self.assertEqual([a.luid for a in first], [a.luid for a in again])
        self.assertEqual([a.luid for a in first], [(1 << 62) | 0x3B << 8 | 2 << 3 | 1,
                                                   (1 << 62) | 0x0A << 8])
        self.assertTrue(all(a.luid >= 1 << 62 for a in first))  # far from the small LUIDs DXGI hands out


class OrderingTests(unittest.TestCase):
    def test_tcc_gpus_sort_after_every_listed_adapter_so_indexes_never_shift(self):
        listed = [listed_adapter(1, 0x8086, 1, 2 * GiB), listed_adapter(2, VENDOR_NVIDIA, 2, 16 * GiB),
                  listed_adapter(3, VENDOR_AMD, 3, 8 * GiB)]
        tcc = tcc_adapter(total=48 * GiB)  # more memory than any of them
        ordered = ordered_adapters(listed + [tcc])
        self.assertEqual(ordered[:-1], ordered_adapters(listed))
        self.assertIs(ordered[-1], tcc)

    def test_two_tcc_gpus_sort_by_memory_then_luid(self):
        small, big, twin = tcc_adapter(0x0A, total=12 * GiB), tcc_adapter(0x0B, total=24 * GiB), tcc_adapter(0x0C, total=12 * GiB)
        self.assertEqual(ordered_adapters([small, big]), [big, small])
        self.assertEqual(ordered_adapters([twin, small]), [small, twin])

    def test_a_machine_with_only_a_tcc_gpu_selects_it(self):
        tcc = tcc_adapter()
        self.assertIs(select_adapter([tcc], 0), tcc)

    def test_gpu_index_after_the_listed_adapters_selects_the_tcc_gpu(self):
        listed, tcc = listed_adapter(1), tcc_adapter()
        self.assertIs(select_adapter([tcc, listed], 0), listed)
        self.assertIs(select_adapter([tcc, listed], 1), tcc)
        with self.assertRaisesRegex(ValueError, "out of range; 2 GPU adapter"):
            select_adapter([tcc, listed], 2)

    def test_json_describes_a_tcc_adapter_like_any_other(self):
        self.assertEqual(set(dataclasses.asdict(tcc_adapter())), set(dataclasses.asdict(listed_adapter(1))))


class CounterTests(TccTestCase):
    def test_reports_nvml_memory_for_the_adapter_and_each_process(self):
        self.api.devices = [device(0x0A, used=9 * GiB, processes=[(100, 6 * GiB), (200, 3 * GiB)])]
        counters, _ = self.counters()
        per_luid, adapters = counters.sample()
        luid = counters.adapter.luid
        self.assertEqual(list(per_luid), [luid])
        self.assertEqual({pid: g.dedicated for pid, g in per_luid[luid].items()}, {100: 6 * GiB, 200: 3 * GiB})
        for g in per_luid[luid].values():
            self.assertEqual(g.available, {"dedicated"})  # no shared, commit or engine counters exist for TCC
            self.assertTrue(g.attributed)
        self.assertEqual((adapters[luid].dedicated, adapters[luid].shared), (9 * GiB, None))

    def test_the_adapter_discovery_returns_is_the_one_the_counters_find(self):
        self.api.devices = [device(0x3B, bus_id="00000000:3B:02.1", used=GiB, processes=[(5, GiB)])]
        (found,) = tcc_adapters([])
        per_luid, adapters = TccCounters(found, Nvml()).sample()
        self.assertEqual((list(per_luid), list(adapters)), ([found.luid], [found.luid]))
        self.assertEqual(per_luid[found.luid][5].dedicated, GiB)

    def test_counters_locate_the_gpu_by_domain_when_bdf_values_match(self):
        self.api.devices = [device(1, bus_id="00000000:01:00.0", used=GiB, processes=[(5, GiB)]),
                            device(2, bus_id="00010000:01:00.0", used=2 * GiB, processes=[(6, GiB)])]
        counters = TccCounters(tcc_adapter(1, domain=0x10000), Nvml())
        per_luid, adapters = counters.sample()
        self.assertEqual(list(per_luid[counters.adapter.luid]), [6])
        self.assertEqual(adapters[counters.adapter.luid].dedicated, 2 * GiB)

    def test_memory_uses_the_v2_query_and_falls_back_to_the_plain_one(self):
        self.api.devices = [device(0x0A, used=GiB)]
        counters, _ = self.counters()
        counters.sample()
        self.assertEqual(self.api.memory_requests, [MEMORY_V2])
        self.api.memory_requests.clear()
        self.api.v2_rejected = True
        _, adapters = counters.sample()
        self.assertEqual(self.api.memory_requests, [MEMORY_V2, None])
        self.assertEqual(adapters[counters.adapter.luid].dedicated, GiB)

    def test_hidden_process_memory_is_unknown_not_zero(self):
        for hidden in (None, NOT_AVAILABLE, 1 << 63):
            with self.subTest(value=hidden):
                self.api.devices = [device(0x0A, used=GiB, processes=[(100, hidden)])]
                counters, _ = self.counters()
                (g,) = counters.sample()[0][counters.adapter.luid].values()
                self.assertFalse(g.attributed)
                self.assertEqual(g.available, set())
                self.assertIsNone(g.shown_vram)

    def test_a_measured_zero_and_the_largest_signed_value_stay_measured(self):
        self.api.devices = [device(0x0A, processes=[(100, 0), (200, (1 << 63) - 1)])]
        counters, _ = self.counters()
        procs = counters.sample()[0][counters.adapter.luid]
        self.assertEqual([(g.attributed, g.dedicated, g.available) for g in (procs[100], procs[200])],
                         [(True, 0.0, {"dedicated"}), (True, float((1 << 63) - 1), {"dedicated"})])

    def test_a_process_listed_twice_is_counted_once_at_its_larger_size(self):
        for entries in ([(100, 2 * GiB), (100, 3 * GiB)], [(100, 3 * GiB), (100, 2 * GiB)]):
            with self.subTest(entries=entries):
                self.api.devices = [device(0x0A, processes=entries)]
                counters, _ = self.counters()
                (g,) = counters.sample()[0][counters.adapter.luid].values()
                self.assertEqual(g.dedicated, 3 * GiB)

    def test_one_unknown_entry_makes_a_process_unknown_in_either_order(self):
        for entries in ([(100, 2 * GiB), (100, None)], [(100, None), (100, 2 * GiB)]):
            with self.subTest(entries=entries):
                self.api.devices = [device(0x0A, processes=entries)]
                counters, _ = self.counters()
                (g,) = counters.sample()[0][counters.adapter.luid].values()
                self.assertFalse(g.attributed)
                self.assertIsNone(g.shown_vram)

    def test_a_gpu_with_no_processes_reports_an_empty_set_not_unavailable(self):
        self.api.devices = [device(0x0A, used=GiB)]
        counters, _ = self.counters()
        per_luid, adapters = counters.sample()
        self.assertEqual(per_luid, {counters.adapter.luid: {}})
        self.assertEqual(adapters[counters.adapter.luid].dedicated, GiB)

    def test_unreadable_is_reported_as_nothing_never_as_a_partial_guess(self):
        for case in ("nvml did not start", "gpu is not in the handle list", "process list failed", "memory failed"):
            with self.subTest(case=case):
                self.api.devices = [device(0x0B if case == "gpu is not in the handle list" else 0x0A,
                                           used=GiB, processes=[(100, GiB)])]
                self.api.faults = {}
                self.api.init_error = NvmlError(9) if case == "nvml did not start" else None
                if case == "process list failed":
                    self.api.faults = {("nvmlDeviceGetComputeRunningProcesses", "handle-0a"): NvmlError(3)}
                if case == "memory failed":  # both the v2 and the plain query fail
                    self.api.faults = {("nvmlDeviceGetMemoryInfo", "handle-0a"): NvmlError(3)}
                counters, _ = self.counters()
                self.assertEqual(counters.sample(), ({}, {}))
                self.assertEqual(nvml_mod.LOST, set())  # an NVML error code is not a fault inside nvml.dll
                self.assertFalse(nvml_mod._unsafe)
                if case != "process list failed" and case != "memory failed":
                    self.assertNotIn("nvmlDeviceGetMemoryInfo", self.names())
                    self.assertNotIn("nvmlDeviceGetComputeRunningProcesses", self.names())

    def test_a_fault_inside_nvml_stops_reading_this_gpu_for_good(self):
        for function in ("nvmlDeviceGetMemoryInfo", "nvmlDeviceGetComputeRunningProcesses"):
            with self.subTest(function=function):
                nvml_mod.LOST.clear()
                nvml_mod._unsafe = False
                self.api.devices = [device(0x0A, used=GiB, processes=[(100, GiB)])]
                self.api.faults = {(function, "handle-0a"): OSError("access violation")}
                counters, _ = self.counters()
                self.assertEqual(counters.sample(), ({}, {}))
                self.assertEqual(nvml_mod.LOST, {"00000000:0A:00.0"})
                self.assertTrue(nvml_mod._unsafe)
                self.api.log.clear()
                self.api.faults = {}
                self.assertEqual(counters.sample(), ({}, {}))
                self.assertEqual(self.api.log, [])  # not one more read of a GPU whose reads faulted

    def test_a_bus_nvml_already_lost_is_matched_by_address_not_by_text(self):
        nvml_mod.LOST.add("00000000:0a:00.0")  # another reader's spelling of the same address
        self.api.devices = [device(0x0A, used=GiB, processes=[(100, GiB)])]
        counters, _ = self.counters()
        self.assertEqual(counters.sample(), ({}, {}))
        self.assertEqual(self.names(), ["nvmlDeviceGetPciInfo"])

    def test_a_fault_flag_left_by_another_gpu_does_not_mark_this_one_lost(self):
        self.api.devices = [device(0x0A, used=GiB)]
        counters, nvml = self.counters()
        nvml._faulted = True  # Nvml.sample() sets it while it reads some other GPU
        per_luid, _ = counters.sample()
        self.assertEqual(per_luid, {counters.adapter.luid: {}})
        self.assertEqual(nvml_mod.LOST, set())
        self.assertFalse(nvml_mod._unsafe)

    def test_it_finds_its_handle_again_after_nvml_reacquires_handles(self):
        self.api.devices = [device(0x0A, used=GiB)]
        counters, nvml = self.counters()
        self.assertIn(counters.adapter.luid, counters.sample()[1])
        self.api.rehandle(self.api.devices[0], "handle-new")
        self.assertEqual(counters.sample(), ({}, {}))  # the old handle no longer works
        nvml.refresh()
        self.assertIn(counters.adapter.luid, counters.sample()[1])

    def test_a_failed_pci_lookup_is_retried_on_the_next_sample(self):
        self.api.devices = [device(0x0A, used=GiB)]
        counters, _ = self.counters()
        self.api.faults = {("nvmlDeviceGetPciInfo", "handle-0a"): NvmlError(3)}
        self.assertEqual(counters.sample(), ({}, {}))
        self.api.faults = {}
        self.assertIn(counters.adapter.luid, counters.sample()[1])

    def test_the_pci_address_is_read_once(self):
        self.api.devices = [device(0x0A, used=GiB)]
        counters, _ = self.counters()
        counters.sample()
        counters.sample()
        self.assertEqual(self.names().count("nvmlDeviceGetPciInfo"), 1)

    def test_close_never_touches_the_collectors_nvml_session(self):
        self.api.devices = [device(0x0A)]
        counters, nvml = self.counters()
        counters.close()
        counters.close()
        self.assertTrue(nvml.ok)
        self.assertEqual(self.api.log, [])

    def test_nothing_is_read_once_the_nvml_session_is_closed(self):
        self.api.devices = [device(0x0A, used=GiB, processes=[(100, GiB)])]
        counters, nvml = self.counters()
        self.assertIn(counters.adapter.luid, counters.sample()[1])
        nvml.close()  # keeps its old handles, which are not valid after nvmlShutdown
        self.api.log.clear()
        self.assertEqual(counters.sample(), ({}, {}))
        self.assertEqual(self.api.log, [])

    def test_each_gpu_sees_only_its_own_processes(self):
        self.api.devices = [device(1, model=0, processes=[(10, GiB)]),
                            device(0x0A, used=3 * GiB, processes=[(20, 2 * GiB)]),
                            device(0x0B, used=5 * GiB, processes=[(30, 4 * GiB)])]
        nvml = Nvml()
        first, second = TccCounters(tcc_adapter(0x0A), nvml), TccCounters(tcc_adapter(0x0B), nvml)
        for counters, pid, used, memory in ((first, 20, 2 * GiB, 3 * GiB), (second, 30, 4 * GiB, 5 * GiB)):
            per_luid, adapters = counters.sample()
            luid = counters.adapter.luid
            self.assertEqual({p: g.dedicated for p, g in per_luid[luid].items()}, {pid: used})
            self.assertEqual(adapters[luid].dedicated, memory)


class CollectorTests(TccTestCase):
    def collector(self, adapter=None, *, listed=()):
        adapter = adapter or tcc_adapter()
        c = Collector.__new__(Collector)
        c.adapter = adapter
        c.adapters = [*listed, adapter]
        c.gpu_index, c.scan_budget, c.host, c.host_status = len(listed), None, None, ""
        c.nvml = Nvml()
        c.counters = TccCounters(adapter, c.nvml)
        c.adl, c.amdsmi = Mock(), Mock()
        c.inspector, c.prober = Mock(), Mock()
        c.inspector.get.side_effect = lambda pid: ProcDetail(pid, f"app{pid}.exe")
        c.inspector.needs_scan.return_value = False
        return c

    def sample(self, c):
        with patch("gputaskman.collector.BACKEND", "wddm"), patch("gputaskman.collector.sys.platform", "win32"):
            return c.sample()

    def test_a_tcc_gpu_gets_its_memory_and_processes_from_nvml(self):
        self.api.devices = [device(0x0A, used=9 * GiB, processes=[(200, 2 * GiB), (100, 6 * GiB)])]
        snap = self.sample(self.collector())
        self.assertEqual((snap.nv.driver_model, snap.nv.mem_total, snap.vram_total), ("TCC", 24 * GiB, 24 * GiB))
        self.assertEqual((snap.driver, snap.cuda, snap.backend), ("581.15", "13.0", "wddm"))
        self.assertEqual((snap.vram_used, snap.shared_used), (9 * GiB, None))
        self.assertTrue(snap.counter_adapter_matched)
        self.assertIn("TCC compute-only", snap.counter_status)
        self.assertEqual([r.pid for r in snap.procs], [100, 200])  # largest first
        self.assertEqual(snap.attributed, 8 * GiB)
        data = to_dict(snap)
        self.assertEqual({p["pid"]: p["vram_bytes"] for p in data["processes"]}, {100: 6 * GiB, 200: 2 * GiB})
        for process in data["processes"]:
            for key in ("shared_bytes", "committed_bytes", "paged_out_bytes", "gpu_pct", "engines"):
                self.assertIsNone(process[key], key)  # no such counters exist for a TCC GPU: unknown, not zero
        self.assertIsNone(data["shared_used"])
        self.assertIn("TCC compute-only", to_text(snap))

    def test_it_is_told_apart_from_a_wddm_gpu_in_the_same_machine(self):
        self.api.devices = [device(1, model=0, name="GTX 1080 Ti", total=11 * GiB, used=2 * GiB),
                            device(0x0A, name="P102-100", total=10 * GiB, used=GiB, processes=[(7, GiB)])]
        snap = self.sample(self.collector(tcc_adapter(0x0A, "P102-100", 10 * GiB), listed=[listed_adapter(1, bus=1)]))
        self.assertEqual((snap.nv.name, snap.nv.driver_model, snap.vram_used), ("P102-100", "TCC", GiB))
        self.assertEqual([(r.pid, r.gpu.dedicated) for r in snap.procs], [(7, GiB)])

    def test_device_telemetry_matches_tcc_adapter_by_domain(self):
        self.api.devices = [device(1, bus_id="00010000:01:00.0", total=24 * GiB, used=GiB)]
        snap = self.sample(self.collector(tcc_adapter(1, domain=0x10000)))
        self.assertEqual((snap.nv.driver_model, snap.nv.mem_total, snap.vram_used),
                         ("TCC", 24 * GiB, GiB))

    def test_hidden_process_memory_leaves_the_total_unknown_but_the_adapter_measured(self):
        self.api.devices = [device(0x0A, used=9 * GiB, processes=[(100, 6 * GiB), (200, None)])]
        snap = self.sample(self.collector())
        self.assertIsNone(snap.attributed)
        self.assertEqual(snap.vram_used, 9 * GiB)
        data = to_dict(snap)
        self.assertEqual({p["pid"]: p["vram_bytes"] for p in data["processes"]}, {100: 6 * GiB, 200: None})
        self.assertIsNone(data["attributed"])

    def test_unreadable_nvml_is_explained_not_guessed(self):
        self.api.init_error = NvmlError(9)
        c = self.collector()
        snap = self.sample(c)
        self.assertFalse(snap.counter_adapter_matched)
        self.assertIn("NVML could not read this TCC GPU", snap.counter_status)
        self.assertEqual(snap.procs, [])
        self.assertIsNone(snap.vram_used)
        self.assertIsNone(snap.attributed)
        self.assertEqual(snap.vram_total, c.adapter.dedicated_vram)

    def test_a_lost_gpu_says_to_restart(self):
        nvml_mod.LOST.add("00000000:0A:00.0")
        self.api.devices = [device(0x0A, used=GiB)]
        snap = self.sample(self.collector())
        self.assertTrue(snap.nv_lost)
        self.assertIn("restart gputaskman", snap.counter_status)
        self.assertIsNone(snap.vram_used)


class CollectorSetupTests(TccTestCase):
    NAMES = ("GpuCounters", "Nvml", "AmdSmi", "Adl", "ProcInspector", "ServerProber")

    def mocks(self, stack):
        mocks = {name: stack.enter_context(patch(f"gputaskman.collector.{name}")) for name in self.NAMES}
        stack.enter_context(patch("gputaskman.collector.BACKEND", "wddm"))
        return mocks

    def test_a_tcc_gpu_never_creates_windows_counters(self):
        with contextlib.ExitStack() as stack:
            mocks = self.mocks(stack)
            c = Collector(0, adapters=[tcc_adapter()], host=False)
        mocks["GpuCounters"].assert_not_called()
        mocks["Nvml"].assert_called_once_with()  # one NVML session, shared with the counters
        self.assertIsInstance(c.counters, TccCounters)
        self.assertIs(c.counters.nvml, mocks["Nvml"].return_value)
        self.assertIs(c.nvml, mocks["Nvml"].return_value)

    def test_a_listed_nvidia_gpu_still_uses_windows_counters(self):
        with contextlib.ExitStack() as stack:
            mocks = self.mocks(stack)
            c = Collector(0, adapters=[listed_adapter(1)], host=False)
        mocks["GpuCounters"].assert_called_once_with()
        self.assertIs(c.counters, mocks["GpuCounters"].return_value)

    def test_a_failure_while_building_closes_the_nvml_session_it_opened(self):
        with contextlib.ExitStack() as stack:
            mocks = self.mocks(stack)
            mocks["Adl"].side_effect = OSError("no ADL")
            with self.assertRaises(OSError):
                Collector(0, adapters=[tcc_adapter()], host=False)
        mocks["Nvml"].return_value.close.assert_called_once()
        mocks["GpuCounters"].assert_not_called()


class CliTests(unittest.TestCase):
    def test_list_gpus_shows_a_tcc_gpu_by_pci_address_after_the_listed_ones(self):
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--list-gpus"]), \
                patch("gputaskman.collector.enumerate_adapters",
                      return_value=[tcc_adapter(0x0A), listed_adapter(0x1234ABCD)]), \
                patch("gputaskman.tcc.tcc_discovery_status", return_value=""), \
                contextlib.redirect_stdout(output):
            main()
        self.assertEqual(output.getvalue().splitlines(), [
            "0: GPU at 1 (NVIDIA, LUID 0x000000001234abcd)",
            "1: Tesla P40 (NVIDIA, TCC compute-only, PCI 0a:00.0)"])

    def test_list_gpus_reports_nvml_discovery_status_on_stderr(self):
        output, error = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["gputaskman", "--list-gpus"]), \
                patch("gputaskman.collector.enumerate_adapters", return_value=[]), \
                patch("gputaskman.tcc.tcc_discovery_status",
                      return_value="NVML initialization failed; TCC GPUs could not be checked."), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            main()
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(error.getvalue().strip(),
                         "TCC discovery: NVML initialization failed; TCC GPUs could not be checked.")

    def test_list_gpus_displays_full_domain_for_tcc_gpu(self):
        output = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--list-gpus"]), \
                patch("gputaskman.collector.enumerate_adapters", return_value=[tcc_adapter(1, domain=0x10000)]), \
                patch("gputaskman.tcc.tcc_discovery_status", return_value=""), \
                contextlib.redirect_stdout(output):
            main()
        self.assertEqual(output.getvalue().strip(),
                         "0: Tesla P40 (NVIDIA, TCC compute-only, PCI 00010000:01:00.0)")


@unittest.skipUnless(sys.platform == "win32", "the DXGI wrapper exists only on Windows")
class WindowsEnumerationTests(unittest.TestCase):
    def test_tcc_gpus_come_after_the_adapters_dxgi_lists(self):
        listed, tcc = [listed_adapter(1)], tcc_adapter()
        with patch("gputaskman.collector.dxgi.enumerate_adapters", return_value=listed) as dxgi, \
                patch("gputaskman.collector.tcc_adapters", return_value=[tcc]) as found:
            self.assertEqual(collector_mod.enumerate_adapters(strict=True), [*listed, tcc])
        dxgi.assert_called_once_with(strict=True)
        found.assert_called_once_with(listed)

    def test_without_tcc_gpus_the_dxgi_list_is_returned_unchanged(self):
        listed = [listed_adapter(1)]
        with patch("gputaskman.collector.dxgi.enumerate_adapters", return_value=listed), \
                patch("gputaskman.collector.tcc_adapters", return_value=[]):
            self.assertEqual(collector_mod.enumerate_adapters(), listed)

    def test_a_strict_dxgi_failure_propagates_before_nvml_is_asked(self):
        with patch("gputaskman.collector.dxgi.enumerate_adapters", side_effect=OSError("DXGI failed")), \
                patch("gputaskman.collector.tcc_adapters") as found:
            with self.assertRaises(OSError):
                collector_mod.enumerate_adapters(strict=True)
        found.assert_not_called()


class RecoveryTests(TccTestCase):
    def test_a_tcc_slot_keeps_its_collector_while_a_listed_gpu_is_re_enabled(self):
        wddm, tcc = listed_adapter(1), tcc_adapter()
        topology, created = [tcc, wddm], []

        def factory(index, *, adapters, scan_budget):
            adapter = adapters[index]
            collector = Mock(adapter=adapter, scan_budget=scan_budget)
            collector.sample.side_effect = lambda: Snapshot(1, adapter, adapters, None, "", "", adapter.dedicated_vram,
                                                            0, 0, counter_adapter_matched=True)
            created.append(collector)
            return collector

        live = LiveCollectors(None, collector_factory=factory, discover=lambda **_kwargs: list(topology))
        self.addCleanup(live.close)
        self.assertEqual([slot.adapter for slot in live.slots], [wddm, tcc])  # TCC last
        tcc_collector = created[1]
        topology[:] = [tcc, listed_adapter(11)]  # Windows re-enabled the listed GPU under a new LUID
        snaps = live.sample(now=live.next_discovery)
        self.assertEqual([s.adapter.luid for s in snaps], [11, tcc.luid])
        self.assertIs(live.slots[1].collector, tcc_collector)
        tcc_collector.close.assert_not_called()
        tcc_collector.nvml.refresh.assert_called_once()  # so TccCounters finds its handle again


if __name__ == "__main__":
    unittest.main()
