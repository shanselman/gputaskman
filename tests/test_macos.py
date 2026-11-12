"""macOS backend. Fixtures follow ioreg/footprint output measured on an M4 Mac mini (macOS 26.3.1)."""

from __future__ import annotations

import json
import plistlib
import unittest
from unittest.mock import Mock, patch

from gputaskman import macos
from gputaskman.collector import Collector, ProcRow, Snapshot
from gputaskman.dxgi import Adapter
from gputaskman.pdh import ProcGpu
from gputaskman.procinfo import ProcDetail, _lib_name, _match_runtimes
from gputaskman.report import macos_memory_label, macos_note, to_dict, to_text

GiB = 1 << 30
ME, ROOT = 501, 0


def accel(util=28, alloc=2202157056):
    return [{"IOClass": "AGXAcceleratorG16G", "model": "Apple M4", "gpu-core-count": 10,
             "PerformanceStatistics": {"Device Utilization %": util, "Renderer Utilization %": util,
                                       "Tiler Utilization %": util, "Alloc system memory": alloc,
                                       "In use system memory": 13598720}}]


def client(pid, name, *times, cid=None):
    entry = {"IOUserClientCreator": f"pid {pid}, {name}",
             "AppUsage": [{"API": "Metal", "accumulatedGPUTime": t} for t in times]}
    if cid is not None:
        entry["IORegistryEntryID"] = cid
    return entry


class FakePoller:
    def __init__(self, sizes=None):
        self.sizes = sizes or {}  # pid -> bytes, for identity "same"
        self.targets = []

    def wait_pass(self, timeout):
        return True

    def start(self):
        pass

    def set_targets(self, targets):
        self.targets = targets

    def get(self, pid, identity=None):
        return self.sizes.get(pid) if identity == "same" else None

    def close(self):
        pass


class ParseTests(unittest.TestCase):
    def test_ioreg_plist_round_trip(self):
        entries = plistlib.loads(plistlib.dumps(accel(99, 5494423552)))
        dev = macos.parse_accelerator(entries)
        self.assertEqual((dev.model, dev.cores, dev.utilization, dev.allocated), ("Apple M4", 10, 99, 5494423552))
        self.assertIsNone(macos.parse_accelerator([{"model": "x"}]))
        self.assertIsNone(macos.parse_accelerator(None))

    def test_missing_device_allocation_is_unknown_not_zero(self):
        stats = accel()[0]["PerformanceStatistics"]
        del stats["Alloc system memory"]
        dev = macos.parse_accelerator([{"model": "Apple M4", "PerformanceStatistics": stats}])
        self.assertIsNone(dev.allocated)

    def test_intel_and_amd_accelerator_counters(self):
        intel = [{"IOClass": "IntelAccelerator", "model": "Intel Iris Plus",
                  "PerformanceStatistics": {"Device Utilization %": 42, "gartUsedBytes": 512 * 1024**2,
                                            "gartSizeBytes": 4 * GiB}}]
        amd = [{"IOClass": "AMDRadeonX5000_AMDNavi14GraphicsAccelerator", "model": "Radeon Pro",
                "PerformanceStatistics": {"GPU Activity(%)": 73, "inUseVidMemoryBytes": 2 * GiB,
                                          "VRAM,totalMB": 8192}}]
        with patch.object(macos, "physical_memory", return_value=16 * GiB):
            igpu = macos.parse_accelerator(intel)
            dgpu = macos.parse_accelerator(amd)
        self.assertEqual((igpu.vendor_id, igpu.utilization, igpu.allocated, igpu.total),
                         (macos.VENDOR_INTEL, 42, 512 * 1024**2, 4 * GiB))
        self.assertEqual((dgpu.vendor_id, dgpu.utilization, dgpu.allocated, dgpu.total),
                         (macos.VENDOR_AMD, 73, 2 * GiB, 8192 * 1024**2))

    def test_pci_parent_vram_data_and_startup_enumeration(self):
        entry = {"IORegistryEntryID": 42, "IOClass": "AMDRadeonX5000",
                 "PerformanceStatistics": {"inUseVidMemoryBytes": (2 * GiB).to_bytes(8, "little")}}
        pci = [{"IOClass": "IOPCIDevice", "model": b"Radeon Pro 5500M\0",
                "VRAM,totalMB": (8192).to_bytes(4, "little"), "IORegistryEntryChildren": [entry]}]
        pci = plistlib.loads(plistlib.dumps(pci))
        with patch.object(macos, "_ioreg", side_effect=[[entry], pci]):
            adapter = macos.enumerate_adapters()[0]
        self.assertEqual((adapter.name, adapter.vendor_id, adapter.dedicated_vram),
                         ("Radeon Pro 5500M", macos.VENDOR_AMD, 8 * GiB))
        dev = macos.parse_accelerator([entry], macos._pci_parents(pci))
        self.assertEqual((dev.allocated, dev.total), (2 * GiB, 8 * GiB))
        pci[0].pop("VRAM,totalMB")
        pci[0]["ATY,memsize"] = (4 * GiB).to_bytes(8, "little")
        self.assertEqual(macos.parse_accelerator([entry], macos._pci_parents(pci)).total, 4 * GiB)
        other = {"IOClass": "IOPCIDevice", "VRAM,totalMB": 4096, "IORegistryEntryChildren": [entry]}
        self.assertNotIn(42, macos._pci_parents([pci[0], other]))

    def test_malformed_and_null_metrics_are_unknown_or_fall_back(self):
        entry = {"IOClass": "IntelAccelerator",
                 "PerformanceStatistics": {"gartUsedBytes": None, "Alloc system memory": 123,
                                           "gartSizeBytes": b"\x01", "Device Utilization %": None,
                                           "GPU Activity(%)": 42.5}}
        dev = macos.parse_accelerator([entry])
        self.assertEqual((dev.allocated, dev.total, dev.utilization), (123, None, 42.5))
        amd = {"IOClass": "AMDRadeon", "VRAM,totalMB": b"\x01",
               "PerformanceStatistics": {"inUseVidMemoryBytes": "bad", "GPU Activity(%)": float("nan")}}
        dev = macos.parse_accelerator([amd])
        self.assertEqual((dev.allocated, dev.total, dev.utilization), (None, None, None))
        amd["PerformanceStatistics"]["inUseVidMemoryBytes"] = "9" * 5000
        amd["PerformanceStatistics"]["GPU Activity(%)"] = 10**5000
        dev = macos.parse_accelerator([amd])
        self.assertEqual((dev.allocated, dev.utilization), (None, None))

    def test_intel_gart_total_is_not_physical_ram_or_fallback_total(self):
        entry = {"IOClass": "IntelAccelerator", "PerformanceStatistics": {"gartUsedBytes": 123}}
        with patch.object(macos, "physical_memory", return_value=16 * GiB):
            self.assertIsNone(macos.parse_accelerator([entry]).total)
            entry["PerformanceStatistics"] = {"gartUsedBytes": None, "gartSizeBytes": 999,
                                               "Alloc system memory": 123}
            dev = macos.parse_accelerator([entry])
            self.assertEqual((dev.allocated, dev.total), (123, None))

    def test_unrecognized_accelerator_memory_stays_unknown(self):
        dev = macos.parse_accelerator([{"IOClass": "nvaccelerator",
                                        "PerformanceStatistics": {"Device Utilization %": 25,
                                                                  "Alloc system memory": GiB}}])
        self.assertEqual(dev.vendor_id, macos.VENDOR_NVIDIA)
        self.assertIsNone(dev.allocated)
        self.assertFalse(dev.process_memory_supported)

    def test_gpu_time_is_kept_per_client(self):
        # WindowServer has several Metal clients; runningboardd has one with no usage yet
        times = macos.parse_clients([client(167, "WindowServer", 53012153708, 17704333, cid=11),
                                     client(167, "WindowServer", 262833, cid=12), client(176, "runningboardd", cid=13),
                                     {"IOUserClientCreator": "garbage"}])
        self.assertEqual(times, {167: {11: 53012153708 + 17704333, 12: 262833}, 176: {13: 0}})

    def test_intel_submitter_has_direct_gpu_time(self):
        self.assertEqual(macos.parse_clients([{"IOUserClientCreator": "pid 12, app",
                                               "accumulatedGPUTime": 123, "IORegistryEntryID": 3}]),
                         {12: {3: 123}})

    def test_graphics_footprint_counts_only_graphics_categories(self):
        process = {"pid": 79379, "categories": {
            "Owned physical footprint (unmapped) (graphics)": {"dirty": 2162163712, "swapped": 0},
            "IOAccelerator (graphics)": {"dirty": 8880128, "swapped": 0},
            "IOAccelerator": {"dirty": 65536}, "MALLOC_SMALL": {"dirty": 58 << 20}}}
        self.assertEqual(macos.graphics_bytes(process), 2162163712 + 8880128)

    def test_physical_memory_without_sysconf(self):
        with patch.object(macos.os, "sysconf", side_effect=AttributeError, create=True):
            self.assertEqual(macos.physical_memory(), 0)  # Windows has no os.sysconf


class CountersTests(unittest.TestCase):
    def counters(self, sizes, owners, root=False, identities=None):
        poller = FakePoller(sizes)
        self.identities = identities if identities is not None else {}
        with patch.object(macos, "_ioreg", return_value=[]), \
                patch.object(macos, "physical_memory", return_value=16 * GiB):  # baseline sees nothing
            c = macos.MacCounters(poller=poller, uid=ME, root=root,
                                  identity=lambda pid: self.identities.get(pid, "same"),
                                  owner=lambda pid: owners.get(pid))
        c._names = {167: "WindowServer", 176: "runningboardd", 900: "backupd"}
        return c, poller

    def sample(self, c, *args, **kw):
        with patch.object(macos, "physical_memory", return_value=16 * GiB):
            return c.sample(*args, **kw)

    def test_gpu_percent_memory_and_honest_unknowns(self):
        owners = {79379: ME, 486: ME, 167: ROOT, 176: ROOT, 900: ROOT}
        c, poller = self.counters({79379: 2162163712 + 8880128}, owners)
        first = [client(79379, "python3.13", 10 * 10**9, cid=1), client(486, "wezterm-gui", 4 * 10**9, cid=2),
                 client(167, "WindowServer", 50 * 10**9, cid=3), client(176, "runningboardd", 0, cid=4),
                 client(900, "backupd", 7 * 10**9, cid=5)]
        self.sample(c, accel(), first, now=100.0)
        second = [client(79379, "python3.13", 10 * 10**9 + 1_968_000_000, cid=1),
                  client(486, "wezterm-gui", 4 * 10**9, cid=2),
                  client(167, "WindowServer", 50 * 10**9 + 40_000_000, cid=3), client(176, "runningboardd", 0, cid=4),
                  client(900, "backupd", 7 * 10**9, cid=5)]
        procs, adapters = self.sample(c, accel(99, 5494423552), second, now=102.0)
        per = procs[macos.ADAPTER_KEY]
        self.assertAlmostEqual(per[79379].gpu_pct, 98.4)  # measured: 98.4% vs Device Utilization 99%
        self.assertEqual(per[79379].dedicated, 2162163712 + 8880128)
        self.assertTrue(per[79379].attributed)
        self.assertFalse(per[486].attributed)  # own process, not measured yet: unknown, not 0
        self.assertFalse(per[167].attributed)  # root's WindowServer: GPU% only
        self.assertAlmostEqual(per[167].gpu_pct, 2.0)
        self.assertNotIn(176, per)  # idle daemons of other users stay out of the table
        self.assertNotIn(900, per)
        self.assertEqual(poller.targets, [(79379, "same", True), (486, "same", False)])  # busiest first
        self.assertEqual(adapters[macos.ADAPTER_KEY].dedicated, 5494423552)
        self.assertEqual((c.device.name, c.device.util_gpu, c.device.driver_model), ("Apple M4", 99, "Metal"))

    def test_intel_submitter_query_fallback_measures_process_gpu_time(self):
        c, _ = self.counters({}, {1: ME})
        accel_entry = [{"IOClass": "IntelAccelerator", "model": "Intel Iris",
                        "PerformanceStatistics": {"Device Utilization %": 50, "gartUsedBytes": GiB}}]
        first = [{"IOUserClientCreator": "pid 1, game", "accumulatedGPUTime": 1_000_000_000}]
        later = [{"IOUserClientCreator": "pid 1, game", "accumulatedGPUTime": 2_000_000_000}]
        with patch.object(macos, "_ioreg", side_effect=[accel_entry, [], first]):
            c.sample(now=100.0)
        with patch.object(macos, "_ioreg", side_effect=[accel_entry, [], later]):
            procs, _ = c.sample(now=102.0)
        self.assertAlmostEqual(procs[macos.ADAPTER_KEY][1].gpu_pct, 50.0)

    def test_selected_gpu_does_not_change_with_order_or_disappearance(self):
        c, poller = self.counters({1: GiB}, {1: ME})
        intel = {"IORegistryEntryID": 11, "IOClass": "IntelAccelerator", "model": "Intel Iris",
                 "PerformanceStatistics": {"gartUsedBytes": 100, "gartSizeBytes": 200}}
        amd = {"IORegistryEntryID": 22, "IOClass": "AMDRadeon", "model": "Radeon",
               "PerformanceStatistics": {"inUseVidMemoryBytes": 300, "VRAM,totalMB": 4096}}
        adapters = [Adapter(11, "Intel Iris", macos.VENDOR_INTEL, 200, 0, False),
                    Adapter(22, "Radeon", macos.VENDOR_AMD, 4 * GiB, 0, False)]
        c.select_adapter(adapters[0], multiple=True)
        with patch.object(macos, "_read_pci_parents", return_value={}):
            procs, measured = self.sample(c, [amd, intel], [client(1, "game", 123, cid=9)], now=1)
            self.assertEqual(measured[11].dedicated, 100)
            self.assertEqual(c.device.name, "Intel Iris")
            self.assertEqual(procs, {})
            self.assertEqual(poller.targets, [])
            self.assertIn("cannot be matched", c.status)
            _, measured = self.sample(c, [intel, amd], [], now=2)
            self.assertEqual(measured[11].dedicated, 100)
            _, measured = self.sample(c, [amd], [], now=3)
            self.assertEqual(measured, {})
            self.assertIsNone(c.device)
            self.assertIn("unavailable", c.status)
            c.select_adapter(adapters[1], multiple=True)
            _, measured = self.sample(c, [intel, amd], [], now=4)
            self.assertEqual(measured[22].dedicated, 300)
            self.assertEqual(c.device.name, "Radeon")
            self.assertFalse(c.process_memory_available)

    def test_selecting_initial_apple_gpu_preserves_rate_baseline(self):
        c, poller = self.counters({1: GiB}, {1: ME})
        apple = accel()
        apple[0]["IORegistryEntryID"] = 42
        self.sample(c, apple, [client(1, "game", 1_000_000_000, cid=9)], now=1)
        c.select_adapter(Adapter(42, "Apple M4", macos.VENDOR_APPLE, 16 * GiB, 0, False))
        self.assertEqual(poller.targets, [(1, "same", False)])
        procs, _ = self.sample(c, apple, [client(1, "game", 2_000_000_000, cid=9)], now=2)
        self.assertAlmostEqual(procs[42][1].gpu_pct, 100.0)

    def test_multiple_accelerators_require_distinct_registry_ids(self):
        intel = {"IORegistryEntryID": 11, "IOClass": "IntelAccelerator", "PerformanceStatistics": {}}
        amd = {"IOClass": "AMDRadeon", "PerformanceStatistics": {}}
        with patch.object(macos, "_ioreg", side_effect=[[amd, intel], []]):
            found = macos.enumerate_adapters()
        self.assertEqual([a.luid for a in found], [11])
        amd["IORegistryEntryID"] = 11
        with patch.object(macos, "_ioreg", side_effect=[[amd, intel], []]):
            self.assertEqual(macos.enumerate_adapters(), [])

    def test_selected_unidentified_intel_gpu_does_not_fall_back_by_name(self):
        c, _ = self.counters({}, {})
        c.select_adapter(Adapter(1, "Intel Iris", macos.VENDOR_INTEL, 0, 0, False))
        intel = {"IOClass": "IntelAccelerator", "model": "Intel Iris",
                 "PerformanceStatistics": {"gartUsedBytes": 10}}
        _, readings = self.sample(c, [intel], [], now=1)
        self.assertEqual(readings, {})
        self.assertIsNone(c.device)
        self.assertIn("cannot be matched", c.status)

    def test_missing_intel_total_is_unknown_in_device_json(self):
        c, _ = self.counters({}, {})
        intel = {"IOClass": "IntelAccelerator", "model": "Intel Iris",
                 "PerformanceStatistics": {"gartUsedBytes": 123}}
        self.sample(c, [intel], [], now=1)
        adapter = Adapter(1, "Intel Iris", macos.VENDOR_INTEL, 0, 0, False)
        snap = Snapshot(taken=1, adapter=adapter, adapters=[adapter], nv=c.device, driver="", cuda="",
                        vram_total=0, vram_used=123, shared_used=None, backend="metal")
        self.assertIsNone(to_dict(snap)["device"]["mem_total"])
        self.assertIn("total is unavailable", c.status)

    def test_collector_does_not_restore_stale_adapter_total(self):
        c, _ = self.counters({}, {})
        adapter = Adapter(11, "Intel Iris", macos.VENDOR_INTEL, 999, 0, False)
        c.select_adapter(adapter)
        entry = {"IORegistryEntryID": 11, "IOClass": "IntelAccelerator", "model": "Intel Iris",
                 "PerformanceStatistics": {"gartUsedBytes": 123}}
        measured = self.sample(c, [entry], [], now=1)
        collector = Collector.__new__(Collector)
        collector.adapter, collector.adapters, collector.counters = adapter, [adapter], c
        collector.drm_card, collector.ascend, collector.ascend_chip = None, None, None
        collector._mac_driver, collector.host, collector.host_status = "macOS", None, ""
        collector.scan_budget, collector.inspector = 0, Mock()
        with patch("gputaskman.collector.BACKEND", "metal"), patch.object(c, "sample", return_value=measured):
            snap = collector.sample()
        self.assertEqual(snap.vram_total, 0)
        self.assertIsNone(to_dict(snap)["vram_total"])
        self.assertIsNone(to_dict(snap)["device"]["mem_total"])

    def test_a_closed_client_does_not_hide_work_on_the_others(self):
        c, _ = self.counters({}, {1: ME})
        self.sample(c, accel(), [client(1, "a", 10 * 10**9, cid=7), client(1, "a", 1 * 10**9, cid=8)], now=1.0)
        procs, _ = self.sample(c, accel(), [client(1, "a", 2 * 10**9, cid=8)], now=2.0)  # 7 closed; 8 ran 1 s
        self.assertAlmostEqual(procs[macos.ADAPTER_KEY][1].gpu_pct, 100.0)

    def test_a_reused_pid_starts_over(self):
        c, poller = self.counters({1: 3 * GiB}, {1: ROOT}, identities={1: "old"})
        self.sample(c, accel(), [client(1, "a", 5 * 10**9, cid=7)], now=1.0)
        self.identities[1] = "same"  # PID 1 now belongs to a new process of this user
        c._owner = lambda pid: ME
        procs, _ = self.sample(c, accel(), [client(1, "a", 1 * 10**9, cid=9)], now=2.0)
        g = procs[macos.ADAPTER_KEY][1]
        self.assertAlmostEqual(g.gpu_pct, 100.0)  # its new client's 1 s, not a negative delta from the old one
        self.assertTrue(g.attributed)  # owner re-read: now measurable
        self.assertEqual(poller.targets, [(1, "same", True)])

    def test_unknown_process_identity_disables_rate_and_footprint_attribution(self):
        c, poller = self.counters({1: 3 * GiB}, {1: ME})
        self.sample(c, accel(), [client(1, "a", 5 * 10**9, cid=7)], now=1.0)
        self.identities[1] = None
        procs, _ = self.sample(c, accel(), [client(1, "a", 6 * 10**9, cid=7)], now=2.0)
        g = procs[macos.ADAPTER_KEY][1]
        self.assertNotIn("engine", g.available)
        self.assertFalse(g.attributed)
        self.assertEqual(poller.targets, [])

        self.identities[1] = "same"
        procs, _ = self.sample(c, accel(), [client(1, "a", 20 * 10**9, cid=7)], now=3.0)
        g = procs[macos.ADAPTER_KEY][1]
        self.assertNotIn("engine", g.available)  # establish a fresh baseline without a fabricated spike
        procs, _ = self.sample(c, accel(), [client(1, "a", 21 * 10**9, cid=7)], now=4.0)
        self.assertAlmostEqual(procs[macos.ADAPTER_KEY][1].gpu_pct, 100.0)

    def test_root_measures_every_process(self):
        c, poller = self.counters({}, {167: ROOT, 486: ME}, root=True)
        self.sample(c, accel(), [client(167, "WindowServer", 0, cid=1), client(486, "wezterm-gui", 0, cid=2)], now=1.0)
        self.assertEqual({pid for pid, _, _ in poller.targets}, {167, 486})

    def test_a_failed_device_read_is_not_kept_as_current(self):
        c, _ = self.counters({}, {})
        self.sample(c, accel(90), [], now=1.0)
        self.assertEqual(c.device.util_gpu, 90)
        _, adapters = self.sample(c, [], [], now=2.0)  # ioreg failed or returned nothing
        self.assertIsNone(c.device)
        self.assertNotIn(macos.ADAPTER_KEY, adapters)

    def test_missing_device_allocation_is_unknown_not_zero(self):
        stats = accel()[0]["PerformanceStatistics"]
        del stats["Alloc system memory"]
        entries = [{"model": "Apple M4", "PerformanceStatistics": stats}]
        c, _ = self.counters({}, {})
        _, adapters = self.sample(c, entries, [], now=1.0)
        self.assertIsNone(c.device.mem_used)
        self.assertIsNone(adapters[macos.ADAPTER_KEY].dedicated)

    def test_discrete_mac_vram_is_not_attributed_from_footprint(self):
        c, _ = self.counters({1: GiB}, {1: ME})
        amd = [{"IOClass": "AMDRadeonX5000", "model": "Radeon Pro",
                "PerformanceStatistics": {"Device Utilization %": 40, "inUseVidMemoryBytes": GiB,
                                          "VRAM,totalMB": 8192}}]
        self.sample(c, amd, [client(1, "game", 1_000_000_000, cid=1)], now=1.0)
        procs, adapters = self.sample(c, amd, [client(1, "game", 2_000_000_000, cid=1)], now=2.0)
        gpu = procs[macos.ADAPTER_KEY][1]
        self.assertFalse(gpu.attributed)
        self.assertNotIn("dedicated", gpu.available)
        self.assertEqual(adapters[macos.ADAPTER_KEY].dedicated, GiB)
        self.assertEqual(c.device.mem_total, 8192 * 1024**2)

    def test_unsupported_memory_still_filters_idle_other_users(self):
        for kind in ("AMDRadeonX5000", "UnknownAccelerator"):
            with self.subTest(kind=kind):
                c, poller = self.counters({}, {1: ME, 167: ROOT, 176: ROOT, 900: ROOT})
                c._name = lambda pid: {167: "WindowServer", 176: "runningboardd", 900: "backupd"}.get(pid, "")
                entries = [{"IOClass": kind, "PerformanceStatistics": {}}]
                first = [client(1, "own", 0, cid=1), client(167, "WindowServer", 0, cid=2),
                         client(176, "daemon", 0, cid=3), client(900, "busy", 0, cid=4)]
                later = [client(1, "own", 0, cid=1), client(167, "WindowServer", 0, cid=2),
                         client(176, "daemon", 0, cid=3), client(900, "busy", 100_000_000, cid=4)]
                self.sample(c, entries, first, now=1.0)
                procs, _ = self.sample(c, entries, later, now=2.0)
                per = procs[macos.ADAPTER_KEY]
                self.assertEqual(set(per), {1, 167, 900})
                self.assertTrue(all(not g.attributed for g in per.values()))
                self.assertEqual(poller.targets, [])

    def test_failed_client_read_preserves_baseline_and_footprint_targets(self):
        c, poller = self.counters({1: GiB}, {1: ME})
        first = [client(1, "python", 10 * 10**9, cid=7)]
        self.sample(c, accel(), first, now=100.0)
        previous = c._prev.copy()
        targets = poller.targets.copy()
        with patch.object(macos, "_ioreg", return_value=None):
            _, adapters = self.sample(c, accel(), None, now=101.0)
        self.assertEqual(c._prev, previous)
        self.assertFalse(c.process_memory_available)
        self.assertIn("process counters are unavailable", c.status)
        self.assertEqual(c._last, 100.0)
        self.assertEqual(poller.targets, targets)
        self.assertEqual(adapters[macos.ADAPTER_KEY].dedicated, accel()[0]["PerformanceStatistics"]["Alloc system memory"])

        later = [client(1, "python", 11 * 10**9, cid=7)]
        procs, _ = self.sample(c, accel(), later, now=102.0)
        self.assertAlmostEqual(procs[macos.ADAPTER_KEY][1].gpu_pct, 50.0)


class PollerTests(unittest.TestCase):
    def footprint(self, sizes, calls):
        def run(args, **kw):
            pid = int(args[-1])
            calls.append(pid)
            if sizes.get(pid) is None:
                return  # writes nothing, like footprint on a process it can't read
            with open(args[args.index("-j") + 1], "w", encoding="utf-8") as f:
                json.dump({"processes": [{"pid": pid, "categories": {
                    "IOAccelerator (graphics)": {"dirty": sizes[pid]}}}]}, f)
        return run

    def test_footprint_json_is_read_per_pid_and_tied_to_identity(self):
        calls = []
        p = macos.FootprintPoller(run=self.footprint({42: 3 * GiB}, calls), clock=lambda: 0.0)
        p.set_targets([(42, "a", True)])
        p.poll_once()
        self.assertEqual(p.get(42, "a"), 3 * GiB)
        self.assertIsNone(p.get(42, "b"))
        p.set_targets([(42, "b", True)])  # reused PID: the old size is gone
        self.assertIsNone(p.get(42, "a"))

    def test_idle_processes_are_measured_less_often(self):
        calls, clock = [], [0.0]
        p = macos.FootprintPoller(run=self.footprint({1: GiB, 2: GiB}, calls), clock=lambda: clock[0])
        p.set_targets([(1, "a", True), (2, "a", False)])
        p.poll_once()
        clock[0] = 2.0
        p.poll_once()
        self.assertEqual(calls, [1, 2, 1])  # idle 2 waits IDLE_EVERY seconds
        clock[0] = 2.0 + p.IDLE_EVERY
        p.poll_once()
        self.assertEqual(calls[-2:], [1, 2])

    def test_one_failure_keeps_the_last_size_and_repeated_failures_back_off(self):
        calls, sizes = [], {5: GiB}
        p = macos.FootprintPoller(run=self.footprint(sizes, calls), clock=lambda: 0.0)
        p.set_targets([(5, "a", True)])
        p.poll_once()
        sizes[5] = None  # footprint times out once
        p.poll_once()
        self.assertEqual(p.get(5, "a"), GiB)
        for _ in range(p.FAIL_LIMIT):
            p.poll_once()
        n = len(calls)
        p.poll_once()
        self.assertEqual(len(calls), n)  # backing off
        self.assertIsNone(p.get(5, "a"))

    def test_a_result_for_a_dropped_pid_is_discarded(self):
        p = macos.FootprintPoller(run=lambda *a, **k: None, clock=lambda: 0.0)
        p.set_targets([(812, "a", True)])

        def measure(pid):
            p.set_targets([])  # the process exits while footprint runs
            return 2 * GiB

        p.measure = measure
        p.poll_once()
        p.set_targets([(812, "a", True)])
        self.assertIsNone(p.get(812, "a"))


class ProcInfoTests(unittest.TestCase):
    def test_macos_libraries_and_metal_driver(self):
        names = {_lib_name(b) for b in ("libggml-metal.dylib", "libllama.dylib", "libmlx.dylib",
                                        "agxmetalg16g_b0", "libtorch_cpu.dylib", "assets.car")}
        self.assertNotIn("assets.car", names)
        self.assertEqual(_match_runtimes(names - {None}), ["llama.cpp", "ggml", "PyTorch", "MLX", "Metal"])


class SkipPathTests(unittest.TestCase):
    def test_protected_containers_and_system_files_are_never_opened(self):
        from gputaskman import procinfo
        with patch.object(procinfo.sys, "platform", "darwin"):
            self.assertTrue(procinfo._skip_path(
                "/Users/me/Library/Containers/com.apple.mediaanalysisd/Data/Library/Caches/PFSceneGeographyData.index"))
            self.assertTrue(procinfo._skip_path("/Users/me/Library/Group Containers/group.x/db"))
            self.assertTrue(procinfo._skip_path("/System/Library/Fonts/SFNS.ttf"))
            self.assertFalse(procinfo._skip_path("/Users/me/.ollama/models/blobs/sha256-abc"))
        with patch.object(procinfo.sys, "platform", "linux"):
            self.assertFalse(procinfo._skip_path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"))


class ReportTests(unittest.TestCase):
    def test_metal_header_and_json(self):
        rows = [ProcRow(79379, "python3.13", "AI", ProcGpu(dedicated=2 * GiB, engines={"gpu": 98.4}),
                        ProcDetail(79379, "python3.13")),
                ProcRow(167, "WindowServer", "compositor", ProcGpu(attributed=False, engines={"gpu": 2.0}),
                        ProcDetail(167, "WindowServer"))]
        snap = Snapshot(taken=1, adapter=Adapter(1, "Apple M4", macos.VENDOR_APPLE, 16 * GiB, 0, False),
                        adapters=[], nv=None, driver="macOS 26.3.1", cuda="",
                        vram_total=16 * GiB, vram_used=5 * GiB, shared_used=0, backend="metal", procs=rows)
        text = to_text(snap)
        self.assertIn("unified", text)
        self.assertNotIn("CUDA", text)
        ws = next(line for line in text.splitlines() if line.lstrip().startswith("167"))
        self.assertIn("n/a", ws)
        data = to_dict(snap)["processes"]
        self.assertEqual((data[0]["vram_bytes"], data[1]["vram_bytes"], data[1]["gpu_pct"]), (2 * GiB, None, 2.0))
        self.assertIsNone(data[0]["shared_bytes"])

    def test_mac_vendor_labels_and_notes(self):
        for vendor, label, note in ((macos.VENDOR_AMD, "VRAM", "Per-process VRAM is unknown"),
                                    (macos.VENDOR_INTEL, "shared system memory", "driver-reported"),
                                    (macos.VENDOR_UNKNOWN, "GPU memory", "unavailable"),
                                    (macos.VENDOR_APPLE, "unified memory", "graphics footprint")):
            with self.subTest(vendor=vendor):
                adapter = Adapter(1, "GPU", vendor, 0, 0, False)
                snap = Snapshot(taken=1, adapter=adapter, adapters=[adapter], nv=None, driver="", cuda="",
                                vram_total=0, vram_used=None, shared_used=None, backend="metal")
                self.assertEqual(macos_memory_label(snap), label)
                self.assertIn(note, macos_note(snap))
                self.assertIn(f"{label} N/A allocated / N/A", to_text(snap))
        snap.adapter = None
        self.assertEqual(macos_memory_label(snap), "GPU memory")


class DisplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_inspector_explains_unavailable_mac_memory(self):
        from gputaskman import app as app_mod
        from gputaskman.app import GpuTaskManager
        from textual.widgets import Static
        for vendor, reason in ((macos.VENDOR_AMD, "isn't measured for this adapter"),
                               (macos.VENDOR_UNKNOWN, "isn't measured for this adapter"),
                               (macos.VENDOR_INTEL, "footprint can't read"),
                               (macos.VENDOR_APPLE, "footprint can't read")):
            with self.subTest(vendor=vendor), patch.object(app_mod, "BACKEND", "metal"), \
                    patch.object(GpuTaskManager, "poll"):
                adapter = Adapter(1, "GPU", vendor, 0, 0, False)
                row = ProcRow(42, "Game", "app", ProcGpu(attributed=False), ProcDetail(42, "Game"))
                snap = Snapshot(taken=1, adapter=adapter, adapters=[adapter], nv=None, driver="", cuda="",
                                vram_total=0, vram_used=None, shared_used=None, backend="metal", procs=[row])
                app = GpuTaskManager()
                async with app.run_test(size=(180, 42)) as pilot:
                    app.apply(snap)
                    await pilot.pause()
                    text = " ".join(str(item) for item in app.query_one("#inspector", Static).content.renderables)
                    self.assertIn(reason, text)

    async def test_charts_and_engine_columns_on_metal(self):
        from gputaskman import app as app_mod
        from gputaskman.app import GpuTaskManager
        from textual.widgets import Select
        row = ProcRow(42, "Game", "3D/game", ProcGpu(dedicated=GiB, engines={"gpu": 80.0}), ProcDetail(42, "Game"))
        snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="macOS 26.3.1", cuda="",
                        vram_total=16 * GiB, vram_used=3 * GiB, shared_used=0, backend="metal", procs=[row])
        with patch.object(app_mod, "BACKEND", "metal"), patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager()
            async with app.run_test(size=(180, 42)) as pilot:
                app.apply(snap)
                await pilot.pause()
                self.assertEqual([app.CHARTS[i] for i in app.available_charts], ["GPU", "VRAM"])
                seen = []
                for _ in range(3):
                    await pilot.press("c")
                    await pilot.pause()
                    seen.append(app.chart_idx)
                self.assertEqual([app.CHARTS[i] if i >= 0 else "all" for i in seen], ["GPU", "VRAM", "all"])
                options = [v for _, v in app.query_one("#chart-mode", Select)._options]
                self.assertEqual(len(options), 3)
                table = app.query_one("#procs")
                self.assertEqual(table.get_cell("42", "GPU%").plain, "80")
                for column in ("3D", "Copy", "Video"):
                    self.assertEqual(table.get_cell("42", column).plain, "n/a")


class ClassifyTests(unittest.TestCase):
    def test_busy_metal_app_is_3d_but_the_compositor_is_not(self):
        from gputaskman.procinfo import classify
        game = ProcDetail(42, "Game")
        game.runtimes = ["Metal"]
        self.assertEqual(classify(game, {"gpu": 80.0}), "3D/game")
        self.assertEqual(classify(game, {"gpu": 3.0}), "app")
        ws = ProcDetail(167, "WindowServer")
        ws.runtimes = ["Metal"]
        self.assertEqual(classify(ws, {"gpu": 80.0}), "compositor")

    def test_handy_log_lives_in_library_logs_on_macos(self):
        import tempfile
        from pathlib import Path
        from gputaskman import handy
        with tempfile.TemporaryDirectory() as home:
            logs = Path(home) / "Library" / "Logs" / "com.pais.handy"
            logs.mkdir(parents=True)
            seen = []
            real = handy._last_model_event
            with patch.object(handy.sys, "platform", "darwin"), \
                    patch.object(handy.os.path, "expanduser", side_effect=lambda p: p.replace("~", home)), \
                    patch.object(handy, "_last_model_event", side_effect=lambda path, t: seen.append(path) or real(path, t)):
                handy.inspect_handy(0.0)
            self.assertEqual(seen, [str(logs / "handy.log")])


if __name__ == "__main__":
    unittest.main()
