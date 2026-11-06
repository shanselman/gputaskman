from __future__ import annotations

import os
import tempfile
import unittest
from itertools import count
from pathlib import Path
from unittest.mock import patch

from gputaskman import amdgpu, linux
from gputaskman.collector import Collector, ordered_adapters
from gputaskman.report import to_dict, to_text


@unittest.skipIf(os.name == "nt", "Linux sysfs fixtures use POSIX PCI paths and directory symlinks")
class AmdGpuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.device = self.root / "devices" / "0000:03:00.0"
        self.device.mkdir(parents=True)
        self.driver = self.root / "drivers" / "amdgpu"
        self.driver.mkdir(parents=True)
        (self.device / "driver").symlink_to(self.driver, target_is_directory=True)
        self.drm = self.root / "drm"
        self.proc = self.root / "proc"
        self.proc.mkdir()
        proc_patch = patch.object(amdgpu, "PROC_ROOT", self.proc)
        proc_patch.start()
        self.addCleanup(proc_patch.stop)
        for name in ("card0", "card0-DP-1", "card1"):
            card = self.drm / name
            card.mkdir(parents=True)
            (card / "device").symlink_to(self.device, target_is_directory=True)
        self.write("vendor", "0x1002")
        self.write("product_name", "AMD test GPU")
        self.write("mem_info_vram_total", str(8 << 30))
        self.write("mem_info_vram_used", str(2 << 30))
        self.write("gpu_busy_percent", "0")
        self.write("hwmon/hwmon0/name", "amdgpu")
        self.write("hwmon/hwmon0/temp1_input", "48500")
        self.write("hwmon/hwmon0/power1_average", "25000000")
        self.write("hwmon/hwmon0/freq1_input", "1200000000")
        self.patch = patch.object(amdgpu, "DRM_ROOT", self.drm)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, relative, value):
        path = self.device / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="ascii")

    def fdinfo(self, pid, fd, client=1, vram="100 MiB", busy=0, extra="", bus="0000:03:00.0", start=1):
        process = self.proc / str(pid)
        (process / "fd").mkdir(parents=True, exist_ok=True)
        (process / "fdinfo").mkdir(exist_ok=True)
        link = process / "fd" / str(fd)
        if not link.is_symlink():
            link.symlink_to("/dev/dri/renderD128")
        (process / "stat").write_text(f"{pid} (test client) S " + "0 " * 18 + str(start), encoding="ascii")
        engine = f"drm-engine-gfx: {busy} ns\n" if busy is not None else ""
        text = (f"drm-driver: amdgpu\ndrm-client-id: {client}\ndrm-pdev: {bus}\n"
                f"drm-resident-vram: {vram}\ndrm-resident-gtt: 0 KiB\n" + engine + extra)
        (process / "fdinfo" / str(fd)).write_text(text, encoding="ascii")

    def read_processes(self, reader, timestamp):
        with patch.object(amdgpu.time, "monotonic_ns", return_value=timestamp):
            return reader.sample(None)

    def test_fdinfo_units_aliases_and_invalid_identifiers(self):
        text = ("drm-driver: amdgpu\ndrm-client-id: 7\ndrm-pdev: 0000:03:00.0\n"
                "drm-memory-vram: 12 KiB\ndrm-resident-gtt: 1 MiB\ndrm-engine-gfx: 12 ns\n")
        c = amdgpu.parse_fdinfo(text)
        self.assertEqual((c.device, c.client, c.vram, c.gtt), (0x3000, 7, 12 * 1024, 1 << 20))
        self.assertEqual(c.engines, {"gfx": 12})
        self.assertEqual(amdgpu.parse_fdinfo(text + "drm-resident-vram: 0\n").vram, 0)
        self.assertIsNone(amdgpu.parse_fdinfo(text + "drm-resident-vram: -1 KiB\n").vram)
        for changed in (text.replace("amdgpu", "i915"), text.replace("drm-client-id: 7", "drm-client-id: -1"),
                        text.replace("0000:03:00.0", "missing"), text.replace("drm-client-id: 7\n", "")):
            self.assertIsNone(amdgpu.parse_fdinfo(changed))

    def test_duplicate_fds_are_counted_once_and_clients_are_combined(self):
        self.fdinfo(10, 3, busy=0)
        self.fdinfo(10, 4, busy=0)
        self.fdinfo(10, 5, client=2, vram="200 MiB", busy=0)
        reader = amdgpu.DrmProcesses([0x3000])
        g = self.read_processes(reader, 1_000_000_000)[0x3000][10]
        self.assertEqual(g.drm_vram, 300 << 20)
        self.assertNotIn("engine", g.available)
        self.assertEqual(g.drm_gtt, 0)
        self.fdinfo(10, 3, busy=200_000_000)
        self.fdinfo(10, 4, busy=200_000_000)
        self.fdinfo(10, 5, client=2, vram="200 MiB", busy=300_000_000)
        g = self.read_processes(reader, 2_000_000_000)[0x3000][10]
        self.assertEqual(g.engines["3d"], 50)
        self.assertIn("engine", g.available)
        self.assertFalse(g.attributed)
        self.assertNotIn("dedicated", g.available)
        self.assertIsNone(g.inferred)
        self.assertEqual(g.shown_vram, 300 << 20)

    def test_shared_client_handles_do_not_invent_a_process_owner(self):
        self.fdinfo(10, 3)
        self.fdinfo(20, 3)
        rows = amdgpu.DrmProcesses([0x3000]).sample(None)[0x3000]
        self.assertEqual(set(rows), {10, 20})
        for g in rows.values():
            self.assertIsNone(g.drm_vram)
            self.assertNotIn("engine", g.available)

    def test_amdgpu_omitted_zero_engines_need_observed_device_support(self):
        self.fdinfo(10, 3, busy=None)
        reader = amdgpu.DrmProcesses([0x3000])
        self.read_processes(reader, 1_000_000_000)
        self.assertNotIn("engine", self.read_processes(reader, 2_000_000_000)[0x3000][10].available)
        self.fdinfo(20, 3, client=2, busy=1_000_000)
        g = self.read_processes(reader, 3_000_000_000)[0x3000][10]
        self.assertIn("engine", g.available)
        self.assertEqual(g.gpu_pct, 0)
        self.fdinfo(10, 3, busy=100_000_000)
        self.assertEqual(self.read_processes(reader, 4_000_000_000)[0x3000][10].gpu_pct, 10)

    def test_engine_capacity_zero_rates_and_pid_reuse(self):
        self.fdinfo(10, 3, extra="drm-engine-capacity-gfx: 2\n")
        reader = amdgpu.DrmProcesses([0x3000])
        self.read_processes(reader, 1_000_000_000)
        self.fdinfo(10, 3, busy=400_000_000, extra="drm-engine-capacity-gfx: 2\n")
        self.assertEqual(self.read_processes(reader, 2_000_000_000)[0x3000][10].gpu_pct, 20)
        g = self.read_processes(reader, 3_000_000_000)[0x3000][10]
        self.assertIn("engine", g.available)
        self.assertEqual(g.gpu_pct, 0)
        self.fdinfo(10, 3, busy=800_000_000, start=2)
        g = self.read_processes(reader, 4_000_000_000)[0x3000][10]
        self.assertNotIn("engine", g.available)
        self.assertEqual(g.drm_vram, 100 << 20)

    def test_counter_regression_waits_for_the_previous_high_water_mark(self):
        reader = amdgpu.DrmProcesses([0x3000])
        for seconds, busy, rate in ((1, 1_000_000_000, None), (2, 500_000_000, 0),
                                    (3, 800_000_000, 0), (4, 1_200_000_000, 20)):
            self.fdinfo(10, 3, busy=busy)
            g = self.read_processes(reader, seconds * 1_000_000_000)[0x3000][10]
            self.assertEqual(g.gpu_pct if "engine" in g.available else None, rate)

    def test_budget_rotates_to_other_pids_without_publishing_partial_counters(self):
        for fd in range(10):
            self.fdinfo(10, fd)
        self.fdinfo(20, 3, client=2)
        reader = amdgpu.DrmProcesses([0x3000])
        ticks = count()
        with patch.object(amdgpu.time, "monotonic", side_effect=lambda: next(ticks) * 0.01):
            self.assertEqual(reader.sample(0.05), {})
            rows = reader.sample(0.05)[0x3000]
        self.assertIn(20, rows)
        self.assertIsNone(rows[20].drm_vram)
        self.assertIn("partial", reader.status)

    def test_missing_engine_does_not_hide_measured_memory_and_wrong_gpu_is_ignored(self):
        self.fdinfo(10, 3, extra="drm-engine-gfx: not available\n")
        self.fdinfo(20, 3, bus="0000:0e:00.0")
        rows = amdgpu.DrmProcesses([0x3000]).sample(None)[0x3000]
        self.assertEqual(set(rows), {10})
        self.assertEqual(rows[10].drm_vram, 100 << 20)
        self.assertNotIn("engine", rows[10].available)

    def test_engine_availability_is_independent_of_memory_availability(self):
        self.fdinfo(10, 3, vram="unavailable")
        reader = amdgpu.DrmProcesses([0x3000])
        self.read_processes(reader, 1_000_000_000)
        self.fdinfo(10, 3, vram="unavailable", busy=100_000_000)
        g = self.read_processes(reader, 2_000_000_000)[0x3000][10]
        self.assertIsNone(g.shown_vram)
        self.assertIn("engine", g.available)
        self.assertEqual(g.gpu_pct, 10)

    def test_unreadable_drm_descriptor_does_not_publish_a_partial_process_total(self):
        self.fdinfo(10, 3)
        self.fdinfo(10, 4, client=2)
        (self.proc / "10/fdinfo/4").unlink()
        reader = amdgpu.DrmProcesses([0x3000])
        g = reader.sample(None)[0x3000][10]
        self.assertIsNone(g.drm_vram)
        self.assertNotIn("engine", g.available)
        self.assertIn("unreadable", reader.status)

    def test_malformed_engine_sample_requires_a_new_valid_baseline(self):
        reader = amdgpu.DrmProcesses([0x3000])
        self.fdinfo(10, 3, busy=100)
        self.read_processes(reader, 1_000_000_000)
        self.fdinfo(10, 3, busy="invalid")
        self.read_processes(reader, 2_000_000_000)
        self.fdinfo(10, 3, busy=300)
        self.assertNotIn("engine", self.read_processes(reader, 3_000_000_000)[0x3000][10].available)

    def collector(self, gpu_index=0):
        patches = (
            patch("gputaskman.collector.BACKEND", "nvml"),
            patch("gputaskman.linux.is_wsl", return_value=False),
            patch("gputaskman.linux.N", None),
            patch("gputaskman.collector.enumerate_adapters", linux.enumerate_adapters),
            patch("gputaskman.collector.GpuCounters", linux.LinuxCounters),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        c = Collector(gpu_index=gpu_index)
        self.addCleanup(c.close)
        return c

    def test_pci_pairing_deduplication_and_units(self):
        adapters = amdgpu.enumerate_adapters()
        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0].luid, linux.pci_key("00000000:03:00.0"))
        c = amdgpu.AmdGpuCounters()
        counters = c.sample()
        d = c.telemetry[adapters[0].luid]
        self.assertEqual(counters[adapters[0].luid].dedicated, 2 << 30)
        self.assertEqual((d.util_gpu, d.temp, d.power_w, d.gfx_clock), (0, 48.5, 25, 1200))
        self.assertIsNone(d.mem_clock)
        self.assertIsNone(d.util_mem)

    def test_missing_malformed_and_zero_counters(self):
        self.write("mem_info_vram_used", "invalid")
        self.write("hwmon/hwmon0/power1_average", "-1")
        self.write("hwmon/hwmon0/power1_input", "0")
        c = amdgpu.AmdGpuCounters()
        counters = c.sample()
        d = next(iter(c.telemetry.values()))
        self.assertIsNone(next(iter(counters.values())).dedicated)
        self.assertIsNone(d.mem_used)
        self.assertEqual(d.power_w, 0)
        (self.device / "mem_info_vram_total").unlink()
        self.assertIsNone(amdgpu._number(self.device / "mem_info_vram_total"))
        self.assertIsNone(amdgpu.enumerate_adapters()[0].dedicated_vram)

    def test_collector_without_nvml_and_honest_output(self):
        s = self.collector().sample()
        self.assertEqual(s.backend, "amdgpu")
        self.assertEqual(s.vram_used, 2 << 30)
        self.assertEqual(s.vram_total, 8 << 30)
        self.assertEqual(s.nv.source, "AMD sysfs")
        self.assertIn("amdgpu (Linux", s.driver)
        data = to_dict(s)
        self.assertIsNone(data["attributed"])
        self.assertIsNone(data["shared_used"])
        self.assertEqual(data["processes"], [])
        self.assertIn("resident client references", data["backend_note"])
        self.assertIn("resident client references", to_text(s))

    def test_unavailable_total_and_measured_zero_are_distinct(self):
        (self.device / "mem_info_vram_total").unlink()
        c = self.collector()
        s = c.sample()
        self.assertIsNone(s.vram_total)
        self.assertIsNone(to_dict(s)["adapter"]["dedicated_vram"])
        self.assertIsNone(to_dict(s)["vram_total"])
        self.assertIn("/ N/A", to_text(s))
        self.write("mem_info_vram_total", "0")
        self.write("mem_info_vram_used", "0")
        s = c.sample()
        self.assertEqual(to_dict(s)["vram_total"], 0)
        self.assertEqual(to_dict(s)["vram_used"], 0)
        self.assertIn("VRAM 0 MB / 0 MB", to_text(s))

    def test_selected_gpu_uses_pci_address_even_when_names_match(self):
        second = self.root / "devices" / "0000:0e:00.0"
        second.mkdir()
        for name, value in {"vendor": "0x1002", "product_name": "AMD test GPU",
                            "mem_info_vram_total": str(512 << 20),
                            "mem_info_vram_used": "1234"}.items():
            (second / name).write_text(value, encoding="ascii")
        (second / "driver").symlink_to(self.driver, target_is_directory=True)
        card = self.drm / "card2"
        card.mkdir()
        (card / "device").symlink_to(second, target_is_directory=True)
        adapters = ordered_adapters(amdgpu.enumerate_adapters())
        self.assertEqual([a.luid for a in adapters], [0x3000, 0xE000])
        s = self.collector(gpu_index=1).sample()
        self.assertEqual(s.adapter.luid, 0xE000)
        self.assertEqual(s.nv.pci_bus, "0000:0e:00.0")
        self.assertEqual(s.nv.index, 1)
        self.assertEqual(s.vram_used, 1234)

    @patch("gputaskman.linux.is_wsl", return_value=True)
    @patch("gputaskman.linux.N", None)
    def test_wsl_does_not_read_native_amd_devices(self, *_):
        self.assertEqual(linux.enumerate_adapters(), [])
        self.assertIsNone(linux.LinuxCounters().amdgpu)

    def test_disappearing_sensors_remain_unavailable(self):
        c = self.collector()
        self.assertEqual(c.sample().vram_used, 2 << 30)
        (self.device / "mem_info_vram_used").unlink()
        (self.device / "hwmon/hwmon0/temp1_input").unlink()
        s = c.sample()
        self.assertIsNone(s.vram_used)
        self.assertIsNone(s.nv.temp)
        self.assertEqual(s.backend, "amdgpu")
        self.assertIsNone(s.attributed)

    def test_inaccessible_topology_is_skipped(self):
        with patch.object(Path, "resolve", side_effect=PermissionError):
            self.assertEqual(amdgpu.enumerate_adapters(), [])

    def test_other_vendors_and_missing_drivers_are_ignored(self):
        self.write("vendor", "0x8086")
        self.assertEqual(amdgpu.enumerate_adapters(), [])
        self.write("vendor", "0x1002")
        (self.device / "driver").unlink()
        self.assertEqual(amdgpu.enumerate_adapters(), [])
