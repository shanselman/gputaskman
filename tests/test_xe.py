"""Xe fdinfo replays and sysfs fixtures based on Linux's xe_drm_client.c ABI.

These are synthetic counter sequences, not captures of live process activity.
"""
from __future__ import annotations

import tempfile
import os
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from gputaskman import drm
from test_drm import Clock, GiB, KiB, MiB, XE_FDINFO, _write_fd


def fdinfo(busy=0, total=1000, *, client=3, shared=False):
    text = "\n".join(line for line in XE_FDINFO.splitlines()
                     if not line.startswith(("drm-cycles-", "drm-total-cycles-"))) + "\n"
    text = text.replace("drm-client-id:\t3", f"drm-client-id:\t{client}")
    if not shared:
        text = text.replace("drm-shared-vram0:\t16 MiB", "drm-shared-vram0:\t0")
    return text + f"drm-cycles-ccs:\t{busy}\ndrm-total-cycles-ccs:\t{total}\n"


class XeParserTests(unittest.TestCase):
    def test_multiple_local_regions_and_cpu_memory_are_kept_apart(self):
        text = fdinfo() + "drm-resident-vram1:\t4 MiB\ndrm-shared-vram1:\t0\n"
        text = text.replace("drm-resident-system:\t0", "drm-resident-system:\t2 MiB")
        info = drm.parse_fdinfo(text)
        self.assertEqual(info["vram"], 23992 * KiB + 4 * MiB)
        self.assertEqual(info["gtt"], 192 * KiB)
        self.assertEqual(info["cpu"], 2 * MiB)
        self.assertEqual(info["shared"], 0)

    def test_missing_region_or_sharing_value_never_becomes_zero(self):
        for line in ("drm-shared-system:\t0\n", "drm-shared-gtt:\t0\n",
                     "drm-shared-vram0:\t0\n", "drm-shared-stolen:\t0\n"):
            with self.subTest(line=line):
                self.assertIsNone(drm.parse_fdinfo(fdinfo().replace(line, ""))["shared"])
        text = fdinfo() + "drm-total-vram1:\t4 MiB\ndrm-shared-vram1:\t0\n"
        self.assertIsNone(drm.parse_fdinfo(text)["vram"])
        text = fdinfo().replace("drm-resident-vram0:\t23992 KiB", "drm-resident-vram0:\tbad")
        self.assertIsNone(drm.parse_fdinfo(text)["vram"])

    def test_incomplete_malformed_and_negative_cycles_are_unknown(self):
        for old, new in (("drm-total-cycles-ccs:\t1000\n", ""),
                         ("drm-cycles-ccs:\t0\n", ""),
                         ("drm-cycles-ccs:\t0", "drm-cycles-ccs:\tbad"),
                         ("drm-cycles-ccs:\t0", "drm-cycles-ccs:\t-1")):
            with self.subTest(new=new):
                self.assertIsNone(drm.parse_fdinfo(fdinfo().replace(old, new))["cycles"])

    def test_integrated_xe_has_no_local_region(self):
        text = "\n".join(line for line in fdinfo().splitlines() if "vram0" not in line)
        info = drm.parse_fdinfo(text)
        self.assertIsNone(info["vram"])
        self.assertEqual((info["gtt"], info["shared"]), (192 * KiB, 0))


class XeScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.proc = root / "proc"
        self.proc.mkdir()
        self.clock = Clock()
        self.card = drm.DrmCard(key=12288, pdev="0000:03:00.0", sysfs=str(root / "device"),
                                driver="xe", vendor_id=0x8086, device_id=0xe222,
                                name="Intel GPU", vram_total=32 * GiB, integrated=False, intel=True)
        self.counters = drm.DrmCounters([self.card], proc=str(self.proc), clock=self.clock)
        self.addCleanup(self.counters.close)

    def sample(self, busy=0, total=1000, *, text=None):
        _write_fd(self.proc, 100, 5, fdinfo(busy, total) if text is None else text)
        self.clock.tick()
        return self.counters.sample()[0][self.card.key][100]

    def test_gpu_ticks_drive_compute_and_private_memory_is_charged(self):
        first = self.sample(100, 1000)
        self.assertNotIn("engine", first.available)
        self.clock.tick(50)  # CPU wall time must have no effect on the GPU ratio
        row = self.sample(850, 2000)
        self.assertEqual(row.engines["compute"], 75.0)
        self.assertEqual(row.gpu_pct, 75.0)
        self.assertEqual((row.dedicated, row.shared), (23992 * KiB, 192 * KiB))
        self.assertEqual(row.drm_reason, "")

    def test_idle_is_measured_zero_but_stalled_gpu_clock_is_unknown(self):
        self.sample()
        idle = self.sample(0, 2000)
        self.assertIn("engine", idle.available)
        self.assertEqual(idle.gpu_pct, 0)
        self.assertNotIn("engine", self.sample(0, 2000).available)

    def test_busy_counter_high_water_and_total_counter_reset(self):
        self.sample(100, 1000)
        self.assertEqual(self.sample(90, 2000).gpu_pct, 0)
        self.assertEqual(self.sample(600, 3000).gpu_pct, 50)
        self.assertNotIn("engine", self.sample(0, 10).available)
        self.assertEqual(self.sample(25, 110).gpu_pct, 25)

    def test_class_capacity_preserves_one_busy_engine_semantics_and_clamps(self):
        self.sample(0, 1000)
        self.assertEqual(self.sample(1000, 2000).gpu_pct, 100)
        self.assertEqual(self.sample(3000, 3000).gpu_pct, 100)

    def test_all_clients_must_have_a_measured_interval(self):
        self.sample()
        _write_fd(self.proc, 100, 6, fdinfo(client=4))
        self.assertNotIn("engine", self.sample(500, 2000).available)
        _write_fd(self.proc, 100, 6, fdinfo(100, 2000, client=4))
        self.assertEqual(self.sample(1000, 3000).gpu_pct, 60)

    def test_duplicate_fds_are_counted_once_and_shared_clients_are_unknown(self):
        self.sample()
        _write_fd(self.proc, 100, 6, fdinfo(500, 2000))
        row = self.sample(500, 2000)
        self.assertEqual((row.gpu_pct, row.dedicated), (50, 23992 * KiB))
        _write_fd(self.proc, 200, 7, fdinfo(1000, 3000))
        row = self.sample(1000, 3000)
        self.assertNotIn("engine", row.available)
        self.assertEqual(row.drm_reason, drm.REASON_SHARED_CLIENT)

    def test_missing_pairs_and_partial_scans_do_not_look_idle(self):
        self.sample()
        broken = fdinfo(500, 2000).replace("drm-total-cycles-ccs:\t2000\n", "")
        self.assertNotIn("engine", self.sample(text=broken).available)
        row = self.sample(1000, 3000)
        self.assertEqual(row.gpu_pct, 50)
        original = self.counters._clients

        def partial():
            result = original()
            self.counters.partial = True
            return result

        with patch.object(self.counters, "_clients", side_effect=partial):
            row = self.sample(1500, 4000)
        self.assertNotIn("engine", row.available)
        self.assertEqual(row.drm_reason, drm.REASON_PARTIAL_SCAN)
        self.assertIsNone(self.card.client_util)

    def test_visible_client_fallback_needs_an_interval_and_deduplicates_clients(self):
        self.sample()
        self.assertIsNone(self.card.client_util)
        _write_fd(self.proc, 100, 6, fdinfo(500, 2000))
        self.sample(500, 2000)
        self.assertEqual(self.card.client_util, 50)
        _write_fd(self.proc, 100, 6, fdinfo(500, 3000))
        self.sample(500, 3000)
        self.assertEqual(self.card.client_util, 0)
        self.assertIsNone(self.card.client_video)  # absent video counters are not idle video

    def test_integrated_xe_charges_gtt_not_cpu_only_memory(self):
        self.card.vram_total = 0
        self.card.integrated = True
        text = "\n".join(line for line in fdinfo().splitlines() if "vram0" not in line)
        row = self.sample(text=text)
        self.assertTrue(self.card.integrated)
        self.assertEqual(row.dedicated, 192 * KiB)
        self.assertNotIn("shared", row.available)


class XeDeviceTests(unittest.TestCase):
    def test_labelled_temperature_frequency_and_throttle(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp)
            hwmon = dev / "hwmon" / "hwmon8"
            hwmon.mkdir(parents=True)
            for name, value in {"temp2_label": "pkg", "temp2_input": "53000",
                                "temp3_label": "vram", "temp3_input": "48000"}.items():
                (hwmon / name).write_text(value)
            gt = dev / "tile0" / "gt0" / "freq0"
            throttle = gt / "throttle"
            throttle.mkdir(parents=True)
            for name, value in {"act_freq": "2400", "cur_freq": "2500", "rp0_freq": "2400"}.items():
                (gt / name).write_text(value)
            (throttle / "reason_thermal").write_text("1")
            (throttle / "reason_pl1").write_text("0")
            card = drm.DrmCard(key=1, pdev="0000:04:00.0", sysfs=str(dev), driver="xe",
                                vendor_id=0x8086, device_id=0xe222, name="Intel GPU",
                                vram_total=32 * GiB, integrated=False, intel=True)
            metric = drm.read_device(card, 0)
            self.assertEqual(metric.temp, 53)
            self.assertEqual(metric.vram_temp, 48)
            self.assertEqual((metric.sm_clock, metric.gfx_clock, metric.sm_clock_max), (2400, 2500, 2400))
            self.assertEqual(metric.throttle, ["thermal"])
            (gt / "act_freq").write_text("0")  # a real idle reading, not a request fallback
            self.assertEqual(drm.read_device(card, 0).sm_clock, 0)
            (gt / "act_freq").unlink()
            self.assertEqual(drm.read_device(card, 0).sm_clock, 2500)
            self.assertIn("Xe device-wide", drm.intel_counter_status(card, metric))
            self.assertIn("unavailable", drm.intel_counter_status(card, metric))

    def test_one_shot_primes_board_power_and_preserves_counter_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp)
            hwmon = dev / "hwmon" / "hwmon8"
            hwmon.mkdir(parents=True)
            energy = hwmon / "energy1_input"
            energy.write_text("1000000")
            card = drm.DrmCard(key=1, pdev="0000:04:00.0", sysfs=str(dev), driver="xe",
                                vendor_id=0x8086, device_id=0xe222, name="Intel GPU",
                                vram_total=32 * GiB, integrated=False, intel=True)
            counters = drm.DrmCounters([card], proc=str(dev / "proc"))
            self.addCleanup(counters.close)
            with patch.object(drm.time, "perf_counter", return_value=1.0):
                counters.start(card)
            energy.write_text("51000000")
            with patch.object(drm.time, "perf_counter", return_value=2.0):
                metric = drm.read_device(card, 0)
            self.assertEqual((metric.power_w, metric.power_kind), (50, "Board"))
            energy.write_text("100")
            with patch.object(drm.time, "perf_counter", return_value=3.0):
                self.assertIsNone(drm.read_device(card, 0).power_w)

    def test_sensors_and_unavailable_reasons_reach_every_output(self):
        from gputaskman.app import GpuTaskManager
        from gputaskman.collector import Snapshot
        from gputaskman.report import to_dict, to_text

        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp)
            hwmon = dev / "hwmon" / "hwmon8"
            hwmon.mkdir(parents=True)
            for name, value in {"fan1_input": "960", "power1_cap": "200000000",
                                "temp3_label": "vram", "temp3_input": "32000"}.items():
                (hwmon / name).write_text(value)
            media = dev / "tile0" / "gt1"
            (media / "engines" / "vcs").mkdir(parents=True)
            (media / "freq0").mkdir()
            (media / "freq0" / "act_freq").write_text("0")
            card = drm.DrmCard(key=1, pdev="0000:04:00.0", sysfs=tmp, driver="xe",
                               vendor_id=0x8086, device_id=0xe222, name="Intel GPU",
                               vram_total=32 * GiB, integrated=False, intel=True,
                               client_util=75, client_video=0)
            self.addCleanup(lambda: card.xe_provider.close() if card.xe_provider else None)
            nv = drm.read_device(card, 0)
            snap = Snapshot(taken=0, adapter=None, adapters=[], nv=nv, driver="", cuda="",
                            backend="drm", vram_total=32 * GiB, vram_used=None, shared_used=None)
            data = to_dict(snap)["device"]
            self.assertEqual((data["fan_rpm"], data["fan"], data["vram_temp"]), (960, None, 32))
            self.assertEqual((data["power_limit_w"], data["video_clock"]), (200, 0))
            self.assertEqual((data["util_gpu"], data["util_kind"]), (75, "visible DRM clients (estimated)"))
            self.assertEqual((data["video"], data["video_kind"]), (0, "visible DRM clients (estimated)"))
            self.assertIsNone(data["pcie_rx_kbs"])
            self.assertIn("pcie_rx_kbs", data["metric_status"])
            panel = "\n".join(line.plain for line in GpuTaskManager().render_device(snap).renderables)
            for text in (to_text(snap), panel):
                self.assertIn("fan 960 RPM", text)
                self.assertIn("visible DRM clients", text)
                self.assertIn("Metric availability:", text)
                self.assertIn("no Xe bandwidth counter", text)
            self.assertIn("VRAM 32°C", panel)


class XeQueryTests(unittest.TestCase):
    """Public Xe query layouts; these fixtures do not claim hardware validation."""

    def provider(self, root="/nonexistent"):
        from gputaskman.xe import XeDevice

        provider = XeDevice(root, "0000:04:00.0", pmu_root=root)
        self.addCleanup(provider.close)
        return provider

    @staticmethod
    def memory_data(*regions):
        # drm_xe_query_mem_regions: 8-byte header, 88-byte drm_xe_mem_region entries.
        return struct.pack("=II", len(regions), 0) + b"".join(
            struct.pack("=HHI4Q6Q", cls, instance, 4096, total, used, total, used, *([0] * 6))
            for cls, instance, total, used in regions)

    def test_vram_query_ignores_system_ram_and_hides_unprivileged_zero(self):
        provider = self.provider()
        data = self.memory_data((0, 0, 64 * GiB, 10 * GiB), (1, 1, 31 * GiB, 0))
        with patch.object(provider, "query", return_value=data), patch("gputaskman.xe.perfmon_capable", return_value=False):
            self.assertEqual(provider.memory(), (31 * GiB, None))
        with patch.object(provider, "query", return_value=data), patch("gputaskman.xe.perfmon_capable", return_value=True):
            self.assertEqual(provider.memory(), (31 * GiB, 0))

    def test_invalid_region_layout_and_duplicate_ids_remain_unknown(self):
        provider = self.provider()
        for data in (self.memory_data((1, 1, GiB, 2 * GiB)),
                     self.memory_data((1, 1, GiB, 0), (1, 1, GiB, 0)),
                     self.memory_data((1, 1, GiB, 0))[:-1]):
            with self.subTest(size=len(data)), patch.object(provider, "query", return_value=data):
                self.assertEqual(provider.memory(), (None, None))

    @unittest.skipUnless(sys.platform.startswith("linux"), "fcntl ioctl is Linux-only")
    def test_ioctl_uses_the_pci_matched_render_node_and_bounds_response(self):
        import ctypes
        from gputaskman.xe import QUERY, XeDevice

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "drm" / "renderD129").mkdir(parents=True)
            (root / "nodes").mkdir()
            (root / "nodes" / "renderD129").touch()
            provider = XeDevice(tmp, "0000:04:00.0", dev_root=str(root / "nodes"))
            self.addCleanup(provider.close)
            data = self.memory_data((1, 1, 31 * GiB, 4 * GiB))

            def ioctl(fd, request, buffer, mutate):
                self.assertEqual(request, 0xC0286440)
                extensions, kind, size, address, r0, r1 = QUERY.unpack(buffer)
                self.assertEqual((extensions, kind, r0, r1), (0, 1, 0, 0))
                if size == 0:
                    QUERY.pack_into(buffer, 0, 0, kind, len(data), 0, 0, 0)
                else:
                    self.assertEqual(size, len(data))
                    ctypes.memmove(address, data, len(data))

            with patch("fcntl.ioctl", side_effect=ioctl):
                self.assertEqual(provider.query(1), data)
            def oversized(fd, request, buffer, mutate):
                QUERY.pack_into(buffer, 0, 0, 1, 1 << 21, 0, 0, 0)
            with patch("fcntl.ioctl", side_effect=oversized) as query:
                self.assertIsNone(provider.query(1))
                self.assertEqual(query.call_count, 1)
                self.assertIn("size", provider.query_error)
            os.fstat(provider.fd)
            fd = provider.fd
            provider.close()
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_pmu_matches_pci_and_uses_engine_ticks_on_the_media_gt(self):
        from gputaskman.xe import XeDevice

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pmu = root / "xe_0000_04_00.0"
            (pmu / "format").mkdir(parents=True)
            (pmu / "events").mkdir()
            for name, value in {"type": "99", "cpumask": "2-3",
                                "format/event": "config:0-11", "format/gt": "config:60-63",
                                "format/engine_class": "config:20-27", "format/engine_instance": "config:12-19",
                                "events/engine-active-ticks": "event=0x02",
                                "events/engine-total-ticks": "event=0x03"}.items():
                (pmu / name).write_text(value)
            provider = XeDevice(tmp, "0000:04:00.0", pmu_root=tmp)
            self.addCleanup(provider.close)
            engine_data = struct.pack("=II4H3Q", 1, 0, 2, 1, 1, 0, 0, 0, 0)
            values = {2: 100, 3: 1000}
            configs = {}

            def open_counter(typ, config, cpu):
                self.assertEqual((typ, cpu), (99, 2))
                self.assertEqual(config & ~0xfff, (1 << 60) | (2 << 20) | (1 << 12))
                fd = os.open(os.devnull, os.O_RDONLY)
                configs[fd] = config & 0xfff
                return fd, None

            def read_counter(fd):
                return values[configs[fd]]

            with patch.object(provider, "query", return_value=engine_data):
                self.assertEqual(provider.rates(open_counter, read_counter), (None, None))
                values.update({2: 850, 3: 2000})
                self.assertEqual(provider.rates(open_counter, read_counter), (75, 75))
                values[3] = 3000
                self.assertEqual(provider.rates(open_counter, read_counter), (0, 0))
                self.assertEqual(provider.rates(open_counter, read_counter), (None, None))
                values.update({2: 0, 3: 10})
                self.assertEqual(provider.rates(open_counter, read_counter), (None, None))
                provider.close()
                for fd in configs:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_denied_pmu_never_reads_like_zero(self):
        provider = self.provider()
        with patch.object(provider, "_spec", return_value=(99, 0, {"ccs0@gt0": (2, 3)})):
            self.assertEqual(provider.rates(lambda *args: (None, "denied"), lambda fd: 0), (None, None))
            self.assertEqual(provider.pmu_error, "denied")


class IntelChartTests(unittest.IsolatedAsyncioTestCase):
    """Hide unsupported Intel charts without hiding missing readings from supported counters."""

    def snapshot(self, source="Intel sysfs", video=0):
        from gputaskman.collector import Snapshot
        from gputaskman.telemetry import DeviceTelemetry

        return Snapshot(taken=0, adapter=None, adapters=[], driver="", cuda="", backend="drm",
                        vram_total=32 * GiB, vram_used=4 * GiB, shared_used=None,
                        nv=DeviceTelemetry(index=0, source=source, util_gpu=25, video=video,
                                           sm_clock=2400, sm_clock_max=2400, temp=32, power_w=100,
                                           power_limit_w=200, fan_rpm=960, vram_temp=30, video_clock=0))

    def chart_text(self, app):
        import io
        from rich.console import Console

        output = io.StringIO()
        Console(file=output, width=140, color_system=None).print(app.query_one("#charts").content)
        return output.getvalue()

    async def test_intel_uses_gt_and_combined_video_without_permanent_na_boxes(self):
        from gputaskman.app import GpuTaskManager

        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager(system=False)
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply(self.snapshot())
                await pilot.pause()
                text = self.chart_text(app)
                for label in ("GPU", "GT clock", "VRAM", "Temperature", "Power", "Video"):
                    self.assertIn(label, text)
                for label in ("Mem activity", "Encoder", "Decoder", "SM clock"):
                    self.assertNotIn(label, text)
                self.assertNotIn("N/A", text)
                self.assertEqual(app.hist_enc[-1], 0)
                panel = "\n".join(line.plain for line in app.render_device(app.snap).renderables)
                for absent in ("P N/A", "PCIe", "mem N/A", "Mem activity"):
                    self.assertNotIn(absent, panel)
                self.assertIn("fan 960 RPM", panel)
                app.apply(self.snapshot(video=None))
                await pilot.pause()
                self.assertIsNone(app.hist_enc[-1])
                self.assertIn("N/A", self.chart_text(app))  # unavailable video is never shown as idle

    async def test_profile_change_resets_unsupported_selection_and_preserves_mixed_gpus(self):
        from gputaskman.app import GpuTaskManager

        with patch.object(GpuTaskManager, "poll"):
            app = GpuTaskManager(system=False)
            async with app.run_test(size=(140, 42)) as pilot:
                app.apply(self.snapshot(source="NVML"))
                await pilot.pause()
                decoder = app.CHARTS.index("Decoder")
                app.query_one("#chart-mode").value = decoder
                await pilot.pause()
                self.assertEqual(app.chart_idx, decoder)
                app.apply(self.snapshot())
                await pilot.pause()
                self.assertEqual(app.chart_idx, -1)
                self.assertNotIn(decoder, app.available_charts)
                app.apply_all([self.snapshot(), self.snapshot(source="NVML")])
                await pilot.pause()
                self.assertIn(decoder, app.available_charts)
                text = self.chart_text(app)
                self.assertIn("GT clock", text)
                self.assertIn("SM clock", text)
                self.assertEqual(text.count("Decoder"), 1)
                self.assertEqual(text.count("Mem activity"), 1)


if __name__ == "__main__":
    unittest.main()
