"""Same-boot, PCI-matched host handoff; no elevation or GPU access in these tests."""
import dataclasses
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from gputaskman import drm
from gputaskman.intel_host import (CLIENT_USED, PCI_TOTAL, XE_TOTAL, XE_USED,
                                  FeedWriter, HostDeviceFeed, MAX_BYTES, filename)
from gputaskman.telemetry import DeviceTelemetry


@unittest.skipUnless(os.name == 'posix', 'POSIX host file ownership and locking')
class HostFeedTests(unittest.TestCase):
    def setUp(self):
        boot = patch('gputaskman.intel_host.boot_id', return_value='00000000-0000-4000-8000-000000000001')
        boot.start()
        self.addCleanup(boot.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / 'feed'
        self.pci = '0000:04:00.0'
        self.writer = FeedWriter(self.pci, 0x8086, 0xE222, directory=self.directory)
        self.addCleanup(self.writer.close)
        self.now = 1000.0
        self.reader = HostDeviceFeed(self.pci, 0x8086, 0xE222, directory=self.directory,
                                     trusted_uid=os.getuid(), boot=self.writer.boot, clock=lambda: self.now)
        self.path = self.directory / filename(self.pci)
        self.card = drm.DrmCard(key=1, pdev=self.pci, sysfs='/missing', driver='xe',
                                vendor_id=0x8086, device_id=0xE222, name='Intel',
                                vram_total=2048, integrated=False, intel=True)

    def publish(self, **fields):
        host = DeviceTelemetry(index=0, source='Intel sysfs', pci_bus=self.pci, **fields)
        self.writer.publish(dataclasses.asdict(host), taken=1000.0)
        return host

    def edit(self, change):
        body = json.loads(self.path.read_text())
        change(body)
        self.path.write_text(json.dumps(body))

    def test_fresh_measured_zero_reaches_device_with_source(self):
        self.publish(util_gpu=0, video=0, mem_read_bps=0, mem_write_bps=0,
                     mem_bandwidth_max_bps=608_000_000_000, mem_bandwidth_pct=0,
                     pcie_rx_kbs=0, pcie_tx_kbs=0, pcie_gen=4, pcie_width=4,
                     metric_sources={'mem_read_bps': 'Intel PMT (BMG G31)'})
        device = DeviceTelemetry(index=0)
        self.reader.apply(self.card, device)
        self.assertEqual((device.util_gpu, device.video, device.mem_read_bps, device.mem_write_bps), (0, 0, 0, 0))
        self.assertEqual((device.pcie_gen, device.pcie_width), (4, 4))
        self.assertIn('Intel PMT', device.metric_sources['mem_read_bps'])
        self.assertIn('via host reader', device.metric_sources['mem_read_bps'])
        self.assertIsNone(device.util_mem)
        self.assertIsNone(device.enc)

    def test_native_readings_take_precedence_and_zero_fan_is_not_hidden(self):
        self.publish(util_gpu=80, temp=50, fan_rpm=1200, power_w=125)
        device = DeviceTelemetry(index=0, util_gpu=0, temp=30, fan_rpm=0, power_w=0)
        self.reader.apply(self.card, device)
        self.assertEqual((device.util_gpu, device.temp, device.fan_rpm, device.power_w), (0, 30, 0, 0))

    def test_auxiliary_temperatures_fill_only_missing_sensors(self):
        self.publish(temperature_sensors_c={'mctrl': 48, 'pcie': 66, 'vram_ch_0': 78})
        device = DeviceTelemetry(index=0, temperature_sensors_c={'mctrl': 0})
        self.reader.apply(self.card, device)
        self.assertEqual(device.temperature_sensors_c, {'mctrl': 0, 'pcie': 66, 'vram_ch_0': 78})
        self.assertIn('via host reader', device.metric_sources['temperature_sensors_c'])

    def test_invalid_auxiliary_temperatures_reject_the_sample(self):
        for value in (None, [], {'pcie': True}, {'pcie': float('nan')}, {'pcie': -274},
                      {'pcie': 10**500}, {'vram_ch_16': 50}, {'untrusted': 50}):
            with self.subTest(value=value):
                self.publish()
                self.edit(lambda body: body['device'].update(temperature_sensors_c=value))
                self.assertIsNone(self.reader.read()[0])

    def test_measured_load_supersedes_visible_client_estimate(self):
        self.publish(util_gpu=0, video=0)
        device = DeviceTelemetry(index=0, util_gpu=70, video=20,
                                 util_kind=CLIENT_USED, video_kind=CLIENT_USED)
        self.reader.apply(self.card, device)
        self.assertEqual((device.util_gpu, device.video, device.util_kind, device.video_kind), (0, 0, '', ''))
        self.card.pmu_state['fail'] = 'denied'
        self.assertEqual(drm.intel_counter_status(self.card, device), '')

    def test_privileged_device_memory_does_not_become_a_client_sum(self):
        self.publish(mem_total=1000, mem_used=0,
                     metric_sources={'mem_total': XE_TOTAL, 'mem_used': XE_USED})
        device = DeviceTelemetry(index=0, mem_total=2048, mem_used=500,
                                 metric_sources={'mem_total': PCI_TOTAL, 'mem_used': CLIENT_USED})
        self.reader.apply(self.card, device)
        self.assertEqual((device.mem_total, device.mem_used), (1000, 0))
        self.assertIn('estimate', device.memory_note)
        self.assertNotIn('visible', device.memory_note)
        self.assertNotIn('memory_query', device.metric_status)

    def test_native_privileged_memory_keeps_precedence(self):
        self.publish(mem_total=1000, mem_used=700, metric_sources={'mem_total': XE_TOTAL, 'mem_used': XE_USED})
        device = DeviceTelemetry(index=0, mem_total=1000, mem_used=500,
                                 metric_sources={'mem_total': XE_TOTAL, 'mem_used': XE_USED})
        self.reader.apply(self.card, device)
        self.assertEqual(device.mem_used, 500)

    def test_rate_families_are_not_spliced_between_sources(self):
        self.publish(mem_read_bps=20, mem_write_bps=30)
        device = DeviceTelemetry(index=0, mem_read_bps=10, mem_write_bps=None,
                                 mem_bandwidth_max_bps=100, mem_bandwidth_pct=90)
        self.reader.apply(self.card, device)
        self.assertEqual((device.mem_read_bps, device.mem_write_bps), (20, 30))
        self.assertIsNone(device.mem_bandwidth_max_bps)
        self.assertIsNone(device.mem_bandwidth_pct)

    def test_complete_native_rates_do_not_borrow_a_host_maximum(self):
        self.publish(mem_read_bps=20, mem_write_bps=30, mem_bandwidth_max_bps=100)
        device = DeviceTelemetry(index=0, mem_read_bps=1, mem_write_bps=2)
        self.reader.apply(self.card, device)
        self.assertEqual((device.mem_read_bps, device.mem_write_bps), (1, 2))
        self.assertIsNone(device.mem_bandwidth_max_bps)

    def test_partial_counter_pair_remains_unknown(self):
        self.publish(mem_read_bps=20, pcie_gen=4)
        device = DeviceTelemetry(index=0)
        self.reader.apply(self.card, device)
        self.assertIsNone(device.mem_read_bps)
        self.assertIsNone(device.mem_write_bps)
        self.assertIsNone(device.pcie_gen)
        self.assertIsNone(device.pcie_width)

    def test_missing_then_recovered_then_stale_sample(self):
        self.assertEqual(self.reader.read(), (None, ''))
        self.publish(util_gpu=0)
        self.assertEqual(self.reader.read()[0]['util_gpu'], 0)
        self.now = 1007
        device = DeviceTelemetry(index=0, temp=30)
        self.reader.apply(self.card, device)
        self.assertIsNone(device.util_gpu)
        self.assertEqual(device.temp, 30)
        self.assertIn('stale', device.metric_status['host_feed'])
        self.writer.publish(dataclasses.asdict(DeviceTelemetry(index=0, source='Intel sysfs', pci_bus=self.pci,
                                                               util_gpu=1)), taken=self.now)
        self.assertEqual(self.reader.read()[0]['util_gpu'], 1)

    def test_wrong_device_boot_schema_and_future_samples_are_rejected(self):
        changes = [('pci_bus', '0000:05:00.0'), ('vendor_id', 0x1002), ('device_id', 123),
                   ('boot_id', 'another-boot'), ('schema', True), ('schema', 2), ('taken', 1002)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                self.publish(util_gpu=50)
                self.edit(lambda body: body.update({key: value}))
                value, reason = self.reader.read()
                self.assertIsNone(value)
                self.assertTrue(reason)

    def test_stopped_reader_has_no_live_values(self):
        self.writer.publish(None, taken=self.now, state='stopped')
        value, reason = self.reader.read()
        self.assertIsNone(value)
        self.assertIn('stopped', reason)

    def test_invalid_numbers_and_memory_bounds_are_rejected(self):
        for key, value in (('util_gpu', float('nan')), ('power_w', float('inf')), ('fan', 101),
                           ('fan_rpm', -1), ('mem_used', 1001), ('mem_total', 1000.0),
                           ('pcie_gen', 0), ('pcie_width', 3), ('video', True), ('power_w', 10**500)):
            with self.subTest(key=key, value=value):
                self.publish(mem_total=1000)
                self.edit(lambda body: body['device'].update({key: value}))
                self.assertIsNone(self.reader.read()[0])

    def test_negative_temperature_can_be_measured(self):
        self.publish(temp=-5)
        self.assertEqual(self.reader.read()[0]['temp'], -5)

    def test_huge_timestamp_and_invalid_provenance_do_not_crash(self):
        for change in (lambda body: body.update(taken=10**500),
                       lambda body: body['device'].update(metric_sources={'temp': None}),
                       lambda body: body['device'].update(throttle='unknown')):
            self.publish()
            self.edit(change)
            self.assertIsNone(self.reader.read()[0])

    def test_unsafe_permissions_symlink_and_fifo_are_rejected(self):
        self.publish(util_gpu=50)
        self.path.chmod(0o666)
        self.assertIsNone(self.reader.read()[0])
        self.path.chmod(0o644)
        self.directory.chmod(0o777)
        self.assertIsNone(self.reader.read()[0])
        self.directory.chmod(0o755)
        elsewhere = Path(self.tmp.name) / 'elsewhere'
        self.path.rename(elsewhere)
        self.path.symlink_to(elsewhere)
        self.assertIsNone(self.reader.read()[0])
        self.path.unlink()
        os.mkfifo(self.path)
        self.assertIsNone(self.reader.read()[0])

    def test_oversized_and_truncated_frames_are_rejected(self):
        self.publish()
        self.path.write_text(' ' * (MAX_BYTES + 1))
        self.assertIsNone(self.reader.read()[0])
        self.path.write_text('{"schema":1')
        self.assertIsNone(self.reader.read()[0])

    def test_atomic_replacement_and_single_writer_lock(self):
        for value in (0, 10, 0):
            self.publish(util_gpu=value)
            self.assertEqual(self.reader.read()[0]['util_gpu'], value)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o644)
        with self.assertRaisesRegex(RuntimeError, 'already publishing'):
            FeedWriter(self.pci, 0x8086, 0xE222, directory=self.directory)
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_private_launcher_umask_does_not_hide_public_device_counters(self):
        directory = Path(self.tmp.name) / 'private-umask'
        previous = os.umask(0o077)
        try:
            writer = FeedWriter(self.pci, 0x8086, 0xE222, directory=directory)
            try:
                writer.publish(dataclasses.asdict(DeviceTelemetry(index=0, source='Intel sysfs', pci_bus=self.pci)))
            finally:
                writer.close()
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE((directory / filename(self.pci)).stat().st_mode), 0o644)

    def test_invalid_filename_cannot_escape_the_feed_directory(self):
        for value in ('../../secret', '04:00.0', '0000:04:00.8'):
            with self.assertRaises(ValueError):
                filename(value)

    def test_disabled_feed_is_not_read_by_the_producer(self):
        from unittest.mock import Mock

        self.card.pmt_provider = Mock()
        self.card.pmt_provider.sample.return_value = ({}, {}, {})
        self.card.sysman_provider = Mock(error='', domain_errors={})
        self.card.sysman_provider.sample.return_value = ({}, {}, {})
        self.card.host_device_feed = Mock()
        with patch.dict(os.environ, {'GPUTASKMAN_DISABLE_HOST_FEED': '1'}), \
                patch('gputaskman.intel_pmt.pci_link', return_value=({}, 'unreadable')):
            drm._xe_extra_telemetry(self.card, DeviceTelemetry(index=0))
        self.card.host_device_feed.apply.assert_not_called()

    def test_host_values_reach_json_text_and_intel_charts_then_expire(self):
        from unittest.mock import Mock
        from gputaskman.app import GpuTaskManager
        from gputaskman.collector import Snapshot
        from gputaskman.report import to_dict, to_text

        self.publish(util_gpu=0, video=0, mem_total=1000, mem_used=700,
                     mem_read_bps=2e9, mem_write_bps=1e9, pcie_gen=4, pcie_width=4,
                     metric_sources={'mem_total': XE_TOTAL, 'mem_used': XE_USED,
                                     'mem_read_bps': 'Intel PMT (BMG G31)'})
        self.card.host_device_feed = self.reader
        self.card.xe_provider = Mock(query_error='device query unavailable', pmu_error='denied')
        self.card.xe_provider.memory.return_value = (None, None)
        self.card.pmt_provider = Mock()
        self.card.pmt_provider.sample.return_value = ({}, {}, {})
        self.card.sysman_provider = Mock(error='', domain_errors={})
        self.card.sysman_provider.sample.return_value = ({}, {}, {})
        self.card.vram_used = 300
        with patch.dict(os.environ, {'GPUTASKMAN_DISABLE_HOST_FEED': '0'}), \
                patch.object(drm, '_intel_pmu_rates', return_value=(None, None)), \
                patch('gputaskman.intel_pmt.pci_link', return_value=({}, 'denied')):
            device = drm.read_device(self.card, 0)
            snapshot = Snapshot(taken=1000, adapter=None, adapters=[], nv=device, driver='kernel test',
                                cuda='', backend='drm', vram_total=device.mem_total,
                                vram_used=device.mem_used, shared_used=None)
            self.assertEqual(to_dict(snapshot)['device']['mem_used'], 700)
            app = GpuTaskManager()
            panel = '\n'.join(line.plain for line in app.render_device(snapshot).renderables)
            for text in (to_text(snapshot), panel):
                self.assertIn('Memory BW read 2.00 / write 1.00 GB/s', text)
                self.assertIn('via host reader', text)
                self.assertIn('PCIe gen4 x4', text)
            self.assertIn(app.CHARTS.index('Mem activity'), app.charts_for(snapshot))
            self.now = 1007
            expired = drm.read_device(self.card, 0)
        self.assertIsNone(expired.util_gpu)
        self.assertIsNone(expired.mem_read_bps)
        self.assertEqual(expired.mem_used, 300)  # native visible-client sum, not the expired host estimate
        self.assertIn('stale', expired.metric_status['host_feed'])
