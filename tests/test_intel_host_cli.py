"""Host-reader orchestration with fake devices; never starts a privileged process."""
import contextlib
import dataclasses
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from gputaskman import drm
from gputaskman.dxgi import Adapter
from gputaskman.intel_host import ReaderRunning
from gputaskman.telemetry import DeviceTelemetry
from tools import intel_host_reader as cli


class HostReaderCliTests(unittest.TestCase):
    def test_native_audit_cannot_borrow_an_existing_privileged_feed(self):
        from tools import check_intel_telemetry as audit

        card = drm.DrmCard(key=1, pdev='0000:04:00.0', sysfs='/fixture', driver='xe',
                           vendor_id=0x8086, device_id=0xe222, name='Intel',
                           vram_total=1024, integrated=False, intel=True)
        def read(card, index):
            self.assertEqual(os.environ.get('GPUTASKMAN_DISABLE_HOST_FEED'), '1')
            return DeviceTelemetry(index=index, source='Intel sysfs', pci_bus=card.pdev)
        with patch.dict(os.environ, {'GPUTASKMAN_DISABLE_HOST_FEED': '0'}), \
                patch.object(sys, 'argv', ['check', '--pci', card.pdev]), \
                patch.object(drm, 'cards', return_value=[card]), \
                patch.object(Path, 'glob', return_value=[]), \
                patch.object(drm, 'read_device', side_effect=read) as reads, \
                patch.object(drm, 'DrmCounters') as counters, \
                patch.object(audit.os, 'uname', return_value=Mock(release='test-kernel'), create=True), \
                patch.object(audit.os, 'geteuid', return_value=1000, create=True), \
                patch.object(audit.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()) as output:
            audit.main()
        import json
        report = json.loads(output.getvalue())
        self.assertEqual(reads.call_count, 2)
        self.assertIn('host-feed imports disabled', report['scope'])
        self.assertIsNone(report['samples'][1]['mem_read_bps'])
        counters.return_value.close.assert_called_once()

    def run_reader(self, *, record=False, wrong_gpu=False, writer_busy=False):
        card = drm.DrmCard(key=1, pdev='0000:04:00.0', sysfs='/fixture', driver='xe',
                           vendor_id=0x8086, device_id=0xE222, name='Intel', vram_total=1024, integrated=False, intel=True)
        adapter = Adapter(1, 'Intel', 0x8086, 1024, 0, False)
        device = DeviceTelemetry(index=0, source='Intel sysfs', pci_bus=card.pdev, util_gpu=0)
        recorded = dataclasses.asdict(device)
        if wrong_gpu:
            recorded['pci_bus'] = '0000:05:00.0'
        writer = Mock()
        native_reads = []
        original_glob = Path.glob
        def glob(path, pattern):
            return iter([Path('/dev/dri/renderD128')]) if path.as_posix() == '/dev/dri' else original_glob(path, pattern)
        def native(*args):
            self.assertEqual(os.environ.get('GPUTASKMAN_DISABLE_HOST_FEED'), '1')
            native_reads.append(True)
            return device
        def frame():
            self.assertEqual(os.environ.get('GPUTASKMAN_DISABLE_HOST_FEED'), '1')
            return {'taken': 1000, 'device': recorded, 'processes': []}
        def run(sample, output, **kwargs):
            for _ in range(2):
                value = sample()
                output.write('record\n')
                output.flush()
            return 2
        args = ['intel_host_reader.py', '--pci', card.pdev, '--duration', '2'] + (['--record'] if record else [])
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ), patch.object(sys, 'argv', args), patch.object(sys, 'platform', 'linux'), \
                patch.object(os, 'geteuid', return_value=0, create=True), patch.object(Path, 'glob', glob), \
                patch.object(drm, 'cards', return_value=[card]), \
                patch('gputaskman.collector.enumerate_adapters', return_value=[adapter]), \
                patch('gputaskman.intel_host.FeedWriter', return_value=writer,
                      side_effect=ReaderRunning('already running') if writer_busy else None), \
                patch.object(drm, 'read_device', side_effect=native), \
                patch.object(drm, 'DrmCounters') as counters, \
                patch('gputaskman.record.source', return_value=contextlib.nullcontext(frame)) as source, \
                patch('gputaskman.record.run', side_effect=run), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if wrong_gpu:
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    cli.main()
            else:
                cli.main()
            source_called = source.called
            counters.return_value.close.assert_called_once()
        return writer, native_reads, source_called, stdout.getvalue(), stderr.getvalue()

    def test_device_only_mode_does_not_scan_processes_or_write_a_trace(self):
        writer, native, source_called, stdout, stderr = self.run_reader()
        self.assertEqual(len(native), 2)
        self.assertFalse(source_called)
        self.assertEqual(stdout, '')
        self.assertIn('intel-0000_04_00.0.json', stderr)
        self.assertEqual(writer.publish.call_args_list[-1].kwargs['state'], 'stopped')
        writer.close.assert_called_once()

    def test_record_mode_uses_the_standard_full_snapshot_source(self):
        writer, native, source_called, stdout, _ = self.run_reader(record=True)
        self.assertEqual(native, [])
        self.assertTrue(source_called)
        self.assertEqual(stdout, 'record\nrecord\n')
        self.assertEqual(writer.publish.call_count, 3)

    def test_wrong_gpu_stops_publication_and_closes_resources(self):
        writer, _, _, _, _ = self.run_reader(record=True, wrong_gpu=True)
        writer.publish.assert_called_once_with(None, state='stopped')
        writer.close.assert_called_once()

    def test_recording_does_not_stop_an_already_running_publisher(self):
        writer, _, source_called, stdout, stderr = self.run_reader(record=True, writer_busy=True)
        self.assertTrue(source_called)
        self.assertEqual(stdout, 'record\nrecord\n')
        writer.publish.assert_not_called()
        writer.close.assert_not_called()
        self.assertIn('without replacing', stderr)

    def test_administrator_preflight_does_not_try_to_elevate(self):
        with patch.object(sys, 'platform', 'linux'), \
                patch.object(sys, 'argv', ['reader', '--pci', '0000:04:00.0']), \
                patch.object(os, 'geteuid', return_value=1000, create=True), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                cli.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn('administrator', stderr.getvalue())
