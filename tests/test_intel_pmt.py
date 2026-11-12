"""Synthetic BMG G31 rev 16 telemetry; no live hardware claims."""
import errno
import struct
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from gputaskman import intel_pmt as pmt


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux PMT sysfs/pread provider")
class PmtTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.device = self.root / 'devices' / '0000:04:00.0'
        self.directory = self.root / 'pmt'
        self.directory.mkdir()
        self.device.mkdir(parents=True)
        self.ns = 1_000_000_000
        self.punit = self.node('telem14', pmt.PUNIT_GUID)
        self.oob = self.node('telem15', pmt.OOB_GUID)
        self.provider = pmt.PmtTelemetry(str(self.device), root=str(self.directory), clock=lambda: self.ns)
        self.addCleanup(self.provider.close)
        self.write()

    def node(self, name, guid, device=None):
        node = (device or self.device) / 'intel_pmt' / name
        node.mkdir(parents=True)
        (node / 'guid').write_text(hex(guid))
        (node / 'size').write_text('4096')
        (node / 'offset').write_text('144')  # BMG offsets are relative to the binary file, not this mapping offset
        (node / 'telem').write_bytes(bytes(4096))
        (self.directory / name).symlink_to(node, target_is_directory=True)
        return node

    def write(self, *, read=0, write=0, rx=0, tx=0, mask=1, mhz=1800, maximum=40000):
        data = bytearray(4096)
        struct.pack_into('<I', data, 56, (maximum << 16) | mhz)
        (self.punit / 'telem').write_bytes(data)
        data = bytearray(4096)
        struct.pack_into('<I', data, 3688, mask)
        # Independent reference offsets: CH0 GT 32-byte reads/writes; all other request counters idle.
        for offset, value in ((376, read), (392, write), (280, rx), (288, tx)):
            struct.pack_into('<Q', data, offset, value)
        (self.oob / 'telem').write_bytes(data)

    def tick(self):
        self.ns += 1_000_000_000

    def test_bytes_rates_use_request_size_and_preserve_measured_idle(self):
        first, status, sources = self.provider.sample()
        self.assertEqual(first['mem_clock'], 1800)
        self.assertEqual(first['mem_bandwidth_max_bps'], 500_000_000_000)
        self.assertNotIn('mem_read_bps', first)
        self.assertIn('second', status['mem_read_bps'])
        self.assertIn('mem_read_bps', sources)
        self.tick()
        self.write(read=10_000_000, write=5_000_000, rx=2_097_152, tx=1_048_576)
        values, status, _ = self.provider.sample()
        self.assertEqual((values['mem_read_bps'], values['mem_write_bps']), (320_000_000, 160_000_000))
        self.assertAlmostEqual(values['mem_bandwidth_pct'], .096)
        self.assertEqual((values['pcie_rx_kbs'], values['pcie_tx_kbs']), (2048, 1024))
        self.assertNotIn('mem_read_bps', status)
        self.tick()
        values, _, _ = self.provider.sample()
        self.assertEqual((values['mem_read_bps'], values['mem_write_bps'], values['pcie_rx_kbs']), (0, 0, 0))

    def test_actual_memory_clock_zero_and_unknown_maximum(self):
        self.write(mhz=0, maximum=0)
        self.provider.sample()
        self.tick()
        values, status, _ = self.provider.sample()
        self.assertEqual(values['mem_clock'], 0)
        self.assertEqual(values['mem_read_bps'], 0)
        self.assertNotIn('mem_bandwidth_pct', values)
        self.assertIn('maximum', status['mem_bandwidth_pct'])

    def test_reset_rearms_without_turning_into_a_spike(self):
        self.write(read=100, rx=100)
        self.provider.sample()
        self.tick()
        self.write(read=1, rx=1)
        values, status, _ = self.provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertNotIn('pcie_rx_kbs', values)
        self.assertIn('reset', status['mem_read_bps'])
        self.tick()
        self.write(read=2, rx=2)
        values, _, _ = self.provider.sample()
        self.assertEqual(values['mem_read_bps'], 32)

    def test_mask_changes_with_same_number_of_channels_reprime(self):
        self.provider.sample()
        self.tick()
        self.write(mask=2)
        values, _, _ = self.provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.tick()
        values, _, _ = self.provider.sample()
        self.assertEqual(values['mem_read_bps'], 0)

    def test_documented_msu6_channel1_read_offset_is_used(self):
        self.write(mask=1 << 6)
        self.provider.sample()
        self.tick()
        data = bytearray((self.oob / 'telem').read_bytes())
        struct.pack_into('<Q', data, 1576, 50)  # Intel rev 16 map: MSU6 CH1 GT32 RD
        struct.pack_into('<Q', data, 1416, 999)  # same slot in a regular-stride layout would be wrong
        (self.oob / 'telem').write_bytes(data)
        values, _, _ = self.provider.sample()
        self.assertEqual(values['mem_read_bps'], 1600)

    def test_other_pci_device_and_neighboring_guid_are_not_guessed(self):
        other = self.root / 'devices' / '0000:05:00.0'
        self.node('telem30', pmt.OOB_GUID, device=other)
        (self.oob / 'guid').write_text('0x5e2f8310')
        values, status, _ = self.provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertIn('unsupported PMT GUID', status['mem_read_bps'])
        self.assertNotIn(pmt.OOB_GUID, self.provider.nodes)

    def test_duplicate_matching_endpoints_are_ambiguous(self):
        self.node('telem16', pmt.OOB_GUID)
        values, status, _ = self.provider.sample()
        self.assertNotIn('pcie_rx_kbs', values)
        self.assertIn('ambiguous', status['pcie_rx_kbs'])

    def test_permission_failure_recovers_without_restart_and_requires_new_interval(self):
        with patch.object(pmt.os, 'open', side_effect=PermissionError(errno.EACCES, 'denied')):
            values, status, _ = self.provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertIn('access denied', status['mem_read_bps'])
        self.ns += 6_000_000_000
        values, _, _ = self.provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.tick()
        values, _, _ = self.provider.sample()
        self.assertEqual(values['mem_read_bps'], 0)

    def test_short_read_and_torn_counter_stay_unknown(self):
        self.provider.sample()
        self.tick()
        (self.oob / 'telem').write_bytes(bytes(100))
        values, status, _ = self.provider.sample()
        self.assertNotIn('pcie_rx_kbs', values)
        self.assertIn('short', status['pcie_rx_kbs'])
        self.write()
        self.tick()
        self.provider.sample()
        self.tick()
        first = bytes(4096)
        last = bytearray(first)
        struct.pack_into('<I', last, 284, 1)
        node = self.provider.nodes[pmt.OOB_GUID]
        with patch.object(node, 'frames', return_value=(first, first, bytes(last))):
            values, status, _ = self.provider.sample()
        self.assertNotIn('pcie_rx_kbs', values)
        self.assertIn('stable', status['pcie_rx_kbs'])

    def test_zero_or_out_of_range_channel_mask_is_unknown(self):
        for mask in (0, 256):
            with self.subTest(mask=mask):
                self.write(mask=mask)
                values, status, _ = self.provider.sample()
                self.assertNotIn('mem_read_bps', values)
                self.assertIn('mask', status['mem_read_bps'])


class PciLinkTests(unittest.TestCase):
    def test_capability_link_status_overrides_untrusted_sysfs_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = bytearray(256)
            struct.pack_into('<H', raw, 6, 0x10)
            raw[0x34], raw[0x40] = 0x40, 0x10
            struct.pack_into('<H', raw, 0x52, 5 | (16 << 4))
            (root / 'config').write_bytes(raw)
            (root / 'current_link_speed').write_text('2.5 GT/s PCIe')
            (root / 'current_link_width').write_text('1')
            self.assertEqual(pmt.pci_link(tmp), ({'pcie_gen': 5, 'pcie_width': 16}, ''))

    def test_truncated_configuration_and_cycle_cannot_become_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = bytearray(64)
            struct.pack_into('<H', raw, 6, 0x10)
            raw[0x34] = 0x40
            (root / 'config').write_bytes(raw)
            values, reason = pmt.pci_link(tmp)
            self.assertEqual(values, {})
            self.assertIn('read access', reason)
            raw.extend(bytes(192))
            raw[0x40], raw[0x41] = 1, 0x40
            (root / 'config').write_bytes(raw)
            self.assertIn('invalid', pmt.pci_link(tmp)[1])
