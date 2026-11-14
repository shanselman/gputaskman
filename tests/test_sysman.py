"""Level Zero ABI and read-only provider fixtures; no runtime or GPU required."""
import ctypes as c
import os
import sys
import unittest
from unittest.mock import patch

from gputaskman import sysman as sm


class AbiTests(unittest.TestCase):
    @unittest.skipUnless(c.sizeof(c.c_void_p) == 8, '64-bit Sysman ABI')
    def test_published_c_layout_sizes_and_offsets(self):
        # Independent 64-bit C ABI values from include/zes_api.h.
        for structure, size in ((sm.Address, 16), (sm.Speed, 16), (sm.PciProperties, 64),
                                (sm.PciState, 48), (sm.PciStats, 56), (sm.MemProperties, 48),
                                (sm.MemBandwidth, 32), (sm.FreqProperties, 56), (sm.FreqState, 64),
                                (sm.FanProperties, 48)):
            self.assertEqual(c.sizeof(structure), size, structure.__name__)
        self.assertEqual(sm.PciProperties.address.offset, 16)
        self.assertEqual(sm.PciState.speed.offset, 32)
        self.assertEqual(sm.FreqState.actual.offset, 48)
        self.assertEqual(sm.FanProperties.maxRPM.offset, 36)
        self.assertEqual(sm.MemBandwidth.timestamp.offset, 24)


class FakeApi:
    def __init__(self):
        self.calls = []
        self.addresses = {100: (0, 4, 0, 0), 200: (0, 5, 0, 0)}
        self.memory_handles = [10, 11]  # local memory, then system RAM
        self.frequency_handles = [20, 21]  # GPU, then memory
        self.fan_handles = [30]
        self.time = 1_000_000
        self.read, self.write, self.rx, self.tx = 1000, 2000, 3000, 4000
        self.maximum = 10_000_000_000
        self.have_pci_counters = 1
        self.gen, self.width, self.frequency = 5, 16, 1800
        self.rpm, self.percent, self.max_rpm, self.units = 800, 40, 2000, 3
        self.memory_error = self.init_error = 0

    def enum(self, name, values, count, array):
        self.calls.append(name)
        count._obj.value = len(values)
        if array is not None:
            for i, value in enumerate(values):
                array[i] = value
        return 0

    def zesInit(self, flags):
        self.calls.append('zesInit')
        return self.init_error

    def zesDriverGet(self, count, array):
        return self.enum('zesDriverGet', [1], count, array)

    def zesDeviceGet(self, driver, count, array):
        return self.enum('zesDeviceGet', list(self.addresses), count, array)

    def zesDevicePciGetProperties(self, device, pointer):
        self.calls.append('zesDevicePciGetProperties')
        prop = pointer._obj
        prop.address = sm.Address(*self.addresses[device])
        prop.haveBandwidthCounters = self.have_pci_counters
        prop.maxSpeed = sm.Speed(5, 16, 64_000_000_000)
        return 0

    def zesDeviceEnumMemoryModules(self, device, count, array):
        return self.enum('zesDeviceEnumMemoryModules', self.memory_handles, count, array)

    def zesMemoryGetProperties(self, handle, pointer):
        pointer._obj.location = 1 if handle == 10 else 0
        return 0

    def zesMemoryGetBandwidth(self, handle, pointer):
        self.calls.append('zesMemoryGetBandwidth')
        assert handle == 10, 'must never query system-RAM bandwidth as VRAM'
        sample = pointer._obj
        sample.readCounter, sample.writeCounter = self.read, self.write
        sample.timestamp, sample.maxBandwidth = self.time, self.maximum
        return self.memory_error

    def zesDeviceEnumFrequencyDomains(self, device, count, array):
        return self.enum('zesDeviceEnumFrequencyDomains', self.frequency_handles, count, array)

    def zesFrequencyGetProperties(self, handle, pointer):
        pointer._obj.type = 1 if handle == 21 else 0
        return 0

    def zesFrequencyGetState(self, handle, pointer):
        self.calls.append('zesFrequencyGetState')
        assert handle == 21
        pointer._obj.actual = self.frequency
        return 0

    def zesDeviceEnumFans(self, device, count, array):
        return self.enum('zesDeviceEnumFans', self.fan_handles, count, array)

    def zesFanGetProperties(self, handle, pointer):
        pointer._obj.supportedUnits, pointer._obj.maxRPM = self.units, self.max_rpm
        return 0

    def zesFanGetState(self, handle, units, pointer):
        self.calls.append('zesFanGetState')
        pointer._obj.value = self.rpm if units == 0 else self.percent
        return 0

    def zesDevicePciGetStats(self, device, pointer):
        self.calls.append('zesDevicePciGetStats')
        sample = pointer._obj
        sample.timestamp, sample.rxCounter, sample.txCounter = self.time, self.rx, self.tx
        return 0

    def zesDevicePciGetState(self, device, pointer):
        self.calls.append('zesDevicePciGetState')
        pointer._obj.speed.gen, pointer._obj.speed.width = self.gen, self.width
        return 0


@unittest.skipUnless(sys.platform.startswith('linux') and c.sizeof(c.c_void_p) == 8, 'native Linux provider')
class SysmanTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'ZES_ENABLE_SYSMAN': '0'})
        env.start()
        self.addCleanup(env.stop)
        self.api = FakeApi()

    def provider(self):
        provider = sm.Sysman('0000:04:00.0', self.api)
        self.addCleanup(provider.close)
        return provider

    def test_pci_match_and_byte_rates_and_read_only_initialization(self):
        provider = self.provider()
        self.assertEqual(provider.handle, 100)
        values, status, _ = provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertIn('second', status['mem_read_bps'])
        self.assertEqual((values['mem_clock'], values['fan'], values['fan_rpm']), (1800, 40, 800))
        self.api.time += 1_000_000
        self.api.read += 1_000_000_000
        self.api.write += 2_000_000_000
        self.api.rx += 2_097_152
        self.api.tx += 1_048_576
        values, _, sources = provider.sample()
        self.assertEqual((values['mem_read_bps'], values['mem_write_bps']), (1e9, 2e9))
        self.assertEqual(values['mem_bandwidth_pct'], 30)
        self.assertEqual((values['pcie_rx_kbs'], values['pcie_tx_kbs']), (2048, 1024))
        self.assertEqual((values['pcie_gen'], values['pcie_width']), (5, 16))
        self.assertIn('Sysman', sources['mem_read_bps'])
        self.assertTrue(all(name.startswith(('zesInit', 'zesDriverGet', 'zesDeviceGet', 'zesDeviceEnum', 'zesDevicePciGet',
                                            'zesMemoryGet', 'zesFrequencyGet', 'zesFanGet')) for name in self.api.calls))

    def test_unknown_is_not_idle_and_advertised_link_is_not_live_link(self):
        self.api.have_pci_counters = 0
        self.api.gen = self.api.width = -1
        self.api.frequency = -1
        provider = self.provider()
        values, status, _ = provider.sample()
        self.assertNotIn('pcie_gen', values)
        self.assertNotIn('pcie_width', values)
        self.assertNotIn('mem_clock', values)
        self.assertNotIn('pcie_rx_kbs', values)
        self.assertIn('no PCIe byte', status['pcie_rx_kbs'])
        self.assertNotIn('zesDevicePciGetStats', self.api.calls)
        self.api.frequency = 0
        self.assertEqual(provider.sample()[0]['mem_clock'], 0)

    def test_measured_idle_stall_and_reset(self):
        provider = self.provider()
        provider.sample()
        self.api.time += 1_000_000
        values, _, _ = provider.sample()
        self.assertEqual((values['mem_read_bps'], values['pcie_rx_kbs']), (0, 0))
        values, status, _ = provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertIn('stalled', status['mem_read_bps'])
        self.api.time += 1_000_000
        self.api.read = 0
        values, _, _ = provider.sample()
        self.assertNotIn('mem_read_bps', values)

    def test_permission_failure_recovers_without_false_zero(self):
        provider = self.provider()
        provider.sample()
        self.api.memory_error = 0x70010000
        values, status, _ = provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertIn('access denied', status['mem_read_bps'])
        self.api.memory_error = 0
        self.api.time += 1_000_000
        self.assertNotIn('mem_read_bps', provider.sample()[0])
        self.api.time += 1_000_000
        self.assertEqual(provider.sample()[0]['mem_read_bps'], 0)

    def test_lost_device_requests_provider_rediscovery(self):
        provider = self.provider()
        provider.sample()
        self.api.memory_error = 0x70000001
        values, status, _ = provider.sample()
        self.assertNotIn('mem_read_bps', values)
        self.assertIn('device lost', status['mem_read_bps'])
        self.assertIn('discovery', provider.error)

    def test_fan_percentage_uses_known_maximum_only(self):
        self.api.units = 1
        provider = self.provider()
        self.assertEqual(provider.sample()[0]['fan'], 40)
        self.api.max_rpm = -1
        provider = self.provider()
        self.assertNotIn('fan', provider.sample()[0])
        self.api.max_rpm = 2000
        self.api.rpm = 3000
        provider = self.provider()
        self.assertNotIn('fan', provider.sample()[0])

    def test_fan_percentage_survives_independent_rpm_failure(self):
        provider = self.provider()
        original = self.api.zesFanGetState
        def state(handle, units, pointer):
            return 0x78000003 if units == 0 else original(handle, units, pointer)
        self.api.zesFanGetState = state
        values, status, _ = provider.sample({'fan', 'fan_rpm'})
        self.assertEqual(values['fan'], 40)
        self.assertNotIn('fan_rpm', values)
        self.assertIn('unsupported', status['fan_rpm'])

    def test_ambiguous_pci_identity_and_domain_are_not_guessed(self):
        self.api.addresses[200] = (0, 4, 0, 0)
        provider = self.provider()
        self.assertIsNone(provider.handle)
        self.assertIn('unique', provider.error)
        self.api.addresses = {100: (1, 4, 0, 0)}
        self.assertIsNone(self.provider().handle)

    def test_optional_loader_is_not_a_required_dependency(self):
        with patch.object(sm.c, 'CDLL', side_effect=OSError('not installed')):
            provider = sm.Sysman('0000:04:00.0')
        values, status, _ = provider.sample()
        self.assertEqual(values, {})
        self.assertIn('not installed', status['sysman'])

    def test_legacy_initialization_is_not_used_on_xe(self):
        with patch.dict(os.environ, {'ZES_ENABLE_SYSMAN': '1'}):
            provider = self.provider()
        self.assertIsNone(provider.handle)
        self.assertEqual(self.api.calls, [])
        self.assertIn('legacy', provider.error)

    def test_native_success_does_not_cause_duplicate_sysman_traffic_probes(self):
        provider = self.provider()
        self.api.calls.clear()
        values, _, _ = provider.sample({'fan'})
        self.assertIn('fan', values)
        self.assertNotIn('zesMemoryGetBandwidth', self.api.calls)
        self.assertNotIn('zesDevicePciGetStats', self.api.calls)
        self.assertNotIn('zesDevicePciGetState', self.api.calls)
        self.assertNotIn('zesFrequencyGetState', self.api.calls)

    def test_duplicate_handles_and_unbounded_enumeration_are_refused(self):
        self.api.memory_handles = [10, 10]
        provider = self.provider()
        self.assertNotIn('memory', provider.domains)
        self.assertIn('enumeration', provider.domain_errors['memory'])
        self.api.memory_handles = list(range(100))
        provider = self.provider()
        self.assertIn('bound', provider.domain_errors['memory'])
