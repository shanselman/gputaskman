"""GPU-owned Intel hwmon labels; no live device or administrator access required."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gputaskman import drm


class IntelHwmonTemperatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.device = Path(self.tmp.name) / 'gpu'
        self.hwmon = self.device / 'hwmon/hwmon8'
        self.hwmon.mkdir(parents=True)
        self.card = drm.DrmCard(key=1, pdev='0000:04:00.0', sysfs=str(self.device), driver='xe',
                                vendor_id=0x8086, device_id=0xe223, name='Intel',
                                vram_total=32 << 30, integrated=False, intel=True)

    def sensor(self, number, label, value):
        (self.hwmon / f'temp{number}_label').write_text(label)
        (self.hwmon / f'temp{number}_input').write_text(str(value))

    def read(self):
        with patch.object(drm, '_intel_pmu_rates', return_value=(None, None)):
            return drm.read_device(self.card, 0).temp

    def test_xe_package_sensor_works_without_temp1(self):
        self.sensor(2, 'pkg\n', 42000)
        self.assertEqual(self.read(), 42)

    def test_package_label_takes_priority_over_gpu_edge_and_unlabelled_first_sensor(self):
        self.sensor(1, 'vram', 90000)
        self.sensor(2, 'edge', 50000)
        self.sensor(3, 'gpu', 45000)
        self.sensor(4, 'pkg', 42000)
        self.assertEqual(self.read(), 42)

    def test_invalid_package_input_can_fall_back_to_another_gpu_temperature_label(self):
        self.sensor(2, 'pkg', 'unreadable')
        self.sensor(3, 'gpu', 43000)
        self.sensor(4, 'edge', 44000)
        self.assertEqual(self.read(), 43)

    def test_measured_zero_is_preserved(self):
        self.sensor(2, 'pkg', 0)
        self.sensor(3, 'gpu', 43000)
        self.assertEqual(self.read(), 0)

    def test_auxiliary_vram_sensor_is_not_used_as_gpu_temperature(self):
        self.sensor(3, 'vram', 90000)
        self.sensor(6, 'vram_ch_0', 91000)
        self.assertIsNone(self.read())

    def test_missing_sensor_does_not_borrow_another_devices_temperature(self):
        other = self.device.parent / 'other-gpu/hwmon/hwmon0'
        other.mkdir(parents=True)
        (other / 'temp2_label').write_text('pkg')
        (other / 'temp2_input').write_text('99000')
        self.assertIsNone(self.read())

    def test_amd_retains_edge_priority(self):
        self.sensor(1, 'edge', 57000)
        self.sensor(2, 'pkg', 99000)
        self.card.intel = False
        self.card.driver = 'amdgpu'
        self.assertEqual(self.read(), 57)

    def test_unlabelled_temp1_fallback_is_preserved(self):
        (self.hwmon / 'temp1_input').write_text('48000')
        self.assertEqual(self.read(), 48)

    def test_unreadable_amd_edge_keeps_existing_unknown_behavior(self):
        (self.hwmon / 'temp1_input').write_text('48000')
        self.sensor(2, 'edge', 'unreadable')
        self.card.intel = False
        self.card.driver = 'amdgpu'
        self.assertIsNone(self.read())


if __name__ == '__main__':
    unittest.main()
