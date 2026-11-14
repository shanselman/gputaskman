"""Auxiliary Xe sensors: independent readings, bounded labels and missing inputs."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from gputaskman import drm
from gputaskman.report import auxiliary_temperature_label
from gputaskman.telemetry import DeviceTelemetry


class AuxiliaryTemperatureTests(unittest.TestCase):
    def test_hwmon_collection_preserves_each_sensor_and_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            hwmon = Path(tmp) / 'hwmon' / 'hwmon0'
            hwmon.mkdir(parents=True)
            readings = [('vram', '78000'), ('mctrl', '48250'), ('pcie', '0'),
                        ('vram_ch_0', '76000'), ('vram_ch_1', None),
                        ('vram_ch_2', 'bad'), ('vram_ch_3', '-274000'),
                        ('vram_ch_16', '99000'), ('arbitrary', '90000')]
            for index, (label, value) in enumerate(readings, 1):
                (hwmon / f'temp{index}_label').write_text(label)
                if value is not None:
                    (hwmon / f'temp{index}_input').write_text(value)
            card = drm.DrmCard(key=1, pdev='0000:04:00.0', sysfs=tmp, driver='xe',
                               vendor_id=0x8086, device_id=0xe222, name='Intel', intel=True,
                               vram_total=32 * 1024**3, integrated=False)
            provider = Mock(query_error='', pmu_error='')
            provider.memory.return_value = (None, None)
            with patch.object(drm, '_xe_device', return_value=provider), \
                    patch.object(drm, '_intel_pmu_rates', return_value=(None, None)), \
                    patch.object(drm, '_xe_extra_telemetry'):
                device = drm.read_device(card, 0)
            self.assertEqual(device.vram_temp, 78)
            self.assertEqual(device.temperature_sensors_c, {'mctrl': 48.25, 'pcie': 0, 'vram_ch_0': 76})
            self.assertEqual(json.loads(json.dumps(dataclasses.asdict(device)))['temperature_sensors_c'],
                             device.temperature_sensors_c)
            self.assertIn('Intel hwmon', device.metric_sources['temperature_sensors_c'])

    def test_display_reports_hottest_read_channel_without_implying_all_are_available(self):
        device = DeviceTelemetry(index=0, temperature_sensors_c={
            'mctrl': 0, 'pcie': 66, 'vram_ch_0': 70, 'vram_ch_12': 78})
        text = auxiliary_temperature_label(device)
        self.assertIn('Memory controller 0°C', text)
        self.assertIn('PCIe 66°C', text)
        self.assertIn('hottest 12 78°C (2 read)', text)
        self.assertEqual(auxiliary_temperature_label(DeviceTelemetry(index=0)), '')
