"""Provider precedence and public output; synthetic, not live B65 validation."""
import unittest
from unittest.mock import Mock, patch

from gputaskman import drm
from gputaskman.telemetry import DeviceTelemetry


class ProviderIntegrationTests(unittest.TestCase):
    def card(self, native, fallback):
        card = drm.DrmCard(key=1, pdev="0000:04:00.0", sysfs="/missing", driver="xe",
                           vendor_id=0x8086, device_id=0xe222, name="Intel", intel=True,
                           vram_total=32 * 1024**3, integrated=False)
        card.pmt_provider = Mock()
        card.pmt_provider.sample.return_value = native
        card.sysman_provider = Mock(error="", domain_errors={})
        card.sysman_provider.sample.return_value = fallback
        return card

    def collect(self, card, **fields):
        dev = DeviceTelemetry(index=0, **fields)
        with patch("gputaskman.intel_pmt.pci_link", return_value=({}, "truncated config")):
            drm._xe_extra_telemetry(card, dev)
        return dev

    def test_native_idle_preserves_zero_and_avoids_duplicate_rate_probes(self):
        fields = {"mem_read_bps": 0, "mem_write_bps": 0, "mem_clock": 0,
                  "pcie_rx_kbs": 0, "pcie_tx_kbs": 0}
        card = self.card((fields, {}, {k: "PMT" for k in fields}), ({}, {}, {}))
        dev = self.collect(card)
        self.assertEqual((dev.mem_read_bps, dev.mem_write_bps, dev.mem_clock), (0, 0, 0))
        wanted = card.sysman_provider.sample.call_args.args[0]
        self.assertFalse(wanted.intersection(fields))
        self.assertNotIn("mem_bandwidth_pct", wanted)
        self.assertEqual(dev.metric_sources["mem_read_bps"], "PMT")
        self.assertNotIn("mem_read_bps", dev.metric_status)

    def test_fallback_replaces_whole_rate_family_and_preserves_hwmon(self):
        native = {"mem_bandwidth_max_bps": 1000, "mem_clock": 1800}
        fallback = {"mem_read_bps": 10, "mem_write_bps": 20, "fan": 40, "fan_rpm": 800}
        card = self.card((native, {"mem_read_bps": "denied"}, {k: "PMT" for k in native}),
                         (fallback, {}, {k: "Sysman" for k in fallback}))
        dev = self.collect(card, fan_rpm=960)
        self.assertEqual((dev.mem_read_bps, dev.mem_write_bps), (10, 20))
        self.assertIsNone(dev.mem_bandwidth_max_bps)  # no PMT max mixed into Sysman rates
        self.assertIsNone(dev.mem_bandwidth_pct)
        self.assertNotIn("mem_bandwidth_max_bps", dev.metric_sources)
        self.assertEqual((dev.mem_clock, dev.fan, dev.fan_rpm), (1800, 40, 960))
        self.assertNotIn("mem_read_bps", dev.metric_status)

    def test_unavailable_data_keeps_reasons_and_nulls(self):
        card = self.card(({}, {"mem_read_bps": "PMT denied"}, {}),
                         ({}, {"sysman": "loader missing"}, {}))
        dev = self.collect(card)
        self.assertIsNone(dev.mem_read_bps)
        self.assertIsNone(dev.pcie_gen)
        self.assertEqual(dev.metric_status["mem_read_bps"], "PMT denied")
        self.assertEqual(dev.metric_status["sysman"], "loader missing")
        self.assertEqual(dev.metric_status["pcie_gen"], "truncated config")

    def test_collector_closes_all_owned_providers(self):
        card = self.card(({}, {}, {}), ({}, {}, {}))
        providers = [card.pmt_provider, card.sysman_provider, Mock()]
        card.xe_provider = providers[-1]
        drm.DrmCounters([card]).close()
        for provider in providers:
            provider.close.assert_called_once()
        self.assertIsNone(card.pmt_provider)
        self.assertIsNone(card.sysman_provider)


if __name__ == "__main__":
    unittest.main()
