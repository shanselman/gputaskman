from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from gputaskman.cli import main
from gputaskman.summary import load, split_runs, summarize, to_text

GiB = 1 << 30
MiB = 1 << 20
_MISSING = object()


def proc(name, vram=None, pct=None, host=None, swapped=_MISSING):
    row = {"pid": 1, "host": host, "name": name, "vram_bytes": vram, "gpu_pct": pct}
    if swapped is not _MISSING:
        row["graphics_swapped_bytes"] = swapped
    return row


def snap(taken, procs=(), used=None, gpu="RTX", backend="wddm", device=None, system=None):
    return {"taken": taken, "gpu": gpu, "backend": backend, "vram_total": 16 * GiB, "vram_used": used,
            "shared_used": None, "device": device, "processes": list(procs),
            **({"system": system} if system is not None else {})}


class SummaryTests(unittest.TestCase):
    def test_nulls_are_skipped_and_values_are_kept_as_recorded(self):
        samples = [snap(0, used=GiB, device={"util_gpu": 50, "temp": None, "power_w": None}),
                   snap(1, used=None, device={"util_gpu": 122.3, "temp": None, "power_w": None}),
                   snap(2, used=3 * GiB, device=None)]
        m = summarize(samples)["runs"][0]["metrics"]
        self.assertEqual(m["vram_used"], {"label": "VRAM used", "unit": "bytes", "avg": 2 * GiB, "peak": 3 * GiB,
                                          "readings": 2})
        self.assertEqual(m["gpu_load"]["peak"], 122.3)  # never capped
        self.assertNotIn("temperature", m)  # no readings: left out, not shown as 0
        self.assertNotIn("shared_used", m)
        self.assertNotIn("system_cpu", m)

    def test_system_section_is_used_when_present(self):
        sys_ok = {"cpu": {"pct": 10.0}, "ram": {"used": 8 * GiB}}
        m = summarize([snap(0, system=sys_ok), snap(1, system={"error": "denied"})])["runs"][0]["metrics"]
        self.assertEqual((m["system_cpu"]["peak"], m["system_cpu"]["readings"]), (10.0, 1))
        self.assertEqual(m["system_ram_used"]["avg"], 8 * GiB)

    def test_processes_sum_same_name_per_sample_and_count_measured_bytes_only(self):
        samples = [snap(0, [proc("llama", GiB, 40), proc("llama", GiB, 30), proc("wsl", None, 5)]),
                   snap(1, [proc("llama", GiB, 90), proc("wsl", None, None), proc("llama", None, None, host="windows")])]
        procs = summarize(samples)["runs"][0]["processes"]
        by = {(p["host"], p["name"]): p for p in procs}
        self.assertEqual(by[(None, "llama")]["peak_vram_bytes"], 2 * GiB)  # two rows in sample 0
        self.assertEqual(by[(None, "llama")]["peak_gpu_pct"], 90)
        self.assertIsNone(by[(None, "wsl")]["peak_vram_bytes"])  # inferred/unknown: n/a, never 0
        self.assertEqual(by[("windows", "llama")]["samples_seen"], 1)  # host rows are kept apart
        self.assertEqual(procs[0]["name"], "llama")
        text = to_text(summarize(samples))
        self.assertIn("W:llama", text)
        self.assertRegex(text, r"wsl\s+n/a\s+5\.0%")
        self.assertNotIn("COMPRESSED", text)  # no swapped readings: the column stays out

    def test_compressed_graphics_peaks_on_its_own_and_a_missing_value_stays_unknown(self):
        samples = [snap(0, [proc("app", GiB, 10, swapped=512 * MiB), proc("app", GiB, 5, swapped=0),
                            proc("blank", GiB, 1, swapped=None), proc("bad", 0, 0, swapped=True),
                            proc("host", GiB, 1, host="windows", swapped=64 * MiB)]),
                   snap(1, [proc("app", 0, 1, swapped=None), proc("zero", 0, 0, swapped=0),
                            proc("bad", 0, 0, swapped=float("nan"))])]
        windows = proc("paged", GiB, 1)
        windows["paged_out_bytes"] = 8 * GiB
        samples[1]["processes"].append(windows)
        procs = summarize(samples)["runs"][0]["processes"]
        by = {(p["host"], p["name"]): p for p in procs}
        self.assertEqual(by[(None, "app")]["peak_vram_bytes"], 2 * GiB)  # swapped is not added in
        self.assertEqual(by[(None, "app")]["peak_graphics_swapped_bytes"], 512 * MiB)  # 512 + a measured 0
        self.assertIsNone(by[(None, "blank")]["peak_graphics_swapped_bytes"])  # null is not a 0
        self.assertEqual(by[(None, "zero")]["peak_graphics_swapped_bytes"], 0)
        self.assertIsNone(by[(None, "bad")]["peak_graphics_swapped_bytes"])  # bool and nan are not byte counts
        self.assertIsNone(by[(None, "paged")]["peak_graphics_swapped_bytes"])  # Windows paged estimate is a different field
        self.assertEqual(by[("windows", "host")]["peak_graphics_swapped_bytes"], 64 * MiB)  # host rows stay apart
        self.assertEqual(procs[0]["name"], "app")  # its VRAM is the larger peak
        text = to_text(summarize(samples))
        self.assertIn("COMPRESSED", text)
        self.assertRegex(text, r"app\s+2\.00 GB\s+512 MB\s+15\.0%")
        self.assertRegex(text, r"zero\s+0 MB\s+0 MB")
        self.assertRegex(text, r"blank\s+1\.00 GB\s+n/a")
        self.assertIn("not added to peak VRAM", text)

    def test_the_larger_memory_peak_ranks_a_process_and_the_peaks_stay_separate(self):
        samples = [snap(0, [proc("vram", 2 * GiB, 1), proc("compressed", 0, 1, swapped=4 * GiB),
                            proc("unknown", None, 90)])]
        procs = summarize(samples)["runs"][0]["processes"]
        self.assertEqual([p["name"] for p in procs], ["compressed", "vram", "unknown"])
        self.assertEqual(procs[0]["peak_vram_bytes"], 0)  # the swapped peak does not replace it
        self.assertEqual(procs[0]["peak_graphics_swapped_bytes"], 4 * GiB)
        text = to_text(summarize(samples), top=1)
        self.assertRegex(text, r"compressed\s+0 MB\s+4\.00 GB")
        self.assertIn("2 more process(es)", text)
        self.assertNotRegex(text, r"(?m)^vram\s")

    def test_process_vram_peaks_ignore_inferred_and_shared_resident_bytes(self):
        samples = [
            snap(0, [
                {**proc("inferred"), "vram_inferred_bytes": 2 * GiB},
                {**proc("shared"), "vram_resident_bytes": 3 * GiB},
                {**proc("measured", GiB), "vram_inferred_bytes": 8 * GiB, "vram_resident_bytes": 7 * GiB},
                {**proc("same-name", GiB), "vram_inferred_bytes": 6 * GiB},
                {**proc("same-name"), "vram_resident_bytes": 5 * GiB},
            ]),
            snap(1, [
                {**proc("inferred"), "vram_inferred_bytes": 4 * GiB},
                {**proc("shared"), "vram_resident_bytes": 5 * GiB},
                {**proc("same-name", 2 * GiB), "vram_resident_bytes": 9 * GiB},
            ]),
        ]

        procs = summarize(samples)["runs"][0]["processes"]
        by = {(p["host"], p["name"]): p for p in procs}
        self.assertIsNone(by[(None, "inferred")]["peak_vram_bytes"])
        self.assertIsNone(by[(None, "shared")]["peak_vram_bytes"])
        self.assertEqual(by[(None, "measured")]["peak_vram_bytes"], GiB)
        self.assertEqual(by[(None, "same-name")]["peak_vram_bytes"], 2 * GiB)
        self.assertEqual(procs[0]["name"], "same-name")

    def test_oversized_values_stay_as_recorded_and_do_not_fuse_columns(self):
        text = to_text(summarize([snap(0, [proc("python.exe", 8 * GiB, 18465941443603.8)])]))
        self.assertIn("8.00 GB 18465941443603.8%", text)

    def test_runs_split_at_long_gaps_clock_steps_and_gpu_changes(self):
        taken = [0, 1, 2, 3, 500, 501, 502, 400, 401]
        samples = [snap(t) for t in taken] + [snap(402, gpu="Other")]
        runs = split_runs(samples)
        self.assertEqual([[s["taken"] for s in r] for r in runs], [[0, 1, 2, 3], [500, 501, 502], [400, 401], [402]])
        # a slow interval is not a gap: 5 s samples with one 30 s hiccup stay one run
        self.assertEqual(len(split_runs([snap(t) for t in (0, 5, 10, 40, 45)])), 1)

    def test_same_name_adapters_have_separate_vram_summaries(self):
        first = dict(snap(0, used=GiB), adapter={"pci_address": [1, 0, 0], "pci_domain": 0})
        second = dict(snap(1, used=3 * GiB), adapter={"pci_address": [1, 0, 0], "pci_domain": 1})
        third = dict(snap(2, used=5 * GiB), adapter={"pci_address": [2, 0, 0], "pci_domain": 1})

        runs = summarize([first, second, third])["runs"]
        self.assertEqual([r["metrics"]["vram_used"]["avg"] for r in runs], [GiB, 3 * GiB, 5 * GiB])

    def test_runs_keep_pci_matches_and_older_snapshots_together(self):
        first = dict(snap(0), adapter={"pci_address": [1, 0, 0], "luid": 1})
        second = dict(snap(1), adapter={"pci_address": [1, 0, 0], "luid": 2})
        older = snap(2)

        self.assertEqual(split_runs([first, second, older]), [[first, second, older]])

    def test_runs_remember_pci_identity_across_unknown_samples(self):
        first = dict(snap(0), adapter={"pci_address": [1, 0, 0], "pci_domain": 0, "luid": 1})
        older = snap(1)
        same = dict(snap(2), adapter={"pci_address": [1, 0, 0], "luid": 2})
        different = dict(snap(3), adapter={"pci_address": [1, 0, 0], "pci_domain": 1, "luid": 2})

        self.assertEqual(split_runs([first, older, same]), [[first, older, same]])
        self.assertEqual(split_runs([first, older, different]), [[first, older], [different]])
        self.assertEqual(split_runs([first, older, same, different]), [[first, older, same], [different]])

    def test_runs_remember_non_windows_keys_across_unknown_samples(self):
        first = dict(snap(0, backend="metal"), adapter={"luid": 1})
        older = snap(1, backend="metal")
        different = dict(snap(2, backend="metal"), adapter={"luid": 2})

        self.assertEqual(split_runs([first, older, different]), [[first, older], [different]])

    def test_runs_keep_windows_luid_only_samples_across_reboots(self):
        first = dict(snap(0), adapter={"luid": 1})
        older = snap(1)
        rebooted = dict(snap(2), adapter={"luid": 2})

        self.assertEqual(split_runs([first, older, rebooted]), [[first, older, rebooted]])

    def test_load_skips_lines_that_do_not_parse(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(snap(0)) + "\n\n" + json.dumps({"no": "snapshot"}) + "\n"
                        + json.dumps(snap(1)) + "\n" + json.dumps(snap(2))[:40])
            samples, skipped = load(path)
            self.assertEqual(([s["taken"] for s in samples], skipped), ([0, 1], 2))

            out = io.StringIO()
            with patch("sys.argv", ["gputaskman", "summary", path]), contextlib.redirect_stdout(out):
                main()
            self.assertIn("2 sample(s) in 1 run(s)", out.getvalue())
            self.assertIn("2 line(s) that do not parse", out.getvalue())
            out = io.StringIO()
            with patch("sys.argv", ["gputaskman", "summary", path, "--json"]), contextlib.redirect_stdout(out):
                main()
            self.assertEqual(json.loads(out.getvalue())["runs"][0]["samples"], 2)

            empty = os.path.join(d, "empty.jsonl")
            open(empty, "w").close()
            err = io.StringIO()
            with patch("sys.argv", ["gputaskman", "summary", empty]), contextlib.redirect_stderr(err), \
                    self.assertRaises(SystemExit):
                main()
            self.assertIn("no gputaskman record snapshots", err.getvalue())


if __name__ == "__main__":
    unittest.main()
