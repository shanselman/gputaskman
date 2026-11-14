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


def proc(name, vram=None, pct=None, host=None):
    return {"pid": 1, "host": host, "name": name, "vram_bytes": vram, "gpu_pct": pct}


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

    def test_runs_split_at_long_gaps_clock_steps_and_gpu_changes(self):
        taken = [0, 1, 2, 3, 500, 501, 502, 400, 401]
        samples = [snap(t) for t in taken] + [snap(402, gpu="Other")]
        runs = split_runs(samples)
        self.assertEqual([[s["taken"] for s in r] for r in runs], [[0, 1, 2, 3], [500, 501, 502], [400, 401], [402]])
        # a slow interval is not a gap: 5 s samples with one 30 s hiccup stay one run
        self.assertEqual(len(split_runs([snap(t) for t in (0, 5, 10, 40, 45)])), 1)

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
