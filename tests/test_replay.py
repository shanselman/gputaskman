"""Offline replay of a gputaskman record file. No GPU, no sleeps: time is a fake clock."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import DataTable, Static

from gputaskman.cli import main
from gputaskman.replay import (FIXED_INTERVAL, Playback, RecordingError, describe, load_recording,
                               process_rows)

GiB = 1 << 30


def snap(taken=1.0, gpu="NVIDIA RTX 4090", processes=None, **extra) -> dict:
    data = {
        "taken": taken, "gpu": gpu, "driver": "560.00", "cuda": "12.6", "backend": "wddm",
        "adapter": {"luid": 7, "name": gpu, "vendor_id": 0x10DE, "dedicated_vram": 24 * GiB,
                    "shared_system": 0, "software": False, "pci_address": [1, 0, 0],
                    "device_instance_id": "", "identity_status": ""},
        "vram_total": 24 * GiB, "vram_used": 12 * GiB, "shared_used": GiB, "attributed": 10 * GiB,
        "device": {"util_gpu": 78.0, "util_mem": 40.0, "temp": 64.0, "power_w": 285.0,
                   "power_limit_w": 320.0, "pci_bus": "00000000:01:00.0"},
        "processes": [proc()] if processes is None else processes,
    }
    data.update(extra)
    return data


def proc(pid=10, name="python.exe", vram=8 * GiB, **extra) -> dict:
    data = {"pid": pid, "host": None, "name": name, "kind": "LLM", "vram_attributed": True,
            "vram_inferred_bytes": None, "vram_inferred_by": None, "vram_bytes": vram,
            "shared_bytes": 0, "gpu_pct": 12.0}
    data.update(extra)
    return data


def write(path: str, records, tail: str = "") -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        if tail:
            handle.write(tail)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class LoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "session.jsonl")

    def test_a_valid_recording_keeps_the_recorded_values(self):
        write(self.path, [snap(1.0), snap(2.5, processes=[proc(20, "ollama.exe", 2 * GiB)])])
        with load_recording(self.path) as recording:
            self.assertEqual(recording.count, 2)
            self.assertTrue(recording.timed)
            self.assertEqual(recording.taken, [1.0, 2.5])
            self.assertEqual(recording.snapshot(1)["processes"][0]["name"], "ollama.exe")
            self.assertEqual(recording.snapshot(1)["vram_used"], 12 * GiB)
            again = recording.snapshot(1)
            self.assertIsNot(recording.snapshot(1), again)  # parsed when asked, not cached copies
        os.remove(self.path)  # the handle is closed, including on Windows

    def test_ctrl_c_while_the_file_is_indexed_closes_the_handle(self):
        write(self.path, [snap(1.0), snap(2.0)])
        opened = []
        builtin = open

        def tracking(*args, **kwargs):
            handle = builtin(*args, **kwargs)
            opened.append(handle)
            return handle

        with patch("builtins.open", tracking), patch("gputaskman.replay._binary_lines", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                load_recording(self.path)
        self.assertTrue(opened[-1].closed)

    def test_a_real_json_snapshot_round_trips(self):
        from gputaskman.collector import ProcRow, Snapshot
        from gputaskman.dxgi import Adapter
        from gputaskman.pdh import ProcGpu
        from gputaskman.procinfo import ProcDetail
        from gputaskman.report import to_json_dict

        row = ProcRow(10, "llama-server.exe", "LLM", ProcGpu(dedicated=8 * GiB), ProcDetail(10, "llama-server.exe"))
        adapter = Adapter(7, "NVIDIA RTX 4090", 0x10DE, 24 * GiB, 0, False)
        recorded = to_json_dict(Snapshot(taken=1.5, adapter=adapter, adapters=[adapter], nv=None, driver="560",
                                         cuda="12.6", vram_total=24 * GiB, vram_used=8 * GiB, shared_used=0,
                                         procs=[row]))
        write(self.path, [recorded])
        with load_recording(self.path) as recording:
            self.assertEqual(recording.snapshot(0)["processes"][0]["vram_bytes"], 8 * GiB)
            self.assertIsNone(recording.snapshot(0)["device"])

    def test_missing_and_empty_files_are_errors(self):
        with self.assertRaises(RecordingError) as missing:
            load_recording(os.path.join(self.dir.name, "missing.jsonl"))
        self.assertIn("cannot read", str(missing.exception))
        open(self.path, "wb").close()
        with self.assertRaises(RecordingError) as empty:
            load_recording(self.path)
        self.assertIn("empty", str(empty.exception))

    def test_invalid_json_and_a_truncated_tail(self):
        write(self.path, [], tail="{not json")
        with self.assertRaises(RecordingError) as bad:
            load_recording(self.path)
        self.assertIn("no complete", str(bad.exception))
        write(self.path, [snap(1.0), snap(2.0)], tail='{"taken":3')
        with load_recording(self.path) as recording:
            self.assertEqual(recording.count, 2)
            self.assertTrue(any("trailing incomplete" in warning for warning in recording.warnings))
            self.assertEqual(recording.snapshot(1)["taken"], 2.0)

    def test_a_bad_line_between_valid_ones_is_reported_and_not_a_frame(self):
        write(self.path, [snap(1.0)], tail="{bad\n")
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(snap(3.0, gpu="Other GPU", processes=[proc(9, "chrome.exe")])) + "\n")
            # the third object needs its own luid or the gpu count stays 1
        raw = json.dumps(snap(3.0))
        # rewrite cleanly: valid, broken, valid with a different gpu
        other = snap(3.0, gpu="AMD Radeon", processes=[proc(9, "chrome.exe")])
        other["adapter"] = dict(other["adapter"], luid=99, name="AMD Radeon", vendor_id=0x1002)
        write(self.path, [], tail=json.dumps(snap(1.0)) + "\n{bad\n" + json.dumps(other) + "\n")
        with load_recording(self.path) as recording:
            self.assertEqual(recording.count, 2)
            self.assertEqual(recording.gpu_count, 2)
            self.assertTrue(any("skipped line 2" in warning for warning in recording.warnings))
            self.assertEqual(recording.snapshot(1)["processes"][0]["name"], "chrome.exe")
            self.assertNotIn("{bad", json.dumps(recording.snapshot(0)))
        del raw

    def test_utf16_and_a_long_file_do_not_keep_every_object(self):
        text = json.dumps(snap(1.0)) + "\n" + json.dumps(snap(2.0)) + "\n"
        with open(self.path, "wb") as handle:
            handle.write(text.encode("utf-16"))
        with load_recording(self.path) as recording:
            self.assertEqual(recording.count, 2)
            self.assertEqual(recording.taken, [1.0, 2.0])
        records = [snap(float(i), processes=[proc(i, f"p{i}.exe", GiB)]) for i in range(1, 1001)]
        write(self.path, records)
        with load_recording(self.path) as recording:
            self.assertEqual(recording.count, 1000)
            self.assertEqual(recording.snapshot(0)["processes"][0]["name"], "p1.exe")
            self.assertEqual(recording.snapshot(999)["processes"][0]["name"], "p1000.exe")
            self.assertIsNotNone(recording._file)
            self.assertFalse(hasattr(recording, "_text"))


class PlaybackTests(unittest.TestCase):
    def play(self, taken=(0.0, 4.0, 5.0), timed=True) -> Playback:
        return Playback(len(taken), list(taken), timed=timed, clock=self.clock)

    def setUp(self) -> None:
        self.clock = Clock()

    def test_steps_and_ends(self):
        play = self.play()
        self.assertEqual(play.index, 0)
        self.assertFalse(play.playing)
        play.step(1)
        play.step(1)
        self.assertEqual(play.index, 2)
        play.step(1)
        self.assertEqual(play.index, 2)
        play.step(-1)
        self.assertEqual(play.index, 1)
        play.first()
        self.assertEqual(play.index, 0)
        play.last()
        self.assertEqual(play.index, 2)
        self.assertFalse(play.playing)

    def test_play_pause_speed_and_stop_at_the_end(self):
        play = self.play()
        play.toggle()
        self.assertTrue(play.playing)
        self.assertEqual(play.delay(), 4.0)
        self.clock.now = 1.0
        play.faster()  # 2x, 3s of the recorded gap left
        self.assertEqual(play.speed, 2.0)
        self.assertEqual(play.delay(), 1.5)
        play.toggle()
        self.assertFalse(play.playing)
        self.assertIsNone(play.delay())
        play.toggle()
        play.faster()
        play.faster()
        play.faster()  # stays at 4x
        self.assertEqual(play.speed, 4.0)
        play.slower()
        play.slower()
        play.slower()
        play.slower()
        self.assertEqual(play.speed, 0.5)
        play.restart()
        self.assertEqual(play.index, 0)
        self.assertTrue(play.playing)
        self.assertTrue(play.advance())
        self.assertTrue(play.advance())
        self.assertEqual(play.index, 2)
        self.assertFalse(play.playing)
        self.assertFalse(play.advance())
        play.toggle()  # already at the end: do not loop
        self.assertFalse(play.playing)
        self.assertEqual(play.index, 2)

    def test_home_keeps_playing_and_a_fixed_interval_is_not_recorded_time(self):
        play = self.play()
        play.toggle()
        play.step(1)
        self.assertTrue(play.playing)
        play.first()
        self.assertEqual(play.index, 0)
        self.assertTrue(play.playing)
        untimed = Playback(3, [None, 5.0, 4.0], timed=False, clock=self.clock)
        untimed.toggle()
        self.assertEqual(untimed.gap(), FIXED_INTERVAL)
        self.assertEqual(untimed.delay(), FIXED_INTERVAL)
        untimed.faster()
        self.assertEqual(untimed.delay(), FIXED_INTERVAL / 2)
        self.assertIn("not recorded time", describe(
            _recording_view(untimed, timed=False), untimed, snap(taken=None, processes=[])))

    def test_one_snapshot_stays_paused(self):
        play = Playback(1, [1.0], timed=False, clock=self.clock)
        play.toggle()
        play.restart()
        self.assertFalse(play.playing)
        self.assertIsNone(play.delay())


class DisplayTests(unittest.TestCase):
    def test_missing_metrics_stay_unavailable_and_zero_stays_zero(self):
        body = snap(processes=[proc(vram=None, gpu_pct=None, shared_bytes=None)])
        body["processes"][0]["vram_bytes"] = None
        for key in ("vram_used", "vram_total", "shared_used", "attributed", "device"):
            body[key] = None
        text = describe(_view(), Playback(1, [1.0], timed=False), body)
        self.assertIn("Memory: N/A / N/A   attributed n/a   shared n/a", text)
        self.assertIn("Utilisation: N/A", text)
        self.assertIn("Temperature: N/A", text)
        self.assertIn("Power: N/A", text)
        self.assertNotIn("0 MB", text)
        rows, _notes = process_rows(body)
        self.assertEqual(rows[0][3], "n/a")
        self.assertEqual(rows[0][5], "n/a")
        measured_zero = snap(processes=[proc(vram=0, gpu_pct=0)])
        measured_zero["vram_used"] = 0
        zero_text = describe(_view(), Playback(1, [1.0], timed=False), measured_zero)
        self.assertIn("Memory: 0 MB / 24.00 GB", zero_text)
        self.assertEqual(process_rows(measured_zero)[0][0][5], "0")

    def test_shared_is_blank_only_when_the_row_has_no_shared_counter(self):
        self.assertEqual(process_rows(snap(processes=[proc(shared_bytes=None)]))[0][0][4], "n/a")
        self.assertEqual(process_rows(snap(processes=[proc(shared_bytes=0)]))[0][0][4], "0 MB")
        linux = snap(backend="nvml", processes=[proc(shared_bytes=None)])
        self.assertEqual(process_rows(linux)[0][0][4], "")
        windows = snap(backend="wsl", processes=[proc(shared_bytes=None, host="windows")])
        self.assertEqual(process_rows(windows)[0][0][4], "n/a")

    def test_inferred_bytes_are_marked_and_not_added_to_the_device_total(self):
        inferred = proc(name="wsl-python", vram=None, vram_attributed=False, vram_inferred_bytes=3 * GiB,
                        vram_inferred_by="order")
        body = snap(processes=[inferred])
        body["attributed"] = None
        text = describe(_view(), Playback(1, [1.0], timed=False), body)
        self.assertIn("attributed n/a", text)
        self.assertIn("12.00 GB / 24.00 GB", text)
        self.assertNotIn("15.00 GB", text)
        self.assertEqual(process_rows(body)[0][0][3], "3.00 GB?")
        self.assertIn("inferred", process_rows(body)[0][0][6])
        unmarked = proc(name="plain", vram=None, vram_attributed=False, vram_inferred_bytes=GiB, vram_inferred_by=None)
        cells = process_rows(snap(processes=[unmarked]))[0][0]
        self.assertEqual(cells[3], "1.00 GB")
        self.assertNotIn("?", cells[3])
        self.assertIn("inferred", cells[6])
        agreed = proc(name="agreed", vram=None, vram_attributed=False, vram_inferred_bytes=GiB,
                      vram_inferred_by="consistent")
        self.assertTrue(process_rows(snap(processes=[agreed]))[0][0][3].endswith("*"))

    def test_processes_and_gpus_follow_the_snapshot(self):
        first = snap(1.0, processes=[proc(1, "python.exe", 9 * GiB), proc(2, "ollama.exe", 2 * GiB)])
        second = snap(2.0, gpu="AMD Radeon", processes=[proc(3, "chrome.exe", GiB // 2)])
        second["adapter"] = dict(second["adapter"], luid=99, name="AMD Radeon", vendor_id=0x1002)
        second["vram_used"] = None
        second["device"] = {"temp": None, "power_w": None, "util_gpu": None}
        recording = _view(count=2, taken=[1.0, 2.0], gpus=2)
        play = Playback(2, [1.0, 2.0], timed=True)
        self.assertEqual([row[1] for row in process_rows(first)[0]], ["python.exe", "ollama.exe"])
        self.assertIn("NVIDIA RTX 4090", describe(recording, play, first))
        self.assertIn("GPUs in this file: 2", describe(recording, play, first))
        play.step(1)
        later = describe(recording, play, second)
        self.assertEqual([row[1] for row in process_rows(second)[0]], ["chrome.exe"])
        self.assertIn("AMD Radeon", later)
        self.assertIn("Utilisation: N/A", later)
        self.assertIn("PAUSED (end)", later)

    def test_a_recorded_line_shows_the_sample_not_a_sum_of_rows(self):
        body = snap(processes=[proc(vram=9 * GiB), proc(vram=GiB, name="ollama.exe")])
        text = describe(_view(), Playback(1, [1.0], timed=False), body)
        self.assertIn("Snapshot: 1 / 1", text)
        self.assertIn("Status: PAUSED", text)
        self.assertIn("Speed: 1.0x", text)
        self.assertIn("Timing: one snapshot", text)
        self.assertIn("] 100.0%", text)
        self.assertIn("Memory: 12.00 GB / 24.00 GB   attributed 10.00 GB", text)
        self.assertIn("Utilisation: 78.0%", text)
        self.assertIn("Temperature: 64.0°C", text)
        self.assertIn("Power: 285/320 W", text)
        self.assertIn("unix (", text)
        self.assertIn("on this computer)", text)
        self.assertNotIn("NVIDIA  NVIDIA", text)
        names = [row[1] for row in process_rows(body)[0]]
        self.assertEqual(names, ["python.exe", "ollama.exe"])


class UiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "session.jsonl")
        write(self.path, [snap(0.0, processes=[proc(1, "python.exe")]),
                          snap(1000.0, processes=[proc(2, "ollama.exe")]),
                          snap(2000.0, processes=[proc(3, "chrome.exe")])])

    def text(self, app) -> str:
        content = app.query_one("#replay", Static).content
        return content.plain if hasattr(content, "plain") else str(content)

    async def test_keys_move_and_play_stops_on_the_last_snapshot(self):
        from gputaskman.replay import ReplayApp

        app = ReplayApp(load_recording(self.path))
        self.addCleanup(app.recording.close)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            self.assertIn("Snapshot: 1 / 3", self.text(app))
            self.assertIn("PAUSED", self.text(app))
            self.assertIn("1.0x", self.text(app))
            self.assertIn("recorded timestamps", self.text(app))
            await pilot.press("right")
            await pilot.pause()
            self.assertIn("Snapshot: 2 / 3", self.text(app))
            self.assertEqual(app.query_one("#procs", DataTable).get_cell("0", "Process").plain, "ollama.exe")
            await pilot.press("left")
            await pilot.pause()
            self.assertIn("Snapshot: 1 / 3", self.text(app))
            await pilot.press("end")
            await pilot.pause()
            self.assertIn("Snapshot: 3 / 3", self.text(app))
            self.assertIn("PAUSED (end)", self.text(app))
            await pilot.press("home")
            await pilot.press("plus")
            await pilot.pause()
            self.assertEqual(app.playback.speed, 2.0)
            self.assertIn("Snapshot: 1 / 3", self.text(app))
            await pilot.press("minus")
            await pilot.pause()
            self.assertEqual(app.playback.speed, 1.0)
            await pilot.press("space")
            await pilot.pause()
            self.assertTrue(app.playback.playing)
            self.assertAlmostEqual(app.scheduled_delay, 1000.0)
            app._tick()
            await pilot.pause()
            self.assertEqual(app.playback.index, 1)
            app._tick()
            await pilot.pause()
            self.assertEqual(app.playback.index, 2)
            self.assertFalse(app.playback.playing)
            app._tick()
            self.assertEqual(app.playback.index, 2)
            await pilot.press("r")
            await pilot.pause()
            self.assertEqual(app.playback.index, 0)
            self.assertTrue(app.playback.playing)
            await pilot.resize_terminal(70, 20)
            await pilot.pause()
            self.assertIn("Snapshot: 1 / 3", self.text(app))

    async def test_replay_never_samples_the_gpu(self):
        from gputaskman.replay import ReplayApp

        banned = patch.multiple(
            "gputaskman.collector",
            Collector=unittest.mock.Mock(side_effect=AssertionError("collector")),
            enumerate_adapters=unittest.mock.Mock(side_effect=AssertionError("discover")),
        )
        others = [
            patch("gputaskman.dxgi.enumerate_adapters", side_effect=AssertionError("dxgi")),
            patch("gputaskman.nvml.Nvml", side_effect=AssertionError("nvml")),
            patch("gputaskman.record.source", side_effect=AssertionError("record")),
            patch("gputaskman.host.HostSampler", side_effect=AssertionError("host")),
        ]
        with banned, others[0], others[1], others[2], others[3]:
            recording = load_recording(self.path)
            app = ReplayApp(recording)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("space", "right")
                app._tick()
                self.assertIn("Snapshot:", self.text(app))
            recording.close()


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "session.jsonl")
        write(self.path, [snap(1.0), snap(2.0)], tail='{"cut')

    def test_help_lists_replay(self):
        out = io.StringIO()
        with patch("sys.argv", ["gputaskman", "--help"]), contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            main()
        self.assertIn("gputaskman replay FILE", out.getvalue())
        out = io.StringIO()
        with patch("sys.argv", ["gputaskman", "replay", "--help"]), contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit):
            main()
        self.assertIn("Offline:", out.getvalue())

    def test_a_missing_file_is_refused_without_sampling(self):
        err = io.StringIO()
        with patch("sys.argv", ["gputaskman", "replay", self.path + ".missing"]), \
                patch("gputaskman.collector.enumerate_adapters", side_effect=AssertionError("discover")), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
            main()
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("cannot read", err.getvalue())

    def test_warnings_print_after_the_screen_and_ctrl_c_closes_the_file(self):
        err = io.StringIO()
        with patch("sys.argv", ["gputaskman", "replay", self.path]), \
                patch("gputaskman.replay.run_replay", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stderr(err):
            main()
        self.assertIn("trailing incomplete", err.getvalue())
        os.remove(self.path)


def _view(count=1, taken=None, timed=False, gpus=1):
    """A recording-shaped object for describe(); it does not open a file."""
    class View:
        path = "session.jsonl"
        warnings: list[str] = []
        gpu_count = gpus

    view = View()
    view.count = count
    view.timed = timed
    view.taken = list(taken or [1.0])
    return view


def _recording_view(playback: Playback, timed: bool):
    view = _view(count=playback.count, taken=playback.taken, timed=timed)
    if not timed and playback.count >= 2:
        view.warnings = ["Playback uses 1 second per snapshot at 1x because recorded timestamps are missing."]
    return view


if __name__ == "__main__":
    unittest.main()
