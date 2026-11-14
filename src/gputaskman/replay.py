"""Offline playback of a file written by ``gputaskman record``.

A recording is the ``--json`` objects ``record`` already writes, one snapshot per line. ``taken`` is
the sample time in Unix seconds. This module only reads that file: it never samples the GPU, probes
an API, or writes the recording back.

Long files are not copied into a list of objects. The reader remembers where each snapshot starts
and parses the one on screen, including a UTF-16 file. Playback timing uses ``taken`` when every
value is finite and the sequence is strictly increasing. The file does not store ``--interval``, so
anything else waits one second per snapshot at 1x and says that wait is not recorded time.
"""

from __future__ import annotations

import json
import math
import os
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Static

from .dxgi import VENDORS
from .llm import gb
from .nvml import format_metric
from .report import DRM_NOTES, INFERRED_NOTE, ORDER_NOTE

SPEEDS = (0.5, 1.0, 2.0, 4.0)
FIXED_INTERVAL = 1.0  # stand-in when `taken` cannot drive playback; not a recovered sample period
MAX_LINE = 32 << 20
MAX_WARNINGS = 20
UNTIMED = ("Playback uses 1 second per snapshot at 1x because recorded timestamps are missing or not "
           "strictly increasing. That interval is not recorded timing.")
_TRAILING = "skipped a trailing incomplete line ({}); an interrupted recording can end that way"
_INCOMPLETE = ("invalid JSON", "invalid text")


class RecordingError(ValueError):
    """The file cannot be replayed. ``str(error)`` is the message for the command line."""


class Recording:
    """One opened JSON Lines recording. Snapshots are parsed when asked for, then discarded."""

    def __init__(self, path: str, spans: list[tuple[int, int]], taken: list[float | None], timed: bool,
                 warnings: list[str], gpu_count: int, file, encoding: str) -> None:
        self.path = path
        self.spans = spans
        self.taken = taken
        self.timed = timed
        self.warnings = warnings
        self.gpu_count = gpu_count
        self._file = file
        self._encoding = encoding

    def __len__(self) -> int:
        return len(self.spans)

    @property
    def count(self) -> int:
        return len(self.spans)

    def snapshot(self, index: int) -> dict:
        """The snapshot at index, parsed from the file (or from decoded text, for UTF-16)."""
        if index < 0 or index >= len(self.spans):
            raise IndexError(index)
        if self._file is None:
            raise ValueError("recording is closed")
        start, length = self.spans[index]
        self._file.seek(start)
        try:
            raw = self._file.read(length).decode(self._encoding)
        except UnicodeDecodeError as error:
            raise ValueError(f"snapshot {index + 1} is not valid text") from error
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"snapshot {index + 1} is not an object")
        return data

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Recording:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Playback:
    """Which snapshot is on screen, and when play moves to the next one.

    The clock is injectable so tests can advance time without sleeping. ``delay`` is the only method
    that starts a wait; ``preview_delay`` reports the same wait without starting it.
    """

    def __init__(self, count: int, taken: list[float | None], *, timed: bool, clock=time.monotonic) -> None:
        if count < 1:
            raise ValueError("nothing to play")
        if len(taken) != count:
            raise ValueError("taken and count differ")
        self.count = count
        self.taken = list(taken)
        self.timed = timed
        self._clock = clock
        self.index = 0
        self.playing = False
        self._speed_index = SPEEDS.index(1.0)
        self._remaining: float | None = None  # unscaled seconds left on this snapshot
        self._mark: float | None = None
        self._mark_speed = 1.0
        if timed:
            previous = None
            for value in self.taken:
                if value is None or (previous is not None and value <= previous):
                    raise ValueError("recorded timing needs strictly increasing timestamps")
                previous = value

    @property
    def speed(self) -> float:
        return SPEEDS[self._speed_index]

    def toggle(self) -> None:
        """Play, or pause. At the last snapshot play does nothing: playback does not loop."""
        if self.playing:
            self._consume()
            self.playing = False
            return
        if self.index >= self.count - 1:
            return
        self.playing = True

    def advance(self) -> bool:
        """Move to the next snapshot while playing. Pause on the last one and return whether it moved."""
        if not self.playing or self.index >= self.count - 1:
            self.playing = False
            self._remaining = None
            self._mark = None
            return False
        self.index += 1
        self._remaining = None
        self._mark = None
        if self.index >= self.count - 1:
            self.playing = False
        return True

    def step(self, delta: int) -> None:
        self._goto(self.index + delta)

    def first(self) -> None:
        """Jump to the first snapshot. A playback that was running keeps running."""
        if self.index != 0:
            self.index = 0
            self._remaining = None
            self._mark = None

    def last(self) -> None:
        self._goto(self.count - 1)

    def faster(self) -> None:
        if self._speed_index < len(SPEEDS) - 1:
            self._speed_index += 1

    def slower(self) -> None:
        if self._speed_index > 0:
            self._speed_index -= 1

    def restart(self) -> None:
        """Back to the first snapshot and play. A one-snapshot recording stays paused."""
        self.index = 0
        self._remaining = None
        self._mark = None
        self.playing = self.count > 1

    def gap(self) -> float | None:
        """Unscaled seconds between this snapshot and the next, or None at the end."""
        if self.index >= self.count - 1:
            return None
        if self.timed:
            return self.taken[self.index + 1] - self.taken[self.index]
        return FIXED_INTERVAL

    def preview_delay(self) -> float | None:
        """Seconds until the next snapshot at the current speed, without starting a wait."""
        remaining = self._remaining_now()
        if remaining is None:
            return None
        return remaining / self.speed

    def delay(self) -> float | None:
        """Seconds until the next snapshot, and start (or continue) the wait for that long."""
        if not self.playing or self.index >= self.count - 1:
            self._consume()
            return None
        self._consume()
        if self._remaining is None:
            self._remaining = self.gap()
        self._mark = self._clock()
        self._mark_speed = self.speed
        return self._remaining / self.speed

    def _remaining_now(self) -> float | None:
        if not self.playing or self.index >= self.count - 1:
            return None
        remaining = self._remaining if self._remaining is not None else self.gap()
        if self._mark is not None and remaining is not None:
            elapsed = self._clock() - self._mark
            if elapsed > 0:
                remaining = max(0.0, remaining - elapsed * self._mark_speed)
        return remaining

    def _consume(self) -> None:
        if self._remaining is None or self._mark is None:
            return
        elapsed = self._clock() - self._mark
        if elapsed > 0:
            self._remaining = max(0.0, self._remaining - elapsed * self._mark_speed)
        self._mark = None

    def _goto(self, index: int) -> None:
        index = min(max(index, 0), self.count - 1)
        if index != self.index:
            self.index = index
            self._remaining = None
            self._mark = None
        if self.index >= self.count - 1:
            self.playing = False


def load_recording(path: str) -> Recording:
    """Open a JSON Lines recording. Raise RecordingError when there is nothing to play.

    A final line that does not parse is the interrupted-recording case and is skipped with a note.
    Any other bad line is skipped and reported too; it is not turned into a snapshot.
    """
    try:
        handle = open(path, "rb")
    except OSError as error:
        raise RecordingError(f"cannot read {path}: {error}") from error
    try:
        mark = handle.read(4)
        handle.seek(0)
        if mark[:2] == b"\xff\xfe":
            return _from_lines(path, _utf16_lines(handle, little=True), file=handle, encoding="utf-16-le")
        if mark[:2] == b"\xfe\xff":
            return _from_lines(path, _utf16_lines(handle, little=False), file=handle, encoding="utf-16-be")
        return _from_lines(path, _binary_lines(handle), file=handle, encoding="utf-8")
    except BaseException:
        # KeyboardInterrupt is not an Exception. Close the handle before it propagates.
        if not handle.closed:
            handle.close()
        raise


def run_replay(recording: Recording) -> None:
    """Run the replay screen. The caller closes the recording, including after Ctrl+C."""
    ReplayApp(recording).run()


def describe(recording: Recording, playback: Playback, snapshot: dict, *, width: int = 40) -> str:
    """The status block for one snapshot. File strings are included as plain text."""
    lines = [
        "GPU Session Replay",
        f"File: {_plain(recording.path)}",
        "",
        f"Snapshot: {playback.index + 1:,} / {playback.count:,}",
        f"Status: {_status(playback)}",
        f"Speed: {playback.speed:.1f}x",
        f"Timing: {_timing(recording, playback)}",
        f"Recorded: {_taken_text(snapshot.get('taken'))}",
        _progress(playback.index, playback.count, width),
    ]
    upcoming = _upcoming(recording, playback)
    if upcoming:
        lines.append(upcoming)
    if recording.gpu_count > 1:
        lines.append(f"GPUs in this file: {recording.gpu_count}. Each snapshot shows the GPU it recorded.")
    if recording.warnings:
        shown = recording.warnings[:3]
        lines.append("Notice: " + " ".join(shown))
        if len(recording.warnings) > 3:
            lines.append(f"{len(recording.warnings) - 3} more notice(s) print after you quit.")
    lines.append("")
    lines.append("GPU: " + _gpu_heading(snapshot))
    backend = snapshot.get("backend")
    if isinstance(backend, str) and backend:
        lines.append(f"Backend: {_plain(backend)}")
    lines.append(_memory_line(snapshot))
    lines.append(_utilisation_line(snapshot))
    lines.append(_temperature_line(snapshot))
    lines.append(_power_line(snapshot))
    system = _system_line(snapshot)
    if system:
        lines.append(system)
    for note in _notes(snapshot):
        lines.append(_plain(note))
    lines.append("")
    lines.append("Space: Play/Pause | ←/→: Step | Home/End: Ends | +/-: Speed | r: Restart | q: Quit")
    return "\n".join(lines)


def process_rows(snapshot: dict) -> tuple[list[tuple[str, str, str, str, str, str, str]], list[str]]:
    """Table cells for the recorded processes, in the order the snapshot stored them.

    Returns the rows and any notes about rows that were not objects. VRAM is the recorded measured
    bytes, or the inferred bytes with ``*`` / ``?``. A missing counter stays ``n/a``."""
    procs = snapshot.get("processes")
    if not isinstance(procs, list):
        return [], ["processes is missing; none are shown"]
    rows = []
    bad = 0
    for proc in procs:
        if not isinstance(proc, dict):
            bad += 1
            continue
        rows.append((
            _pid(proc),
            _plain(proc.get("name")),
            _plain(proc.get("kind")),
            _vram(proc),
            _shared(snapshot, proc),
            _pct(proc.get("gpu_pct")),
            _note(proc),
        ))
    notes = [f"{bad} process row(s) were not objects and are not shown"] if bad else []
    return rows, notes


class ReplayApp(App):
    """The replay screen. It reads the recording it was given and does not collect anything live."""

    TITLE = "gputaskman replay"
    CSS = """
    Screen { layout: vertical; }
    #replay { height: auto; max-height: 70%; overflow-y: auto; border: round $accent; padding: 0 1; }
    #procs { height: 1fr; min-height: 4; }
    """
    BINDINGS = [
        Binding("space", "play_pause", "Play/Pause", priority=True),
        Binding("right", "next", "Next", priority=True),
        Binding("left", "previous", "Previous", priority=True),
        Binding("home", "first", "First", priority=True),
        Binding("end", "last", "Last", priority=True),
        Binding("plus,+", "faster", "Faster", priority=True),
        Binding("minus,-", "slower", "Slower", priority=True),
        Binding("r", "restart", "Restart", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(self, recording: Recording) -> None:
        super().__init__()
        self.recording = recording
        self.playback = Playback(recording.count, recording.taken, timed=recording.timed)
        self._timer = None
        self._mounted = False
        self._drawing = False
        self.scheduled_delay: float | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="replay", markup=False)
        table = DataTable(id="procs", cursor_type="row", zebra_stripes=True)
        yield table
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#procs", DataTable)
        for name in ("PID", "Process", "Kind", "VRAM", "Shared", "GPU%", "Note"):
            table.add_column(name, key=name)
        panel = self.query_one("#replay", Static)
        panel.border_title = "GPU Session Replay"
        panel.border_subtitle = "offline"
        table.focus()
        self._mounted = True
        self.refresh_view()

    def on_resize(self) -> None:
        if self._mounted and self.is_running and not self._drawing:
            self.call_after_refresh(self.refresh_view)

    def on_unmount(self) -> None:
        self._cancel_timer()

    def refresh_view(self) -> None:
        if not self.is_running or self._drawing:
            return
        self._drawing = True
        try:
            self._refresh_view()
        finally:
            self._drawing = False

    def _refresh_view(self) -> None:
        try:
            snapshot = self.recording.snapshot(self.playback.index)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            self.playback.playing = False
            self.query_one("#replay", Static).update(f"Cannot read snapshot {self.playback.index + 1}: {error}")
            self._fill_table([])
            self._arm()
            return
        width = 40
        if self.size.width:
            width = max(10, min(40, self.size.width - 4))
        text = describe(self.recording, self.playback, snapshot, width=width)
        extra = process_rows(snapshot)
        self.query_one("#replay", Static).update(text)
        self._fill_table(extra[0])
        self._arm()

    def action_play_pause(self) -> None:
        self.playback.toggle()
        self.refresh_view()

    def action_next(self) -> None:
        self.playback.step(1)
        self.refresh_view()

    def action_previous(self) -> None:
        self.playback.step(-1)
        self.refresh_view()

    def action_first(self) -> None:
        self.playback.first()
        self.refresh_view()

    def action_last(self) -> None:
        self.playback.last()
        self.refresh_view()

    def action_faster(self) -> None:
        self.playback.faster()
        self.refresh_view()

    def action_slower(self) -> None:
        self.playback.slower()
        self.refresh_view()

    def action_restart(self) -> None:
        self.playback.restart()
        self.refresh_view()

    def _tick(self) -> None:
        self._timer = None
        if not self.is_running:
            return
        self.playback.advance()
        self.refresh_view()

    def _arm(self) -> None:
        self._cancel_timer()
        delay = self.playback.delay()
        self.scheduled_delay = delay
        if delay is not None:
            self._timer = self.set_timer(delay, self._tick)

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.stop()

    def _fill_table(self, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
        table = self.query_one("#procs", DataTable)
        table.clear()
        for index, row in enumerate(rows):
            table.add_row(*(self._cell(value) for value in row), key=str(index))

    @staticmethod
    def _cell(value: str) -> Text:
        style = "grey62" if value == "n/a" else ""
        justify = "right" if value == "n/a" or value[:1].isdigit() or value[:1] == "-" else "left"
        return Text(value, style=style, justify=justify)


def _from_lines(path: str, lines, *, file, encoding: str) -> Recording:
    spans: list[tuple[int, int]] = []
    taken: list[float | None] = []
    problems: list[tuple[int, str]] = []
    gpus: set[str] = set()
    line_no = 0
    last_nonblank = 0
    any_nonblank = False
    for line_no, start, payload in lines:
        if payload is None:
            any_nonblank = True
            last_nonblank = line_no
            problems.append((line_no, "line is too long to be a snapshot"))
            continue
        if isinstance(payload, bytes):
            if not payload.strip(b" \t\r\n\x00"):
                continue
            any_nonblank = True
            last_nonblank = line_no
            try:
                decoded = payload.decode(encoding)
            except UnicodeDecodeError:
                problems.append((line_no, "invalid text"))
                continue
        else:
            if not str(payload).strip():
                continue
            any_nonblank = True
            last_nonblank = line_no
            decoded = str(payload)
        data, reason = _parse_snapshot(decoded)
        if reason:
            problems.append((line_no, reason))
            continue
        length = len(payload)
        spans.append((start, length))
        taken.append(_number(data.get("taken")))
        gpus.add(_gpu_key(data))
    warnings = _warnings(problems, last_nonblank)
    if not spans:
        if not any_nonblank:
            raise RecordingError(f"{path} is empty")
        detail = "; ".join(warnings) if warnings else "no snapshot lines"
        raise RecordingError(f"{path} contains no complete gputaskman snapshot ({detail})")
    timed = _timed(taken)
    if len(spans) >= 2 and not timed:
        warnings.append(UNTIMED)
    return Recording(path, spans, taken, timed, warnings, len(gpus), file, encoding)


def _binary_lines(handle):
    mark = handle.read(3)
    if mark != b"\xef\xbb\xbf":
        handle.seek(0)
    line_no = 0
    while True:
        start = handle.tell()
        chunk = handle.readline(MAX_LINE + 1)
        if not chunk:
            break
        line_no += 1
        if len(chunk) <= MAX_LINE:
            yield line_no, start, chunk
            continue
        while chunk and not chunk.endswith(b"\n"):
            chunk = handle.readline(MAX_LINE)
        yield line_no, start, None  # too long: the rest of the line was discarded


def _utf16_lines(handle, *, little: bool):
    """UTF-16 lines as byte spans. The file stays on disk; only one line is decoded at a time."""
    newline = b"\n\x00" if little else b"\x00\n"
    handle.seek(2)  # the BOM was the reason this reader was chosen
    line_no = 0
    while True:
        start = handle.tell()
        payload, too_long = _take_utf16_line(handle, newline)
        if payload is None and not too_long:
            break
        line_no += 1
        yield line_no, start, None if too_long else payload


def _take_utf16_line(handle, newline: bytes):
    """One UTF-16 line, including its newline, or (None, True) when the line is over the cap.

    (None, False) is end of file. A newline split across two reads is still one line.
    """
    chunks = []
    size = 0
    pending = b""
    keep = len(newline) - 1
    while size <= MAX_LINE:
        block = handle.read(65536)
        if not block:
            if size == 0 and not pending:
                return None, False
            return b"".join(chunks) + pending, False
        data = pending + block
        found = data.find(newline)
        if found != -1:
            end = found + len(newline)
            extra = data[end:]
            if extra:
                handle.seek(-len(extra), os.SEEK_CUR)
            line = b"".join(chunks) + data[:end]
            if len(line) > MAX_LINE:
                return None, True
            return line, False
        if len(data) > keep:
            chunks.append(data[:-keep])
            size += len(data) - keep
            pending = data[-keep:]
        else:
            pending = data
    data = pending
    while True:
        found = data.find(newline)
        if found != -1:
            extra = data[found + len(newline):]
            if extra:
                handle.seek(-len(extra), os.SEEK_CUR)
            return None, True
        block = handle.read(65536)
        if not block:
            return None, True
        data = data[-keep:] + block if data else block


def _parse_snapshot(text: str) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid JSON"
    if not isinstance(data, dict) or not isinstance(data.get("processes"), list):
        return None, "not a gputaskman --json snapshot"
    return data, None


def _warnings(problems: list[tuple[int, str]], last_nonblank: int) -> list[str]:
    problems = list(problems)
    warnings = []
    if problems and problems[-1][0] == last_nonblank and problems[-1][1] in _INCOMPLETE:
        number, _reason = problems.pop()
        warnings.append(_TRAILING.format(number))
    warnings.extend(f"skipped line {number}: {reason}" for number, reason in problems)
    if len(warnings) > MAX_WARNINGS:
        extra = len(warnings) - MAX_WARNINGS
        warnings = warnings[:MAX_WARNINGS]
        warnings.append(f"{extra} more malformed line(s) skipped")
    return warnings


def _timed(taken: list[float | None]) -> bool:
    if len(taken) < 2 or taken[0] is None:
        return False
    previous = taken[0]
    for value in taken[1:]:
        if value is None or value <= previous:
            return False
        previous = value
    return True


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _gpu_key(data: dict) -> str:
    adapter = data.get("adapter") if isinstance(data.get("adapter"), dict) else {}
    device = data.get("device") if isinstance(data.get("device"), dict) else {}
    luid = adapter.get("luid")
    if isinstance(luid, int) and not isinstance(luid, bool):
        return f"luid:{luid}"
    pci = device.get("pci_bus")
    if isinstance(pci, str) and pci:
        return f"pci:{pci}"
    name = data.get("gpu") if isinstance(data.get("gpu"), str) else ""
    if not name and isinstance(adapter.get("name"), str):
        name = adapter["name"]
    return f"name:{name}"


def _plain(value: object) -> str:
    """File text for the screen: no control characters, so a recording cannot redraw the terminal."""
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value if ch >= " " and ch != "\x7f")


def _status(playback: Playback) -> str:
    if playback.playing:
        return "PLAYING"
    if playback.count > 1 and playback.index == playback.count - 1:
        return "PAUSED (end)"
    return "PAUSED"


def _timing(recording: Recording, playback: Playback) -> str:
    if recording.count < 2:
        return "one snapshot"
    if recording.timed:
        return "recorded timestamps"
    return f"{FIXED_INTERVAL:.0f}s per snapshot at 1x (not recorded time); speed is {playback.speed:.1f}x"


def _taken_text(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    try:
        local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(number))
    except (OverflowError, OSError, ValueError):
        return f"{number:.3f} (local time unavailable)"
    return f"{number:.3f} unix ({local} on this computer)"


def _progress(index: int, count: int, width: int) -> str:
    width = max(10, width)
    frac = (index + 1) / count if count else 0.0
    filled = min(width, max(0, round(frac * width)))
    return f"[{'=' * filled}{'-' * (width - filled)}] {frac * 100:.1f}%"


def _upcoming(recording: Recording, playback: Playback) -> str:
    delay = playback.preview_delay()
    gap = playback.gap()
    if delay is None or gap is None:
        return ""
    if recording.timed:
        return f"Next snapshot in {delay:.2f}s ({gap:.2f}s recorded / {playback.speed:.1f}x)"
    return f"Next snapshot in {delay:.2f}s (fixed interval, not recorded time)"


def _gpu_heading(snapshot: dict) -> str:
    adapter = snapshot.get("adapter") if isinstance(snapshot.get("adapter"), dict) else {}
    device = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    name = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), str) and snapshot.get("gpu") else ""
    if not name and isinstance(adapter.get("name"), str):
        name = adapter["name"]
    if not name and isinstance(device.get("name"), str):
        name = device["name"]
    parts = [_plain(name) or "GPU"]
    vendor_id = adapter.get("vendor_id")
    vendor = VENDORS.get(vendor_id, hex(vendor_id)) if isinstance(vendor_id, int) and not isinstance(vendor_id, bool) else ""
    if vendor and vendor.lower() not in parts[0].lower():
        parts.append(vendor)
    luid = adapter.get("luid")
    if isinstance(luid, int) and not isinstance(luid, bool):
        parts.append(f"LUID 0x{luid & 0xffffffffffffffff:016x}")
    pci = device.get("pci_bus") if isinstance(device.get("pci_bus"), str) else ""
    if not pci and isinstance(adapter.get("pci_address"), list) and len(adapter["pci_address"]) == 3:
        bus, dev, function = adapter["pci_address"]
        if all(isinstance(part, int) and not isinstance(part, bool) for part in (bus, dev, function)):
            pci = f"{bus:02x}:{dev:02x}.{function:x}"
    if pci:
        parts.append("PCI " + _plain(pci))
    if isinstance(snapshot.get("driver"), str) and snapshot["driver"]:
        parts.append("driver " + _plain(snapshot["driver"]))
    if isinstance(snapshot.get("cuda"), str) and snapshot["cuda"]:
        parts.append("CUDA " + _plain(snapshot["cuda"]))
    return "  ".join(parts)


def _memory_line(snapshot: dict) -> str:
    """Device memory as the snapshot recorded it. Process rows are not added up here."""
    return (f"Memory: {_total_bytes(snapshot.get('vram_used'))} / {_total_bytes(snapshot.get('vram_total'), total=True)}"
            f"   attributed {_field_bytes(snapshot.get('attributed'))}"
            f"   shared {_field_bytes(snapshot.get('shared_used'))}")


def _total_bytes(value, *, total: bool = False) -> str:
    number = _number(value)
    if number is None or (total and number == 0):
        return "N/A"  # a total of 0 is unknown (an Intel iGPU), matching the live view
    return gb(number)


def _field_bytes(value) -> str:
    number = _number(value)
    return "n/a" if number is None else gb(number)


def _utilisation_line(snapshot: dict) -> str:
    device = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    text = "Utilisation: " + _metric(device.get("util_gpu"), unit="%")
    if _number(device.get("util_mem")) is not None:
        text += "   memory activity " + _metric(device.get("util_mem"), unit="%")
    return text


def _temperature_line(snapshot: dict) -> str:
    device = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    text = "Temperature: " + _metric(device.get("temp"), unit="°C")
    kind = device.get("temp_kind")
    if _number(device.get("temp")) is not None and isinstance(kind, str) and kind:
        text += f" ({_plain(kind)})"
    return text


def _power_line(snapshot: dict) -> str:
    device = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    power = _number(device.get("power_w"))
    limit = _number(device.get("power_limit_w"))
    if power is None:
        text = "Power: N/A"
    elif limit is None:
        text = "Power: " + _metric(power, ".0f") + " W"
    else:
        text = f"Power: {_metric(power, '.0f')}/{_metric(limit, '.0f')} W"
    kind = device.get("power_kind")
    if power is not None and isinstance(kind, str) and kind:
        text += f" ({_plain(kind)})"
    return text


def _metric(value, spec: str = "", unit: str = "") -> str:
    number = _number(value)
    return format_metric(number, spec, unit) if number is not None else "N/A"


def _system_line(snapshot: dict) -> str:
    system = snapshot.get("system")
    if system is None:
        return ""
    if not isinstance(system, dict):
        return "System: n/a"
    if isinstance(system.get("error"), str) and system["error"]:
        return "System RAM/CPU unavailable: " + _plain(system["error"])
    cpu = system.get("cpu") if isinstance(system.get("cpu"), dict) else {}
    ram = system.get("ram") if isinstance(system.get("ram"), dict) else {}
    cpu_pct = _number(cpu.get("pct"))
    used, total = _number(ram.get("used")), _number(ram.get("total"))
    cpu_text = f"{cpu_pct:.0f}%" if cpu_pct is not None else "n/a"
    ram_text = f"{gb(used)} / {gb(total)}" if used is not None and total is not None else "n/a"
    return f"System (OS-reported, not GPU memory): CPU {cpu_text}   RAM {ram_text}"


def _notes(snapshot: dict) -> list[str]:
    procs = snapshot.get("processes") if isinstance(snapshot.get("processes"), list) else []
    notes = []
    if any(isinstance(proc, dict) and proc.get("vram_inferred_by") == "consistent" for proc in procs):
        notes.append(INFERRED_NOTE)
    if any(isinstance(proc, dict) and proc.get("vram_inferred_by") == "order" for proc in procs):
        notes.append(ORDER_NOTE)
    seen = set()
    for proc in procs:
        if not isinstance(proc, dict):
            continue
        reason = proc.get("vram_unattributed_reason")
        text = DRM_NOTES.get(reason, "") if isinstance(reason, str) else ""
        if text and text not in seen:
            seen.add(text)
            notes.append(text)
    for key in ("counter_status", "memory_note", "host_status", "process_status", "wsl_split"):
        value = snapshot.get(key)
        if isinstance(value, str) and value:
            notes.append(value)
    adapter = snapshot.get("adapter") if isinstance(snapshot.get("adapter"), dict) else {}
    if isinstance(adapter.get("identity_status"), str) and adapter["identity_status"]:
        notes.append(adapter["identity_status"])
    rows, extra = process_rows(snapshot)
    del rows
    notes.extend(extra)
    return notes


def _pid(proc: dict) -> str:
    host = proc.get("host")
    pid = proc.get("pid")
    label = "?" if pid is None else str(pid)
    if host == "windows":
        return f"W:{label}"
    if host == "outside":
        return f"H:{label}"
    return label


def _vram(proc: dict) -> str:
    """Measured bytes, or an inferred size. ``*`` and ``?`` are only the marks the snapshot recorded."""
    inferred = _number(proc.get("vram_inferred_bytes"))
    measured = _number(proc.get("vram_bytes"))
    if proc.get("vram_attributed") is False:
        if inferred is None:
            return "n/a"
        mark = {"order": "?", "consistent": "*"}.get(proc.get("vram_inferred_by") or "", "")
        return gb(inferred) + mark
    if measured is None:
        return "n/a"
    return gb(measured)


def _shared(snapshot: dict, proc: dict) -> str:
    """A recorded shared-memory counter. Blank when this row has no such counter; ``n/a`` when it should and didn't."""
    number = _number(proc.get("shared_bytes"))
    if number is not None:
        return gb(number)
    backend = snapshot.get("backend")
    host = proc.get("host")
    applies = backend == "wddm" or host == "windows" or (backend == "drm" and host not in ("windows", "outside"))
    return "n/a" if applies else ""


def _pct(value) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    if number == int(number):
        return str(int(number))
    return f"{number:.1f}"


def _note(proc: dict) -> str:
    parts = []
    if proc.get("vram_attributed") is False and _number(proc.get("vram_inferred_bytes")) is not None:
        parts.append("inferred")
    reason = proc.get("vram_unattributed_reason")
    if isinstance(reason, str) and reason:
        parts.append(_plain(reason))
    label = _model_label(proc)
    if label:
        parts.append(label)
    return "  ".join(parts)


def _model_label(proc: dict) -> str:
    llm = proc.get("llm") if isinstance(proc.get("llm"), dict) else None
    if llm and isinstance(llm.get("models"), list):
        labels = []
        for model in llm["models"]:
            if not isinstance(model, dict):
                continue
            name = model.get("label") or model.get("name")
            if isinstance(name, str) and name:
                labels.append(_plain(name))
        return ", ".join(labels)
    files = proc.get("model_files") if isinstance(proc.get("model_files"), list) else []
    shown = []
    for model in files[:2]:
        if not isinstance(model, dict):
            continue
        sources = model.get("sources") if isinstance(model.get("sources"), list) else []
        name = model.get("store_name") if isinstance(model.get("store_name"), str) else ""
        if not name and isinstance(model.get("path"), str):
            name = os.path.basename(model["path"])
        name = _plain(name)
        if not name:
            continue
        confirmed = any(source in ("api", "mapped", "cmdline") for source in sources)
        if "path-arg" in sources and not confirmed:
            shown.append(f"path-arg (unproven): {name}")
        else:
            shown.append(name)
    return ", ".join(shown)
