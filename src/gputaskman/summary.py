"""Summarize a `gputaskman record` file: average and peak of each recorded reading, per run.

Like `diff`, this only reads a saved file and collects nothing. Readings are taken as recorded: a `null` is
skipped (never counted as 0), a metric with no readings is left out, and values are never capped or adjusted.
Per-process VRAM comes only from the measured `vram_bytes`, so nothing inferred is summed. Recorded
`graphics_swapped_bytes` (macOS footprint swapped) is a separate peak, never added to VRAM, and the list
is ordered by whichever peak is larger.

`record` appends, so one file can hold several recordings. A new run starts when the adapter or backend changes,
when `taken` goes backwards, or when two samples are further apart than RUN_GAP_FACTOR times the file's
median interval (and at least RUN_GAP_MIN seconds). Each run is summarized on its own.
"""

from __future__ import annotations

import json
import math
import statistics
import time

from .diff import adapter_identity, compare_adapter_identities
from .llm import gb

RUN_GAP_FACTOR = 10
RUN_GAP_MIN = 60.0
TOP = 10

# (key, label, kind, reader): kind picks the text format; the reader takes one snapshot and returns a reading.
METRICS = (
    ("vram_used", "VRAM used", "bytes", lambda s: s.get("vram_used")),
    ("shared_used", "shared memory used", "bytes", lambda s: s.get("shared_used")),
    ("gpu_load", "GPU load", "pct", lambda s: _dev(s, "util_gpu")),
    ("temperature", "temperature", "temp", lambda s: _dev(s, "temp")),
    ("power", "power", "watts", lambda s: _dev(s, "power_w")),
    ("system_ram_used", "system RAM used", "bytes", lambda s: _sys(s, "ram", "used")),
    ("system_cpu", "system CPU", "pct", lambda s: _sys(s, "cpu", "pct")),
)


def _number(v) -> float | None:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _dev(s: dict, key: str):
    d = s.get("device")
    return d.get(key) if isinstance(d, dict) else None


def _sys(s: dict, section: str, key: str):
    d = s.get("system")
    d = d.get(section) if isinstance(d, dict) else None
    return d.get(key) if isinstance(d, dict) else None


def load(path: str) -> tuple[list[dict], int]:
    """The snapshots in a record file, in file order, and how many lines were skipped because they don't parse."""
    with open(path, "rb") as f:
        raw = f.read()
    # Windows PowerShell 5.1 redirects (`>`) as UTF-16; record itself and other shells write UTF-8.
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", "replace")
    samples, skipped = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            s = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(s, dict) and isinstance(s.get("processes"), list) and _number(s.get("taken")) is not None:
            samples.append(s)
        else:
            skipped += 1
    return samples, skipped


def _adapter_conflicts(known: dict, current: dict, sample: dict) -> bool:
    try:
        compare_adapter_identities(known, current, sample)
    except ValueError:
        return True
    return False


def split_runs(samples: list[dict]) -> list[list[dict]]:
    gaps = [b["taken"] - a["taken"] for a, b in zip(samples, samples[1:]) if b["taken"] > a["taken"]]
    limit = max(RUN_GAP_MIN, RUN_GAP_FACTOR * statistics.median(gaps)) if gaps else RUN_GAP_MIN
    runs: list[list[dict]] = []
    identities: dict[tuple, dict[int | None, dict]] = {}
    for s in samples:
        prev = runs[-1][-1] if runs else None
        identity = adapter_identity(s)
        gpu, backend = s.get("gpu"), s.get("backend")
        if backend == "wddm":
            identity.pop("key", None)  # A Windows LUID alone cannot establish a conflict.
        if "pci" not in identity:
            identity.pop("pci_domain", None)
        if (prev is None or (gpu, backend) != (prev.get("gpu"), prev.get("backend"))
                or not 0 <= s["taken"] - prev["taken"] <= limit
                or (identity and any(_adapter_conflicts(known, identity, s)
                                     for keys in identities.values() for known in keys.values()))):
            runs.append([])
            identities.clear()
        runs[-1].append(s)
        if identity:
            signature = tuple((field, value) for field, value in identity.items() if field != "key")
            keys = identities.setdefault(signature, {})
            key = identity.get("key")
            # Two distinct keys already prove any future key-only mismatch for this signature.
            if len(keys) < 2:
                keys[key] = identity
            elif None in keys and key is not None:
                del keys[None]
                keys[key] = identity
    return runs


def _stats(values: list) -> dict | None:
    vs = [v for v in (_number(x) for x in values) if v is not None]
    if not vs:
        return None
    return {"avg": sum(vs) / len(vs), "peak": max(vs), "readings": len(vs)}


def _kind(run: list[dict], key: str) -> str:
    """Where a fallback temperature or power reading comes from (for example "CPU package"), if any sample says."""
    return next((k for k in (_dev(s, key) for s in run) if isinstance(k, str) and k), "")


def _larger_peak(p: dict) -> float | None:
    """Whichever measured peak is larger. A missing one is skipped, not taken as 0."""
    peaks = [v for v in (p["peak_vram_bytes"], p["peak_graphics_swapped_bytes"]) if v is not None]
    return max(peaks) if peaks else None


def _processes(run: list[dict]) -> list[dict]:
    """Per (host, name): peaks of measured VRAM and footprint swapped, and peak GPU%, ordered by the larger peak."""
    seen: dict[tuple, int] = {}
    vram: dict[tuple, list] = {}
    swapped: dict[tuple, list] = {}
    busy: dict[tuple, list] = {}
    for s in run:
        v_sample: dict[tuple, float] = {}
        c_sample: dict[tuple, float] = {}
        l_sample: dict[tuple, float] = {}
        keys = set()
        for p in s["processes"]:
            if not isinstance(p, dict):
                continue
            key = (p.get("host"), p.get("name") or "?")
            keys.add(key)
            if (v := _number(p.get("vram_bytes"))) is not None:
                v_sample[key] = v_sample.get(key, 0) + v
            if (c := _number(p.get("graphics_swapped_bytes"))) is not None:
                c_sample[key] = c_sample.get(key, 0) + c
            if (g := _number(p.get("gpu_pct"))) is not None:
                l_sample[key] = l_sample.get(key, 0) + g
        for key in keys:
            seen[key] = seen.get(key, 0) + 1
        for key, v in v_sample.items():
            vram.setdefault(key, []).append(v)
        for key, c in c_sample.items():
            swapped.setdefault(key, []).append(c)
        for key, g in l_sample.items():
            busy.setdefault(key, []).append(g)
    out = [{"name": name, "host": host, "samples_seen": n,
            "peak_vram_bytes": max(vram.get((host, name), ()), default=None),
            "peak_graphics_swapped_bytes": max(swapped.get((host, name), ()), default=None),
            "peak_gpu_pct": max(busy.get((host, name), ()), default=None)} for (host, name), n in seen.items()]
    out.sort(key=lambda p: ((rank := _larger_peak(p)) is None, -(rank or 0), -(p["peak_gpu_pct"] or 0)))
    return out


def summarize_run(run: list[dict]) -> dict:
    metrics = {}
    for key, label, kind, read in METRICS:
        st = _stats([read(s) for s in run])
        if st:
            metrics[key] = {"label": label, "unit": kind, **st}
    if "temperature" in metrics and (k := _kind(run, "temp_kind")):
        metrics["temperature"]["source"] = k
    if "power" in metrics and (k := _kind(run, "power_kind")):
        metrics["power"]["source"] = k
    total = next((t for t in (_number(s.get("vram_total")) for s in run) if t), None)
    return {"gpu": run[0].get("gpu"), "backend": run[0].get("backend"),
            "start": run[0]["taken"], "end": run[-1]["taken"], "seconds": round(run[-1]["taken"] - run[0]["taken"], 1),
            "samples": len(run), "vram_total": total,
            "memory_note": next((s["memory_note"] for s in run if s.get("memory_note")), None),
            "metrics": metrics, "processes": _processes(run)}


def summarize(samples: list[dict], skipped: int = 0) -> dict:
    return {"samples": len(samples), "skipped_lines": skipped,
            "runs": [summarize_run(r) for r in split_runs(samples)]}


# ---------------------------------------------------------------- text


def _fmt(v, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind == "bytes":
        return gb(v)
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "temp":
        return f"{v:.0f} C"
    return f"{v:.0f} W"


def _span(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def _when(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def to_text(d: dict, top: int = TOP) -> str:
    out = [f"gputaskman summary: {d['samples']} sample(s) in {len(d['runs'])} run(s)"]
    if d["skipped_lines"]:
        out.append(f"{d['skipped_lines']} line(s) that do not parse as a snapshot were skipped.")
    if len(d["runs"]) > 1:
        out.append(f"A new run starts when the GPU adapter or backend changes, the clock goes backwards, "
                   f"or samples are more than {RUN_GAP_FACTOR}x the usual "
                   f"interval (at least {RUN_GAP_MIN:g} s) apart.")
    out.append("Values are shown as recorded; unknown readings are skipped, not counted as 0.")
    for i, r in enumerate(d["runs"], 1):
        total = f", {gb(r['vram_total'])} total" if r["vram_total"] else ""
        out += ["", f"Run {i}: {r['gpu'] or 'unknown GPU'} ({r['backend']}){total}",
                f"{r['samples']} sample(s) over {_span(r['seconds'])}, {_when(r['start'])} to {_when(r['end'])}"]
        if r["metrics"]:
            out += ["", f"{'METRIC':<24}{'AVERAGE':>11}{'PEAK':>11}{'READINGS':>11}"]
            for m in r["metrics"].values():
                label = f"{m['label']} ({m['source']})" if m.get("source") else m["label"]
                # A space before each column keeps an oversized value (shown as recorded) from fusing with its neighbour.
                out.append(f"{label[:23]:<24} {_fmt(m['avg'], m['unit']):>10} {_fmt(m['peak'], m['unit']):>10}"
                           f" {m['readings']:>5}/{r['samples']:<4}")
        if r["memory_note"]:
            out.append(r["memory_note"])
        procs = r["processes"][:top]
        if procs:
            compressed = any(p["peak_graphics_swapped_bytes"] is not None for p in r["processes"])
            head = f"{'TOP PROCESSES (same name summed)':<34}{'PEAK VRAM':>11}"
            if compressed:
                head += f"{'COMPRESSED':>11}"
            out += ["", head + f"{'PEAK GPU':>10}{'SEEN':>11}"]
            for p in procs:
                name = (f"{'W' if p['host'] == 'windows' else 'H'}:" if p["host"] else "") + p["name"]
                swapped = (f" {_fmt(p['peak_graphics_swapped_bytes'], 'bytes'):>10}" if compressed else "")
                out.append(f"{name[:33]:<34} {_fmt(p['peak_vram_bytes'], 'bytes'):>10}{swapped}"
                           f" {_fmt(p['peak_gpu_pct'], 'pct'):>9} {p['samples_seen']:>5}/{r['samples']:<4}")
            if len(r["processes"]) > top:
                out.append(f"{len(r['processes']) - top} more process(es); --json lists all.")
            out.append("Peak VRAM counts measured bytes only (vram_bytes); n/a means none were measured.")
            if compressed:
                out.append("Compressed counts recorded graphics_swapped_bytes only (footprint swapped) "
                           "and is not added to peak VRAM.")
    return "\n".join(out)
