"""What's inside an LLM process's VRAM: weights vs KV cache vs compute/runtime.

Evidence, best first:
  1. llama.cpp / Ollama load logs  -> timestamped load-time buffer history
  2. Local server endpoints        -> llama-server /props,/v1/models,/slots,/metrics; Ollama /api/ps;
                                      LM Studio /api/v1/models (falls back to /api/v0/models);
                                      Foundry Local /models/loaded (loaded IDs only)
  3. GGUF header + command line    -> estimates (weights per layer, KV bytes per token x context)
The process's real VRAM total always comes from Windows, so the remainder is never invented.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .gguf import GgufInfo, is_gguf, read_gguf
from .procinfo import SECRET_FLAGS, MappedModel, ProcDetail, file_key, make_model, target_path

MiB = 1 << 20
KV_TYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
                 "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32}

_FLAGS = {
    "-c": "ctx", "--ctx-size": "ctx", "-ngl": "ngl", "--n-gpu-layers": "ngl", "--gpu-layers": "ngl",
    "-np": "parallel", "--parallel": "parallel", "-ctk": "ctk", "--cache-type-k": "ctk",
    "-ctv": "ctv", "--cache-type-v": "ctv", "--port": "port", "--log-file": "log_file",
}
_BOOL_FLAGS = {"--no-mmap": "no_mmap"}


def parse_flags(cmd: list[str]) -> dict:
    out: dict = {}
    for i, arg in enumerate(cmd):
        key, val = arg, None
        if arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
        if key in _BOOL_FLAGS:
            out[_BOOL_FLAGS[key]] = True
        elif key in _FLAGS:
            if val is None and i + 1 < len(cmd):
                val = cmd[i + 1]
            if val is not None:
                name = _FLAGS[key]
                if name in ("ctx", "ngl", "parallel", "port"):
                    try:
                        out[name] = int(val)
                    except ValueError:
                        if name == "ngl" and val in ("all", "auto"):
                            out[name] = -1
                else:
                    out[name] = val
    return out


# ---------------------------------------------------------------- runtime log history

_LLAMA_BUF = re.compile(r"(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
_OLLAMA_BUF = re.compile(r'msg="(model weights|kv cache|compute graph)"\s+device=(\S+)\s+size="([\d.]+)\s*(GiB|MiB|KiB|B)"')
_LOG_TIMESTAMP = re.compile(
    r'^\s*(?:time="?|\[)?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?'
    r'(?:Z|[+-]\d{2}:?\d{2})?)(?=[\s"\]]|$)'
)
_UNIT = {"GiB": 1 << 30, "MiB": 1 << 20, "KiB": 1 << 10, "B": 1}
_KIND = {"model": "weights", "model weights": "weights", "KV": "kv", "kv cache": "kv", "RS": "kv",
         "compute": "compute", "compute graph": "compute", "output": "compute"}


@dataclass
class LoggedBuffer:
    device: str
    kind: str
    bytes: int
    timestamp: str | None
    timestamp_epoch: float | None
    timestamp_issue: str | None


def _log_timestamp(line: str) -> tuple[str | None, float | None, str | None]:
    match = _LOG_TIMESTAMP.search(line)
    if not match:
        return None, None, "missing"
    value = match[1]
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value, None, "unparseable"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return value, None, "timezone-unknown"
    try:
        return value, _datetime_epoch(parsed), None
    except (OSError, OverflowError, ValueError):
        return value, None, "unrepresentable"


def _datetime_epoch(value: datetime.datetime) -> float:
    return value.timestamp()


def is_gpu_device(dev: str) -> bool:
    d = dev.upper()
    return not (d.startswith("CPU") or "HOST" in d or d.startswith("MMAP"))


def parse_load_log(path: str) -> list[LoggedBuffer]:
    """Read recent buffer records, not a proven load block or a set of simultaneous allocations."""
    limit = 512 << 10
    try:
        with open(path, "rb") as f:
            offset = max(f.seek(0, 2) - limit, 0)
            f.seek(offset)
            tail = f.read(limit)
    except OSError:
        return []
    if offset:
        # A clipped line could lose its timestamp or contain only part of a buffer record.
        tail = tail.partition(b"\n")[2]
    lines = tail.decode("utf-8", "replace").splitlines()
    start = None
    for i in range(len(lines) - 1, -1, -1):  # Bound displayed history to the most recent weights run onward.
        if "model buffer size" in lines[i] or 'msg="model weights"' in lines[i]:
            start = i
            while start > 0 and ("model buffer size" in lines[start - 1] or 'msg="model weights"' in lines[start - 1]):
                start -= 1
            break
    if start is None:
        return []
    out = []
    for line in lines[start:]:
        m = _LLAMA_BUF.search(line)
        if m:
            dev, kind, n = m[1].rstrip(":"), _KIND[m[2]], int(float(m[3]) * MiB)
        else:
            m = _OLLAMA_BUF.search(line)
            if not m:
                continue
            dev, kind, n = m[2], _KIND[m[1]], int(float(m[3]) * _UNIT[m[4]])
        out.append(LoggedBuffer(dev, kind, n, *_log_timestamp(line[:m.start()])))
    return out


def ollama_log_path() -> str:
    if sys.platform == "win32":
        return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log")
    # Linux service installs log to the systemd journal; a user-run `ollama serve` may write here.
    return os.path.join(os.path.expanduser("~"), ".ollama", "logs", "server.log")


# ---------------------------------------------------------------- local servers

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# urllib forwards the Authorization header across redirects, so a request that carries a key never follows one.
_keyed_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


UNAUTHORIZED = object()  # _get result for HTTP 401: the server wants a key


def api_key(cmd: list[str]) -> str | None:
    """llama-server's API key from its own launch arguments, if it was given one there.

    `--api-key` may list several comma-separated keys; any of them works. `--api-key-file` is not read: its path
    comes from another process's command line, and opening it would read a file on that process's behalf. A key
    set only through the LLAMA_API_KEY environment variable can't be seen from outside the process.
    The key is only sent back to that process's own localhost port and is never stored or shown.
    """
    for i, arg in enumerate(cmd):
        key, val = arg, None
        if "=" in arg and arg.startswith("--"):
            key, val = arg.split("=", 1)
        if key != "--api-key":
            continue
        if val is None and i + 1 < len(cmd):
            val = cmd[i + 1]
        if val:
            return next((k.strip() for k in val.split(",") if k.strip()), None)
    return None


def redact_cmdline(cmd: list[str]) -> list[str]:
    """Command line with API keys and key-file paths replaced, for display."""
    out, hide = [], False
    for arg in cmd:
        if hide:
            out.append("***")
            hide = False
        elif arg in SECRET_FLAGS:
            out.append(arg)
            hide = True
        elif "=" in arg and arg.split("=", 1)[0] in SECRET_FLAGS:
            out.append(arg.split("=", 1)[0] + "=***")
        else:
            out.append(arg)
    return out


def _get(url: str, timeout: float = 0.5, key: str | None = None, redirects: bool = True):
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with (_keyed_opener if key or not redirects else _opener).open(urllib.request.Request(url, headers=headers),
                                                       timeout=timeout) as r:
            if r.status != 200:
                return None
            body = r.read(4 << 20)
    except urllib.error.HTTPError as e:
        return UNAUTHORIZED if e.code == 401 else None
    except Exception:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace")


@dataclass
class ServerInfo:
    kind: str  # llama-server | ollama | lmstudio | vllm | foundry-local
    url: str
    n_ctx: int | None = None
    slots_total: int | None = None
    slots_busy: int | None = None
    gen_tok_s: float | None = None
    prompt_tok_s: float | None = None
    model_path: str | None = None  # llama-server: the file it says it loaded
    models: list[dict] = field(default_factory=list)  # normalized: name, type, quant, ctx, size, size_vram, path
    # Server-level runtime reports, never per-process or per-GPU memory measurements.
    requests_running: int | None = None
    requests_waiting: int | None = None
    kv_cache_usage: float | None = None  # fraction 0-1 of KV-cache pool, not of GPU memory
    models_complete: bool = True  # internal guard against inferring a singleton from partial API data


def _metric(text: str, name: str) -> float | None:
    m = re.search(rf"^llamacpp:{name}\s+(\S+)", text, re.M)
    if not m:
        return None
    try:
        value = float(m[1])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


_VLLM_GAUGES = {"vllm:num_requests_running": "requests_running",
                "vllm:num_requests_waiting": "requests_waiting",
                "vllm:kv_cache_usage_perc": "kv_cache_usage"}
_PROM_SAMPLE = re.compile(r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?[ \t]+(\S+)(?:[ \t]+-?\d+)?[ \t]*")
_PROM_LABEL = re.compile(r'[ \t]*([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"[ \t]*(?:,|$)')


def _prom_labels(text: str) -> dict[str, str] | None:
    labels, pos = {}, 0
    while pos < len(text):
        m = _PROM_LABEL.match(text, pos)
        if not m or m[1] in labels:
            return None
        labels[m[1]] = re.sub(r"\\(.)", lambda e: "\n" if e[1] == "n" else e[1], m[2])
        pos = m.end()
    return labels


def vllm_metrics(text: str, model_ids: list[str]) -> dict[str, int | float | None]:
    """Three server-level gauges from vLLM's Prometheus text, or unknown for each.

    Accepted only for exactly one engine and one metric model identity. vLLM labels metrics with its first
    --served-model-name, so that one label must be among the advertised /v1/models IDs (aliases allowed).
    A reported 0 stays 0; a missing, duplicated, malformed or out-of-range sample stays None.
    """
    out: dict[str, int | float | None] = dict.fromkeys(_VLLM_GAUGES.values())
    samples: dict[str, list[float | None]] = {}
    engines, names = set(), set()
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _PROM_SAMPLE.fullmatch(line)
        if not m or m[1] not in _VLLM_GAUGES:
            continue
        labels = _prom_labels(m[2] or "")
        if labels is None or not labels.get("engine") or not labels.get("model_name"):
            return out  # identity can't be checked
        engines.add(labels.get("engine"))
        names.add(labels.get("model_name"))
        try:
            value = float(m[3])
        except ValueError:
            value = None
        samples.setdefault(m[1], []).append(value if value is not None and math.isfinite(value) else None)
    if len(engines) != 1 or len(names) != 1 or next(iter(names)) not in model_ids:
        return out
    for metric, key in _VLLM_GAUGES.items():
        values = samples.get(metric, [])
        value = values[0] if len(values) == 1 else None
        if value is None:
            continue
        if key == "kv_cache_usage":
            out[key] = value if 0 <= value <= 1 else None
        else:
            out[key] = int(value) if value >= 0 and value.is_integer() else None
    return out


def server_gauge_text(s: ServerInfo) -> str | None:
    """One runtime-gauge line, or None when none is known. Unknown shows as n/a, 0 as 0."""
    if s.kind not in ("vllm", "llama-server") or all(
            v is None for v in (s.requests_running, s.requests_waiting, s.kv_cache_usage)):
        return None
    running, waiting = ("n/a" if v is None else str(v) for v in (s.requests_running, s.requests_waiting))
    kv = "n/a" if s.kv_cache_usage is None else f"{s.kv_cache_usage * 100:.1f}%"
    source = "vLLM /metrics" if s.kind == "vllm" else "llama-server"
    return f"{source} (runtime-reported): requests running {running}, waiting {waiting}, KV-cache pool {kv}"


def _model_entry(name, type_=None, quant=None, ctx=None, size=None, size_vram=None, path=None) -> dict:
    return {"name": name, "type": type_, "quant": quant, "ctx": ctx, "size": size, "size_vram": size_vram, "path": path}


def _runtime_text(value) -> str | None:
    return value if isinstance(value, str) else None


def _runtime_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


_MODALITY = {"text": "llm", "decisions": "decision", "embeddings": "embedding", "embedding": "embedding"}


def llama_server_models(v1: dict, model_path: str | None) -> list[dict]:
    out = []
    entries = v1.get("data")
    for m in entries if isinstance(entries, list) else []:
        if not isinstance(m, dict):
            continue
        architecture = m.get("architecture")
        outs = architecture.get("output_modalities") if isinstance(architecture, dict) else None
        outs = [o for o in outs if isinstance(o, str)] if isinstance(outs, list) else []
        meta = m.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        model_id = _runtime_text(m.get("id"))
        path = _runtime_text(model_path)
        name = os.path.basename(model_id if model_id else path or "?")
        out.append(_model_entry(name, next((_MODALITY.get(o, o) for o in outs), None),
                                _runtime_text(meta.get("ftype")), _runtime_int(meta.get("n_ctx")),
                                _runtime_int(meta.get("size")), None, path))
    return out


def ollama_models(ps: dict) -> list[dict]:
    entries = ps.get("models")
    out = []
    for m in entries if isinstance(entries, list) else []:
        if not isinstance(m, dict):
            continue
        details = m.get("details")
        details = details if isinstance(details, dict) else {}
        out.append(_model_entry(_runtime_text(m.get("name")), "llm",
                                _runtime_text(details.get("quantization_level")),
                                _runtime_int(m.get("context_length")), _runtime_int(m.get("size")),
                                _runtime_int(m.get("size_vram"))))
    return out


def lmstudio_models(v1: dict | None, v0: dict | None) -> tuple[list[dict], bool] | None:
    """Loaded models from LM Studio's /api/v1/models (preferred) or /api/v0/models."""
    if isinstance(v1, dict) and isinstance(v1.get("models"), list):
        out = []
        complete = True
        for m in v1["models"]:
            if not isinstance(m, dict):
                complete = False
                continue
            instances = m.get("loaded_instances")
            if not isinstance(instances, list):
                complete = False
                continue
            for inst in instances:
                if not isinstance(inst, dict):
                    complete = False
                    continue
                q = m.get("quantization")
                q = q if isinstance(q, dict) else {}
                t = _runtime_text(m.get("type"))
                decision_type = _runtime_text(m.get("decision_type"))
                if t == "decision" and isinstance(decision_type, str) and decision_type:
                    t = f"decision ({decision_type})"
                config = inst.get("config")
                config = config if isinstance(config, dict) else {}
                name = _runtime_text(inst.get("id"))
                if not name or not name.strip():
                    name = _runtime_text(m.get("key"))
                if not name or not name.strip():
                    complete = False
                    continue
                out.append(_model_entry(name, t, _runtime_text(q.get("name")),
                                        _runtime_int(config.get("context_length")), _runtime_int(m.get("size_bytes"))))
        return out, complete
    if isinstance(v0, dict) and isinstance(v0.get("data"), list):
        out = []
        complete = True
        for m in v0["data"]:
            if not isinstance(m, dict) or not isinstance(m.get("state"), str):
                complete = False
                continue
            if m["state"] != "loaded":
                continue
            name = _runtime_text(m.get("id"))
            if not name or not name.strip():
                complete = False
                continue
            ctx = _runtime_int(m.get("loaded_context_length"))
            if ctx is None:
                ctx = _runtime_int(m.get("max_context_length"))
            out.append(_model_entry(name, _runtime_text(m.get("type")), _runtime_text(m.get("quantization")), ctx))
        return out, complete
    return None


def vllm_models(response) -> list[dict] | None:
    """Model identifiers only: the OpenAI list says nothing about bytes or worker placement."""
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        return None
    entries = response["data"]
    if any(not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"].strip() for m in entries):
        return None
    return [_model_entry(m["id"]) for m in entries]


def foundry_models(response) -> list[dict] | None:
    """Foundry Local's loaded IDs, not its catalog or a per-model memory report."""
    if not isinstance(response, list) or any(not isinstance(m, str) or not m.strip() for m in response):
        return None
    return [_model_entry(m) for m in response]


def probe_server(port: int, prev: ServerInfo | None = None, key: str | None = None,
                 kind: str | None = None) -> ServerInfo | None:
    """Read-only GETs. Never calls inference, load or download endpoints.

    Returns UNAUTHORIZED when the port answers 401 and no key (or a wrong one) was sent. Probing stops there,
    so a locked server logs one rejected request per probe instead of one per endpoint.
    """
    base = f"http://127.0.0.1:{port}"
    if kind == "foundry-local":
        response = _get(base + "/models/loaded", redirects=False)
        if response is UNAUTHORIZED:
            return UNAUTHORIZED
        models = foundry_models(response)
        return ServerInfo("foundry-local", base, models=models) if models is not None else None
    if kind == "vllm":
        response = _get(base + "/v1/models", key=key, redirects=False)
        if response is UNAUTHORIZED:
            return UNAUTHORIZED
        models = vllm_models(response)
        if models is None:
            return None
        s = ServerInfo("vllm", base, models=models)
        # Only after /v1/models succeeds. Never sends a key; a 401 or any failure leaves the gauges unknown.
        metrics = _get(base + "/metrics", redirects=False)
        if isinstance(metrics, str):
            for key, value in vllm_metrics(metrics, [m["name"] for m in models]).items():
                setattr(s, key, value)
        return s
    props = _get(base + "/props", key=key)
    if props is UNAUTHORIZED:
        return UNAUTHORIZED
    if isinstance(props, dict) and ("default_generation_settings" in props or "model_path" in props):
        s = ServerInfo("llama-server", base, model_path=_runtime_text(props.get("model_path")))
        dgs = props.get("default_generation_settings")
        dgs = dgs if isinstance(dgs, dict) else {}
        s.n_ctx = _runtime_int(dgs.get("n_ctx"))
        if s.n_ctx is None:
            s.n_ctx = _runtime_int(props.get("n_ctx"))
        s.slots_total = _runtime_int(props.get("total_slots"))
        if prev and prev.kind == "llama-server" and prev.model_path == s.model_path:
            s.models = prev.models  # identity only changes when the model does
        else:
            v1 = _get(base + "/v1/models", key=key)
            s.models = llama_server_models(v1, s.model_path) if isinstance(v1, dict) else []
        slots = _get(base + "/slots", key=key)
        if isinstance(slots, list) and all(
                isinstance(slot, dict) and isinstance(slot.get("is_processing"), bool) for slot in slots):
            s.slots_total = len(slots)
            s.slots_busy = sum(1 for slot in slots if slot["is_processing"])
            s.requests_running = s.slots_busy
        metrics = _get(base + "/metrics", key=key)
        if isinstance(metrics, str):
            s.gen_tok_s = _metric(metrics, "predicted_tokens_seconds")
            s.prompt_tok_s = _metric(metrics, "prompt_tokens_seconds")
            kv_ratio = _metric(metrics, "kv_cache_usage_ratio")
            if kv_ratio is not None and 0.0 <= kv_ratio <= 1.0:
                s.kv_cache_usage = kv_ratio
            req_proc = _metric(metrics, "requests_processing")
            if req_proc is not None and req_proc >= 0 and req_proc.is_integer():
                s.requests_running = int(req_proc)
            req_def = _metric(metrics, "requests_deferred")
            if req_def is not None and req_def >= 0 and req_def.is_integer():
                s.requests_waiting = int(req_def)
        return s
    ps = _get(base + "/api/ps")
    if isinstance(ps, dict) and isinstance(ps.get("models"), list):
        return ServerInfo("ollama", base, models=ollama_models(ps))
    # Older LM Studio answers unknown endpoints with HTTP 200 + {"error": ...}, so fall back on shape, not status
    loaded = lmstudio_models(_get(base + "/api/v1/models"), None)
    if loaded is None:
        loaded = lmstudio_models(None, _get(base + "/api/v0/models"))
    if loaded is not None:
        models, complete = loaded
        return ServerInfo("lmstudio", base, models=models, models_complete=complete)
    return None


class ServerProber:
    """Throttled, cached probes. Only localhost, only read-only GET endpoints."""

    TTL_HIT, TTL_MISS = 3.0, 30.0

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[float, ServerInfo | None]] = {}
        self._last: dict[tuple, ServerInfo] = {}
        self._locked: set[tuple] = set()
        self.locked: set[int] = set()  # ports whose last probe was refused with 401

    @staticmethod
    def _identity(port: int, key: str | None, kind: str | None, origin: object) -> tuple:
        # Never retain the key itself; authenticated own-port results cannot satisfy a parent probe.
        return port, kind, origin, hashlib.sha256(key.encode()).digest() if key else None

    def get(self, ports: list[int], key: str | None = None, *, kind: str | None = None,
            origin: object = "process") -> ServerInfo | None:
        now = time.time()
        for port in ports:
            identity = self._identity(port, key, kind, origin)
            hit = self._cache.get(identity)
            if hit and now - hit[0] < (self.TTL_HIT if hit[1] else self.TTL_MISS):
                info = hit[1]
            else:
                info = probe_server(port, self._last.get(identity), key, kind=kind)
                if info is UNAUTHORIZED:
                    self._locked.add(identity)
                    self.locked.add(port)
                    info = None
                else:
                    self._locked.discard(identity)
                    self.locked.discard(port)
                self._cache[identity] = (now, info)
                if info:
                    self._last[identity] = info
            if info:
                return info
        return None

    def locked_port(self, ports: list[int], key: str | None = None, *, kind: str | None = None,
                    origin: object = "process") -> int | None:
        """First of these ports whose last probe was refused for a missing or wrong API key."""
        return next((p for p in ports if self._identity(p, key, kind, origin) in self._locked), None)


# ---------------------------------------------------------------- the breakdown


@dataclass
class Segment:
    label: str
    bytes: int
    kind: str  # weights | kv | compute | runtime | unknown
    exact: bool = False


@dataclass
class LlmView:
    models: list[GgufInfo] = field(default_factory=list)
    files: list[MappedModel] = field(default_factory=list)  # evidence for models[i] (same order)
    other_files: list[MappedModel] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    server: ServerInfo | None = None
    host: ServerInfo | None = None  # managing front-end (Ollama / LM Studio) when the runner has its own server
    server_source: str | None = None  # process | parent; an API report, not proof of worker residency
    log_source: str | None = None
    log: list[LoggedBuffer] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    kv_per_token: int | None = None
    kv_type: str = "f16"
    n_ctx: int | None = None

    def label(self, i: int) -> str:
        """Best human name for models[i]: model-store name, else GGUF basename/name."""
        f = self.files[i] if i < len(self.files) else None
        g = self.models[i]
        if f and f.kind_hint:
            return f.store_name or os.path.basename(f.path)
        return (f.store_name if f and f.store_name else None) or g.metadata.get("general.basename") or g.name

    def kind_label(self, i: int) -> str:
        f = self.files[i] if i < len(self.files) else None
        g = self.models[i]
        if f and f.kind_hint:
            return f.kind_label
        return g.kind + (f" ({g.decision_type})" if g.decision_type else "")


def gb(n: float) -> str:
    if abs(n) >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / MiB:.0f} MB"


LOG_HISTORY_LABEL = "Load-log history (runtime-reported; not current residency or allocation)"


def log_buffer_text(record: LoggedBuffer, process_start: float) -> str:
    when = record.timestamp or "timestamp unavailable"
    if record.timestamp_issue == "timezone-unknown":
        when += " (has no timezone)"
    elif record.timestamp_issue == "unrepresentable":
        when += " (cannot be converted on this platform)"
    elif record.timestamp_issue == "unparseable":
        when += " (unparseable timestamp)"
    elif record.timestamp_epoch is not None:
        if process_start <= 0:
            when += " (process start time unknown)"
        elif record.timestamp_epoch < process_start - 5:
            when += " (predates this process)"
    size = f"{record.bytes / MiB:.2f}".rstrip("0").rstrip(".") + " MiB" if record.bytes >= MiB else f"{record.bytes} B"
    return f"{record.device} {record.kind}: {size} at {when}"


_SIDE_KINDS = ("vision projector", "LoRA adapter")


def _merge_api(files: list[MappedModel], server: ServerInfo | None, host: ServerInfo | None,
               pid: int | None = None) -> list[dict]:
    """Mark files a runtime API also reports; return the API entries that matched this process."""
    matched: list[dict] = []
    # llama-server reports -m as it was typed, so a relative path belongs to the server's cwd, not ours.
    path = target_path(pid, server.model_path) if server and server.kind == "llama-server" and server.model_path \
        else None
    if path:
        key = file_key(path)
        hit = next((f for f in files if file_key(f.path) == key), None)
        if hit:
            hit.add_source("api")
        elif os.path.isfile(path):  # loaded without mmap or via the router: API-only evidence
            files.insert(0, make_model(path, os.path.getsize(path), None, is_gguf(path), "api"))
    for front in (s for s in (server, host) if s and s.kind == "ollama"):
        for f in files:
            if not f.has_model_evidence:
                continue
            names = set((f.store_name or "").split(", "))
            for om in front.models:
                if om["name"] in names:
                    f.add_source("api")
                    matched.append(om)
    lms = host if host and host.kind == "lmstudio" else (server if server and server.kind == "lmstudio" else None)
    primary = [f for f in files if f.has_model_evidence and f.kind not in _SIDE_KINDS]
    if lms and lms.models_complete and len(lms.models) == 1 and len(primary) == 1:
        # LM Studio keys don't map to paths; only pair a complete one-to-one listing.
        primary[0].add_source("api")
        matched.append(lms.models[0])
    return matched


def build_view(d: ProcDetail, vram: float, shared: float, server: ServerInfo | None,
               log_path: str | None = None, host: ServerInfo | None = None, attributed: bool = True,
               server_source: str | None = None) -> LlmView:
    v = LlmView(flags=parse_flags(d.cmdline), server=server, host=host,
                server_source=(server_source or "process") if server else None)
    if server and server.kind == "vllm":
        source = "parent process" if v.server_source == "parent" else "process's own listener"
        v.notes.append(f"vLLM API reported via {source}; model IDs do not prove this process's "
                       "model residency, weights, KV cache or GPU placement.")
        if server_gauge_text(server):
            v.notes.append("vLLM /metrics gauges describe the API server's engine as a whole, not this process "
                           "or one GPU; KV-cache pool usage is a share of vLLM's own KV-cache allocation, "
                           "not of GPU memory.")
    elif server and server.kind == "llama-server" and server_gauge_text(server):
        v.notes.append("llama-server gauges are runtime-reported for the server as a whole, not one GPU; "
                       "KV-cache pool usage is a share of the server's KV-cache allocation, not of GPU memory. "
                       "Running requests fall back to busy /slots when no valid processing gauge is available.")
    elif server and server.kind == "foundry-local":
        v.notes.append("Foundry Local API reports loaded model IDs from this process's own listener, "
                       "not per-model GPU placement, weights or KV-cache bytes. "
                       "The loaded list can include CPU and NPU models.")
    elif d.vllm_role and not server:
        v.notes.append("vLLM process signature only; no unambiguous local API association was established.")
    elif d.foundry_local and not server:
        v.notes.append("Foundry Local daemon signature only; no unambiguous loaded-model API report is available.")
    files = [dataclasses.replace(m, sources=list(m.sources)) for m in d.models]  # cached details stay untouched
    matched = _merge_api(files, server, host, d.pid)
    for m in files:
        if not m.has_model_evidence:
            v.other_files.append(m)
            continue
        g = read_gguf(m.path) if m.gguf else None
        if g is not None and not g.error:
            v.models.append(g)
            v.files.append(m)
        else:
            v.other_files.append(m)

    if not log_path and v.flags.get("log_file"):
        log_path = target_path(d.pid, v.flags["log_file"])  # --log-file is relative to the server's cwd
    if log_path:
        v.log = parse_load_log(log_path)
        v.log_source = log_path if v.log else None

    main = next((g for g in v.models if g.kind == "LLM" and g.n_layers), None)  # only causal LLMs grow a KV cache
    ctk = str(v.flags.get("ctk", "f16")).lower()
    ctv = str(v.flags.get("ctv", "f16")).lower()
    v.kv_type = ctk if ctk == ctv else f"K {ctk}/V {ctv}"
    if main:
        elems = main.kv_bytes_per_token(1.0)  # K+V elements per token (K and V are usually equal width)
        if elems:
            v.kv_per_token = int(elems * (KV_TYPE_BYTES.get(ctk, 2) + KV_TYPE_BYTES.get(ctv, 2)) / 2)
    v.n_ctx = v.flags.get("ctx") or (server.n_ctx if server and server.kind == "llama-server" else None)
    if main:
        for om in matched or (server.models if server and server.kind == "ollama" else []):
            if om.get("ctx"):
                v.n_ctx = v.n_ctx or om["ctx"]
    primaries = [g for g in v.models if g.kind not in _SIDE_KINDS]
    no_kv = bool(primaries) and all(not g.causal for g in primaries)

    total = int(vram) if attributed else None  # None: the OS can't say how much VRAM this process holds
    ngl = v.flags.get("ngl") if len(v.models) == 1 else None
    weights = sum(g.gpu_weight_bytes(ngl) for g in v.models if g.kind != "LoRA adapter")
    weights += sum(g.tensor_bytes for g in v.models if g.kind == "LoRA adapter")
    if total is not None and weights > total * 1.03 and total > 0:
        v.notes.append(f"Weights need ~{gb(weights)} but the process holds {gb(total)} of VRAM: "
                       "partial offload, paging, or placement on another GPU may explain the gap.")
        weights = total
    if weights:
        label = "weights" if len(v.models) == 1 else f"weights ({len(v.models)} files)"
        v.segments.append(Segment(label, weights, "weights"))
    kv = v.kv_per_token * v.n_ctx if (v.kv_per_token and v.n_ctx) else 0
    rest = total - weights if total is not None else None
    if rest is None:
        if kv:
            v.segments.append(Segment(f"KV cache ({v.n_ctx:,} ctx, {v.kv_type}) est.", kv, "kv"))
        v.notes.append("This process's VRAM total is unknown (WSL charges it to the VM on the Windows host, "
                       "container ownership is inferred, or the driver hides per-process memory); "
                       "segments are estimates from the model file and launch flags.")
    elif kv and kv <= rest:
        swa = main.sliding_window if main else None
        v.segments.append(Segment(f"KV cache ({v.n_ctx:,} ctx, {v.kv_type}) est.{' max' if swa else ''}", kv, "kv"))
        if swa:
            v.notes.append(f"Sliding-window attention ({swa} tokens): most layers keep a short KV window, "
                           "so the real KV cache is smaller than this estimate.")
        rest -= kv
        if rest > 0:
            v.segments.append(Segment("compute buffers + runtime", rest, "runtime"))
    elif rest > 0 and no_kv:
        v.segments.append(Segment("compute buffers + runtime", rest, "runtime"))
    elif rest > 0:
        v.segments.append(Segment("KV cache + compute + runtime", rest, "unknown"))

    if not v.models and total:
        v.notes.append("No GGUF mapped. Model may be loaded with --no-mmap or via another format.")
    if no_kv:
        kinds = ", ".join(sorted({f"{g.decision_type} {g.kind}" if g.decision_type else g.kind for g in primaries}))
        v.notes.append(f"Non-causal encoder ({kinds}): no KV cache grows with use; VRAM is weights + compute.")
    if main and v.kv_per_token:
        v.notes.append(f"KV cost {v.kv_per_token / 1024:.0f} KB/token ({v.kv_type}): "
                       f"4k ctx = {gb(v.kv_per_token * 4096)}, 32k ctx = {gb(v.kv_per_token * 32768)}.")
    if shared > 256 * MiB:
        v.notes.append(f"! {gb(shared)} is in SHARED GPU memory (system RAM over PCIe). "
                       "May include pinned host buffers; under VRAM pressure, fallback can slow inference.")
    if total is not None and total < 64 * MiB and v.models:
        v.notes.append("Model is mapped but almost nothing is in VRAM: CPU inference, or another GPU.")
    for front in (s for s in (server, host) if s and s.kind == "ollama"):
        mine = [om for om in front.models if om in matched]
        for om in mine or front.models:  # unmatched: show everything Ollama has loaded
            size = _runtime_int(om.get("size"))
            if size:
                size_vram = _runtime_int(om.get("size_vram"))
                pct = f"{100 * size_vram / size:.0f}%" if size_vram is not None else "n/a"
                v.notes.append(f"Ollama{'' if mine else ' (not matched to this runner)'}: {om['name']} "
                               f"{gb(size)} total, {pct} on GPU, ctx {om.get('ctx') or '?'}.")
    if host and host.kind == "lmstudio":
        for m in host.models[:4]:
            v.notes.append(f"LM Studio: {m['name']} {m.get('type') or ''} {m.get('quant') or ''} "
                           f"ctx {m.get('ctx') or '?'}.")
    return v
