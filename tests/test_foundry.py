from __future__ import annotations

import contextlib
import http.server
import json
import threading
import unittest
from unittest.mock import Mock, patch

from textual.widgets import Static

from gputaskman.app import GpuTaskManager
from gputaskman.collector import Collector
from gputaskman.llm import ServerProber, UNAUTHORIZED, foundry_models, probe_server
from gputaskman.pdh import AdapterGpu, ProcGpu
from gputaskman.procinfo import ProcDetail, ProcInspector, classify
from gputaskman.report import to_dict, to_text

MiB = 1 << 20
MODEL = "Phi-3.5-mini-instruct-generic-gpu:2"


@contextlib.contextmanager
def listener(*, redirect=False):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.command, self.path, self.headers.get("Authorization")))
            body = json.dumps([MODEL]).encode()
            self.send_response(302 if redirect else 200)
            if redirect:
                self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def collector(ports, *, started=1, available=None):
    c = Collector.__new__(Collector)
    c.adapter, c.adapters = None, []
    c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
    c.nvml = Mock(ok=False)
    c.counters = Mock(outside={})
    gpu = ProcGpu(dedicated=512 * MiB, shared=0)
    if available is not None:
        gpu.available = set(available)
        gpu.dedicated = 0
        gpu.engines = {"3d": 10}
    c.counters.sample.return_value = (
        {0: {10: gpu}}, {0: AdapterGpu(dedicated=512 * MiB, shared=0)})
    c.inspector = ProcInspector()
    c.inspector._cache = {10: ProcDetail(10, "foundrylocald.exe", create_time=started)}
    c.inspector._ports, c.inspector._ports_time = ports, float("inf")
    c.inspector.needs_scan = Mock(return_value=False)
    c.inspector.prune = Mock()
    c.prober = ServerProber()
    c._parent_ports = Mock(side_effect=AssertionError("Foundry must not probe parents"))
    return c


def sample(c):
    with patch("gputaskman.collector.BACKEND", "wddm"), \
            patch("gputaskman.procinfo._create_time", return_value=c.inspector._cache[10].create_time):
        return c.sample()


class FoundryTests(unittest.TestCase):
    def test_only_daemon_signature_enables_discovery_without_mapped_models(self):
        for name in ("foundrylocald.exe", "FoundryLocalD.exe", "foundrylocald"):
            self.assertEqual(classify(ProcDetail(10, name), {}), "AI")
        for name in ("foundry.exe", "foundry", "not-foundrylocald.exe"):
            self.assertFalse(ProcDetail(10, name).foundry_local)

    def test_loaded_ids_are_not_catalog_entries_or_memory_estimates(self):
        for response in (None, {}, {"data": [{"id": MODEL}]}, {"models": [{"id": MODEL}]},
                         [None], [123], [""], ["  "], [MODEL, {}]):
            with self.subTest(response=response):
                self.assertIsNone(foundry_models(response))
        self.assertEqual(foundry_models([]), [])
        model = foundry_models([MODEL])[0]
        self.assertEqual(model["name"], MODEL)
        self.assertTrue(all(v is None for k, v in model.items() if k != "name"))

    def test_own_listener_in_collector_text_and_json_without_file_evidence(self):
        with listener() as (port, seen):
            c = collector({10: [port]})
            snap = sample(c)
            sample(c)
        row = snap.procs[0]
        self.assertEqual(seen, [("GET", "/models/loaded", None)])
        self.assertEqual(row.kind, "AI")
        self.assertEqual(row.inside, f"reported (API): {MODEL}")
        self.assertEqual(row.llm.server_source, "process")
        self.assertEqual(row.llm.files, [])
        self.assertEqual(row.llm.models, [])
        self.assertEqual(row.detail.models, [])
        self.assertIsNone(row.llm.n_ctx)
        self.assertIsNone(row.llm.kv_per_token)
        self.assertEqual([(s.kind, s.bytes) for s in row.llm.segments], [("unknown", 512 * MiB)])
        self.assertEqual(snap.attributed, 512 * MiB)
        self.assertIn("not per-model GPU placement", " ".join(row.llm.notes))
        self.assertIn("CPU and NPU models", " ".join(row.llm.notes))
        data = to_dict(snap)["processes"][0]
        self.assertEqual(data["llm"]["server"]["models"][0]["name"], MODEL)
        self.assertIsNone(data["llm"]["server"]["models"][0]["size_vram"])
        self.assertIn(f"API-reported model ID: {MODEL}", to_text(snap))

    def test_identity_is_available_even_when_memory_counters_are_unknown(self):
        with patch("gputaskman.llm._get", return_value=[MODEL]):
            snap = sample(collector({10: [12345]}, available={"engines"}))
        data = to_dict(snap)["processes"][0]
        self.assertEqual(data["llm"]["server"]["models"][0]["name"], MODEL)
        self.assertIsNone(data["vram_bytes"])
        self.assertIsNone(data["shared_bytes"])
        self.assertIsNone(snap.attributed)
        self.assertEqual(data["llm"]["segments"], [])

    def test_no_parent_shared_multiple_or_unidentified_listener_is_probed(self):
        for ports, started in (({20: [12345]}, 1), ({10: [12345], 20: [12345]}, 1),
                               ({10: [12345, 12346]}, 1), ({10: [12345]}, 0)):
            with self.subTest(ports=ports, started=started), \
                    patch("gputaskman.llm._get") as get:
                snap = sample(collector(ports, started=started))
                get.assert_not_called()
                self.assertIsNone(snap.procs[0].llm.server)
                self.assertIn("no unambiguous loaded-model API report", " ".join(snap.procs[0].llm.notes))

    def test_unload_and_failed_refresh_remove_previous_api_identity(self):
        p = ServerProber()
        with patch("gputaskman.llm._get", side_effect=[[MODEL], [], None]) as get, \
                patch("gputaskman.llm.time.time", side_effect=[100, 101, 104, 108]):
            self.assertEqual(p.get([12345], kind="foundry-local").models[0]["name"], MODEL)
            self.assertEqual(p.get([12345], kind="foundry-local").models[0]["name"], MODEL)
            self.assertEqual(p.get([12345], kind="foundry-local").models, [])
            self.assertIsNone(p.get([12345], kind="foundry-local"))
        self.assertEqual(get.call_count, 3)

    def test_pid_reuse_does_not_reuse_previous_api_report(self):
        p = ServerProber()
        with patch("gputaskman.llm._get", side_effect=[[MODEL], []]) as get:
            p.get([12345], kind="foundry-local", origin=("process", 10, 1))
            self.assertEqual(p.get([12345], kind="foundry-local", origin=("process", 10, 2)).models, [])
        self.assertEqual(get.call_count, 2)

    def test_redirects_and_401_do_not_fall_back_to_catalog_or_other_endpoints(self):
        with listener(redirect=True) as (port, seen):
            self.assertIsNone(probe_server(port, kind="foundry-local"))
        self.assertEqual(seen, [("GET", "/models/loaded", None)])
        p = ServerProber()
        with patch("gputaskman.llm._get", return_value=UNAUTHORIZED) as get:
            self.assertIsNone(p.get([12345], kind="foundry-local"))
        get.assert_called_once_with("http://127.0.0.1:12345/models/loaded", redirects=False)
        self.assertEqual(p.locked_port([12345], kind="foundry-local"), 12345)


class FoundryUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_reported_model_in_inspector(self):
        with patch("gputaskman.llm._get", return_value=[MODEL]):
            snap = sample(collector({10: [12345]}))
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "wddm"):
            app = GpuTaskManager()
            app.selected_key = "10"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = "\n".join(getattr(line, "plain", "") for line in
                                 app.query_one("#inspector", Static).content.renderables)
                self.assertIn(f"API-reported model ID: {MODEL}", text)
                self.assertIn("not per-model GPU placement", text)
