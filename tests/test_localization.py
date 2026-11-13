from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import string
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

from textual.command import CommandInput
from textual.widgets import DataTable, Static

from gputaskman.app import GpuTaskManager
from gputaskman.cli import main
from gputaskman.i18n import current_language, inside, kind_label, language, tr
from gputaskman.nvml import NvDevice
from gputaskman.procinfo import MappedModel
from gputaskman.report import dump, to_dict, to_text
from gputaskman.ru import MESSAGES
from gputaskman.tray import WindowsTray, compact_summary, tooltip_text
from test_core import GiB, snapshot


class LocalizationTests(unittest.TestCase):
    def test_default_cli_is_russian_and_english_is_available(self):
        for args, expected in (([], "показать справку"), (["--lang", "en"], "show this help")):
            out = io.StringIO()
            with patch.dict(os.environ, {"GPUTASKMAN_LANG": "ru"}), \
                    patch("sys.argv", ["gputaskman", *args, "--help"]), \
                    contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn(expected, out.getvalue())
        self.assertEqual(current_language(), "en")

    def test_argparse_errors_are_translated_without_collecting(self):
        out = io.StringIO()
        with patch.dict(os.environ, {"GPUTASKMAN_LANG": "ru"}), \
                patch("sys.argv", ["gputaskman", "--lang", "invalid"]), \
                contextlib.redirect_stderr(out), self.assertRaises(SystemExit):
            main()
        self.assertIn("ошибка:", out.getvalue())
        self.assertIn("недопустимое значение", out.getvalue())

    def test_language_option_also_precedes_subcommands(self):
        for prefix in (["--lang", "en"], ["--lang=en"]):
            for command in ("diff", "record"):
                with self.subTest(prefix=prefix, command=command):
                    out = io.StringIO()
                    with patch("sys.argv", ["gputaskman", *prefix, command, "--help"]), \
                            contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stopped:
                        main()
                    self.assertEqual(stopped.exception.code, 0)
                    self.assertIn("show this help", out.getvalue())

    def test_catalogue_preserves_all_template_fields(self):
        formatter = string.Formatter()
        for source, target in MESSAGES.items():
            with self.subTest(source=source):
                before = [name for _, name, _, _ in formatter.parse(source) if name is not None]
                after = [name for _, name, _, _ in formatter.parse(target) if name is not None]
                self.assertCountEqual(before, after)

    def test_text_is_translated_but_json_and_snapshot_are_unchanged(self):
        snap = snapshot()
        snap.procs[0].detail.models = [MappedModel("C:/models/weights {secret}.gguf", GiB, None, True,
                                                 sources=["path-arg"], kind_hint="decision model")]
        snap.procs[0].llm = None
        before = copy.deepcopy(to_dict(snap))
        with language("ru"):
            text = to_text(snap)
            self.assertIn("ПРОЦЕСС", text)
            self.assertIn("композитор", text)
            self.assertIn("загрузка не подтверждена", text)
            self.assertIn("weights {secret}.gguf", text)
            self.assertEqual(json.loads(dump(snap, True)), before)
        self.assertEqual(to_dict(snap), before)

    def test_names_paths_and_provenance_survive_dynamic_translation(self):
        with language("ru"):
            path = "C:/модели/weights {0} — api.gguf"
            self.assertEqual(tr(f"   exact sizes from log: {path}"),
                             f"   точные размеры выделенных буферов из журнала: {path}")
            self.assertEqual(kind_label("likely decision model (name)"),
                             "предположительно модель принятия решений (по имени)")
            self.assertEqual(tr("confirmed (api+mapped)"), "подтверждено (api+mapped)")
            self.assertIn("не подтверждено", tr("path-arg (unproven): "))
            self.assertEqual(tr("invented runtime error"), "invented runtime error")


class CompactSummaryTests(unittest.TestCase):
    def test_unknown_is_distinct_from_zero_and_inferred_bytes_are_not_summed(self):
        snap = snapshot()
        snap.nv = NvDevice(index=0, util_gpu=0, temp=None)
        snap.vram_used = 0
        snap.procs[0].gpu.inferred = 8 * GiB
        with language("ru"):
            text = compact_summary([snap], True, gpu_index=2)
            self.assertIn("[ПАУЗА]", text)
            self.assertIn("GPU 2: 0%", text)
            self.assertIn("VRAM 0 MB/16.00 GB", text)
            self.assertIn("N/A", text)
            snap.vram_used = None
            snap.nv.util_gpu = None
            self.assertIn("GPU 0: N/A", compact_summary([snap]))
            self.assertIn("VRAM n/a/", compact_summary([snap]))

    def test_multiple_adapters_are_separate_and_intel_is_marked(self):
        first, second = snapshot(), snapshot()
        second.backend = "drm"
        second.nv = NvDevice(index=1, source="Intel sysfs", util_gpu=40)
        text = compact_summary([first, second])
        self.assertEqual(len(text.splitlines()), 2)
        self.assertIn("GPU 1: 40% · VRAM≈", text)

    def test_tooltip_fits_windows_utf16_limit(self):
        result = tooltip_text("я😀" * 200)
        self.assertLessEqual(len(result.encode("utf-16-le")), 254)
        self.assertNotIn("\0", tooltip_text("hello\0world"))

    def test_pseudoconsole_is_never_shown_or_minimized(self):
        tray = WindowsTray(lambda: None)
        tray.hwnd = 99
        tray.kernel = Mock()
        tray.kernel.GetConsoleWindow.return_value = 12
        tray.user = Mock()
        tray.user.IsWindowVisible.return_value = False
        self.assertFalse(tray.minimize())
        self.assertFalse(tray.restore())
        tray.user.ShowWindow.assert_not_called()

    def test_classic_console_uses_minimize_and_restore_not_hide(self):
        tray = WindowsTray(lambda: None)
        tray.hwnd = 99
        tray.kernel = Mock()
        tray.kernel.GetConsoleWindow.return_value = 12
        tray.user = Mock()
        tray.user.IsWindowVisible.return_value = True
        self.assertTrue(tray.minimize())
        tray.user.ShowWindow.assert_called_with(12, 6)
        self.assertTrue(tray.restore())
        tray.user.ShowWindow.assert_called_with(12, 9)


class RussianUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_command_palette_and_empty_search_are_russian(self):
        with language("ru"):
            app = GpuTaskManager(system=False)
        with patch.object(GpuTaskManager, "poll"):
            async with app.run_test(size=(140, 45)) as pilot:
                await pilot.pause()
                await pilot.press("ctrl+p")
                await pilot.pause()
                field = app.screen.query_one(CommandInput)
                self.assertEqual(field.placeholder, "Поиск команд…")
                field.value = "zzzzzzz"
                await pilot.pause(0.8)
                text = "".join(ET.fromstring(app.export_screenshot()).itertext()).replace("\xa0", " ")
                self.assertIn("Совпадений нет", text)
                await pilot.press("escape")
                app.search_themes()
                await pilot.pause()
                self.assertEqual(app.screen.query_one(CommandInput).placeholder, "Поиск тем…")

    async def test_localized_ui_retains_column_keys_and_language_in_callbacks(self):
        with language("ru"):
            app = GpuTaskManager(system=False)
        # Callbacks run outside the construction context, as worker callbacks do.
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "wddm"):
            async with app.run_test(size=(150, 45)) as pilot:
                await pilot.pause()
                app.apply(snapshot())
                await pilot.pause()
                table = app.query_one("#procs", DataTable)
                self.assertIn("Process", table.columns)
                self.assertEqual(str(table.columns["Process"].label), "Процесс")
                self.assertEqual(str(app.query_one("#device", Static).border_title), "Подробная диагностика")
                self.assertEqual(str(table.get_row_at(1)[2]), "композитор")
                svg = app.export_screenshot()
                self.assertIn("Температура", svg)
                self.assertIn("Выход", svg)
                self.assertIn("Кратко/Подробно", svg)
                await pilot.press("question_mark")
                await pilot.pause()
                help_text = "".join(ET.fromstring(app.export_screenshot()).itertext()).replace("\xa0", " ")
                self.assertIn("Что можно узнать", help_text)
                await pilot.press("escape", "p")
                await pilot.pause()
                self.assertTrue(app.paused)
        self.assertEqual(current_language(), "en")

    @unittest.skipUnless(os.name == "nt", "Windows desktop integration")
    async def test_tray_updates_from_same_snapshot_and_closes_on_exit(self):
        desktop = Mock()
        desktop.start.return_value = True
        desktop.panel_error = ""
        with language("ru"):
            app = GpuTaskManager(system=False, tray=True)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.tray.WindowsTray", return_value=desktop):
            async with app.run_test() as pilot:
                await pilot.pause()
                app.apply(snapshot())
                await pilot.pause()
                self.assertIn("VRAM 10.00 GB/16.00 GB", desktop.update.call_args.args[0])
                await pilot.press("p")
                await pilot.pause()
                self.assertIn("[ПАУЗА]", desktop.update.call_args.args[0])
                await pilot.press("t")
                desktop.minimize.assert_called_once()
        desktop.close.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows desktop integration")
    async def test_tray_failure_keeps_the_monitor_usable(self):
        desktop = Mock()
        desktop.start.return_value = False
        desktop.error = "Windows notification area did not respond"
        with language("ru"):
            app = GpuTaskManager(system=False, tray=True)
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.tray.WindowsTray", return_value=desktop):
            async with app.run_test() as pilot:
                await pilot.pause()
                app.apply(snapshot())
                await pilot.pause()
                self.assertEqual(app.query_one("#procs", DataTable).row_count, 2)
                notification = next(iter(app._notifications))
                self.assertIn("Значок в области уведомлений недоступен", notification.message)
        desktop.close.assert_called_once()
