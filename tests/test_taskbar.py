from pathlib import Path
import ctypes as C
from ctypes import wintypes as W
import tempfile
import unittest
from unittest.mock import Mock

from gputaskman.i18n import language
from gputaskman.taskbar import PanelSettings, TaskbarPanel, panel_rect, visible_lines


class TaskbarLayoutTests(unittest.TestCase):
    def test_default_position_stays_before_notification_area(self):
        x, y, width, height = panel_rect((0, 1392, 2560, 1440), 2340, 318)
        self.assertLessEqual(x + width, 2340 - 6)
        self.assertGreaterEqual(y, 1392)
        self.assertLessEqual(y + height, 1440)

    def test_drag_position_scales_and_clamps_on_another_display(self):
        bar = (-1920, 1032, 0, 1080)
        left = panel_rect(bar, -220, 300, 1, -1)
        right = panel_rect(bar, -220, 300, 1, 99)
        middle = panel_rect(bar, -220, 300, 1, 0.5)
        self.assertEqual(left[0], -1914)
        self.assertEqual(right[0] + right[2], -226)
        self.assertEqual(middle[0], round((left[0] + right[0]) / 2))
        scaled = panel_rect((0, 2000, 3840, 2072), 3500, 450, 1.5)
        self.assertEqual(scaled[1], 2009)
        self.assertEqual(scaled[3], 54)

    def test_too_small_and_vertical_taskbars_are_not_covered(self):
        self.assertIsNone(panel_rect((0, 0, 48, 1080), 48, 300))
        self.assertIsNone(panel_rect((0, 1000, 800, 1048), 100, 300))

    def test_multi_gpu_rows_and_pause_remain_distinct(self):
        with language("ru"):
            rows = visible_lines("[ПАУЗА] GPU 0: N/A\nGPU 1: 0%\nGPU 2: 70%")
            self.assertEqual(rows[0], "[ПАУЗА] GPU 0: N/A")
            self.assertEqual(rows[1], "GPU 1: 0% … (ещё GPU: 1)")
            self.assertEqual(visible_lines("GPU 0: 0%"), ["GPU 0: 0%"])

    def test_fast_drag_uses_original_grab_point_even_when_cursor_has_moved(self):
        panel = TaskbarPanel.__new__(TaskbarPanel)
        panel.u, panel.position, panel._save = Mock(), Mock(), Mock()
        panel._rect, panel._bar = (1988, 1398, 360, 36), (0, 1392, 2560, 1440)
        panel.scale, panel._tray_left = 1, 2354
        panel.settings = PanelSettings()
        panel._drag = None
        def final_cursor(pointer):
            point = C.cast(pointer, C.POINTER(W.POINT)).contents
            point.x, point.y = 2028, 1416
            return True
        panel.u.GetCursorPos.side_effect = final_cursor
        # Mouse-down happened at screen x=2228 (client 240); by the time its
        # message is processed the cursor is already 200 px farther left.
        panel._window_proc(1, 0x0201, 0, (18 << 16) | 240)
        panel._window_proc(1, 0x0200, 1, 0)
        panel._window_proc(1, 0x0202, 0, 0)
        result = panel_rect(panel._bar, panel._tray_left, 300, panel.scale, panel.settings.position)
        self.assertEqual(result[0], 1788)
        self.assertIsNone(panel._drag)
        panel._save.assert_called_once()


class TaskbarSettingsTests(unittest.TestCase):
    def test_visibility_and_drag_position_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gputaskman" / "taskbar.json"
            self.assertEqual(PanelSettings.load(path), PanelSettings())
            settings = PanelSettings(False, 0.35)
            settings.save(path)
            self.assertEqual(PanelSettings.load(path), settings)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_malformed_settings_use_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "taskbar.json"
            for value in ('not json', '[]', 'null', '{"enabled":false,"position":true}',
                          '{"enabled":false,"position":NaN}', '{"enabled":false,"position":2}'):
                with self.subTest(value=value):
                    path.write_text(value, encoding="utf-8")
                    self.assertEqual(PanelSettings.load(path), PanelSettings())
