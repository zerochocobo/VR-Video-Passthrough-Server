from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

from ui import settings as settings_module


class SettingsTests(unittest.TestCase):
    def _settings(self):
        root = Path("runtime_cache/test_ui_settings")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        patcher = patch.object(settings_module, "SETTINGS_PATH", root / "ui_settings.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        return settings_module.Settings()

    def test_passthrough_mode_mapping(self) -> None:
        s = self._settings()
        cases = [
            (False, False, "none"),
            (True, False, "green"),
            (False, True, "alpha"),
            (True, True, "all"),
        ]
        for green, alpha, expected in cases:
            with self.subTest(green=green, alpha=alpha):
                s.data["mode_green"] = green
                s.data["mode_alpha"] = alpha
                self.assertEqual(s.passthrough_mode(), expected)

    def test_server_env_omits_blank_subtitle_color(self) -> None:
        s = self._settings()
        s.data["subtitle_color"] = ""
        env = s.server_env()
        self.assertNotIn("PT_SUBTITLE_COLOR", env)
        self.assertEqual(env["PT_COMPOSITE_BG_RGB"], "00FF00")
        self.assertEqual(env["PT_ALPHA_STRIDE"], "3")
        self.assertEqual(env["PT_PASSTHROUGH_MAX_FPS"], "30")
        self.assertEqual(env["PT_DECODE_MAX_SIDE"], "4096")

    def test_server_env_contains_video_dirs(self) -> None:
        s = self._settings()
        s.set_video_dirs([r"D:\VR", r"E:\VR"])
        env = s.server_env()

        self.assertEqual(env["PT_VIDEO_DIR"], r"D:\VR|E:\VR")
        self.assertNotIn("PT_DEBUG_LOGS", env)

    def test_server_env_keeps_zero_decode_max_side(self) -> None:
        s = self._settings()
        s.data["decode_max_side"] = 0
        env = s.server_env()
        self.assertEqual(env["PT_DECODE_MAX_SIDE"], "0")

    def test_restore_default_subtitle_style(self) -> None:
        s = self._settings()
        s.data["subtitle_yaw"] = 22
        s.data["subtitle_pitch"] = -10
        s.data["subtitle_color"] = "FFFFFF"
        s.restore_default_subtitle_style()
        self.assertEqual(s.data["subtitle_yaw"], 0.0)
        self.assertEqual(s.data["subtitle_pitch"], 0.0)
        self.assertEqual(s.data["subtitle_fov"], 60.0)
        self.assertEqual(s.data["subtitle_direction"], "horizontal_bottom")
        self.assertEqual(s.data["subtitle_color"], "")


if __name__ == "__main__":
    unittest.main()
