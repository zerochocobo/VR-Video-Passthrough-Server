from __future__ import annotations

import os
import site
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_DLL_HANDLES = []
if hasattr(os, "add_dll_directory"):
    for site_dir in site.getsitepackages():
        base = Path(site_dir)
        for dll_dir in (base / "PySide6", base / "shiboken6"):
            if dll_dir.exists():
                _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        plugins = base / "PySide6" / "plugins"
        platforms = plugins / "platforms"
        if platforms.exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms))
        if plugins.exists():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QComboBox

from ui.i18n import I18n
from ui.pages.home_page import HomePage
from ui.pages.two_dvr_page import TwoDvrPage


class _FakeSettings:
    def __init__(self) -> None:
        self.data: dict[str, object] = {
            "mode_green": True,
            "mode_alpha": True,
            "mode_two_dvr": True,
            "background_color": "00FF00",
            "subtitle_enable": True,
            "quality_speed": "ultrafast",
            "passthrough_max_fps": 30,
            "decode_max_side": 4096,
            "inference_backend": "cuda",
            "two_dvr_live_model": "base",
            "two_dvr_live_hole_fill": "inverse_warp",
            "two_dvr_live_eye_distance": 65.0,
            "two_dvr_live_strength": 1.0,
        }
        self.save_count = 0

    def save(self) -> None:
        self.save_count += 1

    def video_dirs(self) -> list[str]:
        return []

    def server_env(self) -> dict[str, str]:
        return {}


class _FakeProcess(QObject):
    output = Signal(str)
    state_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.started_args: list[str] | None = None
        self.started_env: dict[str, str] | None = None

    def start(self, args: list[str], env: dict[str, str]) -> None:
        self.started_args = list(args)
        self.started_env = dict(env)

    def stop(self) -> None:
        pass


class TwoDvrUiHoleFillTests(unittest.TestCase):
    def test_strength_maps_to_effective_eye_distance_with_limits(self) -> None:
        from offline.two_dvr_render import effective_eye_distance_mm, strength_multiplier

        self.assertEqual(effective_eye_distance_mm(65.0, 1.5), 97.5)
        self.assertEqual(strength_multiplier(0.01), 0.1)
        self.assertEqual(strength_multiplier(9.0), 3.0)

    def test_server_env_emits_only_realtime_strength_from_ui_settings(self) -> None:
        from ui.settings import DEFAULTS, Settings

        settings = object.__new__(Settings)
        settings.data = dict(DEFAULTS)
        settings.data.update({
            "inference_backend": "cuda",
            "two_dvr_live_model": "base_hd",
            "two_dvr_live_hole_fill": "inverse_warp",
            "two_dvr_live_eye_distance": 80.0,
            "two_dvr_live_strength": 1.5,
        })

        env = Settings.server_env(settings)

        self.assertEqual(env["PT_TWO_DVR_MODEL"], "base")
        self.assertEqual(env["PT_TWO_DVR_HOLE_FILL"], "soft_shift")
        self.assertEqual(env["PT_TWO_DVR_EYE_DISTANCE_MM"], "65.0")
        self.assertEqual(env["PT_TWO_DVR_STRENGTH"], "1.5")

    def test_home_exposes_only_realtime_strength_and_saves_hidden_defaults(self) -> None:
        app = QApplication.instance() or QApplication([])
        settings = _FakeSettings()
        with patch.object(HomePage, "_update_trt_state", lambda self: None):
            page = HomePage(I18n("en_US"), settings)
            try:
                self.assertFalse(hasattr(page, "home_two_dvr_config_button"))
                self.assertTrue(hasattr(page, "home_two_dvr_strength"))
                for combo in page.findChildren(QComboBox):
                    values = {combo.itemData(index) for index in range(combo.count())}
                    self.assertNotIn("inverse_warp", values)
                    self.assertNotIn("small_hd", values)
                    self.assertNotIn("base_hd", values)

                index = page.home_two_dvr_strength.findData(1.5)
                self.assertGreaterEqual(index, 0)
                page.home_two_dvr_strength.setCurrentIndex(index)

                self.assertEqual(settings.data["two_dvr_live_model"], "base")
                self.assertEqual(settings.data["two_dvr_live_hole_fill"], "soft_shift")
                self.assertEqual(settings.data["two_dvr_live_eye_distance"], 65.0)
                self.assertEqual(settings.data["two_dvr_live_strength"], 1.5)
            finally:
                page.close()
            app.processEvents()

    def test_offline_page_hides_hole_fill_choice_and_runs_soft_shift(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.mp4"
            src.write_bytes(b"video")
            process = _FakeProcess()
            page = TwoDvrPage(I18n("en_US"), _FakeSettings(), process)
            try:
                self.assertFalse(hasattr(page, "single_hole_fill"))
                self.assertFalse(hasattr(page, "batch_hole_fill"))
                self.assertFalse(hasattr(page, "single_eye_distance"))
                self.assertFalse(hasattr(page, "batch_eye_distance"))
                self.assertFalse(hasattr(page, "single_quality"))
                self.assertFalse(hasattr(page, "batch_quality"))
                self.assertNotIn("hole_fill", page.single_labels)
                self.assertNotIn("hole_fill", page.batch_labels)
                self.assertNotIn("eye", page.single_labels)
                self.assertNotIn("eye", page.batch_labels)
                self.assertNotIn("quality", page.single_labels)
                self.assertNotIn("quality", page.batch_labels)
                self.assertIn("strength", page.single_labels)
                self.assertIn("strength", page.batch_labels)
                self.assertEqual(page.single_labels["performance"].text(), page.i18n.t("performance.quality_speed"))
                self.assertEqual(page.batch_labels["performance"].text(), page.i18n.t("performance.quality_speed"))
                self.assertEqual(page.single_quality_speed.currentData(), "medium")
                self.assertEqual(page.batch_quality_speed.currentData(), "medium")
                for combo in page.findChildren(QComboBox):
                    values = {combo.itemData(index) for index in range(combo.count())}
                    self.assertNotIn("inverse_warp", values)

                page.single_video.setText(str(src))
                with patch(
                    "ui.pages.two_dvr_page.probe_video_metadata",
                    return_value=SimpleNamespace(timing=SimpleNamespace(duration=600.0)),
                ):
                    page.run_single()

                self.assertIsNotNone(process.started_args)
                args = process.started_args or []
                self.assertEqual(args[args.index("--hole-fill") + 1], "soft_shift")
                self.assertEqual(args[args.index("--strength") + 1], "1.00")
                self.assertEqual(args[args.index("--max-side") + 1], "0")
                self.assertEqual(args[args.index("--preset") + 1], "p4")
                self.assertNotIn("--eye-distance", args)
                self.assertEqual(process.started_env, {"PT_PASSTHROUGH_PYNV_PRESET": "P4"})
            finally:
                page.close()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
