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
        meta_patcher = patch.object(settings_module, "SETTINGS_META_PATH", root / "ui_settings_meta.json")
        meta_patcher.start()
        self.addCleanup(meta_patcher.stop)
        return settings_module.Settings()

    def test_passthrough_mode_mapping(self) -> None:
        s = self._settings()
        self.assertEqual(s.passthrough_mode(), "all")
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
        self.assertEqual(env["PT_ALPHA_STRIDE"], "1")
        self.assertEqual(env["PT_PASSTHROUGH_MAX_FPS"], "30")
        self.assertEqual(env["PT_PASSTHROUGH_PRODUCER_REALTIME_PACING"], "1")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_ENABLED"], "0")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_DLNA"], "0")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_ROUTE_POLICY"], "profile")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_CONTAINER"], "mpegts")
        self.assertEqual(env["PT_DLNA_IMAGE_ENABLED"], "0")
        self.assertEqual(env["PT_DECODE_MAX_SIDE"], "4096")
        self.assertEqual(env["PT_LIGHT_MATCH_PRESET"], "daylight")

    def test_server_env_can_enable_seekable_passthrough_for_ui_start(self) -> None:
        s = self._settings()
        s.data["passthrough_seek_enabled"] = True
        s.data["passthrough_seek_dlna"] = True
        s.data["passthrough_seek_route_policy"] = "all"
        s.data["passthrough_seek_container"] = "mp4"

        env = s.server_env()

        self.assertEqual(env["PT_PASSTHROUGH_SEEK_ENABLED"], "1")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_DLNA"], "1")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_ROUTE_POLICY"], "all")
        self.assertEqual(env["PT_PASSTHROUGH_SEEK_CONTAINER"], "mp4")

    def test_legacy_seekable_passthrough_dlna_migrates_off(self) -> None:
        root = Path("runtime_cache/test_ui_settings_seek_dlna_migration")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text(
            '{"passthrough_seek_enabled": true, "passthrough_seek_dlna": true}',
            encoding="utf-8",
        )

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()

        self.assertTrue(s.data["passthrough_seek_enabled"])
        self.assertFalse(s.data["passthrough_seek_dlna"])
        self.assertEqual(s.server_env()["PT_PASSTHROUGH_SEEK_ENABLED"], "1")
        self.assertEqual(s.server_env()["PT_PASSTHROUGH_SEEK_DLNA"], "0")

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

    def test_server_env_enables_tensorrt_only_when_cache_ready(self) -> None:
        s = self._settings()
        s.data["inference_backend"] = "tensorrt"
        with patch.object(settings_module, "cache_status", return_value="missing"):
            self.assertEqual(s.server_env()["PT_ONNX_PROVIDERS"], "CUDAExecutionProvider,CPUExecutionProvider")
        with patch.object(settings_module, "cache_status", return_value="ready"):
            self.assertEqual(
                s.server_env()["PT_ONNX_PROVIDERS"],
                "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
            )

    def test_server_env_disables_tensorrt_explicitly(self) -> None:
        s = self._settings()
        s.data["inference_backend"] = "cuda"
        env = s.server_env()
        self.assertEqual(env["PT_ONNX_PROVIDERS"], "CUDAExecutionProvider,CPUExecutionProvider")

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

    def test_internal_migration_flags_are_not_user_settings(self) -> None:
        root = Path("runtime_cache/test_ui_settings_migrations")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text(
            '{"light_match_enabled": true, "defaults_migrated_20260517_fps_size": true}',
            encoding="utf-8",
        )

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertNotIn("defaults_migrated_20260517_fps_size", s.data)
            self.assertNotIn("defaults_migrated_20260519_light_match_off", s.data)
            self.assertFalse(s.data["light_match_enabled"])

            s.save()
            saved = settings_path.read_text(encoding="utf-8")
            self.assertNotIn("defaults_migrated_", saved)
            self.assertTrue(meta_path.exists())

    def test_light_match_default_preset_is_daylight(self) -> None:
        s = self._settings()
        self.assertEqual(s.data["light_match_preset"], "daylight")
        self.assertEqual(s.data["light_match_temp_k"], 6500)
        self.assertEqual(s.server_env()["PT_LIGHT_MATCH_PRESET"], "daylight")
        self.assertEqual(s.server_env()["PT_LIGHT_MATCH_TEMP_K"], "6500")

    def test_legacy_disabled_custom_light_match_migrates_to_daylight(self) -> None:
        root = Path("runtime_cache/test_ui_settings_light_match_default")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text('{"light_match_enabled": false, "light_match_preset": "custom"}', encoding="utf-8")

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["light_match_preset"], "daylight")
            self.assertEqual(s.data["light_match_temp_k"], 6500)

    def test_legacy_daylight_light_match_recalibrates_to_d65(self) -> None:
        root = Path("runtime_cache/test_ui_settings_light_match_recalibration")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text(
            '{"light_match_enabled": true, "light_match_preset": "daylight", "light_match_temp_k": 5500}',
            encoding="utf-8",
        )

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["light_match_preset"], "daylight")
            self.assertEqual(s.data["light_match_temp_k"], 6500)
            self.assertEqual(s.data["light_match_saturation"], 1.0)

    def test_legacy_night_cool_light_match_recalibrates_to_8000k(self) -> None:
        root = Path("runtime_cache/test_ui_settings_light_match_night_recalibration")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text(
            '{"light_match_enabled": true, "light_match_preset": "night_cool", "light_match_temp_k": 6500, "light_match_exposure_ev": -0.1, "light_match_saturation": 0.95}',
            encoding="utf-8",
        )

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["light_match_preset"], "night_cool")
            self.assertEqual(s.data["light_match_temp_k"], 8000)
            self.assertEqual(s.data["light_match_exposure_ev"], 0.0)
            self.assertEqual(s.data["light_match_saturation"], 1.0)

    def test_custom_light_match_survives_recalibration_migration(self) -> None:
        root = Path("runtime_cache/test_ui_settings_light_match_custom_recalibration")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text(
            '{"light_match_enabled": true, "light_match_preset": "custom", "light_match_temp_k": 5400}',
            encoding="utf-8",
        )

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["light_match_preset"], "custom")
            self.assertEqual(s.data["light_match_temp_k"], 5400)

    def test_legacy_30fps_default_stays_default(self) -> None:
        root = Path("runtime_cache/test_ui_settings_fps_migration")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text('{"passthrough_max_fps": 30}', encoding="utf-8")

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["passthrough_max_fps"], 30)

    def test_legacy_zero_fps_default_migrates_to_30fps_once(self) -> None:
        root = Path("runtime_cache/test_ui_settings_fps_zero_migration")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text('{"passthrough_max_fps": 0}', encoding="utf-8")

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["passthrough_max_fps"], 30)
            s.data["passthrough_max_fps"] = 0
            s.save()
            reloaded = settings_module.Settings()
            self.assertEqual(reloaded.data["passthrough_max_fps"], 0)

    def test_explicit_non_default_fps_survives_migration(self) -> None:
        root = Path("runtime_cache/test_ui_settings_fps_custom")
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        settings_path = root / "ui_settings.json"
        meta_path = root / "ui_settings_meta.json"
        settings_path.write_text('{"passthrough_max_fps": 24}', encoding="utf-8")

        with (
            patch.object(settings_module, "SETTINGS_PATH", settings_path),
            patch.object(settings_module, "SETTINGS_META_PATH", meta_path),
        ):
            s = settings_module.Settings()
            self.assertEqual(s.data["passthrough_max_fps"], 24)


if __name__ == "__main__":
    unittest.main()
