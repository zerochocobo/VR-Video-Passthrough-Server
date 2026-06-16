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
from PySide6.QtWidgets import QApplication

from ui.i18n import I18n
from ui.pages.offline_page import OfflinePage, _parse_hhmmss_text, _parse_time_text, _resolve_time_range, _resolve_time_segments


class _FakeSettings:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def save(self) -> None:
        pass

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


class OfflinePageTimeRangeTests(unittest.TestCase):
    def test_parse_time_text_accepts_hhmmss_mmss_and_seconds(self) -> None:
        self.assertEqual(_parse_time_text("01:02:03"), 3723.0)
        self.assertEqual(_parse_time_text("02:03"), 123.0)
        self.assertEqual(_parse_time_text("12.5"), 12.5)

    def test_parse_time_text_rejects_invalid_formats(self) -> None:
        for value in ("", "1:60:00", "1::00", "-1", "abc", "1:2:3:4"):
            with self.subTest(value=value):
                self.assertIsNone(_parse_time_text(value))

    def test_parse_hhmmss_text_requires_three_fields(self) -> None:
        self.assertEqual(_parse_hhmmss_text("01:02:03"), 3723.0)
        self.assertIsNone(_parse_hhmmss_text("02:03"))
        self.assertIsNone(_parse_hhmmss_text("1:60:00"))

    def test_custom_end_time_resolves_to_duration(self) -> None:
        start, duration, error = _resolve_time_range(
            "00:01:00",
            "custom_end",
            "5",
            "00:02:30",
            600.0,
        )

        self.assertEqual(error, "")
        self.assertEqual(start, 60.0)
        self.assertEqual(duration, 90.0)

    def test_custom_end_time_must_be_after_start_and_inside_video(self) -> None:
        self.assertEqual(
            _resolve_time_range("00:02:00", "custom_end", "5", "00:02:00", 600.0)[2],
            "offline.time_error_end_before_start",
        )
        self.assertEqual(
            _resolve_time_range("00:02:00", "custom_end", "5", "00:11:00", 600.0)[2],
            "offline.time_error_end_after_video",
        )

    def test_custom_minutes_and_fixed_duration_cannot_overrun_video(self) -> None:
        self.assertEqual(
            _resolve_time_range("00:09:30", "custom", "1", "00:00:00", 600.0)[2],
            "offline.time_error_clip_after_video",
        )
        self.assertEqual(
            _resolve_time_range("00:09:50", 15.0, "5", "00:00:00", 600.0)[2],
            "offline.time_error_clip_after_video",
        )

    def test_time_segments_require_order_and_video_bounds(self) -> None:
        segments, error, row = _resolve_time_segments([(0.0, 60.0), (60.0, 90.0)], 120.0)
        self.assertEqual(error, "")
        self.assertEqual(row, 0)
        self.assertEqual(segments, [(0.0, 60.0), (60.0, 90.0)])
        self.assertEqual(_resolve_time_segments([(30.0, 30.0)], 120.0)[1], "offline.time_error_segment_order")
        self.assertEqual(_resolve_time_segments([(0.0, 90.0), (80.0, 100.0)], 120.0)[1], "offline.time_error_segment_overlap")
        self.assertEqual(_resolve_time_segments([(0.0, 130.0)], 120.0)[1], "offline.time_error_segment_end_after_video")

    def test_run_single_converts_custom_end_time_to_duration(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.mp4"
            src.write_bytes(b"video")
            process = _FakeProcess()
            with patch("ui.pages.offline_page.cache_status", return_value="missing"), patch(
                "ui.pages.offline_page.probe_video_metadata",
                return_value=SimpleNamespace(timing=SimpleNamespace(duration=600.0)),
            ):
                page = OfflinePage(I18n("en_US"), _FakeSettings(), process)
                try:
                    page.single_video.setText(str(src))
                    page.single_duration.setCurrentIndex(page.single_duration.findData("custom_end"))
                    page.single_start.setText("00:01:00")
                    page.single_custom_end.setText("00:02:30")

                    page.run_single()

                    self.assertIsNotNone(process.started_args)
                    args = process.started_args or []
                    self.assertEqual(args[args.index("--start") + 1], "60.0")
                    self.assertEqual(args[args.index("--duration") + 1], "90.0")
                finally:
                    page.close()
                    app.processEvents()

    def test_run_single_uses_configured_time_segments(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.mp4"
            src.write_bytes(b"video")
            process = _FakeProcess()
            with patch("ui.pages.offline_page.cache_status", return_value="missing"), patch(
                "ui.pages.offline_page.probe_video_metadata",
                return_value=SimpleNamespace(timing=SimpleNamespace(duration=600.0)),
            ):
                page = OfflinePage(I18n("en_US"), _FakeSettings(), process)
                try:
                    page.single_video.setText(str(src))
                    page.single_time_mode.setCurrentIndex(page.single_time_mode.findData("segments"))
                    page.single_time_segments = [(0.0, 15.0), (60.0, 90.0)]

                    page.run_single()

                    self.assertIsNotNone(process.started_args)
                    args = process.started_args or []
                    self.assertNotIn("--start", args)
                    self.assertNotIn("--duration", args)
                    first = args.index("--segment")
                    self.assertEqual(args[first + 1], "00:00:00-00:00:15")
                    second = args.index("--segment", first + 2)
                    self.assertEqual(args[second + 1], "00:01:00-00:01:30")
                finally:
                    page.close()
                    app.processEvents()

    def test_run_single_passes_selected_birefnet_prepass(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.mp4"
            src.write_bytes(b"video")
            process = _FakeProcess()
            with patch("ui.pages.offline_page.cache_status", return_value="missing"), patch(
                "ui.pages.offline_page.probe_video_metadata",
                return_value=SimpleNamespace(timing=SimpleNamespace(duration=600.0)),
            ):
                page = OfflinePage(I18n("en_US"), _FakeSettings(), process)
                try:
                    page.single_video.setText(str(src))
                    page.single_engine.setCurrentIndex(page.single_engine.findData("matanyone2"))
                    page.single_recognition.setCurrentIndex(page.single_recognition.findData("yolo26m_birefnet"))

                    page.run_single()

                    self.assertIsNotNone(process.started_args)
                    args = process.started_args or []
                    self.assertEqual(args[args.index("--engine") + 1], "matanyone2_medium")
                    self.assertEqual(args[args.index("--matanyone2-prepass") + 1], "yolo26m_birefnet")
                finally:
                    page.close()
                    app.processEvents()


if __name__ == "__main__":
    unittest.main()
