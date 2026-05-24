from __future__ import annotations

import os
import site
import tempfile
import unittest
from pathlib import Path
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


class FakeI18n:
    def t(self, key: str) -> str:
        values = {
            "trt.title": "TensorRT Acceleration",
            "trt.model": "Model",
            "trt.precision": "TRT precision",
            "trt.gpu": "GPU",
            "trt.driver": "Driver",
            "trt.tensorrt": "TensorRT",
            "trt.cache_path": "Cache path",
            "trt.cache_status": "Cache status",
            "trt.status_missing": "Missing",
            "trt.fps_hint": "TensorRT acceleration can significantly improve FPS for realtime playback and offline generation.",
            "trt.description": "RVM description",
            "trt.description_matanyone2": "MatAnyone2 description",
            "trt.warning": "Warning",
            "trt.auto_download": "Auto download",
            "trt.manual_download": "Manual download",
            "trt.delete_cache": "Delete cache",
            "button.close": "Close",
            "trt.start_build": "Start build",
            "button.cancel": "Cancel",
            "trt.building_model": "Building {model}",
            "trt.engines_built": "Engines built: {count}",
            "trt.source_model_missing": "Source ONNX model does not exist: {path}",
            "trt.build_failed": "TensorRT build failed: {error}",
        }
        return values.get(key, key)


class TensorRTCacheDialogTests(unittest.TestCase):
    def _app(self):
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    def test_download_progress_signal_accepts_large_wheel_size(self) -> None:
        from ui.widgets.trt_cache_dialog import _TensorRTDownloadSignals
        from utils.tensorrt_runtime_libs import TENSORRT_CU12_LIBS_WHL_SIZE_BYTES

        self._app()
        signals = _TensorRTDownloadSignals()
        captured: list[tuple[int, int]] = []
        signals.progress.connect(lambda received, total: captured.append((int(received), int(total))))

        signals.progress.emit(0, TENSORRT_CU12_LIBS_WHL_SIZE_BYTES)

        self.assertEqual(captured, [(0, TENSORRT_CU12_LIBS_WHL_SIZE_BYTES)])

    def test_fps_hint_is_shown_for_realtime_and_offline_dialogs(self) -> None:
        from ui.widgets import trt_cache_dialog
        from utils.tensorrt_runtime_libs import TensorRTRuntimeLibStatus

        self._app()
        with tempfile.TemporaryDirectory() as raw:
            cache_root = Path(raw)
            runtime_status = TensorRTRuntimeLibStatus(frozen=False, lib_dir=cache_root)
            with (
                patch.object(trt_cache_dialog, "check_tensorrt_runtime_libs", return_value=runtime_status),
                patch.object(trt_cache_dialog, "cache_status", return_value="missing"),
                patch.object(trt_cache_dialog, "load_manifest_for_model", return_value={}),
                patch.object(trt_cache_dialog, "collect_fingerprint", return_value={}),
                patch.object(trt_cache_dialog, "manifest_path", side_effect=lambda model_key=None: cache_root / str(model_key or "rvm") / "manifest.json"),
                patch.object(trt_cache_dialog, "source_model_path", side_effect=lambda model_key=None: Path(f"{model_key or 'rvm'}.onnx")),
            ):
                realtime_dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key=None)
                offline_dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="matanyone2")

            try:
                self.assertIn("realtime playback", realtime_dialog.fps_hint_label.text())
                self.assertIn("offline generation", offline_dialog.fps_hint_label.text())
            finally:
                realtime_dialog.close()
                offline_dialog.close()

    def test_missing_source_model_does_not_start_build_process(self) -> None:
        from ui.widgets import trt_cache_dialog
        from utils.tensorrt_runtime_libs import TensorRTRuntimeLibStatus

        self._app()
        with tempfile.TemporaryDirectory() as raw:
            cache_root = Path(raw)
            missing_model = cache_root / "missing.onnx"
            runtime_status = TensorRTRuntimeLibStatus(frozen=False, lib_dir=cache_root)
            with (
                patch.object(trt_cache_dialog, "check_tensorrt_runtime_libs", return_value=runtime_status),
                patch.object(trt_cache_dialog, "cache_status", return_value="missing"),
                patch.object(trt_cache_dialog, "load_manifest_for_model", return_value={}),
                patch.object(trt_cache_dialog, "collect_fingerprint", return_value={}),
                patch.object(trt_cache_dialog, "manifest_path", side_effect=lambda model_key=None: cache_root / str(model_key or "rvm") / "manifest.json"),
                patch.object(trt_cache_dialog, "source_model_path", return_value=missing_model),
            ):
                dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="rvm")
                try:
                    with patch.object(trt_cache_dialog, "HiddenProcess") as hidden_process:
                        dialog._start_build()
                    hidden_process.assert_not_called()
                    self.assertIsNone(dialog.process)
                    self.assertIn("Source ONNX model does not exist", dialog.stage_label.text())
                    self.assertIn(str(missing_model), dialog.stage_label.text())
                finally:
                    dialog.close()

    def test_build_error_survives_finish_refresh(self) -> None:
        from ui.widgets import trt_cache_dialog
        from utils.tensorrt_runtime_libs import TensorRTRuntimeLibStatus

        self._app()
        with tempfile.TemporaryDirectory() as raw:
            cache_root = Path(raw)
            model_path = cache_root / "model.onnx"
            model_path.write_bytes(b"onnx")
            runtime_status = TensorRTRuntimeLibStatus(frozen=False, lib_dir=cache_root)
            with (
                patch.object(trt_cache_dialog, "check_tensorrt_runtime_libs", return_value=runtime_status),
                patch.object(trt_cache_dialog, "cache_status", return_value="missing"),
                patch.object(trt_cache_dialog, "load_manifest_for_model", return_value={}),
                patch.object(trt_cache_dialog, "collect_fingerprint", return_value={}),
                patch.object(trt_cache_dialog, "manifest_path", side_effect=lambda model_key=None: cache_root / str(model_key or "rvm") / "manifest.json"),
                patch.object(trt_cache_dialog, "source_model_path", return_value=model_path),
            ):
                dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="rvm")
                try:
                    dialog._read_process_output("ERROR:MatAnyone2 TensorRT source model not found\n")
                    dialog._build_finished(1)
                    self.assertEqual(dialog.stage_label.text(), "ERROR:MatAnyone2 TensorRT source model not found")
                    self.assertFalse(dialog.build_button.isHidden())
                    self.assertFalse(dialog.close_button.isHidden())
                    self.assertTrue(dialog.cancel_button.isHidden())
                finally:
                    dialog.close()


if __name__ == "__main__":
    unittest.main()
