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

    def test_matanyone2_build_progress_counts_only_model_builds_and_caps_before_exit(self) -> None:
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
                patch.object(
                    trt_cache_dialog,
                    "manifest_path",
                    side_effect=lambda model_key=None, scope=None: cache_root / str(scope or model_key or "rvm") / "manifest.json",
                ),
            ):
                dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="matanyone2")
                try:
                    self.assertEqual(dialog._build_stage_count(), len(trt_cache_dialog.MATANYONE2_MODEL_KEYS))
                    dialog._read_process_output("STAGE:1:start:Building MatAnyone2 512\n")
                    self.assertEqual(dialog.progress.value(), 0)
                    dialog._read_process_output("STAGE:1:done:1\n")
                    self.assertEqual(dialog.progress.value(), 50)
                    dialog._read_process_output("STAGE:2:done:1\n")
                    self.assertEqual(dialog.progress.value(), 99)
                    dialog._read_process_output("STAGE:3:start:Verifying MatAnyone2 TensorRT cache\nSTAGE:4:done:0\n")
                    self.assertEqual(dialog.progress.value(), 99)
                finally:
                    dialog.close()

    def test_rvm_build_progress_caps_at_99_until_process_exit(self) -> None:
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
                patch.object(
                    trt_cache_dialog,
                    "manifest_path",
                    side_effect=lambda model_key=None, scope=None: cache_root / str(scope or model_key or "rvm") / "manifest.json",
                ),
                patch.object(trt_cache_dialog, "source_model_path", side_effect=lambda model_key=None: Path(f"{model_key or 'rvm'}.onnx")),
            ):
                realtime = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="rvm", scope="realtime")
                offline = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="rvm", scope="offline")
                try:
                    self.assertEqual(realtime._build_stage_count(), 3)
                    realtime._read_process_output("STAGE:3:done:1\n")
                    self.assertEqual(realtime.progress.value(), 99)

                    self.assertEqual(offline._build_stage_count(), 6)
                    offline._read_process_output("STAGE:6:done:1\n")
                    self.assertEqual(offline.progress.value(), 99)
                finally:
                    realtime.close()
                    offline.close()

    def test_realtime_rvm_build_uses_rvm_1024_warmup(self) -> None:
        from ui.widgets import trt_cache_dialog
        from utils.tensorrt_runtime_libs import TensorRTRuntimeLibStatus

        class _Signal:
            def connect(self, _callback):
                pass

        class _Process:
            stdout = _Signal()
            stderr = _Signal()
            finished = _Signal()

            def __init__(self, *_args, **_kwargs):
                pass

            def start(self, exe, args, env=None):
                captured["exe"] = exe
                captured["args"] = args
                captured["env"] = env or {}
                return True

            def kill(self):
                pass

        self._app()
        captured: dict[str, object] = {}
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
                patch.object(trt_cache_dialog, "trt_warmup_command", return_value=("python", ["-m", "ui.services.trt_warmup_process"])),
                patch.object(trt_cache_dialog, "HiddenProcess", _Process),
            ):
                dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="rvm", scope="realtime")
                try:
                    dialog._start_build()
                finally:
                    dialog.close()

        args = captured["args"]
        self.assertEqual(args[args.index("--model") + 1], "rvm")
        self.assertEqual(args[args.index("--input-size") + 1], "1024")
        self.assertEqual(args[args.index("--fp16") + 1], "0")

    def test_offline_rvm_build_uses_offline_warmup(self) -> None:
        from ui.widgets import trt_cache_dialog
        from utils.tensorrt_runtime_libs import TensorRTRuntimeLibStatus

        class _Signal:
            def connect(self, _callback):
                pass

        class _Process:
            stdout = _Signal()
            stderr = _Signal()
            finished = _Signal()

            def __init__(self, *_args, **_kwargs):
                pass

            def start(self, exe, args, env=None):
                captured["exe"] = exe
                captured["args"] = args
                return True

            def kill(self):
                pass

        self._app()
        captured: dict[str, object] = {}
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
                patch.object(trt_cache_dialog, "manifest_path", side_effect=lambda model_key=None, scope=None: cache_root / str(scope or model_key or "rvm") / "manifest.json"),
                patch.object(trt_cache_dialog, "source_model_path", return_value=model_path),
                patch.object(trt_cache_dialog, "offline_trt_warmup_command", return_value=("python", ["tools/warmup_offline_trt.py"])),
                patch.object(trt_cache_dialog, "HiddenProcess", _Process),
            ):
                dialog = trt_cache_dialog.TensorRTConfigDialog(FakeI18n(), model_key="rvm", scope="offline")
                try:
                    dialog._start_build()
                finally:
                    dialog.close()

        self.assertEqual(captured["args"], ["tools/warmup_offline_trt.py"])


if __name__ == "__main__":
    unittest.main()
