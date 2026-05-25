"""Tests for predict_warmup_state and the structured startup status.

predict_warmup_state must not raise even when cupy/onnxruntime are missing
(it powers the startup overlay's pre-warmup hint), and must reliably detect
the Blackwell sm_120 "known slow" combination from the marker JSON.
"""
from __future__ import annotations

import json
import logging
import unittest
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from utils import gpu_runtime_cache as grc
from utils.startup_status import (
    get_startup_state,
    reset_startup_progress,
    set_startup_phase,
)
from utils.logger import warmup_event


SAMPLE_KEY = {
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "compute_capability": "12.0",
    "driver_version": "560.94",
    "python_version": "3.12.7",
    "onnxruntime_version": "1.20.0",
    "onnxruntime_providers_cuda_dll_hash": "deadbeefdeadbeef",
    "cupy_version": "13.3.0",
    "cupy_cuda_runtime": "12060",
    "model_name": "rvm_mobilenetv3_fp32.onnx",
    "model_sha256_16": "0123456789abcdef",
    "input_size": 1024,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "shapes": [[1, 3, 1024, 1024], [2, 3, 1024, 1024]],
}


def _write_marker(path: Path, key_dict: dict, elapsed: float = 120.0) -> None:
    payload = {
        "key": key_dict,
        "cuda_cache_path": "C:/tmp/cuda",
        "cupy_cache_dir": "C:/tmp/cupy",
        "cache_size_after_warmup": 1024,
        "cache_file_count_after_warmup": 1,
        "elapsed_sec": elapsed,
        "verified_second_pass_sec": 5.0,
        "created_at": "2026-05-12T00:00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class KnownSlowDetectionTests(unittest.TestCase):
    def test_sm_120_with_old_ort_is_known_slow(self) -> None:
        self.assertTrue(grc._is_known_slow_combo("12.0", "1.20.0"))
        self.assertTrue(grc._is_known_slow_combo("12.0", "1.21.5"))

    def test_sm_120_with_future_ort_is_not_known_slow(self) -> None:
        self.assertFalse(grc._is_known_slow_combo("12.0", "1.22.0"))
        self.assertFalse(grc._is_known_slow_combo("12.5", "1.23.0"))

    def test_pre_blackwell_is_not_known_slow(self) -> None:
        self.assertFalse(grc._is_known_slow_combo("8.9", "1.20.0"))
        self.assertFalse(grc._is_known_slow_combo("7.5", "1.10.0"))

    def test_empty_capability_is_not_known_slow(self) -> None:
        self.assertFalse(grc._is_known_slow_combo("", "1.20.0"))
        self.assertFalse(grc._is_known_slow_combo("?.?", "1.20.0"))


class PredictWarmupStateTests(unittest.TestCase):
    def test_no_marker_returns_cold(self) -> None:
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "marker.json"
            report = grc.predict_warmup_state(marker_path=marker)
        self.assertTrue(report.cold)
        self.assertIn(report.reason, {"marker_missing", "inspect_failed"})
        self.assertFalse(report.marker_exists)
        # Estimate must always be positive so the overlay shows something.
        self.assertGreater(report.estimate_sec, 0.0)

    def test_marker_present_does_not_raise(self) -> None:
        # Even if the running environment doesn't match the marker, the
        # function must still return a ColdStartReport without raising.
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "marker.json"
            _write_marker(marker, SAMPLE_KEY)
            report = grc.predict_warmup_state(marker_path=marker)
        self.assertIsInstance(report, grc.ColdStartReport)
        self.assertTrue(report.marker_exists)
        self.assertGreater(report.previous_elapsed_sec, 0.0)

    def test_inspect_failure_returns_safe_report(self) -> None:
        # Without cupy installed on the build machine, predict_warmup_state
        # should still return a report rather than raising.
        report = grc.predict_warmup_state()
        self.assertIsInstance(report, grc.ColdStartReport)
        self.assertIsInstance(report.estimate_sec, float)
        self.assertIsInstance(report.is_known_slow, bool)


class CompositeWarmupTests(unittest.TestCase):
    def test_resident_warmup_runs_composite_and_alpha_pack_paths(self) -> None:
        key = grc.GpuWarmupKey(
            gpu_name="GPU",
            compute_capability="8.9",
            driver_version="1",
            python_version="3",
            onnxruntime_version="1",
            onnxruntime_providers_cuda_dll_hash="x",
            cupy_version="1",
            cupy_cuda_runtime="1",
            model_name="model.onnx",
            model_sha256_16="abc",
            input_size=1024,
            providers=["CUDAExecutionProvider"],
            shapes=[(1, 3, 32, 32)],
        )

        fake_matter = MagicMock()
        fake_matter.input_dtype = "float32"
        fake_matter.sess.get_providers.return_value = ["CUDAExecutionProvider"]
        fake_matter._get_trt_static_session.return_value = object()
        fake_matter.acquire_nv12_output_slot.return_value = MagicMock(buffer="slot-buffer")

        fake_cp = MagicMock()
        fake_cp.float32 = "float32"
        fake_cp.zeros.return_value = "zeros"
        fake_cp.cuda.Stream.null.synchronize = MagicMock()

        fake_matting_mod = MagicMock()
        fake_matting_mod._CUDA_STREAM = None
        fake_matting_mod.make_zero_gpu_frame.return_value = MagicMock()

        fake_packer = MagicMock()
        fake_alpha_packer_cls = MagicMock(return_value=fake_packer)

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cupy":
                return fake_cp
            if name == "pipeline.matting":
                return fake_matting_mod
            if name == "pipeline.alpha_packer":
                module = MagicMock()
                module.AlphaPacker = fake_alpha_packer_cls
                return module
            return real_import(name, globals, locals, fromlist, level)

        fake_matting_mod.get_matter.return_value = fake_matter
        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch.object(
                grc,
                "configure_gpu_runtime_cache",
                return_value=grc.GpuRuntimeCacheEnv(
                    runtime_cache_dir=".",
                    cuda_cache_path=".",
                    cupy_cache_dir=".",
                    ort_cache_dir=".",
                    marker_path="marker.json",
                ),
            ),
            patch.object(grc, "build_warmup_key", return_value=key),
            patch.object(grc, "marker_matches", return_value=True),
            patch.object(grc, "_cache_stats", return_value=(0, 0)),
            patch.object(grc.config, "WARMUP_COMPOSITE_ENABLE", True),
            patch.object(grc.config, "WARMUP_COMPOSITE_GEOMETRIES", [(64, 128)]),
            patch.object(grc.config, "MATTING_INPUT_SIZE", 32),
        ):
            grc.warmup_gpu_runtime_cache(runs_per_shape=1)

        fake_matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile.assert_called_once()
        fake_matter.composite_green_gpu_p016_frame_to_gpu_nv12_profile.assert_called_once()
        fake_packer.pack_uploaded.assert_called_once()


class StartupStatusTests(unittest.TestCase):
    def test_only_true_terminal_startup_phases_stop_polling(self) -> None:
        poller_source = Path("ui/services/startup_status_poller.py").read_text(encoding="utf-8")
        namespace: dict[str, object] = {}
        terminal_line = next(
            line for line in poller_source.splitlines() if line.startswith("TERMINAL_PHASES = ")
        )
        exec(terminal_line, namespace)
        terminal_phases = namespace["TERMINAL_PHASES"]

        self.assertIn("listening", terminal_phases)
        self.assertIn("failed", terminal_phases)
        self.assertNotIn("warmed", terminal_phases)
        self.assertNotIn("http_starting", terminal_phases)

    def test_set_startup_phase_accepts_structured_kwargs(self) -> None:
        set_startup_phase(
            "warming",
            "GPU init",
            step="ort_session",
            step_index=1,
            step_total=4,
            progress=0.25,
            eta_sec=42.0,
            elapsed_sec=2.0,
            cold=True,
            is_known_slow=True,
            gpu_name="RTX 5090",
            compute_capability="12.0",
            onnxruntime_version="1.21.0",
        )
        state = get_startup_state()
        self.assertEqual(state["phase"], "warming")
        self.assertEqual(state["message"], "GPU init")
        self.assertEqual(state["step"], "ort_session")
        self.assertAlmostEqual(state["progress"], 0.25)
        self.assertTrue(state["cold"])
        self.assertTrue(state["is_known_slow"])
        self.assertEqual(state["gpu_name"], "RTX 5090")

    def test_reset_clears_progress_fields(self) -> None:
        set_startup_phase(
            "warming",
            step="x",
            step_index=2,
            progress=0.5,
            eta_sec=10.0,
            elapsed_sec=5.0,
        )
        reset_startup_progress()
        state = get_startup_state()
        self.assertEqual(state["step"], "")
        self.assertEqual(state["step_index"], 0)
        self.assertAlmostEqual(state["progress"], 0.0)
        self.assertAlmostEqual(state["eta_sec"], 0.0)

    def test_monotonic_progress_flag_does_not_lower_same_phase(self) -> None:
        set_startup_phase("warming", "first", progress=0.6)
        set_startup_phase("warming", "older substep", progress=0.4, monotonic_progress=True)
        state = get_startup_state()
        self.assertAlmostEqual(state["progress"], 0.6)

        set_startup_phase("warming", "newer substep", progress=0.7, monotonic_progress=True)
        state = get_startup_state()
        self.assertAlmostEqual(state["progress"], 0.7)

        set_startup_phase("failed", "new phase may reset", progress=0.0, monotonic_progress=True)
        state = get_startup_state()
        self.assertAlmostEqual(state["progress"], 0.0)


class WarmupLoggingTests(unittest.TestCase):
    def test_warmup_event_writes_json_payload(self) -> None:
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("test_warmup_event")
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            warmup_event(logger, phase="static_trt_preload", batch=1, shape=[1024, 1024], loaded=True)
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = True

        line = stream.getvalue().strip()
        self.assertTrue(line.startswith("[WARMUP] "))
        payload = json.loads(line[len("[WARMUP] ") :])
        self.assertEqual(payload["phase"], "static_trt_preload")
        self.assertEqual(payload["shape"], [1024, 1024])
        self.assertTrue(payload["loaded"])

    def test_startup_steps_include_track_a_a5_phases(self) -> None:
        overlay_source = Path("ui/widgets/startup_overlay.py").read_text(encoding="utf-8")
        for step in (
            "matter_singleton",
            "static_trt_preload",
            "ort_iobinding_runs",
            "composite_jit",
            "reset_state",
            "nvenc_preflight",
            "firewall",
            "ssdp",
            "http_starting",
            "listening",
        ):
            self.assertIn(step, overlay_source)


if __name__ == "__main__":
    unittest.main()
