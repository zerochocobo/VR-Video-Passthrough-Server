"""Tests for predict_warmup_state and the structured startup status.

predict_warmup_state must not raise even when cupy/onnxruntime are missing
(it powers the startup overlay's pre-warmup hint), and must reliably detect
the Blackwell sm_120 "known slow" combination from the marker JSON.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import gpu_runtime_cache as grc
from utils.startup_status import (
    get_startup_state,
    reset_startup_progress,
    set_startup_phase,
)


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


if __name__ == "__main__":
    unittest.main()
