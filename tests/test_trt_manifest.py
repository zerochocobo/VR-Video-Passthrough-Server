from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from utils import trt_manifest


class TrtManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("runtime_cache/test_trt_manifest")
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        patcher = patch.object(config, "ONNX_TRT_ENGINE_CACHE_PATH", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fingerprint(self) -> dict:
        return {
            "gpu_uuid": "GPU-test",
            "gpu_name": "Test GPU",
            "driver_version": "1",
            "trt_model_key": "rvm",
            "cuda_runtime": "12.4",
            "trt_version": "10",
            "ort_version": "1.20",
            "model_sha256": "abc",
            "matting_input_size": 1024,
            "rvm_downsample_ratio": 0.5,
            "trt_fp16": True,
            "trt_cuda_graph": True,
        }

    def test_missing_manifest(self) -> None:
        self.assertEqual(trt_manifest.cache_status(actual_fp=self._fingerprint()), "missing")

    def test_ready_manifest_requires_engine_file(self) -> None:
        fp = self._fingerprint()
        manifest = trt_manifest.build_manifest(
            fp,
            [{"shape": "1x3x1024x1024", "size_mb": 1, "built_at": "2026-05-20T00:00:00Z"}],
            3,
        )
        trt_manifest.save_manifest(manifest)
        self.assertEqual(trt_manifest.cache_status(actual_fp=fp), "failed")
        trt_manifest.shape_inferred_model_path(cache_dir=self.root).write_bytes(b"onnx")
        (self.root / "rvm.engine").write_bytes(b"e" * (1024 * 1024))
        self.assertEqual(trt_manifest.cache_status(actual_fp=fp), "ready")

    def test_shape_inferred_model_and_tiny_engine_are_not_ready_cache(self) -> None:
        fp = self._fingerprint()
        manifest = trt_manifest.build_manifest(
            fp,
            [{"shape": "bad", "size_mb": 0, "built_at": "2026-05-20T00:00:00Z"}],
            3,
        )
        trt_manifest.save_manifest(manifest)
        trt_manifest.shape_inferred_model_path(cache_dir=self.root).write_bytes(b"onnx")
        (self.root / "failed.engine").write_bytes(b"small")
        self.assertEqual(trt_manifest.cache_status(actual_fp=fp), "failed")

    def test_stale_reasons(self) -> None:
        saved = self._fingerprint()
        actual = dict(saved)
        actual["driver_version"] = "2"
        self.assertEqual(trt_manifest.stale_reasons(saved, actual), ["driver_version: 1 -> 2"])
        manifest = trt_manifest.build_manifest(
            saved,
            [{"shape": "1x3x1024x1024", "size_mb": 1, "built_at": "2026-05-20T00:00:00Z"}],
            3,
        )
        trt_manifest.save_manifest(manifest)
        trt_manifest.shape_inferred_model_path(cache_dir=self.root).write_bytes(b"onnx")
        (self.root / "rvm.engine").write_bytes(b"e" * (1024 * 1024))
        self.assertEqual(trt_manifest.cache_status(actual_fp=actual), "stale")

    def test_failed_model_status(self) -> None:
        manifest = trt_manifest.build_manifest(self._fingerprint(), [], 0)
        manifest["models"][0]["status"] = "failed"
        trt_manifest.save_manifest(manifest)
        self.assertEqual(trt_manifest.cache_status(actual_fp=self._fingerprint()), "failed")

    def test_matanyone2_manifest_uses_separate_cache_dir(self) -> None:
        source = self.root / "matanyone2_step_update.onnx"
        source.write_bytes(b"onnx")
        fp = {
            "gpu_uuid": "GPU-test",
            "gpu_name": "Test GPU",
            "driver_version": "1",
            "trt_model_key": "matanyone2",
            "cuda_runtime": "12.4",
            "trt_version": "10",
            "ort_version": "1.20",
            "model_sha256": "def",
            "trt_fp16": True,
            "trt_cuda_graph": True,
            "matanyone2_model_key": "matanyone2_onnx_512_bs1",
            "matanyone2_onnx": "matanyone2_step_update.onnx",
        }
        cache_dir = trt_manifest.cache_dir_for_model(trt_manifest.TRT_MODEL_MATANYONE2)
        manifest = trt_manifest.build_manifest(
            fp,
            [{"shape": "matanyone2_step_update", "size_mb": 1, "built_at": "2026-05-20T00:00:00Z"}],
            3,
            model_key=trt_manifest.TRT_MODEL_MATANYONE2,
        )
        with patch.object(trt_manifest, "matanyone2_trt_source_model_path", return_value=source):
            trt_manifest.save_manifest(manifest, model_key=trt_manifest.TRT_MODEL_MATANYONE2)
            (cache_dir / "step.engine").write_bytes(b"e" * (1024 * 1024))
            self.assertEqual(
                trt_manifest.cache_status(actual_fp=fp, model_key=trt_manifest.TRT_MODEL_MATANYONE2),
                "ready",
            )
        self.assertEqual(trt_manifest.manifest_path(trt_manifest.TRT_MODEL_MATANYONE2).parent.name, "matanyone2_onnx_512_bs1")

    def test_nvidia_smi_fallback_collects_gpu_name_and_driver(self) -> None:
        completed = subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            stdout="GPU-abc, NVIDIA GeForce RTX 2080, 560.94\n",
            stderr="",
        )
        with patch.dict("sys.modules", {"pynvml": None}), patch.object(trt_manifest.subprocess, "run", return_value=completed):
            info = trt_manifest._nvml_info()
        self.assertEqual(info["gpu_uuid"], "GPU-abc")
        self.assertEqual(info["gpu_name"], "NVIDIA GeForce RTX 2080")
        self.assertEqual(info["driver_version"], "560.94")


if __name__ == "__main__":
    unittest.main()
