from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from utils import trt_manifest
from utils.rvm_static_onnx import static_rvm_model_path


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
            "trt_fp16": False,
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

    def test_rvm_ready_requires_all_offline_precision_tier_artifacts(self) -> None:
        fp = self._fingerprint()
        fp["offline_precision_tiers"] = True
        fp["offline_shapes"] = [
            {"input_size": input_size, "batch": batch, "downsample_ratio": downsample}
            for input_size, batch, downsample in trt_manifest.RVM_OFFLINE_TRT_SHAPES
        ]
        manifest = trt_manifest.build_manifest(
            fp,
            [{"shape": f"shape_{idx}", "size_mb": 1, "built_at": "2026-05-20T00:00:00Z"} for idx in range(6)],
            3,
        )
        offline_cache = trt_manifest.cache_dir_for_model(trt_manifest.TRT_MODEL_RVM, scope="offline")
        trt_manifest.save_manifest(manifest, scope="offline")
        trt_manifest.shape_inferred_model_path(cache_dir=offline_cache).write_bytes(b"onnx")
        source = trt_manifest.original_rvm_model_path()
        for input_size, batch, downsample in trt_manifest.RVM_OFFLINE_TRT_SHAPES[:-1]:
            static_rvm_model_path(source, offline_cache, batch, input_size, downsample).write_bytes(b"onnx")
        (offline_cache / "rvm.engine").write_bytes(b"e" * (1024 * 1024))
        self.assertEqual(trt_manifest.cache_status(actual_fp=self._fingerprint(), scope="offline"), "stale")

        input_size, batch, downsample = trt_manifest.RVM_OFFLINE_TRT_SHAPES[-1]
        static_rvm_model_path(source, offline_cache, batch, input_size, downsample).write_bytes(b"onnx")
        self.assertEqual(trt_manifest.cache_status(actual_fp=self._fingerprint(), scope="offline"), "ready")
        self.assertEqual(trt_manifest.cache_status(actual_fp=self._fingerprint()), "missing")

    def test_clearing_realtime_rvm_cache_preserves_offline_scope(self) -> None:
        (self.root / "manifest.json").write_text("{}", encoding="utf-8")
        (self.root / "runtime.engine").write_bytes(b"e" * (1024 * 1024))
        offline_cache = self.root / "offline"
        offline_cache.mkdir()
        (offline_cache / "manifest.json").write_text("{}", encoding="utf-8")
        matanyone_cache = self.root / trt_manifest.MATANYONE2_CACHE_KEY
        matanyone_cache.mkdir()
        (matanyone_cache / "manifest.json").write_text("{}", encoding="utf-8")

        trt_manifest.clear_cache(trt_manifest.TRT_MODEL_RVM)

        self.assertFalse((self.root / "manifest.json").exists())
        self.assertFalse((self.root / "runtime.engine").exists())
        self.assertTrue((offline_cache / "manifest.json").exists())
        self.assertTrue((matanyone_cache / "manifest.json").exists())

    def test_offline_scope_is_idempotent_when_base_is_already_offline(self) -> None:
        offline_base = self.root / "offline"
        with patch.object(config, "ONNX_TRT_ENGINE_CACHE_PATH", offline_base):
            self.assertEqual(
                trt_manifest.cache_dir_for_model(trt_manifest.TRT_MODEL_RVM, scope="offline"),
                offline_base.resolve(),
            )

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
        source_512 = self.root / "matanyone2_512_step_update.onnx"
        source_1024 = self.root / "matanyone2_1024_step_update.onnx"
        source_512.write_bytes(b"onnx512")
        source_1024.write_bytes(b"onnx1024")
        fp = {
            "gpu_uuid": "GPU-test",
            "gpu_name": "Test GPU",
            "driver_version": "1",
            "trt_model_key": "matanyone2",
            "cuda_runtime": "12.4",
            "trt_version": "10",
            "ort_version": "1.20",
            "model_sha256": "def",
            "trt_fp16": False,
            "trt_cuda_graph": True,
            "matanyone2_model_key": "matanyone2_onnx_1024_bs1",
            "matanyone2_model_keys": list(trt_manifest.MATANYONE2_MODEL_KEYS),
            "matanyone2_onnx": "matanyone2_step_update.onnx",
        }
        cache_dir = trt_manifest.cache_dir_for_model(trt_manifest.TRT_MODEL_MATANYONE2)
        manifest = trt_manifest.build_manifest(
            fp,
            [
                {"shape": f"{model_key}/matanyone2_step_update", "size_mb": 1, "built_at": "2026-05-20T00:00:00Z"}
                for model_key in trt_manifest.MATANYONE2_MODEL_KEYS
            ],
            3,
            model_key=trt_manifest.TRT_MODEL_MATANYONE2,
        )
        paths = {
            "matanyone2_onnx_512_bs1": source_512,
            "matanyone2_onnx_1024_bs1": source_1024,
        }
        with patch.object(trt_manifest, "matanyone2_trt_source_model_path", return_value=source_1024), patch.object(
            trt_manifest, "matanyone2_trt_source_model_paths", return_value=paths
        ):
            trt_manifest.save_manifest(manifest, model_key=trt_manifest.TRT_MODEL_MATANYONE2)
            for model_key in trt_manifest.MATANYONE2_MODEL_KEYS:
                model_cache = trt_manifest.matanyone2_trt_cache_dir_for_key(model_key, cache_dir)
                model_cache.mkdir(parents=True, exist_ok=True)
                (model_cache / "step.engine").write_bytes(b"e" * (1024 * 1024))
            self.assertEqual(
                trt_manifest.cache_status(actual_fp=fp, model_key=trt_manifest.TRT_MODEL_MATANYONE2),
                "ready",
            )
        self.assertEqual(trt_manifest.manifest_path(trt_manifest.TRT_MODEL_MATANYONE2).parent.name, trt_manifest.MATANYONE2_CACHE_KEY)

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
