from __future__ import annotations

import os
import shutil
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from utils import trt_manifest


class MatAnyone2TensorRTRuntimePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("runtime_cache/test_matanyone2_runtime_trt")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def _fake_pipeline_matting(self):
        module = types.ModuleType("pipeline.matting")
        module.ONNX_TRT_ENGINE_CACHE_PATH = Path("wrong")
        module._filter_available_providers = lambda providers: list(providers)
        module._provider_config = lambda providers: [
            (
                "FAKE_TRT",
                {
                    "providers": ",".join(providers),
                    "trt_engine_cache_path": str(module.ONNX_TRT_ENGINE_CACHE_PATH),
                },
            )
        ]
        return module

    def _assert_runtime_cache_dir(self, module, model_key: str) -> None:
        model_dir = (config.ROOT / "models" / model_key).resolve()
        expected = self.root / trt_manifest.MATANYONE2_CACHE_KEY / model_key
        with patch.object(config, "ONNX_TRT_ENGINE_CACHE_PATH", self.root), patch.object(
            module, "cache_status", return_value="ready"
        ), patch.object(module, "is_matanyone2_trt_model_dir", return_value=True), patch.dict(
            "sys.modules", {"pipeline.matting": self._fake_pipeline_matting()}
        ), patch.dict(os.environ, {"PT_OFFLINE_MATANYONE2_TRT": "1"}, clear=False):
            providers = module._matanyone2_session_providers(trt_manifest.MATANYONE2_TRT_ONNX_NAME, model_dir)
            actual_cache_dir = Path(os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"])

        self.assertEqual(providers[0][0], "FAKE_TRT")
        self.assertEqual(actual_cache_dir, expected.resolve())
        self.assertEqual(Path(providers[0][1]["trt_engine_cache_path"]), expected.resolve())

    def test_green_runtime_uses_selected_matanyone2_trt_cache_dir(self) -> None:
        import tools.offline_passthrough as offline_passthrough

        self._assert_runtime_cache_dir(offline_passthrough, "matanyone2_onnx_1024_bs1")

    def test_alpha_runtime_uses_selected_matanyone2_trt_cache_dir(self) -> None:
        import tools.offline_alpha_passthrough as offline_alpha_passthrough

        self._assert_runtime_cache_dir(offline_alpha_passthrough, "matanyone2_onnx_512_bs1")


if __name__ == "__main__":
    unittest.main()
