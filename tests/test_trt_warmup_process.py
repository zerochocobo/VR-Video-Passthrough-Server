from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from ui.services.trt_warmup_process import (
    _clean_cache_dir,
    _copy_rvm_shared_1024_artifacts,
    _make_rvm_state_dims_unique,
    _parse_args,
    _rvm_shared_1024_artifacts_available,
)
from utils.rvm_static_onnx import static_rvm_model_path
from utils.trt_manifest import MATANYONE2_CACHE_KEY, shape_inferred_model_path


def _value_info(name: str, dims: list[str | int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


class TrtWarmupProcessTests(unittest.TestCase):
    def test_parse_args_accepts_matanyone2_model(self) -> None:
        args = _parse_args(
            [
                "--model",
                "matanyone2",
                "--cache-dir",
                "runtime_cache/test_matanyone2_trt",
                "--matanyone2-model-key",
                "matanyone2_onnx_512_bs1",
            ]
        )
        self.assertEqual(args.model, "matanyone2")
        self.assertEqual(args.matanyone2_model_key, "matanyone2_onnx_512_bs1")
        self.assertEqual(args.fp16, 0)

    def test_rvm_state_symbolic_dims_do_not_reuse_src_height_width(self) -> None:
        graph = helper.make_graph(
            [],
            "rvm",
            [
                _value_info("src", ["batch_size", 3, "height", "width"]),
                _value_info("r1i", ["batch_size", "channels", "height", "width"]),
                _value_info("r2i", ["batch_size", "channels", "height", "width"]),
                _value_info("downsample_ratio", [1]),
            ],
            [
                _value_info("fgr", ["batch_size", 3, "height", "width"]),
                _value_info("r1o", ["batch_size", 16, "height", "width"]),
            ],
        )
        model = helper.make_model(graph)

        _make_rvm_state_dims_unique(model)

        inputs = {value.name: value for value in model.graph.input}
        outputs = {value.name: value for value in model.graph.output}
        self.assertEqual(_dim_params(inputs["src"]), ["batch_size", "", "height", "width"])
        self.assertEqual(_dim_params(inputs["r1i"]), ["r1i_batch", "r1i_channels", "r1i_height", "r1i_width"])
        self.assertEqual(_dim_params(inputs["r2i"]), ["r2i_batch", "r2i_channels", "r2i_height", "r2i_width"])
        self.assertEqual(_dim_params(outputs["fgr"]), ["batch_size", "", "height", "width"])
        self.assertEqual(_dim_params(outputs["r1o"]), ["r1o_batch", "", "r1o_height", "r1o_width"])

    def test_clean_cache_dir_can_preserve_offline_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cache_dir = Path(raw)
            (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (cache_dir / "runtime.engine").write_bytes(b"e")
            offline = cache_dir / "offline"
            offline.mkdir()
            (offline / "manifest.json").write_text("{}", encoding="utf-8")
            matanyone = cache_dir / MATANYONE2_CACHE_KEY
            matanyone.mkdir()
            (matanyone / "manifest.json").write_text("{}", encoding="utf-8")

            _clean_cache_dir(cache_dir, preserve_names={"offline", MATANYONE2_CACHE_KEY})

            self.assertFalse((cache_dir / "manifest.json").exists())
            self.assertFalse((cache_dir / "runtime.engine").exists())
            self.assertTrue((offline / "manifest.json").exists())
            self.assertTrue((matanyone / "manifest.json").exists())

    def test_copy_rvm_shared_1024_artifacts_between_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_cache = root / "realtime"
            target_cache = root / "offline"
            source_cache.mkdir()
            source_model = root / "rvm_mobilenetv3_fp32.onnx"
            source_model.write_bytes(b"model")
            shape_inferred_model_path(source_model, source_cache).write_bytes(b"shape")
            static_rvm_model_path(source_model, source_cache, 1, 1024, 0.5).write_bytes(b"b1")
            static_rvm_model_path(source_model, source_cache, 2, 1024, 0.5).write_bytes(b"b2")
            (source_cache / "runtime.engine").write_bytes(b"e" * (1024 * 1024))
            (source_cache / "runtime.profile").write_bytes(b"profile")

            copied = _copy_rvm_shared_1024_artifacts(source_cache, target_cache, source_model)

            self.assertGreaterEqual(copied, 4)
            self.assertTrue(_rvm_shared_1024_artifacts_available(target_cache, source_model))
            self.assertTrue((target_cache / "runtime.profile").exists())


def _dim_params(value_info) -> list[str]:
    return [dim.dim_param for dim in value_info.type.tensor_type.shape.dim]


if __name__ == "__main__":
    unittest.main()
