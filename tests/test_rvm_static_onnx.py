from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from utils.rvm_static_onnx import make_static_rvm_model, static_rvm_model_path


def _value_info(name: str, dims: list[str | int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


class RvmStaticOnnxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("runtime_cache/test_rvm_static_onnx")
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_static_model_path_includes_batch_size_and_downsample(self) -> None:
        path = static_rvm_model_path(Path("models/rvm_mobilenetv3_fp32.onnx"), self.root, 2, 1024, 0.5)
        self.assertEqual(path.name, "rvm_mobilenetv3_fp32_static_b2_1024x1024_ds0.5.onnx")

    def test_make_static_rvm_model_freezes_runtime_shapes_and_resize_scale(self) -> None:
        source = self.root / "source.onnx"
        target = self.root / "static.onnx"
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["src"], ["fgr"]),
                helper.make_node("Identity", ["src"], ["pha"]),
                helper.make_node("Constant", [], ["388"], value=numpy_helper.from_array(np.asarray([0.0], dtype=np.float32))),
                helper.make_node("Identity", ["r1i"], ["r1o"]),
                helper.make_node("Identity", ["r2i"], ["r2o"]),
                helper.make_node("Identity", ["r3i"], ["r3o"]),
                helper.make_node("Identity", ["r4i"], ["r4o"]),
            ],
            "rvm",
            [
                _value_info("src", ["batch", 3, "height", "width"]),
                _value_info("r1i", ["batch", 16, "r1h", "r1w"]),
                _value_info("r2i", ["batch", 20, "r2h", "r2w"]),
                _value_info("r3i", ["batch", 40, "r3h", "r3w"]),
                _value_info("r4i", ["batch", 64, "r4h", "r4w"]),
                _value_info("downsample_ratio", [1]),
            ],
            [
                _value_info("fgr", ["batch", 3, "height", "width"]),
                _value_info("pha", ["batch", 1, "height", "width"]),
                _value_info("r1o", ["batch", 16, "r1h", "r1w"]),
                _value_info("r2o", ["batch", 20, "r2h", "r2w"]),
                _value_info("r3o", ["batch", 40, "r3h", "r3w"]),
                _value_info("r4o", ["batch", 64, "r4h", "r4w"]),
            ],
        )
        onnx.save(helper.make_model(graph), str(source))

        make_static_rvm_model(source, target, batch=2, h=1024, w=1024, downsample=0.5)

        model = onnx.load(str(target))
        inputs = {value.name: value for value in model.graph.input}
        outputs = {value.name: value for value in model.graph.output}
        initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        self.assertNotIn("downsample_ratio", inputs)
        np.testing.assert_allclose(initializers["downsample_ratio"], np.asarray([0.5], dtype=np.float32))
        np.testing.assert_allclose(initializers["388"], np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32))
        self.assertFalse(any("388" in node.output for node in model.graph.node))
        self.assertEqual(_dims(inputs["src"]), [2, 3, 1024, 1024])
        self.assertEqual(_dims(inputs["r1i"]), [2, 16, 256, 256])
        self.assertEqual(_dims(inputs["r2i"]), [2, 20, 128, 128])
        self.assertEqual(_dims(inputs["r3i"]), [2, 40, 64, 64])
        self.assertEqual(_dims(inputs["r4i"]), [2, 64, 32, 32])
        self.assertEqual(_dims(outputs["pha"]), [2, 1, 1024, 1024])


def _dims(value_info) -> list[int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


if __name__ == "__main__":
    unittest.main()
