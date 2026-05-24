from __future__ import annotations

import unittest

import onnx
from onnx import TensorProto, helper

from ui.services.trt_warmup_process import _make_rvm_state_dims_unique, _parse_args


def _value_info(name: str, dims: list[str | int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


class TrtWarmupProcessTests(unittest.TestCase):
    def test_parse_args_accepts_matanyone2_model(self) -> None:
        args = _parse_args(["--model", "matanyone2", "--cache-dir", "runtime_cache/test_matanyone2_trt"])
        self.assertEqual(args.model, "matanyone2")

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


def _dim_params(value_info) -> list[str]:
    return [dim.dim_param for dim in value_info.type.tensor_type.shape.dim]


if __name__ == "__main__":
    unittest.main()
