from __future__ import annotations

from pathlib import Path

import numpy as np


def static_rvm_model_path(source: Path, cache_dir: Path, batch: int, input_size: int, downsample: float) -> Path:
    return cache_dir / f"{source.stem}_static_b{int(batch)}_{int(input_size)}x{int(input_size)}_ds{downsample:g}.onnx"


def make_static_rvm_model(source: Path, target: Path, batch: int, h: int, w: int, downsample: float) -> Path:
    import onnx
    from onnx import numpy_helper, shape_inference

    def set_shape(value_info, shape: tuple[int, ...]) -> None:
        dims = value_info.type.tensor_type.shape.dim
        del dims[:]
        for value in shape:
            dim = dims.add()
            dim.dim_value = int(value)

    def remove_graph_input(model, name: str) -> None:
        kept = [value_info for value_info in model.graph.input if value_info.name != name]
        del model.graph.input[:]
        model.graph.input.extend(kept)

    def replace_initializer(model, name: str, value: np.ndarray) -> None:
        kept = [init for init in model.graph.initializer if init.name != name]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), name=name))

    def remove_nodes_by_output(model, output_names: set[str]) -> None:
        kept = [node for node in model.graph.node if not any(output in output_names for output in node.output)]
        del model.graph.node[:]
        model.graph.node.extend(kept)

    model = onnx.load(str(source))
    state_shapes = {
        "r1i": (batch, 16, max(1, int(round(h * downsample / 2))), max(1, int(round(w * downsample / 2)))),
        "r2i": (batch, 20, max(1, int(round(h * downsample / 4))), max(1, int(round(w * downsample / 4)))),
        "r3i": (batch, 40, max(1, int(round(h * downsample / 8))), max(1, int(round(w * downsample / 8)))),
        "r4i": (batch, 64, max(1, int(round(h * downsample / 16))), max(1, int(round(w * downsample / 16)))),
    }
    output_shapes = {
        "fgr": (batch, 3, h, w),
        "pha": (batch, 1, h, w),
        "r1o": state_shapes["r1i"],
        "r2o": state_shapes["r2i"],
        "r3o": state_shapes["r3i"],
        "r4o": state_shapes["r4i"],
    }
    for value_info in model.graph.input:
        if value_info.name == "src":
            set_shape(value_info, (batch, 3, h, w))
        elif value_info.name in state_shapes:
            set_shape(value_info, state_shapes[value_info.name])
    for value_info in model.graph.output:
        if value_info.name in output_shapes:
            set_shape(value_info, output_shapes[value_info.name])

    remove_graph_input(model, "downsample_ratio")
    replace_initializer(model, "downsample_ratio", np.asarray([downsample], dtype=np.float32))

    # RVM's first Resize scale normally comes from Concat([1, 1, downsample, downsample]).
    # TensorRT 10.16 rejects that dynamic shape expression; freezing this scale lets
    # ORT TensorRT EP build a usable static partition for the fixed runtime shape.
    remove_nodes_by_output(model, {"388"})
    replace_initializer(model, "388", np.asarray([1.0, 1.0, downsample, downsample], dtype=np.float32))

    inferred = shape_inference.infer_shapes(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, str(target))
    return target
