from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_compat_shims() -> None:
    """Provide the tiny subset of timm/IPython needed by the vendored NVDS code."""

    if "timm" not in sys.modules and importlib.util.find_spec("timm") is None:
        timm = types.ModuleType("timm")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")
        registry = types.ModuleType("timm.models.registry")
        vision_transformer = types.ModuleType("timm.models.vision_transformer")

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(
                    shape, dtype=x.dtype, device=x.device
                )
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

        def to_2tuple(x: Any) -> tuple[Any, Any]:
            return x if isinstance(x, tuple) else (x, x)

        def trunc_normal_(
            tensor: torch.Tensor,
            mean: float = 0.0,
            std: float = 1.0,
            a: float = -2.0,
            b: float = 2.0,
        ) -> torch.Tensor:
            return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        def register_model(fn: Any) -> Any:
            return fn

        def _cfg(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        layers.DropPath = DropPath
        layers.to_2tuple = to_2tuple
        layers.trunc_normal_ = trunc_normal_
        registry.register_model = register_model
        vision_transformer._cfg = _cfg
        timm.models = models
        models.layers = layers
        models.registry = registry
        models.vision_transformer = vision_transformer

        sys.modules.update(
            {
                "timm": timm,
                "timm.models": models,
                "timm.models.layers": layers,
                "timm.models.registry": registry,
                "timm.models.vision_transformer": vision_transformer,
            }
        )

    if "IPython" not in sys.modules and importlib.util.find_spec("IPython") is None:
        ipython = types.ModuleType("IPython")
        ipython.embed = lambda *args, **kwargs: None
        sys.modules["IPython"] = ipython


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def strip_dataparallel_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


def load_model(source_root: Path, checkpoint_path: Path) -> nn.Module:
    install_compat_shims()
    sys.path.insert(0, str(source_root))

    from full_model import NVDS  # type: ignore

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_dataparallel_prefix(extract_state_dict(checkpoint))

    model = NVDS()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match NVDS model. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.eval()
    return model


class _NvdsBackbone(nn.Module):
    """Per-frame MiT-B5 backbone.

    The original NVDS runs this on all 4 window frames every step, but it is a
    pure per-frame CNN/transformer with no cross-frame ops, so its features are
    batch-independent. Exporting it on a single frame lets the wrapper run it
    once per frame and cache the last 4 outputs instead of recomputing 3/4 of the
    backbone work each step.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone

    def forward(self, frame_rgbd: torch.Tensor):  # [1, 4, H, W]
        c1, c2, c3, c4 = self.backbone(frame_rgbd)
        return c1, c2, c3, c4


class _NvdsHead(nn.Module):
    """Cross-frame stabilizer head.

    Takes the 4 window frames' backbone features (stacked on the batch axis, so
    each scale is ``[num_clips, C, h, w]``) plus the last frame's RGB (for the
    edge branch), and reproduces ``NVDS.forward`` from the backbone output onward.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.edge_conv = model.edge_conv
        self.edge_conv1 = model.edge_conv1
        self.Stabilizer = model.Stabilizer

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        last_rgb: torch.Tensor,  # [1, 3, H, W]
    ) -> torch.Tensor:
        edge_feat = self.edge_conv(last_rgb)
        edge_feat1 = self.edge_conv1(edge_feat)
        out = self.Stabilizer([c1, c2, c3, c4], edge_feat, edge_feat1, num_clips=4)
        return torch.relu(out)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Export the NVDS Stabilizer PyTorch checkpoint to ONNX."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root / "reference" / "NVDS",
        help="Path to the vendored NVDS source directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "NVDS" / "NVDS_Stabilizer.pth",
        help="Path to NVDS_Stabilizer.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output ONNX path. Defaults to "
            "models/NVDS/NVDS_Stabilizer_{width}x{height}.onnx."
        ),
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for the trace forward pass.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Mark only the batch axis dynamic. Height/width remain static.",
    )
    parser.add_argument(
        "--skip-ort-check",
        action="store_true",
        help="Skip ONNX Runtime inference comparison.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Export two graphs instead of the monolith: "
            "NVDS_Backbone_{w}x{h}.onnx (per-frame, cacheable) and "
            "NVDS_Head_{w}x{h}.onnx (cross-frame stabilizer). Cuts the redundant "
            "backbone recompute the sliding window otherwise incurs."
        ),
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def run_ort_check(onnx_path: Path, x: torch.Tensor, y_ref: torch.Tensor) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    y_onnx = session.run(None, {"rgbd_seq": x.detach().cpu().numpy()})[0]
    y_ref_np = y_ref.detach().cpu().numpy()
    return {
        "providers": session.get_providers(),
        "output_shape": list(y_onnx.shape),
        "max_abs_diff": float(np.max(np.abs(y_onnx - y_ref_np))),
        "mean_abs_diff": float(np.mean(np.abs(y_onnx - y_ref_np))),
    }


def default_output_path(width: int, height: int) -> Path:
    return repo_root() / "models" / "NVDS" / f"NVDS_Stabilizer_{width}x{height}.onnx"


def export_onnx(
    model: nn.Module,
    x: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> None:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": x,
        "f": str(output_path),
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["rgbd_seq"],
        "output_names": ["stabilized_depth"],
        "dynamic_axes": dynamic_axes,
    }
    # PyTorch 2.x accepts this and avoids the new dynamo exporter path, while
    # the older PyTorch builds commonly used for NVDS do not expose the keyword.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(**kwargs)


def _ort_session(onnx_path: Path, device: torch.device):
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(str(onnx_path), providers=providers)


def export_split(
    model: nn.Module,
    height: int,
    width: int,
    opset: int,
    device: torch.device,
    skip_ort_check: bool,
) -> dict[str, Any]:
    """Export backbone + head as two graphs and verify against the monolith."""
    import numpy as np

    backbone = _NvdsBackbone(model).to(device).eval()
    head = _NvdsHead(model).to(device).eval()

    out_dir = repo_root() / "models" / "NVDS"
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_path = out_dir / f"NVDS_Backbone_{width}x{height}.onnx"
    head_path = out_dir / f"NVDS_Head_{width}x{height}.onnx"

    dynamo_kw: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        dynamo_kw["dynamo"] = False

    x_single = torch.randn(1, 4, height, width, device=device)
    with torch.no_grad():
        feats = backbone(x_single)
    # The head consumes the 4 window frames stacked on the batch axis.
    c_examples = tuple(f.repeat(4, 1, 1, 1).contiguous() for f in feats)
    last_rgb = torch.randn(1, 3, height, width, device=device)

    print(f"Exporting backbone to {backbone_path}")
    torch.onnx.export(
        backbone,
        (x_single,),
        str(backbone_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["frame_rgbd"],
        output_names=["feat0", "feat1", "feat2", "feat3"],
        **dynamo_kw,
    )
    print(f"Exporting head to {head_path}")
    torch.onnx.export(
        head,
        (*c_examples, last_rgb),
        str(head_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["feat0", "feat1", "feat2", "feat3", "last_rgb"],
        output_names=["stabilized_depth"],
        **dynamo_kw,
    )

    import onnx

    for path in (backbone_path, head_path):
        onnx.checker.check_model(onnx.load(str(path)))
    print("ONNX checker passed for both graphs")

    result: dict[str, Any] = {
        "backbone": str(backbone_path),
        "head": str(head_path),
        "ort_check": None,
    }
    if skip_ort_check:
        return result

    # Compare the split pipeline (per-frame backbone + head) against the
    # monolithic PyTorch forward on the same 4-frame window.
    seq = torch.randn(1, 4, 4, height, width, device=device)
    with torch.no_grad():
        y_ref = model(seq).detach().cpu().numpy()

    bs = _ort_session(backbone_path, device)
    hs = _ort_session(head_path, device)
    per_frame = []
    for t in range(4):
        frame = seq[:, t].detach().cpu().numpy()  # [1, 4, H, W]
        per_frame.append(bs.run(None, {"frame_rgbd": frame}))
    stacked = [
        np.concatenate([per_frame[t][s] for t in range(4)], axis=0) for s in range(4)
    ]
    last_rgb_np = seq[:, -1, 0:3].detach().cpu().numpy()  # [1, 3, H, W]
    y_split = hs.run(
        None,
        {
            "feat0": stacked[0],
            "feat1": stacked[1],
            "feat2": stacked[2],
            "feat3": stacked[3],
            "last_rgb": last_rgb_np,
        },
    )[0]

    result["ort_check"] = {
        "max_abs_diff": float(np.max(np.abs(y_split - y_ref))),
        "mean_abs_diff": float(np.mean(np.abs(y_split - y_ref))),
        "output_shape": list(y_split.shape),
    }
    print(
        "Split vs monolith: "
        f"max_abs_diff={result['ort_check']['max_abs_diff']:.6g} "
        f"mean_abs_diff={result['ort_check']['mean_abs_diff']:.6g}"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("NVDS inference height and width should be multiples of 32.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    source_root = args.source_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = (args.output or default_output_path(args.width, args.height)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    torch.set_grad_enabled(False)

    print(f"Loading model from {checkpoint_path}")
    model = load_model(source_root, checkpoint_path).to(device)

    if args.split:
        result = export_split(
            model,
            args.height,
            args.width,
            args.opset,
            device,
            args.skip_ort_check,
        )
        metadata = {
            "model": "NVDS_Stabilizer_split",
            "checkpoint": str(checkpoint_path),
            "source_root": str(source_root),
            "backbone": result["backbone"],
            "head": result["head"],
            "backbone_input": {"name": "frame_rgbd", "shape": [1, 4, args.height, args.width]},
            "backbone_outputs": ["feat0", "feat1", "feat2", "feat3"],
            "head_inputs": {
                "features": ["feat0", "feat1", "feat2", "feat3"],
                "feature_batch": 4,
                "last_rgb": [1, 3, args.height, args.width],
            },
            "head_output": {"name": "stabilized_depth", "shape": [1, 1, args.height, args.width]},
            "sequence_length": 4,
            "rgb_preprocess": {
                "range": "0..1",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "opset": args.opset,
            "ort_check": result["ort_check"],
        }
        meta_path = (repo_root() / "models" / "NVDS" / f"NVDS_split_{args.width}x{args.height}.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote split metadata to {meta_path}")
        return

    x = torch.randn(args.batch_size, 4, 4, args.height, args.width, device=device)

    print(f"Tracing input shape: {tuple(x.shape)} on {device}")
    with torch.no_grad():
        start = time.perf_counter()
        y_ref = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"PyTorch output shape: {tuple(y_ref.shape)} in {time.perf_counter() - start:.2f}s")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "rgbd_seq": {0: "batch"},
            "stabilized_depth": {0: "batch"},
        }

    print(f"Exporting ONNX to {output_path}")
    start = time.perf_counter()
    export_onnx(model, x, output_path, args.opset, dynamic_axes)
    if device.type == "cuda":
        torch.cuda.synchronize()
    export_seconds = time.perf_counter() - start
    print(f"ONNX export finished in {export_seconds:.2f}s")

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX checker passed")

    ort_result = None
    if not args.skip_ort_check:
        print("Running ONNX Runtime check")
        ort_result = run_ort_check(output_path, x, y_ref)
        print(
            "ORT output shape: "
            f"{ort_result['output_shape']}, max_abs_diff={ort_result['max_abs_diff']:.6g}"
        )

    metadata = {
        "model": "NVDS_Stabilizer",
        "checkpoint": str(checkpoint_path),
        "source_root": str(source_root),
        "output": str(output_path),
        "input_name": "rgbd_seq",
        "input_layout": "N,T,C,H,W",
        "input_shape": [args.batch_size, 4, 4, args.height, args.width],
        "sequence_length": 4,
        "target_frame": "last",
        "input_channels": ["R", "G", "B", "depth_or_disparity"],
        "rgb_preprocess": {
            "range": "0..1",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "depth_preprocess": {
            "range": "0..1",
            "recommended_source": "normalized near/disparity, not raw DA3 distance depth",
            "normalization": "per-frame min/max or project wrapper equivalent",
        },
        "output_name": "stabilized_depth",
        "output_shape": list(y_ref.shape),
        "fixed_resolution": {
            "width": args.width,
            "height": args.height,
            "multiple_of": 32,
        },
        "opset": args.opset,
        "dynamic_batch": bool(args.dynamic_batch),
        "export_seconds": export_seconds,
        "ort_check": ort_result,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
