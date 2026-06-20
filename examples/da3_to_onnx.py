"""Convert Depth Anything 3 (Small / Base) to a depth-only ONNX graph.

DA3 ships as PyTorch + safetensors. This project (PTMediaServer) runs every
model through onnxruntime, so before the 2D->VR pipeline can be ported we need
DA3 as a single ``.onnx`` file.

What gets exported
------------------
Only the *depth-only* sub-graph: ``DepthAnything3Net.forward(..., skip_camera=
True, skip_sky=True)``. That path is the ONNX-friendly subset -- it drops the
camera / sky / Gaussian-Splat branches whose ``torch.quantile`` / ``randint`` /
``.item()`` / boolean-mask control flow cannot be traced. The remaining graph is
DINOv2 (ViT-S for Small, ViT-B for Base) + DualDPT head, all static ops.

Why a fixed square input
------------------------
The transformer uses 2D RoPE + a learned ``pos_embed`` sized for a 37x37 patch
grid (img_size 518, patch 14). Feeding a 518x518 frame hits the
``npatch == N and w == h`` fast path in ``interpolate_pos_encoding`` so the
bicubic pos-embed interpolation is skipped entirely -- cleaner graph, no dynamic
spatial axes. The single-view (S=1) batch keeps the reference-view-selection
branch (needs S >= 3) out of the trace, so control flow is constant. Only the
batch axis (number of frames) is dynamic.

Input  : ``image``  float32 (B, 3, SIZE, SIZE), ImageNet-normalised RGB.
Output : ``depth``  float32 (B, h_out, w_out) -- raw DA3 depth (distance-like,
         smaller = nearer). Resize to frame and invert at runtime, exactly like
         the existing tool_2dvr ``inference_depth_only`` path does.

Usage
-----
    python examples/da3_to_onnx.py --variant both --validate

Run it with the VR_Video_Toolbox_NE venv, which already has the DA3 deps:
    G:/GIT/debug/VR_Video_Toolbox_NE/.venv/Scripts/python.exe \
        examples/da3_to_onnx.py --variant both --validate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- Locations -------------------------------------------------------------
# tool_2dvr vendors the DA3 source under _vendor/da3; weights live under
# VR_Video_Toolbox_NE/models/DA3/<Variant>. Outputs land in this project's
# models/DA3 as da3_small.onnx / da3_base.onnx.
DEFAULT_TOOLBOX = Path(r"G:/GIT/debug/VR_Video_Toolbox_NE")
DEFAULT_VENDOR = DEFAULT_TOOLBOX / "tool_2dvr" / "_vendor" / "da3"
DEFAULT_SRC_ROOT = DEFAULT_TOOLBOX / "models" / "DA3"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "DA3"

VARIANTS = {
    "small": {"src": "Small", "onnx": "da3_small.onnx"},
    "base": {"src": "Base", "onnx": "da3_base.onnx"},
    "large": {"src": "Large", "onnx": "da3_large.onnx"},
}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class DepthOnlyWrapper(nn.Module):
    """Wraps DepthAnything3 so ONNX sees ``-> depth (B,H,W)``.

    Calls the inner ``DepthAnything3Net`` directly (bypassing the api-level
    bf16 autocast) so the whole graph stays fp32, matching the head, which
    already runs autocast-disabled.

    ``fold_preprocess`` makes the graph take a uint8 ``(B,H,W,3)`` letterboxed
    canvas and do the ImageNet normalize + channel transpose on-device, so the
    runtime only pays the cv2 letterbox on the CPU (the numpy normalize was
    ~6 ms/frame at 1080p).
    """

    def __init__(self, da3: nn.Module, fold_preprocess: bool = False):
        super().__init__()
        self.net = da3.model  # DepthAnything3Net
        self.fold_preprocess = fold_preprocess
        if fold_preprocess:
            self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.fold_preprocess:
            # uint8 (B,H,W,3) -> normalized float (B,3,H,W)
            x = image.permute(0, 3, 1, 2).to(torch.float32) / 255.0
            image = (x - self._mean) / self._std
        # (B, 3, H, W) -> (B, S=1, 3, H, W): one independent view per frame.
        x = image.unsqueeze(1)
        out = self.net(
            x,
            None,            # extrinsics
            None,            # intrinsics
            [],              # export_feat_layers
            False,           # infer_gs
            False,           # use_ray_pose
            "middle",        # ref_view_strategy (unused at S=1)
            skip_camera=True,
            skip_sky=True,
        )
        depth = out["depth"]
        # Net returns (B, S, H, W); collapse the singleton view dim -> (B, H, W).
        if depth.dim() == 4:
            depth = depth[:, 0]
        elif depth.dim() == 5:
            depth = depth[:, 0, 0]
        return depth.float()


def _patch_position_getter() -> None:
    """Replace RoPE's ``cartesian_prod`` grid with an ONNX-exportable equivalent.

    ``torch.cartesian_prod`` has no ONNX symbolic. The grid is a constant (it
    only depends on patch dims, not input data), so meshgrid + stack is an exact
    drop-in that the exporter can fold to a constant.
    """
    from depth_anything_3.model.dinov2.layers.rope import PositionGetter

    def _call(self, batch_size, height, width, device):
        key = (height, width)
        if key not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            self.position_cache[key] = torch.stack(
                (yy.reshape(-1), xx.reshape(-1)), dim=-1
            )
        cached = self.position_cache[key]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    PositionGetter.__call__ = _call


def load_da3(model_dir: Path, vendor_root: Path):
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from depth_anything_3.api import DepthAnything3  # noqa: E402

    _patch_position_getter()
    model = DepthAnything3.from_pretrained(str(model_dir))
    model.eval()
    return model


def export_variant(
    variant: str,
    src_root: Path,
    vendor_root: Path,
    out_dir: Path,
    size: int,
    opset: int,
    device: str,
    validate: bool,
    fold_preprocess: bool = False,
) -> Path:
    info = VARIANTS[variant]
    model_dir = src_root / info["src"]
    # Non-native sizes get a size suffix (da3_base_700.onnx); 518 keeps the
    # canonical name so existing callers/caches are unaffected.
    onnx_name = info["onnx"] if size == 518 else info["onnx"][:-5] + f"_{size}.onnx"
    out_path = out_dir / onnx_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DA3 {variant} ===")
    print(f"  weights : {model_dir}")
    print(f"  output  : {out_path}  (size={size}, fold_preprocess={fold_preprocess})")
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"DA3 weights not found under {model_dir}")

    t0 = time.time()
    da3 = load_da3(model_dir, vendor_root)
    wrapper = DepthOnlyWrapper(da3, fold_preprocess=fold_preprocess).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    if fold_preprocess:
        dummy = torch.randint(0, 255, (1, size, size, 3), dtype=torch.uint8, device=device)
    else:
        dummy = torch.randn(1, 3, size, size, dtype=torch.float32, device=device)

    # Static H/W (RoPE + pos_embed are size-locked); only batch is dynamic.
    t0 = time.time()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes={"image": {0: "batch"}, "depth": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  exported in {time.time() - t0:.1f}s "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    if validate:
        validate_variant(wrapper, out_path, size, device, fold_preprocess)
    return out_path


def validate_variant(wrapper: nn.Module, out_path: Path, size: int, device: str,
                     fold_preprocess: bool = False) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    if fold_preprocess:
        sample = rng.integers(0, 255, (2, size, size, 3), dtype=np.uint8)
    else:
        sample = rng.standard_normal((2, 3, size, size)).astype(np.float32)

    with torch.inference_mode():
        torch_depth = wrapper(torch.from_numpy(sample).to(device)).cpu().numpy()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(out_path), providers=providers)
    onnx_depth = sess.run(["depth"], {"image": sample})[0]

    if torch_depth.shape != onnx_depth.shape:
        raise SystemExit(
            f"  [FAIL] shape mismatch torch={torch_depth.shape} onnx={onnx_depth.shape}"
        )
    diff = np.abs(torch_depth - onnx_depth)
    denom = np.abs(torch_depth).mean() + 1e-6
    print(
        f"  [validate] out shape {onnx_depth.shape} | "
        f"max abs {diff.max():.4e} | mean abs {diff.mean():.4e} | "
        f"rel {diff.mean() / denom:.4e} | providers={sess.get_providers()}"
    )
    if diff.mean() / denom > 1e-2:
        print("  [WARN] relative error > 1e-2; inspect before trusting depth output")
    else:
        print("  [OK] torch vs onnxruntime match within tolerance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["small", "base", "large", "both"], default="both")
    p.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT,
                   help="Folder holding Small/ and Base/ DA3 weight dirs")
    p.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR,
                   help="Vendored DA3 source root (contains depth_anything_3/)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Where to write da3_*.onnx (this project's models/DA3)")
    p.add_argument("--size", type=int, default=518,
                   help="Square input side, must be a multiple of 14 (default 518 = native 37x37 grid)")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Trace device; cpu keeps the graph fp32 and deterministic")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--fold-preprocess", dest="fold_preprocess", action="store_true",
                   help="Bake ImageNet normalize into the graph; input becomes uint8 (B,size,size,3)")
    p.set_defaults(validate=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.size % 14 != 0:
        raise SystemExit(f"--size must be a multiple of 14, got {args.size}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    variants = ["small", "base"] if args.variant == "both" else [args.variant]
    written = []
    for v in variants:
        written.append(
            export_variant(v, args.src_root, args.vendor, args.out_dir,
                           args.size, args.opset, args.device, args.validate,
                           args.fold_preprocess)
        )
    print("\nDone. Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
