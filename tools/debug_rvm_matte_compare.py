"""Generate RVM matte comparison screenshots for two input/downsample settings."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = [
    ("full_sbs_batch1", 1024, 0.5, 0, 0),
    ("split_sbs_batch2", 1024, 0.5, 1, 1),
]


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return text.strip("._") or "video"


def _resize_preview(img, max_width: int):
    import cv2

    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    new_h = max(1, int(round(img.shape[0] * max_width / img.shape[1])))
    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)


def _label(img, text: str):
    import cv2
    import numpy as np

    out = img.copy()
    h, w = out.shape[:2]
    band_h = max(34, min(54, h // 12))
    band = out[:band_h, :]
    band[:] = (band.astype(np.float32) * 0.35).astype(np.uint8)
    cv2.putText(
        out,
        text,
        (12, max(24, band_h - 13)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    return out


def _read_or_blank(path: Path, shape: tuple[int, int, int] | None = None):
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    if shape is None:
        shape = (360, 640, 3)
    return np.zeros(shape, dtype=np.uint8)


def _left_half(img):
    w = img.shape[1]
    return img[:, : max(1, w // 2)].copy()


def _write_compare_sheets(out_dir: Path, videos: list[str], frames: int, stack: str) -> None:
    import cv2
    import numpy as np

    compare_dir = out_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        stem = Path(video).stem
        safe = _safe_name(stem)
        for i in range(frames):
            first_matte = out_dir / VARIANTS[0][0] / safe / f"{i:02d}_matte.jpg"
            base = _read_or_blank(first_matte)
            target_h, target_w = base.shape[:2]

            cells = []
            labels = [
                (f"{label} {input_size} / {ratio:g}", out_dir / label / safe / f"{i:02d}_matte.jpg")
                for label, input_size, ratio, _split_sbs, _sbs_batch in VARIANTS
            ]
            imgs = []
            for label, path in labels:
                img = _read_or_blank(path, base.shape)
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                imgs.append((label, img))

            for label, img in imgs:
                cells.append(_label(img, label))
            sheet = np.vstack(cells) if stack == "vertical" else np.hstack(cells)
            cv2.imwrite(str(compare_dir / f"{safe}_{i:02d}_compare.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _reuse_variant_outputs(src_root: Path, dst_root: Path, videos: list[str], frames: int, left_only: bool) -> None:
    import cv2

    dst_root.mkdir(parents=True, exist_ok=True)
    for video in videos:
        safe = _safe_name(Path(video).stem)
        src_dir = src_root / safe
        dst_dir = dst_root / safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            src_path = src_dir / f"{i:02d}_matte.jpg"
            img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(src_path)
            if left_only:
                img = _left_half(img)
            cv2.imwrite(str(dst_dir / f"{i:02d}_matte.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    stats_src = src_root / "stats.csv"
    if stats_src.exists():
        stats_dst = dst_root / "stats.csv"
        stats_dst.write_bytes(stats_src.read_bytes())


def _resolve_video(value: str):
    import config

    p = Path(value)
    candidates = [
        p,
        config.ROOT / value,
        config.VIDEO_DIR / value,
        config.ROOT / "videos" / value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _sample_indices(frame_count: int, frames: int) -> list[int]:
    if frame_count <= 1:
        return [0] * frames
    return [
        min(frame_count - 1, max(0, int(round(((i + 0.5) / frames) * (frame_count - 1)))))
        for i in range(frames)
    ]


def _apply_ui_cuda_dll_environment() -> None:
    """Match the UI-launched CUDA DLL search path for direct debug runs."""
    from ui.services.process_helpers import base_environment

    os.environ.update(base_environment())


def _patch_runtime_tempdir(runtime_tmp: Path) -> None:
    """Keep CUDA/CuPy temporary files under the project runtime cache."""
    runtime_tmp.mkdir(parents=True, exist_ok=True)

    class FixedTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.name = str(runtime_tmp)

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

        def cleanup(self):
            return None

    tempfile.TemporaryDirectory = FixedTemporaryDirectory


def _worker(args: argparse.Namespace) -> int:
    _apply_ui_cuda_dll_environment()
    os.environ["PT_MODEL_PATH"] = str(Path(args.model).resolve())
    os.environ["PT_MATTING_INPUT_SIZE"] = str(args.input_size)
    os.environ["PT_RVM_DOWNSAMPLE_RATIO"] = str(args.downsample_ratio)
    os.environ["PT_MATTING_SBS_BATCH"] = str(args.sbs_batch)
    os.environ["PT_MATTING_SPLIT_SBS"] = str(args.split_sbs)
    os.environ["PT_MATTING_WARMUP_RUNS"] = "0"
    os.environ["PT_STARTUP_GPU_WARMUP"] = "0"

    import config
    from utils.gpu_runtime_cache import configure_gpu_runtime_cache

    gpu_cache_env = configure_gpu_runtime_cache()
    _patch_runtime_tempdir(config.RUNTIME_TMP_DIR)
    print(
        f"[{args.variant_label}] gpu cache cuda={gpu_cache_env.cuda_cache_path} "
        f"cupy={gpu_cache_env.cupy_cache_dir} tmp={config.RUNTIME_TMP_DIR}",
        flush=True,
    )

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available; aborting instead of running CPU-only. "
            f"available={sorted(available)}"
        )

    import cv2
    import numpy as np

    from pipeline.matting import Matter

    out_dir = Path(args.out_dir).resolve() / args.variant_label
    out_dir.mkdir(parents=True, exist_ok=True)
    matter = Matter()
    active = matter.sess.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider is not active for the RVM session; aborting instead of running CPU-only. "
            f"active={active}"
        )
    print(f"[{args.variant_label}] active ORT providers={active}", flush=True)
    rows: list[dict[str, object]] = []

    for video in args.videos:
        src = _resolve_video(video)
        safe = _safe_name(src.stem)
        video_dir = out_dir / safe
        video_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {src}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        indices = _sample_indices(frame_count, int(args.frames))

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {idx} from {src}")
            matter.reset_state()
            matte, timing = matter.composite_green_profile(frame)
            matte_preview = _resize_preview(matte, int(args.preview_width))
            if args.left_only:
                matte_preview = _left_half(matte_preview)

            cv2.imwrite(str(video_dir / f"{i:02d}_matte.jpg"), matte_preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(
                {
                    "variant": args.variant_label,
                    "video": src.name,
                    "sample": i,
                    "frame_index": idx,
                    "fps": f"{fps:.6f}",
                    "frame_count": frame_count,
                    "input_size": args.input_size,
                    "downsample_ratio": args.downsample_ratio,
                    "ort_shape": getattr(matter, "_last_ort_shape", ""),
                    "preprocess_ms": f"{timing.preprocess_ms:.3f}",
                    "ort_ms": f"{timing.ort_ms:.3f}",
                    "composite_ms": f"{timing.composite_ms:.3f}",
                }
            )
            print(
                f"[{args.variant_label}] {src.name} {i + 1}/{args.frames} "
                f"frame={idx} shape={getattr(matter, '_last_ort_shape', '')} "
                f"ort={timing.ort_ms:.1f}ms comp={timing.composite_ms:.1f}ms"
            )
        cap.release()

    with (out_dir / "stats.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return 0


def _parent(args: argparse.Namespace) -> int:
    from ui.services.process_helpers import base_environment

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Path(args.model)
    if not model.is_absolute():
        model = Path.cwd() / model

    reuse_root = Path(args.reuse_first_variant_from).resolve() if args.reuse_first_variant_from else None
    for idx, (label, input_size, ratio, split_sbs, sbs_batch) in enumerate(VARIANTS):
        if idx == 0 and reuse_root is not None:
            src_root = reuse_root / label
            print(f"[compare] reusing {src_root} -> {out_dir / label}")
            _reuse_variant_outputs(src_root, out_dir / label, args.videos, int(args.frames), bool(args.left_only))
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--variant-label",
            label,
            "--input-size",
            str(input_size),
            "--downsample-ratio",
            str(ratio),
            "--split-sbs",
            str(split_sbs),
            "--sbs-batch",
            str(sbs_batch),
            "--model",
            str(model),
            "--frames",
            str(args.frames),
            "--preview-width",
            str(args.preview_width),
            "--out-dir",
            str(out_dir),
            *(["--left-only"] if args.left_only else []),
            *args.videos,
        ]
        print("[compare] running", subprocess.list2cmdline(cmd))
        subprocess.run(cmd, cwd=str(ROOT), env=base_environment(), check=True)

    _write_compare_sheets(out_dir, args.videos, int(args.frames), args.stack)
    print(f"[compare] wrote {out_dir}")
    print(f"[compare] comparison sheets: {out_dir / 'comparison'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RVM matte comparison screenshots.")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--out-dir", default="debug_output/rvm_matte_compare")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--preview-width", type=int, default=1920)
    parser.add_argument("--model", default="models/rvm_mobilenetv3_fp16.onnx")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant-label", default="")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--downsample-ratio", type=float, default=0.5)
    parser.add_argument("--split-sbs", type=int, choices=[0, 1], default=1)
    parser.add_argument("--sbs-batch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--stack", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--left-only", action="store_true", help="save only the left half of each matte image")
    parser.add_argument(
        "--reuse-first-variant-from",
        default="",
        help="reuse the first configured variant from an existing output root instead of regenerating it",
    )
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
