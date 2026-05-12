"""Dump intermediate alpha layers for realtime alpha passthrough diagnosis."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import config  # noqa: E402
from pipeline.alpha_packer import AlphaPacker  # noqa: E402
from pipeline.matting import get_matter  # noqa: E402
from pipeline.pynv_io import GpuP016Frame, PyNvSimpleDecoder  # noqa: E402
from utils.gpu_runtime_cache import configure_gpu_runtime_cache  # noqa: E402
from utils.video_metadata import cfr_source_index, probe_video_metadata  # noqa: E402


configure_gpu_runtime_cache()


def _resolve_video(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = (config.ROOT / value).resolve()
    if candidate.exists():
        return candidate
    candidate = (config.VIDEO_DIR / value).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    cv2.imwrite(str(path), img)


def _save_gray_preview(path: Path, arr: np.ndarray, max_width: int) -> None:
    import cv2

    img = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    if max_width > 0 and img.shape[1] > max_width:
        new_h = max(1, int(round(img.shape[0] * max_width / img.shape[1])))
        img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    _save_gray(path, img)


def _copy_alpha_to_host(alpha) -> np.ndarray:
    try:
        import cupy as cp

        if hasattr(alpha, "data") and hasattr(alpha.data, "ptr"):
            return cp.asnumpy(alpha)
    except Exception:
        pass
    return np.asarray(alpha)


def _copy_fisheye_alpha_to_host(packer: AlphaPacker) -> np.ndarray:
    try:
        import cupy as cp

        return cp.asnumpy(packer._g_fisheye_alpha)
    except Exception:
        return np.asarray(packer._g_fisheye_alpha)


def _diff_stats(prev: np.ndarray | None, cur: np.ndarray) -> tuple[float, int, int, float]:
    if prev is None:
        return 0.0, 0, 0, 0.0
    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    changed = int((diff >= 16).sum())
    total = int(diff.size)
    ratio = changed / total if total else 0.0
    return float(diff.mean()), int(diff.max()), changed, ratio


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump alpha model/fisheye layers for a short passthrough segment.")
    parser.add_argument("video")
    parser.add_argument("--out-dir", default=str(config.ROOT / "debug_output" / "alpha_debug"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--model", default=str(config.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx"))
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--rvm-downsample-ratio", type=float, default=0.5)
    parser.add_argument("--p016-shift", type=int, default=int(config.PASSTHROUGH_PYNV_10BIT_SHIFT))
    parser.add_argument("--preview-max-width", type=int, default=2048)
    args = parser.parse_args()

    src = _resolve_video(args.video)
    out_dir = Path(args.out_dir).resolve() / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    config.MODEL_PATH = str(Path(args.model).resolve())
    config.MATTING_INPUT_SIZE = int(args.input_size)
    config.RVM_DOWNSAMPLE_RATIO = float(args.rvm_downsample_ratio)
    config.MATTING_WARMUP_RUNS = 0
    config.MATTING_SBS_BATCH = False

    meta = probe_video_metadata(src)
    bit_depth = int(meta.codec.bit_depth or 8)
    dec = PyNvSimpleDecoder(src, bit_depth=bit_depth)
    info = dec.info
    source_fps = float(meta.timing.source_fps or info.fps or 30.0)
    fps = min(source_fps, float(args.fps or source_fps))
    start_out = int(round(max(0.0, args.start) * fps))
    frames = max(1, int(args.frames))

    matter = get_matter()
    matter.reset_state()
    packer = AlphaPacker(matter)

    csv_path = out_dir / "alpha_stats.csv"
    prev_model: np.ndarray | None = None
    prev_fisheye: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    for i in range(frames):
        out_idx = start_out + i
        src_idx = min(len(dec) - 1, cfr_source_index(out_idx, source_fps, fps))
        frame = dec.frame_at(src_idx)
        h, w = int(frame.height), int(frame.width)
        if isinstance(frame, GpuP016Frame):
            matter.upload_p016_planes_as_nv12_gpu(
                frame.y.as_cupy(),
                frame.uv.as_cupy(),
                h,
                w,
                shift_bits=int(args.p016_shift),
            )
        else:
            matter.upload_nv12_planes_gpu(frame.y.as_cupy(), frame.uv.as_cupy(), h, w)
        try:
            import cupy as cp

            y_plane = cp.asnumpy(matter._g_frame[:h, :])
        except Exception:
            y_plane = np.asarray(matter._g_frame[:h, :])
        alpha, timing, ort_shape = matter._alpha_low_res_gpu(h, w, use_nv12=True)
        packer.pack_uploaded(alpha, h, w)
        model_alpha = (_copy_alpha_to_host(alpha) * 255.0).astype(np.uint8)
        fisheye_alpha = _copy_fisheye_alpha_to_host(packer)

        model_mean_diff, model_max_diff, model_changed, model_ratio = _diff_stats(prev_model, model_alpha)
        fish_mean_diff, fish_max_diff, fish_changed, fish_ratio = _diff_stats(prev_fisheye, fisheye_alpha)
        rows.append(
            {
                "frame": i,
                "out_idx": out_idx,
                "src_idx": src_idx,
                "ort_shape": ort_shape,
                "preprocess_ms": f"{timing.preprocess_ms:.3f}",
                "ort_ms": f"{timing.ort_ms:.3f}",
                "y_min": int(y_plane.min()),
                "y_max": int(y_plane.max()),
                "y_mean": f"{float(y_plane.mean()):.3f}",
                "model_min": int(model_alpha.min()),
                "model_max": int(model_alpha.max()),
                "model_mean": f"{float(model_alpha.mean()):.3f}",
                "model_diff_mean": f"{model_mean_diff:.3f}",
                "model_diff_max": model_max_diff,
                "model_diff_changed": model_changed,
                "model_diff_ratio": f"{model_ratio:.6f}",
                "fisheye_min": int(fisheye_alpha.min()),
                "fisheye_max": int(fisheye_alpha.max()),
                "fisheye_mean": f"{float(fisheye_alpha.mean()):.3f}",
                "fisheye_diff_mean": f"{fish_mean_diff:.3f}",
                "fisheye_diff_max": fish_max_diff,
                "fisheye_diff_changed": fish_changed,
                "fisheye_diff_ratio": f"{fish_ratio:.6f}",
            }
        )
        _save_gray_preview(out_dir / "luma_y" / f"{i:04d}_src{src_idx:06d}.png", y_plane, args.preview_max_width)
        _save_gray(out_dir / "model_alpha" / f"{i:04d}_src{src_idx:06d}.png", model_alpha)
        _save_gray_preview(out_dir / "fisheye_alpha" / f"{i:04d}_src{src_idx:06d}.png", fisheye_alpha, args.preview_max_width)
        prev_model = model_alpha
        prev_fisheye = fisheye_alpha
        print(
            f"[alpha-debug] {i + 1}/{frames} src_idx={src_idx} model_diff={model_mean_diff:.2f}/{model_max_diff} "
            f"fisheye_diff={fish_mean_diff:.2f}/{fish_max_diff} ort={timing.ort_ms:.1f}ms"
        )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[alpha-debug] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
