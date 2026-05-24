"""Check captured green-screen live video content against the source timeline.

This complements check_live_audio_alignment.py. It decodes one frame from a
captured MPEG-TS file, masks out the green background, and compares the visible
foreground against PyNv-decoded source frames around the requested start time.

The tool is intended for green passthrough captures. Alpha-packed output changes
the image layout and should be checked with a dedicated matcher.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.pynv_io import GpuP016Frame, PyNvSimpleDecoder  # noqa: E402


def _run_frame_extract(path: Path, *, at_sec: float, width: int, height: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if at_sec > 0:
        cmd.extend(["-ss", f"{at_sec:.6f}"])
    cmd.extend(
        [
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    expected = width * height * 3
    if len(proc.stdout) != expected:
        raise RuntimeError(f"expected one RGB frame ({expected} bytes), got {len(proc.stdout)} bytes")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, 3)


def _foreground_mask(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(rgb.shape[:2], dtype=bool)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    green = (g > 170) & (r < 120) & (b < 120) & ((g - r) > 80) & ((g - b) > 80)
    return ~green


def _rgb_luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)


def _source_luma_frame(decoder: PyNvSimpleDecoder, index: int, *, width: int, height: int) -> np.ndarray:
    frame = decoder.frame_at(index)
    if isinstance(frame, GpuP016Frame):
        y_gpu = (frame.y.as_cupy(cp.uint16).reshape(frame.height, frame.width) >> 8).astype(cp.uint8)
    else:
        y_gpu = frame.y.as_cupy(cp.uint8).reshape(frame.height, frame.width)
    y = cp.asnumpy(y_gpu)
    cp.get_default_memory_pool().free_all_blocks()
    return cv2.resize(y, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)


def _normalized_mse(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aa = (aa - float(aa.mean())) / (float(aa.std()) or 1.0)
    bb = (bb - float(bb.mean())) / (float(bb.std()) or 1.0)
    return float(np.mean((aa - bb) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check green live video content alignment against source frames.")
    parser.add_argument("capture", type=Path, help="captured MPEG-TS file")
    parser.add_argument("--source", required=True, type=Path, help="local source video path")
    parser.add_argument("--source-start", required=True, type=float, help="requested source start time in seconds")
    parser.add_argument("--capture-time", type=float, default=0.0, help="time in the capture to sample")
    parser.add_argument("--search-before", type=float, default=0.5)
    parser.add_argument("--search-after", type=float, default=1.5)
    parser.add_argument("--max-offset", type=float, default=0.10)
    parser.add_argument("--scale", default="640x320", help="comparison size, WIDTHxHEIGHT")
    parser.add_argument("--mask", choices=("green", "all"), default="green")
    parser.add_argument("--step-frames", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.capture.exists():
        print(f"missing capture: {args.capture}", file=sys.stderr)
        return 2
    if not args.source.exists():
        print(f"missing source: {args.source}", file=sys.stderr)
        return 2
    try:
        width_s, height_s = str(args.scale).lower().split("x", 1)
        width = int(width_s)
        height = int(height_s)
    except Exception:
        print("--scale must be WIDTHxHEIGHT", file=sys.stderr)
        return 2
    if width <= 0 or height <= 0:
        print("--scale dimensions must be positive", file=sys.stderr)
        return 2

    try:
        out_rgb = _run_frame_extract(args.capture, at_sec=max(0.0, args.capture_time), width=width, height=height)
        mask = _foreground_mask(out_rgb, args.mask)
        mask_pixels = int(mask.sum())
        if mask_pixels < 200:
            raise RuntimeError(f"foreground mask too small: {mask_pixels} pixels")
        out_y = _rgb_luma(out_rgb)

        decoder = PyNvSimpleDecoder(args.source)
        try:
            fps = float(decoder.info.fps or 0.0)
            if fps <= 0:
                raise RuntimeError("source FPS unavailable from PyNv decoder")
            expected_sec = float(args.source_start) + max(0.0, float(args.capture_time))
            expected_index = int(round(expected_sec * fps))
            before = max(0, int(round(max(0.0, args.search_before) * fps)))
            after = max(0, int(round(max(0.0, args.search_after) * fps)))
            lo = max(0, expected_index - before)
            hi = min(len(decoder) - 1, expected_index + after)
            step = max(1, int(args.step_frames))

            best: tuple[float, int] | None = None
            checked = 0
            for index in range(lo, hi + 1, step):
                src_y = _source_luma_frame(decoder, index, width=width, height=height)
                mse = _normalized_mse(out_y[mask], src_y[mask])
                checked += 1
                if best is None or mse < best[0]:
                    best = (mse, index)
        finally:
            decoder.stop()

        if best is None:
            raise RuntimeError("no source frames checked")
        best_mse, matched_index = best
        matched_sec = matched_index / fps
        relative = matched_sec - expected_sec
        record = {
            "capture": str(args.capture),
            "source": str(args.source),
            "source_start_sec": float(args.source_start),
            "capture_time_sec": float(args.capture_time),
            "expected_source_sec": expected_sec,
            "matched_source_sec": matched_sec,
            "relative_to_expected_sec": relative,
            "abs_offset_sec": abs(relative),
            "expected_index": expected_index,
            "matched_index": matched_index,
            "fps": fps,
            "score_mse": best_mse,
            "mask_pixels": mask_pixels,
            "frames_checked": checked,
        }
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(
            f"video_content_offset={relative:.3f}s "
            f"(matched={matched_sec:.3f}s expected={expected_sec:.3f}s mse={best_mse:.3f})"
        )
    return 0 if abs(relative) <= float(args.max_offset) else 1


if __name__ == "__main__":
    raise SystemExit(main())
