"""Benchmark the realtime mosaic-restoration (RM) inference core.

Measures end-to-end throughput of the detector + restoration ONNX models on a
real video so we can judge whether RM can sustain the 60 FPS realtime target.

It reports two figures:
  * detect-only FPS   - every frame is run through the YOLO11-seg detector and
                        only restored where a mosaic is found (the realistic
                        cost on clean footage / between mosaic regions).
  * worst-case FPS    - additionally forces ``--force-regions`` restoration calls
                        per frame (each a 7-frame 256x256 window) to measure the
                        cost when several mosaic regions are on screen at once.

This isolates inference cost (NVDEC decode + NVENC encode in the live path are
cheap by comparison and run on dedicated hardware blocks).

Usage:
    uv run python -m tools.bench_rm --video videos/test_1080p_2d.mp4
    uv run python -m tools.bench_rm --video videos/test_1080p_2d.mp4 --provider trt --force-regions 2
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

# Put the bundled CUDA / cuDNN / TensorRT DLLs on the loader path before any ORT
# session is created -- otherwise the CUDA/TRT EPs fail to load and ORT silently
# falls back to CPU (the server process does this at startup via main.py).
from utils.runtime_dll_paths import apply_runtime_dll_paths

apply_runtime_dll_paths()

from pipeline.demosaic import (
    DemosaicEngines,
    WINDOW,
    CENTER,
    REST_SIZE,
    restore_center_frame,
    rm_trt_cached,
    models_available,
)


def _read_frames(video: str, n: int, start_sec: float = 0.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < WINDOW:
        raise SystemExit(f"need at least {WINDOW} frames, got {len(frames)}")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/test_1080p_2d.mp4")
    ap.add_argument("--provider", default="trt", choices=["trt", "cuda", "cpu"])
    ap.add_argument("--start", type=float, default=0.0, help="seek to this second before reading")
    ap.add_argument("--frames", type=int, default=300, help="frames to process")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--force-regions", type=int, default=0,
                    help="extra forced restoration calls per frame (worst-case probe)")
    ap.add_argument("--gpu", action="store_true",
                    help="run the GPU (cupy) crop/resize/blend pipeline (Opt2)")
    ap.add_argument("--detect-interval", type=int, default=1,
                    help="run the detector every N frames, reuse boxes between (Opt1)")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not models_available():
        raise SystemExit("demosaic models not found under models/demosaic/")

    print(f"[bench] provider={args.provider} trt_cached={rm_trt_cached()}")
    print(f"[bench] loading engines (first TRT build can take minutes)...")
    t0 = time.perf_counter()
    engines = DemosaicEngines(provider=args.provider)
    print(f"[bench] engines ready in {time.perf_counter()-t0:.1f}s "
          f"detector={engines.detector.providers} restorer={engines.restorer.providers}")

    frames = _read_frames(args.video, args.frames + WINDOW, args.start)
    H, W = frames[0].shape[:2]
    print(f"[bench] {len(frames)} frames @ {W}x{H} start={args.start:.1f}s")

    # Pre-build a forced restoration crop window (a fixed centred box) so the
    # worst-case path exercises the restorer regardless of detections.
    fx1, fy1 = W // 2 - 128, H // 2 - 128
    forced_window = [f[fy1:fy1 + 256, fx1:fx1 + 256] for f in frames[:WINDOW]]

    gpu_frames = None
    processor = None
    if args.gpu:
        import cupy as cp
        from pipeline.demosaic import GpuRmProcessor

        # Pre-upload frames to the GPU once. In the live path NVDEC produces these
        # cupy frames directly, so upload cost is excluded from the timed loop.
        gpu_frames = [cp.ascontiguousarray(cp.asarray(f)) for f in frames]
        processor = GpuRmProcessor(engines, detect_interval=args.detect_interval)
        cp.cuda.get_current_stream().synchronize()

    def step(i: int):
        if args.gpu:
            window = gpu_frames[i:i + WINDOW]
            out = processor.process(window, args.conf)
            for _ in range(args.force_regions):
                engines.restorer.restore(forced_window)
            cp.cuda.get_current_stream().synchronize()
            return out
        window = frames[i:i + WINDOW]
        out = restore_center_frame(window, engines.detector, engines.restorer, args.conf)
        for _ in range(args.force_regions):
            engines.restorer.restore(forced_window)
        return out

    # warmup
    for i in range(min(args.warmup, len(frames) - WINDOW)):
        step(i)

    n = min(args.frames, len(frames) - WINDOW)
    # quick pass: how many mosaic regions does the detector actually find here?
    det_counts = []
    for i in range(n):
        dets, _ = engines.detector.detect(frames[i + CENTER], conf=args.conf)
        det_counts.append(len(dets))
    dc = np.array(det_counts)
    print(f"[bench] detections/frame: mean={dc.mean():.2f} max={int(dc.max())} "
          f"frames_with_mosaic={int((dc>0).sum())}/{n}")

    # detect-only timing (force_regions applied if >0)
    det_dt = []
    t_start = time.perf_counter()
    for i in range(n):
        a = time.perf_counter()
        step(i)
        det_dt.append(time.perf_counter() - a)
    total = time.perf_counter() - t_start

    arr = np.array(det_dt) * 1000.0
    fps = n / total
    path_label = "gpu" if args.gpu else "host"
    label = f"{path_label} {'worst-case' if args.force_regions > 0 else 'real'}"
    print("")
    print(f"[bench] === {label} (force_regions={args.force_regions}) ===")
    print(f"[bench] frames={n} total={total:.2f}s  FPS={fps:.1f}")
    print(f"[bench] per-frame ms: mean={arr.mean():.2f} p50={np.percentile(arr,50):.2f} "
          f"p95={np.percentile(arr,95):.2f} max={arr.max():.2f}")

    # isolate single restoration cost
    rt = []
    for _ in range(30):
        a = time.perf_counter()
        engines.restorer.restore(forced_window)
        rt.append(time.perf_counter() - a)
    rt_ms = np.array(rt) * 1000.0
    print(f"[bench] single restoration call ms: mean={rt_ms.mean():.2f} p50={np.percentile(rt_ms,50):.2f}")

    budget = 1000.0 / 60.0
    print("")
    print(f"[bench] 60 FPS budget = {budget:.2f} ms/frame")
    print(f"[bench] verdict: {'PASS' if fps >= 60 else 'BELOW 60FPS'} "
          f"({fps:.1f} FPS, {arr.mean():.2f} ms/frame mean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
