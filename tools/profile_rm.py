"""Per-stage CPU/GPU profiler for the RM GpuRmProcessor pipeline.

Breaks one process() call into stages with CUDA synchronisation so we can see
where time goes and which stages are CPU/host vs GPU -- to decide what to move
off the CPU and how the pipeline scales to 4K/8K.

    uv run python -m tools.profile_rm --video videos/2_2.mp4 --start 0 --iters 40
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from utils.runtime_dll_paths import apply_runtime_dll_paths

apply_runtime_dll_paths()

import cupy as cp

from pipeline.demosaic import (
    DemosaicEngines, WINDOW, CENTER, REST_SIZE,
    segmentation_mask, models_available, _RESIZE_KERNEL_SRC,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/2_2.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if not models_available():
        raise SystemExit("models missing")
    eng = DemosaicEngines("trt")
    det, res = eng.detector, eng.restorer
    size = det.size

    cap = cv2.VideoCapture(args.video)
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    frames = []
    for _ in range(args.iters + WINDOW):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    H, W = frames[0].shape[:2]
    print(f"[profile] {len(frames)} frames @ {W}x{H}")
    gpu = [cp.ascontiguousarray(cp.asarray(f)) for f in frames]

    resize = cp.RawModule(code=_RESIZE_KERNEL_SRC).get_function("bilinear_resize_u8")
    block = (16, 16, 1)

    def resize_into(src_g, dst_g, x0, y0, cw, ch):
        dh, dw = int(dst_g.shape[0]), int(dst_g.shape[1])
        grid = ((dw + 15) // 16, (dh + 15) // 16, 1)
        resize(grid, block, (src_g, dst_g, np.int32(src_g.shape[1]), np.int32(x0), np.int32(y0),
                             np.int32(cw), np.int32(ch), np.int32(dh), np.int32(dw)))

    def sync():
        cp.cuda.get_current_stream().synchronize()

    acc: dict[str, float] = {}
    def add(k, dt):
        acc[k] = acc.get(k, 0.0) + dt

    scale = min(size / W, size / H)
    nw, nh = int(round(W * scale)), int(round(H * scale))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    n = 0
    n_regions = 0
    for i in range(CENTER, CENTER + args.iters):
        if i + CENTER >= len(gpu):
            break
        n += 1
        window = gpu[i - CENTER:i + CENTER + 1]
        center_g = window[CENTER]

        sync(); t = time.perf_counter()
        canvas_g = cp.full((size, size, 3), 114, cp.uint8)
        tmp = cp.empty((nh, nw, 3), cp.uint8)
        resize_into(center_g, tmp, 0, 0, W, H)
        canvas_g[pad_y:pad_y + nh, pad_x:pad_x + nw] = tmp
        sync(); add("GPU det_letterbox", time.perf_counter() - t)

        t = time.perf_counter()
        canvas_host = canvas_g.get()
        add("CPU det_blob_download", time.perf_counter() - t)

        t = time.perf_counter()
        blob = np.ascontiguousarray(canvas_host.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        add("CPU det_normalize", time.perf_counter() - t)

        t = time.perf_counter()
        out0, out1 = det.sess.run(det.output_names, {det.input_name: blob})
        add("ORT det_run", time.perf_counter() - t)

        t = time.perf_counter()
        dets, protos = det.detect_blob(canvas_host, scale, pad_x, pad_y, (H, W), args.conf)
        add("CPU+ORT det_blob_total", time.perf_counter() - t)

        regions = [(d.box, cp.asarray(segmentation_mask(d, protos, (H, W), size))) for d in dets]
        n_regions += len(regions)

        sync(); t = time.perf_counter()
        out_g = center_g.copy()
        sync(); add("GPU center_copy", time.perf_counter() - t)

        for box, m_g in regions:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            sync(); t = time.perf_counter()
            stack_g = cp.empty((WINDOW, REST_SIZE, REST_SIZE, 3), cp.uint8)
            for k in range(WINDOW):
                resize_into(window[k], stack_g[k], x1, y1, bw, bh)
            sync(); add("GPU rest_crop_resize", time.perf_counter() - t)

            t = time.perf_counter()
            stack_host = stack_g.get()
            add("CPU rest_stack_download", time.perf_counter() - t)

            t = time.perf_counter()
            nchw = np.ascontiguousarray((stack_host.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)[None])
            add("CPU rest_normalize", time.perf_counter() - t)

            t = time.perf_counter()
            restored = res.restore_stack(nchw)
            add("ORT rest_run", time.perf_counter() - t)

            sync(); t = time.perf_counter()
            restored_g = cp.asarray(restored)
            resized_g = cp.empty((bh, bw, 3), cp.uint8)
            resize_into(restored_g, resized_g, 0, 0, REST_SIZE, REST_SIZE)
            region = out_g[y1:y2, x1:x2].astype(cp.float32)
            out_g[y1:y2, x1:x2] = (region * (1 - m_g) + resized_g.astype(cp.float32) * m_g).astype(cp.uint8)
            sync(); add("GPU rest_upload_blend", time.perf_counter() - t)

    rpf = n_regions / max(1, n)
    print(f"[profile] iters={n} regions/frame={rpf:.2f}")
    print(f"{'stage':28s} {'ms/frame':>10s}  kind")
    for k in sorted(acc, key=lambda x: -acc[x]):
        print(f"{k:28s} {acc[k]/n*1000.0:10.2f}")
    print("note: 'det_blob_total' overlaps 'det_run'+postproc; count det as letterbox+download+normalize+det_blob_total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
