"""
Measure ONNX Runtime cold-start cost without PyNvVideoCodec.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Matter/ORT cold-start without video decode.")
    parser.add_argument("--providers", default="", help="override providers")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--split-sbs", action="store_true")
    parser.add_argument("--alpha-stride", type=int, default=3)
    parser.add_argument("--cuda-cudnn-search", default="")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    config.MATTING_INPUT_SIZE = int(args.input_size)
    if args.providers:
        config.ONNX_PROVIDERS = [p.strip() for p in args.providers.split(",") if p.strip()]
        os.environ["PT_ONNX_PROVIDERS"] = ",".join(config.ONNX_PROVIDERS)
    if args.split_sbs:
        os.environ["PT_MATTING_SPLIT_SBS"] = "1"
        os.environ["PT_MATTING_SBS_BATCH"] = "1"
    if args.cuda_cudnn_search:
        os.environ["PT_CUDA_CUDNN_CONV_ALGO_SEARCH"] = args.cuda_cudnn_search
    if args.no_warmup:
        config.MATTING_WARMUP_RUNS = 0
    os.environ["PT_ALPHA_STRIDE"] = str(max(1, args.alpha_stride))

    t0 = time.perf_counter()
    print("[ort-cold] import matting", flush=True)
    from pipeline.matting import Matter

    print(f"[timeline] matting import: {time.perf_counter() - t0:.3f}s", flush=True)
    t0 = time.perf_counter()
    print("[ort-cold] Matter init", flush=True)
    matter = Matter()
    print(f"[timeline] Matter init: {time.perf_counter() - t0:.3f}s", flush=True)

    import numpy as np

    h, w = (2048, 4096) if args.split_sbs else (512, 512)
    nv12 = np.zeros((h * 3 // 2, w), dtype=np.uint8)
    t0 = time.perf_counter()
    print("[ort-cold] first mat", flush=True)
    _, timing = matter.composite_green_nv12_to_nv12_profile(nv12.reshape(-1), h, w)
    elapsed = time.perf_counter() - t0
    print(f"[timeline] first mat total: {elapsed:.3f}s")
    print(f"[timeline] first preprocess: {timing.preprocess_ms / 1000.0:.3f}s")
    print(f"[timeline] first ort: {timing.ort_ms / 1000.0:.3f}s")
    print(f"[timeline] first composite: {timing.composite_ms / 1000.0:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
