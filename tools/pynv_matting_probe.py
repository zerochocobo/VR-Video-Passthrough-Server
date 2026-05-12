"""
Experimental PyNvVideoCodec + current matting probe.

This does not replace the production passthrough path. It measures a first
hybrid step: PyNv GPU decode frames, copy GPU planes into the current contiguous
NV12 device buffer, reuse existing RVM/composite kernels, and return NV12 host
output for parity with the current encoder handoff.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.pynv_io import PyNvSimpleDecoder  # noqa: E402


def _configure_local_temp() -> None:
    cache = config.ROOT / ".uv-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("TMP", str(cache))
    os.environ.setdefault("TEMP", str(cache))
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache / "cupy"))


def _patch_cupy_tempdir() -> None:
    """Work around sandbox ACL issues for Python-created TemporaryDirectory."""
    fixed = config.ROOT / "debug_output" / "cupy_tmp_fixed"
    fixed.mkdir(parents=True, exist_ok=True)

    class FixedTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.name = str(fixed)

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

        def cleanup(self):
            return None

    import tempfile

    tempfile.TemporaryDirectory = FixedTemporaryDirectory


def _resolve_video(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = config.VIDEO_DIR / p
    return p.resolve()


def main() -> int:
    _configure_local_temp()
    _patch_cupy_tempdir()
    parser = argparse.ArgumentParser(description="Probe PyNvVideoCodec decode + current matting kernels.")
    parser.add_argument("video", help="video filename under videos/ or absolute path")
    parser.add_argument("-n", "--frames", type=int, default=120, help="number of frames")
    parser.add_argument("--input-size", type=int, default=512, help="override matting input size")
    parser.add_argument("--alpha-stride", type=int, default=3, help="override PT_ALPHA_STRIDE")
    parser.add_argument("--discard", type=int, default=1, help="exclude first N frames from steady-state summary")
    parser.add_argument("--no-warmup", action="store_true", help="disable Matter warmup for first-kernel diagnostics")
    parser.add_argument(
        "--providers",
        default="",
        help="override PT_ONNX_PROVIDERS for this probe, e.g. CUDAExecutionProvider,CPUExecutionProvider",
    )
    parser.add_argument(
        "--cuda-cudnn-search",
        default="",
        choices=("", "DEFAULT", "HEURISTIC", "EXHAUSTIVE"),
        help="set CUDA EP cudnn_conv_algo_search for this probe",
    )
    parser.add_argument("--timeline", action="store_true", help="print phase timings for cold-start diagnosis")
    args = parser.parse_args()

    config.MATTING_INPUT_SIZE = int(args.input_size)
    if args.no_warmup:
        config.MATTING_WARMUP_RUNS = 0
    if args.providers:
        config.ONNX_PROVIDERS = [p.strip() for p in args.providers.split(",") if p.strip()]
        os.environ["PT_ONNX_PROVIDERS"] = ",".join(config.ONNX_PROVIDERS)
    if args.cuda_cudnn_search:
        os.environ["PT_CUDA_CUDNN_CONV_ALGO_SEARCH"] = args.cuda_cudnn_search
    os.environ["PT_ALPHA_STRIDE"] = str(max(1, int(args.alpha_stride)))
    t_script0 = time.perf_counter()
    if args.timeline:
        print("[timeline] before matting import", flush=True)
    from pipeline.matting import Matter
    if args.timeline:
        print(f"[timeline] matting import: {time.perf_counter() - t_script0:.3f}s", flush=True)

    src = _resolve_video(args.video)
    t0 = time.perf_counter()
    print("[pynv-mat] opening decoder", flush=True)
    dec = PyNvSimpleDecoder(src)
    if args.timeline:
        print(f"[timeline] decoder init: {time.perf_counter() - t0:.3f}s", flush=True)
    t0 = time.perf_counter()
    print("[pynv-mat] loading matter", flush=True)
    matter = Matter()
    if args.timeline:
        print(f"[timeline] Matter init: {time.perf_counter() - t0:.3f}s", flush=True)
    t0 = time.perf_counter()
    print("[pynv-mat] reset state", flush=True)
    matter.reset_state()
    if args.timeline:
        print(f"[timeline] reset_state: {time.perf_counter() - t0:.3f}s", flush=True)

    n = min(max(1, args.frames), max(1, len(dec)))
    t_decode = 0.0
    t_mat = 0.0
    t_pre = 0.0
    t_ort = 0.0
    t_comp = 0.0
    steady_n = 0
    steady_decode = 0.0
    steady_mat = 0.0
    steady_pre = 0.0
    steady_ort = 0.0
    steady_comp = 0.0
    steady_start = None
    t0_all = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        if i == 0:
            print("[pynv-mat] decode first frame", flush=True)
        frame = dec.frame_at(i)
        t1 = time.perf_counter()
        if i == 0 and args.timeline:
            print(f"[timeline] first decode: {t1 - t0:.3f}s", flush=True)
        if i == 0:
            print("[pynv-mat] mat first frame", flush=True)
        _, timing = matter.composite_green_gpu_nv12_frame_to_nv12_profile(frame)
        t2 = time.perf_counter()
        if i == 0 and args.timeline:
            print(f"[timeline] first mat total: {t2 - t1:.3f}s", flush=True)
            print(f"[timeline] first preprocess: {timing.preprocess_ms / 1000.0:.3f}s", flush=True)
            print(f"[timeline] first ort: {timing.ort_ms / 1000.0:.3f}s", flush=True)
            print(f"[timeline] first composite: {timing.composite_ms / 1000.0:.3f}s", flush=True)
        t_decode += t1 - t0
        t_mat += t2 - t1
        t_pre += timing.preprocess_ms
        t_ort += timing.ort_ms
        t_comp += timing.composite_ms
        if i >= max(0, args.discard):
            if steady_start is None:
                steady_start = t0
            steady_n += 1
            steady_decode += t1 - t0
            steady_mat += t2 - t1
            steady_pre += timing.preprocess_ms
            steady_ort += timing.ort_ms
            steady_comp += timing.composite_ms
        if (i + 1) % 30 == 0:
            elapsed = time.perf_counter() - t0_all
            steady_elapsed = (time.perf_counter() - steady_start) if steady_start is not None else 0.0
            print(
                f"[pynv-mat] {i + 1:4d}/{n} fps={(i + 1) / elapsed:6.2f} "
                f"dec={t_decode / (i + 1) * 1000:5.1f}ms "
                f"mat={t_mat / (i + 1) * 1000:5.1f}ms "
                f"ort={t_ort / (i + 1):5.1f}ms comp={t_comp / (i + 1):5.1f}ms "
                f"steady_fps={(steady_n / steady_elapsed) if steady_elapsed > 0 else 0:6.2f}"
            )

    elapsed = time.perf_counter() - t0_all
    print("---- summary ----")
    print(f"frames        = {n}")
    print(f"elapsed       = {elapsed:.2f} s")
    print(f"throughput    = {n / elapsed:.2f} fps")
    print(f"avg_decode    = {t_decode / n * 1000:.2f} ms")
    print(f"avg_matting   = {t_mat / n * 1000:.2f} ms")
    print(f"avg_preprocess= {t_pre / n:.2f} ms")
    print(f"avg_ort_run   = {t_ort / n:.2f} ms")
    print(f"avg_composite = {t_comp / n:.2f} ms")
    if steady_n > 0 and steady_start is not None:
        steady_elapsed = time.perf_counter() - steady_start
        print("---- steady ----")
        print(f"discarded     = {max(0, args.discard)}")
        print(f"steady_frames = {steady_n}")
        print(f"steady_elapsed= {steady_elapsed:.2f} s")
        print(f"steady_fps    = {steady_n / steady_elapsed:.2f} fps")
        print(f"steady_decode = {steady_decode / steady_n * 1000:.2f} ms")
        print(f"steady_matting= {steady_mat / steady_n * 1000:.2f} ms")
        print(f"steady_pre    = {steady_pre / steady_n:.2f} ms")
        print(f"steady_ort    = {steady_ort / steady_n:.2f} ms")
        print(f"steady_comp   = {steady_comp / steady_n:.2f} ms")
    dec.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
