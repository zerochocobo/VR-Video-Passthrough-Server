"""Command-line utility for inspecting and building GPU runtime caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or inspect PTMediaServer GPU runtime caches.")
    parser.add_argument("--force", action="store_true", help="run warmup even if marker matches")
    parser.add_argument("--check-only", action="store_true", help="print current warmup key and marker status")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--runs-per-shape", type=int, default=3)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--split-sbs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.providers:
        os.environ["PT_ONNX_PROVIDERS"] = args.providers
    if args.split_sbs:
        os.environ["PT_MATTING_SPLIT_SBS"] = "1"
        os.environ["PT_MATTING_SBS_BATCH"] = "1"

    from utils.gpu_runtime_cache import (
        MARKER_PATH,
        build_warmup_key,
        configure_gpu_runtime_cache,
        marker_matches,
        warmup_gpu_runtime_cache,
    )

    env = configure_gpu_runtime_cache()
    key = build_warmup_key()
    print("[gpu-cache] env=" + json.dumps(asdict(env), ensure_ascii=False))
    print("[gpu-cache] key=" + json.dumps(asdict(key), ensure_ascii=False))
    marker_path = Path(env.marker_path)
    print(f"[gpu-cache] marker={marker_path} matches={marker_matches(key, marker_path)}")
    if args.check_only:
        return 0 if marker_matches(key, marker_path) else 2

    print("[gpu-cache] warmup start")
    start = time.perf_counter()
    try:
        marker = warmup_gpu_runtime_cache(
            force=args.force,
            timeout_sec=max(1.0, args.timeout),
            runs_per_shape=max(1, args.runs_per_shape),
        )
    except TimeoutError as exc:
        print(f"[gpu-cache] warmup timeout: {exc}", file=sys.stderr)
        return 124
    elapsed = time.perf_counter() - start
    print("[gpu-cache] marker=" + json.dumps(asdict(marker), indent=2, ensure_ascii=False))
    print(f"[gpu-cache] warmup elapsed={elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
