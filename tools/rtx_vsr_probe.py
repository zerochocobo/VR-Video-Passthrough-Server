"""Probe RTX VSR capability and optionally evaluate a generated RGBA frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.gpu_runtime_cache import configure_gpu_runtime_cache
from utils.runtime_dll_paths import apply_runtime_dll_paths
from utils.rtx_vsr import load_bridge, probe_capability, target_dimensions


def main() -> int:
    configure_gpu_runtime_cache()
    apply_runtime_dll_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--target-height", type=int, default=720)
    parser.add_argument("--quality", type=int, choices=[0, 1, 2, 3, 4], default=2)
    args = parser.parse_args()

    if not args.evaluate:
        capability = probe_capability(args.gpu)
        payload = {"available": capability.available, "reason": capability.reason, "runtime_dir": str(capability.runtime_dir or "")}
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if capability.available else 2

    # Do not call probe_capability first: it initializes NGX on a bridge-owned
    # CUDA context.  CuPy evaluation must initialize NGX directly on CuPy's
    # current context, otherwise the first CUDA allocation/copy can deadlock.
    try:
        bridge = load_bridge()
    except Exception as exc:
        print(json.dumps({"available": False, "reason": str(exc), "runtime_dir": ""}, ensure_ascii=False))
        return 2
    print(json.dumps({"available": True, "reason": "runtime_loaded", "runtime_dir": str(bridge.runtime_dir)}, ensure_ascii=False))
    sys.stdout.flush()

    out_w, out_h = target_dimensions(args.width, args.height, args.target_height)
    try:
        import cupy as cp
    except ImportError:
        # Keep the native SDK probe useful in a minimal Python environment.
        rgba = bytearray(args.width * args.height * 4)
        for y in range(args.height):
            for x in range(args.width):
                i = (y * args.width + x) * 4
                rgba[i:i + 4] = bytes((x & 255, y & 255, 127, 255))
        try:
            out_bytes = bridge.process_driver_rgba(rgba, (args.width, args.height), (out_w, out_h), args.quality)
            method = "cuda_deviceptr"
        except Exception:
            out_bytes = bridge.process_host_rgba(rgba, (args.width, args.height), (out_w, out_h), args.quality)
            method = "cuda_hostptr"
        checksum = sum(out_bytes)
    else:
        # Generate the diagnostic image on CPU.  CuPy elementwise assignment
        # can trigger a first-run NVRTC compile lasting minutes on a clean
        # Blackwell cache, which would incorrectly look like an NGX timeout.
        import numpy as np

        host = np.empty((args.height, args.width, 4), dtype=np.uint8)
        host[..., 0] = np.arange(args.width, dtype=np.uint16)[None, :] & 255
        host[..., 1] = np.arange(args.height, dtype=np.uint16)[:, None] & 255
        host[..., 2] = 127
        host[..., 3] = 255
        rgba = cp.asarray(host)
        out = bridge.process_cupy_rgba(rgba, (out_w, out_h), args.quality)
        # Keep checksum calculation on CPU so the probe does not depend on a
        # second NVRTC/CUB compilation after VSR has already succeeded.
        checksum = int(out.get().sum(dtype=np.uint64))
        method = "cupy_deviceptr"
    print(json.dumps({"input": [args.width, args.height], "output": [out_w, out_h], "quality": args.quality, "method": method, "checksum": checksum}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
