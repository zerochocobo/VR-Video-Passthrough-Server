"""
Minimal PyNvVideoCodec decode probe.

This is the first development step toward a future zero-copy backend. It decodes
NV12 frames into device memory and verifies that the CUDA planes can be wrapped
and read without going through FFmpeg rawvideo pipes.
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


def _resolve_video(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = config.VIDEO_DIR / p
    return p.resolve()


def _copy_sample(plane, count: int = 64) -> bytes:
    import cupy as cp
    import numpy as np

    arr = plane.as_cupy()
    out = np.empty((min(count, arr.size),), dtype=np.uint8)
    cp.cuda.runtime.memcpy(
        out.ctypes.data,
        arr.data.ptr,
        out.nbytes,
        cp.cuda.runtime.memcpyDeviceToHost,
    )
    return out.tobytes()


def main() -> int:
    _configure_local_temp()
    parser = argparse.ArgumentParser(description="Decode frames with PyNvVideoCodec into GPU NV12 planes.")
    parser.add_argument("video", help="video filename under videos/ or absolute path")
    parser.add_argument("--frames", type=int, default=120, help="number of indexed frames to decode")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    src = _resolve_video(args.video)
    dec = PyNvSimpleDecoder(src, gpu_id=args.gpu)
    info = dec.info
    print(
        f"[pynv] source={src.name} {info.width}x{info.height} "
        f"{info.fps:.3f}fps frames={info.num_frames} codec={info.codec_name} bitrate={info.bitrate:.0f}"
    )

    n = min(max(1, args.frames), max(1, len(dec)))
    t0 = time.perf_counter()
    first = None
    for i in range(n):
        frame = dec.frame_at(i)
        if first is None:
            first = frame
            y_sample = _copy_sample(frame.y)
            uv_sample = _copy_sample(frame.uv)
            print(
                "[pynv] first "
                f"pts={frame.pts} y_shape={frame.y.shape} y_strides={frame.y.strides} "
                f"y_ptr=0x{frame.y.ptr:x} uv_shape={frame.uv.shape} uv_strides={frame.uv.strides} "
                f"uv_ptr=0x{frame.uv.ptr:x}"
            )
            print(f"[pynv] y_sample={y_sample[:16].hex()} uv_sample={uv_sample[:16].hex()}")
    elapsed = time.perf_counter() - t0
    print(f"[pynv] decoded={n} elapsed={elapsed:.3f}s throughput={n / elapsed:.2f} fps")
    dec.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
