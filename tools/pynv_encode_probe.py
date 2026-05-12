"""
Step 1.1 PyNvVideoCodec encoder probe.

This probe does not touch the production server. It creates synthetic NV12
frames on the GPU, encodes them with PyNvVideoCodec/NVENC, writes a raw H.264
Annex-B stream, and optionally asks ffprobe to parse the result.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.pynv_io import GpuNv12AppFrame  # noqa: E402


def _make_nv12_frame(width: int, height: int, frame_idx: int):
    import numpy as np

    frame = np.empty((height * 3 // 2, width), dtype=np.uint8)
    frame[:height, :].fill((64 + frame_idx * 2) & 0xFF)
    frame[height:, 0::2].fill(128)
    frame[height:, 1::2].fill(128)
    return frame


def _ffprobe(path: Path, width: int, height: int, fps: int) -> str:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "h264",
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-count_frames",
        "-of",
        "default=nw=1",
        str(path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    return (p.stdout + p.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode synthetic GPU NV12 frames with PyNvVideoCodec.")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="20000000")
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--codec", default="h264")
    parser.add_argument("--preset", default="")
    parser.add_argument("--bf", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--force-idr-first", action="store_true")
    parser.add_argument("--cpu-input", action="store_true", help="use PyNv CPU input buffer mode")
    args = parser.parse_args()

    import PyNvVideoCodec as nvc

    out = Path(args.out) if args.out else config.ROOT / "debug_output" / "pynv_encode_probe.h264"
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    enc_kwargs = {
        "codec": args.codec,
        "bitrate": str(args.bitrate),
        "fps": str(args.fps),
        "gop": str(args.gop),
        "bf": str(args.bf),
    }
    if args.preset:
        enc_kwargs["preset"] = args.preset

    print(f"[pynv-enc] create encoder {args.width}x{args.height} kwargs={enc_kwargs}")
    enc = nvc.CreateEncoder(args.width, args.height, "NV12", bool(args.cpu_input), **enc_kwargs)
    written = 0
    packets = 0
    packet_sizes: list[int] = []
    t0 = time.perf_counter()
    with out.open("wb") as f:
        for i in range(max(1, args.frames)):
            frame = _make_nv12_frame(args.width, args.height, i)
            if not args.cpu_input:
                import cupy as cp

                frame = GpuNv12AppFrame(cp.asarray(frame), args.width, args.height)
            flags = 0
            if i == 0 and args.force_idr_first:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            bitstream = enc.Encode(frame, flags) if flags else enc.Encode(frame)
            if bitstream:
                f.write(bitstream)
                packets += 1
                packet_sizes.append(len(bitstream))
                written += len(bitstream)
        tail = enc.EndEncode()
        if tail:
            f.write(tail)
            packets += 1
            packet_sizes.append(len(tail))
            written += len(tail)
    elapsed = time.perf_counter() - t0
    print(f"[pynv-enc] out={out}")
    print(f"[pynv-enc] frames={args.frames} packets={packets} bytes={written} elapsed={elapsed:.3f}s fps={args.frames / elapsed:.2f}")
    if packet_sizes:
        print(f"[pynv-enc] packet_min={min(packet_sizes)} packet_max={max(packet_sizes)} packet_avg={sum(packet_sizes) / len(packet_sizes):.1f}")
    print("[ffprobe]")
    print(_ffprobe(out, args.width, args.height, args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
