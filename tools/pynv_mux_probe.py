"""
Step 1.2 PyNvVideoCodec encoder + FFmpeg mux probe.

Encodes synthetic GPU NV12 frames with PyNvVideoCodec and streams the H.264
Annex-B bytes into an FFmpeg subprocess that only muxes with `-c copy`.
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
from tools.pynv_encode_probe import _make_nv12_frame  # noqa: E402
from utils.video_metadata import probe_color_metadata  # noqa: E402


def _ffprobe(path: Path) -> str:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames,color_space,color_range,color_transfer,color_primaries",
        "-of",
        "default=nw=1",
        str(path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    return (p.stdout + p.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode GPU NV12 with PyNv and mux H.264 through FFmpeg.")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="20000000")
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--out", default="")
    parser.add_argument("--color-from", default="", help="source video to copy color metadata from")
    args = parser.parse_args()

    import cupy as cp
    import PyNvVideoCodec as nvc

    out = Path(args.out) if args.out else config.ROOT / "debug_output" / "pynv_mux_probe.mp4"
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    color_args = probe_color_metadata(Path(args.color_from)).ffmpeg_args() if args.color_from else []

    enc = nvc.CreateEncoder(
        args.width,
        args.height,
        "NV12",
        False,
        codec="h264",
        bitrate=str(args.bitrate),
        fps=str(args.fps),
        gop=str(args.gop),
        bf="0",
    )
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "h264",
        "-framerate",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "copy",
        *color_args,
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        str(out),
    ]
    print("[pynv-mux] " + subprocess.list2cmdline(cmd))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    packets = 0
    bytes_written = 0
    t0 = time.perf_counter()
    assert p.stdin is not None
    try:
        for i in range(max(1, args.frames)):
            host = _make_nv12_frame(args.width, args.height, i)
            dev = cp.asarray(host)
            frame = GpuNv12AppFrame(dev, args.width, args.height)
            flags = 0
            if i == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            bitstream = enc.Encode(frame, flags) if flags else enc.Encode(frame)
            if bitstream:
                p.stdin.write(bitstream)
                packets += 1
                bytes_written += len(bitstream)
        tail = enc.EndEncode()
        if tail:
            p.stdin.write(tail)
            packets += 1
            bytes_written += len(tail)
    finally:
        p.stdin.close()
    stderr = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
    rc = p.wait(timeout=60)
    elapsed = time.perf_counter() - t0
    print(f"[pynv-mux] rc={rc} out={out}")
    print(f"[pynv-mux] frames={args.frames} packets={packets} h264_bytes={bytes_written} elapsed={elapsed:.3f}s fps={args.frames / elapsed:.2f}")
    if stderr.strip():
        print("[ffmpeg stderr]")
        print(stderr.strip()[-2000:])
    print("[ffprobe]")
    print(_ffprobe(out))
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
