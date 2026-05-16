"""
Phase 1.3 PyNv decode -> PyNv encode -> FFmpeg mux probe.

This probe validates decoder surface lifecycle and pure GPU transcode behavior
without matting. It intentionally stays outside the production server.
"""
from __future__ import annotations

import argparse
import tempfile
import queue
import shutil
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.pynv_io import GpuNv12AppFrame, PyNvSimpleDecoder  # noqa: E402
from utils.video_metadata import cfr_source_index, probe_color_metadata, probe_timing_metadata  # noqa: E402


def _patch_tempdir() -> None:
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

    tempfile.TemporaryDirectory = FixedTemporaryDirectory


def _resolve_video(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = config.VIDEO_DIR / p
    return p.resolve()


def _open_muxer(out: Path, fps: float, src: Path, codec: str):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    input_format = "hevc" if codec.lower() in {"hevc", "h265"} else "h264"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        input_format,
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "copy",
        *probe_color_metadata(src).ffmpeg_args(),
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        str(out),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return cmd, p


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


def _copy_frame(frame):
    import cupy as cp

    y = cp.ascontiguousarray(frame.y.as_cupy())
    uv = cp.ascontiguousarray(frame.uv.as_cupy())
    nv12 = cp.empty((frame.height * 3 // 2, frame.width), dtype=cp.uint8)
    nv12[:frame.height, :] = y.reshape(frame.height, frame.width)
    nv12[frame.height:, :] = uv.reshape(frame.height // 2, frame.width)
    return GpuNv12AppFrame(nv12, frame.width, frame.height)


def _mem_stats() -> tuple[float, float]:
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / 1e6, pool.total_bytes() / 1e6
    except Exception:
        return 0.0, 0.0


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(len(values) * 0.99))]


def main() -> int:
    _patch_tempdir()
    parser = argparse.ArgumentParser(description="PyNv decoder -> encoder -> ffmpeg mux probe.")
    parser.add_argument("video", help="video filename under videos/ or absolute path")
    parser.add_argument("--frames", type=int, default=0, help="frame count; 0 means derive from --duration or source")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to process when --frames=0")
    parser.add_argument("--delay", type=int, default=0, help="hold decoded surfaces for K frames before encoding")
    parser.add_argument("--copy-out", action="store_true", help="copy decoded surface before delayed encode")
    parser.add_argument("--fps", type=float, default=30.0, help="max output CFR framerate; <=0 keeps source CFR fps")
    parser.add_argument("--bitrate", default="20000000")
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--codec", default="h264", choices=["h264", "hevc", "h265"])
    parser.add_argument("--preset", default="")
    parser.add_argument("--tuning-info", default="")
    parser.add_argument("--rc", default="")
    parser.add_argument("--idrperiod", default="")
    parser.add_argument(
        "--enc-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra PyNv CreateEncoder option; may be repeated",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--progress", type=int, default=30)
    args = parser.parse_args()

    import PyNvVideoCodec as nvc

    src = _resolve_video(args.video)
    dec = PyNvSimpleDecoder(src)
    info = dec.info
    timing = probe_timing_metadata(src)
    source_fps = float(timing.source_fps or info.fps or 30.0)
    cap_fps = float(args.fps or 0.0)
    fps = timing.effective_fps(cap_fps)
    if args.frames > 0:
        target = int(args.frames)
    else:
        duration = min(float(timing.duration or info.duration or args.duration), max(1.0, args.duration))
        target = int(max(1, duration * fps))
    max_target = int((len(dec) - 1) * fps / source_fps) + 1 if source_fps > 0 else len(dec)
    target = min(target, max(1, max_target))
    out = Path(args.out) if args.out else config.ROOT / "debug_output" / f"pynv_transcode_delay{args.delay}.mp4"
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    enc_kwargs = {
        "codec": args.codec,
        "bitrate": str(args.bitrate),
        "fps": f"{fps:.6f}",
        "gop": str(args.gop),
        "bf": "0",
    }
    if args.preset:
        enc_kwargs["preset"] = args.preset
    if args.tuning_info:
        enc_kwargs["tuning_info"] = args.tuning_info
    if args.rc:
        enc_kwargs["rc"] = args.rc
    if args.idrperiod:
        enc_kwargs["idrperiod"] = args.idrperiod
    for item in args.enc_opt:
        if "=" not in item:
            raise SystemExit(f"--enc-opt must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--enc-opt key is empty in {item!r}")
        enc_kwargs[key] = value.strip()
    enc = nvc.CreateEncoder(info.width, info.height, "NV12", False, **enc_kwargs)
    cmd, mux = _open_muxer(out, fps, src, args.codec)
    assert mux.stdin is not None
    print(
        f"[pynv-trans] src={src.name} {info.width}x{info.height} "
        f"source_fps={source_fps:.6f} cap_fps={cap_fps:.6f} effective_fps={fps:.6f} "
        f"is_cfr={timing.is_cfr} fps_diff={timing.fps_diff_ratio:.6f} target={target}"
    )
    print(f"[pynv-trans] encoder kwargs={enc_kwargs}")
    if not timing.is_cfr:
        print("[pynv-trans] WARNING: source is not strong CFR; PyNv raw-H264 mux path will synthesize CFR timestamps.")
    print(f"[pynv-trans] delay={args.delay} copy_out={args.copy_out} mux={' '.join(cmd)}")

    held = deque()
    n_dec = 0
    n_enc_in = 0
    n_packets = 0
    first_src_idx = -1
    last_src_idx_seen = -1
    bytes_written = 0
    t_dec: list[float] = []
    t_hold: list[float] = []
    t_enc: list[float] = []
    t_mux: list[float] = []
    used0, total0 = _mem_stats()
    t0_all = time.perf_counter()

    def encode_one(obj, idx: int) -> None:
        nonlocal n_enc_in, n_packets, bytes_written
        flags = 0
        if idx == 0:
            flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
        t0 = time.perf_counter()
        bs = enc.Encode(obj, flags) if flags else enc.Encode(obj)
        t1 = time.perf_counter()
        n_enc_in += 1
        t_enc.append((t1 - t0) * 1000)
        if bs:
            tm0 = time.perf_counter()
            mux.stdin.write(bs)
            tm1 = time.perf_counter()
            t_mux.append((tm1 - tm0) * 1000)
            n_packets += 1
            bytes_written += len(bs)

    try:
        last_src_idx = -1
        for i in range(target):
            src_idx = min(len(dec) - 1, cfr_source_index(i, source_fps, fps))
            if src_idx <= last_src_idx:
                src_idx = min(len(dec) - 1, last_src_idx + 1)
            last_src_idx = src_idx
            if first_src_idx < 0:
                first_src_idx = src_idx
            last_src_idx_seen = src_idx
            td0 = time.perf_counter()
            frame = dec.frame_at(src_idx)
            td1 = time.perf_counter()
            n_dec += 1
            t_dec.append((td1 - td0) * 1000)
            th0 = time.perf_counter()
            obj = _copy_frame(frame) if args.copy_out else frame.owner
            held.append((i, obj))
            if len(held) > max(0, args.delay):
                enc_idx, enc_obj = held.popleft()
                encode_one(enc_obj, enc_idx)
            th1 = time.perf_counter()
            t_hold.append((th1 - th0) * 1000)
            if args.progress > 0 and (i + 1) % args.progress == 0:
                elapsed = time.perf_counter() - t0_all
                used, total = _mem_stats()
                print(
                    f"[pynv-trans] {i + 1:5d}/{target} fps={(i + 1) / elapsed:7.2f} "
                    f"dec={statistics.fmean(t_dec):5.2f}ms enc={statistics.fmean(t_enc) if t_enc else 0:5.2f}ms "
                    f"mux={statistics.fmean(t_mux) if t_mux else 0:5.2f}ms mem={used:.1f}/{total:.1f}MB"
                )
        while held:
            enc_idx, enc_obj = held.popleft()
            encode_one(enc_obj, enc_idx)
        tail = enc.EndEncode()
        if tail:
            tm0 = time.perf_counter()
            mux.stdin.write(tail)
            tm1 = time.perf_counter()
            t_mux.append((tm1 - tm0) * 1000)
            n_packets += 1
            bytes_written += len(tail)
    finally:
        mux.stdin.close()
    stderr = mux.stderr.read().decode("utf-8", "replace") if mux.stderr else ""
    rc = mux.wait(timeout=60)
    elapsed = time.perf_counter() - t0_all
    used1, total1 = _mem_stats()

    print("---- summary ----")
    print(f"rc              = {rc}")
    print(f"frames_decoded  = {n_dec}")
    print(f"frames_encoded  = {n_enc_in}")
    print(f"source_index    = {first_src_idx}..{last_src_idx_seen}")
    print(f"packets         = {n_packets}")
    print(f"h264_bytes      = {bytes_written}")
    print(f"elapsed         = {elapsed:.3f} s")
    print(f"throughput      = {n_dec / elapsed:.2f} fps")
    print(f"avg_decode      = {statistics.fmean(t_dec):.3f} ms")
    print(f"p99_decode      = {_p99(t_dec):.3f} ms")
    print(f"std_decode      = {statistics.pstdev(t_dec) if len(t_dec) > 1 else 0:.3f} ms")
    print(f"avg_hold        = {statistics.fmean(t_hold):.3f} ms")
    print(f"avg_encode      = {statistics.fmean(t_enc):.3f} ms")
    print(f"p99_encode      = {_p99(t_enc):.3f} ms")
    print(f"std_encode      = {statistics.pstdev(t_enc) if len(t_enc) > 1 else 0:.3f} ms")
    print(f"avg_mux_write   = {statistics.fmean(t_mux) if t_mux else 0:.3f} ms")
    print(f"p99_mux_write   = {_p99(t_mux):.3f} ms")
    print(f"std_mux_write   = {statistics.pstdev(t_mux) if len(t_mux) > 1 else 0:.3f} ms")
    print(f"mem_start       = {used0:.1f}/{total0:.1f} MB")
    print(f"mem_end         = {used1:.1f}/{total1:.1f} MB")
    if stderr.strip():
        print("[ffmpeg stderr]")
        print(stderr.strip()[-2000:])
    print("[ffprobe]")
    print(_ffprobe(out))
    dec.stop()
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
