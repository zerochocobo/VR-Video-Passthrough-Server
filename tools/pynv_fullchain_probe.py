"""
Phase 1.4 PyNv decode -> GPU matting -> PyNv encode -> FFmpeg mux probe.

Production server code is not touched. This validates whether the new GPU
backend can beat the existing FFmpeg rawvideo pipeline end to end.
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
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


def _open_muxer(out: Path, fps: float, src: Path, codec: str, audio: str, start_sec: float = 0.0):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    input_format = "hevc" if codec.lower() in {"hevc", "h265"} else "h264"
    audio_args: list[str] = ["-an"]
    if audio != "off":
        audio_args = [
            "-ss",
            f"{max(0.0, start_sec):.3f}",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:a",
            "copy" if audio == "copy" else "aac",
        ]
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        input_format,
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        *audio_args,
        "-c:v",
        "copy",
        *probe_color_metadata(src).ffmpeg_args(),
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        str(out),
    ]
    return cmd, subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _mux_file_with_audio(raw_video: Path, out: Path, fps: float, src: Path, codec: str, audio: str, start_sec: float = 0.0) -> tuple[int, str, str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    input_format = "hevc" if codec.lower() in {"hevc", "h265"} else "h264"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        input_format,
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(raw_video),
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy" if audio == "copy" else "aac",
        "-shortest",
        *probe_color_metadata(src).ffmpeg_args(),
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        str(out),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", timeout=60)
    return p.returncode, " ".join(cmd), (p.stdout + p.stderr)


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


def _summary(prefix: str, values: list[float]) -> list[str]:
    if not values:
        return [
            f"{prefix}_avg = 0.000 ms",
            f"{prefix}_p99 = 0.000 ms",
            f"{prefix}_std = 0.000 ms",
        ]
    return [
        f"{prefix}_avg = {statistics.fmean(values):.3f} ms",
        f"{prefix}_p99 = {_p99(values):.3f} ms",
        f"{prefix}_std = {statistics.pstdev(values) if len(values) > 1 else 0:.3f} ms",
    ]


def main() -> int:
    _patch_tempdir()
    parser = argparse.ArgumentParser(description="PyNv full GPU passthrough probe with matting.")
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--discard", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0, help="max output CFR framerate; <=0 keeps source CFR fps")
    parser.add_argument("--bitrate", default="20000000")
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--codec", default="h264", choices=["h264", "hevc", "h265"])
    parser.add_argument("--audio", default="off", choices=["off", "copy", "aac"])
    parser.add_argument("--raw-video-out", default="")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--alpha-stride", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--progress", type=int, default=30)
    args = parser.parse_args()

    config.MATTING_INPUT_SIZE = int(args.input_size)
    if args.no_warmup:
        config.MATTING_WARMUP_RUNS = 0
    import os

    os.environ["PT_ALPHA_STRIDE"] = str(max(1, args.alpha_stride))
    os.environ.setdefault("PT_MATTING_SPLIT_SBS", "1")
    os.environ.setdefault("PT_MATTING_SBS_BATCH", "1")

    import PyNvVideoCodec as nvc
    from pipeline.matting import Matter

    src = _resolve_video(args.video)
    dec = PyNvSimpleDecoder(src)
    info = dec.info
    timing = probe_timing_metadata(src)
    source_fps = float(timing.source_fps or info.fps or 30.0)
    cap_fps = float(args.fps or 0.0)
    fps = timing.effective_fps(cap_fps)
    start_out = int(round(max(0.0, args.start) * fps))
    if args.frames > 0:
        target = int(args.frames)
    else:
        duration = min(float(timing.duration or info.duration or args.duration), max(1.0, args.duration))
        target = int(max(1, duration * fps))
    max_target = int((len(dec) - 1) * fps / source_fps) + 1 if source_fps > 0 else len(dec)
    target = min(target, max(1, max_target))
    out = Path(args.out) if args.out else config.ROOT / "debug_output" / f"pynv_fullchain_s{args.alpha_stride}.mp4"
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    matter = Matter()
    matter.reset_state()
    enc = nvc.CreateEncoder(
        info.width,
        info.height,
        "NV12",
        False,
        codec=args.codec,
        bitrate=str(args.bitrate),
        fps=f"{fps:.6f}",
        gop=str(args.gop),
        bf="0",
    )
    raw_video = Path(args.raw_video_out) if args.raw_video_out else None
    if raw_video is not None:
        if not raw_video.is_absolute():
            raw_video = (config.ROOT / raw_video).resolve()
        raw_video.parent.mkdir(parents=True, exist_ok=True)
        mux = raw_video.open("wb")
        cmd = ["raw-video-file", str(raw_video)]
    else:
        cmd, mux = _open_muxer(out, fps, src, args.codec, args.audio, start_sec=args.start)
        assert mux.stdin is not None
    print(
        f"[pynv-full] src={src.name} {info.width}x{info.height} "
        f"source_fps={source_fps:.6f} cap_fps={cap_fps:.6f} effective_fps={fps:.6f} "
        f"is_cfr={timing.is_cfr} fps_diff={timing.fps_diff_ratio:.6f} target={target}"
    )
    if not timing.is_cfr:
        print("[pynv-full] WARNING: source is not strong CFR; PyNv raw-H264 mux path will synthesize CFR timestamps.")
    print(f"[pynv-full] stride={args.alpha_stride} discard={args.discard} mux={' '.join(cmd)}")

    t_dec: list[float] = []
    t_mat: list[float] = []
    t_pre: list[float] = []
    t_ort: list[float] = []
    t_comp: list[float] = []
    t_enc: list[float] = []
    t_mux: list[float] = []
    steady_dec: list[float] = []
    steady_mat: list[float] = []
    steady_pre: list[float] = []
    steady_ort: list[float] = []
    steady_comp: list[float] = []
    steady_enc: list[float] = []
    steady_mux: list[float] = []

    n_packets = 0
    bytes_written = 0
    first_src_idx = -1
    last_src_idx_seen = -1
    used0, total0 = _mem_stats()
    steady_start = None
    t0_all = time.perf_counter()

    try:
        last_src_idx = -1
        for i in range(target):
            out_idx = start_out + i
            src_idx = min(len(dec) - 1, cfr_source_index(out_idx, source_fps, fps))
            if src_idx <= last_src_idx:
                src_idx = min(len(dec) - 1, last_src_idx + 1)
            last_src_idx = src_idx
            if first_src_idx < 0:
                first_src_idx = src_idx
            last_src_idx_seen = src_idx
            td0 = time.perf_counter()
            frame = dec.frame_at(src_idx)
            td1 = time.perf_counter()
            out_nv12, timing = matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(frame)
            tm1 = time.perf_counter()
            app_frame = GpuNv12AppFrame(out_nv12, info.width, info.height)
            flags = 0
            if i == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            te0 = time.perf_counter()
            bs = enc.Encode(app_frame, flags) if flags else enc.Encode(app_frame)
            te1 = time.perf_counter()
            t_dec.append((td1 - td0) * 1000)
            t_mat.append((tm1 - td1) * 1000)
            t_pre.append(timing.preprocess_ms)
            t_ort.append(timing.ort_ms)
            t_comp.append(timing.composite_ms)
            t_enc.append((te1 - te0) * 1000)
            if bs:
                tw0 = time.perf_counter()
                mux.write(bs) if raw_video is not None else mux.stdin.write(bs)
                tw1 = time.perf_counter()
                t_mux.append((tw1 - tw0) * 1000)
                n_packets += 1
                bytes_written += len(bs)
            if i >= max(0, args.discard):
                if steady_start is None:
                    steady_start = td0
                steady_dec.append(t_dec[-1])
                steady_mat.append(t_mat[-1])
                steady_pre.append(t_pre[-1])
                steady_ort.append(t_ort[-1])
                steady_comp.append(t_comp[-1])
                steady_enc.append(t_enc[-1])
                if bs:
                    steady_mux.append(t_mux[-1])
            if args.progress > 0 and (i + 1) % args.progress == 0:
                elapsed = time.perf_counter() - t0_all
                steady_elapsed = (time.perf_counter() - steady_start) if steady_start else 0.0
                used, total = _mem_stats()
                print(
                    f"[pynv-full] {i + 1:5d}/{target} fps={(i + 1) / elapsed:7.2f} "
                    f"steady_fps={(len(steady_mat) / steady_elapsed) if steady_elapsed > 0 else 0:7.2f} "
                    f"mat={statistics.fmean(steady_mat) if steady_mat else 0:6.2f}ms "
                    f"ort={statistics.fmean(steady_ort) if steady_ort else 0:6.2f}ms "
                    f"comp={statistics.fmean(steady_comp) if steady_comp else 0:6.2f}ms "
                    f"enc={statistics.fmean(steady_enc) if steady_enc else 0:5.2f}ms "
                    f"mem={used:.1f}/{total:.1f}MB"
                )
        tail = enc.EndEncode()
        if tail:
            tw0 = time.perf_counter()
            mux.write(tail) if raw_video is not None else mux.stdin.write(tail)
            tw1 = time.perf_counter()
            t_mux.append((tw1 - tw0) * 1000)
            n_packets += 1
            bytes_written += len(tail)
    finally:
        if raw_video is not None:
            mux.close()
        else:
            mux.stdin.close()
    if raw_video is not None:
        stderr = ""
        if args.audio == "off":
            rc, mux_cmd, mux_stderr = _mux_file_with_audio(raw_video, out, fps, src, args.codec, "copy", start_sec=args.start)
        else:
            rc, mux_cmd, mux_stderr = _mux_file_with_audio(raw_video, out, fps, src, args.codec, args.audio, start_sec=args.start)
        stderr = f"[file-mux] {mux_cmd}\n{mux_stderr}"
    else:
        stderr = mux.stderr.read().decode("utf-8", "replace") if mux.stderr else ""
        rc = mux.wait(timeout=60)
    elapsed = time.perf_counter() - t0_all
    steady_elapsed = (time.perf_counter() - steady_start) if steady_start else 0.0
    used1, total1 = _mem_stats()

    print("---- summary ----")
    print(f"rc = {rc}")
    print(f"frames = {target}")
    print(f"source_index = {first_src_idx}..{last_src_idx_seen}")
    print(f"packets = {n_packets}")
    print(f"h264_bytes = {bytes_written}")
    print(f"elapsed = {elapsed:.3f} s")
    print(f"throughput = {target / elapsed:.2f} fps")
    for line in _summary("decode", t_dec): print(line)
    for line in _summary("matting", t_mat): print(line)
    for line in _summary("preprocess", t_pre): print(line)
    for line in _summary("ort", t_ort): print(line)
    for line in _summary("composite", t_comp): print(line)
    for line in _summary("encode", t_enc): print(line)
    for line in _summary("mux_write", t_mux): print(line)
    print(f"mem_start = {used0:.1f}/{total0:.1f} MB")
    print(f"mem_end = {used1:.1f}/{total1:.1f} MB")
    if steady_mat and steady_elapsed > 0:
        print("---- steady ----")
        print(f"discarded = {max(0, args.discard)}")
        print(f"steady_frames = {len(steady_mat)}")
        print(f"steady_elapsed = {steady_elapsed:.3f} s")
        print(f"steady_fps = {len(steady_mat) / steady_elapsed:.2f} fps")
        for line in _summary("steady_decode", steady_dec): print(line)
        for line in _summary("steady_matting", steady_mat): print(line)
        for line in _summary("steady_preprocess", steady_pre): print(line)
        for line in _summary("steady_ort", steady_ort): print(line)
        for line in _summary("steady_composite", steady_comp): print(line)
        for line in _summary("steady_encode", steady_enc): print(line)
        for line in _summary("steady_mux_write", steady_mux): print(line)
    if stderr.strip():
        print("[ffmpeg stderr]")
        print(stderr.strip()[-2000:])
    print("[ffprobe]")
    print(_ffprobe(out))
    dec.stop()
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
