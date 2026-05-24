"""Probe MPEG-TS audio mux timestamp behavior for raw HEVC stdin.

This tool feeds an existing HEVC Annex-B elementary stream into FFmpeg in small
chunks, optionally paced like the live server, while muxing source audio into
MPEG-TS. It reports startup bytes, total bytes, stderr warnings, and stream
packet timestamp summaries so live audio fixes can be tested outside production.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.ffprobe_json import run_ffprobe_json  # noqa: E402


def _resolve_video(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config.VIDEO_DIR / path
    return path.resolve()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config.ROOT / path
    return path.resolve()


def _run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _ffprobe_packets(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,dts_time,duration_time,flags",
        "-of",
        "json",
        str(path),
    ]
    try:
        packets = run_ffprobe_json(cmd, timeout=60.0).get("packets", [])
    except RuntimeError as exc:
        return {"error": str(exc).strip()}
    dts_values: list[float] = []
    pts_values: list[float] = []
    regressions = 0
    last_dts: float | None = None
    for pkt in packets[:300]:
        try:
            dts = float(pkt.get("dts_time"))
            dts_values.append(dts)
            if last_dts is not None and dts < last_dts:
                regressions += 1
            last_dts = dts
        except (TypeError, ValueError):
            pass
        try:
            pts_values.append(float(pkt.get("pts_time")))
        except (TypeError, ValueError):
            pass
    return {
        "packets_checked": min(len(packets), 300),
        "dts_regressions": regressions,
        "first_dts": dts_values[:5],
        "first_pts": pts_values[:5],
        "last_dts": dts_values[-5:],
    }


def _build_cmd(args: argparse.Namespace, out: Path | None) -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
    ]
    if args.fflags:
        cmd += ["-fflags", args.fflags]
    if args.wallclock:
        cmd += ["-use_wallclock_as_timestamps", "1"]
    if args.re_input:
        cmd += ["-re"]
    if args.video_queue_size > 0:
        cmd += ["-thread_queue_size", str(args.video_queue_size)]
    cmd += [
        "-f",
        "hevc",
        "-framerate",
        f"{args.fps:.6f}",
        "-i",
        "-",
    ]
    if args.audio_delay:
        cmd += ["-itsoffset", f"{args.audio_delay:.6f}"]
    if args.start > 0.001:
        cmd += ["-ss", f"{args.start:.3f}"]
    if args.audio_readrate > 0:
        cmd += ["-readrate", f"{args.audio_readrate:.6f}"]
    if args.audio_queue_size > 0:
        cmd += ["-thread_queue_size", str(args.audio_queue_size)]
    cmd += [
        "-t",
        f"{args.duration:.3f}",
        "-i",
        str(args.video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        args.audio,
    ]
    if args.audio_bitrate:
        cmd += ["-b:a", args.audio_bitrate]
    if args.video_bsf:
        cmd += ["-bsf:v", args.video_bsf]
    if args.copyts:
        cmd += ["-copyts", "-start_at_zero"]
    if args.output_t:
        cmd += ["-t", f"{args.duration:.3f}"]
    if args.muxrate:
        cmd += ["-muxrate", str(args.muxrate)]
    if args.max_interleave_delta != "":
        cmd += ["-max_interleave_delta", str(args.max_interleave_delta)]
    cmd += [
        "-flush_packets",
        "1",
        "-muxdelay",
        str(args.muxdelay),
        "-muxpreload",
        str(args.muxpreload),
        "-mpegts_flags",
        "+resend_headers",
        "-pat_period",
        "0.1",
        "-sdt_period",
        "0.5",
        "-pcr_period",
        "20",
        "-f",
        "mpegts",
        "-" if out is None else str(out),
    ]
    return cmd


def _feed(
    proc: subprocess.Popen,
    hevc: Path,
    chunk: int,
    pace: float,
    duration: float,
    close_stdin: bool,
) -> tuple[int, str]:
    assert proc.stdin is not None
    total = 0
    start = time.perf_counter()
    try:
        with hevc.open("rb") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                if duration > 0 and time.perf_counter() - start > duration:
                    break
                proc.stdin.write(data)
                proc.stdin.flush()
                total += len(data)
                if pace > 0:
                    time.sleep(pace)
        if close_stdin:
            proc.stdin.close()
        return total, ""
    except Exception as e:
        try:
            proc.stdin.close()
        except Exception:
            pass
        return total, str(e)


def run_variant(args: argparse.Namespace, name: str) -> dict:
    out = None if args.stdout else args.out_dir / f"{name}.ts"
    cmd = _build_cmd(args, out)
    stderr_chunks: list[str] = []
    stdout_bytes = 0
    stdout_first_at: float | None = None
    stdout_last_at: float | None = None
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE if args.stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )
    start = time.perf_counter()

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line.decode("utf-8", "replace").rstrip())

    def read_stdout() -> None:
        nonlocal stdout_bytes, stdout_first_at, stdout_last_at
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(args.read_chunk)
            if not data:
                break
            now = time.perf_counter() - start
            if stdout_first_at is None:
                stdout_first_at = now
            stdout_last_at = now
            stdout_bytes += len(data)

    stderr_t = threading.Thread(target=read_stderr, daemon=True)
    stderr_t.start()
    stdout_t: threading.Thread | None = None
    if args.stdout:
        stdout_t = threading.Thread(target=read_stdout, daemon=True)
        stdout_t.start()
    fed, feed_error = _feed(proc, args.hevc, args.chunk, args.pace, args.feed_seconds, args.close_stdin)
    if not args.close_stdin and args.hold_open > 0:
        time.sleep(args.hold_open)
        try:
            proc.stdin.close()
        except Exception:
            pass
    try:
        rc = proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait(timeout=10)
    stderr_t.join(timeout=2)
    if stdout_t is not None:
        stdout_t.join(timeout=2)
    warnings = "\n".join(stderr_chunks)
    out_exists = out is not None and out.exists() and out.stat().st_size > 0
    return {
        "name": name,
        "returncode": rc,
        "fed_bytes": fed,
        "feed_error": feed_error,
        "out": "<stdout>" if out is None else str(out),
        "out_bytes": stdout_bytes if out is None else (out.stat().st_size if out.exists() else 0),
        "stdout_first_at": stdout_first_at,
        "stdout_last_at": stdout_last_at,
        "non_monotonic": "Non-monotonic" in warnings,
        "stderr": warnings[-4000:],
        "packets": _ffprobe_packets(out) if out_exists else {},
        "cmd": " ".join(cmd),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe MPEG-TS AAC timestamp variants.")
    parser.add_argument("video")
    parser.add_argument("--hevc", default="debug_output/test8k_fullchain_120.annexb.hevc")
    parser.add_argument("--out-dir", default="debug_output/mpegts_audio_pts_probe")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--audio", choices=["aac", "copy"], default="aac")
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--pace", type=float, default=0.01)
    parser.add_argument("--feed-seconds", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--fflags", default="+genpts")
    parser.add_argument("--wallclock", action="store_true")
    parser.add_argument("--re-input", action="store_true")
    parser.add_argument("--copyts", action="store_true")
    parser.add_argument("--output-t", action="store_true")
    parser.add_argument("--audio-delay", type=float, default=0.0)
    parser.add_argument("--audio-readrate", type=float, default=0.0)
    parser.add_argument("--video-queue-size", type=int, default=0)
    parser.add_argument("--audio-queue-size", type=int, default=0)
    parser.add_argument("--video-bsf", default="")
    parser.add_argument("--audio-bitrate", default="")
    parser.add_argument("--max-interleave-delta", default="0")
    parser.add_argument("--muxdelay", default="0")
    parser.add_argument("--muxpreload", default="0")
    parser.add_argument("--muxrate", default="")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--read-chunk", type=int, default=262144)
    parser.add_argument("--close-stdin", action="store_true")
    parser.add_argument("--hold-open", type=float, default=0.0)
    args = parser.parse_args()

    args.video = _resolve_video(args.video)
    args.hevc = _resolve_path(args.hevc)
    args.out_dir = _resolve_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.hevc.exists() or args.hevc.stat().st_size <= 0:
        raise FileNotFoundError(args.hevc)

    result = run_variant(args, "variant")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["out_bytes"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
