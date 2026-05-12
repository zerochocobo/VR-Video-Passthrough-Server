"""Probe HEVC elementary video + source audio muxing outside the server path."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def _resolve_video(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config.VIDEO_DIR / path
    return path.resolve()


def _run(cmd: list[str], timeout: int = 120, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        timeout=timeout,
        check=False,
    )


def _probe_json(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration,size:stream=index,codec_type,codec_name,channels,sample_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return json.loads(result.stdout.decode("utf-8", "replace"))


def _has_audio(path: Path) -> bool:
    info = _probe_json(path)
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


def _generate_hevc_annexb(width: int, height: int, fps: float, duration: float) -> bytes:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={fps:.6f}:duration={duration:.3f}",
        "-an",
        "-c:v",
        "libx265",
        "-preset",
        "ultrafast",
        "-x265-params",
        "log-level=error:keyint=30:min-keyint=30:scenecut=0",
        "-f",
        "hevc",
        "-",
    ]
    result = _run(cmd, timeout=180)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


def _mux_with_audio(hevc_bytes: bytes, src: Path, out: Path, fps: float, duration: float, audio: str) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    codec = "copy" if audio == "copy" else "aac"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        "hevc",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        codec,
        "-t",
        f"{duration:.3f}",
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        str(out),
    ]
    result = _run(cmd, timeout=180, input_bytes=hevc_bytes)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))


def _extract_audio(src: Path, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(out),
    ]
    result = _run(cmd, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate audio muxing with generated HEVC video.")
    parser.add_argument("video")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--audio", choices=["copy", "aac"], default="copy")
    parser.add_argument("--out", default="debug_output/audio_mux_probe.mp4")
    args = parser.parse_args()

    src = _resolve_video(args.video)
    out = Path(args.out)
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    extracted = out.with_suffix(".audio.wav")

    if not _has_audio(src):
        raise RuntimeError(f"source has no audio stream: {src}")

    hevc = _generate_hevc_annexb(args.width, args.height, args.fps, args.duration)
    _mux_with_audio(hevc, src, out, args.fps, args.duration, args.audio)
    _extract_audio(out, extracted)

    out_info = _probe_json(out)
    audio_info = _probe_json(extracted)
    out_audio = [s for s in out_info.get("streams", []) if s.get("codec_type") == "audio"]
    extracted_audio = [s for s in audio_info.get("streams", []) if s.get("codec_type") == "audio"]
    if not out_audio or not extracted_audio:
        raise RuntimeError("audio stream validation failed")

    print(f"output={out}")
    print(f"extracted_audio={extracted}")
    print(f"output_streams={json.dumps(out_info.get('streams', []), ensure_ascii=False)}")
    print(f"extracted_streams={json.dumps(audio_info.get('streams', []), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
