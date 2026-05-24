"""Check first audio/video start-time delta in an MPEG-TS file.

This is an objective validation helper for live passthrough captures. It uses
ffprobe to read the first video and audio stream start times, then reports the
absolute and signed delta. A large delta means the muxed TS itself is
desynchronized, independent of player behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ffprobe_json import run_ffprobe_json  # noqa: E402


def _ffprobe_json(path: Path) -> dict:
    ffprobe = "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,start_time,duration,time_base,avg_frame_rate,sample_rate,channels,channel_layout,codec_name",
        "-show_entries",
        "format=start_time,duration,bit_rate,format_name",
        "-of",
        "json",
        str(path),
    ]
    return run_ffprobe_json(cmd)


def _stream_start(obj: dict, codec_type: str) -> float | None:
    for stream in obj.get("streams", []):
        if stream.get("codec_type") != codec_type:
            continue
        start = stream.get("start_time")
        if start in (None, ""):
            continue
        try:
            return float(start)
        except (TypeError, ValueError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MPEG-TS audio/video start-time delta.")
    parser.add_argument("ts", type=Path)
    parser.add_argument("--max-delta", type=float, default=0.10, help="allowed absolute delta in seconds")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.ts.exists():
        print(f"missing file: {args.ts}", file=sys.stderr)
        return 2

    obj = _ffprobe_json(args.ts)
    video_start = _stream_start(obj, "video")
    audio_start = _stream_start(obj, "audio")
    if video_start is None or audio_start is None:
        print("missing audio or video stream start_time", file=sys.stderr)
        return 2

    delta = audio_start - video_start
    record = {
        "file": str(args.ts),
        "video_start_sec": video_start,
        "audio_start_sec": audio_start,
        "audio_minus_video_sec": delta,
        "abs_delta_sec": abs(delta),
    }
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(
            f"audio_minus_video={delta:.3f}s "
            f"(video_start={video_start:.3f}s audio_start={audio_start:.3f}s)"
        )
    return 0 if abs(delta) <= args.max_delta else 1


if __name__ == "__main__":
    raise SystemExit(main())
