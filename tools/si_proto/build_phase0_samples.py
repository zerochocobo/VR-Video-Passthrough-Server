"""Generate Phase 0 static SI playback samples.

The generated MP4 files are intentionally written under debug_output by
default, so they are not committed. They are used for real-player testing of
whether progressive/fMP4 files get a seekable timeline in DLNA clients.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.ffmpeg_io import FFMPEG  # noqa: E402
from utils.si_filter import build_si_mix_filter  # noqa: E402


SAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A_progressive.mp4", ("+faststart",)),
    ("B_fmp4.mp4", ("+empty_moov+default_base_moof+frag_keyframe",)),
    ("C_fmp4_dash.mp4", ("+dash",)),
)


def run(cmd: list[str]) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_sample(video: Path, si_wav: Path, output: Path, seconds: float, movflags: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filt = build_si_mix_filter("both", 100, 100, 1.0, True)
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-t",
            f"{seconds:g}",
            "-i",
            str(video),
            "-i",
            str(si_wav),
            "-filter_complex",
            filt,
            "-map",
            "0:v",
            "-c:v",
            "copy",
            "-map",
            "[si_track]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            movflags,
            "-f",
            "mp4",
            str(output),
        ]
    )


def ffprobe_summary(path: Path) -> str:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return " | ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 0 SI static MP4 samples.")
    parser.add_argument("video", type=Path, help="Source MP4 video.")
    parser.add_argument("si_wav", type=Path, help="Same-language SI WAV sidecar.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Clip length in seconds.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "debug_output" / "si_proto")
    args = parser.parse_args()

    video = args.video
    si_wav = args.si_wav
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")
    if not si_wav.is_file():
        raise SystemExit(f"missing si wav: {si_wav}")

    for filename, flags in SAMPLES:
        output = args.out_dir / filename
        build_sample(video, si_wav, output, args.seconds, flags[0])
        print(f"[ok] {output} ({output.stat().st_size:,} bytes) {ffprobe_summary(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
