"""Feasibility proof for a frame-level VMP4 slot layout (real fps, plain MP4).

The 1-IDR-per-slot slot backend plays like a slideshow because each GOP slot is a
single MP4 sample with a multi-second stts. The proposed salvage keeps the plain
``ftyp+moov+mdat`` / single-moov / byte-stable design but stores EVERY frame as
its own sample, each padded with a filler NAL up to a FIXED per-frame byte budget
so the co64 offsets can be fixed before encoding.

This script proves the structural core offline and without the GPU:
  1. ffmpeg (libx265) makes a real CFR HEVC GOP clip.
  2. We re-pack every frame as a sample padded to a fixed budget with one HEVC
     filler NAL, assembled as a plain ftyp+moov+mdat with per-frame 1/fps stts.
  3. ffprobe must report the REAL frame rate / frame count, and ffmpeg must
     decode every frame cleanly.

If this passes, the frame-level fixed-budget layout is viable: real fps is
achievable in plain MP4, and the only remaining work is a background sequential
realtime encoder that fills the fixed frame slots ahead of the read cursor.

    uv run python -m tools.vmp4_frame_filler_proof
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from pipeline.ffmpeg_io import FFMPEG, FFPROBE
from pipeline.passthrough_vmp4_slot import (
    _build_slot_moov,
    _read_top_level_box,
    _source_video_stsd,
)
from pipeline.si_virtual_mp4 import read_media_sample_table


def _filler_nal(pad: int) -> bytes:
    # One 4-byte-length-prefixed HEVC filler-data NAL (type 38), total == pad.
    assert pad >= 8, pad
    inner = b"\x4c\x01" + b"\xff" * (pad - 7) + b"\x80"  # 2 hdr + body + stop == pad-4
    return (len(inner)).to_bytes(4, "big") + inner


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_frames,avg_frame_rate,time_base:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    return json.loads(out.stdout or "{}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--gop", type=int, default=30)
    ap.add_argument("--budget-margin", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("debug_output/vmp4_frame_filler_proof.mp4"))
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        gen = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i",
             f"testsrc2=s={args.width}x{args.height}:r={args.fps}:d={args.frames/args.fps:.4f}",
             "-c:v", "libx265", "-preset", "ultrafast",
             "-x265-params", f"log-level=error:keyint={args.gop}:min-keyint={args.gop}:scenecut=0:bframes=0",
             "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-movflags", "+faststart", str(clip)],
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            print("ffmpeg gen failed:", gen.stderr[-400:]); return 2

        info = _probe(clip)
        tb = "1/15360"
        for s in info.get("streams") or []:
            tb = str(s.get("time_base") or tb)
        timescale = int(tb.split("/")[1]) if "/" in tb else 15360
        delta = max(1, round(timescale / args.fps))

        moov_src = _read_top_level_box(clip, b"moov")
        ftyp = _read_top_level_box(clip, b"ftyp")
        stsd = _source_video_stsd(moov_src)
        table = read_media_sample_table(clip, "video")
        samples = list(table.samples)
        if not samples:
            print("no samples parsed"); return 2

        raw = []
        with clip.open("rb") as fh:
            for s in samples:
                fh.seek(s.source_offset)
                raw.append(fh.read(s.size))

        budget = max(len(b) for b in raw) + args.budget_margin
        n = len(raw)
        sizes = [budget] * n
        durations = [delta] * n
        media_duration = delta * n
        movie_duration_sec = n / float(args.fps)

        # Converge moov size (co64/stsz are fixed-width, so this settles fast).
        moov = _build_slot_moov(
            moov_src, output_stsd=stsd, offsets=[0] * n, sizes=sizes, durations=durations,
            media_duration=media_duration, movie_duration_sec=movie_duration_sec,
        )
        for _ in range(4):
            mdat_start = len(ftyp) + len(moov) + 8
            offsets = [mdat_start + i * budget for i in range(n)]
            new_moov = _build_slot_moov(
                moov_src, output_stsd=stsd, offsets=offsets, sizes=sizes, durations=durations,
                media_duration=media_duration, movie_duration_sec=movie_duration_sec,
            )
            if len(new_moov) == len(moov):
                moov = new_moov
                break
            moov = new_moov
        mdat_start = len(ftyp) + len(moov) + 8
        offsets = [mdat_start + i * budget for i in range(n)]

        payload = bytearray()
        for b in raw:
            payload += b
            payload += _filler_nal(budget - len(b))
        mdat = (len(payload) + 8).to_bytes(4, "big") + b"mdat" + bytes(payload)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(ftyp + moov + mdat)

        keyframes = sum(1 for s in samples if s.keyframe)
        print("=== frame-level filler MP4 proof ===")
        print(f"frames           : {n} ({keyframes} keyframes, gop={args.gop})")
        print(f"per-frame budget : {budget} B (max real frame {max(len(b) for b in raw)} B)")
        print(f"avg real frame   : {sum(len(b) for b in raw)//n} B  -> in-AU filler avg {budget - sum(len(b) for b in raw)//n} B")
        print(f"timescale/delta  : {timescale}/{delta}  (expect {args.fps} fps)")
        print(f"out file         : {args.out} ({args.out.stat().st_size} B)")

        out_info = _probe(args.out)
        st = (out_info.get("streams") or [{}])[0]
        print("\n--- ffprobe (rebuilt) ---")
        print(f"codec={st.get('codec_name')} {st.get('width')}x{st.get('height')} "
              f"nb_frames={st.get('nb_frames')} avg_frame_rate={st.get('avg_frame_rate')} "
              f"duration={(out_info.get('format') or {}).get('duration')}")

        dec = subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "error", "-i", str(args.out), "-f", "null", "-"],
            capture_output=True, text=True,
        )
        ok = dec.returncode == 0 and not dec.stderr.strip()
        if dec.stderr.strip():
            print("\nffmpeg decode stderr:\n" + dec.stderr.strip())
        rate = str(st.get("avg_frame_rate") or "")
        real_fps = rate.startswith(f"{args.fps}/") or rate == f"{args.fps}/1"
        print(f"\nRESULT: decode={'OK' if ok else 'FAIL'} real_fps={'OK' if real_fps else 'NO('+rate+')'} "
              f"frames={st.get('nb_frames')}")
        print("=> Frame-level fixed-budget plain MP4 is viable." if (ok and real_fps)
              else "=> Needs investigation before building the frame-level backend.")
        return 0 if (ok and real_fps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
