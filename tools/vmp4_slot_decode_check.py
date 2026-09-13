"""Offline decodability check for the /passthrough_seek VMP4 slot layout.

Why this exists
---------------
Quest players (Skybox / 4XVR) kept failing on the slot backend even though the
PyNv builder was producing valid HEVC. The open suspicion is structural, not
HTTP: every virtual MP4 *sample* is declared at ``slot_size`` and padded up to
that size with one big HEVC filler NAL *inside the access unit*. Mobile HEVC
hardware decoders have per-access-unit input limits, so a sample that is mostly
filler can be rejected at decode-init even when ffmpeg's tolerant software
decoder accepts it.

This tool builds the *real* slot layout for a real source, fills the first N
slots with a real HEVC IDR access unit (from a generated template, or from a
``--payload-file`` you point at an already-built ``slot_XXXXXX.bin``), serves the
first N slots through the exact production range iterator, and then runs
``ffprobe`` + ``ffmpeg`` decode on the resulting bytes.

Interpretation
--------------
- If ffmpeg (software, tolerant) cannot cleanly decode the dumped file, Quest's
  stricter hardware decoder never will -> the filler-in-access-unit structure
  must change (smaller ``slot_size`` / no in-AU filler / frame-level samples)
  before any further device testing.
- If ffmpeg decodes cleanly, the container structure is at least valid and the
  remaining failure is elsewhere (codec params, cadence, pacing).

This is a structural check. It does not validate byte<->time seek mapping, and it
does not need the GPU: the template payload is generated with ffmpeg (libx265 or
hevc), and ``--payload-file`` lets you inject a real PyNv slot for a faithful
content test.

Usage
-----
    uv run python -m tools.vmp4_slot_decode_check --source videos/72461_3840p.mp4 \
        --slots 5 --out debug_output/vmp4_slot_decode_check.mp4

    # Faithful content test using a real built slot payload:
    uv run python -m tools.vmp4_slot_decode_check --source videos/72461_3840p.mp4 \
        --payload-file runtime_cache/vmp4_slot/<digest>/slot_000000.bin
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.ffmpeg_io import FFMPEG, FFPROBE
from pipeline.passthrough_vmp4_slot import (
    Vmp4SlotLayoutError,
    build_passthrough_vmp4_slot_layout,
    iter_vmp4_slot_range,
    load_vmp4_slot_output_template,
)


def _probe_duration_fps(source: Path) -> tuple[float, float]:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", str(source),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(out.stdout or "{}")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    fps = 0.0
    for stream in data.get("streams") or []:
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = str(stream.get(key) or "").strip()
            if "/" in raw:
                num, _, den = raw.partition("/")
                try:
                    n, d = float(num), float(den)
                    if d > 0 and n > 0:
                        fps = n / d
                        break
                except ValueError:
                    continue
        if fps > 0:
            break
    return duration, fps


def _generate_hevc_template(out_mp4: Path, width: int, height: int, fps: float) -> None:
    base = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={int(width)}x{int(height)}:r={max(1.0, fps):.6f}:d=1",
        "-frames:v", "1", "-an", "-pix_fmt", "yuv420p",
        "-g", "1", "-bf", "0", "-tag:v", "hvc1", "-movflags", "+faststart",
    ]
    attempts = (
        [*base, "-c:v", "libx265", "-preset", "ultrafast",
         "-x265-params", "log-level=error:keyint=1:min-keyint=1:scenecut=0", str(out_mp4)],
        [*base, "-c:v", "hevc", str(out_mp4)],
    )
    last = ""
    for cmd in attempts:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and out_mp4.is_file() and out_mp4.stat().st_size > 0:
            return
        last = (proc.stderr or "").strip()[-400:]
    raise RuntimeError(f"failed to generate HEVC template: {last or 'unknown'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="real source mp4 (provides the slot layout's source moov)")
    parser.add_argument("--out", type=Path, default=Path("debug_output/vmp4_slot_decode_check.mp4"))
    parser.add_argument("--slots", type=int, default=5, help="number of leading slots to materialize and decode")
    parser.add_argument("--width", type=int, default=7680, help="output width advertised in stsd (default mirrors 8K alpha)")
    parser.add_argument("--height", type=int, default=3840)
    parser.add_argument("--fps", type=float, default=0.0, help="override fps (default: probe source)")
    parser.add_argument("--gop-frames", type=int, default=60)
    parser.add_argument("--max-sample-bytes", type=int, default=512 * 1024, help="slot_size cap (the per-AU stsz size)")
    parser.add_argument("--payload-file", type=Path, default=None, help="use this real slot_XXXXXX.bin as every slot payload")
    args = parser.parse_args(argv)

    source = args.source.resolve()
    if not source.is_file():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        return 2

    duration, probed_fps = _probe_duration_fps(source)
    fps = args.fps if args.fps > 0 else (probed_fps or 30.0)
    if duration <= 0:
        print("ERROR: could not probe source duration", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        template_mp4 = tmp_dir / "template.mp4"
        _generate_hevc_template(template_mp4, args.width, args.height, fps)
        template = load_vmp4_slot_output_template(template_mp4)

        try:
            layout = build_passthrough_vmp4_slot_layout(
                source,
                duration_sec=duration,
                fps=fps,
                total_size=source.stat().st_size,
                gop_frames=args.gop_frames,
                max_sample_bytes=args.max_sample_bytes,
                output_stsd=template.stsd,
                output_codec_name=template.codec_name,
                output_width=template.width,
                output_height=template.height,
                placeholder_payload=template.payload,
            )
        except Vmp4SlotLayoutError as exc:
            print(f"ERROR: layout build failed: {exc}", file=sys.stderr)
            return 2

        n = max(1, min(int(args.slots), layout.slot_count))
        payload_bytes = (
            args.payload_file.read_bytes() if args.payload_file and args.payload_file.is_file() else template.payload
        )
        if len(payload_bytes) > layout.slot_size:
            print(
                f"ERROR: payload ({len(payload_bytes)} B) exceeds slot_size ({layout.slot_size} B); "
                f"raise --max-sample-bytes or this slot would be a permanent oversize failure.",
                file=sys.stderr,
            )
            return 2

        # Materialize the first N slots as "ready" payloads.
        cache_dir = tmp_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ready: dict[int, Path] = {}
        for i in range(n):
            p = cache_dir / f"slot_{i:06d}.bin"
            p.write_bytes(payload_bytes)
            ready[i] = p

        # Dump exactly the init + first N slots (truncated virtual file).
        dump_end = layout.mdat_payload_start + n * layout.slot_stride - 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb") as fh:
            for chunk in iter_vmp4_slot_range(layout, 0, dump_end, payload_paths=ready):
                fh.write(chunk)

        filler_per_au = max(0, layout.slot_size - len(payload_bytes))
        print("=== VMP4 slot decode check ===")
        print(f"source         : {source.name}")
        print(f"duration/fps   : {duration:.3f}s @ {fps:.3f}fps")
        print(f"output stsd    : {args.width}x{args.height} hevc nal_len={template.nal_length_size}")
        print(f"slot_count     : {layout.slot_count} (dumping first {n})")
        print(f"slot_size (AU) : {layout.slot_size} B  <- decoder reads this per sample")
        print(f"slot_stride    : {layout.slot_stride} B")
        print(f"slot_gap       : {layout.slot_gap_size} B  <- unreferenced (not in stsz)")
        print(f"payload bytes  : {len(payload_bytes)} B ({'real ' + args.payload_file.name if args.payload_file else 'template IDR'})")
        print(f"in-AU filler   : {filler_per_au} B per sample")
        print(f"slot_duration  : {layout.slot_duration_sec:.3f}s  (effective ~{1.0/max(0.001, layout.slot_duration_sec):.2f} fps)")
        print(f"out file       : {args.out}  ({args.out.stat().st_size} B)")

        # ffprobe
        print("\n--- ffprobe ---")
        probe = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries",
             "stream=codec_name,width,height,nb_frames:format=duration",
             "-of", "default=noprint_wrappers=1", str(args.out)],
            capture_output=True, text=True,
        )
        print(probe.stdout.strip() or "(no stream info)")
        if probe.stderr.strip():
            print("ffprobe stderr:", probe.stderr.strip())

        # ffmpeg decode of the first N frames
        print("\n--- ffmpeg decode (first N frames) ---")
        decode = subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "error", "-i", str(args.out),
             "-frames:v", str(n), "-f", "null", "-"],
            capture_output=True, text=True,
        )
        ok = decode.returncode == 0 and not decode.stderr.strip()
        if decode.stderr.strip():
            print(decode.stderr.strip())
        print(f"\nRESULT: {'DECODE OK' if ok else 'DECODE FAILED / WARNINGS'} (ffmpeg rc={decode.returncode})")
        if not ok:
            print("=> Software decode is unhappy; Quest hardware decode will not be better. "
                  "Treat the filler-in-access-unit structure as the blocker.")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
