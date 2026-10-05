"""Render short real GPU effect VMP4s and verify decode plus nonzero seek.

Run from the repository root with a 2D source, for example:
  python -m tools.vmp4_effects_check --source videos/test_1080p_2d.mp4

Uses the production encoder, MP4 layout, padding, original-audio interleaving
and range iterator. Output is retained under debug_output for visual review.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--modes", default="face_beauty,two_dvr,rm")
    parser.add_argument("--subtitles", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("debug_output/vmp4_effects_check"))
    args = parser.parse_args()

    from utils.gpu_runtime_cache import configure_gpu_runtime_cache
    from utils.runtime_dll_paths import apply_runtime_dll_paths
    configure_gpu_runtime_cache()
    apply_runtime_dll_paths()
    import config
    from pipeline.ffmpeg_io import FFMPEG, FFPROBE, probe_cached
    from pipeline.matting import Matter
    from pipeline.pynv_stream import iter_pynv_passthrough_annexb_frames
    from pipeline.seek_effects import effect_settings
    from pipeline.passthrough_vmp4_frames import build_passthrough_vmp4_frames_layout, iter_vmp4_frames_range
    from pipeline.passthrough_vmp4_slot import hevc_annexb_to_length_prefixed_sample
    from pipeline.si_virtual_mp4 import read_media_sample_table
    from http_app.routes_media import _vmp4_slot_output_template

    config.PASSTHROUGH_GOP = 12
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "source.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(args.source), "-t", "2",
                    "-vf", "scale=640:360,fps=24", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "+faststart", str(source)], check=True)
    if args.subtitles:
        # Local smoke-test configuration only; never change saved UI settings.
        import pipeline.subtitles as subtitles
        subtitles.subtitle_output_enabled = lambda: True
        source.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nVMP4 START\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nVMP4 SEEK\n", encoding="utf-8")
    info = probe_cached(source)
    try:
        audio = read_media_sample_table(source, "audio")
    except Exception:
        audio = None
    report = []
    for mode in args.modes.split(","):
        started = time.monotonic()
        print(f"BEGIN {mode}", flush=True)
        template = _vmp4_slot_output_template(info, mode)
        layout = build_passthrough_vmp4_frames_layout(
            source, duration_sec=info.duration, fps=24, gop_frames=12,
            idr_budget=256 * 1024, p_budget=256 * 1024,
            output_stsd=template.stsd, output_width=template.width, output_height=template.height,
            placeholder_payload=template.payload, audio_table=audio)
        matter = Matter(load_model=False)
        ready = {}
        for start_frame in (0, 12):
            output_dir = args.out / f"{mode}_{start_frame}"
            output_dir.mkdir(exist_ok=True)
            encoded = 0
            for index, au in enumerate(iter_pynv_passthrough_annexb_frames(
                    source, start_sec=start_frame / 24, frame_count=layout.frame_count - start_frame,
                    matter=matter, output_mode=mode, per_frame_cap_bytes=256 * 1024,
                    processing_settings=effect_settings(mode, source))):
                payload = hevc_annexb_to_length_prefixed_sample(au, nal_length_size=layout.nal_length_size)
                if len(payload) > layout.frames[start_frame + index].budget:
                    raise RuntimeError(f"{mode} oversize frame {index}: {len(payload)}")
                target = output_dir / f"frame_{start_frame + index:06d}.bin"
                target.write_bytes(payload)
                ready[start_frame + index] = target
                encoded += 1
            if encoded != layout.frame_count - start_frame:
                raise RuntimeError(f"{mode} frame count {encoded} != {layout.frame_count - start_frame}")
            output = args.out / f"{mode}_{start_frame}.mp4"
            with output.open("wb") as fh:
                for chunk in iter_vmp4_frames_range(layout, 0, layout.total_size - 1, payload_paths=ready):
                    fh.write(chunk)
            for seek in (0, 0.5):
                result = subprocess.run([FFMPEG, "-v", "error", "-err_detect", "explode",
                                         "-ss", str(seek), "-i", str(output), "-f", "null", "-"],
                                        capture_output=True, text=True, errors="replace")
                if result.returncode or result.stderr.strip():
                    raise RuntimeError(f"{mode} decode/seek failed: {result.stderr[:2000]}")
            probe = subprocess.check_output([FFPROBE, "-v", "error", "-show_entries",
                "stream=codec_name,width,height,nb_frames,avg_frame_rate", "-of", "json", str(output)])
            report.append({"mode": mode, "start_frame": start_frame, "frames": encoded,
                           "subtitles": args.subtitles,
                           "output": str(output), "probe": json.loads(probe)})
        print(f"PASS {mode}: {layout.frame_count} frames; {template.width}x{template.height}; "
              f"decode and seek; {time.monotonic() - started:.1f}s", flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
