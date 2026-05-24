"""Check captured live TS audio content against the source timeline.

This complements check_mpegts_sync.py. Stream start_time deltas prove that the
container timestamps line up, but they do not prove that the audio content came
from the requested source time. This helper extracts mono PCM from a live
capture and from a source time window, then uses normalized correlation to find
where the capture audio actually matches the source.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(stderr or f"command failed rc={proc.returncode}: {' '.join(cmd)}")


def _extract_pcm(
    input_path: str,
    out_path: Path,
    *,
    start_sec: float | None,
    duration_sec: float,
    sample_rate: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.6f}"])
    cmd.extend(
        [
            "-i",
            input_path,
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-t",
            f"{duration_sec:.6f}",
            "-f",
            "s16le",
            str(out_path),
        ]
    )
    _run(cmd)


def _read_pcm(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int16).astype(np.float32)
    if data.size <= 0:
        raise RuntimeError(f"empty decoded audio: {path}")
    data -= float(data.mean())
    std = float(data.std()) or 1.0
    return data / std


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _best_offset(
    capture: np.ndarray,
    source: np.ndarray,
    *,
    coarse_step: int,
    refine_radius: int,
) -> tuple[float, int]:
    if capture.size > source.size:
        raise RuntimeError(
            f"capture audio is longer than source search window: capture={capture.size} source={source.size}"
        )
    coarse_step = max(1, int(coarse_step))
    refine_radius = max(0, int(refine_radius))
    best_corr = -1.0
    best_offset = 0
    limit = int(source.size - capture.size)
    cap_norm = np.linalg.norm(capture) or 1.0
    for offset in range(0, limit + 1, coarse_step):
        segment = source[offset : offset + capture.size]
        denom = float(cap_norm * (np.linalg.norm(segment) or 1.0))
        corr = float(np.dot(capture, segment) / denom)
        if corr > best_corr:
            best_corr = corr
            best_offset = offset
    lo = max(0, best_offset - refine_radius)
    hi = min(limit, best_offset + refine_radius)
    for offset in range(lo, hi + 1):
        corr = _norm_corr(capture, source[offset : offset + capture.size])
        if corr > best_corr:
            best_corr = corr
            best_offset = offset
    return best_corr, best_offset


def main() -> int:
    parser = argparse.ArgumentParser(description="Correlate live TS audio content against source audio.")
    parser.add_argument("capture", type=Path, help="captured MPEG-TS file")
    parser.add_argument("--source", required=True, help="source file path or HTTP URL")
    parser.add_argument("--source-start", type=float, required=True, help="requested source start time in seconds")
    parser.add_argument("--duration", type=float, default=5.0, help="capture audio duration to compare")
    parser.add_argument("--search-before", type=float, default=2.0, help="seconds before source-start to include")
    parser.add_argument("--search-after", type=float, default=2.0, help="seconds after compared duration to include")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-offset", type=float, default=0.10, help="allowed content offset from source-start")
    parser.add_argument("--min-corr", type=float, default=0.80, help="minimum acceptable normalized correlation")
    parser.add_argument("--coarse-step-samples", type=int, default=16)
    parser.add_argument("--refine-radius-samples", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.capture.exists():
        print(f"missing capture: {args.capture}", file=sys.stderr)
        return 2
    if args.duration <= 0:
        print("--duration must be positive", file=sys.stderr)
        return 2

    source_window_start = max(0.0, float(args.source_start) - max(0.0, float(args.search_before)))
    source_window_duration = (
        float(args.duration)
        + max(0.0, float(args.search_before))
        + max(0.0, float(args.search_after))
    )
    with tempfile.TemporaryDirectory(prefix="pt_avsync_") as tmp:
        tmpdir = Path(tmp)
        capture_pcm = tmpdir / "capture.pcm"
        source_pcm = tmpdir / "source.pcm"
        _extract_pcm(
            str(args.capture),
            capture_pcm,
            start_sec=None,
            duration_sec=float(args.duration),
            sample_rate=int(args.sample_rate),
        )
        _extract_pcm(
            str(args.source),
            source_pcm,
            start_sec=source_window_start,
            duration_sec=source_window_duration,
            sample_rate=int(args.sample_rate),
        )
        capture_audio = _read_pcm(capture_pcm)
        source_audio = _read_pcm(source_pcm)

    corr, offset_samples = _best_offset(
        capture_audio,
        source_audio,
        coarse_step=int(args.coarse_step_samples),
        refine_radius=int(args.refine_radius_samples),
    )
    matched_source_sec = source_window_start + (offset_samples / float(args.sample_rate))
    relative = matched_source_sec - float(args.source_start)
    record = {
        "capture": str(args.capture),
        "source": str(args.source),
        "source_start_sec": float(args.source_start),
        "matched_source_sec": matched_source_sec,
        "relative_to_source_start_sec": relative,
        "abs_offset_sec": abs(relative),
        "correlation": corr,
        "sample_rate": int(args.sample_rate),
        "capture_samples": int(capture_audio.size),
        "source_window_start_sec": source_window_start,
        "source_window_duration_sec": source_window_duration,
    }
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(
            f"audio_content_offset={relative:.3f}s corr={corr:.3f} "
            f"(matched_source={matched_source_sec:.3f}s expected={float(args.source_start):.3f}s)"
        )
    return 0 if abs(relative) <= float(args.max_offset) and corr >= float(args.min_corr) else 1


if __name__ == "__main__":
    raise SystemExit(main())
