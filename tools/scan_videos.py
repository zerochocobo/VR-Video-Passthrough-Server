"""
Scan local video files and classify future PyNv backend compatibility.

This is an offline planning tool. It does not alter the production server path.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.video_metadata import BackendDecision, probe_video_metadata, select_backend  # noqa: E402

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".webm",
    ".ts",
    ".m2ts",
    ".mts",
}


def _iter_videos(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    yield path
            except OSError:
                continue


def _strict_decode(path: Path) -> tuple[bool, str]:
    try:
        from pipeline.pynv_io import PyNvSimpleDecoder

        dec = PyNvSimpleDecoder(path)
        try:
            dec.frame_at(0)
        finally:
            dec.stop()
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:180]}"


def _bucket_fps(fps: float) -> str:
    if fps <= 0:
        return "unknown"
    if fps < 25:
        return "<25"
    if fps < 31:
        return "25-30"
    if fps < 61:
        return "31-60"
    return ">60"


def _row_for(
    path: Path,
    meta,
    decision: BackendDecision,
    strict: bool,
    strict_msg: str,
    roots: list[Path],
) -> dict[str, str]:
    rel = str(path)
    for root in roots:
        try:
            rel = str(path.relative_to(root))
            rel = str(root / rel)
            break
        except ValueError:
            pass
    return {
        "path": str(path),
        "display_path": rel,
        "verdict": decision.verdict,
        "reason": decision.reason,
        "strict_decode": "pass" if strict else "fail" if strict_msg else "",
        "strict_message": strict_msg,
        "codec": meta.codec.codec_name,
        "profile": meta.codec.profile,
        "level": str(meta.codec.level),
        "pix_fmt": meta.codec.pix_fmt,
        "bit_depth": str(meta.codec.bit_depth),
        "width": str(meta.codec.width),
        "height": str(meta.codec.height),
        "resolution_bucket": meta.codec.resolution_bucket,
        "fps": f"{meta.timing.source_fps:.6f}",
        "r_frame_rate": meta.timing.r_frame_rate,
        "avg_frame_rate": meta.timing.avg_frame_rate,
        "is_cfr": str(meta.timing.is_cfr),
        "fps_diff_ratio": f"{meta.timing.fps_diff_ratio:.6f}",
        "duration": f"{meta.timing.duration:.3f}",
        "nb_frames": str(meta.timing.nb_frames),
        "color_range": meta.color.color_range,
        "color_space": meta.color.color_space,
        "color_transfer": meta.color.color_transfer,
        "color_primaries": meta.color.color_primaries,
        "audio_codec": meta.codec.audio_codec,
        "audio_profile": meta.codec.audio_profile,
    }


def _write_summary(rows: list[dict[str, str]], out_summary: Path, elapsed: float) -> None:
    counters = {
        "verdict": Counter(r["verdict"] for r in rows),
        "codec": Counter(r["codec"] or "unknown" for r in rows),
        "pix_fmt": Counter(r["pix_fmt"] or "unknown" for r in rows),
        "resolution": Counter(r["resolution_bucket"] for r in rows),
        "fps_bucket": Counter(_bucket_fps(float(r["fps"] or 0.0)) for r in rows),
        "color_transfer": Counter(r["color_transfer"] or "unknown" for r in rows),
        "color_primaries": Counter(r["color_primaries"] or "unknown" for r in rows),
        "audio_codec": Counter(r["audio_codec"] or "none" for r in rows),
    }
    data = {
        "scanned": len(rows),
        "elapsed_sec": round(elapsed, 3),
        "histograms": {name: dict(counter.most_common()) for name, counter in counters.items()},
    }
    out_summary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_baseline(rows: list[dict[str, str]], summary_path: Path, csv_path: Path, baseline_path: Path, elapsed: float) -> None:
    verdicts = Counter(r["verdict"] for r in rows)
    codecs = Counter(r["codec"] or "unknown" for r in rows)
    resolutions = Counter(r["resolution_bucket"] for r in rows)
    pix_fmts = Counter(r["pix_fmt"] or "unknown" for r in rows)
    transfers = Counter(r["color_transfer"] or "unknown" for r in rows)
    top_fallback = Counter(r["reason"] for r in rows if r["verdict"] in {"ffmpeg_fallback", "block"})
    lines = [
        "Baseline 2026-05-08 - Video library compatibility scan",
        "======================================================",
        "",
        "Purpose:",
        "  Inventory F:\\VR and G:\\Downloads before Phase 2 PyNv routing.",
        "  This records input/output constraints discovered by ffprobe and the",
        "  shared select_backend() decision function.",
        "",
        "Scanned roots:",
        "  F:\\VR",
        "  G:\\Downloads",
        "",
        f"Files scanned: {len(rows)}",
        f"Elapsed: {elapsed:.3f} s",
        f"CSV: {csv_path}",
        f"Summary JSON: {summary_path}",
        "",
        "Verdict counts:",
    ]
    for key, value in verdicts.most_common():
        lines.append(f"  {key}: {value}")
    lines += ["", "Codec histogram:"]
    for key, value in codecs.most_common(20):
        lines.append(f"  {key}: {value}")
    lines += ["", "Resolution histogram:"]
    for key, value in resolutions.most_common():
        lines.append(f"  {key}: {value}")
    lines += ["", "Pixel format histogram:"]
    for key, value in pix_fmts.most_common(20):
        lines.append(f"  {key}: {value}")
    lines += ["", "Color transfer histogram:"]
    for key, value in transfers.most_common(20):
        lines.append(f"  {key}: {value}")
    lines += ["", "Top fallback/block reasons:"]
    for key, value in top_fallback.most_common(20):
        lines.append(f"  {value}: {key}")
    lines += [
        "",
        "Routing policy snapshot:",
        "  pynv_hevc: CFR SDR 8-bit h264/hevc/vp9 yuv420p/yuvj420p/nv12; production PyNv output is all-HEVC.",
        "  ffmpeg_fallback: VFR, yuv444, HDR/Main10/P010, AV1 and other non-PyNv-safe cases.",
        "  block: unsupported/unknown codecs or missing dimensions.",
        "",
        "Notes:",
        "  --strict was not required for this static scan unless stated above.",
        "  The same select_backend() function should be reused by Phase 2 routing.",
        "",
    ]
    baseline_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan videos and classify PyNv backend compatibility.")
    parser.add_argument("--root", action="append", default=[], help="root directory to scan; can be repeated")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="decode one frame with PyNv for files selected for PyNv")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--progress", type=int, default=50)
    args = parser.parse_args()

    roots = [Path(p) for p in args.root] if args.root else [Path("F:/VR"), Path("G:/Downloads")]
    out_dir = config.ROOT / "baseline"
    out_dir.mkdir(exist_ok=True)
    csv_path = Path(args.out_csv) if args.out_csv else out_dir / "video_scan_20260508.csv"
    summary_path = Path(args.out_summary) if args.out_summary else out_dir / "video_scan_20260508_summary.json"
    baseline_path = Path(args.baseline) if args.baseline else out_dir / "baseline_20260508_video_scan.txt"
    if not csv_path.is_absolute():
        csv_path = (config.ROOT / csv_path).resolve()
    if not summary_path.is_absolute():
        summary_path = (config.ROOT / summary_path).resolve()
    if not baseline_path.is_absolute():
        baseline_path = (config.ROOT / baseline_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    t0 = time.perf_counter()
    for idx, path in enumerate(_iter_videos(roots), start=1):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        try:
            meta = probe_video_metadata(path)
            decision = select_backend(meta.timing, meta.codec, meta.color)
            strict_ok = False
            strict_msg = ""
            if args.strict and decision.verdict == "pynv_hevc":
                strict_ok, strict_msg = _strict_decode(path)
                if not strict_ok:
                    decision = BackendDecision("ffmpeg_fallback", f"strict PyNv decode failed: {strict_msg}")
            row = _row_for(path, meta, decision, strict_ok, strict_msg, roots)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "display_path": str(path),
                    "verdict": "block",
                    "reason": f"probe failed: {type(exc).__name__}: {str(exc).splitlines()[0][:180]}",
                    "strict_decode": "",
                    "strict_message": "",
                    "codec": "",
                    "profile": "",
                    "level": "",
                    "pix_fmt": "",
                    "bit_depth": "",
                    "width": "",
                    "height": "",
                    "resolution_bucket": "unknown",
                    "fps": "0.000000",
                    "r_frame_rate": "",
                    "avg_frame_rate": "",
                    "is_cfr": "False",
                    "fps_diff_ratio": "0.000000",
                    "duration": "0.000",
                    "nb_frames": "0",
                    "color_range": "",
                    "color_space": "",
                    "color_transfer": "",
                    "color_primaries": "",
                    "audio_codec": "",
                    "audio_profile": "",
                }
            )
        if args.progress > 0 and len(rows) % args.progress == 0:
            elapsed = time.perf_counter() - t0
            verdicts = Counter(r["verdict"] for r in rows)
            print(f"[scan] {len(rows)} files in {elapsed:.1f}s verdicts={dict(verdicts)}")

    elapsed = time.perf_counter() - t0
    fieldnames = list(rows[0].keys()) if rows else ["path", "verdict", "reason"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, summary_path, elapsed)
    _write_baseline(rows, summary_path, csv_path, baseline_path, elapsed)

    print(f"[scan] done files={len(rows)} elapsed={elapsed:.3f}s")
    print(f"[scan] csv={csv_path}")
    print(f"[scan] summary={summary_path}")
    print(f"[scan] baseline={baseline_path}")
    print(f"[scan] verdicts={dict(Counter(r['verdict'] for r in rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
