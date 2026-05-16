"""Validate PyNv ThreadedDecoder frame-index mapping stability.

This probe exists because ThreadedDecoder.getPTS() can differ from
SimpleDecoder[index].getPTS() even when decoded frame content matches. It proves
the mapping by hashing frame content while each ThreadedDecoder batch is still
valid, before the next get_batch_frames() call and before end().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.pynv_io import GpuNv12Frame, GpuP016Frame, PyNvSimpleDecoder  # noqa: E402
from utils.video_metadata import probe_video_metadata  # noqa: E402


@dataclass
class SimpleEntry:
    index: int
    pts: int
    y_sha256: str
    uv_sha256: str


def _configure_local_temp() -> None:
    cache = config.ROOT / ".uv-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("TMP", str(cache))
    os.environ.setdefault("TEMP", str(cache))
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache / "cupy"))


def _resolve_video(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    root_relative = (config.ROOT / p).resolve()
    if root_relative.exists():
        return root_relative
    return (config.VIDEO_DIR / p).resolve()


def _frame_from_raw(raw: Any, width: int, height: int, bit_depth: int):
    if bit_depth > 8:
        return GpuP016Frame.from_decoded_frame(raw, width, height)
    return GpuNv12Frame.from_decoded_frame(raw, width, height)


def _hash_plane(plane) -> str:
    import cupy as cp

    return hashlib.sha256(cp.asnumpy(plane.as_cupy()).tobytes()).hexdigest()


def _hash_frame(frame) -> tuple[str, str]:
    return _hash_plane(frame.y), _hash_plane(frame.uv)


def _build_simple_map(src: Path, max_index: int, gpu: int, bit_depth: int) -> dict[int, SimpleEntry]:
    dec = PyNvSimpleDecoder(src, gpu_id=gpu, bit_depth=bit_depth)
    out: dict[int, SimpleEntry] = {}
    try:
        last = min(max_index, len(dec) - 1)
        for idx in range(last + 1):
            frame = dec.frame_at(idx)
            y, uv = _hash_frame(frame)
            out[idx] = SimpleEntry(index=idx, pts=int(frame.pts), y_sha256=y, uv_sha256=uv)
    finally:
        dec.stop()
    return out


def _write_json(path_value: str, data: dict[str, Any]) -> None:
    path = Path(path_value)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pynv-map] wrote json {path}")


def _write_md(path_value: str, text: str) -> None:
    path = Path(path_value)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")
    print(f"[pynv-map] wrote md {path}")


def _default_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = config.ROOT / "baseline" / f"pynv_threaded_mapping_phase2_{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".md")


def _run_case(
    src: Path,
    simple: dict[int, SimpleEntry],
    width: int,
    height: int,
    bit_depth: int,
    start: int,
    frames: int,
    batch_size: int,
    buffer_size: int,
    gpu: int,
    repeat: int,
) -> dict[str, Any]:
    import PyNvVideoCodec as nvc

    mismatches: list[dict[str, Any]] = []
    pts_deltas: list[int] = []
    matched = 0
    decoded = 0
    decoder = nvc.ThreadedDecoder(
        str(src),
        int(buffer_size),
        gpu_id=int(gpu),
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
        start_frame=int(start),
    )
    try:
        while decoded < frames:
            batch = decoder.get_batch_frames(min(batch_size, frames - decoded))
            if not batch:
                mismatches.append({"error": "empty batch before expected frame count", "decoded": decoded})
                break
            for raw in batch:
                expected_idx = start + decoded
                frame = _frame_from_raw(raw, width, height, bit_depth)
                y, uv = _hash_frame(frame)
                expected = simple.get(expected_idx)
                row = {
                    "repeat": repeat,
                    "expected_index": expected_idx,
                    "threaded_pts": int(frame.pts),
                    "simple_pts": int(expected.pts) if expected else None,
                }
                if expected is None:
                    mismatches.append({**row, "error": "missing simple reference"})
                else:
                    delta = int(frame.pts) - int(expected.pts)
                    pts_deltas.append(delta)
                    if y == expected.y_sha256 and uv == expected.uv_sha256:
                        matched += 1
                    else:
                        mismatches.append(
                            {
                                **row,
                                "pts_delta": delta,
                                "threaded_y": y[:16],
                                "simple_y": expected.y_sha256[:16],
                                "threaded_uv": uv[:16],
                                "simple_uv": expected.uv_sha256[:16],
                            }
                        )
                decoded += 1
                if decoded >= frames:
                    break
    finally:
        end = getattr(decoder, "end", None)
        if callable(end):
            end()
    return {
        "start_frame": start,
        "frames": frames,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "repeat": repeat,
        "decoded": decoded,
        "matched": matched,
        "ok": decoded == frames and matched == frames and not mismatches,
        "pts_delta_min": min(pts_deltas) if pts_deltas else None,
        "pts_delta_max": max(pts_deltas) if pts_deltas else None,
        "pts_delta_values": sorted(set(pts_deltas)),
        "pts_delta_avg": statistics.fmean(pts_deltas) if pts_deltas else None,
        "mismatches": mismatches[:20],
    }


def _report(data: dict[str, Any]) -> str:
    cases = data["cases"]
    ok_count = sum(1 for case in cases if case["ok"])
    pts_values = sorted({delta for case in cases for delta in case.get("pts_delta_values", [])})
    return "\n".join(
        [
            "# PyNv ThreadedDecoder Mapping Stability Report",
            "",
            f"- Generated: {data['generated_at']}",
            f"- Video: `{data['video']}`",
            f"- Cases: `{ok_count}/{len(cases)}` passed",
            f"- Overall OK: `{data['ok']}`",
            f"- PTS delta values: `{pts_values}`",
            "",
            "## Interpretation",
            "",
            "ThreadedDecoder frame content is compared against SimpleDecoder[index] by Y/UV SHA256.",
            "Frames are hashed before the next `get_batch_frames()` call and before `end()`, matching the documented frame lifetime.",
            "A nonzero PTS delta is recorded separately and must not be used as the frame identity for ThreadedDecoder.",
            "",
            "## Cases",
            "",
            "| start | batch | repeat | decoded | matched | ok | pts deltas |",
            "|---:|---:|---:|---:|---:|---|---|",
            *[
                f"| {case['start_frame']} | {case['batch_size']} | {case['repeat']} | {case['decoded']} | {case['matched']} | {case['ok']} | {case['pts_delta_values']} |"
                for case in cases
            ],
            "",
        ]
    )


def main() -> int:
    _configure_local_temp()
    parser = argparse.ArgumentParser(description="Validate ThreadedDecoder frame mapping against SimpleDecoder.")
    parser.add_argument("video")
    parser.add_argument("--starts", default="0,1,2,3,4,10,30,58,120,300", help="comma-separated start_frame values")
    parser.add_argument("--frames-per-start", type=int, default=16)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16", help="comma-separated get_batch_frames sizes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    args = parser.parse_args()

    src = _resolve_video(args.video)
    metadata = probe_video_metadata(src)
    bit_depth = int(metadata.codec.bit_depth or 8)
    probe = PyNvSimpleDecoder(src, gpu_id=args.gpu, bit_depth=bit_depth)
    try:
        info = probe.info
        dec_len = len(probe)
    finally:
        probe.stop()

    starts = [int(item.strip()) for item in args.starts.split(",") if item.strip()]
    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    frames = max(1, int(args.frames_per_start))
    max_index = min(dec_len - 1, max(starts) + frames + 2)
    simple = _build_simple_map(src, max_index, args.gpu, bit_depth)
    cases: list[dict[str, Any]] = []
    for start in starts:
        if start + frames > dec_len:
            continue
        for batch_size in batch_sizes:
            if batch_size > args.buffer_size:
                continue
            for repeat in range(max(1, args.repeats)):
                case = _run_case(
                    src,
                    simple,
                    info.width,
                    info.height,
                    bit_depth,
                    start,
                    frames,
                    batch_size,
                    args.buffer_size,
                    args.gpu,
                    repeat,
                )
                print(
                    f"[pynv-map] start={start} batch={batch_size} repeat={repeat} "
                    f"matched={case['matched']}/{case['decoded']} ok={case['ok']} pts={case['pts_delta_values']}"
                )
                cases.append(case)

    default_json, default_md = _default_paths()
    json_out = Path(args.json_out) if args.json_out else default_json
    md_out = Path(args.md_out) if args.md_out else default_md
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video": str(src),
        "width": info.width,
        "height": info.height,
        "bit_depth": bit_depth,
        "frames_per_start": frames,
        "starts": starts,
        "batch_sizes": batch_sizes,
        "repeats": args.repeats,
        "buffer_size": args.buffer_size,
        "cases": cases,
        "ok": bool(cases) and all(case["ok"] for case in cases),
        "paths": {"json": str(json_out), "md": str(md_out)},
    }
    _write_json(str(json_out), data)
    _write_md(str(md_out), _report(data))
    return 0 if data["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
