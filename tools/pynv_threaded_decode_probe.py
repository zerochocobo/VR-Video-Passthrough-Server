"""Probe PyNvVideoCodec ThreadedDecoder for sequential CFR decimation.

This is an isolated Phase 2 tool. It does not touch the live route. The probe
checks whether ThreadedDecoder can feed the current 8K output-frame selection
pattern faster than SimpleDecoder random/indexed access, and verifies selected
NV12/P016 frames against SimpleDecoder hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.pynv_io import GpuNv12Frame, GpuP016Frame, PyNvSimpleDecoder  # noqa: E402
from utils.video_metadata import cfr_source_index, probe_timing_metadata, probe_video_metadata  # noqa: E402


GpuFrame = GpuNv12Frame | GpuP016Frame


@dataclass
class ThreadedResult:
    selected_count: int
    fetched_source_count: int
    elapsed_sec: float
    selected_fps: float
    source_fetch_fps: float
    avg_fetch_ms: float
    p99_fetch_ms: float
    avg_selected_wall_ms: float
    p99_selected_wall_ms: float


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


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(len(values) * 0.99))]


def _frame_from_decoded(raw: Any, width: int, height: int, bit_depth: int) -> GpuFrame:
    if int(bit_depth or 8) > 8:
        return GpuP016Frame.from_decoded_frame(raw, width, height)
    return GpuNv12Frame.from_decoded_frame(raw, width, height)


def _hash_plane(plane) -> str:
    import cupy as cp

    arr = plane.as_cupy()
    host = cp.asnumpy(arr)
    return hashlib.sha256(host.tobytes()).hexdigest()


def _hash_frame(frame: GpuFrame) -> dict[str, str | int]:
    return {
        "pts": int(frame.pts),
        "y_sha256": _hash_plane(frame.y),
        "uv_sha256": _hash_plane(frame.uv),
    }


def _write_json(path_value: str, data: dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pynv-threaded] wrote json {path}")


def _write_md(path_value: str, text: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")
    print(f"[pynv-threaded] wrote md {path}")


def _default_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = config.ROOT / "baseline" / f"pynv_threaded_decode_phase2_{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".md")


def _selected_source_indices(start_out: int, frames: int, source_fps: float, output_fps: float, dec_len: int) -> list[int]:
    indices: list[int] = []
    last_src_idx = -1
    for i in range(frames):
        out_idx = start_out + i
        src_idx = min(dec_len - 1, cfr_source_index(out_idx, source_fps, output_fps))
        if src_idx <= last_src_idx:
            src_idx = min(dec_len - 1, last_src_idx + 1)
        indices.append(src_idx)
        last_src_idx = src_idx
    return indices


def _simple_baseline(src: Path, indices: list[int], args: argparse.Namespace, bit_depth: int) -> dict[str, Any]:
    dec = PyNvSimpleDecoder(src, gpu_id=args.gpu, bit_depth=bit_depth)
    timings: list[float] = []
    t0 = time.perf_counter()
    try:
        for idx in indices:
            a = time.perf_counter()
            dec.frame_at(idx)
            b = time.perf_counter()
            timings.append((b - a) * 1000.0)
    finally:
        dec.stop()
    elapsed = time.perf_counter() - t0
    return {
        "decoded_frames": len(indices),
        "elapsed_sec": elapsed,
        "fps": len(indices) / elapsed if elapsed > 0 else 0.0,
        "avg_ms": statistics.fmean(timings) if timings else 0.0,
        "p99_ms": _p99(timings),
        "first_indices": indices[: min(20, len(indices))],
    }


def _simple_hashes(src: Path, indices: list[int], args: argparse.Namespace, bit_depth: int) -> list[dict[str, str | int]]:
    dec = PyNvSimpleDecoder(src, gpu_id=args.gpu, bit_depth=bit_depth)
    hashes: list[dict[str, str | int]] = []
    try:
        for idx in indices[: max(0, args.hash_frames)]:
            hashes.append(_hash_frame(dec.frame_at(idx)))
    finally:
        dec.stop()
    return hashes


def _threaded_decode(src: Path, indices: list[int], args: argparse.Namespace) -> tuple[ThreadedResult, str]:
    import PyNvVideoCodec as nvc

    start_frame = indices[0] if indices else int(args.start_frame)
    wanted = set(indices)
    selected_count = 0
    fetch_times: list[float] = []
    selected_wall_ms: list[float] = []
    source_idx = start_frame
    fetched = 0
    error = ""
    dec = nvc.ThreadedDecoder(
        str(src),
        int(args.buffer_size),
        gpu_id=int(args.gpu),
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
        start_frame=int(start_frame),
    )
    t0 = time.perf_counter()
    try:
        while selected_count < len(indices):
            a = time.perf_counter()
            batch = dec.get_batch_frames(int(args.batch_size))
            b = time.perf_counter()
            fetch_times.append((b - a) * 1000.0)
            if not batch:
                error = f"decoder returned no frames at source_idx={source_idx}"
                break
            for raw in batch:
                current = source_idx
                source_idx += 1
                fetched += 1
                if current not in wanted:
                    continue
                selected_count += 1
                selected_wall_ms.append((time.perf_counter() - t0) * 1000.0)
                if selected_count >= len(indices):
                    break
            if source_idx > indices[-1] + int(args.batch_size) + 2:
                error = f"passed final wanted index {indices[-1]} at source_idx={source_idx}"
                break
    finally:
        end = getattr(dec, "end", None)
        if callable(end):
            end()
    elapsed = time.perf_counter() - t0
    result = ThreadedResult(
        selected_count=selected_count,
        fetched_source_count=fetched,
        elapsed_sec=elapsed,
        selected_fps=selected_count / elapsed if elapsed > 0 else 0.0,
        source_fetch_fps=fetched / elapsed if elapsed > 0 else 0.0,
        avg_fetch_ms=statistics.fmean(fetch_times) if fetch_times else 0.0,
        p99_fetch_ms=_p99(fetch_times),
        avg_selected_wall_ms=(selected_wall_ms[-1] / selected_count) if selected_count and selected_wall_ms else 0.0,
        p99_selected_wall_ms=_p99(
            [
                selected_wall_ms[i] - (selected_wall_ms[i - 1] if i else 0.0)
                for i in range(len(selected_wall_ms))
            ]
        ),
    )
    return result, error


def _threaded_hashes(src: Path, indices: list[int], args: argparse.Namespace, width: int, height: int, bit_depth: int) -> tuple[list[dict[str, str | int]], str]:
    import PyNvVideoCodec as nvc

    hash_indices = indices[: max(0, args.hash_frames)]
    if not hash_indices:
        return [], ""
    start_frame = hash_indices[0]
    wanted = set(hash_indices)
    wanted_order = {idx: pos for pos, idx in enumerate(hash_indices)}
    hashes_by_pos: dict[int, dict[str, str | int]] = {}
    source_idx = start_frame
    error = ""
    dec = nvc.ThreadedDecoder(
        str(src),
        int(args.buffer_size),
        gpu_id=int(args.gpu),
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
        start_frame=int(start_frame),
    )
    try:
        while len(hashes_by_pos) < len(hash_indices):
            batch = dec.get_batch_frames(int(args.batch_size))
            if not batch:
                error = f"decoder returned no frames during hash pass at source_idx={source_idx}"
                break
            for raw in batch:
                current = source_idx
                source_idx += 1
                if current not in wanted:
                    continue
                frame = _frame_from_decoded(raw, width, height, bit_depth)
                hashes_by_pos[wanted_order[current]] = _hash_frame(frame)
                if len(hashes_by_pos) >= len(hash_indices):
                    break
            if source_idx > hash_indices[-1] + int(args.batch_size) + 2:
                error = f"hash pass exceeded final wanted index {hash_indices[-1]} at source_idx={source_idx}"
                break
    finally:
        end = getattr(dec, "end", None)
        if callable(end):
            end()
    return [hashes_by_pos[i] for i in range(min(len(hash_indices), len(hashes_by_pos)))], error


def _compare_hashes(simple: list[dict[str, str | int]], threaded: list[dict[str, str | int]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    pts_deltas: list[int] = []
    n = min(len(simple), len(threaded))
    for i in range(n):
        s = simple[i]
        t = threaded[i]
        try:
            pts_deltas.append(int(t["pts"]) - int(s["pts"]))
        except Exception:
            pass
        if s.get("y_sha256") != t.get("y_sha256") or s.get("uv_sha256") != t.get("uv_sha256"):
            mismatches.append({"position": i, "simple": s, "threaded": t})
    return {
        "checked": n,
        "matched": n - len(mismatches),
        "mismatches": mismatches,
        "pts_deltas": pts_deltas,
        "ok": n > 0 and not mismatches and len(simple) == len(threaded),
    }


def _report(data: dict[str, Any]) -> str:
    threaded = data["threaded"]
    simple = data["simple"]
    compare = data["hash_compare"]
    return "\n".join(
        [
            "# PyNv Threaded Decode Phase 2 Probe",
            "",
            f"- Generated: {data['generated_at']}",
            f"- Video: `{data['video']}`",
            f"- Source FPS: `{data['source_fps']}`",
            f"- Output FPS: `{data['output_fps']}`",
            f"- Target output frames: `{data['frames']}`",
            f"- CFR selected source range: `{data['source_indices'][0]}`..`{data['source_indices'][-1]}`",
            "",
            "## ThreadedDecoder",
            "",
            f"- Selected FPS: `{threaded['selected_fps']:.2f}`",
            f"- Source fetch FPS: `{threaded['source_fetch_fps']:.2f}`",
            f"- Selected frames: `{threaded['selected_count']}`",
            f"- Fetched source frames: `{threaded['fetched_source_count']}`",
            f"- Average fetch call: `{threaded['avg_fetch_ms']:.3f} ms`",
            f"- P99 fetch call: `{threaded['p99_fetch_ms']:.3f} ms`",
            f"- Average selected wall: `{threaded['avg_selected_wall_ms']:.3f} ms/output frame`",
            f"- P99 selected wall delta: `{threaded['p99_selected_wall_ms']:.3f} ms`",
            "",
            "## SimpleDecoder Baseline",
            "",
            f"- FPS: `{simple['fps']:.2f}`",
            f"- Average indexed decode: `{simple['avg_ms']:.3f} ms`",
            f"- P99 indexed decode: `{simple['p99_ms']:.3f} ms`",
            "",
            "## Hash Check",
            "",
            f"- Checked: `{compare['checked']}`",
            f"- Matched: `{compare['matched']}`",
            f"- OK: `{compare['ok']}`",
            f"- PTS deltas: `{compare.get('pts_deltas', [])}`",
            f"- Error: `{data.get('error', '')}`",
            "",
        ]
    )


def main() -> int:
    _configure_local_temp()
    parser = argparse.ArgumentParser(description="Probe PyNv ThreadedDecoder sequential CFR decimation.")
    parser.add_argument("video", help="video path, root-relative path, or filename under PT_VIDEO_DIR")
    parser.add_argument("--frames", type=int, default=300, help="output frames to select")
    parser.add_argument("--fps", type=float, default=30.0, help="output FPS cap; <=0 uses source FPS")
    parser.add_argument("--start-frame", type=int, default=0, help="output-frame start index")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hash-frames", type=int, default=10)
    parser.add_argument("--json-out", default="", help="optional JSON output path; default writes baseline timestamp")
    parser.add_argument("--md-out", default="", help="optional Markdown output path; default writes baseline timestamp")
    args = parser.parse_args()

    src = _resolve_video(args.video)
    metadata = probe_video_metadata(src)
    timing = metadata.timing or probe_timing_metadata(src)
    source_fps = float(timing.source_fps or 30.0)
    output_fps = timing.effective_fps(float(args.fps or 0.0))
    bit_depth = int(metadata.codec.bit_depth or 8)

    probe_dec = PyNvSimpleDecoder(src, gpu_id=args.gpu, bit_depth=bit_depth)
    try:
        info = probe_dec.info
        dec_len = len(probe_dec)
    finally:
        probe_dec.stop()

    frames = min(max(1, int(args.frames)), max(1, dec_len))
    source_indices = _selected_source_indices(int(args.start_frame), frames, source_fps, output_fps, dec_len)
    print(
        f"[pynv-threaded] src={src.name} {info.width}x{info.height} "
        f"source_fps={source_fps:.6f} output_fps={output_fps:.6f} "
        f"frames={frames} source_range={source_indices[0]}..{source_indices[-1]} bit_depth={bit_depth}"
    )

    simple_result = _simple_baseline(src, source_indices, args, bit_depth)
    print(
        f"[pynv-threaded] SimpleDecoder fps={simple_result['fps']:.2f} "
        f"avg={simple_result['avg_ms']:.3f}ms p99={simple_result['p99_ms']:.3f}ms"
    )
    threaded_result, error = _threaded_decode(
        src,
        source_indices,
        args,
    )
    print(
        f"[pynv-threaded] ThreadedDecoder selected_fps={threaded_result.selected_fps:.2f} "
        f"source_fetch_fps={threaded_result.source_fetch_fps:.2f} "
        f"avg_selected_wall={threaded_result.avg_selected_wall_ms:.3f}ms"
    )
    simple_hashes = _simple_hashes(src, source_indices, args, bit_depth)
    threaded_hashes, hash_error = _threaded_hashes(src, source_indices, args, info.width, info.height, bit_depth)
    error = "; ".join(item for item in (error, hash_error) if item)
    compare = _compare_hashes(simple_hashes, threaded_hashes)
    print(
        f"[pynv-threaded] hash checked={compare['checked']} "
        f"matched={compare['matched']} ok={compare['ok']}"
    )
    if error:
        print(f"[pynv-threaded] ERROR {error}", file=sys.stderr)

    json_out = Path(args.json_out) if args.json_out else None
    md_out = Path(args.md_out) if args.md_out else None
    if json_out is None or md_out is None:
        default_json, default_md = _default_paths()
        json_out = json_out or default_json
        md_out = md_out or default_md

    data: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video": str(src),
        "width": info.width,
        "height": info.height,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "frames": frames,
        "source_indices": source_indices,
        "simple": simple_result,
        "threaded": threaded_result.__dict__,
        "hash_compare": compare,
        "error": error,
        "paths": {"json": str(json_out), "md": str(md_out)},
    }
    _write_json(str(json_out), data)
    _write_md(str(md_out), _report(data))
    return 0 if not error and compare["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
