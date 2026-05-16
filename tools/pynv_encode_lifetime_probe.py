"""Probe whether PyNv NVENC consumes GPU NV12 input before slot reuse.

The test encodes deterministic solid NV12 frames from a configurable ring of GPU
buffers. When ``--slots 1 --overwrite-after-encode`` is used, it immediately
overwrites the same buffer after Encode returns. With ``--slots N`` and no
post-encode overwrite, a buffer is overwritten only when reused N frames later.
The encoded stream is decoded back to raw NV12 and sampled.

If encoded frames show the expected luma values, that reuse distance was safe in
this probe. If they show overwrite values, NVENC still read from a reused input
buffer and the reuse distance is unsafe.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline.pynv_io import GpuNv12AppFrame  # noqa: E402


def _write_json(path_value: str, data: dict[str, Any]) -> None:
    path = Path(path_value)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pynv-life] wrote json {path}")


def _write_md(path_value: str, text: str) -> None:
    path = Path(path_value)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")
    print(f"[pynv-life] wrote md {path}")


def _resolve_out(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (config.ROOT / path).resolve()


def _default_paths(codec: str) -> tuple[Path, Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = config.ROOT / "baseline" / f"pynv_encode_lifetime_stage3_{stamp}"
    artifacts = config.ROOT / "baseline" / f"pynv_encode_lifetime_stage3_{stamp}_artifacts"
    ext = ".hevc" if codec.lower() in {"hevc", "h265"} else ".h264"
    return base.with_suffix(".json"), base.with_suffix(".md"), artifacts / f"encoded{ext}", artifacts / "decoded.nv12"


def _fill_nv12(dev, width: int, height: int, y_value: int, uv_value: int = 128) -> None:
    import cupy as cp

    y_bytes = int(width) * int(height)
    uv_bytes = y_bytes // 2
    cp.cuda.runtime.memset(int(dev.data.ptr), int(y_value) & 0xFF, y_bytes)
    cp.cuda.runtime.memset(int(dev.data.ptr) + y_bytes, int(uv_value) & 0xFF, uv_bytes)


def _decode_raw_nv12(encoded: Path, raw_out: Path, width: int, height: int, fps: int, codec: str) -> tuple[int, str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    fmt = "hevc" if codec.lower() in {"hevc", "h265"} else "h264"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        fmt,
        "-framerate",
        str(fps),
        "-i",
        str(encoded),
        "-an",
        "-pix_fmt",
        "nv12",
        "-f",
        "rawvideo",
        "-y",
        str(raw_out),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", timeout=120)
    return p.returncode, (p.stdout + p.stderr)


def _sample_decoded(raw: Path, width: int, height: int, frames: int) -> list[dict[str, Any]]:
    frame_size = width * height * 3 // 2
    data = raw.read_bytes()
    decoded_frames = len(data) // frame_size
    out: list[dict[str, Any]] = []
    for i in range(min(frames, decoded_frames)):
        off = i * frame_size
        y = data[off : off + width * height]
        sample = y[0 : min(len(y), 4096)]
        avg = sum(sample) / len(sample) if sample else 0.0
        out.append(
            {
                "frame": i,
                "y_first": y[0] if y else None,
                "y_avg_sample": avg,
                "y_min_sample": min(sample) if sample else None,
                "y_max_sample": max(sample) if sample else None,
            }
        )
    return out


def _classify(samples: list[dict[str, Any]], expected: list[int], overwritten: list[int], tolerance: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ok = True
    corrupt = 0
    for sample in samples:
        i = int(sample["frame"])
        avg = float(sample["y_avg_sample"])
        exp = expected[i]
        over = overwritten[i]
        dist_exp = abs(avg - exp)
        dist_over = abs(avg - over)
        status = "expected" if dist_exp <= tolerance else "overwrite" if dist_over <= tolerance else "unknown"
        if status != "expected":
            ok = False
            corrupt += 1
        rows.append({**sample, "expected_y": exp, "overwrite_y": over, "status": status})
    return {"ok": ok, "corrupt_or_unknown": corrupt, "frames": rows}


def _report(data: dict[str, Any]) -> str:
    cls = data["classification"]
    return "\n".join(
        [
            "# PyNv Encode Input Lifetime Probe",
            "",
            f"- Generated: {data['generated_at']}",
            f"- Codec: `{data['codec']}`",
            f"- Size: `{data['width']}x{data['height']}`",
            f"- Frames: `{data['frames']}`",
            f"- Slots: `{data['slots']}`",
            f"- Overwrite after Encode: `{data['overwrite_after_encode']}`",
            f"- Overall OK: `{cls['ok']}`",
            f"- Corrupt/unknown frames: `{cls['corrupt_or_unknown']}`",
            "",
            "## Interpretation",
            "",
            "Each frame is encoded from a deterministic GPU NV12 slot.",
            "With one slot and post-encode overwrite, this tests immediate reuse after `Encode()` returns.",
            "With multiple slots, this tests reuse after the configured ring distance.",
            "",
            "## Sampled Frames",
            "",
            "| frame | expected Y | overwrite Y | decoded avg Y | status |",
            "|---:|---:|---:|---:|---|",
            *[
                f"| {row['frame']} | {row['expected_y']} | {row['overwrite_y']} | {row['y_avg_sample']:.2f} | {row['status']} |"
                for row in cls["frames"][:80]
            ],
            "",
        ]
    )


def run(args: argparse.Namespace) -> int:
    import cupy as cp
    import PyNvVideoCodec as nvc

    default_json, default_md, default_encoded, default_raw = _default_paths(args.codec)
    json_out = _resolve_out(Path(args.json_out)) if args.json_out else default_json
    md_out = _resolve_out(Path(args.md_out)) if args.md_out else default_md
    encoded = _resolve_out(Path(args.encoded_out)) if args.encoded_out else default_encoded
    raw = _resolve_out(Path(args.raw_out)) if args.raw_out else default_raw
    for path in (encoded, raw):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[pynv-life] create encoder {args.width}x{args.height} "
        f"codec={args.codec} frames={args.frames} out={encoded}"
        , flush=True
    )
    enc = nvc.CreateEncoder(
        args.width,
        args.height,
        "NV12",
        False,
        codec=args.codec,
        bitrate=str(args.bitrate),
        fps=str(args.fps),
        gop=str(args.gop),
        bf="0",
    )
    slots = [cp.empty((args.height * 3 // 2, args.width), dtype=cp.uint8) for _ in range(max(1, args.slots))]
    app_frames = [GpuNv12AppFrame(dev, args.width, args.height) for dev in slots]
    expected: list[int] = []
    overwritten: list[int] = []
    packets = 0
    bytes_written = 0
    t0 = time.perf_counter()
    with encoded.open("wb") as f:
        for i in range(max(1, args.frames)):
            y_expected = (32 + (i * 13) % 160) & 0xFF
            y_overwrite = (223 - (i * 17) % 160) & 0xFF
            if abs(y_expected - y_overwrite) < 32:
                y_overwrite = (y_overwrite + 80) & 0xFF
            expected.append(y_expected)
            overwritten.append(y_overwrite)
            slot_idx = i % len(slots)
            dev = slots[slot_idx]
            app_frame = app_frames[slot_idx]
            _fill_nv12(dev, args.width, args.height, y_expected)
            flags = 0
            if i == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            bitstream = enc.Encode(app_frame, flags) if flags else enc.Encode(app_frame)
            if args.post_encode_sync:
                cp.cuda.Stream.null.synchronize()
            if args.overwrite_after_encode:
                _fill_nv12(dev, args.width, args.height, y_overwrite)
            if bitstream:
                f.write(bitstream)
                packets += 1
                bytes_written += len(bitstream)
            if args.progress > 0 and (i + 1) % args.progress == 0:
                print(f"[pynv-life] encoded input frames {i + 1}/{args.frames} packets={packets}", flush=True)
        tail = enc.EndEncode()
        if tail:
            f.write(tail)
            packets += 1
            bytes_written += len(tail)
    elapsed = time.perf_counter() - t0
    print(f"[pynv-life] encode done packets={packets} bytes={bytes_written} elapsed={elapsed:.3f}s", flush=True)
    rc, ffmpeg_log = _decode_raw_nv12(encoded, raw, args.width, args.height, args.fps, args.codec)
    print(f"[pynv-life] ffmpeg decode rc={rc} raw={raw}", flush=True)
    samples = _sample_decoded(raw, args.width, args.height, args.frames) if rc == 0 else []
    classification = _classify(samples, expected, overwritten, args.tolerance) if samples else {"ok": False, "corrupt_or_unknown": args.frames, "frames": []}
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "slots": max(1, args.slots),
        "overwrite_after_encode": args.overwrite_after_encode,
        "fps": args.fps,
        "codec": args.codec,
        "bitrate": args.bitrate,
        "gop": args.gop,
        "post_encode_sync": args.post_encode_sync,
        "packets": packets,
        "encoded_bytes": bytes_written,
        "encode_elapsed_sec": elapsed,
        "encode_fps": args.frames / elapsed if elapsed > 0 else 0.0,
        "ffmpeg_decode_rc": rc,
        "ffmpeg_log": ffmpeg_log[-4000:],
        "classification": classification,
        "paths": {
            "json": str(json_out),
            "md": str(md_out),
            "encoded": str(encoded),
            "raw": str(raw),
        },
    }
    _write_json(str(json_out), data)
    _write_md(str(md_out), _report(data))
    print(f"[pynv-life] ok={classification['ok']} corrupt_or_unknown={classification['corrupt_or_unknown']} encode_fps={data['encode_fps']:.2f}")
    return 0 if rc == 0 and classification["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe PyNv NVENC GPU input lifetime after Encode returns.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", default="h264", choices=["h264", "hevc", "h265"])
    parser.add_argument("--bitrate", default="8000000")
    parser.add_argument("--gop", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--encoded-out", default="")
    parser.add_argument("--raw-out", default="")
    parser.add_argument("--progress", type=int, default=10)
    parser.add_argument("--post-encode-sync", action="store_true", help="synchronize the null stream after Encode returns and before overwrite")
    parser.add_argument("--slots", type=int, default=1, help="number of reusable GPU input slots")
    parser.add_argument("--overwrite-after-encode", action="store_true", help="overwrite the just-encoded slot immediately after Encode returns")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
