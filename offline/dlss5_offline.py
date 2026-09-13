"""Offline DLSS 5 Neural Rendering video converter.

Decodes video with FFmpeg, passes frames through DLSS5 NR on GPU,
and encodes the enhanced output with NVENC.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.gpu_runtime_cache import configure_gpu_runtime_cache

configure_gpu_runtime_cache()

import numpy as np

import config
from models.dlss5.neural_bridge import BRIDGE
from utils.offline_outputs import discard_pending_output, pending_output_path, publish_pending_output
from utils.dlss5 import (
    DLSS5Settings,
    is_dlss5_available,
    prepare_cuda_context_for_dlss5,
    source_block_reason_dlss5,
)
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.video_metadata import VideoProbeMetadata, probe_video_metadata
from utils.vr_naming import dlss5_output_stem


def _format_progress_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_message(frames: int, total_frames: int, elapsed: float) -> str:
    elapsed = max(1e-6, float(elapsed))
    fps = float(frames) / elapsed
    if total_frames > 0:
        pct = min(100.0, float(frames) * 100.0 / float(total_frames))
        rem = max(0, total_frames - frames)
        eta = rem / fps if fps > 0 else 0.0
        return (
            f"progress={pct:.1f}% frames={frames}/{total_frames} "
            f"fps={fps:.2f} elapsed={_format_progress_time(elapsed)} "
            f"eta={_format_progress_time(eta)}"
        )
    return f"frames={frames} fps={fps:.2f} elapsed={_format_progress_time(elapsed)} eta=--:--:--"


def run_dlss5(
    src: Path,
    out: Path,
    meta: VideoProbeMetadata,
    *,
    start: float = 0.0,
    duration: float = 0.0,
    settings: DLSS5Settings | None = None,
    preset: str | None = None,
    cq: int = 19,
    bitrate_mode: str = "auto",
) -> int:
    width = int(meta.codec.width)
    height = int(meta.codec.height)

    reason = source_block_reason_dlss5(width, height)
    if reason:
        print(f"[dlss5] reject source={src.name} reason={reason}", flush=True)
        return 2

    if not config.DLSS5_ENABLED:
        print("[dlss5] disabled by PT_DLSS5_ENABLE", flush=True)
        return 2

    # Must come before anything in this process activates the primary CUDA
    # context: the NR engine needs its blocking-sync flag and the flag cannot be
    # set once the context is live.
    prepare_cuda_context_for_dlss5()

    import cupy as cp

    from pipeline.dlss5_stage import DLSS5Stage

    stage = DLSS5Stage(width=width, height=height, settings=settings)

    fps = meta.timing.source_fps if meta.timing.source_fps > 0 else 30.0
    source_duration = max(0.0, float(getattr(meta.timing, "duration", 0.0) or 0.0))
    available_duration = max(0.0, source_duration - max(0.0, float(start or 0.0)))
    requested_duration = max(0.0, float(duration or 0.0))
    eval_duration = (
        min(available_duration, requested_duration)
        if (requested_duration > 0 and available_duration > 0)
        else (requested_duration or available_duration)
    )
    total_frames = int(round(eval_duration * fps)) if eval_duration > 0 else 0

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    dec_args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        dec_args.extend(["-ss", f"{start:.3f}"])
    dec_args.extend(["-i", str(src)])
    if duration > 0:
        dec_args.extend(["-t", f"{duration:.3f}"])
    dec_args.extend(["-f", "rawvideo", "-pix_fmt", "nv12", "-an", "-sn", "-"])

    enc_args = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "nv12",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.3f}",
        "-i", "-",
    ]
    if start > 0:
        enc_args.extend(["-ss", f"{start:.3f}"])
    enc_args.extend(["-i", str(src)])
    pending = pending_output_path(out)
    if duration > 0:
        enc_args.extend(["-t", f"{duration:.3f}"])

    # PASSTHROUGH_PYNV_PRESET is spelled the way PyNvVideoCodec wants it ("P1",
    # uppercase). FFmpeg's hevc_nvenc only knows the lowercase names and answers
    # an unknown one with "Error applying encoder options: Invalid argument",
    # which reaches the caller as a broken pipe on the next frame. Normalise the
    # same way offline/rtx_vsr.py does.
    enc_preset = str(preset or getattr(config, "PASSTHROUGH_PYNV_PRESET", "p4") or "p4").strip().lower()
    if enc_preset not in {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}:
        enc_preset = "p4"
    enc_args.extend([
        "-c:v", "hevc_nvenc",
        "-preset", enc_preset,
        "-cq", str(cq),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:a", "copy",
        "-map_metadata", "1",
        "-movflags", "+faststart",
        str(pending),
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    dec_proc = subprocess.Popen(dec_args, stdout=subprocess.PIPE, **hidden_subprocess_kwargs())
    # The encoder's own words, kept rather than inherited: when it exits early
    # the next frame fails with a bare "[Errno 22] Invalid argument" that says
    # nothing about why, and its reason has already scrolled past.
    enc_proc = subprocess.Popen(
        enc_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_subprocess_kwargs()
    )
    enc_errors: list[str] = []

    def _drain_encoder_errors() -> None:
        stream = enc_proc.stderr
        if stream is None:
            return
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                enc_errors.append(text)
                print(f"[dlss5][nvenc] {text}", flush=True)

    enc_error_reader = threading.Thread(target=_drain_encoder_errors, name="dlss5-nvenc-stderr", daemon=True)
    enc_error_reader.start()

    frame_bytes = (width * height * 3) // 2
    y_size = width * height
    uv_size = y_size // 2

    # GPU buffers
    d_in_y = cp.empty((height, width), cp.uint8)
    d_in_uv = cp.empty((height // 2, width), cp.uint8)
    d_out_y = cp.empty((height, width), cp.uint8)
    d_out_uv = cp.empty((height // 2, width), cp.uint8)

    processed_frames = 0
    start_time = time.time()
    last_print = 0.0

    print(
        f"[dlss5] start source={src.name} res={width}x{height} "
        f"frames={total_frames or 'unknown'}",
        flush=True,
    )

    try:
        while True:
            raw = dec_proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            # Host -> GPU
            h_y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape((height, width))
            h_uv = np.frombuffer(raw[y_size:], dtype=np.uint8).reshape((height // 2, width))
            d_in_y.set(h_y)
            d_in_uv.set(h_uv)

            stage.process_nv12(
                int(d_in_y.data.ptr),
                int(d_in_uv.data.ptr),
                width,
                width,
                int(d_out_y.data.ptr),
                int(d_out_uv.data.ptr),
                width,
                width,
                reset=(processed_frames == 0),
            )
            cp.cuda.Stream.null.synchronize()

            # GPU -> Host -> Enc
            out_raw = d_out_y.get().tobytes() + d_out_uv.get().tobytes()
            if enc_proc.poll() is not None:
                reason = "; ".join(enc_errors[-3:]) or f"exit code {enc_proc.returncode}"
                raise RuntimeError(
                    f"NVENC stopped after {processed_frames} frames: {reason}"
                )
            enc_proc.stdin.write(out_raw)

            processed_frames += 1
            now = time.time()
            if now - last_print >= 2.0:
                print(_progress_message(processed_frames, total_frames, now - start_time), flush=True)
                last_print = now

    finally:
        if dec_proc.stdout:
            dec_proc.stdout.close()
        dec_proc.wait()
        if enc_proc.stdin:
            try:
                enc_proc.stdin.close()
            except OSError:
                pass
        enc_proc.wait()
        enc_error_reader.join(timeout=5.0)
        if enc_proc.stderr:
            enc_proc.stderr.close()
    if enc_proc.returncode not in (0, None):
        reason = "; ".join(enc_errors[-3:]) or f"exit code {enc_proc.returncode}"
        print(f"[dlss5] encode failed: {reason}", flush=True)
        discard_pending_output(pending)
        return 1
    if not publish_pending_output(pending, out):
        print(f"[dlss5] could not rename {pending.name} to {out.name}", flush=True)
        discard_pending_output(pending)
        return 1

    elapsed = time.time() - start_time
    print(
        f"[dlss5] finished frames={processed_frames} elapsed={_format_progress_time(elapsed)} "
        f"fps={processed_frames / max(1e-6, elapsed):.2f}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from offline.convert import main as convert_main

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("single or batch command required")
    return convert_main([args[0], "--engine", "dlss5", *args[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
