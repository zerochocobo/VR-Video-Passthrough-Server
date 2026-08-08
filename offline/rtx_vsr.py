"""Offline ordinary-2D RTX VSR converter.

FFmpeg decodes RGBA frames, CuPy moves each frame to the GPU, the prebuilt
CUDA bridge evaluates NVIDIA NGX VSR, and FFmpeg NVENC muxes video/audio.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

from utils.gpu_runtime_cache import configure_gpu_runtime_cache

# Standalone offline entry points can bypass main.py. Pin project-local CUDA
# and CuPy caches before the first lazy CuPy import (PROJECT.md sm_120 rules).
configure_gpu_runtime_cache()

import config
from utils.rtx_vsr import (
    effective_offline_target_height, load_bridge, run_evaluation_preflight,
    source_block_reason, target_dimensions, target_resolution,
)
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.superres_bitrate import plan_superres_bitrate
from utils.vr_naming import has_vr_filename_marker, is_half_equirectangular_source, superres_output_stem
from utils.video_metadata import VideoProbeMetadata


def _format_progress_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_message(frames: int, total_frames: int, elapsed: float) -> str:
    elapsed = max(1e-6, float(elapsed))
    processing_fps = float(frames) / elapsed
    if total_frames > 0:
        percent = min(100.0, float(frames) * 100.0 / float(total_frames))
        remaining = max(0, total_frames - frames)
        eta = remaining / processing_fps if processing_fps > 0 else 0.0
        return (
            f"progress={percent:.1f}% frames={frames}/{total_frames} "
            f"fps={processing_fps:.2f} elapsed={_format_progress_time(elapsed)} "
            f"eta={_format_progress_time(eta)}"
        )
    return (
        f"frames={frames} fps={processing_fps:.2f} "
        f"elapsed={_format_progress_time(elapsed)} eta=--:--:--"
    )


def _format_size(size_bytes: float) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f}GB"
    return f"{size_bytes / 1_000_000:.1f}MB"


def run_rtx_vsr(
    src: Path,
    out: Path,
    meta: VideoProbeMetadata,
    *,
    start: float = 0.0,
    duration: float = 0.0,
    target_height: int | None = None,
    quality: int | None = None,
    preset: str | None = None,
    hdr_look: str | None = None,
    cq: int = 19,
    bitrate_mode: str = "auto",
) -> int:
    width = int(meta.codec.width)
    height = int(meta.codec.height)
    requested_target_height = int(target_height or config.RTX_VSR_TARGET_HEIGHT)
    target_height = effective_offline_target_height(requested_target_height, width, height)
    if target_height != requested_target_height:
        print(
            f"[rtx-vsr] 8K is limited to 2:1 SBS VR sources; "
            f"source={width}x{height} fallback=4K",
            flush=True,
        )
    reason = source_block_reason(
        width,
        height,
        is_vr=has_vr_filename_marker(src.stem) or is_half_equirectangular_source(width, height),
        is_10bit=bool(int(getattr(meta.codec, "bit_depth", 8) or 8) > 8),
        target_height=target_height,
        allow_vr=True,
    )
    if reason:
        if reason == "source_exceeds_target_resolution":
            target_w, target_h = target_resolution(target_height, width, height)
            print(
                f"[rtx-vsr] reject source={src.name}: source resolution {width}x{height} "
                f"exceeds target {target_w}x{target_h}; conversion not started",
                flush=True,
            )
        else:
            print(f"[rtx-vsr] reject source={src.name} reason={reason}", flush=True)
        return 2
    if not config.RTX_VSR_OFFLINE_ENABLED:
        print("[rtx-vsr] disabled by PT_RTX_VSR_OFFLINE_ENABLE", flush=True)
        return 2
    fps = meta.timing.source_fps if meta.timing.source_fps > 0 else 30.0
    out_w, out_h = target_dimensions(width, height, target_height)
    from utils.bitrate_estimator import source_video_bitrate

    bitrate_plan = plan_superres_bitrate(
        source_video_bitrate(src), width, height, out_w, out_h, bitrate_mode,
    )
    source_duration = max(0.0, float(getattr(meta.timing, "duration", 0.0) or 0.0))
    available_duration = max(0.0, source_duration - max(0.0, float(start or 0.0)))
    requested_duration = max(0.0, float(duration or 0.0))
    work_duration = min(requested_duration, available_duration) if requested_duration > 0 and available_duration > 0 else (requested_duration or available_duration)
    estimated_size = _format_size(bitrate_plan.target_bps * work_duration / 8.0) if work_duration > 0 else "unknown"
    print(
        f"[rtx-vsr] bitrate_mode={bitrate_plan.mode} "
        f"source_bitrate={bitrate_plan.source_bps / 1_000_000:.2f}Mbps "
        f"pixel_ratio={bitrate_plan.pixel_ratio:.3f} "
        f"target_bitrate={bitrate_plan.target_bps / 1_000_000:.2f}Mbps "
        f"max_bitrate={bitrate_plan.max_bps / 1_000_000:.2f}Mbps "
        f"estimated_video_size={estimated_size}",
        flush=True,
    )
    # Fail before opening FFmpeg pipes if the SDK/driver cannot complete one
    # real evaluation.  The preflight itself is isolated and time-limited;
    # this prevents a partially written output file on known-bad systems.
    preflight = run_evaluation_preflight()
    if not preflight.get("ok"):
        print(
            f"[rtx-vsr] evaluate preflight failed: {preflight.get('reason', 'unknown')} "
            f"{preflight.get('detail', '')}".strip(),
            flush=True,
        )
        return 3
    if config.RTX_VSR_OFFLINE_PYNV_ENABLED:
        try:
            from offline.rtx_vsr_pynv import run_rtx_vsr_pynv

            return run_rtx_vsr_pynv(
                src,
                out,
                meta,
                start=float(start or 0.0),
                duration=float(duration or 0.0),
                target_height=int(target_height or config.RTX_VSR_TARGET_HEIGHT),
                quality=int(config.RTX_VSR_QUALITY if quality is None else quality),
                preset=str(preset or config.PASSTHROUGH_PYNV_PRESET or "p4"),
                cq=int(cq),
                hdr_look=str(hdr_look or config.RTX_VSR_HDR_LOOK),
                target_bitrate=bitrate_plan.target_bps,
                max_bitrate=bitrate_plan.max_bps,
                buffer_bitrate=bitrate_plan.buffer_bps,
            )
        except Exception as exc:
            if int(target_height or 0) >= 4096:
                print(f"[rtx-vsr] split-eye GPU pipeline failed: {type(exc).__name__}: {exc}", flush=True)
            else:
                print(f"[rtx-vsr] GPU pipeline unavailable, falling back to FFmpeg rawvideo: {type(exc).__name__}: {exc}", flush=True)
            try:
                if out.exists():
                    out.unlink()
            except OSError:
                pass
            if int(target_height or 0) >= 4096:
                print(
                    "[rtx-vsr] experimental 8K requires the split-eye GPU pipeline; "
                    "legacy whole-frame fallback is disabled",
                    flush=True,
                )
                return 3
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    decode_cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start > 0:
        decode_cmd += ["-ss", f"{start:.6f}"]
    decode_cmd += ["-i", str(src)]
    if duration > 0:
        decode_cmd += ["-t", f"{duration:.6f}"]
    decode_cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    encode_cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.9f}", "-i", "-",
    ]
    if start > 0:
        encode_cmd += ["-ss", f"{start:.6f}"]
    if duration > 0:
        encode_cmd += ["-t", f"{duration:.6f}"]
    effective_preset = str(preset or config.PASSTHROUGH_PYNV_PRESET or "P1").strip().lower()
    if effective_preset not in {"p1", "p4", "p7"}:
        effective_preset = "p4"
    encode_cmd += [
        "-i", str(src), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "hevc_nvenc", "-preset", effective_preset,
        "-rc", "vbr", "-b:v", str(bitrate_plan.target_bps),
        "-maxrate", str(bitrate_plan.max_bps), "-bufsize", str(bitrate_plan.buffer_bps),
        "-cq", str(int(cq)), "-c:a", "copy", "-shortest",
        *meta.color.ffmpeg_args(), str(out),
    ]
    effective_quality = config.RTX_VSR_QUALITY if quality is None else quality
    from pipeline.hdr_look import apply_hdr_look, create_hdr_look_kernel, normalize_hdr_look
    effective_hdr_look = normalize_hdr_look(hdr_look or config.RTX_VSR_HDR_LOOK)
    print(f"[rtx-vsr] input={width}x{height} output={out_w}x{out_h} fps={fps:.6f} quality={effective_quality} preset={effective_preset} hdr_look={effective_hdr_look}", flush=True)
    print("[rtx-vsr] decode=" + subprocess.list2cmdline(decode_cmd), flush=True)
    print("[rtx-vsr] encode=" + subprocess.list2cmdline(encode_cmd), flush=True)
    import cupy as cp
    import numpy as np

    bridge = load_bridge()
    if not bridge.initialize_cupy(cp):
        print("[rtx-vsr] CUDA/NGX initialization failed for the CuPy context", flush=True)
        return 3
    hdr_look_kernel = create_hdr_look_kernel(cp) if effective_hdr_look != "off" else None

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_subprocess_kwargs())
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_subprocess_kwargs())
    stderr_lines: list[str] = []

    def drain_stderr(proc, label: str) -> None:
        pipe = proc.stderr
        if pipe is None:
            return
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                stderr_lines.append(f"{label}: {line}")
                del stderr_lines[:-50]
                print(f"[rtx-vsr][{label}] {line}", flush=True)
        finally:
            pipe.close()

    decoder_stderr_thread = threading.Thread(target=drain_stderr, args=(decoder, "decode"), daemon=True)
    encoder_stderr_thread = threading.Thread(target=drain_stderr, args=(encoder, "encode"), daemon=True)
    decoder_stderr_thread.start()
    encoder_stderr_thread.start()
    frame_bytes = width * height * 4
    frames = 0
    total_frames = max(0, int(round(work_duration * fps)))
    progress_started = time.monotonic()
    next_progress_at = progress_started + 5.0
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"short RGBA frame: {len(raw)} != {frame_bytes}")
            host = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            rgba = cp.asarray(host)
            result = bridge.process_cupy_rgba(rgba, (out_w, out_h), quality)
            if hdr_look_kernel is not None:
                apply_hdr_look(hdr_look_kernel, result, effective_hdr_look)
            try:
                encoder.stdin.write(result.get().tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"RTX VSR encoder pipe closed: {exc}") from exc
            frames += 1
            now = time.monotonic()
            if now >= next_progress_at:
                print(f"[rtx-vsr] {_progress_message(frames, total_frames, now - progress_started)}", flush=True)
                next_progress_at = now + 5.0
    finally:
        if decoder.stdout:
            decoder.stdout.close()
        try:
            decoder.wait(timeout=30)
        except subprocess.TimeoutExpired:
            decoder.kill()
            decoder.wait(timeout=5)
        if encoder.stdin:
            encoder.stdin.close()
        try:
            encoder.wait(timeout=60)
        except subprocess.TimeoutExpired:
            encoder.kill()
            encoder.wait(timeout=5)
        decoder_stderr_thread.join(timeout=5)
        encoder_stderr_thread.join(timeout=5)
    if decoder.returncode != 0 or encoder.returncode != 0:
        detail = " | ".join(stderr_lines[-10:])
        print(f"[rtx-vsr] failed decode_rc={decoder.returncode} encode_rc={encoder.returncode} {detail}", flush=True)
        return 1
    elapsed = max(1e-6, time.monotonic() - progress_started)
    print(f"[rtx-vsr] complete {_progress_message(frames, total_frames or frames, elapsed)} out={out}", flush=True)
    return 0


def default_output(src: Path) -> Path:
    return src.with_name(superres_output_stem(src.stem, config.RTX_VSR_TARGET_HEIGHT) + ".mp4")
