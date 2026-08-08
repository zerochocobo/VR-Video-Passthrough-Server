"""GPU-resident offline RTX VSR pipeline: NVDEC -> CUDA/NGX -> NVENC."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config
from utils.subprocess_hidden import hidden_subprocess_kwargs


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


_NV12_TO_SPLIT_RGBA = r'''
extern "C" __global__ void nv12_to_split_rgba(
    const unsigned char* Y, const unsigned char* UV,
    unsigned char* left_rgba, unsigned char* right_rgba,
    int W, int H, int eyeW)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;
    float c = (float)Y[(long)y*W+x] - 16.0f;
    int cy=y>>1, cx=(x>>1)<<1;
    float du = (float)UV[(long)cy*W+cx]   - 128.0f;
    float dv = (float)UV[(long)cy*W+cx+1] - 128.0f;
    float R = 1.16438356f*c + 1.79274107f*dv;
    float G = 1.16438356f*c - 0.21324861f*du - 0.53290933f*dv;
    float B = 1.16438356f*c + 2.11240179f*du;
    int ex = x < eyeW ? x : x-eyeW;
    unsigned char* rgba = x < eyeW ? left_rgba : right_rgba;
    long o=((long)y*eyeW+ex)*4;
    rgba[o+0]=(unsigned char)fminf(fmaxf(R,0.f),255.f);
    rgba[o+1]=(unsigned char)fminf(fmaxf(G,0.f),255.f);
    rgba[o+2]=(unsigned char)fminf(fmaxf(B,0.f),255.f);
    rgba[o+3]=255;
}
'''


def _format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _open_video_muxer(out: Path, fps: float, color_args: list[str]):
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-fflags", "+genpts",
        "-f", "hevc", "-framerate", f"{fps:.9f}", "-i", "-",
    ]
    cmd += ["-map", "0:v:0", "-c:v", "copy", "-tag:v", "hvc1"]
    cmd += [*color_args, "-movflags", "+faststart", str(out)]
    return cmd, subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_subprocess_kwargs())


def _mux_source_audio(video: Path, src: Path, out: Path, start: float, duration: float) -> None:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-shortest",
        "-movflags", "+faststart", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, **hidden_subprocess_kwargs())
    if result.returncode != 0:
        error = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"GPU RTX VSR audio mux failed rc={result.returncode}: {error[:500]}")


def run_rtx_vsr_pynv(
    src: Path,
    out: Path,
    meta,
    *,
    start: float,
    duration: float,
    target_height: int,
    quality: int,
    preset: str,
    cq: int,
    hdr_look: str,
    target_bitrate: int,
    max_bitrate: int,
    buffer_bitrate: int,
) -> int:
    import cupy as cp
    import numpy as np
    import PyNvVideoCodec as nvc

    from offline.two_dvr_pynv import _NV12_RGB_KERNELS
    from pipeline.hdr_look import HDR_LOOK_CUDA, apply_hdr_look, normalize_hdr_look
    from pipeline.pynv_io import GpuNv12AppFrame, GpuP016Frame, PyNvSimpleDecoder, PyNvThreadedSerialDecoder
    from utils.rtx_vsr import load_bridge, target_dimensions
    from utils.vr_naming import is_half_equirectangular_source

    bit_depth = int(getattr(meta.codec, "bit_depth", 8) or 8)
    if bit_depth > 8:
        raise RuntimeError("GPU RTX VSR pipeline requires 8-bit NV12 input")

    probe = PyNvSimpleDecoder(src, bit_depth=bit_depth)
    try:
        info = probe.info
        total_source_frames = len(probe)
    finally:
        probe.stop()
    width, height = int(info.width), int(info.height)
    fps = float(meta.timing.source_fps or info.fps or 30.0)
    out_w, out_h = target_dimensions(width, height, target_height)
    split_eyes = int(target_height) >= 2160 and is_half_equirectangular_source(width, height)
    if split_eyes and (width % 2 or out_w != out_h * 2):
        raise RuntimeError(f"split-eye RTX VSR requires even-width 2:1 SBS input; got {width}x{height}")
    start_frame = max(0, int(round(max(0.0, start) * fps)))
    available = max(0, total_source_frames - start_frame)
    frame_count = min(available, int(round(duration * fps))) if duration > 0 else available
    if frame_count <= 0:
        raise RuntimeError("no source frames selected")

    module = cp.RawModule(code=_NV12_RGB_KERNELS + HDR_LOOK_CUDA + _NV12_TO_SPLIT_RGBA)
    k_to_rgb = module.get_function("nv12_to_rgb")
    k_to_split_rgba = module.get_function("nv12_to_split_rgba")
    k_to_nv12 = module.get_function("rgba_to_nv12")
    effective_hdr = normalize_hdr_look(hdr_look)
    k_hdr = module.get_function("hdr_look_rgba") if effective_hdr != "off" else None

    bridge = load_bridge()
    if not bridge.initialize_cupy(cp):
        raise RuntimeError("RTX VSR bridge failed to initialize on the PyNv CUDA context")

    bitrate = max(1, int(target_bitrate))
    effective_preset = str(preset or "p4").strip().upper()
    if effective_preset not in {"P1", "P4", "P7"}:
        effective_preset = "P4"
    enc_kwargs = {
        "codec": "hevc",
        "fps": f"{fps:.9f}",
        "bitrate": str(bitrate),
        "maxbitrate": str(max(1, int(max_bitrate))),
        "vbvbufsize": str(max(1, int(buffer_bitrate))),
        "rc": "vbr",
        "cq": str(max(0, int(cq))),
        "preset": effective_preset,
        "gop": str(config.PASSTHROUGH_GOP),
        "bf": str(config.PASSTHROUGH_HEVC_BF),
        "repeatspspps": "1",
    }
    encoder = nvc.CreateEncoder(out_w, out_h, "NV12", False, **enc_kwargs)
    decoder = PyNvThreadedSerialDecoder(
        src,
        bit_depth=bit_depth,
        start_frame=start_frame,
        info=info,
        num_frames=total_source_frames,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{out.stem}_gpu_", suffix=".mp4", dir=out.parent, delete=False,
    )
    temp_video = Path(temp_handle.name)
    temp_handle.close()
    mux_cmd, mux = _open_video_muxer(
        temp_video,
        fps,
        list(meta.color.ffmpeg_args()),
    )
    if mux.stdin is None:
        decoder.stop()
        mux.terminate()
        mux.wait()
        temp_video.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg mux stdin unavailable")

    rgb = None if split_eyes else cp.empty((height, width, 3), cp.uint8)
    rgba = None if split_eyes else cp.empty((height, width, 4), cp.uint8)
    eye_in_w = width // 2 if split_eyes else 0
    eye_out_w = out_w // 2 if split_eyes else 0
    left_eye = cp.empty((height, eye_in_w, 4), cp.uint8) if split_eyes else None
    right_eye = cp.empty((height, eye_in_w, 4), cp.uint8) if split_eyes else None
    split_output = cp.empty((out_h, out_w, 4), cp.uint8) if split_eyes else None
    ring = [
        cp.empty((out_h * 3 // 2, out_w), cp.uint8)
        for _ in range(2 if out_w >= 8192 else max(2, int(config.PASSTHROUGH_NV12_RING_SLOTS)))
    ]
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    grid_out = ((out_w + 15) // 16, (out_h + 15) // 16, 1)
    started = time.monotonic()
    next_progress = started + 5.0
    count = 0
    mux_error = ""
    timing_enabled = os.environ.get("PT_RTX_VSR_STAGE_TIMING", "0").strip() == "1"
    timing_warmup_frames = 10
    timing_samples = 0
    timing_cpu_total = {
        "decode": 0.0,
        "input_sync": 0.0,
        "left_vsr_call": 0.0,
        "right_vsr_call": 0.0,
        "whole_vsr_call": 0.0,
        "output_sync": 0.0,
        "encode": 0.0,
        "mux": 0.0,
    }
    timing_gpu_total = {
        "nv12_convert": 0.0,
        "split_rgba": 0.0,
        "left_vsr": 0.0,
        "right_vsr": 0.0,
        "whole_vsr": 0.0,
        "hdr_look": 0.0,
        "rgba_to_nv12": 0.0,
    }
    timing_events = (
        {name: cp.cuda.Event() for name in ("start", "rgb", "split", "left", "vsr", "hdr", "nv12")}
        if timing_enabled else None
    )
    print(
        f"[rtx-vsr] pipeline=pynv-gpu input={width}x{height} output={out_w}x{out_h} "
        f"frames={frame_count} fps={fps:.6f} quality={quality} preset={effective_preset.lower()} "
        f"hdr_look={effective_hdr} bitrate={bitrate} "
        f"split_eyes={f'{eye_out_w}x{out_h}+{eye_out_w}x{out_h}' if split_eyes else 'off'}",
        flush=True,
    )
    print("[rtx-vsr] mux=" + subprocess.list2cmdline(mux_cmd), flush=True)
    try:
        for index in range(start_frame, start_frame + frame_count):
            cpu_started = time.perf_counter()
            frame = decoder.frame_at(index)
            cpu_after_decode = time.perf_counter()
            if isinstance(frame, GpuP016Frame):
                raise RuntimeError("unexpected P016 frame in 8-bit RTX VSR pipeline")
            cp.cuda.Device().synchronize()
            cpu_after_input_sync = time.perf_counter()
            y = frame.y.as_cupy(cp.uint8).reshape(height, width)
            uv = frame.uv.as_cupy(cp.uint8).reshape(height // 2, width)
            if timing_events is not None:
                timing_events["start"].record()
            cpu_left_started = cpu_after_input_sync
            cpu_left_done = cpu_after_input_sync
            cpu_right_started = cpu_after_input_sync
            cpu_right_done = cpu_after_input_sync
            cpu_whole_started = cpu_after_input_sync
            cpu_whole_done = cpu_after_input_sync
            if split_eyes:
                k_to_split_rgba(
                    grid,
                    block,
                    (y, uv, left_eye, right_eye, np.int32(width), np.int32(height), np.int32(eye_in_w)),
                )
                if timing_events is not None:
                    timing_events["rgb"].record()
                    timing_events["split"].record()
                cpu_left_started = time.perf_counter()
                split_output[:, :eye_out_w] = bridge.process_cupy_rgba(
                    left_eye, (eye_out_w, out_h), quality,
                )
                cpu_left_done = time.perf_counter()
                if timing_events is not None:
                    timing_events["left"].record()
                cpu_right_started = time.perf_counter()
                split_output[:, eye_out_w:] = bridge.process_cupy_rgba(
                    right_eye, (eye_out_w, out_h), quality,
                )
                cpu_right_done = time.perf_counter()
                enhanced = split_output
            else:
                k_to_rgb(grid, block, (y, uv, rgb, np.int32(width), np.int32(height)))
                if timing_events is not None:
                    timing_events["rgb"].record()
                rgba[:, :, :3] = rgb
                rgba[:, :, 3] = 255
                if timing_events is not None:
                    timing_events["split"].record()
                cpu_whole_started = time.perf_counter()
                enhanced = bridge.process_cupy_rgba(rgba, (out_w, out_h), quality)
                cpu_whole_done = time.perf_counter()
            if timing_events is not None:
                timing_events["vsr"].record()
            if k_hdr is not None:
                apply_hdr_look(k_hdr, enhanced, effective_hdr)
            if timing_events is not None:
                timing_events["hdr"].record()
            nv12 = ring[count % len(ring)]
            k_to_nv12(grid_out, block, (enhanced, nv12, np.int32(out_w), np.int32(out_h)))
            if timing_events is not None:
                timing_events["nv12"].record()
            cpu_output_sync_started = time.perf_counter()
            cp.cuda.get_current_stream().synchronize()
            cpu_after_output_sync = time.perf_counter()
            flags = 0
            if count == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            app_frame = GpuNv12AppFrame(nv12, out_w, out_h)
            cpu_encode_started = time.perf_counter()
            bitstream = encoder.Encode(app_frame, flags) if flags else encoder.Encode(app_frame)
            cpu_after_encode = time.perf_counter()
            if bitstream:
                mux.stdin.write(bitstream)
            cpu_after_mux = time.perf_counter()
            if timing_events is not None and count >= timing_warmup_frames:
                timing_samples += 1
                timing_cpu_total["decode"] += cpu_after_decode - cpu_started
                timing_cpu_total["input_sync"] += cpu_after_input_sync - cpu_after_decode
                timing_cpu_total["left_vsr_call"] += cpu_left_done - cpu_left_started
                timing_cpu_total["right_vsr_call"] += cpu_right_done - cpu_right_started
                timing_cpu_total["whole_vsr_call"] += cpu_whole_done - cpu_whole_started
                timing_cpu_total["output_sync"] += cpu_after_output_sync - cpu_output_sync_started
                timing_cpu_total["encode"] += cpu_after_encode - cpu_encode_started
                timing_cpu_total["mux"] += cpu_after_mux - cpu_after_encode
                elapsed_ms = cp.cuda.get_elapsed_time
                timing_gpu_total["nv12_convert"] += elapsed_ms(timing_events["start"], timing_events["rgb"])
                timing_gpu_total["split_rgba"] += elapsed_ms(timing_events["rgb"], timing_events["split"])
                if split_eyes:
                    timing_gpu_total["left_vsr"] += elapsed_ms(timing_events["split"], timing_events["left"])
                    timing_gpu_total["right_vsr"] += elapsed_ms(timing_events["left"], timing_events["vsr"])
                else:
                    timing_gpu_total["whole_vsr"] += elapsed_ms(timing_events["split"], timing_events["vsr"])
                timing_gpu_total["hdr_look"] += elapsed_ms(timing_events["vsr"], timing_events["hdr"])
                timing_gpu_total["rgba_to_nv12"] += elapsed_ms(timing_events["hdr"], timing_events["nv12"])
            count += 1
            now = time.monotonic()
            if now >= next_progress:
                elapsed = max(1e-6, now - started)
                processing_fps = count / elapsed
                percent = count * 100.0 / frame_count
                eta = (frame_count - count) / max(1e-6, processing_fps)
                print(
                    f"[rtx-vsr] progress={percent:.1f}% frames={count}/{frame_count} "
                    f"fps={processing_fps:.2f} elapsed={_format_time(elapsed)} eta={_format_time(eta)}",
                    flush=True,
                )
                next_progress = now + 5.0
        tail = encoder.EndEncode()
        if tail:
            mux.stdin.write(tail)
        if timing_enabled and timing_samples > 0:
            cpu_parts = " ".join(
                f"{name}={total * 1000.0 / timing_samples:.3f}"
                for name, total in timing_cpu_total.items()
                if total > 0.0
            )
            gpu_parts = " ".join(
                f"{name}={total / timing_samples:.3f}"
                for name, total in timing_gpu_total.items()
                if total > 0.0
            )
            print(f"[rtx-vsr] stage_cpu_avg_ms samples={timing_samples} {cpu_parts}", flush=True)
            print(f"[rtx-vsr] stage_gpu_avg_ms samples={timing_samples} {gpu_parts}", flush=True)
    finally:
        processing_failed = sys.exc_info()[0] is not None
        decoder.stop()
        if mux.stdin and not mux.stdin.closed:
            mux.stdin.close()
        if mux.stderr:
            mux_error = (mux.stderr.read() or b"").decode("utf-8", "replace")
        mux.wait()
        if processing_failed:
            temp_video.unlink(missing_ok=True)
    if mux.returncode != 0:
        temp_video.unlink(missing_ok=True)
        raise RuntimeError(f"GPU RTX VSR mux failed rc={mux.returncode}: {mux_error.strip()[:500]}")
    try:
        _mux_source_audio(temp_video, src, out, start, duration)
    finally:
        temp_video.unlink(missing_ok=True)
    elapsed = max(1e-6, time.monotonic() - started)
    print(
        f"[rtx-vsr] complete pipeline=pynv-gpu progress=100.0% frames={count}/{frame_count} "
        f"fps={count / elapsed:.2f} elapsed={_format_time(elapsed)} out={out}",
        flush=True,
    )
    return 0
