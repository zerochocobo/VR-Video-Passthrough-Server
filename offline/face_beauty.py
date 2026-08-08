"""Offline face-beautification converter (detect -> restore -> retouch).

Mirrors the CLI shape of ``offline/two_dvr.py`` (``single`` / ``batch`` with
``--start`` / ``--duration`` / ``--segment`` / ``--out-dir`` / ``--recursive`` /
``--skip-existing``) and runs:

    ffmpeg decode (bgr24) -> YuNet detect -> 2DFAN4 landmarks -> GFPGAN /
    RestoreFormer++ restore -> BiSeNet-gated retouch -> paste back ->
    ffmpeg encode (hevc_nvenc) + audio copy

The frame geometry is untouched, so an SBS/VR source comes out with the same
layout and markers; only the pixels inside each detected face change. Model
loading, TensorRT caching and the option set live in
:mod:`offline.face_beauty_engine`.
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from utils.runtime_dll_paths import apply_runtime_dll_paths

# Register the venv's TensorRT/cuDNN/CUDA DLL directories before onnxruntime
# loads its CUDA provider, mirroring the other offline entry points.
apply_runtime_dll_paths()

from offline.face_beauty_engine import (
    DEFAULT_ENHANCER,
    DEFAULT_PRESET,
    ENHANCER_MODELS,
    ENHANCER_NONE,
    DETECT_MODES,
    MIN_FACE_MODES,
    PRESET_FIELDS,
    PRESETS,
    BeautyOptions,
    FaceBeautyEngine,
    build_trt_stage,
    ensure_models_available,
    normalize_enhancer,
    preset_options,
    trt_cache_keys,
    trt_cache_ready,
)
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.vr_naming import FACE_BEAUTY_SUFFIX

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}
DEFAULT_MAX_SIDE = 0            # 0 = keep the source resolution
ENHANCER_CHOICES = [ENHANCER_NONE, *ENHANCER_MODELS]


def log(msg: str) -> None:
    print(f"[beauty] {msg}", flush=True)


# --- progress ---------------------------------------------------------------


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if not np.isfinite(value):
        return "--:--"
    total = max(0, int(round(value)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _estimated_frames(fps: float, duration: float) -> int:
    try:
        return max(0, int(round(float(fps) * max(0.0, float(duration)))))
    except (TypeError, ValueError):
        return 0


def _progress_message(done: int, total: int, started: float, faces: int) -> str:
    elapsed = max(0.0, time.time() - started)
    fps = float(done) / max(1e-6, elapsed)
    if total > 0:
        percent = min(100.0, max(0.0, float(done) * 100.0 / float(total)))
        remaining = max(0, int(total) - int(done))
        eta = (float(remaining) / fps) if fps > 1e-6 and remaining > 0 else 0.0
        return (f"{done}/{total} frames ({percent:5.1f}%) faces={faces} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} {fps:.2f} fps")
    return f"{done} frames faces={faces} elapsed={_format_duration(elapsed)} eta=--:-- {fps:.2f} fps"


# --- probing ----------------------------------------------------------------


def probe_video(path: Path) -> tuple[int, int, float, float]:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    raw = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace",
                                  **hidden_subprocess_kwargs())
    data = json.loads(raw)
    stream = (data.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"invalid video metadata for {path}: {width}x{height} fps={fps}")
    return width, height, fps, duration


def _parse_fps(rate: str) -> float:
    text = str(rate or "").strip()
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(text or 0.0)


def _has_audio(path: Path) -> bool:
    cmd = [FFPROBE, "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=index", "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace",
                                      **hidden_subprocess_kwargs())
        return bool(out.strip())
    except Exception:
        return False


def _processing_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0 or max(width, height) <= max_side:
        w, h = width, height
    else:
        scale = max_side / float(max(width, height))
        w = int(round(width * scale))
        h = int(round(height * scale))
    # even dims keep yuv420 / NVENC happy
    return w - (w & 1), h - (h & 1)


# --- output naming ----------------------------------------------------------


def _time_tag(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}{m:02d}{s:02d}"


def output_path(src: Path, out_dir: Path | None, start: float, duration: float,
                segments: list | None = None) -> Path:
    parent = out_dir if out_dir else src.parent
    if segments:
        seg = f"_SEG{len(segments)}_S{_time_tag(segments[0][0])}_E{_time_tag(segments[-1][1])}"
    elif duration > 0:
        seg = f"_S{_time_tag(start)}_E{_time_tag(start + duration)}"
    elif start > 0:
        seg = f"_S{_time_tag(start)}"
    else:
        seg = ""
    return parent / f"{src.stem}{seg}{FACE_BEAUTY_SUFFIX}.mp4"


# --- ffmpeg decode / encode -------------------------------------------------


def _decode_proc(src: Path, start: float, duration: float, proc_w: int, proc_h: int):
    cmd = [FFMPEG, "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src), "-an", "-vf", f"scale={proc_w}:{proc_h}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


def _encode_proc(src: Path, out: Path, out_w: int, out_h: int, fps: float,
                 start: float, duration: float, preset: str, bitrate: str, with_audio: bool):
    cmd = [FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
           "-r", f"{fps:.6f}", "-i", "-"]
    if with_audio:
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        if duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", config.PASSTHROUGH_VCODEC, "-preset", str(preset), "-b:v", str(bitrate),
            "-pix_fmt", "yuv420p", "-shortest", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


FACE_BEAUTY_BITRATE_MULT = 1.5


def _effective_bitrate(args, src: Path) -> int:
    """Beautification keeps the frame size, so the source bitrate is the right
    reference. Cap at 1.5x it: enough headroom that the re-encode does not eat
    the detail the restorer just added, without inflating the file."""
    from utils.bitrate_estimator import projection_capped_bitrate

    return projection_capped_bitrate(getattr(args, "bitrate", "40M"), src, "flat3d",
                                     FACE_BEAUTY_BITRATE_MULT, FACE_BEAUTY_BITRATE_MULT)


# --- options ----------------------------------------------------------------


def _percent(value) -> float:
    try:
        return float(np.clip(float(value) / 100.0, 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def options_from_args(args) -> BeautyOptions:
    """Layer the CLI over the chosen preset.

    Every tunable defaults to ``None``, so an explicit flag overrides the preset
    and an omitted one inherits it. That keeps ``--preset strong --skin-smooth 20``
    doing the obvious thing."""
    options = preset_options(getattr(args, "beauty_preset", None))

    for field in PRESET_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            setattr(options, field, _percent(value))

    padding = tuple(
        float(getattr(args, f"mask_padding_{side}", None) or 0.0)
        for side in ("top", "right", "bottom", "left")
    )
    overrides = {
        "enhancer": normalize_enhancer(getattr(args, "enhancer", None) or DEFAULT_ENHANCER),
        "mask_padding": padding,
        "provider": str(getattr(args, "provider", "trt") or "trt").lower(),
    }
    mask_blur = getattr(args, "mask_blur", None)
    if mask_blur is not None:
        overrides["mask_blur"] = _percent(mask_blur)
    temporal = getattr(args, "temporal_smooth", None)
    if temporal is not None:
        overrides["temporal_smooth"] = _percent(temporal)
    region_mask = getattr(args, "region_mask", None)
    if region_mask is not None:
        overrides["use_region_mask"] = bool(region_mask)
    landmarker = getattr(args, "landmarker", None)
    if landmarker is not None:
        overrides["use_landmarker"] = bool(landmarker)
    detector_score = getattr(args, "detector_score", None)
    if detector_score is not None:
        overrides["detector_score"] = float(detector_score)
    vr_reproject = getattr(args, "vr_reproject", None)
    if vr_reproject is not None:
        overrides["vr_reproject"] = str(vr_reproject)
    detect_mode = getattr(args, "detect_mode", None)
    if detect_mode is not None:
        overrides["detect_mode"] = str(detect_mode)
    detect_roi = getattr(args, "detect_roi", None)
    if detect_roi is not None:
        overrides["detect_roi"] = bool(detect_roi)
    roi_sweep = getattr(args, "roi_sweep_interval", None)
    if roi_sweep is not None:
        overrides["roi_sweep_interval"] = max(1, int(roi_sweep))
    detect_interval = getattr(args, "detect_interval", None)
    if detect_interval is not None:
        overrides["detect_interval"] = max(1, int(detect_interval))
    landmark_interval = getattr(args, "landmark_interval", None)
    if landmark_interval is not None:
        overrides["landmark_interval"] = max(1, int(landmark_interval))
    min_face_mode = getattr(args, "min_face_mode", None)
    if min_face_mode is not None:
        overrides["min_face_mode"] = str(min_face_mode)
    max_faces = getattr(args, "max_faces", None)
    if max_faces is not None:
        overrides["max_faces"] = int(max_faces)
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


# --- GPU-resident clip pipeline (NVDEC -> cupy -> models -> NVENC) ----------


def _mux_proc(src: Path, out: Path, fps: float, start: float, duration: float):
    """Mux our NVENC HEVC elementary stream (stdin) with the source audio,
    trimmed to the same window. Video is copied -- it is already encoded on the
    GPU -- so only audio is transcoded. Mirrors offline/demosaic_offline.py."""
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "hevc", "-r", f"{fps:.6f}", "-i", "-"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


def gpu_pipeline_available() -> bool:
    try:
        import cupy  # noqa: F401
        import PyNvVideoCodec  # noqa: F401
        return True
    except Exception:
        return False


def convert_clip_gpu(src: Path, out: Path, options: BeautyOptions, args,
                     start: float, duration: float) -> int:
    """Frames never leave the GPU: NVDEC -> CuPy RGB -> detect/restore/retouch
    -> NVENC, with ffmpeg only muxing the audio.

    At 8K this is what matters -- the ffmpeg-pipe path moves an 88 MB frame
    across a process boundary twice per frame and downscales 7680x3840 to the
    detector's 640 on the CPU."""
    import cupy as cp
    import PyNvVideoCodec as nvc

    from offline.face_beauty_gpu import GpuFaceBeautyProcessor
    from offline.two_dvr_pynv import _NV12_RGB_KERNELS
    from pipeline.pynv_io import GpuNv12AppFrame, GpuP016Frame, PyNvSimpleDecoder, PyNvThreadedSerialDecoder
    from pipeline.pynv_stream import _pynv_encoder_kwargs
    from utils.video_metadata import probe_video_metadata

    meta = probe_video_metadata(src)
    bit_depth = int(meta.codec.bit_depth if meta.codec and meta.codec.bit_depth > 0 else 8)
    shift_bits = int(config.PASSTHROUGH_PYNV_10BIT_SHIFT)

    meta_dec = PyNvSimpleDecoder(src, bit_depth=bit_depth)
    info = meta_dec.info
    dec_len = len(meta_dec)
    width, height = int(info.width), int(info.height)
    fps = float(info.fps or (meta.timing.source_fps if meta.timing else 0.0) or 30.0)
    meta_dec.stop()

    first = min(max(0, int(round(start * fps))), max(0, dec_len - 1))
    last = min(dec_len, first + int(round(duration * fps))) if duration > 0 else dec_len
    last = max(first + 1, last)
    total = last - first

    processor = GpuFaceBeautyProcessor(options, log=log)
    log(f"{src.name}: {width}x{height}@{fps:.3f} frames[{first},{last}) "
        f"{options.retouch_summary()} {processor.provider_summary()} pipeline=gpu")

    dec = PyNvThreadedSerialDecoder(
        src, bit_depth=bit_depth, start_frame=first,
        batch_size=config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
        buffer_size=config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
        info=info, num_frames=dec_len,
    )
    enc = nvc.CreateEncoder(width, height, "NV12", False,
                            **_pynv_encoder_kwargs(bitrate=str(_effective_bitrate(args, src)),
                                                   fps=f"{fps:.6f}"))
    mux = _mux_proc(src, out, fps, start, duration)
    mux_err: list[bytes] = []

    def _drain() -> None:
        try:
            for line in iter(mux.stderr.readline, b""):
                mux_err.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain, name="beauty-mux-stderr", daemon=True)
    stderr_thread.start()

    mod = cp.RawModule(code=_NV12_RGB_KERNELS)
    k_to_rgb = mod.get_function("nv12_to_rgb")
    k_p016_to_rgb = mod.get_function("p016_to_rgb")
    k_to_nv12 = mod.get_function("rgb_to_nv12")
    rgb_g = cp.empty((height, width, 3), cp.uint8)
    out_nv12 = cp.empty((height * 3 // 2, width), cp.uint8)
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)

    produced = 0
    faces = 0
    started = time.time()
    processor.reset()
    try:
        for index in range(first, last):
            frame = dec.frame_at(index).owned_copy()
            if isinstance(frame, GpuP016Frame):
                y_g = frame.y.as_cupy(cp.uint16).reshape(height, width)
                uv_g = frame.uv.as_cupy(cp.uint16).reshape(height // 2, width)
                k_p016_to_rgb(grid, block, (y_g, uv_g, rgb_g, np.int32(width), np.int32(height),
                                            np.int32(shift_bits)))
            else:
                y_g = frame.y.as_cupy(cp.uint8).reshape(height, width)
                uv_g = frame.uv.as_cupy(cp.uint8).reshape(height // 2, width)
                k_to_rgb(grid, block, (y_g, uv_g, rgb_g, np.int32(width), np.int32(height)))

            _, stats = processor.process(rgb_g)
            faces += stats.processed

            k_to_nv12(grid, block, (rgb_g, out_nv12, np.int32(width), np.int32(height)))
            cp.cuda.get_current_stream().synchronize()
            flags = 0
            if produced == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            bitstream = enc.Encode(GpuNv12AppFrame(out_nv12, width, height), flags)
            if bitstream:
                mux.stdin.write(bitstream)
            produced += 1
            if produced % 32 == 0:
                log(f"  {_progress_message(produced, total, started, faces)}")
        tail = enc.EndEncode()
        if tail:
            mux.stdin.write(tail)
        try:
            mux.stdin.close()
        except Exception:
            pass
        mux.wait(timeout=120)
    finally:
        try:
            dec.stop()
        except Exception:
            pass
        if mux.poll() is None:
            mux.kill()
    stderr_thread.join(timeout=5)
    if mux.returncode not in (0, None) or not out.is_file():
        error = b"".join(mux_err).decode("utf-8", "replace").strip()
        log(f"mux failed rc={mux.returncode}: {error[:800]}")
        return 1
    elapsed = max(1e-3, time.time() - started)
    log(f"done {out.name}: {produced} frames, {faces} faces in {elapsed:.1f}s "
        f"({produced / elapsed:.2f} fps)")
    return 0


# --- per-clip pipeline ------------------------------------------------------


def convert_clip(src: Path, out: Path, engine: FaceBeautyEngine, args, start: float, duration: float) -> int:
    width, height, fps, total = probe_video(src)
    proc_w, proc_h = _processing_size(width, height, args.max_side)
    with_audio = _has_audio(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    log(f"{src.name}: {width}x{height}@{fps:.3f} -> proc {proc_w}x{proc_h} "
        f"{engine.options.retouch_summary()} {engine.provider_summary()}")

    dec = _decode_proc(src, start, duration, proc_w, proc_h)
    enc = _encode_proc(src, out, proc_w, proc_h, fps, start, duration,
                       args.preset, _effective_bitrate(args, src), with_audio)
    started = time.time()
    engine.reset()
    clip_duration = duration if duration > 0 else max(0.0, float(total or 0.0) - max(0.0, start))
    count, faces, dec_err, enc_err = _pump_pipeline(
        dec, enc, engine, proc_w, proc_h, started, total_frames=_estimated_frames(fps, clip_duration),
    )

    if enc.returncode not in (0, None):
        log(f"encode failed rc={enc.returncode}: {enc_err.strip()[:400]}")
        return 1
    if count == 0:
        log(f"no frames decoded: {dec_err.strip()[:400]}")
        return 1
    elapsed = time.time() - started
    log(f"done {out.name}: {count} frames, {faces} faces in {elapsed:.1f}s "
        f"({count / max(1e-6, elapsed):.2f} fps)")
    return 0


def _pump_pipeline(dec, enc, engine, proc_w, proc_h, started,
                   total_frames: int = 0) -> tuple[int, int, str, str]:
    """3-stage pipeline so the ffmpeg pipes overlap the GPU work:

        T-read  : decode ingest             -> q_in
        main    : detect + restore (GPU)    -> q_out
        T-write : encode ingest             -> ffmpeg

    Both pipe stages release the GIL, so decode(N+1) genuinely overlaps
    process(N). Returns (frames, faces, decode_stderr, encode_stderr)."""
    frame_bytes = proc_w * proc_h * 3
    q_in: queue.Queue = queue.Queue(maxsize=4)
    q_out: queue.Queue = queue.Queue(maxsize=4)

    def read_stage() -> None:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            # Writable copy: the engine pastes faces in place.
            q_in.put(np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3).copy())
        q_in.put(None)

    def write_stage() -> None:
        while True:
            frame = q_out.get()
            if frame is None:
                break
            try:
                enc.stdin.write(frame)
            except (BrokenPipeError, OSError):
                break

    read_t = threading.Thread(target=read_stage, name="beauty-read", daemon=True)
    write_t = threading.Thread(target=write_stage, name="beauty-write", daemon=True)
    read_t.start()
    write_t.start()

    count = 0
    faces = 0
    while True:
        frame = q_in.get()
        if frame is None:
            break
        frame, stats = engine.process(frame)
        faces += stats.processed
        q_out.put(np.ascontiguousarray(frame))
        count += 1
        if count % 32 == 0:
            log(f"  {_progress_message(count, total_frames, started, faces)}")
    q_out.put(None)
    write_t.join()
    read_t.join(timeout=1.0)

    if enc.stdin:
        try:
            enc.stdin.close()
        except Exception:
            pass
    dec_err = (dec.stderr.read() or b"").decode("utf-8", "replace") if dec.stderr else ""
    enc.wait()
    enc_err = (enc.stderr.read() or b"").decode("utf-8", "replace") if enc.stderr else ""
    dec.wait()
    return count, faces, dec_err, enc_err


def _run_segments(engine: FaceBeautyEngine, args, src: Path, segments: list, out: Path) -> int:
    """Render multiple time segments concatenated into one output."""
    width, height, fps, _ = probe_video(src)
    proc_w, proc_h = _processing_size(width, height, args.max_side)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = _encode_proc(src, out, proc_w, proc_h, fps, 0.0, 0.0, args.preset,
                       _effective_bitrate(args, src), False)
    total = 0
    faces = 0
    total_expected = sum(_estimated_frames(fps, end - begin) for begin, end in segments)
    started = time.time()
    try:
        for begin, end in segments:
            engine.reset()
            dec = _decode_proc(src, begin, end - begin, proc_w, proc_h)
            count, segment_faces = _pump_segment(
                dec, enc, engine, proc_w, proc_h,
                started=started, progress_offset=total, faces_offset=faces, total_frames=total_expected,
            )
            total += count
            faces += segment_faces
            dec.wait()
    finally:
        if enc.stdin:
            try:
                enc.stdin.close()
            except Exception:
                pass
        enc.wait()
    log(f"done {out.name}: {total} frames, {faces} faces (segments) in {time.time() - started:.1f}s")
    return 0 if total > 0 and enc.returncode in (0, None) else 1


def _pump_segment(dec, enc, engine, proc_w, proc_h, started: float, progress_offset: int,
                  faces_offset: int, total_frames: int) -> tuple[int, int]:
    """Single-segment compute loop writing into a shared (already open) encoder."""
    frame_bytes = proc_w * proc_h * 3
    count = 0
    faces = 0
    while True:
        raw = dec.stdout.read(frame_bytes)
        if not raw or len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3).copy()
        frame, stats = engine.process(frame)
        faces += stats.processed
        try:
            enc.stdin.write(np.ascontiguousarray(frame))
        except (BrokenPipeError, OSError):
            break
        count += 1
        if (progress_offset + count) % 32 == 0:
            log(f"  {_progress_message(progress_offset + count, total_frames, started, faces_offset + faces)}")
    return count, faces


# --- run orchestration ------------------------------------------------------


def _parse_time_text(text: str) -> float | None:
    value = str(text or "").strip()
    if not value:
        return None
    if ":" not in value:
        try:
            return max(0.0, float(value))
        except ValueError:
            return None
    parts = value.split(":")
    if len(parts) not in (2, 3) or any(not p.strip().isdigit() for p in parts):
        return None
    nums = [int(p) for p in parts]
    if len(nums) == 2:
        h, m, s = 0, nums[0], nums[1]
    else:
        h, m, s = nums
    return float(h * 3600 + m * 60 + s)


def _segment_arg(text: str) -> tuple[float, float]:
    value = str(text or "").strip()
    if "-" not in value:
        raise argparse.ArgumentTypeError("segment must be START-END")
    a, b = value.split("-", 1)
    start = _parse_time_text(a)
    end = _parse_time_text(b)
    if start is None or end is None or end <= start:
        raise argparse.ArgumentTypeError("segment times invalid (need START<END, HH:MM:SS)")
    return start, end


def _video_files(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    out = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if path.stem.lower().endswith(FACE_BEAUTY_SUFFIX.lower()):
            continue
        out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


def _use_gpu_pipeline(args) -> bool:
    """The GPU pipeline handles a single contiguous range. Segment lists still
    go through the ffmpeg path, which can seek per segment into one encoder."""
    mode = str(getattr(args, "pipeline", "auto") or "auto").lower()
    if mode == "ffmpeg":
        return False
    if getattr(args, "segment", None):
        return False
    if str(getattr(args, "provider", "trt") or "trt").lower() == "cpu":
        return False
    if int(getattr(args, "max_side", 0) or 0) > 0:
        return False          # the GPU path encodes at the source resolution
    if not gpu_pipeline_available():
        if mode == "gpu":
            raise RuntimeError("cupy / PyNvVideoCodec unavailable")
        return False
    return True


def _run_one(engine_factory, options: BeautyOptions, args, src: Path) -> int:
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", "") else None
    segments = getattr(args, "segment", None) or None
    if segments:
        out = output_path(src, out_dir, 0.0, 0.0, segments)
        if args.skip_existing and out.exists():
            log(f"skip existing: {out.name}")
            return 0
        return _run_segments(engine_factory(), args, src, segments, out)
    start = float(getattr(args, "start", 0.0) or 0.0)
    duration = float(getattr(args, "duration", 0.0) or 0.0)
    out = output_path(src, out_dir, start, duration)
    if args.skip_existing and out.exists():
        log(f"skip existing: {out.name}")
        return 0
    if _use_gpu_pipeline(args):
        try:
            return convert_clip_gpu(src, out, options, args, start, duration)
        except Exception as exc:
            if str(getattr(args, "pipeline", "auto")).lower() == "gpu":
                raise
            log(f"GPU pipeline unavailable ({type(exc).__name__}: {exc}); using the ffmpeg pipeline")
    return convert_clip(src, out, engine_factory(), args, start, duration)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    # Every tunable below defaults to None: unset means "inherit the preset".
    # Named --beauty-preset, not --preset: the video group's --preset is the
    # NVENC preset, matching offline/convert.py and offline/two_dvr.py.
    p.add_argument("--beauty-preset", dest="beauty_preset", choices=list(PRESETS), default=DEFAULT_PRESET,
                   help="beauty strength preset; individual flags below override it")

    face = p.add_argument_group("face restoration")
    face.add_argument("--enhancer", choices=ENHANCER_CHOICES, default=DEFAULT_ENHANCER,
                      help="blind face restoration model; 'none' runs retouch only")
    face.add_argument("--enhancer-blend", dest="enhancer_blend", type=float, default=None,
                      help="0-100 blend of the restored face over the original")

    retouch = p.add_argument_group("retouch (0-100; unset = preset value)")
    retouch.add_argument("--skin-smooth", dest="skin_smooth", type=float, default=None,
                         help="edge-preserving skin smoothing")
    retouch.add_argument("--skin-brighten", dest="skin_brighten", type=float, default=None,
                         help="raise skin luminance")
    retouch.add_argument("--skin-even", dest="skin_even", type=float, default=None,
                         help="pull skin chroma toward its mean (evens blotches/redness)")
    retouch.add_argument("--eye-brighten", dest="eye_brighten", type=float, default=None)
    retouch.add_argument("--teeth-white", dest="teeth_white", type=float, default=None)
    retouch.add_argument("--lip-vivid", dest="lip_vivid", type=float, default=None)
    retouch.add_argument("--sharpen", dest="sharpen", type=float, default=None,
                         help="unsharp mask over the face crop")

    mask = p.add_argument_group("mask")
    mask.add_argument("--mask-blur", dest="mask_blur", type=float, default=None,
                      help="0-100 feather of the face box mask (default 30)")
    mask.add_argument("--mask-padding-top", dest="mask_padding_top", type=float, default=None)
    mask.add_argument("--mask-padding-right", dest="mask_padding_right", type=float, default=None)
    mask.add_argument("--mask-padding-bottom", dest="mask_padding_bottom", type=float, default=None)
    mask.add_argument("--mask-padding-left", dest="mask_padding_left", type=float, default=None)
    mask.add_argument("--region-mask", dest="region_mask", action=argparse.BooleanOptionalAction, default=None,
                      help="restrict edits to parsed face parts (keeps glasses/hair untouched)")

    detect = p.add_argument_group("detection")
    detect.add_argument("--detector-score", dest="detector_score", type=float, default=None)
    detect.add_argument("--vr-reproject", dest="vr_reproject", choices=["auto", "off", "on"],
                        default=None,
                        help="process each face through a gnomonic flat view (auto = SBS/VR "
                             "sources). The affine face warp assumes a perspective camera, so on "
                             "an equirect eye it is only exact near the horizon")
    detect.add_argument("--detect-mode", dest="detect_mode", choices=list(DETECT_MODES), default=None,
                        help="auto tiles the detector over large frames (YuNet's input is a fixed "
                             "640, so a whole-frame letterbox loses small faces); full forces one "
                             "pass, tiled always grids")
    detect.add_argument("--detect-roi", dest="detect_roi", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="between full sweeps, detect only around the faces found last time. "
                             "At 8K the full grid is 18 windows; following two known faces is two")
    detect.add_argument("--roi-sweep-interval", dest="roi_sweep_interval", type=int, default=None,
                        help="detections between full sweeps when --detect-roi is on")
    detect.add_argument("--detect-interval", dest="detect_interval", type=int, default=None,
                        help="run detection every N frames and reuse the boxes in between; "
                             "landmarks still update every frame")
    detect.add_argument("--landmark-interval", dest="landmark_interval", type=int, default=None,
                        help="run 68-point landmark refinement every N frames and reuse stable "
                             "points in between")
    detect.add_argument("--min-face-mode", dest="min_face_mode", choices=list(MIN_FACE_MODES), default=None,
                        help="smallest face to process, as a share of frame height: "
                             "auto (1.5%%, min 24px) / loose (everything) / strict (5%%, foreground only)")
    detect.add_argument("--max-faces", dest="max_faces", type=int, default=None,
                        help="process at most N largest faces per frame (0 = all)")
    detect.add_argument("--landmarker", dest="landmarker", action=argparse.BooleanOptionalAction, default=None,
                        help="2DFAN4 68-point alignment (steadier than detector keypoints)")
    detect.add_argument("--temporal-smooth", dest="temporal_smooth", type=float, default=None,
                        help="0-100 EMA on tracked landmarks; reduces face wobble across frames")

    video = p.add_argument_group("video")
    video.add_argument("--max-side", dest="max_side", type=int, default=DEFAULT_MAX_SIDE,
                       help="downscale longer side before processing (0 = original)")
    video.add_argument("--preset", default="p5")
    video.add_argument("--bitrate", default="40M")
    video.add_argument("--provider", default="trt", choices=["trt", "cuda", "cpu"],
                       help="trt = TensorRT fp16 (fastest, builds a cached engine on first run)")
    video.add_argument("--pipeline", default="auto", choices=["auto", "gpu", "ffmpeg"],
                       help="auto/gpu = GPU-resident NVDEC->NVENC (frames never leave the GPU); "
                            "ffmpeg = CPU rawvideo pipe (used for --segment and --max-side)")
    video.add_argument("--skip-existing", dest="skip_existing", action="store_true")


def build_trt(options: BeautyOptions) -> int:
    """Build every missing engine, each in its own process.

    Freeing the session objects is not enough: building an engine segfaults when
    another TensorRT session exists in the process, even one merely deserialised
    from cache. Isolation therefore has to come from the OS. Reproduced with
    gpen_bfr_512, whose build crashes after the detector/landmarker/parser
    engines have been loaded, while the same build succeeds alone."""
    from offline.face_beauty_engine import trt_engine_cached

    program, base_args = _self_command()
    rc = 0
    for key in trt_cache_keys(options):
        if trt_engine_cached(key):
            continue
        # Re-emit the child's log through ours: the TensorRT dialog watches this
        # process's output for "build-trt:" lines to drive its progress bar.
        child = subprocess.Popen(
            [program, *base_args, "build-trt", "--stage", key,
             "--enhancer", normalize_enhancer(options.enhancer)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **hidden_subprocess_kwargs(),
        )
        for raw in iter(child.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                print(line, flush=True)
        child.wait()
        if child.returncode != 0:
            log(f"build-trt: stage {key} exited with {child.returncode}; "
                f"it will run on CUDA instead")
            rc = 1
    return rc


def _self_command() -> tuple[str, list[str]]:
    """How to re-invoke this tool as a child process."""
    if getattr(sys, "frozen", False):
        return sys.executable, ["face_beauty"]
    return sys.executable, [str(Path(__file__).resolve())]


def _ensure_trt_cache(options: BeautyOptions) -> int:
    if options.provider != "trt" or trt_cache_ready(options):
        return 0
    log("TensorRT engine cache missing; building before conversion (this can take minutes)...")
    return build_trt(options)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline face beautification (restore + retouch)")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="process one video")
    single.add_argument("video")
    single.add_argument("--out-dir", dest="out_dir", default="")
    single.add_argument("--start", type=float, default=0.0)
    single.add_argument("--duration", type=float, default=0.0)
    single.add_argument("--segment", action="append", type=_segment_arg)
    _add_common_args(single)

    batch = sub.add_parser("batch", help="process all videos under a directory")
    batch.add_argument("directory")
    batch.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    _add_common_args(batch)

    buildtrt = sub.add_parser("build-trt", help="pre-build the TensorRT engine cache")
    buildtrt.add_argument("--stage", default=None,
                          help="build exactly this one engine in-process (used by the "
                               "per-stage child processes; see build_trt)")
    _add_common_args(buildtrt)

    dl = sub.add_parser("download", help="download the ONNX models from Hugging Face")
    _add_common_args(dl)

    args = parser.parse_args(argv)
    options = options_from_args(args)

    if not ensure_models_available(options.enhancer, options.use_landmarker, options.needs_parser(), log=log):
        log("model download failed; aborting.")
        return 2
    if args.command == "download":
        log("models ready")
        return 0
    if args.command == "build-trt":
        stage = getattr(args, "stage", None)
        if stage:
            return build_trt_stage(stage, options, log=log)
        return build_trt(options)

    rc = _ensure_trt_cache(options)
    if rc != 0:
        return rc
    # Built lazily: the GPU pipeline creates its own sessions, so an unused CPU
    # engine would double the VRAM footprint for nothing.
    cached_engine: list[FaceBeautyEngine] = []

    def engine_factory() -> FaceBeautyEngine:
        if not cached_engine:
            cached_engine.append(FaceBeautyEngine(options, log=log))
        return cached_engine[0]

    if args.command == "single":
        src = Path(args.video)
        if not src.is_file():
            log(f"input not found: {src}")
            return 2
        return _run_one(engine_factory, options, args, src)

    root = Path(args.directory)
    if not root.is_dir():
        log(f"directory not found: {root}")
        return 2
    files = _video_files(root, args.recursive)
    if not files:
        log(f"no videos found under {root}")
        return 0
    log(f"batch: {len(files)} videos")
    args.out_dir = ""      # batch writes next to each source
    args.start = 0.0
    args.duration = 0.0
    args.segment = None
    rc = 0
    for index, src in enumerate(files, 1):
        log(f"[{index}/{len(files)}] {src.name}")
        rc |= _run_one(engine_factory, options, args, src)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
