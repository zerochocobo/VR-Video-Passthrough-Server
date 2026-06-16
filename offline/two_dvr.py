"""Offline 2D -> VR/3D converter (DA3 depth + stereo + projection).

Mirrors the CLI shape of ``offline/convert.py`` (``single`` / ``batch`` with
``--start`` / ``--duration`` / ``--segment`` / ``--out-dir`` / ``--recursive`` /
``--skip-existing``) but runs a self-contained pipeline:

    ffmpeg decode (rgb24) -> DA3 ONNX depth -> stereo warp -> VR projection
    -> ffmpeg encode (hevc_nvenc) + audio copy

Depth comes from :mod:`offline.da3_depth` (ORT, da3_small/base.onnx). Stereo and
projection are pure numpy in :mod:`offline.two_dvr_render`.
"""
from __future__ import annotations

import argparse
import json
import queue
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
# loads its CUDA provider, mirroring the offline/server entry points. Without
# this the CUDAExecutionProvider silently falls back to CPU.
apply_runtime_dll_paths()

from offline import two_dvr_render as render
from offline.da3_depth import Da3DepthEngine, ensure_model_available, trt_engine_cached
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.vr_naming import TWO_DVR_SUFFIX

import shutil

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}
DEFAULT_MAX_SIDE = 1920
DEFAULT_BATCH = 4
OUTPUT_MARKER = "_2dvr_"


def log(msg: str) -> None:
    print(f"[2dvr] {msg}", flush=True)


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


# --- sizing -----------------------------------------------------------------


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


def output_path(src: Path, out_dir: Path | None, projection: str, variant: str,
                start: float, duration: float, segments: list | None = None) -> Path:
    parent = out_dir if out_dir else src.parent
    proj_tag = {"flat3d": "flat3d", "hequirect": "heq180", "fisheye": "fish180"}.get(projection, projection)
    if segments:
        seg = f"_SEG{len(segments)}_S{_time_tag(segments[0][0])}_E{_time_tag(segments[-1][1])}"
    elif duration > 0:
        seg = f"_S{_time_tag(start)}_E{_time_tag(start + duration)}"
    elif start > 0:
        seg = f"_S{_time_tag(start)}"
    else:
        seg = ""
    if projection == render.PROJECTION_FLAT_3D:
        return parent / f"{src.stem}{seg}{TWO_DVR_SUFFIX}.mp4"
    return parent / f"{src.stem}{OUTPUT_MARKER}{variant}_{proj_tag}_LR_SBS{seg}.mp4"


# --- ffmpeg decode / encode -------------------------------------------------


def _decode_proc(src: Path, start: float, duration: float, proc_w: int, proc_h: int):
    cmd = [FFMPEG, "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src), "-an", "-vf", f"scale={proc_w}:{proc_h}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


def _encode_proc(src: Path, out: Path, out_w: int, out_h: int, fps: float,
                 start: float, duration: float, preset: str, bitrate: str, with_audio: bool):
    cmd = [FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
           "-r", f"{fps:.6f}", "-i", "-"]
    if with_audio:
        # Audio is taken from the source as a second input, trimmed to match the
        # same [start, start+duration) window used for the video frames.
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        if duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", config.PASSTHROUGH_VCODEC, "-preset", preset, "-b:v", bitrate,
            "-pix_fmt", "yuv420p", "-shortest", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


# --- per-clip pipeline ------------------------------------------------------


def _make_renderer(proc_w, proc_h, args):
    """GPU renderer for the flat3d + inverse_warp fast path when available and
    requested; CPU StereoRenderer otherwise."""
    gpu = getattr(args, "gpu_render", "auto")
    eligible = args.hole_fill in (render.HOLE_FILL_INVERSE_WARP, render.HOLE_FILL_SOFT_SHIFT)
    eye_distance = render.effective_eye_distance_mm(args.eye_distance, getattr(args, "strength", render.DEFAULT_STRENGTH))
    if gpu != "off" and eligible:
        try:
            from offline.two_dvr_gpu import GpuStereoRenderer, gpu_available

            if gpu_available():
                renderer = GpuStereoRenderer(
                    proc_w, proc_h, args.projection, eye_distance, args.hole_fill, args.flat_fov
                )
                log("renderer: GPU (cupy)")
                return renderer
            if gpu == "on":
                raise RuntimeError("cupy GPU not available")
        except Exception as exc:
            if gpu == "on":
                raise
            log(f"GPU renderer unavailable ({type(exc).__name__}: {exc}); using CPU")
    return render.StereoRenderer(
        proc_w, proc_h, args.projection, eye_distance, args.hole_fill, args.flat_fov
    )


def convert_clip(src: Path, out: Path, engine: Da3DepthEngine, args, start: float, duration: float) -> int:
    width, height, fps, total = probe_video(src)
    proc_w, proc_h = _processing_size(width, height, args.max_side)
    renderer = _make_renderer(proc_w, proc_h, args)
    out_w, out_h = renderer.out_w, renderer.out_h
    with_audio = _has_audio(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    strength = render.strength_multiplier(getattr(args, "strength", render.DEFAULT_STRENGTH))
    log(f"{src.name}: {width}x{height}@{fps:.3f} -> proc {proc_w}x{proc_h} -> SBS {out_w}x{out_h} "
        f"proj={args.projection} fill={args.hole_fill} strength={strength:.2f} "
        f"model={args.model} depth={engine.providers[0]}")

    dec = _decode_proc(src, start, duration, proc_w, proc_h)
    enc = _encode_proc(src, out, out_w, out_h, fps, start, duration, args.preset, args.bitrate, with_audio)
    started = time.time()
    count, dec_err, enc_err = _pump_pipeline(dec, enc, engine, renderer, proc_w, proc_h, started)

    if enc.returncode not in (0, None):
        log(f"encode failed rc={enc.returncode}: {enc_err.strip()[:400]}")
        return 1
    if count == 0:
        log(f"no frames decoded: {dec_err.strip()[:400]}")
        return 1
    elapsed = time.time() - started
    log(f"done {out.name}: {count} frames in {elapsed:.1f}s ({count / max(1e-6, elapsed):.1f} fps)")
    return 0


def _pump_pipeline(dec, enc, engine, renderer, proc_w, proc_h, started, log_progress: bool = True) -> tuple[int, str, str]:
    """3-stage pipeline so the GPU depth pass overlaps the CPU render:

        T-depth : read frame + DA3 depth (GPU)  -> q_depth
        main    : render (CPU/cv2)              -> q_out
        T-write : encode ingest                 -> ffmpeg

    The heavy stages (TensorRT run, cv2.remap, pipe IO) all release the GIL, so
    depth(frame N+1) genuinely runs while render(frame N) does. Returns
    (frames, decode_stderr, encode_stderr)."""
    frame_bytes = proc_w * proc_h * 3
    q_depth: queue.Queue = queue.Queue(maxsize=4)
    q_out: queue.Queue = queue.Queue(maxsize=4)

    def depth_stage() -> None:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)
            q_depth.put((frame, engine.predict_batch([frame], upscale=False)[0]))
        q_depth.put(None)

    def write_stage() -> None:
        while True:
            sbs = q_out.get()
            if sbs is None:
                break
            try:
                enc.stdin.write(sbs)
            except (BrokenPipeError, OSError):
                break

    depth_t = threading.Thread(target=depth_stage, name="2dvr-depth", daemon=True)
    write_t = threading.Thread(target=write_stage, name="2dvr-write", daemon=True)
    depth_t.start()
    write_t.start()

    count = 0
    while True:
        item = q_depth.get()
        if item is None:
            break
        frame, depth = item
        sbs = renderer.render(frame, depth)
        # Copy out of the renderer's reused buffer; the writer streams the array
        # straight to the pipe via the buffer protocol (no tobytes copy).
        q_out.put(sbs.copy())
        count += 1
        if log_progress and count % 64 == 0:
            log(f"  {count} frames ({count / max(1e-6, time.time() - started):.1f} fps)")
    q_out.put(None)
    write_t.join()
    depth_t.join(timeout=1.0)

    if enc.stdin:
        try:
            enc.stdin.close()
        except Exception:
            pass
    dec_err = (dec.stderr.read() or b"").decode("utf-8", "replace") if dec.stderr else ""
    enc.wait()
    enc_err = (enc.stderr.read() or b"").decode("utf-8", "replace") if enc.stderr else ""
    dec.wait()
    return count, dec_err, enc_err


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
        if OUTPUT_MARKER in path.name or path.stem.lower().endswith(TWO_DVR_SUFFIX.lower()):
            continue
        out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


def _run_one(engine: Da3DepthEngine, args, src: Path) -> int:
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", "") else None
    segments = getattr(args, "segment", None) or None
    if segments:
        out = output_path(src, out_dir, args.projection, args.model, 0.0, 0.0, segments)
        if args.skip_existing and out.exists():
            log(f"skip existing: {out.name}")
            return 0
        # Encode each segment to a temp then concat would add complexity; for v1
        # we process the span covering all segments by running them sequentially
        # into separate outputs is undesirable. Instead render the union as one
        # output by concatenating segment frames.
        return _run_segments(engine, args, src, segments, out)
    start = float(getattr(args, "start", 0.0) or 0.0)
    duration = float(getattr(args, "duration", 0.0) or 0.0)
    out = output_path(src, out_dir, args.projection, args.model, start, duration)
    if args.skip_existing and out.exists():
        log(f"skip existing: {out.name}")
        return 0
    if _use_pynv(args):
        try:
            from offline.two_dvr_pynv import convert_clip_pynv
            return convert_clip_pynv(src, out, engine, args, start, duration, log=log)
        except Exception as exc:
            if getattr(args, "pipeline", "auto") == "pynv":
                raise
            log(f"GPU-resident pipeline unavailable ({type(exc).__name__}: {exc}); using ffmpeg pipeline")
    return convert_clip(src, out, engine, args, start, duration)


def _use_pynv(args) -> bool:
    """GPU-resident pipeline is available for the flat3d + inverse_warp fast path."""
    if getattr(args, "pipeline", "auto") == "ffmpeg":
        return False
    try:
        from offline.two_dvr_pynv import supported
        return supported(args.projection, args.hole_fill)
    except Exception:
        return False


def _run_segments(engine, args, src, segments, out) -> int:
    """Render multiple time segments concatenated into one SBS output."""
    width, height, fps, _ = probe_video(src)
    proc_w, proc_h = _processing_size(width, height, args.max_side)
    renderer = _make_renderer(proc_w, proc_h, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = _encode_proc(src, out, renderer.out_w, renderer.out_h, fps, 0.0, 0.0, args.preset, args.bitrate, False)
    total = 0
    started = time.time()
    try:
        for seg_start, seg_end in segments:
            dec = _decode_proc(src, seg_start, seg_end - seg_start, proc_w, proc_h)
            count = _pump_segment(dec, enc, engine, renderer, proc_w, proc_h)
            total += count
            dec.wait()
    finally:
        if enc.stdin:
            try:
                enc.stdin.close()
            except Exception:
                pass
        enc.wait()
    log(f"done {out.name}: {total} frames (segments) in {time.time() - started:.1f}s")
    return 0 if total > 0 and enc.returncode in (0, None) else 1


def _pump_segment(dec, enc, engine, renderer, proc_w, proc_h) -> int:
    """Single-segment compute loop that writes into a shared (already open)
    encoder; the encoder/decoder lifecycle is managed by the caller."""
    frame_bytes = proc_w * proc_h * 3
    count = 0
    while True:
        raw = dec.stdout.read(frame_bytes)
        if not raw or len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)
        depth = engine.predict_batch([frame], upscale=False)[0]
        sbs = renderer.render(frame, depth)
        try:
            enc.stdin.write(np.ascontiguousarray(sbs).tobytes())
        except (BrokenPipeError, OSError):
            break
        count += 1
    return count


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=["small", "base", "small_hd", "base_hd"], default="base")
    p.add_argument("--projection", choices=sorted(render.PROJECTIONS), default=render.DEFAULT_PROJECTION)
    p.add_argument("--hole-fill", dest="hole_fill", choices=sorted(render.HOLE_FILL_MODES),
                   default=render.DEFAULT_HOLE_FILL_MODE)
    p.add_argument("--eye-distance", dest="eye_distance", type=float, default=render.DEFAULT_EYE_DISTANCE_MM)
    p.add_argument("--strength", type=float, default=render.DEFAULT_STRENGTH,
                   help="3D strength multiplier; 1.0 matches the default 65mm baseline")
    p.add_argument("--flat-fov", dest="flat_fov", type=float, default=render.DEFAULT_FLAT_FOV_DEG)
    p.add_argument("--max-side", dest="max_side", type=int, default=DEFAULT_MAX_SIDE,
                   help="downscale longer side before processing (0 = original)")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--preset", default="p5")
    p.add_argument("--bitrate", default="40M")
    p.add_argument("--provider", default="trt", choices=["trt", "cuda", "cpu"],
                   help="trt = TensorRT fp16 (fastest, builds a cached engine on first run)")
    p.add_argument("--gpu-render", dest="gpu_render", default="auto", choices=["auto", "on", "off"],
                   help="cupy GPU stereo render for flat3d+inverse_warp (auto = use if available)")
    p.add_argument("--pipeline", default="auto", choices=["auto", "pynv", "ffmpeg"],
                   help="auto/pynv = GPU-resident NVDEC->NVENC (flat3d+inverse_warp); ffmpeg = CPU pipe")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true")


def _build_trt(model: str) -> int:
    """Pre-build the DA3 TensorRT fp16 engine cache by running one inference per
    variant. ORT's TensorRT EP compiles + caches the engine on first run
    (~3-15s); doing it here means the first real conversion is instant."""
    from offline.da3_depth import DA3_PRESETS, default_model_path

    variants = list(DA3_PRESETS) if model in {"both", "all"} else [model]
    dummy = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    rc = 0
    for variant in variants:
        if not default_model_path(variant).exists():
            log(f"build-trt: {variant} onnx not found, skipping")
            continue
        log(f"build-trt: building TensorRT engine for DA3 {variant} (first time is slow)...")
        t0 = time.time()
        try:
            engine = Da3DepthEngine(variant=variant, provider="trt")
            engine.predict_batch([dummy], upscale=False)
            log(f"build-trt: {variant} ready on {engine.providers[0]} in {time.time() - t0:.1f}s")
        except Exception as exc:
            log(f"DA3 TensorRT auto-build failed for {variant}: {type(exc).__name__}: {exc}")
            rc = 1
    return rc


def _ensure_trt_cache(model: str, provider: str) -> int:
    if provider != "trt" or trt_engine_cached(model):
        return 0
    log(f"DA3 TensorRT engine cache missing for {model}; building before conversion...")
    return _build_trt(model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline 2D->VR/3D converter (DA3 depth)")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="convert one video")
    single.add_argument("video")
    single.add_argument("--out-dir", dest="out_dir", default="")
    single.add_argument("--start", type=float, default=0.0)
    single.add_argument("--duration", type=float, default=0.0)
    single.add_argument("--segment", action="append", type=_segment_arg)
    _add_common_args(single)

    batch = sub.add_parser("batch", help="convert all videos under a directory")
    batch.add_argument("directory")
    batch.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    _add_common_args(batch)

    buildtrt = sub.add_parser("build-trt", help="pre-build the DA3 TensorRT engine cache")
    buildtrt.add_argument("--model", choices=["small", "base", "small_hd", "base_hd", "both"], default="both")

    dl = sub.add_parser("download", help="download DA3 ONNX model(s) from Hugging Face")
    dl.add_argument("--model", choices=["small", "base", "small_hd", "base_hd", "both"], default="both")

    args = parser.parse_args(argv)

    if args.command == "download":
        from offline.da3_depth import DA3_PRESETS, download_model
        models = list(DA3_PRESETS) if args.model == "both" else [args.model]
        rc = 0
        for m in models:
            try:
                download_model(m, log=log)
            except Exception as exc:
                log(f"download failed for {m}: {type(exc).__name__}: {exc}")
                rc = 1
        return rc

    if args.command == "build-trt":
        if args.model != "both" and not ensure_model_available(args.model, log=log):
            return 2
        return _build_trt(args.model)

    if not ensure_model_available(args.model, log=log):
        log(f"DA3 model {args.model} unavailable and download failed; aborting.")
        return 2
    rc = _ensure_trt_cache(args.model, args.provider)
    if rc != 0:
        return rc
    engine = Da3DepthEngine(variant=args.model, provider=args.provider)

    if args.command == "single":
        src = Path(args.video)
        if not src.is_file():
            log(f"input not found: {src}")
            return 2
        return _run_one(engine, args, src)

    root = Path(args.directory)
    if not root.is_dir():
        log(f"directory not found: {root}")
        return 2
    files = _video_files(root, args.recursive)
    if not files:
        log(f"no videos found under {root}")
        return 0
    log(f"batch: {len(files)} videos")
    args.out_dir = ""  # batch writes next to each source
    args.start = 0.0
    args.duration = 0.0
    args.segment = None
    rc = 0
    for index, src in enumerate(files, 1):
        log(f"[{index}/{len(files)}] {src.name}")
        rc |= _run_one(engine, args, src)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
