"""Offline passthrough video generator.

This tool reuses the optimized PyNv decode -> GPU composite -> NVENC -> MP4
mux path, but writes a finished file instead of serving a live stream. The
default engine is RVM. The MatAnyone2 ONNX engine processes SBS videos per eye:
SAM3 or a supplied mask bootstraps each segment, MatAnyone2 propagates each eye
independently, and the two alpha halves are stitched back for final compositing.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.bitrate_estimator import effective_default_bitrate, parse_bitrate, source_video_bitrate  # noqa: E402
from utils.gpu_runtime_cache import configure_gpu_runtime_cache  # noqa: E402
from utils.subprocess_hidden import hidden_subprocess_kwargs, run_hidden_streaming  # noqa: E402
from utils.scene_detection import SceneCutDetector  # noqa: E402
from utils.trt_manifest import (  # noqa: E402
    MATANYONE2_TRT_ONNX_NAME,
    TRT_MODEL_MATANYONE2,
    TRT_MODEL_RVM,
    TRT_PROVIDER_CHAIN,
    cache_dir_for_model,
    cache_status,
    is_matanyone2_trt_model_dir,
    matanyone2_trt_cache_dir_for_model_dir,
    original_rvm_model_path,
)
from utils.video_metadata import cfr_source_index, probe_color_metadata, probe_timing_metadata, probe_video_metadata, select_backend  # noqa: E402
from utils.vr_naming import offline_passthrough_stem  # noqa: E402
from offline.sam3_matanyone2 import (  # noqa: E402
    Sam3TextMasker,
    apply_sam3_stereo_guard,
    clear_gpu_memory_pools,
    empty_sam3_mask,
    fill_short_inactive_gaps,
)
from offline.decoded_frames import decoded_frame_to_bgr  # noqa: E402
from offline.matanyone2_engine import MatAnyone2OnnxEngine as SharedMatAnyone2OnnxEngine  # noqa: E402

GPU_CACHE_ENV = configure_gpu_runtime_cache()


def _configure_offline_rvm_tensorrt(model_path: Path) -> bool:
    providers = [p.strip() for p in config.ONNX_PROVIDERS if p.strip()]
    wants_trt = "TensorrtExecutionProvider" in providers
    is_mobile = model_path.resolve() == original_rvm_model_path().resolve()
    if wants_trt and is_mobile:
        try:
            if cache_status(model_key=TRT_MODEL_RVM, scope="offline") == "ready":
                cache_dir = cache_dir_for_model(TRT_MODEL_RVM, scope="offline")
                config.ONNX_TRT_ENGINE_CACHE_PATH = cache_dir
                os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(cache_dir)
                config.ONNX_PROVIDERS = [p.strip() for p in TRT_PROVIDER_CHAIN.split(",") if p.strip()]
                print(f"[offline] RVM TensorRT enabled providers={config.ONNX_PROVIDERS}")
                return True
        except Exception as exc:
            print(f"[offline] RVM TensorRT unavailable, using CUDA providers ({type(exc).__name__}: {exc})")
    if wants_trt:
        config.ONNX_PROVIDERS = [p for p in providers if p != "TensorrtExecutionProvider"] or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[offline] RVM TensorRT disabled for model={model_path.name}; providers={config.ONNX_PROVIDERS}")
    return False


def _available_onnx_providers():
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    return [p for p in providers if p in available] or ["CPUExecutionProvider"]


def _matanyone2_session_providers(name: str, model_dir: Path):
    if os.environ.get("PT_OFFLINE_MATANYONE2_TRT") != "1":
        return _available_onnx_providers()
    if name != MATANYONE2_TRT_ONNX_NAME:
        return _available_onnx_providers()
    try:
        if not is_matanyone2_trt_model_dir(model_dir):
            print(f"[offline] MatAnyone2 TensorRT disabled for custom model dir={model_dir}", flush=True)
            return _available_onnx_providers()
        trt_root = cache_dir_for_model(TRT_MODEL_MATANYONE2)
        if cache_status(model_key=TRT_MODEL_MATANYONE2, cache_dir=trt_root) != "ready":
            print("[offline] MatAnyone2 TensorRT cache is not ready; using CUDA providers", flush=True)
            return _available_onnx_providers()
        cache_dir = matanyone2_trt_cache_dir_for_model_dir(model_dir, trt_root)
        config.ONNX_TRT_ENGINE_CACHE_PATH = cache_dir
        os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(cache_dir)
        matting_module = importlib.import_module("pipeline.matting")

        matting_module.ONNX_TRT_ENGINE_CACHE_PATH = cache_dir
        providers = matting_module._filter_available_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        if "TensorrtExecutionProvider" not in providers:
            print("[offline] MatAnyone2 TensorRT provider unavailable; using CUDA providers", flush=True)
            return _available_onnx_providers()
        print(f"[offline] MatAnyone2 TensorRT enabled model={name} cache={cache_dir}", flush=True)
        return matting_module._provider_config(providers)
    except Exception as exc:
        print(f"[offline] MatAnyone2 TensorRT unavailable, using CUDA providers ({type(exc).__name__}: {exc})", flush=True)
        return _available_onnx_providers()


def _sam3_onnx_providers(provider: str, cuda_memory_limit_mb: int):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if provider == "cpu" or "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]
    options = {}
    if cuda_memory_limit_mb > 0:
        options["gpu_mem_limit"] = str(int(cuda_memory_limit_mb) * 1024 * 1024)
        options["arena_extend_strategy"] = "kSameAsRequested"
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]


def _provider_summary(providers) -> str:
    parts = []
    for provider in providers:
        if isinstance(provider, tuple):
            name, options = provider
            cap = options.get("gpu_mem_limit") if isinstance(options, dict) else None
            if cap:
                parts.append(f"{name}(arena_cap={int(cap) // (1024 * 1024)}MiB)")
            else:
                parts.append(str(name))
        else:
            parts.append(str(provider))
    return "[" + ", ".join(parts) + "]"


def _resolve_matanyone2_model_dir(args, width: int = 0, height: int = 0) -> Path:
    supported_sizes = (512, 1024)
    if getattr(args, "model", ""):
        model_dir = Path(args.model).resolve()
        manifest_path = model_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            height = int(manifest.get("height") or 0)
            width = int(manifest.get("width") or 0)
            if (height, width) not in {(size, size) for size in supported_sizes}:
                supported = ", ".join(f"{size}x{size}" for size in supported_sizes)
                raise RuntimeError(
                    f"unsupported MatAnyone2 model size {width}x{height}; supported: {supported}"
                )
        return model_dir
    size = int(getattr(args, "matanyone2_size", 1024) or 1024)
    if size not in supported_sizes:
        supported = ", ".join(str(item) for item in supported_sizes)
        raise RuntimeError(f"unsupported MatAnyone2 size {size}; supported: {supported}")
    batch_arg = str(getattr(args, "matanyone2_batch", "1") or "1").lower()
    if batch_arg == "auto":
        # Batch2 is memory-bandwidth/workspace heavy in ORT CUDA on this
        # pipeline and benchmarks slower than two batch1 eye passes.
        batch = 1
    else:
        batch = int(batch_arg)
    model_dir = config.ROOT / "models" / f"matanyone2_onnx_{size}_bs{batch}"
    if not model_dir.exists():
        fallback = config.ROOT / "models" / "matanyone2_onnx"
        if fallback.exists():
            print(f"[offline] warning: {model_dir} not found, falling back to {fallback}")
            return _resolve_matanyone2_model_dir(type("_Args", (), {"model": str(fallback)})())
    return model_dir.resolve()


def _patch_tempdir() -> None:
    fixed = config.RUNTIME_TMP_DIR
    fixed.mkdir(parents=True, exist_ok=True)

    class FixedTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.name = str(fixed)

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

        def cleanup(self):
            return None

    tempfile.TemporaryDirectory = FixedTemporaryDirectory


def _resolve_video(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        if p.exists():
            return p.resolve()
        p = config.VIDEO_DIR / p
    return p.resolve()


def _default_out(src: Path, width: int = 0, height: int = 0) -> Path:
    return src.with_name(f"{offline_passthrough_stem(src.stem, 'green', width, height)}.mp4")


def _open_muxer(out: Path, fps: float, src: Path, codec: str):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    input_format = "hevc" if codec.lower() in {"hevc", "h265"} else "h264"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        input_format,
        "-framerate",
        f"{fps:.6f}",
        "-i",
        "-",
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        *probe_color_metadata(src).ffmpeg_args(),
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(out),
    ]
    return cmd, subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **hidden_subprocess_kwargs(),
    )


def _extract_audio_sidecar(src: Path, out: Path, audio: str, start_sec: float = 0.0, duration: float = 0.0) -> tuple[list[str], subprocess.CompletedProcess, Path]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    audio_path = out.with_name(f"{out.stem}._audio.aac")
    codec_args = ["-c:a", "copy"] if audio == "copy" else ["-c:a", "aac", "-b:a", "192k"]
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        *(
            ["-ss", f"{start_sec:.6f}"]
            if start_sec > 0
            else []
        ),
        "-i",
        str(src),
        "-map",
        "0:a:0?",
        "-vn",
        *codec_args,
        *(
            ["-t", f"{duration:.6f}"]
            if duration > 0
            else []
        ),
        "-f",
        "adts",
        str(audio_path),
    ]
    return cmd, subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **hidden_subprocess_kwargs(),
    ), audio_path


def _cleanup_audio_sidecar(audio_path: Path | None) -> None:
    if audio_path is None:
        return
    try:
        audio_path.unlink(missing_ok=True)
    except Exception:
        pass


def _mux_audio_sidecar_after(video_only: Path, out: Path, audio_path: Path, src: Path, duration: float = 0.0) -> tuple[list[str], subprocess.CompletedProcess]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(video_only),
        "-f",
        "aac",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        *(
            ["-t", f"{duration:.6f}"]
            if duration > 0
            else []
        ),
        *probe_color_metadata(src).ffmpeg_args(),
        "-movflags",
        "+faststart",
        str(out),
    ]
    return cmd, subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **hidden_subprocess_kwargs(),
    )


def _ffprobe(path: Path) -> str:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames,color_space,color_range,color_transfer,color_primaries",
        "-of",
        "default=nw=1",
        str(path),
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **hidden_subprocess_kwargs(),
    )
    return (p.stdout + p.stderr).strip()


def _probe_keyframe_indices(path: Path, source_fps: float, target: int, output_fps: float) -> list[int]:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,coded_picture_number",
        "-of",
        "json",
        str(path),
    ]
    try:
        data = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs()))
    except Exception:
        return []
    indices = []
    for frame in data.get("frames") or []:
        idx = None
        ts = frame.get("best_effort_timestamp_time") or frame.get("pkt_pts_time")
        if ts is not None:
            try:
                idx = int(round(float(ts) * output_fps))
            except Exception:
                idx = None
        if idx is None:
            try:
                coded = int(frame.get("coded_picture_number"))
                idx = int(round(coded * output_fps / source_fps)) if source_fps > 0 else coded
            except Exception:
                idx = None
        if idx is not None and 0 <= idx < target:
            indices.append(idx)
    return sorted(set(indices))


def _parse_bitrate(value: str, src: Path) -> str:
    raw = str(value or "source").strip().lower()
    if raw in {"live", "realtime", "passthrough"}:
        return str(effective_default_bitrate(src, "pynv_hevc").bps)
    if raw in {"", "source", "auto", "same"}:
        bitrate = source_video_bitrate(src)
        if not bitrate:
            return "40000000"
        return str(bitrate)
    return str(value)


def _encoder_bitrate_kwargs(args: argparse.Namespace, src: Path) -> tuple[dict[str, str], int, int, int]:
    target_text = _parse_bitrate(args.bitrate, src)
    target_bps = parse_bitrate(target_text)
    max_bps = int(target_bps * max(1.0, float(args.maxrate_multiplier)))
    buf_bps = int(target_bps * max(1.0, float(args.bufsize_multiplier)))
    kwargs = {
        "bitrate": str(target_bps),
        "maxbitrate": str(max_bps),
        "vbvbufsize": str(buf_bps),
        "rc": str(args.rc),
    }
    if args.cq >= 0:
        kwargs["cq"] = str(int(args.cq))
    if args.preset:
        kwargs["preset"] = str(args.preset).upper()
    return kwargs, target_bps, max_bps, buf_bps


def _mem_stats() -> tuple[float, float]:
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / 1e6, pool.total_bytes() / 1e6
    except Exception:
        return 0.0, 0.0


def _nvidia_mem_stats() -> tuple[int, int]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0, 0
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            **hidden_subprocess_kwargs(),
        )
        first = out.strip().splitlines()[0]
        used, total = [int(x.strip()) for x in first.split(",")[:2]]
        return used, total
    except Exception:
        return 0, 0


def _require_sam3_vram(args) -> None:
    if args.engine != "matanyone2_onnx" or args.mask or args.matanyone2_prepass != "sam3" or args.sam3_provider != "cuda":
        return
    if args.sam3_min_vram_gb <= 0:
        return
    _used, total = _nvidia_mem_stats()
    if total <= 0:
        print("[offline] warning: cannot query GPU VRAM; SAM3 prepass requires a high-memory CUDA GPU")
        return
    required_mib = int(float(args.sam3_min_vram_gb) * 1024)
    if total < required_mib:
        raise RuntimeError(
            f"SAM3 prepass requires at least {args.sam3_min_vram_gb:g}GB VRAM "
            f"(detected {total / 1024:.1f}GB). Use RVM or provide --mask on lower-VRAM GPUs."
        )


def _summary(prefix: str, values: list[float]) -> list[str]:
    if not values:
        return [f"{prefix}_avg = 0.000 ms", f"{prefix}_p99 = 0.000 ms"]
    ordered = sorted(values)
    p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
    return [f"{prefix}_avg = {statistics.fmean(values):.3f} ms", f"{prefix}_p99 = {p99:.3f} ms"]


def _cuda_sync_ms(device: bool = False) -> float:
    try:
        from pipeline import matting as matting_module

        t0 = time.perf_counter()
        if device:
            cp = getattr(matting_module, "_cp", None)
            if cp is None:
                return 0.0
            cp.cuda.Device(0).synchronize()
        else:
            stream = getattr(matting_module, "_CUDA_STREAM", None)
            if stream is None:
                return 0.0
            stream.synchronize()
        return (time.perf_counter() - t0) * 1000
    except Exception:
        return 0.0


class OfflineMattingEngine:
    def composite_nv12(self, frame):
        raise NotImplementedError


class RvmOfflineEngine(OfflineMattingEngine):
    def __init__(self, model: Path | None = None):
        if model is not None:
            config.MODEL_PATH = model
            _configure_offline_rvm_tensorrt(model)
        from pipeline.matting import Matter

        self.matter = Matter()
        print(f"[offline] RVM runtime provider_diag={self.matter._rvm_provider_diag()}", file=sys.stderr)
        self.matter.reset_state()
        self.profile: dict[str, list[float]] = defaultdict(list)
        self._pending_nv12_slots = []
        self._max_pending_nv12_slots = max(0, int(config.PASSTHROUGH_NV12_RING_SLOTS) - 1)

    def composite_nv12(self, frame):
        from pipeline.pynv_io import GpuP016Frame

        h, w = int(frame.height), int(frame.width)
        out_slot = self.matter.acquire_nv12_output_slot(h, w)
        if isinstance(frame, GpuP016Frame):
            out, timing = self.matter.composite_green_gpu_p016_frame_to_gpu_nv12_profile(
                frame,
                shift_bits=int(config.PASSTHROUGH_PYNV_10BIT_SHIFT),
                out_h=h,
                out_w=w,
                out_slot=out_slot,
            )
        else:
            out, timing = self.matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(
                frame,
                out_h=h,
                out_w=w,
                out_slot=out_slot,
            )
        self._pending_nv12_slots.append(out_slot)
        while len(self._pending_nv12_slots) > self._max_pending_nv12_slots:
            self.matter.release_nv12_output_slot(self._pending_nv12_slots.pop(0))
        self.profile["preprocess"].append(timing.preprocess_ms)
        self.profile["ort"].append(timing.ort_ms)
        self.profile["composite"].append(timing.composite_ms)
        return out, timing

    def release_pending_outputs(self) -> None:
        while self._pending_nv12_slots:
            self.matter.release_nv12_output_slot(self._pending_nv12_slots.pop(0))

    def profile_lines(self) -> list[str]:
        lines = []
        for key in ("preprocess", "ort", "composite"):
            values = self.profile.get(key)
            if values:
                lines.append(f"rvm_{key}_avg = {statistics.fmean(values):.3f} ms n={len(values)}")
        return lines


def _make_engine(args) -> OfflineMattingEngine:
    name = args.engine
    if name == "rvm":
        model_path = Path(args.model).resolve() if args.model else (config.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx").resolve()
        return RvmOfflineEngine(model_path)
    if name == "matanyone2_onnx":
        model_dir = Path(args._matanyone2_model_dir).resolve()
        return SharedMatAnyone2OnnxEngine(
            model_dir,
            Path(args.mask).resolve() if args.mask else None,
            Path(args.sam3_model_dir).resolve(),
            _matanyone2_session_providers,
            sam3_prompt=args.sam3_prompt,
            bootstrap_threshold=args.matanyone2_bootstrap_threshold,
            bootstrap_erode=args.matanyone2_bootstrap_erode,
            bootstrap_dilate=args.matanyone2_bootstrap_dilate,
            bootstrap_soft=args.matanyone2_bootstrap_soft,
            segment_frames=args.matanyone2_segment_frames,
            use_fused_update=args.matanyone2_fused_update,
            use_step_update=args.matanyone2_step_update,
            output_mode="green",
            log_prefix="[offline]",
        )
    raise RuntimeError(f"unknown engine: {name}")


def _object_count_from_infos(infos: list[dict]) -> int:
    return max((len(info.get("selected") or []) for info in infos), default=0)


def _planned_starts_from_sam3_records(
    records: list[dict],
    target: int,
    max_frames: int,
    min_frames: int,
    cut_on_count_change: bool,
    cut_every_active_sample: bool,
    scene_min_frames: int = 0,
) -> list[int]:
    if not records:
        return [0]
    starts = [int(records[0]["frame"])]
    last = starts[0]
    last_active = bool(records[0]["active"])
    last_count = int(records[0]["object_count"])
    for record in records[1:]:
        idx = int(record["frame"])
        active = bool(record["active"])
        count = int(record["object_count"])
        force_cut = active != last_active
        if active and cut_on_count_change and count != last_count:
            force_cut = True
        if active and cut_every_active_sample:
            force_cut = True
        if active and bool(record.get("scene_cut")) and (scene_min_frames <= 0 or idx - last >= scene_min_frames):
            force_cut = True
        timed_cut = (min_frames > 0 and idx - last >= min_frames) or (max_frames > 0 and idx - last >= max_frames)
        if force_cut or timed_cut:
            starts.append(idx)
            last = idx
            last_active = active
            last_count = count
            continue
        last_active = active
        last_count = max(last_count, count) if active else count
    while max_frames > 0 and target - last > max_frames:
        last += max_frames
        starts.append(last)
    return sorted(set(x for x in starts if 0 <= x < target))


def _precompute_sam3_segment_masks(args, src: Path, dec, source_fps: float, fps: float, target: int):
    if args.engine != "matanyone2_onnx" or args.mask:
        return {}, [0]
    if args.sam3_subprocess and not getattr(args, "_sam3_child", False):
        return _precompute_sam3_segment_masks_subprocess(args, src, source_fps, fps, target)

    import cv2
    import numpy as np

    max_segment_frames = max(1, int(args.matanyone2_segment_frames or target))
    min_segment_frames = max(1, int(round(max(0.0, args.matanyone2_min_segment_sec) * fps)))
    scene_detector = (
        SceneCutDetector(
            threshold=config.MATANYONE2_SCENE_THRESHOLD,
            cooldown_frames=config.MATANYONE2_SCENE_COOLDOWN,
            ref_ema_alpha=config.MATANYONE2_SCENE_REF_EMA,
        )
        if config.MATANYONE2_SCENE_RESET
        else None
    )
    scene_min_frames = (
        max(1, int(round(max(0.0, config.MATANYONE2_SCENE_MIN_SEGMENT_SEC) * fps)))
        if scene_detector is not None
        else 0
    )
    if args.sam3_scan in {"keyframe", "hybrid"}:
        candidates = _probe_keyframe_indices(src, source_fps, target, fps)
        if args.sam3_scan == "hybrid":
            step = max(1, int(round(max(0.1, args.sam3_scan_interval_sec) * fps)))
            candidates = sorted(set(candidates) | set(range(0, target, step)))
        if not candidates:
            step = max(1, int(round(max(0.1, args.sam3_scan_interval_sec) * fps)))
            candidates = list(range(0, target, step))
    else:
        step = max(1, int(round(max(0.1, args.sam3_scan_interval_sec) * fps)))
        candidates = list(range(0, target, step))
    scan_points = sorted(set(x for x in candidates if 0 <= x < target))
    if 0 not in scan_points:
        scan_points.insert(0, 0)
    providers = _sam3_onnx_providers(args.sam3_provider, args.sam3_cuda_memory_limit_mb)
    decoder_providers = _sam3_onnx_providers(args.sam3_decoder_provider, args.sam3_decoder_cuda_memory_limit_mb)
    masker = None

    def make_masker():
        return Sam3TextMasker(
            Path(args.sam3_model_dir).resolve(),
            args.sam3_prompt,
            providers,
            decoder_providers=decoder_providers,
            score_threshold=args.sam3_score_threshold,
            min_area_ratio=args.sam3_min_area_ratio,
            max_area_ratio=args.sam3_max_area_ratio,
            top_k=args.sam3_top_k,
            low_memory=args.sam3_low_memory,
        )
    masks_by_start = {}
    debug_dir = Path(args.sam3_debug_dir).resolve() if args.sam3_debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[offline] SAM3 prepass prompt={args.sam3_prompt!r} samples={len(scan_points)} "
        f"scan={args.sam3_scan} max_segment_frames={max_segment_frames} "
        f"scene_reset={int(scene_detector is not None)} "
        f"low_memory={args.sam3_low_memory} "
        f"encoder={_provider_summary(providers)} decoder={_provider_summary(decoder_providers)}"
    )
    release_interval = max(0, int(args.sam3_release_interval))
    records = []
    for n, start in enumerate(scan_points, 1):
        if masker is None:
            masker = make_masker()
        src_idx = min(len(dec) - 1, cfr_source_index(start, source_fps, fps))
        frame = dec.frame_at(src_idx)
        bgr = decoded_frame_to_bgr(frame)
        half = frame.width // 2
        scene_cut = False
        scene_distance = 0.0
        if scene_detector is not None:
            scene_cut = scene_detector.step(bgr[:, :half] if half > 0 else bgr)
            scene_distance = scene_detector.last_distance
        eye_images = [
            cv2.cvtColor(bgr[:, :half], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bgr[:, half:half * 2], cv2.COLOR_BGR2RGB),
        ]
        masks = []
        infos = []
        t0 = time.perf_counter()
        for eye_idx, image_rgb in enumerate(eye_images):
            sam_image, source_size = masker.prepare_image(image_rgb)
            image_out = masker.encode_prepared(None, sam_image)
            del sam_image
            try:
                mask, info = masker.decode_encoded(
                    None,
                    image_out,
                    source_size,
                    out_size=(args._matanyone2_in_w, args._matanyone2_in_h),
                )
            except RuntimeError as exc:
                message = str(exc)
                if "SAM3 returned no masks" not in message and "SAM3 returned empty masks" not in message:
                    raise
                eye_name = "left" if eye_idx == 0 else "right"
                print(
                    f"[offline] SAM3 prepass warning: frame={start} eye={eye_name} "
                    f"has no usable masks; treating this eye as inactive ({message})"
                )
                mask, info = empty_sam3_mask(
                    args._matanyone2_in_w,
                    args._matanyone2_in_h,
                    reason=message,
                )
            infos.append(info)
            masks.append(mask.astype(np.float32, copy=False))
        masks, infos, stereo_mode = apply_sam3_stereo_guard(masks, infos)
        mask_tensors = []
        for eye_idx, (mask, info, image_rgb) in enumerate(zip(masks, infos, eye_images)):
            if debug_dir is not None:
                eye_name = "left" if eye_idx == 0 else "right"
                debug_frame = cv2.resize(
                    image_rgb,
                    (args._matanyone2_in_w, args._matanyone2_in_h),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imwrite(str(debug_dir / f"seg_{start:06d}_{eye_name}_frame.png"), cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(debug_dir / f"seg_{start:06d}_{eye_name}_mask.png"), (mask * 255).astype(np.uint8))
            mask_tensors.append(mask[None, None, :, :].astype(np.float32, copy=False))
        active = (
            infos[0]["union_area_ratio"] >= args.sam3_active_min_area_ratio
            or infos[1]["union_area_ratio"] >= args.sam3_active_min_area_ratio
        )
        object_count = _object_count_from_infos(infos) if active else 0
        if active:
            masks_by_start[start] = mask_tensors
        gpu_used, gpu_total = _nvidia_mem_stats() if args.sam3_log_vram else (0, 0)
        records.append(
            {
                "frame": int(start),
                "src_idx": int(src_idx),
                "active": bool(active),
                "object_count": int(object_count),
                "scene_cut": bool(scene_cut),
            }
        )
        if debug_dir is not None:
            (debug_dir / f"seg_{start:06d}_info.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
        print(
            f"[offline] SAM3 prepass {n}/{len(scan_points)} frame={start} src_idx={src_idx} "
            f"ms={(time.perf_counter() - t0) * 1000:.1f} "
            f"active={active} objects={object_count} "
            f"L={infos[0]['selected']} area={infos[0]['union_area_ratio']:.4f} "
            f"R={infos[1]['selected']} area={infos[1]['union_area_ratio']:.4f} "
            f"stereo={stereo_mode}"
            + (f" scene_cut=1 dist={scene_distance:.3f}" if scene_cut else "")
            + (f" gpu={gpu_used}/{gpu_total}MiB" if gpu_total else "")
        )
        if release_interval and n % release_interval == 0:
            del masker
            masker = None
            clear_gpu_memory_pools()
    if masker is not None:
        del masker
    clear_gpu_memory_pools()
    gap_fill_frames = int(getattr(args, "sam3_gap_fill_frames", max_segment_frames) or 0)
    filled_gaps = fill_short_inactive_gaps(records, masks_by_start, gap_fill_frames)
    if filled_gaps:
        print(
            f"[offline] SAM3 prepass filled short inactive gaps frames={filled_gaps} "
            f"max_gap_frames={gap_fill_frames}",
            flush=True,
        )
    starts = _planned_starts_from_sam3_records(
        records,
        target,
        max_segment_frames,
        min_segment_frames,
        args.sam3_cut_on_count_change,
        args.sam3_cut_every_active_sample,
        scene_min_frames,
    )
    scene_cut_frames = [int(record["frame"]) for record in records if record.get("scene_cut")]
    if scene_cut_frames:
        scene_with_masks = [frame for frame in scene_cut_frames if frame in masks_by_start]
        scene_included = [frame for frame in scene_with_masks if frame in starts]
        scene_ignored = [frame for frame in scene_cut_frames if frame not in scene_included]
        print(
            f"[offline] MatAnyone2 scene cuts detected={scene_cut_frames} "
            f"included={scene_included} ignored={scene_ignored}"
        )
    masks_by_start = {start: masks_by_start[start] for start in starts if start in masks_by_start}
    print(f"[offline] MatAnyone2 segment plan starts={starts} active={sorted(masks_by_start)}")
    return masks_by_start, starts


def _precompute_sam3_segment_masks_subprocess(args, src: Path, source_fps: float, fps: float, target: int):
    import numpy as np

    tmp_dir = config.ROOT / "debug_output" / "_sam3_prepass"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result_path = tmp_dir / f"sam3_prepass_{int(time.time() * 1000)}_{id(args)}.npz"
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "tool", "offline_passthrough"]
    else:
        cmd = [sys.executable, str(Path(__file__).resolve())]
    cmd += [
        str(src),
        "--engine",
        "matanyone2_onnx",
        "--model",
        str(Path(args._matanyone2_model_dir).resolve()),
        "--matanyone2-size",
        str(args.matanyone2_size),
        "--matanyone2-batch",
        str(args.matanyone2_batch),
        "--sam3-model-dir",
        str(Path(args.sam3_model_dir).resolve()),
        "--sam3-prompt",
        str(args.sam3_prompt),
        "--sam3-score-threshold",
        str(args.sam3_score_threshold),
        "--sam3-min-area-ratio",
        str(args.sam3_min_area_ratio),
        "--sam3-max-area-ratio",
        str(args.sam3_max_area_ratio),
        "--sam3-top-k",
        str(args.sam3_top_k),
        "--sam3-scan",
        str(args.sam3_scan),
        "--sam3-scan-interval-sec",
        str(args.sam3_scan_interval_sec),
        "--sam3-active-min-area-ratio",
        str(args.sam3_active_min_area_ratio),
        "--sam3-provider",
        str(args.sam3_provider),
        "--sam3-decoder-provider",
        str(args.sam3_decoder_provider),
        "--sam3-cuda-memory-limit-mb",
        str(args.sam3_cuda_memory_limit_mb),
        "--sam3-decoder-cuda-memory-limit-mb",
        str(args.sam3_decoder_cuda_memory_limit_mb),
        "--sam3-min-vram-gb",
        str(args.sam3_min_vram_gb),
        "--sam3-release-interval",
        str(args.sam3_release_interval),
        "--sam3-low-memory" if args.sam3_low_memory else "--no-sam3-low-memory",
        "--sam3-low-memory-batch",
        str(args.sam3_low_memory_batch),
        "--sam3-log-vram" if args.sam3_log_vram else "--no-sam3-log-vram",
        "--matanyone2-segment-frames",
        str(args.matanyone2_segment_frames),
        "--matanyone2-min-segment-sec",
        str(args.matanyone2_min_segment_sec),
        "--sam3-gap-fill-frames",
        str(args.sam3_gap_fill_frames),
        "--frames",
        str(target),
        "--fps",
        str(fps),
        "--sam3-prepass-out",
        str(result_path),
    ]
    if args.sam3_debug_dir:
        cmd += ["--sam3-debug-dir", str(Path(args.sam3_debug_dir).resolve())]
    if not args.sam3_cut_on_count_change:
        cmd += ["--no-sam3-cut-on-count-change"]
    if args.sam3_cut_every_active_sample:
        cmd += ["--sam3-cut-every-active-sample"]
    print("[offline] SAM3 prepass subprocess=" + subprocess.list2cmdline(cmd))
    run_hidden_streaming(cmd, check=True, exit_label="offline-sam3")
    data = np.load(result_path, allow_pickle=False)
    starts = [int(x) for x in data["segment_starts"].tolist()]
    active_starts = [int(x) for x in data["active_starts"].tolist()]
    masks_by_start = {}
    for idx, start in enumerate(active_starts):
        masks_by_start[start] = [
            data[f"mask_{idx}_left"].astype(np.float32, copy=False),
            data[f"mask_{idx}_right"].astype(np.float32, copy=False),
        ]
    try:
        result_path.unlink(missing_ok=True)
    except Exception:
        pass
    clear_gpu_memory_pools()
    return masks_by_start, starts


def _write_sam3_prepass_result(path: Path, masks_by_start: dict[int, list], starts: list[int]) -> None:
    import numpy as np

    payload = {
        "segment_starts": np.asarray(starts, dtype=np.int64),
        "active_starts": np.asarray(sorted(masks_by_start), dtype=np.int64),
    }
    for idx, start in enumerate(sorted(masks_by_start)):
        payload[f"mask_{idx}_left"] = masks_by_start[start][0].astype(np.float32, copy=False)
        payload[f"mask_{idx}_right"] = masks_by_start[start][1].astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> int:
    _patch_tempdir()
    parser = argparse.ArgumentParser(description="Generate an offline passthrough MP4 from a source video.")
    parser.add_argument("video", help="video path, absolute or relative to PT_VIDEO_DIR")
    parser.add_argument("--out", default="", help="output mp4 path; default is source-stem_passthrough.mp4")
    parser.add_argument("--engine", default="rvm", choices=["rvm", "matanyone2_onnx"])
    parser.add_argument("--model", default="", help="model path; RVM defaults to models/rvm_mobilenetv3_fp32.onnx")
    parser.add_argument("--matanyone2-size", type=int, default=1024, choices=[512, 1024],
                        help="MatAnyone2 ONNX size to auto-select when --model is omitted")
    parser.add_argument("--matanyone2-batch", default="1", choices=["auto", "1", "2"],
                        help="MatAnyone2 ONNX batch to auto-select when --model is omitted; auto currently uses bs1")
    parser.add_argument("--mask", default="", help="first-frame object mask for MatAnyone2")
    parser.add_argument("--matanyone2-prepass", default="sam3", choices=["sam3", "yoloworld_efficientsam", "yolo26m_efficientsam", "yolo26m_birefnet"],
                        help="automatic first-mask prepass backend for MatAnyone2 when --mask is omitted")
    parser.add_argument("--ywes-model-dir", default=str(config.ROOT / "models" / "yoloworld_efficientsam"),
                        help="YOLO-World + EfficientSAM ONNX model directory")
    parser.add_argument("--ywes-txt-feats", default=str(config.ROOT / "models" / "person_txt_feats.npy"),
                        help="precomputed YOLO-World person txt_feats .npy")
    parser.add_argument("--ywes-provider", default="cuda", choices=["cuda", "cpu"],
                        help="execution provider for YOLO-World + EfficientSAM prepass")
    parser.add_argument("--ywes-yolo-model", default="yolov8l-worldv2.onnx",
                        help="YOLO-World ONNX filename under --ywes-model-dir")
    parser.add_argument("--ywes-sam-model", default="efficientsam_s.onnx",
                        help="EfficientSAM ONNX filename under --ywes-model-dir")
    parser.add_argument("--ywes-yolo-size", type=int, default=1280,
                        help="square YOLO-World letterbox input size")
    parser.add_argument("--ywes-score-threshold", type=float, default=0.03,
                        help="YOLO-World person score threshold")
    parser.add_argument("--ywes-nms-threshold", type=float, default=0.6,
                        help="YOLO-World NMS IoU threshold")
    parser.add_argument("--ywes-box-expand", type=float, default=0.08,
                        help="box expansion ratio before EfficientSAM")
    parser.add_argument("--ywes-top-k", type=int, default=1,
                        help="top detected persons per eye to segment")
    parser.add_argument("--ywes-scan", default="hybrid", choices=["keyframe", "interval", "hybrid"],
                        help="YOLO-World + EfficientSAM prepass sample strategy")
    parser.add_argument("--ywes-scan-interval-sec", type=float, default=1.0,
                        help="fallback/interval YOLO-World + EfficientSAM scan step in seconds")
    parser.add_argument("--ywes-active-min-area-ratio", type=float, default=0.001,
                        help="sample is active if either eye union mask area is at least this ratio")
    parser.add_argument("--ywes-gap-fill-frames", type=int, default=300,
                        help="fill short inactive YOLO-World samples between active samples by reusing a neighboring mask; 0 disables")
    parser.add_argument("--ywes-debug-dir", default="",
                        help="optional directory to save YOLO-World + EfficientSAM prepass debug files")
    parser.add_argument("--ywes-cut-on-count-change", action=argparse.BooleanOptionalAction, default=True,
                        help="start a new MatAnyone2 segment when selected person count changes")
    parser.add_argument("--ywes-cut-every-active-sample", action="store_true",
                        help="debug/quality mode: restart MatAnyone2 at every active YOLO-World sample")
    parser.add_argument("--ywes-fail-on-empty", action=argparse.BooleanOptionalAction, default=True,
                        help="fail instead of writing an all-background output when no person masks are found")
    parser.add_argument("--ywes-subprocess", action=argparse.BooleanOptionalAction, default=True,
                        help="run YOLO-World + EfficientSAM prepass in a child process so its CUDA context is released before MatAnyone2")
    parser.add_argument("--ywes-prepass-out", default="", help=argparse.SUPPRESS)
    parser.add_argument("--y26es-model-dir", default=str(config.ROOT / "models" / "yolo26m"),
                        help="YOLO26m ONNX model directory")
    parser.add_argument("--y26es-sam-model-dir", default=str(config.ROOT / "models" / "efficientsam"),
                        help="EfficientSAM ONNX model directory for YOLO26m prepass")
    parser.add_argument("--y26es-provider", default="cuda", choices=["cuda", "cpu"],
                        help="execution provider for YOLO26m + EfficientSAM prepass")
    parser.add_argument("--y26es-yolo-model", default="yolo26m_model.onnx",
                        help="YOLO26m ONNX filename under --y26es-model-dir; default fp32 because the fp16 export silently collapses on ORT CUDAExecutionProvider (person scores ~0.01-0.03)")
    parser.add_argument("--y26es-sam-model", default="efficientsam_s.onnx",
                        help="EfficientSAM ONNX filename under --y26es-sam-model-dir")
    parser.add_argument("--y26es-yolo-size", type=int, default=640,
                        help="YOLO26m letterbox input size; exported ONNX graph is fixed at 640")
    parser.add_argument("--y26es-score-threshold", type=float, default=0.35,
                        help="YOLO26m person sigmoid score threshold")
    parser.add_argument("--y26es-nms-threshold", type=float, default=0.6,
                        help="YOLO26m fallback NMS IoU threshold")
    parser.add_argument("--y26es-box-expand", type=float, default=0.08,
                        help="box expansion ratio before EfficientSAM")
    parser.add_argument("--y26es-top-k", type=int, default=0,
                        help="max persons to segment per eye; 0 = unlimited (every plausible candidate is paired, leftovers projected to the other eye). N>0 caps at N pairs. Default 0 handles 1..many people without a hard limit. EfficientSAM cost scales linearly with the number of selected persons per scan point, so set a small positive cap (e.g., 2-3) if you want to bound prepass time on crowded content.")
    parser.add_argument("--y26es-binarize-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="binarize EfficientSAM mask before writing MatAnyone2 prepass masks")
    parser.add_argument("--y26es-mask-erode-px", type=int, default=1,
                        help="morphological erosion iterations after mask binarization")
    parser.add_argument("--y26es-max-box-area", type=float, default=0.50,
                        help="reject YOLO26m boxes whose area exceeds this fraction of a per-eye frame; also gates the score fallback. Lower this if 'fill the frame' false-positives slip through.")
    parser.add_argument("--y26es-cross-eye-area-ratio", type=float, default=1.5,
                        help="when paired L/R boxes' area ratio exceeds this, project the higher-score side to the other eye to keep masks symmetric")
    parser.add_argument("--y26es-scan", default="hybrid", choices=["keyframe", "interval", "hybrid"],
                        help="YOLO26m + EfficientSAM prepass sample strategy")
    parser.add_argument("--y26es-scan-interval-sec", type=float, default=1.0,
                        help="fallback/interval YOLO26m + EfficientSAM scan step in seconds")
    parser.add_argument("--y26es-active-min-area-ratio", type=float, default=0.001,
                        help="sample is active if either eye union mask area is at least this ratio")
    parser.add_argument("--y26es-gap-fill-frames", type=int, default=600,
                        help="fill inactive YOLO26m samples between two active samples (or at clip start/end) by reusing a neighboring mask; counted in output-fps frames; 0 disables")
    parser.add_argument("--y26es-fill-boundaries", action=argparse.BooleanOptionalAction, default=True,
                        help="forward-fill from first active scan point to frame 0 and backward-fill from last active scan point to the tail (capped by --y26es-gap-fill-frames)")
    parser.add_argument("--y26es-scene-aware-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="when filling middle gaps that contain a scene cut, use the post-cut neighbor for frames after the cut; also blocks boundary fill across a scene-cut anchor")
    parser.add_argument("--y26es-debug-dir", default="",
                        help="optional directory to save YOLO26m + EfficientSAM prepass debug files")
    parser.add_argument("--y26es-cut-on-count-change", action=argparse.BooleanOptionalAction, default=True,
                        help="start a new MatAnyone2 segment when selected person count changes")
    parser.add_argument("--y26es-cut-every-active-sample", action="store_true",
                        help="debug/quality mode: restart MatAnyone2 at every active YOLO26m sample")
    parser.add_argument("--y26es-fail-on-empty", action=argparse.BooleanOptionalAction, default=True,
                        help="fail instead of writing an all-background output when no person masks are found")
    parser.add_argument("--y26es-subprocess", action=argparse.BooleanOptionalAction, default=True,
                        help="run YOLO26m + EfficientSAM prepass in a child process so its CUDA context is released before MatAnyone2")
    parser.add_argument("--y26es-prepass-out", default="", help=argparse.SUPPRESS)
    parser.add_argument("--y26br-model-dir", default=str(config.ROOT / "models" / "yolo26m"),
                        help="YOLO26m ONNX model directory")
    parser.add_argument("--y26br-birefnet-model-dir", default=str(config.ROOT / "models" / "BiRefNet"),
                        help="BiRefNet ONNX model directory for YOLO26m prepass")
    parser.add_argument("--y26br-provider", default="cuda", choices=["cuda", "cpu"],
                        help="execution provider for YOLO26m + BiRefNet prepass")
    parser.add_argument("--y26br-yolo-model", default="yolo26m_model.onnx",
                        help="YOLO26m ONNX filename under --y26br-model-dir; default fp32 because the fp16 export silently collapses on ORT CUDAExecutionProvider (person scores ~0.01-0.03)")
    parser.add_argument("--y26br-birefnet-model", default="model_fp16.onnx",
                        help="BiRefNet ONNX filename under --y26br-birefnet-model-dir")
    parser.add_argument("--y26br-yolo-size", type=int, default=640,
                        help="YOLO26m letterbox input size; exported ONNX graph is fixed at 640")
    parser.add_argument("--y26br-birefnet-input-size", type=int, default=1024,
                        help="BiRefNet square input size; the current ONNX graph is fixed at 1024")
    parser.add_argument("--y26br-score-threshold", type=float, default=0.35,
                        help="YOLO26m person sigmoid score threshold")
    parser.add_argument("--y26br-nms-threshold", type=float, default=0.6,
                        help="YOLO26m fallback NMS IoU threshold")
    parser.add_argument("--y26br-box-expand", type=float, default=0.08,
                        help="box expansion ratio before BiRefNet ROI segmentation")
    parser.add_argument("--y26br-top-k", type=int, default=0,
                        help="max persons to segment per eye; 0 = unlimited. BiRefNet cost scales per selected ROI, so set a small cap if you need bounded prepass time.")
    parser.add_argument("--y26br-binarize-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="binarize BiRefNet mask before writing MatAnyone2 prepass masks")
    parser.add_argument("--y26br-mask-erode-px", type=int, default=1,
                        help="morphological erosion iterations after mask binarization")
    parser.add_argument("--y26br-max-box-area", type=float, default=0.50,
                        help="reject YOLO26m boxes whose area exceeds this fraction of a per-eye frame; also gates the score fallback. Lower this if 'fill the frame' false-positives slip through.")
    parser.add_argument("--y26br-cross-eye-area-ratio", type=float, default=1.5,
                        help="when paired L/R boxes' area ratio exceeds this, project the higher-score side to the other eye to keep masks symmetric")
    parser.add_argument("--y26br-scan", default="hybrid", choices=["keyframe", "interval", "hybrid"],
                        help="YOLO26m + BiRefNet prepass sample strategy")
    parser.add_argument("--y26br-scan-interval-sec", type=float, default=1.0,
                        help="fallback/interval YOLO26m + BiRefNet scan step in seconds")
    parser.add_argument("--y26br-active-min-area-ratio", type=float, default=0.001,
                        help="sample is active if either eye union mask area is at least this ratio")
    parser.add_argument("--y26br-gap-fill-frames", type=int, default=600,
                        help="fill inactive YOLO26m samples between two active samples (or at clip start/end) by reusing a neighboring mask; counted in output-fps frames; 0 disables")
    parser.add_argument("--y26br-fill-boundaries", action=argparse.BooleanOptionalAction, default=True,
                        help="forward-fill from first active scan point to frame 0 and backward-fill from last active scan point to the tail (capped by --y26br-gap-fill-frames)")
    parser.add_argument("--y26br-scene-aware-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="when filling middle gaps that contain a scene cut, use the post-cut neighbor for frames after the cut; also blocks boundary fill across a scene-cut anchor")
    parser.add_argument("--y26br-debug-dir", default="",
                        help="optional directory to save YOLO26m + BiRefNet prepass debug files")
    parser.add_argument("--y26br-cut-on-count-change", action=argparse.BooleanOptionalAction, default=True,
                        help="start a new MatAnyone2 segment when selected person count changes")
    parser.add_argument("--y26br-cut-every-active-sample", action="store_true",
                        help="debug/quality mode: restart MatAnyone2 at every active YOLO26m sample")
    parser.add_argument("--y26br-fail-on-empty", action=argparse.BooleanOptionalAction, default=True,
                        help="fail instead of writing an all-background output when no person masks are found")
    parser.add_argument("--y26br-subprocess", action=argparse.BooleanOptionalAction, default=True,
                        help="run YOLO26m + BiRefNet prepass in a child process so its CUDA context is released before MatAnyone2")
    parser.add_argument("--y26br-prepass-out", default="", help=argparse.SUPPRESS)
    parser.add_argument("--sam3-model-dir", default=str(config.ROOT / "models" / "sam3_onnx"),
                        help="SAM3 ONNX model directory for MatAnyone2 first-frame text mask")
    parser.add_argument("--sam3-prompt", default="person", help="SAM3 text prompt for MatAnyone2 first-frame mask")
    parser.add_argument("--sam3-score-threshold", type=float, default=0.5,
                        help="SAM3 masks with score >= threshold are unioned")
    parser.add_argument("--sam3-min-area-ratio", type=float, default=0.0005,
                        help="drop SAM3 masks smaller than this frame-area ratio")
    parser.add_argument("--sam3-max-area-ratio", type=float, default=0.95,
                        help="drop SAM3 masks larger than this frame-area ratio")
    parser.add_argument("--sam3-top-k", type=int, default=0,
                        help="keep only top K selected SAM3 masks by score; 0 keeps all selected")
    parser.add_argument("--sam3-debug-dir", default="",
                        help="optional directory to save SAM3 prepass frames, masks, and metadata")
    parser.add_argument("--sam3-scan", default="hybrid", choices=["keyframe", "interval", "hybrid"],
                        help="SAM3 prepass sample strategy")
    parser.add_argument("--sam3-scan-interval-sec", type=float, default=1.0,
                        help="fallback/interval SAM3 scan step in seconds")
    parser.add_argument("--sam3-active-min-area-ratio", type=float, default=0.001,
                        help="sample is active if either eye union mask area is at least this ratio")
    parser.add_argument("--sam3-gap-fill-frames", type=int, default=300,
                        help="fill short inactive SAM3 samples between active samples by reusing a neighboring mask; 0 disables")
    parser.add_argument("--sam3-provider", default="cuda", choices=["cuda", "cpu"],
                        help="execution provider for SAM3 image encoder prepass")
    parser.add_argument("--sam3-decoder-provider", default="cuda", choices=["cuda", "cpu"],
                        help="execution provider for SAM3 decoder prepass")
    parser.add_argument("--sam3-cuda-memory-limit-mb", type=int, default=8192,
                        help="CUDA arena cap for SAM3 image encoder session/workspace cache; 0 leaves ORT uncapped")
    parser.add_argument("--sam3-decoder-cuda-memory-limit-mb", type=int, default=4096,
                        help="CUDA arena cap for SAM3 decoder session/workspace cache, not model weight size; 0 leaves ORT uncapped")
    parser.add_argument("--sam3-min-vram-gb", type=float, default=15.5,
                        help="minimum total GPU VRAM required for SAM3 CUDA prepass; set 0 to disable the check")
    parser.add_argument("--sam3-release-interval", type=int, default=0,
                        help="recreate SAM3 ONNX sessions every N sampled frames; 0 keeps sessions for the whole prepass")
    parser.add_argument("--sam3-low-memory", action=argparse.BooleanOptionalAction, default=False,
                        help="load/unload SAM3 sessions per call to reduce peak VRAM; slower and off by default")
    parser.add_argument("--sam3-low-memory-batch", type=int, default=1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--sam3-log-vram", action=argparse.BooleanOptionalAction, default=True,
                        help="print nvidia-smi memory after each SAM3 sample")
    parser.add_argument("--sam3-subprocess", action=argparse.BooleanOptionalAction, default=True,
                        help="run SAM3 prepass in a child process so its CUDA context is released before MatAnyone2")
    parser.add_argument("--sam3-prepass-out", default="", help=argparse.SUPPRESS)
    parser.add_argument("--sam3-cut-on-count-change", action=argparse.BooleanOptionalAction, default=True,
                        help="start a new MatAnyone2 segment when SAM3 selected person count changes")
    parser.add_argument("--sam3-cut-every-active-sample", action="store_true",
                        help="debug/quality mode: restart MatAnyone2 at every active SAM3 sample")
    parser.add_argument("--matanyone2-bootstrap-threshold", type=float, default=0.55,
                        help="SAM3/mask alpha threshold used to seed MatAnyone2 first-frame mask")
    parser.add_argument("--matanyone2-bootstrap-erode", type=int, default=1,
                        help="3x3 erosion iterations for MatAnyone2 bootstrap mask")
    parser.add_argument("--matanyone2-bootstrap-dilate", type=int, default=0,
                        help="3x3 dilation iterations for MatAnyone2 bootstrap mask")
    parser.add_argument("--matanyone2-bootstrap-soft", action="store_true",
                        help="keep soft alpha inside the conservative bootstrap mask")
    parser.add_argument("--matanyone2-segment-frames", type=int, default=config.MATANYONE2_SEGMENT_FRAMES,
                        help="reset MatAnyone2 memory and re-bootstrap with SAM3/mask every N frames; 0 disables")
    parser.add_argument("--matanyone2-min-segment-sec", type=float, default=3.0,
                        help="minimum seconds between SAM3-driven MatAnyone2 segment starts")
    parser.add_argument("--matanyone2-fused-update", action=argparse.BooleanOptionalAction, default=False,
                        help="use optional propagate_update graph when present; off by default because tests were slower")
    parser.add_argument("--matanyone2-step-update", action=argparse.BooleanOptionalAction, default=True,
                        help="use optional full step_update graph when present")
    parser.add_argument("--frames", type=int, default=0, help="limit frames for tests; 0 processes full video")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds; default 0")
    parser.add_argument("--duration", type=float, default=0.0, help="limit seconds for tests; 0 processes full video")
    parser.add_argument("--fps", type=float, default=0.0, help="max output CFR fps; <=0 keeps source CFR fps")
    parser.add_argument("--bitrate", default="source", help="NVENC target bitrate; default 'source' uses source video bitrate")
    parser.add_argument("--maxrate-multiplier", type=float, default=1.2, help="NVENC max bitrate multiplier over target bitrate")
    parser.add_argument("--bufsize-multiplier", type=float, default=2.0, help="NVENC VBV buffer multiplier over target bitrate")
    parser.add_argument("--rc", default="vbr", choices=["vbr", "vbr_hq", "cbr"], help="PyNv NVENC rate-control mode")
    parser.add_argument("--cq", type=int, default=-1, help="PyNv NVENC CQ value; set -1 to omit")
    parser.add_argument("--preset", default=config.PASSTHROUGH_PYNV_PRESET, help="PyNv NVENC preset, e.g. P1..P7")
    parser.add_argument("--codec", default="hevc", choices=["hevc", "h265", "h264"])
    parser.add_argument("--gop", type=int, default=int(config.PASSTHROUGH_GOP))
    parser.add_argument("--audio", default="copy", choices=["off", "copy", "aac"])
    parser.add_argument("--progress", type=int, default=30)
    parser.add_argument("--sync-profile", action="store_true",
                        help="synchronize CUDA after decode/matting/encode to locate async GPU wait time")
    parser.add_argument("--device-sync-profile", action="store_true",
                        help="diagnostic: use full CUDA device synchronize for --sync-profile")
    parser.add_argument("--no-warmup", action="store_true", help="disable matting warmup for quick offline tests")
    parser.add_argument("--input-size", type=int, default=2048, help="override PT_MATTING_INPUT_SIZE before loading Matter")
    parser.add_argument("--rvm-downsample-ratio", type=float, default=0.25,
                        help="override PT_RVM_DOWNSAMPLE_RATIO before loading RVM Matter")
    parser.add_argument("--alpha-stride", type=int, default=1, help="override PT_ALPHA_STRIDE before loading Matter")
    parser.add_argument("--sbs-batch", action=argparse.BooleanOptionalAction, default=False,
                        help="run left/right SBS eyes as a batch when the RVM model supports batch2")
    args = parser.parse_args()
    args._sam3_child = bool(args.sam3_prepass_out)
    args._ywes_child = bool(args.ywes_prepass_out)
    args._y26es_child = bool(args.y26es_prepass_out)
    args._y26br_child = bool(args.y26br_prepass_out)
    args._tool_name = "offline_passthrough"

    import PyNvVideoCodec as nvc
    if args.no_warmup:
        config.MATTING_WARMUP_RUNS = 0
    if args.input_size > 0:
        config.MATTING_INPUT_SIZE = int(args.input_size)
    if args.rvm_downsample_ratio > 0:
        config.RVM_DOWNSAMPLE_RATIO = float(args.rvm_downsample_ratio)
    if args.engine == "rvm":
        config.RVM_SCENE_RESET = True
        config.RVM_ALPHA_SMOOTH = True
    if args.alpha_stride > 0:
        config.ALPHA_STRIDE = int(args.alpha_stride)
    config.MATTING_SBS_BATCH = bool(args.sbs_batch)
    from pipeline.pynv_io import (
        FfmpegNv12SequentialDecoder,
        GpuNv12AppFrame,
        PyNvSimpleDecoder,
        PyNvThreadedSerialDecoder,
        cuda_device_summary,
    )

    src = _resolve_video(args.video)
    meta = probe_video_metadata(src)
    out = Path(args.out) if args.out else _default_out(
        src,
        int(getattr(meta.codec, "width", 0) or 0),
        int(getattr(meta.codec, "height", 0) or 0),
    )
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    final_out = out
    video_only_out = out
    if args.audio != "off":
        suffix = "".join(out.suffixes[-1:]) or ".mp4"
        video_only_out = out.with_name(f"{out.stem}._video_only{suffix}")
    print(
        f"[offline] input codec={meta.codec.codec_name} profile={meta.codec.profile} "
        f"pix_fmt={meta.codec.pix_fmt} bit_depth={meta.codec.bit_depth} "
        f"size={meta.codec.width}x{meta.codec.height}",
        flush=True,
    )
    print(f"[offline] cuda {cuda_device_summary(0)}", flush=True)
    backend_decision = select_backend(meta.timing, meta.codec, meta.color)
    decoder_mode = config.PASSTHROUGH_PYNV_DECODER
    if backend_decision.verdict == "ffmpeg_fallback":
        decoder_mode = "ffmpeg_fallback"
    elif backend_decision.verdict == "block":
        raise RuntimeError(f"unsupported offline source: {backend_decision.reason}")
    engine: OfflineMattingEngine | None = None
    if args.engine == "rvm":
        # Match the live path more closely: initialize Matter/ORT/TRT before
        # constructing the PyNv decoder, so decoder state is not created under
        # the heavy RVM session build.
        engine = _make_engine(args)
    try:
        if decoder_mode == "ffmpeg_fallback":
            dec = FfmpegNv12SequentialDecoder(src)
        elif decoder_mode == "threaded_serial":
            dec = PyNvThreadedSerialDecoder(
                src,
                bit_depth=int(meta.codec.bit_depth or 8),
                batch_size=config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
                buffer_size=config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
            )
        elif decoder_mode == "simple":
            dec = PyNvSimpleDecoder(src, bit_depth=int(meta.codec.bit_depth or 8))
        else:
            raise RuntimeError(f"unknown PT_PASSTHROUGH_PYNV_DECODER={decoder_mode!r}")
    except Exception as exc:
        text = str(exc)
        if "MBCount not supported" in text:
            print(
                "[offline] ERROR: PyNvVideoCodec/NVDEC rejected this video on the selected CUDA device. "
                "This usually means gpu_id=0 is not the expected RTX GPU, the NVIDIA driver/video stack is "
                "reporting the wrong decode capability, or the source codec/profile exceeds NVDEC support.",
                flush=True,
            )
            print(f"[offline] ERROR detail: {text}", flush=True)
            return 3
        raise
    info = dec.info
    print(
        f"[offline] decoder={decoder_mode} "
        f"batch={config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE} "
        f"buffer={config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE} "
        f"backend_reason={backend_decision.reason}",
        flush=True,
    )
    timing = probe_timing_metadata(src)
    source_fps = float(timing.source_fps or info.fps or 30.0)
    fps = float(timing.effective_fps(float(args.fps or 0.0)))
    start_out = int(round(max(0.0, float(args.start or 0.0)) * fps))
    if args.frames > 0:
        target = int(args.frames)
    else:
        total_seconds = float(timing.duration or info.duration or 0.0)
        seconds = max(0.0, total_seconds - max(0.0, float(args.start or 0.0))) if total_seconds > 0 else 0.0
        if args.duration > 0:
            seconds = min(seconds, float(args.duration)) if seconds > 0 else float(args.duration)
        target = int(max(1, round(seconds * fps))) if seconds > 0 else len(dec)
    max_target = int((len(dec) - 1) * fps / source_fps) + 1 if source_fps > 0 else len(dec)
    target = min(target, max(1, max_target - start_out))
    bitrate_kwargs, target_bps, max_bps, buf_bps = _encoder_bitrate_kwargs(args, src)
    output_duration = target / fps if fps > 0 else 0.0
    audio_sidecar: Path | None = None
    if args.audio != "off":
        ta0 = time.perf_counter()
        audio_extract_cmd, audio_extract_proc, audio_sidecar = _extract_audio_sidecar(
            src,
            out,
            args.audio,
            max(0.0, float(args.start or 0.0)),
            output_duration,
        )
        print("[offline] audio_extract=" + subprocess.list2cmdline(audio_extract_cmd))
        print(f"audio_extract_rc = {audio_extract_proc.returncode}")
        print(f"audio_extract_elapsed = {time.perf_counter() - ta0:.3f} s")
        audio_extract_stderr = (audio_extract_proc.stdout or "") + (audio_extract_proc.stderr or "")
        if audio_extract_stderr.strip():
            print("[audio extract stderr]")
            print(audio_extract_stderr.strip()[-2000:])
        if audio_extract_proc.returncode != 0:
            _cleanup_audio_sidecar(audio_sidecar)
            dec.stop()
            return audio_extract_proc.returncode

    if args.engine == "matanyone2_onnx":
        model_dir = _resolve_matanyone2_model_dir(args, info.width, info.height)
        args._matanyone2_model_dir = str(model_dir)
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        args._matanyone2_in_h = int(manifest.get("height") or 512)
        args._matanyone2_in_w = int(manifest.get("width") or 512)
        args._matanyone2_batch_size = int(manifest.get("batch_size") or 1)
    _require_sam3_vram(args)
    if args.engine == "matanyone2_onnx" and not args.mask and args.matanyone2_prepass == "yolo26m_birefnet":
        from offline.yolo26m_birefnet import precompute_segment_masks as _precompute_y26br_segment_masks
        from offline.yolo26m_birefnet import write_prepass_result as _write_y26br_prepass_result

        sam3_masks, segment_starts = _precompute_y26br_segment_masks(args, src, dec, source_fps, fps, target, cfr_source_index)
    elif args.engine == "matanyone2_onnx" and not args.mask and args.matanyone2_prepass == "yolo26m_efficientsam":
        from offline.yolo26m_efficientsam import precompute_segment_masks as _precompute_y26es_segment_masks
        from offline.yolo26m_efficientsam import write_prepass_result as _write_y26es_prepass_result

        sam3_masks, segment_starts = _precompute_y26es_segment_masks(args, src, dec, source_fps, fps, target, cfr_source_index)
    elif args.engine == "matanyone2_onnx" and not args.mask and args.matanyone2_prepass == "yoloworld_efficientsam":
        from offline.yoloworld_efficientsam import precompute_segment_masks as _precompute_ywes_segment_masks
        from offline.yoloworld_efficientsam import write_prepass_result as _write_ywes_prepass_result

        sam3_masks, segment_starts = _precompute_ywes_segment_masks(args, src, dec, source_fps, fps, target, cfr_source_index)
    else:
        sam3_masks, segment_starts = _precompute_sam3_segment_masks(args, src, dec, source_fps, fps, target)
    if args.y26br_prepass_out:
        _write_y26br_prepass_result(Path(args.y26br_prepass_out).resolve(), sam3_masks, segment_starts)
        _cleanup_audio_sidecar(audio_sidecar)
        dec.stop()
        return 0
    if args.y26es_prepass_out:
        _write_y26es_prepass_result(Path(args.y26es_prepass_out).resolve(), sam3_masks, segment_starts)
        _cleanup_audio_sidecar(audio_sidecar)
        dec.stop()
        return 0
    if args.ywes_prepass_out:
        _write_ywes_prepass_result(Path(args.ywes_prepass_out).resolve(), sam3_masks, segment_starts)
        _cleanup_audio_sidecar(audio_sidecar)
        dec.stop()
        return 0
    if args.sam3_prepass_out:
        _write_sam3_prepass_result(Path(args.sam3_prepass_out).resolve(), sam3_masks, segment_starts)
        _cleanup_audio_sidecar(audio_sidecar)
        dec.stop()
        return 0
    if engine is None:
        engine = _make_engine(args)
    enc_w, enc_h = info.width, info.height
    if isinstance(engine, SharedMatAnyone2OnnxEngine):
        engine.set_segment_plan(segment_starts)
        for segment_start, masks in sam3_masks.items():
            engine.set_segment_masks(segment_start, masks)
    from pipeline import matting as matting_module

    enc = nvc.CreateEncoder(
        enc_w,
        enc_h,
        "NV12",
        False,
        codec=args.codec,
        fps=f"{fps:.6f}",
        gop=str(args.gop),
        bf="0",
        **bitrate_kwargs,
    )
    cmd, mux = _open_muxer(video_only_out, fps, src, args.codec)
    assert mux.stdin is not None
    print(
        f"[offline] src={src} out={final_out} engine={args.engine} {info.width}x{info.height} "
        f"source_fps={source_fps:.6f} output_fps={fps:.6f} target={target} audio={args.audio} "
        f"bitrate={target_bps} maxbitrate={max_bps} vbvbufsize={buf_bps} rc={args.rc} cq={args.cq} preset={args.preset}"
    )
    print("[offline] mux=" + subprocess.list2cmdline(cmd))

    t_dec: list[float] = []
    t_mat: list[float] = []
    t_sync: list[float] = []
    t_enc: list[float] = []
    t_mux: list[float] = []
    t_sync_after_dec: list[float] = []
    t_sync_after_mat: list[float] = []
    t_sync_after_enc: list[float] = []
    last_src_idx = -1
    source_index_rewinds = 0
    bytes_written = 0
    started = time.perf_counter()
    used0, total0 = _mem_stats()
    try:
        for i in range(target):
            src_idx = min(len(dec) - 1, cfr_source_index(start_out + i, source_fps, fps))
            if src_idx <= last_src_idx:
                source_index_rewinds += 1
                src_idx = min(len(dec) - 1, last_src_idx + 1)
            last_src_idx = src_idx
            td0 = time.perf_counter()
            frame = dec.frame_at(src_idx)
            td1 = time.perf_counter()
            if args.sync_profile:
                t_sync_after_dec.append(_cuda_sync_ms(args.device_sync_profile))
            if isinstance(engine, SharedMatAnyone2OnnxEngine):
                engine.set_source_frame_index(src_idx)
            if isinstance(engine, SharedMatAnyone2OnnxEngine) and not engine.is_active_frame():
                out_nv12, _ = engine.composite_green_nv12(frame)
            else:
                out_nv12, _ = engine.composite_nv12(frame)
            tm1 = time.perf_counter()
            cuda_stream = getattr(matting_module, "_CUDA_STREAM", None)
            if cuda_stream is not None:
                cuda_stream.synchronize()
            ts1 = time.perf_counter()
            if args.sync_profile:
                t_sync_after_mat.append(_cuda_sync_ms(args.device_sync_profile))
            app_frame = GpuNv12AppFrame(out_nv12, enc_w, enc_h)
            flags = 0
            if i == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            te0 = time.perf_counter()
            bitstream = enc.Encode(app_frame, flags) if flags else enc.Encode(app_frame)
            te1 = time.perf_counter()
            if args.sync_profile:
                t_sync_after_enc.append(_cuda_sync_ms(args.device_sync_profile))
            t_dec.append((td1 - td0) * 1000)
            t_mat.append((tm1 - td1) * 1000)
            t_sync.append((ts1 - tm1) * 1000)
            t_enc.append((te1 - te0) * 1000)
            if bitstream:
                tw0 = time.perf_counter()
                mux.stdin.write(bitstream)
                tw1 = time.perf_counter()
                t_mux.append((tw1 - tw0) * 1000)
                bytes_written += len(bitstream)
            if args.progress > 0 and (i + 1) % args.progress == 0:
                elapsed = time.perf_counter() - started
                dec_avg = statistics.fmean(t_dec)
                mat_avg = statistics.fmean(t_mat)
                sync_avg = statistics.fmean(t_sync)
                print(
                    f"[offline] {i + 1:7d}/{target} fps={(i + 1) / elapsed:7.2f} "
                    f"dec={dec_avg:6.2f}ms mat={mat_avg:6.2f}ms sync={sync_avg:5.2f}ms "
                    f"dec+mat+sync={dec_avg + mat_avg + sync_avg:6.2f}ms "
                    f"enc={statistics.fmean(t_enc):5.2f}ms mux={statistics.fmean(t_mux) if t_mux else 0:5.2f}ms"
                )
        tail = enc.EndEncode()
        if tail:
            mux.stdin.write(tail)
            bytes_written += len(tail)
    finally:
        release_pending = getattr(engine, "release_pending_outputs", None) if "engine" in locals() else None
        if callable(release_pending):
            release_pending()
        try:
            mux.stdin.close()
        except Exception:
            pass
        dec.stop()

    stderr = mux.stderr.read().decode("utf-8", "replace") if mux.stderr else ""
    rc = mux.wait(timeout=120)
    elapsed = time.perf_counter() - started
    used1, total1 = _mem_stats()
    print("---- summary ----")
    print(f"rc = {rc}")
    print(f"frames = {target}")
    print(f"video_bytes = {bytes_written}")
    print(f"elapsed = {elapsed:.3f} s")
    print(f"throughput = {target / elapsed:.2f} fps")
    print(f"source_index_rewinds = {source_index_rewinds}")
    for line in _summary("decode", t_dec): print(line)
    for line in _summary("matting", t_mat): print(line)
    for line in _summary("sync", t_sync): print(line)
    if t_dec and t_mat:
        print(f"decode_matting_avg = {statistics.fmean([d + m for d, m in zip(t_dec, t_mat)]):.3f} ms")
    if t_dec and t_mat and t_sync:
        print(f"decode_matting_sync_avg = {statistics.fmean([d + m + s for d, m, s in zip(t_dec, t_mat, t_sync)]):.3f} ms")
    if hasattr(engine, "profile_lines"):
        for line in engine.profile_lines():
            print(line)
    for line in _summary("encode", t_enc): print(line)
    for line in _summary("mux_write", t_mux): print(line)
    if args.sync_profile:
        for line in _summary("sync_after_decode", t_sync_after_dec): print(line)
        for line in _summary("sync_after_matting", t_sync_after_mat): print(line)
        for line in _summary("sync_after_encode", t_sync_after_enc): print(line)
    print(f"mem_start = {used0:.1f}/{total0:.1f} MB")
    print(f"mem_end = {used1:.1f}/{total1:.1f} MB")
    if stderr.strip():
        print("[ffmpeg stderr]")
        print(stderr.strip()[-2000:])
    audio_mux_proc = None
    if rc == 0 and args.audio != "off" and audio_sidecar is not None:
        ta0 = time.perf_counter()
        audio_mux_cmd, audio_mux_proc = _mux_audio_sidecar_after(video_only_out, final_out, audio_sidecar, src, output_duration)
        print("[offline] audio_mux=" + subprocess.list2cmdline(audio_mux_cmd))
        print(f"audio_mux_rc = {audio_mux_proc.returncode}")
        print(f"audio_mux_elapsed = {time.perf_counter() - ta0:.3f} s")
        audio_mux_stderr = (audio_mux_proc.stdout or "") + (audio_mux_proc.stderr or "")
        if audio_mux_stderr.strip():
            print("[audio mux stderr]")
            print(audio_mux_stderr.strip()[-2000:])
        if audio_mux_proc.returncode != 0:
            rc = audio_mux_proc.returncode
        else:
            try:
                video_only_out.unlink(missing_ok=True)
            except Exception:
                pass
            _cleanup_audio_sidecar(audio_sidecar)
    _cleanup_audio_sidecar(audio_sidecar)
    print("[ffprobe]")
    print(_ffprobe(final_out if rc == 0 else video_only_out))
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
