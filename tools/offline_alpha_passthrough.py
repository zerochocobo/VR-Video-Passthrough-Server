"""Offline DeoVR alpha-packed passthrough test generator.

This experimental tool is copied from offline_passthrough.py but replaces the
final green-screen composite step with DeoVR-style alpha packing for local
tests. The main offline passthrough flow is intentionally untouched.
"""
from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402

import config  # noqa: E402
from utils.bitrate_estimator import effective_default_bitrate, parse_bitrate, source_video_bitrate  # noqa: E402
from utils.gpu_runtime_cache import configure_gpu_runtime_cache  # noqa: E402
from utils.subprocess_hidden import hidden_subprocess_kwargs, run_hidden_streaming  # noqa: E402
from utils.trt_manifest import (  # noqa: E402
    MATANYONE2_TRT_ONNX_NAME,
    TRT_MODEL_MATANYONE2,
    TRT_PROVIDER_CHAIN,
    cache_dir_for_model,
    cache_status,
    matanyone2_model_dir,
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

GPU_CACHE_ENV = configure_gpu_runtime_cache()


def _configure_offline_rvm_tensorrt(model_path: Path) -> bool:
    providers = [p.strip() for p in config.ONNX_PROVIDERS if p.strip()]
    wants_trt = "TensorrtExecutionProvider" in providers
    is_mobile = model_path.resolve() == original_rvm_model_path().resolve()
    if wants_trt and is_mobile:
        try:
            if cache_status() == "ready":
                config.ONNX_PROVIDERS = [p.strip() for p in TRT_PROVIDER_CHAIN.split(",") if p.strip()]
                print(f"[offline-alpha] RVM TensorRT enabled providers={config.ONNX_PROVIDERS}")
                return True
        except Exception as exc:
            print(f"[offline-alpha] RVM TensorRT unavailable, using CUDA providers ({type(exc).__name__}: {exc})")
    if wants_trt:
        config.ONNX_PROVIDERS = [p for p in providers if p != "TensorrtExecutionProvider"] or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"[offline-alpha] RVM TensorRT disabled for model={model_path.name}; providers={config.ONNX_PROVIDERS}")
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
        if model_dir.resolve() != matanyone2_model_dir().resolve():
            print(f"[offline-alpha] MatAnyone2 TensorRT disabled for custom model dir={model_dir}", flush=True)
            return _available_onnx_providers()
        if cache_status(model_key=TRT_MODEL_MATANYONE2) != "ready":
            print("[offline-alpha] MatAnyone2 TensorRT cache is not ready; using CUDA providers", flush=True)
            return _available_onnx_providers()
        from pipeline.matting import _filter_available_providers, _provider_config

        cache_dir = cache_dir_for_model(TRT_MODEL_MATANYONE2)
        config.ONNX_TRT_ENGINE_CACHE_PATH = cache_dir
        os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(cache_dir)
        providers = _filter_available_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        if "TensorrtExecutionProvider" not in providers:
            print("[offline-alpha] MatAnyone2 TensorRT provider unavailable; using CUDA providers", flush=True)
            return _available_onnx_providers()
        print(f"[offline-alpha] MatAnyone2 TensorRT enabled model={name} cache={cache_dir}", flush=True)
        return _provider_config(providers)
    except Exception as exc:
        print(f"[offline-alpha] MatAnyone2 TensorRT unavailable, using CUDA providers ({type(exc).__name__}: {exc})", flush=True)
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
    if getattr(args, "model", ""):
        return Path(args.model).resolve()
    size = int(getattr(args, "matanyone2_size", 512) or 512)
    batch_arg = str(getattr(args, "matanyone2_batch", "1") or "1").lower()
    if batch_arg == "auto":
        is_sbs = width > 0 and height > 0 and width >= 2 * height
        # 1024 batch2 is memory-bandwidth/workspace heavy in ORT CUDA on this
        # pipeline and benchmarks slower than two batch1 eye passes. Keep bs2
        # as the default only for the 512 SBS model unless explicitly forced.
        batch = 2 if size <= 512 and is_sbs else 1
    else:
        batch = int(batch_arg)
    model_dir = config.ROOT / "models" / f"matanyone2_onnx_{size}_bs{batch}"
    if not model_dir.exists():
        fallback = config.ROOT / "models" / "matanyone2_onnx"
        if fallback.exists():
            print(f"[offline] warning: {model_dir} not found, falling back to {fallback}")
            return fallback.resolve()
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


def _default_out(src: Path) -> Path:
    return src.with_name(f"{offline_passthrough_stem(src.stem, 'alpha_passthrough.mp4')}.mp4")


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


def _mux_audio_after(
    video_only: Path,
    out: Path,
    src: Path,
    audio: str,
    start_sec: float = 0.0,
    duration: float = 0.0,
) -> tuple[list[str], subprocess.CompletedProcess]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    audio_codec = "copy" if audio == "copy" else "aac"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(video_only),
        *(
            ["-ss", f"{start_sec:.6f}"]
            if start_sec > 0
            else []
        ),
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        *(
            ["-t", f"{duration:.6f}"]
            if duration > 0
            else []
        ),
        "-map_metadata",
        "1",
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


def _encoder_bitrate_kwargs(
    args: argparse.Namespace,
    src: Path,
    src_w: int = 0,
    src_h: int = 0,
    out_w: int = 0,
    out_h: int = 0,
) -> tuple[dict[str, str], int, int, int]:
    raw_bitrate = str(args.bitrate or "source").strip().lower()
    target_text = _parse_bitrate(args.bitrate, src)
    target_bps = parse_bitrate(target_text)
    if raw_bitrate in {"", "source", "auto", "same"} and src_w > 0 and src_h > 0 and out_w > 0 and out_h > 0:
        src_pixels = max(1, int(src_w) * int(src_h))
        out_pixels = max(1, int(out_w) * int(out_h))
        if out_pixels > src_pixels:
            target_bps = int(target_bps * (out_pixels / src_pixels))
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


class OfflineMattingEngine:
    def composite_nv12(self, frame):
        raise NotImplementedError


def _alpha_packer_from_args(matter, args) -> "AlphaPacker":
    return AlphaPacker(
        matter,
        scale=args.alpha_pack_scale,
        blocks_x=args.alpha_pack_blocks_x,
        blocks_y=args.alpha_pack_blocks_y,
        radius_scale=args.fisheye_radius_scale,
    )


from pipeline.alpha_packer import AlphaPacker, alpha_2d_disparity_px, alpha_output_size


class RvmOfflineEngine(OfflineMattingEngine):
    def __init__(self, model: Path | None = None, args=None):
        if model is not None:
            config.MODEL_PATH = model
            _configure_offline_rvm_tensorrt(model)
        from pipeline.matting import Matter

        self.matter = Matter()
        print(f"[offline-alpha] RVM runtime provider_diag={self.matter._rvm_provider_diag()}", file=sys.stderr)
        self.matter.reset_state()
        self.packer = _alpha_packer_from_args(self.matter, args) if args is not None else AlphaPacker(self.matter)
        self.profile: dict[str, list[float]] = defaultdict(list)

    def composite_nv12(self, frame):
        t_up0 = time.perf_counter()
        h, w = int(frame.height), int(frame.width)
        from pipeline.pynv_io import GpuP016Frame

        if isinstance(frame, GpuP016Frame):
            self.matter.upload_p016_planes_as_nv12_gpu(
                frame.y.as_cupy(),
                frame.uv.as_cupy(),
                h,
                w,
                shift_bits=int(config.PASSTHROUGH_PYNV_10BIT_SHIFT),
            )
        else:
            self.matter.upload_nv12_planes_gpu(frame.y.as_cupy(), frame.uv.as_cupy(), h, w)
        t_up1 = time.perf_counter()
        a_small, timing, _ = self.matter._alpha_low_res_gpu_temporal(h, w, use_nv12=True)
        t0 = time.perf_counter()
        out = self.packer.pack_uploaded(a_small, h, w)
        t1 = time.perf_counter()
        self.profile["preprocess"].append(timing.preprocess_ms)
        self.profile["ort"].append(timing.ort_ms)
        self.profile["upload_nv12"].append((t_up1 - t_up0) * 1000)
        self.profile["alpha_pack"].append((t1 - t0) * 1000)
        return out, timing

    def profile_lines(self) -> list[str]:
        lines = []
        for key in ("upload_nv12", "preprocess", "ort", "alpha_pack"):
            values = self.profile.get(key)
            if values:
                lines.append(f"rvm_{key}_avg = {statistics.fmean(values):.3f} ms n={len(values)}")
        return lines


class MatAnyone2OnnxEngine(OfflineMattingEngine):

    class _EyeState:
        def __init__(self):
            self.sensory = None
            self.memory_key = None
            self.memory_shrinkage = None
            self.memory_msk_value = None
            self.obj_memory = None
            self.last_pix_feat = None
            self.last_mask = None
            self.last_msk_value = None
            self.initialized = False

        def reset(self):
            self.sensory = None
            self.memory_key = None
            self.memory_shrinkage = None
            self.memory_msk_value = None
            self.obj_memory = None
            self.last_pix_feat = None
            self.last_mask = None
            self.last_msk_value = None
            self.initialized = False

    def __init__(
        self,
        model_dir: Path,
        mask: Path | None,
        sam3_dir: Path,
        sam3_prompt: str = "person",
        bootstrap_threshold: float = 0.55,
        bootstrap_erode: int = 1,
        bootstrap_dilate: int = 0,
        bootstrap_soft: bool = False,
        segment_frames: int = 300,
        use_fused_update: bool = False,
        use_step_update: bool = True,
        args=None,
    ):
        import cv2
        import numpy as np
        import onnxruntime as ort

        from pipeline.matting import Matter

        self.cv2 = cv2
        self.np = np
        self.ort = ort
        self.model_dir = model_dir
        self.mask_path = mask
        self.sam3_dir = sam3_dir
        self.sam3_prompt = sam3_prompt
        self.bootstrap_threshold = min(1.0, max(0.0, float(bootstrap_threshold)))
        self.bootstrap_erode = max(0, int(bootstrap_erode))
        self.bootstrap_dilate = max(0, int(bootstrap_dilate))
        self.bootstrap_soft = bool(bootstrap_soft)
        self.segment_frames = max(0, int(segment_frames))
        self.alpha_stride = max(1, int(config.ALPHA_STRIDE))
        self._frame_index = 0
        self._source_frame_index = -1
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"MatAnyone2 manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        self.in_h = int(manifest.get("height") or 512)
        self.in_w = int(manifest.get("width") or 512)
        self.batch_size = int(manifest.get("batch_size") or 1)
        if int(manifest.get("objects") or 1) != 1:
            raise RuntimeError("MatAnyone2 offline engine currently supports one object only")
        provider_report: dict[str, list[str]] = {}
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def sess(name: str):
            path = model_dir / name
            if not path.exists():
                raise RuntimeError(f"MatAnyone2 ONNX file not found: {path}")
            session = ort.InferenceSession(str(path), sess_options=sess_opts, providers=_matanyone2_session_providers(name, model_dir))
            provider_report[name] = list(session.get_providers())
            return session

        self.image_key = sess("matanyone2_image_key.onnx")
        self.mask_memory = sess("matanyone2_mask_memory.onnx")
        self.first_refine = sess("matanyone2_first_frame_refine.onnx")
        self.propagate = sess("matanyone2_propagate.onnx")
        self.propagate_update = None
        if use_fused_update and (model_dir / "matanyone2_propagate_update.onnx").exists():
            self.propagate_update = sess("matanyone2_propagate_update.onnx")
        self.step_update = None
        if use_step_update and (model_dir / "matanyone2_step_update.onnx").exists():
            self.step_update = sess("matanyone2_step_update.onnx")
        self._first_refine_inputs = {i.name for i in self.first_refine.get_inputs()}
        self._propagate_inputs = {i.name for i in self.propagate.get_inputs()}
        self._propagate_update_inputs = {i.name for i in self.propagate_update.get_inputs()} if self.propagate_update else set()
        self._step_update_inputs = {i.name for i in self.step_update.get_inputs()} if self.step_update else set()
        self.tensor_dtype = self.np.float16 if self.image_key.get_inputs()[0].type == "tensor(float16)" else self.np.float32
        image_batch = self.image_key.get_inputs()[0].shape[0]
        self.batch2_enabled = self.batch_size >= 2 or image_batch == 2
        sensory_meta = next(i for i in self.mask_memory.get_inputs() if i.name == "sensory")
        sensory_shape = [int(v) for v in sensory_meta.shape]
        self.sensory_shape = tuple(sensory_shape)
        self.sensory_single_shape = tuple([1, *sensory_shape[1:]])
        # SAM3 provides the bootstrap mask. Matter is used here only for its
        # NV12 GPU upload/preprocess/composite helpers, not for RVM inference.
        self.matter = Matter(config.ROOT / "models" / "rvm_resnet50_fp32.onnx", load_model=False)
        self.matter.reset_state()
        self.packer = _alpha_packer_from_args(self.matter, args) if args is not None else AlphaPacker(self.matter)
        self.eyes = [self._EyeState(), self._EyeState()]
        self._mask_cache: list[np.ndarray] | None = None
        self.segment_masks: dict[int, list[np.ndarray]] = {}
        self._active_segment_start = -1
        self._cached_alpha_sbs = None
        self.profile = defaultdict(list)
        print(
            f"[offline] MatAnyone2 ONNX loaded dir={model_dir} input={self.in_w}x{self.in_h} "
            f"sbs=per-eye bootstrap={'mask' if mask else 'sam3'} sam3_prompt={sam3_prompt!r} "
            f"bootstrap_erode={self.bootstrap_erode} bootstrap_dilate={self.bootstrap_dilate} "
            f"bootstrap_soft={self.bootstrap_soft} segment_frames={self.segment_frames} alpha_stride={self.alpha_stride} "
            f"batch2={self.batch2_enabled} dtype={self.tensor_dtype.__name__} "
            f"fused_update={self.propagate_update is not None} step_update={self.step_update is not None} "
            f"providers={provider_report}"
        )

    @staticmethod
    def _as_numpy(x):
        try:
            import cupy as cp

            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)
        except Exception:
            pass
        return x

    def _tensor(self, x):
        return x.astype(self.tensor_dtype, copy=False)

    def _image_key(self, image: "np.ndarray") -> dict[str, "np.ndarray"]:
        names = ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"]
        t0 = time.perf_counter()
        outs = self.image_key.run(names, {"image": image})
        self.profile["image_key"].append((time.perf_counter() - t0) * 1000)
        return dict(zip(names, outs))

    def _mask_memory(self, image, mask, sensory, pix_feat):
        t0 = time.perf_counter()
        msk_value, new_sensory, obj_memory = self.mask_memory.run(
            ["msk_value", "new_sensory", "obj_memory"],
            {
                "image": image,
                "mask": mask,
                "sensory": sensory,
                "pix_feat": pix_feat,
            },
        )
        self.profile["mask_memory"].append((time.perf_counter() - t0) * 1000)
        return msk_value, new_sensory, obj_memory

    def _first_frame_refine(self, feats, last_msk_value, obj_memory, sensory, last_mask):
        feed = {
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "last_msk_value": last_msk_value,
            "obj_memory": obj_memory,
            "sensory": sensory,
            "last_mask": last_mask,
        }
        t0 = time.perf_counter()
        prob, new_sensory, _logits = self.first_refine.run(
            ["prob", "new_sensory", "logits"],
            {k: v for k, v in feed.items() if k in self._first_refine_inputs},
        )
        self.profile["first_refine"].append((time.perf_counter() - t0) * 1000)
        return prob, new_sensory

    def _propagate(self, feats, state):
        assert state.memory_key is not None
        assert state.memory_shrinkage is not None
        assert state.memory_msk_value is not None
        assert state.obj_memory is not None
        assert state.sensory is not None
        assert state.last_mask is not None
        assert state.last_pix_feat is not None
        assert state.last_msk_value is not None
        feed = {
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "key": feats["key"],
            "selection": feats["selection"],
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": state.last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": state.last_mask,
            "last_msk_value": state.last_msk_value,
        }
        t0 = time.perf_counter()
        prob, new_sensory, _logits, _uncert_prob = self.propagate.run(
            ["prob", "new_sensory", "logits", "uncert_prob"],
            {k: v for k, v in feed.items() if k in self._propagate_inputs},
        )
        self.profile["propagate"].append((time.perf_counter() - t0) * 1000)
        return prob, new_sensory

    def _propagate_update(self, image, feats, state):
        if self.propagate_update is None:
            prob, sensory = self._propagate(feats, state)
            alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.np.float32, copy=False)
            msk_value, sensory, obj_memory = self._mask_memory(image, alpha, sensory, feats["pix_feat"])
            return prob, sensory, msk_value, obj_memory
        feed = {
            "image": image,
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "key": feats["key"],
            "selection": feats["selection"],
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": state.last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": state.last_mask,
            "last_msk_value": state.last_msk_value,
        }
        t0 = time.perf_counter()
        prob, sensory, msk_value, obj_memory, _logits, _uncert_prob = self.propagate_update.run(
            ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "logits", "uncert_prob"],
            {k: v for k, v in feed.items() if k in self._propagate_update_inputs},
        )
        self.profile["propagate_update"].append((time.perf_counter() - t0) * 1000)
        return prob, sensory, msk_value, obj_memory

    def _step_update(self, image, state):
        if self.step_update is None:
            feats = self._image_key(image)
            prob, sensory, msk_value, obj_memory = self._propagate_update(image, feats, state)
            return prob, sensory, msk_value, obj_memory, feats["pix_feat"]
        feed = {
            "image": image,
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": state.last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": state.last_mask,
            "last_msk_value": state.last_msk_value,
        }
        t0 = time.perf_counter()
        prob, sensory, msk_value, obj_memory, pix_feat, _logits, _uncert_prob = self.step_update.run(
            ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "pix_feat", "logits", "uncert_prob"],
            {k: v for k, v in feed.items() if k in self._step_update_inputs},
        )
        self.profile["step_update"].append((time.perf_counter() - t0) * 1000)
        return prob, sensory, msk_value, obj_memory, pix_feat

    def _load_masks(self) -> "list[np.ndarray] | None":
        if self.mask_path is None:
            return None
        if self._mask_cache is not None:
            return self._mask_cache
        mask = self.cv2.imread(str(self.mask_path), self.cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read MatAnyone2 mask: {self.mask_path}")
        mh, mw = mask.shape[:2]
        if mw >= 2 * mh:
            half = mw // 2
            masks = [mask[:, :half], mask[:, half:half * 2]]
        else:
            masks = [mask, mask]
        self._mask_cache = []
        for eye_mask in masks:
            resized = self.cv2.resize(eye_mask, (self.in_w, self.in_h), interpolation=self.cv2.INTER_NEAREST)
            self._mask_cache.append((resized.astype(self.np.float32) / 255.0)[None, None, :, :])
        return self._mask_cache

    def _reset_segment(self):
        for eye in self.eyes:
            eye.reset()
        self._active_segment_start = -1
        self._cached_alpha_sbs = None

    def set_segment_masks(self, segment_start: int, masks: "list[np.ndarray]"):
        self.segment_masks[int(segment_start)] = masks

    def set_segment_plan(self, starts: "list[int]"):
        self.segment_starts = sorted(set(int(x) for x in starts))

    def set_source_frame_index(self, src_idx: int):
        self._source_frame_index = int(src_idx)

    def _current_segment_start(self) -> int:
        starts = getattr(self, "segment_starts", None)
        if starts:
            current = starts[0]
            for start in starts:
                if start > self._frame_index:
                    break
                current = start
            return current
        if self.segment_frames <= 0:
            return 0
        return (self._frame_index // self.segment_frames) * self.segment_frames

    def is_active_frame(self) -> bool:
        return self._current_segment_start() in self.segment_masks

    def composite_green_nv12(self, frame):
        h, w = self._upload(frame)
        import numpy as np

        alpha = np.zeros((self.in_h, self.in_w * 2), dtype=np.float32)
        self._frame_index += 1
        # Alpha passthrough keeps the source image in no-person segments and
        # writes an empty alpha layer instead of producing a green-screen frame.
        t0 = time.perf_counter()
        out = self.packer.pack_uploaded(alpha, h, w)
        self.profile["alpha_pack"].append((time.perf_counter() - t0) * 1000)
        return out, None

    def _bootstrap_mask(self, h: int, w: int, eye_idx: int) -> "np.ndarray":
        masks = self._load_masks()
        if masks is not None:
            return masks[eye_idx]
        segment_start = self._current_segment_start()
        sam3_masks = self.segment_masks.get(segment_start)
        if sam3_masks is None:
            raise RuntimeError(
                f"Missing precomputed MatAnyone2 bootstrap mask for segment_start={segment_start}. "
                "Run the configured prepass or pass --mask."
            )
        alpha = sam3_masks[eye_idx][0, 0]
        # MatAnyone2 writes the first-frame mask into memory, so prefer a
        # conservative foreground seed over a soft or slightly over-wide edge.
        hard = (alpha >= self.bootstrap_threshold).astype(self.np.uint8)
        kernel = self.np.ones((3, 3), self.np.uint8)
        if self.bootstrap_erode > 0:
            hard = self.cv2.erode(hard, kernel, iterations=self.bootstrap_erode)
        if self.bootstrap_dilate > 0:
            hard = self.cv2.dilate(hard, kernel, iterations=self.bootstrap_dilate)
        if self.bootstrap_soft:
            alpha = self.np.minimum(alpha, hard.astype(self.np.float32))
        else:
            alpha = hard.astype(self.np.float32)
        return alpha[None, None, :, :].astype(self.tensor_dtype, copy=False)

    def _preprocess_eye(self, x0: int, eye_w: int) -> "np.ndarray":
        t0 = time.perf_counter()
        image = self.matter._gpu_preprocess_nv12_one(x0, eye_w, self.in_w, self.in_h, copy_to_host=True)
        self.profile["preprocess_eye"].append((time.perf_counter() - t0) * 1000)
        return image.astype(self.tensor_dtype, copy=False)

    def _preprocess_eyes_batch2(self, eye_w: int) -> "np.ndarray":
        t0 = time.perf_counter()
        left = self.matter._gpu_preprocess_nv12_one(0, eye_w, self.in_w, self.in_h, batch=2, batch_idx=0, copy_to_host=True)
        right = self.matter._gpu_preprocess_nv12_one(eye_w, eye_w, self.in_w, self.in_h, batch=2, batch_idx=1, copy_to_host=True)
        self.profile["preprocess_eye"].append((time.perf_counter() - t0) * 1000)
        return self.np.ascontiguousarray(self.np.concatenate([left, right], axis=0)).astype(self.tensor_dtype, copy=False)

    def _upload(self, frame) -> tuple[int, int]:
        from pipeline.pynv_io import GpuP016Frame

        h, w = int(frame.height), int(frame.width)
        t0 = time.perf_counter()
        if isinstance(frame, GpuP016Frame):
            self.matter.upload_p016_planes_as_nv12_gpu(
                frame.y.as_cupy(),
                frame.uv.as_cupy(),
                h,
                w,
                shift_bits=int(config.PASSTHROUGH_PYNV_10BIT_SHIFT),
            )
        else:
            self.matter.upload_nv12_planes_gpu(frame.y.as_cupy(), frame.uv.as_cupy(), h, w)
        self.profile["upload_nv12"].append((time.perf_counter() - t0) * 1000)
        return h, w

    def _run_eye(self, image, h: int, w: int, eye_idx: int) -> "np.ndarray":
        state = self.eyes[eye_idx]
        if not state.initialized:
            feats = self._image_key(image)
            sensory = self.np.zeros(self.sensory_single_shape, dtype=self.tensor_dtype)
            mask = self._bootstrap_mask(h, w, eye_idx)
            msk_value, sensory, obj_memory = self._mask_memory(image, mask, sensory, feats["pix_feat"])
            prob, sensory = self._first_frame_refine(feats, msk_value, obj_memory[:, :, None, :, :], sensory, mask)
            alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
            msk_value, sensory, obj_memory = self._mask_memory(image, alpha, sensory, feats["pix_feat"])
            state.memory_key = feats["key"][:, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.memory_shrinkage = feats["shrinkage"][:, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.memory_msk_value = msk_value[:, :, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.obj_memory = obj_memory[:, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.sensory = sensory.astype(self.tensor_dtype, copy=False)
            state.last_mask = alpha
            state.last_pix_feat = feats["pix_feat"].astype(self.tensor_dtype, copy=False)
            state.last_msk_value = msk_value.astype(self.tensor_dtype, copy=False)
            state.initialized = True
        else:
            prob, sensory, msk_value, _obj_memory, pix_feat = self._step_update(image, state)
            alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
            state.sensory = sensory.astype(self.tensor_dtype, copy=False)
            state.last_mask = alpha
            state.last_pix_feat = pix_feat.astype(self.tensor_dtype, copy=False)
            state.last_msk_value = msk_value.astype(self.tensor_dtype, copy=False)
        return alpha[0, 0]

    def _run_eyes_batch2(self, images, h: int, w: int) -> "np.ndarray":
        states = self.eyes
        if images.shape[0] != 2:
            left = self._run_eye(images[0:1], h, w, 0)
            right = self._run_eye(images[1:2], h, w, 1)
            return self.np.ascontiguousarray(self.np.concatenate([left, right], axis=1))

        if any(not state.initialized for state in states):
            feats = self._image_key(images)
            sensory = self.np.zeros(self.sensory_shape, dtype=self.tensor_dtype)
            masks = self.np.concatenate([self._bootstrap_mask(h, w, 0), self._bootstrap_mask(h, w, 1)], axis=0)
            msk_value, sensory, obj_memory = self._mask_memory(images, masks, sensory, feats["pix_feat"])
            prob, sensory = self._first_frame_refine(
                feats,
                msk_value,
                obj_memory[:, :, None, :, :],
                sensory,
                masks,
            )
            alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
            msk_value, sensory, obj_memory = self._mask_memory(images, alpha, sensory, feats["pix_feat"])
            for idx, state in enumerate(states):
                state.memory_key = feats["key"][idx:idx + 1, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.memory_shrinkage = feats["shrinkage"][idx:idx + 1, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.memory_msk_value = msk_value[idx:idx + 1, :, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.obj_memory = obj_memory[idx:idx + 1, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.sensory = sensory[idx:idx + 1].astype(self.tensor_dtype, copy=False)
                state.last_mask = alpha[idx:idx + 1]
                state.last_pix_feat = feats["pix_feat"][idx:idx + 1].astype(self.tensor_dtype, copy=False)
                state.last_msk_value = msk_value[idx:idx + 1].astype(self.tensor_dtype, copy=False)
                state.initialized = True
            return self.np.ascontiguousarray(self.np.concatenate([alpha[0, 0], alpha[1, 0]], axis=1))

        for other in states[1:]:
            assert other.memory_key is not None
        batched_state = self._EyeState()
        batched_state.memory_key = self.np.concatenate([s.memory_key for s in states], axis=0)
        batched_state.memory_shrinkage = self.np.concatenate([s.memory_shrinkage for s in states], axis=0)
        batched_state.memory_msk_value = self.np.concatenate([s.memory_msk_value for s in states], axis=0)
        batched_state.obj_memory = self.np.concatenate([s.obj_memory for s in states], axis=0)
        batched_state.sensory = self.np.concatenate([s.sensory for s in states], axis=0)
        batched_state.last_mask = self.np.concatenate([s.last_mask for s in states], axis=0)
        batched_state.last_pix_feat = self.np.concatenate([s.last_pix_feat for s in states], axis=0)
        batched_state.last_msk_value = self.np.concatenate([s.last_msk_value for s in states], axis=0)
        batched_state.initialized = True
        prob, sensory, msk_value, _obj_memory, pix_feat = self._step_update(images, batched_state)
        alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
        for idx, state in enumerate(states):
            state.sensory = sensory[idx:idx + 1].astype(self.tensor_dtype, copy=False)
            state.last_mask = alpha[idx:idx + 1]
            state.last_pix_feat = pix_feat[idx:idx + 1].astype(self.tensor_dtype, copy=False)
            state.last_msk_value = msk_value[idx:idx + 1].astype(self.tensor_dtype, copy=False)
        return self.np.ascontiguousarray(self.np.concatenate([alpha[0, 0], alpha[1, 0]], axis=1))

    def composite_nv12(self, frame):
        segment_start = self._current_segment_start()
        if self._active_segment_start != segment_start:
            print(
                f"[offline] MatAnyone2 segment reset at frame={self._frame_index} "
                f"src_idx={self._source_frame_index} segment_start={segment_start}"
            )
            self._reset_segment()
            self._active_segment_start = segment_start
        h, w = self._upload(frame)
        eye_w = w // 2
        if eye_w <= 0 or w < 2 * h:
            raise RuntimeError(f"MatAnyone2 ONNX offline engine expects SBS input, got {w}x{h}")
        should_update = (
            self._cached_alpha_sbs is None
            or self.alpha_stride <= 1
            or self._frame_index % self.alpha_stride == 0
        )
        if should_update and self.batch2_enabled:
            alpha_sbs = self._run_eyes_batch2(self._preprocess_eyes_batch2(eye_w), h, w)
            self._cached_alpha_sbs = alpha_sbs
        elif should_update:
            left = self._run_eye(self._preprocess_eye(0, eye_w), h, w, 0)
            right = self._run_eye(self._preprocess_eye(eye_w, eye_w), h, w, 1)
            t0 = time.perf_counter()
            alpha_sbs = self.np.ascontiguousarray(self.np.concatenate([left, right], axis=1))
            self.profile["alpha_concat"].append((time.perf_counter() - t0) * 1000)
            self._cached_alpha_sbs = alpha_sbs
        else:
            alpha_sbs = self._cached_alpha_sbs
            self.profile["alpha_reuse"].append(0.0)
        self._frame_index += 1
        t0 = time.perf_counter()
        out = self.packer.pack_uploaded(alpha_sbs, h, w)
        self.profile["alpha_pack"].append((time.perf_counter() - t0) * 1000)
        return out, None

    def profile_lines(self) -> list[str]:
        order = [
            "upload_nv12",
            "preprocess_eye",
            "step_update",
            "image_key",
            "propagate_update",
            "propagate",
            "mask_memory",
            "first_refine",
            "alpha_concat",
            "alpha_reuse",
            "alpha_pack",
        ]
        lines = []
        for key in order:
            values = self.profile.get(key)
            if values:
                lines.append(f"matanyone2_{key}_avg = {statistics.fmean(values):.3f} ms n={len(values)}")
        return lines


def _make_engine(args) -> OfflineMattingEngine:
    name = args.engine
    if name == "rvm":
        model_path = Path(args.model).resolve() if args.model else (config.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx").resolve()
        return RvmOfflineEngine(model_path, args=args)
    if name == "matanyone2_onnx":
        model_dir = Path(args._matanyone2_model_dir).resolve()
        return MatAnyone2OnnxEngine(
            model_dir,
            Path(args.mask).resolve() if args.mask else None,
            Path(args.sam3_model_dir).resolve(),
            sam3_prompt=args.sam3_prompt,
            bootstrap_threshold=args.matanyone2_bootstrap_threshold,
            bootstrap_erode=args.matanyone2_bootstrap_erode,
            bootstrap_dilate=args.matanyone2_bootstrap_dilate,
            bootstrap_soft=args.matanyone2_bootstrap_soft,
            segment_frames=args.matanyone2_segment_frames,
            use_fused_update=args.matanyone2_fused_update,
            use_step_update=args.matanyone2_step_update,
            args=args,
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
        cmd = [sys.executable, "tool", "offline_alpha_passthrough"]
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
    parser = argparse.ArgumentParser(description="Generate an offline DeoVR alpha-packed passthrough MP4 from a source video.")
    parser.add_argument("video", help="video path, absolute or relative to PT_VIDEO_DIR")
    parser.add_argument("--out", default="", help="output mp4 path; default is source-stem_SBS_F180_alpha.mp4")
    parser.add_argument("--engine", default="rvm", choices=["rvm", "matanyone2_onnx"])
    parser.add_argument("--model", default="", help="model path; RVM defaults to models/rvm_mobilenetv3_fp32.onnx")
    parser.add_argument("--matanyone2-size", type=int, default=512, choices=[512, 1024],
                        help="MatAnyone2 ONNX size to auto-select when --model is omitted")
    parser.add_argument("--matanyone2-batch", default="1", choices=["auto", "1", "2"],
                        help="MatAnyone2 ONNX batch to auto-select when --model is omitted; auto uses bs2 only for 512 SBS")
    parser.add_argument("--mask", default="", help="first-frame object mask for MatAnyone2")
    parser.add_argument("--matanyone2-prepass", default="sam3", choices=["sam3", "yoloworld_efficientsam"],
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
    parser.add_argument("--matanyone2-segment-frames", type=int, default=300,
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
    parser.add_argument("--no-warmup", action="store_true", help="disable matting warmup for quick offline tests")
    parser.add_argument("--input-size", type=int, default=1024, help="override PT_MATTING_INPUT_SIZE before loading Matter")
    parser.add_argument("--rvm-downsample-ratio", type=float, default=0.5,
                        help="override PT_RVM_DOWNSAMPLE_RATIO before loading RVM Matter")
    parser.add_argument("--alpha-stride", type=int, default=1, help="override PT_ALPHA_STRIDE before loading Matter")
    parser.add_argument("--sbs-batch", action=argparse.BooleanOptionalAction, default=False,
                        help="run left/right SBS eyes as a batch when the RVM model supports batch2")
    parser.add_argument("--fisheye-radius-scale", type=float, default=1.0,
                        help="fisheye circle radius relative to half the per-eye output size")
    parser.add_argument("--alpha-pack-scale", type=float, default=0.4,
                        help="DeoVR alpha packed mask height as a fraction of frame height")
    parser.add_argument("--alpha-pack-blocks-x", type=int, default=3,
                        help="number of horizontal alpha packing blocks")
    parser.add_argument("--alpha-pack-blocks-y", type=int, default=2,
                        help="number of vertical alpha packing blocks")
    args = parser.parse_args()
    args._sam3_child = bool(args.sam3_prepass_out)
    args._ywes_child = bool(args.ywes_prepass_out)
    args._tool_name = "offline_alpha_passthrough"

    import PyNvVideoCodec as nvc
    if args.no_warmup:
        config.MATTING_WARMUP_RUNS = 0
    if args.input_size > 0:
        config.MATTING_INPUT_SIZE = int(args.input_size)
    if args.rvm_downsample_ratio > 0:
        config.RVM_DOWNSAMPLE_RATIO = float(args.rvm_downsample_ratio)
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
    out = Path(args.out) if args.out else _default_out(src)
    if not out.is_absolute():
        out = (config.ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    final_out = out
    video_only_out = out
    if args.audio != "off":
        suffix = "".join(out.suffixes[-1:]) or ".mp4"
        video_only_out = out.with_name(f"{out.stem}._video_only{suffix}")

    meta = probe_video_metadata(src)
    print(
        f"[offline-alpha] input codec={meta.codec.codec_name} profile={meta.codec.profile} "
        f"pix_fmt={meta.codec.pix_fmt} bit_depth={meta.codec.bit_depth} "
        f"size={meta.codec.width}x{meta.codec.height}",
        flush=True,
    )
    print(f"[offline-alpha] cuda {cuda_device_summary(0)}", flush=True)
    backend_decision = select_backend(meta.timing, meta.codec, meta.color)
    decoder_mode = config.PASSTHROUGH_PYNV_DECODER
    if backend_decision.verdict == "ffmpeg_fallback":
        decoder_mode = "ffmpeg_fallback"
    elif backend_decision.verdict == "block":
        raise RuntimeError(f"unsupported offline source: {backend_decision.reason}")
    if decoder_mode == "threaded_serial" and not config.PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER:
        print(
            "[offline-alpha] decoder=threaded_serial requested; using decoder=simple because "
            "PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=0",
            flush=True,
        )
        decoder_mode = "simple"
    alpha_threaded_owned_copy = decoder_mode == "threaded_serial"
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
                "[offline-alpha] ERROR: PyNvVideoCodec/NVDEC rejected this video on the selected CUDA device. "
                "This usually means gpu_id=0 is not the expected RTX GPU, the NVIDIA driver/video stack is "
                "reporting the wrong decode capability, or the source codec/profile exceeds NVDEC support.",
                flush=True,
            )
            print(f"[offline-alpha] ERROR detail: {text}", flush=True)
            return 3
        raise
    info = dec.info
    print(
        f"[offline-alpha] decoder={decoder_mode} "
        f"batch={config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE} "
        f"buffer={config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE} "
        f"owned_copy={alpha_threaded_owned_copy} "
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
    if args.engine == "matanyone2_onnx":
        model_dir = _resolve_matanyone2_model_dir(args, info.width, info.height)
        args._matanyone2_model_dir = str(model_dir)
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        args._matanyone2_in_h = int(manifest.get("height") or 512)
        args._matanyone2_in_w = int(manifest.get("width") or 512)
        args._matanyone2_batch_size = int(manifest.get("batch_size") or 1)
    _require_sam3_vram(args)
    if args.engine == "matanyone2_onnx" and not args.mask and args.matanyone2_prepass == "yoloworld_efficientsam":
        from offline.yoloworld_efficientsam import precompute_segment_masks as _precompute_ywes_segment_masks
        from offline.yoloworld_efficientsam import write_prepass_result as _write_ywes_prepass_result

        sam3_masks, segment_starts = _precompute_ywes_segment_masks(args, src, dec, source_fps, fps, target, cfr_source_index)
    else:
        sam3_masks, segment_starts = _precompute_sam3_segment_masks(args, src, dec, source_fps, fps, target)
    if args.ywes_prepass_out:
        _write_ywes_prepass_result(Path(args.ywes_prepass_out).resolve(), sam3_masks, segment_starts)
        dec.stop()
        return 0
    if args.sam3_prepass_out:
        _write_sam3_prepass_result(Path(args.sam3_prepass_out).resolve(), sam3_masks, segment_starts)
        dec.stop()
        return 0
    if engine is None:
        engine = _make_engine(args)
    enc_w, enc_h = alpha_output_size(info.width, info.height)
    bitrate_kwargs, target_bps, max_bps, buf_bps = _encoder_bitrate_kwargs(args, src, info.width, info.height, enc_w, enc_h)
    projection_mode = AlphaPacker.projection_mode_static(info.width, info.height)
    if isinstance(engine, MatAnyone2OnnxEngine):
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
        f"[offline-alpha] src={src} out={final_out} engine={args.engine} {info.width}x{info.height} "
        f"output={enc_w}x{enc_h} "
        f"source_fps={source_fps:.6f} output_fps={fps:.6f} target={target} audio={args.audio} "
        f"bitrate={target_bps} maxbitrate={max_bps} vbvbufsize={buf_bps} rc={args.rc} cq={args.cq} preset={args.preset}"
    )
    if projection_mode == "flat2d_fisheye":
        alpha_2d_disparity = alpha_2d_disparity_px(enc_w)
        print(
            f"[offline-alpha] projection=flat2d_fisheye -> stereo fisheye sbs "
            f"fov={config.ALPHA_2D_FOV:g} distance={config.ALPHA_2D_DISTANCE_M:g}m "
            f"disparity={alpha_2d_disparity:g}px "
            f"radius_scale={args.fisheye_radius_scale:g}"
        )
    elif projection_mode == "flat2d_3d":
        alpha_2d_disparity = alpha_2d_disparity_px(enc_w)
        print(
            f"[offline-alpha] projection=flat2d_3d -> stereo flat sbs "
            f"safe={config.ALPHA_2D_FLAT3D_SAFE_W:g}x{config.ALPHA_2D_FLAT3D_SAFE_H:g} "
            f"distance={config.ALPHA_2D_DISTANCE_M:g}m disparity={alpha_2d_disparity:g}px"
        )
    else:
        print(
            f"[offline-alpha] projection=sbs half-equirect 180 -> fisheye sbs "
            f"radius_scale={args.fisheye_radius_scale:g}"
        )
    print(
        f"[offline-alpha] packing=DeoVR-style red-channel alpha blocks "
        f"scale={args.alpha_pack_scale:g} layout=alpha-packer-6block"
    )
    print("[offline] mux=" + subprocess.list2cmdline(cmd))

    t_dec: list[float] = []
    t_mat: list[float] = []
    t_sync: list[float] = []
    t_enc: list[float] = []
    t_mux: list[float] = []
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
            if alpha_threaded_owned_copy:
                frame = frame.owned_copy()
            td1 = time.perf_counter()
            if isinstance(engine, MatAnyone2OnnxEngine):
                engine.set_source_frame_index(src_idx)
            if isinstance(engine, MatAnyone2OnnxEngine) and not engine.is_active_frame():
                out_nv12, _ = engine.composite_green_nv12(frame)
            else:
                out_nv12, _ = engine.composite_nv12(frame)
            tm1 = time.perf_counter()
            cuda_stream = getattr(matting_module, "_CUDA_STREAM", None)
            if cuda_stream is not None:
                cuda_stream.synchronize()
            ts1 = time.perf_counter()
            app_frame = GpuNv12AppFrame(out_nv12, enc_w, enc_h)
            flags = 0
            if i == 0:
                flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            te0 = time.perf_counter()
            bitstream = enc.Encode(app_frame, flags) if flags else enc.Encode(app_frame)
            te1 = time.perf_counter()
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
    print(f"mem_start = {used0:.1f}/{total0:.1f} MB")
    print(f"mem_end = {used1:.1f}/{total1:.1f} MB")
    if stderr.strip():
        print("[ffmpeg stderr]")
        print(stderr.strip()[-2000:])
    audio_mux_elapsed = 0.0
    if rc == 0 and args.audio != "off":
        ta0 = time.perf_counter()
        audio_cmd, audio_proc = _mux_audio_after(
            video_only_out,
            final_out,
            src,
            args.audio,
            max(0.0, float(args.start or 0.0)),
            target / fps if fps > 0 else 0.0,
        )
        audio_mux_elapsed = time.perf_counter() - ta0
        print("[offline] audio_mux=" + subprocess.list2cmdline(audio_cmd))
        print(f"audio_mux_rc = {audio_proc.returncode}")
        print(f"audio_mux_elapsed = {audio_mux_elapsed:.3f} s")
        audio_stderr = (audio_proc.stdout or "") + (audio_proc.stderr or "")
        if audio_stderr.strip():
            print("[audio mux stderr]")
            print(audio_stderr.strip()[-2000:])
        if audio_proc.returncode != 0:
            rc = audio_proc.returncode
        else:
            try:
                video_only_out.unlink(missing_ok=True)
            except Exception:
                pass
    print("[ffprobe]")
    print(_ffprobe(final_out if rc == 0 else video_only_out))
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())

