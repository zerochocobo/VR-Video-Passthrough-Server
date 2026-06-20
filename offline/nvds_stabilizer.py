"""NVDS ONNX depth/near stabilizer for offline 2DVR.

Runtime stays ONNX-only. The PyTorch NVDS checkpoint is converted offline by
``examples/export_nvds_onnx.py``; this module only loads the exported ONNX graph.
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import config
from utils import hf_download

NVDS_SEQUENCE = 4

# Public HF repo holding the converted NVDS ONNX graphs (mirror-aware fetch).
NVDS_HF_REPO = "zerochocobo/NVDS_onnx"

# Resolution tiers. The focal-attention head dominates NVDS cost and scales with
# the stride-4 token count (i.e. the input pixel count), so the lower tier is
# markedly faster -- measured ~3.9 NVDS fps at 672x384 vs ~6.8 at 512x288 -- for
# a modest quality cost. Both stay 16:9. The fast tier is the default.
NVDS_RES_FAST = (512, 288)
NVDS_RES_HIQ = (672, 384)
NVDS_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "512x288": NVDS_RES_FAST,
    "672x384": NVDS_RES_HIQ,
}
NVDS_DEFAULT_RES = "512x288"
NVDS_WIDTH, NVDS_HEIGHT = NVDS_RES_FAST  # default tier
NVDS_TRT_SUPPORTED = False

_RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
_NORM_SAMPLE = 16384


def resolve_resolution(res) -> tuple[int, int]:
    """Map a tier name / ``"WxH"`` / ``(w, h)`` to a concrete (width, height)."""
    if res is None:
        return NVDS_RESOLUTIONS[NVDS_DEFAULT_RES]
    if isinstance(res, (tuple, list)) and len(res) == 2:
        return int(res[0]), int(res[1])
    key = str(res).strip().lower().replace(" ", "")
    if key in NVDS_RESOLUTIONS:
        return NVDS_RESOLUTIONS[key]
    if "x" in key:
        w, _, h = key.partition("x")
        try:
            return int(w), int(h)
        except ValueError:
            pass
    return NVDS_RESOLUTIONS[NVDS_DEFAULT_RES]


def _models_dir() -> Path:
    return config.ROOT / "models" / "NVDS"


def monolith_model_path(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> Path:
    return _models_dir() / f"NVDS_Stabilizer_{width}x{height}.onnx"


def backbone_model_path(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> Path:
    return _models_dir() / f"NVDS_Backbone_{width}x{height}.onnx"


def head_model_path(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> Path:
    return _models_dir() / f"NVDS_Head_{width}x{height}.onnx"


# Backwards-compatible alias (trt_cache_dir, warmup, external callers).
def default_model_path(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> Path:
    return monolith_model_path(width, height)


def split_models_available(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> bool:
    """True when both halves of the split (backbone + head) export exist.

    The split path runs the heavy MiT-B5 backbone once per frame instead of
    recomputing 3/4 of it for every sliding window, so it is preferred whenever
    the two graphs are present.
    """
    return backbone_model_path(width, height).exists() and head_model_path(width, height).exists()


def models_available(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> bool:
    return split_models_available(width, height) or monolith_model_path(width, height).exists()


def resolve_available_resolution(res) -> tuple[int, int]:
    """Resolve ``res`` to a tier whose ONNX files are actually present.

    Falls back to any installed tier so a missing model for the requested tier
    does not hard-fail when another tier is available.
    """
    width, height = resolve_resolution(res)
    if models_available(width, height):
        return width, height
    for cand in NVDS_RESOLUTIONS.values():
        if models_available(*cand):
            return cand
    return width, height


# --- model download (split graphs from Hugging Face) ------------------------


def required_filenames(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> list[str]:
    """The split graph filenames PTMediaServer needs for a resolution tier."""
    return [
        backbone_model_path(width, height).name,
        head_model_path(width, height).name,
    ]


def missing_filenames(width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT) -> list[str]:
    return [name for name in required_filenames(width, height) if not (_models_dir() / name).exists()]


def download_urls(filename: str, language: str | None = None) -> list[str]:
    return hf_download.hf_resolve_urls(NVDS_HF_REPO, filename, language)


def download_targets(
    width: int = NVDS_WIDTH, height: int = NVDS_HEIGHT, language: str | None = None
) -> list[tuple[str, Path, list[str]]]:
    """``(filename, dest_path, urls)`` for every missing split graph at a tier."""
    targets: list[tuple[str, Path, list[str]]] = []
    for name in missing_filenames(width, height):
        targets.append((name, _models_dir() / name, download_urls(name, language)))
    return targets


def trt_cache_dir(model_path: Path | None = None) -> Path:
    stem = Path(model_path or default_model_path()).stem
    path = config.ROOT / "runtime_cache" / "nvds_trt" / stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def trt_engine_cached(model_path: Path | None = None) -> bool:
    if not NVDS_TRT_SUPPORTED:
        return False
    cache = trt_cache_dir(model_path)
    return cache.is_dir() and any(cache.glob("*.engine"))


# VRAM (in MiB) reserved for everything that is NOT the NVDS CUDA arena: the DA3
# TensorRT engine + workspace, the CuPy renderer pools, and the NVENC encoder.
# NVDS is capped at (total VRAM - this reserve) so its arena cannot grow into the
# territory that forces a WDDM spill.
_NVDS_VRAM_RESERVE_MB = 4096


def _total_vram_bytes() -> int:
    """Best-effort total device VRAM in bytes (0 if it cannot be determined)."""
    try:
        import cupy as cp  # the renderer already depends on CuPy

        _, total = cp.cuda.runtime.memGetInfo()
        return int(total)
    except Exception:
        return 0


def _cuda_provider_options() -> dict:
    """CUDA EP options that cap NVDS's arena without slowing its kernels.

    NVDS shares the GPU with DA3 (TensorRT), the CuPy renderer, and NVENC. ORT's
    default arena uses ``kNextPowerOfTwo`` growth and never returns memory, so the
    combined working set creeps past the physical VRAM; on Windows WDDM the driver
    then spills to shared system RAM, which shows up as a progressive slowdown
    that ends in an effective freeze.

    The fix is to keep ORT's fast default arena/kernels (do NOT switch to
    ``kSameAsRequested`` or non-exhaustive conv search -- both measurably slow the
    run) and instead bound the arena with ``gpu_mem_limit`` so it physically
    cannot grow into the spill zone.

    ``PTMS_NVDS_GPU_MEM_LIMIT_MB`` (env) overrides the cap explicitly;
    ``PTMS_NVDS_VRAM_RESERVE_MB`` (env) overrides the headroom left for DA3/CuPy/
    NVENC. When neither is set and total VRAM is known, the cap is
    ``total - reserve``.
    """
    options: dict = {"do_copy_in_default_stream": True}

    limit_mb = 0
    raw = os.environ.get("PTMS_NVDS_GPU_MEM_LIMIT_MB", "").strip()
    if raw:
        try:
            limit_mb = int(float(raw))
        except ValueError:
            limit_mb = 0
    else:
        reserve_mb = _NVDS_VRAM_RESERVE_MB
        reserve_raw = os.environ.get("PTMS_NVDS_VRAM_RESERVE_MB", "").strip()
        if reserve_raw:
            try:
                reserve_mb = int(float(reserve_raw))
            except ValueError:
                reserve_mb = _NVDS_VRAM_RESERVE_MB
        total_mb = _total_vram_bytes() // (1024 * 1024)
        if total_mb > reserve_mb:
            limit_mb = total_mb - reserve_mb

    if limit_mb > 0:
        options["gpu_mem_limit"] = limit_mb * 1024 * 1024
    return options


def onnx_providers(provider: str, model_path: Path | None = None) -> list:
    available = set(ort.get_available_providers())
    provider = str(provider or "trt").lower()
    chain: list = []
    # TensorRT EP currently fails NVDS with a >2GB optimized protobuf and reports
    # that no graph will run on TensorRT. Treat `trt` as CUDA for NVDS to avoid a
    # long failed build attempt and noisy logs; DA3 still uses TRT normally.
    if provider == "trt" and NVDS_TRT_SUPPORTED and "TensorrtExecutionProvider" in available:
        chain.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_dir(model_path)),
                "trt_timing_cache_enable": True,
            },
        ))
    if provider in ("trt", "cuda") and "CUDAExecutionProvider" in available:
        chain.append(("CUDAExecutionProvider", _cuda_provider_options()))
    chain.append("CPUExecutionProvider")
    return chain


def _resize_rgb(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = cv2.INTER_AREA if frame_rgb.shape[1] >= width and frame_rgb.shape[0] >= height else cv2.INTER_CUBIC
    return cv2.resize(frame_rgb, (width, height), interpolation=interpolation)


def _normalize_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame_rgb, dtype=np.float32) * np.float32(1.0 / 255.0)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return (chw - _RGB_MEAN) / _RGB_STD


def _depth_to_near(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    inv = np.reciprocal(np.maximum(d, 1e-6))
    sample = inv[::4, ::4].ravel()
    if sample.size > _NORM_SAMPLE:
        sample = sample[:: max(1, sample.size // _NORM_SAMPLE)]
    lo, hi = (float(v) for v in np.percentile(sample, [5.0, 95.0]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(d.shape, dtype=np.float32)
    near = (inv - np.float32(lo)) * np.float32(1.0 / (hi - lo))
    np.clip(near, 0.0, 1.0, out=near)
    return near.astype(np.float32, copy=False)


def _normalize_unit(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros(x.shape, dtype=np.float32)
    lo = float(np.min(x[finite]))
    hi = float(np.max(x[finite]))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.float32)
    out = (x - np.float32(lo)) * np.float32(1.0 / (hi - lo))
    np.clip(out, 0.0, 1.0, out=out)
    return np.ascontiguousarray(out, dtype=np.float32)


def is_16x9(width: int, height: int, tolerance: float = 0.02) -> bool:
    if width <= 0 or height <= 0:
        return False
    return abs((float(width) / float(height)) - (16.0 / 9.0)) <= tolerance


class NvdsDepthStabilizer:
    """Causal 4-frame NVDS wrapper returning a normalized near/disparity map."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        provider: str = "trt",
        resolution=None,
        width: int | None = None,
        height: int | None = None,
        split: bool | None = None,
    ) -> None:
        if width is not None and height is not None:
            self.width, self.height = int(width), int(height)
        else:
            self.width, self.height = resolve_resolution(resolution)
        self.resolution = f"{self.width}x{self.height}"
        self.provider = str(provider)
        self.split = (
            split_models_available(self.width, self.height) if split is None else bool(split)
        )

        if self.split:
            self.mode = "split"
            self.backbone_session = self._make_session(backbone_model_path(self.width, self.height))
            self.head_session = self._make_session(head_model_path(self.width, self.height))
            self.providers = self.backbone_session.get_providers()
            self._backbone_input = self.backbone_session.get_inputs()[0].name
            self._backbone_outputs = [o.name for o in self.backbone_session.get_outputs()]
            head_inputs = self.head_session.get_inputs()
            self._head_last_rgb = next(i.name for i in head_inputs if "rgb" in i.name.lower())
            self._head_feats = [i.name for i in head_inputs if "rgb" not in i.name.lower()]
            self._head_output = self.head_session.get_outputs()[0].name
            # Cache the last NVDS_SEQUENCE backbone outputs (one tuple of feature
            # maps per frame) so each new frame only pays one backbone pass.
            self._feat_history: deque[list[np.ndarray]] = deque(maxlen=NVDS_SEQUENCE)
        else:
            self.mode = "monolith"
            self.model_path = Path(model_path or monolith_model_path(self.width, self.height))
            if not self.model_path.exists():
                raise FileNotFoundError(f"NVDS ONNX model not found: {self.model_path}")
            self.session = self._make_session(self.model_path)
            self.providers = self.session.get_providers()
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self._history: deque[np.ndarray] = deque(maxlen=NVDS_SEQUENCE)

        self.frames = 0
        self.inference_seconds = 0.0

    def _make_session(self, model_path: Path) -> "ort.InferenceSession":
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.provider.lower() == "trt":
            opts.log_severity_level = 4
        return ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=onnx_providers(self.provider, model_path),
        )

    def reset(self) -> None:
        if self.split:
            self._feat_history.clear()
        else:
            self._history.clear()
        self.frames = 0
        self.inference_seconds = 0.0

    @staticmethod
    def _pad_to_window(items: list) -> list:
        if len(items) < NVDS_SEQUENCE:
            return [items[0]] * (NVDS_SEQUENCE - len(items)) + items
        return items

    def _preprocess(self, frame_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        rgb = _resize_rgb(frame_rgb, self.width, self.height)
        rgb_chw = _normalize_rgb(rgb)
        near = _depth_to_near(depth)
        near = cv2.resize(near, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(np.concatenate([rgb_chw, near[None]], axis=0), dtype=np.float32)

    def stabilize(self, frame_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if self.split:
            return self._stabilize_split(frame_rgb, depth)
        return self._stabilize_monolith(frame_rgb, depth)

    def _stabilize_monolith(self, frame_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        current = self._preprocess(frame_rgb, depth)
        self._history.append(current)
        seq = self._pad_to_window(list(self._history))
        tensor = np.ascontiguousarray(np.stack(seq, axis=0)[None], dtype=np.float32)
        start = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: tensor})[0][0, 0]
        self.inference_seconds += time.perf_counter() - start
        self.frames += 1
        return _normalize_unit(output)

    def _stabilize_split(self, frame_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        current = self._preprocess(frame_rgb, depth)  # [4, H, W]
        last_rgb = np.ascontiguousarray(current[None, 0:3], dtype=np.float32)  # [1, 3, H, W]
        start = time.perf_counter()
        # One backbone pass for the current frame; reuse the cached neighbours.
        feats = self.backbone_session.run(
            self._backbone_outputs, {self._backbone_input: current[None]}
        )
        self._feat_history.append(feats)
        window = self._pad_to_window(list(self._feat_history))
        feeds = {
            name: np.ascontiguousarray(
                np.concatenate([window[t][s] for t in range(NVDS_SEQUENCE)], axis=0)
            )
            for s, name in enumerate(self._head_feats)
        }
        feeds[self._head_last_rgb] = last_rgb
        output = self.head_session.run([self._head_output], feeds)[0][0, 0]
        self.inference_seconds += time.perf_counter() - start
        self.frames += 1
        return _normalize_unit(output)

    def fps_summary(self) -> str:
        if self.frames <= 0:
            return f"nvds_frames=0 mode={self.mode}"
        ms = self.inference_seconds * 1000.0 / max(1, self.frames)
        fps = self.frames / max(1e-6, self.inference_seconds)
        return (
            f"nvds_frames={self.frames} mode={self.mode} "
            f"nvds_infer={ms:.1f}ms/frame ({fps:.1f} fps)"
        )


def warmup(model_path: Path | None = None, provider: str = "trt", resolution=None) -> NvdsDepthStabilizer:
    stabilizer = NvdsDepthStabilizer(model_path=model_path, provider=provider, resolution=resolution)
    h, w = stabilizer.height, stabilizer.width
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.linspace(1.0, 2.0, h * w, dtype=np.float32).reshape(h, w)
    stabilizer.stabilize(frame, depth)
    return stabilizer
