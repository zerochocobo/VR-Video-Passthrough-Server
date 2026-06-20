"""Depth Anything 3 depth estimation via ONNX Runtime.

Drives the ``da3_small.onnx`` / ``da3_base.onnx`` graphs exported by
``examples/da3_to_onnx.py``. The graph takes ImageNet-normalised RGB at a fixed
square side (518, the native DINOv2 37x37 patch grid) and returns a same-sized
distance-like depth map. We letterbox each frame to the square, run a batch, then
un-letterbox + resize depth back to the source frame -- mirroring the existing
ORT engines (rvm / birefnet) in this package.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import config

DA3_INPUT_SIZE = 518
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
# HWC-space constants for cheap normalization (mean*255, 1/(std*255)).
_MEAN_HWC = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :] * 255.0
_INV_STD_HWC = np.float32(1.0) / (np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :] * 255.0)

# User-facing model presets. The key is the value carried by `PT_TWO_DVR_MODEL` /
# the CLI `--model` / the UI dropdowns. Each maps to an ONNX file and its fixed
# input size; the TRT engine cache lives in a per-preset subdir (engine shape
# differs by size, so base and base_hd must not share a cache dir).
DA3_PRESETS: dict[str, dict] = {
    "base":     {"variant": "base",  "size": 518,  "file": "da3_base.onnx"},
    "small":    {"variant": "small", "size": 518,  "file": "da3_small.onnx"},
    "base_hd":  {"variant": "base",  "size": 1036, "file": "da3_base_1036.onnx"},
    "small_hd": {"variant": "small", "size": 1036, "file": "da3_small_1036.onnx"},
    "large_hd": {"variant": "large", "size": 1036, "file": "da3_large_1036.onnx"},
}
DEFAULT_MODEL = "base"
# Back-compat alias (only da3_depth referenced it).
MODEL_FILES = {k: v["file"] for k, v in DA3_PRESETS.items()}
_ENGINE_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: dict[tuple[str, str], "Da3DepthEngine"] = {}


def normalize_model(model: str) -> str:
    """Map any model string to a known preset key (default base)."""
    m = str(model or "").strip().lower()
    return m if m in DA3_PRESETS else DEFAULT_MODEL


def model_preset(model: str) -> dict:
    return DA3_PRESETS[normalize_model(model)]


def _trt_cache_dir(model: str) -> Path:
    path = config.ROOT / "runtime_cache" / "da3_trt" / normalize_model(model)
    path.mkdir(parents=True, exist_ok=True)
    return path


def onnx_providers(provider: str, variant: str) -> list:
    """Provider chain. ``trt`` adds the TensorRT EP (fp16 + cached engine) in
    front of CUDA -- it fuses the whole ViT and cuts DA3-Small from ~38 ms to
    ~5 ms/frame on sm_120. Falls back to CUDA, then CPU."""
    available = set(ort.get_available_providers())
    chain: list = []
    if provider == "trt" and "TensorrtExecutionProvider" in available:
        chain.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(_trt_cache_dir(variant)),
                "trt_timing_cache_enable": True,
            },
        ))
    if provider in ("trt", "cuda") and "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


def default_model_path(model: str) -> Path:
    return config.ROOT / "models" / "DA3" / model_preset(model)["file"]


def trt_engine_cached(model: str) -> bool:
    """True if a TensorRT engine has been built+cached for this preset."""
    cache = config.ROOT / "runtime_cache" / "da3_trt" / normalize_model(model)
    return cache.is_dir() and any(cache.glob("*.engine"))


# --- model download ---------------------------------------------------------
# Public HF repo holding the converted ONNX graphs. HF_ENDPOINT lets users point
# at a specific endpoint; otherwise Chinese UI/runtime uses hf-mirror.com
# directly, while other languages try Hugging Face first.
HF_REPO = "zerochocobo/DepthAnything3_ONNX"
HF_DEFAULT_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def model_available(model: str) -> bool:
    return default_model_path(model).exists()


def _download_endpoints() -> list[str]:
    override = str(os.environ.get("HF_ENDPOINT") or "").strip()
    if override:
        return [override.rstrip("/")]
    language = str(os.environ.get("PT_UI_LANGUAGE") or "").lower()
    if language.startswith("zh"):
        return [HF_MIRROR_ENDPOINT]
    return [HF_DEFAULT_ENDPOINT, HF_MIRROR_ENDPOINT]


def model_download_urls(model: str) -> list[str]:
    filename = model_preset(model)["file"]
    return [f"{endpoint}/{HF_REPO}/resolve/main/{filename}" for endpoint in _download_endpoints()]


def model_download_url(model: str) -> str:
    return model_download_urls(model)[0]


def download_target(model: str, language: str | None = None) -> tuple[str, Path, list[str]]:
    """``(filename, dest_path, urls)`` for a preset, for the UI download dialog.

    ``language`` lets the in-process UI force mirror selection; the existing
    ``model_download_urls`` keeps using the env-based endpoint logic.
    """
    from utils import hf_download

    filename = model_preset(model)["file"]
    return filename, default_model_path(model), hf_download.hf_resolve_urls(HF_REPO, filename, language)


def download_model(model: str, log=print, progress=None) -> Path:
    """Download a missing DA3 ONNX preset from Hugging Face (mirror-aware).

    Streams to a ``.part`` file then atomically renames. ``progress(done, total)``
    is called as bytes arrive (total may be 0 if the server omits Content-Length).
    """
    dest = default_model_path(model)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    urls = model_download_urls(model)
    last_error: Exception | None = None
    for index, url in enumerate(urls):
        log(f"DA3 download: {normalize_model(model)} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "PTMediaServer/DA3"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            tmp.replace(dest)
            log(f"DA3 download: {normalize_model(model)} done ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as exc:
            last_error = exc
            try:
                tmp.unlink()
            except OSError:
                pass
            if index + 1 < len(urls):
                log(f"DA3 download failed from {url}: {type(exc).__name__}: {exc}; trying next mirror")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"no download endpoint configured for {normalize_model(model)}")


def ensure_model_available(model: str, log=print, progress=None) -> bool:
    """Return True if the preset's ONNX exists, downloading it if missing.

    Returns False (and logs) instead of raising so callers can fall back."""
    if model_available(model):
        return True
    try:
        download_model(model, log=log, progress=progress)
        return True
    except Exception as exc:
        log(f"DA3 download failed for {normalize_model(model)}: {type(exc).__name__}: {exc}")
        return False


class Da3DepthEngine:
    """ONNX Runtime DA3 depth estimator (batched, letterboxed to a square)."""

    def __init__(self, variant: str = DEFAULT_MODEL, model_path: Path | None = None,
                 provider: str = "trt", size: int | None = None):
        # ``variant`` is the user-facing preset key (base/small/base_hd/small_hd).
        self.model = normalize_model(variant)
        preset = DA3_PRESETS[self.model]
        self.variant = preset["variant"]          # backbone (small/base)
        self.size = int(size) if size else int(preset["size"])
        path = Path(model_path) if model_path else (config.ROOT / "models" / "DA3" / preset["file"])
        if not path.exists():
            raise FileNotFoundError(
                f"DA3 ONNX model not found: {path}. Run examples/da3_to_onnx.py "
                f"(or download it) first."
            )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "trt":
            # Folded-preprocess DA3 models start with UINT8 preprocessing nodes.
            # TensorRT logs parser errors for those nodes, then ORT partitions
            # them to CUDA and keeps the supported depth trunk on TRT. Suppress
            # that known-benign noise while still surfacing our own failures.
            opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=onnx_providers(provider, self.model)
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # Folded-preprocess models take a uint8 (B,size,size,3) canvas and do the
        # ImageNet normalize on-device; plain models take a normalized float NCHW
        # tensor. Branch on the graph's declared input type so either works.
        self.folded = "uint8" in str(self.session.get_inputs()[0].type)
        # Reused letterbox canvas (single-frame hot path).
        self._canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

    # -- preprocessing -------------------------------------------------------

    def _letterbox(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        h, w = frame_rgb.shape[:2]
        scale = min(self.size / max(1, w), self.size / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # INTER_LINEAR, not INTER_AREA: ~3x faster downscale and the depth net is
        # insensitive to the small quality difference at this scale.
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = self._canvas
        canvas[:] = 0
        x0 = (self.size - new_w) // 2
        y0 = (self.size - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas, (x0, y0, new_w, new_h)

    def _normalize(self, canvas_rgb: np.ndarray) -> np.ndarray:
        # Normalize in HWC (contiguous, cache-friendly) then transpose once.
        hwc = (canvas_rgb.astype(np.float32) - _MEAN_HWC) * _INV_STD_HWC
        return np.ascontiguousarray(hwc.transpose(2, 0, 1))

    # -- inference -----------------------------------------------------------

    def predict_batch(self, frames_rgb: list[np.ndarray], upscale: bool = True) -> list[np.ndarray]:
        """Return one depth map per input frame.

        ``upscale=True`` resizes each depth back to its source frame size.
        ``upscale=False`` returns the model-resolution crop (the letterbox box,
        aspect-correct, <=518 on the long side) -- the hot path uses this and
        upscales the *disparity* instead, since normalizing 2M pixels per frame
        is far more expensive than normalizing ~150k.
        """
        if not frames_rgb:
            return []
        boxes = []
        if self.folded:
            tensor = np.empty((len(frames_rgb), self.size, self.size, 3), dtype=np.uint8)
            for i, frame in enumerate(frames_rgb):
                canvas, box = self._letterbox(frame)
                boxes.append(box)
                tensor[i] = canvas
        else:
            tensor = np.empty((len(frames_rgb), 3, self.size, self.size), dtype=np.float32)
            for i, frame in enumerate(frames_rgb):
                canvas, box = self._letterbox(frame)
                boxes.append(box)
                tensor[i] = self._normalize(canvas)
        depths = self.session.run([self.output_name], {self.input_name: tensor})[0]
        out = []
        for i, frame in enumerate(frames_rgb):
            depth = np.asarray(depths[i], dtype=np.float32)
            x0, y0, new_w, new_h = boxes[i]
            crop = depth[y0:y0 + new_h, x0:x0 + new_w]
            if upscale:
                h, w = frame.shape[:2]
                out.append(cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR))
            else:
                out.append(np.ascontiguousarray(crop))
        return out

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_rgb])[0]


def _dummy_input_for_engine(engine: Da3DepthEngine) -> np.ndarray:
    if engine.folded:
        return np.zeros((1, engine.size, engine.size, 3), dtype=np.uint8)
    return np.zeros((1, 3, engine.size, engine.size), dtype=np.float32)


def warmup_depth_engine(variant: str = DEFAULT_MODEL, provider: str = "trt", log=print) -> Da3DepthEngine:
    """Create, warm, and retain a DA3 engine for realtime reuse."""
    model = normalize_model(variant)
    provider_key = str(provider or "trt").lower()
    key = (model, provider_key)
    with _ENGINE_CACHE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None:
            return cached
        engine = Da3DepthEngine(variant=model, provider=provider_key)
        engine.session.run([engine.output_name], {engine.input_name: _dummy_input_for_engine(engine)})
        _ENGINE_CACHE[key] = engine
        log(
            f"DA3 engine warmup retained: variant={model} provider="
            f"{engine.providers[0] if engine.providers else 'unknown'} folded={engine.folded} size={engine.size}"
        )
        return engine
