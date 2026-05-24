from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import config

CacheStatus = Literal["missing", "ready", "stale", "failed"]

MANIFEST_VERSION = 1
TRT_MODEL_RVM = "rvm"
TRT_MODEL_MATANYONE2 = "matanyone2"
MODEL_KEY = "rvm_mobilenetv3"
MODEL_LABEL = "Robust Video Matting"
MATANYONE2_MODEL_KEY = "matanyone2_onnx_512_bs1"
MATANYONE2_MODEL_LABEL = "MatAnyone2 ONNX 512 bs1"
MATANYONE2_TRT_ONNX_NAME = "matanyone2_step_update.onnx"
TRT_PROVIDER_CHAIN = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
_CACHE_METADATA_NAMES = {"manifest.json", "build.log"}
_ENGINE_SUFFIXES = {".engine"}
_MIN_ENGINE_BYTES = 1024 * 1024


def normalized_model_key(model_key: str | None = None) -> str:
    key = str(model_key or TRT_MODEL_RVM).strip().lower()
    if key in {"", "rvm", MODEL_KEY}:
        return TRT_MODEL_RVM
    if key in {"matanyone2", MATANYONE2_MODEL_KEY}:
        return TRT_MODEL_MATANYONE2
    return key


def cache_dir_for_model(model_key: str | None = None, cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).resolve()
    key = normalized_model_key(model_key)
    base = config.ONNX_TRT_ENGINE_CACHE_PATH.resolve()
    if key == TRT_MODEL_MATANYONE2:
        return base if base.name == MATANYONE2_MODEL_KEY else base / MATANYONE2_MODEL_KEY
    return base


def manifest_path(model_key: str | None = None, cache_dir: Path | None = None) -> Path:
    return cache_dir_for_model(model_key, cache_dir) / "manifest.json"


def shape_inferred_model_path(model_path: Path | None = None, cache_dir: Path | None = None) -> Path:
    source = Path(config.MODEL_PATH if model_path is None else model_path)
    target_dir = Path(config.ONNX_TRT_ENGINE_CACHE_PATH if cache_dir is None else cache_dir)
    return target_dir / f"{source.stem}_shape_inferred.onnx"


def original_rvm_model_path() -> Path:
    candidate = Path(config.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx").resolve()
    return candidate if candidate.exists() else Path(config.MODEL_PATH).resolve()


def matanyone2_model_dir() -> Path:
    return (config.ROOT / "models" / MATANYONE2_MODEL_KEY).resolve()


def matanyone2_trt_source_model_path() -> Path:
    return matanyone2_model_dir() / MATANYONE2_TRT_ONNX_NAME


def source_model_path(model_key: str | None = None) -> Path:
    key = normalized_model_key(model_key)
    if key == TRT_MODEL_MATANYONE2:
        return matanyone2_trt_source_model_path()
    return original_rvm_model_path()


def model_label(model_key: str | None = None) -> str:
    key = normalized_model_key(model_key)
    if key == TRT_MODEL_MATANYONE2:
        return MATANYONE2_MODEL_LABEL
    return MODEL_LABEL


def trt_runtime_model_path() -> Path:
    path = shape_inferred_model_path()
    return path if path.exists() else config.MODEL_PATH


def is_engine_artifact(path: Path) -> bool:
    if not path.is_file() or path.name in _CACHE_METADATA_NAMES:
        return False
    if path.suffix.lower() not in _ENGINE_SUFFIXES:
        return False
    try:
        return path.stat().st_size >= _MIN_ENGINE_BYTES
    except OSError:
        return False


def _read_json(path: Path) -> dict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def load_manifest() -> dict | None:
    path = manifest_path()
    if not path.exists():
        return None
    return _read_json(path)


def load_manifest_for_model(model_key: str | None = None, cache_dir: Path | None = None) -> dict | None:
    path = manifest_path(model_key, cache_dir)
    if not path.exists():
        return None
    return _read_json(path)


def save_manifest(manifest: dict, model_key: str | None = None, cache_dir: Path | None = None) -> None:
    path = manifest_path(model_key, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_cache(model_key: str | None = None, cache_dir: Path | None = None) -> None:
    cache_dir = cache_dir_for_model(model_key, cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _onnxruntime_version() -> str:
    try:
        import onnxruntime as ort

        return str(getattr(ort, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _nvml_info() -> dict[str, str]:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            return {
                "gpu_uuid": name_or_text(uuid),
                "gpu_name": name_or_text(name),
                "driver_version": name_or_text(driver),
            }
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return _nvidia_smi_info()


def name_or_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _nvidia_smi_driver_version() -> str:
    return _nvidia_smi_info().get("driver_version", "unknown")


def _nvidia_smi_info() -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return {"gpu_uuid": "unknown", "gpu_name": "unknown", "driver_version": "unknown"}
    line = proc.stdout.splitlines()[0].strip() if proc.stdout.splitlines() else ""
    if not line:
        return {"gpu_uuid": "unknown", "gpu_name": "unknown", "driver_version": "unknown"}
    parts = [part.strip() for part in line.split(",", 2)]
    while len(parts) < 3:
        parts.append("unknown")
    return {
        "gpu_uuid": parts[0] or "unknown",
        "gpu_name": parts[1] or "unknown",
        "driver_version": parts[2] or "unknown",
    }


def _cuda_runtime_version() -> str:
    try:
        import cupy as cp

        version = cp.cuda.runtime.runtimeGetVersion()
        major = int(version) // 1000
        minor = (int(version) % 1000) // 10
        return f"{major}.{minor}"
    except Exception:
        return "unknown"


def _dll_product_version(path: Path) -> str | None:
    if not sys.platform.startswith("win") or not path.exists():
        return None
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        ctypes.windll.version.GetFileVersionInfoW(str(path), 0, size, buffer)
        u_len = ctypes.c_uint()
        u_ptr = ctypes.c_void_p()
        ctypes.windll.version.VerQueryValueW(buffer, "\\", ctypes.byref(u_ptr), ctypes.byref(u_len))
        fixed = ctypes.cast(u_ptr, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        ms = fixed[2]
        ls = fixed[3]
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception:
        return None


def _trt_version() -> str:
    try:
        import tensorrt as trt

        return str(getattr(trt, "__version__", "unknown"))
    except Exception:
        pass
    candidates: list[Path] = [Path.cwd()]
    for raw in sys.path:
        if raw:
            candidates.append(Path(raw))
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw:
            candidates.append(Path(raw))
    for directory in candidates:
        for name in ("nvinfer_10.dll", "nvinfer.dll"):
            version = _dll_product_version(directory / name)
            if version:
                return version
    return "unknown"


def collect_fingerprint(model_key: str | None = None, model_path: Path | None = None) -> dict:
    gpu = _nvml_info()
    key = normalized_model_key(model_key)
    model_path = Path(model_path or source_model_path(key))
    fingerprint = {
        **gpu,
        "trt_model_key": key,
        "cuda_runtime": _cuda_runtime_version(),
        "trt_version": _trt_version(),
        "ort_version": _onnxruntime_version(),
        "model_sha256": _sha256(model_path) if model_path.exists() else "missing",
        "trt_fp16": bool(config.ONNX_TRT_FP16_ENABLE),
        "trt_cuda_graph": bool(config.ONNX_TRT_CUDA_GRAPH_ENABLE),
    }
    if key == TRT_MODEL_MATANYONE2:
        fingerprint.update(
            {
                "matanyone2_model_key": MATANYONE2_MODEL_KEY,
                "matanyone2_onnx": MATANYONE2_TRT_ONNX_NAME,
            }
        )
    else:
        fingerprint.update(
            {
                "matting_input_size": int(config.MATTING_INPUT_SIZE),
                "rvm_downsample_ratio": float(config.RVM_DOWNSAMPLE_RATIO),
            }
        )
    return fingerprint


def stale_reasons(saved_fp: dict, actual_fp: dict) -> list[str]:
    reasons: list[str] = []
    for key, actual in actual_fp.items():
        saved = saved_fp.get(key)
        if saved != actual:
            reasons.append(f"{key}: {saved} -> {actual}")
    return reasons


def _ready_models(manifest: dict) -> list[dict]:
    models = manifest.get("models")
    return [model for model in models if isinstance(model, dict) and model.get("status") == "ready"] if isinstance(models, list) else []


def _manifest_engine_files_exist(manifest: dict, model_key: str | None = None, cache_dir: Path | None = None) -> bool:
    key = normalized_model_key(model_key)
    cache_dir = cache_dir_for_model(key, cache_dir)
    ready = _ready_models(manifest)
    if not ready:
        return False
    for model in ready:
        engines = model.get("engines")
        if not isinstance(engines, list) or not engines:
            return False
    if key == TRT_MODEL_RVM and not shape_inferred_model_path(cache_dir=cache_dir).is_file():
        return False
    if key == TRT_MODEL_MATANYONE2 and not matanyone2_trt_source_model_path().is_file():
        return False
    return any(is_engine_artifact(path) for path in cache_dir.iterdir()) if cache_dir.exists() else False


def cache_status(
    actual_fp: dict | None = None,
    manifest: dict | None = None,
    model_key: str | None = None,
    cache_dir: Path | None = None,
) -> CacheStatus:
    key = normalized_model_key(model_key)
    manifest = load_manifest_for_model(key, cache_dir) if manifest is None else manifest
    if not manifest:
        return "missing"
    if int(manifest.get("version", 0) or 0) != MANIFEST_VERSION:
        return "stale"
    if any(model.get("status") == "failed" for model in manifest.get("models", []) if isinstance(model, dict)):
        return "failed"
    if not _manifest_engine_files_exist(manifest, key, cache_dir):
        return "failed"
    actual = collect_fingerprint(key) if actual_fp is None else actual_fp
    saved = manifest.get("fingerprint")
    if not isinstance(saved, dict) or stale_reasons(saved, actual):
        return "stale"
    return "ready"


def build_manifest(
    fingerprint: dict,
    engines: list[dict],
    total_build_seconds: float,
    model_key: str | None = None,
    label: str | None = None,
) -> dict:
    key = normalized_model_key(model_key)
    manifest_model_key = MATANYONE2_MODEL_KEY if key == TRT_MODEL_MATANYONE2 else MODEL_KEY
    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "version": MANIFEST_VERSION,
        "fingerprint": fingerprint,
        "models": [
            {
                "key": manifest_model_key,
                "label": label or model_label(key),
                "engines": engines,
                "total_build_seconds": int(round(total_build_seconds)),
                "status": "ready",
            }
        ],
        "built_at": built_at,
    }
