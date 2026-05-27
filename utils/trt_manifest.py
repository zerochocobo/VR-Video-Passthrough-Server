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
from utils.subprocess_hidden import hidden_subprocess_kwargs

CacheStatus = Literal["missing", "ready", "stale", "failed"]

MANIFEST_VERSION = 1
TRT_MODEL_RVM = "rvm"
TRT_MODEL_MATANYONE2 = "matanyone2"
MODEL_KEY = "rvm_mobilenetv3"
MODEL_LABEL = "Robust Video Matting"
MATANYONE2_MODEL_KEYS = ("matanyone2_onnx_512_bs1", "matanyone2_onnx_1024_bs1")
MATANYONE2_MODEL_KEY = "matanyone2_onnx_1024_bs1"
MATANYONE2_CACHE_KEY = "matanyone2"
MATANYONE2_MODEL_LABEL = "MatAnyone2 ONNX 512/1024 bs1"
MATANYONE2_TRT_ONNX_NAME = "matanyone2_step_update.onnx"
MATANYONE2_SUPPORTED_SIZES = (512, 1024)
TRT_PROVIDER_CHAIN = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
RVM_OFFLINE_TRT_SHAPES = (
    (1024, 1, 0.5),
    (1024, 2, 0.5),
    (2048, 1, 0.25),
    (2048, 2, 0.25),
    (2048, 1, 0.5),
    (2048, 2, 0.5),
)
_CACHE_METADATA_NAMES = {"manifest.json", "build.log"}
_ENGINE_SUFFIXES = {".engine"}
_MIN_ENGINE_BYTES = 1024 * 1024


def normalized_model_key(model_key: str | None = None) -> str:
    key = str(model_key or TRT_MODEL_RVM).strip().lower()
    if key in {"", "rvm", MODEL_KEY}:
        return TRT_MODEL_RVM
    if key in {"matanyone2", MATANYONE2_CACHE_KEY, *MATANYONE2_MODEL_KEYS}:
        return TRT_MODEL_MATANYONE2
    return key


def cache_dir_for_model(model_key: str | None = None, cache_dir: Path | None = None, scope: str | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).resolve()
    key = normalized_model_key(model_key)
    base = config.ONNX_TRT_ENGINE_CACHE_PATH.resolve()
    if key == TRT_MODEL_RVM and str(scope or "").strip().lower() == "offline":
        if base.name == "offline":
            return base
        return base / "offline"
    if key == TRT_MODEL_MATANYONE2:
        return base if base.name == MATANYONE2_CACHE_KEY else base / MATANYONE2_CACHE_KEY
    return base


def manifest_path(model_key: str | None = None, cache_dir: Path | None = None, scope: str | None = None) -> Path:
    return cache_dir_for_model(model_key, cache_dir, scope) / "manifest.json"


def shape_inferred_model_path(model_path: Path | None = None, cache_dir: Path | None = None) -> Path:
    source = Path(config.MODEL_PATH if model_path is None else model_path)
    target_dir = Path(config.ONNX_TRT_ENGINE_CACHE_PATH if cache_dir is None else cache_dir)
    return target_dir / f"{source.stem}_shape_inferred.onnx"


def original_rvm_model_path() -> Path:
    candidate = Path(config.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx").resolve()
    return candidate if candidate.exists() else Path(config.MODEL_PATH).resolve()


def matanyone2_model_key_for_size(size: int) -> str:
    value = int(size)
    key = f"matanyone2_onnx_{value}_bs1"
    if key not in MATANYONE2_MODEL_KEYS:
        supported = ", ".join(str(item) for item in MATANYONE2_SUPPORTED_SIZES)
        raise ValueError(f"only MatAnyone2 sizes {supported} are supported")
    return key


def matanyone2_model_dir(size: int | None = None) -> Path:
    if size is None:
        key = MATANYONE2_MODEL_KEY
    else:
        key = matanyone2_model_key_for_size(size)
    return (config.ROOT / "models" / key).resolve()


def matanyone2_trt_source_model_path(size: int | None = None) -> Path:
    return matanyone2_model_dir(size) / MATANYONE2_TRT_ONNX_NAME


def matanyone2_trt_source_model_paths() -> dict[str, Path]:
    return {key: (config.ROOT / "models" / key / MATANYONE2_TRT_ONNX_NAME).resolve() for key in MATANYONE2_MODEL_KEYS}


def matanyone2_model_key_for_dir(model_dir: Path) -> str:
    try:
        resolved = model_dir.resolve()
    except OSError:
        resolved = Path(model_dir)
    for key in MATANYONE2_MODEL_KEYS:
        if resolved == (config.ROOT / "models" / key).resolve():
            return key
    return ""


def matanyone2_trt_cache_dir_for_key(model_key: str, cache_dir: Path | None = None) -> Path:
    key = str(model_key or "").strip()
    if key not in MATANYONE2_MODEL_KEYS:
        raise ValueError(f"unsupported MatAnyone2 TensorRT model key: {model_key}")
    root = Path(cache_dir).resolve() if cache_dir is not None else cache_dir_for_model(TRT_MODEL_MATANYONE2)
    return root / key


def matanyone2_trt_cache_dir_for_model_dir(model_dir: Path, cache_dir: Path | None = None) -> Path:
    key = matanyone2_model_key_for_dir(model_dir)
    if not key:
        return cache_dir_for_model(TRT_MODEL_MATANYONE2, cache_dir)
    return matanyone2_trt_cache_dir_for_key(key, cache_dir)


def is_matanyone2_trt_model_dir(model_dir: Path) -> bool:
    return bool(matanyone2_model_key_for_dir(model_dir))


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


def engine_artifact_paths(cache_dir: Path, recursive: bool = False) -> list[Path]:
    if not cache_dir.exists():
        return []
    iterator = cache_dir.rglob("*") if recursive else cache_dir.iterdir()
    return [path for path in iterator if is_engine_artifact(path)]


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


def load_manifest_for_model(model_key: str | None = None, cache_dir: Path | None = None, scope: str | None = None) -> dict | None:
    path = manifest_path(model_key, cache_dir, scope)
    if not path.exists():
        return None
    return _read_json(path)


def save_manifest(manifest: dict, model_key: str | None = None, cache_dir: Path | None = None, scope: str | None = None) -> None:
    path = manifest_path(model_key, cache_dir, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_cache(model_key: str | None = None, cache_dir: Path | None = None, scope: str | None = None) -> None:
    cache_dir = cache_dir_for_model(model_key, cache_dir, scope)
    key = normalized_model_key(model_key)
    offline_scope = str(scope or "").strip().lower() == "offline"
    if key == TRT_MODEL_RVM and not offline_scope and cache_dir.exists():
        for path in cache_dir.iterdir():
            if path.name in {"offline", MATANYONE2_CACHE_KEY, *MATANYONE2_MODEL_KEYS}:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except OSError:
                    pass
    elif cache_dir.exists():
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
            **hidden_subprocess_kwargs(),
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
    if key == TRT_MODEL_MATANYONE2 and model_path == source_model_path(key):
        source_paths = matanyone2_trt_source_model_paths()
        model_sha256 = {
            model_key: _sha256(path) if path.exists() else "missing"
            for model_key, path in source_paths.items()
        }
    else:
        model_sha256 = _sha256(model_path) if model_path.exists() else "missing"
    fingerprint = {
        **gpu,
        "trt_model_key": key,
        "cuda_runtime": _cuda_runtime_version(),
        "trt_version": _trt_version(),
        "ort_version": _onnxruntime_version(),
        "model_sha256": model_sha256,
        "trt_fp16": bool(config.ONNX_TRT_FP16_ENABLE),
        "trt_cuda_graph": bool(config.ONNX_TRT_CUDA_GRAPH_ENABLE),
    }
    if key == TRT_MODEL_MATANYONE2:
        fingerprint.update(
            {
                "matanyone2_model_key": MATANYONE2_MODEL_KEY,
                "matanyone2_model_keys": list(MATANYONE2_MODEL_KEYS),
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


def _manifest_engine_files_exist(
    manifest: dict,
    model_key: str | None = None,
    cache_dir: Path | None = None,
    scope: str | None = None,
) -> bool:
    key = normalized_model_key(model_key)
    cache_dir = cache_dir_for_model(key, cache_dir, scope)
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
    if key == TRT_MODEL_MATANYONE2:
        paths = matanyone2_trt_source_model_paths()
        if not all(path.is_file() for path in paths.values()):
            return False
        fingerprint = manifest.get("fingerprint")
        if not isinstance(fingerprint, dict):
            return False
        saved_keys = fingerprint.get("matanyone2_model_keys")
        if saved_keys != list(MATANYONE2_MODEL_KEYS):
            return False
        for model_name in MATANYONE2_MODEL_KEYS:
            model_cache_dir = matanyone2_trt_cache_dir_for_key(model_name, cache_dir)
            if not engine_artifact_paths(model_cache_dir, recursive=False):
                return False
        return True
    return bool(engine_artifact_paths(cache_dir, recursive=False))


def _rvm_offline_cache_complete(manifest: dict, cache_dir: Path | None = None, scope: str | None = None) -> bool:
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint.get("offline_precision_tiers"):
        return False
    saved_shapes = fingerprint.get("offline_shapes")
    if not isinstance(saved_shapes, list):
        return False
    saved_shape_keys = set()
    for shape in saved_shapes:
        if not isinstance(shape, dict):
            continue
        try:
            saved_shape_keys.add((
                int(shape.get("input_size")),
                int(shape.get("batch")),
                float(shape.get("downsample_ratio")),
            ))
        except (TypeError, ValueError):
            continue
    expected_shape_keys = {(int(size), int(batch), float(downsample)) for size, batch, downsample in RVM_OFFLINE_TRT_SHAPES}
    if not expected_shape_keys.issubset(saved_shape_keys):
        return False

    from utils.rvm_static_onnx import static_rvm_model_path

    source = original_rvm_model_path()
    cache_root = cache_dir_for_model(TRT_MODEL_RVM, cache_dir, scope)
    static_models = [
        static_rvm_model_path(source, cache_root, batch, input_size, downsample)
        for input_size, batch, downsample in RVM_OFFLINE_TRT_SHAPES
    ]
    if not all(path.is_file() for path in static_models):
        return False
    return any(is_engine_artifact(path) for path in cache_root.iterdir()) if cache_root.exists() else False


def cache_status(
    actual_fp: dict | None = None,
    manifest: dict | None = None,
    model_key: str | None = None,
    cache_dir: Path | None = None,
    scope: str | None = None,
) -> CacheStatus:
    key = normalized_model_key(model_key)
    manifest = load_manifest_for_model(key, cache_dir, scope) if manifest is None else manifest
    if not manifest:
        return "missing"
    if int(manifest.get("version", 0) or 0) != MANIFEST_VERSION:
        return "stale"
    if any(model.get("status") == "failed" for model in manifest.get("models", []) if isinstance(model, dict)):
        return "failed"
    if not _manifest_engine_files_exist(manifest, key, cache_dir, scope):
        return "failed"
    offline_scope = str(scope or "").strip().lower() == "offline"
    if key == TRT_MODEL_RVM and offline_scope and not _rvm_offline_cache_complete(manifest, cache_dir, scope):
        return "stale"
    actual = collect_fingerprint(key) if actual_fp is None else actual_fp
    saved = manifest.get("fingerprint")
    if not isinstance(saved, dict):
        return "stale"
    if key == TRT_MODEL_RVM and offline_scope and saved.get("offline_precision_tiers"):
        tier_keys = {"matting_input_size", "rvm_downsample_ratio", "offline_precision_tiers", "offline_shapes"}
        saved_common = {k: v for k, v in saved.items() if k not in tier_keys}
        actual_common = {k: v for k, v in actual.items() if k not in tier_keys}
        if stale_reasons(saved_common, actual_common):
            return "stale"
        return "ready"
    if stale_reasons(saved, actual):
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
    manifest_model_key = MATANYONE2_CACHE_KEY if key == TRT_MODEL_MATANYONE2 else MODEL_KEY
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
