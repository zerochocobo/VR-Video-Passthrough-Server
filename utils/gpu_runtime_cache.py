"""Runtime cache and startup warmup helpers for CUDA, CuPy, and ORT.

The first ONNX Runtime CUDA run can spend a long time loading DLLs, compiling
kernels, and building provider caches. These helpers pin cache directories under
runtime_cache, build a machine/model-specific warmup key, and optionally run a
short matting warmup before the DLNA port starts accepting requests.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import config


RUNTIME_CACHE_DIR = config.RUNTIME_CACHE_DIR
CUDA_CACHE_DIR = config.CUDA_CACHE_PATH
CUPY_CACHE_DIR = config.CUPY_CACHE_DIR
ORT_CACHE_DIR = config.ORT_CACHE_DIR
MARKER_PATH = config.GPU_WARMUP_MARKER
LOCK_PATH = config.GPU_WARMUP_LOCK


@dataclass(frozen=True)
class GpuRuntimeCacheEnv:
    runtime_cache_dir: str
    cuda_cache_path: str
    cupy_cache_dir: str
    ort_cache_dir: str
    marker_path: str


@dataclass(frozen=True)
class GpuWarmupKey:
    gpu_name: str
    compute_capability: str
    driver_version: str
    python_version: str
    onnxruntime_version: str
    onnxruntime_providers_cuda_dll_hash: str
    cupy_version: str
    cupy_cuda_runtime: str
    model_name: str
    model_sha256_16: str
    input_size: int
    providers: list[str]
    shapes: list[list[int]]


@dataclass(frozen=True)
class GpuWarmupMarker:
    key: GpuWarmupKey
    cuda_cache_path: str
    cupy_cache_dir: str
    cache_size_after_warmup: int
    cache_file_count_after_warmup: int
    elapsed_sec: float
    verified_second_pass_sec: float
    created_at: str


class WarmupLock:
    def __init__(self, path: Path, timeout_sec: float = 300.0, poll_sec: float = 1.0, stale_sec: float = 3600.0):
        self.path = path
        self.timeout_sec = timeout_sec
        self.poll_sec = poll_sec
        self.stale_sec = stale_sec
        self._fd: int | None = None

    def _try_clear_stale(self) -> None:
        try:
            age = time.time() - self.path.stat().st_mtime
            if age < self.stale_sec:
                return
            text = self.path.read_text(encoding="ascii", errors="ignore")
            pid = 0
            for part in text.replace("\n", " ").split():
                if part.startswith("pid="):
                    pid = int(part.split("=", 1)[1])
                    break
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    return
                except OSError:
                    pass
            self.path.unlink()
        except Exception:
            return

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_sec
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
                return self
            except FileExistsError:
                self._try_clear_stale()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for GPU warmup lock: {self.path}")
                time.sleep(self.poll_sec)

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def configure_gpu_runtime_cache() -> GpuRuntimeCacheEnv:
    """Set CUDA/CuPy cache env vars before importing CUDA-heavy modules."""
    runtime_dir = Path(config.RUNTIME_CACHE_DIR).resolve()
    cuda_dir = Path(config.CUDA_CACHE_PATH).resolve()
    cupy_dir = Path(config.CUPY_CACHE_DIR).resolve()
    ort_dir = Path(config.ORT_CACHE_DIR).resolve()
    marker_path = Path(config.GPU_WARMUP_MARKER).resolve()
    tmp_dir = Path(config.RUNTIME_TMP_DIR).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)
    cupy_dir.mkdir(parents=True, exist_ok=True)
    ort_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_DISABLE"] = config.CUDA_CACHE_DISABLE
    os.environ["CUDA_CACHE_MAXSIZE"] = config.CUDA_CACHE_MAXSIZE
    os.environ["CUDA_CACHE_PATH"] = str(cuda_dir)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return GpuRuntimeCacheEnv(
        runtime_cache_dir=str(runtime_dir),
        cuda_cache_path=str(cuda_dir),
        cupy_cache_dir=str(cupy_dir),
        ort_cache_dir=str(ort_dir),
        marker_path=str(marker_path),
    )


def _sha256_16(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


def _cache_stats(path: Path) -> tuple[int, int]:
    count = 0
    total = 0
    if not path.exists():
        return 0, 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                count += 1
                total += file.stat().st_size
        except OSError:
            continue
    return count, total


def _driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.splitlines()[0].strip()
    except Exception:
        return ""


def _providers_cuda_hash() -> str:
    try:
        import onnxruntime as ort

        capi = Path(ort.__file__).resolve().parent / "capi"
        candidates = sorted(capi.glob("onnxruntime_providers_cuda*.dll"))
        return _sha256_16(candidates[0]) if candidates else ""
    except Exception:
        return ""


def build_warmup_key(shapes: Iterable[tuple[int, int, int, int]] | None = None) -> GpuWarmupKey:
    configure_gpu_runtime_cache()
    import cupy as cp
    import onnxruntime as ort

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    model_path = Path(config.MODEL_PATH).resolve()
    shape_list = [list(s) for s in (shapes or default_warmup_shapes())]
    return GpuWarmupKey(
        gpu_name=name,
        compute_capability=f"{props['major']}.{props['minor']}",
        driver_version=_driver_version(),
        python_version=sys.version.split()[0],
        onnxruntime_version=str(getattr(ort, "__version__", "")),
        onnxruntime_providers_cuda_dll_hash=_providers_cuda_hash(),
        cupy_version=str(getattr(cp, "__version__", "")),
        cupy_cuda_runtime=str(cp.cuda.runtime.runtimeGetVersion()),
        model_name=model_path.name,
        model_sha256_16=_sha256_16(model_path),
        input_size=int(config.MATTING_INPUT_SIZE),
        providers=[p.strip() for p in config.ONNX_PROVIDERS if p.strip()],
        shapes=shape_list,
    )


def default_warmup_shapes() -> list[tuple[int, int, int, int]]:
    size = int(config.MATTING_INPUT_SIZE)
    shapes = [(1, 3, size, size)]
    if config.MATTING_SPLIT_SBS and config.MATTING_SBS_BATCH:
        shapes.append((2, 3, size, size))
    return shapes


def marker_matches(key: GpuWarmupKey, marker_path: Path = MARKER_PATH) -> bool:
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8-sig"))
        return data.get("key") == asdict(key)
    except Exception:
        return False


def write_marker(marker: GpuWarmupMarker, marker_path: Path = MARKER_PATH) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(asdict(marker), indent=2, ensure_ascii=False), encoding="utf-8")


def warmup_gpu_runtime_cache(force: bool = False, timeout_sec: float = 300.0, runs_per_shape: int = 3) -> GpuWarmupMarker:
    env = configure_gpu_runtime_cache()
    marker_path = Path(env.marker_path)
    key = build_warmup_key()
    if not force and marker_matches(key, marker_path):
        count, size = _cache_stats(Path(env.cuda_cache_path))
        return GpuWarmupMarker(
            key=key,
            cuda_cache_path=env.cuda_cache_path,
            cupy_cache_dir=env.cupy_cache_dir,
            cache_size_after_warmup=size,
            cache_file_count_after_warmup=count,
            elapsed_sec=0.0,
            verified_second_pass_sec=0.0,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    start = time.perf_counter()
    with WarmupLock(LOCK_PATH, timeout_sec=timeout_sec):
        key = build_warmup_key()
        if not force and marker_matches(key, marker_path):
            count, size = _cache_stats(Path(env.cuda_cache_path))
            return GpuWarmupMarker(
                key=key,
                cuda_cache_path=env.cuda_cache_path,
                cupy_cache_dir=env.cupy_cache_dir,
                cache_size_after_warmup=size,
                cache_file_count_after_warmup=count,
                elapsed_sec=time.perf_counter() - start,
                verified_second_pass_sec=0.0,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
        from pipeline.matting import Matter

        old_warmup = config.MATTING_WARMUP_RUNS
        config.MATTING_WARMUP_RUNS = 0
        try:
            matter = Matter()
        finally:
            config.MATTING_WARMUP_RUNS = old_warmup

        import cupy as cp
        import pipeline.matting as matting_mod

        verify_start = 0.0
        verify_elapsed = 0.0
        for shape in key.shapes:
            batch, channels, h, w = shape
            if channels != 3:
                continue
            x = cp.zeros((batch, channels, h, w), dtype=matter.input_dtype)
            matter.reset_state()
            for i in range(max(1, runs_per_shape)):
                t0 = time.perf_counter()
                matter._run_rvm_iobinding_from_dev(x)
                stream = getattr(matting_mod, "_CUDA_STREAM", None)
                if stream is not None:
                    stream.synchronize()
                else:
                    cp.cuda.Stream.null.synchronize()
                if i == max(1, runs_per_shape) - 1:
                    verify_elapsed += time.perf_counter() - t0

        count, size = _cache_stats(Path(env.cuda_cache_path))
        marker = GpuWarmupMarker(
            key=key,
            cuda_cache_path=env.cuda_cache_path,
            cupy_cache_dir=env.cupy_cache_dir,
            cache_size_after_warmup=size,
            cache_file_count_after_warmup=count,
            elapsed_sec=time.perf_counter() - start,
            verified_second_pass_sec=verify_elapsed,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        write_marker(marker, marker_path)
        return marker
