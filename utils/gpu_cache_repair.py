"""Filesystem-only repair helpers for generated GPU runtime caches.

This module intentionally does not import ``config`` or any GPU runtime.  The
UI must be able to repair a broken cache even when CUDA/ORT cannot initialize.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path


GPU_CACHE_TARGETS = (
    "cuda_compute_cache",
    "cupy",
    "ort",
    "tmp",
    "trt_engines",
    "da3_trt",
    "demosaic_trt",
    "nvds_trt",
    "vsr_onnx_trt",
    "gpu_warmup_marker.json",
    ".warmup.lock",
)
QUARANTINE_ROOT_NAME = "gpu_cache_quarantine"


@dataclass(frozen=True)
class GpuCacheRepairResult:
    quarantine_dir: Path | None
    moved: tuple[str, ...]
    errors: tuple[str, ...]


def quarantine_gpu_caches(runtime_cache_dir: Path, *, stamp: str | None = None) -> GpuCacheRepairResult:
    """Move known generated GPU caches aside without touching user state."""
    root = Path(runtime_cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    quarantine = root / QUARANTINE_ROOT_NAME / stamp
    suffix = 1
    while quarantine.exists():
        quarantine = root / QUARANTINE_ROOT_NAME / f"{stamp}_{suffix}"
        suffix += 1
    moved: list[str] = []
    errors: list[str] = []

    for name in GPU_CACHE_TARGETS:
        source = root / name
        if not source.exists():
            continue
        try:
            quarantine.mkdir(parents=True, exist_ok=True)
            destination = quarantine / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"repair destination already exists: {destination}")
            source.replace(destination)
            moved.append(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if not moved:
        try:
            quarantine.rmdir()
            quarantine.parent.rmdir()
        except OSError:
            pass
        quarantine_dir: Path | None = None
    else:
        quarantine_dir = quarantine
    return GpuCacheRepairResult(quarantine_dir, tuple(moved), tuple(errors))


def discard_quarantine(path: Path | None) -> None:
    """Permanently remove one successful repair quarantine."""
    if path is None:
        return
    target = Path(path).resolve()
    if target.parent.name != QUARANTINE_ROOT_NAME:
        raise ValueError(f"not a GPU cache quarantine directory: {target}")
    shutil.rmtree(target, ignore_errors=False)
    try:
        target.parent.rmdir()
    except OSError:
        pass


def cleanup_old_quarantines(
    runtime_cache_dir: Path,
    *,
    max_age_days: float = 7.0,
    now: float | None = None,
) -> tuple[str, ...]:
    """Remove abandoned GPU cache quarantines older than the retention window."""
    root = Path(runtime_cache_dir).resolve() / QUARANTINE_ROOT_NAME
    if not root.is_dir():
        return ()
    cutoff = float(time.time() if now is None else now) - max(0.0, float(max_age_days)) * 86400.0
    removed: list[str] = []
    for candidate in root.iterdir():
        try:
            if not candidate.is_dir() or candidate.stat().st_mtime > cutoff:
                continue
            discard_quarantine(candidate)
            removed.append(candidate.name)
        except OSError:
            continue
    try:
        root.rmdir()
    except OSError:
        pass
    return tuple(removed)
