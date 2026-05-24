from __future__ import annotations

import os
import sys
from pathlib import Path

import config


def apply_runtime_dll_paths(env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    paths = _runtime_dll_candidates(target)
    _prepend_path_value(target, paths)
    if env is None and sys.platform.startswith("win"):
        for path in paths:
            if not path.exists():
                continue
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass
    return target


def _runtime_dll_candidates(env: dict[str, str]) -> list[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
        for path in (base / "_internal" / "tensorrt_libs", base / "_internal"):
            if path.exists():
                candidates.append(path)
    else:
        tensorrt_libs = config.ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"
        if tensorrt_libs.exists():
            candidates.append(tensorrt_libs)
    cudnn_bin = env.get("PT_CUDNN_BIN") or os.environ.get("PT_CUDNN_BIN")
    if cudnn_bin:
        candidates.append(Path(cudnn_bin))
    for key in ("CUDA_PATH", "CUDA_HOME"):
        raw = env.get(key) or os.environ.get(key)
        if raw:
            candidates.append(Path(raw) / "bin")
    return candidates


def _prepend_path_value(env: dict[str, str], paths: list[Path]) -> None:
    existing = env.get("PATH", "")
    seen: set[str] = set()
    parts: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = str(path)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    for raw in existing.split(os.pathsep):
        if not raw:
            continue
        key = raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(raw)
    if parts:
        env["PATH"] = os.pathsep.join(parts)
