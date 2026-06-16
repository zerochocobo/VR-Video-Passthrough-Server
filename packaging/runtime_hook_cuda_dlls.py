"""PyInstaller runtime bootstrap for CUDA/CuPy DLL discovery on Windows.

CuPy's Windows import path probes CUDA DLL locations before application code
can run. In onedir builds PyInstaller places collected CUDA DLLs under
``_internal``; CuPy can then infer the distribution root as ``CUDA_PATH`` and
try to add ``<dist>\\bin``. Keep that directory present and explicitly add the
real bundled DLL locations before CuPy is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_HANDLES: list[object] = []


def _add_dll_dir(path: Path) -> None:
    if not path.exists():
        return
    try:
        _DLL_HANDLES.append(os.add_dll_directory(str(path)))
    except (FileNotFoundError, OSError):
        return


def _prepend_path(paths: list[Path]) -> None:
    existing = os.environ.get("PATH", "")
    seen: set[str] = set()
    parts: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(str(path))
    for raw in existing.split(os.pathsep):
        if not raw:
            continue
        key = raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(raw)
    if parts:
        os.environ["PATH"] = os.pathsep.join(parts)


if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
    # The package now bundles the pip CUDA 12.x NVRTC (>=12.8), which CAN emit
    # native cubins for Blackwell sm_120. So we keep CuPy on its default direct-
    # cubin path (PTX disabled) -- this avoids the PTX->driver JIT cold-start
    # ("cuda compute cache" recompile) that CUPY_COMPILE_WITH_PTX=1 incurred with
    # the old system CUDA 12.6 NVRTC. nvrtc 12.x supports sm_50..sm_120, covering
    # every GPU this product targets. Override to "1" if a future GPU needs the
    # driver-JIT fallback.
    os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "0")

    exe_dir = Path(sys.executable).resolve().parent
    internal_dir = Path(getattr(sys, "_MEIPASS", exe_dir / "_internal")).resolve()
    bundled_cuda_root = exe_dir
    bundled_cuda_bin = exe_dir / "bin"
    onnxruntime_capi_dir = internal_dir / "onnxruntime" / "capi"

    # CuPy checks CUDA_PATH before probing wheel-bundled libraries. In a frozen
    # distribution we want deterministic bundled CUDA components, not a user's
    # stale or broken system CUDA Toolkit.
    if "PT_ORIGINAL_CUDA_PATH" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["PT_ORIGINAL_CUDA_PATH"] = os.environ["CUDA_PATH"]
    if "PT_ORIGINAL_CUDA_HOME" not in os.environ and "CUDA_HOME" in os.environ:
        os.environ["PT_ORIGINAL_CUDA_HOME"] = os.environ["CUDA_HOME"]
    os.environ["CUDA_PATH"] = str(bundled_cuda_root)
    os.environ["CUDA_HOME"] = str(bundled_cuda_root)

    # CuPy calls os.add_dll_directory(CUDA_PATH\bin) during import. The build
    # script places NVRTC sidecar DLLs here and keeps the directory present.
    try:
        bundled_cuda_bin.mkdir(exist_ok=True)
    except OSError:
        pass

    dll_dirs: list[Path] = [
        bundled_cuda_bin,
        onnxruntime_capi_dir,
        internal_dir / "tensorrt_libs",
        internal_dir / "cupy" / ".data" / "lib",
        internal_dir,
    ]

    nvidia_root = internal_dir / "nvidia"
    if nvidia_root.exists():
        dll_dirs.extend(p for p in nvidia_root.rglob("bin") if p.is_dir())
        dll_dirs.extend(p for p in nvidia_root.rglob("x64") if p.is_dir())

    for dll_dir in dll_dirs:
        _add_dll_dir(dll_dir)
    _prepend_path(dll_dirs)
