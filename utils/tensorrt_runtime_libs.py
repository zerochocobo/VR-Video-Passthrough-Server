from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import config
from utils.gpu_requirements import detect_nvidia_gpu_requirement


TENSORRT_CU12_LIBS_WHL_URL = (
    "https://pypi.nvidia.com/tensorrt-cu12-libs/"
    "tensorrt_cu12_libs-10.16.1.11-py3-none-win_amd64.whl"
    "#sha256=ed0d4536f1322aa2f76da54feb3f9bd2d14d89e4325cef02165a98f3a2c1a493"
)
TENSORRT_CU12_LIBS_WHL_SHA256 = "ed0d4536f1322aa2f76da54feb3f9bd2d14d89e4325cef02165a98f3a2c1a493"
TENSORRT_CU12_LIBS_WHL_SIZE_BYTES = 2_206_065_494
STANDARD_TRT_DLLS = ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll")


@dataclass(frozen=True)
class TensorRTRuntimeLibStatus:
    frozen: bool
    lib_dir: Path
    gpu_name: str = ""
    compute_capability: str = ""
    sm_dll: str = ""
    missing_standard: tuple[str, ...] = ()
    missing_sm: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.missing_standard and not self.missing_sm

    @property
    def missing(self) -> tuple[str, ...]:
        return (*self.missing_standard, *self.missing_sm)


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def tensorrt_lib_dir() -> Path:
    if is_frozen_app():
        return Path(sys.executable).resolve().parent / "_internal" / "tensorrt_libs"
    return config.ROOT / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"


def sm_resource_name(compute_capability: str) -> str:
    digits = "".join(ch for ch in str(compute_capability or "") if ch.isdigit())
    return f"nvinfer_builder_resource_sm{digits}_10.dll" if digits else ""


def check_tensorrt_runtime_libs(lib_dir: Path | None = None) -> TensorRTRuntimeLibStatus:
    directory = Path(lib_dir or tensorrt_lib_dir())
    gpu = detect_nvidia_gpu_requirement()
    sm_dll = sm_resource_name(gpu.compute_capability)
    missing_standard = tuple(name for name in STANDARD_TRT_DLLS if not (directory / name).is_file())
    missing_sm = tuple([sm_dll] if sm_dll and not (directory / sm_dll).is_file() else [])
    return TensorRTRuntimeLibStatus(
        frozen=is_frozen_app(),
        lib_dir=directory,
        gpu_name=gpu.name,
        compute_capability=gpu.compute_capability,
        sm_dll=sm_dll,
        missing_standard=missing_standard,
        missing_sm=missing_sm,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tensorrt_wheel(
    destination: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = TENSORRT_CU12_LIBS_WHL_URL.split("#", 1)[0]
    request = urllib.request.Request(url, headers={"User-Agent": "PTMediaServer/1.0"})
    received = 0
    with urllib.request.urlopen(request, timeout=30) as response:
        try:
            total = int(response.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            total = 0
        if total <= 0:
            total = TENSORRT_CU12_LIBS_WHL_SIZE_BYTES
        if progress is not None:
            progress(0, total)
        with destination.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                if progress is not None:
                    progress(received, total)
    actual = _sha256(destination)
    if actual.lower() != TENSORRT_CU12_LIBS_WHL_SHA256:
        try:
            destination.unlink()
        except OSError:
            pass
        raise RuntimeError(f"TensorRT wheel SHA256 mismatch: {actual}")
    return destination


def extract_required_tensorrt_libs(whl_path: Path, lib_dir: Path | None = None) -> list[Path]:
    status = check_tensorrt_runtime_libs(lib_dir)
    names = set(status.missing)
    target_dir = status.lib_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    found: set[str] = set()
    if not names:
        return extracted
    with zipfile.ZipFile(whl_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename).name
            if name not in names:
                continue
            target = target_dir / name
            with archive.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            found.add(name)
            extracted.append(target)
    missing = tuple(sorted(names - found))
    if missing:
        raise RuntimeError(f"TensorRT wheel did not contain required DLLs: {', '.join(missing)}")
    if sys.platform.startswith("win"):
        try:
            os.add_dll_directory(str(target_dir))
        except (AttributeError, OSError):
            pass
    return extracted


def download_and_install_tensorrt_libs(
    progress: Callable[[int, int], None] | None = None,
    lib_dir: Path | None = None,
) -> TensorRTRuntimeLibStatus:
    temp_dir = Path(tempfile.gettempdir()) / "ptmediaserver"
    whl = temp_dir / "tensorrt_cu12_libs-10.16.1.11-py3-none-win_amd64.whl"
    download_tensorrt_wheel(whl, progress=progress)
    extract_required_tensorrt_libs(whl, lib_dir=lib_dir)
    return check_tensorrt_runtime_libs(lib_dir)
