from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]


def python_executable() -> str:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def server_command() -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
        for name in ("pt_core.exe", "vr_pt_server.exe", "VR_Video_Passthrough_Server.exe"):
            exe = base / name
            if exe.exists():
                return str(exe), []
    return python_executable(), [str(ROOT / "main.py")]


def offline_command() -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        server, args = server_command()
        return server, [*args, "offline"]
    return python_executable(), [str(ROOT / "offline" / "convert.py")]


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


def _apply_development_cuda_dll_paths(env: dict[str, str]) -> None:
    if getattr(sys, "frozen", False) or not sys.platform.startswith("win"):
        return
    candidates: list[Path] = []
    cudnn_bin = env.get("PT_CUDNN_BIN") or os.environ.get("PT_CUDNN_BIN")
    if cudnn_bin:
        candidates.append(Path(cudnn_bin))
    cuda_path = env.get("CUDA_PATH") or os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    cuda_home = env.get("CUDA_HOME") or os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin")
    _prepend_path_value(env, candidates)


def base_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra:
        env.update(extra)
    _apply_development_cuda_dll_paths(env)
    return env
