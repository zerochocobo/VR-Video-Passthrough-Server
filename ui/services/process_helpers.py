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


def base_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra:
        env.update(extra)
    return env
