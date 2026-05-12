"""Build-time DLL path bootstrap for PyInstaller isolated child processes.

PySide6-Essentials keeps Qt DLLs under PySide6 and VC/shiboken DLLs under
shiboken6. PyInstaller's Qt hook imports PySide6.QtCore in a clean child
process before our application entry point can call os.add_dll_directory().
Putting this directory on PYTHONPATH during packaging makes that child import
sitecustomize first and gives the Windows loader the package DLL directories.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


_DLL_HANDLES: list[object] = []


def _module_dir(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.origin:
        return Path(spec.origin).resolve().parent
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    return None


if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
    for module_name in ("PySide6.QtCore", "shiboken6"):
        module_dir = _module_dir(module_name)
        if module_dir is not None and module_dir.exists():
            _DLL_HANDLES.append(os.add_dll_directory(str(module_dir)))
