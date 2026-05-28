"""Qt runtime path helpers for direct source and test execution."""
from __future__ import annotations

import os
import site
from pathlib import Path

_DLL_HANDLES = []
_CONFIGURED = False


def configure_qt_runtime_paths() -> None:
    """Add PySide/shiboken DLL and plugin directories when running from source."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    if hasattr(os, "add_dll_directory"):
        for site_dir in site.getsitepackages():
            base = Path(site_dir)
            for dll_dir in (base / "PySide6", base / "shiboken6"):
                if dll_dir.exists():
                    _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            plugins = base / "PySide6" / "plugins"
            platforms = plugins / "platforms"
            if platforms.exists():
                os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms))
            if plugins.exists():
                os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
