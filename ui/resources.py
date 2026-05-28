from __future__ import annotations

from pathlib import Path
import sys

from ui.qt_runtime import configure_qt_runtime_paths

configure_qt_runtime_paths()

from PySide6.QtGui import QIcon


ROOT = Path(sys.executable).resolve().parent / "_internal" if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "resources"
APP_ICON_PATH = RESOURCE_DIR / "app.ico"
SWITCH_OFF_IMAGE_PATH = RESOURCE_DIR / "switch_off.png"
SWITCH_ON_IMAGE_PATH = RESOURCE_DIR / "switch_on.png"
PLAYER_SUPPORT_PATH = RESOURCE_DIR / "player_support.json"


def app_icon() -> QIcon:
    return QIcon(str(APP_ICON_PATH))
