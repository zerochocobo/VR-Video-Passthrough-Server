from __future__ import annotations

from pathlib import Path

from ui.qt_runtime import configure_qt_runtime_paths

configure_qt_runtime_paths()

from PySide6.QtGui import QFont


ROOT = Path(__file__).resolve().parent
STYLE_PATH = ROOT / "app.qss"


def load_app_stylesheet() -> str:
    return STYLE_PATH.read_text(encoding="utf-8-sig")


def font_for_language(language: str) -> QFont:
    font = QFont()
    font.setPointSize(11)
    if language == "zh_CN":
        font.setFamilies(["Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei"])
    elif language == "ja_JP":
        font.setFamilies(["Segoe UI", "Yu Gothic UI", "Meiryo"])
    else:
        font.setFamilies(["Segoe UI"])
    return font
