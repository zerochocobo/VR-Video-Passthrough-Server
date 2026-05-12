from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path


_QT_DLL_HANDLES: list[object] = []


def _module_dir(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.origin:
        return Path(spec.origin).resolve().parent
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    return None


def _prepare_qt_dll_paths() -> None:
    if not sys.platform.startswith("win") or not hasattr(os, "add_dll_directory"):
        return
    candidates: list[Path] = []
    plugin_roots: list[Path] = []
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent / "_internal"
        candidates.extend([base, base / "PySide6", base / "shiboken6"])
        plugin_roots.append(base / "PySide6" / "plugins")
    else:
        pyside_dir = _module_dir("PySide6.QtCore")
        shiboken_dir = _module_dir("shiboken6")
        if pyside_dir is not None:
            candidates.append(pyside_dir)
            plugin_roots.append(pyside_dir / "plugins")
        if shiboken_dir is not None:
            candidates.append(shiboken_dir)
    for path in candidates:
        if path.exists():
            _QT_DLL_HANDLES.append(os.add_dll_directory(str(path)))
    for plugins in plugin_roots:
        platforms = plugins / "platforms"
        if platforms.exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms))
        if plugins.exists():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))


_prepare_qt_dll_paths()

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from ui.i18n import system_language
from ui.main_window import MainWindow
from ui.resources import app_icon
from ui.styles import font_for_language, load_app_stylesheet


def _install_qt_message_filter() -> None:
    previous_handler = qInstallMessageHandler(None)

    def handler(mode: QtMsgType, context, message: str) -> None:
        if context.category == "qt.qpa.fonts" and "CreateFontFaceFromHDC() failed" in message:
            return
        if previous_handler is not None:
            previous_handler(mode, context, message)
            return
        prefix = {
            QtMsgType.QtDebugMsg: "Debug",
            QtMsgType.QtInfoMsg: "Info",
            QtMsgType.QtWarningMsg: "Warning",
            QtMsgType.QtCriticalMsg: "Critical",
            QtMsgType.QtFatalMsg: "Fatal",
        }.get(mode, "Log")
        print(f"{prefix}: {message}", file=sys.stderr)

    qInstallMessageHandler(handler)


def main() -> int:
    _install_qt_message_filter()
    app = QApplication(sys.argv)
    app.setApplicationName("VR Video Passthrough Server")
    app.setWindowIcon(app_icon())
    app.setFont(font_for_language(system_language()))
    app.setStyleSheet(load_app_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
