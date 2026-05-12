from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies, pyside6_library_info


def collect(module_file: str):
    info = pyside6_library_info
    _ = info.location
    package_dir = Path(info.package_location).resolve()
    if (package_dir / "plugins").is_dir():
        loc = info.location
        loc["PrefixPath"] = str(package_dir)
        loc["BinariesPath"] = str(package_dir)
        loc["LibrariesPath"] = str(package_dir)
        loc["LibraryExecutablesPath"] = str(package_dir)
        loc["PluginsPath"] = str(package_dir / "plugins")
        loc["QmlImportsPath"] = str(package_dir / "qml")
        loc["TranslationsPath"] = str(package_dir / "translations")
        loc["DataPath"] = str(package_dir)
        info.qt_inside_package = True
        info.qt_lib_dir = package_dir
    return add_qt6_dependencies(module_file)
