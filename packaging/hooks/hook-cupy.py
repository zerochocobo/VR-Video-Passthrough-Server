"""PyInstaller hook for CuPy wheels.

CuPy 14 imports several Cython extension modules dynamically from package
initializers. PyInstaller's default analysis can miss some of these .pyd files
even with --collect-all cupy, which breaks frozen imports with errors such as
``No module named 'cupy._core._carray'``.
"""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, get_package_paths


def _collect_package_pyds(package_name: str) -> list[tuple[str, str]]:
    try:
        _, package_dir = get_package_paths(package_name)
    except Exception:
        return []
    root = Path(package_dir)
    found: list[tuple[str, str]] = []
    for path in root.rglob("*.pyd"):
        rel_parent = path.parent.relative_to(root.parent)
        found.append((str(path), str(rel_parent)))
    return found


hiddenimports = (
    collect_submodules("cupy_backends")
    + collect_submodules("cuda.pathfinder")
    + [
        "cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess",
    ]
)
binaries = (
    collect_dynamic_libs("cupy")
    + collect_dynamic_libs("cupy_backends")
    + _collect_package_pyds("cupy")
    + _collect_package_pyds("cupy_backends")
)
datas = []
