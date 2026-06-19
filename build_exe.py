from __future__ import annotations

import os
import argparse
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_NAME = "VR_Video_Passthrough_Server"
SERVER_NAME = "pt_core"
DIST_DIR = ROOT / "dist" / APP_NAME
ICON = ROOT / "resources" / "app.ico"
VENV = ROOT / ".venv"
SITE_PACKAGES = VENV / "Lib" / "site-packages"
DEFAULT_COMPARE_DIR = ROOT / f"{APP_NAME}_compare"


class BuildError(RuntimeError):
    pass


def info(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    raise BuildError(message)


def on_rm_error(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, onerror=on_rm_error)
    else:
        try:
            path.chmod(stat.S_IWRITE)
        except OSError:
            pass
        path.unlink()
    if path.exists():
        fail(f"Failed to remove {path}. Stop running packaged processes and try again.")


def copy_file(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            src_stat = src.stat()
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
                return
        except OSError:
            pass
    if dst.exists():
        try:
            dst.chmod(stat.S_IWRITE)
        except OSError:
            pass
    try:
        shutil.copy2(src, dst)
    except PermissionError as e:
        fail(f"Failed to copy locked file:\n  src: {src}\n  dst: {dst}\nClose any running packaged process and retry.\n{e}")


def copy_tree(src: Path, dst: Path, *, ignore=None) -> None:
    if not src.exists():
        fail(f"Required directory not found: {src}")
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if ignore and ignore(item, rel):
            continue
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(item, target)


def find_first_exe(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found).resolve() if found else None


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    uv = find_first_exe("uv")
    python = Path(sys.executable).resolve()

    system_root = Path(env.get("SystemRoot", r"C:\Windows"))
    path_parts = [
        system_root / "System32",
        system_root,
        system_root / "System32" / "Wbem",
        VENV / "Scripts",
    ]
    if uv:
        path_parts.append(uv.parent)
    path_parts.append(python.parent)
    for package_dir in ("PySide6", "shiboken6"):
        p = SITE_PACKAGES / package_dir
        if p.exists():
            path_parts.append(p)

    env["PATH"] = os.pathsep.join(str(p) for p in path_parts if p)
    hook_paths = [ROOT / "packaging", ROOT / "packaging" / "hooks"]
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in hook_paths] + ([old_pythonpath] if old_pythonpath else []))
    return env


def run(cmd: list[str], *, env: dict[str, str]) -> None:
    info(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    if completed.returncode:
        fail(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def python_has_pyinstaller(python: Path, env: dict[str, str]) -> bool:
    completed = subprocess.run(
        [str(python), "-c", "import PyInstaller"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def pyinstaller_cmd(env: dict[str, str]) -> list[str]:
    python = Path(sys.executable).resolve()
    if python_has_pyinstaller(python, env):
        return [str(python), "-m", "PyInstaller"]

    uv = find_first_exe("uv")
    if uv:
        completed = subprocess.run(
            [str(uv), "run", "python", "-c", "import PyInstaller"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return [str(uv), "run", "pyinstaller"]

    fail(
        "PyInstaller is not installed in the active environment.\n"
        "Install it first, for example:\n"
        "  pip install pyinstaller\n"
        "or:\n"
        "  uv add --dev pyinstaller\n"
        "or:\n"
        "  uv pip install pyinstaller"
    )


def clean() -> None:
    for path in (
        ROOT / "build",
        ROOT / "dist" / APP_NAME,
        ROOT / "dist" / SERVER_NAME,
        ROOT / f"{APP_NAME}.spec",
        ROOT / f"{SERVER_NAME}.spec",
    ):
        remove_path(path)


def build_ui(pyi: list[str], env: dict[str, str]) -> None:
    run(
        [
            *pyi,
            "--name", APP_NAME,
            "--noconsole",
            "--onedir",
            "--icon", str(ICON),
            "--additional-hooks-dir", "packaging\\hooks",
            "--runtime-hook", "packaging\\runtime_hook_cuda_dlls.py",
            "--add-data", "resources;resources",
            "--add-data", "ui\\app_metadata.json;ui",
            "--add-data", "ui\\translations;ui\\translations",
            "--add-data", "ui\\styles;ui\\styles",
            "--collect-binaries", "PySide6",
            "--collect-binaries", "shiboken6",
            "--collect-data", "PySide6",
            "--hidden-import", "PySide6.QtCore",
            "--hidden-import", "PySide6.QtGui",
            "--hidden-import", "PySide6.QtWidgets",
            "--hidden-import", "shiboken6",
            "ui\\app.py",
        ],
        env=env,
    )


def build_server(pyi: list[str], env: dict[str, str]) -> None:
    run(
        [
            *pyi,
            "--name", SERVER_NAME,
            "--console",
            "--onedir",
            "--icon", str(ICON),
            "--additional-hooks-dir", "packaging\\hooks",
            "--runtime-hook", "packaging\\runtime_hook_cuda_dlls.py",
            "--add-data", "resources;resources",
            "--hidden-import", "offline.convert",
            "--hidden-import", "offline.two_dvr",
            "--hidden-import", "tools.offline_passthrough",
            "--hidden-import", "tools.offline_alpha_passthrough",
            "--hidden-import", "tools.warmup_offline_trt",
            "--hidden-import", "tools.generate_yoloworld_person_txt_feats",
            "--collect-submodules", "offline",
            "--collect-submodules", "pipeline",
            "--collect-submodules", "http_app",
            "--collect-submodules", "dlna",
            "--collect-submodules", "utils",
            "--hidden-import", "cupy_backends.cuda._softlink",
            "--collect-all", "onnxruntime",
            "--collect-data", "osam",
            "--collect-all", "cupy",
            "--collect-all", "cupy_backends",
            "--collect-submodules", "cupy_backends",
            "--collect-all", "pynvvideocodec",
            "main.py",
        ],
        env=env,
    )


def merge_server_into_app() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    copy_file(ROOT / "dist" / SERVER_NAME / f"{SERVER_NAME}.exe", DIST_DIR / f"{SERVER_NAME}.exe")
    copy_tree(ROOT / "dist" / SERVER_NAME / "_internal", DIST_DIR / "_internal")


def require_glob(pattern: str, message: str) -> list[Path]:
    matches = list(ROOT.glob(pattern))
    if not matches:
        fail(message)
    return matches


def copy_pynv_video_codec() -> None:
    src = SITE_PACKAGES / "PyNvVideoCodec"
    if not src.exists():
        fail("PyNvVideoCodec package not found under .venv\\Lib\\site-packages.")

    def ignore(item: Path, rel: Path) -> bool:
        return item.name == "__pycache__" or item.suffix.lower() == ".pyc"

    copy_tree(src, DIST_DIR / "_internal" / "PyNvVideoCodec", ignore=ignore)
    if not list((DIST_DIR / "_internal" / "PyNvVideoCodec").glob("PyNvVideoCodec_*.pyd")):
        fail(f"Missing PyNvVideoCodec driver extension: {DIST_DIR}\\_internal\\PyNvVideoCodec\\PyNvVideoCodec_*.pyd")


def remove_stale_icu() -> None:
    for name in ("icuuc.dll", "icudt58.dll", "icuin.dll", "icuin58.dll", "icuuc58.dll", "icudata.dll"):
        p = DIST_DIR / "_internal" / name
        if p.exists():
            p.unlink()


def verify_no_duplicate_critical_dlls() -> None:
    names = (
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "pyside6.abi3.dll",
        "shiboken6.abi3.dll",
        "python312.dll",
        "python3.dll",
    )
    internal = DIST_DIR / "_internal"
    for name in names:
        matches = list(internal.rglob(name))
        if len(matches) > 1:
            fail(f"Duplicated DLL detected: {name} has {len(matches)} copies under _internal")


def prepare_resources_and_models() -> None:
    remove_path(DIST_DIR / "resources")
    remove_path(DIST_DIR / "models")
    (DIST_DIR / "models").mkdir(parents=True, exist_ok=True)


def search_roots_for_runtime_dlls() -> list[Path]:
    roots: list[Path] = []
    cudnn_bin = Path(os.environ.get("PT_CUDNN_BIN", ""))
    cudnn_root = Path(os.environ.get("PT_CUDNN_PATH", ""))
    configured_roots = []
    if str(cudnn_bin):
        configured_roots.append(cudnn_bin)
    if str(cudnn_root):
        configured_roots.extend([cudnn_root / "bin", cudnn_root])
    for p in (
        *configured_roots,
        SITE_PACKAGES,
        SITE_PACKAGES / "nvidia",
        Path(os.environ.get("CUDA_PATH", "")),
        Path(os.environ.get("CUDA_HOME", "")),
    ):
        if str(p) and p.exists() and p not in roots:
            try:
                if DIST_DIR.exists() and p.resolve().is_relative_to(DIST_DIR.resolve()):
                    continue
            except OSError:
                pass
            roots.append(p)
    return roots


def copy_named_runtime_dlls(
    names: tuple[str, ...],
    *,
    required: tuple[str, ...] = (),
    copy_to_bin: bool = True,
) -> dict[str, list[Path]]:
    copied: dict[str, list[Path]] = {name.lower(): [] for name in names}
    roots = search_roots_for_runtime_dlls()
    for root in roots:
        for pattern in names:
            for src in root.rglob(pattern):
                if not src.is_file():
                    continue
                target_dirs = [DIST_DIR / "_internal"]
                if copy_to_bin:
                    target_dirs.append(DIST_DIR / "bin")
                for target_dir in target_dirs:
                    copy_file(src, target_dir / src.name)
                copied.setdefault(pattern.lower(), []).append(src)

    for pattern in required:
        found = list((DIST_DIR / "_internal").glob(pattern)) + list((DIST_DIR / "bin").glob(pattern))
        if not found:
            roots_text = "\n  ".join(str(r) for r in roots) or "(none)"
            fail(
                f"Missing required runtime DLL {pattern}. Searched:\n  {roots_text}\n"
                "If cuDNN is installed separately, set one of:\n"
                "  set PT_CUDNN_BIN=C:\\Program Files\\NVIDIA\\CUDNN\\v9.0\\bin\n"
                "  set PT_CUDNN_PATH=C:\\Program Files\\NVIDIA\\CUDNN\\v9.0"
            )
    return copied


# pip CUDA wheels (nvidia-*-cu12) that make the package self-contained and
# Blackwell sm_120-native, independent of any system CUDA toolkit. Each wheel
# drops its DLLs under site-packages/nvidia/<component>/bin.
PIP_CUDA_NVIDIA_ROOT = SITE_PACKAGES / "nvidia"
PIP_CUDA_RUNTIME_INCLUDE = PIP_CUDA_NVIDIA_ROOT / "cuda_runtime" / "include"


def copy_cuda_headers() -> None:
    """Bundle the CUDA headers CuPy's NVRTC JIT needs (cuda_fp16.h, etc.).

    Prefer the pip nvidia-cuda-runtime-cu12 headers (match the bundled pip NVRTC
    and ship without a system CUDA toolkit); fall back to system CUDA_PATH.
    """
    include = PIP_CUDA_RUNTIME_INCLUDE
    if not include.exists():
        cuda_path = Path(os.environ.get("CUDA_PATH", ""))
        include = cuda_path / "include"
    if include.exists():
        remove_path(DIST_DIR / "include")
        copy_tree(
            include,
            DIST_DIR / "include",
            ignore=lambda item, rel: item.suffix.lower() in {".lib", ".pdb"} or item.name == "__pycache__",
        )


def copy_pip_cuda_runtime_dlls() -> None:
    """Bundle the pip CUDA runtime DLLs (cublas/cudart/cufft/nvrtc/...) so the
    frozen app uses sm_120-native 12.x libraries instead of the build machine's
    system CUDA. Copied last so these overwrite any same-named DLLs PyInstaller
    pulled from a system toolkit (e.g. cublasLt64_12.dll)."""
    if not PIP_CUDA_NVIDIA_ROOT.exists():
        fail(
            "pip CUDA wheels not found under .venv\\Lib\\site-packages\\nvidia.\n"
            "Run: uv sync  (expects cupy-cuda12x[ctk] + nvidia-cudnn-cu12)"
        )
    target = DIST_DIR / "_internal"
    bundled: list[str] = []
    for bin_dir in sorted(PIP_CUDA_NVIDIA_ROOT.glob("*/bin")):
        for dll in bin_dir.glob("*.dll"):
            copy_file(dll, target / dll.name)
            bundled.append(dll.name)
    if not bundled:
        fail(f"No DLLs found under {PIP_CUDA_NVIDIA_ROOT}\\*\\bin")
    for required in ("cudart64_12.dll", "cublasLt64_12.dll", "nvrtc64_120_0.dll"):
        if not (target / required).exists():
            fail(f"pip CUDA runtime missing {required} under {target}")
    info(f"Bundled {len(bundled)} pip CUDA runtime DLLs from {PIP_CUDA_NVIDIA_ROOT}")


def copy_cuda_auxiliary_dlls() -> None:
    copy_cuda_headers()
    # cuTENSOR is optional (only if the build machine has it); nvrtc-builtins is
    # also provided by the pip NVRTC wheel and picked up via the nvidia search root.
    copy_named_runtime_dlls(
        (
            "cuTENSOR.dll",
            "cuTENSORMg.dll",
            "nvrtc-builtins*.dll",
        )
    )


def copy_ort_cuda_ep_dependencies() -> None:
    for stale in (DIST_DIR / "bin").glob("cudnn*.dll"):
        stale.unlink()

    copied = copy_named_runtime_dlls(
        (
            "cudnn*.dll",
        ),
        required=("cudnn64_9.dll",),
        copy_to_bin=False,
    )
    for name, sources in copied.items():
        if sources:
            info(f"Bundled {name}:")
            for src in sources:
                info(f"  {src}")


def copy_ort_tensorrt_ep_dependencies() -> None:
    trt_libs = SITE_PACKAGES / "tensorrt_libs"
    if not trt_libs.exists():
        fail(
            "TensorRT wheel libraries not found under .venv\\Lib\\site-packages\\tensorrt_libs.\n"
            "Run: uv sync"
        )
    target_dir = DIST_DIR / "_internal" / "tensorrt_libs"
    excluded = {"nvinfer_10.dll"}
    for pattern in ("nvinfer_10.dll", "nvinfer_builder_*.dll"):
        for stale in target_dir.glob(pattern):
            stale.unlink()
    copy_tree(
        trt_libs,
        target_dir,
        ignore=lambda item, rel: (
            item.name == "__pycache__"
            or item.suffix.lower() in {".pyc", ".lib", ".pdb"}
            or item.name.lower() in excluded
            or item.name.lower().startswith("nvinfer_builder_")
        ),
    )
    for pattern in ("nvinfer_10.dll", "nvinfer_builder_*.dll"):
        for stale in target_dir.glob(pattern):
            stale.unlink()
    for pattern in ("nvinfer_plugin*_10.dll", "nvonnxparser*_10.dll"):
        matches = list(target_dir.glob(pattern))
        if not matches:
            fail(f"Missing TensorRT runtime DLL matching {pattern} under _internal\\tensorrt_libs.")


def copy_clip_tokenizer_cache() -> None:
    src = ROOT / "runtime_cache" / "clip_text_onnx" / "bpe_simple_vocab_16e6.txt.gz"
    if not src.exists():
        info(f"Optional CLIP BPE tokenizer cache not found: {src}")
        return
    copy_file(src, DIST_DIR / "_internal" / "runtime_cache" / "clip_text_onnx" / src.name)


def verify_base_runtime() -> None:
    (DIST_DIR / "bin").mkdir(parents=True, exist_ok=True)
    (DIST_DIR / "bin" / ".keep").write_bytes(b"")

    if not list((DIST_DIR / "_internal" / "cupy" / "_core").glob("_carray*.pyd")):
        fail(f"Missing CuPy extension: {DIST_DIR}\\_internal\\cupy\\_core\\_carray*.pyd")

    if not (DIST_DIR / "_internal" / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll").exists():
        fail("Missing onnxruntime_providers_cuda.dll.")
    if not (DIST_DIR / "_internal" / "onnxruntime" / "capi" / "onnxruntime_providers_shared.dll").exists():
        fail("Missing onnxruntime_providers_shared.dll.")


def verify_clip_tokenizer_runtime() -> None:
    osam_bpe = DIST_DIR / "_internal" / "osam" / "_models" / "yoloworld" / "clip" / "bpe_simple_vocab_16e6.txt.gz"
    fallback_bpe = DIST_DIR / "_internal" / "runtime_cache" / "clip_text_onnx" / "bpe_simple_vocab_16e6.txt.gz"
    if not osam_bpe.exists() and not fallback_bpe.exists():
        fail(
            "Missing CLIP BPE tokenizer data for SAM3 prompt tokenization. "
            f"Expected {osam_bpe} or {fallback_bpe}."
        )


def verify_cuda_auxiliary_runtime() -> None:
    if not list((DIST_DIR / "_internal").glob("nvrtc-builtins*.dll")) and not list((DIST_DIR / "bin").glob("nvrtc-builtins*.dll")):
        fail("Missing nvrtc-builtins*.dll. Install the matching CUDA Toolkit or make sure CUDA_PATH points to it.")

    if not (DIST_DIR / "include" / "cuda_fp16.h").exists():
        fail(f"Missing CUDA headers under {DIST_DIR}\\include. Install the matching CUDA Toolkit or make sure CUDA_PATH points to it.")


def verify_ort_cuda_ep_runtime() -> None:
    if not list((DIST_DIR / "_internal").glob("cudnn64_9.dll")):
        fail("Missing cudnn64_9.dll required by ONNX Runtime CUDAExecutionProvider.")


def verify_ort_tensorrt_ep_runtime() -> None:
    capi = DIST_DIR / "_internal" / "onnxruntime" / "capi"
    if not (capi / "onnxruntime_providers_tensorrt.dll").exists():
        fail("Missing onnxruntime_providers_tensorrt.dll.")
    trt_dir = DIST_DIR / "_internal" / "tensorrt_libs"
    required = (
        "nvinfer_plugin_10.dll",
        "nvonnxparser_10.dll",
    )
    for name in required:
        if not (trt_dir / name).exists():
            fail(f"Missing TensorRT DLL: {trt_dir / name}")


def run_frozen_probe(env: dict[str, str]) -> None:
    run(
        [
            str(DIST_DIR / f"{SERVER_NAME}.exe"),
            "-m",
            "cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess",
            "canary",
            "cudart",
        ],
        env=env,
    )


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(root).as_posix()
        stat_result = item.stat()
        out[rel] = (int(stat_result.st_size), int(stat_result.st_mtime))
    return out


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def compare_dist(compare_dir: Path, new_dir: Path = DIST_DIR) -> Path:
    if not compare_dir.exists():
        fail(f"Compare directory not found: {compare_dir}")
    if not new_dir.exists():
        fail(f"New distribution directory not found: {new_dir}")

    old = snapshot_tree(compare_dir)
    new = snapshot_tree(new_dir)
    old_keys = set(old)
    new_keys = set(new)
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(k for k in old_keys & new_keys if old[k][0] != new[k][0])
    same = len(old_keys & new_keys) - len(changed)

    old_size = sum(size for size, _ in old.values())
    new_size = sum(size for size, _ in new.values())
    report_lines = [
        f"compare_old: {compare_dir}",
        f"compare_new: {new_dir}",
        f"old_files:   {len(old)}",
        f"new_files:   {len(new)}",
        f"same_files:  {same}",
        f"added:       {len(added)}",
        f"removed:     {len(removed)}",
        f"size_changed:{len(changed)}",
        f"old_size:    {format_bytes(old_size)}",
        f"new_size:    {format_bytes(new_size)}",
        f"delta_size:  {format_bytes(new_size - old_size)}",
        "",
    ]

    def append_section(title: str, rows: list[str], formatter) -> None:
        report_lines.append(title)
        if not rows:
            report_lines.append("  (none)")
        else:
            for row in rows:
                report_lines.append(formatter(row))
        report_lines.append("")

    append_section(
        "ADDED",
        added,
        lambda rel: f"  + {rel} ({format_bytes(new[rel][0])})",
    )
    append_section(
        "REMOVED",
        removed,
        lambda rel: f"  - {rel} ({format_bytes(old[rel][0])})",
    )
    append_section(
        "SIZE_CHANGED",
        changed,
        lambda rel: (
            f"  * {rel} "
            f"{format_bytes(old[rel][0])} -> {format_bytes(new[rel][0])} "
            f"({format_bytes(new[rel][0] - old[rel][0])})"
        ),
    )

    output_dir = ROOT / "debug_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"build_compare_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    info("")
    info("Build package comparison:")
    for line in report_lines[:11]:
        info(line)
    info(f"full_report: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PyInstaller onedir release package.")
    parser.add_argument(
        "--compare",
        nargs="?",
        const=str(DEFAULT_COMPARE_DIR),
        default="",
        help="Compare the new dist package with an old package directory after build.",
    )
    parser.add_argument(
        "--compare-only",
        nargs="?",
        const=str(DEFAULT_COMPARE_DIR),
        default="",
        help="Only compare an old package directory with the current dist package; do not build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compare_only:
        try:
            compare_dist(Path(args.compare_only).resolve())
        except BuildError as e:
            print(f"\nCompare failed.\n{e}", file=sys.stderr)
            return 1
        return 0

    env = build_env()
    try:
        pyi = pyinstaller_cmd(env)
        info(f"Using: {' '.join(pyi)}")
        clean()
        build_ui(pyi, env)
        build_server(pyi, env)
        merge_server_into_app()
        verify_base_runtime()
        copy_pynv_video_codec()
        remove_stale_icu()
        verify_no_duplicate_critical_dlls()
        prepare_resources_and_models()
        copy_clip_tokenizer_cache()
        copy_cuda_auxiliary_dlls()
        copy_ort_cuda_ep_dependencies()
        copy_ort_tensorrt_ep_dependencies()
        # Last CUDA step: overlay the pip sm_120-native runtime DLLs so they win
        # over any system-toolkit DLLs PyInstaller/auto-collect pulled in.
        copy_pip_cuda_runtime_dlls()
        verify_base_runtime()
        verify_clip_tokenizer_runtime()
        verify_cuda_auxiliary_runtime()
        verify_ort_cuda_ep_runtime()
        verify_ort_tensorrt_ep_runtime()
        run_frozen_probe(env)
        if args.compare:
            compare_dist(Path(args.compare).resolve())
    except BuildError as e:
        print(f"\nBuild failed.\n{e}", file=sys.stderr)
        return 1

    print("\nBuild complete:")
    print(f"  {DIST_DIR / (APP_NAME + '.exe')}")
    print(f"  {DIST_DIR / (SERVER_NAME + '.exe')}")
    print(f'\nThis is an onedir build. Distribute the whole "{DIST_DIR}" folder, not only the exe.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
