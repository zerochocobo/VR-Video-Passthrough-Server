"""
Probe the local GPU video stack before attempting a zero-copy rewrite.

This tool intentionally does not touch the production passthrough path. It checks
which Python/NVIDIA video bindings are importable and whether FFmpeg can decode
with CUDA hardware frames on the current machine.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", e.stderr or f"timeout after {timeout}s"


def _module_status(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "missing"
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return f"found but import failed: {type(e).__name__}: {e}"
    version = getattr(mod, "__version__", "")
    return f"ok{f' ({version})' if version else ''}"


def _print_module_probe() -> None:
    print("[python modules]")
    for name in (
        "cupy",
        "onnxruntime",
        "PyNvVideoCodec",
        "pynvvideo",
        "PyNvCodec",
        "PytorchNvCodec",
        "nvidia",
    ):
        print(f"{name:18s} { _module_status(name) }")

    try:
        import cupy as cp

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        print(f"cuda_device        {dev.id}: {name}")
        print(f"cuda_runtime       {cp.cuda.runtime.runtimeGetVersion()}")
    except Exception as e:
        print(f"cuda_device        unavailable: {type(e).__name__}: {e}")

    try:
        import onnxruntime as ort

        print(f"ort_providers      {ort.get_available_providers()}")
    except Exception as e:
        print(f"ort_providers      unavailable: {type(e).__name__}: {e}")


def _print_ffmpeg_probe(src: Path | None) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    print("\n[ffmpeg]")
    print(f"ffmpeg             {ffmpeg}")
    print(f"ffprobe            {ffprobe}")

    rc, out, err = _run([ffmpeg, "-hide_banner", "-hwaccels"], timeout=10)
    hwaccels = (out + "\n" + err).strip().replace("\r", "")
    print(f"hwaccels_rc        {rc}")
    print(hwaccels)

    rc, out, err = _run([ffmpeg, "-hide_banner", "-decoders"], timeout=10)
    decoders_text = out + "\n" + err
    cuvid = sorted(
        line.strip()
        for line in decoders_text.splitlines()
        if "_cuvid" in line
    )
    print("\n[cuvid decoders]")
    if cuvid:
        for line in cuvid[:32]:
            print(line)
        if len(cuvid) > 32:
            print(f"... {len(cuvid) - 32} more")
    else:
        print("none found")

    if src is None:
        return

    print("\n[cuda decode smoke]")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "info",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(src),
        "-an",
        "-sn",
        "-frames:v",
        "30",
        "-f",
        "null",
        "-",
    ]
    print("cmd                " + subprocess.list2cmdline(cmd))
    rc, out, err = _run(cmd, timeout=60)
    tail = (out + "\n" + err).replace("\r", "").strip().splitlines()[-20:]
    print(f"rc                 {rc}")
    for line in tail:
        print(line)


def _resolve_video(value: str) -> Path | None:
    if not value:
        candidates = sorted(config.VIDEO_DIR.glob("*.mp4"))
        return candidates[0] if candidates else None
    p = Path(value)
    if not p.is_absolute():
        p = config.VIDEO_DIR / p
    return p.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe GPU video decode/encode prerequisites.")
    parser.add_argument("video", nargs="?", default="", help="optional video file under videos/ or absolute path")
    args = parser.parse_args()

    src = _resolve_video(args.video)
    _print_module_probe()
    _print_ffmpeg_probe(src if src and src.exists() else None)
    if src is None:
        print("\n[video] no input video found; skipped CUDA decode smoke test")
    elif not src.exists():
        print(f"\n[video] not found: {src}; skipped CUDA decode smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
