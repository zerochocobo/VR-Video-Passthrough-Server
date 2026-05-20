from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.services.process_helpers import base_environment  # noqa: E402

os.environ.update(base_environment())

import config  # noqa: E402
from utils.gpu_runtime_cache import configure_gpu_runtime_cache  # noqa: E402

configure_gpu_runtime_cache()

config.RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)


class FixedTemporaryDirectory:
    def __init__(self, *args, **kwargs):
        self.name = str(config.RUNTIME_TMP_DIR)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        return False

    def cleanup(self):
        return None


tempfile.TemporaryDirectory = FixedTemporaryDirectory

import onnxruntime as ort  # noqa: E402

if "CUDAExecutionProvider" not in set(ort.get_available_providers()):
    raise RuntimeError(f"CUDAExecutionProvider unavailable: {ort.get_available_providers()}")

from pipeline.matting import get_matter  # noqa: E402
from pipeline.pynv_stream import PyNvPassthroughStream  # noqa: E402
from utils.video_metadata import probe_video_metadata  # noqa: E402


async def _capture(args: argparse.Namespace) -> None:
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    matter = get_matter()
    active = matter.sess.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"CUDAExecutionProvider not active for Matter session: {active}")
    print(f"[capture] active ORT providers={active}")
    meta = probe_video_metadata(src)
    stream = PyNvPassthroughStream(
        src,
        float(args.start),
        matter,
        meta,
        container="mpegts",
        max_fps=float(args.max_fps),
        audio_mode_override="off",
        output_mode="alpha",
    )
    written = 0
    chunks = 0
    with out.open("wb") as fh:
        async for chunk in stream.iter_bytes():
            fh.write(chunk)
            written += len(chunk)
            chunks += 1
            if written >= int(args.bytes):
                break
    stream.close()
    print(f"[capture] wrote={written} chunks={chunks} out={out}")


def _decode_first_frame(ts_path: Path, png_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(ts_path),
        "-frames:v",
        "1",
        str(png_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"[capture] frame={png_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("--out", default=str(config.ROOT / "debug_output" / "realtime_alpha_capture.ts"))
    parser.add_argument("--frame", default=str(config.ROOT / "debug_output" / "realtime_alpha_capture.png"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--max-fps", type=float, default=30.0)
    parser.add_argument("--bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    asyncio.run(_capture(args))
    _decode_first_frame(Path(args.out).resolve(), Path(args.frame).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
