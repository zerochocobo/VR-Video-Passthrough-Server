"""Offline VSR A/B validation: Real-ESRGAN-style ONNX vs the current RTX VSR.

Decodes a short clip with FFmpeg, upscales it two ways, and reports per-frame
latency, effective fps, peak VRAM and output resolution so we can decide whether
a candidate ONNX super-resolution model is worth wiring into offline.convert as
a real engine (mirroring the existing rvm_fast / matanyone2 ONNX engines).

The ONNX path reuses the project's provider-chain convention
(pipeline.demosaic._onnx_providers): TensorRT fp16 + engine cache, then CUDA,
then CPU. The RTX baseline calls the real offline.rtx_vsr.run_rtx_vsr so the
comparison is against production output, not a re-implementation.

Nothing here is downloaded automatically. Point --model at a local .onnx.

Examples
--------
# Candidate ONNX only, 10s clip, native x4:
python tools/vsr_onnx_compare.py G:/clips/sample.mp4 \
    --model models/realesr/realesr-general-x4v3.onnx --duration 10

# A/B against the current RTX VSR at a matched 2160p target:
python tools/vsr_onnx_compare.py G:/clips/sample.mp4 \
    --model models/realesr/realesr-general-x4v3.onnx --duration 10 \
    --target-height 2160 --rtx-baseline
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np

import config
from utils.runtime_dll_paths import apply_runtime_dll_paths
from utils.subprocess_hidden import hidden_subprocess_kwargs

# ORT's CUDA/TensorRT EPs need the pip NVRTC, TensorRT and CUDA/cuDNN DLL dirs on
# the Windows loader search path (same as the production offline entrypoints).
apply_runtime_dll_paths()


def _add_pip_nvidia_dll_dirs() -> None:
    """Add .venv/.../nvidia/*/bin to the loader path so ORT 1.25 finds the pip
    CUDA 12 / cuDNN 9 runtime (cublasLt64_12.dll, cudnn*_9.dll, ...)."""
    if not sys.platform.startswith("win"):
        return
    nvidia_root = config.ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
    for bin_dir in sorted(nvidia_root.glob("*/bin")):
        try:
            os.add_dll_directory(str(bin_dir))
        except (AttributeError, OSError):
            pass


_add_pip_nvidia_dll_dirs()


# --------------------------------------------------------------------------- #
# ONNX session (matches pipeline.demosaic conventions)
# --------------------------------------------------------------------------- #
def _trt_cache_dir(name: str) -> Path:
    path = config.ROOT / "runtime_cache" / "vsr_onnx_trt" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _onnx_providers(provider: str, cache_name: str) -> list:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chain: list = []
    if provider == "trt" and "TensorrtExecutionProvider" in available:
        chain.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(_trt_cache_dir(cache_name)),
                "trt_timing_cache_enable": True,
            },
        ))
    if provider in ("trt", "cuda") and "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


def _make_session(model_path: Path, provider: str, cache_name: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=_onnx_providers(provider, cache_name))


# --------------------------------------------------------------------------- #
# VRAM sampling
# --------------------------------------------------------------------------- #
class VramSampler:
    """Polls nvidia-smi in a thread and keeps the peak used-MB seen."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_mb = 0.0
        self._exe = shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> float:
        if not self._exe:
            return 0.0
        try:
            out = subprocess.check_output(
                [self._exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs(),
            )
            return float(out.strip().splitlines()[0].strip())
        except Exception:
            return 0.0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self._sample_once())

    def __enter__(self) -> "VramSampler":
        self.peak_mb = self._sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# Tiled ONNX super-resolution
# --------------------------------------------------------------------------- #
class OnnxUpscaler:
    """Real-ESRGAN-style ONNX SR: NCHW float32 RGB in [0,1], x`scale` out.

    Handles the common SRVGGNetCompact / RRDBNet I/O layout. Tiling with overlap
    keeps VRAM bounded on <=16GB cards; set --tile 0 to run whole frames.
    """

    def __init__(self, model_path: Path, provider: str, scale: int, tile: int, overlap: int):
        self.sess = _make_session(model_path, provider, model_path.stem)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.scale = scale
        self.tile = tile
        self.overlap = overlap

    @property
    def providers(self) -> list:
        return self.sess.get_providers()

    def _infer(self, chw: np.ndarray) -> np.ndarray:
        out = self.sess.run([self.out_name], {self.in_name: chw[None]})[0][0]
        return np.clip(out, 0.0, 1.0)

    def upscale(self, rgb_u8: np.ndarray) -> np.ndarray:
        """rgb_u8: HxWx3 uint8 -> (scale*H)x(scale*W)x3 uint8."""
        h, w = rgb_u8.shape[:2]
        chw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1).astype(np.float32) / 255.0)
        s = self.scale
        if self.tile <= 0 or (h <= self.tile and w <= self.tile):
            out = self._infer(chw)
        else:
            out = np.zeros((3, h * s, w * s), dtype=np.float32)
            step = self.tile - self.overlap
            for y in range(0, h, step):
                for x in range(0, w, step):
                    y2, x2 = min(y + self.tile, h), min(x + self.tile, w)
                    y1, x1 = max(0, y2 - self.tile), max(0, x2 - self.tile)
                    tile_out = self._infer(np.ascontiguousarray(chw[:, y1:y2, x1:x2]))
                    # trim the overlap margins before pasting to avoid seams
                    ty1 = 0 if y1 == 0 else self.overlap // 2
                    tx1 = 0 if x1 == 0 else self.overlap // 2
                    oy1, ox1 = (y1 + ty1) * s, (x1 + tx1) * s
                    out[:, oy1:y2 * s, ox1:x2 * s] = tile_out[:, ty1 * s:, tx1 * s:]
        return (out.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# FFmpeg helpers
# --------------------------------------------------------------------------- #
def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _decode_cmd(src: Path, start: float, duration: float) -> list[str]:
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(src)]
    if duration > 0:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return cmd


def _encode_cmd(out: Path, w: int, h: int, fps: float, preset: str, cq: int) -> list[str]:
    return [
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.9f}", "-i", "-",
        "-map", "0:v:0", "-c:v", "hevc_nvenc", "-preset", preset, "-cq", str(cq),
        str(out),
    ]


def _probe(src: Path) -> tuple[int, int, float]:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    out = subprocess.check_output(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", str(src)],
        text=True, **hidden_subprocess_kwargs(),
    ).strip().split(",")
    w, h = int(out[0]), int(out[1])
    num, _, den = out[2].partition("/")
    fps = float(num) / float(den) if den and float(den) else float(num)
    return w, h, (fps if fps > 0 else 30.0)


def run_onnx(src: Path, out: Path, up: OnnxUpscaler, *, start: float, duration: float,
             target_height: int, preset: str, cq: int) -> dict:
    w, h, fps = _probe(src)
    out_w, out_h = w * up.scale, h * up.scale
    resize_to = None
    if target_height > 0 and out_h != target_height:
        rw = int(round(out_w * target_height / out_h)) & ~1
        resize_to = (rw, target_height)
        enc_w, enc_h = resize_to
    else:
        enc_w, enc_h = out_w, out_h

    print(f"[onnx] providers={up.providers}", flush=True)
    print(f"[onnx] in={w}x{h} native={out_w}x{out_h} enc={enc_w}x{enc_h} "
          f"scale=x{up.scale} tile={up.tile} fps={fps:.3f}", flush=True)

    dec = subprocess.Popen(_decode_cmd(src, start, duration), stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs())
    enc = subprocess.Popen(_encode_cmd(out, enc_w, enc_h, fps, preset, cq),
                           stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           **hidden_subprocess_kwargs())
    frame_bytes = w * h * 3
    frames, infer_s, infer_frames = 0, 0.0, 0
    t0 = time.monotonic()
    with VramSampler() as vram:
        try:
            while True:
                raw = dec.stdout.read(frame_bytes)
                if not raw or len(raw) != frame_bytes:
                    break
                rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                ti = time.monotonic()
                sr = up.upscale(rgb)
                dt = time.monotonic() - ti
                # exclude the first frame: it carries one-time TRT engine build /
                # CUDA kernel warmup and would skew steady-state fps.
                if frames > 0:
                    infer_s += dt
                    infer_frames += 1
                if resize_to is not None:
                    import cv2
                    sr = cv2.resize(sr, resize_to, interpolation=cv2.INTER_AREA)
                enc.stdin.write(sr.tobytes())
                frames += 1
        finally:
            if dec.stdout:
                dec.stdout.close()
            dec.wait(timeout=30)
            if enc.stdin:
                enc.stdin.close()
            enc.wait(timeout=120)
    wall = max(1e-6, time.monotonic() - t0)
    return {
        "label": "onnx", "frames": frames, "wall_s": wall,
        "infer_fps": infer_frames / infer_s if infer_s else 0.0,
        "wall_fps": frames / wall, "peak_vram_mb": vram.peak_mb,
        "out": str(out), "out_res": f"{enc_w}x{enc_h}",
    }


def run_rtx_baseline(src: Path, out: Path, *, start: float, duration: float,
                     target_height: int, quality: int, preset: str, cq: int) -> dict:
    from utils.video_metadata import probe_video_metadata
    from offline.rtx_vsr import run_rtx_vsr

    meta = probe_video_metadata(src)
    t0 = time.monotonic()
    with VramSampler() as vram:
        rc = run_rtx_vsr(src, out, meta, start=start, duration=duration,
                         target_height=target_height, quality=quality,
                         preset=preset, cq=cq)
    wall = max(1e-6, time.monotonic() - t0)
    frames = max(0, int(round((duration or float(meta.timing.duration or 0.0))
                              * (meta.timing.source_fps or 30.0))))
    return {
        "label": "rtx_vsr", "rc": rc, "frames": frames, "wall_s": wall,
        "infer_fps": 0.0, "wall_fps": frames / wall if frames else 0.0,
        "peak_vram_mb": vram.peak_mb, "out": str(out) if rc == 0 else "(failed)",
        "out_res": f"h={target_height}",
    }


def _report(rows: list[dict]) -> None:
    print("\n================ VSR A/B RESULT ================", flush=True)
    hdr = f"{'engine':<10}{'frames':>8}{'wall_s':>10}{'wall_fps':>10}{'infer_fps':>11}{'peakVRAM_MB':>13}  out_res"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in rows:
        print(f"{r['label']:<10}{r['frames']:>8}{r['wall_s']:>10.2f}"
              f"{r['wall_fps']:>10.2f}{r['infer_fps']:>11.2f}"
              f"{r['peak_vram_mb']:>13.0f}  {r['out_res']}", flush=True)
    for r in rows:
        print(f"[{r['label']}] out={r['out']}", flush=True)
    print("===============================================\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline VSR A/B: ONNX vs RTX VSR")
    ap.add_argument("video")
    ap.add_argument("--model", type=Path, help="candidate .onnx (Real-ESRGAN-style)")
    ap.add_argument("--provider", choices=["trt", "cuda", "cpu"], default="cuda")
    ap.add_argument("--scale", type=int, default=4, help="model native upscale factor")
    ap.add_argument("--tile", type=int, default=512, help="tile size in px (0=whole frame)")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--target-height", type=int, default=0,
                    help="resize SR output to this height (match RTX target for A/B)")
    ap.add_argument("--preset", default="p4")
    ap.add_argument("--cq", type=int, default=19)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rtx-baseline", action="store_true",
                    help="also run offline.rtx_vsr for a real A/B comparison")
    ap.add_argument("--rtx-quality", type=int, choices=[2, 3, 4],
                    default=int(getattr(config, "RTX_VSR_QUALITY", 3)))
    args = ap.parse_args(argv)

    src = Path(args.video).resolve()
    if not src.exists():
        print(f"[error] source not found: {src}", flush=True)
        return 2
    out_dir = (args.out_dir or src.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.model:
        if not args.model.exists():
            print(f"[error] --model not found: {args.model}", flush=True)
            return 2
        up = OnnxUpscaler(args.model, args.provider, args.scale, args.tile, args.overlap)
        out = out_dir / f"{src.stem}__onnx_{args.model.stem}.mp4"
        rows.append(run_onnx(src, out, up, start=args.start, duration=args.duration,
                             target_height=args.target_height, preset=args.preset, cq=args.cq))

    if args.rtx_baseline:
        th = args.target_height or int(getattr(config, "RTX_VSR_TARGET_HEIGHT", 2160))
        out = out_dir / f"{src.stem}__rtxvsr_{th}.mp4"
        rows.append(run_rtx_baseline(src, out, start=args.start, duration=args.duration,
                                     target_height=th, quality=args.rtx_quality,
                                     preset=args.preset, cq=args.cq))

    if not rows:
        print("[error] nothing to do: pass --model and/or --rtx-baseline", flush=True)
        return 2
    _report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
