"""Offline mosaic-restoration (RM) converter.

Sibling of ``offline/two_dvr.py``. Runs the same detector + chunk
restoration models used by the realtime [RM] DLNA path, but writes a finished
file so the restoration result can be reviewed without a DLNA player.

Pipeline (ffmpeg decode -> GPU restore -> ffmpeg encode):
    ffmpeg (rgb24, optional -ss/-t) -> 8-frame temporal chunk -> GpuRmProcessor
    -> ffmpeg libx-encode muxing the source audio for the same clip.

TensorRT engine cache is built before conversion (``build-trt`` subcommand, or
automatically on first run) so the offline path never silently falls back to a
slow provider.

    uv run python offline/demosaic_offline.py single in.mp4 --start 0 --duration 60
    uv run python offline/demosaic_offline.py batch  DIR --recursive
    uv run python offline/demosaic_offline.py build-trt
"""
from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.runtime_dll_paths import apply_runtime_dll_paths

apply_runtime_dll_paths()

import config
from utils.subprocess_hidden import hidden_subprocess_kwargs

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}
OUTPUT_SUFFIX = "_restored"


def log(msg: str) -> None:
    print(msg, flush=True)


# --- output naming ----------------------------------------------------------


def _time_tag(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}{m:02d}{s:02d}"


def output_path(src: Path, out_dir: Path | None, start: float, duration: float) -> Path:
    parent = out_dir if out_dir else src.parent
    if duration > 0:
        seg = f"_S{_time_tag(start)}_E{_time_tag(start + duration)}"
    elif start > 0:
        seg = f"_S{_time_tag(start)}"
    else:
        seg = ""
    return parent / f"{src.stem}{seg}{OUTPUT_SUFFIX}.mp4"


# --- ffmpeg muxer (audio only) ----------------------------------------------


def _mux_proc(src: Path, out: Path, fps: float, start: float, duration: float):
    """Mux our NVENC HEVC elementary stream (stdin) with the source audio,
    trimmed to the same [start, start+duration) window, into an mp4. Video is
    copied (already encoded on the GPU); only audio is transcoded."""
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "hevc", "-r", f"{fps:.6f}", "-i", "-"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    # No -shortest: the audio is already trimmed by -t and matches the encoded
    # video length; -shortest with a raw-hevc pipe (no container duration) can
    # end the mux before any audio is written (audio:0KiB).
    cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


# --- per-clip pipeline (full GPU: NVDEC -> cupy -> RM -> NVENC) --------------


def _run_clip(processor_factory, src: Path, out: Path, start: float, duration: float,
              conf: float) -> int:
    import cupy as cp
    import PyNvVideoCodec as nvc
    from offline.two_dvr_pynv import _NV12_RGB_KERNELS
    from pipeline.demosaic import WINDOW
    from pipeline.pynv_io import GpuNv12AppFrame, GpuP016Frame, PyNvSimpleDecoder, PyNvThreadedSerialDecoder
    from pipeline.pynv_stream import PYNV_BACKEND_LABEL, _pynv_encoder_kwargs
    from utils.bitrate_estimator import effective_default_bitrate
    from utils.video_metadata import probe_video_metadata

    meta = probe_video_metadata(src)
    bit_depth = int(meta.codec.bit_depth if meta.codec and meta.codec.bit_depth > 0 else 8)
    shift_bits = int(config.PASSTHROUGH_PYNV_10BIT_SHIFT)

    meta_dec = PyNvSimpleDecoder(src, bit_depth=bit_depth)
    info = meta_dec.info
    dec_len = len(meta_dec)
    width, height = int(info.width), int(info.height)
    fps = float(info.fps or (meta.timing.source_fps if meta.timing else 0.0) or 30.0)
    meta_dec.stop()

    start_out = min(max(0, int(round(start * fps))), max(0, dec_len - 1))
    if duration > 0:
        end_idx = min(dec_len, start_out + int(round(duration * fps)))
    else:
        end_idx = dec_len
    end_idx = max(start_out + 1, end_idx)
    last_src = dec_len - 1
    s0 = start_out
    log(f"{src.name}: {width}x{height}@{fps:.3f} frames[{start_out},{end_idx}) -> {out.name}")

    # Build the width-specific detector (1280 for 8K) up front so its one-time
    # TensorRT build/load happens here rather than stalling the first chunk.
    processor = processor_factory()
    processor.prewarm_for_width(width)

    dec = PyNvThreadedSerialDecoder(
        src, bit_depth=bit_depth, start_frame=s0,
        batch_size=config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
        buffer_size=config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
        info=info, num_frames=dec_len,
    )
    bitrate = str(effective_default_bitrate(src, PYNV_BACKEND_LABEL).bps)
    enc = nvc.CreateEncoder(width, height, "NV12", False,
                            **_pynv_encoder_kwargs(bitrate=bitrate, fps=f"{fps:.6f}"))
    mux = _mux_proc(src, out, fps, start, duration)
    # Drain mux stderr concurrently so a full pipe buffer can never block ffmpeg
    # (which would stall its stdin reads and deadlock our bitstream writes).
    mux_err_chunks: list[bytes] = []

    def _drain_stderr() -> None:
        try:
            for line in iter(mux.stderr.readline, b""):
                mux_err_chunks.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, name="rm-mux-stderr", daemon=True)
    stderr_thread.start()

    mod = cp.RawModule(code=_NV12_RGB_KERNELS)
    k_to_rgb = mod.get_function("nv12_to_rgb")
    k_p016_to_rgb = mod.get_function("p016_to_rgb")
    k_to_nv12 = mod.get_function("rgb_to_nv12")
    bx = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    grid_out = ((width + 15) // 16, (height + 15) // 16, 1)

    cache: dict[int, "cp.ndarray"] = {}

    def rgb_at(idx: int):
        idx = max(0, min(last_src, idx))
        cached = cache.get(idx)
        if cached is None:
            frame = dec.frame_at(idx).owned_copy()
            cached = cp.empty((height, width, 3), cp.uint8)
            if isinstance(frame, GpuP016Frame):   # 10-bit -> 8-bit RGB
                y_g = frame.y.as_cupy(cp.uint16).reshape(height, width)
                uv_g = frame.uv.as_cupy(cp.uint16).reshape(height // 2, width)
                k_p016_to_rgb(grid, bx, (y_g, uv_g, cached, np.int32(width), np.int32(height), np.int32(shift_bits)))
            else:
                y_g = frame.y.as_cupy(cp.uint8).reshape(height, width)
                uv_g = frame.uv.as_cupy(cp.uint8).reshape(height // 2, width)
                k_to_rgb(grid, bx, (y_g, uv_g, cached, np.int32(width), np.int32(height)))
            cache[idx] = cached
        return cached

    total = end_idx - start_out
    t0 = time.perf_counter()

    # NVENC + pipe I/O run on their own thread so they overlap the next chunk's
    # decode + restore (NVENC is a separate HW engine from the CUDA cores). The
    # main thread only launches the RGB->NV12 conversion and hands over a ready
    # GPU buffer guarded by a CUDA event; the FIFO queue preserves frame order,
    # so recurrent state and mux output stay identical to the serial path.
    enc_q: "queue.Queue" = queue.Queue(maxsize=WINDOW)
    enc_state = {"produced": 0}
    enc_error: list[BaseException] = []

    def _encode_worker() -> None:
        cp.cuda.Device().use()
        try:
            while True:
                item = enc_q.get()
                if item is None:
                    break
                nv12_g, ev = item
                ev.synchronize()                 # conversion done before NVENC reads
                flags = 0
                if enc_state["produced"] == 0:
                    flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
                bitstream = enc.Encode(GpuNv12AppFrame(nv12_g, width, height), flags)
                if bitstream:
                    mux.stdin.write(bitstream)
                enc_state["produced"] += 1
                p = enc_state["produced"]
                if p % 100 == 0:
                    elapsed = max(1e-3, time.perf_counter() - t0)
                    pct = p / total * 100.0 if total > 0 else 0.0
                    log(f"  {pct:5.1f}%  {p}/{total} frames ({p / elapsed:.1f} fps)")
            tail = enc.EndEncode()
            if tail:
                mux.stdin.write(tail)
        except BaseException as e:               # surface to main; never hang the pipe
            enc_error.append(e)

    enc_thread = threading.Thread(target=_encode_worker, name="rm-encode", daemon=True)
    enc_thread.start()

    def _submit(item) -> None:
        while True:
            if enc_error:
                raise enc_error[0]
            try:
                enc_q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    try:
        for chunk_start in range(start_out, end_idx, WINDOW):
            actual = min(WINDOW, end_idx - chunk_start)
            chunk = [rgb_at(chunk_start + k) for k in range(WINDOW)]
            out_chunk_g = processor.process_chunk(chunk, conf)
            for j in range(actual):
                nv12_g = cp.empty((height * 3 // 2, width), cp.uint8)
                k_to_nv12(grid_out, bx, (out_chunk_g[j], nv12_g, np.int32(width), np.int32(height)))
                ev = cp.cuda.Event()
                ev.record()
                _submit((nv12_g, ev))
            for key in [k for k in cache if k < chunk_start + actual]:
                del cache[key]
        _submit(None)                            # sentinel: flush tail + EndEncode
        enc_thread.join()
        if enc_error:
            raise enc_error[0]
        try:
            mux.stdin.close()
        except Exception:
            pass
        mux.wait(timeout=120)
    finally:
        try:
            dec.stop()
        except Exception:
            pass
        if enc_thread.is_alive():
            try:
                enc_q.put_nowait(None)           # unblock a worker parked on the queue
            except Exception:
                pass
            enc_thread.join(timeout=5)
        if mux.poll() is None:
            mux.kill()
    stderr_thread.join(timeout=5)
    mux_err = b"".join(mux_err_chunks).decode("utf-8", "replace").strip()
    if mux.returncode not in (0, None) or not out.is_file():
        log(f"mux failed rc={mux.returncode}: {mux_err[:800]}")
        return 1
    produced = enc_state["produced"]
    elapsed = max(1e-3, time.perf_counter() - t0)
    log(f"done: {out} ({produced} frames, {produced / elapsed:.1f} fps avg)")
    return 0


def _video_files(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    out = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if path.stem.endswith(OUTPUT_SUFFIX):
            continue
        out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


# --- TensorRT cache ---------------------------------------------------------


def _ensure_trt_cache(provider: str) -> int:
    from pipeline import demosaic

    if not demosaic.models_available():
        log("RM models not found under models/demosaic/; aborting.")
        return 2
    if provider == "trt" and demosaic.rm_trt_cached():
        return 0
    log("RM TensorRT engine cache missing; building before conversion...")
    try:
        demosaic.warmup_rm_engines(provider=provider, log=log)
    except Exception as exc:
        log(f"RM TensorRT build failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline mosaic restoration (RM)")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="restore one video")
    single.add_argument("video")
    single.add_argument("--out-dir", dest="out_dir", default="")
    single.add_argument("--start", type=float, default=0.0)
    single.add_argument("--duration", type=float, default=0.0)
    _add_common_args(single)

    batch = sub.add_parser("batch", help="restore all videos under a directory")
    batch.add_argument("directory")
    batch.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    _add_common_args(batch)

    buildtrt = sub.add_parser("build-trt", help="pre-build the RM TensorRT engine cache")
    buildtrt.add_argument("--provider", default="trt", choices=["trt", "cuda", "cpu"])

    args = parser.parse_args(argv)

    if args.command == "build-trt":
        return _ensure_trt_cache(args.provider)

    rc = _ensure_trt_cache(args.provider)
    if rc != 0:
        return rc

    from pipeline.demosaic import get_shared_engines, GpuRmProcessor

    engines = get_shared_engines(provider=args.provider, log=log)
    detect_interval = max(1, int(args.detect_interval))

    def processor_factory():
        return GpuRmProcessor(engines, detect_interval=detect_interval)

    if args.command == "single":
        src = Path(args.video)
        if not src.is_file():
            log(f"input not found: {src}")
            return 2
        out_dir = Path(args.out_dir) if args.out_dir else None
        out = output_path(src, out_dir, args.start, args.duration)
        if getattr(args, "skip_existing", False) and out.is_file():
            log(f"skip existing: {out}")
            return 0
        return _run_clip(processor_factory, src, out, args.start, args.duration, args.conf)

    root = Path(args.directory)
    if not root.is_dir():
        log(f"directory not found: {root}")
        return 2
    files = _video_files(root, args.recursive)
    if not files:
        log(f"no videos found under {root}")
        return 0
    log(f"batch: {len(files)} videos")
    rc = 0
    for index, src in enumerate(files, 1):
        out = output_path(src, None, 0.0, 0.0)
        if getattr(args, "skip_existing", False) and out.is_file():
            log(f"[{index}/{len(files)}] skip existing: {out.name}")
            continue
        log(f"[{index}/{len(files)}] {src.name}")
        rc |= _run_clip(processor_factory, src, out, 0.0, 0.0, args.conf)
    return rc


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="trt", choices=["trt", "cuda", "cpu"])
    p.add_argument("--conf", type=float, default=config.RM_CONF,
                   help="detector confidence threshold")
    p.add_argument("--detect-interval", dest="detect_interval", type=int,
                   default=config.RM_DETECT_INTERVAL,
                   help="run detector every N frames (reuse boxes between)")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
