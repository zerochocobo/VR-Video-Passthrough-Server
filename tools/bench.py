"""Local diagnostics for playback, HTTP throughput, and pipeline performance.

Examples:
  uv run python tools/bench.py play     test.mp4
  uv run python tools/bench.py play     test.mp4 --raw
  uv run python tools/bench.py bench    test.mp4 -d 30
  uv run python tools/bench.py pipeline test.mp4 -d 10
  uv run python tools/bench.py decode   test.mp4 -d 10
"""
from __future__ import annotations

import argparse
import os
import queue
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

# Allow this script to import project modules when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def _resolve_snapshot_path(value: str, name: str, frame_no: int) -> Path:
    if value:
        p = Path(value)
    else:
        stem = Path(name).stem
        p = config.ROOT / "debug_output" / f"{stem}_matting_frame_{frame_no}.png"
    if not p.is_absolute():
        p = (config.ROOT / p).resolve()
    return p


def _bench_dec_queue_size(args: argparse.Namespace) -> int:
    return max(2, int(getattr(args, "dec_queue", 16)))


def _apply_max_fps_override(args: argparse.Namespace) -> None:
    value = getattr(args, "max_fps", None)
    if value is not None:
        config.PASSTHROUGH_MAX_FPS = max(0.0, float(value))


def _fmt_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def _print_decoder_diag(dec) -> None:
    print(f"[DIAG] decoder selected={dec.strategy_name}")
    print(f"[DIAG] decoder out={dec.out_info.width}x{dec.out_info.height} @ {dec.out_info.fps:.2f}fps")
    print(f"[DIAG] decoder cmd={_fmt_cmd(dec.cmd)}")
    for name, rc, err in dec.failed_strategies:
        err_one_line = err.replace("\r", " ").replace("\n", " | ").strip()
        print(f"[DIAG] decoder failed strategy={name} rc={rc} err={err_one_line[-500:]}")


def _resolve_model_path(value: str) -> Path:
    p = Path(value)
    if not p.suffix:
        p = p.with_suffix(".onnx")
    if not p.is_absolute():
        p = config.ROOT / "models" / p
    return p.resolve()


def _apply_model_override(value: str) -> None:
    if value:
        config.MODEL_PATH = _resolve_model_path(value)


def _url(name: str, passthrough: bool, t: float) -> str:
    base = f"http://127.0.0.1:{config.HTTP_PORT}"
    if passthrough:
        return f"{base}/passthrough/{quote(name)}?t={t}"
    return f"{base}/media/{quote(name)}"


def cmd_play(args: argparse.Namespace) -> int:
    """Open one media URL through ffplay for local playback smoke tests."""

    ffplay = shutil.which("ffplay") or "ffplay"
    url = _url(args.name, not args.raw, args.t)
    print(f"[play] {url}")
    cmd = [ffplay, "-fflags", "nobuffer", "-framedrop", "-stats"]
    if not getattr(args, "seekable", False):
        cmd += ["-seekable", "0"]
    cmd += ["-window_title", f"PT Diag - {args.name}", url]
    return subprocess.call(cmd)


def cmd_bench(args: argparse.Namespace) -> int:
    """Measure HTTP playback throughput by piping the URL through ffmpeg."""

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    url = _url(args.name, not args.raw, args.t)
    print(f"[bench] {args.duration}s of {url}")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "info"]
    if not getattr(args, "seekable", False):
        cmd += ["-seekable", "0"]
    cmd += ["-i", url, "-t", str(args.duration), "-f", "null", "-"]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    # ffmpeg stats stderr frame= xxx fps= xx
    err = proc.stderr or ""
    last_stat = ""
    for line in err.splitlines():
        if line.startswith("frame=") or " frame=" in line:
            last_stat = line.strip()
    print(err[-1500:])
    print("\n=== summary ===")
    print(f"wall_time = {elapsed:.2f}s")
    print(f"last_stat = {last_stat}")
    return proc.returncode


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Benchmark decode plus matting/composite without encoding output."""
    import numpy as np  # noqa: E402

    if args.input_size:
        config.MATTING_INPUT_SIZE = args.input_size
    if args.max_side is not None:
        config.DECODE_MAX_SIDE = args.max_side
    _apply_max_fps_override(args)
    if args.decode_pix_fmt:
        config.DECODE_PIX_FMT = args.decode_pix_fmt.lower()
    _apply_model_override(args.model)
    if args.warmup_runs is not None:
        config.MATTING_WARMUP_RUNS = args.warmup_runs
    if args.alpha_stride is not None:
        os.environ["PT_ALPHA_STRIDE"] = str(args.alpha_stride)
    if args.alpha_mode:
        os.environ["PT_ALPHA_MODE"] = args.alpha_mode

    from pipeline.ffmpeg_io import DecoderProcess, probe_cached  # noqa: E402
    from pipeline.matting import get_matter  # noqa: E402

    src = config.VIDEO_DIR / args.name
    if not src.is_file():
        print(f"not found: {src}", file=sys.stderr)
        return 2

    info = probe_cached(src)
    print(f"[pipeline] src {info.width}x{info.height} {info.codec_name}/{info.pix_fmt} @ {info.fps:.2f}fps  duration={info.duration:.1f}s")
    print(
        f"loading matting model... model={config.MODEL_PATH.name} "
        f"matting_input={config.MATTING_INPUT_SIZE} decode_max_side={config.DECODE_MAX_SIDE}"
    )
    matter = get_matter()
    dec = DecoderProcess(src, args.t, info)
    _print_decoder_diag(dec)
    target = int(args.duration * dec.out_info.fps)
    print(f"target {target} frames ({args.duration}s @ {dec.out_info.fps:.2f}fps output)")

    n = 0
    n_skip = max(target // 10, 1)
    save_frame_no = 0
    save_frame_path: Path | None = None
    saved_frame_path: Path | None = None
    if args.save_frame is not None:
        save_frame_no = args.save_frame if args.save_frame > 0 else random.randint(1, max(target, 1))
        save_frame_path = _resolve_snapshot_path(args.save_frame_path, args.name, save_frame_no)
        print(f"[snapshot] will save frame #{save_frame_no} to {save_frame_path}")
    t_pre = 0.0
    t_ort = 0.0
    t_resize = 0.0
    t_comp = 0.0
    t_wait = 0.0  # Time spent waiting for the decoder prefetch queue.

    # Decoder prefetch isolates matting timings from short decode stalls.
    dec_q: queue.Queue = queue.Queue(maxsize=_bench_dec_queue_size(args))
    stop_dec = threading.Event()

    def decoder_thread() -> None:
        produced = 0
        try:
            while not stop_dec.is_set():
                raw = dec.read_frame()
                if raw is None:
                    break
                while not stop_dec.is_set():
                    try:
                        dec_q.put(raw, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return
                produced += 1
        finally:
            try:
                dec_q.put_nowait(None)
            except Exception:
                pass

    dec_thr = threading.Thread(target=decoder_thread, name="bench-dec", daemon=True)
    dec_thr.start()

    start = time.time()
    try:
        for _ in range(target):
            tw0 = time.perf_counter()
            raw = dec_q.get()
            tw1 = time.perf_counter()
            t_wait += tw1 - tw0
            if raw is None:
                break
            if dec.out_info.pix_fmt == "nv12":
                frame = np.frombuffer(raw, dtype=np.uint8)
                composed, mt = matter.composite_green_nv12_profile(frame, dec.out_info.height, dec.out_info.width)
            else:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(dec.out_info.height, dec.out_info.width, 3)
                composed, mt = matter.composite_green_profile(frame)
            if save_frame_path is not None and saved_frame_path is None and n + 1 >= save_frame_no:
                import cv2  # noqa: E402

                save_frame_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_frame_path), composed)
                saved_frame_path = save_frame_path
                print(f"[snapshot] saved {saved_frame_path}")
            t_pre += mt.preprocess_ms
            t_ort += mt.ort_ms
            t_resize += mt.alpha_resize_ms
            t_comp += mt.composite_ms
            n += 1
            if n % n_skip == 0:
                el = time.time() - start
                print(
                    f"  {n:5d} frames | {el:6.2f}s elapsed | {n/el:6.2f} fps | "
                    f"pre={t_pre/n:5.1f}ms ort={t_ort/n:5.1f}ms "
                    f"alpha_up={t_resize/n:5.1f}ms comp={t_comp/n:5.1f}ms "
                    f"dec_wait={t_wait/n*1000:5.1f}ms "
                    f"shape={getattr(matter, '_last_ort_shape', '?')}"
                )
    finally:
        stop_dec.set()
        # Drain pending frames so decoder_thread cannot block during shutdown.
        try:
            while True:
                dec_q.get_nowait()
        except queue.Empty:
            pass
        dec.close()
        dec_thr.join(timeout=2.0)

    el = time.time() - start
    fps = n / el if el > 0 else 0
    realtime = fps >= dec.out_info.fps * 0.95
    print("\n=== summary ===")
    print(f"frames    = {n}")
    print(f"wall_time = {el:.2f}s")
    print(f"throughput= {fps:.2f} fps")
    print(f"source    = {info.fps:.2f} fps")
    print(f"output    = {dec.out_info.fps:.2f} fps")
    if n > 0:
        print(f"avg_preprocess   = {t_pre/n:.2f} ms")
        print(f"avg_ort_run      = {t_ort/n:.2f} ms")
        print(f"avg_alpha_resize = {t_resize/n:.2f} ms")
        print(f"avg_composite    = {t_comp/n:.2f} ms")
        print(f"avg_dec_wait     = {t_wait/n*1000:.2f} ms  (near 0 means matting fully hides decode wait)")
        print(f"last_ort_shape   = {getattr(matter, '_last_ort_shape', '?')}")
        if saved_frame_path is not None:
            print(f"snapshot         = {saved_frame_path}")
    if realtime:
        print(f"verdict   = REAL-TIME ({fps/dec.out_info.fps:.2f}x output rate)")
    else:
        print(f"verdict   = NOT real-time (lag {dec.out_info.fps/max(fps, 0.01):.2f}x slower than output)")
    return 0 if realtime else 1


def cmd_transcode(args: argparse.Namespace) -> int:
    """Local decode + matting + encode benchmark without HTTP."""
    import numpy as np  # noqa: E402

    if args.input_size:
        config.MATTING_INPUT_SIZE = args.input_size
    if args.max_side is not None:
        config.DECODE_MAX_SIDE = args.max_side
    _apply_max_fps_override(args)
    if args.decode_pix_fmt:
        config.DECODE_PIX_FMT = args.decode_pix_fmt.lower()
    _apply_model_override(args.model)
    if args.warmup_runs is not None:
        config.MATTING_WARMUP_RUNS = args.warmup_runs
    if args.alpha_stride is not None:
        os.environ["PT_ALPHA_STRIDE"] = str(args.alpha_stride)
    if args.alpha_mode:
        os.environ["PT_ALPHA_MODE"] = args.alpha_mode

    from pipeline.ffmpeg_io import DecoderProcess, EncoderProcess, probe_cached  # noqa: E402
    from pipeline.matting import get_matter  # noqa: E402

    src = config.VIDEO_DIR / args.name
    if not src.is_file():
        print(f"not found: {src}", file=sys.stderr)
        return 2

    info = probe_cached(src)
    print(f"[transcode] src {info.width}x{info.height} {info.codec_name}/{info.pix_fmt} @ {info.fps:.2f}fps")
    print(
        f"model={config.MODEL_PATH.name} matting_input={config.MATTING_INPUT_SIZE} "
        f"decode_max_side={config.DECODE_MAX_SIDE} vcodec={config.PASSTHROUGH_VCODEC}"
    )
    print(
        f"encoder preset={config.PASSTHROUGH_PRESET} bitrate={config.PASSTHROUGH_BITRATE} "
        f"gop={config.PASSTHROUGH_GOP} tune={config.PASSTHROUGH_TUNE or '-'} rc={config.PASSTHROUGH_RC or '-'}"
    )
    matter = get_matter()
    dec = DecoderProcess(src, args.t, info)
    encode_input_pix_fmt = args.encode_input_pix_fmt
    if encode_input_pix_fmt == "auto":
        encode_input_pix_fmt = "nv12" if dec.out_info.pix_fmt == "nv12" else "bgr24"
    enc = EncoderProcess(dec.out_info.width, dec.out_info.height, dec.out_info.fps, input_pix_fmt=encode_input_pix_fmt)
    _print_decoder_diag(dec)

    drained = 0
    stop_drain = threading.Event()

    def drain() -> None:
        nonlocal drained
        while not stop_drain.is_set():
            data = enc.read_output(256 * 1024)
            if not data:
                break
            drained += len(data)

    drainer = threading.Thread(target=drain, name="bench-enc-drain", daemon=True)
    drainer.start()

    target = int(args.duration * dec.out_info.fps)
    print(f"target {target} frames ({args.duration}s @ {dec.out_info.fps:.2f}fps output)")
    n = 0
    n_skip = max(target // 10, 1)
    t_dec = t_mat = 0.0  # t_dec is queue.get wait time
    t_pre = 0.0
    t_ort = 0.0
    t_resize = 0.0
    t_comp = 0.0
    enc_q: queue.Queue = queue.Queue(maxsize=args.enc_queue)
    enc_free_q: queue.Queue | None = None
    enc_pool_refs: list[object] = []
    use_nv12_pool = dec.out_info.pix_fmt == "nv12" and encode_input_pix_fmt == "nv12"
    if use_nv12_pool:
        pool = matter.make_pinned_nv12_output_pool(
            dec.out_info.height,
            dec.out_info.width,
            args.enc_queue + 1,
        )
        enc_pool_refs = [mem for mem, _ in pool]
        enc_free_q = queue.Queue(maxsize=len(pool))
        for _, arr in pool:
            enc_free_q.put_nowait(arr)
        print(f"[DIAG] encoder handoff pool: {len(pool)} pinned NV12 frames")
    enc_frames = 0
    enc_write_total = 0.0
    enc_write_samples: list[float] = []
    writer_ok = True
    start = time.time()

    # Decoder prefetch queue mirrors the production stream shape.
    dec_q: queue.Queue = queue.Queue(maxsize=_bench_dec_queue_size(args))
    stop_dec = threading.Event()

    def decoder_thread() -> None:
        try:
            while not stop_dec.is_set():
                raw = dec.read_frame()
                if raw is None:
                    break
                while not stop_dec.is_set():
                    try:
                        dec_q.put(raw, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return
        finally:
            try:
                dec_q.put_nowait(None)
            except Exception:
                pass

    def writer_thread() -> None:
        nonlocal enc_frames, enc_write_total, writer_ok
        try:
            while not stop_dec.is_set():
                item = enc_q.get()
                if item is None:
                    break
                if isinstance(item, tuple):
                    frame, release_to_pool = item
                else:
                    frame, release_to_pool = item, False
                t0 = time.perf_counter()
                try:
                    ok = enc.write_frame(memoryview(np.ascontiguousarray(frame)).cast("B"))
                finally:
                    if release_to_pool and enc_free_q is not None:
                        try:
                            enc_free_q.put_nowait(frame)
                        except queue.Full:
                            pass
                t1 = time.perf_counter()
                enc_elapsed = t1 - t0
                enc_write_total += enc_elapsed
                enc_write_samples.append(enc_elapsed * 1000.0)
                enc_frames += 1
                if not ok:
                    writer_ok = False
                    print("encoder pipe closed", file=sys.stderr)
                    break
        finally:
            try:
                if enc.proc.stdin:
                    enc.proc.stdin.close()
            except Exception:
                pass

    dec_thr = threading.Thread(target=decoder_thread, name="bench-dec", daemon=True)
    dec_thr.start()
    wr_thr = threading.Thread(target=writer_thread, name="bench-enc-write", daemon=True)
    wr_thr.start()

    try:
        for _ in range(target):
            if not writer_ok:
                break
            t0 = time.perf_counter()
            raw = dec_q.get()
            t1 = time.perf_counter()
            if raw is None:
                break
            release_to_pool = False
            if dec.out_info.pix_fmt == "nv12" and encode_input_pix_fmt == "nv12":
                frame = np.frombuffer(raw, dtype=np.uint8)
                out_buf = None
                if enc_free_q is not None:
                    while not stop_dec.is_set():
                        try:
                            out_buf = enc_free_q.get(timeout=0.5)
                            break
                        except queue.Empty:
                            if not writer_ok:
                                break
                            continue
                composed, mt = matter.composite_green_nv12_to_nv12_profile(
                    frame,
                    dec.out_info.height,
                    dec.out_info.width,
                    out=out_buf,
                )
                release_to_pool = out_buf is not None
            elif dec.out_info.pix_fmt == "nv12":
                frame = np.frombuffer(raw, dtype=np.uint8)
                composed, mt = matter.composite_green_nv12_profile(frame, dec.out_info.height, dec.out_info.width)
            else:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(dec.out_info.height, dec.out_info.width, 3)
                composed, mt = matter.composite_green_profile(frame)
            t2 = time.perf_counter()
            while not stop_dec.is_set():
                try:
                    enc_q.put((composed, release_to_pool), timeout=0.5)
                    break
                except queue.Full:
                    if not writer_ok:
                        if release_to_pool and enc_free_q is not None:
                            try:
                                enc_free_q.put_nowait(composed)
                            except queue.Full:
                                pass
                        break
                    continue
            t_dec += t1 - t0
            t_mat += t2 - t1
            t_pre += mt.preprocess_ms
            t_ort += mt.ort_ms
            t_resize += mt.alpha_resize_ms
            t_comp += mt.composite_ms
            n += 1
            if n % n_skip == 0:
                el = time.time() - start
                avg_enc = enc_write_total / max(enc_frames, 1) * 1000
                print(
                    f"  {n:5d} frames | {el:6.2f}s elapsed | {n/el:6.2f} fps | "
                    f"dec_wait={t_dec/n*1000:5.1f}ms mat={t_mat/n*1000:5.1f}ms "
                    f"pre={t_pre/n:5.1f}ms ort={t_ort/n:5.1f}ms "
                    f"alpha_up={t_resize/n:5.1f}ms comp={t_comp/n:5.1f}ms "
                    f"enc_write={avg_enc:5.1f}ms enc_q={enc_q.qsize():2d} "
                    f"shape={getattr(matter, '_last_ort_shape', '?')}"
                )
    finally:
        stop_dec.set()
        try:
            enc_q.put_nowait(None)
        except queue.Full:
            pass
        try:
            while True:
                dec_q.get_nowait()
        except queue.Empty:
            pass
        dec.close()
        dec_thr.join(timeout=2.0)
        wr_thr.join(timeout=10.0)
        drainer.join(timeout=5)
        stop_drain.set()
        enc.close()

    elapsed = time.time() - start
    fps = n / elapsed if elapsed > 0 else 0
    realtime = fps >= dec.out_info.fps * 0.95
    print("\n=== summary ===")
    print(f"frames       = {n}")
    print(f"wall_time    = {elapsed:.2f}s")
    print(f"throughput   = {fps:.2f} fps")
    print(f"source       = {info.fps:.2f} fps")
    print(f"output       = {dec.out_info.fps:.2f} fps")
    print(f"encoded      = {drained / 1_000_000:.2f} MB")
    if n > 0:
        enc_p99 = 0.0
        enc_stdev = 0.0
        if enc_write_samples:
            ordered = sorted(enc_write_samples)
            enc_p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
            enc_stdev = statistics.stdev(enc_write_samples) if len(enc_write_samples) > 1 else 0.0
        print(f"avg_dec_wait = {t_dec/n*1000:.2f} ms  (near 0 means matting fully hides decode wait)")
        print(f"avg_matting  = {t_mat/n*1000:.2f} ms")
        print(f"avg_preprocess= {t_pre/n:.2f} ms")
        print(f"avg_ort_run   = {t_ort/n:.2f} ms")
        print(f"avg_alpha_up  = {t_resize/n:.2f} ms")
        print(f"avg_composite = {t_comp/n:.2f} ms")
        print(f"avg_enc_write= {enc_write_total/max(enc_frames, 1)*1000:.2f} ms")
        print(f"p99_enc_write= {enc_p99:.2f} ms")
        print(f"std_enc_write= {enc_stdev:.2f} ms")
        print(f"encoded_frames= {enc_frames}")
        print(f"last_ort_shape= {getattr(matter, '_last_ort_shape', '?')}")
    print(f"verdict      = {'REAL-TIME' if realtime else 'NOT real-time'}")
    return 0 if realtime else 1


def cmd_decode(args: argparse.Namespace) -> int:
    """ HTTP/ / """
    if args.max_side is not None:
        config.DECODE_MAX_SIDE = args.max_side
    _apply_max_fps_override(args)

    from pipeline.ffmpeg_io import DecoderProcess, probe_cached  # noqa: E402

    src = config.VIDEO_DIR / args.name
    if not src.is_file():
        print(f"not found: {src}", file=sys.stderr)
        return 2

    info = probe_cached(src)
    print(f"[decode] src {info.width}x{info.height} {info.codec_name}/{info.pix_fmt} @ {info.fps:.2f}fps  duration={info.duration:.1f}s")
    dec = DecoderProcess(src, args.t, info)
    _print_decoder_diag(dec)
    target = int(args.duration * dec.out_info.fps)
    print(f"target {target} frames ({args.duration}s @ {dec.out_info.fps:.2f}fps output)")

    n = 0
    n_skip = max(target // 10, 1)
    start = time.time()
    try:
        for _ in range(target):
            raw = dec.read_frame()
            if raw is None:
                break
            n += 1
            if n % n_skip == 0:
                el = time.time() - start
                print(f"  {n:5d} frames | {el:6.2f}s elapsed | {n/el:6.2f} fps")
    finally:
        dec.close()

    dec_err = dec.read_stderr_nonblock()
    if dec_err:
        print("\n[decode stderr]\n" + dec_err[-2000:])

    el = time.time() - start
    fps = n / el if el > 0 else 0
    realtime = fps >= dec.out_info.fps * 0.95
    print("\n=== summary ===")
    print(f"frames    = {n}")
    print(f"wall_time = {el:.2f}s")
    print(f"throughput= {fps:.2f} fps")
    print(f"source    = {info.fps:.2f} fps")
    print(f"output    = {dec.out_info.fps:.2f} fps")
    if realtime:
        print(f"verdict   = REAL-TIME ({fps/dec.out_info.fps:.2f}x output rate)")
    else:
        print(f"verdict   = NOT real-time (lag {dec.out_info.fps/max(fps, 0.01):.2f}x slower than output)")
    return 0 if realtime else 1


def _cuvid_decoder(codec_name: str) -> str | None:
    return {
        "h264": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "av1": "av1_cuvid",
        "vp8": "vp8_cuvid",
        "vp9": "vp9_cuvid",
        "mpeg1video": "mpeg1_cuvid",
        "mpeg2video": "mpeg2_cuvid",
        "mpeg4": "mpeg4_cuvid",
    }.get(codec_name)


def _scaled_size(width: int, height: int) -> tuple[int, int]:
    max_side = config.DECODE_MAX_SIDE
    if max_side <= 0 or max(width, height) <= max_side:
        return width, height
    if width >= height:
        out_w = max_side
        out_h = int(round(height * max_side / width))
    else:
        out_h = max_side
        out_w = int(round(width * max_side / height))
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(out_w, 2), max(out_h, 2)


def _run_ffmpeg_case(label: str, cmd: list[str]) -> tuple[int, float, str]:
    print(f"\n[{label}]")
    print(_fmt_cmd(cmd))
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    err = proc.stderr or ""
    last_stat = ""
    for line in err.replace("\r", "\n").splitlines():
        if "frame=" in line or "bench:" in line:
            last_stat = line.strip()
    print(err[-1200:])
    print(f"[{label}] rc={proc.returncode} wall={elapsed:.2f}s last={last_stat}")
    return proc.returncode, elapsed, last_stat


def cmd_decode_matrix(args: argparse.Namespace) -> int:
    """ ffmpeg decode/resize/pix_fmt/rawvideo/Python pipe """
    _apply_max_fps_override(args)

    from pipeline.ffmpeg_io import DecoderProcess, FFMPEG, probe_cached  # noqa: E402

    src = config.VIDEO_DIR / args.name
    if not src.is_file():
        print(f"not found: {src}", file=sys.stderr)
        return 2

    info = probe_cached(src)
    output_fps = min(info.fps, config.PASSTHROUGH_MAX_FPS) if config.PASSTHROUGH_MAX_FPS > 0 else info.fps
    target = int(args.duration * output_fps)
    out_w, out_h = _scaled_size(info.width, info.height)
    decoder = _cuvid_decoder(info.codec_name)
    print(f"[decode-matrix] src {info.width}x{info.height} {info.codec_name}/{info.pix_fmt} @ {info.fps:.2f}fps")
    print(f"target {target} frames, scaled={out_w}x{out_h}, output_fps={output_fps:.2f}, decoder={decoder or 'n/a'}")
    if not decoder:
        print(f"no cuvid decoder mapping for codec: {info.codec_name}", file=sys.stderr)
        return 2

    base_in = [
        FFMPEG, "-hide_banner", "-loglevel", "info", "-stats", "-benchmark",
        "-c:v", decoder,
    ]
    if (out_w, out_h) != (info.width, info.height):
        base_in += ["-resize", f"{out_w}x{out_h}"]
    base_in += [
        "-threads", "0",
        "-ss", f"{max(0.0, args.t):.3f}",
        "-i", str(src),
        "-an", "-sn",
        "-fps_mode", "passthrough",
    ]
    if output_fps < info.fps * 0.999:
        base_in += ["-vf", f"fps=fps={output_fps:.6f}:round=near"]
    base_in += ["-frames:v", str(target)]

    cases = [
        ("cuvid-null", [*base_in, "-f", "null", "-"]),
        ("cuvid-yuv420p-devnull", [*base_in, "-f", "rawvideo", "-pix_fmt", "yuv420p", os.devnull]),
        ("cuvid-bgr24-devnull", [*base_in, "-f", "rawvideo", "-pix_fmt", "bgr24", os.devnull]),
    ]

    worst_rc = 0
    for label, cmd in cases:
        rc, _, _ = _run_ffmpeg_case(label, cmd)
        worst_rc = worst_rc or rc

    print("\n[python-pipe-bgr24]")
    dec = DecoderProcess(src, args.t, info)
    _print_decoder_diag(dec)
    n = 0
    start = time.time()
    try:
        for _ in range(target):
            raw = dec.read_frame()
            if raw is None:
                break
            n += 1
    finally:
        dec.close()
    elapsed = time.time() - start
    fps = n / elapsed if elapsed > 0 else 0
    print(f"[python-pipe-bgr24] frames={n} wall={elapsed:.2f}s fps={fps:.2f}")
    return worst_rc


def cmd_matting(args: argparse.Namespace) -> int:
    """Profile ONNX matting on one decoded frame, repeated in memory."""
    import numpy as np  # noqa: E402

    if args.input_size:
        config.MATTING_INPUT_SIZE = args.input_size
    if args.max_side is not None:
        config.DECODE_MAX_SIDE = args.max_side
    _apply_model_override(args.model)
    if args.warmup_runs is not None:
        config.MATTING_WARMUP_RUNS = args.warmup_runs

    from pipeline.ffmpeg_io import DecoderProcess, probe_cached  # noqa: E402
    from pipeline.matting import get_matter  # noqa: E402

    src = config.VIDEO_DIR / args.name
    if not src.is_file():
        print(f"not found: {src}", file=sys.stderr)
        return 2

    info = probe_cached(src)
    dec = DecoderProcess(src, args.t, info)
    try:
        raw = dec.read_frame()
        if raw is None:
            print("decoder returned EOF before first frame", file=sys.stderr)
            return 1
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(dec.out_info.height, dec.out_info.width, 3)
        frame = np.ascontiguousarray(frame)
    finally:
        dec.close()

    matter = get_matter()
    print(
        f"[matting] frame={frame.shape[1]}x{frame.shape[0]} "
        f"model={config.MODEL_PATH.name} input_type={matter.input_type} "
        f"matting_input={config.MATTING_INPUT_SIZE} providers={matter.sess.get_providers()}"
    )
    print(f"target {args.frames} repeated frames")

    t_pre = 0.0
    t_ort = 0.0
    t_resize = 0.0
    t_comp = 0.0
    start = time.time()
    for i in range(args.frames):
        _, mt = matter.composite_green_profile(frame)
        t_pre += mt.preprocess_ms
        t_ort += mt.ort_ms
        t_resize += mt.alpha_resize_ms
        t_comp += mt.composite_ms
        if (i + 1) == 1 or (i + 1) % max(args.frames // 10, 1) == 0:
            n = i + 1
            el = time.time() - start
            print(
                f"  {n:5d} frames | {el:6.2f}s elapsed | {n/el:6.2f} fps | "
                f"pre={t_pre/n:5.1f}ms ort={t_ort/n:5.1f}ms "
                f"alpha_up={t_resize/n:5.1f}ms comp={t_comp/n:5.1f}ms "
                f"shape={getattr(matter, '_last_ort_shape', '?')}"
            )

    elapsed = time.time() - start
    fps = args.frames / elapsed if elapsed > 0 else 0
    print("\n=== summary ===")
    print(f"frames           = {args.frames}")
    print(f"wall_time        = {elapsed:.2f}s")
    print(f"throughput       = {fps:.2f} fps")
    print(f"avg_preprocess   = {t_pre/args.frames:.2f} ms")
    print(f"avg_ort_run      = {t_ort/args.frames:.2f} ms")
    print(f"avg_alpha_resize = {t_resize/args.frames:.2f} ms")
    print(f"avg_composite    = {t_comp/args.frames:.2f} ms")
    print(f"last_ort_shape   = {getattr(matter, '_last_ort_shape', '?')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="PTServer local diagnostic")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="open ffplay for raw or passthrough playback")
    p.add_argument("name", help="video filename under videos/ or the configured video directory")
    p.add_argument("--raw", action="store_true", help="play the raw source video instead of passthrough")
    p.add_argument("--seekable", action="store_true", help="allow ffmpeg HTTP byte seek probes")
    p.add_argument("-t", type=float, default=0.0, help="start time in seconds")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("bench", help="measure end-to-end HTTP output FPS")
    p.add_argument("name")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--seekable", action="store_true", help="allow ffmpeg HTTP byte seek probes")
    p.add_argument("-t", type=float, default=0.0)
    p.add_argument("-d", "--duration", type=float, default=20.0)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("pipeline", help="measure local decode + matting without HTTP/encode")
    p.add_argument("name")
    p.add_argument("-t", type=float, default=0.0)
    p.add_argument("-d", "--duration", type=float, default=10.0)
    p.add_argument("--input-size", type=int, default=0, help="override PT_MATTING_INPUT_SIZE for this run")
    p.add_argument("--max-side", type=int, default=None, help="override PT_DECODE_MAX_SIDE for this run")
    p.add_argument("--max-fps", type=float, default=None, help="cap decoded/output fps for this run; 0 keeps source fps")
    p.add_argument("--decode-pix-fmt", choices=["bgr24", "nv12"], default="", help="override decoder rawvideo pixel format")
    p.add_argument("--model", default="", help="model file in models/ or absolute path, e.g. model_fp16.onnx")
    p.add_argument("--warmup-runs", type=int, default=None, help="override PT_MATTING_WARMUP_RUNS for this run")
    p.add_argument("--alpha-stride", type=int, default=None, help="run matting inference every N frames and reuse alpha between runs")
    p.add_argument("--alpha-mode", choices=["reuse"], default="", help="alpha skip mode; currently only reuse is realtime-safe")
    p.add_argument("--dec-queue", type=int, default=16, help="decoded-frame prefetch queue size for bench")
    p.add_argument(
        "--save-frame",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="save one composited PNG frame; omit value for random frame, or pass a 1-based frame number",
    )
    p.add_argument("--save-frame-path", default="", help="PNG output path for --save-frame")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("transcode", help="measure local decode + matting + encode without HTTP")
    p.add_argument("name")
    p.add_argument("-t", type=float, default=0.0)
    p.add_argument("-d", "--duration", type=float, default=10.0)
    p.add_argument("--input-size", type=int, default=0, help="override PT_MATTING_INPUT_SIZE for this run")
    p.add_argument("--max-side", type=int, default=None, help="override PT_DECODE_MAX_SIDE for this run")
    p.add_argument("--max-fps", type=float, default=None, help="cap decoded/output fps for this run; 0 keeps source fps")
    p.add_argument("--decode-pix-fmt", choices=["bgr24", "nv12"], default="", help="override decoder rawvideo pixel format")
    p.add_argument("--model", default="", help="model file in models/ or absolute path, e.g. model_fp16.onnx")
    p.add_argument("--warmup-runs", type=int, default=None, help="override PT_MATTING_WARMUP_RUNS for this run")
    p.add_argument("--alpha-stride", type=int, default=None, help="run matting inference every N frames and reuse alpha between runs")
    p.add_argument("--alpha-mode", choices=["reuse"], default="", help="alpha skip mode; currently only reuse is realtime-safe")
    p.add_argument("--dec-queue", type=int, default=16, help="decoded-frame prefetch queue size for bench")
    p.add_argument("--enc-queue", type=int, default=8, help="encoded input frame queue size for async transcode writer")
    p.add_argument("--encode-input-pix-fmt", choices=["auto", "bgr24", "nv12"], default="bgr24", help="raw pixel format sent to encoder stdin")
    p.set_defaults(func=cmd_transcode)

    p = sub.add_parser("decode", help="measure local decode throughput only")
    p.add_argument("name")
    p.add_argument("-t", type=float, default=0.0)
    p.add_argument("-d", "--duration", type=float, default=10.0)
    p.add_argument("--max-side", type=int, default=None, help="override PT_DECODE_MAX_SIDE for this run")
    p.add_argument("--max-fps", type=float, default=None, help="cap decoded/output fps for this run; 0 keeps source fps")
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("decode-matrix", help="split-test cuvid decode, conversion, rawvideo, and Python pipe")
    p.add_argument("name")
    p.add_argument("-t", type=float, default=0.0)
    p.add_argument("-d", "--duration", type=float, default=10.0)
    p.add_argument("--max-fps", type=float, default=None, help="cap filter output fps for comparable pipe tests; 0 keeps source fps")
    p.set_defaults(func=cmd_decode_matrix)

    p = sub.add_parser("matting", help="repeat one frame to profile ONNX matting/composite")
    p.add_argument("name")
    p.add_argument("-t", type=float, default=0.0)
    p.add_argument("-n", "--frames", type=int, default=120)
    p.add_argument("--input-size", type=int, default=0, help="override PT_MATTING_INPUT_SIZE for this run")
    p.add_argument("--max-side", type=int, default=None, help="override PT_DECODE_MAX_SIDE for this run")
    p.add_argument("--model", default="", help="model file in models/ or absolute path, e.g. model_fp16.onnx")
    p.add_argument("--warmup-runs", type=int, default=None, help="override PT_MATTING_WARMUP_RUNS for this run")
    p.set_defaults(func=cmd_matting)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
