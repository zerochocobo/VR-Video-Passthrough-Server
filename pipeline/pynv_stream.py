"""Production PyNv HEVC passthrough stream.

The stream decodes source frames with PyNvVideoCodec, runs GPU matting and
NV12 compositing, encodes HEVC with NVENC, then asks FFmpeg only to mux the
compressed bitstream into fragmented MP4 or MPEG-TS. It also owns preflight,
diagnostic logging, stderr draining, and per-stream output FPS caps.
"""
from __future__ import annotations

import asyncio
import gc
import os
import itertools
import queue
import shutil
import socket
import subprocess
import sys
import threading
import traceback
import time
from pathlib import Path
from typing import AsyncIterator

import config
from pipeline import matting as matting_module
from pipeline.matting import Matter
from pipeline.pynv_io import GpuNv12AppFrame, GpuP016Frame, PyNvSimpleDecoder, PyNvThreadedSerialDecoder
from pipeline.subtitles import SubtitleRenderer, find_subtitle_for_video
from utils.logger import get, warmup_event
from utils.startup_status import get_startup_state, set_startup_phase
from utils.cache_key import fingerprint, stat_key
from utils.bitrate_estimator import effective_default_bitrate, parse_bitrate
from utils.runtime_settings import get_light_match
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.video_metadata import VideoProbeMetadata, cfr_source_index, probe_color_metadata, probe_timing_metadata

log = get("pynv_stream")

_READ_CHUNK = 256 * 1024
_QUEUE_MAX = 256
_THREAD_JOIN_TIMEOUT = 2.0
PYNV_OUTPUT_CODEC = "hevc"
PYNV_BACKEND_LABEL = "pynv_hevc"
_DIAG_INTERVAL = 30
_AUDIO_CACHE_PROGRESS_INTERVAL = 5.0
_PREFLIGHT_TTL_SEC = 2 * 60 * 60
_PREFLIGHT_CACHE_MAX_ENTRIES = 256
_AUDIO_CACHE_LOCK_MAX_ENTRIES = 512
_BENIGN_FFMPEG_WARNINGS = (
    "Stream #0: not enough frames to estimate rate",
    "Timestamps are unset in a packet",
)
_preflight_lock = threading.Lock()
_preflight_ok: dict[str, float] = {}
_stream_ids = itertools.count(1)
_audio_cache_locks: dict[str, threading.Lock] = {}
_audio_cache_locks_guard = threading.Lock()
_audio_tmp_cleanup_done = False
_audio_tmp_cleanup_lock = threading.Lock()
_pynv_runtime_tainted = threading.Event()
_SUBTITLE_BLEND_Y_KERNEL = None
_SUBTITLE_BLEND_UV_KERNEL = None


def _realtime_pynv_bitrate() -> str:
    """Return configured realtime PyNv HEVC bitrate in bits/second."""
    return str(parse_bitrate(config.PASSTHROUGH_HEVC_BITRATE))


def _drain_async_queue_nowait(q: asyncio.Queue, *, keep_sentinel: bool = True) -> tuple[int, int, bool]:
    """Drop queued byte chunks and optionally preserve an end sentinel."""
    chunks = 0
    bytes_dropped = 0
    saw_sentinel = False
    while True:
        try:
            item = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item is None:
            saw_sentinel = True
            if keep_sentinel:
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                break
            continue
        if isinstance(item, (bytes, bytearray, memoryview)):
            chunks += 1
            bytes_dropped += len(item)
    return chunks, bytes_dropped, saw_sentinel


def _pynv_encoder_kwargs(*, bitrate: str, fps: str) -> dict[str, str]:
    kwargs = {
        "codec": PYNV_OUTPUT_CODEC,
        "bitrate": str(bitrate),
        "fps": str(fps),
        "gop": str(config.PASSTHROUGH_GOP),
        "bf": str(config.PASSTHROUGH_HEVC_BF),
    }
    if config.PASSTHROUGH_PYNV_PRESET:
        kwargs["preset"] = str(config.PASSTHROUGH_PYNV_PRESET)
    if config.PASSTHROUGH_PYNV_TUNING_INFO:
        kwargs["tuning_info"] = str(config.PASSTHROUGH_PYNV_TUNING_INFO)
    if config.PASSTHROUGH_PYNV_RC:
        kwargs["rc"] = str(config.PASSTHROUGH_PYNV_RC)
    if config.PASSTHROUGH_PYNV_IDR_PERIOD:
        kwargs["idrperiod"] = str(config.PASSTHROUGH_PYNV_IDR_PERIOD)
    return kwargs


def pynv_runtime_tainted() -> bool:
    return _pynv_runtime_tainted.is_set()


def _rgb_to_limited_yuv(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    y = 16.0 + (65.738 * r + 129.057 * g + 25.064 * b) / 256.0
    u = 128.0 + (-37.945 * r - 74.494 * g + 112.439 * b) / 256.0
    v = 128.0 + (112.439 * r - 94.154 * g - 18.285 * b) / 256.0
    return tuple(max(0, min(255, int(round(x)))) for x in (y, u, v))


def _format_mib(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _cupy_vram_snapshot() -> tuple[int, int, int, int, int] | None:
    try:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        return (
            int(total_bytes) - int(free_bytes),
            int(total_bytes),
            int(pool.used_bytes()),
            int(pool.total_bytes()),
            int(pinned_pool.n_free_blocks()),
        )
    except Exception:
        return None


def _mpegts_flags() -> str:
    flags = "+resend_headers"
    if config.PASSTHROUGH_AUDIO_MPEGTS_PAT_PMT_AT_FRAMES:
        flags += "+pat_pmt_at_frames"
    return flags


def _mux_fflags() -> str:
    if config.MUX_NOBUFFER_ENABLE:
        # Diagnostic only. With raw HEVC, nobuffer can drop packets before the
        # first muxed GOP and make video content lead audio by about one GOP.
        return "+genpts+nobuffer+flush_packets"
    return "+genpts"


def _pipe_ts_final_mux_fflags() -> str:
    # The split final mux needs generated timestamps, but nobuffer/flush_packets
    # shift live pipe/TCP input start PTS by about one second.
    return "+genpts"


def _mux_loglevel() -> str:
    return config.MUX_FFMPEG_LOGLEVEL or "warning"


def _mux_probe_args(
    probesize: str | None = None,
    *,
    for_raw_video: bool = False,
    analyzeduration_us: str | None = None,
) -> list[str]:
    if for_raw_video:
        args: list[str] = []
        if config.MUX_RAW_VIDEO_PROBESIZE:
            args.extend(["-probesize", config.MUX_RAW_VIDEO_PROBESIZE])
        if config.MUX_RAW_VIDEO_ANALYZEDURATION:
            args.extend(["-analyzeduration", config.MUX_RAW_VIDEO_ANALYZEDURATION])
        return args
    args: list[str] = []
    probe = config.MUX_PROBESIZE_OVERRIDE if probesize is None else probesize
    if probe:
        args.extend(["-probesize", str(probe)])
    analyze = config.MUX_ANALYZEDURATION_US if analyzeduration_us is None else analyzeduration_us
    if analyze:
        args.extend(["-analyzeduration", str(analyze)])
    return args


def _mux_intermediate_ts_probe_args() -> list[str]:
    """Probe controls for the pipe_ts final mux intermediate MPEG-TS stdin.

    The final mux must inspect the intermediate TS long enough to see HEVC
    VPS/SPS/PPS. Too-small values can make strict players classify the final
    stream as audio-only, so the default intentionally keeps FFmpeg defaults.
    """
    args: list[str] = []
    if config.MUX_INTERMEDIATE_TS_PROBESIZE:
        args.extend(["-probesize", config.MUX_INTERMEDIATE_TS_PROBESIZE])
    if config.MUX_INTERMEDIATE_TS_ANALYZEDURATION:
        args.extend(["-analyzeduration", config.MUX_INTERMEDIATE_TS_ANALYZEDURATION])
    return args


def _mpegts_tick_for_fps(fps: float) -> int:
    if abs(fps - (60000.0 / 1001.0)) < 0.001:
        return 1502
    return int((90000.0 / fps) + 0.5) if fps > 0 else 3000


def _mpegts_video_bsf(timestamp_filter: str | None = None) -> list[str]:
    filters: list[str] = []
    if PYNV_OUTPUT_CODEC == "hevc" and config.PASSTHROUGH_MPEGTS_HEVC_AUD and not timestamp_filter:
        filters.append("hevc_metadata=aud=insert")
    if timestamp_filter:
        filters.append(timestamp_filter)
    if not filters:
        return []
    return ["-bsf:v", ",".join(filters)]


def _slate_frame_pace_delay(
    *,
    fps: float,
    sent_frames: int,
    burst_frames: int,
    pace_start: float,
    now: float,
) -> float:
    """Return pre-send delay for the next slate frame.

    The first burst frames may be sent immediately to expose codec headers.
    Later frames are held until their nominal video PTS is due on the wall
    clock, keeping slate video aligned with the realtime silent AAC source.
    """
    if fps <= 0:
        return 0.0
    burst = max(0, int(burst_frames))
    sent = max(0, int(sent_frames))
    if sent < burst:
        return 0.0
    due = pace_start + (sent / fps)
    return max(0.0, due - now)


def _hevc_nal_summary(data: bytes | bytearray | memoryview, *, limit: int = 12) -> str:
    b = bytes(data)
    starts: list[tuple[int, int]] = []
    i = 0
    while i < len(b) - 3 and len(starts) < limit + 1:
        if b[i : i + 3] == b"\x00\x00\x01":
            starts.append((i, 3))
            i += 3
        elif i < len(b) - 4 and b[i : i + 4] == b"\x00\x00\x00\x01":
            starts.append((i, 4))
            i += 4
        else:
            i += 1
    parts: list[str] = []
    for idx, (pos, sc_len) in enumerate(starts[:limit]):
        nal_start = pos + sc_len
        if nal_start >= len(b):
            continue
        nal_end = starts[idx + 1][0] if idx + 1 < len(starts) else len(b)
        nal_type = (b[nal_start] >> 1) & 0x3F
        parts.append(f"{nal_type}:{nal_end - nal_start}")
    return ",".join(parts) if parts else "none"


def _mpegts_color_args(color_meta) -> list[str]:
    args = list(color_meta.ffmpeg_args())
    override = config.PASSTHROUGH_MPEGTS_COLOR_RANGE
    if override in {"tv", "pc", "mpeg", "jpeg"}:
        filtered: list[str] = []
        skip_next = False
        for item in args:
            if skip_next:
                skip_next = False
                continue
            if item == "-color_range":
                skip_next = True
                continue
            filtered.append(item)
        return ["-color_range", override, *filtered]
    return args


def _aac_output_args() -> list[str]:
    args = [
        "-c:a",
        "aac",
        "-ar",
        str(config.PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_RATE),
        "-ac",
        str(config.PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS),
    ]
    if config.PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE:
        args.extend(["-b:a", config.PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE])
    return args


def _aac_output_channel_layout() -> str:
    if config.PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS == 1:
        return "mono"
    if config.PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS == 2:
        return "stereo"
    return f"{config.PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS}c"


def _lock_for_audio_cache(key: str) -> threading.Lock:
    with _audio_cache_locks_guard:
        lock = _audio_cache_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _audio_cache_locks[key] = lock
            if len(_audio_cache_locks) > _AUDIO_CACHE_LOCK_MAX_ENTRIES:
                for old_key, old_lock in list(_audio_cache_locks.items()):
                    if len(_audio_cache_locks) <= _AUDIO_CACHE_LOCK_MAX_ENTRIES:
                        break
                    if old_key != key and not old_lock.locked():
                        _audio_cache_locks.pop(old_key, None)
        return lock


def _file_size_or_zero(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _cleanup_stale_audio_tmp_files() -> None:
    global _audio_tmp_cleanup_done
    if _audio_tmp_cleanup_done:
        return
    with _audio_tmp_cleanup_lock:
        if _audio_tmp_cleanup_done:
            return
        cache_dir = config.PASSTHROUGH_AUDIO_MPEGTS_CACHE_DIR
        removed = 0
        try:
            if cache_dir.exists():
                for stale in cache_dir.glob("*.tmp.aac"):
                    try:
                        stale.unlink()
                        removed += 1
                    except OSError:
                        pass
        except OSError as e:
            log.warning("stale audio tmp cleanup failed: dir=%s error=%s", cache_dir, e)
        if removed:
            log.info("stale audio tmp cleanup removed %d files from %s", removed, cache_dir)
        _audio_tmp_cleanup_done = True


class PyNvPassthroughStream:
    """PyNv decode -> GPU matting -> PyNv HEVC encode -> FFmpeg mux."""

    def __init__(
        self,
        src: Path,
        start_sec: float,
        matter: Matter,
        metadata: VideoProbeMetadata | None = None,
        container: str = "mp4",
        max_fps: float | None = None,
        audio_mode_override: str | None = None,
        output_mode: str | None = None,
    ):
        self.src = Path(src).resolve()
        self.path = self.src
        self.sid = next(_stream_ids)
        self.start_sec = max(0.0, float(start_sec))
        self.matter = matter
        self.metadata = metadata
        self.container = container.lower()
        self.max_fps = max_fps
        self.audio_mode_override = audio_mode_override
        self.output_mode = (output_mode or config.PASSTHROUGH_OUTPUT_MODE).lower()
        if self.output_mode == "all":
            self.output_mode = "green"
        _cleanup_stale_audio_tmp_files()
        self.bytes_emitted = 0
        self.frames_produced = 0
        self.output_fps = 0.0
        self._stop = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._mux: subprocess.Popen | None = None
        self._video_mux: subprocess.Popen | None = None
        self._audio_procs: set[subprocess.Popen] = set()
        self._audio_procs_lock = threading.Lock()
        self._encoder_lock = threading.Lock()
        self._dec: PyNvSimpleDecoder | PyNvThreadedSerialDecoder | None = None
        self._enc = None
        self._slate_audio_thread: threading.Thread | None = None
        self._slate_audio_cache_thread: threading.Thread | None = None
        self._slate_audio_ready = threading.Event()
        self._slate_direct_ready = threading.Event()
        self._slate_audio_failed = threading.Event()
        self._real_video_started = threading.Event()
        self._slate_audio_addr: tuple[str, int] | None = None
        self._slate_audio_cache_path: Path | None = None
        self._slate_audio_direct_only = False
        self._first_chunk_marks: dict[str, float] = {}
        self.startup_error: str | None = None

    def _mark_first(self, key: str) -> None:
        if not config.MUX_LATENCY_DIAG or key in self._first_chunk_marks:
            return
        now = time.perf_counter()
        self._first_chunk_marks[key] = now
        t0 = self._first_chunk_marks.get("T0_mux_spawn", now)
        log.info(
            "[DIAG][MUX][%d] mark key=%s delta_from_T0_ms=%.1f",
            self.sid,
            key,
            (now - t0) * 1000.0,
        )

    def _mark_first_write(self) -> None:
        self._mark_first("T1_first_write")

    def _log_first_chunk_breakdown(self) -> None:
        if not config.MUX_LATENCY_DIAG:
            return
        now = time.perf_counter()
        marks = self._first_chunk_marks
        t0 = marks.get("T0_mux_spawn", now)

        def delta_ms(key: str) -> float:
            mark = marks.get(key)
            if mark is None:
                return -1.0
            return (mark - t0) * 1000.0

        log.info(
            "[DIAG][MUX][%d] first_chunk_breakdown "
            "T0_video_spawn=%.1fms T0_spawn=0.0ms T1_write=%.1fms T2_stderr=%.1fms "
            "T2a_video=%.1fms T2b_final=%.1fms "
            "T3a_vcodec=%.1fms T3b_acodec=%.1fms T3c_output=%.1fms "
            "T4_reader=%.1fms total=%.1fms",
            self.sid,
            delta_ms("T0_video_mux_spawn"),
            delta_ms("T1_first_write"),
            delta_ms("T2_first_stderr"),
            delta_ms("T2a_video_first_stderr"),
            delta_ms("T2b_final_first_stderr"),
            delta_ms("T3a_final_video_codec"),
            delta_ms("T3b_final_audio_codec"),
            delta_ms("T3c_final_output_ready"),
            (now - t0) * 1000.0,
            (now - t0) * 1000.0,
        )

    def _register_audio_proc(self, proc: subprocess.Popen) -> subprocess.Popen:
        with self._audio_procs_lock:
            self._audio_procs.add(proc)
        return proc

    def _unregister_audio_proc(self, proc: subprocess.Popen) -> None:
        with self._audio_procs_lock:
            self._audio_procs.discard(proc)

    def _stop_proc(self, proc: subprocess.Popen, label: str, *, close_pipes: bool = True, wait_timeout: float = 1.0) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        if close_pipes:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
        try:
            if proc.poll() is None:
                proc.wait(timeout=wait_timeout)
        except Exception:
            try:
                if proc.poll() is None:
                    log.info("[PYNV][%d] killing %s pid=%s", self.sid, label, getattr(proc, "pid", None))
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=wait_timeout)
            except Exception:
                pass

    def _stop_audio_procs(self) -> None:
        with self._audio_procs_lock:
            procs = list(self._audio_procs)
        for proc in procs:
            self._stop_proc(proc, "audio ffmpeg", wait_timeout=0.5)

    def _log_vram(self, label: str) -> None:
        if not config.DEBUG_LOGS:
            return
        snapshot = _cupy_vram_snapshot()
        if snapshot is None:
            log.info("[PYNV][%d] VRAM %s: unavailable", self.sid, label)
            return
        used, total, pool_used, pool_total, pinned_free = snapshot
        log.info(
            "[PYNV][%d] VRAM %s: used=%.1f/%.1fMiB cupy_pool=%.1f/%.1fMiB pinned_free_blocks=%d",
            self.sid,
            label,
            _format_mib(used),
            _format_mib(total),
            _format_mib(pool_used),
            _format_mib(pool_total),
            pinned_free,
        )

    def _log_thread_stack(self, thread: threading.Thread) -> None:
        frame = sys._current_frames().get(thread.ident) if thread.ident is not None else None
        if frame is None:
            log.warning("[PYNV][%d] stuck thread stack unavailable: %s", self.sid, thread.name)
            return
        stack = "".join(traceback.format_stack(frame, limit=16))
        log.warning("[PYNV][%d] stuck thread stack name=%s\n%s", self.sid, thread.name, stack)

    @staticmethod
    def preflight(src: Path, metadata: VideoProbeMetadata | None = None) -> None:
        """Fail before HTTP headers if PyNv decoder or HEVC encoder cannot initialize."""
        import PyNvVideoCodec as nvc

        width = int(metadata.codec.width if metadata and metadata.codec.width > 0 else 0)
        height = int(metadata.codec.height if metadata and metadata.codec.height > 0 else 0)
        fps = float(metadata.timing.effective_fps(config.PASSTHROUGH_MAX_FPS) if metadata else 0.0)
        bitrate_estimate = effective_default_bitrate(src, PYNV_BACKEND_LABEL)
        bitrate = str(bitrate_estimate.bps)
        stat = stat_key(src)
        key = "|".join(
            [
                stat[0],
                str(stat[1]),
                str(stat[2]),
                str(width),
                str(height),
                f"{fps:.6f}",
                bitrate,
                str(config.PASSTHROUGH_GOP),
                str(config.PASSTHROUGH_HEVC_BF),
            ]
        )
        now = time.monotonic()
        with _preflight_lock:
            for old_key, expires_at in list(_preflight_ok.items()):
                if expires_at <= now:
                    _preflight_ok.pop(old_key, None)
            while len(_preflight_ok) > _PREFLIGHT_CACHE_MAX_ENTRIES:
                _preflight_ok.pop(next(iter(_preflight_ok)), None)
            if _preflight_ok.get(key, 0.0) > now:
                return

        bit_depth = int(metadata.codec.bit_depth if metadata and metadata.codec.bit_depth > 0 else 8)
        dec = PyNvSimpleDecoder(src, bit_depth=bit_depth)
        try:
            width = int(width or dec.info.width)
            height = int(height or dec.info.height)
            fps = float(fps or dec.info.fps or 30.0)
            enc = nvc.CreateEncoder(width, height, "NV12", False, **_pynv_encoder_kwargs(bitrate=bitrate, fps=f"{fps:.6f}"))
            try:
                end = getattr(enc, "EndEncode", None)
                if callable(end):
                    end()
            finally:
                del enc
                gc.collect()
        finally:
            dec.stop()
        with _preflight_lock:
            _preflight_ok[key] = time.monotonic() + _PREFLIGHT_TTL_SEC
            while len(_preflight_ok) > _PREFLIGHT_CACHE_MAX_ENTRIES:
                _preflight_ok.pop(next(iter(_preflight_ok)), None)

    @staticmethod
    def startup_preflight() -> None:
        """Pay process-level NVENC SDK initialization during startup."""
        if not config.NVENC_PREFLIGHT_ENABLE:
            warmup_event(log, phase="nvenc_preflight", enabled=False, status="disabled")
            return
        import PyNvVideoCodec as nvc

        state = get_startup_state()
        step_total = int(state.get("step_total") or 6)
        set_startup_phase(
            "warming",
            "warming NVENC encoder",
            step="nvenc_preflight",
            step_index=step_total,
            step_total=step_total,
            progress=max(0.0, min(0.99, (step_total - 0.2) / step_total)),
        )
        for width, height, fps_label, bitrate in config.NVENC_PREFLIGHT_GEOMETRIES:
            width = max(2, int(width) & ~1)
            height = max(2, int(height) & ~1)
            fps_label = str(fps_label)
            bitrate = str(bitrate)
            t0 = time.perf_counter()
            try:
                enc = nvc.CreateEncoder(
                    width,
                    height,
                    "NV12",
                    False,
                    **_pynv_encoder_kwargs(bitrate=bitrate, fps=fps_label),
                )
                try:
                    end = getattr(enc, "EndEncode", None)
                    if callable(end):
                        end()
                finally:
                    del enc
                    gc.collect()
                warmup_event(
                    log,
                    phase="nvenc_preflight",
                    status="ok",
                    geometry=[width, height],
                    fps=fps_label,
                    bitrate=int(bitrate),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                )
            except Exception:
                warmup_event(
                    log,
                    phase="nvenc_preflight",
                    status="failed",
                    geometry=[width, height],
                    fps=fps_label,
                    bitrate=int(bitrate),
                )
                log.warning(
                    "nvenc startup preflight failed geometry=%dx%d fps=%s bitrate=%s",
                    width,
                    height,
                    fps_label,
                    bitrate,
                    exc_info=True,
                )

    def _audio_mode(self, duration: float) -> str:
        if config.FORCE_AUDIO_OFF:
            log.info("[PYNV][%d] audio forced off by PT_FORCE_AUDIO_OFF", self.sid)
            return "off"
        configured = self.audio_mode_override
        if configured is None:
            configured = config.PASSTHROUGH_AUDIO_MPEGTS if self.container == "mpegts" else config.PASSTHROUGH_AUDIO
        audio_mode = str(configured or "off").lower()
        if audio_mode == "acc":
            log.warning("[PYNV][%d] passthrough audio mode 'acc' normalized to 'aac'", self.sid)
            audio_mode = "aac"
        if audio_mode not in {"copy", "aac", "off"}:
            log.warning("[PYNV][%d] unknown passthrough audio mode=%r; disabling audio", self.sid, audio_mode)
            audio_mode = "off"
        if audio_mode == "off":
            return "off"
        if duration <= 0.0:
            log.warning("[PYNV][%d] source duration unavailable; disabling audio mux", self.sid)
            return "off"
        return audio_mode

    def _cached_aac_input(self, audio_mode: str) -> Path | None:
        if (
            self.container != "mpegts"
            or audio_mode != "aac"
            or not config.PASSTHROUGH_AUDIO_MPEGTS_CACHE
        ):
            return None
        total_started = time.perf_counter()
        cache_key = fingerprint(self.src, length=20)
        cache_dir = config.PASSTHROUGH_AUDIO_MPEGTS_CACHE_DIR
        cache_path = cache_dir / f"{cache_key}.aac"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            log.info(
                "[PYNV][%d] audio cache hit: %s bytes=%d total_elapsed=%.3fs",
                self.sid, cache_path.name, cache_path.stat().st_size, time.perf_counter() - total_started,
            )
            return cache_path

        lock = _lock_for_audio_cache(cache_key)
        with lock:
            if self._stop.is_set():
                return None
            if cache_path.exists() and cache_path.stat().st_size > 0:
                log.info(
                    "[PYNV][%d] audio cache hit after wait: %s bytes=%d total_elapsed=%.3fs",
                    self.sid, cache_path.name, cache_path.stat().st_size, time.perf_counter() - total_started,
                )
                return cache_path
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".{os.getpid()}.{self.sid}.tmp.aac")
            cmd = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                _mux_loglevel(),
                "-y",
                "-probesize",
                "32768",
                "-analyzeduration",
                "0",
                "-vn",
                "-sn",
                "-dn",
                "-i",
                str(self.src),
                "-vn",
                "-sn",
                "-dn",
                "-map",
                "0:a:0?",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-c:a",
                "copy",
                "-f",
                "adts",
                str(tmp_path),
            ]
            log.info("[PYNV][%d] audio cache build cmd: %s", self.sid, " ".join(cmd))
            started = time.perf_counter()
            if self._stop.is_set():
                return None
            proc = self._register_audio_proc(subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                **hidden_subprocess_kwargs(),
            ))
            interrupted = False
            last_progress_log = started
            last_progress_size = 0
            while proc.poll() is None:
                if self._stop.wait(0.1):
                    interrupted = True
                    self._stop_proc(proc, "audio cache build", close_pipes=False)
                    break
                now = time.perf_counter()
                if config.DEBUG_LOGS and now - last_progress_log >= _AUDIO_CACHE_PROGRESS_INTERVAL:
                    current_size = _file_size_or_zero(tmp_path)
                    delta_size = current_size - last_progress_size
                    log.info(
                        "[PYNV][%d] audio cache building: tmp=%s bytes=%d delta=%d elapsed=%.2fs",
                        self.sid,
                        tmp_path.name,
                        current_size,
                        delta_size,
                        now - started,
                    )
                    last_progress_log = now
                    last_progress_size = current_size
            proc_exit_at = time.perf_counter()
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:
                self._stop_proc(proc, "audio cache build communicate", close_pipes=False)
                stdout, stderr = "", ""
            finally:
                self._unregister_audio_proc(proc)
            communicate_elapsed = time.perf_counter() - proc_exit_at
            elapsed = time.perf_counter() - started
            if interrupted or self._stop.is_set():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                log.info(
                    "[PYNV][%d] audio cache build interrupted; discarded tmp=%s elapsed=%.2fs total_elapsed=%.3fs",
                    self.sid,
                    tmp_path.name,
                    elapsed,
                    time.perf_counter() - total_started,
                )
                return None
            stat_started = time.perf_counter()
            tmp_size = _file_size_or_zero(tmp_path)
            stat_elapsed = time.perf_counter() - stat_started
            if proc.returncode != 0 or tmp_size <= 0:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                stderr = (stderr or "").strip()[-2000:]
                log.warning(
                    "[PYNV][%d] audio cache build failed rc=%s build_elapsed=%.2fs total_elapsed=%.3fs stderr=%s",
                    self.sid, proc.returncode, elapsed, time.perf_counter() - total_started, stderr,
                )
                return None
            replace_started = time.perf_counter()
            os.replace(tmp_path, cache_path)
            replace_elapsed = time.perf_counter() - replace_started
            cache_size = _file_size_or_zero(cache_path)
            log.info(
                "[PYNV][%d] audio cache built: %s bytes=%d ffmpeg_elapsed=%.2fs communicate_elapsed=%.3fs stat_elapsed=%.3fs replace_elapsed=%.3fs total_elapsed=%.3fs",
                self.sid,
                cache_path.name,
                cache_size,
                elapsed,
                communicate_elapsed,
                stat_elapsed,
                replace_elapsed,
                time.perf_counter() - total_started,
            )
            return cache_path

    def _aac_cache_path(self) -> tuple[str, Path]:
        cache_key = fingerprint(self.src, length=20)
        cache_dir = config.PASSTHROUGH_AUDIO_MPEGTS_CACHE_DIR
        return cache_key, cache_dir / f"{cache_key}.aac"

    def _start_aac_cache_build(self, cache_key: str, cache_path: Path) -> threading.Thread:
        thread = threading.Thread(
            target=self._build_aac_cache_worker,
            args=(cache_key, cache_path),
            name="pynv-audio-cache",
            daemon=True,
        )
        thread.start()
        return thread

    def _build_aac_cache_worker(self, cache_key: str, cache_path: Path) -> None:
        lock = _lock_for_audio_cache(cache_key)
        with lock:
            if self._stop.is_set():
                self._slate_audio_failed.set()
                return
            if cache_path.exists() and cache_path.stat().st_size > 0:
                self._slate_audio_cache_path = cache_path
                self._slate_audio_ready.set()
                return
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".{os.getpid()}.{self.sid}.tmp.aac")
            cmd = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                _mux_loglevel(),
                "-y",
                "-probesize",
                "32768",
                "-analyzeduration",
                "0",
                "-vn",
                "-sn",
                "-dn",
                "-i",
                str(self.src),
                "-vn",
                "-sn",
                "-dn",
                "-map",
                "0:a:0?",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-c:a",
                "copy",
                "-f",
                "adts",
                str(tmp_path),
            ]
            log.info("[PYNV][%d] slate audio cache build cmd: %s", self.sid, " ".join(cmd))
            started = time.perf_counter()
            proc = self._register_audio_proc(subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                **hidden_subprocess_kwargs(),
            ))
            interrupted = False
            last_progress_log = started
            last_progress_size = 0
            while proc.poll() is None:
                if self._stop.wait(0.1):
                    interrupted = True
                    self._stop_proc(proc, "slate audio cache build", close_pipes=False)
                    break
                now = time.perf_counter()
                if now - last_progress_log >= _AUDIO_CACHE_PROGRESS_INTERVAL:
                    current_size = _file_size_or_zero(tmp_path)
                    delta_size = current_size - last_progress_size
                    log.info(
                        "[PYNV][%d] slate audio cache building: tmp=%s bytes=%d delta=%d elapsed=%.2fs",
                        self.sid,
                        tmp_path.name,
                        current_size,
                        delta_size,
                        now - started,
                    )
                    last_progress_log = now
                    last_progress_size = current_size
            proc_exit_at = time.perf_counter()
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except Exception:
                self._stop_proc(proc, "slate audio cache build communicate", close_pipes=False)
                stdout, stderr = "", ""
            finally:
                self._unregister_audio_proc(proc)
            communicate_elapsed = time.perf_counter() - proc_exit_at
            elapsed = time.perf_counter() - started
            stat_started = time.perf_counter()
            tmp_size = _file_size_or_zero(tmp_path)
            stat_elapsed = time.perf_counter() - stat_started
            if interrupted or self._stop.is_set():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                log.info(
                    "[PYNV][%d] slate audio cache build interrupted; discarded tmp=%s elapsed=%.2fs",
                    self.sid, tmp_path.name, elapsed,
                )
                return
            if proc.returncode != 0 or tmp_size <= 0:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                stderr = (stderr or "").strip()[-2000:]
                log.warning(
                    "[PYNV][%d] slate audio cache build failed rc=%s elapsed=%.2fs stderr=%s",
                    self.sid, proc.returncode, elapsed, stderr,
                )
                self._slate_audio_failed.set()
                return
            replace_started = time.perf_counter()
            os.replace(tmp_path, cache_path)
            replace_elapsed = time.perf_counter() - replace_started
            cache_size = _file_size_or_zero(cache_path)
            self._slate_audio_cache_path = cache_path
            self._slate_audio_ready.set()
            log.info(
                "[PYNV][%d] slate audio cache ready: %s bytes=%d ffmpeg_elapsed=%.2fs communicate_elapsed=%.3fs stat_elapsed=%.3fs replace_elapsed=%.3fs total_elapsed=%.3fs",
                self.sid,
                cache_path.name,
                cache_size,
                elapsed,
                communicate_elapsed,
                stat_elapsed,
                replace_elapsed,
                time.perf_counter() - started,
            )

    def _start_slate_audio_server(self, cache_key: str, cache_path: Path, *, direct_only: bool = False) -> tuple[str, int]:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(0.25)
        addr = server.getsockname()
        self._slate_audio_addr = (str(addr[0]), int(addr[1]))
        self._slate_audio_direct_only = direct_only
        if direct_only:
            log.info("[PYNV][%d] slate audio direct-only mode; skipping full AAC cache build", self.sid)
        else:
            self._slate_audio_cache_thread = self._start_aac_cache_build(cache_key, cache_path)
        self._slate_audio_thread = threading.Thread(
            target=self._slate_audio_server_loop,
            args=(server,),
            name="pynv-slate-audio",
            daemon=True,
        )
        self._slate_audio_thread.start()
        log.info("[PYNV][%d] slate audio server listening: tcp://%s:%d", self.sid, self._slate_audio_addr[0], self._slate_audio_addr[1])
        return self._slate_audio_addr

    def _make_slate_nv12_gpu(self, width: int, height: int):
        import cupy as cp

        slate_rgb = (0, 0, 0) if self.output_mode == "alpha" else config.COMPOSITE_BG_RGB
        y, u, v = _rgb_to_limited_yuv(slate_rgb)
        frame = cp.empty((height + height // 2, width), dtype=cp.uint8)
        frame[:height, :].fill(y)
        uv = frame[height:, :].reshape(height // 2, width // 2, 2)
        uv[:, :, 0].fill(u)
        uv[:, :, 1].fill(v)
        return frame

    def _subtitle_kernels(self):
        global _SUBTITLE_BLEND_Y_KERNEL, _SUBTITLE_BLEND_UV_KERNEL
        if _SUBTITLE_BLEND_Y_KERNEL is not None and _SUBTITLE_BLEND_UV_KERNEL is not None:
            return _SUBTITLE_BLEND_Y_KERNEL, _SUBTITLE_BLEND_UV_KERNEL
        import cupy as cp

        _SUBTITLE_BLEND_Y_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void blend_rgba_y_to_nv12(
                unsigned char* frame,
                const unsigned char* rgba,
                int frame_w,
                int frame_h,
                int overlay_w,
                int overlay_h,
                int dst_x,
                int dst_y)
            {
                int x = blockDim.x * blockIdx.x + threadIdx.x;
                int y = blockDim.y * blockIdx.y + threadIdx.y;
                if (x >= overlay_w || y >= overlay_h) return;
                int fx = dst_x + x;
                int fy = dst_y + y;
                if (fx < 0 || fy < 0 || fx >= frame_w || fy >= frame_h) return;
                int oi = (y * overlay_w + x) * 4;
                float a = rgba[oi + 3] / 255.0f;
                if (a <= 0.0f) return;
                float r = rgba[oi + 0];
                float g = rgba[oi + 1];
                float b = rgba[oi + 2];
                float yy = 16.0f + (65.738f * r + 129.057f * g + 25.064f * b) / 256.0f;
                int yi = fy * frame_w + fx;
                frame[yi] = (unsigned char)(frame[yi] * (1.0f - a) + yy * a + 0.5f);
            }
            """,
            "blend_rgba_y_to_nv12",
        )
        _SUBTITLE_BLEND_UV_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void blend_rgba_uv_to_nv12(
                unsigned char* frame,
                const unsigned char* rgba,
                int frame_w,
                int frame_h,
                int overlay_w,
                int overlay_h,
                int dst_x,
                int dst_y)
            {
                int ux = blockDim.x * blockIdx.x + threadIdx.x;
                int uy = blockDim.y * blockIdx.y + threadIdx.y;
                int uv_w = (overlay_w + 1) / 2;
                int uv_h = (overlay_h + 1) / 2;
                if (ux >= uv_w || uy >= uv_h) return;
                int ox0 = ux * 2;
                int oy0 = uy * 2;
                int fx0 = dst_x + ox0;
                int fy0 = dst_y + oy0;
                int uv_fx = fx0 & ~1;
                int uv_fy = fy0 & ~1;
                if (uv_fx < 0 || uv_fy < 0 || uv_fx + 1 >= frame_w || uv_fy + 1 >= frame_h) return;

                float a_sum = 0.0f;
                float r_sum = 0.0f;
                float g_sum = 0.0f;
                float b_sum = 0.0f;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int ox = ox0 + dx;
                        int oy = oy0 + dy;
                        int fx = dst_x + ox;
                        int fy = dst_y + oy;
                        if (ox >= overlay_w || oy >= overlay_h || fx < 0 || fy < 0 || fx >= frame_w || fy >= frame_h) continue;
                        int oi = (oy * overlay_w + ox) * 4;
                        float a = rgba[oi + 3] / 255.0f;
                        if (a <= 0.0f) continue;
                        a_sum += a;
                        r_sum += rgba[oi + 0] * a;
                        g_sum += rgba[oi + 1] * a;
                        b_sum += rgba[oi + 2] * a;
                    }
                }
                if (a_sum <= 0.0f) return;
                float a = fminf(1.0f, a_sum / 4.0f);
                float r = r_sum / a_sum;
                float g = g_sum / a_sum;
                float b = b_sum / a_sum;
                float uu = 128.0f + (-37.945f * r - 74.494f * g + 112.439f * b) / 256.0f;
                float vv = 128.0f + (112.439f * r - 94.154f * g - 18.285f * b) / 256.0f;
                int uv_i = frame_w * frame_h + (uv_fy / 2) * frame_w + uv_fx;
                frame[uv_i] = (unsigned char)(frame[uv_i] * (1.0f - a) + uu * a + 0.5f);
                frame[uv_i + 1] = (unsigned char)(frame[uv_i + 1] * (1.0f - a) + vv * a + 0.5f);
            }
            """,
            "blend_rgba_uv_to_nv12",
        )
        return _SUBTITLE_BLEND_Y_KERNEL, _SUBTITLE_BLEND_UV_KERNEL

    def _subtitle_overlay_for_time(self, renderer: SubtitleRenderer | None, pts_sec: float):
        if renderer is None or not renderer.enabled:
            return None
        return renderer.overlay_for_time(pts_sec)

    def _subtitle_overlay_positions(self, out_nv12, renderer: SubtitleRenderer, overlay):
        rgba, left, top = overlay
        if rgba.size <= 0:
            return []
        h, w = int(out_nv12.shape[0] * 2 // 3), int(out_nv12.shape[1])
        del h
        eye_w = w // 2 if w >= 3000 else w
        mode = config.SUBTITLE_MODE
        if mode == "auto":
            mode = "dual" if w >= 3000 else "mono"
        if mode == "left":
            positions = [(left, top)]
        elif mode == "right":
            positions = [(eye_w + left, top)]
        elif mode == "dual":
            parallax = renderer.parallax_px()
            positions = [(left, top), (eye_w + left + parallax, top)]
        else:
            positions = [(max(0, (w - int(rgba.shape[1])) // 2), top)]
        return [(rgba, x, y) for x, y in positions]

    def _blend_subtitle_overlay(self, out_nv12, renderer: SubtitleRenderer, overlay) -> None:
        if overlay is None:
            return
        import cupy as cp

        rgba, left, top = overlay
        if rgba.size <= 0:
            return
        rgba_dev = cp.asarray(rgba)
        y_kernel, uv_kernel = self._subtitle_kernels()
        h, w = int(out_nv12.shape[0] * 2 // 3), int(out_nv12.shape[1])
        positions = [(x, y) for _rgba, x, y in self._subtitle_overlay_positions(out_nv12, renderer, overlay)]
        block = (16, 16)
        grid_y = ((int(rgba.shape[1]) + block[0] - 1) // block[0], (int(rgba.shape[0]) + block[1] - 1) // block[1])
        grid_uv = (((int(rgba.shape[1]) + 1) // 2 + block[0] - 1) // block[0], ((int(rgba.shape[0]) + 1) // 2 + block[1] - 1) // block[1])
        for x, y in positions:
            y_kernel(
                grid_y,
                block,
                (
                    out_nv12,
                    rgba_dev,
                    w,
                    h,
                    int(rgba.shape[1]),
                    int(rgba.shape[0]),
                    int(x),
                    int(y),
                ),
            )
            uv_kernel(
                grid_uv,
                block,
                (
                    out_nv12,
                    rgba_dev,
                    w,
                    h,
                    int(rgba.shape[1]),
                    int(rgba.shape[0]),
                    int(x),
                    int(y),
                ),
            )
        cp.cuda.get_current_stream().synchronize()

    def _blend_positioned_subtitle_overlays(self, out_nv12, positioned_overlays) -> None:
        if not positioned_overlays:
            return
        import cupy as cp

        y_kernel, uv_kernel = self._subtitle_kernels()
        h, w = int(out_nv12.shape[0] * 2 // 3), int(out_nv12.shape[1])
        block = (16, 16)
        for rgba, x, y in positioned_overlays:
            if rgba.size <= 0:
                continue
            rgba_dev = cp.asarray(rgba)
            overlay_h, overlay_w = int(rgba.shape[0]), int(rgba.shape[1])
            grid_y = ((overlay_w + block[0] - 1) // block[0], (overlay_h + block[1] - 1) // block[1])
            grid_uv = (((overlay_w + 1) // 2 + block[0] - 1) // block[0], ((overlay_h + 1) // 2 + block[1] - 1) // block[1])
            y_kernel(grid_y, block, (out_nv12, rgba_dev, w, h, overlay_w, overlay_h, int(x), int(y)))
            uv_kernel(grid_uv, block, (out_nv12, rgba_dev, w, h, overlay_w, overlay_h, int(x), int(y)))
        cp.cuda.get_current_stream().synchronize()

    def _apply_subtitle_overlay(self, out_nv12, renderer: SubtitleRenderer | None, pts_sec: float) -> None:
        overlay = self._subtitle_overlay_for_time(renderer, pts_sec)
        if renderer is None or overlay is None:
            return
        self._blend_subtitle_overlay(out_nv12, renderer, overlay)

    def _slate_audio_server_loop(self, server: socket.socket) -> None:
        conn: socket.socket | None = None
        silence_proc: subprocess.Popen | None = None
        try:
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            silence_cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                _mux_loglevel(),
                "-re",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={config.PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_RATE}:cl={_aac_output_channel_layout()}",
                *_aac_output_args(),
                "-f",
                "adts",
                "-",
            ]
            silence_proc = self._register_audio_proc(subprocess.Popen(
                silence_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **hidden_subprocess_kwargs(),
            ))
            assert silence_proc.stdout is not None
            log.info("[PYNV][%d] slate audio silence cmd: %s", self.sid, " ".join(silence_cmd))
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                    break
                except socket.timeout:
                    continue
            if conn is None:
                return
            conn.settimeout(0.25)
            direct_after = 0.0 if self._slate_audio_direct_only else float(config.PASSTHROUGH_AUDIO_MPEGTS_SLATE_DIRECT_AFTER)
            direct_deadline = time.perf_counter() + direct_after if direct_after > 0.0 else time.perf_counter()
            audio_source_ready = False
            waiting_real_video_logged = False
            while not self._stop.is_set() and not self._slate_audio_failed.is_set():
                if not audio_source_ready and self._slate_audio_ready.is_set():
                    audio_source_ready = True
                    log.info("[PYNV][%d] slate audio source ready; waiting for real video start", self.sid)
                if not audio_source_ready and time.perf_counter() >= direct_deadline:
                    self._slate_direct_ready.set()
                    audio_source_ready = True
                    log.info("[PYNV][%d] slate audio direct deadline reached; waiting for real video start", self.sid)
                if audio_source_ready and self._real_video_started.is_set():
                    break
                if audio_source_ready and not waiting_real_video_logged:
                    waiting_real_video_logged = True
                    log.info("[PYNV][%d] slate audio keeps silence until first real video bitstream", self.sid)
                data = silence_proc.stdout.read(4096)
                if not data:
                    if silence_proc.poll() is not None:
                        log.warning("[PYNV][%d] slate audio silence generator exited rc=%s", self.sid, silence_proc.returncode)
                        self._slate_audio_failed.set()
                        return
                    continue
                try:
                    conn.sendall(data)
                except OSError:
                    return
            self._stop_proc(silence_proc, "slate audio silence")
            self._unregister_audio_proc(silence_proc)
            silence_proc = None
            if self._stop.is_set() or self._slate_audio_failed.is_set():
                return
            cache_path = self._slate_audio_cache_path
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            if cache_path is not None:
                log.info("[PYNV][%d] slate audio switching to cached AAC: %s", self.sid, cache_path.name)
                real_cmd = [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    _mux_loglevel(),
                    "-f",
                    "aac",
                    *(["-ss", f"{self.start_sec:.3f}"] if self.start_sec > 0.001 else []),
                    "-i",
                    str(cache_path),
                    "-c:a",
                    "copy",
                    "-f",
                    "adts",
                    "-",
                ]
                log_label = "slate audio cache stream cmd"
            else:
                log.info(
                    "[PYNV][%d] slate audio switching to direct source demux after %.2fs while full cache continues",
                    self.sid,
                    direct_after,
                )
                real_cmd = [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    _mux_loglevel(),
                    "-probesize",
                    "32768",
                    "-analyzeduration",
                    "0",
                    *(["-ss", f"{self.start_sec:.3f}"] if self.start_sec > 0.001 else []),
                    "-vn",
                    "-sn",
                    "-dn",
                    "-i",
                    str(self.src),
                    "-vn",
                    "-sn",
                    "-dn",
                    "-map",
                    "0:a:0?",
                    "-c:a",
                    "copy",
                    "-f",
                    "adts",
                    "-",
                ]
                log_label = "slate audio direct stream cmd"
            real_proc = self._register_audio_proc(subprocess.Popen(
                real_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **hidden_subprocess_kwargs(),
            ))
            assert real_proc.stdout is not None
            log.info("[PYNV][%d] %s: %s", self.sid, log_label, " ".join(real_cmd))
            try:
                while not self._stop.is_set():
                    data = real_proc.stdout.read(64 * 1024)
                    if not data:
                        break
                    try:
                        conn.sendall(data)
                    except OSError:
                        break
            finally:
                self._stop_proc(real_proc, log_label)
                self._unregister_audio_proc(real_proc)
            log.info("[PYNV][%d] slate audio server done", self.sid)
        finally:
            try:
                if conn is not None:
                    conn.close()
            except OSError:
                pass
            try:
                server.close()
            except OSError:
                pass
            if silence_proc is not None:
                self._stop_proc(silence_proc, "slate audio silence finally")
                self._unregister_audio_proc(silence_proc)

    def _open_pipe_ts_muxer(
        self,
        fps: float,
        duration: float,
        audio_input: Path,
        *,
        audio_input_format: str | None = "aac",
        audio_label: str = "aac",
    ) -> subprocess.Popen:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        color_args = _mpegts_color_args(
            self.metadata.color if self.metadata is not None else probe_color_metadata(self.src)
        )
        input_format = "hevc" if PYNV_OUTPUT_CODEC == "hevc" else "h264"
        audio_probe = (
            config.MUX_AUDIO_PROBESIZE_OVERRIDE
            if audio_input_format == "aac"
            else config.MUX_CONTAINER_PROBESIZE_OVERRIDE
        )
        video_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            _mux_loglevel(),
            "-fflags",
            _mux_fflags(),
            "-thread_queue_size",
            str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE or 1024),
            *_mux_probe_args(for_raw_video=True),
            "-f",
            input_format,
            *(["-raw_packet_size", str(config.PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE)] if config.PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE > 0 else []),
            "-framerate",
            f"{fps:.6f}",
            "-i",
            "-",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            *_mpegts_video_bsf(f"setts=time_base=1/90000:pts=N*{_mpegts_tick_for_fps(fps)}:dts=N*{_mpegts_tick_for_fps(fps)}"),
            *color_args,
            "-flush_packets",
            "1",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-mpegts_flags",
            _mpegts_flags(),
            "-pat_period",
            "0.1",
            "-sdt_period",
            "0.5",
            "-pcr_period",
            "20",
            "-f",
            "mpegts",
            "-",
        ]
        final_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            _mux_loglevel(),
            "-fflags",
            _pipe_ts_final_mux_fflags(),
            "-thread_queue_size",
            str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE or 1024),
            *_mux_intermediate_ts_probe_args(),
            "-f",
            "mpegts",
            "-i",
            "-",
            "-thread_queue_size",
            str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE or 1024),
            *_mux_probe_args(audio_probe),
            *(["-f", audio_input_format] if audio_input_format else []),
            *(["-ss", f"{self.start_sec:.3f}"] if self.start_sec > 0.001 else []),
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_input),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            *_aac_output_args(),
            "-t",
            f"{duration:.3f}",
            *color_args,
        ]
        if config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA:
            final_cmd.extend(["-max_interleave_delta", config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA])
        final_cmd.extend(
            [
                "-flush_packets",
                "1",
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-mpegts_flags",
                _mpegts_flags(),
                "-pat_period",
                "0.1",
                "-sdt_period",
                "0.5",
                "-pcr_period",
                "20",
                "-f",
                "mpegts",
                "-",
            ]
        )
        log.info("[PYNV][%d] pipe_ts video mux cmd: %s", self.sid, " ".join(video_cmd))
        log.info("[PYNV][%d] pipe_ts final mux cmd: %s audio=%s duration=%.3f", self.sid, " ".join(final_cmd), audio_label, duration)
        video_proc = subprocess.Popen(
            video_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
        self._mark_first("T0_video_mux_spawn")
        final_proc = None
        try:
            assert video_proc.stdout is not None
            final_proc = subprocess.Popen(
                final_cmd,
                stdin=video_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **hidden_subprocess_kwargs(),
            )
            self._mark_first("T0_mux_spawn")
            video_proc.stdout.close()
        except BaseException:
            if final_proc is not None:
                self._stop_proc(final_proc, "pipe_ts final mux fallback", wait_timeout=0.2)
            self._stop_proc(video_proc, "pipe_ts video mux fallback", wait_timeout=0.2)
            raise
        self._video_mux = video_proc
        return final_proc

    def _open_slate_pipe_ts_muxer(self, fps: float, duration: float, audio_addr: tuple[str, int]) -> subprocess.Popen:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        color_args = _mpegts_color_args(
            self.metadata.color if self.metadata is not None else probe_color_metadata(self.src)
        )
        input_format = "hevc" if PYNV_OUTPUT_CODEC == "hevc" else "h264"
        video_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            _mux_loglevel(),
            "-fflags",
            _mux_fflags(),
            "-thread_queue_size",
            str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE or 1024),
            *_mux_probe_args(for_raw_video=True),
            "-f",
            input_format,
            *(["-raw_packet_size", str(config.PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE)] if config.PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE > 0 else []),
            "-framerate",
            f"{fps:.6f}",
            "-i",
            "-",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            *_mpegts_video_bsf(f"setts=time_base=1/90000:pts=N*{_mpegts_tick_for_fps(fps)}:dts=N*{_mpegts_tick_for_fps(fps)}"),
            *color_args,
            "-flush_packets",
            "1",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-mpegts_flags",
            _mpegts_flags(),
            "-pat_period",
            "0.1",
            "-sdt_period",
            "0.5",
            "-pcr_period",
            "20",
            "-f",
            "mpegts",
            "-",
        ]
        final_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            _mux_loglevel(),
            "-fflags",
            _pipe_ts_final_mux_fflags(),
            "-thread_queue_size",
            str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE or 1024),
            *_mux_probe_args(config.MUX_AUDIO_PROBESIZE_OVERRIDE),
            "-f",
            "aac",
            "-i",
            f"tcp://{audio_addr[0]}:{audio_addr[1]}",
            "-thread_queue_size",
            str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE or 1024),
            *_mux_intermediate_ts_probe_args(),
            "-f",
            "mpegts",
            "-i",
            "-",
            "-map",
            "1:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
            *_aac_output_args(),
            "-t",
            f"{duration:.3f}",
            *color_args,
        ]
        if config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA:
            final_cmd.extend(["-max_interleave_delta", config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA])
        final_cmd.extend(
            [
                "-flush_packets",
                "1",
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-mpegts_flags",
                _mpegts_flags(),
                "-pat_period",
                "0.1",
                "-sdt_period",
                "0.5",
                "-pcr_period",
                "20",
                "-f",
                "mpegts",
                "-",
            ]
        )
        log.info("[PYNV][%d] slate pipe_ts video mux cmd: %s", self.sid, " ".join(video_cmd))
        log.info("[PYNV][%d] slate pipe_ts final mux cmd: %s audio=tcp-aac duration=%.3f", self.sid, " ".join(final_cmd), duration)
        video_proc = subprocess.Popen(
            video_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
        self._mark_first("T0_video_mux_spawn")
        final_proc = None
        try:
            assert video_proc.stdout is not None
            final_proc = subprocess.Popen(
                final_cmd,
                stdin=video_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **hidden_subprocess_kwargs(),
            )
            self._mark_first("T0_mux_spawn")
            video_proc.stdout.close()
        except BaseException:
            if final_proc is not None:
                self._stop_proc(final_proc, "slate pipe_ts final mux fallback", wait_timeout=0.2)
            self._stop_proc(video_proc, "slate pipe_ts video mux fallback", wait_timeout=0.2)
            raise
        self._video_mux = video_proc
        return final_proc

    def _open_muxer(self, fps: float, duration: float) -> subprocess.Popen:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        if self.container == "mpegts":
            mux_args = [
                "-flush_packets",
                "1",
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-mpegts_flags",
                _mpegts_flags(),
                "-pat_period",
                "0.1",
                "-sdt_period",
                "0.5",
                "-pcr_period",
                "20",
                "-f",
                "mpegts",
            ]
            if config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA:
                mux_args = [
                    "-max_interleave_delta",
                    config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA,
                    *mux_args,
                ]
        else:
            mux_args = [
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                "-frag_duration",
                str(config.PASSTHROUGH_FMP4_FRAG_DURATION_US),
                "-f",
                "mp4",
            ]
        audio_mode = self._audio_mode(duration)
        base_args = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            _mux_loglevel(),
            "-fflags",
            _mux_fflags(),
        ]
        input_format = "hevc" if PYNV_OUTPUT_CODEC == "hevc" else "h264"
        if audio_mode == "off":
            queue_args = (
                ["-thread_queue_size", str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE)]
                if self.container == "mpegts" and config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE > 0
                else []
            )
            input_args = [
                *queue_args,
                *_mux_probe_args(for_raw_video=True),
                "-f",
                input_format,
                "-framerate",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
            ]
            map_args: list[str] = []
        else:
            timestamp_mode = (
                config.PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE
                if self.container == "mpegts"
                else "wallclock"
            )
            if timestamp_mode not in {"wallclock", "demux", "setts", "pipe_ts"}:
                log.warning(
                    "[PYNV][%d] unknown MPEG-TS audio timestamp mode=%r; using wallclock",
                    self.sid, timestamp_mode,
                )
                timestamp_mode = "wallclock"
            audio_input = None
            slate_audio_addr: tuple[str, int] | None = None
            if (
                timestamp_mode == "pipe_ts"
                and self.container == "mpegts"
                and audio_mode == "aac"
                and not config.PASSTHROUGH_AUDIO_MPEGTS_CACHE
            ):
                log.info("[PYNV][%d] audio cache disabled; using source audio in pipe_ts final mux", self.sid)
                return self._open_pipe_ts_muxer(
                    fps,
                    duration,
                    self.src,
                    audio_input_format=None,
                    audio_label="source",
                )
            if (
                timestamp_mode == "pipe_ts"
                and self.container == "mpegts"
                and audio_mode == "aac"
                and config.PASSTHROUGH_MPEGTS_VIDEO_SLATE
            ):
                cache_key, cache_path = self._aac_cache_path()
                if (
                    config.PASSTHROUGH_AUDIO_MPEGTS_CACHE
                    and cache_path.exists()
                    and cache_path.stat().st_size > 0
                ):
                    audio_input = cache_path
                    log.info("[PYNV][%d] audio cache hit: %s bytes=%d total_elapsed=0.000s", self.sid, cache_path.name, cache_path.stat().st_size)
                else:
                    direct_only = self.output_mode in {"alpha", "two_dvr"}
                    slate_audio_addr = self._start_slate_audio_server(cache_key, cache_path, direct_only=direct_only)
            else:
                audio_input = self._cached_aac_input(audio_mode)
            if timestamp_mode == "pipe_ts" and audio_input is not None:
                return self._open_pipe_ts_muxer(fps, duration, audio_input)
            if timestamp_mode == "pipe_ts" and slate_audio_addr is not None:
                return self._open_slate_pipe_ts_muxer(fps, duration, slate_audio_addr)
            if timestamp_mode == "pipe_ts":
                log.warning("[PYNV][%d] pipe_ts audio source unavailable; falling back to setts", self.sid)
                timestamp_mode = "setts"
            seek_args = ["-ss", f"{self.start_sec:.3f}"] if self.start_sec > 0.001 else []
            queue_args = (
                ["-thread_queue_size", str(config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE)]
                if self.container == "mpegts" and config.PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE > 0
                else []
            )
            raw_timestamp_args = (
                ["-use_wallclock_as_timestamps", "1"]
                if self.container == "mpegts" and timestamp_mode == "wallclock"
                else []
            )
            audio_rate_args = (
                ["-readrate", f"{config.PASSTHROUGH_AUDIO_MPEGTS_READRATE:.6f}"]
                if self.container == "mpegts" and config.PASSTHROUGH_AUDIO_MPEGTS_READRATE > 0
                else []
            )
            input_args = [
                *raw_timestamp_args,
                *queue_args,
                *_mux_probe_args(for_raw_video=True),
                "-f",
                input_format,
                *(["-raw_packet_size", str(config.PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE)] if self.container == "mpegts" and config.PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE > 0 else []),
                "-framerate",
                f"{fps:.6f}",
                "-i",
                "-",
                *seek_args,
                *audio_rate_args,
                *queue_args,
                *_mux_probe_args(
                    config.MUX_AUDIO_PROBESIZE_OVERRIDE
                    if audio_input is not None
                    else config.MUX_CONTAINER_PROBESIZE_OVERRIDE
                ),
                *(["-f", "aac"] if audio_input is not None else []),
                "-t",
                f"{duration:.3f}",
                "-i",
                str(audio_input or self.src),
            ]
            map_args = [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-t",
                f"{duration:.3f}",
            ]
            if self.container == "mpegts" and audio_mode == "aac":
                map_args.extend(_aac_output_args())
            else:
                map_args.extend(["-c:a", "copy" if audio_mode == "copy" or audio_input is not None else "aac"])
                if audio_mode == "aac" and audio_input is None and config.PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE:
                    map_args.extend(["-b:a", config.PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE])
            if self.container == "mpegts" and timestamp_mode == "setts":
                tick = _mpegts_tick_for_fps(fps)
                map_args.extend(_mpegts_video_bsf(f"setts=time_base=1/90000:pts=N*{tick}:dts=N*{tick}"))
            elif self.container == "mpegts":
                map_args.extend(_mpegts_video_bsf())
        cmd = [
            *base_args,
            *input_args,
            *map_args,
            "-c:v",
            "copy",
            *(
                _mpegts_color_args(self.metadata.color if self.metadata is not None else probe_color_metadata(self.src))
                if self.container == "mpegts"
                else (self.metadata.color if self.metadata is not None else probe_color_metadata(self.src)).ffmpeg_args()
            ),
            *mux_args,
            "-",
        ]
        log.info(
            "[PYNV][%d] mux cmd: %s audio=%s duration=%.3f",
            self.sid, " ".join(cmd), audio_mode, duration,
        )
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
        self._mark_first("T0_mux_spawn")
        return proc

    def _stderr_loop(self) -> None:
        proc = self._mux
        if not proc or not proc.stderr:
            return
        self._drain_stderr(proc, "ffmpeg")

    def _video_stderr_loop(self) -> None:
        proc = self._video_mux
        if not proc or not proc.stderr:
            return
        self._drain_stderr(proc, "ffmpeg-video")

    def _drain_stderr(self, proc: subprocess.Popen, label: str) -> None:
        if label == "ffmpeg-video":
            first_key = "T2a_video_first_stderr"
        elif label == "ffmpeg":
            first_key = "T2b_final_first_stderr"
        else:
            first_key = None
        enable_stage_marks = label == "ffmpeg" and config.MUX_LATENCY_DIAG_VERBOSE
        try:
            assert proc.stderr is not None
            for raw in iter(proc.stderr.readline, b""):
                if self._stop.is_set():
                    break
                text = raw.decode("utf-8", "replace").strip()
                if text:
                    if first_key:
                        self._mark_first(first_key)
                        self._mark_first("T2_first_stderr")
                    if enable_stage_marks:
                        if "Stream #" in text and ("Video: hevc" in text or "Video: h264" in text):
                            self._mark_first("T3a_final_video_codec")
                        elif "Stream #" in text and "Audio: " in text:
                            self._mark_first("T3b_final_audio_codec")
                        elif text.startswith("Output #0") or "Output #0," in text:
                            self._mark_first("T3c_final_output_ready")
                    if any(marker in text for marker in _BENIGN_FFMPEG_WARNINGS):
                        log.debug("[PYNV][%d][%s] %s", self.sid, label, text)
                    else:
                        log.warning("[PYNV][%d][%s] %s", self.sid, label, text)
        except (ValueError, OSError):
            pass

    def _reader_loop(self) -> None:
        assert self._mux and self._mux.stdout and self._queue is not None and self._loop is not None
        chunks = 0
        bytes_read = 0
        try:
            while not self._stop.is_set():
                try:
                    data = self._mux.stdout.read(_READ_CHUNK)
                except (ValueError, OSError):
                    break
                if not data:
                    break
                bytes_read += len(data)
                if chunks == 0:
                    log.info("[PYNV][%d] reader first stdout chunk len=%d", self.sid, len(data))
                    self._mark_first("T4_reader")
                    self._log_first_chunk_breakdown()
                if not self._put_chunk_from_thread(data, chunks, bytes_read):
                    return
                chunks += 1
                if chunks % 512 == 0:
                    log.debug("[PYNV][%d] reader progress chunks=%d bytes=%d", self.sid, chunks, bytes_read)
        except Exception as e:
            log.error("[PYNV][%d] reader exception: %s\n%s", self.sid, e, traceback.format_exc(limit=4))
        finally:
            self._post_sentinel()
            log.info("[PYNV][%d] reader done chunks=%d bytes=%d stop=%s", self.sid, chunks, bytes_read, self._stop.is_set())

    def _put_chunk_from_thread(self, data: bytes, chunks: int, bytes_read: int) -> bool:
        """Put one reader chunk into the asyncio queue without creating a coroutine task."""
        if self._queue is None or self._loop is None or self._loop.is_closed():
            return False
        done = threading.Event()
        state = {"ok": False, "failed": False}
        queue_ref = self._queue

        def try_put() -> None:
            if self._stop.is_set():
                state["failed"] = True
                done.set()
                return
            try:
                queue_ref.put_nowait(data)
                state["ok"] = True
                done.set()
            except asyncio.QueueFull:
                try:
                    self._loop.call_later(0.01, try_put)
                except RuntimeError:
                    state["failed"] = True
                    done.set()

        try:
            self._loop.call_soon_threadsafe(try_put)
        except RuntimeError:
            return False
        wait_started = time.perf_counter()
        next_wait_log_at = 5.0
        wait_log_interval = 10.0
        while not self._stop.is_set():
            if done.wait(timeout=0.5):
                waited = time.perf_counter() - wait_started
                if config.DEBUG_LOGS and waited >= 1.0:
                    log.warning(
                        "[PYNV][%d] reader async queue recovered after %.3fs chunks=%d bytes=%d",
                        self.sid,
                        waited,
                        chunks,
                        bytes_read,
                    )
                return bool(state["ok"])
            if config.DEBUG_LOGS:
                waited = time.perf_counter() - wait_started
                if waited >= next_wait_log_at:
                    log.warning(
                        "[PYNV][%d] reader waiting for async queue chunks=%d bytes=%d waited=%.3fs",
                        self.sid,
                        chunks,
                        bytes_read,
                        waited,
                    )
                    next_wait_log_at += wait_log_interval
        return False

    def _worker_loop(self) -> None:
        log.info("[PYNV][%d] worker start src=%s path=%s start=%.3f", self.sid, self.src.name, self.src, self.start_sec)
        self._log_vram("worker_start")
        if self.output_mode == "two_dvr":
            self._worker_loop_two_dvr()
            return
        pending_nv12_slots: list[object] = []
        slate_stop = threading.Event()
        slate_thread: threading.Thread | None = None
        try:
            import PyNvVideoCodec as nvc

            codec_meta = self.metadata.codec if self.metadata is not None else None
            bit_depth = int(codec_meta.bit_depth if codec_meta and codec_meta.bit_depth > 0 else 8)
            meta_dec = PyNvSimpleDecoder(self.src, bit_depth=bit_depth)
            self._log_vram("decoder_metadata_created")
            info = meta_dec.info
            dec_len = len(meta_dec)
            timing = self.metadata.timing if self.metadata is not None else probe_timing_metadata(self.src)
            source_fps = float(timing.source_fps or info.fps or 30.0)
            fps_cap = config.PASSTHROUGH_MAX_FPS if self.max_fps is None else float(self.max_fps)
            fps = float(timing.effective_fps(fps_cap))
            producer_pacing = bool(config.PASSTHROUGH_PRODUCER_REALTIME_PACING or fps_cap > 0)
            out_w, out_h = self.matter.pynv_scaled_size(info.width, info.height)
            alpha_projection_mode = ""
            alpha_process_w, alpha_process_h = out_w, out_h
            if self.output_mode == "alpha":
                from pipeline.alpha_packer import AlphaPacker, alpha_output_size

                alpha_projection_mode = AlphaPacker.projection_mode_static(alpha_process_w, alpha_process_h)
                out_w, out_h = alpha_output_size(alpha_process_w, alpha_process_h)
            self.output_fps = fps
            if not timing.is_cfr:
                meta_dec.stop()
                raise RuntimeError("PyNv production stream requires strong CFR source")
            start_out = int(round(self.start_sec * fps))
            target = max(0, int((timing.duration or info.duration or 0.0) * fps) - start_out)
            max_target = int((dec_len - 1) * fps / source_fps) + 1 if source_fps > 0 else dec_len
            target = min(target, max(1, max_target))
            initial_src_idx = min(dec_len - 1, cfr_source_index(start_out, source_fps, fps))
            decoder_mode = config.PASSTHROUGH_PYNV_DECODER
            if (
                self.output_mode == "alpha"
                and decoder_mode == "threaded_serial"
                and not config.PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER
            ):
                log.warning(
                    "[PYNV][%d] alpha output uses decoder=simple; "
                    "threaded_serial is disabled for alpha by PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=0",
                    self.sid,
                )
                decoder_mode = "simple"
            alpha_threaded_owned_copy = self.output_mode == "alpha" and decoder_mode == "threaded_serial"
            if decoder_mode == "threaded_serial":
                meta_dec.stop()
                self._dec = PyNvThreadedSerialDecoder(
                    self.src,
                    bit_depth=bit_depth,
                    start_frame=initial_src_idx,
                    batch_size=config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
                    buffer_size=config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
                    info=info,
                    num_frames=dec_len,
                )
                log.info(
                    "[PYNV][%d] decoder created mode=threaded_serial start_frame=%d batch=%d buffer=%d",
                    self.sid,
                    initial_src_idx,
                    config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
                    config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
                )
            elif decoder_mode == "simple":
                self._dec = meta_dec
                log.info("[PYNV][%d] decoder created mode=simple", self.sid)
            else:
                meta_dec.stop()
                raise RuntimeError(f"unknown PT_PASSTHROUGH_PYNV_DECODER={decoder_mode!r}")
            self._log_vram("decoder_created")
            if self.metadata is not None:
                codec_meta = self.metadata.codec
                color_meta = self.metadata.color
                log.info(
                    "[PYNV][%d] source meta: codec=%s profile=%s pix_fmt=%s bit_depth=%d level=%s "
                    "size=%dx%d process_size=%dx%d output_size=%dx%d source_fps=%.3f output_fps=%.3f "
                    "fps_cap=%.3f duration=%.3f frames=%d color=%s/%s/%s/%s container=%s output_mode=%s",
                    self.sid,
                    codec_meta.codec_name,
                    codec_meta.profile,
                    codec_meta.pix_fmt,
                    codec_meta.bit_depth,
                    codec_meta.level,
                    info.width,
                    info.height,
                    alpha_process_w,
                    alpha_process_h,
                    out_w,
                    out_h,
                    source_fps,
                    fps,
                    fps_cap,
                    timing.duration or info.duration or 0.0,
                    dec_len,
                    color_meta.color_range,
                    color_meta.color_space,
                    color_meta.color_transfer,
                    color_meta.color_primaries,
                    self.container,
                    self.output_mode,
                )
            log.info(
                "[PYNV][%d] runtime config: alpha_stride=%d alpha_mode=%s model=%s "
                "decoder=%s worker_mode=%s output_codec=%s producer_realtime_pacing=%s "
                "send_realtime_pacing=%s max_fps=%.3f output_fps=%.3f rvm_bypass_alpha=%s "
                "nv12_slots=%d",
                self.sid,
                config.ALPHA_STRIDE,
                config.ALPHA_MODE,
                config.MODEL_PATH,
                decoder_mode,
                config.PASSTHROUGH_PYNV_WORKER_MODE,
                PYNV_OUTPUT_CODEC,
                producer_pacing,
                config.PASSTHROUGH_SEND_REALTIME_PACING,
                fps_cap,
                fps,
                config.PASSTHROUGH_RVM_BYPASS_ALPHA,
                config.PASSTHROUGH_NV12_RING_SLOTS,
            )
            if alpha_threaded_owned_copy:
                log.info(
                    "[PYNV][%d] alpha decoder detail: effective_decoder=%s batch=%d buffer=%d owned_copy=%s allow_threaded=%s",
                    self.sid,
                    decoder_mode,
                    config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
                    config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
                    alpha_threaded_owned_copy,
                    config.PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER,
                )
            elif self.output_mode == "alpha":
                log.info(
                    "[PYNV][%d] alpha decoder detail: effective_decoder=%s batch=%d buffer=%d owned_copy=%s allow_threaded=%s",
                    self.sid,
                    decoder_mode,
                    config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
                    config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
                    alpha_threaded_owned_copy,
                    config.PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER,
                )
            subtitle_renderer: SubtitleRenderer | None = None
            log.info(
                "[PYNV][%d] subtitle lookup config: enabled=%s exts=%s",
                self.sid,
                config.SUBTITLE_ENABLE,
                config.SUBTITLE_EXTS,
            )
            subtitle_path = find_subtitle_for_video(self.src)
            if subtitle_path is not None:
                try:
                    subtitle_renderer = SubtitleRenderer(subtitle_path, out_w, out_h)
                    if not subtitle_renderer.enabled:
                        subtitle_renderer = None
                except Exception as e:
                    subtitle_renderer = None
                    log.warning("[PYNV][%d] subtitle load failed: %s error=%s", self.sid, subtitle_path.name, e)
            bitrate_estimate = effective_default_bitrate(self.src, PYNV_BACKEND_LABEL)
            bitrate = str(bitrate_estimate.bps)
            enc_kwargs = _pynv_encoder_kwargs(bitrate=bitrate, fps=f"{fps:.6f}")
            self._enc = nvc.CreateEncoder(out_w, out_h, "NV12", False, **enc_kwargs)
            log.info(
                "[PYNV][%d] encoder created %dx%d source=%dx%d fps=%.3f target=%d kwargs=%s source_bitrate=%s configured_bitrate=%s",
                self.sid,
                out_w,
                out_h,
                info.width,
                info.height,
                fps,
                target,
                enc_kwargs,
                bitrate_estimate.source,
                config.PASSTHROUGH_HEVC_BITRATE,
            )
            self._log_vram("encoder_created")
            if bit_depth > 8:
                log.info(
                    "[PYNV][%d] experimental 10-bit path active: source_bit_depth=%d p016_shift=%d output=NV12/8-bit",
                    self.sid,
                    bit_depth,
                    config.PASSTHROUGH_PYNV_10BIT_SHIFT,
                )
            mux_duration = max(0.0, float(timing.duration or info.duration or 0.0) - self.start_sec)
            if self._stop.is_set():
                log.info("[PYNV][%d] worker stopped before mux open", self.sid)
                return
            self._mux = self._open_muxer(fps, mux_duration)
            if self._stop.is_set():
                log.info("[PYNV][%d] worker stopped after mux open", self.sid)
                return
            mux_input = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
            assert mux_input is not None
            self._reader = threading.Thread(target=self._reader_loop, name="pynv-reader", daemon=True)
            self._stderr_reader = threading.Thread(target=self._stderr_loop, name="pynv-stderr", daemon=True)
            self._reader.start()
            self._stderr_reader.start()
            if self._video_mux is not None:
                threading.Thread(target=self._video_stderr_loop, name="pynv-video-stderr", daemon=True).start()
            self.matter.reset_state()
            log.info(
                "[PYNV][%d] RVM recurrent state reset at %.3fs; first frames after seek may show alpha warmup jitter",
                self.sid, self.start_sec,
            )
            alpha_packer = None
            if self.output_mode == "alpha":
                from pipeline.alpha_packer import AlphaPacker, alpha_2d_disparity_px

                alpha_packer = AlphaPacker(self.matter)
                alpha_2d_disparity = alpha_2d_disparity_px(out_w)
                log.info(
                    "[PYNV][%d] alpha passthrough active: projection=%s process=%dx%d output=%dx%d scale=%.3f radius=%.3f layout=alpha-packer-6block flat2d_fov=%.1f flat2d_distance=%.2fm flat2d_disparity=%.1fpx",
                    self.sid,
                    alpha_projection_mode or alpha_packer.projection_mode(alpha_process_w, alpha_process_h),
                    alpha_process_w,
                    alpha_process_h,
                    out_w,
                    out_h,
                    alpha_packer.scale,
                    alpha_packer.radius_scale,
                    config.ALPHA_2D_FOV,
                    config.ALPHA_2D_DISTANCE_M,
                    alpha_2d_disparity,
                )
            slate_done = threading.Event()
            slate_error: list[str] = []
            slate_frames_box = [0]

            def slate_video_loop() -> None:
                if self._slate_audio_addr is None:
                    slate_done.set()
                    return
                slate_nv12 = None
                slate_alloc_start = time.perf_counter()
                try:
                    slate_nv12 = self._make_slate_nv12_gpu(out_w, out_h)
                    log.info(
                        "[PYNV][%d] slate video begin: continuous until first real frame ready alloc=%.3fs burst_frames=%d fps=%.3f",
                        self.sid,
                        time.perf_counter() - slate_alloc_start,
                        config.PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES,
                        fps,
                    )
                    slate_start = time.perf_counter()
                    while (
                        not self._stop.is_set()
                        and not slate_stop.is_set()
                        and not self._slate_audio_failed.is_set()
                    ):
                        slate_frames = slate_frames_box[0]
                        if fps > 0:
                            while not self._stop.is_set() and not slate_stop.is_set():
                                delay = _slate_frame_pace_delay(
                                    fps=fps,
                                    sent_frames=slate_frames,
                                    burst_frames=config.PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES,
                                    pace_start=slate_start,
                                    now=time.perf_counter(),
                                )
                                if delay <= 0:
                                    break
                                self._stop.wait(min(delay, 0.05))
                            if self._stop.is_set() or slate_stop.is_set():
                                break
                        flags = 0
                        if slate_frames == 0:
                            flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
                        encode_start = time.perf_counter()
                        with self._encoder_lock:
                            if self._enc is None or self._stop.is_set() or slate_stop.is_set():
                                break
                            bitstream = self._enc.Encode(GpuNv12AppFrame(slate_nv12, out_w, out_h), flags)
                        if slate_frames == 0:
                            log.info(
                                "[PYNV][%d] slate first encode: bytes=%d elapsed=%.3fs",
                                self.sid,
                                len(bitstream) if bitstream else 0,
                                time.perf_counter() - encode_start,
                            )
                        if bitstream:
                            mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                            if self._stop.is_set() or not mux_stdin or mux_stdin.closed:
                                break
                            try:
                                self._mark_first_write()
                                mux_stdin.write(bitstream)
                            except (BrokenPipeError, OSError, ValueError) as e:
                                log.info("[PYNV][%d] slate mux stdin write stopped at frame=%d: %s", self.sid, slate_frames + 1, e)
                                break
                        slate_frames_box[0] = slate_frames + 1
                    log.info(
                        "[PYNV][%d] slate video end: frames=%d ready=%s direct=%s failed=%s real_ready=%s elapsed=%.3fs",
                        self.sid,
                        slate_frames_box[0],
                        self._slate_audio_ready.is_set(),
                        self._slate_direct_ready.is_set(),
                        self._slate_audio_failed.is_set(),
                        slate_stop.is_set(),
                        time.perf_counter() - slate_start,
                    )
                except Exception as e:
                    slate_error.append(f"{e}\n{traceback.format_exc(limit=8)}")
                    self._stop.set()
                finally:
                    slate_done.set()

            if self._slate_audio_addr is not None and config.PASSTHROUGH_MPEGTS_VIDEO_SLATE:
                slate_thread = threading.Thread(target=slate_video_loop, name=f"pynv-slate-video-{self.sid}", daemon=True)
                slate_thread.start()
            if (
                config.PASSTHROUGH_PYNV_WORKER_MODE == "two_stage"
                and alpha_packer is None
                and self._slate_audio_addr is None
            ):
                self._worker_loop_two_stage_green(
                    nvc=nvc,
                    info=info,
                    source_fps=source_fps,
                    fps=fps,
                    start_out=start_out,
                    target=target,
                    out_w=out_w,
                    out_h=out_h,
                    subtitle_renderer=subtitle_renderer,
                    pending_nv12_slots=pending_nv12_slots,
                    producer_pacing=producer_pacing,
                )
                return
            if config.PASSTHROUGH_PYNV_WORKER_MODE not in ("serial", "two_stage"):
                raise RuntimeError(f"unknown PT_PASSTHROUGH_PYNV_WORKER_MODE={config.PASSTHROUGH_PYNV_WORKER_MODE!r}")
            if config.PASSTHROUGH_PYNV_WORKER_MODE == "two_stage":
                log.info(
                    "[PYNV][%d] worker mode two_stage requested but output_mode=%s slate_active=%s; using serial",
                    self.sid,
                    self.output_mode,
                    self._slate_audio_addr is not None,
                )
            last_src_idx = -1
            t_start = time.perf_counter()
            interval_start = t_start
            interval_bytes = 0
            sum_decode = 0.0
            sum_composite = 0.0
            sum_sync = 0.0
            sum_encode = 0.0
            sum_mux = 0.0
            sum_mat_pre = 0.0
            sum_mat_ort = 0.0
            sum_mat_kernel = 0.0
            max_decode = 0.0
            max_composite = 0.0
            max_sync = 0.0
            max_encode = 0.0
            max_mux = 0.0
            max_mat_pre = 0.0
            max_mat_ort = 0.0
            max_mat_kernel = 0.0
            max_pending_nv12_slots = max(0, int(config.PASSTHROUGH_NV12_RING_SLOTS) - 1)
            for i in range(target):
                if self._stop.is_set():
                    break
                if producer_pacing and self.container == "mpegts" and fps > 0:
                    due = t_start + (i / fps)
                    now = time.perf_counter()
                    if due > now:
                        self._stop.wait(due - now)
                        if self._stop.is_set():
                            break
                out_idx = start_out + i
                src_idx = min(len(self._dec) - 1, cfr_source_index(out_idx, source_fps, fps))
                if src_idx <= last_src_idx:
                    src_idx = min(len(self._dec) - 1, last_src_idx + 1)
                last_src_idx = src_idx
                t0 = time.perf_counter()
                frame = self._dec.frame_at(src_idx)
                t1 = time.perf_counter()
                if self._stop.is_set():
                    break
                nv12_slot = None
                if alpha_packer is not None:
                    if alpha_threaded_owned_copy:
                        frame = frame.owned_copy()
                    subtitle_overlay = self._subtitle_overlay_for_time(
                        subtitle_renderer,
                        out_idx / fps if fps > 0 else 0.0,
                    )

                    def apply_alpha_subtitle(uploaded_nv12):
                        if subtitle_renderer is None or subtitle_overlay is None:
                            return None
                        return self._subtitle_overlay_positions(
                            uploaded_nv12,
                            subtitle_renderer,
                            subtitle_overlay,
                        )

                    if isinstance(frame, GpuP016Frame):
                        out_nv12, timing = alpha_packer.pack_gpu_p016_frame(
                            frame,
                            shift_bits=config.PASSTHROUGH_PYNV_10BIT_SHIFT,
                            before_pack=apply_alpha_subtitle,
                            out_h=out_h,
                            out_w=out_w,
                        )
                    else:
                        out_nv12, timing = alpha_packer.pack_gpu_nv12_frame(
                            frame,
                            before_pack=apply_alpha_subtitle,
                            out_h=out_h,
                            out_w=out_w,
                        )
                elif isinstance(frame, GpuP016Frame):
                    nv12_slot = self.matter.acquire_nv12_output_slot(out_h, out_w)
                    out_nv12, timing = self.matter.composite_green_gpu_p016_frame_to_gpu_nv12_profile(
                        frame,
                        shift_bits=config.PASSTHROUGH_PYNV_10BIT_SHIFT,
                        out_h=out_h,
                        out_w=out_w,
                        out_slot=nv12_slot,
                    )
                else:
                    nv12_slot = self.matter.acquire_nv12_output_slot(out_h, out_w)
                    out_nv12, timing = self.matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(
                        frame,
                        out_h=out_h,
                        out_w=out_w,
                        out_slot=nv12_slot,
                    )
                t2 = time.perf_counter()
                if self._stop.is_set():
                    self.matter.release_nv12_output_slot(nv12_slot)
                    break
                cuda_stream = getattr(matting_module, "_CUDA_STREAM", None)
                if cuda_stream is not None:
                    cuda_stream.synchronize()
                t3 = time.perf_counter()
                if self._stop.is_set():
                    self.matter.release_nv12_output_slot(nv12_slot)
                    break
                if alpha_packer is None:
                    self._apply_subtitle_overlay(out_nv12, subtitle_renderer, out_idx / fps if fps > 0 else 0.0)
                app_frame = GpuNv12AppFrame(out_nv12, out_w, out_h)
                flags = 0
                first_real_frame = not self._real_video_started.is_set()
                if i == 0:
                    flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
                    log.info(
                        "[PYNV][%d] seek start: requested_sec=%.3f out_idx=%d src_idx=%d fps=%.3f source_fps=%.3f",
                        self.sid, self.start_sec, out_idx, src_idx, fps, source_fps,
                    )
                if first_real_frame and slate_thread is not None:
                    slate_stop.set()
                    if slate_thread.is_alive():
                        slate_thread.join(timeout=2.0)
                    if slate_thread.is_alive():
                        log.warning("[PYNV][%d] slate video thread did not stop before first real frame", self.sid)
                    if slate_error:
                        raise RuntimeError(f"slate video failed: {slate_error[0]}")
                    if self._slate_audio_failed.is_set() and not self._stop.is_set():
                        raise RuntimeError("slate audio cache build failed")
                    log.info("[PYNV][%d] slate-to-real switch: slate_frames=%d first_real_frame=%d", self.sid, slate_frames_box[0], i + 1)
                try:
                    with self._encoder_lock:
                        bitstream = self._enc.Encode(app_frame, flags)
                except Exception:
                    self.matter.release_nv12_output_slot(nv12_slot)
                    raise
                if nv12_slot is not None:
                    pending_nv12_slots.append(nv12_slot)
                    nv12_slot = None
                    while len(pending_nv12_slots) > max_pending_nv12_slots:
                        self.matter.release_nv12_output_slot(pending_nv12_slots.pop(0))
                self.matter.release_nv12_output_slot(nv12_slot)
                t4 = time.perf_counter()
                if bitstream:
                    if self.frames_produced == 0:
                        log.info(
                            "[PYNV][%d] first bitstream len=%d hevc_nals=%s",
                            self.sid,
                            len(bitstream),
                            _hevc_nal_summary(bitstream) if PYNV_OUTPUT_CODEC == "hevc" else "n/a",
                        )
                    mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                    if self._stop.is_set() or not mux_stdin or mux_stdin.closed:
                        log.info("[PYNV][%d] mux stdin closed before write at frame=%d", self.sid, i + 1)
                        break
                    try:
                        write_started = time.perf_counter()
                        self._mark_first_write()
                        mux_stdin.write(bitstream)
                        if first_real_frame:
                            self._real_video_started.set()
                            log.info("[PYNV][%d] first real video bitstream written; slate audio may switch", self.sid)
                            self._log_vram("first_real_video_bitstream")
                        write_elapsed = time.perf_counter() - write_started
                        if write_elapsed >= 1.0:
                            log.warning(
                                "[PYNV][%d] mux stdin write slow: frame=%d len=%d elapsed=%.3fs video_rc=%s final_rc=%s",
                                self.sid,
                                i + 1,
                                len(bitstream),
                                write_elapsed,
                                self._video_mux.poll() if self._video_mux is not None else None,
                                self._mux.poll() if self._mux is not None else None,
                            )
                        interval_bytes += len(bitstream)
                    except (BrokenPipeError, OSError, ValueError) as e:
                        log.info("[PYNV][%d] mux stdin write stopped at frame=%d: %s", self.sid, i + 1, e)
                        break
                t5 = time.perf_counter()
                dt_decode = t1 - t0
                dt_composite = t2 - t1
                dt_sync = t3 - t2
                dt_encode = t4 - t3
                dt_mux = t5 - t4
                sum_decode += dt_decode
                sum_composite += dt_composite
                sum_sync += dt_sync
                sum_encode += dt_encode
                sum_mux += dt_mux
                mat_pre = float(getattr(timing, "preprocess_ms", 0.0)) / 1000.0
                mat_ort = float(getattr(timing, "ort_ms", 0.0)) / 1000.0
                mat_kernel = float(getattr(timing, "composite_ms", 0.0)) / 1000.0
                sum_mat_pre += mat_pre
                sum_mat_ort += mat_ort
                sum_mat_kernel += mat_kernel
                max_decode = max(max_decode, dt_decode)
                max_composite = max(max_composite, dt_composite)
                max_sync = max(max_sync, dt_sync)
                max_encode = max(max_encode, dt_encode)
                max_mux = max(max_mux, dt_mux)
                max_mat_pre = max(max_mat_pre, mat_pre)
                max_mat_ort = max(max_mat_ort, mat_ort)
                max_mat_kernel = max(max_mat_kernel, mat_kernel)
                self.frames_produced = i + 1
                if config.DEBUG_LOGS and self.frames_produced % _DIAG_INTERVAL == 0:
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    interval_elapsed = max(0.001, time.perf_counter() - interval_start)
                    interval_frames = _DIAG_INTERVAL
                    log.info(
                        "[PYNV][%d] frame %d/%d fps=%.2f interval_fps=%.2f src_idx=%d bytes=%d out_bps=%.1fM stage_avg_ms decode=%.2f composite=%.2f sync=%.2f encode=%.2f mux=%.2f mat_avg_ms pre=%.2f ort=%.2f kernel=%.2f stage_max_ms decode=%.2f composite=%.2f sync=%.2f encode=%.2f mux=%.2f mat_max_ms pre=%.2f ort=%.2f kernel=%.2f",
                        self.sid,
                        self.frames_produced,
                        target,
                        self.frames_produced / elapsed,
                        interval_frames / interval_elapsed,
                        src_idx,
                        self.bytes_emitted,
                        (interval_bytes * 8.0 / interval_elapsed) / 1_000_000.0,
                        (sum_decode / interval_frames) * 1000.0,
                        (sum_composite / interval_frames) * 1000.0,
                        (sum_sync / interval_frames) * 1000.0,
                        (sum_encode / interval_frames) * 1000.0,
                        (sum_mux / interval_frames) * 1000.0,
                        (sum_mat_pre / interval_frames) * 1000.0,
                        (sum_mat_ort / interval_frames) * 1000.0,
                        (sum_mat_kernel / interval_frames) * 1000.0,
                        max_decode * 1000.0,
                        max_composite * 1000.0,
                        max_sync * 1000.0,
                        max_encode * 1000.0,
                        max_mux * 1000.0,
                        max_mat_pre * 1000.0,
                        max_mat_ort * 1000.0,
                        max_mat_kernel * 1000.0,
                    )
                    interval_start = time.perf_counter()
                    interval_bytes = 0
                    sum_decode = 0.0
                    sum_composite = 0.0
                    sum_sync = 0.0
                    sum_encode = 0.0
                    sum_mux = 0.0
                    sum_mat_pre = 0.0
                    sum_mat_ort = 0.0
                    sum_mat_kernel = 0.0
                    max_decode = 0.0
                    max_composite = 0.0
                    max_sync = 0.0
                    max_encode = 0.0
                    max_mux = 0.0
                    max_mat_pre = 0.0
                    max_mat_ort = 0.0
                    max_mat_kernel = 0.0
            if not self._stop.is_set():
                log.info("[PYNV][%d] EndEncode begin frames=%d", self.sid, self.frames_produced)
                with self._encoder_lock:
                    tail = self._enc.EndEncode()
                if tail:
                    try:
                        mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                        if mux_stdin:
                            tail_started = time.perf_counter()
                            self._mark_first_write()
                            mux_stdin.write(tail)
                            tail_elapsed = time.perf_counter() - tail_started
                            if tail_elapsed >= 1.0:
                                log.warning(
                                    "[PYNV][%d] EndEncode tail mux write slow: len=%d elapsed=%.3fs video_rc=%s final_rc=%s",
                                    self.sid,
                                    len(tail),
                                    tail_elapsed,
                                    self._video_mux.poll() if self._video_mux is not None else None,
                                    self._mux.poll() if self._mux is not None else None,
                                )
                        log.info("[PYNV][%d] EndEncode tail len=%d", self.sid, len(tail))
                    except (BrokenPipeError, OSError):
                        pass
                log.info("[PYNV][%d] EndEncode done", self.sid)
            while pending_nv12_slots:
                self.matter.release_nv12_output_slot(pending_nv12_slots.pop(0))
        except Exception as e:
            if self._stop.is_set():
                log.info("[PYNV][%d] worker stopped during close: %s", self.sid, e)
                return
            if self.frames_produced == 0 and self.bytes_emitted == 0:
                self.startup_error = str(e)
            log.error("[PYNV][%d] worker exception: %s\n%s", self.sid, e, traceback.format_exc(limit=8))
        finally:
            slate_stop.set()
            if slate_thread is not None and slate_thread.is_alive():
                slate_thread.join(timeout=2.0)
            while pending_nv12_slots:
                self.matter.release_nv12_output_slot(pending_nv12_slots.pop(0))
            self._enc = None
            reader_started = self._reader is not None
            try:
                stdin = self._video_mux.stdin if self._video_mux is not None else (self._mux.stdin if self._mux else None)
                if stdin:
                    log.info("[PYNV][%d] worker closing mux stdin frames=%d stop=%s", self.sid, self.frames_produced, self._stop.is_set())
                    stdin.close()
            except Exception:
                pass
            if not reader_started:
                self._post_sentinel()
            try:
                gc.collect()
            except Exception:
                pass
            self._log_vram("worker_done")
            log.info("[PYNV][%d] worker done frames=%d bytes_emitted=%d reader_started=%s", self.sid, self.frames_produced, self.bytes_emitted, reader_started)

    def _worker_loop_two_dvr(self) -> None:
        reader_started = False
        try:
            import numpy as np
            import cupy as cp
            import PyNvVideoCodec as nvc
            from offline.da3_depth import ensure_model_available, warmup_depth_engine
            from offline.two_dvr import _ensure_trt_cache
            from offline.two_dvr_gpu import GpuStereoRenderer
            from offline.two_dvr_pynv import DA3_SIZE, _NV12_RGB_KERNELS, _letterbox_box
            from offline.two_dvr_render import (
                DEFAULT_FLAT_FOV_DEG,
                PROJECTION_FLAT_3D,
                effective_eye_distance_mm,
                strength_multiplier,
            )

            codec_meta = self.metadata.codec if self.metadata is not None else None
            bit_depth = int(codec_meta.bit_depth if codec_meta and codec_meta.bit_depth > 0 else 8)
            if bit_depth > 8:
                raise RuntimeError("2D->3D live currently supports 8-bit NV12 sources only")
            meta_dec = PyNvSimpleDecoder(self.src, bit_depth=bit_depth)
            self._log_vram("two_dvr_decoder_metadata_created")
            info = meta_dec.info
            if int(info.width) > 4096:
                meta_dec.stop()
                raise RuntimeError("2D->3D live source width exceeds 4096px; SBS output would exceed 8K")
            dec_len = len(meta_dec)
            timing = self.metadata.timing if self.metadata is not None else probe_timing_metadata(self.src)
            if not timing.is_cfr:
                meta_dec.stop()
                raise RuntimeError("2D->3D live requires strong CFR source")
            source_fps = float(timing.source_fps or info.fps or 30.0)
            fps_cap = config.PASSTHROUGH_MAX_FPS if self.max_fps is None else float(self.max_fps)
            fps = float(timing.effective_fps(fps_cap))
            self.output_fps = fps
            producer_pacing = bool(config.PASSTHROUGH_PRODUCER_REALTIME_PACING or fps_cap > 0)
            start_out = int(round(self.start_sec * fps))
            target = max(0, int((timing.duration or info.duration or 0.0) * fps) - start_out)
            max_target = int((dec_len - 1) * fps / source_fps) + 1 if source_fps > 0 else dec_len
            target = min(target, max(1, max_target))
            initial_src_idx = min(dec_len - 1, cfr_source_index(start_out, source_fps, fps))
            meta_dec.stop()
            self._dec = PyNvThreadedSerialDecoder(
                self.src,
                bit_depth=bit_depth,
                start_frame=initial_src_idx,
                batch_size=config.PASSTHROUGH_PYNV_THREADED_BATCH_SIZE,
                buffer_size=config.PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE,
                info=info,
                num_frames=dec_len,
            )
            self._log_vram("two_dvr_decoder_created")

            model_variant = config.TWO_DVR_MODEL if config.TWO_DVR_MODEL in {"small", "base", "small_hd", "base_hd"} else "base"
            hole_fill = config.TWO_DVR_HOLE_FILL if config.TWO_DVR_HOLE_FILL in {"inverse_warp", "soft_shift"} else "soft_shift"
            strength = strength_multiplier(config.TWO_DVR_STRENGTH)
            eye_distance = effective_eye_distance_mm(config.TWO_DVR_EYE_DISTANCE_MM, strength)
            temporal_kwargs = {
                "temporal_norm": bool(config.TWO_DVR_TEMPORAL_NORM),
                "temporal_norm_alpha": float(config.TWO_DVR_TEMPORAL_NORM_ALPHA),
                "temporal_norm_reset": float(config.TWO_DVR_TEMPORAL_NORM_RESET),
                "temporal_depth": bool(config.TWO_DVR_TEMPORAL_DEPTH),
                "temporal_depth_mode": str(config.TWO_DVR_TEMPORAL_DEPTH_MODE),
                "temporal_depth_alpha": float(config.TWO_DVR_TEMPORAL_DEPTH_ALPHA),
                "temporal_flow_diff": float(config.TWO_DVR_TEMPORAL_FLOW_DIFF),
                "temporal_flow_consistency": float(config.TWO_DVR_TEMPORAL_FLOW_CONSISTENCY),
                "temporal_flow_motion_gate": float(config.TWO_DVR_TEMPORAL_FLOW_MOTION_GATE),
                "temporal_affine": bool(config.TWO_DVR_TEMPORAL_AFFINE),
                "temporal_affine_max_scale": float(config.TWO_DVR_TEMPORAL_AFFINE_MAX_SCALE),
                "temporal_affine_max_bias": float(config.TWO_DVR_TEMPORAL_AFFINE_MAX_BIAS),
                "temporal_static_deadband_px": float(config.TWO_DVR_TEMPORAL_STATIC_DEADBAND_PX),
                "temporal_static_max_step_px": float(config.TWO_DVR_TEMPORAL_STATIC_MAX_STEP_PX),
                "temporal_motion_max_step_px": float(config.TWO_DVR_TEMPORAL_MOTION_MAX_STEP_PX),
            }
            log.info("[PYNV][%d] 2D->3D live model=%s ensure available", self.sid, model_variant)
            if not ensure_model_available(model_variant, log=lambda m: log.info("[PYNV][%d] %s", self.sid, m)):
                raise RuntimeError(f"DA3 model {model_variant} unavailable and download failed")
            log.info("[PYNV][%d] 2D->3D live TensorRT cache ensure begin model=%s", self.sid, model_variant)
            _ensure_trt_cache(model_variant, "trt")
            engine = warmup_depth_engine(
                variant=model_variant,
                provider="trt",
                log=lambda msg: log.info("[PYNV][%d] %s", self.sid, msg),
            )
            if not engine.folded:
                raise RuntimeError("2D->3D live requires folded-preprocess DA3 model")
            self._log_vram("two_dvr_da3_created")

            width, height = int(info.width), int(info.height)
            renderer = GpuStereoRenderer(
                width,
                height,
                PROJECTION_FLAT_3D,
                eye_distance,
                hole_fill,
                DEFAULT_FLAT_FOV_DEG,
                **temporal_kwargs,
            )
            renderer.reset()
            cut_detector = None
            if config.TWO_DVR_SCENE_CUT:
                from utils.scene_detection import SceneCutDetector
                cut_detector = SceneCutDetector(threshold=config.TWO_DVR_SCENE_CUT_THRESHOLD)
            out_w, out_h = int(renderer.out_w), int(renderer.out_h)
            if out_w > 8192:
                raise RuntimeError(f"2D->3D live output width exceeds 8K: {out_w}")
            da3_size = int(engine.size)   # 518 (base/small) or 1036 (hd presets)
            x0, y0, nw, nh = _letterbox_box(width, height, da3_size)
            subtitle_renderer: SubtitleRenderer | None = None
            subtitle_path = find_subtitle_for_video(self.src)
            if subtitle_path is not None:
                try:
                    subtitle_renderer = SubtitleRenderer(subtitle_path, out_w, out_h)
                    if not subtitle_renderer.enabled:
                        subtitle_renderer = None
                except Exception as e:
                    subtitle_renderer = None
                    log.warning("[PYNV][%d] subtitle load failed: %s error=%s", self.sid, subtitle_path.name, e)

            bitrate_estimate = effective_default_bitrate(self.src, PYNV_BACKEND_LABEL)
            bitrate = str(bitrate_estimate.bps)
            enc_kwargs = _pynv_encoder_kwargs(bitrate=bitrate, fps=f"{fps:.6f}")
            self._enc = nvc.CreateEncoder(out_w, out_h, "NV12", False, **enc_kwargs)
            self._log_vram("two_dvr_encoder_created")
            temporal_depth_on = temporal_kwargs["temporal_depth"] and temporal_kwargs["temporal_depth_mode"] != "off"
            log.info(
                "[PYNV][%d] 2D->3D live start: source=%dx%d output=%dx%d source_fps=%.3f output_fps=%.3f "
                "target=%d model=%s provider=%s hole_fill=%s strength=%.2f eye_distance=%.1fmm "
                "temporal_norm=%s norm_alpha=%.2f temporal_depth=%s depth_mode=%s depth_alpha=%.2f "
                "affine=%s deadband_px=%.2f static_step_px=%.2f motion_step_px=%.2f container=%s",
                self.sid,
                width,
                height,
                out_w,
                out_h,
                source_fps,
                fps,
                target,
                model_variant,
                engine.providers[0] if engine.providers else "unknown",
                hole_fill,
                strength,
                eye_distance,
                "on" if temporal_kwargs["temporal_norm"] else "off",
                temporal_kwargs["temporal_norm_alpha"],
                "on" if temporal_depth_on else "off",
                temporal_kwargs["temporal_depth_mode"],
                temporal_kwargs["temporal_depth_alpha"],
                "on" if temporal_kwargs["temporal_affine"] else "off",
                temporal_kwargs["temporal_static_deadband_px"],
                temporal_kwargs["temporal_static_max_step_px"],
                temporal_kwargs["temporal_motion_max_step_px"],
                self.container,
            )

            mux_duration = max(0.0, float(timing.duration or info.duration or 0.0) - self.start_sec)
            self._mux = self._open_muxer(fps, mux_duration)
            mux_input = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
            assert mux_input is not None
            self._reader = threading.Thread(target=self._reader_loop, name="pynv-reader", daemon=True)
            self._stderr_reader = threading.Thread(target=self._stderr_loop, name="pynv-stderr", daemon=True)
            self._reader.start()
            self._stderr_reader.start()
            if self._video_mux is not None:
                threading.Thread(target=self._video_stderr_loop, name="pynv-video-stderr", daemon=True).start()
            reader_started = True

            mod = cp.RawModule(code=_NV12_RGB_KERNELS)
            k_to_rgb = mod.get_function("nv12_to_rgb")
            k_lb = mod.get_function("nv12_to_rgb_letterbox")
            k_to_nv12 = mod.get_function("rgb_to_nv12")
            rgb_g = cp.empty((height, width, 3), cp.uint8)
            canvas_g = cp.empty((da3_size, da3_size, 3), cp.uint8)
            out_nv12 = cp.empty((out_h * 3 // 2, out_w), cp.uint8)
            bx = (16, 16, 1)
            grid = ((width + 15) // 16, (height + 15) // 16, 1)
            grid_lb = ((da3_size + 15) // 16, (da3_size + 15) // 16, 1)
            grid_out = ((out_w + 15) // 16, (out_h + 15) // 16, 1)

            t_start = time.perf_counter()
            interval_start = t_start
            interval_bytes = 0
            last_src_idx = -1
            sum_decode = sum_depth = sum_render = sum_encode = sum_mux = 0.0
            max_decode = max_depth = max_render = max_encode = max_mux = 0.0
            for i in range(target):
                if self._stop.is_set():
                    break
                if producer_pacing and self.container == "mpegts" and fps > 0:
                    due = t_start + (i / fps)
                    now = time.perf_counter()
                    if due > now:
                        self._stop.wait(due - now)
                        if self._stop.is_set():
                            break
                out_idx = start_out + i
                src_idx = min(len(self._dec) - 1, cfr_source_index(out_idx, source_fps, fps))
                if src_idx <= last_src_idx:
                    src_idx = min(len(self._dec) - 1, last_src_idx + 1)
                last_src_idx = src_idx
                t0 = time.perf_counter()
                if i == 0:
                    log.info("[PYNV][%d] 2D->3D first frame decode begin src_idx=%d", self.sid, src_idx)
                frame = self._dec.frame_at(src_idx).owned_copy()
                if isinstance(frame, GpuP016Frame):
                    raise RuntimeError("2D->3D live received P016 frame despite 8-bit preflight")
                y_g = frame.y.as_cupy(cp.uint8).reshape(height, width)
                uv_g = frame.uv.as_cupy(cp.uint8).reshape(height // 2, width)
                t1 = time.perf_counter()
                if i == 0:
                    log.info("[PYNV][%d] 2D->3D first frame depth begin decode_ms=%.1f", self.sid, (t1 - t0) * 1000.0)
                k_lb(
                    grid_lb,
                    bx,
                    (
                        y_g,
                        uv_g,
                        canvas_g,
                        np.int32(width),
                        np.int32(height),
                        np.int32(da3_size),
                        np.int32(x0),
                        np.int32(y0),
                        np.int32(nw),
                        np.int32(nh),
                    ),
                )
                canvas = canvas_g.get()[None]
                if cut_detector is not None and cut_detector.step(canvas[0]):
                    renderer.reset()  # scene-cut: re-seed depth band/base for the new shot
                depth = engine.session.run([engine.output_name], {engine.input_name: canvas})[0][0]
                near_g = renderer.prepare_near_gpu(
                    depth[y0:y0 + nh, x0:x0 + nw],
                    canvas_g[y0:y0 + nh, x0:x0 + nw],
                )
                t2 = time.perf_counter()
                if i == 0:
                    log.info("[PYNV][%d] 2D->3D first frame render begin depth_ms=%.1f", self.sid, (t2 - t1) * 1000.0)
                k_to_rgb(grid, bx, (y_g, uv_g, rgb_g, np.int32(width), np.int32(height)))
                sbs_rgb = renderer.render_into_gpu(rgb_g, near_g)
                k_to_nv12(grid_out, bx, (sbs_rgb, out_nv12, np.int32(out_w), np.int32(out_h)))
                self._apply_subtitle_overlay(out_nv12, subtitle_renderer, out_idx / fps if fps > 0 else 0.0)
                cp.cuda.get_current_stream().synchronize()
                t3 = time.perf_counter()
                if i == 0:
                    log.info("[PYNV][%d] 2D->3D first frame encode begin render_ms=%.1f", self.sid, (t3 - t2) * 1000.0)
                flags = 0
                if i == 0:
                    flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
                    log.info(
                        "[PYNV][%d] 2D->3D seek start: requested_sec=%.3f out_idx=%d src_idx=%d fps=%.3f source_fps=%.3f",
                        self.sid,
                        self.start_sec,
                        out_idx,
                        src_idx,
                        fps,
                        source_fps,
                    )
                with self._encoder_lock:
                    bitstream = self._enc.Encode(GpuNv12AppFrame(out_nv12, out_w, out_h), flags)
                t4 = time.perf_counter()
                if i == 0:
                    log.info(
                        "[PYNV][%d] 2D->3D first frame encode done encode_ms=%.1f bitstream=%d",
                        self.sid,
                        (t4 - t3) * 1000.0,
                        len(bitstream) if bitstream else 0,
                    )
                if bitstream:
                    if self.frames_produced == 0:
                        log.info(
                            "[PYNV][%d] first 2D->3D bitstream len=%d hevc_nals=%s",
                            self.sid,
                            len(bitstream),
                            _hevc_nal_summary(bitstream) if PYNV_OUTPUT_CODEC == "hevc" else "n/a",
                        )
                    mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                    if self._stop.is_set() or not mux_stdin or mux_stdin.closed:
                        break
                    try:
                        write_started = time.perf_counter()
                        self._mark_first_write()
                        mux_stdin.write(bitstream)
                        if not self._real_video_started.is_set():
                            self._real_video_started.set()
                            self._log_vram("first_2dvr_video_bitstream")
                        interval_bytes += len(bitstream)
                        write_elapsed = time.perf_counter() - write_started
                        if write_elapsed >= 1.0:
                            log.warning("[PYNV][%d] 2D->3D mux stdin write slow: frame=%d elapsed=%.3fs", self.sid, i + 1, write_elapsed)
                    except (BrokenPipeError, OSError, ValueError) as e:
                        log.info("[PYNV][%d] 2D->3D mux stdin write stopped at frame=%d: %s", self.sid, i + 1, e)
                        break
                t5 = time.perf_counter()
                self.frames_produced = i + 1
                sum_decode += t1 - t0
                sum_depth += t2 - t1
                sum_render += t3 - t2
                sum_encode += t4 - t3
                sum_mux += t5 - t4
                max_decode = max(max_decode, t1 - t0)
                max_depth = max(max_depth, t2 - t1)
                max_render = max(max_render, t3 - t2)
                max_encode = max(max_encode, t4 - t3)
                max_mux = max(max_mux, t5 - t4)
                if config.DEBUG_LOGS and self.frames_produced % _DIAG_INTERVAL == 0:
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    interval_elapsed = max(0.001, time.perf_counter() - interval_start)
                    interval_frames = _DIAG_INTERVAL
                    log.info(
                        "[PYNV][%d] 2D->3D frame %d/%d fps=%.2f interval_fps=%.2f src_idx=%d bytes=%d out_bps=%.1fM "
                        "stage_avg_ms decode=%.2f depth=%.2f render=%.2f encode=%.2f mux=%.2f "
                        "stage_max_ms decode=%.2f depth=%.2f render=%.2f encode=%.2f mux=%.2f",
                        self.sid,
                        self.frames_produced,
                        target,
                        self.frames_produced / elapsed,
                        interval_frames / interval_elapsed,
                        src_idx,
                        self.bytes_emitted,
                        (interval_bytes * 8.0 / interval_elapsed) / 1_000_000.0,
                        (sum_decode / interval_frames) * 1000.0,
                        (sum_depth / interval_frames) * 1000.0,
                        (sum_render / interval_frames) * 1000.0,
                        (sum_encode / interval_frames) * 1000.0,
                        (sum_mux / interval_frames) * 1000.0,
                        max_decode * 1000.0,
                        max_depth * 1000.0,
                        max_render * 1000.0,
                        max_encode * 1000.0,
                        max_mux * 1000.0,
                    )
                    interval_start = time.perf_counter()
                    interval_bytes = 0
                    sum_decode = sum_depth = sum_render = sum_encode = sum_mux = 0.0
                    max_decode = max_depth = max_render = max_encode = max_mux = 0.0
            if not self._stop.is_set():
                log.info("[PYNV][%d] 2D->3D EndEncode begin frames=%d", self.sid, self.frames_produced)
                with self._encoder_lock:
                    tail = self._enc.EndEncode()
                if tail:
                    mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                    if mux_stdin:
                        self._mark_first_write()
                        mux_stdin.write(tail)
                log.info("[PYNV][%d] 2D->3D EndEncode done", self.sid)
        except Exception as e:
            if self._stop.is_set():
                log.info("[PYNV][%d] 2D->3D worker stopped during close: %s", self.sid, e)
                return
            if self.frames_produced == 0 and self.bytes_emitted == 0:
                self.startup_error = str(e)
            log.error("[PYNV][%d] 2D->3D worker exception: %s\n%s", self.sid, e, traceback.format_exc(limit=8))
        finally:
            self._enc = None
            try:
                if self._dec is not None:
                    self._dec.stop()
            except Exception:
                pass
            try:
                stdin = self._video_mux.stdin if self._video_mux is not None else (self._mux.stdin if self._mux else None)
                if stdin:
                    log.info("[PYNV][%d] 2D->3D worker closing mux stdin frames=%d stop=%s", self.sid, self.frames_produced, self._stop.is_set())
                    stdin.close()
            except Exception:
                pass
            if not reader_started:
                self._post_sentinel()
            try:
                gc.collect()
            except Exception:
                pass
            self._log_vram("two_dvr_worker_done")
            log.info("[PYNV][%d] 2D->3D worker done frames=%d bytes_emitted=%d reader_started=%s", self.sid, self.frames_produced, self.bytes_emitted, reader_started)

    def _worker_loop_two_stage_green(
        self,
        *,
        nvc,
        info,
        source_fps: float,
        fps: float,
        start_out: int,
        target: int,
        out_w: int,
        out_h: int,
        subtitle_renderer: SubtitleRenderer | None,
        pending_nv12_slots: list[object],
        producer_pacing: bool,
    ) -> None:
        log.info("[PYNV][%d] worker mode=two_stage green begin target=%d", self.sid, target)
        encode_q: queue.Queue = queue.Queue(maxsize=max(1, int(config.PASSTHROUGH_NV12_RING_SLOTS)))
        sentinel = object()
        errors: list[str] = []
        max_pending_nv12_slots = max(0, int(config.PASSTHROUGH_NV12_RING_SLOTS) - 1)
        t_start = time.perf_counter()
        interval_start = t_start
        interval_bytes = 0
        sum_decode = 0.0
        sum_composite = 0.0
        sum_sync = 0.0
        sum_encode = 0.0
        sum_mux = 0.0
        max_decode = 0.0
        max_composite = 0.0
        max_sync = 0.0
        max_encode = 0.0
        max_mux = 0.0

        def put_or_stop(item) -> bool:
            while not self._stop.is_set():
                try:
                    encode_q.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def force_put_sentinel() -> None:
            while True:
                try:
                    encode_q.put(sentinel, timeout=0.05)
                    return
                except queue.Full:
                    if self._stop.is_set():
                        try:
                            old = encode_q.get_nowait()
                            if old is not sentinel and isinstance(old, tuple) and len(old) >= 4:
                                self.matter.release_nv12_output_slot(old[3])
                        except queue.Empty:
                            pass

        def matting_worker() -> None:
            last_src_idx = -1
            try:
                assert self._dec is not None
                for i in range(target):
                    if self._stop.is_set():
                        break
                    out_idx = start_out + i
                    src_idx = min(len(self._dec) - 1, cfr_source_index(out_idx, source_fps, fps))
                    if src_idx <= last_src_idx:
                        src_idx = min(len(self._dec) - 1, last_src_idx + 1)
                    last_src_idx = src_idx
                    t0 = time.perf_counter()
                    frame = self._dec.frame_at(src_idx)
                    t1 = time.perf_counter()
                    slot = None
                    try:
                        slot = self.matter.acquire_nv12_output_slot(
                            out_h,
                            out_w,
                            timeout=config.PASSTHROUGH_NV12_SLOT_WAIT_SEC,
                        )
                        if isinstance(frame, GpuP016Frame):
                            out_nv12, _ = self.matter.composite_green_gpu_p016_frame_to_gpu_nv12_profile(
                                frame,
                                shift_bits=config.PASSTHROUGH_PYNV_10BIT_SHIFT,
                                out_h=out_h,
                                out_w=out_w,
                                out_slot=slot,
                            )
                        else:
                            out_nv12, _ = self.matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(
                                frame,
                                out_h=out_h,
                                out_w=out_w,
                                out_slot=slot,
                            )
                        t2 = time.perf_counter()
                        cuda_stream = getattr(matting_module, "_CUDA_STREAM", None)
                        if cuda_stream is not None:
                            cuda_stream.synchronize()
                        t3 = time.perf_counter()
                        self._apply_subtitle_overlay(out_nv12, subtitle_renderer, out_idx / fps if fps > 0 else 0.0)
                        app_frame = GpuNv12AppFrame(out_nv12, out_w, out_h)
                        if not put_or_stop((i, src_idx, app_frame, slot, t0, t1, t2, t3)):
                            self.matter.release_nv12_output_slot(slot)
                            slot = None
                            break
                        slot = None
                    except Exception:
                        self.matter.release_nv12_output_slot(slot)
                        raise
            except Exception as e:
                errors.append(f"matting: {e}\n{traceback.format_exc(limit=8)}")
                self._stop.set()
            finally:
                force_put_sentinel()

        worker = threading.Thread(target=matting_worker, name=f"pynv-two-stage-matting-{self.sid}", daemon=True)
        worker.start()
        last_src_idx = -1
        try:
            while not self._stop.is_set():
                item = encode_q.get()
                if item is sentinel:
                    break
                i, src_idx, app_frame, slot, t0, t1, t2, t3 = item
                if producer_pacing and self.container == "mpegts" and fps > 0:
                    due = t_start + (i / fps)
                    now = time.perf_counter()
                    if due > now:
                        self._stop.wait(due - now)
                        if self._stop.is_set():
                            self.matter.release_nv12_output_slot(slot)
                            break
                last_src_idx = src_idx
                flags = 0
                if i == 0:
                    flags = int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
                    log.info(
                        "[PYNV][%d] seek start: requested_sec=%.3f out_idx=%d src_idx=%d fps=%.3f source_fps=%.3f",
                        self.sid, self.start_sec, start_out + i, src_idx, fps, source_fps,
                    )
                try:
                    bitstream = self._enc.Encode(app_frame, flags)
                except Exception:
                    self.matter.release_nv12_output_slot(slot)
                    raise
                pending_nv12_slots.append(slot)
                while len(pending_nv12_slots) > max_pending_nv12_slots:
                    self.matter.release_nv12_output_slot(pending_nv12_slots.pop(0))
                t4 = time.perf_counter()
                if bitstream:
                    if self.frames_produced == 0:
                        log.info(
                            "[PYNV][%d] first bitstream len=%d hevc_nals=%s",
                            self.sid,
                            len(bitstream),
                            _hevc_nal_summary(bitstream) if PYNV_OUTPUT_CODEC == "hevc" else "n/a",
                        )
                    mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                    if self._stop.is_set() or not mux_stdin or mux_stdin.closed:
                        log.info("[PYNV][%d] mux stdin closed before write at frame=%d", self.sid, i + 1)
                        break
                    try:
                        write_started = time.perf_counter()
                        self._mark_first_write()
                        mux_stdin.write(bitstream)
                        if not self._real_video_started.is_set():
                            self._real_video_started.set()
                            log.info("[PYNV][%d] first real video bitstream written; slate audio may switch", self.sid)
                            self._log_vram("first_real_video_bitstream")
                        write_elapsed = time.perf_counter() - write_started
                        if write_elapsed >= 1.0:
                            log.warning(
                                "[PYNV][%d] mux stdin write slow: frame=%d len=%d elapsed=%.3fs video_rc=%s final_rc=%s",
                                self.sid,
                                i + 1,
                                len(bitstream),
                                write_elapsed,
                                self._video_mux.poll() if self._video_mux is not None else None,
                                self._mux.poll() if self._mux is not None else None,
                            )
                    except (BrokenPipeError, OSError, ValueError) as e:
                        log.info("[PYNV][%d] mux stdin write stopped at frame=%d: %s", self.sid, i + 1, e)
                        break
                t5 = time.perf_counter()
                sum_decode += t1 - t0
                sum_composite += t2 - t1
                sum_sync += t3 - t2
                sum_encode += t4 - t3
                sum_mux += t5 - t4
                max_decode = max(max_decode, t1 - t0)
                max_composite = max(max_composite, t2 - t1)
                max_sync = max(max_sync, t3 - t2)
                max_encode = max(max_encode, t4 - t3)
                max_mux = max(max_mux, t5 - t4)
                if bitstream:
                    interval_bytes += len(bitstream)
                self.frames_produced = i + 1
                if config.DEBUG_LOGS and self.frames_produced % _DIAG_INTERVAL == 0:
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    interval_elapsed = max(0.001, time.perf_counter() - interval_start)
                    interval_frames = _DIAG_INTERVAL
                    log.info(
                        "[PYNV][%d] frame %d/%d fps=%.2f interval_fps=%.2f src_idx=%d bytes=%d out_bps=%.1fM stage_avg_ms decode=%.2f composite=%.2f sync=%.2f encode=%.2f mux=%.2f mat_avg_ms pre=%.2f ort=%.2f kernel=%.2f stage_max_ms decode=%.2f composite=%.2f sync=%.2f encode=%.2f mux=%.2f mat_max_ms pre=%.2f ort=%.2f kernel=%.2f",
                        self.sid,
                        self.frames_produced,
                        target,
                        self.frames_produced / elapsed,
                        interval_frames / interval_elapsed,
                        last_src_idx,
                        self.bytes_emitted,
                        (interval_bytes * 8.0 / interval_elapsed) / 1_000_000.0,
                        (sum_decode / interval_frames) * 1000.0,
                        (sum_composite / interval_frames) * 1000.0,
                        (sum_sync / interval_frames) * 1000.0,
                        (sum_encode / interval_frames) * 1000.0,
                        (sum_mux / interval_frames) * 1000.0,
                        (sum_mat_pre / interval_frames) * 1000.0,
                        (sum_mat_ort / interval_frames) * 1000.0,
                        (sum_mat_kernel / interval_frames) * 1000.0,
                        max_decode * 1000.0,
                        max_composite * 1000.0,
                        max_sync * 1000.0,
                        max_encode * 1000.0,
                        max_mux * 1000.0,
                        max_mat_pre * 1000.0,
                        max_mat_ort * 1000.0,
                        max_mat_kernel * 1000.0,
                    )
                    interval_start = time.perf_counter()
                    interval_bytes = 0
                    sum_decode = sum_composite = sum_sync = sum_encode = sum_mux = 0.0
                    max_decode = max_composite = max_sync = max_encode = max_mux = 0.0
        finally:
            if worker.is_alive():
                self._stop.set()
                worker.join(timeout=2.0)
        if errors:
            raise RuntimeError(errors[0])
        if not self._stop.is_set():
            log.info("[PYNV][%d] EndEncode begin frames=%d", self.sid, self.frames_produced)
            tail = self._enc.EndEncode()
            if tail:
                mux_stdin = self._video_mux.stdin if self._video_mux is not None else self._mux.stdin
                if mux_stdin:
                    self._mark_first_write()
                    mux_stdin.write(tail)
                log.info("[PYNV][%d] EndEncode tail len=%d", self.sid, len(tail))
            log.info("[PYNV][%d] EndEncode done", self.sid)
        while pending_nv12_slots:
            self.matter.release_nv12_output_slot(pending_nv12_slots.pop(0))
        log.info("[PYNV][%d] worker mode=two_stage green done frames=%d", self.sid, self.frames_produced)

    def _post_sentinel(self) -> None:
        if self._queue is None or self._loop is None or self._loop.is_closed():
            return
        queue_ref = self._queue

        def _put_sentinel() -> None:
            try:
                queue_ref.put_nowait(None)
                log.info("[PYNV][%d] sentinel posted", self.sid)
            except asyncio.QueueFull:
                try:
                    queue_ref.get_nowait()
                    queue_ref.put_nowait(None)
                    log.info("[PYNV][%d] sentinel posted after dropping queued chunk", self.sid)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    log.debug("[PYNV][%d] sentinel skipped because async queue is full", self.sid)

        try:
            self._loop.call_soon_threadsafe(_put_sentinel)
        except RuntimeError:
            pass

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._worker_loop, name="pynv-worker", daemon=True)
        self._worker.start()
        log.info("[PYNV][%d] iter start", self.sid)
        try:
            chunks = 0
            light_match_version = get_light_match().version
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    log.info("[PYNV][%d] iter got sentinel chunks=%d bytes=%d", self.sid, chunks, self.bytes_emitted)
                    break
                current_light_match_version = get_light_match().version
                if (
                    config.LIGHT_MATCH_FLUSH_QUEUES
                    and self.container == "mpegts"
                    and current_light_match_version != light_match_version
                ):
                    dropped_chunks, dropped_bytes, saw_sentinel = _drain_async_queue_nowait(self._queue)
                    log.info(
                        "[PYNV][%d] light match changed v%d->v%d; dropped current mux chunk plus pending chunks=%d bytes=%d sentinel=%s",
                        self.sid,
                        light_match_version,
                        current_light_match_version,
                        dropped_chunks,
                        dropped_bytes + len(chunk),
                        saw_sentinel,
                    )
                    light_match_version = current_light_match_version
                    continue
                self.bytes_emitted += len(chunk)
                chunks += 1
                if chunks == 1:
                    log.info("[PYNV][%d] iter first chunk len=%d bytes=%d", self.sid, len(chunk), self.bytes_emitted)
                elif chunks % 512 == 0:
                    log.debug("[PYNV][%d] iter yield chunk=%d len=%d bytes=%d", self.sid, chunks, len(chunk), self.bytes_emitted)
                yield chunk
        finally:
            log.info("[PYNV][%d] iter finally bytes=%d frames=%d", self.sid, self.bytes_emitted, self.frames_produced)
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        log.info("[PYNV][%d] close begin bytes=%d frames=%d", self.sid, self.bytes_emitted, self.frames_produced)
        self._log_vram("close_begin")
        self._stop_audio_procs()
        for proc in (self._video_mux, self._mux):
            if not proc:
                continue
            self._stop_proc(proc, "mux ffmpeg", wait_timeout=0.1)
        current = threading.current_thread()
        worker_alive_after_first_join = False
        for thread in (self._slate_audio_thread, self._slate_audio_cache_thread, self._worker, self._reader, self._stderr_reader):
            if thread and thread.is_alive() and thread is not current:
                log.info("[PYNV][%d] close: waiting thread name=%s", self.sid, thread.name)
                timeout = (
                    config.PASSTHROUGH_CLOSE_WORKER_TIMEOUT_SEC
                    if thread is self._worker
                    else _THREAD_JOIN_TIMEOUT
                )
                thread.join(timeout=timeout)
                if thread is self._worker and thread.is_alive():
                    worker_alive_after_first_join = True
                    log.info(
                        "[PYNV][%d] close: worker still alive after %.1fs, stopping decoder",
                        self.sid,
                        timeout,
                    )
        self._enc = None
        if worker_alive_after_first_join:
            if self._dec:
                self._dec = None
            self._log_vram("worker_join_timeout_decoder_detached")
            log.warning(
                "[PYNV][%d] close: worker did not stop after first join; decoder stop is skipped because native decoder stop is not reliable after a stuck worker",
                self.sid,
            )
        elif self._dec:
            dec = self._dec
            self._dec = None
            sid = self.sid

            def _stop_decoder() -> None:
                try:
                    log.info("[PYNV][%d] decoder stop thread begin", sid)
                    dec.stop()
                    log.info("[PYNV][%d] decoder stop thread done", sid)
                except Exception as e:
                    log.warning("[PYNV][%d] decoder stop thread failed: %s", sid, e)

            dec_thread = threading.Thread(target=_stop_decoder, name=f"pynv-dec-stop-{sid}", daemon=True)
            dec_thread.start()
            dec_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            self._log_vram("decoder_stop_joined")
        try:
            gc.collect()
        except Exception:
            pass
        for thread in (self._reader, self._stderr_reader, self._worker):
            if thread and thread.is_alive():
                log.info("[PYNV][%d] close: thread still alive name=%s", self.sid, thread.name)
                self._log_vram(f"thread_still_alive:{thread.name}")
                self._log_thread_stack(thread)
                if thread is self._worker:
                    _pynv_runtime_tainted.set()
                    log.error(
                        "[PYNV][%d] PyNv runtime marked tainted because worker did not stop; restart the server before continuing alpha passthrough",
                        self.sid,
                    )
        self._log_vram("close_done")
        log.info("[PYNV][%d] close done", self.sid)
