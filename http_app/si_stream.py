"""Realtime SI audio mixing for virtual DLNA MP4 items."""
from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.ffmpeg_io import FFMPEG, probe_cached
from utils.logger import get
from utils.runtime_settings import SIMixRuntime, get_si_mix
from utils.si_filter import SIMixParams
from utils.subprocess_hidden import hidden_subprocess_kwargs


AUDIO_BITRATE_BPS = 192_000
SIZE_OVERHEAD_FACTOR = 1.05
MIN_ESTIMATED_SIZE = 64 * 1024
DEFAULT_CHUNK_SIZE = 64 * 1024
DEFAULT_REUSE_TOLERANCE_BYTES = 1024 * 1024
DEFAULT_SEEK_COOLDOWN_SECONDS = 0.2
DEFAULT_ESTIMATE_CACHE_LIMIT = 512
DEFAULT_STARTUP_PROBE_WINDOW_SECONDS = 45.0
SI_VIDEO_EXTS = {".mp4"}

log = get("si_stream")


def parse_range_header(value: str | None) -> tuple[int, int | None, bool]:
    """Parse one HTTP bytes range, falling back to a full stream."""
    header = (value or "").strip()
    if not header.lower().startswith("bytes="):
        return 0, None, False
    spec = header[6:].split(",", 1)[0].strip()
    if not spec or spec.startswith("-") or "-" not in spec:
        return 0, None, True
    start_text, end_text = spec.split("-", 1)
    try:
        start = int(start_text)
        end = int(end_text) if end_text.strip() else None
    except ValueError:
        return 0, None, True
    if start < 0 or (end is not None and end < start):
        return 0, None, True
    return start, end, True


@dataclass(frozen=True)
class SIStreamOpenResult:
    chunks: Iterator[bytes]
    content_length: int
    total_size: int
    status_code: int
    start: int
    end: int
    start_time: float


class ConfigHolder:
    def __init__(self, config: SIMixParams | None = None) -> None:
        self._config = config or SIMixParams()
        self._lock = threading.RLock()

    def get(self) -> SIMixParams:
        with self._lock:
            return self._config

    def set(self, config: SIMixParams) -> None:
        with self._lock:
            self._config = config


class LiveStreamSession:
    """One active ffmpeg stdout pipe for a virtual SI stream."""

    def __init__(
        self,
        video: Path,
        si_wav: Path,
        config: SIMixParams,
        start_time: float,
        estimated_total: int,
        start_byte: int = 0,
    ) -> None:
        self.video = video
        self.si_wav = si_wav
        self.config = config
        self.estimated_total = max(1, int(estimated_total))
        self.start_time = max(0.0, float(start_time))
        self.byte_cursor = max(0, int(start_byte))
        self.last_used = time.monotonic()
        self.proc: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()
        self._closed = False
        self._start_ffmpeg(self.start_time)

    def _start_ffmpeg(self, start_time: float) -> None:
        seek = f"{max(0.0, start_time):.3f}"
        # Output MPEG-TS, not fragmented MP4. SKYBOX (and the other VR players we
        # captured in tools/si_proto) handle a linear MPEG-TS live stream cleanly,
        # but treated fMP4 as a pseudo-file and issued overlapping Range probes
        # (3.6x-6.35x the byte volume) that stalled playback. This mirrors the
        # passthrough_live transport, which already serves video/MP2T. Video is
        # stream-copied; only the SI mix audio is encoded, so the stream is
        # produced faster than realtime with no full-file cache.
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            seek,
            "-i",
            str(self.video),
            "-ss",
            seek,
            "-i",
            str(self.si_wav),
            "-filter_complex",
            self.config.filter_string(),
            "-map",
            "0:v",
            "-c:v",
            "copy",
            "-map",
            "[si_track]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-muxpreload",
            "0",
            "-muxdelay",
            "0",
            "-f",
            "mpegts",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
        log.info("started SI ffmpeg stream video=%s si=%s seek=%s", self.video, self.si_wav, seek)

    def is_usable(self) -> bool:
        proc = self.proc
        return not self._closed and proc is not None and proc.stdout is not None and proc.poll() is None

    def read(self, n: int) -> bytes:
        with self.lock:
            proc = self.proc
            if not self.is_usable() or proc is None or proc.stdout is None:
                return b""
            try:
                chunk = proc.stdout.read(max(1, int(n)))
            except (OSError, ValueError):
                return b""
            if chunk:
                self.byte_cursor += len(chunk)
                self.last_used = time.monotonic()
            return chunk

    def discard(self, n: int) -> int:
        remaining = max(0, int(n))
        discarded = 0
        while remaining > 0:
            chunk = self.read(min(DEFAULT_CHUNK_SIZE, remaining))
            if not chunk:
                break
            discarded += len(chunk)
            remaining -= len(chunk)
        return discarded

    def close(self) -> None:
        self._closed = True
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception as exc:
            log.warning("failed to terminate SI ffmpeg stream for %s: %s", self.video, exc)


class SIStreamService:
    """Coordinates active SI sessions and runtime config hot reloads."""

    def __init__(
        self,
        *,
        config_holder: ConfigHolder | None = None,
        session_factory: Callable[..., LiveStreamSession] = LiveStreamSession,
        reuse_tolerance_bytes: int = DEFAULT_REUSE_TOLERANCE_BYTES,
        seek_cooldown_seconds: float = DEFAULT_SEEK_COOLDOWN_SECONDS,
    ) -> None:
        runtime = get_si_mix()
        self._config_holder = config_holder or ConfigHolder(runtime.params())
        self._runtime_version = runtime.version
        self._session_factory = session_factory
        self._reuse_tolerance_bytes = max(0, int(reuse_tolerance_bytes))
        self._seek_cooldown_seconds = max(0.0, float(seek_cooldown_seconds))
        self._sessions: dict[str, Any] = {}
        self._last_start_at: dict[str, float] = {}
        self._estimate_cache: dict[str, tuple[int, int, int, float, int]] = {}
        self._estimate_cache_limit = DEFAULT_ESTIMATE_CACHE_LIMIT
        self._startup_probe_window_seconds = DEFAULT_STARTUP_PROBE_WINDOW_SECONDS
        self._sessions_lock = threading.Lock()

    def current_config(self) -> SIMixParams:
        runtime = get_si_mix()
        if runtime.version != self._runtime_version:
            self.reload_config(runtime.params(), version=runtime.version)
        return self._config_holder.get()

    def has_si_source(self, video: Path) -> Path | None:
        video = Path(video)
        if video.suffix.lower() not in SI_VIDEO_EXTS:
            return None
        sibling = video.with_suffix(".si.wav")
        return sibling if sibling.is_file() else None

    def estimate_output_size(self, video: Path) -> int:
        video = Path(video).resolve()
        try:
            stat = video.stat()
            mtime_ns = int(stat.st_mtime_ns)
            size = int(stat.st_size)
        except OSError:
            mtime_ns = 0
            size = 0
        key = str(video)
        cached = self._estimate_cache.get(key)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            return cached[4]

        duration = self._duration(video)
        audio_size = int(AUDIO_BITRATE_BPS * duration / 8) if duration > 0 else 0
        estimated = int((size + audio_size) * SIZE_OVERHEAD_FACTOR)
        estimated = max(estimated, size, MIN_ESTIMATED_SIZE)
        if len(self._estimate_cache) >= self._estimate_cache_limit:
            self._estimate_cache.pop(next(iter(self._estimate_cache)))
        self._estimate_cache[key] = (mtime_ns, size, audio_size, duration, estimated)
        return estimated

    def _duration(self, video: Path) -> float:
        try:
            info = probe_cached(video)
            return max(0.0, float(info.duration or 0.0))
        except Exception as exc:
            log.warning("SI duration probe failed for %s: %s", video, exc)
            return 0.0

    def _start_time_for_range(self, video: Path, range_start: int, total_size: int) -> float:
        duration = self._duration(video)
        if duration <= 0 or total_size <= 0:
            return 0.0
        ratio = min(1.0, max(0.0, range_start / total_size))
        return ratio * duration

    def _session_key(self, video: Path, client_id: str | None = None) -> str:
        base = str(Path(video).resolve())
        normalized_client = str(client_id or "").strip()
        return f"{base}\0{normalized_client}" if normalized_client else base

    def is_startup_probe_range(
        self,
        video: Path,
        range_start: int,
        range_end: int | None,
        *,
        client_id: str | None,
        user_agent: str | None,
    ) -> tuple[bool, int]:
        """Detect SKYBOX/libmpv startup byte probes without disrupting the main stream."""
        total_size = self.estimate_output_size(video)
        ua = (user_agent or "").lower()
        if "skybox" not in ua and "libmpv" not in ua:
            return False, total_size
        if range_end is not None or int(range_start) <= 0:
            return False, total_size
        key = self._session_key(video, client_id)
        with self._sessions_lock:
            session = self._sessions.get(key)
            last_start = self._last_start_at.get(key, 0.0)
        if session is None:
            return False, total_size
        if hasattr(session, "is_usable") and not session.is_usable():
            return False, total_size
        age = time.monotonic() - last_start if last_start > 0 else 0.0
        if age > self._startup_probe_window_seconds:
            return False, total_size
        cursor = int(getattr(session, "byte_cursor", 0))
        if int(range_start) <= cursor + self._reuse_tolerance_bytes:
            return False, total_size
        # SKYBOX/libmpv can issue many open-ended byte probes across the file
        # immediately after the real bytes=0- stream starts. Those offsets do
        # not map to stable bytes in our generated fMP4 stream, so treating
        # them as seeks causes a restart storm and starves the main stream.
        return int(range_start) < total_size, total_size

    def _can_reuse(self, session: Any, config: SIMixParams, si_wav: Path, range_start: int) -> bool:
        if getattr(session, "config", None) != config:
            return False
        if Path(getattr(session, "si_wav", "")) != si_wav:
            return False
        if hasattr(session, "is_usable") and not session.is_usable():
            return False
        cursor = int(getattr(session, "byte_cursor", 0))
        return cursor <= range_start <= cursor + self._reuse_tolerance_bytes

    def _close_session(self, session: Any) -> None:
        try:
            session.close()
        except Exception as exc:
            log.warning("failed to close SI stream session: %s", exc)

    def _get_or_start_session(
        self,
        video: Path,
        si_wav: Path,
        config: SIMixParams,
        range_start: int,
        total_size: int,
        client_id: str | None,
    ) -> tuple[Any, float]:
        key = self._session_key(video, client_id)
        with self._sessions_lock:
            session = self._sessions.get(key)
            if session is not None and self._can_reuse(session, config, si_wav, range_start):
                return session, float(getattr(session, "start_time", 0.0))
            if session is not None:
                self._close_session(session)
                self._sessions.pop(key, None)

            last_start = self._last_start_at.get(key, 0.0)
            wait_seconds = self._seek_cooldown_seconds - (time.monotonic() - last_start)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            start_time = self._start_time_for_range(video, range_start, total_size)
            session = self._session_factory(
                video=video,
                si_wav=si_wav,
                config=config,
                start_time=start_time,
                estimated_total=total_size,
                start_byte=range_start,
            )
            self._sessions[key] = session
            self._last_start_at[key] = time.monotonic()
            return session, start_time

    def _drop_session_if_current(
        self,
        video: Path,
        session: Any,
        *,
        client_id: str | None,
        close: bool,
    ) -> None:
        key = self._session_key(video, client_id)
        with self._sessions_lock:
            if self._sessions.get(key) is not session:
                return
            self._sessions.pop(key, None)
        if close:
            self._close_session(session)

    def open_stream(
        self,
        video: Path,
        range_start: int = 0,
        range_end: int | None = None,
        *,
        range_requested: bool = False,
        client_id: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> SIStreamOpenResult:
        video = Path(video).resolve()
        config = self.current_config()
        if not config.enabled:
            raise FileNotFoundError("SI streaming is disabled")
        si_wav = self.has_si_source(video)
        if si_wav is None:
            raise FileNotFoundError("No sibling SI WAV file")

        total_size = self.estimate_output_size(video)
        safe_start = min(max(0, int(range_start)), max(0, total_size - 1))
        safe_end = min(int(range_end), total_size - 1) if range_end is not None else total_size - 1
        if safe_end < safe_start:
            safe_end = total_size - 1
        content_length = max(0, safe_end - safe_start + 1)
        status_code = 206 if range_requested else 200
        session, start_time = self._get_or_start_session(video, si_wav, config, safe_start, total_size, client_id)

        def chunks() -> Iterator[bytes]:
            remaining = content_length
            saw_eof = False
            try:
                cursor = int(getattr(session, "byte_cursor", safe_start))
                if cursor < safe_start and hasattr(session, "discard"):
                    session.discard(safe_start - cursor)
                while remaining > 0:
                    chunk = session.read(min(chunk_size, remaining))
                    if not chunk:
                        saw_eof = True
                        break
                    remaining -= len(chunk)
                    yield chunk
            finally:
                if saw_eof:
                    self._drop_session_if_current(video, session, client_id=client_id, close=True)

        return SIStreamOpenResult(
            chunks=chunks(),
            content_length=content_length,
            total_size=total_size,
            status_code=status_code,
            start=safe_start,
            end=safe_end,
            start_time=start_time,
        )

    def reload_config(self, new_config: SIMixParams, *, version: int | None = None) -> None:
        self._config_holder.set(new_config)
        if version is not None:
            self._runtime_version = int(version)
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._close_session(session)
        log.info("reloaded DLNA SI config: %s version=%s", new_config.to_dict(), self._runtime_version)

    def shutdown(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            self._close_session(session)


def iter_si_mpegts(
    video: Path,
    si_wav: Path,
    config: SIMixParams,
    start_time: float,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[bytes]:
    """Yield a realtime MPEG-TS SI mix stream starting at ``start_time`` seconds.

    Each call spawns one ffmpeg process (source video copied, SI mix encoded) and
    streams its stdout until EOF. Seeking is handled by the caller re-requesting a
    new ``?t=`` offset, exactly like the passthrough_live transport.
    """
    session = LiveStreamSession(
        video=Path(video),
        si_wav=Path(si_wav),
        config=config,
        start_time=max(0.0, float(start_time)),
        estimated_total=1,
        start_byte=0,
    )
    try:
        while True:
            chunk = session.read(max(1, int(chunk_size)))
            if not chunk:
                break
            yield chunk
    finally:
        session.close()


_service_lock = threading.Lock()
_service: SIStreamService | None = None


def get_si_stream_service() -> SIStreamService:
    global _service
    with _service_lock:
        if _service is None:
            _service = SIStreamService()
        return _service


def reload_si_stream_service(runtime: SIMixRuntime | None = None) -> None:
    with _service_lock:
        service = _service
    if service is None:
        return
    runtime = runtime or get_si_mix()
    service.reload_config(runtime.params(), version=runtime.version)


def shutdown_si_stream_service() -> None:
    global _service
    with _service_lock:
        service = _service
        _service = None
    if service is not None:
        service.shutdown()
