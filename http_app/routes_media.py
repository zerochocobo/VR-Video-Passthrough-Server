"""Media HTTP routes.

- /media/{name}: serve the source file with standard HTTP Range support.
- /passthrough/{name}: pseudo-VOD passthrough with byte/time seek mapping.
- /passthrough_live/{name}: MPEG-TS live passthrough for clients that dislike pseudo-VOD byte seeking.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Header, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from config import (
    HTTP_PORT,
    DLNA_IMAGE_ENABLED,
    IMAGE_EXTS,
    IMAGE_MIME_BY_EXT,
    LAN_IP,
    PASSTHROUGH_CONTAINER,
    PASSTHROUGH_BUSY_WAIT_SEC,
    PASSTHROUGH_LIVE_ADAPTIVE_FPS,
    PASSTHROUGH_LIVE_CACHE_BYTES,
    PASSTHROUGH_LIVE_CACHE_TTL_SEC,
    PASSTHROUGH_LIVE_HIGH_BITRATE_BPS,
    PASSTHROUGH_LIVE_HIGH_BITRATE_FPS,
    PASSTHROUGH_LIVE_DEFAULT_PROFILE,
    PASSTHROUGH_LIVE_FIRST_CHUNK_TIMEOUT_SEC,
    PASSTHROUGH_LIVE_LAVF_POLICY,
    PASSTHROUGH_LIVE_SUB_QUEUE_CHUNKS,
    PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC,
    PASSTHROUGH_LIVE_VLC_PREROLL_BYTES,
    PASSTHROUGH_LIVE_VLC_PREROLL_TIMEOUT_SEC,
    PASSTHROUGH_LIVE_VLC_PSEUDO_VOD,
    PASSTHROUGH_OUTPUT_MODE,
    PASSTHROUGH_AUDIO_MPEGTS_VLC,
    PASSTHROUGH_FALLBACK_MAX_FPS,
    PASSTHROUGH_GOP,
    PASSTHROUGH_HEVC_BITRATE,
    PASSTHROUGH_MAX_FPS,
    PASSTHROUGH_MAX_CONCURRENT,
    PASSTHROUGH_MKV_LIVE_POLICY,
    PASSTHROUGH_PAD_TO_LENGTH,
    PASSTHROUGH_SEND_MIN_BPS,
    PASSTHROUGH_SEND_PACING_MULTIPLIER,
    PASSTHROUGH_SEND_REALTIME_PACING,
    PASSTHROUGH_SEEK_MODE,
    PASSTHROUGH_SEEK_ENABLED,
    PASSTHROUGH_SEEK_CONTAINER,
    PASSTHROUGH_SEEK_HEADER_BYTES,
    PASSTHROUGH_SEEK_PROFILES,
    PASSTHROUGH_SEEK_ROUTE_POLICY,
    SI_PROGRESSIVE_ENABLED,
    DEBUG_LOGS,
    LIVE_REQUEST_HEADER_DUMP,
    LIGHT_MATCH_FLUSH_QUEUES,
    MEDIA_LIBRARY,
    ROOT,
    USE_PYNV,
    VIDEO_EXTS,
)
from dlna.profiles import passthrough_dlna_pn, passthrough_frame_rate
from http_app.si_stream import DEFAULT_CHUNK_SIZE, get_si_stream_service, iter_si_mpegts, parse_range_header
from pipeline.ffmpeg_io import probe_cached
from pipeline.si_virtual_mp4 import build_progressive_si_virtual_mp4, iter_virtual_range
from pipeline.matting import acquire_matter, release_matter
from pipeline.stream import PassthroughStream
from pipeline.pynv_stream import PYNV_BACKEND_LABEL, PYNV_OUTPUT_CODEC, PyNvPassthroughStream
from pipeline.thumbnail import get_thumb
from utils.bitrate_estimator import estimate_for_media, parse_bitrate, record_actual_bps
from utils.byte_seek_map import map_byte_start_to_time
from utils.logger import get
from utils.player_compat import (
    is_lavf_user_agent,
    is_libmpv_screenshot_probe_ua,
    is_nplayer_user_agent,
    live_response_profile_from_ua,
)
from utils.request_history import annotate_request
from utils.runtime_settings import get_light_match
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.mkv_cues import probe_mkv_cues
from utils.offline_outputs import has_offline_two_dvr_output
from utils.subtitles import find_external_subtitles, is_subtitle_path, subtitle_mime
from utils.video_metadata import probe_video_metadata, select_backend
from utils.vr_naming import has_vr_filename_marker, is_half_equirectangular_source

log = get("media")
router = APIRouter()
DLNA_FLAGS_BASE = "01700000000000000000000000000000"
DLNA_FLAGS_TIME_SEEK = "41700000000000000000000000000000"
DLNA_FLAGS_BYTE_AND_TIME_SEEK = "01F00000000000000000000000000000"
# Historical compatibility note: the legacy passthrough/live branches in this
# project treat OP=01 as the byte-style compatibility advertisement and OP=10
# as the time-style advertisement, even though DLNA's BA bit wording is often
# read the other way around. Do not "fix" these legacy values without
# re-testing the affected players. The new seek route uses OP=11, which is
# unambiguous because both bits are set. Its flags intentionally use file-like
# transfer bits only; do not re-add lop-npt/lop-bytes flags such as 0x6170...
# while advertising OP=11, because clients may downgrade that contradiction to
# a limited/live-style resource.
DLNA_OP_BYTE_SEEK = "01"
DLNA_OP_TIME_SEEK = "10"
DLNA_OP_BYTE_AND_TIME_SEEK = "11"
_request_ids = itertools.count(1)

# ---- Passthrough concurrency guard ----
# Keep passthrough concurrency low to avoid NVENC session exhaustion, blocked
# ffmpeg pipes, and concurrent access to the shared Matter/ONNX session.
_active_lock = asyncio.Lock()
_active_streams: dict[object, tuple[str, str]] = {}
_active_started: dict[object, float] = {}
# Matter instance bound to each active slot key (slot_token or stream). Tracked
# separately from _active_streams so the lifecycle survives the slot_token ->
# stream key swap inside _replace_active_slot.
_active_matter: dict[object, object] = {}
_probe_cache_lock = asyncio.Lock()
_probe_cache: dict[str, bytes] = {}
# Benign without an async lock: the estimate inputs are deterministic, so a
# concurrent miss only recalculates the same declared size for the same key.
_seek_declared_size_cache: dict[tuple, int] = {}
_thumb_lock = asyncio.Lock()
_PROBE_CACHE_LIMIT = 16 * 1024 * 1024
_PROBE_CACHE_TOTAL_LIMIT = 64 * 1024 * 1024
_SEEK_DECLARED_SIZE_CACHE_LIMIT = 512
_SMALL_PROBE_LIMIT = 64 * 1024
_PREFIX_CACHE_WAIT_SEC = 5.0
_PREFIX_CACHE_IDLE_SEC = 2.0
_TAIL_PROBE_RATIO = 0.95
# nPlayer performs startup EOF probes with a few growing open-ended tail
# ranges before the visible progress bar is usable. Treat these as probes so
# they do not start a real producer at the final second and auto-advance.
_TAIL_PROBE_MAX_BYTES = 2 * 1024 * 1024
_SEEK_PREFIX_CACHE_FLUSH_STEP = 64 * 1024
_SEEK_ROUTE_SUFFIXES = (
    (".seek.ts", "mpegts"),
    (".seek.mp4", "mp4"),
)
_LIVE_SEND_PACE_CHUNK_BYTES = 64 * 1024
_LIVE_SEND_PACE_BURST_SEC = 1.5
_LIVE_PROGRESS_INTERVAL_BYTES = 50 * 1024 * 1024
_LIVE_FIRST_CHUNK_TIMEOUT_SEC = PASSTHROUGH_LIVE_FIRST_CHUNK_TIMEOUT_SEC
_LIVE_VLC_PREROLL_BYTES = PASSTHROUGH_LIVE_VLC_PREROLL_BYTES
_LIVE_VLC_PREROLL_TIMEOUT_SEC = PASSTHROUGH_LIVE_VLC_PREROLL_TIMEOUT_SEC
_LIVE_REQUEST_DUMP_DIR = ROOT / "debug_output" / "live_requests"
_live_session_lock = asyncio.Lock()
_live_sessions: dict[tuple[str, str, float, str, float, str], "LiveSession"] = {}
_live_starting: dict[tuple[str, str, float, str, float, str], float] = {}
_LIVE_NPLAYER_START_DEBOUNCE_SEC = max(1.5, _LIVE_FIRST_CHUNK_TIMEOUT_SEC)
_SI_STARTUP_PROBE_BYTES = 1024 * 1024


def _seek_container() -> str:
    return PASSTHROUGH_SEEK_CONTAINER if PASSTHROUGH_SEEK_CONTAINER in {"mpegts", "mp4"} else "mpegts"


def _seek_media_type(container: str | None = None) -> str:
    return "video/mp4" if (container or _seek_container()) == "mp4" else "video/MP2T"


def _seek_dlna_pn(container: str | None = None) -> str:
    return "HEVC_MP4_MAIN" if (container or _seek_container()) == "mp4" else "HEVC_TS_NA_ISO"


def _split_seek_route_name(name: str) -> tuple[str, str | None]:
    decoded = unquote(name)
    lower = decoded.lower()
    for suffix, container in _SEEK_ROUTE_SUFFIXES:
        if lower.endswith(suffix):
            return decoded[: -len(suffix)], container
    return decoded, None


_LIVE_END = object()
_SI_EOF = object()


def _drain_live_queue_nowait(q: asyncio.Queue[bytes | object]) -> tuple[int, int, bool]:
    chunks = 0
    bytes_dropped = 0
    saw_end = False
    while True:
        try:
            item = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item is _LIVE_END:
            saw_end = True
            try:
                q.put_nowait(_LIVE_END)
            except asyncio.QueueFull:
                pass
            break
        if isinstance(item, (bytes, bytearray, memoryview)):
            chunks += 1
            bytes_dropped += len(item)
    return chunks, bytes_dropped, saw_end


def _set_probe_cache_locked(key: str, data: bytes) -> None:
    """Store a probe prefix while keeping the process-wide cache bounded."""
    if not data:
        _probe_cache.pop(key, None)
        return
    _probe_cache.pop(key, None)
    _probe_cache[key] = data[:_PROBE_CACHE_LIMIT]
    total = sum(len(value) for value in _probe_cache.values())
    while total > _PROBE_CACHE_TOTAL_LIMIT and _probe_cache:
        old_key = next(iter(_probe_cache))
        if old_key == key and len(_probe_cache) == 1:
            break
        old_value = _probe_cache.pop(old_key)
        total -= len(old_value)


def _release_active_slot_nowait(stream: object) -> None:
    removed = _active_streams.pop(stream, None)
    _active_started.pop(stream, None)
    matter = _active_matter.pop(stream, None)
    if removed is not None:
        log.info("passthrough active slot released: active=%d owner=%s", len(_active_streams), _owner_log_value(removed))
    if matter is not None:
        release_matter(matter)


async def _clear_live_starting(key: tuple[str, str, float, str, float, str], started_at: float | None) -> None:
    if started_at is None:
        return
    async with _live_session_lock:
        if _live_starting.get(key) == started_at:
            _live_starting.pop(key, None)


@dataclass(eq=False)
class LiveSubscriber:
    rid: int
    queue: asyncio.Queue[bytes | object]
    primary: bool


class LiveSession:
    """Short-lived shared producer for duplicate live MPEG-TS requests."""

    def __init__(
        self,
        key: tuple[str, str, float, str, float, str],
        stream: object,
        headers: dict[str, str],
        first_chunk: bytes,
        owner: tuple[str, str],
        producer_rid: int,
        send_bps: int,
        send_pacing: bool,
    ) -> None:
        self.key = key
        self.stream = stream
        self.headers = dict(headers)
        self.owner = owner
        self.producer_rid = producer_rid
        self.send_bps = send_bps
        self.send_pacing = send_pacing
        self.created = asyncio.get_running_loop().time()
        self.last_used = self.created
        self.cache = bytearray()
        self.cache_limit = PASSTHROUGH_LIVE_CACHE_BYTES
        self.subscribers: set[LiveSubscriber] = set()
        self.lock = asyncio.Lock()
        self.closed = False
        self.close_reason = ""
        self.total_bytes = 0
        self.first_chunk = first_chunk
        self.producer_task: asyncio.Task | None = None
        self.expire_task: asyncio.Task | None = None
        self._stream_iter = None
        self._producer_start = self.created
        self._light_match_version = get_light_match().version
        self._append_cache(first_chunk)

    def start(self, stream_iter) -> None:
        self._stream_iter = stream_iter
        self.producer_task = asyncio.create_task(self._run())
        self._schedule_expire()

    @property
    def bytes_emitted(self) -> int:
        return int(getattr(self.stream, "bytes_emitted", self.total_bytes))

    @property
    def frames_produced(self) -> int:
        return int(getattr(self.stream, "frames_produced", 0))

    @property
    def output_fps(self) -> float:
        return float(getattr(self.stream, "output_fps", 0.0))

    @property
    def source_path(self) -> Path | None:
        path = getattr(self.stream, "path", None)
        return path if isinstance(path, Path) else None

    def _append_cache(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if self.cache_limit <= 0:
            return
        remaining = self.cache_limit - len(self.cache)
        if remaining > 0:
            self.cache.extend(chunk[:remaining])

    async def _publish(self, chunk: bytes) -> None:
        stale: list[LiveSubscriber] = []
        async with self.lock:
            current_light_match_version = get_light_match().version
            if LIGHT_MATCH_FLUSH_QUEUES and current_light_match_version != self._light_match_version:
                cache_bytes = len(self.cache)
                self.cache.clear()
                dropped_chunks = 0
                dropped_bytes = 0
                saw_end = False
                for subscriber in list(self.subscribers):
                    d_chunks, d_bytes, d_end = _drain_live_queue_nowait(subscriber.queue)
                    dropped_chunks += d_chunks
                    dropped_bytes += d_bytes
                    saw_end = saw_end or d_end
                log.info(
                    "passthrough_live light match changed v%d->v%d; cleared cache=%d queued_chunks=%d queued_bytes=%d end=%s key=%s",
                    self._light_match_version,
                    current_light_match_version,
                    cache_bytes,
                    dropped_chunks,
                    dropped_bytes,
                    saw_end,
                    _live_session_log_key(self.key),
                )
                self._light_match_version = current_light_match_version
            self._append_cache(chunk)
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            if subscriber.primary:
                await subscriber.queue.put(chunk)
                continue
            try:
                subscriber.queue.put_nowait(chunk)
            except asyncio.QueueFull:
                stale.append(subscriber)
        for subscriber in stale:
            self.subscribers.discard(subscriber)
            try:
                subscriber.queue.put_nowait(_LIVE_END)
            except asyncio.QueueFull:
                pass
            log.info(
                "passthrough_live[%d] live cache duplicate subscriber dropped: key=%s subscribers=%d",
                subscriber.rid,
                _live_session_log_key(self.key),
                len(self.subscribers),
            )

    async def _run(self) -> None:
        try:
            async for chunk in self._stream_iter:
                if not chunk:
                    continue
                await self._publish(chunk)
        except asyncio.CancelledError:
            self.close_reason = self.close_reason or "cancelled"
            raise
        except Exception as e:
            self.close_reason = self.close_reason or f"producer error: {e}"
            log.warning("live session producer failed: key=%s error=%s", _live_session_log_key(self.key), e)
        finally:
            if not self.closed:
                self.closed = True
            for subscriber in list(self.subscribers):
                try:
                    subscriber.queue.put_nowait(_LIVE_END)
                except asyncio.QueueFull:
                    pass
            self.subscribers.clear()
            await asyncio.to_thread(self.stream.close)
            await _release_active_slot(self)
            async with _live_session_lock:
                if _live_sessions.get(self.key) is self:
                    _live_sessions.pop(self.key, None)
            log.info(
                "live session closed: key=%s bytes=%d stream_bytes=%d frames=%d reason=%s",
                _live_session_log_key(self.key),
                self.total_bytes,
                getattr(self.stream, "bytes_emitted", -1),
                getattr(self.stream, "frames_produced", -1),
                self.close_reason or "ended",
            )

    async def close(self, reason: str = "closed") -> None:
        self.close_reason = reason
        self.closed = True
        current_task = asyncio.current_task()
        if self.expire_task is not None and self.expire_task is not current_task:
            self.expire_task.cancel()
        if self.producer_task is not None:
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                pass
        else:
            await asyncio.to_thread(self.stream.close)
            await _release_active_slot(self)
            async with _live_session_lock:
                if _live_sessions.get(self.key) is self:
                    _live_sessions.pop(self.key, None)

    async def subscribe(
        self,
        rid: int,
        *,
        primary: bool | None = None,
        snapshot_only: bool = True,
    ):
        """Subscribe to this LiveSession.

        ``primary=True`` participants back-pressure the producer via a
        blocking ``queue.put`` in ``_publish``; non-primary participants are
        added with ``put_nowait`` and silently dropped if their queue fills,
        so they can never stall the producer.

        ``snapshot_only=True`` (the default for non-primary subscribers)
        means the subscriber receives the current cache prefix and then
        ends. This is the 2026-05-10 behaviour: it stops Skybox/libmpv's
        duplicate-startup GETs from competing for the same Wi-Fi link as
        the primary decoder connection, which on an 8K SBS HEVC alpha link
        is already bandwidth-limited. A 3-way bandwidth split made the
        primary drop below realtime (22fps instead of 30fps) and produced
        a permanent loading spinner on Skybox — see HANDOVER 2026-05-31.
        """
        self.last_used = asyncio.get_running_loop().time()
        if self.expire_task is not None:
            self.expire_task.cancel()
            self.expire_task = None
        queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=PASSTHROUGH_LIVE_SUB_QUEUE_CHUNKS)
        async with self.lock:
            if primary is None:
                primary = not any(subscriber.primary for subscriber in self.subscribers)
            subscriber = LiveSubscriber(rid=rid, queue=queue, primary=bool(primary))
            if not self.closed:
                self.subscribers.add(subscriber)
            snapshot = bytes(self.cache)
        log.info(
            "passthrough_live[%d] live cache subscribe: key=%s snapshot=%d primary=%s snapshot_only=%s closed=%s subscribers=%d",
            rid,
            _live_session_log_key(self.key),
            len(snapshot),
            subscriber.primary,
            snapshot_only,
            self.closed,
            len(self.subscribers),
        )
        sent = 0
        pace_start = asyncio.get_running_loop().time()
        try:
            if snapshot:
                for offset in range(0, len(snapshot), _LIVE_SEND_PACE_CHUNK_BYTES):
                    chunk = snapshot[offset : offset + _LIVE_SEND_PACE_CHUNK_BYTES]
                    sent += len(chunk)
                    if self.send_pacing:
                        await _pace_live_send(pace_start, sent, self.send_bps)
                    yield chunk
            if snapshot_only and not subscriber.primary:
                log.info(
                    "passthrough_live[%d] live cache snapshot-only complete: key=%s sent=%d",
                    rid,
                    _live_session_log_key(self.key),
                    sent,
                )
                return
            while not self.closed:
                item = await queue.get()
                if item is _LIVE_END:
                    break
                chunk = item
                sent += len(chunk)
                if self.send_pacing:
                    await _pace_live_send(pace_start, sent, self.send_bps)
                yield chunk
        finally:
            self.subscribers.discard(subscriber)
            self.last_used = asyncio.get_running_loop().time()
            log.info(
                "passthrough_live[%d] live cache unsubscribe: key=%s primary=%s subscribers=%d",
                rid,
                _live_session_log_key(self.key),
                subscriber.primary,
                len(self.subscribers),
            )
            if not self.subscribers and not self.closed:
                self._schedule_expire()

    def _schedule_expire(self) -> None:
        if PASSTHROUGH_LIVE_CACHE_TTL_SEC <= 0:
            self.expire_task = asyncio.create_task(self.close("no subscribers"))
            return
        if self.expire_task is None or self.expire_task.done():
            self.expire_task = asyncio.create_task(self._expire_later())

    async def _expire_later(self) -> None:
        try:
            await asyncio.sleep(PASSTHROUGH_LIVE_CACHE_TTL_SEC)
            if not self.subscribers:
                await self.close("ttl expired")
        except asyncio.CancelledError:
            pass


def _live_session_log_key(key: tuple[str, str, float, str, float, str]) -> str:
    path, client, start, codec, fps, profile = key
    return f"{Path(path).name}@{start:.2f}/{codec}/{fps:.3f}/{profile}/{client}"


async def _get_live_session(key: tuple[str, str, float, str, float, str]) -> LiveSession | None:
    async with _live_session_lock:
        session = _live_sessions.get(key)
        if session is None or session.closed:
            return None
        return session


async def _put_live_session(key: tuple[str, str, float, str, float, str], session: LiveSession) -> None:
    async with _live_session_lock:
        old = _live_sessions.get(key)
        if old is not None and old is not session:
            asyncio.create_task(old.close("replaced"))
        _live_sessions[key] = session


async def _close_idle_live_sessions_for_request(
    key: tuple[str, str, float, str, float, str],
    rid: int,
) -> None:
    stale: list[LiveSession] = []
    _path, client, _start, _codec, _fps, profile = key
    async with _live_session_lock:
        for old_key, session in list(_live_sessions.items()):
            if old_key == key or session.closed:
                continue
            _old_path, old_client, _old_start, _old_codec, _old_fps, old_profile = old_key
            if old_client != client or old_profile != profile:
                continue
            async with session.lock:
                idle = not session.subscribers
            if idle:
                stale.append(session)
    for session in stale:
        log.info(
            "passthrough_live[%d] close idle live session before new request: old=%s new=%s",
            rid,
            _live_session_log_key(session.key),
            _live_session_log_key(key),
        )
        await session.close("superseded by new request")


def _close_stream_if_possible(stream: object) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception as e:
            log.warning("passthrough preempt close failed: %s", e)


async def _close_preempted_stream(stream: object, who: str) -> None:
    log.info("passthrough preempt close begin: %s stream=%s", who, type(stream).__name__)
    close = getattr(stream, "close", None)
    try:
        if callable(close):
            if isinstance(stream, LiveSession):
                await stream.close("preempted")
            else:
                await asyncio.to_thread(close)
    except Exception as e:
        log.warning("passthrough preempt close failed: %s", e)
    finally:
        # _take_active_slot removes the preempted key from _active_streams
        # before this function runs, but the old key can still own a Matter.
        # Release it after close so a new request does not wait for the old
        # StreamingResponse finally block before acquiring the pool slot.
        await _release_active_slot(stream)
    log.info("passthrough preempt close done: %s stream=%s", who, type(stream).__name__)


async def _close_active_two_dvr_for_client(client_host: str, rid: int, keep_key: tuple | None = None) -> None:
    stale: list[object] = []
    async with _active_lock:
        for active_stream, active_owner in list(_active_streams.items()):
            if _owner_base(active_owner) != ("live", client_host):
                continue
            stream_obj = active_stream.stream if isinstance(active_stream, LiveSession) else active_stream
            if getattr(stream_obj, "output_mode", "") != "two_dvr":
                continue
            if keep_key is not None and isinstance(active_stream, LiveSession) and active_stream.key == keep_key:
                continue
            stale.append(active_stream)
            del _active_streams[active_stream]
            _active_started.pop(active_stream, None)
    for stream in stale:
        log.info("passthrough_live[%d] close previous 2D->3D live stream before new request: stream=%s", rid, type(stream).__name__)
        await _close_preempted_stream(stream, "two_dvr superseded")


def _owner_base(owner: tuple) -> tuple:
    return owner[:2] if len(owner) >= 2 else owner


def _owner_kind(owner: tuple) -> str:
    return str(owner[2]) if len(owner) >= 3 else ""


def _owner_client(owner: tuple) -> str:
    return str(owner[1]) if len(owner) >= 2 else ""


def _can_preempt_same_client_for_seek_test(active_owner: tuple, new_owner: tuple) -> bool:
    """During seek-route testing, one client switching files/players should not
    be blocked by its own stale generated streams.
    """
    client = _owner_client(new_owner)
    return bool(client and _owner_client(active_owner) == client)


def _client_log_id(client: object) -> str:
    text = str(client or "")
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
    return f"client-{digest}"


def _owner_log_value(owner: tuple) -> tuple:
    if not owner:
        return owner
    try:
        first = str(owner[0])
        client = _client_log_id(owner[1] if len(owner) >= 2 else "")
        if first == "live":
            return ("live", client, *owner[2:])
        return (Path(first).name or "<path>", client, *owner[2:])
    except Exception:
        return ("<owner>",)


def _can_preempt_owner(active_owner: tuple, new_owner: tuple) -> bool:
    new_base = _owner_base(new_owner)
    is_live_owner = len(new_base) > 0 and new_base[0] == "live"
    if active_owner == new_owner:
        kind = _owner_kind(new_owner)
        if is_live_owner:
            # libmpv same-owner preempt is safe because same-live_key duplicates
            # are caught earlier by the libmpv startup debounce in
            # passthrough_live_get and join via subscribe(primary=False).
            # Reaching this point means different-live_key (typically a
            # different t= chapter probe from the same Skybox client) and the
            # old slot should yield to the newer probe.
            return kind in ("nplayer", "4xvr", "avpro", "libmpv")
        return kind in ("", "libmpv", "vlc", "nplayer", "seek")
    if _owner_base(active_owner) != new_base:
        return False
    active_kind = _owner_kind(active_owner)
    new_kind = _owner_kind(new_owner)
    if is_live_owner and new_kind in ("nplayer", "4xvr", "avpro", "libmpv"):
        return True
    if active_kind == "lavf" and new_kind in ("vlc", "libmpv"):
        return True
    if active_kind == "lavf" and new_kind in ("default", ""):
        return True
    if is_live_owner:
        return False
    if new_kind == "libmpv":
        return True
    return False


def _nvidia_smi_path() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for root in (os.environ.get("SystemRoot"), os.environ.get("WINDIR")):
        if not root:
            continue
        system32 = Path(root) / "System32" / "nvidia-smi.exe"
        if system32.exists():
            return str(system32)
    return None


def _query_vram_mib() -> tuple[float, float] | None:
    exe = _nvidia_smi_path()
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            **hidden_subprocess_kwargs(),
        )
    except Exception:
        return None
    best: tuple[float, float] | None = None
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used = float(parts[0])
            total = float(parts[1])
        except ValueError:
            continue
        if best is None or used > best[0]:
            best = (used, total)
    return best


def _runtime_stream_info(stream: object, started_at: float | None, now: float) -> dict:
    inner = getattr(stream, "stream", stream)
    frames = int(getattr(stream, "frames_produced", getattr(inner, "frames_produced", 0)) or 0)
    output_fps = float(getattr(stream, "output_fps", getattr(inner, "output_fps", 0.0)) or 0.0)
    bytes_emitted = int(getattr(stream, "bytes_emitted", getattr(inner, "bytes_emitted", 0)) or 0)
    source = getattr(stream, "source_path", None)
    if source is None:
        source = getattr(inner, "path", getattr(inner, "src", None))
    elapsed = max(0.0, now - started_at) if started_at is not None else 0.0
    produced_fps = frames / elapsed if elapsed > 0.25 and frames > 0 else 0.0
    return {
        "active": True,
        "source": str(source) if source else "",
        "source_name": Path(source).name if source else "",
        "output_fps": output_fps,
        "produced_fps": produced_fps,
        "frames": frames,
        "bytes": bytes_emitted,
        "elapsed_sec": elapsed,
    }


@router.get("/runtime_status")
async def runtime_status():
    now = asyncio.get_running_loop().time()
    async with _active_lock:
        active_items = [(stream, _active_started.get(stream)) for stream in _active_streams.keys()]
    stream_info = None
    for stream, started_at in active_items:
        stream_info = _runtime_stream_info(stream, started_at, now)
        if stream_info["frames"] > 0 or stream_info["source"]:
            break
    vram = await asyncio.to_thread(_query_vram_mib)
    status = {
        "ok": True,
        "active": stream_info is not None,
        "source": "",
        "source_name": "",
        "output_fps": 0.0,
        "produced_fps": 0.0,
        "frames": 0,
        "bytes": 0,
        "elapsed_sec": 0.0,
        "vram_used_mib": None,
        "vram_total_mib": None,
    }
    if stream_info is not None:
        status.update(stream_info)
    if vram is not None:
        status["vram_used_mib"], status["vram_total_mib"] = vram
    return status


def _is_real_active_stream(active_stream: object) -> bool:
    """True when active_stream is a real producer (PyNvPassthroughStream or
    LiveSession), False when it is the raw slot_token placeholder returned by
    object() during slot acquisition. Slot_tokens have no close(); real
    producers always expose a callable close().
    """
    return callable(getattr(active_stream, "close", None))


async def _take_active_slot(
    new_stream: object,
    who: str,
    owner: tuple,
    *,
    allow_same_owner_preempt: bool = True,
    allow_same_client_preempt: bool = False,
) -> object | None | bool:
    deadline = asyncio.get_running_loop().time() + PASSTHROUGH_BUSY_WAIT_SEC
    warned = False
    while True:
        async with _active_lock:
            if len(_active_streams) < PASSTHROUGH_MAX_CONCURRENT:
                _active_streams[new_stream] = owner
                _active_started[new_stream] = asyncio.get_running_loop().time()
                return None
            if allow_same_owner_preempt or allow_same_client_preempt:
                for active_stream, active_owner in list(_active_streams.items()):
                    can_preempt = allow_same_owner_preempt and _can_preempt_owner(active_owner, owner)
                    if allow_same_client_preempt:
                        can_preempt = can_preempt or _can_preempt_same_client_for_seek_test(active_owner, owner)
                    if can_preempt and active_owner == owner and _is_real_active_stream(active_stream):
                        # Same-owner preempt of a real in-flight producer would
                        # kill its build / iter_bytes mid-stream and turn into
                        # 409/503 for that request — the failure mode seen when
                        # a Skybox chapter-probe burst cascades through N libmpv
                        # requests. Only raw slot_tokens may be preempted by
                        # same-owner; established producers wait/503 instead so
                        # the working stream stays intact.
                        can_preempt = False
                    if can_preempt:
                        del _active_streams[active_stream]
                        _active_started.pop(active_stream, None)
                        _active_streams[new_stream] = owner
                        _active_started[new_stream] = asyncio.get_running_loop().time()
                        log.info("passthrough preempt previous range: %s owner=%s", who, _owner_log_value(owner))
                        return active_stream
            active = len(_active_streams)
        if PASSTHROUGH_BUSY_WAIT_SEC <= 0 or asyncio.get_running_loop().time() >= deadline:
            log.warning(
                "passthrough busy: reject %s active=%d max=%d waited=%.1fs",
                who, active, PASSTHROUGH_MAX_CONCURRENT, PASSTHROUGH_BUSY_WAIT_SEC,
            )
            return False
        if not warned:
            log.info(
                "passthrough busy: wait %s active=%d max=%d timeout=%.1fs",
                who, active, PASSTHROUGH_MAX_CONCURRENT, PASSTHROUGH_BUSY_WAIT_SEC,
            )
            warned = True
        await asyncio.sleep(0.1)


async def _release_active_slot(stream: object) -> None:
    async with _active_lock:
        _release_active_slot_nowait(stream)


async def _replace_active_slot(
    old_stream: object,
    new_stream: object,
    *,
    close_on_failure: object | None = None,
) -> bool:
    """Migrate slot/started/matter bookkeeping from old key to new key.

    On failure (old key already preempted) the popped Matter is returned to
    the pool inline. Without this, callers that hit the failure branch and
    return 409 would leak the Matter, because the failure path leaves the
    handler before any ``_release_active_slot`` call can recover it.

    Critically, failure cleanup closes any running stream before releasing the
    Matter. For slot-token callers the running stream is ``close_on_failure``;
    for session-key callers it is usually ``old_stream``.
    """
    leaked_matter = None
    async with _active_lock:
        owner = _active_streams.pop(old_stream, None)
        started_at = _active_started.pop(old_stream, None)
        matter = _active_matter.pop(old_stream, None)
        if owner is None:
            leaked_matter = matter
        else:
            _active_streams[new_stream] = owner
            _active_started[new_stream] = started_at or asyncio.get_running_loop().time()
            if matter is not None:
                _active_matter[new_stream] = matter
    if owner is None:
        close_targets: list[object] = []
        if close_on_failure is not None:
            close_targets.append(close_on_failure)
        if all(target is not old_stream for target in close_targets):
            close_targets.append(old_stream)
        for target in close_targets:
            close = getattr(target, "close", None)
            if not callable(close):
                continue
            try:
                await asyncio.to_thread(close)
            except Exception as e:
                log.warning(
                    "replace_active_slot close failed: %s err=%s",
                    type(target).__name__, e,
                )
    if leaked_matter is not None:
        # Release only after failure streams have been stopped so another
        # request cannot acquire a Matter still used by an old worker.
        release_matter(leaked_matter)
    return owner is not None


def _safe_video_path_from_key(name: str) -> Path:
    p = MEDIA_LIBRARY.key_to_path(name)
    if p is None:
        raise HTTPException(403, "forbidden")
    p = p.resolve()
    # Reject path traversal outside configured media roots.
    if not MEDIA_LIBRARY.contains(p):
        raise HTTPException(403, "forbidden")
    if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(404, "not found")
    return p


def _safe_video_path(name: str) -> Path:
    return _safe_video_path_from_key(unquote(name))


# Suffixes that the live route silently strips from the URL path before
# resolving the underlying source file. Skybox/2.0.x (and reportedly older
# Skybox builds) pick their HTTP playback pipeline from the URL file
# extension: a request to ``/passthrough_live/<...>.mp4`` is routed to the
# MP4 byte-range pipeline (pipeline=basic in Skybox debug), the MP4 parser
# then finds no ftyp/moov atoms in our MPEG-TS bytes and the entire
# response is dropped — Skybox's debug overlay shows zero network traffic
# even though the server clearly delivered megabytes. Appending ``.ts`` to
# the DLNA URL pushes Skybox onto its TS pipeline. Compliant clients (4XVR,
# HereSphere, nPlayer, VLC) ignore the URL extension and key off
# Content-Type, so the extra suffix is harmless there.
_LIVE_ROUTE_HINT_SUFFIXES = (".ts", ".m2ts", ".mpegts")


def _strip_live_route_hint_suffix(name: str) -> str:
    """Drop an optional MPEG-TS pipeline hint suffix from the URL path. See
    ``_LIVE_ROUTE_HINT_SUFFIXES``. The suffix is treated as a client hint
    only; the file lookup uses the original source filename.
    """
    decoded = unquote(name)
    lower = decoded.lower()
    for suffix in _LIVE_ROUTE_HINT_SUFFIXES:
        if lower.endswith(suffix):
            return decoded[: -len(suffix)]
    return decoded


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _media_type_for_path(path: Path) -> str:
    if _is_image_path(path):
        return IMAGE_MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
    return "video/mp4"


def _safe_media_path_from_key(name: str) -> Path:
    p = MEDIA_LIBRARY.key_to_path(name)
    if p is None:
        raise HTTPException(403, "forbidden")
    p = p.resolve()
    if not MEDIA_LIBRARY.contains(p):
        raise HTTPException(403, "forbidden")
    suffix = p.suffix.lower()
    if p.is_file() and suffix in VIDEO_EXTS:
        return p
    if p.is_file() and DLNA_IMAGE_ENABLED and suffix in IMAGE_EXTS:
        return p
    raise HTTPException(404, "not found")


def _safe_media_path(name: str) -> Path:
    return _safe_media_path_from_key(unquote(name))


def _safe_seek_video_path(name: str) -> tuple[Path, str | None]:
    key, route_container = _split_seek_route_name(name)
    return _safe_video_path_from_key(key), route_container


def _safe_subtitle_path(name: str) -> Path:
    name = unquote(name)
    p = MEDIA_LIBRARY.key_to_path(name)
    if p is None:
        raise HTTPException(403, "forbidden")
    p = p.resolve()
    if not MEDIA_LIBRARY.contains(p):
        raise HTTPException(403, "forbidden")
    if not p.is_file() or not is_subtitle_path(p):
        raise HTTPException(404, "not found")
    return p


def _subtitle_headers_for_video(path: Path) -> dict[str, str]:
    tracks = find_external_subtitles(path)
    if not tracks:
        return {}
    try:
        rel = MEDIA_LIBRARY.path_to_key(tracks[0].path)
    except Exception:
        return {}
    url = f"http://{LAN_IP}:{HTTP_PORT}/subs/{quote(rel)}"
    return {
        "CaptionInfo.sec": url,
        "getCaptionInfo.sec": "1",
    }


def _reject_unsafe_mkv_live_path(path: Path) -> None:
    if path.suffix.lower() != ".mkv":
        return
    policy = PASSTHROUGH_MKV_LIVE_POLICY
    if policy not in {"block", "head_cues", "allow"}:
        policy = "block"
    if policy == "allow":
        return
    if policy == "block":
        log.warning("passthrough_live reject MKV by policy: path=%s", path.name)
        raise HTTPException(409, "MKV live passthrough is disabled")
    info = probe_mkv_cues(path)
    if info.needs_fix:
        log.warning(
            "passthrough_live reject MKV without head Cues: path=%s status=%s position=%d reason=%s",
            path.name,
            info.status,
            info.position,
            info.reason,
        )
        raise HTTPException(409, "MKV needs remux before live passthrough")


# ---- Raw MP4 Range serving ----
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_NPT_RE = re.compile(r"npt\s*=\s*([0-9:.]+)\s*-", re.IGNORECASE)


@dataclass(frozen=True)
class ByteRange:
    """Parsed HTTP byte range with inclusive start/end offsets."""

    start: int
    end: int
    total: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)


def _parse_byte_range(value: str | None, size: int) -> ByteRange | None:
    if not value:
        return None
    m = _RANGE_RE.match(value)
    if not m:
        raise HTTPException(416, "invalid range")
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else size - 1
    end = min(end, size - 1)
    byte_range = ByteRange(start=start, end=end, total=size)
    if start >= size or byte_range.length <= 0:
        raise HTTPException(416, "range not satisfiable")
    return byte_range


def _file_range_response(path: Path, media_type: str, range_header: str | None, extra_headers: dict[str, str] | None = None) -> Response:
    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        **(extra_headers or {}),
    }
    byte_range = _parse_byte_range(range_header, size)
    if byte_range is None:
        return FileResponse(path, media_type=media_type, headers=headers)

    length = byte_range.length

    def gen():
        with open(path, "rb") as f:
            f.seek(byte_range.start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers.update(
        {
            "Content-Range": f"bytes {byte_range.start}-{byte_range.end}/{size}",
            "Content-Length": str(length),
            "Content-Type": media_type,
        }
    )
    return StreamingResponse(gen(), status_code=206, headers=headers, media_type=media_type)


@router.get("/subs/{name:path}")
async def subtitle_get(request: Request, name: str, range: str | None = Header(default=None)):
    path = _safe_subtitle_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path))
    headers = {
        "Content-Disposition": "inline",
        "Access-Control-Allow-Origin": "*",
    }
    return _file_range_response(path, subtitle_mime(path), range, headers)


@router.head("/subs/{name:path}")
async def subtitle_head(request: Request, name: str, range: str | None = Header(default=None)):
    path = _safe_subtitle_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path))
    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": "inline",
        "Access-Control-Allow-Origin": "*",
        "Content-Type": subtitle_mime(path),
    }
    byte_range = _parse_byte_range(range, size)
    if byte_range is not None:
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{size}"
        headers["Content-Length"] = str(byte_range.length)
        return Response(status_code=206, headers=headers)
    headers["Content-Length"] = str(size)
    return Response(status_code=200, headers=headers)


@router.head("/media/{name:path}")
async def media_head(request: Request, name: str, range: str | None = Header(default=None)):
    path = _safe_media_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path))
    size = path.stat().st_size
    media_type = _media_type_for_path(path)
    subtitle_headers = _subtitle_headers_for_video(path) if path.suffix.lower() in VIDEO_EXTS else {}
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": media_type,
        **subtitle_headers,
    }
    byte_range = _parse_byte_range(range, size)
    if byte_range is not None:
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{size}"
        headers["Content-Length"] = str(byte_range.length)
        return Response(status_code=206, headers=headers)
    headers["Content-Length"] = str(size)
    return Response(status_code=200, headers=headers)


def _si_dlna_content_features() -> str:
    return (
        "DLNA.ORG_PN=HEVC_MP4_MAIN;"
        f"DLNA.ORG_OP={DLNA_OP_BYTE_SEEK};"
        f"DLNA.ORG_CI=0;DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
    )


def _si_base_headers(content_length: int) -> dict[str, str]:
    features = _si_dlna_content_features()
    return {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
        "Content-Length": str(max(0, int(content_length))),
        "contentFeatures.dlna.org": features,
        "transferMode.dlna.org": "Streaming",
    }


def _si_startup_probe_body(path: Path, start: int, total: int) -> tuple[bytes, int, int]:
    safe_start = min(max(0, int(start)), max(0, int(total) - 1))
    length = min(_SI_STARTUP_PROBE_BYTES, max(0, int(total) - safe_start))
    if length <= 0:
        return b"", safe_start, safe_start
    body = b""
    try:
        with path.open("rb") as fh:
            fh.seek(safe_start)
            body = fh.read(length)
    except OSError:
        body = b""
    if len(body) < length:
        body += b"\x00" * (length - len(body))
    safe_end = safe_start + len(body) - 1 if body else safe_start
    return body, safe_start, safe_end


def _safe_si_video_path(name: str) -> Path:
    path = _safe_video_path(name)
    if path.suffix.lower() != ".mp4":
        raise HTTPException(404, "SI streaming currently supports MP4 sources only")
    return path


def _si_virtual_disabled() -> bool:
    return not bool(SI_PROGRESSIVE_ENABLED)


def _si_range_headers(layout, start: int, end: int, status_code: int) -> dict[str, str]:
    content_length = max(0, int(end) - int(start) + 1)
    headers = _si_base_headers(content_length)
    headers["ETag"] = f'"{layout.etag}"'
    headers["X-SI-Enabled"] = "1"
    headers["X-SI-Transport"] = "progressive-virtual"
    headers["X-SI-Moov-Bytes"] = str(layout.moov_size)
    headers["X-SI-Samples"] = f"{layout.video_samples}+{layout.audio_samples}"
    headers["X-SI-Audio-Edit"] = str(getattr(layout, "audio_edit_mode", "preserve"))
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{layout.content_length}"
    return headers


async def _si_virtual_layout(path: Path):
    if _si_virtual_disabled():
        raise HTTPException(404, "SI progressive virtual MP4 is disabled")
    service = get_si_stream_service()
    config = service.current_config()
    si_wav = service.has_si_source(path)
    if not config.enabled or si_wav is None:
        raise HTTPException(404, "SI stream not available")
    try:
        return await asyncio.to_thread(build_progressive_si_virtual_mp4, path, si_wav, config)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        log.warning("SI progressive audio sidecar build failed: %s", exc)
        raise HTTPException(500, "SI audio sidecar build failed") from exc
    except Exception as exc:
        log.warning("SI progressive virtual MP4 build failed for %s: %s", path, exc)
        raise HTTPException(500, "SI virtual MP4 build failed") from exc


def _si_resolve_range(range_header: str | None, total: int) -> tuple[int, int, bool, int]:
    start, end, range_requested = parse_range_header(range_header)
    if total <= 0:
        return 0, -1, range_requested, 416 if range_requested else 200
    safe_start = max(0, int(start))
    if range_requested and safe_start >= total:
        return safe_start, total - 1, True, 416
    safe_start = min(safe_start, total - 1)
    safe_end = min(int(end), total - 1) if end is not None else total - 1
    if safe_end < safe_start:
        safe_end = total - 1
    return safe_start, safe_end, range_requested, 206 if range_requested else 200


@router.head("/media_si/{name:path}")
async def media_si_head(request: Request, name: str, range: str | None = Header(default=None)):
    path = _safe_si_video_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path), passthrough_route="si_mix")
    layout = await _si_virtual_layout(path)
    start, end, range_requested, status_code = _si_resolve_range(range, layout.content_length)
    annotate_request(request, total_estimated_size=layout.content_length)
    if status_code == 416:
        return Response(
            status_code=416,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes */{layout.content_length}",
                "ETag": f'"{layout.etag}"',
            },
        )
    headers = _si_range_headers(layout, start, end, status_code)
    return Response(status_code=status_code, headers=headers, media_type="video/mp4")


@router.get("/media_si/{name:path}")
async def media_si_get(
    request: Request,
    name: str,
    range: str | None = Header(default=None),
    user_agent: str | None = Header(default=None, alias="User-Agent"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
):
    rid = next(_request_ids)
    path = _safe_si_video_path(name)
    client_id = request.client.host if request.client else ""
    annotate_request(request, media_name=path.name, media_path=str(path), passthrough_route="si_mix")
    layout = await _si_virtual_layout(path)
    start, end, range_requested, status_code = _si_resolve_range(range, layout.content_length)
    annotate_request(request, total_estimated_size=layout.content_length)
    if status_code == 416:
        return Response(
            status_code=416,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes */{layout.content_length}",
                "ETag": f'"{layout.etag}"',
            },
        )
    headers = _si_range_headers(layout, start, end, status_code)
    log.info(
        "media_si[%d] progressive virtual start: path=%s status=%d range=%r resolved=%d-%d/%d client=%s client_id=%s ua=%r time_seek=%r transfer=%r getfeatures=%r moov=%d samples=%d+%d",
        rid,
        path.name,
        status_code,
        range,
        start,
        end,
        layout.content_length,
        request.client,
        client_id,
        (user_agent or "")[:160],
        time_seek_range,
        transfer_mode,
        get_content_features,
        layout.moov_size,
        layout.video_samples,
        layout.audio_samples,
    )

    async def gen():
        sent = 0
        iterator = iter_virtual_range(layout.regions, start, end, chunk_size=DEFAULT_CHUNK_SIZE)
        content_length = max(0, end - start + 1)
        try:
            while sent < content_length:
                chunk = await asyncio.to_thread(next, iterator, _SI_EOF)
                if chunk is _SI_EOF:
                    break
                if not chunk:
                    break
                if sent + len(chunk) > content_length:
                    chunk = chunk[: content_length - sent]
                sent += len(chunk)
                yield chunk
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
            log.info(
                "media_si[%d] progressive virtual end: path=%s sent=%d content_length=%d",
                rid,
                path.name,
                sent,
                content_length,
            )

    return StreamingResponse(gen(), status_code=status_code, headers=headers, media_type="video/mp4")


def _si_live_content_features() -> str:
    return (
        "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
        "DLNA.ORG_OP=10;DLNA.ORG_CI=0;"
        "DLNA.ORG_FLAGS=41700000000000000000000000000000"
    )


@router.get("/si_live/{name:path}")
async def si_live_get(
    request: Request,
    name: str,
    t: float = 0.0,
    user_agent: str | None = Header(default=None, alias="User-Agent"),
):
    """Realtime MPEG-TS SI mix stream.

    Unlike the cached progressive `/media_si` path, this mixes the SI audio on the
    fly and streams MPEG-TS (video/MP2T) like `/passthrough_live`, so playback
    starts in ~1-2s with no sidecar cache. Seeking is done by re-requesting a new
    `?t=<seconds>` offset (the DLNA `[SI]` time-index leaves do exactly that).
    """
    for suffix in _LIVE_ROUTE_HINT_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    path = _safe_si_video_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path), passthrough_route="si_live")
    service = get_si_stream_service()
    config = service.current_config()
    si_wav = service.has_si_source(path)
    if not config.enabled or si_wav is None:
        raise HTTPException(404, "SI stream not available")
    start_time = max(0.0, float(t or 0.0))
    log.info("si_live start: path=%s t=%.3f si=%s ua=%r", path, start_time, si_wav, user_agent)
    headers = {
        "Accept-Ranges": "none",
        "X-SI-Enabled": "1",
        "X-SI-Transport": "mpegts-live",
        "contentFeatures.dlna.org": _si_live_content_features(),
        "transferMode.dlna.org": "Streaming",
    }
    return StreamingResponse(
        iter_si_mpegts(path, si_wav, config, start_time, chunk_size=DEFAULT_CHUNK_SIZE),
        status_code=200,
        headers=headers,
        media_type="video/MP2T",
    )


@router.get("/media/{name:path}")
async def media_get(
    request: Request,
    name: str,
    range: str | None = Header(default=None),
    user_agent: str | None = Header(default=None, alias="User-Agent"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
):
    rid = next(_request_ids)
    path = _safe_media_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path))
    size = path.stat().st_size
    media_type = _media_type_for_path(path)
    subtitle_headers = _subtitle_headers_for_video(path) if path.suffix.lower() in VIDEO_EXTS else {}
    if DEBUG_LOGS:
        log.info(
            "media[%d] request: path=%s size=%d range=%r time_seek=%r transfer=%r getfeatures=%r ua=%r client=%s",
            rid,
            path.name,
            size,
            range,
            time_seek_range,
            transfer_mode,
            get_content_features,
            (user_agent or "")[:240],
            request.client,
        )

    if range:
        m = _RANGE_RE.match(range)
        if not m:
            if DEBUG_LOGS:
                log.info("media[%d] return 416 invalid range: %r path=%s", rid, range, path.name)
            raise HTTPException(416, "invalid range")
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        length = end - start + 1
        if start >= size or length <= 0:
            if DEBUG_LOGS:
                log.info(
                    "media[%d] return 416 unsatisfiable range=%r parsed=%d-%d/%d path=%s",
                    rid,
                    range,
                    start,
                    end,
                    size,
                    path.name,
                )
            raise HTTPException(416, "range not satisfiable")

        def gen():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": media_type,
            **subtitle_headers,
        }
        if DEBUG_LOGS:
            log.info(
                "media[%d] response: status=206 path=%s range=%d-%d/%d length=%d open=%s suffix=%s",
                rid,
                path.name,
                start,
                end,
                size,
                length,
                not bool(m.group(2)),
                not bool(m.group(1)),
            )
        return StreamingResponse(gen(), status_code=206, headers=headers, media_type=media_type)

    if DEBUG_LOGS:
        log.info("media[%d] response: status=200 path=%s size=%d full-file", rid, path.name, size)
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes", **subtitle_headers})


def _parse_npt_seconds(value: str | None) -> float | None:
    """Parse DLNA TimeSeekRange.dlna.org values like npt=120.5- or npt=00:02:00-."""
    if not value:
        return None
    m = _NPT_RE.search(value)
    if not m:
        return None
    token = m.group(1)
    try:
        if ":" not in token:
            return max(0.0, float(token))
        parts = [float(p) for p in token.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    else:
        return None
    return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)


def _format_npt(seconds: float) -> str:
    return f"{max(0.0, seconds):.3f}"


def _estimated_passthrough_size(path: Path, duration: float, codec: str = "") -> int:
    if duration <= 0:
        return 0
    size, _, _ = estimate_for_media(path, duration, codec)
    return size


def _estimated_passthrough_bps(path: Path, codec: str = "") -> int:
    _, bps, _ = estimate_for_media(path, 1.0, codec)
    paced_bps = int(float(bps) * PASSTHROUGH_SEND_PACING_MULTIPLIER)
    return max(1, PASSTHROUGH_SEND_MIN_BPS, paced_bps)


def _estimated_live_pynv_size(duration: float) -> int:
    if duration <= 0:
        return 0
    return int(parse_bitrate(PASSTHROUGH_HEVC_BITRATE) * float(duration) / 8.0)


def _estimated_live_pynv_send_bps() -> int:
    paced_bps = int(float(parse_bitrate(PASSTHROUGH_HEVC_BITRATE)) * PASSTHROUGH_SEND_PACING_MULTIPLIER)
    return max(1, PASSTHROUGH_SEND_MIN_BPS, paced_bps)


async def _pace_live_send(start_wall: float, sent_bytes: int, bps: int) -> None:
    if not PASSTHROUGH_SEND_REALTIME_PACING or sent_bytes <= 0 or bps <= 0:
        return
    target_elapsed = sent_bytes * 8.0 / float(bps)
    elapsed = asyncio.get_running_loop().time() - start_wall
    delay = target_elapsed - elapsed - _LIVE_SEND_PACE_BURST_SEC
    if delay > 0:
        await asyncio.sleep(min(0.05, delay))


def _codec_from_ffmpeg_vcodec() -> str:
    from config import PASSTHROUGH_VCODEC

    text = (PASSTHROUGH_VCODEC or "").lower()
    if "hevc" in text or "h265" in text:
        return "hevc"
    if "h264" in text or "x264" in text or "avc" in text:
        return "h264"
    return ""


def _passthrough_estimate_codec(path: Path) -> str:
    if not USE_PYNV:
        return _codec_from_ffmpeg_vcodec()
    try:
        meta = probe_video_metadata(path)
        decision = select_backend(meta.timing, meta.codec, meta.color)
        if decision.verdict == "pynv_hevc":
            return PYNV_OUTPUT_CODEC
    except Exception:
        return _codec_from_ffmpeg_vcodec()
    return _codec_from_ffmpeg_vcodec()


def _passthrough_backend_verdict(path: Path) -> str:
    if not USE_PYNV:
        return ""
    try:
        meta = probe_video_metadata(path)
        decision = select_backend(meta.timing, meta.codec, meta.color)
        return decision.verdict
    except Exception:
        return ""


def _probe_cache_key(path: Path, codec: str, duration: float) -> str:
    total = _estimated_passthrough_size(path, duration, codec)
    return f"{path.resolve()}|{codec}|{total}"


def _seek_declared_size_key(path: Path, codec: str, duration: float, client_host: str, container: str | None = None) -> tuple:
    try:
        st = path.stat()
        stat_part = (st.st_size, st.st_mtime_ns)
    except OSError:
        stat_part = (0, 0)
    return (
        str(path.resolve()),
        stat_part[0],
        stat_part[1],
        codec,
        container or _seek_container(),
        round(float(duration or 0.0), 3),
        int(PASSTHROUGH_SEEK_HEADER_BYTES),
        client_host or "",
    )


def _estimated_seek_passthrough_size(path: Path, duration: float, codec: str, client_host: str = "", container: str | None = None) -> int:
    key = _seek_declared_size_key(path, codec, duration, client_host, container)
    cached = _seek_declared_size_cache.get(key)
    if cached is not None:
        return cached
    base = _estimated_passthrough_size(path, duration, codec)
    total = max(0, int(PASSTHROUGH_SEEK_HEADER_BYTES)) + base
    if len(_seek_declared_size_cache) >= _SEEK_DECLARED_SIZE_CACHE_LIMIT:
        _seek_declared_size_cache.pop(next(iter(_seek_declared_size_cache)), None)
    _seek_declared_size_cache[key] = total
    return total


def _seek_probe_cache_key(path: Path, codec: str, duration: float, total: int, container: str | None = None) -> str:
    return f"seek|{container or _seek_container()}|{path.resolve()}|{codec}|{round(float(duration or 0.0), 3)}|{total}"


def _seek_route_allowed(user_agent: str) -> tuple[bool, str, str]:
    profile = _live_response_profile(user_agent)
    if not PASSTHROUGH_SEEK_ENABLED:
        return False, "disabled", profile
    policy = PASSTHROUGH_SEEK_ROUTE_POLICY
    if policy == "off":
        return False, "route_policy_off", profile
    if policy == "all":
        return True, "route_policy_all", profile
    if profile in set(PASSTHROUGH_SEEK_PROFILES):
        return True, "profile_allowed", profile
    return False, f"profile_{profile}_blocked", profile


def _seek_blocked_response(reason: str) -> Response:
    if reason == "disabled":
        return Response("seekable passthrough disabled", status_code=404)
    return Response(
        "seekable passthrough disabled for this client; use /passthrough_live",
        status_code=403,
    )


def _seek_output_fps(info) -> float:
    source_fps = float(getattr(info, "fps", 0.0) or 0.0)
    cap = float(PASSTHROUGH_MAX_FPS or 0.0)
    if cap > 0 and source_fps > 0:
        return min(source_fps, cap)
    return source_fps if source_fps > 0 else cap


def _seek_prefix_cache_limit() -> int:
    return min(_PROBE_CACHE_LIMIT, max(0, int(PASSTHROUGH_SEEK_HEADER_BYTES or 0)))


def _cache_prefix_limit() -> int:
    return _PROBE_CACHE_LIMIT


def _apply_seek_diag_headers(
    headers: dict[str, str],
    *,
    start_sec: float,
    output_mode: str,
    container: str | None = None,
    mapped=None,
) -> None:
    headers["X-Passthrough-Seek-Time"] = f"{start_sec:.3f}"
    headers["X-Passthrough-Mode"] = f"seek-{container or _seek_container()}-{output_mode}"
    if mapped is not None:
        headers["X-Passthrough-Seek-Ratio"] = f"{mapped.ratio:.6f}"
        headers["X-Passthrough-Seek-Raw-Time"] = f"{mapped.time_sec:.3f}"
        headers["X-Passthrough-Seek-Gop"] = f"{mapped.gop_seconds:.3f}"


def _seek_headers(
    *,
    path: Path,
    duration: float,
    codec: str,
    total: int,
    start_sec: float,
    range_header: str | None,
    include_length: bool,
    container: str | None = None,
    backend_verdict: str | None = None,
    info=None,
) -> dict[str, str]:
    resolved_container = container or _seek_container()
    headers = {
        "Content-Type": _seek_media_type(resolved_container),
        "Cache-Control": "no-cache",
        "transferMode.dlna.org": "Interactive",
        "contentFeatures.dlna.org": (
            f"DLNA.ORG_PN={_seek_dlna_pn(resolved_container)};"
            f"DLNA.ORG_OP={DLNA_OP_BYTE_AND_TIME_SEEK};"
            "DLNA.ORG_CI=0;"
            f"DLNA.ORG_FLAGS={DLNA_FLAGS_BYTE_AND_TIME_SEEK}"
        ),
        "Accept-Ranges": "bytes",
        "X-Passthrough-Seekable": "1",
        "X-Passthrough-Estimated-Size": str(total),
    }
    if info is None:
        info = probe_cached(path)
    frame_rate = passthrough_frame_rate(_seek_output_fps(info))
    if frame_rate:
        headers["X-Passthrough-FrameRate"] = frame_rate
    _, estimated_bps, estimate = estimate_for_media(path, duration, codec)
    headers["X-Passthrough-Estimated-Bps"] = str(estimated_bps)
    headers["X-Passthrough-Estimate-Source"] = estimate.source
    if backend_verdict:
        headers["X-Passthrough-Backend-Verdict"] = backend_verdict
    byte_range = _parse_byte_range(range_header, total)
    response_range = byte_range or ByteRange(start=0, end=max(0, total - 1), total=total)
    if include_length:
        headers["Content-Length"] = str(response_range.length)
    if byte_range is not None:
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{total}"
    if duration > 0:
        start_npt = _format_npt(start_sec)
        end_npt = _format_npt(duration)
        headers["TimeSeekRange.dlna.org"] = f"npt={start_npt}-{end_npt}/{end_npt}"
        headers["availableSeekRange.dlna.org"] = f"1 npt=0.000-{end_npt}"
        headers["X-AvailableSeekRange.dlna.org"] = f"1 npt=0.000-{end_npt}"
    return headers


def _range_start(value: str | None) -> int | None:
    if not value:
        return None
    m = _RANGE_RE.match(value)
    if not m:
        return None
    if not m.group(1):
        return 0
    try:
        return max(0, int(m.group(1)))
    except ValueError:
        return None


def _range_end(value: str | None) -> int | None:
    if not value:
        return None
    m = _RANGE_RE.match(value)
    if not m or not m.group(2):
        return None
    try:
        return max(0, int(m.group(2)))
    except ValueError:
        return None


def _parse_byte_range(value: str | None, total: int) -> ByteRange | None:
    if not value or total <= 0:
        return None
    m = _RANGE_RE.fullmatch(value.strip())
    if not m:
        return None
    start_s, end_s = m.group(1), m.group(2)
    if not start_s and not end_s:
        return None
    if start_s:
        start = int(start_s)
        if start >= total:
            return None
        end = int(end_s) if end_s else total - 1
        end = min(end, total - 1)
        if end < start:
            return None
        return ByteRange(start=start, end=end, total=total)
    suffix_len = int(end_s)
    if suffix_len <= 0:
        return None
    suffix_len = min(suffix_len, total)
    return ByteRange(start=total - suffix_len, end=total - 1, total=total)


def _is_small_probe_range(byte_range: ByteRange | None) -> bool:
    return byte_range is not None and byte_range.start == 0 and byte_range.end < _SMALL_PROBE_LIMIT


def _is_zero_open_range(value: str | None, byte_range: ByteRange | None) -> bool:
    return bool(value) and byte_range is not None and byte_range.start == 0 and _range_end(value) is None


def _is_open_range(value: str | None) -> bool:
    return bool(value) and _range_end(value) is None


def _is_tail_probe_range(byte_range: ByteRange | None) -> bool:
    if byte_range is None or byte_range.total <= 0:
        return False
    return (
        byte_range.start > 0
        and byte_range.start >= int(byte_range.total * _TAIL_PROBE_RATIO)
        and byte_range.length <= _TAIL_PROBE_MAX_BYTES
    )


def _is_header_only_range(byte_range: ByteRange | None) -> bool:
    if byte_range is None:
        return False
    return (
        byte_range.start < PASSTHROUGH_SEEK_HEADER_BYTES
        and byte_range.end < PASSTHROUGH_SEEK_HEADER_BYTES
    )


def _is_header_crossing_range(byte_range: ByteRange | None) -> bool:
    if byte_range is None:
        return False
    header_limit = _seek_prefix_cache_limit()
    return header_limit > 0 and byte_range.start < header_limit <= byte_range.end


def _seek_from_byte_range(value: str | None, path: Path, duration: float, codec: str = "") -> float | None:
    total = _estimated_passthrough_size(path, duration, codec)
    byte_range = _parse_byte_range(value, total)
    if byte_range is None or duration <= 0:
        return None
    ratio = min(1.0, max(0.0, byte_range.start / total))
    log.info(
        "passthrough byte seek map: range=%r total=%d ratio=%.6f mapped_t=%.3fs codec=%s",
        value, total, ratio, ratio * duration, codec,
    )
    return ratio * duration


def _range_unsatisfiable(value: str | None, path: Path, duration: float, codec: str = "") -> bool:
    total = _estimated_passthrough_size(path, duration, codec)
    return bool(value) and total > 0 and _parse_byte_range(value, total) is None


def _range_416(path: Path, duration: float, codec: str = "") -> Response:
    total = _estimated_passthrough_size(path, duration, codec)
    return Response(
        status_code=416,
        headers={
            "Content-Range": f"bytes */{total}",
            "Accept-Ranges": "bytes",
        },
    )


def _seek_range_416(total: int) -> Response:
    return Response(
        status_code=416,
        headers={
            "Content-Range": f"bytes */{max(0, int(total))}",
            "Accept-Ranges": "bytes",
        },
    )


def _dump_live_request_headers(
    rid: int,
    path: Path,
    request_headers: dict[str, str],
) -> None:
    if not LIVE_REQUEST_HEADER_DUMP:
        return
    try:
        _LIVE_REQUEST_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        prefix = f"live_{rid:04d}_{path.stem[:80]}"
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
        header_lines = [
            f"path: {path}",
            "",
            "[request headers]",
            *[f"{k}: {v}" for k, v in sorted(request_headers.items())],
        ]
        out = _LIVE_REQUEST_DUMP_DIR / f"{safe_prefix}_request_headers.txt"
        out.write_text(
            "\n".join(header_lines) + "\n",
            encoding="utf-8",
        )
        log.info("passthrough_live[%d] request headers dumped: %s", rid, out)
    except OSError as e:
        log.warning("passthrough_live[%d] request header dump failed: %s", rid, e)


def _passthrough_media_type() -> str:
    return "video/MP2T" if PASSTHROUGH_CONTAINER == "mpegts" else "video/mp4"


def _passthrough_content_features(backend_verdict: str | None = None) -> str:
    dlna_pn = passthrough_dlna_pn(backend_verdict)
    if PASSTHROUGH_SEEK_MODE == "bytes":
        op = DLNA_OP_BYTE_SEEK
        flags = "01700000000000000000000000000000"
    else:
        op = DLNA_OP_TIME_SEEK
        flags = DLNA_FLAGS_TIME_SEEK
    return (
        f"DLNA.ORG_PN={dlna_pn};DLNA.ORG_OP={op};DLNA.ORG_CI=1;"
        f"DLNA.ORG_FLAGS={flags}"
    )


def _live_adaptive_max_fps(path: Path, meta) -> float | None:
    base = float(PASSTHROUGH_MAX_FPS)
    if (
        not PASSTHROUGH_LIVE_ADAPTIVE_FPS
        or meta is None
        or PASSTHROUGH_LIVE_HIGH_BITRATE_FPS <= 0
    ):
        return base
    duration = float(getattr(getattr(meta, "timing", None), "duration", 0.0) or 0.0)
    try:
        src_bps = (path.stat().st_size * 8.0 / duration) if duration > 0 else 0.0
    except OSError:
        src_bps = 0.0
    if src_bps >= float(PASSTHROUGH_LIVE_HIGH_BITRATE_BPS):
        adaptive = float(PASSTHROUGH_LIVE_HIGH_BITRATE_FPS)
        selected = adaptive if base <= 0 else min(base, adaptive)
        log.info(
            "passthrough_live adaptive fps: %s src_bps=%.1fM threshold=%.1fM base=%.3f selected=%.3f",
            path.name,
            src_bps / 1_000_000.0,
            float(PASSTHROUGH_LIVE_HIGH_BITRATE_BPS) / 1_000_000.0,
            base,
            selected,
        )
        return selected
    return base


def _live_response_profile(user_agent: str) -> str:
    return live_response_profile_from_ua(user_agent, PASSTHROUGH_LIVE_DEFAULT_PROFILE)


def _is_nplayer_client(user_agent: str) -> bool:
    return is_nplayer_user_agent(user_agent)


def _is_lavf_client(user_agent: str) -> bool:
    return is_lavf_user_agent(user_agent)


def _format_fps_header(fps: float | None) -> str | None:
    if fps is None or fps <= 0:
        return None
    return str(int(fps)) if float(fps).is_integer() else f"{fps:.3f}".rstrip("0").rstrip(".")


_LIVE_MAX_SIDE = 8192


def _configured_passthrough_modes() -> tuple[str, ...]:
    raw = str(PASSTHROUGH_OUTPUT_MODE or "none").strip().lower()
    if raw == "none":
        return ()
    if raw == "all":
        return ("green", "alpha")
    out: list[str] = []
    for token in re.split(r"[,;\s]+", raw):
        if token == "all":
            tokens = ("green", "alpha")
        else:
            tokens = (token,)
        for mode in tokens:
            if mode in {"green", "alpha", "two_dvr"} and mode not in out:
                out.append(mode)
    return tuple(out)


def _select_live_output_mode(requested_mode: str) -> str:
    modes = _configured_passthrough_modes()
    if requested_mode in modes:
        return requested_mode
    if "green" in modes:
        return "green"
    if "alpha" in modes:
        return "alpha"
    if "two_dvr" in modes:
        return "two_dvr"
    return "green"


def _is_two_d_source(path: Path, width: int = 0, height: int = 0) -> bool:
    return (
        not has_vr_filename_marker(path.stem)
        and not is_half_equirectangular_source(width, height)
    )


def _live_block_reason(path: Path, meta) -> str:
    if path.suffix.lower() == ".mkv":
        policy = PASSTHROUGH_MKV_LIVE_POLICY
        if policy not in {"block", "head_cues", "allow"}:
            policy = "block"
        if policy == "block":
            return "MKV live passthrough is disabled"
        if policy == "head_cues":
            info = probe_mkv_cues(path)
            if info.needs_fix:
                return "MKV needs remux before live passthrough"
    codec = meta.codec
    if codec.width <= 0 or codec.height <= 0:
        return "missing video dimensions"
    if codec.width > _LIVE_MAX_SIDE or codec.height > _LIVE_MAX_SIDE:
        return f"video dimensions exceed live limit {_LIVE_MAX_SIDE}px"
    decision = select_backend(meta.timing, meta.codec, meta.color)
    if decision.verdict != "pynv_hevc":
        return decision.reason
    return ""


def _two_dvr_live_block_reason(path: Path, meta) -> str:
    codec = meta.codec
    if not _is_two_d_source(path, codec.width, codec.height):
        return "2D->3D live is only available for 2D source videos"
    if has_offline_two_dvr_output(path):
        return "offline 2D->3D output already exists"
    if int(codec.width or 0) > 4096:
        return "2D->3D live source width exceeds 4096px; SBS output would exceed 8K"
    bit_depth = int(codec.bit_depth if codec and codec.bit_depth > 0 else 8)
    if bit_depth > 8:
        return "2D->3D live currently supports 8-bit NV12 sources only"
    # 2D->3D live only runs on the GPU NV12 (PyNv) path; sources the backend would
    # route to the ffmpeg fallback (VFR, unsupported codec/pixel format, HDR/10-bit)
    # cannot be served and must be rejected cleanly instead of failing at startup.
    try:
        decision = select_backend(meta.timing, meta.codec, meta.color)
    except Exception as e:
        return f"2D->3D live source probe failed: {e}"
    if decision.verdict != "pynv_hevc":
        return f"2D->3D live requires the GPU NV12 path (source ineligible: {decision.reason})"
    return ""


def _probe_live_request_metadata(path: Path):
    info = probe_cached(path)
    live_meta = probe_video_metadata(path)
    return info, live_meta, _live_block_reason(path, live_meta)


def _select_passthrough_stream(
    path: Path,
    start_sec: float,
    matter,
    container: str = "mp4",
    max_fps: float | None = None,
    audio_mode_override: str | None = None,
    output_mode: str | None = None,
    preflight: bool = True,
):
    output_mode = (output_mode or PASSTHROUGH_OUTPUT_MODE).lower()
    if output_mode == "all":
        output_mode = "green"
    elif output_mode not in {"green", "alpha", "two_dvr"}:
        output_mode = _select_live_output_mode("")
    fallback_container = "mpegts" if container == "mpegts" else None
    fallback_max_fps = max_fps
    if fallback_container == "mpegts" and PASSTHROUGH_FALLBACK_MAX_FPS > 0:
        fallback_max_fps = PASSTHROUGH_FALLBACK_MAX_FPS
        if max_fps and max_fps > 0:
            fallback_max_fps = min(float(max_fps), fallback_max_fps)
    fallback_audio_mode = None
    if fallback_container == "mpegts":
        fallback_audio_mode = (audio_mode_override or "aac").lower()

    def fallback_stream() -> PassthroughStream:
        if output_mode == "alpha":
            raise RuntimeError("alpha passthrough requires the PyNv NV12 live path")
        if output_mode == "two_dvr":
            raise RuntimeError("2D->3D live requires the PyNv NV12 live path")
        return PassthroughStream(
            path,
            start_sec,
            matter,
            container=fallback_container,
            max_fps=fallback_max_fps,
            audio_mode=fallback_audio_mode,
        )

    if not USE_PYNV:
        return fallback_stream(), "ffmpeg_disabled", "ffmpeg_disabled"
    try:
        meta = probe_video_metadata(path)
        decision = select_backend(meta.timing, meta.codec, meta.color)
    except Exception as e:
        log.warning("PyNv metadata probe failed, fallback ffmpeg: %s", e)
        return fallback_stream(), "ffmpeg_probe_failed", "ffmpeg_probe_failed"
    if decision.verdict == "pynv_hevc":
        if preflight:
            try:
                PyNvPassthroughStream.preflight(path, meta)
            except Exception as e:
                log.warning("PyNv preflight failed, fallback ffmpeg: %s", e)
                return fallback_stream(), "ffmpeg_pynv_preflight_failed", "ffmpeg_fallback"
        return (
            PyNvPassthroughStream(
                path,
                start_sec,
                matter,
                meta,
                container=container,
                max_fps=max_fps,
                audio_mode_override=audio_mode_override,
                output_mode=output_mode,
            ),
            PYNV_BACKEND_LABEL,
            decision.verdict,
        )
    log.info("PyNv fallback: %s -> %s (%s)", path.name, decision.verdict, decision.reason)
    return fallback_stream(), decision.verdict, decision.verdict


def _passthrough_headers(
    media_type: str,
    start_sec: float,
    duration: float,
    path: Path,
    codec: str = "",
    range_header: str | None = None,
    include_length: bool = False,
    backend_verdict: str | None = None,
) -> dict[str, str]:
    headers = {
        "Content-Type": media_type,
        "Cache-Control": "no-store",
        "transferMode.dlna.org": "Streaming",
        "contentFeatures.dlna.org": _passthrough_content_features(backend_verdict),
    }
    frame_rate = passthrough_frame_rate()
    if frame_rate:
        headers["X-Passthrough-FrameRate"] = frame_rate
    if PASSTHROUGH_SEEK_MODE == "bytes":
        total = _estimated_passthrough_size(path, duration, codec)
        byte_range = _parse_byte_range(range_header, total)
        response_range = byte_range or ByteRange(start=0, end=max(0, total - 1), total=total)
        headers["Accept-Ranges"] = "bytes"
        headers["X-Passthrough-Estimated-Size"] = str(total)
        _, estimated_bps, estimate = estimate_for_media(path, duration, codec)
        headers["X-Passthrough-Estimated-Bps"] = str(estimated_bps)
        headers["X-Passthrough-Estimate-Source"] = estimate.source
        if include_length:
            headers["Content-Length"] = str(response_range.length)
        if byte_range is not None:
            headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{total}"
    else:
        # Transcoded streams do not have a stable byte Range map in this mode;
        # advertise DLNA time seek only.
        headers["Accept-Ranges"] = "none"
    if duration > 0:
        start_npt = _format_npt(start_sec)
        end_npt = _format_npt(duration)
        headers["TimeSeekRange.dlna.org"] = f"npt={start_npt}-{end_npt}/{end_npt}"
        headers["X-AvailableSeekRange.dlna.org"] = f"1 npt=0.000-{end_npt}"
    return headers


# ---- Thumbnails ----
@router.get("/thumb/{name:path}")
async def thumb_get(request: Request, name: str, pt: int = Query(default=0)):
    path = _safe_video_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path), thumb_passthrough=bool(pt))
    async with _thumb_lock:
        out = await asyncio.to_thread(get_thumb, path, bool(pt))
    if out is None or not out.exists():
        log.warning("thumb unavailable after generation: path=%s passthrough=%s out=%s", path.name, bool(pt), out)
        raise HTTPException(404, "thumb not available")
    return FileResponse(out, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


# ---- Passthrough streams ----
@router.get("/passthrough_live/{name:path}")
async def passthrough_live_get(
    request: Request,
    name: str,
    t: float = Query(default=0.0, ge=0.0),
    mode: str | None = Query(default=None),
    range_header: str | None = Header(default=None, alias="Range"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
):
    rid = next(_request_ids)
    # Skybox keys its HTTP pipeline on the URL extension; we advertise the
    # live URL with a ``.ts`` suffix so its TS pipeline activates. Strip
    # the suffix here so the source-file lookup still resolves the real
    # ``.mp4`` (or other source) on disk.
    name = _strip_live_route_hint_suffix(name)
    path = _safe_video_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path), passthrough_route="live")
    try:
        info, live_meta, live_block_reason = await asyncio.to_thread(_probe_live_request_metadata, path)
    except Exception as e:
        log.warning("passthrough_live[%d] metadata probe failed; reject live source: %s", rid, e)
        return Response("live passthrough unsupported: metadata probe failed", status_code=409)
    requested_t = t
    npt_t = _parse_npt_seconds(time_seek_range)
    if npt_t is not None:
        t = npt_t
    if info.duration > 0:
        t = min(t, max(0.0, info.duration - 0.01))
    requested_mode = (mode or "").lower()
    live_output_mode = _select_live_output_mode(requested_mode)

    user_agent = request.headers.get("user-agent", "")
    accept = request.headers.get("accept", "")
    request_headers = {k: v for k, v in request.headers.items()}
    x_av_client_info = request.headers.get("x-av-client-info")
    x_dlna_doc = request.headers.get("x-dlna-doc")
    host = request.headers.get("host")
    _dump_live_request_headers(rid, path, request_headers)
    log.info(
        (
            "passthrough_live[%d] request headers: ua=%r accept=%r range=%r "
            "time_seek=%r transfer=%r getfeatures=%r x_av=%r x_dlna=%r host=%r client=%s"
        ),
        rid,
        user_agent[:240],
        accept[:160],
        range_header,
        time_seek_range,
        transfer_mode,
        get_content_features,
        (x_av_client_info or "")[:240] or None,
        x_dlna_doc,
        host,
        request.client,
    )
    log.info(
        "passthrough_live[%d] start: %s @ %.2fs from %s requested_t=%.2fs mode=%s requested_mode=%r time_seek=%r",
        rid, path.name, t, request.client, requested_t, live_output_mode, requested_mode or None, time_seek_range,
    )

    if live_block_reason:
        log.info("passthrough_live[%d] reject unsupported live source: %s reason=%s", rid, path.name, live_block_reason)
        return Response(f"live passthrough unsupported: {live_block_reason}", status_code=409)
    if live_output_mode == "two_dvr":
        two_dvr_block_reason = _two_dvr_live_block_reason(path, live_meta)
        if two_dvr_block_reason:
            log.info(
                "passthrough_live[%d] reject unsupported 2D->3D live source: %s reason=%s",
                rid,
                path.name,
                two_dvr_block_reason,
            )
            return Response(f"2D->3D live unsupported: {two_dvr_block_reason}", status_code=409)
    live_max_fps = _live_adaptive_max_fps(path, live_meta)
    live_profile = _live_response_profile(user_agent)
    is_nplayer = _is_nplayer_client(user_agent)
    annotate_request(
        request,
        route_profile=live_profile,
        passthrough_mode=live_output_mode,
        requested_t=round(float(t), 3),
    )
    use_managed_live_session = live_profile in {"4xvr", "avpro", "libmpv"} or is_nplayer
    live_total = _estimated_passthrough_size(path, max(0.0, info.duration - t), PYNV_OUTPUT_CODEC)
    annotate_request(request, total_estimated_size=live_total)
    live_send_bps = _estimated_passthrough_bps(path, PYNV_OUTPUT_CODEC)
    live_send_pacing = PASSTHROUGH_SEND_REALTIME_PACING and live_profile != "libmpv"
    use_vlc_pseudo_vod = (
        live_profile == "vlc"
        and not is_nplayer
        and PASSTHROUGH_LIVE_VLC_PSEUDO_VOD
        and live_total > 0
    )
    live_byte_range = _parse_byte_range(range_header, live_total)
    if is_nplayer and range_header:
        log.info(
            "passthrough_live[%d] ignore nPlayer live range for LiveSession key stability: range=%r parsed=%r total=%d",
            rid, range_header, live_byte_range, live_total,
        )
    if (
        live_profile not in {"4xvr", "avpro", "libmpv"}
        and not is_nplayer
        and range_header
        and not _is_zero_open_range(range_header, live_byte_range)
    ):
        log.info(
            "passthrough_live[%d] return 416 for %s non-start live range before stream: range=%r total=%d",
            rid, live_profile, range_header, live_total,
        )
        return Response(
            status_code=416,
            headers={
                "Content-Range": f"bytes */{live_total}",
                "Accept-Ranges": "none",
            },
        )

    client_host = request.client.host if request.client else ""
    live_key = (
        str(path.resolve()),
        client_host,
        round(float(t), 3),
        PYNV_OUTPUT_CODEC,
        round(float(live_max_fps or 0.0), 3),
        f"{live_profile}:{live_output_mode}",
    )
    lavf_policy = PASSTHROUGH_LIVE_LAVF_POLICY
    if lavf_policy not in {"active_only", "reject", "allow"}:
        lavf_policy = "active_only"
    active_same_device = False
    if live_profile == "lavf":
        if lavf_policy == "reject":
            log.info(
                "passthrough_live[%d] return 409 for Lavf side request by policy: range=%r path=%s",
                rid, range_header, path.name,
            )
            return Response("passthrough live side request rejected", status_code=409, headers={"Retry-After": "1"})
        if lavf_policy == "active_only":
            async with _active_lock:
                active_same_device = any(
                    _owner_base(active_owner) == ("live", client_host)
                    and _owner_kind(active_owner) in ("vlc", "default", "")
                    for active_owner in _active_streams.values()
                )
    if lavf_policy == "active_only" and active_same_device:
        log.info(
            "passthrough_live[%d] return 409 for Lavf side request while VLC/default stream is active: range=%r path=%s",
            rid, range_header, path.name,
        )
        return Response("passthrough live active", status_code=409, headers={"Retry-After": "1"})
    # Skybox fires bare "libmpv" UA chapter-thumbnail probes for every
    # time-sliced DLNA item the moment the file is opened. Real chapter
    # thumbnails come from /thumb (returns JPEG); the libmpv probes here
    # are just connectivity checks Skybox does as a side-effect.
    #
    # We must NOT serve them from the existing playback session's cache
    # snapshot: 8K SBS HEVC alpha has a 30MB-class prefix, and a burst of
    # ~10 probes will dump 300MB into Wi-Fi at once, starving the real
    # SKYBOX UA playback connection. Pcap evidence: bandwidth-split made
    # the primary fall from 30fps target to 17fps actual, locking the
    # player on a permanent loading spinner.
    #
    # The probes are user-confirmed discardable. Always fast-fail 503 so
    # they never reserve a slot, start GPU work, or consume bandwidth.
    # SKYBOX/x.y.z UA — the real playback path — is unaffected.
    if is_libmpv_screenshot_probe_ua(user_agent):
        annotate_request(request, screenshot_probe=True)
        log.info(
            "passthrough_live[%d] libmpv screenshot probe rejected (preserves bandwidth for real playback): key=%s",
            rid,
            _live_session_log_key(live_key),
        )
        return Response(
            "passthrough live screenshot probe rejected",
            status_code=503,
            headers={"Retry-After": "1"},
        )

    cached_session = await _get_live_session(live_key)
    if cached_session is not None:
        async with cached_session.lock:
            take_primary = not any(subscriber.primary for subscriber in cached_session.subscribers)
        log.info(
            "passthrough_live[%d] live cache hit: key=%s bytes=%d frames=%d primary=%s",
            rid,
            _live_session_log_key(live_key),
            cached_session.bytes_emitted,
            cached_session.frames_produced,
            take_primary,
        )
        return StreamingResponse(
            cached_session.subscribe(rid, primary=take_primary),
            status_code=200,
            headers=dict(cached_session.headers),
            media_type="video/MP2T",
        )
    await _close_idle_live_sessions_for_request(live_key, rid)
    live_starting_at: float | None = None
    # libmpv/Skybox enables same-owner preempt below for different-t chapter
    # probes. Same-live_key duplicates must NOT take that path; they need to
    # wait briefly for the in-flight starter to register a LiveSession and
    # then join via subscribe(primary=False). Without this debounce, a near-
    # simultaneous duplicate GET would race past _get_live_session, fall into
    # _take_active_slot, and preempt the original starter before it produces
    # any bytes — the failure mode HANDOVER 2026-05-10 warned about.
    needs_startup_debounce = is_nplayer or live_profile == "libmpv"
    if needs_startup_debounce:
        debounce_label = "nPlayer" if is_nplayer else "libmpv"
        now = asyncio.get_running_loop().time()
        async with _live_session_lock:
            started_at = _live_starting.get(live_key)
        if started_at is not None and now - started_at < _LIVE_NPLAYER_START_DEBOUNCE_SEC:
            annotate_request(
                request,
                duplicate_startup=True,
                duplicate_startup_age_ms=int(max(0.0, now - started_at) * 1000),
            )
            deadline = now + _LIVE_NPLAYER_START_DEBOUNCE_SEC
            while asyncio.get_running_loop().time() < deadline:
                cached_session = await _get_live_session(live_key)
                if cached_session is not None:
                    log.info(
                        "passthrough_live[%d] %s duplicate startup joined cache: key=%s age=%.3fs",
                        rid,
                        debounce_label,
                        _live_session_log_key(live_key),
                        asyncio.get_running_loop().time() - started_at,
                    )
                    return StreamingResponse(
                        cached_session.subscribe(rid, primary=False),
                        status_code=200,
                        headers=dict(cached_session.headers),
                        media_type="video/MP2T",
                    )
                await asyncio.sleep(0.05)
            async with _live_session_lock:
                still_starting = _live_starting.get(live_key) == started_at
            if still_starting:
                log.info(
                    "passthrough_live[%d] return 409 %s duplicate startup still pending: key=%s age=%.3fs",
                    rid,
                    debounce_label,
                    _live_session_log_key(live_key),
                    asyncio.get_running_loop().time() - started_at,
                )
                return Response("passthrough live duplicate startup", status_code=409, headers={"Retry-After": "1"})
        async with _live_session_lock:
            _live_starting[live_key] = now
        live_starting_at = now

    if live_output_mode == "two_dvr":
        await _close_active_two_dvr_for_client(client_host, rid, keep_key=live_key)

    slot_token = object()
    # nPlayer, 4xvr/avpro, and libmpv may replace an old live stream from the
    # same device. nPlayer does not reliably notify the server when the user
    # leaves a live item; libmpv/Skybox sends bursts of different-t chapter
    # probes that would otherwise queue 10s and 503. Same-live_key duplicates
    # for these profiles are caught earlier by the startup debounce above and
    # join via subscribe(primary=False), so reaching this point with a same-
    # owner active slot means a genuinely different request that should win.
    owner = ("live", client_host, live_profile)
    preempted = await _take_active_slot(
        slot_token,
        who=f"live:{path.name}@{t:.2f}s",
        owner=owner,
        allow_same_owner_preempt=is_nplayer or live_profile in {"4xvr", "avpro", "libmpv"},
        allow_same_client_preempt=PASSTHROUGH_SEEK_ENABLED,
    )
    if preempted is False:
        await _clear_live_starting(live_key, live_starting_at)
        log.info("passthrough_live[%d] return 503 busy", rid)
        return Response("passthrough live busy", status_code=503, headers={"Retry-After": "2"})
    if preempted is not None:
        await _close_preempted_stream(preempted, f"live:{path.name}@{t:.2f}s")

    try:
        live_audio_override = (
            PASSTHROUGH_AUDIO_MPEGTS_VLC
            if live_profile in {"vlc", "lavf"} and PASSTHROUGH_AUDIO_MPEGTS_VLC != "auto"
            else None
        )

        live_acquire_timeout = max(PASSTHROUGH_BUSY_WAIT_SEC, 1.0)

        def build_stream():
            matter = None if live_output_mode == "two_dvr" else acquire_matter(blocking=True, timeout=live_acquire_timeout)
            if matter is None:
                if live_output_mode != "two_dvr":
                    return None
            try:
                stream_tuple = _select_passthrough_stream(
                    path,
                    t,
                    matter,
                    container="mpegts",
                    max_fps=live_max_fps,
                    audio_mode_override=live_audio_override,
                    output_mode=live_output_mode,
                    preflight=False,
                )
            except BaseException:
                release_matter(matter)
                raise
            return stream_tuple, matter

        built = await asyncio.to_thread(build_stream)
        if built is None:
            await _release_active_slot(slot_token)
            await _clear_live_starting(live_key, live_starting_at)
            log.warning(
                "passthrough_live[%d] return 503 matter pool exhausted after %.1fs",
                rid, live_acquire_timeout,
            )
            return Response(
                "passthrough live busy", status_code=503, headers={"Retry-After": "2"}
            )
        (stream, stream_backend, stream_verdict), live_matter = built
        if live_output_mode == "two_dvr" and float(getattr(stream, "output_fps", 0.0) or 0.0) <= 0.0:
            try:
                stream.output_fps = float(live_meta.timing.effective_fps(live_max_fps))
            except Exception:
                pass
        async with _active_lock:
            if slot_token in _active_streams:
                if live_matter is not None:
                    _active_matter[slot_token] = live_matter
                live_matter = None
        if live_matter is not None:
            # The slot was preempted while we were building; do not leak the matter.
            release_matter(live_matter)
        if not await _replace_active_slot(slot_token, stream, close_on_failure=stream):
            await _clear_live_starting(live_key, live_starting_at)
            log.info("passthrough_live[%d] return 409 preempted before stream", rid)
            return Response("passthrough live preempted", status_code=409, headers={"Retry-After": "1"})
    except asyncio.CancelledError:
        await _release_active_slot(slot_token)
        await _clear_live_starting(live_key, live_starting_at)
        raise
    except Exception:
        await _release_active_slot(slot_token)
        await _clear_live_starting(live_key, live_starting_at)
        raise

    headers = {
        "Content-Type": "video/MP2T",
        "Cache-Control": "no-store",
        "transferMode.dlna.org": "Streaming",
        "X-Passthrough-Mode": f"live-mpegts-{live_output_mode}",
        "X-Passthrough-Seek-Time": f"{t:.3f}",
        "X-Passthrough-Backend": stream_backend,
        "X-Passthrough-Backend-Verdict": stream_verdict,
    }
    if live_total > 0:
        headers["X-Passthrough-Estimated-Size"] = str(live_total)
    if info.duration > 0:
        start_npt = _format_npt(t)
        end_npt = _format_npt(info.duration)
        headers["TimeSeekRange.dlna.org"] = f"npt={start_npt}-{end_npt}/{end_npt}"
        headers["X-AvailableSeekRange.dlna.org"] = f"1 npt=0.000-{end_npt}"
    response_fps = float(getattr(stream, "output_fps", 0.0) or 0.0)
    if response_fps <= 0:
        try:
            response_fps = float(live_meta.timing.effective_fps(live_max_fps))
        except Exception:
            response_fps = float(getattr(stream, "max_fps", 0.0) or live_max_fps or 0.0)
    frame_rate = _format_fps_header(response_fps) or passthrough_frame_rate()
    if frame_rate:
        headers["X-Passthrough-FrameRate"] = frame_rate
    if not use_managed_live_session:
        if use_vlc_pseudo_vod:
            headers["Accept-Ranges"] = "bytes"
            headers["Content-Range"] = f"bytes 0-{live_total - 1}/{live_total}"
            headers.pop("Content-Length", None)
            headers["contentFeatures.dlna.org"] = (
                "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
                f"DLNA.ORG_OP={DLNA_OP_BYTE_SEEK};"
                "DLNA.ORG_CI=1;"
                f"DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
            )
        else:
            headers["Accept-Ranges"] = "none"
            headers.pop("Content-Range", None)
            headers.pop("Content-Length", None)
            headers["contentFeatures.dlna.org"] = (
                "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
                f"DLNA.ORG_OP={DLNA_OP_TIME_SEEK};"
                "DLNA.ORG_CI=1;"
                f"DLNA.ORG_FLAGS={DLNA_FLAGS_TIME_SEEK}"
            )
    else:
        headers["contentFeatures.dlna.org"] = (
            "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
            f"DLNA.ORG_OP={DLNA_OP_TIME_SEEK};"
            "DLNA.ORG_CI=1;"
            f"DLNA.ORG_FLAGS={DLNA_FLAGS_TIME_SEEK}"
        )

    log.info(
        "passthrough_live[%d] response: profile=%s status=%s backend=%s verdict=%s ignored_range=%r live_total_est=%d send_bps=%d send_pacing=%s headers=%s",
        rid,
        live_profile,
        206 if use_vlc_pseudo_vod else 200,
        stream_backend,
        stream_verdict,
        range_header,
        live_total,
        live_send_bps,
        live_send_pacing,
        headers,
    )

    if not use_managed_live_session:
        effective_stall_timeout = PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC
        if is_nplayer and effective_stall_timeout <= 0:
            effective_stall_timeout = 6.0
        stream_iter = stream.iter_bytes()
        preroll_chunks: list[bytes] = []
        preroll_bytes = 0
        preroll_started = asyncio.get_running_loop().time()
        try:
            while True:
                first_live_chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=_LIVE_FIRST_CHUNK_TIMEOUT_SEC,
                )
                if first_live_chunk:
                    preroll_chunks.append(first_live_chunk)
                    preroll_bytes += len(first_live_chunk)
                    break
                log.warning("passthrough_live[%d] ignored empty first chunk before VLC response", rid)
            preroll_target = (
                _LIVE_VLC_PREROLL_BYTES
                if live_profile == "vlc" and not is_nplayer
                else 0
            )
            preroll_deadline = asyncio.get_running_loop().time() + _LIVE_VLC_PREROLL_TIMEOUT_SEC
            while preroll_target > 0 and preroll_bytes < preroll_target:
                remaining = preroll_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if not chunk:
                    continue
                preroll_chunks.append(chunk)
                preroll_bytes += len(chunk)
        except StopAsyncIteration:
            if not preroll_chunks:
                await asyncio.to_thread(stream.close)
                await _release_active_slot(stream)
                log.warning("passthrough_live[%d] return 503 no stream data before VLC response", rid)
                return Response("passthrough live no data", status_code=503, headers={"Retry-After": "2"})
        except asyncio.TimeoutError:
            await asyncio.to_thread(stream.close)
            await _release_active_slot(stream)
            log.warning(
                "passthrough_live[%d] return 504 VLC first chunk timeout after %.1fs",
                rid,
                _LIVE_FIRST_CHUNK_TIMEOUT_SEC,
            )
            return Response("passthrough live first chunk timeout", status_code=504, headers={"Retry-After": "2"})
        except Exception:
            await asyncio.to_thread(stream.close)
            await _release_active_slot(stream)
            raise
        log.info(
            "passthrough_live[%d] preroll ready: profile=%s nplayer=%s chunks=%d bytes=%d target=%d elapsed=%.3fs",
            rid,
            live_profile,
            is_nplayer,
            len(preroll_chunks),
            preroll_bytes,
            preroll_target,
            asyncio.get_running_loop().time() - preroll_started,
        )

        async def vlc_gen():
            sent = 0
            first_chunk = True
            next_progress = _LIVE_PROGRESS_INTERVAL_BYTES
            disconnect_task: asyncio.Task | None = None
            pump_task: asyncio.Task | None = None
            last_send_wall = asyncio.get_running_loop().time()
            pace_start_wall = last_send_wall
            delivery_queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=PASSTHROUGH_LIVE_SUB_QUEUE_CHUNKS)
            released = False
            light_match_version = get_light_match().version

            async def close_and_release(reason: str) -> None:
                nonlocal released
                if released:
                    return
                released = True
                try:
                    try:
                        await asyncio.wait_for(asyncio.to_thread(stream.close), timeout=3.0)
                    except asyncio.TimeoutError:
                        log.warning("passthrough_live[%d] stream close timeout during %s", rid, reason)
                    except Exception as e:
                        log.warning("passthrough_live[%d] stream close failed during %s: %s", rid, reason, e)
                finally:
                    await _release_active_slot(stream)

            def signal_live_end() -> None:
                try:
                    delivery_queue.put_nowait(_LIVE_END)
                    return
                except asyncio.QueueFull:
                    pass
                try:
                    delivery_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    delivery_queue.put_nowait(_LIVE_END)
                except asyncio.QueueFull:
                    log.warning("passthrough_live[%d] unable to enqueue live end marker", rid)

            async def pump_stream():
                try:
                    async for item in stream_iter:
                        if item:
                            await delivery_queue.put(item)
                finally:
                    signal_live_end()

            async def disconnect_watchdog():
                nonlocal last_send_wall, released
                while True:
                    await asyncio.sleep(0.25)
                    try:
                        disconnected = await request.is_disconnected()
                    except Exception as e:
                        log.info("passthrough_live[%d] disconnect watchdog stopped: %s", rid, e)
                        return
                    if disconnected:
                        log.info("passthrough_live[%d] disconnect watchdog stopped response", rid)
                        return
                    if (
                        effective_stall_timeout > 0
                        and sent > 0
                        and asyncio.get_running_loop().time() - last_send_wall > effective_stall_timeout
                    ):
                        log.info(
                            "passthrough_live[%d] send stall watchdog closing stream: sent=%d stream_bytes=%d frames=%d idle=%.1fs",
                            rid,
                            sent,
                            getattr(stream, "bytes_emitted", -1),
                            getattr(stream, "frames_produced", -1),
                            asyncio.get_running_loop().time() - last_send_wall,
                        )
                        if pump_task is not None:
                            pump_task.cancel()
                        signal_live_end()
                        await close_and_release("stall watchdog")
                        return

            try:
                disconnect_task = asyncio.create_task(disconnect_watchdog())
                pump_task = asyncio.create_task(pump_stream())
                for chunk in preroll_chunks:
                    sent += len(chunk)
                    if live_send_pacing:
                        await _pace_live_send(pace_start_wall, sent, live_send_bps)
                    last_send_wall = asyncio.get_running_loop().time()
                    if first_chunk:
                        first_chunk = False
                        log.info(
                            "passthrough_live[%d] first chunk: len=%d sent=%d stream_bytes=%d",
                            rid, len(chunk), sent, getattr(stream, "bytes_emitted", -1),
                        )
                    yield chunk
                while True:
                    item = await delivery_queue.get()
                    if item is _LIVE_END:
                        break
                    chunk = item
                    current_light_match_version = get_light_match().version
                    if LIGHT_MATCH_FLUSH_QUEUES and current_light_match_version != light_match_version:
                        dropped_chunks, dropped_bytes, saw_end = _drain_live_queue_nowait(delivery_queue)
                        log.info(
                            "passthrough_live[%d] light match changed v%d->v%d; dropped VLC delivery current plus queued_chunks=%d queued_bytes=%d end=%s",
                            rid,
                            light_match_version,
                            current_light_match_version,
                            dropped_chunks,
                            dropped_bytes + len(chunk),
                            saw_end,
                        )
                        light_match_version = current_light_match_version
                        continue
                    sent += len(chunk)
                    last_send_wall = asyncio.get_running_loop().time()
                    if first_chunk:
                        first_chunk = False
                        log.info(
                            "passthrough_live[%d] first chunk: len=%d sent=%d stream_bytes=%d",
                            rid, len(chunk), sent, getattr(stream, "bytes_emitted", -1),
                        )
                    if sent >= next_progress:
                        log.info(
                            "passthrough_live[%d] progress: sent=%d stream_bytes=%d frames=%d",
                            rid, sent, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1),
                        )
                        while next_progress <= sent:
                            next_progress += _LIVE_PROGRESS_INTERVAL_BYTES
                    if live_send_pacing:
                        await _pace_live_send(pace_start_wall, sent, live_send_bps)
                    yield chunk
            finally:
                log.info(
                    "passthrough_live[%d] finally begin: sent=%d stream_bytes=%d frames=%d",
                    rid, sent, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1),
                )
                if disconnect_task is not None:
                    disconnect_task.cancel()
                if pump_task is not None:
                    pump_task.cancel()
                await close_and_release("response cleanup")
                pending_tasks = [task for task in (disconnect_task, pump_task) if task is not None]
                if pending_tasks:
                    try:
                        await asyncio.wait_for(asyncio.gather(*pending_tasks, return_exceptions=True), timeout=1.0)
                    except asyncio.TimeoutError:
                        log.warning("passthrough_live[%d] cleanup task wait timeout", rid)
                log.info("passthrough_live[%d] finally done: sent=%d", rid, sent)

        vlc_status_code = 206 if use_vlc_pseudo_vod else 200
        return StreamingResponse(vlc_gen(), status_code=vlc_status_code, headers=headers, media_type="video/MP2T")

    stream_iter = stream.iter_bytes()
    try:
        while True:
            first_live_chunk = await asyncio.wait_for(
                stream_iter.__anext__(),
                timeout=_LIVE_FIRST_CHUNK_TIMEOUT_SEC,
            )
            if first_live_chunk:
                break
            log.warning("passthrough_live[%d] ignored empty first chunk", rid)
    except StopAsyncIteration:
        await asyncio.to_thread(stream.close)
        await _release_active_slot(stream)
        await _clear_live_starting(live_key, live_starting_at)
        log.warning("passthrough_live[%d] return 503 no stream data before response", rid)
        return Response("passthrough live no data", status_code=503, headers={"Retry-After": "2"})
    except asyncio.TimeoutError:
        await asyncio.to_thread(stream.close)
        await _release_active_slot(stream)
        await _clear_live_starting(live_key, live_starting_at)
        log.warning(
            "passthrough_live[%d] return 504 first chunk timeout after %.1fs",
            rid,
            _LIVE_FIRST_CHUNK_TIMEOUT_SEC,
        )
        return Response("passthrough live first chunk timeout", status_code=504, headers={"Retry-After": "2"})
    except Exception:
        await asyncio.to_thread(stream.close)
        await _release_active_slot(stream)
        await _clear_live_starting(live_key, live_starting_at)
        raise

    session = LiveSession(live_key, stream, headers, first_live_chunk, owner, rid, live_send_bps, live_send_pacing)
    if not await _replace_active_slot(stream, session):
        # _replace_active_slot has already closed `stream` (it owned old_stream
        # lifecycle on the failure path, closing it before releasing the Matter
        # to prevent another acquirer from grabbing a Matter still in use).
        await _clear_live_starting(live_key, live_starting_at)
        log.info("passthrough_live[%d] return 409 preempted before live session", rid)
        return Response("passthrough live preempted", status_code=409, headers={"Retry-After": "1"})
    await _put_live_session(live_key, session)
    await _clear_live_starting(live_key, live_starting_at)
    session.start(stream_iter)
    effective_stall_timeout = PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC
    if is_nplayer and effective_stall_timeout <= 0:
        effective_stall_timeout = 6.0

    async def gen():
        sent = 0
        first_chunk = True
        next_progress = _LIVE_PROGRESS_INTERVAL_BYTES
        disconnect_task: asyncio.Task | None = None
        stream_task: asyncio.Task | None = None
        last_send_wall = asyncio.get_running_loop().time()

        async def disconnect_watchdog():
            nonlocal last_send_wall
            while True:
                await asyncio.sleep(0.25)
                try:
                    disconnected = await request.is_disconnected()
                except Exception as e:
                    log.info("passthrough_live[%d] disconnect watchdog stopped: %s", rid, e)
                    return
                if disconnected:
                    log.info("passthrough_live[%d] disconnect watchdog stopped response", rid)
                    return
                if (
                    effective_stall_timeout > 0
                    and sent > 0
                    and asyncio.get_running_loop().time() - last_send_wall > effective_stall_timeout
                ):
                    log.info(
                        "passthrough_live[%d] send stall watchdog closing stream: sent=%d stream_bytes=%d frames=%d idle=%.1fs",
                        rid,
                        sent,
                        getattr(stream, "bytes_emitted", -1),
                        getattr(stream, "frames_produced", -1),
                        asyncio.get_running_loop().time() - last_send_wall,
                    )
                    await session.close("send stall watchdog")
                    if stream_task is not None and not stream_task.done():
                        stream_task.cancel()
                    return

        try:
            stream_task = asyncio.current_task()
            disconnect_task = asyncio.create_task(disconnect_watchdog())
            async for chunk in session.subscribe(rid, primary=True):
                sent += len(chunk)
                last_send_wall = asyncio.get_running_loop().time()
                if first_chunk:
                    first_chunk = False
                    log.info(
                        "passthrough_live[%d] first chunk: len=%d sent=%d stream_bytes=%d",
                        rid, len(chunk), sent, getattr(stream, "bytes_emitted", -1),
                    )
                if sent >= next_progress:
                    log.info(
                        "passthrough_live[%d] progress: sent=%d stream_bytes=%d frames=%d",
                        rid, sent, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1),
                    )
                    while next_progress <= sent:
                        next_progress += _LIVE_PROGRESS_INTERVAL_BYTES
                yield chunk
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
            log.info(
                "passthrough_live[%d] finally begin: sent=%d stream_bytes=%d frames=%d",
                rid, sent, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1),
            )
            log.info("passthrough_live[%d] finally done: sent=%d", rid, sent)

    return StreamingResponse(
        gen(),
        status_code=200,
        headers=headers,
        media_type="video/MP2T",
    )


def _seek_output_mode(requested_mode: str | None) -> str:
    requested = (requested_mode or "").lower()
    modes = tuple(mode for mode in _configured_passthrough_modes() if mode in {"green", "alpha"})
    if "green" in modes and "alpha" in modes:
        return requested if requested in {"green", "alpha"} else "green"
    if "alpha" in modes:
        return "alpha"
    return "green"


async def _serve_seek_prefix_or_retry(
    *,
    rid: int,
    path: Path,
    media_type: str,
    headers: dict[str, str],
    byte_range: ByteRange,
    probe_key: str,
    range_header: str | None,
) -> Response:
    async with _probe_cache_lock:
        cached = _probe_cache.get(probe_key, b"")
    if byte_range.end < len(cached):
        body = cached[byte_range.start:byte_range.end + 1]
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{byte_range.total}"
        headers["Content-Length"] = str(len(body))
        headers["X-Passthrough-Probe-Source"] = "seek-prefix-cache"
        log.info(
            "passthrough_seek[%d] prefix cache hit: %s range=%s served=%d-%d cached=%d len=%d",
            rid, path.name, range_header, byte_range.start, byte_range.end, len(cached), len(body),
        )
        return Response(body, status_code=206, headers=headers, media_type=media_type)
    log.info(
        "passthrough_seek[%d] prefix cache miss: %s range=%s start=%d cached=%d header=%d limit=%d",
        rid, path.name, range_header, byte_range.start, len(cached),
        PASSTHROUGH_SEEK_HEADER_BYTES, _seek_prefix_cache_limit(),
    )
    return Response(
        "seek prefix cache not ready",
        status_code=503,
        headers={
            "Retry-After": "1",
            "Accept-Ranges": "bytes",
            "Content-Type": media_type,
            "X-Passthrough-Probe-Source": "seek-prefix-cache-not-ready",
        },
        media_type=media_type,
    )


async def _seek_prefix_splice_or_retry(
    *,
    rid: int,
    path: Path,
    media_type: str,
    headers: dict[str, str],
    byte_range: ByteRange,
    probe_key: str,
    range_header: str | None,
) -> tuple[bytes, int] | Response:
    header_limit = _seek_prefix_cache_limit()
    deadline = asyncio.get_running_loop().time() + _PREFIX_CACHE_WAIT_SEC
    cached = b""
    while header_limit > 0:
        async with _probe_cache_lock:
            cached = _probe_cache.get(probe_key, b"")
        if len(cached) >= header_limit:
            prefix = cached[byte_range.start:header_limit]
            headers["X-Passthrough-Probe-Source"] = "seek-prefix-cache-crossing"
            log.info(
                "passthrough_seek[%d] prefix crossing cache hit: %s range=%s served=%d-%d cached=%d skip=%d len=%d",
                rid, path.name, range_header, byte_range.start, header_limit - 1, len(cached), header_limit, len(prefix),
            )
            return prefix, header_limit
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.05)
    log.info(
        "passthrough_seek[%d] prefix crossing cache miss: %s range=%s start=%d cached=%d header=%d limit=%d",
        rid, path.name, range_header, byte_range.start, len(cached),
        PASSTHROUGH_SEEK_HEADER_BYTES, header_limit,
    )
    return Response(
        "seek prefix cache not ready",
        status_code=503,
        headers={
            "Retry-After": "1",
            "Accept-Ranges": "bytes",
            "Content-Type": media_type,
            "X-Passthrough-Probe-Source": "seek-prefix-cache-not-ready",
        },
        media_type=media_type,
    )


@router.head("/passthrough_seek/{name:path}")
async def passthrough_seek_head(
    request: Request,
    name: str,
    mode: str | None = Query(default=None),
    range_header: str | None = Header(default=None, alias="Range"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
):
    rid = next(_request_ids)
    path, route_container = _safe_seek_video_path(name)
    user_agent = request.headers.get("user-agent", "")
    allowed, reason, route_profile = _seek_route_allowed(user_agent)
    output_mode = _seek_output_mode(mode)
    annotate_request(
        request,
        media_name=path.name,
        media_path=str(path),
        passthrough_route="seekable",
        route_profile=route_profile,
        passthrough_mode=output_mode,
    )
    if not allowed:
        log.info("passthrough_seek[%d] HEAD blocked: reason=%s profile=%s", rid, reason, route_profile)
        return _seek_blocked_response(reason)
    info = probe_cached(path)
    codec = PYNV_OUTPUT_CODEC
    client_host = request.client.host if request.client else ""
    container = route_container or _seek_container()
    media_type = _seek_media_type(container)
    total = _estimated_seek_passthrough_size(path, info.duration, codec, client_host, container)
    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None:
        return _seek_range_416(total)
    t = 0.0
    mapped = None
    npt_t = _parse_npt_seconds(time_seek_range)
    if npt_t is not None:
        t = npt_t
    elif byte_range is not None:
        mapped = map_byte_start_to_time(
            start=byte_range.start,
            total=total,
            duration_sec=info.duration,
            header_bytes=PASSTHROUGH_SEEK_HEADER_BYTES,
            output_fps=_seek_output_fps(info),
            gop_frames=PASSTHROUGH_GOP,
        )
        t = mapped.snapped_time_sec
    if info.duration > 0:
        t = min(t, max(0.0, info.duration - 0.01))
    headers = _seek_headers(
        path=path,
        duration=info.duration,
        codec=codec,
        total=total,
        start_sec=t,
        range_header=range_header,
        include_length=True,
        container=container,
        info=info,
    )
    # Intentional compatibility deviation from RFC 7233: `bytes=0-` is treated
    # as a full-start probe and answered like the non-Range startup path. Some
    # DLNA/VR clients behave better when the first open-ended request is not a
    # partial-content response; non-zero ranges still return 206.
    status_code = 206 if range_header and byte_range is not None and not _is_zero_open_range(range_header, byte_range) else 200
    if status_code == 200:
        headers.pop("Content-Range", None)
    _apply_seek_diag_headers(headers, start_sec=t, output_mode=output_mode, container=container, mapped=mapped)
    log.info(
        "passthrough_seek[%d] HEAD: %s @ %.2fs profile=%s reason=%s range=%r time_seek=%r getfeatures=%r transfer=%r",
        rid, path.name, t, route_profile, reason, range_header, time_seek_range, get_content_features, transfer_mode,
    )
    return Response(status_code=status_code, headers=headers, media_type=media_type)


@router.get("/passthrough_seek/{name:path}")
async def passthrough_seek_get(
    request: Request,
    name: str,
    mode: str | None = Query(default=None),
    range_header: str | None = Header(default=None, alias="Range"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
):
    rid = next(_request_ids)
    path, route_container = _safe_seek_video_path(name)
    user_agent = request.headers.get("user-agent", "")
    accept = request.headers.get("accept", "")
    allowed, reason, route_profile = _seek_route_allowed(user_agent)
    output_mode = _seek_output_mode(mode)
    annotate_request(
        request,
        media_name=path.name,
        media_path=str(path),
        passthrough_route="seekable",
        route_profile=route_profile,
        passthrough_mode=output_mode,
    )
    if not allowed:
        log.info("passthrough_seek[%d] blocked: reason=%s profile=%s ua=%r", rid, reason, route_profile, user_agent[:160])
        return _seek_blocked_response(reason)

    info = probe_cached(path)
    codec = PYNV_OUTPUT_CODEC
    client_host = request.client.host if request.client else ""
    container = route_container or _seek_container()
    media_type = _seek_media_type(container)
    total = _estimated_seek_passthrough_size(path, info.duration, codec, client_host, container)
    annotate_request(request, total_estimated_size=total)
    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None:
        return _seek_range_416(total)

    t = 0.0
    mapped = None
    npt_t = _parse_npt_seconds(time_seek_range)
    if npt_t is not None:
        t = npt_t
    elif byte_range is not None and not _is_zero_open_range(range_header, byte_range):
        mapped = map_byte_start_to_time(
            start=byte_range.start,
            total=total,
            duration_sec=info.duration,
            header_bytes=PASSTHROUGH_SEEK_HEADER_BYTES,
            output_fps=_seek_output_fps(info),
            gop_frames=PASSTHROUGH_GOP,
        )
        t = mapped.snapped_time_sec
    if info.duration > 0:
        t = min(t, max(0.0, info.duration - 0.01))

    probe_key = _seek_probe_cache_key(path, codec, info.duration, total, container)
    headers = _seek_headers(
        path=path,
        duration=info.duration,
        codec=codec,
        total=total,
        start_sec=t,
        range_header=range_header,
        include_length=True,
        container=container,
        info=info,
    )
    _apply_seek_diag_headers(headers, start_sec=t, output_mode=output_mode, container=container, mapped=mapped)

    if byte_range is not None and _is_tail_probe_range(byte_range):
        body = b"\x00" * byte_range.length
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{total}"
        headers["Content-Length"] = str(len(body))
        headers["X-Passthrough-Probe-Source"] = "seek-tail-empty"
        log.info(
            "passthrough_seek[%d] tail probe ignored: %s range=%s total=%d",
            rid, path.name, range_header, total,
        )
        return Response(body, status_code=206, headers=headers, media_type=media_type)

    if (
        byte_range is not None
        and not _is_zero_open_range(range_header, byte_range)
        and _is_header_only_range(byte_range)
    ):
        return await _serve_seek_prefix_or_retry(
            rid=rid,
            path=path,
            media_type=media_type,
            headers=headers,
            byte_range=byte_range,
            probe_key=probe_key,
            range_header=range_header,
        )

    prefix_splice = b""
    skip_initial_bytes = 0
    if (
        byte_range is not None
        and not _is_zero_open_range(range_header, byte_range)
        and _is_header_crossing_range(byte_range)
    ):
        splice = await _seek_prefix_splice_or_retry(
            rid=rid,
            path=path,
            media_type=media_type,
            headers=headers,
            byte_range=byte_range,
            probe_key=probe_key,
            range_header=range_header,
        )
        if isinstance(splice, Response):
            return splice
        prefix_splice, skip_initial_bytes = splice
    if skip_initial_bytes:
        log.info(
            "passthrough_seek[%d] stream header-crossing range after cached prefix with skip=%d range=%r prefix=%d",
            rid, skip_initial_bytes, range_header, len(prefix_splice),
        )

    log.info(
        "passthrough_seek[%d] start: %s @ %.2fs profile=%s mode=%s range=%r ua=%r accept=%r time_seek=%r transfer=%r",
        rid, path.name, t, route_profile, output_mode, range_header, user_agent[:160], accept[:160], time_seek_range, transfer_mode,
    )
    slot_token = object()
    owner = (str(path.resolve()), client_host, "seek")
    preempted = await _take_active_slot(
        slot_token,
        who=f"seek:{path.name}@{t:.2f}s",
        owner=owner,
        allow_same_client_preempt=PASSTHROUGH_SEEK_ENABLED,
    )
    if preempted is False:
        log.info("passthrough_seek[%d] return 503 busy", rid)
        return Response("seekable passthrough busy", status_code=503, headers={"Retry-After": "2"})
    if preempted is not None:
        await _close_preempted_stream(preempted, f"seek:{path.name}@{t:.2f}s")

    try:
        acquire_timeout = max(PASSTHROUGH_BUSY_WAIT_SEC, 1.0)
        matter = await asyncio.to_thread(acquire_matter, blocking=True, timeout=acquire_timeout)
        if matter is None:
            await _release_active_slot(slot_token)
            log.warning("passthrough_seek[%d] return 503 matter pool exhausted after %.1fs", rid, acquire_timeout)
            return Response("seekable passthrough busy", status_code=503, headers={"Retry-After": "2"})
        async with _active_lock:
            if slot_token in _active_streams:
                _active_matter[slot_token] = matter
                matter_tracked = True
            else:
                matter_tracked = False
        if not matter_tracked:
            release_matter(matter)
            log.info("passthrough_seek[%d] return 409 preempted before stream", rid)
            return Response("seekable passthrough preempted", status_code=409, headers={"Retry-After": "1"})
        stream, stream_backend, stream_verdict = _select_passthrough_stream(
            path,
            t,
            matter,
            container=container,
            output_mode=output_mode,
        )
        if not await _replace_active_slot(slot_token, stream, close_on_failure=stream):
            log.info("passthrough_seek[%d] return 409 preempted before stream", rid)
            return Response("seekable passthrough preempted", status_code=409, headers={"Retry-After": "1"})
    except Exception:
        await _release_active_slot(slot_token)
        raise

    headers["X-Passthrough-Backend"] = stream_backend
    headers["X-Passthrough-Backend-Verdict"] = stream_verdict
    # Keep the same `bytes=0-` startup compatibility behavior as HEAD above.
    status_code = 206 if range_header and byte_range is not None and not _is_zero_open_range(range_header, byte_range) else 200
    if status_code == 200:
        headers.pop("Content-Range", None)
    content_length = int(headers.get("Content-Length") or "0")
    cache_prefix = byte_range is None or byte_range.start == 0
    log.info(
        "passthrough_seek[%d] response: status=%d backend=%s verdict=%s total=%d content_length=%d byte_range=%s headers_range=%r",
        rid, status_code, stream_backend, stream_verdict, total, content_length, byte_range, headers.get("Content-Range"),
    )

    async def gen():
        sent = 0
        probe_prefix = bytearray()
        probe_prefix_limit = _seek_prefix_cache_limit() if cache_prefix else 0
        probe_prefix_flushed_len = 0
        probe_prefix_next_flush_len = (
            min(_SEEK_PREFIX_CACHE_FLUSH_STEP, probe_prefix_limit)
            if probe_prefix_limit > 0
            else 0
        )
        first_chunk = True
        skip_remaining = skip_initial_bytes
        disconnect_task: asyncio.Task | None = None

        async def disconnect_watchdog():
            while True:
                await asyncio.sleep(0.25)
                try:
                    disconnected = await request.is_disconnected()
                except Exception as e:
                    log.info("passthrough_seek[%d] disconnect watchdog stopped: %s", rid, e)
                    return
                if disconnected:
                    log.info("passthrough_seek[%d] disconnect watchdog closing stream", rid)
                    await asyncio.to_thread(stream.close)
                    return

        try:
            disconnect_task = asyncio.create_task(disconnect_watchdog())
            if prefix_splice:
                chunk = prefix_splice
                if content_length > 0 and len(chunk) > content_length:
                    chunk = chunk[:content_length]
                sent += len(chunk)
                first_chunk = False
                log.info(
                    "passthrough_seek[%d] first chunk: len=%d sent=%d source=prefix-splice",
                    rid, len(chunk), sent,
                )
                yield chunk
            async for chunk in stream.iter_bytes():
                if skip_remaining > 0:
                    if len(chunk) <= skip_remaining:
                        skip_remaining -= len(chunk)
                        continue
                    chunk = chunk[skip_remaining:]
                    skip_remaining = 0
                if content_length > 0:
                    remaining = content_length - sent
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                if probe_prefix_limit > 0 and len(probe_prefix) < probe_prefix_limit:
                    need = probe_prefix_limit - len(probe_prefix)
                    probe_prefix.extend(chunk[:need])
                    if len(probe_prefix) >= probe_prefix_next_flush_len:
                        async with _probe_cache_lock:
                            _set_probe_cache_locked(probe_key, bytes(probe_prefix))
                        probe_prefix_flushed_len = len(probe_prefix)
                        if probe_prefix_flushed_len >= probe_prefix_limit:
                            probe_prefix_next_flush_len = probe_prefix_limit + 1
                        else:
                            probe_prefix_next_flush_len = min(
                                probe_prefix_limit,
                                probe_prefix_flushed_len + _SEEK_PREFIX_CACHE_FLUSH_STEP,
                            )
                sent += len(chunk)
                if first_chunk:
                    first_chunk = False
                    log.info("passthrough_seek[%d] first chunk: len=%d sent=%d stream_bytes=%d", rid, len(chunk), sent, getattr(stream, "bytes_emitted", -1))
                yield chunk
            if PASSTHROUGH_PAD_TO_LENGTH and content_length > 0 and sent < content_length:
                log.info("passthrough_seek[%d] padding begin: sent=%d content_length=%d", rid, sent, content_length)
                pad = b"\x00" * min(64 * 1024, content_length - sent)
                while sent < content_length:
                    chunk = pad[: min(len(pad), content_length - sent)]
                    sent += len(chunk)
                    yield chunk
                log.info("passthrough_seek[%d] padding end: sent=%d", rid, sent)
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
            # Put the scarce Matter/slot back before any awaited diagnostics or
            # cache writes. Starlette may cancel the streaming task as soon as a
            # client disconnects; if cleanup is interrupted after PyNv closes
            # but before _release_active_slot(), later players see false 503
            # busy even though the worker has stopped.
            _close_stream_if_possible(stream)
            _release_active_slot_nowait(stream)
            if probe_prefix and len(probe_prefix) != probe_prefix_flushed_len:
                async with _probe_cache_lock:
                    _set_probe_cache_locked(probe_key, bytes(probe_prefix))
            if stream.bytes_emitted > 0 and (content_length <= 0 or sent >= content_length):
                if stream.frames_produced > 0 and stream.output_fps > 0:
                    elapsed_media = stream.frames_produced / stream.output_fps
                else:
                    elapsed_media = max(0.001, info.duration - t)
                record_actual_bps(
                    path,
                    codec,
                    None,
                    stream.bytes_emitted * 8 / elapsed_media,
                    elapsed_media,
                )
            log.info("passthrough_seek[%d] finally done: sent=%d", rid, sent)

    return StreamingResponse(gen(), status_code=status_code, headers=headers, media_type=media_type)


@router.head("/passthrough/{name:path}")
async def passthrough_head(
    request: Request,
    name: str,
    t: float = Query(default=0.0, ge=0.0),
    range_header: str | None = Header(default=None, alias="Range"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
):
    rid = next(_request_ids)
    path = _safe_video_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path), passthrough_route="pseudo_vod")
    info = probe_cached(path)
    estimate_codec = _passthrough_estimate_codec(path) or info.codec_name
    backend_verdict = _passthrough_backend_verdict(path)
    if PASSTHROUGH_SEEK_MODE == "bytes" and _range_unsatisfiable(range_header, path, info.duration, estimate_codec):
        return _range_416(path, info.duration, estimate_codec)
    requested_t = t
    npt_t = _parse_npt_seconds(time_seek_range)
    if npt_t is not None:
        t = npt_t
    elif PASSTHROUGH_SEEK_MODE == "bytes":
        byte_t = _seek_from_byte_range(range_header, path, info.duration, estimate_codec)
        if byte_t is not None:
            t = byte_t
    if info.duration > 0:
        t = min(t, max(0.0, info.duration - 0.01))
    media_type = _passthrough_media_type()
    log.info(
        "passthrough HEAD: %s @ %.2fs from %s container=%s media_type=%s requested_t=%.2fs time_seek=%r range=%r getfeatures=%r transfer=%r",
        path.name, t, request.client, PASSTHROUGH_CONTAINER, media_type,
        requested_t, time_seek_range, range_header, get_content_features, transfer_mode,
    )
    total = _estimated_passthrough_size(path, info.duration, estimate_codec)
    annotate_request(request, total_estimated_size=total)
    byte_range = _parse_byte_range(range_header, total)
    status_code = 206 if PASSTHROUGH_SEEK_MODE == "bytes" and range_header and not _is_zero_open_range(range_header, byte_range) else 200
    headers = _passthrough_headers(media_type, t, info.duration, path, estimate_codec, range_header, include_length=True, backend_verdict=backend_verdict)
    if status_code == 200:
        headers.pop("Content-Range", None)
    headers["X-Passthrough-Seek-Time"] = f"{t:.3f}"
    return Response(
        status_code=status_code,
        headers=headers,
        media_type=media_type,
    )


@router.get("/passthrough/{name:path}")
async def passthrough_get(
    request: Request,
    name: str,
    t: float = Query(default=0.0, ge=0.0),
    range_header: str | None = Header(default=None, alias="Range"),
    time_seek_range: str | None = Header(default=None, alias="TimeSeekRange.dlna.org"),
    get_content_features: str | None = Header(default=None, alias="getcontentFeatures.dlna.org"),
    transfer_mode: str | None = Header(default=None, alias="transferMode.dlna.org"),
):
    rid = next(_request_ids)
    path = _safe_video_path(name)
    annotate_request(request, media_name=path.name, media_path=str(path), passthrough_route="pseudo_vod")
    info = probe_cached(path)
    estimate_codec = _passthrough_estimate_codec(path) or info.codec_name
    backend_verdict = _passthrough_backend_verdict(path)
    user_agent = request.headers.get("user-agent", "")
    accept = request.headers.get("accept", "")
    log.info(
        "passthrough[%d] request headers: ua=%r accept=%r range=%r host=%s",
        rid, user_agent[:160], accept[:160], range_header, request.client,
    )
    if PASSTHROUGH_SEEK_MODE == "bytes" and _range_unsatisfiable(range_header, path, info.duration, estimate_codec):
        return _range_416(path, info.duration, estimate_codec)
    requested_t = t
    npt_t = _parse_npt_seconds(time_seek_range)
    if npt_t is not None:
        t = npt_t
    elif PASSTHROUGH_SEEK_MODE == "bytes":
        byte_t = _seek_from_byte_range(range_header, path, info.duration, estimate_codec)
        if byte_t is not None:
            t = byte_t
    if info.duration > 0:
        t = min(t, max(0.0, info.duration - 0.01))
    media_type = _passthrough_media_type()
    log.info(
        "passthrough[%d] start: %s @ %.2fs from %s container=%s media_type=%s requested_t=%.2fs time_seek=%r range=%r getfeatures=%r transfer=%r",
        rid, path.name, t, request.client, PASSTHROUGH_CONTAINER, media_type,
        requested_t, time_seek_range, range_header, get_content_features, transfer_mode,
    )
    total = _estimated_passthrough_size(path, info.duration, estimate_codec)
    annotate_request(request, total_estimated_size=total)
    byte_range = _parse_byte_range(range_header, total)
    if PASSTHROUGH_SEEK_MODE == "bytes" and _is_tail_probe_range(byte_range):
        assert byte_range is not None
        headers = _passthrough_headers(
            media_type,
            t,
            info.duration,
            path,
            estimate_codec,
            range_header,
            include_length=True,
            backend_verdict=backend_verdict,
        )
        headers["X-Passthrough-Probe-Source"] = "tail-probe-empty"
        headers["X-Passthrough-Seek-Time"] = f"{t:.3f}"
        body = b"\x00" * byte_range.length
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{total}"
        headers["Content-Length"] = str(len(body))
        log.info(
            "passthrough[%d] tail probe ignored: %s range=%s total=%d start_ratio=%.6f",
            rid, path.name, range_header, total, byte_range.start / total if total else 0.0,
        )
        return Response(body, status_code=206, headers=headers, media_type=media_type)
    if PASSTHROUGH_SEEK_MODE == "bytes" and byte_range is not None and byte_range.start > 0:
        probe_key = _probe_cache_key(path, estimate_codec, info.duration)
        deadline = asyncio.get_running_loop().time() + _PREFIX_CACHE_WAIT_SEC
        cached = b""
        while True:
            async with _probe_cache_lock:
                cached = _probe_cache.get(probe_key, b"")
            if byte_range.start < len(cached):
                break
            if byte_range.start >= _PROBE_CACHE_LIMIT or asyncio.get_running_loop().time() >= deadline:
                break
            if int((deadline - asyncio.get_running_loop().time()) * 10) % 10 == 0:
                log.info(
                    "passthrough[%d] prefix cache wait: range=%s start=%d cached=%d",
                    rid, range_header, byte_range.start, len(cached),
                )
            await asyncio.sleep(0.05)
        if byte_range.start < len(cached):
            headers = _passthrough_headers(
                media_type,
                t,
                info.duration,
                path,
                estimate_codec,
                range_header,
                include_length=True,
                backend_verdict=backend_verdict,
            )
            headers["X-Passthrough-Probe-Source"] = "prefix-cache"
            headers["X-Passthrough-Seek-Time"] = f"{t:.3f}"
            if _is_open_range(range_header):
                end = min(byte_range.end, len(cached) - 1)
                body = cached[byte_range.start:end + 1]
                headers["Content-Range"] = f"bytes {byte_range.start}-{end}/{total}"
                headers["Content-Length"] = str(len(body))
                log.info(
                    "passthrough[%d] prefix cache open bounded hit: %s range=%s served=%d-%d cached=%d len=%d",
                    rid, path.name, range_header, byte_range.start, end, len(cached), len(body),
                )
                return Response(body, status_code=206, headers=headers, media_type=media_type)
            end = min(byte_range.end, len(cached) - 1)
            body = cached[byte_range.start:end + 1]
            headers["Content-Range"] = f"bytes {byte_range.start}-{end}/{total}"
            headers["Content-Length"] = str(len(body))
            log.info(
                "passthrough[%d] prefix cache hit: %s range=%s served=%d-%d cached=%d len=%d",
                rid, path.name, range_header, byte_range.start, end, len(cached), len(body),
            )
            return Response(body, status_code=206, headers=headers, media_type=media_type)
        log.info(
            "passthrough[%d] prefix cache miss: %s range=%s cached=%d limit=%d",
            rid, path.name, range_header, len(cached), _PROBE_CACHE_LIMIT,
        )
        if _is_open_range(range_header) and byte_range.start < _PROBE_CACHE_LIMIT:
            log.info(
                "passthrough[%d] prefix cache not ready; refusing probe without starting new stream: %s range=%s cached=%d",
                rid, path.name, range_header, len(cached),
            )
            return Response(
                "prefix cache not ready",
                status_code=503,
                headers={
                    "Retry-After": "1",
                    "Accept-Ranges": "bytes",
                    "X-Passthrough-Probe-Source": "prefix-cache-not-ready",
                },
            )
    if PASSTHROUGH_SEEK_MODE == "bytes" and _is_small_probe_range(byte_range):
        headers = _passthrough_headers(media_type, t, info.duration, path, estimate_codec, range_header, include_length=True, backend_verdict=backend_verdict)
        headers["X-Passthrough-Seek-Time"] = f"{t:.3f}"
        assert byte_range is not None
        probe_len = byte_range.length
        headers["Content-Length"] = str(probe_len)
        if probe_len > 0:
            headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{total}"
        async with _probe_cache_lock:
            cached = _probe_cache.get(_probe_cache_key(path, estimate_codec, info.duration), b"")
        if len(cached) >= probe_len:
            body = cached[:probe_len]
            headers["X-Passthrough-Probe-Source"] = "cache"
        else:
            prefix = b"\x00\x00\x00\x1cftypmp42\x00\x00\x02\x00mp42isomiso6"
            body = (prefix + b"\x00" * max(0, probe_len - len(prefix)))[:probe_len]
            headers["X-Passthrough-Probe-Source"] = "synthetic"
        return Response(body, status_code=206, headers=headers, media_type=media_type)

    slot_token = object()
    owner = (str(path.resolve()), request.client.host if request.client else "")
    preempted = await _take_active_slot(slot_token, who=f"{path.name}@{t:.2f}s", owner=owner)
    if preempted is False:
        log.info("passthrough[%d] return 503 busy", rid)
        return Response("passthrough busy", status_code=503, headers={"Retry-After": "2"})
    if preempted is not None:
        # Close the preempted stream synchronously so its Matter is returned to
        # the pool before we try to acquire one. Without this, when
        # MAX_CONCURRENT=1 the new request would block the event loop in
        # acquire_matter() waiting for a Matter that only the old stream's
        # generator can release, deadlocking the loop.
        await _close_preempted_stream(preempted, f"{path.name}@{t:.2f}s")

    try:
        # Run acquire_matter on a worker thread so the blocking
        # threading.Condition.wait() never freezes the event loop. Bound the
        # wait so a stuck producer cannot pin this request forever; on timeout
        # rollback the slot and return 503.
        acquire_timeout = max(PASSTHROUGH_BUSY_WAIT_SEC, 1.0)
        matter = await asyncio.to_thread(
            acquire_matter, blocking=True, timeout=acquire_timeout
        )
        if matter is None:
            await _release_active_slot(slot_token)
            log.warning(
                "passthrough[%d] return 503 matter pool exhausted after %.1fs",
                rid, acquire_timeout,
            )
            return Response(
                "passthrough busy", status_code=503, headers={"Retry-After": "2"}
            )
        async with _active_lock:
            if slot_token in _active_streams:
                _active_matter[slot_token] = matter
                matter_tracked = True
            else:
                matter_tracked = False
        if not matter_tracked:
            release_matter(matter)
            log.info("passthrough[%d] return 409 preempted before stream", rid)
            return Response("passthrough preempted", status_code=409, headers={"Retry-After": "1"})
        try:
            stream, stream_backend, stream_verdict = _select_passthrough_stream(path, t, matter)
        except BaseException:
            raise
        if not await _replace_active_slot(slot_token, stream, close_on_failure=stream):
            log.info("passthrough[%d] return 409 preempted before stream", rid)
            return Response("passthrough preempted", status_code=409, headers={"Retry-After": "1"})
    except Exception:
        await _release_active_slot(slot_token)
        raise

    selected_codec = PYNV_OUTPUT_CODEC
    if selected_codec != estimate_codec:
        log.info(
            "passthrough estimate codec changed after backend selection: %s -> %s backend=%s verdict=%s",
            estimate_codec, selected_codec, stream_backend, stream_verdict,
        )
        estimate_codec = selected_codec
        total = _estimated_passthrough_size(path, info.duration, estimate_codec)
        byte_range = _parse_byte_range(range_header, total)

    headers = _passthrough_headers(media_type, t, info.duration, path, estimate_codec, range_header, include_length=True, backend_verdict=stream_verdict)
    headers["X-Passthrough-Seek-Time"] = f"{t:.3f}"
    headers["X-Passthrough-Backend"] = stream_backend
    headers["X-Passthrough-Backend-Verdict"] = stream_verdict
    status_code = 206 if PASSTHROUGH_SEEK_MODE == "bytes" and range_header and not _is_zero_open_range(range_header, byte_range) else 200
    if status_code == 200:
        headers.pop("Content-Range", None)

    content_length = int(headers.get("Content-Length") or "0")
    probe_key = _probe_cache_key(path, estimate_codec, info.duration)
    cache_probe_prefix = True
    pad_to_declared_length = True
    log.info(
        "passthrough[%d] response: status=%d backend=%s verdict=%s codec=%s content_length=%d byte_range=%s headers_range=%r",
        rid, status_code, stream_backend, stream_verdict, estimate_codec, content_length, byte_range, headers.get("Content-Range"),
    )

    async def gen():
        nonlocal cache_probe_prefix, pad_to_declared_length
        sent = 0
        probe_prefix = bytearray()
        probe_prefix_limit = (
            _cache_prefix_limit()
            if PASSTHROUGH_SEEK_MODE == "bytes" and (byte_range is None or byte_range.start == 0)
            else 0
        )
        probe_prefix_flushed_len = 0
        first_chunk = True
        next_progress = 1024 * 1024
        disconnect_task: asyncio.Task | None = None

        async def disconnect_watchdog():
            while True:
                await asyncio.sleep(0.25)
                try:
                    disconnected = await request.is_disconnected()
                except Exception as e:
                    log.info("passthrough[%d] disconnect watchdog stopped: %s", rid, e)
                    return
                if disconnected:
                    log.info("passthrough[%d] disconnect watchdog closing stream", rid)
                    await asyncio.to_thread(stream.close)
                    return

        try:
            disconnect_task = asyncio.create_task(disconnect_watchdog())
            async for chunk in stream.iter_bytes():
                if content_length > 0:
                    remaining = content_length - sent
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                if (
                    probe_prefix_limit > 0
                    and len(probe_prefix) < probe_prefix_limit
                ):
                    need = probe_prefix_limit - len(probe_prefix)
                    probe_prefix.extend(chunk[:need])
                    if len(probe_prefix) >= probe_prefix_limit:
                        async with _probe_cache_lock:
                            _set_probe_cache_locked(probe_key, bytes(probe_prefix))
                        probe_prefix_flushed_len = len(probe_prefix)
                sent += len(chunk)
                if first_chunk:
                    first_chunk = False
                    log.info("passthrough[%d] first chunk: len=%d sent=%d stream_bytes=%d", rid, len(chunk), sent, getattr(stream, "bytes_emitted", -1))
                if sent >= next_progress:
                    log.info("passthrough[%d] progress: sent=%d stream_bytes=%d frames=%d cache=%d", rid, sent, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1), len(probe_prefix))
                    while next_progress <= sent:
                        next_progress += 1024 * 1024
                yield chunk

            log.info("passthrough[%d] stream loop ended: sent=%d stream_bytes=%d frames=%d startup_error=%r", rid, sent, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1), getattr(stream, "startup_error", None))
            if (
                isinstance(stream, PyNvPassthroughStream)
                and stream.bytes_emitted == 0
                and stream.startup_error
            ):
                log.warning("PyNv startup failed before first byte, fallback to FFmpeg: %s", stream.startup_error)
                fallback = PassthroughStream(path, t, matter)
                fallback_codec = PYNV_OUTPUT_CODEC
                fallback_probe_key = _probe_cache_key(path, fallback_codec, info.duration)
                fallback_probe_prefix = bytearray()
                fallback_content_length = _estimated_passthrough_size(path, info.duration, fallback_codec)
                try:
                    async for chunk in fallback.iter_bytes():
                        if fallback_content_length > 0:
                            remaining = fallback_content_length - sent
                            if remaining <= 0:
                                break
                            if len(chunk) > remaining:
                                chunk = chunk[:remaining]
                        if (
                            PASSTHROUGH_SEEK_MODE == "bytes"
                            and (byte_range is None or byte_range.start == 0)
                            and len(fallback_probe_prefix) < _PROBE_CACHE_LIMIT
                        ):
                            need = _PROBE_CACHE_LIMIT - len(fallback_probe_prefix)
                            fallback_probe_prefix.extend(chunk[:need])
                        sent += len(chunk)
                        yield chunk
                finally:
                    fallback.close()
                    cache_probe_prefix = False
                    if fallback_probe_prefix:
                        async with _probe_cache_lock:
                            _set_probe_cache_locked(fallback_probe_key, bytes(fallback_probe_prefix))
                    if fallback.bytes_emitted > 0:
                        stream.bytes_emitted = fallback.bytes_emitted
                        stream.frames_produced = fallback.frames_produced
                        stream.output_fps = fallback.output_fps

            if (
                PASSTHROUGH_PAD_TO_LENGTH
                and pad_to_declared_length
                and content_length > 0
                and sent < content_length
            ):
                log.info("passthrough[%d] padding begin: sent=%d content_length=%d", rid, sent, content_length)
                pad = b"\x00" * min(64 * 1024, content_length - sent)
                while sent < content_length:
                    chunk = pad[: min(len(pad), content_length - sent)]
                    sent += len(chunk)
                    yield chunk
                log.info("passthrough[%d] padding end: sent=%d", rid, sent)
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
            log.info("passthrough[%d] finally begin: sent=%d cache_probe=%s pad=%s stream_bytes=%d frames=%d", rid, sent, cache_probe_prefix, pad_to_declared_length, getattr(stream, "bytes_emitted", -1), getattr(stream, "frames_produced", -1))
            _close_stream_if_possible(stream)
            _release_active_slot_nowait(stream)
            if cache_probe_prefix and probe_prefix and len(probe_prefix) != probe_prefix_flushed_len:
                async with _probe_cache_lock:
                    _set_probe_cache_locked(probe_key, bytes(probe_prefix))
            if stream.bytes_emitted > 0 and (content_length <= 0 or sent >= content_length):
                if stream.frames_produced > 0 and stream.output_fps > 0:
                    elapsed_media = stream.frames_produced / stream.output_fps
                else:
                    elapsed_media = max(0.001, info.duration - t)
                record_actual_bps(
                    path,
                    estimate_codec,
                    None,
                    stream.bytes_emitted * 8 / elapsed_media,
                    elapsed_media,
                )
            log.info("passthrough[%d] finally done: sent=%d", rid, sent)

    return StreamingResponse(gen(), status_code=status_code, headers=headers, media_type=media_type)
