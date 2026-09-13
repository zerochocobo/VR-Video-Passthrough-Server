"""Media HTTP routes.

- /media/{name}: serve the source file with standard HTTP Range support.
- /passthrough/{name}: pseudo-VOD passthrough with byte/time seek mapping.
- /passthrough_live/{name}: MPEG-TS live passthrough for clients that dislike pseudo-VOD byte seeking.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import itertools
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Header, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

import config
from config import (
    HTTP_PORT,
    DLNA_IMAGE_ENABLED,
    IMAGE_EXTS,
    IMAGE_MIME_BY_EXT,
    LAN_IP,
    FACE_BEAUTY_LIVE_FIRST_CHUNK_TIMEOUT_SEC,
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
    RTX_VSR_REALTIME_ENABLED,
    RTX_VSR_TARGET_HEIGHT,
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
    PASSTHROUGH_SEEK_VMP4,
    PASSTHROUGH_SEEK_VMP4_BACKEND,
    PASSTHROUGH_SEEK_VMP4_BUILD_BITRATE,
    PASSTHROUGH_SEEK_VMP4_BUILD_ENGINE,
    PASSTHROUGH_SEEK_VMP4_BUILD_MAX_ACTIVE,
    PASSTHROUGH_SEEK_VMP4_BUILD_MISSING,
    PASSTHROUGH_SEEK_VMP4_BUILD_MODES,
    PASSTHROUGH_SEEK_VMP4_BUILD_PRESET,
    PASSTHROUGH_SEEK_VMP4_BUILD_TWO_DVR_BITRATE,
    PASSTHROUGH_SEEK_VMP4_BUILD_TWO_DVR_MAX_SIDE,
    PASSTHROUGH_SEEK_VMP4_BUILD_TWO_DVR_PROVIDER,
    PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER,
    PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY,
    PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT,
    PASSTHROUGH_SEEK_VMP4_SLOT_MAX_SAMPLE_BYTES,
    PASSTHROUGH_SEEK_VMP4_SLOT_MATTER_TIMEOUT,
    PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_BASE,
    PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_MAX,
    PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES,
    PASSTHROUGH_SEEK_VMP4_FRAMES_SOURCE_BUDGET,
    seek_budget_scale,
    seek_declared_total_bytes,
    seek_output_fps,
    PASSTHROUGH_SEEK_VMP4_FRAMES_BUDGET_FLATTEN,
    PASSTHROUGH_SEEK_VMP4_FRAMES_FLOOR_BYTES,
    seek_frame_floor_bytes,
    PASSTHROUGH_SEEK_VMP4_FRAMES_PROBE_BYTES,
    PASSTHROUGH_SEEK_VMP4_FRAMES_IDR_WEIGHT,
    PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_FPS,
    PASSTHROUGH_SEEK_VMP4_FRAMES_PREBUFFER_FRAMES,
    PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_SPAN_BYTES,
    PASSTHROUGH_SEEK_VMP4_FRAMES_MIN_SPAN_BYTES,
    PASSTHROUGH_SEEK_VMP4_FRAMES_RANGED_WAIT,
    PASSTHROUGH_SEEK_VMP4_FRAMES_BODY_FRAME_WAIT,
    PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_ACTIVE,
    PASSTHROUGH_SEEK_VMP4_FRAMES_READY_WAIT,
    RUNTIME_CACHE_DIR,
    SI_PROGRESSIVE_ENABLED,
    TWO_DVR_HOLE_FILL,
    TWO_DVR_MODEL,
    TWO_DVR_STRENGTH,
    DEBUG_LOGS,
    DECODE_MAX_SIDE,
    LIVE_REQUEST_HEADER_DUMP,
    LIGHT_MATCH_FLUSH_QUEUES,
    MEDIA_LIBRARY,
    ROOT,
    USE_PYNV,
    VIDEO_EXTS,
)
from dlna.profiles import passthrough_dlna_pn, passthrough_frame_rate
from media_library import safe_resolve_path
from http_app.si_stream import DEFAULT_CHUNK_SIZE, get_si_stream_service, iter_si_mpegts, parse_range_header
from pipeline.ffmpeg_io import FFMPEG, probe_cached
from pipeline.passthrough_vmp4_slot import (
    PassthroughVmp4SlotLayout,
    Vmp4SlotCacheStatus,
    Vmp4SlotLayoutError,
    Vmp4SlotOutputTemplate,
    build_passthrough_vmp4_slot_layout,
    ensure_vmp4_slot_manifest,
    iter_vmp4_slot_range,
    load_vmp4_slot_output_template,
    update_vmp4_slot_manifest_slot_state,
    vmp4_slot_index_for_offset,
    write_vmp4_slot_hevc_annexb_payload,
    hevc_annexb_to_length_prefixed_sample,
    split_hevc_annexb_access_units,
)
from pipeline.source_budget_plan import (
    SourceBudgetError,
    SourceBudgetPlan,
    build_source_budget_plan,
)
from pipeline.passthrough_vmp4_frames import (
    PassthroughVmp4FramesLayout,
    build_passthrough_vmp4_frames_layout,
    iter_vmp4_frames_range,
    vmp4_frame_index_for_offset,
)
from pipeline.si_virtual_mp4 import build_progressive_si_virtual_mp4, iter_virtual_range, read_media_sample_table
from pipeline.matting import acquire_matter, release_matter
from pipeline.stream import PassthroughStream
from pipeline.pynv_stream import SEEK_FRAME_MODES
from utils.rtx_vsr import seek_supported_target
from pipeline.pynv_stream import (
    PYNV_BACKEND_LABEL,
    PYNV_OUTPUT_CODEC,
    PyNvPassthroughStream,
    build_passthrough_hevc_annexb_gop,
)
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
from utils.runtime_dll_paths import apply_runtime_dll_paths
from utils.runtime_settings import get_light_match
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.mkv_cues import probe_mkv_cues
from utils.offline_outputs import has_offline_two_dvr_output, is_internal_intermediate_name, matches_offline_output_for_source, matches_offline_two_dvr_output_for_source
from utils.subtitles import find_external_subtitles, is_subtitle_path, subtitle_mime
from utils.video_metadata import probe_video_metadata, select_backend
from utils.vr_naming import has_vr_filename_marker, is_half_equirectangular_source, offline_passthrough_stem, two_dvr_stem

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
    if PASSTHROUGH_SEEK_VMP4:
        return "mp4"
    return PASSTHROUGH_SEEK_CONTAINER if PASSTHROUGH_SEEK_CONTAINER in {"mpegts", "mp4"} else "mpegts"


def _seek_media_type(container: str | None = None) -> str:
    return "video/mp4" if (container or _seek_container()) == "mp4" else "video/MP2T"


def _seek_dlna_pn(container: str | None = None) -> str:
    return "HEVC_MP4_MAIN" if (container or _seek_container()) == "mp4" else "HEVC_TS_NA_ISO"


# The output mode rides in the path, not only in the query: players re-issue a
# seek URL for their range requests and some drop the query when they do. A
# dropped `mode=superres` then fell back to another mode and played the wrong
# picture, so the route carries `<key>.<mode>.seek.<ext>` and the query stays
# as a hint for links that predate this.
_SEEK_ROUTE_MODES = ("green", "alpha", "superres", "dlss5")


def _split_seek_route_name(name: str) -> tuple[str, str | None, str | None]:
    decoded = unquote(name)
    lower = decoded.lower()
    for suffix, container in _SEEK_ROUTE_SUFFIXES:
        if lower.endswith(suffix):
            key = decoded[: -len(suffix)]
            key_lower = key.lower()
            for mode in _SEEK_ROUTE_MODES:
                tag = f".{mode}"
                if key_lower.endswith(tag):
                    return key[: -len(tag)], container, mode
            return key, container, None
    return decoded, None, None


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
        "provider_kind": (
            "trt"
            if config.ONNX_PROVIDERS and config.ONNX_PROVIDERS[0] == "TensorrtExecutionProvider"
            else "cuda"
            if config.ONNX_PROVIDERS and config.ONNX_PROVIDERS[0] == "CUDAExecutionProvider"
            else "cpu"
        ),
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
    p = safe_resolve_path(p)
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
    p = safe_resolve_path(p)
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


def _safe_seek_video_path(name: str) -> tuple[Path, str | None, str | None]:
    key, route_container, route_mode = _split_seek_route_name(name)
    return _safe_video_path_from_key(key), route_container, route_mode


def _safe_subtitle_path(name: str) -> Path:
    name = unquote(name)
    p = MEDIA_LIBRARY.key_to_path(name)
    if p is None:
        raise HTTPException(403, "forbidden")
    p = safe_resolve_path(p)
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
    config, duck_key = service.resolve_stream(path)
    si_wav = service.has_si_source(path)
    if not config.enabled or si_wav is None:
        raise HTTPException(404, "SI stream not available")
    start_time = max(0.0, float(t or 0.0))
    log.info(
        "si_live start: path=%s t=%.3f si=%s duck=%s ua=%r",
        path, start_time, si_wav, duck_key, user_agent,
    )
    headers = {
        "Accept-Ranges": "none",
        "X-SI-Enabled": "1",
        "X-SI-Transport": "mpegts-live",
        "X-SI-Dub-Mode": "1" if duck_key is not None else "0",
        "contentFeatures.dlna.org": _si_live_content_features(),
        "transferMode.dlna.org": "Streaming",
    }
    return StreamingResponse(
        iter_si_mpegts(path, si_wav, config, start_time, duck_key=duck_key, chunk_size=DEFAULT_CHUNK_SIZE),
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
    return f"{safe_resolve_path(path)}|{codec}|{total}"


def _seek_declared_size_key(path: Path, codec: str, duration: float, client_host: str, container: str | None = None) -> tuple:
    try:
        st = path.stat()
        stat_part = (st.st_size, st.st_mtime_ns)
    except OSError:
        stat_part = (0, 0)
    return (
        str(safe_resolve_path(path)),
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


def _seek_vmp4_enabled(container: str | None = None) -> bool:
    return bool(PASSTHROUGH_SEEK_VMP4 and (container or _seek_container()) == "mp4")


def _seek_vmp4_backend() -> str:
    raw = str(PASSTHROUGH_SEEK_VMP4_BACKEND or "cache_file").strip().lower().replace("-", "_")
    return raw if raw in {"cache_file", "slot", "slot_frames"} else "cache_file"


def _source_declared_size(path: Path) -> int:
    try:
        return max(0, int(Path(path).stat().st_size))
    except OSError:
        return 0


def _seek_declared_total(
    path: Path,
    duration: float,
    codec: str,
    client_host: str = "",
    container: str | None = None,
) -> int:
    # Deliberately NOT scaled by the budget rule: this is shared by every seek
    # backend, including the prebuilt-cache path that pads a real file up to the
    # source size. Only the frames layout stretches the file, and that path
    # serves layout.total_size, which carries the scale already.
    if _seek_vmp4_enabled(container):
        return _source_declared_size(path)
    return _estimated_seek_passthrough_size(path, duration, codec, client_host, container)


def _seek_probe_cache_key(path: Path, codec: str, duration: float, total: int, container: str | None = None) -> str:
    flavor = "vmp4" if _seek_vmp4_enabled(container) else "seek"
    return f"{flavor}|{container or _seek_container()}|{safe_resolve_path(path)}|{codec}|{round(float(duration or 0.0), 3)}|{total}"


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
    # config owns the rule: the budget scale depends on it and the DLNA layer has
    # to arrive at the same number this path will.
    return seek_output_fps(float(getattr(info, "fps", 0.0) or 0.0))


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
    prefix = "seek-vmp4" if _seek_vmp4_enabled(container) else "seek"
    headers["X-Passthrough-Mode"] = f"{prefix}-{container or _seek_container()}-{output_mode}"
    if mapped is not None:
        headers["X-Passthrough-Seek-Ratio"] = f"{mapped.ratio:.6f}"
        headers["X-Passthrough-Seek-Raw-Time"] = f"{mapped.time_sec:.3f}"
        headers["X-Passthrough-Seek-Gop"] = f"{mapped.gop_seconds:.3f}"


def _apply_seek_vmp4_headers(
    headers: dict[str, str],
    *,
    path: Path,
    duration: float,
    total: int,
    container: str,
) -> None:
    if not _seek_vmp4_enabled(container):
        return
    headers["X-Passthrough-VMP4"] = "1"
    headers["X-Passthrough-VMP4-Phase"] = "source-size-harness"
    headers["X-Passthrough-VMP4-Layout-Size"] = str(max(0, int(total)))
    headers["X-Passthrough-VMP4-Source-Size"] = str(_source_declared_size(path))
    headers["X-Passthrough-VMP4-Duration"] = f"{max(0.0, float(duration or 0.0)):.3f}"


def _header_safe_text(value: object) -> str:
    return quote(str(value), safe="._-+~")


@dataclass
class _Vmp4CacheBuild:
    key: str
    source: Path
    mode: str
    target: Path
    work_dir: Path
    temp_target: Path
    log_path: Path
    cmd: tuple[str, ...]
    state: str = "queued"
    pid: int = 0
    returncode: int | None = None
    reason: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass(frozen=True)
class _Vmp4CacheBuildView:
    state: str
    target: Path | None
    log_path: Path | None = None
    pid: int = 0
    returncode: int | None = None
    reason: str = ""


_VMP4_BUILD_LOG_DIR = ROOT / "debug_output" / "vmp4_cache_build"
_VMP4_BUILD_STATES_KEEP = 128
_vmp4_build_lock = threading.RLock()
_vmp4_builds: dict[str, _Vmp4CacheBuild] = {}
_vmp4_active_builds = 0


@dataclass
class _Vmp4SlotBuild:
    key: str
    slot_index: int
    output_mode: str
    cache_dir: Path
    payload_path: Path
    manifest_path: Path
    state: str = "queued"
    payload_size: int = 0
    reason: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: int = 0
    permanent: bool = False
    next_retry_at: float = 0.0


@dataclass(frozen=True)
class _Vmp4SlotBuildView:
    state: str
    payload_path: Path | None = None
    payload_size: int = 0
    reason: str = ""


@dataclass(frozen=True)
class _Vmp4SlotReadyWait:
    status: Vmp4SlotCacheStatus
    build: _Vmp4SlotBuildView
    slot_state: str
    ready: bool
    waited_sec: float


_vmp4_slot_build_lock = threading.RLock()
_vmp4_slot_builds: dict[str, _Vmp4SlotBuild] = {}
_VMP4_SLOT_LAYOUT_CACHE_LIMIT = 64
_vmp4_slot_layout_cache_lock = threading.RLock()
_vmp4_slot_layout_cache: dict[tuple, PassthroughVmp4SlotLayout] = {}


def _vmp4_cache_candidate_ignored(candidate: Path) -> bool:
    return is_internal_intermediate_name(candidate.name)


def _vmp4_mode_candidate_stems(path: Path, output_mode: str, info) -> list[str]:
    width = int(getattr(info, "width", 0) or 0)
    height = int(getattr(info, "height", 0) or 0)
    stems: list[str] = []
    mode = (output_mode or "green").lower()
    if mode == "alpha":
        stems.append(offline_passthrough_stem(path.stem, "alpha", width, height))
    elif mode == "two_dvr":
        stems.append(two_dvr_stem(path.stem))
    else:
        stems.append(offline_passthrough_stem(path.stem, "green", width, height))
        stems.append(f"{path.stem}_passthrough")
    return list(dict.fromkeys(stems))


def _vmp4_expected_cache_names(path: Path, output_mode: str, info) -> list[str]:
    return [f"{stem}.mp4" for stem in _vmp4_mode_candidate_stems(path, output_mode, info)]


def _vmp4_expected_cache_target(path: Path, output_mode: str, info) -> Path | None:
    names = _vmp4_expected_cache_names(path, output_mode, info)
    return path.with_name(names[0]) if names else None


def _vmp4_mode_candidate(path: Path, output_mode: str, info) -> Path | None:
    mode = (output_mode or "green").lower()
    seen: set[Path] = set()
    for stem in _vmp4_mode_candidate_stems(path, output_mode, info):
        candidate = path.with_name(f"{stem}.mp4")
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and candidate.resolve() != path.resolve():
            return candidate
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        siblings = []
    for candidate in siblings:
        if candidate == path or candidate.suffix.lower() != ".mp4":
            continue
        if _vmp4_cache_candidate_ignored(candidate):
            continue
        stem_l = candidate.stem.lower()
        if mode == "two_dvr":
            if matches_offline_two_dvr_output_for_source(path, candidate):
                return candidate
        elif mode == "alpha":
            if matches_offline_output_for_source(path, candidate) and "alpha" in stem_l:
                return candidate
        else:
            if matches_offline_output_for_source(path, candidate) and "alpha" not in stem_l:
                return candidate
    return None


def _vmp4_build_view(build: _Vmp4CacheBuild) -> _Vmp4CacheBuildView:
    return _Vmp4CacheBuildView(
        state=build.state,
        target=build.target,
        log_path=build.log_path,
        pid=build.pid,
        returncode=build.returncode,
        reason=build.reason,
    )


def _vmp4_disabled_build_view(reason: str, target: Path | None = None) -> _Vmp4CacheBuildView:
    return _Vmp4CacheBuildView(state="disabled", target=target, reason=reason)


def _vmp4_python_command(*args: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    python = venv_python if venv_python.exists() else Path(sys.executable)
    return [str(python), str(ROOT / "main.py"), *args]


def _vmp4_build_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    # Match the offline GUI path: do not apply the realtime decode-size cap to
    # full-file cache generation.
    env["PT_DECODE_MAX_SIDE"] = "0"
    apply_runtime_dll_paths(env)
    return env


def _vmp4_build_paths(path: Path, output_mode: str, target: Path, key: str) -> tuple[Path, Path, Path]:
    digest = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:12]
    safe_mode = re.sub(r"[^a-z0-9_]+", "_", (output_mode or "green").lower())
    work_dir = path.parent / f".{target.stem}.vmp4build_{safe_mode}_{digest}"
    temp_target = work_dir / target.name
    log_path = _VMP4_BUILD_LOG_DIR / f"{path.stem}_{safe_mode}_{digest}.log"
    return work_dir, temp_target, log_path


def _vmp4_budget_bitrate(info, total: int, output_mode: str, observed_cache_size: int = 0) -> str:
    duration = max(0.0, float(getattr(info, "duration", 0.0) or 0.0))
    budget_bytes = max(0, int(total))
    if duration <= 0.0 or budget_bytes <= 0:
        return "12000000"
    raw_budget_bps = budget_bytes * 8.0 / duration
    mode = (output_mode or "green").lower()
    safety = 0.84 if mode == "alpha" else 0.90
    if observed_cache_size > budget_bytes:
        safety = min(safety, 0.90 * (budget_bytes / max(1, int(observed_cache_size))))
    target_bps = int(raw_budget_bps * safety)
    return str(max(1_000_000, target_bps))


def _vmp4_build_bitrate(info, total: int, output_mode: str, observed_cache_size: int = 0) -> str:
    raw = str(PASSTHROUGH_SEEK_VMP4_BUILD_BITRATE or "budget").strip()
    if raw.lower() in {"", "auto", "budget"}:
        return _vmp4_budget_bitrate(info, total, output_mode, observed_cache_size)
    return raw


def _vmp4_two_dvr_build_bitrate(info, total: int, output_mode: str, observed_cache_size: int = 0) -> str:
    raw = str(PASSTHROUGH_SEEK_VMP4_BUILD_TWO_DVR_BITRATE or "budget").strip()
    if raw.lower() in {"", "auto", "budget"}:
        return _vmp4_budget_bitrate(info, total, output_mode, observed_cache_size)
    return raw


def _vmp4_build_command(
    path: Path,
    output_mode: str,
    target: Path,
    temp_target: Path,
    work_dir: Path,
    bitrate: str,
) -> list[str]:
    mode = (output_mode or "green").lower()
    if mode in SEEK_FRAME_MODES:
        return _vmp4_python_command(
            "offline",
            "single",
            str(path),
            "--mode",
            mode,
            "--engine",
            PASSTHROUGH_SEEK_VMP4_BUILD_ENGINE,
            "--out",
            str(temp_target),
            "--skip-frames",
            "0",
            "--bitrate",
            str(bitrate),
            "--preset",
            str(PASSTHROUGH_SEEK_VMP4_BUILD_PRESET),
        )
    if mode == "two_dvr":
        return _vmp4_python_command(
            "two_dvr",
            "single",
            str(path),
            "--out-dir",
            str(work_dir),
            "--projection",
            "flat3d",
            "--model",
            str(TWO_DVR_MODEL),
            "--hole-fill",
            str(TWO_DVR_HOLE_FILL),
            "--strength",
            f"{float(TWO_DVR_STRENGTH):g}",
            "--max-side",
            str(PASSTHROUGH_SEEK_VMP4_BUILD_TWO_DVR_MAX_SIDE),
            "--provider",
            str(PASSTHROUGH_SEEK_VMP4_BUILD_TWO_DVR_PROVIDER),
            "--bitrate",
            str(bitrate),
        )
    raise ValueError(f"unsupported VMP4 cache build mode: {output_mode!r}")


def _vmp4_build_key(path: Path, output_mode: str, target: Path, bitrate: str, reason: str = "") -> str:
    try:
        st = path.stat()
        source_fingerprint = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        source_fingerprint = "missing"
    return (
        f"{_media_key_path(path)}|{(output_mode or 'green').lower()}|"
        f"{target.name}|{source_fingerprint}|bitrate={bitrate}|reason={reason}"
    )


def _vmp4_prune_builds_locked() -> None:
    if len(_vmp4_builds) <= _VMP4_BUILD_STATES_KEEP:
        return
    removable = [
        (build.finished_at or build.started_at or 0.0, key)
        for key, build in _vmp4_builds.items()
        if build.state in {"ready", "failed"}
    ]
    for _, key in sorted(removable)[: max(0, len(_vmp4_builds) - _VMP4_BUILD_STATES_KEEP)]:
        _vmp4_builds.pop(key, None)


def _vmp4_start_build_locked(build: _Vmp4CacheBuild) -> None:
    global _vmp4_active_builds
    if build.state != "queued":
        return
    build.state = "starting"
    build.started_at = time.time()
    _vmp4_active_builds += 1
    thread = threading.Thread(
        target=_vmp4_build_worker,
        args=(build.key,),
        name=f"vmp4-cache-{build.mode}",
        daemon=True,
    )
    thread.start()


def _vmp4_start_next_build_locked() -> None:
    if _vmp4_active_builds >= max(1, int(PASSTHROUGH_SEEK_VMP4_BUILD_MAX_ACTIVE)):
        return
    for build in _vmp4_builds.values():
        if build.state == "queued":
            _vmp4_start_build_locked(build)
            return


def _vmp4_build_worker(key: str) -> None:
    global _vmp4_active_builds
    with _vmp4_build_lock:
        build = _vmp4_builds.get(key)
        if build is None:
            _vmp4_active_builds = max(0, _vmp4_active_builds - 1)
            return
        build.state = "running"
    rc: int | None = None
    failure = ""
    try:
        shutil.rmtree(build.work_dir, ignore_errors=True)
        build.work_dir.mkdir(parents=True, exist_ok=True)
        build.log_path.parent.mkdir(parents=True, exist_ok=True)
        with build.log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
            log_fh.write(f"[vmp4-cache] source={build.source}\n")
            log_fh.write(f"[vmp4-cache] mode={build.mode} target={build.target}\n")
            log_fh.write("[vmp4-cache] command=" + subprocess.list2cmdline(list(build.cmd)) + "\n")
            log_fh.flush()
            proc = subprocess.Popen(
                list(build.cmd),
                cwd=str(ROOT),
                env=_vmp4_build_env(),
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                **hidden_subprocess_kwargs(),
            )
            with _vmp4_build_lock:
                current = _vmp4_builds.get(key)
                if current is not None:
                    current.pid = int(proc.pid or 0)
            log.info(
                "passthrough_seek VMP4 cache build started: mode=%s pid=%s source=%s target=%s log=%s cmd=%s",
                build.mode,
                proc.pid,
                build.source.name,
                build.target.name,
                build.log_path,
                subprocess.list2cmdline(list(build.cmd)),
            )
            rc = int(proc.wait())
            log_fh.write(f"\n[vmp4-cache] rc={rc}\n")
        if rc == 0 and build.temp_target.is_file():
            build.target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(build.temp_target, build.target)
            shutil.rmtree(build.work_dir, ignore_errors=True)
        elif rc == 0:
            failure = f"output missing: {build.temp_target}"
        else:
            failure = f"process exited rc={rc}"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        with _vmp4_build_lock:
            current = _vmp4_builds.get(key)
            if current is not None:
                current.returncode = rc
                current.finished_at = time.time()
                if failure:
                    current.state = "failed"
                    current.reason = failure
                    log.warning(
                        "passthrough_seek VMP4 cache build failed: mode=%s source=%s target=%s rc=%s reason=%s log=%s",
                        current.mode,
                        current.source.name,
                        current.target.name,
                        rc,
                        failure,
                        current.log_path,
                    )
                else:
                    current.state = "ready"
                    current.reason = ""
                    log.info(
                        "passthrough_seek VMP4 cache build ready: mode=%s source=%s target=%s log=%s",
                        current.mode,
                        current.source.name,
                        current.target.name,
                        current.log_path,
                    )
            _vmp4_active_builds = max(0, _vmp4_active_builds - 1)
            _vmp4_prune_builds_locked()
            _vmp4_start_next_build_locked()


def _vmp4_schedule_cache_build(
    path: Path,
    info,
    output_mode: str,
    target: Path | None,
    *,
    total: int,
    force: bool = False,
    reason: str = "",
    observed_cache_size: int = 0,
) -> _Vmp4CacheBuildView:
    mode = (output_mode or "green").lower()
    if target is None:
        return _vmp4_disabled_build_view("target-missing", target)
    if not PASSTHROUGH_SEEK_VMP4_BUILD_MISSING:
        return _vmp4_disabled_build_view("build-disabled", target)
    if mode not in set(PASSTHROUGH_SEEK_VMP4_BUILD_MODES):
        return _vmp4_disabled_build_view("mode-disabled", target)
    if target.is_file() and not force:
        return _Vmp4CacheBuildView(state="ready", target=target)
    bitrate = (
        _vmp4_two_dvr_build_bitrate(info, total, mode, observed_cache_size)
        if mode == "two_dvr"
        else _vmp4_build_bitrate(info, total, mode, observed_cache_size)
    )
    key = _vmp4_build_key(path, mode, target, bitrate, reason if force else "")
    work_dir, temp_target, log_path = _vmp4_build_paths(path, mode, target, key)
    try:
        cmd = tuple(_vmp4_build_command(path, mode, target, temp_target, work_dir, bitrate))
    except Exception as exc:
        return _vmp4_disabled_build_view(f"command-error:{type(exc).__name__}", target)
    with _vmp4_build_lock:
        build = _vmp4_builds.get(key)
        if build is None:
            build = _Vmp4CacheBuild(
                key=key,
                source=path,
                mode=mode,
                target=target,
                work_dir=work_dir,
                temp_target=temp_target,
                log_path=log_path,
                cmd=cmd,
            )
            _vmp4_builds[key] = build
            log.info(
                "passthrough_seek VMP4 cache build queued: mode=%s source=%s target=%s log=%s",
                mode,
                path.name,
                target.name,
                log_path,
            )
            _vmp4_start_next_build_locked()
        return _vmp4_build_view(build)


def _apply_vmp4_build_headers(headers: dict[str, str], build: _Vmp4CacheBuildView) -> None:
    headers["X-Passthrough-VMP4-Build"] = build.state
    if build.target is not None:
        headers["X-Passthrough-VMP4-Build-Target"] = _header_safe_text(build.target.name)
    if build.log_path is not None:
        headers["X-Passthrough-VMP4-Build-Log"] = _header_safe_text(build.log_path.name)
    if build.pid:
        headers["X-Passthrough-VMP4-Build-Pid"] = str(build.pid)
    if build.returncode is not None:
        headers["X-Passthrough-VMP4-Build-RC"] = str(build.returncode)
    if build.reason:
        headers["X-Passthrough-VMP4-Build-Reason"] = _header_safe_text(build.reason)


def _vmp4_slot_build_view(build: _Vmp4SlotBuild) -> _Vmp4SlotBuildView:
    return _Vmp4SlotBuildView(
        state=build.state,
        payload_path=build.payload_path,
        payload_size=build.payload_size,
        reason=build.reason,
    )


def _vmp4_slot_disabled_build_view(reason: str) -> _Vmp4SlotBuildView:
    return _Vmp4SlotBuildView(state="disabled", reason=reason)


def _vmp4_slot_build_key(status: Vmp4SlotCacheStatus, slot_index: int) -> str:
    return f"{status.digest}:slot={int(slot_index)}"


def _vmp4_slot_start_build_locked(build: _Vmp4SlotBuild, layout: PassthroughVmp4SlotLayout) -> None:
    if build.state not in {"queued", "failed"}:
        return
    build.state = "running"
    build.started_at = time.time()
    update_vmp4_slot_manifest_slot_state(build.manifest_path, build.slot_index, "building")
    thread = threading.Thread(
        target=_vmp4_slot_build_worker,
        args=(build.key, layout),
        name=f"vmp4-slot-{build.slot_index}",
        daemon=True,
    )
    thread.start()


def _vmp4_slot_schedule_placeholder_build(
    layout: PassthroughVmp4SlotLayout,
    status: Vmp4SlotCacheStatus,
    slot_index: int,
    output_mode: str = "green",
) -> _Vmp4SlotBuildView:
    if not PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER:
        return _vmp4_slot_disabled_build_view("slot-build-disabled")
    slot = status.slot(slot_index)
    if slot is None:
        return _vmp4_slot_disabled_build_view("slot-index-missing")
    if slot.state == "ready":
        return _Vmp4SlotBuildView(
            state="ready",
            payload_path=slot.payload_path,
            payload_size=slot.payload_size,
        )
    payload_path = status.cache_dir / f"slot_{int(slot_index):06d}.bin"
    key = _vmp4_slot_build_key(status, slot_index)
    with _vmp4_slot_build_lock:
        build = _vmp4_slot_builds.get(key)
        if build is None:
            build = _Vmp4SlotBuild(
                key=key,
                slot_index=int(slot_index),
                output_mode=(output_mode or "green").lower(),
                cache_dir=status.cache_dir,
                payload_path=payload_path,
                manifest_path=status.manifest_path,
            )
            _vmp4_slot_builds[key] = build
            _vmp4_slot_start_build_locked(build, layout)
        elif build.state == "failed" and not build.permanent and time.time() >= build.next_retry_at:
            # Transient failure (e.g. matter timeout, GPU hiccup): retry with the
            # accumulated backoff. Permanent failures (oversize/unsupported) are
            # never auto-retried so a player's Retry-After loop cannot thrash the
            # GPU with a slot that can never fit.
            build.state = "queued"
            build.reason = ""
            _vmp4_slot_start_build_locked(build, layout)
        return _vmp4_slot_build_view(build)


def _vmp4_slot_state_with_build(
    slot_state: str,
    build: _Vmp4SlotBuildView,
) -> str:
    if build.state in {"queued", "running"} and slot_state != "ready":
        return "building"
    return slot_state


def _vmp4_slot_refresh_status(
    layout: PassthroughVmp4SlotLayout,
    status: Vmp4SlotCacheStatus,
    output_mode: str,
    fps: float,
) -> Vmp4SlotCacheStatus:
    return ensure_vmp4_slot_manifest(
        layout,
        status.cache_dir.parent,
        output_mode=output_mode,
        fps=fps,
        gop_frames=PASSTHROUGH_GOP,
    )


def _vmp4_slot_wait_for_ready(
    layout: PassthroughVmp4SlotLayout,
    status: Vmp4SlotCacheStatus,
    slot_index: int,
    output_mode: str,
    fps: float,
    timeout_sec: float,
) -> _Vmp4SlotReadyWait:
    start = time.time()
    deadline = start + max(0.0, float(timeout_sec or 0.0))
    current_status = status
    build = _vmp4_slot_disabled_build_view("slot-build-not-started")
    slot_state = "missing"

    while True:
        slot = current_status.slot(slot_index)
        if slot is not None:
            slot_state = slot.state
            if slot.state == "ready":
                return _Vmp4SlotReadyWait(
                    status=current_status,
                    build=build,
                    slot_state=slot_state,
                    ready=True,
                    waited_sec=time.time() - start,
                )

        build = _vmp4_slot_schedule_placeholder_build(layout, current_status, slot_index, output_mode)
        current_status = _vmp4_slot_refresh_status(layout, current_status, output_mode, fps)
        slot = current_status.slot(slot_index)
        slot_state = slot.state if slot is not None else "missing"
        if slot is not None and slot.state == "ready":
            return _Vmp4SlotReadyWait(
                status=current_status,
                build=build,
                slot_state=slot_state,
                ready=True,
                waited_sec=time.time() - start,
            )

        slot_state = _vmp4_slot_state_with_build(slot_state, build)
        now = time.time()
        if timeout_sec <= 0.0 or now >= deadline or build.state == "disabled":
            return _Vmp4SlotReadyWait(
                status=current_status,
                build=build,
                slot_state=slot_state,
                ready=False,
                waited_sec=now - start,
            )
        time.sleep(min(0.05, max(0.001, deadline - now)))


def _vmp4_slot_build_worker(key: str, layout: PassthroughVmp4SlotLayout) -> None:
    failure = ""
    payload_size = 0
    matter = None
    try:
        with _vmp4_slot_build_lock:
            build = _vmp4_slot_builds.get(key)
        if build is None:
            return
        sample = layout.samples[build.slot_index]
        mode = (build.output_mode or "green").lower()
        if mode not in SEEK_FRAME_MODES:
            raise RuntimeError(f"slot-real-builder-unsupported-mode:{mode}")
        matter = acquire_matter(blocking=True, timeout=PASSTHROUGH_SEEK_VMP4_SLOT_MATTER_TIMEOUT)
        if matter is None:
            raise RuntimeError("slot-real-builder-matter-timeout")
        annexb = build_passthrough_hevc_annexb_gop(
            layout.source_path,
            start_sec=sample.start_time_sec,
            frame_count=1,
            matter=matter,
            output_mode=mode,
            max_bytes=max(0, int(sample.sample_size)),
        )
        payload_size = write_vmp4_slot_hevc_annexb_payload(layout, build.slot_index, build.payload_path, annexb)
        if payload_size <= 0:
            failure = "payload-empty"
            try:
                build.payload_path.unlink(missing_ok=True)
            except OSError:
                pass
        elif payload_size > sample.sample_size:
            failure = "payload-oversize"
            try:
                build.payload_path.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            update_vmp4_slot_manifest_slot_state(
                build.manifest_path,
                build.slot_index,
                "ready",
                payload_size=payload_size,
            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        release_matter(matter)
        with _vmp4_slot_build_lock:
            build = _vmp4_slot_builds.get(key)
            if build is not None:
                build.payload_size = payload_size
                build.finished_at = time.time()
                if failure:
                    build.state = "failed"
                    build.reason = failure
                    build.attempts += 1
                    lowered = failure.lower()
                    build.permanent = "oversize" in lowered or "unsupported" in lowered
                    backoff = min(
                        PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_MAX,
                        PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_BASE * (2 ** min(build.attempts - 1, 16)),
                    )
                    build.next_retry_at = build.finished_at + backoff
                    update_vmp4_slot_manifest_slot_state(
                        build.manifest_path,
                        build.slot_index,
                        "failed",
                        payload_size=payload_size,
                        reason=failure,
                    )
                    log.warning(
                        "passthrough_seek VMP4 slot build failed: slot=%d payload=%s reason=%s permanent=%s attempts=%d retry_in=%.1fs",
                        build.slot_index,
                        build.payload_path.name,
                        failure,
                        build.permanent,
                        build.attempts,
                        0.0 if build.permanent else backoff,
                    )
                else:
                    build.state = "ready"
                    build.reason = ""
                    log.info(
                        "passthrough_seek VMP4 slot build ready: slot=%d payload=%s bytes=%d",
                        build.slot_index,
                        build.payload_path.name,
                        payload_size,
                    )


_vmp4_slot_template_lock = threading.RLock()


def _vmp4_slot_even(value: int) -> int:
    return max(2, int(value) & ~1)


def _vmp4_slot_scaled_size(width: int, height: int) -> tuple[int, int]:
    width = max(2, int(width or 0))
    height = max(2, int(height or 0))
    max_side = int(DECODE_MAX_SIDE or 0)
    if max_side <= 0 or max(width, height) <= max_side:
        return _vmp4_slot_even(width), _vmp4_slot_even(height)
    if width >= height:
        out_w = max_side
        out_h = int(round(height * max_side / width))
    else:
        out_h = max_side
        out_w = int(round(width * max_side / height))
    return _vmp4_slot_even(out_w), _vmp4_slot_even(out_h)


def _vmp4_slot_output_size(info, output_mode: str) -> tuple[int, int]:
    out_w, out_h = _vmp4_slot_scaled_size(
        int(getattr(info, "width", 0) or 0),
        int(getattr(info, "height", 0) or 0),
    )
    mode = str(output_mode or "green").lower()
    if mode == "alpha":
        from pipeline.alpha_packer import alpha_output_size

        out_w, out_h = alpha_output_size(out_w, out_h)
    elif mode == "two_dvr":
        out_w = min(8192, _vmp4_slot_even(out_w * 2))
    elif mode == "superres":
        from utils.rtx_vsr import seek_target_height, target_dimensions

        # The MP4 shell must declare what the stage actually encodes, so both
        # sides ask the same function for the seek-sustainable target. The
        # stage works from the source size, not the decode-capped one.
        out_w, out_h = target_dimensions(
            int(getattr(info, "width", 0) or 0),
            int(getattr(info, "height", 0) or 0),
            seek_target_height(),
        )
    elif mode == "dlss5":
        # Neural Rendering is 1x, but it is still a stage: like SuperRes it
        # works from the source size and never sees DECODE_MAX_SIDE, so the
        # shell has to declare the source's geometry rather than the
        # decode-capped one it would otherwise inherit.
        out_w = int(getattr(info, "width", 0) or 0)
        out_h = int(getattr(info, "height", 0) or 0)
    return _vmp4_slot_even(out_w), _vmp4_slot_even(out_h)


def _vmp4_slot_template_path(width: int, height: int, fps: float) -> Path:
    fps_milli = int(round(max(1.0, float(fps or 0.0)) * 1000.0))
    return RUNTIME_CACHE_DIR / "vmp4_slot_template" / f"hevc_{int(width)}x{int(height)}_{fps_milli}.mp4"


def _vmp4_slot_output_template(info, output_mode: str) -> Vmp4SlotOutputTemplate:
    width, height = _vmp4_slot_output_size(info, output_mode)
    fps = max(1.0, float(_seek_output_fps(info) or 0.0))
    path = _vmp4_slot_template_path(width, height, fps)
    with _vmp4_slot_template_lock:
        if not path.is_file() or path.stat().st_size <= 0:
            _vmp4_slot_generate_hevc_template(path, width, height, fps)
        template = load_vmp4_slot_output_template(path)
        return Vmp4SlotOutputTemplate(
            stsd=template.stsd,
            payload=template.payload,
            codec_name="hevc",
            nal_length_size=template.nal_length_size,
            width=width,
            height=height,
        )


def _vmp4_slot_layout_cache_key(
    path: Path,
    info,
    output_mode: str,
    total: int,
    output_template: Vmp4SlotOutputTemplate,
) -> tuple:
    try:
        st = Path(path).stat()
        stat_key = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_key = (0, 0)
    return (
        _media_key_path(path),
        stat_key,
        str(output_mode or "green").lower(),
        round(float(getattr(info, "duration", 0.0) or 0.0), 6),
        round(float(_seek_output_fps(info) or 0.0), 6),
        int(total),
        int(PASSTHROUGH_GOP),
        int(PASSTHROUGH_SEEK_VMP4_SLOT_MAX_SAMPLE_BYTES),
        hashlib.sha256(output_template.stsd).hexdigest(),
        hashlib.sha256(output_template.payload).hexdigest(),
        str(output_template.codec_name or "hevc").lower(),
        int(output_template.nal_length_size),
        int(output_template.width),
        int(output_template.height),
    )


def _vmp4_slot_cached_layout(
    path: Path,
    info,
    output_mode: str,
    total: int,
    output_template: Vmp4SlotOutputTemplate,
) -> PassthroughVmp4SlotLayout:
    key = _vmp4_slot_layout_cache_key(path, info, output_mode, total, output_template)
    with _vmp4_slot_layout_cache_lock:
        cached = _vmp4_slot_layout_cache.get(key)
        if cached is not None:
            _vmp4_slot_layout_cache.pop(key, None)
            _vmp4_slot_layout_cache[key] = cached
            return cached
    layout = build_passthrough_vmp4_slot_layout(
        path,
        duration_sec=float(getattr(info, "duration", 0.0) or 0.0),
        fps=_seek_output_fps(info),
        total_size=total,
        gop_frames=PASSTHROUGH_GOP,
        max_sample_bytes=PASSTHROUGH_SEEK_VMP4_SLOT_MAX_SAMPLE_BYTES,
        output_stsd=output_template.stsd,
        output_codec_name=output_template.codec_name,
        output_width=output_template.width,
        output_height=output_template.height,
        placeholder_payload=output_template.payload,
    )
    with _vmp4_slot_layout_cache_lock:
        _vmp4_slot_layout_cache[key] = layout
        while len(_vmp4_slot_layout_cache) > _VMP4_SLOT_LAYOUT_CACHE_LIMIT:
            old_key = next(iter(_vmp4_slot_layout_cache))
            _vmp4_slot_layout_cache.pop(old_key, None)
    return layout


def _vmp4_slot_first_intersecting_slot(
    layout: PassthroughVmp4SlotLayout,
    start: int,
    end_inclusive: int,
) -> int | None:
    if end_inclusive < layout.mdat_payload_start or layout.slot_stride <= 0:
        return None
    cursor = max(int(start), int(layout.mdat_payload_start))
    slot_index = max(0, (cursor - layout.mdat_payload_start) // layout.slot_stride)
    while slot_index < layout.slot_count:
        sample = layout.samples[int(slot_index)]
        slot_start = layout.mdat_payload_start + sample.slot_index * layout.slot_stride
        slot_end = slot_start + sample.sample_size - 1
        if slot_end < cursor:
            slot_index += 1
            continue
        if slot_start > end_inclusive:
            return None
        return int(slot_index)
    return None


def _vmp4_slot_first_unready_slot(
    layout: PassthroughVmp4SlotLayout,
    status: Vmp4SlotCacheStatus,
    start: int,
    end_inclusive: int,
) -> int | None:
    if end_inclusive < layout.mdat_payload_start or layout.slot_stride <= 0:
        return None
    cursor = max(int(start), int(layout.mdat_payload_start))
    slot_index = max(0, (cursor - layout.mdat_payload_start) // layout.slot_stride)
    while slot_index < layout.slot_count:
        sample = layout.samples[int(slot_index)]
        slot_start = layout.mdat_payload_start + sample.slot_index * layout.slot_stride
        slot_end = slot_start + sample.sample_size - 1
        if slot_end < cursor:
            slot_index += 1
            continue
        if slot_start > end_inclusive:
            return None
        slot = status.slot(int(slot_index))
        if slot is None or slot.state != "ready":
            return int(slot_index)
        slot_index += 1
    return None


def _iter_vmp4_slot_range_ready_only(
    layout: PassthroughVmp4SlotLayout,
    status: Vmp4SlotCacheStatus,
    output_mode: str,
    fps: float,
    start: int,
    end_inclusive: int,
    *,
    chunk_size: int,
    wait_timeout_sec: float,
    rid: int = 0,
):
    cursor = max(0, int(start))
    stop = min(max(0, int(end_inclusive)), max(0, layout.total_size - 1))
    if cursor > stop:
        return

    init_end = len(layout.init)
    if cursor < init_end:
        init_stop = min(stop, init_end - 1)
        yield layout.init[cursor : init_stop + 1]
        cursor = init_stop + 1
    if cursor > stop:
        return

    current_status = status
    for sample in layout.samples:
        slot_start = layout.mdat_payload_start + sample.slot_index * layout.slot_stride
        slot_end = slot_start + sample.sample_size
        if slot_end <= cursor:
            continue
        if slot_start > stop:
            break
        if cursor < slot_start:
            gap_end = min(stop + 1, slot_start)
            yield from iter_vmp4_slot_range(
                layout,
                cursor,
                gap_end - 1,
                chunk_size=chunk_size,
                payload_paths=current_status.ready_payloads,
            )
            cursor = gap_end
            if cursor > stop:
                return

        overlap_start = max(cursor, slot_start)
        overlap_end = min(stop + 1, slot_end)
        if overlap_start >= overlap_end:
            continue

        wait = _vmp4_slot_wait_for_ready(
            layout,
            current_status,
            sample.slot_index,
            output_mode,
            fps,
            wait_timeout_sec,
        )
        current_status = wait.status
        if not wait.ready:
            log.warning(
                "passthrough_seek[%d] VMP4 slot stream wait failed: slot=%d state=%s build=%s waited=%.3fs reason=%s",
                rid,
                sample.slot_index,
                wait.slot_state,
                wait.build.state,
                wait.waited_sec,
                wait.build.reason,
            )
            raise RuntimeError(f"VMP4 slot not ready: slot={sample.slot_index} state={wait.slot_state}")

        yield from iter_vmp4_slot_range(
            layout,
            overlap_start,
            overlap_end - 1,
            chunk_size=chunk_size,
            payload_paths=current_status.ready_payloads,
        )
        cursor = overlap_end
        if cursor > stop:
            return

    if cursor <= stop:
        yield from iter_vmp4_slot_range(
            layout,
            cursor,
            stop,
            chunk_size=chunk_size,
            payload_paths=current_status.ready_payloads,
        )


def _vmp4_slot_generate_hevc_template(path: Path, width: int, height: int, fps: float) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.tmp.mp4")
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    pyav_error = _vmp4_slot_generate_hevc_template_pyav(tmp, width, height, fps)
    if not pyav_error and tmp.is_file() and tmp.stat().st_size > 0:
        os.replace(tmp, target)
        log.info(
            "passthrough_seek VMP4 slot HEVC template ready via pyav: %s size=%dx%d fps=%.3f bytes=%d",
            target.name,
            width,
            height,
            fps,
            target.stat().st_size,
        )
        return
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    if pyav_error:
        log.warning(
            "passthrough_seek VMP4 slot PyAV HEVC template failed, falling back to ffmpeg: %s",
            pyav_error,
        )
    base = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={int(width)}x{int(height)}:r={max(1.0, float(fps)):.6f}:d=1",
        "-frames:v",
        "1",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "1",
        "-bf",
        "0",
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
    ]
    attempts = (
        [*base, "-c:v", "hevc_nvenc", "-preset", "p1", "-b:v", "2M", str(tmp)],
        [
            *base,
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "log-level=error:keyint=1:min-keyint=1:scenecut=0",
            "-b:v",
            "2M",
            str(tmp),
        ],
    )
    last_error = ""
    for cmd in attempts:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=45,
                **hidden_subprocess_kwargs(),
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            os.replace(tmp, target)
            log.info(
                "passthrough_seek VMP4 slot HEVC template ready: %s size=%dx%d fps=%.3f bytes=%d",
                target.name,
                width,
                height,
                fps,
                target.stat().st_size,
            )
            return
        stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
        last_error = stderr[-500:] if stderr else f"rc={proc.returncode}"
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    raise Vmp4SlotLayoutError(f"slot-output-template-build-failed:{last_error or 'unknown'}")


def _vmp4_slot_generate_hevc_template_pyav(path: Path, width: int, height: int, fps: float) -> str:
    try:
        from fractions import Fraction

        import av  # type: ignore[import-not-found]
        import numpy as np
    except Exception as exc:
        return f"pyav-import:{type(exc).__name__}: {exc}"

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(max(1.0, float(fps or 0.0))).limit_denominator(1_000_000)
    attempts = (
        (
            "hevc_nvenc",
            "nv12",
            {
                "preset": "p1",
                "b": "2M",
                "g": "1",
                "bf": "0",
            },
        ),
        (
            "libx265",
            "yuv420p",
            {
                "preset": "ultrafast",
                "b": "2M",
                "x265-params": "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0",
            },
        ),
        (
            "hevc",
            "yuv420p",
            {
                "preset": "ultrafast",
                "b": "2M",
                "x265-params": "log-level=error:keyint=1:min-keyint=1:scenecut=0:bframes=0",
            },
        ),
    )
    errors: list[str] = []
    for codec_name, pix_fmt, options in attempts:
        container = None
        try:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            container = av.open(str(target), "w", format="mp4")
            stream = container.add_stream(codec_name, rate=rate, options=options)
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = pix_fmt
            ctx = stream.codec_context
            try:
                ctx.gop_size = 1
            except Exception:
                pass
            try:
                ctx.max_b_frames = 0
            except Exception:
                pass
            try:
                ctx.bit_rate = 2_000_000
            except Exception:
                pass
            try:
                ctx.codec_tag = "hvc1"
            except Exception:
                pass
            frame = av.VideoFrame.from_ndarray(
                np.zeros((int(height), int(width), 3), dtype=np.uint8),
                format="rgb24",
            )
            frame = frame.reformat(width=int(width), height=int(height), format=pix_fmt)
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            if target.is_file() and target.stat().st_size > 0:
                return ""
            errors.append(f"{codec_name}:empty-output")
        except Exception as exc:
            errors.append(f"{codec_name}:{type(exc).__name__}: {exc}")
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass
    return "; ".join(errors[-3:]) or "pyav-unknown"


def _mp4_free_box_header(size: int) -> bytes:
    size = int(size)
    if size < 8:
        return b""
    if size <= 0xFFFFFFFF:
        return size.to_bytes(4, "big") + b"free"
    return (1).to_bytes(4, "big") + b"free" + size.to_bytes(8, "big")


def _iter_mp4_free_box_range(
    box_size: int,
    start: int,
    end: int,
    *,
    chunk_size: int = 64 * 1024,
):
    header = _mp4_free_box_header(box_size)
    cursor = max(0, int(start))
    stop = min(max(0, int(end)), max(0, int(box_size) - 1))
    if cursor > stop or not header:
        return
    if cursor < len(header):
        header_end = min(stop, len(header) - 1)
        yield header[cursor : header_end + 1]
        cursor = header_end + 1
    if cursor <= stop:
        pad = b"\x00" * min(max(1, int(chunk_size)), stop - cursor + 1)
        while cursor <= stop:
            chunk = pad[: min(len(pad), stop - cursor + 1)]
            cursor += len(chunk)
            yield chunk


def _iter_mp4_file_with_free_pad_range(
    path: Path,
    start: int,
    end: int,
    *,
    declared_total: int,
    chunk_size: int = 64 * 1024,
):
    file_size = max(0, int(path.stat().st_size))
    cursor = max(0, int(start))
    stop = min(max(0, int(end)), max(0, int(declared_total) - 1))
    if cursor <= stop and cursor < file_size:
        file_end = min(stop, file_size - 1)
        with path.open("rb") as fh:
            fh.seek(cursor)
            remaining = file_end - cursor + 1
            while remaining > 0:
                chunk = fh.read(min(max(1, int(chunk_size)), remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        cursor = file_end + 1
    if cursor <= stop and file_size < declared_total:
        pad_size = declared_total - file_size
        pad_start = max(0, cursor - file_size)
        pad_end = min(pad_size - 1, stop - file_size)
        yield from _iter_mp4_free_box_range(
            pad_size,
            pad_start,
            pad_end,
            chunk_size=chunk_size,
        )


def _mp4_top_level_offsets(path: Path, max_boxes: int = 256) -> tuple[int | None, int | None, str]:
    try:
        file_size = max(0, int(path.stat().st_size))
        with path.open("rb") as fh:
            offset = 0
            moov_offset: int | None = None
            mdat_offset: int | None = None
            boxes = 0
            while offset + 8 <= file_size and boxes < max_boxes:
                fh.seek(offset)
                header = fh.read(16)
                if len(header) < 8:
                    return moov_offset, mdat_offset, "short-header"
                box_size = int.from_bytes(header[:4], "big")
                box_type = header[4:8]
                header_size = 8
                if box_size == 1:
                    if len(header) < 16:
                        return moov_offset, mdat_offset, "short-large-header"
                    box_size = int.from_bytes(header[8:16], "big")
                    header_size = 16
                elif box_size == 0:
                    box_size = file_size - offset
                if box_size < header_size or offset + box_size > file_size:
                    return moov_offset, mdat_offset, "invalid-box-size"
                if box_type == b"moov":
                    moov_offset = offset
                    return moov_offset, mdat_offset, ""
                if box_type == b"mdat" and mdat_offset is None:
                    mdat_offset = offset
                offset += box_size
                boxes += 1
            if boxes >= max_boxes:
                return moov_offset, mdat_offset, "too-many-boxes"
            return moov_offset, mdat_offset, ""
    except OSError as e:
        return None, None, type(e).__name__


def _seek_vmp4_cache_response(
    *,
    path: Path,
    info,
    output_mode: str,
    total: int,
    range_header: str | None,
    headers: dict[str, str],
    head_only: bool,
    rid: int = 0,
) -> Response:
    headers["Accept-Ranges"] = "bytes"
    headers["Content-Type"] = "video/mp4"
    headers["X-Passthrough-VMP4-Backend"] = "cache_file"
    if total <= 0:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "source-size-missing"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 source size unavailable", status_code=409, headers=error_headers, media_type="text/plain")
    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "range-unsatisfiable"
        error_headers["Content-Range"] = f"bytes */{max(0, int(total))}"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Type", None)
        return Response(status_code=416, headers=error_headers)
    cache_path = _vmp4_mode_candidate(path, output_mode, info)
    if cache_path is None:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-missing"
        expected = _vmp4_expected_cache_names(path, output_mode, info)
        if expected:
            error_headers["X-Passthrough-VMP4-Expected-Cache"] = ",".join(_header_safe_text(name) for name in expected)
        target = _vmp4_expected_cache_target(path, output_mode, info)
        build = _vmp4_schedule_cache_build(path, info, output_mode, target, total=total)
        _apply_vmp4_build_headers(error_headers, build)
        log.info(
            "passthrough_seek[%d] VMP4 cache missing: %s mode=%s expected=%s build=%s target=%s log=%s reason=%s",
            rid,
            path.name,
            output_mode,
            ",".join(expected),
            build.state,
            build.target.name if build.target else "",
            build.log_path.name if build.log_path else "",
            build.reason,
        )
        error_headers["Retry-After"] = "2"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 cache not ready", status_code=503, headers=error_headers, media_type="text/plain")
    cache_size = max(0, int(cache_path.stat().st_size))
    pad_size = max(0, int(total) - cache_size)
    headers["X-Passthrough-VMP4-Phase"] = "cache-file"
    headers["X-Passthrough-VMP4-Cache"] = _header_safe_text(cache_path.name)
    headers["X-Passthrough-VMP4-Cache-Size"] = str(cache_size)
    headers["X-Passthrough-VMP4-Pad"] = "free" if pad_size else "none"
    headers["X-Passthrough-VMP4-Pad-Bytes"] = str(pad_size)
    moov_offset, mdat_offset, moov_error = _mp4_top_level_offsets(cache_path)
    if moov_offset is not None:
        headers["X-Passthrough-VMP4-Moov-Offset"] = str(moov_offset)
    if mdat_offset is not None:
        headers["X-Passthrough-VMP4-Mdat-Offset"] = str(mdat_offset)
    if moov_error:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-moov-probe-error"
        error_headers["X-Passthrough-VMP4-Moov-Probe-Error"] = moov_error
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 cache MP4 box probe failed", status_code=409, headers=error_headers, media_type="text/plain")
    if moov_offset is None:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-moov-missing"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 cache moov box missing", status_code=409, headers=error_headers, media_type="text/plain")
    if mdat_offset is not None and mdat_offset < moov_offset:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-moov-after-mdat"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 cache must be faststart with moov before mdat", status_code=409, headers=error_headers, media_type="text/plain")
    try:
        cache_info = probe_cached(cache_path)
        cache_duration = max(0.0, float(getattr(cache_info, "duration", 0.0) or 0.0))
        source_duration = max(0.0, float(getattr(info, "duration", 0.0) or 0.0))
        source_fps = max(0.0, float(getattr(info, "fps", 0.0) or 0.0))
        duration_tolerance = max(0.5, (2.0 / source_fps) if source_fps > 0 else 0.0)
        duration_delta = abs(cache_duration - source_duration)
        headers["X-Passthrough-VMP4-Cache-Duration"] = f"{cache_duration:.3f}"
        headers["X-Passthrough-VMP4-Duration-Delta"] = f"{duration_delta:.3f}"
        headers["X-Passthrough-VMP4-Duration-Tolerance"] = f"{duration_tolerance:.3f}"
        if source_duration > 0 and (cache_duration <= 0 or duration_delta > duration_tolerance):
            error_headers = dict(headers)
            error_headers["X-Passthrough-VMP4-Phase"] = "cache-duration-mismatch"
            error_headers.pop("Content-Length", None)
            error_headers.pop("Content-Range", None)
            error_headers.pop("Content-Type", None)
            return Response("VMP4 cache duration does not match source", status_code=409, headers=error_headers, media_type="text/plain")
    except Exception as e:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-probe-error"
        error_headers["X-Passthrough-VMP4-Cache-Probe-Error"] = type(e).__name__
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 cache probe failed", status_code=409, headers=error_headers, media_type="text/plain")
    if cache_size > total:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-oversize"
        build = _vmp4_schedule_cache_build(
            path,
            info,
            output_mode,
            cache_path,
            total=total,
            force=True,
            reason="cache-oversize",
            observed_cache_size=cache_size,
        )
        _apply_vmp4_build_headers(error_headers, build)
        log.info(
            "passthrough_seek[%d] VMP4 cache oversize: %s mode=%s cache=%s cache_size=%d total=%d rebuild=%s log=%s reason=%s",
            rid,
            path.name,
            output_mode,
            cache_path.name,
            cache_size,
            total,
            build.state,
            build.log_path.name if build.log_path else "",
            build.reason,
        )
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        if build.state != "disabled":
            error_headers["Retry-After"] = "2"
            return Response("VMP4 cache rebuilding", status_code=503, headers=error_headers, media_type="text/plain")
        return Response("VMP4 cache exceeds source-size budget", status_code=409, headers=error_headers, media_type="text/plain")
    if 0 < pad_size < 8:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "cache-pad-too-small"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 cache leaves too little room for MP4 free padding", status_code=409, headers=error_headers, media_type="text/plain")
    response_range = byte_range or ByteRange(start=0, end=max(0, total - 1), total=total)
    zero_open = _is_zero_open_range(range_header, byte_range)
    status_code = 206 if byte_range is not None and not zero_open else 200
    headers["Content-Length"] = str(response_range.length)
    if status_code == 206:
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{total}"
    else:
        headers.pop("Content-Range", None)
    log.info(
        "passthrough_seek[%d] VMP4 cache response: status=%d mode=%s cache=%s cache_size=%d pad=%d total=%d content_length=%d range=%r content_range=%r head=%s",
        rid,
        status_code,
        output_mode,
        cache_path.name,
        cache_size,
        pad_size,
        total,
        response_range.length,
        range_header,
        headers.get("Content-Range"),
        head_only,
    )
    if head_only:
        return Response(status_code=status_code, headers=headers, media_type="video/mp4")
    body = _iter_mp4_file_with_free_pad_range(
        cache_path,
        response_range.start,
        response_range.end,
        declared_total=total,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )
    return StreamingResponse(
        body,
        status_code=status_code,
        headers=headers,
        media_type="video/mp4",
    )


def _seek_vmp4_slot_response(
    *,
    path: Path,
    info,
    output_mode: str,
    total: int,
    range_header: str | None,
    headers: dict[str, str],
    head_only: bool,
    rid: int = 0,
) -> Response:
    headers["Accept-Ranges"] = "bytes"
    headers["Content-Type"] = "video/mp4"
    headers["X-Passthrough-VMP4-Backend"] = "slot"
    if total <= 0:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "source-size-missing"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 source size unavailable", status_code=409, headers=error_headers, media_type="text/plain")

    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "range-unsatisfiable"
        error_headers["Content-Range"] = f"bytes */{max(0, int(total))}"
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Type", None)
        return Response(status_code=416, headers=error_headers)

    try:
        output_template = _vmp4_slot_output_template(info, output_mode)
        layout = _vmp4_slot_cached_layout(path, info, output_mode, total, output_template)
    except Vmp4SlotLayoutError as exc:
        error_headers = dict(headers)
        error_headers["X-Passthrough-VMP4-Phase"] = "slot-layout-error"
        error_headers["X-Passthrough-VMP4-Slot-Error"] = _header_safe_text(str(exc))
        error_headers.pop("Content-Length", None)
        error_headers.pop("Content-Range", None)
        error_headers.pop("Content-Type", None)
        return Response("VMP4 slot layout unavailable", status_code=409, headers=error_headers, media_type="text/plain")
    output_fps = _seek_output_fps(info)
    cache_status = ensure_vmp4_slot_manifest(
        layout,
        RUNTIME_CACHE_DIR / "vmp4_slot",
        output_mode=output_mode,
        fps=output_fps,
        gop_frames=PASSTHROUGH_GOP,
    )

    response_range = byte_range or ByteRange(start=0, end=max(0, total - 1), total=total)
    range_bounded_for_ready_only = False
    ready_only_blocking_slot: int | None = None
    ready_wait_slot: int | None = None
    ready_wait: _Vmp4SlotReadyWait | None = None
    if PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY and not head_only:
        ready_wait_slot = _vmp4_slot_first_unready_slot(
            layout,
            cache_status,
            response_range.start,
            response_range.end,
        )
        if ready_wait_slot is not None:
            ready_wait = _vmp4_slot_wait_for_ready(
                layout,
                cache_status,
                ready_wait_slot,
                output_mode,
                output_fps,
                PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT,
            )
            cache_status = ready_wait.status
    stream_ready_wait = bool(PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY and not head_only and not range_header)
    if ready_wait is not None and not ready_wait.ready:
        ready_only_blocking_slot = ready_wait_slot
        if ready_only_blocking_slot is not None and range_header and ready_only_blocking_slot > 0:
            blocking_sample = layout.samples[ready_only_blocking_slot]
            blocking_start = layout.mdat_payload_start + blocking_sample.slot_index * layout.slot_stride
            if response_range.start < blocking_start:
                response_range = ByteRange(
                    start=response_range.start,
                    end=min(response_range.end, blocking_start - 1),
                    total=total,
                )
                range_bounded_for_ready_only = True
    elif PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY and not head_only and not stream_ready_wait:
        ready_only_blocking_slot = _vmp4_slot_first_unready_slot(
            layout,
            cache_status,
            response_range.start,
            response_range.end,
        )
        if ready_only_blocking_slot is not None and range_header:
            blocking_sample = layout.samples[ready_only_blocking_slot]
            blocking_start = layout.mdat_payload_start + blocking_sample.slot_index * layout.slot_stride
            if response_range.start < blocking_start:
                response_range = ByteRange(
                    start=response_range.start,
                    end=min(response_range.end, blocking_start - 1),
                    total=total,
                )
                range_bounded_for_ready_only = True
    zero_open = _is_zero_open_range(range_header, byte_range)
    status_code = 206 if (byte_range is not None and not zero_open) or range_bounded_for_ready_only else 200
    headers["X-Passthrough-VMP4-Phase"] = "slot-layout"
    headers["X-Passthrough-VMP4-Source-Codec"] = layout.source_codec_name
    headers["X-Passthrough-VMP4-Output-Codec"] = layout.codec_name
    headers["X-Passthrough-VMP4-Output-Size"] = f"{output_template.width}x{output_template.height}"
    headers["X-Passthrough-VMP4-Slot-Count"] = str(layout.slot_count)
    headers["X-Passthrough-VMP4-Slot-Size"] = str(layout.slot_size)
    headers["X-Passthrough-VMP4-Slot-Stride"] = str(layout.slot_stride)
    headers["X-Passthrough-VMP4-Slot-Gap"] = str(layout.slot_gap_size)
    headers["X-Passthrough-VMP4-Slot-Duration"] = f"{layout.slot_duration_sec:.3f}"
    headers["X-Passthrough-VMP4-Slot-Ready-Only"] = "1" if PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY else "0"
    if range_bounded_for_ready_only:
        headers["X-Passthrough-VMP4-Slot-Ready-Bounded"] = "1"
    if ready_wait_slot is not None and ready_wait is not None:
        headers["X-Passthrough-VMP4-Slot-Wait"] = str(ready_wait_slot)
        headers["X-Passthrough-VMP4-Slot-Wait-Ready"] = "1" if ready_wait.ready else "0"
        headers["X-Passthrough-VMP4-Slot-Wait-Seconds"] = f"{ready_wait.waited_sec:.3f}"
    headers["X-Passthrough-VMP4-Moov-Size"] = str(layout.moov_size)
    headers["X-Passthrough-VMP4-Mdat-Payload-Start"] = str(layout.mdat_payload_start)
    headers["X-Passthrough-VMP4-Mdat-Payload-Size"] = str(layout.mdat_payload_size)
    headers["X-Passthrough-VMP4-Slot-Cache"] = cache_status.digest
    headers["X-Passthrough-VMP4-Slot-Manifest"] = _header_safe_text(cache_status.manifest_path.name)
    counts = cache_status.state_counts
    if counts:
        headers["X-Passthrough-VMP4-Slot-States"] = ",".join(f"{key}:{counts[key]}" for key in sorted(counts))
    slot_index = (
        ready_only_blocking_slot
        if ready_only_blocking_slot is not None and not range_bounded_for_ready_only
        else vmp4_slot_index_for_offset(layout, response_range.start)
    )
    if slot_index is None:
        slot_index = _vmp4_slot_first_intersecting_slot(layout, response_range.start, response_range.end)
    if slot_index is not None:
        sample = layout.samples[slot_index]
        slot_cache = cache_status.slot(slot_index)
        slot_build = _vmp4_slot_schedule_placeholder_build(layout, cache_status, slot_index, output_mode)
        slot_state = slot_cache.state if slot_cache is not None else "placeholder"
        slot_state = _vmp4_slot_state_with_build(slot_state, slot_build)
        payload_bytes = slot_cache.payload_size if slot_cache is not None and slot_cache.state == "ready" else len(layout.placeholder_payload)
        headers["X-Passthrough-VMP4-Slot"] = str(slot_index)
        headers["X-Passthrough-VMP4-Slot-Time"] = f"{sample.start_time_sec:.3f}"
        headers["X-Passthrough-VMP4-Slot-Source-Sample"] = str(sample.source_sample_index)
        headers["X-Passthrough-VMP4-Slot-State"] = slot_state
        headers["X-Passthrough-VMP4-Slot-Build"] = slot_build.state
        if slot_build.reason:
            headers["X-Passthrough-VMP4-Slot-Build-Reason"] = _header_safe_text(slot_build.reason)
        headers["X-Passthrough-VMP4-Slot-Sample-Size"] = str(sample.sample_size)
        headers["X-Passthrough-VMP4-Slot-Source-Bytes"] = str(sample.source_size)
        headers["X-Passthrough-VMP4-Slot-Placeholder-Bytes"] = str(len(layout.placeholder_payload))
        headers["X-Passthrough-VMP4-Slot-Payload-Bytes"] = str(payload_bytes)
        headers["X-Passthrough-VMP4-Slot-Filler-Bytes"] = str(max(0, sample.sample_size - payload_bytes))
        headers["X-Passthrough-VMP4-Slot-Filler"] = "nal"
        if PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY and slot_state != "ready" and not head_only:
            not_ready_headers = dict(headers)
            not_ready_headers["X-Passthrough-VMP4-Phase"] = "slot-not-ready"
            not_ready_headers["Retry-After"] = "1"
            not_ready_headers.pop("Content-Length", None)
            not_ready_headers.pop("Content-Range", None)
            not_ready_headers.pop("Content-Type", None)
            log.info(
                "passthrough_seek[%d] VMP4 slot not ready: slot=%d state=%s build=%s range=%r ready_only=%s",
                rid,
                slot_index,
                slot_state,
                slot_build.state,
                range_header,
                PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY,
            )
            return Response(
                "VMP4 slot not ready",
                status_code=503,
                headers=not_ready_headers,
                media_type="text/plain",
            )
    if range_bounded_for_ready_only and ready_only_blocking_slot is not None:
        blocking_cache = cache_status.slot(ready_only_blocking_slot)
        blocking_build = _vmp4_slot_schedule_placeholder_build(layout, cache_status, ready_only_blocking_slot, output_mode)
        blocking_state = blocking_cache.state if blocking_cache is not None else "placeholder"
        blocking_state = _vmp4_slot_state_with_build(blocking_state, blocking_build)
        headers["X-Passthrough-VMP4-Next-Slot"] = str(ready_only_blocking_slot)
        headers["X-Passthrough-VMP4-Next-Slot-State"] = blocking_state
        headers["X-Passthrough-VMP4-Next-Slot-Build"] = blocking_build.state
        if blocking_build.reason:
            headers["X-Passthrough-VMP4-Next-Slot-Build-Reason"] = _header_safe_text(blocking_build.reason)
    headers["Content-Length"] = str(response_range.length)
    if status_code == 206:
        headers["Content-Range"] = f"bytes {response_range.start}-{response_range.end}/{total}"
    else:
        headers.pop("Content-Range", None)
    log.info(
        "passthrough_seek[%d] VMP4 slot response: status=%d mode=%s source_codec=%s output_codec=%s output=%s total=%d slots=%d slot_size=%d slot_stride=%d range=%r content_range=%r head=%s",
        rid,
        status_code,
        output_mode,
        layout.source_codec_name,
        layout.codec_name,
        headers.get("X-Passthrough-VMP4-Output-Size"),
        total,
        layout.slot_count,
        layout.slot_size,
        layout.slot_stride,
        range_header,
        headers.get("Content-Range"),
        head_only,
    )
    if head_only:
        return Response(status_code=status_code, headers=headers, media_type="video/mp4")
    if stream_ready_wait:
        body = _iter_vmp4_slot_range_ready_only(
            layout,
            cache_status,
            output_mode,
            output_fps,
            response_range.start,
            response_range.end,
            chunk_size=DEFAULT_CHUNK_SIZE,
            wait_timeout_sec=PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT,
            rid=rid,
        )
    else:
        body = iter_vmp4_slot_range(
            layout,
            response_range.start,
            response_range.end,
            chunk_size=DEFAULT_CHUNK_SIZE,
            payload_paths=cache_status.ready_payloads,
        )
    return StreamingResponse(
        body,
        status_code=status_code,
        headers=headers,
        media_type="video/mp4",
    )


# --- slot_frames backend: frame-level layout, real fps, background GOP build ---

from pipeline.passthrough_vmp4_slot import _write_bytes_atomic as _vmp4_write_bytes_atomic


@dataclass
class _Vmp4FramesFiller:
    """A persistent forward streaming-encode run that fills per-frame files.

    Instead of rebuilding the PyNv pipeline per GOP (setup dominated -> ~10fps),
    one filler sets up decoder/encoder/matter once and streams frames in order at
    live throughput, writing frame_XXXXXX.bin as it goes. A seek far beyond the
    write cursor (or before the run's start) restarts the filler at that point.
    """
    digest: str
    cache_dir: Path
    output_mode: str
    start_frame: int = 0
    cursor: int = 0          # next frame index this run will write
    state: str = "running"   # running | done | stopped | failed
    reason: str = ""
    started_at: float = 0.0
    first_frame_at: float = 0.0   # when this run wrote its first frame
    last_hit_at: float = 0.0      # last time a request was served from its range
    stop_event: threading.Event = dataclass_field(default_factory=threading.Event)
    # Set when the worker has exited and released its Matter, so a replacement
    # knows when the pool slot is actually free.
    done_event: threading.Event = dataclass_field(default_factory=threading.Event)


# Minimum seconds between filler repositions. Players open several concurrent
# range connections at different offsets; without this debounce each one's
# ensure_filler would kill+restart the single filler at its own offset, so it
# ping-pongs and never produces a frame. With it, one run gets to make real
# forward progress before a genuine (persistent) seek can reposition it.
#
# The cooldown runs from the first frame written, not from the start: an 8K setup
# (open decoder, CreateEncoder, matter reset) takes 5-10s on its own, so timing
# from the start spent the whole window on setup and let the next connection
# preempt the run just as it began producing. Measured with six concurrent
# connections that way: 26 pipeline rebuilds over 18 requests, and the sequential
# playback connection needed 203s for its first 4 MB.
_VMP4_FRAMES_RESTART_COOLDOWN = 8.0
# Once a run is actually producing frames it is doing the useful work, so give it
# a longer guaranteed window before another offset may take it over.
_VMP4_FRAMES_PRODUCTIVE_HOLD = 15.0
# How long a replacement filler waits for the run it supersedes to release its
# Matter before starting anyway.
_VMP4_FRAMES_HANDOVER_TIMEOUT = 20.0
# A filler is "being followed" while requests keep landing inside the range it
# covers. Once nothing has for this long, the player has moved on (it seeks by
# dropping the old connection and opening a new one elsewhere) and the run may be
# repositioned immediately - the debounce below exists to stop concurrent
# connections from fighting, not to make a real seek wait.
_VMP4_FRAMES_FOLLOW_WINDOW = 2.0
_vmp4_frames_lock = threading.RLock()
_vmp4_frames_fillers: dict[str, _Vmp4FramesFiller] = {}
_vmp4_frames_layout_cache_lock = threading.RLock()
_vmp4_frames_layout_cache: dict[tuple, PassthroughVmp4FramesLayout] = {}
_VMP4_FRAMES_LAYOUT_CACHE_LIMIT = 32


def _media_key_path(path) -> str:
    """Stable identity for a media file, for cache keys - without touching it.

    ``Path.resolve()`` was used here, which on Windows opens the file to ask for
    its final path name. On the user's mounted library volume that call raises
    WinError 1005 ("the volume does not contain a recognized file system") while
    ``open`` and ``stat`` on the very same path work fine, so every seek request
    for a title on that mount died with a 500 before it reached any of our code -
    the player reported the item as unsupported. ``abspath`` is pure string work
    and gives the same value on volumes where ``resolve`` succeeds, so cache keys
    (and the frame caches they name) are unchanged there.
    """
    return os.path.abspath(str(path))


def _vmp4_frames_layout_key(path: Path, info, output_mode: str, template: Vmp4SlotOutputTemplate) -> tuple:
    try:
        st = Path(path).stat()
        stat_key = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_key = (0, 0)
    return (
        _media_key_path(path), stat_key, str(output_mode or "green").lower(),
        round(float(getattr(info, "duration", 0.0) or 0.0), 6),
        round(float(_seek_output_fps(info) or 0.0), 6),
        int(PASSTHROUGH_GOP),
        int(PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES),
        seek_budget_scale(
            stat_key[0],
            float(getattr(info, "duration", 0.0) or 0.0),
            int(getattr(info, "width", 0) or 0),
            int(getattr(info, "height", 0) or 0),
            float(getattr(info, "fps", 0.0) or 0.0),
            output_mode,
        ),
        bool(PASSTHROUGH_SEEK_VMP4_FRAMES_SOURCE_BUDGET),
        _vmp4_frames_audio_key(path),
        seek_frame_floor_bytes(
            int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
        ),
        float(PASSTHROUGH_SEEK_VMP4_FRAMES_IDR_WEIGHT),
        float(PASSTHROUGH_SEEK_VMP4_FRAMES_BUDGET_FLATTEN),
        hashlib.sha256(template.stsd).hexdigest(), int(template.width), int(template.height),
    )


_vmp4_audio_table_cache: dict[tuple, object] = {}
_vmp4_audio_table_lock = threading.Lock()


def _vmp4_frames_audio_key(path: Path) -> tuple:
    try:
        st = Path(path).stat()
        return (_media_key_path(path), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return (str(path), 0, 0)


def _vmp4_frames_audio_table(path: Path):
    """Source audio sample table, or None when the source has no audio.

    Parsing a long title's audio table is not cheap (a 50min 8K file carries
    ~140k AAC samples) and the layout is built twice - a flat probe for the init
    size, then the real one - so keep it.
    """
    key = _vmp4_frames_audio_key(path)
    with _vmp4_audio_table_lock:
        if key in _vmp4_audio_table_cache:
            return _vmp4_audio_table_cache[key]
    table = None
    try:
        table = read_media_sample_table(Path(path), "audio")
        if not table.samples:
            table = None
    except Exception as exc:
        log.info("vmp4 frames: no audio track on %s (%s)", Path(path).name, type(exc).__name__)
        table = None
    with _vmp4_audio_table_lock:
        _vmp4_audio_table_cache[key] = table
        while len(_vmp4_audio_table_cache) > 8:
            _vmp4_audio_table_cache.pop(next(iter(_vmp4_audio_table_cache)), None)
    return table


def _vmp4_frames_source_budget(
    path: Path, layout: PassthroughVmp4FramesLayout, output_mode: str, info=None
) -> SourceBudgetPlan | None:
    """Per-GOP budgets inherited from the source, or None to keep the flat ones.

    ``layout`` must be a already-built flat layout; only its init size (which the
    budget values cannot change) and frame count are used.
    """
    if not PASSTHROUGH_SEEK_VMP4_FRAMES_SOURCE_BUDGET:
        return None
    if str(output_mode or "").lower() == "superres":
        # Superres emits more pixels than the source, so the source's byte
        # budget no longer describes the same picture.
        return None
    try:
        source_size = int(Path(path).stat().st_size)
    except OSError:
        return None
    init_size = int(layout.total_size) - int(layout.mdat_payload_size)
    # Audio is copied through at its source size, so it comes off the top: only
    # what is left funds the video budgets, and the total still lands on the
    # size DIDL advertises. That is the source size at 4K and below; above
    # _BUDGET_SCALE_PIXELS it is source_size * _BUDGET_SCALE, because inheriting
    # the source's bytes exactly assumes we encode as well as the source did and
    # at 8K we visibly do not (offline multi-pass x265 vs realtime NVENC).
    declared_total = seek_declared_total_bytes(
        source_size,
        float(getattr(info, "duration", 0.0) or 0.0),
        int(getattr(info, "width", 0) or 0),
        int(getattr(info, "height", 0) or 0),
        float(getattr(info, "fps", 0.0) or 0.0),
        output_mode,
    )
    payload_total = declared_total - init_size - int(layout.audio_bytes)
    if payload_total <= 0:
        return None
    try:
        return build_source_budget_plan(
            path,
            payload_total=payload_total,
            frame_count=layout.frame_count,
            gop_frames=layout.gop_frames,
            output_fps=layout.fps,
            floor_bytes_per_frame=seek_frame_floor_bytes(
                int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
            ),
            idr_weight=PASSTHROUGH_SEEK_VMP4_FRAMES_IDR_WEIGHT,
            flatten=PASSTHROUGH_SEEK_VMP4_FRAMES_BUDGET_FLATTEN,
        )
    except SourceBudgetError as exc:
        log.info(
            "vmp4 frames source budget unavailable for %s (%s); using flat %dKiB budget",
            Path(path).name, exc, PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES // 1024,
        )
        return None
    except Exception as exc:
        log.warning(
            "vmp4 frames source budget failed for %s: %s: %s",
            Path(path).name, type(exc).__name__, exc,
        )
        return None


def _vmp4_frames_probe_range(
    layout: PassthroughVmp4FramesLayout, rng: ByteRange, *, ranged: bool
) -> str:
    """Classify a range a player issues to inspect the file, not to play it.

    Returns a short reason (used as a diagnostic header) or "" when the range is
    real playback and must wait for encoded frames.
    """
    if rng.end < layout.mdat_payload_start:
        return "init"                   # ftyp+moov only; no media bytes at all
    if not ranged:
        return ""                       # no Range header == sequential playback
    if rng.length > PASSTHROUGH_SEEK_VMP4_FRAMES_PROBE_BYTES:
        return ""
    tail_start = layout.total_size - max(
        PASSTHROUGH_SEEK_VMP4_FRAMES_PROBE_BYTES, layout.mdat_payload_size // 64
    )
    if rng.start >= tail_start:
        return "tail"                   # looking for a trailing moov we do not have
    if rng.start < layout.mdat_payload_start:
        return "header-crossing"        # header read that spills into the first frames
    return ""


def _vmp4_frames_frame_budget(output_mode: str, template: Vmp4SlotOutputTemplate) -> int:
    """Per-frame byte budget for the frames layout.

    The configured value is tuned for Green/Alpha, whose output is either the
    source geometry or a matte-packed picture that codes cheaply. SuperRes
    emits detail-dense frames at an enlarged size, and at 6K that truncated
    real frames (measured: frame 3 at 267030 bytes against a 262144 budget),
    which corrupts the stream the player receives. Scale the budget with output
    pixels above the 4K reference; HTTP bandwidth is budget*fps*8, so this
    raises the bandwidth an enlarged SuperRes stream needs.
    """
    base = max(32 * 1024, int(PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES))
    if str(output_mode or "").lower() != "superres":
        return base
    pixels = max(1, int(template.width) * int(template.height))
    reference = 3840 * 2160
    if pixels <= reference:
        return base
    scaled = int(base * pixels / reference * 1.15)
    scaled = ((scaled + 65535) // 65536) * 65536
    return min(4 * base, scaled)


def _vmp4_frames_cached_layout(path: Path, info, output_mode: str, template: Vmp4SlotOutputTemplate) -> PassthroughVmp4FramesLayout:
    key = _vmp4_frames_layout_key(path, info, output_mode, template)
    with _vmp4_frames_layout_cache_lock:
        cached = _vmp4_frames_layout_cache.get(key)
        if cached is not None:
            _vmp4_frames_layout_cache.pop(key, None)
            _vmp4_frames_layout_cache[key] = cached
            return cached
    audio_table = _vmp4_frames_audio_table(path)
    build = functools.partial(
        build_passthrough_vmp4_frames_layout,
        path,
        audio_table=audio_table,
        duration_sec=float(getattr(info, "duration", 0.0) or 0.0),
        fps=_seek_output_fps(info),
        gop_frames=PASSTHROUGH_GOP,
        idr_budget=_vmp4_frames_frame_budget(output_mode, template),
        p_budget=_vmp4_frames_frame_budget(output_mode, template),
        output_stsd=template.stsd,
        output_codec_name=template.codec_name,
        output_width=template.width,
        output_height=template.height,
        placeholder_payload=template.payload,
    )
    layout = build()
    plan = _vmp4_frames_source_budget(path, layout, output_mode, info)
    if plan is not None:
        # The moov is fixed-width (co64 64-bit, stsz 32-bit, one stts run), so
        # its size does not depend on the budget values - the flat layout above
        # already tells us the exact init size to subtract.
        layout = build(frame_budgets=plan.frame_budgets)
    with _vmp4_frames_layout_cache_lock:
        _vmp4_frames_layout_cache[key] = layout
        while len(_vmp4_frames_layout_cache) > _VMP4_FRAMES_LAYOUT_CACHE_LIMIT:
            _vmp4_frames_layout_cache.pop(next(iter(_vmp4_frames_layout_cache)), None)
    return layout


def _vmp4_frames_budget_fingerprint(layout: PassthroughVmp4FramesLayout) -> str:
    """Short hash of the per-frame budgets, for cache keys."""
    if not layout.frames:
        return "0"
    h = hashlib.sha256()
    for fr in layout.frames:
        h.update(fr.budget.to_bytes(5, "big"))
    return h.hexdigest()[:16]


def _vmp4_frames_digest(path: Path, layout: PassthroughVmp4FramesLayout, output_mode: str) -> str:
    try:
        st = Path(path).stat()
        stat_key = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_key = (0, 0)
    raw = "|".join(str(x) for x in (
        _media_key_path(path), stat_key, str(output_mode or "green").lower(),
        round(float(layout.fps), 6), int(layout.gop_frames), int(layout.frame_count),
        int(layout.idr_budget), int(layout.p_budget), layout.output_stsd_sha256, int(layout.total_size),
        bool(layout.source_budget),
        # idr_budget/p_budget are the flat-mode constants and do not move with a
        # source-derived plan, and total_size is pinned to the source size, so
        # without the actual per-frame budgets a re-planned layout would reuse
        # frames encoded against the previous budgets.
        _vmp4_frames_budget_fingerprint(layout),
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _vmp4_frames_ready_payloads(cache_dir: Path) -> dict[int, Path]:
    res: dict[int, Path] = {}
    try:
        for name in os.listdir(cache_dir):
            if name.startswith("frame_") and name.endswith(".bin"):
                try:
                    idx = int(name[6:-4])
                except ValueError:
                    continue
                res[idx] = cache_dir / name
    except OSError:
        pass
    return res


def _vmp4_frames_gop_ready(layout: PassthroughVmp4FramesLayout, gop_index: int, ready: dict[int, Path]) -> bool:
    indices = layout.gop_frame_indices(gop_index)
    return bool(indices) and all(i in ready for i in indices)


def _vmp4_frames_first_frame_of_gop(layout: PassthroughVmp4FramesLayout, gop_index: int) -> int:
    indices = layout.gop_frame_indices(gop_index)
    return int(indices[0]) if indices else 0


def _vmp4_frames_served_end(
    layout: PassthroughVmp4FramesLayout,
    ready: dict[int, Path],
    start_byte: int,
    req_end: int,
) -> int:
    """Last inclusive byte serveable from already-built frames at ``start_byte``.

    Returns -1 if the frame at ``start_byte`` is not built yet. Bounding a 206 to
    this extent means every byte we promise (Content-Length) is on disk, so the
    body never blocks mid-stream and can never come up short of Content-Length.
    """
    start_idx = vmp4_frame_index_for_offset(layout, int(start_byte))
    if start_idx is None:
        # In the init (ftyp+moov+mdat header) region: serve at least the init,
        # extended into the leading contiguous ready frames from frame 0.
        if not layout.frames or 0 not in ready:
            return min(int(req_end), max(layout.mdat_payload_start - 1, int(start_byte)))
        start_idx = 0
    idx = start_idx
    last = -1
    while idx < layout.frame_count and idx in ready:
        last = idx
        idx += 1
    if last < 0:
        return -1
    end_byte = layout.frames[last].offset + layout.frames[last].budget - 1
    return min(int(req_end), int(end_byte))


def _vmp4_frames_lead_frames(layout: PassthroughVmp4FramesLayout) -> int:
    # How far ahead of the write cursor a requested frame may be and still be
    # reachable by waiting (vs. triggering a filler restart at the seek point).
    return max(int(layout.gop_frames) * 2, int(round(float(layout.fps) * 20.0)))


def _vmp4_frames_filler_worker(
    filler: _Vmp4FramesFiller,
    layout: PassthroughVmp4FramesLayout,
    predecessor: _Vmp4FramesFiller | None = None,
) -> None:
    from pipeline.pynv_stream import iter_pynv_passthrough_annexb_frames

    matter = None
    mode = (filler.output_mode or "green").lower()
    try:
        if mode not in SEEK_FRAME_MODES:
            filler.state = "failed"
            filler.reason = f"unsupported-mode:{mode}"
            return
        if predecessor is not None and not predecessor.done_event.is_set():
            # The run we are replacing still holds a Matter, and the pool only has
            # PT_MAX_CONCURRENT of them. Asking for one now would either take the
            # last slot (starving the next seek) or block here holding nothing
            # useful. Wait for it to actually let go first.
            waited = time.time()
            if not predecessor.done_event.wait(timeout=_VMP4_FRAMES_HANDOVER_TIMEOUT):
                log.warning(
                    "passthrough_seek VMP4 frames filler: predecessor did not release after %.1fs",
                    time.time() - waited,
                )
            else:
                log.info(
                    "passthrough_seek VMP4 frames filler: took over after %.1fs",
                    time.time() - waited,
                )
        if filler.stop_event.is_set():
            filler.state = "stopped"
            return
        matter = acquire_matter(blocking=True, timeout=PASSTHROUGH_SEEK_VMP4_SLOT_MATTER_TIMEOUT)
        if matter is None:
            filler.state = "failed"
            filler.reason = "matter-timeout"
            return
        start_frame = int(filler.start_frame)
        start_sec = start_frame / max(1.0, float(layout.fps))
        remaining = max(1, layout.frame_count - start_frame)
        cap = int(layout.frames[start_frame].budget) if layout.frames else 0
        # Under a source-derived plan the budget varies per GOP, so hand the
        # encoder the whole schedule from this start frame on; it retargets
        # itself with Reconfigure at each change instead of being rebuilt.
        schedule = (
            [int(f.budget) for f in layout.frames[start_frame:]]
            if layout.source_budget else None
        )
        idx = start_frame
        for au in iter_pynv_passthrough_annexb_frames(
            layout.source_path,
            start_sec=start_sec,
            frame_count=remaining,
            matter=matter,
            output_mode=mode,
            per_frame_cap_bytes=cap,
            per_frame_cap_schedule=schedule,
            cancel=filler.stop_event,
        ):
            if filler.stop_event.is_set():
                filler.state = "stopped"
                return
            if idx >= layout.frame_count:
                break
            fr = layout.frames[idx]
            payload = hevc_annexb_to_length_prefixed_sample(au, nal_length_size=layout.nal_length_size)
            target = filler.cache_dir / f"frame_{idx:06d}.bin"
            if not target.exists():
                if len(payload) > fr.budget:
                    # Should be rare with CBR; clamp to the budget (one corrupt
                    # frame -> brief glitch) rather than leaving a permanent hole
                    # that 503s the whole GOP forever.
                    log.warning(
                        "passthrough_seek VMP4 frames oversize idx=%d %d>%d (clamped)",
                        idx, len(payload), fr.budget,
                    )
                    payload = payload[: fr.budget]
                _vmp4_write_bytes_atomic(target, payload)
            if not filler.first_frame_at:
                filler.first_frame_at = time.time()
            filler.cursor = idx + 1
            idx += 1
        if filler.stop_event.is_set():
            # Cancelled during setup or between frames: the generator returns
            # normally in that case, so without this check the tail-fill below
            # would mark every remaining frame of the title as an empty payload
            # and the player would be served filler for the rest of the run.
            filler.state = "stopped"
            return
        # Reaching here means the encode generator ran dry rather than being
        # preempted (a stop request returns above). frame_count comes from
        # duration*fps and can overshoot what the decoder actually yields - one
        # frame short is normal - so the trailing samples would never be written
        # and their GOP would never count as ready. Every request for that GOP
        # then restarted the filler, forever: a 6s 8K pipeline rebuild on a loop,
        # and a 503 for anyone playing to the end of the title. Leave
        # zero-length payloads instead; the range iterator serves those samples
        # as filler, which is what they are.
        for missing in range(idx, layout.frame_count):
            tail_target = filler.cache_dir / f"frame_{missing:06d}.bin"
            if not tail_target.exists():
                _vmp4_write_bytes_atomic(tail_target, b"")
        if idx < layout.frame_count:
            log.info(
                "passthrough_seek VMP4 frames filler ran dry at %d/%d; %d trailing samples left as filler",
                idx, layout.frame_count, layout.frame_count - idx,
            )
        filler.cursor = layout.frame_count
        filler.state = "done"
    except Exception as exc:
        filler.state = "failed"
        filler.reason = f"{type(exc).__name__}: {exc}"
        log.warning("passthrough_seek VMP4 frames filler failed: start=%d reason=%s", filler.start_frame, filler.reason)
    finally:
        release_matter(matter)
        filler.done_event.set()


def _vmp4_frames_ensure_filler(
    layout: PassthroughVmp4FramesLayout,
    cache_dir: Path,
    digest: str,
    output_mode: str,
    gop_index: int,
) -> _Vmp4FramesFiller | None:
    if not PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    need_frame = _vmp4_frames_first_frame_of_gop(layout, gop_index)
    lead = _vmp4_frames_lead_frames(layout)
    now = time.time()
    with _vmp4_frames_lock:
        filler = _vmp4_frames_fillers.get(digest)
        covered = (
            filler is not None
            and filler.state == "running"
            and filler.start_frame <= need_frame <= filler.cursor + lead
        )
        if covered:
            filler.last_hit_at = now
            return filler
        # Debounce: a running filler that simply hasn't reached this GOP yet keeps
        # running (the request waits/503s and retries) until the cooldown lapses.
        # This stops concurrent probes at different offsets from ping-ponging the
        # single filler so it can actually make forward progress. It only applies
        # while something is still reading from where the filler is: a genuine
        # seek leaves it with no readers, and then it must move at once or the
        # player waits out its timeout on a position nobody is producing.
        if filler is not None and filler.state == "running":
            # A seek is not a competing probe and must not wait the debounce out.
            # The debounce exists for connections the player opens AHEAD of
            # itself, which are always close by; a request thousands of frames
            # away, or behind a run that only moves forward, can only be a seek.
            # Distance says so, and "nobody is reading the old position" does
            # not: the connection the player abandoned keeps draining for a
            # second or two afterwards and holds last_hit_at fresh, so a real
            # seek sat out the full follow window. Measured on 8K, 11 of 18
            # cold starts spent 2.0-2.7s here before the encoder was even
            # allowed to move - the bulk of the stall where the player still
            # shows the previous picture.
            seeked_away = (
                need_frame < filler.start_frame
                or need_frame > filler.cursor + 2 * lead
            )
            followed = (
                not seeked_away
                and (now - (filler.last_hit_at or filler.started_at)) < _VMP4_FRAMES_FOLLOW_WINDOW
            )
            if filler.first_frame_at:
                hold, since = _VMP4_FRAMES_PRODUCTIVE_HOLD, filler.first_frame_at
            else:
                hold, since = _VMP4_FRAMES_RESTART_COOLDOWN, filler.started_at
            if followed and (now - since) < hold:
                return filler
        # (Re)start a forward filler at the requested GOP. Already-written frame
        # files from earlier runs stay valid; the worker skips existing files.
        predecessor = None
        if filler is not None and filler.state == "running":
            filler.stop_event.set()
            predecessor = filler
        # Runs for OTHER titles are not stopped by the reposition above; without
        # this they keep encoding to the end of a title nobody is watching and
        # hold their Matter, so the pool runs out and the next title never
        # starts. Retire the least recently read ones down to the limit.
        others = [
            (d, f) for d, f in _vmp4_frames_fillers.items()
            if d != digest and f.state == "running"
        ]
        if others:
            others.sort(key=lambda item: item[1].last_hit_at or item[1].started_at)
            for other_digest, other in others[: max(0, len(others) - (PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_ACTIVE - 1))]:
                other.stop_event.set()
                log.info(
                    "passthrough_seek VMP4 frames retiring idle filler: digest=%s idle=%.1fs",
                    other_digest[:8], now - (other.last_hit_at or other.started_at),
                )
                if predecessor is None:
                    predecessor = other
        new = _Vmp4FramesFiller(
            digest=digest,
            cache_dir=cache_dir,
            output_mode=(output_mode or "green").lower(),
            start_frame=int(need_frame),
            cursor=int(need_frame),
            state="running",
            started_at=now,
        )
        _vmp4_frames_fillers[digest] = new
        threading.Thread(
            target=_vmp4_frames_filler_worker,
            args=(new, layout, predecessor),
            name=f"vmp4-frames-filler-{digest[:8]}",
            daemon=True,
        ).start()
        return new


def _vmp4_frames_wait_gop(
    layout: PassthroughVmp4FramesLayout,
    cache_dir: Path,
    digest: str,
    gop_index: int,
    output_mode: str,
    timeout_sec: float,
) -> tuple[bool, dict[int, Path], _Vmp4FramesFiller | None]:
    deadline = time.time() + max(0.0, float(timeout_sec or 0.0))
    filler: _Vmp4FramesFiller | None = None
    while True:
        ready = _vmp4_frames_ready_payloads(cache_dir)
        if _vmp4_frames_gop_ready(layout, gop_index, ready):
            return True, ready, filler
        filler = _vmp4_frames_ensure_filler(layout, cache_dir, digest, output_mode, gop_index)
        now = time.time()
        if filler is None or filler.state == "failed" or now >= deadline:
            return False, ready, filler
        time.sleep(min(0.05, max(0.001, deadline - now)))


def _vmp4_frames_streaming(body, digest: str):
    """Mark the run as being read while its bytes are going out.

    Whether a filler may be repositioned turns on how recently a request landed
    inside the range it covers, but that was only recorded when a request
    ARRIVED. Once a reply started streaming nothing touched it again, so about
    _VMP4_FRAMES_FOLLOW_WINDOW into playback the run looked abandoned and the
    player's next connection - opened ahead, as they do - was free to drag it
    elsewhere. The reply being read at that moment lost its producer mid-stream:
    picture for a second or two after a seek, then nothing.
    """
    for chunk in body:
        filler = _vmp4_frames_fillers.get(digest)
        if filler is not None:
            filler.last_hit_at = time.time()
        yield chunk


def _vmp4_frames_body_waiter(cache_dir: Path, timeout: float, digest: str):
    """Callback letting a response body block on a frame the encoder owes it.

    Giving up produces a lone filler-data NAL - an access unit with no VCL NAL,
    which no conforming decoder accepts - so this waits as long as the run is
    still producing rather than padding after a short deadline. It returns None
    (and the caller emits filler) only when the run has stopped or failed, where
    the frame is genuinely never coming.
    """
    if timeout <= 0:
        return None

    def wait(index: int) -> Path | None:
        target = cache_dir / f"frame_{int(index):06d}.bin"
        deadline = time.time() + timeout
        while True:
            if target.exists():
                return target
            filler = _vmp4_frames_fillers.get(digest)
            if filler is None or filler.state != "running":
                return None
            if time.time() >= deadline:
                log.warning(
                    "passthrough_seek VMP4 frames body wait exhausted: frame=%d after %.1fs",
                    index, timeout,
                )
                return None
            time.sleep(0.02)

    return wait


def _vmp4_frames_wait_frame(
    layout: PassthroughVmp4FramesLayout,
    cache_dir: Path,
    digest: str,
    frame_index: int,
    output_mode: str,
    timeout_sec: float,
    want_frames: int = 1,
) -> tuple[bool, dict[int, Path], _Vmp4FramesFiller | None]:
    """Wait for a contiguous run of frames from ``frame_index``.

    A 206 is trimmed to whatever contiguous prefix is built, so waiting for the
    whole GOP is unnecessary - but waiting for just one frame is too little. A
    player that has just seeked then receives a fraction of a second of video,
    plays it, finds nothing after it and moves to the next item. Asking for a
    GOP's worth of frames costs about a second and answers with a few MB.
    """
    deadline = time.time() + max(0.0, float(timeout_sec or 0.0))
    gop_index = layout.frames[frame_index].gop_index if 0 <= frame_index < len(layout.frames) else 0
    need = max(1, int(want_frames or 1))
    filler: _Vmp4FramesFiller | None = None
    while True:
        ready = _vmp4_frames_ready_payloads(cache_dir)
        run, idx = 0, frame_index
        while idx < layout.frame_count and idx in ready and run < need:
            run += 1
            idx += 1
        # Short of `need` only because the title ends there is still enough.
        if run >= need or (run > 0 and idx >= layout.frame_count):
            return True, ready, filler
        filler = _vmp4_frames_ensure_filler(layout, cache_dir, digest, output_mode, gop_index)
        now = time.time()
        if filler is None or filler.state == "failed" or now >= deadline:
            return run > 0, ready, filler
        # A filler only ever encodes forward. If another request has since
        # repositioned it past (or away from) this frame, nothing will ever
        # produce it and waiting here is waiting forever - answer with whatever
        # exists. Players that fire two requests at one offset and then a third
        # slightly later (OPlayer does) otherwise leave the first two hanging.
        if filler.state == "running" and filler.start_frame > frame_index:
            log.info(
                "passthrough_seek VMP4 frames wait abandoned: frame=%d filler_start=%d",
                frame_index, filler.start_frame,
            )
            return run > 0, ready, filler
        time.sleep(min(0.05, max(0.001, deadline - now)))


def _iter_vmp4_frames_ready_stream(
    layout: PassthroughVmp4FramesLayout,
    cache_dir: Path,
    digest: str,
    output_mode: str,
    start: int,
    end_inclusive: int,
    *,
    chunk_size: int,
    wait_timeout: float,
    rid: int = 0,
):
    """Stream frame bytes, building/waiting each GOP before it is emitted.

    This is the realtime background-encode-ahead behaviour: as the player pulls
    bytes the stream ensures the current GOP is built (and prefetches ahead via
    _vmp4_frames_wait_gop) instead of emitting placeholder bytes. Runs inside the
    StreamingResponse threadpool, so the per-GOP waits do not block the loop.
    """
    cursor = max(0, int(start))
    stop = min(max(0, int(end_inclusive)), max(0, layout.total_size - 1))
    if cursor > stop:
        return
    init_end = len(layout.init)
    if cursor < init_end:
        init_stop = min(stop, init_end - 1)
        yield layout.init[cursor : init_stop + 1]
        cursor = init_stop + 1
    if cursor > stop:
        return
    start_idx = vmp4_frame_index_for_offset(layout, cursor)
    if start_idx is None:
        start_idx = 0
    last_gop = -1
    ready: dict[int, Path] = {}
    for fr in layout.frames[start_idx:]:
        frame_end = fr.offset + fr.budget
        if frame_end <= cursor:
            continue
        if fr.offset > stop:
            break
        if fr.gop_index != last_gop:
            ok, ready, _build = _vmp4_frames_wait_gop(
                layout, cache_dir, digest, fr.gop_index, output_mode, wait_timeout
            )
            last_gop = fr.gop_index
            if not ok:
                log.warning(
                    "passthrough_seek[%d] VMP4 frames stream wait failed: gop=%d at byte=%d",
                    rid, fr.gop_index, cursor,
                )
                return
        sub_end = min(stop, frame_end - 1)
        yield from iter_vmp4_frames_range(
            layout, cursor, sub_end, chunk_size=chunk_size, payload_paths=ready
        )
        cursor = sub_end + 1
        if cursor > stop:
            return


def _vmp4_frames_cap_to_reach(
    layout: PassthroughVmp4FramesLayout,
    build: _Vmp4FramesFiller | None,
    start_byte: int,
    served_end: int,
) -> int:
    """Keep a reply's end inside the range the running encode can still reach.

    A player asks for the byte after the reply it just consumed, so the end of
    every reply picks the start of the next request. The span was measured from
    the request start (up to _MAX_SPAN_BYTES, ~450 frames) while whether a run
    can serve an offset is measured from its write cursor (``cursor + lead``).
    Two different origins: each reply handed the player a next-offset a few
    hundred frames further ahead than the encoder had moved, so after a handful
    of replies the player was asking past ``cursor + lead`` - and that request
    does not wait, it repositions the run. A ~3s pipeline rebuild, and every body
    still streaming from the old position loses its producer. That is the loop
    behind "seek, watch a second or two, jump to the next video".

    Capping the end at the last frame the run still covers means the next request
    always lands inside the window, so it is answered by waiting (milliseconds)
    instead of by restarting. A seek into fresh ground is untouched: there the
    cursor sits at the request, and ``lead`` is wider than _MAX_SPAN_BYTES.
    """
    if build is None or build.state != "running" or not layout.frames:
        return served_end
    reach = int(build.cursor) + _vmp4_frames_lead_frames(layout)
    reach = min(reach, layout.frame_count - 1)
    if reach < 0:
        return served_end
    fr = layout.frames[reach]
    limit = fr.offset + fr.budget - 1
    # Never cap below the request itself - an offset already out of reach needs
    # the reposition, and trimming it to nothing would only answer 503.
    return served_end if limit < start_byte else min(served_end, limit)


def _seek_vmp4_frames_response(
    *,
    path: Path,
    info,
    output_mode: str,
    range_header: str | None,
    headers: dict[str, str],
    head_only: bool,
    rid: int = 0,
) -> Response:
    headers["Accept-Ranges"] = "bytes"
    headers["Content-Type"] = "video/mp4"
    headers["X-Passthrough-VMP4-Backend"] = "slot_frames"
    mode = (output_mode or "green").lower()
    try:
        template = _vmp4_slot_output_template(info, mode)
        layout = _vmp4_frames_cached_layout(path, info, mode, template)
    except Vmp4SlotLayoutError as exc:
        eh = dict(headers)
        eh["X-Passthrough-VMP4-Phase"] = "frames-layout-error"
        eh["X-Passthrough-VMP4-Frames-Error"] = _header_safe_text(str(exc))
        for k in ("Content-Length", "Content-Range", "Content-Type"):
            eh.pop(k, None)
        return Response("VMP4 frames layout unavailable", status_code=409, headers=eh, media_type="text/plain")

    total = layout.total_size
    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None:
        eh = dict(headers)
        eh["X-Passthrough-VMP4-Phase"] = "range-unsatisfiable"
        eh["Content-Range"] = f"bytes */{max(0, int(total))}"
        for k in ("Content-Length", "Content-Type"):
            eh.pop(k, None)
        log.info(
            "passthrough_seek[%d] VMP4 frames 416: range=%r total=%d", rid, range_header, total
        )
        return Response(status_code=416, headers=eh)

    digest = _vmp4_frames_digest(path, layout, mode)
    cache_dir = RUNTIME_CACHE_DIR / "vmp4_frames" / digest
    response_range = byte_range or ByteRange(start=0, end=max(0, total - 1), total=total)
    # "bytes=0-" is answered like a static file would answer it: 206 with a full
    # Content-Range, so the player learns the total size. Serving it as a 200
    # chunked body (no Content-Length, the live-path shape) left Skybox without a
    # size; it then probed bytes=<total>- , got the 416 that offset deserves, and
    # gave up on the item. A real file survives that same probe because the first
    # response already told it how big the file is. Only a request with no Range
    # header at all keeps the chunked shape.
    status_code = 206 if byte_range is not None else 200

    headers["X-Passthrough-VMP4-Phase"] = "frames-layout"
    headers["X-Passthrough-VMP4-Output-Codec"] = layout.codec_name
    headers["X-Passthrough-VMP4-Output-Size"] = f"{template.width}x{template.height}"
    headers["X-Passthrough-VMP4-Frames-Count"] = str(layout.frame_count)
    headers["X-Passthrough-VMP4-Frames-Fps"] = f"{layout.fps:.3f}"
    headers["X-Passthrough-VMP4-Frames-Gop"] = str(layout.gop_frames)
    headers["X-Passthrough-VMP4-Frames-Budget"] = str(layout.idr_budget)
    headers["X-Passthrough-VMP4-Frames-Source-Budget"] = "1" if layout.source_budget else "0"
    headers["X-Passthrough-VMP4-Frames-Audio"] = (
        f"{len(layout.audio_samples)}/{layout.audio_bytes}" if layout.audio_samples else "none"
    )
    if layout.source_budget and layout.frames:
        budgets = [f.budget for f in layout.frames]
        headers["X-Passthrough-VMP4-Frames-Budget-Range"] = (
            f"{min(budgets)}-{max(budgets)}/{layout.mdat_payload_size // max(1, layout.frame_count)}"
        )
    headers["X-Passthrough-VMP4-Moov-Size"] = str(layout.moov_size)
    headers["X-Passthrough-VMP4-Cache"] = digest

    ready: dict[int, Path] = _vmp4_frames_ready_payloads(cache_dir)
    if mode == "superres" and not seek_supported_target(RTX_VSR_TARGET_HEIGHT):
        # Only native 1x SuperRes is offered as a virtual file; anything else
        # belongs to the live chapter container and must not be half-served.
        eh = dict(headers)
        eh["X-Passthrough-VMP4-Phase"] = "superres-target-not-seekable"
        for k in ("Content-Length", "Content-Range", "Content-Type"):
            eh.pop(k, None)
        return Response(
            "SuperRes virtual-file playback requires the native 1x target",
            status_code=409, headers=eh, media_type="text/plain",
        )
    if mode not in SEEK_FRAME_MODES:
        eh = dict(headers)
        eh["X-Passthrough-VMP4-Phase"] = "frames-mode-unsupported"
        for k in ("Content-Length", "Content-Range", "Content-Type"):
            eh.pop(k, None)
        return Response(f"VMP4 frames backend does not support mode={mode}", status_code=409, headers=eh, media_type="text/plain")

    if head_only:
        # HEAD: declare full size, no body, no build wait.
        headers["Content-Length"] = str(response_range.length)
        if status_code == 206:
            headers["Content-Range"] = f"bytes {response_range.start}-{response_range.end}/{total}"
        else:
            headers.pop("Content-Range", None)
        return Response(status_code=status_code, headers=headers, media_type="video/mp4")

    start_idx = vmp4_frame_index_for_offset(layout, response_range.start)
    gop_index = layout.frames[start_idx].gop_index if start_idx is not None else 0

    # MP4 players open a file by reading its header and probing its tail before
    # they play anything. Those bytes are structural: the header lives in our
    # init and the tail bytes are samples the player only inspects, so answering
    # them with 503 (waiting on an encode that will never matter) is what makes
    # a player give up before the first frame. Serve any such probe straight
    # from the layout - real bytes where we have them, filler where we do not.
    probe = _vmp4_frames_probe_range(layout, response_range, ranged=status_code == 206)
    if probe:
        headers["X-Passthrough-VMP4-Frames-Probe"] = probe
        headers["X-Passthrough-VMP4-Frames-Gop-Index"] = str(gop_index)
        ready = _vmp4_frames_ready_payloads(cache_dir)
        log.info(
            "passthrough_seek[%d] VMP4 frames probe (%s): range=%r len=%d",
            rid, probe, range_header, response_range.length,
        )
        body = iter_vmp4_frames_range(
            layout, response_range.start, response_range.end,
            chunk_size=DEFAULT_CHUNK_SIZE, payload_paths=ready,
        )
        headers["Content-Length"] = str(response_range.length)
        if status_code == 206:
            headers["Content-Range"] = f"bytes {response_range.start}-{response_range.end}/{total}"
        else:
            headers.pop("Content-Range", None)
        return StreamingResponse(body, status_code=status_code, headers=headers, media_type="video/mp4")

    t_wait0 = time.time()
    if status_code == 206 and start_idx is not None:
        # Ranged: the body is trimmed to the built prefix anyway, so one frame is
        # enough to answer. Whole-GOP waits belong to the chunked path below.
        ok, ready, build = _vmp4_frames_wait_frame(
            layout, cache_dir, digest, start_idx, mode,
            PASSTHROUGH_SEEK_VMP4_FRAMES_RANGED_WAIT,
            want_frames=PASSTHROUGH_SEEK_VMP4_FRAMES_PREBUFFER_FRAMES or 1,
        )
        if not ok and build is not None and build.state == "running":
            # The encoder is working towards this position but has not reached
            # it. Answering 503 makes a player abandon the item; answer with
            # filler for the part that is not encoded and let it keep playing
            # while the encoder closes the gap.
            ok = True
    else:
        ok, ready, build = _vmp4_frames_wait_gop(
            layout, cache_dir, digest, gop_index, mode, PASSTHROUGH_SEEK_VMP4_FRAMES_READY_WAIT
        )
    headers["X-Passthrough-VMP4-Frames-Gop-Index"] = str(gop_index)
    headers["X-Passthrough-VMP4-Frames-Wait-Ready"] = "1" if ok else "0"
    if build is not None:
        headers["X-Passthrough-VMP4-Frames-Build"] = build.state
        if build.reason:
            headers["X-Passthrough-VMP4-Frames-Build-Reason"] = _header_safe_text(build.reason)
    if not ok:
        eh = dict(headers)
        eh["X-Passthrough-VMP4-Phase"] = "frames-gop-not-ready"
        eh["Retry-After"] = "1"
        for k in ("Content-Length", "Content-Range", "Content-Type"):
            eh.pop(k, None)
        log.info(
            "passthrough_seek[%d] VMP4 frames gop not ready: gop=%d range=%r waited=%.1fs",
            rid, gop_index, range_header, time.time() - t_wait0,
        )
        return Response("VMP4 frames GOP not ready", status_code=503, headers=eh, media_type="text/plain")

    if status_code == 206:
        # Serve only the already-built contiguous prefix so Content-Length is
        # exact and the body never blocks mid-stream (which previously raised
        # "Response content shorter than Content-Length"). The player re-requests
        # the next byte range; the filler keeps building ahead.
        served_end = _vmp4_frames_served_end(layout, ready, response_range.start, response_range.end)
        if build is not None and build.state == "running":
            # A reply is not limited to what is already encoded. Trimming it there
            # made a seek into un-encoded ground answer with whatever few MB
            # existed - 7.9 MB where the same player had been receiving 50 MB per
            # reply over cached ground - so it came back for more within a second
            # and had to wait, which is when it gives up and moves on.
            #
            # The body can block on frames the run has not reached yet (it is
            # producing above realtime, so they are milliseconds away), so promise
            # the same span here as anywhere else and let it deliver at the rate
            # frames appear. Only the reply's pace differs between cached and
            # un-cached ground, not its shape.
            served_end = max(served_end, response_range.start - 1)
        if PASSTHROUGH_SEEK_VMP4_FRAMES_MIN_SPAN_BYTES > 0 and served_end >= response_range.start - 1:
            served_end = max(
                served_end,
                min(
                    response_range.end,
                    response_range.start + PASSTHROUGH_SEEK_VMP4_FRAMES_MIN_SPAN_BYTES - 1,
                ),
            )
        if build is not None and build.state == "running" and PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_SPAN_BYTES > 0:
            served_end = max(
                served_end,
                min(
                    response_range.end,
                    response_range.start + PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_SPAN_BYTES - 1,
                ),
            )
        if PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_SPAN_BYTES > 0:
            served_end = min(
                served_end,
                response_range.start + PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_SPAN_BYTES - 1,
            )
        served_end = _vmp4_frames_cap_to_reach(layout, build, response_range.start, served_end)
        if served_end < response_range.start:
            eh = dict(headers)
            eh["X-Passthrough-VMP4-Phase"] = "frames-gop-not-ready"
            eh["Retry-After"] = "1"
            for k in ("Content-Length", "Content-Range", "Content-Type"):
                eh.pop(k, None)
            return Response("VMP4 frames range not ready", status_code=503, headers=eh, media_type="text/plain")
        response_range = ByteRange(start=response_range.start, end=served_end, total=total)
        headers["Content-Length"] = str(response_range.length)
        headers["Content-Range"] = f"bytes {response_range.start}-{response_range.end}/{total}"
        log.info(
            "passthrough_seek[%d] VMP4 frames 206: gop=%d range=%r content_range=%s",
            rid, gop_index, range_header, headers.get("Content-Range"),
        )
        body = iter_vmp4_frames_range(
            layout, response_range.start, response_range.end,
            chunk_size=DEFAULT_CHUNK_SIZE, payload_paths=ready,
            wait_for_frame=_vmp4_frames_body_waiter(
                cache_dir, PASSTHROUGH_SEEK_VMP4_FRAMES_BODY_FRAME_WAIT, digest
            ),
        )
        return StreamingResponse(
            _vmp4_frames_streaming(body, digest),
            status_code=206, headers=headers, media_type="video/mp4",
        )

    # No-Range / zero-open playback: stream chunked (no Content-Length, like the
    # live path) and block per GOP as needed. Without a declared length the body
    # can stall when the encoder is briefly behind without erroring.
    headers.pop("Content-Length", None)
    headers.pop("Content-Range", None)
    log.info(
        "passthrough_seek[%d] VMP4 frames 200 chunked: mode=%s output=%s total=%d frames=%d fps=%.3f range=%r",
        rid, mode, headers.get("X-Passthrough-VMP4-Output-Size"), total,
        layout.frame_count, layout.fps, range_header,
    )
    body = _iter_vmp4_frames_ready_stream(
        layout, cache_dir, digest, mode,
        response_range.start, response_range.end,
        chunk_size=DEFAULT_CHUNK_SIZE,
        wait_timeout=PASSTHROUGH_SEEK_VMP4_FRAMES_READY_WAIT,
        rid=rid,
    )
    return StreamingResponse(body, status_code=200, headers=headers, media_type="video/mp4")


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
            if mode in {"green", "alpha", "two_dvr", "superres", "dlss5"} and mode not in out:
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
    # 2D->3D live only runs on the GPU NV12/P016 (PyNv) path; sources the backend
    # would route to the ffmpeg fallback (VFR, unsupported codec/pixel format)
    # cannot be served and must be rejected cleanly instead of failing at startup.
    try:
        decision = select_backend(meta.timing, meta.codec, meta.color)
    except Exception as e:
        return f"2D->3D live source probe failed: {e}"
    if decision.verdict != "pynv_hevc":
        return f"2D->3D live requires the GPU NV12 path (source ineligible: {decision.reason})"
    return ""


def _rm_live_block_reason(path: Path, meta) -> str:
    """RM (realtime mosaic restoration) live runs on the GPU NV12 (PyNv) path.

    2D and VR are both supported: realtime RM runs the same GpuRmProcessor as
    offline, which reprojects each VR region to a flat view and back. This gate
    has to stay in step with ``_rm_dlna_enabled`` in dlna/content_directory.py --
    when the two disagreed, the [RM] entry was offered for VR sources and then
    rejected at playback, and SKYBOX answered the 409 by silently re-requesting
    the same item with no mode and no seek, i.e. alpha from 00:00."""
    from utils.runtime_settings import get_rm

    if not get_rm().enabled:
        return "mosaic restoration is disabled"
    try:
        decision = select_backend(meta.timing, meta.codec, meta.color)
    except Exception as e:
        return f"RM live source probe failed: {e}"
    if decision.verdict != "pynv_hevc":
        return f"RM live requires the GPU NV12 path (source ineligible: {decision.reason})"
    return ""


def _face_beauty_live_block_reason(path: Path, meta) -> str:
    """Keep the FaceBeauty playback gate aligned with its DLNA listing gate."""
    from utils.runtime_settings import get_face_beauty

    if not get_face_beauty().enabled:
        return "face beautification is disabled"
    try:
        decision = select_backend(meta.timing, meta.codec, meta.color)
    except Exception as e:
        return f"face beauty live source probe failed: {e}"
    if decision.verdict != "pynv_hevc":
        return f"face beauty live requires the GPU NV12 path (source ineligible: {decision.reason})"
    return ""


def _live_first_chunk_timeout(output_mode: str) -> float:
    if output_mode == "face_beauty":
        return FACE_BEAUTY_LIVE_FIRST_CHUNK_TIMEOUT_SEC
    return _LIVE_FIRST_CHUNK_TIMEOUT_SEC


async def _superres_live_block_reason(path: Path, meta) -> str:
    if not RTX_VSR_REALTIME_ENABLED:
        return "RTX VSR realtime is disabled"
    from utils.rtx_vsr import evaluation_preflight_status, run_evaluation_preflight, source_block_reason

    codec = meta.codec
    is_vr = has_vr_filename_marker(path.stem) or is_half_equirectangular_source(codec.width, codec.height)
    if is_vr and int(codec.width or 0) > 4096:
        return "8K VR sources are hidden from realtime SuperRes"
    reason = source_block_reason(
        codec.width,
        codec.height,
        is_vr=is_vr,
        is_10bit=bool(int(getattr(codec, "bit_depth", 8) or 8) > 8),
        allow_vr=True,
    )
    if reason:
        return reason
    preflight = evaluation_preflight_status()
    if preflight is None or not preflight.get("ok"):
        # NGX first-run evaluation may take up to the configured timeout. Do
        # not run subprocess.run() on FastAPI's event-loop thread.
        preflight = await asyncio.to_thread(run_evaluation_preflight)
    if preflight is not None and not preflight.get("ok"):
        return f"evaluate preflight failed: {preflight.get('reason', 'unknown')}"
    return ""


async def _dlss5_live_block_reason(path: Path, meta) -> str:
    """Why this source cannot ride the realtime DLSS5 channel, or "".

    Unlike SuperRes there is no NGX evaluation preflight to run: the CUDA
    zero-copy path fails loudly on its first frame, and what it needs from the
    process (a blocking-sync primary context) is decided at startup, so
    source_block_reason_dlss5 can answer without touching the GPU.
    """
    from utils.dlss5 import source_block_reason_dlss5

    codec = meta.codec
    timing = getattr(meta, "timing", None)
    return source_block_reason_dlss5(
        int(codec.width or 0),
        int(codec.height or 0),
        is_10bit=bool(int(getattr(codec, "bit_depth", 8) or 8) > 8),
        fps=float(getattr(timing, "source_fps", 0.0) or 0.0),
        realtime=True,
    ) or ""


# Modes that put a GPU stage between the decoder and NVENC, and the name each
# one answers a rejected client with.
_ENHANCED_MODE_LABELS = {"superres": "RTX VSR", "dlss5": "DLSS5"}


async def _enhanced_mode_block_reason(path: Path, mode: str, meta=None) -> str:
    """Why an enhancement mode cannot run for this source, or "".

    Every route that can start one of these stages asks here, so a route cannot
    admit a source the worker then refuses - which is exactly how SuperRes once
    came back as a 409 after the listing had already offered it.
    """
    name = str(mode or "").strip().lower()
    if name not in _ENHANCED_MODE_LABELS:
        return ""
    try:
        if meta is None:
            meta = await asyncio.to_thread(probe_video_metadata, path)
        if name == "superres":
            return await _superres_live_block_reason(path, meta)
        return await _dlss5_live_block_reason(path, meta)
    except Exception as exc:
        return f"metadata probe failed: {exc}"


def _enhanced_mode_label(mode: str) -> str:
    return _ENHANCED_MODE_LABELS.get(str(mode or "").strip().lower(), str(mode or "").upper())


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
    elif output_mode not in {"green", "alpha", "two_dvr", "rm", "superres", "dlss5", "face_beauty"}:
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
        if output_mode == "rm":
            raise RuntimeError("RM live requires the PyNv NV12 live path")
        if output_mode == "face_beauty":
            raise RuntimeError("face beauty live requires the PyNv NV12 live path")
        if output_mode == "superres":
            raise RuntimeError("RTX VSR live requires the PyNv CUDA path")
        if output_mode == "dlss5":
            raise RuntimeError("DLSS5 live requires the PyNv CUDA path")
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
    # RM (realtime mosaic restoration) is a runtime-toggled mode independent of
    # the server-configured passthrough modes, so it bypasses _select_live_output_mode.
    if requested_mode in {"rm", "face_beauty"}:
        # Both are runtime-toggled modes independent of the server-configured
        # passthrough modes, so they bypass _select_live_output_mode.
        live_output_mode = requested_mode
    else:
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
    if live_output_mode == "rm":
        rm_block_reason = _rm_live_block_reason(path, live_meta)
        if rm_block_reason:
            log.info(
                "passthrough_live[%d] reject unsupported RM live source: %s reason=%s",
                rid,
                path.name,
                rm_block_reason,
            )
            return Response(f"RM live unsupported: {rm_block_reason}", status_code=409)
    if live_output_mode == "face_beauty":
        face_beauty_block_reason = _face_beauty_live_block_reason(path, live_meta)
        if face_beauty_block_reason:
            log.info(
                "passthrough_live[%d] reject unsupported FaceBeauty live source: %s reason=%s",
                rid,
                path.name,
                face_beauty_block_reason,
            )
            return Response(
                f"FaceBeauty live unsupported: {face_beauty_block_reason}",
                status_code=409,
            )
    if live_output_mode in _ENHANCED_MODE_LABELS:
        enhanced_block_reason = await _enhanced_mode_block_reason(path, live_output_mode, live_meta)
        if enhanced_block_reason:
            label = _enhanced_mode_label(live_output_mode)
            log.info(
                "passthrough_live[%d] reject unsupported %s source: %s reason=%s",
                rid, label, path.name, enhanced_block_reason,
            )
            return Response(f"{label} live unsupported: {enhanced_block_reason}", status_code=409)
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
        str(safe_resolve_path(path)),
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
            matter = None if live_output_mode in {"two_dvr", "rm", "face_beauty"} else acquire_matter(blocking=True, timeout=live_acquire_timeout)
            if matter is None:
                if live_output_mode not in {"two_dvr", "rm", "face_beauty"}:
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
    first_chunk_timeout = _live_first_chunk_timeout(live_output_mode)

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
                    timeout=first_chunk_timeout,
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
                first_chunk_timeout,
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
                timeout=first_chunk_timeout,
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
            first_chunk_timeout,
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


def _seek_output_mode(requested_mode: str | None, route_mode: str | None = None) -> str:
    """Resolve the seek route's output mode.

    The path-borne mode wins: it survives a player dropping the query.
    """
    requested = (str(route_mode or "") or str(requested_mode or "")).lower()
    modes = tuple(
        mode for mode in _configured_passthrough_modes()
        if mode in {"green", "alpha", "superres", "dlss5"}
    )
    if requested in modes:
        return requested
    # A single enhancement mode is the whole configuration, so it is also the
    # fallback; with green or alpha alongside it the plain modes win, because an
    # enhancement stage is the expensive answer to give a client that asked for
    # nothing in particular.
    if len(modes) == 1 and modes[0] in {"superres", "dlss5"}:
        return modes[0]
    if "green" in modes and "alpha" in modes:
        return "green"
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
    path, route_container, route_mode = _safe_seek_video_path(name)
    user_agent = request.headers.get("user-agent", "")
    allowed, reason, route_profile = _seek_route_allowed(user_agent)
    output_mode = _seek_output_mode(mode, route_mode)
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
    if output_mode in _ENHANCED_MODE_LABELS:
        seek_reason = await _enhanced_mode_block_reason(path, output_mode)
        if seek_reason:
            return Response(
                f"{_enhanced_mode_label(output_mode)} seek unsupported: {seek_reason}",
                status_code=409,
            )
    codec = PYNV_OUTPUT_CODEC
    client_host = request.client.host if request.client else ""
    container = _seek_container() if PASSTHROUGH_SEEK_VMP4 else (route_container or _seek_container())
    media_type = _seek_media_type(container)
    total = _seek_declared_total(path, info.duration, codec, client_host, container)
    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None and not _seek_vmp4_enabled(container):
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
    _apply_seek_vmp4_headers(headers, path=path, duration=info.duration, total=total, container=container)
    if _seek_vmp4_enabled(container):
        if _seek_vmp4_backend() == "slot_frames":
            return _seek_vmp4_frames_response(
                path=path,
                info=info,
                output_mode=output_mode,
                range_header=range_header,
                headers=headers,
                head_only=True,
                rid=rid,
            )
        if _seek_vmp4_backend() == "slot":
            return _seek_vmp4_slot_response(
                path=path,
                info=info,
                output_mode=output_mode,
                total=total,
                range_header=range_header,
                headers=headers,
                head_only=True,
                rid=rid,
            )
        return _seek_vmp4_cache_response(
            path=path,
            info=info,
            output_mode=output_mode,
            total=total,
            range_header=range_header,
            headers=headers,
            head_only=True,
            rid=rid,
        )
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
    _t_req_start = time.time()
    path, route_container, route_mode = _safe_seek_video_path(name)
    user_agent = request.headers.get("user-agent", "")
    accept = request.headers.get("accept", "")
    allowed, reason, route_profile = _seek_route_allowed(user_agent)
    output_mode = _seek_output_mode(mode, route_mode)
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
    if output_mode in _ENHANCED_MODE_LABELS:
        seek_reason = await _enhanced_mode_block_reason(path, output_mode)
        if seek_reason:
            return Response(
                f"{_enhanced_mode_label(output_mode)} seek unsupported: {seek_reason}",
                status_code=409,
            )
    codec = PYNV_OUTPUT_CODEC
    client_host = request.client.host if request.client else ""
    container = _seek_container() if PASSTHROUGH_SEEK_VMP4 else (route_container or _seek_container())
    media_type = _seek_media_type(container)
    total = _seek_declared_total(path, info.duration, codec, client_host, container)
    annotate_request(request, total_estimated_size=total)
    byte_range = _parse_byte_range(range_header, total)
    if range_header and byte_range is None and not _seek_vmp4_enabled(container):
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
    _apply_seek_vmp4_headers(headers, path=path, duration=info.duration, total=total, container=container)
    if _seek_vmp4_enabled(container):
        log.info(
            "passthrough_seek[%d] VMP4 %s request: %s mode=%s range=%r total=%d",
            rid, _seek_vmp4_backend(), path.name, output_mode, range_header, total,
        )
        if _seek_vmp4_backend() == "slot_frames":
            # Frame-level backend blocks waiting for a GOP build; run off the
            # event loop so a waiting request cannot freeze other connections.
            return await run_in_threadpool(
                _seek_vmp4_frames_response,
                path=path,
                info=info,
                output_mode=output_mode,
                range_header=range_header,
                headers=headers,
                head_only=False,
                rid=rid,
            )
        if _seek_vmp4_backend() == "slot":
            # The slot backend may block up to PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT
            # waiting for a slot payload. Run it off the event loop so a waiting
            # request cannot freeze every other connection (StreamingResponse
            # chunks, other players, DLNA browse) while it sleeps.
            return await run_in_threadpool(
                _seek_vmp4_slot_response,
                path=path,
                info=info,
                output_mode=output_mode,
                total=total,
                range_header=range_header,
                headers=headers,
                head_only=False,
                rid=rid,
            )
        return _seek_vmp4_cache_response(
            path=path,
            info=info,
            output_mode=output_mode,
            total=total,
            range_header=range_header,
            headers=headers,
            head_only=False,
            rid=rid,
        )

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
    owner = (str(safe_resolve_path(path)), client_host, "seek")
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
    # The pseudo-VOD route historically bypassed the live route's mode gate.
    # Keep it consistent when RTX VSR is selected globally: reject unsupported
    # sources before advertising a stream that cannot be produced.
    vod_mode = _select_live_output_mode("")
    if vod_mode in _ENHANCED_MODE_LABELS:
        try:
            _, vod_meta, _ = await asyncio.to_thread(_probe_live_request_metadata, path)
        except Exception as exc:
            vod_meta = None
            vod_reason = f"metadata probe failed: {exc}"
        else:
            vod_reason = await _enhanced_mode_block_reason(path, vod_mode, vod_meta)
        if vod_reason:
            return Response(
                f"{_enhanced_mode_label(vod_mode)} pseudo-VOD unsupported: {vod_reason}",
                status_code=409,
            )
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
    vod_mode = _select_live_output_mode("")
    if vod_mode in _ENHANCED_MODE_LABELS:
        try:
            _, vod_meta, _ = await asyncio.to_thread(_probe_live_request_metadata, path)
        except Exception as exc:
            vod_meta = None
            vod_reason = f"metadata probe failed: {exc}"
        else:
            vod_reason = await _enhanced_mode_block_reason(path, vod_mode, vod_meta)
        if vod_reason:
            return Response(
                f"{_enhanced_mode_label(vod_mode)} pseudo-VOD unsupported: {vod_reason}",
                status_code=409,
            )
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
    owner = (str(safe_resolve_path(path)), request.client.host if request.client else "")
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
