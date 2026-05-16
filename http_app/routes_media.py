"""Media HTTP routes.

- /media/{name}: serve the source file with standard HTTP Range support.
- /passthrough/{name}: pseudo-VOD passthrough with byte/time seek mapping.
- /passthrough_live/{name}: MPEG-TS live passthrough for clients that dislike pseudo-VOD byte seeking.
"""
from __future__ import annotations

import asyncio
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Header, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from config import (
    HTTP_PORT,
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
    PASSTHROUGH_MAX_FPS,
    PASSTHROUGH_MAX_CONCURRENT,
    PASSTHROUGH_MKV_LIVE_POLICY,
    PASSTHROUGH_PAD_TO_LENGTH,
    PASSTHROUGH_SEND_MIN_BPS,
    PASSTHROUGH_SEND_PACING_MULTIPLIER,
    PASSTHROUGH_SEND_REALTIME_PACING,
    PASSTHROUGH_SEEK_MODE,
    DEBUG_LOGS,
    LIVE_REQUEST_HEADER_DUMP,
    MEDIA_LIBRARY,
    ROOT,
    USE_PYNV,
    VIDEO_EXTS,
)
from dlna.profiles import passthrough_dlna_pn, passthrough_frame_rate
from pipeline.ffmpeg_io import probe_cached
from pipeline.matting import get_matter
from pipeline.stream import PassthroughStream
from pipeline.pynv_stream import PYNV_BACKEND_LABEL, PYNV_OUTPUT_CODEC, PyNvPassthroughStream
from pipeline.thumbnail import get_thumb
from utils.bitrate_estimator import estimate_for_media, record_actual_bps
from utils.logger import get
from utils.mkv_cues import probe_mkv_cues
from utils.subtitles import find_external_subtitles, is_subtitle_path, subtitle_mime
from utils.video_metadata import probe_video_metadata, select_backend

log = get("media")
router = APIRouter()
DLNA_FLAGS_BASE = "01700000000000000000000000000000"
DLNA_FLAGS_TIME_SEEK = "41700000000000000000000000000000"
_request_ids = itertools.count(1)

# ---- Passthrough concurrency guard ----
# Keep passthrough concurrency low to avoid NVENC session exhaustion, blocked
# ffmpeg pipes, and concurrent access to the shared Matter/ONNX session.
_active_lock = asyncio.Lock()
_active_streams: dict[object, tuple[str, str]] = {}
_probe_cache_lock = asyncio.Lock()
_probe_cache: dict[str, bytes] = {}
_thumb_lock = asyncio.Lock()
_PROBE_CACHE_LIMIT = 16 * 1024 * 1024
_PROBE_CACHE_TOTAL_LIMIT = 64 * 1024 * 1024
_SMALL_PROBE_LIMIT = 64 * 1024
_PREFIX_CACHE_WAIT_SEC = 5.0
_PREFIX_CACHE_IDLE_SEC = 2.0
_TAIL_PROBE_RATIO = 0.95
_TAIL_PROBE_MAX_BYTES = 512 * 1024
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


_LIVE_END = object()


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

    async def subscribe(self, rid: int, *, primary: bool | None = None):
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
            "passthrough_live[%d] live cache subscribe: key=%s snapshot=%d primary=%s closed=%s subscribers=%d",
            rid,
            _live_session_log_key(self.key),
            len(snapshot),
            subscriber.primary,
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
            if not subscriber.primary:
                log.info(
                    "passthrough_live[%d] live cache duplicate snapshot complete: key=%s sent=%d",
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
    if callable(close):
        try:
            if isinstance(stream, LiveSession):
                await stream.close("preempted")
            else:
                await asyncio.to_thread(close)
        except Exception as e:
            log.warning("passthrough preempt close failed: %s", e)
    log.info("passthrough preempt close done: %s stream=%s", who, type(stream).__name__)


def _owner_base(owner: tuple) -> tuple:
    return owner[:2] if len(owner) >= 2 else owner


def _owner_kind(owner: tuple) -> str:
    return str(owner[2]) if len(owner) >= 3 else ""


def _can_preempt_owner(active_owner: tuple, new_owner: tuple) -> bool:
    new_base = _owner_base(new_owner)
    is_live_owner = len(new_base) > 0 and new_base[0] == "live"
    if active_owner == new_owner:
        kind = _owner_kind(new_owner)
        if is_live_owner:
            return kind in ("nplayer", "quest_dalvik")
        return kind in ("", "libmpv", "vlc", "nplayer")
    if _owner_base(active_owner) != new_base:
        return False
    active_kind = _owner_kind(active_owner)
    new_kind = _owner_kind(new_owner)
    if is_live_owner and new_kind in ("nplayer", "quest_dalvik"):
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


async def _take_active_slot(
    new_stream: object,
    who: str,
    owner: tuple,
    *,
    allow_same_owner_preempt: bool = True,
) -> object | None | bool:
    deadline = asyncio.get_running_loop().time() + PASSTHROUGH_BUSY_WAIT_SEC
    warned = False
    while True:
        async with _active_lock:
            if len(_active_streams) < PASSTHROUGH_MAX_CONCURRENT:
                _active_streams[new_stream] = owner
                return None
            if allow_same_owner_preempt:
                for active_stream, active_owner in list(_active_streams.items()):
                    if _can_preempt_owner(active_owner, owner):
                        del _active_streams[active_stream]
                        _active_streams[new_stream] = owner
                        log.info("passthrough preempt previous range: %s owner=%s", who, owner)
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
        removed = _active_streams.pop(stream, None)
        if removed is not None:
            log.info("passthrough active slot released: active=%d owner=%s", len(_active_streams), removed)


async def _replace_active_slot(old_stream: object, new_stream: object) -> bool:
    async with _active_lock:
        owner = _active_streams.pop(old_stream, None)
        if owner is None:
            return False
        _active_streams[new_stream] = owner
        return True


def _safe_video_path(name: str) -> Path:
    name = unquote(name)
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
async def subtitle_get(name: str, range: str | None = Header(default=None)):
    path = _safe_subtitle_path(name)
    headers = {
        "Content-Disposition": "inline",
        "Access-Control-Allow-Origin": "*",
    }
    return _file_range_response(path, subtitle_mime(path), range, headers)


@router.head("/subs/{name:path}")
async def subtitle_head(name: str, range: str | None = Header(default=None)):
    path = _safe_subtitle_path(name)
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
async def media_head(name: str, range: str | None = Header(default=None)):
    path = _safe_video_path(name)
    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
        **_subtitle_headers_for_video(path),
    }
    byte_range = _parse_byte_range(range, size)
    if byte_range is not None:
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{size}"
        headers["Content-Length"] = str(byte_range.length)
        return Response(status_code=206, headers=headers)
    headers["Content-Length"] = str(size)
    return Response(status_code=200, headers=headers)


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
    path = _safe_video_path(name)
    size = path.stat().st_size
    subtitle_headers = _subtitle_headers_for_video(path)
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
            "Content-Type": "video/mp4",
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
        return StreamingResponse(gen(), status_code=206, headers=headers, media_type="video/mp4")

    if DEBUG_LOGS:
        log.info("media[%d] response: status=200 path=%s size=%d full-file", rid, path.name, size)
    return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes", **subtitle_headers})


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
        op = "01"
        flags = "01700000000000000000000000000000"
    else:
        op = "10"
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
    ua = user_agent.lower()
    if "nplayer" in ua:
        return "nplayer"
    if "libmpv" in ua or "skybox" in ua:
        return "libmpv"
    if "dalvik/" in ua and "quest" in ua:
        return "quest_dalvik"
    if "vlc" in ua or "libvlc" in ua or "moonvr" in ua:
        return "vlc"
    if "lavf/" in ua:
        return "lavf"
    return PASSTHROUGH_LIVE_DEFAULT_PROFILE


def _is_nplayer_client(user_agent: str) -> bool:
    return "nplayer" in user_agent.lower()


def _is_lavf_client(user_agent: str) -> bool:
    return "lavf/" in user_agent.lower()


def _format_fps_header(fps: float | None) -> str | None:
    if fps is None or fps <= 0:
        return None
    return str(int(fps)) if float(fps).is_integer() else f"{fps:.3f}".rstrip("0").rstrip(".")


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
async def thumb_get(name: str, pt: int = Query(default=0)):
    path = _safe_video_path(name)
    async with _thumb_lock:
        out = await asyncio.to_thread(get_thumb, path, bool(pt))
    if out is None:
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
    path = _safe_video_path(name)
    _reject_unsafe_mkv_live_path(path)
    info = probe_cached(path)
    requested_t = t
    npt_t = _parse_npt_seconds(time_seek_range)
    if npt_t is not None:
        t = npt_t
    if info.duration > 0:
        t = min(t, max(0.0, info.duration - 0.01))
    requested_mode = (mode or "").lower()
    if PASSTHROUGH_OUTPUT_MODE == "all":
        live_output_mode = requested_mode if requested_mode in {"green", "alpha"} else "green"
    elif PASSTHROUGH_OUTPUT_MODE == "alpha":
        live_output_mode = "alpha"
    else:
        live_output_mode = "green"

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

    live_meta = None
    try:
        live_meta = probe_video_metadata(path)
    except Exception as e:
        log.warning("passthrough_live[%d] adaptive metadata probe failed: %s", rid, e)
    live_max_fps = _live_adaptive_max_fps(path, live_meta)
    live_profile = _live_response_profile(user_agent)
    is_nplayer = _is_nplayer_client(user_agent)
    use_managed_live_session = live_profile in {"libmpv", "quest_dalvik"} or is_nplayer
    live_total = _estimated_passthrough_size(path, max(0.0, info.duration - t), PYNV_OUTPUT_CODEC)
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
        live_profile not in {"libmpv", "quest_dalvik"}
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
    if is_nplayer:
        now = asyncio.get_running_loop().time()
        async with _live_session_lock:
            started_at = _live_starting.get(live_key)
        if started_at is not None and now - started_at < _LIVE_NPLAYER_START_DEBOUNCE_SEC:
            deadline = now + _LIVE_NPLAYER_START_DEBOUNCE_SEC
            while asyncio.get_running_loop().time() < deadline:
                cached_session = await _get_live_session(live_key)
                if cached_session is not None:
                    log.info(
                        "passthrough_live[%d] nPlayer duplicate startup joined cache: key=%s age=%.3fs",
                        rid,
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
                    "passthrough_live[%d] return 409 nPlayer duplicate startup still pending: key=%s age=%.3fs",
                    rid,
                    _live_session_log_key(live_key),
                    asyncio.get_running_loop().time() - started_at,
                )
                return Response("passthrough live duplicate startup", status_code=409, headers={"Retry-After": "1"})
        async with _live_session_lock:
            _live_starting[live_key] = now
        live_starting_at = now

    slot_token = object()
    # nPlayer does not reliably notify the server when the user leaves a live
    # item, so only nPlayer may replace an old live stream from the same device.
    # Other players keep their narrower duplicate-request/session behavior.
    owner = ("live", client_host, live_profile)
    preempted = await _take_active_slot(
        slot_token,
        who=f"live:{path.name}@{t:.2f}s",
        owner=owner,
        allow_same_owner_preempt=is_nplayer or live_profile == "quest_dalvik",
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

        def build_stream():
            matter = get_matter()
            return _select_passthrough_stream(
                path,
                t,
                matter,
                container="mpegts",
                max_fps=live_max_fps,
                audio_mode_override=live_audio_override,
                output_mode=live_output_mode,
                preflight=False,
            )

        stream, stream_backend, stream_verdict = await asyncio.to_thread(build_stream)
        if not await _replace_active_slot(slot_token, stream):
            stream.close()
            await _clear_live_starting(live_key, live_starting_at)
            log.info("passthrough_live[%d] return 409 preempted before stream", rid)
            return Response("passthrough live preempted", status_code=409)
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
                "DLNA.ORG_OP=01;"
                "DLNA.ORG_CI=1;"
                f"DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
            )
        else:
            headers["Accept-Ranges"] = "none"
            headers.pop("Content-Range", None)
            headers.pop("Content-Length", None)
            headers["contentFeatures.dlna.org"] = (
                "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
                "DLNA.ORG_OP=10;"
                "DLNA.ORG_CI=1;"
                f"DLNA.ORG_FLAGS={DLNA_FLAGS_TIME_SEEK}"
            )
    else:
        headers["contentFeatures.dlna.org"] = (
            "DLNA.ORG_PN=HEVC_TS_NA_ISO;"
            "DLNA.ORG_OP=10;"
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
        await asyncio.to_thread(stream.close)
        await _clear_live_starting(live_key, live_starting_at)
        log.info("passthrough_live[%d] return 409 preempted before live session", rid)
        return Response("passthrough live preempted", status_code=409)
    await _put_live_session(live_key, session)
    await _clear_live_starting(live_key, live_starting_at)
    session.start(stream_iter)

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
                    PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC > 0
                    and sent > 0
                    and asyncio.get_running_loop().time() - last_send_wall > PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC
                ):
                    log.info(
                        "passthrough_live[%d] send stall watchdog closing stream: sent=%d stream_bytes=%d frames=%d idle=%.1fs",
                        rid,
                        sent,
                        getattr(stream, "bytes_emitted", -1),
                        getattr(stream, "frames_produced", -1),
                        asyncio.get_running_loop().time() - last_send_wall,
                    )
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
        log.info(
            "passthrough preempt deferred close: %s stream=%s",
            f"{path.name}@{t:.2f}s",
            type(preempted).__name__,
        )

    try:
        matter = get_matter()
        stream, stream_backend, stream_verdict = _select_passthrough_stream(path, t, matter)
        if not await _replace_active_slot(slot_token, stream):
            stream.close()
            log.info("passthrough[%d] return 409 preempted before stream", rid)
            return Response("passthrough preempted", status_code=409)
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
                    PASSTHROUGH_SEEK_MODE == "bytes"
                    and (byte_range is None or byte_range.start == 0)
                    and len(probe_prefix) < _PROBE_CACHE_LIMIT
                ):
                    need = _PROBE_CACHE_LIMIT - len(probe_prefix)
                    probe_prefix.extend(chunk[:need])
                    async with _probe_cache_lock:
                        _set_probe_cache_locked(probe_key, bytes(probe_prefix))
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
            if cache_probe_prefix and probe_prefix:
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
            await asyncio.to_thread(stream.close)
            await _release_active_slot(stream)
            log.info("passthrough[%d] finally done: sent=%d", rid, sent)

    return StreamingResponse(gen(), status_code=status_code, headers=headers, media_type=media_type)
