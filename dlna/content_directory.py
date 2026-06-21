"""UPnP ContentDirectory implementation for the local media library.

The root ObjectID maps to VIDEO_DIR. Physical subdirectories are exposed as
DIDL containers with ids of the form ``d_<relative/path>``. Each normal video
file is exposed as the raw media item plus a passthrough-live item. When enabled
by config, still images are also exposed as photo items using the same /media
route. The older pseudo-VOD passthrough endpoint still exists in HTTP code but
is hidden from DLNA while client seek behavior is being evaluated. The
passthrough-live item is a chapter container, allowing clients to choose a start
time without relying on HTTP Range seeking.
"""
from __future__ import annotations

import html
import math
import re
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

from config import (
    HTTP_PORT,
    DLNA_IMAGE_ENABLED,
    IMAGE_DLNA_PN_BY_EXT,
    IMAGE_EXTS,
    IMAGE_MIME_BY_EXT,
    LAN_IP,
    PASSTHROUGH_BITRATE,
    PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS,
    PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC,
    PASSTHROUGH_MKV_LIVE_POLICY,
    PASSTHROUGH_OUTPUT_MODE,
    PASSTHROUGH_SEEK_DLNA,
    PASSTHROUGH_SEEK_ENABLED,
    PASSTHROUGH_SEEK_CONTAINER,
    PASSTHROUGH_SEEK_HEADER_BYTES,
    PASSTHROUGH_SEEK_MODE,
    SI_PROGRESSIVE_DLNA,
    SI_PROGRESSIVE_ENABLED,
    MEDIA_LIBRARY,
    VIDEO_DIR,
    VIDEO_EXTS,
)
from dlna.profiles import passthrough_frame_rate
from pipeline.alpha_packer import alpha_output_size
from pipeline.ffmpeg_io import probe_cached
from utils.bitrate_estimator import estimate_for_media
from utils.logger import get
from utils.media_index import IndexedChild, get_media_index
from utils.offline_outputs import (
    has_offline_passthrough_output,
    has_offline_two_dvr_output,
    is_offline_passthrough_output_name,
)
from utils.runtime_settings import get_si_mix
from utils.subtitles import SubtitleTrack, find_external_subtitles, subtitle_output_enabled
from utils.video_metadata import probe_video_metadata, select_backend
from utils.vr_naming import (
    has_vr_filename_marker,
    is_half_equirectangular_source,
    live_passthrough_title,
    source_display_stem,
)

log = get("cds")

ROOT_ID = "0"
FOLDER_PREFIX = "d_"
LEGACY_FOLDER_PREFIX = "d:"
LIVE_PREFIX = "pl_"
LEGACY_LIVE_PREFIX = "pl:"
ALPHA_LIVE_PREFIX = "pla_"
LEGACY_ALPHA_LIVE_PREFIX = "pla:"
TWO_DVR_LIVE_PREFIX = "pl3_"
LIVE_ITEM_PREFIX = "lg_"
ALPHA_LIVE_ITEM_PREFIX = "la_"
TWO_DVR_LIVE_ITEM_PREFIX = "l3_"
LIVE_TIME_INDEX_PREFIX = "lix_"
LIVE_TIME_GROUP_PREFIX = "lig_"
LIVE_TIME_MINUTE_PREFIX = "lim_"
LIVE_TIME_POINT_PREFIX = "li5_"
SEEK_ITEM_PREFIX = "sg_"
ALPHA_SEEK_ITEM_PREFIX = "sa_"
IMAGE_ITEM_PREFIX = "img_"
SI_ITEM_PREFIX = "si_"
# Realtime SI ([SI]) directory + time-index object ids. The [SI] entry is a
# container whose children are a time-index tree (group -> minute -> 5s point);
# each point leaf plays the realtime MPEG-TS `/si_live` stream at that offset.
# None of these prefixes start with "si_"/"sg_"/"sa_", so they never collide with
# the legacy SI item, seek, or alpha-seek dispatch.
SI_DIR_PREFIX = "six_"
SI_CHAPTER_ITEM_PREFIX = "sic_"
SI_TIME_INDEX_PREFIX = "sxi_"
SI_TIME_GROUP_PREFIX = "sig_"
SI_TIME_MINUTE_PREFIX = "sin_"
SI_TIME_POINT_PREFIX = "sit_"
PYNV_OUTPUT_CODEC = "hevc"
DLNA_FLAGS_BASE = "01700000000000000000000000000000"
DLNA_FLAGS_TIME_SEEK = "41700000000000000000000000000000"
DLNA_FLAGS_BYTE_AND_TIME_SEEK = "01F00000000000000000000000000000"
# Keep these legacy OP names aligned with http_app.routes_media. The old
# passthrough/live compatibility branches intentionally keep their historic
# OP=01/10 behavior. New seek items advertise OP=11 with file-like transfer
# flags only; lop-npt/lop-bytes flags conflict with full random access and can
# make clients present the item as live/limited.
DLNA_OP_BYTE_SEEK = "01"
DLNA_OP_TIME_SEEK = "10"
DLNA_OP_BYTE_AND_TIME_SEEK = "11"
DIDL_NS = "urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/"

_DIR_ITEMS_CACHE_MAX = 256
_DIDL_SCHEMA_VERSION = 10
_SYSTEM_UPDATE_ID = _DIDL_SCHEMA_VERSION
_OBJECT_ID_VERSION_PREFIX = f"ptv{_DIDL_SCHEMA_VERSION}_"
_dir_items_cache: dict[tuple, list[dict]] = {}
_LIVE_MAX_SIDE = 8192
_NO_LIVE_PREFIX = "[NoLive] "
_OFFLINE_PREFIX = "[Offline] "
_CDS_CLIENT_DEOVR = "deovr"
_TIME_INDEX_GROUP_SEC = 10 * 60
_TIME_INDEX_MINUTE_SEC = 60
_TIME_INDEX_POINT_SEC = 5
_LIVE_TIME_MODE_TOKEN_BY_MODE = {"green": "g", "alpha": "a", "two_dvr": "3"}
_LIVE_TIME_MODE_BY_TOKEN = {token: mode for mode, token in _LIVE_TIME_MODE_TOKEN_BY_MODE.items()}
_SELECT_TIME_INDEX_LABELS = {
    "en_US": "Select Time Index",
    "zh_CN": "选择时间索引",
    "ja_JP": "時間インデックス選択",
}


def clear_dir_items_cache() -> None:
    _dir_items_cache.clear()


def _system_update_id() -> int:
    return _SYSTEM_UPDATE_ID + int(get_si_mix().version)


def _parse_bitrate(s: str) -> int:
    s = s.strip().upper()
    try:
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith("K"):
            return int(float(s[:-1]) * 1_000)
        return int(s)
    except ValueError:
        return 20_000_000


def _fmt_duration(sec: float) -> str:
    if sec <= 0:
        return "0:00:00.000"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:06.3f}"


def _fmt_title_time(sec: int) -> str:
    if sec <= 0:
        return "00:00"
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _fmt_index_time(sec: int, force_hours: bool = False) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if force_hours or h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _normalise_ui_language(language: str | None) -> str:
    value = str(language or "").strip().lower().replace("-", "_")
    if value.startswith("zh"):
        return "zh_CN"
    if value.startswith("ja"):
        return "ja_JP"
    return "en_US"


def _select_time_index_label(language: str | None = None) -> str:
    return _SELECT_TIME_INDEX_LABELS[_normalise_ui_language(language)]


def _root() -> Path:
    return VIDEO_DIR.resolve()


def _rel_key(path: Path) -> str:
    return MEDIA_LIBRARY.path_to_key(path)


def _versioned_rel(rel: str) -> str:
    rel = str(rel or "")
    return f"{_OBJECT_ID_VERSION_PREFIX}{rel}" if rel else rel


def _strip_object_id_version(rel: str) -> str:
    rel = str(rel or "")
    if rel.startswith(_OBJECT_ID_VERSION_PREFIX):
        return rel[len(_OBJECT_ID_VERSION_PREFIX):]
    match = re.match(r"^ptv\d+_(.+)$", rel)
    return match.group(1) if match else rel


def _folder_id(path: Path) -> str:
    rel = _rel_key(path)
    return ROOT_ID if not rel or rel == "." else f"{FOLDER_PREFIX}{_versioned_rel(rel)}"


def _id_to_dir(object_id: str) -> Path | None:
    object_id = object_id or ROOT_ID
    if object_id == ROOT_ID:
        return _root() if not MEDIA_LIBRARY.multi_root else None
    if object_id.startswith(FOLDER_PREFIX):
        rel = object_id[len(FOLDER_PREFIX):].replace("\\", "/").strip("/")
    elif object_id.startswith(LEGACY_FOLDER_PREFIX):
        rel = object_id[len(LEGACY_FOLDER_PREFIX):].replace("\\", "/").strip("/")
    else:
        return None
    rel = _strip_object_id_version(rel)
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path):
        return path
    return None


def _id_to_live(object_id: str) -> tuple[Path, str] | None:
    mode = "green"
    prefix = LIVE_PREFIX
    if object_id.startswith(TWO_DVR_LIVE_ITEM_PREFIX):
        mode = "two_dvr"
        prefix = TWO_DVR_LIVE_ITEM_PREFIX
    elif object_id.startswith(ALPHA_LIVE_ITEM_PREFIX):
        mode = "alpha"
        prefix = ALPHA_LIVE_ITEM_PREFIX
    elif object_id.startswith(LIVE_ITEM_PREFIX):
        prefix = LIVE_ITEM_PREFIX
    elif object_id.startswith(TWO_DVR_LIVE_PREFIX):
        mode = "two_dvr"
        prefix = TWO_DVR_LIVE_PREFIX
    elif object_id.startswith(ALPHA_LIVE_PREFIX):
        mode = "alpha"
        prefix = ALPHA_LIVE_PREFIX
    elif object_id.startswith(LEGACY_ALPHA_LIVE_PREFIX):
        mode = "alpha"
        prefix = LEGACY_ALPHA_LIVE_PREFIX
    elif not object_id.startswith(LIVE_PREFIX):
        if object_id.startswith(LEGACY_LIVE_PREFIX):
            prefix = LEGACY_LIVE_PREFIX
        else:
            return None
    rel = object_id[len(prefix):].replace("\\", "/").strip("/")
    rel = _strip_object_id_version(rel)
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return path, mode
    return None


def _live_time_index_id(prefix: str, rel: str, mode: str, payload: str | None = None) -> str:
    token = _LIVE_TIME_MODE_TOKEN_BY_MODE.get(mode, _LIVE_TIME_MODE_TOKEN_BY_MODE["green"])
    object_id = f"{prefix}{token}_{_versioned_rel(rel)}"
    return f"{object_id}@{payload}" if payload else object_id


def _id_to_live_time_index(object_id: str) -> tuple[Path, str, str, int, int] | None:
    prefixes = (
        (LIVE_TIME_INDEX_PREFIX, "index"),
        (LIVE_TIME_GROUP_PREFIX, "group"),
        (LIVE_TIME_MINUTE_PREFIX, "minute"),
    )
    prefix = ""
    level = ""
    for candidate, candidate_level in prefixes:
        if object_id.startswith(candidate):
            prefix = candidate
            level = candidate_level
            break
    if not prefix:
        return None

    rest = object_id[len(prefix):]
    payload = ""
    if level != "index":
        if "@" not in rest:
            return None
        rest, payload = rest.rsplit("@", 1)
    if "_" not in rest:
        return None
    token, rel = rest.split("_", 1)
    mode = _LIVE_TIME_MODE_BY_TOKEN.get(token)
    if mode is None:
        return None

    start = 0
    end = 0
    try:
        if level == "group":
            start_text, end_text = payload.split("-", 1)
            start = max(0, int(start_text))
            end = max(start, int(end_text))
        elif level == "minute":
            start = max(0, int(payload))
    except ValueError:
        return None

    rel = _strip_object_id_version(rel).replace("\\", "/").strip("/")
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return path, mode, level, start, end
    return None


def _id_to_seek(object_id: str) -> tuple[Path, str] | None:
    mode = "green"
    if object_id.startswith(ALPHA_SEEK_ITEM_PREFIX):
        mode = "alpha"
        prefix = ALPHA_SEEK_ITEM_PREFIX
    elif object_id.startswith(SEEK_ITEM_PREFIX):
        prefix = SEEK_ITEM_PREFIX
    else:
        return None
    rel = object_id[len(prefix):].replace("\\", "/").strip("/")
    rel = _strip_object_id_version(rel)
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return path, mode
    return None


def _id_to_image(object_id: str) -> Path | None:
    if not DLNA_IMAGE_ENABLED or not object_id.startswith(IMAGE_ITEM_PREFIX):
        return None
    rel = object_id[len(IMAGE_ITEM_PREFIX):].replace("\\", "/").strip("/")
    rel = _strip_object_id_version(rel)
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return path
    return None


def _id_to_si(object_id: str) -> Path | None:
    if not object_id.startswith(SI_ITEM_PREFIX):
        return None
    rel = object_id[len(SI_ITEM_PREFIX):].replace("\\", "/").strip("/")
    rel = _strip_object_id_version(rel)
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and _has_si_sidecar(path):
        return path
    return None


def _si_time_index_id(prefix: str, rel: str, payload: str | None = None) -> str:
    object_id = f"{prefix}{_versioned_rel(rel)}"
    return f"{object_id}@{payload}" if payload else object_id


def _si_path_from_rel(rel: str) -> Path | None:
    rel = _strip_object_id_version(rel.replace("\\", "/").strip("/"))
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and _has_si_sidecar(path):
        return path
    return None


def _id_to_si_dir(object_id: str) -> Path | None:
    if not object_id.startswith(SI_DIR_PREFIX):
        return None
    return _si_path_from_rel(object_id[len(SI_DIR_PREFIX):])


def _id_to_si_time_index(object_id: str) -> tuple[Path, str, int, int] | None:
    if object_id.startswith(SI_TIME_INDEX_PREFIX):
        path = _si_path_from_rel(object_id[len(SI_TIME_INDEX_PREFIX):])
        return (path, "index", 0, 0) if path is not None else None
    for prefix, level in ((SI_TIME_GROUP_PREFIX, "group"), (SI_TIME_MINUTE_PREFIX, "minute")):
        if not object_id.startswith(prefix):
            continue
        rest = object_id[len(prefix):]
        if "@" not in rest:
            return None
        rel, payload = rest.rsplit("@", 1)
        path = _si_path_from_rel(rel)
        if path is None:
            return None
        start = 0
        end = 0
        try:
            if level == "group":
                start_text, end_text = payload.split("-", 1)
                start = max(0, int(start_text))
                end = max(start, int(end_text))
            else:
                start = max(0, int(payload))
        except ValueError:
            return None
        return path, level, start, end
    return None


def _id_to_si_point(object_id: str) -> tuple[Path, int, str] | None:
    """Resolve a playable SI leaf id (quick chapter ``sic_`` or 5s point ``sit_``)."""
    for prefix in (SI_CHAPTER_ITEM_PREFIX, SI_TIME_POINT_PREFIX):
        if not object_id.startswith(prefix):
            continue
        rest = object_id[len(prefix):]
        if "@" not in rest:
            return None
        rel, payload = rest.rsplit("@", 1)
        path = _si_path_from_rel(rel)
        if path is None:
            return None
        try:
            offset = max(0, int(payload))
        except ValueError:
            return None
        return path, offset, prefix
    return None


def _parent_id_for_dir(path: Path) -> str:
    path = path.resolve()
    if MEDIA_LIBRARY.multi_root:
        for root in MEDIA_LIBRARY.roots:
            if path == root.path:
                return ROOT_ID
    elif path == _root():
        return "-1"
    parent = path.parent
    return _folder_id(parent)


def _video_item_count(path: Path, child: IndexedChild | None = None) -> int:
    si_items = 1 if _si_dlna_enabled() and get_si_mix().enabled and _has_si_sidecar(path) else 0
    if (
        is_offline_passthrough_output_name(path.name)
        or PASSTHROUGH_OUTPUT_MODE == "none"
        or has_offline_passthrough_output(path)
        or _hide_passthrough_for_path(path, child)
    ):
        return 1 + si_items
    passthrough_modes = len(_passthrough_modes())
    seek_items = passthrough_modes if _seek_passthrough_dlna_enabled() else 0
    return 1 + si_items + passthrough_modes + seek_items


def _marked_original_title(path: Path, child: IndexedChild | None = None) -> str:
    width, height = _indexed_video_dimensions(child)
    title = source_display_stem(path.stem, width, height)
    if is_offline_passthrough_output_name(path.name) and not title.startswith(_OFFLINE_PREFIX.strip()):
        title = f"{_OFFLINE_PREFIX}{title}"
    if _hide_passthrough_for_path(path, child) and not title.startswith(_NO_LIVE_PREFIX.strip()):
        return f"{_NO_LIVE_PREFIX}{title}"
    return title


def _indexed_video_dimensions(child: IndexedChild | None) -> tuple[int, int]:
    video = child.video if child is not None else None
    if video is None:
        return 0, 0
    width = int(getattr(video, "width", 0) or 0)
    height = int(getattr(video, "height", 0) or 0)
    if width <= 0 or height <= 0:
        width, height = _parse_resolution(getattr(video, "resolution", ""))
    return width, height


def _live_passthrough_block_reason(path: Path, child: IndexedChild | None) -> str:
    video = child.video if child is not None else None
    if path.suffix.lower() == ".mkv" and PASSTHROUGH_MKV_LIVE_POLICY == "block":
        return "mkv_disabled"
    if path.suffix.lower() == ".mkv" and PASSTHROUGH_MKV_LIVE_POLICY == "head_cues":
        if video is None or video.mkv_needs_fix:
            return "mkv_needs_remux"
    if video is not None and video.mkv_needs_fix:
        return "mkv_needs_remux"
    if video is not None and getattr(video, "probe_error", ""):
        return "probe_error"
    width, height = _indexed_video_dimensions(child)
    if video is not None and (width <= 0 or height <= 0):
        return "missing_dimensions"
    if width > _LIVE_MAX_SIDE or height > _LIVE_MAX_SIDE:
        return "resolution_too_large"
    verdict = str(getattr(video, "backend_verdict", "") if video is not None else "")
    if verdict and verdict != "pynv_hevc":
        return verdict
    return ""


def _hide_passthrough_for_path(path: Path, child: IndexedChild | None) -> bool:
    return bool(_live_passthrough_block_reason(path, child))


def _passthrough_modes() -> tuple[str, ...]:
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


def _subtitle_item(track: SubtitleTrack) -> dict:
    rel = _rel_key(track.path)
    return {
        "url": f"http://{LAN_IP}:{HTTP_PORT}/subs/{quote(rel)}",
        "lang": track.lang,
        "type": track.kind,
        "mime": track.mime,
    }


def _image_mime(path: Path) -> str:
    return IMAGE_MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")


def _image_protocol_info(path: Path) -> str:
    mime = _image_mime(path)
    dlna_pn = IMAGE_DLNA_PN_BY_EXT.get(path.suffix.lower())
    if not dlna_pn:
        return f"http-get:*:{mime}:*"
    return (
        f"http-get:*:{mime}:DLNA.ORG_PN={dlna_pn};"
        "DLNA.ORG_OP=00;"
        f"DLNA.ORG_CI=0;DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}"
    )


def _has_si_sidecar(path: Path) -> bool:
    return path.suffix.lower() == ".mp4" and path.with_suffix(".si.wav").is_file()


def _si_dlna_enabled() -> bool:
    return bool(SI_PROGRESSIVE_ENABLED and SI_PROGRESSIVE_DLNA)


def _image_resolution(path: Path) -> str:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return _resolution_str(int(width), int(height))
    except Exception:
        return ""


def _image_item_from_index(path: Path, parent_id: str, child: IndexedChild | None = None) -> dict:
    base = f"http://{LAN_IP}:{HTTP_PORT}"
    rel = _rel_key(path)
    quoted = quote(rel)
    url = f"{base}/media/{quoted}"
    size = child.size if child is not None else path.stat().st_size
    return {
        "id": f"{IMAGE_ITEM_PREFIX}{_versioned_rel(rel)}",
        "parent_id": parent_id,
        "title": path.stem,
        "url": url,
        "thumb": url,
        "thumb_profile": IMAGE_DLNA_PN_BY_EXT.get(path.suffix.lower(), ""),
        "size": size,
        "resolution": _image_resolution(path),
        "mime": _image_mime(path),
        "protocol_info": _image_protocol_info(path),
        "upnp_class": "object.item.imageItem.photo",
    }


def _si_directory_title(path: Path, width: int = 0, height: int = 0) -> str:
    return f"[SI]{source_display_stem(path.stem, width, height)}"


def _si_time_index_title(path: Path, width: int, height: int, language: str | None = None) -> str:
    return f"[{_select_time_index_label(language)}]_{_si_directory_title(path, width, height)}"


def _si_directory_child_count(duration: float) -> int:
    # Mirror the Live container: N quick-play chapter leaves + 1 "Select Time
    # Index" subdirectory.
    return len(_live_chapter_offsets(duration)) + 1


def _si_container_item(path: Path, parent_id: str, duration: float, width: int, height: int) -> dict:
    """The `[SI]` directory shown beside the source video.

    Like the Live container, it lists up to N quick-play chapter leaves plus a
    "Select Time Index" subdirectory. Every leaf plays the realtime `/si_live`
    MPEG-TS stream from its offset; no sidecar cache is built.
    """
    rel = _rel_key(path)
    return {
        "container": True,
        "id": _si_time_index_id(SI_DIR_PREFIX, rel),
        "parent_id": parent_id,
        "title": _si_directory_title(path, width, height),
        "child_count": _si_directory_child_count(duration),
    }


def _si_time_index_root_item(
    path: Path,
    parent_id: str,
    duration: float,
    width: int,
    height: int,
    language: str | None = None,
) -> dict:
    rel = _rel_key(path)
    return {
        "container": True,
        "id": _si_time_index_id(SI_TIME_INDEX_PREFIX, rel),
        "parent_id": parent_id,
        "title": _si_time_index_title(path, width, height, language),
        "child_count": _live_time_index_child_count(duration, "index"),
    }


def _si_play_leaf(
    path: Path,
    item_id: str,
    parent_id: str,
    title: str,
    offset: int,
    duration: float,
    width: int,
    height: int,
    source_fps: float,
    pt_bps_est: int,
    client_profile: str | None,
) -> dict:
    base = f"http://{LAN_IP}:{HTTP_PORT}"
    quoted = quote(_rel_key(path))
    live_route_suffix = _live_route_hint_suffix(client_profile)
    omit_filelike_attrs = not _is_deovr_cds_client(client_profile)
    remain = max(0.0, duration - float(offset)) if duration > 0 else 0.0
    return {
        "id": item_id,
        "parent_id": parent_id,
        "title": title,
        "url": f"{base}/si_live/{quoted}{live_route_suffix}?t={offset}&ptv={_DIDL_SCHEMA_VERSION}",
        "thumb": f"{base}/thumb/{quoted}",
        "size": 0,
        "duration": remain,
        "resolution": _resolution_str(width, height),
        "bitrate": pt_bps_est,
        "mime": "video/MP2T",
        "dlna_pn": "HEVC_TS_NA_ISO",
        "frame_rate": passthrough_frame_rate(source_fps),
        "passthrough": True,
        "passthrough_mode": "si_mix",
        "protocol_info": _live_passthrough_protocol_info(client_profile),
        "omit_duration": omit_filelike_attrs,
        "omit_bitrate": omit_filelike_attrs,
    }


def _si_chapter_items(
    path: Path,
    client_profile: str | None = None,
    language: str | None = None,
) -> list[dict]:
    """Children of the `[SI]` container: a Select-Time-Index entry + quick chapters."""
    rel = _rel_key(path)
    duration, width, height, source_fps = _probe_live_directory_context(path, rel)
    parent_id = _si_time_index_id(SI_DIR_PREFIX, rel)
    virtual_title = source_display_stem(path.stem, width, height)
    if duration > 0:
        _, pt_bps_est, _ = estimate_for_media(path, duration, PYNV_OUTPUT_CODEC)
    else:
        pt_bps_est = _parse_bitrate(PASSTHROUGH_BITRATE)
    items: list[dict] = [_si_time_index_root_item(path, parent_id, duration, width, height, language)]
    for offset in _live_chapter_offsets(duration):
        items.append(
            _si_play_leaf(
                path,
                _si_time_index_id(SI_CHAPTER_ITEM_PREFIX, rel, str(offset)),
                parent_id,
                f"{_fmt_title_time(offset)}_[SI]{virtual_title}",
                offset,
                duration,
                width,
                height,
                source_fps,
                pt_bps_est,
                client_profile,
            )
        )
    return items


def _si_time_minute_items(
    path: Path,
    parent_id: str,
    start: int,
    end: int,
    duration: float,
    virtual_title: str,
) -> list[dict]:
    rel = _rel_key(path)
    force_hours = _live_time_force_hours(duration)
    items: list[dict] = []
    for minute in _live_time_minute_offsets(start, end):
        items.append(
            {
                "container": True,
                "id": _si_time_index_id(SI_TIME_MINUTE_PREFIX, rel, str(minute)),
                "parent_id": parent_id,
                "title": f"{_fmt_index_time(minute, force_hours)}_[SI]{virtual_title}",
                "child_count": _live_time_index_child_count(duration, "minute", minute),
            }
        )
    return items


def _si_time_index_items(
    path: Path,
    level: str,
    start: int = 0,
    end: int = 0,
    client_profile: str | None = None,
    language: str | None = None,
) -> list[dict]:
    rel = _rel_key(path)
    duration, width, height, source_fps = _probe_live_directory_context(path, rel)
    virtual_title = source_display_stem(path.stem, width, height)
    force_hours = _live_time_force_hours(duration)
    if level == "index":
        parent_id = _si_time_index_id(SI_TIME_INDEX_PREFIX, rel)
        groups = _live_time_group_ranges(duration)
        if len(groups) == 1:
            group_start, group_end = groups[0]
            return _si_time_minute_items(path, parent_id, group_start, group_end, duration, virtual_title)
        return [
            {
                "container": True,
                "id": _si_time_index_id(SI_TIME_GROUP_PREFIX, rel, f"{group_start}-{group_end}"),
                "parent_id": parent_id,
                "title": (
                    f"{_fmt_index_time(group_start, force_hours)}-{_fmt_index_time(group_end, force_hours)}"
                    f"_[SI]{virtual_title}"
                ),
                "child_count": _live_time_index_child_count(duration, "group", group_start, group_end),
            }
            for group_start, group_end in groups
        ]
    if level == "group":
        parent_id = _si_time_index_id(SI_TIME_GROUP_PREFIX, rel, f"{start}-{end}")
        return _si_time_minute_items(path, parent_id, start, end, duration, virtual_title)
    if level != "minute":
        return []

    if duration > 0:
        _, pt_bps_est, _ = estimate_for_media(path, duration, PYNV_OUTPUT_CODEC)
    else:
        pt_bps_est = _parse_bitrate(PASSTHROUGH_BITRATE)
    parent_id = _si_time_index_id(SI_TIME_MINUTE_PREFIX, rel, str(start))
    items: list[dict] = []
    for offset in _live_time_point_offsets(start, duration):
        items.append(
            _si_play_leaf(
                path,
                _si_time_index_id(SI_TIME_POINT_PREFIX, rel, str(offset)),
                parent_id,
                f"{_fmt_index_time(offset, force_hours)}_[SI]{virtual_title}",
                offset,
                duration,
                width,
                height,
                source_fps,
                pt_bps_est,
                client_profile,
            )
        )
    return items


def _si_time_index_metadata_item(
    path: Path,
    level: str,
    start: int = 0,
    end: int = 0,
    language: str | None = None,
) -> dict | None:
    rel = _rel_key(path)
    duration, width, height, _source_fps = _probe_live_directory_context(path, rel)
    virtual_title = source_display_stem(path.stem, width, height)
    force_hours = _live_time_force_hours(duration)
    if level == "index":
        return _si_time_index_root_item(
            path, _si_time_index_id(SI_DIR_PREFIX, rel), duration, width, height, language
        )
    if level == "group":
        return {
            "container": True,
            "id": _si_time_index_id(SI_TIME_GROUP_PREFIX, rel, f"{start}-{end}"),
            "parent_id": _si_time_index_id(SI_TIME_INDEX_PREFIX, rel),
            "title": (
                f"{_fmt_index_time(start, force_hours)}-{_fmt_index_time(end, force_hours)}_[SI]{virtual_title}"
            ),
            "child_count": _live_time_index_child_count(duration, "group", start, end),
        }
    if level == "minute":
        groups = _live_time_group_ranges(duration)
        parent_id = _si_time_index_id(SI_TIME_INDEX_PREFIX, rel)
        if len(groups) > 1:
            group_start, group_end = _live_time_parent_group_range(start, duration)
            parent_id = _si_time_index_id(SI_TIME_GROUP_PREFIX, rel, f"{group_start}-{group_end}")
        return {
            "container": True,
            "id": _si_time_index_id(SI_TIME_MINUTE_PREFIX, rel, str(start)),
            "parent_id": parent_id,
            "title": f"{_fmt_index_time(start, force_hours)}_[SI]{virtual_title}",
            "child_count": _live_time_index_child_count(duration, "minute", start),
        }
    return None


def _si_point_metadata_item(path: Path, offset: int, prefix: str) -> dict | None:
    suffix = f"@{int(offset)}"
    if prefix == SI_CHAPTER_ITEM_PREFIX:
        candidates = _si_chapter_items(path)
    else:
        minute_start = (max(0, int(offset)) // _TIME_INDEX_MINUTE_SEC) * _TIME_INDEX_MINUTE_SEC
        candidates = _si_time_index_items(path, "minute", start=minute_start)
    for item in candidates:
        item_id = str(item.get("id", ""))
        if item_id.startswith(prefix) and item_id.endswith(suffix):
            return item
    return None


def _passthrough_virtual_title(path: Path, mode: str, width: int = 0, height: int = 0) -> str:
    return live_passthrough_title(path.stem, mode, width, height)


def _passthrough_seek_title(path: Path, mode: str, width: int = 0, height: int = 0) -> str:
    title = _passthrough_virtual_title(path, mode, width, height)
    if title.endswith("_live"):
        return f"{title[:-5]}_seek"
    return f"{title}_seek"


def _passthrough_live_prefix(mode: str) -> str:
    if mode == "two_dvr":
        return TWO_DVR_LIVE_PREFIX
    return ALPHA_LIVE_PREFIX if mode == "alpha" else LIVE_PREFIX


def _passthrough_live_item_prefix(mode: str) -> str:
    if mode == "two_dvr":
        return TWO_DVR_LIVE_ITEM_PREFIX
    return ALPHA_LIVE_ITEM_PREFIX if mode == "alpha" else LIVE_ITEM_PREFIX


def _passthrough_seek_item_prefix(mode: str) -> str:
    return ALPHA_SEEK_ITEM_PREFIX if mode == "alpha" else SEEK_ITEM_PREFIX


def _passthrough_live_query(mode: str) -> str:
    version = f"ptv={_DIDL_SCHEMA_VERSION}"
    if mode in {"green", "alpha", "two_dvr"}:
        return f"mode={mode}&{version}"
    return version


def _passthrough_seek_query(mode: str) -> str:
    return _passthrough_live_query(mode)


def _is_deovr_cds_client(client_profile: str | None) -> bool:
    return str(client_profile or "").strip().lower() == _CDS_CLIENT_DEOVR


def _live_route_hint_suffix(client_profile: str | None = None) -> str:
    return "" if _is_deovr_cds_client(client_profile) else ".ts"


def _live_passthrough_protocol_info(client_profile: str | None = None) -> str:
    if _is_deovr_cds_client(client_profile):
        return (
            "http-get:*:video/MP2T:DLNA.ORG_PN=HEVC_TS_NA_ISO;"
            f"DLNA.ORG_OP={DLNA_OP_TIME_SEEK};"
            f"DLNA.ORG_CI=1;DLNA.ORG_FLAGS={DLNA_FLAGS_TIME_SEEK}"
        )
    return "http-get:*:video/MP2T:DLNA.ORG_PN=HEVC_TS_NA_ISO;DLNA.ORG_OP=00;DLNA.ORG_CI=1"


def _seek_passthrough_protocol_info() -> str:
    container = _seek_passthrough_container()
    return (
        f"http-get:*:{_seek_passthrough_mime(container)}:DLNA.ORG_PN={_seek_passthrough_dlna_pn(container)};"
        f"DLNA.ORG_OP={DLNA_OP_BYTE_AND_TIME_SEEK};"
        f"DLNA.ORG_CI=0;DLNA.ORG_FLAGS={DLNA_FLAGS_BYTE_AND_TIME_SEEK}"
    )


def _seek_passthrough_dlna_enabled() -> bool:
    return bool(PASSTHROUGH_SEEK_ENABLED and PASSTHROUGH_SEEK_DLNA)


def _seek_passthrough_container() -> str:
    return PASSTHROUGH_SEEK_CONTAINER if PASSTHROUGH_SEEK_CONTAINER in {"mpegts", "mp4"} else "mpegts"


def _seek_passthrough_mime(container: str | None = None) -> str:
    return "video/mp4" if (container or _seek_passthrough_container()) == "mp4" else "video/MP2T"


def _seek_passthrough_dlna_pn(container: str | None = None) -> str:
    return "HEVC_MP4_MAIN" if (container or _seek_passthrough_container()) == "mp4" else "HEVC_TS_NA_ISO"


def _seek_passthrough_route_suffix(container: str | None = None) -> str:
    return ".seek.mp4" if (container or _seek_passthrough_container()) == "mp4" else ".seek.ts"


def _resolution_str(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}" if int(width) > 0 and int(height) > 0 else ""


def _parse_resolution(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*x\s*(\d+)", str(value or ""), re.IGNORECASE)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _passthrough_resolution(width: int, height: int, mode: str) -> str:
    if mode == "alpha" and width > 0 and height > 0:
        out_w, out_h = alpha_output_size(width, height)
        return _resolution_str(out_w, out_h)
    if mode == "two_dvr" and width > 0 and height > 0:
        proc_w = min(int(width), 4096)
        if proc_w != int(width):
            proc_h = max(2, int(round(int(height) * (proc_w / float(width)))))
            proc_h -= proc_h & 1
        else:
            proc_h = int(height)
        return _resolution_str(proc_w * 2, proc_h)
    return _resolution_str(width, height)


def _is_two_d_source(path: Path, width: int = 0, height: int = 0) -> bool:
    return (
        not has_vr_filename_marker(path.stem)
        and not is_half_equirectangular_source(width, height)
    )


def _live_chapter_offsets(duration: float) -> list[int]:
    """Return start offsets for a live chapter directory.

    The first entry is always 0. Additional entries are spaced as evenly as
    possible while keeping the interval at or above the configured minimum and
    the total number of entries at or below the configured maximum.
    """
    max_items = max(1, int(PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS))
    min_interval = max(1, int(PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC))
    if duration <= min_interval or max_items == 1:
        return [0]
    duration_sec = int(math.ceil(duration))
    raw_interval = int(math.ceil(duration_sec / max_items))
    interval_sec = max(min_interval, int(math.ceil(raw_interval / 60.0)) * 60)
    offsets: list[int] = []
    offset = 0
    while len(offsets) < max_items and offset < duration_sec:
        if duration_sec - offset <= 60 and offset != 0:
            break
        offsets.append(offset)
        offset += interval_sec
    return offsets or [0]


def _uses_live_chapter_container(duration: float) -> bool:
    return len(_live_chapter_offsets(duration)) > 1


def _duration_seconds(duration: float) -> int:
    return max(0, int(math.ceil(float(duration or 0.0))))


def _live_time_group_ranges(duration: float) -> list[tuple[int, int]]:
    duration_sec = _duration_seconds(duration)
    if duration_sec <= 0:
        return [(0, 0)]
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < duration_sec:
        end = min(start + _TIME_INDEX_GROUP_SEC, duration_sec)
        ranges.append((start, end))
        start += _TIME_INDEX_GROUP_SEC
    return ranges or [(0, 0)]


def _live_time_minute_offsets(start: int, end: int) -> list[int]:
    start = max(0, int(start))
    end = max(start, int(end))
    offsets = list(range(start, end, _TIME_INDEX_MINUTE_SEC))
    return offsets or [start]


def _live_time_point_offsets(minute_start: int, duration: float) -> list[int]:
    duration_sec = _duration_seconds(duration)
    minute_start = max(0, int(minute_start))
    if duration_sec <= 0:
        return [minute_start]
    if minute_start >= duration_sec:
        return []
    minute_end = min(minute_start + _TIME_INDEX_MINUTE_SEC, duration_sec)
    offsets = list(range(minute_start, minute_end, _TIME_INDEX_POINT_SEC))
    return offsets or [minute_start]


def _live_time_force_hours(duration: float) -> bool:
    return _duration_seconds(duration) >= 3600


def _live_time_index_child_count(duration: float, level: str, start: int = 0, end: int = 0) -> int:
    if level == "index":
        groups = _live_time_group_ranges(duration)
        if len(groups) == 1:
            return len(_live_time_minute_offsets(*groups[0]))
        return len(groups)
    if level == "group":
        return len(_live_time_minute_offsets(start, end))
    if level == "minute":
        return len(_live_time_point_offsets(start, duration))
    return 0


def _child_count(path: Path) -> int:
    try:
        return get_media_index().child_count(path)
    except Exception as e:
        log.warning("indexed child count %s failed: %s", path, e)
        return 0


def _video_items(path: Path, parent_id: str, client_profile: str | None = None) -> list[dict]:
    return _video_items_from_index(path, parent_id, None, client_profile=client_profile)


def _video_items_from_index(
    path: Path,
    parent_id: str,
    child: IndexedChild | None,
    siblings: list[Path] | None = None,
    client_profile: str | None = None,
) -> list[dict]:
    base = f"http://{LAN_IP}:{HTTP_PORT}"
    pt_bps = _parse_bitrate(PASSTHROUGH_BITRATE)
    rel = _rel_key(path)
    quoted = quote(rel)
    size = child.size if child is not None else path.stat().st_size
    if child is not None and child.video is not None:
        duration = child.video.duration
        width = int(getattr(child.video, "width", 0) or 0)
        height = int(getattr(child.video, "height", 0) or 0)
        if width <= 0 or height <= 0:
            width, height = _parse_resolution(getattr(child.video, "resolution", ""))
        resolution = _resolution_str(width, height)
        backend_verdict = child.video.backend_verdict
        source_fps = float(getattr(child.video, "fps", 0.0) or 0.0)
    else:
        try:
            info = probe_cached(path)
            duration = info.duration
            width = int(info.width)
            height = int(info.height)
            resolution = _resolution_str(width, height)
            meta = probe_video_metadata(path)
            backend = select_backend(meta.timing, meta.codec, meta.color)
            backend_verdict = backend.verdict
            source_fps = float(meta.timing.source_fps or getattr(info, "fps", 0.0) or 0.0)
        except Exception as e:
            log.warning("probe %s failed: %s", rel, e)
            duration = 0.0
            width = 0
            height = 0
            resolution = ""
            backend_verdict = ""
            source_fps = 0.0

    if child is not None and child.video is not None and child.video.probe_error:
        duration = 0.0
        resolution = ""

    items: list[dict] = [
        {
            "id": f"v_{_versioned_rel(rel)}",
            "parent_id": parent_id,
            "title": _marked_original_title(path, child),
            "url": f"{base}/media/{quoted}",
            "thumb": f"{base}/thumb/{quoted}",
            "size": size,
            "duration": duration,
            "resolution": resolution,
            "bitrate": int(size * 8 / duration) if duration > 0 else 0,
            "mime": "video/mp4",
            "dlna_pn": "AVC_MP4_HP_HD_AAC",
            "frame_rate": None,
            "passthrough": False,
            "subtitles": [_subtitle_item(track) for track in find_external_subtitles(path)],
        }
    ]
    if _si_dlna_enabled() and get_si_mix().enabled and _has_si_sidecar(path):
        items.append(_si_container_item(path, parent_id, duration, width, height))
    if (
        is_offline_passthrough_output_name(path.name)
        or has_offline_passthrough_output(path, siblings)
        or _hide_passthrough_for_path(path, child)
    ):
        return items

    estimate_codec = PYNV_OUTPUT_CODEC
    if duration > 0:
        pt_size, pt_bps_est, _ = estimate_for_media(path, duration, estimate_codec)
    else:
        pt_size, pt_bps_est = 0, pt_bps
    # Keep the legacy pseudo-VOD /passthrough endpoint hidden from DLNA. When
    # seekable passthrough testing is explicitly enabled, add /passthrough_seek
    # beside the live entry instead of replacing it, so unknown or blocked
    # clients still have the stable /passthrough_live fallback visible.
    for mode in _passthrough_modes():
        if mode == "two_dvr" and (
            not _is_two_d_source(path, width, height)
            or has_offline_two_dvr_output(path, siblings)
            or width > 4096
        ):
            continue
        if mode != "two_dvr" and _seek_passthrough_dlna_enabled():
            query = _passthrough_seek_query(mode)
            seek_size = max(0, int(PASSTHROUGH_SEEK_HEADER_BYTES)) + pt_size if duration > 0 else 0
            seek_container = _seek_passthrough_container()
            items.append(
                {
                    "id": f"{_passthrough_seek_item_prefix(mode)}{_versioned_rel(rel)}",
                    "parent_id": parent_id,
                    "title": _passthrough_seek_title(path, mode, width, height),
                    "url": f"{base}/passthrough_seek/{quoted}{_seek_passthrough_route_suffix(seek_container)}" + (f"?{query}" if query else ""),
                    "thumb": f"{base}/thumb/{quoted}",
                    "size": seek_size,
                    "duration": duration,
                    "resolution": _passthrough_resolution(width, height, mode),
                    "bitrate": pt_bps_est,
                    "mime": _seek_passthrough_mime(seek_container),
                    "dlna_pn": _seek_passthrough_dlna_pn(seek_container),
                    "frame_rate": passthrough_frame_rate(source_fps),
                    "passthrough": True,
                    "passthrough_mode": mode,
                    "protocol_info": _seek_passthrough_protocol_info(),
                }
            )
        live_id = f"{_passthrough_live_prefix(mode)}{_versioned_rel(rel)}"
        items.append(
            {
                "container": True,
                "id": live_id,
                "parent_id": parent_id,
                "title": _prefixed_live_directory_title(path, mode, width, height),
                "child_count": len(_live_chapter_offsets(duration)) + 1,
            }
        )
    return items


def _probe_live_directory_context(path: Path, rel: str) -> tuple[float, int, int, float]:
    try:
        info = probe_cached(path)
        return (
            float(info.duration),
            int(info.width),
            int(info.height),
            float(getattr(info, "fps", 0.0) or 0.0),
        )
    except Exception as e:
        log.warning("probe live directory %s failed: %s", rel, e)
        return 0.0, 0, 0, 0.0


def _live_time_index_title(path: Path, mode: str, width: int, height: int, language: str | None = None) -> str:
    return f"[{_select_time_index_label(language)}]_{_prefixed_live_directory_title(path, mode, width, height)}"


def _live_directory_prefix(mode: str) -> str:
    if mode == "alpha":
        return "[ALPHA]"
    if mode == "green":
        return "[GREEN]"
    return ""


def _prefixed_live_directory_title(path: Path, mode: str, width: int = 0, height: int = 0) -> str:
    title = _passthrough_virtual_title(path, mode, width, height)
    prefix = _live_directory_prefix(mode)
    return f"{prefix}_{title}" if prefix else title


def _prefixed_time_index_directory_title(prefix: str, mode: str, virtual_title: str) -> str:
    mode_prefix = _live_directory_prefix(mode)
    return f"{prefix}_{mode_prefix}_{virtual_title}" if mode_prefix else f"{prefix}_{virtual_title}"


def _live_time_index_root_item(
    path: Path,
    mode: str,
    parent_id: str,
    duration: float,
    width: int,
    height: int,
    language: str | None = None,
) -> dict:
    rel = _rel_key(path)
    return {
        "container": True,
        "id": _live_time_index_id(LIVE_TIME_INDEX_PREFIX, rel, mode),
        "parent_id": parent_id,
        "title": _live_time_index_title(path, mode, width, height, language),
        "child_count": _live_time_index_child_count(duration, "index"),
    }


def _live_time_parent_group_range(minute_start: int, duration: float) -> tuple[int, int]:
    groups = _live_time_group_ranges(duration)
    for group_start, group_end in groups:
        if group_start <= minute_start < group_end or (group_start == group_end == minute_start):
            return group_start, group_end
    return groups[-1]


def _live_time_minute_items(
    path: Path,
    mode: str,
    parent_id: str,
    start: int,
    end: int,
    duration: float,
    virtual_title: str,
) -> list[dict]:
    rel = _rel_key(path)
    force_hours = _live_time_force_hours(duration)
    items: list[dict] = []
    for minute in _live_time_minute_offsets(start, end):
        items.append(
            {
                "container": True,
                "id": _live_time_index_id(LIVE_TIME_MINUTE_PREFIX, rel, mode, str(minute)),
                "parent_id": parent_id,
                "title": _prefixed_time_index_directory_title(_fmt_index_time(minute, force_hours), mode, virtual_title),
                "child_count": _live_time_index_child_count(duration, "minute", minute),
            }
        )
    return items


def _live_time_index_items(
    path: Path,
    mode: str,
    level: str,
    start: int = 0,
    end: int = 0,
    client_profile: str | None = None,
    language: str | None = None,
) -> list[dict]:
    if has_offline_passthrough_output(path):
        return []
    base = f"http://{LAN_IP}:{HTTP_PORT}"
    rel = _rel_key(path)
    quoted = quote(rel)
    duration, width, height, source_fps = _probe_live_directory_context(path, rel)
    if mode == "two_dvr" and (
        not _is_two_d_source(path, width, height)
        or has_offline_two_dvr_output(path)
        or width > 4096
    ):
        return []

    virtual_title = _passthrough_virtual_title(path, mode, width, height)
    force_hours = _live_time_force_hours(duration)
    if level == "index":
        parent_id = _live_time_index_id(LIVE_TIME_INDEX_PREFIX, rel, mode)
        groups = _live_time_group_ranges(duration)
        if len(groups) == 1:
            group_start, group_end = groups[0]
            return _live_time_minute_items(path, mode, parent_id, group_start, group_end, duration, virtual_title)
        return [
            {
                "container": True,
                "id": _live_time_index_id(LIVE_TIME_GROUP_PREFIX, rel, mode, f"{group_start}-{group_end}"),
                "parent_id": parent_id,
                "title": (
                    _prefixed_time_index_directory_title(
                        f"{_fmt_index_time(group_start, force_hours)}-{_fmt_index_time(group_end, force_hours)}",
                        mode,
                        virtual_title,
                    )
                ),
                "child_count": _live_time_index_child_count(duration, "group", group_start, group_end),
            }
            for group_start, group_end in groups
        ]
    if level == "group":
        parent_id = _live_time_index_id(LIVE_TIME_GROUP_PREFIX, rel, mode, f"{start}-{end}")
        return _live_time_minute_items(path, mode, parent_id, start, end, duration, virtual_title)
    if level != "minute":
        return []

    pt_bps = _parse_bitrate(PASSTHROUGH_BITRATE)
    if duration > 0:
        _, pt_bps_est, _ = estimate_for_media(path, duration, PYNV_OUTPUT_CODEC)
    else:
        pt_bps_est = pt_bps
    query = _passthrough_live_query(mode)
    parent_id = _live_time_index_id(LIVE_TIME_MINUTE_PREFIX, rel, mode, str(start))
    live_route_suffix = _live_route_hint_suffix(client_profile)
    live_omit_filelike_attrs = not _is_deovr_cds_client(client_profile)
    items: list[dict] = []
    for offset in _live_time_point_offsets(start, duration):
        remain = max(0.0, duration - float(offset)) if duration > 0 else 0.0
        items.append(
            {
                "id": _live_time_index_id(LIVE_TIME_POINT_PREFIX, rel, mode, str(offset)),
                "parent_id": parent_id,
                "title": f"{_fmt_index_time(offset, force_hours)}_{virtual_title}",
                "url": f"{base}/passthrough_live/{quoted}{live_route_suffix}?t={offset}" + (f"&{query}" if query else ""),
                "thumb": f"{base}/thumb/{quoted}",
                "size": 0,
                "duration": remain,
                "resolution": _passthrough_resolution(width, height, mode),
                "bitrate": pt_bps_est,
                "mime": "video/MP2T",
                "dlna_pn": "HEVC_TS_NA_ISO",
                "frame_rate": passthrough_frame_rate(source_fps),
                "passthrough": True,
                "passthrough_mode": mode,
                "protocol_info": _live_passthrough_protocol_info(client_profile),
                "omit_duration": live_omit_filelike_attrs,
                "omit_bitrate": live_omit_filelike_attrs,
            }
        )
    return items


def _live_time_index_metadata_item(
    path: Path,
    mode: str,
    level: str,
    start: int = 0,
    end: int = 0,
    language: str | None = None,
) -> dict | None:
    rel = _rel_key(path)
    duration, width, height, _source_fps = _probe_live_directory_context(path, rel)
    live_parent_id = f"{_passthrough_live_prefix(mode)}{_versioned_rel(rel)}"
    virtual_title = _passthrough_virtual_title(path, mode, width, height)
    force_hours = _live_time_force_hours(duration)
    if level == "index":
        return _live_time_index_root_item(path, mode, live_parent_id, duration, width, height, language)
    if level == "group":
        return {
            "container": True,
            "id": _live_time_index_id(LIVE_TIME_GROUP_PREFIX, rel, mode, f"{start}-{end}"),
            "parent_id": _live_time_index_id(LIVE_TIME_INDEX_PREFIX, rel, mode),
            "title": _prefixed_time_index_directory_title(
                f"{_fmt_index_time(start, force_hours)}-{_fmt_index_time(end, force_hours)}",
                mode,
                virtual_title,
            ),
            "child_count": _live_time_index_child_count(duration, "group", start, end),
        }
    if level == "minute":
        groups = _live_time_group_ranges(duration)
        parent_id = _live_time_index_id(LIVE_TIME_INDEX_PREFIX, rel, mode)
        if len(groups) > 1:
            group_start, group_end = _live_time_parent_group_range(start, duration)
            parent_id = _live_time_index_id(LIVE_TIME_GROUP_PREFIX, rel, mode, f"{group_start}-{group_end}")
        return {
            "container": True,
            "id": _live_time_index_id(LIVE_TIME_MINUTE_PREFIX, rel, mode, str(start)),
            "parent_id": parent_id,
            "title": _prefixed_time_index_directory_title(_fmt_index_time(start, force_hours), mode, virtual_title),
            "child_count": _live_time_index_child_count(duration, "minute", start),
        }
    return None


def _live_chapter_items(
    path: Path,
    mode: str,
    client_profile: str | None = None,
    language: str | None = None,
) -> list[dict]:
    if has_offline_passthrough_output(path):
        return []
    base = f"http://{LAN_IP}:{HTTP_PORT}"
    rel = _rel_key(path)
    quoted = quote(rel)
    parent_id = f"{_passthrough_live_prefix(mode)}{_versioned_rel(rel)}"
    query = _passthrough_live_query(mode)
    try:
        info = probe_cached(path)
        duration = info.duration
        width = int(info.width)
        height = int(info.height)
        source_fps = float(getattr(info, "fps", 0.0) or 0.0)
    except Exception as e:
        log.warning("probe live chapters %s failed: %s", rel, e)
        duration = 0.0
        width = 0
        height = 0
        source_fps = 0.0
    if mode == "two_dvr" and (
        not _is_two_d_source(path, width, height)
        or has_offline_two_dvr_output(path)
        or width > 4096
    ):
        return []
    pt_bps = _parse_bitrate(PASSTHROUGH_BITRATE)
    if duration > 0:
        _, pt_bps_est, _ = estimate_for_media(path, duration, PYNV_OUTPUT_CODEC)
    else:
        pt_bps_est = pt_bps
    items: list[dict] = []
    virtual_title = _passthrough_virtual_title(path, mode, width, height)
    suffix = {"alpha": "a", "two_dvr": "3"}.get(mode, "g")
    live_route_suffix = _live_route_hint_suffix(client_profile)
    live_omit_filelike_attrs = not _is_deovr_cds_client(client_profile)
    items.append(_live_time_index_root_item(path, mode, parent_id, duration, width, height, language))
    for offset in _live_chapter_offsets(duration):
        title = f"{_fmt_title_time(offset)}_{virtual_title}"
        remain = max(0.0, duration - float(offset)) if duration > 0 else 0.0
        items.append(
            {
                "id": f"lt{suffix}_{_versioned_rel(rel)}@{offset}",
                "parent_id": parent_id,
                "title": title,
                # See note above re: the optional ``.ts`` Skybox pipeline hint.
                "url": f"{base}/passthrough_live/{quoted}{live_route_suffix}?t={offset}" + (f"&{query}" if query else ""),
                "thumb": f"{base}/thumb/{quoted}",
                "size": 0,
                "duration": remain,
                "resolution": _passthrough_resolution(width, height, mode),
                "bitrate": pt_bps_est,
                "mime": "video/MP2T",
                "dlna_pn": "HEVC_TS_NA_ISO",
                "frame_rate": passthrough_frame_rate(source_fps),
                "passthrough": True,
                "passthrough_mode": mode,
                "protocol_info": _live_passthrough_protocol_info(client_profile),
                "omit_duration": live_omit_filelike_attrs,
                "omit_bitrate": live_omit_filelike_attrs,
            }
        )
    return items


def _children_for_dir(directory: Path, client_profile: str | None = None) -> list[dict]:
    directory = directory.resolve()
    parent_id = _folder_id(directory)
    items: list[dict] = []
    try:
        snapshot = get_media_index().list_directory(directory)
    except Exception as e:
        log.warning("index list %s failed: %s", directory, e)
        return items
    cache_key = (
        snapshot.key,
        snapshot.signature,
        PASSTHROUGH_OUTPUT_MODE,
        int(subtitle_output_enabled()),
        int(PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS),
        int(PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC),
        int(DLNA_IMAGE_ENABLED),
        int(get_si_mix().version),
        _DIDL_SCHEMA_VERSION,
        str(client_profile or ""),
    )
    cached = _dir_items_cache.get(cache_key)
    if cached is not None:
        return list(cached)
    sibling_paths = [child.path for child in snapshot.children]
    for child in snapshot.children:
        if child.is_dir:
            items.append(
                {
                    "container": True,
                    "id": _folder_id(child.path),
                    "parent_id": parent_id,
                    "title": child.name,
                    "child_count": _child_count(child.path),
                }
            )
        elif child.path.suffix.lower() in VIDEO_EXTS:
            items.extend(
                _video_items_from_index(
                    child.path,
                    parent_id,
                    child,
                    sibling_paths,
                    client_profile,
                )
            )
        elif DLNA_IMAGE_ENABLED and child.path.suffix.lower() in IMAGE_EXTS:
            items.append(_image_item_from_index(child.path, parent_id, child))
    if len(_dir_items_cache) >= _DIR_ITEMS_CACHE_MAX:
        _dir_items_cache.pop(next(iter(_dir_items_cache)))
    _dir_items_cache[cache_key] = list(items)
    return items


def _root_items(client_profile: str | None = None) -> list[dict]:
    if not MEDIA_LIBRARY.multi_root:
        return _children_for_dir(_root(), client_profile)
    return [
        {
            "container": True,
            "id": f"{FOLDER_PREFIX}{_versioned_rel(root.label)}",
            "parent_id": ROOT_ID,
            "title": root.label,
            "child_count": _child_count(root.path),
        }
        for root in MEDIA_LIBRARY.roots
    ]


def _items() -> list[dict]:
    return _root_items()


def _didl_for(items: list[dict]) -> str:
    out = [
        '<DIDL-Lite '
        f'xmlns="{DIDL_NS}" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" '
        'xmlns:dlna="urn:schemas-dlna-org:metadata-1-0/" '
        'xmlns:sec="http://www.sec.co.kr/">'
    ]
    for it in items:
        title = html.escape(it["title"])
        parent_id = html.escape(it.get("parent_id", ROOT_ID))
        if it.get("container"):
            out.append(
                f'<container id="{html.escape(it["id"])}" parentID="{parent_id}" '
                f'childCount="{int(it.get("child_count", 0))}" restricted="1">'
                f"<dc:title>{title}</dc:title>"
                f"<upnp:class>object.container.storageFolder</upnp:class>"
                f"</container>"
            )
            continue

        url = html.escape(it["url"])
        thumb = html.escape(str(it.get("thumb") or ""))
        size = int(it.get("size", 0) or 0)
        duration_value = it.get("duration")
        duration = _fmt_duration(float(duration_value or 0.0))
        resolution = str(it.get("resolution") or "")
        bitrate = int(it.get("bitrate", 0) or 0)
        mime = it["mime"]
        upnp_class = str(it.get("upnp_class") or "object.item.videoItem")

        if "protocol_info" in it:
            proto = it["protocol_info"]
        elif "op" in it:
            op = it["op"]
            ci = it.get("ci", "1")
            flags = it.get("flags", DLNA_FLAGS_BASE)
            proto = (
                f"http-get:*:{mime}:DLNA.ORG_PN={it['dlna_pn']};"
                f"DLNA.ORG_OP={op};"
                f"DLNA.ORG_CI={ci};"
                f"DLNA.ORG_FLAGS={flags}"
            )
        else:
            if it.get("passthrough") and PASSTHROUGH_SEEK_MODE == "bytes":
                op = DLNA_OP_BYTE_SEEK
            else:
                op = DLNA_OP_TIME_SEEK if it.get("passthrough") else DLNA_OP_BYTE_SEEK
            ci = "1" if it.get("passthrough") else "0"
            if it.get("passthrough") and PASSTHROUGH_SEEK_MODE == "bytes":
                flags = DLNA_FLAGS_BASE
            else:
                flags = DLNA_FLAGS_TIME_SEEK if it.get("passthrough") else DLNA_FLAGS_BASE
            proto = (
                f"http-get:*:{mime}:DLNA.ORG_PN={it['dlna_pn']};"
                f"DLNA.ORG_OP={op};"
                f"DLNA.ORG_CI={ci};"
                f"DLNA.ORG_FLAGS={flags}"
            )

        attrs: list[str] = []
        if size > 0:
            attrs.append(f'size="{size}"')
        if duration_value is not None and not it.get("omit_duration"):
            attrs.append(f'duration="{duration}"')
        if bitrate > 0 and not it.get("omit_bitrate"):
            attrs.append(f'bitrate="{bitrate}"')
        if resolution:
            attrs.append(f'resolution="{resolution}"')
        if it.get("frame_rate"):
            attrs.append(f'frameRate="{it["frame_rate"]}"')
        attrs.append(f'protocolInfo="{proto}"')
        res_attrs = " ".join(attrs)

        subtitle_xml = []
        for sub in it.get("subtitles", []):
            sub_url = html.escape(sub["url"])
            sub_mime = html.escape(sub["mime"])
            sub_type = html.escape(sub["type"])
            lang = str(sub.get("lang") or "")
            lang_attr = f' xml:lang="{html.escape(lang)}"' if lang else ""
            subtitle_xml.append(f'<res protocolInfo="http-get:*:{sub_mime}:*"{lang_attr}>{sub_url}</res>')
            subtitle_xml.append(f'<sec:CaptionInfoEx sec:type="{sub_type}">{sub_url}</sec:CaptionInfoEx>')
            subtitle_xml.append(f'<sec:CaptionInfo sec:type="{sub_type}">{sub_url}</sec:CaptionInfo>')

        thumb_profile = html.escape(str(it.get("thumb_profile") or "JPEG_TN"))
        album_art_xml = (
            f'<upnp:albumArtURI dlna:profileID="{thumb_profile}">{thumb}</upnp:albumArtURI>'
            if thumb and thumb_profile
            else (f"<upnp:albumArtURI>{thumb}</upnp:albumArtURI>" if thumb else "")
        )
        out.append(
            f'<item id="{html.escape(it["id"])}" parentID="{parent_id}" restricted="1">'
            f"<dc:title>{title}</dc:title>"
            f"<upnp:class>{html.escape(upnp_class)}</upnp:class>"
            f"{album_art_xml}"
            f"<res {res_attrs}>{url}</res>"
            f"{''.join(subtitle_xml)}"
            f"</item>"
        )
    out.append("</DIDL-Lite>")
    return "".join(out)


def _metadata_didl_for_item(item: dict) -> str:
    return _didl_for([item])


_SOAP_RE = re.compile(r"<([\w:]+)>([\s\S]*?)</\1>")
_MAX_SOAP_BODY_BYTES = 1024 * 1024
_UNSAFE_XML_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


def _parse_soap_args(body: bytes) -> dict:
    if len(body) > _MAX_SOAP_BODY_BYTES:
        log.warning("SOAP body rejected: too large (%d bytes)", len(body))
        return {}
    text = body.decode("utf-8", errors="ignore")
    if _UNSAFE_XML_RE.search(text):
        log.warning("SOAP body rejected: DTD/entity declarations are not allowed")
        return {}
    args: dict = {}
    try:
        root = ET.fromstring(text)
        for elem in root.iter():
            tag = elem.tag.rsplit("}", 1)[-1].split(":")[-1]
            value = (elem.text or "").strip()
            if value:
                args[tag] = value
        return args
    except ET.ParseError:
        pass
    for m in _SOAP_RE.finditer(text):
        tag = m.group(1).split(":")[-1]
        args[tag] = m.group(2).strip()
    return args


def _wrap_soap(action: str, body_xml: str) -> bytes:
    env = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        f'<u:{action}Response xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">'
        f"{body_xml}"
        f"</u:{action}Response>"
        "</s:Body></s:Envelope>"
    )
    return env.encode("utf-8")


def _metadata_didl_for_dir(directory: Path) -> str:
    if MEDIA_LIBRARY.multi_root and directory is None:
        return (
            f'<DIDL-Lite xmlns="{DIDL_NS}" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            f'<container id="{ROOT_ID}" parentID="-1" childCount="{len(MEDIA_LIBRARY.roots)}" restricted="1">'
            "<dc:title>PT Videos</dc:title>"
            "<upnp:class>object.container.storageFolder</upnp:class>"
            "</container></DIDL-Lite>"
        )
    directory = (directory or _root()).resolve()
    title = "PT Videos" if directory == _root() and not MEDIA_LIBRARY.multi_root else _rel_key(directory).split("/", 1)[0] if MEDIA_LIBRARY.multi_root and directory in [root.path for root in MEDIA_LIBRARY.roots] else directory.name
    return (
        f'<DIDL-Lite xmlns="{DIDL_NS}" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        f'<container id="{html.escape(_folder_id(directory))}" '
        f'parentID="{html.escape(_parent_id_for_dir(directory))}" '
        f'childCount="{_child_count(directory)}" restricted="1">'
        f"<dc:title>{html.escape(title)}</dc:title>"
        "<upnp:class>object.container.storageFolder</upnp:class>"
        "</container></DIDL-Lite>"
    )


def _metadata_didl_for_live(path: Path, mode: str) -> str:
    rel = _rel_key(path)
    live_id = f"{_passthrough_live_prefix(mode)}{_versioned_rel(rel)}"
    try:
        info = probe_cached(path)
        duration = info.duration
        width = int(info.width)
        height = int(info.height)
    except Exception:
        duration = 0.0
        width = 0
        height = 0
    return (
        f'<DIDL-Lite xmlns="{DIDL_NS}" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        f'<container id="{html.escape(live_id)}" '
        f'parentID="{html.escape(_folder_id(path.parent))}" '
        f'childCount="{len(_live_chapter_offsets(duration)) + 1}" restricted="1">'
        f"<dc:title>{html.escape(_prefixed_live_directory_title(path, mode, width, height))}</dc:title>"
        "<upnp:class>object.container.storageFolder</upnp:class>"
        "</container></DIDL-Lite>"
    )


def _metadata_didl_for_si_dir(path: Path) -> str:
    rel = _rel_key(path)
    duration, width, height, _fps = _probe_live_directory_context(path, rel)
    si_id = _si_time_index_id(SI_DIR_PREFIX, rel)
    return (
        f'<DIDL-Lite xmlns="{DIDL_NS}" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        f'<container id="{html.escape(si_id)}" '
        f'parentID="{html.escape(_folder_id(path.parent))}" '
        f'childCount="{_si_directory_child_count(duration)}" restricted="1">'
        f"<dc:title>{html.escape(_si_directory_title(path, width, height))}</dc:title>"
        "<upnp:class>object.container.storageFolder</upnp:class>"
        "</container></DIDL-Lite>"
    )


def handle_soap(
    soap_action: str,
    body: bytes,
    client_profile: str | None = None,
    language: str | None = None,
) -> tuple[bytes, int]:
    action = soap_action.strip('"').split("#")[-1]
    args = _parse_soap_args(body)

    if action == "Browse":
        object_id = args.get("ObjectID", ROOT_ID)
        flag = args.get("BrowseFlag", "BrowseDirectChildren")
        start = int(args.get("StartingIndex", "0") or 0)
        count = int(args.get("RequestedCount", "0") or 0)
        directory = _id_to_dir(object_id)
        live = _id_to_live(object_id)
        time_index = _id_to_live_time_index(object_id)
        seek = _id_to_seek(object_id)
        si = _id_to_si(object_id)
        si_dir = _id_to_si_dir(object_id)
        si_time = _id_to_si_time_index(object_id)
        si_point = _id_to_si_point(object_id)
        image = _id_to_image(object_id)

        if time_index is not None:
            index_path, index_mode, index_level, index_start, index_end = time_index
            all_items = _live_time_index_items(
                index_path,
                index_mode,
                index_level,
                index_start,
                index_end,
                client_profile,
                language,
            )
        elif si_time is not None:
            si_path, si_level, si_start, si_end = si_time
            all_items = _si_time_index_items(si_path, si_level, si_start, si_end, client_profile, language)
        elif si_dir is not None:
            all_items = _si_chapter_items(si_dir, client_profile, language)
        elif live is not None:
            live_path, live_mode = live
            all_items = _live_chapter_items(live_path, live_mode, client_profile, language)
        elif object_id == ROOT_ID and MEDIA_LIBRARY.multi_root:
            all_items = _root_items(client_profile)
        elif directory is None or not directory.is_dir():
            all_items: list[dict] = []
        else:
            all_items = _children_for_dir(directory, client_profile)
        if flag == "BrowseMetadata":
            if seek is not None:
                seek_path, seek_mode = seek
                seek_items = [
                    item for item in _video_items(seek_path, _folder_id(seek_path.parent))
                    if item.get("passthrough") and item.get("passthrough_mode") == seek_mode and "/passthrough_seek/" in item.get("url", "")
                ]
                didl = _metadata_didl_for_item(seek_items[0]) if seek_items else _didl_for([])
            elif time_index is not None:
                index_path, index_mode, index_level, index_start, index_end = time_index
                item = _live_time_index_metadata_item(
                    index_path,
                    index_mode,
                    index_level,
                    index_start,
                    index_end,
                    language,
                )
                didl = _metadata_didl_for_item(item) if item else _didl_for([])
            elif live is not None:
                live_path, live_mode = live
                didl = _metadata_didl_for_live(live_path, live_mode)
            elif image is not None:
                didl = _metadata_didl_for_item(_image_item_from_index(image, _folder_id(image.parent)))
            elif si_point is not None:
                point_path, point_offset, point_prefix = si_point
                point_item = _si_point_metadata_item(point_path, point_offset, point_prefix)
                didl = _metadata_didl_for_item(point_item) if point_item else _didl_for([])
            elif si_time is not None:
                meta_path, meta_level, meta_start, meta_end = si_time
                meta_item = _si_time_index_metadata_item(meta_path, meta_level, meta_start, meta_end, language)
                didl = _metadata_didl_for_item(meta_item) if meta_item else _didl_for([])
            elif si_dir is not None:
                didl = _metadata_didl_for_si_dir(si_dir)
            elif si is not None:
                si_items = [
                    item for item in _video_items(si, _folder_id(si.parent), client_profile)
                    if item.get("passthrough_mode") == "si_mix"
                ]
                didl = _metadata_didl_for_item(si_items[0]) if si_items else _didl_for([])
            else:
                didl = _metadata_didl_for_dir(directory or _root())
            return _wrap_soap(
                "Browse",
                f"<Result>{html.escape(didl)}</Result>"
                f"<NumberReturned>1</NumberReturned>"
                f"<TotalMatches>1</TotalMatches>"
                f"<UpdateID>{_system_update_id()}</UpdateID>",
            ), 200

        end = start + count if count > 0 else len(all_items)
        page = all_items[start:end]
        didl = _didl_for(page)
        body_xml = (
            f"<Result>{html.escape(didl)}</Result>"
            f"<NumberReturned>{len(page)}</NumberReturned>"
            f"<TotalMatches>{len(all_items)}</TotalMatches>"
            f"<UpdateID>{_system_update_id()}</UpdateID>"
        )
        return _wrap_soap("Browse", body_xml), 200

    if action == "GetSearchCapabilities":
        return _wrap_soap("GetSearchCapabilities", "<SearchCaps></SearchCaps>"), 200
    if action == "GetSortCapabilities":
        return _wrap_soap("GetSortCapabilities", "<SortCaps></SortCaps>"), 200
    if action == "GetSystemUpdateID":
        return _wrap_soap("GetSystemUpdateID", f"<Id>{_system_update_id()}</Id>"), 200

    fault = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        "<s:Body><s:Fault><faultcode>s:Client</faultcode>"
        "<faultstring>UPnPError</faultstring><detail>"
        '<UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
        "<errorCode>401</errorCode><errorDescription>Invalid Action</errorDescription>"
        "</UPnPError></detail></s:Fault></s:Body></s:Envelope>"
    )
    return fault.encode("utf-8"), 401
