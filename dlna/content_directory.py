"""UPnP ContentDirectory implementation for the local media library.

The root ObjectID maps to VIDEO_DIR. Physical subdirectories are exposed as
DIDL containers with ids of the form ``d_<relative/path>``. Each normal video
file is exposed as the raw media item plus a passthrough-live item. The older
pseudo-VOD passthrough endpoint still exists in HTTP code but is hidden from
DLNA while client seek behavior is being evaluated. The passthrough-live item
is a chapter container, allowing clients to choose a start time without relying
on HTTP Range seeking.
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
    LAN_IP,
    PASSTHROUGH_BITRATE,
    PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS,
    PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC,
    PASSTHROUGH_MKV_LIVE_POLICY,
    PASSTHROUGH_OUTPUT_MODE,
    PASSTHROUGH_SEEK_MODE,
    PASSTHROUGH_SUFFIX,
    ALPHA_2D_ENABLE,
    ALPHA_2D_PROJECTION,
    MEDIA_LIBRARY,
    VIDEO_DIR,
    VIDEO_EXTS,
)
from dlna.profiles import passthrough_frame_rate
from pipeline.alpha_packer import alpha_output_size, is_sbs_vr_size
from pipeline.ffmpeg_io import probe_cached
from utils.bitrate_estimator import estimate_for_media
from utils.logger import get
from utils.media_index import IndexedChild, get_media_index
from utils.offline_outputs import has_offline_passthrough_output, is_offline_passthrough_output_name
from utils.subtitles import SubtitleTrack, find_external_subtitles, subtitle_output_enabled
from utils.video_metadata import probe_video_metadata, select_backend

log = get("cds")

ROOT_ID = "0"
FOLDER_PREFIX = "d_"
LEGACY_FOLDER_PREFIX = "d:"
LIVE_PREFIX = "pl_"
LEGACY_LIVE_PREFIX = "pl:"
ALPHA_LIVE_PREFIX = "pla_"
LEGACY_ALPHA_LIVE_PREFIX = "pla:"
LIVE_ITEM_PREFIX = "lg_"
ALPHA_LIVE_ITEM_PREFIX = "la_"
PYNV_OUTPUT_CODEC = "hevc"
DLNA_FLAGS_BASE = "01700000000000000000000000000000"
DLNA_FLAGS_TIME_SEEK = "41700000000000000000000000000000"
DIDL_NS = "urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/"

_DIR_ITEMS_CACHE_MAX = 256
_DIDL_SCHEMA_VERSION = 4
_SYSTEM_UPDATE_ID = _DIDL_SCHEMA_VERSION
_dir_items_cache: dict[tuple, list[dict]] = {}
_LIVE_MAX_SIDE = 8192
_NO_LIVE_PREFIX = "[NoLive] "


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


def _root() -> Path:
    return VIDEO_DIR.resolve()


def _rel_key(path: Path) -> str:
    return MEDIA_LIBRARY.path_to_key(path)


def _folder_id(path: Path) -> str:
    rel = _rel_key(path)
    return ROOT_ID if not rel or rel == "." else f"{FOLDER_PREFIX}{rel}"


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
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path):
        return path
    return None


def _id_to_live(object_id: str) -> tuple[Path, str] | None:
    mode = "green"
    prefix = LIVE_PREFIX
    if object_id.startswith(ALPHA_LIVE_ITEM_PREFIX):
        mode = "alpha"
        prefix = ALPHA_LIVE_ITEM_PREFIX
    elif object_id.startswith(LIVE_ITEM_PREFIX):
        prefix = LIVE_ITEM_PREFIX
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
    path = MEDIA_LIBRARY.key_to_path(rel)
    if path is not None and MEDIA_LIBRARY.contains(path) and path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return path, mode
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
    if (
        is_offline_passthrough_output_name(path.name)
        or PASSTHROUGH_OUTPUT_MODE == "none"
        or has_offline_passthrough_output(path)
        or _hide_passthrough_for_path(path, child)
    ):
        return 1
    return 3 if PASSTHROUGH_OUTPUT_MODE == "all" else 2


def _marked_original_title(path: Path, child: IndexedChild | None = None) -> str:
    title = path.stem
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
    if PASSTHROUGH_OUTPUT_MODE == "none":
        return ()
    if PASSTHROUGH_OUTPUT_MODE == "all":
        return ("green", "alpha")
    if PASSTHROUGH_OUTPUT_MODE == "alpha":
        return ("alpha",)
    return ("green",)


def _subtitle_item(track: SubtitleTrack) -> dict:
    rel = _rel_key(track.path)
    return {
        "url": f"http://{LAN_IP}:{HTTP_PORT}/subs/{quote(rel)}",
        "lang": track.lang,
        "type": track.kind,
        "mime": track.mime,
    }


def _alpha_virtual_suffix(width: int = 0, height: int = 0) -> str:
    if (
        ALPHA_2D_ENABLE
        and str(ALPHA_2D_PROJECTION).lower() == "flat3d"
        and int(width) > 0
        and int(height) > 0
        and not is_sbs_vr_size(int(width), int(height))
    ):
        return "3D_alpha"
    return "FISHEYE_alpha"


def _passthrough_virtual_title(path: Path, mode: str, width: int = 0, height: int = 0) -> str:
    if mode == "alpha":
        return f"{path.stem}_{_alpha_virtual_suffix(width, height)}_live"
    return f"{path.stem}{PASSTHROUGH_SUFFIX}_live"


def _passthrough_live_prefix(mode: str) -> str:
    return ALPHA_LIVE_PREFIX if mode == "alpha" else LIVE_PREFIX


def _passthrough_live_item_prefix(mode: str) -> str:
    return ALPHA_LIVE_ITEM_PREFIX if mode == "alpha" else LIVE_ITEM_PREFIX


def _passthrough_live_query(mode: str) -> str:
    version = f"ptv={_DIDL_SCHEMA_VERSION}"
    if mode in {"green", "alpha"}:
        return f"mode={mode}&{version}"
    return version


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
    return _resolution_str(width, height)


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


def _child_count(path: Path) -> int:
    try:
        return get_media_index().child_count(path)
    except Exception as e:
        log.warning("indexed child count %s failed: %s", path, e)
        return 0


def _video_items(path: Path, parent_id: str) -> list[dict]:
    return _video_items_from_index(path, parent_id, None)


def _video_items_from_index(
    path: Path,
    parent_id: str,
    child: IndexedChild | None,
    siblings: list[Path] | None = None,
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
        except Exception as e:
            log.warning("probe %s failed: %s", rel, e)
            duration = 0.0
            width = 0
            height = 0
            resolution = ""
            backend_verdict = ""

    if child is not None and child.video is not None and child.video.probe_error:
        duration = 0.0
        resolution = ""

    items: list[dict] = [
        {
            "id": f"v_{rel}",
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
    # Keep the pseudo-VOD /passthrough endpoint implemented but hidden from
    # DLNA listings for now. Several clients issue aggressive probe/seek
    # requests that are a poor fit for on-demand generated media.
    for mode in _passthrough_modes():
        query = _passthrough_live_query(mode)
        suffix = "alpha" if mode == "alpha" else "green"
        if _uses_live_chapter_container(duration):
            live_id = f"{_passthrough_live_prefix(mode)}{rel}"
            items.append(
                {
                    "container": True,
                    "id": live_id,
                    "parent_id": parent_id,
                    "title": _passthrough_virtual_title(path, mode, width, height),
                    "child_count": len(_live_chapter_offsets(duration)),
                }
            )
        else:
            items.append(
                {
                    "id": f"{_passthrough_live_item_prefix(mode)}{rel}",
                    "parent_id": parent_id,
                    "title": _passthrough_virtual_title(path, mode, width, height),
                    "url": f"{base}/passthrough_live/{quoted}" + (f"?{query}" if query else ""),
                    "thumb": f"{base}/thumb/{quoted}",
                    "size": 0,
                    "duration": duration,
                    "resolution": _passthrough_resolution(width, height, mode),
                    "bitrate": pt_bps_est,
                    "mime": "video/MP2T",
                    "dlna_pn": "HEVC_TS_NA_ISO",
                    "frame_rate": passthrough_frame_rate(),
                    "passthrough": True,
                    "passthrough_mode": mode,
                    "op": "10",
                    "ci": "1",
                    "flags": DLNA_FLAGS_TIME_SEEK,
                }
            )
    return items


def _live_chapter_items(path: Path, mode: str) -> list[dict]:
    if has_offline_passthrough_output(path):
        return []
    base = f"http://{LAN_IP}:{HTTP_PORT}"
    rel = _rel_key(path)
    quoted = quote(rel)
    parent_id = f"{_passthrough_live_prefix(mode)}{rel}"
    query = _passthrough_live_query(mode)
    try:
        info = probe_cached(path)
        duration = info.duration
        width = int(info.width)
        height = int(info.height)
    except Exception as e:
        log.warning("probe live chapters %s failed: %s", rel, e)
        duration = 0.0
        width = 0
        height = 0
    pt_bps = _parse_bitrate(PASSTHROUGH_BITRATE)
    if duration > 0:
        _, pt_bps_est, _ = estimate_for_media(path, duration, PYNV_OUTPUT_CODEC)
    else:
        pt_bps_est = pt_bps
    items: list[dict] = []
    virtual_title = _passthrough_virtual_title(path, mode, width, height)
    suffix = "alpha" if mode == "alpha" else "green"
    for offset in _live_chapter_offsets(duration):
        title = f"{_fmt_title_time(offset)}_{virtual_title}"
        remain = max(0.0, duration - float(offset)) if duration > 0 else 0.0
        items.append(
            {
                "id": f"lt{suffix[0]}_{rel}@{offset}",
                "parent_id": parent_id,
                "title": title,
                "url": f"{base}/passthrough_live/{quoted}?t={offset}" + (f"&{query}" if query else ""),
                "thumb": f"{base}/thumb/{quoted}",
                "size": 0,
                "duration": remain,
                "resolution": _passthrough_resolution(width, height, mode),
                "bitrate": pt_bps_est,
                "mime": "video/MP2T",
                "dlna_pn": "HEVC_TS_NA_ISO",
                "frame_rate": passthrough_frame_rate(),
                "passthrough": True,
                "passthrough_mode": mode,
                "op": "10",
                "ci": "1",
                "flags": DLNA_FLAGS_TIME_SEEK,
            }
        )
    return items


def _children_for_dir(directory: Path) -> list[dict]:
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
        _DIDL_SCHEMA_VERSION,
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
            items.extend(_video_items_from_index(child.path, parent_id, child, sibling_paths))
    if len(_dir_items_cache) >= _DIR_ITEMS_CACHE_MAX:
        _dir_items_cache.pop(next(iter(_dir_items_cache)))
    _dir_items_cache[cache_key] = list(items)
    return items


def _root_items() -> list[dict]:
    if not MEDIA_LIBRARY.multi_root:
        return _children_for_dir(_root())
    return [
        {
            "container": True,
            "id": f"{FOLDER_PREFIX}{root.label}",
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
        thumb = html.escape(it["thumb"])
        size = it["size"]
        duration = _fmt_duration(it["duration"])
        resolution = it["resolution"]
        bitrate = it["bitrate"]
        mime = it["mime"]

        if "op" in it:
            op = it["op"]
            ci = it.get("ci", "1")
            flags = it.get("flags", DLNA_FLAGS_BASE)
        else:
            if it["passthrough"] and PASSTHROUGH_SEEK_MODE == "bytes":
                op = "01"
            else:
                op = "10" if it["passthrough"] else "01"
            ci = "1" if it["passthrough"] else "0"
            if it["passthrough"] and PASSTHROUGH_SEEK_MODE == "bytes":
                flags = DLNA_FLAGS_BASE
            else:
                flags = DLNA_FLAGS_TIME_SEEK if it["passthrough"] else DLNA_FLAGS_BASE
        proto = (
            f"http-get:*:{mime}:DLNA.ORG_PN={it['dlna_pn']};"
            f"DLNA.ORG_OP={op};"
            f"DLNA.ORG_CI={ci};"
            f"DLNA.ORG_FLAGS={flags}"
        )

        attrs: list[str] = []
        if size > 0:
            attrs.append(f'size="{size}"')
        attrs.append(f'duration="{duration}"')
        if bitrate > 0:
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

        out.append(
            f'<item id="{html.escape(it["id"])}" parentID="{parent_id}" restricted="1">'
            f"<dc:title>{title}</dc:title>"
            f"<upnp:class>object.item.videoItem</upnp:class>"
            f'<upnp:albumArtURI dlna:profileID="JPEG_TN">{thumb}</upnp:albumArtURI>'
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
    live_id = f"{_passthrough_live_prefix(mode)}{rel}"
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
        f'childCount="{len(_live_chapter_offsets(duration))}" restricted="1">'
        f"<dc:title>{html.escape(_passthrough_virtual_title(path, mode, width, height))}</dc:title>"
        "<upnp:class>object.container.storageFolder</upnp:class>"
        "</container></DIDL-Lite>"
    )


def handle_soap(soap_action: str, body: bytes) -> tuple[bytes, int]:
    action = soap_action.strip('"').split("#")[-1]
    args = _parse_soap_args(body)

    if action == "Browse":
        object_id = args.get("ObjectID", ROOT_ID)
        flag = args.get("BrowseFlag", "BrowseDirectChildren")
        start = int(args.get("StartingIndex", "0") or 0)
        count = int(args.get("RequestedCount", "0") or 0)
        directory = _id_to_dir(object_id)
        live = _id_to_live(object_id)

        if live is not None:
            live_path, live_mode = live
            all_items = _live_chapter_items(live_path, live_mode)
        elif object_id == ROOT_ID and MEDIA_LIBRARY.multi_root:
            all_items = _root_items()
        elif directory is None or not directory.is_dir():
            all_items: list[dict] = []
        else:
            all_items = _children_for_dir(directory)
        if flag == "BrowseMetadata":
            if live is not None:
                live_path, live_mode = live
                try:
                    info = probe_cached(live_path)
                    duration = info.duration
                except Exception:
                    duration = 0.0
                if _uses_live_chapter_container(duration):
                    didl = _metadata_didl_for_live(live_path, live_mode)
                else:
                    live_items = [
                        item for item in _video_items(live_path, _folder_id(live_path.parent))
                        if item.get("passthrough") and item.get("passthrough_mode") == live_mode
                    ]
                    didl = _metadata_didl_for_item(live_items[0]) if live_items else _didl_for([])
            else:
                didl = _metadata_didl_for_dir(directory or _root())
            return _wrap_soap(
                "Browse",
                f"<Result>{html.escape(didl)}</Result>"
                f"<NumberReturned>1</NumberReturned>"
                f"<TotalMatches>1</TotalMatches>"
                f"<UpdateID>{_SYSTEM_UPDATE_ID}</UpdateID>",
            ), 200

        end = start + count if count > 0 else len(all_items)
        page = all_items[start:end]
        didl = _didl_for(page)
        body_xml = (
            f"<Result>{html.escape(didl)}</Result>"
            f"<NumberReturned>{len(page)}</NumberReturned>"
            f"<TotalMatches>{len(all_items)}</TotalMatches>"
            f"<UpdateID>{_SYSTEM_UPDATE_ID}</UpdateID>"
        )
        return _wrap_soap("Browse", body_xml), 200

    if action == "GetSearchCapabilities":
        return _wrap_soap("GetSearchCapabilities", "<SearchCaps></SearchCaps>"), 200
    if action == "GetSortCapabilities":
        return _wrap_soap("GetSortCapabilities", "<SortCaps></SortCaps>"), 200
    if action == "GetSystemUpdateID":
        return _wrap_soap("GetSystemUpdateID", f"<Id>{_SYSTEM_UPDATE_ID}</Id>"), 200

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
