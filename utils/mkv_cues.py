"""Lightweight Matroska Cues placement probe.

The real-time PyNv path uses random frame access. Large MKV files whose Cues
are missing from the head area can make that path stall inside native decoder
code. This probe only reads a small prefix and looks for a Segment SeekHead
entry that points to the Cues element.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MATROSKA_SEGMENT_ID = 0x18538067
MATROSKA_SEEKHEAD_ID = 0x114D9B74
MATROSKA_SEEK_ID = 0x4DBB
MATROSKA_SEEK_ID_ID = 0x53AB
MATROSKA_SEEK_POSITION_ID = 0x53AC
MATROSKA_CUES_ID = 0x1C53BB6B

PROBE_PREFIX_BYTES = 2 * 1024 * 1024
HEAD_CUES_MAX_ABSOLUTE_BYTES = 64 * 1024 * 1024
HEAD_CUES_MAX_RATIO = 0.02


@dataclass(frozen=True)
class MkvCuesInfo:
    status: str = ""
    position: int = -1
    reason: str = ""

    @property
    def needs_fix(self) -> bool:
        return self.status in {"tail", "missing", "unknown"}


def _read_vint(data: bytes, offset: int, *, keep_marker: bool) -> tuple[int, int] | None:
    if offset >= len(data):
        return None
    first = data[offset]
    mask = 0x80
    length = 1
    while length <= 8 and not (first & mask):
        mask >>= 1
        length += 1
    if length > 8 or offset + length > len(data):
        return None
    value = first if keep_marker else (first & (mask - 1))
    for i in range(1, length):
        value = (value << 8) | data[offset + i]
    return value, length


def _read_uint(data: bytes, offset: int, size: int) -> int | None:
    if size < 0 or size > 8 or offset + size > len(data):
        return None
    value = 0
    for b in data[offset : offset + size]:
        value = (value << 8) | b
    return value


def _read_element(data: bytes, offset: int) -> tuple[int, int, int, int] | None:
    elem_id = _read_vint(data, offset, keep_marker=True)
    if elem_id is None:
        return None
    size_pos = offset + elem_id[1]
    elem_size = _read_vint(data, size_pos, keep_marker=False)
    if elem_size is None:
        return None
    header_size = elem_id[1] + elem_size[1]
    payload_pos = offset + header_size
    return elem_id[0], elem_size[0], payload_pos, header_size


def _find_segment(data: bytes) -> tuple[int, int] | None:
    offset = 0
    limit = len(data)
    while offset < limit:
        elem = _read_element(data, offset)
        if elem is None:
            offset += 1
            continue
        elem_id, elem_size, payload_pos, header_size = elem
        if elem_id == MATROSKA_SEGMENT_ID:
            return payload_pos, header_size
        next_offset = payload_pos + elem_size
        offset = next_offset if next_offset > offset else offset + 1
    return None


def _find_direct_cues(data: bytes, segment_start: int, file_size: int) -> MkvCuesInfo | None:
    offset = segment_start
    while offset < len(data):
        elem = _read_element(data, offset)
        if elem is None:
            break
        elem_id, elem_size, payload_pos, _header_size = elem
        if elem_id == MATROSKA_CUES_ID:
            return _classify_cues_position(offset, file_size, "Cues element is in file prefix")
        if payload_pos + elem_size <= offset:
            break
        offset = payload_pos + elem_size
    return None


def _find_seekhead(data: bytes, segment_start: int) -> tuple[int, int] | None:
    offset = segment_start
    while offset < len(data):
        elem = _read_element(data, offset)
        if elem is None:
            break
        elem_id, elem_size, payload_pos, _header_size = elem
        if elem_id == MATROSKA_SEEKHEAD_ID:
            return payload_pos, min(payload_pos + elem_size, len(data))
        next_offset = payload_pos + elem_size
        if next_offset <= offset or next_offset > len(data):
            break
        offset = next_offset
    return None


def _seekhead_cues_position(data: bytes, start: int, end: int) -> int | None:
    offset = start
    while offset < end:
        elem = _read_element(data, offset)
        if elem is None:
            break
        elem_id, elem_size, payload_pos, _header_size = elem
        elem_end = min(payload_pos + elem_size, end)
        if elem_id == MATROSKA_SEEK_ID:
            seek_id = None
            seek_position = None
            inner = payload_pos
            while inner < elem_end:
                child = _read_element(data, inner)
                if child is None:
                    break
                child_id, child_size, child_payload, _child_header = child
                if child_id == MATROSKA_SEEK_ID_ID:
                    seek_id = _read_uint(data, child_payload, child_size)
                elif child_id == MATROSKA_SEEK_POSITION_ID:
                    seek_position = _read_uint(data, child_payload, child_size)
                next_inner = child_payload + child_size
                if next_inner <= inner:
                    break
                inner = next_inner
            if seek_id == MATROSKA_CUES_ID:
                return seek_position
        next_offset = payload_pos + elem_size
        if next_offset <= offset:
            break
        offset = next_offset
    return None


def _classify_cues_position(position: int, file_size: int, reason: str) -> MkvCuesInfo:
    ratio_limit = int(file_size * HEAD_CUES_MAX_RATIO) if file_size > 0 else 0
    head_limit = max(PROBE_PREFIX_BYTES, min(HEAD_CUES_MAX_ABSOLUTE_BYTES, ratio_limit or HEAD_CUES_MAX_ABSOLUTE_BYTES))
    if position <= head_limit:
        return MkvCuesInfo("head", int(position), reason)
    return MkvCuesInfo("tail", int(position), f"Cues position {position} is beyond head limit {head_limit}")


def probe_mkv_cues(path: Path) -> MkvCuesInfo:
    if path.suffix.lower() != ".mkv":
        return MkvCuesInfo()
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            data = f.read(PROBE_PREFIX_BYTES)
    except OSError as e:
        return MkvCuesInfo("unknown", -1, f"read failed: {e}")
    if not data:
        return MkvCuesInfo("unknown", -1, "empty or unreadable file prefix")

    segment = _find_segment(data)
    if segment is None:
        return MkvCuesInfo("unknown", -1, "Matroska Segment not found in prefix")
    segment_start, _segment_header_size = segment

    direct = _find_direct_cues(data, segment_start, file_size)
    if direct is not None:
        return direct

    seekhead = _find_seekhead(data, segment_start)
    if seekhead is None:
        return MkvCuesInfo("missing", -1, "SeekHead not found in prefix")
    cues_relative = _seekhead_cues_position(data, seekhead[0], seekhead[1])
    if cues_relative is None:
        return MkvCuesInfo("missing", -1, "SeekHead has no Cues entry")
    cues_position = segment_start + cues_relative
    return _classify_cues_position(cues_position, file_size, "SeekHead points to Cues in head area")
