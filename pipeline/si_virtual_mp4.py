"""Building blocks for SI virtual MP4 remuxing.

The progressive path exposes a stable logical MP4 without materialising the
large video payload: ``ftyp + moov + mdat-header`` live in memory, video samples
are sliced from the source MP4, and the small AAC SI audio sidecar is cached on
disk.  Sample table boxes are copied from the source/audio MP4s where possible;
only chunk mapping is rewritten to one-sample-per-chunk ``co64`` tables.
"""
from __future__ import annotations

import hashlib
import itertools
import io
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from config import (
    RUNTIME_CACHE_DIR,
    SI_AUDIO_EDIT_MODE,
    SI_AUDIO_EXTRACT_MODE,
    SI_MIX_ENCODER,
    SI_MIX_PARALLEL_MAX,
    SI_MIX_SEGMENTED_AAC,
    SI_MIX_SEGMENT_WARMUP_MS,
    SI_PREWARM_QUEUE_MAX,
)
from pipeline.ffmpeg_io import FFMPEG
from utils.cache_key import stat_key
from utils.logger import get
from utils.si_filter import SIMixParams, build_si_mix_filter
from utils.subprocess_hidden import hidden_subprocess_kwargs


log = get("si_virtual_mp4")

RegionKind = Literal["memory", "file"]
StreamKind = Literal["video", "audio"]
_PROGRESSIVE_CACHE_SCHEMA = 1


def _u32(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=False)


def _u64(value: int) -> bytes:
    return int(value).to_bytes(8, "big", signed=False)


def _box(box_type: bytes | str, payload: bytes) -> bytes:
    typ = box_type.encode("latin1") if isinstance(box_type, str) else bytes(box_type)
    size = 8 + len(payload)
    if size <= 0xFFFFFFFF:
        return _u32(size) + typ + payload
    return _u32(1) + typ + _u64(size) + payload


def _full_box(box_type: bytes | str, version: int, flags: int, payload: bytes) -> bytes:
    return _box(box_type, bytes([int(version) & 0xFF]) + (int(flags) & 0xFFFFFF).to_bytes(3, "big") + payload)


@dataclass(frozen=True)
class VideoSample:
    index: int
    source_offset: int
    size: int
    pts: int | None
    dts: int | None
    keyframe: bool


@dataclass(frozen=True)
class VideoSampleTable:
    path: Path
    codec_name: str
    width: int
    height: int
    time_base: Fraction
    duration_seconds: float
    extradata: bytes
    samples: tuple[VideoSample, ...]


@dataclass(frozen=True)
class MediaSample:
    index: int
    source_offset: int
    size: int
    pts: int | None
    dts: int | None
    keyframe: bool
    time_seconds: float


@dataclass(frozen=True)
class MediaSampleTable:
    path: Path
    stream_kind: StreamKind
    codec_name: str
    time_base: Fraction
    duration_seconds: float
    extradata: bytes
    samples: tuple[MediaSample, ...]


@dataclass(frozen=True)
class TopLevelBox:
    type: str
    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass(frozen=True)
class VirtualRegion:
    """A contiguous byte range in the logical virtual MP4 file."""

    start: int
    size: int
    kind: RegionKind
    data: bytes = b""
    path: Path | None = None
    source_offset: int = 0

    @classmethod
    def memory(cls, start: int, data: bytes) -> "VirtualRegion":
        return cls(start=max(0, int(start)), size=len(data), kind="memory", data=bytes(data))

    @classmethod
    def file(cls, start: int, path: Path, source_offset: int, size: int) -> "VirtualRegion":
        return cls(
            start=max(0, int(start)),
            size=max(0, int(size)),
            kind="file",
            path=Path(path),
            source_offset=max(0, int(source_offset)),
        )

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass(frozen=True)
class ProgressiveSIVirtualMp4:
    video_path: Path
    audio_path: Path
    regions: tuple[VirtualRegion, ...]
    content_length: int
    etag: str
    video_samples: int
    audio_samples: int
    moov_size: int
    mdat_payload_size: int
    audio_edit_mode: str = "preserve"


@dataclass(frozen=True)
class _BoxRef:
    type: bytes
    start: int
    size: int
    header_size: int

    @property
    def payload_start(self) -> int:
        return self.start + self.header_size

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass(frozen=True)
class _TrackSource:
    path: Path
    moov: bytes
    trak: _BoxRef
    kind: StreamKind


@dataclass(frozen=True)
class _SampleLayout:
    kind: StreamKind
    sample_index: int
    path: Path
    source_offset: int
    size: int
    time_seconds: float


class BoxSplittingSink(io.RawIOBase):
    """Append-only sink that records complete top-level ISO BMFF boxes."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.boxes: list[TopLevelBox] = []
        self._scan_cursor = 0

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return len(self.buffer)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        # libav may call seek/tell on Python file-like outputs. The fMP4 paths
        # we use are append-only, so report EOF and keep writes append-only.
        return len(self.buffer)

    def write(self, data: bytes | bytearray | memoryview) -> int:
        chunk = bytes(data)
        self.buffer.extend(chunk)
        self._scan_complete_boxes()
        return len(chunk)

    def box_bytes(self, box: TopLevelBox) -> bytes:
        return bytes(self.buffer[box.start:box.end])

    def _scan_complete_boxes(self) -> None:
        while self._scan_cursor + 8 <= len(self.buffer):
            offset = self._scan_cursor
            size = int.from_bytes(self.buffer[offset:offset + 4], "big")
            box_type = bytes(self.buffer[offset + 4:offset + 8]).decode("latin1", "replace")
            header_size = 8
            if size == 1:
                if offset + 16 > len(self.buffer):
                    return
                size = int.from_bytes(self.buffer[offset + 8:offset + 16], "big")
                header_size = 16
            if size < header_size:
                return
            if offset + size > len(self.buffer):
                return
            self.boxes.append(TopLevelBox(box_type, offset, size))
            self._scan_cursor = offset + size


def virtual_size(regions: Sequence[VirtualRegion]) -> int:
    """Return logical file size after validating region contiguity."""
    cursor = 0
    for region in sorted(regions, key=lambda r: r.start):
        if region.size < 0:
            raise ValueError("virtual region size must be non-negative")
        if region.start != cursor:
            raise ValueError(f"virtual regions are not contiguous at byte {cursor}")
        cursor = region.end
    return cursor


def _iter_boxes(data: bytes, start: int, end: int) -> Iterator[_BoxRef]:
    pos = int(start)
    limit = min(len(data), int(end))
    while pos + 8 <= limit:
        raw_size = int.from_bytes(data[pos:pos + 4], "big")
        box_type = bytes(data[pos + 4:pos + 8])
        header_size = 8
        if raw_size == 1:
            if pos + 16 > limit:
                break
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header_size = 16
        elif raw_size == 0:
            size = limit - pos
        else:
            size = raw_size
        if size < header_size or pos + size > limit:
            break
        yield _BoxRef(box_type, pos, size, header_size)
        pos += size


def _box_bytes(data: bytes, ref: _BoxRef) -> bytes:
    return bytes(data[ref.start:ref.end])


def _children(data: bytes, ref: _BoxRef) -> list[_BoxRef]:
    return list(_iter_boxes(data, ref.payload_start, ref.end))


def _find_child(data: bytes, ref: _BoxRef, box_type: bytes) -> _BoxRef | None:
    for child in _children(data, ref):
        if child.type == box_type:
            return child
    return None


def _require_child(data: bytes, ref: _BoxRef, box_type: bytes) -> _BoxRef:
    child = _find_child(data, ref, box_type)
    if child is None:
        raise ValueError(f"missing MP4 box {box_type!r}")
    return child


def _read_top_level_box(path: Path, box_type: bytes) -> bytes:
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as fh:
        pos = 0
        while pos + 8 <= file_size:
            fh.seek(pos)
            header = fh.read(16)
            if len(header) < 8:
                break
            raw_size = int.from_bytes(header[:4], "big")
            typ = header[4:8]
            header_size = 8
            if raw_size == 1:
                if len(header) < 16:
                    break
                size = int.from_bytes(header[8:16], "big")
                header_size = 16
            elif raw_size == 0:
                size = file_size - pos
            else:
                size = raw_size
            if size < header_size:
                break
            if typ == box_type:
                fh.seek(pos)
                return fh.read(size)
            pos += size
    raise ValueError(f"{source} has no top-level {box_type!r} box")


def _handler_type(moov: bytes, trak: _BoxRef) -> bytes:
    mdia = _require_child(moov, trak, b"mdia")
    hdlr = _require_child(moov, mdia, b"hdlr")
    offset = hdlr.payload_start + 8
    if offset + 4 > hdlr.end:
        raise ValueError("invalid hdlr box")
    return bytes(moov[offset:offset + 4])


def _find_track(moov: bytes, kind: StreamKind) -> _TrackSource:
    moov_ref = next(_iter_boxes(moov, 0, len(moov)), None)
    if moov_ref is None or moov_ref.type != b"moov":
        raise ValueError("expected a moov box")
    wanted = b"vide" if kind == "video" else b"soun"
    for child in _children(moov, moov_ref):
        if child.type == b"trak" and _handler_type(moov, child) == wanted:
            return _TrackSource(Path(), moov, child, kind)
    raise ValueError(f"missing {kind} track in moov")


def _track_stbl(moov: bytes, trak: _BoxRef) -> _BoxRef:
    mdia = _require_child(moov, trak, b"mdia")
    minf = _require_child(moov, mdia, b"minf")
    return _require_child(moov, minf, b"stbl")


def _parse_mdhd_time_base(moov: bytes, trak: _BoxRef) -> tuple[Fraction, float]:
    mdia = _require_child(moov, trak, b"mdia")
    mdhd = _require_child(moov, mdia, b"mdhd")
    version = moov[mdhd.payload_start]
    if version == 1:
        timescale_offset = mdhd.payload_start + 20
        duration_offset = mdhd.payload_start + 24
        duration_size = 8
    else:
        timescale_offset = mdhd.payload_start + 12
        duration_offset = mdhd.payload_start + 16
        duration_size = 4
    if duration_offset + duration_size > mdhd.end:
        raise ValueError("invalid mdhd box")
    timescale = int.from_bytes(moov[timescale_offset:timescale_offset + 4], "big")
    if timescale <= 0:
        raise ValueError("invalid mdhd timescale")
    duration = int.from_bytes(moov[duration_offset:duration_offset + duration_size], "big")
    return Fraction(1, timescale), duration / float(timescale)


def _parse_stsd_codec_name(moov: bytes, stbl: _BoxRef) -> str:
    stsd = _find_child(moov, stbl, b"stsd")
    if stsd is None:
        return ""
    entry_count_offset = stsd.payload_start + 4
    first_entry = entry_count_offset + 4
    if first_entry + 8 > stsd.end:
        return ""
    sample_entry_type = bytes(moov[first_entry + 4:first_entry + 8])
    return {
        b"avc1": "h264",
        b"avc3": "h264",
        b"hvc1": "hevc",
        b"hev1": "hevc",
        b"mp4v": "mpeg4",
        b"mp4a": "aac",
        b"ac-3": "ac3",
        b"ec-3": "eac3",
        b"alac": "alac",
    }.get(sample_entry_type, sample_entry_type.decode("latin1", "replace"))


def _parse_stsz(moov: bytes, stbl: _BoxRef) -> list[int]:
    stsz = _require_child(moov, stbl, b"stsz")
    offset = stsz.payload_start + 4
    if offset + 8 > stsz.end:
        raise ValueError("invalid stsz box")
    sample_size = int.from_bytes(moov[offset:offset + 4], "big")
    sample_count = int.from_bytes(moov[offset + 4:offset + 8], "big")
    if sample_size:
        return [sample_size] * sample_count
    cursor = offset + 8
    if cursor + sample_count * 4 > stsz.end:
        raise ValueError("invalid stsz sample table")
    return [int.from_bytes(moov[cursor + i * 4:cursor + i * 4 + 4], "big") for i in range(sample_count)]


def _parse_chunk_offsets(moov: bytes, stbl: _BoxRef) -> list[int]:
    stco = _find_child(moov, stbl, b"stco")
    co64 = _find_child(moov, stbl, b"co64")
    if co64 is not None:
        offset = co64.payload_start + 4
        if offset + 4 > co64.end:
            raise ValueError("invalid co64 box")
        count = int.from_bytes(moov[offset:offset + 4], "big")
        cursor = offset + 4
        if cursor + count * 8 > co64.end:
            raise ValueError("invalid co64 table")
        return [int.from_bytes(moov[cursor + i * 8:cursor + i * 8 + 8], "big") for i in range(count)]
    if stco is None:
        raise ValueError("missing stco/co64 box")
    offset = stco.payload_start + 4
    if offset + 4 > stco.end:
        raise ValueError("invalid stco box")
    count = int.from_bytes(moov[offset:offset + 4], "big")
    cursor = offset + 4
    if cursor + count * 4 > stco.end:
        raise ValueError("invalid stco table")
    return [int.from_bytes(moov[cursor + i * 4:cursor + i * 4 + 4], "big") for i in range(count)]


def _parse_stsc(moov: bytes, stbl: _BoxRef) -> list[tuple[int, int]]:
    stsc = _require_child(moov, stbl, b"stsc")
    offset = stsc.payload_start + 4
    if offset + 4 > stsc.end:
        raise ValueError("invalid stsc box")
    entry_count = int.from_bytes(moov[offset:offset + 4], "big")
    cursor = offset + 4
    if cursor + entry_count * 12 > stsc.end:
        raise ValueError("invalid stsc table")
    entries: list[tuple[int, int]] = []
    for i in range(entry_count):
        base = cursor + i * 12
        first_chunk = int.from_bytes(moov[base:base + 4], "big")
        samples_per_chunk = int.from_bytes(moov[base + 4:base + 8], "big")
        if first_chunk <= 0 or samples_per_chunk <= 0:
            raise ValueError("invalid stsc entry")
        entries.append((first_chunk, samples_per_chunk))
    if not entries:
        raise ValueError("empty stsc table")
    entries.sort(key=lambda item: item[0])
    return entries


def _expand_sample_offsets(chunk_offsets: Sequence[int], stsc: Sequence[tuple[int, int]], sizes: Sequence[int]) -> list[int]:
    sample_offsets: list[int] = []
    sample_index = 0
    stsc_index = 0
    for chunk_index, chunk_offset in enumerate(chunk_offsets, start=1):
        while stsc_index + 1 < len(stsc) and chunk_index >= stsc[stsc_index + 1][0]:
            stsc_index += 1
        samples_per_chunk = stsc[stsc_index][1]
        cursor = int(chunk_offset)
        for _ in range(samples_per_chunk):
            if sample_index >= len(sizes):
                break
            sample_offsets.append(cursor)
            cursor += int(sizes[sample_index])
            sample_index += 1
    if len(sample_offsets) != len(sizes):
        raise ValueError(f"sample offset count mismatch: offsets={len(sample_offsets)} sizes={len(sizes)}")
    return sample_offsets


def _parse_stts(moov: bytes, stbl: _BoxRef, sample_count: int) -> list[int]:
    stts = _require_child(moov, stbl, b"stts")
    offset = stts.payload_start + 4
    if offset + 4 > stts.end:
        raise ValueError("invalid stts box")
    entry_count = int.from_bytes(moov[offset:offset + 4], "big")
    cursor = offset + 4
    if cursor + entry_count * 8 > stts.end:
        raise ValueError("invalid stts table")
    dts_values: list[int] = []
    dts = 0
    for i in range(entry_count):
        base = cursor + i * 8
        count = int.from_bytes(moov[base:base + 4], "big")
        delta = int.from_bytes(moov[base + 4:base + 8], "big")
        for _ in range(count):
            dts_values.append(dts)
            dts += delta
            if len(dts_values) >= sample_count:
                break
        if len(dts_values) >= sample_count:
            break
    if len(dts_values) != sample_count:
        raise ValueError(f"stts sample count mismatch: dts={len(dts_values)} samples={sample_count}")
    return dts_values


def _parse_ctts(moov: bytes, stbl: _BoxRef, sample_count: int) -> list[int]:
    ctts = _find_child(moov, stbl, b"ctts")
    if ctts is None:
        return [0] * sample_count
    version = moov[ctts.payload_start]
    offset = ctts.payload_start + 4
    if offset + 4 > ctts.end:
        raise ValueError("invalid ctts box")
    entry_count = int.from_bytes(moov[offset:offset + 4], "big")
    cursor = offset + 4
    if cursor + entry_count * 8 > ctts.end:
        raise ValueError("invalid ctts table")
    offsets: list[int] = []
    for i in range(entry_count):
        base = cursor + i * 8
        count = int.from_bytes(moov[base:base + 4], "big")
        raw = moov[base + 4:base + 8]
        sample_offset = int.from_bytes(raw, "big", signed=version == 1)
        offsets.extend([sample_offset] * count)
        if len(offsets) >= sample_count:
            break
    if len(offsets) != sample_count:
        raise ValueError(f"ctts sample count mismatch: offsets={len(offsets)} samples={sample_count}")
    return offsets


def _parse_stss(moov: bytes, stbl: _BoxRef, sample_count: int) -> set[int] | None:
    stss = _find_child(moov, stbl, b"stss")
    if stss is None:
        return None
    offset = stss.payload_start + 4
    if offset + 4 > stss.end:
        raise ValueError("invalid stss box")
    entry_count = int.from_bytes(moov[offset:offset + 4], "big")
    cursor = offset + 4
    if cursor + entry_count * 4 > stss.end:
        raise ValueError("invalid stss table")
    keyframes: set[int] = set()
    for i in range(entry_count):
        sample_number = int.from_bytes(moov[cursor + i * 4:cursor + i * 4 + 4], "big")
        if 1 <= sample_number <= sample_count:
            keyframes.add(sample_number - 1)
    return keyframes


def _audio_edit_media_time_from_moov(moov: bytes) -> int:
    try:
        trak = _find_track(moov, "audio").trak
        edts = _find_child(moov, trak, b"edts")
        if edts is None:
            return 0
        elst = _find_child(moov, edts, b"elst")
        if elst is None:
            return 0
        version = moov[elst.payload_start]
        offset = elst.payload_start + 4
        if offset + 4 > elst.end:
            return 0
        entry_count = int.from_bytes(moov[offset:offset + 4], "big")
        if entry_count <= 0:
            return 0
        cursor = offset + 4
        if version == 1:
            if cursor + 20 > elst.end:
                return 0
            media_time = int.from_bytes(moov[cursor + 8:cursor + 16], "big", signed=True)
        else:
            if cursor + 12 > elst.end:
                return 0
            media_time = int.from_bytes(moov[cursor + 4:cursor + 8], "big", signed=True)
        return max(0, int(media_time))
    except Exception:
        return 0


def _audio_priming_frames_from_moov(moov: bytes, *, sample_delta: int = 1024) -> int:
    media_time = _audio_edit_media_time_from_moov(moov)
    if media_time <= 0 or sample_delta <= 0:
        return 0
    return int(math.ceil(media_time / float(sample_delta)))


def _parse_stts_durations(moov: bytes, stbl: _BoxRef, sample_count: int) -> list[int]:
    stts = _require_child(moov, stbl, b"stts")
    offset = stts.payload_start + 4
    if offset + 4 > stts.end:
        raise ValueError("invalid stts box")
    entry_count = int.from_bytes(moov[offset:offset + 4], "big")
    cursor = offset + 4
    if cursor + entry_count * 8 > stts.end:
        raise ValueError("invalid stts table")
    durations: list[int] = []
    for i in range(entry_count):
        base = cursor + i * 8
        count = int.from_bytes(moov[base:base + 4], "big")
        delta = int.from_bytes(moov[base + 4:base + 8], "big")
        durations.extend([delta] * count)
        if len(durations) >= sample_count:
            break
    if len(durations) != sample_count:
        raise ValueError(f"stts sample count mismatch: durations={len(durations)} samples={sample_count}")
    return durations


def _read_sample_table_from_moov(path: Path, moov: bytes, stream_kind: StreamKind) -> MediaSampleTable:
    source = Path(path)
    track = _find_track(moov, stream_kind).trak
    stbl = _track_stbl(moov, track)
    time_base, duration_seconds = _parse_mdhd_time_base(moov, track)
    timescale = time_base.denominator
    sizes = _parse_stsz(moov, stbl)
    chunk_offsets = _parse_chunk_offsets(moov, stbl)
    sample_offsets = _expand_sample_offsets(chunk_offsets, _parse_stsc(moov, stbl), sizes)
    dts_values = _parse_stts(moov, stbl, len(sizes))
    ctts_offsets = _parse_ctts(moov, stbl, len(sizes))
    keyframes = _parse_stss(moov, stbl, len(sizes))
    samples = [
        MediaSample(
            index=i,
            source_offset=int(sample_offsets[i]),
            size=int(sizes[i]),
            pts=int(dts_values[i] + ctts_offsets[i]),
            dts=int(dts_values[i]),
            keyframe=True if keyframes is None else i in keyframes,
            time_seconds=float(dts_values[i]) / float(timescale),
        )
        for i in range(len(sizes))
    ]
    return MediaSampleTable(
        path=source,
        stream_kind=stream_kind,
        codec_name=_parse_stsd_codec_name(moov, stbl),
        time_base=time_base,
        duration_seconds=duration_seconds,
        extradata=b"",
        samples=tuple(samples),
    )


def _read_video_sample_table_from_moov(path: Path, moov: bytes) -> MediaSampleTable:
    return _read_sample_table_from_moov(path, moov, "video")


def _track_id(trak_data: bytes) -> int:
    trak = next(_iter_boxes(trak_data, 0, len(trak_data)), None)
    if trak is None or trak.type != b"trak":
        raise ValueError("expected trak box")
    tkhd = _require_child(trak_data, trak, b"tkhd")
    version = trak_data[tkhd.payload_start]
    track_id_offset = tkhd.start + (28 if version == 1 else 20)
    if track_id_offset + 4 > tkhd.end:
        raise ValueError("invalid tkhd track id")
    return int.from_bytes(trak_data[track_id_offset:track_id_offset + 4], "big")


def _patch_tkhd_track_id(tkhd: bytes, track_id: int) -> bytes:
    if len(tkhd) < 24 or tkhd[4:8] != b"tkhd":
        raise ValueError("expected tkhd box")
    out = bytearray(tkhd)
    version = out[8]
    offset = 28 if version == 1 else 20
    out[offset:offset + 4] = _u32(track_id)
    return bytes(out)


def _patch_tkhd_duration(tkhd: bytes, duration: int) -> bytes:
    if len(tkhd) < 32 or tkhd[4:8] != b"tkhd":
        raise ValueError("expected tkhd box")
    out = bytearray(tkhd)
    version = out[8]
    if version == 1:
        offset = 36
        out[offset:offset + 8] = _u64(max(0, int(duration)))
    else:
        offset = 28
        out[offset:offset + 4] = _u32(min(max(0, int(duration)), 0xFFFFFFFF))
    return bytes(out)


def _patch_mdhd_duration(mdhd: bytes, duration: int) -> bytes:
    if len(mdhd) < 28 or mdhd[4:8] != b"mdhd":
        raise ValueError("expected mdhd box")
    out = bytearray(mdhd)
    version = out[8]
    if version == 1:
        offset = 32
        out[offset:offset + 8] = _u64(max(0, int(duration)))
    else:
        offset = 24
        out[offset:offset + 4] = _u32(min(max(0, int(duration)), 0xFFFFFFFF))
    return bytes(out)


def _mvhd_timescale(mvhd: bytes) -> int:
    if len(mvhd) < 28 or mvhd[4:8] != b"mvhd":
        raise ValueError("expected mvhd box")
    version = mvhd[8]
    offset = 28 if version == 1 else 20
    if offset + 4 > len(mvhd):
        raise ValueError("invalid mvhd timescale")
    timescale = int.from_bytes(mvhd[offset:offset + 4], "big")
    if timescale <= 0:
        raise ValueError("invalid mvhd timescale")
    return timescale


def _patch_mvhd_duration_and_next_track_id(mvhd: bytes, duration: int, next_track_id: int) -> bytes:
    if len(mvhd) < 32 or mvhd[4:8] != b"mvhd":
        raise ValueError("expected mvhd box")
    out = bytearray(mvhd)
    version = out[8]
    if version == 1:
        offset = 32
        out[offset:offset + 8] = _u64(max(0, int(duration)))
    else:
        offset = 24
        out[offset:offset + 4] = _u32(min(max(0, int(duration)), 0xFFFFFFFF))
    out[-4:] = _u32(max(1, int(next_track_id)))
    return bytes(out)


def _patch_mvhd_next_track_id(mvhd: bytes, next_track_id: int) -> bytes:
    if len(mvhd) < 12 or mvhd[4:8] != b"mvhd":
        raise ValueError("expected mvhd box")
    out = bytearray(mvhd)
    out[-4:] = _u32(max(1, int(next_track_id)))
    return bytes(out)


def _make_stsc_one_sample_per_chunk() -> bytes:
    payload = _u32(1) + _u32(1) + _u32(1) + _u32(1)
    return _full_box("stsc", 0, 0, payload)


def _make_co64(offsets: Sequence[int]) -> bytes:
    payload = bytearray()
    payload += _u32(len(offsets))
    for offset in offsets:
        payload += _u64(offset)
    return _full_box("co64", 0, 0, bytes(payload))


def _make_stsz(sizes: Sequence[int]) -> bytes:
    payload = bytearray()
    payload += _u32(0)
    payload += _u32(len(sizes))
    for size in sizes:
        payload += _u32(max(0, int(size)))
    return _full_box("stsz", 0, 0, bytes(payload))


def _make_stts(sample_count: int, sample_delta: int = 1024) -> bytes:
    payload = _u32(1) + _u32(max(0, int(sample_count))) + _u32(max(0, int(sample_delta)))
    return _full_box("stts", 0, 0, payload)


def _make_stts_from_durations(durations: Sequence[int]) -> bytes:
    entries: list[tuple[int, int]] = []
    for duration in durations:
        delta = max(0, int(duration))
        if entries and entries[-1][1] == delta:
            count, _ = entries[-1]
            entries[-1] = (count + 1, delta)
        else:
            entries.append((1, delta))
    payload = bytearray()
    payload += _u32(len(entries))
    for count, delta in entries:
        payload += _u32(count)
        payload += _u32(delta)
    return _full_box("stts", 0, 0, bytes(payload))


def _make_edts_elst(segment_duration: int, media_time: int) -> bytes:
    payload = _u32(1)
    payload += _u32(min(max(0, int(segment_duration)), 0xFFFFFFFF))
    payload += int(media_time).to_bytes(4, "big", signed=True)
    payload += (1).to_bytes(2, "big", signed=True)
    payload += (0).to_bytes(2, "big", signed=False)
    return _box("edts", _full_box("elst", 0, 0, payload))


def _rewrite_stbl(moov: bytes, stbl: _BoxRef, offsets: Sequence[int]) -> bytes:
    children: list[bytes] = []
    inserted_stsc = False
    inserted_offsets = False
    for child in _children(moov, stbl):
        if child.type == b"stsc":
            if not inserted_stsc:
                children.append(_make_stsc_one_sample_per_chunk())
                inserted_stsc = True
            continue
        if child.type in {b"stco", b"co64"}:
            if not inserted_offsets:
                children.append(_make_co64(offsets))
                inserted_offsets = True
            continue
        children.append(_box_bytes(moov, child))
    if not inserted_stsc:
        children.append(_make_stsc_one_sample_per_chunk())
    if not inserted_offsets:
        children.append(_make_co64(offsets))
    return _box("stbl", b"".join(children))


def _rewrite_audio_stbl_samples(
    moov: bytes,
    stbl: _BoxRef,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    *,
    sample_delta: int = 1024,
) -> bytes:
    stsd = _require_child(moov, stbl, b"stsd")
    stts = _make_stts_from_durations(durations) if durations else _make_stts(len(sizes), sample_delta)
    return _box(
        "stbl",
        _box_bytes(moov, stsd)
        + stts
        + _make_stsc_one_sample_per_chunk()
        + _make_stsz(sizes)
        + _make_co64(offsets),
    )


def _rewrite_minf(moov: bytes, minf: _BoxRef, offsets: Sequence[int]) -> bytes:
    stbl = _require_child(moov, minf, b"stbl")
    children = [
        _rewrite_stbl(moov, child, offsets) if child.start == stbl.start else _box_bytes(moov, child)
        for child in _children(moov, minf)
    ]
    return _box("minf", b"".join(children))


def _rewrite_audio_minf_samples(
    moov: bytes,
    minf: _BoxRef,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    *,
    sample_delta: int = 1024,
) -> bytes:
    stbl = _require_child(moov, minf, b"stbl")
    children = [
        _rewrite_audio_stbl_samples(moov, child, offsets, sizes, durations, sample_delta=sample_delta)
        if child.start == stbl.start
        else _box_bytes(moov, child)
        for child in _children(moov, minf)
    ]
    return _box("minf", b"".join(children))


def _rewrite_mdia(moov: bytes, mdia: _BoxRef, offsets: Sequence[int]) -> bytes:
    minf = _require_child(moov, mdia, b"minf")
    children = [
        _rewrite_minf(moov, child, offsets) if child.start == minf.start else _box_bytes(moov, child)
        for child in _children(moov, mdia)
    ]
    return _box("mdia", b"".join(children))


def _rewrite_audio_mdia_samples(
    moov: bytes,
    mdia: _BoxRef,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    *,
    media_duration: int,
    sample_delta: int = 1024,
) -> bytes:
    minf = _require_child(moov, mdia, b"minf")
    children: list[bytes] = []
    for child in _children(moov, mdia):
        if child.start == minf.start:
            children.append(_rewrite_audio_minf_samples(moov, child, offsets, sizes, durations, sample_delta=sample_delta))
        elif child.type == b"mdhd":
            children.append(_patch_mdhd_duration(_box_bytes(moov, child), media_duration))
        else:
            children.append(_box_bytes(moov, child))
    return _box("mdia", b"".join(children))


def _rewrite_trak(
    moov: bytes,
    trak: _BoxRef,
    offsets: Sequence[int],
    *,
    track_id: int | None = None,
    drop_edts: bool = False,
) -> bytes:
    mdia = _require_child(moov, trak, b"mdia")
    children: list[bytes] = []
    for child in _children(moov, trak):
        if drop_edts and child.type == b"edts":
            continue
        if child.start == mdia.start:
            children.append(_rewrite_mdia(moov, child, offsets))
        elif child.type == b"tkhd" and track_id is not None:
            children.append(_patch_tkhd_track_id(_box_bytes(moov, child), track_id))
        else:
            children.append(_box_bytes(moov, child))
    return _box("trak", b"".join(children))


def _rewrite_audio_trak_samples(
    moov: bytes,
    trak: _BoxRef,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    *,
    media_duration: int,
    movie_duration: int,
    sample_delta: int = 1024,
    drop_edts: bool = True,
    edit_segment_duration: int | None = None,
    edit_media_time: int | None = None,
) -> bytes:
    mdia = _require_child(moov, trak, b"mdia")
    children: list[bytes] = []
    inserted_edts = False
    for child in _children(moov, trak):
        if child.type == b"edts":
            if edit_media_time is not None and edit_media_time > 0 and not inserted_edts:
                children.append(_make_edts_elst(edit_segment_duration or 0, edit_media_time))
                inserted_edts = True
            elif not drop_edts:
                children.append(_box_bytes(moov, child))
            continue
        if child.start == mdia.start:
            if edit_media_time is not None and edit_media_time > 0 and not inserted_edts:
                children.append(_make_edts_elst(edit_segment_duration or 0, edit_media_time))
                inserted_edts = True
            children.append(
                _rewrite_audio_mdia_samples(
                    moov,
                    child,
                    offsets,
                    sizes,
                    durations,
                    media_duration=media_duration,
                    sample_delta=sample_delta,
                )
            )
        elif child.type == b"tkhd":
            children.append(_patch_tkhd_duration(_box_bytes(moov, child), movie_duration))
        else:
            children.append(_box_bytes(moov, child))
    return _box("trak", b"".join(children))


def _build_moov(
    source_moov: bytes,
    audio_moov: bytes,
    video_offsets: Sequence[int],
    audio_offsets: Sequence[int],
    *,
    audio_edit_mode: str = "preserve",
) -> bytes:
    source_moov_ref = next(_iter_boxes(source_moov, 0, len(source_moov)), None)
    if source_moov_ref is None or source_moov_ref.type != b"moov":
        raise ValueError("expected source moov")
    video_trak = _find_track(source_moov, "video").trak
    audio_trak = _find_track(audio_moov, "audio").trak
    video_trak_data = _rewrite_trak(source_moov, video_trak, video_offsets)
    video_id = _track_id(video_trak_data)
    audio_id = max(video_id + 1, 2)
    audio_trak_data = _rewrite_trak(
        audio_moov,
        audio_trak,
        audio_offsets,
        track_id=audio_id,
        drop_edts=audio_edit_mode == "remove",
    )
    next_track_id = max(video_id, audio_id) + 1

    children: list[bytes] = []
    inserted_video = False
    inserted_audio = False
    for child in _children(source_moov, source_moov_ref):
        if child.type == b"mvhd":
            children.append(_patch_mvhd_next_track_id(_box_bytes(source_moov, child), next_track_id))
        elif child.type == b"trak":
            if child.start == video_trak.start:
                children.append(video_trak_data)
                inserted_video = True
                if not inserted_audio:
                    children.append(audio_trak_data)
                    inserted_audio = True
            else:
                # Drop the source audio/subtitle tracks; SI output carries the
                # copied source video plus the newly mixed SI AAC track only.
                continue
        else:
            children.append(_box_bytes(source_moov, child))
    if not inserted_video:
        children.append(video_trak_data)
    if not inserted_audio:
        children.append(audio_trak_data)
    return _box("moov", b"".join(children))


def _build_stitched_audio_moov(
    template_moov: bytes,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    *,
    sample_delta: int = 1024,
    edit_media_time: int = 0,
) -> bytes:
    source_moov_ref = next(_iter_boxes(template_moov, 0, len(template_moov)), None)
    if source_moov_ref is None or source_moov_ref.type != b"moov":
        raise ValueError("expected template moov")
    mvhd = _require_child(template_moov, source_moov_ref, b"mvhd")
    audio_trak = _find_track(template_moov, "audio").trak
    media_timescale = _parse_mdhd_time_base(template_moov, audio_trak)[0].denominator
    media_duration = sum(max(0, int(duration)) for duration in durations) if durations else len(sizes) * int(sample_delta)
    edit_media_time = max(0, int(edit_media_time))
    presentation_media_duration = max(0, media_duration - edit_media_time)
    movie_timescale = _mvhd_timescale(_box_bytes(template_moov, mvhd))
    movie_duration = int(math.ceil(presentation_media_duration * movie_timescale / media_timescale))
    edit_segment_duration = int(presentation_media_duration * movie_timescale / media_timescale)
    audio_trak_data = _rewrite_audio_trak_samples(
        template_moov,
        audio_trak,
        offsets,
        sizes,
        durations,
        media_duration=media_duration,
        movie_duration=movie_duration,
        sample_delta=sample_delta,
        drop_edts=True,
        edit_segment_duration=edit_segment_duration if edit_media_time > 0 else None,
        edit_media_time=edit_media_time if edit_media_time > 0 else None,
    )
    audio_id = _track_id(audio_trak_data)
    return _box(
        "moov",
        _patch_mvhd_duration_and_next_track_id(_box_bytes(template_moov, mvhd), movie_duration, audio_id + 1)
        + audio_trak_data,
    )


def _build_audio_only_moov(source_moov: bytes, audio_offsets: Sequence[int]) -> bytes:
    source_moov_ref = next(_iter_boxes(source_moov, 0, len(source_moov)), None)
    if source_moov_ref is None or source_moov_ref.type != b"moov":
        raise ValueError("expected source moov")
    mvhd = _require_child(source_moov, source_moov_ref, b"mvhd")
    audio_trak = _find_track(source_moov, "audio").trak
    audio_trak_data = _rewrite_trak(source_moov, audio_trak, audio_offsets)
    audio_id = _track_id(audio_trak_data)
    return _box(
        "moov",
        _patch_mvhd_next_track_id(_box_bytes(source_moov, mvhd), audio_id + 1) + audio_trak_data,
    )


def _mdat_header(payload_size: int) -> bytes:
    total = int(payload_size) + 16
    return _u32(1) + b"mdat" + _u64(total)


def _sample_time(sample: MediaSample) -> float:
    return sample.time_seconds


def _layout_samples(
    video_path: Path,
    audio_path: Path,
    video_samples: Sequence[MediaSample],
    audio_samples: Sequence[MediaSample],
    mdat_payload_start: int,
) -> tuple[list[_SampleLayout], list[int], list[int], int]:
    items: list[_SampleLayout] = []
    for sample in video_samples:
        items.append(
            _SampleLayout(
                kind="video",
                sample_index=sample.index,
                path=video_path,
                source_offset=sample.source_offset,
                size=sample.size,
                time_seconds=_sample_time(sample),
            )
        )
    for sample in audio_samples:
        items.append(
            _SampleLayout(
                kind="audio",
                sample_index=sample.index,
                path=audio_path,
                source_offset=sample.source_offset,
                size=sample.size,
                time_seconds=_sample_time(sample),
            )
        )
    items.sort(key=lambda item: (item.time_seconds, 0 if item.kind == "video" else 1, item.sample_index))

    cursor = int(mdat_payload_start)
    video_offsets = [0] * len(video_samples)
    audio_offsets = [0] * len(audio_samples)
    ordered: list[_SampleLayout] = []
    for item in items:
        if item.size <= 0:
            continue
        if item.kind == "video":
            video_offsets[item.sample_index] = cursor
        else:
            audio_offsets[item.sample_index] = cursor
        ordered.append(item)
        cursor += item.size
    return ordered, video_offsets, audio_offsets, cursor - int(mdat_payload_start)


def read_media_sample_table(path: Path, stream_kind: StreamKind, *, limit: int | None = None) -> MediaSampleTable:
    """Read packet offsets for one MP4 stream with PyAV."""
    if limit is None:
        source = Path(path)
        try:
            return _read_sample_table_from_moov(source, _read_top_level_box(source, b"moov"), stream_kind)
        except Exception as exc:
            log.warning("SI %s sample table moov parse failed for %s; falling back to PyAV demux: %s", stream_kind, source, exc)

    import av  # type: ignore[import-not-found]

    source = Path(path)
    samples: list[MediaSample] = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0] if stream_kind == "video" else container.streams.audio[0]
        ctx = stream.codec_context
        time_base = Fraction(stream.time_base)
        duration_seconds = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        sample_index = 0
        for packet in container.demux(stream):
            if packet.size <= 0 or packet.pos is None:
                continue
            timestamp = packet.dts if packet.dts is not None else packet.pts
            time_seconds = float(timestamp * stream.time_base) if timestamp is not None else 0.0
            samples.append(
                MediaSample(
                    index=sample_index,
                    source_offset=int(packet.pos),
                    size=int(packet.size),
                    pts=None if packet.pts is None else int(packet.pts),
                    dts=None if packet.dts is None else int(packet.dts),
                    keyframe=bool(packet.is_keyframe),
                    time_seconds=time_seconds,
                )
            )
            sample_index += 1
            if limit is not None and len(samples) >= max(0, int(limit)):
                break
    return MediaSampleTable(
        path=source,
        stream_kind=stream_kind,
        codec_name=str(ctx.name),
        time_base=time_base,
        duration_seconds=duration_seconds,
        extradata=bytes(ctx.extradata or b""),
        samples=tuple(samples),
    )


def _params_cache_dict(params: SIMixParams) -> dict:
    return params.to_dict() if hasattr(params, "to_dict") else {
        "enabled": bool(params.enabled),
        "mix_channel": params.mix_channel,
        "original_volume_percent": params.original_volume_percent,
        "si_volume_percent": params.si_volume_percent,
        "si_delay_seconds": params.si_delay_seconds,
        "duck_original": params.duck_original,
    }


_encoder_probe_lock = threading.Lock()
_encoder_probe_cache: dict[str, bool] = {}


def _ffmpeg_encoder_available(encoder: str) -> bool:
    name = str(encoder).strip()
    if not name:
        return False
    with _encoder_probe_lock:
        cached = _encoder_probe_cache.get(name)
    if cached is not None:
        return cached
    available = False
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "error", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_subprocess_kwargs(low_priority=False),
            timeout=10,
        )
        if proc.returncode == 0:
            available = any(line.split()[1:2] == [name] for line in proc.stdout.splitlines() if line.strip())
    except Exception:
        available = False
    with _encoder_probe_lock:
        _encoder_probe_cache[name] = available
    return available


def _selected_mix_encoder() -> str:
    configured = str(SI_MIX_ENCODER).strip().lower()
    if configured == "aac_mf":
        if _ffmpeg_encoder_available("aac_mf"):
            return "aac_mf"
        log.warning("PT_SI_MIX_ENCODER=aac_mf requested but ffmpeg aac_mf encoder is unavailable; using native aac")
        return "aac"
    if configured == "aac":
        return "aac"
    if os.name == "nt" and _ffmpeg_encoder_available("aac_mf"):
        return "aac_mf"
    return "aac"


def _use_segmented_aac(encoder: str) -> bool:
    return bool(SI_MIX_SEGMENTED_AAC) and encoder == "aac" and int(SI_MIX_PARALLEL_MAX) > 1


def _mix_cache_variant() -> dict[str, object]:
    encoder = _selected_mix_encoder()
    segmented = _use_segmented_aac(encoder)
    return {
        "encoder": encoder,
        "implementation": "segmented" if segmented else "single",
        "warmup_ms": int(SI_MIX_SEGMENT_WARMUP_MS) if segmented else 0,
    }


def _cache_digest(video: Path, si_wav: Path, params: SIMixParams) -> str:
    payload = {
        "schema": _PROGRESSIVE_CACHE_SCHEMA,
        "video": stat_key(video),
        "si_wav": stat_key(si_wav),
        "params": _params_cache_dict(params),
        "mix": _mix_cache_variant(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_audio_cache_digest(video: Path) -> str:
    payload = {
        "schema": _PROGRESSIVE_CACHE_SCHEMA,
        "kind": "source-audio-v1",
        "video": stat_key(video),
        "track": 0,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _layout_cache_digest(sidecar_digest: str, audio_edit_mode: str) -> str:
    payload = {
        "schema": _PROGRESSIVE_CACHE_SCHEMA,
        "sidecar_digest": sidecar_digest,
        "audio_edit_mode": audio_edit_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ensure_cache_dir() -> Path:
    candidates = (
        RUNTIME_CACHE_DIR / "si_virtual_mp4",
        RUNTIME_CACHE_DIR / "tmp" / "si_virtual_mp4",
        RUNTIME_CACHE_DIR.parent / "debug_output" / "si_virtual_mp4_cache",
    )
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as exc:
            last_error = exc
            log.warning("SI virtual MP4 cache dir unavailable: %s (%s)", candidate, exc)
    assert last_error is not None
    raise last_error


class _BuildCancelled(RuntimeError):
    pass


@dataclass
class _BuildContext:
    digest: str
    video: Path
    priority: int
    reason: str
    cancelable: bool
    cancel_event: threading.Event


_BUILD_PRIORITY_PLAYBACK = 0
_BUILD_PRIORITY_PREWARM = 10
_build_slot = threading.Lock()
_current_build_lock = threading.Lock()
_current_build: _BuildContext | None = None
_source_audio_locks_lock = threading.Lock()
_source_audio_locks: dict[str, threading.Lock] = {}


def _enter_build_slot(digest: str, video: Path, *, priority: int, reason: str, cancelable: bool) -> _BuildContext:
    global _current_build
    with _current_build_lock:
        current = _current_build
        if current is not None and current.cancelable and priority < current.priority:
            current.cancel_event.set()
            log.info(
                "cancelling lower-priority SI progressive build: current=%s reason=%s requested=%s",
                current.video,
                current.reason,
                reason,
            )
    _build_slot.acquire()
    context = _BuildContext(
        digest=digest,
        video=Path(video),
        priority=int(priority),
        reason=str(reason),
        cancelable=bool(cancelable),
        cancel_event=threading.Event(),
    )
    with _current_build_lock:
        _current_build = context
    return context


def _leave_build_slot(context: _BuildContext) -> None:
    global _current_build
    with _current_build_lock:
        if _current_build is context:
            _current_build = None
    _build_slot.release()


def _source_audio_lock_for_digest(digest: str) -> threading.Lock:
    with _source_audio_locks_lock:
        lock = _source_audio_locks.get(digest)
        if lock is None:
            lock = threading.Lock()
            _source_audio_locks[digest] = lock
        return lock


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _BuildCancelled("SI progressive build cancelled")


def _subprocess_kwargs(*, low_priority: bool = False) -> dict:
    kwargs = hidden_subprocess_kwargs()
    if low_priority:
        priority_flag = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        if priority_flag:
            kwargs["creationflags"] = int(kwargs.get("creationflags", 0) or 0) | int(priority_flag)
    return kwargs


def _run_ffmpeg_sidecar(
    cmd: Sequence[str],
    *,
    cancel_event: threading.Event | None,
    low_priority: bool,
) -> None:
    _raise_if_cancelled(cancel_event)
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_subprocess_kwargs(low_priority=low_priority),
    )
    while True:
        if cancel_event is not None and cancel_event.is_set():
            proc.terminate()
            try:
                proc.communicate(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            raise _BuildCancelled("SI sidecar ffmpeg cancelled")
        try:
            stdout, stderr = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            continue
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd), output=stdout, stderr=stderr)


def _drain_pipe(pipe, chunks: list[bytes]) -> None:
    if pipe is None:
        return
    while True:
        data = pipe.read(4096)
        if not data:
            break
        chunks.append(bytes(data))


def _run_ffmpeg_sidecar_with_source_audio_pipe(
    cmd: Sequence[str],
    *,
    video: Path,
    source_audio_output: Path,
    cancel_event: threading.Event | None,
    low_priority: bool,
) -> None:
    _raise_if_cancelled(cancel_event)
    temp_source = source_audio_output.with_suffix(".tmp.mp4")
    if temp_source.exists():
        temp_source.unlink()
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_subprocess_kwargs(low_priority=low_priority),
    )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    writer_errors: list[BaseException] = []

    def writer() -> None:
        try:
            assert proc.stdin is not None
            with temp_source.open("wb") as cache_out:
                with proc.stdin:
                    _write_source_audio_mp4(video, [cache_out, proc.stdin], cancel_event=cancel_event)
            temp_source.replace(source_audio_output)
        except BaseException as exc:  # propagate worker failure after process exits.
            writer_errors.append(exc)
            try:
                proc.terminate()
            except Exception:
                pass

    writer_thread = threading.Thread(target=writer, name=f"si-source-audio-pipe-{Path(video).stem[:24]}", daemon=True)
    stdout_thread = threading.Thread(target=_drain_pipe, args=(proc.stdout, stdout_chunks), name="si-mix-stdout", daemon=True)
    stderr_thread = threading.Thread(target=_drain_pipe, args=(proc.stderr, stderr_chunks), name="si-mix-stderr", daemon=True)
    writer_thread.start()
    stdout_thread.start()
    stderr_thread.start()
    cancelled = False
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            proc.terminate()
            break
        time.sleep(0.25)
    if cancelled:
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    writer_thread.join(timeout=5.0)
    stdout_thread.join(timeout=5.0)
    stderr_thread.join(timeout=5.0)
    if cancelled:
        if temp_source.exists():
            temp_source.unlink()
        raise _BuildCancelled("SI sidecar ffmpeg cancelled")
    stdout = b"".join(stdout_chunks)
    stderr = b"".join(stderr_chunks)
    if proc.returncode != 0:
        if temp_source.exists():
            temp_source.unlink()
        raise subprocess.CalledProcessError(proc.returncode, list(cmd), output=stdout, stderr=stderr)
    if writer_errors:
        if temp_source.exists():
            temp_source.unlink()
        raise writer_errors[0]


def _sample_runs(samples: Sequence[MediaSample]) -> Iterator[tuple[int, int]]:
    run_offset = -1
    run_size = 0
    for sample in samples:
        if sample.size <= 0:
            continue
        sample_offset = int(sample.source_offset)
        sample_size = int(sample.size)
        if run_size > 0 and sample_offset == run_offset + run_size:
            run_size += sample_size
            continue
        if run_size > 0:
            yield run_offset, run_size
        run_offset = sample_offset
        run_size = sample_size
    if run_size > 0:
        yield run_offset, run_size


def _write_to_all(outputs: Sequence, data: bytes) -> None:
    for output in outputs:
        output.write(data)


def _copy_file_ranges_to_outputs(
    source: Path,
    outputs: Sequence,
    samples: Sequence[MediaSample],
    *,
    cancel_event: threading.Event | None,
    chunk_size: int = 1024 * 1024,
) -> int:
    copied = 0
    with Path(source).open("rb") as src:
        for offset, size in _sample_runs(samples):
            _raise_if_cancelled(cancel_event)
            src.seek(offset)
            remaining = size
            while remaining > 0:
                _raise_if_cancelled(cancel_event)
                chunk = src.read(min(chunk_size, remaining))
                if not chunk:
                    raise OSError(f"unexpected EOF while copying audio sample range from {source}")
                _write_to_all(outputs, chunk)
                copied += len(chunk)
                remaining -= len(chunk)
    return copied


def _copy_file_ranges_sequential_to_outputs(
    source: Path,
    outputs: Sequence,
    samples: Sequence[MediaSample],
    *,
    cancel_event: threading.Event | None,
    chunk_size: int = 32 * 1024 * 1024,
) -> int:
    active_samples = tuple(sample for sample in samples if sample.size > 0)
    if not active_samples:
        return 0
    if any(int(cur.source_offset) < int(prev.source_offset) for prev, cur in zip(active_samples, active_samples[1:], strict=False)):
        log.warning("SI sequential audio extraction saw non-monotonic sample offsets in %s; falling back to run copy", source)
        return _copy_file_ranges_to_outputs(source, outputs, samples, cancel_event=cancel_event)
    chunk_size = max(1024 * 1024, int(chunk_size))
    first_offset = min(int(sample.source_offset) for sample in active_samples)
    last_end = max(int(sample.source_offset) + int(sample.size) for sample in active_samples)
    sample_index = 0
    copied = 0
    with Path(source).open("rb") as src:
        src.seek(first_offset)
        window_start = first_offset
        while window_start < last_end:
            _raise_if_cancelled(cancel_event)
            read_size = min(chunk_size, last_end - window_start)
            data = src.read(read_size)
            if not data:
                raise OSError(f"unexpected EOF while sequentially copying audio ranges from {source}")
            window_end = window_start + len(data)
            window_audio = bytearray()
            while sample_index < len(active_samples):
                sample = active_samples[sample_index]
                sample_start = int(sample.source_offset)
                sample_end = sample_start + int(sample.size)
                if sample_end <= window_start:
                    sample_index += 1
                    continue
                if sample_end > window_end:
                    if sample_start >= window_end:
                        break
                    extra = src.read(sample_end - window_end)
                    if len(extra) != sample_end - window_end:
                        raise OSError(f"unexpected EOF while extending audio sample range from {source}")
                    data += extra
                    window_end = sample_end
                rel_start = sample_start - window_start
                rel_end = sample_end - window_start
                window_audio += data[rel_start:rel_end]
                copied += sample.size
                sample_index += 1
            if window_audio:
                _write_to_all(outputs, bytes(window_audio))
            window_start = window_end
    if sample_index < len(active_samples):
        raise OSError(f"sequential audio extraction missed {len(active_samples) - sample_index} samples from {source}")
    return copied


def _copy_file_ranges(
    source: Path,
    output,
    samples: Sequence[MediaSample],
    *,
    cancel_event: threading.Event | None,
    chunk_size: int = 1024 * 1024,
) -> int:
    if SI_AUDIO_EXTRACT_MODE == "sequential":
        return _copy_file_ranges_sequential_to_outputs(source, [output], samples, cancel_event=cancel_event)
    return _copy_file_ranges_to_outputs(source, [output], samples, cancel_event=cancel_event, chunk_size=chunk_size)


def _source_audio_sidecar_output(video: Path) -> tuple[str, Path]:
    digest = _source_audio_cache_digest(video)
    return digest, _ensure_cache_dir() / f"{digest}.source-audio.mp4"


def _source_audio_mp4_plan(video: Path, cancel_event: threading.Event | None) -> tuple[bytes, MediaSampleTable, int]:
    _raise_if_cancelled(cancel_event)
    ftyp = _read_top_level_box(video, b"ftyp")
    source_moov = _read_top_level_box(video, b"moov")
    audio_table = _read_sample_table_from_moov(video, source_moov, "audio")
    if not audio_table.samples:
        raise ValueError(f"no source audio samples found in {video}")
    mdat_payload_size = sum(sample.size for sample in audio_table.samples)
    moov = b""
    for _ in range(3):
        _raise_if_cancelled(cancel_event)
        mdat_start = len(ftyp) + len(moov)
        mdat_payload_start = mdat_start + len(_mdat_header(mdat_payload_size))
        cursor = mdat_payload_start
        audio_offsets = []
        for sample in audio_table.samples:
            audio_offsets.append(cursor)
            cursor += sample.size
        new_moov = _build_audio_only_moov(source_moov, audio_offsets)
        if len(new_moov) == len(moov):
            moov = new_moov
            break
        moov = new_moov
    return ftyp + moov + _mdat_header(mdat_payload_size), audio_table, mdat_payload_size


def _write_source_audio_mp4(
    video: Path,
    outputs: Sequence,
    *,
    cancel_event: threading.Event | None,
) -> tuple[int, int]:
    init, audio_table, mdat_payload_size = _source_audio_mp4_plan(video, cancel_event)
    _write_to_all(outputs, init)
    if SI_AUDIO_EXTRACT_MODE == "sequential":
        copied = _copy_file_ranges_sequential_to_outputs(video, outputs, audio_table.samples, cancel_event=cancel_event)
    else:
        copied = _copy_file_ranges_to_outputs(video, outputs, audio_table.samples, cancel_event=cancel_event)
    if copied != mdat_payload_size:
        raise OSError(f"source audio sidecar payload mismatch: copied={copied} expected={mdat_payload_size}")
    return mdat_payload_size, len(audio_table.samples)


def build_source_audio_sidecar(
    video: Path,
    *,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Build or reuse a small MP4 containing only the source audio track.

    This avoids using ffmpeg to scan the full source video just to decode the
    original audio for SI mixing. The output is keyed only by the source video
    stat, so SI volume/delay/duck changes can reuse it.
    """
    video = Path(video)
    digest, output = _source_audio_sidecar_output(video)
    if output.is_file() and output.stat().st_size > 0:
        log.info("reusing SI source audio sidecar: video=%s out=%s digest=%s", video, output, digest)
        return output

    with _source_audio_lock_for_digest(digest):
        if output.is_file() and output.stat().st_size > 0:
            log.info("reusing SI source audio sidecar: video=%s out=%s digest=%s", video, output, digest)
            return output

        started = time.monotonic()
        temp = output.with_suffix(".tmp.mp4")
        if temp.exists():
            temp.unlink()
        try:
            with temp.open("wb") as out:
                mdat_payload_size, sample_count = _write_source_audio_mp4(video, [out], cancel_event=cancel_event)
            temp.replace(output)
        except _BuildCancelled:
            try:
                if temp.exists():
                    temp.unlink()
            finally:
                raise
        except Exception:
            if temp.exists():
                temp.unlink()
            raise
        log.info(
            "built SI source audio sidecar: video=%s out=%s digest=%s size=%d payload=%d samples=%d elapsed=%.3fs",
            video,
            output,
            digest,
            output.stat().st_size,
            mdat_payload_size,
            sample_count,
            time.monotonic() - started,
        )
        return output


def _plan_mix_segments(total_frames: int, requested_segments: int, warmup_frames: int) -> tuple[_MixSegment, ...]:
    total_frames = max(0, int(total_frames))
    if total_frames <= 0:
        return tuple()
    segment_count = max(1, min(int(requested_segments), total_frames))
    boundaries = [int(round(i * total_frames / segment_count)) for i in range(segment_count + 1)]
    boundaries[0] = 0
    boundaries[-1] = total_frames
    segments: list[_MixSegment] = []
    for index in range(segment_count):
        start = max(0, boundaries[index])
        end = max(start, boundaries[index + 1])
        encode_start = 0 if index == 0 else max(0, start - max(0, int(warmup_frames)))
        segments.append(
            _MixSegment(
                index=index,
                start_frame=start,
                end_frame=end,
                encode_start_frame=encode_start,
                encode_end_frame=end,
            )
        )
    return tuple(segments)


def _seconds_for_frame(frame: int, *, sample_rate: int = 48000, frame_samples: int = 1024) -> float:
    return max(0, int(frame)) * float(frame_samples) / float(sample_rate)


def _fmt_ffmpeg_seconds(value: float) -> str:
    return f"{max(0.0, float(value)):.6f}"


def _effective_mix_segments(total_frames: int, *, low_priority: bool) -> int:
    configured = max(1, int(SI_MIX_PARALLEL_MAX))
    if low_priority:
        configured = max(1, configured // 2)
    return max(1, min(configured, max(1, int(total_frames))))


def _encode_mix_segment(
    source_audio: Path,
    si_wav: Path,
    params: SIMixParams,
    segment: _MixSegment,
    output: Path,
    *,
    cancel_event: threading.Event | None,
    low_priority: bool,
) -> Path:
    start_seconds = _seconds_for_frame(segment.encode_start_frame)
    duration_seconds = _seconds_for_frame(segment.encode_end_frame - segment.encode_start_frame)
    filt = build_si_mix_filter(
        params.mix_channel,
        params.original_volume_percent,
        params.si_volume_percent,
        params.si_delay_seconds,
        params.duck_original,
    )
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-ss",
        _fmt_ffmpeg_seconds(start_seconds),
        "-t",
        _fmt_ffmpeg_seconds(duration_seconds),
        "-i",
        str(source_audio),
        "-ss",
        _fmt_ffmpeg_seconds(start_seconds),
        "-t",
        _fmt_ffmpeg_seconds(duration_seconds),
        "-i",
        str(si_wav),
        "-filter_complex",
        filt,
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
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output),
    ]
    _run_ffmpeg_sidecar(cmd, cancel_event=cancel_event, low_priority=low_priority)
    if not output.is_file() or output.stat().st_size <= 0:
        raise OSError(f"SI mix segment was not written: {output}")
    return output


def _select_segment_samples(segment_path: Path, segment: _MixSegment) -> _SegmentSelection:
    moov = _read_top_level_box(segment_path, b"moov")
    audio_trak = _find_track(moov, "audio").trak
    stbl = _track_stbl(moov, audio_trak)
    table = _read_sample_table_from_moov(segment_path, moov, "audio")
    durations = _parse_stts_durations(moov, stbl, len(table.samples))
    priming_media_time = _audio_edit_media_time_from_moov(moov)
    priming_frames = _audio_priming_frames_from_moov(moov)
    if segment.index == 0:
        drop = 0
        keep = priming_frames + segment.keep_frames
    else:
        drop = priming_frames + segment.leading_frames
        keep = segment.keep_frames
    selected = table.samples[drop:drop + keep]
    selected_durations = durations[drop:drop + keep]
    if len(selected) != keep:
        raise ValueError(
            f"segment {segment.index} has insufficient AAC frames: "
            f"drop={drop} keep={keep} available={len(table.samples)} path={segment_path}"
        )
    return _SegmentSelection(
        path=segment_path,
        samples=tuple(selected),
        durations=tuple(selected_durations),
        priming_media_time=priming_media_time if segment.index == 0 else 0,
        priming_frames=priming_frames if segment.index == 0 else 0,
    )


def _stitch_aac_segments(
    segment_paths: Sequence[Path],
    segments: Sequence[_MixSegment],
    output: Path,
    *,
    cancel_event: threading.Event | None,
) -> Path:
    if len(segment_paths) != len(segments):
        raise ValueError("segment path/count mismatch")
    selections = [_select_segment_samples(path, segment) for path, segment in zip(segment_paths, segments, strict=True)]
    if not selections:
        raise ValueError("no SI mix segments to stitch")
    ftyp = _read_top_level_box(selections[0].path, b"ftyp")
    template_moov = _read_top_level_box(selections[0].path, b"moov")
    selected_samples = [sample for selection in selections for sample in selection.samples]
    sample_delta = 1024
    normalized_durations: list[int] = []
    for index, selection in enumerate(selections):
        durations = list(selection.durations)
        if index == 0:
            if index + 1 < len(selections) and durations:
                # Segment-local -t trimming can shorten the final packet in the
                # first output. The stitched stream is continuous, so only the
                # initial encoder/edit-list timing belongs in the final stts.
                durations[-1] = sample_delta
        else:
            durations = [sample_delta] * len(durations)
        normalized_durations.extend(durations)
    sizes = [sample.size for sample in selected_samples]
    payload_size = sum(sizes)
    edit_media_time = selections[0].priming_media_time
    moov = b""
    offsets: list[int] = []
    for _ in range(3):
        mdat_start = len(ftyp) + len(moov)
        mdat_payload_start = mdat_start + len(_mdat_header(payload_size))
        cursor = mdat_payload_start
        offsets = []
        for size in sizes:
            offsets.append(cursor)
            cursor += size
        new_moov = _build_stitched_audio_moov(
            template_moov,
            offsets,
            sizes,
            normalized_durations,
            edit_media_time=edit_media_time,
        )
        if len(new_moov) == len(moov):
            moov = new_moov
            break
        moov = new_moov
    temp = output.with_suffix(".tmp.mp4")
    if temp.exists():
        temp.unlink()
    try:
        with temp.open("wb") as out:
            out.write(ftyp)
            out.write(moov)
            out.write(_mdat_header(payload_size))
            for selection in selections:
                _raise_if_cancelled(cancel_event)
                _copy_file_ranges_to_outputs(selection.path, [out], selection.samples, cancel_event=cancel_event)
        temp.replace(output)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    return output


def build_mixed_audio_sidecar_parallel(
    video: Path,
    si_wav: Path,
    params: SIMixParams,
    output: Path,
    *,
    cancel_event: threading.Event | None = None,
    low_priority: bool = False,
) -> Path:
    source_audio = build_source_audio_sidecar(video, cancel_event=cancel_event)
    source_table = read_media_sample_table(source_audio, "audio")
    total_frames = max(1, len(source_table.samples))
    segment_count = _effective_mix_segments(total_frames, low_priority=low_priority)
    if segment_count <= 1:
        raise ValueError("parallel SI mix disabled by segment count")
    base_warmup_frames = int(math.ceil((max(0, int(SI_MIX_SEGMENT_WARMUP_MS)) / 1000.0) * 48000.0 / 1024.0))
    delay_warmup_frames = int(math.ceil(max(0.0, float(params.si_delay_seconds)) * 48000.0 / 1024.0))
    warmup_frames = base_warmup_frames + delay_warmup_frames
    segments = _plan_mix_segments(total_frames, segment_count, warmup_frames)
    if len(segments) <= 1:
        raise ValueError("parallel SI mix produced one segment")

    cache_dir = _ensure_cache_dir()
    temp_prefix = f"{output.stem}.{os.getpid()}.{uuid.uuid4().hex[:12]}"
    segment_paths = [cache_dir / f"{temp_prefix}.seg{segment.index:03d}.mp4" for segment in segments]
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=len(segments), thread_name_prefix="si-mix-seg") as executor:
            futures = [
                executor.submit(
                    _encode_mix_segment,
                    source_audio,
                    si_wav,
                    params,
                    segment,
                    segment_path,
                    cancel_event=cancel_event,
                    low_priority=low_priority,
                )
                for segment, segment_path in zip(segments, segment_paths, strict=True)
            ]
            for future in as_completed(futures):
                _raise_if_cancelled(cancel_event)
                future.result()
        _stitch_aac_segments(segment_paths, segments, output, cancel_event=cancel_event)
    finally:
        for segment_path in segment_paths:
            try:
                if segment_path.exists():
                    segment_path.unlink()
            except OSError:
                pass
    log.info(
        "built parallel SI mixed AAC sidecar: video=%s si=%s out=%s segments=%d warmup_frames=%d size=%d elapsed=%.3fs",
        video,
        si_wav,
        output,
        len(segments),
        warmup_frames,
        output.stat().st_size if output.exists() else 0,
        time.monotonic() - started,
    )
    return output


def build_mixed_audio_sidecar(
    video: Path,
    si_wav: Path,
    params: SIMixParams,
    *,
    cancel_event: threading.Event | None = None,
    low_priority: bool = False,
) -> Path:
    """Build or reuse the small AAC MP4 sidecar for the current SI mix."""
    video = Path(video)
    si_wav = Path(si_wav)
    digest = _cache_digest(video, si_wav, params)
    cache_dir = _ensure_cache_dir()
    output = cache_dir / f"{digest}.audio.mp4"
    if output.is_file() and output.stat().st_size > 0:
        log.info(
            "reusing SI mixed AAC sidecar: video=%s si=%s out=%s digest=%s delay=%.1f mix=%s orig_vol=%s si_vol=%s duck=%s",
            video,
            si_wav,
            output,
            digest,
            params.si_delay_seconds,
            params.mix_channel,
            params.original_volume_percent,
            params.si_volume_percent,
            params.duck_original,
        )
        return output

    encoder = _selected_mix_encoder()
    if _use_segmented_aac(encoder):
        try:
            return build_mixed_audio_sidecar_parallel(
                video,
                si_wav,
                params,
                output,
                cancel_event=cancel_event,
                low_priority=low_priority,
            )
        except _BuildCancelled:
            raise
        except Exception as exc:
            log.warning("parallel SI mixed AAC sidecar build failed for %s; falling back to single ffmpeg: %s", video, exc)
            if output.exists():
                try:
                    output.unlink()
                except OSError:
                    pass

    temp = output.with_suffix(".tmp.mp4")
    if temp.exists():
        temp.unlink()
    audio_input_arg = str(video)
    audio_input_kind = "source-video"
    source_audio_output: Path | None = None
    source_audio_lock: threading.Lock | None = None
    try:
        source_audio_digest, source_audio_output = _source_audio_sidecar_output(video)
        source_audio_lock = _source_audio_lock_for_digest(source_audio_digest)
        source_audio_lock.acquire()
        if source_audio_output.is_file() and source_audio_output.stat().st_size > 0:
            audio_input_arg = str(source_audio_output)
            audio_input_kind = "source-audio-sidecar"
            source_audio_lock.release()
            source_audio_lock = None
        else:
            audio_input_arg = "pipe:0"
            audio_input_kind = "source-audio-pipe"
    except _BuildCancelled:
        if source_audio_lock is not None:
            source_audio_lock.release()
        raise
    except Exception as exc:
        if source_audio_lock is not None:
            source_audio_lock.release()
            source_audio_lock = None
        log.warning("SI source audio sidecar setup failed for %s; falling back to full source input: %s", video, exc)
    filt = build_si_mix_filter(
        params.mix_channel,
        params.original_volume_percent,
        params.si_volume_percent,
        params.si_delay_seconds,
        params.duck_original,
    )
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        audio_input_arg,
        "-i",
        str(si_wav),
        "-filter_complex",
        filt,
        "-map",
        "[si_track]",
        "-c:a",
        encoder,
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temp),
    ]
    log.info(
        "building SI mixed AAC sidecar: video=%s audio_input=%s input_kind=%s encoder=%s si=%s out=%s digest=%s delay=%.1f mix=%s orig_vol=%s si_vol=%s duck=%s",
        video,
        audio_input_arg,
        audio_input_kind,
        encoder,
        si_wav,
        output,
        digest,
        params.si_delay_seconds,
        params.mix_channel,
        params.original_volume_percent,
        params.si_volume_percent,
        params.duck_original,
    )
    try:
        if audio_input_kind == "source-audio-pipe":
            assert source_audio_output is not None
            try:
                _run_ffmpeg_sidecar_with_source_audio_pipe(
                    cmd,
                    video=video,
                    source_audio_output=source_audio_output,
                    cancel_event=cancel_event,
                    low_priority=low_priority,
                )
            except _BuildCancelled:
                raise
            except Exception as exc:
                if temp.exists():
                    temp.unlink()
                log.warning("SI source audio pipe mix failed for %s; falling back to full source input: %s", video, exc)
                fallback_cmd = list(cmd)
                first_input = fallback_cmd.index("-i") + 1
                fallback_cmd[first_input] = str(video)
                _run_ffmpeg_sidecar(fallback_cmd, cancel_event=cancel_event, low_priority=low_priority)
        else:
            _run_ffmpeg_sidecar(cmd, cancel_event=cancel_event, low_priority=low_priority)
        temp.replace(output)
    except _BuildCancelled:
        try:
            if temp.exists():
                temp.unlink()
        finally:
            raise
    finally:
        if source_audio_lock is not None:
            source_audio_lock.release()
    return output


_layout_cache_lock = threading.Lock()
_layout_cache: dict[str, ProgressiveSIVirtualMp4] = {}
_layout_build_locks: dict[str, threading.Lock] = {}
_layout_prewarm_inflight: set[str] = set()
_layout_prewarm_queue: queue.PriorityQueue[tuple[int, int, "_PrewarmTask"]] | None = None
_layout_prewarm_counter = itertools.count()
_layout_prewarm_worker_started = False
_LAYOUT_CACHE_LIMIT = 8


@dataclass(frozen=True)
class _PrewarmTask:
    digest: str
    video: Path
    si_wav: Path
    params: SIMixParams
    reason: str


@dataclass(frozen=True)
class _MixSegment:
    index: int
    start_frame: int
    end_frame: int
    encode_start_frame: int
    encode_end_frame: int

    @property
    def keep_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def leading_frames(self) -> int:
        return max(0, self.start_frame - self.encode_start_frame)


@dataclass(frozen=True)
class _SegmentSelection:
    path: Path
    samples: tuple[MediaSample, ...]
    durations: tuple[int, ...]
    priming_media_time: int = 0
    priming_frames: int = 0


def _layout_etag(digest: str, content_length: int, moov_size: int) -> str:
    raw = f"{digest}:{content_length}:{moov_size}:progressive-v{_PROGRESSIVE_CACHE_SCHEMA}".encode("ascii")
    return hashlib.sha256(raw).hexdigest()[:32]


def _build_lock_for_digest(digest: str) -> threading.Lock:
    with _layout_cache_lock:
        lock = _layout_build_locks.get(digest)
        if lock is None:
            lock = threading.Lock()
            _layout_build_locks[digest] = lock
        return lock


def build_progressive_si_virtual_mp4(
    video: Path,
    si_wav: Path,
    params: SIMixParams,
    *,
    priority: int = _BUILD_PRIORITY_PLAYBACK,
    reason: str = "playback",
    cancelable: bool = False,
) -> ProgressiveSIVirtualMp4:
    """Build a progressive virtual MP4 layout for SI M1 testing.

    The first call scans the source MP4 and encodes the mixed AAC sidecar. The
    resulting layout is cached in-process and the AAC sidecar is cached on disk.
    """
    video = Path(video)
    si_wav = Path(si_wav)
    sidecar_digest = _cache_digest(video, si_wav, params)
    audio_edit_mode = SI_AUDIO_EDIT_MODE
    digest = _layout_cache_digest(sidecar_digest, audio_edit_mode)
    with _layout_cache_lock:
        cached = _layout_cache.get(digest)
        if cached is not None:
            return cached

    with _build_lock_for_digest(digest):
        with _layout_cache_lock:
            cached = _layout_cache.get(digest)
            if cached is not None:
                return cached

        context = _enter_build_slot(digest, video, priority=priority, reason=reason, cancelable=cancelable)
        try:
            with _layout_cache_lock:
                cached = _layout_cache.get(digest)
                if cached is not None:
                    return cached

            started = time.monotonic()
            _raise_if_cancelled(context.cancel_event)
            audio = build_mixed_audio_sidecar(
                video,
                si_wav,
                params,
                cancel_event=context.cancel_event,
                low_priority=priority > _BUILD_PRIORITY_PLAYBACK,
            )
            _raise_if_cancelled(context.cancel_event)
            ftyp = _read_top_level_box(video, b"ftyp")
            source_moov = _read_top_level_box(video, b"moov")
            audio_moov = _read_top_level_box(audio, b"moov")
            video_table = read_media_sample_table(video, "video")
            _raise_if_cancelled(context.cancel_event)
            audio_table = read_media_sample_table(audio, "audio")
            _raise_if_cancelled(context.cancel_event)
            if not video_table.samples:
                raise ValueError(f"no video samples found in {video}")
            if not audio_table.samples:
                raise ValueError(f"no audio samples found in {audio}")

            mdat_payload_size = sum(sample.size for sample in video_table.samples) + sum(sample.size for sample in audio_table.samples)
            # co64 table sizes are independent of the actual offset values. Iterate
            # once more after the first moov build so the mdat start offset is exact.
            moov = b""
            ordered: list[_SampleLayout] = []
            for _ in range(3):
                _raise_if_cancelled(context.cancel_event)
                mdat_start = len(ftyp) + len(moov)
                mdat_payload_start = mdat_start + len(_mdat_header(mdat_payload_size))
                ordered, video_offsets, audio_offsets, payload = _layout_samples(
                    video,
                    audio,
                    video_table.samples,
                    audio_table.samples,
                    mdat_payload_start,
                )
                if payload != mdat_payload_size:
                    raise ValueError("sample layout payload size mismatch")
                new_moov = _build_moov(
                    source_moov,
                    audio_moov,
                    video_offsets,
                    audio_offsets,
                    audio_edit_mode=audio_edit_mode,
                )
                if len(new_moov) == len(moov):
                    moov = new_moov
                    break
                moov = new_moov

            mdat_header = _mdat_header(mdat_payload_size)
            init = ftyp + moov + mdat_header
            regions: list[VirtualRegion] = [VirtualRegion.memory(0, init)]
            cursor = len(init)
            for item in ordered:
                regions.append(VirtualRegion.file(cursor, item.path, item.source_offset, item.size))
                cursor += item.size
            content_length = virtual_size(regions)
            layout = ProgressiveSIVirtualMp4(
                video_path=video,
                audio_path=audio,
                regions=tuple(regions),
                content_length=content_length,
                etag=_layout_etag(digest, content_length, len(moov)),
                video_samples=len(video_table.samples),
                audio_samples=len(audio_table.samples),
                moov_size=len(moov),
                mdat_payload_size=mdat_payload_size,
                audio_edit_mode=audio_edit_mode,
            )
            with _layout_cache_lock:
                _layout_cache[digest] = layout
                while len(_layout_cache) > _LAYOUT_CACHE_LIMIT:
                    _layout_cache.pop(next(iter(_layout_cache)))
            log.info(
                "built SI progressive virtual MP4: video=%s audio=%s audio_edit=%s reason=%s size=%d moov=%d samples=%d+%d regions=%d elapsed=%.3fs",
                video,
                audio,
                layout.audio_edit_mode,
                reason,
                layout.content_length,
                layout.moov_size,
                layout.video_samples,
                layout.audio_samples,
                len(layout.regions),
                time.monotonic() - started,
            )
            return layout
        finally:
            _leave_build_slot(context)


def _ensure_prewarm_queue_locked() -> queue.PriorityQueue[tuple[int, int, _PrewarmTask]]:
    global _layout_prewarm_queue
    if _layout_prewarm_queue is None:
        _layout_prewarm_queue = queue.PriorityQueue(maxsize=max(1, int(SI_PREWARM_QUEUE_MAX)))
    return _layout_prewarm_queue


def _ensure_prewarm_worker_locked() -> None:
    global _layout_prewarm_worker_started
    if _layout_prewarm_worker_started:
        return
    _layout_prewarm_worker_started = True
    thread = threading.Thread(target=_prewarm_worker, name="si-mp4-prewarm-worker", daemon=True)
    thread.start()


def _prewarm_worker() -> None:
    while True:
        with _layout_cache_lock:
            work_queue = _ensure_prewarm_queue_locked()
        priority, _sequence, task = work_queue.get()
        try:
            with _layout_cache_lock:
                if task.digest in _layout_cache:
                    continue
            log.info("prewarm SI progressive virtual MP4 begin: video=%s reason=%s", task.video, task.reason)
            build_progressive_si_virtual_mp4(
                task.video,
                task.si_wav,
                task.params,
                priority=priority,
                reason=task.reason,
                cancelable=True,
            )
            log.info("prewarm SI progressive virtual MP4 done: video=%s reason=%s", task.video, task.reason)
        except _BuildCancelled:
            log.info("prewarm SI progressive virtual MP4 cancelled: video=%s reason=%s", task.video, task.reason)
        except Exception as exc:
            log.warning("prewarm SI progressive virtual MP4 failed: video=%s reason=%s error=%s", task.video, task.reason, exc)
        finally:
            with _layout_cache_lock:
                _layout_prewarm_inflight.discard(task.digest)
            work_queue.task_done()


def prewarm_progressive_si_virtual_mp4(video: Path, si_wav: Path, params: SIMixParams, *, reason: str = "dlna") -> bool:
    """Start a background layout build if it is not cached or already queued."""
    if SI_PREWARM_QUEUE_MAX <= 0:
        return False
    video = Path(video)
    si_wav = Path(si_wav)
    sidecar_digest = _cache_digest(video, si_wav, params)
    digest = _layout_cache_digest(sidecar_digest, SI_AUDIO_EDIT_MODE)
    with _layout_cache_lock:
        if digest in _layout_cache or digest in _layout_prewarm_inflight:
            return False
        work_queue = _ensure_prewarm_queue_locked()
        task = _PrewarmTask(digest=digest, video=video, si_wav=si_wav, params=params, reason=reason)
        try:
            work_queue.put_nowait((_BUILD_PRIORITY_PREWARM, next(_layout_prewarm_counter), task))
        except queue.Full:
            log.info("SI progressive prewarm queue full; skipping video=%s reason=%s", video, reason)
            return False
        _layout_prewarm_inflight.add(digest)
        _ensure_prewarm_worker_locked()
    return True


def iter_file_slice(path: Path, offset: int, size: int, *, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    remaining = max(0, int(size))
    if remaining <= 0:
        return
    with Path(path).open("rb") as fh:
        fh.seek(max(0, int(offset)))
        while remaining > 0:
            chunk = fh.read(min(max(1, int(chunk_size)), remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def iter_virtual_range(
    regions: Sequence[VirtualRegion],
    start: int,
    end_inclusive: int,
    *,
    chunk_size: int = 64 * 1024,
) -> Iterator[bytes]:
    """Yield bytes for an inclusive logical range across memory/file regions."""
    if end_inclusive < start:
        return
    range_start = max(0, int(start))
    range_end = int(end_inclusive) + 1
    logical_size = virtual_size(regions)
    if range_start >= logical_size:
        return
    range_end = min(range_end, logical_size)

    handles: dict[Path, object] = {}
    try:
        for region in sorted(regions, key=lambda r: r.start):
            overlap_start = max(range_start, region.start)
            overlap_end = min(range_end, region.end)
            if overlap_start >= overlap_end:
                continue
            inside = overlap_start - region.start
            length = overlap_end - overlap_start
            if region.kind == "memory":
                yield region.data[inside:inside + length]
            elif region.kind == "file":
                if region.path is None:
                    raise ValueError("file virtual region requires a path")
                path = Path(region.path)
                fh = handles.get(path)
                if fh is None:
                    fh = path.open("rb")
                    handles[path] = fh
                fh.seek(region.source_offset + inside)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(max(1, int(chunk_size)), remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
            else:
                raise ValueError(f"unsupported virtual region kind: {region.kind!r}")
    finally:
        for fh in handles.values():
            close = getattr(fh, "close", None)
            if callable(close):
                close()


def read_video_sample_table(path: Path, *, limit: int | None = None) -> VideoSampleTable:
    """Read source MP4 video sample offsets with PyAV.

    PyAV is imported lazily so the server can still start in environments that
    have not enabled the Phase 1 dependency yet.
    """
    import av  # type: ignore[import-not-found]

    source = Path(path)
    table = read_media_sample_table(source, "video", limit=limit)
    with av.open(str(source)) as container:
        ctx = container.streams.video[0].codec_context
        return VideoSampleTable(
            path=source,
            codec_name=table.codec_name,
            width=int(getattr(ctx, "width", 0) or 0),
            height=int(getattr(ctx, "height", 0) or 0),
            time_base=table.time_base,
            duration_seconds=table.duration_seconds,
            extradata=table.extradata,
            samples=tuple(
                VideoSample(
                    index=sample.index,
                    source_offset=sample.source_offset,
                    size=sample.size,
                    pts=sample.pts,
                    dts=sample.dts,
                    keyframe=sample.keyframe,
                )
                for sample in table.samples
            ),
        )


def verify_sample_splice_identity(path: Path, *, limit: int = 16) -> bool:
    """Verify that demuxed packet bytes match source file byte slices."""
    import av  # type: ignore[import-not-found]

    source = Path(path)
    checked = 0
    with av.open(str(source)) as container, source.open("rb") as raw:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size <= 0 or packet.pos is None:
                continue
            raw.seek(int(packet.pos))
            if raw.read(int(packet.size)) != bytes(packet):
                return False
            checked += 1
            if checked >= max(1, int(limit)):
                return True
    return checked > 0
