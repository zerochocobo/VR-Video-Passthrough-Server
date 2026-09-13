"""Experimental video-only VMP4 slot layout for /passthrough_seek.

This is a deliberately small backend skeleton: it builds a byte-stable
``ftyp + moov + mdat`` file whose video track maps one keyframe sample per GOP
slot.  The samples still come from the source MP4; later slices can replace
those per-slot bytes with generated passthrough GOP payloads without changing
the HTTP range contract.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.si_virtual_mp4 import (
    MediaSample,
    _box,
    _box_bytes,
    _children,
    _find_track,
    _full_box,
    _iter_boxes,
    _make_co64,
    _make_stsc_one_sample_per_chunk,
    _make_stsz,
    _make_stts_from_durations,
    _mvhd_timescale,
    _patch_mdhd_duration,
    _patch_mvhd_duration_and_next_track_id,
    _patch_tkhd_duration,
    _read_top_level_box,
    _require_child,
    _track_id,
    _u32,
    _u64,
    read_media_sample_table,
)

_SLOT_CACHE_SCHEMA = 2
_manifest_write_locks_guard = threading.RLock()
_manifest_write_locks: dict[str, threading.RLock] = {}


class Vmp4SlotLayoutError(ValueError):
    """Raised when a source cannot be represented by the slot skeleton."""


@dataclass(frozen=True)
class Vmp4SlotSample:
    slot_index: int
    source_sample_index: int
    source_offset: int
    source_size: int
    sample_size: int
    start_time_sec: float


@dataclass(frozen=True)
class PassthroughVmp4SlotLayout:
    source_path: Path
    init: bytes
    total_size: int
    mdat_payload_start: int
    mdat_payload_size: int
    slot_count: int
    slot_size: int
    slot_stride: int
    slot_gap_size: int
    slot_duration_sec: float
    source_codec_name: str
    codec_name: str
    nal_length_size: int
    output_stsd_sha256: str
    placeholder_payload: bytes
    samples: tuple[Vmp4SlotSample, ...]
    moov_size: int


@dataclass(frozen=True)
class Vmp4SlotOutputTemplate:
    stsd: bytes
    payload: bytes
    codec_name: str
    nal_length_size: int
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class Vmp4SlotCacheSlot:
    index: int
    state: str
    payload_path: Path | None
    payload_size: int = 0
    reason: str = ""


@dataclass(frozen=True)
class Vmp4SlotCacheStatus:
    cache_dir: Path
    manifest_path: Path
    digest: str
    slots: tuple[Vmp4SlotCacheSlot, ...]

    @property
    def ready_payloads(self) -> dict[int, Path]:
        return {
            slot.index: slot.payload_path
            for slot in self.slots
            if slot.state == "ready" and slot.payload_path is not None
        }

    @property
    def state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for slot in self.slots:
            counts[slot.state] = counts.get(slot.state, 0) + 1
        return counts

    def slot(self, index: int) -> Vmp4SlotCacheSlot | None:
        if 0 <= index < len(self.slots):
            return self.slots[index]
        return None


def build_passthrough_vmp4_slot_layout(
    path: Path,
    *,
    duration_sec: float,
    fps: float,
    total_size: int,
    gop_frames: int,
    slot_duration_sec: float | None = None,
    max_sample_bytes: int = 1024 * 1024,
    output_stsd: bytes | None = None,
    output_codec_name: str = "hevc",
    output_width: int = 0,
    output_height: int = 0,
    placeholder_payload: bytes = b"",
) -> PassthroughVmp4SlotLayout:
    source = Path(path)
    total = max(0, int(total_size))
    duration = max(0.0, float(duration_sec or 0.0))
    if total <= 0:
        raise Vmp4SlotLayoutError("source-size-missing")
    if duration <= 0.0:
        raise Vmp4SlotLayoutError("duration-missing")

    output_fps = max(0.0, float(fps or 0.0))
    if slot_duration_sec is not None and slot_duration_sec > 0:
        slot_sec = float(slot_duration_sec)
    elif output_fps > 0 and gop_frames > 0:
        slot_sec = max(0.25, float(gop_frames) / output_fps)
    else:
        slot_sec = 4.0
    slot_count = max(1, int(math.ceil(duration / slot_sec)))

    try:
        ftyp = _read_top_level_box(source, b"ftyp")
    except Exception:
        ftyp = _box("ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")
    try:
        source_moov = _read_top_level_box(source, b"moov")
        table = read_media_sample_table(source, "video")
    except Exception as exc:
        raise Vmp4SlotLayoutError(f"source-video-table-error:{type(exc).__name__}") from exc
    if not table.samples:
        raise Vmp4SlotLayoutError("source-video-samples-missing")

    candidates = tuple(sample for sample in table.samples if sample.size > 0 and sample.keyframe)
    if not candidates:
        candidates = tuple(sample for sample in table.samples if sample.size > 0)
    if not candidates:
        raise Vmp4SlotLayoutError("source-video-samples-empty")

    selected = _select_slot_samples(candidates, slot_count, slot_sec)
    media_timescale = max(1, int(table.time_base.denominator))
    sample_durations = _slot_sample_durations(slot_count, slot_sec, duration, media_timescale)
    media_duration = sum(sample_durations)
    source_codec_name = str(table.codec_name or "").lower()
    source_stsd = _source_video_stsd(source_moov)
    output_stsd = bytes(output_stsd or b"")
    if not output_stsd:
        if source_codec_name != "hevc":
            raise Vmp4SlotLayoutError("slot-output-stsd-missing")
        output_stsd = source_stsd
    codec_name = _normalize_output_codec(output_codec_name) or _stsd_codec_name(output_stsd) or "hevc"
    nal_length_size = _probe_nal_length_size(output_stsd)
    placeholder = bytes(placeholder_payload or b"")
    max_payload_size = len(placeholder)
    output_stsd_sha256 = hashlib.sha256(output_stsd).hexdigest()

    moov = _build_slot_moov(
        source_moov,
        output_stsd=output_stsd,
        offsets=[0] * slot_count,
        sizes=[1] * slot_count,
        durations=sample_durations,
        media_duration=media_duration,
        movie_duration_sec=duration,
        output_width=output_width,
        output_height=output_height,
    )
    mdat_header = b""
    mdat_payload_start = 0
    mdat_payload_size = 0
    slot_size = 0
    slot_stride = 0
    converged = False
    for _ in range(4):
        remaining_for_mdat = total - len(ftyp) - len(moov)
        if remaining_for_mdat <= 16:
            raise Vmp4SlotLayoutError("layout-header-exceeds-source-size")
        mdat_payload_size = remaining_for_mdat - 16
        mdat_header = _extended_mdat_header(mdat_payload_size)
        mdat_payload_start = len(ftyp) + len(moov) + len(mdat_header)
        slot_stride = mdat_payload_size // slot_count
        if slot_stride <= 0:
            raise Vmp4SlotLayoutError("slot-size-zero")
        sample_cap = max(0, int(max_sample_bytes or 0))
        slot_size = min(slot_stride, sample_cap) if sample_cap > 0 else slot_stride
        if max_payload_size > slot_size:
            if max_payload_size <= slot_stride:
                slot_size = max_payload_size
            else:
                raise Vmp4SlotLayoutError("slot-placeholder-oversize")
        offsets = [mdat_payload_start + i * slot_stride for i in range(slot_count)]
        fixed_sample_sizes = [slot_size] * slot_count
        new_moov = _build_slot_moov(
            source_moov,
            output_stsd=output_stsd,
            offsets=offsets,
            sizes=fixed_sample_sizes,
            durations=sample_durations,
            media_duration=media_duration,
            movie_duration_sec=duration,
            output_width=output_width,
            output_height=output_height,
        )
        if len(new_moov) == len(moov):
            moov = new_moov
            converged = True
            break
        moov = new_moov
    if not converged:
        raise Vmp4SlotLayoutError("layout-moov-offsets-not-converged")

    init = ftyp + moov + mdat_header
    samples = tuple(
        Vmp4SlotSample(
            slot_index=i,
            source_sample_index=int(sample.index),
            source_offset=int(sample.source_offset),
            source_size=int(sample.size),
            sample_size=int(slot_size),
            start_time_sec=min(duration, i * slot_sec),
        )
        for i, sample in enumerate(selected)
    )
    return PassthroughVmp4SlotLayout(
        source_path=source,
        init=init,
        total_size=total,
        mdat_payload_start=mdat_payload_start,
        mdat_payload_size=mdat_payload_size,
        slot_count=slot_count,
        slot_size=slot_size,
        slot_stride=slot_stride,
        slot_gap_size=max(0, int(slot_stride) - int(slot_size)),
        slot_duration_sec=slot_sec,
        source_codec_name=source_codec_name,
        codec_name=codec_name,
        nal_length_size=nal_length_size,
        output_stsd_sha256=output_stsd_sha256,
        placeholder_payload=placeholder[:slot_size],
        samples=samples,
        moov_size=len(moov),
    )


def load_vmp4_slot_output_template(path: Path) -> Vmp4SlotOutputTemplate:
    template = Path(path)
    try:
        moov = _read_top_level_box(template, b"moov")
        table = read_media_sample_table(template, "video")
    except Exception as exc:
        raise Vmp4SlotLayoutError(f"slot-output-template-error:{type(exc).__name__}") from exc
    stsd = _source_video_stsd(moov)
    codec_name = str(table.codec_name or _stsd_codec_name(stsd)).lower()
    if codec_name != "hevc":
        raise Vmp4SlotLayoutError(f"slot-output-template-codec:{codec_name or 'unknown'}")
    if not table.samples:
        raise Vmp4SlotLayoutError("slot-output-template-empty")
    sample = next((item for item in table.samples if item.size > 0 and item.keyframe), table.samples[0])
    with template.open("rb") as fh:
        fh.seek(sample.source_offset)
        payload = fh.read(max(0, int(sample.size)))
    if not payload:
        raise Vmp4SlotLayoutError("slot-output-template-payload-empty")
    return Vmp4SlotOutputTemplate(
        stsd=stsd,
        payload=payload,
        codec_name="hevc",
        nal_length_size=_probe_nal_length_size(stsd),
        width=0,
        height=0,
    )


def iter_vmp4_slot_range(
    layout: PassthroughVmp4SlotLayout,
    start: int,
    end_inclusive: int,
    *,
    chunk_size: int = 64 * 1024,
    payload_paths: Mapping[int, Path] | None = None,
) -> Iterator[bytes]:
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

    chunk = max(1, int(chunk_size))
    for sample in layout.samples:
        slot_start = layout.mdat_payload_start + sample.slot_index * layout.slot_stride
        slot_end = slot_start + sample.sample_size
        if slot_end <= cursor:
            continue
        if slot_start > stop:
            break
        if cursor < slot_start:
            gap_end = min(stop + 1, slot_start)
            yield from _iter_zero_bytes(gap_end - cursor, chunk)
            cursor = gap_end
            if cursor > stop:
                return
        overlap_start = max(cursor, slot_start)
        overlap_end = min(stop + 1, slot_end)
        inside = overlap_start - slot_start
        payload_path = Path(payload_paths[sample.slot_index]) if payload_paths and sample.slot_index in payload_paths else None
        emitted_payload_size = 0
        if payload_path is not None:
            try:
                file_payload_size = min(max(0, int(payload_path.stat().st_size)), sample.sample_size)
            except OSError:
                payload_path = None
                file_payload_size = 0
        else:
            file_payload_size = 0
        if payload_path is not None and file_payload_size > 0:
            emitted_payload_size = file_payload_size
            if inside < emitted_payload_size:
                sample_read = min(emitted_payload_size - inside, overlap_end - overlap_start)
                yield from _iter_file_range(payload_path, inside, sample_read, chunk)
                overlap_start += sample_read
        else:
            placeholder = layout.placeholder_payload[:sample.sample_size]
            emitted_payload_size = len(placeholder)
            if inside < emitted_payload_size:
                sample_read = min(emitted_payload_size - inside, overlap_end - overlap_start)
                yield placeholder[inside:inside + sample_read]
                overlap_start += sample_read
        if overlap_start < overlap_end:
            filler_start = max(0, overlap_start - slot_start - emitted_payload_size)
            filler_end = max(0, overlap_end - slot_start - emitted_payload_size)
            yield from _iter_filler_range(
                layout.codec_name,
                layout.nal_length_size,
                max(0, sample.sample_size - emitted_payload_size),
                filler_start,
                filler_end,
                chunk,
            )
        cursor = max(cursor, overlap_end)
        if cursor > stop:
            return

    if cursor <= stop:
        yield from _iter_zero_bytes(stop - cursor + 1, chunk)


def vmp4_slot_index_for_offset(layout: PassthroughVmp4SlotLayout, offset: int) -> int | None:
    rel = int(offset) - int(layout.mdat_payload_start)
    if rel < 0 or layout.slot_stride <= 0 or layout.slot_size <= 0:
        return None
    slot_index = rel // layout.slot_stride
    if slot_index < 0 or slot_index >= layout.slot_count:
        return None
    if rel % layout.slot_stride >= layout.slot_size:
        return None
    return int(slot_index)


def ensure_vmp4_slot_manifest(
    layout: PassthroughVmp4SlotLayout,
    cache_root: Path,
    *,
    output_mode: str,
    fps: float,
    gop_frames: int,
) -> Vmp4SlotCacheStatus:
    digest = _slot_cache_digest(
        layout,
        output_mode=output_mode,
        fps=fps,
        gop_frames=gop_frames,
    )
    cache_dir = Path(cache_root) / digest
    manifest_path = cache_dir / "manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest(manifest_path)
    existing_slots = {
        int(slot.get("index")): slot
        for slot in existing.get("slots", [])
        if isinstance(slot, dict) and str(slot.get("index", "")).isdigit()
    } if existing.get("digest") == digest else {}

    slots: list[Vmp4SlotCacheSlot] = []
    manifest_slots: list[dict[str, object]] = []
    now = time.time()
    for sample in layout.samples:
        payload_name = f"slot_{sample.slot_index:06d}.bin"
        payload_path = cache_dir / payload_name
        payload_size = 0
        state = "placeholder"
        reason = ""
        if payload_path.is_file():
            payload_size = max(0, int(payload_path.stat().st_size))
            if 0 < payload_size <= sample.sample_size:
                state = "ready"
            elif payload_size == 0:
                state = "failed"
                reason = "payload-empty"
            else:
                state = "failed"
                reason = "payload-oversize"
        else:
            old = existing_slots.get(sample.slot_index, {})
            old_state = str(old.get("state") or "")
            if old_state in {"building", "failed"}:
                state = old_state
                reason = str(old.get("reason") or "")
        slots.append(
            Vmp4SlotCacheSlot(
                index=sample.slot_index,
                state=state,
                payload_path=payload_path if state == "ready" else None,
                payload_size=payload_size,
                reason=reason,
            )
        )
        manifest_slots.append(
            {
                "index": sample.slot_index,
                "state": state,
                "payload": payload_name,
                "payload_size": payload_size,
                "sample_size": sample.sample_size,
                "source_sample_index": sample.source_sample_index,
                "source_size": sample.source_size,
                "start_time_sec": sample.start_time_sec,
                "reason": reason,
            }
        )

    manifest = {
        "schema": _SLOT_CACHE_SCHEMA,
        "digest": digest,
        "source": str(layout.source_path),
        "source_stat": _source_stat_dict(layout.source_path),
        "output_mode": str(output_mode or "green"),
        "fps": float(fps or 0.0),
        "gop_frames": int(gop_frames or 0),
        "total_size": layout.total_size,
        "slot_count": layout.slot_count,
        "slot_size": layout.slot_size,
        "slot_stride": layout.slot_stride,
        "slot_gap_size": layout.slot_gap_size,
        "slot_duration_sec": layout.slot_duration_sec,
        "mdat_payload_start": layout.mdat_payload_start,
        "mdat_payload_size": layout.mdat_payload_size,
        "moov_size": layout.moov_size,
        "source_codec_name": layout.source_codec_name,
        "codec_name": layout.codec_name,
        "nal_length_size": layout.nal_length_size,
        "output_stsd_sha256": layout.output_stsd_sha256,
        "placeholder_payload_size": len(layout.placeholder_payload),
        "updated_at": now,
        "slots": manifest_slots,
    }
    if not _manifest_equivalent(existing, manifest):
        _write_manifest_atomic(manifest_path, manifest)
    return Vmp4SlotCacheStatus(
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        digest=digest,
        slots=tuple(slots),
    )


def write_vmp4_slot_placeholder_payload(
    layout: PassthroughVmp4SlotLayout,
    slot_index: int,
    target: Path,
) -> int:
    if slot_index < 0 or slot_index >= len(layout.samples):
        raise Vmp4SlotLayoutError("slot-index-out-of-range")
    sample = layout.samples[slot_index]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = layout.placeholder_payload[:sample.sample_size]
    _write_bytes_atomic(target, payload)
    return len(payload)


def write_vmp4_slot_source_payload(
    layout: PassthroughVmp4SlotLayout,
    slot_index: int,
    target: Path,
) -> int:
    return write_vmp4_slot_placeholder_payload(layout, slot_index, target)


def write_vmp4_slot_hevc_annexb_payload(
    layout: PassthroughVmp4SlotLayout,
    slot_index: int,
    target: Path,
    bitstream: bytes | bytearray | memoryview,
) -> int:
    if slot_index < 0 or slot_index >= len(layout.samples):
        raise Vmp4SlotLayoutError("slot-index-out-of-range")
    sample = layout.samples[slot_index]
    payload = hevc_annexb_to_length_prefixed_sample(bitstream, nal_length_size=layout.nal_length_size)
    if len(payload) > sample.sample_size:
        raise Vmp4SlotLayoutError("slot-payload-oversize")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(target, payload)
    return len(payload)


def hevc_annexb_to_length_prefixed_sample(
    bitstream: bytes | bytearray | memoryview,
    *,
    nal_length_size: int = 4,
) -> bytes:
    length_size = min(4, max(1, int(nal_length_size or 4)))
    out = bytearray()
    for nal in iter_annexb_nal_units(bitstream):
        if len(nal) >= 1 << (8 * length_size):
            raise Vmp4SlotLayoutError("nal-size-exceeds-length-prefix")
        out += int(len(nal)).to_bytes(length_size, "big")
        out += nal
    if not out:
        raise Vmp4SlotLayoutError("annexb-nal-missing")
    return bytes(out)


def split_hevc_annexb_access_units(
    bitstream: bytes | bytearray | memoryview,
) -> list[bytes]:
    """Split an HEVC Annex-B stream into per-frame access units.

    Used to cut a whole-GOP encoder output into one length-prefix-able payload
    per frame for the frame-level VMP4 layout. Each access unit is the leading
    non-VCL NALs (VPS/SPS/PPS/SEI/AUD) followed by exactly one VCL NAL; a new AU
    starts at each VCL NAL. This matches single-slice-per-frame NVENC output
    (HEVC_BF=0), where one VCL NAL == one frame.
    """
    units: list[bytearray] = []
    pending = bytearray()
    have_vcl = False
    for nal in iter_annexb_nal_units(bitstream):
        nal_type = (nal[0] >> 1) & 0x3F
        is_vcl = nal_type <= 31
        if is_vcl and have_vcl:
            units.append(pending)
            pending = bytearray()
            have_vcl = False
        # Re-emit each NAL with a 4-byte Annex-B start code.
        pending += b"\x00\x00\x00\x01"
        pending += nal
        if is_vcl:
            have_vcl = True
    if pending and have_vcl:
        units.append(pending)
    elif pending and units:
        # Trailing non-VCL NALs (rare): attach to the last access unit.
        units[-1] += pending
    return [bytes(u) for u in units]


def iter_annexb_nal_units(bitstream: bytes | bytearray | memoryview) -> Iterator[bytes]:
    data = bytes(bitstream)
    starts = list(_annexb_start_codes(data))
    if not starts:
        return
    for index, (prefix_start, payload_start) in enumerate(starts):
        next_start = starts[index + 1][0] if index + 1 < len(starts) else len(data)
        nal = data[payload_start:next_start].rstrip(b"\x00")
        if nal:
            yield nal


def _annexb_start_codes(data: bytes) -> Iterator[tuple[int, int]]:
    """Yield (prefix_start, payload_start) for every Annex-B start code.

    Scanning byte by byte in Python cost ~15ms on a 42 KiB frame, and the frames
    path scans every frame twice (splitting access units, then length-prefixing),
    which measured as a third of the whole encode pipeline's throughput. Let
    ``bytes.find`` do the search in C instead: a 4-byte start code is just a
    3-byte one preceded by a zero, so one scan finds both forms.
    """
    i = data.find(b"\x00\x00\x01")
    while i >= 0:
        yield (i - 1 if i > 0 and data[i - 1] == 0 else i), i + 3
        i = data.find(b"\x00\x00\x01", i + 3)


def update_vmp4_slot_manifest_slot_state(
    manifest_path: Path,
    slot_index: int,
    state: str,
    *,
    payload_size: int | None = None,
    reason: str = "",
) -> None:
    manifest = _read_manifest(Path(manifest_path))
    slots = manifest.get("slots")
    if not isinstance(slots, list):
        return
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        try:
            current_index = int(slot.get("index"))
        except (TypeError, ValueError):
            continue
        if current_index != int(slot_index):
            continue
        slot["state"] = str(state)
        slot["reason"] = str(reason or "")
        if payload_size is not None:
            slot["payload_size"] = max(0, int(payload_size))
        manifest["updated_at"] = time.time()
        _write_manifest_atomic(Path(manifest_path), manifest)
        return


def _select_slot_samples(samples: Sequence[MediaSample], slot_count: int, slot_sec: float) -> tuple[MediaSample, ...]:
    ordered = sorted(samples, key=lambda sample: (float(sample.time_seconds), int(sample.index)))
    selected: list[MediaSample] = []
    cursor = 0
    for slot_index in range(slot_count):
        slot_time = slot_index * slot_sec
        while cursor + 1 < len(ordered) and float(ordered[cursor + 1].time_seconds) <= slot_time + 1e-6:
            cursor += 1
        selected.append(ordered[cursor])
    return tuple(selected)


def _slot_sample_durations(slot_count: int, slot_sec: float, duration: float, timescale: int) -> tuple[int, ...]:
    durations: list[int] = []
    prev_tick = 0
    for slot_index in range(slot_count):
        if slot_index == slot_count - 1:
            end_sec = duration
        else:
            end_sec = min(duration, (slot_index + 1) * slot_sec)
        end_tick = int(round(end_sec * timescale))
        delta = end_tick - prev_tick
        durations.append(max(1, delta))
        prev_tick += max(1, delta)
    return tuple(durations)


def _extended_mdat_header(payload_size: int) -> bytes:
    return _u32(1) + b"mdat" + _u64(max(0, int(payload_size)) + 16)


def _probe_nal_length_size(source_moov: bytes) -> int:
    hvc = _find_box_payload_by_type(source_moov, b"hvcC")
    if hvc is not None and len(hvc) >= 22:
        return (hvc[21] & 0x03) + 1
    avc = _find_box_payload_by_type(source_moov, b"avcC")
    if avc is not None and len(avc) >= 5:
        return (avc[4] & 0x03) + 1
    return 4


def _source_video_stsd(source_moov: bytes) -> bytes:
    video_track = _find_track(source_moov, "video").trak
    mdia = _require_child(source_moov, video_track, b"mdia")
    minf = _require_child(source_moov, mdia, b"minf")
    stbl = _require_child(source_moov, minf, b"stbl")
    stsd = _require_child(source_moov, stbl, b"stsd")
    return _box_bytes(source_moov, stsd)


def _stsd_codec_name(stsd: bytes) -> str:
    if len(stsd) < 24 or stsd[4:8] != b"stsd":
        return ""
    first_entry = 16
    if first_entry + 8 > len(stsd):
        return ""
    sample_entry_type = bytes(stsd[first_entry + 4:first_entry + 8])
    return {
        b"avc1": "h264",
        b"avc3": "h264",
        b"hvc1": "hevc",
        b"hev1": "hevc",
    }.get(sample_entry_type, sample_entry_type.decode("latin1", "replace")).lower()


def _normalize_output_codec(value: str) -> str:
    codec = str(value or "").strip().lower()
    return {
        "h265": "hevc",
        "hvc1": "hevc",
        "hev1": "hevc",
        "avc1": "h264",
    }.get(codec, codec)


def _patch_tkhd_duration_and_size(tkhd: bytes, duration: int, width: int, height: int) -> bytes:
    patched = bytearray(_patch_tkhd_duration(tkhd, duration))
    if width > 0 and height > 0 and len(patched) >= 16:
        # ISO BMFF tkhd stores width/height as 16.16 fixed-point values in the
        # last two 32-bit fields. Synthetic tests may use short tkhd boxes, so
        # only patch real boxes that have those fields.
        patched[-8:-4] = _u32(max(0, int(width)) << 16)
        patched[-4:] = _u32(max(0, int(height)) << 16)
    return bytes(patched)


def _find_box_payload_by_type(data: bytes, box_type: bytes) -> bytes | None:
    needle = bytes(box_type)
    offset = 0
    while True:
        pos = data.find(needle, offset)
        if pos < 4:
            return None
        size_pos = pos - 4
        raw_size = int.from_bytes(data[size_pos:pos], "big")
        header_size = 8
        payload_start = pos + 4
        if raw_size == 1:
            if pos + 12 > len(data):
                offset = pos + 4
                continue
            raw_size = int.from_bytes(data[pos + 4:pos + 12], "big")
            header_size = 16
            payload_start = pos + 12
        if raw_size >= header_size and size_pos + raw_size <= len(data):
            return data[payload_start:size_pos + raw_size]
        offset = pos + 4


def _make_stss_all(sample_count: int) -> bytes:
    payload = bytearray()
    payload += _u32(max(0, int(sample_count)))
    for index in range(max(0, int(sample_count))):
        payload += _u32(index + 1)
    return _full_box("stss", 0, 0, bytes(payload))


def _rewrite_slot_stbl(
    source_moov: bytes,
    stbl,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    output_stsd: bytes,
) -> bytes:
    return _box(
        "stbl",
        bytes(output_stsd)
        + _make_stts_from_durations(durations)
        + _make_stsc_one_sample_per_chunk()
        + _make_stsz(sizes)
        + _make_co64(offsets)
        + _make_stss_all(len(sizes)),
    )


def _rewrite_slot_minf(
    source_moov: bytes,
    minf,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    output_stsd: bytes,
) -> bytes:
    stbl = _require_child(source_moov, minf, b"stbl")
    children = [
        _rewrite_slot_stbl(source_moov, child, offsets, sizes, durations, output_stsd)
        if child.start == stbl.start
        else _box_bytes(source_moov, child)
        for child in _children(source_moov, minf)
    ]
    return _box("minf", b"".join(children))


def _rewrite_slot_mdia(
    source_moov: bytes,
    mdia,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    media_duration: int,
    output_stsd: bytes,
) -> bytes:
    minf = _require_child(source_moov, mdia, b"minf")
    children = []
    for child in _children(source_moov, mdia):
        if child.start == minf.start:
            children.append(_rewrite_slot_minf(source_moov, child, offsets, sizes, durations, output_stsd))
        elif child.type == b"mdhd":
            children.append(_patch_mdhd_duration(_box_bytes(source_moov, child), media_duration))
        else:
            children.append(_box_bytes(source_moov, child))
    return _box("mdia", b"".join(children))


def _rewrite_slot_trak(
    source_moov: bytes,
    trak,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    media_duration: int,
    movie_duration: int,
    output_stsd: bytes,
    output_width: int,
    output_height: int,
) -> bytes:
    mdia = _require_child(source_moov, trak, b"mdia")
    children = []
    for child in _children(source_moov, trak):
        if child.start == mdia.start:
            children.append(_rewrite_slot_mdia(source_moov, child, offsets, sizes, durations, media_duration, output_stsd))
        elif child.type == b"tkhd":
            children.append(_patch_tkhd_duration_and_size(_box_bytes(source_moov, child), movie_duration, output_width, output_height))
        elif child.type == b"edts":
            continue
        else:
            children.append(_box_bytes(source_moov, child))
    return _box("trak", b"".join(children))


def _build_slot_moov(
    source_moov: bytes,
    *,
    output_stsd: bytes,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    media_duration: int,
    movie_duration_sec: float,
    output_width: int = 0,
    output_height: int = 0,
) -> bytes:
    moov_ref = next(_iter_boxes(source_moov, 0, len(source_moov)), None)
    if moov_ref is None or moov_ref.type != b"moov":
        raise Vmp4SlotLayoutError("source-moov-invalid")
    mvhd = _require_child(source_moov, moov_ref, b"mvhd")
    movie_timescale = _mvhd_timescale(_box_bytes(source_moov, mvhd))
    movie_duration = int(round(max(0.0, float(movie_duration_sec)) * movie_timescale))
    video_trak = _find_track(source_moov, "video").trak
    video_trak_data = _rewrite_slot_trak(
        source_moov,
        video_trak,
        offsets,
        sizes,
        durations,
        media_duration,
        movie_duration,
        output_stsd,
        output_width,
        output_height,
    )
    video_id = _track_id(video_trak_data)
    children = []
    inserted_video = False
    for child in _children(source_moov, moov_ref):
        if child.type == b"mvhd":
            children.append(
                _patch_mvhd_duration_and_next_track_id(
                    _box_bytes(source_moov, child),
                    movie_duration,
                    video_id + 1,
                )
            )
        elif child.type == b"trak":
            if child.start == video_trak.start and not inserted_video:
                children.append(video_trak_data)
                inserted_video = True
            continue
        else:
            children.append(_box_bytes(source_moov, child))
    if not inserted_video:
        children.append(video_trak_data)
    return _box("moov", b"".join(children))


def _iter_zero_bytes(size: int, chunk_size: int) -> Iterator[bytes]:
    remaining = max(0, int(size))
    if remaining <= 0:
        return
    pad = b"\x00" * min(max(1, int(chunk_size)), remaining)
    while remaining > 0:
        chunk = pad[: min(len(pad), remaining)]
        remaining -= len(chunk)
        yield chunk


def _iter_file_range(path: Path, offset: int, size: int, chunk_size: int) -> Iterator[bytes]:
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


def _iter_filler_range(
    codec_name: str,
    nal_length_size: int,
    size: int,
    start: int,
    end: int,
    chunk_size: int,
) -> Iterator[bytes]:
    filler_size = max(0, int(size))
    cursor = max(0, int(start))
    stop = min(max(0, int(end)), filler_size)
    if cursor >= stop:
        return
    length_size = min(4, max(1, int(nal_length_size or 4)))
    codec = str(codec_name or "").lower()
    nal_header = b"\x4c\x01" if codec in {"hevc", "h265", "hvc1", "hev1"} else b"\x0c"
    min_size = length_size + len(nal_header) + 1
    if filler_size < min_size:
        yield from _iter_zero_bytes(stop - cursor, chunk_size)
        return
    nal_size = filler_size - length_size
    prefix = nal_size.to_bytes(length_size, "big", signed=False)
    regions = (
        (0, prefix),
        (length_size, nal_header),
    )
    for region_start, data in regions:
        region_end = region_start + len(data)
        overlap_start = max(cursor, region_start)
        overlap_end = min(stop, region_end)
        if overlap_start < overlap_end:
            yield data[overlap_start - region_start:overlap_end - region_start]
        if cursor < overlap_end:
            cursor = overlap_end

    body_start = length_size + len(nal_header)
    trailing_start = filler_size - 1
    ff_start = max(cursor, body_start)
    ff_end = min(stop, trailing_start)
    if ff_start < ff_end:
        pad = b"\xff" * min(max(1, int(chunk_size)), ff_end - ff_start)
        remaining = ff_end - ff_start
        while remaining > 0:
            chunk = pad[: min(len(pad), remaining)]
            remaining -= len(chunk)
            yield chunk
    if stop > trailing_start and cursor <= trailing_start:
        yield b"\x80"


def _slot_cache_digest(
    layout: PassthroughVmp4SlotLayout,
    *,
    output_mode: str,
    fps: float,
    gop_frames: int,
) -> str:
    st = _source_stat_dict(layout.source_path)
    raw = {
        "schema": _SLOT_CACHE_SCHEMA,
        "source": str(layout.source_path.resolve()),
        "source_stat": st,
        "output_mode": str(output_mode or "green").lower(),
        "fps": round(float(fps or 0.0), 6),
        "gop_frames": int(gop_frames or 0),
        "total_size": layout.total_size,
        "slot_count": layout.slot_count,
        "slot_size": layout.slot_size,
        "slot_stride": layout.slot_stride,
        "slot_gap_size": layout.slot_gap_size,
        "slot_duration_sec": round(float(layout.slot_duration_sec), 6),
        "source_codec_name": layout.source_codec_name,
        "codec_name": layout.codec_name,
        "nal_length_size": layout.nal_length_size,
        "output_stsd_sha256": layout.output_stsd_sha256,
        "placeholder_payload_size": len(layout.placeholder_payload),
    }
    data = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def _source_stat_dict(path: Path) -> dict[str, int]:
    try:
        st = Path(path).stat()
        return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except OSError:
        return {"size": 0, "mtime_ns": 0}


def _read_manifest(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _manifest_equivalent(existing: dict, manifest: dict) -> bool:
    if not isinstance(existing, dict):
        return False
    if existing.get("digest") != manifest.get("digest"):
        return False

    def without_updated_at(value: dict) -> dict:
        return {key: item for key, item in value.items() if key != "updated_at"}

    return without_updated_at(existing) == without_updated_at(manifest)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    tmp_path: Path | None = None
    try:
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        tmp_path = Path(raw_tmp)
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(data)
        _replace_atomic_with_retry(tmp_path, target)
        tmp_path = None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _write_manifest_atomic(path: Path, manifest: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _manifest_write_lock(target)
    fd = -1
    tmp_path: Path | None = None
    with lock:
        try:
            fd, raw_tmp = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=str(target.parent),
            )
            tmp_path = Path(raw_tmp)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fd = -1
                fh.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            _replace_atomic_with_retry(tmp_path, target)
            tmp_path = None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _manifest_write_lock(path: Path) -> threading.RLock:
    key = str(Path(path).resolve()).lower()
    with _manifest_write_locks_guard:
        lock = _manifest_write_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _manifest_write_locks[key] = lock
        return lock


def _replace_atomic_with_retry(source: Path, target: Path) -> None:
    attempts = 20 if os.name == "nt" else 1
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(0.005 * (attempt + 1))
    if last_error is not None:
        raise last_error
