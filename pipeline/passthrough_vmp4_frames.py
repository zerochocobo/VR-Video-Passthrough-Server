"""Frame-level VMP4 layout for /passthrough_seek (real fps, plain MP4).

The 1-sample-per-slot slot layout plays like a slideshow because each GOP is one
MP4 sample with a multi-second ``stts``. This layout keeps the same plain
``ftyp + moov + mdat`` / single-moov / byte-stable contract but stores EVERY
frame as its own sample with ``stts = 1/fps`` -> real frame rate.

Variable frame sizes would break fixed ``co64`` offsets, so each frame gets a
FIXED byte budget (a larger one for GOP-head IDR frames, a smaller one for the
rest). The real per-frame HEVC bytes are padded up to that budget with one filler
NAL, exactly like the slot backend but at frame granularity, so all offsets are
known before encoding. ``stss`` marks only the real GOP heads, so players seek to
true keyframes (avoids decoding a P-frame as a sync sample -> green frames).

Because realtime HEVC at 4K/8K cannot fit the source byte budget, this layout
drops the ``virtual_size == source_size`` constraint and declares ``total`` from
the per-frame budgets (a realtime-bitrate estimate), the same spirit as the live
path's ``live_total_est``.

The validated offline prototype is ``tools/vmp4_frame_filler_proof.py``.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.si_virtual_mp4 import (
    _box,
    _rewrite_audio_trak_samples,
    _box_bytes,
    _children,
    _find_track,
    _iter_boxes,
    _make_co64,
    _make_stsz,
    _make_stts_from_durations,
    MediaSampleTable,
    _mvhd_timescale,
    _patch_mdhd_duration,
    _patch_mvhd_duration_and_next_track_id,
    _patch_tkhd_duration,
    _require_child,
    _track_id,
    _u32,
    read_media_sample_table,
)
from pipeline.passthrough_vmp4_slot import (
    Vmp4SlotLayoutError,
    _extended_mdat_header,
    _iter_file_range,
    _iter_filler_range,
    _iter_zero_bytes,
    _normalize_output_codec,
    _probe_nal_length_size,
    _source_video_stsd,
    _stsd_codec_name,
)

_FRAMES_LAYOUT_SCHEMA = 1


@dataclass(frozen=True)
class Vmp4Frame:
    index: int            # global frame index (presentation order)
    gop_index: int        # GOP / slot this frame belongs to
    frame_in_gop: int     # 0 == GOP head (IDR)
    offset: int           # absolute byte offset of the sample in the virtual file
    budget: int           # fixed stsz sample size for this frame
    keyframe: bool
    start_time_sec: float


@dataclass(frozen=True)
class Vmp4AudioSample:
    """One source audio sample, copied through into the virtual file.

    Audio is never re-encoded here, so its bytes are read straight out of the
    source at ``source_offset``; only the position it occupies in the virtual
    file (``offset``) is ours.
    """
    index: int
    gop_index: int
    offset: int
    source_offset: int
    size: int


@dataclass(frozen=True)
class PassthroughVmp4FramesLayout:
    source_path: Path
    init: bytes
    total_size: int
    mdat_payload_start: int
    mdat_payload_size: int
    frame_count: int
    gop_frames: int
    slot_count: int
    fps: float
    idr_budget: int
    p_budget: int
    codec_name: str
    nal_length_size: int
    output_stsd_sha256: str
    placeholder_payload: bytes
    frames: tuple[Vmp4Frame, ...]
    moov_size: int
    media_timescale: int
    # True when per-frame budgets came from the source's own bit distribution
    # (pipeline/source_budget_plan.py) instead of the flat idr/p constants.
    source_budget: bool = False
    # Source audio copied through, interleaved one block per GOP. Empty when the
    # source has no audio track.
    audio_samples: tuple[Vmp4AudioSample, ...] = ()
    audio_bytes: int = 0
    # Byte span [start, end) of each GOP's interleaved block, for locating an
    # arbitrary offset (which may land in audio) on a GOP.
    gop_spans: tuple[tuple[int, int], ...] = ()

    def gop_frame_indices(self, gop_index: int) -> tuple[int, ...]:
        return tuple(f.index for f in self.frames if f.gop_index == gop_index)


def build_passthrough_vmp4_frames_layout(
    path: Path,
    *,
    duration_sec: float,
    fps: float,
    gop_frames: int,
    idr_budget: int,
    p_budget: int,
    output_stsd: bytes | None = None,
    output_codec_name: str = "hevc",
    output_width: int = 0,
    output_height: int = 0,
    placeholder_payload: bytes = b"",
    frame_budgets: Sequence[int] | None = None,
    audio_table: MediaSampleTable | None = None,
) -> PassthroughVmp4FramesLayout:
    source = Path(path)
    duration = max(0.0, float(duration_sec or 0.0))
    if duration <= 0.0:
        raise Vmp4SlotLayoutError("duration-missing")
    out_fps = float(fps or 0.0)
    if out_fps <= 0.0:
        raise Vmp4SlotLayoutError("fps-missing")
    gop = max(1, int(gop_frames or 1))
    idr_b = max(1024, int(idr_budget))
    p_b = max(1024, int(p_budget))

    frame_count = max(1, int(round(duration * out_fps)))
    slot_count = (frame_count + gop - 1) // gop

    supplied_budgets: list[int] | None = None
    if frame_budgets is not None:
        supplied_budgets = [max(1024, int(b)) for b in frame_budgets]
        if len(supplied_budgets) != frame_count:
            raise Vmp4SlotLayoutError(
                f"frames-budget-count:{len(supplied_budgets)}/{frame_count}"
            )

    try:
        from pipeline.si_virtual_mp4 import _read_top_level_box

        ftyp = _read_top_level_box(source, b"ftyp")
    except Exception:
        ftyp = _box("ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")
    try:
        from pipeline.si_virtual_mp4 import _read_top_level_box

        source_moov = _read_top_level_box(source, b"moov")
    except Exception as exc:
        raise Vmp4SlotLayoutError(f"source-moov-error:{type(exc).__name__}") from exc

    source_stsd = _source_video_stsd(source_moov)
    source_codec_name = _stsd_codec_name(source_stsd)
    out_stsd = bytes(output_stsd or b"")
    if not out_stsd:
        if source_codec_name != "hevc":
            raise Vmp4SlotLayoutError("frames-output-stsd-missing")
        out_stsd = source_stsd
    codec_name = _normalize_output_codec(output_codec_name) or _stsd_codec_name(out_stsd) or "hevc"
    nal_length_size = _probe_nal_length_size(out_stsd)
    placeholder = bytes(placeholder_payload or b"")

    # Media timescale chosen so each frame is exactly one tick of 1/fps.
    media_timescale = max(1, int(round(out_fps * 1000.0)))
    delta = max(1, int(round(media_timescale / out_fps)))
    media_duration = delta * frame_count
    movie_duration_sec = frame_count / out_fps

    budgets = supplied_budgets if supplied_budgets is not None else [
        idr_b if (i % gop == 0) else p_b
        for i in range(frame_count)
    ]
    durations = [delta] * frame_count
    keyframe_numbers = [i + 1 for i in range(frame_count) if i % gop == 0]

    # Audio is copied through untouched, so its sizes and durations are the
    # source's; only its position in our file is ours. Assign each sample to the
    # GOP its presentation time falls in, so the file interleaves one audio block
    # per GOP instead of parking the whole track at one end.
    gop_sec = gop / out_fps
    audio_sizes: list[int] = []
    audio_durations: list[int] = []
    audio_src_offsets: list[int] = []
    audio_gop_of: list[int] = []
    if audio_table is not None and audio_table.samples:
        a_ts = audio_table.time_base.denominator
        prev_dts: int | None = None
        for s in audio_table.samples:
            audio_sizes.append(int(s.size))
            audio_src_offsets.append(int(s.source_offset))
            audio_gop_of.append(min(slot_count - 1, max(0, int(float(s.time_seconds) / gop_sec))))
            if prev_dts is not None and s.dts is not None:
                audio_durations.append(max(1, int(s.dts) - prev_dts))
            elif audio_durations:
                audio_durations.append(audio_durations[-1])
            else:
                audio_durations.append(1024)
            prev_dts = int(s.dts) if s.dts is not None else prev_dts
        if audio_durations:
            audio_durations = audio_durations[1:] + [audio_durations[-1]]
        audio_media_duration = sum(audio_durations)
    else:
        a_ts = 0
        audio_media_duration = 0

    # Number of audio samples belonging to each GOP, in order.
    audio_per_gop: list[list[int]] = [[] for _ in range(slot_count)]
    for ai, gi in enumerate(audio_gop_of):
        audio_per_gop[gi].append(ai)

    def place(moov_len: int) -> tuple[list[int], list[int], list[tuple[int, int]], int]:
        """Lay frames and audio out interleaved; returns offsets and GOP spans."""
        start = len(ftyp) + moov_len + mdat_header_len
        v_off = [0] * frame_count
        a_off = [0] * len(audio_sizes)
        spans: list[tuple[int, int]] = []
        cur = start
        for gi in range(slot_count):
            span_start = cur
            first = gi * gop
            for i in range(first, min(first + gop, frame_count)):
                v_off[i] = cur
                cur += budgets[i]
            for ai in audio_per_gop[gi]:
                a_off[ai] = cur
                cur += audio_sizes[ai]
            spans.append((span_start, cur))
        return v_off, a_off, spans, cur - start

    mdat_payload_size = sum(budgets) + sum(audio_sizes)
    mdat_header = _extended_mdat_header(mdat_payload_size)
    mdat_header_len = len(mdat_header)

    # Build moov once to learn its size, then place offsets and rebuild. co64 is
    # 64-bit fixed-width and stsz is uncompressed, so this settles immediately.
    build_kwargs = dict(
        output_stsd=out_stsd,
        sizes=budgets,
        durations=durations,
        keyframe_numbers=keyframe_numbers,
        media_timescale=media_timescale,
        media_duration=media_duration,
        movie_duration_sec=movie_duration_sec,
        output_width=output_width,
        output_height=output_height,
        audio_sizes=audio_sizes,
        audio_durations=audio_durations,
        audio_media_duration=audio_media_duration,
        audio_timescale=a_ts,
    )
    moov = _build_frames_moov(
        source_moov, offsets=[0] * frame_count, audio_offsets=[0] * len(audio_sizes), **build_kwargs
    )
    converged = False
    offsets: list[int] = []
    audio_offsets: list[int] = []
    gop_spans: list[tuple[int, int]] = []
    for _ in range(4):
        offsets, audio_offsets, gop_spans, _span = place(len(moov))
        new_moov = _build_frames_moov(
            source_moov, offsets=offsets, audio_offsets=audio_offsets, **build_kwargs
        )
        if len(new_moov) == len(moov):
            moov = new_moov
            converged = True
            break
        moov = new_moov
    if not converged:
        raise Vmp4SlotLayoutError("frames-moov-offsets-not-converged")
    offsets, audio_offsets, gop_spans, _span = place(len(moov))

    mdat_payload_start = len(ftyp) + len(moov) + len(mdat_header)
    init = ftyp + moov + mdat_header
    frames = [
        Vmp4Frame(
            index=i,
            gop_index=i // gop,
            frame_in_gop=i % gop,
            offset=offsets[i],
            budget=budgets[i],
            keyframe=(i % gop == 0),
            start_time_sec=min(duration, i / out_fps),
        )
        for i in range(frame_count)
    ]
    audio_samples = tuple(
        Vmp4AudioSample(
            index=ai,
            gop_index=audio_gop_of[ai],
            offset=audio_offsets[ai],
            source_offset=audio_src_offsets[ai],
            size=audio_sizes[ai],
        )
        for ai in range(len(audio_sizes))
    )
    total_size = mdat_payload_start + mdat_payload_size

    return PassthroughVmp4FramesLayout(
        source_path=source,
        init=init,
        total_size=total_size,
        mdat_payload_start=mdat_payload_start,
        mdat_payload_size=mdat_payload_size,
        frame_count=frame_count,
        gop_frames=gop,
        slot_count=slot_count,
        fps=out_fps,
        idr_budget=idr_b,
        p_budget=p_b,
        codec_name=codec_name,
        nal_length_size=nal_length_size,
        output_stsd_sha256=hashlib.sha256(out_stsd).hexdigest(),
        placeholder_payload=placeholder,
        frames=tuple(frames),
        moov_size=len(moov),
        media_timescale=media_timescale,
        source_budget=supplied_budgets is not None,
        audio_samples=audio_samples,
        audio_bytes=sum(audio_sizes),
        gop_spans=tuple(gop_spans),
    )


def _first_region_ending_after(regions, pos: int, size_of) -> int:
    """Index of the first region whose end is past ``pos`` (regions are ordered)."""
    lo, hi = 0, len(regions)
    while lo < hi:
        mid = (lo + hi) // 2
        if regions[mid].offset + size_of(regions[mid]) <= pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


def vmp4_gop_for_offset(layout: PassthroughVmp4FramesLayout, offset: int) -> int | None:
    """GOP an offset belongs to, including offsets inside an audio block."""
    pos = int(offset)
    spans = layout.gop_spans
    if not spans or pos < layout.mdat_payload_start:
        return None
    lo, hi = 0, len(spans) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end = spans[mid]
        if pos < start:
            hi = mid - 1
        elif pos >= end:
            lo = mid + 1
        else:
            return mid
    return None


def vmp4_frame_index_for_offset(layout: PassthroughVmp4FramesLayout, offset: int) -> int | None:
    """Frame an mdat offset belongs to, or the next one when it belongs to none.

    Audio samples are interleaved between the GOPs, so an offset can legitimately
    land in a gap that no frame covers. Returning None there made every caller
    treat a mid-title offset as "before the first frame": the seek path read that
    as GOP 0 and restarted the encoder at 0:00 - a player reopening a few MB
    further along sent the run back to the start of the film, stranding the reply
    it was feeding. Report the first frame ending after ``pos`` instead, which is
    the frame such an offset is actually reading towards.

    None still means the offset is outside the frame region entirely: before the
    mdat payload, or past the last frame.
    """
    pos = int(offset)
    if pos < layout.mdat_payload_start or not layout.frames:
        return None
    # Binary search the frame whose [offset, offset+budget) contains pos. On a
    # miss `lo` lands on the first frame ending after pos, which is what an
    # offset inside an audio block wants.
    lo, hi = 0, len(layout.frames) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        fr = layout.frames[mid]
        if pos < fr.offset:
            hi = mid - 1
        elif pos >= fr.offset + fr.budget:
            lo = mid + 1
        else:
            return mid
    return lo if lo < len(layout.frames) else None


def iter_vmp4_frames_range(
    layout: PassthroughVmp4FramesLayout,
    start: int,
    end_inclusive: int,
    *,
    chunk_size: int = 64 * 1024,
    payload_paths: Mapping[int, Path] | None = None,
    wait_for_frame=None,
) -> Iterator[bytes]:
    """Serve a byte range of the virtual file.

    ``wait_for_frame(index) -> Path | None`` is consulted for a frame that is not
    in ``payload_paths`` yet. It lets the body block briefly on the encoder
    instead of substituting filler, which both keeps the picture real and paces
    the reader to the rate frames are actually produced - players read far faster
    than realtime and otherwise overrun the encoder within seconds of a seek.
    The declared length never changes either way: a frame occupies its budget
    whether it holds real bytes or filler.
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

    chunk = max(1, int(chunk_size))
    frames = layout.frames
    audio = layout.audio_samples
    vi = _first_region_ending_after(frames, cursor, lambda f: f.budget)
    ai = _first_region_ending_after(audio, cursor, lambda a: a.size)

    # Video frames and audio samples are interleaved in the mdat, so walk both in
    # file order.
    while cursor <= stop:
        nv = frames[vi] if vi < len(frames) else None
        na = audio[ai] if ai < len(audio) else None
        if nv is None and na is None:
            break
        is_video = na is None or (nv is not None and nv.offset <= na.offset)
        region = nv if is_video else na
        region_size = region.budget if is_video else region.size
        region_end = region.offset + region_size
        if region.offset > stop:
            break
        if cursor < region.offset:
            gap_end = min(stop + 1, region.offset)
            yield from _iter_zero_bytes(gap_end - cursor, chunk)
            cursor = gap_end
            if cursor > stop:
                return
        overlap_start = max(cursor, region.offset)
        overlap_end = min(stop + 1, region_end)
        inside = overlap_start - region.offset

        if not is_video:
            # Copied through from the source; nothing to encode or pad.
            read = overlap_end - overlap_start
            if read > 0:
                yield from _iter_file_range(
                    layout.source_path, region.source_offset + inside, read, chunk
                )
            ai += 1
            cursor = max(cursor, overlap_end)
            continue

        fr = region
        payload_path = (
            Path(payload_paths[fr.index]) if payload_paths and fr.index in payload_paths else None
        )
        if payload_path is None and wait_for_frame is not None:
            waited = wait_for_frame(fr.index)
            if waited is not None:
                payload_path = Path(waited)
        emitted = 0
        if payload_path is not None:
            try:
                emitted = min(max(0, int(payload_path.stat().st_size)), fr.budget)
            except OSError:
                payload_path = None
                emitted = 0
        if payload_path is not None and emitted > 0:
            if inside < emitted:
                read = min(emitted - inside, overlap_end - overlap_start)
                yield from _iter_file_range(payload_path, inside, read, chunk)
                overlap_start += read
        else:
            placeholder = layout.placeholder_payload[: fr.budget]
            emitted = len(placeholder)
            if inside < emitted:
                read = min(emitted - inside, overlap_end - overlap_start)
                yield placeholder[inside : inside + read]
                overlap_start += read

        if overlap_start < overlap_end:
            filler_start = max(0, overlap_start - fr.offset - emitted)
            filler_end = max(0, overlap_end - fr.offset - emitted)
            yield from _iter_filler_range(
                layout.codec_name,
                layout.nal_length_size,
                max(0, fr.budget - emitted),
                filler_start,
                filler_end,
                chunk,
            )
        vi += 1
        cursor = max(cursor, overlap_end)

    if cursor <= stop:
        yield from _iter_zero_bytes(stop - cursor + 1, chunk)


def _make_stss(sample_numbers: Sequence[int]) -> bytes:
    from pipeline.si_virtual_mp4 import _full_box

    payload = bytearray()
    payload += _u32(len(sample_numbers))
    for number in sample_numbers:
        payload += _u32(int(number))
    return _full_box("stss", 0, 0, bytes(payload))


def _make_stsc_one_sample_per_chunk() -> bytes:
    from pipeline.si_virtual_mp4 import _full_box

    return _full_box("stsc", 0, 0, _u32(1) + _u32(1) + _u32(1) + _u32(1))


def _rewrite_frames_stbl(offsets, sizes, durations, keyframe_numbers, output_stsd) -> bytes:
    return _box(
        "stbl",
        bytes(output_stsd)
        + _make_stts_from_durations(durations)
        + _make_stsc_one_sample_per_chunk()
        + _make_stsz(sizes)
        + _make_co64(offsets)
        + _make_stss(keyframe_numbers),
    )


def _rewrite_frames_minf(source_moov, minf, offsets, sizes, durations, keyframe_numbers, output_stsd) -> bytes:
    stbl = _require_child(source_moov, minf, b"stbl")
    children = [
        _rewrite_frames_stbl(offsets, sizes, durations, keyframe_numbers, output_stsd)
        if child.start == stbl.start
        else _box_bytes(source_moov, child)
        for child in _children(source_moov, minf)
    ]
    return _box("minf", b"".join(children))


def _patch_mdhd_timescale_and_duration(mdhd: bytes, timescale: int, duration: int) -> bytes:
    # Set the media timescale AND duration so 1/fps stts deltas map to real fps.
    # _patch_mdhd_duration alone keeps the source timescale, which skews fps.
    patched = bytearray(_patch_mdhd_duration(mdhd, duration))
    if len(patched) < 9:
        return bytes(patched)
    version = patched[8]
    ts_off = 20 if version == 0 else 28
    if ts_off + 4 <= len(patched):
        patched[ts_off : ts_off + 4] = _u32(max(1, int(timescale)))
    return bytes(patched)


def _rewrite_frames_mdia(source_moov, mdia, offsets, sizes, durations, keyframe_numbers, media_timescale, media_duration, output_stsd) -> bytes:
    minf = _require_child(source_moov, mdia, b"minf")
    children = []
    for child in _children(source_moov, mdia):
        if child.start == minf.start:
            children.append(_rewrite_frames_minf(source_moov, child, offsets, sizes, durations, keyframe_numbers, output_stsd))
        elif child.type == b"mdhd":
            children.append(_patch_mdhd_timescale_and_duration(_box_bytes(source_moov, child), media_timescale, media_duration))
        else:
            children.append(_box_bytes(source_moov, child))
    return _box("mdia", b"".join(children))


def _patch_tkhd_size(tkhd: bytes, duration: int, width: int, height: int) -> bytes:
    patched = bytearray(_patch_tkhd_duration(tkhd, duration))
    if width > 0 and height > 0 and len(patched) >= 16:
        patched[-8:-4] = _u32(max(0, int(width)) << 16)
        patched[-4:] = _u32(max(0, int(height)) << 16)
    return bytes(patched)


def _rewrite_frames_trak(source_moov, trak, offsets, sizes, durations, keyframe_numbers, media_timescale, media_duration, movie_duration, output_stsd, output_width, output_height) -> bytes:
    mdia = _require_child(source_moov, trak, b"mdia")
    children = []
    for child in _children(source_moov, trak):
        if child.start == mdia.start:
            children.append(_rewrite_frames_mdia(source_moov, child, offsets, sizes, durations, keyframe_numbers, media_timescale, media_duration, output_stsd))
        elif child.type == b"tkhd":
            children.append(_patch_tkhd_size(_box_bytes(source_moov, child), movie_duration, output_width, output_height))
        elif child.type == b"edts":
            continue
        else:
            children.append(_box_bytes(source_moov, child))
    return _box("trak", b"".join(children))


def _build_frames_moov(
    source_moov: bytes,
    *,
    output_stsd: bytes,
    offsets: Sequence[int],
    sizes: Sequence[int],
    durations: Sequence[int],
    keyframe_numbers: Sequence[int],
    media_timescale: int,
    media_duration: int,
    movie_duration_sec: float,
    output_width: int = 0,
    output_height: int = 0,
    audio_offsets: Sequence[int] = (),
    audio_sizes: Sequence[int] = (),
    audio_durations: Sequence[int] = (),
    audio_media_duration: int = 0,
    audio_timescale: int = 0,
) -> bytes:
    moov_ref = next(_iter_boxes(source_moov, 0, len(source_moov)), None)
    if moov_ref is None or moov_ref.type != b"moov":
        raise Vmp4SlotLayoutError("source-moov-invalid")
    mvhd = _require_child(source_moov, moov_ref, b"mvhd")
    movie_timescale = _mvhd_timescale(_box_bytes(source_moov, mvhd))
    movie_duration = int(round(max(0.0, float(movie_duration_sec)) * movie_timescale))
    video_trak = _find_track(source_moov, "video").trak
    video_trak_data = _rewrite_frames_trak(
        source_moov, video_trak, offsets, sizes, durations, keyframe_numbers,
        media_timescale, media_duration, movie_duration, output_stsd, output_width, output_height,
    )
    video_id = _track_id(video_trak_data)

    # Audio is a straight copy of the source's samples, so its track keeps the
    # source stsd/timescale and only its chunk offsets are rewritten to where the
    # samples sit in our interleaved mdat.
    audio_trak_data = b""
    audio_id = video_id + 1
    if audio_sizes:
        try:
            audio_trak = _find_track(source_moov, "audio").trak
        except Exception:
            audio_trak = None
        if audio_trak is not None:
            audio_trak_data = _rewrite_audio_trak_samples(
                source_moov,
                audio_trak,
                audio_offsets,
                audio_sizes,
                audio_durations,
                media_duration=int(audio_media_duration),
                movie_duration=movie_duration,
            )
            audio_id = _track_id(audio_trak_data)

    next_track_id = max(video_id, audio_id) + 1
    children = []
    inserted = False
    for child in _children(source_moov, moov_ref):
        if child.type == b"mvhd":
            children.append(
                _patch_mvhd_duration_and_next_track_id(
                    _box_bytes(source_moov, child), movie_duration, next_track_id
                )
            )
        elif child.type == b"trak":
            if child.start == video_trak.start and not inserted:
                children.append(video_trak_data)
                if audio_trak_data:
                    children.append(audio_trak_data)
                inserted = True
            continue
        else:
            children.append(_box_bytes(source_moov, child))
    if not inserted:
        children.append(video_trak_data)
        if audio_trak_data:
            children.append(audio_trak_data)
    return _box("moov", b"".join(children))
