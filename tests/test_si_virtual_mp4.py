from __future__ import annotations

import queue
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pipeline.si_virtual_mp4 as siv
from pipeline.si_virtual_mp4 import (
    BoxSplittingSink,
    VirtualRegion,
    _build_moov,
    _copy_file_ranges_sequential_to_outputs,
    _copy_file_ranges_to_outputs,
    _find_child,
    _find_track,
    _plan_mix_segments,
    iter_virtual_range,
    read_media_sample_table,
    virtual_size,
)
from utils.si_filter import SIMixParams


def _box(box_type: str, payload: bytes) -> bytes:
    body = box_type.encode("latin1") + payload
    return (len(body) + 4).to_bytes(4, "big") + body


def _extended_box(box_type: str, payload: bytes) -> bytes:
    size = 16 + len(payload)
    return b"\x00\x00\x00\x01" + box_type.encode("latin1") + size.to_bytes(8, "big") + payload


def _full_box(box_type: str, payload: bytes, *, version: int = 0, flags: int = 0) -> bytes:
    return _box(box_type, bytes([version]) + int(flags).to_bytes(3, "big") + payload)


def _mdhd(timescale: int, duration: int) -> bytes:
    payload = (
        b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + int(timescale).to_bytes(4, "big")
        + int(duration).to_bytes(4, "big")
        + b"\x55\xc4\x00\x00"
    )
    return _full_box("mdhd", payload)


def _stsd_video(sample_entry_type: str = "hvc1") -> bytes:
    sample_entry = _box(sample_entry_type, b"\x00" * 8)
    return _full_box("stsd", (1).to_bytes(4, "big") + sample_entry)


def _stsd_audio(sample_entry_type: str = "mp4a") -> bytes:
    sample_entry = _box(sample_entry_type, b"\x00" * 8)
    return _full_box("stsd", (1).to_bytes(4, "big") + sample_entry)


def _stts(entries: list[tuple[int, int]]) -> bytes:
    payload = bytearray()
    payload += len(entries).to_bytes(4, "big")
    for count, delta in entries:
        payload += int(count).to_bytes(4, "big")
        payload += int(delta).to_bytes(4, "big")
    return _full_box("stts", bytes(payload))


def _ctts(entries: list[tuple[int, int]]) -> bytes:
    payload = bytearray()
    payload += len(entries).to_bytes(4, "big")
    for count, offset in entries:
        payload += int(count).to_bytes(4, "big")
        payload += int(offset).to_bytes(4, "big")
    return _full_box("ctts", bytes(payload))


def _stsc(entries: list[tuple[int, int, int]]) -> bytes:
    payload = bytearray()
    payload += len(entries).to_bytes(4, "big")
    for first_chunk, samples_per_chunk, sample_description_index in entries:
        payload += int(first_chunk).to_bytes(4, "big")
        payload += int(samples_per_chunk).to_bytes(4, "big")
        payload += int(sample_description_index).to_bytes(4, "big")
    return _full_box("stsc", bytes(payload))


def _stsz(sizes: list[int]) -> bytes:
    payload = bytearray()
    payload += (0).to_bytes(4, "big")
    payload += len(sizes).to_bytes(4, "big")
    for size in sizes:
        payload += int(size).to_bytes(4, "big")
    return _full_box("stsz", bytes(payload))


def _stco(offsets: list[int]) -> bytes:
    payload = bytearray()
    payload += len(offsets).to_bytes(4, "big")
    for offset in offsets:
        payload += int(offset).to_bytes(4, "big")
    return _full_box("stco", bytes(payload))


def _stss(sample_numbers: list[int]) -> bytes:
    payload = bytearray()
    payload += len(sample_numbers).to_bytes(4, "big")
    for number in sample_numbers:
        payload += int(number).to_bytes(4, "big")
    return _full_box("stss", bytes(payload))


def _elst(media_time: int = 1024) -> bytes:
    payload = (1).to_bytes(4, "big")
    payload += (0).to_bytes(4, "big")
    payload += int(media_time).to_bytes(4, "big", signed=True)
    payload += (1).to_bytes(2, "big")
    payload += (0).to_bytes(2, "big")
    return _box("edts", _full_box("elst", payload))


def _tkhd(track_id: int) -> bytes:
    payload = bytearray(72)
    payload[12:16] = int(track_id).to_bytes(4, "big")
    return _box("tkhd", bytes(payload))


def _hdlr(handler_type: bytes) -> bytes:
    payload = bytearray(24)
    payload[8:12] = handler_type
    return _box("hdlr", bytes(payload))


def _trak(track_id: int, handler_type: bytes, *, include_edts: bool) -> bytes:
    stbl = _box("stbl", _box("stsc", b"old") + _box("stco", b"old") + _box("stsz", b"sizes"))
    mdia = _box("mdia", _hdlr(handler_type) + _box("minf", stbl))
    children = [_tkhd(track_id)]
    if include_edts:
        children.append(_box("edts", _box("elst", b"edit")))
    children.append(mdia)
    return _box("trak", b"".join(children))


def _minimal_moov(track: bytes) -> bytes:
    return _box("moov", _box("mvhd", b"\x00" * 32) + track)


def _minimal_moov_with_timescale(track: bytes, *, timescale: int = 1000, duration: int = 0) -> bytes:
    payload = bytearray(100)
    payload[12:16] = int(timescale).to_bytes(4, "big")
    payload[16:20] = int(duration).to_bytes(4, "big")
    payload[-4:] = (3).to_bytes(4, "big")
    return _box("moov", _box("mvhd", bytes(payload)) + track)


class SIVirtualMp4Tests(unittest.TestCase):
    def test_virtual_regions_read_crosses_memory_and_file_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.bin"
            source.write_bytes(b"0123456789abcdef")
            regions = [
                VirtualRegion.memory(0, b"HEAD"),
                VirtualRegion.file(4, source, 5, 6),
                VirtualRegion.memory(10, b"TAIL"),
            ]
            self.assertEqual(virtual_size(regions), 14)
            data = b"".join(iter_virtual_range(regions, 2, 12, chunk_size=3))
        self.assertEqual(data, b"AD56789aTAI")

    def test_virtual_regions_require_contiguous_layout(self) -> None:
        regions = [
            VirtualRegion.memory(0, b"HEAD"),
            VirtualRegion.memory(8, b"TAIL"),
        ]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            virtual_size(regions)

    def test_box_splitting_sink_records_split_top_level_boxes(self) -> None:
        sink = BoxSplittingSink()
        data = _box("ftyp", b"1234") + _box("moov", b"abcdefgh")
        self.assertEqual(sink.write(data[:7]), 7)
        self.assertEqual(sink.boxes, [])
        self.assertEqual(sink.write(data[7:13]), 6)
        self.assertEqual([box.type for box in sink.boxes], ["ftyp"])
        self.assertEqual(sink.write(data[13:]), len(data) - 13)
        self.assertEqual([box.type for box in sink.boxes], ["ftyp", "moov"])
        self.assertEqual(sink.box_bytes(sink.boxes[1]), _box("moov", b"abcdefgh"))

    def test_box_splitting_sink_records_extended_size_box(self) -> None:
        sink = BoxSplittingSink()
        data = _extended_box("uuid", b"payload")
        sink.write(data[:10])
        self.assertEqual(sink.boxes, [])
        sink.write(data[10:])
        self.assertEqual(len(sink.boxes), 1)
        self.assertEqual(sink.boxes[0].type, "uuid")
        self.assertEqual(sink.boxes[0].size, len(data))

    def test_iter_virtual_range_random_ranges_match_ground_truth_and_are_stable(self) -> None:
        """Byte-stability invariant: across many random ranges and chunk sizes a
        mixed memory/file virtual layout must (a) equal the ground-truth bytes
        and (b) return identical bytes when sliced again. This is the property a
        virtual-remux MP4 relies on for stable Content-Length / ETag / seek.
        """
        import random

        rng = random.Random(20260617)
        with tempfile.TemporaryDirectory() as tmp:
            file_payload = bytes(rng.randrange(256) for _ in range(8192))
            file_path = Path(tmp) / "src.bin"
            file_path.write_bytes(file_payload)

            ground = bytearray()
            regions: list[VirtualRegion] = []
            cursor = 0
            file_offset = 0
            for k in range(20):
                if k % 2 == 0:
                    chunk = bytes(rng.randrange(256) for _ in range(rng.randint(1, 300)))
                    regions.append(VirtualRegion.memory(cursor, chunk))
                    ground += chunk
                    cursor += len(chunk)
                else:
                    size = min(rng.randint(1, 400), len(file_payload) - file_offset)
                    if size <= 0:
                        continue
                    regions.append(VirtualRegion.file(cursor, file_path, file_offset, size))
                    ground += file_payload[file_offset:file_offset + size]
                    cursor += size
                    file_offset += size

            ground_bytes = bytes(ground)
            self.assertEqual(virtual_size(regions), len(ground_bytes))
            self.assertGreater(len(regions), 4)

            for _ in range(300):
                start = rng.randint(0, len(ground_bytes) - 1)
                end = rng.randint(start, len(ground_bytes) - 1)
                chunk_size = rng.choice([1, 7, 64, 4096, 65536])
                first = b"".join(iter_virtual_range(regions, start, end, chunk_size=chunk_size))
                self.assertEqual(first, ground_bytes[start:end + 1])
                second = b"".join(iter_virtual_range(regions, start, end, chunk_size=64 * 1024))
                self.assertEqual(first, second)

            whole = b"".join(iter_virtual_range(regions, 0, len(ground_bytes) - 1))
            self.assertEqual(whole, ground_bytes)

    def test_video_sample_table_is_read_from_moov_without_demux_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "movie.mp4"
            stbl = _box(
                "stbl",
                _stsd_video("hvc1")
                + _stts([(3, 40)])
                + _ctts([(2, 5), (1, 10)])
                + _stsc([(1, 2, 1), (2, 1, 1)])
                + _stsz([100, 110, 120])
                + _stco([1000, 2000])
                + _stss([1, 3]),
            )
            mdia = _box("mdia", _mdhd(1000, 120) + _hdlr(b"vide") + _box("minf", stbl))
            moov = _minimal_moov(_box("trak", _tkhd(1) + mdia))
            path.write_bytes(_box("ftyp", b"isom") + moov)

            table = read_media_sample_table(path, "video")

        self.assertEqual(table.codec_name, "hevc")
        self.assertEqual(table.time_base, Fraction(1, 1000))
        self.assertEqual(table.duration_seconds, 0.12)
        self.assertEqual([sample.source_offset for sample in table.samples], [1000, 1100, 2000])
        self.assertEqual([sample.size for sample in table.samples], [100, 110, 120])
        self.assertEqual([sample.dts for sample in table.samples], [0, 40, 80])
        self.assertEqual([sample.pts for sample in table.samples], [5, 45, 90])
        self.assertEqual([sample.time_seconds for sample in table.samples], [0.0, 0.04, 0.08])
        self.assertEqual([sample.keyframe for sample in table.samples], [True, False, True])

    def test_plan_mix_segments_is_frame_aligned_and_complete(self) -> None:
        self.assertEqual([(s.start_frame, s.end_frame, s.encode_start_frame) for s in _plan_mix_segments(10, 1, 3)], [(0, 10, 0)])
        segments = _plan_mix_segments(100, 8, 5)
        self.assertEqual(segments[0].start_frame, 0)
        self.assertEqual(segments[-1].end_frame, 100)
        for prev, cur in zip(segments, segments[1:], strict=False):
            self.assertEqual(prev.end_frame, cur.start_frame)
            self.assertLessEqual(cur.encode_start_frame, cur.start_frame)
            self.assertGreaterEqual(cur.leading_frames, 0)
        self.assertEqual(sum(segment.keep_frames for segment in segments), 100)

    def test_sequential_audio_extraction_matches_run_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparse.bin"
            payload = bytearray(b"v" * 2048)
            samples = []
            chunks = [(100, b"aaa"), (777, b"bbbb"), (1600, b"cc")]
            for index, (offset, data) in enumerate(chunks):
                payload[offset:offset + len(data)] = data
                samples.append(
                    siv.MediaSample(
                        index=index,
                        source_offset=offset,
                        size=len(data),
                        pts=index,
                        dts=index,
                        keyframe=True,
                        time_seconds=float(index),
                    )
                )
            path.write_bytes(bytes(payload))
            run_out = bytearray()
            seq_out = bytearray()

            class Sink:
                def __init__(self, target: bytearray) -> None:
                    self.target = target

                def write(self, data: bytes) -> None:
                    self.target.extend(data)

            run_bytes = _copy_file_ranges_to_outputs(path, [Sink(run_out)], samples, cancel_event=None)
            seq_bytes = _copy_file_ranges_sequential_to_outputs(path, [Sink(seq_out)], samples, cancel_event=None, chunk_size=256)

        self.assertEqual(run_bytes, seq_bytes)
        self.assertEqual(bytes(seq_out), b"aaabbbbcc")

    def test_source_audio_sidecar_copies_only_audio_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "movie.mp4"
            cache = root / "cache"
            cache.mkdir()
            samples = [b"aaa", b"bbbb", b"cc"]
            payload = b"".join(samples)
            ftyp = _box("ftyp", b"isom")

            def make_moov(offsets: list[int]) -> bytes:
                stbl = _box(
                    "stbl",
                    _stsd_audio("mp4a")
                    + _stts([(3, 1024)])
                    + _stsc([(1, 2, 1), (2, 1, 1)])
                    + _stsz([len(sample) for sample in samples])
                    + _stco(offsets),
                )
                mdia = _box("mdia", _mdhd(48000, 3072) + _hdlr(b"soun") + _box("minf", stbl))
                return _minimal_moov(_box("trak", _tkhd(2) + mdia))

            moov = make_moov([0, 0])
            for _ in range(3):
                mdat_payload_start = len(ftyp) + len(moov) + 8
                offsets = [mdat_payload_start, mdat_payload_start + len(samples[0]) + len(samples[1])]
                new_moov = make_moov(offsets)
                if len(new_moov) == len(moov):
                    moov = new_moov
                    break
                moov = new_moov
            path.write_bytes(ftyp + moov + _box("mdat", payload))

            with patch.object(siv, "_ensure_cache_dir", return_value=cache):
                out = siv.build_source_audio_sidecar(path)
                table = read_media_sample_table(out, "audio")
                data = out.read_bytes()
        mdat_type = data.find(b"mdat")
        self.assertGreater(mdat_type, 0)
        self.assertEqual(data[mdat_type + 12:], payload)
        self.assertEqual(table.codec_name, "aac")
        self.assertEqual(table.time_base, Fraction(1, 48000))
        self.assertEqual([sample.size for sample in table.samples], [3, 4, 2])
        self.assertEqual([sample.dts for sample in table.samples], [0, 1024, 2048])

    def test_mixed_audio_sidecar_uses_cached_source_audio_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            video = root / "movie.mp4"
            si_wav = root / "movie.si.wav"
            source_audio = root / "source-audio.mp4"
            video.write_bytes(b"video")
            si_wav.write_bytes(b"si")
            source_audio.write_bytes(b"audio")
            recorded: dict[str, list[str]] = {}

            def fake_run(cmd, *, cancel_event, low_priority) -> None:
                recorded["cmd"] = list(cmd)
                Path(cmd[-1]).write_bytes(b"mixed")

            with (
                patch.object(siv, "_source_audio_sidecar_output", return_value=("digest", source_audio)),
                patch.object(siv, "_selected_mix_encoder", return_value="aac"),
                patch.object(siv, "_run_ffmpeg_sidecar", side_effect=fake_run),
            ):
                out = siv.build_mixed_audio_sidecar(video, si_wav, SIMixParams(si_delay_seconds=0.3))
                output_bytes = out.read_bytes()

        self.assertEqual(output_bytes, b"mixed")
        cmd = recorded["cmd"]
        inputs = [cmd[i + 1] for i, part in enumerate(cmd[:-1]) if part == "-i"]
        self.assertEqual(inputs, [str(source_audio), str(si_wav)])
        self.assertNotIn(str(video), inputs)

    def test_mixed_audio_sidecar_pipes_source_audio_when_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            video = root / "movie.mp4"
            si_wav = root / "movie.si.wav"
            source_audio = root / "source-audio.mp4"
            video.write_bytes(b"video")
            si_wav.write_bytes(b"si")
            recorded: dict[str, list[str]] = {}

            def fake_pipe_run(cmd, *, video, source_audio_output, cancel_event, low_priority) -> None:
                recorded["cmd"] = list(cmd)
                Path(cmd[-1]).write_bytes(b"mixed")
                source_audio_output.write_bytes(b"audio-cache")

            with (
                patch.object(siv, "_ensure_cache_dir", return_value=cache),
                patch.object(siv, "_source_audio_sidecar_output", return_value=("digest-pipe", source_audio)),
                patch.object(siv, "_selected_mix_encoder", return_value="aac"),
                patch.object(siv, "_run_ffmpeg_sidecar_with_source_audio_pipe", side_effect=fake_pipe_run) as pipe_run,
                patch.object(siv, "_run_ffmpeg_sidecar") as normal_run,
            ):
                out = siv.build_mixed_audio_sidecar(video, si_wav, SIMixParams(si_delay_seconds=0.5))
                output_bytes = out.read_bytes()
                source_audio_bytes = source_audio.read_bytes()

        self.assertEqual(output_bytes, b"mixed")
        pipe_run.assert_called_once()
        normal_run.assert_not_called()
        self.assertEqual(source_audio_bytes, b"audio-cache")
        cmd = recorded["cmd"]
        inputs = [cmd[i + 1] for i, part in enumerate(cmd[:-1]) if part == "-i"]
        self.assertEqual(inputs, ["pipe:0", str(si_wav)])

    def test_stitch_aac_segments_preserves_first_priming_and_drops_middle_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_segment(path: Path, frames: list[bytes]) -> None:
                ftyp = _box("ftyp", b"isom")

                def make_moov(offsets: list[int]) -> bytes:
                    stbl = _box(
                        "stbl",
                        _stsd_audio("mp4a")
                        + _stts([(len(frames), 1024)])
                        + _stsc([(1, 1, 1)])
                        + _stsz([len(frame) for frame in frames])
                        + _stco(offsets),
                    )
                    mdia = _box("mdia", _mdhd(48000, len(frames) * 1024) + _hdlr(b"soun") + _box("minf", stbl))
                    return _minimal_moov_with_timescale(_box("trak", _tkhd(2) + _elst(1024) + mdia))

                moov = make_moov([0] * len(frames))
                for _ in range(3):
                    cursor = len(ftyp) + len(moov) + 8
                    offsets = []
                    for frame in frames:
                        offsets.append(cursor)
                        cursor += len(frame)
                    new_moov = make_moov(offsets)
                    if len(new_moov) == len(moov):
                        moov = new_moov
                        break
                    moov = new_moov
                path.write_bytes(ftyp + moov + _box("mdat", b"".join(frames)))

            seg0 = root / "seg0.mp4"
            seg1 = root / "seg1.mp4"
            out = root / "stitched.mp4"
            write_segment(seg0, [b"P", b"A", b"B"])
            write_segment(seg1, [b"Q", b"W", b"C", b"D"])
            segments = (
                siv._MixSegment(index=0, start_frame=0, end_frame=2, encode_start_frame=0, encode_end_frame=2),
                siv._MixSegment(index=1, start_frame=2, end_frame=4, encode_start_frame=1, encode_end_frame=4),
            )

            siv._stitch_aac_segments([seg0, seg1], segments, out, cancel_event=None)
            table = read_media_sample_table(out, "audio")
            moov = siv._read_top_level_box(out, b"moov")
            audio_trak = _find_track(moov, "audio").trak
            data = out.read_bytes()

        mdat_type = data.find(b"mdat")
        self.assertGreater(mdat_type, 0)
        self.assertEqual(data[mdat_type + 12:], b"PABCD")
        self.assertEqual([sample.size for sample in table.samples], [1, 1, 1, 1, 1])
        self.assertEqual([sample.dts for sample in table.samples], [0, 1024, 2048, 3072, 4096])
        self.assertIsNotNone(_find_child(moov, audio_trak, b"edts"))
        self.assertEqual(siv._audio_edit_media_time_from_moov(moov), 1024)
        self.assertEqual(siv._audio_priming_frames_from_moov(moov), 1)

    def test_prewarm_queue_is_bounded_and_deduped(self) -> None:
        old_queue = siv._layout_prewarm_queue
        old_inflight = set(siv._layout_prewarm_inflight)
        old_started = siv._layout_prewarm_worker_started
        try:
            siv._layout_prewarm_queue = queue.PriorityQueue(maxsize=1)
            siv._layout_prewarm_inflight.clear()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = root / "first.mp4"
                second = root / "second.mp4"
                first.write_bytes(b"video-a")
                second.write_bytes(b"video-b")
                first_si = first.with_suffix(".si.wav")
                second_si = second.with_suffix(".si.wav")
                first_si.write_bytes(b"si-a")
                second_si.write_bytes(b"si-b")

                with (
                    patch.object(siv, "SI_PREWARM_QUEUE_MAX", 1),
                    patch.object(siv, "_ensure_prewarm_worker_locked") as ensure_worker,
                ):
                    self.assertTrue(siv.prewarm_progressive_si_virtual_mp4(first, first_si, SIMixParams(), reason="test"))
                    self.assertFalse(siv.prewarm_progressive_si_virtual_mp4(first, first_si, SIMixParams(), reason="test"))
                    self.assertFalse(siv.prewarm_progressive_si_virtual_mp4(second, second_si, SIMixParams(), reason="test"))

            self.assertEqual(siv._layout_prewarm_queue.qsize(), 1)
            ensure_worker.assert_called_once()
        finally:
            siv._layout_prewarm_queue = old_queue
            siv._layout_prewarm_inflight.clear()
            siv._layout_prewarm_inflight.update(old_inflight)
            siv._layout_prewarm_worker_started = old_started

    def test_audio_edit_mode_preserve_keeps_imported_audio_edts(self) -> None:
        source_moov = _minimal_moov(_trak(1, b"vide", include_edts=True))
        audio_moov = _minimal_moov(_trak(2, b"soun", include_edts=True))

        moov = _build_moov(
            source_moov,
            audio_moov,
            [100],
            [200],
            audio_edit_mode="preserve",
        )

        video_trak = _find_track(moov, "video").trak
        audio_trak = _find_track(moov, "audio").trak
        self.assertIsNotNone(_find_child(moov, video_trak, b"edts"))
        self.assertIsNotNone(_find_child(moov, audio_trak, b"edts"))

    def test_audio_edit_mode_remove_drops_only_imported_audio_edts(self) -> None:
        source_moov = _minimal_moov(_trak(1, b"vide", include_edts=True))
        audio_moov = _minimal_moov(_trak(2, b"soun", include_edts=True))

        moov = _build_moov(
            source_moov,
            audio_moov,
            [100],
            [200],
            audio_edit_mode="remove",
        )

        video_trak = _find_track(moov, "video").trak
        audio_trak = _find_track(moov, "audio").trak
        self.assertIsNotNone(_find_child(moov, video_trak, b"edts"))
        self.assertIsNone(_find_child(moov, audio_trak, b"edts"))


if __name__ == "__main__":
    unittest.main()
