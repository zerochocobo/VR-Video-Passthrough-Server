from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.passthrough_vmp4_frames import (
    build_passthrough_vmp4_frames_layout,
    iter_vmp4_frames_range,
    vmp4_frame_index_for_offset,
)
from pipeline.si_virtual_mp4 import read_media_sample_table

# Reuse the synthetic MP4 writer / stsd helper from the slot test module.
from tests.test_passthrough_vmp4_slot import _write_source_mp4, _stsd_video
from pipeline.passthrough_vmp4_slot import Vmp4SlotLayoutError


def _build(tmp: Path, *, fps=10.0, gop=5, duration=2.0, idr=4096, p=1024):
    src = tmp / "movie.mp4"
    _write_source_mp4(src, [b"A" * 32, b"B" * 32, b"C" * 32], target_size=8192)
    return build_passthrough_vmp4_frames_layout(
        src,
        duration_sec=duration,
        fps=fps,
        gop_frames=gop,
        idr_budget=idr,
        p_budget=p,
        output_stsd=_stsd_video("hvc1"),
        output_codec_name="hevc",
        placeholder_payload=b"HEVCFRAME",
    )


class Vmp4FramesLayoutTests(unittest.TestCase):
    def test_frame_count_and_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp), fps=10.0, gop=5, duration=2.0, idr=4096, p=1024)
        self.assertEqual(layout.frame_count, 20)          # 2.0s * 10fps
        self.assertEqual(layout.slot_count, 4)            # 20 / gop 5
        self.assertEqual(len(layout.frames), 20)
        # GOP heads are keyframes with the IDR budget; others use the P budget.
        self.assertTrue(layout.frames[0].keyframe)
        self.assertEqual(layout.frames[0].budget, 4096)
        self.assertFalse(layout.frames[1].keyframe)
        self.assertEqual(layout.frames[1].budget, 1024)
        self.assertTrue(layout.frames[5].keyframe)        # next GOP head
        # total == init + sum(budgets)
        self.assertEqual(
            layout.total_size,
            layout.mdat_payload_start + sum(f.budget for f in layout.frames),
        )

    def test_offsets_are_contiguous_and_lookup_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp))
        cursor = layout.mdat_payload_start
        for fr in layout.frames:
            self.assertEqual(fr.offset, cursor)
            self.assertEqual(vmp4_frame_index_for_offset(layout, fr.offset), fr.index)
            self.assertEqual(
                vmp4_frame_index_for_offset(layout, fr.offset + fr.budget - 1), fr.index
            )
            cursor += fr.budget
        self.assertIsNone(vmp4_frame_index_for_offset(layout, layout.mdat_payload_start - 1))

    def test_range_reads_are_stable_and_filler_padded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp))
            fr = layout.frames[3]
            a = b"".join(iter_vmp4_frames_range(layout, fr.offset, fr.offset + 50))
            b = b"".join(iter_vmp4_frames_range(layout, fr.offset, fr.offset + 50))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 51)
        # No ready payload -> placeholder then filler NAL (HEVC filler header 0x4c01).
        self.assertTrue(a.startswith(b"HEVCFRAME"))

    def test_ready_payload_is_served_then_filler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp))
            fr = layout.frames[2]
            cache = Path(tmp) / "c"
            cache.mkdir()
            payload = cache / "f2.bin"
            payload.write_bytes(b"REALFRAMEDATA")
            got = b"".join(
                iter_vmp4_frames_range(
                    layout, fr.offset, fr.offset + len(b"REALFRAMEDATA") - 1,
                    payload_paths={2: payload},
                )
            )
        self.assertEqual(got, b"REALFRAMEDATA")

    def test_full_virtual_file_parses_as_real_fps_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp), fps=10.0, gop=5, duration=2.0)
            virtual = Path(tmp) / "virtual.mp4"
            virtual.write_bytes(b"".join(iter_vmp4_frames_range(layout, 0, layout.total_size - 1)))
            table = read_media_sample_table(virtual, "video")
        # One sample per frame, and the sample sizes match the fixed budgets.
        self.assertEqual(len(table.samples), layout.frame_count)
        self.assertEqual(
            [s.size for s in table.samples],
            [f.budget for f in layout.frames],
        )
        # Keyframes land on GOP heads only.
        kf = [i for i, s in enumerate(table.samples) if s.keyframe]
        self.assertEqual(kf, [f.index for f in layout.frames if f.keyframe])


class Vmp4FramesServedEndTests(unittest.TestCase):
    def test_served_end_bounds_to_contiguous_ready_frames(self) -> None:
        import http_app.routes_media as routes_media

        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp), fps=10.0, gop=5, duration=2.0)  # 20 frames
            # Frames 0,1,2 ready then a hole at 3.
            ready = {i: Path(tmp) / f"frame_{i:06d}.bin" for i in (0, 1, 2)}
            # A range starting in frame 0 serves through the end of frame 2 only.
            start = layout.frames[0].offset
            end = routes_media._vmp4_frames_served_end(layout, ready, start, layout.total_size - 1)
            self.assertEqual(end, layout.frames[2].offset + layout.frames[2].budget - 1)
            # A range whose start frame is not built returns -1 (caller -> 503).
            start3 = layout.frames[3].offset
            self.assertEqual(routes_media._vmp4_frames_served_end(layout, ready, start3, layout.total_size - 1), -1)
            # A request capped by req_end stays capped.
            capped = routes_media._vmp4_frames_served_end(layout, ready, start, layout.frames[1].offset + 3)
            self.assertEqual(capped, layout.frames[1].offset + 3)


class Vmp4FramesFillerTests(unittest.TestCase):
    def test_persistent_filler_streams_per_frame_files_and_serves_them(self) -> None:
        import time
        from unittest.mock import patch

        import http_app.routes_media as routes_media
        import pipeline.pynv_stream as pynv_stream
        from pipeline.passthrough_vmp4_slot import hevc_annexb_to_length_prefixed_sample

        def nal(t: int, body: bytes) -> bytes:
            return b"\x00\x00\x00\x01" + bytes([t << 1, 1]) + body

        # 10 frames (fps 10 * 1.0s): each a single-AU Annex-B access unit; the
        # GOP heads (every 5) carry VPS/SPS/PPS, the rest are P slices.
        def fake_frames(src, *, start_sec, frame_count, matter, output_mode, per_frame_cap_bytes, **kw):
            start = int(round(start_sec * 10.0))
            for i in range(frame_count):
                gi = start + i
                if gi % 5 == 0:
                    yield nal(32, b"v") + nal(33, b"s") + nal(34, b"p") + nal(19, bytes([gi & 0xFF]))
                else:
                    yield nal(1, bytes([gi & 0xFF]))

        original = dict(routes_media._vmp4_frames_fillers)
        try:
            routes_media._vmp4_frames_fillers.clear()
            with tempfile.TemporaryDirectory() as tmp:
                layout = _build(Path(tmp), fps=10.0, gop=5, duration=1.0, idr=4096, p=4096)
                self.assertEqual(layout.frame_count, 10)
                cache_dir = Path(tmp) / "cache"
                digest = "testdigest"
                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "acquire_matter", return_value=object()),
                    patch.object(routes_media, "release_matter", return_value=None),
                    patch.object(pynv_stream, "iter_pynv_passthrough_annexb_frames", fake_frames),
                ):
                    ok, ready, filler = routes_media._vmp4_frames_wait_gop(
                        layout, cache_dir, digest, 0, "green", 3.0
                    )
                    self.assertTrue(ok)
                    # The single persistent filler streams forward over BOTH GOPs
                    # (one setup, not one per GOP). Wait for the whole run.
                    deadline = time.time() + 3.0
                    while time.time() < deadline and filler.state == "running":
                        time.sleep(0.02)
                ready = routes_media._vmp4_frames_ready_payloads(cache_dir)
                self.assertTrue(routes_media._vmp4_frames_gop_ready(layout, 0, ready))
                self.assertTrue(routes_media._vmp4_frames_gop_ready(layout, 1, ready))
                # Frame files hold the converted access-unit bytes for each frame.
                for i in range(layout.frame_count):
                    au = (
                        nal(32, b"v") + nal(33, b"s") + nal(34, b"p") + nal(19, bytes([i & 0xFF]))
                        if i % 5 == 0
                        else nal(1, bytes([i & 0xFF]))
                    )
                    self.assertEqual(
                        (cache_dir / f"frame_{i:06d}.bin").read_bytes(),
                        hevc_annexb_to_length_prefixed_sample(au),
                    )
                # Served frame 7 bytes match its real payload.
                fr = layout.frames[7]
                exp = hevc_annexb_to_length_prefixed_sample(nal(1, bytes([7])))
                got = b"".join(
                    iter_vmp4_frames_range(layout, fr.offset, fr.offset + len(exp) - 1, payload_paths=ready)
                )
                self.assertEqual(got, exp)
        finally:
            for f in routes_media._vmp4_frames_fillers.values():
                f.stop_event.set()
            routes_media._vmp4_frames_fillers.clear()
            routes_media._vmp4_frames_fillers.update(original)


if __name__ == "__main__":
    unittest.main()


class Vmp4FramesSourceBudgetTests(unittest.TestCase):
    """Per-frame budgets supplied by pipeline/source_budget_plan."""

    def _source(self, tmp: Path, sizes: list[int], *, target_size: int) -> Path:
        src = tmp / "movie.mp4"
        _write_source_mp4(src, [b"A" * n for n in sizes], target_size=target_size)
        return src

    def test_supplied_budgets_drive_stsz_and_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._source(tmp, [64, 64, 64], target_size=8192)
            budgets = [8192, 2048, 2048, 2048, 2048] * 4      # 20 frames, varied
            layout = build_passthrough_vmp4_frames_layout(
                src,
                duration_sec=2.0,
                fps=10.0,
                gop_frames=5,
                idr_budget=4096,
                p_budget=1024,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                frame_budgets=budgets,
            )
        self.assertTrue(layout.source_budget)
        self.assertEqual([f.budget for f in layout.frames], budgets)
        self.assertEqual(layout.mdat_payload_size, sum(budgets))
        self.assertEqual(layout.total_size, layout.mdat_payload_start + sum(budgets))
        # Offsets stay contiguous so co64 still describes a gapless mdat.
        cursor = layout.mdat_payload_start
        for fr in layout.frames:
            self.assertEqual(fr.offset, cursor)
            cursor += fr.budget
        self.assertEqual(cursor, layout.total_size)

    def test_budget_count_must_match_the_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._source(tmp, [64], target_size=4096)
            with self.assertRaises(Vmp4SlotLayoutError):
                build_passthrough_vmp4_frames_layout(
                    src,
                    duration_sec=2.0,
                    fps=10.0,
                    gop_frames=5,
                    idr_budget=4096,
                    p_budget=1024,
                    output_stsd=_stsd_video("hvc1"),
                    output_codec_name="hevc",
                    frame_budgets=[4096] * 19,       # layout needs 20
                )

    def test_flat_layout_reports_no_source_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp))
        self.assertFalse(layout.source_budget)

    def test_init_size_is_independent_of_budget_values(self) -> None:
        # The routes layer builds a flat probe layout to learn the init size and
        # then rebuilds with real budgets; that only works because the moov is
        # fixed-width, so assert it here.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = self._source(tmp, [64, 64], target_size=8192)
            common = dict(
                duration_sec=2.0, fps=10.0, gop_frames=5,
                output_stsd=_stsd_video("hvc1"), output_codec_name="hevc",
            )
            flat = build_passthrough_vmp4_frames_layout(
                src, idr_budget=4096, p_budget=4096, **common
            )
            varied = build_passthrough_vmp4_frames_layout(
                src, idr_budget=4096, p_budget=1024,
                frame_budgets=[9000, 1500, 70000, 2000, 3000] * 4, **common
            )
        self.assertEqual(flat.moov_size, varied.moov_size)
        self.assertEqual(
            flat.total_size - flat.mdat_payload_size,
            varied.total_size - varied.mdat_payload_size,
        )


class Vmp4FramesProbeRangeTests(unittest.TestCase):
    """Structural probes (header read, tail probe) must not wait on an encode."""

    def setUp(self) -> None:
        from http_app import routes_media
        self.rm = routes_media
        with tempfile.TemporaryDirectory() as tmp:
            self.layout = _build(Path(tmp), fps=10.0, gop=5, duration=20.0,
                                 idr=64 * 1024, p=64 * 1024)

    def _range(self, start, end):
        return self.rm.ByteRange(start=start, end=end, total=self.layout.total_size)

    def test_header_only_range_is_an_init_probe(self) -> None:
        end = self.layout.mdat_payload_start - 1
        self.assertEqual(
            self.rm._vmp4_frames_probe_range(self.layout, self._range(0, end), ranged=True),
            "init",
        )
        # Even with no Range header: a header read never needs encoded bytes.
        self.assertEqual(
            self.rm._vmp4_frames_probe_range(self.layout, self._range(0, end), ranged=False),
            "init",
        )

    def test_tail_probe_is_recognised(self) -> None:
        total = self.layout.total_size
        rng = self._range(total - 6608, total - 1)
        self.assertEqual(
            self.rm._vmp4_frames_probe_range(self.layout, rng, ranged=True), "tail"
        )

    def test_unranged_playback_is_not_a_probe(self) -> None:
        rng = self._range(self.layout.mdat_payload_start, self.layout.total_size - 1)
        self.assertEqual(
            self.rm._vmp4_frames_probe_range(self.layout, rng, ranged=False), ""
        )

    def test_mid_file_bounded_read_still_waits(self) -> None:
        mid = self.layout.mdat_payload_start + self.layout.mdat_payload_size // 2
        rng = self._range(mid, mid + 65535)
        self.assertEqual(
            self.rm._vmp4_frames_probe_range(self.layout, rng, ranged=True), ""
        )

    def test_header_crossing_read_is_served_not_stalled(self) -> None:
        rng = self._range(0, self.layout.mdat_payload_start + 4096)
        self.assertEqual(
            self.rm._vmp4_frames_probe_range(self.layout, rng, ranged=True),
            "header-crossing",
        )

    def test_probe_bytes_zero_disables_ranged_probes(self) -> None:
        from unittest.mock import patch
        total = self.layout.total_size
        rng = self._range(total - 6608, total - 1)
        with patch.object(self.rm, "PASSTHROUGH_SEEK_VMP4_FRAMES_PROBE_BYTES", 0):
            self.assertEqual(
                self.rm._vmp4_frames_probe_range(self.layout, rng, ranged=True), ""
            )
            # The init range is structural regardless of the knob.
            init = self._range(0, self.layout.mdat_payload_start - 1)
            self.assertEqual(
                self.rm._vmp4_frames_probe_range(self.layout, init, ranged=True), "init"
            )


class Vmp4FramesAudioTests(unittest.TestCase):
    """Source audio copied through and interleaved one block per GOP."""

    def _layout(self, tmp: Path, *, with_audio: bool):
        from types import SimpleNamespace
        from fractions import Fraction

        src = tmp / "movie.mp4"
        _write_source_mp4(src, [b"A" * 32, b"B" * 32, b"C" * 32], target_size=8192)
        table = None
        if with_audio:
            # 40 samples over 2.0s: 20 per GOP at gop=5 frames / 10fps (0.5s).
            samples = [
                SimpleNamespace(
                    index=i, size=100 + i, source_offset=1000 + i * 200,
                    pts=i * 1024, dts=i * 1024, keyframe=True,
                    time_seconds=i * 0.05,
                )
                for i in range(40)
            ]
            table = SimpleNamespace(
                samples=tuple(samples), time_base=Fraction(1, 48000),
                duration_seconds=2.0, codec_name="aac",
            )
        return build_passthrough_vmp4_frames_layout(
            src,
            duration_sec=2.0, fps=10.0, gop_frames=5,
            idr_budget=4096, p_budget=1024,
            output_stsd=_stsd_video("hvc1"), output_codec_name="hevc",
            audio_table=table,
        )

    def test_audio_samples_are_placed_and_sized_from_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp), with_audio=True)
        self.assertEqual(len(layout.audio_samples), 40)
        self.assertEqual(layout.audio_bytes, sum(100 + i for i in range(40)))
        for i, a in enumerate(layout.audio_samples):
            self.assertEqual(a.size, 100 + i)             # source size, untouched
            self.assertEqual(a.source_offset, 1000 + i * 200)

    def test_regions_tile_the_mdat_without_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp), with_audio=True)
        regions = sorted(
            [(f.offset, f.budget) for f in layout.frames]
            + [(a.offset, a.size) for a in layout.audio_samples]
        )
        cursor = layout.mdat_payload_start
        for offset, size in regions:
            self.assertEqual(offset, cursor)
            cursor += size
        self.assertEqual(cursor, layout.total_size)
        self.assertEqual(
            layout.mdat_payload_size,
            sum(f.budget for f in layout.frames) + layout.audio_bytes,
        )

    def test_audio_is_interleaved_per_gop_not_parked_at_one_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp), with_audio=True)
        # Each GOP's audio must sit after that GOP's frames and before the next.
        for gi, (start, end) in enumerate(layout.gop_spans):
            for a in layout.audio_samples:
                if a.gop_index == gi:
                    self.assertGreaterEqual(a.offset, start)
                    self.assertLess(a.offset, end)
            for f in layout.frames:
                if f.gop_index == gi:
                    self.assertGreaterEqual(f.offset, start)
                    self.assertLess(f.offset, end)

    def test_gop_lookup_works_for_offsets_inside_audio(self) -> None:
        from pipeline.passthrough_vmp4_frames import vmp4_gop_for_offset

        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp), with_audio=True)
        for a in layout.audio_samples:
            self.assertEqual(vmp4_gop_for_offset(layout, a.offset), a.gop_index)
        for f in layout.frames:
            self.assertEqual(vmp4_gop_for_offset(layout, f.offset), f.gop_index)
        self.assertIsNone(vmp4_gop_for_offset(layout, 0))

    def test_frame_lookup_inside_audio_reports_the_next_frame(self) -> None:
        """An offset in an interleaved audio block must not read as "before frame 0".

        Returning None there let the seek path fall back to GOP 0 and restart the
        encoder at the start of the title whenever a player reopened on an audio
        byte in the middle of a film.
        """
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp), with_audio=True)
        checked = 0
        for a in layout.audio_samples:
            idx = vmp4_frame_index_for_offset(layout, a.offset)
            later = [f.index for f in layout.frames if f.offset + f.budget > a.offset]
            if not later:                      # audio after the final frame
                self.assertIsNone(idx)
                continue
            self.assertEqual(idx, later[0])
            checked += 1
        self.assertGreater(checked, 0)
        # Offsets before the mdat payload still have no frame at all.
        self.assertIsNone(
            vmp4_frame_index_for_offset(layout, layout.mdat_payload_start - 1)
        )

    def test_a_source_without_audio_still_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp), with_audio=False)
        self.assertEqual(layout.audio_samples, ())
        self.assertEqual(layout.audio_bytes, 0)
        self.assertEqual(
            layout.mdat_payload_size, sum(f.budget for f in layout.frames)
        )


class SeekBypassesTheFillerDebounceTests(unittest.TestCase):
    """A seek must reposition the encoder at once; a read-ahead probe must not."""

    def _run(self, need_gop: int, cursor: int):
        import threading as _threading
        from unittest.mock import patch

        import http_app.routes_media as rm

        with tempfile.TemporaryDirectory() as tmp:
            layout = _build(Path(tmp), fps=30.0, gop=5, duration=200.0)
            cache = Path(tmp) / "cache"
            existing = rm._Vmp4FramesFiller(
                digest="d", cache_dir=cache, output_mode="green",
                start_frame=100, cursor=cursor, state="running",
                started_at=time.time(), 
            )
            # Still being read this instant: the debounce would normally hold.
            existing.last_hit_at = time.time()
            existing.first_frame_at = time.time()
            started = []
            with (
                patch.dict(rm._vmp4_frames_fillers, {"d": existing}, clear=True),
                patch.object(rm, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                patch.object(_threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: started.append(kw))),
            ):
                got = rm._vmp4_frames_ensure_filler(layout, cache, "d", "green", need_gop)
            return existing, got, started

    def test_a_nearby_request_is_debounced(self) -> None:
        # lead = max(gop*2, fps*20) = 600 frames; 30 frames ahead is a read-ahead.
        existing, got, started = self._run(need_gop=26, cursor=100)
        self.assertIs(got, existing)
        self.assertEqual(started, [])

    def test_a_seek_far_ahead_repositions_immediately(self) -> None:
        existing, got, started = self._run(need_gop=800, cursor=100)
        self.assertIsNot(got, existing)
        self.assertTrue(existing.stop_event.is_set())
        self.assertEqual(len(started), 1)

    def test_a_seek_backwards_repositions_immediately(self) -> None:
        # The run only moves forward, so it can never reach an earlier frame -
        # waiting the debounce out there is waiting for nothing.
        existing, got, started = self._run(need_gop=2, cursor=100)
        self.assertIsNot(got, existing)
        self.assertEqual(len(started), 1)
