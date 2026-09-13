import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import source_budget_plan
from pipeline.source_budget_plan import (
    SourceBudgetError,
    build_source_budget_plan,
)


def _table(sizes, *, fps=60.0):
    """A fake video sample table: ``sizes`` bytes, one sample per 1/fps."""
    samples = [
        SimpleNamespace(size=int(sz), time_seconds=i / fps)
        for i, sz in enumerate(sizes)
    ]
    return SimpleNamespace(
        samples=samples,
        duration_seconds=len(sizes) / fps,
    )


class SourceBudgetPlanTests(unittest.TestCase):
    def _build(self, sizes, *, payload_total, frame_count, gop_frames=60,
               output_fps=30.0, floor=48 * 1024, flatten=0.0, src_fps=60.0, audio=None):
        def fake_read(path, kind, **kw):
            if kind == "video":
                return _table(sizes, fps=src_fps)
            if audio is None:
                raise RuntimeError("no audio")
            return _table(audio, fps=src_fps)

        with (
            patch.object(source_budget_plan, "read_media_sample_table", fake_read),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_size=payload_total)),
        ):
            return build_source_budget_plan(
                Path("movie.mp4"),
                payload_total=payload_total,
                frame_count=frame_count,
                gop_frames=gop_frames,
                output_fps=output_fps,
                floor_bytes_per_frame=floor,
                flatten=flatten,
            )

    def test_budgets_sum_to_the_exact_payload_total(self) -> None:
        # 4 output GOPs (240 frames @30fps = 8s) from an 8s 60fps source.
        sizes = [80_000] * 480
        total = 240 * 100_000 + 7    # deliberately not divisible by frame count
        plan = self._build(sizes, payload_total=total, frame_count=240)
        self.assertEqual(sum(plan.frame_budgets), total)
        self.assertEqual(len(plan.frame_budgets), 240)
        self.assertEqual(sum(plan.gop_budgets), total)
        self.assertEqual(len(plan.gop_budgets), 4)

    def test_rich_stretches_get_more_bytes_than_quiet_ones(self) -> None:
        # GOP 0 is 4x the weight of GOP 1 in the source.
        sizes = [160_000] * 120 + [40_000] * 120
        total = 120 * 100_000
        plan = self._build(sizes, payload_total=total, frame_count=120, gop_frames=60)
        self.assertEqual(len(plan.gop_budgets), 2)
        self.assertGreater(plan.gop_budgets[0], plan.gop_budgets[1] * 3)
        self.assertEqual(sum(plan.gop_budgets), total)

    def test_flatten_gives_every_gop_the_same_budget(self) -> None:
        """The shipped default ignores the source's shape.

        Inheriting it starved the stretches our realtime encoder needs most:
        offline x265 found them cheap and NVENC does not, so those frames were
        truncated. Measured on an 8K title, flattening cut truncated frames from
        21 to 3 and raised delivered picture from 47.5 to 62.6 Mbps at identical
        total bytes.
        """
        import config

        self.assertEqual(config.PASSTHROUGH_SEEK_VMP4_FRAMES_BUDGET_FLATTEN, 1.0)
        # A 4x/1x split in the source that a flat plan must ignore entirely.
        sizes = [160_000] * 120 + [40_000] * 120
        total = 120 * 100_000
        flat = self._build(sizes, payload_total=total, frame_count=120, gop_frames=60, flatten=1.0)
        self.assertEqual(len(flat.gop_budgets), 2)
        self.assertLessEqual(abs(flat.gop_budgets[0] - flat.gop_budgets[1]), 1)
        self.assertEqual(sum(flat.gop_budgets), total)
        # Half way keeps half the difference.
        half = self._build(sizes, payload_total=total, frame_count=120, gop_frames=60, flatten=0.5)
        shaped = self._build(sizes, payload_total=total, frame_count=120, gop_frames=60, flatten=0.0)
        self.assertGreater(half.gop_budgets[0], flat.gop_budgets[0])
        self.assertLess(half.gop_budgets[0], shaped.gop_budgets[0])
        self.assertEqual(sum(half.gop_budgets), total)

    def test_starved_gop_is_raised_to_the_floor_conserving_total(self) -> None:
        # A near-black GOP in the source: 2 KiB/frame.
        sizes = [2 * 1024] * 120 + [200_000] * 360
        total = 240 * 120 * 1024
        floor = 48 * 1024
        plan = self._build(sizes, payload_total=total, frame_count=240, floor=floor)
        self.assertEqual(sum(plan.frame_budgets), total)
        self.assertGreaterEqual(min(plan.frame_budgets), floor)
        self.assertGreater(plan.clamped_gops, 0)

    def test_source_too_small_for_the_floor_is_rejected(self) -> None:
        sizes = [1000] * 480
        with self.assertRaises(SourceBudgetError):
            self._build(sizes, payload_total=240 * 4096, frame_count=240, floor=48 * 1024)

    def test_gop_head_carries_the_rounding_remainder(self) -> None:
        sizes = [100_000] * 480
        total = 240 * 100_000 + 13
        plan = self._build(sizes, payload_total=total, frame_count=240)
        for gi in range(len(plan.gop_budgets)):
            head = plan.frame_budgets[gi * 60]
            rest = plan.frame_budgets[gi * 60 + 1: (gi + 1) * 60]
            self.assertGreaterEqual(head, max(rest))
            self.assertEqual(len(set(rest)), 1)

    def test_gop_bitrate_tracks_the_budget(self) -> None:
        sizes = [160_000] * 120 + [40_000] * 120
        plan = self._build(sizes, payload_total=120 * 100_000, frame_count=120, gop_frames=60)
        self.assertGreater(plan.gop_bitrate_bps(0), plan.gop_bitrate_bps(1))
        # 60 frames of budget at 30fps == 2s of media.
        self.assertAlmostEqual(
            plan.gop_bitrate_bps(0), plan.gop_budgets[0] * 8 / 2.0, delta=1.0
        )

    def test_audio_bytes_are_reported_when_present(self) -> None:
        plan = self._build(
            [100_000] * 480, payload_total=240 * 100_000, frame_count=240,
            audio=[500] * 480,
        )
        self.assertEqual(plan.source_audio_bytes, 480 * 500)


class TrailingGopTests(unittest.TestCase):
    """The last GOP of a title usually holds fewer frames than the rest."""

    def _plan(self, *, frame_count, gop_frames=60, sizes=None, payload=None, src_fps=60.0):
        sizes = sizes or [100_000] * (frame_count * 2)
        total = payload or frame_count * 100_000

        def fake_read(path, kind, **kw):
            if kind != "video":
                raise RuntimeError("no audio")
            samples = [
                SimpleNamespace(size=int(sz), time_seconds=i / src_fps)
                for i, sz in enumerate(sizes)
            ]
            return SimpleNamespace(samples=samples, duration_seconds=len(sizes) / src_fps)

        with (
            patch.object(source_budget_plan, "read_media_sample_table", fake_read),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_size=total)),
        ):
            return build_source_budget_plan(
                Path("movie.mp4"),
                payload_total=total,
                frame_count=frame_count,
                gop_frames=gop_frames,
                output_fps=30.0,
                floor_bytes_per_frame=48 * 1024,
                flatten=0.0,
            )

    def test_two_frame_tail_does_not_get_a_whole_gop_budget(self) -> None:
        # 1802 frames at gop 60 leaves a 2-frame tail. Scaling that window up to
        # a full GOP's worth once produced a 90 MB frame budget, which NVENC
        # rejects outright (error 8) - that stretch then never encodes at all.
        plan = self._plan(frame_count=1802)
        self.assertEqual(len(plan.frame_budgets), 1802)
        self.assertEqual(sum(plan.frame_budgets), plan.payload_total)
        biggest = max(plan.frame_budgets)
        mean = plan.payload_total // 1802
        self.assertLess(biggest, mean * 12, f"largest frame budget {biggest} vs mean {mean}")
        for b in plan.frame_budgets[-2:]:
            self.assertLess(b, mean * 12)

    def test_tail_frames_still_clear_the_floor(self) -> None:
        plan = self._plan(frame_count=1802)
        self.assertGreaterEqual(min(plan.frame_budgets), 48 * 1024)

    def test_various_tail_lengths_stay_sane(self) -> None:
        for extra in (1, 2, 7, 30, 59):
            plan = self._plan(frame_count=600 + extra)
            mean = plan.payload_total // (600 + extra)
            self.assertEqual(sum(plan.frame_budgets), plan.payload_total)
            self.assertLess(
                max(plan.frame_budgets), mean * 12,
                f"tail of {extra} frames produced {max(plan.frame_budgets)}",
            )

    def test_exact_multiple_is_unaffected(self) -> None:
        plan = self._plan(frame_count=600)
        self.assertEqual(sum(plan.frame_budgets), plan.payload_total)
        self.assertEqual(len(plan.gop_budgets), 10)


if __name__ == "__main__":
    unittest.main()
