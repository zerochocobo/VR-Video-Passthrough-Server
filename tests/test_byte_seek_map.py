from __future__ import annotations

import unittest

from utils.byte_seek_map import gop_duration_seconds, map_byte_start_to_time, snap_back_to_gop


class ByteSeekMapTests(unittest.TestCase):
    def test_prefix_region_does_not_map_to_media_time(self) -> None:
        mapped = map_byte_start_to_time(
            start=512 * 1024,
            total=100 * 1024 * 1024,
            duration_sec=100.0,
            header_bytes=1024 * 1024,
            output_fps=30.0,
            gop_frames=60,
        )

        self.assertTrue(mapped.prefix)
        self.assertEqual(mapped.snapped_time_sec, 0.0)
        self.assertEqual(mapped.ratio, 0.0)

    def test_maps_after_header_region_and_snaps_back_to_gop(self) -> None:
        mapped = map_byte_start_to_time(
            start=51 * 1024 * 1024,
            total=101 * 1024 * 1024,
            duration_sec=100.0,
            header_bytes=1 * 1024 * 1024,
            output_fps=30.0,
            gop_frames=60,
        )

        self.assertFalse(mapped.prefix)
        self.assertAlmostEqual(mapped.time_sec, 50.0, places=3)
        self.assertEqual(mapped.gop_seconds, 2.0)
        self.assertEqual(mapped.snapped_time_sec, 50.0)

    def test_snap_back_uses_floor_boundary(self) -> None:
        mapped = map_byte_start_to_time(
            start=52 * 1024 * 1024,
            total=101 * 1024 * 1024,
            duration_sec=100.0,
            header_bytes=1 * 1024 * 1024,
            output_fps=30.0,
            gop_frames=60,
        )

        self.assertAlmostEqual(mapped.time_sec, 51.0, places=3)
        self.assertEqual(mapped.snapped_time_sec, 50.0)

    def test_ratio_is_clamped_to_duration(self) -> None:
        mapped = map_byte_start_to_time(
            start=250,
            total=200,
            duration_sec=60.0,
            header_bytes=100,
            output_fps=24.0,
            gop_frames=48,
        )

        self.assertEqual(mapped.ratio, 1.0)
        self.assertEqual(mapped.time_sec, 60.0)
        self.assertEqual(mapped.snapped_time_sec, 60.0)

    def test_header_region_covering_total_is_treated_as_prefix(self) -> None:
        mapped = map_byte_start_to_time(
            start=99,
            total=100,
            duration_sec=60.0,
            header_bytes=200,
            output_fps=24.0,
            gop_frames=48,
        )

        self.assertTrue(mapped.prefix)
        self.assertEqual(mapped.header_bytes, 100)
        self.assertEqual(mapped.time_sec, 0.0)
        self.assertEqual(mapped.snapped_time_sec, 0.0)

    def test_negative_gop_frames_disable_snap(self) -> None:
        mapped = map_byte_start_to_time(
            start=76,
            total=101,
            duration_sec=100.0,
            header_bytes=1,
            output_fps=25.0,
            gop_frames=-1,
        )

        self.assertEqual(mapped.gop_seconds, 0.0)
        self.assertAlmostEqual(mapped.time_sec, 75.0, places=3)
        self.assertAlmostEqual(mapped.snapped_time_sec, mapped.time_sec, places=3)

    def test_near_end_request_stays_within_duration(self) -> None:
        mapped = map_byte_start_to_time(
            start=200,
            total=201,
            duration_sec=60.0,
            header_bytes=1,
            output_fps=30.0,
            gop_frames=60,
        )

        self.assertLessEqual(mapped.ratio, 1.0)
        self.assertLessEqual(mapped.time_sec, 60.0)
        self.assertLessEqual(mapped.snapped_time_sec, 60.0)

    def test_gop_helpers_tolerate_disabled_values(self) -> None:
        self.assertEqual(gop_duration_seconds(0, 30), 0.0)
        self.assertEqual(gop_duration_seconds(60, 0), 0.0)
        self.assertEqual(snap_back_to_gop(5.5, 0), 5.5)


if __name__ == "__main__":
    unittest.main()
