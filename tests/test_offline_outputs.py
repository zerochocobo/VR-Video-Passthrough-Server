from __future__ import annotations

import unittest
from pathlib import Path

from utils.offline_outputs import (
    has_offline_passthrough_output,
    is_offline_passthrough_output_name,
    matches_offline_output_for_source,
)


class OfflineOutputDetectionTests(unittest.TestCase):
    def test_detects_default_and_segment_outputs_for_source(self) -> None:
        source = Path("movie.mp4")

        self.assertTrue(matches_offline_output_for_source(source, Path("movie_passthrough.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_FISHEYE_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_rvm2_S000000_ALL_FISHEYE_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_matanyone2_S000005_5M_3D_alpha.mp4")))

    def test_rejects_unrelated_names(self) -> None:
        source = Path("movie.mp4")

        self.assertFalse(matches_offline_output_for_source(source, Path("movie2_FISHEYE_alpha.mp4")))
        self.assertFalse(matches_offline_output_for_source(source, Path("movie_notes.mp4")))
        self.assertFalse(matches_offline_output_for_source(source, source))

    def test_passthrough_output_name_suffixes(self) -> None:
        self.assertTrue(is_offline_passthrough_output_name("movie_FISHEYE180_alpha.mp4"))
        self.assertTrue(is_offline_passthrough_output_name("movie_FISHEYE190_alpha.mp4"))
        self.assertFalse(is_offline_passthrough_output_name("movie_alpha_notes.mp4"))

    def test_has_offline_output_accepts_snapshot_siblings(self) -> None:
        source = Path("movie.mp4")
        siblings = [
            source,
            Path("movie_rvm1_S000000_ALL_passthrough.mp4"),
        ]

        self.assertTrue(has_offline_passthrough_output(source, siblings))


if __name__ == "__main__":
    unittest.main()
