from __future__ import annotations

import unittest
from pathlib import Path

from utils.offline_outputs import (
    discard_pending_output,
    has_offline_passthrough_output,
    is_internal_intermediate_name,
    is_offline_passthrough_output_name,
    pending_output_path,
    publish_pending_output,
    matches_offline_output_for_source,
    matches_offline_two_dvr_output_for_source,
)


class OfflineOutputDetectionTests(unittest.TestCase):
    def test_detects_default_and_segment_outputs_for_source(self) -> None:
        source = Path("movie.mp4")

        self.assertTrue(matches_offline_output_for_source(source, Path("movie_passthrough.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_LR_180_SBS_passthrough.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_LR_180_passthrough.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_SBS_180_passthrough.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_FISHEYE_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_LR_180_FISHEYE_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_LR_180_FISHEYE_F180_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_SBS_F180_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_rvm1_S000000_ALL_FISHEYE_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_rvm1_S000000_ALL_LR_180_FISHEYE_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_rvm1_S000000_ALL_LR_180_FISHEYE_F180_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_rvm1_S000000_ALL_SBS_F180_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_matanyone2_S000005_5M_3D_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(source, Path("movie_matanyone2_S000005_E000505_5M_3D_alpha.mp4")))

    def test_matches_alpha_output_that_dropped_the_source_projection_marker(self) -> None:
        source = Path("TEST_4096p_91983_FISHEYE190_x265.mp4")
        self.assertTrue(matches_offline_output_for_source(
            source, Path("TEST_4096p_91983_x265_LR_180_FISHEYE_F180_alpha.mp4")))
        self.assertTrue(matches_offline_output_for_source(
            source, Path("TEST_4096p_91983_x265_rvm1_S000000_ALL_LR_180_FISHEYE_F180_alpha.mp4")))
        # The legacy name, still produced by older builds, keeps matching.
        self.assertTrue(matches_offline_output_for_source(
            source, Path("TEST_4096p_91983_FISHEYE190_x265_LR_180_FISHEYE_F180_alpha.mp4")))

    def test_rejects_unrelated_names(self) -> None:
        source = Path("movie.mp4")

        self.assertFalse(matches_offline_output_for_source(source, Path("movie2_FISHEYE_alpha.mp4")))
        self.assertFalse(matches_offline_output_for_source(source, Path("movie_notes.mp4")))
        self.assertFalse(matches_offline_output_for_source(source, source))

    def test_passthrough_output_name_suffixes(self) -> None:
        self.assertTrue(is_offline_passthrough_output_name("movie_FISHEYE180_alpha.mp4"))
        self.assertTrue(is_offline_passthrough_output_name("movie_FISHEYE190_alpha.mp4"))
        self.assertTrue(is_offline_passthrough_output_name("movie_LR_180_FISHEYE_alpha.mp4"))
        self.assertTrue(is_offline_passthrough_output_name("movie_LR_180_FISHEYE_F180_alpha.mp4"))
        self.assertTrue(is_offline_passthrough_output_name("movie_SBS_F180_alpha.mp4"))
        self.assertFalse(is_offline_passthrough_output_name("movie_alpha_notes.mp4"))

    def test_detects_two_dvr_outputs_for_source(self) -> None:
        source = Path("movie.mp4")

        self.assertTrue(matches_offline_two_dvr_output_for_source(source, Path("movie_3D_LR_Screen.mp4")))
        self.assertTrue(matches_offline_two_dvr_output_for_source(source, Path("movie_S000130_3D_LR_Screen.mp4")))
        self.assertTrue(matches_offline_two_dvr_output_for_source(source, Path("movie_S000130_E000200_3D_LR_Screen.mp4")))
        self.assertTrue(matches_offline_two_dvr_output_for_source(source, Path("movie_SEG2_S000130_E000505_3D_LR_Screen.mp4")))
        # Legacy flat3d SBS output naming is still recognized.
        self.assertTrue(matches_offline_two_dvr_output_for_source(source, Path("movie_2dvr_base_flat3d_LR_SBS.mp4")))

    def test_two_dvr_rejects_prefix_collisions(self) -> None:
        source = Path("movie.mp4")

        # A different source whose stem starts with "movie_" must not be treated
        # as movie's own 2D->3D output.
        self.assertFalse(matches_offline_two_dvr_output_for_source(source, Path("movie_part2_3D_LR_Screen.mp4")))
        self.assertFalse(matches_offline_two_dvr_output_for_source(source, Path("movie_part2_2dvr_base_flat3d_LR_SBS.mp4")))
        self.assertFalse(matches_offline_two_dvr_output_for_source(source, source))

    def test_has_offline_output_accepts_snapshot_siblings(self) -> None:
        source = Path("movie.mp4")
        siblings = [
            source,
            Path("movie_rvm1_S000000_ALL_passthrough.mp4"),
        ]

        self.assertTrue(has_offline_passthrough_output(source, siblings))


class InternalIntermediateNameTests(unittest.TestCase):
    def test_detects_work_files_every_batch_scanner_must_skip(self) -> None:
        for name in (
            "movie_LR_180_SBS_passthrough._video_only.mp4",
            "movie_FISHEYE190_alpha._video_only.mp4",
            "movie_LR_180_SBS_passthrough._audio.4321.aac",
            "movie_restored.partial.4321.mp4",
            ".movie_2dvr_gpu_ab12.mp4",
            "movie.vmp4build.mp4",
            "movie.mp4.tmp",
        ):
            self.assertTrue(is_internal_intermediate_name(name), name)

    def test_keeps_sources_and_finished_outputs(self) -> None:
        for name in (
            "movie.mp4",
            "movie_LR_180_SBS_passthrough.mp4",
            "movie_FISHEYE190_alpha.mp4",
            "movie_2dvr_da3_FLAT3D_LR_SBS.mp4",
            "movie_restored.mp4",
            "movie_beauty.mp4",
        ):
            self.assertFalse(is_internal_intermediate_name(name), name)


class OfflineBatchScannerTests(unittest.TestCase):
    """Every offline batch mode shares one rule for in-flight work files."""

    def test_all_batch_scanners_skip_intermediates(self) -> None:
        import tempfile

        from offline.convert import _video_files as convert_video_files
        from offline.demosaic_offline import _video_files as rm_video_files
        from offline.face_beauty import _video_files as beauty_video_files
        from offline.two_dvr import _video_files as two_dvr_video_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "movie.mp4"
            source.write_bytes(b"")
            for leftover in (
                "movie_LR_180_SBS_passthrough._video_only.mp4",
                "movie_FISHEYE190_alpha._video_only.mp4",
                "movie_restored.partial.4321.mp4",
                ".movie_2dvr_gpu_ab12.mp4",
            ):
                (root / leftover).write_bytes(b"")

            for scanner in (convert_video_files, two_dvr_video_files, rm_video_files, beauty_video_files):
                self.assertEqual(
                    scanner(root, False),
                    [source],
                    scanner.__module__,
                )


class PendingOutputTests(unittest.TestCase):
    """An offline run writes here and renames on success, so the final name
    only ever exists as a complete file."""

    def test_pending_path_sits_next_to_the_output_and_is_skipped_by_scanners(self) -> None:
        import os

        out = Path("/videos/movie_restored.mp4")
        pending = pending_output_path(out)

        self.assertEqual(pending.parent, out.parent)
        self.assertEqual(pending.suffix, out.suffix)
        self.assertIn(str(os.getpid()), pending.name)
        self.assertTrue(is_internal_intermediate_name(pending.name))

    def test_publish_renames_onto_the_final_name(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "movie_restored.mp4"
            pending = pending_output_path(out)
            pending.write_bytes(b"finished")

            self.assertTrue(publish_pending_output(pending, out))
            self.assertFalse(pending.exists())
            self.assertEqual(out.read_bytes(), b"finished")

    def test_publish_replaces_an_earlier_output(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "movie_restored.mp4"
            out.write_bytes(b"stale")
            pending = pending_output_path(out)
            pending.write_bytes(b"fresh")

            self.assertTrue(publish_pending_output(pending, out))
            self.assertEqual(out.read_bytes(), b"fresh")

    def test_discard_leaves_no_output_behind(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "movie_restored.mp4"
            pending = pending_output_path(out)
            pending.write_bytes(b"truncated")

            discard_pending_output(pending)

            self.assertFalse(pending.exists())
            self.assertFalse(out.exists())
        discard_pending_output(None)


if __name__ == "__main__":
    unittest.main()
