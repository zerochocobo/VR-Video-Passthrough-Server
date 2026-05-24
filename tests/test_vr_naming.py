from __future__ import annotations

import unittest

from utils.vr_naming import (
    alpha_passthrough_stem,
    has_vr_filename_marker,
    live_passthrough_title,
    offline_passthrough_stem,
    source_display_stem,
)


class VrNamingTests(unittest.TestCase):
    def test_detects_player_suffixes(self) -> None:
        for name in [
            "scene_LR",
            "scene-RLF",
            "scene_HSBS",
            "scene-Half-SBS",
            "scene_Left_Right",
            "scene-TB",
            "scene_HOU",
            "scene-Half-OU",
            "scene_Top_Bottom",
            "scene_F180_alpha",
            "scene_FISHEYE190",
            "scene_RF52",
            "scene_MKX200",
            "scene_MKX22",
            "scene_VRCA220",
            "scene_EAC360",
            "scene_360EAC",
        ]:
            with self.subTest(name=name):
                self.assertTrue(has_vr_filename_marker(name))

    def test_does_not_treat_embedded_words_as_suffixes(self) -> None:
        self.assertFalse(has_vr_filename_marker("holiday180clip"))
        self.assertFalse(has_vr_filename_marker("alphauser"))
        self.assertFalse(has_vr_filename_marker("scene F180"))
        self.assertFalse(has_vr_filename_marker("scene - F180"))

    def test_half_equirectangular_source_gets_lr_180_display_suffix(self) -> None:
        self.assertEqual(source_display_stem("movie", 3840, 1920), "movie_LR_180_SBS")
        self.assertEqual(source_display_stem("movie_180", 3840, 1920), "movie_180")
        self.assertEqual(source_display_stem("movie", 1920, 1080), "movie")
        self.assertEqual(source_display_stem("movie_LR_180", 3840, 1920), "movie_LR_180_SBS")
        self.assertEqual(source_display_stem("movie_LR_180_FISHEYE", 3840, 1920), "movie_LR_180_FISHEYE")
        self.assertEqual(source_display_stem("movie_LR_180_SBS", 3840, 1920), "movie_LR_180_SBS")

    def test_stem_with_dots_is_not_truncated(self) -> None:
        stem = "xxxx.com@atvr00067_1_8k"
        self.assertEqual(source_display_stem(stem, 7680, 3840), f"{stem}_LR_180_SBS")
        self.assertEqual(source_display_stem(f"{stem}.mp4", 7680, 3840), f"{stem}_LR_180_SBS")

    def test_non_video_dotted_suffix_is_part_of_stem(self) -> None:
        self.assertFalse(has_vr_filename_marker("scene.F180"))
        self.assertEqual(source_display_stem("scene.F180", 3840, 1920), "scene.F180_LR_180_SBS")

    def test_live_titles_use_green_base_and_alpha_original_base(self) -> None:
        self.assertEqual(live_passthrough_title("movie", "green", 3840, 1920), "movie_LR_180_SBS_passthrough_live")
        self.assertEqual(live_passthrough_title("movie", "alpha", 3840, 1920), "movie_LR_180_FISHEYE_F180_alpha_live")

    def test_alpha_rewrites_fisheye_to_f180(self) -> None:
        self.assertEqual(alpha_passthrough_stem("movie_FISHEYE"), "movie_FISHEYE_LR_180_FISHEYE_F180_alpha")

    def test_offline_stems_use_default_player_suffixes(self) -> None:
        self.assertEqual(offline_passthrough_stem("movie", "green", 3840, 1920), "movie_LR_180_SBS_passthrough")
        self.assertEqual(offline_passthrough_stem("movie", "green", 1920, 1080), "movie_passthrough")
        self.assertEqual(offline_passthrough_stem("movie", "alpha"), "movie_LR_180_FISHEYE_F180_alpha")


if __name__ == "__main__":
    unittest.main()
