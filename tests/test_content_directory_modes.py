from __future__ import annotations

import unittest
import shutil
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import dlna.content_directory as cds
from media_library import MediaLibrary, build_media_roots


class ContentDirectoryModeTests(unittest.TestCase):
    def test_passthrough_modes(self) -> None:
        cases = {
            "none": (),
            "green": ("green",),
            "alpha": ("alpha",),
            "all": ("green", "alpha"),
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode), patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", mode):
                self.assertEqual(cds._passthrough_modes(), expected)

    def test_video_item_count(self) -> None:
        source = Path("movie.mp4")
        derived = Path("movie_passthrough.mp4")
        needs_fix = SimpleNamespace(video=SimpleNamespace(mkv_needs_fix=True))
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "none"):
            self.assertEqual(cds._video_item_count(source), 1)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"):
            self.assertEqual(cds._video_item_count(source), 2)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._video_item_count(source), 3)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._video_item_count(derived), 1)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._video_item_count(Path("movie.mkv"), needs_fix), 1)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"), patch.object(cds, "has_offline_passthrough_output", return_value=True):
            self.assertEqual(cds._video_item_count(source), 1)
        with (
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"),
            patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", True),
            patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True),
        ):
            self.assertEqual(cds._video_item_count(source), 3)
        with (
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
            patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", True),
            patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True),
        ):
            self.assertEqual(cds._video_item_count(source), 5)

    def test_live_ids_distinguish_alpha(self) -> None:
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._passthrough_live_prefix("green"), "pl_")
            self.assertEqual(cds._passthrough_live_prefix("alpha"), "pla_")
            self.assertEqual(cds._passthrough_live_item_prefix("green"), "lg_")
            self.assertEqual(cds._passthrough_live_item_prefix("alpha"), "la_")
            self.assertIn("mode=green", cds._passthrough_live_query("green"))
            self.assertIn("mode=alpha", cds._passthrough_live_query("alpha"))

    def test_short_live_items_keep_distinct_modes(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                fps=24.0,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
            patch.object(cds, "_uses_live_chapter_container", return_value=False),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        live = [item for item in items if item.get("container") and str(item["id"]).startswith(("pl_", "pla_"))]
        self.assertEqual([item["id"] for item in live], ["pl_ptv10_movie.mp4", "pla_ptv10_movie.mp4"])
        self.assertEqual([item["title"] for item in live], ["[GREEN]_movie_passthrough_live", "[ALPHA]_movie_LR_180_FISHEYE_F180_alpha_live"])
        self.assertEqual([item["child_count"] for item in live], [2, 2])

    def test_short_live_entry_is_directory(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                fps=24.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"),
            patch.object(cds, "_uses_live_chapter_container", return_value=False),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
            patch("dlna.profiles.PASSTHROUGH_MAX_FPS", 30.0),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        live = [item for item in items if item.get("container") and item["id"] == "pl_ptv10_movie.mp4"]
        self.assertEqual(live[0]["child_count"], 2)

    def test_seek_dlna_switch_adds_live_fallback_virtual_entry(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=600.0,
                fps=30.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"),
            patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", True),
            patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True),
            patch.object(cds, "PASSTHROUGH_SEEK_HEADER_BYTES", 2_000_000),
            patch.object(cds, "_uses_live_chapter_container", return_value=True),
            patch.object(cds, "estimate_for_media", return_value=(12_345_678, 20_000_000, None)),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        self.assertEqual(len(items), 3)
        seek = next(item for item in items if "/passthrough_seek/" in item.get("url", ""))
        live = next(item for item in items if item.get("container"))
        self.assertFalse(seek.get("container"))
        self.assertEqual(seek["id"], "sg_ptv10_movie.mp4")
        self.assertIn("/passthrough_seek/movie.mp4.seek.ts", seek["url"])
        self.assertIn("_seek", seek["title"])
        self.assertEqual(seek["protocol_info"].split(";")[1], "DLNA.ORG_OP=11")
        self.assertIn("DLNA.ORG_CI=0", seek["protocol_info"])
        self.assertIn("DLNA.ORG_FLAGS=01F00000000000000000000000000000", seek["protocol_info"])
        self.assertEqual(seek["size"], 14_345_678)
        self.assertEqual(live["id"], "pl_ptv10_movie.mp4")
        self.assertIn("_live", live["title"])

    def test_seek_dlna_requires_route_master_switch(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=600.0,
                fps=30.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"),
            patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", False),
            patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True),
            patch.object(cds, "_uses_live_chapter_container", return_value=True),
            patch.object(cds, "estimate_for_media", return_value=(12_345_678, 20_000_000, None)),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        self.assertTrue(items[1].get("container"))
        self.assertEqual(items[1]["id"], "pl_ptv10_movie.mp4")

    def test_seek_dlna_can_advertise_true_fmp4_experiment(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                fps=30.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"),
            patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", True),
            patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True),
            patch.object(cds, "PASSTHROUGH_SEEK_CONTAINER", "mp4"),
            patch.object(cds, "_uses_live_chapter_container", return_value=False),
            patch.object(cds, "estimate_for_media", return_value=(12_345_678, 20_000_000, None)),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        seek = next(item for item in items if "/passthrough_seek/" in item.get("url", ""))
        self.assertEqual(seek["mime"], "video/mp4")
        self.assertEqual(seek["dlna_pn"], "HEVC_MP4_MAIN")
        self.assertIn("/passthrough_seek/movie.mp4.seek.mp4", seek["url"])
        self.assertIn("http-get:*:video/mp4:DLNA.ORG_PN=HEVC_MP4_MAIN", seek["protocol_info"])

    def test_existing_offline_output_hides_virtual_modes(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
            patch.object(cds, "has_offline_passthrough_output", return_value=True),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        self.assertEqual(len(items), 1)
        self.assertFalse(items[0].get("passthrough"))

    def test_short_live_metadata_is_alpha_directory(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=60.0, width=3840, height=2160)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
        ):
            didl = cds._metadata_didl_for_live(source, "alpha")

        self.assertIn("pla_ptv10_movie.mp4", didl)
        self.assertIn("<container", didl)
        self.assertIn('childCount="2"', didl)
        self.assertIn("[ALPHA]_movie_LR_180_FISHEYE_F180_alpha_live", didl)

    def test_alpha_virtual_title_uses_file_name(self) -> None:
        self.assertEqual(cds._passthrough_virtual_title(Path("movie.mp4"), "alpha"), "movie_LR_180_FISHEYE_F180_alpha_live")

    def test_green_virtual_title_uses_half_equirectangular_source_display_name(self) -> None:
        self.assertEqual(
            cds._passthrough_virtual_title(Path("movie.mp4"), "green", 3840, 1920),
            "movie_LR_180_SBS_passthrough_live",
        )

    def test_original_title_uses_half_equirectangular_source_display_name(self) -> None:
        child = SimpleNamespace(
            video=SimpleNamespace(
                width=3840,
                height=1920,
                resolution="3840x1920",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            )
        )
        self.assertEqual(cds._marked_original_title(Path("movie.mp4"), child), "movie_LR_180_SBS")

    def test_offline_passthrough_output_title_is_marked(self) -> None:
        child = SimpleNamespace(
            video=SimpleNamespace(
                width=3840,
                height=1920,
                resolution="3840x1920",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            )
        )
        self.assertEqual(
            cds._marked_original_title(Path("movie_LR_180_SBS_passthrough.mp4"), child),
            "[Offline] movie_LR_180_SBS_passthrough",
        )
        self.assertEqual(
            cds._marked_original_title(Path("movie_LR_180_FISHEYE_F180_alpha.mp4"), child),
            "[Offline] movie_LR_180_FISHEYE_F180_alpha",
        )

    def test_mkv_needs_fix_hides_passthrough_and_marks_title(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=True,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mkv"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
            patch.object(cds, "PASSTHROUGH_MKV_LIVE_POLICY", "head_cues"),
        ):
            items = cds._video_items_from_index(Path("movie.mkv"), "0", child)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "[NoLive] movie")

    def test_mkv_live_passthrough_is_hidden_by_default_policy(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mkv"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
            patch.object(cds, "PASSTHROUGH_MKV_LIVE_POLICY", "block"),
        ):
            items = cds._video_items_from_index(Path("movie.mkv"), "0", child)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "[NoLive] movie")

    def test_non_pynv_backend_hides_passthrough_and_marks_title(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                width=3840,
                height=2160,
                resolution="3840x2160",
                backend_verdict="ffmpeg_fallback",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "[NoLive] movie")

    def test_oversized_video_hides_passthrough_and_marks_title(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                width=9000,
                height=4096,
                resolution="9000x4096",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "[NoLive] movie")

    def test_probe_error_hides_passthrough_and_marks_title(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=0.0,
                width=0,
                height=0,
                resolution="",
                backend_verdict="",
                probe_error="pending",
                mkv_needs_fix=False,
            ),
        )
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
        ):
            items = cds._video_items_from_index(Path("movie.mp4"), "0", child)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "[NoLive] movie")

    def test_live_chapter_titles_sort_by_time(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=720.0, width=3840, height=2160)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
            patch.object(cds, "_live_chapter_offsets", return_value=[0, 300]),
        ):
            items = cds._live_chapter_items(source, "alpha")

        chapters = [item for item in items if not item.get("container")]
        self.assertEqual(
            [item["title"] for item in chapters],
            ["00:00_movie_LR_180_FISHEYE_F180_alpha_live", "00:05_movie_LR_180_FISHEYE_F180_alpha_live"],
        )
        self.assertEqual(items[0]["id"], "lix_a_ptv10_movie.mp4")
        self.assertEqual(items[0]["title"], "[Select Time Index]_[ALPHA]_movie_LR_180_FISHEYE_F180_alpha_live")

    def test_live_time_index_title_uses_requested_language(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=720.0, width=3840, height=2160)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
        ):
            items = cds._live_chapter_items(source, "green", language="zh_CN")

        self.assertEqual(items[0]["title"], "[选择时间索引]_[GREEN]_movie_passthrough_live")

    def test_live_time_index_groups_minutes_and_five_second_links(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=3330.0, width=3840, height=2160, fps=24.0)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
        ):
            groups = cds._live_time_index_items(source, "green", "index")
            minutes = cds._live_time_index_items(source, "green", "group", 600, 1200)
            points = cds._live_time_index_items(source, "green", "minute", 660)

        self.assertEqual(groups[0]["title"], "00:00-10:00_[GREEN]_movie_passthrough_live")
        self.assertEqual(groups[1]["title"], "10:00-20:00_[GREEN]_movie_passthrough_live")
        self.assertEqual(groups[-1]["title"], "50:00-55:30_[GREEN]_movie_passthrough_live")
        self.assertEqual(groups[-1]["child_count"], 6)
        self.assertEqual(
            [item["title"] for item in minutes[:3]],
            ["10:00_[GREEN]_movie_passthrough_live", "11:00_[GREEN]_movie_passthrough_live", "12:00_[GREEN]_movie_passthrough_live"],
        )
        self.assertEqual(
            [item["title"] for item in points[:3]],
            ["11:00_movie_passthrough_live", "11:05_movie_passthrough_live", "11:10_movie_passthrough_live"],
        )
        self.assertIn("/passthrough_live/movie.mp4.ts?t=660&mode=green", points[0]["url"])

    def test_live_time_index_single_group_skips_group_layer(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=95.0, width=3840, height=2160, fps=24.0)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
        ):
            items = cds._live_time_index_items(source, "green", "index")

        self.assertEqual([item["id"] for item in items], ["lim_g_ptv10_movie.mp4@0", "lim_g_ptv10_movie.mp4@60"])
        self.assertEqual([item["title"] for item in items], ["00:00_[GREEN]_movie_passthrough_live", "01:00_[GREEN]_movie_passthrough_live"])

    def test_live_time_index_uses_hour_format_for_long_videos(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=3665.0, width=3840, height=2160, fps=24.0)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
        ):
            groups = cds._live_time_index_items(source, "green", "index")

        self.assertEqual(groups[0]["title"], "0:00:00-0:10:00_[GREEN]_movie_passthrough_live")
        self.assertEqual(groups[-1]["title"], "1:00:00-1:01:05_[GREEN]_movie_passthrough_live")

    def test_live_chapter_res_metadata_is_not_seekable_vod(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=720.0, width=3840, height=2160)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
            patch.object(cds, "_live_chapter_offsets", return_value=[0, 300]),
        ):
            didl = cds._didl_for(cds._live_chapter_items(source, "green"))

        self.assertIn("DLNA.ORG_OP=00", didl)
        self.assertNotIn("DLNA.ORG_OP=10", didl)
        self.assertNotIn("DLNA.ORG_FLAGS", didl)
        self.assertNotIn("duration=", didl)
        self.assertNotIn("bitrate=", didl)
        self.assertIn('resolution="3840x2160"', didl)

    def test_deovr_short_live_chapter_item_uses_legacy_cds_shape(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=60.0, width=3840, height=2160, fps=24.0)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
        ):
            live = [item for item in cds._live_chapter_items(source, "green", client_profile="deovr") if not item.get("container")][0]
            didl = cds._metadata_didl_for_item(live)

        self.assertIn("/passthrough_live/movie.mp4?t=0&amp;mode=green", didl)
        self.assertNotIn("/passthrough_live/movie.mp4.ts", didl)
        self.assertIn("DLNA.ORG_OP=10", didl)
        self.assertIn("DLNA.ORG_FLAGS=41700000000000000000000000000000", didl)
        self.assertIn('duration="0:01:00.000"', didl)
        self.assertIn('bitrate="20000000"', didl)

    def test_deovr_live_chapter_items_use_legacy_cds_shape(self) -> None:
        source = Path("movie.mp4")
        info = SimpleNamespace(duration=720.0, width=3840, height=2160)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
            patch.object(cds, "_live_chapter_offsets", return_value=[0, 300]),
        ):
            didl = cds._didl_for(cds._live_chapter_items(source, "green", client_profile="deovr"))

        self.assertIn("/passthrough_live/movie.mp4?t=0&amp;mode=green", didl)
        self.assertNotIn("/passthrough_live/movie.mp4.ts", didl)
        self.assertIn("DLNA.ORG_OP=10", didl)
        self.assertIn("DLNA.ORG_FLAGS=41700000000000000000000000000000", didl)
        self.assertIn('duration="0:12:00.000"', didl)
        self.assertIn('bitrate="20000000"', didl)

    def test_didl_res_url_escapes_query_ampersands(self) -> None:
        didl = cds._didl_for(
            [
                {
                    "id": "lg_movie.mp4",
                    "parent_id": "0",
                    "title": "movie",
                    "url": "http://127.0.0.1:8200/passthrough_live/movie.mp4?t=0&mode=green&ptv=7",
                    "thumb": "http://127.0.0.1:8200/thumb/movie.mp4",
                    "size": 0,
                    "duration": 60.0,
                    "resolution": "3840x2160",
                    "bitrate": 20_000_000,
                    "mime": "video/MP2T",
                    "dlna_pn": "HEVC_TS_NA_ISO",
                    "frame_rate": None,
                    "passthrough": True,
                    "protocol_info": "http-get:*:video/MP2T:DLNA.ORG_PN=HEVC_TS_NA_ISO;DLNA.ORG_OP=00",
                    "subtitles": [],
                }
            ]
        )

        self.assertIn("?t=0&amp;mode=green&amp;ptv=7", didl)
        self.assertNotIn("?t=0&mode=green&ptv=7", didl)

    def test_multi_root_items_are_virtual_folders(self) -> None:
        roots = build_media_roots([Path(r"D:\VR"), Path(r"E:\VR")])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library), patch.object(cds, "_child_count", return_value=0):
            items = cds._root_items()

        self.assertEqual([item["title"] for item in items], ["VR", "VR2"])
        self.assertEqual([item["id"] for item in items], ["d_ptv10_VR", "d_ptv10_VR2"])

    def test_didl_namespace_has_trailing_slash(self) -> None:
        didl = cds._didl_for([])

        self.assertIn('xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/"', didl)
        self.assertNotIn('xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite" ', didl)
        self.assertIn('xmlns:sec="http://www.sec.co.kr/"', didl)

    def test_didl_includes_external_subtitles(self) -> None:
        didl = cds._didl_for(
            [
                {
                    "id": "v_movie.mp4",
                    "parent_id": "0",
                    "title": "movie",
                    "url": "http://127.0.0.1:8200/media/movie.mp4",
                    "thumb": "http://127.0.0.1:8200/thumb/movie.mp4",
                    "size": 1024,
                    "duration": 60.0,
                    "resolution": "1920x1080",
                    "bitrate": 1000,
                    "mime": "video/mp4",
                    "dlna_pn": "AVC_MP4_HP_HD_AAC",
                    "frame_rate": None,
                    "passthrough": False,
                    "subtitles": [
                        {
                            "url": "http://127.0.0.1:8200/subs/movie.zh.srt",
                            "lang": "zh",
                            "type": "srt",
                            "mime": "application/x-subrip",
                        }
                    ],
                }
            ]
        )

        self.assertIn('protocolInfo="http-get:*:application/x-subrip:*" xml:lang="zh"', didl)
        self.assertIn("<sec:CaptionInfoEx sec:type=\"srt\">http://127.0.0.1:8200/subs/movie.zh.srt</sec:CaptionInfoEx>", didl)
        self.assertIn("<sec:CaptionInfo sec:type=\"srt\">http://127.0.0.1:8200/subs/movie.zh.srt</sec:CaptionInfo>", didl)

    def test_directory_cache_key_includes_subtitle_toggle(self) -> None:
        child = SimpleNamespace(is_dir=False, path=Path("movie.mp4"))
        snapshot = SimpleNamespace(key="root", signature="sig", children=[child])
        video_item = {
            "id": "v_movie.mp4",
            "parent_id": "0",
            "title": "movie",
            "url": "http://127.0.0.1:8200/media/movie.mp4",
            "thumb": "http://127.0.0.1:8200/thumb/movie.mp4",
            "size": 1,
            "duration": 1.0,
            "resolution": "",
            "bitrate": 1,
            "mime": "video/mp4",
            "dlna_pn": "AVC_MP4_HP_HD_AAC",
            "frame_rate": None,
            "passthrough": False,
            "subtitles": [],
        }
        with (
            patch.object(cds, "get_media_index") as get_index,
            patch.object(cds, "_folder_id", return_value="0"),
            patch.object(cds, "_video_items_from_index", side_effect=[[dict(video_item, title="off")], [dict(video_item, title="on")]]),
            patch.object(cds, "subtitle_output_enabled", side_effect=[False, True]),
        ):
            get_index.return_value.list_directory.return_value = snapshot
            cds._dir_items_cache.clear()
            off_items = cds._children_for_dir(Path("."))
            on_items = cds._children_for_dir(Path("."))

        self.assertEqual(off_items[0]["title"], "off")
        self.assertEqual(on_items[0]["title"], "on")
        cds._dir_items_cache.clear()

    def test_children_for_dir_includes_image_items_when_enabled(self) -> None:
        child = SimpleNamespace(
            is_dir=False,
            path=Path("photo.jpg"),
            name="photo.jpg",
            size=1234,
        )
        snapshot = SimpleNamespace(key="root", signature="sig", children=[child])
        with (
            patch.object(cds, "DLNA_IMAGE_ENABLED", True),
            patch.object(cds, "get_media_index") as get_index,
            patch.object(cds, "_folder_id", return_value="0"),
            patch.object(cds, "_rel_key", return_value="photo.jpg"),
            patch.object(cds, "_image_resolution", return_value="640x480"),
        ):
            get_index.return_value.list_directory.return_value = snapshot
            cds._dir_items_cache.clear()
            items = cds._children_for_dir(Path("."))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "img_ptv10_photo.jpg")
        self.assertEqual(items[0]["mime"], "image/jpeg")
        self.assertEqual(items[0]["protocol_info"], "http-get:*:image/jpeg:DLNA.ORG_PN=JPEG_LRG;DLNA.ORG_OP=00;DLNA.ORG_CI=0;DLNA.ORG_FLAGS=01700000000000000000000000000000")
        didl = cds._didl_for(items)
        self.assertIn("<upnp:class>object.item.imageItem.photo</upnp:class>", didl)
        self.assertIn('protocolInfo="http-get:*:image/jpeg:', didl)
        self.assertIn("/media/photo.jpg", didl)
        cds._dir_items_cache.clear()

    def test_children_for_dir_skips_image_items_when_disabled(self) -> None:
        child = SimpleNamespace(
            is_dir=False,
            path=Path("photo.jpg"),
            name="photo.jpg",
            size=1234,
        )
        snapshot = SimpleNamespace(key="root", signature="sig", children=[child])
        with (
            patch.object(cds, "DLNA_IMAGE_ENABLED", False),
            patch.object(cds, "get_media_index") as get_index,
            patch.object(cds, "_folder_id", return_value="0"),
        ):
            get_index.return_value.list_directory.return_value = snapshot
            cds._dir_items_cache.clear()
            items = cds._children_for_dir(Path("."))

        self.assertEqual(items, [])
        cds._dir_items_cache.clear()

    def test_legacy_colon_folder_id_still_resolves(self) -> None:
        roots = build_media_roots([Path(r"D:\VR")])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library):
            self.assertEqual(cds._id_to_dir("d:Movies"), Path(r"D:\VR\Movies").resolve())

    def test_versioned_folder_id_resolves_without_visible_title_change(self) -> None:
        roots = build_media_roots([Path(r"D:\VR")])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library):
            self.assertEqual(cds._id_to_dir("d_ptv7_Movies"), Path(r"D:\VR\Movies").resolve())

    def test_versioned_live_id_resolves(self) -> None:
        source = Path("runtime_cache/test_content_directory_versioned_live/movie.mp4")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("video", encoding="utf-8")
        self.addCleanup(lambda: shutil.rmtree(source.parent, ignore_errors=True))
        roots = build_media_roots([source.parent])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library):
            self.assertEqual(cds._id_to_live("lg_ptv7_movie.mp4"), (source.resolve(), "green"))

    def test_versioned_live_time_index_ids_resolve(self) -> None:
        source = Path("runtime_cache/test_content_directory_versioned_live_time_index/movie.mp4")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("video", encoding="utf-8")
        self.addCleanup(lambda: shutil.rmtree(source.parent, ignore_errors=True))
        roots = build_media_roots([source.parent])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library):
            self.assertEqual(cds._id_to_live_time_index("lix_a_ptv7_movie.mp4"), (source.resolve(), "alpha", "index", 0, 0))
            self.assertEqual(cds._id_to_live_time_index("lig_g_ptv7_movie.mp4@600-1200"), (source.resolve(), "green", "group", 600, 1200))
            self.assertEqual(cds._id_to_live_time_index("lim_g_ptv7_movie.mp4@660"), (source.resolve(), "green", "minute", 660, 0))

    def test_soap_parser_rejects_entity_declarations(self) -> None:
        body = b"""<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY x "boom">]>
<s:Envelope><s:Body><ObjectID>&x;</ObjectID></s:Body></s:Envelope>"""

        self.assertEqual(cds._parse_soap_args(body), {})

    def test_soap_parser_rejects_oversized_body(self) -> None:
        body = b"<Envelope>" + (b"x" * (cds._MAX_SOAP_BODY_BYTES + 1)) + b"</Envelope>"

        self.assertEqual(cds._parse_soap_args(body), {})


if __name__ == "__main__":
    unittest.main()
