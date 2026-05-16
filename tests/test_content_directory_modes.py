from __future__ import annotations

import unittest
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

    def test_live_ids_distinguish_alpha(self) -> None:
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._passthrough_live_prefix("green"), "pl_")
            self.assertEqual(cds._passthrough_live_prefix("alpha"), "pla_")
            self.assertEqual(cds._passthrough_live_item_prefix("green"), "lg_")
            self.assertEqual(cds._passthrough_live_item_prefix("alpha"), "la_")
            self.assertEqual(cds._passthrough_live_query("green"), "mode=green")
            self.assertEqual(cds._passthrough_live_query("alpha"), "mode=alpha")

    def test_short_live_items_keep_distinct_modes(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
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

        passthrough = [item for item in items if item.get("passthrough")]
        self.assertEqual([item["id"] for item in passthrough], ["lg_movie.mp4", "la_movie.mp4"])
        self.assertIn("mode=green", passthrough[0]["url"])
        self.assertIn("mode=alpha", passthrough[1]["url"])
        self.assertEqual([item["passthrough_mode"] for item in passthrough], ["green", "alpha"])

    def test_short_live_metadata_keeps_alpha_mode(self) -> None:
        source = Path("movie.mp4")
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
                resolution="3840x2160",
                backend_verdict="pynv_hevc",
                probe_error="",
                mkv_needs_fix=False,
            ),
        )
        info = SimpleNamespace(duration=60.0, width=3840, height=2160)
        with (
            patch.object(cds, "_rel_key", return_value="movie.mp4"),
            patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"),
            patch.object(cds, "_uses_live_chapter_container", return_value=False),
            patch.object(cds, "probe_cached", return_value=info),
            patch.object(cds, "estimate_for_media", return_value=(0, 20_000_000, None)),
            patch.object(cds, "find_external_subtitles", return_value=[]),
        ):
            didl = cds._metadata_didl_for_item(
                [
                    item for item in cds._video_items_from_index(source, "0", child)
                    if item.get("passthrough") and item.get("passthrough_mode") == "alpha"
                ][0]
            )

        self.assertIn("la_movie.mp4", didl)
        self.assertIn("mode=alpha", didl)
        self.assertNotIn("mode=green", didl)

    def test_alpha_virtual_title_uses_file_name(self) -> None:
        self.assertEqual(cds._passthrough_virtual_title(Path("movie.mp4"), "alpha"), "movie_FISHEYE180_alpha_live")

    def test_mkv_needs_fix_hides_passthrough_without_marking_title(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
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
        self.assertEqual(items[0]["title"], "movie")

    def test_mkv_live_passthrough_is_hidden_by_default_policy(self) -> None:
        child = SimpleNamespace(
            size=1024,
            video=SimpleNamespace(
                duration=60.0,
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
        self.assertEqual(items[0]["title"], "movie")

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

        self.assertEqual(
            [item["title"] for item in items],
            ["00:00_movie_FISHEYE180_alpha_live", "00:05_movie_FISHEYE180_alpha_live"],
        )

    def test_multi_root_items_are_virtual_folders(self) -> None:
        roots = build_media_roots([Path(r"D:\VR"), Path(r"E:\VR")])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library), patch.object(cds, "_child_count", return_value=0):
            items = cds._root_items()

        self.assertEqual([item["title"] for item in items], ["VR", "VR2"])
        self.assertEqual([item["id"] for item in items], ["d_VR", "d_VR2"])

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

    def test_legacy_colon_folder_id_still_resolves(self) -> None:
        roots = build_media_roots([Path(r"D:\VR")])
        library = MediaLibrary(roots)
        with patch.object(cds, "MEDIA_LIBRARY", library):
            self.assertEqual(cds._id_to_dir("d:Movies"), Path(r"D:\VR\Movies").resolve())

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
