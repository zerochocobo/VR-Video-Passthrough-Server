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
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "none"):
            self.assertEqual(cds._video_item_count(source), 1)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green"):
            self.assertEqual(cds._video_item_count(source), 2)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._video_item_count(source), 3)
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._video_item_count(derived), 1)

    def test_live_ids_distinguish_alpha(self) -> None:
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "all"):
            self.assertEqual(cds._passthrough_live_prefix("green"), "pl_")
            self.assertEqual(cds._passthrough_live_prefix("alpha"), "pla_")
            self.assertEqual(cds._passthrough_live_query("green"), "mode=green")
            self.assertEqual(cds._passthrough_live_query("alpha"), "mode=alpha")

    def test_alpha_virtual_title_uses_file_name(self) -> None:
        self.assertEqual(cds._passthrough_virtual_title(Path("movie.mp4"), "alpha"), "movie_FISHEYE180_alpha_live")

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
