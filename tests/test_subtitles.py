from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from media_library import MediaLibrary, build_media_roots
from utils.subtitles import find_external_subtitles, subtitle_mime


class SubtitleDiscoveryTests(unittest.TestCase):
    def test_same_stem_subtitles_are_sorted_by_language_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            (root / "movie.eng.srt").write_text("one", encoding="utf-8")
            (root / "movie.zh.ass").write_text("two", encoding="utf-8")
            (root / "movie.srt").write_text("three", encoding="utf-8")
            library = MediaLibrary(build_media_roots([root]))

            with patch("utils.subtitles.config.MEDIA_LIBRARY", library):
                tracks = find_external_subtitles(video)

        self.assertEqual([track.path.name for track in tracks], ["movie.srt", "movie.zh.ass", "movie.eng.srt"])
        self.assertEqual([track.lang for track in tracks], ["", "zh", "eng"])
        self.assertEqual(tracks[1].mime, "application/x-ass")

    def test_subtitle_discovery_respects_enable_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            (root / "movie.srt").write_text("one", encoding="utf-8")
            library = MediaLibrary(build_media_roots([root]))

            with (
                patch("utils.subtitles.config.MEDIA_LIBRARY", library),
                patch("utils.subtitles.config.SUBTITLE_ENABLE", False),
            ):
                tracks = find_external_subtitles(video)

        self.assertEqual(tracks, [])

    def test_subtitle_mime_defaults(self) -> None:
        self.assertEqual(subtitle_mime(Path("movie.srt")), "application/x-subrip")
        self.assertEqual(subtitle_mime(Path("movie.vtt")), "text/vtt")


if __name__ == "__main__":
    unittest.main()
