from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from media_library import MediaLibrary, build_media_roots, parse_video_dirs
from utils.media_index import MediaIndex


class MediaLibraryTests(unittest.TestCase):
    def test_parse_pipe_separated_video_dirs(self) -> None:
        roots = parse_video_dirs(r"D:\VR|E:\VR", Path("videos"))

        self.assertEqual(len(roots), 2)
        self.assertTrue(str(roots[0]).endswith("D:\\VR"))
        self.assertTrue(str(roots[1]).endswith("E:\\VR"))

    def test_parse_skips_unusable_video_dir(self) -> None:
        original_resolve = Path.resolve

        def fake_resolve(path: Path, *args, **kwargs) -> Path:
            if str(path) == r"R:\PikPak":
                raise OSError(1005, "bad rclone mount")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", autospec=True, side_effect=fake_resolve):
            roots = parse_video_dirs(r"R:\PikPak|D:\VR", Path("videos"))

        self.assertEqual(len(roots), 1)
        self.assertTrue(str(roots[0]).endswith("D:\\VR"))

    def test_parse_falls_back_when_all_video_dirs_are_unusable(self) -> None:
        default = Path("videos")
        expected = default.resolve()
        original_resolve = Path.resolve

        def fake_resolve(path: Path, *args, **kwargs) -> Path:
            if str(path) == r"R:\PikPak":
                raise OSError(1005, "bad rclone mount")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", autospec=True, side_effect=fake_resolve):
            roots = parse_video_dirs(r"R:\PikPak", default)

        self.assertEqual(roots, [expected])

    def test_duplicate_names_are_numbered(self) -> None:
        roots = build_media_roots([Path(r"D:\VR"), Path(r"E:\VR"), Path(r"F:\Movies")])

        self.assertEqual([root.label for root in roots], ["VR", "VR2", "Movies"])

    def test_multi_root_virtual_key_roundtrip(self) -> None:
        roots = build_media_roots([Path(r"D:\VR"), Path(r"E:\VR")])
        library = MediaLibrary(roots)

        self.assertEqual(library.path_to_key(Path(r"E:\VR\demo.mp4")), "VR2/demo.mp4")
        self.assertEqual(library.key_to_path("VR2/demo.mp4"), Path(r"E:\VR\demo.mp4").resolve())

    def test_key_to_path_rejects_absolute_key(self) -> None:
        library = MediaLibrary(build_media_roots([Path(r"D:\VR")]))

        self.assertIsNone(library.key_to_path(r"C:\Windows\notepad.exe"))

    def test_multi_root_key_to_path_rejects_absolute_rest(self) -> None:
        library = MediaLibrary(build_media_roots([Path(r"D:\VR"), Path(r"E:\VR")]))

        self.assertIsNone(library.key_to_path(r"VR2/C:/Windows/notepad.exe"))

    def test_media_index_includes_images_when_dlna_images_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            photo = root / "photo.jpg"
            photo.write_bytes(b"image")
            library = MediaLibrary(build_media_roots([root]))
            index = MediaIndex(root / "index.db")
            try:
                with (
                    patch("utils.media_index.config.MEDIA_LIBRARY", library),
                    patch("utils.media_index.config.DLNA_IMAGE_ENABLED", True),
                ):
                    snapshot = index.list_directory(root)
            finally:
                index.close()

            self.assertEqual([child.name for child in snapshot.children], ["photo.jpg"])
            self.assertIsNone(snapshot.children[0].video)

    def test_media_index_signature_tracks_si_sidecar_without_listing_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            movie = root / "movie.mp4"
            movie.write_bytes(b"video")
            library = MediaLibrary(build_media_roots([root]))
            index = MediaIndex(root / "index.db")
            try:
                with patch("utils.media_index.config.MEDIA_LIBRARY", library):
                    before = index.list_directory(root)
                    movie.with_suffix(".si.wav").write_bytes(b"si")
                    after = index.list_directory(root)
            finally:
                index.close()

            self.assertEqual([child.name for child in before.children], ["movie.mp4"])
            self.assertEqual([child.name for child in after.children], ["movie.mp4"])
            self.assertNotEqual(before.signature, after.signature)


if __name__ == "__main__":
    unittest.main()
