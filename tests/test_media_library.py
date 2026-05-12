from __future__ import annotations

import unittest
from pathlib import Path

from media_library import MediaLibrary, build_media_roots, parse_video_dirs


class MediaLibraryTests(unittest.TestCase):
    def test_parse_pipe_separated_video_dirs(self) -> None:
        roots = parse_video_dirs(r"D:\VR|E:\VR", Path("videos"))

        self.assertEqual(len(roots), 2)
        self.assertTrue(str(roots[0]).endswith("D:\\VR"))
        self.assertTrue(str(roots[1]).endswith("E:\\VR"))

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


if __name__ == "__main__":
    unittest.main()
