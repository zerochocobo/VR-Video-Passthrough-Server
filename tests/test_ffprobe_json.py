from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from ui.subtitle_preview import get_video_info
from utils.bitrate_estimator import source_video_bitrate
from utils.ffprobe_json import run_ffprobe_json


def _completed(cmd: list[str], stdout: dict, stderr: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    payload = json.dumps(stdout, ensure_ascii=False).encode("utf-8")
    return subprocess.CompletedProcess(cmd, returncode, stdout=payload, stderr=stderr)


class FfprobeJsonTests(unittest.TestCase):
    def test_decodes_utf8_json_bytes_without_text_mode(self) -> None:
        calls: list[dict] = []

        def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
            calls.append(kwargs)
            return _completed(cmd, {"format": {"tags": {"title": "日本語 中文"}}})

        with patch("utils.ffprobe_json.subprocess.run", side_effect=fake_run):
            data = run_ffprobe_json(["ffprobe", "-of", "json", "日本語.mp4"])

        self.assertEqual(data["format"]["tags"]["title"], "日本語 中文")
        self.assertNotIn("text", calls[0])
        self.assertNotIn("encoding", calls[0])
        self.assertNotIn("errors", calls[0])
        self.assertNotIn("universal_newlines", calls[0])

    def test_decodes_utf8_stderr_on_failure(self) -> None:
        def fake_run(cmd: list[str], **_kwargs) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr="无法读取 日本語.mp4".encode("utf-8"))

        with patch("utils.ffprobe_json.subprocess.run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "无法读取 日本語"):
                run_ffprobe_json(["ffprobe", "-of", "json", "日本語.mp4"])

    def test_subtitle_video_info_uses_binary_json_probe(self) -> None:
        calls: list[dict] = []

        def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
            calls.append(kwargs)
            return _completed(
                cmd,
                {
                    "streams": [{"width": 3840, "height": 2160, "codec_name": "hevc"}],
                    "format": {"duration": "12.5", "tags": {"title": "中文标题"}},
                },
            )

        with patch("utils.ffprobe_json.subprocess.run", side_effect=fake_run):
            info = get_video_info("G:/视频/日本語.mp4")

        self.assertEqual(info["width"], 3840)
        self.assertEqual(info["height"], 2160)
        self.assertEqual(info["duration"], 12.5)
        self.assertEqual(info["codec"], "hevc")
        self.assertNotIn("text", calls[0])

    def test_source_video_bitrate_uses_binary_json_probe(self) -> None:
        calls: list[dict] = []

        def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
            calls.append(kwargs)
            return _completed(
                cmd,
                {
                    "streams": [{"bit_rate": "12345678"}],
                    "format": {"bit_rate": "87654321", "tags": {"title": "動画 中文"}},
                },
            )

        with patch("utils.ffprobe_json.subprocess.run", side_effect=fake_run):
            bitrate = source_video_bitrate(Path("G:/视频/動画.mp4"))

        self.assertEqual(bitrate, 12_345_678)
        self.assertNotIn("text", calls[0])


if __name__ == "__main__":
    unittest.main()
