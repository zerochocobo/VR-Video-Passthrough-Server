from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import offline_alpha_passthrough as offline_alpha


class OfflineAlphaBitrateTests(unittest.TestCase):
    def test_source_bitrate_scales_with_alpha_output_pixels(self) -> None:
        args = argparse.Namespace(bitrate="source", maxrate_multiplier=1.2, bufsize_multiplier=2.0, rc="vbr", cq=-1, preset="P1")
        with (
            patch.object(offline_alpha, "source_video_bitrate", return_value=926_552),
            patch.object(offline_alpha.config, "PASSTHROUGH_HEVC_BITRATE", "50M"),
        ):
            _kwargs, target_bps, _max_bps, _buf_bps = offline_alpha._encoder_bitrate_kwargs(
                args,
                Path("dance.mp4"),
                720,
                1280,
                5760,
                2880,
            )

        self.assertEqual(target_bps, 16_677_936)

    def test_source_bitrate_keeps_unscaled_alpha_output_near_source(self) -> None:
        args = argparse.Namespace(bitrate="source", maxrate_multiplier=1.2, bufsize_multiplier=2.0, rc="vbr", cq=-1, preset="P1")
        with (
            patch.object(offline_alpha, "source_video_bitrate", return_value=40_000_000),
            patch.object(offline_alpha.config, "PASSTHROUGH_HEVC_BITRATE", "50M"),
        ):
            _kwargs, target_bps, _max_bps, _buf_bps = offline_alpha._encoder_bitrate_kwargs(
                args,
                Path("vr.mp4"),
                4096,
                2048,
                4096,
                2048,
            )

        self.assertEqual(target_bps, 40_000_000)


if __name__ == "__main__":
    unittest.main()
