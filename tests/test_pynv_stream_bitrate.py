from __future__ import annotations

import unittest
from unittest.mock import patch

from http_app import routes_media
from pipeline import pynv_stream


class PyNvStreamBitrateTests(unittest.TestCase):
    def test_realtime_bitrate_uses_configured_hevc_bitrate(self) -> None:
        with patch.object(pynv_stream.config, "PASSTHROUGH_HEVC_BITRATE", "50M"):
            self.assertEqual(pynv_stream._realtime_pynv_bitrate(), "50000000")

    def test_realtime_bitrate_does_not_use_source_multiplier(self) -> None:
        with (
            patch.object(pynv_stream.config, "PASSTHROUGH_HEVC_BITRATE", "12M"),
            patch.object(pynv_stream.config, "PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER", 0.1),
        ):
            self.assertEqual(pynv_stream._realtime_pynv_bitrate(), "12000000")

    def test_live_pynv_estimate_uses_configured_hevc_bitrate(self) -> None:
        with (
            patch.object(routes_media, "PASSTHROUGH_HEVC_BITRATE", "50M"),
            patch.object(routes_media, "PASSTHROUGH_SEND_PACING_MULTIPLIER", 2.0),
            patch.object(routes_media, "PASSTHROUGH_SEND_MIN_BPS", 1),
        ):
            self.assertEqual(routes_media._estimated_live_pynv_size(10.0), 62_500_000)
            self.assertEqual(routes_media._estimated_live_pynv_send_bps(), 100_000_000)


if __name__ == "__main__":
    unittest.main()
