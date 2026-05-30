from __future__ import annotations

import unittest

import config


class ConfigDefaultTests(unittest.TestCase):
    def test_mpegts_slate_is_disabled_by_default(self) -> None:
        self.assertFalse(config.PASSTHROUGH_MPEGTS_VIDEO_SLATE)
        self.assertFalse(config.PASSTHROUGH_AUDIO_MPEGTS_SLATE)
        self.assertEqual(config.PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE, "pipe_ts")
        self.assertEqual(config.PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES, 1)
        self.assertFalse(config.PASSTHROUGH_AUDIO_MPEGTS_CACHE)

    def test_composite_warmup_is_enabled_by_default(self) -> None:
        self.assertTrue(config.WARMUP_COMPOSITE_ENABLE)
        self.assertIn((4096, 8192), config.WARMUP_COMPOSITE_GEOMETRIES)
        self.assertIn((2048, 4096), config.WARMUP_COMPOSITE_GEOMETRIES)
        self.assertEqual(config.WARMUP_RAMPUP_DIAG_FRAMES, 0)

    def test_nvenc_preflight_is_enabled_by_default(self) -> None:
        self.assertTrue(config.NVENC_PREFLIGHT_ENABLE)
        self.assertIn((8192, 4096, "59.94006", "50000000"), config.NVENC_PREFLIGHT_GEOMETRIES)
        self.assertIn((4096, 2048, "59.94006", "25000000"), config.NVENC_PREFLIGHT_GEOMETRIES)

    def test_mux_latency_defaults_are_low_latency(self) -> None:
        self.assertTrue(config.MUX_LATENCY_DIAG)
        self.assertFalse(config.MUX_LATENCY_DIAG_VERBOSE)
        self.assertEqual(config.MUX_FFMPEG_LOGLEVEL, "warning")
        self.assertFalse(config.FORCE_AUDIO_OFF)
        self.assertEqual(config.MUX_RAW_VIDEO_PROBESIZE, "1000000")
        self.assertEqual(config.MUX_RAW_VIDEO_ANALYZEDURATION, "1000000")
        self.assertEqual(config.MUX_INTERMEDIATE_TS_PROBESIZE, "16384")
        self.assertEqual(config.MUX_INTERMEDIATE_TS_ANALYZEDURATION, "0")
        self.assertEqual(config.MUX_PROBESIZE_OVERRIDE, "32")
        self.assertEqual(config.MUX_CONTAINER_PROBESIZE_OVERRIDE, "32768")
        self.assertEqual(config.MUX_AUDIO_PROBESIZE_OVERRIDE, "32768")
        self.assertEqual(config.MUX_ANALYZEDURATION_US, "0")
        self.assertFalse(config.MUX_NOBUFFER_ENABLE)
        self.assertEqual(config.PASSTHROUGH_FMP4_FRAG_DURATION_US, 100000)
        self.assertEqual(config.PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA, "500000000")

    def test_seekable_passthrough_is_hidden_by_default(self) -> None:
        self.assertFalse(config.PASSTHROUGH_SEEK_ENABLED)
        self.assertFalse(config.PASSTHROUGH_SEEK_DLNA)
        self.assertEqual(config.PASSTHROUGH_SEEK_ROUTE_POLICY, "profile")
        self.assertIn("nplayer", config.PASSTHROUGH_SEEK_PROFILES)
        self.assertEqual(config.PASSTHROUGH_SEEK_CONTAINER, "mpegts")
        self.assertEqual(config.PASSTHROUGH_SEEK_HEADER_BYTES, 2 * 1024 * 1024)

    def test_dlna_images_are_disabled_by_default(self) -> None:
        self.assertFalse(config.DLNA_IMAGE_ENABLED)
        self.assertIn(".jpg", config.IMAGE_EXTS)
        self.assertEqual(config.IMAGE_MIME_BY_EXT[".png"], "image/png")


if __name__ == "__main__":
    unittest.main()
