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

    def test_si_progressive_is_enabled_by_default_for_m1_testing(self) -> None:
        self.assertTrue(config.SI_PROGRESSIVE_ENABLED)
        self.assertTrue(config.SI_PROGRESSIVE_DLNA)
        self.assertEqual(config.SI_AUDIO_EDIT_MODE, "remove")
        self.assertEqual(config.SI_BROWSE_PREWARM_LIMIT, 1)
        self.assertEqual(config.SI_PREWARM_QUEUE_MAX, 2)
        self.assertEqual(config.SI_AUDIO_EXTRACT_MODE, "sequential")
        self.assertGreaterEqual(config.SI_MIX_PARALLEL_MAX, 1)
        self.assertLessEqual(config.SI_MIX_PARALLEL_MAX, 8)
        self.assertEqual(config.SI_MIX_ENCODER, "auto")
        self.assertFalse(config.SI_MIX_SEGMENTED_AAC)
        self.assertEqual(config.SI_MIX_SEGMENT_WARMUP_MS, 1000)

    def test_dlna_images_are_disabled_by_default(self) -> None:
        self.assertFalse(config.DLNA_IMAGE_ENABLED)
        self.assertIn(".jpg", config.IMAGE_EXTS)
        self.assertEqual(config.IMAGE_MIME_BY_EXT[".png"], "image/png")

    def test_two_dvr_temporal_stability_defaults_enable_base_detail_stabilizer(self) -> None:
        self.assertTrue(config.TWO_DVR_TEMPORAL_NORM)
        self.assertEqual(config.TWO_DVR_TEMPORAL_NORM_ALPHA, 0.10)
        self.assertEqual(config.TWO_DVR_TEMPORAL_NORM_RESET, 1.0)
        # On by default: the base/detail rewrite (mode=ema). The per-pixel px
        # limiters stay 0 so the stabilizer only smooths the low-frequency base.
        self.assertTrue(config.TWO_DVR_TEMPORAL_DEPTH)
        self.assertEqual(config.TWO_DVR_TEMPORAL_DEPTH_MODE, "ema")
        self.assertEqual(config.TWO_DVR_TEMPORAL_DEPTH_ALPHA, 0.20)
        self.assertEqual(config.TWO_DVR_TEMPORAL_FLOW_DIFF, 35.0)
        self.assertEqual(config.TWO_DVR_TEMPORAL_FLOW_CONSISTENCY, 0.0)
        self.assertEqual(config.TWO_DVR_TEMPORAL_FLOW_MOTION_GATE, 0.0)
        self.assertTrue(config.TWO_DVR_TEMPORAL_AFFINE)
        self.assertEqual(config.TWO_DVR_TEMPORAL_AFFINE_MAX_SCALE, 0.20)
        self.assertEqual(config.TWO_DVR_TEMPORAL_AFFINE_MAX_BIAS, 0.12)
        self.assertEqual(config.TWO_DVR_TEMPORAL_STATIC_DEADBAND_PX, 0.0)
        self.assertEqual(config.TWO_DVR_TEMPORAL_STATIC_MAX_STEP_PX, 0.0)
        self.assertEqual(config.TWO_DVR_TEMPORAL_MOTION_MAX_STEP_PX, 0.0)


if __name__ == "__main__":
    unittest.main()
