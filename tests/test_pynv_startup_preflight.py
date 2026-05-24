from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pipeline.pynv_stream import PyNvPassthroughStream


class StartupPreflightTests(unittest.TestCase):
    def test_startup_preflight_creates_configured_nvenc_encoders(self) -> None:
        fake_nvc = MagicMock()
        fake_encoder = MagicMock()
        fake_nvc.CreateEncoder.return_value = fake_encoder

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "PyNvVideoCodec":
                return fake_nvc
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("pipeline.pynv_stream.config.NVENC_PREFLIGHT_ENABLE", True),
            patch(
                "pipeline.pynv_stream.config.NVENC_PREFLIGHT_GEOMETRIES",
                [(128, 64, "60.000000", "1000000"), (256, 128, "30.000000", "2000000")],
            ),
        ):
            PyNvPassthroughStream.startup_preflight()

        self.assertEqual(fake_nvc.CreateEncoder.call_count, 2)
        fake_nvc.CreateEncoder.assert_any_call(
            128,
            64,
            "NV12",
            False,
            codec="hevc",
            bitrate="1000000",
            fps="60.000000",
            gop=str(__import__("config").PASSTHROUGH_GOP),
            bf=str(__import__("config").PASSTHROUGH_HEVC_BF),
            preset=str(__import__("config").PASSTHROUGH_PYNV_PRESET),
            tuning_info=str(__import__("config").PASSTHROUGH_PYNV_TUNING_INFO),
            rc=str(__import__("config").PASSTHROUGH_PYNV_RC),
        )
        self.assertEqual(fake_encoder.EndEncode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
