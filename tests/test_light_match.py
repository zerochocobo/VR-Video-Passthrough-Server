from __future__ import annotations

import unittest

from pipeline.light_match import (
    LightMatchParams,
    apply_light_match_yuv,
    build_light_match_tables,
    normalize_light_match_params,
)
from utils import runtime_settings


class LightMatchCoeffTests(unittest.TestCase):
    def test_identity_params_build_identity_tables(self) -> None:
        tables = build_light_match_tables(
            LightMatchParams(enabled=True, temp_k=6500, tint=0, exposure_ev=0, contrast=1, gamma=1, saturation=1)
        )
        self.assertTrue(tables.identity)
        self.assertEqual(apply_light_match_yuv(100, 120, 130, tables), (100, 120, 130))

    def test_daylight_preset_is_d65_identity(self) -> None:
        tables = build_light_match_tables(
            LightMatchParams(
                enabled=True,
                temp_k=6500,
                tint=0,
                exposure_ev=0.0,
                contrast=1.0,
                gamma=1.0,
                saturation=1.0,
                preset="daylight",
            )
        )
        self.assertTrue(tables.identity)
        self.assertEqual(apply_light_match_yuv(128, 90, 200, tables), (128, 90, 200))

    def test_night_cool_preset_shifts_toward_blue(self) -> None:
        tables = build_light_match_tables(
            LightMatchParams(
                enabled=True,
                temp_k=8000,
                tint=0,
                exposure_ev=0.0,
                contrast=1.0,
                gamma=1.0,
                saturation=1.0,
                preset="night_cool",
            )
        )
        self.assertFalse(tables.identity)
        _y, u, v = apply_light_match_yuv(128, 128, 128, tables)
        self.assertGreater(u, 128)
        self.assertLess(v, 128)

    def test_warm_temperature_shifts_white_warm(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=3000))
        self.assertFalse(tables.identity)
        y, u, v = apply_light_match_yuv(235, 128, 128, tables)
        self.assertGreaterEqual(y, 232)
        self.assertLess(u, 128)
        self.assertGreater(v, 128)

    def test_cool_temperature_preserves_white_luma(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=9000))
        y, u, v = apply_light_match_yuv(235, 128, 128, tables)
        self.assertGreaterEqual(y, 232)
        self.assertGreater(u, 128)
        self.assertLess(v, 128)

    def test_temperature_preserves_neutral_gray_luma(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=3000))
        y, _u, _v = apply_light_match_yuv(128, 128, 128, tables)
        self.assertGreaterEqual(y, 126)
        self.assertLessEqual(y, 130)

    def test_soft_warm_temperature_keeps_chroma_shift_moderate(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=4000, saturation=1.0))
        _y, u, v = apply_light_match_yuv(128, 128, 128, tables)
        self.assertGreaterEqual(u, 114)
        self.assertLessEqual(v, 142)

    def test_exposure_controls_luma(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=6500, exposure_ev=1.0))
        y, u, v = apply_light_match_yuv(100, 128, 128, tables)
        self.assertGreater(y, 100)
        self.assertEqual((u, v), (128, 128))

    def test_contrast_keeps_midpoint_pivot(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=6500, contrast=0.5))
        self.assertEqual(apply_light_match_yuv(128, 128, 128, tables), (128, 128, 128))
        low, _, _ = apply_light_match_yuv(80, 128, 128, tables)
        high, _, _ = apply_light_match_yuv(180, 128, 128, tables)
        self.assertGreater(low, 80)
        self.assertLess(high, 180)

    def test_gamma_lut_is_monotonic(self) -> None:
        for gamma in (0.7, 1.0, 1.4):
            tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=6500, gamma=gamma))
            values = [int(v) for v in tables.gamma_lut]
            self.assertEqual(values, sorted(values))

    def test_saturation_zero_collapses_uv_to_neutral(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=6500, saturation=0.0))
        _, u, v = apply_light_match_yuv(128, 90, 200, tables)
        self.assertEqual(u, 128)
        self.assertEqual(v, 128)

    def test_gamma_half_boosts_midtones(self) -> None:
        tables = build_light_match_tables(LightMatchParams(enabled=True, temp_k=6500, gamma=0.5))
        self.assertGreater(int(tables.gamma_lut[64]), 64)

    def test_normalize_clamps_ranges(self) -> None:
        params = normalize_light_match_params(
            {
                "enabled": True,
                "temp_k": 99999,
                "tint": -99,
                "exposure_ev": 9,
                "contrast": 9,
                "gamma": 9,
                "saturation": 9,
                "preset": "bad",
            }
        )
        self.assertEqual(params.temp_k, 9000)
        self.assertEqual(params.tint, -50.0)
        self.assertEqual(params.exposure_ev, 2.0)
        self.assertEqual(params.contrast, 1.5)
        self.assertEqual(params.gamma, 1.4)
        self.assertEqual(params.saturation, 2.0)
        self.assertEqual(params.preset, "custom")


class LightMatchRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_settings.reset_for_test()

    def test_set_light_match_increments_version_only_on_change(self) -> None:
        start = runtime_settings.get_light_match()
        first = runtime_settings.set_light_match({"enabled": not start.enabled})
        second = runtime_settings.set_light_match(first.params())
        self.assertEqual(second.version, first.version)
        self.assertNotEqual(first.version, start.version)

    def test_reset_for_test_reloads_default_state(self) -> None:
        runtime_settings.set_light_match({"enabled": True, "temp_k": 3000})
        reset = runtime_settings.reset_for_test()
        self.assertFalse(reset.enabled)
        self.assertEqual(reset.temp_k, 6500)
        self.assertEqual(reset.preset, "daylight")
        self.assertEqual(reset.version, 0)


if __name__ == "__main__":
    unittest.main()
