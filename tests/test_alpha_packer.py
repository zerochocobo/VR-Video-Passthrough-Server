from __future__ import annotations

import unittest
import math

from unittest import mock

import config
from pipeline.alpha_packer import AlphaPacker, alpha_src_fisheye_fov, fisheye_src_radial_scale


class _Plane:
    def __init__(self, value: str) -> None:
        self.value = value

    def as_cupy(self) -> str:
        return self.value


class _Frame:
    width = 3840
    height = 2160
    y = _Plane("y16")
    uv = _Plane("uv16")


class _DeviceData:
    ptr = 1


class _DeviceAlpha:
    shape = (128, 256)
    data = _DeviceData()

    def astype(self, _dtype, copy: bool = False):
        return self


class _FakeArray:
    def __init__(self, shape) -> None:
        self.shape = shape


class _FakeCp:
    float32 = "float32"

    @staticmethod
    def empty(shape, dtype=None):
        return _FakeArray(shape)


class _Matter:
    def __init__(self) -> None:
        self.upload_args = None
        self._g_frame = "uploaded_nv12"
        self.temporal_called = False

    def upload_p016_planes_as_nv12_gpu(self, *args, **kwargs) -> None:
        self.upload_args = (args, kwargs)

    def _alpha_low_res_gpu(self, h: int, w: int, use_nv12: bool = False):
        return "alpha", "timing", "shape"

    def _alpha_low_res_gpu_temporal(self, h: int, w: int, use_nv12: bool = False):
        self.temporal_called = True
        return "temporal_alpha", "timing", "shape"


class AlphaPackerTests(unittest.TestCase):
    def test_sbs_half_equirect_to_fisheye_current_projection_anchors(self) -> None:
        eye_w = 4096
        out_w = eye_w * 2
        out_h = 4096
        radius = min(eye_w, out_h) * 0.5

        def map_pixel(x: float, y: float) -> tuple[float, float]:
            eye = 1 if x >= eye_w else 0
            lx = x - eye * eye_w + 0.5
            ly = y + 0.5
            cx = eye_w * 0.5
            cy = out_h * 0.5
            nx = (lx - cx) / radius
            ny = (ly - cy) / radius
            rr = math.hypot(nx, ny)
            if rr > 1.0:
                return -1.0, -1.0
            ff_theta = math.pi * 0.5 * (1.0 - rr)
            ff_phi = math.atan2(ny, nx)
            cos_t = math.cos(ff_theta)
            dir_x = cos_t * math.cos(ff_phi)
            dir_y = cos_t * math.sin(ff_phi)
            dir_z = math.sin(ff_theta)
            src_phi = math.atan2(dir_x, dir_z) / (math.pi * 0.5)
            src_theta = math.asin(dir_y) / (math.pi * 0.5)
            u = (0.5 * src_phi + 0.5) * (eye_w - 1)
            v = (0.5 * src_theta + 0.5) * (out_h - 1)
            return u + eye * eye_w, v

        center_x, center_y = map_pixel(eye_w / 2 - 0.5, out_h / 2 - 0.5)
        right_x, right_y = map_pixel(eye_w - 0.5, out_h / 2 - 0.5)
        top_x, top_y = map_pixel(eye_w / 2 - 0.5, 0)
        corner_x, corner_y = map_pixel(0, 0)

        self.assertAlmostEqual(center_x, (eye_w - 1) * 0.5, places=4)
        self.assertAlmostEqual(center_y, (out_h - 1) * 0.5, places=4)
        self.assertAlmostEqual(right_x, eye_w - 1, delta=1.0)
        self.assertAlmostEqual(right_y, (out_h - 1) * 0.5, places=4)
        self.assertAlmostEqual(top_x, (eye_w - 1) * 0.5, places=4)
        self.assertAlmostEqual(top_y, 1.0, delta=1.0)
        self.assertEqual((corner_x, corner_y), (-1.0, -1.0))

    def test_fisheye_source_keeps_the_picture_in_fisheye(self) -> None:
        # A dual-fisheye SBS frame and a half-equirect SBS frame are the same
        # shape, so only the filename-derived FOV can pick the fisheye path.
        self.assertEqual(AlphaPacker.projection_mode_static(8192, 4096, 190.0), "fisheye_src")
        self.assertEqual(AlphaPacker.projection_mode_static(8192, 4096, 0.0), "sbs_half_equirect")
        # A flat 2D source is never fisheye, whatever a marker claims.
        self.assertEqual(AlphaPacker.projection_mode_static(1920, 1080, 190.0), "flat2d_fisheye")

    def test_fisheye_source_fov_reads_the_filename_and_honours_the_override(self) -> None:
        with mock.patch.object(config, "ALPHA_SRC_FISHEYE", "auto"), mock.patch.object(config, "ALPHA_SRC_FOV", 0.0):
            self.assertEqual(alpha_src_fisheye_fov("TEST_4096p_91983_FISHEYE190_x265"), 190.0)
            self.assertEqual(alpha_src_fisheye_fov("movie_LR_180_SBS"), 0.0)
        with mock.patch.object(config, "ALPHA_SRC_FISHEYE", "auto"), mock.patch.object(config, "ALPHA_SRC_FOV", 200.0):
            self.assertEqual(alpha_src_fisheye_fov("movie_LR_180_SBS"), 200.0)
        with mock.patch.object(config, "ALPHA_SRC_FISHEYE", "off"), mock.patch.object(config, "ALPHA_SRC_FOV", 200.0):
            self.assertEqual(alpha_src_fisheye_fov("TEST_FISHEYE190"), 0.0)

    def test_fisheye_source_radial_scale_fit_keeps_the_whole_circle(self) -> None:
        with mock.patch.object(config, "ALPHA_SRC_RADIUS_SCALE", 1.0):
            with mock.patch.object(config, "ALPHA_SRC_FIT", "fit"):
                # fit maps circle onto circle: nothing is cut and 180 is a copy.
                self.assertAlmostEqual(fisheye_src_radial_scale(190.0), 1.0, places=6)
                self.assertAlmostEqual(fisheye_src_radial_scale(200.0), 1.0, places=6)
            with mock.patch.object(config, "ALPHA_SRC_FIT", "crop"):
                self.assertAlmostEqual(fisheye_src_radial_scale(190.0), 180.0 / 190.0, places=6)
                self.assertAlmostEqual(fisheye_src_radial_scale(200.0), 0.9, places=6)
                # A 180 source has no ring to drop, so crop is also a copy.
                self.assertAlmostEqual(fisheye_src_radial_scale(180.0), 1.0, places=6)

    def test_fisheye_source_mapping_anchors(self) -> None:
        eye_w = 4096
        out_w = eye_w * 2
        out_h = 4096
        radius = min(eye_w, out_h) * 0.5

        def map_pixel(x: int, y: int, src_radial_scale: float) -> tuple[float, float]:
            eye = 1 if x >= eye_w else 0
            fx = float(x - eye * eye_w)
            fy = float(y)
            cx = (eye_w - 1) * 0.5
            cy = (out_h - 1) * 0.5
            nx = (fx - cx) / radius
            ny = (fy - cy) / radius
            if nx * nx + ny * ny > 1.0:
                return -1.0, -1.0
            u = min(max(cx + (fx - cx) * src_radial_scale, 0.0), eye_w - 1)
            v = min(max(cy + (fy - cy) * src_radial_scale, 0.0), out_h - 1)
            return u + eye * eye_w, v

        # fit: an exact copy, so every output pixel reads its own source pixel.
        for x, y in [(0, out_h // 2), (eye_w // 2, out_h // 2), (eye_w // 2, 0), (eye_w + 10, out_h // 2)]:
            with self.subTest(pixel=(x, y)):
                self.assertEqual(map_pixel(x, y, 1.0), (float(x), float(y)))
        # crop: the rim samples the source's 180-degree radius, not its edge.
        crop = 180.0 / 190.0
        u, _v = map_pixel(0, out_h // 2, crop)
        self.assertAlmostEqual(u, (eye_w - 1) * 0.5 * (1.0 - crop), places=3)
        # Outside the circle stays outside in both modes.
        self.assertEqual(map_pixel(0, 0, 1.0), (-1.0, -1.0))
        self.assertEqual(map_pixel(0, 0, crop), (-1.0, -1.0))

    def test_pack_gpu_p016_frame_uploads_as_nv12_before_packing(self) -> None:
        matter = _Matter()
        packer = AlphaPacker.__new__(AlphaPacker)
        packer.matter = matter
        packer.pack_uploaded = lambda alpha, h, w, subtitle_overlay=None: (  # type: ignore[method-assign]
            "packed",
            alpha,
            h,
            w,
            subtitle_overlay,
        )

        packed, timing = packer.pack_gpu_p016_frame(_Frame(), shift_bits=6)

        self.assertEqual(packed, ("packed", "alpha", 2160, 3840, None))
        self.assertEqual(timing, "timing")
        self.assertEqual(matter.upload_args, (("y16", "uv16", 2160, 3840), {"shift_bits": 6}))
        self.assertFalse(matter.temporal_called)

    def test_before_pack_overlay_return_is_passed_to_pack_uploaded(self) -> None:
        matter = _Matter()
        packer = AlphaPacker.__new__(AlphaPacker)
        packer.matter = matter
        seen = {}

        def before_pack(uploaded):
            seen["uploaded"] = uploaded
            return [("rgba", 1, 2)]

        def pack_uploaded(alpha, h, w, subtitle_overlay=None):
            return ("packed", alpha, h, w, subtitle_overlay)

        packer.pack_uploaded = pack_uploaded  # type: ignore[method-assign]

        packed, timing = packer.pack_gpu_p016_frame(_Frame(), shift_bits=6, before_pack=before_pack)

        self.assertEqual(seen["uploaded"], "uploaded_nv12")
        self.assertEqual(packed, ("packed", "alpha", 2160, 3840, [("rgba", 1, 2)]))
        self.assertEqual(timing, "timing")

    def test_pack_uploaded_uses_matter_uploaded_frame_buffer(self) -> None:
        matter = _Matter()
        packer = AlphaPacker.__new__(AlphaPacker)
        packer.matter = matter
        packer.scale = 0.4
        packer.radius_scale = 1.0
        packer.alpha_cutoff = 0.0
        packer.alpha_hard_edge = False
        packer.alpha_contrast = 1.0
        packer._cp = _FakeCp()
        packer._g_alpha = None
        packer._g_fisheye_alpha = _FakeArray((2160, 3840))
        calls = []
        packer._project_kernel = lambda grid, block, args: calls.append(("project", args))  # type: ignore[method-assign]
        packer._overlay_kernel = lambda grid, block, args: calls.append(("overlay", args))  # type: ignore[method-assign]
        matter._ensure_dev_nv12_out = lambda h, w: "out_nv12"  # type: ignore[attr-defined]

        out = packer.pack_uploaded(_DeviceAlpha(), 2160, 3840)

        self.assertEqual(out, "out_nv12")
        self.assertEqual(calls[0][1][0], "uploaded_nv12")

    def test_pack_uploaded_blends_subtitles_after_projection_before_alpha_layout(self) -> None:
        matter = _Matter()
        packer = AlphaPacker.__new__(AlphaPacker)
        packer.matter = matter
        packer.scale = 0.4
        packer.radius_scale = 1.0
        packer.alpha_cutoff = 0.0
        packer.alpha_hard_edge = False
        packer.alpha_contrast = 1.0
        packer._cp = _FakeCp()
        packer._g_alpha = None
        packer._g_fisheye_alpha = _FakeArray((2160, 3840))
        calls = []
        packer._project_kernel = lambda grid, block, args: calls.append(("project", args))  # type: ignore[method-assign]
        packer._overlay_kernel = lambda grid, block, args: calls.append(("alpha_layout", args))  # type: ignore[method-assign]
        packer._blend_projected_subtitles = lambda *args: calls.append(("subtitle_layer", args))  # type: ignore[method-assign]
        matter._ensure_dev_nv12_out = lambda h, w: "out_nv12"  # type: ignore[attr-defined]

        out = packer.pack_uploaded(_DeviceAlpha(), 2160, 3840, subtitle_overlay=[("rgba", 1, 2)])

        self.assertEqual(out, "out_nv12")
        self.assertEqual([name for name, _args in calls], ["project", "subtitle_layer", "alpha_layout"])


if __name__ == "__main__":
    unittest.main()
