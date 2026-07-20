import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from offline.decoded_frames import decoded_frame_to_bgr, decoded_frame_to_sbs_eye_rgbs, scene_bgr_from_rgb


class _FakePlane:
    def __init__(self, arr):
        self.arr = arr

    def as_cupy(self):
        return self.arr


class _FakeFrame:
    def __init__(self, y, uv):
        self.height, self.width = y.shape
        self.y = _FakePlane(y)
        self.uv = _FakePlane(uv)


def _fake_cupy():
    return types.SimpleNamespace(
        uint8=np.uint8,
        uint16=np.uint16,
        asnumpy=lambda arr: np.asarray(arr),
        right_shift=np.right_shift,
    )


class DecodedFrameTests(unittest.TestCase):
    def test_sbs_eye_rgbs_match_full_frame_bgr_split_for_nv12(self):
        y = np.arange(32, dtype=np.uint8).reshape(4, 8) * 3
        uv = np.array(
            [
                [90, 120, 100, 130, 110, 140, 120, 150],
                [95, 125, 105, 135, 115, 145, 125, 155],
            ],
            dtype=np.uint8,
        )
        frame = _FakeFrame(y, uv)

        with patch.dict(sys.modules, {"cupy": _fake_cupy()}):
            full_bgr = decoded_frame_to_bgr(frame)
            eyes = decoded_frame_to_sbs_eye_rgbs(frame)

        expected = [
            cv2.cvtColor(full_bgr[:, :4], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(full_bgr[:, 4:8], cv2.COLOR_BGR2RGB),
        ]
        self.assertEqual(len(eyes), 2)
        np.testing.assert_array_equal(eyes[0], expected[0])
        np.testing.assert_array_equal(eyes[1], expected[1])

    def test_sbs_eye_rgbs_downconverts_p016(self):
        y8 = np.arange(32, dtype=np.uint8).reshape(4, 8) * 2
        uv8 = np.array(
            [
                [80, 118, 96, 126, 112, 134, 128, 142],
                [84, 122, 100, 130, 116, 138, 132, 146],
            ],
            dtype=np.uint8,
        )
        frame8 = _FakeFrame(y8, uv8)
        frame16 = _FakeFrame(y8.astype(np.uint16) << 8, uv8.astype(np.uint16) << 8)

        with patch.dict(sys.modules, {"cupy": _fake_cupy()}):
            eyes8 = decoded_frame_to_sbs_eye_rgbs(frame8)
            eyes16 = decoded_frame_to_sbs_eye_rgbs(frame16)

        np.testing.assert_array_equal(eyes16[0], eyes8[0])
        np.testing.assert_array_equal(eyes16[1], eyes8[1])

    def test_scene_bgr_from_rgb_downsamples_before_channel_swap(self):
        rgb = np.zeros((20, 40, 3), dtype=np.uint8)
        rgb[..., 0] = 20
        rgb[..., 1] = np.arange(40, dtype=np.uint8)
        rgb[..., 2] = 200

        scene = scene_bgr_from_rgb(rgb, downsample_height=10)
        expected_rgb = cv2.resize(rgb, (20, 10), interpolation=cv2.INTER_AREA)
        expected_bgr = cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR)

        self.assertEqual(scene.shape, (10, 20, 3))
        np.testing.assert_array_equal(scene, expected_bgr)


if __name__ == "__main__":
    unittest.main()
