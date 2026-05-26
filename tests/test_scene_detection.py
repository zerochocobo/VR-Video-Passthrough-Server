import unittest

import numpy as np

from utils.scene_detection import SceneCutDetector


class SceneCutDetectorTests(unittest.TestCase):
    def test_detects_large_color_change_and_respects_cooldown(self):
        detector = SceneCutDetector(threshold=0.2, cooldown_frames=1, ref_ema_alpha=0.95)
        red = np.zeros((32, 32, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        blue = np.zeros((32, 32, 3), dtype=np.uint8)
        blue[:, :, 0] = 255
        green = np.zeros((32, 32, 3), dtype=np.uint8)
        green[:, :, 1] = 255

        self.assertFalse(detector.step(red))
        self.assertTrue(detector.step(blue))
        self.assertFalse(detector.step(green))

        detector.reset()
        self.assertFalse(detector.step(green))


if __name__ == "__main__":
    unittest.main()
