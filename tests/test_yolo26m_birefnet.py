import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from offline import yolo26m_birefnet as mod


class _FakeSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path = str(path)
        self.kind = "birefnet" if "BiRefNet" in self.path or "model_fp16" in self.path else "yolo"
        self.last_inputs = None

    def get_inputs(self):
        name = "input_image" if self.kind == "birefnet" else "pixel_values"
        return [SimpleNamespace(name=name)]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        self.last_inputs = inputs
        if "pixel_values" in inputs:
            logits = np.full((1, 300, 80), -10.0, dtype=np.float32)
            boxes = np.zeros((1, 300, 4), dtype=np.float32)
            logits[0, 1, 0] = 1.0
            boxes[0, 1] = [0.5, 0.5, 0.25, 0.5]
            return [logits, boxes]
        mask = np.full((1, 1, 8, 8), -8.0, dtype=np.float32)
        mask[0, 0, 2:6, 2:6] = 8.0
        return [mask]


def _make_masker(**kwargs):
    defaults = {"birefnet_input_size": 8}
    defaults.update(kwargs)
    with patch.object(mod.ort, "InferenceSession", _FakeSession):
        return mod.Yolo26mBiRefNetMasker(Path("models/yolo26m"), Path("models/BiRefNet"), provider="cpu", **defaults)


class Yolo26mBiRefNetTests(unittest.TestCase):
    def test_sigmoid_handles_extreme_logits_without_overflow_warning(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", RuntimeWarning)
            out = mod._sigmoid(np.array([-1000.0, 0.0, 1000.0], dtype=np.float32))

        self.assertFalse([item for item in seen if issubclass(item.category, RuntimeWarning)])
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)

    def test_detect_uses_yolo_pixel_values(self):
        masker = _make_masker()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        dets = masker.detect(image)

        self.assertEqual(list(masker.yolo.last_inputs), ["pixel_values"])
        self.assertEqual(len(dets), 1)
        np.testing.assert_allclose(dets[0].box_xyxy, [75.0, 0.0, 125.0, 99.0], atol=1e-4)

    def test_birefnet_preprocess_uses_input_image_and_imagenet_normalization(self):
        masker = _make_masker(binarize_mask=False)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        box = np.array([8, 8, 24, 24], dtype=np.float32)

        masker._birefnet_mask_for_box(image, box, (16, 16))

        self.assertEqual(list(masker.birefnet.last_inputs), ["input_image"])
        inp = masker.birefnet.last_inputs["input_image"]
        self.assertEqual(inp.shape, (1, 3, 8, 8))
        np.testing.assert_allclose(inp[0, :, 0, 0], -np.array([0.485 / 0.229, 0.456 / 0.224, 0.406 / 0.225]), atol=1e-5)

    def test_birefnet_mask_pastes_roi_and_keeps_outside_zero(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        box = np.array([8, 8, 24, 24], dtype=np.float32)
        mask = _make_masker(binarize_mask=True, mask_erode_px=0)._birefnet_mask_for_box(image, box, (16, 16))

        self.assertEqual(mask.shape, (16, 16))
        self.assertEqual(set(np.unique(mask).tolist()), {0.0, 1.0})
        self.assertGreater(float(mask[4:12, 4:12].sum()), 0.0)
        self.assertEqual(float(mask[:3, :].sum()), 0.0)
        self.assertEqual(float(mask[:, :3].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
