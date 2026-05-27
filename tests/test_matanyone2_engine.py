import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import config
from offline.matanyone2_engine import MatAnyone2OnnxEngine
from pipeline.matanyone2_roi import RoiMeta, roi_from_mask


class _FakeMeta:
    def __init__(self, name, type_name="tensor(float)", shape=None):
        self.name = name
        self.type = type_name
        self.shape = shape or [1, 1]


class _FakeSession:
    def __init__(self, outputs=None, providers=None):
        self._outputs = outputs or []
        self._providers = providers or ["CPUExecutionProvider"]

    def get_inputs(self):
        return [_FakeMeta("image")]

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers


class MatAnyone2EngineTests(unittest.TestCase):
    def _engine(self):
        engine = MatAnyone2OnnxEngine.__new__(MatAnyone2OnnxEngine)
        engine.np = np
        engine.tensor_dtype = np.float32
        engine.profile = defaultdict(list)
        engine.mask_path = None
        engine.cv2 = __import__("cv2")
        engine._last_mask_gate_failed = False
        engine._step_io_outputs = {0: {}, 1: {}}
        engine._step_io_slots = [0, 0]
        engine.ort = type(
            "Ort",
            (),
            {
                "OrtValue": type(
                    "OrtValue",
                    (),
                    {
                        "ortvalue_from_numpy": staticmethod(lambda value, *_args: ("cuda", value)),
                        "ortvalue_from_shape_and_type": staticmethod(lambda shape, *_args: _FakeOrtValue(shape)),
                    },
                )
            },
        )
        engine.step_update = _FakeSession(outputs=[_FakeMeta("prob", shape=[1, 2, 4, 4])])
        engine._step_output_dtypes = {"prob": np.float32}
        return engine

    def test_output_ortvalue_is_reused_for_same_shape(self):
        engine = self._engine()

        first = engine._step_output_ortvalue("prob", (1, 2, 4, 4))
        second = engine._step_output_ortvalue("prob", (1, 2, 4, 4))

        self.assertIs(first, second)

    def test_output_ortvalue_uses_independent_slots(self):
        engine = self._engine()

        first = engine._step_output_ortvalue("prob", (1, 2, 4, 4), slot=0)
        second = engine._step_output_ortvalue("prob", (1, 2, 4, 4), slot=1)

        self.assertIsNot(first, second)

    def test_output_ortvalue_uses_independent_eye_buckets(self):
        engine = self._engine()

        left = engine._step_output_ortvalue("prob", (1, 2, 4, 4), slot=0, eye_idx=0)
        right = engine._step_output_ortvalue("prob", (1, 2, 4, 4), slot=0, eye_idx=1)

        self.assertIsNot(left, right)

    def test_should_use_iobinding_requires_cuda_image_and_initialized_state(self):
        engine = self._engine()
        engine._iobinding_enabled = True
        engine._iobinding_failed = False
        image = np.zeros((1, 3, 4, 4), dtype=np.float32)
        state = type("State", (), {})()
        for attr in ("memory_key", "memory_shrinkage", "memory_msk_value", "obj_memory", "sensory", "last_mask", "last_pix_feat"):
            setattr(state, attr, np.zeros((1,), dtype=np.float32))

        self.assertFalse(engine._should_use_iobinding(image, state))

    def test_static_mask_path_is_active(self):
        engine = self._engine()
        engine.mask_path = Path("mask.png")

        self.assertTrue(engine.is_active_frame())

    def test_last_mask_uncert_gate_shape_mismatch_falls_back(self):
        engine = self._engine()
        state = type("State", (), {})()
        state.last_mask = np.ones((1, 1, 4, 4), dtype=np.float32)
        state.last_uncert = np.ones((2, 1, 4, 4), dtype=np.float32)

        with patch.object(config, "MATANYONE2_LAST_MASK_UNCERT_GATE", 0.7):
            out = engine._last_mask_input(state)

        self.assertIs(out, state.last_mask)

    def test_last_mask_uncert_gate_resizes_spatial_map(self):
        engine = self._engine()
        state = type("State", (), {})()
        state.last_mask = np.ones((1, 1, 4, 4), dtype=np.float32)
        state.last_uncert = np.ones((1, 1, 2, 2), dtype=np.float32)

        with patch.object(config, "MATANYONE2_LAST_MASK_UNCERT_GATE", 0.5):
            out = engine._last_mask_input(state)

        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, np.full((1, 1, 4, 4), 0.5, dtype=np.float32))

    def test_sensory_decay_preserves_dtype(self):
        engine = self._engine()
        sensory = np.ones((1, 2, 2), dtype=np.float16)

        out = engine._decay_sensory(sensory, 0.9)

        self.assertEqual(out.dtype, np.float16)
        np.testing.assert_allclose(out, np.full((1, 2, 2), 0.9, dtype=np.float16), rtol=1e-3)

    def test_last_pred_binarize_threshold(self):
        engine = self._engine()
        last_mask = np.array([[[[0.49, 0.50, 0.51]]]], dtype=np.float32)

        with (
            patch.object(config, "MATANYONE2_LAST_PRED_BINARIZE", True),
            patch.object(config, "MATANYONE2_LAST_PRED_BIN_THRESHOLD", 0.5),
        ):
            out = engine._last_pred_mask_input(last_mask)

        np.testing.assert_array_equal(out, np.array([[[[0.0, 0.0, 1.0]]]], dtype=np.float32))

    def test_bootstrap_refine_iters_default_3(self):
        engine = self._engine()
        engine.eyes = [MatAnyone2OnnxEngine._EyeState(), MatAnyone2OnnxEngine._EyeState()]
        engine.sensory_single_shape = (1, 1, 1, 2, 2)
        engine.bootstrap_refine_iters = config.MATANYONE2_BOOTSTRAP_REFINE_ITERS
        engine._image_key = Mock(
            return_value={
                "f16": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "f8": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "f4": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "f2": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "f1": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "pix_feat": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "key": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "shrinkage": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "selection": np.zeros((1, 1, 2, 2), dtype=np.float32),
            }
        )
        sensory = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
        msk_value = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
        obj_memory = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
        engine._bootstrap_mask = Mock(return_value=np.ones((1, 1, 2, 2), dtype=np.float32))
        engine._mask_memory = Mock(return_value=(msk_value, sensory, obj_memory))
        prob = np.zeros((1, 2, 2, 2), dtype=np.float32)
        prob[:, 1:2] = 0.8
        engine._first_frame_refine = Mock(return_value=(prob, sensory))
        engine._smooth_eye_alpha = Mock(side_effect=lambda alpha, _eye_idx: alpha)

        out = engine._run_eye(np.zeros((1, 3, 2, 2), dtype=np.float32), h=2, w=4, eye_idx=0)

        self.assertEqual(config.MATANYONE2_BOOTSTRAP_REFINE_ITERS, 3)
        self.assertEqual(engine._first_frame_refine.call_count, 3)
        self.assertEqual(engine._mask_memory.call_count, 4)
        self.assertEqual(out.shape, (2, 2))

    def test_maybe_refine_alpha_calls_guided_upsampler(self):
        engine = self._engine()
        engine._guided_upsample_enabled = True
        engine._guided_upsample_failed = False
        engine.profile = defaultdict(list)
        engine.matter = type("Matter", (), {"_g_frame": np.zeros((12, 8), dtype=np.uint8)})()
        alpha = np.zeros((2, 4), dtype=np.float32)
        refined = object()

        with patch("pipeline.alpha_guided_filter.fast_guided_filter_upsample", return_value=refined) as upsample:
            out = engine._maybe_refine_alpha(alpha, 8, 8)

        self.assertIs(out, refined)
        upsample.assert_called_once()
        self.assertEqual(upsample.call_args.kwargs["band_lo"], config.MATANYONE2_GUIDED_BAND_LO)
        self.assertEqual(upsample.call_args.kwargs["band_hi"], config.MATANYONE2_GUIDED_BAND_HI)
        self.assertEqual(len(engine.profile["guided_upsample"]), 1)

    def test_segment_roi_is_derived_from_bootstrap_mask(self):
        engine = self._engine()
        engine._roi_enabled = True
        engine._roi_failed = False
        engine.mask_path = None
        engine.in_w = 8
        engine.in_h = 8
        engine._segment_rois = {}
        mask = np.zeros((1, 1, 8, 8), dtype=np.float32)
        mask[0, 0, 2:6, 3:5] = 1.0
        engine.segment_masks = {0: [mask, mask.copy()]}

        roi = engine._segment_roi(0, 0, eye_w=80, h=80)

        self.assertIsNotNone(roi)
        self.assertLess(roi.roi_w * roi.roi_h, 80 * 80)

    def test_roi_from_mask_rejects_large_eye_fraction(self):
        mask = np.ones((8, 8), dtype=np.float32)

        roi = roi_from_mask(mask, eye_w=80, eye_h=80, model_w=8, model_h=8, max_eye_fraction=0.2)

        self.assertIsNone(roi)

    def test_segment_roi_pair_requires_both_eyes(self):
        engine = self._engine()
        engine._roi_enabled = True
        engine._roi_failed = False
        engine.mask_path = None
        engine.in_w = 8
        engine.in_h = 8
        engine._segment_rois = {}
        left = np.zeros((1, 1, 8, 8), dtype=np.float32)
        left[0, 0, 2:6, 3:5] = 1.0
        right = np.ones((1, 1, 8, 8), dtype=np.float32)
        engine.segment_masks = {0: [left, right]}

        pair = engine._segment_roi_pair(0, eye_w=80, h=80)

        self.assertEqual(pair, (None, None))

    def test_preprocess_right_roi_uses_eye_source_offset(self):
        engine = self._engine()
        engine.in_w = 8
        engine.in_h = 8
        engine._iobinding_enabled = False
        engine._iobinding_failed = False
        captured = {}

        class _Matter:
            def _gpu_preprocess_nv12_roi_one(self, *args, **kwargs):
                captured["source_x0"] = kwargs.get("source_x0")
                captured["batch"] = kwargs.get("batch")
                captured["batch_idx"] = kwargs.get("batch_idx")
                return np.zeros((1, 3, 8, 8), dtype=np.float32)

        engine.matter = _Matter()
        roi = RoiMeta(2, 1, 6, 5, 0, 0, 8, 8, 40, 20, 8, 8)

        engine._preprocess_eye_roi(roi, eye_idx=1)

        self.assertEqual(captured["source_x0"], 40)
        self.assertEqual(captured["batch"], 2)
        self.assertEqual(captured["batch_idx"], 1)

    def test_preprocess_eye_uses_eye_specific_batch_slot(self):
        engine = self._engine()
        engine.in_w = 8
        engine.in_h = 8
        engine._iobinding_enabled = False
        engine._iobinding_failed = False
        captured = {}

        class _Matter:
            def _gpu_preprocess_nv12_one(self, *args, **kwargs):
                captured["batch"] = kwargs.get("batch")
                captured["batch_idx"] = kwargs.get("batch_idx")
                return np.zeros((1, 3, 8, 8), dtype=np.float32)

        engine.matter = _Matter()

        engine._preprocess_eye(40, 40, eye_idx=1)

        self.assertEqual(captured["batch"], 2)
        self.assertEqual(captured["batch_idx"], 1)

    def test_roi_bootstrap_mask_is_crop_letterboxed(self):
        import cv2

        engine = self._engine()
        engine.cv2 = cv2
        engine.bootstrap_threshold = 0.5
        engine.bootstrap_erode = 0
        engine.bootstrap_dilate = 0
        engine.bootstrap_soft = False
        engine.in_w = 8
        engine.in_h = 8
        engine._frame_index = 0
        engine.segment_starts = [0]
        mask = np.zeros((1, 1, 8, 8), dtype=np.float32)
        mask[0, 0, 2:6, 3:5] = 1.0
        engine.segment_masks = {0: [mask, mask.copy()]}
        roi = RoiMeta(30, 20, 50, 60, 2, 0, 4, 8, 80, 80, 8, 8)

        boot = engine._bootstrap_mask(80, 160, 0, roi=roi)[0, 0]

        self.assertEqual(boot.shape, (8, 8))
        self.assertTrue(np.all(boot[:, :2] == 0.0))
        self.assertTrue(np.all(boot[:, 6:] == 0.0))
        self.assertGreater(float(boot[:, 2:6].sum()), 0.0)


class _FakeOrtValue:
    def __init__(self, shape):
        self._shape = tuple(shape)

    def shape(self):
        return self._shape


if __name__ == "__main__":
    unittest.main()
