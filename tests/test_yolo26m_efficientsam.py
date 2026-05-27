import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from offline import yolo26m_efficientsam as mod


class _FakeSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path = str(path)
        self.last_inputs = None

    def run(self, output_names, inputs):
        self.last_inputs = inputs
        if "pixel_values" in inputs:
            logits = np.full((1, 300, 80), -10.0, dtype=np.float32)
            boxes = np.zeros((1, 300, 4), dtype=np.float32)
            logits[0, 0, 0] = 0.0
            logits[0, 1, 0] = 1.0
            boxes[0, 1] = [0.5, 0.5, 0.25, 0.5]
            return [logits, boxes]
        mask = np.zeros((1, 1, 1, 1, 8, 8), dtype=np.float32)
        mask[0, 0, 0, 0, 2:6, 2:6] = 0.9
        ious = np.array([[[[0.8]]]], dtype=np.float32)
        return [mask, ious]


def _make_masker(**kwargs):
    with patch.object(mod.ort, "InferenceSession", _FakeSession):
        return mod.Yolo26mEfficientSamMasker(Path("models/yolo26m"), Path("models/efficientsam"), provider="cpu", **kwargs)


class Yolo26mEfficientSamTests(unittest.TestCase):
    def test_detect_uses_pixel_values_and_converts_normalized_boxes(self):
        masker = _make_masker()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        dets = masker.detect(image)

        self.assertEqual(list(masker.yolo.last_inputs), ["pixel_values"])
        self.assertEqual(len(dets), 1)
        np.testing.assert_allclose(dets[0].box_xyxy, [75.0, 0.0, 125.0, 99.0], atol=1e-4)
        self.assertAlmostEqual(dets[0].score, float(mod._sigmoid(np.array([1.0], dtype=np.float32))[0]))
        self.assertEqual(dets[0].class_id, 0)

    def test_detect_default_threshold_filters_low_scores(self):
        masker = _make_masker()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        dets = masker.detect(image, top_k=8)

        self.assertEqual(len(dets), 1)
        self.assertGreaterEqual(dets[0].score, 0.35)

    def test_plausibility_boundaries_include_new_ranges_and_score_gate(self):
        masker = _make_masker()  # default max_box_area=0.50
        shape = (100, 100, 3)

        # In-bounds: area 5%, aspect 0.2, h 50%, score 0.45 -> passes
        self.assertTrue(masker._is_plausible_person_box(mod.Detection(np.array([0, 0, 10, 50], dtype=np.float32), 0.45), shape))
        # In-bounds at upper area boundary: area 50%, score 0.45 -> passes
        self.assertTrue(masker._is_plausible_person_box(mod.Detection(np.array([0, 0, 100, 50], dtype=np.float32), 0.45), shape))
        # Just above upper area boundary: area 51% -> fails
        self.assertFalse(masker._is_plausible_person_box(mod.Detection(np.array([0, 0, 100, 51], dtype=np.float32), 0.45), shape))
        # Score gate: 0.44 below 0.45 -> fails
        self.assertFalse(masker._is_plausible_person_box(mod.Detection(np.array([0, 0, 10, 50], dtype=np.float32), 0.44), shape))
        # Too small (area 0.5%) -> fails
        self.assertFalse(masker._is_plausible_person_box(mod.Detection(np.array([0, 0, 1, 5], dtype=np.float32), 0.45), shape))
        # Too short (height 9%) -> fails
        self.assertFalse(masker._is_plausible_person_box(mod.Detection(np.array([0, 0, 100, 9], dtype=np.float32), 0.45), shape))

    def test_stereo_pairing_prefers_high_score_pair_when_geometry_is_reasonable(self):
        masker = _make_masker()
        left = np.zeros((100, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 100, 3), dtype=np.uint8)
        left_high = mod.Detection(np.array([10, 10, 40, 70], dtype=np.float32), 0.95)
        right_high = mod.Detection(np.array([14, 12, 44, 72], dtype=np.float32), 0.92)
        left_low = mod.Detection(np.array([60, 10, 90, 70], dtype=np.float32), 0.45)
        right_low = mod.Detection(np.array([60, 10, 90, 70], dtype=np.float32), 0.45)

        with patch.object(masker, "detect", side_effect=[[left_high, left_low], [right_high, right_low]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)

        self.assertEqual(info["stereo_mode"], "paired")
        self.assertIs(left_sel[0], left_high)
        self.assertIs(right_sel[0], right_high)

    def test_fallback_score_gate_rejects_low_scores_and_keeps_high_score(self):
        masker = _make_masker()
        left = np.zeros((100, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 100, 3), dtype=np.uint8)
        low = mod.Detection(np.array([0, 0, 2, 2], dtype=np.float32), 0.44)
        high = mod.Detection(np.array([0, 0, 2, 2], dtype=np.float32), 0.60)

        with patch.object(masker, "detect", side_effect=[[low], [low]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "no_detection")
        self.assertEqual(left_sel, [])
        self.assertEqual(right_sel, [])

        with patch.object(masker, "detect", side_effect=[[high], []]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "fallback_score_gate")
        self.assertEqual(left_sel, [high])
        self.assertEqual(right_sel, [])

    def test_top_k_0_unlimited_returns_all_plausible_pairs(self):
        # 5 plausible persons per eye, top_k=0 (unlimited) -> 5 pairs.
        masker = _make_masker(top_k=0)
        left = np.zeros((1000, 1000, 3), dtype=np.uint8)
        right = np.zeros((1000, 1000, 3), dtype=np.uint8)
        lefts = [mod.Detection(np.array([100 + i * 150, 200, 200 + i * 150, 700], dtype=np.float32), 0.95 - 0.02 * i) for i in range(5)]
        rights = [mod.Detection(np.array([110 + i * 150, 205, 210 + i * 150, 705], dtype=np.float32), 0.93 - 0.02 * i) for i in range(5)]
        with patch.object(masker, "detect", side_effect=[lefts, rights]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "paired")
        self.assertEqual(info["pairs"], 5)
        self.assertEqual(len(left_sel), 5)
        self.assertEqual(len(right_sel), 5)

    def test_top_k_0_asymmetric_projects_leftovers_to_missing_eye(self):
        # 3 plausible left, 2 plausible right -> 2 pairs from greedy, plus
        # 1 leftover left projected to right.
        masker = _make_masker(top_k=0)
        left = np.zeros((1000, 1000, 3), dtype=np.uint8)
        right = np.zeros((1000, 1000, 3), dtype=np.uint8)
        l_a = mod.Detection(np.array([100, 200, 200, 700], dtype=np.float32), 0.95)
        l_b = mod.Detection(np.array([400, 200, 500, 700], dtype=np.float32), 0.90)
        l_c = mod.Detection(np.array([700, 200, 800, 700], dtype=np.float32), 0.85)
        r_a = mod.Detection(np.array([110, 205, 210, 705], dtype=np.float32), 0.93)
        r_b = mod.Detection(np.array([410, 210, 510, 710], dtype=np.float32), 0.88)
        with patch.object(masker, "detect", side_effect=[[l_a, l_b, l_c], [r_a, r_b]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["pairs"], 3)
        # 2 pairs from greedy + 1 leftover (l2r) -> at least one "l2r" in projection_dirs
        self.assertIn("l2r", info["projection_dirs"])
        self.assertEqual(len(left_sel), 3)
        self.assertEqual(len(right_sel), 3)
        # Last entry should be the leftover l_c projected to right.
        self.assertIs(left_sel[2], l_c)

    def test_top_k_2_caps_pairs_even_when_unlimited_would_yield_more(self):
        # 5 plausible per eye, top_k=2 -> only 2 pairs, no leftover projection.
        masker = _make_masker(top_k=2)
        left = np.zeros((1000, 1000, 3), dtype=np.uint8)
        right = np.zeros((1000, 1000, 3), dtype=np.uint8)
        lefts = [mod.Detection(np.array([100 + i * 150, 200, 200 + i * 150, 700], dtype=np.float32), 0.95 - 0.02 * i) for i in range(5)]
        rights = [mod.Detection(np.array([110 + i * 150, 205, 210 + i * 150, 705], dtype=np.float32), 0.93 - 0.02 * i) for i in range(5)]
        with patch.object(masker, "detect", side_effect=[lefts, rights]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["pairs"], 2)
        self.assertEqual(len(left_sel), 2)

    def test_top_k_2_returns_two_disjoint_pairs_for_couples(self):
        # Two plausible persons per eye, well separated. With top_k=2 both pairs
        # should be returned, each candidate used at most once, and mode=paired.
        masker = _make_masker(top_k=2)
        left = np.zeros((1000, 1000, 3), dtype=np.uint8)
        right = np.zeros((1000, 1000, 3), dtype=np.uint8)
        l_a = mod.Detection(np.array([100, 200, 200, 700], dtype=np.float32), 0.90)  # area 5%
        l_b = mod.Detection(np.array([700, 200, 800, 700], dtype=np.float32), 0.85)  # area 5%
        r_a = mod.Detection(np.array([110, 205, 210, 705], dtype=np.float32), 0.88)
        r_b = mod.Detection(np.array([705, 210, 805, 710], dtype=np.float32), 0.83)
        with patch.object(masker, "detect", side_effect=[[l_a, l_b], [r_a, r_b]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "paired")
        self.assertEqual(info["pairs"], 2)
        self.assertEqual(len(left_sel), 2)
        self.assertEqual(len(right_sel), 2)
        # Best pair (a, a) by lower geom cost + higher score; second pair (b, b).
        self.assertIs(left_sel[0], l_a)
        self.assertIs(right_sel[0], r_a)
        self.assertIs(left_sel[1], l_b)
        self.assertIs(right_sel[1], r_b)

    def test_top_k_1_preserves_single_pair_legacy_mode_label(self):
        masker = _make_masker(top_k=1)
        left = np.zeros((100, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 100, 3), dtype=np.uint8)
        l = mod.Detection(np.array([10, 20, 30, 70], dtype=np.float32), 0.80)
        r = mod.Detection(np.array([11, 22, 31, 72], dtype=np.float32), 0.82)
        with patch.object(masker, "detect", side_effect=[[l], [r]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "paired")
        self.assertEqual(info["pairs"], 1)
        self.assertEqual(len(left_sel), 1)
        self.assertEqual(len(right_sel), 1)

    def test_top_k_2_with_one_pair_projection_uses_mixed_label_only_when_mixed(self):
        # Pair 1 healthy; Pair 2 has cross-eye size mismatch -> single
        # projection (r2l). Two pairs, only one projected -> "paired_project_mixed".
        masker = _make_masker(top_k=2, cross_eye_area_ratio=1.5)
        left = np.zeros((1000, 1000, 3), dtype=np.uint8)
        right = np.zeros((1000, 1000, 3), dtype=np.uint8)
        l_a = mod.Detection(np.array([100, 200, 200, 700], dtype=np.float32), 0.90)  # area 5%
        l_b_huge = mod.Detection(np.array([700, 100, 1000, 800], dtype=np.float32), 0.70)  # area 21%
        r_a = mod.Detection(np.array([110, 205, 210, 705], dtype=np.float32), 0.88)
        r_b_tight = mod.Detection(np.array([750, 250, 850, 700], dtype=np.float32), 0.80)  # area 4.5%
        with patch.object(masker, "detect", side_effect=[[l_a, l_b_huge], [r_a, r_b_tight]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["pairs"], 2)
        self.assertEqual(info["stereo_mode"], "paired_project_mixed")
        # Pair 1 (a, a) untouched; pair 2 (b, b) had r_b higher score and
        # smaller box -> projected to L
        np.testing.assert_allclose(left_sel[1].box_xyxy, r_b_tight.box_xyxy, atol=1e-3)

    def test_top_k_2_projects_all_plausible_when_one_eye_missing(self):
        # No right-eye plausible; project all left up to top_k=2 to right.
        masker = _make_masker(top_k=2)
        left = np.zeros((1000, 1000, 3), dtype=np.uint8)
        right = np.zeros((1000, 1000, 3), dtype=np.uint8)
        l_a = mod.Detection(np.array([100, 200, 200, 700], dtype=np.float32), 0.90)
        l_b = mod.Detection(np.array([700, 200, 800, 700], dtype=np.float32), 0.85)
        with patch.object(masker, "detect", side_effect=[[l_a, l_b], []]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "project_left_to_right")
        self.assertEqual(info["pairs"], 2)
        self.assertEqual(len(left_sel), 2)
        self.assertEqual(len(right_sel), 2)

    def test_cross_eye_area_mismatch_projects_higher_score_side(self):
        # Simulate t=2s on test_8k_3.mp4: L box covers 24% of eye, R box covers 8%
        # (3x area ratio). Higher score on R should project to L.
        masker = _make_masker(cross_eye_area_ratio=1.5)
        left = np.zeros((100, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 100, 3), dtype=np.uint8)
        l_big = mod.Detection(np.array([10, 10, 70, 50], dtype=np.float32), 0.70)  # area=0.24
        r_tight = mod.Detection(np.array([20, 20, 60, 40], dtype=np.float32), 0.80)  # area=0.08
        with patch.object(masker, "detect", side_effect=[[l_big], [r_tight]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        self.assertEqual(info["stereo_mode"], "paired_project_r2l")
        # Both eyes now have the R-tight box dimensions, projected to each eye
        np.testing.assert_allclose(left_sel[0].box_xyxy, r_tight.box_xyxy, atol=1e-3)
        np.testing.assert_allclose(right_sel[0].box_xyxy, r_tight.box_xyxy, atol=1e-3)
        # Within-threshold ratio leaves mode as plain paired
        masker_loose = _make_masker(cross_eye_area_ratio=5.0)
        with patch.object(masker_loose, "detect", side_effect=[[l_big], [r_tight]]):
            _, _, info_loose = masker_loose.select_stereo_detections(left, right)
        self.assertEqual(info_loose["stereo_mode"], "paired")

    def test_max_box_area_rejects_fill_the_frame_in_plausibility_and_fallback(self):
        # Simulate t=20-24s on test_8k_3.mp4: 0.58-0.62 area "fill the frame"
        # boxes that fail plausibility AND should not slip through fallback.
        masker = _make_masker(max_box_area=0.50)
        left = np.zeros((100, 100, 3), dtype=np.uint8)
        right = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box covering 58% of eye, aspect 1.7 (mirrors actual t=20s)
        fill = mod.Detection(np.array([0, 30, 100, 88], dtype=np.float32), 0.80)
        self.assertFalse(masker._is_plausible_person_box(fill, left.shape))
        self.assertFalse(masker._is_within_max_box_area(fill, left.shape))
        with patch.object(masker, "detect", side_effect=[[fill], [fill]]):
            left_sel, right_sel, info = masker.select_stereo_detections(left, right)
        # Plausibility rejects -> falls to fallback -> max_box_area gate also rejects -> no_detection
        self.assertEqual(info["stereo_mode"], "no_detection")
        self.assertEqual(left_sel, [])
        self.assertEqual(right_sel, [])

    def test_gap_fill_boundary_extends_first_active_backward_and_last_active_forward(self):
        # Records: [inactive, inactive, active(frame=120, mask=A), active(frame=180, mask=B),
        #           inactive, inactive]. With fill_boundaries=True, samples 0/1 take mask A
        # and samples 4/5 take mask B. boundary_filled markers should be set.
        records = [
            {"frame": 0, "src_idx": 0, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 60, "src_idx": 60, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 120, "src_idx": 120, "active": True, "object_count": 1, "scene_cut": False},
            {"frame": 180, "src_idx": 180, "active": True, "object_count": 1, "scene_cut": False},
            {"frame": 240, "src_idx": 240, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 300, "src_idx": 300, "active": False, "object_count": 0, "scene_cut": False},
        ]
        mask_a = np.ones((4, 4), dtype=np.float32)
        mask_b = np.full((4, 4), 0.5, dtype=np.float32)
        masks = {120: [mask_a], 180: [mask_b]}

        filled = mod._fill_short_inactive_gaps(records, masks, max_gap_frames=600, fill_boundaries=True, respect_scene_cuts=True)

        self.assertIn(0, filled)
        self.assertIn(60, filled)
        self.assertIn(240, filled)
        self.assertIn(300, filled)
        np.testing.assert_array_equal(masks[0][0], mask_a)
        np.testing.assert_array_equal(masks[60][0], mask_a)
        np.testing.assert_array_equal(masks[240][0], mask_b)
        np.testing.assert_array_equal(masks[300][0], mask_b)
        self.assertEqual(records[0].get("boundary_filled"), "start")
        self.assertEqual(records[5].get("boundary_filled"), "end")

    def test_gap_fill_scene_aware_splits_middle_gap_at_scene_cut(self):
        # Inactive run [r1, r2, r3] between active r0(prev) and r4(next). r2 has
        # scene_cut=True. r1 should take prev mask; r2 and r3 should take next mask.
        records = [
            {"frame": 0, "src_idx": 0, "active": True, "object_count": 1, "scene_cut": False},
            {"frame": 60, "src_idx": 60, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 120, "src_idx": 120, "active": False, "object_count": 0, "scene_cut": True},
            {"frame": 180, "src_idx": 180, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 240, "src_idx": 240, "active": True, "object_count": 1, "scene_cut": False},
        ]
        prev_mask = np.ones((4, 4), dtype=np.float32)
        next_mask = np.full((4, 4), 0.5, dtype=np.float32)
        masks = {0: [prev_mask], 240: [next_mask]}

        filled = mod._fill_short_inactive_gaps(records, masks, max_gap_frames=600, fill_boundaries=False, respect_scene_cuts=True)

        self.assertEqual(sorted(filled), [60, 120, 180])
        np.testing.assert_array_equal(masks[60][0], prev_mask)   # before cut -> prev
        np.testing.assert_array_equal(masks[120][0], next_mask)  # cut sample -> next
        np.testing.assert_array_equal(masks[180][0], next_mask)  # after cut -> next

    def test_gap_fill_scene_aware_blocks_boundary_when_first_active_is_scene_cut(self):
        # First active is a scene cut: pre-cut scene is genuinely different,
        # so backward fill must be skipped.
        records = [
            {"frame": 0, "src_idx": 0, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 60, "src_idx": 60, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 120, "src_idx": 120, "active": True, "object_count": 1, "scene_cut": True},
            {"frame": 180, "src_idx": 180, "active": True, "object_count": 1, "scene_cut": False},
        ]
        mask = np.ones((4, 4), dtype=np.float32)
        masks = {120: [mask], 180: [mask]}

        filled = mod._fill_short_inactive_gaps(records, masks, max_gap_frames=600, fill_boundaries=True, respect_scene_cuts=True)

        self.assertEqual(filled, [])
        self.assertNotIn(0, masks)
        self.assertNotIn(60, masks)

    def test_gap_fill_boundary_capped_by_max_gap_frames(self):
        # If the first active sits beyond max_gap_frames, don't backward-fill the
        # entire intro: leave it as no-mask.
        records = [
            {"frame": 0, "src_idx": 0, "active": False, "object_count": 0, "scene_cut": False},
            {"frame": 800, "src_idx": 800, "active": True, "object_count": 1, "scene_cut": False},
        ]
        mask = np.ones((4, 4), dtype=np.float32)
        masks = {800: [mask]}

        filled = mod._fill_short_inactive_gaps(records, masks, max_gap_frames=600, fill_boundaries=True, respect_scene_cuts=True)

        self.assertEqual(filled, [])
        self.assertNotIn(0, masks)

    def test_sam_mask_postprocess_binarizes_and_erodes(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        box = np.array([8, 8, 24, 24], dtype=np.float32)
        eroded = _make_masker(binarize_mask=True, mask_erode_px=1)._sam_mask_for_box(image, box, (8, 8))
        soft = _make_masker(binarize_mask=False, mask_erode_px=1)._sam_mask_for_box(image, box, (8, 8))

        self.assertEqual(set(np.unique(eroded).tolist()), {0.0, 1.0})
        self.assertLess(float(eroded.sum()), float((soft >= 0.5).sum()))
        self.assertNotEqual(set(np.unique(soft).tolist()), {0.0, 1.0})
        self.assertAlmostEqual(float(soft.max()), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
