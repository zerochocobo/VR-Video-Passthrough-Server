from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from offline import face_beauty_engine as fb
from offline.face_beauty_gpu import GpuFaceBeautyProcessor
from pipeline.pynv_stream import _face_beauty_cfr_plan


class _Tracker:
    def smooth(self, faces) -> None:
        pass


def _face() -> fb.DetectedFace:
    return fb.DetectedFace(
        np.array([100, 100, 300, 300], np.float32),
        0.9,
        np.array([[140, 170], [250, 170], [195, 210], [155, 260], [235, 260]], np.float32),
    )


def test_landmarker_interval_reuses_points_between_refinements() -> None:
    processor = object.__new__(GpuFaceBeautyProcessor)
    processor.cp = None
    processor.options = fb.BeautyOptions(
        enhancer="none",
        use_region_mask=False,
        use_landmarker=True,
        detect_interval=2,
        landmark_interval=2,
        max_faces=1,
    )
    processor.landmarker = object()
    processor.tracker = _Tracker()
    processor._cached_faces = []
    processor._frame_index = 0
    processor._detect = lambda frame, score: [_face()]
    processor._vr_enabled = lambda width, height: False
    refined: list[int] = []
    processor._refine_landmarks = lambda frame, face: refined.append(processor._frame_index)
    processor._process_face = lambda frame, face, width, height: None

    frame = SimpleNamespace(shape=(1080, 1920, 3))
    for _ in range(4):
        _, stats = processor._process(frame)
        assert stats.processed == 1

    assert refined == [1, 3]


def test_face_beauty_cfr_plan_caps_5994_source_at_40fps_and_keeps_seek() -> None:
    timing = SimpleNamespace(
        source_fps=60000.0 / 1001.0,
        duration=2046.344,
        effective_fps=lambda cap: min(60000.0 / 1001.0, float(cap)),
    )
    source_fps, output_fps, start_out, target, initial_src = _face_beauty_cfr_plan(
        timing, 0.0, 122658, 1200.0, 40.0)

    assert abs(source_fps - 59.94006) < 0.001
    assert output_fps == 40.0
    assert start_out == 48000
    assert initial_src == 71928
    assert target > 0


def test_blind_enhancer_rejects_low_source_face_coverage() -> None:
    face = _face()
    face.landmark_score = 0.9

    assert not fb.enhancer_is_safe(face, fb.MIN_ENHANCER_FACE_COVERAGE - 0.001)
    assert fb.enhancer_is_safe(face, fb.MIN_ENHANCER_FACE_COVERAGE)


def test_blind_enhancer_uses_landmark_score_when_parser_is_disabled() -> None:
    face = _face()
    face.landmark_score = fb.MIN_ENHANCER_LANDMARK_SCORE - 0.001
    assert not fb.enhancer_is_safe(face, None)

    face.landmark_score = fb.MIN_ENHANCER_LANDMARK_SCORE
    assert fb.enhancer_is_safe(face, None)


def test_cpu_engine_parses_source_before_enhancement_and_skips_hallucination(monkeypatch) -> None:
    processor = object.__new__(fb.FaceBeautyEngine)
    processor.options = fb.BeautyOptions(enhancer="gpen_bfr_256", use_region_mask=True)
    processor.crop_size = 4
    processor.template = "arcface_128"
    processor._box_mask = np.ones((4, 4), np.float32)

    source_crop = np.zeros((4, 4, 3), np.uint8)
    parsed_inputs: list[np.ndarray] = []

    class _Parser:
        def parse(self, crop):
            parsed_inputs.append(crop.copy())
            return np.zeros((4, 4), np.uint8)

    class _Enhancer:
        calls = 0

        def enhance(self, crop):
            self.calls += 1
            return np.full_like(crop, 255)

    processor.parser = _Parser()
    processor.enhancer = _Enhancer()
    monkeypatch.setattr(
        fb, "warp_face",
        lambda frame, points, size, template: (source_crop.copy(), np.eye(2, 3, dtype=np.float32)),
    )
    monkeypatch.setattr(fb, "retouch_crop", lambda crop, labels, options, mask: crop)
    monkeypatch.setattr(fb, "paste_back", lambda frame, crop, mask, matrix: frame)

    processor._process_face(np.zeros((8, 8, 3), np.uint8), _face())

    assert processor.enhancer.calls == 0
    assert len(parsed_inputs) == 1
    assert np.array_equal(parsed_inputs[0], source_crop)
