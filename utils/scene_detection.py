"""Lightweight scene-cut detection helpers."""
from __future__ import annotations

import cv2
import numpy as np


class SceneCutDetector:
    """HSV-Bhattacharyya scene cut detector with reference EMA and cooldown."""

    def __init__(
        self,
        threshold: float = 0.4,
        cooldown_frames: int = 24,
        ref_ema_alpha: float = 0.95,
        downsample_height: int = 540,
    ) -> None:
        self.threshold = float(threshold)
        self.cooldown = max(0, int(cooldown_frames))
        self.ref_ema_alpha = float(ref_ema_alpha)
        self.downsample_height = max(2, int(downsample_height))
        self.last_distance = 0.0
        self._ref_hist: np.ndarray | None = None
        self._cooldown_left = 0

    def step(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        if h > self.downsample_height:
            new_w = max(2, int(w * self.downsample_height / h))
            frame_bgr = cv2.resize(frame_bgr, (new_w, self.downsample_height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

        if self._ref_hist is None:
            self._ref_hist = hist
            self.last_distance = 0.0
            return False

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self.last_distance = 0.0
            self._update_ref(hist)
            return False

        self.last_distance = float(cv2.compareHist(self._ref_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        if self.last_distance > self.threshold:
            self._ref_hist = hist
            self._cooldown_left = self.cooldown
            return True
        self._update_ref(hist)
        return False

    def _update_ref(self, hist: np.ndarray) -> None:
        assert self._ref_hist is not None
        self._ref_hist = self._ref_hist * self.ref_ema_alpha + hist * (1.0 - self.ref_ema_alpha)

    def reset(self) -> None:
        self.last_distance = 0.0
        self._ref_hist = None
        self._cooldown_left = 0
