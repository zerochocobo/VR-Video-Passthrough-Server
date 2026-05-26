"""MatAnyone2 ROI helpers for fixed-input quality mode."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoiMeta:
    x0: int
    y0: int
    x1: int
    y1: int
    model_x0: int
    model_y0: int
    model_w: int
    model_h: int
    eye_w: int
    eye_h: int
    model_w_total: int
    model_h_total: int

    @property
    def roi_w(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def roi_h(self) -> int:
        return max(1, self.y1 - self.y0)


def mask_bbox_xyxy(mask_2d: np.ndarray, threshold: float = 0.05) -> tuple[int, int, int, int] | None:
    mask = np.asarray(mask_2d)
    if mask.ndim != 2 or mask.size <= 0:
        return None
    ys, xs = np.where(mask > float(threshold))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _scale_bbox_to_eye(
    bbox: tuple[int, int, int, int],
    mask_w: int,
    mask_h: int,
    eye_w: int,
    eye_h: int,
) -> tuple[float, float, float, float]:
    sx = float(eye_w) / max(1.0, float(mask_w))
    sy = float(eye_h) / max(1.0, float(mask_h))
    x0, y0, x1, y1 = bbox
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def roi_from_mask(
    mask_2d: np.ndarray,
    eye_w: int,
    eye_h: int,
    model_w: int,
    model_h: int,
    expand: float = 0.30,
    max_eye_fraction: float = 0.70,
) -> RoiMeta | None:
    """Build a fixed segment ROI from a bootstrap mask.

    Returns None when the ROI is empty or too large to be useful.
    """
    mask = np.asarray(mask_2d)
    if mask.ndim != 2:
        return None
    bbox = mask_bbox_xyxy(mask)
    if bbox is None:
        return None
    mh, mw = (int(v) for v in mask.shape[:2])
    fx0, fy0, fx1, fy1 = _scale_bbox_to_eye(bbox, mw, mh, eye_w, eye_h)
    bw = max(1.0, fx1 - fx0)
    bh = max(1.0, fy1 - fy0)
    pad_x = bw * max(0.0, float(expand))
    pad_y = bh * max(0.0, float(expand))
    x0 = max(0, int(np.floor(fx0 - pad_x)))
    y0 = max(0, int(np.floor(fy0 - pad_y)))
    x1 = min(int(eye_w), int(np.ceil(fx1 + pad_x)))
    y1 = min(int(eye_h), int(np.ceil(fy1 + pad_y)))
    if x1 <= x0 or y1 <= y0:
        return None
    frac = ((x1 - x0) * (y1 - y0)) / float(max(1, int(eye_w) * int(eye_h)))
    if frac > max(0.0, min(1.0, float(max_eye_fraction))):
        return None

    roi_w = x1 - x0
    roi_h = y1 - y0
    scale = min(float(model_w) / float(roi_w), float(model_h) / float(roi_h))
    scaled_w = max(1, min(int(model_w), int(round(roi_w * scale))))
    scaled_h = max(1, min(int(model_h), int(round(roi_h * scale))))
    model_x0 = max(0, (int(model_w) - scaled_w) // 2)
    model_y0 = max(0, (int(model_h) - scaled_h) // 2)
    return RoiMeta(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        model_x0=model_x0,
        model_y0=model_y0,
        model_w=scaled_w,
        model_h=scaled_h,
        eye_w=int(eye_w),
        eye_h=int(eye_h),
        model_w_total=int(model_w),
        model_h_total=int(model_h),
    )
