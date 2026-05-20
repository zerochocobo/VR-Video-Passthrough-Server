"""Decoded-frame conversion helpers for offline prepass sampling."""
from __future__ import annotations

import cv2
import numpy as np


def decoded_frame_to_bgr(frame) -> np.ndarray:
    """Convert a PyNv decoded NV12/P016 frame to CPU BGR for CV prepasses."""
    import cupy as cp

    h = int(frame.height)
    w = int(frame.width)
    y = frame.y.as_cupy()
    uv = frame.uv.as_cupy()
    if y.dtype == cp.uint16 or uv.dtype == cp.uint16:
        y8 = cp.right_shift(y.reshape(h, w), 8).astype(cp.uint8, copy=False)
        uv8 = cp.right_shift(uv.reshape(h // 2, w), 8).astype(cp.uint8, copy=False)
        nv12 = cp.asnumpy(cp.concatenate([y8, uv8], axis=0))
    else:
        nv12 = cp.asnumpy(cp.concatenate([
            y.reshape(h, w),
            uv.reshape(h // 2, w),
        ], axis=0))
    if nv12.dtype != np.uint8:
        nv12 = nv12.astype(np.uint8, copy=False)
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
