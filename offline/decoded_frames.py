"""Decoded-frame conversion helpers for offline prepass sampling."""
from __future__ import annotations

import cv2
import numpy as np


def _decoded_frame_planes_to_nv12_cpu(frame) -> tuple[np.ndarray, np.ndarray]:
    """Copy decoded NV12/P016 planes to CPU 8-bit NV12 Y and UV arrays."""
    import cupy as cp

    h = int(frame.height)
    w = int(frame.width)
    y = frame.y.as_cupy()
    uv = frame.uv.as_cupy()
    if y.dtype == cp.uint16 or uv.dtype == cp.uint16:
        y_cpu = cp.asnumpy(cp.right_shift(y.reshape(h, w), 8).astype(cp.uint8, copy=False))
        uv_cpu = cp.asnumpy(cp.right_shift(uv.reshape(h // 2, w), 8).astype(cp.uint8, copy=False))
    else:
        y_cpu = cp.asnumpy(y.reshape(h, w))
        uv_cpu = cp.asnumpy(uv.reshape(h // 2, w))
    if y_cpu.dtype != np.uint8:
        y_cpu = y_cpu.astype(np.uint8, copy=False)
    if uv_cpu.dtype != np.uint8:
        uv_cpu = uv_cpu.astype(np.uint8, copy=False)
    return np.ascontiguousarray(y_cpu), np.ascontiguousarray(uv_cpu)


def decoded_frame_to_bgr(frame) -> np.ndarray:
    """Convert a PyNv decoded NV12/P016 frame to CPU BGR for CV prepasses."""
    y_cpu, uv_cpu = _decoded_frame_planes_to_nv12_cpu(frame)
    nv12 = np.empty((y_cpu.shape[0] + uv_cpu.shape[0], y_cpu.shape[1]), dtype=np.uint8)
    nv12[: y_cpu.shape[0], :] = y_cpu
    nv12[y_cpu.shape[0] :, :] = uv_cpu
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def decoded_frame_to_sbs_eye_rgbs(frame) -> list[np.ndarray]:
    """Convert an SBS decoded NV12/P016 frame directly into left/right RGB eyes.

    This avoids materializing a full-width BGR frame, which is a large transient
    allocation for 8K SBS sources during MatAnyone2/SAM3 prepasses.
    """
    y_cpu, uv_cpu = _decoded_frame_planes_to_nv12_cpu(frame)
    h, w = y_cpu.shape
    half = w // 2
    if half <= 0 or half % 2:
        raise ValueError(f"SBS eye width must be a positive even number, got frame_width={w}")
    eye_rgbs: list[np.ndarray] = []
    for x0 in (0, half):
        nv12_eye = np.empty((h + h // 2, half), dtype=np.uint8)
        nv12_eye[:h, :] = y_cpu[:, x0 : x0 + half]
        nv12_eye[h:, :] = uv_cpu[:, x0 : x0 + half]
        eye_rgbs.append(cv2.cvtColor(nv12_eye, cv2.COLOR_YUV2RGB_NV12))
        del nv12_eye
    return eye_rgbs


def scene_bgr_from_rgb(image_rgb: np.ndarray, downsample_height: int = 540) -> np.ndarray:
    """Return a BGR scene-detection image without allocating a full-res copy."""
    h, w = image_rgb.shape[:2]
    target_h = max(2, int(downsample_height))
    if h > target_h:
        target_w = max(2, int(w * target_h / h))
        image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
