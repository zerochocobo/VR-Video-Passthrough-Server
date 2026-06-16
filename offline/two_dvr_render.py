"""Depth -> stereo -> VR projection rendering (pure numpy).

Ported from the standalone tool_2dvr ``logic.py`` so the offline 2D->VR
pipeline has no torch dependency: depth comes from the DA3 ONNX engine
(:mod:`offline.da3_depth`) and everything here is numpy. Covers the two
production hole-fill modes (``soft_shift`` forward warp, ``inverse_warp`` fast
inverse sampling) and the three projections (flat3d / hequirect-180 / fisheye-180).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Percentile normalization only needs a representative sample of pixels; sampling
# instead of scanning the whole frame keeps _normalize_near cheap at 4K.
_NORM_SAMPLE = 16384

PROJECTION_FLAT_3D = "flat3d"
PROJECTION_HEQUIRECT = "hequirect"
PROJECTION_FISHEYE = "fisheye"
PROJECTIONS = {PROJECTION_FLAT_3D, PROJECTION_HEQUIRECT, PROJECTION_FISHEYE}
DEFAULT_PROJECTION = PROJECTION_FLAT_3D

DEFAULT_EYE_DISTANCE_MM = 65.0
DEFAULT_MAX_DISPARITY_RATIO = 0.035
DEFAULT_STRENGTH = 1.0
MIN_STRENGTH = 0.1
MAX_STRENGTH = 3.0

HOLE_FILL_SOFT_SHIFT = "soft_shift"
HOLE_FILL_INVERSE_WARP = "inverse_warp"
HOLE_FILL_MODES = {HOLE_FILL_SOFT_SHIFT, HOLE_FILL_INVERSE_WARP}
DEFAULT_HOLE_FILL_MODE = HOLE_FILL_SOFT_SHIFT

DEFAULT_FLAT_FOV_DEG = 80.0
MIN_FLAT_FOV_DEG = 1.0
MAX_FLAT_FOV_DEG = 179.0


@dataclass(frozen=True)
class ProjectionMap:
    out_w: int
    out_h: int
    map_x: np.ndarray
    map_y: np.ndarray
    mask: np.ndarray
    # flat3d is a 1:1 identity map -- callers skip resampling entirely.
    is_identity: bool = False


# --- depth -> near/disparity ------------------------------------------------


def _smooth_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.size == 0:
        return arr
    # 3x3 binomial blur == separable Gaussian; cv2 runs it far faster than the
    # numpy shifted-add form.
    return cv2.GaussianBlur(arr, (3, 3), 0, borderType=cv2.BORDER_REPLICATE)


def _normalize_near(depth: np.ndarray) -> np.ndarray:
    """Map DA3 distance-like depth to a 0..1 near buffer (1 = closest).

    Stereo disparity is approximately proportional to inverse depth, so we
    invert, clip to the 5/95 percentile band, and normalize. DA3 depth is dense
    and positive, so we skip the per-pixel validity mask (the old full-array
    boolean gather cost ~40 ms/frame at 1080p) and estimate the band from a
    strided subsample instead.
    """
    d = np.asarray(depth, dtype=np.float32)
    inv = np.reciprocal(np.maximum(d, 1e-6))
    sample = inv[::4, ::4].ravel()
    if sample.size > _NORM_SAMPLE:
        sample = sample[:: max(1, sample.size // _NORM_SAMPLE)]
    lo, hi = (float(v) for v in np.percentile(sample, [5.0, 95.0]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(d.shape, dtype=np.float32)
    near = (inv - np.float32(lo)) * np.float32(1.0 / (hi - lo))
    np.clip(near, 0.0, 1.0, out=near)
    return near


def _dilate_near_fg(near: np.ndarray) -> np.ndarray:
    """Grow the foreground (near) by a few px so the disocclusion gap is bounded
    by clean background.

    Depth boundaries never align exactly with the colour silhouette: a thin ring
    of object-coloured pixels gets an intermediate/low near and, instead of
    travelling with the foreground, stays put as a foreground-coloured sliver on
    the background side of the gap -- which the hole-fill then smears in. Dilating
    near makes that boundary ring travel WITH the foreground; the leftover
    intermediate-near pixels are now background-coloured, so any residual sliver
    blends invisibly into the filled background. Radius scales with the near-map
    width so it matches the warp resolution (toggle/snap still hardens the edge).
    """
    n = np.asarray(near, dtype=np.float32)
    if n.ndim != 2 or n.shape[1] < 3:
        return n
    r = max(1, int(round(n.shape[1] / 512.0)))
    k = 2 * r + 1
    return cv2.dilate(n, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def near_from_depth(depth: np.ndarray, hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE) -> np.ndarray:
    """Depth -> normalized near buffer, smoothing only when it helps.

    ``inverse_warp`` builds a continuous resampling map, so a 3x3 depth blur
    suppresses stair-stepping along disparity edges. ``soft_shift`` (forward
    warp) must NOT pre-smooth: the blurred depth discontinuity gives the boundary
    pixel an intermediate disparity that scatters a 1px foreground sliver into
    the disocclusion gap. The directional hole-fill then grabs that sliver
    instead of the revealed background and fattens foreground objects (faces /
    limbs visibly widen, asymmetrically between the eyes). Sharp depth edges keep
    the gap clean so it fills from the true background.
    """
    if hole_fill_mode == HOLE_FILL_SOFT_SHIFT:
        # No pre-smooth; grow foreground so the gap is bounded by clean
        # background (the GPU warp kernel still snaps the edge hard via _near_at).
        return _dilate_near_fg(_normalize_near(depth))
    return _normalize_near(_smooth_depth(depth))


def _sharpen_near_edges(near: np.ndarray, window: int = 6) -> np.ndarray:
    """Toggle (morphological-contrast) sharpen of the near map's horizontal edges.

    soft_shift's forward warp maps the intermediate near values of a soft depth
    contour to scattered foreground slivers inside the disocclusion gap; the
    hole-fill then smears foreground into it (faces/limbs fatten). Snapping each
    pixel to the nearer of its local horizontal min/max collapses the soft
    contour to a hard edge in place (no net widening), so the gap stays a clean
    hole bounded by solid fg/bg and fills from the true background. The GPU
    renderer does the equivalent snap inside its warp kernel (_near_at).
    """
    n = _dilate_near_fg(np.asarray(near, dtype=np.float32))
    if n.ndim != 2 or n.shape[1] < 3:
        return n
    k = max(3, 2 * int(window) + 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    lo = cv2.erode(n, se)
    hi = cv2.dilate(n, se)
    return np.where((n - lo) >= (hi - n), hi, lo).astype(np.float32)


def _max_disparity_pixels(src_w: int, eye_distance_mm: float) -> float:
    eye_scale = max(0.1, float(eye_distance_mm) / DEFAULT_EYE_DISTANCE_MM)
    return max(2.0, min(96.0, float(src_w) * DEFAULT_MAX_DISPARITY_RATIO * eye_scale))


def strength_multiplier(value: float | str | None = DEFAULT_STRENGTH) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError):
        strength = DEFAULT_STRENGTH
    return max(MIN_STRENGTH, min(MAX_STRENGTH, strength))


def effective_eye_distance_mm(
    eye_distance_mm: float | str | None = DEFAULT_EYE_DISTANCE_MM,
    strength: float | str | None = DEFAULT_STRENGTH,
) -> float:
    try:
        base_eye = float(eye_distance_mm)
    except (TypeError, ValueError):
        base_eye = DEFAULT_EYE_DISTANCE_MM
    return max(1.0, base_eye) * strength_multiplier(strength)


# --- sampling helpers -------------------------------------------------------


def _sample_horizontal_rgb(image: np.ndarray, map_x: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    mx = np.ascontiguousarray(map_x, dtype=np.float32)
    my = np.ascontiguousarray(
        np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    )
    return cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_DILATE_KERNEL = np.ones((3, 3), np.uint8)


def _dilate_mask_np(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    dilated = cv2.dilate(np.asarray(mask, dtype=np.uint8), _DILATE_KERNEL, iterations=max(1, int(iterations)))
    return dilated.astype(bool)


def _box_blur_rgb(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.blur(image, (kernel_size, kernel_size))


def _shift_fill_holes_rgb(image: np.ndarray, holes: np.ndarray, direction: int) -> np.ndarray:
    """Vectorized horizontal hole fill from the preferred ``direction``.

    Each hole takes the nearest valid pixel on the preferred side, falling back
    to the other side. Replaces the previous per-column iterative shift, which
    looped up to width times per frame.
    """
    holes = np.asarray(holes, dtype=bool)
    if not np.any(holes):
        return image
    h, w, _ = image.shape
    cols = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    valid = ~holes
    left_idx = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)
    right_idx = np.minimum.accumulate(np.where(valid, cols, w)[:, ::-1], axis=1)[:, ::-1]
    left_ok = left_idx >= 0
    right_ok = right_idx < w
    if direction < 0:
        primary_idx, primary_ok, secondary_idx = left_idx, left_ok, right_idx
    else:
        primary_idx, primary_ok, secondary_idx = right_idx, right_ok, left_idx
    pick = np.clip(np.where(primary_ok, primary_idx, secondary_idx), 0, w - 1)
    rows = np.arange(h, dtype=np.int32)[:, None]
    src = image[rows, pick]
    out = image.copy()
    fillable = holes & (left_ok | right_ok)
    out[fillable] = src[fillable]
    return out


def _soft_blend_holes_rgb(image: np.ndarray, holes: np.ndarray,
                          near_buffer: np.ndarray | None = None) -> np.ndarray:
    """Soften only the filled disocclusion holes, keeping the foreground
    silhouette hard.

    The blur is masked to exclude foreground (the occluder): foreground pixels
    are kept out of the box-blur average, so the stretched-background fill is
    only smoothed against real background -- the seam the fill creates -- and the
    hard fg/bg boundary is never feathered. ``near_buffer`` is the warped near
    map (>=0 where written, <0 at holes); foreground is the high-near side of the
    local depth step. The blend touches hole pixels only.
    """
    holes = np.asarray(holes, dtype=bool)
    if not np.any(holes):
        return image
    base = image.astype(np.float32)
    if near_buffer is None:
        # No depth info: fall back to a plain hole-only soft blend.
        bg = np.ones(holes.shape, dtype=np.float32)
    else:
        written = near_buffer >= 0.0
        nv = near_buffer[written]
        if nv.size:
            nmin, nmax = float(nv.min()), float(nv.max())
        else:
            nmin, nmax = 0.0, 0.0
        thr = 0.5 * (nmin + nmax) if (nmax - nmin) > 0.30 else np.inf
        # Background or filled hole = blur source; foreground (occluder) excluded.
        bg = ((~written) | (near_buffer <= thr)).astype(np.float32)
    k = 5
    num = cv2.blur(base * bg[:, :, None], (k, k))
    den = cv2.blur(bg, (k, k))
    blurred = num / np.maximum(den[:, :, None], 1e-6)
    alpha = holes[:, :, None].astype(np.float32) * 0.35
    return np.clip(base * (1.0 - alpha) + blurred * alpha, 0.0, 255.0).astype(np.uint8)


# --- stereo warp ------------------------------------------------------------


def _forward_warp_eye_rgb(frame_rgb, near, max_shift, eye_sign):
    h, w, _ = frame_rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    target_x = np.rint(xx.astype(np.float32) + near * (max_shift * 0.5 * eye_sign)).astype(np.int32)
    valid = (target_x >= 0) & (target_x < w)

    flat_target = (yy * w + target_x).reshape(-1)
    valid_flat = valid.reshape(-1)
    priority = near.reshape(-1).astype(np.float32, copy=False)
    target_valid = flat_target[valid_flat]

    zbuf_flat = np.full(h * w, -1.0, dtype=np.float32)
    if target_valid.size:
        np.maximum.at(zbuf_flat, target_valid, priority[valid_flat])

    safe_target = np.clip(flat_target, 0, h * w - 1)
    winners = valid_flat & (priority >= zbuf_flat[safe_target] - 1e-6)
    out_flat = np.zeros((h * w, 3), dtype=np.uint8)
    out_flat[safe_target[winners]] = frame_rgb.reshape(-1, 3)[winners]

    near_buffer = zbuf_flat.reshape(h, w)
    holes = near_buffer < 0.0
    return out_flat.reshape(h, w, 3), holes, near_buffer


def _make_soft_shift_pair(frame_rgb, depth, eye_distance_mm):
    h, w, _ = frame_rgb.shape
    # soft_shift must use the un-smoothed depth -- see near_from_depth -- then
    # snap soft contours to hard edges so the forward warp doesn't spray
    # foreground slivers into the disocclusion gap (see _sharpen_near_edges).
    near = _sharpen_near_edges(_normalize_near(depth))
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    left_raw, left_holes, left_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, 1.0)
    right_raw, right_holes, right_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, -1.0)
    left = _soft_blend_holes_rgb(_shift_fill_holes_rgb(left_raw, left_holes, -1), left_holes, left_near)
    right = _soft_blend_holes_rgb(_shift_fill_holes_rgb(right_raw, right_holes, 1), right_holes, right_near)
    return left, right


def _make_inverse_warp_pair(frame_rgb, depth, eye_distance_mm):
    h, w, _ = frame_rgb.shape
    near = _normalize_near(_smooth_depth(depth))
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    disparity = near * max_shift
    cols = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    left = _sample_horizontal_rgb(frame_rgb, cols - disparity * 0.5)
    right = _sample_horizontal_rgb(frame_rgb, cols + disparity * 0.5)
    return left, right


def make_stereo_pair(frame_rgb, depth, eye_distance_mm=DEFAULT_EYE_DISTANCE_MM,
                     hole_fill_mode=DEFAULT_HOLE_FILL_MODE):
    if depth.shape != frame_rgb.shape[:2]:
        raise ValueError(f"depth {depth.shape} does not match frame {frame_rgb.shape[:2]}")
    if hole_fill_mode == HOLE_FILL_INVERSE_WARP:
        return _make_inverse_warp_pair(frame_rgb, depth, eye_distance_mm)
    return _make_soft_shift_pair(frame_rgb, depth, eye_distance_mm)


# --- projection -------------------------------------------------------------


def _coerce_flat_fov_deg(flat_fov_deg: float) -> float:
    try:
        value = float(flat_fov_deg)
    except (TypeError, ValueError):
        value = DEFAULT_FLAT_FOV_DEG
    if not math.isfinite(value):
        value = DEFAULT_FLAT_FOV_DEG
    return max(MIN_FLAT_FOV_DEG, min(MAX_FLAT_FOV_DEG, value))


def _ceil_even(value: float) -> int:
    out = int(math.ceil(float(value)))
    return out if (out & 1) == 0 else out + 1


# Per-eye VR side is capped so the SBS width (2*side) stays within NVENC's 8192
# limit -- otherwise hevc_nvenc rejects e.g. 1080p/fov80 fisheye (eye 4320 ->
# 8640 wide). Both the CPU and GPU pipelines honor this.
MAX_EYE_SIDE = 4096


def _flat_vr_eye_size(src_w: int, src_h: int, flat_fov_deg: float) -> int:
    fov = _coerce_flat_fov_deg(flat_fov_deg)
    side = _ceil_even(max(1, int(src_w), int(src_h)) * 180.0 / fov)
    return max(2, min(MAX_EYE_SIDE, side))


def _flat_camera_rays_to_source(dir_x, dir_y_down, dir_z, src_w, src_h, flat_fov_deg):
    fov_rad = math.radians(_coerce_flat_fov_deg(flat_fov_deg))
    plane_scale = max(1e-6, math.tan(fov_rad * 0.5))
    valid = dir_z > 1.0e-6
    px = np.zeros_like(dir_x, dtype=np.float32)
    py = np.zeros_like(dir_y_down, dtype=np.float32)
    np.divide(dir_x, dir_z * plane_scale, out=px, where=valid)
    np.divide(dir_y_down, dir_z * plane_scale, out=py, where=valid)
    valid &= (px >= -1.0) & (px <= 1.0) & (py >= -1.0) & (py <= 1.0)
    canvas = float(max(1, int(src_w), int(src_h)))
    x0 = (canvas - float(src_w)) * 0.5
    y0 = (canvas - float(src_h)) * 0.5
    map_x = (px * 0.5 + 0.5) * (canvas - 1.0) - x0
    map_y = (py * 0.5 + 0.5) * (canvas - 1.0) - y0
    valid &= (
        (map_x >= 0.0) & (map_x <= float(src_w - 1))
        & (map_y >= 0.0) & (map_y <= float(src_h - 1))
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def _make_hequirect_projection(src_w, src_h, flat_fov_deg, eye_size=None):
    side = int(eye_size) if eye_size else _flat_vr_eye_size(src_w, src_h, flat_fov_deg)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    yaw = ((xx + 0.5) / float(side) - 0.5) * math.pi
    pitch = (0.5 - (yy + 0.5) / float(side)) * math.pi
    cos_pitch = np.cos(pitch)
    dir_x = cos_pitch * np.sin(yaw)
    dir_y_down = -np.sin(pitch)
    dir_z = cos_pitch * np.cos(yaw)
    map_x, map_y, mask = _flat_camera_rays_to_source(dir_x, dir_y_down, dir_z, src_w, src_h, flat_fov_deg)
    return ProjectionMap(side, side, map_x, map_y, mask)


def _make_fisheye_projection(src_w, src_h, flat_fov_deg, eye_size=None):
    side = int(eye_size) if eye_size else _flat_vr_eye_size(src_w, src_h, flat_fov_deg)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    cx = cy = float(side) * 0.5
    radius = max(1.0, float(side) * 0.5)
    nx_disk = (xx + 0.5 - cx) / radius
    ny_disk = (yy + 0.5 - cy) / radius
    rr = np.sqrt(nx_disk * nx_disk + ny_disk * ny_disk)
    disk_mask = rr <= 1.0
    theta = rr * (math.pi * 0.5)
    azimuth = np.arctan2(-ny_disk, nx_disk)
    sin_theta = np.sin(theta)
    dir_x = sin_theta * np.cos(azimuth)
    dir_y_down = -sin_theta * np.sin(azimuth)
    dir_z = np.cos(theta)
    map_x, map_y, source_mask = _flat_camera_rays_to_source(dir_x, dir_y_down, dir_z, src_w, src_h, flat_fov_deg)
    return ProjectionMap(side, side, map_x, map_y, disk_mask & source_mask)


def _make_flat3d_projection(src_w, src_h):
    mask = np.ones((src_h, src_w), dtype=bool)
    # Identity map: render_sbs_frame short-circuits and never resamples it.
    return ProjectionMap(src_w, src_h, None, None, mask, is_identity=True)


def make_projection_map(src_w, src_h, projection, flat_fov_deg=DEFAULT_FLAT_FOV_DEG, eye_size=None):
    if projection == PROJECTION_FLAT_3D:
        return _make_flat3d_projection(src_w, src_h)
    if projection == PROJECTION_FISHEYE:
        return _make_fisheye_projection(src_w, src_h, flat_fov_deg, eye_size)
    return _make_hequirect_projection(src_w, src_h, flat_fov_deg, eye_size)


def _sample_rgb_xy(image: np.ndarray, pmap: ProjectionMap) -> np.ndarray:
    mapped = cv2.remap(
        image,
        np.ascontiguousarray(pmap.map_x, dtype=np.float32),
        np.ascontiguousarray(pmap.map_y, dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    mapped[~pmap.mask] = 0
    return mapped


def render_sbs_frame(frame_rgb, depth, eye_distance_mm, pmap,
                     hole_fill_mode=DEFAULT_HOLE_FILL_MODE):
    """Full per-frame render: depth -> stereo pair -> projection -> LR-SBS."""
    left, right = make_stereo_pair(frame_rgb, depth, eye_distance_mm, hole_fill_mode)
    if pmap.is_identity:
        # flat3d: eye images are already the output, no projection resample.
        return np.concatenate([left, right], axis=1)
    left_proj = _sample_rgb_xy(left, pmap)
    right_proj = _sample_rgb_xy(right, pmap)
    return np.concatenate([left_proj, right_proj], axis=1)


def output_dimensions(src_w, src_h, projection, flat_fov_deg=DEFAULT_FLAT_FOV_DEG):
    """Return (out_w, out_h) of the final SBS frame without building the map."""
    if projection == PROJECTION_FLAT_3D:
        return src_w * 2, src_h
    side = _flat_vr_eye_size(src_w, src_h, flat_fov_deg)
    return side * 2, side


class StereoRenderer:
    """Stateful per-resolution renderer (caches grids, reuses output buffers).

    Building the column grid / map_y and allocating the SBS frame every call cost
    ~20 ms/frame at 1080p. Holding them on the instance and remapping straight
    into the two halves of one preallocated buffer removes that per-frame churn.
    Use this on the hot path; the free functions stay for one-off calls/tests.
    """

    def __init__(self, src_w, src_h, projection, eye_distance_mm=DEFAULT_EYE_DISTANCE_MM,
                 hole_fill_mode=DEFAULT_HOLE_FILL_MODE, flat_fov_deg=DEFAULT_FLAT_FOV_DEG):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.projection = projection
        self.eye_distance_mm = float(eye_distance_mm)
        self.hole_fill_mode = hole_fill_mode
        self.pmap = make_projection_map(src_w, src_h, projection, flat_fov_deg)
        self.out_w = self.pmap.out_w * 2
        self.out_h = self.pmap.out_h
        self.max_shift = _max_disparity_pixels(self.src_w, self.eye_distance_mm)
        # Cached grids (built once per resolution).
        self._cols = np.broadcast_to(
            np.arange(self.src_w, dtype=np.float32)[None, :], (self.src_h, self.src_w)
        )
        self._map_y = np.ascontiguousarray(
            np.broadcast_to(np.arange(self.src_h, dtype=np.float32)[:, None], (self.src_h, self.src_w))
        )
        self._out = np.empty((self.out_h, self.out_w, 3), dtype=np.uint8)
        # Preallocated per-frame scratch for the inverse-warp fast path so the
        # hot loop does no allocations (disparity + the two eye sample maps).
        self._half = np.empty((self.src_h, self.src_w), dtype=np.float32)
        self._map_l = np.empty((self.src_h, self.src_w), dtype=np.float32)
        self._map_r = np.empty((self.src_h, self.src_w), dtype=np.float32)

    def render(self, frame_rgb, depth):
        """Render one SBS frame. ``depth`` may be at any resolution -- the fast
        inverse-warp path normalizes it cheaply and upscales the disparity."""
        if self.hole_fill_mode == HOLE_FILL_INVERSE_WARP and self.pmap.is_identity:
            # Fast path: normalize at depth res, upscale the (smooth) half-
            # disparity into preallocated scratch, build both eye maps in place,
            # then inverse-sample straight into the SBS halves.
            near = _normalize_near(_smooth_depth(depth))
            half_low = near * (self.max_shift * 0.5)
            if half_low.shape != (self.src_h, self.src_w):
                cv2.resize(half_low, (self.src_w, self.src_h), dst=self._half,
                           interpolation=cv2.INTER_LINEAR)
            else:
                self._half[:] = half_low
            w = self.src_w
            np.subtract(self._cols, self._half, out=self._map_l)
            np.add(self._cols, self._half, out=self._map_r)
            cv2.remap(frame_rgb, self._map_l, self._map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE, dst=self._out[:, :w])
            cv2.remap(frame_rgb, self._map_r, self._map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE, dst=self._out[:, w:])
            return self._out
        # General path (soft_shift, or projected modes) needs frame-res depth.
        if depth.shape != (self.src_h, self.src_w):
            depth = cv2.resize(depth, (self.src_w, self.src_h), interpolation=cv2.INTER_LINEAR)
        left, right = make_stereo_pair(frame_rgb, depth, self.eye_distance_mm, self.hole_fill_mode)
        if self.pmap.is_identity:
            np.concatenate([left, right], axis=1, out=self._out)
            return self._out
        _sample_rgb_xy_into(left, self.pmap, self._out[:, :self.pmap.out_w])
        _sample_rgb_xy_into(right, self.pmap, self._out[:, self.pmap.out_w:])
        return self._out


def _sample_rgb_xy_into(image, pmap, dst):
    cv2.remap(image, np.ascontiguousarray(pmap.map_x, dtype=np.float32),
              np.ascontiguousarray(pmap.map_y, dtype=np.float32),
              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE, dst=dst)
    dst[~pmap.mask] = 0
