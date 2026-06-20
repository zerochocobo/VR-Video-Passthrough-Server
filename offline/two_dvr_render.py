"""Depth -> stereo -> VR projection rendering (pure numpy).

Ported from the standalone tool_2dvr ``logic.py`` so the offline 2D->VR
pipeline has no torch dependency: depth comes from the DA3 ONNX engine
(:mod:`offline.da3_depth`) and everything here is numpy. Covers the two
production hole-fill modes (``soft_shift`` forward warp, ``inverse_warp`` fast
inverse sampling) and the three projections (flat3d / hequirect-180 / fisheye-180).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

# Percentile normalization only needs a representative sample of pixels; sampling
# instead of scanning the whole frame keeps _normalize_near cheap at 4K.
_NORM_SAMPLE = 16384

# VVPS stabilizer (in-house algorithm) -- base/detail split for the depth
# stabilizer (mode=ema is the VVPS path; mode=flow is the experimental
# motion-compensated variant). The low-frequency
# "base" carries the overall per-object depth level that causes the disparity
# "swimming"; the high-frequency "detail" carries edges/structure. We only
# smooth the base across time and ALWAYS re-inject the current frame's detail,
# so the disparity surface stays spatially coherent and soft_shift/inverse_warp
# hole-fill never tears phantom holes inside a foreground body. The base low-pass
# kernel is min(H, W) / _BASE_LOWPASS_DIV (odd), i.e. object-scale, not pixel-scale.
_BASE_LOWPASS_DIV = 16

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

DEFAULT_TEMPORAL_NORM = os.environ.get("PT_TWO_DVR_TEMPORAL_NORM", "1") != "0"
DEFAULT_TEMPORAL_NORM_ALPHA = max(
    0.0,
    min(1.0, float(os.environ.get("PT_TWO_DVR_TEMPORAL_NORM_ALPHA", 0.10))),
)
DEFAULT_TEMPORAL_NORM_RESET = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_NORM_RESET", 1.0)),
)
TEMPORAL_DEPTH_OFF = "off"
TEMPORAL_DEPTH_EMA = "ema"
TEMPORAL_DEPTH_FLOW = "flow"
TEMPORAL_DEPTH_MODES = {TEMPORAL_DEPTH_OFF, TEMPORAL_DEPTH_EMA, TEMPORAL_DEPTH_FLOW}
# On by default: the stabilizer is the base/detail rewrite (see _BASE_LOWPASS_DIV
# and _stabilize_base_detail). The earlier per-pixel deadband/EMA version that
# locked foreground motion and tore soft_shift holes has been replaced; the
# per-pixel px limiters below stay 0 by default. Set PT_TWO_DVR_DEPTH_STABILIZER=0
# (or mode=off) to hard-bypass.
_DEFAULT_TEMPORAL_DEPTH_FLAG = os.environ.get(
    "PT_TWO_DVR_DEPTH_STABILIZER",
    os.environ.get("PT_TWO_DVR_TEMPORAL_DEPTH", "1"),
) != "0"
_DEFAULT_TEMPORAL_DEPTH_MODE_RAW = os.environ.get(
    "PT_TWO_DVR_TEMPORAL_DEPTH_MODE",
    TEMPORAL_DEPTH_EMA if _DEFAULT_TEMPORAL_DEPTH_FLAG else TEMPORAL_DEPTH_OFF,
).strip().lower()
if _DEFAULT_TEMPORAL_DEPTH_MODE_RAW not in TEMPORAL_DEPTH_MODES:
    _DEFAULT_TEMPORAL_DEPTH_MODE_RAW = TEMPORAL_DEPTH_EMA if _DEFAULT_TEMPORAL_DEPTH_FLAG else TEMPORAL_DEPTH_OFF
DEFAULT_TEMPORAL_DEPTH_MODE = _DEFAULT_TEMPORAL_DEPTH_MODE_RAW
DEFAULT_TEMPORAL_DEPTH = _DEFAULT_TEMPORAL_DEPTH_FLAG and DEFAULT_TEMPORAL_DEPTH_MODE != TEMPORAL_DEPTH_OFF
DEFAULT_TEMPORAL_DEPTH_ALPHA = max(
    0.0,
    min(1.0, float(os.environ.get("PT_TWO_DVR_TEMPORAL_DEPTH_ALPHA", 0.20))),
)
DEFAULT_TEMPORAL_FLOW_DIFF = max(0.0, float(os.environ.get("PT_TWO_DVR_TEMPORAL_FLOW_DIFF", 35.0)))
DEFAULT_TEMPORAL_FLOW_CONSISTENCY = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_FLOW_CONSISTENCY", 0.0)),
)
DEFAULT_TEMPORAL_FLOW_MOTION_GATE = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_FLOW_MOTION_GATE", 0.0)),
)
DEFAULT_TEMPORAL_AFFINE = os.environ.get("PT_TWO_DVR_TEMPORAL_AFFINE", "1") != "0"
DEFAULT_TEMPORAL_AFFINE_MAX_SCALE = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_AFFINE_MAX_SCALE", 0.20)),
)
DEFAULT_TEMPORAL_AFFINE_MAX_BIAS = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_AFFINE_MAX_BIAS", 0.12)),
)
# VVPS Phase-2 (summary 8.5/8.6): global motion compensation. Before the base
# EMA, warp the previous base into the current frame by an estimated global
# translation (phaseCorrelate on a downsampled gray). This implements
# warp-then-filter so a camera pan no longer smears/locks the base (a pure pan is
# fully explained by the translation -> base barely changes -> correctly locked;
# a dolly leaves residual that the EMA still smooths while real parallax rides in
# the always-current detail).
# Defaulted ON for validation. IMPORTANT: this currently only affects the CPU
# StereoRenderer path -- the GPU base/detail runs in kernels and does not yet do
# motion compensation, so on the GPU/realtime path this flag is a silent no-op
# until the GPU port lands. Validate via offline `--gpu-render off`. Set
# PT_TWO_DVR_TEMPORAL_MOTION_COMP=0 to disable.
DEFAULT_TEMPORAL_MOTION_COMP = os.environ.get("PT_TWO_DVR_TEMPORAL_MOTION_COMP", "1") != "0"
# Phase correlation runs on a downsampled gray: at least /_MC_DOWNSAMPLE, and
# enough to keep the longest side <= _MC_MAX_WORK (bounds FFT cost at 4K while
# staying accurate -- aggressive downsampling loses the shift on fine structure).
_MC_DOWNSAMPLE = 2
_MC_MAX_WORK = 512
_MC_MAX_SHIFT_PX = 96.0
# Phase-correlation peak response gate: degenerate/flat frames (e.g. a black
# frame) return a spurious half-size shift with response ~0, so reject anything
# below this. Real textured frames score well above it.
_MC_MIN_RESPONSE = 0.05

# VVPS 8.6.3 per-tile evidence gate. After global motion compensation, the
# residual |cur_gray - warped_prev_gray| is the local "should this change?"
# evidence: ~0 means the motion is fully explained (static/pan) -> lock the base
# harder; large means unexplained local motion (a moving object, parallax, or a
# disocclusion) -> let the base follow the current frame. The per-pixel base-EMA
# alpha is driven from a tile-smoothed residual, ramping from a_lock (= depth
# alpha * lock_scale) at <=_EVID_R_LO gray levels to 1.0 at >=_EVID_R_HI.
# Scene-cut reset: on a hard cut, blending the new shot's depth with the old
# one's (band EMA / base EMA / motion comp) produces a few frames of wrong depth.
# An HSV-histogram cut detector resets the temporal state at the boundary -- more
# reliable than the depth-band-jump heuristic alone (which misses cuts with a
# similar depth range and false-fires on in-shot range changes).
DEFAULT_SCENE_CUT = os.environ.get("PT_TWO_DVR_SCENE_CUT", "1") != "0"
DEFAULT_SCENE_CUT_THRESHOLD = max(0.0, float(os.environ.get("PT_TWO_DVR_SCENE_CUT_THRESHOLD", 0.4)))
DEFAULT_TEMPORAL_EVIDENCE_GATE = os.environ.get("PT_TWO_DVR_TEMPORAL_EVIDENCE_GATE", "1") != "0"
DEFAULT_TEMPORAL_EVIDENCE_LOCK = max(
    0.0, min(1.0, float(os.environ.get("PT_TWO_DVR_TEMPORAL_EVIDENCE_LOCK", 0.5)))
)
_EVID_TILE = 16       # residual is smoothed at ~1/_EVID_TILE of the frame (tile scale)
_EVID_R_LO = 6.0      # gray-level residual at/below which a tile is fully locked
_EVID_R_HI = 36.0     # gray-level residual at/above which a tile fully follows current
DEFAULT_TEMPORAL_STATIC_DEADBAND_PX = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_STATIC_DEADBAND_PX", 0.0)),
)
DEFAULT_TEMPORAL_STATIC_MAX_STEP_PX = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_STATIC_MAX_STEP_PX", 0.0)),
)
DEFAULT_TEMPORAL_MOTION_MAX_STEP_PX = max(
    0.0,
    float(os.environ.get("PT_TWO_DVR_TEMPORAL_MOTION_MAX_STEP_PX", 0.0)),
)

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


def _unit_float(value: float | str | None, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return max(0.0, min(1.0, out))


def _nonnegative_float(value: float | str | None, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return max(0.0, out)


def _temporal_depth_mode(value: str | None, enabled: bool) -> str:
    if not enabled:
        return TEMPORAL_DEPTH_OFF
    text = str(value or "").strip().lower()
    if text in TEMPORAL_DEPTH_MODES:
        return text
    return TEMPORAL_DEPTH_EMA


class TemporalDepthStabilizer:
    """Per-clip state for stabilizing DA3 depth before stereo disparity.

    Two layers: (1) the percentile normalization band is smoothed across frames;
    (2) the near/disparity map itself is stabilized by the VVPS stabilizer
    (in-house base/detail algorithm, mode=ema) -- it smooths only the
    low-frequency per-object depth "level" that causes disparity swimming and
    always re-injects the current frame's high-frequency detail, so foreground
    motion and hole-fill stay intact. mode=flow is an experimental
    motion-compensated variant; mode=off bypasses layer (2).
    """

    def __init__(
        self,
        *,
        norm_enabled: bool = DEFAULT_TEMPORAL_NORM,
        norm_alpha: float = DEFAULT_TEMPORAL_NORM_ALPHA,
        norm_reset_threshold: float = DEFAULT_TEMPORAL_NORM_RESET,
        depth_enabled: bool = DEFAULT_TEMPORAL_DEPTH,
        depth_mode: str = DEFAULT_TEMPORAL_DEPTH_MODE,
        depth_alpha: float = DEFAULT_TEMPORAL_DEPTH_ALPHA,
        flow_diff_threshold: float = DEFAULT_TEMPORAL_FLOW_DIFF,
        flow_consistency_threshold: float = DEFAULT_TEMPORAL_FLOW_CONSISTENCY,
        flow_motion_gate: float = DEFAULT_TEMPORAL_FLOW_MOTION_GATE,
        affine_enabled: bool = DEFAULT_TEMPORAL_AFFINE,
        affine_max_scale_delta: float = DEFAULT_TEMPORAL_AFFINE_MAX_SCALE,
        affine_max_bias: float = DEFAULT_TEMPORAL_AFFINE_MAX_BIAS,
        max_disparity_px: float = 0.0,
        static_deadband_px: float = 0.0,
        static_max_step_px: float = 0.0,
        motion_max_step_px: float = 0.0,
        motion_comp_enabled: bool = DEFAULT_TEMPORAL_MOTION_COMP,
        evidence_gate_enabled: bool = DEFAULT_TEMPORAL_EVIDENCE_GATE,
        evidence_lock_scale: float = DEFAULT_TEMPORAL_EVIDENCE_LOCK,
        scene_cut_enabled: bool = DEFAULT_SCENE_CUT,
        scene_cut_threshold: float = DEFAULT_SCENE_CUT_THRESHOLD,
    ) -> None:
        self.norm_enabled = bool(norm_enabled)
        self.norm_alpha = _unit_float(norm_alpha, DEFAULT_TEMPORAL_NORM_ALPHA)
        self.norm_reset_threshold = _nonnegative_float(norm_reset_threshold, DEFAULT_TEMPORAL_NORM_RESET)
        self.depth_mode = _temporal_depth_mode(depth_mode, bool(depth_enabled))
        self.depth_enabled = self.depth_mode != TEMPORAL_DEPTH_OFF
        self.depth_alpha = _unit_float(depth_alpha, DEFAULT_TEMPORAL_DEPTH_ALPHA)
        self.flow_diff_threshold = _nonnegative_float(flow_diff_threshold, DEFAULT_TEMPORAL_FLOW_DIFF)
        self.flow_consistency_threshold = _nonnegative_float(
            flow_consistency_threshold,
            DEFAULT_TEMPORAL_FLOW_CONSISTENCY,
        )
        self.flow_motion_gate = _nonnegative_float(flow_motion_gate, DEFAULT_TEMPORAL_FLOW_MOTION_GATE)
        self.affine_enabled = bool(affine_enabled)
        self.affine_max_scale_delta = _nonnegative_float(affine_max_scale_delta, DEFAULT_TEMPORAL_AFFINE_MAX_SCALE)
        self.affine_max_bias = _nonnegative_float(affine_max_bias, DEFAULT_TEMPORAL_AFFINE_MAX_BIAS)
        self.max_disparity_px = _nonnegative_float(max_disparity_px, 0.0)
        self.static_deadband_px = _nonnegative_float(static_deadband_px, 0.0)
        self.static_max_step_px = _nonnegative_float(static_max_step_px, 0.0)
        self.motion_max_step_px = _nonnegative_float(motion_max_step_px, 0.0)
        self.motion_comp_enabled = bool(motion_comp_enabled)
        self.evidence_gate_enabled = bool(evidence_gate_enabled)
        self.evidence_lock_scale = _unit_float(evidence_lock_scale, DEFAULT_TEMPORAL_EVIDENCE_LOCK)
        self.scene_cut_enabled = bool(scene_cut_enabled)
        self._scene_detector = None
        if self.scene_cut_enabled:
            from utils.scene_detection import SceneCutDetector
            self._scene_detector = SceneCutDetector(threshold=float(scene_cut_threshold))
        self._lo_ema: float | None = None
        self._hi_ema: float | None = None
        self._near_prev: np.ndarray | None = None
        self._gray_prev: np.ndarray | None = None
        # base/detail (mode=ema) state: only the smoothed low-frequency base.
        self._base_prev: np.ndarray | None = None
        # motion-compensation state.
        self._mc_gray_prev: np.ndarray | None = None
        self._mc_window: np.ndarray | None = None

    def reset(self) -> None:
        self._lo_ema = None
        self._hi_ema = None
        self._near_prev = None
        self._gray_prev = None
        self._base_prev = None
        self._mc_gray_prev = None
        if self._scene_detector is not None:
            self._scene_detector.reset()

    def begin_frame(self, frame_rgb: np.ndarray | None) -> bool:
        """Step the scene-cut detector; reset temporal state on a hard cut.

        Called once per frame BEFORE normalization so the new shot re-seeds the
        depth band and base state instead of blending across the cut. Returns
        True on a detected cut. The detector keeps its own EMA reference, so we
        re-seed it after reset() clears it.
        """
        if self._scene_detector is None or frame_rgb is None:
            return False
        arr = np.asarray(frame_rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return False
        if self._scene_detector.step(arr[:, :, :3]):
            self.reset()
            self._scene_detector.step(arr[:, :, :3])  # re-seed reference for the new shot
            return True
        return False

    @property
    def norm_band(self) -> tuple[float, float] | None:
        if self._lo_ema is None or self._hi_ema is None:
            return None
        return self._lo_ema, self._hi_ema

    def normalization_band(self, lo_raw: float, hi_raw: float) -> tuple[float, float]:
        if not self.norm_enabled:
            return lo_raw, hi_raw
        if self._lo_ema is None or self._hi_ema is None:
            self._lo_ema, self._hi_ema = lo_raw, hi_raw
            return lo_raw, hi_raw

        prev_span = max(self._hi_ema - self._lo_ema, 1e-6)
        jump = max(abs(lo_raw - self._lo_ema), abs(hi_raw - self._hi_ema)) / prev_span
        if self.norm_reset_threshold > 0.0 and jump >= self.norm_reset_threshold:
            self.reset()
            self._lo_ema, self._hi_ema = lo_raw, hi_raw
            return lo_raw, hi_raw

        a = self.norm_alpha
        self._lo_ema = (1.0 - a) * self._lo_ema + a * lo_raw
        self._hi_ema = (1.0 - a) * self._hi_ema + a * hi_raw
        return self._lo_ema, self._hi_ema

    def _gray_for_flow(self, frame_rgb: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
        if frame_rgb is None:
            return None
        arr = np.asarray(frame_rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            return None
        if gray.shape != shape:
            interpolation = cv2.INTER_AREA if gray.shape[0] >= shape[0] and gray.shape[1] >= shape[1] else cv2.INTER_LINEAR
            gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=interpolation)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(gray)

    def _flow_align_previous(self, cur_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        prev_near = self._near_prev
        prev_gray = self._gray_prev
        if prev_near is None or prev_gray is None or prev_near.shape != cur_gray.shape or prev_gray.shape != cur_gray.shape:
            return None
        h, w = cur_gray.shape
        bwd = cv2.calcOpticalFlowFarneback(
            cur_gray,
            prev_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xx + bwd[:, :, 0]
        map_y = yy + bwd[:, :, 1]
        valid = (map_x >= 0.0) & (map_x <= float(w - 1)) & (map_y >= 0.0) & (map_y <= float(h - 1))
        aligned = cv2.remap(
            prev_near,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if self.flow_diff_threshold > 0.0:
            prev_gray_aligned = cv2.remap(
                prev_gray,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            diff = np.abs(prev_gray_aligned.astype(np.float32) - cur_gray.astype(np.float32))
            valid &= diff <= self.flow_diff_threshold
        if self.flow_consistency_threshold > 0.0:
            fwd = cv2.calcOpticalFlowFarneback(
                prev_gray,
                cur_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            fwd_x = cv2.remap(fwd[:, :, 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fwd_y = cv2.remap(fwd[:, :, 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            err = np.sqrt((bwd[:, :, 0] + fwd_x) ** 2 + (bwd[:, :, 1] + fwd_y) ** 2)
            valid &= err <= self.flow_consistency_threshold
        motion = np.sqrt(bwd[:, :, 0] ** 2 + bwd[:, :, 1] ** 2)
        return aligned.astype(np.float32, copy=False), valid, motion.astype(np.float32, copy=False)

    def _same_pixel_valid(self, cur_gray: np.ndarray | None) -> np.ndarray | None:
        prev_gray = self._gray_prev
        if cur_gray is None or prev_gray is None or prev_gray.shape != cur_gray.shape:
            return None
        if self.flow_diff_threshold <= 0.0:
            return np.ones(cur_gray.shape, dtype=bool)
        diff = np.abs(prev_gray.astype(np.float32) - cur_gray.astype(np.float32))
        return diff <= self.flow_diff_threshold

    def _affine_match_to_previous(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        valid: np.ndarray | None,
    ) -> np.ndarray:
        if not self.affine_enabled or previous.shape != current.shape:
            return current
        if valid is None:
            mask = np.ones(current.shape, dtype=bool)
        else:
            mask = np.asarray(valid, dtype=bool)
        # Ignore clipped extremes; they carry little scale information and make
        # histogram matching react to disocclusions or black borders.
        mask &= (current > 0.02) & (current < 0.98) & (previous > 0.02) & (previous < 0.98)
        count = int(np.count_nonzero(mask))
        min_count = max(128, int(current.size * 0.05))
        if count < min_count:
            return current
        cur = current[mask]
        prev = previous[mask]
        if cur.size > _NORM_SAMPLE:
            step = max(1, cur.size // _NORM_SAMPLE)
            cur = cur[::step]
            prev = prev[::step]
        c20, c50, c80 = (float(v) for v in np.percentile(cur, [20.0, 50.0, 80.0]))
        p20, p50, p80 = (float(v) for v in np.percentile(prev, [20.0, 50.0, 80.0]))
        cur_span = c80 - c20
        prev_span = p80 - p20
        if cur_span < 0.03 or prev_span < 0.03:
            return current
        scale = prev_span / cur_span
        scale = max(1.0 - self.affine_max_scale_delta, min(1.0 + self.affine_max_scale_delta, scale))
        bias = p50 - scale * c50
        bias = max(-self.affine_max_bias, min(self.affine_max_bias, bias))
        if abs(scale - 1.0) < 1e-4 and abs(bias) < 1e-4:
            return current
        out = current * np.float32(scale) + np.float32(bias)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _px_to_near(self, px: float) -> float:
        if self.max_disparity_px <= 1.0e-6 or px <= 0.0:
            return 0.0
        return float(px) / self.max_disparity_px

    def _limit_delta_by_evidence(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        stable_mask: np.ndarray | None,
    ) -> np.ndarray:
        deadband = self._px_to_near(self.static_deadband_px)
        static_step = self._px_to_near(self.static_max_step_px)
        motion_step = self._px_to_near(self.motion_max_step_px)
        if deadband <= 0.0 and static_step <= 0.0 and motion_step <= 0.0:
            return current
        delta = current - previous
        out = current.copy()
        if stable_mask is None:
            stable = np.ones(current.shape, dtype=bool)
        else:
            stable = np.asarray(stable_mask, dtype=bool)
        if deadband > 0.0:
            out[stable & (np.abs(delta) < np.float32(deadband))] = previous[
                stable & (np.abs(delta) < np.float32(deadband))
            ]
            delta = out - previous
        if static_step > 0.0:
            m = stable & (np.abs(delta) > np.float32(static_step))
            out[m] = previous[m] + np.clip(delta[m], -static_step, static_step)
            delta = out - previous
        if stable_mask is not None and motion_step > 0.0:
            moving = ~stable
            m = moving & (np.abs(delta) > np.float32(motion_step))
            out[m] = previous[m] + np.clip(delta[m], -motion_step, motion_step)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _base_kernel(self, h: int, w: int) -> int:
        k = int(round(min(int(h), int(w)) / float(_BASE_LOWPASS_DIV)))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        return k

    def _lowpass(self, near: np.ndarray) -> np.ndarray:
        h, w = near.shape[:2]
        k = self._base_kernel(h, w)
        if k <= 1:
            return near.astype(np.float32, copy=True)
        # Box blur (== uniform_filter) is an O(n) separable low-pass; the GPU path
        # uses the same box so CPU/GPU base/detail match.
        return cv2.blur(near, (k, k), borderType=cv2.BORDER_REPLICATE)

    def _estimate_global_translation(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[float, float]:
        """Global (dx, dy) that registers prev onto cur, via phase correlation.

        Estimated on a downsampled, Hann-windowed gray pair, then scaled back to
        full resolution and clamped. Returns (0, 0) on degenerate/low-response
        input. Note: strongly periodic textures (fences, brick) can alias the
        phase peak to shift-modulo-period; natural broadband footage is fine.
        """
        h, w = cur_gray.shape
        ds = max(_MC_DOWNSAMPLE, -(-max(h, w) // _MC_MAX_WORK))
        sw = max(8, w // ds)
        sh = max(8, h // ds)
        a = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(cur_gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._mc_window is None or self._mc_window.shape != (sh, sw):
            self._mc_window = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
        (sx, sy), response = cv2.phaseCorrelate(a, b, self._mc_window)
        if not np.isfinite(response) or not np.isfinite(sx) or not np.isfinite(sy):
            return 0.0, 0.0
        if response < _MC_MIN_RESPONSE:  # flat/degenerate frame -> no usable shift
            return 0.0, 0.0
        dx = float(sx) * (float(w) / float(sw))
        dy = float(sy) * (float(h) / float(sh))
        dx = max(-_MC_MAX_SHIFT_PX, min(_MC_MAX_SHIFT_PX, dx))
        dy = max(-_MC_MAX_SHIFT_PX, min(_MC_MAX_SHIFT_PX, dy))
        return dx, dy

    def _warp_translate(self, img: np.ndarray, dx: float, dy: float, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Translate img by (dx, dy): out(x,y)=img(x-dx,y-dy), replicate border."""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return img
        h, w = img.shape[:2]
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(img, m, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)

    def _evidence_alpha(self, cur_gray: np.ndarray, prev_gray_aligned: np.ndarray) -> np.ndarray:
        """Per-pixel base-EMA alpha from the motion-compensated residual (8.6.3).

        Tile-smoothed |cur - aligned_prev| ramps alpha from a_lock (locked) at
        <=_EVID_R_LO to 1.0 (follow current) at >=_EVID_R_HI.
        """
        h, w = cur_gray.shape
        resid = np.abs(cur_gray.astype(np.float32) - prev_gray_aligned.astype(np.float32))
        tw = max(1, w // _EVID_TILE)
        th = max(1, h // _EVID_TILE)
        small = cv2.resize(resid, (tw, th), interpolation=cv2.INTER_AREA)
        coarse = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        t = np.clip((coarse - _EVID_R_LO) / max(1e-3, _EVID_R_HI - _EVID_R_LO), 0.0, 1.0)
        a_lock = np.float32(self.depth_alpha * self.evidence_lock_scale)
        return (a_lock + (np.float32(1.0) - a_lock) * t).astype(np.float32)

    def _motion_compensate_base(self, base_prev: np.ndarray, cur_gray: np.ndarray | None) -> np.ndarray:
        """Estimate the global shift and warp base_prev into the current frame."""
        prev_gray = self._mc_gray_prev
        if (
            cur_gray is None
            or prev_gray is None
            or prev_gray.shape != cur_gray.shape
            or prev_gray.shape != base_prev.shape
        ):
            return base_prev
        dx, dy = self._estimate_global_translation(prev_gray, cur_gray)
        return self._warp_translate(base_prev, dx, dy)

    def _stabilize_base_detail(self, current: np.ndarray, frame_rgb: np.ndarray | None = None) -> np.ndarray:
        """VVPS stabilizer: stabilize only the low-frequency base; re-inject detail.

        This is the default (mode=ema) in-house stabilizer used for the realtime
        path (VVPS). The offline path will additionally offer NVDS as an option
        in a later stage. It kills the per-object
        disparity "swimming" (a low-frequency, level effect) without locking
        foreground motion and without disturbing the high-frequency disparity
        structure that hole-fill keys off -- the two failure modes of the earlier
        per-pixel deadband/EMA version.
        """
        base_cur = self._lowpass(current)
        detail = current - base_cur
        prev = self._base_prev
        need_gray = self.motion_comp_enabled or self.evidence_gate_enabled
        cur_gray = self._gray_for_flow(frame_rgb, current.shape) if need_gray else None
        if prev is None or prev.shape != base_cur.shape:
            self._base_prev = base_cur.copy()
            self._mc_gray_prev = cur_gray.copy() if cur_gray is not None else None
            return current  # first frame is identity (base + detail == near)
        # warp-then-filter (8.6): estimate the global shift once, use it to align
        # the previous base into the current frame (motion comp) and to build the
        # per-tile evidence gate from the motion-compensated gray residual.
        alpha_map: np.ndarray | None = None
        prev_gray = self._mc_gray_prev
        if need_gray and cur_gray is not None and prev_gray is not None and prev_gray.shape == cur_gray.shape:
            dx, dy = self._estimate_global_translation(prev_gray, cur_gray)
            if self.motion_comp_enabled:
                prev = self._warp_translate(prev, dx, dy)
            if self.evidence_gate_enabled:
                aligned_prev_gray = self._warp_translate(prev_gray, dx, dy)
                alpha_map = self._evidence_alpha(cur_gray, aligned_prev_gray)
        base_aligned = self._affine_match_to_previous(base_cur, prev, None)
        base_aligned = self._limit_delta_by_evidence(base_aligned, prev, None)
        if alpha_map is None:
            a = np.float32(self.depth_alpha)
            base_stable = prev * (np.float32(1.0) - a) + base_aligned * a
        else:
            base_stable = prev * (np.float32(1.0) - alpha_map) + base_aligned * alpha_map
        np.clip(base_stable, 0.0, 1.0, out=base_stable)
        self._base_prev = base_stable
        self._mc_gray_prev = cur_gray.copy() if cur_gray is not None else None
        out = base_stable + detail
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(np.float32, copy=False)

    def _stabilize_flow(self, current: np.ndarray, frame_rgb: np.ndarray | None) -> np.ndarray:
        """Experimental (mode=flow): motion-compensated full-near EMA.

        Opt-in only. Unlike base/detail this blends the full-resolution near map,
        so it can trail/lock on fast subjects -- kept for experimentation behind
        --temporal-depth-mode flow.
        """
        cur_gray = self._gray_for_flow(frame_rgb, current.shape)
        if self._near_prev is None or self._near_prev.shape != current.shape:
            self._near_prev = current.copy()
            self._gray_prev = cur_gray.copy() if cur_gray is not None else None
            return current

        prev = self._near_prev
        valid: np.ndarray | None = None
        motion: np.ndarray | None = None
        if cur_gray is not None:
            aligned = self._flow_align_previous(cur_gray)
            if aligned is not None:
                prev, valid, motion = aligned

        a = self.depth_alpha
        current = self._affine_match_to_previous(current, prev, valid)
        current = self._limit_delta_by_evidence(current, prev, valid)
        if valid is None:
            out = prev * np.float32(1.0 - a) + current * np.float32(a)
        else:
            alpha = np.full(current.shape, np.float32(a), dtype=np.float32)
            alpha[~valid] = 1.0
            if self.flow_motion_gate > 0.0 and motion is not None:
                alpha = np.maximum(alpha, np.minimum(1.0, motion / np.float32(self.flow_motion_gate)))
            out = prev * (1.0 - alpha) + current * alpha
        np.clip(out, 0.0, 1.0, out=out)
        self._near_prev = out.copy()
        self._gray_prev = cur_gray.copy() if cur_gray is not None else None
        return out

    def stabilize_near(self, near: np.ndarray, frame_rgb: np.ndarray | None = None) -> np.ndarray:
        if not self.depth_enabled:
            return near
        current = np.asarray(near, dtype=np.float32)
        if self.depth_mode == TEMPORAL_DEPTH_FLOW:
            return self._stabilize_flow(current, frame_rgb)
        return self._stabilize_base_detail(current, frame_rgb)


# VVPS 8.5.3 B -- offline symmetric (non-causal) temporal window. Offline only
# (needs lookahead). Gaussian-weighted symmetric mean of the base over 2r+1
# frames -> lag-free smoothing that matches the realtime causal EMA. Default
# radius 6 (13-frame window); a small radius under-smooths static-scene jitter.
# Replaces the causal base/detail stabilizer for offline conversion. 0 disables.
_WIN_MAX_RADIUS = 10
DEFAULT_TEMPORAL_WINDOW = max(0, min(_WIN_MAX_RADIUS, int(os.environ.get("PT_TWO_DVR_TEMPORAL_WINDOW", "6"))))


def _base_lowpass(near: np.ndarray, div: float = _BASE_LOWPASS_DIV) -> np.ndarray:
    h, w = near.shape[:2]
    k = int(round(min(int(h), int(w)) / float(div)))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return cv2.blur(near, (k, k), borderType=cv2.BORDER_REPLICATE)


class SymmetricBaseWindow:
    """Offline-only symmetric temporal smoothing of the VVPS base.

    Offline conversion can look ahead, so instead of the causal base EMA we take
    a temporal **median** of the base over a window centred on the current frame.
    Because the window is symmetric, static depth stops flickering with *no lag
    and no ghosting* (the causal EMA always trades one for the other).

    Each ``push`` buffers (base, detail, gray, payload) for one frame and returns
    the now-ready centre frame(s) -- output is delayed by ``radius`` frames;
    ``flush`` drains the tail at clip end. Neighbours are globally
    motion-compensated to the centre before the median (so a pan does not smear),
    and the median is only used where the motion-compensated residual says the
    region is static -- moving objects / disocclusions keep the centre base
    (global alignment can't follow them, so the median would corrupt them).
    """

    def __init__(self, radius: int = 2, lowpass_div: float = _BASE_LOWPASS_DIV) -> None:
        self.radius = max(1, min(_WIN_MAX_RADIUS, int(radius)))
        self.lowpass_div = float(lowpass_div)
        self._buf: list[dict] = []
        self._base_idx = 0   # absolute index of self._buf[0]
        self._next = 0       # absolute index of the next frame to emit
        self._window = None  # cv2 Hanning window cache

    def reset(self) -> None:
        self._buf.clear()
        self._base_idx = 0
        self._next = 0

    @property
    def delay(self) -> int:
        return self.radius

    def _gray_small(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        ds = max(_MC_DOWNSAMPLE, -(-max(h, w) // _MC_MAX_WORK))
        sw = max(8, w // ds)
        sh = max(8, h // ds)
        # np.array(copy=True): cv2.resize can hand back a recycled internal
        # buffer, so own the data before it lands in the multi-frame ring buffer.
        return np.array(cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA), dtype=np.float32)

    def _shift_small(self, prev_small: np.ndarray, cur_small: np.ndarray) -> tuple[float, float]:
        sh, sw = cur_small.shape
        if self._window is None or self._window.shape != (sh, sw):
            self._window = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
        # phaseCorrelate multiplies its inputs by the window IN PLACE, so pass
        # copies -- prev_small/cur_small are reused from the frame ring buffer.
        (dx, dy), response = cv2.phaseCorrelate(prev_small.copy(), cur_small.copy(), self._window)
        if not np.isfinite(response) or response < _MC_MIN_RESPONSE:
            return 0.0, 0.0
        if not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0, 0.0
        return float(dx), float(dy)

    def push(self, near: np.ndarray, align_gray: np.ndarray, payload, band=None) -> list[tuple[np.ndarray, object]]:
        """Buffer one frame; return [(near_out, payload)] for any centre now ready.

        A frame becomes emittable once its right neighbour ``j + radius`` has
        arrived, so output trails input by ``radius`` frames. ``band`` is the
        ``(lo_used, hi_used, lo_raw, hi_raw)`` normalization band for this frame;
        when present for all windowed frames, the centre is re-normalized to the
        symmetric (lookahead) band -- a zero-phase smoothing of the raw depth
        range that removes the causal band EMA's lag.
        """
        near = np.asarray(near, dtype=np.float32)
        base = _base_lowpass(near, self.lowpass_div)
        self._buf.append({
            "base": base,
            "detail": near - base,
            "small": self._gray_small(np.asarray(align_gray)),
            "payload": payload,
            "band": band,
        })
        out: list[tuple[np.ndarray, object]] = []
        total = self._base_idx + len(self._buf)
        while self._next + self.radius < total:
            out.append(self._emit_abs(self._next))
            self._next += 1
            self._drop()
        return out

    def flush(self) -> list[tuple[np.ndarray, object]]:
        """Drain the tail at clip end (centres use the clamped right window)."""
        out: list[tuple[np.ndarray, object]] = []
        total = self._base_idx + len(self._buf)
        while self._next < total:
            out.append(self._emit_abs(self._next))
            self._next += 1
            self._drop()
        return out

    def _drop(self) -> None:
        # Keep frames still needed as left context for future centres (>= next-R).
        while self._buf and self._base_idx < self._next - self.radius:
            self._buf.pop(0)
            self._base_idx += 1

    def _emit_abs(self, j: int) -> tuple[np.ndarray, object]:
        c = j - self._base_idx
        return self._emit_centre(c)

    def _emit_centre(self, c: int) -> tuple[np.ndarray, object]:
        centre = self._buf[c]
        base_c = centre["base"]
        small_c = centre["small"]
        h, w = base_c.shape
        sh, sw = small_c.shape
        sx = float(w) / float(sw)
        sy = float(h) / float(sh)
        # Gaussian-weighted symmetric mean of the (motion-compensated) base. A
        # plain median over a small window under-smooths the Gaussian-ish per-frame
        # DA3 level jitter on static scenes (worse than the realtime causal EMA);
        # a tapered mean over a wider window matches/beats it with no lag.
        sigma = max(1e-3, self.radius / 1.5)
        # Band lookahead: symmetric (zero-phase) smoothing of the raw depth range
        # over the window, re-normalizing each frame to it -> removes the causal
        # band EMA's lag. (lo_s, hi_s) is the Gaussian-weighted mean of raw bands.
        lo_s, hi_s, r_c = self._symmetric_band(c, sigma)

        def _reband(item, arr):
            if lo_s is None:
                return arr
            lo_u, hi_u, _, _ = item["band"]
            span_s = hi_s - lo_s
            r = max(hi_u - lo_u, 1e-6) / span_s
            s = (lo_u - lo_s) / span_s
            return arr * np.float32(r) + np.float32(s)

        acc = _reband(centre, base_c).astype(np.float32).copy()  # centre weight = 1
        wsum = 1.0
        max_resid_small = np.zeros_like(small_c)
        for i, item in enumerate(self._buf):
            if i == c:
                continue
            wt = float(np.exp(-((i - c) ** 2) / (2.0 * sigma * sigma)))
            dx_s, dy_s = self._shift_small(item["small"], small_c)
            if abs(dx_s) >= 0.5 or abs(dy_s) >= 0.5:
                m_small = np.array([[1.0, 0.0, dx_s], [0.0, 1.0, dy_s]], dtype=np.float32)
                small_aligned = cv2.warpAffine(item["small"], m_small, (sw, sh),
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                m_base = np.array([[1.0, 0.0, dx_s * sx], [0.0, 1.0, dy_s * sy]], dtype=np.float32)
                base_aligned = cv2.warpAffine(item["base"], m_base, (w, h),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                small_aligned = item["small"]
                base_aligned = item["base"]
            acc += wt * _reband(item, base_aligned)
            wsum += wt
            np.maximum(max_resid_small, np.abs(small_c - small_aligned), out=max_resid_small)
        base_agg = (acc / wsum).astype(np.float32)
        base_c_s = _reband(centre, base_c)
        # Static-only blend: smoothed where the window agrees (low residual), centre
        # base where it doesn't (moving object / disocclusion).
        m = 1.0 - np.clip((max_resid_small - _EVID_R_LO) / max(1e-3, _EVID_R_HI - _EVID_R_LO), 0.0, 1.0)
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        base_out = base_c_s + m * (base_agg - base_c_s)
        out = base_out + centre["detail"] * np.float32(r_c)
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(np.float32, copy=False), centre["payload"]

    def _symmetric_band(self, c: int, sigma: float):
        """Gaussian-weighted symmetric (lo, hi) over the windowed raw bands, and
        the centre frame's re-band scale r_c. Returns (None, None, 1.0) when any
        frame lacks band info (lookahead disabled)."""
        if any(it["band"] is None for it in self._buf):
            return None, None, 1.0
        wsum = 0.0
        lo_acc = 0.0
        hi_acc = 0.0
        for i, it in enumerate(self._buf):
            wt = 1.0 if i == c else float(np.exp(-((i - c) ** 2) / (2.0 * sigma * sigma)))
            _, _, lo_raw, hi_raw = it["band"]
            lo_acc += wt * lo_raw
            hi_acc += wt * hi_raw
            wsum += wt
        lo_s = lo_acc / wsum
        hi_s = hi_acc / wsum
        if not (np.isfinite(lo_s) and np.isfinite(hi_s) and hi_s - lo_s > 1e-6):
            return None, None, 1.0
        lo_uc, hi_uc, _, _ = self._buf[c]["band"]
        r_c = max(hi_uc - lo_uc, 1e-6) / (hi_s - lo_s)
        return lo_s, hi_s, r_c


def _smooth_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.size == 0:
        return arr
    # 3x3 binomial blur == separable Gaussian; cv2 runs it far faster than the
    # numpy shifted-add form.
    return cv2.GaussianBlur(arr, (3, 3), 0, borderType=cv2.BORDER_REPLICATE)


def _normalize_near(depth: np.ndarray, temporal: TemporalDepthStabilizer | None = None) -> np.ndarray:
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
    lo_raw, hi_raw = (float(v) for v in np.percentile(sample, [5.0, 95.0]))
    if not np.isfinite(lo_raw) or not np.isfinite(hi_raw) or hi_raw <= lo_raw:
        if temporal is not None:
            temporal.reset()
        return np.zeros(d.shape, dtype=np.float32)
    lo, hi = temporal.normalization_band(lo_raw, hi_raw) if temporal is not None else (lo_raw, hi_raw)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        if temporal is not None:
            temporal.reset()
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


def near_from_depth(
    depth: np.ndarray,
    hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
    temporal: TemporalDepthStabilizer | None = None,
    frame_rgb: np.ndarray | None = None,
    *,
    soft_shift_sharpen: bool = False,
) -> np.ndarray:
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
    if temporal is not None:
        temporal.begin_frame(frame_rgb)  # scene-cut reset before band/base smoothing
    if hole_fill_mode == HOLE_FILL_SOFT_SHIFT:
        # No pre-smooth; grow foreground so the gap is bounded by clean
        # background (the GPU warp kernel still snaps the edge hard via _near_at).
        near = _normalize_near(depth, temporal)
        if temporal is not None:
            near = temporal.stabilize_near(near, frame_rgb)
        return _sharpen_near_edges(near) if soft_shift_sharpen else _dilate_near_fg(near)
    near = _normalize_near(_smooth_depth(depth), temporal)
    if temporal is not None:
        near = temporal.stabilize_near(near, frame_rgb)
    return near


def near_for_render(
    near: np.ndarray,
    hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
    *,
    soft_shift_sharpen: bool = False,
) -> np.ndarray:
    """Prepare an already-normalized near/disparity map for the renderer."""
    n = np.asarray(near, dtype=np.float32)
    if n.size == 0:
        return n
    n = np.ascontiguousarray(n)
    np.nan_to_num(n, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(n, 0.0, 1.0, out=n)
    if hole_fill_mode == HOLE_FILL_SOFT_SHIFT:
        return _sharpen_near_edges(n) if soft_shift_sharpen else _dilate_near_fg(n)
    return n


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


def _make_soft_shift_pair_from_near(frame_rgb, near, eye_distance_mm):
    h, w, _ = frame_rgb.shape
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    left_raw, left_holes, left_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, 1.0)
    right_raw, right_holes, right_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, -1.0)
    left = _soft_blend_holes_rgb(_shift_fill_holes_rgb(left_raw, left_holes, -1), left_holes, left_near)
    right = _soft_blend_holes_rgb(_shift_fill_holes_rgb(right_raw, right_holes, 1), right_holes, right_near)
    return left, right


def _make_soft_shift_pair(frame_rgb, depth, eye_distance_mm):
    # soft_shift must use the un-smoothed depth -- see near_from_depth -- then
    # snap soft contours to hard edges so the forward warp doesn't spray
    # foreground slivers into the disocclusion gap (see _sharpen_near_edges).
    return _make_soft_shift_pair_from_near(
        frame_rgb,
        near_from_depth(depth, HOLE_FILL_SOFT_SHIFT, frame_rgb=frame_rgb, soft_shift_sharpen=True),
        eye_distance_mm,
    )


def _make_inverse_warp_pair_from_near(frame_rgb, near, eye_distance_mm):
    h, w, _ = frame_rgb.shape
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    disparity = near * max_shift
    cols = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    left = _sample_horizontal_rgb(frame_rgb, cols - disparity * 0.5)
    right = _sample_horizontal_rgb(frame_rgb, cols + disparity * 0.5)
    return left, right


def _make_inverse_warp_pair(frame_rgb, depth, eye_distance_mm):
    return _make_inverse_warp_pair_from_near(
        frame_rgb,
        near_from_depth(depth, HOLE_FILL_INVERSE_WARP, frame_rgb=frame_rgb),
        eye_distance_mm,
    )


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

    def __init__(
        self,
        src_w,
        src_h,
        projection,
        eye_distance_mm=DEFAULT_EYE_DISTANCE_MM,
        hole_fill_mode=DEFAULT_HOLE_FILL_MODE,
        flat_fov_deg=DEFAULT_FLAT_FOV_DEG,
        *,
        temporal_norm: bool = DEFAULT_TEMPORAL_NORM,
        temporal_norm_alpha: float = DEFAULT_TEMPORAL_NORM_ALPHA,
        temporal_norm_reset: float = DEFAULT_TEMPORAL_NORM_RESET,
        temporal_depth: bool = DEFAULT_TEMPORAL_DEPTH,
        temporal_depth_mode: str = DEFAULT_TEMPORAL_DEPTH_MODE,
        temporal_depth_alpha: float = DEFAULT_TEMPORAL_DEPTH_ALPHA,
        temporal_flow_diff: float = DEFAULT_TEMPORAL_FLOW_DIFF,
        temporal_flow_consistency: float = DEFAULT_TEMPORAL_FLOW_CONSISTENCY,
        temporal_flow_motion_gate: float = DEFAULT_TEMPORAL_FLOW_MOTION_GATE,
        temporal_affine: bool = DEFAULT_TEMPORAL_AFFINE,
        temporal_affine_max_scale: float = DEFAULT_TEMPORAL_AFFINE_MAX_SCALE,
        temporal_affine_max_bias: float = DEFAULT_TEMPORAL_AFFINE_MAX_BIAS,
        temporal_static_deadband_px: float = DEFAULT_TEMPORAL_STATIC_DEADBAND_PX,
        temporal_static_max_step_px: float = DEFAULT_TEMPORAL_STATIC_MAX_STEP_PX,
        temporal_motion_max_step_px: float = DEFAULT_TEMPORAL_MOTION_MAX_STEP_PX,
    ):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.projection = projection
        self.eye_distance_mm = float(eye_distance_mm)
        self.hole_fill_mode = hole_fill_mode
        self.max_shift = _max_disparity_pixels(self.src_w, self.eye_distance_mm)
        self.temporal = TemporalDepthStabilizer(
            norm_enabled=temporal_norm,
            norm_alpha=temporal_norm_alpha,
            norm_reset_threshold=temporal_norm_reset,
            depth_enabled=temporal_depth,
            depth_mode=temporal_depth_mode,
            depth_alpha=temporal_depth_alpha,
            flow_diff_threshold=temporal_flow_diff,
            flow_consistency_threshold=temporal_flow_consistency,
            flow_motion_gate=temporal_flow_motion_gate,
            affine_enabled=temporal_affine,
            affine_max_scale_delta=temporal_affine_max_scale,
            affine_max_bias=temporal_affine_max_bias,
            max_disparity_px=self.max_shift,
            static_deadband_px=temporal_static_deadband_px,
            static_max_step_px=temporal_static_max_step_px,
            motion_max_step_px=temporal_motion_max_step_px,
        )
        self.pmap = make_projection_map(src_w, src_h, projection, flat_fov_deg)
        self.out_w = self.pmap.out_w * 2
        self.out_h = self.pmap.out_h
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

    def reset(self) -> None:
        self.temporal.reset()

    def prepare_near(self, depth: np.ndarray, frame_rgb: np.ndarray | None = None) -> np.ndarray:
        return near_from_depth(
            depth,
            self.hole_fill_mode,
            self.temporal,
            frame_rgb,
            soft_shift_sharpen=self.hole_fill_mode == HOLE_FILL_SOFT_SHIFT,
        )

    def render(self, frame_rgb, depth):
        """Render one SBS frame. ``depth`` may be at any resolution -- the fast
        inverse-warp path normalizes it cheaply and upscales the disparity."""
        if self.hole_fill_mode == HOLE_FILL_INVERSE_WARP and self.pmap.is_identity:
            # Fast path: normalize at depth res, upscale the (smooth) half-
            # disparity into preallocated scratch, build both eye maps in place,
            # then inverse-sample straight into the SBS halves.
            near = self.prepare_near(depth, frame_rgb)
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
        near = self.prepare_near(depth, frame_rgb)
        if self.hole_fill_mode == HOLE_FILL_INVERSE_WARP:
            left, right = _make_inverse_warp_pair_from_near(frame_rgb, near, self.eye_distance_mm)
        else:
            left, right = _make_soft_shift_pair_from_near(frame_rgb, near, self.eye_distance_mm)
        if self.pmap.is_identity:
            np.concatenate([left, right], axis=1, out=self._out)
            return self._out
        _sample_rgb_xy_into(left, self.pmap, self._out[:, :self.pmap.out_w])
        _sample_rgb_xy_into(right, self.pmap, self._out[:, self.pmap.out_w:])
        return self._out

    def render_near(self, frame_rgb, near):
        """Render one SBS frame from an already-normalized near/disparity map."""
        if self.hole_fill_mode == HOLE_FILL_INVERSE_WARP and self.pmap.is_identity:
            near = near_for_render(near, self.hole_fill_mode)
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
        near = np.asarray(near, dtype=np.float32)
        if near.shape != (self.src_h, self.src_w):
            near = cv2.resize(near, (self.src_w, self.src_h), interpolation=cv2.INTER_LINEAR)
        near = near_for_render(
            near,
            self.hole_fill_mode,
            soft_shift_sharpen=self.hole_fill_mode == HOLE_FILL_SOFT_SHIFT,
        )
        if self.hole_fill_mode == HOLE_FILL_INVERSE_WARP:
            left, right = _make_inverse_warp_pair_from_near(frame_rgb, near, self.eye_distance_mm)
        else:
            left, right = _make_soft_shift_pair_from_near(frame_rgb, near, self.eye_distance_mm)
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
