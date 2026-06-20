from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest


def _depth_from_inverse(lo: float, hi: float, h: int = 64, w: int = 64) -> np.ndarray:
    inv = np.linspace(lo, hi, w, dtype=np.float32)[None, :]
    inv = np.repeat(inv, h, axis=0)
    return np.reciprocal(inv)


def _raw_band(depth: np.ndarray) -> tuple[float, float]:
    inv = np.reciprocal(np.maximum(depth.astype(np.float32), 1e-6))
    sample = inv[::4, ::4].ravel()
    lo, hi = np.percentile(sample, [5.0, 95.0])
    return float(lo), float(hi)


def test_temporal_normalization_smooths_percentile_band() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer, _normalize_near

    d1 = _depth_from_inverse(1.0, 2.0)
    d2 = _depth_from_inverse(2.0, 3.0)
    state = TemporalDepthStabilizer(
        norm_enabled=True,
        norm_alpha=0.25,
        norm_reset_threshold=100.0,
        depth_enabled=False,
    )

    _normalize_near(d1, state)
    lo1, hi1 = state.norm_band or (0.0, 0.0)
    lo2_raw, hi2_raw = _raw_band(d2)
    _normalize_near(d2, state)

    lo2, hi2 = state.norm_band or (0.0, 0.0)
    assert lo2 == pytest.approx((0.75 * lo1) + (0.25 * lo2_raw), rel=1e-6)
    assert hi2 == pytest.approx((0.75 * hi1) + (0.25 * hi2_raw), rel=1e-6)


def test_temporal_normalization_resets_on_large_band_jump() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer, _normalize_near

    d1 = _depth_from_inverse(1.0, 2.0)
    d2 = _depth_from_inverse(8.0, 9.0)
    state = TemporalDepthStabilizer(
        norm_enabled=True,
        norm_alpha=0.10,
        norm_reset_threshold=0.5,
        depth_enabled=True,
        depth_alpha=0.25,
    )

    _normalize_near(d1, state)
    state.stabilize_near(np.zeros((8, 8), dtype=np.float32))
    assert state.norm_band is not None
    _normalize_near(d2, state)

    assert state.norm_band == pytest.approx(_raw_band(d2), rel=1e-6)
    assert state._near_prev is None


def test_optional_near_map_ema_uses_depth_alpha() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        norm_enabled=False,
        depth_enabled=True,
        depth_mode="ema",
        depth_alpha=0.25,
        affine_enabled=False,
    )
    first = np.zeros((4, 4), dtype=np.float32)
    second = np.ones((4, 4), dtype=np.float32)

    np.testing.assert_array_equal(state.stabilize_near(first), first)
    blended = state.stabilize_near(second)

    np.testing.assert_allclose(blended, np.full((4, 4), 0.25, dtype=np.float32))


def test_invalid_depth_mode_falls_back_to_lightweight_ema() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(norm_enabled=False, depth_enabled=True, depth_mode="bad")

    assert state.depth_mode == "ema"


def test_base_detail_converges_to_input_when_stable() -> None:
    # Key safety property vs the old per-pixel stabilizer: when the input is
    # steady, the output must converge to the input and leave object structure
    # (sharp depth edges) intact -- no permanent distortion, nothing for
    # soft_shift/inverse_warp to tear into phantom holes.
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        norm_enabled=False,
        depth_enabled=True,
        depth_mode="ema",
        depth_alpha=0.20,
        affine_enabled=False,
    )
    cur = np.full((64, 64), 0.4, dtype=np.float32)
    cur[:, 32:] = 0.9  # sharp object boundary

    state.stabilize_near(cur)  # frame 1: identity, seeds the base
    out = state.stabilize_near(cur)  # frame 2: input unchanged -> converged

    np.testing.assert_allclose(out, cur, atol=1e-6)


def test_base_detail_attenuates_base_level_jitter() -> None:
    # The whole point: a frame-to-frame shift in the overall (low-frequency)
    # depth level -- the "swimming" that causes nausea -- is attenuated by the
    # base EMA instead of passing straight through.
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        norm_enabled=False,
        depth_enabled=True,
        depth_mode="ema",
        depth_alpha=0.20,
        affine_enabled=False,
    )
    a = np.full((16, 16), 0.50, dtype=np.float32)
    b = np.full((16, 16), 0.60, dtype=np.float32)

    state.stabilize_near(a)  # identity, seeds base at 0.50
    out = state.stabilize_near(b)

    # base pulled toward prev: 0.50 * 0.8 + 0.60 * 0.2 = 0.52, not the full 0.60.
    np.testing.assert_allclose(out, np.full((16, 16), 0.52, dtype=np.float32), atol=1e-6)


def test_motion_comp_default_follows_env_default() -> None:
    from offline.two_dvr_render import DEFAULT_TEMPORAL_MOTION_COMP, TemporalDepthStabilizer

    state = TemporalDepthStabilizer(norm_enabled=False, depth_enabled=True, depth_mode="ema")
    assert state.motion_comp_enabled == DEFAULT_TEMPORAL_MOTION_COMP
    # explicit override always wins regardless of the default
    assert (
        TemporalDepthStabilizer(depth_enabled=True, depth_mode="ema", motion_comp_enabled=False).motion_comp_enabled
        is False
    )


def test_motion_comp_warps_base_to_follow_global_pan() -> None:
    # warp-then-filter (summary 8.6): when the camera pans by K px, the previous
    # base must be warped into the current frame by the same K before blending,
    # so static content lines up instead of smearing. Verifies the estimate sign
    # and the warp on a known circular shift.
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        norm_enabled=False, depth_enabled=True, depth_mode="ema", motion_comp_enabled=True
    )
    h = w = 128
    k = 6
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tex = np.sin(2 * np.pi * xx / 21.0) + np.cos(2 * np.pi * yy / 17.0) + 0.5 * np.sin(
        2 * np.pi * (xx + yy) / 13.0
    )
    tex = ((tex - tex.min()) / np.ptp(tex) * 255).astype(np.uint8)
    prev_gray = tex
    cur_gray = np.roll(tex, k, axis=1)  # content shifted right by k px
    state._mc_gray_prev = prev_gray

    base_prev = tex.astype(np.float32) / 255.0  # base lives in prev-frame coords
    warped = state._motion_compensate_base(base_prev, cur_gray)

    truth = np.roll(base_prev, k, axis=1)  # base shifted to follow the pan
    inner = (slice(2, h - 2), slice(k + 2, w - 2))  # ignore wrap/replicate borders
    err_warped = float(np.abs(warped[inner] - truth[inner]).mean())
    err_nocomp = float(np.abs(base_prev[inner] - truth[inner]).mean())
    # Motion comp must materially reduce misalignment (right direction + size).
    assert err_warped < 0.4 * err_nocomp, (err_warped, err_nocomp)


def test_evidence_gate_default_follows_env_default() -> None:
    from offline.two_dvr_render import DEFAULT_TEMPORAL_EVIDENCE_GATE, TemporalDepthStabilizer

    state = TemporalDepthStabilizer(depth_enabled=True, depth_mode="ema")
    assert state.evidence_gate_enabled == DEFAULT_TEMPORAL_EVIDENCE_GATE


def test_evidence_alpha_locks_static_and_follows_motion() -> None:
    # 8.6.3: where the motion-compensated residual is ~0 (static), the base-EMA
    # alpha drops to a_lock (= depth_alpha * lock_scale) so the depth locks
    # harder; where the residual is large (unexplained local motion), alpha -> 1
    # so the base follows the current frame.
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        depth_enabled=True, depth_mode="ema", depth_alpha=0.2, evidence_lock_scale=0.5
    )
    h = w = 64
    prev = np.zeros((h, w), dtype=np.uint8)

    static_alpha = state._evidence_alpha(np.zeros((h, w), dtype=np.uint8), prev)
    np.testing.assert_allclose(static_alpha, 0.1, atol=1e-4)  # 0.2 * 0.5

    cur = np.zeros((h, w), dtype=np.uint8)
    cur[:, w // 2:] = 255  # right half changes a lot
    alpha = state._evidence_alpha(cur, prev)
    assert float(alpha[:, 48:].mean()) > 0.9   # changed region follows current
    assert float(alpha[:, :16].mean()) < 0.2   # static region stays locked


def test_band_lookahead_reduces_depth_range_flicker() -> None:
    # With per-frame band info, the window re-normalizes each frame to the
    # symmetric (lookahead) raw band, so a flickering depth range on otherwise
    # static content no longer produces a flickering global near level.
    from offline.two_dvr_render import SymmetricBaseWindow

    h = w = 48
    rng = np.random.default_rng(0)
    inv = np.tile(np.linspace(1.0, 3.0, w, dtype=np.float32), (h, 1))  # static depth
    gray = (rng.random((h, w)) * 255).astype(np.uint8)
    n = 18
    raw_means: list[float] = []
    win = SymmetricBaseWindow(radius=3)
    res: dict = {}
    for i in range(n):
        lo = 1.0 + float(rng.standard_normal() * 0.08)
        hi = 3.0 + float(rng.standard_normal() * 0.08)  # jittery depth range
        near = np.clip((inv - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        raw_means.append(float(near.mean()))
        for o, p in win.push(near, gray, i, band=(lo, hi, lo, hi)):
            res[p] = o
    for o, p in win.flush():
        res[p] = o
    out_means = np.array([res[i].mean() for i in range(n)])
    assert out_means.std() < np.array(raw_means).std() * 0.6


def test_band_lookahead_disabled_without_band_info() -> None:
    # push without band -> no re-band (backward compatible); output equals the
    # plain Gaussian-mean window.
    from offline.two_dvr_render import SymmetricBaseWindow

    win = SymmetricBaseWindow(radius=2)
    near = np.full((16, 16), 0.5, dtype=np.float32)
    gray = np.zeros((16, 16), dtype=np.uint8)
    out = []
    for i in range(6):
        out += win.push(near, gray, i)  # band defaults to None
    out += win.flush()
    assert all(np.allclose(o, 0.5, atol=1e-5) for o, _ in out)


def test_scene_cut_resets_temporal_state() -> None:
    # On a hard cut, begin_frame must reset the temporal state so the new shot's
    # depth isn't blended with the old one's.
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(depth_enabled=True, depth_mode="ema", scene_cut_enabled=True)
    rng = np.random.default_rng(0)
    scene_a = (rng.random((120, 200, 3)) * 255).astype(np.uint8)
    scene_b = np.zeros((120, 200, 3), dtype=np.uint8)
    scene_b[:, :, 2] = 220  # very different colour histogram

    state.stabilize_near(np.full((120, 200), 0.5, dtype=np.float32), scene_a)
    assert state._base_prev is not None  # state seeded within shot A
    assert state.begin_frame(scene_a) is False  # same shot -> no reset
    assert state.begin_frame(scene_b) is True    # hard cut -> reset
    assert state._base_prev is None              # temporal state cleared for shot B


def test_scene_cut_disabled_has_no_detector() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(depth_enabled=True, depth_mode="ema", scene_cut_enabled=False)
    assert state._scene_detector is None
    assert state.begin_frame(np.zeros((8, 8, 3), dtype=np.uint8)) is False


def test_first_frame_is_identity() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(norm_enabled=False, depth_enabled=True, depth_mode="ema")
    near = np.random.default_rng(0).random((48, 80), dtype=np.float32)

    np.testing.assert_array_equal(state.stabilize_near(near), near)


def test_affine_matching_reduces_global_near_gain_jitter() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        norm_enabled=False,
        depth_enabled=True,
        depth_mode="ema",
        depth_alpha=1.0,
        flow_diff_threshold=0.0,
        affine_enabled=True,
        affine_max_scale_delta=0.50,
        affine_max_bias=0.50,
    )
    prev = np.tile(np.linspace(0.1, 0.9, 64, dtype=np.float32), (32, 1))
    cur = np.clip(prev * 1.20 + 0.05, 0.0, 1.0)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)

    state.stabilize_near(prev, frame)
    out = state.stabilize_near(cur, frame)

    assert abs(float(out.mean()) - float(prev.mean())) < abs(float(cur.mean()) - float(prev.mean()))
    assert abs(float(out.std()) - float(prev.std())) < abs(float(cur.std()) - float(prev.std()))


def test_static_deadband_and_max_step_are_in_disparity_pixels() -> None:
    from offline.two_dvr_render import TemporalDepthStabilizer

    state = TemporalDepthStabilizer(
        norm_enabled=False,
        depth_enabled=True,
        depth_mode="ema",
        depth_alpha=1.0,
        flow_diff_threshold=10.0,
        affine_enabled=False,
        max_disparity_px=10.0,
        static_deadband_px=0.25,
        static_max_step_px=0.75,
        motion_max_step_px=3.0,
        motion_comp_enabled=False,
        evidence_gate_enabled=False,
    )
    prev = np.full((4, 4), 0.50, dtype=np.float32)
    small = np.full((4, 4), 0.52, dtype=np.float32)  # 0.2px at max_shift=10, inside deadband.
    large = np.full((4, 4), 0.70, dtype=np.float32)  # 2.0px request, clamped to 0.75px.
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    state.stabilize_near(prev, frame)
    np.testing.assert_allclose(state.stabilize_near(small, frame), prev)
    out = state.stabilize_near(large, frame)

    np.testing.assert_allclose(out, np.full((4, 4), 0.575, dtype=np.float32), atol=1e-6)


def test_flow_mode_warps_previous_near_before_blending(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    from offline.two_dvr_render import TemporalDepthStabilizer

    def fake_flow(cur, prev, *_args):
        flow = np.zeros((cur.shape[0], cur.shape[1], 2), dtype=np.float32)
        flow[:, :, 0] = -1.0
        return flow

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", fake_flow)
    state = TemporalDepthStabilizer(
        norm_enabled=False,
        depth_enabled=True,
        depth_mode="flow",
        depth_alpha=0.25,
        flow_diff_threshold=0.0,
        flow_consistency_threshold=0.0,
        affine_enabled=False,
    )
    prev = np.tile(np.arange(6, dtype=np.float32), (4, 1)) / 5.0
    cur = np.zeros((4, 6), dtype=np.float32)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    state.stabilize_near(prev, frame)
    out = state.stabilize_near(cur, frame)

    x = np.broadcast_to(np.arange(6, dtype=np.float32)[None, :] - 1.0, (4, 6))
    y = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], (4, 6))
    expected_prev = cv2.remap(prev, x, y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    np.testing.assert_allclose(out, expected_prev * 0.75, atol=1e-6)


def test_symmetric_window_emits_every_frame_once_in_order() -> None:
    from offline.two_dvr_render import SymmetricBaseWindow

    win = SymmetricBaseWindow(radius=2)
    h = w = 32
    gray = np.zeros((h, w), dtype=np.uint8)
    emitted: list = []
    n = 10
    for i in range(n):
        near = np.full((h, w), 0.5, dtype=np.float32)
        emitted += win.push(near, gray, i)
    emitted += win.flush()
    assert [p for _, p in emitted] == list(range(n))  # every frame, exactly once, in order


def test_symmetric_window_reduces_static_base_jitter() -> None:
    from offline.two_dvr_render import SymmetricBaseWindow

    win = SymmetricBaseWindow(radius=2)
    h = w = 48
    rng = np.random.default_rng(0)
    gray = (rng.random((h, w)) * 255).astype(np.uint8)  # static textured frame
    n = 14
    inp: list[float] = []
    out_pairs: list[tuple[int, float]] = []
    for i in range(n):
        flick = float(rng.standard_normal() * 0.05)  # per-frame global depth jitter
        near = np.full((h, w), 0.5 + flick, dtype=np.float32)
        inp.append(0.5 + flick)
        for o, p in win.push(near, gray, i):
            out_pairs.append((p, float(o.mean())))
    for o, p in win.flush():
        out_pairs.append((p, float(o.mean())))
    out_pairs.sort()
    out_means = np.array([v for _, v in out_pairs])
    # Symmetric median over the window strips the per-frame jitter (no lag).
    assert out_means.std() < np.array(inp).std() * 0.7


def test_symmetric_window_preserves_moving_region() -> None:
    from offline.two_dvr_render import SymmetricBaseWindow

    win = SymmetricBaseWindow(radius=2)
    h = w = 64
    rng = np.random.default_rng(1)
    gray_base = (rng.random((h, w)) * 255).astype(np.uint8)
    n = 7
    results: dict = {}
    for i in range(n):
        g = gray_base.copy()
        g[20:44, 30 + i * 3:50 + i * 3] = 240  # bright object sweeping right
        near = np.full((h, w), 0.4, dtype=np.float32)
        near[20:44, 30 + i * 3:50 + i * 3] = 0.8  # object at a different depth
        for o, p in win.push(near, g, i):
            results[p] = o
    for o, p in win.flush():
        results[p] = o
    out3 = results[3]  # object spans cols 39..59 in frame 3
    assert float(out3[32, 48]) > 0.6        # moving object keeps its depth (not medianed away)
    assert abs(float(out3[5, 5]) - 0.4) < 0.1  # static background unchanged


def test_stereo_renderer_reset_clears_temporal_state() -> None:
    from offline import two_dvr_render as render

    renderer = render.StereoRenderer(
        64,
        64,
        render.PROJECTION_FLAT_3D,
        hole_fill_mode=render.HOLE_FILL_INVERSE_WARP,
        temporal_norm=True,
        temporal_norm_alpha=0.10,
        temporal_norm_reset=100.0,
    )

    renderer.prepare_near(_depth_from_inverse(1.0, 2.0))
    assert renderer.temporal.norm_band is not None

    renderer.reset()

    assert renderer.temporal.norm_band is None


def test_two_dvr_cli_accepts_temporal_flags() -> None:
    from offline import two_dvr

    parser = argparse.ArgumentParser()
    two_dvr._add_common_args(parser)
    args = parser.parse_args([
        "--no-temporal-norm",
        "--temporal-depth",
        "--temporal-depth-alpha",
        "0.30",
        "--temporal-static-deadband-px",
        "0.40",
        "--temporal-static-max-step-px",
        "0.80",
        "--temporal-motion-max-step-px",
        "2.50",
        "--depth-stabilizer",
        "nvds",
    ])

    assert args.temporal_norm is False
    assert args.temporal_depth is True
    assert args.temporal_depth_mode == "ema"
    assert args.temporal_depth_alpha == pytest.approx(0.30)
    assert args.temporal_static_deadband_px == pytest.approx(0.40)
    assert args.temporal_static_max_step_px == pytest.approx(0.80)
    assert args.temporal_motion_max_step_px == pytest.approx(2.50)
    assert args.depth_stabilizer == "nvds"


def test_stereo_renderer_can_render_precomputed_near() -> None:
    from offline import two_dvr_render as render

    frame = np.full((16, 16, 3), 128, dtype=np.uint8)
    near = np.linspace(0.0, 1.0, 8 * 8, dtype=np.float32).reshape(8, 8)
    renderer = render.StereoRenderer(
        16,
        16,
        render.PROJECTION_FLAT_3D,
        hole_fill_mode=render.HOLE_FILL_INVERSE_WARP,
        temporal_depth=False,
    )

    out = renderer.render_near(frame, near)

    assert out.shape == (16, 32, 3)
    assert out.dtype == np.uint8


def test_make_renderer_passes_temporal_options_to_cpu_renderer() -> None:
    from offline import two_dvr
    from offline import two_dvr_render as render

    args = SimpleNamespace(
        gpu_render="off",
        hole_fill=render.HOLE_FILL_INVERSE_WARP,
        projection=render.PROJECTION_FLAT_3D,
        eye_distance=render.DEFAULT_EYE_DISTANCE_MM,
        strength=render.DEFAULT_STRENGTH,
        flat_fov=render.DEFAULT_FLAT_FOV_DEG,
        temporal_norm=False,
        temporal_norm_alpha=0.20,
        temporal_norm_reset=2.0,
        temporal_depth=True,
        temporal_depth_mode="ema",
        temporal_depth_alpha=0.30,
        temporal_static_deadband_px=0.40,
        temporal_static_max_step_px=0.80,
        temporal_motion_max_step_px=2.50,
    )

    renderer = two_dvr._make_renderer(32, 32, args)

    assert isinstance(renderer, render.StereoRenderer)
    assert renderer.temporal.norm_enabled is False
    assert renderer.temporal.depth_enabled is True
    assert renderer.temporal.depth_mode == "ema"
    assert renderer.temporal.depth_alpha == pytest.approx(0.30)
    assert renderer.temporal.static_deadband_px == pytest.approx(0.40)
    assert renderer.temporal.static_max_step_px == pytest.approx(0.80)
    assert renderer.temporal.motion_max_step_px == pytest.approx(2.50)
