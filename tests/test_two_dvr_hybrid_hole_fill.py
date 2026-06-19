"""Review prototype for 2D->3D soft_shift hole filling.

This file intentionally does not change the production renderer. It documents
and tests the proposed hybrid semantics: keep the soft_shift forward-warp result
where z-buffer wrote a pixel, and fill only soft_shift holes from inverse_warp.
The GPU implementation still needs a production-safe merge path.
"""

from __future__ import annotations

import numpy as np
import pytest


def hybrid_fill_soft_holes_from_inverse(
    soft_sbs_rgb: np.ndarray,
    inverse_sbs_rgb: np.ndarray,
    soft_zbuf: np.ndarray,
) -> np.ndarray:
    """Return soft_shift SBS with zbuf holes replaced by inverse_warp pixels.

    ``soft_zbuf == 0`` is the same hole definition used by the GPU soft_shift
    kernels. All non-hole pixels remain byte-for-byte identical to soft_shift.
    """
    soft = np.asarray(soft_sbs_rgb)
    inverse = np.asarray(inverse_sbs_rgb)
    zbuf = np.asarray(soft_zbuf)
    if soft.shape != inverse.shape:
        raise ValueError(f"soft/inverse shape mismatch: {soft.shape} vs {inverse.shape}")
    if soft.ndim != 3 or soft.shape[2] != 3:
        raise ValueError(f"expected RGB image shape (H,W,3), got {soft.shape}")
    if zbuf.shape != soft.shape[:2]:
        raise ValueError(f"zbuf shape {zbuf.shape} does not match image {soft.shape[:2]}")

    out = soft.copy()
    holes = zbuf == 0
    out[holes] = inverse[holes]
    return out


def _mean_vertical_step(image: np.ndarray, mask: np.ndarray) -> float:
    luma = image.astype(np.float32).mean(axis=2)
    both = mask[1:, :] & mask[:-1, :]
    if not np.any(both):
        return 0.0
    return float(np.abs(luma[1:, :] - luma[:-1, :])[both].mean())


def test_hybrid_fill_replaces_only_zbuf_holes() -> None:
    soft = np.zeros((2, 4, 3), dtype=np.uint8)
    inverse = np.full((2, 4, 3), 200, dtype=np.uint8)
    soft[:, :, 0] = np.arange(8, dtype=np.uint8).reshape(2, 4)
    zbuf = np.ones((2, 4), dtype=np.int32)
    zbuf[0, 1] = 0
    zbuf[1, 3] = 0

    out = hybrid_fill_soft_holes_from_inverse(soft, inverse, zbuf)

    holes = zbuf == 0
    np.testing.assert_array_equal(out[holes], inverse[holes])
    np.testing.assert_array_equal(out[~holes], soft[~holes])


def test_hybrid_fill_reduces_synthetic_row_copy_banding() -> None:
    h, w = 24, 32
    soft = np.full((h, w, 3), 196, dtype=np.uint8)
    inverse = soft.copy()
    zbuf = np.ones((h, w), dtype=np.int32)

    hole = np.zeros((h, w), dtype=bool)
    hole[4:20, 10:18] = True
    zbuf[hole] = 0

    # soft_shift's directional row-copy fill can turn wall texture into visible
    # horizontal bands inside a wide disocclusion hole.
    for y in range(h):
        soft[y, 10:18] = 130 if y % 2 else 220
    inverse[hole] = 190

    out = hybrid_fill_soft_holes_from_inverse(soft, inverse, zbuf)

    assert _mean_vertical_step(out, hole) < _mean_vertical_step(soft, hole) * 0.25
    np.testing.assert_array_equal(out[~hole], soft[~hole])


def test_gpu_rim_cleanup_defaults_to_auto_and_allows_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from offline.two_dvr_gpu import _two_dvr_rim_width

    monkeypatch.delenv("PT_TWO_DVR_RIM", raising=False)
    assert _two_dvr_rim_width(1920) == 16
    assert _two_dvr_rim_width(3840) == 32
    assert _two_dvr_rim_width(720) == 6

    monkeypatch.setenv("PT_TWO_DVR_RIM", "8")
    assert _two_dvr_rim_width(1920) == 8

    monkeypatch.setenv("PT_TWO_DVR_RIM", "-3")
    assert _two_dvr_rim_width(1920) == 0

    monkeypatch.setenv("PT_TWO_DVR_RIM", "bad")
    assert _two_dvr_rim_width(1920) == 0


def test_gpu_fg_bad_cleanup_defaults_to_auto_and_allows_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from offline.two_dvr_gpu import _two_dvr_fg_bad_width

    monkeypatch.delenv("PT_TWO_DVR_FG_BAD", raising=False)
    assert _two_dvr_fg_bad_width(1920) == 8
    assert _two_dvr_fg_bad_width(3840) == 16
    assert _two_dvr_fg_bad_width(720) == 3

    monkeypatch.setenv("PT_TWO_DVR_FG_BAD", "8")
    assert _two_dvr_fg_bad_width(1920) == 8

    monkeypatch.setenv("PT_TWO_DVR_FG_BAD", "-3")
    assert _two_dvr_fg_bad_width(1920) == 0

    monkeypatch.setenv("PT_TWO_DVR_FG_BAD", "bad")
    assert _two_dvr_fg_bad_width(1920) == 0


@pytest.mark.parametrize(
    ("soft", "inverse", "zbuf"),
    [
        (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 3, 3), np.uint8), np.ones((2, 2), np.int32)),
        (np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), np.ones((2, 2), np.int32)),
        (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), np.ones((2, 3), np.int32)),
    ],
)
def test_hybrid_fill_rejects_shape_mismatches(
    soft: np.ndarray,
    inverse: np.ndarray,
    zbuf: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        hybrid_fill_soft_holes_from_inverse(soft, inverse, zbuf)
