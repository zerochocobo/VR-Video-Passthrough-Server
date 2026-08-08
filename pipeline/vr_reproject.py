"""GPU hequirect <-> flat (gnomonic) reprojection for VR-to-flat restoration.

Used by the "VR转平面后解码" (VR-to-flat before restoration) path: a mosaic
region on a VR180 hequirect eye is gnomonic-projected to a flat rectilinear view
(where the mosaic becomes an axis-aligned square the restorer handles best),
restored, then reprojected and alpha-composited back.

Conventions match ffmpeg's ``v360=hequirect:flat`` (validated pixel-for-pixel):
  * hequirect eye spans 180x180 deg, pixel<->angle linear.
  * forward  (equirect->flat): rotate flat-local rays by R, sample equirect.
  * inverse  (flat->equirect): rotate equirect rays by R^T, project to flat.
The host builds R once (yaw/pitch/roll) and the inverse uses its transpose, so
forward and inverse are exact inverses of each other.

Both directions address the eye in-place inside a full SBS frame (stride +
origin), so no per-region copy of the eye is needed. ``blend_into`` composites
straight into the destination frame, so the caller allocates nothing per region.
"""
from __future__ import annotations

import math

import numpy as np

# Forward: flat-local ray -> world dir -> sample the hequirect eye living at
# (eye_x0, eye_y0) inside a src of row stride src_stride_w.
_FWD_SRC = r"""
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// One flat-view sample: pixel (x,y) of an fw x fh view -> bilinear tap on the eye.
__device__ __forceinline__ void sample_eye(
    const unsigned char* src, const float* R, float range_x, float range_y,
    int ew, int eh, int fw, int fh, int src_stride_w, int eye_x0, int eye_y0,
    int x, int y, float* out) {
  // flat pixel -> local camera ray (z forward, y up); image y grows downward.
  float lx = (2.f * (x + 0.5f) / fw - 1.f) * range_x;
  float ly = -(2.f * (y + 0.5f) / fh - 1.f) * range_y;
  float lz = 1.f;
  float inv = rsqrtf(lx*lx + ly*ly + lz*lz);
  lx *= inv; ly *= inv; lz *= inv;
  float wx = R[0]*lx + R[1]*ly + R[2]*lz;   // world = R * local
  float wy = R[3]*lx + R[4]*ly + R[5]*lz;
  float wz = R[6]*lx + R[7]*ly + R[8]*lz;
  float lon = atan2f(wx, wz);
  float lat = asinf(fmaxf(-1.f, fminf(1.f, wy)));
  float u = lon / (float)M_PI + 0.5f;       // 180 deg across the eye width
  float v = 0.5f - lat / (float)M_PI;
  float sx = u * ew - 0.5f;
  float sy = v * eh - 0.5f;
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  float ax = sx - x0, ay = sy - y0;
  int x1 = x0 + 1, y1 = y0 + 1;
  if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
  if (x1 > ew - 1) x1 = ew - 1; if (y1 > eh - 1) y1 = eh - 1;
  if (x0 > ew - 1) x0 = ew - 1; if (y0 > eh - 1) y0 = eh - 1;
  const unsigned char* s00 = src + ((eye_y0 + y0) * src_stride_w + (eye_x0 + x0)) * 3;
  const unsigned char* s01 = src + ((eye_y0 + y0) * src_stride_w + (eye_x0 + x1)) * 3;
  const unsigned char* s10 = src + ((eye_y0 + y1) * src_stride_w + (eye_x0 + x0)) * 3;
  const unsigned char* s11 = src + ((eye_y0 + y1) * src_stride_w + (eye_x0 + x1)) * 3;
  for (int c = 0; c < 3; c++)
    out[c] = s00[c]*(1-ax)*(1-ay) + s01[c]*ax*(1-ay)
           + s10[c]*(1-ax)*ay     + s11[c]*ax*ay;
}

extern "C" __global__ void equirect_to_flat_u8(
    const unsigned char* src, unsigned char* flat, const float* R,
    float range_x, float range_y, int ew, int eh, int fw, int fh,
    int src_stride_w, int eye_x0, int eye_y0) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= fw || y >= fh) return;
  float rgb[3];
  sample_eye(src, R, range_x, range_y, ew, eh, fw, fh,
             src_stride_w, eye_x0, eye_y0, x, y, rgb);
  unsigned char* d = flat + (y * fw + x) * 3;
  for (int c = 0; c < 3; c++) d[c] = (unsigned char)(rgb[c] + 0.5f);
}

// Supersampled variant: one thread per OUTPUT pixel, accumulating the ss x ss
// taps of the same view rendered at ss x resolution. Identical sample positions
// to rendering into an (fh*ss, fw*ss) buffer and box-averaging it down, but the
// accumulation stays in registers -- no 12 MB temporary, no second pass over it,
// and the average is taken before the u8 round instead of after.
extern "C" __global__ void equirect_to_flat_ss_u8(
    const unsigned char* src, unsigned char* flat, const float* R,
    float range_x, float range_y, int ew, int eh, int fw, int fh,
    int src_stride_w, int eye_x0, int eye_y0, int ss) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= fw || y >= fh) return;
  int bw = fw * ss, bh = fh * ss;
  float acc[3] = {0.f, 0.f, 0.f};
  for (int j = 0; j < ss; j++) {
    for (int i = 0; i < ss; i++) {
      float rgb[3];
      sample_eye(src, R, range_x, range_y, ew, eh, bw, bh,
                 src_stride_w, eye_x0, eye_y0, x * ss + i, y * ss + j, rgb);
      for (int c = 0; c < 3; c++) acc[c] += rgb[c];
    }
  }
  float inv_n = 1.f / (float)(ss * ss);
  unsigned char* d = flat + (y * fw + x) * 3;
  for (int c = 0; c < 3; c++) d[c] = (unsigned char)(acc[c] * inv_n + 0.5f);
}
"""

# Inverse: shared device math, then two kernels -- one emitting patch+alpha (for
# validation) and one blending straight into the destination frame (production).
_INV_SRC = r"""
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
__device__ __forceinline__ bool project_to_flat(
    int ex, int ey, int ew, int eh, const float* Rt,
    float range_x, float range_y, int fw, int fh, float* ofx, float* ofy) {
  float u = (ex + 0.5f) / ew;
  float v = (ey + 0.5f) / eh;
  float lon = (u - 0.5f) * (float)M_PI;
  float lat = (0.5f - v) * (float)M_PI;
  float cl = cosf(lat);
  float wx = cl * sinf(lon), wy = sinf(lat), wz = cl * cosf(lon);
  float lx = Rt[0]*wx + Rt[1]*wy + Rt[2]*wz;   // local = R^T * world
  float ly = Rt[3]*wx + Rt[4]*wy + Rt[5]*wz;
  float lz = Rt[6]*wx + Rt[7]*wy + Rt[8]*wz;
  if (lz <= 1e-6f) return false;
  float fx = (lx / lz / range_x * 0.5f + 0.5f) * fw - 0.5f;   // gnomonic
  float fy = (-ly / lz / range_y * 0.5f + 0.5f) * fh - 0.5f;
  if (fx < 0.f || fx > fw - 1.f || fy < 0.f || fy > fh - 1.f) return false;
  *ofx = fx; *ofy = fy;
  return true;
}

__device__ __forceinline__ void sample_flat(
    const unsigned char* flat, int fw, int fh, float fx, float fy, float* out) {
  int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
  float ax = fx - x0, ay = fy - y0;
  int x1 = x0 + 1, y1 = y0 + 1;
  if (x1 > fw - 1) x1 = fw - 1; if (y1 > fh - 1) y1 = fh - 1;
  const unsigned char* s00 = flat + (y0 * fw + x0) * 3;
  const unsigned char* s01 = flat + (y0 * fw + x1) * 3;
  const unsigned char* s10 = flat + (y1 * fw + x0) * 3;
  const unsigned char* s11 = flat + (y1 * fw + x1) * 3;
  for (int c = 0; c < 3; c++)
    out[c] = s00[c]*(1-ax)*(1-ay) + s01[c]*ax*(1-ay)
           + s10[c]*(1-ax)*ay     + s11[c]*ax*ay;
}

// Feather the alpha toward the flat view's border so the composite has no hard
// seam. ``feather`` is the fading band as a fraction of the flat half-extent.
__device__ __forceinline__ float edge_alpha(
    float fx, float fy, int fw, int fh, float feather) {
  if (feather <= 0.f) return 1.f;
  float u = fx / (fw - 1.f), v = fy / (fh - 1.f);
  float d = fminf(fminf(u, 1.f - u), fminf(v, 1.f - v));
  float a = d / feather;
  return a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
}

extern "C" __global__ void flat_to_equirect_u8(
    const unsigned char* flat, unsigned char* patch, unsigned char* alpha,
    const float* Rt, float range_x, float range_y,
    int ew, int eh, int fw, int fh,
    int cx0, int cy0, int cw, int ch, float feather) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cw || y >= ch) return;
  unsigned char* a = alpha + (y * cw + x);
  unsigned char* p = patch + (y * cw + x) * 3;
  float fx, fy;
  if (!project_to_flat(cx0 + x, cy0 + y, ew, eh, Rt, range_x, range_y, fw, fh, &fx, &fy)) {
    *a = 0; p[0] = p[1] = p[2] = 0; return;
  }
  float rgb[3];
  sample_flat(flat, fw, fh, fx, fy, rgb);
  for (int c = 0; c < 3; c++) p[c] = (unsigned char)(rgb[c] + 0.5f);
  *a = (unsigned char)(edge_alpha(fx, fy, fw, fh, feather) * 255.f + 0.5f);
}

// Mask-aware production path: same as flat_to_equirect_blend_u8, but the flat
// view carries a per-pixel alpha plane (e.g. a face mask) that is multiplied
// into the edge feather. Pixels the caller did not touch keep alpha 0 and are
// left exactly as they were, so nothing outside the mask is re-resampled.
extern "C" __global__ void flat_to_equirect_blend_masked_u8(
    const unsigned char* flat, const float* flat_alpha, unsigned char* dst,
    const float* Rt, float range_x, float range_y, int ew, int eh, int fw, int fh,
    int cx0, int cy0, int cw, int ch,
    int dst_stride_w, int eye_x0, int eye_y0, float feather) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cw || y >= ch) return;
  int ex = cx0 + x, ey = cy0 + y;
  float fx, fy;
  if (!project_to_flat(ex, ey, ew, eh, Rt, range_x, range_y, fw, fh, &fx, &fy)) return;
  float a = edge_alpha(fx, fy, fw, fh, feather);
  if (a <= 0.f) return;
  // Bilinear the alpha plane so the mask edge stays smooth after reprojection.
  int ax0 = (int)floorf(fx), ay0 = (int)floorf(fy);
  float tx = fx - ax0, ty = fy - ay0;
  int ax1 = min(fw - 1, ax0 + 1), ay1 = min(fh - 1, ay0 + 1);
  if (ax0 < 0) ax0 = 0; if (ay0 < 0) ay0 = 0;
  float m = flat_alpha[ay0 * fw + ax0] * (1-tx) * (1-ty)
          + flat_alpha[ay0 * fw + ax1] * tx * (1-ty)
          + flat_alpha[ay1 * fw + ax0] * (1-tx) * ty
          + flat_alpha[ay1 * fw + ax1] * tx * ty;
  a *= m;
  if (a <= 0.001f) return;
  if (a > 1.f) a = 1.f;
  float rgb[3];
  sample_flat(flat, fw, fh, fx, fy, rgb);
  unsigned char* d = dst + ((eye_y0 + ey) * dst_stride_w + (eye_x0 + ex)) * 3;
  for (int c = 0; c < 3; c++)
    d[c] = (unsigned char)(rgb[c] * a + d[c] * (1.f - a) + 0.5f);
}

// Production path: blend the restored flat view straight into dst (a full frame
// of row stride dst_stride_w) at the eye-local crop offset. No temporaries.
extern "C" __global__ void flat_to_equirect_blend_u8(
    const unsigned char* flat, unsigned char* dst, const float* Rt,
    float range_x, float range_y, int ew, int eh, int fw, int fh,
    int cx0, int cy0, int cw, int ch,
    int dst_stride_w, int eye_x0, int eye_y0, float feather) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cw || y >= ch) return;
  int ex = cx0 + x, ey = cy0 + y;             // eye-local equirect pixel
  float fx, fy;
  if (!project_to_flat(ex, ey, ew, eh, Rt, range_x, range_y, fw, fh, &fx, &fy)) return;
  float a = edge_alpha(fx, fy, fw, fh, feather);
  if (a <= 0.f) return;
  float rgb[3];
  sample_flat(flat, fw, fh, fx, fy, rgb);
  unsigned char* d = dst + ((eye_y0 + ey) * dst_stride_w + (eye_x0 + ex)) * 3;
  for (int c = 0; c < 3; c++)
    d[c] = (unsigned char)(rgb[c] * a + d[c] * (1.f - a) + 0.5f);
}
"""

_block = (16, 16, 1)


def _grid(w: int, h: int) -> tuple[int, int, int]:
    return ((w + 15) // 16, (h + 15) // 16, 1)


def _axis(ax: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if ax == "x":   # pitch
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)
    if ax == "y":   # yaw
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)  # z / roll


def v360_rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """flat-local -> world rotation matching ffmpeg ``v360`` (rorder=ypr).

    Locked to v360 pixel-for-pixel by the alignment probes:
    ``R = Ry(yaw) @ Rx(-pitch) @ Rz(-roll)`` -- 48.3 dB at roll=0, and 50.0 /
    47.4 dB at roll=+25 / -40 (every other sign/ordering variant scores ~15 dB).
    Both pitch and roll are negated: v360's sign for them is opposite the naive
    right-handed Rx/Rz used here."""
    return _axis("y", math.radians(yaw)) @ _axis("x", math.radians(-pitch)) \
        @ _axis("z", math.radians(-roll))


def fov_to_range(d_fov_deg: float, fw: int, fh: int) -> tuple[float, float]:
    """d_fov (diagonal) -> (tan(hfov/2), tan(vfov/2)), matching v360 fov_from_dfov."""
    da = math.tan(0.5 * math.radians(d_fov_deg))
    d = math.hypot(fw, fh)
    return da * fw / d, da * fh / d


# --------------------------------------------------------------------------------------
# REVIEW NOTE (added 2026-08-07) -- forward-projection supersampling.
#
# Problem found by reproducing this pipeline on real clips (lada/deploy_repro.py):
# equirect_to_flat_u8 samples the eye with a 2-tap bilinear kernel and NO prefilter. That is
# correct only when the flat view is magnifying or roughly 1:1. Real VR mosaic regions near
# the poles are wide equirect bands (measured on test_demosaic_1.mp4: 1046x309, 1298x312,
# 2025x314 px) that _region_view maps to a SQUARE flat view, so the source is minified
# heavily and anisotropically -- 2.7x to 5x horizontally against ~1.2x vertically, and more
# than that once the 1/cos(lat) pole stretch is accounted for.
#
# Undersampling by 5x with 2 taps discards most of the pixels that should have been averaged.
# Measured effect at frame 6952: rendering the SAME view with the SAME kernel at 4x and
# box-averaging down gives 0.05x the Laplacian variance, i.e. ~95% of the high-frequency
# content the restorer currently receives is aliasing rather than signal. It is plainly
# visible as moire swirls on patterned fabric that do not exist in the source.
#
# This matters beyond looks: the mosaic CELL LATTICE is resampled the same way, so the
# restorer is fed a beat pattern that appears in none of its training data. That is why the
# model benchmarks at +6.80 dB region gain on synthetic morphologies yet looks blurred and
# distorted on device -- the two are not the same input distribution.
#
# Fix: average the SxS taps the view would have had if rendered at S x resolution. Same
# projection math, same sampling code (sample_eye), exactly equivalent to SxS supersampling.
# S is derived from the measured source footprint (see _source_footprint) rather than guessed.
# equirect_to_flat_ss_u8 accumulates the taps in registers -- the first version of this went
# through an (fh*S, fw*S) uint8 buffer plus a cupy box-average, which cost ~0.6 ms/call in
# temporaries and a second memory pass, on top of the ~0.6 ms of HOST time _source_footprint
# takes. At 16 to_flat calls per chunk (WINDOW frames x 2 regions) that was ~14 ms/chunk of
# largely avoidable work, most of it CPU-side, which realtime felt directly. Both are now
# memoised/fused; see VrReprojector._ss_cache and _rot_g.
#
# The INVERSE direction is deliberately NOT changed: flat_to_equirect_* iterates over
# destination equirect pixels and gathers from the flat view, and the equirect region is
# LARGER than the 256 view, so that direction magnifies. Bilinear magnification does not
# alias. (It is soft, but that is a resolution limit of restoring a 1296px band through a
# 256px view -- fixable only by tiling wide bands into several views, not by filtering.)
# --------------------------------------------------------------------------------------

# 8x8 = 64 samples/px worst case, i.e. one 2048x2048 render for a 256x256 view. That is a few
# hundred microseconds of a trivially parallel kernel, against ~18 ms/frame for the restorer.
# Measured need: pole bands on test_demosaic_1.mp4 reach a footprint of ~5-10 eye px per flat
# px, and capping at 4 left ~5x more high-frequency energy than a properly filtered reference.
MAX_SUPERSAMPLE = 8


def _source_footprint(R: np.ndarray, range_x: float, range_y: float,
                      ew: int, eh: int, fw: int, fh: int) -> float:
    """How many eye pixels one flat pixel spans, typically, across the view.

    Replicates equirect_to_flat_u8's flat-pixel -> eye-pixel mapping on a coarse grid and
    finite-differences it, so the answer includes the 1/cos(lat) pole stretch instead of
    assuming an equator-like view. 1.0 means 1:1; >2 means the 2-tap kernel undersamples.

    Returns the 75th PERCENTILE, not the max and not the median.

    Not the max: these views are centred on near-pole bands and their fov often reaches over
    the pole itself, where equirect's longitude derivative is singular. The max is then set
    by one degenerate grid point and is meaningless -- 252 against a true ~5 on
    test_demosaic_1.mp4 frame 6952.

    Not the median either, although it tracks the nominal ratio well (measured 5.30/4.97/3.41
    against box-width/256 of 5.06/4.03/2.68 on three real regions). The footprint distribution
    is long-tailed (frame 6952: p50 5.3, p75 7.8, p90 12.7), and the most compressed parts of
    the view are exactly the ones that alias hardest, so they dominate the residual. Sizing on
    the median left 2.3x more high-frequency energy than a true 8x reference on that frame;
    p75 closes it. The extra cost is bounded by MAX_SUPERSAMPLE regardless.
    """
    n = 33
    gx, gy = np.meshgrid(np.linspace(0, fw - 1, n), np.linspace(0, fh - 1, n))

    def to_eye(x, y):
        lx = (2.0 * (x + 0.5) / fw - 1.0) * range_x
        ly = -(2.0 * (y + 0.5) / fh - 1.0) * range_y
        lz = np.ones_like(lx)
        inv = 1.0 / np.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx * inv, ly * inv, lz * inv
        wx = R[0, 0] * lx + R[0, 1] * ly + R[0, 2] * lz
        wy = R[1, 0] * lx + R[1, 1] * ly + R[1, 2] * lz
        wz = R[2, 0] * lx + R[2, 1] * ly + R[2, 2] * lz
        lon = np.arctan2(wx, wz)
        lat = np.arcsin(np.clip(wy, -1.0, 1.0))
        return (lon / math.pi + 0.5) * ew - 0.5, (0.5 - lat / math.pi) * eh - 0.5

    sx, sy = to_eye(gx, gy)
    sx1, sy1 = to_eye(gx + 1.0, gy)          # d/dx
    sx2, sy2 = to_eye(gx, gy + 1.0)          # d/dy
    # guard the atan2 wrap: a genuine step is never half the eye width
    dxx, dyx = np.abs(sx1 - sx), np.abs(sy1 - sy)
    dxy, dyy = np.abs(sx2 - sx), np.abs(sy2 - sy)
    ok = (dxx < ew * 0.5) & (dxy < ew * 0.5)
    if not ok.any():
        return 1.0
    # eye px moved per +1 flat px, along each axis; take the larger of the two per point
    stretch = np.maximum(np.hypot(dxx, dyx), np.hypot(dxy, dyy))[ok]
    return float(np.percentile(stretch, 75))


class VrReprojector:
    """Cached CuPy kernels for hequirect<->flat reprojection of one eye."""

    def __init__(self, cp):
        self.cp = cp
        fwd = cp.RawModule(code=_FWD_SRC)
        self._fwd = fwd.get_function("equirect_to_flat_u8")
        self._fwd_ss = fwd.get_function("equirect_to_flat_ss_u8")
        # _source_footprint is ~0.7 ms of HOST time and depends only on the view
        # geometry, while the caller runs one view over a whole 8-frame chunk (and
        # keeps the same regions across several chunks between detections). Memo it
        # so it is paid once per view instead of once per frame per region.
        self._ss_cache: dict[tuple, int] = {}
        # Same idea for the 3x3 itself: one H2D copy per call is ~0.05 ms, and a chunk
        # issues 2*WINDOW of them per region with identical angles.
        self._rot_cache: dict[tuple, "cp.ndarray"] = {}
        inv = cp.RawModule(code=_INV_SRC)
        self._inv = inv.get_function("flat_to_equirect_u8")
        self._blend = inv.get_function("flat_to_equirect_blend_u8")
        self._blend_masked = inv.get_function("flat_to_equirect_blend_masked_u8")

    def _rot_g(self, yaw: float, pitch: float, roll: float, transpose: bool):
        """Device-side flattened rotation, memoised on the angles. Read-only."""
        key = (round(yaw, 4), round(pitch, 4), round(roll, 4), transpose)
        g = self._rot_cache.get(key)
        if g is None:
            m = v360_rotation(yaw, pitch, roll)
            g = self.cp.asarray((m.T if transpose else m).ravel(), self.cp.float32)
            if len(self._rot_cache) > 512:
                self._rot_cache.clear()
            self._rot_cache[key] = g
        return g

    def to_flat(self, src_g, yaw: float, pitch: float, d_fov: float,
                fw: int, fh: int, roll: float = 0.0,
                eye_origin: tuple[int, int] = (0, 0), eye_size=None, out=None,
                supersample: int = 0):
        """Project one hequirect eye to a flat view.

        ``src_g`` is a contiguous (H,W,3) uint8 frame; ``eye_origin``/``eye_size``
        address the eye inside it, so a SBS frame needs no per-eye copy.
        Writes into ``out`` when given (e.g. a slice of the restorer batch).

        ``supersample``: 0 = auto (pick S from the measured source footprint), 1 = off
        (the old behaviour, kept so the change can be A/B'd), >1 = force that factor.
        See the REVIEW NOTE above _source_footprint for why this exists."""
        cp = self.cp
        src_g = cp.ascontiguousarray(src_g)      # no-op for a whole frame
        H, W = int(src_g.shape[0]), int(src_g.shape[1])
        ex0, ey0 = eye_origin
        ew, eh = eye_size if eye_size is not None else (W, H)
        R = self._rot_g(yaw, pitch, roll, False)
        rx, ry = fov_to_range(d_fov, fw, fh)

        if supersample <= 0:
            # One flat pixel spanning F eye pixels needs ~F samples across to be filtered
            # correctly. Round up, clamp: below 1.5 the 2-tap kernel is already adequate.
            # The key is quantised to half a degree: S is a rounded-up integer of a
            # quantity that varies smoothly with the angles, so half a degree can only
            # ever move it by one step, and this is what makes the memo hit for callers
            # whose view drifts every frame (face beauty tracks a moving face) rather
            # than only for the RM path, whose view is fixed for a whole chunk.
            key = (round(yaw * 2), round(pitch * 2), round(roll * 2), round(d_fov * 2),
                   ew, eh, fw, fh)
            s = self._ss_cache.get(key)
            if s is None:
                f = _source_footprint(v360_rotation(yaw, pitch, roll),
                                      rx, ry, ew, eh, fw, fh)
                s = 1 if f < 1.5 else min(MAX_SUPERSAMPLE, int(math.ceil(f)))
                if len(self._ss_cache) > 512:      # regions come and go; keep it bounded
                    self._ss_cache.clear()
                self._ss_cache[key] = s
        else:
            s = min(MAX_SUPERSAMPLE, int(supersample))

        flat_g = cp.empty((fh, fw, 3), cp.uint8) if out is None else out
        if s <= 1:
            self._fwd(_grid(fw, fh), _block,
                      (src_g, flat_g, R, np.float32(rx), np.float32(ry),
                       np.int32(ew), np.int32(eh), np.int32(fw), np.int32(fh),
                       np.int32(W), np.int32(ex0), np.int32(ey0)))
            return flat_g

        # Sample the SAME view as if rendered at s x resolution and box-averaged down.
        # rx/ry depend only on the aspect ratio (fov_to_range normalises by hypot(fw,fh)),
        # so scaling both sides by s leaves the view identical -- this is exactly sxs
        # supersampling of it, accumulated in registers by equirect_to_flat_ss_u8.
        self._fwd_ss(_grid(fw, fh), _block,
                     (src_g, flat_g, R, np.float32(rx), np.float32(ry),
                      np.int32(ew), np.int32(eh), np.int32(fw), np.int32(fh),
                      np.int32(W), np.int32(ex0), np.int32(ey0), np.int32(s)))
        return flat_g

    def to_equirect(self, flat_g, yaw: float, pitch: float, d_fov: float,
                    ew: int, eh: int, crop=None, roll: float = 0.0,
                    feather: float = 0.0):
        """Reproject to eye space, returning (patch, alpha) over ``crop``
        (eye-local x_off,y_off,cw,ch). Validation/debug helper; the production
        path uses :meth:`blend_into`."""
        cp = self.cp
        flat_g = cp.ascontiguousarray(flat_g)
        fh, fw = int(flat_g.shape[0]), int(flat_g.shape[1])
        cx0, cy0, cw, ch = (0, 0, ew, eh) if crop is None else crop
        Rt = self._rot_g(yaw, pitch, roll, True)
        rx, ry = fov_to_range(d_fov, fw, fh)
        patch_g = cp.empty((ch, cw, 3), cp.uint8)
        alpha_g = cp.empty((ch, cw), cp.uint8)
        self._inv(_grid(cw, ch), _block,
                  (flat_g, patch_g, alpha_g, Rt, np.float32(rx), np.float32(ry),
                   np.int32(ew), np.int32(eh), np.int32(fw), np.int32(fh),
                   np.int32(cx0), np.int32(cy0), np.int32(cw), np.int32(ch),
                   np.float32(feather)))
        return patch_g, alpha_g

    def blend_into(self, flat_g, dst_g, yaw: float, pitch: float, d_fov: float,
                   ew: int, eh: int, crop, eye_origin: tuple[int, int] = (0, 0),
                   roll: float = 0.0, feather: float = 0.0) -> None:
        """Alpha-composite the restored flat view into ``dst_g`` in place.

        ``dst_g`` is a contiguous (H,W,3) uint8 frame; ``crop`` is the eye-local
        window (x_off,y_off,cw,ch) and ``eye_origin`` places the eye in the frame.
        Allocates nothing per call beyond the 3x3 rotation."""
        cp = self.cp
        flat_g = cp.ascontiguousarray(flat_g)
        # dst is written in place, so it must already be contiguous (copying it
        # here would silently discard the composite).
        if not dst_g.flags.c_contiguous:
            raise ValueError("blend_into requires a C-contiguous destination frame")
        fh, fw = int(flat_g.shape[0]), int(flat_g.shape[1])
        cx0, cy0, cw, ch = crop
        ex0, ey0 = eye_origin
        Rt = self._rot_g(yaw, pitch, roll, True)
        rx, ry = fov_to_range(d_fov, fw, fh)
        self._blend(_grid(cw, ch), _block,
                    (flat_g, dst_g, Rt, np.float32(rx), np.float32(ry),
                     np.int32(ew), np.int32(eh), np.int32(fw), np.int32(fh),
                     np.int32(cx0), np.int32(cy0), np.int32(cw), np.int32(ch),
                     np.int32(int(dst_g.shape[1])), np.int32(ex0), np.int32(ey0),
                     np.float32(feather)))

    def blend_into_masked(self, flat_g, alpha_g, dst_g, yaw: float, pitch: float, d_fov: float,
                          ew: int, eh: int, crop, eye_origin: tuple[int, int] = (0, 0),
                          roll: float = 0.0, feather: float = 0.0) -> None:
        """:meth:`blend_into` gated by a flat-space alpha plane.

        ``alpha_g`` is float32 (fh,fw) in 0..1 and is multiplied into the edge
        feather, so pixels the caller left at 0 are not written at all. Used by
        face beauty, where only the face mask should come back and the untouched
        surroundings must not be re-resampled."""
        cp = self.cp
        flat_g = cp.ascontiguousarray(flat_g)
        alpha_g = cp.ascontiguousarray(alpha_g)
        if not dst_g.flags.c_contiguous:
            raise ValueError("blend_into_masked requires a C-contiguous destination frame")
        fh, fw = int(flat_g.shape[0]), int(flat_g.shape[1])
        cx0, cy0, cw, ch = crop
        ex0, ey0 = eye_origin
        Rt = self._rot_g(yaw, pitch, roll, True)
        rx, ry = fov_to_range(d_fov, fw, fh)
        self._blend_masked(_grid(cw, ch), _block,
                           (flat_g, alpha_g, dst_g, Rt, np.float32(rx), np.float32(ry),
                            np.int32(ew), np.int32(eh), np.int32(fw), np.int32(fh),
                            np.int32(cx0), np.int32(cy0), np.int32(cw), np.int32(ch),
                            np.int32(int(dst_g.shape[1])), np.int32(ex0), np.int32(ey0),
                            np.float32(feather)))


def project_points_to_flat(points, ew: int, eh: int, yaw: float, pitch: float,
                           d_fov: float, fw: int, fh: int, roll: float = 0.0):
    """Host-side twin of the kernel's ``project_to_flat``, for a handful of points.

    Face beauty already has the detector's 5 keypoints in eye-local equirect
    coordinates; projecting them directly is exact and avoids re-running the
    detector inside the flat view. Returns (points_xy, valid_mask)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    u = (pts[:, 0] + 0.5) / ew
    v = (pts[:, 1] + 0.5) / eh
    lon = (u - 0.5) * math.pi
    lat = (0.5 - v) * math.pi
    cl = np.cos(lat)
    world = np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)
    local = world @ v360_rotation(yaw, pitch, roll)      # (R^T . w) == w . R
    rx, ry = fov_to_range(d_fov, fw, fh)
    lz = local[:, 2]
    valid = lz > 1e-6
    safe = np.where(valid, lz, 1.0)
    fx = (local[:, 0] / safe / rx * 0.5 + 0.5) * fw - 0.5
    fy = (-local[:, 1] / safe / ry * 0.5 + 0.5) * fh - 0.5
    return np.stack([fx, fy], axis=-1).astype(np.float32), valid
