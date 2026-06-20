"""CuPy GPU renderer for the offline 2D->3D/VR fast path (flat3d inverse_warp).

The CPU renderer (offline.two_dvr_render.StereoRenderer) is fine at <=1080p but
the two full-res cv2.remap calls dominate at 4K. This moves the whole stereo
inverse-warp into a single CuPy RawKernel that produces the SBS frame directly
from the source frame + the (low-res) DA3 depth, bilinearly upscaling the
disparity on-device.

Only RawKernel/ElementwiseKernel are used -- no cccl/thrust ops (percentile,
sort, ndimage) -- because CuPy 14.0.1's libcudacxx headers fail to JIT those.
The 5/95 percentile normalization stays on the CPU at depth resolution (~150k
px, sub-millisecond). Falls back to None if CuPy is unavailable or the GPU
cannot JIT, so the caller can use the CPU renderer.
"""
from __future__ import annotations

import os

# sm_120/Blackwell: CuPy must emit native cubins via NVRTC >= 12.8 (the uv venv
# ships pip NVRTC 12.9). CuPy reads CUPY_COMPILE_WITH_PTX once, into
# compiler._use_ptx, when cupy.cuda.compiler is first imported -- so a stale
# `CUPY_COMPILE_WITH_PTX=1` left in the shell forces the slow PTX->driver-JIT
# path (a fresh RawModule then "hangs" 60-120s). The standalone offline.two_dvr
# entry may import this module before main.py calls configure_gpu_runtime_cache(),
# so pin the project CUDA/CuPy cache env here before this module's lazy
# `import cupy`. This also avoids the default user CuPy cache if it contains a
# stale lock/corrupt entry that can make RawModule compilation look hung.
try:
    from utils.gpu_runtime_cache import configure_gpu_runtime_cache

    configure_gpu_runtime_cache()
except Exception:
    os.environ["CUPY_COMPILE_WITH_PTX"] = "0"

import cv2
import numpy as np

from offline.two_dvr_render import (
    _BASE_LOWPASS_DIV,
    _EVID_R_HI,
    _EVID_R_LO,
    _EVID_TILE,
    _MC_DOWNSAMPLE,
    _MC_MAX_SHIFT_PX,
    _MC_MAX_WORK,
    _MC_MIN_RESPONSE,
    _WIN_MAX_RADIUS,
    DEFAULT_EYE_DISTANCE_MM,
    DEFAULT_FLAT_FOV_DEG,
    DEFAULT_HOLE_FILL_MODE,
    DEFAULT_TEMPORAL_DEPTH,
    DEFAULT_TEMPORAL_DEPTH_ALPHA,
    DEFAULT_TEMPORAL_DEPTH_MODE,
    DEFAULT_TEMPORAL_AFFINE,
    DEFAULT_TEMPORAL_AFFINE_MAX_BIAS,
    DEFAULT_TEMPORAL_AFFINE_MAX_SCALE,
    DEFAULT_TEMPORAL_FLOW_CONSISTENCY,
    DEFAULT_TEMPORAL_FLOW_DIFF,
    DEFAULT_TEMPORAL_FLOW_MOTION_GATE,
    DEFAULT_TEMPORAL_MOTION_MAX_STEP_PX,
    DEFAULT_TEMPORAL_NORM,
    DEFAULT_TEMPORAL_NORM_ALPHA,
    DEFAULT_TEMPORAL_NORM_RESET,
    DEFAULT_TEMPORAL_STATIC_DEADBAND_PX,
    DEFAULT_TEMPORAL_STATIC_MAX_STEP_PX,
    HOLE_FILL_INVERSE_WARP,
    HOLE_FILL_SOFT_SHIFT,
    PROJECTION_FLAT_3D,
    TemporalDepthStabilizer,
    _flat_vr_eye_size,
    _max_disparity_pixels,
    make_projection_map,
    near_from_depth,
    near_for_render,
)


def _two_dvr_rim_width(src_w: int) -> int:
    """Background-side non-hole rim cleanup width.

    Disocclusion holes can have a thin non-hole seam on the background side:
    low-near pixels were written by soft_shift, so pure zbuf==0 hybrid filling
    does not touch them. The kernel only replaces low-near pixels when a hole
    lies in the eye-specific background-side direction, which keeps foreground
    silhouettes protected. Set PT_TWO_DVR_RIM=0 to disable for diagnostics.
    """
    raw = os.environ.get("PT_TWO_DVR_RIM", "").strip()
    if not raw:
        return max(2, round(int(src_w) / 120))
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _two_dvr_fg_bad_width(src_w: int) -> int:
    """Window (full-res px) for the embedded-background cleanup (fw_fg_bad_local).

    soft_shift leaves background-coloured slivers/cracks INSIDE the body -- white
    vertical stripes through arms/torso, both zbuf==0 cracks and low-near written
    pixels. fw_fg_bad_local replaces such a pixel with the nearest foreground
    colour only when foreground encloses it on BOTH horizontal sides within this
    window, so true background (foreground on at most one side) is never touched.

    Defaults to an auto-scaled window (`1920w -> 8`) because the visible hair
    edge failure is common in normal soft_shift output. Set PT_TWO_DVR_FG_BAD to
    a full-res pixel window to override; a non-positive or unparseable value
    disables it.
    """
    raw = os.environ.get("PT_TWO_DVR_FG_BAD", "").strip()
    if not raw:
        return max(2, round(int(src_w) / 240))
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


_SBS_INV_WARP_KERNEL = r'''
extern "C" __global__ void sbs_inv_warp(
    const unsigned char* frame,   // (H, W, 3)
    const float* nearmap,         // (h, w), normalized 0..1
    unsigned char* out,           // (H, 2W, 3)
    int H, int W, int h, int w, float max_shift)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * 2 * W;
    if (idx >= total) return;
    int twoW = 2 * W;
    int px = (int)(idx % twoW);
    int y  = (int)(idx / twoW);
    int ex = px % W;            // source x within the eye
    int eye = px / W;           // 0 = left, 1 = right
    float sign = eye == 0 ? -1.0f : 1.0f;

    // Bilinear-upsample the low-res near map to this full-res pixel.
    float fy = (H > 1) ? (float)y * (float)(h - 1) / (float)(H - 1) : 0.0f;
    float fx = (W > 1) ? (float)ex * (float)(w - 1) / (float)(W - 1) : 0.0f;
    int y0 = (int)fy, x0 = (int)fx;
    int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
    float wy = fy - y0, wx = fx - x0;
    float n = nearmap[(long)y0 * w + x0] * (1 - wx) * (1 - wy)
            + nearmap[(long)y0 * w + x1] * wx * (1 - wy)
            + nearmap[(long)y1 * w + x0] * (1 - wx) * wy
            + nearmap[(long)y1 * w + x1] * wx * wy;

    float sx = (float)ex + sign * (n * max_shift) * 0.5f;
    if (sx < 0.0f) sx = 0.0f;
    if (sx > W - 1) sx = (float)(W - 1);
    int sx0 = (int)sx, sx1 = min(sx0 + 1, W - 1);
    float sw = sx - sx0;

    long o = ((long)y * twoW + px) * 3;
    long base = (long)y * W * 3;
    for (int c = 0; c < 3; c++) {
        float a = frame[base + (long)sx0 * 3 + c];
        float b = frame[base + (long)sx1 * 3 + c];
        out[o + c] = (unsigned char)(a * (1 - sw) + b * sw + 0.5f);
    }
}
'''

# Combined stereo-warp + VR projection (fisheye / hequirect 180). For each VR
# output pixel, the projection map gives the source flat coordinate; we then
# shift it by the per-eye disparity and bilinearly sample the source frame.
# Equivalent to the CPU "warp the flat frame, then project each eye" path.
_SBS_PROJECT_WARP_KERNEL = r'''
extern "C" __global__ void sbs_project_warp(
    const unsigned char* frame,   // (H, W, 3)
    const float* nearmap,         // (h, w) normalized 0..1
    const float* mapx,            // (side, side) source x
    const float* mapy,            // (side, side) source y
    const unsigned char* mask,    // (side, side) 1=valid
    unsigned char* out,           // (side, 2*side, 3)
    int H, int W, int h, int w, int side, float max_shift)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)side * 2 * side;
    if (idx >= total) return;
    int twoS = 2 * side;
    int px = (int)(idx % twoS);
    int oy = (int)(idx / twoS);
    int eye = px / side;          // 0 left, 1 right
    int ex = px % side;
    long mi = (long)oy * side + ex;
    long o = ((long)oy * twoS + px) * 3;
    if (!mask[mi]) { out[o]=0; out[o+1]=0; out[o+2]=0; return; }
    float mx = mapx[mi], my = mapy[mi];   // source flat coordinate
    float sign = eye == 0 ? -1.0f : 1.0f;

    float fy = (H > 1) ? my * (float)(h - 1) / (float)(H - 1) : 0.0f;
    float fx = (W > 1) ? mx * (float)(w - 1) / (float)(W - 1) : 0.0f;
    int ny0=(int)fy, nx0=(int)fx; int ny1=min(ny0+1,h-1), nx1=min(nx0+1,w-1);
    float wy=fy-ny0, wx=fx-nx0;
    float n = nearmap[(long)ny0*w+nx0]*(1-wx)*(1-wy)+nearmap[(long)ny0*w+nx1]*wx*(1-wy)
            + nearmap[(long)ny1*w+nx0]*(1-wx)*wy+nearmap[(long)ny1*w+nx1]*wx*wy;

    float sx = mx + sign * (n * max_shift) * 0.5f;
    if (sx < 0.0f) sx = 0.0f; if (sx > W-1) sx = (float)(W-1);
    float sy = my; if (sy < 0.0f) sy = 0.0f; if (sy > H-1) sy = (float)(H-1);
    int sx0=(int)sx, sx1=min(sx0+1,W-1); int sy0=(int)sy, sy1=min(sy0+1,H-1);
    float swx=sx-sx0, swy=sy-sy0;
    for (int c = 0; c < 3; c++) {
        float a=frame[((long)sy0*W+sx0)*3+c], b=frame[((long)sy0*W+sx1)*3+c];
        float cc=frame[((long)sy1*W+sx0)*3+c], d=frame[((long)sy1*W+sx1)*3+c];
        float top=a*(1-swx)+b*swx, bot=cc*(1-swx)+d*swx;
        out[o+c]=(unsigned char)(top*(1-swy)+bot*swy+0.5f);
    }
}
'''

# soft_shift (forward warp + z-buffer + hole fill). Produces a flat side-by-side
# (H, 2W) buffer: flat3d uses it directly; VR projects it. Three passes -- z-buffer
# scatter (atomicMax on encoded near), winner color write, horizontal hole fill --
# then an optional projection sample. Faithful to the CPU forward-warp path.
_SOFT_SHIFT_KERNELS = r'''
__device__ __forceinline__ float _near_at(const float* nm, int h, int w, int H, int W, int y, int x) {
    float fy = (H > 1) ? (float)y * (float)(h - 1) / (float)(H - 1) : 0.0f;
    float fx = (W > 1) ? (float)x * (float)(w - 1) / (float)(W - 1) : 0.0f;
    int y0=(int)fy, x0=(int)fx; int y1=min(y0+1,h-1), x1=min(x0+1,w-1);
    float wy=fy-y0, wx=fx-x0;
    float n = nm[(long)y0*w+x0]*(1-wx)*(1-wy)+nm[(long)y0*w+x1]*wx*(1-wy)
            + nm[(long)y1*w+x0]*(1-wx)*wy+nm[(long)y1*w+x1]*wx*wy;
    // Toggle (morphological-contrast) sharpen: snap to the nearer of the local
    // horizontal min/max so a soft depth contour becomes a hard edge. DA3 depth
    // ramps over several px at object boundaries; left soft, the forward warp
    // maps those intermediate disparities to scattered foreground slivers inside
    // the disocclusion gap, and the hole-fill then smears foreground into it
    // (faces/limbs visibly fattened/stretched). Snapping collapses the gap to a
    // clean hole bounded by solid fg/bg so it fills from true background. Window
    // approximates +/-6 full-res px; vertical stays bilinear (contours ~vertical).
    float s = (W > 1) ? (float)(w - 1) / (float)(W - 1) : 1.0f;
    int win = max(1, (int)ceilf(6.0f * s));
    float lo = n, hi = n;
    for (int dx = -win; dx <= win; ++dx) {
        int xx = min(max(x0 + dx, 0), w - 1);
        float a = nm[(long)y0*w + xx], b = nm[(long)y1*w + xx];
        lo = fminf(lo, fminf(a, b)); hi = fmaxf(hi, fmaxf(a, b));
    }
    return ((n - lo) >= (hi - n)) ? hi : lo;
}

extern "C" __global__ void fw_zbuf(
    const float* nearmap, int* zbuf, int H, int W, int h, int w, float max_shift)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)H*W) return;
    int x = (int)(idx % W), y = (int)(idx / W);
    float n = _near_at(nearmap, h, w, H, W, y, x);
    int pr = (int)(n * 1000000.0f) + 1;
    int W2 = 2 * W;
    for (int eye = 0; eye < 2; eye++) {
        float sign = eye == 0 ? 1.0f : -1.0f;   // matches CPU forward-warp eye_sign
        int tx = (int)lroundf((float)x + n * (max_shift * 0.5f) * sign);
        if (tx >= 0 && tx < W) atomicMax(&zbuf[(long)y*W2 + eye*W + tx], pr);
    }
}

extern "C" __global__ void fw_color(
    const unsigned char* frame, const float* nearmap, const int* zbuf,
    unsigned char* out, int H, int W, int h, int w, float max_shift)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)H*W) return;
    int x = (int)(idx % W), y = (int)(idx / W);
    float n = _near_at(nearmap, h, w, H, W, y, x);
    int pr = (int)(n * 1000000.0f) + 1;
    int W2 = 2 * W;
    long si = ((long)y*W + x) * 3;
    for (int eye = 0; eye < 2; eye++) {
        float sign = eye == 0 ? 1.0f : -1.0f;
        int tx = (int)lroundf((float)x + n * (max_shift * 0.5f) * sign);
        if (tx >= 0 && tx < W && zbuf[(long)y*W2 + eye*W + tx] == pr) {
            long o = ((long)y*W2 + eye*W + tx) * 3;
            out[o]=frame[si]; out[o+1]=frame[si+1]; out[o+2]=frame[si+2];
        }
    }
}

// Hybrid hole fill: replace soft_shift disocclusion holes (zbuf==0) with the
// inverse_warp result for the same pixel. inverse_warp is a hole-free backward
// sample, so it avoids soft_shift's directional row-copy (which drags wall
// texture / hair into a horizontal band across a wide gap). Non-hole pixels are
// left untouched (soft_shift keeps the correct occlusion + hard silhouette).
extern "C" __global__ void fw_hole_from_inv(
    unsigned char* out, const int* zbuf, const unsigned char* inv, int H, int W, int rim)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int W2 = 2 * W;
    if (idx >= (long)H*W2) return;
    int ox = (int)(idx % W2), y = (int)(idx / W2);
    long o = ((long)y*W2 + ox) * 3;
    int zb = zbuf[(long)y*W2 + ox];
    if (zb == 0) { out[o]=inv[o]; out[o+1]=inv[o+1]; out[o+2]=inv[o+2]; return; }  // hole
    // Background-side rim cleanup: matting/depth leaves object-coloured pixels
    // (hair/cloth) at the foreground's original edge with LOW depth, so they
    // don't shift and form a contaminated rim on the BACKGROUND edge of the gap.
    // Replace those with inverse too -- but only on the eye's background side
    // (left eye = left of the gap, right eye = right) and only if the pixel is
    // background (low near); never foreground (protects the silhouette and thin
    // foreground strips between narrow gaps). near = (zbuf-1)/1e6.
    if (rim <= 0) return;
    if ((float)(zb - 1) * 1e-6f >= 0.5f) return;   // foreground -> keep soft
    int eye = ox / W, lo = eye*W, hi = lo + W;
    int dir = (eye == 0) ? 1 : -1;                 // scan toward the gap
    for (int s = 1; s <= rim; ++s) {
        int nx = ox + dir * s;
        if (nx < lo || nx >= hi) break;
        if (zbuf[(long)y*W2 + nx] == 0) {          // a gap lies on the foreground side
            out[o]=inv[o]; out[o+1]=inv[o+1]; out[o+2]=inv[o+2];
            return;
        }
    }
}

// Fill disocclusion holes (zbuf==0) from the nearest written pixel within the
// same eye. The two eyes warp in opposite directions (left eye shifts the
// foreground right, right eye shifts it left), so the disocclusion gap opens on
// the opposite side for each eye and must be filled from the opposite side too:
// left eye fills from its left (background) neighbour, right eye from its right.
// This mirrors the CPU _shift_fill_holes_rgb direction (-1 for the left eye, +1
// for the right) -- using one shared "prefer smaller near" rule for both eyes
// pulls the foreground into the gap on one eye (stretched) and clips it on the
// other.
extern "C" __global__ void fw_fill(
    unsigned char* out, const int* zbuf, const float* nearmap,
    int H, int W, int h, int w)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int W2 = 2 * W;
    if (idx >= (long)H*W2) return;
    int ox = (int)(idx % W2), y = (int)(idx / W2);
    if (zbuf[(long)y*W2 + ox] != 0) return;
    int eye = ox / W; int lo = eye*W, hi = lo + W;
    int li = -1, ri = -1;
    for (int j = ox-1; j >= lo; --j) { if (zbuf[(long)y*W2+j]) { li = j; break; } }
    for (int j = ox+1; j <  hi; ++j) { if (zbuf[(long)y*W2+j]) { ri = j; break; } }
    int pick;
    if (li < 0 && ri < 0) return;
    else if (li < 0) pick = ri;
    else if (ri < 0) pick = li;
    else pick = (eye == 0) ? li : ri;  // left eye -> left side, right eye -> right side
    long o = ((long)y*W2 + ox) * 3, p = ((long)y*W2 + pick) * 3;
    out[o]=out[p]; out[o+1]=out[p+1]; out[o+2]=out[p+2];
}

// Soft-blend ONLY the filled disocclusion holes (zbuf==0), softening the seam
// where the stretched background fill meets the real background. The foreground
// silhouette must stay hard: the blur excludes foreground neighbours (the
// occluder), and foreground pixels themselves are passed through untouched. The
// foreground is identified per-pixel from the local near range encoded in zbuf
// (priority = near*1e6+1); when the window straddles a depth step we drop the
// near half. Reads `flat`, writes `out` (both H x 2W).
extern "C" __global__ void fw_blend(
    const unsigned char* flat, const int* zbuf, unsigned char* out, int H, int W)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int W2 = 2 * W;
    if (idx >= (long)H*W2) return;
    int ox = (int)(idx % W2), y = (int)(idx / W2);
    long o = ((long)y*W2 + ox) * 3;
    // Only filled holes are softened; real (written) pixels pass through so the
    // foreground silhouette and the untouched background stay crisp.
    if (zbuf[(long)y*W2 + ox] != 0) { out[o]=flat[o]; out[o+1]=flat[o+1]; out[o+2]=flat[o+2]; return; }
    int eye = ox / W, lo = eye*W, hi = lo + W;
    const int K = 3, V = 2;
    // Near range over written neighbours -> threshold to exclude the occluder.
    float nmin = 1e9f, nmax = -1e9f;
    for (int dy=-V; dy<=V; ++dy) { int ny=min(max(y+dy,0),H-1);
        for (int dx=-K; dx<=K; ++dx) { int nx=ox+dx; if (nx<lo||nx>=hi) continue;
            int zb=zbuf[(long)ny*W2+nx]; if (zb!=0) { float nr=(float)(zb-1)*1e-6f;
                nmin=fminf(nmin,nr); nmax=fmaxf(nmax,nr); } } }
    float thr = (nmax - nmin > 0.30f) ? 0.5f*(nmin+nmax) : 1e9f;  // gate only across a depth step
    for (int c=0; c<3; ++c) {
        float s=0.f; int n=0;
        for (int dy=-V; dy<=V; ++dy) { int ny=min(max(y+dy,0),H-1);
            for (int dx=-K; dx<=K; ++dx) { int nx=ox+dx; if (nx<lo||nx>=hi) continue;
                int zb=zbuf[(long)ny*W2+nx];
                if (zb!=0 && (float)(zb-1)*1e-6f > thr) continue;  // skip foreground
                s += flat[((long)ny*W2+nx)*3+c]; ++n; } }
        float blur = n>0 ? s/n : (float)flat[o+c];
        out[o+c] = (unsigned char)(flat[o+c]*0.65f + blur*0.35f + 0.5f);
    }
}

// Clean up background contamination embedded INSIDE the foreground body. The
// soft_shift forward warp leaves two kinds of bad pixels inside a person, both
// of which read as bright/white vertical slivers cutting through an arm/torso:
//   (a) zbuf==0 cracks -- foreground pixels that no source disparity warped onto
//       (later filled from inverse_warp, which can be background-coloured), and
//   (b) zbuf!=0 but LOW-near pixels -- background colour that warped into the
//       body region.
// A pixel is "bad" when it is background-ish (a hole, or near < FG_THR) yet is
// locally enclosed on BOTH horizontal sides, within `win` px of the same eye, by
// high-near foreground written pixels. Such a pixel is a sliver embedded in the
// body, so replace it with the nearest enclosing foreground colour. True
// background has foreground on at most ONE side (the silhouette), so it is never
// touched -- this is the targeted, silhouette-safe version of the rim cleanup.
// Foreground pixels are never bad, so they are never overwritten, which makes the
// in-place RGB copy free of read-after-write hazards. Runs on the blended SBS
// (`img`, H x 2W) using the flat zbuf for the near classification.
extern "C" __global__ void fw_fg_bad_local(
    unsigned char* img, const int* zbuf, int H, int W, int win)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int W2 = 2 * W;
    if (idx >= (long)H*W2) return;
    int ox = (int)(idx % W2), y = (int)(idx / W2);
    const float FG_THR = 0.5f;
    int zb = zbuf[(long)y*W2 + ox];
    if (zb != 0 && (float)(zb - 1) * 1e-6f >= FG_THR) return;  // foreground -> keep
    int eye = ox / W, lo = eye*W, hi = lo + W;
    int lfg = -1;
    for (int s = 1; s <= win; ++s) {
        int nx = ox - s; if (nx < lo) break;
        int z = zbuf[(long)y*W2 + nx];
        if (z != 0 && (float)(z - 1) * 1e-6f >= FG_THR) { lfg = nx; break; }
    }
    if (lfg < 0) return;                       // open to background on the left
    int rfg = -1;
    for (int s = 1; s <= win; ++s) {
        int nx = ox + s; if (nx >= hi) break;
        int z = zbuf[(long)y*W2 + nx];
        if (z != 0 && (float)(z - 1) * 1e-6f >= FG_THR) { rfg = nx; break; }
    }
    if (rfg < 0) return;                       // open to background on the right
    int pick = (ox - lfg <= rfg - ox) ? lfg : rfg;   // nearest foreground source
    long o = ((long)y*W2 + ox) * 3, p = ((long)y*W2 + pick) * 3;
    img[o]=img[p]; img[o+1]=img[p+1]; img[o+2]=img[p+2];
}

// Project the flat (H, 2W) SBS into a VR (side, 2*side) SBS via the projection map.
extern "C" __global__ void project_flat_lr(
    const unsigned char* flat, const float* mapx, const float* mapy,
    const unsigned char* mask, unsigned char* out, int H, int W, int side)
{
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long total = (long)side*2*side;
    if (idx >= total) return;
    int twoS = 2*side; int px=(int)(idx % twoS), oy=(int)(idx / twoS);
    int eye = px / side, ex = px % side;
    long mi = (long)oy*side + ex; long o = ((long)oy*twoS + px)*3;
    if (!mask[mi]) { out[o]=0; out[o+1]=0; out[o+2]=0; return; }
    float mx = mapx[mi], my = mapy[mi];
    if (mx<0) mx=0; if (mx>W-1) mx=(float)(W-1); if (my<0) my=0; if (my>H-1) my=(float)(H-1);
    int W2 = 2*W; int base = eye*W;
    int sx0=(int)mx, sx1=min(sx0+1,W-1), sy0=(int)my, sy1=min(sy0+1,H-1);
    float wx=mx-sx0, wy=my-sy0;
    for (int c=0;c<3;c++){
        float a=flat[((long)sy0*W2 + base+sx0)*3+c], b=flat[((long)sy0*W2 + base+sx1)*3+c];
        float cc=flat[((long)sy1*W2 + base+sx0)*3+c], d=flat[((long)sy1*W2 + base+sx1)*3+c];
        float top=a*(1-wx)+b*wx, bot=cc*(1-wx)+d*wx;
        out[o+c]=(unsigned char)(top*(1-wy)+bot*wy+0.5f);
    }
}
'''

_GPU_NEAR_KERNELS = r'''
extern "C" __global__ void depth_smooth3x3(
    const float* depth, float* out, int H, int W)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * W;
    if (idx >= total) return;
    int x = (int)(idx % W), y = (int)(idx / W);
    float acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = min(max(y + dy, 0), H - 1);
        int wy = (dy == 0) ? 2 : 1;
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = min(max(x + dx, 0), W - 1);
            int wx = (dx == 0) ? 2 : 1;
            acc += depth[(long)yy * W + xx] * (float)(wx * wy);
        }
    }
    out[idx] = acc * 0.0625f;
}

extern "C" __global__ void inv_minmax_blocks(
    const float* depth, float* mins, float* maxs, int n)
{
    extern __shared__ float s[];
    float* smin = s;
    float* smax = s + blockDim.x;
    float mn = 3.402823466e+38f;
    float mx = -3.402823466e+38f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        float d = depth[i];
        if (isfinite(d)) {
            float inv = 1.0f / fmaxf(d, 1.0e-6f);
            mn = fminf(mn, inv);
            mx = fmaxf(mx, inv);
        }
    }
    smin[threadIdx.x] = mn;
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x + stride]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mins[blockIdx.x] = smin[0];
        maxs[blockIdx.x] = smax[0];
    }
}

extern "C" __global__ void reduce_minmax(
    const float* mins, const float* maxs, float* minmax, int blocks)
{
    extern __shared__ float s[];
    float* smin = s;
    float* smax = s + blockDim.x;
    float mn = 3.402823466e+38f;
    float mx = -3.402823466e+38f;
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        mn = fminf(mn, mins[i]);
        mx = fmaxf(mx, maxs[i]);
    }
    smin[threadIdx.x] = mn;
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x + stride]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        minmax[0] = smin[0];
        minmax[1] = smax[0];
    }
}

extern "C" __global__ void inv_histogram(
    const float* depth, const float* minmax, unsigned int* hist, int n, int bins)
{
    float mn = minmax[0];
    float mx = minmax[1];
    float span = mx - mn;
    if (!isfinite(mn) || !isfinite(mx) || span <= 1.0e-12f) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        float d = depth[i];
        if (!isfinite(d)) continue;
        float inv = 1.0f / fmaxf(d, 1.0e-6f);
        int bin = (int)(((inv - mn) / span) * (float)(bins - 1));
        bin = min(max(bin, 0), bins - 1);
        atomicAdd(&hist[bin], 1u);
    }
}

extern "C" __global__ void percentile_norm_update(
    const unsigned int* hist,
    const float* minmax,
    float* band,
    int* flags,
    int bins,
    int total,
    float alpha,
    float reset_threshold,
    int norm_enabled)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    flags[0] = 0;  // reset/initialize depth history
    flags[1] = 0;  // valid band
    float mn = minmax[0];
    float mx = minmax[1];
    float span = mx - mn;
    if (!isfinite(mn) || !isfinite(mx) || span <= 1.0e-12f || total <= 0) {
        band[0] = 0.0f;
        band[1] = 1.0f;
        band[2] = 0.0f;
        flags[0] = 1;
        return;
    }
    unsigned int target_lo = (unsigned int)floorf((float)total * 0.05f);
    unsigned int target_hi = (unsigned int)floorf((float)total * 0.95f);
    unsigned int acc = 0u;
    int lo_bin = 0, hi_bin = bins - 1;
    for (int i = 0; i < bins; ++i) {
        acc += hist[i];
        if (acc >= target_lo) {
            lo_bin = i;
            break;
        }
    }
    acc = 0u;
    for (int i = 0; i < bins; ++i) {
        acc += hist[i];
        if (acc >= target_hi) {
            hi_bin = i;
            break;
        }
    }
    float lo_raw = mn + ((float)lo_bin + 0.5f) * span / (float)bins;
    float hi_raw = mn + ((float)hi_bin + 0.5f) * span / (float)bins;
    if (!isfinite(lo_raw) || !isfinite(hi_raw) || hi_raw <= lo_raw) {
        band[0] = 0.0f;
        band[1] = 1.0f;
        band[2] = 0.0f;
        flags[0] = 1;
        return;
    }
    flags[1] = 1;
    if (!norm_enabled) {
        band[0] = lo_raw;
        band[1] = hi_raw;
        return;
    }
    if (band[2] < 0.5f) {
        band[0] = lo_raw;
        band[1] = hi_raw;
        band[2] = 1.0f;
        flags[0] = 1;
        return;
    }
    float prev_lo = band[0];
    float prev_hi = band[1];
    float prev_span = fmaxf(prev_hi - prev_lo, 1.0e-6f);
    float jump = fmaxf(fabsf(lo_raw - prev_lo), fabsf(hi_raw - prev_hi)) / prev_span;
    if (reset_threshold > 0.0f && jump >= reset_threshold) {
        band[0] = lo_raw;
        band[1] = hi_raw;
        flags[0] = 1;
        return;
    }
    band[0] = (1.0f - alpha) * prev_lo + alpha * lo_raw;
    band[1] = (1.0f - alpha) * prev_hi + alpha * hi_raw;
}

extern "C" __global__ void normalize_depth_to_near(
    const float* depth, float* near, const float* band, const int* flags, int n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (flags[1] == 0) {
        near[idx] = 0.0f;
        return;
    }
    float lo = band[0];
    float hi = band[1];
    float span = hi - lo;
    if (!isfinite(lo) || !isfinite(hi) || span <= 1.0e-12f) {
        near[idx] = 0.0f;
        return;
    }
    float d = depth[idx];
    float nval = 0.0f;
    if (isfinite(d)) {
        float inv = 1.0f / fmaxf(d, 1.0e-6f);
        nval = (inv - lo) / span;
    }
    near[idx] = fminf(1.0f, fmaxf(0.0f, nval));
}

extern "C" __global__ void rgb_to_gray_resize(
    const unsigned char* rgb,
    unsigned char* gray,
    int H,
    int W,
    int h,
    int w)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)h * w;
    if (idx >= total) return;
    int x = (int)(idx % w);
    int y = (int)(idx / w);
    int sx = (w > 1) ? (int)roundf((float)x * (float)(W - 1) / (float)(w - 1)) : 0;
    int sy = (h > 1) ? (int)roundf((float)y * (float)(H - 1) / (float)(h - 1)) : 0;
    sx = min(max(sx, 0), W - 1);
    sy = min(max(sy, 0), H - 1);
    long si = ((long)sy * W + sx) * 3;
    float r = (float)rgb[si];
    float g = (float)rgb[si + 1];
    float b = (float)rgb[si + 2];
    gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
}

extern "C" __global__ void affine_stats_blocks(
    const float* cur,
    const float* prev,
    const unsigned char* gray_cur,
    const unsigned char* gray_prev,
    float* stats,
    int n,
    float diff_threshold,
    int use_gray)
{
    extern __shared__ float s[];
    float* scount = s;
    float* scur = s + blockDim.x;
    float* sprev = s + 2 * blockDim.x;
    float* scur2 = s + 3 * blockDim.x;
    float* sprev2 = s + 4 * blockDim.x;
    float count = 0.0f, sum_cur = 0.0f, sum_prev = 0.0f, sum_cur2 = 0.0f, sum_prev2 = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        float c = cur[i];
        float p = prev[i];
        bool ok = isfinite(c) && isfinite(p) && c > 0.02f && c < 0.98f && p > 0.02f && p < 0.98f;
        if (ok && use_gray && diff_threshold > 0.0f) {
            ok = fabsf((float)gray_cur[i] - (float)gray_prev[i]) <= diff_threshold;
        }
        if (ok) {
            count += 1.0f;
            sum_cur += c;
            sum_prev += p;
            sum_cur2 += c * c;
            sum_prev2 += p * p;
        }
    }
    scount[threadIdx.x] = count;
    scur[threadIdx.x] = sum_cur;
    sprev[threadIdx.x] = sum_prev;
    scur2[threadIdx.x] = sum_cur2;
    sprev2[threadIdx.x] = sum_prev2;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scount[threadIdx.x] += scount[threadIdx.x + stride];
            scur[threadIdx.x] += scur[threadIdx.x + stride];
            sprev[threadIdx.x] += sprev[threadIdx.x + stride];
            scur2[threadIdx.x] += scur2[threadIdx.x + stride];
            sprev2[threadIdx.x] += sprev2[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        long o = (long)blockIdx.x * 5;
        stats[o] = scount[0];
        stats[o + 1] = scur[0];
        stats[o + 2] = sprev[0];
        stats[o + 3] = scur2[0];
        stats[o + 4] = sprev2[0];
    }
}

extern "C" __global__ void affine_finalize(
    const float* stats,
    float* scale_bias,
    int blocks,
    int total,
    float max_scale_delta,
    float max_bias)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float count = 0.0f, sum_cur = 0.0f, sum_prev = 0.0f, sum_cur2 = 0.0f, sum_prev2 = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        long o = (long)i * 5;
        count += stats[o];
        sum_cur += stats[o + 1];
        sum_prev += stats[o + 2];
        sum_cur2 += stats[o + 3];
        sum_prev2 += stats[o + 4];
    }
    float min_count = fmaxf(128.0f, (float)total * 0.05f);
    float scale = 1.0f;
    float bias = 0.0f;
    if (count >= min_count) {
        float mean_cur = sum_cur / count;
        float mean_prev = sum_prev / count;
        float var_cur = fmaxf(sum_cur2 / count - mean_cur * mean_cur, 0.0f);
        float var_prev = fmaxf(sum_prev2 / count - mean_prev * mean_prev, 0.0f);
        if (var_cur > 1.0e-6f && var_prev > 1.0e-6f) {
            scale = sqrtf(var_prev / var_cur);
            scale = fminf(1.0f + max_scale_delta, fmaxf(1.0f - max_scale_delta, scale));
            bias = mean_prev - scale * mean_cur;
            bias = fminf(max_bias, fmaxf(-max_bias, bias));
        }
    }
    scale_bias[0] = scale;
    scale_bias[1] = bias;
    scale_bias[2] = count;
}

extern "C" __global__ void stabilize_ema(
    const float* cur,
    float* prev,
    const unsigned char* gray_cur,
    unsigned char* gray_prev,
    float* out,
    const float* scale_bias,
    const int* flags,
    int n,
    float alpha,
    float diff_threshold,
    float static_deadband,
    float static_max_step,
    float motion_max_step,
    int depth_enabled,
    int use_gray,
    int initialized)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float c = cur[idx];
    if (depth_enabled && initialized && flags[0] == 0) {
        c = fminf(1.0f, fmaxf(0.0f, c * scale_bias[0] + scale_bias[1]));
        float a = alpha;
        bool changed = false;
        if (use_gray && diff_threshold > 0.0f
            && fabsf((float)gray_cur[idx] - (float)gray_prev[idx]) > diff_threshold) {
            changed = true;
            a = 1.0f;
        }
        float d = c - prev[idx];
        if (!changed) {
            if (static_deadband > 0.0f && fabsf(d) < static_deadband) {
                c = prev[idx];
                d = 0.0f;
            }
            if (static_max_step > 0.0f && fabsf(d) > static_max_step) {
                d = fminf(static_max_step, fmaxf(-static_max_step, d));
                c = prev[idx] + d;
            }
        } else if (motion_max_step > 0.0f && fabsf(d) > motion_max_step) {
            d = fminf(motion_max_step, fmaxf(-motion_max_step, d));
            c = prev[idx] + d;
        }
        c = prev[idx] * (1.0f - a) + c * a;
    }
    c = fminf(1.0f, fmaxf(0.0f, c));
    out[idx] = c;
    prev[idx] = c;
    if (use_gray) {
        gray_prev[idx] = gray_cur[idx];
    }
}

extern "C" __global__ void dilate_near_rect(
    const float* src, float* dst, int H, int W, int radius)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * W;
    if (idx >= total) return;
    int x = (int)(idx % W);
    int y = (int)(idx / W);
    float mx = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = min(max(y + dy, 0), H - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = min(max(x + dx, 0), W - 1);
            mx = fmaxf(mx, src[(long)yy * W + xx]);
        }
    }
    dst[idx] = mx;
}

// Separable box low-pass (replicate border) == cv2.blur, used to split the
// near map into a low-frequency base and high-frequency detail. Plain CUDA so
// it compiles in the same RawModule (cupyx.scipy.ndimage.uniform_filter pulls a
// cccl header that fails to compile for sm_120 on CUDA 12.9).
extern "C" __global__ void box_blur_h(
    const float* src, float* dst, int H, int W, int radius)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * W;
    if (idx >= total) return;
    int x = (int)(idx % W);
    int y = (int)(idx / W);
    float acc = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        int xx = min(max(x + dx, 0), W - 1);
        acc += src[(long)y * W + xx];
    }
    dst[idx] = acc / (float)(2 * radius + 1);
}

extern "C" __global__ void box_blur_v(
    const float* src, float* dst, int H, int W, int radius)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * W;
    if (idx >= total) return;
    int x = (int)(idx % W);
    int y = (int)(idx / W);
    float acc = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = min(max(y + dy, 0), H - 1);
        acc += src[(long)yy * W + x];
    }
    dst[idx] = acc / (float)(2 * radius + 1);
}

// Base/detail temporal combine. Smooth only the low-frequency base across time
// (optional affine match + EMA), always re-inject the CURRENT frame's detail
// (= near_raw - base_cur). When initialized==0 this is the identity (seeds
// base_prev = base_cur, out = near_raw) so the first frame is untouched.
extern "C" __global__ void base_detail_combine(
    const float* near_raw,
    const float* base_cur,
    const float* base_prev_in,
    float* base_prev_out,
    float* out,
    const float* scale_bias,
    const float* alpha_map,
    int n,
    float alpha,
    int initialized,
    int use_alpha_map)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float raw = near_raw[idx];
    float bcur = base_cur[idx];
    float detail = raw - bcur;
    float bstab;
    if (initialized) {
        float a = use_alpha_map ? alpha_map[idx] : alpha;  // per-tile evidence gate
        float aligned = fminf(1.0f, fmaxf(0.0f, bcur * scale_bias[0] + scale_bias[1]));
        float bprev = base_prev_in[idx];
        bstab = bprev * (1.0f - a) + aligned * a;
        bstab = fminf(1.0f, fmaxf(0.0f, bstab));
    } else {
        bstab = bcur;
    }
    base_prev_out[idx] = bstab;
    float o = bstab + detail;
    out[idx] = fminf(1.0f, fmaxf(0.0f, o));
}

// Per-pixel median over k stacked maps (k <= 9), stack laid out (k, n) row-major.
// Used by the offline symmetric temporal window (8.5.3 B).
extern "C" __global__ void median_stack(const float* stack, float* out, int k, int n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v[9];
    for (int j = 0; j < k; ++j) v[j] = stack[(long)j * n + idx];
    for (int a = 1; a < k; ++a) {            // insertion sort (k is tiny)
        float key = v[a];
        int b = a - 1;
        while (b >= 0 && v[b] > key) { v[b + 1] = v[b]; --b; }
        v[b + 1] = key;
    }
    out[idx] = (k & 1) ? v[k / 2] : 0.5f * (v[k / 2 - 1] + v[k / 2]);
}

// Per-pixel weighted mean over k stacked maps (stack laid out (k, n)). Used by
// the offline symmetric window for a Gaussian-weighted temporal mean of the base.
extern "C" __global__ void weighted_mean_stack(
    const float* stack, const float* w, float* out, int k, int n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int j = 0; j < k; ++j) { float wj = w[j]; acc += wj * stack[(long)j * n + idx]; wsum += wj; }
    out[idx] = (wsum > 1e-12f) ? acc / wsum : stack[idx];
}

// Weighted mean with a per-map affine re-band: out = sum_j w_j (a_j*stack_j + b_j) / sum_j w_j.
// Used by the offline symmetric window's band lookahead (re-normalize each frame
// to the symmetric depth range before averaging). a_j=1,b_j=0 == plain weighted mean.
extern "C" __global__ void weighted_affine_mean_stack(
    const float* stack, const float* w, const float* a, const float* b, float* out, int k, int n)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int j = 0; j < k; ++j) {
        float wj = w[j];
        acc += wj * (a[j] * stack[(long)j * n + idx] + b[j]);
        wsum += wj;
    }
    out[idx] = (wsum > 1e-12f) ? acc / wsum : (a[0] * stack[idx] + b[0]);
}

// Bilinear upsample a small (sh, sw) map to (H, W); used to upsample the
// per-tile evidence-gate alpha from the motion-comp working grid to full res.
extern "C" __global__ void upsample_bilinear(
    const float* src, float* dst, int sh, int sw, int H, int W)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * W;
    if (idx >= total) return;
    int x = (int)(idx % W);
    int y = (int)(idx / W);
    float fx = (W > 1) ? (float)x * (float)(sw - 1) / (float)(W - 1) : 0.0f;
    float fy = (H > 1) ? (float)y * (float)(sh - 1) / (float)(H - 1) : 0.0f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, sw - 1);
    int y1 = min(y0 + 1, sh - 1);
    float ax = fx - (float)x0;
    float ay = fy - (float)y0;
    float v00 = src[(long)y0 * sw + x0];
    float v01 = src[(long)y0 * sw + x1];
    float v10 = src[(long)y1 * sw + x0];
    float v11 = src[(long)y1 * sw + x1];
    float top = v00 * (1.0f - ax) + v01 * ax;
    float bot = v10 * (1.0f - ax) + v11 * ax;
    dst[idx] = top * (1.0f - ay) + bot * ay;
}

// Translate by (dx, dy) with bilinear sampling + replicate border:
// dst(x,y) = src(x - dx, y - dy). Matches cv2.warpAffine([[1,0,dx],[0,1,dy]]),
// used to motion-compensate the previous base into the current frame (VVPS 8.6).
extern "C" __global__ void translate_bilinear(
    const float* src, float* dst, int H, int W, float dx, float dy)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H * W;
    if (idx >= total) return;
    int x = (int)(idx % W);
    int y = (int)(idx / W);
    float sxf = fminf(fmaxf((float)x - dx, 0.0f), (float)(W - 1));
    float syf = fminf(fmaxf((float)y - dy, 0.0f), (float)(H - 1));
    int x0 = (int)floorf(sxf);
    int y0 = (int)floorf(syf);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    float fx = sxf - (float)x0;
    float fy = syf - (float)y0;
    float v00 = src[(long)y0 * W + x0];
    float v01 = src[(long)y0 * W + x1];
    float v10 = src[(long)y1 * W + x0];
    float v11 = src[(long)y1 * W + x1];
    float top = v00 * (1.0f - fx) + v01 * fx;
    float bot = v10 * (1.0f - fx) + v11 * fx;
    dst[idx] = top * (1.0f - fy) + bot * fy;
}
'''

# NVENC width limit mirror (the authoritative cap lives in two_dvr_render).
_MAX_EYE_SIDE = 4096


def gpu_available() -> bool:
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


class _GpuNearPreprocessor:
    """GPU depth->near normalization and lightweight temporal stabilization.

    This mirrors the realtime-safe CPU `TemporalDepthStabilizer` path without
    pulling the low-res DA3 map back through numpy/cv2. DA3/ORT currently still
    returns a host depth tensor; after that single upload, percentile
    normalization, temporal norm, luma gate, affine scale/bias, EMA, and
    soft_shift dilation stay on the CUDA stream.
    """

    _BINS = 256
    _MAX_BLOCKS = 1024

    def __init__(
        self,
        cp,
        hole_fill_mode: str,
        temporal: TemporalDepthStabilizer,
        max_disparity_px: float,
        threads: int = 256,
    ) -> None:
        self.cp = cp
        self.hole_fill_mode = hole_fill_mode
        self.temporal = temporal
        self.max_disparity_px = max(0.0, float(max_disparity_px))
        self.threads = int(threads)
        self.mod = cp.RawModule(code=_GPU_NEAR_KERNELS)
        self.k_smooth = self.mod.get_function("depth_smooth3x3")
        self.k_minmax = self.mod.get_function("inv_minmax_blocks")
        self.k_reduce_minmax = self.mod.get_function("reduce_minmax")
        self.k_hist = self.mod.get_function("inv_histogram")
        self.k_norm_update = self.mod.get_function("percentile_norm_update")
        self.k_normalize = self.mod.get_function("normalize_depth_to_near")
        self.k_gray = self.mod.get_function("rgb_to_gray_resize")
        self.k_affine_stats = self.mod.get_function("affine_stats_blocks")
        self.k_affine_finalize = self.mod.get_function("affine_finalize")
        self.k_stabilize = self.mod.get_function("stabilize_ema")
        self.k_dilate = self.mod.get_function("dilate_near_rect")
        self.k_box_h = self.mod.get_function("box_blur_h")
        self.k_box_v = self.mod.get_function("box_blur_v")
        self.k_base_combine = self.mod.get_function("base_detail_combine")
        self.k_translate = self.mod.get_function("translate_bilinear")
        self.k_upsample = self.mod.get_function("upsample_bilinear")
        self.k_median = self.mod.get_function("median_stack")
        self.k_weighted = self.mod.get_function("weighted_mean_stack")
        self.k_weighted_affine = self.mod.get_function("weighted_affine_mean_stack")
        self.shape: tuple[int, int] | None = None
        self.blocks = 1
        self.band_g = cp.zeros((3,), cp.float32)
        self.flags_g = cp.zeros((2,), cp.int32)
        self.minmax_g = cp.zeros((2,), cp.float32)
        self.hist_g = cp.zeros((self._BINS,), cp.uint32)
        self.scale_bias_g = cp.asarray([1.0, 0.0, 0.0], dtype=cp.float32)
        self.min_blocks_g = None
        self.max_blocks_g = None
        self.stats_g = None
        self.depth_smooth_g = None
        self.near_raw_g = None
        self.near_prev_g = None
        self.near_out_g = None
        self.near_dilate_g = None
        self.gray_cur_g = None
        self.gray_prev_g = None
        # base/detail (mode=ema) device state: smoothed low-frequency base.
        self.base_cur_g = None
        self.base_tmp_g = None
        self.base_prev_g = None
        self.base_warp_g = None
        # motion-compensation (8.6) state.
        self.gray_small_g = None
        self._mc_sw = 0
        self._mc_sh = 0
        self._mc_prev_small = None  # host np.float32 gray of previous frame
        self._mc_window = None       # host cv2 Hanning window
        # evidence gate (8.6.3) state.
        self.alpha_small_g = None
        self.alpha_full_g = None
        self._evid_alpha_small = None  # host np.float32 (sh, sw) or None
        self.depth_initialized = False
        self.gray_initialized = False

    def reset(self) -> None:
        self.band_g.fill(0)
        self.flags_g.fill(0)
        self.scale_bias_g[0] = np.float32(1.0)
        self.scale_bias_g[1] = np.float32(0.0)
        self.scale_bias_g[2] = np.float32(0.0)
        # base_prev_g buffer is kept; depth_initialized=False re-seeds it on the
        # next frame (base_detail_combine identity branch).
        self._mc_prev_small = None
        self._evid_alpha_small = None
        self.depth_initialized = False
        self.gray_initialized = False

    def _ensure_shape(self, h: int, w: int) -> None:
        cp = self.cp
        shape = (int(h), int(w))
        n = shape[0] * shape[1]
        blocks = max(1, min(self._MAX_BLOCKS, (n + self.threads - 1) // self.threads))
        if self.shape == shape and self.blocks == blocks:
            return
        self.shape = shape
        self.blocks = blocks
        self.min_blocks_g = cp.empty((blocks,), cp.float32)
        self.max_blocks_g = cp.empty((blocks,), cp.float32)
        self.stats_g = cp.empty((blocks, 5), cp.float32)
        self.depth_smooth_g = cp.empty(shape, cp.float32)
        self.near_raw_g = cp.empty(shape, cp.float32)
        self.near_prev_g = cp.empty(shape, cp.float32)
        self.near_out_g = cp.empty(shape, cp.float32)
        self.near_dilate_g = cp.empty(shape, cp.float32)
        self.gray_cur_g = cp.empty(shape, cp.uint8)
        self.gray_prev_g = cp.empty(shape, cp.uint8)
        self.base_cur_g = cp.empty(shape, cp.float32)
        self.base_tmp_g = cp.empty(shape, cp.float32)
        self.base_prev_g = cp.empty(shape, cp.float32)
        self.base_warp_g = cp.empty(shape, cp.float32)
        # Motion-comp working (downsampled) gray size: >= /_MC_DOWNSAMPLE and
        # bounded so the longest side <= _MC_MAX_WORK (phaseCorrelate FFT cost).
        ds = max(_MC_DOWNSAMPLE, -(-max(shape) // _MC_MAX_WORK))
        self._mc_sw = max(8, shape[1] // ds)
        self._mc_sh = max(8, shape[0] // ds)
        self.gray_small_g = cp.empty((self._mc_sh, self._mc_sw), cp.uint8)
        self.alpha_small_g = cp.empty((self._mc_sh, self._mc_sw), cp.float32)
        self.alpha_full_g = cp.empty(shape, cp.float32)
        self._mc_prev_small = None
        self._mc_window = None
        self._evid_alpha_small = None
        self.depth_initialized = False
        self.gray_initialized = False
        # A resolution change invalidates both temporal states.
        self.band_g.fill(0)
        self.flags_g.fill(0)

    def _prepare_gray(self, frame_rgb_g, h: int, w: int):
        if frame_rgb_g is None:
            return None
        cp = self.cp
        rgb = cp.asarray(frame_rgb_g)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        if not rgb.flags.c_contiguous:
            rgb = cp.ascontiguousarray(rgb[:, :, :3])
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        n = h * w
        grid = ((n + self.threads - 1) // self.threads,)
        self.k_gray(grid, (self.threads,), (rgb, self.gray_cur_g, np.int32(H), np.int32(W), np.int32(h), np.int32(w)))
        return self.gray_cur_g

    def _normalize_to_near(self, depth_g, h: int, w: int):
        cp = self.cp
        n = h * w
        total_blocks = self.blocks
        src_g = depth_g
        if self.hole_fill_mode == HOLE_FILL_INVERSE_WARP:
            grid = ((n + self.threads - 1) // self.threads,)
            self.k_smooth(grid, (self.threads,), (depth_g, self.depth_smooth_g, np.int32(h), np.int32(w)))
            src_g = self.depth_smooth_g

        shared_minmax = self.threads * 2 * 4
        self.k_minmax(
            (total_blocks,),
            (self.threads,),
            (src_g, self.min_blocks_g, self.max_blocks_g, np.int32(n)),
            shared_mem=shared_minmax,
        )
        self.k_reduce_minmax(
            (1,),
            (self.threads,),
            (self.min_blocks_g, self.max_blocks_g, self.minmax_g, np.int32(total_blocks)),
            shared_mem=shared_minmax,
        )
        self.hist_g.fill(0)
        self.k_hist(
            (total_blocks,),
            (self.threads,),
            (src_g, self.minmax_g, self.hist_g, np.int32(n), np.int32(self._BINS)),
        )
        self.k_norm_update(
            (1,),
            (1,),
            (
                self.hist_g,
                self.minmax_g,
                self.band_g,
                self.flags_g,
                np.int32(self._BINS),
                np.int32(n),
                np.float32(self.temporal.norm_alpha),
                np.float32(self.temporal.norm_reset_threshold),
                np.int32(1 if self.temporal.norm_enabled else 0),
            ),
        )
        grid = ((n + self.threads - 1) // self.threads,)
        self.k_normalize(grid, (self.threads,), (src_g, self.near_raw_g, self.band_g, self.flags_g, np.int32(n)))
        return self.near_raw_g

    def _update_affine(self, current_g, gray_g, n: int) -> None:
        cp = self.cp
        use_gray = bool(gray_g is not None and self.gray_initialized)
        if not (self.temporal.depth_enabled and self.temporal.affine_enabled and self.depth_initialized):
            self.scale_bias_g[0] = np.float32(1.0)
            self.scale_bias_g[1] = np.float32(0.0)
            self.scale_bias_g[2] = np.float32(0.0)
            return
        self.k_affine_stats(
            (self.blocks,),
            (self.threads,),
            (
                current_g,
                self.near_prev_g,
                self.gray_cur_g,
                self.gray_prev_g,
                self.stats_g,
                np.int32(n),
                np.float32(self.temporal.flow_diff_threshold),
                np.int32(1 if use_gray else 0),
            ),
            shared_mem=self.threads * 5 * 4,
        )
        self.k_affine_finalize(
            (1,),
            (1,),
            (
                self.stats_g,
                self.scale_bias_g,
                np.int32(self.blocks),
                np.int32(n),
                np.float32(self.temporal.affine_max_scale_delta),
                np.float32(self.temporal.affine_max_bias),
            ),
        )

    def _px_to_near(self, px: float) -> np.float32:
        if self.max_disparity_px <= 1.0e-6 or px <= 0.0:
            return np.float32(0.0)
        return np.float32(float(px) / self.max_disparity_px)

    def _base_kernel(self, h: int, w: int) -> int:
        k = int(round(min(int(h), int(w)) / float(_BASE_LOWPASS_DIV)))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        return k

    def _estimate_translation_gpu(self, frame_rgb_g, h: int, w: int) -> tuple[float, float]:
        """Global (dx, dy) registering prev->cur via phase correlation.

        The downsampled gray is produced on the GPU (rgb_to_gray_resize) and only
        the small (<=512px) gray is copied to the host for cv2.phaseCorrelate, so
        the per-frame host transfer is tiny. Returns (0, 0) on the first frame or
        degenerate/low-response input. Mirrors the CPU
        TemporalDepthStabilizer._estimate_global_translation (same gates/scaling).
        """
        cp = self.cp
        self._evid_alpha_small = None
        if frame_rgb_g is None:
            self._mc_prev_small = None
            return 0.0, 0.0
        rgb = cp.asarray(frame_rgb_g)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            self._mc_prev_small = None
            return 0.0, 0.0
        if not rgb.flags.c_contiguous:
            rgb = cp.ascontiguousarray(rgb[:, :, :3])
        big_h, big_w = int(rgb.shape[0]), int(rgb.shape[1])
        sw, sh = self._mc_sw, self._mc_sh
        grid = ((sh * sw + self.threads - 1) // self.threads,)
        self.k_gray(
            grid,
            (self.threads,),
            (rgb, self.gray_small_g, np.int32(big_h), np.int32(big_w), np.int32(sh), np.int32(sw)),
        )
        cur_small = cp.asnumpy(self.gray_small_g).astype(np.float32)
        prev_small = self._mc_prev_small
        self._mc_prev_small = cur_small
        if prev_small is None or prev_small.shape != cur_small.shape:
            return 0.0, 0.0
        if self._mc_window is None or self._mc_window.shape != (sh, sw):
            self._mc_window = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
        # phaseCorrelate windows its inputs IN PLACE; cur_small is retained as
        # _mc_prev_small (and reused below for the residual), so pass copies.
        (dx_s, dy_s), response = cv2.phaseCorrelate(prev_small.copy(), cur_small.copy(), self._mc_window)
        if not np.isfinite(response) or not np.isfinite(dx_s) or not np.isfinite(dy_s):
            return 0.0, 0.0
        if response < _MC_MIN_RESPONSE:
            return 0.0, 0.0
        # Evidence gate (8.6.3): residual between cur and motion-compensated prev,
        # on the small grid (shift there is the raw phaseCorrelate dx_s/dy_s). The
        # alpha map is upsampled to full res by the caller.
        if self.temporal.evidence_gate_enabled:
            aligned = prev_small
            if abs(dx_s) >= 0.5 or abs(dy_s) >= 0.5:
                m = np.array([[1.0, 0.0, dx_s], [0.0, 1.0, dy_s]], dtype=np.float32)
                aligned = cv2.warpAffine(
                    prev_small, m, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
            resid = np.abs(cur_small - aligned)
            tw = max(1, sw // (_EVID_TILE // max(1, _MC_DOWNSAMPLE) or 1))
            th = max(1, sh // (_EVID_TILE // max(1, _MC_DOWNSAMPLE) or 1))
            rc = cv2.resize(cv2.resize(resid, (tw, th), interpolation=cv2.INTER_AREA), (sw, sh), interpolation=cv2.INTER_LINEAR)
            t = np.clip((rc - _EVID_R_LO) / max(1e-3, _EVID_R_HI - _EVID_R_LO), 0.0, 1.0)
            a_lock = np.float32(self.temporal.depth_alpha * self.temporal.evidence_lock_scale)
            self._evid_alpha_small = (a_lock + (np.float32(1.0) - a_lock) * t).astype(np.float32)
        dx = float(dx_s) * (float(w) / float(sw))
        dy = float(dy_s) * (float(h) / float(sh))
        dx = max(-_MC_MAX_SHIFT_PX, min(_MC_MAX_SHIFT_PX, dx))
        dy = max(-_MC_MAX_SHIFT_PX, min(_MC_MAX_SHIFT_PX, dy))
        return dx, dy

    def prepare(self, depth, frame_rgb_g=None):
        cp = self.cp
        depth_g = cp.asarray(depth, dtype=cp.float32)
        if depth_g.ndim != 2:
            raise ValueError(f"depth must be 2D, got {depth_g.shape}")
        if not depth_g.flags.c_contiguous:
            depth_g = cp.ascontiguousarray(depth_g)
        h, w = int(depth_g.shape[0]), int(depth_g.shape[1])
        self._ensure_shape(h, w)
        n = h * w
        current_g = self._normalize_to_near(depth_g, h, w)
        if not self.temporal.depth_enabled:
            out_g = current_g
            if self.hole_fill_mode == HOLE_FILL_SOFT_SHIFT:
                grid = ((n + self.threads - 1) // self.threads,)
                radius = max(1, int(round(w / 512.0)))
                self.k_dilate(
                    grid,
                    (self.threads,),
                    (out_g, self.near_dilate_g, np.int32(h), np.int32(w), np.int32(radius)),
                )
                out_g = self.near_dilate_g
            return out_g

        # VVPS stabilizer (in-house) -- base/detail temporal stabilization
        # (mode=ema), mirroring the CPU
        # TemporalDepthStabilizer._stabilize_base_detail: smooth only the
        # low-frequency base and always re-inject the current frame's detail, so
        # the disparity surface stays spatially coherent and hole-fill never tears
        # phantom holes inside a foreground body. The fused per-pixel k_stabilize
        # kernel is intentionally bypassed (kept only for reference).
        grid = ((n + self.threads - 1) // self.threads,)
        radius = max(1, self._base_kernel(h, w) // 2)
        self.k_box_h(grid, (self.threads,), (current_g, self.base_tmp_g, np.int32(h), np.int32(w), np.int32(radius)))
        self.k_box_v(grid, (self.threads,), (self.base_tmp_g, self.base_cur_g, np.int32(h), np.int32(w), np.int32(radius)))
        # warp-then-filter (8.6): align the previous base into the current frame
        # by the estimated global translation before the affine/EMA. mc_prev_g is
        # the (possibly warped) previous base used as the EMA reference; the new
        # state is always written back to base_prev_g.
        mc_prev_g = self.base_prev_g
        use_alpha_map = 0
        if self.temporal.motion_comp_enabled or self.temporal.evidence_gate_enabled:
            dx, dy = self._estimate_translation_gpu(frame_rgb_g, h, w)
            if self.temporal.motion_comp_enabled and self.depth_initialized and (abs(dx) >= 0.5 or abs(dy) >= 0.5):
                self.k_translate(
                    grid,
                    (self.threads,),
                    (self.base_prev_g, self.base_warp_g, np.int32(h), np.int32(w), np.float32(dx), np.float32(dy)),
                )
                mc_prev_g = self.base_warp_g
            if (
                self.temporal.evidence_gate_enabled
                and self.depth_initialized
                and self._evid_alpha_small is not None
            ):
                self.alpha_small_g.set(self._evid_alpha_small)
                self.k_upsample(
                    grid,
                    (self.threads,),
                    (self.alpha_small_g, self.alpha_full_g, np.int32(self._mc_sh), np.int32(self._mc_sw), np.int32(h), np.int32(w)),
                )
                use_alpha_map = 1
        # Affine-match the current base to the previous stabilized base (global
        # scale/bias, bounded), reusing the existing stats kernels with use_gray=0.
        if self.depth_initialized and self.temporal.affine_enabled:
            self.k_affine_stats(
                (self.blocks,),
                (self.threads,),
                (
                    self.base_cur_g,
                    mc_prev_g,
                    self.gray_cur_g,
                    self.gray_prev_g,
                    self.stats_g,
                    np.int32(n),
                    np.float32(0.0),
                    np.int32(0),
                ),
                shared_mem=self.threads * 5 * 4,
            )
            self.k_affine_finalize(
                (1,),
                (1,),
                (
                    self.stats_g,
                    self.scale_bias_g,
                    np.int32(self.blocks),
                    np.int32(n),
                    np.float32(self.temporal.affine_max_scale_delta),
                    np.float32(self.temporal.affine_max_bias),
                ),
            )
        else:
            self.scale_bias_g[0] = np.float32(1.0)
            self.scale_bias_g[1] = np.float32(0.0)
            self.scale_bias_g[2] = np.float32(0.0)
        self.k_base_combine(
            grid,
            (self.threads,),
            (
                current_g,
                self.base_cur_g,
                mc_prev_g,
                self.base_prev_g,
                self.near_out_g,
                self.scale_bias_g,
                self.alpha_full_g,
                np.int32(n),
                np.float32(self.temporal.depth_alpha),
                np.int32(1 if self.depth_initialized else 0),
                np.int32(use_alpha_map),
            ),
        )
        self.depth_initialized = True
        out_g = self.near_out_g
        if self.hole_fill_mode == HOLE_FILL_SOFT_SHIFT:
            radius = max(1, int(round(w / 512.0)))
            self.k_dilate(
                grid,
                (self.threads,),
                (out_g, self.near_dilate_g, np.int32(h), np.int32(w), np.int32(radius)),
            )
            out_g = self.near_dilate_g
        return out_g


class GpuSymmetricWindow:
    """GPU offline symmetric temporal window (8.5.3 B) -- mirrors the numpy
    SymmetricBaseWindow but keeps base/detail/rgb on device.

    Reuses the preprocessor's compiled kernels (box blur, translate, upsample,
    median). Output trails input by ``radius`` frames; ``flush`` drains the tail.
    Used only for offline conversion, where the causal stabilizer is disabled and
    this provides lag-free symmetric smoothing of the base instead.
    """

    def __init__(self, pre: "_GpuNearPreprocessor", radius: int = 2) -> None:
        self.cp = pre.cp
        self.pre = pre
        self.threads = pre.threads
        self.radius = max(1, min(_WIN_MAX_RADIUS, int(radius)))
        self.k_max = 2 * self.radius + 1
        self._buf: list[dict] = []
        self._base_idx = 0
        self._next = 0
        self._window = None
        self._scratch_shape: tuple[int, int] | None = None
        self.weights_g = self.cp.empty((self.k_max,), self.cp.float32)
        self.reband_a_g = self.cp.empty((self.k_max,), self.cp.float32)
        self.reband_b_g = self.cp.empty((self.k_max,), self.cp.float32)

    def reset(self) -> None:
        self._buf.clear()
        self._base_idx = 0
        self._next = 0

    @property
    def delay(self) -> int:
        return self.radius

    def _ensure_scratch(self, h: int, w: int, sh: int, sw: int) -> None:
        cp = self.cp
        if self._scratch_shape == (h, w):
            return
        self._scratch_shape = (h, w)
        self.stack_g = cp.empty((self.k_max, h, w), cp.float32)
        self.base_med_g = cp.empty((h, w), cp.float32)
        self.mask_small_g = cp.empty((sh, sw), cp.float32)
        self.mask_full_g = cp.empty((h, w), cp.float32)

    def _base_detail(self, near_g):
        cp = self.cp
        h, w = int(near_g.shape[0]), int(near_g.shape[1])
        radius = max(1, self.pre._base_kernel(h, w) // 2)
        grid = ((h * w + self.threads - 1) // self.threads,)
        tmp = cp.empty((h, w), cp.float32)
        base = cp.empty((h, w), cp.float32)
        self.pre.k_box_h(grid, (self.threads,), (near_g, tmp, np.int32(h), np.int32(w), np.int32(radius)))
        self.pre.k_box_v(grid, (self.threads,), (tmp, base, np.int32(h), np.int32(w), np.int32(radius)))
        return base, near_g - base

    def _gray_small_host(self, frame_g) -> np.ndarray:
        cp = self.cp
        rgb = cp.asarray(frame_g)
        if not rgb.flags.c_contiguous:
            rgb = cp.ascontiguousarray(rgb[:, :, :3])
        big_h, big_w = int(rgb.shape[0]), int(rgb.shape[1])
        sw, sh = self.pre._mc_sw, self.pre._mc_sh
        grid = ((sh * sw + self.threads - 1) // self.threads,)
        self.pre.k_gray(grid, (self.threads,), (rgb, self.pre.gray_small_g, np.int32(big_h), np.int32(big_w), np.int32(sh), np.int32(sw)))
        return cp.asnumpy(self.pre.gray_small_g).astype(np.float32)

    def _shift(self, prev_small: np.ndarray, cur_small: np.ndarray) -> tuple[float, float]:
        sh, sw = cur_small.shape
        if self._window is None or self._window.shape != (sh, sw):
            self._window = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
        # phaseCorrelate windows its inputs IN PLACE; both are buffered, copy.
        (dx, dy), response = cv2.phaseCorrelate(prev_small.copy(), cur_small.copy(), self._window)
        if not np.isfinite(response) or response < _MC_MIN_RESPONSE or not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0, 0.0
        return float(dx), float(dy)

    def push(self, near_g, align_frame_g, render_rgb_g, payload, band=None) -> list[tuple]:
        base, detail = self._base_detail(near_g)
        self._buf.append({
            "base": base,
            "detail": detail,
            "rgb": render_rgb_g.copy(),
            "small": self._gray_small_host(align_frame_g),
            "payload": payload,
            "band": band,
        })
        out: list[tuple] = []
        total = self._base_idx + len(self._buf)
        while self._next + self.radius < total:
            out.append(self._emit_abs(self._next))
            self._next += 1
            self._drop()
        return out

    def flush(self) -> list[tuple]:
        out: list[tuple] = []
        total = self._base_idx + len(self._buf)
        while self._next < total:
            out.append(self._emit_abs(self._next))
            self._next += 1
            self._drop()
        return out

    def _drop(self) -> None:
        while self._buf and self._base_idx < self._next - self.radius:
            self._buf.pop(0)
            self._base_idx += 1

    def _emit_abs(self, j: int) -> tuple:
        cp = self.cp
        c = j - self._base_idx
        centre = self._buf[c]
        base_c = centre["base"]
        detail_c = centre["detail"]
        small_c = centre["small"]
        h, w = int(base_c.shape[0]), int(base_c.shape[1])
        sh, sw = small_c.shape
        n = h * w
        grid = ((n + self.threads - 1) // self.threads,)
        self._ensure_scratch(h, w, sh, sw)
        cp.copyto(self.stack_g[0], base_c)
        sigma = max(1e-3, self.radius / 1.5)
        # Band lookahead: symmetric raw-band re-norm coefficients (a_j*base + b_j).
        lo_s, hi_s, r_c, s_c = self._symmetric_band(c, sigma)
        bands = [b["band"] for b in self._buf]

        def _reband_ab(item):
            if lo_s is None:
                return 1.0, 0.0
            lo_u, hi_u, _, _ = item["band"]
            span_s = hi_s - lo_s
            return max(hi_u - lo_u, 1e-6) / span_s, (lo_u - lo_s) / span_s

        weights = [1.0]  # centre, offset 0
        a_c, b_c = _reband_ab(centre)
        reband_a = [a_c]
        reband_b = [b_c]
        slot = 1
        max_resid_small = np.zeros((sh, sw), dtype=np.float32)
        for i, item in enumerate(self._buf):
            if i == c:
                continue
            weights.append(float(np.exp(-((i - c) ** 2) / (2.0 * sigma * sigma))))
            aj, bj = _reband_ab(item)
            reband_a.append(aj)
            reband_b.append(bj)
            dx_s, dy_s = self._shift(item["small"], small_c)
            if abs(dx_s) >= 0.5 or abs(dy_s) >= 0.5:
                m = np.array([[1.0, 0.0, dx_s], [0.0, 1.0, dy_s]], dtype=np.float32)
                small_aligned = cv2.warpAffine(item["small"], m, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                self.pre.k_translate(
                    grid, (self.threads,),
                    (item["base"], self.stack_g[slot], np.int32(h), np.int32(w),
                     np.float32(dx_s * (float(w) / float(sw))), np.float32(dy_s * (float(h) / float(sh)))),
                )
            else:
                small_aligned = item["small"]
                cp.copyto(self.stack_g[slot], item["base"])
            np.maximum(max_resid_small, np.abs(small_c - small_aligned), out=max_resid_small)
            slot += 1
        # Gaussian-weighted symmetric mean with per-frame band re-norm (lag-free).
        self.weights_g[:slot].set(np.asarray(weights, dtype=np.float32))
        self.reband_a_g[:slot].set(np.asarray(reband_a, dtype=np.float32))
        self.reband_b_g[:slot].set(np.asarray(reband_b, dtype=np.float32))
        self.pre.k_weighted_affine(
            grid, (self.threads,),
            (self.stack_g, self.weights_g, self.reband_a_g, self.reband_b_g, self.base_med_g, np.int32(slot), np.int32(n)),
        )
        m_small = (1.0 - np.clip((max_resid_small - _EVID_R_LO) / max(1e-3, _EVID_R_HI - _EVID_R_LO), 0.0, 1.0)).astype(np.float32)
        self.mask_small_g.set(m_small)
        self.pre.k_upsample(grid, (self.threads,), (self.mask_small_g, self.mask_full_g, np.int32(sh), np.int32(sw), np.int32(h), np.int32(w)))
        base_c_s = base_c * cp.float32(a_c) + cp.float32(b_c)  # centre base in symmetric band
        base_out = base_c_s + self.mask_full_g * (self.base_med_g - base_c_s)
        near_out = base_out + detail_c * cp.float32(r_c)
        cp.clip(near_out, 0.0, 1.0, out=near_out)
        return near_out, centre["rgb"], centre["payload"]

    def _symmetric_band(self, c: int, sigma: float):
        """Symmetric (Gaussian-weighted) raw band over the window + centre re-band
        scale/offset. Returns (None, None, 1.0, 0.0) when band info is absent."""
        if any(b["band"] is None for b in self._buf):
            return None, None, 1.0, 0.0
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
            return None, None, 1.0, 0.0
        lo_uc, hi_uc, _, _ = self._buf[c]["band"]
        span_s = hi_s - lo_s
        return lo_s, hi_s, max(hi_uc - lo_uc, 1e-6) / span_s, (lo_uc - lo_s) / span_s


class GpuStereoRenderer:
    """GPU stereo renderer (flat3d, fisheye-180, hequirect-180).

    Same call shape as the CPU StereoRenderer. inverse_warp uses a single
    gather kernel (combined warp+projection for VR). soft_shift uses the
    forward-warp + z-buffer + hole-fill passes into a flat (H, 2W) SBS, which
    flat3d returns directly and VR projects through the map.
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
        if hole_fill_mode not in (HOLE_FILL_INVERSE_WARP, HOLE_FILL_SOFT_SHIFT):
            raise ValueError(f"GpuStereoRenderer: unsupported hole_fill {hole_fill_mode}")
        import cupy as cp

        self.cp = cp
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.projection = projection
        self.eye_distance_mm = float(eye_distance_mm)
        self.hole_fill_mode = hole_fill_mode
        self._threads = 256
        self.max_shift = np.float32(_max_disparity_pixels(self.src_w, self.eye_distance_mm))
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
            max_disparity_px=float(self.max_shift),
            static_deadband_px=temporal_static_deadband_px,
            static_max_step_px=temporal_static_max_step_px,
            motion_max_step_px=temporal_motion_max_step_px,
        )
        self._near_pre = _GpuNearPreprocessor(cp, hole_fill_mode, self.temporal, float(self.max_shift), self._threads)
        self._sh, self._sw = np.int32(self.src_h), np.int32(self.src_w)
        self._soft = hole_fill_mode == HOLE_FILL_SOFT_SHIFT
        # Debug: skip hole-fill/blend and paint disocclusion holes magenta so the
        # raw forward-warp is visible -- shows whether holes are bounded by clean
        # background (=> fill-algorithm question) or by leftover foreground
        # slivers (=> depth/segmentation question). flat3d soft_shift only.
        self._debug_holes = bool(int(os.environ.get("PT_TWO_DVR_DEBUG_HOLES", "0") or 0))
        # Background-side rim cleanup width (px) for the hybrid hole fill. This
        # handles low-near non-hole seam pixels next to true disocclusion holes;
        # PT_TWO_DVR_RIM=0 disables it for diagnostics.
        self._rim = _two_dvr_rim_width(self.src_w)
        # Cleanup of background slivers embedded inside the body or hair edge
        # (white vertical stripes). Auto-enabled by default; set
        # PT_TWO_DVR_FG_BAD=0 to disable for diagnostics.
        self._fg_bad = _two_dvr_fg_bad_width(self.src_w)
        self._project = projection != PROJECTION_FLAT_3D
        self._near_g = None
        self._frame_g = cp.empty((self.src_h, self.src_w, 3), cp.uint8)

        if self._project:
            side = _flat_vr_eye_size(self.src_w, self.src_h, flat_fov_deg)
            side = min(side, _MAX_EYE_SIDE)
            side -= side % 2  # NVENC wants even dimensions
            pmap = make_projection_map(self.src_w, self.src_h, projection, flat_fov_deg, eye_size=side)
            self.side = int(side)
            self.out_w, self.out_h = side * 2, side
            self._mapx_g = cp.asarray(np.ascontiguousarray(pmap.map_x, np.float32))
            self._mapy_g = cp.asarray(np.ascontiguousarray(pmap.map_y, np.float32))
            self._mask_g = cp.asarray(np.ascontiguousarray(pmap.mask.astype(np.uint8)))
        else:
            self.out_w, self.out_h = self.src_w * 2, self.src_h

        self._out_g = cp.empty((self.out_h, self.out_w, 3), cp.uint8)
        self._blocks = (self.out_h * self.out_w + self._threads - 1) // self._threads

        if self._soft:
            mod = cp.RawModule(code=_SOFT_SHIFT_KERNELS)
            self._k_zbuf = mod.get_function("fw_zbuf")
            self._k_color = mod.get_function("fw_color")
            self._k_fill = mod.get_function("fw_fill")
            self._k_blend = mod.get_function("fw_blend")
            self._k_hole_inv = mod.get_function("fw_hole_from_inv")
            self._k_fg_bad = mod.get_function("fw_fg_bad_local")
            self._k_project = mod.get_function("project_flat_lr") if self._project else None
            # Hybrid (flat3d): an inverse_warp sub-render supplies the hole pixels
            # so wide disocclusion gaps don't get the row-copy banding/smear.
            if not self._project:
                self._inv_kernel = cp.RawKernel(_SBS_INV_WARP_KERNEL, "sbs_inv_warp")
                self._inv_g = cp.empty((self.src_h, self.src_w * 2, 3), cp.uint8)
            # Flat (H, 2W) SBS: _flat_g = warped+filled scratch, _flatb_g = soft-
            # blended result. For flat3d the blended result IS the output.
            self._flat_g = cp.empty((self.src_h, self.src_w * 2, 3), cp.uint8)
            self._flatb_g = (cp.empty((self.src_h, self.src_w * 2, 3), cp.uint8)
                             if self._project else self._out_g)
            self._zbuf_g = cp.empty((self.src_h, self.src_w * 2), cp.int32)
            self._src_blocks = (self.src_h * self.src_w + self._threads - 1) // self._threads
            self._flat_blocks = (self.src_h * self.src_w * 2 + self._threads - 1) // self._threads
        elif self._project:
            self._kernel = cp.RawKernel(_SBS_PROJECT_WARP_KERNEL, "sbs_project_warp")
        else:
            self._kernel = cp.RawKernel(_SBS_INV_WARP_KERNEL, "sbs_inv_warp")

        self._host_out = cp.cuda.alloc_pinned_memory(self._out_g.nbytes)
        self._out_view = np.frombuffer(self._host_out, np.uint8, self._out_g.size).reshape(
            self.out_h, self.out_w, 3
        )
        # Warm the JIT so the first real frame isn't slow.
        self._near_g = cp.zeros((2, 2), cp.float32)
        self._launch(self._frame_g, self._near_g)
        cp.cuda.Stream.null.synchronize()

    def _launch(self, frame_g, near_g):
        h, w = np.int32(near_g.shape[0]), np.int32(near_g.shape[1])
        if self._soft:
            self._zbuf_g.fill(0)
            self._flat_g.fill(0)
            self._k_zbuf((self._src_blocks,), (self._threads,),
                         (near_g, self._zbuf_g, self._sh, self._sw, h, w, self.max_shift))
            self._k_color((self._src_blocks,), (self._threads,),
                          (frame_g, near_g, self._zbuf_g, self._flat_g, self._sh, self._sw, h, w, self.max_shift))
            if self._debug_holes and not self._project:
                # Raw forward warp, holes painted magenta, no fill/blend.
                cp = self.cp
                self._out_g[:] = self._flat_g
                self._out_g[self._zbuf_g == 0] = cp.asarray((255, 0, 255), dtype=cp.uint8)
                return
            if not self._project:
                # Hybrid: fill soft_shift holes from an inverse_warp sub-render
                # (no holes -> no row-copy band), then feather only the seam.
                self._inv_kernel((self._blocks,), (self._threads,),
                                 (frame_g, near_g, self._inv_g, self._sh, self._sw, h, w, self.max_shift))
                self._k_hole_inv((self._flat_blocks,), (self._threads,),
                                 (self._flat_g, self._zbuf_g, self._inv_g, self._sh, self._sw,
                                  np.int32(self._rim)))
            else:
                self._k_fill((self._flat_blocks,), (self._threads,),
                             (self._flat_g, self._zbuf_g, near_g, self._sh, self._sw, h, w))
            self._k_blend((self._flat_blocks,), (self._threads,),
                          (self._flat_g, self._zbuf_g, self._flatb_g, self._sh, self._sw))
            if self._fg_bad > 0:
                # Replace background slivers embedded inside the body (enclosed by
                # foreground on both sides) with the local foreground colour. Runs
                # on the blended SBS so nothing downstream re-softens it.
                self._k_fg_bad((self._flat_blocks,), (self._threads,),
                               (self._flatb_g, self._zbuf_g, self._sh, self._sw,
                                np.int32(self._fg_bad)))
            if self._project:
                self._k_project((self._blocks,), (self._threads,),
                                (self._flatb_g, self._mapx_g, self._mapy_g, self._mask_g, self._out_g,
                                 self._sh, self._sw, np.int32(self.side)))
        elif self._project:
            self._kernel((self._blocks,), (self._threads,),
                         (frame_g, near_g, self._mapx_g, self._mapy_g, self._mask_g, self._out_g,
                          self._sh, self._sw, h, w, np.int32(self.side), self.max_shift))
        else:
            self._kernel((self._blocks,), (self._threads,),
                         (frame_g, near_g, self._out_g, self._sh, self._sw, h, w, self.max_shift))

    def render_into_gpu(self, frame_g, near_g):
        """Warp/project from a GPU RGB frame + GPU normalized low-res near map,
        writing into (and returning) the reused GPU SBS buffer. No host transfer."""
        self._launch(frame_g, near_g)
        return self._out_g

    def reset(self) -> None:
        self.temporal.reset()
        self._near_pre.reset()

    def prepare_near(self, depth, frame_rgb=None) -> np.ndarray:
        return near_from_depth(depth, self.hole_fill_mode, self.temporal, frame_rgb)

    def prepare_near_gpu(self, depth, frame_rgb_g=None):
        return self._near_pre.prepare(depth, frame_rgb_g)

    def render(self, frame_rgb, depth):
        cp = self.cp
        self._frame_g.set(np.ascontiguousarray(frame_rgb))
        near_g = self.prepare_near_gpu(depth, self._frame_g)
        self._launch(self._frame_g, near_g)
        self._out_g.get(out=self._out_view)
        return self._out_view

    def render_near(self, frame_rgb, near):
        cp = self.cp
        self._frame_g.set(np.ascontiguousarray(frame_rgb))
        near_cpu = near_for_render(
            near,
            self.hole_fill_mode,
            soft_shift_sharpen=self.hole_fill_mode == HOLE_FILL_SOFT_SHIFT,
        )
        near_g = cp.asarray(np.ascontiguousarray(near_cpu, dtype=np.float32))
        self._launch(self._frame_g, near_g)
        self._out_g.get(out=self._out_view)
        return self._out_view
