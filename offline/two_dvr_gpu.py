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
# entry never calls configure_gpu_runtime_cache(), so hard-set it here, before
# this module's lazy `import cupy`, to neutralize any stale shell value.
os.environ["CUPY_COMPILE_WITH_PTX"] = "0"

import numpy as np

from offline.two_dvr_render import (
    DEFAULT_EYE_DISTANCE_MM,
    DEFAULT_FLAT_FOV_DEG,
    DEFAULT_HOLE_FILL_MODE,
    HOLE_FILL_INVERSE_WARP,
    HOLE_FILL_SOFT_SHIFT,
    PROJECTION_FLAT_3D,
    _dilate_near_fg,
    _flat_vr_eye_size,
    _max_disparity_pixels,
    _normalize_near,
    _smooth_depth,
    make_projection_map,
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


class GpuStereoRenderer:
    """GPU stereo renderer (flat3d, fisheye-180, hequirect-180).

    Same call shape as the CPU StereoRenderer. inverse_warp uses a single
    gather kernel (combined warp+projection for VR). soft_shift uses the
    forward-warp + z-buffer + hole-fill passes into a flat (H, 2W) SBS, which
    flat3d returns directly and VR projects through the map.
    """

    def __init__(self, src_w, src_h, projection, eye_distance_mm=DEFAULT_EYE_DISTANCE_MM,
                 hole_fill_mode=DEFAULT_HOLE_FILL_MODE, flat_fov_deg=DEFAULT_FLAT_FOV_DEG):
        if hole_fill_mode not in (HOLE_FILL_INVERSE_WARP, HOLE_FILL_SOFT_SHIFT):
            raise ValueError(f"GpuStereoRenderer: unsupported hole_fill {hole_fill_mode}")
        import cupy as cp

        self.cp = cp
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.projection = projection
        self.eye_distance_mm = float(eye_distance_mm)
        self.max_shift = np.float32(_max_disparity_pixels(self.src_w, self.eye_distance_mm))
        self._sh, self._sw = np.int32(self.src_h), np.int32(self.src_w)
        self._threads = 256
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

    def render(self, frame_rgb, depth):
        cp = self.cp
        # Normalize the (low-res) depth on the CPU -- cheap, and avoids cupy cccl.
        # soft_shift skips the depth blur and grows the foreground (see
        # near_from_depth / _dilate_near_fg) so the gap is bounded by clean
        # background; inverse_warp keeps the smooth.
        near = _dilate_near_fg(_normalize_near(depth)) if self._soft else _normalize_near(_smooth_depth(depth))
        h, w = near.shape
        if self._near_g is None or self._near_g.shape != (h, w):
            self._near_g = cp.empty((h, w), cp.float32)
        self._frame_g.set(np.ascontiguousarray(frame_rgb))
        self._near_g.set(np.ascontiguousarray(near))
        self._launch(self._frame_g, self._near_g)
        self._out_g.get(out=self._out_view)
        return self._out_view
