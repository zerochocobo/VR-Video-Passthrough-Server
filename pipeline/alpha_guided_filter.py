"""GPU guided alpha upsampling helpers."""
from __future__ import annotations

import numpy as np


_RESIZE_Y_TO_FLOAT_KERNEL = None
_RESIZE_FLOAT_KERNEL = None
_GUIDED_RECONSTRUCT_KERNEL = None
_BOX_FILTER_H_KERNEL = None
_BOX_FILTER_V_KERNEL = None


def _cupy():
    import cupy as cp

    return cp


def _ensure_box_filter_kernels(cp):
    global _BOX_FILTER_H_KERNEL, _BOX_FILTER_V_KERNEL
    if _BOX_FILTER_H_KERNEL is None:
        _BOX_FILTER_H_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void box_filter_h(
                const float* __restrict__ src,
                float* __restrict__ dst,
                int w,
                int h,
                int radius
            ) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= w || y >= h) return;
                int size = radius * 2 + 1;
                float sum = 0.f;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0) xx = 0;
                    if (xx >= w) xx = w - 1;
                    sum += src[y * w + xx];
                }
                dst[y * w + x] = sum / (float)size;
            }
            """,
            "box_filter_h",
        )
    if _BOX_FILTER_V_KERNEL is None:
        _BOX_FILTER_V_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void box_filter_v(
                const float* __restrict__ src,
                float* __restrict__ dst,
                int w,
                int h,
                int radius
            ) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= w || y >= h) return;
                int size = radius * 2 + 1;
                float sum = 0.f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = y + dy;
                    if (yy < 0) yy = 0;
                    if (yy >= h) yy = h - 1;
                    sum += src[yy * w + x];
                }
                dst[y * w + x] = sum / (float)size;
            }
            """,
            "box_filter_v",
        )
    return _BOX_FILTER_H_KERNEL, _BOX_FILTER_V_KERNEL


def _box_filter(image, radius: int):
    cp = _cupy()
    image = cp.ascontiguousarray(image.astype(cp.float32, copy=False))
    radius = max(0, int(radius))
    if radius <= 0:
        return image
    h, w = (int(v) for v in image.shape[:2])
    tmp = cp.empty_like(image)
    out = cp.empty_like(image)
    block = (32, 8, 1)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
    k_h, k_v = _ensure_box_filter_kernels(cp)
    k_h(grid, block, (image, tmp, np.int32(w), np.int32(h), np.int32(radius)))
    k_v(grid, block, (tmp, out, np.int32(w), np.int32(h), np.int32(radius)))
    return out


def _ensure_resize_float_kernel(cp):
    global _RESIZE_FLOAT_KERNEL
    if _RESIZE_FLOAT_KERNEL is None:
        _RESIZE_FLOAT_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void resize_float_bilinear(
                const float* __restrict__ src,
                float* __restrict__ dst,
                int src_w,
                int src_h,
                int dst_w,
                int dst_h
            ) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= dst_w || y >= dst_h) return;

                float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
                float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
                int x0 = (int)floorf(sx); if (x0 < 0) x0 = 0; if (x0 > src_w - 1) x0 = src_w - 1;
                int y0 = (int)floorf(sy); if (y0 < 0) y0 = 0; if (y0 > src_h - 1) y0 = src_h - 1;
                int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
                int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
                float dx = sx - floorf(sx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
                float dy = sy - floorf(sy); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;
                float v00 = src[y0 * src_w + x0];
                float v01 = src[y0 * src_w + x1];
                float v10 = src[y1 * src_w + x0];
                float v11 = src[y1 * src_w + x1];
                float v = (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                        +        dy  * ((1.f - dx) * v10 + dx * v11);
                dst[y * dst_w + x] = v;
            }
            """,
            "resize_float_bilinear",
        )
    return _RESIZE_FLOAT_KERNEL


def _resize_float(image, out_h: int, out_w: int):
    cp = _cupy()
    image = cp.ascontiguousarray(image.astype(cp.float32, copy=False))
    src_h, src_w = (int(v) for v in image.shape[:2])
    out_h, out_w = int(out_h), int(out_w)
    if src_h == out_h and src_w == out_w:
        return image
    out = cp.empty((out_h, out_w), dtype=cp.float32)
    block = (16, 16, 1)
    grid = ((out_w + block[0] - 1) // block[0], (out_h + block[1] - 1) // block[1], 1)
    _ensure_resize_float_kernel(cp)(
        grid,
        block,
        (
            image,
            out,
            np.int32(src_w),
            np.int32(src_h),
            np.int32(out_w),
            np.int32(out_h),
        ),
    )
    return out


def _ensure_resize_y_to_float_kernel(cp):
    global _RESIZE_Y_TO_FLOAT_KERNEL
    if _RESIZE_Y_TO_FLOAT_KERNEL is None:
        _RESIZE_Y_TO_FLOAT_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void resize_y_to_float(
                const unsigned char* __restrict__ src,
                float* __restrict__ dst,
                int src_w,
                int src_h,
                int dst_w,
                int dst_h
            ) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= dst_w || y >= dst_h) return;

                float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
                float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
                int x0 = (int)floorf(sx); if (x0 < 0) x0 = 0; if (x0 > src_w - 1) x0 = src_w - 1;
                int y0 = (int)floorf(sy); if (y0 < 0) y0 = 0; if (y0 > src_h - 1) y0 = src_h - 1;
                int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
                int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
                float dx = sx - floorf(sx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
                float dy = sy - floorf(sy); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;

                float v00 = (float)src[y0 * src_w + x0];
                float v01 = (float)src[y0 * src_w + x1];
                float v10 = (float)src[y1 * src_w + x0];
                float v11 = (float)src[y1 * src_w + x1];
                float v = (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                        +        dy  * ((1.f - dx) * v10 + dx * v11);
                dst[y * dst_w + x] = v * 0.00392156862745098f;
            }
            """,
            "resize_y_to_float",
        )
    return _RESIZE_Y_TO_FLOAT_KERNEL


def _ensure_guided_reconstruct_kernel(cp):
    global _GUIDED_RECONSTRUCT_KERNEL
    if _GUIDED_RECONSTRUCT_KERNEL is None:
        _GUIDED_RECONSTRUCT_KERNEL = cp.RawKernel(
            r"""
            __device__ float sample_coeff(
                const float* __restrict__ src,
                int src_w,
                int src_h,
                int out_w,
                int out_h,
                int x,
                int y
            ) {
                float sx = ((float)x + 0.5f) * (float)src_w / (float)out_w - 0.5f;
                float sy = ((float)y + 0.5f) * (float)src_h / (float)out_h - 0.5f;
                int x0 = (int)floorf(sx); if (x0 < 0) x0 = 0; if (x0 > src_w - 1) x0 = src_w - 1;
                int y0 = (int)floorf(sy); if (y0 < 0) y0 = 0; if (y0 > src_h - 1) y0 = src_h - 1;
                int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
                int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
                float dx = sx - floorf(sx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
                float dy = sy - floorf(sy); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;
                float v00 = src[y0 * src_w + x0];
                float v01 = src[y0 * src_w + x1];
                float v10 = src[y1 * src_w + x0];
                float v11 = src[y1 * src_w + x1];
                return (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                     +        dy  * ((1.f - dx) * v10 + dx * v11);
            }

            __device__ float sample_y_norm(
                const unsigned char* __restrict__ guide_y,
                int guide_w,
                int guide_h,
                int out_w,
                int out_h,
                int x,
                int y
            ) {
                float sx = ((float)x + 0.5f) * (float)guide_w / (float)out_w - 0.5f;
                float sy = ((float)y + 0.5f) * (float)guide_h / (float)out_h - 0.5f;
                int x0 = (int)floorf(sx); if (x0 < 0) x0 = 0; if (x0 > guide_w - 1) x0 = guide_w - 1;
                int y0 = (int)floorf(sy); if (y0 < 0) y0 = 0; if (y0 > guide_h - 1) y0 = guide_h - 1;
                int x1 = x0 + 1; if (x1 > guide_w - 1) x1 = guide_w - 1;
                int y1 = y0 + 1; if (y1 > guide_h - 1) y1 = guide_h - 1;
                float dx = sx - floorf(sx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
                float dy = sy - floorf(sy); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;
                float v00 = (float)guide_y[y0 * guide_w + x0];
                float v01 = (float)guide_y[y0 * guide_w + x1];
                float v10 = (float)guide_y[y1 * guide_w + x0];
                float v11 = (float)guide_y[y1 * guide_w + x1];
                float v = (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                        +        dy  * ((1.f - dx) * v10 + dx * v11);
                return v * 0.00392156862745098f;
            }

            extern "C" __global__
            void guided_reconstruct(
                const float* __restrict__ coeff_a,
                const float* __restrict__ coeff_b,
                const unsigned char* __restrict__ guide_y,
                float* __restrict__ out,
                int coeff_w,
                int coeff_h,
                int guide_w,
                int guide_h,
                int out_w,
                int out_h
            ) {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= out_w || y >= out_h) return;
                float a = sample_coeff(coeff_a, coeff_w, coeff_h, out_w, out_h, x, y);
                float b = sample_coeff(coeff_b, coeff_w, coeff_h, out_w, out_h, x, y);
                float I = sample_y_norm(guide_y, guide_w, guide_h, out_w, out_h, x, y);
                float q = a * I + b;
                if (q < 0.f) q = 0.f;
                if (q > 1.f) q = 1.f;
                out[y * out_w + x] = q;
            }
            """,
            "guided_reconstruct",
        )
    return _GUIDED_RECONSTRUCT_KERNEL


def _resize_y_to_float(guide_y, out_h: int, out_w: int):
    cp = _cupy()
    guide_y = cp.ascontiguousarray(guide_y)
    src_h, src_w = (int(v) for v in guide_y.shape[:2])
    out = cp.empty((int(out_h), int(out_w)), dtype=cp.float32)
    block = (16, 16, 1)
    grid = ((int(out_w) + block[0] - 1) // block[0], (int(out_h) + block[1] - 1) // block[1], 1)
    _ensure_resize_y_to_float_kernel(cp)(
        grid,
        block,
        (
            guide_y,
            out,
            np.int32(src_w),
            np.int32(src_h),
            np.int32(out_w),
            np.int32(out_h),
        ),
    )
    return out


def _guided_reconstruct(coeff_a, coeff_b, guide_y, out_h: int, out_w: int):
    cp = _cupy()
    coeff_a = cp.ascontiguousarray(coeff_a.astype(cp.float32, copy=False))
    coeff_b = cp.ascontiguousarray(coeff_b.astype(cp.float32, copy=False))
    guide_y = cp.ascontiguousarray(guide_y)
    coeff_h, coeff_w = (int(v) for v in coeff_a.shape[:2])
    guide_h, guide_w = (int(v) for v in guide_y.shape[:2])
    out = cp.empty((int(out_h), int(out_w)), dtype=cp.float32)
    block = (16, 16, 1)
    grid = ((int(out_w) + block[0] - 1) // block[0], (int(out_h) + block[1] - 1) // block[1], 1)
    _ensure_guided_reconstruct_kernel(cp)(
        grid,
        block,
        (
            coeff_a,
            coeff_b,
            guide_y,
            out,
            np.int32(coeff_w),
            np.int32(coeff_h),
            np.int32(guide_w),
            np.int32(guide_h),
            np.int32(out_w),
            np.int32(out_h),
        ),
    )
    return out


def fast_guided_filter_upsample(
    alpha_lr_gpu,
    guide_y_hr_gpu,
    radius: int = 8,
    eps: float = 0.0025,
    fullres_scale: float = 1.0,
    support_floor: float = 0.02,
    max_delta: float = 0.08,
    band_lo: float = 0.05,
    band_hi: float = 0.95,
):
    """Refine low-res alpha with a fast guided filter using NV12 luma.

    `alpha_lr_gpu` is a low-res alpha array on CPU or GPU in [0, 1].
    `guide_y_hr_gpu` is the uploaded NV12 Y plane on GPU. The result is a
    CuPy float32 array; at `fullres_scale=1.0` it has source resolution.
    """
    cp = _cupy()
    alpha = cp.asarray(alpha_lr_gpu, dtype=cp.float32)
    if alpha.ndim != 2:
        raise ValueError(f"alpha_lr_gpu must be 2D, got shape={tuple(alpha.shape)}")
    guide_y = cp.asarray(guide_y_hr_gpu)
    if guide_y.ndim != 2:
        raise ValueError(f"guide_y_hr_gpu must be 2D, got shape={tuple(guide_y.shape)}")
    if guide_y.dtype != cp.uint8:
        guide_y = cp.clip(guide_y, 0, 255).astype(cp.uint8, copy=False)

    alpha = cp.ascontiguousarray(cp.clip(alpha, 0.0, 1.0))
    ah, aw = (int(v) for v in alpha.shape[:2])
    gh, gw = (int(v) for v in guide_y.shape[:2])
    if ah <= 0 or aw <= 0 or gh <= 0 or gw <= 0:
        raise ValueError(f"invalid alpha/guide shape alpha={alpha.shape} guide={guide_y.shape}")

    scale = max(0.05, min(1.0, float(fullres_scale)))
    coeff_h = max(2, int(round(ah * scale)))
    coeff_w = max(2, int(round(aw * scale)))
    alpha_coeff = _resize_float(alpha, coeff_h, coeff_w)
    guide_lr = _resize_y_to_float(guide_y, coeff_h, coeff_w)
    radius = max(0, int(radius))
    eps = max(1.0e-8, float(eps))

    mean_i = _box_filter(guide_lr, radius)
    mean_p = _box_filter(alpha_coeff, radius)
    corr_i = _box_filter(guide_lr * guide_lr, radius)
    corr_ip = _box_filter(guide_lr * alpha_coeff, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    coeff_a = cov_ip / (var_i + eps)
    coeff_b = mean_p - coeff_a * mean_i
    mean_a = _box_filter(coeff_a, radius)
    mean_b = _box_filter(coeff_b, radius)

    out_h = max(ah, int(round(gh * scale)))
    out_w = max(aw, int(round(gw * scale)))
    refined = _guided_reconstruct(mean_a, mean_b, guide_y, out_h, out_w)

    base = _resize_float(alpha, out_h, out_w)
    band_lo = max(0.0, min(1.0, float(band_lo)))
    band_hi = max(0.0, min(1.0, float(band_hi)))
    if band_hi < band_lo:
        band_lo, band_hi = band_hi, band_lo
    if band_lo > 0.0 or band_hi < 1.0:
        refined = cp.where((base < band_lo) | (base > band_hi), base, refined)

    support_floor = max(0.0, min(1.0, float(support_floor)))
    if support_floor > 0.0:
        refined = cp.where(base < support_floor, cp.float32(0.0), refined)
    max_delta = float(max_delta)
    if max_delta >= 0.0:
        refined = cp.minimum(refined, cp.clip(base + cp.float32(max_delta), 0.0, 1.0))
    return cp.clip(refined, 0.0, 1.0)
