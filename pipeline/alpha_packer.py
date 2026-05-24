"""GPU alpha-packed passthrough helper.

This mirrors the offline DeoVR alpha-packer experiment: source SBS
half-equirect 180 frames are projected to fisheye SBS, then the fisheye-space
alpha mask is packed into the visible frame using the six-block layout.
"""
from __future__ import annotations

import math

import numpy as np

import config
from pipeline.light_match import LIGHT_MATCH_DEVICE_SRC
from utils.logger import get

log = get("alpha_packer")

PACK_SCALE = 0.4
FISHEYE_RADIUS_SCALE = 1.0
IPD_METERS = 0.063


def _ceil_even(value: float) -> int:
    out = int(math.ceil(value))
    return out if (out & 1) == 0 else out + 1


def alpha_2d_disparity_px(output_w: int) -> float:
    """Convert configured flat-2D alpha depth in meters to stereo disparity px."""
    eye_w = max(2, int(output_w) // 2)
    distance_m = max(0.1, float(config.ALPHA_2D_DISTANCE_M))
    angle_rad = 2.0 * math.atan(IPD_METERS / (2.0 * distance_m))
    return max(0.0, float(round((angle_rad / math.pi) * eye_w)))


def is_sbs_vr_size(w: int, h: int) -> bool:
    return int(w) > 0 and int(h) > 0 and int(w) >= 2 * int(h)


def alpha_output_size(src_w: int, src_h: int) -> tuple[int, int]:
    """Return alpha passthrough output size for an already-scaled source frame."""
    src_w = max(2, int(src_w) & ~1)
    src_h = max(2, int(src_h) & ~1)
    if is_sbs_vr_size(src_w, src_h) or not config.ALPHA_2D_ENABLE:
        return src_w, src_h
    if str(config.ALPHA_2D_PROJECTION).lower() == "flat3d":
        safe_w = max(0.01, min(1.0, float(config.ALPHA_2D_FLAT3D_SAFE_W)))
        safe_h = max(0.01, min(1.0, float(config.ALPHA_2D_FLAT3D_SAFE_H)))
        max_eye = max(2, int(config.ALPHA_2D_MAX_EYE_SIZE) & ~1)
        eye = _ceil_even(max(src_w / safe_w, src_h / safe_h))
        eye = max(2, min(max_eye, eye))
        return eye * 2, eye
    fov = max(1.0, min(179.0, float(config.ALPHA_2D_FOV)))
    eye = int(math.ceil(max(src_w, src_h) * 180.0 / fov))
    if eye & 1:
        eye += 1
    max_eye = max(2, int(config.ALPHA_2D_MAX_EYE_SIZE) & ~1)
    eye = min(max_eye, max(2, eye))
    return eye * 2, eye


class AlphaPacker:
    """Convert uploaded NV12 + alpha to DeoVR-style alpha-packed fisheye NV12."""

    _KERNEL_SRC = LIGHT_MATCH_DEVICE_SRC + r"""
    __device__ float adjust_alpha(float a, float cutoff, int hard_edge, float contrast) {
        a = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
        if (contrast != 1.f) {
            a = (a - 0.5f) * contrast + 0.5f;
            a = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
        }
        if (cutoff > 0.f) {
            if (hard_edge) {
                a = a >= cutoff ? 1.f : 0.f;
            } else if (a < cutoff) {
                a = 0.f;
            }
        }
        return a;
    }

    __device__ float sample_alpha_lr(
        const float* __restrict__ alpha_lr,
        int aw, int ah,
        int out_w, int out_h,
        int x, int y
    ) {
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= out_w) x = out_w - 1;
        if (y >= out_h) y = out_h - 1;
        float scale_x = (float)aw / (float)out_w;
        float scale_y = (float)ah / (float)out_h;
        float fx = ((float)x + 0.5f) * scale_x - 0.5f;
        float fy = ((float)y + 0.5f) * scale_y - 0.5f;
        int x0 = (int)floorf(fx); if (x0 < 0) x0 = 0; if (x0 > aw - 1) x0 = aw - 1;
        int y0 = (int)floorf(fy); if (y0 < 0) y0 = 0; if (y0 > ah - 1) y0 = ah - 1;
        int x1 = x0 + 1; if (x1 > aw - 1) x1 = aw - 1;
        int y1 = y0 + 1; if (y1 > ah - 1) y1 = ah - 1;
        float dx = fx - floorf(fx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
        float dy = fy - floorf(fy); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;
        float a00 = alpha_lr[y0 * aw + x0];
        float a01 = alpha_lr[y0 * aw + x1];
        float a10 = alpha_lr[y1 * aw + x0];
        float a11 = alpha_lr[y1 * aw + x1];
        float a = (1.f - dy) * ((1.f - dx) * a00 + dx * a01)
                +        dy  * ((1.f - dx) * a10 + dx * a11);
        return a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
    }

    __device__ bool fisheye_to_half_equirect(
        int x, int y,
        int out_w, int out_h,
        float radius_scale,
        float* src_x,
        float* src_y
    ) {
        int eye_w = out_w >> 1;
        int eye = x >= eye_w ? 1 : 0;
        float lx = (float)(x - eye * eye_w) + 0.5f;
        float ly = (float)y + 0.5f;
        float cx = (float)eye_w * 0.5f;
        float cy = (float)out_h * 0.5f;
        float radius = fminf((float)eye_w, (float)out_h) * 0.5f * radius_scale;
        float nx = (lx - cx) / radius;
        float ny = (ly - cy) / radius;
        float rr = sqrtf(nx * nx + ny * ny);
        if (rr > 1.f) {
            return false;
        }

        // Match FFmpeg v360=input=hequirect:output=fisheye for the visible
        // fisheye circle, while keeping the square corners black for alpha blocks.
        float ff_theta = 1.5707963267948966f * (1.f - rr);
        float ff_phi = atan2f(ny, nx);
        float cos_t = cosf(ff_theta);
        float dir_x = cos_t * cosf(ff_phi);
        float dir_y = cos_t * sinf(ff_phi);
        float dir_z = sinf(ff_theta);
        float src_phi = atan2f(dir_x, dir_z) / 1.5707963267948966f;
        float src_theta = asinf(dir_y) / 1.5707963267948966f;
        float u = (0.5f * src_phi + 0.5f) * (float)(eye_w - 1);
        float v = (0.5f * src_theta + 0.5f) * (float)(out_h - 1);
        u = u < 0.f ? 0.f : (u > (float)(eye_w - 1) ? (float)(eye_w - 1) : u);
        v = v < 0.f ? 0.f : (v > (float)(out_h - 1) ? (float)(out_h - 1) : v);
        *src_x = u + (float)(eye * eye_w);
        *src_y = v;
        return true;
    }

    __device__ bool fisheye_to_flat_2d(
        int x, int y,
        int out_w, int out_h,
        int src_w, int src_h,
        float radius_scale,
        float fov_rad,
        float disparity_px,
        float* src_x,
        float* src_y
    ) {
        int eye_w = out_w >> 1;
        int eye = x >= eye_w ? 1 : 0;
        float lx = (float)(x - eye * eye_w) + 0.5f;
        float ly = (float)y + 0.5f;
        float cx = (float)eye_w * 0.5f;
        float cy = (float)out_h * 0.5f;
        float radius = fminf((float)eye_w, (float)out_h) * 0.5f * radius_scale;
        float nx = (lx - cx) / radius;
        float ny = (ly - cy) / radius;
        float rr = sqrtf(nx * nx + ny * ny);
        if (rr > 1.f) {
            return false;
        }

        float theta = rr * 1.5707963267948966f;
        float az = atan2f(-ny, nx);
        float sin_t = sinf(theta);
        float dir_x = sin_t * cosf(az);
        float dir_y = sin_t * sinf(az);
        float dir_z = cosf(theta);
        if (dir_z <= 1.0e-6f) {
            return false;
        }

        float plane_scale = tanf(fov_rad * 0.5f);
        float px = dir_x / dir_z / plane_scale;
        float py = -dir_y / dir_z / plane_scale;
        if (px < -1.f || px > 1.f || py < -1.f || py > 1.f) {
            return false;
        }

        float canvas = (float)(src_w > src_h ? src_w : src_h);
        float x0 = (canvas - (float)src_w) * 0.5f;
        float y0 = (canvas - (float)src_h) * 0.5f;
        float u = (px * 0.5f + 0.5f) * canvas - x0;
        float v = (py * 0.5f + 0.5f) * canvas - y0;
        float eye_shift = eye == 0 ? (disparity_px * 0.5f) : (-disparity_px * 0.5f);
        u -= eye_shift;
        if (u < 0.f || u > (float)(src_w - 1) || v < 0.f || v > (float)(src_h - 1)) {
            return false;
        }
        *src_x = u;
        *src_y = v;
        return true;
    }

    __device__ bool flat3d_to_source(
        int x, int y,
        int out_w, int out_h,
        int src_w, int src_h,
        float safe_w_frac,
        float safe_h_frac,
        float disparity_px,
        float* src_x,
        float* src_y
    ) {
        int eye_w = out_w >> 1;
        int eye = x >= eye_w ? 1 : 0;
        float lx = (float)(x - eye * eye_w) + 0.5f;
        float ly = (float)y + 0.5f;
        float safe_w = (float)eye_w * safe_w_frac;
        float safe_h = (float)out_h * safe_h_frac;
        if (safe_w < 1.f || safe_h < 1.f) return false;
        float scale = fminf(safe_w / (float)src_w, safe_h / (float)src_h);
        if (scale <= 0.f) return false;
        float draw_w = (float)src_w * scale;
        float draw_h = (float)src_h * scale;
        float left = ((float)eye_w - draw_w) * 0.5f;
        float top = ((float)out_h - draw_h) * 0.5f;
        float eye_shift = eye == 0 ? (disparity_px * 0.5f) : (-disparity_px * 0.5f);
        float u = (lx - left - eye_shift) / scale;
        float v = (ly - top) / scale;
        if (u < 0.f || u > (float)(src_w - 1) || v < 0.f || v > (float)(src_h - 1)) {
            return false;
        }
        *src_x = u;
        *src_y = v;
        return true;
    }

    __device__ unsigned char sample_y_bilinear(
        const unsigned char* __restrict__ src_nv12,
        int w, int h,
        float sx, float sy
    ) {
        int x0 = (int)floorf(sx); if (x0 < 0) x0 = 0; if (x0 > w - 1) x0 = w - 1;
        int y0 = (int)floorf(sy); if (y0 < 0) y0 = 0; if (y0 > h - 1) y0 = h - 1;
        int x1 = x0 + 1; if (x1 > w - 1) x1 = w - 1;
        int y1 = y0 + 1; if (y1 > h - 1) y1 = h - 1;
        float dx = sx - floorf(sx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
        float dy = sy - floorf(sy); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;
        float v00 = (float)src_nv12[y0 * w + x0];
        float v01 = (float)src_nv12[y0 * w + x1];
        float v10 = (float)src_nv12[y1 * w + x0];
        float v11 = (float)src_nv12[y1 * w + x1];
        float out = (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                  +        dy  * ((1.f - dx) * v10 + dx * v11);
        return (unsigned char)(out + 0.5f);
    }

    __device__ void sample_uv_nearest(
        const unsigned char* __restrict__ src_nv12,
        int w, int h,
        float sx, float sy,
        unsigned char* u,
        unsigned char* v
    ) {
        int ux = ((int)floorf(sx)) & ~1;
        int uy = ((int)floorf(sy)) >> 1;
        if (ux < 0) ux = 0;
        if (ux > w - 2) ux = w - 2;
        if (uy < 0) uy = 0;
        if (uy > (h >> 1) - 1) uy = (h >> 1) - 1;
        int idx = w * h + uy * w + ux;
        *u = src_nv12[idx];
        *v = src_nv12[idx + 1];
    }

    __device__ void rgb_to_yuv_limited(float r, float g, float b, unsigned char* y, unsigned char* u, unsigned char* v) {
        float yf = 16.f + 0.257f * r + 0.504f * g + 0.098f * b;
        float uf = 128.f - 0.148f * r - 0.291f * g + 0.439f * b;
        float vf = 128.f + 0.439f * r - 0.368f * g - 0.071f * b;
        yf = yf < 0.f ? 0.f : (yf > 255.f ? 255.f : yf);
        uf = uf < 0.f ? 0.f : (uf > 255.f ? 255.f : uf);
        vf = vf < 0.f ? 0.f : (vf > 255.f ? 255.f : vf);
        *y = (unsigned char)(yf + 0.5f);
        *u = (unsigned char)(uf + 0.5f);
        *v = (unsigned char)(vf + 0.5f);
    }

    __device__ bool is_alpha_layout_pixel(
        int x, int y,
        int out_w, int out_h,
        int alpha_w, int alpha_h
    ) {
        int half_w = alpha_w >> 1;
        int half_h = alpha_h >> 1;
        int quarter_w = alpha_w >> 2;
        int x_topleft = (out_w >> 1) - (half_w >> 1);
        int y_bottomleft = out_h - half_h;
        int x2_topleft = out_w - quarter_w;
        int y2_topleft = out_h - half_h;

        if (x >= x_topleft && x < x_topleft + half_w && y >= y_bottomleft && y < y_bottomleft + half_h) return true;
        if (x >= x_topleft && x < x_topleft + half_w && y >= 0 && y < half_h) return true;
        if (x >= x2_topleft && x < x2_topleft + quarter_w && y >= y2_topleft && y < y2_topleft + half_h) return true;
        if (x >= 0 && x < quarter_w && y >= y_bottomleft && y < y_bottomleft + half_h) return true;
        if (x >= x2_topleft && x < x2_topleft + half_w && y >= 0 && y < half_h) return true;
        if (x >= 0 && x < quarter_w && y >= 0 && y < half_h) return true;
        return false;
    }

    __device__ bool alpha_layout_source(
        int x, int y,
        int out_w, int out_h,
        int alpha_w, int alpha_h,
        int* src_ax,
        int* src_ay
    ) {
        int half_w = alpha_w >> 1;
        int half_h = alpha_h >> 1;
        int quarter_w = alpha_w >> 2;
        int right2_x = alpha_w - quarter_w;
        int x_topleft = (out_w >> 1) - (half_w >> 1);
        int y_bottomleft = out_h - half_h;
        int x2_topleft = out_w - quarter_w;
        int y2_topleft = out_h - half_h;
        *src_ax = -1;
        *src_ay = -1;

        if (x >= x_topleft && x < x_topleft + half_w && y >= y_bottomleft && y < y_bottomleft + half_h) {
            *src_ax = x - x_topleft;
            *src_ay = y - y_bottomleft;
        } else if (x >= x_topleft && x < x_topleft + half_w && y >= 0 && y < half_h) {
            *src_ax = x - x_topleft;
            *src_ay = y + half_h;
        } else if (x >= x2_topleft && x < x2_topleft + quarter_w && y >= y2_topleft && y < y2_topleft + half_h) {
            *src_ax = (x - x2_topleft) + half_w;
            *src_ay = y - y2_topleft;
        } else if (x >= 0 && x < quarter_w && y >= y_bottomleft && y < y_bottomleft + half_h) {
            *src_ax = x + right2_x;
            *src_ay = y - y_bottomleft;
        } else if (x >= x2_topleft && x < x2_topleft + half_w && y >= 0 && y < half_h) {
            *src_ax = (x - x2_topleft) + half_w;
            *src_ay = y + half_h;
        } else if (x >= 0 && x < quarter_w && y >= 0 && y < half_h) {
            *src_ax = x + right2_x;
            *src_ay = y + half_h;
        }
        return *src_ax >= 0 && *src_ay >= 0;
    }

    __device__ unsigned char alpha_layout_mask_at(
        const unsigned char* __restrict__ fisheye_alpha,
        int x, int y,
        int out_w, int out_h,
        int alpha_w, int alpha_h
    ) {
        int src_ax = -1;
        int src_ay = -1;
        if (!alpha_layout_source(x, y, out_w, out_h, alpha_w, alpha_h, &src_ax, &src_ay)) {
            return 0;
        }
        int fisheye_x = src_ax * out_w / alpha_w;
        int fisheye_y = src_ay * out_h / alpha_h;
        return fisheye_alpha[fisheye_y * out_w + fisheye_x];
    }

    extern "C" __global__
    void project_fisheye_nv12_alpha(
        const unsigned char* __restrict__ src_nv12,
        const float* __restrict__ alpha_lr,
        int out_w, int out_h,
        int aw, int ah,
        float radius_scale,
        float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
        unsigned char* __restrict__ out_nv12,
        unsigned char* __restrict__ fisheye_alpha,
        const float* __restrict__ light_coeffs,
        const unsigned char* __restrict__ light_gamma_lut,
        int light_identity
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= out_w || y >= out_h) return;

        int y_idx = y * out_w + x;
        float src_x = 0.f;
        float src_y = 0.f;
        bool inside = fisheye_to_half_equirect(x, y, out_w, out_h, radius_scale, &src_x, &src_y);
        unsigned char yv = inside ? sample_y_bilinear(src_nv12, out_w, out_h, src_x, src_y) : (unsigned char)16;
        unsigned char uv_u = 128;
        unsigned char uv_v = 128;
        if (inside) {
            sample_uv_nearest(src_nv12, out_w, out_h, src_x, src_y, &uv_u, &uv_v);
            float yy = (float)yv;
            float uu = (float)uv_u;
            float vv = (float)uv_v;
            apply_light_match(&yy, &uu, &vv, light_coeffs, light_gamma_lut, light_identity);
            yv = (unsigned char)(yy + 0.5f);
            uv_u = (unsigned char)(uu + 0.5f);
            uv_v = (unsigned char)(vv + 0.5f);
        }
        out_nv12[y_idx] = yv;

        float a = 0.f;
        if (inside) {
            a = adjust_alpha(
                sample_alpha_lr(alpha_lr, aw, ah, out_w, out_h, (int)src_x, (int)src_y),
                alpha_cutoff, alpha_hard_edge, alpha_contrast
            );
        }
        fisheye_alpha[y_idx] = (unsigned char)(a * 255.f + 0.5f);

        if (((x | y) & 1) == 0) {
            int uv_idx = out_w * out_h + (y >> 1) * out_w + x;
            out_nv12[uv_idx] = uv_u;
            out_nv12[uv_idx + 1] = uv_v;
        }
    }

    extern "C" __global__
    void project_flat2d_fisheye_nv12_alpha(
        const unsigned char* __restrict__ src_nv12,
        const float* __restrict__ alpha_lr,
        int src_w, int src_h,
        int out_w, int out_h,
        int aw, int ah,
        float radius_scale,
        float fov_rad,
        float disparity_px,
        float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
        unsigned char* __restrict__ out_nv12,
        unsigned char* __restrict__ fisheye_alpha,
        const float* __restrict__ light_coeffs,
        const unsigned char* __restrict__ light_gamma_lut,
        int light_identity
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= out_w || y >= out_h) return;

        int y_idx = y * out_w + x;
        float src_x = 0.f;
        float src_y = 0.f;
        bool inside = fisheye_to_flat_2d(
            x, y, out_w, out_h, src_w, src_h, radius_scale, fov_rad, disparity_px, &src_x, &src_y
        );
        unsigned char yv = inside ? sample_y_bilinear(src_nv12, src_w, src_h, src_x, src_y) : (unsigned char)16;
        unsigned char uv_u = 128;
        unsigned char uv_v = 128;
        if (inside) {
            sample_uv_nearest(src_nv12, src_w, src_h, src_x, src_y, &uv_u, &uv_v);
            float yy = (float)yv;
            float uu = (float)uv_u;
            float vv = (float)uv_v;
            apply_light_match(&yy, &uu, &vv, light_coeffs, light_gamma_lut, light_identity);
            yv = (unsigned char)(yy + 0.5f);
            uv_u = (unsigned char)(uu + 0.5f);
            uv_v = (unsigned char)(vv + 0.5f);
        }
        out_nv12[y_idx] = yv;

        float a = 0.f;
        if (inside) {
            a = adjust_alpha(
                sample_alpha_lr(alpha_lr, aw, ah, src_w, src_h, (int)src_x, (int)src_y),
                alpha_cutoff, alpha_hard_edge, alpha_contrast
            );
        }
        fisheye_alpha[y_idx] = (unsigned char)(a * 255.f + 0.5f);

        if (((x | y) & 1) == 0) {
            int uv_idx = out_w * out_h + (y >> 1) * out_w + x;
            out_nv12[uv_idx] = uv_u;
            out_nv12[uv_idx + 1] = uv_v;
        }
    }

    extern "C" __global__
    void project_flat2d_3d_nv12_alpha(
        const unsigned char* __restrict__ src_nv12,
        const float* __restrict__ alpha_lr,
        int src_w, int src_h,
        int out_w, int out_h,
        int aw, int ah,
        float safe_w_frac,
        float safe_h_frac,
        float disparity_px,
        float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
        unsigned char* __restrict__ out_nv12,
        unsigned char* __restrict__ fisheye_alpha,
        const float* __restrict__ light_coeffs,
        const unsigned char* __restrict__ light_gamma_lut,
        int light_identity
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= out_w || y >= out_h) return;

        int y_idx = y * out_w + x;
        float src_x = 0.f;
        float src_y = 0.f;
        bool inside = flat3d_to_source(
            x, y, out_w, out_h, src_w, src_h, safe_w_frac, safe_h_frac, disparity_px, &src_x, &src_y
        );
        unsigned char yv = inside ? sample_y_bilinear(src_nv12, src_w, src_h, src_x, src_y) : (unsigned char)16;
        unsigned char uv_u = 128;
        unsigned char uv_v = 128;
        if (inside) {
            sample_uv_nearest(src_nv12, src_w, src_h, src_x, src_y, &uv_u, &uv_v);
            float yy = (float)yv;
            float uu = (float)uv_u;
            float vv = (float)uv_v;
            apply_light_match(&yy, &uu, &vv, light_coeffs, light_gamma_lut, light_identity);
            yv = (unsigned char)(yy + 0.5f);
            uv_u = (unsigned char)(uu + 0.5f);
            uv_v = (unsigned char)(vv + 0.5f);
        }
        out_nv12[y_idx] = yv;

        float a = 0.f;
        if (inside) {
            a = adjust_alpha(
                sample_alpha_lr(alpha_lr, aw, ah, src_w, src_h, (int)src_x, (int)src_y),
                alpha_cutoff, alpha_hard_edge, alpha_contrast
            );
        }
        fisheye_alpha[y_idx] = (unsigned char)(a * 255.f + 0.5f);

        if (((x | y) & 1) == 0) {
            int uv_idx = out_w * out_h + (y >> 1) * out_w + x;
            out_nv12[uv_idx] = uv_u;
            out_nv12[uv_idx + 1] = uv_v;
        }
    }

    extern "C" __global__
    void blend_projected_overlay_to_fisheye(
        unsigned char* __restrict__ out_nv12,
        unsigned char* __restrict__ fisheye_alpha,
        int out_w, int out_h,
        int alpha_w, int alpha_h,
        float radius_scale,
        const unsigned char* __restrict__ rgba,
        int overlay_w, int overlay_h,
        int dst_x, int dst_y
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= out_w || y >= out_h) return;
        if (is_alpha_layout_pixel(x, y, out_w, out_h, alpha_w, alpha_h)) return;

        float src_x = 0.f;
        float src_y = 0.f;
        if (!fisheye_to_half_equirect(x, y, out_w, out_h, radius_scale, &src_x, &src_y)) return;
        float ox_f = src_x - (float)dst_x;
        float oy_f = src_y - (float)dst_y;
        int ox = (int)floorf(ox_f + 0.5f);
        int oy = (int)floorf(oy_f + 0.5f);
        if (ox < 0 || oy < 0 || ox >= overlay_w || oy >= overlay_h) return;
        int oi = (oy * overlay_w + ox) * 4;
        float a = rgba[oi + 3] / 255.0f;
        if (a <= 0.0f) return;

        float r = rgba[oi + 0];
        float g = rgba[oi + 1];
        float b = rgba[oi + 2];
        unsigned char yy, uu, vv;
        rgb_to_yuv_limited(r, g, b, &yy, &uu, &vv);

        int yi = y * out_w + x;
        out_nv12[yi] = (unsigned char)(out_nv12[yi] * (1.0f - a) + yy * a + 0.5f);
        unsigned char alpha_v = (unsigned char)(a * 255.f + 0.5f);
        if (fisheye_alpha[yi] < alpha_v) {
            fisheye_alpha[yi] = alpha_v;
        }

        if (((x | y) & 1) == 0) {
            int uv_idx = out_w * out_h + (y >> 1) * out_w + x;
            out_nv12[uv_idx] = (unsigned char)(out_nv12[uv_idx] * (1.0f - a) + uu * a + 0.5f);
            out_nv12[uv_idx + 1] = (unsigned char)(out_nv12[uv_idx + 1] * (1.0f - a) + vv * a + 0.5f);
        }
    }

    extern "C" __global__
    void overlay_alpha_packer_layout(
        const unsigned char* __restrict__ fisheye_alpha,
        int out_w, int out_h,
        int alpha_w, int alpha_h,
        unsigned char* __restrict__ out_nv12
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= out_w || y >= out_h) return;

        int src_ax = -1;
        int src_ay = -1;

        if (alpha_layout_source(x, y, out_w, out_h, alpha_w, alpha_h, &src_ax, &src_ay)) {
            int fisheye_x = src_ax * out_w / alpha_w;
            int fisheye_y = src_ay * out_h / alpha_h;
            unsigned char mask = fisheye_alpha[fisheye_y * out_w + fisheye_x];
            if (mask == 0) {
                return;
            }
            unsigned char yv = 16;
            unsigned char uv_u = 128;
            unsigned char uv_v = 128;
            rgb_to_yuv_limited((float)mask, 0.f, 0.f, &yv, &uv_u, &uv_v);
            out_nv12[y * out_w + x] = yv;
            if (((x | y) & 1) == 0) {
                unsigned char m01 = (x + 1 < out_w) ? alpha_layout_mask_at(fisheye_alpha, x + 1, y, out_w, out_h, alpha_w, alpha_h) : 0;
                unsigned char m10 = (y + 1 < out_h) ? alpha_layout_mask_at(fisheye_alpha, x, y + 1, out_w, out_h, alpha_w, alpha_h) : 0;
                unsigned char m11 = (x + 1 < out_w && y + 1 < out_h) ? alpha_layout_mask_at(fisheye_alpha, x + 1, y + 1, out_w, out_h, alpha_w, alpha_h) : 0;
                unsigned char uv_mask = mask;
                if (m01 > uv_mask) uv_mask = m01;
                if (m10 > uv_mask) uv_mask = m10;
                if (m11 > uv_mask) uv_mask = m11;
                rgb_to_yuv_limited((float)uv_mask, 0.f, 0.f, &yv, &uv_u, &uv_v);
                int uv_idx = out_w * out_h + (y >> 1) * out_w + x;
                out_nv12[uv_idx] = uv_u;
                out_nv12[uv_idx + 1] = uv_v;
            }
        }
    }
    """

    def __init__(
        self,
        matter,
        scale: float = PACK_SCALE,
        blocks_x: int = 3,
        blocks_y: int = 2,
        radius_scale: float = FISHEYE_RADIUS_SCALE,
        alpha_cutoff: float | None = None,
        alpha_hard_edge: bool | None = None,
        alpha_contrast: float | None = None,
    ) -> None:
        import cupy as cp

        self.matter = matter
        self.scale = float(scale)
        self.blocks_x = int(blocks_x)
        self.blocks_y = int(blocks_y)
        self.radius_scale = float(radius_scale)
        self.alpha_cutoff = config.ALPHA_CUTOFF if alpha_cutoff is None else float(alpha_cutoff)
        self.alpha_hard_edge = config.ALPHA_HARD_EDGE if alpha_hard_edge is None else bool(alpha_hard_edge)
        self.alpha_contrast = config.ALPHA_CONTRAST if alpha_contrast is None else float(alpha_contrast)
        self._cp = cp
        self._project_kernel = cp.RawKernel(self._KERNEL_SRC, "project_fisheye_nv12_alpha")
        self._project_flat2d_kernel = cp.RawKernel(self._KERNEL_SRC, "project_flat2d_fisheye_nv12_alpha")
        self._project_flat3d_kernel = cp.RawKernel(self._KERNEL_SRC, "project_flat2d_3d_nv12_alpha")
        self._overlay_kernel = cp.RawKernel(self._KERNEL_SRC, "overlay_alpha_packer_layout")
        self._blend_projected_overlay_kernel = cp.RawKernel(self._KERNEL_SRC, "blend_projected_overlay_to_fisheye")
        self._g_alpha = None
        self._g_fisheye_alpha = None
        self._g_overlay = None

    def output_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        return alpha_output_size(src_w, src_h)

    @staticmethod
    def projection_mode_static(src_w: int, src_h: int) -> str:
        if is_sbs_vr_size(src_w, src_h) or not config.ALPHA_2D_ENABLE:
            return "sbs_half_equirect"
        if str(config.ALPHA_2D_PROJECTION).lower() == "flat3d":
            return "flat2d_3d"
        return "flat2d_fisheye"

    def projection_mode(self, src_w: int, src_h: int) -> str:
        return self.projection_mode_static(src_w, src_h)

    def _blend_one_projected_subtitle(self, out_nv12, h: int, w: int, alpha_w: int, alpha_h: int, subtitle_overlay) -> None:
        cp = self._cp
        rgba, left, top = subtitle_overlay
        if rgba.size <= 0:
            return
        if self._g_overlay is None or self._g_overlay.shape != rgba.shape:
            self._g_overlay = cp.empty(rgba.shape, dtype=cp.uint8)
        self._g_overlay.set(rgba)
        overlay_h, overlay_w = int(rgba.shape[0]), int(rgba.shape[1])
        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        self._blend_projected_overlay_kernel(
            grid,
            block,
            (
                out_nv12,
                self._g_fisheye_alpha,
                np.int32(w),
                np.int32(h),
                np.int32(alpha_w),
                np.int32(alpha_h),
                np.float32(self.radius_scale),
                self._g_overlay,
                np.int32(overlay_w),
                np.int32(overlay_h),
                np.int32(left),
                np.int32(top),
            ),
        )

    def _blend_projected_subtitles(self, out_nv12, h: int, w: int, alpha_w: int, alpha_h: int, subtitle_overlay) -> None:
        if subtitle_overlay is None:
            return
        if isinstance(subtitle_overlay, list):
            overlays = subtitle_overlay
        else:
            overlays = [subtitle_overlay]
        for overlay in overlays:
            self._blend_one_projected_subtitle(out_nv12, h, w, alpha_w, alpha_h, overlay)

    def pack_uploaded(self, alpha, h: int, w: int, subtitle_overlay=None, out_h: int | None = None, out_w: int | None = None):
        cp = self._cp
        if hasattr(alpha, "data") and hasattr(alpha.data, "ptr"):
            alpha_dev = alpha.astype(cp.float32, copy=False)
        else:
            if self._g_alpha is None or self._g_alpha.shape != alpha.shape or self._g_alpha.dtype != cp.float32:
                self._g_alpha = cp.empty(alpha.shape, dtype=cp.float32)
            if alpha.dtype != np.float32:
                alpha = alpha.astype(np.float32, copy=False)
            if not alpha.flags["C_CONTIGUOUS"]:
                alpha = np.ascontiguousarray(alpha)
            self._g_alpha.set(alpha)
            alpha_dev = self._g_alpha

        src_h = int(h)
        src_w = int(w)
        if out_w is None or out_h is None:
            out_w, out_h = self.output_size(src_w, src_h)
        out_w, out_h = int(out_w), int(out_h)
        alpha_w = max(4, int(round(out_w * self.scale)) & ~3)
        alpha_h = max(2, int(round(out_h * self.scale)) & ~1)
        if alpha_w > out_w or alpha_h > out_h:
            raise RuntimeError(f"alpha pack does not fit: frame={out_w}x{out_h} alpha={alpha_w}x{alpha_h}")

        out_nv12 = self.matter._ensure_dev_nv12_out(out_h, out_w)
        if self._g_fisheye_alpha is None or self._g_fisheye_alpha.shape != (out_h, out_w):
            self._g_fisheye_alpha = cp.empty((out_h, out_w), dtype=getattr(cp, "uint8", np.uint8))

        ah, aw = alpha_dev.shape[:2]
        block = (16, 16, 1)
        grid = ((out_w + block[0] - 1) // block[0], (out_h + block[1] - 1) // block[1], 1)
        if hasattr(self, "_project_flat2d_kernel") and hasattr(self, "_project_flat3d_kernel"):
            projection_mode = self.projection_mode(src_w, src_h)
        else:
            log.warning("alpha packer flat projection kernels unavailable; falling back to fisheye projection")
            projection_mode = "fisheye"
        flat2d_disparity_px = alpha_2d_disparity_px(out_w)
        if hasattr(self.matter, "light_match_kernel_args"):
            light_coeffs, light_gamma_lut, light_identity = self.matter.light_match_kernel_args()
        else:
            log.warning("Matter.light_match_kernel_args unavailable; alpha packer disables light matching")
            light_coeffs, light_gamma_lut, light_identity = None, None, np.int32(1)
        if projection_mode == "flat2d_fisheye":
            self._project_flat2d_kernel(
                grid,
                block,
                (
                    self.matter._g_frame,
                    alpha_dev,
                    np.int32(src_w),
                    np.int32(src_h),
                    np.int32(out_w),
                    np.int32(out_h),
                    np.int32(aw),
                    np.int32(ah),
                    np.float32(self.radius_scale),
                    np.float32(np.deg2rad(config.ALPHA_2D_FOV)),
                    np.float32(flat2d_disparity_px),
                    np.float32(self.alpha_cutoff),
                    np.int32(1 if self.alpha_hard_edge else 0),
                    np.float32(self.alpha_contrast),
                    out_nv12,
                    self._g_fisheye_alpha,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
        elif projection_mode == "flat2d_3d":
            self._project_flat3d_kernel(
                grid,
                block,
                (
                    self.matter._g_frame,
                    alpha_dev,
                    np.int32(src_w),
                    np.int32(src_h),
                    np.int32(out_w),
                    np.int32(out_h),
                    np.int32(aw),
                    np.int32(ah),
                    np.float32(config.ALPHA_2D_FLAT3D_SAFE_W),
                    np.float32(config.ALPHA_2D_FLAT3D_SAFE_H),
                    np.float32(flat2d_disparity_px),
                    np.float32(self.alpha_cutoff),
                    np.int32(1 if self.alpha_hard_edge else 0),
                    np.float32(self.alpha_contrast),
                    out_nv12,
                    self._g_fisheye_alpha,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
        else:
            self._project_kernel(
                grid,
                block,
                (
                    self.matter._g_frame,
                    alpha_dev,
                    np.int32(out_w),
                    np.int32(out_h),
                    np.int32(aw),
                    np.int32(ah),
                    np.float32(self.radius_scale),
                    np.float32(self.alpha_cutoff),
                    np.int32(1 if self.alpha_hard_edge else 0),
                    np.float32(self.alpha_contrast),
                    out_nv12,
                    self._g_fisheye_alpha,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
        self._blend_projected_subtitles(out_nv12, out_h, out_w, alpha_w, alpha_h, subtitle_overlay)
        self._overlay_kernel(
            grid,
            block,
            (
                self._g_fisheye_alpha,
                np.int32(out_w),
                np.int32(out_h),
                np.int32(alpha_w),
                np.int32(alpha_h),
                out_nv12,
            ),
        )
        return out_nv12

    def pack_gpu_nv12_frame(
        self,
        frame,
        before_pack=None,
        subtitle_overlay=None,
        out_h: int | None = None,
        out_w: int | None = None,
        use_config_scale: bool = True,
    ):
        src_h, src_w = int(frame.height), int(frame.width)
        if use_config_scale and hasattr(self.matter, "pynv_scaled_size"):
            w, h = self.matter.pynv_scaled_size(src_w, src_h)
        else:
            if use_config_scale:
                log.warning("Matter.pynv_scaled_size unavailable; alpha packer uses source size")
            w, h = src_w, src_h
        if out_w is not None and out_h is not None:
            pack_w, pack_h = int(out_w), int(out_h)
        else:
            pack_w, pack_h = self.output_size(w, h)
        self.matter.upload_nv12_planes_gpu_scaled(frame.y.as_cupy(), frame.uv.as_cupy(), src_h, src_w, h, w)
        alpha, timing, _ = self.matter._alpha_low_res_gpu(h, w, use_nv12=True)
        if before_pack is not None:
            result = before_pack(self.matter._g_frame)
            if result is not None:
                subtitle_overlay = result
        try:
            packed = self.pack_uploaded(alpha, h, w, subtitle_overlay=subtitle_overlay, out_h=pack_h, out_w=pack_w)
        except TypeError as exc:
            if "out_h" not in str(exc) and "out_w" not in str(exc):
                raise
            log.warning("alpha packer pack_uploaded does not accept out_h/out_w; falling back to implicit output size")
            packed = self.pack_uploaded(alpha, h, w, subtitle_overlay=subtitle_overlay)
        return packed, timing

    def pack_gpu_p016_frame(
        self,
        frame,
        shift_bits: int = 8,
        before_pack=None,
        subtitle_overlay=None,
        out_h: int | None = None,
        out_w: int | None = None,
        use_config_scale: bool = True,
    ):
        src_h, src_w = int(frame.height), int(frame.width)
        if use_config_scale and hasattr(self.matter, "pynv_scaled_size"):
            w, h = self.matter.pynv_scaled_size(src_w, src_h)
        else:
            if use_config_scale:
                log.warning("Matter.pynv_scaled_size unavailable; alpha packer uses source size")
            w, h = src_w, src_h
        if out_w is not None and out_h is not None:
            pack_w, pack_h = int(out_w), int(out_h)
        else:
            pack_w, pack_h = self.output_size(w, h)
        if hasattr(self.matter, "upload_p016_planes_as_nv12_gpu_scaled"):
            self.matter.upload_p016_planes_as_nv12_gpu_scaled(
                frame.y.as_cupy(),
                frame.uv.as_cupy(),
                src_h,
                src_w,
                h,
                w,
                shift_bits=shift_bits,
            )
        else:
            log.warning("Matter.upload_p016_planes_as_nv12_gpu_scaled unavailable; alpha packer uses unscaled P016 upload")
            self.matter.upload_p016_planes_as_nv12_gpu(
                frame.y.as_cupy(),
                frame.uv.as_cupy(),
                src_h,
                src_w,
                shift_bits=shift_bits,
            )
        alpha, timing, _ = self.matter._alpha_low_res_gpu(h, w, use_nv12=True)
        if before_pack is not None:
            result = before_pack(self.matter._g_frame)
            if result is not None:
                subtitle_overlay = result
        try:
            packed = self.pack_uploaded(alpha, h, w, subtitle_overlay=subtitle_overlay, out_h=pack_h, out_w=pack_w)
        except TypeError as exc:
            if "out_h" not in str(exc) and "out_w" not in str(exc):
                raise
            log.warning("alpha packer pack_uploaded does not accept out_h/out_w; falling back to implicit output size")
            packed = self.pack_uploaded(alpha, h, w, subtitle_overlay=subtitle_overlay)
        return packed, timing
