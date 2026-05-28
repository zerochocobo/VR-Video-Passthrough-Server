"""ONNX Runtime matting and green-screen compositing.

The Matter class loads the RVM ONNX model, prepares SBS-aware inputs, keeps
recurrent RVM state, and composites the foreground over a green background. It
supports both the legacy CPU/BGR path and the production GPU/NV12 path used by
PyNv HEVC passthrough.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from pipeline.light_match import LIGHT_MATCH_DEVICE_SRC, build_light_match_tables
from utils.runtime_settings import get_light_match
from config import (
    ALPHA_CONTRAST,
    ALPHA_CUTOFF,
    ALPHA_HARD_EDGE,
    ALPHA_MODE,
    ALPHA_STRIDE,
    CUDA_CUDNN_CONV_ALGO_SEARCH,
    CUDA_SHARED_STREAM,
    DEBUG_LOGS,
    FAST_UV_ALPHA,
    GREEN_BGR,
    MATTING_DEVICE,
    MATTING_INPUT_SIZE,
    MATTING_MODEL_KIND,
    MATTING_SBS_BATCH,
    MATTING_SQUARE,
    MATTING_SPLIT_SBS,
    MATTING_WARMUP_RUNS,
    MODEL_PATH,
    ONNX_TRT_CUDA_GRAPH_ENABLE,
    ONNX_TRT_DETAILED_BUILD_LOG,
    ONNX_TRT_DUMP_SUBGRAPHS,
    ONNX_TRT_ENGINE_CACHE_ENABLE,
    ONNX_TRT_ENGINE_CACHE_PATH,
    ONNX_TRT_FP16_ENABLE,
    ONNX_PROVIDERS,
    PASSTHROUGH_RVM_BYPASS_ALPHA,
    PASSTHROUGH_PYNV_SYNC_PROBE,
    PASSTHROUGH_NV12_RING_SLOTS,
    DECODE_MAX_SIDE,
    RVM_DOWNSAMPLE_RATIO,
    RVM_ALPHA_SMOOTH,
    RVM_ALPHA_SMOOTH_WEIGHT,
    RVM_IOBINDING,
    RVM_SCENE_COOLDOWN,
    RVM_SCENE_REF_EMA,
    RVM_SCENE_RESET,
    RVM_SCENE_THRESHOLD,
    SPLIT_NV12_COMPOSITE,
    TRT_RVM_IOBINDING,
    WARMUP_RAMPUP_DIAG_FRAMES,
)
from utils.logger import get

log = get("matting")

if not (ONNX_TRT_DUMP_SUBGRAPHS or ONNX_TRT_DETAILED_BUILD_LOG):
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass


_cp = None
_CUDA_STREAM = None
_composite_kernel = None
_composite_nv12_upsample_kernel = None
_composite_nv12_to_nv12_kernel = None
_composite_nv12_y_kernel = None
_composite_nv12_uv_kernel = None
_preprocess_nv12_kernel = None
_preprocess_kernel_fp16 = None
_preprocess_nv12_kernel_fp16 = None
_preprocess_nv12_roi_kernel = None
_preprocess_nv12_roi_kernel_fp16 = None
_alpha_roi_unwarp_kernel = None
_GPU_OK = False


def _should_enable_rvm_iobinding(active_providers: list[str]) -> bool:
    if not RVM_IOBINDING:
        return False
    if "TensorrtExecutionProvider" in active_providers:
        return TRT_RVM_IOBINDING
    return "CUDAExecutionProvider" in active_providers


class _SceneCutDetector:
    """HSV-Bhattacharyya scene cut detector with reference EMA and cooldown."""

    def __init__(
        self,
        threshold: float = 0.4,
        cooldown_frames: int = 24,
        ref_ema_alpha: float = 0.95,
        downsample_height: int = 540,
    ) -> None:
        self.threshold = float(threshold)
        self.cooldown = max(0, int(cooldown_frames))
        self.ref_ema_alpha = float(ref_ema_alpha)
        self.downsample_height = max(2, int(downsample_height))
        self._ref_hist: np.ndarray | None = None
        self._cooldown_left = 0

    def step(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        if h > self.downsample_height:
            new_w = max(2, int(w * self.downsample_height / h))
            frame_bgr = cv2.resize(frame_bgr, (new_w, self.downsample_height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

        if self._ref_hist is None:
            self._ref_hist = hist
            return False

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._update_ref(hist)
            return False

        dist = float(cv2.compareHist(self._ref_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        if dist > self.threshold:
            self._ref_hist = hist
            self._cooldown_left = self.cooldown
            return True
        self._update_ref(hist)
        return False

    def _update_ref(self, hist: np.ndarray) -> None:
        assert self._ref_hist is not None
        self._ref_hist = self._ref_hist * self.ref_ema_alpha + hist * (1.0 - self.ref_ema_alpha)

    def reset(self) -> None:
        self._ref_hist = None
        self._cooldown_left = 0


class _AlphaSmoother:
    def __init__(self, alpha_weight: float = 0.6) -> None:
        self.alpha_weight = float(alpha_weight)
        self._prev = None

    def step(self, alpha_batch):
        if self._prev is None or self._prev.shape != alpha_batch.shape:
            self._prev = alpha_batch.copy()
            return alpha_batch
        out = self._prev * self.alpha_weight + alpha_batch * (1.0 - self.alpha_weight)
        self._prev = out
        return out

    def reset(self) -> None:
        self._prev = None


_LIGHT_MATCH_DEVICE_SRC = LIGHT_MATCH_DEVICE_SRC

if MATTING_DEVICE in ("auto", "gpu"):
    try:
        import cupy as _cp_mod  # type: ignore

        _cp = _cp_mod
        if CUDA_SHARED_STREAM:
            _CUDA_STREAM = _cp.cuda.Stream(non_blocking=True)
        _COMPOSITE_KERNEL_SRC = r"""
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

        extern "C" __global__
        void composite_green(
            const unsigned char* __restrict__ frame,
            const float* __restrict__ alpha,
            unsigned char gB, unsigned char gG, unsigned char gR,
            float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
            int total_pixels,
            unsigned char* __restrict__ out
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_pixels) return;
            float a = adjust_alpha(alpha[idx], alpha_cutoff, alpha_hard_edge, alpha_contrast);
            float inv = 1.f - a;
            int p = idx * 3;
            float b = (float)frame[p]   * a + (float)gB * inv;
            float g = (float)frame[p+1] * a + (float)gG * inv;
            float r = (float)frame[p+2] * a + (float)gR * inv;
            b = b < 0.f ? 0.f : (b > 255.f ? 255.f : b);
            g = g < 0.f ? 0.f : (g > 255.f ? 255.f : g);
            r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
            out[p]   = (unsigned char)(b + 0.5f);
            out[p+1] = (unsigned char)(g + 0.5f);
            out[p+2] = (unsigned char)(r + 0.5f);
        }
        """
        _composite_kernel = _cp.RawKernel(_COMPOSITE_KERNEL_SRC, "composite_green")

        # Fused alpha upsample + BGR composite, replacing CPU cv2.resize on the
        # hot path when the source frame is already on the GPU.
        _COMPOSITE_UPSAMPLE_KERNEL_SRC = r"""
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

        extern "C" __global__
        void composite_green_upsample(
            const unsigned char* __restrict__ frame,   // out_h x out_w x 3
            const float* __restrict__ alpha_lr,        // ah x aw
            int out_w, int out_h,
            int aw, int ah,
            float scale_x, float scale_y,              // (aw / out_w), (ah / out_h)
            unsigned char gB, unsigned char gG, unsigned char gR,
            float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
            unsigned char* __restrict__ out
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            float fx = (x + 0.5f) * scale_x - 0.5f;
            float fy = (y + 0.5f) * scale_y - 0.5f;
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
            a = adjust_alpha(a, alpha_cutoff, alpha_hard_edge, alpha_contrast);
            float inv = 1.f - a;

            int p = (y * out_w + x) * 3;
            float b = (float)frame[p]   * a + (float)gB * inv;
            float g = (float)frame[p+1] * a + (float)gG * inv;
            float r = (float)frame[p+2] * a + (float)gR * inv;
            b = b < 0.f ? 0.f : (b > 255.f ? 255.f : b);
            g = g < 0.f ? 0.f : (g > 255.f ? 255.f : g);
            r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
            out[p]   = (unsigned char)(b + 0.5f);
            out[p+1] = (unsigned char)(g + 0.5f);
            out[p+2] = (unsigned char)(r + 0.5f);
        }
        """
        _composite_upsample_kernel = _cp.RawKernel(
            _COMPOSITE_UPSAMPLE_KERNEL_SRC, "composite_green_upsample"
        )

        # GPU BGR preprocess: crop, box-resample, convert BGR->RGB, normalize
        # to [-1, 1], and write CHW float32 for ORT input binding.
        _PREPROCESS_KERNEL_SRC = r"""
 extern "C" __global__
 void preprocess_bgr_chw_norm(
 const unsigned char* __restrict__ frame, // in_h x in_w x 3
 int in_w, int in_h,
 int src_x0, int src_w, // crop [src_x0, src_x0+src_w)
 int out_w, int out_h,
 float norm_scale, float norm_bias,
 float* __restrict__ out // 3 x out_h x out_w (RGB normalized)
 ) {
 int x = blockIdx.x * blockDim.x + threadIdx.x;
 int y = blockIdx.y * blockDim.y + threadIdx.y;
 if (x >= out_w || y >= out_h) return;

 float fsx0 = (float)x * (float)src_w / (float)out_w;
 float fsx1 = (float)(x + 1) * (float)src_w / (float)out_w;
 float fsy0 = (float)y * (float)in_h / (float)out_h;
 float fsy1 = (float)(y + 1) * (float)in_h / (float)out_h;

 int sx0 = (int)floorf(fsx0);
 int sx1 = (int)ceilf(fsx1);
 int sy0 = (int)floorf(fsy0);
 int sy1 = (int)ceilf(fsy1);
 if (sx1 <= sx0) sx1 = sx0 + 1;
 if (sy1 <= sy0) sy1 = sy0 + 1;
 if (sx1 > src_w) sx1 = src_w;
 if (sy1 > in_h) sy1 = in_h;

 float sumB = 0.f, sumG = 0.f, sumR = 0.f;
 int count = 0;
 for (int yy = sy0; yy < sy1; ++yy) {
 int row = yy * in_w;
 for (int xx = sx0; xx < sx1; ++xx) {
 int p = (row + (xx + src_x0)) * 3;
 sumB += (float)frame[p];
 sumG += (float)frame[p + 1];
 sumR += (float)frame[p + 2];
 ++count;
 }
 }
 float invc = 1.f / (float)count;
 float B = sumB * invc;
 float G = sumG * invc;
 float R = sumR * invc;
 float nB = B * norm_scale + norm_bias;
 float nG = G * norm_scale + norm_bias;
 float nR = R * norm_scale + norm_bias;

 int plane = out_w * out_h;
 int idx = y * out_w + x;
 out[0 * plane + idx] = nR;
 out[1 * plane + idx] = nG;
 out[2 * plane + idx] = nB;
 }
 """
        _preprocess_kernel = _cp.RawKernel(
            _PREPROCESS_KERNEL_SRC, "preprocess_bgr_chw_norm"
        )

        # fp16 __half ORT fp32 fp16 cast
        _PREPROCESS_KERNEL_FP16_SRC = r"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void preprocess_bgr_chw_norm_fp16(
            const unsigned char* __restrict__ frame,
            int in_w, int in_h,
            int src_x0, int src_w,
            int out_w, int out_h,
            float norm_scale, float norm_bias,
            __half* __restrict__ out
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            float fsx0 = (float)x       * (float)src_w / (float)out_w;
            float fsx1 = (float)(x + 1) * (float)src_w / (float)out_w;
            float fsy0 = (float)y       * (float)in_h  / (float)out_h;
            float fsy1 = (float)(y + 1) * (float)in_h  / (float)out_h;

            int sx0 = (int)floorf(fsx0);
            int sx1 = (int)ceilf(fsx1);
            int sy0 = (int)floorf(fsy0);
            int sy1 = (int)ceilf(fsy1);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sy1 <= sy0) sy1 = sy0 + 1;
            if (sx1 > src_w) sx1 = src_w;
            if (sy1 > in_h)  sy1 = in_h;

            float sumB = 0.f, sumG = 0.f, sumR = 0.f;
            int count = 0;
            for (int yy = sy0; yy < sy1; ++yy) {
                int row = yy * in_w;
                for (int xx = sx0; xx < sx1; ++xx) {
                    int p = (row + (xx + src_x0)) * 3;
                    sumB += (float)frame[p];
                    sumG += (float)frame[p + 1];
                    sumR += (float)frame[p + 2];
                    ++count;
                }
            }
            float invc = 1.f / (float)count;
            float B = sumB * invc;
            float G = sumG * invc;
            float R = sumR * invc;
            float nB = B * norm_scale + norm_bias;
            float nG = G * norm_scale + norm_bias;
            float nR = R * norm_scale + norm_bias;

            int plane = out_w * out_h;
            int idx = y * out_w + x;
            out[0 * plane + idx] = __float2half_rn(nR);
            out[1 * plane + idx] = __float2half_rn(nG);
            out[2 * plane + idx] = __float2half_rn(nB);
        }
        """
        _preprocess_kernel_fp16 = _cp.RawKernel(
            _PREPROCESS_KERNEL_FP16_SRC, "preprocess_bgr_chw_norm_fp16"
        )

        _PREPROCESS_NV12_KERNEL_SRC = r"""
        extern "C" __global__
        void preprocess_nv12_chw_norm(
            const unsigned char* __restrict__ nv12,
            int in_w, int in_h,
            int src_x0, int src_w,
            int out_w, int out_h,
            float norm_scale, float norm_bias,
            float* __restrict__ out
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            float fsx0 = (float)x       * (float)src_w / (float)out_w;
            float fsx1 = (float)(x + 1) * (float)src_w / (float)out_w;
            float fsy0 = (float)y       * (float)in_h  / (float)out_h;
            float fsy1 = (float)(y + 1) * (float)in_h  / (float)out_h;
            int sx0 = (int)floorf(fsx0);
            int sx1 = (int)ceilf(fsx1);
            int sy0 = (int)floorf(fsy0);
            int sy1 = (int)ceilf(fsy1);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sy1 <= sy0) sy1 = sy0 + 1;
            if (sx1 > src_w) sx1 = src_w;
            if (sy1 > in_h)  sy1 = in_h;

            const unsigned char* y_plane = nv12;
            const unsigned char* uv_plane = nv12 + in_w * in_h;
            float sumB = 0.f, sumG = 0.f, sumR = 0.f;
            int count = 0;
            for (int yy = sy0; yy < sy1; ++yy) {
                int row = yy * in_w;
                int uv_row = (yy >> 1) * in_w;
                for (int xx = sx0; xx < sx1; ++xx) {
                    int px = xx + src_x0;
                    int uvx = px & ~1;
                    float Y = (float)y_plane[row + px];
                    float U = (float)uv_plane[uv_row + uvx];
                    float V = (float)uv_plane[uv_row + uvx + 1];
                    float C = Y - 16.f; if (C < 0.f) C = 0.f;
                    float D = U - 128.f;
                    float E = V - 128.f;
                    float R = 1.16438356f * C + 1.59602678f * E;
                    float G = 1.16438356f * C - 0.39176229f * D - 0.81296765f * E;
                    float B = 1.16438356f * C + 2.01723214f * D;
                    sumB += B < 0.f ? 0.f : (B > 255.f ? 255.f : B);
                    sumG += G < 0.f ? 0.f : (G > 255.f ? 255.f : G);
                    sumR += R < 0.f ? 0.f : (R > 255.f ? 255.f : R);
                    ++count;
                }
            }
            float invc = 1.f / (float)count;
            float B = sumB * invc;
            float G = sumG * invc;
            float R = sumR * invc;
            int plane = out_w * out_h;
            int idx = y * out_w + x;
            out[0 * plane + idx] = R * norm_scale + norm_bias;
            out[1 * plane + idx] = G * norm_scale + norm_bias;
            out[2 * plane + idx] = B * norm_scale + norm_bias;
        }
        """
        _preprocess_nv12_kernel = _cp.RawKernel(
            _PREPROCESS_NV12_KERNEL_SRC, "preprocess_nv12_chw_norm"
        )

        _PREPROCESS_NV12_KERNEL_FP16_SRC = r"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void preprocess_nv12_chw_norm_fp16(
            const unsigned char* __restrict__ nv12,
            int in_w, int in_h,
            int src_x0, int src_w,
            int out_w, int out_h,
            float norm_scale, float norm_bias,
            __half* __restrict__ out
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            float fsx0 = (float)x       * (float)src_w / (float)out_w;
            float fsx1 = (float)(x + 1) * (float)src_w / (float)out_w;
            float fsy0 = (float)y       * (float)in_h  / (float)out_h;
            float fsy1 = (float)(y + 1) * (float)in_h  / (float)out_h;
            int sx0 = (int)floorf(fsx0);
            int sx1 = (int)ceilf(fsx1);
            int sy0 = (int)floorf(fsy0);
            int sy1 = (int)ceilf(fsy1);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sy1 <= sy0) sy1 = sy0 + 1;
            if (sx1 > src_w) sx1 = src_w;
            if (sy1 > in_h)  sy1 = in_h;

            const unsigned char* y_plane = nv12;
            const unsigned char* uv_plane = nv12 + in_w * in_h;
            float sumB = 0.f, sumG = 0.f, sumR = 0.f;
            int count = 0;
            for (int yy = sy0; yy < sy1; ++yy) {
                int row = yy * in_w;
                int uv_row = (yy >> 1) * in_w;
                for (int xx = sx0; xx < sx1; ++xx) {
                    int px = xx + src_x0;
                    int uvx = px & ~1;
                    float Y = (float)y_plane[row + px];
                    float U = (float)uv_plane[uv_row + uvx];
                    float V = (float)uv_plane[uv_row + uvx + 1];
                    float C = Y - 16.f; if (C < 0.f) C = 0.f;
                    float D = U - 128.f;
                    float E = V - 128.f;
                    float R = 1.16438356f * C + 1.59602678f * E;
                    float G = 1.16438356f * C - 0.39176229f * D - 0.81296765f * E;
                    float B = 1.16438356f * C + 2.01723214f * D;
                    sumB += B < 0.f ? 0.f : (B > 255.f ? 255.f : B);
                    sumG += G < 0.f ? 0.f : (G > 255.f ? 255.f : G);
                    sumR += R < 0.f ? 0.f : (R > 255.f ? 255.f : R);
                    ++count;
                }
            }
            float invc = 1.f / (float)count;
            float B = sumB * invc;
            float G = sumG * invc;
            float R = sumR * invc;
            int plane = out_w * out_h;
            int idx = y * out_w + x;
            out[0 * plane + idx] = __float2half_rn(R * norm_scale + norm_bias);
            out[1 * plane + idx] = __float2half_rn(G * norm_scale + norm_bias);
            out[2 * plane + idx] = __float2half_rn(B * norm_scale + norm_bias);
        }
        """
        _preprocess_nv12_kernel_fp16 = _cp.RawKernel(
            _PREPROCESS_NV12_KERNEL_FP16_SRC, "preprocess_nv12_chw_norm_fp16"
        )

        _PREPROCESS_NV12_ROI_KERNEL_SRC = r"""
        extern "C" __global__
        void preprocess_nv12_roi_chw_norm(
            const unsigned char* __restrict__ nv12,
            int in_w, int in_h,
            int roi_x0, int roi_y0, int roi_w, int roi_h,
            int model_x0, int model_y0, int model_w, int model_h,
            int out_w, int out_h,
            float norm_scale, float norm_bias,
            float* __restrict__ out
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            int plane = out_w * out_h;
            int idx = y * out_w + x;
            float nR = norm_bias;
            float nG = norm_bias;
            float nB = norm_bias;
            if (x >= model_x0 && x < model_x0 + model_w && y >= model_y0 && y < model_y0 + model_h) {
                float sx = (float)(x - model_x0) * (float)roi_w / (float)model_w;
                float sy = (float)(y - model_y0) * (float)roi_h / (float)model_h;
                int px = roi_x0 + (int)floorf(sx);
                int py = roi_y0 + (int)floorf(sy);
                if (px < 0) px = 0; if (px > in_w - 1) px = in_w - 1;
                if (py < 0) py = 0; if (py > in_h - 1) py = in_h - 1;
                int uvx = px & ~1;
                int uv_row = (py >> 1) * in_w;
                const unsigned char* y_plane = nv12;
                const unsigned char* uv_plane = nv12 + in_w * in_h;
                float Y = (float)y_plane[py * in_w + px];
                float U = (float)uv_plane[uv_row + uvx];
                float V = (float)uv_plane[uv_row + uvx + 1];
                float C = Y - 16.f; if (C < 0.f) C = 0.f;
                float D = U - 128.f;
                float E = V - 128.f;
                float R = 1.16438356f * C + 1.59602678f * E;
                float G = 1.16438356f * C - 0.39176229f * D - 0.81296765f * E;
                float B = 1.16438356f * C + 2.01723214f * D;
                if (R < 0.f) R = 0.f; if (R > 255.f) R = 255.f;
                if (G < 0.f) G = 0.f; if (G > 255.f) G = 255.f;
                if (B < 0.f) B = 0.f; if (B > 255.f) B = 255.f;
                nR = R * norm_scale + norm_bias;
                nG = G * norm_scale + norm_bias;
                nB = B * norm_scale + norm_bias;
            }
            out[0 * plane + idx] = nR;
            out[1 * plane + idx] = nG;
            out[2 * plane + idx] = nB;
        }
        """
        _preprocess_nv12_roi_kernel = _cp.RawKernel(
            _PREPROCESS_NV12_ROI_KERNEL_SRC, "preprocess_nv12_roi_chw_norm"
        )

        _PREPROCESS_NV12_ROI_KERNEL_FP16_SRC = r"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void preprocess_nv12_roi_chw_norm_fp16(
            const unsigned char* __restrict__ nv12,
            int in_w, int in_h,
            int roi_x0, int roi_y0, int roi_w, int roi_h,
            int model_x0, int model_y0, int model_w, int model_h,
            int out_w, int out_h,
            float norm_scale, float norm_bias,
            __half* __restrict__ out
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            int plane = out_w * out_h;
            int idx = y * out_w + x;
            float nR = norm_bias;
            float nG = norm_bias;
            float nB = norm_bias;
            if (x >= model_x0 && x < model_x0 + model_w && y >= model_y0 && y < model_y0 + model_h) {
                float sx = (float)(x - model_x0) * (float)roi_w / (float)model_w;
                float sy = (float)(y - model_y0) * (float)roi_h / (float)model_h;
                int px = roi_x0 + (int)floorf(sx);
                int py = roi_y0 + (int)floorf(sy);
                if (px < 0) px = 0; if (px > in_w - 1) px = in_w - 1;
                if (py < 0) py = 0; if (py > in_h - 1) py = in_h - 1;
                int uvx = px & ~1;
                int uv_row = (py >> 1) * in_w;
                const unsigned char* y_plane = nv12;
                const unsigned char* uv_plane = nv12 + in_w * in_h;
                float Y = (float)y_plane[py * in_w + px];
                float U = (float)uv_plane[uv_row + uvx];
                float V = (float)uv_plane[uv_row + uvx + 1];
                float C = Y - 16.f; if (C < 0.f) C = 0.f;
                float D = U - 128.f;
                float E = V - 128.f;
                float R = 1.16438356f * C + 1.59602678f * E;
                float G = 1.16438356f * C - 0.39176229f * D - 0.81296765f * E;
                float B = 1.16438356f * C + 2.01723214f * D;
                if (R < 0.f) R = 0.f; if (R > 255.f) R = 255.f;
                if (G < 0.f) G = 0.f; if (G > 255.f) G = 255.f;
                if (B < 0.f) B = 0.f; if (B > 255.f) B = 255.f;
                nR = R * norm_scale + norm_bias;
                nG = G * norm_scale + norm_bias;
                nB = B * norm_scale + norm_bias;
            }
            out[0 * plane + idx] = __float2half_rn(nR);
            out[1 * plane + idx] = __float2half_rn(nG);
            out[2 * plane + idx] = __float2half_rn(nB);
        }
        """
        _preprocess_nv12_roi_kernel_fp16 = _cp.RawKernel(
            _PREPROCESS_NV12_ROI_KERNEL_FP16_SRC, "preprocess_nv12_roi_chw_norm_fp16"
        )

        _ALPHA_ROI_UNWARP_KERNEL_SRC = r"""
        extern "C" __global__
        void alpha_roi_unwarp(
            const float* __restrict__ alpha_roi,
            float* __restrict__ alpha_eye,
            int eye_w, int eye_h,
            int roi_x0, int roi_y0, int roi_w, int roi_h,
            int model_x0, int model_y0, int model_w, int model_h,
            int src_w, int src_h,
            int feather
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= eye_w || y >= eye_h) return;
            float out = 0.f;
            if (x >= roi_x0 && x < roi_x0 + roi_w && y >= roi_y0 && y < roi_y0 + roi_h) {
                float mx = ((float)(x - roi_x0) + 0.5f) * (float)model_w / (float)roi_w + (float)model_x0 - 0.5f;
                float my = ((float)(y - roi_y0) + 0.5f) * (float)model_h / (float)roi_h + (float)model_y0 - 0.5f;
                int x0 = (int)floorf(mx); if (x0 < 0) x0 = 0; if (x0 > src_w - 1) x0 = src_w - 1;
                int y0 = (int)floorf(my); if (y0 < 0) y0 = 0; if (y0 > src_h - 1) y0 = src_h - 1;
                int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
                int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
                float dx = mx - floorf(mx); if (dx < 0.f) dx = 0.f; if (dx > 1.f) dx = 1.f;
                float dy = my - floorf(my); if (dy < 0.f) dy = 0.f; if (dy > 1.f) dy = 1.f;
                float a00 = alpha_roi[y0 * src_w + x0];
                float a01 = alpha_roi[y0 * src_w + x1];
                float a10 = alpha_roi[y1 * src_w + x0];
                float a11 = alpha_roi[y1 * src_w + x1];
                out = (1.f - dy) * ((1.f - dx) * a00 + dx * a01)
                    +        dy  * ((1.f - dx) * a10 + dx * a11);
                if (feather > 0) {
                    int dl = x - roi_x0;
                    int dr = roi_x0 + roi_w - 1 - x;
                    int dt = y - roi_y0;
                    int db = roi_y0 + roi_h - 1 - y;
                    int d = dl;
                    if (dr < d) d = dr;
                    if (dt < d) d = dt;
                    if (db < d) d = db;
                    float edge = ((float)d + 1.f) / (float)feather;
                    if (edge < 0.f) edge = 0.f;
                    if (edge > 1.f) edge = 1.f;
                    out *= edge;
                }
            }
            if (out < 0.f) out = 0.f;
            if (out > 1.f) out = 1.f;
            alpha_eye[y * eye_w + x] = out;
        }
        """
        _alpha_roi_unwarp_kernel = _cp.RawKernel(
            _ALPHA_ROI_UNWARP_KERNEL_SRC, "alpha_roi_unwarp"
        )

        _COMPOSITE_NV12_UPSAMPLE_KERNEL_SRC = _LIGHT_MATCH_DEVICE_SRC + r"""
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

        extern "C" __global__
        void composite_green_nv12_upsample(
            const unsigned char* __restrict__ nv12,
            const float* __restrict__ alpha_lr,
            int out_w, int out_h,
            int aw, int ah,
            float scale_x, float scale_y,
            unsigned char gB, unsigned char gG, unsigned char gR,
            float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
            unsigned char* __restrict__ out,
            const float* __restrict__ light_coeffs,
            const unsigned char* __restrict__ light_gamma_lut,
            int light_identity
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            float fx = (x + 0.5f) * scale_x - 0.5f;
            float fy = (y + 0.5f) * scale_y - 0.5f;
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
            a = adjust_alpha(a, alpha_cutoff, alpha_hard_edge, alpha_contrast);
            float inv = 1.f - a;

            const unsigned char* y_plane = nv12;
            const unsigned char* uv_plane = nv12 + out_w * out_h;
            int uvx = x & ~1;
            float Y = (float)y_plane[y * out_w + x];
            float U = (float)uv_plane[(y >> 1) * out_w + uvx];
            float V = (float)uv_plane[(y >> 1) * out_w + uvx + 1];
            float C = Y - 16.f; if (C < 0.f) C = 0.f;
            float D = U - 128.f;
            float E = V - 128.f;
            float r0 = 1.16438356f * C + 1.79274107f * E;
            float g0 = 1.16438356f * C - 0.21324861f * D - 0.53290933f * E;
            float b0 = 1.16438356f * C + 2.11240179f * D;
            b0 = b0 < 0.f ? 0.f : (b0 > 255.f ? 255.f : b0);
            g0 = g0 < 0.f ? 0.f : (g0 > 255.f ? 255.f : g0);
            r0 = r0 < 0.f ? 0.f : (r0 > 255.f ? 255.f : r0);
            if (!light_identity) {
                float yy, uu, vv;
                yy = 16.f + 0.182586f * r0 + 0.614231f * g0 + 0.062007f * b0;
                uu = 128.f - 0.100644f * r0 - 0.338572f * g0 + 0.439216f * b0;
                vv = 128.f + 0.439216f * r0 - 0.398942f * g0 - 0.040274f * b0;
                apply_light_match(&yy, &uu, &vv, light_coeffs, light_gamma_lut, light_identity);
                float C2 = yy - 16.f; if (C2 < 0.f) C2 = 0.f;
                float D2 = uu - 128.f;
                float E2 = vv - 128.f;
                r0 = 1.16438356f * C2 + 1.79274107f * E2;
                g0 = 1.16438356f * C2 - 0.21324861f * D2 - 0.53290933f * E2;
                b0 = 1.16438356f * C2 + 2.11240179f * D2;
                b0 = b0 < 0.f ? 0.f : (b0 > 255.f ? 255.f : b0);
                g0 = g0 < 0.f ? 0.f : (g0 > 255.f ? 255.f : g0);
                r0 = r0 < 0.f ? 0.f : (r0 > 255.f ? 255.f : r0);
            }
            int p = (y * out_w + x) * 3;
            out[p]     = (unsigned char)(b0 * a + (float)gB * inv + 0.5f);
            out[p + 1] = (unsigned char)(g0 * a + (float)gG * inv + 0.5f);
            out[p + 2] = (unsigned char)(r0 * a + (float)gR * inv + 0.5f);
        }
        """
        _composite_nv12_upsample_kernel = _cp.RawKernel(
            _COMPOSITE_NV12_UPSAMPLE_KERNEL_SRC, "composite_green_nv12_upsample"
        )

        _COMPOSITE_NV12_TO_NV12_KERNEL_SRC = _LIGHT_MATCH_DEVICE_SRC + r"""
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

        extern "C" __global__
        void composite_green_nv12_to_nv12(
            const unsigned char* __restrict__ src_nv12,
            const float* __restrict__ alpha_lr,
            int out_w, int out_h,
            int aw, int ah,
            unsigned char gY, unsigned char gU, unsigned char gV,
            int fast_uv_alpha,
            float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
            unsigned char* __restrict__ out_nv12,
            const float* __restrict__ light_coeffs,
            const unsigned char* __restrict__ light_gamma_lut,
            int light_identity
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;

            int y_idx = y * out_w + x;
            float a = adjust_alpha(
                sample_alpha_lr(alpha_lr, aw, ah, out_w, out_h, x, y),
                alpha_cutoff, alpha_hard_edge, alpha_contrast
            );
            float src_y = (float)src_nv12[y_idx];
            apply_light_match_y_only(&src_y, light_coeffs, light_gamma_lut, light_identity);
            float yv = src_y * a + (float)gY * (1.f - a);
            yv = yv < 0.f ? 0.f : (yv > 255.f ? 255.f : yv);
            out_nv12[y_idx] = (unsigned char)(yv + 0.5f);

            if (((x | y) & 1) == 0) {
                int uv_idx = out_w * out_h + (y >> 1) * out_w + x;
                float auv = a;
                if (!fast_uv_alpha) {
                    float a01 = adjust_alpha(sample_alpha_lr(alpha_lr, aw, ah, out_w, out_h, x + 1, y), alpha_cutoff, alpha_hard_edge, alpha_contrast);
                    float a10 = adjust_alpha(sample_alpha_lr(alpha_lr, aw, ah, out_w, out_h, x, y + 1), alpha_cutoff, alpha_hard_edge, alpha_contrast);
                    float a11 = adjust_alpha(sample_alpha_lr(alpha_lr, aw, ah, out_w, out_h, x + 1, y + 1), alpha_cutoff, alpha_hard_edge, alpha_contrast);
                    auv = 0.25f * (a + a01 + a10 + a11);
                }
                float src_u = (float)src_nv12[uv_idx];
                float src_v = (float)src_nv12[uv_idx + 1];
                apply_light_match_uv_only(&src_u, &src_v, light_coeffs, light_identity);
                float u = src_u * auv + (float)gU * (1.f - auv);
                float v = src_v * auv + (float)gV * (1.f - auv);
                u = u < 0.f ? 0.f : (u > 255.f ? 255.f : u);
                v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
                out_nv12[uv_idx] = (unsigned char)(u + 0.5f);
                out_nv12[uv_idx + 1] = (unsigned char)(v + 0.5f);
            }
        }
        """
        _composite_nv12_to_nv12_kernel = _cp.RawKernel(
            _COMPOSITE_NV12_TO_NV12_KERNEL_SRC, "composite_green_nv12_to_nv12"
        )

        _COMPOSITE_NV12_Y_KERNEL_SRC = _LIGHT_MATCH_DEVICE_SRC + r"""
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

        __device__ float sample_alpha_lr_y(
            const float* __restrict__ alpha_lr,
            int aw, int ah,
            int out_w, int out_h,
            int x, int y
        ) {
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

        extern "C" __global__
        void composite_green_nv12_y(
            const unsigned char* __restrict__ src_nv12,
            const float* __restrict__ alpha_lr,
            int out_w, int out_h,
            int aw, int ah,
            unsigned char gY,
            float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
            unsigned char* __restrict__ out_nv12,
            const float* __restrict__ light_coeffs,
            const unsigned char* __restrict__ light_gamma_lut,
            int light_identity
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= out_w || y >= out_h) return;
            int idx = y * out_w + x;
            float a = adjust_alpha(
                sample_alpha_lr_y(alpha_lr, aw, ah, out_w, out_h, x, y),
                alpha_cutoff, alpha_hard_edge, alpha_contrast
            );
            float src_y = (float)src_nv12[idx];
            apply_light_match_y_only(&src_y, light_coeffs, light_gamma_lut, light_identity);
            float yv = src_y * a + (float)gY * (1.f - a);
            yv = yv < 0.f ? 0.f : (yv > 255.f ? 255.f : yv);
            out_nv12[idx] = (unsigned char)(yv + 0.5f);
        }
        """
        _composite_nv12_y_kernel = _cp.RawKernel(
            _COMPOSITE_NV12_Y_KERNEL_SRC, "composite_green_nv12_y"
        )

        _COMPOSITE_NV12_UV_KERNEL_SRC = _LIGHT_MATCH_DEVICE_SRC + r"""
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

        __device__ float sample_alpha_lr_uv(
            const float* __restrict__ alpha_lr,
            int aw, int ah,
            int out_w, int out_h,
            int x, int y
        ) {
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

        extern "C" __global__
        void composite_green_nv12_uv(
            const unsigned char* __restrict__ src_nv12,
            const float* __restrict__ alpha_lr,
            int out_w, int out_h,
            int aw, int ah,
            unsigned char gU, unsigned char gV,
            int fast_uv_alpha,
            float alpha_cutoff, int alpha_hard_edge, float alpha_contrast,
            unsigned char* __restrict__ out_nv12,
            const float* __restrict__ light_coeffs,
            int light_identity
        ) {
            int ux = blockIdx.x * blockDim.x + threadIdx.x;
            int uy = blockIdx.y * blockDim.y + threadIdx.y;
            int uv_w = out_w >> 1;
            int uv_h = out_h >> 1;
            if (ux >= uv_w || uy >= uv_h) return;
            int x = ux << 1;
            int y = uy << 1;
            float a = adjust_alpha(
                sample_alpha_lr_uv(alpha_lr, aw, ah, out_w, out_h, x, y),
                alpha_cutoff, alpha_hard_edge, alpha_contrast
            );
            float auv = a;
            if (!fast_uv_alpha) {
                float a01 = adjust_alpha(sample_alpha_lr_uv(alpha_lr, aw, ah, out_w, out_h, x + 1, y), alpha_cutoff, alpha_hard_edge, alpha_contrast);
                float a10 = adjust_alpha(sample_alpha_lr_uv(alpha_lr, aw, ah, out_w, out_h, x, y + 1), alpha_cutoff, alpha_hard_edge, alpha_contrast);
                float a11 = adjust_alpha(sample_alpha_lr_uv(alpha_lr, aw, ah, out_w, out_h, x + 1, y + 1), alpha_cutoff, alpha_hard_edge, alpha_contrast);
                auv = 0.25f * (a + a01 + a10 + a11);
            }
            int uv_idx = out_w * out_h + uy * out_w + x;
            float src_u = (float)src_nv12[uv_idx];
            float src_v = (float)src_nv12[uv_idx + 1];
            apply_light_match_uv_only(&src_u, &src_v, light_coeffs, light_identity);
            float u = src_u * auv + (float)gU * (1.f - auv);
            float v = src_v * auv + (float)gV * (1.f - auv);
            u = u < 0.f ? 0.f : (u > 255.f ? 255.f : u);
            v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
            out_nv12[uv_idx] = (unsigned char)(u + 0.5f);
            out_nv12[uv_idx + 1] = (unsigned char)(v + 0.5f);
        }
        """
        _composite_nv12_uv_kernel = _cp.RawKernel(
            _COMPOSITE_NV12_UV_KERNEL_SRC, "composite_green_nv12_uv"
        )

        # CUDA timing
        _cp.cuda.Device(0).use()
        _GPU_OK = True
        # print stderr logger setup
        print(f"[matting] CuPy {_cp.__version__} detected, GPU composite enabled.", file=sys.stderr)
        log.info("CuPy detected: %s, GPU composite enabled.", _cp.__version__)
    except Exception as e:
        if MATTING_DEVICE == "gpu":
            raise RuntimeError(f"PT_MATTING_DEVICE=gpu but CuPy unavailable: {e}")
        print(f"[matting] CuPy unavailable, falling back to CPU composite: {e}", file=sys.stderr)
        log.warning("CuPy unavailable, fallback to CPU composite: %s", e)
else:
    print(f"[matting] PT_MATTING_DEVICE={MATTING_DEVICE} -> CPU composite path.", file=sys.stderr)

print(
    f"[matting] options: input_size={MATTING_INPUT_SIZE} square={MATTING_SQUARE} "
    f"split_sbs={MATTING_SPLIT_SBS} sbs_batch={MATTING_SBS_BATCH} "
    f"alpha_stride={ALPHA_STRIDE} alpha_mode={ALPHA_MODE} "
    f"alpha_cutoff={ALPHA_CUTOFF} alpha_hard_edge={ALPHA_HARD_EDGE} "
    f"alpha_contrast={ALPHA_CONTRAST}",
    file=sys.stderr,
)


def matter_device() -> str:
    return "gpu" if _GPU_OK else "cpu"


def _detect_model_kind(model_path: Path) -> str:
    if MATTING_MODEL_KIND in {"rmbg", "ben2", "rvm"}:
        return MATTING_MODEL_KIND
    name = model_path.name.lower()
    if name.startswith("rvm_") or "_rvm_" in name:
        return "rvm"
    if "ben2" in name:
        return "ben2"
    if "rmbg" in name:
        return "rmbg"
    return "rvm"


class MattingTiming(NamedTuple):
    preprocess_ms: float
    ort_ms: float
    alpha_resize_ms: float
    composite_ms: float


class Nv12OutputSlot(NamedTuple):
    index: int
    buffer: object


@dataclass(frozen=True)
class _GpuPlane:
    data: object

    def as_cupy(self):
        return self.data


@dataclass(frozen=True)
class ZeroGpuFrame:
    height: int
    width: int
    y: _GpuPlane
    uv: _GpuPlane


def make_zero_gpu_frame(h: int, w: int, bit_depth: int = 8) -> ZeroGpuFrame:
    """Return a duck-typed GPU NV12/P016 frame for startup warmup."""
    if not _GPU_OK or _cp is None:
        raise RuntimeError("make_zero_gpu_frame requires CuPy/GPU matting")
    h = max(2, int(h) & ~1)
    w = max(2, int(w) & ~1)
    if int(bit_depth) > 8:
        y = _cp.zeros((h, w), dtype=_cp.uint16)
        uv = _cp.zeros((h // 2, w), dtype=_cp.uint16)
    else:
        y = _cp.zeros((h, w), dtype=_cp.uint8)
        uv = _cp.full((h // 2, w), 128, dtype=_cp.uint8)
    return ZeroGpuFrame(height=h, width=w, y=_GpuPlane(y), uv=_GpuPlane(uv))


def _filter_available_providers(wanted: list[str]) -> list[str]:
    available_list = ort.get_available_providers()
    available = set(available_list)
    out = [p.strip() for p in wanted if p.strip() in available]
    if not out:
        out = ["CPUExecutionProvider"]
    log.info("[DIAG] ONNX providers wanted=%s available=%s selected=%s", wanted, available_list, out)
    return out


def _onnx_tensor_dtype(type_name: str):
    if "float16" in type_name:
        return np.float16
    if type_name in {"tensor(float)", "tensor(float32)"}:
        return np.float32
    return None


def _provider_config(providers: list[str]):
    configured = []
    for provider in providers:
        if provider == "TensorrtExecutionProvider":
            trt_options = {
                "trt_fp16_enable": "True" if ONNX_TRT_FP16_ENABLE else "False",
                "trt_engine_cache_enable": "True" if ONNX_TRT_ENGINE_CACHE_ENABLE else "False",
                "trt_cuda_graph_enable": "True" if ONNX_TRT_CUDA_GRAPH_ENABLE else "False",
            }
            if ONNX_TRT_ENGINE_CACHE_ENABLE:
                ONNX_TRT_ENGINE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
                trt_options["trt_engine_cache_path"] = str(ONNX_TRT_ENGINE_CACHE_PATH)
            if ONNX_TRT_DUMP_SUBGRAPHS:
                trt_options["trt_dump_subgraphs"] = "True"
            if ONNX_TRT_DETAILED_BUILD_LOG:
                trt_options["trt_detailed_build_log"] = "True"
            configured.append((provider, trt_options))
        elif provider == "CUDAExecutionProvider" and _CUDA_STREAM is not None and CUDA_SHARED_STREAM:
            cuda_options = {
                "user_compute_stream": str(int(_CUDA_STREAM.ptr)),
                "do_copy_in_default_stream": "0",
            }
            cudnn_search = CUDA_CUDNN_CONV_ALGO_SEARCH.strip()
            if cudnn_search:
                cuda_options["cudnn_conv_algo_search"] = cudnn_search
            configured.append((
                provider,
                cuda_options,
            ))
        else:
            configured.append(provider)
    return configured


class Matter:
    def __init__(self, model_path: Path = MODEL_PATH, load_model: bool = True, warmup_runs: int | None = None):
        self.model_kind = "utility"
        self._norm_scale = 1.0 / 255.0
        self._norm_bias = 0.0
        self.input_dtype = np.float32
        self.input_name = "src"
        self.input_type = "tensor(float)"
        self.input_shape = []
        self.input_names = []
        self.output_names = []
        self.output_name = ""
        self._model_supports_batch2 = False
        self._supports_batch2 = False
        self.rvm_downsample_dtype = np.float32
        self._rvm_rec: list[np.ndarray] | None = None
        self._rvm_rec_ort: list[ort.OrtValue] | None = None
        self._rvm_io_sig: tuple[int, int, int] | None = None
        self._rvm_io_outputs: dict[str, ort.OrtValue] = {}
        self._rvm_io_downsample: ort.OrtValue | None = None
        self._rvm_state_slots: dict[str, dict[str, object]] = {}
        self._rvm_input_channels: dict[str, int] = {}
        self._rvm_output_channels: dict[str, int] = {}
        self._rvm_trt_active = False
        self._rvm_static_trt_available = False
        self._rvm_iobinding_enabled = False
        self._rvm_iobinding_failed = False
        self._trt_static_sessions: dict[tuple[int, int, int], object] = {}
        self._trt_static_states: dict[tuple[int, int, int], list[ort.OrtValue]] = {}
        self._trt_static_outputs: dict[tuple[int, int, int], dict[str, ort.OrtValue]] = {}
        self._scene_detector = _SceneCutDetector(
            threshold=RVM_SCENE_THRESHOLD,
            cooldown_frames=RVM_SCENE_COOLDOWN,
            ref_ema_alpha=RVM_SCENE_REF_EMA,
        ) if RVM_SCENE_RESET else None
        self._alpha_smoother = _AlphaSmoother(RVM_ALPHA_SMOOTH_WEIGHT) if RVM_ALPHA_SMOOTH else None
        self._tmp_float: np.ndarray | None = None
        self._call_count = 0
        self._last_ort_shape = ""
        self._temporal_frame_idx = 0
        self._cached_alpha_small: np.ndarray | None = None
        self._cached_alpha_shape: tuple[int, int, int, int, int] | None = None
        self._cached_alpha_ort_shape = ""
        self._g_frame = None
        self._g_alpha = None
        self._g_bypass_alpha = None
        self._g_out = None
        self._nv12_slots: list[object] = []
        self._nv12_slot_shape: tuple[int, int] | None = None
        self._nv12_slot_in_use: list[bool] = []
        self._nv12_next_slot = 0
        self._nv12_slot_cond = threading.Condition()
        self._g_chw = None
        self._p016_to_nv12_kernel = None
        self._h_chw = None
        self._h_chw_mem = None
        self._h_out = None
        self._h_out_mem = None
        self._h_out_nv12 = None
        self._h_out_nv12_mem = None
        self._sync_probe_count = 0
        self._preproc_diag_count = 0
        self._light_match_version = -1
        self._light_match_identity = 1
        self._g_light_coeffs = None
        self._g_light_gamma_lut = None
        if not load_model:
            print("[matting] utility mode: ONNX matting model not loaded", file=sys.stderr)
            return
        if not model_path.exists():
            raise FileNotFoundError(
                f"RVM ONNX not found: {model_path}. "
                "Download rvm_mobilenetv3_fp32.onnx into models/."
            )
        self.model_kind = _detect_model_kind(model_path)
        if self.model_kind == "rmbg":
            self._norm_scale = 1.0 / 255.0
            self._norm_bias = -0.5
        elif self.model_kind in {"ben2", "rvm"}:
            self._norm_scale = 1.0 / 255.0
            self._norm_bias = 0.0
        else:
            self._norm_scale = 1.0 / 127.5
            self._norm_bias = -1.0
        providers = _filter_available_providers(ONNX_PROVIDERS)
        self._rvm_static_trt_available = (
            self.model_kind == "rvm"
            and _GPU_OK
            and "TensorrtExecutionProvider" in providers
            and self._trt_static_model_path(1, MATTING_INPUT_SIZE, MATTING_INPUT_SIZE) is not None
            and self._trt_static_model_path(2, MATTING_INPUT_SIZE, MATTING_INPUT_SIZE) is not None
        )
        session_providers = providers
        if self._rvm_static_trt_available:
            session_providers = [provider for provider in providers if provider != "TensorrtExecutionProvider"]
            log.info("[DIAG] static TensorRT RVM cache detected; loading main RVM session without dynamic TensorRT EP")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if "DmlExecutionProvider" in session_providers:
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider_config = _provider_config(session_providers)
        self.sess = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=provider_config
        )
        self.input_metas = self.sess.get_inputs()
        self.output_metas = self.sess.get_outputs()
        self.input_names = [i.name for i in self.input_metas]
        self.output_names = [o.name for o in self.output_metas]
        self._rvm_input_channels = {}
        for meta in self.input_metas:
            shape = list(meta.shape)
            if len(shape) >= 2 and isinstance(shape[1], int):
                self._rvm_input_channels[meta.name] = int(shape[1])
        self._rvm_output_channels = {}
        for meta in self.output_metas:
            shape = list(meta.shape)
            if len(shape) >= 2 and isinstance(shape[1], int):
                self._rvm_output_channels[meta.name] = int(shape[1])
        self.input_dtypes = {
            meta.name: _onnx_tensor_dtype(meta.type) for meta in self.input_metas
        }
        input_meta = self.input_metas[0]
        self.input_name = input_meta.name
        self.input_type = input_meta.type
        self.input_shape = list(input_meta.shape)
        self._model_supports_batch2 = self._detect_batch2_support(self.input_shape)
        self._supports_batch2 = MATTING_SBS_BATCH and self._model_supports_batch2
        self.input_dtype = self.input_dtypes.get(self.input_name)
        if self.input_dtype is None:
            raise ValueError(f"Unsupported matting model input type: {self.input_type}")
        self.rvm_downsample_dtype = self.input_dtypes.get("downsample_ratio")
        if self.rvm_downsample_dtype is None and len(self.input_names) >= 6:
            self.rvm_downsample_dtype = self.input_dtypes.get(self.input_names[5])
        if self.rvm_downsample_dtype is None:
            self.rvm_downsample_dtype = np.float32
        self.output_name = self.output_metas[0].name
        active_providers = self.sess.get_providers()
        self._rvm_trt_active = "TensorrtExecutionProvider" in active_providers
        self._rvm_iobinding_enabled = self.model_kind == "rvm" and _GPU_OK and _should_enable_rvm_iobinding(active_providers)
        if self._rvm_static_trt_available:
            self._rvm_iobinding_enabled = True
        log.info(
            "Matting model loaded: kind=%s input=%s shape=%s type=%s output=%s wanted=%s active=%s composite_device=%s batch2=%s model_batch2=%s rvm_iobinding=%s",
            self.model_kind, self.input_name, self.input_shape, self.input_type, self.output_name, providers, active_providers,
            matter_device(),
            self._supports_batch2,
            self._model_supports_batch2,
            self._rvm_iobinding_enabled,
        )
        print(
            f"[matting] model kind={self.model_kind} input={self.input_name} shape={self.input_shape} "
            f"type={self.input_type} batch2={self._supports_batch2} "
            f"model_batch2={self._model_supports_batch2} "
            f"rvm_iobinding={self._rvm_iobinding_enabled} "
            f"static_trt={self._rvm_static_trt_available} "
            f"providers={active_providers} inputs={self.input_names} outputs={self.output_names}",
            file=sys.stderr,
        )
        self.warmup(MATTING_WARMUP_RUNS if warmup_runs is None else warmup_runs)

    def _trt_static_model_path(self, batch: int, h: int, w: int) -> Path | None:
        if self.model_kind != "rvm":
            return None
        if h != MATTING_INPUT_SIZE or w != MATTING_INPUT_SIZE:
            return None
        try:
            from utils.rvm_static_onnx import static_rvm_model_path
            from utils.trt_manifest import original_rvm_model_path

            path = static_rvm_model_path(original_rvm_model_path(), ONNX_TRT_ENGINE_CACHE_PATH, batch, MATTING_INPUT_SIZE, RVM_DOWNSAMPLE_RATIO)
        except Exception:
            return None
        return path if path.exists() else None

    def _rvm_provider_diag(self) -> dict[str, object]:
        return {
            "main": self.sess.get_providers(),
            "static_trt": self._rvm_static_trt_available,
            "iobinding": self._rvm_iobinding_enabled and not self._rvm_iobinding_failed,
        }

    def _get_trt_static_session(self, batch: int, h: int, w: int):
        if not self._rvm_static_trt_available:
            return None
        key = (batch, h, w)
        session = self._trt_static_sessions.get(key)
        if session is not None:
            return session
        path = self._trt_static_model_path(batch, h, w)
        if path is None:
            return None
        providers = _filter_available_providers(ONNX_PROVIDERS)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not (ONNX_TRT_DUMP_SUBGRAPHS or ONNX_TRT_DETAILED_BUILD_LOG):
            sess_opts.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=sess_opts, providers=_provider_config(providers))
        if "TensorrtExecutionProvider" not in session.get_providers():
            log.warning("[DIAG] static TensorRT RVM session did not activate TRT: %s", session.get_providers())
            self._rvm_static_trt_available = False
            self._rvm_iobinding_enabled = self.model_kind == "rvm" and _GPU_OK and _should_enable_rvm_iobinding(self.sess.get_providers())
            return None
        self._trt_static_sessions[key] = session
        log.info("[DIAG] static TensorRT RVM session loaded: batch=%d shape=%dx%d model=%s", batch, w, h, path)
        return session

    def _ensure_light_match_tables(self):
        cp = _cp
        if cp is None:
            return None, None, np.int32(1)
        state = get_light_match()
        if (
            self._g_light_coeffs is None
            or self._g_light_gamma_lut is None
            or self._light_match_version != state.version
        ):
            tables = build_light_match_tables(state.params())
            coeffs = tables.coeffs.astype(np.float32, copy=False)
            gamma_lut = tables.gamma_lut.astype(np.uint8, copy=False)
            if self._g_light_coeffs is None or self._g_light_coeffs.shape != coeffs.shape:
                self._g_light_coeffs = cp.empty(coeffs.shape, dtype=cp.float32)
            if self._g_light_gamma_lut is None or self._g_light_gamma_lut.shape != gamma_lut.shape:
                self._g_light_gamma_lut = cp.empty(gamma_lut.shape, dtype=cp.uint8)
            self._g_light_coeffs.set(coeffs)
            self._g_light_gamma_lut.set(gamma_lut)
            self._light_match_identity = 1 if tables.identity else 0
            self._light_match_version = state.version
            if state.version > 0 or not tables.identity:
                log.info(
                    "light match updated: enabled=%s temp_k=%d tint=%.1f ev=%.2f contrast=%.2f gamma=%.2f saturation=%.2f preset=%s version=%d identity=%s",
                    state.enabled,
                    state.temp_k,
                    state.tint,
                    state.exposure_ev,
                    state.contrast,
                    state.gamma,
                    state.saturation,
                    state.preset,
                    state.version,
                    bool(tables.identity),
                )
        return self._g_light_coeffs, self._g_light_gamma_lut, np.int32(self._light_match_identity)

    def light_match_kernel_args(self, include_gamma: bool = True):
        coeffs, gamma_lut, identity = self._ensure_light_match_tables()
        if include_gamma:
            return coeffs, gamma_lut, identity
        return coeffs, identity

    def reset_state(self) -> None:
        """Clear temporal matting state between independent videos/requests."""
        self._rvm_rec = None
        self._rvm_rec_ort = None
        self._rvm_rec_sig = None
        self._rvm_io_sig = None
        self._rvm_io_outputs = {}
        self._rvm_io_downsample = None
        self._trt_static_states = {}
        self._trt_static_outputs = {}
        self._rvm_state_slots = {}
        self._rvm_iobinding_failed = False
        if self._scene_detector is not None:
            self._scene_detector.reset()
        if self._alpha_smoother is not None:
            self._alpha_smoother.reset()
        self._cached_alpha_small = None
        self._cached_alpha_shape = None
        self._cached_alpha_ort_shape = ""
        self._temporal_frame_idx = 0
        with self._nv12_slot_cond:
            for i in range(len(self._nv12_slot_in_use)):
                self._nv12_slot_in_use[i] = False
            self._nv12_slot_cond.notify_all()

    @staticmethod
    def _detect_batch2_support(shape: list) -> bool:
        if not shape:
            return False
        batch_dim = shape[0]
        if isinstance(batch_dim, int):
            return batch_dim >= 2
        return batch_dim is None or isinstance(batch_dim, str)

    def warmup(self, runs: int) -> None:
        if runs <= 0:
            return
        import time as _time

        warmup_size = MATTING_INPUT_SIZE if MATTING_INPUT_SIZE > 0 else 1024
        frame = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
        start = _time.perf_counter()
        for _ in range(runs):
            self.alpha(frame)
        # GPU kernel JIT
        try:
            _ = self.composite_green(frame)
        except Exception as e:
            log.warning("warmup composite failed: %s", e)
        elapsed = (_time.perf_counter() - start) * 1000
        log.info("[DIAG] matting warmup: runs=%d elapsed=%.1fms device=%s", runs, elapsed, matter_device())

    # ---------- preprocess / ONNX ----------
    def _matting_size_for(self, w: int, h: int) -> tuple[int, int]:
        """ _resize_to_matting_input target shape GPU """
        ref = MATTING_INPUT_SIZE
        if self.model_kind == "rvm" and ref <= 0:
            new_w, new_h = w, h
            new_w -= new_w % 32
            new_h -= new_h % 32
            return max(new_w, 32), max(new_h, 32)
        if self.model_kind in {"rmbg", "ben2"}:
            return ref, ref
        if MATTING_SQUARE:
            return ref, ref
        if min(h, w) > ref:
            if w >= h:
                new_h = ref
                new_w = int(w / h * ref)
            else:
                new_w = ref
                new_h = int(h / w * ref)
        else:
            new_h, new_w = h, w
        new_w -= new_w % 32
        new_h -= new_h % 32
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)
        return new_w, new_h

    # ---- pinned host ----
    def _alloc_pinned(self, shape: tuple, dtype):
        """ pinned host memory numpy (PinnedMemoryPointer, ndarray) """
        cp = _cp
        n = int(np.prod(shape))
        nbytes = n * np.dtype(dtype).itemsize
        mem = cp.cuda.alloc_pinned_memory(nbytes)
        arr = np.frombuffer(mem, dtype=dtype, count=n).reshape(shape)
        return mem, arr

    def _ensure_pinned_chw(self, out_w: int, out_h: int, batch: int = 1) -> np.ndarray:
        shape = (batch, 3, out_h, out_w)
        if (self._h_chw is None or self._h_chw.shape != shape
                or self._h_chw.dtype != self.input_dtype):
            self._h_chw_mem, self._h_chw = self._alloc_pinned(shape, self.input_dtype)
        return self._h_chw

    def _ensure_pinned_out(self, h: int, w: int) -> np.ndarray:
        shape = (h, w, 3)
        if self._h_out is None or self._h_out.shape != shape:
            self._h_out_mem, self._h_out = self._alloc_pinned(shape, np.uint8)
        return self._h_out

    def _ensure_pinned_nv12_out(self, h: int, w: int) -> np.ndarray:
        shape = (h * 3 // 2, w)
        if self._h_out_nv12 is None or self._h_out_nv12.shape != shape:
            self._h_out_nv12_mem, self._h_out_nv12 = self._alloc_pinned(shape, np.uint8)
        return self._h_out_nv12

    def _ensure_dev_chw(self, out_w: int, out_h: int, batch: int = 1):
        cp = _cp
        cp_dtype = cp.float16 if self.input_dtype == np.float16 else cp.float32
        if (self._g_chw is None or self._g_chw.shape != (batch, 3, out_h, out_w)
                or self._g_chw.dtype != cp_dtype):
            self._g_chw = cp.empty((batch, 3, out_h, out_w), dtype=cp_dtype)
        return self._g_chw

    def _ensure_dev_frame(self, frame_shape: tuple):
        cp = _cp
        if self._g_frame is None or self._g_frame.shape != frame_shape:
            self._g_frame = cp.empty(frame_shape, dtype=cp.uint8)
            self._g_out = cp.empty(frame_shape, dtype=cp.uint8)
        return self._g_frame

    def _upload_frame_gpu(self, frame_bgr: np.ndarray) -> None:
        """ _g_frame preprocess composite device """
        self._ensure_dev_frame(frame_bgr.shape)
        self._g_frame.set(frame_bgr)

    def _upload_nv12_gpu(self, frame_nv12: np.ndarray, h: int, w: int) -> None:
        cp = _cp
        shape = (h * 3 // 2, w)
        if self._g_frame is None or self._g_frame.shape != shape:
            self._g_frame = cp.empty(shape, dtype=cp.uint8)
            self._g_out = cp.empty((h, w, 3), dtype=cp.uint8)
        self._g_frame.set(frame_nv12.reshape(shape))

    def upload_nv12_planes_gpu(self, y_dev, uv_dev, h: int, w: int) -> None:
        """Experimental: load GPU NV12 planes into the existing contiguous NV12 layout."""
        cp = _cp
        shape = (h * 3 // 2, w)
        if self._g_frame is None or self._g_frame.shape != shape:
            self._g_frame = cp.empty(shape, dtype=cp.uint8)
            self._g_out = cp.empty((h, w, 3), dtype=cp.uint8)
        self._g_frame[:h, :] = y_dev.reshape(h, w)
        self._g_frame[h:, :] = uv_dev.reshape(h // 2, w)

    def pynv_scaled_size(self, w: int, h: int) -> tuple[int, int]:
        """Return the PyNv post-decode GPU processing size for PT_DECODE_MAX_SIDE."""
        max_side = int(DECODE_MAX_SIDE or 0)
        w = max(2, int(w))
        h = max(2, int(h))
        if max_side <= 0 or max(w, h) <= max_side:
            return w & ~1, h & ~1
        if w >= h:
            out_w = max_side
            out_h = int(round(h * max_side / w))
        else:
            out_h = max_side
            out_w = int(round(w * max_side / h))
        return max(2, out_w & ~1), max(2, out_h & ~1)

    def _ensure_nv12_resize_kernel(self):
        cp = _cp
        if getattr(self, "_nv12_resize_kernel", None) is None:
            self._nv12_resize_kernel = cp.RawKernel(r"""
            extern "C" __global__
            void resize_nv12_to_nv12(
                const unsigned char* y_src,
                const unsigned char* uv_src,
                unsigned char* dst,
                int src_w,
                int src_h,
                int dst_w,
                int dst_h,
                int y_stride,
                int uv_stride)
            {
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
                float v00 = (float)y_src[y0 * y_stride + x0];
                float v01 = (float)y_src[y0 * y_stride + x1];
                float v10 = (float)y_src[y1 * y_stride + x0];
                float v11 = (float)y_src[y1 * y_stride + x1];
                float yf = (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                         +        dy  * ((1.f - dx) * v10 + dx * v11);
                dst[y * dst_w + x] = (unsigned char)(yf + 0.5f);

                if (((x | y) & 1) == 0) {
                    int ux = ((int)floorf(sx)) & ~1;
                    int uy = ((int)floorf(sy)) >> 1;
                    if (ux < 0) ux = 0;
                    if (ux > src_w - 2) ux = src_w - 2;
                    if (uy < 0) uy = 0;
                    int max_uy = (src_h >> 1) - 1;
                    if (uy > max_uy) uy = max_uy;
                    int src_i = uy * uv_stride + ux;
                    int dst_i = dst_w * dst_h + (y >> 1) * dst_w + x;
                    dst[dst_i] = uv_src[src_i];
                    dst[dst_i + 1] = uv_src[src_i + 1];
                }
            }
            """, "resize_nv12_to_nv12")
        return self._nv12_resize_kernel

    def upload_nv12_planes_gpu_scaled(self, y_dev, uv_dev, src_h: int, src_w: int, out_h: int, out_w: int) -> None:
        cp = _cp
        src_h, src_w = int(src_h), int(src_w)
        out_h, out_w = int(out_h), int(out_w)
        if out_h == src_h and out_w == src_w:
            self.upload_nv12_planes_gpu(y_dev, uv_dev, src_h, src_w)
            return
        shape = (out_h * 3 // 2, out_w)
        if self._g_frame is None or self._g_frame.shape != shape:
            self._g_frame = cp.empty(shape, dtype=cp.uint8)
            self._g_out = cp.empty((out_h, out_w, 3), dtype=cp.uint8)
        y_stride = int(y_dev.strides[0]) if y_dev.strides else int(src_w)
        uv_stride = int(uv_dev.strides[0]) if uv_dev.strides else int(src_w)
        block = (32, 8, 1)
        grid = ((out_w + block[0] - 1) // block[0], (out_h + block[1] - 1) // block[1], 1)
        self._ensure_nv12_resize_kernel()(
            grid,
            block,
            (
                y_dev,
                uv_dev,
                self._g_frame,
                np.int32(src_w),
                np.int32(src_h),
                np.int32(out_w),
                np.int32(out_h),
                np.int32(y_stride),
                np.int32(uv_stride),
            ),
        )

    def _ensure_p016_to_nv12_kernel(self):
        cp = _cp
        if self._p016_to_nv12_kernel is None:
            self._p016_to_nv12_kernel = cp.RawKernel(r"""
            extern "C" __global__
            void p016_to_nv12(
                const unsigned short* y_src,
                const unsigned short* uv_src,
                unsigned char* dst,
                int width,
                int height,
                int y_stride_elems,
                int uv_stride_elems,
                int dst_stride,
                int shift_bits)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= width || y >= height) return;

                unsigned short yv = y_src[y * y_stride_elems + x];
                dst[y * dst_stride + x] = (unsigned char)(yv >> shift_bits);

                if ((y & 1) == 0) {
                    int uv_row = y >> 1;
                    unsigned short uvv = uv_src[uv_row * uv_stride_elems + x];
                    dst[(height + uv_row) * dst_stride + x] = (unsigned char)(uvv >> shift_bits);
                }
            }
            """, "p016_to_nv12")
        return self._p016_to_nv12_kernel

    def upload_p016_planes_as_nv12_gpu(self, y_dev, uv_dev, h: int, w: int, shift_bits: int = 8) -> None:
        """Convert GPU P016/P010-like planes into the existing contiguous NV12 layout."""
        cp = _cp
        shape = (h * 3 // 2, w)
        if self._g_frame is None or self._g_frame.shape != shape:
            self._g_frame = cp.empty(shape, dtype=cp.uint8)
            self._g_out = cp.empty((h, w, 3), dtype=cp.uint8)
        y_stride_bytes = int(y_dev.strides[0]) if y_dev.strides else int(w * y_dev.dtype.itemsize)
        uv_stride_bytes = int(uv_dev.strides[0]) if uv_dev.strides else int(w * uv_dev.dtype.itemsize)
        y_stride_elems = y_stride_bytes // y_dev.dtype.itemsize
        uv_stride_elems = uv_stride_bytes // uv_dev.dtype.itemsize
        block = (32, 8, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        self._ensure_p016_to_nv12_kernel()(
            grid,
            block,
            (
                y_dev,
                uv_dev,
                self._g_frame,
                np.int32(w),
                np.int32(h),
                np.int32(y_stride_elems),
                np.int32(uv_stride_elems),
                np.int32(w),
                np.int32(shift_bits),
            ),
        )

    def _ensure_p016_resize_to_nv12_kernel(self):
        cp = _cp
        if getattr(self, "_p016_resize_to_nv12_kernel", None) is None:
            self._p016_resize_to_nv12_kernel = cp.RawKernel(r"""
            extern "C" __global__
            void resize_p016_to_nv12(
                const unsigned short* y_src,
                const unsigned short* uv_src,
                unsigned char* dst,
                int src_w,
                int src_h,
                int dst_w,
                int dst_h,
                int y_stride_elems,
                int uv_stride_elems,
                int shift_bits)
            {
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
                float v00 = (float)(y_src[y0 * y_stride_elems + x0] >> shift_bits);
                float v01 = (float)(y_src[y0 * y_stride_elems + x1] >> shift_bits);
                float v10 = (float)(y_src[y1 * y_stride_elems + x0] >> shift_bits);
                float v11 = (float)(y_src[y1 * y_stride_elems + x1] >> shift_bits);
                float yf = (1.f - dy) * ((1.f - dx) * v00 + dx * v01)
                         +        dy  * ((1.f - dx) * v10 + dx * v11);
                dst[y * dst_w + x] = (unsigned char)(yf + 0.5f);

                if (((x | y) & 1) == 0) {
                    int ux = ((int)floorf(sx)) & ~1;
                    int uy = ((int)floorf(sy)) >> 1;
                    if (ux < 0) ux = 0;
                    if (ux > src_w - 2) ux = src_w - 2;
                    if (uy < 0) uy = 0;
                    int max_uy = (src_h >> 1) - 1;
                    if (uy > max_uy) uy = max_uy;
                    int src_i = uy * uv_stride_elems + ux;
                    int dst_i = dst_w * dst_h + (y >> 1) * dst_w + x;
                    dst[dst_i] = (unsigned char)(uv_src[src_i] >> shift_bits);
                    dst[dst_i + 1] = (unsigned char)(uv_src[src_i + 1] >> shift_bits);
                }
            }
            """, "resize_p016_to_nv12")
        return self._p016_resize_to_nv12_kernel

    def upload_p016_planes_as_nv12_gpu_scaled(
        self,
        y_dev,
        uv_dev,
        src_h: int,
        src_w: int,
        out_h: int,
        out_w: int,
        shift_bits: int = 8,
    ) -> None:
        cp = _cp
        src_h, src_w = int(src_h), int(src_w)
        out_h, out_w = int(out_h), int(out_w)
        if out_h == src_h and out_w == src_w:
            self.upload_p016_planes_as_nv12_gpu(y_dev, uv_dev, src_h, src_w, shift_bits=shift_bits)
            return
        shape = (out_h * 3 // 2, out_w)
        if self._g_frame is None or self._g_frame.shape != shape:
            self._g_frame = cp.empty(shape, dtype=cp.uint8)
            self._g_out = cp.empty((out_h, out_w, 3), dtype=cp.uint8)
        y_stride_bytes = int(y_dev.strides[0]) if y_dev.strides else int(src_w * y_dev.dtype.itemsize)
        uv_stride_bytes = int(uv_dev.strides[0]) if uv_dev.strides else int(src_w * uv_dev.dtype.itemsize)
        y_stride_elems = y_stride_bytes // y_dev.dtype.itemsize
        uv_stride_elems = uv_stride_bytes // uv_dev.dtype.itemsize
        block = (32, 8, 1)
        grid = ((out_w + block[0] - 1) // block[0], (out_h + block[1] - 1) // block[1], 1)
        self._ensure_p016_resize_to_nv12_kernel()(
            grid,
            block,
            (
                y_dev,
                uv_dev,
                self._g_frame,
                np.int32(src_w),
                np.int32(src_h),
                np.int32(out_w),
                np.int32(out_h),
                np.int32(y_stride_elems),
                np.int32(uv_stride_elems),
                np.int32(shift_bits),
            ),
        )

    def composite_green_gpu_nv12_frame_to_nv12_profile(
        self,
        frame_gpu_nv12,
        out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, MattingTiming]:
        """Experimental PyNvVideoCodec path: GPU NV12 planes -> matting -> NV12 host output."""
        import time as _time

        if not _GPU_OK:
            raise RuntimeError("GPU NV12 frame path requires CuPy/GPU composite")
        h, w = int(frame_gpu_nv12.height), int(frame_gpu_nv12.width)
        stream = _CUDA_STREAM
        ctx = stream if stream is not None else nullcontext()
        with ctx:
            t_up0 = _time.perf_counter()
            self.upload_nv12_planes_gpu(frame_gpu_nv12.y.as_cupy(), frame_gpu_nv12.uv.as_cupy(), h, w)
            t_up1 = _time.perf_counter()
            a_small, timing, _ = self._alpha_low_res_gpu_temporal(h, w, use_nv12=True)
            t0 = _time.perf_counter()
            out = self._composite_nv12_to_nv12_using_uploaded_frame(a_small, h, w, out=out)
            t1 = _time.perf_counter()
        return out, MattingTiming(
            timing.preprocess_ms,
            timing.ort_ms,
            0.0,
            (t1 - t0) * 1000 + (t_up1 - t_up0) * 1000,
        )

    def _ensure_dev_nv12_out(self, h: int, w: int):
        cp = _cp
        shape = (h * 3 // 2, w)
        if self._g_out is None or self._g_out.shape != shape:
            self._g_out = cp.empty(shape, dtype=cp.uint8)
        return self._g_out

    def _reset_nv12_slots(self, shape: tuple[int, int]) -> None:
        self._nv12_slots = []
        self._nv12_slot_in_use = []
        self._nv12_next_slot = 0
        self._nv12_slot_shape = shape

    def acquire_nv12_output_slot(self, h: int, w: int, timeout: float | None = None) -> Nv12OutputSlot:
        """Return an exclusive GPU NV12 output slot for one composite/encode handoff."""
        import time as _time

        cp = _cp
        if cp is None:
            raise RuntimeError("NV12 output slots require CuPy")
        shape = (h * 3 // 2, w)
        count = max(1, int(PASSTHROUGH_NV12_RING_SLOTS))
        deadline = None if timeout is None else _time.perf_counter() + max(0.0, float(timeout))
        with self._nv12_slot_cond:
            if self._nv12_slot_shape != shape or len(self._nv12_slots) != count:
                self._reset_nv12_slots(shape)
                self._nv12_slots = [cp.empty(shape, dtype=cp.uint8) for _ in range(count)]
                self._nv12_slot_in_use = [False] * count
                self._nv12_slot_cond.notify_all()
            while True:
                for offset in range(count):
                    idx = (self._nv12_next_slot + offset) % count
                    if not self._nv12_slot_in_use[idx]:
                        self._nv12_slot_in_use[idx] = True
                        self._nv12_next_slot = (idx + 1) % count
                        return Nv12OutputSlot(idx, self._nv12_slots[idx])
                if timeout is None:
                    raise RuntimeError(f"no free NV12 output slot: count={count} shape={shape}")
                remaining = deadline - _time.perf_counter() if deadline is not None else 0.0
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for NV12 output slot: count={count} shape={shape}")
                self._nv12_slot_cond.wait(min(remaining, 0.05))

    def release_nv12_output_slot(self, slot: Nv12OutputSlot | int | None) -> None:
        if slot is None:
            return
        idx = int(slot.index if isinstance(slot, Nv12OutputSlot) else slot)
        with self._nv12_slot_cond:
            if 0 <= idx < len(self._nv12_slot_in_use):
                self._nv12_slot_in_use[idx] = False
                self._nv12_slot_cond.notify()

    def _gpu_preprocess_one(self, src_x0: int, src_w: int,
                             out_w: int, out_h: int,
                             batch: int = 1, batch_idx: int = 0,
                             copy_to_host: bool = True) -> np.ndarray:
        """ _g_frame preprocess kernel pinned host CHW buffer
 (1, 3, out_h, out_w) numpy float32 pinned ORT
 buffer ORT.run """
        cp = _cp
        in_h, in_w = self._g_frame.shape[:2]
        chw_dev = self._ensure_dev_chw(out_w, out_h, batch)
        chw_dev_one = chw_dev[batch_idx]

        block = (16, 16, 1)
        grid = ((out_w + 15) // 16, (out_h + 15) // 16, 1)
        kernel = _preprocess_kernel_fp16 if self.input_dtype == np.float16 else _preprocess_kernel
        kernel(
            grid, block,
            (
                self._g_frame,
                np.int32(in_w), np.int32(in_h),
                np.int32(src_x0), np.int32(src_w),
                np.int32(out_w), np.int32(out_h),
                np.float32(self._norm_scale), np.float32(self._norm_bias),
                chw_dev_one,
            ),
        )
        # pinned host buffer
        if not copy_to_host:
            return chw_dev[batch_idx:batch_idx + 1]
        chw_host = self._ensure_pinned_chw(out_w, out_h, batch)
        cp.asnumpy(chw_dev_one, out=chw_host[batch_idx])
        return chw_host[batch_idx:batch_idx + 1]

    def _gpu_preprocess_nv12_one(self, src_x0: int, src_w: int,
                                  out_w: int, out_h: int,
                                  batch: int = 1, batch_idx: int = 0,
                                  copy_to_host: bool = True) -> np.ndarray:
        cp = _cp
        in_h = self._g_frame.shape[0] * 2 // 3
        in_w = self._g_frame.shape[1]
        chw_dev = self._ensure_dev_chw(out_w, out_h, batch)
        chw_dev_one = chw_dev[batch_idx]

        block = (16, 16, 1)
        grid = ((out_w + 15) // 16, (out_h + 15) // 16, 1)
        kernel = _preprocess_nv12_kernel_fp16 if self.input_dtype == np.float16 else _preprocess_nv12_kernel
        kernel(
            grid, block,
            (
                self._g_frame,
                np.int32(in_w), np.int32(in_h),
                np.int32(src_x0), np.int32(src_w),
                np.int32(out_w), np.int32(out_h),
                np.float32(self._norm_scale), np.float32(self._norm_bias),
                chw_dev_one,
            ),
        )
        if not copy_to_host:
            return chw_dev[batch_idx:batch_idx + 1]
        chw_host = self._ensure_pinned_chw(out_w, out_h, batch)
        cp.asnumpy(chw_dev_one, out=chw_host[batch_idx])
        return chw_host[batch_idx:batch_idx + 1]

    def _gpu_preprocess_nv12_roi_one(
        self,
        roi,
        out_w: int,
        out_h: int,
        batch: int = 1,
        batch_idx: int = 0,
        source_x0: int = 0,
        copy_to_host: bool = True,
    ) -> np.ndarray:
        cp = _cp
        in_h = self._g_frame.shape[0] * 2 // 3
        in_w = self._g_frame.shape[1]
        chw_dev = self._ensure_dev_chw(out_w, out_h, batch)
        chw_dev_one = chw_dev[batch_idx]
        kernel = _preprocess_nv12_roi_kernel_fp16 if self.input_dtype == np.float16 else _preprocess_nv12_roi_kernel
        block = (16, 16, 1)
        grid = ((out_w + 15) // 16, (out_h + 15) // 16, 1)
        kernel(
            grid,
            block,
            (
                self._g_frame,
                np.int32(in_w), np.int32(in_h),
                np.int32(int(source_x0) + roi.x0), np.int32(roi.y0), np.int32(roi.roi_w), np.int32(roi.roi_h),
                np.int32(roi.model_x0), np.int32(roi.model_y0), np.int32(roi.model_w), np.int32(roi.model_h),
                np.int32(out_w), np.int32(out_h),
                np.float32(self._norm_scale), np.float32(self._norm_bias),
                chw_dev_one,
            ),
        )
        if not copy_to_host:
            return chw_dev[batch_idx:batch_idx + 1]
        chw_host = self._ensure_pinned_chw(out_w, out_h, batch)
        cp.asnumpy(chw_dev_one, out=chw_host[batch_idx])
        return chw_host[batch_idx:batch_idx + 1]

    def _gpu_unwarp_roi_alpha_to_eye(self, alpha_roi, roi, eye_w: int, eye_h: int, feather: int = 0):
        cp = _cp
        if hasattr(alpha_roi, "data") and hasattr(alpha_roi.data, "ptr"):
            alpha_dev = alpha_roi.astype(cp.float32, copy=False)
        else:
            alpha_dev = cp.asarray(alpha_roi, dtype=cp.float32)
        if alpha_dev.ndim != 2:
            raise RuntimeError(f"ROI alpha must be 2D, got shape={tuple(alpha_dev.shape)}")
        out = cp.empty((int(eye_h), int(eye_w)), dtype=cp.float32)
        src_h, src_w = (int(v) for v in alpha_dev.shape[:2])
        block = (16, 16, 1)
        grid = ((int(eye_w) + 15) // 16, (int(eye_h) + 15) // 16, 1)
        _alpha_roi_unwarp_kernel(
            grid,
            block,
            (
                alpha_dev,
                out,
                np.int32(eye_w), np.int32(eye_h),
                np.int32(roi.x0), np.int32(roi.y0), np.int32(roi.roi_w), np.int32(roi.roi_h),
                np.int32(roi.model_x0), np.int32(roi.model_y0), np.int32(roi.model_w), np.int32(roi.model_h),
                np.int32(src_w), np.int32(src_h),
                np.int32(max(0, int(feather))),
            ),
        )
        return out

    def _resize_to_matting_input(self, frame_bgr: np.ndarray) -> np.ndarray:
        """CPU matting """
        h, w = frame_bgr.shape[:2]
        new_w, new_h = self._matting_size_for(w, h)

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb *= self._norm_scale
        rgb += self._norm_bias
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        if chw.dtype != self.input_dtype:
            chw = chw.astype(self.input_dtype)
        return chw

    def _run_matting(self, x: np.ndarray) -> np.ndarray:
        """ alpha (H' W' float32) """
        if self.model_kind == "rvm":
            return self._run_rvm(x)[0]
        if x.dtype != self.input_dtype:
            x = x.astype(self.input_dtype)
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        return self._postprocess_alpha(self._extract_alpha(out))

    def _run_matting_batch(self, x: np.ndarray) -> np.ndarray:
        """Return alpha batch as (N, H', W') float32."""
        if self.model_kind == "rvm":
            return self._run_rvm(x)
        if x.dtype != self.input_dtype:
            x = x.astype(self.input_dtype)
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        return self._postprocess_alpha_batch(self._extract_alpha_batch(out))

    def _capture_rvm_state(self) -> dict[str, object]:
        return {
            "rec": self._rvm_rec,
            "rec_ort": self._rvm_rec_ort,
            "rec_sig": getattr(self, "_rvm_rec_sig", None),
            "io_sig": self._rvm_io_sig,
            "io_outputs": self._rvm_io_outputs,
            "io_downsample": self._rvm_io_downsample,
            "trt_static_states": self._trt_static_states,
            "trt_static_outputs": self._trt_static_outputs,
            "alpha_smooth_prev": self._alpha_smoother._prev if self._alpha_smoother is not None else None,
        }

    def _restore_rvm_state(self, state: dict[str, object] | None) -> None:
        if state is None:
            self._rvm_rec = None
            self._rvm_rec_ort = None
            self._rvm_rec_sig = None
            self._rvm_io_sig = None
            self._rvm_io_outputs = {}
            self._rvm_io_downsample = None
            self._trt_static_states = {}
            self._trt_static_outputs = {}
            if self._alpha_smoother is not None:
                self._alpha_smoother._prev = None
            return
        self._rvm_rec = state.get("rec")  # type: ignore[assignment]
        self._rvm_rec_ort = state.get("rec_ort")  # type: ignore[assignment]
        self._rvm_rec_sig = state.get("rec_sig")
        self._rvm_io_sig = state.get("io_sig")  # type: ignore[assignment]
        self._rvm_io_outputs = state.get("io_outputs") or {}  # type: ignore[assignment]
        self._rvm_io_downsample = state.get("io_downsample")  # type: ignore[assignment]
        self._trt_static_states = state.get("trt_static_states") or {}  # type: ignore[assignment]
        self._trt_static_outputs = state.get("trt_static_outputs") or {}  # type: ignore[assignment]
        if self._alpha_smoother is not None:
            self._alpha_smoother._prev = state.get("alpha_smooth_prev")

    def _run_matting_in_rvm_slot(self, slot: str, x: np.ndarray) -> np.ndarray:
        if self.model_kind != "rvm":
            return self._run_matting(x)
        current = self._capture_rvm_state()
        self._restore_rvm_state(self._rvm_state_slots.get(slot))
        try:
            out = self._run_matting(x)
            self._rvm_state_slots[slot] = self._capture_rvm_state()
            return out
        finally:
            self._restore_rvm_state(current)

    def _run_rvm_iobinding_in_slot(self, slot: str, x_dev) -> np.ndarray:
        current = self._capture_rvm_state()
        self._restore_rvm_state(self._rvm_state_slots.get(slot))
        try:
            out = self._run_rvm_iobinding_from_dev(x_dev)
            self._rvm_state_slots[slot] = self._capture_rvm_state()
            return out
        finally:
            self._restore_rvm_state(current)

    def _clear_rvm_temporal_state(self) -> None:
        self._rvm_rec = None
        self._rvm_rec_ort = None
        self._rvm_rec_sig = None
        self._rvm_io_sig = None
        self._rvm_io_outputs = {}
        self._rvm_io_downsample = None
        self._trt_static_states = {}
        self._trt_static_outputs = {}
        self._rvm_state_slots = {}
        self._cached_alpha_small = None
        self._cached_alpha_shape = None
        self._cached_alpha_ort_shape = ""
        if self._alpha_smoother is not None:
            self._alpha_smoother.reset()

    def _smooth_rvm_alpha_batch(self, alpha_batch):
        if self._alpha_smoother is None:
            return alpha_batch
        return self._alpha_smoother.step(alpha_batch)

    def _maybe_reset_rvm_for_scene(self, frame_bgr: np.ndarray) -> None:
        if self.model_kind != "rvm" or self._scene_detector is None:
            return
        if self._scene_detector.step(frame_bgr):
            self._clear_rvm_temporal_state()
            log.debug("scene cut detected, RVM rec reset, frame_idx=%d", self._call_count)

    def _scene_bgr_from_nv12(self, frame_nv12: np.ndarray, h: int, w: int) -> np.ndarray | None:
        if self._scene_detector is None:
            return None
        try:
            split_sbs_active = MATTING_SPLIT_SBS and w >= 2 * h
            src_w = w // 2 if split_sbs_active else w
            src_w -= src_w % 2
            src_h = h - (h % 2)
            if src_w <= 0 or src_h <= 0:
                return None
            target_h = min(self._scene_detector.downsample_height, src_h)
            target_h -= target_h % 2
            target_h = max(2, target_h)
            target_w = max(2, int(src_w * target_h / src_h))
            target_w -= target_w % 2
            target_w = max(2, target_w)
            nv12 = frame_nv12.reshape((h * 3 // 2, w))
            y = nv12[:h, :src_w]
            uv = nv12[h:h + h // 2, :src_w].reshape((h // 2, src_w // 2, 2))
            y_small = cv2.resize(y, (target_w, target_h), interpolation=cv2.INTER_AREA)
            uv_small = cv2.resize(uv, (target_w // 2, target_h // 2), interpolation=cv2.INTER_AREA)
            yuv_small = np.vstack((y_small, uv_small.reshape((target_h // 2, target_w))))
            return cv2.cvtColor(yuv_small, cv2.COLOR_YUV2BGR_NV12)
        except Exception as exc:
            log.debug("scene detector NV12 sample failed: %s", exc)
            return None

    def _reset_rvm_rec_if_needed(self, batch: int, h: int, w: int) -> None:
        sig = (batch, h, w)
        if (self._rvm_rec is None
                or self._rvm_rec[0].shape[0] != batch
                or getattr(self, "_rvm_rec_sig", None) != sig):
            self._rvm_rec = [
                np.zeros((batch, 1, 1, 1), dtype=self.input_dtype)
                for _ in range(4)
            ]
            self._rvm_rec_sig = sig
        if self._rvm_io_sig != sig:
            self._rvm_rec_ort = None
            self._rvm_io_outputs = {}
            self._rvm_io_downsample = None
            self._rvm_io_sig = sig

    def _reset_rvm_rec_ort_if_needed(self, batch: int, h: int, w: int) -> None:
        self._reset_rvm_rec_if_needed(batch, h, w)
        if self._rvm_rec_ort is None:
            self._rvm_rec_ort = [
                ort.OrtValue.ortvalue_from_shape_and_type(
                    self._rvm_initial_state_shape(name, batch, h, w), self.input_dtype, "cuda", 0
                )
                for name in self.input_names[1:5]
            ]
            for state in self._rvm_rec_ort:
                zeros = np.zeros(tuple(int(v) for v in state.shape()), dtype=self.input_dtype)
                self._copy_numpy_to_cuda_ortvalue(zeros, state)
        if self._rvm_io_downsample is None:
            self._rvm_io_downsample = ort.OrtValue.ortvalue_from_numpy(
                np.asarray([RVM_DOWNSAMPLE_RATIO], dtype=self.rvm_downsample_dtype),
                "cuda",
                0,
            )

    @staticmethod
    def _copy_numpy_to_cuda_ortvalue(src: np.ndarray, dst: ort.OrtValue) -> None:
        # OrtValue.numpy() returns a CPU copy for CUDA values; use update_inplace
        # when available to initialize persistent CUDA state buffers.
        if hasattr(dst, "update_inplace"):
            dst.update_inplace(src)
            return
        raise RuntimeError("onnxruntime OrtValue.update_inplace is required for CUDA RVM state init")

    def _rvm_output_ortvalue(self, name: str, shape: tuple[int, ...]) -> ort.OrtValue:
        value = self._rvm_io_outputs.get(name)
        if value is None or tuple(value.shape()) != shape:
            value = ort.OrtValue.ortvalue_from_shape_and_type(shape, self.input_dtype, "cuda", 0)
            self._rvm_io_outputs[name] = value
        return value

    def _rvm_initial_state_shape(self, name: str, batch: int, h: int, w: int) -> tuple[int, ...]:
        rec_scale = {
            "r1": 2,
            "r2": 4,
            "r3": 8,
            "r4": 16,
        }
        scale = rec_scale.get(name.lower()[:2])
        channels = self._rvm_input_channels.get(name)
        if channels is None:
            channels = {"r1": 16, "r2": 20, "r3": 40, "r4": 64}.get(name.lower()[:2])
        if scale is None or channels is None:
            return (batch, 1, 1, 1)
        if not (self._rvm_trt_active or self._rvm_static_trt_available) and self.input_shape and self.input_shape[0] not in {1, 2}:
            return (batch, 1, 1, 1)
        return (
            batch,
            channels,
            max(1, int(round(h * RVM_DOWNSAMPLE_RATIO / scale))),
            max(1, int(round(w * RVM_DOWNSAMPLE_RATIO / scale))),
        )

    def _rvm_output_shape_for(self, name: str, batch: int, h: int, w: int) -> tuple[int, ...] | None:
        lname = name.lower()
        if "pha" in lname or "alpha" in lname:
            return (batch, 1, h, w)
        if "fgr" in lname or "foreground" in lname:
            return (batch, 3, h, w)

        rec_scale = {
            "r1": 2,
            "r2": 4,
            "r3": 8,
            "r4": 16,
        }
        prefix = lname[:2]
        scale = rec_scale.get(prefix)
        channels = self._rvm_output_channels.get(name)
        if channels is None:
            channels = {"r1": 16, "r2": 20, "r3": 40, "r4": 64}.get(prefix)
        if scale is not None and channels is not None:
            return (
                batch,
                channels,
                max(1, int(round(h * RVM_DOWNSAMPLE_RATIO / scale))),
                max(1, int(round(w * RVM_DOWNSAMPLE_RATIO / scale))),
            )
        return None

    def _ortvalue_to_cupy(self, value: ort.OrtValue):
        cp = _cp
        shape = tuple(int(v) for v in value.shape())
        dtype = cp.dtype(self.input_dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        mem = cp.cuda.UnownedMemory(int(value.data_ptr()), nbytes, value)
        ptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray(shape, dtype=dtype, memptr=ptr)

    def _copy_cuda_ortvalue(self, value: ort.OrtValue) -> ort.OrtValue:
        cp = _cp
        arr = cp.ascontiguousarray(self._ortvalue_to_cupy(value))
        host = cp.asnumpy(arr)
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(host), "cuda", 0)

    def _rvm_rec_outputs_from(self, outputs_by_name: dict, outputs: list) -> list:
        if len(outputs) >= 6:
            return list(outputs[2:6])
        return [
            outputs_by_name[k]
            for k in self.output_names
            if k.lower().startswith("r") and k.lower().endswith("o")
        ]

    def _run_rvm_iobinding_from_dev(self, x_dev):
        cp = _cp
        batch, _, h, w = (int(v) for v in x_dev.shape)
        static = self._run_rvm_static_trt_iobinding_from_dev(x_dev, batch, h, w)
        if static is not None:
            return static
        self._reset_rvm_rec_ort_if_needed(batch, h, w)
        assert self._rvm_rec_ort is not None

        binding = self.sess.io_binding()
        binding.bind_input(
            self.input_name,
            "cuda",
            0,
            self.input_dtype,
            tuple(x_dev.shape),
            int(x_dev.data.ptr),
        )
        for name, state in zip(self.input_names[1:5], self._rvm_rec_ort):
            binding.bind_ortvalue_input(name, state)
        if len(self.input_names) >= 6 and self._rvm_io_downsample is not None:
            binding.bind_ortvalue_input(self.input_names[5], self._rvm_io_downsample)

        use_ort_allocated_outputs = "TensorrtExecutionProvider" in self.sess.get_providers()
        for meta in self.output_metas:
            shape = None if use_ort_allocated_outputs else self._rvm_output_shape_for(meta.name, batch, h, w)
            if shape is None:
                binding.bind_output(meta.name, "cuda", 0)
            else:
                binding.bind_ortvalue_output(meta.name, self._rvm_output_ortvalue(meta.name, shape))

        self.sess.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        out_by_name = dict(zip(self.output_names, outputs))
        pha_name = next(
            (k for k in self.output_names if "pha" in k.lower() or "alpha" in k.lower()),
            self.output_names[1] if len(self.output_names) > 1 else self.output_names[0],
        )
        pha_ort = out_by_name.get(pha_name, outputs[1] if len(outputs) > 1 else outputs[0])

        rec_outs = self._rvm_rec_outputs_from(out_by_name, outputs)
        if len(rec_outs) >= 4:
            if use_ort_allocated_outputs:
                self._rvm_rec_ort = [self._copy_cuda_ortvalue(v) for v in rec_outs[:4]]
            else:
                self._rvm_rec_ort = rec_outs[:4]

        pha_cp = self._ortvalue_to_cupy(pha_ort)
        if pha_cp.ndim == 4:
            return self._smooth_rvm_alpha_batch(pha_cp[:, 0])
        if pha_cp.ndim == 3:
            return self._smooth_rvm_alpha_batch(pha_cp)
        squeezed = cp.squeeze(pha_cp)
        if squeezed.ndim == 2:
            return self._smooth_rvm_alpha_batch(squeezed[None, ...])
        return self._smooth_rvm_alpha_batch(squeezed)

    def _run_rvm_static_trt_iobinding_from_dev(self, x_dev, batch: int, h: int, w: int):
        session = self._get_trt_static_session(batch, h, w)
        if session is None:
            return None

        key = (batch, h, w)
        input_metas = session.get_inputs()
        output_metas = session.get_outputs()
        input_names = [meta.name for meta in input_metas]
        output_names = [meta.name for meta in output_metas]
        states = self._trt_static_states.get(key)
        if states is None:
            states = []
            for name in input_names[1:5]:
                state = ort.OrtValue.ortvalue_from_shape_and_type(
                    self._rvm_initial_state_shape(name, batch, h, w), self.input_dtype, "cuda", 0
                )
                zeros = np.zeros(tuple(int(v) for v in state.shape()), dtype=self.input_dtype)
                self._copy_numpy_to_cuda_ortvalue(zeros, state)
                states.append(state)
            self._trt_static_states[key] = states

        outputs_cache = self._trt_static_outputs.setdefault(key, {})
        binding = session.io_binding()
        binding.bind_input(
            input_names[0],
            "cuda",
            0,
            self.input_dtype,
            tuple(x_dev.shape),
            int(x_dev.data.ptr),
        )
        for name, state in zip(input_names[1:5], states):
            binding.bind_ortvalue_input(name, state)
        for meta in output_metas:
            shape = self._rvm_output_shape_for(meta.name, batch, h, w)
            if shape is None:
                binding.bind_output(meta.name, "cuda", 0)
                continue
            value = outputs_cache.get(meta.name)
            if value is None or tuple(value.shape()) != shape:
                value = ort.OrtValue.ortvalue_from_shape_and_type(shape, self.input_dtype, "cuda", 0)
                outputs_cache[meta.name] = value
            binding.bind_ortvalue_output(meta.name, value)

        session.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        out_by_name = dict(zip(output_names, outputs))
        rec_outs = self._rvm_rec_outputs_from(out_by_name, outputs)
        if len(rec_outs) >= 4:
            self._trt_static_states[key] = rec_outs[:4]

        pha_name = next(
            (name for name in output_names if "pha" in name.lower() or "alpha" in name.lower()),
            output_names[1] if len(output_names) > 1 else output_names[0],
        )
        pha_ort = out_by_name.get(pha_name, outputs[1] if len(outputs) > 1 else outputs[0])
        pha_cp = self._ortvalue_to_cupy(pha_ort)
        if pha_cp.ndim == 4:
            return self._smooth_rvm_alpha_batch(pha_cp[:, 0])
        if pha_cp.ndim == 3:
            return self._smooth_rvm_alpha_batch(pha_cp)
        squeezed = cp.squeeze(pha_cp)
        if squeezed.ndim == 2:
            return self._smooth_rvm_alpha_batch(squeezed[None, ...])
        return self._smooth_rvm_alpha_batch(squeezed)

    def _run_rvm(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != self.input_dtype:
            x = x.astype(self.input_dtype)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        batch, _, h, w = x.shape
        self._reset_rvm_rec_if_needed(int(batch), int(h), int(w))

        feed: dict[str, np.ndarray] = {self.input_name: x}
        rec_inputs = self.input_names[1:5]
        for name, rec in zip(rec_inputs, self._rvm_rec or []):
            feed[name] = rec
        if len(self.input_names) >= 6:
            feed[self.input_names[5]] = np.asarray([RVM_DOWNSAMPLE_RATIO], dtype=self.rvm_downsample_dtype)

        outputs = self.sess.run(self.output_names, feed)
        out_by_name = dict(zip(self.output_names, outputs))
        pha = next(
            (v for k, v in out_by_name.items() if "pha" in k.lower() or "alpha" in k.lower()),
            outputs[1] if len(outputs) > 1 else outputs[0],
        )
        rec_outs = self._rvm_rec_outputs_from(out_by_name, outputs)
        if len(rec_outs) >= 4:
            self._rvm_rec = [np.ascontiguousarray(r.astype(self.input_dtype, copy=False)) for r in rec_outs[:4]]
        elif len(outputs) >= 6:
            self._rvm_rec = [np.ascontiguousarray(r.astype(self.input_dtype, copy=False)) for r in outputs[-4:]]
        return self._smooth_rvm_alpha_batch(self._postprocess_alpha_batch(self._extract_alpha_batch(pha)))

    @staticmethod
    def _extract_alpha(out: np.ndarray) -> np.ndarray:
        if out.ndim == 4:
            return out[0, 0]
        if out.ndim == 3:
            return out[0]
        return np.squeeze(out)

    @staticmethod
    def _extract_alpha_batch(out: np.ndarray) -> np.ndarray:
        if out.ndim == 4:
            return out[:, 0]
        if out.ndim == 3:
            return out
        squeezed = np.squeeze(out)
        if squeezed.ndim == 2:
            return squeezed[None, ...]
        return squeezed

    def _postprocess_alpha(self, alpha: np.ndarray) -> np.ndarray:
        alpha = alpha.astype(np.float32, copy=False)
        if self.model_kind not in {"rmbg", "ben2"}:
            return alpha
        amin = float(alpha.min())
        amax = float(alpha.max())
        denom = amax - amin
        if denom <= 1e-6:
            return np.zeros_like(alpha, dtype=np.float32)
        return (alpha - amin) / denom

    def _postprocess_alpha_batch(self, alpha_batch: np.ndarray) -> np.ndarray:
        alpha_batch = alpha_batch.astype(np.float32, copy=False)
        if self.model_kind not in {"rmbg", "ben2"}:
            return alpha_batch
        out = np.empty_like(alpha_batch, dtype=np.float32)
        for i in range(alpha_batch.shape[0]):
            out[i] = self._postprocess_alpha(alpha_batch[i])
        return out

    @staticmethod
    def _adjust_alpha_for_composite(alpha: np.ndarray) -> np.ndarray:
        """Apply optional matte edge cleanup before CPU compositing."""
        if ALPHA_CUTOFF <= 0.0 and not ALPHA_HARD_EDGE and ALPHA_CONTRAST == 1.0:
            return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)
        out = alpha.astype(np.float32, copy=True)
        if ALPHA_CONTRAST != 1.0:
            out -= 0.5
            out *= ALPHA_CONTRAST
            out += 0.5
        np.clip(out, 0.0, 1.0, out=out)
        if ALPHA_CUTOFF > 0.0:
            if ALPHA_HARD_EDGE:
                out = (out >= ALPHA_CUTOFF).astype(np.float32)
            else:
                out[out < ALPHA_CUTOFF] = 0.0
        return out

    # ONNX matting alpha cv2
    # SBS (ah, 2*aw)
    # CPU frame_bgr resize/normalize/transpose
    def _alpha_low_res(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, MattingTiming, str]:
        import time as _time

        t0 = _time.perf_counter()
        split_sbs_active = MATTING_SPLIT_SBS and frame_bgr.shape[1] >= 2 * frame_bgr.shape[0]
        if split_sbs_active:
            half = frame_bgr.shape[1] // 2
            left = frame_bgr[:, :half]
            right = frame_bgr[:, half:half * 2]
            self._maybe_reset_rvm_for_scene(left)
            xL = self._resize_to_matting_input(left)
            xR = self._resize_to_matting_input(right)
            t1 = _time.perf_counter()
            if self._supports_batch2 and xL.shape == xR.shape:
                x = np.concatenate([xL, xR], axis=0)
                a_batch = self._run_matting_batch(x)
                aL_small = a_batch[0]
                aR_small = a_batch[1]
                ort_shape = str(x.shape)
            else:
                aL_small = self._run_matting_in_rvm_slot("sbs_left", xL)
                aR_small = self._run_matting_in_rvm_slot("sbs_right", xR)
                ort_shape = f"2x{xL.shape}"
            t2 = _time.perf_counter()
            a_small = np.concatenate([aL_small, aR_small], axis=1)
        else:
            self._maybe_reset_rvm_for_scene(frame_bgr)
            x = self._resize_to_matting_input(frame_bgr)
            t1 = _time.perf_counter()
            a_small = self._run_matting(x)
            t2 = _time.perf_counter()
            ort_shape = str(x.shape)

        self._call_count += 1
        self._last_ort_shape = ort_shape
        if self._call_count == 1 or self._call_count % 100 == 0:
            log.info(
                "[DIAG] alpha #%d: frame=%dx%d input_shape=%s preprocess=%.1fms ort_run=%.1fms "
                "square=%s split_sbs=%s split_active=%s providers=%s",
                self._call_count,
                frame_bgr.shape[1], frame_bgr.shape[0], ort_shape,
                (t1 - t0) * 1000, (t2 - t1) * 1000,
                MATTING_SQUARE, MATTING_SPLIT_SBS, split_sbs_active, self._rvm_provider_diag(),
            )
        return a_small, MattingTiming((t1 - t0) * 1000, (t2 - t1) * 1000, 0.0, 0.0), ort_shape

    # GPU _g_frame _upload_frame_gpu
    # preprocess GPU CHW pinned host buffer ORT
    def _alpha_low_res_gpu(self, h: int, w: int, use_nv12: bool = False) -> tuple[np.ndarray, MattingTiming, str]:
        import time as _time

        t0 = _time.perf_counter()
        preprocess_one = self._gpu_preprocess_nv12_one if use_nv12 else self._gpu_preprocess_one
        split_sbs_active = MATTING_SPLIT_SBS and w >= 2 * h
        if split_sbs_active:
            half = w // 2
            out_w, out_h = self._matting_size_for(half, h)
            if self._supports_batch2:
                use_iobinding = self._rvm_iobinding_enabled and not self._rvm_iobinding_failed
                preprocess_one(0, half, out_w, out_h, batch=2, batch_idx=0, copy_to_host=not use_iobinding)
                preprocess_one(half, half, out_w, out_h, batch=2, batch_idx=1, copy_to_host=not use_iobinding)
                x = self._g_chw if use_iobinding else self._h_chw
                t1 = _time.perf_counter()
                if use_iobinding:
                    try:
                        a_batch = self._run_rvm_iobinding_from_dev(x)
                    except Exception as e:
                        self._rvm_iobinding_failed = True
                        log.warning("[DIAG] RVM IOBinding failed; falling back to sess.run: %s", e)
                        cp = _cp
                        self._ensure_pinned_chw(out_w, out_h, 2)
                        if self._g_chw is None or tuple(self._g_chw.shape) != tuple(self._h_chw.shape):
                            raise RuntimeError(
                                f"RVM fallback buffer mismatch: gpu={getattr(self._g_chw, 'shape', None)} host={self._h_chw.shape}"
                            )
                        cp.asnumpy(self._g_chw, out=self._h_chw)
                        a_batch = self._run_matting_batch(self._h_chw)
                else:
                    a_batch = self._run_matting_batch(x)
                t2 = _time.perf_counter()
                aL_small = a_batch[0]
                aR_small = a_batch[1]
                ort_shape = f"(2,3,{out_h},{out_w})"
                pre_ms = (t1 - t0) * 1000
                ort_ms = (t2 - t1) * 1000
            else:
                use_iobinding = self._rvm_iobinding_enabled and not self._rvm_iobinding_failed
                xL = preprocess_one(0, half, out_w, out_h, copy_to_host=not use_iobinding)
                t1 = _time.perf_counter()
                if use_iobinding:
                    try:
                        aL_small = self._run_rvm_iobinding_in_slot("sbs_left", xL)[0]
                    except Exception as e:
                        self._rvm_iobinding_failed = True
                        log.warning("[DIAG] RVM IOBinding failed; falling back to sess.run: %s", e)
                        cp = _cp
                        cp.asnumpy(xL, out=self._ensure_pinned_chw(out_w, out_h, 1))
                        aL_small = self._run_matting_in_rvm_slot("sbs_left", self._h_chw)
                else:
                    aL_small = self._run_matting_in_rvm_slot("sbs_left", xL)
                t_pre_R = _time.perf_counter()
                xR = preprocess_one(half, half, out_w, out_h, copy_to_host=not use_iobinding)
                t_ort_R0 = _time.perf_counter()
                if use_iobinding and not self._rvm_iobinding_failed:
                    try:
                        aR_small = self._run_rvm_iobinding_in_slot("sbs_right", xR)[0]
                    except Exception as e:
                        self._rvm_iobinding_failed = True
                        log.warning("[DIAG] RVM IOBinding failed; falling back to sess.run: %s", e)
                        cp = _cp
                        cp.asnumpy(xR, out=self._ensure_pinned_chw(out_w, out_h, 1))
                        aR_small = self._run_matting_in_rvm_slot("sbs_right", self._h_chw)
                else:
                    aR_small = self._run_matting_in_rvm_slot("sbs_right", xR)
                t2 = _time.perf_counter()
                ort_shape = f"2x(1,3,{out_h},{out_w})"
                pre_ms = (t1 - t0) * 1000 + (t_ort_R0 - t_pre_R) * 1000
                ort_ms = (t_pre_R - t1) * 1000 + (t2 - t_ort_R0) * 1000
            if _cp is not None and hasattr(aL_small, "data") and hasattr(aL_small.data, "ptr"):
                a_small = _cp.concatenate([aL_small, aR_small], axis=1)
            else:
                a_small = np.concatenate([aL_small, aR_small], axis=1)
        else:
            out_w, out_h = self._matting_size_for(w, h)
            use_iobinding = self._rvm_iobinding_enabled and not self._rvm_iobinding_failed
            x = preprocess_one(0, w, out_w, out_h, copy_to_host=not use_iobinding)
            t1 = _time.perf_counter()
            if use_iobinding:
                try:
                    a_small = self._run_rvm_iobinding_from_dev(x)[0]
                except Exception as e:
                    self._rvm_iobinding_failed = True
                    log.warning("[DIAG] RVM IOBinding failed; falling back to sess.run: %s", e)
                    cp = _cp
                    self._ensure_pinned_chw(out_w, out_h, 1)
                    if self._g_chw is None or tuple(self._g_chw.shape) != tuple(self._h_chw.shape):
                        raise RuntimeError(
                            f"RVM fallback buffer mismatch: gpu={getattr(self._g_chw, 'shape', None)} host={self._h_chw.shape}"
                        )
                    cp.asnumpy(self._g_chw, out=self._h_chw)
                    a_small = self._run_matting(self._h_chw)
            else:
                a_small = self._run_matting(x)
            t2 = _time.perf_counter()
            ort_shape = f"(1,3,{out_h},{out_w})"
            pre_ms = (t1 - t0) * 1000
            ort_ms = (t2 - t1) * 1000

        self._call_count += 1
        self._last_ort_shape = ort_shape
        if self._call_count == 1 or self._call_count % 100 == 0:
            log.info(
                "[DIAG] alpha #%d (gpu-pre): frame=%dx%d input_shape=%s preprocess=%.1fms ort_run=%.1fms "
                "square=%s split_sbs=%s split_active=%s providers=%s",
                self._call_count, w, h, ort_shape, pre_ms, ort_ms,
                MATTING_SQUARE, MATTING_SPLIT_SBS, split_sbs_active, self._rvm_provider_diag(),
            )
        return a_small, MattingTiming(pre_ms, ort_ms, 0.0, 0.0), ort_shape

    def _alpha_low_res_gpu_temporal(self, h: int, w: int, use_nv12: bool = False) -> tuple[np.ndarray, MattingTiming, str]:
        """Reuse the latest low-res alpha for skipped frames when PT_ALPHA_STRIDE > 1."""
        cache_key = (h, w, MATTING_INPUT_SIZE, 1 if MATTING_SPLIT_SBS and w >= 2 * h else 0, 1 if use_nv12 else 0)
        if PASSTHROUGH_RVM_BYPASS_ALPHA:
            cp = _cp
            if MATTING_SPLIT_SBS and w >= 2 * h:
                half = w // 2
                out_w, out_h = self._matting_size_for(half, h)
                shape = (out_h, out_w * 2)
                ort_shape = f"(2,3,{out_h},{out_w}):bypass_all_ones"
            else:
                out_w, out_h = self._matting_size_for(w, h)
                shape = (out_h, out_w)
                ort_shape = f"(1,3,{out_h},{out_w}):bypass_all_ones"
            if self._g_bypass_alpha is None or self._g_bypass_alpha.shape != shape:
                self._g_bypass_alpha = cp.ones(shape, dtype=cp.float32)
            self._call_count += 1
            self._last_ort_shape = ort_shape
            if self._call_count == 1 or self._call_count % 100 == 0:
                log.info(
                    "[DIAG] alpha #%d bypass: frame=%dx%d alpha_shape=%s use_nv12=%s",
                    self._call_count,
                    w,
                    h,
                    shape,
                    use_nv12,
                )
            return self._g_bypass_alpha, MattingTiming(0.0, 0.0, 0.0, 0.0), ort_shape

        infer = (
            ALPHA_STRIDE <= 1
            or self._cached_alpha_small is None
            or self._cached_alpha_shape != cache_key
            or self._temporal_frame_idx % ALPHA_STRIDE == 0
        )
        self._temporal_frame_idx += 1

        if infer:
            a_small, timing, ort_shape = self._alpha_low_res_gpu(h, w, use_nv12=use_nv12)
            if _cp is not None and hasattr(a_small, "data") and hasattr(a_small.data, "ptr"):
                self._cached_alpha_small = a_small.astype(_cp.float32, copy=False)
            else:
                self._cached_alpha_small = np.ascontiguousarray(a_small.astype(np.float32, copy=False))
            self._cached_alpha_shape = cache_key
            self._cached_alpha_ort_shape = ort_shape
            return self._cached_alpha_small, timing, ort_shape

        self._last_ort_shape = f"{self._cached_alpha_ort_shape}:reuse/{ALPHA_STRIDE}"
        return (
            self._cached_alpha_small,
            MattingTiming(0.0, 0.0, 0.0, 0.0),
            self._last_ort_shape,
        )

    def alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        a, _ = self.alpha_profile(frame_bgr)
        return a

    def alpha_profile(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, MattingTiming]:
        """ API alpha bench/
 composite_green_profile fused kernel GPU """
        import time as _time

        h, w = frame_bgr.shape[:2]
        a_small, timing, _ = self._alpha_low_res(frame_bgr)
        t0 = _time.perf_counter()
        a = cv2.resize(a_small, (w, h), interpolation=cv2.INTER_LINEAR)
        a = self._adjust_alpha_for_composite(a)
        a_resize_ms = (_time.perf_counter() - t0) * 1000
        return a, MattingTiming(timing.preprocess_ms, timing.ort_ms, a_resize_ms, 0.0)

    # ---------- composite ----------
    def composite_green(self, frame_bgr: np.ndarray) -> np.ndarray:
        out, _ = self.composite_green_profile(frame_bgr)
        return out

    def composite_green_profile(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, MattingTiming]:
        """
 - GPU preprocess fused composite _g_frame
 preprocess GPU box-resample+normalize+CHW alpha pinned host
 ORT composite fused upsample+composite kernel pinned D2H
 - CPU fallback cv2 + numpy """
        import time as _time

        if _GPU_OK:
            h, w = frame_bgr.shape[:2]
            stream = _CUDA_STREAM
            ctx = stream if stream is not None else nullcontext()
            with ctx:
                t_up0 = _time.perf_counter()
                self._upload_frame_gpu(frame_bgr)
                t_up1 = _time.perf_counter()
                a_small, timing, _ = self._alpha_low_res_gpu_temporal(h, w)
                t0 = _time.perf_counter()
                out = self._composite_using_uploaded_frame(a_small, h, w)
                t1 = _time.perf_counter()
            comp_ms = (t1 - t0) * 1000 + (t_up1 - t_up0) * 1000
            return out, MattingTiming(
                timing.preprocess_ms,
                timing.ort_ms,
                0.0,  # alpha_resize
                comp_ms,
            )

        # CPU fallback cv2 + numpy
        a, timing = self.alpha_profile(frame_bgr)
        t0 = _time.perf_counter()
        out = self._composite_cpu(frame_bgr, a)
        t1 = _time.perf_counter()
        return out, MattingTiming(
            timing.preprocess_ms,
            timing.ort_ms,
            timing.alpha_resize_ms,
            (t1 - t0) * 1000,
        )

    def composite_green_nv12_profile(self, frame_nv12: np.ndarray, h: int, w: int) -> tuple[np.ndarray, MattingTiming]:
        """GPU hot path for NV12 raw frames from ffmpeg: upload NV12 once, infer alpha, composite to BGR."""
        import time as _time

        if not _GPU_OK:
            raise RuntimeError("NV12 matting path requires CuPy/GPU composite")
        stream = _CUDA_STREAM
        ctx = stream if stream is not None else nullcontext()
        scene_bgr = self._scene_bgr_from_nv12(frame_nv12, h, w)
        if scene_bgr is not None:
            self._maybe_reset_rvm_for_scene(scene_bgr)
        with ctx:
            t_up0 = _time.perf_counter()
            self._upload_nv12_gpu(frame_nv12, h, w)
            t_up1 = _time.perf_counter()
            a_small, timing, _ = self._alpha_low_res_gpu_temporal(h, w, use_nv12=True)
            t0 = _time.perf_counter()
            out = self._composite_nv12_using_uploaded_frame(a_small, h, w)
            t1 = _time.perf_counter()
        return out, MattingTiming(
            timing.preprocess_ms,
            timing.ort_ms,
            0.0,
            (t1 - t0) * 1000 + (t_up1 - t_up0) * 1000,
        )

    def composite_green_nv12_to_nv12_profile(
        self,
        frame_nv12: np.ndarray,
        h: int,
        w: int,
        out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, MattingTiming]:
        """GPU hot path for NV12 input and NV12 output, intended for direct NVENC raw input."""
        import time as _time

        if not _GPU_OK:
            raise RuntimeError("NV12 matting path requires CuPy/GPU composite")
        stream = _CUDA_STREAM
        ctx = stream if stream is not None else nullcontext()
        scene_bgr = self._scene_bgr_from_nv12(frame_nv12, h, w)
        if scene_bgr is not None:
            self._maybe_reset_rvm_for_scene(scene_bgr)
        with ctx:
            t_up0 = _time.perf_counter()
            self._upload_nv12_gpu(frame_nv12, h, w)
            t_up1 = _time.perf_counter()
            a_small, timing, _ = self._alpha_low_res_gpu_temporal(h, w, use_nv12=True)
            t0 = _time.perf_counter()
            out = self._composite_nv12_to_nv12_using_uploaded_frame(a_small, h, w, out=out)
            t1 = _time.perf_counter()
        return out, MattingTiming(
            timing.preprocess_ms,
            timing.ort_ms,
            0.0,
            (t1 - t0) * 1000 + (t_up1 - t_up0) * 1000,
        )

    # CPU fallback CuPy
    def _tmp_for(self, shape: tuple[int, int]) -> np.ndarray:
        if self._tmp_float is None or self._tmp_float.shape != shape:
            self._tmp_float = np.empty(shape, dtype=np.float32)
        return self._tmp_float

    def _composite_cpu(self, frame_bgr: np.ndarray, a: np.ndarray) -> np.ndarray:
        out = np.empty_like(frame_bgr)
        tmp = self._tmp_for(a.shape)

        b, g, r = GREEN_BGR
        np.multiply(frame_bgr[:, :, 0], a, out=tmp, casting="unsafe")
        if b:
            tmp += b * (1.0 - a)
        out[:, :, 0] = tmp

        np.multiply(frame_bgr[:, :, 1], a, out=tmp, casting="unsafe")
        if g:
            tmp += g * (1.0 - a)
        out[:, :, 1] = tmp

        np.multiply(frame_bgr[:, :, 2], a, out=tmp, casting="unsafe")
        if r:
            tmp += r * (1.0 - a)
        out[:, :, 2] = tmp
        return out

    # GPU alpha fallback /
    def _composite_gpu(self, frame_bgr: np.ndarray, alpha_f32: np.ndarray) -> np.ndarray:
        cp = _cp
        h, w = frame_bgr.shape[:2]
        total = h * w

        if (self._g_frame is None or self._g_frame.shape != frame_bgr.shape):
            self._g_frame = cp.empty(frame_bgr.shape, dtype=cp.uint8)
            self._g_out = cp.empty(frame_bgr.shape, dtype=cp.uint8)
        if (self._g_alpha is None or self._g_alpha.shape != alpha_f32.shape
                or self._g_alpha.dtype != cp.float32):
            self._g_alpha = cp.empty(alpha_f32.shape, dtype=cp.float32)

        self._g_frame.set(frame_bgr)
        self._g_alpha.set(alpha_f32)

        gB, gG, gR = GREEN_BGR
        threads = 256
        blocks = (total + threads - 1) // threads
        _composite_kernel(
            (blocks,), (threads,),
            (
                self._g_frame, self._g_alpha,
                np.uint8(gB), np.uint8(gG), np.uint8(gR),
                np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                np.int32(total),
                self._g_out,
            ),
        )
        out = cp.asnumpy(self._g_out)
        return out

    # GPU _g_frame _upload_frame_gpu
    # alpha fused kernel pinned host buffer D2H
    # numpy pinned buffer caller composite
    def _composite_using_uploaded_frame(self, a_small: np.ndarray, h: int, w: int) -> np.ndarray:
        cp = _cp
        ah, aw = a_small.shape[:2]

        if hasattr(a_small, "data") and hasattr(a_small.data, "ptr"):
            alpha_dev = a_small.astype(cp.float32, copy=False)
        else:
            if (self._g_alpha is None or self._g_alpha.shape != a_small.shape
                    or self._g_alpha.dtype != cp.float32):
                self._g_alpha = cp.empty(a_small.shape, dtype=cp.float32)
            if a_small.dtype != np.float32:
                a_small = a_small.astype(np.float32, copy=False)
            if not a_small.flags["C_CONTIGUOUS"]:
                a_small = np.ascontiguousarray(a_small)
            self._g_alpha.set(a_small)
            alpha_dev = self._g_alpha

        gB, gG, gR = GREEN_BGR
        scale_x = aw / float(w)
        scale_y = ah / float(h)
        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        _composite_upsample_kernel(
            grid, block,
            (
                self._g_frame, alpha_dev,
                np.int32(w), np.int32(h),
                np.int32(aw), np.int32(ah),
                np.float32(scale_x), np.float32(scale_y),
                np.uint8(gB), np.uint8(gG), np.uint8(gR),
                np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                self._g_out,
            ),
        )
        # Pinned D2H host buffer numpy + pageable
        out_host = self._ensure_pinned_out(h, w)
        cp.asnumpy(self._g_out, out=out_host)
        return out_host

    def _composite_nv12_using_uploaded_frame(self, a_small: np.ndarray, h: int, w: int) -> np.ndarray:
        cp = _cp
        ah, aw = a_small.shape[:2]

        if hasattr(a_small, "data") and hasattr(a_small.data, "ptr"):
            alpha_dev = a_small.astype(cp.float32, copy=False)
        else:
            if (self._g_alpha is None or self._g_alpha.shape != a_small.shape
                    or self._g_alpha.dtype != cp.float32):
                self._g_alpha = cp.empty(a_small.shape, dtype=cp.float32)
            if a_small.dtype != np.float32:
                a_small = a_small.astype(np.float32, copy=False)
            if not a_small.flags["C_CONTIGUOUS"]:
                a_small = np.ascontiguousarray(a_small)
            self._g_alpha.set(a_small)
            alpha_dev = self._g_alpha

        gB, gG, gR = GREEN_BGR
        scale_x = aw / float(w)
        scale_y = ah / float(h)
        light_coeffs, light_gamma_lut, light_identity = self.light_match_kernel_args()
        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        _composite_nv12_upsample_kernel(
            grid, block,
            (
                self._g_frame, alpha_dev,
                np.int32(w), np.int32(h),
                np.int32(aw), np.int32(ah),
                np.float32(scale_x), np.float32(scale_y),
                np.uint8(gB), np.uint8(gG), np.uint8(gR),
                np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                self._g_out,
                light_coeffs,
                light_gamma_lut,
                light_identity,
            ),
        )
        out_host = self._ensure_pinned_out(h, w)
        cp.asnumpy(self._g_out, out=out_host)
        return out_host

    def make_pinned_nv12_output_pool(self, h: int, w: int, count: int) -> list[tuple[object, np.ndarray]]:
        """Allocate reusable pinned host buffers for async encoder handoff."""
        return [self._alloc_pinned((h * 3 // 2, w), np.uint8) for _ in range(max(0, count))]

    def _composite_nv12_to_nv12_using_uploaded_frame(
        self,
        a_small: np.ndarray,
        h: int,
        w: int,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        cp = _cp
        ah, aw = a_small.shape[:2]

        if hasattr(a_small, "data") and hasattr(a_small.data, "ptr"):
            alpha_dev = a_small.astype(cp.float32, copy=False)
        else:
            if (self._g_alpha is None or self._g_alpha.shape != a_small.shape
                    or self._g_alpha.dtype != cp.float32):
                self._g_alpha = cp.empty(a_small.shape, dtype=cp.float32)
            if a_small.dtype != np.float32:
                a_small = a_small.astype(np.float32, copy=False)
            if not a_small.flags["C_CONTIGUOUS"]:
                a_small = np.ascontiguousarray(a_small)
            self._g_alpha.set(a_small)
            alpha_dev = self._g_alpha
        out_nv12 = self._ensure_dev_nv12_out(h, w)

        gY, gU, gV = self._green_yuv()
        light_coeffs, light_gamma_lut, light_identity = self.light_match_kernel_args()
        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        if SPLIT_NV12_COMPOSITE:
            _composite_nv12_y_kernel(
                grid, block,
                (
                    self._g_frame, alpha_dev,
                    np.int32(w), np.int32(h),
                    np.int32(aw), np.int32(ah),
                    np.uint8(gY),
                    np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                    out_nv12,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
            uv_grid = (((w >> 1) + block[0] - 1) // block[0], ((h >> 1) + block[1] - 1) // block[1], 1)
            _composite_nv12_uv_kernel(
                uv_grid, block,
                (
                    self._g_frame, alpha_dev,
                    np.int32(w), np.int32(h),
                    np.int32(aw), np.int32(ah),
                    np.uint8(gU), np.uint8(gV),
                    np.int32(1 if FAST_UV_ALPHA else 0),
                    np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                    out_nv12,
                    light_coeffs,
                    light_identity,
                ),
            )
        else:
            _composite_nv12_to_nv12_kernel(
                grid, block,
                (
                    self._g_frame, alpha_dev,
                    np.int32(w), np.int32(h),
                    np.int32(aw), np.int32(ah),
                    np.uint8(gY), np.uint8(gU), np.uint8(gV),
                    np.int32(1 if FAST_UV_ALPHA else 0),
                    np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                    out_nv12,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
        out_host = out if out is not None else self._ensure_pinned_nv12_out(h, w)
        cp.asnumpy(out_nv12, out=out_host)
        return out_host

    def composite_green_gpu_nv12_frame_to_gpu_nv12_profile(
        self,
        frame_gpu_nv12,
        out_h: int | None = None,
        out_w: int | None = None,
        out_slot: Nv12OutputSlot | None = None,
    ):
        """Experimental PyNv path: GPU NV12 planes -> matting -> GPU NV12 output."""
        import time as _time

        if not _GPU_OK:
            raise RuntimeError("GPU NV12 frame path requires CuPy/GPU composite")
        src_h, src_w = int(frame_gpu_nv12.height), int(frame_gpu_nv12.width)
        w, h = self.pynv_scaled_size(src_w, src_h)
        if out_w is not None and out_h is not None:
            w, h = int(out_w), int(out_h)
        stream = _CUDA_STREAM
        ctx = stream if stream is not None else nullcontext()
        diag_enabled = WARMUP_RAMPUP_DIAG_FRAMES > 0 and self._preproc_diag_count < WARMUP_RAMPUP_DIAG_FRAMES
        t_total0 = _time.perf_counter()
        with ctx:
            t_up0 = _time.perf_counter()
            self.upload_nv12_planes_gpu_scaled(frame_gpu_nv12.y.as_cupy(), frame_gpu_nv12.uv.as_cupy(), src_h, src_w, h, w)
            if (PASSTHROUGH_PYNV_SYNC_PROBE or diag_enabled) and stream is not None:
                stream.synchronize()
            t_up1 = _time.perf_counter()
            a_small, timing, _ = self._alpha_low_res_gpu_temporal(h, w, use_nv12=True)
            t_alpha_done = _time.perf_counter()
            if (PASSTHROUGH_PYNV_SYNC_PROBE or diag_enabled) and stream is not None:
                stream.synchronize()
            t_alpha_sync = _time.perf_counter()
            t0 = _time.perf_counter()
            out = self._composite_nv12_to_nv12_gpu_using_uploaded_frame(
                a_small,
                h,
                w,
                out=out_slot.buffer if out_slot is not None else None,
            )
            if (PASSTHROUGH_PYNV_SYNC_PROBE or diag_enabled) and stream is not None:
                stream.synchronize()
            t1 = _time.perf_counter()
        if diag_enabled:
            self._preproc_diag_count += 1
            log.info(
                "[DIAG][PREPROC] kind=nv12 frame=%d src=%dx%d out=%dx%d upload=%.2fms "
                "alpha_call=%.2fms alpha_tail_sync=%.2fms composite=%.2fms total=%.2fms "
                "mat_pre=%.2fms mat_ort=%.2fms mat_kernel=%.2fms",
                self._preproc_diag_count,
                src_w,
                src_h,
                w,
                h,
                (t_up1 - t_up0) * 1000.0,
                (t_alpha_done - t_up1) * 1000.0,
                (t_alpha_sync - t_alpha_done) * 1000.0,
                (t1 - t0) * 1000.0,
                (t1 - t_total0) * 1000.0,
                timing.preprocess_ms,
                timing.ort_ms,
                timing.composite_ms,
            )
        if PASSTHROUGH_PYNV_SYNC_PROBE:
            self._sync_probe_count += 1
            if self._sync_probe_count == 1 or self._sync_probe_count % 30 == 0:
                log.info(
                    "[DIAG] pynv sync probe nv12 frame=%d upload_sync=%.2fms alpha_call=%.2fms "
                    "alpha_tail_sync=%.2fms composite_sync=%.2fms mat_timing pre=%.2fms ort=%.2fms kernel=%.2fms",
                    self._sync_probe_count,
                    (t_up1 - t_up0) * 1000.0,
                    (t_alpha_done - t_up1) * 1000.0,
                    (t_alpha_sync - t_alpha_done) * 1000.0,
                    (t1 - t0) * 1000.0,
                    timing.preprocess_ms,
                    timing.ort_ms,
                    timing.composite_ms,
                )
        return out, MattingTiming(
            timing.preprocess_ms,
            timing.ort_ms + ((t_alpha_sync - t_alpha_done) * 1000.0 if PASSTHROUGH_PYNV_SYNC_PROBE else 0.0),
            0.0,
            (t1 - t0) * 1000 + (t_up1 - t_up0) * 1000,
        )

    def composite_green_gpu_p016_frame_to_gpu_nv12_profile(
        self,
        frame_gpu_p016,
        shift_bits: int = 8,
        out_h: int | None = None,
        out_w: int | None = None,
        out_slot: Nv12OutputSlot | None = None,
    ):
        """Experimental PyNv path: GPU P016/P010 planes -> NV12 -> matting -> GPU NV12 output."""
        import time as _time

        if not _GPU_OK:
            raise RuntimeError("GPU P016 frame path requires CuPy/GPU composite")
        src_h, src_w = int(frame_gpu_p016.height), int(frame_gpu_p016.width)
        w, h = self.pynv_scaled_size(src_w, src_h)
        if out_w is not None and out_h is not None:
            w, h = int(out_w), int(out_h)
        stream = _CUDA_STREAM
        ctx = stream if stream is not None else nullcontext()
        diag_enabled = WARMUP_RAMPUP_DIAG_FRAMES > 0 and self._preproc_diag_count < WARMUP_RAMPUP_DIAG_FRAMES
        t_total0 = _time.perf_counter()
        with ctx:
            t_up0 = _time.perf_counter()
            self.upload_p016_planes_as_nv12_gpu_scaled(
                frame_gpu_p016.y.as_cupy(),
                frame_gpu_p016.uv.as_cupy(),
                src_h,
                src_w,
                h,
                w,
                shift_bits=shift_bits,
            )
            if (PASSTHROUGH_PYNV_SYNC_PROBE or diag_enabled) and stream is not None:
                stream.synchronize()
            t_up1 = _time.perf_counter()
            a_small, timing, _ = self._alpha_low_res_gpu_temporal(h, w, use_nv12=True)
            t_alpha_done = _time.perf_counter()
            if (PASSTHROUGH_PYNV_SYNC_PROBE or diag_enabled) and stream is not None:
                stream.synchronize()
            t_alpha_sync = _time.perf_counter()
            t0 = _time.perf_counter()
            out = self._composite_nv12_to_nv12_gpu_using_uploaded_frame(
                a_small,
                h,
                w,
                out=out_slot.buffer if out_slot is not None else None,
            )
            if (PASSTHROUGH_PYNV_SYNC_PROBE or diag_enabled) and stream is not None:
                stream.synchronize()
            t1 = _time.perf_counter()
        if diag_enabled:
            self._preproc_diag_count += 1
            log.info(
                "[DIAG][PREPROC] kind=p016 frame=%d src=%dx%d out=%dx%d upload=%.2fms "
                "alpha_call=%.2fms alpha_tail_sync=%.2fms composite=%.2fms total=%.2fms "
                "mat_pre=%.2fms mat_ort=%.2fms mat_kernel=%.2fms",
                self._preproc_diag_count,
                src_w,
                src_h,
                w,
                h,
                (t_up1 - t_up0) * 1000.0,
                (t_alpha_done - t_up1) * 1000.0,
                (t_alpha_sync - t_alpha_done) * 1000.0,
                (t1 - t0) * 1000.0,
                (t1 - t_total0) * 1000.0,
                timing.preprocess_ms,
                timing.ort_ms,
                timing.composite_ms,
            )
        if PASSTHROUGH_PYNV_SYNC_PROBE:
            self._sync_probe_count += 1
            if self._sync_probe_count == 1 or self._sync_probe_count % 30 == 0:
                log.info(
                    "[DIAG] pynv sync probe p016 frame=%d upload_sync=%.2fms alpha_call=%.2fms "
                    "alpha_tail_sync=%.2fms composite_sync=%.2fms mat_timing pre=%.2fms ort=%.2fms kernel=%.2fms",
                    self._sync_probe_count,
                    (t_up1 - t_up0) * 1000.0,
                    (t_alpha_done - t_up1) * 1000.0,
                    (t_alpha_sync - t_alpha_done) * 1000.0,
                    (t1 - t0) * 1000.0,
                    timing.preprocess_ms,
                    timing.ort_ms,
                    timing.composite_ms,
                )
        return out, MattingTiming(
            timing.preprocess_ms,
            timing.ort_ms + ((t_alpha_sync - t_alpha_done) * 1000.0 if PASSTHROUGH_PYNV_SYNC_PROBE else 0.0),
            0.0,
            (t1 - t0) * 1000 + (t_up1 - t_up0) * 1000,
        )

    def _composite_nv12_to_nv12_gpu_using_uploaded_frame(
        self,
        a_small: np.ndarray,
        h: int,
        w: int,
        out=None,
    ):
        import time as _time

        cp = _cp
        ah, aw = a_small.shape[:2]
        debug = bool(DEBUG_LOGS)

        if hasattr(a_small, "data") and hasattr(a_small.data, "ptr"):
            alpha_dev = a_small.astype(cp.float32, copy=False)
        else:
            if (self._g_alpha is None or self._g_alpha.shape != a_small.shape
                    or self._g_alpha.dtype != cp.float32):
                self._g_alpha = cp.empty(a_small.shape, dtype=cp.float32)
            if a_small.dtype != np.float32:
                a_small = a_small.astype(np.float32, copy=False)
            if not a_small.flags["C_CONTIGUOUS"]:
                a_small = np.ascontiguousarray(a_small)
            self._g_alpha.set(a_small)
            alpha_dev = self._g_alpha
        out_nv12 = out if out is not None else self._ensure_dev_nv12_out(h, w)
        gY, gU, gV = self._green_yuv()
        light_coeffs, light_gamma_lut, light_identity = self.light_match_kernel_args()
        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        if SPLIT_NV12_COMPOSITE:
            t_kernel = _time.perf_counter()
            _composite_nv12_y_kernel(
                grid, block,
                (
                    self._g_frame, alpha_dev,
                    np.int32(w), np.int32(h),
                    np.int32(aw), np.int32(ah),
                    np.uint8(gY),
                    np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                    out_nv12,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
            # Per-frame kernel timing logs are too noisy for realtime playback.
            # Re-enable locally only for focused CUDA kernel profiling.
            # if debug:
            #     log.info("[DIAG] nv12->nv12 y kernel returned in %.3fms", (_time.perf_counter() - t_kernel) * 1000.0)
            uv_grid = (((w >> 1) + block[0] - 1) // block[0], ((h >> 1) + block[1] - 1) // block[1], 1)
            t_kernel = _time.perf_counter()
            _composite_nv12_uv_kernel(
                uv_grid, block,
                (
                    self._g_frame, alpha_dev,
                    np.int32(w), np.int32(h),
                    np.int32(aw), np.int32(ah),
                    np.uint8(gU), np.uint8(gV),
                    np.int32(1 if FAST_UV_ALPHA else 0),
                    np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                    out_nv12,
                    light_coeffs,
                    light_identity,
                ),
            )
            # if debug:
            #     log.info("[DIAG] nv12->nv12 uv kernel returned in %.3fms", (_time.perf_counter() - t_kernel) * 1000.0)
        else:
            t_kernel = _time.perf_counter()
            _composite_nv12_to_nv12_kernel(
                grid, block,
                (
                    self._g_frame, alpha_dev,
                    np.int32(w), np.int32(h),
                    np.int32(aw), np.int32(ah),
                    np.uint8(gY), np.uint8(gU), np.uint8(gV),
                    np.int32(1 if FAST_UV_ALPHA else 0),
                    np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                    out_nv12,
                    light_coeffs,
                    light_gamma_lut,
                    light_identity,
                ),
            )
            # if debug:
            #     log.info("[DIAG] nv12->nv12 mono kernel returned in %.3fms", (_time.perf_counter() - t_kernel) * 1000.0)
        return out_nv12

    @staticmethod
    def _green_yuv() -> tuple[int, int, int]:
        b, g, r = GREEN_BGR
        y = 16.0 + 0.257 * r + 0.504 * g + 0.098 * b
        u = 128.0 - 0.148 * r - 0.291 * g + 0.439 * b
        v = 128.0 + 0.439 * r - 0.368 * g - 0.071 * b
        return (
            int(np.clip(round(y), 0, 255)),
            int(np.clip(round(u), 0, 255)),
            int(np.clip(round(v), 0, 255)),
        )

    # GPU alpha bilinear fused kernel
    # frame_bgr: (H, W, 3) uint8;  a_small: (ah, aw) float32
    def _composite_gpu_fused(self, frame_bgr: np.ndarray, a_small: np.ndarray) -> np.ndarray:
        cp = _cp
        h, w = frame_bgr.shape[:2]
        ah, aw = a_small.shape[:2]

        if (self._g_frame is None or self._g_frame.shape != frame_bgr.shape):
            self._g_frame = cp.empty(frame_bgr.shape, dtype=cp.uint8)
            self._g_out = cp.empty(frame_bgr.shape, dtype=cp.uint8)
        if (self._g_alpha is None or self._g_alpha.shape != a_small.shape
                or self._g_alpha.dtype != cp.float32):
            self._g_alpha = cp.empty(a_small.shape, dtype=cp.float32)

        self._g_frame.set(frame_bgr)
        if a_small.dtype != np.float32:
            a_small = a_small.astype(np.float32, copy=False)
        if not a_small.flags["C_CONTIGUOUS"]:
            a_small = np.ascontiguousarray(a_small)
        self._g_alpha.set(a_small)

        gB, gG, gR = GREEN_BGR
        scale_x = aw / float(w)
        scale_y = ah / float(h)

        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        _composite_upsample_kernel(
            grid, block,
            (
                self._g_frame, self._g_alpha,
                np.int32(w), np.int32(h),
                np.int32(aw), np.int32(ah),
                np.float32(scale_x), np.float32(scale_y),
                np.uint8(gB), np.uint8(gG), np.uint8(gR),
                np.float32(ALPHA_CUTOFF), np.int32(1 if ALPHA_HARD_EDGE else 0), np.float32(ALPHA_CONTRAST),
                self._g_out,
            ),
        )
        out = cp.asnumpy(self._g_out)
        return out


_pool_lock = threading.Lock()
_pool_cond = threading.Condition(_pool_lock)
_pool_all: list[Matter] = []
_pool_available: list[Matter] = []
_pool_max: int = 1
_pool_warmup_runs: int | None = None


def configure_matter_pool(max_concurrent: int, *, warmup_runs: int | None = None) -> None:
    """Set the maximum number of Matter instances the pool may hand out.

    Instances are lazily created on first acquire. Safe to call repeatedly;
    shrinking the cap below the current pool size does not destroy existing
    instances but prevents new ones from being created.
    """
    global _pool_max, _pool_warmup_runs
    with _pool_lock:
        _pool_max = max(1, int(max_concurrent))
        if warmup_runs is not None:
            _pool_warmup_runs = warmup_runs


def _create_matter_locked(*, mark_available: bool) -> Matter:
    instance = Matter(warmup_runs=_pool_warmup_runs)
    _pool_all.append(instance)
    if mark_available:
        _pool_available.append(instance)
    return instance


def acquire_matter(*, blocking: bool = True, timeout: float | None = None) -> Matter | None:
    """Acquire a Matter instance from the pool, creating one lazily if room.

    Blocks until an instance is available unless ``blocking=False``. Returns
    ``None`` on timeout or non-blocking failure. Each acquirer must call
    :func:`release_matter` exactly once when done.
    """
    import time as _time

    deadline = _time.monotonic() + timeout if timeout is not None else None
    with _pool_cond:
        while True:
            if _pool_available:
                return _pool_available.pop()
            if len(_pool_all) < _pool_max:
                return _create_matter_locked(mark_available=False)
            if not blocking:
                return None
            if deadline is not None:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    return None
                _pool_cond.wait(timeout=remaining)
            else:
                _pool_cond.wait()


def release_matter(instance: Matter | None) -> None:
    """Return a Matter instance to the pool."""
    if instance is None:
        return
    with _pool_cond:
        if instance not in _pool_all or instance in _pool_available:
            return
        _pool_available.append(instance)
        _pool_cond.notify()


def get_matter(*, warmup_runs: int | None = None) -> Matter:
    """Return the shared utility Matter instance (slot 0).

    Used by startup warmup, offline tools, and benchmarks. The instance is
    created on first call and stays available in the pool, so realtime
    passthrough callers can later acquire it via :func:`acquire_matter`.
    Utility callers must not run concurrently with realtime passthrough on the
    same instance — by convention warmup completes before serving begins.
    """
    global _pool_warmup_runs
    with _pool_cond:
        if warmup_runs is not None:
            _pool_warmup_runs = warmup_runs
        if not _pool_all:
            return _create_matter_locked(mark_available=True)
        return _pool_all[0]
