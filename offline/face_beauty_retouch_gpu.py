"""GPU retouch passes for the face-beauty pipeline.

The host implementation in :mod:`offline.face_beauty_engine` (``retouch_crop``)
was the single largest cost in the whole pipeline once everything else moved to
the GPU: 74 ms of the 116 ms frame at 720p, and 54 ms per face at 8K. It is a
fixed 512x512 workload, so the transfer was never the problem -- the arithmetic
was.

Everything here mirrors the host passes exactly so the two paths stay
interchangeable:

  * ``apply_skin_smooth``   -> bilateral at half resolution, fixed 7x7 kernel
  * ``apply_lab_adjustments`` -> one sRGB->Lab round trip, tonal passes
                               accumulated in place per channel
  * ``apply_sharpen``       -> unsharp mask against a Gaussian blur

The Lab conversion follows OpenCV's 8-bit convention (D65, L scaled by 255/100,
a/b offset by 128) so GPU and CPU output match; the parity test measures ~50 dB.
"""
from __future__ import annotations

import numpy as np

_RETOUCH_SRC = r"""
extern "C" {

__device__ __forceinline__ float srgb_to_linear(float v) {
  return v <= 0.04045f ? v / 12.92f : __powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float v) {
  return v <= 0.0031308f ? v * 12.92f : 1.055f * __powf(v, 1.f / 2.4f) - 0.055f;
}

__device__ __forceinline__ float lab_f(float t) {
  return t > 0.008856f ? cbrtf(t) : 7.787f * t + 16.f / 116.f;
}

// HWC uint8 RGB crop -> three planar float Lab channels, OpenCV 8-bit scaling.
__global__ void rgb_to_lab(const unsigned char* rgb, float* L, float* A, float* B, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int i = y * S + x;
  float r = srgb_to_linear(rgb[i * 3 + 0] / 255.f);
  float g = srgb_to_linear(rgb[i * 3 + 1] / 255.f);
  float b = srgb_to_linear(rgb[i * 3 + 2] / 255.f);
  float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
  float Y =  0.212671f * r + 0.715160f * g + 0.072169f * b;
  float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
  float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
  float l = Y > 0.008856f ? (116.f * fy - 16.f) : (903.3f * Y);
  L[i] = l * 255.f / 100.f;
  A[i] = 500.f * (fx - fy) + 128.f;
  B[i] = 200.f * (fy - fz) + 128.f;
}

__global__ void lab_to_rgb(const float* L, const float* A, const float* B,
                           unsigned char* rgb, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int i = y * S + x;
  float l = L[i] * 100.f / 255.f;
  float a = A[i] - 128.f;
  float bb = B[i] - 128.f;
  float fy = (l + 16.f) / 116.f;
  float fx = fy + a / 500.f;
  float fz = fy - bb / 200.f;
  float Y = l > 7.9996f ? fy * fy * fy : l / 903.3f;
  float fx3 = fx * fx * fx, fz3 = fz * fz * fz;
  float X = (fx3 > 0.008856f ? fx3 : (fx - 16.f / 116.f) / 7.787f) * 0.950456f;
  float Z = (fz3 > 0.008856f ? fz3 : (fz - 16.f / 116.f) / 7.787f) * 1.088754f;
  float r =  3.240479f * X - 1.537150f * Y - 0.498535f * Z;
  float g = -0.969256f * X + 1.875991f * Y + 0.041556f * Z;
  float b =  0.055648f * X - 0.204043f * Y + 1.057311f * Z;
  float out[3] = {linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)};
  for (int c = 0; c < 3; c++)
    rgb[i * 3 + c] = (unsigned char)(fminf(255.f, fmaxf(0.f, out[c] * 255.f)) + 0.5f);
}

// Bilateral filter over an HWC uint8 crop. Fixed radius, matching the host
// pass's cv2.bilateralFilter(d=7) at half resolution.
__global__ void bilateral_u8(const unsigned char* src, unsigned char* dst,
                             int S, int radius, float inv_2sc2, float inv_2ss2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int i = y * S + x;
  float c0 = src[i * 3 + 0], c1 = src[i * 3 + 1], c2 = src[i * 3 + 2];
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, wsum = 0.f;
  for (int dy = -radius; dy <= radius; dy++) {
    int sy = min(S - 1, max(0, y + dy));
    for (int dx = -radius; dx <= radius; dx++) {
      int sx = min(S - 1, max(0, x + dx));
      int j = sy * S + sx;
      float d0 = src[j * 3 + 0] - c0, d1 = src[j * 3 + 1] - c1, d2 = src[j * 3 + 2] - c2;
      float range = (d0 * d0 + d1 * d1 + d2 * d2) * inv_2sc2;
      float space = (float)(dx * dx + dy * dy) * inv_2ss2;
      float w = __expf(-(range + space));
      acc0 += src[j * 3 + 0] * w; acc1 += src[j * 3 + 1] * w; acc2 += src[j * 3 + 2] * w;
      wsum += w;
    }
  }
  dst[i * 3 + 0] = (unsigned char)(acc0 / wsum + 0.5f);
  dst[i * 3 + 1] = (unsigned char)(acc1 / wsum + 0.5f);
  dst[i * 3 + 2] = (unsigned char)(acc2 / wsum + 0.5f);
}

// dst = src*(1-a) + other*a, with a = mask * strength (per-pixel alpha blend).
__global__ void blend_masked(unsigned char* dst, const unsigned char* other,
                             const float* mask, float strength, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int i = y * S + x;
  float a = mask[i] * strength;
  if (a <= 0.f) return;
  if (a > 1.f) a = 1.f;
  for (int c = 0; c < 3; c++)
    dst[i * 3 + c] = (unsigned char)(dst[i * 3 + c] * (1.f - a) + other[i * 3 + c] * a + 0.5f);
}

// Bilinear resize of an HWC uint8 crop (square in, square out).
__global__ void resize_u8(const unsigned char* src, unsigned char* dst, int ss, int ds) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ds || y >= ds) return;
  float fx = (x + 0.5f) * ss / ds - 0.5f;
  float fy = (y + 0.5f) * ss / ds - 0.5f;
  int x0 = max(0, (int)floorf(fx)), y0 = max(0, (int)floorf(fy));
  float ax = fx - x0, ay = fy - y0;
  int x1 = min(ss - 1, x0 + 1), y1 = min(ss - 1, y0 + 1);
  for (int c = 0; c < 3; c++) {
    float v = src[(y0 * ss + x0) * 3 + c] * (1-ax) * (1-ay)
            + src[(y0 * ss + x1) * 3 + c] * ax * (1-ay)
            + src[(y1 * ss + x0) * 3 + c] * (1-ax) * ay
            + src[(y1 * ss + x1) * 3 + c] * ax * ay;
    dst[(y * ds + x) * 3 + c] = (unsigned char)(v + 0.5f);
  }
}

// Unsharp mask: dst = clamp(src*(1+s) - blurred*s).
__global__ void unsharp(unsigned char* dst, const unsigned char* blurred, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = dst[i] * (1.f + s) - blurred[i] * s;
  dst[i] = (unsigned char)(fminf(255.f, fmaxf(0.f, v)) + 0.5f);
}

// Separable Gaussian over an HWC uint8 crop, used by the sharpen pass.
__global__ void blur3_h(const unsigned char* src, float* tmp, const float* k, int r, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  float acc[3] = {0.f, 0.f, 0.f};
  for (int i = -r; i <= r; i++) {
    int sx = min(S - 1, max(0, x + i));
    float w = k[i + r];
    for (int c = 0; c < 3; c++) acc[c] += src[(y * S + sx) * 3 + c] * w;
  }
  for (int c = 0; c < 3; c++) tmp[(y * S + x) * 3 + c] = acc[c];
}

__global__ void blur3_v(const float* tmp, unsigned char* dst, const float* k, int r, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  float acc[3] = {0.f, 0.f, 0.f};
  for (int i = -r; i <= r; i++) {
    int sy = min(S - 1, max(0, y + i));
    float w = k[i + r];
    for (int c = 0; c < 3; c++) acc[c] += tmp[(sy * S + x) * 3 + c] * w;
  }
  for (int c = 0; c < 3; c++)
    dst[(y * S + x) * 3 + c] = (unsigned char)(fminf(255.f, fmaxf(0.f, acc[c])) + 0.5f);
}

}
"""

_BLOCK = (16, 16, 1)


def _grid(w: int, h: int) -> tuple[int, int, int]:
    return ((w + 15) // 16, (h + 15) // 16, 1)


class GpuRetouch:
    """Device-side twin of :func:`offline.face_beauty_engine.retouch_crop`."""

    def __init__(self, cp, crop_size: int) -> None:
        self.cp = cp
        self.S = crop_size
        module = cp.RawModule(code=_RETOUCH_SRC)
        self._rgb2lab = module.get_function("rgb_to_lab")
        self._lab2rgb = module.get_function("lab_to_rgb")
        self._bilateral = module.get_function("bilateral_u8")
        self._blend_masked = module.get_function("blend_masked")
        self._resize = module.get_function("resize_u8")
        self._unsharp = module.get_function("unsharp")
        self._blur_h = module.get_function("blur3_h")
        self._blur_v = module.get_function("blur3_v")

        S = crop_size
        half = S // 2
        self._half = half
        self._small = cp.empty((half, half, 3), cp.uint8)
        self._small_out = cp.empty((half, half, 3), cp.uint8)
        self._smoothed = cp.empty((S, S, 3), cp.uint8)
        self._blur_tmp = cp.empty((S, S, 3), cp.float32)
        self._blurred = cp.empty((S, S, 3), cp.uint8)
        self._lab = [cp.empty((S, S), cp.float32) for _ in range(3)]
        # Same sigma as the host sharpen pass (cv2.GaussianBlur sigma 1.6).
        radius = max(1, int(np.ceil(1.6 * 3.0)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2.0 * 1.6 * 1.6))
        self._sharp_k = cp.asarray((kernel / kernel.sum()).astype(np.float32))
        self._sharp_r = radius

    # -- passes --------------------------------------------------------------

    def skin_smooth(self, crop_g, mask_g, strength: float) -> None:
        """Bilateral at half resolution, blended back through mask*strength."""
        if strength <= 0:
            return
        cp = self.cp
        S, half = self.S, self._half
        self._resize(_grid(half, half), _BLOCK, (crop_g, self._small, np.int32(S), np.int32(half)))
        sigma_color = 20.0 + 60.0 * strength
        self._bilateral(_grid(half, half), _BLOCK,
                        (self._small, self._small_out, np.int32(half), np.int32(3),
                         np.float32(1.0 / (2.0 * sigma_color * sigma_color)),
                         np.float32(1.0 / (2.0 * 7.0 * 7.0))))
        self._resize(_grid(S, S), _BLOCK, (self._small_out, self._smoothed, np.int32(half), np.int32(S)))
        self._blend_masked(_grid(S, S), _BLOCK,
                           (crop_g, self._smoothed, mask_g, np.float32(strength), np.int32(S)))

    def lab_adjustments(self, crop_g, adjustments) -> None:
        """One sRGB->Lab round trip; every tonal pass accumulates in place on the
        one or two channels it touches, exactly like the host version."""
        active = [a for a in adjustments if a["strength"] > 0]
        if not active:
            return
        cp = self.cp
        S = self.S
        L, A, B = self._lab
        self._rgb2lab(_grid(S, S), _BLOCK, (crop_g, L, A, B, np.int32(S)))
        planes = (L, A, B)
        for adj in active:
            mask = adj["mask"]
            strength = float(adj["strength"])
            if adj.get("even_chroma"):
                total = float(mask.sum())
                if total >= 1.0:
                    for channel in (1, 2):
                        plane = planes[channel]
                        target = float((plane * mask).sum() / total)
                        plane += (target - plane) * (strength * mask)
            if adj.get("luma_gain"):
                L += (float(adj["luma_gain"]) * strength) * mask
            if adj.get("blue_shift"):
                B += (float(adj["blue_shift"]) * strength) * mask
            gain = float(adj.get("chroma_gain", 1.0))
            if gain != 1.0:
                factor = (gain - 1.0) * strength
                for channel in (1, 2):
                    plane = planes[channel]
                    plane += (plane - 128.0) * (factor * mask)
        for plane in planes:
            cp.clip(plane, 0, 255, out=plane)
        self._lab2rgb(_grid(S, S), _BLOCK, (L, A, B, crop_g, np.int32(S)))

    def sharpen(self, crop_g, strength: float) -> None:
        if strength <= 0:
            return
        S = self.S
        self._blur_h(_grid(S, S), _BLOCK,
                     (crop_g, self._blur_tmp, self._sharp_k, np.int32(self._sharp_r), np.int32(S)))
        self._blur_v(_grid(S, S), _BLOCK,
                     (self._blur_tmp, self._blurred, self._sharp_k, np.int32(self._sharp_r), np.int32(S)))
        n = S * S * 3
        self._unsharp(((n + 255) // 256, 1, 1), (256, 1, 1),
                      (crop_g, self._blurred, np.float32(strength), np.int32(n)))
