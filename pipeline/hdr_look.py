"""Shared CUDA SDR "HDR look" post-processing for realtime and offline VSR."""
from __future__ import annotations


HDR_LOOK_MODES = ("off", "natural", "vivid")


def normalize_hdr_look(value: object, default: str = "natural") -> str:
    mode = str(value or default).strip().lower()
    return mode if mode in HDR_LOOK_MODES else default


def hdr_look_mode_value(value: object) -> int:
    return {"off": 0, "natural": 1, "vivid": 2}[normalize_hdr_look(value)]


HDR_LOOK_CUDA = r'''
extern "C" __global__ void hdr_look_rgba(unsigned char* rgba, int pixels, int mode) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= pixels || mode <= 0) return;
    int p = i * 4;
    float r = rgba[p] / 255.0f;
    float g = rgba[p + 1] / 255.0f;
    float b = rgba[p + 2] / 255.0f;
    float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    float contrast = mode == 1 ? 1.08f : 1.16f;
    float shadow = mode == 1 ? 0.012f : 0.020f;
    float highlight = mode == 1 ? 0.040f : 0.080f;
    float saturation = mode == 1 ? 1.08f : 1.18f;

    float mapped = 0.5f + (y - 0.5f) * contrast;
    mapped += shadow * (1.0f - y) * (1.0f - y);
    mapped -= highlight * y * y;
    mapped = fminf(1.0f, fmaxf(0.0f, mapped));

    float scale = y > 0.0001f ? mapped / y : 0.0f;
    r *= scale;
    g *= scale;
    b *= scale;
    float scaled_y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    r = mapped + (r - scaled_y) * saturation;
    g = mapped + (g - scaled_y) * saturation;
    b = mapped + (b - scaled_y) * saturation;

    rgba[p] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, r)) * 255.0f + 0.5f);
    rgba[p + 1] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, g)) * 255.0f + 0.5f);
    rgba[p + 2] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, b)) * 255.0f + 0.5f);
}
'''


def create_hdr_look_kernel(cp):
    return cp.RawKernel(HDR_LOOK_CUDA, "hdr_look_rgba")


def apply_hdr_look(kernel, rgba, mode: object) -> None:
    mode_value = hdr_look_mode_value(mode)
    if mode_value <= 0:
        return
    pixels = int(rgba.shape[0]) * int(rgba.shape[1])
    threads = 256
    kernel(((pixels + threads - 1) // threads,), (threads,), (rgba, pixels, mode_value))
