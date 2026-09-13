"""NGX TrueHDR (SDR -> HDR10) output support shared by realtime and offline VSR.

The RTX Video SDK writes TrueHDR results as packed 10:10:10:2 ("ABGR10"):
one ``uint32`` per pixel with red in the low bits.  Those samples are PQ
encoded BT.2020 RGB, so the only work left here is converting them to the
limited-range BT.2020 non-constant-luminance P010 surface NVENC expects for
10-bit HEVC, and describing that colorimetry to the muxer.
"""
from __future__ import annotations

from utils.video_metadata import VideoColorMetadata


TRUE_HDR_MODE = "truehdr"


def is_true_hdr(mode: object) -> bool:
    return str(mode or "").strip().lower() == TRUE_HDR_MODE


def true_hdr_color_metadata() -> VideoColorMetadata:
    """Colorimetry of the TrueHDR output, independent of the SDR source."""
    return VideoColorMetadata(
        color_range="tv",
        color_space="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
    )


TRUE_HDR_CUDA = r'''
// One thread per 2x2 luma block: four Y samples and one chroma pair.
extern "C" __global__ void abgr10_to_p010(
    const unsigned int* __restrict__ src,
    unsigned short* __restrict__ dst,
    int W, int H)
{
    int bx = blockIdx.x*blockDim.x + threadIdx.x;
    int by = blockIdx.y*blockDim.y + threadIdx.y;
    int x0 = bx << 1;
    int y0 = by << 1;
    if (x0 >= W || y0 >= H) return;

    float cb_sum = 0.0f;
    float cr_sum = 0.0f;
    int samples = 0;
    for (int dy = 0; dy < 2; ++dy) {
        int y = y0 + dy;
        if (y >= H) continue;
        for (int dx = 0; dx < 2; ++dx) {
            int x = x0 + dx;
            if (x >= W) continue;
            unsigned int v = src[(long)y*W + x];
            float r = (float)( v        & 1023u) * (1.0f/1023.0f);
            float g = (float)((v >> 10) & 1023u) * (1.0f/1023.0f);
            float b = (float)((v >> 20) & 1023u) * (1.0f/1023.0f);
            // BT.2020 non-constant luminance.
            float luma = 0.2627f*r + 0.6780f*g + 0.0593f*b;
            cb_sum += (b - luma) * (1.0f/1.8814f);
            cr_sum += (r - luma) * (1.0f/1.4746f);
            ++samples;
            // Limited range 10-bit luma, stored in the high bits for P010.
            float yv = 64.0f + 876.0f * luma;
            unsigned short y10 = (unsigned short)fminf(fmaxf(yv + 0.5f, 0.0f), 1023.0f);
            dst[(long)y*W + x] = (unsigned short)(y10 << 6);
        }
    }

    float inv = samples > 0 ? 1.0f / (float)samples : 0.0f;
    float cb = 512.0f + 896.0f * (cb_sum * inv);
    float cr = 512.0f + 896.0f * (cr_sum * inv);
    unsigned short cb10 = (unsigned short)fminf(fmaxf(cb + 0.5f, 0.0f), 1023.0f);
    unsigned short cr10 = (unsigned short)fminf(fmaxf(cr + 0.5f, 0.0f), 1023.0f);
    long uv = (long)(H + by)*W + x0;
    dst[uv] = (unsigned short)(cb10 << 6);
    if (x0 + 1 < W) dst[uv + 1] = (unsigned short)(cr10 << 6);
}
'''


def p010_grid(out_w: int, out_h: int, block: tuple[int, int, int] = (16, 16, 1)) -> tuple[int, int, int]:
    """Grid covering one thread per 2x2 block of the output frame."""
    chroma_w = (int(out_w) + 1) // 2
    chroma_h = (int(out_h) + 1) // 2
    return (
        (chroma_w + block[0] - 1) // block[0],
        (chroma_h + block[1] - 1) // block[1],
        1,
    )
