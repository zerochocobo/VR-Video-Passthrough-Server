"""GPU-resident DLSS 5 Neural Rendering stage.

Converts a decoded NV12 surface to RGB float32 [0.0, 1.0] on the GPU, evaluates
DLSS5 NR on it and writes NV12 back, so the realtime paths never round-trip a
frame through the host.

Neural Rendering is a 1x filter: the output is exactly the source size. That is
what lets it ride the seek channel unchanged, where an enlarging stage needs the
MP4 shell and the frame budget resized to match.
"""
from __future__ import annotations

import logging
from typing import Any

import config
from models.dlss5.neural_bridge import BRIDGE, RenderParametersV6
from utils.dlss5 import DLSS5Settings

log = logging.getLogger(__name__)

# CUDA kernel for NV12 <-> Float32 RGB conversion
_DLSS5_CUDA_KERNELS = r"""
extern "C" __global__
void nv12_to_rgb_f32(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    float* __restrict__ out_rgb,
    int width,
    int height,
    int y_stride,
    int uv_stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // BT.709 limited range, the same matrix offline/two_dvr_pynv.py's shared
    // _NV12_RGB_KERNELS uses for every other stage. It was BT.601 here (the
    // 298.082/408.583/516.412 integer set) under a comment claiming 709, so a
    // 709 source - which is every HD and 4K title - was decoded on the wrong
    // matrix, enhanced in those wrong colours and written back out on 709.
    // Skin is where that shows first.
    float c = (float)y_plane[y * y_stride + x] - 16.0f;
    int uv_idx = (y >> 1) * uv_stride + ((x >> 1) << 1);
    float du = (float)uv_plane[uv_idx]     - 128.0f;
    float dv = (float)uv_plane[uv_idx + 1] - 128.0f;

    float r = 1.16438356f * c + 1.79274107f * dv;
    float g = 1.16438356f * c - 0.21324861f * du - 0.53290933f * dv;
    float b = 1.16438356f * c + 2.11240179f * du;

    r = fminf(fmaxf(r, 0.0f), 255.0f) / 255.0f;
    g = fminf(fmaxf(g, 0.0f), 255.0f) / 255.0f;
    b = fminf(fmaxf(b, 0.0f), 255.0f) / 255.0f;

    int out_idx = (y * width + x) * 3;
    out_rgb[out_idx]     = r;
    out_rgb[out_idx + 1] = g;
    out_rgb[out_idx + 2] = b;
}

extern "C" __global__
void rgb_f32_to_nv12(
    const float* __restrict__ in_rgb,
    unsigned char* __restrict__ y_plane,
    unsigned char* __restrict__ uv_plane,
    int width,
    int height,
    int y_stride,
    int uv_stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int in_idx = (y * width + x) * 3;
    float r = fminf(fmaxf(in_rgb[in_idx],     0.0f), 1.0f) * 255.0f;
    float g = fminf(fmaxf(in_rgb[in_idx + 1], 0.0f), 1.0f) * 255.0f;
    float b = fminf(fmaxf(in_rgb[in_idx + 2], 0.0f), 1.0f) * 255.0f;

    // BT.709 limited range, matching the inverse above and the shared kernels.
    float Yv = 16.0f + 0.182586f * r + 0.614231f * g + 0.062007f * b;
    y_plane[y * y_stride + x] = (unsigned char)fminf(fmaxf(roundf(Yv), 0.0f), 255.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        // Average the 2x2 block for chroma, as the shared kernels do. Sampling
        // only this corner pixel is point subsampling: it aliases every colour
        // edge it lands on, and NR has just sharpened those edges.
        float cr = 0.0f, cg = 0.0f, cb = 0.0f;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int xx = min(x + dx, width - 1);
                int yy = min(y + dy, height - 1);
                int j = (yy * width + xx) * 3;
                cr += fminf(fmaxf(in_rgb[j],     0.0f), 1.0f);
                cg += fminf(fmaxf(in_rgb[j + 1], 0.0f), 1.0f);
                cb += fminf(fmaxf(in_rgb[j + 2], 0.0f), 1.0f);
            }
        }
        cr *= 255.0f / 4.0f;
        cg *= 255.0f / 4.0f;
        cb *= 255.0f / 4.0f;

        float U = 128.0f - 0.100644f * cr - 0.338572f * cg + 0.439216f * cb;
        float V = 128.0f + 0.439216f * cr - 0.398942f * cg - 0.040274f * cb;
        int uv_idx = (y >> 1) * uv_stride + x;
        uv_plane[uv_idx]     = (unsigned char)fminf(fmaxf(roundf(U), 0.0f), 255.0f);
        uv_plane[uv_idx + 1] = (unsigned char)fminf(fmaxf(roundf(V), 0.0f), 255.0f);
    }
}
"""


class DLSS5Stage:
    """One DLSS 5 Neural Rendering session: NV12 in, NV12 out on GPU."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        settings: DLSS5Settings | None = None,
        is_10bit: bool = False,
        fps: float = 0.0,
        ring_slots: int | None = None,
        realtime: bool = False,
        logger=None,
        label: str = "",
    ) -> None:
        from utils.dlss5 import prepare_cuda_context_for_dlss5, source_block_reason_dlss5

        self._log = logger or log
        self._label = f"{label} " if label else ""
        self.width = int(width)
        self.height = int(height)
        self.settings = settings or DLSS5Settings.from_config()
        # The NR engine reaches the picture through CUDA interop and refuses to
        # set that up unless the primary context carries the blocking-sync flag.
        # The flag can only be claimed while the context is inactive, so this is
        # a no-op here on a healthy process and a clear failure on one that let
        # something else create the context first.
        prepare_cuda_context_for_dlss5()
        # The same function the route and the DLNA listing ask, so a source
        # that reaches here and should not is a mismatch with one answer, not
        # three. Without it a 10-bit source got all the way to the first frame
        # before anything said no.
        reason = source_block_reason_dlss5(
            self.width, self.height, is_10bit=bool(is_10bit), fps=float(fps or 0.0),
            realtime=realtime,
        )
        if reason:
            raise RuntimeError(f"DLSS5 source rejected: {reason}")

        import cupy as cp

        self._cp = cp
        BRIDGE.initialize(0)

        module = cp.RawModule(code=_DLSS5_CUDA_KERNELS)
        self._k_to_rgb = module.get_function("nv12_to_rgb_f32")
        self._k_to_nv12 = module.get_function("rgb_f32_to_nv12")

        # Buffers
        self._rgb_in = cp.empty((self.height, self.width, 3), cp.float32)
        self._rgb_out = cp.empty((self.height, self.width, 3), cp.float32)
        # Output ring, as in SuperResStage: NVENC may still be reading the
        # previous input after Encode() returns, so one frame must not land on
        # the surface the encoder is still holding.
        slots = int(config.PASSTHROUGH_NV12_RING_SLOTS if ring_slots is None else ring_slots)
        self._ring = [
            cp.empty((self.height * 3 // 2, self.width), cp.uint8)
            for _ in range(2 if self.width >= 8192 else max(1, slots))
        ]
        self._seen_first_frame = False
        self._log.info(
            "%sDLSS5 NR active source=%dx%d style=%d intensity=%.2f nr_passes=%d shimmer=%.2f ring=%d",
            self._label, self.width, self.height, self.settings.style, self.settings.intensity,
            self.settings.nr_passes, self.settings.shimmer_suppression, len(self._ring),
        )

    @property
    def output_size(self) -> tuple[int, int]:
        """Neural Rendering never resizes, so this is always the source size."""
        return self.width, self.height

    def process(self, frame: Any, index: int) -> Any:
        """Enhance one decoded NV12 frame; returns a GPU NV12 surface.

        Same shape as SuperResStage.process so the live worker and the seek
        frame generator can hold either behind one variable.
        """
        from pipeline.pynv_io import GpuP016Frame

        if isinstance(frame, GpuP016Frame):
            raise RuntimeError("DLSS5 realtime requires 8-bit NV12 input")
        cp = self._cp
        y = frame.y.as_cupy(cp.uint8).reshape(self.height, self.width)
        uv = frame.uv.as_cupy(cp.uint8).reshape(self.height // 2, self.width)
        nv12 = self._ring[int(index) % len(self._ring)]
        # reset tells NR the temporal history no longer describes this picture.
        # It is true exactly once per stage: a stage is built per seek/live
        # session, so its first frame is always a cut.
        reset = not self._seen_first_frame
        self._seen_first_frame = True
        self.process_nv12(
            int(y.data.ptr),
            int(uv.data.ptr),
            int(y.strides[0]),
            int(uv.strides[0]),
            int(nv12.data.ptr),
            int(nv12.data.ptr) + self.height * int(nv12.strides[0]),
            int(nv12.strides[0]),
            int(nv12.strides[0]),
            reset=reset,
        )
        return nv12

    def process_nv12(
        self,
        y_ptr: int,
        uv_ptr: int,
        y_stride: int,
        uv_stride: int,
        out_y_ptr: int,
        out_uv_ptr: int,
        out_y_stride: int,
        out_uv_stride: int,
        reset: bool = False,
    ) -> None:
        cp = self._cp
        stream = cp.cuda.get_current_stream()
        grid = ((self.width + 15) // 16, (self.height + 15) // 16)
        block = (16, 16)

        # 1. Convert NV12 -> RGB Float32
        self._k_to_rgb(
            grid,
            block,
            (
                cp.uint64(y_ptr),
                cp.uint64(uv_ptr),
                self._rgb_in.data.ptr,
                cp.int32(self.width),
                cp.int32(self.height),
                cp.int32(y_stride),
                cp.int32(uv_stride),
            ),
            stream=stream,
        )

        # 2. Evaluate DLSS 5 NR
        #
        # NR runs on D3D12 and only reaches these buffers through CUDA interop,
        # so it carries no ordering against this stream in either direction:
        # the conversion above must have landed before NR reads its input, and
        # NR must have finished before the conversion below reads its output.
        # RTX VSR brackets its own evaluate with exactly this pair. Without it
        # the encoder picked up half-written surfaces - grey and cyan blocks, a
        # bright line along the top edge, and a picture that flickered frame to
        # frame - which is what the realtime channel showed while offline (which
        # synchronises after every frame) looked correct.
        params = self.settings.to_params(reset=reset)
        stream.synchronize()
        BRIDGE.process_cuda_pointers(
            int(self._rgb_in.data.ptr),
            int(self._rgb_out.data.ptr),
            self.width,
            self.height,
            params,
            stream_ptr=int(stream.ptr),
        )
        stream.synchronize()

        # 3. Convert RGB Float32 -> NV12
        self._k_to_nv12(
            grid,
            block,
            (
                self._rgb_out.data.ptr,
                cp.uint64(out_y_ptr),
                cp.uint64(out_uv_ptr),
                cp.int32(self.width),
                cp.int32(self.height),
                cp.int32(out_y_stride),
                cp.int32(out_uv_stride),
            ),
            stream=stream,
        )
