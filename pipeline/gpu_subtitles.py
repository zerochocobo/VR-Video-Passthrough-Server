"""GPU subtitle blending shared by live and virtual-file playback."""
from __future__ import annotations
import config
from pipeline.subtitles import SubtitleRenderer

_SUBTITLE_BLEND_Y_KERNEL = None
_SUBTITLE_BLEND_UV_KERNEL = None


class GpuSubtitleOverlay:
    def _subtitle_kernels(self):
        global _SUBTITLE_BLEND_Y_KERNEL, _SUBTITLE_BLEND_UV_KERNEL
        if _SUBTITLE_BLEND_Y_KERNEL is not None and _SUBTITLE_BLEND_UV_KERNEL is not None:
            return _SUBTITLE_BLEND_Y_KERNEL, _SUBTITLE_BLEND_UV_KERNEL
        import cupy as cp

        _SUBTITLE_BLEND_Y_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void blend_rgba_y_to_nv12(
                unsigned char* frame,
                const unsigned char* rgba,
                int frame_w,
                int frame_h,
                int overlay_w,
                int overlay_h,
                int dst_x,
                int dst_y)
            {
                int x = blockDim.x * blockIdx.x + threadIdx.x;
                int y = blockDim.y * blockIdx.y + threadIdx.y;
                if (x >= overlay_w || y >= overlay_h) return;
                int fx = dst_x + x;
                int fy = dst_y + y;
                if (fx < 0 || fy < 0 || fx >= frame_w || fy >= frame_h) return;
                int oi = (y * overlay_w + x) * 4;
                float a = rgba[oi + 3] / 255.0f;
                if (a <= 0.0f) return;
                float r = rgba[oi + 0];
                float g = rgba[oi + 1];
                float b = rgba[oi + 2];
                float yy = 16.0f + (65.738f * r + 129.057f * g + 25.064f * b) / 256.0f;
                int yi = fy * frame_w + fx;
                frame[yi] = (unsigned char)(frame[yi] * (1.0f - a) + yy * a + 0.5f);
            }
            """,
            "blend_rgba_y_to_nv12",
        )
        _SUBTITLE_BLEND_UV_KERNEL = cp.RawKernel(
            r"""
            extern "C" __global__
            void blend_rgba_uv_to_nv12(
                unsigned char* frame,
                const unsigned char* rgba,
                int frame_w,
                int frame_h,
                int overlay_w,
                int overlay_h,
                int dst_x,
                int dst_y)
            {
                int ux = blockDim.x * blockIdx.x + threadIdx.x;
                int uy = blockDim.y * blockIdx.y + threadIdx.y;
                int uv_w = (overlay_w + 1) / 2;
                int uv_h = (overlay_h + 1) / 2;
                if (ux >= uv_w || uy >= uv_h) return;
                int ox0 = ux * 2;
                int oy0 = uy * 2;
                int fx0 = dst_x + ox0;
                int fy0 = dst_y + oy0;
                int uv_fx = fx0 & ~1;
                int uv_fy = fy0 & ~1;
                if (uv_fx < 0 || uv_fy < 0 || uv_fx + 1 >= frame_w || uv_fy + 1 >= frame_h) return;

                float a_sum = 0.0f;
                float r_sum = 0.0f;
                float g_sum = 0.0f;
                float b_sum = 0.0f;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int ox = ox0 + dx;
                        int oy = oy0 + dy;
                        int fx = dst_x + ox;
                        int fy = dst_y + oy;
                        if (ox >= overlay_w || oy >= overlay_h || fx < 0 || fy < 0 || fx >= frame_w || fy >= frame_h) continue;
                        int oi = (oy * overlay_w + ox) * 4;
                        float a = rgba[oi + 3] / 255.0f;
                        if (a <= 0.0f) continue;
                        a_sum += a;
                        r_sum += rgba[oi + 0] * a;
                        g_sum += rgba[oi + 1] * a;
                        b_sum += rgba[oi + 2] * a;
                    }
                }
                if (a_sum <= 0.0f) return;
                float a = fminf(1.0f, a_sum / 4.0f);
                float r = r_sum / a_sum;
                float g = g_sum / a_sum;
                float b = b_sum / a_sum;
                float uu = 128.0f + (-37.945f * r - 74.494f * g + 112.439f * b) / 256.0f;
                float vv = 128.0f + (112.439f * r - 94.154f * g - 18.285f * b) / 256.0f;
                int uv_i = frame_w * frame_h + (uv_fy / 2) * frame_w + uv_fx;
                frame[uv_i] = (unsigned char)(frame[uv_i] * (1.0f - a) + uu * a + 0.5f);
                frame[uv_i + 1] = (unsigned char)(frame[uv_i + 1] * (1.0f - a) + vv * a + 0.5f);
            }
            """,
            "blend_rgba_uv_to_nv12",
        )
        return _SUBTITLE_BLEND_Y_KERNEL, _SUBTITLE_BLEND_UV_KERNEL

    def _subtitle_overlay_for_time(self, renderer: SubtitleRenderer | None, pts_sec: float):
        if renderer is None or not renderer.enabled:
            return None
        return renderer.overlay_for_time(pts_sec)

    def _subtitle_overlay_positions(self, out_nv12, renderer: SubtitleRenderer, overlay):
        rgba, left, top = overlay
        if rgba.size <= 0:
            return []
        w = int(out_nv12.shape[1])
        stereo = getattr(renderer, "stereo", w >= 3000)
        eye_w = w // 2 if stereo else w
        mode = getattr(renderer, "config", config).SUBTITLE_MODE
        if mode == "auto":
            mode = "dual" if stereo else "mono"
        if mode == "left":
            positions = [(left, top)]
        elif mode == "right":
            positions = [(eye_w + left, top)]
        elif mode == "dual":
            parallax = renderer.parallax_px()
            positions = [(left, top), (eye_w + left + parallax, top)]
        else:
            positions = [(max(0, (w - int(rgba.shape[1])) // 2), top)]
        return [(rgba, x, y) for x, y in positions]

    def _blend_subtitle_overlay(self, out_nv12, renderer: SubtitleRenderer, overlay) -> None:
        if overlay is None:
            return
        import cupy as cp

        rgba, left, top = overlay
        if rgba.size <= 0:
            return
        rgba_dev = cp.asarray(rgba)
        y_kernel, uv_kernel = self._subtitle_kernels()
        h, w = int(out_nv12.shape[0] * 2 // 3), int(out_nv12.shape[1])
        positions = [(x, y) for _rgba, x, y in self._subtitle_overlay_positions(out_nv12, renderer, overlay)]
        block = (16, 16)
        grid_y = ((int(rgba.shape[1]) + block[0] - 1) // block[0], (int(rgba.shape[0]) + block[1] - 1) // block[1])
        grid_uv = (((int(rgba.shape[1]) + 1) // 2 + block[0] - 1) // block[0], ((int(rgba.shape[0]) + 1) // 2 + block[1] - 1) // block[1])
        for x, y in positions:
            y_kernel(
                grid_y,
                block,
                (
                    out_nv12,
                    rgba_dev,
                    w,
                    h,
                    int(rgba.shape[1]),
                    int(rgba.shape[0]),
                    int(x),
                    int(y),
                ),
            )
            uv_kernel(
                grid_uv,
                block,
                (
                    out_nv12,
                    rgba_dev,
                    w,
                    h,
                    int(rgba.shape[1]),
                    int(rgba.shape[0]),
                    int(x),
                    int(y),
                ),
            )
        cp.cuda.get_current_stream().synchronize()

    def _blend_positioned_subtitle_overlays(self, out_nv12, positioned_overlays) -> None:
        if not positioned_overlays:
            return
        import cupy as cp

        y_kernel, uv_kernel = self._subtitle_kernels()
        h, w = int(out_nv12.shape[0] * 2 // 3), int(out_nv12.shape[1])
        block = (16, 16)
        for rgba, x, y in positioned_overlays:
            if rgba.size <= 0:
                continue
            rgba_dev = cp.asarray(rgba)
            overlay_h, overlay_w = int(rgba.shape[0]), int(rgba.shape[1])
            grid_y = ((overlay_w + block[0] - 1) // block[0], (overlay_h + block[1] - 1) // block[1])
            grid_uv = (((overlay_w + 1) // 2 + block[0] - 1) // block[0], ((overlay_h + 1) // 2 + block[1] - 1) // block[1])
            y_kernel(grid_y, block, (out_nv12, rgba_dev, w, h, overlay_w, overlay_h, int(x), int(y)))
            uv_kernel(grid_uv, block, (out_nv12, rgba_dev, w, h, overlay_w, overlay_h, int(x), int(y)))
        cp.cuda.get_current_stream().synchronize()

    def _apply_subtitle_overlay(self, out_nv12, renderer: SubtitleRenderer | None, pts_sec: float) -> None:
        overlay = self._subtitle_overlay_for_time(renderer, pts_sec)
        if renderer is None or overlay is None:
            return
        self._blend_subtitle_overlay(out_nv12, renderer, overlay)
