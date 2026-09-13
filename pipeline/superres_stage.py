"""GPU-resident RTX VSR stage shared by the live and seekable passthrough paths.

Both paths decode NV12 on the GPU, hand each frame to NGX VSR and encode the
result with NVENC. Keeping the setup and the per-frame work here means the two
callers cannot drift apart on eye splitting, buffer reuse or the SDR look.

Output surfaces come from a ring: NVENC may still be reading the previous input
after Encode() returns, so a frame must not overwrite the one before it.
"""
from __future__ import annotations

import logging
from typing import Any

import config

log = logging.getLogger(__name__)


def realtime_hdr_look(logger=None, label: str = "") -> str:
    """The SDR look a realtime path may apply.

    TrueHDR needs the offline P010/HDR10 chain, so realtime keeps the SDR grade
    instead of silently producing something the 8-bit NV12 output cannot carry.
    """
    from pipeline.hdr_look import normalize_hdr_look
    from pipeline.true_hdr import is_true_hdr

    mode = normalize_hdr_look(config.RTX_VSR_HDR_LOOK)
    if is_true_hdr(mode):
        (logger or log).warning(
            "%sTrueHDR is offline-only; realtime SuperRes keeps the SDR look",
            f"{label} " if label else "",
        )
        return "off"
    return mode


class SuperResStage:
    """One RTX VSR session: NV12 in, NV12 out, both on the GPU."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        source_stem: str = "",
        is_10bit: bool = False,
        target_height: int | None = None,
        quality: int | None = None,
        hdr_look: str | None = None,
        ring_slots: int | None = None,
        logger=None,
        label: str = "",
    ) -> None:
        from utils.rtx_vsr import is_native_target, source_block_reason, target_dimensions
        from utils.vr_naming import has_vr_filename_marker, is_half_equirectangular_source

        self._log = logger or log
        self._label = f"{label} " if label else ""
        self.width, self.height = int(width), int(height)
        self.quality = config.RTX_VSR_QUALITY if quality is None else int(quality)
        is_sbs_vr = is_half_equirectangular_source(self.width, self.height)
        reason = source_block_reason(
            self.width,
            self.height,
            is_vr=has_vr_filename_marker(source_stem) or is_sbs_vr,
            is_10bit=bool(is_10bit),
            target_height=target_height,
            allow_vr=True,
        )
        if reason:
            raise RuntimeError(f"RTX VSR source rejected: {reason}")
        self.out_w, self.out_h = target_dimensions(self.width, self.height, target_height)
        # Each eye is evaluated on its own, at 1x as well as when enlarging.
        self.split_eyes = bool(
            is_sbs_vr
            and (is_native_target(target_height) or int(target_height or config.RTX_VSR_TARGET_HEIGHT) >= 2160)
            and self.out_w == self.out_h * 2
        )
        if self.split_eyes and self.width % 2:
            raise RuntimeError(f"split-eye RTX VSR requires even-width SBS input: {self.width}x{self.height}")

        import cupy as cp

        from offline.two_dvr_pynv import _NV12_RGB_KERNELS
        from pipeline.hdr_look import HDR_LOOK_CUDA
        from utils.rtx_vsr import load_bridge

        self._cp = cp
        self._bridge = load_bridge()
        if not self._bridge.initialize_cupy(cp):
            raise RuntimeError("RTX VSR feature unavailable")
        module = cp.RawModule(code=_NV12_RGB_KERNELS + HDR_LOOK_CUDA)
        self._k_to_rgb = module.get_function("nv12_to_rgb")
        self._k_to_nv12 = module.get_function("rgba_to_nv12")
        self.hdr_look = realtime_hdr_look(self._log, label) if hdr_look is None else str(hdr_look)
        self._k_hdr = module.get_function("hdr_look_rgba") if self.hdr_look != "off" else None

        self._rgb = cp.empty((self.height, self.width, 3), cp.uint8)
        self.eye_in_w = self.width // 2 if self.split_eyes else 0
        self.eye_out_w = self.out_w // 2 if self.split_eyes else 0
        if self.split_eyes:
            self._left_eye = cp.empty((self.height, self.eye_in_w, 4), cp.uint8)
            self._right_eye = cp.empty((self.height, self.eye_in_w, 4), cp.uint8)
            self._left_out = cp.empty((self.out_h, self.eye_out_w, 4), cp.uint8)
            self._right_out = cp.empty((self.out_h, self.eye_out_w, 4), cp.uint8)
            self._split_output = cp.empty((self.out_h, self.out_w, 4), cp.uint8)
            self._rgba = None
            self._whole_out = None
        else:
            self._left_eye = self._right_eye = self._left_out = self._right_out = None
            self._split_output = None
            self._rgba = cp.empty((self.height, self.width, 4), cp.uint8)
            self._whole_out = cp.empty((self.out_h, self.out_w, 4), cp.uint8)
        slots = int(config.PASSTHROUGH_NV12_RING_SLOTS if ring_slots is None else ring_slots)
        self._ring = [
            cp.empty((self.out_h * 3 // 2, self.out_w), cp.uint8)
            for _ in range(2 if self.out_w >= 8192 else max(1, slots))
        ]
        self._block = (16, 16, 1)
        self._grid = ((self.width + 15) // 16, (self.height + 15) // 16, 1)
        self._grid_out = ((self.out_w + 15) // 16, (self.out_h + 15) // 16, 1)
        self._log.info(
            "%sRTX VSR active source=%dx%d output=%dx%d quality=%d hdr_look=%s split_eyes=%s",
            self._label, self.width, self.height, self.out_w, self.out_h,
            self.quality, self.hdr_look,
            f"{self.eye_out_w}x{self.out_h}+{self.eye_out_w}x{self.out_h}" if self.split_eyes else "off",
        )

    @property
    def output_size(self) -> tuple[int, int]:
        return self.out_w, self.out_h

    def process(self, frame: Any, index: int) -> Any:
        """Enhance one decoded NV12 frame; returns a GPU NV12 surface."""
        from pipeline.pynv_io import GpuP016Frame

        if isinstance(frame, GpuP016Frame):
            raise RuntimeError("RTX VSR realtime currently requires 8-bit NV12 input")
        cp = self._cp
        import numpy as np

        y = frame.y.as_cupy(cp.uint8).reshape(self.height, self.width)
        uv = frame.uv.as_cupy(cp.uint8).reshape(self.height // 2, self.width)
        self._k_to_rgb(
            self._grid, self._block,
            (y, uv, self._rgb, np.int32(self.width), np.int32(self.height)),
        )
        if self.split_eyes:
            self._left_eye[:, :, :3] = self._rgb[:, : self.eye_in_w]
            self._left_eye[:, :, 3] = 255
            self._right_eye[:, :, :3] = self._rgb[:, self.eye_in_w :]
            self._right_eye[:, :, 3] = 255
            self._split_output[:, : self.eye_out_w] = self._bridge.process_cupy_rgba(
                self._left_eye, (self.eye_out_w, self.out_h), self.quality, out=self._left_out,
            )
            self._split_output[:, self.eye_out_w :] = self._bridge.process_cupy_rgba(
                self._right_eye, (self.eye_out_w, self.out_h), self.quality, out=self._right_out,
            )
            enhanced = self._split_output
        else:
            self._rgba[:, :, :3] = self._rgb
            self._rgba[:, :, 3] = 255
            enhanced = self._bridge.process_cupy_rgba(
                self._rgba, (self.out_w, self.out_h), self.quality, out=self._whole_out,
            )
        if self._k_hdr is not None:
            from pipeline.hdr_look import apply_hdr_look

            apply_hdr_look(self._k_hdr, enhanced, self.hdr_look)
        nv12 = self._ring[int(index) % len(self._ring)]
        self._k_to_nv12(
            self._grid_out, self._block,
            (enhanced, nv12, np.int32(self.out_w), np.int32(self.out_h)),
        )
        return nv12
