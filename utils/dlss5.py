"""DLSS 5 Neural Rendering utility and parameter helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from models.dlss5.neural_bridge import BRIDGE, RenderParametersV6

log = logging.getLogger(__name__)


# Each control's usable span and the value that means "leave this alone". Most
# neutrals are 1.0, and Skin Structure Strength's is -1.0, so a zeroed block is
# not a neutral one. The UI reads these, so a slider cannot offer a range the
# runtime rejects.
DLSS5_RANGES: dict[str, tuple[float, float]] = {
    "intensity": (0.0, 2.0),
    "local_tone": (0.0, 2.0),
    "local_structure": (0.0, 2.0),
    "skin_structure": (-1.0, 2.0),
    "color_strength": (0.0, 1.0),
    "tone_preservation": (0.0, 1.0),
    "face_skin_protection": (0.0, 1.0),
    "grain_preservation": (0.0, 1.0),
    "shimmer_suppression": (0.0, 1.0),
}
DLSS5_NR_PASSES_RANGE = (1, 4)
DLSS5_STYLES = (0, 1, 2)  # default, natural, cinematic


@dataclass(slots=True)
class DLSS5Settings:
    style: int = 0
    intensity: float = 1.0
    local_tone: float = 1.0
    local_structure: float = 1.0
    skin_structure: float = -1.0
    auto_mask: bool = False
    color_strength: float = 1.0
    tone_preservation: float = 0.0
    face_skin_protection: float = 0.0
    grain_preservation: float = 0.0
    nr_passes: int = 1
    shimmer_suppression: float = 0.70
    prefer_nvof: bool = False

    @classmethod
    def from_config(cls) -> "DLSS5Settings":
        return cls(
            style=getattr(config, "DLSS5_STYLE", 0),
            intensity=getattr(config, "DLSS5_INTENSITY", 1.0),
            local_tone=getattr(config, "DLSS5_LOCAL_TONE", 1.0),
            local_structure=getattr(config, "DLSS5_LOCAL_STRUCTURE", 1.0),
            skin_structure=getattr(config, "DLSS5_SKIN_STRUCTURE", -1.0),
            auto_mask=getattr(config, "DLSS5_AUTO_MASK", False),
            color_strength=getattr(config, "DLSS5_COLOR_STRENGTH", 1.0),
            tone_preservation=getattr(config, "DLSS5_TONE_PRESERVATION", 0.0),
            face_skin_protection=getattr(config, "DLSS5_FACE_SKIN_PROTECTION", 0.0),
            grain_preservation=getattr(config, "DLSS5_GRAIN_PRESERVATION", 0.0),
            nr_passes=getattr(config, "DLSS5_NR_PASSES", 1),
            shimmer_suppression=getattr(config, "DLSS5_SHIMMER_SUPPRESSION", 0.70),
            prefer_nvof=getattr(config, "DLSS5_PREFER_NVOF", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "intensity": self.intensity,
            "local_tone": self.local_tone,
            "local_structure": self.local_structure,
            "skin_structure": self.skin_structure,
            "auto_mask": self.auto_mask,
            "color_strength": self.color_strength,
            "tone_preservation": self.tone_preservation,
            "face_skin_protection": self.face_skin_protection,
            "grain_preservation": self.grain_preservation,
            "nr_passes": self.nr_passes,
            "shimmer_suppression": self.shimmer_suppression,
            "prefer_nvof": self.prefer_nvof,
        }

    def to_params(self, reset: bool = False) -> RenderParametersV6:
        return RenderParametersV6.from_settings(self.to_dict(), reset=reset)


def is_dlss5_available() -> bool:
    """Return whether the DLSS 5 runtime DLLs are present."""
    return BRIDGE.is_available


# The NR engine reaches the picture through CUDA-D3D12 interop, and it refuses
# to set that up unless the device's primary context carries the blocking-sync
# scheduling flag FFmpeg's CUDA hwcontext uses ("active CUDA primary context
# does not use FFmpeg blocking-sync flags"). The engine tries to set the flag
# itself, but cuDevicePrimaryCtxSetFlags answers CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
# (708) once anything has activated the context - and in the server process
# cupy, PyNvVideoCodec and TensorRT all get there first. Offline only worked
# because it constructs the stage before its first cupy allocation.
#
# So the flag is claimed at process start, while the primary context is still
# inactive. That costs one cuInit and touches no DLSS5 DLL, so a build without
# the runtime pays nothing for it.
_CU_CTX_SCHED_BLOCKING_SYNC = 0x04
_cuda_context_prepared: bool | None = None


def prepare_cuda_context_for_dlss5(device: int = 0) -> bool:
    """Set the blocking-sync flag on the primary context; True when it holds.

    Call this BEFORE any CUDA context exists in the process. It is idempotent
    and never raises: a machine without a driver simply reports False and the
    DLSS5 channel stays out of the listing.
    """
    global _cuda_context_prepared
    if _cuda_context_prepared is not None:
        return _cuda_context_prepared
    _cuda_context_prepared = False
    try:
        import ctypes

        cuda = ctypes.WinDLL("nvcuda.dll")
        if cuda.cuInit(0) != 0:
            return False
        flags = ctypes.c_uint(0)
        active = ctypes.c_int(0)
        if cuda.cuDevicePrimaryCtxGetState(int(device), ctypes.byref(flags), ctypes.byref(active)) != 0:
            return False
        if not (flags.value & _CU_CTX_SCHED_BLOCKING_SYNC):
            status = cuda.cuDevicePrimaryCtxSetFlags(int(device), _CU_CTX_SCHED_BLOCKING_SYNC)
            if status != 0:
                log.warning(
                    "DLSS5 could not claim blocking-sync on the primary context "
                    "(cuDevicePrimaryCtxSetFlags=%d, already active=%d); "
                    "the CUDA zero-copy path will refuse to run",
                    status, active.value,
                )
                return False
        _cuda_context_prepared = True
        return True
    except Exception as exc:  # pragma: no cover - driver-specific
        log.warning("DLSS5 CUDA context preparation failed: %s: %s", type(exc).__name__, exc)
        return False


def cuda_context_ready() -> bool:
    """Whether prepare_cuda_context_for_dlss5 has succeeded in this process."""
    return bool(_cuda_context_prepared)


def warmup_dlss5_runtime(width: int = 256, height: int = 144) -> dict[str, Any]:
    """Load the NGX runtime and push one frame through it, at startup.

    The first Neural Rendering frame in a process pays for the D3D12 device,
    the NGX snippet load and the first feature build. On a cold driver cache
    that is tens of seconds, and the realtime channel used to pay it on the
    viewer's first play - where a player that is already buffering simply looks
    hung. Doing it here puts the wait inside the startup bar instead.

    A small host frame is enough: what is slow is the runtime coming up, not
    the pixels. Never raises - a failure here only means the first play pays
    for it again, so it is reported and the server carries on.
    """
    import time

    started = time.time()
    info: dict[str, Any] = {"ok": False}
    try:
        import numpy as np

        BRIDGE.initialize(0)
        info["gpu"] = BRIDGE.gpu_name
        info["runtime"] = BRIDGE.version
        frame = np.full((int(height), int(width), 3), 0.5, dtype=np.float32)
        out = np.empty_like(frame)
        BRIDGE.process_host(frame, out, DLSS5Settings.from_config().to_params(reset=True))
        info["ok"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    info["seconds"] = round(time.time() - started, 2)
    return info


def source_block_reason_dlss5(
    width: int,
    height: int,
    *,
    is_10bit: bool = False,
    fps: float = 0.0,
    realtime: bool = False,
) -> str | None:
    """Validate source dimensions for DLSS5 NR.

    ``realtime`` adds the limits that only the seek/live channel has: it pulls
    at playback speed and shares the GPU with the decoder and NVENC, so it stops
    where offline happily keeps going.
    """
    if width <= 0 or height <= 0:
        return "invalid_dimensions"
    if width % 2 != 0 or height % 2 != 0:
        return "odd_dimensions"
    if not is_dlss5_available():
        return "dlss5_runtime_missing"
    if not realtime:
        return None
    if not config.DLSS5_REALTIME_ENABLED:
        return "dlss5_realtime_disabled"
    # The kernel writes BT.709 limited-range 8-bit NV12 and the decode is forced
    # to nv12, so a 10-bit source would be silently truncated. Offline has the
    # same limit; here it is refused rather than shown and played wrong.
    if is_10bit:
        return "dlss5_10bit_unsupported"
    if height < config.DLSS5_INPUT_MIN_HEIGHT:
        return "dlss5_source_too_small"
    if height > config.DLSS5_INPUT_MAX_HEIGHT or width > config.DLSS5_INPUT_MAX_WIDTH:
        return "dlss5_source_too_large"
    # Resolution alone does not decide it: 4K at 24fps plays and the same 4K at
    # 60fps does not, because the channel is pulled at playback speed. An
    # unknown frame rate is not treated as a refusal - the probe simply has
    # nothing to say yet, and the resolution ceiling still applies.
    budget = config.DLSS5_REALTIME_PIXEL_RATE
    if budget > 0 and fps > 0 and width * height * float(fps) > budget:
        return "dlss5_source_too_fast"
    if not cuda_context_ready():
        # Without the blocking-sync primary context the CUDA zero-copy path
        # refuses every frame; see prepare_cuda_context_for_dlss5.
        return "dlss5_cuda_context_unprepared"
    return None


def realtime_source_allowed(
    width: int, height: int, *, is_10bit: bool = False, fps: float = 0.0
) -> bool:
    """Whether the realtime DLSS5 channel may be offered for this source."""
    return source_block_reason_dlss5(
        width, height, is_10bit=is_10bit, fps=fps, realtime=True
    ) is None
