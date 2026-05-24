"""DLNA protocolInfo helpers for derived passthrough resources."""
from __future__ import annotations

from config import PASSTHROUGH_CONTAINER, PASSTHROUGH_DLNA_PN, PASSTHROUGH_MAX_FPS


def passthrough_dlna_pn(backend_verdict: str | None = None) -> str:
    """Return the DLNA profile name advertised for passthrough output."""
    if PASSTHROUGH_DLNA_PN:
        return PASSTHROUGH_DLNA_PN
    if PASSTHROUGH_CONTAINER == "mpegts":
        return "HEVC_TS_NA_ISO"
    return "HEVC_MP4_MAIN"


def _fmt_fps(fps: float) -> str:
    return str(int(fps)) if float(fps).is_integer() else f"{float(fps):.3f}".rstrip("0").rstrip(".")


def passthrough_frame_rate(source_fps: float | None = None) -> str | None:
    """Return the advertised passthrough frame rate, or None for uncapped."""
    if PASSTHROUGH_MAX_FPS <= 0:
        return None
    fps = float(PASSTHROUGH_MAX_FPS)
    if source_fps and source_fps > 0:
        fps = min(fps, float(source_fps))
    return _fmt_fps(fps)
