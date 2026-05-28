"""Helpers for mapping virtual byte ranges to passthrough start times."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ByteSeekMapping:
    """Result of interpreting a byte Range start inside a virtual CBR resource."""

    prefix: bool
    start: int
    total: int
    header_bytes: int
    ratio: float
    time_sec: float
    snapped_time_sec: float
    gop_seconds: float


def gop_duration_seconds(gop_frames: int | float, output_fps: int | float) -> float:
    frames = float(gop_frames or 0)
    fps = float(output_fps or 0)
    if frames <= 0 or fps <= 0:
        return 0.0
    return max(0.0, frames / fps)


def snap_back_to_gop(seconds: float, gop_seconds: float) -> float:
    value = max(0.0, float(seconds or 0.0))
    gop = float(gop_seconds or 0.0)
    if gop <= 0:
        return value
    return math.floor(value / gop) * gop


def map_byte_start_to_time(
    *,
    start: int,
    total: int,
    duration_sec: float,
    header_bytes: int,
    output_fps: float,
    gop_frames: int,
) -> ByteSeekMapping:
    """Map an HTTP byte Range start to a generated media start time.

    Bytes below ``header_bytes`` belong to the stable prefix region and should
    be served from a real prefix cache instead of starting a new producer.
    """

    total_i = max(0, int(total or 0))
    start_i = max(0, int(start or 0))
    duration = max(0.0, float(duration_sec or 0.0))
    header = max(0, int(header_bytes or 0))
    if total_i > 0:
        header = min(header, total_i)
    prefix = start_i < header
    gop_seconds = gop_duration_seconds(gop_frames, output_fps)
    if prefix or total_i <= 0 or duration <= 0 or header >= total_i:
        return ByteSeekMapping(
            prefix=prefix,
            start=start_i,
            total=total_i,
            header_bytes=header,
            ratio=0.0,
            time_sec=0.0,
            snapped_time_sec=0.0,
            gop_seconds=gop_seconds,
        )
    media_span = max(1, total_i - header)
    ratio = min(1.0, max(0.0, float(start_i - header) / float(media_span)))
    time_sec = min(duration, ratio * duration)
    snapped = min(duration, snap_back_to_gop(time_sec, gop_seconds))
    return ByteSeekMapping(
        prefix=False,
        start=start_i,
        total=total_i,
        header_bytes=header,
        ratio=ratio,
        time_sec=time_sec,
        snapped_time_sec=snapped,
        gop_seconds=gop_seconds,
    )
