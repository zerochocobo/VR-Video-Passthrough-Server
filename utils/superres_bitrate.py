"""Bitrate planning for offline RTX Super Resolution outputs."""
from __future__ import annotations

import math
from dataclasses import dataclass


SUPERRES_BITRATE_MODES = ("auto", "1.2", "1.5", "2", "3")
DEFAULT_SOURCE_BITRATE = 20_000_000
AUTO_MAX_BITRATE = 80_000_000
MANUAL_MAX_BITRATE = 120_000_000


@dataclass(frozen=True)
class SuperResBitratePlan:
    mode: str
    source_bps: int
    target_bps: int
    max_bps: int
    buffer_bps: int
    pixel_ratio: float


def normalize_superres_bitrate_mode(value: str | float | int | None) -> str:
    text = str(value or "auto").strip().lower().removesuffix("x")
    aliases = {"default": "auto", "recommended": "auto", "2.0": "2", "3.0": "3"}
    text = aliases.get(text, text)
    return text if text in SUPERRES_BITRATE_MODES else "auto"


def plan_superres_bitrate(
    source_bps: int | None,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    mode: str | float | int | None = "auto",
) -> SuperResBitratePlan:
    """Return a VBR plan based on source bitrate and the selected UI mode.

    Auto scales by the square root of the output/input pixel ratio. Manual
    modes are direct source-bitrate multipliers, independent of resolution.
    """
    normalized = normalize_superres_bitrate_mode(mode)
    source = max(1, int(source_bps or DEFAULT_SOURCE_BITRATE))
    input_pixels = max(1, int(in_w) * int(in_h))
    output_pixels = max(1, int(out_w) * int(out_h))
    pixel_ratio = output_pixels / float(input_pixels)
    if normalized == "auto":
        multiplier = math.sqrt(pixel_ratio)
        ceiling = AUTO_MAX_BITRATE
    else:
        multiplier = float(normalized)
        ceiling = MANUAL_MAX_BITRATE
    target = max(1, min(ceiling, int(round(source * multiplier))))
    return SuperResBitratePlan(
        mode=normalized,
        source_bps=source,
        target_bps=target,
        max_bps=int(round(target * 1.25)),
        buffer_bps=int(round(target * 2.0)),
        pixel_ratio=pixel_ratio,
    )
