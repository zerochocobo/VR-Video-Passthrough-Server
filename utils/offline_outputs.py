"""Helpers for detecting generated offline passthrough outputs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import config


OFFLINE_PASSTHROUGH_SUFFIXES = (
    "_passthrough",
    "_LR_180_SBS_passthrough",
    "_LR_180_passthrough",
    "_SBS_180_passthrough",
    "_FISHEYE_alpha",
    "_FISHEYE180_alpha",
    "_FISHEYE190_alpha",
    "_3D_alpha",
    "_LR_180_FISHEYE_alpha",
    "_LR_180_FISHEYE_F180_alpha",
    "_SBS_F180_alpha",
)

_ENGINE_SEGMENT_RE = re.compile(
    r"_(?:rvm1|rvm2|rvm|matanyone2m|matanyone2)_s\d{6}_(?:all|\d+s|\d+m)$",
    re.IGNORECASE,
)


def is_offline_passthrough_output_name(name: str) -> bool:
    stem = Path(name).stem.lower()
    return any(stem.endswith(suffix.lower()) for suffix in OFFLINE_PASSTHROUGH_SUFFIXES)


def matches_offline_output_for_source(source: Path, candidate: Path) -> bool:
    if candidate == source or candidate.suffix.lower() not in config.VIDEO_EXTS:
        return False
    source_stem = source.stem.lower()
    candidate_stem = candidate.stem.lower()
    for suffix in OFFLINE_PASSTHROUGH_SUFFIXES:
        suffix_l = suffix.lower()
        if candidate_stem == f"{source_stem}{suffix_l}":
            return True
        if not candidate_stem.startswith(f"{source_stem}_") or not candidate_stem.endswith(suffix_l):
            continue
        middle = candidate_stem[len(source_stem):-len(suffix_l)]
        if _ENGINE_SEGMENT_RE.fullmatch(middle):
            return True
    return False


def has_offline_passthrough_output(source: Path, siblings: Iterable[Path] | None = None) -> bool:
    if source.suffix.lower() not in config.VIDEO_EXTS:
        return False
    try:
        candidates = list(source.parent.iterdir()) if siblings is None else siblings
    except OSError:
        return False
    return any(matches_offline_output_for_source(source, candidate) for candidate in candidates)
