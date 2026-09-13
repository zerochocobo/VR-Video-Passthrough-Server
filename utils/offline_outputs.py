"""Helpers for naming and publishing generated offline outputs."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import config
from utils.vr_naming import strip_projection_markers


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

OFFLINE_TWO_DVR_SUFFIXES = (
    "_3D_LR_Screen",
)

OFFLINE_GENERATED_SUFFIXES = OFFLINE_PASSTHROUGH_SUFFIXES + OFFLINE_TWO_DVR_SUFFIXES

_ENGINE_SEGMENT_RE = re.compile(
    r"_(?:rvm1|rvm|matanyone2m|matanyone2)_s\d{6}(?:_e\d{6})?_(?:all|\d+s|\d+m)$",
    re.IGNORECASE,
)

# Optional segment tag that offline 2D->3D flat outputs append between the source
# stem and the ``_3D_LR_Screen`` suffix (see offline/two_dvr.py output_path):
#   ""  |  "_S<hhmmss>"  |  "_S<hhmmss>_E<hhmmss>"  |  "_SEG<n>_S<hhmmss>_E<hhmmss>"
_TWO_DVR_SEGMENT_RE = re.compile(
    r"(?:_SEG\d+_S\d{6}_E\d{6}|_S\d{6}(?:_E\d{6})?)?",
    re.IGNORECASE,
)


# Work files the offline tools write while a run is in flight: the video-only
# encode before its audio mux, the audio sidecar, and the VMP4 build scratch.
# They are neither sources nor finished outputs, so every batch scanner must
# skip them -- a leftover from an interrupted run is not a video to convert.
INTERNAL_INTERMEDIATE_MARKERS = ("._video_only", "._audio", ".partial.", ".vmp4build")


def is_internal_intermediate_name(name: str) -> bool:
    lowered = Path(name).name.lower()
    # A leading dot is how the tools hide their scratch files next to the
    # output they are building (rtx_vsr_pynv's GPU video, segment work dirs).
    if lowered.startswith("."):
        return True
    return lowered.endswith(".tmp") or any(marker in lowered for marker in INTERNAL_INTERMEDIATE_MARKERS)


def pending_output_path(out: Path) -> Path:
    """Where a run writes while it is still producing ``out``.

    Offline modes encode straight into their final name, so a crash, a GPU
    error or a user stopping the run leaves a truncated mp4 that has no moov
    box -- and the next ``--skip-existing`` batch treats it as finished work.
    Writing here first and renaming on success keeps the final name meaning
    "this file is complete". The pid keeps two runs aimed at one output from
    writing into the same scratch file.
    """
    return out.with_name(f"{out.stem}.partial.{os.getpid()}{out.suffix}")


def publish_pending_output(pending: Path, out: Path) -> bool:
    """Rename a finished pending file onto its final name."""
    try:
        os.replace(pending, out)
        return True
    except OSError:
        return False


def discard_pending_output(pending: Path | None) -> None:
    if pending is None:
        return
    try:
        pending.unlink(missing_ok=True)
    except Exception:
        pass


def is_offline_passthrough_output_name(name: str) -> bool:
    stem = Path(name).stem.lower()
    return any(stem.endswith(suffix.lower()) for suffix in OFFLINE_GENERATED_SUFFIXES)


def _source_stem_variants(source_stem: str) -> tuple[str, ...]:
    """Stems an output of this source can be named after.

    Alpha output drops the source's own projection markers (``_FISHEYE190``),
    so the generated name no longer starts with the full source stem.
    """
    stripped = strip_projection_markers(source_stem).lower()
    return (source_stem,) if stripped == source_stem else (source_stem, stripped)


def matches_offline_output_for_source(source: Path, candidate: Path) -> bool:
    if candidate == source or candidate.suffix.lower() not in config.VIDEO_EXTS:
        return False
    candidate_stem = candidate.stem.lower()
    for source_stem in _source_stem_variants(source.stem.lower()):
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


def matches_offline_two_dvr_output_for_source(source: Path, candidate: Path) -> bool:
    if candidate == source or candidate.suffix.lower() not in config.VIDEO_EXTS:
        return False
    source_stem = source.stem.lower()
    candidate_stem = candidate.stem.lower()
    if candidate_stem.startswith(f"{source_stem}_2dvr_") and "_flat3d_lr_sbs" in candidate_stem:
        return True
    for suffix in OFFLINE_TWO_DVR_SUFFIXES:
        suffix_l = suffix.lower()
        if candidate_stem == f"{source_stem}{suffix_l}":
            return True
        if candidate_stem.startswith(f"{source_stem}_") and candidate_stem.endswith(suffix_l):
            middle = candidate_stem[len(source_stem):-len(suffix_l)]
            if _TWO_DVR_SEGMENT_RE.fullmatch(middle):
                return True
    return False


def has_offline_two_dvr_output(source: Path, siblings: Iterable[Path] | None = None) -> bool:
    if source.suffix.lower() not in config.VIDEO_EXTS:
        return False
    try:
        candidates = list(source.parent.iterdir()) if siblings is None else siblings
    except OSError:
        return False
    return any(matches_offline_two_dvr_output_for_source(source, candidate) for candidate in candidates)
