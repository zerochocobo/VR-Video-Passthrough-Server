"""Centralized VR filename marker and generated-title helpers."""
from __future__ import annotations

import re


# Player naming references checked for this rule set:
# - HereSphere/SKYBOX/DeoVR: this project treats only underscore and hyphen as
#   reliable filename marker separators.
# - SKYBOX: 3D/angle/fisheye keywords can appear in any order/case.
# - DeoVR: local filenames use markers such as _LR_180 and _fisheye190.
_MARKER_RE = re.compile(
    r"(^|[_\-])("
    r"lr|rl|lrf|rlf|3dh|3dhf|sbs|sbsf|hsbs|"
    r"left[-_]*right|left[-_]*by[-_]*right|"
    r"half[-_]*sbs|half[-_]*side[-_]*by[-_]*side|"
    r"side[-_]*by[-_]*side|"
    r"tb|bt|tbf|btf|ou|ouf|3dv|3dvf|hou|"
    r"top[-_]*bottom|top[-_]*by[-_]*bottom|"
    r"over[-_]*under|half[-_]*ou|half[-_]*over[-_]*under|"
    r"3d|3dh|3dv|2d|"
    r"180|360|180x180|vr180|"
    r"f180|180f|fisheye|fisheye180|fisheye190|rf52|"
    r"mkx200|mkx22|vrca220|eac360|360eac|"
    r"alpha|passthrough|3d_alpha|fisheye_alpha|f180_alpha|sbs_f180_alpha|"
    r"lr_180_fisheye_alpha|lr_180_fisheye_f180_alpha"
    r")($|[_\-])",
    re.IGNORECASE,
)

# Fisheye source markers -> the lens field of view they stand for. Players use
# these in local filenames (DeoVR reads _fisheye190, SLR ships MKX200/VRCA220),
# and they are the only reliable source of the FOV: 180, 190 and 200 deg circles
# all fill the same inscribed circle, so no pixel test can tell them apart.
_FISHEYE_FOV_BY_MARKER = {
    "fisheye": 180.0,
    "f180": 180.0,
    "180f": 180.0,
    "rf52": 190.0,
    "mkx200": 200.0,
    "mkx22": 220.0,
    "mkx220": 220.0,
    "vrca220": 220.0,
}
_FISHEYE_MARKER_RE = re.compile(
    r"(?:^|[_\-])(fisheye(\d{3})?|f180|180f|mkx220|mkx200|mkx22|vrca220|rf52)(?=$|[_\-])",
    re.IGNORECASE,
)
# Markers that describe how a stem was produced rather than what the source is.
# An alpha output fed back in must not be read as a fisheye source, and its own
# markers must not survive into a new generated name.
_GENERATED_MARKER_RE = re.compile(
    r"(?:^|[_\-])(alpha|passthrough|3d_alpha|fisheye_alpha|f180_alpha|sbs_f180_alpha)(?=$|[_\-])",
    re.IGNORECASE,
)
# Lowest and highest lens FOV this project will honour from a filename.
FISHEYE_FOV_MIN = 140.0
FISHEYE_FOV_MAX = 280.0

SBS_180_SOURCE_SUFFIX = "_LR_180_SBS"
LEGACY_LR_180_SOURCE_SUFFIX = "_LR_180"
GREEN_LIVE_PASSTHROUGH_SUFFIX = "_passthrough"
GREEN_OFFLINE_PASSTHROUGH_SUFFIX = "_LR_180_SBS_passthrough"
ALPHA_PASSTHROUGH_SUFFIX = "_LR_180_FISHEYE_F180_alpha"
TWO_DVR_SUFFIX = "_3D_LR_Screen"
SUPERRES_PREFIX = "[SuperRes]"
DLSS5_PREFIX = "[DLSS5]"
# Face beautification keeps the source geometry, so the marker is appended after
# any existing VR markers (which stay matchable, being followed by "_").
FACE_BEAUTY_SUFFIX = "_beauty"
# Display prefix for the realtime DLNA entry, alongside [RM] / [2D>3D].
FACE_BEAUTY_PREFIX = "[FaceBeauty]"
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".ts", ".m2ts"}


def _as_stem(stem_or_name: str) -> str:
    value = str(stem_or_name)
    dot = value.rfind(".")
    if dot > 0:
        ext = value[dot + 1:].lower()
        if f".{ext}" in _VIDEO_SUFFIXES:
            return value[:dot]
    return value


def has_vr_filename_marker(stem_or_name: str) -> bool:
    """Return whether a filename stem/name already carries known VR markers."""
    stem = _as_stem(stem_or_name)
    return bool(_MARKER_RE.search(stem))


def is_half_equirectangular_source(width: int = 0, height: int = 0) -> bool:
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 or height <= 0:
        return False
    return abs((width / height) - 2.0) <= 0.02


def source_display_stem(stem_or_name: str, width: int = 0, height: int = 0) -> str:
    """Return the DLNA display stem for a source video.

    2:1 half-equirectangular sources without known VR/player markers are exposed
    as SBS 180 virtual names so VR players enter the intended VR180 mode.
    """
    stem = _as_stem(stem_or_name)
    if is_half_equirectangular_source(width, height) and not has_vr_filename_marker(stem):
        return f"{stem}{SBS_180_SOURCE_SUFFIX}"
    if stem.lower().endswith(LEGACY_LR_180_SOURCE_SUFFIX.lower()):
        return f"{stem}_SBS"
    return stem


def green_passthrough_stem(stem_or_name: str, width: int = 0, height: int = 0) -> str:
    return f"{source_display_stem(stem_or_name, width, height)}{GREEN_LIVE_PASSTHROUGH_SUFFIX}"


def parse_fisheye_fov(stem_or_name: str) -> float:
    """Return the source lens FOV in degrees from filename markers, else 0.0.

    ``_fisheye190`` -> 190, ``MKX200`` -> 200, a bare ``_fisheye`` -> 180.
    Generated markers (``_alpha``, ``_passthrough``) are ignored first, so an
    alpha output that is fed back in is not mistaken for a fisheye source.
    """
    stem = _GENERATED_MARKER_RE.sub("_", _as_stem(stem_or_name))
    for match in _FISHEYE_MARKER_RE.finditer(stem):
        marker = match.group(1).lower()
        digits = match.group(2)
        if digits:
            fov = float(digits)
            if FISHEYE_FOV_MIN <= fov <= FISHEYE_FOV_MAX:
                return fov
            continue
        fov = _FISHEYE_FOV_BY_MARKER.get(marker)
        if fov:
            return float(fov)
    return 0.0


def strip_projection_markers(stem_or_name: str) -> str:
    """Drop fisheye/generated markers so a new one can state the real geometry.

    ``TEST_91983_FISHEYE190_x265`` -> ``TEST_91983_x265``. Without this the
    generated name would carry both the source's ``FISHEYE190`` and this
    project's ``FISHEYE_F180``, and the player would read the wrong one.
    """
    stem = _as_stem(stem_or_name)
    for pattern in (_FISHEYE_MARKER_RE, _GENERATED_MARKER_RE):
        while True:
            stripped = pattern.sub("_", stem)
            if stripped == stem:
                break
            stem = stripped
    stem = re.sub(r"[_\-]{2,}", "_", stem)
    return stem.strip("_- ") or _as_stem(stem_or_name)


def alpha_passthrough_stem(stem_or_name: str) -> str:
    """Return alpha passthrough stem using LR 180 + fisheye markers."""
    stem = _as_stem(stem_or_name)
    if stem.lower().endswith(ALPHA_PASSTHROUGH_SUFFIX.lower()):
        return stem
    return f"{strip_projection_markers(stem)}{ALPHA_PASSTHROUGH_SUFFIX}"


def two_dvr_stem(stem_or_name: str) -> str:
    stem = _as_stem(stem_or_name)
    if stem.lower().endswith(TWO_DVR_SUFFIX.lower()):
        return stem
    return f"{stem}{TWO_DVR_SUFFIX}"


def superres_stem(stem_or_name: str) -> str:
    """Return a stable RTX VSR output stem with the SuperRes marker."""
    stem = _as_stem(stem_or_name)
    if stem.lower().startswith(SUPERRES_PREFIX.lower()):
        return stem
    return f"{SUPERRES_PREFIX}{stem}"


def dlss5_stem(stem_or_name: str) -> str:
    """Return a stable DLSS5 output stem with the DLSS5 marker."""
    stem = _as_stem(stem_or_name)
    if stem.lower().startswith(DLSS5_PREFIX.lower()):
        return stem
    return f"{DLSS5_PREFIX}{stem}"


def dlss5_output_stem(stem_or_name: str) -> str:
    """Return the offline DLSS5 NR stem."""
    return dlss5_stem(stem_or_name)


def superres_target_tag(target_height: int | None) -> str:
    """Short user-facing name for a SuperRes target: 1X, 2K, 4K, 6K or 8K."""
    height = 2160 if target_height is None else int(target_height)
    if height <= 0:
        return "1X"
    if height <= 1440:
        return "2K"
    if height >= 4096:
        return "8K"
    if height >= 3072:
        return "6K"
    return "4K"


def superres_output_stem(stem_or_name: str, target_height: int = 2160) -> str:
    """Return the offline RTX VSR stem with a user-facing resolution suffix."""
    stem = _as_stem(stem_or_name)
    stem = re.sub(r"_(?:1X|2K|4K|6K|8K)$", "", stem, flags=re.IGNORECASE)
    return f"{stem}_{superres_target_tag(target_height)}"


def live_passthrough_title(stem_or_name: str, mode: str, width: int = 0, height: int = 0) -> str:
    if mode == "alpha":
        return f"{alpha_passthrough_stem(stem_or_name)}_live"
    if mode == "two_dvr":
        return f"[2D>3D]{two_dvr_stem(stem_or_name)}_live"
    if mode == "rm":
        return f"[RM]{_as_stem(stem_or_name)}_live"
    if mode == "face_beauty":
        return f"{FACE_BEAUTY_PREFIX}{_as_stem(stem_or_name)}_live"
    if mode == "superres":
        return f"{superres_stem(source_display_stem(stem_or_name, width, height))}_live"
    if mode == "dlss5":
        # 1x: the output keeps the source geometry, so it keeps the source's
        # display stem and the VR markers a player reads out of it.
        return f"{dlss5_stem(source_display_stem(stem_or_name, width, height))}_live"
    return f"{green_passthrough_stem(stem_or_name, width, height)}_live"


def offline_passthrough_stem(stem_or_name: str, mode: str, width: int = 0, height: int = 0) -> str:
    if mode == "alpha":
        return alpha_passthrough_stem(stem_or_name)
    stem = _as_stem(stem_or_name)
    if stem.lower().endswith(GREEN_OFFLINE_PASSTHROUGH_SUFFIX.lower()):
        return stem
    if is_half_equirectangular_source(width, height):
        return f"{source_display_stem(stem, width, height)}{GREEN_LIVE_PASSTHROUGH_SUFFIX}"
    return f"{stem}{GREEN_LIVE_PASSTHROUGH_SUFFIX}"
