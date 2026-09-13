"""Shared SuperRes target choices for the dashboard, dialogs and offline page.

The values are output heights, with 0 meaning native 1x enhancement. 6K and 8K
are VR targets: ordinary 2D sources fall back to 4K (see
`utils.rtx_vsr.effective_offline_target_height`).
"""
from __future__ import annotations

from utils.rtx_vsr import NATIVE_TARGET_HEIGHT

TARGET_CHOICES: tuple[int, ...] = (NATIVE_TARGET_HEIGHT, 1440, 2160, 3072, 4096)
# Realtime defaults to native 1x: it is the target that can be served as a
# virtual file (draggable progress bar), and it keeps up with playback.
REALTIME_DEFAULT_TARGET = NATIVE_TARGET_HEIGHT
# Offline has no playback-speed constraint, so enlarging stays its default.
OFFLINE_DEFAULT_TARGET = 4096
DEFAULT_TARGET = OFFLINE_DEFAULT_TARGET


def target_i18n_key(target_height: object) -> str:
    """Map any stored target height onto its user-facing label."""
    try:
        height = int(target_height)
    except (TypeError, ValueError):
        height = DEFAULT_TARGET
    if height <= NATIVE_TARGET_HEIGHT:
        return "superres.target_native"
    if height <= 1440:
        return "superres.target_2k"
    if height >= 4096:
        return "superres.target_8k_vr"
    if height >= 3072:
        return "superres.target_6k_vr"
    return "superres.target_4k"
