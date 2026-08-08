"""Thread-safe runtime settings updated by the local UI control API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading

import config
from pipeline.light_match import LightMatchParams, normalize_light_match_params
from utils.si_filter import DEFAULT_DUB_MODE, SIMixParams, normalize_si_mix_params


@dataclass(frozen=True)
class LightMatchRuntime:
    enabled: bool
    temp_k: int
    tint: float
    exposure_ev: float
    contrast: float
    gamma: float
    saturation: float
    preset: str
    version: int = 0

    def params(self) -> LightMatchParams:
        return LightMatchParams(
            enabled=self.enabled,
            temp_k=self.temp_k,
            tint=self.tint,
            exposure_ev=self.exposure_ev,
            contrast=self.contrast,
            gamma=self.gamma,
            saturation=self.saturation,
            preset=self.preset,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SIMixRuntime:
    enabled: bool
    mix_channel: str
    original_volume_percent: int
    si_volume_percent: int
    si_delay_seconds: float
    duck_original: bool
    duck_preset: str
    dub_mode_enabled: bool = DEFAULT_DUB_MODE
    version: int = 0

    def params(self) -> SIMixParams:
        return SIMixParams(
            enabled=self.enabled,
            mix_channel=self.mix_channel,
            original_volume_percent=self.original_volume_percent,
            si_volume_percent=self.si_volume_percent,
            si_delay_seconds=self.si_delay_seconds,
            duck_original=self.duck_original,
            duck_preset=self.duck_preset,
            dub_mode_enabled=self.dub_mode_enabled,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RmRuntime:
    """Realtime mosaic-restoration toggle. ``version`` bumps on every change so
    the DLNA SystemUpdateID refreshes and clients re-Browse."""
    enabled: bool
    version: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FaceBeautyRuntime:
    """Realtime face-beautification state. ``enabled`` gates the [FaceBeauty]
    DLNA entry; the rest are the strength knobs the dashboard dialog edits, kept
    as 0-100 percentages so they match the offline UI and CLI one to one.
    ``version`` bumps on every change so DLNA's SystemUpdateID refreshes."""
    enabled: bool
    preset: str = "standard"
    enhancer_blend: int = 80
    skin_smooth: int = 40
    skin_brighten: int = 15
    skin_even: int = 20
    eye_brighten: int = 20
    teeth_white: int = 20
    lip_vivid: int = 10
    sharpen: int = 15
    version: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def strengths(self) -> dict:
        return {k: getattr(self, k) for k in FACE_BEAUTY_STRENGTH_KEYS}


FACE_BEAUTY_STRENGTH_KEYS = (
    "enhancer_blend", "skin_smooth", "skin_brighten", "skin_even",
    "eye_brighten", "teeth_white", "lip_vivid", "sharpen",
)


def _face_beauty_defaults() -> dict:
    """Preset percentages from the offline engine, so realtime and offline
    cannot drift apart. Falls back to the dataclass defaults if the offline
    module is unavailable (it pulls in cv2/onnxruntime)."""
    try:
        from offline.face_beauty_engine import preset_percentages

        return preset_percentages(config.FACE_BEAUTY_PRESET)
    except Exception:
        return {}


_lock = threading.RLock()
_light_match_state = LightMatchRuntime(
    **normalize_light_match_params(config.LIGHT_MATCH_DICT).__dict__,
    version=0,
)
_si_mix_state = SIMixRuntime(
    **normalize_si_mix_params(config.SI_MIX_DICT).to_dict(),
    version=0,
)
_rm_state = RmRuntime(enabled=bool(config.RM_ENABLED), version=0)


def reset_for_test(data: dict | LightMatchParams | None = None) -> LightMatchRuntime:
    global _light_match_state
    params = normalize_light_match_params(config.LIGHT_MATCH_DICT if data is None else data)
    with _lock:
        _light_match_state = LightMatchRuntime(**params.__dict__, version=0)
        return _light_match_state


def get_light_match() -> LightMatchRuntime:
    with _lock:
        return _light_match_state


def set_light_match(data: dict | LightMatchParams) -> LightMatchRuntime:
    global _light_match_state
    params = normalize_light_match_params(data)
    with _lock:
        if params == _light_match_state.params():
            return _light_match_state
        _light_match_state = LightMatchRuntime(**params.__dict__, version=_light_match_state.version + 1)
        return _light_match_state


def reset_si_mix_for_test(data: dict | SIMixParams | None = None) -> SIMixRuntime:
    global _si_mix_state
    params = normalize_si_mix_params(config.SI_MIX_DICT if data is None else data)
    with _lock:
        _si_mix_state = SIMixRuntime(**params.to_dict(), version=0)
        return _si_mix_state


def get_si_mix() -> SIMixRuntime:
    with _lock:
        return _si_mix_state


def set_si_mix(data: dict | SIMixParams) -> SIMixRuntime:
    global _si_mix_state
    params = normalize_si_mix_params(data)
    with _lock:
        if params == _si_mix_state.params():
            return _si_mix_state
        _si_mix_state = SIMixRuntime(**params.to_dict(), version=_si_mix_state.version + 1)
        return _si_mix_state


def reset_rm_for_test(enabled: bool | None = None) -> RmRuntime:
    global _rm_state
    with _lock:
        _rm_state = RmRuntime(
            enabled=bool(config.RM_ENABLED if enabled is None else enabled),
            version=0,
        )
        return _rm_state


def get_rm() -> RmRuntime:
    with _lock:
        return _rm_state


def set_rm(data: dict | bool) -> RmRuntime:
    global _rm_state
    if isinstance(data, dict):
        enabled = bool(data.get("enabled", _rm_state.enabled))
    else:
        enabled = bool(data)
    with _lock:
        if enabled == _rm_state.enabled:
            return _rm_state
        _rm_state = RmRuntime(enabled=enabled, version=_rm_state.version + 1)
        return _rm_state


def reset_face_beauty_for_test(enabled: bool | None = None) -> FaceBeautyRuntime:
    global _face_beauty_state
    with _lock:
        _face_beauty_state = FaceBeautyRuntime(
            enabled=bool(config.FACE_BEAUTY_ENABLED if enabled is None else enabled),
            preset=str(config.FACE_BEAUTY_PRESET),
            version=0,
            **_face_beauty_defaults(),
        )
        return _face_beauty_state


def get_face_beauty() -> FaceBeautyRuntime:
    with _lock:
        return _face_beauty_state


def set_face_beauty(data: dict | bool) -> FaceBeautyRuntime:
    global _face_beauty_state
    with _lock:
        current = _face_beauty_state
        if isinstance(data, dict):
            values = {
                "enabled": bool(data.get("enabled", current.enabled)),
                "preset": str(data.get("preset", current.preset)),
            }
            for key in FACE_BEAUTY_STRENGTH_KEYS:
                raw = data.get(key, getattr(current, key))
                try:
                    values[key] = max(0, min(100, int(raw)))
                except (TypeError, ValueError):
                    values[key] = getattr(current, key)
        else:
            values = {"enabled": bool(data), "preset": current.preset,
                      **current.strengths()}
        if all(getattr(current, k) == v for k, v in values.items()):
            return current
        _face_beauty_state = FaceBeautyRuntime(version=current.version + 1, **values)
        return _face_beauty_state


_face_beauty_state = FaceBeautyRuntime(
    enabled=bool(config.FACE_BEAUTY_ENABLED),
    preset=str(config.FACE_BEAUTY_PRESET),
    **_face_beauty_defaults(),
)
