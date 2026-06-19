"""Thread-safe runtime settings updated by the local UI control API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading

import config
from pipeline.light_match import LightMatchParams, normalize_light_match_params
from utils.si_filter import SIMixParams, normalize_si_mix_params


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
    version: int = 0

    def params(self) -> SIMixParams:
        return SIMixParams(
            enabled=self.enabled,
            mix_channel=self.mix_channel,
            original_volume_percent=self.original_volume_percent,
            si_volume_percent=self.si_volume_percent,
            si_delay_seconds=self.si_delay_seconds,
            duck_original=self.duck_original,
        )

    def to_dict(self) -> dict:
        return asdict(self)


_lock = threading.RLock()
_light_match_state = LightMatchRuntime(
    **normalize_light_match_params(config.LIGHT_MATCH_DICT).__dict__,
    version=0,
)
_si_mix_state = SIMixRuntime(
    **normalize_si_mix_params(config.SI_MIX_DICT).to_dict(),
    version=0,
)


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
