"""Thread-safe runtime settings updated by the local UI control API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading

import config
from pipeline.light_match import LightMatchParams, normalize_light_match_params


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


_lock = threading.RLock()
_state = LightMatchRuntime(
    **normalize_light_match_params(config.LIGHT_MATCH_DICT).__dict__,
    version=0,
)


def reset_for_test(data: dict | LightMatchParams | None = None) -> LightMatchRuntime:
    global _state
    params = normalize_light_match_params(config.LIGHT_MATCH_DICT if data is None else data)
    with _lock:
        _state = LightMatchRuntime(**params.__dict__, version=0)
        return _state


def get_light_match() -> LightMatchRuntime:
    with _lock:
        return _state


def set_light_match(data: dict | LightMatchParams) -> LightMatchRuntime:
    global _state
    params = normalize_light_match_params(data)
    with _lock:
        if params == _state.params():
            return _state
        _state = LightMatchRuntime(**params.__dict__, version=_state.version + 1)
        return _state
