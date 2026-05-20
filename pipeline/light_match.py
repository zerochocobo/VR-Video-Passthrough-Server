"""Ambient light matching coefficients for passthrough foreground video."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


LIGHT_MATCH_DEVICE_SRC = r"""
// coeffs layout:
// [0]=Y gain, [1]=Y bias,
// [2]=U from U, [3]=U from V, [4]=U bias,
// [5]=V from U, [6]=V from V, [7]=V bias.
__device__ unsigned char clamp_light_y(float v) {
    v = v < 16.f ? 16.f : (v > 235.f ? 235.f : v);
    return (unsigned char)(v + 0.5f);
}

__device__ unsigned char clamp_light_uv(float v) {
    v = v < 16.f ? 16.f : (v > 240.f ? 240.f : v);
    return (unsigned char)(v + 0.5f);
}

__device__ void apply_light_match(
    float* y,
    float* u,
    float* v,
    const float* __restrict__ coeffs,
    const unsigned char* __restrict__ gamma_lut,
    int identity
) {
    if (identity) return;
    float yin = *y;
    float uin = *u - 128.f;
    float vin = *v - 128.f;
    unsigned char yidx = clamp_light_y(yin * coeffs[0] + coeffs[1]);
    *y = (float)gamma_lut[yidx];
    *u = (float)clamp_light_uv(uin * coeffs[2] + vin * coeffs[3] + coeffs[4] + 128.f);
    *v = (float)clamp_light_uv(uin * coeffs[5] + vin * coeffs[6] + coeffs[7] + 128.f);
}

__device__ void apply_light_match_y_only(
    float* y,
    const float* __restrict__ coeffs,
    const unsigned char* __restrict__ gamma_lut,
    int identity
) {
    if (identity) return;
    unsigned char yidx = clamp_light_y((*y) * coeffs[0] + coeffs[1]);
    *y = (float)gamma_lut[yidx];
}

__device__ void apply_light_match_uv_only(
    float* u,
    float* v,
    const float* __restrict__ coeffs,
    int identity
) {
    if (identity) return;
    float uin = *u - 128.f;
    float vin = *v - 128.f;
    *u = (float)clamp_light_uv(uin * coeffs[2] + vin * coeffs[3] + coeffs[4] + 128.f);
    *v = (float)clamp_light_uv(uin * coeffs[5] + vin * coeffs[6] + coeffs[7] + 128.f);
}
"""


TEMP_K_MIN = 2700
TEMP_K_MAX = 9000
NEUTRAL_TEMP_K = 6500
DEFAULT_TEMP_K = 5500
TINT_MIN = -50.0
TINT_MAX = 50.0
EXPOSURE_EV_MIN = -2.0
EXPOSURE_EV_MAX = 2.0
CONTRAST_MIN = 0.5
CONTRAST_MAX = 1.5
GAMMA_MIN = 0.7
GAMMA_MAX = 1.4
SATURATION_MIN = 0.0
SATURATION_MAX = 2.0
PRESETS = {"custom", "home_warm", "daylight", "night_cool"}


@dataclass(frozen=True)
class LightMatchParams:
    enabled: bool = False
    temp_k: int = DEFAULT_TEMP_K
    tint: float = 0.0
    exposure_ev: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    preset: str = "custom"


@dataclass(frozen=True)
class LightMatchTables:
    coeffs: np.ndarray
    gamma_lut: np.ndarray
    identity: bool


def _clamp_float(value, minimum: float, maximum: float, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if math.isnan(out) or math.isinf(out):
        out = default
    return max(minimum, min(maximum, out))


def _clamp_int(value, minimum: int, maximum: int, default: int) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        out = default
    return max(minimum, min(maximum, out))


def normalize_light_match_params(data: dict | LightMatchParams | None = None, **overrides) -> LightMatchParams:
    if isinstance(data, LightMatchParams):
        base = data.__dict__.copy()
    elif isinstance(data, dict):
        base = dict(data)
    else:
        base = {}
    base.update({k: v for k, v in overrides.items() if v is not None})
    preset = str(base.get("preset", "custom") or "custom").strip().lower()
    if preset not in PRESETS:
        preset = "custom"
    return LightMatchParams(
        enabled=bool(base.get("enabled", False)),
        temp_k=_clamp_int(base.get("temp_k", DEFAULT_TEMP_K), TEMP_K_MIN, TEMP_K_MAX, DEFAULT_TEMP_K),
        tint=_clamp_float(base.get("tint", 0.0), TINT_MIN, TINT_MAX, 0.0),
        exposure_ev=_clamp_float(base.get("exposure_ev", 0.0), EXPOSURE_EV_MIN, EXPOSURE_EV_MAX, 0.0),
        contrast=_clamp_float(base.get("contrast", 1.0), CONTRAST_MIN, CONTRAST_MAX, 1.0),
        gamma=_clamp_float(base.get("gamma", 1.0), GAMMA_MIN, GAMMA_MAX, 1.0),
        saturation=_clamp_float(base.get("saturation", 1.0), SATURATION_MIN, SATURATION_MAX, 1.0),
        preset=preset,
    )


def _kelvin_to_rgb(temp_k: int) -> tuple[float, float, float]:
    temp = max(1000.0, min(40000.0, float(temp_k))) / 100.0
    if temp <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(temp) - 161.1195681661
        b = 0.0 if temp <= 19.0 else 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        b = 255.0
    return tuple(max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b))


def _rgb_gains(temp_k: int, tint: float, exposure_ev: float) -> tuple[float, float, float]:
    target = _kelvin_to_rgb(temp_k)
    neutral = _kelvin_to_rgb(NEUTRAL_TEMP_K)
    exposure = 2.0 ** exposure_ev
    # Positive tint is magenta, negative tint is green.
    tint_magenta = max(0.0, tint) / 50.0
    tint_green = max(0.0, -tint) / 50.0
    r = target[0] / max(neutral[0], 1.0e-6)
    g = target[1] / max(neutral[1], 1.0e-6)
    b = target[2] / max(neutral[2], 1.0e-6)
    r *= 1.0 + 0.12 * tint_magenta
    b *= 1.0 + 0.12 * tint_magenta
    g *= 1.0 + 0.12 * tint_green - 0.08 * tint_magenta
    return r * exposure, g * exposure, b * exposure


def _rgb_to_yuv_limited(r: float, g: float, b: float) -> tuple[float, float, float]:
    return (
        16.0 + 0.182586 * r + 0.614231 * g + 0.062007 * b,
        128.0 - 0.100644 * r - 0.338572 * g + 0.439216 * b,
        128.0 + 0.439216 * r - 0.398942 * g - 0.040274 * b,
    )


def _yuv_limited_to_rgb(y: float, u: float, v: float) -> tuple[float, float, float]:
    c = max(0.0, y - 16.0)
    d = u - 128.0
    e = v - 128.0
    return (
        1.16438356 * c + 1.79274107 * e,
        1.16438356 * c - 0.21324861 * d - 0.53290933 * e,
        1.16438356 * c + 2.11240179 * d,
    )


def _apply_rgb_gains_to_yuv(y: float, u: float, v: float, gains: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = _yuv_limited_to_rgb(y, u, v)
    return _rgb_to_yuv_limited(r * gains[0], g * gains[1], b * gains[2])


def _luma_preserving_gains(gains: tuple[float, float, float]) -> tuple[float, float, float]:
    luma_gain = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2]
    if luma_gain <= 1.0e-6:
        return gains
    return gains[0] / luma_gain, gains[1] / luma_gain, gains[2] / luma_gain


def build_light_match_tables(params: LightMatchParams | dict | None = None) -> LightMatchTables:
    p = normalize_light_match_params(params)
    identity = (
        not p.enabled
        or (
            p.temp_k == NEUTRAL_TEMP_K
            and abs(p.tint) < 1.0e-6
            and abs(p.exposure_ev) < 1.0e-6
            and abs(p.contrast - 1.0) < 1.0e-6
            and abs(p.gamma - 1.0) < 1.0e-6
            and abs(p.saturation - 1.0) < 1.0e-6
        )
    )
    if identity:
        coeffs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return LightMatchTables(coeffs=coeffs, gamma_lut=np.arange(256, dtype=np.uint8), identity=True)

    chroma_gains = _luma_preserving_gains(_rgb_gains(p.temp_k, p.tint, 0.0))
    exposure = 2.0 ** p.exposure_ev
    _y0, u0, v0 = _apply_rgb_gains_to_yuv(128.0, 128.0, 128.0, chroma_gains)
    _, u_u, v_u = _apply_rgb_gains_to_yuv(128.0, 129.0, 128.0, chroma_gains)
    _, u_v, v_v = _apply_rgb_gains_to_yuv(128.0, 128.0, 129.0, chroma_gains)

    y_gain = exposure * p.contrast
    y_bias = 128.0 * (1.0 - p.contrast)

    uu = (u_u - u0) * p.saturation
    uv = (u_v - u0) * p.saturation
    vu = (v_u - v0) * p.saturation
    vv = (v_v - v0) * p.saturation
    u_bias = u0 - 128.0
    v_bias = v0 - 128.0

    coeffs = np.array([y_gain, y_bias, uu, uv, u_bias, vu, vv, v_bias], dtype=np.float32)
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(x, p.gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return LightMatchTables(coeffs=coeffs, gamma_lut=lut, identity=False)


def apply_light_match_yuv(y: int, u: int, v: int, tables: LightMatchTables) -> tuple[int, int, int]:
    if tables.identity:
        return int(y), int(u), int(v)
    c = tables.coeffs.astype(np.float32, copy=False)
    yf = float(y) * float(c[0]) + float(c[1])
    yi = int(max(16.0, min(235.0, yf)) + 0.5)
    yi = int(tables.gamma_lut[max(0, min(255, yi))])
    uf = (float(u) - 128.0) * float(c[2]) + (float(v) - 128.0) * float(c[3]) + float(c[4]) + 128.0
    vf = (float(u) - 128.0) * float(c[5]) + (float(v) - 128.0) * float(c[6]) + float(c[7]) + 128.0
    return yi, int(max(16.0, min(240.0, uf)) + 0.5), int(max(16.0, min(240.0, vf)) + 0.5)
