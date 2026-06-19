from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SI_MIX_CHANNELS = ("left", "right", "both")
ORIGINAL_VOLUME_CHOICES = (70, 80, 90, 100)
SI_VOLUME_CHOICES = (50, 60, 70, 80, 90, 100)
SI_DELAY_SECONDS_CHOICES = (0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0)
DEFAULT_SI_MIX_ENABLED = True
DEFAULT_SI_MIX_CHANNEL = "both"
DEFAULT_ORIGINAL_VOLUME_PERCENT = 100
DEFAULT_SI_VOLUME_PERCENT = 100
DEFAULT_SI_DELAY_SECONDS = 0.0
DEFAULT_DUCK_ORIGINAL = True
SI_DUCK_THRESHOLD = "0.025"
SI_DUCK_RATIO = "5"
SI_DUCK_ATTACK_MS = "30"
SI_DUCK_RELEASE_MS = "600"
SI_DUCK_MAKEUP = "1"


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_choice(value: object, choices: tuple[Any, ...], default: Any) -> Any:
    if value in choices:
        return value
    for choice in choices:
        try:
            if type(choice)(value) == choice:
                return choice
        except (TypeError, ValueError):
            continue
    return default


@dataclass(frozen=True)
class SIMixParams:
    enabled: bool = DEFAULT_SI_MIX_ENABLED
    mix_channel: str = DEFAULT_SI_MIX_CHANNEL
    original_volume_percent: int = DEFAULT_ORIGINAL_VOLUME_PERCENT
    si_volume_percent: int = DEFAULT_SI_VOLUME_PERCENT
    si_delay_seconds: float = DEFAULT_SI_DELAY_SECONDS
    duck_original: bool = DEFAULT_DUCK_ORIGINAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _coerce_bool(self.enabled, DEFAULT_SI_MIX_ENABLED))
        object.__setattr__(
            self,
            "mix_channel",
            _coerce_choice(str(self.mix_channel).strip().lower(), SI_MIX_CHANNELS, DEFAULT_SI_MIX_CHANNEL),
        )
        object.__setattr__(
            self,
            "original_volume_percent",
            _coerce_choice(
                self.original_volume_percent,
                ORIGINAL_VOLUME_CHOICES,
                DEFAULT_ORIGINAL_VOLUME_PERCENT,
            ),
        )
        object.__setattr__(
            self,
            "si_volume_percent",
            _coerce_choice(self.si_volume_percent, SI_VOLUME_CHOICES, DEFAULT_SI_VOLUME_PERCENT),
        )
        try:
            delay = round(float(self.si_delay_seconds), 1)
        except (TypeError, ValueError):
            delay = DEFAULT_SI_DELAY_SECONDS
        object.__setattr__(
            self,
            "si_delay_seconds",
            _coerce_choice(delay, SI_DELAY_SECONDS_CHOICES, DEFAULT_SI_DELAY_SECONDS),
        )
        object.__setattr__(self, "duck_original", _coerce_bool(self.duck_original, DEFAULT_DUCK_ORIGINAL))

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SIMixParams":
        source = data or {}
        return cls(
            enabled=source.get(
                "enabled",
                source.get("si_enabled", source.get("dlna_si_enabled", DEFAULT_SI_MIX_ENABLED)),
            ),
            mix_channel=source.get(
                "mix_channel",
                source.get("si_mix_channel", source.get("dlna_si_mix_channel", DEFAULT_SI_MIX_CHANNEL)),
            ),
            original_volume_percent=source.get(
                "original_volume_percent",
                source.get(
                    "si_original_volume_percent",
                    source.get("dlna_si_original_volume_percent", DEFAULT_ORIGINAL_VOLUME_PERCENT),
                ),
            ),
            si_volume_percent=source.get(
                "si_volume_percent",
                source.get("dlna_si_volume_percent", DEFAULT_SI_VOLUME_PERCENT),
            ),
            si_delay_seconds=source.get(
                "si_delay_seconds",
                source.get("dlna_si_delay_seconds", DEFAULT_SI_DELAY_SECONDS),
            ),
            duck_original=source.get(
                "duck_original",
                source.get("si_duck_original", source.get("dlna_si_duck_original", DEFAULT_DUCK_ORIGINAL)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def filter_string(self) -> str:
        return build_si_mix_filter(
            self.mix_channel,
            self.original_volume_percent,
            self.si_volume_percent,
            self.si_delay_seconds,
            self.duck_original,
        )


def normalize_si_mix_params(data: dict[str, Any] | SIMixParams) -> SIMixParams:
    if isinstance(data, SIMixParams):
        return SIMixParams(**data.to_dict())
    return SIMixParams.from_mapping(data)


def _validate_si_mix_channel(channel: str) -> str:
    normalized = (channel or "").strip().lower()
    if normalized not in SI_MIX_CHANNELS:
        raise ValueError(f"Unsupported SI mix channel: {channel}")
    return normalized


def _validate_original_volume(percent: int | float) -> int:
    try:
        value = int(percent)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid volume percent: {percent}") from exc
    if value not in ORIGINAL_VOLUME_CHOICES:
        raise ValueError("Original volume percent must be one of 70, 80, 90, 100.")
    return value


def _validate_si_volume(percent: int | float) -> int:
    try:
        value = int(percent)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SI volume percent: {percent}") from exc
    if value not in SI_VOLUME_CHOICES:
        raise ValueError("SI volume percent must be one of 50, 60, 70, 80, 90, 100.")
    return value


def _validate_si_delay_seconds(seconds: int | float) -> float:
    try:
        value = round(float(seconds), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SI delay seconds: {seconds}") from exc
    if value not in SI_DELAY_SECONDS_CHOICES:
        raise ValueError("SI delay must be one of 0, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 2 seconds.")
    return value


def _filter_number(value: int | float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _sidechain_compressor() -> str:
    return (
        f"threshold={SI_DUCK_THRESHOLD}:"
        f"ratio={SI_DUCK_RATIO}:"
        f"attack={SI_DUCK_ATTACK_MS}:"
        f"release={SI_DUCK_RELEASE_MS}:"
        f"makeup={SI_DUCK_MAKEUP}"
    )


def build_si_mix_filter(
    mix_channel: str,
    original_volume_percent: int | float,
    si_volume_percent: int | float,
    si_delay_seconds: int | float = DEFAULT_SI_DELAY_SECONDS,
    duck_original: bool = False,
) -> str:
    channel = _validate_si_mix_channel(mix_channel)
    original_volume = _filter_number(_validate_original_volume(original_volume_percent) / 100.0)
    si_volume = _filter_number(_validate_si_volume(si_volume_percent) / 100.0)
    si_delay_ms = int(round(_validate_si_delay_seconds(si_delay_seconds) * 1000))

    if channel == "both":
        if duck_original:
            compressor = _sidechain_compressor()
            return (
                "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={original_volume}[orig_base];"
                "[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"adelay={si_delay_ms},volume={si_volume},apad,asplit=2[si_key][si_mono];"
                f"[orig_base][si_key]sidechaincompress={compressor}[orig];"
                "[si_mono]aformat=channel_layouts=stereo[si];"
                "[orig][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                "alimiter=limit=0.95[si_track]"
            )
        return (
            f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume={original_volume}[orig];"
            f"[1:a:0]aresample=48000,aformat=channel_layouts=mono,adelay={si_delay_ms},"
            f"volume={si_volume},aformat=channel_layouts=stereo[si];"
            "[orig][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95[si_track]"
        )

    if channel == "left":
        mix_part = (
            "[ol][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[left_mix_raw];"
            "[left_mix_raw][or]join=inputs=2:channel_layout=stereo,"
            "alimiter=limit=0.95[si_track]"
        )
    else:
        mix_part = (
            "[or][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[right_mix_raw];"
            "[ol][right_mix_raw]join=inputs=2:channel_layout=stereo,"
            "alimiter=limit=0.95[si_track]"
        )

    if duck_original:
        compressor = _sidechain_compressor()
        return (
            "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={original_volume}[orig_base];"
            "[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={si_delay_ms},volume={si_volume},apad,asplit=2[si_key][si];"
            f"[orig_base][si_key]sidechaincompress={compressor}[orig];"
            "[orig]channelsplit=channel_layout=stereo[ol][or];"
            f"{mix_part}"
        )

    return (
        f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume={original_volume}[orig];"
        f"[1:a:0]aresample=48000,aformat=channel_layouts=mono,adelay={si_delay_ms},volume={si_volume}[si];"
        "[orig]channelsplit=channel_layout=stereo[ol][or];"
        f"{mix_part}"
    )
