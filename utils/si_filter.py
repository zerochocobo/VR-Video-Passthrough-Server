from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SI_MIX_CHANNELS = ("left", "right", "both")
ORIGINAL_VOLUME_CHOICES = (70, 80, 90, 100)
SI_VOLUME_CHOICES = (50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150)
SI_DELAY_SECONDS_CHOICES = (0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0)
DEFAULT_SI_MIX_ENABLED = True
DEFAULT_SI_MIX_CHANNEL = "both"
DEFAULT_ORIGINAL_VOLUME_PERCENT = 100
DEFAULT_SI_VOLUME_PERCENT = 100
DEFAULT_SI_DELAY_SECONDS = 0.0
DEFAULT_DUCK_ORIGINAL = True
DEFAULT_DUCK_PRESET = "normal"
DEFAULT_DUB_MODE = True
SI_DUCK_PRESET_CHOICES = ("light", "normal", "strong")
SI_DUCK_PRESETS = {
    "light": {"threshold": "0.03", "ratio": "2.5", "release": "400"},
    "normal": {"threshold": "0.025", "ratio": "5", "release": "600"},
    "strong": {"threshold": "0.015", "ratio": "10", "release": "800"},
}
SI_DUCK_ATTACK_MS = "30"
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
    duck_preset: str = DEFAULT_DUCK_PRESET
    dub_mode_enabled: bool = DEFAULT_DUB_MODE

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
        object.__setattr__(
            self,
            "duck_preset",
            _coerce_choice(str(self.duck_preset).strip().lower(), SI_DUCK_PRESET_CHOICES, DEFAULT_DUCK_PRESET),
        )
        object.__setattr__(self, "dub_mode_enabled", _coerce_bool(self.dub_mode_enabled, DEFAULT_DUB_MODE))

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
            duck_preset=source.get(
                "duck_preset",
                source.get("si_duck_preset", source.get("dlna_si_duck_preset", DEFAULT_DUCK_PRESET)),
            ),
            dub_mode_enabled=source.get(
                "dub_mode_enabled",
                source.get("si_dub_mode", source.get("dlna_si_dub_mode", DEFAULT_DUB_MODE)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dubbing_variant(self) -> "SIMixParams":
        """Config used when a ``.si.duck.wav`` key drives dubbing playback. This
        is fully fixed and ignores the user's SI settings: original at 100%, dub
        voice at 120%, no delay, overlaid on both channels, with the original
        ducked strongly across every subtitle span via the duck key."""
        return SIMixParams(
            enabled=self.enabled,
            mix_channel="both",
            original_volume_percent=100,
            si_volume_percent=120,
            si_delay_seconds=0.0,
            duck_original=True,
            duck_preset="strong",
            dub_mode_enabled=self.dub_mode_enabled,
        )

    def filter_string(self, duck_key_input: bool = False) -> str:
        return build_si_mix_filter(
            self.mix_channel,
            self.original_volume_percent,
            self.si_volume_percent,
            self.si_delay_seconds,
            self.duck_original,
            self.duck_preset,
            duck_key_input=bool(duck_key_input) and self.duck_original,
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
        raise ValueError("SI volume percent must be one of 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150.")
    return value


def _validate_si_delay_seconds(seconds: int | float) -> float:
    try:
        value = round(float(seconds), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SI delay seconds: {seconds}") from exc
    if value not in SI_DELAY_SECONDS_CHOICES:
        raise ValueError("SI delay must be one of 0, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 2 seconds.")
    return value


def _validate_duck_preset(preset: str) -> str:
    normalized = str(preset or "").strip().lower()
    if normalized not in SI_DUCK_PRESET_CHOICES:
        raise ValueError(f"Unsupported SI duck preset: {preset}")
    return normalized


def _filter_number(value: int | float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _sidechain_compressor(duck_preset: str = DEFAULT_DUCK_PRESET) -> str:
    preset = SI_DUCK_PRESETS[_validate_duck_preset(duck_preset)]
    return (
        f"threshold={preset['threshold']}:"
        f"ratio={preset['ratio']}:"
        f"attack={SI_DUCK_ATTACK_MS}:"
        f"release={preset['release']}:"
        f"makeup={SI_DUCK_MAKEUP}"
    )


def build_si_mix_filter(
    mix_channel: str,
    original_volume_percent: int | float,
    si_volume_percent: int | float,
    si_delay_seconds: int | float = DEFAULT_SI_DELAY_SECONDS,
    duck_original: bool = False,
    duck_preset: str = DEFAULT_DUCK_PRESET,
    duck_key_input: bool = False,
) -> str:
    channel = _validate_si_mix_channel(mix_channel)
    original_volume = _filter_number(_validate_original_volume(original_volume_percent) / 100.0)
    si_volume = _filter_number(_validate_si_volume(si_volume_percent) / 100.0)
    si_delay_ms = int(round(_validate_si_delay_seconds(si_delay_seconds) * 1000))
    duck_preset = _validate_duck_preset(duck_preset)

    if channel == "both":
        if duck_original:
            compressor = _sidechain_compressor(duck_preset)
            if duck_key_input:
                # The sidechain key comes from a dedicated ``.si.duck.wav`` (input
                # [2]) that marks the original-subtitle spans, so the original is
                # ducked across those spans instead of only where the dub voice is.
                return (
                    "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"volume={original_volume}[orig_base];"
                    "[2:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,apad[si_key];"
                    "[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,"
                    f"adelay={si_delay_ms},volume={si_volume},apad[si_mono];"
                    f"[orig_base][si_key]sidechaincompress={compressor}[orig];"
                    "[si_mono]aformat=channel_layouts=stereo[si];"
                    "[orig][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                    "alimiter=limit=0.95[si_track]"
                )
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
        compressor = _sidechain_compressor(duck_preset)
        if duck_key_input:
            return (
                "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={original_volume}[orig_base];"
                "[2:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,apad[si_key];"
                "[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"adelay={si_delay_ms},volume={si_volume},apad[si];"
                f"[orig_base][si_key]sidechaincompress={compressor}[orig];"
                "[orig]channelsplit=channel_layout=stereo[ol][or];"
                f"{mix_part}"
            )
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
