from __future__ import annotations

import json
import sys
from pathlib import Path

from ui.i18n import system_language

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "runtime_cache" / "ui_settings.json"


def _setting_value(data: dict, key: str, default):
    value = data.get(key)
    return default if value is None or value == "" else value

DEFAULTS = {
    "language": system_language(),
    "video_dirs": [str(ROOT / "videos")],
    "mode_green": True,
    "mode_alpha": True,
    "background_color": "00FF00",
    "alpha_stride": 1,
    "quality_speed": "ultrafast",
    "offline_quality_speed": "medium",
    "passthrough_max_fps": 30,
    "decode_max_side": 4096,
    "subtitle_enable": True,
    "subtitle_mode": "auto",
    "subtitle_direction": "horizontal_bottom",
    "subtitle_distance_m": 4.0,
    "subtitle_fov": 60.0,
    "subtitle_yaw": 0.0,
    "subtitle_pitch": 0.0,
    "subtitle_font_scale": 0.045,
    "subtitle_outline_scale": 0.08,
    "subtitle_margin_v_scale": 0.08,
    "subtitle_alpha": 1.0,
    "subtitle_color": "",
    "subtitle_outline_color": "000000",
    "subtitle_v360": True,
    "defaults_migrated_20260517_fps_size": True,
}


QUALITY_SPEED_PRESETS = {
    "ultrafast": "P1",
    "medium": "P4",
    "veryslow": "P7",
}


def quality_speed_value(value, default: str | None = None) -> str:
    fallback = str(default or DEFAULTS["quality_speed"])
    key = str(value or fallback).strip().lower()
    return key if key in QUALITY_SPEED_PRESETS else fallback


def quality_speed_preset(value, default: str | None = None) -> str:
    return QUALITY_SPEED_PRESETS[quality_speed_value(value, default)]


def quality_speed_env(value) -> dict[str, str]:
    return {
        "PT_PASSTHROUGH_PYNV_PRESET": quality_speed_preset(value),
        "PT_PASSTHROUGH_PYNV_DECODER": "simple",
        "PT_PASSTHROUGH_PYNV_THREADED_BATCH_SIZE": "1",
        "PT_PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE": "2",
    }


class Settings:
    def __init__(self) -> None:
        self.data = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        if SETTINGS_PATH.exists():
            try:
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
                    if "quality_speed" in loaded and "offline_quality_speed" not in loaded:
                        self.data["offline_quality_speed"] = quality_speed_value(loaded.get("quality_speed"), "medium")
                        self.data["quality_speed"] = DEFAULTS["quality_speed"]
                    if not loaded.get("defaults_migrated_20260517_fps_size"):
                        if int(loaded.get("passthrough_max_fps", 0) or 0) == 0:
                            self.data["passthrough_max_fps"] = DEFAULTS["passthrough_max_fps"]
                        if int(loaded.get("decode_max_side", DEFAULTS["decode_max_side"]) or 0) == 0:
                            self.data["decode_max_side"] = DEFAULTS["decode_max_side"]
                        self.data["defaults_migrated_20260517_fps_size"] = True
                    elif "passthrough_max_fps" not in loaded:
                        self.data["passthrough_max_fps"] = DEFAULTS["passthrough_max_fps"]
            except Exception:
                pass

    def save(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def passthrough_mode(self) -> str:
        green = bool(self.data.get("mode_green"))
        alpha = bool(self.data.get("mode_alpha"))
        if green and alpha:
            return "all"
        if green:
            return "green"
        if alpha:
            return "alpha"
        return "none"

    def server_env(self) -> dict[str, str]:
        passthrough_max_fps = _setting_value(self.data, "passthrough_max_fps", 0)
        env = {
            "PT_VIDEO_DIR": "|".join(self.video_dirs()),
            "PT_UI_LANGUAGE": str(self.data.get("language") or system_language()),
            "PT_PASSTHROUGH_OUTPUT_MODE": self.passthrough_mode(),
            "PT_COMPOSITE_BG_RGB": str(self.data.get("background_color") or "00FF00"),
            "PT_ALPHA_STRIDE": str(_setting_value(self.data, "alpha_stride", 1)),
            "PT_PASSTHROUGH_MAX_FPS": str(passthrough_max_fps),
            "PT_PASSTHROUGH_PRODUCER_REALTIME_PACING": "1" if float(passthrough_max_fps or 0) > 0 else "0",
            "PT_DECODE_MAX_SIDE": str(_setting_value(self.data, "decode_max_side", 4096)),
            "PT_SUBTITLE_ENABLE": "1" if self.data.get("subtitle_enable") else "0",
            "PT_SUBTITLE_MODE": str(self.data.get("subtitle_mode") or "auto"),
            "PT_SUBTITLE_DIRECTION": str(self.data.get("subtitle_direction") or "horizontal_bottom"),
            "PT_SUBTITLE_DISTANCE_M": str(self.data.get("subtitle_distance_m") or 4.0),
            "PT_SUBTITLE_FOV": str(self.data.get("subtitle_fov") or 60.0),
            "PT_SUBTITLE_YAW": str(self.data.get("subtitle_yaw") or 0.0),
            "PT_SUBTITLE_PITCH": str(self.data.get("subtitle_pitch") or 0.0),
            "PT_SUBTITLE_FONT_SCALE": str(self.data.get("subtitle_font_scale") or 0.045),
            "PT_SUBTITLE_OUTLINE_SCALE": str(self.data.get("subtitle_outline_scale") or 0.08),
            "PT_SUBTITLE_MARGIN_V_SCALE": str(self.data.get("subtitle_margin_v_scale") or 0.08),
            "PT_SUBTITLE_ALPHA": str(self.data.get("subtitle_alpha") or 1.0),
            "PT_SUBTITLE_OUTLINE_COLOR": str(self.data.get("subtitle_outline_color") or "000000"),
            "PT_SUBTITLE_V360": "1" if self.data.get("subtitle_v360") else "0",
        }
        env.update(quality_speed_env(self.data.get("quality_speed")))
        color = str(self.data.get("subtitle_color") or "").strip()
        if color:
            env["PT_SUBTITLE_COLOR"] = color
        else:
            env.pop("PT_SUBTITLE_COLOR", None)
        return env

    def video_dirs(self) -> list[str]:
        raw = self.data.get("video_dirs")
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
        elif isinstance(raw, str):
            values = [item.strip() for item in raw.split("|") if item.strip()]
        else:
            values = []
        return values or [str(ROOT / "videos")]

    def set_video_dirs(self, directories: list[str]) -> None:
        values = [str(Path(directory).expanduser()) for directory in directories if str(directory).strip()]
        self.data["video_dirs"] = values or [str(ROOT / "videos")]

    def restore_default_subtitle_style(self) -> None:
        for key in (
            "subtitle_mode",
            "subtitle_direction",
            "subtitle_distance_m",
            "subtitle_fov",
            "subtitle_yaw",
            "subtitle_pitch",
            "subtitle_font_scale",
            "subtitle_outline_scale",
            "subtitle_margin_v_scale",
            "subtitle_alpha",
            "subtitle_color",
            "subtitle_outline_color",
            "subtitle_v360",
        ):
            self.data[key] = DEFAULTS[key]
