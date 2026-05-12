from __future__ import annotations

import json
import sys
from pathlib import Path

from ui.i18n import system_language

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "runtime_cache" / "ui_settings.json"

DEFAULTS = {
    "language": system_language(),
    "video_dirs": [str(ROOT / "videos")],
    "mode_green": True,
    "mode_alpha": False,
    "background_color": "808080",
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
        env = {
            "PT_VIDEO_DIR": "|".join(self.video_dirs()),
            "PT_PASSTHROUGH_OUTPUT_MODE": self.passthrough_mode(),
            "PT_COMPOSITE_BG_RGB": str(self.data.get("background_color") or "808080"),
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
