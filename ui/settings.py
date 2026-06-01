from __future__ import annotations

import json
import sys
from pathlib import Path

from ui.i18n import system_language
from utils.trt_manifest import TRT_PROVIDER_CHAIN, cache_status

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "runtime_cache" / "ui_settings.json"
SETTINGS_META_PATH = ROOT / "runtime_cache" / "ui_settings_meta.json"


def _setting_value(data: dict, key: str, default):
    value = data.get(key)
    return default if value is None or value == "" else value

LIGHT_MATCH_PRESETS = {
    "home_warm": {"temp_k": 4000, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "daylight": {"temp_k": 6500, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "night_cool": {"temp_k": 8000, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
}

DEFAULTS = {
    "language": system_language(),
    "video_dirs": [str(ROOT / "videos")],
    "mode_green": True,
    "mode_alpha": True,
    "background_color": "00FF00",
    "alpha_stride": 1,
    "quality_speed": "ultrafast",
    "offline_quality_speed": "medium",
    "offline_sam3_prompt": "person",
    "offline_single_time_segments": [],
    "offline_single_trt_rvm_enabled": True,
    "offline_single_trt_matanyone2_enabled": True,
    "offline_batch_trt_rvm_enabled": True,
    "offline_batch_trt_matanyone2_enabled": True,
    "passthrough_max_fps": 30,
    "passthrough_seek_enabled": False,
    "passthrough_seek_dlna": False,
    "passthrough_seek_route_policy": "profile",
    "passthrough_seek_container": "mpegts",
    "dlna_image_enabled": False,
    "decode_max_side": 4096,
    "inference_backend": "cuda",
    "light_match_enabled": False,
    "light_match_temp_k": 6500,
    "light_match_tint": 0.0,
    "light_match_exposure_ev": 0.0,
    "light_match_contrast": 1.0,
    "light_match_gamma": 1.0,
    "light_match_saturation": 1.0,
    "light_match_preset": "daylight",
    "alpha_2d_projection": "fisheye",
    "alpha_2d_distance_m": 4.0,
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
        self._meta = self._load_meta()
        self.load()

    def _load_meta(self) -> dict:
        if not SETTINGS_META_PATH.exists():
            return {"migrations": []}
        try:
            loaded = json.loads(SETTINGS_META_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            return {"migrations": []}
        if not isinstance(loaded, dict):
            return {"migrations": []}
        migrations = loaded.get("migrations")
        return {"migrations": migrations if isinstance(migrations, list) else []}

    def _migration_done(self, name: str, loaded: dict) -> bool:
        return name in self._meta.get("migrations", []) or bool(loaded.get(f"defaults_migrated_{name}"))

    def _mark_migration_done(self, name: str) -> None:
        migrations = self._meta.setdefault("migrations", [])
        if name not in migrations:
            migrations.append(name)

    def load(self) -> None:
        if SETTINGS_PATH.exists():
            try:
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    self.data.update({k: v for k, v in loaded.items() if not str(k).startswith("defaults_migrated_")})
                    if "quality_speed" in loaded and "offline_quality_speed" not in loaded:
                        self.data["offline_quality_speed"] = quality_speed_value(loaded.get("quality_speed"), "medium")
                        self.data["quality_speed"] = DEFAULTS["quality_speed"]
                    if not self._migration_done("20260517_fps_size", loaded):
                        if int(loaded.get("passthrough_max_fps", 0) or 0) == 0:
                            self.data["passthrough_max_fps"] = DEFAULTS["passthrough_max_fps"]
                        if int(loaded.get("decode_max_side", DEFAULTS["decode_max_side"]) or 0) == 0:
                            self.data["decode_max_side"] = DEFAULTS["decode_max_side"]
                        self._mark_migration_done("20260517_fps_size")
                    elif "passthrough_max_fps" not in loaded:
                        self.data["passthrough_max_fps"] = DEFAULTS["passthrough_max_fps"]
                    if not self._migration_done("20260524_fps_default_30", loaded):
                        if int(loaded.get("passthrough_max_fps", DEFAULTS["passthrough_max_fps"]) or 0) == 0:
                            self.data["passthrough_max_fps"] = DEFAULTS["passthrough_max_fps"]
                        self._mark_migration_done("20260524_fps_default_30")
                    if not self._migration_done("20260519_light_match_off", loaded):
                        self.data["light_match_enabled"] = False
                        self._mark_migration_done("20260519_light_match_off")
                    if not self._migration_done("20260524_light_match_daylight_default", loaded):
                        preset = str(loaded.get("light_match_preset", "custom") or "custom").strip().lower()
                        enabled = bool(loaded.get("light_match_enabled"))
                        if not enabled and preset == "custom":
                            self.data["light_match_preset"] = DEFAULTS["light_match_preset"]
                        self._mark_migration_done("20260524_light_match_daylight_default")
                    if not self._migration_done("20260525_light_match_temps_recalibrated", loaded):
                        preset = str(self.data.get("light_match_preset", "custom") or "custom").strip().lower()
                        values = LIGHT_MATCH_PRESETS.get(preset)
                        if values is not None:
                            for key, value in values.items():
                                self.data[f"light_match_{key}"] = value
                        self._mark_migration_done("20260525_light_match_temps_recalibrated")
            except Exception:
                pass

    def save(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(
            json.dumps({k: v for k, v in self.data.items() if not str(k).startswith("defaults_migrated_")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        SETTINGS_META_PATH.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
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
        seek_route_policy = str(
            self.data.get("passthrough_seek_route_policy") or DEFAULTS["passthrough_seek_route_policy"]
        ).strip().lower()
        if seek_route_policy not in {"profile", "all", "off"}:
            seek_route_policy = DEFAULTS["passthrough_seek_route_policy"]
        seek_container = str(
            self.data.get("passthrough_seek_container") or DEFAULTS["passthrough_seek_container"]
        ).strip().lower()
        if seek_container not in {"mpegts", "mp4"}:
            seek_container = DEFAULTS["passthrough_seek_container"]
        env = {
            "PT_VIDEO_DIR": "|".join(self.video_dirs()),
            "PT_UI_LANGUAGE": str(self.data.get("language") or system_language()),
            "PT_PASSTHROUGH_OUTPUT_MODE": self.passthrough_mode(),
            "PT_COMPOSITE_BG_RGB": str(self.data.get("background_color") or "00FF00"),
            "PT_ALPHA_STRIDE": str(_setting_value(self.data, "alpha_stride", 1)),
            "PT_PASSTHROUGH_MAX_FPS": str(passthrough_max_fps),
            "PT_PASSTHROUGH_PRODUCER_REALTIME_PACING": "1",
            "PT_PASSTHROUGH_SEEK_ENABLED": "1" if self.data.get("passthrough_seek_enabled") else "0",
            "PT_PASSTHROUGH_SEEK_DLNA": "1" if self.data.get("passthrough_seek_dlna") else "0",
            "PT_PASSTHROUGH_SEEK_ROUTE_POLICY": seek_route_policy,
            "PT_PASSTHROUGH_SEEK_CONTAINER": seek_container,
            "PT_DLNA_IMAGE_ENABLED": "1" if self.data.get("dlna_image_enabled") else "0",
            "PT_DECODE_MAX_SIDE": str(_setting_value(self.data, "decode_max_side", 4096)),
            "PT_LIGHT_MATCH_ENABLED": "1" if self.data.get("light_match_enabled") else "0",
            "PT_LIGHT_MATCH_TEMP_K": str(_setting_value(self.data, "light_match_temp_k", DEFAULTS["light_match_temp_k"])),
            "PT_LIGHT_MATCH_TINT": str(_setting_value(self.data, "light_match_tint", 0.0)),
            "PT_LIGHT_MATCH_EXPOSURE_EV": str(_setting_value(self.data, "light_match_exposure_ev", 0.0)),
            "PT_LIGHT_MATCH_CONTRAST": str(_setting_value(self.data, "light_match_contrast", 1.0)),
            "PT_LIGHT_MATCH_GAMMA": str(_setting_value(self.data, "light_match_gamma", 1.0)),
            "PT_LIGHT_MATCH_SATURATION": str(_setting_value(self.data, "light_match_saturation", 1.0)),
            "PT_LIGHT_MATCH_PRESET": str(self.data.get("light_match_preset") or DEFAULTS["light_match_preset"]),
            "PT_ALPHA_2D_PROJECTION": str(self.data.get("alpha_2d_projection") or "fisheye"),
            "PT_ALPHA_2D_DISTANCE_M": str(_setting_value(self.data, "alpha_2d_distance_m", 4.0)),
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
            "PT_ONNX_PROVIDERS": "CUDAExecutionProvider,CPUExecutionProvider",
        }
        env.update(quality_speed_env(self.data.get("quality_speed")))
        color = str(self.data.get("subtitle_color") or "").strip()
        if color:
            env["PT_SUBTITLE_COLOR"] = color
        else:
            env.pop("PT_SUBTITLE_COLOR", None)
        if str(self.data.get("inference_backend") or "cuda").lower() == "tensorrt":
            try:
                if cache_status() == "ready":
                    env["PT_ONNX_PROVIDERS"] = TRT_PROVIDER_CHAIN
            except Exception:
                pass
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
