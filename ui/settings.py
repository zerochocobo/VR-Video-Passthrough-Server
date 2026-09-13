from __future__ import annotations

import json
import sys
from pathlib import Path

from media_library import is_unc_path
from ui.i18n import system_language
from utils.si_filter import (
    DEFAULT_DUB_MODE,
    DEFAULT_DUCK_ORIGINAL,
    DEFAULT_DUCK_PRESET,
    DEFAULT_ORIGINAL_VOLUME_PERCENT,
    DEFAULT_SI_DELAY_SECONDS,
    DEFAULT_SI_MIX_CHANNEL,
    DEFAULT_SI_MIX_ENABLED,
    DEFAULT_SI_VOLUME_PERCENT,
    SI_DUCK_PRESET_CHOICES,
)
from utils.trt_manifest import TRT_PROVIDER_CHAIN

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "runtime_cache" / "ui_settings.json"
SETTINGS_META_PATH = ROOT / "runtime_cache" / "ui_settings_meta.json"


def _setting_value(data: dict, key: str, default):
    value = data.get(key)
    return default if value is None or value == "" else value


def _two_dvr_strength_value(value, default: float = 1.0) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError):
        strength = default
    return max(0.1, min(3.0, strength))

LIGHT_MATCH_PRESETS = {
    "home_warm": {"temp_k": 4000, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "daylight": {"temp_k": 6500, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "night_cool": {"temp_k": 8000, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
}

DEFAULT_HTTP_PORT = 8200
DEFAULT_SERVER_NAME = "VR Passthrough Server"  # config.py fallback when PT_SERVER_NAME is unset.

DEFAULTS = {
    "language": system_language(),
    "video_dirs": [str(ROOT / "videos")],
    "server_name": "",
    "http_port": DEFAULT_HTTP_PORT,
    "mode_green": True,
    "mode_alpha": True,
    "mode_2d": True,
    "mode_two_dvr": True,
    "mode_superres": False,
    # Native 1x: the target that can also be played as a virtual file.
    "superres_target_height": 0,
    "superres_quality": 4,
    "superres_hdr_look": "natural",
    "superres_offline_hdr_look": "natural",
    "superres_offline_bitrate_mode": "auto",
    # NGX TrueHDR controls; ranges come from the RTX Video SDK headers.
    "superres_truehdr_contrast": 100,
    "superres_truehdr_saturation": 100,
    "superres_truehdr_middle_gray": 50,
    "superres_truehdr_max_nits": 1000,
    "mode_dlss5": False,
    # DLSS5 is parked: the pipeline, the offline tool and the probe all stay,
    # but on this library NR did not beat RTX VSR 1x, so nothing of it is shown.
    # While this is False the dashboard card, the offline tools card and the
    # [DLSS5] DLNA entries are all absent, whatever mode_dlss5 says. There is
    # deliberately no switch in the settings page; set it in ui_settings.json.
    "dlss5_card_visible": False,
    # Neural Rendering controls. The neutral value of each is what the runtime
    # treats as "no adjustment", which is 1.0 for most of them and -1.0 for skin
    # structure; see config.DLSS5_* and utils.dlss5.DLSS5_RANGES.
    "dlss5_style": 0,
    "dlss5_intensity": 1.0,
    "dlss5_nr_passes": 1,
    "dlss5_shimmer_suppression": 0.7,
    "dlss5_local_tone": 1.0,
    "dlss5_local_structure": 1.0,
    "dlss5_skin_structure": -1.0,
    "dlss5_color_strength": 1.0,
    "dlss5_tone_preservation": 0.0,
    "dlss5_face_skin_protection": 0.0,
    "dlss5_grain_preservation": 0.0,
    "dlss5_auto_mask": False,
    "two_dvr_live_model": "base",
    "two_dvr_live_hole_fill": "soft_shift",
    "two_dvr_live_eye_distance": 65.0,
    "two_dvr_live_strength": 1.0,
    "background_color": "00FF00",
    "alpha_stride": 1,
    "quality_speed": "ultrafast",
    "offline_quality_speed": "medium",
    "two_dvr_depth_stabilizer": "default",
    "offline_sam3_prompt": "person",
    "offline_single_time_segments": [],
    "offline_single_trt_rvm_enabled": True,
    "offline_single_trt_matanyone2_enabled": True,
    "offline_batch_trt_rvm_enabled": True,
    "offline_batch_trt_matanyone2_enabled": True,
    "passthrough_max_fps": 30,
    # How a passthrough mode is offered to the player. "virtual" advertises the
    # draggable virtual-MP4 entry (seek), "live" the chapter/live entry. One or
    # the other, never both - see PT_PASSTHROUGH_SEEK_DLNA.
    "passthrough_playback_mode": "virtual",
    "passthrough_seek_enabled": False,
    "passthrough_seek_dlna": False,
    "passthrough_seek_route_policy": "profile",
    "passthrough_seek_container": "mp4",
    "passthrough_seek_vmp4": True,
    "passthrough_seek_vmp4_backend": "slot_frames",
    "passthrough_seek_vmp4_slot_ready_only": True,
    "passthrough_seek_vmp4_build_missing": True,
    "passthrough_seek_vmp4_build_max_active": 1,
    "dlna_image_enabled": False,
    "dlna_all_videos_enabled": False,
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
    "si_enabled": DEFAULT_SI_MIX_ENABLED,
    "si_mix_channel": DEFAULT_SI_MIX_CHANNEL,
    "si_original_volume_percent": DEFAULT_ORIGINAL_VOLUME_PERCENT,
    "si_volume_percent": DEFAULT_SI_VOLUME_PERCENT,
    "si_delay_seconds": DEFAULT_SI_DELAY_SECONDS,
    "si_duck_original": DEFAULT_DUCK_ORIGINAL,
    "si_duck_preset": DEFAULT_DUCK_PRESET,
    "si_dub_mode": DEFAULT_DUB_MODE,
    "rm_enabled": False,
    "face_beauty_enabled": False,
    "face_beauty_card_visible": False,
    # Realtime strength knobs; empty means "use the offline standard preset".
    "face_beauty_live": {},
    "rm_vr2flat_decode": True,
    "rm_card_visible": False,
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
        if not SETTINGS_PATH.exists():
            return
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
                    if not self._migration_done("20260616_two_dvr_strength", loaded):
                        if "two_dvr_live_strength" not in loaded:
                            try:
                                old_eye = float(loaded.get("two_dvr_live_eye_distance", DEFAULTS["two_dvr_live_eye_distance"]))
                            except (TypeError, ValueError):
                                old_eye = DEFAULTS["two_dvr_live_eye_distance"]
                            self.data["two_dvr_live_strength"] = _two_dvr_strength_value(
                                old_eye / DEFAULTS["two_dvr_live_eye_distance"]
                            )
                        self.data["two_dvr_live_model"] = DEFAULTS["two_dvr_live_model"]
                        self.data["two_dvr_live_hole_fill"] = DEFAULTS["two_dvr_live_hole_fill"]
                        self.data["two_dvr_live_eye_distance"] = DEFAULTS["two_dvr_live_eye_distance"]
                        self._mark_migration_done("20260616_two_dvr_strength")
                    if not self._migration_done("20260620_seek_vmp4_frames_default", loaded):
                        # Move the superseded one-fps "slot" backend onto the
                        # real-fps "slot_frames" backend; leave explicit
                        # cache_file choices untouched.
                        if str(self.data.get("passthrough_seek_vmp4_backend") or "").lower() == "slot":
                            self.data["passthrough_seek_vmp4_backend"] = "slot_frames"
                        self._mark_migration_done("20260620_seek_vmp4_frames_default")
                    if not self._migration_done("20260620_seek_dlna_default_off", loaded):
                        self.data["passthrough_seek_dlna"] = False
                        self._mark_migration_done("20260620_seek_dlna_default_off")
                    if not self._migration_done("20260910_playback_mode", loaded):
                        # The virtual-file entry is the default way a passthrough
                        # mode is offered now; the dialog on each mode's card
                        # switches an install back to the live entry.
                        self.data["passthrough_playback_mode"] = DEFAULTS["passthrough_playback_mode"]
                        self._mark_migration_done("20260910_playback_mode")
                    if not self._migration_done("20260720_superres_adaptive_8k_default", loaded):
                        if int(loaded.get("superres_target_height", 2160) or 2160) == 2160:
                            self.data["superres_target_height"] = 4096
                        self._mark_migration_done("20260720_superres_adaptive_8k_default")
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

    def dlss5_enabled(self) -> bool:
        """mode_dlss5, unless DLSS5 is hidden - a switch left on from before
        must not keep a [DLSS5] entry in the player's list."""
        return bool(self.data.get("dlss5_card_visible")) and bool(self.data.get("mode_dlss5"))

    def passthrough_mode(self) -> str:
        modes: list[str] = []
        if bool(self.data.get("mode_green")):
            modes.append("green")
        if bool(self.data.get("mode_alpha")):
            modes.append("alpha")
        if bool(self.data.get("mode_two_dvr")):
            modes.append("two_dvr")
        if bool(self.data.get("mode_superres")):
            modes.append("superres")
        if self.dlss5_enabled():
            modes.append("dlss5")
        if modes == ["green", "alpha"]:
            return "all"
        return ",".join(modes) if modes else "none"

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
        si_duck_preset = str(self.data.get("si_duck_preset") or DEFAULTS["si_duck_preset"]).strip().lower()
        if si_duck_preset not in SI_DUCK_PRESET_CHOICES:
            si_duck_preset = DEFAULTS["si_duck_preset"]
        seek_vmp4_backend = str(
            self.data.get("passthrough_seek_vmp4_backend") or DEFAULTS["passthrough_seek_vmp4_backend"]
        ).strip().lower().replace("-", "_")
        if seek_vmp4_backend not in {"cache_file", "slot", "slot_frames"}:
            seek_vmp4_backend = DEFAULTS["passthrough_seek_vmp4_backend"]
        playback_virtual = str(
            self.data.get("passthrough_playback_mode") or DEFAULTS["passthrough_playback_mode"]
        ).strip().lower() != "live"
        env = {
            "PT_VIDEO_DIR": "|".join(self.video_dirs()),
            "PT_HTTP_PORT": str(self.http_port()),
            "PT_UI_LANGUAGE": str(self.data.get("language") or system_language()),
            "PT_PASSTHROUGH_OUTPUT_MODE": self.passthrough_mode(),
            "PT_RTX_VSR_REALTIME_ENABLE": "1" if self.data.get("mode_superres") else "0",
            "PT_RTX_VSR_TARGET_HEIGHT": str(_setting_value(self.data, "superres_target_height", DEFAULTS["superres_target_height"])),
            "PT_RTX_VSR_QUALITY": str(_setting_value(self.data, "superres_quality", DEFAULTS["superres_quality"])),
            "PT_RTX_VSR_HDR_LOOK": str(self.data.get("superres_hdr_look") or DEFAULTS["superres_hdr_look"]),
            "PT_DLSS5_REALTIME_ENABLE": "1" if self.dlss5_enabled() else "0",
            "PT_DLSS5_STYLE": str(_setting_value(self.data, "dlss5_style", DEFAULTS["dlss5_style"])),
            "PT_DLSS5_INTENSITY": str(_setting_value(self.data, "dlss5_intensity", DEFAULTS["dlss5_intensity"])),
            "PT_DLSS5_NR_PASSES": str(_setting_value(self.data, "dlss5_nr_passes", DEFAULTS["dlss5_nr_passes"])),
            "PT_DLSS5_SHIMMER_SUPPRESSION": str(
                _setting_value(self.data, "dlss5_shimmer_suppression", DEFAULTS["dlss5_shimmer_suppression"])
            ),
            "PT_DLSS5_LOCAL_TONE": str(_setting_value(self.data, "dlss5_local_tone", DEFAULTS["dlss5_local_tone"])),
            "PT_DLSS5_LOCAL_STRUCTURE": str(
                _setting_value(self.data, "dlss5_local_structure", DEFAULTS["dlss5_local_structure"])
            ),
            "PT_DLSS5_SKIN_STRUCTURE": str(
                _setting_value(self.data, "dlss5_skin_structure", DEFAULTS["dlss5_skin_structure"])
            ),
            "PT_DLSS5_COLOR_STRENGTH": str(
                _setting_value(self.data, "dlss5_color_strength", DEFAULTS["dlss5_color_strength"])
            ),
            "PT_DLSS5_TONE_PRESERVATION": str(
                _setting_value(self.data, "dlss5_tone_preservation", DEFAULTS["dlss5_tone_preservation"])
            ),
            "PT_DLSS5_FACE_SKIN_PROTECTION": str(
                _setting_value(self.data, "dlss5_face_skin_protection", DEFAULTS["dlss5_face_skin_protection"])
            ),
            "PT_DLSS5_GRAIN_PRESERVATION": str(
                _setting_value(self.data, "dlss5_grain_preservation", DEFAULTS["dlss5_grain_preservation"])
            ),
            "PT_DLSS5_AUTO_MASK": "1" if self.data.get("dlss5_auto_mask") else "0",
            "PT_COMPOSITE_BG_RGB": str(self.data.get("background_color") or "00FF00"),
            "PT_ALPHA_STRIDE": str(_setting_value(self.data, "alpha_stride", 1)),
            "PT_PASSTHROUGH_MAX_FPS": str(passthrough_max_fps),
            "PT_PASSTHROUGH_PRODUCER_REALTIME_PACING": "1",
            # Both come from one choice: the route and the advertisement have to
            # agree, and a user who picks "virtual file" in the dialog means both.
            "PT_PASSTHROUGH_SEEK_ENABLED": "1" if playback_virtual else "0",
            "PT_PASSTHROUGH_SEEK_DLNA": "1" if playback_virtual else "0",
            "PT_PASSTHROUGH_SEEK_ROUTE_POLICY": seek_route_policy,
            "PT_PASSTHROUGH_SEEK_CONTAINER": seek_container,
            "PT_PASSTHROUGH_SEEK_VMP4": "1" if self.data.get("passthrough_seek_vmp4") else "0",
            "PT_PASSTHROUGH_SEEK_VMP4_BACKEND": seek_vmp4_backend,
            "PT_PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY": (
                "1" if self.data.get("passthrough_seek_vmp4_slot_ready_only", True) else "0"
            ),
            "PT_PASSTHROUGH_SEEK_VMP4_BUILD_MISSING": "1" if self.data.get("passthrough_seek_vmp4_build_missing") else "0",
            "PT_PASSTHROUGH_SEEK_VMP4_BUILD_MAX_ACTIVE": str(
                max(1, int(_setting_value(self.data, "passthrough_seek_vmp4_build_max_active", 1) or 1))
            ),
            "PT_DLNA_IMAGE_ENABLED": "1" if self.data.get("dlna_image_enabled") else "0",
            "PT_DLNA_ALL_VIDEOS_ENABLED": "1" if self.data.get("dlna_all_videos_enabled") else "0",
            "PT_DECODE_MAX_SIDE": str(_setting_value(self.data, "decode_max_side", 4096)),
            "PT_LIGHT_MATCH_ENABLED": "1" if self.data.get("light_match_enabled") else "0",
            "PT_LIGHT_MATCH_TEMP_K": str(_setting_value(self.data, "light_match_temp_k", DEFAULTS["light_match_temp_k"])),
            "PT_LIGHT_MATCH_TINT": str(_setting_value(self.data, "light_match_tint", 0.0)),
            "PT_LIGHT_MATCH_EXPOSURE_EV": str(_setting_value(self.data, "light_match_exposure_ev", 0.0)),
            "PT_LIGHT_MATCH_CONTRAST": str(_setting_value(self.data, "light_match_contrast", 1.0)),
            "PT_LIGHT_MATCH_GAMMA": str(_setting_value(self.data, "light_match_gamma", 1.0)),
            "PT_LIGHT_MATCH_SATURATION": str(_setting_value(self.data, "light_match_saturation", 1.0)),
            "PT_LIGHT_MATCH_PRESET": str(self.data.get("light_match_preset") or DEFAULTS["light_match_preset"]),
            "PT_SI_MIX_ENABLED": "1" if self.data.get("si_enabled") else "0",
            "PT_SI_MIX_CHANNEL": str(self.data.get("si_mix_channel") or DEFAULTS["si_mix_channel"]),
            "PT_SI_ORIGINAL_VOLUME_PERCENT": str(
                _setting_value(self.data, "si_original_volume_percent", DEFAULTS["si_original_volume_percent"])
            ),
            "PT_SI_VOLUME_PERCENT": str(_setting_value(self.data, "si_volume_percent", DEFAULTS["si_volume_percent"])),
            "PT_SI_DELAY_SECONDS": str(_setting_value(self.data, "si_delay_seconds", DEFAULTS["si_delay_seconds"])),
            "PT_SI_DUCK_ORIGINAL": "1" if self.data.get("si_duck_original", DEFAULTS["si_duck_original"]) else "0",
            "PT_SI_DUCK_PRESET": si_duck_preset,
            "PT_SI_DUB_MODE": "1" if self.data.get("si_dub_mode", DEFAULTS["si_dub_mode"]) else "0",
            "PT_RM_ENABLED": "1" if self.data.get("rm_enabled", DEFAULTS["rm_enabled"]) else "0",
            "PT_FACE_BEAUTY_ENABLED": "1" if self.data.get(
                "face_beauty_enabled", DEFAULTS["face_beauty_enabled"]) else "0",
            "PT_FACE_BEAUTY_PRESET": str(
                (self.data.get("face_beauty_live") or {}).get("preset") or "standard"),
            "PT_RM_VR2FLAT_DECODE": "1" if self.data.get("rm_vr2flat_decode", DEFAULTS["rm_vr2flat_decode"]) else "0",
            "PT_ALPHA_2D_ENABLE": "1" if self.data.get("mode_2d", DEFAULTS["mode_2d"]) else "0",
            "PT_ALPHA_2D_PROJECTION": str(self.data.get("alpha_2d_projection") or "fisheye"),
            "PT_ALPHA_2D_DISTANCE_M": str(_setting_value(self.data, "alpha_2d_distance_m", 4.0)),
            "PT_TWO_DVR_MODEL": str(DEFAULTS["two_dvr_live_model"]),
            "PT_TWO_DVR_HOLE_FILL": str(DEFAULTS["two_dvr_live_hole_fill"]),
            "PT_TWO_DVR_EYE_DISTANCE_MM": str(DEFAULTS["two_dvr_live_eye_distance"]),
            "PT_TWO_DVR_STRENGTH": str(_two_dvr_strength_value(
                _setting_value(self.data, "two_dvr_live_strength", DEFAULTS["two_dvr_live_strength"])
            )),
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
        server_name = self.server_name()
        if server_name:
            env["PT_SERVER_NAME"] = server_name
        color = str(self.data.get("subtitle_color") or "").strip()
        if color:
            env["PT_SUBTITLE_COLOR"] = color
        else:
            env.pop("PT_SUBTITLE_COLOR", None)
        if str(self.data.get("inference_backend") or "cuda").lower() == "tensorrt":
            # The UI stays GPU-runtime-free.  The server process validates the
            # manifest/fingerprint and falls back to CUDA when TRT is stale.
            env["PT_ONNX_PROVIDERS"] = TRT_PROVIDER_CHAIN
        return env

    def http_port(self) -> int:
        try:
            port = int(self.data.get("http_port") or DEFAULT_HTTP_PORT)
        except (TypeError, ValueError):
            return DEFAULT_HTTP_PORT
        return port if 1 <= port <= 65535 else DEFAULT_HTTP_PORT

    def server_name(self) -> str:
        return str(self.data.get("server_name") or "").strip()

    def video_dirs(self) -> list[str]:
        raw = self.data.get("video_dirs")
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip() and not is_unc_path(item)]
        elif isinstance(raw, str):
            values = [item.strip() for item in raw.split("|") if item.strip() and not is_unc_path(item)]
        else:
            values = []
        return values or [str(ROOT / "videos")]

    def set_video_dirs(self, directories: list[str]) -> None:
        values = [
            str(Path(directory).expanduser())
            for directory in directories
            if str(directory).strip() and not is_unc_path(directory)
        ]
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
