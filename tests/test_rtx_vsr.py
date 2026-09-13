from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config
import utils.rtx_vsr as rtx_vsr
from offline.rtx_vsr import _format_progress_time, _progress_message
from utils.rtx_vsr import (
    effective_offline_target_height, source_block_reason,
    source_exceeds_target_resolution, target_dimensions, target_resolution,
)
from utils.vr_naming import superres_output_stem, superres_stem
from utils.superres_bitrate import normalize_superres_bitrate_mode, plan_superres_bitrate
from ui.settings import Settings
from pipeline.hdr_look import hdr_look_mode_value, normalize_hdr_look


def test_target_dimensions_preserves_aspect_and_even_size():
    assert target_dimensions(1280, 720, 2160) == (3840, 2160)
    width, height = target_dimensions(853, 480, 1080)
    assert width % 2 == 0 and height % 2 == 0
    assert abs((width / height) - (853 / 480)) < 0.002
    assert target_dimensions(720, 1280, 2160) == (2160, 3840)
    assert target_dimensions(3840, 1920, 2160) == (4096, 2048)
    assert target_dimensions(4096, 2048, 2160) == (4096, 2048)
    assert target_dimensions(3840, 1920, 4096) == (8192, 4096)
    assert target_dimensions(4096, 2048, 4096) == (8192, 4096)
    assert target_resolution(4096, 3840, 1920) == (8192, 4096)
    assert target_resolution(4096, 1920, 1080) == (3840, 2160)
    assert target_dimensions(1920, 1080, 4096) == (3840, 2160)
    assert target_dimensions(2560, 1080, 1440) == (2560, 1080)
    assert target_resolution(1440) == (2560, 1440)
    assert source_exceeds_target_resolution(3000, 1200, 1440) is True
    assert source_exceeds_target_resolution(1200, 2000, 1440) is False


def test_offline_superres_auto_bitrate_uses_sqrt_pixel_ratio():
    plan = plan_superres_bitrate(30_000_000, 4096, 2048, 8192, 4096, "auto")
    assert plan.pixel_ratio == 4.0
    assert plan.target_bps == 60_000_000
    assert plan.max_bps == 75_000_000
    assert plan.buffer_bps == 120_000_000


def test_offline_superres_manual_bitrate_modes_use_source_multiplier():
    plan = plan_superres_bitrate(30_000_000, 3840, 1920, 8192, 4096, "1.5")
    assert plan.target_bps == 45_000_000
    assert plan.max_bps == 56_250_000
    assert plan.buffer_bps == 90_000_000
    assert normalize_superres_bitrate_mode("2.0x") == "2"
    assert normalize_superres_bitrate_mode("invalid") == "auto"


def test_offline_superres_bitrate_caps_auto_and_manual_modes():
    auto = plan_superres_bitrate(100_000_000, 3840, 1920, 8192, 4096, "auto")
    manual = plan_superres_bitrate(100_000_000, 3840, 1920, 8192, 4096, "3")
    assert auto.target_bps == 80_000_000
    assert manual.target_bps == 120_000_000


def test_offline_8k_target_is_limited_to_sbs_vr():
    assert effective_offline_target_height(4096, 3840, 1920) == 4096
    assert effective_offline_target_height(4096, 4096, 2048) == 4096
    assert effective_offline_target_height(4096, 1920, 1080) == 2160
    assert effective_offline_target_height(4096, 3840, 2160) == 2160


def test_offline_progress_message_contains_percent_fps_elapsed_and_eta():
    message = _progress_message(150, 450, 30.0)
    assert "progress=33.3%" in message
    assert "fps=5.00" in message
    assert "elapsed=00:00:30" in message
    assert "eta=00:01:00" in message
    assert _format_progress_time(3661) == "01:01:01"


def test_hdr_look_modes_are_stable_and_default_to_natural():
    assert normalize_hdr_look(None) == "natural"
    assert normalize_hdr_look("vivid") == "vivid"
    assert normalize_hdr_look("invalid") == "natural"
    assert hdr_look_mode_value("off") == 0
    assert hdr_look_mode_value("natural") == 1
    assert hdr_look_mode_value("vivid") == 2


def test_source_policy_rejects_vr_and_out_of_policy(monkeypatch):
    monkeypatch.setattr("config.RTX_VSR_ENABLED", True)
    monkeypatch.setattr("config.RTX_VSR_INPUT_MIN_HEIGHT", 360)
    monkeypatch.setattr("config.RTX_VSR_INPUT_MAX_HEIGHT", 1440)
    # These assert the UPSCALE input policy, so they name a target: with the
    # native 1x default the policy does not apply (see the native gate test).
    assert source_block_reason(1920, 1080, target_height=2160) is None
    assert source_block_reason(1920, 1080, is_10bit=True, target_height=2160) == "unsupported_10bit_source"
    assert source_block_reason(4096, 2048, is_vr=True, target_height=2160) == "unsupported_vr_source"
    assert source_block_reason(3840, 1920, is_vr=True, allow_vr=True, target_height=2160) is None
    assert source_block_reason(4096, 2048, is_vr=True, allow_vr=True, target_height=2160) is None
    assert source_block_reason(3840, 2160, target_height=2160) == "project_resolution_policy"
    assert source_block_reason(426, 240, target_height=2160) == "project_resolution_policy"
    assert source_block_reason(3000, 1200, target_height=1440) == "source_exceeds_target_resolution"


def test_superres_marker_is_stable():
    assert superres_stem("movie") == "[SuperRes]movie"
    assert superres_stem("[SuperRes]movie") == "[SuperRes]movie"
    assert superres_output_stem("movie", 1440) == "movie_2K"
    assert superres_output_stem("movie", 2160) == "movie_4K"
    assert superres_output_stem("movie", 4096) == "movie_8K"
    assert superres_output_stem("movie_4K", 4096) == "movie_8K"


def test_realtime_superres_has_independent_output_mode():
    settings = Settings.__new__(Settings)
    settings.data = {
        "mode_green": False,
        "mode_alpha": False,
        "mode_two_dvr": False,
        "mode_superres": True,
    }
    assert settings.passthrough_mode() == "superres"


def test_dlna_superres_gate_rejects_10bit_indexed_source(monkeypatch):
    import dlna.content_directory as cds

    monkeypatch.setattr(cds, "RTX_VSR_ENABLED", True)
    monkeypatch.setattr(cds, "RTX_VSR_REALTIME_ENABLED", True)
    monkeypatch.setattr(cds, "RTX_VSR_INPUT_MIN_HEIGHT", 360)
    monkeypatch.setattr(cds, "RTX_VSR_INPUT_MAX_HEIGHT", 1440)
    monkeypatch.setattr(cds, "RTX_VSR_TARGET_HEIGHT", 1440)
    path = Path("movie.mp4")

    def child_for(pix_fmt: str):
        return SimpleNamespace(video=SimpleNamespace(width=1920, height=1080, pix_fmt=pix_fmt))

    with patch.object(cds, "_is_two_d_source", return_value=True):
        assert cds._superres_dlna_enabled(path, 1920, 1080, child_for("yuv420p")) is True
        # IndexedVideo carries no bit_depth field, so the gate must read pix_fmt.
        assert cds._superres_dlna_enabled(path, 1920, 1080, child_for("yuv420p10le")) is False
        assert cds._superres_dlna_enabled(path, 3000, 1200, child_for("yuv420p")) is False
    vr_path = Path("movie_LR_180.mp4")
    monkeypatch.setattr(cds, "RTX_VSR_TARGET_HEIGHT", 2160)
    assert cds._superres_dlna_enabled(vr_path, 3840, 1920, child_for("yuv420p")) is True
    assert cds._superres_dlna_enabled(vr_path, 4096, 2048, child_for("yuv420p")) is True


def test_failed_preflight_is_retried_only_after_cooldown(monkeypatch):
    monkeypatch.setattr(rtx_vsr, "_preflight_result", None)
    monkeypatch.setattr(rtx_vsr, "_preflight_failed_at", 0.0)
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(rtx_vsr.subprocess, "run", fake_run)
    assert rtx_vsr.run_evaluation_preflight()["ok"] is False
    assert calls["n"] == 1
    # Within the cooldown the cached failure is served without a new child.
    assert rtx_vsr.run_evaluation_preflight()["ok"] is False
    assert calls["n"] == 1
    monkeypatch.setattr(
        rtx_vsr, "_preflight_failed_at", rtx_vsr.time.monotonic() - rtx_vsr._PREFLIGHT_FAILURE_RETRY_SEC - 1
    )
    rtx_vsr.run_evaluation_preflight()
    assert calls["n"] == 2


def test_realtime_superres_uses_rgba_stride_aware_nv12_kernel():
    root = Path(__file__).resolve().parents[1]
    stage_source = (root / "pipeline" / "superres_stage.py").read_text(encoding="utf-8-sig")
    kernel_source = (root / "offline" / "two_dvr_pynv.py").read_text(encoding="utf-8-sig")
    assert 'get_function("rgba_to_nv12")' in stage_source
    assert "enhanced[:, :, :3]" not in stage_source
    assert 'extern "C" __global__ void rgba_to_nv12' in kernel_source
    assert 'get_function("hdr_look_rgba")' in stage_source
    assert "self.split_eyes" in stage_source
    assert "self._left_eye, (self.eye_out_w, self.out_h)" in stage_source
    assert "self._right_eye, (self.eye_out_w, self.out_h)" in stage_source
    assert stage_source.index("apply_hdr_look(self._k_hdr, enhanced") < stage_source.index("self._k_to_nv12(")


def test_live_and_seek_paths_share_one_superres_stage():
    """Two callers, one factory: eye splitting and the SDR look cannot drift.

    The callers used to construct SuperResStage each for themselves, which is
    how the two drifted apart once already. They now both go through
    pynv_stream.make_frame_stage, so the stage exists in exactly one place and a
    new mode is one branch rather than two edits someone can half-make.
    """
    root = Path(__file__).resolve().parents[1]
    stream_source = (root / "pipeline" / "pynv_stream.py").read_text(encoding="utf-8-sig")
    routes_source = (root / "http_app" / "routes_media.py").read_text(encoding="utf-8-sig")
    assert stream_source.count("from pipeline.superres_stage import SuperResStage") == 1
    assert stream_source.count("SuperResStage(") == 1
    assert stream_source.count("make_frame_stage(") == 3      # the def plus both callers
    # The seek route is pulled at playback speed, so it takes the sustainable
    # target, and the MP4 shell has to declare exactly what the stage encodes.
    assert "target_height=seek_target_height() if seek else None" in stream_source
    assert stream_source.count("seek=True") == 1      # only the seek generator asks
    assert "seek_target_height()" in routes_source
    # One authoritative set, read by the route guards and the DLNA browse side.
    assert 'SEEK_FRAME_MODES = frozenset({"green", "alpha", "superres", "dlss5"})' in stream_source
    assert "from pipeline.pynv_stream import SEEK_FRAME_MODES" in routes_source
    assert routes_source.count("in SEEK_FRAME_MODES") >= 4
    dlna_source = (root / "dlna" / "content_directory.py").read_text(encoding="utf-8-sig")
    assert "from pipeline.pynv_stream import SEEK_FRAME_MODES" in dlna_source


def test_offline_split_eye_superres_fuses_nv12_conversion_and_eye_split():
    root = Path(__file__).resolve().parents[1]
    source = (root / "offline" / "rtx_vsr_pynv.py").read_text(encoding="utf-8-sig")
    assert 'get_function("nv12_to_split_rgba")' in source
    assert "left_eye[:, :, :3]" not in source
    assert "right_eye[:, :, :3]" not in source
    assert 'PT_RTX_VSR_STAGE_TIMING' in source


def test_native_target_is_a_real_height_not_a_missing_value():
    assert rtx_vsr.resolve_target_height(None) == config.RTX_VSR_TARGET_HEIGHT
    assert rtx_vsr.resolve_target_height(0) == rtx_vsr.NATIVE_TARGET_HEIGHT
    assert rtx_vsr.resolve_target_height(-5) == rtx_vsr.NATIVE_TARGET_HEIGHT
    assert rtx_vsr.resolve_target_height(3072) == 3072
    assert rtx_vsr.is_native_target(0) is True
    assert rtx_vsr.is_native_target(2160) is False


def test_native_target_keeps_the_source_size():
    assert target_dimensions(3840, 1920, 0) == (3840, 1920)
    assert target_dimensions(1281, 721, 0) == (1282, 722)
    assert target_resolution(0, 3840, 1920) == (3840, 1920)
    # Native never enlarges, so it can never overshoot its own target.
    assert source_exceeds_target_resolution(7680, 3840, 0) is False
    assert effective_offline_target_height(0, 3840, 2160) == 0


def test_six_k_is_a_vr_target_and_two_d_falls_back_to_four_k():
    assert target_resolution(3072, 3840, 1920) == (6144, 3072)
    assert target_dimensions(3840, 1920, 3072) == (6144, 3072)
    assert target_resolution(3072, 1920, 1080) == (3840, 2160)
    assert effective_offline_target_height(3072, 3840, 1920) == 3072
    assert effective_offline_target_height(3072, 1920, 1080) == 2160
    # The 8K target is unchanged by the new middle step.
    assert target_dimensions(3840, 1920, 4096) == (8192, 4096)


def test_native_gate_replaces_the_upscale_input_policy():
    # A 4K 2D source is rejected for upscaling but allowed at 1x.
    assert source_block_reason(3840, 2160, target_height=2160) == "project_resolution_policy"
    assert source_block_reason(3840, 2160, target_height=0) is None
    # 8K SBS VR fits the encoder envelope at 1x.
    assert source_block_reason(8192, 4096, is_vr=True, allow_vr=True, target_height=0) is None
    assert source_block_reason(9000, 4096, is_vr=True, allow_vr=True, target_height=0) == "source_exceeds_encoder_limit"
    # The native ceiling and the shared minimum still apply.
    assert source_block_reason(8192, 4320, is_vr=True, allow_vr=True, target_height=0) == "project_resolution_policy"
    assert source_block_reason(320, 180, target_height=0) == "project_resolution_policy"
    assert source_block_reason(0, 1080, target_height=0) == "invalid_source_size"
    assert source_block_reason(1920, 1080, is_10bit=True, target_height=0) == "unsupported_10bit_source"


def test_superres_output_stem_marks_native_and_six_k():
    assert superres_output_stem("clip", 0) == "clip_1X"
    assert superres_output_stem("clip", 3072) == "clip_6K"
    assert superres_output_stem("clip", 4096) == "clip_8K"
    assert superres_output_stem("clip", 2160) == "clip_4K"
    assert superres_output_stem("clip", 1440) == "clip_2K"
    # Re-running against an existing output replaces the marker.
    assert superres_output_stem("clip_8K", 0) == "clip_1X"
    assert superres_output_stem("clip_1x", 3072) == "clip_6K"


def test_ui_target_choices_cover_every_label():
    from ui.superres_targets import DEFAULT_TARGET, TARGET_CHOICES, target_i18n_key

    assert TARGET_CHOICES == (0, 1440, 2160, 3072, 4096)
    assert DEFAULT_TARGET in TARGET_CHOICES
    keys = [target_i18n_key(target) for target in TARGET_CHOICES]
    assert keys == [
        "superres.target_native",
        "superres.target_2k",
        "superres.target_4k",
        "superres.target_6k_vr",
        "superres.target_8k_vr",
    ]
    assert len(set(keys)) == len(keys)
    assert target_i18n_key("bogus") == target_i18n_key(DEFAULT_TARGET)


def test_only_native_superres_is_offered_as_a_virtual_file():
    """Enlarging cannot be pulled at playback speed, so it stays on live."""
    from pathlib import Path as _Path

    import dlna.content_directory as cds

    assert rtx_vsr.seek_supported_target(0) is True
    assert rtx_vsr.seek_supported_target(3072) is False
    assert rtx_vsr.seek_supported_target(4096) is False
    # The route renders the configured target; it never substitutes another.
    assert rtx_vsr.seek_target_height(3072) == 3072
    with patch.object(rtx_vsr.config, "RTX_VSR_SEEK_ALLOW_UPSCALE", True):
        assert rtx_vsr.seek_supported_target(4096) is True

    with patch.object(cds, "RTX_VSR_TARGET_HEIGHT", 0):
        assert cds._seek_supported_mode("superres") is True
    with patch.object(cds, "RTX_VSR_TARGET_HEIGHT", 3072):
        assert cds._seek_supported_mode("superres") is False
    # Green and Alpha are unaffected by the SuperRes target.
    assert cds._seek_supported_mode("green") is True
    assert cds._seek_supported_mode("alpha") is True
    # Skybox caches item titles, so the label must stay stable.
    assert cds._passthrough_seek_title(_Path("MOVIE.mp4"), "superres", 1920, 1080).startswith("[SUPERRES]")


def test_frames_budget_scales_for_enlarged_superres():
    """A 4K-tuned budget truncates 6K SuperRes frames and corrupts the stream."""
    import http_app.routes_media as rm

    template = lambda w, h: SimpleNamespace(width=w, height=h)
    base = rm._vmp4_frames_frame_budget("green", template(3840, 2160))
    # Green/Alpha and native-1x SuperRes keep the configured budget.
    assert rm._vmp4_frames_frame_budget("alpha", template(8192, 4096)) == base
    assert rm._vmp4_frames_frame_budget("superres", template(3840, 1920)) == base
    # Enlarged SuperRes gets room for its extra pixels, capped at 4x.
    six_k = rm._vmp4_frames_frame_budget("superres", template(6144, 3072))
    eight_k = rm._vmp4_frames_frame_budget("superres", template(8192, 4096))
    assert base < six_k < eight_k <= 4 * base
    assert six_k % 65536 == 0 and eight_k % 65536 == 0
