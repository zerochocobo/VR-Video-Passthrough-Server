from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    assert source_block_reason(1920, 1080) is None
    assert source_block_reason(1920, 1080, is_10bit=True) == "unsupported_10bit_source"
    assert source_block_reason(4096, 2048, is_vr=True) == "unsupported_vr_source"
    assert source_block_reason(3840, 1920, is_vr=True, allow_vr=True) is None
    assert source_block_reason(4096, 2048, is_vr=True, allow_vr=True) is None
    assert source_block_reason(3840, 2160) == "project_resolution_policy"
    assert source_block_reason(426, 240) == "project_resolution_policy"
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
    stream_source = (root / "pipeline" / "pynv_stream.py").read_text(encoding="utf-8-sig")
    kernel_source = (root / "offline" / "two_dvr_pynv.py").read_text(encoding="utf-8-sig")
    assert 'get_function("rgba_to_nv12")' in stream_source
    assert "sr_out[:, :, :3]" not in stream_source
    assert 'extern "C" __global__ void rgba_to_nv12' in kernel_source
    assert 'get_function("hdr_look_rgba")' in stream_source
    assert "sr_split_eyes" in stream_source
    assert "sr_left_eye, (sr_eye_out_w, out_h)" in stream_source
    assert "sr_right_eye, (sr_eye_out_w, out_h)" in stream_source
    assert stream_source.index("apply_hdr_look(sr_k_hdr, sr_out") < stream_source.index("sr_k_to_nv12(grid_out")


def test_offline_split_eye_superres_fuses_nv12_conversion_and_eye_split():
    root = Path(__file__).resolve().parents[1]
    source = (root / "offline" / "rtx_vsr_pynv.py").read_text(encoding="utf-8-sig")
    assert 'get_function("nv12_to_split_rgba")' in source
    assert "left_eye[:, :, :3]" not in source
    assert "right_eye[:, :, :3]" not in source
    assert 'PT_RTX_VSR_STAGE_TIMING' in source
