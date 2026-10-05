from dataclasses import replace
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import config
import dlna.content_directory as cds
import http_app.routes_media as routes
from pipeline.seek_effects import RmStage, effect_settings, iter_stage_frames, settings_digest
from tests.test_passthrough_vmp4_frames import _build
from media_library import MediaLibrary, build_media_roots
from utils import runtime_settings


def child(width=1920, height=1080):
    return SimpleNamespace(size=1000000, video=SimpleNamespace(
        duration=30.0, fps=30.0, width=width, height=height,
        resolution=f"{width}x{height}", backend_verdict="pynv_hevc",
        probe_error="", mkv_needs_fix=False))


@pytest.mark.parametrize("mode,prefix,width", [("two_dvr", "s3_", 3840),
                                              ("rm", "srm_", 1920),
                                              ("face_beauty", "sfb_", 1920)])
def test_new_modes_publish_one_seek_file_and_round_trip(tmp_path, mode, prefix, width):
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"source")
    library = MediaLibrary(build_media_roots([tmp_path]))
    with (
        patch.object(cds, "MEDIA_LIBRARY", library),
        patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "two_dvr" if mode == "two_dvr" else "none"),
        patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", True),
        patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True),
        patch.object(cds, "PASSTHROUGH_SEEK_VMP4", True),
        patch.object(cds, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot_frames"),
        patch.object(cds, "_rm_dlna_enabled", return_value=mode == "rm"),
        patch.object(cds, "_face_beauty_dlna_enabled", return_value=mode == "face_beauty"),
        patch.object(cds, "_seek_declared_size", return_value=1000000),
    ):
        items = cds._video_items_from_index(source, "0", child())
        assert len(items) == 2
        item = items[1]
        assert not item.get("container")
        assert item["id"].startswith(prefix)
        assert f".{mode}.seek.mp4" in item["url"]
        assert item["resolution"] == f"{width}x1080"
        assert cds._id_to_seek(item["id"]) == (source, mode)
        key, container, resolved = routes._split_seek_route_name(f"movie.mp4.{mode}.seek.mp4")
        assert (key, container, resolved) == ("movie.mp4", "mp4", mode)
        if mode == "two_dvr":
            assert "_3D_LR_Screen" in item["title"]


@pytest.mark.parametrize("backend", ["slot", "cache_file"])
def test_legacy_backends_keep_new_effects_as_live_directories(backend):
    with patch.object(cds, "PASSTHROUGH_SEEK_VMP4_BACKEND", backend):
        assert not cds._seek_supported_mode("rm")
        assert not cds._seek_supported_mode("face_beauty")
        assert not cds._seek_supported_mode("two_dvr")


def test_seek_route_rejects_disabled_effect_instead_of_playing_alpha():
    with patch.object(runtime_settings, "get_face_beauty", return_value=SimpleNamespace(enabled=False)):
        with pytest.raises(HTTPException, match="disabled"):
            routes._seek_output_mode("alpha", "face_beauty")


def test_source_geometry_is_not_decode_capped_for_new_effects():
    info = SimpleNamespace(width=7680, height=3840)
    with patch.object(routes, "DECODE_MAX_SIDE", 1920):
        assert routes._vmp4_slot_output_size(info, "face_beauty") == (7680, 3840)
        assert routes._vmp4_slot_output_size(info, "rm") == (7680, 3840)
        assert routes._vmp4_slot_output_size(SimpleNamespace(width=3840, height=2160), "two_dvr") == (7680, 2160)


@pytest.mark.parametrize("count", [1, 7, 8, 9, 17])
def test_rm_chunk_adapter_preserves_every_timestamp_and_pads_only_the_tail(count):
    stage = object.__new__(RmStage)
    stage.window = 8
    stage.config = SimpleNamespace(RM_CONF=0.2)
    stage.rgb = lambda frame: frame
    chunks = []
    def process(frames, conf):
        assert conf == 0.2
        chunks.append(list(frames))
        return [value + 100 for value in frames]
    stage.processor = SimpleNamespace(process_chunk=process)
    stage.nv12 = lambda frame, index: frame
    output = list(iter_stage_frames(stage, ((i + 60, i) for i in range(count))))
    assert output == [(i + 60, i + 100) for i in range(count)]
    assert all(len(chunk) == 8 for chunk in chunks)
    assert chunks[-1][-1] == count - 1


def test_effect_cache_changes_with_strength_and_subtitle_content(tmp_path):
    from utils.runtime_settings import FaceBeautyRuntime
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"source")
    subtitle = source.with_suffix(".srt")
    subtitle.write_text("first", encoding="utf-8")
    with patch("pipeline.subtitles.find_subtitle_for_video", return_value=subtitle):
        with patch.object(runtime_settings, "get_face_beauty", return_value=FaceBeautyRuntime(enabled=True, skin_smooth=20)):
            first = effect_settings("face_beauty", source)
        with patch.object(runtime_settings, "get_face_beauty", return_value=FaceBeautyRuntime(enabled=True, skin_smooth=70)):
            second = effect_settings("face_beauty", source)
        assert settings_digest(first) != settings_digest(second)
        subtitle.write_text("changed subtitle", encoding="utf-8")
        third = effect_settings("face_beauty", source)
        assert settings_digest(second) != settings_digest(third)
    layout = _build(tmp_path)
    assert routes._vmp4_frames_digest(layout.source_path, replace(layout, processing_settings=first), "face_beauty") != routes._vmp4_frames_digest(layout.source_path, replace(layout, processing_settings=second), "face_beauty")


def test_identical_effect_values_reuse_cache_despite_runtime_version():
    assert settings_digest({"light_match": {"gamma": 1.1, "version": 1}}) == settings_digest({"light_match": {"gamma": 1.1, "version": 9}})


@pytest.mark.parametrize("width,stereo,eyes", [(1280, True, 2), (3840, False, 1)])
def test_seek_subtitle_geometry_uses_picture_type_not_width(tmp_path, width, stereo, eyes):
    import numpy as np
    from pipeline.subtitles import SubtitleRenderer
    from pipeline.gpu_subtitles import GpuSubtitleOverlay

    subtitle = tmp_path / "movie.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\nSubtitle", encoding="utf-8")
    renderer = SubtitleRenderer(subtitle, width, 360, stereo=stereo,
                                settings={"SUBTITLE_MODE": "auto", "SUBTITLE_V360": False},
                                blocking=True)
    # A virtual frame must include its subtitle even at the first frame after seek.
    overlay = renderer.overlay_for_time(0.5)
    assert overlay is not None
    positions = GpuSubtitleOverlay()._subtitle_overlay_positions(
        SimpleNamespace(shape=(540, width)), renderer, overlay)
    assert renderer.eye_width == width // eyes
    assert len(positions) == eyes
    if eyes == 2:
        assert positions[1][1] - positions[0][1] == width // 2 + renderer.parallax_px()
    else:
        assert positions[0][1] == (width - overlay[0].shape[1]) // 2
    # A running seek layout keeps its subtitle style after UI configuration changes.
    with patch.object(config, "SUBTITLE_ALPHA", 0.0), patch.object(config, "SUBTITLE_MODE", "mono"):
        pinned = renderer._render_text_overlay(renderer.cues[0].lines)
        assert np.array_equal(overlay[0], pinned[0])
        assert len(GpuSubtitleOverlay()._subtitle_overlay_positions(
            SimpleNamespace(shape=(540, width)), renderer, pinned)) == eyes


@pytest.mark.parametrize("mode", ["two_dvr", "face_beauty", "rm"])
def test_new_seek_routes_head_and_range_keep_path_mode(tmp_path, mode):
    layout = _build(tmp_path)
    source = layout.source_path
    info = SimpleNamespace(duration=2.0, fps=10.0, width=1920, height=1080)
    meta = object()
    template = SimpleNamespace(width=3840 if mode == "two_dvr" else 1920, height=1080)
    request = SimpleNamespace(headers={"user-agent": "VLC/3.0"}, client=SimpleNamespace(host="test-client"))
    range_header = f"bytes=0-{layout.mdat_payload_start - 1}"
    kwargs = dict(mode="alpha", range_header=range_header, time_seek_range=None,
                  get_content_features=None, transfer_mode=None)
    gate_name = {"two_dvr": "_two_dvr_live_block_reason", "rm": "_rm_live_block_reason",
                 "face_beauty": "_face_beauty_live_block_reason"}[mode]
    request_thread = threading.get_ident()
    def cached_layout(*args, **kwargs):
        # HEAD and GET layout construction must not block other SI connections.
        assert threading.get_ident() != request_thread
        return layout
    with (
        patch.object(routes, "_safe_seek_video_path", return_value=(source, "mp4", mode)),
        patch.object(routes, "_seek_route_allowed", return_value=(True, "allowed", "vlc")),
        patch.object(routes, "PASSTHROUGH_OUTPUT_MODE", "alpha,two_dvr"),
        patch.object(runtime_settings, "get_rm", return_value=SimpleNamespace(enabled=True)),
        patch.object(runtime_settings, "get_face_beauty", return_value=SimpleNamespace(enabled=True)),
        patch.object(routes, "PASSTHROUGH_SEEK_VMP4", True),
        patch.object(routes, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot_frames"),
        patch.object(routes, "_probe_live_request_metadata", return_value=(info, meta, "")),
        patch.object(routes, gate_name, return_value="") as gate,
        patch.object(routes, "probe_cached", return_value=info),
        patch.object(routes, "annotate_request"),
        patch.object(routes, "_seek_declared_total", return_value=layout.total_size),
        patch.object(routes, "_seek_headers", return_value={}),
        patch.object(routes, "_vmp4_slot_output_template", return_value=template) as make_template,
        patch.object(routes, "_vmp4_frames_cached_layout", side_effect=cached_layout),
        patch.object(routes, "_vmp4_frames_ensure_filler", side_effect=AssertionError("init/HEAD must not encode")),
    ):
        head = asyncio.run(routes.passthrough_seek_head(request, f"movie.mp4.{mode}.seek.mp4", **kwargs))
        get = asyncio.run(routes.passthrough_seek_get(request, f"movie.mp4.{mode}.seek.mp4", **kwargs))
        async def read_body():
            return b"".join([part async for part in get.body_iterator])
        body = asyncio.run(read_body())
        assert head.status_code == get.status_code == 206
        assert head.headers["content-range"] == get.headers["content-range"]
        assert len(body) == int(get.headers["content-length"]) == layout.mdat_payload_start
        assert body == layout.init
        assert make_template.call_args.args[1] == mode
        gate.assert_called_with(source, meta)


@pytest.mark.parametrize("backend,reason", [("slot", "slot_frames"), ("slot_frames", "unsupported-source")])
def test_new_seek_effect_gate_rejects_ineligible_request(backend, reason):
    with (
        patch.object(routes, "PASSTHROUGH_SEEK_VMP4", True),
        patch.object(routes, "PASSTHROUGH_SEEK_VMP4_BACKEND", backend),
        patch.object(routes, "_probe_live_request_metadata", return_value=(None, None, "unsupported-source")),
    ):
        with pytest.raises(HTTPException, match=reason):
            asyncio.run(routes._validate_seek_effect(Path("movie.mp4"), "rm"))


def test_si_uses_virtual_file_or_live_directory_without_building_on_browse(tmp_path):
    from utils.si_filter import SIMixParams
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"source")
    source.with_suffix(".si.wav").write_bytes(b"audio")
    with (patch.object(cds, "_rel_key", return_value="movie.mp4"),
          patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "none"),
          patch.object(cds, "get_si_mix", return_value=SimpleNamespace(enabled=True, params=lambda: SIMixParams(enabled=True))),
          patch.object(cds, "progressive_si_info", return_value=None),
          patch.object(cds, "prewarm_progressive_si_virtual_mp4") as prewarm,
          patch.object(cds, "PASSTHROUGH_SEEK_ENABLED", True),
          patch.object(cds, "PASSTHROUGH_SEEK_DLNA", True)):
        item = cds._video_items_from_index(source, "0", child())[1]
        assert item["passthrough_mode"] == "si_mix"
        assert "/media_si/" in item["url"]
        assert item["size"] > 0
        assert item["size_estimated"]
        prewarm.assert_called_once()
        assert not item.get("container")
        with patch.object(cds, "PASSTHROUGH_SEEK_DLNA", False):
            item = cds._video_items_from_index(source, "0", child())[1]
            assert item["container"]


def test_sbs_budget_accounts_for_both_eyes():
    with (patch.object(config, "PASSTHROUGH_HEVC_BITRATE", "1M"),
          patch.object(config, "PASSTHROUGH_SEEK_VMP4_FRAMES_BUDGET_MAX_SCALE", 100.0)):
        assert config.seek_declared_total_bytes(1000000, 10, 3840, 2160, 30, "two_dvr") > config.seek_declared_total_bytes(1000000, 10, 3840, 2160, 30, "green")


def test_dashboard_feature_settings_preserve_global_playback_choice():
    import os
    from ui.qt_runtime import configure_qt_runtime_paths
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    configure_qt_runtime_paths()
    from PySide6.QtWidgets import QApplication, QDialog
    from ui.dialogs.feature_dialogs import (
        TwoDvrSettingsDialog, RmSettingsDialog, FaceBeautySettingsDialog,
        SISettingsDialog, GreenScreenSettingsDialog, SuperResSettingsDialog,
        DLSS5SettingsDialog,
    )
    from ui.pages.dashboard_page import DashboardPage
    from ui.settings import DEFAULTS
    from ui.i18n import I18n
    app = QApplication.instance() or QApplication([])
    settings = SimpleNamespace(data=dict(DEFAULTS), save=lambda: None)
    page = DashboardPage(I18n("zh_CN"), settings)
    try:
        for mode in ("virtual", "live"):
            settings.data["passthrough_playback_mode"] = mode
            for cls, handler in (
                (TwoDvrSettingsDialog, page._configure_two_dvr),
                (RmSettingsDialog, page._configure_rm),
                (FaceBeautySettingsDialog, page._configure_face_beauty),
                (SISettingsDialog, page._configure_si),
                (GreenScreenSettingsDialog, page._configure_green),
                (SuperResSettingsDialog, page._configure_superres),
                (DLSS5SettingsDialog, page._configure_dlss5),
            ):
                def accept(dialog):
                    assert not hasattr(dialog, "playback")
                    return QDialog.DialogCode.Accepted

                with (patch.object(cls, "exec", accept),
                      patch("ui.pages.dashboard_page.send_control")):
                    handler()
                assert settings.data["passthrough_playback_mode"] == mode
    finally:
        page.close()
        app.processEvents()
