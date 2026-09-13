from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import utils.rtx_vsr as rtx_vsr
from pipeline.hdr_look import HDR_LOOK_MODES, hdr_look_mode_value, normalize_hdr_look
from pipeline.true_hdr import is_true_hdr, p010_grid, true_hdr_color_metadata
from utils.rtx_vsr import TrueHdrSettings


def _bridge(tmp_path, *, with_truehdr_dll: bool = True) -> tuple[rtx_vsr.RtxVsrBridge, MagicMock]:
    if with_truehdr_dll:
        (tmp_path / "nvngx_truehdr.dll").write_bytes(b"")
    dll = MagicMock()
    dll.rtx_video_api_cuda_create.return_value = 1
    return rtx_vsr.RtxVsrBridge(tmp_path, dll), dll


def test_true_hdr_is_a_recognised_hdr_look_but_not_an_sdr_grade():
    assert normalize_hdr_look("truehdr") == "truehdr"
    assert normalize_hdr_look("TrueHDR") == "truehdr"
    # The SDR look kernel must stay a no-op: NGX produces the whole HDR grade.
    assert hdr_look_mode_value("truehdr") == 0
    assert is_true_hdr("truehdr") is True
    assert is_true_hdr("natural") is False
    assert is_true_hdr(None) is False
    assert "truehdr" in HDR_LOOK_MODES


def test_true_hdr_output_is_tagged_hdr10():
    args = true_hdr_color_metadata().ffmpeg_args()
    assert args == [
        "-color_range", "tv",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
    ]


def test_p010_grid_covers_every_two_by_two_block():
    block = (16, 16, 1)
    assert p010_grid(8192, 4096, block) == (256, 128, 1)
    # Odd sizes still get a thread for the trailing partial block.
    assert p010_grid(34, 34, block) == (2, 2, 1)


def test_true_hdr_settings_clamp_to_sdk_ranges():
    packed = TrueHdrSettings(contrast=999, saturation=-5, middle_gray=0, max_luminance=99_999).to_struct()
    assert (packed.contrast, packed.saturation) == (200, 0)
    assert (packed.middle_gray, packed.max_luminance) == (10, 2000)
    default = TrueHdrSettings().to_struct()
    assert (default.contrast, default.saturation, default.middle_gray, default.max_luminance) == (100, 100, 50, 1000)


def test_bridge_requests_the_true_hdr_feature_and_rebuilds_when_the_mode_changes(tmp_path):
    bridge, dll = _bridge(tmp_path)
    assert bridge.initialize(0, cu_context=11, cu_stream=22, true_hdr=True)
    assert dll.rtx_video_api_cuda_create.call_args[0][3] == 1
    # Same request again reuses the native session.
    assert bridge.initialize(0, cu_context=11, cu_stream=22, true_hdr=True)
    assert dll.rtx_video_api_cuda_create.call_count == 1
    # TrueHDR is chosen at create time, so switching output tears it down.
    assert bridge.initialize(0, cu_context=11, cu_stream=22, true_hdr=False)
    dll.rtx_video_api_cuda_shutdown.assert_called_once()
    assert dll.rtx_video_api_cuda_create.call_args[0][3] == 0


def test_bridge_reports_a_missing_true_hdr_runtime(tmp_path):
    bridge, _ = _bridge(tmp_path, with_truehdr_dll=False)
    assert bridge.true_hdr_runtime_available() is False
    with pytest.raises(RuntimeError, match="TrueHDR runtime missing"):
        bridge.initialize(0, cu_context=1, cu_stream=2, true_hdr=True)


def test_evaluate_rejects_a_mode_the_bridge_was_not_created_for(tmp_path):
    bridge, dll = _bridge(tmp_path)
    assert bridge.initialize(0, cu_context=1, cu_stream=2, true_hdr=False)
    with pytest.raises(RuntimeError, match="SDR output"):
        bridge.evaluate_deviceptr(1, 2, (640, 360), (1280, 720), 2, true_hdr=True)
    dll.rtx_video_api_cuda_evaluate_deviceptr.assert_not_called()


def test_evaluate_passes_the_configured_true_hdr_controls(tmp_path):
    bridge, dll = _bridge(tmp_path)
    dll.rtx_video_api_cuda_evaluate_deviceptr.return_value = 1
    assert bridge.initialize(0, cu_context=1, cu_stream=2, true_hdr=True)
    bridge.evaluate_deviceptr(
        1, 2, (640, 360), (1280, 720), 3,
        true_hdr=True,
        hdr_settings=TrueHdrSettings(contrast=120, saturation=90, middle_gray=44, max_luminance=1400),
    )
    call = dll.rtx_video_api_cuda_evaluate_deviceptr.call_args[0]
    assert call[4]._obj.quality_level == 3
    thdr = call[5]._obj
    assert (thdr.contrast, thdr.saturation, thdr.middle_gray, thdr.max_luminance) == (120, 90, 44, 1400)
