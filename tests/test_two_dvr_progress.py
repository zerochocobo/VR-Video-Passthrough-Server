from __future__ import annotations

from unittest.mock import patch


def test_progress_message_includes_percent_elapsed_eta_and_fps() -> None:
    from offline.two_dvr import _progress_message

    with patch("offline.two_dvr.time.time", return_value=110.0):
        text = _progress_message(done=50, total=200, started=100.0)

    assert "50/200 frames" in text
    assert "( 25.0%)" in text
    assert "elapsed=00:10" in text
    assert "eta=00:30" in text
    assert "5.0 fps" in text


def test_progress_message_handles_unknown_total() -> None:
    from offline.two_dvr import _progress_message

    with patch("offline.two_dvr.time.time", return_value=105.0):
        text = _progress_message(done=25, total=0, started=100.0)

    assert "25 frames" in text
    assert "elapsed=00:05" in text
    assert "eta=--:--" in text
    assert "5.0 fps" in text


def test_clip_expected_frames_uses_remaining_duration_when_duration_is_zero() -> None:
    from offline.two_dvr import _clip_expected_frames

    assert _clip_expected_frames(fps=24.0, total_duration=120.0, start=30.0, duration=0.0) == 2160
    assert _clip_expected_frames(fps=24.0, total_duration=120.0, start=30.0, duration=10.0) == 240


def test_pynv_encoder_kwargs_disable_b_frames_for_stable_mux_timestamps() -> None:
    from offline.two_dvr_pynv import _encoder_kwargs

    kwargs = _encoder_kwargs("20M", 60000 / 1001)

    assert kwargs["bf"] == str(__import__("config").PASSTHROUGH_HEVC_BF)
    assert kwargs["bf"] == "0"
    assert kwargs["preset"]


def test_pynv_encoder_kwargs_repeat_parameter_sets_for_decoder_compatibility() -> None:
    from offline.two_dvr_pynv import _encoder_kwargs

    kwargs = _encoder_kwargs("20M", 60000 / 1001)

    assert kwargs["repeatspspps"] == "1"
