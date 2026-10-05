"""Prepared toolbox mixes must skip encoding and invalidate with their inputs."""
import json
import asyncio
import queue
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

import dlna.content_directory as cds
import http_app.routes_media as routes
import pipeline.si_virtual_mp4 as siv
from tests.test_si_virtual_playback import _sample_file
from utils.si_filter import SIMixParams, resolve_si_mix_inputs
from utils.si_prepared_audio import input_signature, make_manifest, prepared_paths


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    video = tmp_path / "movie.mp4"
    voice = video.with_suffix(".si.wav")
    voice.write_bytes(b"voice")
    _sample_file(video, [b"video-zero", b"video-one", b"video-two"])
    monkeypatch.setattr(siv, "_layout_cache", {})
    monkeypatch.setattr(siv, "_info_cache", {})
    monkeypatch.setattr(siv, "_ensure_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(siv, "_selected_mix_encoder", lambda: "aac")
    params = SIMixParams()

    def publish(samples=(b"audio-zero", b"audio-one", b"audio-two"), *, settings=params):
        audio, meta = prepared_paths(video)
        _sample_file(audio, samples, audio=True)
        effective, duck = resolve_si_mix_inputs(video, settings)
        meta.write_text(json.dumps(make_manifest(
            input_signature(video, voice, duck), effective.filter_string(duck is not None), audio,
        )), encoding="utf-8")
        return audio, meta

    return video, voice, params, publish


@pytest.mark.parametrize("dub", [False, True])
def test_prepared_browse_playback_and_ranges_never_encode(prepared, dub):
    video, voice, params, publish = prepared
    if dub:
        video.with_suffix(".si.duck.wav").write_bytes(b"key")
        params = replace(params, si_delay_seconds=0.7, mix_channel="left")
    audio, _ = publish(settings=params)
    with (patch.object(siv, "_selected_mix_encoder", side_effect=AssertionError("No encoder probe")),
          patch.object(siv, "_ensure_cache_dir", side_effect=AssertionError("No runtime audio cache access")),
          patch.object(siv, "_run_ffmpeg_sidecar", side_effect=AssertionError("No encoding")),
          patch.object(siv, "_run_ffmpeg_sidecar_with_source_audio_pipe", side_effect=AssertionError("No extraction"))):
        info = siv.progressive_si_info(video, voice, params)
        layout = siv.build_progressive_si_virtual_mp4(video, voice, params)
    assert layout.audio_path == audio
    assert layout.content_length == info.content_length
    payload = b"".join(siv.iter_virtual_range(layout.regions, 0, layout.content_length - 1))
    assert len(payload) == info.content_length
    for start, end in [(0, 127), (len(payload) - 32, len(payload) - 1)]:
        assert b"".join(siv.iter_virtual_range(layout.regions, start, end)) == payload[start:end + 1]
    assert not (video.parent / "cache").exists()


@pytest.mark.parametrize("dub", [False, True])
def test_prepared_dlna_head_get_and_prewarm_never_access_runtime(prepared, monkeypatch, dub):
    video, voice, params, publish = prepared
    if dub:
        video.with_suffix(".si.duck.wav").write_bytes(b"subtitle-span key")
        params = replace(params, si_delay_seconds=0.7, mix_channel="left")
    audio, _ = publish(settings=params)
    work_queue = queue.PriorityQueue(maxsize=1)
    monkeypatch.setattr(siv, "_layout_prewarm_queue", work_queue)
    monkeypatch.setattr(siv, "_layout_prewarm_inflight", set())
    monkeypatch.setattr(siv, "SI_PREWARM_QUEUE_MAX", 1)
    # Consume a single queued task in the test; avoid leaving a daemon worker.
    monkeypatch.setattr(siv, "_ensure_prewarm_worker_locked", lambda: None)
    with (patch.object(siv, "_ensure_cache_dir", side_effect=AssertionError("No runtime cache")),
          patch.object(siv, "_source_audio_sidecar_output", side_effect=AssertionError("No source cache")),
          patch.object(siv, "build_source_audio_sidecar", side_effect=AssertionError("No extraction")),
          patch.object(siv, "_selected_mix_encoder", side_effect=AssertionError("No encoder probe")),
          patch.object(siv, "_run_ffmpeg_sidecar", side_effect=AssertionError("No encoding")),
          patch.object(siv, "_run_ffmpeg_sidecar_with_source_audio_pipe", side_effect=AssertionError("No pipe")),
          patch.object(cds, "get_si_mix", return_value=SimpleNamespace(params=lambda: params)),
          patch.object(cds, "_rel_key", return_value="movie.mp4")):
        item = cds._si_file_item(video, "0", 1.0, 8192, 4096, source_size=video.stat().st_size)
        priority, _, task = work_queue.get_nowait()
        layout = siv.build_progressive_si_virtual_mp4(task.video, task.si_wav, task.params,
                                                     priority=priority, reason=task.reason, cancelable=True)
        assert layout.audio_path == audio
        assert item["size"] == layout.content_length and not item["size_estimated"]
        expected = b"".join(siv.iter_virtual_range(layout.regions, 0, layout.content_length - 1))
        app = FastAPI()
        app.include_router(routes.router)
        service = SimpleNamespace(current_config=lambda: params, has_si_source=lambda _: voice)
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                head = await client.head("/media_si/movie.mp4")
                assert head.status_code == 200
                assert int(head.headers["content-length"]) == len(expected)
                assert head.headers["x-si-audio-source"] == "prepared-m4a"
                for start, end in [(0, 127), (len(expected) // 2, len(expected) - 1)]:
                    response = await client.get("/media_si/movie.mp4", headers={"Range": f"bytes={start}-{end}"})
                    assert response.status_code == 206
                    assert response.headers["x-si-audio-source"] == "prepared-m4a"
                    assert response.content == expected[start:end + 1]
        with (patch.object(routes, "_safe_si_video_path", return_value=video),
              patch.object(routes, "get_si_stream_service", return_value=service)):
            asyncio.run(check())
    assert not (video.parent / "cache").exists()


@pytest.mark.parametrize("changed", ["video", "voice", "duck", "audio", "metadata", "settings"])
def test_stale_assets_are_ignored(prepared, changed):
    video, voice, params, publish = prepared
    audio, metadata = publish()
    assert siv._prepared_mixed_audio(video, voice, params) == audio
    if changed == "video":
        video.write_bytes(video.read_bytes() + b"replacement")
    elif changed == "voice":
        voice.write_bytes(b"new voice")
    elif changed == "duck":
        video.with_suffix(".si.duck.wav").write_bytes(b"new duck key")
    elif changed == "audio":
        audio.write_bytes(b"truncated")
    elif changed == "metadata":
        metadata.write_text("{unfinished", encoding="utf-8")
    else:
        params = replace(params, si_volume_percent=80)
    assert siv._prepared_mixed_audio(video, voice, params) is None
    assert siv.progressive_si_info(video, voice, params) is None


def test_replacement_and_removal_invalidate_existing_layout(prepared):
    video, voice, params, publish = prepared
    audio, metadata = publish()
    first = siv.build_progressive_si_virtual_mp4(video, voice, params)
    publish([b"a bigger replacement AAC sample", b"tail"])
    second = siv.build_progressive_si_virtual_mp4(video, voice, params)
    assert first.etag != second.etag
    assert second.content_length == siv.progressive_si_info(video, voice, params).content_length
    metadata.unlink()
    assert siv.progressive_si_info(video, voice, params) is None


def test_bad_container_with_matching_manifest_falls_back(prepared):
    video, voice, params, publish = prepared
    audio, metadata = publish()
    audio.write_bytes(b"not an MP4")
    metadata.write_text(json.dumps(make_manifest(
        input_signature(video, voice, None), params.filter_string(), audio,
    )), encoding="utf-8")
    assert siv._prepared_mixed_audio(video, voice, params) is None
    # Exercise the real fallback dispatcher, without needing a media encoder.
    with patch.object(siv, "_use_segmented_aac", return_value=True), \
         patch.object(siv, "build_mixed_audio_sidecar_parallel", return_value="rebuilt") as encode:
        assert siv.build_mixed_audio_sidecar(video, voice, params) == "rebuilt"
    encode.assert_called_once()


def test_mix_assets_participate_in_media_index_signature():
    from utils.media_index import _si_sidecar_source_name
    assert _si_sidecar_source_name("Movie.si.mix.m4a") == "Movie.mp4"
    assert _si_sidecar_source_name("Movie.SI.MIX.JSON") == "Movie.mp4"
