"""SI Browse metadata and progressive HTTP startup regressions."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree as ET

import httpx
import pytest
from fastapi import FastAPI

import dlna.content_directory as cds
import http_app.routes_media as routes
import pipeline.si_virtual_mp4 as siv
from tests.test_si_virtual_mp4 import (
    _box, _elst, _hdlr, _mdhd, _minimal_moov_with_timescale,
    _stco, _stsc, _stsd_audio, _stsd_video, _stsz, _stts, _tkhd,
)
from utils.si_filter import SIMixParams, resolve_si_mix_inputs


def _sample_file(path, samples, codec="hvc1", *, audio=False):
    ftyp = _box("ftyp", b"isom")
    def moov(offsets):
        stsd = _stsd_audio() if audio else _stsd_video(codec)
        stbl = _box("stbl", stsd + _stts([(len(samples), 1024)])
                    + _stsc([(1, 1, 1)]) + _stsz([len(s) for s in samples]) + _stco(offsets))
        mdia = _box("mdia", _mdhd(48000, len(samples) * 1024)
                    + _hdlr(b"soun" if audio else b"vide") + _box("minf", stbl))
        track = _box("trak", _tkhd(2 if audio else 1) + (_elst() if audio else b"") + mdia)
        return _minimal_moov_with_timescale(track)
    header = moov([0] * len(samples))
    cursor = len(ftyp) + len(header) + 8
    offsets = []
    for sample in samples:
        offsets.append(cursor)
        cursor += len(sample)
    path.write_bytes(ftyp + moov(offsets) + _box("mdat", b"".join(samples)))


@pytest.fixture
def movie(tmp_path, monkeypatch):
    monkeypatch.setattr(siv, "_ensure_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(siv, "_selected_mix_encoder", lambda: "aac")
    monkeypatch.setattr(siv, "_layout_cache", {})
    monkeypatch.setattr(siv, "_info_cache", {})
    monkeypatch.setattr(siv, "_metadata_version", 0)
    source = tmp_path / "movie.mp4"
    wav = source.with_suffix(".si.wav")
    wav.write_bytes(b"SI PCM placeholder")
    params = SIMixParams(enabled=True)
    def prepare(codec="hvc1", audio_samples=(b"A", b"BC", b"DEF")):
        _sample_file(source, [b"0123456789", b"video-two", b"last-video"], codec)
        digest = siv._cache_digest(source, wav, params)
        audio = tmp_path / f"{digest}.audio.mp4"
        _sample_file(audio, audio_samples, audio=True)
        return audio
    return source, wav, params, prepare


@pytest.mark.parametrize("codec,expected", [("hvc1", "hevc"), ("hev1", "hevc"), ("avc1", "h264")])
@pytest.mark.parametrize("edit", ["preserve", "remove"])
def test_browse_exact_size_matches_real_layout_without_encoding_or_packet_scan(movie, monkeypatch, codec, expected, edit):
    source, wav, params, prepare = movie
    audio = prepare(codec)
    monkeypatch.setattr(siv, "SI_AUDIO_EDIT_MODE", edit)
    with (patch.object(siv, "build_mixed_audio_sidecar", side_effect=AssertionError("Browse cannot encode")),
          patch.object(siv, "read_media_sample_table", side_effect=AssertionError("Browse cannot scan packets"))):
        info = siv.progressive_si_info(source, wav, params)
    with patch.object(siv, "build_mixed_audio_sidecar", return_value=audio):
        layout = siv.build_progressive_si_virtual_mp4(source, wav, params)
    assert info.content_length == layout.content_length
    assert info.video_codec_name == layout.video_codec_name == expected
    assert siv.progressive_si_metadata_version() == 1
    assert siv.progressive_si_info(source, wav, params) == info
    assert len(b"".join(siv.iter_virtual_range(layout.regions, 0, layout.content_length - 1))) == info.content_length


def test_browse_metadata_follows_audio_replacement_and_mix_settings(movie):
    source, wav, params, prepare = movie
    audio = prepare()
    before = siv.progressive_si_info(source, wav, params)
    _sample_file(audio, [b"much-longer-audio-sample", b"tail"], audio=True)
    after = siv.progressive_si_info(source, wav, params)
    assert after != before
    with patch.object(siv, "build_mixed_audio_sidecar", side_effect=AssertionError("Cannot encode")):
        assert siv.progressive_si_info(source, wav, replace(params, si_delay_seconds=0.7)) is None
        wav.write_bytes(b"changed SI audio")
        assert siv.progressive_si_info(source, wav, params) is None


def test_live_and_virtual_share_dubbing_rules_and_duck_cache_identity(movie):
    from http_app.si_stream import ConfigHolder, SIStreamService
    source, wav, params, prepare = movie
    prepare()
    ordinary = siv._cache_digest(source, wav, params)
    duck = source.with_suffix(".si.duck.wav")
    duck.write_bytes(b"subtitle-span-control")
    custom = replace(params, mix_channel="left", original_volume_percent=70,
                     si_volume_percent=50, si_delay_seconds=0.7, duck_original=False)
    effective, key = resolve_si_mix_inputs(source, custom)
    assert effective == params.dubbing_variant()
    assert key == duck
    service = SIStreamService(config_holder=ConfigHolder(custom))
    with patch.object(service, "current_config", return_value=custom):
        assert service.resolve_stream(source) == (effective, key)
    dubbed = siv._cache_digest(source, wav, custom)
    assert dubbed != ordinary
    assert dubbed == siv._cache_digest(source, wav, params)
    assert siv.progressive_si_info(source, wav, params) is None  # Cannot reuse non-duck audio.
    duck.write_bytes(b"changed subtitle-span-control")
    assert siv._cache_digest(source, wav, custom) != dubbed
    disabled = replace(custom, dub_mode_enabled=False)
    assert resolve_si_mix_inputs(source, disabled) == (disabled, None)
    disabled_digest = siv._cache_digest(source, wav, disabled)
    duck.unlink()
    assert siv._cache_digest(source, wav, params) == ordinary
    assert siv._cache_digest(source, wav, disabled) == disabled_digest
    assert resolve_si_mix_inputs(source, custom) == (custom, None)


@pytest.mark.parametrize("cached_source", [False, True])
@pytest.mark.parametrize("dub_enabled", [False, True])
def test_mixed_cache_uses_dedicated_duck_input_with_source_pipe_and_cached_audio(movie, cached_source, dub_enabled):
    source, wav, params, prepare = movie
    prepare()
    duck = source.with_suffix(".si.duck.wav")
    duck.write_bytes(b"control")
    params = replace(params, mix_channel="left", si_volume_percent=50, si_delay_seconds=0.7,
                     duck_original=False, dub_mode_enabled=dub_enabled)
    # Force a cache miss without changing the waveform inputs.
    source_audio = source.parent / "original.audio.mp4"
    if cached_source:
        source_audio.write_bytes(b"original")
    recorded = []
    def run(cmd, **kwargs):
        recorded.append(cmd)
        source_audio.write_bytes(b"original")
        from pathlib import Path
        Path(cmd[-1]).write_bytes(b"encoded mix")
    with (patch.object(siv, "_source_audio_sidecar_output", return_value=("test-original", source_audio)),
          patch.object(siv, "_use_segmented_aac", return_value=False),
          patch.object(siv, "_run_ffmpeg_sidecar", side_effect=run),
          patch.object(siv, "_run_ffmpeg_sidecar_with_source_audio_pipe", side_effect=run)):
        output = siv.build_mixed_audio_sidecar(source, wav, params)
    assert output.read_bytes() == b"encoded mix"
    cmd, = recorded
    inputs = [cmd[i + 1] for i, token in enumerate(cmd[:-1]) if token == "-i"]
    assert inputs == ([str(source_audio)] if cached_source else ["pipe:0"]) + [str(wav)] + ([str(duck)] if dub_enabled else [])
    expected = params.dubbing_variant() if dub_enabled else params
    assert cmd[cmd.index("-filter_complex") + 1] == expected.filter_string(duck_key_input=dub_enabled)


def test_segmented_duck_input_seeks_to_the_same_time_as_original_and_voice(movie):
    source, wav, params, prepare = movie
    duck = source.with_suffix(".si.duck.wav")
    duck.write_bytes(b"control")
    effective, _ = resolve_si_mix_inputs(source, params)
    segment = siv._MixSegment(index=1, start_frame=60, end_frame=144, encode_start_frame=48, encode_end_frame=144)
    output = source.parent / "segment.mp4"
    def run(cmd, **kwargs):
        output.write_bytes(b"segment")
    with patch.object(siv, "_run_ffmpeg_sidecar", side_effect=run) as encode:
        siv._encode_mix_segment(source, wav, effective, segment, output, cancel_event=None,
                                low_priority=False, duck_key=duck)
    cmd = encode.call_args.args[0]
    inputs = [i for i, token in enumerate(cmd) if token == "-i"]
    assert len(inputs) == 3
    for i in inputs:
        assert cmd[i - 4:i] == ["-ss", "1.024000", "-t", "2.048000"]
    assert "[2:a:0]" in cmd[cmd.index("-filter_complex") + 1]


def test_parallel_mix_passes_resolved_duck_key_and_dubbing_params_to_every_segment(movie):
    source, wav, params, prepare = movie
    duck = source.with_suffix(".si.duck.wav")
    duck.write_bytes(b"control")
    output = source.parent / "parallel.mp4"
    def encode(original, voice, settings, segment, target, **kwargs):
        target.write_bytes(b"segment")
        return target
    def stitch(paths, segments, target, **kwargs):
        target.write_bytes(b"complete")
    with (patch.object(siv, "build_source_audio_sidecar", return_value=source),
          patch.object(siv, "read_media_sample_table", return_value=SimpleNamespace(samples=range(200))),
          patch.object(siv, "_effective_mix_segments", return_value=2),
          patch.object(siv, "_encode_mix_segment", side_effect=encode) as mix,
          patch.object(siv, "_stitch_aac_segments", side_effect=stitch)):
        siv.build_mixed_audio_sidecar_parallel(source, wav, params, output)
    assert mix.call_count == 2
    assert all(call.kwargs["duck_key"] == duck and call.args[2] == params.dubbing_variant() for call in mix.call_args_list)


@pytest.mark.parametrize("block_size", [1, 16, 64])
def test_http_blocks_combine_small_samples_and_bound_large_init(tmp_path, block_size):
    source = tmp_path / "samples.bin"
    source.write_bytes(bytes(range(256)))
    init = b"moov" * 30
    regions = [siv.VirtualRegion.memory(0, init)]
    regions += [siv.VirtualRegion.file(len(init) + i, source, i, 1) for i in range(256)]
    expected = init + source.read_bytes()
    chunks = list(siv.iter_virtual_range(regions, 2, len(expected) - 3, chunk_size=block_size))
    assert b"".join(chunks) == expected[2:-2]
    assert all(len(chunk) == block_size for chunk in chunks[:-1])
    assert 0 < len(chunks[-1]) <= block_size


def test_truncated_source_cannot_silently_end_an_http_body(tmp_path):
    source = tmp_path / "short.bin"
    source.write_bytes(b"short")
    with pytest.raises(EOFError, match="SI source truncated"):
        list(siv.iter_virtual_range([siv.VirtualRegion.file(0, source, 0, 20)], 0, 19))


@pytest.mark.parametrize("codec,profile", [("hev1", "HEVC_MP4_MAIN"), ("avc1", "AVC_MP4_HP_HD_AAC")])
def test_dlna_resource_publishes_exact_cached_size_and_copied_codec(movie, codec, profile):
    source, wav, params, prepare = movie
    prepare(codec)
    info = siv.progressive_si_info(source, wav, params)
    with (patch.object(cds, "get_si_mix", return_value=SimpleNamespace(params=lambda: params)),
          patch.object(cds, "_rel_key", return_value="movie.mp4"),
          patch.object(cds, "prewarm_progressive_si_virtual_mp4") as prewarm,
          patch.object(siv, "build_mixed_audio_sidecar", side_effect=AssertionError("Browse cannot encode"))):
        item = cds._si_file_item(source, "0", 1.0, 8192, 4096, source_size=source.stat().st_size)
        didl = ET.fromstring(cds._didl_for([item]))
    resource = didl.find(".//{urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/}res")
    assert int(resource.attrib["size"]) == info.content_length
    assert profile in resource.attrib["protocolInfo"]
    assert not item["size_estimated"]
    prewarm.assert_called_once_with(source, wav, params, reason="dlna-virtual-file")


def test_browse_cache_refreshes_when_background_layout_is_ready(tmp_path):
    child = SimpleNamespace(is_dir=False, path=tmp_path / "movie.mp4")
    snapshot = SimpleNamespace(key="si-ready", signature="unchanged", children=[child])
    with (patch.object(cds, "get_media_index") as index,
          patch.object(cds, "_folder_id", return_value="0"),
          patch.object(cds, "_dir_items_cache", {}),
          patch.object(cds, "progressive_si_metadata_version", side_effect=[0, 0, 1]),
          patch.object(cds, "_video_items_from_index", side_effect=[[{"size": 123}], [{"size": 456}]]) as build):
        index.return_value.list_directory.return_value = snapshot
        assert cds._children_for_dir(tmp_path)[0]["size"] == 123
        assert cds._children_for_dir(tmp_path)[0]["size"] == 123
        assert cds._children_for_dir(tmp_path)[0]["size"] == 456
        assert build.call_count == 2


def test_si_http_head_and_ranges_use_exact_layout_size_and_stable_bytes(movie):
    source, wav, params, prepare = movie
    audio = prepare("avc1")
    with patch.object(siv, "build_mixed_audio_sidecar", return_value=audio):
        layout = siv.build_progressive_si_virtual_mp4(source, wav, params)
    expected = b"".join(siv.iter_virtual_range(layout.regions, 0, layout.content_length - 1))
    app = FastAPI()
    app.include_router(routes.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            head = await client.head("/media_si/movie.mp4")
            assert int(head.headers["content-length"]) == len(expected)
            assert "AVC_MP4_HP_HD_AAC" in head.headers["contentFeatures.dlna.org"]
            assert not head.content
            for start, end in [(0, 31), (2, len(expected) - 3), (len(expected) - 8, len(expected) - 1)]:
                response = await client.get("/media_si/movie.mp4", headers={"Range": f"bytes={start}-{end}"})
                assert response.status_code == 206
                assert response.headers["content-range"] == f"bytes {start}-{end}/{len(expected)}"
                assert int(response.headers["content-length"]) == end - start + 1
                assert response.content == expected[start:end + 1]
            response = await client.get("/media_si/movie.mp4", headers={"Range": "bytes=0-"})
            assert response.content == expected
            response = await client.get("/media_si/movie.mp4", headers={"Range": f"bytes={len(expected)}-"})
            assert response.status_code == 416
            assert response.headers["content-range"] == f"bytes */{len(expected)}"
    with (patch.object(routes, "_safe_si_video_path", return_value=source),
          patch.object(routes, "_si_virtual_layout", new=AsyncMock(return_value=layout))):
        asyncio.run(check())
