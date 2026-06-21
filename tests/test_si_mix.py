from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import dlna.content_directory as cds
import http_app.routes_media as routes_media
from pipeline.si_virtual_mp4 import VirtualRegion
from ui import settings as settings_module
from http_app.server import create_app
from http_app.si_stream import ConfigHolder, SIStreamService, parse_range_header
from utils import runtime_settings
from utils.si_filter import SIMixParams, build_si_mix_filter


class FakeSession:
    instances: list["FakeSession"] = []
    payload = b"x" * 1024

    def __init__(self, **kwargs) -> None:
        self.video = kwargs["video"]
        self.si_wav = kwargs["si_wav"]
        self.config = kwargs["config"]
        self.start_time = kwargs["start_time"]
        self.byte_cursor = kwargs["start_byte"]
        self.closed = False
        self.offset = 0
        FakeSession.instances.append(self)

    def is_usable(self) -> bool:
        return not self.closed

    def read(self, n: int) -> bytes:
        if self.closed:
            return b""
        chunk = self.payload[self.offset : self.offset + n]
        self.offset += len(chunk)
        self.byte_cursor += len(chunk)
        return chunk

    def discard(self, n: int) -> int:
        skipped = min(max(0, n), len(self.payload) - self.offset)
        self.offset += skipped
        self.byte_cursor += skipped
        return skipped

    def close(self) -> None:
        self.closed = True


class SIMixTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_settings.reset_si_mix_for_test({"enabled": False})
        FakeSession.instances.clear()
        FakeSession.payload = b"x" * 1024

    def test_filter_uses_fixed_first_audio_track_and_ducking(self) -> None:
        filt = build_si_mix_filter("left", 100, 50, 1.0, duck_original=True)
        self.assertIn("[0:a:0]", filt)
        self.assertIn("[1:a:0]", filt)
        self.assertIn("sidechaincompress", filt)
        self.assertIn("[si_track]", filt)

    def test_si_defaults_are_enabled_both_channel_and_full_si_volume(self) -> None:
        params = SIMixParams()
        self.assertTrue(params.enabled)
        self.assertEqual(params.mix_channel, "both")
        self.assertEqual(params.si_volume_percent, 100)

    def test_runtime_params_clamp_supported_choices(self) -> None:
        params = SIMixParams(
            enabled="yes",
            mix_channel="bad",
            original_volume_percent=77,
            si_volume_percent=55,
            si_delay_seconds=1.4,
            duck_original="off",
        )
        self.assertTrue(params.enabled)
        self.assertEqual(params.mix_channel, "both")
        self.assertEqual(params.original_volume_percent, 100)
        self.assertEqual(params.si_volume_percent, 100)
        self.assertEqual(params.si_delay_seconds, 0.0)
        self.assertFalse(params.duck_original)

    def test_parse_range_header_is_lenient(self) -> None:
        self.assertEqual(parse_range_header("bytes=1024-2047"), (1024, 2047, True))
        self.assertEqual(parse_range_header("bytes=1024-"), (1024, None, True))
        self.assertEqual(parse_range_header("bytes=-512"), (0, None, True))
        self.assertEqual(parse_range_header("bad"), (0, None, False))

    def test_open_stream_maps_range_to_time_and_reuses_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            service = SIStreamService(
                config_holder=ConfigHolder(SIMixParams(enabled=True)),
                session_factory=FakeSession,
                seek_cooldown_seconds=0,
            )
            with (
                patch.object(service, "estimate_output_size", return_value=1000),
                patch.object(service, "_duration", return_value=100.0),
            ):
                first = service.open_stream(video, 100, 109, range_requested=True, client_id="client")
                self.assertEqual(first.status_code, 206)
                self.assertEqual(first.start_time, 10.0)
                self.assertEqual(first.content_length, 10)
                self.assertEqual(b"".join(first.chunks), b"x" * 10)

                second = service.open_stream(video, 110, 119, range_requested=True, client_id="client")
                self.assertIs(FakeSession.instances[0], FakeSession.instances[-1])
                self.assertEqual(b"".join(second.chunks), b"x" * 10)

    def test_skybox_startup_probe_does_not_replace_active_zero_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            service = SIStreamService(
                config_holder=ConfigHolder(SIMixParams(enabled=True)),
                session_factory=FakeSession,
                reuse_tolerance_bytes=10,
                seek_cooldown_seconds=0,
            )
            with (
                patch.object(service, "estimate_output_size", return_value=1000),
                patch.object(service, "_duration", return_value=100.0),
            ):
                service.open_stream(video, 0, None, range_requested=True, client_id="client")
                probe, total = service.is_startup_probe_range(
                    video,
                    100,
                    None,
                    client_id="client",
                    user_agent="SKYBOX/2.0.2",
                )
        self.assertTrue(probe)
        self.assertEqual(total, 1000)
        self.assertEqual(len(FakeSession.instances), 1)
        self.assertFalse(FakeSession.instances[0].closed)

    def test_skybox_large_startup_probe_is_ignored_while_zero_stream_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            service = SIStreamService(
                config_holder=ConfigHolder(SIMixParams(enabled=True)),
                session_factory=FakeSession,
                reuse_tolerance_bytes=10,
                seek_cooldown_seconds=0,
            )
            with (
                patch.object(service, "estimate_output_size", return_value=10_000_000_000),
                patch.object(service, "_duration", return_value=3000.0),
            ):
                service.open_stream(video, 0, None, range_requested=True, client_id="client")
                probe, total = service.is_startup_probe_range(
                    video,
                    3_000_000_000,
                    None,
                    client_id="client",
                    user_agent="SKYBOX/2.0.2",
                )
        self.assertTrue(probe)
        self.assertEqual(total, 10_000_000_000)
        self.assertEqual(len(FakeSession.instances), 1)
        self.assertFalse(FakeSession.instances[0].closed)

    def test_short_connection_does_not_close_session_but_eof_does(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            service = SIStreamService(
                config_holder=ConfigHolder(SIMixParams(enabled=True)),
                session_factory=FakeSession,
                seek_cooldown_seconds=0,
            )
            with (
                patch.object(service, "estimate_output_size", return_value=1000),
                patch.object(service, "_duration", return_value=100.0),
            ):
                result = service.open_stream(video, 0, 99, range_requested=True, client_id="client")
                iterator = result.chunks
                self.assertEqual(next(iterator), b"x" * 100)
                iterator.close()
                self.assertFalse(FakeSession.instances[-1].closed)

                FakeSession.payload = b"abc"
                result = service.open_stream(video, 300, 309, range_requested=True, client_id="other")
                self.assertEqual(b"".join(result.chunks), b"abc")
                self.assertTrue(FakeSession.instances[-1].closed)

    def test_control_route_updates_si_runtime(self) -> None:
        client = TestClient(create_app())
        before = client.get("/control/si_mix")
        self.assertEqual(before.status_code, 200)
        response = client.put(
            "/control/si_mix",
            json={
                "enabled": True,
                "mix_channel": "right",
                "original_volume_percent": 80,
                "si_volume_percent": 90,
                "si_delay_seconds": 0.7,
                "duck_original": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["enabled"])
        self.assertEqual(data["mix_channel"], "right")
        self.assertEqual(data["original_volume_percent"], 80)
        self.assertEqual(data["si_delay_seconds"], 0.7)
        self.assertFalse(data["duck_original"])
        self.assertGreater(data["version"], before.json()["version"])

    def test_settings_server_env_contains_si_mix_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(settings_module, "SETTINGS_PATH", root / "ui_settings.json"),
                patch.object(settings_module, "SETTINGS_META_PATH", root / "ui_settings_meta.json"),
            ):
                settings = settings_module.Settings()
                default_env = settings.server_env()
                self.assertEqual(default_env["PT_SI_MIX_ENABLED"], "1")
                self.assertEqual(default_env["PT_SI_MIX_CHANNEL"], "both")
                self.assertEqual(default_env["PT_SI_VOLUME_PERCENT"], "100")

                settings.data["si_enabled"] = True
                settings.data["si_mix_channel"] = "right"
                settings.data["si_original_volume_percent"] = 80
                settings.data["si_volume_percent"] = 90
                settings.data["si_delay_seconds"] = 0.7
                settings.data["si_duck_original"] = False
                env = settings.server_env()
        self.assertEqual(env["PT_SI_MIX_ENABLED"], "1")
        self.assertEqual(env["PT_SI_MIX_CHANNEL"], "right")
        self.assertEqual(env["PT_SI_ORIGINAL_VOLUME_PERCENT"], "80")
        self.assertEqual(env["PT_SI_VOLUME_PERCENT"], "90")
        self.assertEqual(env["PT_SI_DELAY_SECONDS"], "0.7")
        self.assertEqual(env["PT_SI_DUCK_ORIGINAL"], "0")

    def test_media_si_head_uses_virtual_layout_without_opening_legacy_stream(self) -> None:
        class Service:
            opened = False

            def current_config(self):
                return SIMixParams(enabled=True)

            def has_si_source(self, _path):
                return Path("movie.si.wav")

            def estimate_output_size(self, _path):
                return 1000

            def open_stream(self, *_args, **_kwargs):
                self.opened = True
                raise AssertionError("HEAD must not start ffmpeg")

        service = Service()
        layout = SimpleNamespace(
            content_length=1000,
            etag="etag",
            moov_size=123,
            video_samples=7,
            audio_samples=5,
            audio_edit_mode="remove",
        )
        request = SimpleNamespace(state=SimpleNamespace(), client=SimpleNamespace(host="127.0.0.1"))
        with (
            patch.object(routes_media, "_safe_si_video_path", return_value=Path("movie.mp4")),
            patch.object(routes_media, "get_si_stream_service", return_value=service),
            patch.object(routes_media, "build_progressive_si_virtual_mp4", return_value=layout),
        ):
            response = asyncio.run(routes_media.media_si_head(request, "movie.mp4", range="bytes=10-19"))
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.headers["content-length"], "10")
        self.assertEqual(response.headers["content-range"], "bytes 10-19/1000")
        self.assertEqual(response.headers["x-si-transport"], "progressive-virtual")
        self.assertEqual(response.headers["x-si-audio-edit"], "remove")
        self.assertFalse(service.opened)

    def test_media_si_get_serves_virtual_range_without_opening_legacy_stream(self) -> None:
        class Service:
            opened = False

            def current_config(self):
                return SIMixParams(enabled=True)

            def has_si_source(self, _path):
                return Path("movie.si.wav")

            def open_stream(self, *_args, **_kwargs):
                self.opened = True
                raise AssertionError("virtual media_si must not start legacy ffmpeg")

        service = Service()
        request = SimpleNamespace(state=SimpleNamespace(), client=SimpleNamespace(host="127.0.0.1"))
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"0123456789" * 200)
            layout = SimpleNamespace(
                content_length=2000,
                etag="etag",
                moov_size=123,
                video_samples=7,
                audio_samples=5,
                audio_edit_mode="remove",
                regions=(VirtualRegion.memory(0, b"0123456789" * 200),),
            )
            with (
                patch.object(routes_media, "_safe_si_video_path", return_value=video),
                patch.object(routes_media, "get_si_stream_service", return_value=service),
                patch.object(routes_media, "build_progressive_si_virtual_mp4", return_value=layout),
            ):
                async def call_and_collect():
                    resp = await routes_media.media_si_get(
                        request,
                        "movie.mp4",
                        range="bytes=100-115",
                        user_agent="SKYBOX/2.0.2",
                    )
                    payload = b""
                    async for chunk in resp.body_iterator:
                        payload += chunk
                    return resp, payload

                response, body = asyncio.run(call_and_collect())
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.headers["content-length"], "16")
        self.assertEqual(response.headers["content-range"], "bytes 100-115/2000")
        self.assertEqual(response.headers["x-si-transport"], "progressive-virtual")
        self.assertEqual(response.headers["x-si-audio-edit"], "remove")
        self.assertEqual(body, b"0123456789012345")
        self.assertFalse(service.opened)

    def test_dlna_adds_si_directory_container_with_vr_name(self) -> None:
        runtime_settings.reset_si_mix_for_test({"enabled": True})
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            child = SimpleNamespace(
                size=1000,
                video=SimpleNamespace(
                    duration=60.0,
                    fps=24.0,
                    width=3840,
                    height=1920,
                    resolution="3840x1920",
                    backend_verdict="pynv_hevc",
                    probe_error="",
                    mkv_needs_fix=False,
                ),
            )
            with (
                patch.object(cds, "_rel_key", return_value="movie.mp4"),
                patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "none"),
                patch.object(cds, "find_external_subtitles", return_value=[]),
            ):
                items = cds._video_items_from_index(video, "0", child)
        # The [SI] entry is now a realtime directory (chapters + time index),
        # not a single cached item.
        self.assertTrue(items[1].get("container"))
        self.assertEqual(items[1]["id"], "six_ptv10_movie.mp4")
        self.assertEqual(items[1]["title"], "[SI]movie_LR_180_SBS")
        # 60s source -> one quick chapter (t=0) + one "Select Time Index" entry.
        self.assertEqual(items[1]["child_count"], 2)
        self.assertNotIn("url", items[1])

    def test_dlna_si_container_lists_chapters_and_time_index(self) -> None:
        runtime_settings.reset_si_mix_for_test({"enabled": True})
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            with (
                patch.object(cds, "_rel_key", return_value="movie.mp4"),
                patch.object(cds, "_probe_live_directory_context", return_value=(60.0, 3840, 1920, 24.0)),
                patch.object(cds, "estimate_for_media", return_value=(0, 1000, 0)),
            ):
                children = cds._si_chapter_items(video)
        # First child is the "Select Time Index" subdirectory.
        self.assertTrue(children[0].get("container"))
        self.assertEqual(children[0]["id"], "sxi_ptv10_movie.mp4")
        self.assertIn("[SI]movie_LR_180_SBS", children[0]["title"])
        # Then up to N quick-play chapter leaves hitting /si_live.
        self.assertEqual(children[1]["id"], "sic_ptv10_movie.mp4@0")
        self.assertIn("/si_live/movie.mp4", children[1]["url"])
        self.assertIn("t=0", children[1]["url"])
        self.assertEqual(children[1]["mime"], "video/MP2T")
        self.assertEqual(children[1]["passthrough_mode"], "si_mix")

    def test_dlna_si_time_index_leaf_plays_si_live(self) -> None:
        runtime_settings.reset_si_mix_for_test({"enabled": True})
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "movie.mp4"
            video.write_bytes(b"video")
            video.with_suffix(".si.wav").write_bytes(b"si")
            with (
                patch.object(cds, "_rel_key", return_value="movie.mp4"),
                patch.object(cds, "_probe_live_directory_context", return_value=(60.0, 3840, 1920, 24.0)),
                patch.object(cds, "estimate_for_media", return_value=(0, 1000, 0)),
            ):
                index_items = cds._si_time_index_items(video, "index")
                minute_items = cds._si_time_index_items(video, "minute", start=0)
        # 60s -> a single 10-min group, so [SI] shows the minute directories directly.
        self.assertTrue(index_items[0].get("container"))
        self.assertEqual(index_items[0]["id"], "sin_ptv10_movie.mp4@0")
        # Each 5s point is a playable leaf hitting the realtime MPEG-TS route.
        first_leaf = minute_items[0]
        self.assertEqual(first_leaf["id"], "sit_ptv10_movie.mp4@0")
        self.assertIn("/si_live/movie.mp4", first_leaf["url"])
        self.assertIn("t=0", first_leaf["url"])
        self.assertEqual(first_leaf["mime"], "video/MP2T")
        self.assertEqual(first_leaf["passthrough_mode"], "si_mix")


if __name__ == "__main__":
    unittest.main()
