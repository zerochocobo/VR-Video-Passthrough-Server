from __future__ import annotations

import unittest
import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import http_app.routes_media as routes_media
from pipeline.pynv_stream import _drain_async_queue_nowait


class ProbeCacheTests(unittest.TestCase):
    def test_probe_cache_is_bounded_by_total_limit(self) -> None:
        original_cache = dict(routes_media._probe_cache)
        try:
            routes_media._probe_cache.clear()
            with (
                patch.object(routes_media, "_PROBE_CACHE_LIMIT", 10),
                patch.object(routes_media, "_PROBE_CACHE_TOTAL_LIMIT", 25),
            ):
                routes_media._set_probe_cache_locked("a", b"a" * 10)
                routes_media._set_probe_cache_locked("b", b"b" * 10)
                routes_media._set_probe_cache_locked("c", b"c" * 10)

            self.assertLessEqual(sum(len(v) for v in routes_media._probe_cache.values()), 25)
            self.assertNotIn("a", routes_media._probe_cache)
            self.assertIn("c", routes_media._probe_cache)
        finally:
            routes_media._probe_cache.clear()
            routes_media._probe_cache.update(original_cache)


class SourceMediaRouteTests(unittest.TestCase):
    def test_media_type_for_images_uses_real_image_mime(self) -> None:
        self.assertEqual(routes_media._media_type_for_path(Path("photo.jpg")), "image/jpeg")
        self.assertEqual(routes_media._media_type_for_path(Path("photo.png")), "image/png")
        self.assertEqual(routes_media._media_type_for_path(Path("movie.mp4")), "video/mp4")

    def test_media_head_serves_image_content_type(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "photo.png"
            path.write_bytes(b"abcdef")
            with (
                patch.object(routes_media, "_safe_media_path", return_value=path),
                patch.object(routes_media, "annotate_request", return_value=None),
            ):
                response = asyncio.run(routes_media.media_head(SimpleNamespace(), "photo.png", range=None))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["content-length"], "6")


class ExistingPassthroughStrategyCharacterizationTests(unittest.TestCase):
    def test_owner_kind_and_preempt_matrix_preserve_current_behavior(self) -> None:
        self.assertEqual(routes_media._owner_kind((r"G:\VR\movie.mp4", "client")), "")
        self.assertEqual(routes_media._owner_kind(("live", "client", "nplayer")), "nplayer")

        self.assertTrue(routes_media._can_preempt_owner(("live", "client", "nplayer"), ("live", "client", "nplayer")))
        self.assertTrue(routes_media._can_preempt_owner(("live", "client", "4xvr"), ("live", "client", "4xvr")))
        # libmpv same-owner preempt is enabled; same-live_key duplicates are
        # caught earlier by the libmpv startup debounce in
        # passthrough_live_get, so reaching the slot path means a different
        # request that should win (typically a different t= chapter probe).
        self.assertTrue(routes_media._can_preempt_owner(("live", "client", "libmpv"), ("live", "client", "libmpv")))
        self.assertTrue(routes_media._can_preempt_owner(("live", "client", "vlc"), ("live", "client", "nplayer")))
        self.assertFalse(routes_media._can_preempt_owner(("live", "client-a", "nplayer"), ("live", "client-b", "nplayer")))
        self.assertTrue(routes_media._can_preempt_owner(("live", "client", "lavf"), ("live", "client", "vlc")))
        self.assertTrue(routes_media._can_preempt_owner((r"G:\VR\movie.mp4", "client", "vlc"), (r"G:\VR\movie.mp4", "client", "libmpv")))
        self.assertTrue(routes_media._can_preempt_owner((r"G:\VR\movie.mp4", "client", "seek"), (r"G:\VR\movie.mp4", "client", "seek")))
        self.assertFalse(routes_media._can_preempt_owner((r"G:\VR\movie-a.mp4", "client", "vlc"), (r"G:\VR\movie-b.mp4", "client", "libmpv")))

    def test_seek_test_same_client_preempt_allows_switching_files(self) -> None:
        self.assertTrue(
            routes_media._can_preempt_same_client_for_seek_test(
                (r"G:\VR\movie-a.mp4", "client", "seek"),
                (r"G:\VR\movie-b.mp4", "client", "seek"),
            )
        )
        self.assertTrue(
            routes_media._can_preempt_same_client_for_seek_test(
                (r"G:\VR\movie-a.mp4", "client", "seek"),
                ("live", "client", "4xvr"),
            )
        )
        self.assertFalse(
            routes_media._can_preempt_same_client_for_seek_test(
                (r"G:\VR\movie-a.mp4", "client-a", "seek"),
                (r"G:\VR\movie-b.mp4", "client-b", "seek"),
            )
        )

    def test_probe_range_helpers_preserve_current_thresholds(self) -> None:
        self.assertTrue(routes_media._is_small_probe_range(routes_media.ByteRange(start=0, end=64 * 1024 - 1, total=10_000_000)))
        self.assertFalse(routes_media._is_small_probe_range(routes_media.ByteRange(start=0, end=64 * 1024, total=10_000_000)))

        self.assertTrue(routes_media._is_tail_probe_range(routes_media.ByteRange(start=950_000, end=950_511, total=1_000_000)))
        self.assertFalse(routes_media._is_tail_probe_range(routes_media.ByteRange(start=949_999, end=950_510, total=1_000_000)))
        self.assertTrue(routes_media._is_tail_probe_range(routes_media.ByteRange(start=7_647_068_160, end=7_648_096_887, total=7_648_096_888)))
        self.assertFalse(routes_media._is_tail_probe_range(routes_media.ByteRange(start=95_000_000, end=97_200_000, total=100_000_000)))

    def test_seek_route_guard_separates_master_and_profile_policy(self) -> None:
        with patch.object(routes_media, "PASSTHROUGH_SEEK_ENABLED", False):
            allowed, reason, profile = routes_media._seek_route_allowed("VLC/3.0")
        self.assertFalse(allowed)
        self.assertEqual(reason, "disabled")
        self.assertEqual(profile, "vlc")

        with (
            patch.object(routes_media, "PASSTHROUGH_SEEK_ENABLED", True),
            patch.object(routes_media, "PASSTHROUGH_SEEK_ROUTE_POLICY", "profile"),
            patch.object(routes_media, "PASSTHROUGH_SEEK_PROFILES", ("vlc",)),
        ):
            allowed, reason, profile = routes_media._seek_route_allowed("VLC/3.0")
            blocked, blocked_reason, blocked_profile = routes_media._seek_route_allowed("AVProMobileVideo/2.0")

        self.assertTrue(allowed)
        self.assertEqual(reason, "profile_allowed")
        self.assertEqual(profile, "vlc")
        self.assertFalse(blocked)
        self.assertEqual(blocked_reason, "profile_avpro_blocked")
        self.assertEqual(blocked_profile, "avpro")

    def test_seek_declared_size_is_cached_per_client(self) -> None:
        source = Path("movie.mp4")
        original_cache = dict(routes_media._seek_declared_size_cache)
        routes_media._seek_declared_size_cache.clear()
        try:
            with (
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=10, st_mtime_ns=20)),
                patch.object(routes_media, "_estimated_passthrough_size", side_effect=[1000, 2000]),
                patch.object(routes_media, "PASSTHROUGH_SEEK_HEADER_BYTES", 100),
            ):
                first = routes_media._estimated_seek_passthrough_size(source, 60.0, "hevc", "client")
                second = routes_media._estimated_seek_passthrough_size(source, 60.0, "hevc", "client")
                other_client = routes_media._estimated_seek_passthrough_size(source, 60.0, "hevc", "other")

            self.assertEqual(first, 1100)
            self.assertEqual(second, 1100)
            self.assertEqual(other_client, 2100)
        finally:
            routes_media._seek_declared_size_cache.clear()
            routes_media._seek_declared_size_cache.update(original_cache)

    def test_seek_headers_advertise_byte_and_time_seek(self) -> None:
        info = SimpleNamespace(fps=30.0)
        estimate = SimpleNamespace(source="test")
        with patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, estimate)):
            headers = routes_media._seek_headers(
                path=Path("movie.mp4"),
                info=info,
                duration=60.0,
                codec="hevc",
                total=10_000,
                start_sec=5.0,
                range_header="bytes=100-",
                include_length=True,
            )

        self.assertIn("DLNA.ORG_OP=11", headers["contentFeatures.dlna.org"])
        self.assertIn("DLNA.ORG_CI=0", headers["contentFeatures.dlna.org"])
        self.assertIn("DLNA.ORG_FLAGS=01F00000000000000000000000000000", headers["contentFeatures.dlna.org"])
        self.assertEqual(headers["transferMode.dlna.org"], "Interactive")
        self.assertEqual(headers["Cache-Control"], "no-cache")
        self.assertEqual(headers["availableSeekRange.dlna.org"], "1 npt=0.000-60.000")
        self.assertEqual(headers["Accept-Ranges"], "bytes")

    def test_seek_headers_can_advertise_true_fmp4_experiment(self) -> None:
        info = SimpleNamespace(fps=30.0)
        estimate = SimpleNamespace(source="test")
        with patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, estimate)):
            headers = routes_media._seek_headers(
                path=Path("movie.mp4"),
                info=info,
                duration=60.0,
                codec="hevc",
                total=10_000,
                start_sec=5.0,
                range_header="bytes=100-",
                include_length=True,
                container="mp4",
            )

        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertIn("DLNA.ORG_PN=HEVC_MP4_MAIN", headers["contentFeatures.dlna.org"])
        self.assertEqual(headers["transferMode.dlna.org"], "Interactive")

    def test_seek_route_suffix_selects_container_without_changing_media_key(self) -> None:
        self.assertEqual(
            routes_media._split_seek_route_name("folder/movie.mp4.seek.ts"),
            ("folder/movie.mp4", "mpegts"),
        )
        self.assertEqual(
            routes_media._split_seek_route_name("folder/movie.mp4.seek.mp4"),
            ("folder/movie.mp4", "mp4"),
        )
        self.assertEqual(
            routes_media._split_seek_route_name("folder/movie.mp4"),
            ("folder/movie.mp4", None),
        )

    def test_seek_diag_headers_include_head_get_common_fields(self) -> None:
        headers: dict[str, str] = {}
        mapped = SimpleNamespace(ratio=0.5, time_sec=30.0, gop_seconds=2.0)

        routes_media._apply_seek_diag_headers(
            headers,
            start_sec=28.0,
            output_mode="green",
            mapped=mapped,
        )

        self.assertEqual(headers["X-Passthrough-Mode"], "seek-mpegts-green")
        self.assertEqual(headers["X-Passthrough-Seek-Time"], "28.000")
        self.assertEqual(headers["X-Passthrough-Seek-Ratio"], "0.500000")
        self.assertEqual(headers["X-Passthrough-Seek-Raw-Time"], "30.000")
        self.assertEqual(headers["X-Passthrough-Seek-Gop"], "2.000")

    def test_seek_head_prefix_range_gets_mapping_diag_headers(self) -> None:
        request = SimpleNamespace(
            headers={"user-agent": "VLC/3.0"},
            client=SimpleNamespace(host="client"),
        )
        info = SimpleNamespace(duration=60.0, fps=30.0)
        estimate = SimpleNamespace(source="test")
        path = Path("movie.mp4")

        with (
            patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None)),
            patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "vlc")),
            patch.object(routes_media, "annotate_request", return_value=None),
            patch.object(routes_media, "probe_cached", return_value=info),
            patch.object(routes_media, "_estimated_seek_passthrough_size", return_value=10_000),
            patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, estimate)),
            patch.object(routes_media, "PASSTHROUGH_SEEK_HEADER_BYTES", 2_000),
        ):
            response = asyncio.run(
                routes_media.passthrough_seek_head(
                    request,
                    "movie.mp4",
                    mode=None,
                    range_header="bytes=100-200",
                    time_seek_range=None,
                    get_content_features=None,
                    transfer_mode=None,
                )
            )

        self.assertEqual(response.headers["x-passthrough-seek-ratio"], "0.000000")
        self.assertEqual(response.headers["x-passthrough-seek-raw-time"], "0.000")
        self.assertEqual(response.headers["x-passthrough-seek-gop"], "2.000")

    def test_seek_blocked_response_uses_404_only_for_master_disable(self) -> None:
        disabled = routes_media._seek_blocked_response("disabled")
        profile_blocked = routes_media._seek_blocked_response("profile_avpro_blocked")

        self.assertEqual(disabled.status_code, 404)
        self.assertEqual(profile_blocked.status_code, 403)
        self.assertIn(b"/passthrough_live", profile_blocked.body)

    def test_seek_prefix_cache_limit_uses_reserved_header_bytes(self) -> None:
        with (
            patch.object(routes_media, "_PROBE_CACHE_LIMIT", 16 * 1024 * 1024),
            patch.object(routes_media, "PASSTHROUGH_SEEK_HEADER_BYTES", 2 * 1024 * 1024),
        ):
            self.assertEqual(routes_media._seek_prefix_cache_limit(), 2 * 1024 * 1024)

    def test_seek_prefix_cache_hit_serves_real_cached_slice(self) -> None:
        async def run():
            return await routes_media._serve_seek_prefix_or_retry(
                rid=1,
                path=Path("movie.mp4"),
                media_type="video/MP2T",
                headers={"Content-Type": "video/MP2T"},
                byte_range=routes_media.ByteRange(start=1, end=3, total=10),
                probe_key="seek-test",
                range_header="bytes=1-3",
            )

        original_cache = dict(routes_media._probe_cache)
        routes_media._probe_cache.clear()
        try:
            routes_media._probe_cache["seek-test"] = b"abcdef"
            response = asyncio.run(run())
        finally:
            routes_media._probe_cache.clear()
            routes_media._probe_cache.update(original_cache)

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.body, b"bcd")
        self.assertEqual(response.headers["x-passthrough-probe-source"], "seek-prefix-cache")
        self.assertEqual(response.headers["content-range"], "bytes 1-3/10")

    def test_seek_prefix_cache_requires_full_requested_slice(self) -> None:
        async def run():
            return await routes_media._serve_seek_prefix_or_retry(
                rid=1,
                path=Path("movie.mp4"),
                media_type="video/MP2T",
                headers={"Content-Type": "video/MP2T"},
                byte_range=routes_media.ByteRange(start=1, end=5, total=10),
                probe_key="seek-test-short",
                range_header="bytes=1-5",
            )

        original_cache = dict(routes_media._probe_cache)
        routes_media._probe_cache.clear()
        try:
            routes_media._probe_cache["seek-test-short"] = b"abc"
            response = asyncio.run(run())
        finally:
            routes_media._probe_cache.clear()
            routes_media._probe_cache.update(original_cache)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["x-passthrough-probe-source"], "seek-prefix-cache-not-ready")

    def test_seek_prefix_cache_miss_returns_retry_after(self) -> None:
        async def run():
            return await routes_media._serve_seek_prefix_or_retry(
                rid=1,
                path=Path("movie.mp4"),
                media_type="video/MP2T",
                headers={"Content-Type": "video/MP2T"},
                byte_range=routes_media.ByteRange(start=1, end=3, total=10),
                probe_key="seek-test-miss",
                range_header="bytes=1-3",
            )

        original_cache = dict(routes_media._probe_cache)
        routes_media._probe_cache.clear()
        try:
            response = asyncio.run(run())
        finally:
            routes_media._probe_cache.clear()
            routes_media._probe_cache.update(original_cache)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "1")
        self.assertEqual(response.headers["content-type"], "video/MP2T")
        self.assertEqual(response.headers["x-passthrough-probe-source"], "seek-prefix-cache-not-ready")

    def test_seek_prefix_crossing_open_range_splices_cached_prefix_then_streams(self) -> None:
        class _FakeStream:
            bytes_emitted = 20
            frames_produced = 1
            output_fps = 1.0

            async def iter_bytes(self):
                yield b"ABCDE"
                yield b"FGHIJ"
                yield b"KLMNOPQRST"

            def close(self) -> None:
                pass

        fake_stream = _FakeStream()

        class _FakeRequest:
            headers = {"user-agent": "nPlayer/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        request = _FakeRequest()
        info = SimpleNamespace(duration=20.0, fps=1.0)
        estimate = SimpleNamespace(source="test")
        path = Path("movie.mp4")

        async def run() -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                request,
                "movie.mp4",
                mode=None,
                range_header="bytes=3-",
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            async for chunk in response.body_iterator:
                body.extend(chunk)
            return response.status_code, dict(response.headers), bytes(body)

        with (
            patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None)),
            patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "nplayer")),
            patch.object(routes_media, "annotate_request", return_value=None),
            patch.object(routes_media, "probe_cached", return_value=info),
            patch.object(routes_media, "_estimated_seek_passthrough_size", return_value=20),
            patch.object(routes_media, "_seek_probe_cache_key", return_value="seek-crossing"),
            patch.object(routes_media, "estimate_for_media", return_value=(20, 8, estimate)),
            patch.object(routes_media, "_select_passthrough_stream", return_value=(fake_stream, "fake", "pynv_hevc")),
            patch.object(routes_media, "acquire_matter", return_value=object()),
            patch.object(routes_media, "release_matter", return_value=None),
            patch.object(routes_media, "record_actual_bps", return_value=None),
            patch.object(routes_media, "PASSTHROUGH_BUSY_WAIT_SEC", 0),
            patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1),
            patch.object(routes_media, "PASSTHROUGH_PAD_TO_LENGTH", False),
            patch.object(routes_media, "PASSTHROUGH_SEEK_HEADER_BYTES", 5),
        ):
            original_cache = dict(routes_media._probe_cache)
            original_streams = dict(routes_media._active_streams)
            original_started = dict(routes_media._active_started)
            original_matter = dict(routes_media._active_matter)
            routes_media._probe_cache.clear()
            routes_media._probe_cache["seek-crossing"] = b"abcde"
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            try:
                status_code, headers, body = asyncio.run(run())
            finally:
                routes_media._probe_cache.clear()
                routes_media._probe_cache.update(original_cache)
                routes_media._active_streams.clear()
                routes_media._active_started.clear()
                routes_media._active_matter.clear()
                routes_media._active_streams.update(original_streams)
                routes_media._active_started.update(original_started)
                routes_media._active_matter.update(original_matter)

        self.assertEqual(status_code, 206)
        self.assertEqual(headers["content-range"], "bytes 3-19/20")
        self.assertEqual(headers["content-length"], "17")
        self.assertEqual(headers["x-passthrough-probe-source"], "seek-prefix-cache-crossing")
        self.assertEqual(body, b"deFGHIJKLMNOPQRST")

    def test_seek_prefix_crossing_open_range_requires_cached_header(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "nPlayer/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        request = _FakeRequest()
        info = SimpleNamespace(duration=20.0, fps=1.0)
        estimate = SimpleNamespace(source="test")
        path = Path("movie.mp4")

        async def run() -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                request,
                "movie.mp4",
                mode=None,
                range_header="bytes=3-",
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            return response.status_code, dict(response.headers), response.body

        with (
            patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None)),
            patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "nplayer")),
            patch.object(routes_media, "annotate_request", return_value=None),
            patch.object(routes_media, "probe_cached", return_value=info),
            patch.object(routes_media, "_estimated_seek_passthrough_size", return_value=20),
            patch.object(routes_media, "_seek_probe_cache_key", return_value="seek-crossing-miss"),
            patch.object(routes_media, "estimate_for_media", return_value=(20, 8, estimate)),
            patch.object(routes_media, "_select_passthrough_stream", side_effect=AssertionError("stream should not start")),
            patch.object(routes_media, "PASSTHROUGH_SEEK_HEADER_BYTES", 5),
            patch.object(routes_media, "_PREFIX_CACHE_WAIT_SEC", 0),
        ):
            original_cache = dict(routes_media._probe_cache)
            routes_media._probe_cache.clear()
            routes_media._probe_cache["seek-crossing-miss"] = b"abc"
            try:
                status_code, headers, body = asyncio.run(run())
            finally:
                routes_media._probe_cache.clear()
                routes_media._probe_cache.update(original_cache)

        self.assertEqual(status_code, 503)
        self.assertEqual(body, b"seek prefix cache not ready")
        self.assertEqual(headers["retry-after"], "1")
        self.assertEqual(headers["x-passthrough-probe-source"], "seek-prefix-cache-not-ready")

    def test_seek_stream_close_releases_active_slot(self) -> None:
        events: list[str] = []
        matter = object()

        class _FakeStream:
            bytes_emitted = 4
            frames_produced = 1
            output_fps = 1.0

            async def iter_bytes(self):
                yield b"abcd"
                await asyncio.sleep(60)

            def close(self) -> None:
                events.append("close")

        fake_stream = _FakeStream()

        class _FakeRequest:
            headers = {"user-agent": "nPlayer/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        request = _FakeRequest()
        info = SimpleNamespace(duration=10.0, fps=1.0)
        estimate = SimpleNamespace(source="test")
        path = Path("movie.mp4")
        released: list[object] = []

        async def run() -> bytes:
            response = await routes_media.passthrough_seek_get(
                request,
                "movie.mp4",
                mode=None,
                range_header=None,
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            iterator = response.body_iterator
            first = await iterator.__anext__()
            await iterator.aclose()
            return first

        with (
            patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None)),
            patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "nplayer")),
            patch.object(routes_media, "annotate_request", return_value=None),
            patch.object(routes_media, "probe_cached", return_value=info),
            patch.object(routes_media, "_estimated_seek_passthrough_size", return_value=10),
            patch.object(routes_media, "estimate_for_media", return_value=(10, 8, estimate)),
            patch.object(routes_media, "_select_passthrough_stream", return_value=(fake_stream, "fake", "pynv_hevc")),
            patch.object(routes_media, "acquire_matter", return_value=matter),
            patch.object(routes_media, "release_matter", side_effect=released.append),
            patch.object(routes_media, "PASSTHROUGH_BUSY_WAIT_SEC", 0),
            patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1),
            patch.object(routes_media, "PASSTHROUGH_PAD_TO_LENGTH", False),
            patch.object(routes_media, "PASSTHROUGH_SEEK_HEADER_BYTES", 2),
        ):
            original_streams = dict(routes_media._active_streams)
            original_started = dict(routes_media._active_started)
            original_matter = dict(routes_media._active_matter)
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            try:
                first = asyncio.run(run())
            finally:
                routes_media._active_streams.clear()
                routes_media._active_started.clear()
                routes_media._active_matter.clear()
                routes_media._active_streams.update(original_streams)
                routes_media._active_started.update(original_started)
                routes_media._active_matter.update(original_matter)

        self.assertEqual(first, b"abcd")
        self.assertIn("close", events)
        self.assertEqual(released, [matter])

    def test_passthrough_prefix_cache_writes_once_at_limit_and_finally_skips_duplicate(self) -> None:
        class _FakeStream:
            bytes_emitted = 8
            frames_produced = 1
            output_fps = 1.0

            async def iter_bytes(self):
                yield b"aa"
                yield b"bb"
                yield b"cc"
                yield b"dd"

            def close(self) -> None:
                pass

        fake_stream = _FakeStream()

        class _FakeRequest:
            headers: dict[str, str] = {}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        request = _FakeRequest()
        info = SimpleNamespace(duration=8.0, codec_name="hevc")
        estimate = SimpleNamespace(source="test")
        path = Path("movie.mp4")
        writes: list[bytes] = []

        class _NoopLock:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def run() -> None:
            response = await routes_media.passthrough_get(
                request,
                "movie.mp4",
                t=0.0,
                range_header="bytes=0-",
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            async for _ in response.body_iterator:
                pass

        with (
            patch.object(routes_media, "_safe_video_path", return_value=path),
            patch.object(routes_media, "annotate_request", return_value=None),
            patch.object(routes_media, "probe_cached", return_value=info),
            patch.object(routes_media, "_passthrough_estimate_codec", return_value="hevc"),
            patch.object(routes_media, "_passthrough_backend_verdict", return_value="pynv_hevc"),
            patch.object(routes_media, "_range_unsatisfiable", return_value=False),
            patch.object(routes_media, "_estimated_passthrough_size", return_value=128 * 1024),
            patch.object(routes_media, "estimate_for_media", return_value=(128 * 1024, 2000, estimate)),
            patch.object(routes_media, "_select_passthrough_stream", return_value=(fake_stream, "fake", "pynv_hevc")),
            patch.object(routes_media, "acquire_matter", return_value=object()),
            patch.object(routes_media, "release_matter", return_value=None),
            patch.object(routes_media, "record_actual_bps", return_value=None),
            patch.object(routes_media, "PASSTHROUGH_SEEK_MODE", "bytes"),
            patch.object(routes_media, "PASSTHROUGH_PAD_TO_LENGTH", False),
            patch.object(routes_media, "PASSTHROUGH_BUSY_WAIT_SEC", 0),
            patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1),
            patch.object(routes_media, "_PROBE_CACHE_LIMIT", 4),
            patch.object(routes_media, "_probe_cache_lock", _NoopLock()),
            patch.object(routes_media, "_set_probe_cache_locked", side_effect=lambda key, data: writes.append(data)),
        ):
            original_streams = dict(routes_media._active_streams)
            original_started = dict(routes_media._active_started)
            original_matter = dict(routes_media._active_matter)
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            try:
                asyncio.run(run())
            finally:
                routes_media._active_streams.clear()
                routes_media._active_started.clear()
                routes_media._active_matter.clear()
                routes_media._active_streams.update(original_streams)
                routes_media._active_started.update(original_started)
                routes_media._active_matter.update(original_matter)

        self.assertEqual(writes, [b"aabb"])


class LiveQueueTests(unittest.TestCase):
    def test_drain_live_queue_preserves_end_marker(self) -> None:
        async def run() -> tuple[int, int, bool, object]:
            queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=4)
            queue.put_nowait(b"abc")
            queue.put_nowait(b"defg")
            queue.put_nowait(routes_media._LIVE_END)
            dropped = routes_media._drain_live_queue_nowait(queue)
            return (*dropped, queue.get_nowait())

        chunks, bytes_dropped, saw_end, marker = asyncio.run(run())
        self.assertEqual(chunks, 2)
        self.assertEqual(bytes_dropped, 7)
        self.assertTrue(saw_end)
        self.assertIs(marker, routes_media._LIVE_END)

    def test_drain_pynv_queue_preserves_sentinel(self) -> None:
        async def run() -> tuple[int, int, bool, object]:
            queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=4)
            queue.put_nowait(b"abc")
            queue.put_nowait(b"defg")
            queue.put_nowait(None)
            dropped = _drain_async_queue_nowait(queue)
            return (*dropped, queue.get_nowait())

        chunks, bytes_dropped, saw_sentinel, marker = asyncio.run(run())
        self.assertEqual(chunks, 2)
        self.assertEqual(bytes_dropped, 7)
        self.assertTrue(saw_sentinel)
        self.assertIsNone(marker)


class ReplaceActiveSlotMatterReleaseTests(unittest.TestCase):
    """Regression: _replace_active_slot must return Matter to pool when the
    old key has already been preempted, otherwise the 409 paths leak.
    """

    def _run(self, coro):
        return asyncio.run(coro)

    def test_take_active_slot_can_preempt_same_client_when_seek_testing(self) -> None:
        old_stream = object()
        new_stream = object()
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_streams[old_stream] = (r"G:\VR\old.mp4", "client", "seek")

        try:
            with patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1):
                result = self._run(
                    routes_media._take_active_slot(
                        new_stream,
                        "test",
                        (r"G:\VR\new.mp4", "client", "seek"),
                        allow_same_owner_preempt=False,
                        allow_same_client_preempt=True,
                    )
                )
            self.assertIs(result, old_stream)
            self.assertIn(new_stream, routes_media._active_streams)
            self.assertNotIn(old_stream, routes_media._active_streams)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_close_preempted_stream_releases_matter_after_close(self) -> None:
        events: list[str] = []
        sentinel_matter = object()

        class _OldStream:
            def close(self) -> None:
                events.append("close")

        old_stream = _OldStream()
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_matter[old_stream] = sentinel_matter

        def _record_release(m: object) -> None:
            events.append("release")

        try:
            with patch.object(routes_media, "release_matter", side_effect=_record_release):
                self._run(routes_media._close_preempted_stream(old_stream, "test"))
            self.assertEqual(events, ["close", "release"])
            self.assertNotIn(old_stream, routes_media._active_matter)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_close_preempted_stream_releases_matter_when_close_fails(self) -> None:
        events: list[str] = []
        sentinel_matter = object()

        class _BadStream:
            def close(self) -> None:
                events.append("close")
                raise RuntimeError("simulated close failure")

        old_stream = _BadStream()
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_matter[old_stream] = sentinel_matter

        def _record_release(m: object) -> None:
            events.append("release")

        try:
            with patch.object(routes_media, "release_matter", side_effect=_record_release):
                self._run(routes_media._close_preempted_stream(old_stream, "test"))
            self.assertEqual(events, ["close", "release"])
            self.assertNotIn(old_stream, routes_media._active_matter)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_close_preempted_slot_token_releases_bound_matter(self) -> None:
        sentinel_matter = object()
        released: list[object] = []

        slot_token = object()
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_matter[slot_token] = sentinel_matter

        try:
            with patch.object(routes_media, "release_matter", side_effect=released.append):
                self._run(routes_media._close_preempted_stream(slot_token, "test"))
            self.assertEqual(released, [sentinel_matter])
            self.assertNotIn(slot_token, routes_media._active_matter)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_failure_path_releases_orphan_matter(self) -> None:
        sentinel_matter = object()
        released: list[object] = []

        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        slot_token = object()
        new_stream = object()
        # Simulate the post-preempt state: slot_token already evicted from
        # _active_streams by a competing _take_active_slot, but _active_matter
        # still holds the matter that the original acquirer bound.
        routes_media._active_matter[slot_token] = sentinel_matter

        try:
            with patch.object(routes_media, "release_matter", side_effect=released.append):
                ok = self._run(routes_media._replace_active_slot(slot_token, new_stream))
            self.assertFalse(ok)
            self.assertEqual(released, [sentinel_matter])
            self.assertNotIn(slot_token, routes_media._active_matter)
            self.assertNotIn(new_stream, routes_media._active_streams)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_failure_path_closes_new_stream_before_releasing_slot_token_matter(self) -> None:
        events: list[str] = []
        sentinel_matter = object()

        class _NewStream:
            def close(self) -> None:
                events.append("close_new")

        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        slot_token = object()
        new_stream = _NewStream()
        routes_media._active_matter[slot_token] = sentinel_matter

        def _record_release(m: object) -> None:
            events.append("release")

        try:
            with patch.object(routes_media, "release_matter", side_effect=_record_release):
                ok = self._run(
                    routes_media._replace_active_slot(
                        slot_token,
                        new_stream,
                        close_on_failure=new_stream,
                    )
                )
            self.assertFalse(ok)
            self.assertEqual(events, ["close_new", "release"])
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_failure_path_closes_old_stream_before_releasing_matter(self) -> None:
        """When old_stream has close(), it must run BEFORE release_matter so
        a concurrent acquirer cannot pick up a Matter still in use by the
        old worker.
        """
        events: list[str] = []
        sentinel_matter = object()

        class _OldStream:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                # Tag the moment close runs so we can assert ordering against
                # release_matter below.
                events.append("close")
                self.closed = True

        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        old_stream = _OldStream()
        new_stream = object()
        # Post-preempt state: old_stream's matter is still tracked, but
        # _active_streams[old_stream] is gone.
        routes_media._active_matter[old_stream] = sentinel_matter

        def _record_release(m: object) -> None:
            events.append("release")

        try:
            with patch.object(routes_media, "release_matter", side_effect=_record_release):
                ok = self._run(routes_media._replace_active_slot(old_stream, new_stream))
            self.assertFalse(ok)
            self.assertTrue(old_stream.closed, "old_stream.close() must run on failure path")
            self.assertEqual(events, ["close", "release"], "close must precede release_matter")
            self.assertNotIn(old_stream, routes_media._active_matter)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_failure_path_release_swallows_close_exception(self) -> None:
        """Even if old_stream.close() raises, the Matter must still be
        released; otherwise a misbehaving close() would leak the pool.
        """
        events: list[str] = []
        sentinel_matter = object()

        class _BadStream:
            def close(self) -> None:
                events.append("close")
                raise RuntimeError("simulated close failure")

        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        old_stream = _BadStream()
        new_stream = object()
        routes_media._active_matter[old_stream] = sentinel_matter

        def _record_release(m: object) -> None:
            events.append("release")

        try:
            with patch.object(routes_media, "release_matter", side_effect=_record_release):
                ok = self._run(routes_media._replace_active_slot(old_stream, new_stream))
            self.assertFalse(ok)
            self.assertEqual(events, ["close", "release"])
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_success_path_migrates_matter_without_release(self) -> None:
        sentinel_matter = object()
        released: list[object] = []

        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        slot_token = object()
        new_stream = object()
        # Live slot present; matter bound. Expect migration to new_stream key
        # and no release.
        routes_media._active_streams[slot_token] = ("path", "client")
        routes_media._active_started[slot_token] = 0.0
        routes_media._active_matter[slot_token] = sentinel_matter

        try:
            with patch.object(routes_media, "release_matter", side_effect=released.append):
                ok = self._run(routes_media._replace_active_slot(slot_token, new_stream))
            self.assertTrue(ok)
            self.assertEqual(released, [])
            self.assertNotIn(slot_token, routes_media._active_matter)
            self.assertIs(routes_media._active_matter[new_stream], sentinel_matter)
            self.assertIn(new_stream, routes_media._active_streams)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)


class LiveProfileTests(unittest.TestCase):
    def test_quest_dalvik_uses_4xvr_live_profile(self) -> None:
        ua = "Dalvik/2.1.0 (Linux; U; Android 14; Quest 3 Build/UP1A.231005.007.A1)"
        self.assertEqual(routes_media._live_response_profile(ua), "4xvr")

    def test_known_profiles_still_win(self) -> None:
        self.assertEqual(routes_media._live_response_profile("nPlayer/3.12"), "nplayer")
        self.assertEqual(routes_media._live_response_profile("libmpv Android"), "libmpv")
        self.assertEqual(routes_media._live_response_profile("com.heresphere.vrvideoplayerdemo/29 Cronet/95.0"), "4xvr")
        self.assertEqual(routes_media._live_response_profile("VLC/3.0"), "vlc")

    def test_4xvr_live_owner_can_preempt_same_device(self) -> None:
        owner = ("live", "192.168.31.112", "4xvr")
        self.assertTrue(routes_media._can_preempt_owner(owner, owner))

    def test_libmpv_live_owner_can_preempt_same_device(self) -> None:
        # Skybox/libmpv chapter probes hit different live_keys (different t=),
        # so same-owner preempt is required to avoid 10s busy-wait pile-up.
        # Same-live_key duplicates are kept safe by the libmpv startup debounce
        # in passthrough_live_get.
        owner = ("live", "192.168.31.112", "libmpv")
        self.assertTrue(routes_media._can_preempt_owner(owner, owner))

    def test_owner_log_value_hides_local_path_and_client_ip(self) -> None:
        owner = (r"G:\VR\private_movie.mp4", "192.168.31.112")
        text = str(routes_media._owner_log_value(owner))

        self.assertIn("private_movie.mp4", text)
        self.assertIn("client-", text)
        self.assertNotIn(r"G:\VR", text)
        self.assertNotIn("192.168.31.112", text)

    def test_live_owner_log_value_hides_client_ip(self) -> None:
        owner = ("live", "192.168.31.112", "vlc")
        text = str(routes_media._owner_log_value(owner))

        self.assertIn("live", text)
        self.assertIn("vlc", text)
        self.assertIn("client-", text)
        self.assertNotIn("192.168.31.112", text)


class LibmpvLivePreemptTests(unittest.TestCase):
    """Regression: Skybox/libmpv chapter probe bursts must not pile up behind
    a stale active slot, while same-live_key duplicates must still join the
    existing LiveSession instead of preempting the original starter.
    """

    def _run(self, coro):
        return asyncio.run(coro)

    def test_take_active_slot_lets_libmpv_preempt_prior_same_owner_slot(self) -> None:
        old_stream = object()
        new_stream = object()
        owner = ("live", "192.168.31.112", "libmpv")
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_streams[old_stream] = owner

        try:
            with patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1):
                result = self._run(
                    routes_media._take_active_slot(
                        new_stream,
                        "live:sample.mp4@720.00s",
                        owner,
                        allow_same_owner_preempt=True,
                        allow_same_client_preempt=False,
                    )
                )
            self.assertIs(result, old_stream)
            self.assertIn(new_stream, routes_media._active_streams)
            self.assertNotIn(old_stream, routes_media._active_streams)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_take_active_slot_refuses_to_preempt_real_same_owner_producer(self) -> None:
        """Regression: a Skybox chapter-probe burst must not kill a libmpv
        producer that has already started building or serving bytes. Only raw
        slot_tokens (object()) may be preempted by same-owner; a real producer
        (PyNvPassthroughStream or LiveSession surrogate with .close()) is
        protected and the new request must fall through to the wait/503 path.
        """

        class _FakeRealStream:
            def close(self) -> None:
                pass

        old_real_stream = _FakeRealStream()
        new_stream = object()
        owner = ("live", "192.168.31.112", "libmpv")
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_streams[old_real_stream] = owner

        try:
            with (
                patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1),
                patch.object(routes_media, "PASSTHROUGH_BUSY_WAIT_SEC", 0),
            ):
                result = self._run(
                    routes_media._take_active_slot(
                        new_stream,
                        "live:sample.mp4@720.00s",
                        owner,
                        allow_same_owner_preempt=True,
                        allow_same_client_preempt=False,
                    )
                )
            self.assertIs(result, False)
            self.assertIn(old_real_stream, routes_media._active_streams)
            self.assertNotIn(new_stream, routes_media._active_streams)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def test_take_active_slot_returns_busy_for_libmpv_without_preempt_flag(self) -> None:
        old_stream = object()
        new_stream = object()
        owner = ("live", "192.168.31.112", "libmpv")
        original_streams = dict(routes_media._active_streams)
        original_started = dict(routes_media._active_started)
        original_matter = dict(routes_media._active_matter)
        routes_media._active_streams.clear()
        routes_media._active_started.clear()
        routes_media._active_matter.clear()
        routes_media._active_streams[old_stream] = owner

        try:
            with (
                patch.object(routes_media, "PASSTHROUGH_MAX_CONCURRENT", 1),
                patch.object(routes_media, "PASSTHROUGH_BUSY_WAIT_SEC", 0),
            ):
                result = self._run(
                    routes_media._take_active_slot(
                        new_stream,
                        "live:sample.mp4@720.00s",
                        owner,
                        allow_same_owner_preempt=False,
                        allow_same_client_preempt=False,
                    )
                )
            self.assertIs(result, False)
            self.assertIn(old_stream, routes_media._active_streams)
            self.assertNotIn(new_stream, routes_media._active_streams)
        finally:
            routes_media._active_streams.clear()
            routes_media._active_started.clear()
            routes_media._active_matter.clear()
            routes_media._active_streams.update(original_streams)
            routes_media._active_started.update(original_started)
            routes_media._active_matter.update(original_matter)

    def _make_fake_session(self, events: list[str]):
        """Build a minimal LiveSession-shaped stub that records subscribe calls
        and yields no real bytes. Used by debounce/probe path tests below.
        """

        class _FakeLiveSession:
            def __init__(self) -> None:
                self.headers = {"Content-Type": "video/MP2T"}
                self.closed = False
                self.subscribers: list = []
                self.lock = asyncio.Lock()
                self.bytes_emitted = 0
                self.frames_produced = 0

            def subscribe(self, rid: int, *, primary: bool = False, snapshot_only: bool = True):
                events.append(f"subscribe primary={primary} snapshot_only={snapshot_only}")

                async def _gen():
                    yield b""

                return _gen()

        return _FakeLiveSession()

    def test_strip_live_route_hint_suffix_drops_ts_for_skybox_pipeline(self) -> None:
        """Skybox keys its HTTP pipeline on the URL extension, so DLNA URLs
        end in ``.ts``. The route strips that suffix back off before the
        source-file lookup so the underlying ``.mp4`` is still found.
        """
        self.assertEqual(
            routes_media._strip_live_route_hint_suffix("Downloads/x/movie.mp4.ts"),
            "Downloads/x/movie.mp4",
        )
        self.assertEqual(
            routes_media._strip_live_route_hint_suffix("Downloads/x/movie.mp4.TS"),
            "Downloads/x/movie.mp4",
        )
        self.assertEqual(
            routes_media._strip_live_route_hint_suffix("Downloads/x/movie.mp4.m2ts"),
            "Downloads/x/movie.mp4",
        )
        # Backward compat: existing/cached URLs without the hint suffix
        # must still resolve to the same file path.
        self.assertEqual(
            routes_media._strip_live_route_hint_suffix("Downloads/x/movie.mp4"),
            "Downloads/x/movie.mp4",
        )
        # URL-encoded names round-trip through unquote.
        self.assertEqual(
            routes_media._strip_live_route_hint_suffix("Downloads/x/movie%20file.mp4.ts"),
            "Downloads/x/movie file.mp4",
        )

    def test_libmpv_screenshot_probe_does_not_drain_existing_session_cache(self) -> None:
        """Bare libmpv UA probes must NOT be served from an existing playback
        session's cache snapshot — even when one is available. Pcap evidence
        showed ~9 simultaneous probes dumping ~30MB each (~270MB total) into
        the Wi-Fi link, starving the real SKYBOX UA playback connection and
        causing a permanent loading spinner. They must 503 immediately so the
        playback's bandwidth stays uncontested.
        """
        events: list[str] = []
        fake_session = self._make_fake_session(events)
        path = Path("sivr00170_B.mp4")
        live_meta = SimpleNamespace(
            codec=SimpleNamespace(width=3840, height=2160),
            timing=SimpleNamespace(effective_fps=lambda *_: 30.0),
            color=SimpleNamespace(),
        )
        info = SimpleNamespace(duration=1500.0, fps=30.0)
        client_host = "192.168.31.112"

        class _FakeRequest:
            headers = {"user-agent": "libmpv", "accept": "*/*"}
            client = SimpleNamespace(host=client_host)

        request = _FakeRequest()

        async def run() -> object:
            existing_key = (
                str(path.resolve()),
                client_host,
                round(240.0, 3),  # real Skybox playback at t=240
                routes_media.PYNV_OUTPUT_CODEC,
                round(30.0, 3),
                "libmpv:green",
            )
            fake_session.key = existing_key
            async with routes_media._live_session_lock:
                routes_media._live_sessions[existing_key] = fake_session
            try:
                return await routes_media.passthrough_live_get(
                    request,
                    "sivr00170_B.mp4",
                    t=1440.0,
                    mode=None,
                    range_header="bytes=0-",
                    time_seek_range=None,
                    transfer_mode=None,
                    get_content_features=None,
                )
            finally:
                async with routes_media._live_session_lock:
                    routes_media._live_sessions.pop(existing_key, None)

        original_sessions = dict(routes_media._live_sessions)
        original_starting = dict(routes_media._live_starting)
        original_streams = dict(routes_media._active_streams)
        try:
            routes_media._live_sessions.clear()
            routes_media._live_starting.clear()
            routes_media._active_streams.clear()
            with (
                patch.object(routes_media, "_safe_video_path", return_value=path),
                patch.object(routes_media, "annotate_request", return_value=None),
                patch.object(routes_media, "_probe_live_request_metadata", return_value=(info, live_meta, "")),
                patch.object(routes_media, "_live_adaptive_max_fps", return_value=30.0),
                patch.object(routes_media, "_dump_live_request_headers", return_value=None),
                patch.object(routes_media, "_estimated_passthrough_size", return_value=4_000_000_000),
                patch.object(routes_media, "_estimated_passthrough_bps", return_value=100_000_000),
                patch.object(routes_media, "PASSTHROUGH_OUTPUT_MODE", "green"),
                patch.object(
                    routes_media,
                    "_take_active_slot",
                    side_effect=AssertionError(
                        "libmpv screenshot probe must not reserve a slot"
                    ),
                ),
            ):
                response = self._run(run())
        finally:
            routes_media._live_sessions.clear()
            routes_media._live_starting.clear()
            routes_media._active_streams.clear()
            routes_media._live_sessions.update(original_sessions)
            routes_media._live_starting.update(original_starting)
            routes_media._active_streams.update(original_streams)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("retry-after"), "1")
        # Must NOT subscribe to the existing session and pull bytes from it.
        self.assertEqual(events, [], f"probe must not call subscribe(); got {events}")

    def test_libmpv_screenshot_probe_returns_503_when_no_producer(self) -> None:
        """Bare libmpv UA probe with no matching active session must fast-fail
        with 503 Retry-After, NOT start a GPU pipeline. This is the user-
        approved behaviour: thumbnail probes are discardable.
        """
        path = Path("sivr00170_B.mp4")
        live_meta = SimpleNamespace(
            codec=SimpleNamespace(width=3840, height=2160),
            timing=SimpleNamespace(effective_fps=lambda *_: 30.0),
            color=SimpleNamespace(),
        )
        info = SimpleNamespace(duration=1500.0, fps=30.0)
        client_host = "192.168.31.112"

        class _FakeRequest:
            headers = {"user-agent": "libmpv", "accept": "*/*"}
            client = SimpleNamespace(host=client_host)

        request = _FakeRequest()

        async def run() -> object:
            return await routes_media.passthrough_live_get(
                request,
                "sivr00170_B.mp4",
                t=1440.0,
                mode=None,
                range_header="bytes=0-",
                time_seek_range=None,
                transfer_mode=None,
                get_content_features=None,
            )

        original_sessions = dict(routes_media._live_sessions)
        original_starting = dict(routes_media._live_starting)
        original_streams = dict(routes_media._active_streams)
        try:
            routes_media._live_sessions.clear()
            routes_media._live_starting.clear()
            routes_media._active_streams.clear()
            with (
                patch.object(routes_media, "_safe_video_path", return_value=path),
                patch.object(routes_media, "annotate_request", return_value=None),
                patch.object(routes_media, "_probe_live_request_metadata", return_value=(info, live_meta, "")),
                patch.object(routes_media, "_live_adaptive_max_fps", return_value=30.0),
                patch.object(routes_media, "_dump_live_request_headers", return_value=None),
                patch.object(routes_media, "_estimated_passthrough_size", return_value=4_000_000_000),
                patch.object(routes_media, "_estimated_passthrough_bps", return_value=100_000_000),
                patch.object(routes_media, "PASSTHROUGH_OUTPUT_MODE", "green"),
                patch.object(
                    routes_media,
                    "_take_active_slot",
                    side_effect=AssertionError(
                        "libmpv screenshot probe must not reserve a slot"
                    ),
                ),
                patch.object(
                    routes_media,
                    "_select_passthrough_stream",
                    side_effect=AssertionError(
                        "libmpv screenshot probe must not start GPU pipeline"
                    ),
                ),
            ):
                response = self._run(run())
        finally:
            routes_media._live_sessions.clear()
            routes_media._live_starting.clear()
            routes_media._active_streams.clear()
            routes_media._live_sessions.update(original_sessions)
            routes_media._live_starting.update(original_starting)
            routes_media._active_streams.update(original_streams)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("retry-after"), "1")


class LiveSupportTests(unittest.TestCase):
    def _meta(self, *, width: int = 3840, height: int = 2160, verdict: str = "pynv_hevc"):
        return SimpleNamespace(
            codec=SimpleNamespace(width=width, height=height),
            timing=SimpleNamespace(),
            color=SimpleNamespace(),
            _verdict=verdict,
        )

    def test_live_block_reason_rejects_oversized_source(self) -> None:
        meta = self._meta(width=9000, height=4096)
        self.assertIn("exceed", routes_media._live_block_reason(Path("movie.mp4"), meta))

    def test_live_block_reason_rejects_non_pynv_backend(self) -> None:
        meta = self._meta(verdict="ffmpeg_fallback")
        decision = SimpleNamespace(verdict="ffmpeg_fallback", reason="codec needs fallback")
        with patch.object(routes_media, "select_backend", return_value=decision):
            self.assertEqual(routes_media._live_block_reason(Path("movie.mp4"), meta), "codec needs fallback")


if __name__ == "__main__":
    unittest.main()
