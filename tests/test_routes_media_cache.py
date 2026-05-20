from __future__ import annotations

import unittest
import asyncio
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


class LiveProfileTests(unittest.TestCase):
    def test_quest_dalvik_uses_managed_live_profile(self) -> None:
        ua = "Dalvik/2.1.0 (Linux; U; Android 14; Quest 3 Build/UP1A.231005.007.A1)"
        self.assertEqual(routes_media._live_response_profile(ua), "quest_dalvik")

    def test_known_profiles_still_win(self) -> None:
        self.assertEqual(routes_media._live_response_profile("nPlayer/3.12"), "nplayer")
        self.assertEqual(routes_media._live_response_profile("libmpv Android"), "libmpv")
        self.assertEqual(routes_media._live_response_profile("VLC/3.0"), "vlc")

    def test_quest_dalvik_live_owner_can_preempt_same_device(self) -> None:
        owner = ("live", "192.168.31.112", "quest_dalvik")
        self.assertTrue(routes_media._can_preempt_owner(owner, owner))


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
