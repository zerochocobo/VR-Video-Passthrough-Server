from __future__ import annotations

import unittest
from unittest.mock import patch

import http_app.routes_media as routes_media


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


if __name__ == "__main__":
    unittest.main()
