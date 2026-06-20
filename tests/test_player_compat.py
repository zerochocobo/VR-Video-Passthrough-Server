from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import Request

from http_app import routes_dlna
from http_app.server import create_app
from utils import player_compat
from utils.request_history import RequestHistory, annotate_request, build_record, get_request_history


class PlayerCompatProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        player_compat.clear_device_sessions_for_test()

    def test_live_response_profile_preserves_existing_mapping(self) -> None:
        cases = [
            ("nPlayer/3.12", "nplayer"),
            ("AVProMobileVideo/2.0", "avpro"),
            ("ExoPlayerLib/2.18", "avpro"),
            ("Mozilla/5.0 SkyboxVR libmpv", "libmpv"),
            ("Dalvik/2.1.0 (Linux; U; Android 12; Quest 3)", "4xvr"),
            ("com.heresphere.vrvideoplayerdemo/29 (Linux; U; Android 14; Quest 3; Cronet/95.0)", "4xvr"),
            ("VLC/3.0 LibVLC/3.0", "vlc"),
            ("Lavf/58.45.100", "lavf"),
            ("UnknownClient/1.0", "vlc"),
        ]
        for ua, expected in cases:
            with self.subTest(ua=ua):
                self.assertEqual(player_compat.live_response_profile_from_ua(ua, "vlc"), expected)

    def test_deovr_cds_profile_detects_blank_ua_browse_fingerprint(self) -> None:
        fields = {
            "BrowseFlag": "BrowseDirectChildren",
            "Filter": "res,res@size,res@duration,dc:date,upnp:albumArtURI",
            "RequestedCount": "0",
        }

        self.assertEqual(routes_dlna._cds_client_profile({}, fields), "deovr")  # noqa: SLF001
        self.assertEqual(
            routes_dlna._cds_client_profile(  # noqa: SLF001
                {"user-agent": "Mozilla/5.0 [DEO15.6.3755]/meta-store"},
                {},
            ),
            "deovr",
        )
        self.assertIsNone(
            routes_dlna._cds_client_profile(  # noqa: SLF001
                {"user-agent": "Android/14 UPnP/1.0 Cling/2.0"},
                fields,
            )
        )
        self.assertIsNone(routes_dlna._cds_client_profile({}, {**fields, "RequestedCount": "999"}))  # noqa: SLF001

    def test_soap_history_fields_keep_deovr_filter(self) -> None:
        body = (
            b"<s:Envelope><s:Body><u:Browse>"
            b"<BrowseFlag>BrowseDirectChildren</BrowseFlag>"
            b"<Filter>res,res@size,res@duration,dc:date,upnp:albumArtURI</Filter>"
            b"<RequestedCount>0</RequestedCount>"
            b"</u:Browse></s:Body></s:Envelope>"
        )

        fields = routes_dlna._soap_history_fields(body)  # noqa: SLF001

        self.assertEqual(fields["Filter"], "res,res@size,res@duration,dc:date,upnp:albumArtURI")
        self.assertEqual(routes_dlna._cds_client_profile({}, fields), "deovr")  # noqa: SLF001

    def test_cds_ui_language_prefers_accept_language_quality(self) -> None:
        headers = {"Accept-Language": "en-US;q=0.3, zh-CN;q=0.9, ja-JP;q=0.5"}

        self.assertEqual(routes_dlna._cds_ui_language(headers), "zh_CN")  # noqa: SLF001

    def test_skybox_player_ua_matches_versioned_skybox(self) -> None:
        # Real Skybox playback path.
        self.assertTrue(player_compat.is_skybox_player_ua("SKYBOX/2.0.2"))
        self.assertTrue(player_compat.is_skybox_player_ua("skybox/3.0"))
        self.assertTrue(player_compat.is_skybox_player_ua("Mozilla/5.0 SkyboxVR libmpv"))
        # Bare libmpv (Skybox's screenshot prober) is NOT the player UA.
        self.assertFalse(player_compat.is_skybox_player_ua("libmpv"))
        self.assertFalse(player_compat.is_skybox_player_ua(""))
        self.assertFalse(player_compat.is_skybox_player_ua(None))  # type: ignore[arg-type]

    def test_libmpv_screenshot_probe_ua_matches_bare_libmpv_only(self) -> None:
        # Skybox's bare "libmpv" UA = chapter-thumbnail probe.
        self.assertTrue(player_compat.is_libmpv_screenshot_probe_ua("libmpv"))
        self.assertTrue(player_compat.is_libmpv_screenshot_probe_ua("LibMpv"))
        self.assertTrue(player_compat.is_libmpv_screenshot_probe_ua("  libmpv  "))
        # Skybox's actual playback UA must not match — it needs the full pipeline.
        self.assertFalse(player_compat.is_libmpv_screenshot_probe_ua("SKYBOX/2.0.2"))
        self.assertFalse(player_compat.is_libmpv_screenshot_probe_ua("Mozilla/5.0 SkyboxVR libmpv"))
        # Real libmpv builds advertise a version or other tokens.
        self.assertFalse(player_compat.is_libmpv_screenshot_probe_ua("libmpv/0.40.0"))
        self.assertFalse(player_compat.is_libmpv_screenshot_probe_ua(""))
        self.assertFalse(player_compat.is_libmpv_screenshot_probe_ua(None))  # type: ignore[arg-type]

    def test_lavf_is_side_probe_intent_not_player_class(self) -> None:
        profile = player_compat.match_profile("Lavf/58.45.100")
        intent = player_compat.match_intent(
            method="GET",
            path="/passthrough_live/movie.mp4",
            user_agent="Lavf/58.45.100",
            route_profile=profile.route_profile,
        )

        self.assertEqual(profile.route_profile, "lavf")
        self.assertEqual(profile.profile_class, player_compat.PROFILE_UNKNOWN)
        self.assertEqual(intent.intent, player_compat.INTENT_SIDE_PROBE)

    def test_passthrough_probe_decision_rejects_or_caches_before_gpu(self) -> None:
        decision = player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"user-agent": "Lavf/58.45.100", "range": "bytes=564-"},
            client_host="192.0.2.10",
        )

        self.assertEqual(decision.intent.intent, player_compat.INTENT_SIDE_PROBE)
        self.assertEqual(decision.decision, player_compat.DECISION_REJECT_OR_CACHE_BEFORE_GPU)

    def test_duplicate_startup_annotation_takes_priority_over_startup_range(self) -> None:
        decision = player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"user-agent": "nPlayer/3.12", "range": "bytes=12345-"},
            client_host="192.0.2.10",
            annotations={"duplicate_startup": True},
        )

        self.assertEqual(decision.intent.intent, player_compat.INTENT_DUPLICATE_STARTUP)
        self.assertEqual(decision.decision, player_compat.DECISION_REUSE_SESSION_OR_DEBOUNCE)

    def test_tail_probe_uses_total_size_ratio_not_absolute_offset(self) -> None:
        not_tail = player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough/movie.mp4",
            headers={"user-agent": "Player/1.0", "range": "bytes=629145600-629146111"},
            client_host="192.0.2.10",
            annotations={"total_estimated_size": 30 * 1024 * 1024 * 1024},
        )
        tail = player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough/movie.mp4",
            headers={"user-agent": "Player/1.0", "range": "bytes=1020054732-1020055243"},
            client_host="192.0.2.10",
            annotations={"total_estimated_size": 1024 * 1024 * 1024},
        )

        self.assertNotEqual(not_tail.intent.intent, player_compat.INTENT_TAIL_PROBE)
        self.assertEqual(tail.intent.intent, player_compat.INTENT_TAIL_PROBE)

    def test_lavf_side_probe_stays_linked_to_host_session_without_overwriting_profile(self) -> None:
        host = "192.0.2.10"
        vlc_ua = "VLC/3.0 LibVLC/3.0"
        player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"user-agent": vlc_ua},
            client_host=host,
        )
        player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"user-agent": "Lavf/58.45.100"},
            client_host=host,
        )

        session = player_compat._sessions[player_compat.session_key(host, vlc_ua)]  # noqa: SLF001 - characterization of in-memory diagnostics.
        self.assertEqual(session.profile_class, player_compat.PROFILE_VLC_LIKE)
        self.assertEqual(session.side_probe_total, 1)

    def test_same_host_different_player_user_agents_get_separate_sessions(self) -> None:
        host = "192.0.2.10"
        skybox_ua = "SkyboxVR libmpv"
        vlc_ua = "VLC/3.0 LibVLC/3.0"

        player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"User-Agent": skybox_ua},
            client_host=host,
        )
        player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"User-Agent": vlc_ua},
            client_host=host,
        )

        self.assertNotEqual(player_compat.session_key(host, skybox_ua), player_compat.session_key(host, vlc_ua))
        self.assertEqual(player_compat._sessions[player_compat.session_key(host, skybox_ua)].profile_class, player_compat.PROFILE_LIBMPV_LIKE)  # noqa: SLF001
        self.assertEqual(player_compat._sessions[player_compat.session_key(host, vlc_ua)].profile_class, player_compat.PROFILE_VLC_LIKE)  # noqa: SLF001

    def test_raw_media_preview_is_observation_only(self) -> None:
        decision = player_compat.classify_request_shadow(
            method="GET",
            path="/media/movie.mp4",
            headers={"user-agent": "Player/1.0", "range": "bytes=0-1023"},
            client_host="192.0.2.10",
        )

        self.assertEqual(decision.intent.intent, player_compat.INTENT_RAW_MEDIA_PREVIEW)
        self.assertEqual(decision.decision, player_compat.DECISION_OBSERVE_ONLY)

    def test_scenario_replay_classifies_recorded_sequence(self) -> None:
        records = [
            {
                "method": "POST",
                "path": "/control/cds",
                "headers": {"user-agent": "nPlayer/3.12"},
                "client_host": "192.0.2.10",
                "annotations": {"BrowseFlag": "BrowseDirectChildren"},
            },
            {
                "method": "GET",
                "path": "/media/movie.mp4",
                "headers": {"user-agent": "nPlayer/3.12", "range": "bytes=0-4095"},
                "client_host": "192.0.2.10",
            },
            {
                "method": "GET",
                "path": "/passthrough_live/movie.mp4",
                "headers": {"user-agent": "nPlayer/3.12", "range": "bytes=0-"},
                "client_host": "192.0.2.10",
            },
        ]

        decisions = player_compat.replay_scenario(records)

        self.assertEqual([item.intent.intent for item in decisions], [
            player_compat.INTENT_BROWSE,
            player_compat.INTENT_RAW_MEDIA_PREVIEW,
            player_compat.INTENT_PLAYBACK_PRIMARY,
        ])
        self.assertEqual(decisions[-1].profile.profile_class, player_compat.PROFILE_NPLAYER_LIKE)


class RequestHistoryTests(unittest.TestCase):
    def _request(self, path: str, *, client_host: str = "192.168.1.55", query: bytes = b"") -> Request:
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": path,
                "raw_path": path.encode("ascii", "ignore"),
                "scheme": "http",
                "server": ("testserver", 80),
                "client": (client_host, 50000),
                "headers": [(b"host", b"testserver"), (b"user-agent", b"nPlayer/3.12")],
                "query_string": query,
            }
        )

    def test_request_history_bounds_memory_and_writes_jsonl(self) -> None:
        from utils.request_history import RequestRecord

        with tempfile.TemporaryDirectory() as tmp:
            history = RequestHistory(
                max_records=2,
                jsonl_dir=Path(tmp),
                jsonl_enabled=True,
                flush_every=2,
            )
            for i in range(3):
                history.add(
                    RequestRecord(
                        trace_id=f"pt-{i}",
                        ts="2026-05-28T00:00:00.000",
                        method="GET",
                        path=f"/media/{i}.mp4",
                        query="",
                        client_host="127.0.0.1",
                        user_agent="test",
                        range="",
                        time_seek_range="",
                        transfer_mode="",
                        get_content_features="",
                        status_code=200,
                        elapsed_ms=1.0,
                    )
                )
            history.flush()

            snapshot = history.snapshot()
            self.assertEqual([item["trace_id"] for item in snapshot], ["pt-1", "pt-2"])
            files = list(Path(tmp).glob("request_history_*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(files[0].read_text(encoding="utf-8").splitlines()), 3)

    def test_request_history_redacts_client_and_media_identifiers(self) -> None:
        request = self._request("/media/private_movie.mp4", query=b"t=12345&mode=alpha&ptv=7")
        annotate_request(
            request,
            media_path=r"G:\VR\private_movie.mp4",
            media_name="private_movie.mp4",
            ObjectID="0$private_movie.mp4",
        )

        record = build_record(
            request=request,
            trace_id="pt-test-00000001",
            status_code=200,
            elapsed_ms=1.0,
            redact=True,
        )
        payload = record.to_jsonable()

        self.assertTrue(payload["client_host"].startswith("client-"))
        self.assertNotIn("192.168.1.55", str(payload))
        self.assertNotIn("private_movie.mp4", str(payload))
        self.assertEqual(payload["path"].split("/", 2)[1], "media")
        self.assertEqual(payload["query"], "mode=alpha&ptv=7")

    def test_server_adds_trace_header_and_records_history(self) -> None:
        with TestClient(create_app()) as client:
            get_request_history().clear_for_test()
            response = client.get("/description.xml")
            self.assertEqual(response.status_code, 200)
            self.assertRegex(response.headers.get("X-PT-Request-Trace-Id", ""), r"^pt-[0-9a-f]{6}-\d+$")

            snapshot = get_request_history().snapshot()
            self.assertEqual(len(snapshot), 1)
            self.assertEqual(snapshot[0]["path"], "/description.xml")
            self.assertEqual(snapshot[0]["status_code"], 200)

    def test_debug_request_history_endpoint_is_localhost_only(self) -> None:
        with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
            get_request_history().clear_for_test()
            response = client.get("/debug/request_history")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])
            self.assertEqual(get_request_history().snapshot(), [])

        with TestClient(create_app(), client=("192.0.2.10", 50000)) as client:
            response = client.get("/debug/request_history")
            self.assertEqual(response.status_code, 403)

    def test_debug_clear_device_sessions_endpoint_is_localhost_only(self) -> None:
        player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"user-agent": "VLC/3.0 LibVLC/3.0"},
            client_host="192.0.2.10",
        )
        self.assertTrue(player_compat._sessions)  # noqa: SLF001

        with TestClient(create_app(), client=("127.0.0.1", 50000)) as client:
            response = client.post("/debug/clear_device_sessions")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"ok": True})
            self.assertEqual(player_compat._sessions, {})  # noqa: SLF001

        player_compat.classify_request_shadow(
            method="GET",
            path="/passthrough_live/movie.mp4",
            headers={"user-agent": "VLC/3.0 LibVLC/3.0"},
            client_host="192.0.2.10",
        )
        with TestClient(create_app(), client=("192.0.2.10", 50000)) as client:
            response = client.post("/debug/clear_device_sessions")
            self.assertEqual(response.status_code, 403)
            self.assertTrue(player_compat._sessions)  # noqa: SLF001

    def test_exception_path_records_history_without_swallowing_error(self) -> None:
        app = create_app()

        @app.get("/boom")
        async def boom():
            raise RuntimeError("boom")

        with TestClient(app) as client, self.assertRaises(RuntimeError):
            get_request_history().clear_for_test()
            client.get("/boom")

        snapshot = get_request_history().snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]["path"], "/boom")
        self.assertEqual(snapshot[0]["status_code"], 500)

    def test_streaming_response_header_immediate_case(self) -> None:
        """Covers immediate iterables; long passthrough streams need real-device/curl validation."""
        app = create_app()

        @app.get("/stream-test")
        async def stream_test():
            return StreamingResponse(iter([b"ok"]), media_type="text/plain")

        with TestClient(app) as client:
            response = client.get("/stream-test")

        self.assertEqual(response.status_code, 200)
        self.assertRegex(response.headers.get("X-PT-Request-Trace-Id", ""), r"^pt-[0-9a-f]{6}-\d+$")


if __name__ == "__main__":
    unittest.main()
