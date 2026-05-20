from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from http_app.server import create_app
from utils import runtime_settings


class LightMatchControlTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_settings.reset_for_test()

    def test_put_light_match_updates_runtime_settings(self) -> None:
        client = TestClient(create_app())
        before = client.get("/control/light_match")
        self.assertEqual(before.status_code, 200)
        response = client.put(
            "/control/light_match",
            json={
                "enabled": True,
                "temp_k": 3000,
                "tint": 10,
                "exposure_ev": 0.25,
                "contrast": 1.1,
                "gamma": 0.9,
                "saturation": 1.2,
                "preset": "home_warm",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["enabled"])
        self.assertEqual(data["temp_k"], 3000)
        self.assertEqual(data["preset"], "home_warm")
        self.assertGreaterEqual(data["version"], before.json()["version"])

    def test_put_light_match_clamps_payload(self) -> None:
        client = TestClient(create_app())
        response = client.put("/control/light_match", json={"enabled": True, "temp_k": 100000, "gamma": 9})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["temp_k"], 9000)
        self.assertEqual(data["gamma"], 1.4)


if __name__ == "__main__":
    unittest.main()
