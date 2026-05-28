from __future__ import annotations

import json
import unittest
from pathlib import Path

from ui.metadata import AppMetadata, load_app_metadata


class MetadataTests(unittest.TestCase):
    def test_packaged_metadata_loads_version(self) -> None:
        metadata = load_app_metadata()
        payload = json.loads(Path("ui/app_metadata.json").read_text(encoding="utf-8-sig"))
        expected_version = payload["version"]

        self.assertEqual(metadata.version, expected_version)
        self.assertEqual(metadata.display_version, f"v{expected_version}")

    def test_version_tuple_is_comparable(self) -> None:
        metadata = AppMetadata(version="1.2.10")

        self.assertEqual(metadata.version_tuple, (1, 2, 10))
        self.assertGreater(metadata.version_tuple, AppMetadata(version="1.2.9").version_tuple)


if __name__ == "__main__":
    unittest.main()
