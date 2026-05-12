from __future__ import annotations

import unittest

from ui.metadata import AppMetadata, load_app_metadata


class MetadataTests(unittest.TestCase):
    def test_packaged_metadata_loads_version(self) -> None:
        metadata = load_app_metadata()

        self.assertEqual(metadata.version, "0.1.0-alpha.1")
        self.assertEqual(metadata.display_version, "v0.1.0-alpha.1")

    def test_version_tuple_is_comparable(self) -> None:
        metadata = AppMetadata(version="1.2.10")

        self.assertEqual(metadata.version_tuple, (1, 2, 10))
        self.assertGreater(metadata.version_tuple, AppMetadata(version="1.2.9").version_tuple)


if __name__ == "__main__":
    unittest.main()
