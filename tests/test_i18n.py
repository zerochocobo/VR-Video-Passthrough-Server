from __future__ import annotations

import json
import unittest
from pathlib import Path


class I18nTests(unittest.TestCase):
    def test_translation_keys_match(self) -> None:
        base = Path("ui/translations")
        translations = {
            path.stem: set(json.loads(path.read_text(encoding="utf-8-sig")))
            for path in base.glob("*.json")
        }
        self.assertIn("zh_CN", translations)
        self.assertIn("en_US", translations)
        self.assertIn("ja_JP", translations)
        reference = translations["zh_CN"]
        for language, keys in translations.items():
            with self.subTest(language=language):
                self.assertEqual(reference, keys)


if __name__ == "__main__":
    unittest.main()
