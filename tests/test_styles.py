from __future__ import annotations

import unittest

from ui.styles import font_for_language, load_app_stylesheet


class StyleTests(unittest.TestCase):
    def test_stylesheet_loads(self) -> None:
        qss = load_app_stylesheet()
        self.assertIn("QPushButton", qss)
        self.assertIn("QScrollBar:vertical", qss)

    def test_font_for_language_switches_family(self) -> None:
        zh = font_for_language("zh_CN").families()
        en = font_for_language("en_US").families()
        self.assertIn("Microsoft YaHei UI", zh)
        self.assertIn("Segoe UI", en)


if __name__ == "__main__":
    unittest.main()
