from __future__ import annotations

import unittest

from ui.log_sanitizer import clean_log_text


class LogSanitizerTests(unittest.TestCase):
    def test_removes_ansi_nul_and_control_chars(self) -> None:
        raw = "\x1b[0;93mwarning\x1b[m\x00\r\nok\b\x07\n"
        self.assertEqual(clean_log_text(raw), "warning\nok\n")

    def test_removes_nul_separated_ansi_sequences(self) -> None:
        raw = "\x1b\x00[\x000\x00;\x009\x003\x00m\x00warning\x1b\x00[\x00m\x00"
        self.assertEqual(clean_log_text(raw), "warning")


if __name__ == "__main__":
    unittest.main()
