from __future__ import annotations

import json
import locale
import sys
from pathlib import Path

TRANSLATION_DIR = Path(__file__).resolve().parent / "translations"
SUPPORTED = ("zh_CN", "en_US", "ja_JP")


def system_language() -> str:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            lang_id = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
            primary = lang_id & 0x3FF
            if primary == 0x04:
                return "zh_CN"
            if primary == 0x11:
                return "ja_JP"
        except Exception:
            pass
    lang, _encoding = locale.getlocale()
    if not lang:
        return "en_US"
    if lang.startswith("zh"):
        return "zh_CN"
    if lang.startswith("ja"):
        return "ja_JP"
    return "en_US"


class I18n:
    def __init__(self, language: str | None = None) -> None:
        self.language = language if language in SUPPORTED else system_language()
        self._data: dict[str, str] = {}
        self.load(self.language)

    def load(self, language: str) -> None:
        if language not in SUPPORTED:
            language = "en_US"
        path = TRANSLATION_DIR / f"{language}.json"
        self._data = json.loads(path.read_text(encoding="utf-8-sig"))
        self.language = language

    def t(self, key: str) -> str:
        return self._data.get(key, key)
