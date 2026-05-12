from __future__ import annotations

import re


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_log_text(text: str) -> str:
    """Return UI-safe plain text for QTextEdit log panes."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\b", "")
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or ord(ch) >= 32
    )
