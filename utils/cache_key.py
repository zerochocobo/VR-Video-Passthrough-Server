"""Stable cache keys derived from a file path and its current stat data.

The helpers intentionally include absolute path, size, and nanosecond mtime so
cached probes are invalidated when a media file is replaced in place.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from media_library import safe_resolve_path

DEFAULT_LEN = 12  # 12 hex chars, 48 bits.


def fingerprint(src: Path | str, length: int = DEFAULT_LEN) -> str:
    """Return a short SHA1 fingerprint for a file identity/version."""
    p = Path(src)
    st = p.stat()
    raw = f"{safe_resolve_path(p)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:length]


def stat_key(src: Path | str) -> tuple[str, int, int]:
    """Return a hashable key suitable for stat-sensitive in-memory caches."""
    p = Path(src)
    st = p.stat()
    return (os.fspath(safe_resolve_path(p)), st.st_size, st.st_mtime_ns)
