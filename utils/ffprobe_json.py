from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Any

from utils.subprocess_hidden import hidden_subprocess_kwargs


def _decode_process_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8-sig", "replace")
    return value


def run_ffprobe_json(cmd: Sequence[str], *, timeout: float | None = None) -> dict[str, Any]:
    """Run ffprobe JSON commands without locale-dependent text decoding."""
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        **hidden_subprocess_kwargs(),
    )
    if proc.returncode != 0:
        detail = _decode_process_text(proc.stderr or proc.stdout).strip()
        raise RuntimeError(detail or f"ffprobe failed rc={proc.returncode}")
    payload = _decode_process_text(proc.stdout).strip()
    return json.loads(payload or "{}")
