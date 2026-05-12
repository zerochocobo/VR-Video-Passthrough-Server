from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppMetadata:
    version: str
    version_scheme: str = "semver"

    @property
    def display_version(self) -> str:
        return f"v{self.version}"

    @property
    def version_tuple(self) -> tuple[int, ...]:
        parts = re.findall(r"\d+", self.version)
        return tuple(int(part) for part in parts)


def load_app_metadata() -> AppMetadata:
    path = Path(__file__).with_name("app_metadata.json")
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    version = str(data.get("version") or "0.1").strip()
    version_scheme = str(data.get("version_scheme") or "semver").strip()
    return AppMetadata(version=version, version_scheme=version_scheme)
