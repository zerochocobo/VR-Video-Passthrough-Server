from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaRoot:
    label: str
    path: Path


def _safe_resolve(path: Path, *, warn: bool = True) -> Path | None:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError) as e:
        if warn:
            log.warning("skip unusable media directory %s: %s", path, e)
        return None


def _absolute_fallback(path: Path) -> Path:
    try:
        return path.expanduser().absolute()
    except (OSError, RuntimeError):
        return path.absolute()


def parse_video_dirs(raw: object, default: Path) -> list[Path]:
    text = str(raw or "").strip()
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if not parts:
        parts = [str(default)]
    fallback = _safe_resolve(Path(default), warn=False) or _absolute_fallback(Path(default))
    roots: list[Path] = []
    seen: set[str] = set()
    for part in parts:
        path = _safe_resolve(Path(part))
        if path is None:
            continue
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots or [fallback]


def build_media_roots(paths: list[Path]) -> list[MediaRoot]:
    used: dict[str, int] = {}
    roots: list[MediaRoot] = []
    for path in paths:
        resolved = _safe_resolve(path)
        if resolved is None:
            continue
        base = path.name or path.drive.rstrip(":\\") or "Videos"
        index = used.get(base.casefold(), 0) + 1
        used[base.casefold()] = index
        label = base if index == 1 else f"{base}{index}"
        roots.append(MediaRoot(label=label, path=resolved))
    return roots


class MediaLibrary:
    def __init__(self, roots: list[MediaRoot]) -> None:
        if not roots:
            raise ValueError("media library requires at least one root")
        self.roots = roots

    @property
    def multi_root(self) -> bool:
        return len(self.roots) > 1

    @property
    def first_root(self) -> MediaRoot:
        return self.roots[0]

    def path_to_key(self, path: Path) -> str:
        resolved = path.resolve()
        for root in self.roots:
            try:
                rel = resolved.relative_to(root.path)
            except ValueError:
                continue
            rel_text = rel.as_posix()
            if self.multi_root:
                return root.label if not rel_text or rel_text == "." else f"{root.label}/{rel_text}"
            return "" if not rel_text or rel_text == "." else rel_text
        raise ValueError(f"path is outside media roots: {path}")

    def key_to_path(self, key: str) -> Path | None:
        rel = str(key or "").replace("\\", "/").strip("/")
        if Path(rel).is_absolute():
            return None
        if not self.multi_root:
            return (self.first_root.path / rel).resolve()
        label, _, rest = rel.partition("/")
        if not label:
            return None
        if Path(rest).is_absolute():
            return None
        root = self.root_by_label(label)
        if root is None:
            return None
        return (root.path / rest).resolve()

    def root_by_label(self, label: str) -> MediaRoot | None:
        wanted = str(label or "").casefold()
        for root in self.roots:
            if root.label.casefold() == wanted:
                return root
        return None

    def contains(self, path: Path) -> bool:
        resolved = path.resolve()
        for root in self.roots:
            if resolved == root.path or root.path in resolved.parents:
                return True
        return False
