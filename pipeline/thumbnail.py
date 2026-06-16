"""Thumbnail generation and cache.

Source thumbnails are extracted near 10% of the video duration, capped at 30s.
DLNA live/passthrough entries normally reuse the raw thumbnail so browsing does
not load the matting model or block on a GPU pass. Output names include the
source fingerprint so stale thumbnails naturally age out when the source path,
size, or mtime changes.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from config import RUNTIME_CACHE_DIR, THUMB_FFMPEG_TIMEOUT_SEC
from pipeline.ffmpeg_io import FFMPEG, probe_cached
from utils.cache_key import fingerprint
from utils.logger import get
from utils.subprocess_hidden import hidden_subprocess_kwargs

log = get("thumb")

THUMB_DIR = RUNTIME_CACHE_DIR / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

THUMB_WIDTH = 480
_thumb_matter = None


def _get_thumb_matter():
    global _thumb_matter
    if _thumb_matter is None:
        from pipeline.matting import Matter
        _thumb_matter = Matter()
    _thumb_matter.reset_state()
    return _thumb_matter


def _out_path(src: Path, passthrough: bool, fp: str) -> Path:
    suffix = "-pt" if passthrough else ""
    return THUMB_DIR / f"{src.stem}_{fp}{suffix}.jpg"


def _cleanup_stale(src: Path, passthrough: bool, keep: Path) -> None:
    """Delete stale thumbnails for the same source stem and passthrough mode."""
    cutoff = time.time() - 3600.0
    for p in THUMB_DIR.glob(f"{src.stem}_*.jpg"):
        if p == keep:
            continue
        if p.stem.endswith("-pt") != passthrough:
            continue
        try:
            if p.stat().st_mtime >= cutoff:
                continue
            p.unlink()
        except OSError as e:
            log.debug("cleanup skip %s: %s", p.name, e)


def _extract_frame(src: Path, seek_sec: float) -> np.ndarray | None:
    """Extract one BGR frame through a PNG stdout pipe."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{seek_sec:.3f}",
        "-i", str(src),
        "-frames:v", "1",
        "-vf", f"scale={THUMB_WIDTH}:-2",
        "-c:v", "png", "-f", "image2pipe", "-",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=THUMB_FFMPEG_TIMEOUT_SEC,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg thumb extract timed out after %.1fs: %s", THUMB_FFMPEG_TIMEOUT_SEC, src)
        return None
    if r.returncode != 0 or not r.stdout:
        log.warning("ffmpeg thumb extract failed: %s", r.stderr.decode("utf-8", "ignore")[:200])
        return None
    arr = np.frombuffer(r.stdout, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def get_thumb(src: Path, passthrough: bool) -> Path | None:
    """Generate or return a cached thumbnail path."""
    # Passthrough thumbnails used to run one matting pass, but DLNA clients can
    # request many thumbnails while browsing. Reusing raw thumbnails keeps
    # directory navigation responsive and avoids Ctrl+C waiting on GPU work.
    passthrough = False
    fp = fingerprint(src)
    out = _out_path(src, passthrough, fp)
    if out.exists():
        log.info("thumb cache hit: src=%s passthrough=%s out=%s", src.name, passthrough, out.name)
        return out

    try:
        info = probe_cached(src)
    except Exception as e:
        log.warning("probe failed: %s", e)
        return None

    seek = min(max(info.duration * 0.1, 1.0), 30.0)
    img = _extract_frame(src, seek)
    if img is None:
        return None

    if passthrough:
        try:
            img = _get_thumb_matter().composite_green(img)
        except Exception as e:
            log.warning("passthrough thumb matting failed, fallback raw: %s", e)

    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok or encoded is None:
        log.warning("jpeg thumb encode failed: %s", src)
        return None
    tmp = out.with_name(f".{out.name}.tmp")
    try:
        tmp.write_bytes(encoded.tobytes())
        tmp.replace(out)
    except OSError as e:
        log.warning("jpeg thumb write failed: %s error=%s", out, e)
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    if not out.exists():
        log.warning("jpeg thumb missing after write: %s", out)
        return None
    _cleanup_stale(src, passthrough=passthrough, keep=out)
    log.info("thumb generated: src=%s passthrough=%s out=%s", src.name, passthrough, out.name)
    return out
