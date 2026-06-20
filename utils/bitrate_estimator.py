"""Persistent bitrate estimator for virtual passthrough resources.

Passthrough output is generated on demand, so HTTP and DLNA need an estimated
Content-Length before all bytes exist. Defaults come from configured bitrates;
successful streams feed an EWMA cache keyed by file stat data and encoder
settings so future estimates get closer without reusing stale results.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from config import PASSTHROUGH_BITRATE, PASSTHROUGH_HEVC_BITRATE, PASSTHROUGH_MAX_FPS
from utils.cache_key import stat_key
from utils.ffprobe_json import run_ffprobe_json

CACHE_PATH = config.BITRATE_ESTIMATES
LOCK_PATH = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".lock")
_SAFETY_MULTIPLIER = 1.08
_MIN_ESTIMATED_SIZE = 1024 * 1024
_MIN_RECORD_SECONDS = 3.0
_MAX_DEFAULT_MULTIPLIER = 3.0
_LOCK_TIMEOUT_SEC = 5.0
_STALE_LOCK_SEC = 30.0


@dataclass(frozen=True)
class BitrateEstimate:
    key: str
    bps: int
    source: str
    samples: int = 0


def parse_bitrate(value: str | int | float | None, default: int = 20_000_000) -> int:
    if value is None:
        return default
    text = str(value).strip().upper()
    try:
        if text.endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        if text.endswith("K"):
            return int(float(text[:-1]) * 1_000)
        return int(float(text))
    except ValueError:
        return default


def projection_capped_bitrate(
    configured: str | int | float | None,
    src_path: Path,
    projection: str,
    mult_3d: float = 3.0,
    mult_vr: float = 4.0,
) -> int:
    """Cap the 2D->3D output bitrate at a multiple of the source bitrate.

    flat3d -> mult_3d (SBS, ~2x pixels), VR projections (fisheye/hequirect) ->
    mult_vr. Returns min(configured, mult * source). A 0/unknown multiplier or an
    unreadable source falls back to the configured bitrate unchanged.
    """
    cfg = parse_bitrate(configured)
    mult = float(mult_vr if str(projection).lower() in {"fisheye", "hequirect"} else mult_3d)
    if mult <= 0:
        return cfg
    source_bps = source_video_bitrate(Path(src_path))
    if not source_bps:
        return cfg
    return min(cfg, max(1, int(source_bps * mult)))


def _default_bitrate_for_codec(codec: str = "") -> str:
    return PASSTHROUGH_HEVC_BITRATE if codec.lower() in {"hevc", "h265", "pynv_hevc"} else PASSTHROUGH_BITRATE


def source_video_bitrate(path: Path) -> int | None:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=bit_rate:stream=bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        data = run_ffprobe_json(cmd)
    except Exception:
        return None
    for stream in data.get("streams") or []:
        value = stream.get("bit_rate")
        if value not in (None, "N/A", ""):
            try:
                bitrate = int(float(value))
            except (TypeError, ValueError):
                continue
            if bitrate > 0:
                return bitrate
    value = (data.get("format") or {}).get("bit_rate")
    if value not in (None, "N/A", ""):
        try:
            bitrate = int(float(value))
        except (TypeError, ValueError):
            return None
        if bitrate > 0:
            return bitrate
    return None


def effective_default_bitrate(path: Path, codec: str = "") -> BitrateEstimate:
    configured = parse_bitrate(_default_bitrate_for_codec(codec))
    if codec.lower() not in {"hevc", "h265", "pynv_hevc"}:
        return BitrateEstimate(key="", bps=configured, source="default", samples=0)
    multiplier = float(getattr(config, "PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER", 0.0) or 0.0)
    if multiplier <= 0:
        return BitrateEstimate(key="", bps=configured, source="default", samples=0)
    source_bps = source_video_bitrate(path)
    if not source_bps:
        return BitrateEstimate(key="", bps=configured, source="default", samples=0)
    capped = min(configured, max(1, int(source_bps * multiplier)))
    if capped < configured:
        return BitrateEstimate(key="", bps=capped, source=f"source_x{multiplier:g}", samples=0)
    return BitrateEstimate(key="", bps=configured, source="default", samples=0)


def _effective_vcodec_for_key(codec: str = "") -> str:
    return "pynv_hevc" if codec.lower() in {"hevc", "h265", "pynv_hevc"} else config.PASSTHROUGH_VCODEC


def make_key(path: Path, codec: str = "", params: dict[str, Any] | None = None) -> str:
    p = Path(path).resolve()
    stat = stat_key(p)
    parts = [
        stat[0],
        str(stat[1]),
        str(stat[2]),
        codec or "auto",
        f"fps{PASSTHROUGH_MAX_FPS:g}",
        f"container{config.PASSTHROUGH_CONTAINER}",
        f"vcodec{_effective_vcodec_for_key(codec)}",
        f"bitrate{_default_bitrate_for_codec(codec)}",
        f"source_bitrate_multiplier{getattr(config, 'PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER', 0.0):g}",
        f"gop{config.PASSTHROUGH_GOP}",
        f"preset{config.PASSTHROUGH_PRESET}",
        f"tune{config.PASSTHROUGH_TUNE}",
        f"rc{config.PASSTHROUGH_RC}",
        f"lookahead{config.PASSTHROUGH_RC_LOOKAHEAD}",
        f"bf{config.PASSTHROUGH_BF}",
        f"multipass{config.PASSTHROUGH_MULTIPASS}",
        f"no_scenecut{config.PASSTHROUGH_NO_SCENECUT}",
        f"spatial_aq{config.PASSTHROUGH_SPATIAL_AQ}",
        f"temporal_aq{config.PASSTHROUGH_TEMPORAL_AQ}",
        f"surfaces{config.PASSTHROUGH_SURFACES}",
        f"delay{config.PASSTHROUGH_DELAY}",
        f"zerolatency{config.PASSTHROUGH_ZERO_LATENCY}",
        f"strict_gop{config.PASSTHROUGH_STRICT_GOP}",
        f"aud{config.PASSTHROUGH_AUD}",
    ]
    for key, value in sorted((params or {}).items()):
        parts.append(f"{key}{value}")
    return "|".join(parts)


def _load() -> dict[str, Any]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


@contextmanager
def _file_lock(timeout_sec: float = _LOCK_TIMEOUT_SEC):
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii", "ignore"))
            break
        except FileExistsError:
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
                if age > _STALE_LOCK_SEC:
                    LOCK_PATH.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for bitrate estimate lock: {LOCK_PATH}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def _save(data: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=CACHE_PATH.name, suffix=".tmp", dir=str(CACHE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_name, CACHE_PATH)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def estimate_bps(path: Path, codec: str = "", params: dict[str, Any] | None = None) -> BitrateEstimate:
    key = make_key(path, codec, params)
    row = _load().get(key)
    if isinstance(row, dict):
        try:
            bps = int(row.get("avg_bps") or 0)
            if bps > 0:
                return BitrateEstimate(key=key, bps=bps, source="cache", samples=int(row.get("samples") or 0))
        except Exception:
            pass
    estimate = effective_default_bitrate(path, codec)
    return BitrateEstimate(key=key, bps=estimate.bps, source=estimate.source, samples=0)


def estimated_size(duration_sec: float, estimate: BitrateEstimate, overhead_bytes: int = 512 * 1024) -> int:
    if duration_sec <= 0:
        return 0
    raw = int(duration_sec * estimate.bps / 8) + overhead_bytes
    return max(_MIN_ESTIMATED_SIZE, int(raw * _SAFETY_MULTIPLIER))


def estimate_for_media(path: Path, duration_sec: float, codec: str = "", params: dict[str, Any] | None = None) -> tuple[int, int, BitrateEstimate]:
    estimate = estimate_bps(path, codec, params)
    return estimated_size(duration_sec, estimate), estimate.bps, estimate


def record_actual_bps(
    path: Path,
    codec: str,
    params: dict[str, Any] | None,
    actual_bps: float,
    elapsed_media_sec: float,
) -> None:
    if actual_bps <= 0 or elapsed_media_sec < _MIN_RECORD_SECONDS:
        return
    default_bps = effective_default_bitrate(path, codec).bps
    actual_bps = min(float(actual_bps), default_bps * _MAX_DEFAULT_MULTIPLIER)
    key = make_key(path, codec, params)
    with _file_lock():
        data = _load()
        row = data.get(key) if isinstance(data.get(key), dict) else {}
        old_bps = float(row.get("avg_bps") or 0)
        samples = int(row.get("samples") or 0)
        if old_bps > 0 and samples > 0:
            avg_bps = int(old_bps * 0.7 + float(actual_bps) * 0.3)
            samples += 1
        else:
            avg_bps = int(actual_bps)
            samples = 1
        data[key] = {
            "avg_bps": avg_bps,
            "samples": samples,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(data)
