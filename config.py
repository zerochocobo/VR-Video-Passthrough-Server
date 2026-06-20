"""Global configuration.

Most runtime-facing values can be overridden with an environment variable using
the `PT_` prefix. For example, `HTTP_PORT` is controlled by `PT_HTTP_PORT`.

Defaults here are the current single-user Windows production profile. Runtime
values remain overridable with PT_* environment variables for diagnostics and
client-specific A/B tests.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

from utils.gpu_requirements import resolve_passthrough_max_concurrent

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).parent.resolve()


def _env(key: str, default):
    """Read PT_<key>, returning default when unset."""
    return os.environ.get(f"PT_{key}", default)


def _env_any(keys: tuple[str, ...], default):
    """Read the first set PT_<key> from a list of compatible aliases."""
    for key in keys:
        value = os.environ.get(f"PT_{key}")
        if value is not None:
            return value
    return default


def _rgb_hex(value: str, default: str = "000000") -> tuple[int, int, int]:
    """Parse RRGGBB or #RRGGBB into an RGB tuple."""
    text = str(value or default).strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        text = default
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        fallback = default.lstrip("#")
        return int(fallback[0:2], 16), int(fallback[2:4], 16), int(fallback[4:6], 16)


# ---- Network ----
# PT_LAN_IP:
#   Address advertised in SSDP, device XML, and media URLs. Default auto-detects
#   the outbound LAN address by opening a UDP socket to 8.8.8.8. Override this
#   when the machine has multiple NICs/VPNs and DLNA clients see the wrong IP.
def _detect_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


LAN_IP: str = _env("LAN_IP", _detect_lan_ip())

# PT_HTTP_PORT:
#   Main HTTP port for UPnP XML, SOAP control, raw media, thumbnails, and
#   passthrough streams. If changed, firewall rules and DLNA rediscovery may be
#   needed. Default: 8200.
HTTP_PORT: int = int(_env("HTTP_PORT", 8200))

# PT_STARTUP_STATUS_PORT:
#   Optional localhost-only status endpoint used while startup GPU warmup is
#   running and the main HTTP port is not listening yet. Set <=0 to disable.
#   Default: 8299.
STARTUP_STATUS_PORT: int = int(_env("STARTUP_STATUS_PORT", 8299))

# SSDP alive NOTIFY heartbeat interval in seconds. Not currently env-backed
# because clients tolerate the default and changing it rarely helps.
SSDP_INTERVAL_SEC: int = 60


# ---- DLNA identity ----
# PT_SERVER_NAME:
#   Friendly name shown by DLNA clients during discovery. Some clients cache it,
#   so restart the server and refresh/re-discover the client after changing it.
SERVER_NAME = _env("SERVER_NAME", "VR Passthrough Server")

# Static strings included in device description XML.
MANUFACTURER = "PT"
MODEL_NAME = "PT-DLNA"

# UUID is stable for one LAN_IP/HTTP_PORT pair. Changing IP or port changes the
# advertised UDN, helping clients treat the endpoint as a different device.
DEVICE_UUID = str(uuid5(NAMESPACE_DNS, f"ptserver-{LAN_IP}-{HTTP_PORT}"))
DEVICE_USN = f"uuid:{DEVICE_UUID}"


# ---- Media library and thumbnails ----
# PT_VIDEO_DIR:
#   Root directory exposed through ContentDirectory. Multiple roots can be
#   separated with `|`. With multiple roots, clients first see virtual folders
#   named after each physical directory, with numeric suffixes for conflicts.
from media_library import MediaLibrary, build_media_roots, parse_video_dirs
from utils.si_filter import (
    DEFAULT_DUCK_ORIGINAL,
    DEFAULT_ORIGINAL_VOLUME_PERCENT,
    DEFAULT_SI_DELAY_SECONDS,
    DEFAULT_SI_MIX_CHANNEL,
    DEFAULT_SI_MIX_ENABLED,
    DEFAULT_SI_VOLUME_PERCENT,
)


VIDEO_DIRS: list[Path] = parse_video_dirs(_env("VIDEO_DIR", ROOT / "videos"), ROOT / "videos")
MEDIA_ROOTS = build_media_roots(VIDEO_DIRS)
MEDIA_LIBRARY = MediaLibrary(MEDIA_ROOTS)
VIDEO_DIR: Path = VIDEO_DIRS[0]

# File extensions considered video media during directory scans.
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}

# PT_DLNA_IMAGE_ENABLED:
#   1 exposes still images in DLNA Browse and allows /media/{name} to serve
#   those image files with their real image Content-Type. Default is off because
#   most tested VR players do not support image playback through this DLNA path.
DLNA_IMAGE_ENABLED = _env("DLNA_IMAGE_ENABLED", "0") == "1"

# File extensions considered image media when PT_DLNA_IMAGE_ENABLED=1.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}
IMAGE_DLNA_PN_BY_EXT = {
    ".jpg": "JPEG_LRG",
    ".jpeg": "JPEG_LRG",
    ".png": "PNG_LRG",
}

# ---- Burned-in subtitle overlay ----
# PT_SUBTITLE_ENABLE:
#   1 enables automatic burned-in subtitles for passthrough streams. The server
#   looks for subtitle files next to the source video, preferring same-stem
#   `.ass`/`.srt` files. When no subtitle exists, the path is skipped.
SUBTITLE_ENABLE = _env("SUBTITLE_ENABLE", "1") == "1"

# PT_SUBTITLE_EXTS:
#   Ordered subtitle extensions to try beside the video source.
SUBTITLE_EXTS = tuple(
    ext if ext.startswith(".") else f".{ext}"
    for ext in str(_env("SUBTITLE_EXTS", ".ass,.srt")).replace(";", ",").split(",")
    if ext.strip()
)

# PT_SUBTITLE_MODE:
#   auto draws once for 2D and as dual-eye overlay for SBS-like frames.
#   dual draws both eyes, left/right draw one eye only, mono draws one centered
#   overlay across the full frame.
SUBTITLE_MODE = str(_env("SUBTITLE_MODE", "auto")).lower()

# PT_SUBTITLE_DISTANCE_M:
#   Stereo subtitle depth in meters for dual-eye SBS mode. Lower values create
#   more parallax. Used with the same IPD formula as the built-in renderer.
SUBTITLE_DISTANCE_M = max(0.1, float(_env("SUBTITLE_DISTANCE_M", 4.0)))

# PT_SUBTITLE_DIRECTION:
#   horizontal_bottom, horizontal_middle, horizontal_top, vertical_left,
#   vertical_middle, vertical_right.
SUBTITLE_DIRECTION = str(_env("SUBTITLE_DIRECTION", "horizontal_bottom")).lower()

# PT_SUBTITLE_COLOR / PT_SUBTITLE_OUTLINE_COLOR:
#   RRGGBB colors for text and outline. Empty SUBTITLE_COLOR uses the inverse
#   of the configured passthrough/green-screen background color at render time.
SUBTITLE_COLOR_RAW = str(_env("SUBTITLE_COLOR", "")).strip()
SUBTITLE_COLOR = _rgb_hex(SUBTITLE_COLOR_RAW, "FFFFFF") if SUBTITLE_COLOR_RAW else None
SUBTITLE_OUTLINE_COLOR = _rgb_hex(str(_env("SUBTITLE_OUTLINE_COLOR", "000000")), "000000")

# PT_SUBTITLE_ALPHA:
#   Text/outline opacity. 1.0 is fully opaque.
SUBTITLE_ALPHA = max(0.0, min(1.0, float(_env("SUBTITLE_ALPHA", 1.0))))

# PT_SUBTITLE_FONT:
#   Optional font file or font family name. Empty uses common Windows CJK fonts
#   when available, then Pillow's default font.
SUBTITLE_FONT = str(_env("SUBTITLE_FONT", "")).strip()

# PT_SUBTITLE_FONT_SCALE:
#   Font size as a fraction of one eye height. Default 0.045 is readable on 8K
#   SBS without dominating the view.
SUBTITLE_FONT_SCALE = max(0.005, float(_env("SUBTITLE_FONT_SCALE", 0.045)))

# PT_SUBTITLE_OUTLINE_SCALE:
#   Outline width relative to font size.
SUBTITLE_OUTLINE_SCALE = max(0.0, float(_env("SUBTITLE_OUTLINE_SCALE", 0.08)))

# PT_SUBTITLE_MARGIN_V_SCALE:
#   Vertical margin relative to one eye height.
SUBTITLE_MARGIN_V_SCALE = max(0.0, float(_env("SUBTITLE_MARGIN_V_SCALE", 0.08)))

# PT_SUBTITLE_V360:
#   1 projects the per-eye subtitle canvas as a flat plane into the equirect eye
#   image, matching the built-in flat->hequirect behavior.
SUBTITLE_V360 = _env("SUBTITLE_V360", "1") == "1"

# PT_SUBTITLE_FOV / PT_SUBTITLE_YAW / PT_SUBTITLE_PITCH:
#   Flat subtitle plane projection parameters in degrees.
SUBTITLE_FOV = max(1.0, min(179.0, float(_env("SUBTITLE_FOV", 60.0))))
SUBTITLE_YAW = float(_env("SUBTITLE_YAW", 0.0))
SUBTITLE_PITCH = float(_env("SUBTITLE_PITCH", 0.0))

# Virtual item suffix shown in DLNA titles, e.g. `movie-passthrough-live`.
PASSTHROUGH_SUFFIX = "_passthrough"

# PT_PASSTHROUGH_OUTPUT_MODE:
#   Output layout for generated passthrough streams.
#   Values:
#     green - existing green-background chroma-key stream.
#     alpha - experimental DeoVR alpha-packed fisheye stream, based on the
#             offline alpha passthrough test path.
#     all   - expose both green and alpha live entries in DLNA.
#     none  - plain DLNA server; expose source media only.
PASSTHROUGH_OUTPUT_MODE = _env("PASSTHROUGH_OUTPUT_MODE", "green").lower()

# PT_ALPHA_PASSTHROUGH_TITLE:
#   DLNA virtual item title used when PT_PASSTHROUGH_OUTPUT_MODE=alpha.
ALPHA_PASSTHROUGH_TITLE = _env("ALPHA_PASSTHROUGH_TITLE", "Alpha Passthrough")

# PT_THUMB_FFMPEG_TIMEOUT_SEC:
#   Maximum time spent extracting one thumbnail frame with ffmpeg. Keep this
#   short because DLNA clients can request many thumbnails during browsing and
#   shutdown waits for active HTTP requests. Default: 3 seconds.
THUMB_FFMPEG_TIMEOUT_SEC = max(1.0, float(_env("THUMB_FFMPEG_TIMEOUT_SEC", 3.0)))

# PT_LIVE_REQUEST_HEADER_DUMP:
#   1 writes complete `/passthrough_live` request headers to
#   `debug_output/live_requests` for client compatibility diagnostics.
LIVE_REQUEST_HEADER_DUMP = _env("LIVE_REQUEST_HEADER_DUMP", "0") == "1"

# PT_REQUEST_HISTORY_ENABLED:
#   1 records a bounded in-memory trace of DLNA/media requests. This is a
#   lightweight compatibility diagnostic layer and does not change playback
#   policy by itself.
REQUEST_HISTORY_ENABLED = _env("REQUEST_HISTORY_ENABLED", "1") == "1"

# PT_REQUEST_HISTORY_MAX_RECORDS:
#   Maximum number of request records kept in memory for local diagnostics and
#   the debug endpoint.
REQUEST_HISTORY_MAX_RECORDS = max(0, int(_env("REQUEST_HISTORY_MAX_RECORDS", 500)))

# PT_REQUEST_HISTORY_JSONL:
#   1 appends request-history records to rolling JSONL files under
#   debug_output/request_history. Writes are buffered and flushed every
#   PT_REQUEST_HISTORY_FLUSH_EVERY records to avoid per-request fsync costs.
REQUEST_HISTORY_JSONL = _env("REQUEST_HISTORY_JSONL", "1") == "1"
REQUEST_HISTORY_DIR: Path = Path(_env("REQUEST_HISTORY_DIR", ROOT / "debug_output" / "request_history")).resolve()
REQUEST_HISTORY_FLUSH_EVERY = max(1, int(_env("REQUEST_HISTORY_FLUSH_EVERY", 16)))

# PT_REQUEST_HISTORY_REDACT:
#   1 redacts client IPs and media identifiers in request-history output so
#   JSONL traces can be shared for compatibility debugging with less local
#   path/network leakage. This only affects request-history JSONL/in-memory
#   output; server.log remains the raw operational log and must be filtered
#   separately before sharing.
REQUEST_HISTORY_REDACT = _env("REQUEST_HISTORY_REDACT", "1") == "1"

# PT_REQUEST_HISTORY_DEBUG_ENDPOINT:
#   1 enables localhost-only GET /debug/request_history for the recent in-memory
#   history. Remote clients are rejected.
REQUEST_HISTORY_DEBUG_ENDPOINT = _env("REQUEST_HISTORY_DEBUG_ENDPOINT", "1") == "1"

# PT_PASSTHROUGH_LIVE_DEFAULT_PROFILE:
#   Fallback live response profile for clients that are not explicitly matched.
#   This deliberately preserves the legacy behavior as the observation baseline;
#   switching unknown clients to strict_live remains a Phase 3 real-device data
#   decision.
#   Values:
#     vlc    - direct streaming path with no shared live-session cache.
#     libmpv - managed live-session path with first-chunk gating and reuse.
#   Default: vlc. This is the safer choice for unknown players because it
#   avoids the busy-lock and cache coupling used by libmpv/Skybox clients.
PASSTHROUGH_LIVE_DEFAULT_PROFILE = _env("PASSTHROUGH_LIVE_DEFAULT_PROFILE", "vlc").lower()


# ---- Matting and startup warmup ----
# PT_MODEL_PATH:
#   ONNX matting model path. Realtime defaults to RVM MobileNetV3 FP32 while
#   investigating FP16 alpha stability on some 2D sources.
MODEL_PATH: Path = Path(_env("MODEL_PATH", ROOT / "models" / "rvm_mobilenetv3_fp32.onnx")).resolve()

# PT_MATTING_DEVICE:
#   Matting backend selection.
#   Values:
#     auto - prefer GPU when CuPy is available, otherwise CPU fallback.
#     gpu  - require GPU path and raise if unavailable.
#     cpu  - force CPU path.
MATTING_DEVICE = _env("MATTING_DEVICE", "auto").lower()

# PT_MATTING_SQUARE:
#   1 enables square crop before matting. Mostly useful for older non-RVM
#   experiments. Current RVM default is rectangular input.
MATTING_SQUARE = _env("MATTING_SQUARE", "0") == "1"

# PT_MATTING_SPLIT_SBS:
#   1 splits side-by-side VR sources into left/right halves before matting.
#   This is the production default for stereo sources.
MATTING_SPLIT_SBS = _env("MATTING_SPLIT_SBS", "1") == "1"

# PT_MATTING_SBS_BATCH:
#   1 batches both SBS halves into a single ORT call when the exported model
#   and current runtime shape support it. This remains the production default;
#   RVM also respects this switch, and batch=2 benchmarked faster than two
#   separate batch=1 eye inferences.
MATTING_SBS_BATCH = _env("MATTING_SBS_BATCH", "1") == "1"

# PT_MATTING_MODEL_KIND:
#   Explicit model family override.
#   Values:
#     auto - infer from the ONNX graph and input/output names.
#     rvm  - force the RVM recurrent path.
MATTING_MODEL_KIND = _env("MATTING_MODEL_KIND", "rvm").lower()

# PT_RVM_DOWNSAMPLE_RATIO:
#   Downsample factor used by the RVM recurrent path. 1.0 keeps full working
#   resolution. Lower values reduce cost but can soften masks.
RVM_DOWNSAMPLE_RATIO = float(_env("RVM_DOWNSAMPLE_RATIO", 0.5))

# PT_RVM_SCENE_RESET:
#   1 detects scene cuts via HSV (H,S) histogram Bhattacharyya distance and
#   resets RVM recurrent state across scene boundaries. Default off globally;
#   offline tools enable it explicitly.
RVM_SCENE_RESET = _env("RVM_SCENE_RESET", "0") == "1"

# PT_RVM_SCENE_THRESHOLD:
#   Bhattacharyya distance threshold; above this value triggers a scene cut.
RVM_SCENE_THRESHOLD = float(_env("RVM_SCENE_THRESHOLD", 0.4))

# PT_RVM_SCENE_COOLDOWN:
#   Minimum frames between two scene-cut resets.
RVM_SCENE_COOLDOWN = int(_env("RVM_SCENE_COOLDOWN", 24))

# PT_RVM_SCENE_REF_EMA:
#   EMA weight on the reference histogram for slow within-scene drift tracking.
RVM_SCENE_REF_EMA = float(_env("RVM_SCENE_REF_EMA", 0.95))

# PT_RVM_ALPHA_SMOOTH:
#   1 enables temporal EMA smoothing on RVM alpha output. Default off globally;
#   offline tools enable it explicitly.
RVM_ALPHA_SMOOTH = _env("RVM_ALPHA_SMOOTH", "0") == "1"

# PT_RVM_ALPHA_SMOOTH_WEIGHT:
#   EMA weight for historical alpha. 0.6 = 60% history + 40% new frame.
RVM_ALPHA_SMOOTH_WEIGHT = float(_env("RVM_ALPHA_SMOOTH_WEIGHT", 0.6))

# PT_MATANYONE2_SCENE_RESET:
#   1 detects scene cuts during MatAnyone2 offline prepass and merges detected
#   cuts into the segment plan when a bootstrap mask is available. This avoids
#   carrying propagation state across hard scene boundaries.
MATANYONE2_SCENE_RESET = _env("MATANYONE2_SCENE_RESET", "1") == "1"

# PT_MATANYONE2_SCENE_THRESHOLD:
#   HSV Bhattacharyya distance threshold; above this value triggers a scene cut.
MATANYONE2_SCENE_THRESHOLD = float(_env("MATANYONE2_SCENE_THRESHOLD", 0.4))

# PT_MATANYONE2_SCENE_COOLDOWN:
#   Minimum scanned frames between two MatAnyone2 scene-cut triggers.
MATANYONE2_SCENE_COOLDOWN = int(_env("MATANYONE2_SCENE_COOLDOWN", 24))

# PT_MATANYONE2_SCENE_REF_EMA:
#   EMA weight on the reference histogram for slow within-scene drift tracking.
MATANYONE2_SCENE_REF_EMA = float(_env("MATANYONE2_SCENE_REF_EMA", 0.95))

# PT_MATANYONE2_SCENE_MIN_SEGMENT_SEC:
#   Minimum spacing between two MatAnyone2 segment starts caused by scene cuts.
MATANYONE2_SCENE_MIN_SEGMENT_SEC = float(_env("MATANYONE2_SCENE_MIN_SEGMENT_SEC", 3.0))

# PT_MATANYONE2_SEGMENT_FRAMES:
#   Maximum MatAnyone2 propagation length before re-bootstrapping from the
#   prepass mask. Phase-1 state gating allows a longer default than the earlier
#   60-frame drag workaround while keeping bootstrap overhead lower.
MATANYONE2_SEGMENT_FRAMES = max(0, int(_env("MATANYONE2_SEGMENT_FRAMES", 240)))

# PT_MATANYONE2_LAST_MASK_UNCERT_GATE:
#   Scale the previous-frame last_mask down in high-uncertainty regions before
#   MatAnyone2 propagation. 0 disables the gate; typical values are 0.5-0.9.
MATANYONE2_LAST_MASK_UNCERT_GATE = max(0.0, min(1.0, float(_env("MATANYONE2_LAST_MASK_UNCERT_GATE", 0.7))))

# PT_MATANYONE2_SENSORY_DECAY_INTERVAL:
#   Soft-reset MatAnyone2 recurrent sensory state every N output frames. 0
#   disables the decay. This reduces hidden-state drag without segment reset.
MATANYONE2_SENSORY_DECAY_INTERVAL = max(0, int(_env("MATANYONE2_SENSORY_DECAY_INTERVAL", 8)))

# PT_MATANYONE2_SENSORY_DECAY_FACTOR:
#   Multiplier used by the sensory soft reset. Values below 0.7 can cause
#   flicker; the default is intentionally mild.
MATANYONE2_SENSORY_DECAY_FACTOR = max(0.0, min(1.0, float(_env("MATANYONE2_SENSORY_DECAY_FACTOR", 0.9))))

# PT_MATANYONE2_LAST_PRED_BINARIZE:
#   Use a thresholded previous mask for last_pred_mask, while keeping last_mask
#   soft. This decouples pred_uncertainty from alpha trails.
MATANYONE2_LAST_PRED_BINARIZE = _env("MATANYONE2_LAST_PRED_BINARIZE", "1") == "1"

# PT_MATANYONE2_LAST_PRED_BIN_THRESHOLD:
#   Threshold for the optional last_pred_mask binarization.
MATANYONE2_LAST_PRED_BIN_THRESHOLD = max(0.0, min(1.0, float(_env("MATANYONE2_LAST_PRED_BIN_THRESHOLD", 0.5))))

# PT_MATANYONE2_BOOTSTRAP_REFINE_ITERS:
#   Number of recurrent first-frame refinement passes used to build stronger
#   segment memory. 1 preserves the previous single-refine behavior.
MATANYONE2_BOOTSTRAP_REFINE_ITERS = max(1, int(_env("MATANYONE2_BOOTSTRAP_REFINE_ITERS", 3)))

# PT_MATANYONE2_ALPHA_SMOOTH:
#   1 enables temporal EMA smoothing on MatAnyone2 alpha output. Smoothers reset
#   whenever the MatAnyone2 segment plan resets.
MATANYONE2_ALPHA_SMOOTH = _env("MATANYONE2_ALPHA_SMOOTH", "0") == "1"

# PT_MATANYONE2_ALPHA_SMOOTH_WEIGHT:
#   EMA weight for historical alpha. 0.6 = 60% history + 40% new alpha.
MATANYONE2_ALPHA_SMOOTH_WEIGHT = float(_env("MATANYONE2_ALPHA_SMOOTH_WEIGHT", 0.6))

# PT_MATANYONE2_EDGE_AWARE_UPSAMPLE:
#   1 refines MatAnyone2 low-res alpha with a fast guided filter using the
#   uploaded NV12 Y plane as guide before green/alpha output. Default off after
#   smoke testing showed visible background halos on some foregrounds.
MATANYONE2_EDGE_AWARE_UPSAMPLE = _env("MATANYONE2_EDGE_AWARE_UPSAMPLE", "0") == "1"

# PT_MATANYONE2_GUIDED_RADIUS:
#   Box radius used by the MatAnyone2 guided alpha upsampler.
MATANYONE2_GUIDED_RADIUS = int(_env("MATANYONE2_GUIDED_RADIUS", 8))

# PT_MATANYONE2_GUIDED_EPS:
#   Regularization epsilon for the MatAnyone2 guided alpha upsampler.
MATANYONE2_GUIDED_EPS = float(_env("MATANYONE2_GUIDED_EPS", 0.0025))

# PT_MATANYONE2_GUIDED_FULLRES_SCALE:
#   Output scale for guided alpha refinement. 1.0 returns source-resolution
#   alpha; the default 0.5 keeps 8K cost bounded and lets the composite kernel
#   finish the remaining bilinear upscale.
MATANYONE2_GUIDED_FULLRES_SCALE = max(0.05, min(1.0, float(_env("MATANYONE2_GUIDED_FULLRES_SCALE", 0.5))))

# PT_MATANYONE2_GUIDED_SUPPORT_FLOOR:
#   Suppress guided-refine pixels where the original bilinear alpha has almost
#   no support. This prevents luma-guided background halos around the subject.
MATANYONE2_GUIDED_SUPPORT_FLOOR = max(0.0, min(1.0, float(_env("MATANYONE2_GUIDED_SUPPORT_FLOOR", 0.02))))

# PT_MATANYONE2_GUIDED_MAX_DELTA:
#   Clamp guided alpha growth over the original bilinear alpha. Negative values
#   disable the clamp. Default is conservative to avoid visible background rings.
MATANYONE2_GUIDED_MAX_DELTA = float(_env("MATANYONE2_GUIDED_MAX_DELTA", 0.08))

# PT_MATANYONE2_GUIDED_BAND_LO / PT_MATANYONE2_GUIDED_BAND_HI:
#   Only allow guided refinement in this base-alpha confidence band. Outside
#   the band, keep the original bilinear alpha to avoid luma-guided halos.
MATANYONE2_GUIDED_BAND_LO = max(0.0, min(1.0, float(_env("MATANYONE2_GUIDED_BAND_LO", 0.05))))
MATANYONE2_GUIDED_BAND_HI = max(0.0, min(1.0, float(_env("MATANYONE2_GUIDED_BAND_HI", 0.95))))
if MATANYONE2_GUIDED_BAND_HI < MATANYONE2_GUIDED_BAND_LO:
    MATANYONE2_GUIDED_BAND_LO, MATANYONE2_GUIDED_BAND_HI = MATANYONE2_GUIDED_BAND_HI, MATANYONE2_GUIDED_BAND_LO

# PT_MATANYONE2_ROI_CROP:
#   Experimental quality mode: crop/letterbox the segment foreground ROI to
#   the fixed MatAnyone2 input size. This can improve far-subject quality but
#   does not reduce ONNX token count or guarantee speedup.
MATANYONE2_ROI_CROP = _env("MATANYONE2_ROI_CROP", "0") == "1"

# PT_MATANYONE2_ROI_EXPAND:
#   Fraction of bbox size added to each side of the detected bootstrap mask ROI.
MATANYONE2_ROI_EXPAND = float(_env("MATANYONE2_ROI_EXPAND", 0.30))

# PT_MATANYONE2_ROI_MAX_EYE_FRACTION:
#   Fallback to full-eye path when the expanded ROI covers more than this
#   fraction of the eye area.
MATANYONE2_ROI_MAX_EYE_FRACTION = float(_env("MATANYONE2_ROI_MAX_EYE_FRACTION", 0.70))

# PT_MATANYONE2_ROI_FEATHER:
#   Feather width in full-eye pixels when pasting ROI alpha back to the eye.
MATANYONE2_ROI_FEATHER = int(_env("MATANYONE2_ROI_FEATHER", 16))

# PT_ALPHA_STRIDE:
#   Reuse the previous alpha mask for N-1 frames and recompute every Nth frame.
#   Production default 1 recomputes every frame for temporal fidelity.
ALPHA_STRIDE = max(1, int(_env("ALPHA_STRIDE", 1)))

# PT_ALPHA_MODE:
#   Alpha reuse strategy.
#   Values:
#     reuse - keep the previous mask until the next scheduled refresh.
#     refresh - recompute whenever the current frame differs enough.
ALPHA_MODE = _env("ALPHA_MODE", "reuse").lower()

# PT_ALPHA_CUTOFF:
#   Optional alpha threshold applied during final compositing. Default 0.0 keeps
#   the original soft mask. Use values around 0.35-0.55 to remove semi-
#   transparent edge pixels that make VR chroma-key playback look dirty.
ALPHA_CUTOFF = min(1.0, max(0.0, float(_env("ALPHA_CUTOFF", 0.0))))

# PT_ALPHA_HARD_EDGE:
#   1 turns PT_ALPHA_CUTOFF into a binary matte: alpha >= cutoff becomes fully
#   foreground, alpha < cutoff becomes fully background. This gives the cleanest
#   chroma-key edge, but can look jagged. Keep 0 to only drop weak alpha values.
ALPHA_HARD_EDGE = _env("ALPHA_HARD_EDGE", "0") == "1"

# PT_ALPHA_CONTRAST:
#   Multiplier around alpha 0.5 before cutoff. 1.0 keeps the model output.
#   Values like 1.5-3.0 steepen soft edges while retaining some antialiasing.
ALPHA_CONTRAST = max(0.01, float(_env("ALPHA_CONTRAST", 1.0)))

# PT_ALPHA_2D_ENABLE:
#   1 lets alpha passthrough convert non-SBS / non-2:1 flat videos into a
#   stereo fisheye SBS output. Existing SBS VR sources keep the original path.
ALPHA_2D_ENABLE = _env("ALPHA_2D_ENABLE", "1") == "1"

# PT_ALPHA_2D_PROJECTION:
#   Projection used for non-SBS / non-2:1 flat videos in alpha passthrough.
#     fisheye - existing flat-2D-to-fisheye SBS projection.
#     flat3d  - stereo SBS without fisheye projection; the image is centered in
#               a safe area so it does not overlap the alpha packer blocks.
ALPHA_2D_PROJECTION = _env("ALPHA_2D_PROJECTION", "fisheye").lower()

# PT_ALPHA_2D_FOV:
#   Field of view in degrees used when projecting a flat 2D video plane into
#   each fisheye eye. 90 keeps the plane natural and avoids excessive zoom.
ALPHA_2D_FOV = max(1.0, min(179.0, float(_env("ALPHA_2D_FOV", 80.0))))

# PT_ALPHA_2D_MAX_EYE_SIZE:
#   Maximum square size for one eye in flat-2D alpha passthrough. The full SBS
#   output is 2x this value by this value. Default 4096 keeps flat-2D alpha
#   output within an 8192x4096 HEVC/NVENC-friendly envelope.
ALPHA_2D_MAX_EYE_SIZE = max(2, int(_env("ALPHA_2D_MAX_EYE_SIZE", 4096)) & ~1)

# PT_ALPHA_2D_FLAT3D_SAFE_W / PT_ALPHA_2D_FLAT3D_SAFE_H:
#   In flat3d mode, the visible source image is centered inside this fraction of
#   one eye canvas. Defaults keep the source away from the alpha packer blocks.
ALPHA_2D_FLAT3D_SAFE_W = max(0.01, min(1.0, float(_env("ALPHA_2D_FLAT3D_SAFE_W", 0.7))))
ALPHA_2D_FLAT3D_SAFE_H = max(0.01, min(1.0, float(_env("ALPHA_2D_FLAT3D_SAFE_H", 0.6))))

# PT_ALPHA_2D_DISTANCE_M:
#   Stereo depth in meters for flat 2D alpha passthrough. Lower values create
#   more parallax. Runtime code converts this to pixels using the same IPD
#   formula as subtitles and the current output eye width.
ALPHA_2D_DISTANCE_M = max(0.1, float(_env("ALPHA_2D_DISTANCE_M", 4.0)))

# PT_TWO_DVR_MODEL:
#   DA3 depth model preset for 2D->3D: base/small (518) or base_hd/small_hd (1036).
TWO_DVR_MODEL = _env("TWO_DVR_MODEL", "base").strip().lower()
if TWO_DVR_MODEL not in {"small", "base", "small_hd", "base_hd"}:
    TWO_DVR_MODEL = "base"

# PT_TWO_DVR_HOLE_FILL:
#   Stereo hole-fill mode used by realtime 2D->3D flat SBS output.
TWO_DVR_HOLE_FILL = _env("TWO_DVR_HOLE_FILL", "soft_shift").strip().lower()
if TWO_DVR_HOLE_FILL not in {"inverse_warp", "soft_shift"}:
    TWO_DVR_HOLE_FILL = "soft_shift"

# PT_TWO_DVR_EYE_DISTANCE_MM:
#   Backward-compatible base eye distance for realtime 2D->3D live output.
#   User-facing UI keeps this at 65mm and exposes PT_TWO_DVR_STRENGTH instead.
TWO_DVR_EYE_DISTANCE_MM = max(1.0, float(_env("TWO_DVR_EYE_DISTANCE_MM", 65.0)))

# PT_TWO_DVR_STRENGTH:
#   User-facing 3D strength multiplier. 1.0 matches the default 65mm baseline;
#   render code still clamps final pixel disparity per output width.
TWO_DVR_STRENGTH = max(0.1, min(3.0, float(_env("TWO_DVR_STRENGTH", 1.0))))

# PT_TWO_DVR_BITRATE_MULT_3D / _VR:
#   Cap the 2D->3D output bitrate at this multiple of the SOURCE video bitrate.
#   SBS (flat3d) doubles the pixels, VR projections (fisheye/hequirect) stretch
#   more, so 3x / 4x of source keeps quality without the heavy over-spend that
#   made high-bitrate output stutter on playback. The configured/--bitrate value
#   is still the ceiling. 0 disables the cap (use the configured bitrate as-is).
TWO_DVR_BITRATE_MULT_3D = max(0.0, float(_env("TWO_DVR_BITRATE_MULT_3D", 3.0)))
TWO_DVR_BITRATE_MULT_VR = max(0.0, float(_env("TWO_DVR_BITRATE_MULT_VR", 4.0)))

# PT_TWO_DVR_SCENE_CUT / _THRESHOLD:
#   Reset the depth temporal state (normalization band, base EMA, motion comp) on
#   a detected hard cut, so the new shot doesn't blend with the previous one's
#   depth. Uses the HSV-histogram SceneCutDetector. 0 disables.
TWO_DVR_SCENE_CUT = _env("TWO_DVR_SCENE_CUT", "1") != "0"
TWO_DVR_SCENE_CUT_THRESHOLD = max(0.0, float(_env("TWO_DVR_SCENE_CUT_THRESHOLD", 0.4)))

# PT_TWO_DVR_BAND_LOOKAHEAD:
#   Offline only (needs the symmetric window). Replace the causal 5/95 depth-range
#   EMA with a zero-phase symmetric smoothing over the window's raw bands -- the
#   normalization band sees future frames, removing the causal EMA's lag. The
#   causal band EMA is turned off and the window re-normalizes each frame to the
#   symmetric band. 0 keeps the causal band EMA.
TWO_DVR_BAND_LOOKAHEAD = _env("TWO_DVR_BAND_LOOKAHEAD", "1") != "0"

# PT_TWO_DVR_TEMPORAL_NORM:
#   1 smooths the depth percentile normalization band across frames. This is the
#   default because per-frame normalization directly amplifies DA3 scale flicker.
TWO_DVR_TEMPORAL_NORM = _env("TWO_DVR_TEMPORAL_NORM", "1") != "0"

# PT_TWO_DVR_TEMPORAL_NORM_ALPHA:
#   EMA alpha for temporal normalization. Lower values are steadier but slower
#   to adapt; scene-cut reset below prevents long tails on large jumps.
TWO_DVR_TEMPORAL_NORM_ALPHA = max(0.0, min(1.0, float(_env("TWO_DVR_TEMPORAL_NORM_ALPHA", 0.10))))

# PT_TWO_DVR_TEMPORAL_NORM_RESET:
#   Reset normalization state when raw lo/hi moves by this many previous band
#   spans. 1.0 catches hard scene changes without resetting on ordinary motion.
TWO_DVR_TEMPORAL_NORM_RESET = max(0.0, float(_env("TWO_DVR_TEMPORAL_NORM_RESET", 1.0)))

# PT_TWO_DVR_DEPTH_STABILIZER / PT_TWO_DVR_TEMPORAL_DEPTH / PT_TWO_DVR_TEMPORAL_DEPTH_MODE:
#   Stabilize the normalized near/disparity map itself. `flow` aligns the
#   previous near map to the current frame using OpenCV Farneback flow before
#   blending, which targets DA3's per-frame disparity flicker. Set
#   PT_TWO_DVR_DEPTH_STABILIZER=0, PT_TWO_DVR_TEMPORAL_DEPTH=0, or mode=off to
#   hard-bypass the stabilizer. Default is on: the stabilizer is now the
#   base/detail rewrite (smooth low-frequency base, re-inject current detail),
#   which fixes the V1-lite foreground locking + soft_shift hole artifacts. The
#   per-pixel px limiters below stay 0 by default.
_TWO_DVR_TEMPORAL_DEPTH_FLAG = _env_any(("TWO_DVR_DEPTH_STABILIZER", "TWO_DVR_TEMPORAL_DEPTH"), "1") != "0"
TWO_DVR_TEMPORAL_DEPTH_MODE = str(
    _env("TWO_DVR_TEMPORAL_DEPTH_MODE", "ema" if _TWO_DVR_TEMPORAL_DEPTH_FLAG else "off")
).strip().lower()
if TWO_DVR_TEMPORAL_DEPTH_MODE not in {"off", "ema", "flow"}:
    TWO_DVR_TEMPORAL_DEPTH_MODE = "ema" if _TWO_DVR_TEMPORAL_DEPTH_FLAG else "off"
TWO_DVR_TEMPORAL_DEPTH = _TWO_DVR_TEMPORAL_DEPTH_FLAG and TWO_DVR_TEMPORAL_DEPTH_MODE != "off"
TWO_DVR_TEMPORAL_DEPTH_ALPHA = max(0.0, min(1.0, float(_env("TWO_DVR_TEMPORAL_DEPTH_ALPHA", 0.20))))

# PT_TWO_DVR_TEMPORAL_FLOW_*:
#   Flow-mode rejection gates. DIFF rejects warped previous pixels when luma
#   changes too much; CONSISTENCY enables optional forward/backward flow check;
#   MOTION_GATE raises current-frame weight for very large motion. 0 disables.
TWO_DVR_TEMPORAL_FLOW_DIFF = max(0.0, float(_env("TWO_DVR_TEMPORAL_FLOW_DIFF", 35.0)))
TWO_DVR_TEMPORAL_FLOW_CONSISTENCY = max(0.0, float(_env("TWO_DVR_TEMPORAL_FLOW_CONSISTENCY", 0.0)))
TWO_DVR_TEMPORAL_FLOW_MOTION_GATE = max(0.0, float(_env("TWO_DVR_TEMPORAL_FLOW_MOTION_GATE", 0.0)))

# PT_TWO_DVR_TEMPORAL_AFFINE:
#   1 enables a cheap global scale/bias correction of the current near map
#   against the previous stable near map, estimated on luma-stable pixels. This
#   specifically targets subtle whole-frame disparity gain drift.
TWO_DVR_TEMPORAL_AFFINE = _env("TWO_DVR_TEMPORAL_AFFINE", "1") != "0"
TWO_DVR_TEMPORAL_AFFINE_MAX_SCALE = max(0.0, float(_env("TWO_DVR_TEMPORAL_AFFINE_MAX_SCALE", 0.20)))
TWO_DVR_TEMPORAL_AFFINE_MAX_BIAS = max(0.0, float(_env("TWO_DVR_TEMPORAL_AFFINE_MAX_BIAS", 0.12)))

# PT_TWO_DVR_TEMPORAL_*_PX:
#   Stabilizer thresholds in output stereo disparity pixels. These are converted
#   to near-map units using the current max_shift, so the same setting behaves
#   consistently across output sizes and 3D strengths.
TWO_DVR_TEMPORAL_STATIC_DEADBAND_PX = max(0.0, float(_env("TWO_DVR_TEMPORAL_STATIC_DEADBAND_PX", 0.0)))
TWO_DVR_TEMPORAL_STATIC_MAX_STEP_PX = max(0.0, float(_env("TWO_DVR_TEMPORAL_STATIC_MAX_STEP_PX", 0.0)))
TWO_DVR_TEMPORAL_MOTION_MAX_STEP_PX = max(0.0, float(_env("TWO_DVR_TEMPORAL_MOTION_MAX_STEP_PX", 0.0)))

# PT_RVM_IOBINDING:
#   1 enables ORT IOBinding for the RVM path. This usually reduces copies and
#   improves GPU throughput.
RVM_IOBINDING = _env("RVM_IOBINDING", "1") == "1"

# PT_TRT_RVM_IOBINDING:
#   1 allows the experimental TensorRT + RVM IOBinding path. The default stays
#   off because ORT TensorRT EP can hang in run_with_iobinding on this model.
TRT_RVM_IOBINDING = _env("TRT_RVM_IOBINDING", "0") == "1"

# PT_MATANYONE2_IOBINDING:
#   1 enables ORT IOBinding for the MatAnyone2 offline step_update hot path.
#   The implementation is limited to batch=1 and automatically falls back to
#   the NumPy path on the first runtime failure.
MATANYONE2_IOBINDING = _env("MATANYONE2_IOBINDING", "1") == "1"

# PT_CUDA_SHARED_STREAM:
#   1 reuses a shared CuPy CUDA stream for matting and composite kernels.
CUDA_SHARED_STREAM = _env("CUDA_SHARED_STREAM", "1") == "1"

# PT_FAST_UV_ALPHA:
#   1 uses the faster UV alpha kernel variant during NV12 compositing. This is
#   an optional performance/quality tradeoff.
FAST_UV_ALPHA = _env("FAST_UV_ALPHA", "0") == "1"

# PT_SPLIT_NV12_COMPOSITE:
#   1 splits Y and UV composite work into separate kernels for A/B testing.
SPLIT_NV12_COMPOSITE = _env("SPLIT_NV12_COMPOSITE", "0") == "1"

# PT_CUDA_CUDNN_CONV_ALGO_SEARCH:
#   Optional ONNX Runtime CUDA provider option for cuDNN convolution search.
#   Leave empty to keep the provider default.
CUDA_CUDNN_CONV_ALGO_SEARCH = _env("CUDA_CUDNN_CONV_ALGO_SEARCH", "")

# PT_MATTING_INPUT_SIZE:
#   Reference model input size. RVM defaults to 1024 for matte quality.
#   2048 + PT_RVM_DOWNSAMPLE_RATIO=0.125 was faster in realtime benchmarks,
#   but quality validation showed the mask became too weak.
#   Set 0 only for RVM experiments that feed the current source/eye working
#   size directly and rely on PT_RVM_DOWNSAMPLE_RATIO.
MATTING_INPUT_SIZE = int(_env("MATTING_INPUT_SIZE", 1024 if MATTING_MODEL_KIND == "rvm" else 512))

# PT_MATTING_WARMUP_RUNS:
#   Number of dummy matting runs inside Matter initialization. This is distinct
#   from startup GPU cache warmup. Keep >=1 to avoid first stream including a
#   small local warmup cost.
MATTING_WARMUP_RUNS = int(_env("MATTING_WARMUP_RUNS", 1))

# PT_STARTUP_GPU_WARMUP:
#   1 warms CUDA/CuPy/ORT before opening the DLNA HTTP port. This hides expensive
#   ORT CUDA first-run JIT from the first playback request. 0 disables it.
STARTUP_GPU_WARMUP = _env("STARTUP_GPU_WARMUP", "1") == "1"

# PT_STARTUP_GPU_WARMUP_FORCE:
#   0 skips warmup when the marker key matches. 1 always runs warmup.
STARTUP_GPU_WARMUP_FORCE = _env("STARTUP_GPU_WARMUP_FORCE", "0") == "1"

# PT_STARTUP_GPU_WARMUP_TIMEOUT:
#   Seconds to wait for the global warmup lock before failing startup.
STARTUP_GPU_WARMUP_TIMEOUT = float(_env("STARTUP_GPU_WARMUP_TIMEOUT", 300))

# PT_STARTUP_GPU_WARMUP_RUNS_PER_SHAPE:
#   Number of warmup runs per configured shape. More runs provide stronger
#   second-pass verification but increase startup time.
STARTUP_GPU_WARMUP_RUNS_PER_SHAPE = int(_env("STARTUP_GPU_WARMUP_RUNS_PER_SHAPE", 3))

# PT_ONNX_PROVIDERS:
#   Comma-separated ONNX Runtime provider preference list.
#   Common values:
#     CUDAExecutionProvider,CPUExecutionProvider
#       Production NVIDIA path. Requires onnxruntime-gpu, CUDA runtime, cuDNN.
#     DmlExecutionProvider,CPUExecutionProvider
#       Windows DirectML fallback path. Slower but broadly available.
#     CPUExecutionProvider
#       Debug-only CPU path.
ONNX_PROVIDERS: list[str] = _env(
    "ONNX_PROVIDERS",
    "CUDAExecutionProvider,CPUExecutionProvider",
).split(",")

# PT_ONNX_TRT_ENGINE_CACHE_ENABLE / PATH:
#   Optional TensorRT EP settings when PT_ONNX_PROVIDERS includes
#   TensorrtExecutionProvider. Keep disabled unless explicitly A/B testing TRT
#   because the first engine build can take significant startup time.
ONNX_TRT_ENGINE_CACHE_ENABLE = _env("ONNX_TRT_ENGINE_CACHE_ENABLE", "1") == "1"
ONNX_TRT_ENGINE_CACHE_PATH: Path = Path(
    _env("ONNX_TRT_ENGINE_CACHE_PATH", ROOT / "runtime_cache" / "trt_engines")
).resolve()
# TensorRT builds default to FP32. FP16 TRT engines have shown alpha flicker
# on some RVM and MatAnyone2 paths.
ONNX_TRT_FP16_ENABLE = _env("ONNX_TRT_FP16_ENABLE", "0") == "1"
ONNX_TRT_CUDA_GRAPH_ENABLE = _env("ONNX_TRT_CUDA_GRAPH_ENABLE", "0") == "1"
ONNX_TRT_DUMP_SUBGRAPHS = _env("ONNX_TRT_DUMP_SUBGRAPHS", "0") == "1"
ONNX_TRT_DETAILED_BUILD_LOG = _env("ONNX_TRT_DETAILED_BUILD_LOG", "0") == "1"

# PT_PASSTHROUGH_PYNV_SYNC_PROBE:
#   1 enables extra CUDA synchronizations inside the PyNv green matting path to
#   attribute the normal outer sync wait to upload, RVM/ORT, or composite. This
#   is a diagnostic mode and intentionally changes timing.
PASSTHROUGH_PYNV_SYNC_PROBE = _env("PASSTHROUGH_PYNV_SYNC_PROBE", "0") == "1"

# PT_WARMUP_RAMPUP_DIAG_FRAMES:
#   Number of first PyNv GPU composite calls to log with synchronous ramp-up
#   timing. Diagnostic only; 0 keeps the hot path unchanged.
WARMUP_RAMPUP_DIAG_FRAMES = max(0, int(_env("WARMUP_RAMPUP_DIAG_FRAMES", 0)))

# PT_WARMUP_COMPOSITE_ENABLE:
#   1 warms the CuPy upload/composite/alpha-pack kernels during startup GPU
#   warmup so the first real stream does not pay their JIT/allocation cost.
WARMUP_COMPOSITE_ENABLE = _env("WARMUP_COMPOSITE_ENABLE", "1") == "1"

# PT_WARMUP_COMPOSITE_GEOMETRIES:
#   Semicolon-separated HxW list of representative source geometries to warm.
#   Defaults cover common SBS 8K and 4K VR180 sources.
_WARMUP_COMPOSITE_GEOMETRIES_RAW = _env("WARMUP_COMPOSITE_GEOMETRIES", "4096x8192;2048x4096")
WARMUP_COMPOSITE_GEOMETRIES: list[tuple[int, int]] = []
for _warmup_geometry in _WARMUP_COMPOSITE_GEOMETRIES_RAW.replace(",", ";").split(";"):
    _warmup_geometry = _warmup_geometry.strip().lower()
    if not _warmup_geometry or "x" not in _warmup_geometry:
        continue
    try:
        _gh, _gw = _warmup_geometry.split("x", 1)
        _h = max(2, int(_gh) & ~1)
        _w = max(2, int(_gw) & ~1)
        WARMUP_COMPOSITE_GEOMETRIES.append((_h, _w))
    except ValueError:
        pass
if not WARMUP_COMPOSITE_GEOMETRIES:
    WARMUP_COMPOSITE_GEOMETRIES = [(4096, 8192)]

# PT_NVENC_PREFLIGHT_ENABLE:
#   1 creates and releases representative NVENC encoders during startup so the
#   first real stream does not pay process-level NVENC SDK initialization.
NVENC_PREFLIGHT_ENABLE = _env("NVENC_PREFLIGHT_ENABLE", "1") == "1"

# PT_NVENC_PREFLIGHT_GEOMETRIES:
#   Semicolon-separated WxH@FPS:BITRATE list. Defaults cover full 8K SBS alpha
#   and downscaled 4K SBS alpha using the current low-latency encoder settings.
_NVENC_PREFLIGHT_GEOMETRIES_RAW = _env("NVENC_PREFLIGHT_GEOMETRIES", "8192x4096@59.940060:50000000;4096x2048@59.940060:25000000")
NVENC_PREFLIGHT_GEOMETRIES: list[tuple[int, int, str, str]] = []
for _nvenc_geometry in _NVENC_PREFLIGHT_GEOMETRIES_RAW.replace(",", ";").split(";"):
    _nvenc_geometry = _nvenc_geometry.strip().lower()
    if not _nvenc_geometry or "x" not in _nvenc_geometry:
        continue
    try:
        _size, _rest = _nvenc_geometry.split("@", 1)
        _fps, _bitrate = _rest.split(":", 1)
        _w_raw, _h_raw = _size.split("x", 1)
        _w = max(2, int(_w_raw) & ~1)
        _h = max(2, int(_h_raw) & ~1)
        NVENC_PREFLIGHT_GEOMETRIES.append((_w, _h, str(float(_fps)), str(int(_bitrate))))
    except ValueError:
        pass
if not NVENC_PREFLIGHT_GEOMETRIES:
    NVENC_PREFLIGHT_GEOMETRIES = [(8192, 4096, "59.940060", "50000000")]

# PT_PASSTHROUGH_RVM_BYPASS_ALPHA:
#   Diagnostic only. 1 bypasses RVM inference on the PyNv/CuPy green path and
#   uses an all-foreground alpha mask. This isolates decode/composite/encode
#   throughput and must not be used as a production visual mode.
PASSTHROUGH_RVM_BYPASS_ALPHA = _env("PASSTHROUGH_RVM_BYPASS_ALPHA", "0") == "1"

# PT_RUNTIME_CACHE_DIR:
#   Root directory for all runtime-generated caches and warmup markers.
RUNTIME_CACHE_DIR: Path = Path(_env("RUNTIME_CACHE_DIR", ROOT / "runtime_cache")).resolve()

# PT_CUDA_CACHE_PATH:
#   CUDA JIT cache directory used by the CUDA driver.
CUDA_CACHE_PATH: Path = Path(_env("CUDA_CACHE_PATH", RUNTIME_CACHE_DIR / "cuda_compute_cache")).resolve()

# PT_CUPY_CACHE_DIR:
#   CuPy kernel cache directory.
CUPY_CACHE_DIR: Path = Path(_env("CUPY_CACHE_DIR", RUNTIME_CACHE_DIR / "cupy")).resolve()

# PT_ORT_CACHE_DIR:
#   Optional ONNX Runtime cache directory for provider artifacts.
ORT_CACHE_DIR: Path = Path(_env("ORT_CACHE_DIR", RUNTIME_CACHE_DIR / "ort")).resolve()

# PT_GPU_WARMUP_MARKER:
#   JSON marker written after warmup succeeds for a particular machine/model key.
GPU_WARMUP_MARKER: Path = Path(_env("GPU_WARMUP_MARKER", RUNTIME_CACHE_DIR / "gpu_warmup_marker.json")).resolve()

# PT_GPU_WARMUP_LOCK:
#   File lock used to avoid multiple simultaneous warmup runs.
GPU_WARMUP_LOCK: Path = Path(_env("GPU_WARMUP_LOCK", RUNTIME_CACHE_DIR / ".warmup.lock")).resolve()

# PT_RUNTIME_TMP_DIR:
#   Temporary directory used while building cache artifacts and during warmup.
RUNTIME_TMP_DIR: Path = Path(_env("RUNTIME_TMP_DIR", RUNTIME_CACHE_DIR / "tmp")).resolve()

# PT_BITRATE_ESTIMATES:
#   JSON cache of per-file bitrate estimates used by pseudo-VOD size math.
BITRATE_ESTIMATES: Path = Path(_env("BITRATE_ESTIMATES", RUNTIME_CACHE_DIR / "bitrate_estimates.json")).resolve()

# PT_LIBRARY_INDEX_DB:
#   SQLite cache for DLNA directory listings and video metadata. Browse requests
#   still check directory/file stat data so external file changes invalidate the
#   affected rows without requiring a server restart.
LIBRARY_INDEX_DB: Path = Path(_env("LIBRARY_INDEX_DB", RUNTIME_CACHE_DIR / "library_index.sqlite")).resolve()

# Driver-facing cache variables. These remain overridable through the host
# environment, but config.py now owns the canonical defaults.
CUDA_CACHE_DISABLE = _env("CUDA_CACHE_DISABLE", "0")
CUDA_CACHE_MAXSIZE = _env("CUDA_CACHE_MAXSIZE", "4294967296")


# ---- Composite background color ----
# PT_COMPOSITE_BG_RGB:
#   Background color used after matting, written as RGB hex `RRGGBB` or
#   `#RRGGBB`. Default `00FF00` is the ChromaKey green used by the UI default.
#   Use gray values such as `808080` for players that key better on gray.
COMPOSITE_BG_RGB_HEX = _env("COMPOSITE_BG_RGB", "00FF00")
COMPOSITE_BG_RGB = _rgb_hex(COMPOSITE_BG_RGB_HEX, "00FF00")

# OpenCV/Numpy frames and legacy GPU BGR kernels use BGR channel order.
GREEN_BGR = (COMPOSITE_BG_RGB[2], COMPOSITE_BG_RGB[1], COMPOSITE_BG_RGB[0])


# ---- Foreground ambient light matching ----
# PT_LIGHT_MATCH_ENABLED:
#   Enables foreground-only color/luma correction for passthrough output.
LIGHT_MATCH_ENABLED = str(_env("LIGHT_MATCH_ENABLED", "0")).lower() in {"1", "true", "yes", "on"}
LIGHT_MATCH_TEMP_K = int(float(_env("LIGHT_MATCH_TEMP_K", "6500")))
LIGHT_MATCH_TINT = float(_env("LIGHT_MATCH_TINT", "0"))
LIGHT_MATCH_EXPOSURE_EV = float(_env("LIGHT_MATCH_EXPOSURE_EV", "0.0"))
LIGHT_MATCH_CONTRAST = float(_env("LIGHT_MATCH_CONTRAST", "1.0"))
LIGHT_MATCH_GAMMA = float(_env("LIGHT_MATCH_GAMMA", "1.0"))
LIGHT_MATCH_SATURATION = float(_env("LIGHT_MATCH_SATURATION", "1.0"))
LIGHT_MATCH_PRESET = str(_env("LIGHT_MATCH_PRESET", "daylight")).lower()
LIGHT_MATCH_FLUSH_QUEUES = str(_env("LIGHT_MATCH_FLUSH_QUEUES", "0")).lower() in {"1", "true", "yes", "on"}
LIGHT_MATCH_DICT = {
    "enabled": LIGHT_MATCH_ENABLED,
    "temp_k": LIGHT_MATCH_TEMP_K,
    "tint": LIGHT_MATCH_TINT,
    "exposure_ev": LIGHT_MATCH_EXPOSURE_EV,
    "contrast": LIGHT_MATCH_CONTRAST,
    "gamma": LIGHT_MATCH_GAMMA,
    "saturation": LIGHT_MATCH_SATURATION,
    "preset": LIGHT_MATCH_PRESET,
}

# ---- Simultaneous interpretation audio mixing ----
# PT_SI_MIX_ENABLED:
#   Exposes virtual [SI] MP4 items in DLNA for MP4 files with same-stem
#   `.si.wav` sidecar audio. Runtime changes are handled by /control/si_mix.
SI_MIX_ENABLED = str(
    _env_any(("SI_MIX_ENABLED", "DLNA_SI_ENABLED"), "1" if DEFAULT_SI_MIX_ENABLED else "0")
).lower() in {"1", "true", "yes", "on"}
# PT_SI_PROGRESSIVE_ENABLED:
#   Enables the M1 progressive virtual MP4 `/media_si` transport. Default 1 so
#   `[SI]` sidecar entries are immediately testable without enabling the legacy
#   `/passthrough_seek` experiment.
SI_PROGRESSIVE_ENABLED = str(_env("SI_PROGRESSIVE_ENABLED", "1")).lower() in {"1", "true", "yes", "on"}
# PT_SI_PROGRESSIVE_DLNA:
#   Adds `[SI]` progressive virtual MP4 entries to DLNA Browse when SI mixing is
#   enabled and a same-stem `.si.wav` exists. Set 0 to keep manual URLs only.
SI_PROGRESSIVE_DLNA = str(_env("SI_PROGRESSIVE_DLNA", "1")).lower() in {"1", "true", "yes", "on"}
# PT_SI_BROWSE_PREWARM_LIMIT:
#   Maximum number of `[SI]` items per DLNA Browse directory listing that may
#   enqueue background progressive MP4 prewarm. This keeps large folders from
#   queueing every SI sidecar at once. Set 0 to disable Browse-triggered prewarm.
SI_BROWSE_PREWARM_LIMIT = max(0, int(_env("SI_BROWSE_PREWARM_LIMIT", 1)))
# PT_SI_PREWARM_QUEUE_MAX:
#   Bounded background queue for low-priority SI progressive prewarm jobs. The
#   playback path still builds on demand with higher priority.
SI_PREWARM_QUEUE_MAX = max(0, int(_env("SI_PREWARM_QUEUE_MAX", 2)))
# PT_SI_AUDIO_EXTRACT_MODE:
#   Source-audio sidecar extraction strategy.
#   sequential - forward large-buffer scan over the source file span, avoiding
#                cold random seeks when audio samples are sparse.
#   runs       - old seek-per-audio-run extractor, useful for A/B diagnostics.
SI_AUDIO_EXTRACT_MODE = str(_env("SI_AUDIO_EXTRACT_MODE", "sequential")).strip().lower()
if SI_AUDIO_EXTRACT_MODE not in {"sequential", "runs"}:
    SI_AUDIO_EXTRACT_MODE = "sequential"
# PT_SI_MIX_PARALLEL_MAX:
#   Maximum ffmpeg segment encoders for SI mixed AAC sidecar builds. Set 1 to
#   force the single-process fallback path.
SI_MIX_PARALLEL_MAX = max(1, int(_env("SI_MIX_PARALLEL_MAX", min(8, os.cpu_count() or 1))))
# PT_SI_MIX_ENCODER:
#   auto   - prefer Windows MediaFoundation AAC when available, else ffmpeg aac.
#   aac    - ffmpeg native AAC.
#   aac_mf - Windows MediaFoundation AAC.
#   Default uses `auto` for best first-play responsiveness on Windows. Quest3
#   players validated the `aac_mf` fast path; use `aac` explicitly only for
#   quality/diagnostic A/B tests.
SI_MIX_ENCODER = str(_env("SI_MIX_ENCODER", "auto")).strip().lower()
if SI_MIX_ENCODER not in {"auto", "aac", "aac_mf"}:
    SI_MIX_ENCODER = "auto"
# PT_SI_MIX_SEGMENTED_AAC:
#   Enables experimental native-AAC segmented parallel encoding. Default off:
#   real-file PCM checks showed independent native AAC segments do not decode
#   sample-identically to one continuous native-AAC encode after seams.
SI_MIX_SEGMENTED_AAC = str(_env("SI_MIX_SEGMENTED_AAC", "0")).lower() in {"1", "true", "yes", "on"}
# PT_SI_MIX_SEGMENT_WARMUP_MS:
#   Leading warmup before non-first AAC mix segments. It absorbs encoder priming
#   and lets ducking/limiter envelopes settle before kept frames.
SI_MIX_SEGMENT_WARMUP_MS = max(0, int(_env("SI_MIX_SEGMENT_WARMUP_MS", 1000)))
# PT_SI_AUDIO_EDIT_MODE:
#   M1 compatibility switch for the imported AAC sidecar audio track.
#   preserve - copy the sidecar audio trak including edts/elst exactly.
#   remove   - drop audio trak edts/elst from the virtual moov only. This tests
#              players that mishandle AAC priming edit lists differently from
#              the source video's edit-list/ctts timeline.
SI_AUDIO_EDIT_MODE = str(_env("SI_AUDIO_EDIT_MODE", "remove")).strip().lower()
if SI_AUDIO_EDIT_MODE not in {"preserve", "remove"}:
    SI_AUDIO_EDIT_MODE = "remove"
SI_MIX_CHANNEL = str(_env_any(("SI_MIX_CHANNEL", "DLNA_SI_MIX_CHANNEL"), DEFAULT_SI_MIX_CHANNEL)).lower()
SI_ORIGINAL_VOLUME_PERCENT = int(float(_env_any(
    ("SI_ORIGINAL_VOLUME_PERCENT", "DLNA_SI_ORIGINAL_VOLUME_PERCENT"),
    DEFAULT_ORIGINAL_VOLUME_PERCENT,
)))
SI_VOLUME_PERCENT = int(float(_env_any(("SI_VOLUME_PERCENT", "DLNA_SI_VOLUME_PERCENT"), DEFAULT_SI_VOLUME_PERCENT)))
SI_DELAY_SECONDS = float(_env_any(("SI_DELAY_SECONDS", "DLNA_SI_DELAY_SECONDS"), DEFAULT_SI_DELAY_SECONDS))
SI_DUCK_ORIGINAL = str(_env_any(
    ("SI_DUCK_ORIGINAL", "DLNA_SI_DUCK_ORIGINAL"),
    "1" if DEFAULT_DUCK_ORIGINAL else "0",
)).lower() in {"1", "true", "yes", "on"}
SI_MIX_DICT = {
    "enabled": SI_MIX_ENABLED,
    "mix_channel": SI_MIX_CHANNEL,
    "original_volume_percent": SI_ORIGINAL_VOLUME_PERCENT,
    "si_volume_percent": SI_VOLUME_PERCENT,
    "si_delay_seconds": SI_DELAY_SECONDS,
    "duck_original": SI_DUCK_ORIGINAL,
}


# ---- Passthrough encoding and DLNA behavior ----
# PT_CONTAINER:
#   Container for the pseudo-VOD `/passthrough` path and FFmpeg mux helpers.
#   Values:
#     mp4    - fragmented MP4 (fMP4), useful for clients that expect MP4.
#     mpegts - MPEG-TS, useful for live-style DLNA playback.
#   `/passthrough_live` uses live-friendly route behavior regardless.
PASSTHROUGH_CONTAINER = _env("CONTAINER", "mp4").lower()

# PT_VCODEC:
#   FFmpeg fallback encoder. PyNv production output is always HEVC and does not
#   use this value for its encoder, but legacy fallback still does.
#   Common values: h264_nvenc, hevc_nvenc, libx264.
PASSTHROUGH_VCODEC = _env("VCODEC", "hevc_nvenc")

# PT_PASSTHROUGH_PRESET or PT_PRESET:
#   Encoder speed/quality preset for FFmpeg NVENC fallback. NVENC p1 is fastest,
#   p7 is slowest/best compression. p4 is a balanced default. libx264 uses names
#   like veryfast/faster/medium when selected.
PASSTHROUGH_PRESET = "p4"

# PT_PASSTHROUGH_BITRATE or PT_BITRATE:
#   Default FFmpeg fallback bitrate. Accepts ffmpeg-style strings such as 15M,
#   20M, 8000K, or integer bits/sec.
PASSTHROUGH_BITRATE = "20M"

# PT_PASSTHROUGH_HEVC_BITRATE:
#   PyNv HEVC target bitrate ceiling and fallback estimator bitrate for PyNv
#   output. The actual realtime target can be capped by source bitrate below.
PASSTHROUGH_HEVC_BITRATE = _env("PASSTHROUGH_HEVC_BITRATE", "50M")

# PT_PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER:
#   When source video bitrate is known, cap PyNv HEVC realtime target bitrate to
#   source_video_bps * this multiplier, while never exceeding
#   PT_PASSTHROUGH_HEVC_BITRATE. Set <=0 to disable source-based capping.
PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER = float(_env("PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER", 2.0))

# PT_PASSTHROUGH_HEVC_BF:
#   PyNv HEVC B-frame count. 0 minimizes latency and mux/seek complexity. Higher
#   values may improve compression but are not the current realtime default.
PASSTHROUGH_HEVC_BF = _env("PASSTHROUGH_HEVC_BF", "0")

# PT_PASSTHROUGH_PYNV_PRESET:
#   PyNvVideoCodec NVENC preset. P1 is the fastest SDK 10+ preset and is required
#   for 8K realtime headroom; lowercase p1 is not accepted by PyNv 2.1.0.
PASSTHROUGH_PYNV_PRESET = _env("PASSTHROUGH_PYNV_PRESET", "P1")

# PT_PASSTHROUGH_PYNV_TUNING_INFO:
#   PyNvVideoCodec tuning_info string. ultra_low_latency matches the live
#   streaming path and avoids the slow default/high-quality encoder settings.
PASSTHROUGH_PYNV_TUNING_INFO = _env("PASSTHROUGH_PYNV_TUNING_INFO", "ultra_low_latency")

# PT_PASSTHROUGH_PYNV_RC:
#   PyNvVideoCodec rate-control mode. cbr keeps bitrate predictable for live
#   MPEG-TS delivery. Leave empty to let PyNv choose its default.
PASSTHROUGH_PYNV_RC = _env("PASSTHROUGH_PYNV_RC", "cbr")

# PT_PASSTHROUGH_PYNV_IDR_PERIOD:
#   Optional PyNv idrperiod override. Empty keeps PyNv default/GOP behavior.
PASSTHROUGH_PYNV_IDR_PERIOD = _env("PASSTHROUGH_PYNV_IDR_PERIOD", "")

# PT_PASSTHROUGH_GOP or PT_GOP:
#   GOP/keyframe interval in output frames. With 30fps, 60 means one keyframe
#   about every 2 seconds.
PASSTHROUGH_GOP = 60

PASSTHROUGH_PRESET = _env_any(("PASSTHROUGH_PRESET", "PRESET"), PASSTHROUGH_PRESET)
PASSTHROUGH_BITRATE = _env_any(("PASSTHROUGH_BITRATE", "BITRATE"), PASSTHROUGH_BITRATE)
PASSTHROUGH_GOP = int(_env_any(("PASSTHROUGH_GOP", "GOP"), PASSTHROUGH_GOP))

# Optional FFmpeg NVENC fallback flags. Empty string means "do not pass this
# option" so FFmpeg chooses its default. Valid values differ between encoder,
# driver, and ffmpeg builds, so these remain opt-in A/B switches.
#
# PT_PASSTHROUGH_TUNE / PT_TUNE:
#   Examples: ull, ll, hq. Empty by default.
PASSTHROUGH_TUNE = _env_any(("PASSTHROUGH_TUNE", "TUNE"), "")

# PT_PASSTHROUGH_RC / PT_RC:
#   Rate-control mode. Examples: cbr, vbr, cbr_ld_hq. Empty by default.
PASSTHROUGH_RC = _env_any(("PASSTHROUGH_RC", "RC"), "")

# PT_PASSTHROUGH_RC_LOOKAHEAD / PT_RC_LOOKAHEAD:
#   NVENC lookahead frame count. 0 can reduce latency; empty keeps default.
PASSTHROUGH_RC_LOOKAHEAD = _env_any(("PASSTHROUGH_RC_LOOKAHEAD", "RC_LOOKAHEAD"), "")

# PT_PASSTHROUGH_BF / PT_BF:
#   FFmpeg fallback B-frame count. 0 reduces latency. Empty keeps default.
PASSTHROUGH_BF = _env_any(("PASSTHROUGH_BF", "BF"), "")

# PT_PASSTHROUGH_MULTIPASS / PT_MULTIPASS:
#   NVENC multipass mode. Empty keeps default; 0 disables when supported.
PASSTHROUGH_MULTIPASS = _env_any(("PASSTHROUGH_MULTIPASS", "MULTIPASS"), "")

# PT_PASSTHROUGH_NO_SCENECUT / PT_NO_SCENECUT:
#   NVENC scene-cut control. Often 1 for stricter GOP cadence, if supported.
PASSTHROUGH_NO_SCENECUT = _env_any(("PASSTHROUGH_NO_SCENECUT", "NO_SCENECUT"), "")

# PT_PASSTHROUGH_SPATIAL_AQ / PT_SPATIAL_AQ:
#   NVENC spatial adaptive quantization, typically 0/1.
PASSTHROUGH_SPATIAL_AQ = _env_any(("PASSTHROUGH_SPATIAL_AQ", "SPATIAL_AQ"), "")

# PT_PASSTHROUGH_TEMPORAL_AQ / PT_TEMPORAL_AQ:
#   NVENC temporal adaptive quantization, typically 0/1.
PASSTHROUGH_TEMPORAL_AQ = _env_any(("PASSTHROUGH_TEMPORAL_AQ", "TEMPORAL_AQ"), "")

# PT_PASSTHROUGH_SURFACES / PT_SURFACES:
#   Number of NVENC surfaces for FFmpeg fallback. Empty keeps FFmpeg default.
PASSTHROUGH_SURFACES = _env_any(("PASSTHROUGH_SURFACES", "SURFACES"), "")

# PT_PASSTHROUGH_DELAY / PT_DELAY:
#   Encoder delay option for some FFmpeg/NVENC builds. Empty keeps default.
PASSTHROUGH_DELAY = _env_any(("PASSTHROUGH_DELAY", "DELAY"), "")

# PT_PASSTHROUGH_ZERO_LATENCY / PT_ZERO_LATENCY:
#   Optional zero-latency flag for supported encoders, typically 0/1.
PASSTHROUGH_ZERO_LATENCY = _env_any(("PASSTHROUGH_ZERO_LATENCY", "ZERO_LATENCY"), "")

# PT_PASSTHROUGH_STRICT_GOP / PT_STRICT_GOP:
#   Request stricter GOP cadence for supported encoders, typically 0/1.
PASSTHROUGH_STRICT_GOP = _env_any(("PASSTHROUGH_STRICT_GOP", "STRICT_GOP"), "")

# PT_PASSTHROUGH_AUD / PT_AUD:
#   Access Unit Delimiter insertion for supported encoders, typically 0/1.
PASSTHROUGH_AUD = _env_any(("PASSTHROUGH_AUD", "AUD"), "")

# PT_PASSTHROUGH_AUDIO:
#   Audio handling for the PyNv production passthrough muxer.
#   Values:
#     copy - copy the first source audio stream into the output container.
#     aac  - transcode the first source audio stream to AAC.
#     off  - keep the previous video-only behavior.
#   The production muxer uses an explicit remaining duration instead of
#   `-shortest`; raw HEVC input plus `-shortest` can create an empty audio track.
PASSTHROUGH_AUDIO = _env("PASSTHROUGH_AUDIO", "copy").lower()

# PT_PASSTHROUGH_AUDIO_MPEGTS:
#   Audio handling when the PyNv production muxer outputs MPEG-TS, which is the
#   current `/passthrough_live` container. Default is `off` because several DLNA
#   clients are sensitive to startup buffering and stream layout changes in
#   live MPEG-TS. Set to `copy` or `aac` only for targeted audio compatibility
#   tests after video-only live playback is confirmed.
#   For live MPEG-TS, `aac` is preferred over `copy`: stream tests showed that
#   copied source audio can make FFmpeg hold the first TS bytes for many seconds,
#   which looks like a black loading screen to DLNA clients.
PASSTHROUGH_AUDIO_MPEGTS = _env("PASSTHROUGH_AUDIO_MPEGTS", "aac").lower()

# PT_PASSTHROUGH_AUDIO_MPEGTS_VLC:
#   Optional audio override for VLC/LibVLC/MoonVR live MPEG-TS requests.
#   Values are the same as PT_PASSTHROUGH_AUDIO_MPEGTS plus `auto`.
#     auto - use PT_PASSTHROUGH_AUDIO_MPEGTS.
#     off  - force video-only output for VLC/MoonVR if AAC timestamps regress.
#   Default auto. If MoonVR loads forever while server logs show MPEG-TS bytes
#   and `Non-monotonic DTS`, set this to `off` as a compatibility fallback.
PASSTHROUGH_AUDIO_MPEGTS_VLC = _env("PASSTHROUGH_AUDIO_MPEGTS_VLC", "auto").lower()

# PT_PASSTHROUGH_AUDIO_MPEGTS_SETTS:
#   Experimental AAC-in-MPEG-TS timestamp fix for the PyNv live path.
#   When enabled, FFmpeg applies a video bitstream timestamp filter:
#     setts=pts=N/(fps*TB):dts=N/(fps*TB)
#   This synthesizes strict CFR video timestamps for raw HEVC stdin and disables
#   the older wall-clock timestamp mode. Default 0 keeps the last stable
#   production behavior for existing players.
PASSTHROUGH_AUDIO_MPEGTS_SETTS = _env("PASSTHROUGH_AUDIO_MPEGTS_SETTS", "0") == "1"

# PT_PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE:
#   Video timestamp strategy for live MPEG-TS when audio is enabled.
#     wallclock - add -use_wallclock_as_timestamps 1. This is the old path and
#                 emits continuous TS, but can create non-monotonic DTS.
#     demux     - rely on the raw HEVC demuxer's -framerate CFR timestamps.
#                 Intended for AAC-cache tests without MP4 audio demux pressure.
#     setts     - apply the setts bitstream filter below. This stalled with MP4
#                 audio input, but can be retested with AAC cache.
#     pipe_ts   - two-stage experiment: first mux raw HEVC to video-only MPEG-TS
#                 with CFR timestamps, then mux that timestamped video stream
#                 with cached ADTS AAC. Current stable default despite first
#                 chunk latency; single-stage setts was incompatible with
#                 nPlayer/SKYBOX because output stalled after the first chunks.
#   If unset, PT_PASSTHROUGH_AUDIO_MPEGTS_SETTS=1 maps to setts; otherwise the
#   default uses the stable two-stage pipe_ts path.
PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE = _env(
    "PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE",
    "pipe_ts",
).lower()

# PT_PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE:
#   Optional FFmpeg input thread queue size for the experimental audio mux path.
#   0 leaves the command unchanged. Expert suggestion was 1024.
PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE = max(0, int(_env("PASSTHROUGH_AUDIO_MPEGTS_QUEUE_SIZE", 1024)))

# PT_PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE:
#   Raw HEVC demux packet size for the live MPEG-TS mux path.
#   Larger values help keep raw HEVC access units closer to packet boundaries.
#   65536 is a practical default for the current 8K S3D samples.
PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE = max(0, int(_env("PASSTHROUGH_AUDIO_MPEGTS_RAW_PACKET_SIZE", 65536)))

# PT_MUX_LATENCY_DIAG:
#   1 logs first-chunk mux timing marks for PyNv live streams. The log is
#   lightweight and helps separate encoder latency from FFmpeg mux probing.
MUX_LATENCY_DIAG = _env("MUX_LATENCY_DIAG", "1") == "1"

# PT_MUX_LATENCY_DIAG_VERBOSE:
#   1 enables extra FFmpeg stderr stage markers for first-chunk diagnostics.
#   T2 first-stderr markers remain enabled whenever PT_MUX_LATENCY_DIAG=1.
MUX_LATENCY_DIAG_VERBOSE = _env("MUX_LATENCY_DIAG_VERBOSE", "0") == "1"

# PT_MUX_FFMPEG_LOGLEVEL:
#   FFmpeg loglevel for PyNv live mux processes. Keep warning by default; set
#   info only for temporary first-chunk diagnostics.
MUX_FFMPEG_LOGLEVEL = _env("MUX_FFMPEG_LOGLEVEL", "warning").strip() or "warning"

# PT_FORCE_AUDIO_OFF:
#   Diagnostic override for PyNv live muxing. 1 forces video-only mux output to
#   compare single-stage mux startup against audio/pipe_ts startup.
FORCE_AUDIO_OFF = _env("FORCE_AUDIO_OFF", "0") == "1"

# PT_MUX_PROBESIZE_OVERRIDE:
#   Optional FFmpeg input probesize override for raw PyNv HEVC/H264 stdin.
#   Empty string omits the option and restores FFmpeg defaults. Raw HEVC/H264
#   inputs are explicitly declared, so the default 32 avoids stdin probe delay.
MUX_PROBESIZE_OVERRIDE = _env("MUX_PROBESIZE_OVERRIDE", "32").strip()

# PT_MUX_RAW_VIDEO_PROBESIZE:
#   Optional probesize override for PyNv raw HEVC/H264 stdin. Defaults to the
#   A8.P1.B 1MB winner. Do not test below 65536 because strict players
#   previously regressed when raw video probing was too small.
MUX_RAW_VIDEO_PROBESIZE = _env("MUX_RAW_VIDEO_PROBESIZE", "1000000").strip()

# PT_MUX_RAW_VIDEO_ANALYZEDURATION:
#   Optional analyzeduration override in microseconds for PyNv raw HEVC/H264
#   stdin. Defaults to the A8.P1.B 1s winner.
MUX_RAW_VIDEO_ANALYZEDURATION = _env("MUX_RAW_VIDEO_ANALYZEDURATION", "1000000").strip()

# PT_MUX_INTERMEDIATE_TS_PROBESIZE:
#   Optional probesize override for the intermediate MPEG-TS stdin used by the
#   pipe_ts final mux. Defaults to the A8.P2.A.1 16KB winner; 8192 showed no
#   nPlayer first-chunk gain and increased the post-output reader gap.
#   diagnostics. Do not test below 4096 because strict players previously
#   regressed to audio-only when intermediate TS probing was too small.
MUX_INTERMEDIATE_TS_PROBESIZE = _env("MUX_INTERMEDIATE_TS_PROBESIZE", "16384").strip()

# PT_MUX_INTERMEDIATE_TS_ANALYZEDURATION:
#   Optional analyzeduration override in microseconds for the intermediate
#   MPEG-TS stdin used by the pipe_ts final mux. Defaults to the A8.P2.A.1
#   first-chunk latency setting.
MUX_INTERMEDIATE_TS_ANALYZEDURATION = _env("MUX_INTERMEDIATE_TS_ANALYZEDURATION", "0").strip()

# PT_MUX_CONTAINER_PROBESIZE_OVERRIDE:
#   Optional FFmpeg input probesize override for already-muxed local container
#   inputs, especially the pipe_ts final-stage MPEG-TS stdin. 32768 avoids the
#   old 5MB default while leaving enough data for HEVC codec parameters.
MUX_CONTAINER_PROBESIZE_OVERRIDE = _env("MUX_CONTAINER_PROBESIZE_OVERRIDE", "32768").strip()

# PT_MUX_AUDIO_PROBESIZE_OVERRIDE:
#   Optional FFmpeg input probesize override for AAC/file audio inputs.
MUX_AUDIO_PROBESIZE_OVERRIDE = _env("MUX_AUDIO_PROBESIZE_OVERRIDE", "32768").strip()

# PT_MUX_ANALYZEDURATION_US:
#   Optional FFmpeg input analyzeduration override in microseconds. Empty string
#   omits the option and restores FFmpeg defaults.
MUX_ANALYZEDURATION_US = _env("MUX_ANALYZEDURATION_US", "0").strip()

# PT_MUX_NOBUFFER_ENABLE:
#   Historical first-chunk latency switch for raw HEVC/H264 FFmpeg mux inputs.
#   Keep it disabled by default: `+nobuffer` can make FFmpeg discard the first
#   GOP from raw HEVC before MPEG-TS muxing, shifting video content about one
#   second ahead of audio when PT_PASSTHROUGH_GOP=60.
MUX_NOBUFFER_ENABLE = _env("MUX_NOBUFFER_ENABLE", "0") == "1"

# PT_FMP4_FRAG_DURATION_US:
#   Fragment duration for fragmented MP4 output. Lower values reduce first
#   fragment latency; 250000 restores the previous default.
PASSTHROUGH_FMP4_FRAG_DURATION_US = max(1, int(_env("FMP4_FRAG_DURATION_US", 100000)))

# PT_PASSTHROUGH_AUDIO_MPEGTS_READRATE:
#   Optional FFmpeg audio input read rate. 0 leaves audio file reads unrestricted.
#   Expert suggestion was 1, but local probes showed it can delay the first TS
#   output by many seconds, so the default remains 0 for live startup.
PASSTHROUGH_AUDIO_MPEGTS_READRATE = max(0.0, float(_env("PASSTHROUGH_AUDIO_MPEGTS_READRATE", 0)))

# PT_PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA:
#   Optional override for FFmpeg -max_interleave_delta in the live MPEG-TS mux.
#   Empty string omits the option and lets FFmpeg use its default. The A8.2
#   0-value experiment did not reduce T4-T3c and has historical stall risk, so
#   the stable default remains 500000000.
PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA = _env("PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA", "500000000").strip()

# PT_PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE:
#   Optional AAC bitrate for live MPEG-TS audio transcoding, for example 192k.
#   Empty string lets FFmpeg choose its default.
PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE = _env("PASSTHROUGH_AUDIO_MPEGTS_AAC_BITRATE", "192k").strip()

# PT_PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_RATE:
#   Target live MPEG-TS AAC sample rate. Library scan on 2026-05-10 showed the
#   dominant source format is AAC LC 48 kHz stereo, so output is normalized to
#   that format while disk cache remains a fast ADTS copy.
PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_RATE = int(_env("PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_RATE", 48000))

# PT_PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS:
#   Target live MPEG-TS AAC channel count. Default 2 = stereo.
PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS = int(_env("PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS", 2))

# PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE:
#   1 extracts the first source audio stream into an ADTS AAC cache file before
#   live MPEG-TS muxing, then uses that cache for subsequent requests. This can
#   remove per-request MP4 audio demux overhead and avoid source-container edit
#   list differences while testing AAC sync behavior. Default 0 disables disk
#   cache reuse and lets the pipe_ts final mux read source audio directly.
PASSTHROUGH_AUDIO_MPEGTS_CACHE = _env("PASSTHROUGH_AUDIO_MPEGTS_CACHE", "0") == "1"

# PT_PASSTHROUGH_MPEGTS_VIDEO_SLATE:
#   Total switch for the live MPEG-TS video slate. 1 starts a generated
#   green/black video slate during AAC cache misses; 0 disables that video
#   padding path. PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE is kept as a legacy alias.
PASSTHROUGH_MPEGTS_VIDEO_SLATE = (
    _env_any(("PASSTHROUGH_MPEGTS_VIDEO_SLATE", "PASSTHROUGH_AUDIO_MPEGTS_SLATE"), "0") == "1"
)
PASSTHROUGH_AUDIO_MPEGTS_SLATE = PASSTHROUGH_MPEGTS_VIDEO_SLATE

# PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE_DIRECT_AFTER:
#   On an AAC cache miss, keep building the full source-level AAC cache in the
#   background, but stop waiting for it after this many seconds and feed the
#   current playback from a direct source demux. This keeps first playback from
#   showing a long green slate on large MP4 files.
PASSTHROUGH_AUDIO_MPEGTS_SLATE_DIRECT_AFTER = max(
    0.0,
    float(_env("PASSTHROUGH_AUDIO_MPEGTS_SLATE_DIRECT_AFTER", 1.0)),
)

# PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES:
#   Number of initial green-screen frames sent immediately so the MPEG-TS muxer
#   can see HEVC VPS/SPS/PPS quickly. Keep this low: every unpaced slate frame
#   can advance video PTS ahead of wall-clock audio during seek startup.
PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES = max(
    0,
    int(_env("PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES", 1)),
)

# PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE_DIR:
#   Directory for extracted ADTS AAC cache files. File identity includes source
#   absolute path, size, and mtime, so replacing a video invalidates its cache.
PASSTHROUGH_AUDIO_MPEGTS_CACHE_DIR: Path = Path(
    _env("PASSTHROUGH_AUDIO_MPEGTS_CACHE_DIR", RUNTIME_CACHE_DIR / "audio")
).resolve()

# PT_PASSTHROUGH_AUDIO_MPEGTS_PAT_PMT_AT_FRAMES:
#   1 adds FFmpeg mpegts flag pat_pmt_at_frames in addition to resend_headers.
#   This increases TS table repetition for stricter VLC/LibVLC clients that may
#   lock onto AAC but fail to expose the HEVC video PID during live startup.
PASSTHROUGH_AUDIO_MPEGTS_PAT_PMT_AT_FRAMES = (
    _env("PASSTHROUGH_AUDIO_MPEGTS_PAT_PMT_AT_FRAMES", "1") == "1"
)

# PT_PASSTHROUGH_MPEGTS_HEVC_AUD:
#   1 inserts HEVC Access Unit Delimiters before MPEG-TS muxing. Some
#   LibVLC/hardware-player paths are more reliable when HEVC-in-TS has AUD NALs
#   for access-unit boundary detection.
PASSTHROUGH_MPEGTS_HEVC_AUD = _env("PASSTHROUGH_MPEGTS_HEVC_AUD", "1") == "1"

# PT_PASSTHROUGH_PYNV_10BIT:
#   Experimental path for SDR Main10/P010/P016 sources. It decodes with PyNv,
#   downconverts GPU P016/P010 planes to 8-bit NV12, then reuses the existing
#   NV12 matting/HEVC path. HDR PQ/HLG remains blocked.
PASSTHROUGH_PYNV_10BIT = _env("PASSTHROUGH_PYNV_10BIT", "1") == "1"

# PT_PASSTHROUGH_PYNV_10BIT_SHIFT:
#   Right shift used for P016/P010 -> NV12 conversion. NVIDIA P016 output is
#   normally MSB-aligned, so 8 extracts the high byte. Use 2 only for diagnostic
#   testing if decoded values appear LSB-aligned.
PASSTHROUGH_PYNV_10BIT_SHIFT = int(_env("PASSTHROUGH_PYNV_10BIT_SHIFT", 8))

# PT_PASSTHROUGH_PYNV_DECODER:
#   Decoder used by the production PyNv live worker.
#     threaded_serial - sequential ThreadedDecoder pull with CFR source-frame
#                       dropping inside the existing serial worker.
#     simple          - legacy SimpleDecoder[index] random access.
PASSTHROUGH_PYNV_DECODER = _env("PASSTHROUGH_PYNV_DECODER", "simple").lower()

# PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER:
#   Diagnostic only. Alpha passthrough has shown red/gray alpha flicker with
#   ThreadedDecoder on some sources. Keep alpha on SimpleDecoder by default;
#   set to 1 only when explicitly testing threaded alpha.
PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER = _env("PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER", "0") == "1"

# PT_PASSTHROUGH_PYNV_THREADED_BATCH_SIZE / BUFFER_SIZE:
#   ThreadedDecoder tuning for the serial production reader. Returned frames
#   must still be consumed before the next get_batch_frames() call.
PASSTHROUGH_PYNV_THREADED_BATCH_SIZE = max(1, int(_env("PASSTHROUGH_PYNV_THREADED_BATCH_SIZE", 1)))
PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE = max(1, int(_env("PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE", 2)))

# PT_PASSTHROUGH_PYNV_WORKER_MODE:
#   serial    - existing single worker loop.
#   two_stage - experimental green-only overlap: decode+matting in one worker,
#               encode+mux in another. ThreadedDecoder frames never cross the
#               worker boundary; only Matter-owned NV12 slots are handed off.
PASSTHROUGH_PYNV_WORKER_MODE = _env("PASSTHROUGH_PYNV_WORKER_MODE", "serial").lower()

# PT_PASSTHROUGH_NV12_SLOT_WAIT_SEC:
#   Maximum wait for a Matter NV12 output slot in staged/overlapped workers.
PASSTHROUGH_NV12_SLOT_WAIT_SEC = max(0.0, float(_env("PASSTHROUGH_NV12_SLOT_WAIT_SEC", 5.0)))

# PT_PASSTHROUGH_LIVE_VLC_PSEUDO_VOD:
#   1 returns VLC/LibVLC/MoonVR live requests as 206 Partial Content with a
#   synthetic Content-Range. Do not send Content-Length for generated live
#   streams because early disconnects or startup failures make the body shorter
#   than the estimate and uvicorn rejects the response.
PASSTHROUGH_LIVE_VLC_PSEUDO_VOD = _env("PASSTHROUGH_LIVE_VLC_PSEUDO_VOD", "0") == "1"

# PT_PASSTHROUGH_LIVE_LAVF_POLICY:
#   Behavior for Lavf/FFmpeg live requests. MoonVR/LibVLC can issue these as
#   background thumbnail/probe requests through the same video endpoint; letting
#   them start production can steal the single NVENC slot from real playback.
#     active_only - reject Lavf only while a VLC/default stream is active.
#     reject      - always reject Lavf live requests.
#     allow       - let Lavf behave like a normal live client.
PASSTHROUGH_LIVE_LAVF_POLICY = _env("PASSTHROUGH_LIVE_LAVF_POLICY", "reject").lower()

# PT_PASSTHROUGH_MPEGTS_COLOR_RANGE:
#   Color range metadata to declare on generated MPEG-TS output. `source` keeps
#   ffprobe metadata, while `tv` forces limited-range signaling. MoonVR/LibVLC
#   showed audio-only playback on at least one full-range (`pc`) HEVC TS stream
#   even though server output was continuous.
PASSTHROUGH_MPEGTS_COLOR_RANGE = _env("PASSTHROUGH_MPEGTS_COLOR_RANGE", "tv").lower()

# PT_PASSTHROUGH_MAX_FPS:
#   Output/processing FPS cap. 0 keeps source FPS. Positive values cap output
#   frames, encoder/mux FPS, and live producer pacing. If the source FPS is
#   lower than the cap, realtime output keeps the source FPS. Set 0 to keep the
#   source frame rate unthrottled.
PASSTHROUGH_MAX_FPS = float(_env("PASSTHROUGH_MAX_FPS", 30))

# PT_PASSTHROUGH_REALTIME_PACING:
#   Backward-compatible alias for PT_PASSTHROUGH_SEND_REALTIME_PACING.
#
# PT_PASSTHROUGH_SEND_REALTIME_PACING:
#   1 paces HTTP live delivery near the configured output bitrate so players are
#   not flooded by a faster-than-realtime producer.
#
# PT_PASSTHROUGH_PRODUCER_REALTIME_PACING:
#   1 paces the PyNv worker itself to output FPS. A positive
#   PT_PASSTHROUGH_MAX_FPS also enables producer pacing so UI FPS caps limit the
#   actual live output rate, not only timestamps.
_REALTIME_PACING_DEFAULT = _env("PASSTHROUGH_REALTIME_PACING", "1")
PASSTHROUGH_SEND_REALTIME_PACING = _env("PASSTHROUGH_SEND_REALTIME_PACING", _REALTIME_PACING_DEFAULT) == "1"
PASSTHROUGH_PRODUCER_REALTIME_PACING = (
    _env("PASSTHROUGH_PRODUCER_REALTIME_PACING", "1" if PASSTHROUGH_MAX_FPS > 0 else "0") == "1"
)

# PT_PASSTHROUGH_SEND_PACING_MULTIPLIER:
#   Multiplier applied to estimated live output bitrate for HTTP send pacing.
#   MPEG-TS plus VBV bursts can exceed the nominal encoder target; a multiplier
#   gives the player buffer headroom without disabling pacing completely.
PASSTHROUGH_SEND_PACING_MULTIPLIER = max(
    1.0,
    float(_env("PASSTHROUGH_SEND_PACING_MULTIPLIER", 2.0)),
)

# PT_PASSTHROUGH_SEND_MIN_BPS:
#   Lower bound for paced live HTTP delivery.
PASSTHROUGH_SEND_MIN_BPS = max(1, int(_env("PASSTHROUGH_SEND_MIN_BPS", 100_000_000)))

# PT_DEBUG_LOGS:
#   0: suppress very frequent diagnostic/progress logs in server.log.
#   1: keep verbose per-stream progress diagnostics for troubleshooting.
DEBUG_LOGS = _env("DEBUG_LOGS", "0") == "1"

# PT_PASSTHROUGH_LIVE_ADAPTIVE_FPS:
#   1 enables live-only FPS lowering for extremely high source-bitrate files.
#   This avoids visible buffering when decode/input bandwidth is the bottleneck.
PASSTHROUGH_LIVE_ADAPTIVE_FPS = _env("PASSTHROUGH_LIVE_ADAPTIVE_FPS", "0") == "1"

# PT_PASSTHROUGH_LIVE_HIGH_BITRATE_BPS:
#   Source bitrate threshold for adaptive live FPS. Estimated from file size /
#   duration. Default 120 Mbps.
PASSTHROUGH_LIVE_HIGH_BITRATE_BPS = int(_env("PASSTHROUGH_LIVE_HIGH_BITRATE_BPS", 120_000_000))

# PT_PASSTHROUGH_LIVE_HIGH_BITRATE_FPS:
#   FPS selected for high-bitrate live sources when adaptive mode triggers.
#   Default 24. Global PASSTHROUGH_MAX_FPS still caps this if lower.
PASSTHROUGH_LIVE_HIGH_BITRATE_FPS = float(_env("PASSTHROUGH_LIVE_HIGH_BITRATE_FPS", 24))

# PT_PASSTHROUGH_FALLBACK_MAX_FPS:
#   FPS cap for the legacy FFmpeg fallback path used when PyNv rejects a source
#   such as Main10/P010. 0 means use PT_PASSTHROUGH_MAX_FPS.
PASSTHROUGH_FALLBACK_MAX_FPS = float(_env("PASSTHROUGH_FALLBACK_MAX_FPS", 24))

# PT_PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC:
#   Safety watchdog for live HTTP clients that stop reading after initial bytes
#   but do not close the TCP connection cleanly. When positive, the live stream
#   is closed if no response bytes advance for this many seconds after output
#   starts. Set to 0 to disable during low-level client tests.
PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC = max(0.0, float(_env("PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC", 0)))

# PT_PASSTHROUGH_LIVE_FIRST_CHUNK_TIMEOUT_SEC:
#   Timeout for the libmpv/Skybox path while waiting for the first live chunk
#   before headers are sent. Audio readrate experiments can need more than the
#   old hard-coded 15 seconds, so keep this configurable.
PASSTHROUGH_LIVE_FIRST_CHUNK_TIMEOUT_SEC = max(
    1.0,
    float(_env("PASSTHROUGH_LIVE_FIRST_CHUNK_TIMEOUT_SEC", 30)),
)

# PT_PASSTHROUGH_LIVE_VLC_PREROLL_BYTES:
#   For VLC/LibVLC/MoonVR, buffer this many initial MPEG-TS bytes before
#   returning the HTTP response. This gives stricter players enough PAT/PMT and
#   video PES data to initialize HEVC video instead of sometimes starting audio
#   only from a tiny first TS chunk. Set to 0 to disable.
PASSTHROUGH_LIVE_VLC_PREROLL_BYTES = max(
    0,
    int(_env("PASSTHROUGH_LIVE_VLC_PREROLL_BYTES", 1048576)),
)

# PT_PASSTHROUGH_LIVE_VLC_PREROLL_TIMEOUT_SEC:
#   Extra time after the first VLC/MoonVR TS chunk to wait for the preroll byte
#   target. If the target is not reached in time, the buffered data is still
#   sent rather than failing the request.
PASSTHROUGH_LIVE_VLC_PREROLL_TIMEOUT_SEC = max(
    0.0,
    float(_env("PASSTHROUGH_LIVE_VLC_PREROLL_TIMEOUT_SEC", 3)),
)

# PT_PASSTHROUGH_LIVE_CACHE_BYTES:
#   Maximum prefix bytes kept in memory for one `/passthrough_live` session.
#   This is mainly for strict clients such as libmpv/Skybox that first probe a
#   live MPEG-TS URL, disconnect, then immediately request the same URL again
#   for actual playback. The second request can replay this prefix without
#   starting another PyNv/NVENC pipeline. Default 128 MiB gives high-bitrate 8K
#   SBS streams enough probe headroom while concurrency remains 1. Set to 0 to
#   disable live prefix cache.
PASSTHROUGH_LIVE_CACHE_BYTES = max(0, int(_env("PASSTHROUGH_LIVE_CACHE_BYTES", 128 * 1024 * 1024)))

# PT_PASSTHROUGH_LIVE_CACHE_TTL_SEC:
#   How long a live session keeps running after the last HTTP subscriber leaves.
#   A short grace period lets probe/play retries reuse the same PyNv pipeline.
#   Keep this small because the GPU pipeline continues producing during grace.
PASSTHROUGH_LIVE_CACHE_TTL_SEC = max(0.0, float(_env("PASSTHROUGH_LIVE_CACHE_TTL_SEC", 10)))

# PT_PASSTHROUGH_LIVE_SUB_QUEUE_CHUNKS:
#   Per-subscriber live chunk queue depth. If a client stops reading and the
#   queue fills, that subscriber is dropped so it cannot block the shared PyNv
#   producer or other subscribers.
PASSTHROUGH_LIVE_SUB_QUEUE_CHUNKS = max(1, int(_env("PASSTHROUGH_LIVE_SUB_QUEUE_CHUNKS", 256)))

# PT_PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS:
#   Maximum number of DLNA live chapter entries shown inside one
#   `*-passthrough-live` chapter container.
PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS = max(1, int(_env("PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS", 10)))

# PT_PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC:
#   Minimum spacing between generated live chapter start points. Also acts as
#   the short-video threshold when only one useful chapter would be generated.
PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC = max(1, int(_env("PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC", 180)))

# PT_PASSTHROUGH_MKV_LIVE_POLICY:
#   MKV is risky for the current PyNv random-index live path. Files with Cues
#   near the head are allowed for diagnostics; files with missing/tail Cues are
#   hidden/rejected because PyNv SimpleDecoder[index] can block in native code.
#     block     - hide/reject all MKV live passthrough.
#     head_cues - allow only MKV whose Cues are detected near the file head.
#     allow     - allow MKV live passthrough for diagnostics.
PASSTHROUGH_MKV_LIVE_POLICY = _env("PASSTHROUGH_MKV_LIVE_POLICY", "block").lower()

# PT_PASSTHROUGH_DLNA_PN or PT_DLNA_PN:
#   Override DLNA.ORG_PN for passthrough resources. Leave empty for automatic
#   selection in dlna/profiles.py. Use only for target-client compatibility.
PASSTHROUGH_DLNA_PN = _env_any(("PASSTHROUGH_DLNA_PN", "DLNA_PN"), "")

# PT_SEEK_MODE:
#   Legacy pseudo-VOD `/passthrough` seek advertisement mode.
#     time  - advertise DLNA TimeSeekRange.
#     bytes - advertise estimated byte Range and map Range start to time.
#   The pseudo-VOD route is currently hidden from DLNA. Live chapter containers
#   are the preferred coarse seek strategy.
PASSTHROUGH_SEEK_MODE = _env("SEEK_MODE", "bytes").lower()

# PT_PASSTHROUGH_SEEK_ENABLED:
#   Master switch for the experimental seekable passthrough endpoint
#   `/passthrough_seek`. Default 0 keeps the existing `/passthrough_live`
#   behavior untouched. When this is 0, direct/manual `/passthrough_seek` URLs
#   are rejected even if DLNA exposure is enabled.
PASSTHROUGH_SEEK_ENABLED = _env("PASSTHROUGH_SEEK_ENABLED", "0") == "1"

# PT_PASSTHROUGH_SEEK_DLNA:
#   1 adds seekable passthrough entries to DLNA Browse while keeping the
#   existing live/chapter fallback items visible. This is only an advertisement
#   switch: it does not by itself enable the HTTP route. Keep it 0 for
#   hidden/manual URL testing while PT_PASSTHROUGH_SEEK_ENABLED=1.
PASSTHROUGH_SEEK_DLNA = _env("PASSTHROUGH_SEEK_DLNA", "0") == "1"

# PT_PASSTHROUGH_SEEK_ROUTE_POLICY:
#   Runtime guard for `/passthrough_seek` based on the request User-Agent:
#     profile - allow only route profiles listed in PT_PASSTHROUGH_SEEK_PROFILES.
#     all     - allow every client while the master switch is on.
#     off     - reject even when the master switch is on.
#   Aliases: auto/manual/list are treated as profile.
PASSTHROUGH_SEEK_ROUTE_POLICY = _env("PASSTHROUGH_SEEK_ROUTE_POLICY", "profile").lower()
if PASSTHROUGH_SEEK_ROUTE_POLICY in {"auto", "manual", "list"}:
    PASSTHROUGH_SEEK_ROUTE_POLICY = "profile"
if PASSTHROUGH_SEEK_ROUTE_POLICY not in {"profile", "all", "off"}:
    PASSTHROUGH_SEEK_ROUTE_POLICY = "profile"

# PT_PASSTHROUGH_SEEK_PROFILES:
#   Comma-separated live-response profiles allowed when route policy is
#   `profile`. Known values include nplayer, vlc, libmpv, 4xvr, avpro, lavf,
#   and default. This is the manual per-player rollout list.
PASSTHROUGH_SEEK_PROFILES = tuple(
    part.strip().lower()
    for part in str(_env("PASSTHROUGH_SEEK_PROFILES", "nplayer,vlc,libmpv")).replace(";", ",").split(",")
    if part.strip()
)

# PT_PASSTHROUGH_SEEK_CONTAINER:
#   Container emitted by the experimental `/passthrough_seek` endpoint.
#     mpegts - current default; true MPEG-TS bytes with byte-range mapping.
#     mp4    - true fragmented MP4 experiment for clients that refuse TS VOD.
#   Do not fake this with only headers: the body container must match.
PASSTHROUGH_SEEK_CONTAINER = _env("PASSTHROUGH_SEEK_CONTAINER", "mpegts").lower()
if PASSTHROUGH_SEEK_CONTAINER not in {"mpegts", "mp4"}:
    PASSTHROUGH_SEEK_CONTAINER = "mpegts"

# PT_PASSTHROUGH_SEEK_HEADER_BYTES:
#   Stable real prefix region for `/passthrough_seek`. Ranges intersecting this
#   region must be served from cached muxer bytes, not synthetic headers.
PASSTHROUGH_SEEK_HEADER_BYTES = max(0, int(_env("PASSTHROUGH_SEEK_HEADER_BYTES", 2 * 1024 * 1024)))

# PT_PASSTHROUGH_MAX_CONCURRENT:
#   Maximum concurrent passthrough streams. Each concurrent stream needs its own
#   Matter instance (independent ORT session + RVM recurrent state + GPU buffers),
#   roughly 1.5-2GB VRAM. Use "auto" to pick a value based on detected VRAM.
PASSTHROUGH_MAX_CONCURRENT = resolve_passthrough_max_concurrent(_env("PASSTHROUGH_MAX_CONCURRENT", "auto"))

# PT_PASSTHROUGH_BUSY_WAIT_SEC:
#   How long a new passthrough request waits for the active slot before 503.
#   Same-owner range/probe requests may preempt the prior stream.
PASSTHROUGH_BUSY_WAIT_SEC = max(0.0, float(_env("PASSTHROUGH_BUSY_WAIT_SEC", 10)))

# PT_PASSTHROUGH_PAD_TO_LENGTH:
#   Legacy pseudo-VOD compatibility option. Padding generated responses to an
#   estimated length is risky and mostly kept for controlled client experiments.
PASSTHROUGH_PAD_TO_LENGTH = _env("PASSTHROUGH_PAD_TO_LENGTH", "1") == "1"

# PT_PASSTHROUGH_NV12_RING_SLOTS:
#   GPU NV12 output slot count for passthrough compositing. Multiple slots let
#   the stream hand distinct output buffers to NVENC instead of reusing one
#   Matter-owned buffer every frame. Phase 3 default is 3.
PASSTHROUGH_NV12_RING_SLOTS = max(1, int(_env("PASSTHROUGH_NV12_RING_SLOTS", 3)))

# PT_PASSTHROUGH_CLOSE_WORKER_TIMEOUT_SEC:
#   Maximum time to wait for a PyNv passthrough worker to exit during close.
#   If it is still inside native decoder code after this, the stream attempts
#   decoder stop and logs the stuck thread for diagnostics.
PASSTHROUGH_CLOSE_WORKER_TIMEOUT_SEC = max(
    0.1,
    float(_env("PASSTHROUGH_CLOSE_WORKER_TIMEOUT_SEC", 3.0)),
)

# PT_USE_PYNV:
#   1 routes eligible CFR SDR 8-bit sources to PyNv decode/matting/HEVC encode.
#   0 forces the legacy FFmpeg fallback path.
USE_PYNV = _env("USE_PYNV", "1") == "1"


# ---- FFmpeg decode, mainly legacy path and benchmarks ----
# PT_HWACCEL:
#   FFmpeg decoder hardware acceleration for the legacy path and benchmarks.
#   Values:
#     auto    - let FFmpeg choose.
#     none    - force software decode.
#     cuda    - NVIDIA CUDA/NVDEC path.
#     d3d11va - Windows D3D11VA.
#     dxva2   - older Windows DXVA2.
FFMPEG_HWACCEL = _env("HWACCEL", "cuda")

# PT_HWACCEL_OUTPUT:
#   1 requests hardware frames for supported FFmpeg paths and downloads/formats
#   them where required by the old pipeline. 0 keeps simpler software-frame
#   behavior. Mostly useful for decode A/B diagnostics.
FFMPEG_HWACCEL_OUTPUT = _env("HWACCEL_OUTPUT", "1") == "1"

# PT_DECODE_MAX_SIDE:
#   Optional decode-stage downscale for the old FFmpeg rawvideo path.
#   0 keeps source size. Positive values clamp the longest edge while preserving
#   aspect ratio. PyNv production path preserves source resolution and is not
#   controlled by this old-path scaler.
DECODE_MAX_SIDE = int(_env("DECODE_MAX_SIDE", 4096))

# PT_DECODE_PIX_FMT:
#   Raw frame format produced by legacy DecoderProcess.
#   Values:
#     nv12  - preferred for GPU composite and lower pipe bandwidth.
#     bgr24 - older OpenCV-friendly CPU path.
DECODE_PIX_FMT = _env("DECODE_PIX_FMT", "nv12").lower()


# Ensure default local folders exist for direct developer runs.
for _video_dir in VIDEO_DIRS:
    _video_dir.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(parents=True, exist_ok=True)
