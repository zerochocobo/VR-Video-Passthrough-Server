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


VIDEO_DIRS: list[Path] = parse_video_dirs(_env("VIDEO_DIR", ROOT / "videos"), ROOT / "videos")
MEDIA_ROOTS = build_media_roots(VIDEO_DIRS)
MEDIA_LIBRARY = MediaLibrary(MEDIA_ROOTS)
VIDEO_DIR: Path = VIDEO_DIRS[0]

# File extensions considered video media during directory scans.
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}

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

# PT_PASSTHROUGH_LIVE_DEFAULT_PROFILE:
#   Fallback live response profile for clients that are not explicitly matched.
#   Values:
#     vlc    - direct streaming path with no shared live-session cache.
#     libmpv - managed live-session path with first-chunk gating and reuse.
#   Default: vlc. This is the safer choice for unknown players because it
#   avoids the busy-lock and cache coupling used by libmpv/Skybox clients.
PASSTHROUGH_LIVE_DEFAULT_PROFILE = _env("PASSTHROUGH_LIVE_DEFAULT_PROFILE", "vlc").lower()


# ---- Matting and startup warmup ----
# PT_MODEL_PATH:
#   ONNX matting model path. Current recommended production model is RVM
#   MobileNetV3 fp32. Typical value: models\rvm_mobilenetv3_fp32.onnx.
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
#   and current runtime shape support it.
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

# PT_ALPHA_STRIDE:
#   Reuse the previous alpha mask for N-1 frames and recompute every Nth frame.
#   Higher values improve throughput at the cost of temporal fidelity.
ALPHA_STRIDE = max(1, int(_env("ALPHA_STRIDE", 3)))

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

# PT_RVM_IOBINDING:
#   1 enables ORT IOBinding for the RVM path. This usually reduces copies and
#   improves GPU throughput.
RVM_IOBINDING = _env("RVM_IOBINDING", "1") == "1"

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
#   Reference model input size for non-RVM matting models. RVM production uses
#   its model default below and intentionally does not expose this as a startup
#   profile override; changing it can make 8K SBS masks miss small people.
MATTING_INPUT_SIZE = 1024 if MATTING_MODEL_KIND == "rvm" else int(_env("MATTING_INPUT_SIZE", 512))

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
#                 with cached ADTS AAC.
#   If unset, PT_PASSTHROUGH_AUDIO_MPEGTS_SETTS=1 maps to setts; otherwise the
#   default is wallclock for backward compatibility.
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

# PT_PASSTHROUGH_AUDIO_MPEGTS_READRATE:
#   Optional FFmpeg audio input read rate. 0 leaves audio file reads unrestricted.
#   Expert suggestion was 1, but local probes showed it can delay the first TS
#   output by many seconds, so the default remains 0 for live startup.
PASSTHROUGH_AUDIO_MPEGTS_READRATE = max(0.0, float(_env("PASSTHROUGH_AUDIO_MPEGTS_READRATE", 0)))

# PT_PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA:
#   Optional override for FFmpeg -max_interleave_delta in the live MPEG-TS mux.
#   Empty string omits the option and lets FFmpeg use its default. Avoid 0 for
#   AAC live muxing because it can deadlock interleaving with raw HEVC stdin.
#   A useful experimental override is 500000000.
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
#   list differences while testing AAC sync behavior.
PASSTHROUGH_AUDIO_MPEGTS_CACHE = _env("PASSTHROUGH_AUDIO_MPEGTS_CACHE", "1") == "1"

# PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE:
#   1 enables a green-screen slate during live MPEG-TS AAC cache misses. The
#   stream starts immediately with the final video resolution/codec and a silent
#   AAC input, then switches to real video/audio after the cache is ready.
PASSTHROUGH_AUDIO_MPEGTS_SLATE = _env("PASSTHROUGH_AUDIO_MPEGTS_SLATE", "1") == "1"

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
#   Number of initial green-screen frames sent without realtime pacing so the
#   MPEG-TS muxer can emit headers/data quickly.
PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES = max(
    0,
    int(_env("PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES", 90)),
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

# PT_PASSTHROUGH_REALTIME_PACING:
#   Backward-compatible alias for PT_PASSTHROUGH_SEND_REALTIME_PACING.
#
# PT_PASSTHROUGH_SEND_REALTIME_PACING:
#   1 paces HTTP live delivery near the configured output bitrate so players are
#   not flooded by a faster-than-realtime producer.
#
# PT_PASSTHROUGH_PRODUCER_REALTIME_PACING:
#   1 paces the PyNv worker itself to output FPS. Default 0 keeps the GPU
#   producer unthrottled so logs show real processing headroom above 30fps.
_REALTIME_PACING_DEFAULT = _env("PASSTHROUGH_REALTIME_PACING", "1")
PASSTHROUGH_SEND_REALTIME_PACING = _env("PASSTHROUGH_SEND_REALTIME_PACING", _REALTIME_PACING_DEFAULT) == "1"
PASSTHROUGH_PRODUCER_REALTIME_PACING = _env("PASSTHROUGH_PRODUCER_REALTIME_PACING", "0") == "1"

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

# PT_PASSTHROUGH_MAX_FPS:
#   Output/processing FPS cap. 0 keeps source FPS. Positive values cap output
#   frames and encoder/mux FPS. Current real-device profile uses 30.
PASSTHROUGH_MAX_FPS = float(_env("PASSTHROUGH_MAX_FPS", 30))

# PT_DEBUG_LOGS:
#   0: suppress very frequent diagnostic/progress logs in server.log.
#   1: keep verbose per-stream progress diagnostics for troubleshooting.
DEBUG_LOGS = _env("DEBUG_LOGS", "0") == "1"

# PT_PASSTHROUGH_LIVE_ADAPTIVE_FPS:
#   1 enables live-only FPS lowering for extremely high source-bitrate files.
#   This avoids visible buffering when decode/input bandwidth is the bottleneck.
PASSTHROUGH_LIVE_ADAPTIVE_FPS = _env("PASSTHROUGH_LIVE_ADAPTIVE_FPS", "1") == "1"

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

# PT_PASSTHROUGH_MAX_CONCURRENT:
#   Maximum concurrent passthrough streams. Keep 1 for consumer GPUs and shared
#   Matter state unless explicit multi-session testing proves safe.
PASSTHROUGH_MAX_CONCURRENT = max(1, int(_env("PASSTHROUGH_MAX_CONCURRENT", 1)))

# PT_PASSTHROUGH_BUSY_WAIT_SEC:
#   How long a new passthrough request waits for the active slot before 503.
#   Same-owner range/probe requests may preempt the prior stream.
PASSTHROUGH_BUSY_WAIT_SEC = max(0.0, float(_env("PASSTHROUGH_BUSY_WAIT_SEC", 10)))

# PT_PASSTHROUGH_PAD_TO_LENGTH:
#   Legacy pseudo-VOD compatibility option. Padding generated responses to an
#   estimated length is risky and mostly kept for controlled client experiments.
PASSTHROUGH_PAD_TO_LENGTH = _env("PASSTHROUGH_PAD_TO_LENGTH", "1") == "1"

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
