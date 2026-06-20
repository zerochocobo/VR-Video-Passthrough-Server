# VR Video Passthrough Server - Project Guide

VR Video Passthrough Server is a local DLNA/UPnP media server for VR video playback.
It exposes a local video library to DLNA clients and can generate a realtime
passthrough stream by applying RVM video matting, compositing the foreground over
a green background, and encoding the result for playback.

The current production-oriented path is PyNvVideoCodec + GPU matting + HEVC
output. The older FFmpeg subprocess pipeline remains available as a fallback and
for diagnostics.

## Quick Start

Recommended Windows startup profile:

```bat
run_server.bat
```

Direct development start:

```bat
uv run python main.py
```

Useful environment variables are prefixed with `PT_`. The current production
defaults live in `config.py`; use `run_server.bat` only as a convenience
launcher unless temporarily overriding values for diagnostics.

## Current Product Behavior

- DLNA discovery uses SSDP on UDP/1900.
- HTTP media/control defaults to port `8200`.
- The advertised server name defaults to `VR Video Passthrough Server` and can be
  changed with `PT_SERVER_NAME`.
- The video directory defaults to `G:\Downloads` and can be changed with
  `PT_VIDEO_DIR`.
- Physical folders under `VIDEO_DIR` are exposed as DLNA folders.
- Each normal video is listed as:
  - the original media item;
  - a `*-passthrough-live` item.
- `PT_PASSTHROUGH_OUTPUT_MODE` controls generated live entries:
  - `green`: existing green-background chroma-key live entry;
  - `alpha`: experimental DeoVR alpha-packed fisheye live entry titled
    `Alpha Passthrough`;
  - `all`: expose both green and alpha live entries.
- The older pseudo-VOD `/passthrough/{name}` endpoint still exists in HTTP code
  but is hidden from the DLNA catalog for now.
- `*-passthrough-live` is directly playable for short videos.
- For longer videos, `*-passthrough-live` is a chapter container. Entering it
  shows coarse start points backed by `/passthrough_live/{name}?t=<seconds>`.
- Live chapter rules:
  - max items: `PT_PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS`, default `10`;
  - minimum interval: `PT_PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC`, default
    `180` seconds;
  - first item is always `Play from start` at `0` seconds;
  - titles use `HH:MM`, not seconds;
  - computed intervals round up to whole minutes;
  - candidate starts with `<= 60` seconds remaining are omitted.
- Files whose name contains `passthrough` are treated as already-derived media
  and do not get additional passthrough/live virtual entries.
- Thumbnails are cached and live/passthrough entries reuse the raw thumbnail to
  avoid loading matting during browsing.

## Top-Level Layout

```text
VR Video Passthrough Server/
|-- main.py                    Process entry point and startup sequence.
|-- config.py                  Global config with PT_* environment overrides.
|-- run_server.bat             Recommended Windows startup profile.
|-- pyproject.toml             Python dependencies and project metadata.
|-- PROJECT.md                 This document.
|-- dlna/                      UPnP/DLNA discovery and ContentDirectory.
|-- http_app/                  FastAPI app and HTTP routes.
|-- pipeline/                  Decode, matting, encode, thumbnail pipelines.
|-- ui/                        PySide6 desktop UI, pages, i18n, and process control.
|-- offline/                   Production offline conversion entry points.
|-- utils/                     Shared helpers for cache, logs, metadata, etc.
|-- tools/                     Benchmarks, probes, scans, warmup utilities.
|-- models/                    ONNX matting models.
|-- videos/                    Default local input video directory.
|-- runtime_cache/             Runtime caches, thumbnails, and warmup marker.
|-- debug_output/              Runtime logs and diagnostic outputs.
|-- baseline/                  Performance baseline records and scan outputs.
|-- prompt/                    Handover notes and investigation reports.
```

## Application Entry Points

### `main.py`

Startup order:

1. Configure runtime cache directories before CUDA-heavy imports.
2. Start the optional local startup-status endpoint.
3. Run startup GPU warmup when enabled.
4. Log the active pipeline configuration.
5. Ensure Windows Firewall rules when possible.
6. Start the SSDP background thread.
7. Create the FastAPI app and run uvicorn.
8. Stop SSDP/status helpers during shutdown.

Shutdown behavior is tuned for Ctrl+C: uvicorn graceful shutdown timeout is kept
short and KeyboardInterrupt tracebacks are suppressed where possible.

### `config.py`

Central configuration. Most values are overridable with `PT_*` environment
variables through `_env()` / `_env_any()`.

Important groups:

- Network and DLNA identity: `LAN_IP`, `HTTP_PORT`, `SERVER_NAME`, UUID/USN.
- Media library: `VIDEO_DIR`, `VIDEO_EXTS`, `PASSTHROUGH_SUFFIX`.
- Matting: model path, input size, warmup settings.
- Passthrough encode: container, codec, bitrate, GOP, FPS cap, live chapters.
- FFmpeg decode: hardware acceleration, output pixel format, optional downscale.
- Thumbnail timeout and runtime cache settings.

### `run_server.bat`

Windows convenience launcher for real-device testing. Runtime defaults are kept
in `config.py`; add temporary `set PT_*` lines here only for diagnostics or
one-off client compatibility tests. The script exits with Python's error code.

## DLNA Package

### `dlna/ssdp.py`

Minimal SSDP responder/notifier:

- listens on `239.255.255.250:1900`;
- replies to `M-SEARCH`;
- sends periodic `ssdp:alive` notifications;
- sends `ssdp:byebye` on shutdown.

### `dlna/descriptions.py`

Builds static UPnP XML documents:

- `/description.xml` device description;
- `/cds.xml` ContentDirectory SCPD;
- `/cm.xml` ConnectionManager SCPD.

### `dlna/content_directory.py`

Implements ContentDirectory SOAP Browse logic:

- maps physical subdirectories to DIDL containers (`d:<relative/path>`);
- emits original video items;
- emits passthrough-live direct items or chapter containers;
- generates chapter items with `?t=<seconds>` start offsets;
- estimates sizes/bitrates for virtual resources;
- skips derived links for filenames containing `passthrough`.

### `dlna/profiles.py`

Small helpers for DLNA protocolInfo details:

- passthrough DLNA profile name;
- advertised frame rate when FPS cap is set.

## HTTP App Package

### `http_app/server.py`

FastAPI app factory. It mounts DLNA routes and media routes.

### `http_app/routes_dlna.py`

UPnP HTTP endpoints:

- `GET /description.xml`;
- `GET /cds.xml`;
- `GET /cm.xml`;
- `POST /control/cds`;
- `POST /control/cm`.

### `http_app/routes_media.py`

Media and passthrough routes:

- `GET /media/{name}`: raw source file with HTTP Range support.
- `GET /thumb/{name}`: cached JPEG thumbnail.
- `GET /passthrough_live/{name}`: live passthrough stream, normally MPEG-TS.
- `HEAD/GET /passthrough/{name}`: pseudo-VOD byte/time-seek experiment; still
  implemented but hidden from DLNA listings.

This module also owns passthrough concurrency control. The default server allows
one active passthrough stream at a time to avoid NVENC/NVDEC and shared-matting
resource contention.

## Pipeline Package

### `pipeline/pynv_stream.py`

Production PyNv passthrough stream:

```text
PyNv decode -> GPU matting/composite -> PyNv HEVC encode -> FFmpeg mux
```

Responsibilities:

- PyNv preflight and preflight cache;
- PyNv decoder/encoder lifecycle;
- GPU stream synchronization before NVENC reads composited NV12;
- FFmpeg muxing to fMP4 or MPEG-TS;
- stderr draining;
- reader/worker thread coordination;
- per-stream FPS cap for adaptive live playback;
- throughput and stage timing diagnostics.

### `pipeline/pynv_io.py`

Thin PyNvVideoCodec adapters:

- exposes decoded GPU NV12 planes through CUDA Array Interface;
- wraps composited contiguous NV12 CuPy buffers as PyNv encoder AppFrame input.

### `pipeline/matting.py`

ONNX Runtime matting and compositing:

- supports the RVM ONNX matting model used by the realtime path;
- handles SBS splitting and RVM recurrent state;
- supports CPU/BGR and GPU/NV12 paths;
- includes CuPy RawKernel preprocess and composite kernels;
- exposes the shared `Matter` instance through `get_matter()`.

### `pipeline/ffmpeg_io.py`

Legacy FFmpeg subprocess wrappers:

- `probe()` / `probe_cached()` through ffprobe;
- `DecoderProcess` for source seek/decode to raw BGR24/NV12;
- `EncoderProcess` for raw frame input to fMP4/MPEG-TS output.

### `pipeline/stream.py`

Legacy passthrough stream using FFmpeg decode + Matter + FFmpeg encode.
It remains useful as fallback and for comparing with the PyNv path.

### `pipeline/thumbnail.py`

Thumbnail generation and cache:

- extracts one frame near 10% of duration, capped at 30 seconds;
- writes JPEG cache files under `runtime_cache/thumbs/`;
- uses stat-based fingerprints to invalidate stale thumbnails;
- reuses raw thumbnails for live/passthrough catalog items;
- has a configurable ffmpeg timeout to avoid browse/shutdown stalls.

## Utils Package

### `utils/cache_key.py`

Stat-sensitive file identity helpers:

- short SHA1 fingerprint from path, size, and mtime;
- tuple `stat_key()` for in-memory caches.

### `utils/bitrate_estimator.py`

Persistent bitrate/size estimator for virtual passthrough resources:

- default estimates from configured bitrates;
- EWMA cache of actual emitted bitrate;
- stat/config keyed to avoid stale hits.

### `utils/video_metadata.py`

ffprobe metadata and backend routing policy:

- color metadata;
- timing/CFR metadata;
- codec/pixel-format/bit-depth metadata;
- PyNv-vs-FFmpeg decision rules;
- CFR source-index mapping for capped output FPS.

### `utils/gpu_runtime_cache.py`

CUDA/CuPy/ORT runtime cache setup and startup warmup:

- pins cache directories under `runtime_cache/`;
- builds a model/GPU/provider warmup key;
- writes `gpu_warmup_marker.json`;
- uses a warmup lock to avoid concurrent cache builds.

### `utils/startup_status.py`

Tiny localhost status server used during startup warmup so tests can see the
process is alive before the main DLNA HTTP port is listening.

### `utils/firewall.py`

Windows Firewall helper:

- ensures inbound TCP for HTTP port;
- ensures inbound UDP/1900 for SSDP;
- uses direct netsh when elevated or UAC `runas` batch otherwise.

### `utils/logger.py`

Shared logging setup for stdout and `debug_output/server.log`.

## Tools Package

### `tools/bench.py`

Local diagnostics and benchmarks:

- `play`: open a raw or passthrough URL with ffplay;
- `bench`: pull HTTP output through ffmpeg and report stats;
- `pipeline`: decode + matting/composite benchmark;
- `transcode`: decode + matting + encode benchmark;
- `decode`: decode-only benchmark;
- `decode-matrix`: split FFmpeg decode/download/pipe costs;
- `matting`: repeat one-frame matting path tests.

### `tools/scan_videos.py`

Offline library scanner:

- probes metadata for many files;
- classifies backend decisions;
- optional strict PyNv first-frame decode test;
- writes CSV, JSON summary, and baseline text.

### `tools/warmup_gpu_cache.py`

CLI for GPU runtime cache inspection/build:

- print warmup key and marker status;
- force warmup;
- check-only mode for startup/install flows.

### PyNv probe tools

Development probes used while building the PyNv path:

- `pynv_decode_probe.py`;
- `pynv_encode_probe.py`;
- `pynv_mux_probe.py`;
- `pynv_transcode_probe.py`;
- `pynv_matting_probe.py`;
- `pynv_fullchain_probe.py`.

### Other diagnostics

- `gpu_video_probe.py`: inspect local GPU/video stack.
- `ort_cold_probe.py`: isolate ONNX Runtime CUDA cold-start behavior.

## Runtime Data Directories

These directories are generated or environment-specific:

- `runtime_cache/thumbs/`: JPEG thumbnails.
- `runtime_cache/`: CUDA ComputeCache, CuPy cache, ORT cache, warmup marker.
- `debug_output/`: server log and local diagnostic outputs.
- `baseline/`: manually kept performance baselines and scan outputs.
- `.venv/`, `.uv-cache/`: uv/Python environment data.

## Key Configuration Reference

| Variable | Default / Current Intent | Purpose |
|---|---:|---|
| `PT_SERVER_NAME` | `VR Video Passthrough Server` | DLNA/SSDP friendly name. |
| `PT_LAN_IP` | auto-detect | LAN address advertised to DLNA clients. |
| `PT_HTTP_PORT` | `8200` | HTTP media/control port. |
| `PT_VIDEO_DIR` | `G:\Downloads` | Media library root. |
| `PT_MODEL_PATH` | RVM MobileNetV3 fp16 | ONNX matting model. |
| `PT_MATTING_INPUT_SIZE` | RVM `1024`, other models `512` | Matting reference input size. |
| `PT_MATTING_SPLIT_SBS` | `1` | Split side-by-side VR frames before matting. |
| `PT_ALPHA_STRIDE` | `3` | Run matting once every N output frames, reuse alpha otherwise. |
| `PT_MATANYONE2_IOBINDING` | `1` | Enable MatAnyone2 offline `step_update` IOBinding for batch-1 hot path, with automatic fallback. |
| `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE` | `0` | Experimental MatAnyone2 offline guided alpha refinement using the uploaded NV12 Y plane; default off because it can create background halos on some subjects. |
| `PT_MATANYONE2_GUIDED_FULLRES_SCALE` | `0.5` | Guided alpha refinement output scale; default half-res keeps 8K cost bounded before final composite upsample. |
| `PT_MATANYONE2_GUIDED_SUPPORT_FLOOR` | `0.02` | Suppress guided refinement where the original alpha has almost no support, preventing background halos. |
| `PT_MATANYONE2_GUIDED_MAX_DELTA` | `0.08` | Limit guided alpha growth over the original bilinear alpha; set negative to disable. |
| `PT_MATANYONE2_GUIDED_BAND_LO` | `0.05` | Lower base-alpha confidence threshold; guided refinement is bypassed below this value. |
| `PT_MATANYONE2_GUIDED_BAND_HI` | `0.95` | Upper base-alpha confidence threshold; guided refinement is bypassed above this value. |
| `PT_MATANYONE2_ROI_CROP` | `0` | Experimental MatAnyone2 offline ROI crop/letterbox quality mode; default off and not a token/FLOP speedup with the fixed 1024 model. |
| `PT_MATANYONE2_ROI_EXPAND` | `0.30` | Expand the bootstrap-mask ROI by this bbox fraction before crop/letterbox preprocessing. |
| `PT_MATANYONE2_ROI_MAX_EYE_FRACTION` | `0.70` | Disable ROI and fall back to full-eye processing when the expanded ROI covers too much of an eye. |
| `PT_MATANYONE2_ROI_FEATHER` | `16` | Feather ROI alpha edges in full-eye pixels when pasting the ROI result back into the SBS alpha. |
| `PT_MATANYONE2_SEGMENT_FRAMES` | `240` | Re-bootstrap MatAnyone2 from prepass masks after this many frames; Phase-1 state gating lets the default move beyond the earlier 60-frame workaround, `0` disables the fixed interval. |
| `PT_MATANYONE2_LAST_MASK_UNCERT_GATE` | `0.7` | Scale down MatAnyone2 `last_mask` in high-uncertainty regions before propagation, reducing old-position drag while keeping stable areas anchored. |
| `PT_MATANYONE2_SENSORY_DECAY_INTERVAL` | `8` | Apply a mild recurrent sensory-state decay every N output frames; `0` disables this soft reset. |
| `PT_MATANYONE2_SENSORY_DECAY_FACTOR` | `0.9` | Multiplier used by MatAnyone2 sensory-state decay. Lower values reduce residual state more aggressively but can flicker. |
| `PT_MATANYONE2_LAST_PRED_BINARIZE` | `1` | Feed a thresholded previous mask into MatAnyone2 `last_pred_mask` while keeping `last_mask` soft. |
| `PT_MATANYONE2_LAST_PRED_BIN_THRESHOLD` | `0.5` | Threshold used when `PT_MATANYONE2_LAST_PRED_BINARIZE=1`. |
| `PT_MATANYONE2_BOOTSTRAP_REFINE_ITERS` | `3` | Recurrent first-frame refinement passes used to build stronger segment memory; `1` restores the previous behavior. |
| `PT_MATANYONE2_SCENE_RESET` | `1` | Merge detected scene cuts into MatAnyone2 offline segment planning when a bootstrap mask is available. |
| `PT_MATANYONE2_ALPHA_SMOOTH` | `0` | Optional per-eye EMA smoothing for MatAnyone2 offline alpha; default off because it can add motion afterimages. |
| `PT_USE_PYNV` | `1` | Enable PyNv backend for eligible sources. |
| `PT_PASSTHROUGH_MAX_FPS` | `30` | Output FPS cap. |
| `PT_PASSTHROUGH_OUTPUT_MODE` | `green` | Generated passthrough layout: `none`, `green`, `alpha`, or `all` to expose both passthrough entries. |
| `PT_ALPHA_PASSTHROUGH_TITLE` | `Alpha Passthrough` | DLNA virtual item title when alpha output mode is enabled. |
| `PT_COMPOSITE_BG_RGB` | `808080` | Green-screen/composite background color as RGB hex. UI presets: `808080`, `C8C8C8`, `2BE640`, `0047BB`. |
| `PT_PASSTHROUGH_SEND_REALTIME_PACING` | `1` | Pace live HTTP delivery to player-safe bitrate. |
| `PT_PASSTHROUGH_SEND_PACING_MULTIPLIER` | `2.0` | Multiplier applied to estimated live send bitrate so MPEG-TS/VBV bursts do not starve player buffering. |
| `PT_PASSTHROUGH_SEND_MIN_BPS` | `100000000` | Minimum paced live HTTP send bitrate. |
| `PT_PASSTHROUGH_PRODUCER_REALTIME_PACING` | `0` | Optional PyNv worker wall-clock pacing; default off to preserve producer FPS headroom. |
| `PT_PASSTHROUGH_REALTIME_PACING` | `1` | Compatibility alias for send pacing. |
| `PT_PASSTHROUGH_HEVC_BITRATE` | `50M` | PyNv HEVC output target bitrate. |
| `PT_PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_RATE` | `48000` | Live MPEG-TS AAC output sample rate. |
| `PT_PASSTHROUGH_AUDIO_MPEGTS_OUTPUT_CHANNELS` | `2` | Live MPEG-TS AAC output channels, stereo. |
| `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE` | `1` | Use green-screen slate while first AAC cache is built. Slate video continues while the first real frame is prepared; source audio starts only after the first real video bitstream is written. |
| `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE_DIRECT_AFTER` | `1.0` | On cache miss, stop waiting for full AAC cache after this many seconds and feed this playback from direct source demux while the full cache continues. |
| `PT_PASSTHROUGH_GOP` | `60` | Encoder GOP size. |
| `PT_PASSTHROUGH_MAX_CONCURRENT` | `1` | Concurrent passthrough stream limit. |
| `PT_PASSTHROUGH_LIVE_CHAPTER_MIN_INTERVAL_SEC` | `180` | Minimum live chapter spacing and short-video threshold. |
| `PT_PASSTHROUGH_LIVE_CHAPTER_MAX_ITEMS` | `10` | Max live chapter entries. |
| `PT_THUMB_FFMPEG_TIMEOUT_SEC` | `3` | Thumbnail extraction timeout. |
| `PT_STARTUP_GPU_WARMUP` | `1` | Warm CUDA/ORT caches before serving. |
| `PT_STARTUP_STATUS_PORT` | `8299` | Local status port during warmup. |

## Typical Request Flow

```text
DLNA client
  |
  |-- SSDP M-SEARCH -----------------> dlna/ssdp.py
  |
  |-- GET /description.xml ----------> routes_dlna -> descriptions.py
  |-- POST /control/cds Browse ------> routes_dlna -> content_directory.py
  |                                      returns folders, raw items, live chapter containers/items
  |
  |-- GET /media/{name} -------------> routes_media -> FileResponse with Range
  |-- GET /thumb/{name} -------------> routes_media -> thumbnail.py -> runtime_cache/thumbs/
  |
  |-- GET /passthrough_live/{name}?t=seconds
        -> routes_media
        -> PyNvPassthroughStream when eligible
        -> PyNv decode
        -> Matter GPU matting/composite
        -> PyNv HEVC encode
        -> FFmpeg mux
        -> StreamingResponse
```

## Notes for Future Work

- The pseudo-VOD `/passthrough` path is intentionally still in the code but is
  hidden from DLNA because several clients probe/seek generated media in ways
  that do not match realtime generation.
- Live chapter containers are the current coarse seek strategy.
- Startup GPU warmup is important on systems where ORT CUDA first-run JIT is
  expensive.
- Direct development diagnostics that need ONNX Runtime CUDA must launch with
  the same DLL environment as the UI. Reuse
  `ui.services.process_helpers.base_environment()` or prepend equivalent
  `PT_CUDNN_BIN`, `CUDA_PATH\bin`, and `CUDA_HOME\bin` entries to `PATH` before
  running `uv run`/Python tools; otherwise ORT can fail to load CUDA provider
  DLLs such as `cublasLt64_12.dll` / `cudnn64_9.dll` and silently fall back to
  CPU.
- Debug tools that instantiate `Matter`/ORT directly must also call
  `utils.gpu_runtime_cache.configure_gpu_runtime_cache()` before importing
  CUDA-heavy modules, and should keep `tempfile.TemporaryDirectory` under
  `config.RUNTIME_TMP_DIR` like `tools/offline_alpha_passthrough.py`. After
  creating the ONNX session, verify that `CUDAExecutionProvider` is both
  available and active; if it is not, fail fast instead of continuing on CPU.
- Translation JSON files under `ui/translations/` are intentionally kept as
  UTF-8 with BOM. When editing or generating them programmatically, read/write
  with `utf-8-sig` and preserve the BOM.
- MatAnyone2 medium now defaults to `yolo26m_efficientsam`; `yolo26m_birefnet`
  remains available from the offline recognition selector for higher-VRAM
  systems, and `yoloworld_efficientsam` remains a command-line legacy fallback.
- Keep handover notes in `prompt/HANDOVER_YYYYMMDD.md` when making meaningful
  project changes.

## CuPy kernel compile "hangs" on sm_120 (NVRTC pitfall)

**Symptom:** the first launch / `get_function()` of any CuPy `RawModule` /
`RawKernel` -- even a one-line noop kernel -- takes 60-120s (looks hung), while
already-cached kernels run instantly. Hits the 2D->3D GPU renderer
(`offline/two_dvr_gpu.py`), matting CuPy kernels, etc.

**Root cause:** on the RTX 5060 Ti (sm_120 / Blackwell), CuPy must compile kernels
with **NVRTC >= 12.8** to emit sm_120 cubins directly. If CuPy instead uses the
**system CUDA 12.6 NVRTC** (system `CUDA_PATH` is `...\CUDA\v12.6`), that NVRTC
doesn't know sm_120, so CuPy falls back to emitting PTX and lets the **driver
JIT-compile PTX -> sm_120 at launch** -- a cold start that is extremely slow on the
13.x driver and looks like a hang. Forcing `CUPY_COMPILE_WITH_PTX=1` takes the same
slow PTX path on purpose. See
`summary/summary_20260615_CUPY_SM120_NVRTC_UPGRADE_CN.md`.

**Why it works in this repo's uv venv:** `pyproject.toml` pins
`cupy-cuda12x[ctk]` (+ `nvidia-cudnn-cu12`), which bundles **pip NVRTC 12.9**
(`nvrtc64_120_0.dll` / `nvrtc-builtins64_129.dll`). Running via `uv run python ...`
loads that 12.9 NVRTC even though system `CUDA_PATH` still points at 12.6, so a
fresh kernel compiles in <1s. The frozen build sets `CUPY_COMPILE_WITH_PTX=0` and
bundles the pip NVRTC (`packaging/runtime_hook_cuda_dlls.py`, `build_exe.py`).

**Run it the way that works:**
- Always `uv run python ...` / `uv run python -m offline.two_dvr ...` (the project
  `.venv`); run `uv sync` first so the pip CUDA 12.9 runtime + NVRTC are installed.
- Do NOT set `CUPY_COMPILE_WITH_PTX=1` (leave unset, or `0`).
- Do NOT prepend `%CUDA_PATH%\bin` (system 12.6) ahead of the venv on `PATH`, or it
  can shadow the pip NVRTC. (Note: ORT/TensorRT DLL setup via
  `apply_runtime_dll_paths()` is unrelated -- it does not change which NVRTC CuPy
  loads.)
- Folding a new kernel into an existing cached `RawModule` (e.g.
  `_SOFT_SHIFT_KERNELS`) and reusing existing `RawKernel`s avoids extra cold
  compiles.

**3-line self-check in the process that hangs:**
```python
import os, cupy as cp
from cupy.cuda import compiler
print(cp.cuda.nvrtc.getVersion(), os.environ.get("CUPY_COMPILE_WITH_PTX"),
      compiler._use_ptx, compiler._get_arch_for_options_for_nvrtc())
# want: (12, 9)  '0'/None  False  ('-arch=sm_120', 'cubin')
# (12, 6), '1', _use_ptx True, or ('...compute_120','ptx') => the slow PTX-JIT path.
```
A fresh unique-source kernel should compile+launch in <1s; if it's 60-120s you are
still on PTX -> driver JIT.

**Gotcha -- the env var is a red herring once cupy is imported.** CuPy captures
`CUPY_COMPILE_WITH_PTX` into `compiler._use_ptx` **at import time** (one read,
`compiler.py` module level). If a stale `CUPY_COMPILE_WITH_PTX=1` is in the shell
when `import cupy` first runs, `_use_ptx` is `True` for the whole process even if
you later set the env var back to `0` and print `0`. The decisive value is
`compiler._use_ptx`, NOT `os.environ`. Two consequences:
- `os.environ.setdefault("CUPY_COMPILE_WITH_PTX","0")` does NOT fix a stale `=1`
  (setdefault is a no-op when the key exists). `configure_gpu_runtime_cache()` and
  `offline/two_dvr_gpu.py` hard-set it to `"0"` for this reason.
- If you hit the hang, clear the shell var (`Remove-Item Env:CUPY_COMPILE_WITH_PTX`
  in PowerShell) or open a fresh shell, then re-run -- and confirm with
  `compiler._use_ptx is False`.

**Gotcha -- the default user CuPy cache can also make a correct NVRTC path look
hung.** On 2026-06-18, `_GPU_NEAR_KERNELS` was tested with the correct fast-path
diagnostics:

```text
nvrtc=(12, 9)
compiler._use_ptx=False
arch=('-arch=sm_120', 'cubin')
```

but `RawModule.get_function(...)` still appeared to hang when CuPy used the
default cache:

```text
C:\Users\dennis\.cupy\kernel_cache
```

Switching only `CUPY_CACHE_DIR` to a fresh project-local directory made the same
11-kernel RawModule compile in about 1.25s. The practical lesson is:

```text
Do not rely on the default user CuPy cache for project tools.
Always call utils.gpu_runtime_cache.configure_gpu_runtime_cache()
before the first import cupy / RawModule / RawKernel.
```

For standalone modules that may be imported outside `main.py`, set the runtime
cache at module import before any lazy CuPy import. `offline/two_dvr_gpu.py` does
this because offline 2DVR tools can bypass the normal server startup path.

If a kernel still looks hung despite the NVRTC self-check being correct, run one
more self-check with an explicit fresh cache:

```powershell
$env:CUPY_CACHE_DIR="G:\GIT\debug\PTMediaServer\runtime_cache\cupy_jit_probe"
uv run python your_probe.py
```

If the fresh cache is fast, the cause is stale/locked/corrupt default CuPy cache
state, not CUDA source code and not the NVRTC/PTX path.

**"It hangs for me but not for you" -- developer triage checklist.** This is almost
always an environment delta, NOT a kernel bug and NOT directory permissions. A
brand-new multi-kernel `RawModule` (e.g. `_GPU_NEAR_KERNELS` in
`offline/two_dvr_gpu.py`) is one compile and is fine on the fast path. Run the
3-line self-check **in the exact launcher that hangs** (same interpreter + same
shell/IDE run config -- a clean terminal won't reproduce the env delta), then walk
these in order:

1. **Wrong interpreter (most common).** `print(sys.executable)` must be
   `...\PTMediaServer\.venv\Scripts\python.exe`. Running via an IDE (PyCharm/VSCode)
   with a different interpreter, or bare `python x.py`, loads a CuPy whose only
   NVRTC is the **system CUDA 12.6** one -> PTX -> driver JIT -> 60-120s "hang".
   Fix: launch with `uv run python ...` or activate the project `.venv`.
2. **`uv sync` never ran in their checkout.** The pip NVRTC 12.9 must exist at
   `.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll` (~89 MB). Run
   `uv sync` first.
3. **DLL shadowing on PATH.** System CUDA 12.6 `bin` ahead of the venv on PATH wins
   the `nvrtc64_120_0.dll` name (same filename in both!) even with the right venv.
   `cp.cuda.nvrtc.getVersion()` returning `(12, 6)` is the tell.
4. **Stale `CUPY_COMPILE_WITH_PTX=1`** captured at import (see gotcha above) --
   `Remove-Item Env:CUPY_COMPILE_WITH_PTX`, restart the process.

**Is it a slow compile or a real deadlock?** Don't kill it -- let it run 2-3 min
once. If it finishes AND a second launch of the same kernel is instant, it was
PTX->driver JIT (env problem above), not a deadlock; `nvidia-smi` shows the python
process busy meanwhile.

**Permissions are a red herring for the hang.** A non-writable `CUPY_CACHE_DIR` only
defeats *caching* (you recompile every run); on the fast NVRTC 12.9 path that's
still <1s/run. It cannot turn a cubin compile into the 60-120s PTX-JIT path -- fix
the NVRTC path first.

**One-shot fix:**
```powershell
uv sync
Remove-Item Env:CUPY_COMPILE_WITH_PTX -EA SilentlyContinue
uv run python <script>   # self-check should print (12, 9) ... False ('-arch=sm_120', 'cubin')
```
