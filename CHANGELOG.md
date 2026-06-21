# Changelog

This file only keeps version releases, major bug fixes, major UI/UX updates, and major core or performance upgrades.

## English

### 2026-06-21

- **SI realtime MPEG-TS transport:** Added the `/si_live/{name}` streaming route for same-stem `.si.wav` sidecars, switching SI playback from full-file progressive virtual MP4 preparation to zero-cache realtime MPEG-TS with `?t=` start offsets.
- **DLNA `[SI]` time index:** `[SI]` entries are now directories with a `[Select Time Index]` folder, quick chapter leaves, 10-minute groups, minute folders, and 5-second playable start points that all target `/si_live/*.ts`.
- **SI cache path cleanup:** Kept the older `/media_si` progressive virtual MP4 route as a dormant fallback, while removing DLNA-side SI prewarm wiring and dead progressive-cache item helpers from the active browse path.

### 2026-06-20

- **2D-to-3D / VR model update:** Added NVDS ONNX temporal stabilization for offline 16:9 jobs, split backbone/head runtime support, a 512x288 default NVDS tier, the DA3 Large 1036 preset, and a pre-run model download dialog for missing DA3/NVDS files.
- **DLNA Live time index:** Passthrough Live entries are now directories with localized `[Select Time Index]` folders, 10-minute groups, minute folders, and 5-second playable start points. Green and alpha Live directories show `[GREEN]` and `[ALPHA]` prefixes.
- **Default and wording cleanup:** Experimental `passthrough_seek` DLNA entries are hidden by default, and the 2DVR skip-existing checkbox now refers to the target file instead of a passthrough file.

### 2026-06-19

- **2DVR stabilizer iteration:** Added a GPU near-map stabilizer experiment, then defaulted the destructive stabilizer behavior off after real-device artifact validation while keeping diagnostic controls available.
- **Offline 2DVR UI update:** Replaced the old processing-precision selector with a quality-speed control shared by single and batch 2DVR jobs.
- **Offline 2DVR playback fix:** Investigated generated-output stutter and fixed the current PyNv HEVC path by repeating VPS/SPS/PPS parameter sets (`repeatspspps=1`) without changing the GOP-length defaults.

### 2026-06-18

- **SI progressive transport default-on:** Added SI-specific progressive virtual MP4 switches, enabled `/media_si` and DLNA `[SI]` entries by default for same-stem `.si.wav` sidecars, and separated this path from the legacy seekable passthrough experiment.
- **SI first-play and sync improvements:** Added background prewarm with bounded queues, sample-table/layout caching, source-audio sidecar decoupling, AAC mix acceleration, audio edit-list controls, better size estimates, and UI wording/help updates.
- **2DVR temporal stability update:** Added renderer-side depth normalization and near-map temporal stabilization, scene-cut resets, offline progress logging, and a GPU near-preprocessing baseline for later optimization.

### 2026-06-17

- **SI feature groundwork:** Added same-stem `.si.wav` detection, `[SI]` DLNA entries, local SI mix control APIs, Home-page SI settings, and the first `/media_si` implementation before pivoting the transport layer to progressive virtual MP4.
- **Progressive MP4 research path:** Added PyAV prototype tooling, player Phase 0 sample generation, PyInstaller/PyAV compatibility probes, and the first `pipeline.si_virtual_mp4` sample-table and virtual-range primitives.
- **2DVR artifact cleanup:** Added GPU cleanup for vertical stripe/rim contamination, hardened CuPy sm_120 direct-cubin behavior, and fixed the DA3 startup heartbeat call so realtime DA3 prewarm can run.

### 2026-06-16

- **Realtime 2D-to-3D UI:** Added realtime 2D-to-3D settings, server environment propagation, a user-facing 3D strength control, and DA3 engine prewarm to reduce first-play startup latency.
- **2DVR soft-shift quality fixes:** Fixed per-eye hole-fill direction, depth-edge smearing, hard-silhouette/background-seam handling, foreground dilation, hybrid inverse-warp hole fill, and background-side rim cleanup. The UI now treats soft-shift as the product path and hides `inverse_warp`.
- **DA3 model preset update:** Added normal/HD DA3 presets, per-preset TensorRT caches, Hugging Face / mirror-aware model downloads, and updated DA3 exporter documentation.

### 2026-06-15

- **Offline 2D-to-3D / VR generation:** Added DA3 ONNX-based offline 2D-to-SBS 3D/VR conversion with CLI and PySide UI support for single, batch, start/duration, and segment workflows.
- **GPU-resident 2DVR pipeline:** Added the PyNv/CuPy GPU path for decode, DA3 depth, stereo rendering, NVENC output, and VR projections, with TensorRT as the default DA3 acceleration path.
- **Realtime 2D-to-3D groundwork:** Added the `two_dvr` passthrough mode, DLNA `[2D>3D]` live entries, `/passthrough_live mode=two_dvr`, offline output detection/naming, and DA3 TensorRT startup prewarm.
- **CUDA runtime hardening:** Moved toward self-contained pip CUDA runtime components for CuPy/ONNX Runtime and sm_120-class GPUs, reducing dependence on a system CUDA toolkit.

### 2026-06-01

- **v0.1.0 released.**
- **Skybox live playback fix:** DLNA passthrough-live URLs now append a `.ts` route-hint suffix so Skybox selects its MPEG-TS pipeline instead of rejecting TS bytes behind an `.mp4` URL. The live route strips `.ts`, `.m2ts`, and `.mpegts` before source lookup, keeping older cached URLs and other clients compatible.
- **Skybox/libmpv probe guard:** Bare `libmpv` screenshot probes now return `503` before acquiring a slot, starting GPU work, or draining an active session. Same-key libmpv startup requests are debounced, and same-owner preemption no longer closes already-built realtime streams or `LiveSession` objects.

### 2026-05-31

- **Offline single-video UI update:** Renamed the single-video tab away from test wording, added a mode selector for start/duration versus time-slice merge, and added persisted `HH:MM:SS` slice configuration with validation.
- **Offline segment merge:** `offline.convert single --segment START-END` can now process multiple configured time slices through the existing offline pipeline and concatenate the generated MP4 parts into one final output with FFmpeg concat copy.

### 2026-05-30

- **Startup crash fix:** Media directory parsing now skips unresolvable Windows cloud/rclone roots such as PikPak mounts instead of crashing UI startup or backend config import. Other valid local media folders continue to load.

### 2026-05-29

- **Offline single-video time controls:** Added a custom end-time option with video-duration probing and validation, then converted it to the existing `--start` and `--duration` offline command contract. English, Chinese, and Japanese translations were added.
- **Offline output naming:** Positive-duration single-video outputs now include an end-time tag such as `S000500_E000615_75S`, while generated-output detection still accepts legacy start-duration names.
- **DLNA image browse test switch:** Added a disabled-by-default `PT_DLNA_IMAGE_ENABLED` / UI setting for optional DLNA image browsing tests, including image MIME/protocol advertisement, DIDL image items, and `/media` serving for supported image formats.

### 2026-05-28

- **Realtime concurrency update:** Replaced the process-wide Matter singleton with a lazy Matter pool, added adaptive `PT_PASSTHROUGH_MAX_CONCURRENT=auto` based on NVIDIA VRAM, and wired live and non-live passthrough routes to acquire and release per-stream Matter instances without recurrent-state crosstalk or deadlock.
- **Seekable passthrough Stage 1:** Added the opt-in `/passthrough_seek/{name}` route, byte-range-to-time mapping, virtual MPEG-TS declared-size handling, UA/profile guards, and additive DLNA seek items behind explicit enable/DLNA switches while leaving existing live routes untouched.
- **DLNA/player compatibility diagnostics:** Added request-history middleware, redacted JSONL dumps, trace IDs, localhost debug endpoints, and shadow profile/intent/decision classification for DLNA, media, and passthrough requests without changing playback behavior.

### 2026-05-27

- **v0.1.0-beta.6 released.**
- **MatAnyone2/recognition update:** Replaced the MatAnyone2 Medium default bootstrap chain with `YOLO26m -> EfficientSAM -> MatAnyone2`, moved the active EfficientSAM model path to `models/efficientsam/`, and removed the legacy YOLOWorld-EfficientSAM option from the offline UI.
- **YOLO26m quality fix:** Defaulted YOLO26m to the FP32 ONNX model after confirming the FP16 export collapses on ORT CUDA, added a startup sanity warning for broken FP16 usage, improved stereo asymmetry handling, added scene-aware/boundary gap filling, and enabled unlimited multi-person pairing with `top_k=0`.
- **MatAnyone2 quality fix:** Reduced motion drag and afterimages with anti-drag defaults, uncertainty gating, sensory-state decay, last-mask binarization, bootstrap refinement loops, and a segment default of 240 frames after validation. Also fixed the SBS left-eye buffer overwrite that made the left eye drag much more than the right.
- **TensorRT update:** TensorRT cache builds now default to FP32 for realtime RVM, offline RVM, and offline MatAnyone2. MatAnyone2 TensorRT now builds only batch-1 caches for `512` and `1024`, with separate per-model cache directories under `runtime_cache/trt_engines/matanyone2/`.
- **UI update:** Added a 50 FPS realtime output option between 40 and 60, exposed MatAnyone2 offline precision as `512` and `1024` with `1024` still selected by default, and kept the offline recognition choices to YOLO26m-EfficientSAM and SAM3.
- **Offline/DLNA cleanup:** DLNA now marks generated passthrough outputs with an `[Offline]` title prefix, and offline passthrough cleans temporary `.aac` audio sidecars on success, failure, and early prepass exits.
- **Documentation update:** Updated the EfficientSAM, YOLO26m, and MatAnyone2 model setup notes for the current YOLO26m-EfficientSAM-MatAnyone2 pipeline, FP32 YOLO26m default, and MatAnyone2 `512`/`1024` TensorRT requirements.
- **TensorRT UI fix:** TensorRT cache build progress now counts MatAnyone2 `512` and `1024` as two model-build units and caps realtime RVM, offline RVM, and MatAnyone2 progress at 99% until the child process exits successfully, avoiding premature 100% completion during finalization.

### 2026-05-26

- **MatAnyone2 offline core update:** Extracted a shared MatAnyone2 offline engine for green and alpha outputs, added the batch-1 `step_update` IOBinding hot path, and kept the fallback NumPy path for unsupported or failed IOBinding runs.
- **MatAnyone2 stability update:** Added scene-reset planning, MatAnyone2-specific alpha smoothing controls, experimental guided alpha upsample with confidence-band gating, and an experimental ROI quality mode while keeping the guided and ROI paths off by default after validation.
- **MatAnyone2 TensorRT fix:** Isolated MatAnyone2 TensorRT builds by model-specific cache subdirectories, split multi-precision builds into separate subprocesses to avoid process-global ORT/TensorRT reuse, and synchronized runtime provider cache paths for green and alpha offline runs.
- **Offline process fix:** Stopping an offline conversion now terminates the whole child process tree on Windows, preventing MatAnyone2 prepass, conversion, or FFmpeg helpers from leaving GPU memory allocated after the UI reports the job as stopped.
- **Offline RVM TensorRT update:** Completed the realtime/offline RVM TensorRT cache split, offline precision-tier readiness checks, watcher freeze fixes, offline engine output directory fix, idempotent offline runtime path handling, and 1024-engine reuse between realtime and offline caches.

### 2026-05-25

- **UI/color fix:** Recalibrated Light Matching presets to the D65 display white point. `daylight` is now neutral 6500K, `night_cool` is now a visibly cool 8000K preset, and existing built-in preset settings are migrated so the default no longer appears yellow.
- **Startup UX update:** Added cold-start startup heartbeat/status APIs, overlay progress details, localized reassurance text, diagnostics logging, proxy-safe local polling, and monotonic warmup progress so long CUDA/ONNX/TensorRT startup is visible instead of appearing stuck.
- **Offline RVM update:** Added offline RVM processing precision tiers, removed the offline ResNet50 balanced path, added offline-only scene reset and alpha smoothing, and introduced offline RVM TensorRT warmup/cache completeness checks.
- **TensorRT cache update:** Split realtime and offline RVM TensorRT cache handling so the Home page builds only realtime 1024 engines while offline mode owns the broader precision-tier cache, including safer status checks and clearer long-build messaging.

### 2026-05-24

- **v0.1.0-beta.5 released.**
- **Bug fix:** Fixed a one-GOP live A/V sync offset caused by `+nobuffer` in the raw HEVC-to-MPEG-TS mux path. The default now disables `PT_MUX_NOBUFFER_ENABLE`, avoiding dropped first-GOP video while keeping the diagnostic override available.
- **UI/packaging update:** Added a TensorRT runtime library download flow for packaged Windows builds. The TensorRT dialog can detect missing runtime DLLs, offer automatic or manual NVIDIA wheel download, verify the wheel hash, extract only required DLLs, and show progress even when `Content-Length` is unavailable.
- **Core/UI update:** Added offline TensorRT cache support for MatAnyone2 `step_update`, separated RVM and MatAnyone2 TensorRT manifests/cache directories, added offline-only TensorRT switches for single and batch conversion, and removed the RVM balanced option from the offline UI.
- **UI/core update:** Updated light matching defaults to daylight, softened the warm preset, and refreshed Home-page TensorRT/FPS labels and help text. Realtime FPS defaults returned to 30 FPS, while the `Same as source` option remains available.
- **DLNA compatibility update:** Updated VR naming for 4XVR compatibility from `_LR_180` to `_LR_180_SBS`, preserved legacy generated-output detection, and added invisible ObjectID versioning so DLNA clients such as SKYBOX refresh cached virtual names.
- **Logging fix:** Added a targeted uvicorn socket-send noise filter that suppresses repeated `socket.send() raised exception` messages while keeping other uvicorn warnings and errors.
- **Packaging/UI fix:** Hid transient Windows console windows from packaged startup, server start/stop, offline conversion, TensorRT cache build, and runtime probe processes while preserving streamed logs.
- **Bug fix:** Hardened offline conversion child-process output forwarding so cold CUDA/ONNX/TensorRT startup no longer appears to stop silently, and conversion logs now show both child and UI-level exit status.
- **Compatibility fix:** Hardened ffprobe JSON handling on Windows by decoding JSON output as UTF-8 bytes instead of locale text, improving reliability for Chinese/Japanese paths and metadata.
- **TensorRT fix:** TensorRT runtime installation no longer overwrites already-loaded DLLs, and TensorRT build errors such as missing ONNX models now stay visible instead of flashing away.
- **Runtime fix:** Improved UI process lifecycle handling, safer forced server stop, startup polling stability, redacted passthrough owner logs, and added retry hints for busy/preempted passthrough requests.

### 2026-05-23

- **Performance/core update:** Added Track A first-chunk latency diagnostics and startup warmups, including mux timing marks, CuPy composite/alpha warmup, NVENC startup preflight, and validated low-latency mux defaults for the pipe-TS path.
- **Bug fix:** Resolved the nPlayer audio-only regression by removing `hevc_metadata=aud=insert` from the pipe-TS video-stage bitstream filter when `setts` timestamps are applied. Strict players now receive usable HEVC codec parameters and enter video playback mode.
- **Bug fix:** Fixed the FastAPI startup deprecation warning by moving runtime startup registration to a lifespan-based `create_app(startup_hook=...)` flow.
- **A/V sync update:** Reduced slate startup A/V skew by limiting immediate slate burst frames, pacing subsequent slate frames, and adding slate and MPEG-TS sync validation tools.
- **A/V sync investigation/update:** Added live MPEG-TS capture, audio-content alignment, and video-content alignment tools. Reworked the default live path to avoid slate/cache startup skew, use source audio directly when the AAC cache is disabled, and disable generated video slate by default for real-device A/B testing.
- **UI/runtime update:** Migrated Home-page realtime output FPS handling toward source-cadence testing, added source-FPS pacing, and documented that remaining subjective A/V issues may be client playback behavior when server-side captures are objectively aligned.

### 2026-05-22

- **Performance/core update:** Improved TensorRT cold-start warmup by warming the process-global `Matter` singleton, preloading static TensorRT sessions for warmup shapes, resetting recurrent state after warmup, and separating startup warmup control from global matting config mutation.
- **Performance/core update:** Added composite/alpha CuPy warmup and NVENC startup preflight so first-playback latency no longer pays common kernel JIT or encoder initialization costs.
- **Core update:** Enabled offline RVM fast mode to use ready TensorRT caches, while forcing CUDA/CPU providers for unsupported offline engines and adding clearer TensorRT provider diagnostics.
- **Bug fix:** Fixed offline TensorRT disable behavior so UI-launched child processes no longer inherit stale TensorRT provider settings, and failed static TensorRT activation no longer retries and floods logs every frame.
- **Performance bug fix:** Fixed offline alpha RVM throughput by adding the missing CUDA stream synchronization before NVENC encode. Offline alpha throughput on the 8K test path rose from about 36 FPS to about 75 FPS.
- **Bug fix:** Fixed misleading live headers for FPS-capped sources by advertising the actual effective stream FPS instead of the configured cap when the source FPS is lower.
- **Bug fix:** Reduced shutdown noise in live audio-cache cleanup by avoiding premature pipe closing during interrupted `communicate()` calls and logging interrupted slate cache builds as expected cleanup.

### 2026-05-21

- **DLNA/offline naming update:** Centralized VR/player filename handling in `utils.vr_naming` and reused it across DLNA titles, offline default output names, and generated-output detection.
- **Compatibility update:** Updated generated 2:1 half-equirectangular naming from `_SBS_180` to `_LR_180`, and alpha naming to `_LR_180_FISHEYE_F180_alpha` for broader player compatibility, including HereSphere.
- **DLNA update:** Bumped the DIDL schema version and refreshed live/raw naming rules so generated green and alpha entries use player-compatible markers while preserving original playback URLs.
- **Offline update:** Updated RVM and alpha offline output defaults to use the new VR naming rules while keeping legacy generated-output suffixes detectable.
- **Bug fix:** Cleaned up audio-cache interruption handling to avoid misleading `ValueError: I/O operation on closed file` reader-thread tracebacks during stream shutdown.

### 2026-05-20


- **v0.1.0-beta.4 patches and additions.**

- **Performance/UI update:** Added TensorRT acceleration cache management for realtime RVM inference, including a Performance-panel toggle, cache configure/build dialog, startup cache validation, CUDA fallback when the cache is missing or stale, and PyInstaller runtime DLL handling for TensorRT.
- **Bug fix:** Offline generation now targets the source video bitrate by default for all engines, including RVM fast/balanced and MatAnyone2 medium/slow. This keeps generated video size closer to the original source. If source bitrate is unavailable, offline generation falls back to 40 Mbps.
- **Bug fix:** Fixed lingering FFmpeg child processes after playback or server stop. PyNv streams now track and stop audio FFmpeg subprocesses, wait for slate audio/cache threads during close, clean up partially spawned pipe-TS muxers, remove stale temporary AAC files, and forced UI server stop now terminates child processes through `taskkill /T /F` on Windows.
- **Bug fix:** Fixed realtime 2D alpha blocks turning gray during playback by switching the default realtime RVM model back to FP32. The issue was caused by FP16 precision loss accumulating in RVM's recurrent `rec1`-`rec4` state across frames, not by alpha packing or realtime bitrate control.

### 2026-05-19

- **Bug fix:** Fixed MatAnyone2 medium offline alpha prepass crashes on HEVC Main10/P016-style decoded frames by converting 16-bit NV12/P010 planes to 8-bit BGR before YOLO-World/EfficientSAM or SAM3 prepass processing.
- **Bug fix:** Updated AV1 backend routing so GPUs without AV1 NVDEC support, such as RTX 20/Turing, use the FFmpeg decode fallback instead of failing later inside PyNv decode.
- **UI update:** Added foreground-only Light Matching for realtime passthrough, including a dedicated Home-page panel, presets, custom settings dialog, persisted UI settings, and live runtime updates during playback.
- **Core update:** Added DLNA `[NoLive]` labeling and realtime-source rejection for known unsupported live sources, avoiding confusing realtime fallback attempts.

### 2026-05-18

- **v0.1.0-beta.3 patches and additions.**
- **Core/performance upgrade:** Added MatAnyone2 medium offline mode using YOLO-World + EfficientSAM as the bootstrap recognizer before MatAnyone2 propagation.
- **Core/performance upgrade:** Reduced MatAnyone2 medium peak VRAM by moving the YOLO-World/EfficientSAM prepass into a subprocess and defaulting MatAnyone2 offline processing to batch 1 without SBS batching.
- **Core upgrade:** Improved SAM3-backed MatAnyone2 slow mode with shared SAM3 helper code, stereo mask consistency guarding, short inactive-gap filling, and configurable SAM3 text prompts.
- **UI update:** Updated the offline UI so MatAnyone2 is selected once, with a recognition-model selector for `YOLOWorld-EfficientSAM` or `SAM3 (16GB+ VRAM)`, plus a SAM3-only prompt dialog.
- **UI/core update:** Added flat 2D alpha output controls, including fisheye/flat3d projection mode, distance-based disparity, square-eye flat3d sizing, and Home-page 2D alpha settings.

### 2026-05-17

- **v0.1.0-beta.1 officially released for public beta testing.**
- **v0.1.0-beta.2 patches and additions.**
- **UI update:** Replaced the previous realtime/offline VRAM profile UI with Quality / Speed presets, including separate offline quality settings.
- **UI update:** Added `GET /runtime_status` and a centered main status-bar indicator for current FPS and VRAM usage.
- **Bug fix:** Added shared NVIDIA compute capability checks and hard gates for realtime startup and offline conversion.
- **Core update:** Added offline source-codec preflight plus an FFmpeg NV12 decode fallback for non-PyNv sources such as MPEG-4 Visual / `mp4v-20`.
- **Core update:** Added a dedicated 4XVR live playback profile so AVPro/ExoPlayer reconnects can reuse managed live sessions.
- **UI/core update:** Added flat 2D alpha passthrough for non-SBS 2D videos, projecting them into stereo fisheye SBS output with configurable FOV and disparity.

### 2026-05-16

- **Core/performance upgrade:** Added producer pacing for positive realtime FPS caps, so capped output also throttles production.
- **Core/performance upgrade:** Added realtime GPU resize after PyNv decode via `PT_DECODE_MAX_SIDE`.
- **Bug fix:** Kept offline green and alpha generation at original source resolution regardless of realtime output-size settings.
- **Core update:** Deduplicated alpha packing so offline alpha uses the shared `pipeline.alpha_packer.AlphaPacker` implementation.
- **Bug fix:** Removed the default 30 FPS cap from offline generation.
- **Bug fix:** Added an early startup GPU capability gate for compute capability below 7.5.

### 2026-05-15

- **Core/performance upgrade:** Added staged PyNv/8K performance tooling and probes for decode, encode, mux, and end-to-end passthrough measurement.
- **Core/performance upgrade:** Added PyNv threaded decoder experiments, slot ownership handling, encode input lifetime safeguards, FP16 RVM benchmark support, and TensorRT/CUDA provider diagnostics.

### 2026-05-14

- **Packaging/core fix:** Added CuPy/CUDA packaging dependency handling and improved runtime CUDA DLL loading for frozen Windows builds.

### 2026-05-13

- **v0.1.0-alpha.1 officially released for limited public beta testing.**
- **UI update:** Added the cold-start startup overlay for long first GPU warmup, local startup status polling, one-click diagnostic report copying, and structured startup failure reporting.

### 2026-05-12

- **Packaging fix:** Added PyInstaller build fixes for Qt/ICU DLL conflicts and defensive checks to reduce duplicate or incompatible DLL collection.

### 2026-05-11

- **UI update:** Added the first PySide6 desktop UI, including realtime server controls, quick configuration, version display, status bar, log side panel, language selector, and multi-video-directory configuration.
- **UI/core update:** Added subtitle settings and preview UI work.
- **Core update:** Added server alpha passthrough entries, dual green/alpha passthrough listings, alpha fisheye output, alpha block layout correction, transparent zero-alpha overlay behavior, and audio post-mux support for alpha output.

### 2026-05-10

- **Core update:** Added offline RVM passthrough generation.
- **Core update:** Added MatAnyone2 ONNX export tooling and first offline runtime integration.
- **Core update:** Added SAM3/MatAnyone2 experimental segmentation workflow, including low-memory modes and active segment planning.
- **Core update:** Added AAC cache, audio normalization, and live-session cache improvements for live playback.

### 2026-05-09

- **Core update:** Added player-specific live passthrough handling for MoonVR/VLC, Skybox/libmpv, nPlayer/OPlayer-style clients, and default clients.
- **Core update:** Added live passthrough active-slot ownership and preemption rules.
- **Core update:** Added PyNv production audio mux integration and AAC/MPEG-TS timestamp handling.
- **Bug fix:** Added Main10/P010/P016 compatibility experiments and conversion paths for PyNv passthrough.

### 2026-05-08

- **Core/performance upgrade:** Added PyNv production stream initial integration, including encoder, mux, decode-to-encode, and GPU matting probes.
- **Core update:** Added pseudo-VOD byte seek integration, passthrough live mode, and HEVC live support.
- **Core update:** Added DLNA physical directory browsing, thumbnails, live-only listing adjustments, live chapter containers, and short-video direct play behavior.
- **Core/performance upgrade:** Added GPU runtime cache support and ONNX Runtime CUDA cold-start support tooling.

### 2026-05-07

- **Core/performance upgrade:** Added output FPS cap configuration, alpha stride reuse, RVM model selection, CUDA IOBinding experiments, GPU NV12 preprocess, and fused NV12-to-NV12 green composite kernels.
- **Core update:** Added PyNvVideoCodec dependency and initial PyNv decode/matting bridge code.

### 2026-05-06

- **Core/performance upgrade:** Added CUDA decoder diagnostics, FFmpeg hardware decode candidate selection, decode output FPS/dimension propagation, and matting profiling.
- **Core/performance upgrade:** Added optimized green-screen composite path that avoids full-frame green background allocation.
- **Core update:** Added initial DLNA time-seek metadata, passthrough HEAD support, and `PT_CONTAINER` support for MP4 and MPEG-TS passthrough output.

## 中文

### 2026-06-21

- **SI 实时 MPEG-TS 传输：** 新增 `/si_live/{name}` 流式路由，基于同名 `.si.wav` sidecar 实时输出 SI 混音 MPEG-TS，将 SI 播放从整片 progressive virtual MP4 预生成切换为零缓存实时传输，并支持 `?t=` 起播偏移。
- **DLNA `[SI]` 时间索引：** `[SI]` 入口现在显示为目录，包含 `[选择时间索引]` 文件夹、快速章节叶子、10 分钟分组、分钟目录和 5 秒一个的可播放起点，播放 URL 统一指向 `/si_live/*.ts`。
- **SI 缓存路径清理：** 保留旧 `/media_si` progressive virtual MP4 路由作为休眠 fallback，同时从当前 DLNA 浏览路径移除 SI 预热接线和 progressive cache 旧 item helper。

### 2026-06-20

- **2D 转 3D / VR 模型更新：** 新增面向离线 16:9 任务的 NVDS ONNX 时域稳定、backbone/head 拆分运行支持、默认 512x288 NVDS 档位、DA3 Large 1036 预设，以及运行前缺失 DA3/NVDS 模型下载对话框。
- **DLNA Live 时间索引：** Passthrough Live 入口现在统一显示为目录，包含本地化的 `[选择时间索引]` 目录、10 分钟分组、分钟目录和 5 秒一个的可播放起点；绿幕和 Alpha Live 目录分别显示 `[GREEN]` 与 `[ALPHA]` 前缀。
- **默认值与文案清理：** 实验性 `passthrough_seek` DLNA 入口默认隐藏，2DVR“存在则跳过”复选框改为描述目标文件，而不是透视文件。

### 2026-06-19

- **2DVR 稳定器迭代：** 新增 GPU near-map 稳定器实验；真机验证发现破坏性伪影后，将该稳定器默认关闭，同时保留诊断控制。
- **离线 2DVR UI 更新：** 旧的处理精度选择改为画质速度控制，并在单文件和批量 2DVR 任务之间共享设置。
- **离线 2DVR 播放修复：** 排查生成文件卡顿问题，并通过在 PyNv HEVC 输出中重复 VPS/SPS/PPS 参数集（`repeatspspps=1`）修复当前路径，同时保持 GOP 长度默认值不变。

### 2026-06-18

- **SI progressive 传输默认开启：** 新增 SI 专用 progressive virtual MP4 开关，默认启用 `/media_si` 和基于同名 `.si.wav` sidecar 的 DLNA `[SI]` 入口，并与旧的可 seek passthrough 实验解耦。
- **SI 首播与同步改进：** 新增后台预热和有界队列、样本表/布局缓存、源音频 sidecar 解耦、AAC 混音加速、音频 edit-list 控制、更准确的大小估算，以及 UI 文案和帮助说明更新。
- **2DVR 时域稳定更新：** 新增渲染端深度归一化和 near-map 时域稳定、场景切换重置、离线进度日志，以及后续优化用的 GPU near 预处理基线。

### 2026-06-17

- **SI 功能基座：** 新增同名 `.si.wav` 检测、DLNA `[SI]` 入口、本地 SI 混音控制 API、首页 SI 设置，以及第一版 `/media_si` 实现；随后传输层方向切换为 progressive virtual MP4。
- **Progressive MP4 预研路径：** 新增 PyAV 原型工具、播放器 Phase 0 样本生成、PyInstaller/PyAV 兼容探针，以及首版 `pipeline.si_virtual_mp4` 样本表和虚拟 Range 基础件。
- **2DVR 伪影清理：** 新增 GPU 竖线/边缘污染清理，强化 CuPy sm_120 direct-cubin 行为，并修复 DA3 启动 heartbeat 调用，使实时 DA3 预热可以正常运行。

### 2026-06-16

- **实时 2D 转 3D UI：** 新增实时 2D 转 3D 设置、服务端环境变量传递、面向用户的 3D 强度控制，以及用于降低首次播放等待的 DA3 引擎预热。
- **2DVR soft-shift 质量修复：** 修复左右眼 hole-fill 方向、深度边缘拖影、硬轮廓/背景接缝处理、前景膨胀、混合 inverse-warp 补洞，以及背景侧 rim 清理；UI 现在把 soft-shift 作为产品路径，并隐藏 `inverse_warp`。
- **DA3 模型预设更新：** 新增普通/高清 DA3 预设、按预设隔离的 TensorRT 缓存、Hugging Face / 镜像感知模型下载，并更新 DA3 导出文档。

### 2026-06-15

- **离线 2D 转 3D / VR 生成：** 新增基于 DA3 ONNX 的离线 2D 转 SBS 3D/VR，支持 CLI 和 PySide UI 中的单文件、批量、开始/时长和片段流程。
- **GPU 常驻 2DVR 链路：** 新增 PyNv/CuPy GPU 路径，覆盖解码、DA3 深度、立体渲染、NVENC 输出和 VR 投影，并默认使用 TensorRT 加速 DA3。
- **实时 2D 转 3D 基座：** 新增 `two_dvr` passthrough 模式、DLNA `[2D>3D]` live 入口、`/passthrough_live mode=two_dvr`、离线输出识别/命名，以及 DA3 TensorRT 启动预热。
- **CUDA 运行时加固：** 转向使用自包含的 pip CUDA 运行时组件支持 CuPy/ONNX Runtime 和 sm_120 级 GPU，降低对系统 CUDA toolkit 的依赖。

### 2026-06-01

- **v0.1.0 发布。**
- **Skybox 实时播放修复：** DLNA 广播的 passthrough-live URL 末尾追加 `.ts` 路由提示后缀，避免 Skybox 因 `.mp4` URL 选择 MP4 pipeline 后拒绝实际 MPEG-TS 字节；路由端会在查找源文件前剥离 `.ts`、`.m2ts` 和 `.mpegts`，旧缓存 URL 与其它客户端保持兼容。
- **Skybox/libmpv 探测保护：** 纯 `libmpv` UA 的截图探测现在会在占用 slot、启动 GPU 工作或读取现有 live session 前快速返回 `503`；libmpv 同 live key 启动请求会 debounce，same-owner preempt 也不再关闭已经构建完成的实时流或 `LiveSession`。

### 2026-05-31

- **离线单视频 UI 更新：** 单视频页签移除“测试”措辞，时间行新增“开始时间和时长”与“时间片段合并”模式选择，并加入可持久化的 `HH:MM:SS` 片段配置和校验。
- **离线片段合并：** `offline.convert single --segment START-END` 现在支持多个时间片段复用现有离线链路生成分段 MP4，再通过 FFmpeg concat copy 合并为一个最终输出。

### 2026-05-30

- **启动崩溃修复：** 媒体目录解析现在会跳过无法 `resolve()` 的 Windows 云盘/rclone 根目录（如 PikPak 挂载），不再让 UI 启动或后端 `config.py` 导入崩溃；其它有效本地媒体目录继续可用。

### 2026-05-29

- **离线单视频时间控制：** 新增自定义结束时间选项，启动前会探测视频时长并校验开始、结束和时长范围，再转换为现有 `--start` 与 `--duration` 离线命令参数；同步新增英文、中文和日文翻译。
- **离线输出命名：** 带正时长的单视频输出文件名现在包含结束时间标记，例如 `S000500_E000615_75S`；旧的开始时间加时长命名仍可被识别为生成的离线输出。
- **DLNA 图片浏览测试开关：** 新增默认关闭的 `PT_DLNA_IMAGE_ENABLED` / UI 设置，用于可选 DLNA 图片浏览测试；启用时支持图片 MIME/protocol 广播、DIDL 图片项，以及 `/media` 对支持图片格式的服务。

### 2026-05-28

- **实时并发更新：** 将进程级 Matter singleton 替换为惰性 Matter 池，新增按 NVIDIA 显存解析的 `PT_PASSTHROUGH_MAX_CONCURRENT=auto`，并让 live 与 non-live passthrough 按流 acquire/release Matter，避免并发串用循环状态或死锁。
- **可 seek passthrough 阶段 1：** 新增 opt-in 的 `/passthrough_seek/{name}` 路由、byte range 到时间的映射、虚拟 MPEG-TS 大小处理、UA/profile 保护，以及由显式 enable/DLNA 开关控制的附加 DLNA seek 条目；原 live 路由保持不变。
- **DLNA/播放器兼容诊断：** 新增请求历史 middleware、脱敏 JSONL、trace id、本机 debug endpoint，以及对 DLNA、media、passthrough 请求的影子 profile/intent/decision 分类；当前仅观察审计，不改变播放行为。

### 2026-05-27

- **v0.1.0-beta.6 发布。**
- **MatAnyone2/识别链路更新：** MatAnyone2 中速默认前置链路改为 `YOLO26m -> EfficientSAM -> MatAnyone2`，当前 EfficientSAM 模型路径迁移到 `models/efficientsam/`，并从离线 UI 移除旧的 YOLOWorld-EfficientSAM 选项。
- **YOLO26m 质量修复：** YOLO26m 默认改用 FP32 ONNX 模型，因为已确认 FP16 导出在 ORT CUDA 下会发生输出塌缩；新增 FP16 异常探测警告，并改进左右眼不对称处理、场景感知/边界 gap-fill，以及 `top_k=0` 的不限人数配对。
- **MatAnyone2 质量修复：** 通过抗拖影默认参数、不确定区域门控、sensory 状态衰减、last-mask 二值化、bootstrap 多轮 refine，以及验证后的 240 帧分段默认值，降低运动拖影和残影；同时修复 SBS 左眼缓冲区被右眼预处理覆盖导致左眼拖影更重的问题。
- **TensorRT 更新：** 实时 RVM、离线 RVM、离线 MatAnyone2 的 TensorRT cache 构建默认都改为 FP32。MatAnyone2 TensorRT 现在只构建 batch 1 的 `512` 和 `1024` cache，并在 `runtime_cache/trt_engines/matanyone2/` 下按模型目录隔离。
- **UI 更新：** 首页实时输出 FPS 在 40 和 60 之间新增 50；离线 MatAnyone2 精度显示 `512` 与 `1024`，默认仍选 `1024`；离线识别模型只保留 YOLO26m-EfficientSAM 和 SAM3。
- **离线/DLNA 清理：** DLNA 对已生成的离线 passthrough 输出增加 `[Offline]` 虚拟标题前缀；离线 passthrough 在成功、失败和前置识别提前退出时都会清理临时 `.aac` 音频 sidecar。
- **文档更新：** 更新 EfficientSAM、YOLO26m、MatAnyone2 模型准备说明，匹配当前 YOLO26m-EfficientSAM-MatAnyone2 链路、YOLO26m FP32 默认模型，以及 MatAnyone2 `512`/`1024` TensorRT 要求。
- **TensorRT UI 修复：** TensorRT cache 构建进度现在把 MatAnyone2 `512` 和 `1024` 只计为两个模型构建单元，并让实时 RVM、离线 RVM 和 MatAnyone2 在子进程成功退出前最高显示 99%，避免 finalization 期间过早显示 100%。

### 2026-05-26

- **MatAnyone2 离线内核更新：** 抽出绿幕和 alpha 共用的 MatAnyone2 离线引擎，新增 batch 1 `step_update` IOBinding 热路径，并保留不支持或出错时回退到 NumPy 的路径。
- **MatAnyone2 稳定性更新：** 新增场景重置规划、MatAnyone2 专用 alpha 平滑控制、带 confidence-band 的实验性 guided alpha upsample，以及实验性 ROI 质量模式；经过验证后 guided 与 ROI 仍默认关闭。
- **MatAnyone2 TensorRT 修复：** MatAnyone2 TensorRT 按模型精度使用独立 cache 子目录，多精度构建拆成独立子进程以避免 ORT/TensorRT 进程全局状态串用，并同步绿幕与 alpha 离线路径的运行时 provider cache path。
- **离线进程修复：** 停止离线转换时现在会在 Windows 上终止完整子进程树，避免 MatAnyone2 prepass、转换进程或 FFmpeg helper 在 UI 显示已停止后继续占用 GPU 显存。
- **离线 RVM TensorRT 更新：** 完成实时/离线 RVM TensorRT cache 拆分、离线精度档完整性检查、watcher 卡顿修复、离线 engine 输出目录修复、离线路径幂等处理，以及实时与离线 cache 之间的 1024 engine 复用。

### 2026-05-25

- **UI/色彩修复：** 将光照匹配预设重新校准到 D65 显示白点。`daylight` 现在是中性的 6500K，`night_cool` 现在是明显偏冷的 8000K，并迁移现有内置预设设置，避免默认自然日光继续偏黄。
- **启动体验更新：** 新增冷启动 heartbeat/status API、启动遮罩进度细节、本地化等待提示、诊断日志、绕过代理的本地轮询，以及单调递增的 warmup 进度，让较长的 CUDA/ONNX/TensorRT 启动过程可见而不是像卡住。
- **离线 RVM 更新：** 新增离线 RVM 处理精度档，移除离线 ResNet50 balanced 路径，加入离线专用场景重置和 alpha 平滑，并引入离线 RVM TensorRT warmup/cache 完整性检查。
- **TensorRT cache 更新：** 拆分实时与离线 RVM TensorRT cache 处理：首页只构建实时 1024 engine，离线模式维护更完整的精度档 cache，同时增加更安全的状态检查和更明确的长时间构建提示。

### 2026-05-24

- **v0.1.0-beta.5 发布。**
- **BUG 修复：** 修复 raw HEVC 到 MPEG-TS mux 路径中 `+nobuffer` 导致首个 GOP 视频被丢弃、进而产生约一个 GOP A/V 偏移的问题。默认关闭 `PT_MUX_NOBUFFER_ENABLE`，同时保留诊断覆盖开关。
- **UI/打包更新：** 为 Windows 打包版新增 TensorRT 运行库下载流程。TensorRT 对话框可检测缺失运行时 DLL，提供自动/手动下载 NVIDIA wheel，校验哈希，仅解压所需 DLL，并在缺少 `Content-Length` 时仍显示进度。
- **内核/UI 更新：** 新增 MatAnyone2 `step_update` 离线 TensorRT cache 支持，分离 RVM 与 MatAnyone2 的 TensorRT manifest/cache 目录，增加单文件/批量离线专用 TensorRT 开关，并从离线 UI 移除 RVM 均衡选项。
- **UI/内核更新：** 光照匹配默认改为自然日光，暖黄光预设降温并降低饱和度；同步刷新首页 TensorRT/FPS 文案与帮助说明。实时 FPS 默认回到 30 FPS，同时保留“同原视频”选项。
- **DLNA 兼容性更新：** 为 4XVR 兼容性将自动 VR 命名从 `_LR_180` 调整为 `_LR_180_SBS`，保留旧生成文件识别，并新增不可见 ObjectID 版本号以促使 SKYBOX 等 DLNA 客户端刷新虚拟名称缓存。
- **日志修复：** 新增精确的 uvicorn socket-send 噪声过滤器，只抑制重复的 `socket.send() raised exception`，保留其他 uvicorn 警告和错误。
- **UI/打包修复：** 隐藏打包版启动、服务启停、离线转换、TensorRT 缓存构建和运行时探测时闪现的 Windows 黑色控制台窗口，同时保留日志输出。
- **BUG 修复：** 加固离线转换子进程日志转发，冷启动 CUDA/ONNX/TensorRT 时不再表现为静默停止，并在日志中显示子进程和 UI 外层退出状态。
- **兼容性修复：** 加固 Windows 下 ffprobe JSON 读取，改为按 UTF-8 字节解码，提升含中文/日文路径和元数据时的可靠性。
- **TensorRT 修复：** TensorRT 运行库安装不再覆盖已加载 DLL；TensorRT 构建错误（如 ONNX 模型缺失）会持续显示，不再一闪而过。
- **运行时修复：** 改进 UI 进程生命周期处理、服务强制停止、启动状态轮询稳定性，脱敏透视任务日志，并为繁忙或被抢占的透视请求返回重试提示。

### 2026-05-23

- **性能/内核更新：** 新增 Track A 首包延迟诊断和启动 warmup，包括 mux 时间标记、CuPy composite/alpha warmup、NVENC 启动预检，以及 pipe-TS 路径验证后的低延迟 mux 默认值。
- **BUG 修复：** 修复 nPlayer 进入 audio-only 模式的回归问题：在 pipe-TS 视频阶段应用 `setts` 时间戳时移除 `hevc_metadata=aud=insert`，使严格播放器能拿到可用 HEVC codec 参数并进入视频播放。
- **BUG 修复：** 将 FastAPI 启动注册迁移到基于 lifespan 的 `create_app(startup_hook=...)`，修复 `on_event` 弃用警告。
- **A/V 同步更新：** 通过限制 slate 初始突发帧并对后续 slate 帧按 wallclock pacing，降低启动 slate A/V 偏移；同时新增 slate 和 MPEG-TS 同步验证工具。
- **A/V 同步调查/更新：** 新增 live MPEG-TS capture、音频内容对齐和视频内容对齐工具；调整默认 live 路径以避开 slate/cache 启动偏移，在禁用 AAC cache 时直接使用源音频，并默认关闭生成的视频 slate 供真机 A/B 测试。
- **UI/运行时更新：** 调整首页实时输出 FPS 处理以支持源帧率测试，增加源帧率 pacing，并记录当服务器侧 capture 已客观对齐时，剩余主观 A/V 问题可能来自客户端播放行为。

### 2026-05-22

- **性能/内核更新：** 改进 TensorRT 冷启动 warmup：启动阶段预热进程全局 `Matter` singleton，预加载 warmup shape 的 static TensorRT session，warmup 后重置循环状态，并将启动 warmup 控制从全局 matting 配置变更中剥离。
- **性能/内核更新：** 新增 composite/alpha CuPy warmup 和 NVENC 启动预检，降低首次播放承担 kernel JIT 或编码器初始化成本的概率。
- **内核更新：** 离线 RVM fast 模式可在 cache ready 时使用 TensorRT；不支持的离线引擎强制使用 CUDA/CPU provider，并增加更清晰的 TensorRT provider 诊断输出。
- **BUG 修复：** 修复 UI 启动的子进程在关闭 TensorRT 后仍可能继承旧 provider 环境的问题；static TensorRT 激活失败后也不再逐帧重试并刷屏日志。
- **性能 BUG 修复：** 修复离线 alpha RVM 在 NVENC 编码前缺少 CUDA stream 同步导致的吞吐问题。8K 测试路径离线 alpha 从约 36 FPS 提升到约 75 FPS。
- **BUG 修复：** 修复实时 FPS cap 下响应头误报帧率的问题；当源视频帧率低于配置 cap 时，live header 现在使用实际有效输出 FPS。
- **BUG 修复：** 改进 live 音频 cache 清理逻辑，避免中断 `communicate()` 时过早关闭 pipe 导致噪声 traceback，并将被中断的 slate cache 构建作为正常清理记录。

### 2026-05-21

- **DLNA/离线命名更新：** 将 VR/player 文件名处理集中到 `utils.vr_naming`，并统一用于 DLNA 标题、离线默认输出名和生成文件识别。
- **兼容性更新：** 将自动生成的 2:1 half-equirectangular 命名从 `_SBS_180` 调整为 `_LR_180`，并将 alpha 命名调整为 `_LR_180_FISHEYE_F180_alpha`，提升包括 HereSphere 在内的播放器兼容性。
- **DLNA 更新：** 提升 DIDL schema 版本并刷新 live/raw 命名规则，使生成的绿幕和 alpha 入口使用更兼容播放器的标记，同时保持播放 URL 不变。
- **离线更新：** RVM 与 alpha 离线默认输出名改用新的 VR 命名规则，同时继续识别旧版生成文件后缀。
- **BUG 修复：** 改进音频 cache 中断清理，避免 stream 关闭时出现误导性的 `ValueError: I/O operation on closed file` reader-thread traceback。

### 2026-05-20

- **v0.1.0-beta.4 修补和新增功能。**
- **BUG 修复：** 所有离线生成引擎默认改为按源视频码率输出，包括 RVM 快速/均衡和 MatAnyone2 中速/慢速，使生成文件大小更接近原始视频。读取不到源码率时回退到 40 Mbps。
- **BUG 修复：** 修复播放或停止服务器后 FFmpeg 子进程后台常驻的问题。PyNv 流现在会跟踪并停止音频 FFmpeg 子进程，关闭时等待 slate 音频/缓存线程，清理部分启动失败的 pipe-TS muxer，删除残留临时 AAC 文件；UI 强制停止服务器时也会在 Windows 上通过 `taskkill /T /F` 终止子进程。
- **BUG 修复：** 修复实时 2D alpha 播放中 alpha 区块逐渐变灰的问题，默认实时 RVM 模型切回 FP32。根因是 RVM 的 `rec1`-`rec4` 循环状态在 FP16 下逐帧累积精度误差，不是 alpha 打包或实时码率控制。

### 2026-05-19

- **BUG 修复：** 修复 MatAnyone2 中速离线 alpha 在 HEVC Main10/P016 风格解码帧上的前置识别崩溃，YOLO-World/EfficientSAM 或 SAM3 预处理前会先把 16-bit NV12/P010 平面转换成 8-bit BGR。
- **BUG 修复：** 更新 AV1 后端路由，RTX 20/Turing 等不支持 AV1 NVDEC 的显卡会走 FFmpeg 解码 fallback，不再等到 PyNv 解码取帧阶段才失败。
- **UI 更新：** 新增实时透视前景光照匹配功能，包括首页独立面板、预设、自定义设置对话框、持久化 UI 设置，以及播放中的运行时更新。
- **内核更新：** 新增 DLNA `[NoLive]` 标记，并对已知不支持实时处理的源直接拒绝实时入口，避免误走实时 fallback。

### 2026-05-18

- **v0.1.0-beta.3 修补和新增功能。**
- **内核/性能升级：** 新增 MatAnyone2 中速离线模式，使用 YOLO-World + EfficientSAM 作为 MatAnyone2 传播前的前置识别模型。
- **内核/性能升级：** 降低 MatAnyone2 中速峰值显存，YOLO-World/EfficientSAM 前置改为子进程运行，MatAnyone2 离线默认 batch 1 且关闭 SBS batch。
- **内核升级：** 改进 SAM3 前置的 MatAnyone2 慢速模式，包括共享 SAM3 helper、左右眼 mask 一致性防护、短 inactive 缺口填补，以及可配置 SAM3 文本提示词。
- **UI 更新：** 调整离线 UI，MatAnyone2 作为统一引擎显示，下方增加识别模型选择，可选 `YOLOWorld-EfficientSAM` 或 `SAM3 (16GB+ VRAM)`，并增加 SAM3 专用提示词对话框。
- **UI/内核更新：** 新增 2D alpha 输出控制，包括 fisheye/flat3d 投影模式、按距离计算视差、flat3d 方形单眼画布，以及首页 2D alpha 设置入口。

### 2026-05-17

- **v0.1.0-beta.1 正式发布公测。**
- **v0.1.0-beta.2 修补和新增功能。**
- **UI 更新：** 将实时/离线界面的显存配置替换为画质/速度预设，并增加独立离线画质设置。
- **UI 更新：** 新增 `GET /runtime_status`，并在主窗口状态栏中间显示当前 FPS 和显存占用。
- **BUG 修复：** 新增共享 NVIDIA 计算能力检测，并在实时启动和离线转换中加入硬性拦截。
- **内核更新：** 新增离线源编码预检，并为 MPEG-4 Visual / `mp4v-20` 等非 PyNv 源增加 FFmpeg NV12 解码 fallback。
- **内核更新：** 新增 4XVR live 播放配置，使 AVPro/ExoPlayer 的重连可以复用托管 live session。
- **UI/内核更新：** 新增普通 2D 视频 alpha 直通，将非 SBS 2D 视频投影为双眼鱼眼 SBS 输出，并支持 FOV 和视差配置。

### 2026-05-16

- **内核/性能升级：** 为正数实时 FPS cap 增加 producer pacing，使限制 FPS 时生产端也同步节流。
- **内核/性能升级：** 新增实时 PyNv 解码后的 GPU 缩放，配置项为 `PT_DECODE_MAX_SIDE`。
- **BUG 修复：** 离线绿幕和离线 alpha 生成保持源视频原尺寸输出，不跟随实时输出尺寸设置。
- **内核更新：** 去重 alpha packer，离线 alpha 改用共享的 `pipeline.alpha_packer.AlphaPacker` 实现。
- **BUG 修复：** 移除离线生成默认 30 FPS 限制。
- **BUG 修复：** 新增启动阶段 GPU 算力门槛检查，compute capability 低于 7.5 时快速失败。

### 2026-05-15

- **内核/性能升级：** 新增 PyNv/8K 分阶段性能工具和 decode、encode、mux、端到端 passthrough probe。
- **内核/性能升级：** 新增 PyNv threaded decoder 实验、slot ownership 处理、encode input lifetime 保护、FP16 RVM benchmark 支持，以及 TensorRT/CUDA provider 诊断。

### 2026-05-14

- **打包/内核修复：** 新增 frozen build 下的 CuPy/CUDA 打包依赖处理，并改进 Windows 打包版本的运行时 CUDA DLL 加载。

### 2026-05-13

- **v0.1.0-alpha.1 正式发布小范围公测。**
- **UI 更新：** 新增首次 GPU warmup 长等待场景的启动遮罩、本地启动状态轮询、一键复制诊断报告，以及结构化启动失败提示。

### 2026-05-12

- **打包修复：** 新增 PyInstaller 打包中 Qt/ICU DLL 冲突修复和防御检查，减少重复或不兼容 DLL 被收集。

### 2026-05-11

- **UI 更新：** 新增第一版 PySide6 桌面 UI，包括实时服务器控制、快速配置、版本显示、状态栏、日志侧栏、语言选择和多视频目录配置。
- **UI/内核更新：** 新增字幕设置和预览 UI。
- **内核更新：** 新增服务端 alpha 直通入口、绿幕/alpha 双入口列表、alpha 鱼眼输出、alpha block 布局修正、透明零 alpha overlay 行为，以及 alpha 输出音频后混流。

### 2026-05-10

- **内核更新：** 新增离线 RVM passthrough 生成。
- **内核更新：** 新增 MatAnyone2 ONNX 导出工具和首个离线运行时集成。
- **内核更新：** 新增 SAM3/MatAnyone2 实验性分割流程，包括低显存模式和 active segment plan。
- **内核更新：** 新增 live 播放 AAC 缓存、音频归一化和 live-session 缓存改进。

### 2026-05-09

- **内核更新：** 新增 MoonVR/VLC、Skybox/libmpv、nPlayer/OPlayer 风格客户端和默认客户端的播放器专用 live passthrough 处理。
- **内核更新：** 新增 live passthrough active-slot ownership 和 preemption 规则。
- **内核更新：** 新增 PyNv 生产音频 mux 集成和 AAC/MPEG-TS 时间戳处理。
- **BUG 修复：** 新增 Main10/P010/P016 兼容性实验和 PyNv passthrough 转换路径。

### 2026-05-08

- **内核/性能升级：** 新增 PyNv 生产流初始集成，包括 encoder、mux、decode-to-encode 和 GPU matting probe。
- **内核更新：** 新增 pseudo-VOD byte seek 集成、passthrough live 模式和 HEVC live 支持。
- **内核更新：** 新增 DLNA 物理目录浏览、缩略图、live-only 列表调整、live chapter 容器和短视频 direct play 行为。
- **内核/性能升级：** 新增 GPU runtime cache 和 ONNX Runtime CUDA cold-start 支持工具。

### 2026-05-07

- **内核/性能升级：** 新增输出 FPS cap 配置、alpha stride reuse、RVM 模型选择、CUDA IOBinding 实验、GPU NV12 preprocess 和 fused NV12-to-NV12 green composite kernel。
- **内核更新：** 新增 PyNvVideoCodec 依赖和初始 PyNv decode/matting bridge。

### 2026-05-06

- **内核/性能升级：** 新增 CUDA decoder 诊断、FFmpeg 硬件解码候选选择、decoder 输出 FPS/尺寸传播和 matting profiling。
- **内核/性能升级：** 新增优化版绿幕 composite 路径，避免整帧绿色背景分配。
- **内核更新：** 新增初始 DLNA time-seek metadata、passthrough HEAD 支持，以及 `PT_CONTAINER` 对 MP4 和 MPEG-TS passthrough 输出的支持。
