# Changelog

This file only keeps version releases, major bug fixes, major UI/UX updates, and major core or performance upgrades.

## English

### 2026-05-20

- **Major bug fix:** Offline generation now targets the source video bitrate by default for all engines, including RVM fast/balanced and MatAnyone2 medium/slow. This keeps generated video size closer to the original source. If source bitrate is unavailable, offline generation falls back to 40 Mbps.
- **Major bug fix:** Fixed lingering FFmpeg child processes after playback or server stop. PyNv streams now track and stop audio FFmpeg subprocesses, wait for slate audio/cache threads during close, clean up partially spawned pipe-TS muxers, remove stale temporary AAC files, and forced UI server stop now terminates child processes through `taskkill /T /F` on Windows.

### 2026-05-19

- **Major bug fix:** Fixed MatAnyone2 medium offline alpha prepass crashes on HEVC Main10/P016-style decoded frames by converting 16-bit NV12/P010 planes to 8-bit BGR before YOLO-World/EfficientSAM or SAM3 prepass processing.
- **Major bug fix:** Updated AV1 backend routing so GPUs without AV1 NVDEC support, such as RTX 20/Turing, use the FFmpeg decode fallback instead of failing later inside PyNv decode.
- **Major UI update:** Added foreground-only Light Matching for realtime passthrough, including a dedicated Home-page panel, presets, custom settings dialog, persisted UI settings, and live runtime updates during playback.
- **Major core update:** Added DLNA `[NoLive]` labeling and realtime-source rejection for known unsupported live sources, avoiding confusing realtime fallback attempts.

### 2026-05-18

- **v0.1.0-beta.3 patches and additions.**
- **Major core/performance upgrade:** Added MatAnyone2 medium offline mode using YOLO-World + EfficientSAM as the bootstrap recognizer before MatAnyone2 propagation.
- **Major core/performance upgrade:** Reduced MatAnyone2 medium peak VRAM by moving the YOLO-World/EfficientSAM prepass into a subprocess and defaulting MatAnyone2 offline processing to batch 1 without SBS batching.
- **Major core upgrade:** Improved SAM3-backed MatAnyone2 slow mode with shared SAM3 helper code, stereo mask consistency guarding, short inactive-gap filling, and configurable SAM3 text prompts.
- **Major UI update:** Updated the offline UI so MatAnyone2 is selected once, with a recognition-model selector for `YOLOWorld-EfficientSAM` or `SAM3 (16GB+ VRAM)`, plus a SAM3-only prompt dialog.
- **Major UI/core update:** Added flat 2D alpha output controls, including fisheye/flat3d projection mode, distance-based disparity, square-eye flat3d sizing, and Home-page 2D alpha settings.

### 2026-05-17

- **v0.1.0-beta.1 officially released for public beta testing.**
- **v0.1.0-beta.2 patches and additions.**
- **Major UI update:** Replaced the previous realtime/offline VRAM profile UI with Quality / Speed presets, including separate offline quality settings.
- **Major UI update:** Added `GET /runtime_status` and a centered main status-bar indicator for current FPS and VRAM usage.
- **Major bug fix:** Added shared NVIDIA compute capability checks and hard gates for realtime startup and offline conversion.
- **Major core update:** Added offline source-codec preflight plus an FFmpeg NV12 decode fallback for non-PyNv sources such as MPEG-4 Visual / `mp4v-20`.
- **Major core update:** Added a dedicated 4XVR live playback profile so AVPro/ExoPlayer reconnects can reuse managed live sessions.
- **Major UI/core update:** Added flat 2D alpha passthrough for non-SBS 2D videos, projecting them into stereo fisheye SBS output with configurable FOV and disparity.

### 2026-05-16

- **Major core/performance upgrade:** Added producer pacing for positive realtime FPS caps, so capped output also throttles production.
- **Major core/performance upgrade:** Added realtime GPU resize after PyNv decode via `PT_DECODE_MAX_SIDE`.
- **Major bug fix:** Kept offline green and alpha generation at original source resolution regardless of realtime output-size settings.
- **Major core update:** Deduplicated alpha packing so offline alpha uses the shared `pipeline.alpha_packer.AlphaPacker` implementation.
- **Major bug fix:** Removed the default 30 FPS cap from offline generation.
- **Major bug fix:** Added an early startup GPU capability gate for compute capability below 7.5.

### 2026-05-15

- **Major core/performance upgrade:** Added staged PyNv/8K performance tooling and probes for decode, encode, mux, and end-to-end passthrough measurement.
- **Major core/performance upgrade:** Added PyNv threaded decoder experiments, slot ownership handling, encode input lifetime safeguards, FP16 RVM benchmark support, and TensorRT/CUDA provider diagnostics.

### 2026-05-14

- **Major packaging/core fix:** Added CuPy/CUDA packaging dependency handling and improved runtime CUDA DLL loading for frozen Windows builds.

### 2026-05-13

- **v0.1.0-alpha.1 officially released for limited public beta testing.**
- **Major UI update:** Added the cold-start startup overlay for long first GPU warmup, local startup status polling, one-click diagnostic report copying, and structured startup failure reporting.

### 2026-05-12

- **Major packaging fix:** Added PyInstaller build fixes for Qt/ICU DLL conflicts and defensive checks to reduce duplicate or incompatible DLL collection.

### 2026-05-11

- **Major UI update:** Added the first PySide6 desktop UI, including realtime server controls, quick configuration, version display, status bar, log side panel, language selector, and multi-video-directory configuration.
- **Major UI/core update:** Added subtitle settings and preview UI work.
- **Major core update:** Added server alpha passthrough entries, dual green/alpha passthrough listings, alpha fisheye output, alpha block layout correction, transparent zero-alpha overlay behavior, and audio post-mux support for alpha output.

### 2026-05-10

- **Major core update:** Added offline RVM passthrough generation.
- **Major core update:** Added MatAnyone2 ONNX export tooling and first offline runtime integration.
- **Major core update:** Added SAM3/MatAnyone2 experimental segmentation workflow, including low-memory modes and active segment planning.
- **Major core update:** Added AAC cache, audio normalization, and live-session cache improvements for live playback.

### 2026-05-09

- **Major core update:** Added player-specific live passthrough handling for MoonVR/VLC, Skybox/libmpv, nPlayer/OPlayer-style clients, and default clients.
- **Major core update:** Added live passthrough active-slot ownership and preemption rules.
- **Major core update:** Added PyNv production audio mux integration and AAC/MPEG-TS timestamp handling.
- **Major bug fix:** Added Main10/P010/P016 compatibility experiments and conversion paths for PyNv passthrough.

### 2026-05-08

- **Major core/performance upgrade:** Added PyNv production stream initial integration, including encoder, mux, decode-to-encode, and GPU matting probes.
- **Major core update:** Added pseudo-VOD byte seek integration, passthrough live mode, and HEVC live support.
- **Major core update:** Added DLNA physical directory browsing, thumbnails, live-only listing adjustments, live chapter containers, and short-video direct play behavior.
- **Major core/performance upgrade:** Added GPU runtime cache support and ONNX Runtime CUDA cold-start support tooling.

### 2026-05-07

- **Major core/performance upgrade:** Added output FPS cap configuration, alpha stride reuse, RVM model selection, CUDA IOBinding experiments, GPU NV12 preprocess, and fused NV12-to-NV12 green composite kernels.
- **Major core update:** Added PyNvVideoCodec dependency and initial PyNv decode/matting bridge code.

### 2026-05-06

- **Major core/performance upgrade:** Added CUDA decoder diagnostics, FFmpeg hardware decode candidate selection, decode output FPS/dimension propagation, and matting profiling.
- **Major core/performance upgrade:** Added optimized green-screen composite path that avoids full-frame green background allocation.
- **Major core update:** Added initial DLNA time-seek metadata, passthrough HEAD support, and `PT_CONTAINER` support for MP4 and MPEG-TS passthrough output.

## 中文

### 2026-05-20

- **重大 BUG 修复：** 所有离线生成引擎默认改为按源视频码率输出，包括 RVM 快速/均衡和 MatAnyone2 中速/慢速，使生成文件大小更接近原始视频。读取不到源码率时回退到 40 Mbps。
- **重大 BUG 修复：** 修复播放或停止服务器后 FFmpeg 子进程后台常驻的问题。PyNv 流现在会跟踪并停止音频 FFmpeg 子进程，关闭时等待 slate 音频/缓存线程，清理部分启动失败的 pipe-TS muxer，删除残留临时 AAC 文件；UI 强制停止服务器时也会在 Windows 上通过 `taskkill /T /F` 终止子进程。

### 2026-05-19

- **重大 BUG 修复：** 修复 MatAnyone2 中速离线 alpha 在 HEVC Main10/P016 风格解码帧上的前置识别崩溃，YOLO-World/EfficientSAM 或 SAM3 预处理前会先把 16-bit NV12/P010 平面转换成 8-bit BGR。
- **重大 BUG 修复：** 更新 AV1 后端路由，RTX 20/Turing 等不支持 AV1 NVDEC 的显卡会走 FFmpeg 解码 fallback，不再等到 PyNv 解码取帧阶段才失败。
- **重大 UI 更新：** 新增实时透视前景光照匹配功能，包括首页独立面板、预设、自定义设置对话框、持久化 UI 设置，以及播放中的运行时更新。
- **重大内核更新：** 新增 DLNA `[NoLive]` 标记，并对已知不支持实时处理的源直接拒绝实时入口，避免误走实时 fallback。

### 2026-05-18

- **v0.1.0-beta.3 修补和新增功能。**
- **重大内核/性能升级：** 新增 MatAnyone2 中速离线模式，使用 YOLO-World + EfficientSAM 作为 MatAnyone2 传播前的前置识别模型。
- **重大内核/性能升级：** 降低 MatAnyone2 中速峰值显存，YOLO-World/EfficientSAM 前置改为子进程运行，MatAnyone2 离线默认 batch 1 且关闭 SBS batch。
- **重大内核升级：** 改进 SAM3 前置的 MatAnyone2 慢速模式，包括共享 SAM3 helper、左右眼 mask 一致性防护、短 inactive 缺口填补，以及可配置 SAM3 文本提示词。
- **重大 UI 更新：** 调整离线 UI，MatAnyone2 作为统一引擎显示，下方增加识别模型选择，可选 `YOLOWorld-EfficientSAM` 或 `SAM3 (16GB+ VRAM)`，并增加 SAM3 专用提示词对话框。
- **重大 UI/内核更新：** 新增 2D alpha 输出控制，包括 fisheye/flat3d 投影模式、按距离计算视差、flat3d 方形单眼画布，以及首页 2D alpha 设置入口。

### 2026-05-17

- **v0.1.0-beta.1 正式发布公测。**
- **v0.1.0-beta.2 修补和新增功能。**
- **重大 UI 更新：** 将实时/离线界面的显存配置替换为画质/速度预设，并增加独立离线画质设置。
- **重大 UI 更新：** 新增 `GET /runtime_status`，并在主窗口状态栏中间显示当前 FPS 和显存占用。
- **重大 BUG 修复：** 新增共享 NVIDIA 计算能力检测，并在实时启动和离线转换中加入硬性拦截。
- **重大内核更新：** 新增离线源编码预检，并为 MPEG-4 Visual / `mp4v-20` 等非 PyNv 源增加 FFmpeg NV12 解码 fallback。
- **重大内核更新：** 新增 4XVR live 播放配置，使 AVPro/ExoPlayer 的重连可以复用托管 live session。
- **重大 UI/内核更新：** 新增普通 2D 视频 alpha 直通，将非 SBS 2D 视频投影为双眼鱼眼 SBS 输出，并支持 FOV 和视差配置。

### 2026-05-16

- **重大内核/性能升级：** 为正数实时 FPS cap 增加 producer pacing，使限制 FPS 时生产端也同步节流。
- **重大内核/性能升级：** 新增实时 PyNv 解码后的 GPU 缩放，配置项为 `PT_DECODE_MAX_SIDE`。
- **重大 BUG 修复：** 离线绿幕和离线 alpha 生成保持源视频原尺寸输出，不跟随实时输出尺寸设置。
- **重大内核更新：** 去重 alpha packer，离线 alpha 改用共享的 `pipeline.alpha_packer.AlphaPacker` 实现。
- **重大 BUG 修复：** 移除离线生成默认 30 FPS 限制。
- **重大 BUG 修复：** 新增启动阶段 GPU 算力门槛检查，compute capability 低于 7.5 时快速失败。

### 2026-05-15

- **重大内核/性能升级：** 新增 PyNv/8K 分阶段性能工具和 decode、encode、mux、端到端 passthrough probe。
- **重大内核/性能升级：** 新增 PyNv threaded decoder 实验、slot ownership 处理、encode input lifetime 保护、FP16 RVM benchmark 支持，以及 TensorRT/CUDA provider 诊断。

### 2026-05-14

- **重大打包/内核修复：** 新增 frozen build 下的 CuPy/CUDA 打包依赖处理，并改进 Windows 打包版本的运行时 CUDA DLL 加载。

### 2026-05-13

- **v0.1.0-alpha.1 正式发布小范围公测。**
- **重大 UI 更新：** 新增首次 GPU warmup 长等待场景的启动遮罩、本地启动状态轮询、一键复制诊断报告，以及结构化启动失败提示。

### 2026-05-12

- **重大打包修复：** 新增 PyInstaller 打包中 Qt/ICU DLL 冲突修复和防御检查，减少重复或不兼容 DLL 被收集。

### 2026-05-11

- **重大 UI 更新：** 新增第一版 PySide6 桌面 UI，包括实时服务器控制、快速配置、版本显示、状态栏、日志侧栏、语言选择和多视频目录配置。
- **重大 UI/内核更新：** 新增字幕设置和预览 UI。
- **重大内核更新：** 新增服务端 alpha 直通入口、绿幕/alpha 双入口列表、alpha 鱼眼输出、alpha block 布局修正、透明零 alpha overlay 行为，以及 alpha 输出音频后混流。

### 2026-05-10

- **重大内核更新：** 新增离线 RVM passthrough 生成。
- **重大内核更新：** 新增 MatAnyone2 ONNX 导出工具和首个离线运行时集成。
- **重大内核更新：** 新增 SAM3/MatAnyone2 实验性分割流程，包括低显存模式和 active segment plan。
- **重大内核更新：** 新增 live 播放 AAC 缓存、音频归一化和 live-session 缓存改进。

### 2026-05-09

- **重大内核更新：** 新增 MoonVR/VLC、Skybox/libmpv、nPlayer/OPlayer 风格客户端和默认客户端的播放器专用 live passthrough 处理。
- **重大内核更新：** 新增 live passthrough active-slot ownership 和 preemption 规则。
- **重大内核更新：** 新增 PyNv 生产音频 mux 集成和 AAC/MPEG-TS 时间戳处理。
- **重大 BUG 修复：** 新增 Main10/P010/P016 兼容性实验和 PyNv passthrough 转换路径。

### 2026-05-08

- **重大内核/性能升级：** 新增 PyNv 生产流初始集成，包括 encoder、mux、decode-to-encode 和 GPU matting probe。
- **重大内核更新：** 新增 pseudo-VOD byte seek 集成、passthrough live 模式和 HEVC live 支持。
- **重大内核更新：** 新增 DLNA 物理目录浏览、缩略图、live-only 列表调整、live chapter 容器和短视频 direct play 行为。
- **重大内核/性能升级：** 新增 GPU runtime cache 和 ONNX Runtime CUDA cold-start 支持工具。

### 2026-05-07

- **重大内核/性能升级：** 新增输出 FPS cap 配置、alpha stride reuse、RVM 模型选择、CUDA IOBinding 实验、GPU NV12 preprocess 和 fused NV12-to-NV12 green composite kernel。
- **重大内核更新：** 新增 PyNvVideoCodec 依赖和初始 PyNv decode/matting bridge。

### 2026-05-06

- **重大内核/性能升级：** 新增 CUDA decoder 诊断、FFmpeg 硬件解码候选选择、decoder 输出 FPS/尺寸传播和 matting profiling。
- **重大内核/性能升级：** 新增优化版绿幕 composite 路径，避免整帧绿色背景分配。
- **重大内核更新：** 新增初始 DLNA time-seek metadata、passthrough HEAD 支持，以及 `PT_CONTAINER` 对 MP4 和 MPEG-TS passthrough 输出的支持。
