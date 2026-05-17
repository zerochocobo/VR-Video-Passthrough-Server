# Changelog

This file lists implemented user-facing and technically relevant changes. Entries are grouped by date. Research findings, validation notes, and superseded conclusions are intentionally omitted.

## English

### 2026-05-17

- **v0.1.0-beta.1 officially released for public beta testing.**
- **v0.1.0-beta.2 Patches and Additions**
- Replaced the previous realtime/offline VRAM profile UI with Quality / Speed presets.
- Set realtime Quality / Speed default to `P1` / ultrafast.
- Added separate offline Quality / Speed settings with offline default `P4` / medium and optional `P7` / veryslow.
- Returned realtime RVM default model to FP32 (`rvm_mobilenetv3_fp32.onnx`).
- Set main UI realtime defaults to 30 FPS and 4096 (4K) output size.
- Labeled the unlimited realtime FPS option as `Unlimited (For Test)`.
- Reset the startup overlay fully before each new server start so stale failure text is cleared.
- Added distinct short-video DLNA ids/metadata for green and alpha live virtual files.
- Added `GET /runtime_status` and a centered main status-bar indicator for current FPS and VRAM usage.
- Added a server start/stop pending-state guard to prevent rapid repeated start/stop clicks.
- Added offline GPU warmup notices and startup prediction logs for first-run offline conversions.
- Added hidden-window handling for all `nvidia-smi` probes used by UI, diagnostics, runtime status, warmup, and offline tools.
- Added shared NVIDIA compute capability checks and hard gates for realtime startup and offline conversion.
- Added offline source-codec preflight plus an FFmpeg NV12 decode fallback for non-PyNv sources such as MPEG-4 Visual / `mp4v-20`.
- Added a dedicated 4XVR live playback profile so AVPro/ExoPlayer reconnects can reuse managed live sessions.
- Added flat 2D alpha passthrough: non-SBS 2D videos are projected into stereo fisheye SBS output with configurable FOV and disparity.
- Updated flat 2D alpha output sizing so each fisheye eye scales by `180 / PT_ALPHA_2D_FOV`, preserving visible detail for the configured FOV.
- Updated realtime and offline alpha paths to create encoders at the computed alpha output size before processing starts.
- Fixed the main UI runtime status URL crash caused by a missing `os` import.
- Added explicit DLNA passthrough mode/version query parameters and alpha-aware resolution metadata for green/alpha virtual files.

### 2026-05-16

- Added producer pacing behavior for positive realtime FPS caps, so capped output also throttles production.
- Added realtime GPU resize after PyNv decode via `PT_DECODE_MAX_SIDE`.
- Kept offline green and alpha generation at original source resolution regardless of realtime output-size settings.
- Updated UI-launched offline jobs to inherit the system CUDA/CUDNN environment.
- Deduplicated alpha packing so offline alpha uses the shared `pipeline.alpha_packer.AlphaPacker` implementation.
- Removed the default 30 FPS cap from offline generation.
- Added decoder/preset diagnostics to offline process logs.
- Reduced noisy per-frame matting diagnostic logs.
- Added an alpha-threaded-decoder safety gate; production alpha falls back to simple decode unless explicitly overridden.
- Added an early startup GPU capability gate for compute capability below 7.5.

### 2026-05-15

- Added staged PyNv/8K performance tooling and probes for decode, encode, mux, and end-to-end passthrough measurement.
- Added PyNv threaded decoder experiments, slot ownership handling, and encode input lifetime safeguards.
- Added FP16 RVM benchmark support and TensorRT/CUDA provider diagnostics for the PyNv path.
- Added FPS cap discovery documentation and superseded markers for contaminated performance summaries.

### 2026-05-14

- Added CuPy/CUDA packaging dependency handling for frozen builds.
- Improved runtime CUDA DLL loading behavior for packaged Windows builds.
- Added packaging diagnostics around CUDA, CuPy, and ONNX Runtime provider availability.

### 2026-05-13

- **v0.1.0-alpha.1 officially released for limited public beta testing.**
- Added the cold-start startup overlay for long first GPU warmup.
- Added startup status polling on the local status port.
- Added one-click diagnostic report copying from the startup overlay.
- Added fallback overlay close behavior when the main server starts but the status poller is unavailable.
- Added structured startup failure reporting for unsupported or failed GPU initialization.

### 2026-05-12

- Added PyInstaller build fixes for Qt/ICU DLL conflicts.
- Added defensive packaging checks to reduce duplicate or incompatible DLL collection.
- Added build-time diagnostics for packaged runtime dependency issues.

### 2026-05-11

- Added the first PySide6 desktop UI pass.
- Added realtime server controls, quick configuration, version display, status bar, and log side panel.
- Added green/alpha mode toggles in the UI.
- Added subtitle settings and preview UI work.
- Added multi-video-directory configuration support.
- Added application icon, shared UI styling, and language selector simplification.
- Added server alpha passthrough entries and dual green/alpha passthrough listings.
- Added alpha fisheye output, alpha block layout correction, transparent zero-alpha overlay behavior, and audio post-mux support for alpha output.

### 2026-05-10

- Added offline RVM passthrough generation.
- Added offline bitrate controls and VR-quality defaults.
- Added subtitle overlay phase 1, including subtitle matching, color handling, and source-path diagnostics.
- Added MatAnyone2 ONNX export tooling and first offline runtime integration.
- Added SAM3/MatAnyone2 experimental segmentation workflow, including low-memory modes and active segment planning.
- Added AAC cache and audio normalization work for live playback.
- Added live-session cache improvements for duplicate player requests.

### 2026-05-09

- Added player-specific live passthrough handling for MoonVR/VLC, Skybox/libmpv, nPlayer/OPlayer-style clients, and default clients.
- Added live passthrough active-slot ownership and preemption rules.
- Added alpha edge cleanup controls and default green composite background.
- Added PyNv production audio mux integration and AAC/MPEG-TS timestamp handling work.
- Added Main10/P010/P016 compatibility experiments and conversion paths for PyNv passthrough.
- Added server log truncation/startup log handling improvements.

### 2026-05-08

- Added PyNv encoder, mux, decode-to-encode, and GPU matting probes.
- Added PyNv production stream initial integration.
- Added pseudo-VOD byte seek integration for passthrough streams.
- Added passthrough live mode and HEVC live support.
- Added adaptive live FPS behavior for very high bitrate 8K sources.
- Added DLNA physical directory browsing, thumbnails, live-only listing adjustments, live chapter containers, and short-video direct play behavior.
- Added centralized environment reads in `config.py` and expanded config documentation.
- Added GPU runtime cache support and ORT CUDA cold-start support tooling.
- Added live request header dump and stalled-client cleanup watchdogs.

### 2026-05-07

- Added output FPS cap configuration via `PT_PASSTHROUGH_MAX_FPS`.
- Added alpha stride reuse via `PT_ALPHA_STRIDE` and `PT_ALPHA_MODE=reuse`.
- Added RVM model selection and RVM MobileNetV3 support.
- Added CUDA IOBinding and shared-stream support for RVM experiments.
- Added GPU NV12 preprocess and fused NV12-to-NV12 green composite kernels.
- Added NVENC tuning environment switches for A/B testing.
- Added PyNvVideoCodec dependency and initial PyNv decode/matting bridge code.
- Added GPU/video probe tooling for CUDA, FFmpeg hwaccels, ONNX Runtime, and PyNv availability.

### 2026-05-06

- Added CUDA decoder diagnostics and FFmpeg hardware decode candidate selection.
- Added decode output FPS/dimension propagation from decoder output metadata.
- Added matting profiling for preprocess, ORT, alpha resize, and composite timing.
- Added optimized green-screen composite path that avoids full-frame green background allocation.
- Added model and throughput controls for matting input size, decode max side, and model path.
- Added initial DLNA time-seek metadata and passthrough HEAD support.
- Added `PT_CONTAINER` support for MP4 and MPEG-TS passthrough output.

## 中文

### 2026-05-17

- **v0.1.0-beta.1正式发布公测**
- **v0.1.0-beta.2修补和新增功能**

- 将实时/离线界面的“显存占用”配置替换为“画质速度”预设。
- 实时模式默认画质速度设为 `P1` / 极速。
- 新增离线独立画质速度设置，离线默认 `P4` / 均衡，并提供 `P7` / 高画质。
- 实时 RVM 默认模型恢复为 FP32 (`rvm_mobilenetv3_fp32.onnx`)。
- UI 主界面实时默认输出 FPS 改为 30，默认输出尺寸改为 4096 (4K)。
- 将实时输出 FPS 的无限制选项标注为测试用途。
- 修复启动遮罩重复启动时的旧失败信息残留问题。
- 为短视频 DLNA 绿幕/alpha live 虚拟文件增加独立 id 和元数据。
- 新增 `GET /runtime_status`，并在主窗口状态栏中间显示当前 FPS 和显存占用。
- 新增服务器启动/停止 pending 状态，避免快速重复点击导致 start/stop 交错。
- 为首次离线生成增加 GPU warmup 提示和启动预测日志。
- 为 UI、诊断、运行状态、warmup 和离线工具中的 `nvidia-smi` 调用统一增加隐藏窗口处理。
- 新增共享的 NVIDIA 计算能力检测，并在实时启动和离线转换中加入硬性拦截。
- 新增离线源编码预检，并为 MPEG-4 Visual / `mp4v-20` 等非 PyNv 源增加 FFmpeg NV12 解码 fallback。
- 新增 4XVR live 播放配置，使 AVPro/ExoPlayer 的重连可以复用托管 live session。
- 新增普通 2D 视频 alpha 直通：非 SBS 2D 视频会投影为双眼鱼眼 SBS 输出，并支持 FOV 和视差配置。
- 更新 2D alpha 输出尺寸规则，单眼鱼眼尺寸按 `180 / PT_ALPHA_2D_FOV` 放大，保留配置 FOV 下的可见细节。
- 实时和离线 alpha 路径现在会在处理开始前按计算后的 alpha 输出尺寸创建编码器。
- 修复主 UI 运行状态 URL 因缺少 `os` import 导致的崩溃。
- 为 DLNA 绿幕/alpha 虚拟文件增加显式模式/版本 query，并补齐 alpha 输出解析度元数据。

### 2026-05-16

- 为正数实时 FPS cap 增加 producer pacing，使限制 FPS 时生产端也同步节流。
- 新增实时 PyNv 解码后的 GPU 缩放，配置项为 `PT_DECODE_MAX_SIDE`。
- 离线绿幕和离线 alpha 生成保持源视频原尺寸输出，不跟随实时输出尺寸设置。
- UI 启动的离线任务改为继承系统 CUDA/CUDNN 环境。
- 去重 alpha packer，离线 alpha 改用共享的 `pipeline.alpha_packer.AlphaPacker`。
- 移除离线生成默认 30 FPS 限制。
- 为离线进程日志增加 decoder/preset 诊断信息。
- 降低 matting 逐帧诊断日志噪声。
- 新增 alpha threaded decoder 安全门；生产 alpha 默认回落 simple decode，除非显式覆盖。
- 新增启动阶段 GPU 算力门槛检查，compute capability 低于 7.5 时快速失败。

### 2026-05-15

- 新增 PyNv/8K 分阶段性能工具和 decode、encode、mux、端到端 passthrough probe。
- 新增 PyNv threaded decoder 实验、slot ownership 处理和 encode input lifetime 保护。
- 新增 PyNv 路径的 FP16 RVM benchmark 支持，以及 TensorRT/CUDA provider 诊断。
- 新增 FPS cap 发现报告，并为受污染的性能 summary 添加废止标记。

### 2026-05-14

- 新增 frozen build 下的 CuPy/CUDA 打包依赖处理。
- 改进 Windows 打包版本的运行时 CUDA DLL 加载行为。
- 新增 CUDA、CuPy、ONNX Runtime provider 可用性的打包诊断。

### 2026-05-13

- **v0.1.0-alpha.1正式发布小范围公测**
- 新增首次 GPU warmup 长等待场景的启动遮罩。
- 新增本地状态端口的启动状态轮询。
- 新增启动遮罩中的一键复制诊断报告功能。
- 新增当状态轮询不可用但主服务已启动时的遮罩关闭 fallback。
- 新增 GPU 初始化失败或不支持时的结构化启动失败提示。

### 2026-05-12

- 新增 PyInstaller 打包中 Qt/ICU DLL 冲突修复。
- 新增打包防御检查，减少重复或不兼容 DLL 被收集。
- 新增打包运行时依赖问题的构建期诊断。

### 2026-05-11

- 新增第一版 PySide6 桌面 UI。
- 新增实时服务器控制、快捷配置、版本显示、状态栏和日志侧栏。
- 新增 UI 中的绿幕/alpha 模式开关。
- 新增字幕设置和预览 UI。
- 新增多视频目录配置。
- 新增应用图标、共享 UI 样式和语言选择简化。
- 新增服务器 alpha 直通条目和绿幕/alpha 双模式列表。
- 新增 alpha fisheye 输出、alpha block 布局修正、透明零 alpha 覆盖行为，以及 alpha 输出的音频后混流。

### 2026-05-10

- 新增离线 RVM passthrough 生成。
- 新增离线 bitrate 控制和 VR 质量默认值。
- 新增字幕叠加 Phase 1，包括字幕匹配、颜色处理和源路径诊断。
- 新增 MatAnyone2 ONNX 导出工具和第一版离线 runtime 接入。
- 新增 SAM3/MatAnyone2 实验性分割流程，包括低显存模式和 active segment planning。
- 新增 live 播放的 AAC cache 和音频标准化工作。
- 新增重复播放器请求的 live-session cache 改进。

### 2026-05-09

- 新增 MoonVR/VLC、Skybox/libmpv、nPlayer/OPlayer 类客户端和默认客户端的播放器特定 live passthrough 处理。
- 新增 live passthrough active-slot ownership 和 preemption 规则。
- 新增 alpha 边缘清理控制和默认绿色合成背景。
- 新增 PyNv 生产路径音频 mux 接入和 AAC/MPEG-TS 时间戳处理。
- 新增 Main10/P010/P016 兼容性实验和 PyNv passthrough 转换路径。
- 新增 server log 截断和启动日志处理改进。

### 2026-05-08

- 新增 PyNv encoder、mux、decode-to-encode 和 GPU matting probe。
- 新增 PyNv production stream 初始集成。
- 新增 passthrough pseudo-VOD byte seek 集成。
- 新增 passthrough live mode 和 HEVC live 支持。
- 新增高码率 8K 源的 adaptive live FPS 行为。
- 新增 DLNA 物理目录浏览、缩略图、live-only 列表调整、live 分段目录和短视频直接播放行为。
- 新增 `config.py` 中的集中环境变量读取和配置文档扩展。
- 新增 GPU runtime cache 和 ORT CUDA 冷启动支持工具。
- 新增 live request header dump 和 stalled-client cleanup watchdog。

### 2026-05-07

- 新增 `PT_PASSTHROUGH_MAX_FPS` 输出 FPS cap 配置。
- 新增 `PT_ALPHA_STRIDE` 和 `PT_ALPHA_MODE=reuse` alpha stride 复用。
- 新增 RVM 模型选择和 RVM MobileNetV3 支持。
- 新增 RVM 实验用 CUDA IOBinding 和 shared-stream 支持。
- 新增 GPU NV12 preprocess 和 fused NV12-to-NV12 绿幕合成 kernel。
- 新增 NVENC 调参环境变量，用于 A/B 测试。
- 新增 PyNvVideoCodec 依赖和初始 PyNv decode/matting bridge 代码。
- 新增 CUDA、FFmpeg hwaccel、ONNX Runtime 和 PyNv 可用性的 GPU/video probe 工具。

### 2026-05-06

- 新增 CUDA decoder 诊断和 FFmpeg 硬件解码候选选择。
- 新增从 decoder 输出元数据传递输出 FPS/尺寸。
- 新增 matting profiling，拆分 preprocess、ORT、alpha resize 和 composite timing。
- 新增优化后的绿幕合成路径，避免 full-frame green background 分配。
- 新增 matting input size、decode max side 和 model path 的模型/吞吐控制。
- 新增初始 DLNA time-seek 元数据和 passthrough HEAD 支持。
- 新增 `PT_CONTAINER`，支持 MP4 和 MPEG-TS passthrough 输出。
