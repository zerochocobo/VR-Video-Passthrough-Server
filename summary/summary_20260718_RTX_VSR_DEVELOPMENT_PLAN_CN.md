# NVIDIA RTX VSR 高清功能开发计划（中文）

日期：2026-07-18  
状态：开发前计划，尚未实现功能代码

## 1. 目标与范围

针对普通 2D 视频接入 NVIDIA RTX Video Super Resolution（VSR）：

- 普通 2D 视频：仅当输入解析度在 360p–1440p 范围内时支持实时高清和离线高清，输出名称带 `[SuperRes]` 前缀。
- 4K VR、单眼裁剪 VR、SBS/OU/equirectangular VR：不接入 RTX VSR，继续使用现有路径。

默认行为应保持现有链路不变；不满足 RTX VSR 条件时自动回退到现有路径，并明确记录原因。

## 2. 调研结论与技术边界

- SDK 位置为 `reference/RTX_Video_SDK_v1.1.0`，后续不得从业务代码直接引用该目录。
- SDK 的 VSR 质量级别为 Bicubic(0)、Low(1)、Medium(2)、High(3)、Ultra(4)；实现对外完整暴露 0–4，其中 0 为明确的 Bicubic 基线。
- SDK Programming Guide 未声明 360p–1440p 的硬限制。首版按项目保守策略门控为 360p–1440p；低于或高于该范围先走现有路径，待 native POC 验证后再决定是否开放。
- SDK 提供 DX11、DX12、CUDA 接口，但 Python/PyNvVideoCodec 当前没有现成 NGX VSR 绑定。需要增加 Windows x64 native bridge（优先 CUDA 或 DX11，依据帧资源互操作验证结果选择）。
- VSR 是空间超分，不是编码器；仍需沿用现有 NVENC/FFmpeg 封装、音频复用、时间戳和播放器兼容逻辑。
- 普通 2D 的目标分辨率必须由配置矩阵和运行时 capability 决定，不能无条件放大所有视频。

## 3. 计划新增的配置项

建议统一使用 `PT_RTX_VSR_*` 环境变量，并在 `config.py` 提供解析、默认值和注释：

| 配置项 | 作用 | 建议默认 |
|---|---|---|
| `PT_RTX_VSR_ENABLE` | 总开关 | `0`（首版验证后再评估改为 `1`） |
| `PT_RTX_VSR_REALTIME_ENABLE` | 实时链路开关 | `0` |
| `PT_RTX_VSR_OFFLINE_ENABLE` | 离线链路开关 | `0` |
| `PT_RTX_VSR_QUALITY` | 质量级别 0–4 | `2`（Medium） |
| `PT_RTX_VSR_INPUT_MIN_HEIGHT` | 项目首版输入解析度下限，不代表 SDK 硬限制 | `360` |
| `PT_RTX_VSR_INPUT_MAX_HEIGHT` | 项目首版输入解析度上限，不代表 SDK 硬限制 | `1440` |
| `PT_RTX_VSR_MAX_OUTPUT_PIXELS` | 输出像素上限，防止 4K/8K 显存爆炸 | 按 GPU/实测设定 |
| `PT_RTX_VSR_2D_TARGET` | 普通 2D 目标档位；受官方能力范围限制 | `auto` |
| `PT_RTX_VSR_SOURCE_POLICY` | `2d` 或 `auto`；VR 自动禁用 | `auto` |
| `PT_RTX_VSR_FALLBACK` | 不可用时 `passthrough` 或 `error` | `passthrough` |
| `PT_RTX_VSR_FRAME_QUEUE` | 实时 GPU 队列深度 | `2` |
| `PT_RTX_VSR_MAX_LATENCY_MS` | 实时延迟预算，超限时降级/跳过 | `100` |
| `PT_RTX_VSR_PREFIX` | 文件名标记 | `[SuperRes]` |
| `PT_RTX_VSR_SDK_DIR` | 打包后 SDK runtime 相对目录（仅诊断/覆盖） | `rtx_video_sdk` |

还要定义输入分类规则：2:1/VR 标记视为 VR；普通 2D 仍须满足 360p–1440p 输入范围。分类、目标分辨率和实际回退原因写入诊断日志。

## 4. 实现阶段

### 阶段 A：SDK 资产与 native bridge

1. 从 `reference/RTX_Video_SDK_v1.1.0` 复制必要的头文件、x64 release DLL（至少 `nvngx_vsr.dll`，按 SDK 依赖补齐）到 `models/rtx_vsr/` 专用目录；`native/` 保存桥接源码、头文件和链接库，`runtime/` 保存运行 DLL、许可证和版本说明。
2. 在 `models/rtx_vsr/native/` 维护 native bridge，提供稳定的 C API：初始化、能力探测、输入帧格式、目标尺寸、质量级别、处理、释放、错误码。开发/发布机预编译 DLL，终端用户安装 exe 时不编译 C++。
3. 优先验证 CUDA/CuPy 内存互操作；若 PyNv 资源无法安全共享，改用 DX11 texture bridge，禁止在实时热路径中发生不受控 CPU 往返。
4. 增加 driver、GPU 架构、`nvngx_vsr.dll`、NGX feature availability 预检。

### 阶段 B：实时高清

1. 在 `pipeline/pynv_stream.py` 中增加可选 VSR stage，位置为解码后、现有合成/编码前；尽量保持帧驻留 GPU。
2. 对 VR 标记、2:1/equirectangular 比例和 VR 播放模式执行硬门控；普通 2D 才进入目标尺寸、显存和吞吐门控，实时超预算时按 `fallback` 回到原始分辨率路径。
3. 在 `utils/vr_naming.py` 增加 `[SuperRes]` 命名函数，并同步 DLNA/HTTP display title，避免重复添加前缀。
4. 增加 UI/运行时开关、当前状态、质量级别和回退原因展示；兼容已有 live mode（green、2D→3D、RM 等），首版先限定普通视频/指定 VR 模式。

### 阶段 C：离线高清

1. 在 `offline/convert.py` 或独立 `offline/rtx_vsr.py` 增加单文件、批量、时间段处理参数。
2. 使用 GPU 解码/桥接/ VSR / NVENC，音频直接复用或按现有离线策略封装；避免把整段视频解码为 CPU 图片序列。
3. 输出命名统一为 `[SuperRes]<原文件名>`，与重复运行和 `--skip-existing` 规则兼容。
4. UI 增加 RTX VSR 引擎、目标分辨率、质量、回退策略和进度/错误显示。

### 阶段 D：打包与发布

1. 修改 `build_exe.py`、两个 `.spec` 和必要 runtime hook，把预编译 bridge 与 `models/rtx_vsr/runtime`（`nvngx_vsr.dll`、CUDA runtime、许可证）复制到最终 `dist\...\models\rtx_vsr\runtime`。
2. 构建时只复制 Windows x64 release 资产，不复制 SDK samples、dev/debug DLL 或 ARM64 资产。
3. 增加打包后文件存在性、DLL 加载、bridge 启动和 VSR capability smoke check；运行时路径使用 `sys._MEIPASS`/应用根目录解析。
4. 明确 NVIDIA SDK 许可证、驱动最低版本和 RTX GPU 要求，写入发布说明。

## 5. 测试与验收

- 单元测试：配置解析、360p–1440p 输入门控、输入分类、目标尺寸、`[SuperRes]` 命名、重复前缀、fallback 决策。
- bridge 测试：DLL 缺失、驱动过旧、非 RTX GPU、质量 0–4、NV12/P010/RGBA 格式、资源尺寸错误、初始化/释放循环。
- 实时测试：360p、720p、1080p、1440p 2D，以及低于 360p、高于 1440p 和 VR 拒绝用例，验证 FPS、端到端延迟、显存峰值、音画同步、客户端兼容性。
- 离线测试：短片、长片、无音频、多音轨、10-bit、批处理、时间段、断点/跳过已有输出。
- 打包测试：全新机器仅使用 `dist` 目录运行，确认不读取 `reference`，缺失 VSR 资产时主程序仍可启动并回退。

验收门槛：功能关闭时行为/性能回归为零；功能开启时输入在 360p–1440p、输出达到目标尺寸且带 `[SuperRes]`；所有失败场景可解释、可回退、无进程/显存泄漏。

## 6. 风险与待确认事项

- NGX/VSR 的 Python GPU 资源互操作是最大技术风险，需先做最小 native proof-of-concept。
- RTX VSR 对输入/输出格式、尺寸比例、驱动版本和显存有约束，必须以 capability 查询和实测为准。
- native POC 必须实测 360p、480p、720p、1080p、1440p、2160p 和非 16:9 输入，区分项目策略与 SDK 实际边界。
- SDK DLL 的再分发许可、NGX feature 初始化所需的 app id/模型下载行为，需要在 POC 阶段核验。

## 7. 建议执行顺序

先完成“复制 release SDK + native bridge capability/单帧测试”，再接入实时链路，最后复用 bridge 接入离线链路和打包流程。只有 POC 能稳定处理 GPU 帧后，才进入 UI 和批处理扩展。
