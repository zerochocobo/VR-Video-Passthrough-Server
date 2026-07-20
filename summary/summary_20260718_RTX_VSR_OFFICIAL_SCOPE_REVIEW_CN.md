# RTX Video SDK / VSR 官方范围复核（中文）

> 说明：本文件关于“360p–1440p”的结论已由 `summary_20260718_RTX_VSR_PROGRAMMING_GUIDE_REVIEW_CN.md` 修正；该范围是首版项目策略，不是 Programming Guide 明文硬限制。

日期：2026-07-18

## 结论

本项目应将 RTX VSR 的正式支持范围收窄为“输入解析度 360p–1440p 的普通 2D 视频帧”。低于 360p 或高于 1440p 不进入 VSR；因此 4K 2D、4K VR 全帧不支持，把 VR 视频裁成高解析度单眼也不应作为支持方案。

更严谨地说：

- **不是 API 语义上的 2D-only**：NGX VSR 接收输入资源、输入矩形、输出矩形和质量级别，不接收“2D/VR”类型字段。
- **是产品支持范围上的普通视频策略**：SDK 自带 VSR sample 以 YUV/RGB 视频文件为输入，通过常规视频处理器转换后调用 VSR；没有 VR 投影、双眼、球面坐标、SBS/OU 解包或逐眼处理流程。
- **4K VR 全帧明确不适合作为 VSR 输入**：全帧通常为 2:1 或更宽的超大画布，且常见宽度已达到 4K/8K；这不符合“低分辨率视频超分到更高显示分辨率”的目标。
- **单眼也不应承诺支持**：即使单眼裁剪后的像素数看似接近普通视频，SDK 文档和 sample 没有对 VR 单眼/球面内容给出支持声明或专用处理，无法形成官方支持依据；单眼裁剪还会改变项目的 VR 播放几何和拼接流程。

因此，本项目应描述为：**RTX VSR 仅用于输入 360p–1440p 的普通 2D 视频实时/离线超分；VR、低于 360p 和高于 1440p 的输入走现有路径，输出标记为 `[SuperRes]`。**

## SDK 证据

1. `include/nvsdk_ngx_defs_vsr.h` 只定义质量级别 `Bicubic/Low/Medium/High/Ultra` 以及 availability、driver、feature init 等参数，没有 VR/双眼/投影类型参数。
2. `include/nvsdk_ngx_helpers_vsr.h` 的 DX11、DX12、CUDA evaluate 参数只有输入/输出资源、输入/输出矩形和质量级别；这证明它是通用帧/矩形处理接口，但不构成 VR 支持承诺。
3. `samples/ReadMe.md` 将 VSRDemo 描述为播放 YUV/RGB data file，TrueHDR_VSR sample 也是常规视频帧处理；命令行只提供输入文件、显示尺寸、帧率、质量级别等选项，没有 VR layout、SBS/OU、per-eye 或 equirectangular 选项。
4. VSR sample 的 DX11/DX12 路径先用 Video Processor 将 YUV 转成 NGX 输入，再按普通 source/output rectangle 执行 VSR；没有球面重映射或眼睛分离。

SDK 自带 Programming Guide 没有声明 360p–1440p 硬限制。计划可按该范围建立首版项目门控，但必须在 POC 中通过运行时返回值和边界样片验证实际尺寸能力。

## 对原开发计划的修订

- 删除“4K VR 实时 VSR”作为目标和配置矩阵。
- `PT_RTX_VSR_SOURCE_POLICY` 只保留 `2d`/`auto`；`auto` 遇到 VR 标记、2:1/半球面比例或 VR 模式时必须禁用并记录 `unsupported_vr_source`。
- 删除 `PT_RTX_VSR_4K_VR_TARGET`。
- `PT_RTX_VSR_2D_TARGET` 保留，但目标分辨率必须受 Programming Guide 与运行时 capability 限制，不能默认写死 4K/8K。
- 新增 360p 下限和 1440p 上限配置；超出范围记录 `project_resolution_policy`，不能误写成 SDK 官方不支持。
- `[SuperRes]` 只用于范围内的普通 2D RTX VSR 输出；其他输出不添加该标记。

## 最终建议

下一步先制作普通 2D 720p/1080p/1440p 的 VSR native POC，验证实际输入/输出分辨率、质量级别、驱动和显存限制；同时加入 VR 源硬门控。没有必要为 4K VR 或单眼建立 VSR bridge 分支。
