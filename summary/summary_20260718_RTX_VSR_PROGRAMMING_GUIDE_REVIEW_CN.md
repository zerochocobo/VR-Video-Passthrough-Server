# NVIDIA RTX Video SDK v1.1.0 Programming Guide 复核

日期：2026-07-18  
文档：`reference/RTX_Video_SDK_v1.1.0/doc/NVIDIA_RTX_Video_SDK_Programming_Guide.pdf`

## 最终结论

这份 SDK Programming Guide **没有声明 VSR 输入解析度必须为 360p–1440p，也没有声明只支持 2D 视频**。

360p–1440p 可能来自 NVIDIA 面向浏览器/播放器的 GeForce RTX Video Super Resolution 产品说明，但不能直接等同于 RTX Video SDK v1.1.0 API 的硬限制。本项目可以把 360p–1440p 作为首版保守产品策略，却必须注明它是项目门控，不是此 PDF 明文规定。

同理，PDF 没有 VR、SBS、OU、equirectangular 或 per-eye 支持声明，也没有明确说 VR 禁止。对 4K VR 的正确结论是：**官方指南未覆盖、未保证，首版不支持；不能说 SDK 官方明确禁止。**

## PDF 原文要点

### 第 46 页：VSR 的定义

指南描述：VSR 输入一个 RGB SDR surface，对其进行 upscale、sharpen 和 deblock，输出 RGB surface。

这说明：

- VSR 的直接输入/输出是 RGB GPU surface。
- 它处理的是二维纹理矩形，但没有“2D 视频类型”字段。
- NV12/P010 视频需要先经 Video Processor 转换为 RGB；sample 正是这样实现。

### 第 46–48 页：受支持参数

VSR evaluate 参数包括：

- input/output GPU resource
- input/output subrect base
- input/output subrect size
- quality level 0–4

没有：

- 固定解析度枚举
- 360p/1440p 上下限
- VR layout、投影方式或左右眼参数
- 固定缩放倍数

### 第 10 页附近：官方运行示例

指南给出：

- 480p 输入放大到 1920×1080：`VSRDemo.exe ... -size 1920 1080`
- 1080p 输入同时应用 VSR 和 TrueHDR
- sample 参数允许 `-size w h` 指定输出尺寸

这些例子证明文档覆盖 480p、720p、1080p，但不能反向证明 360p 是下限或 1440p 是上限。

### 第 9 页附近：高解析度样片提示

指南建议在准备 sample 原始 YUV/RGB 文件时，用 FFmpeg downscale 高解析度视频。该段落是 sample 文件准备建议，没有写成 SDK evaluate 的最大输入限制。

### 第 54 页：GPU/驱动要求

指南明确写出：

- Turing RTX 20xx 或更新 GPU
- 550.58 或更新驱动
- CUDA API 路径的构建需要 CUDA Toolkit 12.8 或更新版本

这些是 PDF 中明确可作为 capability/preflight 依据的要求。

## 对项目计划的正确处理

1. 首版正式功能仍只开放普通 2D 视频，VR/4K VR 不进入产品支持范围，理由是未获官方覆盖且风险高，而不是宣称 SDK 明确禁止。
2. 360p–1440p 可作为首版默认门控，配置保持可调整；日志注明 `project_resolution_policy`，不要写成 `sdk_unsupported_resolution`。
3. native POC 必须实测 360p、480p、720p、1080p、1440p、2160p，以及非 16:9 输入，记录 create/evaluate 的返回码、显存和性能。
4. 最终硬限制应由 SDK runtime 返回值、NVIDIA 后续正式说明和 POC 数据确定。
5. 输出标记保持 `[SuperRes]`。

## 推荐表述

> RTX Video SDK v1.1.0 Programming Guide 没有规定 360p–1440p 的硬性输入范围，也没有 VR 支持声明。PTMediaServer 首版将 RTX VSR 限定为 360p–1440p 普通 2D 视频，这是保守的产品策略；4K、VR 和其他边界输入待 native POC 验证后再决定是否开放。
