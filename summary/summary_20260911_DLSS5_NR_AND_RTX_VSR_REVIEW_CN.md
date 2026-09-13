# DLSS 5 Visual Enhancer 调研 + 本项目 RTX 超分改进评估（中文）

日期：2026-09-11
状态：调研报告，未改动任何功能代码
参考源码：`reference/dlss5-visual-enhancer-main`（GitHub: Merserk/dlss5-visual-enhancer）

---

## 1. 结论先行

1. **在 RTX VSR 这一块，我们的实现比该项目更好，不需要照抄。** 它的 Upscale（VSR/TrueHDR）走的是「独立 worker 进程 + stdin/stdout 传整帧像素」的 CPU 往返路径（`src/upscale/video/native.py`），我们是 NVDEC → CuPy → NGX CUDA deviceptr → NVENC 的全 GPU 常驻路径。
2. **它真正有价值的部分是 DLSS 5 Neural Rendering（NGX feature 18）的调用架构**，以及它已经启用而我们没启用的 **NGX TrueHDR**、**VSR 1x 原生增强**。
3. **DLSS 5 NR 目前没有官方 SDK**：需要 `nvngx_dlssnr.dll`（不可再分发、用户自备、约 158MB）、一个自研 D3D12/NGX 桥 DLL、以及一个「调用方校验绕过 shim」。该项目的桥 DLL（`neuroframe_engine.dll` / `neuroframe_caller.dll`）**并未开源**，只随 Release 二进制分发。
4. **NR 不做放大**，它是同分辨率的细节/色调/皮肤重建。要提分辨率仍然要 VSR。两者是叠加成本，不是二选一。
5. **对我们的场景，实时 8K VR 上 NR 没有可行性**；现实落点是「离线 2D/VR 增强」和「实时 2D ≤1080p」。
6. **我们当前 8K VR 实时超分的 23–24 FPS 是 NGX 计算瓶颈，工程优化救不回来**（CHANGELOG 已记录实测）。真正能改的是画质/功能维度，以及增加中间分辨率档位。

---

## 2. 该项目技术拆解

### 2.1 它的四条功能线

| 功能 | 底层 | 实现方式 |
|---|---|---|
| Neural Rendering（主打） | NGX **feature 18**，`nvngx_dlssnr.dll` | 自研闭源桥 `neuroframe_engine.dll`，ctypes 调用 |
| Frame Interpolation | NGX **feature 11**（DLSSG） | 独立 worker 进程 `DLSSG_WORKER`，二进制流传帧 |
| Upscale（VSR / TrueHDR） | NGX feature 16 / TrueHDR，RTX Video SDK | 独立 worker 进程，**host 像素管道往返** |
| Live | ffmpeg/PyAV 解码 → NR → HLS 分段 → MPV 播放 | 1/2/4 秒分段 + 2–30 秒播放缓冲 |

### 2.2 NR 桥的 ABI（`src/core/neural_bridge.py`，值得借鉴的部分）

- 帧描述符 `FrameDescriptorV1`：memory_type（host / CUDA / none）、pixel_format（**RGBA8 / NV12 / P010**）、planes[3]、strides[3]、color_matrix、color_range、rotation、timestamp。
  → 说明它的桥**可以直接吃 NVDEC 的 NV12/P010 显存指针**，和我们 VSR 桥的思路一致。
- 渲染参数演进到 ABI v6：style / intensity / tone / structure / skin / automask / **reset** / color_strength / tone_preservation / mask / face_skin_protection / grain_preservation / **nr_passes** / **shimmer_suppression** / **prefer_nvof**。
- 入口分三类：`dlss5nr_process*`（host）、`dlss5nr_process_cuda*`（CUDA 进 / CUDA 出）、`dlss5nr_process_frame_v*`（帧描述符）。
- NGX 是**进程生命周期状态**：源码注释明确写了「正常关闭时绝不调用 `NVSDK_NGX_D3D12_Shutdown`、绝不卸载驱动模块，两者在 feature-18 成功 evaluate 之后都会卡死」。这条踩坑经验对我们将来做任何 NGX 桥都适用（我们 VSR 桥目前是有 `shutdown` 的，切换 CUDA context 时会调用——如果以后接 NR 要特别小心）。
- 有看门狗（45s）、结构化 JSON 帧日志、`verify_feature_18` 证据收集。

### 2.3 它的时间稳定做法（比我们弱）

`TemporalGuideGenerator`（`neural_rendering/video/guides.py`）只是：缩到 640 宽 → 灰度 → `absdiff` → 均值 > 0.24 判定场景切换 → 给 NR 传 `reset`。全部在 CPU 用 cv2 做。
真正的抖动抑制（shimmer suppression / NVOF）在闭源桥内部。
→ 我们 VVPS 那套（base/detail 分离 + 运动补偿 + 证据门控 + 场景切换 reset）在思路上更完整，不用回头学它。唯一值得抄的是**「场景切换必须给神经网络送 reset」这个契约本身**。

---

## 3. DLSS 5 NR（feature 18）的真实门槛

来自社区项目与公开报道的核实结果：

| 项目 | 结论 |
|---|---|
| 接口状态 | **未公开、pre-release**，无官方 SDK 文档，NVIDIA 不支持 |
| 运行时 DLL | `nvngx_dlssnr.dll`（~158MB，v310.8+）+ 驱动内 `_nvngx.dll` |
| 分发 | **无再分发许可**，所有社区项目都要求用户自行获取 DLL |
| 调用方校验 | NR DLL 拒绝未知调用方：`0xBAD00002`。社区项目靠一个伪装成 `nvngx.dll` 的 shim 通过校验 |
| GPU 门槛 | 官方 DLL **硬锁 RTX 50 系**；40 系 create 返回 `0xBAD00001`（社区有报告称部分场景可用，不可依赖）。驱动 ≥ 616.56 |
| 像素路径 | NV12 → **RGBA16F** → feature 18 → RGBA8，D3D12 纹理 |
| 是否放大 | **否**，同分辨率增强；放大仍需 VSR |
| 时域 | OBS 插件明确「NR runs per-frame (no temporal flow yet)」，逐帧无时域一致性；闭源桥的 shimmer suppression 是自己加的后处理 |
| 成本量级 | 游戏侧公开数据：RTX 5060 Ti @1080p 开 DLSS 5 从 177 → 62 FPS（含 SR/FG，非纯视频路径）。**纯视频 NR 的 ms/帧必须实测，不能引用游戏数字** |

### 对我们的直接含义

- **不能打包进安装包**。我们现在能打包 `nvngx_vsr.dll` 是因为 RTX Video SDK 明确给了再分发许可；`nvngx_dlssnr.dll` 没有。
- **caller-validation shim 是绕过厂商访问控制**。我的建议是不要把它做进产品：一是许可风险，二是 NVIDIA 随时可以在驱动更新里封掉，我们会背上一个随时碎掉的功能。要推进的正路是向 NVIDIA 申请 DLSS NR SDK / RTX Video SDK 后续版本的访问。
- 真要做，只能做成「**默认关闭 + 用户自备 DLL + 明确免责 + 离线优先**」的实验特性。

---

## 4. 我们当前 RTX 超分的现状盘点

代码位置：`utils/rtx_vsr.py`、`pipeline/pynv_stream.py:2440-2910`、`offline/rtx_vsr_pynv.py`、`models/rtx_vsr/native/`

已经做对的：
- NGX CUDA deviceptr 路径，帧全程不落 CPU；NV12→RGB 与左右眼 RGBA 准备已融合成一个 kernel。
- SBS VR 分眼超分 → 8192×4096 重组；NVENC 输入用 ring buffer 防异步持有。
- 门控清晰（分辨率/位深/VR/目标尺寸），不可用时明确拒绝而非静默降级。
- 打包已含 bridge + `nvngx_vsr.dll` + CUDA runtime + 许可证。

已知硬瓶颈（CHANGELOG 已实测记录）：
- **RTX 5060 Ti 上 8K VR Ultra ≈ 23–24 FPS，NGX 计算受限**；整帧 evaluate、去同步、异步编码、双实例均无法安全达到 60 FPS。

### 现存的可改进点（按性价比排序）

#### A. 启用 NGX TrueHDR —— 收益最大，工作量明确

- 现状：native 桥 `rtx_video_api_cuda_impl.cpp` **已完整实现 TrueHDR**（create/evaluate/VSR→THDR 串联/中间纹理），但 Python 侧 `utils/rtx_vsr.py:184` 调用 `rtx_video_api_cuda_create(ctx, stream, gpu, 0, 1)` —— **THDREnable 恒为 0**，`_ThdrSetting` 也一直传全零。
- 现在的「HDR 观感」是 `pipeline/hdr_look.py` 里我们自己写的 SDR 内调色 kernel（对比度/阴影/高光/饱和度），不是真 HDR。
- `nvngx_truehdr.dll` 在 `reference/RTX_Video_SDK_v1.1.0/bin/Windows/x64/rel/` 里存在，**与 VSR 同一份许可，可再分发**，但没有复制进 `models/rtx_vsr/runtime/`，打包脚本也没引用。
- 要做的事：复制 DLL → Python 侧开 THDR 开关与参数（contrast / saturation / middle gray / max luminance）→ THDR 输出是 **10bit ABGR10**，需要新增 `abgr10_to_p010` kernel → NVENC HEVC Main10 + HDR10 元数据 → 离线封装与 DLNA 声明。
- 建议：**先只上离线**（实时再叠一次 NGX 会进一步压低本就 23 FPS 的 VR 路径）。

#### B. 开放 VSR 1x「原生增强」 —— 新功能，低成本

- 现状：`source_block_reason()` 在 `source_exceeds_target_resolution()` 为真时直接拒绝，所以 4K 源根本进不了 VSR。
- SDK 支持 1x（质量级别照常生效），用于不放大的去压缩伪影/锐化。该项目的 Upscale 页面就明确提供 1×。
- 收益：4K 2D、4K VR 源可以做「同分辨率画质修复」，成本远低于放大到 8K，实时可行性也高得多。
- 要做的事：目标尺寸计算允许 `out == in`；命名/DLNA 需要区分 `[SuperRes]` 与「原生增强」，避免用户以为分辨率变了。

#### C. 增加中间目标分辨率档位

> **2026-09-11 实测更正**：本节下面「帧率随输出像素线性下降、6K 能到 40 FPS」的推断**是错的**。实测表明 NGX VSR 的耗时由**输入**分辨率主导，几乎与输出分辨率无关：同一 4K VR 源（每眼 1920×1920）下，8K 输出 17.2 ms/眼、6K 输出 16.1 ms/眼，只差 7%；把输入换成 2K VR（每眼 960×960）后，同样的 8K 输出只要 6.2 ms/眼。因此 6K 档的价值是**输出体积和头显解码负担**（像素数为 8K 的 56%），不是帧率。详见 [summary_20260911_VSR_NATIVE_AND_6K_CN.md](summary_20260911_VSR_NATIVE_AND_6K_CN.md)。

- 现状：`ui/pages/superres_page.py:74` 只有 2160 与 4096 两档；VR 走 4096 就是 8192×4096（相对 4K VR 源约 4.5× 像素）。
- 既然是 compute-bound，帧率基本随输出像素线性掉。加 6144×3072（约 2.5× 像素）一档，理论上能把 23 FPS 拉到 40 FPS 附近，是「实时 VR 超分」从不可用变可用的关键档。
- 需要同时确认头显端对 6K HEVC 的解码能力，以及 `utils/rtx_vsr.py:target_resolution()` / 命名规则（现在只有 `_2K/_4K/_8K`）。

#### D. 10-bit 源不再硬失败

- 现状：`pipeline/pynv_stream.py:2878` 遇到 `GpuP016Frame` 直接 `raise`；门控里 `is_10bit` 也直接拒。
- 至少应在实时路径上给出明确的降级（P016 → 8bit RGBA 后超分）而不是抛错中断会话；配合 A 项做完后可以走 P010 全 10bit 链路。

#### E. 热路径微优化 —— 有收益但不要指望帧率

- `process_cupy_rgba()` 每帧都调 `initialize_cupy()`（含 `Device().use()`、`runtime.free(0)`、`ctxGetCurrent`、取 stream）、每帧 `cp.empty` 新建输出、每次 evaluate 前后各一次 `stream.synchronize()`。分眼时以上全部 ×2。
- 之后还有 `split_output[:, :w] = ...` 的整帧拷贝、`hdr_look` 的一趟读写、`rgba_to_nv12` 的一趟读写。8K 下每帧多搬约 400MB，5060 Ti 上折合约 0.9ms/帧 —— 相对 43ms 的 NGX 计算只占 2%。
- 所以：**做，但理由是显存占用、分配抖动和代码清晰度，不是帧率**。建议会话级预分配左右眼输出缓冲，并把 hdr_look 融进 `rgba_to_nv12`（直接从两个眼缓冲拼出全宽 NV12）。

#### F. 组合能力

VSR 目前只服务于独立的 `superres` 输出模式，和 RM 去码、2D→3D、Alpha 等模式互斥。是否要支持「RM + SuperRes」这类串联，取决于产品优先级，但成本是两个模型叠加，8K 下不现实，2D 下可行。

---

## 5. 引入 DLSS 5 NR 的路线建议

### 前提判断

- **实时 8K VR + NR：不可行。** VSR 本身已经吃满 43ms/帧，NR 是同量级甚至更贵的第二次神经网络评估，且要额外做 RGBA16F 转换。
- **实时 2D ≤1080p：有可能**，需要实测。
- **离线（2D 与 VR）：这是唯一稳妥的落点**，和我们已有的 `offline/rtx_vsr_pynv.py` 链路形态一致。

### 分阶段

**阶段 0 —— 合规与能力确认（先做，不写代码）**
1. 确认 `nvngx_dlssnr.dll` 的获取方式与许可状态；确认我们不打算、也不能随安装包分发。
2. 确认目标 GPU：5060 Ti（sm120）在锁内，但要接受「40 系及以下用户拿不到这个功能」。
3. 明确产品定位：默认关闭的实验特性，DLL 由用户放到指定目录，缺失时功能卡片直接隐藏。

**阶段 1 —— 离线 PoC（一次性，不进产品代码）**
- 目标只有一个数字：**NR 在 5060 Ti 上 1080p / 4K / 4096×4096（单眼）各是多少 ms/帧**。
- 形态：独立小程序或独立进程 worker，D3D12 + NGX，RGBA16F 进 / RGBA8 出。
- 同时观察：连续帧是否闪烁（无时域一致性）、场景切换 reset 的必要性、皮肤/纹理的实际增益是否值得这个代价。
- 注意 `neural_bridge.py` 的那条经验：**不要在退出时调 NGX Shutdown**。

**阶段 2 —— 若 PoC 数字可接受，接离线链路**
- 复用现有 NVDEC → CuPy 骨架，新增 `nv12 → rgba16f` 与 `rgba8 → nv12` kernel。
- VR 必须**分眼**处理（和 VSR 一样），并注意 equirect 上下极点的畸变会被细节增强放大。
- 与 VSR 的顺序建议：**先 VSR 放大，再 NR 增强**（NR 在目标分辨率上工作，效果与该项目的设计一致）；但成本是两次神经网络，需要在 PoC 阶段就量出叠加后的总时间。
- 场景切换检测复用我们 VVPS 已有的能力，不必再写一套 cv2 版本。

**阶段 3 —— 实时（只对 2D，且严格门控）**
- 上限 1080p（必要时 1440p），超过直接拒绝。
- 复用现有的预检/超时/隔离子进程机制，失败一律回退 passthrough。

### 我不建议做的

- 不建议实现 caller-validation shim 并随产品分发。这是绕过厂商的调用方校验，许可风险与稳定性风险都由我们承担，而且一次驱动更新就可能失效。如果这个功能对产品足够重要，正路是去拿官方 SDK 授权。

---

## 6. 建议的执行顺序

1. **TrueHDR 离线链路**（A）—— 资产就在 SDK 里，桥代码已经写好，是当前投入产出比最高的一项。
2. **VSR 1x 原生增强**（B）—— 小改动，直接扩大可处理片源范围。
3. **6K 中间档**（C）—— 让实时 VR 超分真正可用。
4. **10-bit 不再硬失败**（D）+ **热路径预分配**（E）—— 稳定性清理。
5. **DLSS 5 NR 阶段 0/1**（PoC 拿数字）—— 在上面四项之后再启动，避免把工期压在一个没有官方支持的接口上。

---

## 参考链接

- [Merserk/dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer)
- [Zonnery/dlss5-nr-player](https://github.com/Zonnery/dlss5-nr-player)
- [Saganaki22/obs-dlss5-nr](https://github.com/Saganaki22/obs-dlss5-nr)
- [NIGos/dlss5-bridge](https://github.com/NIGos/dlss5-bridge)
- [TechPowerUp: NVIDIA DLSS 5 Technical Preview](https://www.techpowerup.com/review/nvidia-dlss-5-technical-preview/6.html)
- [NVIDIA: DLSS 5 3D-Guided Neural Rendering](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/)
