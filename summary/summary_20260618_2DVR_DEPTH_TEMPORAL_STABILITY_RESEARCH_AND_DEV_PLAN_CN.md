# 2DVR 深度时间一致性（消抖动）调研 + 详细开发方案 — 2026-06-18

## 0. 背景与问题

当前 2D→3D 链路（逐帧独立）：

```
ffmpeg 解码 → DA3 ONNX（单帧深度）→ _normalize_near 归一化 → 立体平移(视差) → 编码
```

现象：视差持续轻微变化，画面抖动，观感差。即使做了额外处理仍无法控制很多画面的抖动。

### 抖动的两个独立来源（叠加）

**来源 1：DA3 ONNX 是单帧推理，没有时间一致性。**
`offline/da3_depth.py:247` `predict_batch` 把每帧当独立图片送进 `da3_base.onnx`。单目深度网络逐帧独立估计，绝对尺度/相对深度逐帧漂移。这是根本原因。

**来源 2：渲染端逐帧百分位归一化，把漂移放大了。**
`offline/two_dvr_render.py:80`：

```python
lo, hi = np.percentile(sample, [5.0, 95.0])   # 每帧重新算 5/95 分位
near = (inv - lo) * (1.0 / (hi - lo))          # 视差 ∝ near
```

`lo/hi` 是**每帧**按当前画面内容重算的。画面里只要有物体进出、明暗变化，分位带就漂移，同一静止物体每帧视差都不同——这是一个与 DA3 无关、且很强的抖动放大器。

> 关键结论：**即使换更好的深度模型，只要不改这段逐帧归一化，抖动仍会残留。来源 2 必须修。**

---

## 1. 调研目标

是否存在可以"叠加"在现有逐帧 DA3 深度之上的**现成 ONNX 模型**（视频深度稳定化 / 时间一致性），从而不用手写时域算法。如果没有，就只做"第一步"（渲染端时域稳定化）。

---

## 2. 调研结论（核心）

> **不存在可以直接下载、即插即用的"稳定化 ONNX"。** 公开的 ONNX/TensorRT 导出全部是**单帧** Depth Anything V1/V2/V3（即本项目已经在用的那类），没有现成的"时域稳定化 ONNX"。

候选分两类：

### 2A. "叠加型"稳定器（架构上符合"叠加 ONNX"的设想）—— NVDS（已有可用权重，Phase 2 首选）

**NVDS+ / NVDS（Neural Video Depth Stabilizer，ICCV 2023 / TPAMI 2024）**

- 设计就是"depth predictor（任意单帧深度模型）+ stabilization network（稳定网络）"，**plug-and-play**，把闪烁的 disparity 精修成时间一致的 disparity。完全契合"在 DA3 后面叠加一个模型"的思路。
- 滑动窗口（每窗 4 帧），cross-attention。
- **推理阶段不需要光流**：光流（GMFlow）只用在**训练 loss（OPW）+ 可选评测指标**里。推理是纯前馈：输入 = 窗口内 RGB + 闪烁 disparity，输出 = 目标帧精修 disparity。→ **走 NVDS 路线就不需要 GMFlow（与 2A' 二选一，不叠加）。**
- 输入格式：NVDS 吃归一化 disparity，与本项目 `_normalize_near` 输出形态天然吻合；predictor-agnostic，喂 DA3 disparity 正是标准用法。输出已是稳定 disparity → **可同时吸收来源 1 和来源 2**。

**2026-06-18 复核：NVDS 已有可用权重（release）。** 这是最强 Phase 2 候选——直接满足"用模型、不手写时域算法"的最初诉求。

- Release 资产（均 354 MB）：
  - **`NVDS_Stabilizer.pth`（VDW 通用版）→ 选这个**，处理任意 in-the-wild 片源。
  - `NVDS_Stabilizer_NYUDV2_Finetuned.pth`（NYU 室内 finetune）：仅复现该 benchmark / 纯室内场景，**不要**用于一般片源（会过拟合室内域）。
- **重要更正**：该 release 是**原版 NVDS**，**不是**轻量版 NVDS+。354 MB ≈ 88M 参数（fp32），DPT-Large 量级逐帧推理——比 NVDS+「5M/35fps」重得多，**实时链路会与 DA3 抢 GPU**。之前文中的 5M/35fps 数据不适用于此权重。
- **2026-06-18 复核：NVDS+ 的 5.04M 轻量版权重未公开发布（论文 only，repo 只放了原版 NVDS）**，通过 GitHub API 确认 release 仅 `NVDS_Stabilizer.pth` + NYU finetune 两个资产。即「轻量版」目前不可得，不能作为方案依据；走 NVDS 路线只能用原版 354MB。→ 这也抬高了 2A'（GMFlow，更轻、有现成 ONNX/TRT）的相对吸引力。
- **主要风险 = ONNX 导出**：cross-attention 依赖 `mmcv-full==1.3.0` + `mmseg==0.11.0`（很老、含自定义算子）+ 4 帧滑窗 cross-attention，导出不平凡，是这条路线的真正工作量与风险点。
- 效果风险：NVDS 原始 depth predictor 是 DPT/MiDaS 系；喂 DA3 disparity 属"未训练过的搭配"，需实测验证（架构 plug-and-play，泛化质量要试）。
- 仓库 / release：`github.com/RaymondWang987/NVDS` ，`github.com/RaymondWang987/NVDS/releases`

**落地次序（避免一上来就做 ONNX）：**
1. 先做第一步改造 A（便宜、立竿见影，且是 NVDS 导出搞不定时的兜底）。
2. **NVDS 验证 spike**：临时 torch+mmcv 环境，用官方 PyTorch demo 把**本项目 DA3 的 disparity** 喂进 `NVDS_Stabilizer.pth`，在 2~3 段代表性片源上肉眼 + ROI 量化，确认确实消抖且无伪影。
3. 确认有效后，再投入 ONNX 导出（mmcv/mmseg → ONNX）。导出代价过高则回退到 2A'（GMFlow）或仅改造 A+B。

### 2A'. GMFlow / UniMatch 光流 —— 推荐的 Phase 2 路线（替代 NVDS+）

**重要定位：GMFlow 是纯光流模型，本身不稳定深度。** 它输出相邻帧的逐像素运动向量，作用是给"运动补偿式时域深度平滑"（见 4.2 改造 B 进阶版）提供高质量运动估计。它**不能**替代 NVDS 稳定器，**不能**消除来源 2（仍需改造 A），**仍需手写** warp + 融合逻辑——它只是把"手写运动估计（Farneback，弱/CPU）"升级成"强模型估运动"。

为什么它比 NVDS+ 更适配本项目：

- **有可用权重 + 现成 ONNX/TensorRT 导出**：GMFlow 升级版 UniMatch（同作者，统一 flow/stereo/depth）有社区仓库 `github.com/fateshelled/unimatch_onnx`（ONNX + TensorRT 推理 demo），与本项目现有 ONNX/TRT/exe 基建同路子。而 NVDS 稳定器找不到可用模型。
- **根治朴素 EMA 的"运动拖影"命门**：有准确光流后，把上一帧深度 warp 到当前帧、只在真正对应的同一表面点上融合 → 去抖动但不拖影；再加前后向一致性检查屏蔽遮挡/失配区域（那些区域退回当前帧深度）。
- 速度：GMFlow basic（无 refinement）约 26ms@A100 / 57ms@V100（436×1024），量级与 DA3 相当。

代价：

- **额外逐帧 GPU 推理**：离线转换可接受；**实时链路会与 DA3 抢 GPU**，需评估。
- 社区 ONNX 导出需自行验证/适配**固定输入分辨率**（global matching 的 softmax 对 H×W 敏感，导出须锁形状；UniMatch issue #29 提到 dict 输出导出报错，需改成 tensor 输出）。
- warp + 融合 + 遮挡掩码逻辑仍需手写（不复杂，但非零）。

仓库：`github.com/haofeixu/gmflow`（原始）、`github.com/autonomousvision/unimatch`（升级版）、`github.com/fateshelled/unimatch_onnx`（ONNX/TRT）

### 2B. "替换型"视频深度模型（不是叠加，是替换整个 DA3 阶段）

这些把时序能力**烤进模型本体**，不能叠加在 DA3 之上，只能替换掉现有 DA3 阶段：

- **Video Depth Anything（ByteDance, CVPR 2025）**：DAv2 换成时空头（temporal self-attention）。Small 28.4M / 7.5ms、Large 381.8M / 14ms（A100, FP16, 32 帧 batch, 518²）。有 2025-07 的实验性 streaming 模式（缓存时序 attention 隐藏态、单帧进），但官方说**有掉点**，建议再 finetune。**无官方 ONNX/TensorRT。**
- **Online Video Depth Anything（arXiv 2510.09182）**：低显存、流式版本，同样无官方 ONNX。
- **FlashDepth（arXiv 2504.07093）**：2K 实时流式视频深度，研究级，无官方 ONNX。

替换型的问题：① 都没有官方 ONNX；② 等于推翻现有 DA3/ONNX/TensorRT/exe 基建重做；③ 与本项目"轻量、ONNX、可打包 exe"的设计冲突。

### 2C. 关于 DA3-Streaming（用户最初提到的）

`reference/da3/da3_streaming/` 不是"输出稳定深度视频"的模块，而是一套**类 SLAM 的离线三维重建管线**（基于 VGGT-Long）：分 chunk 多帧联合推理 → 估计相机位姿/内参/深度/置信度 → chunk 间 Sim3 对齐 + 闭环 → 融合成全局点云 `combined_pcd.ply`。

- 真正治抖动的只有"chunk 内多帧联合推理（跨帧注意力）→ 帧间一致深度"这一机制，其余（点云、轨迹、闭环）你都用不上。
- 硬障碍：依赖 **PyTorch 模型 + safetensors**（非 ONNX）；显存 16–28GB；读磁盘 PNG 序列 + 大量临时文件；一堆额外依赖（triton/numba/SALAD）。对本项目 ONNX/exe 设计是重负担。
- 结论：**整体搬入是错的方向**（重、慢、输出形态不对）。

---

## 3. 决策结论

1. **没有"下载即用的叠加式稳定化 ONNX"。** 用户设定的前提成立 → **执行第一步（渲染端时域稳定化）**。
2. **Phase 2 首选 = NVDS 稳定器（已有 release 权重 `NVDS_Stabilizer.pth`）。** 直接满足"用模型不手写算法"，DA3 disparity 进→稳定 disparity 出。但是原版 NVDS（~88M/354MB，较重）+ ONNX 导出依赖老旧 mmcv/mmseg（主要风险）→ 先 PyTorch spike 验证效果，再决定是否做 ONNX。推理不需要 GMFlow。
3. **Phase 2 备选 = GMFlow/UniMatch（2A'）**：若 NVDS ONNX 导出代价过高再走。它是**光流不是稳定器**，需手写 warp+融合，作改造 B 的高质量后端（B2）。与 NVDS **二选一**。
4. **替换型（Video Depth Anything 等）/ DA3-Streaming 不建议**：无 ONNX、推翻现有基建、与 exe 打包设计冲突。
5. 无论走哪条 Phase 2，**改造 A 都先做**（最便宜、且是兜底）。

---

## 4. 第一步：渲染端时域稳定化 —— 详细开发方案

目标：在**不动模型、不加依赖**的前提下，消掉来源 2、并平滑来源 1 的高频抖动。改动集中在 `offline/two_dvr_render.py`，并需让渲染器从"无状态逐帧"升级为"按视频序列保持状态"。

### 4.0 现状与核心改造点

渲染器目前是**无状态逐帧** `renderer.render(frame, depth)`（见 `offline/two_dvr.py:278` `_pump_pipeline` 主循环、`offline/two_dvr.py:415` `_pump_segment`）。时域稳定化需要跨帧状态（上一帧的统计量 / 上一帧的 near），因此核心改造是：

1. 让 `StereoRenderer`（及 GPU/pynv 版本）**持有 per-clip 时域状态**；
2. 提供 `reset()`，在每个 clip / segment 起始处调用；
3. 三条渲染路径（CPU `StereoRenderer`、`two_dvr_gpu.GpuStereoRenderer`、`two_dvr_pynv` GPU 常驻管线）都要同步实现；
4. 注意 clip / segment 边界、镜头切换时的状态重置。

### 4.1 改造 A — 归一化分位带做时域平滑（**优先级最高、收益最大、必做**）

把 `_normalize_near` 里**每帧重算的 `lo/hi`** 改成跨帧 EMA（或滑动中位数）稳定：

```python
# 伪代码：renderer 持有 self._lo_ema / self._hi_ema（per-clip 状态）
lo_raw, hi_raw = np.percentile(sample, [5.0, 95.0])
if self._lo_ema is None:           # clip 第一帧
    self._lo_ema, self._hi_ema = lo_raw, hi_raw
else:
    a = NORM_EMA_ALPHA             # 例如 0.05~0.15，越小越稳但越迟钝
    self._lo_ema = (1 - a) * self._lo_ema + a * lo_raw
    self._hi_ema = (1 - a) * self._hi_ema + a * hi_raw
near = (inv - self._lo_ema) * (1.0 / (self._hi_ema - self._lo_ema))
```

- 直接消除"分位带逐帧漂移"这个抖动放大器，通常立竿见影。
- `NORM_EMA_ALPHA` 需可配置（环境变量 / CLI），并实测调参。
- 镜头切换（场景突变）时 EMA 会"拖尾"几帧：可加突变检测（lo/hi 相对跳变超阈值就重置 EMA），属可选增强。
- **注意**：本项目存在多种 max_side / 分辨率（`_processing_size`），`sample` 采样数 `_NORM_SAMPLE` 不变即可，EMA 与分辨率无关。

### 4.2 改造 B — 深度/视差帧间 EMA（平滑来源 1 的残余高频抖动，视情况做）

对归一化后的 `near`（或最终 disparity）做帧间 EMA：

```python
if self._near_prev is None:
    near_s = near
else:
    b = DEPTH_EMA_ALPHA            # 例如 0.3~0.6
    near_s = (1 - b) * near_prev_aligned + b * near
self._near_prev = near_s
```

- **风险：运动鬼影/拖影。** 静止背景平滑效果好，运动物体会拖影。
- 三档强度（**关键：把"光流后端"设计成可插拔**，融合逻辑只写一次）：
  - **B0 简单版**：直接像素级 EMA（实现最简单，运动区域轻微拖影，多数素材可接受）。
  - **B1 进阶版（Farneback）**：用上一帧到当前帧的运动把 `near_prev` warp 对齐后再 EMA（`near_prev_aligned`），并对运动剧烈区域降低 EMA 权重（运动越大越偏向当前帧）+ 前后向一致性检查屏蔽遮挡区。运动估计用 OpenCV `calcOpticalFlowFarneback`（CPU，**无新依赖**）。质量更好但实现复杂。
  - **B2 高质量版（GMFlow/UniMatch ONNX）**：与 B1 **完全相同的 warp+融合逻辑**，仅把光流后端从 Farneback 换成 GMFlow/UniMatch ONNX（见 2A'）。流场更干净 → 去抖不拖影上限最高。代价：额外逐帧 GPU 推理、需自行验证 ONNX 导出、实时链路与 DA3 抢 GPU。
- 实现要点：定义一个 `flow_backend(prev_rgb, cur_rgb) -> flow` 接口，B1/B2 共用 warp+融合；后端通过参数切换（`farneback` / `gmflow`）。
- 建议顺序：**先只做改造 A → 实测**；不够再上 B0；B0 拖影明显再上 B1（Farneback）；B1 质量仍不够再换 B2（GMFlow）。

### 4.3 涉及文件清单（开发对照）

| 文件 | 改动 |
|---|---|
| `offline/two_dvr_render.py` | `_normalize_near` / `prepare_near` 实现改造 A、B；`StereoRenderer` 增加 per-clip 时域状态字段 + `reset()` |
| `offline/two_dvr_gpu.py` | `GpuStereoRenderer` 同步实现时域状态（cupy） |
| `offline/two_dvr_pynv.py` | GPU 常驻管线同步实现 + 边界 reset |
| `offline/two_dvr.py` | `convert_clip`、`_run_segments`、`_pump_segment` 在每个 clip/segment 起始处 `renderer.reset()`；`_add_common_args` 新增 CLI 参数 |
| 实时链路（如适用） | `pipeline/` 下若有复用 renderer 的实时路径，同样需 reset 入口 |
| 配置/UI | 暴露稳定化强度参数（`NORM_EMA_ALPHA`、`DEPTH_EMA_ALPHA`、开关），可后置 |

### 4.4 新增参数（建议）

- `--temporal-norm`（开/关，默认开）+ `NORM_EMA_ALPHA`（默认 ~0.1）
- `--temporal-depth`（开/关，默认关）+ `DEPTH_EMA_ALPHA`（默认 ~0.4）
- 环境变量等价项（与项目现有 `PT_TWO_DVR_*` 风格一致）

### 4.5 验证方法

- 选 2~3 段代表性素材（含静止大背景 + 局部运动 + 镜头平移 + 镜头切换）。
- 量化：取若干静止区域 ROI，统计**相邻帧视差差分的标准差/均值**（OPW 风格时间一致性度量），改造前后对比。
- 主观：VR 头显 / SBS 播放器实看抖动是否消除、是否引入拖影、镜头切换是否有可见拖尾。
- 回归：确认未破坏现有 hole-fill（soft_shift / inverse_warp）与三种投影（flat3d / heq180 / fish180）。

### 4.6 开发顺序建议

1. CPU `StereoRenderer` 加 per-clip 状态 + `reset()` + 改造 A → 单视频实测调 `NORM_EMA_ALPHA`。
2. clip/segment 边界 reset 接好（`two_dvr.py`）。
3. GPU / pynv 路径同步改造 A。
4. 实测；不足则加改造 B 简单版（先 CPU，再 GPU）。
5. 暴露参数到 CLI/UI。

---

## 5. 一句话总结

**没有"下载即用"的稳定化 ONNX，但 NVDS 稳定器有可用 PyTorch 权重（`NVDS_Stabilizer.pth`，VDW 版）。当前先做"第一步"——渲染端时域稳定化（改造 A 必做、改造 B 视情况），改动集中在 `two_dvr_render.py` 且无新依赖。** 改造 A（修逐帧分位归一化）性价比最高、必须做，且是任何 Phase 2 的兜底。Phase 2 首选 **NVDS_Stabilizer.pth**（DA3 disparity 进→稳定 disparity 出，不手写算法；但 ~88M 较重、ONNX 导出依赖老旧 mmcv 是主风险 → 先 PyTorch spike 验证再导出）；备选 GMFlow/UniMatch ONNX（光流，需手写 warp+融合，与 NVDS 二选一）。

---

## 附：来源

- NVDS / NVDS+：https://raymondwang987.github.io/NVDS/ ，https://arxiv.org/abs/2307.08695 ，https://github.com/RaymondWang987/NVDS
- GMFlow / UniMatch（光流，推荐 Phase 2 后端）：https://github.com/haofeixu/gmflow ，https://github.com/autonomousvision/unimatch ，https://github.com/fateshelled/unimatch_onnx （ONNX/TensorRT），UniMatch ONNX 导出注意点 issue #29
- Video Depth Anything：https://arxiv.org/abs/2501.12375 ，https://github.com/DepthAnything/Video-Depth-Anything
- Online Video Depth Anything：https://arxiv.org/html/2510.09182v1
- FlashDepth：https://arxiv.org/pdf/2504.07093
- DA3-Streaming（参考源码）：`reference/da3/da3_streaming/`
- 现有单帧 DA ONNX/TRT 导出参考：https://github.com/spacewalk01/depth-anything-tensorrt ，https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
