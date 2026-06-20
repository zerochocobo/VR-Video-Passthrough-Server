# NVDS 接入与性能问题外部评审 Summary

日期：2026-06-20  
项目：PTMediaServer / 离线 2D 转 3D/VR  
目标：评估 RaymondWang987/NVDS 是否适合作为本项目离线 2D 转 3D 的时序深度稳定器。

## 1. 背景与目标

当前项目的 2D 转 3D 离线管线基于 DA3 ONNX 深度估计：

```text
视频帧 -> DA3 ONNX depth -> depth/near 归一化与时序稳定 -> stereo/VR 渲染 -> NVENC 输出
```

用户希望引入 NVDS，用模型方式解决 DA3 单帧深度在视频中的时序抖动问题。

约束条件：

- 主项目运行时必须保持 ONNX/ONNX Runtime，不引入 PyTorch runtime。
- NVDS 只能作为离线功能，不进入实时链路。
- 期望通过 UI 可选：
  - 默认算法：现有自研 VVPS/base-detail 时序稳定。
  - NVDS 模型：使用导出的 NVDS ONNX，限 16:9。

## 2. NVDS 模型与导出状态

本地源代码：

```text
reference/NVDS
```

原始权重：

```text
models/NVDS/NVDS_Stabilizer.pth
```

大小约：

```text
354,794,886 bytes
```

已导出 ONNX：

```text
models/NVDS/NVDS_Stabilizer_384x384.onnx
models/NVDS/NVDS_Stabilizer_672x384.onnx
```

其中 16:9 正式测试模型：

```text
models/NVDS/NVDS_Stabilizer_672x384.onnx
```

模型签名：

```text
input:
  name: rgbd_seq
  dtype: float32
  shape: [1, 4, 4, 384, 672]

output:
  name: stabilized_depth
  dtype: float32
  shape: [1, 1, 384, 672]
```

含义：

- batch = 1
- temporal window = 4 帧
- channel = 4，即 RGB + depth/near
- 输出目标帧稳定后的单通道 depth/near

ONNX 静态检查结果：

- opset 17
- 只有标准 `ai.onnx` domain
- 无 `ATen`
- 无自定义 domain node
- `onnx.checker.check_model(...)` 通过
- ONNX Runtime CUDA 对比 PyTorch 导出检查通过：
  - `max_abs_diff ~= 4.44e-05`
  - `mean_abs_diff ~= 5.21e-06`

导出脚本：

```text
examples/export_nvds_onnx.py
```

导出脚本已支持按尺寸自动命名：

```powershell
python examples\export_nvds_onnx.py --width 672 --height 384 --device cuda
```

## 3. 当前项目接入方式

新增 wrapper：

```text
offline/nvds_stabilizer.py
```

职责：

- 仅使用 ONNX Runtime，不依赖 PyTorch。
- 维护 4 帧 RGBD history。
- 将 DA3 distance-like depth 转为 normalized near/disparity。
- 按 NVDS 输入要求构造：

```text
[1, 4, 4, 384, 672]
```

- 输出 normalized stable near/disparity。
- 统计 NVDS 单模型推理耗时：

```text
nvds_frames=N nvds_infer=...ms/frame (... fps)
```

新增渲染入口：

```text
offline/two_dvr_render.py: StereoRenderer.render_near(...)
offline/two_dvr_gpu.py: GpuStereoRenderer.render_near(...)
```

原因：

- 现有 `renderer.render(frame, depth)` 会把输入当作 DA3 raw depth，再执行 reciprocal + percentile normalization。
- NVDS 输出已经是 stabilized near/disparity，不能再作为 distance depth 二次处理。
- 因此 NVDS 路径必须走 `render_near(frame, stable_near)`。

新增 CLI：

```powershell
--depth-stabilizer default
--depth-stabilizer nvds
```

UI 位置：

```text
ui/pages/two_dvr_page.py
```

在“画质速度”下方新增“时序稳定”：

- 默认算法
- NVDS模型（限16:9）

设置项：

```text
ui/settings.py: two_dvr_depth_stabilizer
```

默认：

```text
default
```

## 4. TensorRT 结论

DA3 当前仍可使用 TensorRT EP 缓存。

NVDS 当前不能有效使用 TensorRT EP。

用户日志中出现：

```text
onnx.ModelProto exceeded maximum protobuf size of 2GB: 7050012025
[TensorRT EP] No graph will run on TensorRT execution provider
```

同时 NVDS ONNX 图中包含大量 `ScatterND` 等算子。

当前判断：

- NVDS 导出的 ONNX 可以被 CUDA EP 加载。
- TensorRT EP 对当前图无法生成可用 engine。
- 即使请求 `--provider trt`，NVDS wrapper 也会自动使用：

```text
CUDAExecutionProvider -> CPUExecutionProvider
```

- DA3 仍继续使用 TensorRT。

因此当前 NVDS 实际运行组合是：

```text
DA3: TensorRT EP
NVDS: CUDA EP
renderer: CPU 或 CuPy GPU renderer
encoder: NVENC
```

## 5. 已观察性能问题

测试输入：

```text
videos/test_4k2d.mp4
```

视频信息日志显示：

```text
1920x1080@59.940
proc 1920x1080
SBS 3840x1080
proj=flat3d
fill=soft_shift
model=base
depth=TensorrtExecutionProvider
depth_stabilizer=nvds
```

用户反馈第一次 NVDS 测试速度极慢并最终卡住：

```text
64/1798 frames   elapsed=00:34  1.9 fps
128/1798 frames  elapsed=01:02  2.1 fps
192/1798 frames  elapsed=01:30  2.1 fps
256/1798 frames  elapsed=03:41  1.2 fps
```

之后尝试改成串行处理，避免 DA3/NVDS/CuPy 多线程争 GPU，结果更差：

```text
约 0.1 fps
```

该串行改动已回滚。

当前保留状态：

- 保留 NVDS 接入。
- 保留 NVDS 使用 CUDA EP，不走 TensorRT。
- 保留 bitrate int -> str 的崩溃修复。
- 回滚 NVDS 串行 pipeline。
- NVDS 路径恢复原来的 3-stage pipeline。

## 6. 曾发生的非性能崩溃

用户日志中最终 traceback：

```text
TypeError: expected str, bytes or os.PathLike object, not int
```

原因：

```text
offline/two_dvr.py:_encode_proc(...)
```

中将整数 bitrate 传给 `subprocess.Popen`。

已修复：

```python
str(preset)
str(bitrate)
```

该问题与 NVDS 模型性能无关。

## 7. 当前运行路径细节

NVDS 每帧流程：

```text
1. DA3 对当前 frame 输出 low-res depth crop
2. wrapper 将 DA3 depth 转 reciprocal near/disparity
3. resize near 到 672x384
4. resize RGB 到 672x384
5. RGB 使用 ImageNet mean/std 归一化
6. 拼接 RGB + near，形成单帧 4-channel
7. 维护 4 帧序列
8. ORT CUDA EP 运行 NVDS
9. 输出 384x672 stable near
10. stable near 进入 render_near
11. 渲染 SBS
12. NVENC 编码
```

当前使用的是 causal 4-frame window：

- 前几帧通过重复首帧补齐。
- 没有实现 NVDS 官方 demo 中的 bidirectional/mix 推理。
- 没有 GMFlow，GMFlow 只用于官方评估/训练相关路径，不参与本接入。

## 8. 可能的问题方向

需要外部专家判断的核心问题：

### 8.1 原版 NVDS 是否本身太重

当前权重 354MB，明显不是轻量版 NVDS+。

官方 README 中也提到：

- 896x384 下 stabilizer 大约需要 5GB VRAM。
- 这还不包括 DA3、渲染和编码。

当前 672x384 下 1-2 fps，可能符合原版 NVDS 在该硬件/ORT CUDA 下的实际开销。

### 8.2 ONNX 导出图是否不适合 ORT CUDA

图中包含大量：

- `ScatterND`
- `Slice`
- `Where`
- `Range`
- `Reshape`
- window attention 相关动态/半动态结构

CUDA EP 可以跑，但可能没有理想 kernel fusion。

TensorRT EP 完全无法有效接管。

需要专家判断是否可通过改写 PyTorch forward / ONNX graph，减少 ScatterND 和巨大图优化开销。

### 8.3 是否应该重新导出更小分辨率模型

当前用于 16:9 的模型是：

```text
672x384
```

可考虑：

```text
384x216
512x288
```

但风险：

- 深度边界精度下降。
- 3D 输出边缘伪影可能增加。
- 对 4K/1080p 的 stereo 渲染可能不够。

### 8.4 是否应该降低 NVDS 运行频率

例如：

- 每 2 帧运行一次 NVDS，中间帧插值/复用。
- 每 4 帧运行一次 NVDS，其余帧用默认算法。
- 只对低频 base 层使用 NVDS，保留当前帧 detail。

这可能比逐帧 NVDS 更实用。

### 8.5 输入 depth 分布是否匹配 NVDS 训练域

NVDS 官方 demo 使用 DPT/MiDaS depth。

本项目输入是 DA3 distance-like depth，再转 normalized near/disparity。

虽然架构上 plug-and-play，但分布可能不完全一致。

需要专家判断：

- 应喂 DA3 raw depth？
- 应喂 reciprocal near？
- 应喂 percentile-normalized near？
- 是否需要对 NVDS 输出做 scale/bias 或 temporal norm？

### 8.6 是否应该放弃原版 NVDS，转向轻量时序/光流方案

如果目标是实用离线速度，可能需要考虑：

- 默认 VVPS 继续强化。
- GMFlow/UniMatch 只做运动估计，再手写 near warp+blend。
- 小模型 learned stabilizer。
- 训练或蒸馏一个轻量 ONNX stabilizer。

## 9. 复现命令

默认算法：

```powershell
.\.venv\Scripts\python.exe offline\two_dvr.py single videos\test_4k2d.mp4 --duration 10 --depth-stabilizer default --provider trt --preset p4
```

NVDS：

```powershell
.\.venv\Scripts\python.exe offline\two_dvr.py single videos\test_4k2d.mp4 --duration 10 --depth-stabilizer nvds --provider trt --preset p4
```

注意：

- `--provider trt` 对 DA3 有效。
- 对 NVDS 会自动降为 CUDA EP。

## 10. 当前代码改动文件

核心新增/修改：

```text
offline/nvds_stabilizer.py
offline/two_dvr.py
offline/two_dvr_render.py
offline/two_dvr_gpu.py
ui/pages/two_dvr_page.py
ui/settings.py
ui/translations/zh_CN.json
ui/translations/en_US.json
examples/export_nvds_onnx.py
```

测试：

```text
tests/test_two_dvr_temporal_stability.py
tests/test_da3_download_and_trt.py
tests/test_two_dvr_progress.py
```

最近测试结果：

```text
32 passed
```

## 11. 需要专家给出的判断

希望专家重点判断：

1. 当前 NVDS ONNX 图是否有优化/重导出的空间，尤其是 TensorRT 不可用和 ScatterND 过多的问题。
2. 672x384 原版 NVDS 在 ORT CUDA 下 1-2 fps 是否属于合理预期。
3. 是否有推荐的 ONNX graph rewrite / PyTorch forward rewrite 方式，能显著提升 CUDA EP 或 TRT EP 性能。
4. 对本项目这种 DA3 -> stereo 的场景，NVDS 输入应使用 raw depth、inverse depth、near/disparity，还是其他归一化方式。
5. 如果原版 NVDS 不适合生产，是否建议改为低频 NVDS、轻量光流补偿、或重新训练/蒸馏小模型。

## 12. 调查结论与已落地修复（2026-06-20 更新）

经过代码走查 + 带 `nvidia-smi` 显存曲线的实测复现，问题拆成**两个独立问题**：

### 12.1 卡死 / 进行性变慢 —— 根因 = VRAM 溢出（已修复）

不是「NVDS 太慢」，恒定开销不会给出 `1.9→2.1→1.2 fps 然后冻死` 的曲线。

实测显存：DA3(TRT) + NVDS(CUDA EP) + CuPy renderer 同享 GPU，真实工作集已达
**~14.9GB / 16GB**。ORT 默认 arena 用 `kNextPowerOfTwo` 增长且**永不归还**，把组合
占用顶过物理显存 → Windows WDDM 把显存分页到共享系统内存（shared GPU memory）→
每帧越来越慢直至接近冻结。这也解释了串行版更差（0.1fps）：串行没有缓解显存压力，
反而每帧多付一次 page-in。

**修复**（`offline/nvds_stabilizer.py`）：给 NVDS 的 `CUDAExecutionProvider` 设
`gpu_mem_limit`，自动 = 总显存 − `_NVDS_VRAM_RESERVE_MB`（默认 4096MB，留给
DA3/CuPy/NVENC）。**关键是保留 ORT 的快速默认 arena 与 cudnn 内核** —— 实测改用
`arena_extend_strategy=kSameAsRequested` + `cudnn_conv_algo_search=DEFAULT` 虽然也
止住卡死，但把 fps 从 2.8 砍到 0.4，是错误的旋钮。

env 覆盖：`PTMS_NVDS_GPU_MEM_LIMIT_MB`（直接指定上限）、
`PTMS_NVDS_VRAM_RESERVE_MB`（余量）。

实测对比（`videos/test_4k2d.mp4`，同帧位）：

| frame | 修复前（坏） | 修复后 |
|---|---|---|
| 64  | 1.9 fps | 2.7 fps |
| 128 | 2.1 | 2.8 |
| 192 | 2.1 | 2.8 |
| 256 | 1.2（崩溃中→冻死） | 2.8 |
| 320 | 已冻 | 2.8 |
| 384 | 已冻 | 2.9 |

修复后 VRAM 平稳在 ~12.6GB（峰值 12.9GB），永不触顶，全程不再冻死，且比原来更快
（有界 arena 顺带规避碎片开销）。`tests/test_two_dvr_temporal_stability.py` 21 passed。

### 12.2 ~2.8fps 是架构天花板（待优化，非 bug）

针对 §11 的问题 1/2/3 给出答复：

`reference/NVDS/full_model.py` 的 backbone = `mit_b5()`（SegFormer MiT-B5，是 354MB
权重的主体）；`backbone.py:398` 把 `[N,T,C,H,W]` 的 4 帧全部展进 batch 跑完整
backbone。causal 滑窗每帧步进 1 → **相邻两次推理有 3/4 的 backbone 计算完全重复**。
所以 672×384 下 ORT CUDA ~2-3fps 属于该结构的合理预期，不是 EP 配置问题。

优化路线（按性价比）：

1. **拆 ONNX 图（进行中）**：backbone 单帧导出 + wrapper 缓存最近 4 帧特征，
   stabilizer 跨帧 attention 头单独跑。理论砍掉 ~3/4 backbone 开销，同时显著降低
   显存峰值。改导出脚本即可，不需重训。
2. **降频跑 NVDS**：每 N 帧一次当 base 层，中间帧用 DA3/VVPS detail（和现有
   base/detail 架构契合，工程量最小，2–4×）。
3. **降分辨率**：512×288 / 384×216。

针对 §11 问题 3（TRT）：TRT 路线放弃。2GB protobuf 上限 + `ScatterND` + 动态分支
（`backbone.py:240` 的 `if x.shape[1]==4`）使 TRT EP 无法接管；拆图后 backbone
部分（基本是 conv/matmul）理论上才有重上 TRT 的空间。

针对 §11 问题 4（输入分布）：`*.onnx.json` metadata 已注明应喂 normalized
near/disparity，wrapper 的 `_depth_to_near` 即按此实现，非性能瓶颈，暂不动。

## 13. 拆图实测结论（2026-06-20，纠正 §12.2 的假设）

已实现「backbone 单帧 + head 跨帧」拆图（`examples/export_nvds_onnx.py --split`，
wrapper `offline/nvds_stabilizer.py` 自动检测 split 模型并逐帧缓存最近 4 帧 backbone
特征）。两半图数值与原整图一致：`max_abs_diff≈3.5e-05`（与原整图自身导出误差
4.4e-05 同量级）；wrapper 端到端复算与整图 wrapper 逐位相同（CPU 上 diff=0）。

**但实测推翻了「backbone 是瓶颈」的假设。** NVDS 单模型耗时（CUDA EP，672×384，
40 帧均值）：

| 配置 | backbone | head | NVDS 合计 | NVDS-only fps |
|---|---|---|---|---|
| 整图 monolith | （4 帧 ~84ms） | — | 292.7ms | 3.4 |
| 拆图 split | 26.3ms（1 帧） | 208.3ms | 258.0ms | 3.9 |

- **head（focal 3D 跨帧注意力解码器）才是大头：208ms，是 backbone 单帧的 8 倍。**
- backbone 的 4 帧冗余只占 ~84ms；拆图去冗余仅省 ~35ms → NVDS 提速 1.13×，
  端到端 2.8→3.1 fps（pipeline 里 NVDS 与 DA3/渲染/编码 重叠，且 head 无法像
  backbone 那样跨窗复用——它本就是逐输出帧的跨帧融合）。

**真正的杠杆是分辨率**（head 开销 ∝ stride-4 token 数，即输入像素数）。实测拆图：

| 分辨率 | backbone | head | NVDS 合计 | NVDS-only fps |
|---|---|---|---|---|
| 672×384 | 26.3ms | 208.3ms | 258ms | 3.9 |
| 512×288 | 20.9ms | 125.7ms | 146.6ms | 6.8 |

512×288（仍 16:9）像素降到 0.57×，head 降到 0.60×，NVDS 整体 **1.76× 提速**
（3.9→6.8 fps），质量代价中等。导出与整图数值一致（`max_abs_diff≈2.6e-05`）。

**结论与下一步**：
1. 拆图保留——数值精确、略快、单次推理显存更低，且让低分辨率重导出变简单；但单靠
   它收益有限（1.13×）。
2. **优先把 NVDS 分辨率降到 512×288 作为默认（或可选档位）**，这是当前最大且最稳的
   提速来源（~1.76×）。
3. 仍可叠加「降频跑 NVDS」（每 N 帧一次，~150-260ms/次），进一步 2–4×。
4. 进一步压 head 需改 focal attention 结构并重训，暂不做。

复现工具：`tools/nvds_split_bench.py [split|mono]`（NVDS 单模型微基准，含 backbone/
head 拆分计时）。
