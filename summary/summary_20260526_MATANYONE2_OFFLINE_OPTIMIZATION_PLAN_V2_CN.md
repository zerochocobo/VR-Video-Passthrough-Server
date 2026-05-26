# MatAnyone2 离线优化 V2 — 五项质量/速度增强实施计划

**日期**：2026-05-26
**版本**：V2（V1 已完成，进入下一阶段）
**作者**：架构调研 + 开发者反馈整合（Claude 协同）
**目标读者**：执行本次重构的工程师
**关联报告**：
- `summary/summary_20260526_MATANYONE2_OFFLINE_OPTIMIZATION_PLAN_CN.md`（V1 计划，**已完成**）
- `summary/summary_20260526_MATANYONE2_RESEARCH_CN.md`（模型机制综合调研）
- `summary/summary_20260525_RVM_OFFLINE_PRECISION_TIERS_PLAN_CN.md`（RVM 计划，含 F/G 节实现）
- `summary/summary_20260526_MATANYONE2_OFFLINE_V2_DEVELOPMENT_PLAN_CN.md`（开发者反馈版，已整合）

---

## 0. 开发者反馈整合摘要（2026-05-26 二次修订）

收到开发者审阅后的修订版，本计划在此处吸收以下**关键纠错与改进**，并在对应章节落地：

| 编号 | 内容 | 落地位置 |
|---|---|---|
| C1 | **ROI 收益认知纠错**：固定 1024 ONNX 输入下，ROI crop 不会降低 attention token 数和 FLOPs，G3 仅是远景**质量功能**，不承诺提速；真正提速需要 ROI-B（512/768 ROI 模型） | §1.2、§6 全面改写 |
| C2 | **新增 Phase 0：共享 engine 抽取**——`tools/offline_passthrough.py` 和 `tools/offline_alpha_passthrough.py` 各持一份 `MatAnyone2OnnxEngine`，V2 改两遍危险，必须先抽 `offline/matanyone2_engine.py` | 新增 §2.5、§11 |
| C3 | **IOBinding 收敛范围**：只优化 `step_update` 热路径，首帧 `image_key/mask_memory/first_refine` 仍走 `session.run`，避免一次性 boil-the-ocean | §5 改写 |
| C4 | **Scene cut 改为 prepass 合并方案**：scene cut frame 加入 `segment_starts`，由 prepass（YWES / SAM3）补跑生成 bootstrap mask；主循环不做 inline bootstrap | §3 改写 |
| C5 | **Guided filter guide 用 NV12 Y 平面**（无需 BGR 转换） | §7.3 修订 |
| C6 | **ROI 拆 ROI-A / ROI-B**：A 质量优先（1024 输入），B 速度优先（512/768 输入，V1 保留的 512 容器在此启用） | §6 改写 |
| C7 | **验证矩阵 5 档**：V1 baseline / IOBinding only / +guided / Full V2 / +ROI-A | §9.2 修订 |

---

---

## 1. 背景与起点

### 1.1 V1 已完成

V1 已经把 MatAnyone2 从"实验性 2048 不可用状态"收敛为：
- 仅保留 1024 ONNX 精度档（512 文件保留，2048 已清理）
- UI 容器保留，单项 `setEnabled(False)`
- FP16 维持，定义了回退触发条件
- TRT warmup 覆盖 1024-bs1/bs2 两套

V1 完成后的实测基线：
- `1024 + TRT step_update`：~8 FPS，step_update 平均 49.7 ms
- `1024 + CUDA fallback`：~3.9 FPS

### 1.2 V2 目标

按 V1 第 9 节列出的后续 backlog，本计划落地以下 5 项（已按开发者反馈修订收益预期）：

| 项 | 投入 | 期望收益 | 备注 |
|---|---|---|---|
| **G0. 共享 engine 抽取**（新增）| 小 | 减少 V2 改两遍风险 | 两份 `MatAnyone2OnnxEngine` 合并到 `offline/matanyone2_engine.py` |
| **G1. GPU IOBinding（step_update only）** | 大 | 1024 FPS：8 → 10-14（消除 PCIe 往返）| **范围收敛**：首帧 graphs 不动 |
| **G2. Edge-aware alpha upsampler** | 中 | 8K 画质达 RVM+FGF 同等边缘锐度 | guide 用 NV12 Y 平面 |
| **G3-A. 1024 ROI crop（质量）** | 中 | **远景人物质量提升，不承诺 FPS** | 固定 1024 输入，token 数不变 |
| **G3-B. 512/768 ROI 模型（速度）** | 中 | 远景内容真正提速 | 启用 V1 保留的 512 容器；3A 验证后再决定 |
| **G4. 场景检测 → prepass segment plan** | 小（复用 RVM `_SceneCutDetector`）| 跨场景 mask 漂移消除 | scene cut 合并进 `segment_starts`，prepass 补 bootstrap |
| **G5. Alpha smoother (EMA α=0.6)** | 小（复用 RVM `_AlphaSmoother`） | 时序抖动抑制 | reset 必须与 segment / scene cut / ROI 变更同步 |

### 1.3 现有可复用资产（**重要**）

调研发现项目已有以下资产可**直接复用**，避免重写：

| 资产 | 位置 | 用途 |
|---|---|---|
| `_SceneCutDetector` 类 | `pipeline/matting.py:98-147` | G4 直接 import |
| `_AlphaSmoother` 类 | `pipeline/matting.py:150-164` | G5 直接 import |
| RVM IOBinding 范式 | `pipeline/matting.py:2103-2210` | G1 参考实现 |
| SAM3 `union_bbox_xyxy` | `offline/sam3_matanyone2.py:254,261` | G3 bbox 来源之一（重模型分支）|
| YOLOWorld+EfficientSAM `box_xyxy` | `offline/yoloworld_efficientsam.py:23,250,308` | G3 bbox 来源之一（轻量分支，**默认**）|
| 双前置选择 argparse | `tools/offline_passthrough.py:1432` | `--matanyone2-prepass=sam3 \| yoloworld_efficientsam` |
| `Matter._gpu_preprocess_nv12_one` GPU 路径 | `pipeline/matting.py` | G1 输入端复用（仅需关闭 `copy_to_host=True`） |

这意味着 V2 的 5 项里 G4/G5 几乎是**纯接线工作**；G1 是**改造 + 借鉴**；G3 是**新增小模块 + 借鉴**；只有 G2 是相对独立的新算法。

---

## 2. 总体决策

### 2.1 V2 五项均**仅对离线 MatAnyone2 路径生效**

实时 RVM、离线 RVM 路径均不动。所有改动在 `tools/offline_passthrough.py` 的 `MatAnyone2OnnxEngine` 内部或 `pipeline/` 新增模块。

### 2.2 5+2 项之间的依赖与执行顺序

```
G0（engine 抽取）─→ 所有后续都基于统一 engine
       ↓
G1（IOBinding step_update）─→ G2（edge-aware upsample）
       ↓                              ↓
G4（场景检测 → prepass plan）─→ G5（alpha smoother，reset 信号闭环）
       ↓                              ↓
G3-A（1024 ROI 质量）─→ G3-B（512/768 ROI 速度，3A 验证后决定）
```

**推荐执行顺序（吸收开发者反馈，调整为）**：**G0 → G1 → G2 → G4 → G5 → G3-A →（条件触发）G3-B**

- **G0** 先做：抽 `offline/matanyone2_engine.py`，避免后续每项改 green/alpha 两份代码
- **G1** 紧跟：性能基线提升的核心改动，且只动 `step_update` 热路径
- **G2** 与 G1 后做：拿到 GPU-resident alpha 后，guided filter 可直接消费 CuPy buffer
- **G4 → G5**：时序稳定性，G5 reset 信号依赖 G4
- **G3-A 后置**：固定 1024 模型下 G3 只是质量实验，不是核心性能路径
- **G3-B 条件触发**：仅当 G3-A 质量验证通过且仍需进一步加速时启动

> **重大顺序变更说明**：上一版把 G4/G5 放最前（理由是复用 RVM 类工时小），现采纳开发者建议把 G0/G1 前置。G1 提供的 GPU-resident alpha 是 G2/G5 的高效前提；先做 G0 避免 V2 中后期发现两份代码改不一致。

### 2.3 全部 5 项默认行为

- 配置项均加在 `config.py`，环境变量名按 `PT_MATANYONE2_*` 前缀
- 离线模式默认**开启** G4、G5；G1/G2 通过编译期开关（默认开），G3 默认关（实验性）
- 实时路径全部默认关闭（即便复用了类，实时分支不调用）
- UI 一律不暴露（V2 是内置质量优化）

### 2.4 FP16 精度策略

V1 已锁定 FP16 维持。V2 不改动这一决策。**G1 IOBinding 改造时 OrtValue 数据类型必须从 ONNX 入参元数据动态读取**（`tensor(float16)` → fp16 OrtValue），不能硬编码。

### 2.5 G0：共享 engine 抽取（V2 改动的前置条件）

**问题**：`tools/offline_passthrough.py`（green 输出）和 `tools/offline_alpha_passthrough.py`（alpha-packed 输出）各持一份几乎相同的 `MatAnyone2OnnxEngine` 实现。V2 五项中 G1/G2/G3/G5 都要动 engine 内部，若不抽出，所有改动需双写双测，极易不一致。

**改动**：

1. 新增 `offline/matanyone2_engine.py`，将 `MatAnyone2OnnxEngine`、`_EyeState`、`_run_eye`、`_run_eyes_batch2` 迁入
2. 输出模式差异通过策略接口注入：
   - green 路径传入 `lambda alpha, h, w: matter._composite_nv12_to_nv12_gpu_using_uploaded_frame(alpha, h, w)`
   - alpha-packed 路径传入 `lambda alpha, h, w: AlphaPacker.pack_uploaded(alpha, h, w)`
3. CLI / 解码 / 编码逻辑**留在原工具内**，仅 engine 共享
4. 新增 profile 明细字段（V2 全程沿用）：
   - `matanyone2_preprocess_eye_avg`
   - `matanyone2_step_update_avg`
   - `matanyone2_iobinding_copy_avg`
   - `matanyone2_guided_upsample_avg`
   - `matanyone2_composite_avg`
5. 日志 summary 增加 `matanyone2_iobinding=0/1`、`matanyone2_alpha_refine=off|guided`

**验收**：

- green/alpha 两路输出与 V1 像素一致（`--alpha-stride 1`）
- `tests/test_offline_convert.py`、`tests/test_matanyone2_trt_runtime_paths.py` 全通过
- 新增 `tests/test_matanyone2_engine.py` 覆盖 engine 单元测试

**V2 前基准**：

- 8K SBS 15 s 视频，`--alpha-stride 1`，1024+TRT
- 记录 `matanyone2_step_update_avg`、端到端 FPS、显存峰值（作为后续 G1/G2/G3 对照）

**工时**：0.5-1 d

---

## 3. G4：场景检测 + MatAnyone2 segment 自动 reset

**优先级：最高**（投入最小，立刻拿到稳定性）

### 3.1 复用 RVM 的 `_SceneCutDetector`

直接 `from pipeline.matting import _SceneCutDetector` 即可使用。该类已经支持：
- HSV (H,S) 二维直方图 + Bhattacharyya 距离
- 参考帧 EMA 慢更新（α=0.95）
- 冷却窗口（默认 24 帧）
- 540p 预降采样

### 3.2 接入位置

`tools/offline_passthrough.py` 的 `MatAnyone2OnnxEngine`：

**3.2.1 `__init__` 新增成员**

```python
from pipeline.matting import _SceneCutDetector
from config import (
    MATANYONE2_SCENE_RESET,
    MATANYONE2_SCENE_THRESHOLD,
    MATANYONE2_SCENE_COOLDOWN,
    MATANYONE2_SCENE_REF_EMA,
)

self._scene_detector = _SceneCutDetector(
    threshold=MATANYONE2_SCENE_THRESHOLD,
    cooldown_frames=MATANYONE2_SCENE_COOLDOWN,
    ref_ema_alpha=MATANYONE2_SCENE_REF_EMA,
) if MATANYONE2_SCENE_RESET else None
```

**3.2.2 在 `composite_nv12` 入口检测**

`composite_nv12`（`tools/offline_passthrough.py:1012` 附近）开头加：

```python
def composite_nv12(self, frame):
    # 场景切换检测（仅对左眼即可，SBS 必然同步）
    if self._scene_detector is not None:
        # 从 frame 拿一份小尺寸 BGR 用于直方图。priority：
        # 1) 如果 frame 已有 cpu 缓存（少见）
        # 2) 否则用 self.matter 的现有 NV12→BGR helper（小尺寸即可）
        scene_bgr = self._scene_detector_frame_bgr(frame)
        if self._scene_detector.step(scene_bgr):
            print(f"[offline] MatAnyone2 scene cut detected at frame={self._frame_index}; forcing segment reset")
            self._reset_segment()
            # 标记下一帧必须重新 bootstrap
            self._force_resegment = True
    ...（原有逻辑）
```

**3.2.3 与 segment_starts 协同（重大设计修订）**

> **原方案**（已废弃）：主循环 inline 检测，命中后 reset state + 等下一个预定 segment_starts。问题是会有数秒"哑帧期"，且 inline bootstrap 复杂。
>
> **新方案**（采纳开发者建议）：scene cut 检测**前置到 prepass 扫描阶段**，与 prepass 一起生成 `segment_starts`，每个 scene cut 起点都补跑一次 prepass mask。主循环只按计划 reset，不做任何 inline bootstrap。

具体流程：

1. **Prepass 扫描时同步跑 scene cut**
   - 复用 `pipeline.matting._SceneCutDetector`（建议抽到 `utils/scene_detection.py` 作为通用模块）
   - 只检测 SBS 左眼，540p 降采样
   - 输出 `scene_cut_frames: list[int]`

2. **合并 segment starts**
   ```python
   base_starts = matanyone2_segment_frames_starts | prepass_active_sample_starts
   merged = sorted(set(base_starts) | set(scene_cut_frames))
   # min gap 过滤
   segment_starts = _enforce_min_gap(merged, min_seconds=3.0, fps=source_fps)
   ```

3. **为 scene cut start 补跑 bootstrap mask**
   - YWES 分支：对 cut frame 跑 `_sam_mask_for_box(image_rgb, box_xyxy, out_size)`
   - SAM3 分支：对 cut frame 跑 SAM3 mask 生成
   - 失败时：**丢弃**该 scene cut start，日志 `scene cut ignored: bootstrap failed`

4. **主循环不变**：仍按 `segment_starts` 在固定帧位 reset state。轻量 scene detector 可在主循环跑，仅用于 debug 对比/统计，**不再触发 reset**。

这个方案的优点：
- 主循环零复杂度增量
- bootstrap mask 在 prepass 阶段统一生成（避免 inline 调用 SAM3/YWES）
- min-gap 过滤抑制误检导致的频繁 reset

### 3.3 `config.py` 新增配置项

```python
# PT_MATANYONE2_SCENE_RESET:
#   Enable HSV-Bhattacharyya scene cut detection that forces a MatAnyone2
#   segment reset between scenes. Reuses pipeline.matting._SceneCutDetector.
#   Default on for offline MatAnyone2; ignored elsewhere.
MATANYONE2_SCENE_RESET = _env("MATANYONE2_SCENE_RESET", "1") == "1"

# PT_MATANYONE2_SCENE_THRESHOLD: Bhattacharyya threshold. Default 0.4.
MATANYONE2_SCENE_THRESHOLD = float(_env("MATANYONE2_SCENE_THRESHOLD", 0.4))

# PT_MATANYONE2_SCENE_COOLDOWN: frames between two cut triggers. Default 24.
MATANYONE2_SCENE_COOLDOWN = int(_env("MATANYONE2_SCENE_COOLDOWN", 24))

# PT_MATANYONE2_SCENE_REF_EMA: EMA weight on reference histogram. Default 0.95.
MATANYONE2_SCENE_REF_EMA = float(_env("MATANYONE2_SCENE_REF_EMA", 0.95))

# PT_MATANYONE2_SCENE_MIN_SEGMENT_SEC:
#   Minimum spacing between two segment starts (including scene-cut ones).
#   Suppresses false-positive cuts on quick cross-fades. Default 3.0 s.
MATANYONE2_SCENE_MIN_SEGMENT_SEC = float(_env("MATANYONE2_SCENE_MIN_SEGMENT_SEC", 3.0))
```

### 3.4 SBS 处理

SBS 视频左右眼来自同一原始画面，只对左眼半侧（或者 SBS 整帧）做一次检测即可。开销 ~1 ms/帧，可忽略。

### 3.5 工时：**0.5 d**

---

## 4. G5：MatAnyone2 Alpha Smoother

**优先级：高**（依赖 G4 的 reset 信号闭环跨场景残影）

### 4.1 复用 RVM 的 `_AlphaSmoother`

直接 `from pipeline.matting import _AlphaSmoother`。

### 4.2 接入位置

**4.2.1 `__init__`**

```python
from pipeline.matting import _AlphaSmoother
from config import MATANYONE2_ALPHA_SMOOTH, MATANYONE2_ALPHA_SMOOTH_WEIGHT

self._alpha_smoother = _AlphaSmoother(
    MATANYONE2_ALPHA_SMOOTH_WEIGHT
) if MATANYONE2_ALPHA_SMOOTH else None
```

**4.2.2 在 alpha 输出后接平滑**

MatAnyone2 ONNX 路径有两处 alpha 出口：
- `_run_eye` 末尾返回 `alpha[0, 0]`（`tools/offline_passthrough.py:956`）
- `_run_eyes_batch2` 末尾的 `np.concatenate([alpha[0, 0], alpha[1, 0]], axis=1)`

平滑应该在 alpha **拼接成 SBS 完整 alpha 之前**做，分别对左右眼 state 平滑：

```python
def _smooth_eye_alpha(self, alpha_2d: np.ndarray, eye_idx: int) -> np.ndarray:
    if self._alpha_smoother is None:
        return alpha_2d
    # alpha_2d shape (H, W); reshape to (1, H, W) for batch alignment
    batch = alpha_2d[None, ...]
    smoothed = self._eye_smoothers[eye_idx].step(batch)
    return smoothed[0]
```

**注意**：左右眼必须各持一份 `_AlphaSmoother` 实例（`self._eye_smoothers = [_AlphaSmoother(w), _AlphaSmoother(w)]`），不能共享一个 prev buffer。

### 4.3 与 G4 联动

`_reset_segment` 内必须同步调 smoother reset：

```python
def _reset_segment(self):
    for eye in self.eyes:
        eye.reset()
    for sm in getattr(self, "_eye_smoothers", []):
        if sm is not None:
            sm.reset()
    self._active_segment_start = -1
    self._cached_alpha_sbs = None
```

### 4.4 `config.py` 新增

```python
# PT_MATANYONE2_ALPHA_SMOOTH:
#   Temporal EMA smoothing on MatAnyone2 alpha output. Reset on segment cut.
MATANYONE2_ALPHA_SMOOTH = _env("MATANYONE2_ALPHA_SMOOTH", "1") == "1"

# PT_MATANYONE2_ALPHA_SMOOTH_WEIGHT: EMA weight on previous alpha. Default 0.6.
MATANYONE2_ALPHA_SMOOTH_WEIGHT = float(_env("MATANYONE2_ALPHA_SMOOTH_WEIGHT", 0.6))
```

### 4.5 与 alpha_stride 的互动

当前代码有 `alpha_stride` 缓存机制（`tools/offline_passthrough.py:1025-1029`），即每 N 帧才跑一次 ONNX。如果 `alpha_stride > 1`，缓存帧上不调 smoother（因为没有新数据），但下次更新时按"上一次更新的输出 + 这次新输出"做 EMA 即可（smoother 内部 `_prev` 自然就是上次的 smoothed 值）。**无需额外处理**。

### 4.6 工时：**0.5 d**

---

## 5. G1：GPU IOBinding 改造（消除 PCIe 往返）

**优先级：最高（V2 最大性能杠杆）**

### 5.0 范围收敛（吸收开发者反馈）

**只优化 `step_update` 热路径**，首帧 `image_key` / `mask_memory` / `first_frame_refine` 仍走 `session.run()`（频率低，一个 segment 一次，~50 ms 总开销可接受）。理由：

- 主循环 99% 时间在 step_update，优化收益集中在这一点
- 首帧 graphs 输入张量种类多（features dict、obj_memory、sensory…），全部 OrtValue 化的代码复杂度高
- 先把热路径做对、做稳，再决定是否把 bootstrap graphs 也 GPU-resident

**子阶段拆分**：

| 子阶段 | 内容 | 可独立验收 |
|---|---|---|
| 1A | CuPy ptr → ORT IOBinding probe，跑单次 `step_update` | ✅ |
| 1B | 初始化后上传 state，单眼连续 30 帧 GPU-resident | ✅ |
| 1C | SBS 左右眼顺序跑，GPU alpha concat + composite | ✅ |
| 1D | TensorRT EP + CUDA EP 双 provider 验证 | ✅ |

### 5.1 现状瓶颈

`tools/offline_passthrough.py:899-910`：

```python
def _preprocess_eye(self, x0, eye_w):
    image = self.matter._gpu_preprocess_nv12_one(
        x0, eye_w, self.in_w, self.in_h,
        copy_to_host=True,  # ← 强制 D2H
    )
    return image.astype(self.tensor_dtype, copy=False)
```

每帧都把 GPU 张量复制回 NumPy，再让 ORT 复制回 CUDA。单眼 1024 fp16 image = 6 MB，左右眼 12 MB，加上 9 路 state 张量（memory_key / memory_shrinkage / msk_value / obj_memory / sensory / last_pix_feat / last_mask / last_msk_value，加上 first frame 时还有 pix_feat / key / shrinkage / selection 等 feature dict），每帧 H2D + D2H 总量 ~30-50 MB。在 PCIe Gen3 x16 实测带宽下吃 ~5-10 ms / 帧。

### 5.2 改造策略

**5.2.1 输入端**：复用 `Matter._gpu_preprocess_nv12_one` 的 GPU 输出（CuPy ndarray），用 `OrtValue` 包裹

```python
def _preprocess_eye_gpu(self, x0, eye_w) -> "ort.OrtValue":
    image_cp = self.matter._gpu_preprocess_nv12_one(
        x0, eye_w, self.in_w, self.in_h,
        copy_to_host=False,  # ← 关键
    )
    # image_cp 是 CuPy ndarray (1, 3, H, W) on GPU
    # 转 fp16 / fp32 仍在 GPU 上完成
    if image_cp.dtype != self.tensor_dtype:
        image_cp = image_cp.astype(self.tensor_dtype, copy=False)
    return ort.OrtValue.ortvalue_from_numpy(...)  # ← 不能用 numpy 路径
    # 应该用 OrtValue.ortvalue_from_shape_and_type + ptr 绑定，参考 RVM 实现
```

**5.2.2 State 全部 OrtValue 化**

state 8 路改成 OrtValue 字典：

```python
class _EyeState:
    def __init__(self, runner):
        self.tensors: dict[str, ort.OrtValue] = {}  # name → OrtValue on cuda:0
        self.initialized = False

    def get_or_alloc(self, name: str, shape, dtype) -> ort.OrtValue:
        ov = self.tensors.get(name)
        if ov is None or tuple(ov.shape()) != tuple(shape):
            ov = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, "cuda", 0)
            self.tensors[name] = ov
        return ov
```

**5.2.3 IOBinding 调用**

参考 RVM `_run_rvm_iobinding_from_dev`（`pipeline/matting.py:2103-2158`）：

```python
def _step_update_iobinding(self, image_ortval, state) -> ...:
    binding = self.step_update.io_binding()
    binding.bind_input("image", "cuda", 0, self.tensor_dtype,
                       tuple(image_ortval.shape()), int(image_ortval.data_ptr()))
    for name in ("memory_key", "memory_shrinkage", "msk_value", "obj_memory",
                 "sensory", "last_mask", "last_pix_feat",
                 "last_pred_mask", "last_msk_value"):
        binding.bind_ortvalue_input(name, state.tensors[name])

    # 输出预分配
    for out_name in ("prob", "new_sensory", "new_msk_value",
                     "new_obj_memory", "pix_feat", "logits", "uncert_prob"):
        binding.bind_output(out_name, "cuda", 0)  # TRT 会自动分配

    self.step_update.run_with_iobinding(binding)
    return binding.get_outputs()
```

**5.2.4 alpha 输出全 GPU 直传 composite**

`Matter._composite_nv12_to_nv12_gpu_using_uploaded_frame()` **已经接受 CuPy alpha**（开发者反馈确认）。因此 G1 直接构造 `alpha_eye_dev` `[1024,1024]` CuPy ndarray，左右眼拼成 `alpha_sbs_dev` `[1024,2048]`，无需 D2H。

新增小 CuPy kernel 处理 `prob` `[1,2,1024,1024]` → `alpha_eye_dev` `[1024,1024]`：
- 抽取 foreground channel
- clamp [0, 1]
- dtype 转换（fp16 → fp32 仅当下游需要）

**5.2.5 双缓冲 ping-pong**

ORT IOBinding 不允许输入输出 alias 同一 OrtValue，需要双缓冲：

```python
# state_t -> step_update -> state_{t+1}
state_a = {name: ortvalue_from_shape_and_type(shape, dtype, "cuda", 0) ...}
state_b = {name: ortvalue_from_shape_and_type(shape, dtype, "cuda", 0) ...}
# 每帧 swap 输入/输出绑定
```

### 5.3 batch 策略（吸收开发者反馈）

**V2 优先支持 `matanyone2_onnx_1024_bs1`**，bs2 不作为 V2 主线。理由：

- bs2 IOBinding 需要 batched concat/split GPU kernel，工程量大
- bs1 SBS 按"左眼→右眼顺序跑"在 IOBinding 下已可接近 bs2 吞吐
- V1 已 warmup 两套 TRT engine，bs2 路径仍可走 V1 numpy fallback

bs2 IOBinding 作为 V2.x 后续可选改进，不阻塞主线。

### 5.4 性能验证目标（按开发者反馈调整）

- 1024-bs1 + TRT + IOBinding：`step_update` 单次降低 25%-50%（从 49.7 ms → 25-37 ms）
- 端到端 FPS：**10-14 FPS**（vs V1 base 8 FPS，开发者反馈的实际目标区间）
- PCIe 监控（nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,utilization.memory --format=csv）应能看到带宽利用率下降

### 5.5 回退方案

`config.py` 增加开关：

```python
# PT_MATANYONE2_IOBINDING:
#   Use IOBinding + GPU-resident OrtValue path for MatAnyone2.
#   Falls back to NumPy/copy-to-host path on first failure.
MATANYONE2_IOBINDING = _env("MATANYONE2_IOBINDING", "1") == "1"
```

代码中 try/except 包住 IOBinding 路径，失败一次记 warning 并回退到现有 numpy 路径（参考 RVM 的 `log.warning("[DIAG] RVM IOBinding failed; falling back to sess.run: %s", e)`）。

### 5.6 工时：**2-3 d**（范围收敛后）

- 0.5 d：1A — CuPy ptr → ORT IOBinding probe
- 0.5 d：1B — state OrtValue 化 + ping-pong 双缓冲
- 0.5 d：1C — SBS 左右眼 GPU alpha concat + composite 衔接
- 0.5 d：1D — TRT EP + CUDA EP 双 provider 验证
- 0.5 d：回退路径 + 配置项 + 日志
- 0.5 d：性能 profile + 调优

**bs2 IOBinding / 首帧 graphs IOBinding 不在本次范围**

---

## 6. G3：ROI / person crop（**重大修订**：拆 A 质量 / B 速度）

**优先级：中-低**（G3-A 后置；G3-B 条件触发）

### 6.0 认知纠错（吸收开发者反馈）

**原方案的错误**：声称 G3 通过 ROI 把 attention token 数砍 60-80% 进而"平方级加速"。

**实际情况**：MatAnyone2 ONNX 输入固定 `1024×1024`，**无论原始 ROI 多大，都 letterbox resize 到 1024 后送模型**。token 数 = `(1024/stride)²`，**完全不变**。FLOPs 也不变。

因此 G3 必须按"做什么"重新分类：

| 类型 | 模型输入尺寸 | token 数是否降低 | 真实价值 |
|---|---|---|---|
| **G3-A（1024 ROI crop）** | 1024×1024（letterbox）| ❌ 不变 | 远景人物被放大到满输入，**alpha 质量提升** |
| **G3-B（512 / 768 ROI 模型）**| 512 / 768（V1 保留的 512 容器在此用上）| ✅ 大幅降低 | **真正提速**，但需要权衡质量 |

V2 路径：

1. **先做 G3-A**：仅作为"远景质量功能"，**不承诺 FPS 提升**
2. **G3-A 质量验证通过后**，再决定是否启动 G3-B
3. G3-B 启用时复用 V1 保留的 512 ONNX 文件，UI 仍不暴露 512 选项，仅作为内部 ROI speed mode

### 6.1 G3-A：1024 ROI crop（质量优先）

**bbox 来源（两套前置任一一套，均零额外成本）**：

| 前置 | bbox 字段 | 模块 | 备注 |
|---|---|---|---|
| SAM3（重）| `union_bbox_xyxy` | `offline/sam3_matanyone2.py:254` | 多人合并 union；模型权重 GB 级 |
| YOLOWorld + EfficientSAM（轻，**默认**）| `box_xyxy`（per detection）| `offline/yoloworld_efficientsam.py:23` | 多人需自行 union；模型几十 MB |

由 `--matanyone2-prepass` 选择前置。G3 必须**两套都支持**，不应耦合任一前置的内部结构。

**流程**：

1. 第一帧 / segment 起点：从前置缓存拿 bbox（统一为 `np.ndarray([x0, y0, x1, y1])`）
2. padding 扩展（`expand_factor = 0.25-0.40`）保持正方形
3. ROI letterbox 到模型输入 `1024×1024`
4. MatAnyone2 propagate
5. 输出的 1024² alpha 逆变换回 ROI 原坐标，paste 到 full-eye canvas
6. ROI 外 alpha=0，边界 feather 16-32 px
7. 再走 G2 的 edge-aware upsample 到 8K

**适用 / 禁用规则（吸收开发者反馈）**：

| 情况 | 行为 |
|---|---|
| VR 远景人物占单眼面积 < 20% | ✅ 启用 |
| 单人或主目标稳定 | ✅ 启用 |
| bbox 过大（> 70% 单眼面积）| ⛔ fallback 全眼 |
| bbox 置信度低 / 左右眼不匹配 | ⛔ fallback 全眼 |
| segment 内目标大幅移动 | ⛔ fallback 全眼 或扩大 bbox |

### 6.2 关键工程难点

**6.2.1 ROI 跟随策略（吸收开发者反馈）**

| 策略 | 优势 | 劣势 |
|---|---|---|
| **A. 上一帧 alpha 滚动更新 bbox + EMA** | 跟随快速移动 | 短期抖动会泄露到 alpha |
| **B. 同一 segment 内固定 bbox**（开发者推荐）| memory 坐标系稳定，无抖动 | 人物大幅移动出框需 fallback |

**采纳 B（开发者方案）**：同一 segment 内 bbox 固定，只在 segment reset / scene cut 时按前置 bbox 重锚定。这与 MatAnyone2 memory-based 架构吻合（memory 坐标系不能漂移）。

策略 A 留作未来选项，需配合更复杂的 memory 坐标重映射。

**6.2.2 ROI shape 变化导致 TRT engine 重建**

TRT engine 是 shape-specific。如果 ROI 每帧 shape 都不同：
- 方案 1：**固定 ROI 输出为 1024×1024**（letterbox），shape 固定 → TRT engine 不变
- 方案 2：用 ORT CUDA EP 而不是 TRT EP（动态 shape 支持，性能稍差）

**推荐方案 1**：letterbox 保持 1024² 固定输入。padding 区域用 0 填充，模型会自然学到"边界忽略"。

**6.2.3 ROI 坐标管理**

每帧维护 `roi_x0, roi_y0, roi_w, roi_h, letterbox_pad_x, letterbox_pad_y, scale`，alpha 输出回投到原坐标。

### 6.3 接入位置

新增 `pipeline/matanyone2_roi.py`：

```python
class RoiTracker:
    """Prepass-agnostic ROI tracker.

    Accepts a single normalized bbox (x0, y0, x1, y1) from any source:
    - SAM3   : `union_bbox_xyxy` (offline/sam3_matanyone2.py:254)
    - YWES   : `box_xyxy`        (offline/yoloworld_efficientsam.py:23)
                 — for multi-detection frames, caller computes union first.
    """

    def __init__(self, expand_factor=0.30, model_size=1024,
                 max_eye_fraction=0.70):
        self.expand_factor = expand_factor  # 0.25-0.40 per developer feedback
        self.model_size = model_size
        self.max_eye_fraction = max_eye_fraction  # fallback threshold
        self._segment_roi = None  # (x0, y0, x1, y1), fixed per segment

    def bootstrap(self, bbox_xyxy: np.ndarray, frame_w: int, frame_h: int):
        """Segment-start bbox from whichever prepass is active.

        Returns None when bbox area exceeds max_eye_fraction → caller
        must fallback to full-eye path.
        """
        expanded = self._expand_square(bbox_xyxy, frame_w, frame_h)
        if self._area(expanded) > self.max_eye_fraction * frame_w * frame_h:
            self._segment_roi = None
            return None
        self._segment_roi = expanded
        return expanded

    # Strategy B: segment-fixed ROI; no per-frame update.
    # update_from_alpha intentionally omitted in V2 (see §6.2.1).

    def warp_to_model(self, frame_bgr, roi) -> tuple[np.ndarray, dict]:
        """Crop ROI + letterbox to model_size×model_size; return image + meta."""
        ...

    def unwarp_alpha(self, alpha_model, meta, frame_w, frame_h) -> np.ndarray:
        """Project model_size alpha back to frame coordinates with zero pad."""
        ...
```

**统一 bbox 提取 helper**（消除 G3 对前置模块的耦合），新增到 `tools/offline_passthrough.py`：

```python
def _extract_segment_bbox(prepass_record) -> np.ndarray | None:
    """Return [x0, y0, x1, y1] from either prepass output.

    - SAM3 records carry `union_bbox_xyxy` (already union over all subjects).
    - YWES records carry a list of `Detection.box_xyxy`; union here.
    """
    if prepass_record is None:
        return None
    if hasattr(prepass_record, "union_bbox_xyxy") and len(prepass_record.union_bbox_xyxy) == 4:
        return np.asarray(prepass_record.union_bbox_xyxy, dtype=np.float32)
    if hasattr(prepass_record, "detections") and prepass_record.detections:
        boxes = np.stack([np.asarray(d.box_xyxy, dtype=np.float32)
                          for d in prepass_record.detections], axis=0)
        return np.array([boxes[:, 0].min(), boxes[:, 1].min(),
                         boxes[:, 2].max(), boxes[:, 3].max()], dtype=np.float32)
    return None
```

`MatAnyone2OnnxEngine.composite_nv12` 内加分支：

```python
if config.MATANYONE2_ROI_CROP and self._roi_tracker is not None:
    image_eye_roi, roi_meta = self._roi_tracker.warp_to_model(eye_frame, roi)
    alpha_roi = self._run_eye(image_eye_roi, ...)
    alpha_eye_full = self._roi_tracker.unwarp_alpha(alpha_roi, roi_meta, eye_w, h)
else:
    # 现有全眼 1024 路径
    ...
```

### 6.4 与 G1 IOBinding 的关系

ROI 模式下输入 shape 仍是 1024×1024（letterbox），**所以 G1 IOBinding 完全兼容**。这是为什么 G3 必须在 G1 之后做：ROI 路径需要 GPU 上原地裁剪 + letterbox，否则又退回 CPU。

裁剪 + letterbox 需要 CuPy kernel 或 `cv2.cuda` 实现。考虑到现有 NV12 GPU kernel 已经成熟，建议**新写一个 `Matter._gpu_crop_letterbox_roi_to_eye_input(x0, y0, w, h, target_size)` kernel**，复用 NV12 → fp16 RGB 的现有路径。

### 6.5 `config.py` 新增

```python
# PT_MATANYONE2_ROI_CROP:
#   Crop ROI around the foreground subject for MatAnyone2 inference.
#   Quality-only at fixed 1024 input; does not reduce token count.
#   Experimental; default off.
MATANYONE2_ROI_CROP = _env("MATANYONE2_ROI_CROP", "0") == "1"

# PT_MATANYONE2_ROI_EXPAND:
#   ROI expansion factor over the detected bbox (each side).
#   Default 0.30 per developer recommendation (range 0.25-0.40).
MATANYONE2_ROI_EXPAND = float(_env("MATANYONE2_ROI_EXPAND", 0.30))

# PT_MATANYONE2_ROI_MAX_EYE_FRACTION:
#   Fallback to full-eye path when expanded ROI exceeds this eye fraction.
MATANYONE2_ROI_MAX_EYE_FRACTION = float(_env("MATANYONE2_ROI_MAX_EYE_FRACTION", 0.70))

# PT_MATANYONE2_ROI_SPEED_MODE:
#   G3-B: route ROI to smaller model (512 / 768) for real speedup.
#   Requires G2 edge-aware upsample. Internal only; UI never exposes 512.
MATANYONE2_ROI_SPEED_MODE = _env("MATANYONE2_ROI_SPEED_MODE", "0") == "1"
```

### 6.6 与 V1 RVM 计划的方向冲突

V1 `summary_20260525_RVM_OFFLINE_PRECISION_TIERS_PLAN_CN.md` 第 7 节：

> **Person ROI crop（零模型方案已评估）**：技术可行但与 SBS batch=2 冲突、与场景 reset 冲突、bbox 抖动难控、VR 视频人物占比高收益有限。**本次不做**

那个否决是**针对 RVM**的。换到 MatAnyone2 的修订理由（吸收开发者纠错）：
- RVM 是 RNN，attention 不是 N²，ROI 收益线性
- MatAnyone2 的 G3-A 因为输入仍是固定 1024×1024，**实际上不降低 token 数和 FLOPs**，但远景人物从"占 20% 像素"放大到"占 80% 像素"，**alpha 质量收益明确**（这是与 RVM 决策的真正区别）
- 真正提速需要 G3-B（512/768 模型），由 V1 保留的 512 容器承担
- 抖动通过 segment 内固定 bbox 解决（不再依赖 EMA）
- SBS batch=2 兼容性：letterbox 后输入 shape 固定，没问题
- 与场景 reset 兼容性：segment 重启时重锚定前置 bbox（SAM3 或 YWES 都可），没问题
- **前置无关性**：bbox 来源是 SAM3 还是 YOLOWorld+EfficientSAM 不影响 G3 设计；`RoiTracker.bootstrap(bbox_xyxy, ...)` 接受统一格式，由 `_extract_segment_bbox` helper 适配两种前置输出

**结论**：V2 G3-A 与 V1 RVM 7 节决策**方向相反但理由不同**——RVM 7 节否决是因为"VR 人物占比高收益有限"，G3-A 翻案是因为"远景质量提升是 RVM 不具备的独立维度"。同时 G3 **必须与两套前置（SAM3 + YWES）都兼容**，不得耦合任一前置的内部数据结构。

### 6.7 G3-B：512 / 768 ROI 模型（速度优先，条件触发）

**仅当 G3-A 质量验证通过且远景内容仍需进一步加速时启动**。

候选实现：

| 选项 | 模型 | 代价 |
|---|---|---|
| **B.1**：复用 V1 保留的 `matanyone2_onnx_512_bs1` | 无导出成本 | 512 token 数仅为 1024 的 25%，约 4× 提速；但 alpha 精度依赖更强的 G2 |
| **B.2**：新导出 `matanyone2_onnx_768_bs1` | 一次性导出 + warmup | 速度/质量折中 |

**约束**：

- V1 决策"512 不暴露 UI"仍然有效。G3-B 仅作为**内部 ROI speed mode**
- UI 选择项继续保留 1024 单档，G3-B 通过环境变量 `PT_MATANYONE2_ROI_SPEED_MODE=1` 启用
- G3-B 启用时强制要求 G2 (edge-aware upsample) 同时打开，否则 alpha 质量不够

G3-B 设计细节留待 G3-A 验证后专门补一份小 patch 计划，不在本 V2 主线工时内。

### 6.8 工时（仅 G3-A）：**1.5-2 d**

- 0.5 d：`RoiTracker` 类（策略 B 固定 bbox）+ 单元测试
- 0.5 d：CuPy crop + letterbox kernel
- 0.5 d：unwarp alpha + feather paste + 黑背景填充
- 0.5 d：与 G1 IOBinding 衔接 + fallback 规则
- 0.5 d：验证 + 调参（expand_factor / max_eye_fraction）

**G3-B 工时单独评估**：2-4 d，3A 审核通过后立项

---

## 7. G2：Edge-aware Alpha Upsampler

**优先级：中**（独立项，最后做）

### 7.1 现状

MatAnyone2 输出 1024² alpha。当前上采样到 8K eye（4096²）走 `pipeline/matting.py:_COMPOSITE_UPSAMPLE_KERNEL_SRC:165` 的 bilinear。bilinear 在硬边界（人体轮廓、发丝）上会产生 4 像素宽的模糊带，吃掉 MatAnyone2 的边缘细节优势。

### 7.2 算法选型

| 算法 | GPU 友好度 | 边缘锐度 | 实现复杂度 | 推荐度 |
|---|---|---|---|---|
| Bilinear（现状） | ✅ | ❌ | 0 | basleine |
| Fast Guided Filter (FGF) | ✅ | ✅ | 中 | **首选** |
| Joint Bilateral Upsample | ⚠️（取决于实现） | ✅ | 中-高 | 备选 |
| Deep Guided Filter (DGF) | ❌ 需要模型 | ✅ | 高 | 否决（额外模型） |
| Bilateral grid upsample | ⚠️ | ✅ | 高 | 否决（复杂度） |

**采纳 Fast Guided Filter**：
- 复杂度 O(N)，可 box-filter 实现
- 已有现成 CuPy / CUDA kernel 实现可参考
- RVM 已用同类技术（其内部的 stage2 上采样就是 FGF）

### 7.3 接入位置（**guide 选型修订**：用 NV12 Y 平面，吸收开发者反馈）

新增 `pipeline/alpha_guided_filter.py`（命名与开发者方案对齐）：

```python
def fast_guided_filter_upsample(
    alpha_lr_gpu: cp.ndarray,    # (H_lr, W_lr) on GPU, [0, 1]
    guide_y_hr_gpu: cp.ndarray,  # (H_hr, W_hr) NV12 luma (Y) on GPU, [0, 255]
    radius: int = 8,
    eps: float = 0.0025,
) -> cp.ndarray:
    """Fast Guided Filter alpha upsampling using NV12 Y plane as guide.

    Returns alpha_hr on GPU shape (H_hr, W_hr).

    Why Y plane instead of BGR?
    - NV12 Y is already on GPU (uploaded for composite); zero extra upload.
    - Single-channel guide halves box-filter cost vs 3-channel BGR.
    - Luma carries the edge structure needed for alpha refinement.
    """
    # 1. Downsample Y to alpha_lr resolution
    # 2. Compute mean_I, mean_p, mean_Ip, var_I in low res via box_filter
    # 3. a = (mean_Ip - mean_I * mean_p) / (var_I + eps)
    # 4. b = mean_p - a * mean_I
    # 5. Upsample a, b to hr (bilinear)
    # 6. alpha_hr = a * Y_hr + b ; clamp [0, 1]
    ...
```

实现要点：
- **guide 直接复用已上传的 NV12 Y 平面**（无需 BGR 转换、无需额外 H2D）
- box_filter 用 CuPy 实现（separable moving-average kernel，未来可优化为 integral image）
- 整个过程 GPU-resident，与 G1 IOBinding 输出的 CuPy alpha 衔接
- 对 SBS：左右眼各跑一次（或拼成 1024×2048 一起跑）
- **初版只支持 NV12 8-bit**；P016（10-bit）先 fallback 到 bilinear，待后续 patch

### 7.4 调用位置

`tools/offline_passthrough.py:1030-1043` 当前的 alpha 输出后：

```python
if config.MATANYONE2_EDGE_AWARE_UPSAMPLE and self._upsampler is not None:
    # alpha_lr_gpu: 1024×2048 SBS alpha on GPU
    # guide_hr_gpu: NV12 → BGR on GPU at 8K SBS
    alpha_hr_gpu = self._upsampler(alpha_lr_gpu, guide_hr_gpu)
    composite_input = alpha_hr_gpu  # 直接喂 composite kernel
else:
    composite_input = self._cached_alpha_sbs  # 现有 bilinear 路径
```

注意 composite kernel 当前接受 alpha shape 是 1024×2048（模型输出尺寸），内部做 bilinear upsample 到 NVENC 输入。FGF 模式下 alpha 已经在 8K，需要 composite kernel 跳过它的 bilinear 段。新增 kernel 变体 `_composite_nv12_with_full_res_alpha`。

### 7.5 性能预算（开发者反馈对齐）

| 分辨率 | 目标额外耗时 |
|---|---:|
| 4K SBS | ≤ 3 ms |
| 8K SBS | ≤ 8 ms |

如 8K kernel 慢于预算，回退到 half-res refine（`PT_MATANYONE2_GUIDED_FULLRES_SCALE=0.5`）。

### 7.6 `config.py` 新增

```python
# PT_MATANYONE2_EDGE_AWARE_UPSAMPLE:
#   Use Fast Guided Filter to upsample MatAnyone2 alpha to source resolution
#   with edge-aware refinement.
MATANYONE2_EDGE_AWARE_UPSAMPLE = _env("MATANYONE2_EDGE_AWARE_UPSAMPLE", "1") == "1"

# PT_MATANYONE2_GUIDED_RADIUS: Guided filter box radius. Default 8.
MATANYONE2_GUIDED_RADIUS = int(_env("MATANYONE2_GUIDED_RADIUS", 8))

# PT_MATANYONE2_GUIDED_EPS: Guided filter regularization epsilon. Default 0.0025.
MATANYONE2_GUIDED_EPS = float(_env("MATANYONE2_GUIDED_EPS", 0.0025))

# PT_MATANYONE2_GUIDED_FULLRES_SCALE:
#   Scale at which to compute guided filter coefficients before upsampling.
#   1.0 = full-res alpha refine; 0.5 = half-res refine then upsample (fallback
#   for slow GPUs).
MATANYONE2_GUIDED_FULLRES_SCALE = float(_env("MATANYONE2_GUIDED_FULLRES_SCALE", 1.0))
```

### 7.7 工时：**2 d**

- 1 d：FGF CuPy kernel 实现 + 验证
- 0.5 d：composite kernel 新增 full-res alpha 入口
- 0.5 d：性能 profile + 调参

---

## 8. 文件级修改清单总览（V2 修订版）

| 文件 | G0 | G1 | G2 | G3-A | G4 | G5 |
|---|---|---|---|---|---|---|
| `config.py` | + profile 开关 | + IOBinding 开关 | + 3 项 guided | + 4 项 ROI | + 5 项 scene | + 2 项 smooth |
| `tools/offline_passthrough.py` | 削薄（保留 CLI / IO）| 接入 engine | 接入 | 接入 | 接入 | 接入 |
| `tools/offline_alpha_passthrough.py` | 削薄（保留 CLI / IO）| — | — | — | — | — |
| `offline/matanyone2_engine.py` | **新增**（含 engine 抽取）| 大改 IOBinding | 接入 | 接入 | 接入 | 接入 |
| `pipeline/matting.py` | — | composite 已支持 CuPy alpha | + full-res alpha 入口 | + ROI crop/letterbox kernel | （无） | （无） |
| `pipeline/matanyone2_roi.py` | — | — | — | **新增** | — | — |
| `pipeline/alpha_guided_filter.py` | — | — | **新增** | — | — | — |
| `utils/scene_detection.py` | — | — | — | — | **新增**（从 `pipeline/matting._SceneCutDetector` 抽出）| — |
| `pipeline/matting.py` 已有的 `_AlphaSmoother` | — | — | — | — | — | 复用 |
| `tests/test_matanyone2_engine.py` | **新增** | — | — | — | — | — |
| `tests/test_alpha_guided_filter.py` | — | — | **新增** | — | — | — |
| `tests/test_scene_detection.py` | — | — | — | — | **新增** | — |
| `summary/summary_20260525_RVM_OFFLINE_PRECISION_TIERS_PLAN_CN.md` | — | — | — | 加附注：MatAnyone2 G3-A 翻案理由不同 | — | — |

**新增源文件 4 个**（matanyone2_engine / matanyone2_roi / alpha_guided_filter / scene_detection），**新增测试 3 个**，修改 5 个，更新 1 个 summary。

---

## 9. 验证清单

### 9.1 单项验证（每节完成后）

**G4 场景检测**：
- [ ] 日志出现 `MatAnyone2 scene cut detected`
- [ ] 跨场景第 1-3 帧无残影
- [ ] 长镜头不误触发
- [ ] 淡入淡出转场不连续触发

**G5 Alpha smoother**：
- [ ] 同场景内静止镜头 alpha 抖动减小（人眼可辨）
- [ ] 与 G4 reset 联动：场景切换瞬间 smoother 重置
- [ ] alpha_stride > 1 时无异常

**G1 IOBinding**：
- [ ] step_update 单次 < 30 ms（vs 49.7 ms）
- [ ] 端到端 FPS ~15（vs 8）
- [ ] bs1 / bs2 均正常
- [ ] 关闭 `PT_MATANYONE2_IOBINDING=0` 后回退到 numpy 路径
- [ ] 单次 IOBinding 失败自动回退并 warning 打印

**G3 ROI crop**：
- [ ] 远景 VR 测试视频（人物占画面 < 30%）FPS 翻倍
- [ ] 近景测试视频 FPS 与全眼模式接近（pad_factor=1.5 时 ROI ≈ 全眼）
- [ ] ROI 跟随人物移动无掉框
- [ ] 与 SBS batch=2 兼容
- [ ] 与 G4 segment reset 兼容（reset 时回前置 bbox 锚定）
- [ ] **`--matanyone2-prepass=sam3` 路径下 G3 正常工作**（用 `union_bbox_xyxy`）
- [ ] **`--matanyone2-prepass=yoloworld_efficientsam` 路径下 G3 正常工作**（用 `box_xyxy` 多框 union）
- [ ] 切换前置不需要修改 `RoiTracker` 内部代码

**G2 Edge-aware upsample**：
- [ ] 发丝边缘锐度肉眼优于 bilinear
- [ ] 与 RVM `2048+FGF` 对比，质量接近或反超
- [ ] 性能开销 < 10 ms / 帧

### 9.2 端到端验证（吸收开发者 5 档矩阵）

固定视频：`videos/test_8k.mp4`，15 秒，8K SBS（含 ≥3 场景切换 + 多人 + 发丝）。

| Case | 开关 | 期望端到端 FPS | 期望质量变化 |
|---|---|---|---|
| **V1 baseline** | 当前 1024+TRT | 8 | baseline |
| **IOBinding only** | `PT_MATANYONE2_IOBINDING=1` | 10-14 | 与 V1 一致 |
| **IOBinding + guided** | + `PT_MATANYONE2_EDGE_UPSAMPLE=1` | 10-13 | 边缘锐度提升 |
| **Full V2（无 ROI）**| + `PT_MATANYONE2_SCENE_RESET=1 PT_MATANYONE2_ALPHA_SMOOTH=1` | 10-13 | 跨场景残影消除 + 抖动减小 |
| **Full V2 + G3-A** | + `PT_MATANYONE2_ROI_CROP=1` | 10-13（不承诺提速）| 远景人物质量提升 |

记录指标（沿用 G0 新增的 profile 字段）：

- 端到端 FPS、显存峰值
- `matanyone2_preprocess_eye_avg`
- `matanyone2_step_update_avg`
- `matanyone2_iobinding_copy_avg`
- `matanyone2_guided_upsample_avg`
- `matanyone2_composite_avg`

### 9.2.1 质量对比截图（开发者建议）

输出对比图（同一帧）：
- 原图 crop
- V1 bilinear alpha
- V2 guided alpha
- V2 guided + smoother alpha
- RVM balanced/hq 参考

重点观察：
- 头发/手指/衣服边缘锐度
- alpha 是否吸入背景纹理（FGF 风险）
- scene cut 后是否残留上一场景轮廓
- EMA 是否拖影
- ROI-A 是否提升远景边界 vs 放大误差

### 9.3 FP16 监测（继续 V1 4.3 标准）

V2 五项任何一项启用后，仍需监控：
- [ ] alpha 通道整体不变灰
- [ ] 长 segment 末尾不失真
- [ ] 边缘无溶解

如触发，**先关 G3 再关 G1**（这两项最可能放大 FP16 误差），最后才考虑 FP32 全面回退。

---

## 10. 风险与回退

### 10.1 G1 IOBinding 风险

| 风险 | 触发 | 回退 |
|---|---|---|
| OrtValue 与 CuPy 互操作 bug | 启动报错 | `PT_MATANYONE2_IOBINDING=0` |
| TRT engine 与 IOBinding 不兼容 | first run 失败 | 自动 fallback 到 CUDA EP + IOBinding |
| state shape 不匹配导致显存爆 | OOM | 关闭 IOBinding，回 numpy 路径 |

### 10.2 G3 ROI 风险

| 风险 | 触发 | 回退 |
|---|---|---|
| ROI 跟丢人物（快速移动）| alpha 错误覆盖 | EMA 调更小 / pad_factor 调更大 |
| 多人场景 ROI 包不全 | 部分人物被裁 | 关 G3 |
| bbox 抖动导致背景闪烁 | 视觉抖动 | EMA 调更大（0.85+） |

### 10.3 G2 FGF 风险

| 风险 | 触发 | 回退 |
|---|---|---|
| FGF 引入光晕（halo）| 强光梯度场景 | 减小 radius / 减小 eps |
| 上采样模糊感反而增强 | alpha 本身就模糊 | 关 G2 退回 bilinear |

### 10.4 G4/G5 风险

| 风险 | 触发 | 回退 |
|---|---|---|
| 场景检测过敏 | 频繁 reset 导致 mask 反复 bootstrap | 阈值调高 0.5+ |
| Alpha smoother 残影 | EMA 权重过大 | weight 调到 0.4-0.5 |

### 10.5 集中开关

最坏情况一次性关闭所有 V2 增强，回到 V1 base：

```bash
PT_MATANYONE2_IOBINDING=0
PT_MATANYONE2_ROI_CROP=0
PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0
PT_MATANYONE2_SCENE_RESET=0
PT_MATANYONE2_ALPHA_SMOOTH=0
```

---

## 11. 工时估算与执行顺序（V2 修订版，吸收开发者反馈）

| 阶段 | 内容 | 工时 |
|---|---|---|
| 0 | **G0 共享 engine 抽取** + V2 前基准 | 0.5-1 d |
| 1 | **G1 IOBinding (step_update only)** 子阶段 1A-1D | 2-3 d |
| 2 | **G2 Guided Filter Upsampler**（NV12 Y guide）| 2-3 d |
| 3 | **G4 Scene cut → prepass plan merge** | 1-1.5 d |
| 4 | **G5 Alpha smoother**（复用 RVM 类）| 0.5 d |
| 5 | **G3-A 1024 ROI crop**（质量，固定 bbox 策略 B）| 1.5-2 d |
| 6 | **端到端 5 档矩阵验证** | 1.5 d |
| **V2 主线合计** | | **9-12.5 d** |
| —（条件）| **G3-B 512/768 ROI speed mode**（仅 3A 通过后立项）| 2-4 d |

可分三阶段交付：

| Stage | 内容 | 工时 | 交付价值 |
|---|---|---|---|
| **Stage 1** | G0 + G1 + G2 | 4.5-7 d | 性能提升至 10-14 FPS + 边缘锐度提升 |
| **Stage 2** | G4 + G5 | 1.5-2 d | 跨场景稳定性 + 时序平滑 |
| **Stage 3** | G3-A | 1.5-2 d + 1.5 d 验证 | 远景质量提升（不承诺提速）|
| **Stage 4（条件）**| G3-B | 2-4 d | 真实远景加速 |

> 与旧版相比的主要差异：
> - 总工时维持 ~10 d，但**前置 G0** 避免后期改两份代码
> - G1 收敛到 step_update only，工时不降但风险大幅降低
> - G3 拆分后 G3-A 工时下调（策略 B 比策略 A 简单）
> - G3-B 显式从主线移出，作为条件触发的后续 patch

---

## 12. 不在 V2 范围内（明确推迟）

- ❌ **MatAnyone2 FP32 全面回退**：仅在 FP16 出问题时触发
- ❌ **2048 ONNX 重新启用**：V1 已永久否决
- ❌ **改实时 RVM / 改 RVM 离线精度档 / 重新训练或微调 MatAnyone2**：硬性边界
- ⏸ **bs2 IOBinding**：V2 主线只支持 bs1；bs2 走 numpy fallback，作为 V2.x 后续 patch
- ⏸ **首帧 graphs IOBinding**（image_key / mask_memory / first_refine）：本次只动 step_update 热路径
- ⏸ **1280 / 1536 中间档**：V1 第 7 节已推迟，本次仍不做
- ⏸ **G3-B（512/768 ROI speed mode）**：3A 验证通过后单独立项，复用 V1 保留的 512 容器
- ⏸ **多目标 / 多人独立 mask**：当前 ONNX 导出只支持 1 object，需要重新导出
- ⏸ **完整动态 shape MatAnyone2 ONNX 导出**：不做
- ⏸ **alpha 写盘缓存（offline GT mode）**：V2 全部在线计算
- ⏸ **MatAnyone2 实时模式**：永远不做
- ⏸ **MQE / Alpha Fusion**（训练流程）：不引入

---

## 13. 与既有计划的关系

| 既有计划 | 关系 |
|---|---|
| `summary_20260526_MATANYONE2_OFFLINE_OPTIMIZATION_PLAN_CN.md`（V1）| V2 接续，V1 已完成 |
| `summary_20260525_RVM_OFFLINE_PRECISION_TIERS_PLAN_CN.md`（RVM 计划）| **复用其 F/G 节实现的 `_SceneCutDetector` 和 `_AlphaSmoother`**；G3 ROI 决策与该计划第 7 节相反，需附注翻案理由 |
| `summary_20260526_MATANYONE2_RESEARCH_CN.md`（研究报告）| V2 是研究报告里"1024 + edge-aware upsample + ROI"建议的工程落地 |

---

## 14. 成功标准

V2 完成后，**MatAnyone2 离线 1024 模式**应满足（按开发者反馈修订）：

1. **性能**：端到端 **10-14 FPS**（vs V1 base 8 FPS）；G3-A 不承诺提速；G3-B 启用后远景内容才会进一步加速
2. **质量**：8K 画面边缘锐度不低于 RVM `2048+FGF`
3. **稳定性**：跨场景无残影、长 segment 无明显漂移
4. **兼容性**：FP16 全程稳定；**bs1 IOBinding 主线**，bs2 走 numpy fallback
5. **可回退**：6+1 项均有独立开关，任一项可关闭回 V1 base
6. **可观察**：日志包含 `matanyone2_iobinding`、`matanyone2_alpha_refine`、scene cut 计数、ROI 跟随状态、profile 明细字段

达到以上标准后，MatAnyone2 才真正具备"离线质量优先工具"的产品定位。

---

**END**
