# RVM 离线模式精度档位重构 — 实施文档 v2

**日期**：2026-05-25（v2 增补 ①② 时序优化）
**作者**：架构调研结论（Claude 协同）
**目标读者**：执行本次重构的工程师
**v2 变更**：追加 F 节（场景检测 + RVM rec reset）、G 节（AlphaSmoother α=0.6，UI 不暴露）；Person ROI 维持推迟
**v2.1 变更**：F 节算法升级为 HSV (H,S) 二维直方图 + Bhattacharyya 距离 + 参考帧 EMA 慢更新 + 冷却窗口（替换原灰度 Pearson 方案）
**关联报告**：
- `summary/summary_20260520_RVM_INPUT_SIZE_DOWNSAMPLE_FPS_CN.md`（FPS 基准）
- `summary/summary_20260525_RVM_RESEARCH_CN.md`（综合调研）
- `CHANGELOG.md` line 57 / 200（FP16 alpha 灰条记录，影响本方案决策）

---

## 1. 背景

当前离线 RVM 配置：
- 输入 1024×1024，`downsample_ratio = 0.5` → stage1 编码器在 512×512 运行
- 模型可选 `rvm_fast` (MobileNetV3 fp32) / `rvm_balanced` (ResNet50 fp32)
- SBS 视频在每次前向把左右眼合并成 `batch=2`（`MATTING_SBS_BATCH=1`）

FPS 基准报告里有一个**被忽略的甜点**：`2048 + 0.25` → stage1 同样在 512×512 跑（编码器开销基本相同），但 stage2 Fast Guided Filter 在 2048² 上做精修，对头发/边缘等高频细节有明显增益，ORT 延迟仅从 15.55 ms 升至 24.08 ms。

ResNet50 在多次实测中相对 MobileNetV3 增益有限，不值得占用第二份 TRT 引擎缓存。

FP16 已被 CHANGELOG 明确排除：RVM 的 rec1–rec4 RNN 状态在 FP16 下逐帧累积精度误差，导致 alpha 通道在某些视频里逐渐变灰。**本次重构所有 RVM 路径强制 fp32**。

---

## 2. 总体决策

### 2.1 离线 UI 暴露"处理精度"档位（替代裸暴露 input_size/ratio）

| 档位 | input_size | downsample_ratio | stage1 实际像素 | FGF 精修分辨率 | 适用 |
|---|---|---|---|---|---|
| **低（fast）** | 1024 | 0.5 | 512² | 1024² | 快速预览、低显存兜底 |
| **中（balanced，默认）** | 2048 | 0.25 | 512² | 2048² | 出片首选 |
| **高（hq）** | 2048 | 0.5 | 1024² | 2048² | 极致细节（编码器超出训练域 256–512，谨慎使用） |

### 2.2 删除 ResNet50（`rvm_balanced` 引擎）

ResNet50 实测增益不足以独立成档。完全移除，**不保留向后兼容**。模型文件 `rvm_resnet50_fp32.onnx` 由用户自行清理。

### 2.3 实时模式**完全不动**

继续 `1024 + 0.5 + SBS-batch=2`。所有改动只影响离线代码路径与 UI 离线面板。

### 2.4 离线 TRT 引擎缓存 — 4 套全集

| Engine | input_size | batch | 用途 |
|---|---|---|---|
| `rvm_mnv3_1024_bs1` | 1024 | 1 | 单眼 2D / 低精度 + 非 SBS |
| `rvm_mnv3_1024_bs2` | 1024 | 2 | 低精度 + SBS 双眼合 batch |
| `rvm_mnv3_2048_bs1` | 2048 | 1 | 中/高精度 + 单眼 2D 或显存兜底 |
| `rvm_mnv3_2048_bs2` | 2048 | 2 | 中/高精度 + SBS 双眼合 batch（首发主力） |

注：实时模式现有 `1024×512×bs1/bs2` 引擎不动，与本批离线引擎在 `runtime_cache/trt_engines/` 同目录共存（ORT TRT EP 用 onnx hash + shape 作文件名，互不冲突）。

---

## 3. 文件级修改清单

> 路径均相对 `D:\p\PTServer\`。以下顺序即推荐改动顺序。

### A. 移除 ResNet50（`rvm_balanced`）

**A.1 `offline/convert.py`**
- 第 31-36 行 `ENGINES` 字典：删 `"rvm_balanced"` 整行
- 第 38-43 行 `ENGINE_TAGS` 字典：删 `"rvm_balanced"` 整行
- 全文件搜 `rvm_balanced` / `resnet50` / `balanced`，删干净

**A.2 `tools/offline_passthrough.py` & `tools/offline_alpha_passthrough.py`**
- 搜 `resnet50` / `rvm_balanced`，删任何分支
- argparse 选项中若有 `--rvm-variant`、`--rvm-model` 等，去掉 `balanced`/`resnet50` 选项

**A.3 `ui/pages/offline_page.py`**
- 第 163-167 行 `_engine_combo` 当前只有 `rvm_fast` 和 `matanyone2`，**已无 ResNet50**，无需改 ✓
- 如其他 UI 文件出现 `rvm_balanced`，一并清理

**A.4 `config.py`**
- 第 230 行 `MODEL_PATH` 默认 `rvm_mobilenetv3_fp32.onnx`，**已正确**，无需改 ✓

**A.5 TRT manifest 与缓存**
- 搜 `utils/trt_manifest.py` 等位置是否硬编码引用 resnet50 / rvm_balanced，移除
- 手动删除 `runtime_cache/trt_engines/` 下名字含 `resnet50` 的 `.engine`（如果有）

**A.6 测试**
- 全仓 grep `resnet50` + `rvm_balanced` + `rvm2`（ENGINE_TAGS 缩写）确认零引用
- `tests/test_offline_convert.py` 若 mock 了 `rvm_balanced`，改成 `rvm_fast` 或删

---

### B. 离线模式默认值改为"中精度（2048 + 0.25）"

**B.1 `tools/offline_passthrough.py`**
```python
# 第 1538-1539 行
parser.add_argument("--input-size", type=int, default=2048, ...)        # 1024 → 2048
parser.add_argument("--rvm-downsample-ratio", type=float, default=0.25, ...)  # 0.5 → 0.25
```

**B.2 `tools/offline_alpha_passthrough.py`** — 同上同位（约 line 1518 附近，结构一致）

**B.3 `offline/convert.py`**
```python
# 第 45-52 行 RVM_DEFAULT_ARGS
RVM_DEFAULT_ARGS = {
    "input_size": 2048,            # 1024 → 2048
    "downsample_ratio": 0.25,      # 新增字段（如果原来没有）
    ...
}
```
然后 `_tool_command` 内（line 189-191 附近）把 `--rvm-downsample-ratio` 也透传出去。

**B.4 `config.py`**
- **不改** `RVM_DOWNSAMPLE_RATIO` (line 267) 与 `MATTING_INPUT_SIZE` (line 367) 的默认值——这两个是**实时模式**的运行时配置，按 2.3 决策保持 `1024 / 0.5` 不动。
- 离线模式通过命令行参数显式覆盖 `config.MATTING_INPUT_SIZE` 和 `config.RVM_DOWNSAMPLE_RATIO`（`offline_passthrough.py:1552-1555` 现有逻辑已支持）。

---

### C. 离线 UI 暴露"处理精度"档位下拉

**C.1 `ui/pages/offline_page.py`**

在 `_engine_combo` 之后新增方法：
```python
def _precision_combo(self) -> QComboBox:
    combo = _fit_combo(QComboBox())
    # data 用 tuple (input_size, downsample_ratio) 便于直接消费
    combo.addItem("低 (1024 / 0.5) — 预览/低显存", ("fast", 1024, 0.5))
    combo.addItem("中 (2048 / 0.25) — 推荐",       ("balanced", 2048, 0.25))
    combo.addItem("高 (2048 / 0.5) — 极致细节",    ("hq", 2048, 0.5))
    combo.setCurrentIndex(1)  # 默认中
    return combo
```

在 single / batch 面板各新增一个 `self.single_precision`、`self.batch_precision` 实例（参考第 254 / 319 行 engine_combo 的布局方式）。

仅当 `engine == "rvm_fast"` 时该下拉可见可用；切到 `matanyone2` 时灰显或隐藏（参照第 263-265 行 `_update_recognition_visibility` 的联动写法）。

**C.2 把档位传给 offline runner**

找到 `_build_command` / `_run_single` / `_run_batch` 等组装命令行的位置（offline_page.py 内），把 `(input_size, ratio)` 转成：
```python
cmd.extend(["--input-size", str(input_size)])
cmd.extend(["--rvm-downsample-ratio", str(ratio)])
```

`offline/convert.py:_tool_command` 已能透传，注意把 `args.input_size` / `args.rvm_downsample_ratio` 透传链路打通。

**C.3 国际化文案**
- 若 UI 走 `tr()`，新增三档的翻译 key
- 默认中文文案如上表 2.1，英文文案：`low / balanced / hq`

---

### D. 离线 TRT 引擎缓存 — 4 套预生成

**D.1 新增预热脚本 `tools/warmup_offline_trt.py`**

参考 `tools/warmup_gpu_cache.py` 写一个离线专用版，关键差异：循环跑 4 组 shape，让 ORT TRT EP 把 4 个 engine 文件 build 出来：

```python
SHAPES = [
    # (input_size, batch)
    (1024, 1),
    (1024, 2),
    (2048, 1),
    (2048, 2),
]
```

对每组 shape：
1. 设 `config.MATTING_INPUT_SIZE = input_size`
2. 设 `config.MATTING_SBS_BATCH = (batch == 2)` 以及 `config.MATTING_SPLIT_SBS = 1`
3. 用 `get_matter(warmup_runs=0)` 拿到 Matter 实例
4. 喂 `runs_per_shape` 次（建议 2）dummy frame，触发 ORT TRT EP build engine
5. dispose Matter，进入下一组（避免 ORT session 占住 GPU）

预期产物：`runtime_cache/trt_engines/` 多出 4 个 `*.engine`（命名由 ORT TRT EP 自动 hash），加一个统一 marker 文件 `offline_trt_engines.marker.json` 记录 4 组 shape + onnxruntime 版本 + onnx 文件 sha256。

**D.2 UI 触发入口**

在 `ui/pages/offline_page.py` 顶部加一个"生成离线 TRT 缓存"按钮（或在设置页 `ui/settings.py` 加入口）：
- 点击后弹进度条对话框
- 后台 QProcess 调 `python tools/warmup_offline_trt.py`，把 stdout 行喂进度
- 完成后刷新 `_update_trt_cache_rows()` 显示"4/4 已就绪"

**D.3 启动时不要阻塞实时 warmup**

主程序 `main.py` / `app.py` 的 `STARTUP_GPU_WARMUP` 流程**只 warmup 实时 1024**，**不**自动 build 离线 4 套（避免冷启动多花数分钟）。离线 4 套由 D.2 的 UI 按钮按需触发。

**D.4 引擎冲突检查**

ORT TRT EP 用 onnx hash + shape 作 engine 文件名，所以 4 套 + 实时 2 套（共 6 个 engine 文件）能在同一缓存目录共存。验证：
```bash
ls runtime_cache/trt_engines/*.engine | wc -l
# 期望 ≥ 6（实时 1024-bs1 + 1024-bs2 + 离线 4 套；如果实时和离线 1024 共享 engine，则为 4）
```

实际上**实时和离线 1024 系列的 engine 在 onnx + shape 完全一致时会自动复用**——这是好事，意味着离线 warmup 也顺便覆盖实时。最终缓存大小：
- 共享情况下：1024-bs1, 1024-bs2, 2048-bs1, 2048-bs2 共 4 个 engine
- mobilenet fp32 模型一个 engine 文件约 50–80 MB → 总缓存 ≈ 250–350 MB

---

### F. 场景检测 + RVM rec reset（v2 新增）

**目标**：跨场景切换时清掉 RVM 的 rec1–rec4 RNN 状态，避免旧场景上下文污染新场景 alpha；同时为 G 节 AlphaSmoother 提供 reset 信号。

**F.1 算法**：**HSV (H,S) 二维直方图 + Bhattacharyya 距离 + 参考帧 EMA 慢更新 + 冷却窗口**。

> 算法选型经过横向对比（详见本节末尾"算法选型说明"）。最终方案融合 VRAutoMatte 的"参考帧 + 冷却"思路，将统计量升级为更适合 VR 内容的 HSV Bhattacharyya，并预降采样以摊薄 8K 输入开销。

要点：
- **HSV (H,S) 二维直方图**：H 维度 30 bin（色相 0–180）、S 维度 32 bin（饱和度 0–256），对色温/色调突变（夜→日、暖→冷光）远比灰度敏感
- **Bhattacharyya 距离** `cv2.HISTCMP_BHATTACHARYYA`：取值 [0, 1]，0=完全相同，1=完全不同；阈值默认 **0.4**（>0.4 判为切换）
- **参考帧 EMA 慢更新**（0.95 老 + 0.05 新）：避免长镜头内累积漂移误触发；又避免"只比上一帧"导致渐变镜头漏检
- **冷却 24 帧** ≈ 1 秒 @24fps：避免淡入淡出/转场序列内连续触发多次 reset（reset 频繁本身有害于 rec 学习）
- **预降采样到 540p**：8K 算 hist 没必要满分辨率，缩到 540 高再算可降 10× 开销

**F.2 实现位置**：`pipeline/matting.py` Matter 类（RVM 分支）

新增 `_SceneCutDetector` 内嵌小类：
```python
class _SceneCutDetector:
    """HSV-Bhattacharyya scene cut detector with reference-frame EMA and cooldown."""

    def __init__(
        self,
        threshold: float = 0.4,      # Bhattacharyya 距离；> threshold 判为切换
        cooldown_frames: int = 24,   # 触发后 N 帧内不再触发
        ref_ema_alpha: float = 0.95, # 同场景内参考帧慢更新系数
        downsample_height: int = 540,
    ):
        self.threshold = float(threshold)
        self.cooldown = int(cooldown_frames)
        self.ref_ema_alpha = float(ref_ema_alpha)
        self.downsample_height = int(downsample_height)
        self._ref_hist = None
        self._cooldown_left = 0

    def step(self, frame_bgr: np.ndarray) -> bool:
        """Returns True iff a scene cut is detected at this frame."""
        # 降采样到 540p 加速 (8K 不需要满分辨率算 hist)
        h, w = frame_bgr.shape[:2]
        if h > self.downsample_height:
            new_w = w * self.downsample_height // h
            frame_bgr = cv2.resize(
                frame_bgr, (new_w, self.downsample_height),
                interpolation=cv2.INTER_AREA,
            )
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [30, 32], [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

        if self._ref_hist is None:
            self._ref_hist = hist
            return False

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            # 冷却期内也用慢更新跟随当前场景，避免冷却结束后参考帧滞后
            self._ref_hist = (
                self._ref_hist * self.ref_ema_alpha
                + hist * (1.0 - self.ref_ema_alpha)
            )
            return False

        dist = float(cv2.compareHist(
            self._ref_hist, hist, cv2.HISTCMP_BHATTACHARYYA
        ))
        if dist > self.threshold:
            self._ref_hist = hist
            self._cooldown_left = self.cooldown
            return True
        # 同场景内慢更新参考帧
        self._ref_hist = (
            self._ref_hist * self.ref_ema_alpha
            + hist * (1.0 - self.ref_ema_alpha)
        )
        return False

    def reset(self):
        self._ref_hist = None
        self._cooldown_left = 0
```

**性能预算**：
- 8K (7680×3840) 降采样到 540p ≈ 0.6 ms (INTER_AREA)
- HSV 转换 + 2D hist (30×32) + 归一化 ≈ 0.3 ms
- Bhattacharyya 距离 ≈ 0.05 ms
- **合计 < 1 ms / 帧**（CPU 单核），完全可接受

### F.1.x 算法选型说明（保留供 review）

| 候选 | 开销 | VR 适配 | 否决 / 采纳理由 |
|---|---|---|---|
| 灰度 hist + Pearson（VRAutoMatte 原版） | ~1 ms | ⚠️ 一般 | 丢色彩信息；纯色调切换漏检 |
| **HSV (H,S) hist + Bhattacharyya** | ~1 ms | ✅ 好 | **采纳**：对色温敏感、阈值 [0,1] 直观、OpenCV 高度优化 |
| Edge Change Ratio (Zabih 1995) | ~5-10 ms | ✅ 好 | 抗光照变化最强但实现复杂、CPU 贵；本场景过度 |
| pHash / dHash 汉明距离 | ~0.3 ms | ✅ 好 | 最快但 64-bit 信息量小，"同主体姿态剧变"可能漏检 |
| PySceneDetect ContentDetector | ~2-3 ms | ⚠️ 重型 | 引入额外依赖、配置项多 |
| TransNetV2 ONNX | ~10 ms + GPU | ❌ | **违反"不引入新模型"前提**，否决 |

**F.3 SBS 处理策略**

SBS 视频左右眼来自同一原始画面，**只对左眼做检测**即可（右眼必然同步切）。降低一半开销。

**F.4 重置 RVM rec**

`pipeline/matting.py` Matter 类持有 `self._rvm_rec`（4 个 GPU 张量）。新增：
```python
def _reset_rvm_rec(self):
    # rec 张量的 shape 在 Matter 初始化时已知；这里把内容清零或重新分配
    for i in range(1, 5):
        rec = getattr(self, f"_rec{i}", None)
        if rec is not None:
            rec.fill(0)  # CuPy 张量
    self._alpha_smoother_reset()  # G 节联动
```

在每次 RVM forward 之前：
```python
if self._scene_detector.step(frame_bgr):
    self._reset_rvm_rec()
    log.debug("scene cut detected, RVM rec reset, frame_idx=%d", self._frame_idx)
```

**F.5 配置项**

`config.py` 增加（默认全局关闭，离线脚本主动打开；阈值/冷却/EMA 系数全部可调）：
```python
# PT_RVM_SCENE_RESET:
#   1 detects scene cuts via HSV (H,S) histogram Bhattacharyya distance and
#   resets the RVM recurrent state across scene boundaries. Default off
#   globally; the offline tools enable it explicitly.
RVM_SCENE_RESET = _env("RVM_SCENE_RESET", "0") == "1"

# PT_RVM_SCENE_THRESHOLD:
#   Bhattacharyya distance threshold; above this value triggers a scene cut.
#   Range [0, 1]; 0 = identical, 1 = completely different.
#   0.4 is a balanced default; lower to 0.3 for more sensitive resets,
#   raise to 0.5 to be conservative.
RVM_SCENE_THRESHOLD = float(_env("RVM_SCENE_THRESHOLD", 0.4))

# PT_RVM_SCENE_COOLDOWN:
#   Minimum frames between two cut triggers, to avoid back-to-back resets
#   during fade-out/fade-in transitions. Default ~1 s at 24 fps.
RVM_SCENE_COOLDOWN = int(_env("RVM_SCENE_COOLDOWN", 24))

# PT_RVM_SCENE_REF_EMA:
#   EMA weight on the reference histogram for slow within-scene drift
#   tracking. 0.95 = 95% old + 5% new each frame. Keep close to 1.0;
#   too low will mask real cuts.
RVM_SCENE_REF_EMA = float(_env("RVM_SCENE_REF_EMA", 0.95))
```

**F.6 离线 vs 实时**

- **仅离线开启**。`config.py` 全局默认 `RVM_SCENE_RESET = False`（保护实时模式零变化）
- 离线脚本启动时主动打开：
  ```python
  # tools/offline_passthrough.py / tools/offline_alpha_passthrough.py
  config.RVM_SCENE_RESET = True
  ```
- 实时模式不开的原因：场景检测增加 ~1 ms/帧 CPU 开销虽小但非零；且实时直通场景中"跨场景"很少见（用户多半在播一部片）；与"实时模式完全不动"约定一致

**F.7 UI**

不暴露。属于"内置质量优化"，用户无感知。日志里出现 `scene cut detected` 即可。

**F.8 工时**：0.5–0.75 d（算法本身简单，但要手测 2-3 段不同类型 VR 片段验证阈值）

---

### G. AlphaSmoother (EMA α=0.6, UI 不暴露)（v2 新增）

**目标**：对 RVM 输出 alpha 做时序 EMA 平滑，抑制帧间小幅抖动；**必须依赖 F 节场景检测**联动重置，避免跨场景残影。

**G.1 算法**

```
a_smooth_t = α * a_smooth_{t-1} + (1 - α) * a_raw_t,  α = 0.6
```

α=0.6 含义：新帧 alpha 占 40%，历史占 60%。残影衰减到 < 10% 需约 5 帧（@ 24 fps ≈ 0.2 s）；场景切换时 F 节会强制 reset，所以跨场景残影问题闭环。

**G.2 实现位置**：`pipeline/matting.py` RVM forward 后、composite 之前

新增 `_AlphaSmoother` 内嵌类：
```python
class _AlphaSmoother:
    def __init__(self, alpha_weight: float = 0.6):
        self.alpha_weight = float(alpha_weight)
        self._prev = None  # CuPy 张量，shape 与 RVM alpha 输出一致

    def step(self, alpha_gpu):
        """In-place EMA blend; returns smoothed alpha on GPU."""
        if self._prev is None or self._prev.shape != alpha_gpu.shape:
            self._prev = alpha_gpu.copy()
            return alpha_gpu
        # smoothed = w * prev + (1-w) * alpha
        out = self._prev * self.alpha_weight + alpha_gpu * (1.0 - self.alpha_weight)
        self._prev = out
        return out

    def reset(self):
        self._prev = None
```

**G.3 SBS batch=2 适配**

RVM forward 输出 alpha shape 是 `(B, 1, H, W)`，B ∈ {1, 2}。EMA 直接逐 batch 元素做即可，左右眼各自维护历史。建议存为 `(B, 1, H, W)` 同形状的 `_prev`。

**G.4 与 F 节联动**

F.4 的 `_reset_rvm_rec()` 内**必须**同步调 `self._alpha_smoother.reset()`，否则跨场景残影。

**G.5 配置项**

```python
# PT_RVM_ALPHA_SMOOTH:
#   1 enables temporal EMA smoothing on RVM alpha output. Reset on scene cut.
#   Default on for offline; not exposed in the UI.
RVM_ALPHA_SMOOTH = _env("RVM_ALPHA_SMOOTH", "1") == "1"

# PT_RVM_ALPHA_SMOOTH_WEIGHT:
#   EMA weight for the historical alpha. 0.6 = 60% history + 40% new frame.
RVM_ALPHA_SMOOTH_WEIGHT = float(_env("RVM_ALPHA_SMOOTH_WEIGHT", 0.6))
```

**G.6 离线 vs 实时**

仅离线开启。实时模式：
- 实时 fps 预算紧（NVENC + 网络）
- 实时直通的目标是"现在的画面"，残影更敏感
- 与 F.6 同步：`config.RVM_ALPHA_SMOOTH = False` 全局默认，离线脚本 `tools/offline_passthrough.py` 启动时打开

**G.7 性能开销**

EMA 是逐元素 `a*x + b*y`，GPU 上 ~0.3 ms / 2048² alpha 张量。可忽略。

**G.8 UI**

不暴露。

**G.9 工时**：0.5 d（依赖 F 完成）

---

### E. 缩放算法核对（已完成调查）

调查结果：**当前管线无 nearest-neighbor 引入锯齿，无需修复**。详见调研报告 Section 一。仅记录已核对的关键路径：

| 路径 | 文件:行 | 算法 |
|---|---|---|
| BGR 输入降采样（CPU） | `pipeline/matting.py:1753` | `cv2.INTER_AREA` ✓ |
| BGR 输入降采样（GPU） | `_PREPROCESS_KERNEL_SRC:215` | box-resample ✓ |
| NV12 输入降采样（GPU fp32/fp16） | `:337`, `:405` | box-resample ✓ |
| NV12 中间 resize | `_nv12_resize_kernel:1366` | bilinear ✓ |
| P016→NV12 resize | `:1511` | Y bilinear + UV nearest（4:2:0 标准） ✓ |
| Alpha matte 上采样（CPU） | `:2393` | `cv2.INTER_LINEAR` ✓ |
| Alpha matte 上采样（GPU 合成路径） | `_COMPOSITE_UPSAMPLE_KERNEL_SRC:165` | bilinear ✓ |

MatAnyone2 first-frame seed mask 的 `INTER_NEAREST`（`offline_passthrough.py:830`, `offline_alpha_passthrough.py:804`）是设计上正确的——MA2 期待二值硬边界 mask，与 RVM alpha 链路无关。

**此节工程师无需动手**，列出供 review。

---

## 4. 验证清单

### 4.1 功能验证

- [ ] UI 离线面板新出现"处理精度"下拉，默认"中"
- [ ] 切换三档分别跑同一段 8K SBS 测试视频，输出无报错
- [ ] alpha 通道无渐变灰（CHANGELOG-201 验证视频）
- [ ] ResNet50 引擎不再被任何代码路径调用（grep 验证）
- [ ] 实时模式 FPS / 显存与重构前一致（确认实时不受影响）

### 4.2 质量验证（人眼对比）

针对 1 段带头发/碎发的 8K SBS 片段：
- [ ] 低档 vs 中档：边缘细节明显改善（FGF 从 1024² → 2048²）
- [ ] 中档 vs 高档：细节差异需肉眼可辨；若不明显，考虑将"高"档隐藏或加 expert 模式开关

### 4.2b 场景检测 + AlphaSmoother 验证

针对 1 段**含明显场景切换**（≥3 个 cut）的 8K SBS 片段：
- [ ] 日志出现 `scene cut detected`，cut 数与人工标注接近（漏检允许，误检需 < 10%）
- [ ] 场景切换后第 1-3 帧 alpha **无残影**（旧场景人物轮廓不应出现在新场景）
- [ ] 同场景内**长镜头缓慢推/拉/摇**不被误判（参考帧 EMA 慢更新生效）
- [ ] **淡入淡出转场**不会触发 ≥ 2 次连续 reset（冷却窗口生效）
- [ ] **色温剧变**（如夜→日切换、室内→室外）能正确检出（HSV 优于灰度的关键场景）
- [ ] 同场景内静止镜头下 alpha 抖动**明显减小**（与不开 G 节对比）
- [ ] 物体快速运动场景下**无明显延迟感**（α=0.6 应在可接受范围）
- [ ] 若 α=0.6 残影超出预期，调到 0.5；若抖动抑制不足，调到 0.7（改 `PT_RVM_ALPHA_SMOOTH_WEIGHT` 环境变量验证）
- [ ] 若场景检测漏检多，把 `PT_RVM_SCENE_THRESHOLD` 调到 0.3；若误检多，调到 0.5

### 4.3 性能验证

| 档位 | 单帧 ORT 延迟（目标） | 8K SBS 端到端 fps（目标） |
|---|---|---|
| 低 | ~15.5 ms | ≥ 当前 |
| 中 | ~24 ms | 当前 × 0.65 |
| 高 | ~40 ms（stage1 翻 4 倍） | 当前 × 0.4 |

### 4.4 TRT 缓存验证

- [ ] 点 UI "生成离线 TRT 缓存"按钮，4 个 engine 文件 build 成功
- [ ] marker 文件正确写入，二次启动跳过 build
- [ ] 模型文件或 onnxruntime 版本变化时 marker 失效，自动重建

---

## 5. 风险与回退

### 5.1 显存风险

`2048 + bs=2` 在 stage2 FGF 阶段需要 2048×2048×2 的工作 buffer。2080 11G 测算：
- input tensor (NCHW fp32) = 2×3×2048×2048×4 = 96 MB
- stage1 编码器中间 feature 在 512×512 上，与现状相当
- FGF guide + alpha 全分辨率 = 2×2048×2048×(3+1)×4 = 64 MB
- 加 ORT/cuDNN workspace ≈ 1.5–2 GB

总占用预计 < 3 GB，余量充足。若实测 OOM，回退到"高"档隐藏 + 中档限制 batch=1。

### 5.2 兼容性回退

- 如果发现某些视频在"中"档出现非预期 artifact，UI 上**降回"低"档**即可，无需代码回滚
- 如果整个重构出问题，git revert 即可——所有改动都集中在 6-8 个文件

### 5.3 用户教育

档位 tooltip 必须写清楚 trade-off，避免用户盲目用"高"档导致 fps 暴跌投诉。

---

## 6. 工时估算与执行顺序

| 阶段 | 内容 | 预估 |
|---|---|---|
| 1 | A 节：清理 ResNet50 | 0.5 d |
| 2 | B 节：默认值切换 + 命令行参数透传 | 0.5 d |
| 3 | C 节：UI 精度下拉 + 联动 | 1 d |
| 4 | D 节：离线 TRT warmup 脚本 + UI 触发 | 1.5 d |
| 5 | **F 节：场景检测（HSV-Bhattacharyya）+ RVM rec reset** | **0.75 d** |
| 6 | **G 节：AlphaSmoother α=0.6（依赖 F）** | **0.5 d** |
| 7 | 验证清单 (4.1–4.4 + 4.2b) | 1.5 d |
| **合计** | | **6.25 d** |

建议按 A → B → C → D → F → G 顺序执行：F/G 是时序质量优化，依赖 A-D 的基础重构稳定后再叠加。每节完成后跑一次 `tests/test_offline_convert.py` 等单元测试。

---

## 7. 不在本次范围内（明确推迟）

以下来自 RVM 调研报告，**本次不做**：
- DeoVR Alpha 通道输出（用户自研中）
- ~~场景切换检测自动 reset RVM rec 状态~~ → **v2 已纳入，见 F 节**
- ~~AlphaSmoother / EMA 时序平滑~~ → **v2 已纳入，见 G 节**（机制与 fp16 灰条不同，且依赖 F 节场景 reset 闭环避免残影）
- SAM gating（属于 P2 级未来工作）
- **Person ROI crop（零模型方案已评估）**：技术可行但与 SBS batch=2 冲突、与场景 reset 冲突、bbox 抖动难控、VR 视频人物占比高收益有限。**本次不做**；如未来引入轻量人体检测器再启动
- 实时模式任何改动（包括 F/G 节也仅在离线开启）

---

## 附录 A：现有 SBS batch 机制要点

`config.MATTING_SBS_BATCH=1` 时：
- `pipeline/matting.py:2227-2228` 把 SBS 左右眼合 batch=2 进 ORT
- 离线 RVM 路径默认开启（`offline/convert.py:189-191` 强制 `--sbs-batch`）
- 实时 RVM 路径默认开启（`config.py:255` 默认 `"1"`）
- MatAnyone2 离线路径强制关闭（`offline/convert.py:182` `--no-sbs-batch`）

本次重构**不改动这一机制**，仅确认 4 套 TRT 引擎正好覆盖 `(input_size ∈ {1024, 2048}) × (batch ∈ {1, 2})`。

## 附录 B：ORT TensorRT EP engine 缓存路径

- `pipeline/matting.py:932-944` 注册 trt provider option
- `trt_engine_cache_enable = True`
- `trt_engine_cache_path = config.ONNX_TRT_ENGINE_CACHE_PATH`（默认 `runtime_cache/trt_engines`）
- ORT 内部根据 (onnx 模型 hash, 输入 shape, 精度模式) 计算 engine 文件名，shape 变化自动 build 新文件

---

**END**
