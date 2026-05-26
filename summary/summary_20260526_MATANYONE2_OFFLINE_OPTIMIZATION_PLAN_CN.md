# MatAnyone2 离线优化与定位重置 — 实施计划 v1

**日期**：2026-05-26
**作者**：架构调研 + 开发者反馈整合（Claude 协同）
**目标读者**：执行本次重构的工程师
**关联报告**：
- `summary/summary_20260526_MATANYONE2_RESEARCH_CN.md`（MatAnyone2 模型机制综合调研）
- `summary/summary_20260510_OFFLINE_MATANYONE2_PLAN.md`（首版离线接入设计）
- `summary/summary_20260525_RVM_OFFLINE_PRECISION_TIERS_PLAN_CN.md`（RVM 精度档位方案，本计划与之并列）
- `examples/matanyone2_onnx_huggingface_readme.md`（ONNX 导出说明）
- `tools/offline_passthrough.py:541-1010`（当前 `MatAnyone2OnnxEngine` 实现）

---

## 1. 背景

### 1.1 实测现状

| 配置 | step_update 平均耗时 | 端到端 FPS |
|---|---|---|
| `1024 + TensorRT step_update` | 49.7 ms | ~8 FPS |
| `1024 + CUDA fallback` | 117 ms | ~3.9 FPS |
| `2048 + 任意 provider` | — | ~0.3 FPS |

`2048` 在 8K SBS（每眼 4096²）上没有可用空间，0.3 FPS 既不是 bug 也不是优化不足，而是 MatAnyone2 架构本身的代价。

### 1.2 已确认的两条瓶颈

1. **算法层 (O(N²))**：MatAnyone2 是 memory-based + cross-attention，空间 token 数 N = (H/stride)·(W/stride)。1024→2048 是 ~16× attention FLOPs。
2. **实现层（CPU/GPU 往返）**：`tools/offline_passthrough.py:901,907-908` 调 `_gpu_preprocess_nv12_one(..., copy_to_host=True)`；所有 recurrent state（`memory_key/msk_value/sensory/last_pix_feat`）都以 NumPy 在每帧反复进出 ORT。
   - 单眼 2048 image fp32 = 50 MB，左右眼 100 MB，加 state 总和可达 300+ MB / 帧
   - PCIe Gen3 x16 实际 ~12 GB/s，仅 H2D+D2H 就 ~60 ms / 帧
   - 这是 RVM 路径（已 GPU-resident IOBinding）和 MatAnyone2 路径的实质差距

### 1.3 已做对、不动的部分

以下设计已经被验证合理，**本计划不修改**：

- 首帧/分段用 SAM3 或 YOLOWorld+EfficientSAM 生成 bootstrap mask
- 长视频分段 reset、重新 bootstrap
- 后续帧优先用 fused `step_update` 图
- SBS 左右眼独立维护 state（不复制单眼）

---

## 2. 总体决策

### 2.1 把 MatAnyone2 重新定位为 "离线质量优先 / 非实时" 工具

- 实时模式继续走 RVM（已 8 FPS+ on 1024+TRT）
- MatAnyone2 仅用于离线模式的"高质量档"
- 不再试图让 MatAnyone2 顶替 RVM

### 2.2 精度档位裁剪 — **只保留 1024**

| 状态 | 决策 |
|---|---|
| `matanyone2_onnx_512_*` | 当前已存在，**本次不删模型文件**，但 UI/CLI 暂不暴露 512 选项 |
| `matanyone2_onnx_1024_*` | **唯一正式档位** |
| `matanyone2_onnx_2048_*`（如有） | 从 UI/CLI 完全移除；任何代码路径不允许指定 2048 |

理由：
- 1024 是当前 8 FPS 的可用甜点
- 512 暂留 ONNX 文件，方便未来加回"快速预览档"，**不需要重新导出**
- 2048 经评估为不可用，留着是坑

### 2.3 UI 档位下拉保留容器，仅暴露一项

UI 离线面板的"MatAnyone2 精度"下拉**保留控件**（避免删了未来再加回的工程开销），但当前只有一项：

```
高（1024） — 推荐
```

未来如确认 512 在某些场景有价值，再加：

```
快（512） — 预览/低显存
高（1024） — 推荐（默认）
```

### 2.4 精度策略 — 维持 FP16

**继续使用 FP16**，理由：
- 当前 8 FPS 测算建立在 FP16 上
- RVM 那条线 FP16 灰条问题来自 rec1-rec4 RNN 的长链累积，MatAnyone2 的 memory bank 累积特性不同（每段 segment 重置），短段内 FP16 误差未观察到明显画质退化
- 实际运行未观察到 alpha 灰条或边缘退化
- **触发条件回退 FP32**：若后续测试中出现下列任一现象，再切 FP32 重新评估：
  - alpha 通道整体变灰或漂移
  - 长 segment 末尾 mask 明显失真
  - 边缘出现非锯齿的"溶解"或晕染

`tools/offline_passthrough.py:636` 当前的 dtype 自动从 ONNX 入参类型推断（`tensor(float16)` → np.float16），**本计划不强制改 fp32**，保持灵活。

### 2.5 实时模式完全不动

继续 RVM `1024 + 0.5 + SBS-batch=2`。本计划任何改动都只影响离线 MatAnyone2 路径。

---

## 3. 文件级修改清单

> 路径均相对 `D:\p\PTServer\`。顺序即推荐改动顺序。

### A. 移除 2048 选项

**A.1 `tools/offline_passthrough.py`**

- L1411（推测，参考 grep 输出）`--matanyone2-size` argparse `choices=[512, 1024]` — **保持现状**（本来就没有 2048）
- 全文件搜 `2048` / `matanyone2_onnx_2048` — 删除任何相关分支或注释
- 验证 `_resolve_matanyone2_model_dir`（L134 附近）只接受 512/1024

**A.2 `tools/offline_alpha_passthrough.py`**
- 同 A.1 处理

**A.3 `offline/convert.py`**
- `ENGINES`（L31-35）保留 `matanyone2_medium` / `matanyone2` 不变
- 检查 `_tool_command` 传递的 `--matanyone2-size` 不会出现 2048

**A.4 `utils/trt_manifest.py`**
- 确认 `MATANYONE2_TRT_ONNX_NAME` / `TRT_MODEL_MATANYONE2` 只引用 1024 模型路径
- 如有 2048 manifest 条目，删除

**A.5 模型文件**
- `models/matanyone2_onnx_512_bs1/bs2`：**保留**（未来可能用）
- `models/matanyone2_onnx_1024_bs1/bs2`：**保留并作为主用**
- `models/matanyone2_onnx_2048_*`（如存在）：用户手动删除即可，代码层不再引用

---

### B. UI 档位下拉保留容器，只暴露 1024

**B.1 `ui/pages/offline_page.py`**

新增 `_matanyone2_precision_combo` 方法（参考 RVM 精度档位的 `_precision_combo`）：

```python
def _matanyone2_precision_combo(self) -> QComboBox:
    combo = _fit_combo(QComboBox())
    # 当前只暴露 1024；未来加 512 时在前面 insert 一项
    combo.addItem("高 (1024) — 推荐", ("high", 1024, "auto"))
    combo.setCurrentIndex(0)
    combo.setEnabled(False)  # 单项时禁用，避免用户误以为有选择
    return combo
```

设计要点：
- 控件**始终存在**，未来加 512 时只需 insert 一项 + 解除 `setEnabled(False)`
- data tuple 第三项是 `--matanyone2-batch`（保持 `auto`）
- 当 `engine != "matanyone2"` 时整组隐藏（参考 `_update_recognition_visibility`）

**B.2 命令行透传**

`_build_command` / `_run_single` / `_run_batch` 内：

```python
cmd.extend(["--matanyone2-size", str(size)])  # 当前固定 1024
cmd.extend(["--matanyone2-batch", str(batch)])  # 当前 "auto"
```

**B.3 国际化文案**

- 中文："高 (1024) — 推荐"
- 英文："High (1024) — Recommended"
- 日文："高品質 (1024) — 推奨"

i18n key 建议：`offline.matanyone2.precision.high`

---

### C. TRT 引擎缓存 — 只覆盖 1024

**C.1 `tools/warmup_offline_trt.py`**

如该脚本已存在（参考 RVM 计划 D 节），MatAnyone2 部分只 warmup 一组 shape：

```python
MATANYONE2_SHAPES = [
    # (input_size, batch)
    (1024, 1),
    (1024, 2),  # SBS 双眼合 batch
]
```

不 warmup 512（暂未启用）、不 warmup 2048（已废弃）。

**C.2 启动时不阻塞**

`main.py` / `app.py` 启动 warmup **不包含**离线 MatAnyone2 引擎，由 UI 按钮按需触发。

**C.3 缓存验证**

```bash
ls runtime_cache/trt_engines/*matanyone2*.engine
# 期望：2 个 engine 文件（1024-bs1, 1024-bs2）
# 每个 mat2 engine 约 150-300 MB，总和 ≤ 600 MB
```

---

### D. （可选 / 低优先）配置项默认值

**D.1 `config.py`**

如有 `PT_MATANYONE2_DEFAULT_SIZE` 类环境变量，确认默认值 = 1024：

```python
# PT_MATANYONE2_DEFAULT_SIZE:
#   ONNX input size for MatAnyone2 offline engine. Only 1024 is supported in
#   the current build. 512 ONNX files are kept on disk for future preview tier.
MATANYONE2_DEFAULT_SIZE = int(_env("MATANYONE2_DEFAULT_SIZE", 1024))
```

**D.2 segment 长度**

`tools/offline_passthrough.py` 当前 `segment_frames` 默认推测为 300（约 12.5 s @ 24fps）。本计划**不调整**，留待未来观察长 segment 漂移情况后再改。

---

## 4. 验证清单

### 4.1 功能验证

- [ ] UI 离线面板的 MatAnyone2 精度下拉只出现一项 "高 (1024)"，且禁用状态
- [ ] 切到 `engine = matanyone2_medium / matanyone2` 时档位下拉可见，切到 `rvm_fast` 时隐藏
- [ ] CLI `--matanyone2-size 2048` 应被 argparse 拒绝（或 `_resolve_matanyone2_model_dir` 显式报错）
- [ ] `tools/offline_passthrough.py --help` 不再出现 2048
- [ ] 一段 8K SBS 测试视频在 1024+TRT step_update 下端到端 ≥ 6 FPS（容忍 1024+CUDA fallback ~4 FPS）

### 4.2 质量验证

- [ ] 1024 在 FP16 下连续跑 600 帧（约 25 s），alpha 通道无整体变灰、无边缘溶解
- [ ] 分段 reset 后第 1-3 帧无残影（已有逻辑）
- [ ] 与 RVM `2048+0.25` 对比同一片段，MatAnyone2 在多人 / 背景人像场景下能正确锁定目标（这是 MatAnyone2 真正的卖点）

### 4.3 FP16 回退触发监测

如下任一现象出现，记录原始视频 + 帧号，并切 FP32 重新跑作为对照：

- [ ] alpha 通道整体变灰（参考 CHANGELOG line 57 / 200 的 FP16 灰条记录方式）
- [ ] 长 segment 末尾 mask 明显失真
- [ ] 边缘出现非锯齿的溶解 / 晕染

### 4.4 缓存验证

- [ ] 点 UI "生成离线 TRT 缓存"按钮后，`matanyone2_*1024*.engine` 出现 2 个
- [ ] 二次启动不重建
- [ ] 模型文件或 onnxruntime 版本变化时 marker 失效

---

## 5. 风险与回退

### 5.1 显存

`1024 + bs=2 + FP16` 在 2080 11G 上：
- image (bs=2 fp16) = 2×3×1024×1024×2 = 12 MB
- memory state 累积上限（10 帧 memory bank）≈ 200-400 MB
- ORT/cuDNN/TRT workspace ≈ 1.5-2 GB
- 总占用 < 2.5 GB，余量充足

若实测 OOM，回退顺序：
1. 关闭 bs2，强制 sequential 左右眼
2. 缩短 segment_frames（减少 memory bank 累积）
3. 切 512 ONNX（如已 enable）

### 5.2 FP16 失败回退

如 4.3 任一现象触发，临时方案：
- 在 `tools/offline_passthrough.py:636` 附近强制 `self.tensor_dtype = np.float32`
- 重新导出 1024 FP32 ONNX（参考 `tools/export_matanyone2_onnx.py`）
- 重新 TRT warmup

工程代价：1-2 天（重导出 + 重 warmup + 验证）

### 5.3 用户教育

UI tooltip 必须写明：
- MatAnyone2 是 **离线质量优先** 工具，FPS 远低于 RVM
- 单段视频建议 < 12 s（避免 memory bank 漂移）
- 多人场景或需要锁定特定目标时优先选 MatAnyone2

---

## 6. 工时估算

| 阶段 | 内容 | 预估 |
|---|---|---|
| 1 | A 节：清理 2048 相关代码路径 | 0.5 d |
| 2 | B 节：UI 档位下拉容器（保留扩展位） + i18n | 1 d |
| 3 | C 节：TRT warmup 脚本调整为只覆盖 1024 | 0.5 d |
| 4 | D 节：config 默认值校对 | 0.25 d |
| 5 | 验证清单 4.1 / 4.2 | 1 d |
| 6 | 验证清单 4.3（FP16 监测，可与 5 并行）| 0.5 d |
| 7 | 验证清单 4.4（TRT 缓存）| 0.25 d |
| **合计** | | **~4 d** |

建议按 A → C → D → B → 验证 顺序执行。B 节涉及 UI 工时最长，放在代码层稳定后再做。

---

## 7. 不在本次范围内（明确推迟 / 已否决）

以下来自调研讨论，**本次不做**：

- ❌ **2048 ONNX 支持** — 已确认架构不可用，UI/CLI 全面移除
- ⏸ **512 ONNX 重新启用** — 文件保留，UI 容器保留，等未来需要"快速预览档"时再开
- ⏸ **1280 / 1536 ONNX 中间档** — 工程代价高（重新导出 + 重 TRT warmup），收益不明，跳过
- ⏸ **bs2 进一步优化** — 现有 `1024_bs2` 保留 warmup，不投入额外优化
- ⏸ **GPU IOBinding 改造（消除 copy_to_host）** — 工程代价大（要重写 MatAnyone2OnnxEngine 全部 state 管理为 CuPy / OrtValue），收益大但优先级不在本次。如后续 MatAnyone2 重要性提升再启动
- ⏸ **ROI / person crop** — 与 RVM 时的"暂不做"决定方向不同（MatAnyone2 attention N² 收益更大），但本次不混入，单独立项
- ⏸ **edge-aware alpha upsampler（Fast Guided Filter / DGF）** — 是把 1024 alpha 落到 8K 画质的关键工程，但属于上采样链路改造，独立立项
- ⏸ **segment_frames 缩短到 120-180** — 等观察到漂移再调
- ❌ **强制 FP32** — 本次维持 FP16，仅在 4.3 触发条件下回退
- ❌ **实时模式改动** — 完全不动

---

## 8. 与既有计划的关系

| 既有计划 | 关系 |
|---|---|
| `summary_20260525_RVM_OFFLINE_PRECISION_TIERS_PLAN_CN.md` | **并列**。RVM 三档（低/中/高）与 MatAnyone2 单档（高）在 UI 上是 engine 切换后各自显示，互不影响 |
| `summary_20260510_OFFLINE_MATANYONE2_PLAN.md` | **本计划替代其"模型尺寸"决策**。首版计划没有讨论 2048 vs 1024，本计划明确锁定 1024 |
| `summary_20260526_MATANYONE2_RESEARCH_CN.md` | **研究依据**。本计划是研究结论的落地版 |

---

## 9. 后续可能的工作（明确不在本次范围）

按"投入产出比"排序，作为未来 backlog（不承诺执行）：

1. **GPU IOBinding** — 消除 PCIe 瓶颈，预期 1024 FPS 翻倍
2. **edge-aware alpha upsampler** — 把 1024 alpha 在 8K 画面上做出 RVM+FGF 同等边缘锐度
3. **ROI / person crop** — 进一步降低 MatAnyone2 实际 token 数，最适合 VR 远景
4. **场景检测 + 自动 segment reset** — 借用 RVM 计划 F 节的 HSV-Bhattacharyya 实现
5. **alpha smoother**（EMA α=0.6）— 借用 RVM 计划 G 节
6. **512 档位重新启用** — 作为快速预览
7. **MatAnyone2 FP32 兜底导出** — 如 FP16 在生产中出问题

---

**END**
