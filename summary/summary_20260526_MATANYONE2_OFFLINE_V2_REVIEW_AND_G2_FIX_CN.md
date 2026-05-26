# MatAnyone2 Offline V2 代码审阅与 G2 光晕修复建议

**日期**：2026-05-26
**范围**：V2 主线代码审阅 + G2 guided alpha refine 光晕问题分析与修复路径
**面向**：开发人员，作为继续修复 G2 的输入

---

## 1. 审阅结论概览

V2 主线（G0/G1/G3-A/G4/G5）实现完整且质量良好，可以合入主线。当前默认组合：

- `PT_MATANYONE2_IOBINDING=1`
- `PT_MATANYONE2_ALPHA_SMOOTH=1`
- `PT_MATANYONE2_SCENE_RESET=1`
- `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0`
- `PT_MATANYONE2_ROI_CROP=0`

这个组合是干净的、稳定的、保留了 IOBinding 收益，**应作为 V2 默认合入**。

唯一需要继续修复的是 **G2 guided alpha refine 的光晕**——但不建议永久关闭，建议按本文 §4 的方案 A 做一次小改动后重新消融。

---

## 2. 各模块审阅细项

| 模块 | 文件 | 评价 |
|---|---|---|
| G0 共享 engine | `offline/matanyone2_engine.py` | ✅ 单类驱动 green/alpha 两条出口，差异通过 `output_mode` + `alpha_packer_factory` 注入，无重复逻辑 |
| G1 IOBinding | `matanyone2_engine.py:355-452` | ✅ 仅在 `step_update` 热路径启用，batch-1 守卫，双 slot ping-pong，异常自动永久回退 |
| G2 Guided refine | `pipeline/alpha_guided_filter.py` | ⚠ 实现正确，但算法本身有光晕缺陷，见 §3 |
| G3-A ROI 质量模式 | `pipeline/matanyone2_roi.py` + engine `_segment_roi_pair` | ✅ segment-fixed bbox（策略 B），左右眼必须同时成功，否则回退 full-eye |
| G4 scene cut | `utils/scene_detection.py` | ✅ HSV-Bhattacharyya + EMA + cooldown，结果合并进 prepass plan，不在 engine 内联检测 |
| G5 alpha smoother | `engine._smooth_eye_alpha` + reset 钩子 | ✅ 左右眼独立，segment reset 时清空 GPU prev |

### 值得肯定的细节
- `_should_use_iobinding` 严格校验 `image.data.ptr` 且 batch==1，避免误绑 CPU 数组；首次异常立刻置 `_iobinding_failed=True` 永久回退。
- ROI bootstrap 用 `_roi_bootstrap_alpha` 在 mask 上一次性 crop/letterbox 对齐到 ROI 坐标系，运行时无重复坐标转换。
- `_reset_segment` 显式清理 `_step_io_outputs`/`_step_io_slot`/`_segment_rois`/`_eye_smoother_gpu_prev`，segment 切换无残留。
- `SceneCutDetector` 用 540p HSV hist，避免 4K 输入下 CPU 直方图开销。

### 几个轻微毛刺（非阻塞，记录供后续优化）
- `_maybe_refine_alpha` 只在 alpha shape ≠ (h, w) 时才 refine，意味着 ROI 路径（alpha 已 unwarp 到 eye 尺寸）会跳过 guided。当前默认 ROI=off + guided=off 一致，但 ROI+guided 联动路径建议加注释说明。
- `SceneCutDetector` cooldown 期内仍对 `_ref_hist` 做 EMA，会让旧场景把参考向后拖；未来可改成 cooldown 期不更新参考、或改用滑窗。
- `roi_from_mask` 的 `max_eye_fraction=0.70`：远景人物刚好接近 70% 时会被剔除，目前是静默回退；建议日志输出被剔除的 frac，便于后续调参。

这些都不影响合入，留待 V2 之后处理。

---

## 3. G2 光晕根因分析

读 `alpha_guided_filter.py:339-400` 并结合开发者的消融结果（guided_off 干净、guided_on 有圈），光晕成因是 **Fast Guided Filter 在大 box 半径下的固有性质**。

### 数学层面
低分辨率 alpha（512×512）上计算线性系数 `(a, b)`，box-filter radius=8：
- 512 分辨率上 radius 8 ≈ 1.5% 帧宽
- 反向到 4K 实际作用半径 ≈ 60 像素

在人像边缘外侧的 box 窗口内：
- `corr_ip = mean(I·p)`，p（alpha）窗口内既含人体又含背景，I（luma）随之相关 → `cov_ip ≠ 0`
- 高分辨率重建时 `q(x) = a·I(x) + b`：person 外但 box 半径内的背景像素，guide 取背景 luma，但 `a` 已被边缘传染 → `q > 0`

这就是用户看到的"人像周围一圈"。

### 现有两道闸门为什么不够
开发者已加：

```python
if support_floor > 0.0:
    refined = cp.where(base < support_floor, cp.float32(0.0), refined)
if max_delta >= 0.0:
    refined = cp.minimum(refined, cp.clip(base + cp.float32(max_delta), 0.0, 1.0))
```

- `support_floor=0.02`：只杀 bilinear base alpha < 0.02 的像素。但人像边缘外 60px 内，bilinear 上采的 alpha 通常 0.03~0.2，落不到下限。
- `max_delta=0.08`：限 refined ≤ base + 0.08。bilinear base ~0.1 时，refined 被 clip 到 0.18 —— 在 1.0 满 alpha 旁仍是肉眼明显的脏边。

**结论**：两道闸门是缝补，没有杀掉根因。

---

## 4. 修复方案（按工程代价升序）

### 方案 A：confidence-band 限制 ⭐ 推荐落地
只在 base alpha 落在过渡带 `[α_lo, α_hi]`（建议 0.05~0.95）内做 refine，带外强制保留 base。等价于隐式 trimap。

**改动位置**：`pipeline/alpha_guided_filter.py` 末尾，紧跟现有 support_floor / max_delta 之后加一行：

```python
band_lo, band_hi = 0.05, 0.95
refined = cp.where((base < band_lo) | (base > band_hi), base, refined)
```

**为什么有效**：远离 person 的背景像素 base alpha 接近 0 → 强制保持 0，halo 彻底消失。Person 内部完整像素 base alpha 接近 1 → 保持 1，不被 guided 反复抹边。只在真正的"边缘过渡带"内 refine，正是 alpha matting 该做的范围。

**代价**：单行 GPU select，工作量 < 0.5d，几乎无额外计算开销。

**风险**：若 base bilinear alpha 在真实软边（头发、半透明）也 < 0.05，会被强制压回 0，损失软边。但实际上 512 → 4K 双线性会把任何 > 0 的低分像素抹成 ≥ 0.05 的小区域，软边不会全跌进 < 0.05 的"硬背景"判定。建议先用 0.05/0.95 跑消融，若软边受损再放宽到 0.02/0.98。

**这是 MODNet / RVM / closed-form matting 系列的标准做法**：所有 alpha refine 都隐式或显式只在 unknown band 内跑。

### 方案 B：mask-aware 加权 box filter
计算 a/b 时按 base alpha 的"非中间度" `w = 1 - 4·base·(1-base)` 加权（核心像素权重高、过渡带权重低），等价于不让中间值污染线性回归。

**代价**：要把 `_box_filter` 改成加权版（分子 `sum(w·x)` / 分母 `sum(w)`），约 1-1.5d。
**收益**：比 A 更柔顺，软边过渡更自然。
**建议**：先做 A 验收；A 还不达标再升级 B。

### 方案 C：换上采技术（joint bilateral upsample）
小半径 JBU (r=2~3) 不会跨大区域传染，但锐度收益弱于 FGF。约 1d。**不推荐**——本质上是退回更保守的滤波器，丢失 V2 边缘锐化收益。

### 方案 D：trimap-based refine
对 base alpha 做 threshold + dilate/erode 算 trimap，refine 只跑 unknown 区。算法上等价于 A 的更精细版（带 morphology），约 1d。如果 A 在某些素材上软边判定不准再考虑升级。

### 方案 E：永久关闭 G2
零风险，但放弃 V2 边缘锐化预期。**不推荐**——算法没坏，只缺一道过渡带门。

---

## 5. 建议执行顺序

1. **立即合入当前 V2 主线**（默认 `EDGE_AWARE_UPSAMPLE=0`）。这是已验证干净的路径，无阻塞。
2. **G2 落地方案 A**（单行 `where` 改动），重新跑 §6 的消融。
   - 如果 `v2_default_with_band` 与 `guided_off` 视觉一致 + 边缘比 `guided_off` 更锐 → 把 `EDGE_AWARE_UPSAMPLE` 默认改回 1。
   - 如果仍有可见 halo → 升级方案 B 或 D。
   - 如果软边受损 → 把 band 放宽到 0.02/0.98 再试。
3. **G3-A ROI**：保持默认关闭，等远景素材实际质量收益验证后再决定是否上默认。
4. **15 秒五档完整性能/质量矩阵**（V1 / IOBinding only / IOBinding+guided-with-A / Full V2 / Full V2+ROI）按现 V2 plan §9.2 跑完。

---

## 6. 验收消融建议

复用现有的 `debug_output/matanyone2_ablation_*` 框架，新增一档：

| 档位 | guided | smoother | scene_reset | band | 预期 |
|---|---|---|---|---|---|
| `v1_like` | off | off | off | - | 干净 baseline |
| `guided_off` | off | on | on | - | 干净（当前默认） |
| `v2_default` (旧) | on | on | on | 无 band | 明显光晕 |
| `v2_band_a` (新) | on | on | on | 0.05/0.95 | **目标：干净 + 比 guided_off 边缘更锐** |
| `v2_band_loose` (备) | on | on | on | 0.02/0.98 | 软边保留更好 |

逐帧对比第 8 / 50 / 200 帧的人像边缘 + 远离人像的背景区。

---

## 7. 配置项建议

如果方案 A 上线，可以加两个 env：

```python
# config.py
MATANYONE2_GUIDED_BAND_LO = float(os.environ.get("PT_MATANYONE2_GUIDED_BAND_LO", "0.05"))
MATANYONE2_GUIDED_BAND_HI = float(os.environ.get("PT_MATANYONE2_GUIDED_BAND_HI", "0.95"))
```

`fast_guided_filter_upsample` 增加 `band_lo, band_hi` 参数，`engine._maybe_refine_alpha` 传入。这样若个别素材软边受损可调，不必改代码。

---

## 8. 一句话总结

**V2 主线代码质量良好，可直接合入；G2 不需要弃，只需在 `pipeline/alpha_guided_filter.py` 末尾加一行 confidence-band `where` 限制。**

开发者落地后只需重新跑一档消融即可决定默认开关。
