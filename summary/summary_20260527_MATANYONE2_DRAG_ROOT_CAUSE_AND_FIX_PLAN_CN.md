# MatAnyone2 拖影根因深入分析与治本开发计划

**日期**：2026-05-27
**范围**：MatAnyone2 离线 V2 抠图拖影（drag / afterimage）问题
**面向**：开发人员，作为继续 "60 帧重启" 之后的下一步治本方案
**结论提要**：当前 `PT_MATANYONE2_SEGMENT_FRAMES=60` 是控制 drag 累积时长的剂量调节，不是治本。本计划给出三档治本路径：Phase 1（纯代码门控，1.5d）、Phase 2（光流补偿，2-3d）、Phase 3（双向传播，2-3d）。Phase 1 落地后即可把 `segment_frames` 回退到 120-240 或关闭定时重启。

---

## 1. 现状 / 你为什么觉得"治标不治本"

开发者文档（`summary_20260527_MATANYONE2_DRAG_AFTERIMAGE_HANDLING_CN.md`）已采用：

- `PT_MATANYONE2_ALPHA_SMOOTH` 默认 `0`
- `PT_MATANYONE2_SEGMENT_FRAMES` 默认 `60`

这套配置在测试视频 `72456_3840p.mp4` 上视觉效果接受。但是其作用机制是**每 60 帧把传播状态硬复位回 bootstrap mask**，所以：

| 段内位置 | drag 表现 |
|---|---|
| frame 0 / 60 / 120 …（segment 起点） | drag 最小，但有 bootstrap mask 一致性切换风险 |
| 段中（frame 30 / 90） | drag 已部分累积 |
| 段尾（frame 59 / 119） | drag 最大 |

视觉上呈"每秒喘息一次"的周期性伪影，不是干净。要治本必须改 drag 的产生机制，不是缩短累积窗口。

---

## 2. 真正的 drag 来源（从导出代码与论文一起推）

### 2.1 模型内部 fusion 公式（导出代码 `tools/export_matanyone2_onnx.py:447-477`）

```
affinity      = get_affinity(memory_key, memory_shrinkage, key, selection)
pixel_readout = readout(affinity, flat_msk_value)               # 从长期 memory 读
last_value    = flat_msk_value[:, :, -1]                        # bank 最后 slice
uncert        = pred_uncertainty(last_pix_feat, pix_feat,
                                  last_pred_mask, ...)
pixel_readout = pixel_readout * uncert + last_value * (1 - uncert)   # 论文 region-adaptive fusion
pixel_readout = pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)
new_sensory, logits, prob = segment([feats...], pixel_readout,
                                     sensory, last_mask=last_mask)
```

### 2.2 关键观察（按重要性）

1. **长期 memory bank 在我们的引擎里 segment 内完全冻结**
   `matanyone2_engine.py:684-691` 设置 `memory_key/shrinkage/msk_value/obj_memory`；`694-699` 的 update 路径只更新 `sensory / last_mask / last_pix_feat / last_msk_value`。memory bank 永远是 bootstrap 帧的语义快照。

2. **drag 来自三个"短期状态"，不是来自长期 memory**
   - `last_mask`：上一帧的 alpha，作为强位置先验进 `pixel_fusion` 和 `segment`
   - `last_pix_feat` + `last_pred_mask`：上一帧像素特征与预测，进入 `pred_uncertainty`
   - `sensory`：GRU 风格的递归隐状态，每帧 `update_sensory=True` 累积

   快速运动 → `last_mask` 落在 N-1 位置 → 模型把它当强位置先验 → 输出被牵引回旧位置。而 `sensory` 是连续累积的隐状态，即便 `last_mask` 被替换，过去的轮廓痕迹仍残留在 sensory 里。

3. **论文 fusion 只对 `pixel_readout` 做 uncertainty 加权，对 `last_mask` 不做加权**
   公式 `pixel_readout * uncert + last_value * (1 - uncert)` 只调和 memory-readout 通道。`last_mask` 在 `pixel_fusion` 和 `segment` 中是无条件的强先验。这是论文设计的 fundamental 限制。

4. **`uncert_prob` 当前被完全丢弃**
   `matanyone2_engine.py:283, 317, 348` 三处都是 `_uncert_prob`，模型训练好的、能区分"稳定/变化"区域的免费信号没有取出来用。

5. **未做 Recurrent refinement（论文推荐推理策略）**
   原文建议 bootstrap 帧重复 N 次，只取最后一次的 memory。我们 `_run_eye` 的 `not state.initialized` 分支只跑一次。segment 起点 memory 不饱满，drag 累积更快。

6. **训练 sequence length 上限是 8 帧**
   论文 staged training 从 3 → 8，意味着模型设计上对长程稳定的容忍上限就是 ~8 帧。`segment_frames=60` 已远超训练分布，drag 必然累积，再小的窗口也只是缓解。

---

## 3. 治本路径总览

| Phase | 思路 | 预期 drag 改善 | 工作量 | 性能影响 | 适用 |
|---|---|---|---|---|---|
| 1 | 输入端门控 + 状态衰减 + bootstrap 加强 | "明显"→"轻微" | ~1.5d | ≈ 0 | 全量合入 |
| 2 | 光流补偿 last_mask | "轻微"→"几乎消失" | 2-3d | +1-2ms/eye | 高质量档 |
| 3 | 离线双向传播 | "完全抵消"（对称） | 2-3d | 2× 时间 | 短片高质量档 |

**推荐顺序**：先 Phase 1 全量落地并验收 → 若仍残留可见 drag 再上 Phase 2 → 若发布要求"零 drag"再上 Phase 3。

---

## 4. Phase 1：纯代码门控（1.5d，零风险，可独立开关）

### 4.1 T1-A：uncert_prob 反向门控 last_mask（核心）

**思路**：模型输出 `uncert_prob` 已经区分了"变化大/稳定"区域。把 uncert 高的区域的 last_mask 压低，模型不再被旧位置牵引；uncert 低的稳定区保留 last_mask，防止内部闪烁。

**修改点**：

1. `offline/matanyone2_engine.py:22-31` `_STATE_NAMES`：增加 `"last_uncert"` 字段（仅 Python 侧 state，不改 ONNX）。
2. `_EyeState.__init__` 和 `reset()`：增加 `self.last_uncert = None`。
3. `_step_update` 三处（line 283, 317, 348）：把 `_uncert_prob` 接出来：
   ```python
   prob, sensory, msk_value, obj_memory, pix_feat, _logits, uncert_prob = self.step_update.run(...)
   ```
   返回值新增一个 `uncert_prob`。IOBinding 路径同步取 `outputs["uncert_prob"]`。
4. `_run_eye` (line 694-699)：保存 `state.last_uncert = uncert_prob`。
5. 在喂 `last_mask` 进下一帧 step_update 之前做门控：
   ```python
   if state.last_uncert is not None:
       gate = 1.0 - config.MATANYONE2_LAST_MASK_UNCERT_GATE * state.last_uncert
       last_mask_in = state.last_mask * gate
   else:
       last_mask_in = state.last_mask
   ```
   注意 `uncert_prob` 的空间分辨率可能 = `last_mask` / 4 或 / 8，需要 bilinear upsample 对齐。

**配置项**：
```python
# config.py
# 0 关闭；典型 0.6~0.9
MATANYONE2_LAST_MASK_UNCERT_GATE = float(_env("MATANYONE2_LAST_MASK_UNCERT_GATE", 0.7))
```

**风险**：
- 在 uncert 信号噪声大的素材上可能过度压制 last_mask 导致段内 alpha 闪烁。先调参 0.5 → 0.7 → 0.9 阶梯验证。
- `uncert_prob` 的 dtype 与 spatial shape 与 `last_mask` 不一致时需要对齐；建议加 fallback：shape mismatch 时不门控。

**工作量**：0.5d

---

### 4.2 T1-B：sensory 周期衰减

**思路**：sensory 是 GRU 风格累积，drag 的隐状态主要藏在这里。每 K 帧做一次 `sensory *= γ` 软重置，不杀 state 只稀释累积。

**修改点**：

1. `offline/matanyone2_engine.py` `composite_nv12` 或 `_run_eye` 进入 update 分支前：
   ```python
   if (config.MATANYONE2_SENSORY_DECAY_INTERVAL > 0
       and self._frame_index > 0
       and self._frame_index % config.MATANYONE2_SENSORY_DECAY_INTERVAL == 0):
       decay = config.MATANYONE2_SENSORY_DECAY_FACTOR
       for eye in self.eyes:
           if eye.sensory is not None:
               # OrtValue 路径需要先转 CuPy 再转回
               eye.sensory = self._decay_sensory(eye.sensory, decay)
   ```
2. 实现 `_decay_sensory(value, factor)`：处理 OrtValue / CuPy / NumPy 三种情况，乘 factor 后回写。

**配置项**：
```python
MATANYONE2_SENSORY_DECAY_INTERVAL = int(_env("MATANYONE2_SENSORY_DECAY_INTERVAL", 8))  # 0 关闭
MATANYONE2_SENSORY_DECAY_FACTOR = float(_env("MATANYONE2_SENSORY_DECAY_FACTOR", 0.9))
```

**风险**：
- 衰减过强（< 0.7）会让人像内部 alpha 闪烁；从 0.9 起调。
- IOBinding 双 slot 下要保证衰减后的 sensory 不破坏 ping-pong 槽位。建议衰减时直接覆写当前 slot。

**工作量**：0.4d

---

### 4.3 T1-C：last_pred_mask 二值化解耦

**思路**：引擎 line 277-279 和 343-344 把同一个 `state.last_mask` 既当 `last_mask` 又当 `last_pred_mask`。论文里这两个应该不同（refined vs raw prediction）。把 `last_pred_mask` 换成硬阈值二值化，削弱"软轨迹"传染。

**修改点**：
1. `_propagate` (line 279)、`_propagate_update` (line 313)、`_step_update` (line 344)：
   ```python
   last_pred = (state.last_mask > 0.5).astype(state.last_mask.dtype)
   ...
   "last_pred_mask": self._to_numpy(last_pred),
   ```
   IOBinding 路径 (line 428) 同样替换为二值化版本（GPU 域 `cp.where`）。
2. 加 config 开关：
   ```python
   MATANYONE2_LAST_PRED_BINARIZE = _env("MATANYONE2_LAST_PRED_BINARIZE", "1") == "1"
   MATANYONE2_LAST_PRED_BIN_THRESHOLD = float(_env("MATANYONE2_LAST_PRED_BIN_THRESHOLD", 0.5))
   ```

**风险**：
- 软边区域被硬化可能让 `pred_uncertainty` 误判。consistency 测试需要做。
- IOBinding GPU 域多一个 element-wise kernel，~0.1ms/eye。

**工作量**：0.3d

---

### 4.4 T1-D：Bootstrap 阶段 Recurrent Refinement

**思路**：MatAnyone 论文推荐推理策略。bootstrap 帧重复 N 次走 `image_key → mask_memory → first_frame_refine → mask_memory`，最后一次的 memory 作为 segment memory。

**修改点**：`_run_eye` line 676-692 当前是单次 bootstrap，改成循环：

```python
if not state.initialized:
    feats = self._image_key(image)
    sensory = self.np.zeros(self.sensory_single_shape, dtype=self.tensor_dtype)
    mask = self._bootstrap_mask(h, w, eye_idx, roi=roi)
    msk_value, sensory, obj_memory = self._mask_memory(image, mask, sensory, feats["pix_feat"])

    for _ in range(config.MATANYONE2_BOOTSTRAP_REFINE_ITERS):
        prob, sensory = self._first_frame_refine(feats, msk_value, obj_memory[:, :, None, :, :], sensory, mask)
        alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
        msk_value, sensory, obj_memory = self._mask_memory(image, alpha, sensory, feats["pix_feat"])
        mask = alpha    # 下一轮用上一轮 refined alpha 当 mask

    state.memory_key = feats["key"][...].astype(...)
    state.memory_shrinkage = feats["shrinkage"][...].astype(...)
    state.memory_msk_value = msk_value[...].astype(...)
    state.obj_memory = obj_memory[...].astype(...)
    state.sensory = sensory.astype(...)
    state.last_mask = alpha
    state.last_pix_feat = feats["pix_feat"].astype(...)
    state.last_msk_value = msk_value.astype(...)
    state.initialized = True
```

**配置项**：
```python
MATANYONE2_BOOTSTRAP_REFINE_ITERS = max(1, int(_env("MATANYONE2_BOOTSTRAP_REFINE_ITERS", 3)))
```

**成本**：
- 每次 bootstrap 多 `(N-1) × (first_refine + mask_memory) ≈ (N-1) × ~130ms`
- N=3 时 segment_frames=60，每 60 帧多 260ms ≈ 4.3ms/frame 摊薄
- N=3 时 segment_frames=240，每 240 帧多 260ms ≈ 1.1ms/frame 摊薄

**这是 Phase 1 里最值的一项**——重启质量提升后可以放心把 segment_frames 拉到 120-240。

**工作量**：0.3d

---

### 4.5 Phase 1 验收

#### 4.5.1 消融矩阵

复用 `debug_output/72456_drag_*` 目录，跑：

| 档位 | last_mask gate | sensory decay | last_pred binarize | bootstrap iters | segment_frames |
|---|---|---|---|---|---|
| baseline_60 | off | off | off | 1 | 60（当前默认） |
| t1a_only | 0.7 | off | off | 1 | 60 |
| t1b_only | off | 8 / 0.9 | off | 1 | 60 |
| t1c_only | off | off | on | 1 | 60 |
| t1d_only | off | off | off | 3 | 60 |
| t1_full_60 | 0.7 | 8 / 0.9 | on | 3 | 60 |
| t1_full_120 | 0.7 | 8 / 0.9 | on | 3 | 120 |
| t1_full_240 | 0.7 | 8 / 0.9 | on | 3 | 240 |
| t1_full_scene | 0.7 | 8 / 0.9 | on | 3 | 0（仅 scene reset） |

逐档对比快速运动后 5/15/30 帧的人像周围拖影。目标：`t1_full_120` 或 `t1_full_240` 视觉上不劣于 `baseline_60`，且吞吐 ≥ baseline。

#### 4.5.2 单元测试

- `tests/test_matanyone2_engine.py` 新增：
  - `test_last_mask_uncert_gate_shape_mismatch_falls_back`：uncert shape 不一致时不门控
  - `test_sensory_decay_preserves_dtype`：衰减后 dtype 不变
  - `test_last_pred_binarize_threshold`：边界值
  - `test_bootstrap_refine_iters_default_3`：bootstrap 调用次数符合预期

#### 4.5.3 默认配置决策树

跑完消融矩阵后按下表决策：

| 观察 | 默认配置 |
|---|---|
| `t1_full_240` ≥ `baseline_60` | 升级到 `segment_frames=240`，吞吐回收 |
| `t1_full_120` ≥ `baseline_60` 但 `t1_full_240` 略劣 | 默认 `segment_frames=120` |
| `t1_full_60` 显著优于 `baseline_60`，但拉长 segment 仍劣 | 保持 `segment_frames=60` 但全开 T1，进入 Phase 2 |

---

## 5. Phase 2：光流补偿 last_mask（治本核心，2-3d）

### 5.1 思路

`last_mask` 在 N-1 帧位置 → 用 backward flow `F(N → N-1)` 把 `last_mask` 前向 warp 到 N 帧位置：

```
last_mask_warped[x] = last_mask[ x + F(x) ]
```

喂 warped 版本进 step_update。模型仍然得到位置先验，但**这个先验已被搬到当前帧位置**，drag 几乎完全消失。这是 VOS 文献的标准 motion-compensated propagation。

### 5.2 实现选项

#### 5.2.1 NVIDIA Optical Flow SDK (NVOFA)（推荐）

- RTX 2080 (Turing) 原生硬件加速
- 4K eye 大约 1-2ms
- 通过 `cv2.cuda.NvidiaOpticalFlow_2_0` 或直接调 NVOF SDK
- 输出 4×4 块的 flow，需要 bilinear upsample 到 alpha 分辨率

#### 5.2.2 NV12 域简单块匹配（备选）

- 16×16 块 SAD on Y plane
- 自己写 CUDA kernel，~0.5ms
- 精度低于 NVOFA，运动大时容易失锁

**建议优先 NVOFA，备选块匹配作为 fallback。**

### 5.3 修改点

1. 新增 `pipeline/optical_flow_nvof.py`：
   - `class NvofEstimator`：封装 NvidiaOpticalFlow_2_0 句柄
   - `estimate(prev_nv12_y, curr_nv12_y) -> cp.ndarray`：返回 (H/4, W/4, 2) 的 flow，单位是源像素

2. 新增 `pipeline/mask_warp.py`：
   - `warp_mask_by_flow(mask_2d, flow, out_h, out_w) -> mask_warped`：CUDA kernel，bilinear sample
   - 必须支持 alpha 在 model 分辨率 (512×512)、flow 在 eye 分辨率 (4K/4 = 1K)、out 在 model 分辨率的坐标变换

3. `offline/matanyone2_engine.py`：
   - 增加 `self._nvof_estimator`（lazy init）
   - 增加 `self._prev_eye_y[2]` 缓存
   - `_run_eye` update 分支前：
     ```python
     if config.MATANYONE2_FLOW_WARP and self._prev_eye_y[eye_idx] is not None:
         curr_y = self.matter._extract_eye_y_plane(...)
         flow = self._nvof_estimator.estimate(self._prev_eye_y[eye_idx], curr_y)
         state.last_mask = warp_mask_by_flow(state.last_mask, flow, model_h, model_w)
         self._prev_eye_y[eye_idx] = curr_y
     ```
   - segment reset 时清空 `self._prev_eye_y`

### 5.4 配置项

```python
MATANYONE2_FLOW_WARP = _env("MATANYONE2_FLOW_WARP", "0") == "1"
MATANYONE2_FLOW_WARP_BACKEND = _env("MATANYONE2_FLOW_WARP_BACKEND", "nvof")  # nvof | sad
```

### 5.5 风险与回退

- NVOF SDK 在某些驱动版本可能不可用 → 启动失败自动回退 `block_match`，再失败禁用 flow warp
- 4K SBS eye 单眼分辨率 ~3840×2160，NVOF 句柄初始化是否成功需测试
- Flow 估计在低纹理区不稳 → warp 应该用 confidence map，confidence 低时不 warp（保留原 last_mask）

### 5.6 验收

- 加 `flow_warp_on` / `flow_warp_off` 两档对比
- 检查 step_update 输出 alpha 在快速运动帧的人像周围 drag 是否消失
- 性能：18 fps → 16-17 fps 可接受

### 5.7 工作量

2-3d（含 NVOF 集成调试 + warp kernel + 单测 + 消融）

---

## 6. Phase 3：离线双向传播（短片发布质量，2-3d）

### 6.1 思路

本项目是 offline passthrough。可以两遍跑：

- Forward pass: `0 → T` 正向传播，drag 向后（拖在身后）
- Backward pass: `T → 0` 反向传播，drag 向前（拖在身前）
- 输出：`alpha = w_f · alpha_forward + w_b · alpha_backward`
  - `w_f` 在 segment 起点 = 1，向后线性降到 0
  - `w_b` 在 segment 终点 = 1，向前线性降到 0

drag 在对称融合下完全抵消（前向的"旧位置"被反向的"未来位置"反向 drag 抵消）。

### 6.2 实现点

1. 新增 `tools/offline_passthrough_bidir.py` 或在 `offline_passthrough.py` 加 `--bidirectional` 旗标
2. Forward pass：跑完保存每帧 alpha 到中间文件（或内存 buffer）
3. Backward pass：把视频帧反向读入，用相同的 engine 跑，输出反向 alpha 序列
4. Blend pass：按 segment 内位置做线性权重融合
5. 最终 composite 用融合后的 alpha

### 6.3 配置项

```python
MATANYONE2_BIDIRECTIONAL = _env("MATANYONE2_BIDIRECTIONAL", "0") == "1"
```

### 6.4 性能

- 2× 时间（18 fps → 9 fps）
- 内存：每帧 alpha (h × w × 2 eye) ≈ 4K × 4K × float16 = 32MB；30s × 30fps = 900 帧 × 32MB = 28.8GB → 必须落盘到临时文件
- 替代方案：分块（chunk）双向跑，每块 60 帧内做双向 → 内存 ~2GB，无需落盘

### 6.5 风险

- 反向传播时 first_frame_refine 用的"反向 bootstrap"质量难保证（end-frame mask 不一定是好的 anchor）
- 建议：反向跑时 segment 起点反过来用 forward pass 的高质量 alpha 帧作 bootstrap mask（半监督）

### 6.6 工作量

2-3d

---

## 7. 与现有 V2 模块的兼容性

| V2 模块 | 与 Phase 1 兼容 | 与 Phase 2 兼容 | 与 Phase 3 兼容 |
|---|---|---|---|
| G0 共享 engine | ✅ 直接受益 | ✅ | ✅ |
| G1 IOBinding | ⚠ T1-A 需要从 step_update 输出取 uncert_prob，IOBinding 输出绑定需相应增加；T1-B 衰减 sensory 需要在 ping-pong slot 上覆写 | ✅ | ✅ |
| G2 Guided refine | ✅ 无关 | ✅ | ✅ |
| G3-A ROI | ✅ 但要注意 ROI 分辨率下 uncert shape 也要 ROI | ⚠ flow 估计需在 ROI 坐标系 | ✅ |
| G4 Scene cut | ✅ 配合更好（自适应 reset） | ✅ | ⚠ 双向 + scene cut 段落策略需要梳理 |
| G5 alpha smoother | 已默认 off，无影响 | 同 | 同 |

**Phase 1 与 IOBinding 联动注意**：T1-A 取 `uncert_prob` 时需在 `_step_io_outputs` 的 slot 列表里给 uncert_prob 也开 ping-pong slot。当前 `_step_io_outputs` 是 `dict[str, list[2]]`，扩展即可。

---

## 8. 工作量与排期

| Phase | 子项 | 工作量 |
|---|---|---|
| **Phase 1** | T1-A uncert 门控 | 0.5d |
| | T1-B sensory 衰减 | 0.4d |
| | T1-C last_pred 二值化 | 0.3d |
| | T1-D Bootstrap recurrent refine | 0.3d |
| | 消融矩阵 + 单测 + 默认决策 | 0.5d |
| | **小计** | **2.0d** |
| **Phase 2** | NVOF estimator | 0.7d |
| | mask warp kernel | 0.5d |
| | engine 集成 + fallback | 0.5d |
| | 单测 + 消融 | 0.5d |
| | **小计** | **2.2d** |
| **Phase 3** | 双向调度 + chunk buffer | 0.8d |
| | blend 权重 + segment 对齐 | 0.5d |
| | 反向 bootstrap 策略 | 0.5d |
| | 单测 + 视觉验收 | 0.5d |
| | **小计** | **2.3d** |
| | **总计（三档全做）** | **~6.5d** |

---

## 9. 验收标准

### 9.1 Phase 1 验收

- 视觉：`t1_full_120` 或 `t1_full_240` 在 `72456_3840p.mp4` 上人像周围拖影 ≤ 当前 `baseline_60`
- 性能：吞吐 ≥ 当前 baseline 16.7 fps；目标 `t1_full_240` 吞吐 ≥ 19 fps
- 默认 `segment_frames` 从 60 调整到 120/240/0 之一（看消融）
- 默认 `MATANYONE2_LAST_MASK_UNCERT_GATE` 落点（0.5/0.7/0.9 之一）
- 单测：所有现有 25 passed 不变，新增 4 个测试通过

### 9.2 Phase 2 验收

- 视觉：快速运动帧（手臂挥动、转身）人像周围拖影几乎消失
- 性能：吞吐 ≥ 16 fps（4K SBS）
- NVOF 失败时自动回退块匹配；块匹配失败时禁用 flow warp，不崩溃
- 单测：新增 `tests/test_optical_flow.py`、`tests/test_mask_warp.py`

### 9.3 Phase 3 验收

- 视觉：30s 短片输出对比 forward-only 版本，拖影完全消失
- 性能：吞吐 ≥ 8 fps（接受 2× 减速）
- 内存：分块双向不超 4GB GPU
- 单测：新增 `tests/test_bidirectional.py` 验证 alpha 序列对称融合权重

---

## 10. 决策建议

**当前阶段（合入下一版）**：

1. **Phase 1 全量落地**（T1-A + T1-B + T1-C + T1-D）。这是治本工作里的"低垂果实"，预期单这个 Phase 就能把拖影问题从"明显"降到"轻微可察"。
2. 跑完消融矩阵后**把 `MATANYONE2_SEGMENT_FRAMES` 默认从 60 改到 120 或 240**（看消融结果）。当前 60 是 V2 的剂量补救，Phase 1 后不再需要这么紧。
3. 保留 `MATANYONE2_ALPHA_SMOOTH=0` 默认（开发者已确认）。

**下一版决策点**：

- 如果 Phase 1 落地后用户视觉验收"可接受、不发布级别"，停在 Phase 1
- 如果还需要发布级别（"几乎完全无 drag"），立刻进 Phase 2 光流补偿
- Phase 3 只在"短片高质量档发布"场景立项，不作为通用路径

**不应该做的事**：

- ❌ 不要继续把 `segment_frames` 缩到 30 或更短。短窗口意味着更频繁的 bootstrap mask 切换，prepass mask 一致性风险陡增；视觉上"喘息"周期更密集，更难看。
- ❌ 不要恢复 `MATANYONE2_ALPHA_SMOOTH=1`。EMA smoother 让旧 alpha 衰减过慢，加重视觉拖影。
- ❌ 不要在 G2 guided refine 上试图修拖影。G2 处理的是边缘锐度，与时序传播状态无关。两个问题独立。

---

## 11. 与"脚下地毯被纳入前景"问题的边界

开发者文档提到"脚下地毯、地面被纳入前景的现象"。这**不是 drag**，是 bootstrap mask 的前景判定污染。处理路径完全不同：

- 走 SAM3 / YWES prepass 的 person-class filter（COCO class 0）
- 或 bootstrap mask 后处理（基于人体关键点投影做下肢边界裁剪）
- 或 ROI 限制（G3-A 启用并收紧 max_eye_fraction）

这条线建议作为单独的 prepass 质量改进项立项，不混进本文的 drag 修复。

---

## 12. 一句话总结

**60 帧重启是控制 drag 累积时长的剂量调节，不是治本。真正的 drag 源于 last_mask / last_pix_feat / sensory 三个短期状态在快速运动下的位置滞后。Phase 1（输入端门控 + 状态衰减 + bootstrap 加强）能在不动模型、不动 pipeline 形状的前提下大幅降 drag，并允许把 segment 拉长回收吞吐；Phase 2 光流补偿是 VOS 文献的标准治本方案；Phase 3 双向传播是离线短片高质量档的终极手段。**

建议按 Phase 1 → 视觉验收 → 决定是否继续 Phase 2 / Phase 3 的顺序推进。
