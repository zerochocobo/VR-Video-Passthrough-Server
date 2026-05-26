# MatAnyone2 Offline V2 执行审阅小结

**日期**：2026-05-26  
**范围**：MatAnyone2 离线模式 V2 质量、稳定性、性能优化  
**状态**：V2 主线实现已完成；G2 guided refine 经消融确认存在光晕风险，已改为默认关闭。

## 1. 总览

本轮已完成 MatAnyone2 Offline V2 主线实现：

- G0：抽取共享 `offline/matanyone2_engine.py`。
- G1：实现 batch-1 `step_update` IOBinding 热路径。
- G2：实现 guided alpha refine，但目前作为实验功能默认关闭。
- G4：scene cut 检测并入 prepass segment plan。
- G5：左右眼独立 alpha EMA smoother，并随 segment reset 重置。
- G3-A：1024 ROI crop/letterbox 质量模式，默认关闭。

当前保守默认路径为：

- `PT_MATANYONE2_IOBINDING=1`
- `PT_MATANYONE2_ALPHA_SMOOTH=1`
- `PT_MATANYONE2_SCENE_RESET=1`
- `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0`
- `PT_MATANYONE2_ROI_CROP=0`

这个默认组合保留 IOBinding 和稳定性收益，同时避免 G2 guided refine 带来的可见背景圈。

## 2. 已完成内容

### G0 共享 Engine

- 新增 `offline/matanyone2_engine.py`。
- `tools/offline_passthrough.py` 与 `tools/offline_alpha_passthrough.py` 共用同一套 MatAnyone2 engine。
- green/alpha 输出差异通过 `output_mode="green"` 或 `"alpha"` 注入。

### G1 IOBinding

- 对热路径 `matanyone2_step_update.onnx` 增加 batch-1 GPU IOBinding。
- 首帧 bootstrap graphs 仍走 `session.run()`，降低实现风险。
- IOBinding 首次失败会自动回退。
- 增加双输出 slot，避免 recurrent state 同时作为输入和输出绑定。

### G2 Guided Alpha Refine

- 新增 `pipeline/alpha_guided_filter.py`。
- 使用已上传的 NV12 Y 平面作为 guide。
- 增加约束参数：
  - `PT_MATANYONE2_GUIDED_SUPPORT_FLOOR=0.02`
  - `PT_MATANYONE2_GUIDED_MAX_DELTA=0.08`
- 经视觉消融确认，该模块会在部分素材上放大低置信背景 alpha，形成明显背景圈，因此已改为默认关闭。

### G4/G5 稳定性

- 新增 `utils.scene_detection.SceneCutDetector`。
- SAM3 与 YOLOWorld+EfficientSAM prepass 都会把有效 scene cut 合并进 MatAnyone2 segment starts。
- 增加左右眼独立 alpha EMA smoothing，并在 segment reset 时清理 smoother 状态。

### G3-A ROI 质量模式

- 新增 `pipeline/matanyone2_roi.py`。
- 在 `pipeline/matting.py` 中新增 GPU NV12 ROI crop/letterbox 预处理 kernel，以及 ROI alpha unwarp/feather kernel。
- ROI 从每个 segment 的 bootstrap mask 派生，并按 segment 缓存。
- ROI 要求左右眼都生成有效 ROI；任一失败则回退 full-eye 路径，避免左右 alpha 尺寸混用。
- 修复右眼 SBS 源图偏移和 ROI bootstrap mask 坐标对齐。
- ROI 默认关闭，因为固定 1024 输入下它是质量功能，不承诺提速。

## 3. 验证结果

静态与单元验证：

- `py_compile` 通过：MatAnyone2 engine、matting、guided filter、ROI helper、工具入口和测试文件。
- `pytest tests/test_alpha_guided_filter.py tests/test_matanyone2_engine.py tests/test_scene_detection.py tests/test_offline_convert.py tests/test_matanyone2_trt_runtime_paths.py`
- 结果：`25 passed, 2 skipped`。
- `git diff --check` 通过。

运行时 smoke：

- ROI green TRT+IOBinding smoke：通过。
- ROI alpha TRT+IOBinding smoke：通过。
- 光晕修复后的默认 smoke：通过，日志显示 `alpha_refine=off`。
- 消融输出已生成在 `debug_output/matanyone2_ablation_*`，对应帧图为 `*.frame8.png`。

## 4. 抠图背景圈问题结论

现象：

- V2 默认输出中，人像周围出现明显的脏背景圈。
- V2 前或 V1-like 输出是干净的。

消融结果：

- `v2_default`：guided on + smoother on，背景圈明显。
- `smooth_off`：guided on + smoother off，背景圈仍明显。
- `guided_off`：guided off + smoother on，干净。
- `v1_like`：guided off + smoother off + scene reset off，干净。

结论：

- 主因是 G2 guided alpha refine，不是 EMA smoother，也不是 ROI。
- guided filter 使用亮度 guide 时，把人体边缘附近的低置信背景 alpha 放大了。

已采取措施：

- guided refine 增加 support floor 和 max-delta 约束。
- `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE` 默认值从 `1` 改为 `0`。
- guided refine 保留为实验功能，后续需要重新设计质量策略后再考虑默认开启。

## 5. 未完成或待审阅事项

- 15 秒五档完整性能/质量矩阵尚未完成：
  - V1 baseline
  - IOBinding only
  - IOBinding + guided
  - Full V2 without ROI
  - Full V2 + ROI-A
- ROI 的 SAM3 prepass 路径尚未做 runtime smoke；当前 ROI smoke 使用 YOLOWorld+EfficientSAM。
- ROI-B 512/768 speed mode 未实现；它是 G3-A 验证通过后的条件后续项。
- guided refine 需要重新评估算法或改为更保守的 edge-aware 策略。

## 6. 建议审阅重点

- 先确认默认 `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0` 时输出质量是否恢复到 V2 前水平。
- 审阅 G2 是否继续保留为实验功能，或直接进入重设计。
- 审阅 ROI-A 对远景人物素材是否有实际质量收益。
- 决定是否立项 G3-B 512/768 ROI speed mode。

## 7. 安全回退开关

```bat
set PT_MATANYONE2_IOBINDING=0
set PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0
set PT_MATANYONE2_ROI_CROP=0
set PT_MATANYONE2_SCENE_RESET=0
set PT_MATANYONE2_ALPHA_SMOOTH=0
```
