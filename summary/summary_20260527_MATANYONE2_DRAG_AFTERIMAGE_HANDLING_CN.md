# MatAnyone2 拖影问题处置经验小结

日期：2026-05-27

## 背景

本次针对 `videos/72456_3840p.mp4` 排查 MatAnyone2 抠像拖影问题。现象是人物快速运动后，手臂、身体、头发附近会残留上一段姿态的半透明轮廓。该问题在 V2 优化前已经存在，不是 G2 guided refine 或 V2 的新增问题。

排查中对比了既有输出：

- `videos/72456_3840p_matanyone2_S000000_30S_LR_180_SBS_passthrough.mp4`
- `videos/72456_3840p_matanyone2m_S000000_15S_LR_180_SBS_passthrough.mp4`
- `videos/72456_3840p_rvm1_S000000_30S_LR_180_SBS_passthrough.mp4`

辅助诊断图位于：

- `debug_output/72456_drag_existing/m2_30s.contact.png`
- `debug_output/72456_drag_existing/rvm_30s.contact.png`
- `debug_output/72456_drag_compare/contact_crop_left_variants.png`
- `debug_output/72456_drag_compare/contact_crop_reset120_vs_reset60.png`
- `debug_output/72456_drag_compare/contact_default_fix.png`

## 判断结论

拖影的主要来源是 MatAnyone2 的传播状态残留。模型在连续帧传播时会把历史前景形状带入后续 alpha，快速动作时就表现为旧手臂、旧身体边缘或旧发丝轮廓。

`PT_MATANYONE2_ALPHA_SMOOTH` 的 EMA 平滑会加重视觉拖影，但不是根因。只关闭平滑可以略微改善边缘和速度，但无法消除传播状态中残留的旧轮廓。

G2 guided refine 主要处理边缘上采样和局部细化，不负责 MatAnyone2 的时序传播状态。此前 frame 360/540/720 出现的浅色边缘外扩是另一类边缘细化副作用；本次拖影问题的根因不同。

地毯、地面在脚下被带入前景的现象属于初始/重启 mask 的前景选择污染，不完全是时序拖影。这个问题需要后续从 bootstrap mask 过滤、ROI 或 person-only 约束处理。

## 实验记录

基线默认路径：

- 输出：`debug_output/72456_drag_baseline_10s.mp4`
- 600 帧，返回码 0，吞吐 16.70 fps，`matting_avg=56.924 ms`
- 视觉结果：快速动作后手臂和身体周围存在明显旧轮廓。

仅关闭 alpha EMA 平滑：

- 输出：`debug_output/72456_drag_nosmooth_10s.mp4`
- 600 帧，返回码 0，吞吐 19.30 fps，`matting_avg=49.222 ms`
- 视觉结果：拖影略减且速度更快，但旧轮廓仍然明显。

每 120 帧重新 bootstrap，关闭平滑：

- 输出：`debug_output/72456_drag_reset120_nosmooth_10s.mp4`
- 600 帧，返回码 0，吞吐 19.10 fps，`matting_avg=49.808 ms`
- 视觉结果：重启帧附近改善明显，但两个重启点之间仍有可见残影。

每 60 帧重新 bootstrap，关闭平滑：

- 输出：`debug_output/72456_drag_reset60_nosmooth_10s.mp4`
- 600 帧，返回码 0，吞吐 17.94 fps，`matting_avg=53.179 ms`
- 视觉结果：目前测试中效果最好，旧手臂、旧身体轮廓显著减少。

修复后默认路径：

- 输出：`debug_output/72456_drag_default_after_fix_10s.mp4`
- 日志显示：`segment_frames=60`，`alpha_smooth=0`，`alpha_refine=off`，`iobinding=1`
- `matanyone2_step_update.onnx` 使用 `TensorrtExecutionProvider`
- 分段计划：`[0, 60, 120, 180, 240, 300, 360, 420, 480, 540]`
- 600 帧，返回码 0，吞吐 18.01 fps，`matting_avg=52.992 ms`
- `matanyone2_step_update_avg=21.938 ms n=1180`
- `matanyone2_first_refine_avg=112.094 ms n=20`
- 视觉对比图 `debug_output/72456_drag_compare/contact_default_fix.png` 与 `reset60_nosmooth` 的最佳结果一致。

## 已采用策略

默认关闭 MatAnyone2 alpha EMA 平滑：

- `PT_MATANYONE2_ALPHA_SMOOTH` 默认值从 `1` 改为 `0`

新增固定间隔重启传播状态：

- 新增 `PT_MATANYONE2_SEGMENT_FRAMES`
- 初始默认值为 `60`，Phase 1 和左右眼修复验证后调整为 `240`
- 设置为 `0` 时关闭固定间隔重启

该参数已接入：

- `tools/offline_passthrough.py`
- `tools/offline_alpha_passthrough.py`

同时更新了 `PROJECT.md` 中的关键配置说明。

## 验证

语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile config.py tools\offline_passthrough.py tools\offline_alpha_passthrough.py offline\matanyone2_engine.py
```

测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alpha_guided_filter.py tests\test_matanyone2_engine.py tests\test_scene_detection.py tests\test_offline_convert.py tests\test_matanyone2_trt_runtime_paths.py
```

结果：`25 passed, 2 skipped`。

## 经验结论

MatAnyone2 这类传播式视频抠像遇到快速动作拖影时，应优先区分三类问题：

- 传播状态残留：旧姿态、旧肢体轮廓跟随到后续帧。本次主要问题属于这一类。
- 后处理平滑残留：EMA 或时序滤波让旧 alpha 衰减过慢。关闭平滑有帮助，但不能替代状态重置。
- 初始 mask 污染：地面、地毯、物体被纳入前景。需要改进 bootstrap mask，而不是只调时序参数。

早期最有效的实用策略是短分段重启 MatAnyone2 传播状态，并默认关闭 alpha EMA 平滑。60 帧重启在该视频上取得了质量和性能之间相对可接受的平衡；Phase 1 和左右眼修复后，默认已放宽到 240 帧。

## 遗留风险

频繁 re-bootstrap 会增加 first refine 调用次数，使首帧/重启帧 p99 延迟上升。默认 240 帧减少了这部分开销，但在更长视频或更高并发场景中仍需关注尾延迟。

如果 prepass mask 在重启点质量不稳定，可能产生每秒一次的轻微 mask 一致性变化。需要后续用更多视频覆盖验证。

脚下地毯、地面被纳入前景的问题仍未完全解决。它应作为 bootstrap mask 质量问题单独处理，例如增加 person-only 约束、ROI 限制或更强的初始 mask 过滤。

## 后续执行更新

Phase 1 根因修复和 SBS 左右眼 buffer 修复落地后，重新复用 `videos/72456_3840p.mp4` 的 SAM3 prepass mask 跑了 10 秒消融：

- `segment_frames=60`：14.31 fps，`matting_avg=67.123 ms`
- `segment_frames=120`：17.11 fps，`matting_avg=55.881 ms`
- `segment_frames=240`：17.60 fps，`matting_avg=54.159 ms`

视觉接触图：`debug_output/72456_phase1_ablation_compare/seg60_120_240_mid_end_contact.png`。

结论：Phase 1 和左右眼修复后，`segment_frames=240` 在该测试片段上未表现出比 60 帧更差的段尾拖影，同时显著减少 bootstrap 成本。因此默认值已从 `60` 调整为 `240`。

用户复测后确认 Phase 1 效果已经足够好。Phase 2 光流补偿和 Phase 3 双向传播暂不实施，除非后续出现新的发布级质量要求再重新立项。
