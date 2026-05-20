# RVM 输入尺寸和 Downsample FPS 小结

日期：2026-05-20

## 范围

本小结记录 `videos/test_8k.mp4` 在生产 DLNA 实时路径下的 RVM FPS 实验。比较内容包括：

- RVM fp32 和 fp16。
- SBS batch=2 和左右眼分别 batch=1。
- RVM 输入尺寸 `1024`、`2048`、每眼原始 `4096`。
- 多组 `PT_RVM_DOWNSAMPLE_RATIO`。

## 最终默认值决定

生产默认值为：

- `PT_MODEL_PATH=models/rvm_mobilenetv3_fp16.onnx`
- `PT_MATTING_INPUT_SIZE=1024`
- `PT_RVM_DOWNSAMPLE_RATIO=0.5`
- `PT_MATTING_SBS_BATCH=1`

最终敲定 `1024 + 0.5` 作为均衡默认值。`2048 + 0.125` 在 producer FPS 诊断里更快，但后续绿色 matte 截图检查发现 downsample 太低，几乎无法可靠抠出人像；`1024 + 1.0` 有些帧更干净，但也会抠出更多无关区域；`2048 + 0.5` 明显更慢。综合抠像质量、误抠区域和性能，`1024 + 0.5` 最均衡。

## 关键结果

以下数据均来自同一套 uncapped 实时 DLNA 路径：`PT_PASSTHROUGH_MAX_FPS=0`、`PT_ALPHA_STRIDE=1`、`PT_DECODE_MAX_SIDE=0`，除特别说明外均使用 fp16 模型。

| 配置 | 实际 RVM 输入 | Downsample | interval FPS 均值 | interval FPS 中位数 | ORT 均值 |
|---|---:|---:|---:|---:|---:|
| 最终均衡默认 | `(2,3,1024,1024)` | `0.5` | `43.13` | `42.29` | `18.81 ms` |
| 更快但 matte 质量不合格 | `(2,3,2048,2048)` | `0.125` | `53.20` | `49.73` | `14.64 ms` |
| 2048 更快组合 | `(2,3,2048,2048)` | `0.0625` | `54.62` | `51.94` | `13.91 ms` |
| 2048 偏质量组合 | `(2,3,2048,2048)` | `0.25` | `34.83` | `34.65` | `24.08 ms` |
| 每眼原始尺寸 | `(2,3,4096,4096)` | `0.125` | `20.25` | `20.43` | `43.73 ms` |
| 每眼原始尺寸 | `(2,3,4096,4096)` | `0.0625` | `25.26` | `25.27` | `34.30 ms` |

## 其它结论

- fp16 在测试路径下比 fp32 快约 16-17%。
- SBS batch=2 比左右眼分别 batch=1 更快，因此 `PT_MATTING_SBS_BATCH=1` 继续作为默认值。
- 每眼原始 `4096x4096` 输入在这台机器上不适合作为实时默认，即使用较低 downsample ratio 也明显低于实时目标。
- `2048 + 0.125` 在 FPS 上优于 `1024 + 0.5`，但因为 downsample 太低导致人像 matte 质量不可接受，不能作为生产默认。
- 最终肉眼评估选择 `1024 + 0.5`：抠像质量可接受，误抠区域少于 `1024 + 1.0`，速度明显优于 `2048 + 0.5`，整体均衡性最好。

## SBS Batch 1 和 Batch 2 对比

以下测试固定 `PT_MATTING_SPLIT_SBS=1`。Batch=2 表示左右眼一起以 `(2,3,H,W)` 输入 RVM，只调用一次 ORT；Batch=1 表示左右眼分别以 `2x(1,3,H,W)` 推理，并且左右眼使用独立的 RVM recurrent state。

| 模型 | Batch 模式 | 实际 RVM 输入 | interval FPS 均值 | interval FPS 中位数 | ORT 均值 |
|---|---:|---:|---:|---:|---:|
| fp32 | batch=2 | `(2,3,1024,1024)` | `37.17` | `36.87` | `22.68 ms` |
| fp32 | 左右眼分别 batch=1 | `2x(1,3,1024,1024)` | `35.88` | `35.17` | `23.74 ms` |
| fp16 | batch=2 | `(2,3,1024,1024)` | `43.13` | `42.29` | `18.81 ms` |
| fp16 | 左右眼分别 batch=1 | `2x(1,3,1024,1024)` | `41.93` | `40.38` | `19.70 ms` |

结论：

- fp32 下，batch=2 比左右眼分别 batch=1 快约 `3.6%`。
- fp16 下，batch=2 比左右眼分别 batch=1 快约 `2.9%`。
- 提升不大，但方向稳定，因此 `PT_MATTING_SBS_BATCH=1` 继续作为默认值。
- 代码已修正为 RVM 也尊重 `PT_MATTING_SBS_BATCH`；修正前即使设置为 `0`，RVM 仍会强制走 batch=2。

## Resize 行为

当前 `PT_MATTING_SQUARE=0` 时，不会把非 1:1 视频强行拉成正方形。

- 对 SBS 8K `8192x4096`，每眼先成为 `4096x4096`，再按默认 RVM reference path 变成 `1024x1024`。
- 普通非正方形 2D 视频会尽量保持宽高比，只会把尺寸向下对齐到 32 的倍数。
- 较小的 2D 输入不会再被放大到 RVM reference 尺寸，只保留源尺寸附近的 32 倍数对齐。

## 追加截图验证

为了肉眼比较 matte 质量，又生成了一组固定 `PT_MATTING_INPUT_SIZE=1024` 的绿色 matte 对比：

- `PT_RVM_DOWNSAMPLE_RATIO=1.0`
- `PT_RVM_DOWNSAMPLE_RATIO=0.5`

输出目录：`debug_output/rvm_matte_compare_1024_dr1_vs_dr0p5`。该组只输出绿色 matte，不输出原图。截图运行中两组都确认使用 `CUDAExecutionProvider`。

截图运行的稳态耗时，按每个视频排除 sample 0 后统计：

| 配置 | ORT 均值 | Composite 均值 |
|---|---:|---:|
| `1024 + 1.0` | `48.16 ms` | `26.56 ms` |
| `1024 + 0.5` | `15.55 ms` | `26.11 ms` |

## 产物

- `baseline/auto_tune_8k_phase1_20260520_114353.*`：`1024 + 0.5`
- `baseline/auto_tune_8k_phase1_20260520_130418.*`：`2048 + 0.125`
- `baseline/auto_tune_8k_phase1_20260520_130454.*`：`2048 + 0.0625`
- `baseline/auto_tune_8k_phase1_20260520_123117.*`：`4096 + 0.125`
- `baseline/auto_tune_8k_phase1_20260520_123243.*`：`4096 + 0.0625`
- `debug_output/rvm_matte_compare_1024_dr1_vs_dr0p5`：`1024 + 1.0` 对比 `1024 + 0.5` 的绿色 matte 截图。
- `debug_output/rvm_matte_compare_left_1024_dr0p5_vs_2048_dr0p5`：只看左半边的 `1024 + 0.5` 对比 `2048 + 0.5` 绿色 matte 截图。

## Matte 截图追加验证

肉眼检查发现，`1024 + 1.0` 有些帧抠得更干净，但也更容易抠出无关区域；因此追加比较了 `1024 + 0.5` 和 `2048 + 0.5`。其中 `1024 + 0.5` 直接复用上一轮输出，只重新生成 `2048 + 0.5`。

这轮单项图和对比图都只输出左半边。

| 截图比较 | ORT 均值，排除 sample 0 | Composite 均值，排除 sample 0 | 输出 |
|---|---:|---:|---|
| `1024 + 1.0` | `48.16 ms` | `26.56 ms` | `debug_output/rvm_matte_compare_1024_dr1_vs_dr0p5` |
| `1024 + 0.5` | `15.55 ms` | `26.11 ms` | `debug_output/rvm_matte_compare_1024_dr1_vs_dr0p5` |
| `2048 + 0.5` | `45.25 ms` | `24.52 ms` | `debug_output/rvm_matte_compare_left_1024_dr0p5_vs_2048_dr0p5` |

## 完整 SBS 和拆分左右眼 Batch2 的 Matte 检查

在最终 `1024 + 0.5` 参数固定不变的前提下，又追加比较了完整 SBS 单图输入和拆分左右眼 batch2：

| 模式 | 设置 | 实际 RVM shape | ORT 均值，排除 sample 0 | Composite 均值，排除 sample 0 |
|---|---|---:|---:|---:|
| 完整 SBS batch1 | `PT_MATTING_SPLIT_SBS=0`，`PT_MATTING_SBS_BATCH=0` | `(1,3,1024,2048)` | `17.25 ms` | `37.27 ms` |
| 拆分 SBS batch2 | `PT_MATTING_SPLIT_SBS=1`，`PT_MATTING_SBS_BATCH=1` | `(2,3,1024,1024)` | `14.60 ms` | `31.48 ms` |

产物：

- `debug_output/rvm_matte_compare_full_sbs_vs_split_batch2`
- 视频：`test_4k.mp4`、`test_8k.mp4`、`72456_3840p.mp4`
- 每个视频 10 张采样帧
- 比较图为上下堆叠；8K 样例中，单项 matte 图为 `1280x640`，比较图为 `1280x1280`。
