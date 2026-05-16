> ⚠️ SUPERSEDED 2026-05-16
> 本报告中的 FPS、sync、瓶颈归因结论建立在旧默认值 PT_PASSTHROUGH_MAX_FPS=30 和/或非目标 PT_ALPHA_STRIDE=3 条件下，已被后续复核证伪或降级为仅适用于旧诊断条件。
> 重新基线与有效结论入口见 summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_CN.md。
> 仅保留为研究过程档案；与 cap 无关的实现结论仍需按正文中的适用范围判断。
# 阶段 4R 小结 - RVM FP16 模型基线

## 背景

根据外部复核意见，本阶段不采用 DirectML，不把 TensorRT 作为默认路径，而是优先验证已有 RVM FP16 ONNX 模型在当前 ORT CUDA EP + IOBinding 管线中的实际收益。

## 测试范围更正

本文件最初记录的 FP16 基线，除非命令中显式写出 `PT_ALPHA_STRIDE=1`，都使用项目默认值 `PT_ALPHA_STRIDE=3`。

这意味着前面的 `36-37fps` 结论只能代表 stride=3 诊断结果，不能代表最初目标“`ALPHA_STRIDE=1` 下 8K 达到 40fps”。

本阶段使用用户放入的模型：

- `models/rvm_mobilenetv3_fp16.onnx`

该模型已确认输入输出为 FP16：

- `src`: `tensor(float16)`
- `r1i/r2i/r3i/r4i`: `tensor(float16)`
- `downsample_ratio`: `tensor(float)`
- `pha/fgr/r1o-r4o`: `tensor(float16)`

## 测试命令

短测：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 8 --startup-timeout 240 --client-timeout 120 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx
```

60s 正式基线：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 180 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx
```

说明：

- `videos/test_8k_2.mp4` 约 26s，适合作快速 sanity check。
- `videos/test_8k.mp4` 为 8192x4096、约 60.06s、59.94fps，更适合作 60s 稳态基线。

## 测试结果

### FP16 短测：test_8k_2.mp4

报告：

- `baseline/auto_tune_8k_phase1_20260516_162123.md`
- `baseline/auto_tune_8k_phase1_20260516_162123.json`

结果：

- active providers: `CUDAExecutionProvider`, `CPUExecutionProvider`
- model input type: `tensor(float16)`
- latest interval FPS: `36.90`
- average interval FPS: `35.60`
- stage avg:
  - decode: `0.07 ms`
  - composite: `14.02 ms`
  - sync: `11.97 ms`
  - encode: `0.43 ms`
  - mux: `0.60 ms`
- mat avg:
  - preprocess: `0.03 ms`
  - ORT/RVM: `13.48 ms`
  - kernel: `0.39 ms`

### FP16 全片：test_8k_2.mp4

报告：

- `baseline/auto_tune_8k_phase1_20260516_162213.md`
- `baseline/auto_tune_8k_phase1_20260516_162213.json`

结果：

- latest interval FPS: `36.93`
- average interval FPS: `36.63`
- stage avg:
  - decode: `0.05 ms`
  - composite: `13.99 ms`
  - sync: `12.67 ms`
  - encode: `0.34 ms`
  - mux: `0.02 ms`
- mat avg:
  - preprocess: `0.08 ms`
  - ORT/RVM: `13.23 ms`
  - kernel: `0.59 ms`

### FP16 60s 正式基线：test_8k.mp4

报告：

- `baseline/auto_tune_8k_phase1_20260516_162343.md`
- `baseline/auto_tune_8k_phase1_20260516_162343.json`

结果：

- HTTP status: `200`
- first byte: `3.913 s`
- bytes read: `156943716`
- average client bitrate: `24.29 Mbps`
- latest interval FPS: `36.94`
- average interval FPS: `36.89`
- slow mux warnings: `0`
- pacing/stall/timeout lines: `0`
- stage avg:
  - decode: `0.06 ms`
  - composite: `14.25 ms`
  - sync: `12.34 ms`
  - encode: `0.34 ms`
  - mux: `0.07 ms`
- mat avg:
  - preprocess: `0.04 ms`
  - ORT/RVM: `13.68 ms`
  - kernel: `0.43 ms`

### FP16 60s 正确目标基线：test_8k.mp4 + ALPHA_STRIDE=1

命令：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 240 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx --server-env PT_ALPHA_STRIDE=1
```

报告：

- `baseline/auto_tune_8k_phase1_20260516_170451.md`
- `baseline/auto_tune_8k_phase1_20260516_170451.json`

结果：

- HTTP status: `200`
- first byte: `2.858 s`
- latest interval FPS: `35.56`
- average interval FPS: `35.03`
- slow mux warnings: `0`
- pacing/stall/timeout lines: `0`
- stage avg:
  - decode: `0.08 ms`
  - composite: `25.65 ms`
  - sync: `1.84 ms`
  - encode: `0.50 ms`
  - mux: `0.05 ms`
- mat avg:
  - preprocess: `0.17 ms`
  - ORT/RVM: `24.60 ms`
  - kernel: `0.55 ms`

结论：

- stride=1 与 stride=3 确实存在明显差异。
- 在 stride=1 下，瓶颈重新明确回到 ORT/RVM/composite，而不是前面 stride=3 诊断中的 sync 等待。
- 当前 FP16 + ThreadedDecoder + HEVC 路径在 `videos/test_8k.mp4` 上约 `35.03fps`，距离 40fps 仍差约 `5fps`。

## 与 FP32 基线对比

参考 FP32 基线：

- `baseline/auto_tune_8k_phase1_20260516_154731.md`
- 视频：`videos/test_8k_2.mp4`
- latest interval FPS: `36.40`
- average interval FPS: `36.13`
- ORT/RVM: `15.17 ms`
- composite: `15.98 ms`
- sync: `11.00 ms`

同视频 FP16 全片：

- `baseline/auto_tune_8k_phase1_20260516_162213.md`
- latest interval FPS: `36.93`
- average interval FPS: `36.63`
- ORT/RVM: `13.23 ms`
- composite: `13.99 ms`
- sync: `12.67 ms`

stride=3 结论：

- FP16 模型确实生效，stride=3 下 ORT/RVM 从约 `15.17 ms` 降到约 `13.23 ms`，约减少 `1.94 ms`。
- stride=3 下端到端 FPS 只从约 `36.13` 提升到约 `36.63`，增益约 `0.5 fps`。
- 60s `test_8k.mp4` stride=3 稳态约 `36.89 fps`，仍未达到 40fps。
- sync 等待从约 `11 ms` 升到约 `12-13 ms`，说明 stride=3 下 ORT 降低后，剩余瓶颈被 GPU 等待/跨流同步吸收。

## 风险与注意事项

- 目前只验证了性能与稳定性，没有做逐帧画质/alpha 边缘主观对比。
- FP16 输入输出可能带来轻微 alpha 边缘差异，需要在真实 VR 播放场景中肉眼确认。
- `test_8k.mp4` 与 `test_8k_2.mp4` 码率、内容不同，不能把两者的 client bitrate 或首包时间直接互相比。
- FP16 不是最终 40fps 解法，只是降低了 RVM 子项成本。
- 必须显式区分 `PT_ALPHA_STRIDE=1` 与默认 `PT_ALPHA_STRIDE=3`；二者性能瓶颈不同。

## 建议下一步

FP16 可以保留为可选或默认候选模型，但不能单独完成 stride=1 下 40fps 目标。下一步建议按反馈意见继续：

1. 补一次正确抓服务进程的 Nsight profile，解释 `sync/upload_sync` 等待来源。
2. 如果继续优化 ORT/RVM，评估 CUDA Graph，但要先确认固定 IOBinding 指针和固定输入形状的约束。
3. 暂不恢复 DirectML、默认 TRT、自定义 composite kernel、Python 多阶段流水线。

## 验证

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\matting.py pipeline\pynv_stream.py tools\auto_tune_8k.py
```
