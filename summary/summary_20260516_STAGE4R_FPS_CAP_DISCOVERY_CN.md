# 阶段 4R 小结 - FPS Cap 发现与旧性能结论废止

## 结论

2026-05-16 复核确认，Stage 4R 之前多数组合测试受到默认 `PT_PASSTHROUGH_MAX_FPS=30` 污染；同时部分测试还运行在非目标条件 `PT_ALPHA_STRIDE=3` 下。因此，过去关于 `~37fps` 物理上限、sync 等待归因、RVM bypass 无增益、slot/decoder/codec 改动无效等 FPS 结论，不能再作为 8K 40fps / `ALPHA_STRIDE=1` 目标的依据。

本次处理不删除旧 summary，而是在相关文件顶部加入 `SUPERSEDED 2026-05-16` 横幅。旧文件仅保留为研究过程档案；与 FPS cap 无关的实现结论仍按各自适用范围保留。

## 直接原因

`config.py` 中旧默认值为：

```python
PASSTHROUGH_MAX_FPS = float(_env("PASSTHROUGH_MAX_FPS", 30))
```

该值会通过 `utils/video_metadata.py::effective_fps()` 把输出媒体 FPS 压到 30。此前 `videos/test_8k.mp4` 是约 59.94fps 源，但 live cache key 出现 `/hevc/30.000/`，说明输出端实际按 30fps 媒体流生成。

这会导致 producer 以快于实时的方式生成 30fps 输出流，因此日志中出现 `36-37fps` 并不代表 8K 59.94fps/无 cap 的真实吞吐上限。

补充说明：`PT_PASSTHROUGH_MAX_FPS=30` 是输出媒体 timestamp / 选帧 FPS cap，不是 producer wall-clock 速率门。无论输出媒体 fps 是 30 还是 59.94，producer 都会尽可能快地生成目标帧。因此旧数据里 “cap=30 但 producer 仍显示 36-37fps” 并不矛盾；它只是碰巧接近后续发现的 PyNv 8K HEVC 编码吞吐。

## 已执行修正

- `config.py` 默认改为无 cap：`PT_PASSTHROUGH_MAX_FPS` 默认值从 `30` 改为 `0`。
- 保留配置项：需要客户端兼容性诊断时仍可显式设置 `PT_PASSTHROUGH_MAX_FPS=30`。
- `pipeline/pynv_stream.py` 已加入 runtime config 日志，用于记录实际 `alpha_stride`、`max_fps`、`output_fps`、decoder、worker mode、model、是否 bypass RVM 等。
- 已对 Stage 4R threaded/two-stage、profiling/TRT blocker、FP16 baseline、downstream barrier probes 中英文 summary 加废止横幅。

## 旧结论处理

仍成立：

- ThreadedDecoder frame 不能跨线程排队持有；旧 batch 底层 GPU 指针可能在下一次 `get_batch_frames()` 后失效，native crash 风险成立。
- `PyNvThreadedSerialDecoder` 的顺序消费/lifetime 设计仍成立。
- MKV unsafe Cues 走 block 策略仍成立。
- TRT runtime DLL 缺失导致 TensorRT EP 回落 CUDA EP 的部署结论仍成立。
- DirectML 与当前 CUDA 常驻管线架构不兼容的判断仍成立。
- FP16 模型可以被 ORT CUDA EP 加载并运行；但性能收益必须在无 cap、目标 stride 下重测。

废止或需重测：

- `~37fps` 是物理硬上限：废止。
- sync 等待代表 GPU/NVENC 真实瓶颈：废止，旧数据中可能只是 cap 后的等待位置迁移。
- RVM bypass 仍约 37fps 因此下游才是瓶颈：废止，该实验在 cap 条件下不能说明目标场景。
- slot=3 vs slot=8 无差异：需在无 cap 条件下重测。
- FP16、decoder、worker mode 的 FPS 收益：需在无 cap 且显式 stride 条件下重测。

## 第一条干净事实

已运行一次 `videos/test_8k.mp4`、FP16、`PT_ALPHA_STRIDE=1`、`PT_PASSTHROUGH_MAX_FPS=0`：

- 报告：`baseline/auto_tune_8k_phase1_20260516_171415.md/json`
- 目标帧数：`3600`，不再是 30fps cap 下的约 `1801`
- latest interval FPS：`36.32`
- average interval FPS：`36.50`
- stage avg：decode `0.06ms`，composite `24.95ms`，sync `2.06ms`，encode `0.43ms`，mux `0.03ms`
- mat avg：pre `0.13ms`，ORT/RVM `24.13ms`，kernel `0.44ms`

这条数据说明：在无 cap 且 `stride=1` 时，RVM/ORT 回到关键路径，不能再使用旧的 stride=3/bypass 数据推断目标瓶颈。

## 后续重测计划

1. 基线 A：`stride=1`、FP32、simple decoder、slot=3、`PT_PASSTHROUGH_MAX_FPS=0`。
2. 对比 B：基于 A，分别跑 `stride=1/3/6`，确认 stride 灵敏度。
3. 对比 C：在 `stride=1` 下比较 FP32 vs FP16，确认 FP16 真正 FPS 收益。
4. 对比 D：在胜出配置下比较 simple vs threaded_serial。
5. 对比 E：胜出配置 + RVM bypass，测纯下游上限。

每次报告必须记录 runtime config 日志，确认 `alpha_stride`、`max_fps=0`、`output_fps≈59.94`，否则结果不得用于优化决策。
## 无 cap 重测结果补充

以下测试均使用 `videos/test_8k.mp4`，并显式设置 `PT_PASSTHROUGH_MAX_FPS=0`。报告中的目标帧数均为 `3600`，cache key 为 `/hevc/0.000/`，不再是旧的 `/hevc/30.000/`。

| 测试 | 报告 | 条件 | Avg FPS | Latest FPS | 关键 stage |
|---|---|---|---:|---:|---|
| A | `baseline/auto_tune_8k_phase1_20260516_172105.md` | FP32, simple, stride=1 | 34.37 | 34.30 | decode 3.27, composite 23.48, sync 1.98, ORT 22.71 |
| B1 | `baseline/auto_tune_8k_phase1_20260516_172245.md` | FP32, simple, stride=3 | 36.68 | 36.95 | decode 17.38, composite 7.60, sync 1.67, ORT 7.04 |
| B2 | `baseline/auto_tune_8k_phase1_20260516_172408.md` | FP32, simple, stride=6 | 37.05 | 37.01 | decode 21.07, composite 4.03, sync 1.53, ORT 3.49 |
| C | `baseline/auto_tune_8k_phase1_20260516_172542.md` | FP16, simple, stride=1 | 36.59 | 36.34 | decode 7.00, composite 18.58, sync 1.45, ORT 17.70 |
| D | `baseline/auto_tune_8k_phase1_20260516_171415.md` | FP16, threaded_serial, stride=1 | 36.50 | 36.32 | decode 0.06, composite 24.95, sync 2.06, ORT 24.13 |
| E1 | `baseline/auto_tune_8k_phase1_20260516_172723.md` | FP16, simple, stride=1, RVM bypass | 37.30 | 37.14 | decode 23.94, composite 0.58, sync 1.81, ORT 0.00 |

## 终局修正：PyNv P1 大写 preset 解除 8K 编码瓶颈

前一版“PyNv 8K HEVC 只能约 37fps”的判断仍然不完整。严格按 NVIDIA PyNvVideoCodec API 文档复核后，关键遗漏是 `preset` 大小写：PyNv 2.1.0 接受 `preset="P1"`（大写 P + 数字），但不接受 `preset="p1"`。

因此，正确结论是：

- 旧 `PT_PASSTHROUGH_MAX_FPS=30` 确实污染了前期 FPS 归因；
- 解除 cap 后看到的 `~37fps` 不是硬件上限，也不是 PyNv 必然上限；
- 它是 PyNv encoder 默认/错误 preset 配置造成的性能瓶颈；
- 使用 `preset="P1"` + `tuning_info="ultra_low_latency"` + `rc="cbr"` 后，PyNv 8K HEVC 编码和生产路径均突破 40fps。

## 官方格式复核

外部复核指出 NVIDIA PyNvVideoCodec API Programming Guide 示例使用：

```python
nvc.CreateEncoder(
    width=1920,
    height=1080,
    format="NV12",
    codec="hevc",
    preset="P2",
    tuning_info="low_latency",
)
```

本地验证确认 PyNv 2.1.0 的 `preset` 对大小写敏感：

- `preset="p1"`：初始化失败；
- `preset="P1"`：初始化成功，并显著提速；
- `preset="P2"`：初始化成功，但 8K 速度接近旧慢路径；
- 旧式 `LOW_LATENCY_HQ` 可初始化但更慢。

## P1 纯编码验证

所有测试使用 `tools/pynv_encode_probe.py --reuse-gpu-frame`，避免 CPU 造帧和每帧上传污染结果。

| 条件 | 结果 |
|---|---:|
| `preset=P1` | `72.83fps` |
| `preset=P1, tuning_info=ultra_low_latency` | `76.84fps` |
| `preset=P1, tuning_info=ultra_low_latency, rc=cbr` | `76.90fps` |
| `preset=P1, tuning_info=ultra_low_latency, rc=cbr, gop=30, idrperiod=30` | `76.11fps` |
| `preset=P2, tuning_info=ultra_low_latency, rc=cbr` | `38.68fps` |
| `preset=P3, tuning_info=ultra_low_latency, rc=cbr` | `35.54fps` |

结论：`P1` 是决定性参数；P2/P3/旧 preset 会回到 35-39fps 量级。

## P1 转码验证

`tools/pynv_transcode_probe.py` 已加入 `--preset`、`--tuning-info`、`--rc`、`--idrperiod` 和 `--enc-opt`，用于和生产路径一致地透传 PyNv encoder 参数。

命令：

```powershell
.venv\Scripts\python.exe tools\pynv_transcode_probe.py test_8k.mp4 `
  --duration 20 --fps 0 --codec hevc --bitrate 60000000 --gop 60 `
  --preset P1 --tuning-info ultra_low_latency --rc cbr `
  --progress 300 --out debug_output\probe_transcode_8k_hevc_p1_ull_cbr.mp4
```

结果：

- `throughput=79.46fps`；
- `avg_decode=11.267ms`；
- `avg_encode=0.331ms`；
- 输出仍为 `8192x4096 HEVC`。

这与 FFmpeg CUDA decode + `hevc_nvenc -preset p1 -tune ull` 的 `~80fps` 对照一致，说明 PyNv 在正确 P1 参数下可以达到同级性能。

## 生产路径修改

已在 `config.py` 增加生产配置：

- `PT_PASSTHROUGH_PYNV_PRESET`，默认 `P1`；
- `PT_PASSTHROUGH_PYNV_TUNING_INFO`，默认 `ultra_low_latency`；
- `PT_PASSTHROUGH_PYNV_RC`，默认 `cbr`；
- `PT_PASSTHROUGH_PYNV_IDR_PERIOD`，默认空。

已在 `pipeline/pynv_stream.py` 中统一生产与 preflight 的 encoder kwargs：

- `codec=hevc`；
- `bitrate=<effective_default_bitrate>`；
- `fps=<effective_fps>`；
- `gop=<PT_PASSTHROUGH_GOP>`；
- `bf=<PT_PASSTHROUGH_HEVC_BF>`；
- `preset=P1`；
- `tuning_info=ultra_low_latency`；
- `rc=cbr`。

## 生产 8K / stride=1 验证

使用 `videos/test_8k.mp4`，`PT_ALPHA_STRIDE=1`，FP16 RVM，`PT_PASSTHROUGH_MAX_FPS=0`，`threaded_serial` decoder。

| 报告 | 时长 | Avg FPS | Latest FPS | 关键 stage |
|---|---:|---:|---:|---|
| `baseline/auto_tune_8k_phase1_20260516_185207.md` | 20s | `55.62` | `56.41` | decode 0.05, composite 16.23, sync 1.10, encode 0.32, mux 0.02, ORT 15.61 |
| `baseline/auto_tune_8k_phase1_20260516_185313.md` | 60s | `56.16` | `56.57` | decode 0.04, composite 16.05, sync 1.24, encode 0.31, mux 0.02, ORT 15.47 |

结论：原始目标“8K、`ALPHA_STRIDE=1`、超过 40fps”已经达成，当前 60 秒稳态约 `56fps`。

## 更新后的有效判断

1. `PT_PASSTHROUGH_MAX_FPS=30` 是输出媒体 timestamp / 选帧 cap，不是 producer wall-clock 速率门；旧 cap=30 仍显示 36-37fps 并不矛盾。
2. `preset="P1"` 大写是 PyNv 2.1.0 8K HEVC 性能关键；小写 `p1` 会失败，P2/P3 仍慢。
3. 旧“PyNv 8K HEVC 约 37fps 上限”“应转 FFmpeg/PyAV”的判断废止。
4. 当前瓶颈重新回到 RVM/ORT：60 秒生产路径中 `ORT≈15.47ms`，总 FPS 约 `56fps`。
5. 后续若继续优化，应重新以 P1 配置为基线，再考虑 FP16/TRT/CUDA Graph/RVM 优化；不要再基于旧 37fps 数据决策。

## 验证命令

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\pynv_stream.py tools\pynv_transcode_probe.py tools\pynv_encode_probe.py
```

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 `
  --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 `
  --startup-timeout 240 --client-timeout 240 `
  --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx `
  --server-env PT_ALPHA_STRIDE=1 `
  --server-env PT_PASSTHROUGH_MAX_FPS=0 `
  --server-env PT_PASSTHROUGH_PYNV_DECODER=threaded_serial
```
