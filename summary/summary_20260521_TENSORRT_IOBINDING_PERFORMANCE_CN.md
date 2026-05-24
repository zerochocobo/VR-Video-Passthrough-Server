# TensorRT IO Binding 性能与卡死问题小结

日期：2026-05-21

## 结论

当前我不能把这个问题完整解决到“TensorRT 实时性能稳定超过纯 CUDA”的状态。

已确认 TensorRT cache 可以构建、服务器可以启用 TensorRT、实时播放也能跑通。但在实时 RVM 路径上存在核心矛盾：

- `TensorRTExecutionProvider + RVM IOBinding`：会在 `onnxruntime.InferenceSession.run_with_iobinding()` 内卡死，导致首包 30 秒超时和 HTTP 504。
- `TensorRTExecutionProvider + 普通 sess.run()`：能稳定播放，但 4K alpha 实时 FPS 明显低于纯 CUDA + IOBinding。
- `CUDAExecutionProvider + RVM IOBinding`：当前项目中实时性能更好，是现有生产路径。

这个问题更像 ONNX Runtime TensorRT EP、IOBinding、RVM 多输入多输出 recurrent state、动态 shape/profile 之间的交互问题，需要熟悉 ORT TensorRT EP 内部行为或 TensorRT profile/graph partition 的专家继续处理。

## 环境

- Windows
- GPU：NVIDIA GeForce RTX 5060 Ti
- Driver：581.57
- ONNX Runtime：1.25.1
- Providers：
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`
- TensorRT Python package：10.16.1.11
- 模型：`models/rvm_mobilenetv3_fp32.onnx`
- TensorRT runtime model：
  - `runtime_cache/trt_engines/rvm_mobilenetv3_fp32_shape_inferred.onnx`
- TensorRT cache：
  - `runtime_cache/trt_engines/manifest.json`
  - usable engine about 10.6 MB

## 已完成的 TensorRT 功能

- UI 增加 TensorRT 配置、构建、状态显示、启用开关。
- TensorRT cache manifest：
  - fingerprint 包含 GPU、driver、CUDA runtime、TensorRT、ORT、model sha256、input size、downsample、FP16、CUDA graph。
  - cache 状态支持 `missing` / `ready` / `stale` / `failed`。
- warmup 构建流程：
  - shape inference 生成 `rvm_mobilenetv3_fp32_shape_inferred.onnx`。
  - 强制 TensorRT/CUDA/CPU provider chain。
  - 检测 ORT provider fallback，防止 false-ready manifest。
  - ready cache 需要 shape-inferred ONNX 和大于 1 MiB 的 `.engine`。
- 运行时：
  - `main.py` 在 TensorRT cache ready 时把 `config.MODEL_PATH` 切到 shape-inferred ONNX。
  - dev Python 和 frozen exe 均注入 TensorRT/CUDA DLL path。
- RVM dynamic symbol 修复：
  - 原 ONNX 中 `src` 和 recurrent states 复用了 `height` / `width` symbol。
  - TensorRT 误认为 recurrent states 与源图像 H/W 相同，导致 profile constraint 冲突。
  - 已在 warmup 前重命名 recurrent state input/output symbolic dims。

## 关键问题 1：TensorRT + IOBinding 卡死

出错日志来自 `debug_output/server.log`。

播放 8K green 实时流时，TensorRT provider 已启用：

```text
Matting model loaded ... active=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] ... rvm_iobinding=True
```

随后 worker 卡在：

```text
pipeline/matting.py", line 1823, in _run_rvm_iobinding_from_dev
    self.sess.run_with_iobinding(binding)
```

结果：

```text
return 504 first chunk timeout after 30.0s
PyNv runtime marked tainted because worker did not stop
```

因为 `run_with_iobinding()` 不返回，Python 层的异常 fallback 无法触发。

## 当前规避方案

为避免实时服务器卡死，我增加了策略：

- CUDA-only 时保留 RVM IOBinding。
- TensorRT provider active 时禁用 RVM IOBinding。

相关代码：

- `pipeline/matting.py`
  - `_should_enable_rvm_iobinding(active_providers)`

当前 TensorRT 路径日志：

```text
active=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
rvm_iobinding=False
```

这个规避方案能保证稳定播放，但牺牲性能。

## 关键问题 2：TensorRT 普通 sess.run 性能低于 CUDA IOBinding

4K alpha 测试日志显示 TensorRT 确实启用：

```text
ONNX providers requested=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
trt cache ready; ONNX providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
runtime_model=...\runtime_cache\trt_engines\rvm_mobilenetv3_fp32_shape_inferred.onnx
Matting model loaded ... active=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] ... rvm_iobinding=False
```

4K alpha 视频：

```text
size=4096x2048
output_mode=alpha
input_shape=(2,3,1024,1024)
```

性能日志：

```text
frame 300 ... fps=26.53 ... mat_avg_ms pre=4.61 ort=24.76
frame 600 ... fps=26.94 ... mat_avg_ms pre=5.04 ort=27.64
frame 840 ... fps=26.50 ... mat_avg_ms pre=5.22 ort=27.47
```

结论：

- decode / encode / mux 都不是主要瓶颈。
- 主要瓶颈是 RVM matting 推理。
- TensorRT 普通 `sess.run()` 下 `ort_run` 大约 24-27ms。
- 加上 preprocess 后 matting 已接近 30ms，整体只能约 26-27 FPS。

纯 CUDA + IOBinding 的历史日志显示 8K green 路径下：

```text
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
rvm_iobinding=True
frame=8192x4096 input_shape=(2,3,1024,1024)
ort_run=18-19ms
```

虽然不是同一 alpha/green 场景的完全严格 A/B，但已经足够说明当前 TensorRT 普通 `sess.run()` 没有性能优势。

## 外部专家建议重点检查

1. ONNX Runtime TensorRT EP 是否支持当前 RVM 多输入、多输出、recurrent state 的 CUDA IOBinding。
2. `run_with_iobinding()` 卡死是否是：
   - TensorRT EP bug；
   - output binding shape/type 不匹配；
   - recurrent OrtValue 生命周期/ownership 问题；
   - CUDA stream/synchronization 问题；
   - TensorRT partition fallback 后与 CUDA EP 混合执行导致的问题。
3. 是否需要为 TensorRT 单独禁用/调整某些 ORT provider options。
4. 是否可以把 RVM recurrent state 改为更 TensorRT-friendly 的 graph/state 管理方式。
5. 是否可以离线导出更静态 shape 的 RVM ONNX，避免 ORT TensorRT EP 对 Resize/dynamic shape 的部分 partition。
6. 是否可以用 TensorRT engine 直接推理，而不是通过 ORT TensorRT EP。

## 当前建议

在专家解决 `TensorRT + IOBinding` 之前，不建议把 TensorRT 作为默认实时后端。

建议策略：

- 默认实时播放：CUDA + RVM IOBinding。
- TensorRT：保留为实验/手动启用选项。
- UI 应提示 TensorRT 当前可能不提升实时 FPS。

## 相关文件

- `pipeline/matting.py`
- `ui/services/trt_warmup_process.py`
- `utils/trt_manifest.py`
- `utils/runtime_dll_paths.py`
- `main.py`
- `ui/settings.py`
- `ui/pages/home_page.py`
- `tests/test_matting_runtime_policy.py`
- `tests/test_trt_warmup_process.py`
- `tests/test_runtime_dll_paths.py`

## 已通过测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_exe.py tests/test_runtime_dll_paths.py tests/test_matting_runtime_policy.py tests/test_trt_manifest.py tests/test_trt_warmup_process.py tests/test_settings.py tests/test_i18n.py tests/test_ui_smoke.py tests/test_main_args.py -q
```

结果：

```text
31 passed, 7 subtests passed
```
