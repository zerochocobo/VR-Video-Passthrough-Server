> ⚠️ SUPERSEDED 2026-05-16
> 本报告中的 FPS、sync、瓶颈归因结论建立在旧默认值 PT_PASSTHROUGH_MAX_FPS=30 和/或非目标 PT_ALPHA_STRIDE=3 条件下，已被后续复核证伪或降级为仅适用于旧诊断条件。
> 重新基线与有效结论入口见 summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_CN.md。
> 仅保留为研究过程档案；与 cap 无关的实现结论仍需按正文中的适用范围判断。
# 阶段 4R/P0 小结 - Profiling 复核与 TensorRT 阻塞点

## 背景

本阶段根据外部专家复核意见继续推进，优先回答三个疑惑点：

1. `composite` 约 15 ms 中，RVM 推理和绿幕合成 kernel 各占多少。
2. `cuda_stream.synchronize()` 约 10-11 ms 到底在等哪一类 GPU 工作。
3. TensorRT EP 是否可以作为下一阶段 P1 直接推进。

本阶段没有继续 Python 多阶段流水线，也没有恢复已确认有 native crash 风险的 ThreadedDecoder staged 设计。

## 代码变更

- `pipeline/pynv_stream.py`
  - 生产诊断日志新增 `mat_avg_ms pre/ort/kernel` 和 `mat_max_ms pre/ort/kernel`。
  - 目的是把原来的 `composite` 大项拆成 matting preprocess、ORT/RVM、NV12 composite kernel。

- `tools/auto_tune_8k.py`
  - parser 支持新的 matting 子阶段字段。
  - Markdown 报告新增 `Latest mat avg ms`。
  - 新增 `--server-env KEY=VALUE`，用于把实验环境变量显式传给 auto_tune 启动的服务进程，并写入报告，避免环境覆盖不可复现。

- `config.py`
  - 新增 TensorRT EP 相关配置：
    - `PT_ONNX_TRT_ENGINE_CACHE_ENABLE`
    - `PT_ONNX_TRT_ENGINE_CACHE_PATH`
    - `PT_ONNX_TRT_FP16_ENABLE`
    - `PT_ONNX_TRT_CUDA_GRAPH_ENABLE`
  - 新增 `PT_PASSTHROUGH_PYNV_SYNC_PROBE` 诊断开关。

- `pipeline/matting.py`
  - `TensorrtExecutionProvider` 被选中时传入 TRT FP16、engine cache、CUDA Graph provider options。
  - 新增 ONNX provider 诊断日志：wanted / available / selected。
  - `PT_PASSTHROUGH_PYNV_SYNC_PROBE=1` 时，在 PyNv green GPU 路径内部对 upload、RVM/alpha、composite 分段同步并打印 `[DIAG] pynv sync probe ...`。

## P0 测试与结论

### 1. 60s / 全片 green 稳态基线

报告：

- `baseline/auto_tune_8k_phase1_20260516_154731.md`
- `baseline/auto_tune_8k_phase1_20260516_154731.json`

说明：用户指定 60s，但 `videos/test_8k_2.mp4` 本身约 26s，实际跑完整片 785 帧。

结果：

- HTTP status: `200`
- first byte: `2.765 s`
- latest interval FPS: `36.40`
- average interval FPS: `36.13`
- stage avg:
  - decode: `0.05 ms`
  - composite: `15.98 ms`
  - sync: `11.00 ms`
  - encode: `0.40 ms`
  - mux: `0.02 ms`
- mat avg:
  - preprocess: `0.11 ms`
  - ORT/RVM: `15.17 ms`
  - kernel: `0.62 ms`

结论：

- 专家判断“不要继续堆 Python 多阶段”是正确的。
- `composite` 主要不是绿幕合成 kernel，而是 RVM/ORT 推理。
- 当前自定义 composite kernel 只占约 `0.5-0.6 ms`，远低于 30%，所以 P3 custom composite kernel 不是近期优先项。

### 2. Nsight Systems 尝试

已确认 Nsight Systems CLI 存在：

- `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe`

尝试结果：

- 第一次 `--trace=osrt` 在当前 Windows Nsight 版本中无效。
- 第二次直接 profile `uv` shim 失败。
- 第三次 profile `.venv\Scripts\python.exe tools\auto_tune_8k.py ...` 成功生成：
  - `baseline/nsys_stage4r_green_20260516_154929.nsys-rep`
  - `baseline/nsys_stage4r_green_20260516_154929.sqlite`

但该 Nsight 报告不可作为 CUDA timeline 依据：

- `nsys stats` 只看到 parent process 的少量 CUDA API，例如 `cudaStreamCreateWithPriority`。
- 没有 CUDA kernels、GPU memory、NVTX、NVVIDEO 等关键数据。
- 原因：被 profile 的是 `auto_tune_8k.py` 父进程，真正 CUDA 工作在它启动的服务子进程中。

结论：

- 需要后续改为直接 profile 服务进程，或使用 Nsight 支持的 attach / child-process 方式。
- 当前 `.nsys-rep` 只能记录“抓错进程”这个事实，不能支持 GPU 重叠结论。

### 3. `PT_PASSTHROUGH_PYNV_SYNC_PROBE=1` 同步归因短测

命令：

```powershell
$env:PT_PASSTHROUGH_PYNV_SYNC_PROBE='1'
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 6 --startup-timeout 240 --client-timeout 120
```

报告：

- `baseline/auto_tune_8k_phase1_20260516_155457.md`
- `baseline/auto_tune_8k_phase1_20260516_155457.json`

结果：

- latest interval FPS: `36.46`
- stage avg:
  - decode: `0.07 ms`
  - composite: `26.35 ms`
  - sync: `0.01 ms`
  - encode: `0.45 ms`
  - mux: `0.54 ms`
- mat avg:
  - preprocess: `0.05 ms`
  - ORT/RVM: `7.22 ms`
  - kernel: `18.98 ms`

关键日志例子：

```text
[DIAG] pynv sync probe nv12 frame=240 upload_sync=20.74ms alpha_call=0.03ms alpha_tail_sync=0.02ms composite_sync=1.64ms
```

解释：

- 开启 sync probe 后，外层 `sync` 从约 `10-11 ms` 降到 `0.01 ms`，说明原外层 sync 主要是在等前面已经 enqueue 的 GPU 工作完成。
- 但该模式改变了时间归因：强制同步会把等待时间前移到 upload/composite 调用内部。
- 日志显示非推理帧也可能在 `upload_sync` 等待约 `20-25 ms`，这更像是在等 PyNv/ThreadedDecoder 返回帧背后的 GPU 生产或跨流可见性，而不是绿幕合成 kernel 本身慢。
- 因此 `mat_kernel=18.98 ms` 不能解释为 kernel 算法成本，它是诊断同步导致等待被归入 kernel/upload 区间。

结论：

- `cuda_stream.synchronize()` 的 10-11 ms 不是 CPU 空转问题，而是在等待前序 GPU 工作完成。
- 仅把同步点换位置不能提升单 session FPS。
- 下一步如果要彻底证明等待对象，仍需要正确抓服务进程的 Nsight timeline。

## TensorRT EP P1 预检查

### Provider 可用性

本地 venv 检查：

```powershell
@'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
'@ | .venv\Scripts\python.exe -
```

结果：

- ORT: `1.25.1`
- available providers:
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`

### 显式 TRT auto_tune 短测

命令：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 `
  --video videos\test_8k_2.mp4 `
  --profile quest `
  --prefer green `
  --duration 4 `
  --startup-timeout 900 `
  --client-timeout 240 `
  --server-env PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider `
  --server-env PT_ONNX_TRT_ENGINE_CACHE_ENABLE=1 `
  --server-env PT_ONNX_TRT_FP16_ENABLE=1 `
  --server-env PT_ONNX_TRT_CUDA_GRAPH_ENABLE=1
```

报告：

- `baseline/auto_tune_8k_phase1_20260516_155825.md`
- `baseline/auto_tune_8k_phase1_20260516_155825.json`

结果：

- latest interval FPS: `36.53`
- mat ORT/RVM: `15.21 ms`
- 实际 active providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

关键日志：

```text
[DIAG] ONNX providers wanted=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] available=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] selected=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
*************** EP Error ***************
Please install TensorRT libraries as mentioned in the GPU requirements page, make sure they're in the PATH or LD_LIBRARY_PATH, and that your GPU is supported.
```

结论：

- ORT Python 包暴露了 TensorRT EP。
- 服务进程也正确接收了 `PT_ONNX_PROVIDERS=TensorrtExecutionProvider,...`。
- 但 TensorRT runtime libraries 没有被 ORT 找到，导致 Session 初始化时自动回退到 CUDA EP。
- 本机未发现明显 TensorRT 安装目录；`runtime_cache/trt_engines` 也没有生成 engine cache。
- 因此 P1 不能直接进入“性能优化实现”，必须先解决 TensorRT runtime DLL / PATH / 版本兼容问题。

## 当前对专家建议的逐条判断

- P0 profiling：部分完成。
  - 已通过内置计时证明 RVM/ORT 是主要 `composite` 成本。
  - 已证明 custom composite kernel 不是近期优先项。
  - Nsight timeline 仍未完成，因为第一次抓到了父进程而不是服务子进程。

- P1 TensorRT EP：方向正确，但当前被环境阻塞。
  - 代码开关和 provider options 已准备好。
  - 阻塞点是 TensorRT native libraries 不在 PATH 或未安装。

- P2 CUDA Graph：暂缓。
  - TRT EP 尚未真正启用，无法验证 TRT 内置 CUDA Graph。
  - CUDA EP 显式 graph capture 需要固定 IOBinding 指针，风险和工作量高于先解决 TRT runtime。

- P3 composite custom kernel：不建议现在做。
  - 稳态数据中 kernel 约 `0.5-0.6 ms`。

- P4 encode/composite event chain：暂缓。
  - 可能释放 CPU worker 等待，但当前单 session FPS 主要受 GPU 等待限制。

## 阻碍与风险

- TensorRT EP 需要安装或加入 PATH 的 TensorRT runtime DLL，当前环境不满足。
- Nsight 必须抓服务进程，否则无法回答 NVDEC/RVM/composite 是否真正 overlap。
- `PT_PASSTHROUGH_PYNV_SYNC_PROBE=1` 只用于短跑诊断，不能作为性能优化默认配置。
- `DEBUG_LOGS=1` 下 `nv12->nv12 gpu composite begin` 日志非常多，长测会放大日志体积。
- 不能恢复 `tools/pynv_fullchain_probe.py --decoder threaded` staged 路径；该路径仍然有已知 native crash 风险。

## 建议下一阶段操作

1. 先解决 TensorRT runtime 环境：
   - 安装与当前 ORT/CUDA 兼容的 TensorRT；
   - 或把现有 TensorRT `lib/bin` 路径加入服务进程 PATH；
   - 然后复跑 `baseline/auto_tune_8k_phase1_20260516_155825.md` 对应命令。

2. 若 TRT active providers 变成 `['TensorrtExecutionProvider', ...]` 且生成 engine cache，再跑完整片 green 基线。

3. 同时准备一个“直接 profile 服务进程”的 Nsight 命令或脚本，避免再次抓到 auto_tune 父进程。

4. 在 TRT 或 Nsight 有明确结果前，不继续新增 Python staged pipeline。

## 简单验证命令

编译：

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\matting.py pipeline\pynv_stream.py tools\auto_tune_8k.py
```

检查 ORT provider：

```powershell
@'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
'@ | .venv\Scripts\python.exe -
```

CUDA EP green 基线：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 10 --startup-timeout 240 --client-timeout 120
```

TRT EP 环境复测：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 4 --startup-timeout 900 --client-timeout 240 --server-env PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider --server-env PT_ONNX_TRT_ENGINE_CACHE_ENABLE=1 --server-env PT_ONNX_TRT_FP16_ENABLE=1 --server-env PT_ONNX_TRT_CUDA_GRAPH_ENABLE=1
```
