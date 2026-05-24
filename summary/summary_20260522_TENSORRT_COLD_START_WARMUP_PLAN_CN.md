# TensorRT 首帧 Warmup/加载时间优化计划

日期：2026-05-22

## 背景

当前 TensorRT static RVM 路径的稳态性能已经接近目标。以 `[VenusReality]Hannah02-8K.mp4` 为例，关闭 live adaptive FPS 后，输出从 24FPS 限制恢复到接近 40FPS：

- `source_fps=59.940 output_fps=40.000 fps_cap=40.000`
- `static_trt=True, iobinding=True`
- 稳态 `ort_run` 常见在 `5.5-8.7ms`
- 后段累计 FPS 达到 `39.9FPS`

但首帧/首包阶段仍有明显等待，播放器端可能把启动阶段计入平均 FPS 或表现为起播慢。

## 日志证据

来自 `debug_output/server.log` 的同一次请求：

- 请求开始：`12:18:36.144`
- 首次 RVM 诊断：`alpha #1 ... ort_run=1133.6ms`
- worker start：`12:18:38.856`
- source meta：`output_fps=40.000`
- batch=2 static TRT session 加载发生在请求中
- first real video bitstream：`12:18:39.867`
- first stdout chunk：约 `12:18:41` 附近

启动前几十帧累计 FPS 较低：

- frame 30: `fps=17.60`
- frame 60: `fps=23.22`
- frame 120: `fps=29.22`

稳态恢复：

- frame 600: `fps=39.03`
- frame 900: `fps=39.62`
- frame 1110: `fps=39.90`

## 当前判断

首帧慢主要不是稳态 TensorRT 性能问题，而是冷启动成本叠加：

1. ONNX Runtime / TensorRT static session 第一次加载。
2. batch=1 warmup 和实际 batch=2 SBS/alpha 路径不完全一致。
3. 第一次 CUDA/ORT/TensorRT 调用触发上下文、buffer、kernel、engine runtime 初始化。
4. PyNv / NVDEC / NVENC / FFmpeg mux pipeline 首次建立也有固定成本。

## 建议方案

### 方案 A：server 启动后台预热 TensorRT static sessions

在 server 启动后启动后台线程，提前创建 `Matter` 并加载 static TensorRT RVM session。

建议预热内容：

- batch=1 static RVM session
- batch=2 static RVM session
- 当前 `MATTING_INPUT_SIZE=1024`
- 当前 `RVM_DOWNSAMPLE_RATIO=0.5`
- 当前 `ONNX_TRT_FP16_ENABLE=1`
- CUDA Graph 维持当前关闭策略

目标：把第一次播放时的 session load / warmup 成本提前支付到 server 启动后。

优点：

- 改动面相对小。
- 不改变播放主路径。
- 用户等待主要发生在 server 启动阶段或后台空闲阶段。

风险：

- 启动后 GPU 显存会提前占用。
- 如果 UI/配置改变了模型、input size、downsample、provider，需要 invalidation/re-warmup。
- 需要避免多个 warmup 和真实请求同时构建同一个 TensorRT session。

### 方案 B：按实际播放路径预热 batch=2

当前日志显示先出现 batch=1 慢调用，随后请求中才加载 batch=2 static session。SBS alpha 实际稳态使用 batch=2，因此 warmup 应覆盖真实路径。

建议：

- warmup API 支持明确指定 batch 列表：`[1, 2]`
- 对 alpha/SBS 默认至少预热 batch=2
- warmup 记录日志，明确输出 batch、shape、provider、elapsed

### 方案 C：全局 session / Matter runtime 池

建立按配置 key 缓存的 runtime：

- model path
- providers
- input size
- downsample ratio
- fp16/cuda graph/trt cache path

可共享：

- ORT InferenceSession
- static TRT sessions
- 固定 model metadata

不可直接共享或需要隔离：

- RVM recurrent state OrtValue / CuPy buffers
- per-stream input/output binding buffers
- per-stream CUDA buffer lifecycle

这个方案收益可能最大，但需要专家重点审阅状态隔离风险。

### 方案 D：PyNv/NVDEC/NVENC 轻量预初始化

在后台做一次轻量 preflight：

- 初始化 CUDA/PyNv runtime
- 创建并释放一个小尺寸或目标尺寸 NVENC encoder
- 可选创建 decoder preflight

预期收益小于 TRT session warmup，但可以降低首包抖动。

## 推荐实施顺序

1. 先实现后台 warmup batch=1 + batch=2 static TRT sessions。
2. 加锁/状态标记，防止真实请求和 warmup 同时构建同一 session。
3. 在 UI 或日志中暴露 warmup 状态：`idle / warming / ready / failed`。
4. 再评估是否做全局 session 池。
5. 最后评估 PyNv/NVENC/NVDEC preflight。

## 需要专家审阅的问题

1. ORT InferenceSession 是否可以在多 stream 间安全共享？如果可以，是否只需隔离 RVM recurrent state 和 IOBinding buffers？
2. 当前 static TensorRT session 加载是否已经被 `Matter` 实例缓存？如果是，缓存粒度在哪里？
3. warmup 期间是否可能污染 RVM recurrent state？是否应使用专门的 throwaway state？
4. batch=1 是否仍然必要，还是 alpha/SBS 场景只预热 batch=2 即可？
5. TensorRT engine/context 是否存在 per-thread 或 per-CUDA-stream 限制？
6. 显存占用是否可接受？是否需要 warmup 可取消或延迟到首次进入 TensorRT 模式？
7. exe 打包环境下后台 warmup 是否需要额外 DLL/path 初始化顺序？

## 当前建议

优先做方案 A+B：server 启动后后台预热 batch=1/batch=2 static TensorRT RVM session。先不急于做全局 session 池，避免把 RVM state 隔离问题和冷启动优化绑在一起。
