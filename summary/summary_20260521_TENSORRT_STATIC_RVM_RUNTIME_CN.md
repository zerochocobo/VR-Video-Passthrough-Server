# TensorRT Static RVM Runtime Summary

日期：2026-05-21

## 背景

本阶段继续处理 TensorRT 在实时 RVM matting 路径中的性能和卡顿问题。此前已确认动态 ONNX Runtime TensorRT EP 路径存在两个核心问题：

- `TensorRTExecutionProvider + RVM IOBinding` 可能在 `run_with_iobinding()` 中长时间阻塞，导致首包超时。
- `TensorRTExecutionProvider + sess.run()` 可以跑通，但因为 RVM 被 ORT TensorRT EP 切成多个 fallback 子图，单帧耗时达到秒级，明显慢于纯 CUDA + IOBinding。

专家反馈指出根因很可能是 ORT TensorRT EP 对 RVM 动态图的 partition 碎片化，建议优先验证 static-shape ONNX 路线。

## 已完成

1. 新增静态 RVM ONNX 生成逻辑：
   - 新增 `utils/rvm_static_onnx.py`。
   - 为 batch 1 和 batch 2 生成固定输入尺寸的 TensorRT 专用 ONNX。
   - 固定 `src`、`r1i..r4i`、`fgr`、`pha`、`r1o..r4o` shape。
   - 移除运行时 graph input `downsample_ratio`。
   - 将 RVM 中触发 TensorRT parser 问题的动态 Resize scale tensor `388` 固定为 `[1, 1, downsample, downsample]`。

2. Runtime 改为静态 TensorRT 快路径：
   - `Matter` 检测 batch1/batch2 静态 TRT ONNX cache 是否存在。
   - 静态 cache 存在时，主 RVM session 改用 CUDA/CPU，只承担元数据和兜底。
   - 真正实时推理走独立的静态 TensorRT session + CUDA IOBinding。
   - 这样避免主 runtime session 再触发动态 TensorRT `Resize_3` parser 错误。

3. RVM recurrent state 修正：
   - 修复 TensorRT 路径下 `r1/r2/r3/r4` state channel fallback。
   - 静态 TensorRT recurrent state 和输出 OrtValue cache 已接入原有 SBS slot 机制，避免左右眼状态串线。

4. Warmup/cache 构建流程调整：
   - `ui/services/trt_warmup_process.py` 现在会生成 static batch1 和 batch2 ONNX。
   - 静态 engine 直接用 ORT TensorRT session 构建。
   - 移除旧的动态 stage 3 构建/固化慢路径。
   - 正式构建结果：`DONE:total_seconds=186`，其中 stage 3 为 `0s`。

5. 诊断日志改进：
   - matting 日志中现在会显示类似：
     `providers={'main': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'static_trt': True, 'iobinding': True}`
   - 这样不会因为主 session providers 是 CUDA/CPU 而误判 TensorRT 没启用。

6. 注释掉限制 FPS 的逐帧 debug 日志：
   - 已注释 `pipeline/matting.py` 中的：
     - `nv12->nv12 y kernel returned ...`
     - `nv12->nv12 uv kernel returned ...`
     - `nv12->nv12 mono kernel returned ...`
   - 用户报告的日志：
     `nv12->nv12 mono kernel returned in 0.02xms`
     是 info 级逐帧输出，会污染 `server.log`，在 debug 开启时也可能影响实时 FPS。
   - 现在默认不会输出这些 kernel-return 逐帧日志；需要做 CUDA kernel profiling 时再局部恢复。

## 本地验证结果

- `uv` 环境已能导入 TensorRT EP：
  - `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- 静态 TRT runtime probe：
  - 主 session providers：`['CUDAExecutionProvider', 'CPUExecutionProvider']`
  - `static_trt=True`
  - `iobinding=True`
  - `(2,3,1024,1024)` 后续推理约 `5-10ms`
- 正式 warmup：
  - 命令：`.\.venv\Scripts\python.exe main.py trt_warmup --input-size 1024 --downsample 0.5 --fp16 1 --cuda-graph 0 --progress-stdout`
  - 结果：`DONE:total_seconds=186`
  - cache status：`ready`
- 测试：
  - `tests/test_rvm_static_onnx.py`
  - `tests/test_trt_warmup_process.py`
  - `tests/test_trt_manifest.py`
  - `tests/test_matting_runtime_policy.py`
  - 结果：`13 passed`
- 编译检查：
  - `pipeline/matting.py`
  - `ui/services/trt_warmup_process.py`
  - `utils/rvm_static_onnx.py`
  - 通过。

## 还需要解决的问题

1. 首次静态 TRT 推理仍有 lazy session/cache 开销：
   - probe 中第一帧可能是数百毫秒。
   - 后续帧已降到 `5-10ms`。
   - 后续可考虑在 server startup 或 warmup 阶段预加载 static batch1/batch2 session，减少第一帧延迟。

2. 需要真实 4K/8K 播放复测端到端 FPS：
   - 当前只确认 RVM 静态 TRT 推理本身很快。
   - 还需要看完整链路：decode、preprocess、matting、NV12 composite、encode、mux、DLNA client backpressure。

3. TensorRT static session 仍会输出部分 warning：
   - 主要是 unused/empty initializer 相关 warning。
   - 目前不影响 cache ready 和推理性能。
   - 如果日志仍过多，可再做 ONNX cleanup，移除静态模型中的无用 initializer。

4. TensorRT cache 构建仍较久：
   - 当前约 186 秒。
   - 已比动态旧流程短很多，但首次构建仍然需要等待。
   - 后续可考虑 UI 中更明确展示 stage 耗时和提示。

5. 动态 ORT TensorRT EP 路线不建议继续作为实时主路径：
   - 动态 RVM 图仍存在 `Resize_3` parser/partition 问题。
   - 产品路径应优先使用 static TRT + IOBinding，或继续保留 CUDA + IOBinding 作为稳定兜底。

