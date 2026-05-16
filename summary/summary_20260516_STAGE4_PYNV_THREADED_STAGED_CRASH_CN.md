# 阶段 4 问题报告 - PyNv Threaded Staged Pipeline 崩溃

## 背景

本文记录一次失败的 8K realtime passthrough 第四阶段优化尝试。

第四阶段原目标是把当前串行 live 处理路径拆成三段：

- decode；
- RVM matting 和 GPU NV12 composite；
- NVENC encode 和 mux。

预期收益是让 NVDEC、CUDA/ORT、NVENC/mux 互相重叠，长期目标是达到 8K realtime `40fps` 且 `ALPHA_STRIDE=1`。

本次工作没有得到可用结果，并且在 Windows 上触发了 native `python.exe` 崩溃弹窗。在专家复核前，应视为不安全路径。

## 环境和测试素材

- 仓库：`G:\GIT\debug\PTMediaServer`
- 测试视频：`videos/test_8k_2.mp4`
- 平台：Windows
- 相关 native 组件：PyNvVideoCodec、CUDA、CuPy、ONNX Runtime CUDA EP、NVENC
- 第四阶段前相对稳定的生产路径：
  - PyNv SimpleDecoder；
  - Matter GPU matting/composite；
  - PyNv NVENC；
  - FFmpeg mux；
  - `pipeline/pynv_stream.py` 中的串行 worker loop。

## 第四阶段依赖的前置安全结论

阶段 1 已证明 ThreadedDecoder 内容映射可以正确，但必须遵守严格生命周期规则：

- `ThreadedDecoder(start_frame=N)` 的内容映射到 `SimpleDecoder[N + local_sequence]`；
- 不能用 `getPTS()` 作为帧身份；
- 返回帧必须在下一次 `get_batch_frames()` 调用前、以及 `end()` 前消费完。

阶段 3 已证明 NVENC 输入生命周期也不能立即复用：

- `Encode()` 返回后立刻复用或覆盖同一个 GPU NV12 输入 slot 是不安全的；
- 3-slot 延迟复用 ring 通过了合成 8K HEVC 探针；
- 因此 live green 路径改为延迟释放 NV12 输出 slot。

## 第四阶段尝试中改动了什么

本次没有把 live route 正式改成 staged pipeline。

主要改动发生在离线探针：

- `tools/pynv_fullchain_probe.py`
  - 实现 `--pipeline staged`；
  - 添加 decode / matting / encode 三个 worker 线程；
  - 添加有界队列；
  - 通过 `configure_gpu_runtime_cache()` 对齐生产环境的 runtime cache 初始化；
  - 默认禁用旧的固定 `tempfile.TemporaryDirectory` monkey patch；
  - 添加 `--decoder simple|threaded`；
  - 发生 native 崩溃后，显式禁用 `--decoder threaded`。

另在以下文件临时加入诊断日志：

- `pipeline/matting.py`
  - 受 `DEBUG_LOGS` 控制；
  - 用于观察 GPU NV12-to-NV12 composite kernel 边界。

## 过程中发现的一个离线探针问题

`tools/pynv_fullchain_probe.py` 过去会把 `tempfile.TemporaryDirectory` patch 到一个固定目录。

这会导致 CuPy / RawKernel 路径在 NV12 composite 阶段卡住。修复方式：

- 使用 `configure_gpu_runtime_cache()`；
- 默认恢复正常临时目录行为；
- 旧 fixed-tempdir 行为改成仅在 `PT_PYNV_FULLCHAIN_FIXED_TEMPDIR=1` 时启用。

这个问题看起来和后续 native 崩溃不是同一个问题。

## 已通过的命令

SimpleDecoder staged smoke 通过：

```powershell
uv run python tools\pynv_fullchain_probe.py videos\test_8k_2.mp4 --pipeline staged --frames 30 --discard 3 --fps 30 --codec hevc --bitrate 50000000 --alpha-stride 3 --input-size 1024 --raw-video-out debug_output\stage4_staged_30.hevc --out debug_output\stage4_staged_30.mp4 --json-out baseline\pynv_fullchain_stage4_staged_smoke_20260515.json --progress 10
```

更长的 SimpleDecoder staged 也通过：

```powershell
uv run python tools\pynv_fullchain_probe.py videos\test_8k_2.mp4 --pipeline staged --frames 120 --discard 10 --fps 30 --codec hevc --bitrate 50000000 --alpha-stride 3 --input-size 1024 --raw-video-out debug_output\stage4_staged_120.hevc --out debug_output\stage4_staged_120.mp4 --json-out baseline\pynv_fullchain_stage4_staged_120_20260515.json --progress 30
```

报告：

- `baseline/pynv_fullchain_stage4_staged_smoke_20260515.json`
- `baseline/pynv_fullchain_stage4_staged_120_20260515.json`

120 帧 SimpleDecoder staged 结果概要：

- 能跑完；
- steady FPS 约 `31.76`；
- 仍然达不到 40fps 目标；
- decode 仍然昂贵，因为还是 indexed/random `SimpleDecoder` 访问。

## 为什么 SimpleDecoder Staged 不够

SimpleDecoder staged 仍然使用当前 indexed decode 模式。对于 8K 59.94fps 源到 30fps 输出，代码会反复选择 CFR source index 并调用 indexed decode。

因此 staged pipeline 虽然能重叠部分工作，但 decode 仍然是瓶颈：

- decode average 仍约 `27 ms`；
- matting steady average 约 `8.9 ms`；
- encode steady average 约 `6.4 ms`。

所以这不是有效的第四阶段性能突破。

## 失败的 ThreadedDecoder Staged 尝试

为了降低 decode 成本，离线 staged probe 尝试接入 `PyNvVideoCodec.ThreadedDecoder`。

尝试命令形态：

```powershell
uv run python tools\pynv_fullchain_probe.py videos\test_8k_2.mp4 --pipeline staged --decoder threaded --frames 120 --discard 10 --fps 30 --codec hevc --bitrate 50000000 --alpha-stride 3 --input-size 1024 --raw-video-out debug_output\stage4_staged_threaded_120.hevc --out debug_output\stage4_staged_threaded_120.mp4 --json-out baseline\pynv_fullchain_stage4_staged_threaded_120_20260515.json --progress 30
```

观察到的行为：

- 控制台进度能跑到约 `90/120`；
- 随后进程失败，未写出 JSON；
- Windows 反复弹出 native `python.exe` 崩溃对话框。

用户看到的崩溃文本：

```text
python.exe - 应用程序错误:
0x00007FFFCCC27880 指令引用了 0x0000000B69200000 内存。
该内存不能为 read。
```

这是 native access violation，不是普通 Python 异常，不能靠 `try/except` 安全处理。

## 最可能原因

最可能原因是 PyNv ThreadedDecoder 返回帧的生命周期处理错误。

阶段 1 映射工作已经确认：

- ThreadedDecoder 返回帧只保证在下一次 `get_batch_frames()` 前有效；
- 必须在拉下一批之前消费数据；
- 在消费返回帧之前调用 `end()` 是无效用法。

失败的 staged 设计违反了这个模型：

- decode worker 调用 `get_batch_frames()`；
- 选中的帧被 wrap 后放入跨线程队列；
- decode worker 继续拉后续 batch；
- matting worker 稍后才消费早先的 frame 对象。

这意味着 matting 阶段可能读取了 PyNv 已经失效或复用的底层 GPU 指针。

结合 native access violation 和内存地址特征，表现符合 CUDA/PyNv use-after-free 或无效 device pointer read。

## 额外发现：Slot 回压风险

最初 SimpleDecoder staged 实现还遇到过：

```text
RuntimeError: no free NV12 output slot: count=3 shape=(6144, 8192)
```

原因：

- 阶段 3 的延迟释放策略会保留最近的 NV12 输出 slot，以保护 NVENC 输入生命周期；
- staged matting 可能跑得比 encode 快，请求第 4 个 slot；
- 当前 Matter slot API 没有阻塞等待，而是立即抛错。

离线探针临时处理：

- staged probe 里对 slot acquire 做等待重试。

需要专家判断：

- `Matter.acquire_nv12_output_slot()` 是否应该演进成 staged pipeline 可用的阻塞 API；
- 或者 staged pipeline 应该在 Matter 外部自己管理 slot 调度。

## 已采取的止血措施

`tools/pynv_fullchain_probe.py --decoder threaded` 现在会直接 `SystemExit` 禁止运行。

当前提示：

```text
ThreadedDecoder is temporarily disabled in the staged full-chain probe.
PyNv ThreadedDecoder frames are only valid until the next get_batch_frames() call;
passing them across worker threads caused native Python/PyNv crashes during Phase 4 testing.
```

在专家复核前，不应继续运行 Python/GPU 相关探针。

## 希望专家重点回答的问题

1. PyNvVideoCodec `ThreadedDecoder.get_batch_frames()` 返回帧的正确 ownership/lifetime 模型是什么？
2. 是否有官方方式可以 retain/copy decoded frame，让它在下一次 `get_batch_frames()` 后继续有效？
3. 是否可以在拉下一批之前，把 decoded frame device-to-device 复制到用户持有的 CuPy NV12 buffer？
4. 如果可以，应该用什么 copy primitive：CuPy assignment、`cudaMemcpy2DAsync`、PyNv API，还是其他 CUDA interop？
5. ThreadedDecoder 输出属于哪个 CUDA stream？用户 stream 应该如何等待 decode 完成？
6. 如果 Python 层持有 PyNv frame object 引用，`frame.cuda()` 内存是否可以跨线程使用？还是仍然受 batch 生命周期限制？
7. 是否应该把 decode 和 matting 放在同一个 worker 里，保证 ThreadedDecoder frame 在下一批前被消费？
8. 没有 owned intermediate decoded-frame ring 的情况下，ThreadedDecoder 三段流水线是否可行？
9. 多线程和队列参与时，ThreadedDecoder 的正确 shutdown 顺序是什么？
10. PyNvVideoCodec 是否暴露 CUDA event、stream handle 或引用计数帧所有权 API？

## 可能更安全的设计

### 方案 A - 自有 Decode Ring

Decode 线程：

1. 调用 `get_batch_frames()`；
2. 对每个选中帧，立刻把 Y/UV planes 复制到用户持有的 GPU NV12 ring buffer；
3. 只把 owned buffer 放入 matting 队列；
4. 当前 batch 的选中帧复制完以后，才调用下一次 `get_batch_frames()`。

这个方案遵守 ThreadedDecoder frame 生命周期，但会增加一次 GPU copy。

### 方案 B - Decode 和 Matting 合并为一段

把 ThreadedDecoder 和 matting 放在同一个 worker：

1. 拉 batch；
2. 对选中帧先完成 matting/composite；
3. 在拉下一批之前，只把 Matter-owned NV12 encode slot 放给 encode worker。

这个方案可能降低并发度，但避免跨阶段传递 ThreadedDecoder frame。

### 方案 C - 第四阶段继续使用 SimpleDecoder

只使用 SimpleDecoder staged，把性能优化转向其他方向。

这个方案更安全，但 120 帧探针没有达到目标。

## 当前建议

不要继续 ThreadedDecoder staged 实现，除非先得到外部专家对 ownership/lifetime 的明确建议。

下一次尝试应先做一个很小的独立 ownership probe：

```text
ThreadedDecoder get_batch_frames()
把选中帧复制到自有 GPU NV12 buffer
拉下一批
在原 batch 失效后使用 copied buffer
hash/encode/compare copied content
跨多批重复验证
```

只有这个探针通过后，才能恢复第四阶段。

## 相关文件

- `tools/pynv_fullchain_probe.py`
- `tools/pynv_threaded_mapping_probe.py`
- `tools/pynv_threaded_decode_probe.py`
- `tools/pynv_encode_lifetime_probe.py`
- `pipeline/pynv_io.py`
- `pipeline/matting.py`
- `pipeline/pynv_stream.py`

## 状态

阶段 4 阻塞。

本次工作产生了一些诊断结论，但没有交付预期性能提升，并且引入了 native 崩溃风险。在重新设计和隔离验证 ownership/lifetime 模型之前，ThreadedDecoder staged mode 必须保持禁用。
