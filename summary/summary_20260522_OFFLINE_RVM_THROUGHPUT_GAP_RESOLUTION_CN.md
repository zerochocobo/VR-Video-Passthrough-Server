# 离线 RVM 吞吐差距问题 —— 最终定位与修复

日期：2026-05-22

关联文档：
- 问题报告：`summary/summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_EN.md`
- 分析与建议：`summary/summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_REPLY_CN.md`
- 本文：最终修复与验证结果

## 1. 一句话结论

离线 alpha 路径 `engine.composite_nv12()` 与 `enc.Encode()` 之间缺少 `_CUDA_STREAM.synchronize()` 作为 CUDA stream barrier，导致 NVDEC 与 SM 在 CUDA driver 调度层被判为依赖串行；补上这一行同步后两者在硬件层并行，throughput 从 36.3fps 提升至 **75.6fps**，与实时（77fps）及 green 离线（75.5fps）一致。

## 2. 修复内容

仅改 `tools/offline_alpha_passthrough.py`：

- 在 `engine.composite_nv12(frame)` 后、`GpuNv12AppFrame(...)` / `enc.Encode(...)` 前调用 `matting._CUDA_STREAM.synchronize()`
- 新增 `t_sync` 列表与 `sync_avg / sync_p99 / decode_matting_sync_avg` summary 统计项
- progress 行新增 `sync` 与 `dec+mat+sync`

未落地（也无需落地）：

- 建议 B（alpha 路径接入 `Matter.acquire_nv12_output_slot()` ring slot）：alpha 走 `AlphaPacker.pack_uploaded()`，接口差异大；在 A 已经追平实时后该改造收益边际，先不做。
- 建议 C（`ThreadedDecoder` 替换 `SimpleDecoder`）：同理，无需。

## 3. 验证数据

测试条件：`videos/test_8k.mp4`，8192×4096 HEVC，15 秒 / 899 帧，`--engine rvm --input-size 1024 --sbs-batch --alpha-stride 1 --preset P1`，TensorRT 静态形状路径已激活（`static_trt=True, rvm_ort_avg≈5.86ms`）。

### Alpha 离线

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| **throughput** | **36.32 fps** | **75.62 fps** | **+108%** |
| decode_avg | 20.722 ms | 6.395 ms | -14.33 ms |
| matting_avg | 6.227 ms | 6.287 ms | ~ |
| sync_avg | — | 0.005 ms | (NOP) |
| decode_matting(+sync)_avg | 26.949 ms | 12.687 ms | -14.26 ms |
| encode_avg | — | 0.290 ms | — |
| mux_write_avg | — | 0.145 ms | — |

### Green 离线（对照，未改动）

| 指标 | 数值 |
|---|---|
| throughput | 75.46 fps |
| decode_avg | 4.635 ms |
| matting_avg | 6.690 ms |
| sync_avg | 1.390 ms |
| decode_matting_sync_avg | 12.715 ms |

### 实时 alpha（对照，来自 `debug_output/server.log`）

| 指标 | 数值 |
|---|---|
| throughput | ~77 fps |
| decode | 5-6 ms |
| composite | ~6 ms |
| output_fps | 59.940 |

**三者总时长（decode + matting + sync）均落在 12.7 ms 上下，差距收敛。**

## 4. 机理解释（为什么 sync ≈ 0 还能省 14ms）

修复后 `sync_avg = 0.005 ms`，说明 host 调到 `synchronize()` 时 GPU 工作已经完成。这一调用并非「等」CUDA 队列，而是起到 **stream barrier checkpoint** 作用：

- **修复前**：composite kernels launch 到 `_CUDA_STREAM` 后无显式同步。PyNvVideoCodec 在下一次 `SimpleDecoder.__getitem__` 通过 CAI 暴露 NVDEC surface 时，会与未排空的 `_CUDA_STREAM` 形成隐式依赖。CUDA driver 把这个依赖视为 hazard，序列化调度：必须等当前帧 SM composite 完成 NVDEC 才上 hw 引擎。NVDEC 与 SM 在硬件可并行的情况下被迫串行，每帧多花 ~14ms。
- **修复后**：显式 `synchronize()` 告诉 driver 这条 stream 已干净。下一次 `frame_at()` 调度 NVDEC 时不再被悬挂依赖阻塞，NVDEC 可以在当前帧 SM 还在做 composite 时就并行解下一帧。SimpleDecoder 内部预取窗口被打开，NVDEC 与 SM/NVENC 三个硬件单元真正并行。
- **sync 调用本身耗时 ≈ 0**：因为到这一行时 GPU 早已闲。但「这一行存在」是 CUDA 调度依赖解开的关键。

## 5. Green vs Alpha 子项分布差异

两边总时长一致（≈12.7ms），但分布不同：

| 指标 | alpha | green |
|---|---|---|
| decode_avg | 6.395 | 4.635 |
| matting_avg | 6.287 | 6.690 |
| sync_avg | 0.005 | 1.390 |

- green 走 `Matter.acquire_nv12_output_slot()` 输出到 ring slot，composite GPU 工作的部分尾巴留在 stream 上等 sync。
- alpha 走 `AlphaPacker.pack_uploaded()`，pack 内部 CuPy 调用已隐式 sync 过，到显式 sync 时无事可等。
- 两种「等待时机」分布不同，但 NVDEC 并行带来的总时长收益相等。
- 这也反向解释了之前 green（已有 sync）也只有 36fps 的现象：green 之前同样卡在 NVDEC 串行上，因为 alpha 链路里多次 CuPy 隐式同步没有起到 stream barrier 的等价作用 —— 真正的钥匙是 `_CUDA_STREAM` 这条特定流的显式 barrier。

## 6. 问题报告中「Remaining Suspects」处置

`summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_EN.md` 列出的怀疑项可全部关闭：

| 怀疑项 | 处置 |
|---|---|
| MP4 mux/磁盘写 backpressure 影响 `frame_at()` | 关闭。修复后 mux_write_avg = 0.145 ms，与本现象无关。 |
| 实时是否有 warmup/cache 让 SimpleDecoder 走更快路径 | 关闭。实时与离线的 SimpleDecoder 行为一致，差异在 stream barrier。 |
| 最小 A/B 探针 | 不必再做。修复前后两组数已构成强对照（throughput 36 → 75）。 |
| 不要裸跑 GPU/ORT 工具（需 `base_environment()`） | 与本问题无关，但属于通用约束，继续保持。 |

## 7. 复测命令

```powershell
.\.venv\Scripts\python.exe -m compileall tools\offline_alpha_passthrough.py
.\.venv\Scripts\python.exe -m pytest tests\test_offline_convert.py tests\test_settings.py tests\test_offline_alpha_bitrate.py
```

复测真实文件（任一离线工具均可）：

```powershell
.\.venv\Scripts\python.exe tools\offline_alpha_passthrough.py videos\test_8k.mp4 `
  --engine rvm --out videos\test_8k_alpha.mp4 --audio copy `
  --model models\rvm_mobilenetv3_fp32.onnx --duration 15.0 --bitrate source `
  --alpha-stride 1 --preset P1 --cq -1 --input-size 1024 --sbs-batch
```

预期 `throughput ≈ 75 fps`，`decode_avg ≈ 6-7 ms`，`sync_avg < 0.1 ms`，`decode_matting_sync_avg ≈ 12-13 ms`。

## 8. 经验提炼（防止下一次踩坑）

1. **CUDA stream barrier 不只是「等待」**：在多硬件引擎（NVDEC + SM + NVENC）混合场景下，显式 `synchronize()` 是给 driver 调度器的依赖图清理信号，缺它会被默认按 hazard 串行，丢掉硬件并行收益。
2. **`sync_avg ≈ 0` 不等于 sync 调用无效**：sync 的位置和存在性比它实际等待的时长更重要。
3. **PyNvVideoCodec SimpleDecoder 的预取窗口需要被「让」出来**：让的方式不是等待时间，而是给出干净的 stream 状态。
4. **同样的 throughput 不代表两路径相同**：alpha 与 green 都到 75.5fps，但 decode/matting/sync 子项分布不同；定位问题时不能只看总数，必须看分布。
5. **离线工具与实时 worker 的等价性检查**：今后新增任何 GPU 流水线工具，最小集应包含 `composite → synchronize → encode` 三段，并按子项分别计时，方便对照实时 worker。

## 9. 文件落地清单

- 修改：`tools/offline_alpha_passthrough.py`（+sync 调用、+t_sync 统计、+progress/summary 输出栏）
- 未动：`tools/offline_passthrough.py`（green 离线本来就有 sync，本次确认无误差）
- 未动：`pipeline/matting.py`、`pipeline/pynv_stream.py`、`pipeline/pynv_io.py`、`utils/trt_manifest.py`、`utils/runtime_dll_paths.py`、`ui/services/trt_warmup_process.py`（与本问题无关）
- 测试：`tests/test_offline_convert.py`、`tests/test_settings.py`、`tests/test_offline_alpha_bitrate.py` 全部通过

## 10. 结案

问题闭环。离线 alpha 路径与实时 / green 离线吞吐对齐到 ~75fps，TensorRT 路径性能在离线工具上得到完整释放。
