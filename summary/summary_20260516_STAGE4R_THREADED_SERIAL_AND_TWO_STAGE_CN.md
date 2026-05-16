> ⚠️ SUPERSEDED 2026-05-16
> 本报告中的 FPS、sync、瓶颈归因结论建立在旧默认值 PT_PASSTHROUGH_MAX_FPS=30 和/或非目标 PT_ALPHA_STRIDE=3 条件下，已被后续复核证伪或降级为仅适用于旧诊断条件。
> 重新基线与有效结论入口见 summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_CN.md。
> 仅保留为研究过程档案；与 cap 无关的实现结论仍需按正文中的适用范围判断。
# 阶段 4R 小结 - ThreadedDecoder 串行接入与两段式验证

## 背景

本阶段是在专家建议 `summary/summary_20260516_STAGE4_PYNV_THREADED_STAGED_CRASH_ADVICE_EN.md` 基础上继续推进。原三段式 ThreadedDecoder staged pipeline 已确认存在生命周期违约风险：ThreadedDecoder 返回帧不能跨线程排队后再消费，否则 decode worker 的下一次 `get_batch_frames()` 可能让旧 batch 的底层 GPU 指针失效。

本阶段目标改为低风险路线：先验证 ThreadedDecoder 顺序拉帧，再接入生产串行 worker；如果不足，再尝试只跨线程传递 Matter-owned NV12 slot 的两段式。

## 已完成工作

1. 运行 ThreadedDecoder decode-only 探针：

```powershell
uv run python tools\pynv_threaded_decode_probe.py videos\test_8k_2.mp4 --frames 300 --fps 30 --batch-size 8 --buffer-size 32 --hash-frames 20
```

结果：

- `threaded.selected_fps=86.24`；
- `source_fetch_fps=171.91`；
- `hash checked=20 matched=20 ok=True`；
- 报告：`baseline/pynv_threaded_decode_phase2_20260516_150052.md/json`。

结论：专家建议中的 Path A 通过门槛，可以进入生产串行接入。

2. 新增 `PyNvThreadedSerialDecoder`：

- 文件：`pipeline/pynv_io.py`；
- 只支持 monotonic source index；
- 使用 `ThreadedDecoder(start_frame=initial_src_idx)`；
- 内部按 batch 拉帧并跳过未选 source frame；
- 不跨线程保存或传递 ThreadedDecoder frame；
- 调用者必须在下一次 `frame_at()` 前消费完返回 frame。

3. 生产 live worker 接入可切换 decoder：

- `PT_PASSTHROUGH_PYNV_DECODER=threaded_serial` 默认启用；
- `PT_PASSTHROUGH_PYNV_DECODER=simple` 可立即回退旧路径；
- 文件：`config.py`、`pipeline/pynv_stream.py`。

4. 修复自动化 harness 的开发环境 CUDA DLL 路径：

- 文件：`tools/auto_tune_8k.py`；
- 复用 `ui.services.process_helpers.base_environment()`；
- 默认补 `PT_CUDNN_BIN=C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64`；
- 修复前 auto_tune 子进程会退到 `CPUExecutionProvider`，修复后恢复 `CUDAExecutionProvider`。

5. 实现 Path D：Matter NV12 slot 可等待 acquire：

- 文件：`pipeline/matting.py`；
- `acquire_nv12_output_slot(h, w, timeout=None)` 默认保持旧行为：无空位立即抛错；
- 传入 timeout 时可条件变量等待；
- `release_nv12_output_slot()` 会 notify；
- `reset_state()` 会释放所有 slot 占用并 notify。

6. 实现实验性 Path B：green-only two-stage worker：

- 配置：`PT_PASSTHROUGH_PYNV_WORKER_MODE=two_stage`；
- 默认仍是 `serial`；
- 只覆盖 green 路径，alpha 仍自动回到 serial；
- matting worker 内部完成 `ThreadedDecoder frame -> composite -> sync -> Matter NV12 slot`；
- encode/mux worker 只接收 `GpuNv12AppFrame + Matter slot`，不接收 PyNv decoded frame；
- 使用 `PT_PASSTHROUGH_NV12_RING_SLOTS=5` 测试。

## 测试结果

### Decode-only 门槛

- `baseline/pynv_threaded_decode_phase2_20260516_150052.md`
- 通过：`selected_fps=86.24`，hash 全匹配。

### Threaded serial green 10 秒

命令：

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 10 --startup-timeout 240 --client-timeout 120
```

报告：`baseline/auto_tune_8k_phase1_20260516_150735.md`

结果：

- latest interval FPS：`36.43`；
- average interval FPS：`36.02`；
- decode：`0.07 ms`；
- composite：`15.08 ms`；
- sync：`11.43 ms`；
- slow mux warnings：`0`。

结论：ThreadedDecoder 串行接入安全，decode 成本从旧路径约 `18 ms` 降到约 `0.07 ms`，但总 FPS 未突破 40。

### 同环境 SimpleDecoder 对照

报告：`baseline/auto_tune_8k_phase1_20260516_150912.md`

结果：

- latest interval FPS：`36.55`；
- average interval FPS：`35.94`；
- decode：`18.26 ms`；
- composite：`6.90 ms`；
- sync：`1.29 ms`。

结论：两者总 FPS 接近，但耗时归因不同。Threaded serial 消除了 random decode 成本，但由于现有单 worker 和 CUDA 同步结构，等待被转移到 composite/sync，未形成端到端吞吐提升。

### Alpha smoke

报告：`baseline/auto_tune_8k_phase1_20260516_150832.md`

结果：

- latest interval FPS：`34.99`；
- decode：`0.06 ms`；
- composite：`27.59 ms`；
- 无 worker exception。

结论：alpha 路径未被 Threaded serial 接入破坏，但性能瓶颈仍在 alpha pack/composite。

### Two-stage green 10 秒

报告：`baseline/auto_tune_8k_phase1_20260516_151600.md`

结果：

- latest interval FPS：`36.32`；
- average interval FPS：`35.95`；
- decode：`0.13 ms`；
- composite：`18.31 ms`；
- sync：`9.03 ms`；
- encode：`0.75 ms`；
- slow mux warnings：`0`；
- 无 slot timeout、无 worker exception。

结论：两段式 green 路径安全性初步通过，但没有性能收益，仍在 `36 fps` 左右。

## 关键结论

1. 原三段式 ThreadedDecoder staged 不能恢复，除非先做 owned decoded GPU ring lifetime probe。
2. ThreadedDecoder 串行接入是安全的，且帧映射稳定、hash 通过。
3. 当前 40fps 阻塞点不再是 decode，而是 composite/sync/RVM CUDA 执行结构。
4. 简单拆成两线程不能突破 40fps，因为关键 GPU 工作仍在同一 CUDA stream 和同步点上串行。
5. Path B 的基础安全设施已经具备：Matter NV12 slot 可等待，跨线程只传 owned slot，不传 PyNv frame。

## 风险和阻碍

- `PyNvThreadedSerialDecoder` 只支持 monotonic source index；如果未来 live seek 在同一个 decoder 实例内倒退，需要重建 decoder。
- `ThreadedDecoder` frame 生命周期仍是硬约束：不能跨线程传递 decoded frame，不能在消费前拉下一批。
- two-stage 当前只是实验路径，不应设为默认。
- alpha 不适合直接套 green two-stage，因为 alpha packer 的输出生命周期和 slot 所有权不同。
- 40fps 的下一突破点需要更底层的 CUDA stream/event 或减少 sync，而不是继续增加 Python 线程。

## 建议下一步

建议下一阶段不要继续扩展 staged worker，而是转向以下方向之一：

1. 分析 `pipeline/matting.py` 中 `_CUDA_STREAM.synchronize()` 的必要性，研究能否用 CUDA event 将 composite 完成信号交给 encode，而不是每帧全 stream sync。
2. 把 RVM preprocess/ORT/composite 的等待拆开测量到 CUDA event 级别，确认 `composite` 和 `sync` 实际分别包含什么。
3. 若事件交接不可行，进入 CUDA Graph 或 TensorRT EP 优化 RVM/alpha 路径。
4. two-stage 继续保留为实验配置，不作为默认生产路径。

## 验证命令

```powershell
python -m py_compile config.py pipeline\matting.py pipeline\pynv_io.py pipeline\pynv_stream.py tools\auto_tune_8k.py
python -m unittest tests.test_content_directory_modes tests.test_subtitles
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 10 --startup-timeout 240 --client-timeout 120
```