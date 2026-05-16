> ⚠️ SUPERSEDED 2026-05-16
> 本报告中的 FPS、sync、瓶颈归因结论建立在旧默认值 PT_PASSTHROUGH_MAX_FPS=30 和/或非目标 PT_ALPHA_STRIDE=3 条件下，已被后续复核证伪或降级为仅适用于旧诊断条件。
> 重新基线与有效结论入口见 summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_CN.md。
> 仅保留为研究过程档案；与 cap 无关的实现结论仍需按正文中的适用范围判断。
# 阶段 4R 小结 - 下游串行屏障排查

## 背景

外部反馈认为 FP16 节省的 ORT/RVM 时间被 `sync` 吸收，怀疑存在约 27ms 的结构性串行屏障。反馈中提出的优先验证项包括：

1. 确认是否仍在使用 H264 NVENC。
2. 调大 NV12 output slot 池。
3. 短路 RVM，隔离 decode -> composite -> encode/mux 下游瓶颈。

本阶段按这些低成本验证项执行。

## 测试范围更正

本文件记录的 slot=8 与 RVM bypass 隔离测试，均是在默认 `PT_ALPHA_STRIDE=3` 或 bypass 条件下完成的。它们只能说明 stride=3 诊断路径中存在固定等待/吞吐上限，不能代表最初目标 `PT_ALPHA_STRIDE=1` 的性能瓶颈。

后续补测 `PT_ALPHA_STRIDE=1` 后，正确目标基线为：

- `baseline/auto_tune_8k_phase1_20260516_170451.md`
- average interval FPS: `35.03`
- ORT/RVM: `24.60 ms`
- sync: `1.84 ms`

这说明 stride=1 下瓶颈仍主要在 ORT/RVM/composite，而不是本文件 stride=3 诊断中表现出的 sync 屏障。

## 关键前提纠正：当前生产路径已经是 HEVC

代码确认：

- `pipeline/pynv_stream.py` 中 `PYNV_OUTPUT_CODEC = "hevc"`。
- `PYNV_BACKEND_LABEL = "pynv_hevc"`。
- live cache key 日志为 `/hevc/30.000/...`。

因此反馈中的“从 h264_nvenc 改为 hevc_nvenc”测试，在当前 PyNv 生产路径上已经默认成立。当前约 37fps 上限不能归因于仍在使用 H264 NVENC。

注意：

- `config.PASSTHROUGH_VCODEC=hevc_nvenc` 是 FFmpeg fallback 路径配置。
- PyNv 生产 live path 不使用该值，而是固定 `PYNV_OUTPUT_CODEC="hevc"`。

## 测试 1：NV12 slot 池从 3 调到 8

命令：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 180 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx --server-env PT_PASSTHROUGH_NV12_RING_SLOTS=8
```

报告：

- `baseline/auto_tune_8k_phase1_20260516_164945.md`
- `baseline/auto_tune_8k_phase1_20260516_164945.json`

对比：

| 配置 | average FPS | latest FPS | composite | sync | ORT/RVM |
|---|---:|---:|---:|---:|---:|
| FP16 slot=3 | 36.89 | 36.94 | 14.25ms | 12.34ms | 13.68ms |
| FP16 slot=8 | 36.87 | 36.97 | 15.36ms | 11.08ms | 14.55ms |

结论：

- slot 从 3 增加到 8 没有提升端到端 FPS。
- `composite` 与 `sync` 只是计时桶有波动，平均吞吐不变。
- “NV12 output slot 池太浅”不是当前 37fps 上限的主因。

## 测试 2：RVM bypass，下游瓶颈隔离

新增诊断开关：

- `PT_PASSTHROUGH_RVM_BYPASS_ALPHA=1`

实现位置：

- `config.py`
- `pipeline/matting.py`

行为：

- 仅用于诊断。
- 在 PyNv/CuPy green path 中跳过 RVM 推理。
- 使用全 1 alpha mask，即全前景。
- 保留 decode、NV12 composite kernel、sync、HEVC encode、mux。
- 默认关闭，不影响生产视觉路径。

命令：

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 180 --server-env PT_PASSTHROUGH_RVM_BYPASS_ALPHA=1
```

报告：

- `baseline/auto_tune_8k_phase1_20260516_165209.md`
- `baseline/auto_tune_8k_phase1_20260516_165209.json`

结果：

- latest interval FPS: `37.11`
- average interval FPS: `37.02`
- stage avg:
  - decode: `7.02 ms`
  - composite: `0.69 ms`
  - sync: `18.42 ms`
  - encode: `0.49 ms`
  - mux: `0.29 ms`
- mat avg:
  - preprocess: `0.00 ms`
  - ORT/RVM: `0.00 ms`
  - kernel: `0.63 ms`

关键日志：

```text
[DIAG] alpha #1800 bypass: frame=8192x4096 alpha_shape=(1024, 2048) use_nv12=True
```

结论：

- RVM 完全短路后，FPS 仍然只有约 `37.02`。
- 这证明当前 37fps 上限不是 RVM/ORT 算力瓶颈。
- 等待从 `composite` 转移到 `decode/sync`，说明瓶颈是下游 GPU 同步/解码帧可见性/编码尾部屏障中的某一项。

## 综合结论

目前已排除：

- H264 编码假设：当前生产路径已经是 HEVC。
- slot 池太浅：slot=8 与 slot=3 吞吐等价。
- RVM/ORT 主瓶颈：RVM bypass 后仍然约 37fps。
- custom composite kernel：kernel 约 0.4-0.7ms，不是主因。

stride=3 诊断下的判断：

- 存在一个约 27ms 的结构性 GPU/编码/解码同步屏障。
- 在 stride=3 或 RVM bypass 诊断中，单纯优化 ORT/RVM 会被 `sync` 或 decode wait 吸收。
- 但该结论不能直接外推到 stride=1；stride=1 补测显示 ORT/RVM 仍是主要瓶颈。
- 下一步如果继续查 stride=3 固定屏障，必须抓服务进程 Nsight timeline，重点看 NVDEC、NVENC、CuPy stream、ORT stream 和 host API wait。

## 下一步建议

1. 直接 profile 服务进程，而不是 auto_tune 父进程。
2. Nsight 必看：
   - NVENC encode submit 到 bitstream 可用之间的延迟；
   - ThreadedDecoder/NVDEC 内部 GPU 工作是否和 ORT/CuPy 争抢；
   - CuPy stream、ORT stream、PyNv stream 是否串行化；
   - `cuda_stream.synchronize()` 前累积了哪些 kernel/memcpy。
3. 对 stride=3 暂缓 CUDA Graph；但对 stride=1，CUDA Graph / ORT 优化仍可能有意义，因为 stride=1 补测中 ORT/RVM 为 `24.60 ms`。
4. 保留 `PT_PASSTHROUGH_RVM_BYPASS_ALPHA` 作为诊断开关，但不得作为生产模式。

## 验证

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\matting.py pipeline\pynv_stream.py tools\auto_tune_8k.py
```
