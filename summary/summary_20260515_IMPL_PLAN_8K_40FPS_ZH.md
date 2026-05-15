# 8K 实时透视 40fps 实施方案 — 2026-05-15

## 目标

把生产环境的实时透视从"8K + `ALPHA_STRIDE=3` 勉强 30fps"提升到 **8K + `ALPHA_STRIDE=1`（不跳 alpha）稳定 40fps**，且不要求用户升级 GPU。

本文档是以下两份资料的工程化落地：
- `prompt/HANDOVER_20260515.md`（可行性研究 + reviewer 反馈）
- `baseline/baseline_20260508_pynv_8k.txt`（目前唯一一份 8K 全链路实测）

方案分为 1 个自动化前置阶段和 6 个实施阶段，实施阶段按 reviewer 给出的优先级排序。每个实施阶段都有明确的验收门槛；**前一阶段不通过禁止开始下一阶段**。

---

## Phase 0 — 自动化测试前置准备

在开始任何性能改造前，先消除对人工真实 DLNA 客户端播放的依赖。本方案中的验收门槛不再要求用户手动打开播放器、选择 DLNA 条目并观察播放；真实设备测试只作为自动化门槛通过后的兼容性 soak，不阻塞阶段推进。

### 复用现有测试代码
- `tools/pynv_fullchain_probe.py`：离线全链路 probe 基础，用于统计 decode → matting → encode → mux。
- `tools/pynv_decode_probe.py`：纯解码基线，也是 SimpleDecoder 的对照点。
- `tools/bench.py`：辅助做本地 decode / pipeline / playback 快速检查。它可以做 smoke test，但不能作为 8K 自动化主验收，因为它不覆盖 DLNA Browse 和当前 live 条目选择路径。
- `tools/dlna_client_probe.py`：自动化 DLNA SOAP Browse + HTTP 拉流模拟器。默认应优先选择 alpha-live，passthrough-live 作为显式可选项。

### Phase 1 前必须完成的测试框架改造
1. 新增顶层编排脚本，建议命名为 `tools/auto_tune_8k.py`，支持按阶段运行并输出机器可读结果。
2. 编排脚本负责用 `PT_DEBUG_LOGS=1` 子进程启动 PTMediaServer，等待 `/description.xml` 和 `/control/cds` 可访问，测试结束后停止服务。
3. 编排脚本调用 `tools/dlna_client_probe.py` 浏览媒体库、定位指定视频、优先选择 alpha-live 条目、GET DIDL 返回的 `<res>` URL，并按固定时长读取流。
4. 增加 `debug_output/server.log` 解析器，提取 `[PYNV][sid] ... interval_fps=... stage_avg_ms decode=... composite=... sync=... encode=... mux=...`、`mux stdin write slow`、HTTP pacing 警告、实际模式和 stream session id。
5. 扩展 `tools/pynv_fullchain_probe.py`，增加 `--pipeline=serial|staged` 和 JSON 输出，让 Phase 4 可以先离线验收，再进入 live 路径。
6. 新增或排期两个缺失的独立 probe：`tools/pynv_threaded_decode_probe.py` 用于验证 ThreadedDecoder，`tools/trt_rvm_probe.py` 用于验证 TensorRT EP。
7. 所有输出写入 `baseline/`，例如：
   - `baseline/auto_tune_8k_phase1_YYYYMMDD_HHMMSS.json`
   - `baseline/auto_tune_8k_phase1_YYYYMMDD_HHMMSS.md`
   - 截取后的 server log 和 client probe JSON。

### Phase 0 验收门槛
下面一条命令可以在没有用户交互的情况下完成 Phase 1 测量：

```powershell
uv run python tools/auto_tune_8k.py phase1 --video <8k-file> --profile quest --prefer alpha --duration 60
```

该命令必须自动启动服务、完成 DLNA Browse 和 live HTTP 拉流、解析生产端/客户端指标、停止服务，并写出 baseline 报告。如果这个门槛尚未完成，不应开始 Phase 1 测量，因为那仍然会依赖人工测试。

---

## 背景：8K 单帧时间预算（已实测）

来自 `baseline_20260508_pynv_8k.txt` 稳态数据，RTX 级 GPU，`MATTING_INPUT_SIZE=512`，SBS batch=2，`RVM_DOWNSAMPLE_RATIO=0.5`：

| 阶段 | stride=3 | stride=1 | 备注 |
|---|---:|---:|---|
| `frame_at(src_idx)` 解码 | 18.8 ms | 5.0 ms | stride=3 时 `SimpleDecoder[index]` 随机访问导致虚高 |
| matting (preprocess+ORT 均摊) | 5.8 ms | 20.7 ms | 单次 ORT 实际 ~17 ms，与 stride 无关 |
| composite | 0.16 ms | 0.16 ms | CuPy RawKernel，GPU |
| encode (NVENC HEVC) | 0.33 ms | 0.40 ms | |
| mux_write (FFmpeg stdin) | 1.92 ms | 3.61 ms | |
| 串行总和 | ~27 ms | ~30 ms | 与 probe 35.9 / 31.9 fps 吻合 |

生产目标：stride=1 下每帧 ≤ 25 ms → 40 fps。

---

## Phase 1 — 排查 HTTP 实时路径与 probe 的差距（不改代码）

### 为什么先做
Probe 是 35.9 fps，但用户反映生产只有 ~30 fps。在动流水线之前，先确认瓶颈到底是生产端还是 HTTP 投递端。

### 步骤
1. 运行 `tools/auto_tune_8k.py phase1 --video <8k-file> --profile quest --prefer alpha --duration 60`。
2. 编排脚本用 `PT_DEBUG_LOGS=1` 启动服务，等待 DLNA 就绪，运行 `tools/dlna_client_probe.py`，并把 client JSON 与 server log 摘要保存到 `baseline/`。
3. 从 `debug_output/server.log` 自动提取周期日志：`[PYNV][sid] frame N/M ... interval_fps=... stage_avg_ms decode=... composite=... sync=... encode=... mux=...`（按 `_DIAG_INTERVAL` 输出）。
4. 看三个信号：
   - `interval_fps`：生产端真实吞吐。
   - `mux_write` 均值，以及是否触发 `mux stdin write slow` 告警（`pipeline/pynv_stream.py:1779-1787`）。
   - 与 `dlna_client_probe.py` 记录的首字节时间、读取字节数、平均码率、stall/timeout 和 HTTP 状态对比。

### 决策矩阵
- **interval_fps ≈ 35 且客户端 ≈ 26** → HTTP 发送端节流或队列是瓶颈。处置：
  - `PT_PASSTHROUGH_SEND_PACING_MULTIPLIER` 从 `2.0` 升到 `3.0`–`4.0`。
  - 临时设 `PT_PASSTHROUGH_SEND_REALTIME_PACING=0` 复测。
  - 检查 `_audio_cache` 锁的获取延迟。
- **interval_fps ≈ 26** 且 mux_write 健康 → 真的是生产端瓶颈，进入 Phase 2+。
- **mux_write 出现 > 100 ms 尖峰** → FFmpeg mux 子进程被反压；先查下游（TS muxer / slate / audio 路径）再动 encode。

### 验收门槛
把"丢失的 5–10 fps 归因于（HTTP pacing / mux 反压 / 生产端封顶）"写成一段明确的结论，落到 `baseline/` 下一份自动生成的 probe 笔记。该报告必须由自动化框架生成，不再依赖人工播放器观察。

### 风险
无 —— 本阶段只读。

---

## Phase 2 — `ThreadedDecoder` 顺序拉流 probe（独立验证）

### 为什么
`pipeline/pynv_io.py:201` 的 `PyNvSimpleDecoder.frame_at` 调的是 `SimpleDecoder[index]`。NVIDIA 文档明确 `ThreadedDecoder` 是给推理 / 高吞吐顺序场景设计的，内置后台预取。8K 源 60fps → 输出 30fps 正好是 1:2 顺序抽帧，是 ThreadedDecoder 的甜蜜场景；`SimpleDecoder` 随机访问的 ~12 ms 额外开销应当消失。

### 步骤
1. 新建 `tools/pynv_threaded_decode_probe.py`：
   - `nvc.ThreadedDecoder(filename, gpu_id=..., use_device_memory=True)`（按已安装的 PyNvVideoCodec 2.1.0 实际 API 名称）。
   - 必要时 `seek_to_frame(start_idx)`，然后循环 `get_next_frame()`（或同等接口），按现 `pynv_stream.py:1704` 的 `cfr_source_index` 规则丢掉中间帧。
   - 报告每帧 `t_decode` 和 300 帧稳态 fps。
2. 与 `tools/pynv_decode_probe.py`（当前 SimpleDecoder，141fps 纯解码）对比。
3. 校验前 10 帧 NV12 plane 与 SimpleDecoder 输出 SHA256 一致。

### 决策矩阵
- ThreadedDecoder 稳态每输出帧解码 ≤ 8 ms：**进入 Phase 3**。
- PyNv 2.1.0 没暴露 ThreadedDecoder，或与 SimpleDecoder 持平/更慢：保留 SimpleDecoder，**Phase 3 仍然继续**——ring buffer + 去除同步本身也能拿一部分收益。
- 帧不一致：停下来排查；线上路径依赖 CFR 映射的确定性。

### 验收门槛
独立 probe 脚本能复现 ≤ 8 ms 解码时间，并写出与 `cfr_source_index` 一致的 CFR 跳帧规则。**本阶段不动线上路径。**

### 风险
PyNv 2.1.0 的 `ThreadedDecoder` 接口可能与公开文档不符。如果构造签名、取帧方法、seek 行为不能配合"先 seek 再顺序播放"的现状，probe 必须明确写出来；那种情况 Phase 4 仍然适用，但 decode 阶段保留 SimpleDecoder。

---

## Phase 3 — GPU NV12 ring buffer（2–3 槽位）

### 为什么
`pipeline/matting.py:_ensure_dev_nv12_out` 现在返回单例 `_g_out`。下一帧的 composite 直接覆盖上一帧 NVENC 还在读的内存，所以线上循环必须在 `pipeline/pynv_stream.py:1751` 硬同步 `cuda_stream.synchronize()`。没有按槽位归属，就没法安全重叠。

### 具体改动落点
- `pipeline/matting.py`：
  - `Matter._ensure_dev_nv12_out(h, w)` → 改成 `Matter._acquire_nv12_slot(h, w)`，从固定大小池子（`config.PASSTHROUGH_NV12_RING_SLOTS` 默认 `3`）返回索引。
  - 新增 `Matter._release_nv12_slot(idx)` 及槽位状态掩码（`free` / `compositing` / `encoding`）。
- `pipeline/pynv_stream.py`：
  - `out_nv12, _ = self.matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(frame)` 调用前先 acquire 一个空闲槽位。
  - 跟踪当前 NVENC 占用的槽位；`self._enc.Encode(app_frame, flags)` 返回比特流后释放（或在下一次 `Encode` 返回时释放，看 PyNv 重入语义）。

### 同步策略（按安全程度排序）
1. **延迟槽位复用（不依赖 event API）**：N=3 时，等到生产端再次想要槽位 0，前一次的 `Encode` 早已返回并把输入读完。前提是 `self._enc.Encode()` 同步等待 NVENC 把输入完整读入内部队列（PyNv 公开文档的标准行为）。
2. **CUDA event 传递（仅当 PyNv 暴露此接口）**：composite 之后 record event，`Encode(input, wait_event=...)` 等待。**Phase 2 probe 必须确认已安装 wheel 中的 2.1.0 是否有这个签名**，再决定是否走这条路。

### 验收门槛
- 生产端不再每帧调 `cp.cuda.get_current_stream().synchronize()` 来做 NV12 槽位交接。
- `videos/test_8k.mp4` 与当前构建做 A/B 视觉对比：前 200 帧无任何瑕疵（NV12 复用错乱会表现为横向撕裂或色块）。
- `interval_fps` 至少多出原 sync 时间对应的帧率（诊断行里 `sum_sync` 应当降到 ~0）。

### 风险
- 如果 PyNv `Encode()` 是异步的（输入还没读完就返回），延迟复用就不安全。**Phase 2 probe 必须测**：调 `Encode()` 后立即把输入 buffer 写垃圾，再编码 N 帧，看输出有无损坏。如不安全，退化方案是每"轮转一圈"做一次 `cuda_stream.synchronize()`，而不是每帧。

---

## Phase 4 — 三段流水线（decode / ORT+composite / encode+mux）

### 为什么
Phase 2、3 完成后，三种硬件引擎（NVDEC、CUDA SM/Tensor、NVENC）才能真正并发。理论单帧墙时变成 `max(decode, ORT+composite, encode+mux)`。按 stride=1 的基线数：
- decode（ThreadedDecoder）：~7 ms
- ORT + composite：~20 ms
- encode + mux：~4 ms
- max ≈ 20 ms → ~50 fps（保守估计，留出竞争损耗）

### 架构
- 每个阶段一个 Python 线程，通过 `queue.Queue(maxsize=ring_slots)` 通信。
- Stage A（decode）：从 `ThreadedDecoder`（或 SimpleDecoder fallback）拉帧，push `(slot_idx, GpuNv12Frame)` token。
- Stage B（matting）：消费 A 的 token，跑 ORT IOBinding + composite 写入槽位 NV12，push `(slot_idx, app_frame)` token。
- Stage C（encode+mux）：消费 B 的 token，调 `self._enc.Encode(app_frame)`，写比特流到 mux stdin，释放槽位。
- 现 `_worker_loop`（`pipeline/pynv_stream.py:1489`）改为协调者：启动/汇合三个线程，传播 `_stop`，聚合诊断。

### 跨阶段 GPU 同步
- Stage A → Stage B：NVDEC 写入设备内存；CuPy 导入是零拷贝（`PyNvSimpleDecoder.frame_at` 已是这样）。NVDEC 的 stream 在 PyNv 内部，matting 用的是 `_CUDA_STREAM`。每帧 token 上：在 PyNv 的 stream 上 `cudaEventRecord`，在 `_CUDA_STREAM` 上 `cudaStreamWaitEvent`。如果 PyNv 不暴露其 decode stream，退化成生产线程入队前 `cp.cuda.runtime.deviceSynchronize()` —— 粗粒度但安全。
- Stage B → Stage C：composite 完成后在 `_CUDA_STREAM` 上 record per-slot event；Stage C 在 `Encode()` 前 wait。或者，依赖 Phase 3 的延迟复用，省掉 event。

### 具体改动落点
- 只需要拆 `pipeline/pynv_stream.py:_worker_loop`。其余（preflight、mux 启动、AAC cache、slate、字幕）保持不变。
- 新增 `pipeline/pynv_pipeline.py` 装三个线程体和共享 `Slot` 数据类，避免 `pynv_stream.py` 继续膨胀。

### 验收门槛
- 新增 8K 全链路 probe（`tools/pynv_fullchain_probe.py --pipeline=staged`）`ALPHA_STRIDE=1` 稳态 ≥ 40 fps。
- 自动化 live HTTP probe（`tools/auto_tune_8k.py phase4 --profile quest --prefer alpha --duration 60`）显示 `interval_fps ≥ 38` 持续 ≥ 60 s，且无 `mux stdin write slow` 告警；客户端拉流侧持续收到数据/码率稳定，没有 timeout 或提前断开。
- 字幕叠加路径（`_apply_subtitle_overlay`）仍正常；alpha-pack 路径仍正常。

### 风险
- **`Matter` 内部并发**：`_g_alpha`、`_g_chw`、`_rvm_io_outputs`、`_rvm_rec_ort`、`_cached_alpha_small` 全是单例状态。Stage B 是唯一调用方，所以内部仍然单线程；但要确保 Stage A 不在任何 preflight 副作用里碰 matter 单例。
- **顺序保持**：队列必须保序；Stage B 不要并行池化，除非每个 slot 带序号且 Stage C 重排。v1 保持 Stage B 单线程。

---

## Phase 5 — RVM 上 ORT CUDA Graph

### 为什么
此时 ORT（~17 ms）变成新瓶颈。ORT 在 IOBinding + 输入输出地址固定时支持 CUDA Graph 捕获（`enable_cuda_graph` provider option）。预期降低 20–35% kernel-launch 开销 → ORT 降到 ~12 ms。

### 前提：递归状态地址必须持久
`pipeline/matting.py:1416` 的 `self._rvm_rec_ort = rec_outs[:4]` **替换了 OrtValue 对象**。CUDA Graph 在捕获时把指针写死，replay 时换地址会爆。需要的改动：

- 按分辨率一次性分配 4 个持久 `OrtValue` 给 `r1i..r4i`，再分配 4 个给 `r1o..r4o`，shape 用 `_rvm_output_shape_for` 算。
- 每次 `run_with_iobinding` 后，把 `r*o` 的设备 buffer 原地拷回 `r*i`（`cudaMemcpyAsync` on `_CUDA_STREAM`，或暴露一个 OrtValue 的 device-to-device copy helper）。
- 每次都通过 `bind_ortvalue_input` / `bind_ortvalue_output` 绑定到**同一组** OrtValue 对象。

### 启用
地址稳定之后，在 `pipeline/matting.py:828` 的 `_provider_config` 加 `"enable_cuda_graph": "1"`。

### 校验
- 每个会话第一次推理：常规 CUDA EP 跑（图捕获）。
- 后续推理：图 replay；`nvidia-smi dmon` 看到 kernel launch 大幅下降。
- 同一 shape 100 帧的 alpha 输出与非 graph 构建逐位一致。

### 验收门槛
- 同硬件 ORT 均值从 ~17 ms 降到 ≤ 13 ms。
- 无精度回归（合成输出与非 graph 参考 PSNR ≥ 50 dB）。

### 风险
- 任何输入 shape（含 SBS active 与否、batch=1 与 2）变化都会让 graph 失效。**强制每个会话固定 shape**；若 SBS 切换就 fallback 到非 graph 模式并打日志。
- 不同 ORT 版本对递归图的 CUDA Graph 稳定性差异较大；当前 `1.19.2` —— 用 5 分钟 soak 验证。

---

## Phase 6 — TensorRT EP probe（可选，方差最大）

### 为什么
TensorRT EP 通常比 CUDA EP fp32 再快 1.5–2.5×，特别是 Turing（RTX 2080）的 Tensor Core fp16 利用得很好。reviewer 已确认本机 `onnxruntime==1.19.2` 包含 `TensorrtExecutionProvider`。

### 步骤
1. 独立 probe `tools/trt_rvm_probe.py`：
   - 用 `providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": "1", "trt_engine_cache_enable": "1", "trt_engine_cache_path": "runtime_cache/trt_engines"})]` 加载同一份 RVM ONNX。
   - 喂入生产实际形状（8K SBS 半帧 4096×4096 下采样到 512×512，batch=2，加 rec 输入）。
   - 1 次 warmup 后 100 次推理计时。
2. 与现有 baseline 的 CUDA EP IOBinding 数据对比。
3. 输出与 fp32 CUDA EP 逐位比对（或 PSNR ≥ 50 dB）。

### 决策矩阵
- TRT 推理 ≤ 8 ms 且无质量回归：单独排一个集成阶段。
- TRT 推理 8–13 ms：仍值得集成，余量更大。
- TRT engine 在 RVM 递归输入上构建失败 / 推理质量明显下降：弃用 TRT，把 Phase 5 的 CUDA Graph 当作 ORT 侧天花板。

### 验收门槛
`baseline/` 下一份独立 probe 报告，含：build 时长、engine cache 大小、稳态 fps、输出 PSNR、FP16 动态范围问题（如有）。

### 风险
- RVM 递归循环时常让 TRT shape inference 困惑；可能要显式钉死所有 `r*i` 输入 shape。
- 第一次 engine build 需要 1–3 分钟；必须藏到启动 warmup（`utils/gpu_runtime_cache.py`）里。

---

## 新增配置开关

加到 `config.py`，支持 `PT_*` 环境变量覆盖（默认值如下）：

| 变量 | 默认 | 用途 |
|---|---:|---|
| `PT_PASSTHROUGH_NV12_RING_SLOTS` | `3` | Phase 3 NV12 输出槽位数 |
| `PT_PASSTHROUGH_PIPELINE_MODE` | `staged` | `serial`（当前）或 `staged`（Phase 4） |
| `PT_RVM_CUDA_GRAPH` | `0` | Phase 5 稳定后开启 |
| `PT_RVM_TENSORRT_EP` | `0` | Phase 6 PoC 通过后开启 |

默认迁移计划：Phase 3 + 4 通过验收门槛后，将 `PT_PASSTHROUGH_PIPELINE_MODE=staged` 设为默认；Phase 5、6 持续放在 flag 后面，等 soak 测试通过再开默认。

---

## 累计预期收益（保守版）

| 完成阶段 | ORT 路径 | 预期 8K stride=1 稳态 fps |
|---|---|---:|
| 0（当前） | CUDA EP IOBinding，串行 | 31.9 |
| 1（HTTP 修） | 同上 | 32–35（仅 HTTP 侧）|
| 3（ring buffer） | 同上，去同步 | 35–38 |
| 4（三段流水线） | 同上 | 40–48 |
| 5（CUDA Graph） | CUDA Graph | 45–55 |
| 6（TRT EP） | TRT fp16 | 55–70+ |

**40 fps 目标在 Phase 4 后达成。**Phase 5、6 是给后续新功能买余量。

---

## 仍可能让计划失败的因素

1. PyNv 2.1.0 `Encode()` 输入读和输出发都是同步的，且没有 event 接口暴露。Phase 4 的 encode 阶段就不能与 NVENC 硅工作并行——但 NVENC 这里本来 < 1 ms，损失很小。
2. 用户 GPU 的 NVDEC 是 gen-1（Maxwell）。8K HEVC 解码本身就跑不到 30 fps，再多流水线也救不回来。Phase 2 probe 检测到这种情况；这类用户唯一现实选项是边解码边降分辨率，或走预算 alpha 离线路径（已存在）。
3. ORT 1.19.2 CUDA Graph 在这个 RVM 递归拓扑上行为异常。退到普通 IOBinding，接受 Phase 4 的数。
4. 现存线上路径的 audio cache（`_lock_for_audio_cache`，`pipeline/pynv_stream.py:125`）锁持有时间过长，导致 Phase 4 的三个线程仍然在锁上串行化。Phase 4 实施时审一遍锁的 scope。

---

## 建议的开发节奏

1. **Day 0** — Phase 0 自动化框架：串起 `auto_tune_8k.py`、服务子进程生命周期、`dlna_client_probe.py`、日志解析和 baseline 报告生成。
2. **Day 1** — Phase 1 自动化诊断：运行框架，根据结果尝试一个可能的 env 调整，把生成的结论写到 `baseline/`。
3. **Day 1–2** — Phase 2 probe：独立脚本，不影响生产。
4. **Day 2–3** — Phase 3：ring buffer + 延迟复用。**这阶段正确性 review 最重**。
5. **Day 3–5** — Phase 4：流水线拆分。**主要收益在这。**
6. **Day 5+** — Phase 5（graph）和 Phase 6（TRT），在 Phase 4 上线并稳定运行几次会话之后再做。

每完成一个阶段，把结果追加到本文档末尾，让项目保留单一真实记录。
