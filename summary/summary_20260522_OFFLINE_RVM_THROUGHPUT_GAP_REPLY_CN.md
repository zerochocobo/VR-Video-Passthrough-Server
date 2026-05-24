# 离线 RVM 吞吐与实时差距 —— 分析与调整建议

日期：2026-05-22

对应问题报告：`summary/summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_EN.md`

## 1. 现象复述

- 同一 `videos/test_8k.mp4`、同一模型、同一 alpha 模式：
  - 实时 alpha：`decode=5-6ms`，`composite≈6ms`，`output_fps=59.940`，吞吐 ~77fps。
  - 离线 alpha（TRT 已激活，`static_trt=True`，`rvm_ort_avg≈5.8ms`）：`decode_avg=20.722ms`，`matting_avg=6.227ms`，`decode_matting_avg=26.949ms`，吞吐 36.32fps。
- 文档已排除：输出 FPS 上限、TRT 未激活、源文件并发音频、alpha pack、`source_index_rewind`、Matter/ORT 初始化顺序、NVENC 参数。

## 2. 根因定位（代码层）

实时 worker `pipeline/pynv_stream.py:2031-2161` 与离线脚本 `tools/offline_alpha_passthrough.py:1668-1700` 的**循环主体一致**，但有一处关键差异：

| 步骤 | 实时 | 离线 |
|---|---|---|
| frame_at | `t0→t1` 计入 `dt_decode` | `td0→td1` 计入 `t_dec` |
| composite | `t1→t2` 异步 launch | `td1→tm1` 异步 launch |
| **sync** | `_CUDA_STREAM.synchronize()`，`t2→t3` 单独计入 `dt_sync` | **缺失** |
| encode | `with encoder_lock: enc.Encode(...)`，`t3→t4` | `enc.Encode(...)`，`tm1→te1` |
| ring slot | `acquire_nv12_output_slot` / `pending_nv12_slots`，强制环形换页 | 直接构造 `GpuNv12AppFrame(out_nv12)` |
| mux write | 写 ffmpeg 管道 → StreamingResponse | 写 ffmpeg 管道 → 本地 MP4 文件 |

`pipeline/pynv_io.py:291-295` 的 `frame_at()` 函数体两端一致，差异不在它本身。

### 真正的开销迁移

- 实时显式 `cuda_stream.synchronize()` 把「上一帧 composite/NVENC 在 CUDA 队列里的尾巴」切到 `dt_sync` 单独统计，因此 `decode=5-6ms` 是**纯净 NVDEC 等待时间**。
- 离线没有显式同步，`composite_nv12()` 仅完成 launch 即返回（6.2ms 是 launch 开销），真正 kernel 还在排队；`enc.Encode()` 在 PyNvVideoCodec 层是阻塞返回 bitstream 的，会把 NVENC 队列里能立即完成的部分等掉，但若 NVENC 未饱和则不阻塞太久。
- 真正等队列的时机出现在下一次 `dec.frame_at()`：PyNvVideoCodec 的 `SimpleDecoder.__getitem__` 内部在把 NVDEC surface 通过 CAI 暴露给 Python 时，会与同一 CUDA context 上挂着的 launch 形成隐式 barrier，将「上一帧 composite/NVENC 残留 GPU 工作」吸收到当前 `frame_at()` 的耗时里。

### 数值核对

- 离线 `decode_avg≈20.7ms` ≈「纯解码 5-6ms」+「上一帧 GPU 队列尾巴 ~14-15ms」。
- 实时单帧累计 `decode(5.5)+composite(6)+sync(5)+encode+mux ≈ 13ms` → 77fps。
- 离线单帧累计 `decode_matting(27)+encode+mux ≈ 27.5ms` → 36fps。
- **实际差距 ≈ 14ms**，与离线缺失的 sync 时段长度一致。

### 实时为什么能并行省下这 14ms？

- `cuda_stream.synchronize()` 在 sync 时段里把 GPU 占满，但 NVDEC 拥有独立硬件单元（与 SM/NVENC 物理分离），可以在此时段并行**预解下一帧**。`SimpleDecoder` 内部维持小型预取队列，下次 `__getitem__` 命中预取后接近零等待。
- 离线缺 sync 时段，前一帧 composite/encode launch 与下一帧 NVDEC 启动的 CUDA 调用挤在同一时间线，NVDEC 拿不到稳定的预取窗口。
- 实时 `acquire_nv12_output_slot` / `pending_nv12_slots` 环缓冲（`pipeline/pynv_stream.py:2079-2129`）还会在 encode 阶段释放 slot，进一步给 composite/NVDEC 让出显存与 CAI 引用计数路径。

## 3. 调整建议（按风险/收益排序）

### 建议 A（最小改动，先验证根因）

在离线两个工具的内层循环 `composite_*` 之后、`enc.Encode(...)` 之前，加入一次显式 CUDA stream 同步，参照实时 worker 的 `t2→t3`：

涉及位置：
- `tools/offline_alpha_passthrough.py:1668-1700`（alpha 主循环）
- `tools/offline_passthrough.py` 中绿幕主循环（与上面对称）
- 同步对象：`matting._CUDA_STREAM` 或 `Matter` 持有的同一 stream 句柄

为避免阻塞测量结果失真，建议**新增一个单独耗时栏目** `t_sync`（与实时 `dt_sync` 一致），并把现有 `t_dec` 解读为「纯解码」而非「解码+残留」。

**预期效果**：离线 `decode_avg` 应回落到 5-7ms，`t_sync` 出现 ~14ms，整体吞吐由 36fps 提升至接近 70fps（不会完全等于实时，受 NVENC 后端写文件耗时与 `acquire_nv12_output_slot` 环缓冲差异影响）。

### 建议 B（让离线更贴近实时拓扑）

在离线脚本中复用 `Matter.acquire_nv12_output_slot()` + `release_nv12_output_slot()` 环缓冲（RVM 绿幕的 `tools/offline_passthrough.py` 已部分采用，alpha 路径未对齐）：

- 让 composite 输出落入显存 ring slot，encode 之后再 release。
- 与建议 A 联合使用，可消除「上一帧设备内存被下一帧 launch 引用」的隐式序列化。

### 建议 C（A/B/C 探针验证）

按 `summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_EN.md` 第 3 条建议落地，在 `base_environment()` 下用三段最小化探针定位耗时归属：

| 探针 | 内容 | 预测 `decode_avg` |
|---|---|---|
| A | 仅 `dec.frame_at()` 跑 899 帧 | 5-6ms |
| B | A + RVM composite + 显式 `_CUDA_STREAM.synchronize()` | 5-6ms（composite ~6ms） |
| C | B 上接 NVENC，但**省略** sync | 跳回 ~20ms（重现现象） |

如果 B vs C 重现 5ms→20ms 的跃迁，本回复定位即被验证；若不重现，则需要再看 NVDEC 内部预取队列大小、CUDA context primary stream 设置。

### 建议 D（与 TensorRT 路线无关，不要混淆）

最近 `utils/trt_manifest.py` 新增的 `original_rvm_model_path()`、`ui/services/trt_warmup_process.py` 新增的 `_static_model_path()/_run_static_shape()` 以及 `utils/rvm_static_onnx`，是为「TRT IOBinding 卡死 / 分区碎片」服务的（参见 `summary_20260521_TENSORRT_IOBINDING_PERFORMANCE_REPLY_CN.md` 与 `summary_20260521_TENSORRT_STATIC_RVM_RUNTIME_*.md`）。本次离线吞吐问题与之**正交**：离线日志 `rvm_ort_avg≈5.8ms` 表示 TRT 静态形状路径正常，本现象不是 TRT 路径回退。

### 建议 E（DLL 路径）

`utils/runtime_dll_paths.py:10-22` 的 `os.add_dll_directory()` 仅在 Windows 且未冻结时生效；裸跑 GPU/ORT 时若未经 `ui.services.process_helpers.base_environment()` 包装，可能引发 CUDA EP 加载静默 fallback。本次离线脚本测得 `static_trt=True` 表明已经走对路径，但调试探针时务必保留同一包装，避免引入新的混淆变量。

## 4. 落地顺序

1. 跑探针 C（仅在工具内部加几行计时，无侵入修改），确认 14ms 跃迁。
2. 落地建议 A：把 `_CUDA_STREAM.synchronize()` 加进两个离线脚本的 composite→encode 之间，并新增 `t_sync` 统计。
3. 若仍有 5-7ms 残差，再落地建议 B（环缓冲对齐）。
4. 通过 `pytest tests/test_offline_convert.py tests/test_settings.py tests/test_offline_alpha_bitrate.py` 验证回归；命令行复测 `videos/test_8k.mp4` 并对照 `decode_avg / matting_avg / t_sync / throughput` 四个数。

## 5. 排除项汇总（沿用问题报告并补充结论）

| 排除项 | 结论 |
|---|---|
| 输出 FPS 上限 | 与本现象无关。 |
| TRT 未激活 | 已激活，`rvm_ort_avg≈5.8ms`。 |
| 音频并发读 | 已抽取 AAC sidecar，仍 36fps，与本现象无关。 |
| alpha vs green pack | alpha pack 0.064ms，绿幕同样 36fps，与本现象无关。 |
| `source_index_rewinds` | 已为 0，与本现象无关。 |
| Matter/ORT 初始化顺序 | 已对齐实时，仍 36fps，与本现象无关。 |
| NVENC 参数 | 已加 `--realtime-encoder-args`，仍 36fps，与本现象无关。 |
| **CUDA stream 同步缺失** | **本次定位的真正根因。** |
| `acquire_nv12_output_slot` 环缓冲缺失 | 次要因素，建议同时对齐。 |
| `PyNvSimpleDecoder.frame_at()` 本身 | 函数体一致，非根因。 |

## 6. 重要提醒

- 本回复只做行为分析与调整建议，不附带代码修改；按工程节奏先由开发执行建议 C 探针验证，再决定建议 A/B 是否落地。
- 不要把本问题当成 TensorRT 通道回退去查，方向会偏。
- 调整离线脚本时仍需通过 `base_environment()` 启动（或经 `offline.convert` / `main.py` / UI 入口间接启动），保持 DLL 与 ORT provider 一致性。
