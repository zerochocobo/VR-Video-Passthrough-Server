# 首包输出延迟（First Chunk Latency）诊断与修复 Patch

日期：2026-05-23

> 父文档：
> - `summary_20260522_ROADMAP_POST_TRT_WARMUP_CN.md`
> - `summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md`（Track A 已落地 A1-A3）
>
> 触发现象：Track A 性能目标全部达成（frame 30=54.92, frame 60=59.92, 稳态 77-78fps）后，仍观察到首播 ~1.77s 静默：
> ```
> 11:44:41.263  mux 命令启动
> 11:44:41.364  first alpha（+101ms，GPU 路径完全就绪）
> 11:44:43.035  frame 120 已处理完（+1772ms）
> 11:44:43.140  reader first stdout chunk（+1877ms）
> ```
> GPU 链路在 1.77s 内已经跑完 120 帧（68fps，比实时还快），但 HTTP 端首字节延迟到 1.88s。瓶颈在 FFmpeg mux 进程内部，与 TensorRT/NVENC 无关。
>
> 本文档记录 A6（诊断打点）与 A7（修复）的可交付 patch 描述。

---

## 0. 已知关键事实（代码侧）

### 0.1 mux 拓扑

`pipeline/pynv_stream.py` 中存在 **3 套 muxer**：

| 路径 | 函数 | 进程数 | 调用条件 |
|---|---|---|---|
| 直 mux | `_open_muxer`（行 1443） | 1 段（直接 mux） | audio_mode 不是 pipe_ts，或 mp4 容器 |
| pipe_ts | `_open_pipe_ts_muxer`（行 1176） | **2 段**（video_proc → final_proc） | mpegts + audio=aac + pipe_ts + cache hit |
| slate pipe_ts | `_open_slate_pipe_ts_muxer`（行 1307） | **2 段** | 同上但 cache miss，走 slate audio server |

播放器实际命中哪条要在 server.log 中确认（每个函数有日志 `pipe_ts video mux cmd` / `pipe_ts final mux cmd`）。**1.77s 静默几乎肯定是 pipe_ts 双段 mux 各自 probe 累加**。

### 0.2 已有的「快速首包」配置（已生效）

直 mux 与 pipe_ts 都已经设置：
- `-flush_packets 1`
- `-muxdelay 0`
- `-muxpreload 0`
- mpegts: `-mpegts_flags +resend_headers`，`-pat_period 0.1`，`-sdt_period 0.5`
- fMP4: `+frag_keyframe+empty_moov+default_base_moof`，`-frag_duration 250000` (250ms)

### 0.3 **关键缺口：主 video pipe 没设 probesize/analyzeduration**

| 路径 | stdin 输入 | probesize | analyzeduration |
|---|---|---|---|
| `_open_muxer` 主输入（行 1495-1503） | stdin HEVC | **未设（FFmpeg 默认 5MB）** | **未设（默认 5s）** |
| `_open_pipe_ts_muxer.video_cmd`（行 1182-1222） | stdin HEVC | **未设** | **未设** |
| `_open_pipe_ts_muxer.final_cmd`（行 1223-1278） | stdin mpegts + aac 文件 | **未设** | **未设** |
| audio 子流程（行 525-532 等） | 文件输入 | 32768 | 0 |

FFmpeg 对 stdin pipe input 在 raw HEVC 模式下，默认要收够 `probesize` 字节或者 `analyzeduration` 时长才开始 mux 输出。8K HEVC bitrate ~50Mbps，5MB 需要 ~0.8s；两段 mux 各自付 → 接近实测 1.77s。

---

## 1. A6：mux 链路细分时间戳（诊断不改逻辑）

### 1.1 目标

在 server.log 中打出 5 个关键时间戳，得到 T0-T5 实测 delta，**用数据验证** 0.3 的猜想，并精确定位剩余 latency 来源。

### 1.2 打点位置

| 标识 | 含义 | 落地位置 |
|---|---|---|
| **T0** | mux Popen 返回（subprocess 启动完成） | `_open_pipe_ts_muxer:1281`（video_proc）与 `:1291`（final_proc）`Popen()` 返回后 |
| **T1** | 首个 HEVC packet 写入 video mux stdin | encoder 输出循环里第一次 `self._video_mux.stdin.write(...)` 或 `self._mux.stdin.write(...)` 之前 |
| **T2** | video mux 首次 stderr 行 | `_stderr_reader` / `_stderr_loop` 内首次拿到 video_proc 行 |
| **T3** | final mux 首次 stdout 字节 | `_reader_loop:1654-1662`，把现有 `reader first stdout chunk` 计入；video_proc 没有 stdout reader（管道直接 pipe 给 final_proc），故 T3 无独立采点 |
| **T4** | reader 首次拿到 chunk（即现有 `reader first stdout chunk`） | `_reader_loop:1662`（已有） |
| **T5** | HTTP route 首次 yield 给客户端 | `http_app/routes_media.py` 的 streaming response 生成器首次 `yield`（如果是 starlette `StreamingResponse(iter)`，加 wrapper） |

### 1.3 数据结构

每个 stream 维护一个 dict：

```python
self._first_chunk_marks: dict[str, float] = {}

def _mark_first(self, key: str) -> None:
    if key not in self._first_chunk_marks:
        self._first_chunk_marks[key] = time.perf_counter()
        log.info(
            "[DIAG][MUX][%d] mark key=%s monotonic=%.6f delta_from_T0_ms=%.1f",
            self.sid, key, self._first_chunk_marks[key],
            (self._first_chunk_marks[key] - self._first_chunk_marks.get("T0_mux_spawn", self._first_chunk_marks[key])) * 1000.0,
        )
```

落地点调用 `self._mark_first("T0_mux_spawn")` 等。

### 1.4 汇总日志

`_reader_loop` 拿到首字节后追加一行 summary：

```python
if first:
    first = False
    t_now = time.perf_counter()
    t0 = self._first_chunk_marks.get("T0_mux_spawn", t_now)
    log.info(
        "[DIAG][MUX][%d] first_chunk_breakdown "
        "T0_spawn=%.1fms T1_write=%.1fms T2_stderr=%.1fms "
        "T4_reader=%.1fms total=%.1fms",
        self.sid,
        0.0,
        (self._first_chunk_marks.get("T1_first_write", t_now) - t0) * 1000.0,
        (self._first_chunk_marks.get("T2_first_stderr", t_now) - t0) * 1000.0,
        (t_now - t0) * 1000.0,
        (t_now - t0) * 1000.0,
    )
```

### 1.5 配置项

```python
# config.py
PT_MUX_LATENCY_DIAG = bool(int(os.environ.get("PT_MUX_LATENCY_DIAG", "1")))
```

默认开启（开销可忽略）。

### 1.6 验证

冷启动后跑一次首播，server.log 中应看到 `[DIAG][MUX][...] first_chunk_breakdown ...` 一行。把 5 个 delta 贴回本文档 1.7 节。

### 1.7 数据落点（A6 跑完填入）

```
T0_spawn:        0.0ms
T1_first_write:  __ms     # video pipeline 首次写 HEVC packet 到 mux stdin
T2_first_stderr: __ms     # FFmpeg 自己说 "Stream mapping" 等启动行
T4_reader:       __ms     # reader 拿到首字节
total:           __ms

判断：
- T1 大（>500ms）→ encoder 输出节奏问题（不太可能，A3 已优化）
- T2-T1 > 500ms → FFmpeg probe 等待 stdin
- T4-T2 > 500ms → mux 内部 first-fragment 凑齐 / interleave 阻塞
```

### 1.8 风险

- 写 stdin 的代码可能在多处（`_reader_loop` 之外，比如 encoder 输出循环 `_mux_write_loop`）。打点 T1 需要找到所有 stdin.write 路径并 OR-mark。
- HTTP layer T5 打点涉及修改 `routes_media.py` streaming generator，可选 — 优先 T1-T4。

---

## 2. A7：FFmpeg mux 首包参数调优（修复）

### 2.1 优先级最高的修复（基于 0.3 的代码事实）

#### A7.1 给所有 stdin pipe input 加 probesize/analyzeduration

**改 3 处**：

| 文件:行 | 函数 | 加在哪 |
|---|---|---|
| `pipeline/pynv_stream.py:1488-1503` | `_open_muxer` 直 mux | `base_args` 之后、`-i -` 之前的 input_args 段开头 |
| `pipeline/pynv_stream.py:1182-1197` | `_open_pipe_ts_muxer.video_cmd` | `-thread_queue_size` 与 `-f input_format` 之间 |
| `pipeline/pynv_stream.py:1228-1234` | `_open_pipe_ts_muxer.final_cmd` | `-thread_queue_size` 与 `-f mpegts` 之间 |
| `pipeline/pynv_stream.py:1307+` | `_open_slate_pipe_ts_muxer`（同 pipe_ts 结构） | 同上对应位置 |

插入：

```python
"-probesize", "32",
"-analyzeduration", "0",
```

最小值 `32` 而不是 audio path 用的 `32768` —— raw HEVC stdin 不需要 probe（已经声明 `-f hevc`），让 FFmpeg 立即开始 mux。

**预期收益**：T2 从 ~1s+ → < 100ms。

#### A7.2 fMP4 frag_duration 降低

`pipeline/pynv_stream.py:1474-1475`：

```python
"-frag_duration", "250000",  # 250ms
```

→

```python
"-frag_duration", str(config.PASSTHROUGH_FMP4_FRAG_DURATION_US),  # 默认 100ms
```

配置：

```python
# config.py
PASSTHROUGH_FMP4_FRAG_DURATION_US = int(os.environ.get("PT_FMP4_FRAG_DURATION_US", "100000"))
```

**预期收益**：fMP4 容器首 fragment 等待 250ms → 100ms。注意：太小（< 50ms）会让 fragment header overhead 上升，DeoVR 兼容性需要测。

#### A7.3 增加 `-fflags +nobuffer`

`-fflags +genpts` → `-fflags +genpts+nobuffer+flush_packets`

落地点（3 处）：
- `_open_muxer:1485-1487`
- `_open_pipe_ts_muxer.video_cmd:1187-1188`
- `_open_pipe_ts_muxer.final_cmd:1228-1229`

`+nobuffer` 减少输入 buffer，`+flush_packets` 与已有 `-flush_packets 1` 互补（一个是 mux 输出层，一个是 demux 输入层）。

#### A7.4 video_proc 中间 mpegts 加 `-flush_packets 1`

`_open_pipe_ts_muxer` 的 `video_cmd` 已经有 `-flush_packets 1`（行 1205-1206）。**已 OK，不动**。

#### A7.5 final_proc audio aac 输入 `-fflags +nobuffer`

audio 输入是 file 不是 pipe，但仍然有 demux probe。加 `-probesize 32 -analyzeduration 0` 到 audio input args。

### 2.2 验证矩阵

按 A7.1 - A7.5 顺序逐个落地，每个落地后跑一次冷启动首播，记录 first_chunk_breakdown：

| 步骤 | T2 期望 | T4 期望 | 备注 |
|---|---|---|---|
| baseline（A7 落地前） | ~1000ms | ~1770ms | A6 跑完先记 |
| A7.1（probesize） | < 100ms | < 800ms | 单步主收益 |
| + A7.2（fMP4 frag） | < 100ms | < 600ms | 仅 mp4 容器有效 |
| + A7.3（nobuffer） | < 50ms | < 500ms | 边际 |
| + A7.5（audio probe） | < 50ms | < 400ms | 仅 audio enabled 有效 |

**最终目标**：reader first chunk < 300ms（含 frame 0 GPU 处理 + encoder + mux）。

### 2.3 排除性实验（A6.5）

如果 A7.1 落地后 T4 仍然 > 1s，跑一次：

```
PT_FORCE_AUDIO_OFF=1
```

（强制 `_audio_mode` 返回 "off"，绕过 pipe_ts 双段 mux 直接走单段 `_open_muxer`）

如果此时 first chunk < 300ms：问题在双段 pipe_ts 串联，而非 probesize。下一步改造 audio interleave 逻辑或评估单段 mux 直接喂 aac。

如果此时 first chunk 仍 > 1s：问题不在 probesize/audio，而在更深处（reader 线程、HTTP layer、NVENC 输出节奏）。

### 2.4 风险

- **`-probesize 32`** 极小，对正常 raw HEVC 输入无害（已声明 `-f hevc`），但如果未来切换到非声明输入会失败。当前所有 mux 都显式 `-f hevc`/`-f h264`，安全。
- **fMP4 frag_duration 100ms** 可能让 DeoVR/Quest3 兼容性下降。必须真机测试。
- **`+nobuffer`** 在网络/UDP 输入下会丢包；当前都是 stdin/file 输入，安全。
- **audio interleave** 关闭 `max_interleave_delta` 在某些播放器会引起音画不同步。如果出现，回到当前默认。

### 2.5 回退路径

每个 flag 都通过 `config.py` 暴露，环境变量可一键关闭：

```
PT_MUX_PROBESIZE_OVERRIDE=         # 空 = 不加 probesize（回退到 baseline）
PT_FMP4_FRAG_DURATION_US=250000    # 回到原 250ms
PT_MUX_NOBUFFER_ENABLE=0           # 不加 +nobuffer
```

### 2.6 完成判定

- A6 数据出齐，1.7 节 5 个 delta 填完
- A7.1 落地后 reader first chunk delta < 800ms
- A7.1-A7.5 全部落地后 reader first chunk delta < 300ms
- 真机播放（VLC、Quest3 DeoVR）无音画不同步、无起播失败
- 16+ 现有测试通过

---

## 3. 推荐落地顺序

| 步 | 内容 | 工作量 | 阻塞性 |
|---|---|---|---|
| **A6** | T0-T4 时间戳打点 + first_chunk_breakdown 汇总日志 | 0.5d | 高（无数据不动 A7） |
| **A6.5** | `PT_FORCE_AUDIO_OFF=1` 跑一次对照 | 0.5h | 高（区分 pipe_ts vs 普通 mux） |
| **A7.1** | 3 处 stdin pipe 加 probesize=32 + analyzeduration=0 | 0.5d | 主收益 |
| **A7.2** | fMP4 frag_duration → 100ms（仅 mp4 容器） | 0.5h | 中 |
| **A7.3** | `+nobuffer+flush_packets` | 0.5h | 边际 |
| **A7.5** | audio file 输入 probesize/analyzeduration | 0.5h | 仅 audio enabled |
| **A7.v** | 真机重测（VLC + Quest3 DeoVR） | 0.5d | 验收 |

总预算：**1.5-2 天**。

---

## 4. 不在本 patch 中

- **A6 中 T5（HTTP layer yield）**：可选，A7 落地后若仍有间隙再加。
- **NVENC look-ahead / B 帧关闭**：当前 BF 推断为 0（看 `_pynv_encoder_kwargs`）。若 A7 完成后仍有问题再查。
- **GOP/IDR 间隔调整**：frame 0 已是 IDR，不阻塞首包。
- **reader 线程异步 yield 重构**：当前 `_reader_loop` + async queue 已是合理结构，先排除上游。
- **encoder pool**：A3 startup_preflight 已经付掉 process-级 SDK init；per-stream encoder create 实测应该 < 30ms，不再优化。

---

## 5. 与 Track A 的关系

本 patch 是 Track A 收尾的补充。Track A 原计划 A1-A5 攻 GPU 侧 ramp-up，全部达标。但「首播 1.7s 静默」是一个独立维度的体验问题（FFmpeg mux 首包），**建议作为 Track A 的 A6/A7 收入**，而不是新开一个 Track，以保持「首播平滑」的整体语义闭合。

完成后回写父文档 `summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md` 增加 A6/A7 章节与实施记录。

---

## 6. 实施记录

```
2026-05-23: A6 落地，新增 PT_MUX_LATENCY_DIAG=1 默认开启。
            PyNvPassthroughStream 记录 T0_video_mux_spawn / T0_mux_spawn /
            T1_first_write / T2_first_stderr / T4_reader，并在 reader 首包时输出：
            [DIAG][MUX][sid] first_chunk_breakdown ...

2026-05-23: A6.5 落地，新增 PT_FORCE_AUDIO_OFF=1，用于强制单段 video-only mux 对照实验。

2026-05-23: A7.1 / A7.3 / A7.5 落地。
            新增 PT_MUX_PROBESIZE_OVERRIDE=32、PT_MUX_ANALYZEDURATION_US=0、
            PT_MUX_NOBUFFER_ENABLE=1；直 mux、pipe_ts、slate pipe_ts 的 video stdin、
            中间 mpegts stdin、audio aac/file/tcp 输入均应用可回退 probe/analyze 参数。
            FFmpeg -fflags 默认从 +genpts 改为 +genpts+nobuffer+flush_packets。

2026-05-23: A7.2 落地。
            fMP4 -frag_duration 从硬编码 250000 改为 PT_FMP4_FRAG_DURATION_US，
            默认 100000，设置 250000 可回退。

2026-05-23: 自动化验证：
            compileall config.py pipeline\pynv_stream.py tests\test_config_defaults.py
            tests\test_pynv_mux_latency.py 通过；
            uv run python -m pytest tests\test_config_defaults.py tests\test_predict_warmup_state.py
            tests\test_alpha_packer.py tests\test_vr_naming.py tests\test_pynv_startup_preflight.py
            tests\test_pynv_mux_latency.py tests\test_main_args.py => 38 passed；
            git diff --check 仅 CRLF 提示。

2026-05-23: 待真机复测填入：
            first_chunk_breakdown:
            T0_video_spawn=__ms T0_spawn=0.0ms T1_write=__ms
            T2_stderr=__ms T4_reader=__ms total=__ms
            VLC __，Quest3 DeoVR __

2026-05-23: A7 首轮真机日志：
            命令参数已生效，但 first_chunk_breakdown 为
            T0_video_spawn=-5.5ms T0_spawn=0.0ms T1_write=176.8ms
            T2_stderr=300.2ms T4_reader=1908.9ms total=1908.9ms。
            结论：raw HEVC stdin probe 已不是瓶颈；剩余延迟在 final mux
            T2->T4。日志同时出现 final mux 的
            `Could not find codec parameters ... unspecified size` 与
            `Consider increasing ... analyzeduration/probesize`，说明把
            `-probesize 32` 应用到 final-stage MPEG-TS stdin 过于激进。

2026-05-23: probe 分档修正：
            raw HEVC/H264 stdin 保持 PT_MUX_PROBESIZE_OVERRIDE=32；
            新增 PT_MUX_CONTAINER_PROBESIZE_OVERRIDE=32768，供 pipe_ts final
            MPEG-TS stdin 和普通容器输入使用；
            新增 PT_MUX_AUDIO_PROBESIZE_OVERRIDE=32768，供 AAC/file/tcp audio
            输入使用。自动化验证：compileall 通过，
            uv run python -m pytest tests\test_config_defaults.py
            tests\test_pynv_mux_latency.py tests\test_pynv_startup_preflight.py
            => 8 passed。

2026-05-23: probe 分档后第二轮真机日志：
            pipe_ts video raw stdin 使用 probesize=32，final mpegts/audio 使用
            probesize=32768，但 first_chunk_breakdown 仍为
            T1_write=190.7ms T2_stderr=298.4ms T4_reader=1909.5ms total=1909.5ms；
            final mux 仍报 `Could not find codec parameters ... unspecified size`。
            结论：剩余 1.9s 不是简单 probesize 数值问题，而是 double-stage
            pipe_ts final mux 对中间 TS 的 HEVC 参数解析/等待。

2026-05-23: 默认路径改为 single-stage setts：
            PT_PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE 默认从 pipe_ts 改为 setts，
            复用 AAC cache 但绕过 video_proc -> final_proc 双段 mux。
            同时修正 single-stage setts 的时间戳步长：从旧硬编码
            pts=N*3000/dts=N*3000 改为按 fps 计算；59.94fps 容差内固定为
            1502 tick。pipe_ts 仍可通过
            PT_PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE=pipe_ts 回退。
            自动化验证：相关回归 39 passed；git diff --check 仅 CRLF 提示。

2026-05-23: single-stage setts 回退：
            用户实测表现为播放器启动卡住/无内容。日志确认 setts 单段 mux 能输出首包
            （短样例 total=1118.2ms），但后续输出量异常低，多个 nPlayer seek/retry
            只发送数百 KB 即被 preempt，播放器无法进入正常播放。结论：single-stage
            setts 与当前 live alpha 客户端路径不兼容，不能作为默认。
            默认 PT_PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE 已恢复为 pipe_ts；
            setts 的 59.94fps tick 修正保留为实验能力。
            同时修复 `_worker_loop` 早期失败清理路径中的
            `UnboundLocalError: slate_stop`：在函数入口预初始化 slate_stop/slate_thread。
            自动化验证：相关回归 39 passed；git diff --check 仅 CRLF 提示。

2026-05-23: audio-only 回归根因修复：
            用户反馈真实播放器进入音频模式而非视频模式。专家指出根因为 raw HEVC
            stdin 使用 `-probesize 32 -analyzeduration 0`，导致 FFmpeg 无法稳定解析
            VPS/SPS/PPS，最终 MPEG-TS 缺 video codec params，严格播放器判定为
            audio-only。已按方案 A 修复：
            `_mux_probe_args(..., for_raw_video=True)` 对 raw HEVC/H264 输入返回空参数；
            pipe_ts video、slate pipe_ts video、direct mux video-only、direct mux audio-enabled
            四处 raw video stdin 均改为 `for_raw_video=True`。
            container/audio 输入仍保留 32768/0。测试更新断言 raw video 输入前不再出现
            `-probesize` / `-analyzeduration`，final/audio 仍有 32768。
            自动化验证：compileall 通过，相关回归 39 passed，git diff --check 仅 CRLF 提示。
```
