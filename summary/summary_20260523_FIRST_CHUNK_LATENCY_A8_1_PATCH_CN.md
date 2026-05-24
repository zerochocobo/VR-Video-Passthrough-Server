# A8.1 patch description — Final mux T2→T4 split-stage timing diagnostic

日期: 2026-05-23
所属轨道: Track A / 阶段 A8 (Post-rollback first-chunk latency diagnostic)
前置阶段: A6 (T0/T1/T2/T4 framework) + A7 (probe tuning, partial rollback)
状态: WAITING_FOR_IMPLEMENTATION (本文档为开发可执行的设计稿)

---

## 0. 背景

当前 first-chunk breakdown 输出大致形如：

```
[DIAG][MUX][sid] first_chunk_breakdown
  T0_video_spawn=12.3ms
  T0_spawn=0.0ms
  T1_write=180.5ms
  T2_stderr=420.8ms
  T4_reader=2870.4ms
  total=2870.4ms
```

- A6/A7 着陆后，`T2 → T4` 仍存在 **2.4–3.6s** 黑箱。
- 当前 `_drain_stderr` 在 `label in {"ffmpeg", "ffmpeg-video"}` 时统一打 `T2_first_stderr`，
  这意味着 `video_proc`（前级 mpegts）和 `final_proc`（后级 mux + AAC 合流）的 stderr 首行被合并为单一时间点。
- 必须先把 `T2 → T4` 拆细，才能定位真正瓶颈（视频参数集解析？AAC 缓存抓取？interleave 等待？输出头写入？）。

> 边界约束：本阶段 **不** 修改 HEVC bsf / parser / 编码器路径；仅向 `_drain_stderr` / `_log_first_chunk_breakdown` 注入可观测点，**纯加测量、不改业务**。

---

## 1. 目标

将单点 `T2_first_stderr` 拆为 5 个独立时间戳：

| key | 含义 | 触发条件 |
|---|---|---|
| `T2a_video_first_stderr` | `video_proc` (前级 mpegts mux) 输出 stderr 首行 | label == `"ffmpeg-video"` |
| `T2b_final_first_stderr` | `final_proc` (后级合流 mux) 输出 stderr 首行 | label == `"ffmpeg"` |
| `T3a_final_video_codec` | `final_proc` 解出视频流编解码参数 | label == `"ffmpeg"` 且行内匹配 `Stream #0:0` 或 `Video: hevc` |
| `T3b_final_audio_codec` | `final_proc` 解出音频流编解码参数 | label == `"ffmpeg"` 且行内匹配 `Stream #0:1` / `Stream #1:0` 或 `Audio: aac` |
| `T3c_final_output_ready` | `final_proc` 进入输出阶段（`Output #0` 行） | label == `"ffmpeg"` 且行内匹配 `Output #0` |

直接 mux 路径 (`_open_muxer`) 同样适用 T2b / T3a / T3b / T3c（label 为 `"ffmpeg"`）；
T2a 仅在 pipe_ts / slate pipe_ts 双段链路中出现。

---

## 2. 改动定位

文件: `pipeline/pynv_stream.py`

涉及函数（**只改这两处**）:
- `_drain_stderr` (现行 line 1751-1766)
- `_log_first_chunk_breakdown` (现行 line 379-402)

新增配置（可选，default ON）:
- `config.MUX_LATENCY_DIAG_VERBOSE` — 控制 T3a/T3b/T3c 的正则匹配是否启用；
  即使关闭，T2a/T2b 也照常打点（成本零）。

---

## 3. 设计细节

### 3.1 `_drain_stderr` 改造

伪代码（仅描述行为，开发同学按现有代码风格落地）:

```python
# 在 _drain_stderr 进入循环前预编译 / 选择 mark key:
if label == "ffmpeg-video":
    first_key = "T2a_video_first_stderr"
elif label == "ffmpeg":
    first_key = "T2b_final_first_stderr"
else:
    first_key = None

# 仅对 final_proc 启用 stage marker (因为前级输出是 raw mpegts 字节流，没有 "Stream"/"Output" 文本)
enable_stage_marks = (label == "ffmpeg") and getattr(config, "MUX_LATENCY_DIAG_VERBOSE", True)

# 循环内（每读到 text 后）:
if first_key:
    self._mark_first(first_key)

if enable_stage_marks:
    # 视频流参数解析完成
    if "Stream #" in text and ("Video: hevc" in text or "Video: h264" in text):
        self._mark_first("T3a_final_video_codec")
    # 音频流参数解析完成
    elif "Stream #" in text and "Audio: " in text:
        self._mark_first("T3b_final_audio_codec")
    # 输出 header 写入阶段
    elif text.startswith("Output #0") or "Output #0," in text:
        self._mark_first("T3c_final_output_ready")
```

要点:
1. **保留** 现有 `T2_first_stderr` mark（向后兼容；可在 A8 收尾时再决定是否移除）。
   推荐：在 `first_key` 被打点的同时，仍调用一次 `self._mark_first("T2_first_stderr")`，
   行为与现行相同（首位先到者写入），不破坏旧日志解析脚本。
2. **不要** 引入复杂正则；`text in ...`/`startswith` 即可，避免每行 stderr 多开销。
3. T3a/T3b/T3c 匹配失败时 `_mark_first` 不会被调用，breakdown 输出对应字段为 `-1.0`，
   可由 `delta_ms` 自然降级。

### 3.2 `_log_first_chunk_breakdown` 扩展

伪代码:

```python
log.info(
    "[DIAG][MUX][%d] first_chunk_breakdown "
    "T0_video_spawn=%.1fms T0_spawn=0.0ms T1_write=%.1fms "
    "T2a_video=%.1fms T2b_final=%.1fms "
    "T3a_vcodec=%.1fms T3b_acodec=%.1fms T3c_output=%.1fms "
    "T4_reader=%.1fms total=%.1fms",
    self.sid,
    delta_ms("T0_video_mux_spawn"),
    delta_ms("T1_first_write"),
    delta_ms("T2a_video_first_stderr"),
    delta_ms("T2b_final_first_stderr"),
    delta_ms("T3a_final_video_codec"),
    delta_ms("T3b_final_audio_codec"),
    delta_ms("T3c_final_output_ready"),
    delta_ms("T4_reader"),
    (now - t0) * 1000.0,
)
```

注意:
- 直接 mux 路径下 `T2a_video=-1.0` 是正常现象（不存在前级）。
- 如 final mux 一开始就被 kill（startup_error），整组 T3 可能全 -1.0，属预期。

---

## 4. 数据采集与解读

完成落地后，跑一次冷启动 + 首次播放，期望日志:

```
[DIAG][MUX][1] mark key=T0_video_mux_spawn delta_from_T0_ms=-XX.X
[DIAG][MUX][1] mark key=T0_mux_spawn delta_from_T0_ms=0.0
[DIAG][MUX][1] mark key=T1_first_write delta_from_T0_ms=XXX.X
[DIAG][MUX][1] mark key=T2a_video_first_stderr delta_from_T0_ms=XXX.X
[DIAG][MUX][1] mark key=T2b_final_first_stderr delta_from_T0_ms=XXX.X
[DIAG][MUX][1] mark key=T3a_final_video_codec delta_from_T0_ms=XXX.X
[DIAG][MUX][1] mark key=T3b_final_audio_codec delta_from_T0_ms=XXX.X
[DIAG][MUX][1] mark key=T3c_final_output_ready delta_from_T0_ms=XXX.X
[DIAG][MUX][1] mark key=T4_reader delta_from_T0_ms=XXX.X
[DIAG][MUX][1] first_chunk_breakdown ... total=XXXX.Xms
```

### 4.1 解读矩阵

| 区段 | 含义 | 大值（>X ms）含义 |
|---|---|---|
| `T2b − T2a` | 前级 mpegts 到达后级 stdin 的延迟 | 通常 < 20ms，>100ms 说明 OS pipe 阻塞/前级首包慢 |
| `T3a − T2b` | final mux 从 stdin 探测出视频参数耗时 | >300ms → mpegts container probe 仍在拉数据 / VPS-SPS-PPS 延迟到达 |
| `T3b − T3a` | final mux 探测出音频参数耗时 | >300ms → AAC 文件解析慢 / I/O 等待 |
| `T3c − T3b` | mux 头写入耗时 | >50ms → fMP4/mpegts header 计算异常 |
| `T4 − T3c` | 写出第一字节到 stdout 的延迟 | >100ms → interleave delta / fragment 凑包 / HTTP queue |

### 4.2 数据槽（待填）

| 场景 | T1 | T2a | T2b | T3a | T3b | T3c | T4 | total |
|---|---|---|---|---|---|---|---|---|
| pipe_ts (默认) |  |  |  |  |  |  |  |  |
| slate pipe_ts |  |  |  |  |  |  |  |  |
| 直接 mux fmp4 |  |  |  |  |  |  |  |  |
| 直接 mux mpegts |  |  |  |  |  |  |  |  |

---

## 5. 风险与回退

| 风险 | 等级 | 缓解 |
|---|---|---|
| FFmpeg 不同版本 stderr 文本格式变化（`Stream #0:0` vs `Stream 0:0`） | 低 | 匹配失败时降级为 `-1.0`，不影响主流程 |
| 每行 stderr 多 3 次子串 `in` 检查 | 极低 | stderr 总行数有限（几十行），<<1ms |
| `MUX_LATENCY_DIAG_VERBOSE=0` 临时关闭 | n/a | 已设开关，问题时一秒回退 |
| 兼容旧日志解析 | 低 | 保留 `T2_first_stderr` 字段，新字段为追加 |

无业务路径改动，**A8.1 不会引起 audio-only / 卡顿 / 解码失败回归**。

---

## 6. 验收

1. 默认 `MUX_LATENCY_DIAG=1`、`MUX_LATENCY_DIAG_VERBOSE=1`。
2. 重启服务 → 冷启动播放一次 → 关闭播放器。
3. 日志包含至少 `T0_video_mux_spawn / T0_mux_spawn / T1_first_write / T2a_video_first_stderr / T2b_final_first_stderr / T3a_final_video_codec / T3b_final_audio_codec / T3c_final_output_ready / T4_reader` 9 个 mark 行（直接 mux 缺 T2a）。
4. `first_chunk_breakdown` 单行包含全部新字段，`total` 与原口径一致（±2ms 内）。
5. 用 `MUX_LATENCY_DIAG_VERBOSE=0` 再跑一次，确认 T3a/T3b/T3c 为 `-1.0` 而 T2a/T2b 仍存在。

---

## 7. 下游决策树

依据采集到的区段值，触发后续阶段:

- `T3a − T2b > 300ms` → 进入 **A8.x-video-probe**: 评估给 final mux stdin 加 `-fflags +discardcorrupt`、或在前级 mux 输出 mpegts PAT/PMT 周期 (`-mpegts_pat_period 0.02`)。
- `T3b − T3a > 300ms` → 进入 **A8.3 AAC cache 预热** 或评估 `-thread_queue_size`。
- `T4 − T3c > 100ms` → 进入 **A8.2 `max_interleave_delta=0`** 试验、或评估 `-flush_packets 1`。
- `T2b − T2a > 100ms` → 进入 **A8.x-pipe**：检查 OS pipe buffer / 前级 stdout 阻塞，评估缩短前级链路。

> 各下游阶段的 patch 文档会在 A8.1 采集到首批数据后单独输出，避免无证据盲改。

---

## 8. 不要做的事（再次强调）

- ❌ 不要在 A8.1 顺手加 `hevc_metadata=aud=insert` / `-probesize 32` 给 raw HEVC stdin（已在 nPlayer 回归中证实致命）。
- ❌ 不要修改 `_open_pipe_ts_muxer` / `_open_muxer` / `_open_slate_pipe_ts_muxer` 任何命令行参数。
- ❌ 不要把 stderr 解析提取到独立线程池（一行就是一次 readline，本就阻塞 IO，多线程无意义）。
- ❌ 不要 backport 到 `MUX_LATENCY_DIAG=0` 路径——观测代码全部包裹在 diag 分支内。

---

## 9. 交付物

1. 本设计稿（已在 `summary/summary_20260523_FIRST_CHUNK_LATENCY_A8_1_PATCH_CN.md`）。
2. 落地后开发同学在本文件 **附录** 追加：
   - 实际改动 diff（仅 `_drain_stderr` / `_log_first_chunk_breakdown` 两处）
   - 三种场景（pipe_ts / slate / 直接 mux）的实际 breakdown 单行日志
   - 解读结论：T2→T4 最大瓶颈段是哪段
3. 完成后将"下一阶段编号 + 假设"返回到 Track A 主文档 `summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md` 的第 10 节。

---

## 附录 A. 落地记录（2026-05-23）

### A.1 实际改动

- `pipeline/pynv_stream.py`
  - `_log_first_chunk_breakdown()` 保留旧字段 `T2_stderr`，追加：
    - `T2a_video`
    - `T2b_final`
    - `T3a_vcodec`
    - `T3b_acodec`
    - `T3c_output`
  - `_drain_stderr()` 按 label 拆分首行 stderr：
    - `ffmpeg-video` → `T2a_video_first_stderr`
    - `ffmpeg` → `T2b_final_first_stderr`
    - 同时保留旧 `T2_first_stderr`
  - `_drain_stderr()` 仅在 final mux (`label == "ffmpeg"`) 且 `MUX_LATENCY_DIAG_VERBOSE=1` 时匹配：
    - `Stream #` + `Video: hevc/h264` → `T3a_final_video_codec`
    - `Stream #` + `Audio:` → `T3b_final_audio_codec`
    - `Output #0` → `T3c_final_output_ready`
- `config.py`
  - 新增 `PT_MUX_LATENCY_DIAG_VERBOSE`，默认 `1`。
- `tests/test_pynv_mux_latency.py`
  - 增加 stderr drain 单元测试，覆盖 T2a/T2b/T3a/T3b/T3c 标记与 verbose 关闭降级。
- `tests/test_config_defaults.py`
  - 锁定 `MUX_LATENCY_DIAG_VERBOSE=True` 默认值。

### A.2 明确未改动

- 未修改 `_open_pipe_ts_muxer()` / `_open_muxer()` / `_open_slate_pipe_ts_muxer()` 的 FFmpeg 命令行参数。
- 未恢复 `hevc_metadata=aud=insert,setts=...`。
- 未给 raw HEVC stdin 或 intermediate TS stdin 添加低 `-probesize` / `-analyzeduration`。

### A.3 验证

```powershell
python -m compileall config.py pipeline\pynv_stream.py tests\test_pynv_mux_latency.py tests\test_config_defaults.py
uv run python -m pytest tests\test_pynv_mux_latency.py tests\test_config_defaults.py tests\test_pynv_startup_preflight.py
```

结果：`13 passed`。

### A.4 待采集

重启 server 后播放一轮，收集：

```text
[DIAG][MUX][sid] mark key=T2a_video_first_stderr ...
[DIAG][MUX][sid] mark key=T2b_final_first_stderr ...
[DIAG][MUX][sid] mark key=T3a_final_video_codec ...
[DIAG][MUX][sid] mark key=T3b_final_audio_codec ...
[DIAG][MUX][sid] mark key=T3c_final_output_ready ...
[DIAG][MUX][sid] first_chunk_breakdown ... T2a_video=... T2b_final=... T3a_vcodec=... T3b_acodec=... T3c_output=... T4_reader=... total=...
```

拿到新日志后再根据第 7 节决策树选择 A8.2 / A8.3 / A8.x-video-probe / A8.x-pipe。
