# 中段 seek 音画不同步根因分析与缓解方向

日期：2026-05-23
背景：用户报告 passthrough live MPEG-TS 流在中间章节点入时音画不同步可达数秒，
TensorRT 模式比 CUDA EP 模式更严重。垫片（slate）机制因 strict 播放器音频化降级
问题已确认不可去除（见 `summary_20260523_NPLAYER_AUDIO_ONLY_REGRESSION_CN.md`）。

---

## 1. 数据流概览

播放 URL：`GET /passthrough_live/{name}?t=T`
默认 `PT_PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE=pipe_ts`，启用 `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE=1`、`PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE=1`。

两段 mux：

1. `video_proc`：`pipeline/pynv_stream.py:1332-1373` / `1462-1509`（slate 变体）
   - stdin = NVENC 原始 HEVC bitstream（slate 阶段 + 真实阶段共用）
   - 关键 bsf：`setts=time_base=1/90000:pts=N*1502:dts=N*1502`
     （`_mpegts_video_bsf` + `_mpegts_tick_for_fps(fps)`）
   - 输出：中间 MPEG-TS

2. `final_proc`：`pipeline/pynv_stream.py:1374-1431` / `:1510-1564`（slate 变体）
   - 输入 1（aac）：cache 或 TCP `tcp://127.0.0.1:port`（slate 期间是 anullsrc 静音，切换后是真实源）
   - 输入 0（mpegts）：video_proc 的 stdout（slate HEVC + 真实 HEVC 拼接）
   - 输出：最终 TS

垫片视频在 `pipeline/pynv_stream.py:2142-2211` `slate_video_loop`，垫片音频服务在
`:1165-1311` `_slate_audio_server_loop`。

---

## 2. 时间戳模型对比

| 维度 | 视频侧 | 音频侧 |
|---|---|---|
| 编码源 | NVENC 重编 NV12（绿屏帧 + 真实合成帧） | `ffmpeg -re -f lavfi -i anullsrc=...`（静音）+ 真实 AAC 源 |
| 进入 mux 的载体 | video_proc.stdin → bsf `setts=N*1502` | TCP socket → final_proc 的 `-f aac` 输入 |
| PTS 推进方式 | **按包序号 N**（与 wallclock 解耦） | **wallclock 同步**（受 `-re` 节流） |
| start 偏移控制 | 无（PTS 从 0 开始累加，slate + real 同一条流） | slate 静音从 0 开始；切换后用 `-ss start_sec` 选源，但 PTS 仍由 ffmpeg 给 final mux 时按 TCP 流接续 |

**核心错配**：视频 bsf 完全无视真实 wallclock，PTS 直接乘以帧序号；音频静音严格按
wallclock 输出。两条流在 final mux 里以"同一个起点 0"对齐，于是 wallclock
差异 = PTS 差异。

---

## 3. burst 突发帧机制

`pipeline/pynv_stream.py:2188-2196`：

```python
slate_frames_box[0] = slate_frames + 1
if fps > 0 and slate_frames_box[0] > config.PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES:
    if slate_pace_start is None:
        slate_pace_start = time.perf_counter()
    paced_frames = slate_frames_box[0] - config.PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES
    due = slate_pace_start + (paced_frames / fps)
    delay = due - time.perf_counter()
    if delay > 0:
        self._stop.wait(min(delay, 0.05))
```

配置：`config.py:892-898`

```
PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES = 90  # 默认
```

- 前 90 帧**完全不节流**：绿屏 NV12 NVENC 编码 ≈ 几 ms/帧，全部
  bitstream 立即写入 video_proc.stdin。
- 写入到 video_proc 后，`setts=N*1502` 给这 90 帧打的 PTS 占
  `90/fps` 秒（fps=30 → 3.0 s；fps=60 → 1.5 s）。
- 同期 wallclock 只过去 ~100–200 ms（取决于 NVENC + Popen.stdin 写入），所以
  90 帧瞬间塞进了"视频未来 3 秒"的 PTS。
- 音频侧 `anullsrc + -re` 老老实实按 wallclock 出包，这 200 ms wallclock 只产生
  ~200 ms 的静音 AAC PTS。

→ **slate 阶段一开始视频 PTS 就比音频 PTS 提前约 `burst/fps − burst_wallclock`**。

burst 之后进入 paced 阶段：每帧 due ≈ `slate_pace_start + paced_frames/fps`，与
wallclock 步调一致，所以差距**不再扩大**但**也不会自动收敛**。

---

## 4. 切换瞬间发生了什么

`pipeline/pynv_stream.py:2357-2399`：

```python
if first_real_frame and slate_thread is not None:
    slate_stop.set()
    if slate_thread.is_alive():
        slate_thread.join(timeout=2.0)
    ...
try:
    ...
    bitstream = self._enc.Encode(app_frame, flags)
...
mux_stdin.write(bitstream)
if first_real_frame:
    self._real_video_started.set()
```

`_real_video_started` 被 audio loop 监听（`:1206-1230`）：

```python
if audio_source_ready and self._real_video_started.is_set():
    break
...
self._stop_proc(silence_proc, "slate audio silence")
...
real_cmd = [..., "-ss", f"{self.start_sec:.3f}", ..., "-i", str(cache_path or self.src), ...]
real_proc = subprocess.Popen(real_cmd, ...)
while not self._stop.is_set():
    data = real_proc.stdout.read(64 * 1024)
    conn.sendall(data)
```

注意：

- 音频 TCP socket **从头到尾是同一条连接**（`conn`），final mux 在 `-f aac -i tcp://...`
  里把它当一条连续 AAC 流。所以 silence + real 拼起来 ffmpeg 给统一的递增 PTS。
- 真实音频用 `-ss start_sec` 是选**源时间位置**，但**进入 TCP 后被 final mux 重打
  PTS**，时间起点等于"已经发了多少静音"= slate wallclock。
- 视频侧 bsf 拿到的 N（包序号）是连续的：slate_frames + real_frames。所以真实
  第 0 帧 PTS = `slate_frames * 1502 / 90000` 秒。

切换瞬间两侧时间：

```
video_pts_switch = slate_frames / fps                      (帧序号驱动)
audio_pts_switch ≈ slate_wallclock_elapsed                 (wallclock 驱动)
desync = video_pts_switch - audio_pts_switch
       = (burst/fps - burst_wallclock) + (paced_pts - paced_wallclock)
       ≈  burst/fps - burst_wallclock         (paced 阶段两者相等)
```

burst=90, fps=30, burst_wallclock≈0.1 s → **desync ≈ 2.9 s**
burst=90, fps=60, burst_wallclock≈0.1 s → **desync ≈ 1.4 s**

**结论：播放器看到的音频比视频晚 `burst/fps − burst_wallclock` 秒。**

---

## 5. TensorRT 比 CUDA 严重的物理原因

`pipeline/pynv_stream.py:2113`：

```python
self.matter.reset_state()
log.info("[PYNV][%d] RVM recurrent state reset at %.3fs; ...", ...)
```

`pipeline/matting.py:1205-1224`：

```python
def reset_state(self) -> None:
    self._rvm_rec = None
    self._rvm_rec_ort = None
    self._rvm_rec_sig = None
    self._rvm_io_sig = None
    self._rvm_io_outputs = {}
    self._rvm_io_downsample = None
    self._trt_static_states = {}      # <-- TRT 关键
    self._trt_static_outputs = {}     # <-- TRT 关键
    ...
```

`_run_rvm_static_trt_iobinding_from_dev`（`pipeline/matting.py:2015-2065`）拿到
空 `_trt_static_states` 时：

```python
states = self._trt_static_states.get(key)
if states is None:
    states = []
    for name in input_names[1:5]:
        state = ort.OrtValue.ortvalue_from_shape_and_type(
            self._rvm_initial_state_shape(name, batch, h, w),
            self.input_dtype, "cuda", 0
        )
        zeros = np.zeros(tuple(int(v) for v in state.shape()), dtype=self.input_dtype)
        self._copy_numpy_to_cuda_ortvalue(zeros, state)
        states.append(state)
    self._trt_static_states[key] = states
```

每次中段 seek，TRT 都要：

1. 重新分配 4 个递归状态 CUDA OrtValue（含 numpy zeros → CUDA H2D 拷贝）。
2. 重新走 io_binding bind_input / bind_ortvalue_input / bind_output 全套绑定。
3. 触发 TRT engine context 在新 binding 上的首次 `run_with_iobinding`，附带
   CUDA Graph capture（如果 `ONNX_TRT_CUDA_GRAPH_ENABLE=1`）。

实际表现（参见今天 handover 的 T2a/T2b 分布）：TRT 首推迟通常在 **1.5–3 s**，CUDA
EP 在 **0.3–0.8 s**。

→ T_slate（首真实帧到达前 wallclock）：
- CUDA：~0.5 s。burst=90 在这段时间内"也许都没打满"（90 帧需要 NVENC 一直跑），实测
  slate 视频可能只发了 30–60 帧。
- TRT：~2–3 s。burst=90 已经全部打完，paced 阶段又跑了 1.5–2 s。

| 模式 | T_slate | slate_frames | video_pts_switch | audio_pts_switch | desync |
|---|---:|---:|---:|---:|---:|
| CUDA, fps=30, T=0.5s | 0.5 | ~40 | 1.3 s | 0.5 s | **~0.8 s** |
| TRT, fps=30, T=2.5s | 2.5 | 90 + 72 = 162 | 5.4 s | 2.5 s | **~2.9 s** |
| TRT, fps=60, T=2.5s | 2.5 | 90 + 144 = 234 | 3.9 s | 2.5 s | **~1.4 s** |

这是用户感受到"TRT 模式更糟，CUDA 好点"的精确量化解释。**desync 上限被 burst/fps 钳住**，
TRT 模式恰好让 slate 跑得足够久，把上限打满。

注：`reset_state` 还顺带把 `_rvm_iobinding_failed`、`_cached_alpha_small` 等清掉，
还有 RVM 递归状态 4 帧 alpha 噪声（`pipeline/pynv_stream.py:2114-2117` 注释），这些
是画质问题，不直接计入 sync 偏移。

---

## 6. 垫片不能去除的硬约束

确认事实（按今天的回归过程）：

1. `summary/summary_20260523_NPLAYER_AUDIO_ONLY_REGRESSION_CN.md`：
   final mux 在 video_proc stdin 长时间无数据时 `Could not find codec parameters
   ... Video: hevc ..., none / unspecified size`，strict 播放器（nPlayer/Quest3
   AVProMobileVideo/SKYBOX libmpv）直接判 audio-only。

2. P2.A 系列把 `MUX_INTERMEDIATE_TS_PROBESIZE` 砍到 16384 后，FFmpeg 对中间 TS
   的 codec 参数容错窗口变得更小，越发依赖**video 流从第一刻起就有可解析的 VPS/SPS/PPS+slice**。
   slate 第一帧用 `FORCEIDR | OUTPUT_SPSPPS`（`pipeline/pynv_stream.py:2164-2165`）
   正是为此。

3. AAC cache 构建在 first-play / cache-miss 时耗 1–3 s，没有垫片音频静音占位，
   final mux 的 audio 输入会迟到，触发 mux 阻塞或 audio-only。

→ "去掉垫片"不是可选方案。必须在保留垫片的前提下消除 PTS 偏移。

---

## 7. 缓解方向详细说明

### 方向 A：把 burst 调小或关掉（最小代码改动）

**原理**：burst 是唯一让 video PTS 跑赢 wallclock 的来源；paced 阶段两侧时基天然
一致。如果 `PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES = 0`，slate 视频每一帧
都按 fps 节流，PTS 严格跟随 wallclock，与 audio anullsrc 完全同步。

**预期效果**：desync 直接 → 0（误差只剩 ffmpeg `-re` 与 Python 节流的几十 ms 抖动）。

**代价分析**（看代码即可定位）：

- burst 当初的目的是**让 final mux 尽快拿到足够 HEVC 包来识别 codec 参数**，避免
  P2.A.1 把 probesize 砍到 16384 后再次出现 `Could not find codec parameters`。
- 关掉 burst 后，第一秒只产生 fps 个 HEVC 包（30 帧/s × ~几十 KB ≈ 1–3 MB），仍远
  超 16384 字节阈值。**风险点不在 probesize**，而在 final mux 内部对 PAT/PMT/PCR
  的最小观察窗口：`pat_period=0.1`、`pcr_period=20` 都已经设到最快档，单帧足以
  触发 PAT 输出。
- 真实风险是**首 chunk 延迟**：T1_write 从 ~170 ms 上升到 ~33 ms × 90 = 3 s 吗？
  不会——T1_write 由 slate **第一帧**触发（line 2183 `self._mark_first_write()`），
  与 burst 长度无关。受影响的只是后续填充密度。
- 推荐第一步：把默认改成 `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES=0`，跑
  Quest3 / nPlayer / SKYBOX 三家。如果 final mux `T2b_final_first_stderr` 上涨
  超过 300 ms 或出现 `Could not find codec parameters` 退化，再试 `burst=8` /
  `burst=16` 这种小值（fps=30 → 0.27 s / 0.53 s desync，依然比现在好一个数量级）。

**操作**：仅改 `config.py:895-898` 默认；`run_server.bat` 可加 `set
PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES=0` 做无重打包对照。无代码逻辑改动。

### 方向 B：slate 视频改用 wallclock 时基（彻底解耦帧序号）

**原理**：让 slate 阶段的 `setts` 改成基于真实 wallclock，或者让 video_proc 直接
用 `-use_wallclock_as_timestamps 1` 替代 `setts=N*1502`。

**代码位置**：
- `pipeline/pynv_stream.py:1354`（pipe_ts video mux）
- `pipeline/pynv_stream.py:1490`（slate pipe_ts video mux）

两处 bsf 都是 `setts=time_base=1/90000:pts=N*1502:dts=N*1502`。

**两种实现思路**：

B1. **video_proc 输入端改时基**：把 `-framerate {fps}` 之外加上
`-use_wallclock_as_timestamps 1`（FFmpeg 对 `-f hevc` 是否生效需实测；对 raw HEVC
通常不行，因为没有内建容器时间戳，必须靠 bsf 重写）。

B2. **bsf 切换时机**：slate 阶段用 `setpts=PTS-STARTPTS` 之类的 filter（注意 bsf 是
比特流级，setpts 是 filter 级，需要走 `-filter:v`，这会触发 video_proc 解码再
编码——破坏 `-c:v copy`，不可行）。

B3. **不改 bsf，改 slate 节流策略**（与方向 A 等价但代码不同）：直接删掉 burst
分支，让每一帧都通过节流判断（即 `if fps > 0: ... due = slate_pace_start +
slate_frames/fps ...`），且 `slate_pace_start` 在循环开始时即刻赋值。这与方向 A
等效，唯一差异是 burst=0 走配置路径，B3 是写死。

**结论**：方向 B 在不破坏 `-c:v copy` 的前提下没有比方向 A 更优的实现；推荐先走 A。

### 方向 C：在切换时用 `-itsoffset` 把音频前推

**原理**：保留 burst 的 mux 启动收益，但在 audio loop 启 `real_proc` 时给真实
AAC 输入加 `-itsoffset -<desync>`，让真实音频 PTS 倒回 desync 秒，从而在 final
mux 那里与 video PTS 对齐。

**关键问题**：
- desync 不是常数：它取决于 slate 跑了多久、burst 是否打满。需要在 audio loop
  里**实时读取** `slate_frames_box[0]` 与 `slate_start_wallclock`，算出 desync，再
  传给 `real_cmd`。
- 但 `real_cmd` 启动的时刻就是切换点，那时 `slate_frames_box[0]` 已稳定，可读。
  代码改动点：
  - `pipeline/pynv_stream.py:1235-1296` 构造 `real_cmd` 前插入：
    ```python
    desync_sec = max(0.0, slate_frames_count / fps - (time.perf_counter() - slate_start_t))
    itsoffset_args = ["-itsoffset", f"-{desync_sec:.3f}"] if desync_sec > 0.001 else []
    ```
  - 把 `itsoffset_args` 放在 `-i str(cache_path)` 之前。
- **风险**：
  - `-itsoffset` 配合 `-c:a copy` 时只改 demux 起点 PTS，不会真正裁掉前面的样本。
    如果 desync=3s，audio 的前 3s 实际上是被裁掉的——这恰好是我们想要的（用户期望
    seek 到 T 时音频是 T 而非 T+3）。但实现细节要测：cache 路径走 `-ss start_sec`
    本来就裁过；direct 路径走 `-ss start_sec` 也是裁过；再加 `-itsoffset` 是叠加，
    要小心方向。
  - `-itsoffset` 是 input option，必须放在对应 `-i` 之前。
- 这个方向**会动音频源切换的命令**，但不动 mux 命令、不动 bsf、不动 burst 行为，
  对 final mux 的稳定性最友好。

**推荐顺位**：作为方向 A 的兜底——如果方向 A 在某家播放器上引发 codec 识别退化，
就保留 burst=90 + 方向 C 校正。

### 方向 D：缩短 TRT 中段 seek 的冷启动

**原理**：TRT desync 上限是 burst/fps（固定），但中段 seek 冷启动让 slate 跑满了
这个上限。如果让 TRT 首推延迟降到 CUDA 量级，slate 还没跑完 burst 就被打断，
desync 也跟着下降。

**代码切入点**：

D1. `pipeline/pynv_stream.py:2113` 之后、`:2265` 主循环之前，插入一个 dummy
inference 预绑定 TRT engine：
```python
self.matter.reset_state()
# 预热：用零帧驱动 TRT engine 重新 bind I/O，不污染真实第一帧的 alpha
dummy_h, dummy_w = MATTING_INPUT_SIZE, MATTING_INPUT_SIZE
self.matter.prewarm_for_shape(batch=1 or 2, h=dummy_h, w=dummy_w)
```
其中 `prewarm_for_shape` 是新方法，等价于跑一次 `_run_rvm_static_trt_iobinding_from_dev`
传零张量，**但跑完后再 `reset_state` 一次清掉脏的递归状态**，避免污染真实第一帧。

不过看 `pipeline/matting.py:1235-1256` 的现有 `warmup` 已经是 numpy 零帧驱动 `alpha`
方法，可以复用：在 worker 启动时（reset_state 之后）调用 `self.matter.warmup(1)`
+ 再次 `reset_state`。

D2. 现有启动时 warmup 已覆盖了模型加载和 engine 编译；问题不在编译，在
**reset_state 后 OrtValue 重新分配与 io_binding 重绑**。这两个 cost 在每次 seek
都重来。可以考虑：

- `_trt_static_states` 不在 `reset_state` 里清空，而是**只 zero-fill** 已有的
  OrtValue（保留分配，避免重分配）。这是 `pipeline/matting.py:1213-1214` 的改动：
  ```python
  # 现状
  self._trt_static_states = {}
  self._trt_static_outputs = {}
  # 改为：
  for states in self._trt_static_states.values():
      for state in states:
          zeros = np.zeros(tuple(int(v) for v in state.shape()), dtype=self.input_dtype)
          self._copy_numpy_to_cuda_ortvalue(zeros, state)
  # _trt_static_outputs 输出 buffer 无状态语义，可不动
  ```
- 这样 seek 不再触发重分配，只剩一次 H2D zero。`_run_rvm_static_trt_iobinding_from_dev`
  的 `states = self._trt_static_states.get(key)` 直接命中。
- 风险：跨视频分辨率切换时旧 key 的 states 不会被清，需要在 worker 启动时按
  `(batch, h, w)` 检查并清掉其他 key。

D2 是结构性优化，D1 是补丁性预热；建议先验 D1 是否能在测量上把 T_slate 拉回
CUDA 量级。

### 方向 E：让 slate 阶段视频走 wallclock 节流但不打 burst（A 的等价 + 显式工程化）

把 burst 改成"先打**视频流头**（VPS/SPS/PPS + 1 个 IDR slice）然后立刻切节流"。
即 burst 改成"打 1 帧就停"，目的明确为给 final mux 一份 codec 参数样本。

代码改动：
```python
# pipeline/pynv_stream.py:2188-2196
slate_frames_box[0] = slate_frames + 1
if fps > 0 and slate_frames_box[0] > 1:           # 原本是 > BURST
    if slate_pace_start is None:
        slate_pace_start = time.perf_counter()
    paced_frames = slate_frames_box[0] - 1         # 原本是 - BURST
    due = slate_pace_start + (paced_frames / fps)
    delay = due - time.perf_counter()
    if delay > 0:
        self._stop.wait(min(delay, 0.05))
```

这样：第一帧（含 SPS/PPS）立刻发出 → final mux 可识别 codec → 后续每帧都按 wallclock。
desync ≈ 0（只有 0–33 ms 抖动）。

这本质上是方向 A 的子集（burst=1），优点是把"为什么留 1 帧 burst"显式写进代码，
不依赖配置默认。

---

## 8. 推荐的执行顺序

1. **先做方向 A**：环境变量 `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE_BURST_FRAMES=0`，
   不动代码，用 nPlayer / Quest3 DeoVR / SKYBOX 各测一次。记录：
   - `T1_write`、`T2a_video`、`T2b_final`、`T3a/b/c`、`T4_reader`（看 first-chunk
     是否退化）；
   - mux stderr 是否再现 `Could not find codec parameters` / `unspecified size`；
   - 实际听感+画面同步（>30 s 段，中段 seek 5 次取平均）。

2. **若 A 在某家退化**：把默认改回 90，转方向 E（强制 `burst=1`），重测。

3. **若 E 仍退化**：保留 burst=90，落地方向 C 的 `-itsoffset` 校正逻辑。这条改动
   只在 `_slate_audio_server_loop` 启动 `real_proc` 时插入数行计算 + ffmpeg args，
   对其他路径零影响。

4. **不论 1/2/3 结果如何**，独立推进方向 D2（`reset_state` 不再清空 TRT
   `_trt_static_states`，改为 zero-fill），从根本上把中段 seek 的 TRT 冷启动拉
   平到 CUDA 水准。**这是与同步问题正交的性能优化**，同时也能把 first-chunk 总
   时延从 ~2 s 降到 ~1 s。

---

## 9. 不应做的事

- **不要回到 `setts` 单段 mux**（已被 `2026-05-23 single-stage setts rollback`
  否决，nPlayer 字节率异常）。
- **不要在 video bsf 上加 `hevc_metadata=aud=insert`**（`NPLAYER_AUDIO_ONLY_REGRESSION`
  根因）。
- **不要在 raw HEVC stdin 上设 `-probesize 32 -analyzeduration 0`**（同上）。
- **不要试图通过加大 final mux 的 `-max_interleave_delta` 解决 sync**（A8.2 单
  点测试无效）。
- **不要在 audio 侧关 `-re`**：会让静音瞬间灌满 final mux 的 audio 输入 buffer，
  反而把 audio PTS 提前，制造反向 desync。

---

## 10. 待验证假设清单

- [ ] burst=0 时 final mux 是否仍能在 P2.A.1 (probesize 16384) 下稳定识别 codec？
- [ ] burst=0 时 T2b_final 是否上涨 < 300 ms？
- [ ] burst=1 时各家播放器是否都满足"视频模式"且 sync 在 ±100 ms 内？
- [ ] `-itsoffset -<desync>` 配合 `-c:a copy` 在 cache 路径 / direct 路径是否
  正确裁掉前段样本而非引入静默？
- [ ] `reset_state` 改为 zero-fill 后，中段 seek 首推迟从 2–3 s 降到多少？是否
  影响连续播放时 RVM alpha 收敛？

代码索引：
- 节流 / burst：`pipeline/pynv_stream.py:2188-2196`
- audio 切换：`pipeline/pynv_stream.py:1230-1298`
- video bsf：`pipeline/pynv_stream.py:1354`、`:1490`
- TRT 状态清理：`pipeline/matting.py:1205-1224`
- TRT iobinding：`pipeline/matting.py:2015-2065`
- 配置默认：`config.py:875-898`、`:986-1010`
