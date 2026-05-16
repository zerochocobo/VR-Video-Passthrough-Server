# MKV PyNv SimpleDecoder 卡死问题说明

日期：2026-05-16

## 问题概述

在实时 alpha 直通模式下测试 MKV 视频时，服务端可以成功启动 PyNv 解码、抠像、编码和 MPEG-TS live 输出，客户端也能收到一段数据。但当客户端停止播放或 live session 进入关闭流程后，`pynv-worker` 线程偶发卡死在 PyNvVideoCodec 的 `SimpleDecoder[index]` 调用中，native 解码调用不返回。

卡死后 Python 侧无法安全终止该线程，导致当前进程内 PyNv/CUDA/NVDEC/NVENC 状态被污染，表现为显存无法完全释放、后续 alpha 播放可能卡顿或红色 alpha 通道闪烁。当前日志已经能明确定位到卡死栈。

## 测试场景

- 输入文件：`test_mkv_8k.mkv`
- 容器：MKV
- 视频：HEVC，8192x4096，约 59.94 fps
- 实时输出模式：alpha passthrough
- 客户端：nPlayer 3.0
- 请求时间点：`t=240.00s`，即 00:04:00
- 服务端输出：MPEG-TS live stream
- PyNv 路径：`SimpleDecoder[index]` 随机帧访问

## 关键日志证据

客户端请求 live 流时没有 HTTP Range：

```text
passthrough_live[1] request headers: ua='nPlayer/3.0' accept='*/*' range=None time_seek=None
passthrough_live[1] start: test_mkv_8k.mkv @ 240.00s
```

这说明本次 live 路径里，nPlayer 没有像普通文件播放那样主动请求 MKV 文件尾部或做 Range 探测。

PyNv 成功创建 decoder/encoder，并开始输出：

```text
[PYNV][1] worker start src=test_mkv_8k.mkv start=240.000
[PYNV][1] source meta: codec=hevc ... size=8192x4096 source_fps=59.940 output_fps=30.000
[PYNV][1] encoder created 8192x4096 fps=30.000
[PYNV][1] first real video bitstream written
passthrough_live[1] live cache subscribe ... closed=False subscribers=1
passthrough_live[1] first chunk: len=65536 sent=65536 stream_bytes=65536
```

播放过程中帧率进入稳定状态：

```text
[PYNV][1] frame 420/60241 fps=30.27 interval_fps=34.95
[PYNV][1] frame 600/60241 fps=31.56 interval_fps=35.05
[PYNV][1] frame 750/60241 fps=32.19 interval_fps=35.10
```

客户端关闭或停止订阅后，服务端进入关闭流程：

```text
passthrough_live[1] live cache unsubscribe ... subscribers=0
passthrough_live[1] finally begin: sent=93490708 stream_bytes=93490708 frames=756
```

之后 live session 因 TTL 过期关闭，但 worker 线程没有退出：

```text
[PYNV][1] iter finally bytes=113533952 frames=917
[PYNV][1] close begin bytes=113533952 frames=917
[PYNV][1] close: waiting thread name=pynv-worker
[PYNV][1] close: worker still alive after 3.0s, stopping decoder
[PYNV][1] decoder stop thread begin
[PYNV][1] decoder stop thread done
[PYNV][1] close: waiting worker after decoder stop 1.5s
[PYNV][1] close: thread still alive name=pynv-worker
```

最终捕获到 stuck worker 栈：

```text
stuck thread stack name=pynv-worker
  File "pipeline\pynv_stream.py", line 1783, in _worker_loop
    frame = self._dec.frame_at(src_idx)
  File "pipeline\pynv_io.py", line 202, in frame_at
    frame = self._decoder[index]
  File ".venv\Lib\site-packages\PyNvVideoCodec\decoders\SimpleDecoder.py", line 86, in __getitem__
    return self.simple_decoder[key]
```

服务端随后将 PyNv runtime 标记为 tainted：

```text
PyNv runtime marked tainted because worker did not stop; restart the server before continuing alpha passthrough
```

## 显存现象

本次 VRAM 日志显示：

```text
worker_start: used=1619.4/16310.4MiB
encoder_created: used=2056.9/16310.4MiB
first_real_video_bitstream: used=4001.1/16310.4MiB
close_begin: used=4145.1/16310.4MiB
decoder_stop_joined: used=3611.6/16310.4MiB
thread_still_alive:pynv-worker: used=3611.6/16310.4MiB
close_done: used=3611.6/16310.4MiB
```

关闭后显存从峰值回落了一部分，但仍比 worker 启动时高约 2GB。结合 stuck stack，当前判断是 native decoder 调用仍占住线程和部分 GPU/native 资源，Python 侧 close 流程无法完整释放。

## 已排除或基本确认的点

1. 这次不是“客户端强制读取 MKV 文件尾”直接导致的问题。

   本次 live 请求日志里 `range=None`，客户端没有对 live URL 发 Range 请求。

2. 这次不是服务端过早关闭 live session 导致的首包失败。

   之前曾给 MKV 添加“无订阅者立即关闭”逻辑，导致首个 64KB 后立刻关闭，nPlayer 逐个跳下一个视频。该逻辑已移除。本次日志显示服务端持续输出到 917 帧，客户端收到约 93MB 后才 unsubscribe。

3. 播放阶段性能不是主要矛盾。

   首帧较慢，但稳定输出后 `interval_fps` 约 35fps，能满足 30fps 目标。真正的问题发生在关闭阶段。

4. 卡死位置集中在 PyNvVideoCodec `SimpleDecoder[index]`。

   栈显示 worker 阻塞在 `self._decoder[index]`，即 SimpleDecoder 的随机帧索引访问，而不是 Python 层的 mux、queue、HTTP response 或 ffmpeg pipe 写入。

## 当前技术判断

MKV 实时 alpha 路径当前使用 PyNv `SimpleDecoder[index]` 做随机帧访问，以支持按时间点 seek 后输出目标帧序列。对于 MP4，该路径通常能正常关闭；但对于 MKV，尤其是大分辨率、大文件、复杂索引或网络盘来源的 MKV，`SimpleDecoder[index]` 可能在关闭或抢占阶段卡在 native 解码调用里。

一旦卡住，Python 线程无法被强杀。即使服务端调用 decoder stop，也只能释放一部分资源，无法保证 native 调用返回。因此继续在同一进程内播放其他 alpha 视频存在风险。

## 对用户可见的影响

- MKV live 可能能播放一段，但关闭后显存不完全释放。
- 后续播放 MP4 或其他视频时，可能出现红色 alpha 通道闪烁、卡顿、409、超时或显存持续增长。
- 如果多次触发，服务端进程可能逐渐进入不可恢复状态，需要重启服务端进程。

## 待专家研判的问题

1. PyNvVideoCodec 的 `SimpleDecoder[index]` 对 MKV 随机帧访问是否存在已知卡死风险？

2. 对 MKV 是否应避免 `SimpleDecoder[index]`，改用顺序解码路径？

   例如：
   - 先 seek 到接近目标时间点；
   - 使用 sequential decode 逐帧前进；
   - 避免每帧通过 `decoder[index]` 做随机访问。

3. PyNvVideoCodec 是否提供可中断的 decode/seek API？

   当前 Python 线程卡在 native 层后无法安全退出。如果没有可中断 API，in-process MKV live 可能天然不安全。

4. 是否应该将 MKV PyNv 解码隔离到子进程？

   子进程方案的优势是 native 卡死时可以 kill 子进程，主服务进程不被污染；缺点是需要设计进程间传输、启动延迟、错误恢复和显存占用策略。

5. 对 MKV 是否应强制 remux 成 MP4 后再允许实时 alpha？

   如果 PyNv 对 MKV 的随机访问不稳定，而 MP4 路径稳定，则产品策略上可以阻止 MKV live alpha，提示用户转封装。

## 建议的下一步验证

1. 准备三类文件对比：

   - 原始 MKV
   - Cues/SeekHead 已修复到头部的 MKV
   - remux 后 MP4

2. 对每类文件测试：

   - 是否能稳定首帧输出；
   - 是否能持续 30fps；
   - 停止播放后 worker 是否退出；
   - `SimpleDecoder[index]` 是否卡死；
   - close 后 VRAM 是否回落到接近启动前水平。

3. 单独验证 PyNv sequential decode 对 MKV 的关闭行为。

4. 如果 sequential decode 仍不可控，优先设计 MKV 子进程隔离或产品层禁用 MKV live alpha。

---

## 代码研判与解决办法（2026-05-16 补充）

### 一、问题机理（严格基于现有代码 + 日志）

#### 1. 卡死位置的确认

`pipeline/pynv_stream.py` worker 循环的结构：

```python
for i in range(target):
    if self._stop.is_set(): break          # 帧间检查
    frame = self._dec.frame_at(src_idx)    # 一旦进入此处就锁死，不可中断
    ...
```

`frame_at` → `pipeline/pynv_io.py:202` 的 `self._decoder[index]` → PyNvVideoCodec C++ 层 demux + seek + decode 串行同步调用。这一步对 Python 没有暴露任何中断点。

#### 2. close 流程为什么救不回来

`pipeline/pynv_stream.py:2096-2106` 走的是"另起一个线程调 `dec.stop()`，主线程 `join(timeout)`"的策略：

```python
def _stop_decoder() -> None:
    dec.stop()                      # 立刻返回
dec_thread = threading.Thread(...)
dec_thread.start()
dec_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
```

但 `pipeline/pynv_io.py:207-215` 注释已经写得很清楚：

> PyNvVideoCodec 2.1.0 exposes SimpleDecoder.stop() in Python, but the wrapped native object does not implement stop().

也就是说 **`dec.stop()` 是个空壳**。日志里 `decoder stop thread begin → decoder stop thread done` 之间几乎不耗时就是证明：Python 层认为已经"停了"，但 native 那边继续在 `__getitem__` 里跑。

#### 3. 为什么偏偏 MKV 容易触发

MKV 在 PyNv 底层走的是 ffmpeg / libavformat demuxer。`SimpleDecoder[index]` 做随机访问要：

- 在 Matroska SeekHead / Cues 里找最近 keyframe；
- demux 到该位置 → 喂 NVDEC → 顺序解码到目标 index 后丢弃中间帧。

对 8K HEVC + 大文件 + 复杂 GOP 的 MKV，这条路径本身耗时就是秒级（前几次 handover 提到的 "decoder_created 用了约 63 秒" 就是同一现象的另一面）。它不是真正的"死锁"，而是**单次 native 调用执行时间无界**。MP4 的 moov 在头部时这条路径稳定毫秒级，所以从来没事。

#### 4. 显存为何只半释放

`close_begin: used=4145MB → close_done: used=3611MB`，回收约 534MB。worker 还持有：

- 当前帧 NV12 plane（CAI owner = `frame` Python 对象）；
- 解码线程内部的 NVDEC 输出环形缓冲；
- libavformat 内部分配的 buffer。

这些只能等 native 调用返回才能回收，否则只能等进程退出。

---

### 二、解决办法分层评估

#### 方案 A：用 `ThreadedDecoder` 顺序拉替换 `SimpleDecoder[index]`（首选）

**根据**：5/15 已经用 `tools/pynv_threaded_mapping_probe.py` 全矩阵验证 ThreadedDecoder + `start_frame` + `get_batch_frames(N)` 的内容映射稳定，包括 8K。Stage 4 的 native crash 是**跨线程传递帧对象**违反生命周期规则，不是 ThreadedDecoder 本身的问题。

**关键约束**：保持 worker 单线程（不要顺势做 Stage 4 的多线程流水线），改动只在 decoder 接口层：

- 构造时一次性 `ThreadedDecoder(file, buffer_size=32, start_frame=src_idx_at_start)`；
- worker 循环里 `get_batch_frames(B)`（B 取 1~4），逐帧消费完再拉下一批；
- CFR 跳帧逻辑：在 Python 端按 `cfr_source_index(out_idx, source_fps, fps)` 序列匹配，对不需要的源帧丢弃。

**为什么能缓解卡死**：

1. 没有任何 `[index]` 随机访问；demuxer 顺序拉，不再每帧重新 seek 解码到目标 index；
2. `get_batch_frames(1)` 单次的 native 工作量是"取出 prefetch buffer 中已 ready 的一帧"，比 `[index]` 短得多，碰到 MKV 病态结构卡死的窗口大幅收窄；
3. close 时即便某一次 `get_batch_frames` 慢，下一帧间隔还有机会被 `_stop` 中断，窗口短。

**风险**：ThreadedDecoder 的 `end()` 也是 native，MKV 病态情况下仍可能卡。但比 `[index]` 的概率低得多。**Stage 2 已铺好基础设施，这是落地代价最小的方案**。

#### 方案 B：close 阶段的硬约束 watchdog（必加）

无论是否换 decoder，close 路径都要加防线：

1. `close()` 里 worker `join(timeout=K)` 失败后，**不要再依赖 `dec.stop()`**（已证实是空壳），直接：
   - 把 `self._dec` 引用置 None；
   - 把 `self._enc` 引用置 None；
   - **不再二次 join**，立刻返回，让 daemon 线程随进程死；
   - 标 tainted（已实现）。
2. 不要让"等 worker 退"阻塞 HTTP 请求关闭。当前 `iter_bytes` finally 调 `close()`，如果 close 慢，response 也慢，下个请求体验受影响。

**收益**：MKV 真卡死时，HTTP 层和后续无关请求至少不被牵连。

#### 方案 C：preflight 提前甄别病态 MKV（强烈推荐）

`PyNvPassthroughStream.preflight`（`pipeline/pynv_stream.py:195-229`）已经在做 PyNv 解码器初始化测试。在已有 preflight 里加一个**短超时的首帧 probe**：

```text
在子线程里调一次 dec.frame_at(start_idx)
主线程 join(timeout=8s)
若超时 → preflight 失败，refuse 502/503 + 日志，
        并在 utils/video_metadata.py 路由策略里
        对该文件的 PyNv 资格永久打 X（可加缓存）
```

落到 `utils/video_metadata.py` 的 `should_use_pynv()` 决策里。一次失败的 MKV 自动降级走 `pipeline/stream.py` 的 ffmpeg 子进程路径——子进程是真的能 SIGTERM 中断的。

#### 方案 D：子进程隔离 PyNv 解码（长期最干净，但工作量大）

把 NVDEC + matting + NVENC 整段塞进子进程，主服务进程只管 mux + HTTP。MKV native 卡死 → `Process.kill()` 一刀切。

**代价**：

- IPC：8K NV12 单帧 50MB，大概率走 GPU 共享（CUDA IPC handle）而不是 shm，复杂；
- 启动延迟：每次 live 起一个子进程 + ORT cold start（已经被坑过）；
- 跨进程 GPU 资源（NVDEC / NVENC session）数限制。

建议**不做**，除非方案 A+B+C 仍频繁出现 MKV 卡死。

#### 方案 E：MKV 走 ffmpeg 子进程路径（兜底）

`pipeline/stream.py` 里的老 FFmpeg subprocess 路径仍然在。MKV 在 `should_use_pynv()` 里直接判 false，强制走 FFmpeg：

- ffmpeg 子进程可以 SIGTERM 干净退出；
- 8K 性能更差，但**不会污染主进程 PyNv state**；
- 用户能看到画面，VRAM 不会泄漏。

代价：MKV 用户失去 PyNv 性能。但从产品语义看，"能播 + 能关" > "高帧率"。

#### 方案 F：MKV 自动 remux 到 MP4（避坑神器）

第一次见 MKV 文件时后台跑 `ffmpeg -i in.mkv -c copy out.mp4`：

- 8K HEVC 视频流不重编码，仅换容器，速度等于 IO 速度（数 GB/min）；
- 缓存到 `runtime_cache/remux/<sha1>.mp4`；
- 后续走 PyNv MP4 路径。

**优点**：从根上避开 MKV demuxer。
**缺点**：首次 remux 需要等待，磁盘多用一倍空间。

---

### 三、判断与落地建议

按 **风险 / 收益 / 改动量** 三轴权衡：

| 方案 | 解决根因 | 兼容性 | 开发量 | 优先级 |
|---|---|---|---|---|
| **A. ThreadedDecoder 替换 `[index]`** | ✅ 大幅缩短 native 调用窗口 | ✅ Stage 2 已验证 | 中（worker 循环 + CFR 改造） | **P0** |
| **B. close watchdog 不再 join** | ❌ 治标 | ✅ | 极小 | **P0** |
| **C. preflight 首帧超时探测 + 路由降级** | ✅ 隔离病态文件 | ✅ | 小 | **P0** |
| E. MKV 路由到 FFmpeg 兜底 | ✅ 完全规避 | ✅ | 小 | P1（A 落地前的临时开关） |
| F. MKV 自动 remux 缓存 | ✅ 治本 | ✅ | 中（缓存管理） | P1 |
| D. 子进程隔离 | ✅ 治本 | ⚠️ IPC / CUDA 复杂 | 大 | P2 |

#### 推荐路径

**第 1 步（今天可发）**：方案 B + 方案 E 临时开关

- 给 `close()` 加 watchdog：worker 二次 join 失败立刻返回 + 标 tainted，**不再二次 join**；
- `utils/video_metadata.py` 的 `should_use_pynv()` 加一个 `if container in ("matroska", "webm"): return False`（或用 env `PT_PYNV_ALLOW_MKV=0` 控制）；
- 上线后 MKV 退到 FFmpeg 路径，PyNv 不会再被 MKV 污染；MP4 不受影响。

**第 2 步（本周可发）**：方案 C

- `preflight` 加 8 秒首帧 probe，失败的文件落到内存 set + `runtime_cache/pynv_blocklist.json`，永久走 ffmpeg；
- 这一步落定后可以试着把第 1 步里的 MKV 黑名单放开（让 MKV 也走 PyNv，preflight 自然过滤掉病态的）。

**第 3 步（按 8K 优化计划同步推进）**：方案 A

- 借 `IMPL_PLAN_8K_40FPS_20260515` Phase 2 / 4 的 ThreadedDecoder 切换；
- **单线程串行先上**，回避 Stage 4 跨线程帧生命周期的坑；
- 每个 `get_batch_frames(1)` 之间检查 `_stop`，让 close 时间从"无界"压回"≤ 一次顺序解码时间"。

**不必做**：方案 D（子进程）。F（remux）作为 A 落地不顺利时的备胎。

---

### 四、实现方案 A 时必须遵守的 native 行为约束

来自 5/15 Stage 2 / Stage 4 已经踩过的坑：

1. ThreadedDecoder 返回的 `frame` **只在下一次 `get_batch_frames` 之前有效**。当前帧的 `frame.cuda()` 拿到的 NV12 plane CAI 也只在这个窗口里活。worker 必须**在拿到下一批之前完整完成 composite + 入 NVENC**。
2. 不要用 `frame.getPTS()` 做身份；用 `start_frame + 已消费序号` 做身份。`test_8k_2.mp4` 的 PTS 与内容偏移约一个源帧（2002 ticks，首帧 3003）。
3. ThreadedDecoder 的 `end()` 和 `__del__` 也是 native，close 时也要走方案 B 的 watchdog 路线，不要无限期 join。
4. **不能跨 worker 线程传递 `frame` 对象**（Stage 4 native crash 的根因）。本方案保持单线程串行就规避了这一点。

---

### 五、与现有工作的衔接

- 方案 A 与 `prompt/IMPL_PLAN_8K_40FPS_20260515` Phase 2-4 完全同向，可以合并推进；
- 方案 B 与 5/15 "Live PyNv 8K Startup Timeout" + "PyNv Stuck Worker Taint Guard" 已落地的 close 改造一脉相承，只是把 "已 join 失败 → 再 join" 改成 "已 join 失败 → 直接返回 + tainted"；
- 方案 C 利用现有 `preflight` + `should_use_pynv()` + `runtime_cache/` 三处已存在的基础设施，无新概念；
- 方案 E 复用已存在但被边缘化的 `pipeline/stream.py` FFmpeg 路径，路由开关即可。

