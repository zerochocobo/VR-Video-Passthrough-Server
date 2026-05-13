这份报告非常详细，抓住了导致 FFmpeg 管道堵塞的核心矛盾。处理 8K S3D 这种极高规格的 VR 视频时，系统的 I/O 调度和缓冲机制本来就处于高压状态，而在实时流中混流（Muxing）裸流视频和文件音频，踩中 FFmpeg 交织死锁（Interleaving Deadlock）的概率非常高。

你遇到的现象——只输出 11-13 KB（基本上是 PAT、PMT、SDT 和非常少量的 PES 头部），然后 FFmpeg 的 `stdout` 彻底卡死，但前面的编码器仍在以 30fps 正常吐出数据——这是典型的**时间戳对齐与交织缓冲溢出**引发的阻塞。

以下是针对报告中几个核心问题（Specific questions）的深度分析和解决策略。

### 1. 为什么会在生产环境中卡死，而 8 秒的受控测试却没有？

**根本原因在于 I/O 阻塞特性的差异：**
在受控测试（`tools/mpegts_audio_pts_probe.py`）中，HEVC 数据来自一个静态的 8 秒文件。此时，FFmpeg 从 `stdin` 读取视频的速度和从硬盘读取 MP4 音频的速度都极快，它可以在几百毫秒内将这两路流塞入内存，迅速计算并生成对齐的时间戳，完成交织。
但在生产环境的实时流（Live）中：

* **视频源 (`stdin`)：** 是被 PyNvVideoCodec 严格以 30fps 的物理时间速率（或受限于 GPU 处理速度）“滴漏”进来的。
* **音频源 (`<source>.mp4`)：** FFmpeg 的音频解复用器会试图以极高的速度（磁盘 I/O 速度）读取音频流。

此时，FFmpeg 的交织器（Interleaver）收到了大量的、未来的音频包，但迟迟等不到对应的视频包（因为视频是实时的）。为了保证 MPEG-TS 的规范，FFmpeg 会将音频包缓冲起来等待视频。当缓冲队列达到上限（尤其你还设置了激进的 `-max_interleave_delta 0`），FFmpeg 就会陷入死锁：它不再从 `stdin` 读取数据（导致 Python 端管道可能最终也会写阻塞），也不往 `stdout` 写数据。

### 2. 针对 `-use_wallclock_as_timestamps` 和非单调 DTS 的问题

`-use_wallclock_as_timestamps 1` 确实赋予了裸流时间戳，打破了上述的死锁，让包得以流转。但这对于 CFR（固定帧率）的视频是灾难性的，因为系统的真实时钟存在微小的抖动，导致生成的 DTS 不均匀，MPEG-TS 封装对此极为敏感，最终导致 MoonVR / VLC 播放器丢帧或画面卡死。

**解决纯裸流打时间戳的问题，`setts` 是正确的方向（解答你的第 3 个问题）：**
你提到的比特流过滤器（BSF）非常合适。对于严格 30fps 的输入，你可以强制覆写 PTS 和 DTS，使其严格单调递增，完全无视物理送入的速度。

建议在视频输出前应用此过滤器：
`-bsf:v "setts=pts=N/(30*TB):dts=N/(30*TB)"`

### 3. 破局与参数优化方案（解答第 5 个问题）

要解决这个 Live 场景下的混流问题，你需要同时处理**时间戳生成**和**读取速率控制**。请尝试在生产命令中做以下调整：

* **扩大输入队列的缓冲（防止管道阻塞）：** 在 `-i -` 和音频 `-i` 之前分别加上 `-thread_queue_size 1024`。
* **限制音频读取速率（极其重要）：** 在音频输入文件前加上 `-re`（以本地帧频读取）或 `-readrate 1`。这能强制 FFmpeg 不要一口气吞掉整个文件的音频，而是和视频一样以 1x 的速度读取，从而防止交织器瞬间被音频包塞满而锁死。
* **放宽交织条件：** 移除 `-max_interleave_delta 0`，让 FFmpeg 使用默认的交织逻辑，或者显式设置为 `500M`。`0` 过于苛刻，在缺少可靠时间戳的实时源之间混流几乎必死。
* **处理 Seek 偏移：** （这点在报告的命令中未体现，但至关重要）由于你的架构响应 `?t=<start_seconds>`，你的视频推流是从 `<start_seconds>` 开始的。但如果 FFmpeg 命令行中的音频输入没有使用 `-ss <start_seconds>`，音频会从 0 秒开始。这会导致音视频 PTS 相差巨大，直接触发交织死锁。

**优化后的测试命令参考：**

```bat
ffmpeg -hide_banner -loglevel warning ^
  -thread_queue_size 1024 -f hevc -framerate 30 -i - ^
  -thread_queue_size 1024 -readrate 1 -i "<source>.mp4" ^
  -map 0:v:0 -map 1:a:0? ^
  -c:a aac -b:a 192k ^
  -c:v copy ^
  -bsf:v "setts=pts=N/(30*TB):dts=N/(30*TB)" ^
  -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 ^
  -flush_packets 1 ^
  -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.5 -pcr_period 20 ^
  -f mpegts -

```

*(注意：如果启用了 Seek，必须在音频 `-i` 前加上 `-ss <seconds>`)*

### 4. 关于架构层面的终极解法（解答第 4 个问题）

你提到：“Would a timestamped intermediate container be the correct architecture?”

**答案是肯定的，且这是最健壮的方案。**

目前的痛点全都在于 `raw HEVC` 缺乏容器级时间戳。如果你在 Python 端将 GPU 取回的 NALU 封装成一个轻量级的中间格式（比如 **NUT** 格式，或者非常简单的附带时间戳的 **FLV/Matroska** 内存流），再 Pipe 给 FFmpeg，FFmpeg 的负担将大幅降低，你也不再需要 `genpts` 或 `setts` 这种容易在音视频同步上翻车的补丁。对于追求低延迟和稳定的 DLNA 串流服务，带有时间戳的中间件会一劳永逸地解决 DTS 非单调和交织卡死的问题。

### 5. 关于 AAC 与 MPEG-TS 的兼容性（解答第 6 个问题）

在 MPEG-TS 中封装 AAC（实际上是打成了 ADTS 头）是非常标准且被广泛支持的，MoonVR / LibVLC 和 Skybox / libmpv 对其解码都毫无压力。你现在遇到的单纯是外层的 TS 容器交织崩溃，而非 AAC 本身的解码兼容性问题。所以，坚持使用 AAC 是没问题的。

---

在 Python 端，PyNvVideoCodec 吐出 H.265 NALU 时，如果要实现中间层封装，你目前是否有现成的方案能拿到每一帧硬件编码后的准确时间戳并做轻量级打包？