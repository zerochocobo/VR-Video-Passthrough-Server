# A/V 同步问题外部研判摘要

日期：2026-05-23
对象：PTMediaServer / VR passthrough live MPEG-TS
目的：给无法查看源码的外部专家使用。本文只描述系统行为、实验配置、日志现象、客观检测结果和剩余假设。

---

## 1. 摘要

问题现象：

- 在 Quest/VR 播放器上播放实时 passthrough 流时，用户主观听到音画不同步。
- 初期 TensorRT 模式比 CUDA EP 模式更严重；用户描述音频显著晚于画面，量级约 1-2 秒。
- 绿幕模式和 alpha 模式在同一时间点的错位几乎一致。
- 更换另一个播放器后情况依旧，不像是单一播放器设置问题。
- 关闭视频垫片后有所好转，但仍感觉音频晚约 1 秒。

当前核心矛盾：

- 服务端抓取到的 live MPEG-TS 文件，经 ffprobe 时间戳检查和音频内容相关性检查，显示音频与请求源时间对齐在约 21 ms 内。
- 真机播放听感仍然明显偏晚。
- 因此目前不能简单给音频加固定 offset；需要先解释“服务端抓包正常，但真机播放不正常”的差异。

当前倾向：

- 早期“视频垫片 + 静音音频 + AAC 缓存”的确存在可解释的大偏移风险，尤其 TensorRT 冷启动/seek 首帧慢时会放大。
- 但现在默认已经关闭 AAC 缓存和视频垫片，改为直接从源 MP4 读音频进最终 MPEG-TS mux；此路径下服务端抓包显示近似同步。
- 剩余更可疑的方向是：播放器/客户端对 live MPEG-TS、HTTP Range、DLNA live seek headers、HEVC-in-TS 或内部缓冲策略的处理，或我们尚未客观测量“输出视频内容相对源时间”的偏差。

---

## 2. 系统模型

目标功能：

- 本地 DLNA/UPnP 媒体服务器。
- 对普通 MP4 视频生成实时 passthrough 版本。
- 播放 URL 类似：`GET /passthrough_live/{name}?t=<seconds>&mode=green|alpha`

实时链路：

1. 源文件：MP4，HEVC 视频 + AAC 音频。
2. 视频：PyNvVideoCodec/NVDEC 解码，RVM 人像抠像，CUDA/TensorRT 推理，NVENC HEVC 重新编码。
3. 音频：通常转 AAC，目标 48 kHz stereo。
4. 容器：FFmpeg mux 为 MPEG-TS，返回给 DLNA/VR 播放器。

项目背景中最重要的一点：

- 视频和音频不是作为一个整体转封装处理。
- 视频被单独取出，进入 GPU 视频处理链：按请求时间定位源帧，NVDEC 解码，做人像抠像/alpha 或绿幕合成，再由 NVENC 重新编码成新的 HEVC 码流。
- 音频不参与 GPU 抠像链，也不跟随每一帧视频同步处理。它是单独从源文件按请求时间读取/seek，再转成 AAC，最后才与新生成的视频码流重新 mux 到 MPEG-TS。
- 因此所有 A/V 同步都依赖三个边界是否一致：视频解码起点、音频读取起点、最终 MPEG-TS 中的 PTS/DTS/PCR。任何一个边界对 `t=<seconds>` 的解释不同，都可能造成可听/可见错位。

当前主要 MPEG-TS mux 策略：

- 两段 mux：
  - 第一段把 NVENC 输出的 raw HEVC 写成“只有视频”的中间 MPEG-TS，并用 `setts` 合成 CFR PTS/DTS。
  - 第二段把中间 TS 视频和源 MP4 音频 mux 成最终 MPEG-TS。
- 当前关闭 AAC disk cache 时，第二段直接读源 MP4 音频，而不是先生成 `.aac` 缓存。

相关技术标签/搜索词：

- HEVC in MPEG-TS
- AAC in MPEG-TS
- MPEG-TS PTS/DTS/PCR
- FFmpeg `setts` bitstream filter
- FFmpeg `-fflags +genpts`
- FFmpeg `-muxdelay 0`, `-muxpreload 0`, `-max_interleave_delta`
- DLNA `TimeSeekRange.dlna.org`
- DLNA `transferMode.dlna.org: Streaming`
- HTTP Range requests on live/generated streams
- Quest 3 / Android / VR player live MPEG-TS buffering
- 4XVR / SKYBOX / nPlayer / LibVLC HEVC TS compatibility
- NVENC B-frames disabled / low-latency HEVC
- keyframe seeking / GOP-aligned seek / non-keyframe accurate seek
- ONNX Runtime CUDAExecutionProvider vs TensorRTExecutionProvider first-run latency
- RVM recurrent state reset after seek

---

## 3. 测试素材

用户指定测试文件：

`videos/urvrsp00566_1_8k.mp4`

ffprobe 关键信息：

```json
{
  "video": {
    "codec": "hevc",
    "resolution": "8192x4096",
    "avg_frame_rate": "60000/1001",
    "start_time": "0.000000",
    "duration": "1280.729450"
  },
  "audio": {
    "codec": "aac",
    "sample_rate": 48000,
    "channels": 2,
    "start_time": "0.000000",
    "duration": "1280.768167"
  },
  "format_duration": "1280.768167"
}
```

源文件本身从 ffprobe 看不到明显的音频起始 offset：音视频 `start_time` 都是 0。

---

## 4. 已做过的实验与结果

| 序号 | 实验/变更 | 观察结果 | 当前解释 |
|---:|---|---|---|
| 1 | TensorRT 与 CUDA EP 主观对比 | TensorRT 模式更严重，CUDA 也有但轻一些 | 早期符合 TensorRT seek 后首帧/首推理更慢，导致垫片阶段更长的模型 |
| 2 | 绿幕模式与 alpha 模式对比 | 同一时间点错位几乎一样 | 不像 alpha packing 或绿幕合成特有问题 |
| 3 | 运行 `check_av_sync_slate.py` | 有时提示没有 slate 记录，因为 AAC 缓存命中绕过 slate；后续 TensorRT/CUDA 记录显示 `estimated_video_lead_sec=0.0`，burst=1 | slate burst 被压到 1 后，日志层面没有发现 slate 视频显著超前 |
| 4 | 关闭 AAC 缓存 `PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE=0` | 一开始无法播放，后来修复 cache-off 路径；修复后能播但仍主观不同步 | cache-off 开关现在有意义：无缓存时直接走源 MP4 音频进最终 mux |
| 5 | 输出 FPS 改为源 FPS，`PT_PASSTHROUGH_MAX_FPS=0` | 日志显示 `source_fps=59.940`, `output_fps=59.940`；主观仍不同步 | 不是旧 30 fps cap 单独造成 |
| 6 | 关闭视频垫片 `PT_PASSTHROUGH_MPEGTS_VIDEO_SLATE=0` | 用户反馈有所好转，但音频仍晚约 1 秒 | 说明早期垫片可能放大问题，但不是当前唯一问题 |
| 7 | 更换播放器 | 情况依旧 | 不像单一播放器设置；仍可能是多个 Quest/Android 播放器共同的 live TS/Range 行为 |
| 8 | 服务端抓取 live TS 并检查 start_time | alpha/green `t=180` 均约 `audio_minus_video=-0.021333s` | 服务端输出容器 start_time 未显示 1-2 秒偏移 |
| 9 | live TS 音频内容与源音频相关性 | 匹配源时间约 `179.9786875s`，相对请求 `180s` 偏移 `-0.0213125s`，相关性 `0.994846` | 输出音频内容本身没有晚 1-2 秒 |
| 10 | PyNv 解码帧 PTS sanity check | `frame_at(10789).pts / 60000 = 179.963117s`，接近请求 180s | 视频源索引没有明显跳到未来 1-2 秒 |

---

## 5. 关键日志现象

当前修复后的目标状态，日志应出现：

```text
source_fps=59.940
output_fps=59.940
audio cache disabled; using source audio in pipe_ts final mux
pipe_ts final mux cmd: ... -i <source.mp4> -map 0:v:0 -map 1:a:0? -c:a aac ...
```

当前修复后的目标状态，日志不应出现：

```text
slate audio server listening
slate video begin
```

真机请求日志中反复出现的另一个重要现象：

```text
request headers: range='bytes=1072693248-1073741823' ...
live cache hit: ... snapshot=<bytes> ...
live cache subscribe: ... primary=True/False ...
```

含义：

- 某些 VR 播放器会对 live/generated stream 发多个非零 HTTP Range 请求。
- 服务端为了支持实时流和重复请求，会把同一个 live producer 的开头若干数据缓存起来，并让后续请求订阅这个 live session。
- 对 managed live profile，服务端更偏向“Streaming”语义，而不是严格按 Range 返回精确字节段。
- 如果播放器把这些非零 Range 当作 VOD 字节 seek 语义，而服务端返回的是 live snapshot/从头片段，可能引发播放器内部时间线或缓冲误判。这是目前仍需专家评估的方向。

---

## 6. 客观验证工具

已经添加/使用过的工具：

### 6.1 slate 风险检查

命令示例：

```bat
uv run python tools\check_av_sync_slate.py --json --max-lead 0.10
```

用户实测样例：

```json
{
  "sid": 1,
  "frames": 86,
  "fps": 59.94,
  "burst_frames": 1,
  "elapsed_sec": 1.45,
  "video_pts_sec": 1.4347681014347682,
  "estimated_video_lead_sec": 0.0
}
```

解释：

- 该工具只检查 slate 视频阶段是否因为 burst 或过快写入造成“视频 PTS 超前于 wall-clock”。
- 如果 AAC 缓存命中或当前已经禁用 slate，它会提示没有 slate 记录。
- 它不能证明真机端最终播放同步，只能排除/定位 slate 阶段风险。

### 6.2 MPEG-TS 音视频 start_time 检查

命令示例：

```bat
uv run python tools\check_mpegts_sync.py debug_output\urvrsp_current_alpha_t180.ts --json --max-delta 0.10
```

结果摘要：

```text
audio_minus_video_sec = -0.021333
```

解释：

- 负值表示音频 start_time 略早于视频，大约一个 AAC frame 量级。
- 这不是“音频晚 1-2 秒”。

### 6.3 音频内容相关性检查

命令示例：

```bat
uv run python tools\check_live_audio_alignment.py ^
  debug_output\urvrsp_current_alpha_t180.ts ^
  --source videos\urvrsp00566_1_8k.mp4 ^
  --source-start 180 ^
  --duration 5 ^
  --json
```

结果摘要：

```json
{
  "matched_source_sec": 179.9786875,
  "relative_to_source_start_sec": -0.0213125,
  "correlation": 0.994846
}
```

解释：

- 捕获到的 live TS 音频内容，与源文件 180s 附近音频高度相关。
- 音频内容没有晚 1-2 秒。

---

## 7. 当前配置/实现状态

当前重要行为：

- AAC 缓存默认关闭：`PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE=0`
- 视频垫片默认关闭：`PT_PASSTHROUGH_MPEGTS_VIDEO_SLATE=0`
- 旧配置名 `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE` 保留为兼容别名，但实际现在表示同一个“视频垫片总开关”。
- live 输出 FPS 默认跟随源 FPS：`PT_PASSTHROUGH_MAX_FPS=0`
- producer pacing 开启，避免源 FPS 下突发发送。
- NVENC HEVC 配置中 B-frames 为 0，倾向低延迟。
- MPEG-TS 中 PAT/PMT 会重复发送，HEVC AUD 也已启用，用于提高播放器兼容性。

已通过的本地测试：

```text
uv run python -m pytest tests\test_config_defaults.py tests\test_pynv_mux_latency.py
21 passed

uv run python -m pytest tests\test_settings.py tests\test_pynv_mux_latency.py tests\test_config_defaults.py
30 passed
```

---

## 8. 早期 slate 根因与当前状态的关系

早期模型：

- 旧路径在 AAC cache miss 时可能先发送 video slate，同时音频侧发送静音 AAC。
- 视频 slate 用 `setts` 按帧序号生成 PTS。
- 静音音频按 wall-clock 节流。
- 如果 slate 视频前若干帧突发写入，视频 PTS 会比音频 wall-clock 走得更快。
- TensorRT seek 后首帧更慢，slate 时间更长，更容易打满偏移上限，所以 TensorRT 主观更严重。

后续处理：

- slate burst 已被压到 1。
- AAC cache-off 路径不再走 slate/TCP 音频。
- 视频 slate 已加总开关并默认关闭。

当前解释：

- 早期 slate 是一个真实风险，并且关闭后用户确实感觉改善。
- 但在当前 `audio cache disabled; using source audio in pipe_ts final mux` 路径中，理论上不应再有 slate 静音拼接造成的 1 秒音频晚。
- 因此剩余问题需要转向：客户端 live stream 行为、HTTP Range/live cache、MPEG-TS PCR/PTS 细节、或视频内容客观检测缺失。

---

## 9. 不建议的方案

不建议硬编码音频 offset，例如固定提前/延后 1.024 秒。

理由：

- TensorRT 与 CUDA 的主观延迟不完全相同。
- 绿幕和 alpha 一致，但不同素材、不同 seek 点、不同播放器可能不同。
- 当前服务端抓包的音频内容已经接近请求源时间，硬编码 offset 可能让抓包变坏。
- 如果真正问题在播放器 Range/live TS 解释，offset 只能掩盖某些片段，不能解决根因。

---

## 10. 尚未排除的假设

### H1：真机播放器对 live MPEG-TS + 非零 Range 请求处理异常

证据：

- 日志中 Quest/Android 客户端反复发 `Range: bytes=<large>-1073741823`。
- 服务端当前对 managed live session 会复用 live producer/cache，而不是严格返回请求的字节范围。
- 如果播放器内部把响应当作 VOD Range 成功，可能出现时间轴错判。

需要专家判断：

- live/generated MPEG-TS 是否应该完全拒绝非零 Range，例如返回 416 或明确 `Accept-Ranges: none`？
- DLNA `contentFeatures.dlna.org` 的 `OP=10`、`TimeSeekRange`、`X-AvailableSeekRange` 是否会诱导播放器做 Range/time seek？
- 对 4XVR/Quest 这类播放器，live MPEG-TS 更适合 200 chunked streaming、206 pseudo-VOD，还是不暴露 Range？

### H2：服务端只验证了音频内容，没有严格验证“输出视频内容相对源时间”

证据：

- 已验证源解码索引接近请求时间，但还没有用图像相关性/闪帧标记去自动证明输出视频内容正好对应源 180s。
- 用户听到“音频晚”，也可能等价于“视频内容提前”。

需要补充：

- 生成或使用带有可检测视觉/音频标记的测试片，如每秒闪白帧 + beep + 画面帧号。
- 在服务端抓到的 TS 中同时检测 beep 时间和 flash/frame-number 时间。
- 如果服务端抓包中 flash/beep 同步，而真机不同步，则问题基本在播放器/设备端。

### H3：MPEG-TS PCR/PTS 细节被 ffprobe start_time 检查掩盖

证据：

- `check_mpegts_sync.py` 当前只检查 stream start_time delta。
- 它不直接检查 PCR jitter、PCR-vs-PTS lead、长时间 drift、packet interleave、initial buffering policy。

需要补充：

- 用 `ffprobe -show_packets` 或专门 TS analyzer 检查首 10 秒 audio/video packet PTS/DTS/PCR 分布。
- 检查音频包是否在物理字节流中过晚出现，虽然 PTS 正确但播放器等待/缓冲策略不同。

### H4：设备端 HEVC-in-TS 或高码率 8K 流缓冲策略导致呈现时间偏差

证据：

- 源是 8192x4096 59.94fps HEVC，输出也是 live HEVC TS。
- 发送端有 pacing，播放器端仍可能因为硬解、TS demux 或 live buffer 建模产生非预期延迟。

需要专家判断：

- Quest/Android 播放器对 HEVC-in-MPEGTS 的低延迟要求是什么？
- 是否需要 CBR muxrate、固定 PCR 周期、特定 PAT/PMT 策略、AUD、IDR/GOP 策略或更低 mux interleave？

### H5：中段 seek 的关键帧/GOP 对齐导致视频起点与音频起点不同

背景：

- 对压缩视频来说，从任意 `t=<seconds>` 开始播放时，解码通常不能真正从任意 P/B 帧独立开始。
- 实际解码经常需要回退到该时间点之前的最近关键帧/IDR 帧，再从那里解码到目标帧。
- 如果视频处理链实际输出的是“关键帧附近的较早画面”，而音频链直接从请求时间 `t` 开始，就会表现为视频提前、音频相对变晚。
- 反过来，如果视频链内部丢弃了关键帧到目标帧之间的预滚帧，但时间戳仍按输出帧从 0 开始，也可能出现另一种起点解释差异。

本案疑点：

- 这个假设能解释“从中段章节/seek 点开始播放”时产生错位。
- 但用户反馈从片头开始播放也会出现类似错位。片头 `t=0` 正常应当就是关键帧起点，不应该有“回退到更早关键帧”的问题。
- 因此关键帧/GOP 对齐可能是中段 seek 的风险因素，但不能单独解释全部现象。

需要补充验证：

- 在日志和抓包里同时记录请求 `t`、视频实际首帧源 PTS、音频实际首包源 PTS。
- 对同一文件测试 `t=0`、`t=180`、`t=360` 等多个点，看错位是否与 GOP 间隔/最近关键帧距离相关。
- 用带帧号/时间码烧录的视频测试片，客观检测输出首帧到底来自源文件哪个时间。

---

## 11. 建议下一步实验

优先级从高到低：

1. 制作“自动判定用同步测试片”：
   - 视频：每秒一个明显 flash/数字帧号。
   - 音频：同一时间点 beep/click。
   - 通过同一 passthrough live 链路播放。
   - 服务端抓 TS 后自动检测 flash 与 beep 的 offset。

2. 增加“视频内容相关性”工具：
   - 从 live TS 抽帧。
   - 从源文件对应时间窗口抽帧。
   - 用感知 hash、SSIM、OCR 帧号或 flash 检测匹配输出视频的源时间。
   - 与现有音频相关性工具一起判断是音频晚还是视频早。

3. 增加 seek/keyframe 起点审计：
   - 记录视频实际首帧源 PTS 与请求 `t` 的差值。
   - 记录音频实际首包源 PTS 与请求 `t` 的差值。
   - 分别测试 `t=0` 和多个中段时间点，检查是否与 GOP/关键帧距离相关。

4. 对真机 Range 行为做 A/B：
   - A：对 live stream 明确返回 `Accept-Ranges: none`，并对非零 Range 返回 416。
   - B：完全禁用 live session snapshot cache，每次请求独立 stream 或只允许主请求。
   - C：改为 pseudo-VOD 206 语义，认真响应 Range/Content-Range。
   - 观察主观同步和日志请求模式是否变化。

5. 深入分析 TS packet/PCR：
   - 提取首 10 秒 audio/video packet PTS/DTS/PCR。
   - 检查 packet order、PCR lead、initial audio packet 到达字节位置。
   - 与播放器兼容建议对照。

6. 对原始源文件做同播放器基线：
   - 直接播放 `urvrsp00566_1_8k.mp4` 原片。
   - 如果原片在同播放器也音频晚，问题可能是素材或播放器源解码。
   - 如果原片同步而 live 不同步，问题在生成流/传输/容器。

---

## 12. 给专家的核心问题

1. 对 Quest/Android VR 播放器，HEVC + AAC 的 live MPEG-TS 应如何处理 HTTP Range？应该拒绝、忽略、还是实现 pseudo-VOD？
2. ffprobe 显示 audio/video start_time 只差 21 ms，且音频内容相关性也只差 21 ms，但真机听感晚 1 秒，这种情况最常见原因是什么？
3. 是否存在播放器根据 PCR/包到达顺序而不是 PTS start_time 做初始缓冲，导致“文件检查同步但播放不同步”的案例？
4. `setts` 给 raw HEVC 合成 CFR PTS，再和源 MP4 AAC mux 成 TS，这种两段 mux 是否有已知兼容坑？
5. 在 DLNA headers 中同时暴露 `TimeSeekRange`、`X-AvailableSeekRange` 和 live streaming，对某些播放器是否会诱导错误 seek/buffer 策略？
6. 从非关键帧时间点开始实时转码时，视频回退到关键帧预滚、音频精确 seek、最终 PTS 重基准三者最容易在哪里产生 0.5-2 秒级错位？若 `t=0` 也错位，应优先排除哪些非关键帧因素？
