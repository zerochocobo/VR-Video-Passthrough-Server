我看完附件后，结论比较明确：**现在不应该先做固定 audio offset**。你现在遇到的不是典型“mux 后音频 PTS 晚 1 秒”的问题，而更像是：

> 服务端抓到的 TS 文件在时间戳和音频内容上基本同步，但 Quest/VR 播放器在 live/generated MPEG-TS 的 HTTP Range、DLNA seek/header、PCR/包到达顺序或初始缓冲策略上，把这条流解释错了。
> 另一个必须补测的是：你只客观验证了音频接近源时间，还没有同等级验证“输出视频画面内容到底对应源文件哪个时间点”。

## 先判断：最可疑的不是音频本身，而是“播放端如何理解这条流”

附件里的关键证据是：

* `check_mpegts_sync.py` 显示 `audio_minus_video = -0.021333s`，音频还略早于视频约 21ms。
* 音频内容相关性检查匹配到源文件 `179.9786875s`，相对请求 `180s` 也是约 `-21ms`。
* 关闭 AAC cache、关闭 video slate 后，服务端路径已经变成“源 MP4 音频直接进入最终 mux”。
* 真机仍主观感觉音频晚约 1 秒。

所以如果现在硬加 `-itsoffset -1.0` 之类，服务端抓包会被你人为搞坏。真正需要解释的是：**为什么一个本地抓包看似同步的 MPEG-TS，到 Quest/播放器里会播放成不同步。**

我会把优先级排成这样。

---

## 最高优先级 1：先把 live stream 的 Range 语义改干净

你日志里反复出现：

```text
Range: bytes=1072693248-1073741823
live cache hit
live cache subscribe
```

这是很危险的信号。HTTP Range 的语义是：客户端请求资源的某一段字节，服务端如果支持，就应该返回 `206 Partial Content`，且响应体就是请求的那段内容；如果范围无效，应返回 `416 Range Not Satisfiable`。MDN 对 `206` 和 `Range` 的定义也是这个意思：`206` 响应体包含请求 Range 指定的数据，`Range` 请求头表示客户端要求返回资源的一部分。([MDN 文档][1])

但你的 managed live stream 实际不是稳定 VOD 文件。它是实时生成的 MPEG-TS。如果播放器请求一个巨大非零 byte range，而服务端返回的是“live session snapshot / 从头片段 / 订阅已有 producer”，那客户端很可能以为自己拿到了某个 VOD 字节区间，内部时间线、缓冲区、demux 初始点就可能错。

**建议立刻做 A/B 实验：**

### A 方案：live passthrough URL 完全不支持 byte range

对 `/passthrough_live/...` 这类实时生成流：

```http
HTTP/1.1 200 OK
Content-Type: video/MP2T
Accept-Ranges: none
Transfer-Encoding: chunked
```

并且：

```text
如果请求头有 Range 且不是 bytes=0-，直接返回 416 或忽略 Range 改 200 从头发。
```

我倾向于先用更强硬的策略：**非零 Range 直接 416**。因为这样能快速观察播放器是否停止走错误路径。如果它完全不能播，再退一步做“忽略 Range，返回 200 chunked”。

### B 方案：如果要支持 Range，就必须真的做 pseudo-VOD

也就是要能保证：

```text
Range: bytes=X-Y
```

返回的就是全局稳定字节流中的 X-Y。实时生成流一般做不到，除非你先完整生成/缓存一个确定文件，或者按固定码率、固定索引维护映射。对你的场景，这条路复杂度很高，不建议先做。

### C 方案：DLNA 层不要同时暗示“可 seek”与“live”

微软的 DLNA 相关文档提到，如果 DMS 支持通过 `TimeSeekRange.dlna.org` 做 DLNA Seek Media Operation，HTTP 200 响应里应包含 `X-AvailableSeekRange`，并且其范围表示服务端愿意接受的 TimeSeekRange 区间。([微软学习][2])

这意味着：你如果暴露了 `TimeSeekRange.dlna.org` / `X-AvailableSeekRange` / contentFeatures 里的 seek 能力，播放器可能会真的按 seekable media 去操作。对于你的实时生成 TS，建议分两种 URL：

```text
/passthrough_live/...      纯 live，不承诺 byte seek，不承诺 time seek
/passthrough_vod/...       先生成或缓存后，按 VOD/Range 语义严格支持 seek
```

不要在 live URL 上同时给播放器“这是 live streaming”和“你可以按时间/字节 seek”的混合信号。

---

## 最高优先级 2：补一个“视频内容时间”检测工具

你现在已经验证了音频，但还没有同等级验证视频。附件里也把 H2 列为关键假设：服务端只验证了音频内容，没有严格验证输出视频画面相对源时间。

这是必须补的。因为用户说“音频晚”，主观上等价于：

```text
音频真的晚了
或
视频画面提前了
```

你目前只能证明“音频没晚”，还不能证明“视频没提前”。

建议做一个专门测试片：

```text
视频：每秒大号数字时间码 + 每秒闪白帧
音频：同一秒发 beep/click
帧率：59.94 或 60
分辨率：先 1920x1080，再 4096x2048，再 8192x4096
编码：HEVC + AAC
```

通过你的完整 passthrough live 链路输出后，服务端抓 TS，自动检测：

```text
beep 出现时间
flash 出现时间
画面 OCR/帧号对应源时间
```

只要这个测试成立，你就能把问题分成三类：

| 结果                           | 结论                                   |
| ---------------------------- | ------------------------------------ |
| 服务端 TS 里 beep/flash 同步，真机不同步 | 播放器/HTTP/DLNA/TS 解释问题                |
| 服务端 TS 里视频比音频早 1 秒           | 视频 seek / 首帧 / setts / PTS 重基准问题     |
| 服务端 TS 里音频比视频晚 1 秒           | 音频 seek / AAC 编码 / mux interleave 问题 |

这一步比继续猜 FFmpeg 参数更重要。

---

## 高优先级 3：检查 TS 包级别，而不是只看 stream start_time

`ffprobe stream start_time` 只能说明第一批 PTS 的相对起点，不足以排除播放器端初始缓冲问题。MPEG-TS 播放器还会受到 PCR、packet order、PAT/PMT 周期、音频包在物理字节流中的出现位置、PCR 与 PTS 的 lead 等影响。

FFmpeg 的 MPEG-TS muxer有几个与你情况直接相关的选项：`muxrate` 可以设置 constant muxrate，默认是 VBR；`pcr_period` 可以覆盖 PCR 重发周期，默认自动选择，CBR 使用 20ms，VBR 使用低于 100ms 的帧周期倍数；`pat_period` 默认 0.1 秒，`sdt_period` 默认 0.5 秒；`mpegts_flags` 里有 `resend_headers`、`pat_pmt_at_frames`、`initial_discontinuity` 等选项。([FFmpeg][3])

建议加一个 packet 审计脚本，抓前 10 秒：

```bash
ffprobe -hide_banner -show_packets -select_streams v \
  -show_entries packet=pts_time,dts_time,pos,flags,stream_index \
  -of csv debug_output.ts > video_packets.csv

ffprobe -hide_banner -show_packets -select_streams a \
  -show_entries packet=pts_time,dts_time,pos,flags,stream_index \
  -of csv debug_output.ts > audio_packets.csv
```

重点看：

```text
1. 第一个 video packet 的 pts/dts/pos
2. 第一个 audio packet 的 pts/dts/pos
3. 前 1 秒内 audio packet 是否物理上很晚才出现
4. PTS 是否从接近 0 开始，是否有 1.4s 这类 MPEG-TS 常见初始偏移
5. video DTS 是否单调
6. PCR 所在 PID 是否稳定
7. PAT/PMT 是否足够早且重复
```

如果音频 PTS 正确但音频包物理上很晚才到，某些播放器可能会先建立视频缓冲，再等音频，导致启动阶段的 A/V 呈现策略异常。

---

## 高优先级 4：减少“两段 mux + setts”的不确定性

你当前路径是：

```text
raw HEVC -> 中间 video-only TS，用 setts 合成 CFR PTS/DTS
中间 TS + 源 MP4 audio -> 最终 TS
```

FFmpeg 文档对 bitstream filter 的定义是：bitstream filter 在编码后码流层面操作，不做解码。([FFmpeg][4]) 这意味着 `setts` 这类方案本质上是在“包时间戳层”补时间，而不是从同一个滤镜图里用统一时钟生成 A/V。

它不是一定错，但对实时播放器兼容性不是最稳。

我建议做一个对照实验：**临时绕开中间 video-only TS**，让最终 mux 直接从两个 pipe 输入：

```text
pipe 0: HEVC elementary stream，带明确 packet pacing / 时间戳
pipe 1: audio
最终 ffmpeg 一次性 mux 成 MPEG-TS
```

如果 FFmpeg 对 raw HEVC pipe 无法可靠感知时间戳，可以考虑两种更稳的结构：

### 方案 1：视频编码侧输出 Annex B + 外部严格按帧送入，最终 mux 使用 `-r source_fps`

适合快速验证，但仍有时间戳合成风险。

### 方案 2：用 libavformat/PyAV 自己写最终 TS

你自己控制：

```text
video packet pts/dts/duration
audio packet pts/dts/duration
PCR/stream time_base
```

这工程量更高，但长期最稳。你的系统是“实时生成播放器兼容流”，最终可能还是需要自己掌控 packet 级时钟。

---

## 中优先级 5：重新审计 seek 起点，尤其是视频首帧源 PTS

附件里 H5 很关键：视频链路可能从关键帧预滚，音频链路从精确 `t` 开始。

即使 `frame_at(10789).pts / 60000 = 179.963117s` 接近 180s，也还要记录最终实际写入第一帧的源 PTS：

```text
request_t
video_decode_seek_target
video_first_decoded_pts
video_first_output_pts
video_first_encoded_pts
audio_seek_target
audio_first_packet_source_pts
final_ts_first_video_pts
final_ts_first_audio_pts
```

你要特别区分：

```text
第一帧解码了什么
第一帧输出了什么
第一帧编码进 TS 的 PTS 是什么
播放器看到的第一个 IDR 是什么
```

如果从 `t=180` 请求，但实际输出了 `178.9s` 的画面，而音频是 `179.98s`，那听感就是音频晚约 1 秒。
如果 `t=0` 也错位，则优先看 Range/live/TS/PCR，而不是 keyframe seek。

---

## FFmpeg mux 参数建议：先作为实验组，不要盲目全开

你可以做一个“保守兼容 MPEG-TS”实验 profile：

```bash
-f mpegts
-muxdelay 0
-muxpreload 0
-mpegts_flags +resend_headers+pat_pmt_at_frames
-pcr_period 20
-pat_period 0.05
-sdt_period 0.25
```

FFmpeg 官方文档说明 `-muxdelay` 是最大 demux-decode delay，`-muxpreload` 是初始 demux-decode delay。([FFmpeg][5]) MPEG-TS muxer 里 `pcr_period`、`pat_period`、`sdt_period` 也都可调。([FFmpeg][3])

但注意：这些参数不是根治方案，只是为了验证“播放器是否受 TS 初始结构/缓冲策略影响”。

我建议你做四组输出对比：

```text
G0 当前默认参数
G1 muxdelay=0 muxpreload=0
G2 G1 + pcr_period=20 + pat_period=0.05
G3 G2 + pat_pmt_at_frames/resend_headers
```

每组都做：

```text
服务端抓包同步检测
Quest 真机主观/录像检测
请求日志中的 Range 行为
首 10 秒 packet 分布
```

---

## 我认为最值得立刻改的代码策略

按性价比排序：

### 1. live URL 禁止非零 Range

这是最可能立刻改变真机表现的地方。

伪逻辑：

```python
range_header = request.headers.get("Range")

if is_passthrough_live:
    headers["Accept-Ranges"] = "none"

    if range_header and not range_header.startswith("bytes=0-"):
        return Response(status_code=416, headers={
            "Accept-Ranges": "none",
            "Content-Range": "bytes */*",
        })

    return StreamingResponse(
        producer(),
        status_code=200,
        media_type="video/MP2T",
        headers=headers,
    )
```

如果 416 导致某播放器无法播放，再改成：

```text
忽略 Range，始终 200，从当前 live session 头开始发
```

但不要返回 206，除非你真的按 Range 返回了指定字节。

### 2. live DLNA contentFeatures 不暴露 seek 能力

对于 passthrough live profile：

```text
不要给 TimeSeekRange
不要给 X-AvailableSeekRange
不要声明 byte seek
transferMode.dlna.org 更偏 Streaming
```

对于 VOD/finalized profile 才提供 seek。

### 3. 禁用 live snapshot cache 做 A/B

你现在的 `live cache hit / subscribe` 很可能把多个 Range 请求混进同一个 producer 时间线。做一个环境变量：

```text
PT_PASSTHROUGH_LIVE_CACHE=0
```

每个请求独立 producer，或者只允许 primary 请求，secondary 非零 Range 直接拒绝。看真机 A/V 是否变化。

### 4. 加视频内容相关性工具

不要只靠音频相关性。输出 TS 抽帧，与源视频对应窗口做匹配。哪怕先用简单的 pHash/SSIM，也能快速判断“视频是否提前”。

### 5. 加 packet/PCR 审计

检查“PTS 正确但包到达顺序/初始音频包位置不利于播放器”的情况。

---

## 对你附件中 12 个专家问题的简答

1. **Quest/Android VR 播放器的 live MPEG-TS 应如何处理 Range？**
   live/generated TS 不应假装支持 byte range。要么明确不支持并返回 200 chunked / `Accept-Ranges: none`，要么对非零 Range 返回 416；只有能稳定映射全局字节时才返回 206。

2. **ffprobe 只差 21ms，但真机晚 1 秒，常见原因是什么？**
   最常见是播放器没有按你以为的 PTS 起点播放：Range/seek 语义错、PCR/包顺序/初始缓冲策略、或者实际视频内容时间提前但没被检测到。

3. **播放器是否可能根据 PCR/包到达顺序而不是 stream start_time 做初始缓冲？**
   可能。MPEG-TS 播放不是只看 stream start_time；PCR、DTS/PTS、packet interleave、PAT/PMT、首包位置都会影响 demux 和呈现。

4. **`setts` + 两段 mux 有无兼容坑？**
   有风险。它能构造时间戳，但不是从统一 A/V 时钟自然 mux 出来的。对普通 ffprobe 可能正常，对部分硬件播放器/live 模式未必最稳。

5. **DLNA headers 混合暴露 TimeSeekRange / X-AvailableSeekRange / live streaming 是否危险？**
   是。会诱导客户端按 seekable media 处理。live 和 VOD 最好拆 profile。

6. **非关键帧 seek 最容易哪里错？**
   视频回退关键帧预滚、音频精确 seek、最终 PTS 重基准三者不一致时最容易错。若 `t=0` 也错，则优先查 Range/live/TS/PCR，而不是 keyframe。

---

## 最终建议路线

你下一轮不要再优先调 offset，而是这样走：

```text
第一步：live URL 禁止非零 Range，关闭/简化 DLNA seek 暴露
第二步：关闭 live snapshot cache 做 A/B
第三步：做 beep+flash+帧号测试片，验证视频内容时间
第四步：做 TS packet/PCR 首 10 秒审计
第五步：实验 muxdelay/pcr/pat 参数
第六步：如果仍有问题，再考虑重构为单阶段 mux 或自己写 packet 时间戳
```

我目前的判断是：**H1 live MPEG-TS + 非零 Range / DLNA seek 语义冲突，是最高概率根因；H2 视频内容未客观验证，是最高优先级证据缺口。**

[1]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/206?utm_source=chatgpt.com "206 Partial Content - HTTP - MDN Web Docs - Mozilla"
[2]: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-dlnhnd/50fc1557-dc3d-424a-96b0-cc4bdab9e7c7?utm_source=chatgpt.com "[MS-DLNHND]: Requesting to Start Streaming Using HTTP"
[3]: https://ffmpeg.org/ffmpeg-formats.html "      FFmpeg Formats Documentation
"
[4]: https://ffmpeg.org/ffmpeg-bitstream-filters.html?utm_source=chatgpt.com "FFmpeg Bitstream Filters Documentation"
[5]: https://ffmpeg.org/ffmpeg.html?utm_source=chatgpt.com "ffmpeg Documentation"
