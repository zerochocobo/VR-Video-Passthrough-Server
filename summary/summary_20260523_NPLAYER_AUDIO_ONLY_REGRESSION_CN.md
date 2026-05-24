# nPlayer 音频模式回归问题定位与首包优化边界

日期：2026-05-23

## 1. 现象

今天做首包延迟优化后，nPlayer/SKYBOX 类真实播放器打开 alpha live MPEG-TS 时，不再进入视频播放界面，而是直接进入音频模式。浏览器/测试客户端不容易暴露这个问题，因为真实播放器会对 TS 内部 video codec params 做更严格判断。

最终用户复测确认：去掉 pipe_ts video 阶段的 `hevc_metadata=aud=insert` 后，播放器终于正常进入视频模式。

## 2. 最终有效日志证据

成功日志中的关键命令：

```text
pipe_ts video mux cmd: ... -f hevc ... -i - ... -c:v copy -bsf:v setts=time_base=1/90000:pts=N*1502:dts=N*1502 ... -f mpegts -
pipe_ts final mux cmd: ... -f mpegts -i - ... -probesize 32768 -analyzeduration 0 -f aac ... -f mpegts -
```

成功轮次里已经不再出现：

```text
PPS id out of range
Skipping invalid undecodable NALU
Could not find codec parameters for stream 0 (Video: hevc ..., none): unspecified size
```

只剩一个非致命 AAC warning：

```text
[aac] Estimating duration from bitrate, this may be inaccurate
```

这说明最终 TS 的 video stream 已经能被播放器识别，音频模式回归已解除。

## 3. 根因

根因不是 DLNA headers，也不是 SKYBOX 文件名规范，也不是 alpha 命名。

真正触发点是 pipe_ts 第一段 video mux 的 bitstream filter 链：

```text
hevc_metadata=aud=insert,setts=...
```

`hevc_metadata=aud=insert` 会强制 FFmpeg HEVC parser 解析输入码流并插入 AUD。生产 live pipe 中，FFmpeg 在这个阶段出现 HEVC 参数集解析失败：

```text
PPS id out of range
Skipping invalid undecodable NALU
Video: hevc ..., none
```

随后 final mux 输出的 MPEG-TS 缺少可用 video codec params，真实播放器把流判定为只有音频，于是进入音频模式。

最终修复是：pipe_ts video 阶段只保留时间戳修正 `setts=...`，不再插入 `hevc_metadata=aud=insert`。

## 4. 修复状态

已保留：

- `pipe_ts` 仍是默认稳定路径。
- pipe_ts video 阶段继续使用 `setts=time_base=1/90000:pts=N*1502:dts=N*1502`。
- 59.94fps tick 修正保留为 `1502`。
- final mux 的 AAC 输入继续使用 `-probesize 32768 -analyzeduration 0`。
- `-fflags +genpts+nobuffer+flush_packets` 当前成功日志中仍在使用，可保留。
- fMP4 `-frag_duration 100000` 不影响本次 MPEG-TS alpha 路径，可保留。

已禁用/回退：

- pipe_ts video 阶段不能再使用 `hevc_metadata=aud=insert,setts=...`。
- raw HEVC stdin 不能再强行加 `-probesize 32 -analyzeduration 0`。
- pipe_ts final mux 的中间 MPEG-TS stdin 不能再强行加低 probe/analyze。
- single-stage `setts` 不能作为默认路径；之前真实播放器出现打开但无内容/卡住，已回退为实验选项。

## 5. 对今天首包优化的影响

没有回退的优化：

- TensorRT/static TRT warmup。
- composite/alpha CuPy kernel warmup。
- NVENC startup preflight。
- `Matter.__init__` warmup runs 形参隔离相关阶段 2 修复。
- alpha 黑色 slate 要求。
- half-equirect 到 fisheye 的 alpha 投影修正。

受限或不能继续推进的首包优化：

- 不能通过压低 raw HEVC probe 来抢首包。
- 不能通过压低中间 MPEG-TS probe 来抢首包。
- 不能在 pipe_ts video 阶段插入 HEVC AUD。
- 不能把默认链路切到 single-stage `setts`。

当前首包数据仍偏慢：

```text
vrkm01797_1_8k @360s: first_chunk total=2726ms
test_8k @0s: first_chunk total=2701ms / 3888ms
```

但播放器识别恢复是优先级更高的正确性修复。后续首包优化必须以“真实播放器仍识别为视频”为硬约束。

## 6. 后续建议

下一轮如果继续优化首包，建议只做可回退、可对照的实验：

- 保持 `pipe_ts` 双段结构不变。
- 不碰 HEVC parser/filter 链。
- 优先定位 final mux 为什么等待到 2.7-3.9s 才输出首包。
- 对照 `PT_FORCE_AUDIO_OFF=1`，确认剩余延迟是否主要来自 AAC 输入/音视频 interleave。
- 如需减少日志噪声，可以把 AAC duration warning 加入 benign warning 列表，但这不影响播放正确性。
