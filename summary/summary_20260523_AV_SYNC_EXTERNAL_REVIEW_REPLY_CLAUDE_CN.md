我把这份文档完整读完了。结构很清晰，实验也做得扎实。下面是我作为流媒体/封装侧的分析，先给结论，再给可执行的诊断和修复顺序，最后逐条回答你列给专家的 6 个问题。

## 一句话核心判断

你目前的"矛盾"其实**还没被完整证伪**。你证明的是"输出 TS 里的**音频内容**与请求源时间对齐在 21ms 内"，但你**从未客观证明输出 TS 里的视频内容也对应同一源时间**，也**从未在真机端做端到端测量**。所以"音频晚 1 秒"这个描述本身可能等价于"视频早 1 秒"，而"服务端抓包正常"对视频侧是未经证实的假设。在补上这两块之前，任何关于根因的结论都站不稳。

更重要的一点:文件级检查全部通过（21ms），但真机一致性偏移、多播放器复现、且 **t=0 也偏移**——这个组合的典型含义是**问题不在内容/PTS，而在"实时投递 + 播放器缓冲"的交互层**，也就是 H1+H3，而不是 H5（seek/GOP）。H5 你自己也已经排除了（t=0 无预滚却仍偏移）。

## 你现在最大的盲点（也是最快出真相的两个实验）

这两个实验比你列表里的任何修复都更值得先做，因为它们是**决定性**的，能把问题域从"无限假设"砍成"一个明确方向":

**A. 原片基线（最便宜、最决定性）。** 用**同一台 DLNA 服务器、同一个播放器**直接播 `urvrsp00566_1_8k.mp4` 原片，不走 passthrough。
- 原片也晚 → 问题在播放器/源解码/传输容器层，passthrough 链路基本无辜。
- 原片同步、passthrough 晚 → 问题被锁定在"生成流 + 两段 mux + 实时投递"。

这是你列表里的第 6 项，但它应该是**第 0 项**。在拿到这个结果之前做服务端调参是在盲打。

**B. 烧录标记的端到端测试片。** 每秒一个白闪 + beep + 烧录帧号，走同一条 live 链路:
- 服务端抓 TS，自动测 flash↔beep offset（你已规划）。
- **关键补充**:同时用 Quest 内录或外部高帧率相机拍真机播放画面+声音，测真机上的 flash↔beep offset。

如果服务端抓包 flash/beep 同步、真机不同步 → 问题确定在设备/播放器/投递，服务端再怎么 mux 也没用，应该转向传输协议和 player profile。如果服务端抓包就已经偏移 → 你之前的"音频内容相关性 0.994"和"start_time 21ms"之所以没抓到，是因为它们测的是 PTS 元数据，不是**字节流里包的物理到达顺序**。

## 根因按概率排序

**第一梯队（最可能，且能解释"文件正常/真机异常/多播放器/t=0"全部现象）:**

1. **MPEG-TS 字节流的交织（interleaving）与 PCR 节奏。** `start_time` 只描述每条流的第一个 PTS，完全不反映音视频包在字节流里的**物理交错顺序**。轻量级/VR/live 播放器普遍**不做严格的 PTS 对齐缓冲**——它们用 PCR 做时钟恢复，然后各路解码器"缓冲填够就开始渲染"。如果你的两段 mux 让视频包在字节流里前置一大段、音频包物理上晚出现，音频解码器就会晚启动 → 听感音频晚。这与症状方向完全一致，且与文件 PTS 正常并不矛盾。

2. **live cache / snapshot / Range 的混合语义（H1）。** 这是你日志里最可疑的现象。播放器发 `Range: bytes=<large>-1073741823`（典型 VOD 字节 seek 探测），而服务端按 live 语义返回 snapshot/从头片段。播放器若把这个响应当成 VOD Range 成功，它内部的"字节↔时间"映射就错了，时间轴/缓冲建模随之错位。这种"既宣告可 seek、又返回 live 片段"的中间态是**最坏的组合**，且天然会在多个播放器上一致复现。

**第二梯队（真实但量级不足以单独解释 1 秒）:**

3. **slate 残留**——你已证实关闭后改善，说明它是贡献项之一，但不是当前主因。
4. **AAC encoder priming / mux preload**——48kHz 下 priming 约 1024–2048 samples ≈ 21–43ms，正好能解释那个 21ms 量级的 `audio_minus_video`，但解释不了 1 秒。

**基本可排除:** H5（GOP/seek），因为 t=0 也偏移；alpha/green 一致也说明不是合成链特有。

## 服务端可立即尝试的修复（B 实验之后，或并行）

**针对交织/PCR（最高优先）:**
- 最终 mux 显式加 `-muxdelay 0 -muxpreload 0`，并把 `-max_interleave_delta` 设很小（谨慎用 0）。
- 用 `ffprobe -show_packets -show_entries packet=stream_index,pts_time,dts_time,pos` 看输出 TS **前 10 秒**:对照每个包的 `pos`（字节位置）和 `pts_time`。如果第一个音频包的字节位置远远落后于同 PTS 的视频包，就是交织问题坐实。
- 检查 PCR 是否存在且周期合理（典型 ≤40ms 一次），PCR-to-PTS lead 是否够。
- **架构性建议**:两段 mux 里，pass 2 的交织质量取决于 FFmpeg 是否同步读两路输入。如果 pass 1 已经把视频整段生成，pass 2 很可能不是良好交织。考虑让音频在**产生最终时基的同一次 mux**里进入，而不是事后拼。

**针对 Range/live（H1）:**
- 对 live/generated 流，**选定一个一致的模型**，别走中间态:
  - 推荐 live 语义:`200 OK` + sender-paced streaming + `Accept-Ranges: none` + `transferMode.dlna.org: Streaming`，对非零 Range 直接忽略并从当前 live 点流出（或返回 416），且 **DLNA contentFeatures 不宣告 byte-seek/time-seek**。
  - 不要同时暴露 `TimeSeekRange` / `X-AvailableSeekRange` / `OP=10`——这些会诱导播放器去做 VOD 式 seek 和缓冲建模。

## 逐条回答你给专家的 6 个问题

**Q1（Range 怎么处理）:** 选一个一致模型，别混。对真正的 live/generated 流，最稳的是 sender-paced live streaming:`Accept-Ranges: none`、`transferMode.dlna.org: Streaming`、contentFeatures 只宣告流式不宣告可 seek，非零 Range 忽略或 416。pseudo-VOD 206 只有在你能**真正准确**把 byte-range 映射到时间时才用——实时生成器一般做不到。当前"宣告可 seek + 返回 snapshot"是最坏选择。

**Q2（21ms 文件 vs 1 秒真机的最常见原因）:** 最常见就是**问题不在文件内容，而在播放器对 live 流的初始缓冲不对称 + 字节流交织/PCR + 传输层 Range 误判**。`start_time` 只说每条流第一个 PTS 在哪，对交织顺序、PCR 节奏、包到达顺序、live 缓冲填充策略一无所知。而且——你还没证明视频内容对齐，"音频晚"可能是"视频早"，这必须先排除。

**Q3（按 PCR/到达顺序而非 PTS 缓冲）:** 是的，非常常见，尤其轻量级/嵌入式/VR 播放器。很多对 live 流不做严格 PTS 对齐:用 PCR 恢复时钟，各路解码器缓冲填够就开始渲染，起点取决于包在字节流里的到达顺序。视频前置 → 音频解码器晚启动 → 音频滞后。这是你这个症状的头号嫌疑。

**Q4（setts + 两段 mux 的兼容坑）:** 有几个。(1) 音频被 input-seek 后保留残余起始 offset，不归零（`-avoid_negative_ts make_zero` / `-start_at_zero` / `muxpreload/muxdelay 0`）会产生固定 A/V 偏移——你只有 21ms 说明这块控制得不错。(2) raw HEVC 无时基，`setts` 必须保证 DTS≤PTS 单调；若帧时长/`r_frame_rate` 与真实 59.94 不完全一致，会在几分钟尺度累积 drift。(3) **两段 mux 最大的坑就是交织**:pass 2 的字节流交织取决于 FFmpeg 是否锁步读两路输入，PTS 对但交织可能很差。(4) 从中间 TS 再 mux 容易丢/重 PCR 或制造 discontinuity。建议把音频放进产生最终时基的那次 mux，并显式控交织。

**Q5（同时暴露 TimeSeekRange + X-AvailableSeekRange + live）:** 会。同时宣告时间 seek 和字节 seek 能力，等于告诉播放器"你可以 seek"，部分播放器就会发 seek/Range 探测并建立 VOD 式时间轴/缓冲模型，随后服务端返回 live snapshot 时映射错位。live 就只宣告流式、不给 seek 头;要 seek 就彻底做可 seek VOD。混合信号是公认的 flaky 来源。

**Q6（预滚/精确 seek/重基准三者哪里最易出 0.5–2s；t=0 也错先排什么）:** 中段最经典的偏移源是视频预滚:从 `t` 之前最近 IDR 解码，如果这些预滚帧被**输出而非丢弃**，视频起点就早于 `t`、音频精确在 `t` → 视频早/音频晚，量级最多一个 GOP（可达 1–2s）。验证:同时记录视频首输出帧的源 PTS 和音频首包源 PTS 对请求 `t` 的差，测 t=0/180/360 看是否与 GOP 距离相关。但既然 **t=0 也偏移**（无预滚），优先排除这些非关键帧因素:① 原片基线（到底是不是 pipeline）;② 交织/PCR/初始缓冲;③ 传输 Range/live-cache;④ 任何固定的音频 priming 或 mux preload。t=0 那份偏移几乎可以肯定是传输/播放器缓冲，不是 seek/GOP。

---

总结一下行动顺序:**先做原片基线 + 烧录标记端到端（含真机录制）这两个决定性实验**，把"服务端 vs 设备"一刀切开;在此之前别调 mux 参数。如果证明是服务端,优先查交织/PCR 并修 Range/live 语义;如果证明是设备,转向 player profile 和传输协议,服务端 mux 调参收益有限。

需要的话,我可以帮你写那个"烧录 flash+beep+帧号 的测试片生成脚本",以及"从抓到的 TS 里自动检测 flash↔beep offset + 视频内容相关性"的检测工具——这正好补上你 H2 缺的视频侧客观验证。要我直接写吗?