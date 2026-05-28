# Seekable Passthrough 实验复盘与经验教训（2026-05-28）

## 1. 结论摘要

本轮实验目标是把现有实时 passthrough 从“直播流 + 章节选择”改造成播放器可拖动的普通进度条体验。经过 TS 伪 VOD、DLNA seek 广告、HTTP byte Range 映射、真实 fMP4 输出、URL 后缀/MIME/header 调整、prefix cache 和多轮 nPlayer/Skybox/HereSphere 实机测试后，结论是：

**实时生成的 passthrough 流不能靠 HTTP/DLNA header 伪装成普通可 seek 文件。**

原因不是某一个 header 写错，而是播放器对“文件式 seek”的判断依赖容器本身的索引和字节稳定性：

- MPEG-TS 即使给 `Content-Length`、`Accept-Ranges`、`transferMode=Interactive`、`OP=11`，nPlayer/Skybox 仍倾向按 live stream 处理，不显示普通文件时间轴。
- fMP4 虽然 `Content-Type: video/mp4`，但在线生成的 fragments 不是一份完整、字节稳定、可随机访问的静态 MP4 文件。nPlayer 会像读取真实 MP4 一样发 tail/mid Range 去找 `moov`/sample table，在线 fMP4 无法满足。
- 直接复用原始 MP4 的 `moov` 索引也不可行，因为处理后视频重新编码，sample size、chunk offset、关键帧、codec config 和音频 interleave 都会变化。

因此这条“实时流伪装成可拖动文件”的路线暂时放弃。未来第二次尝试应只考虑两条真实可行路径：

1. **离线/近线生成完整 MP4 文件后再按 `/media` 静态 Range 服务。**这是最稳路径，nPlayer 抓包已经证明真实 MP4 文件会显示进度并按 Range seek。
2. **实现真正的虚拟 MP4 文件系统。**需要生成可信 `moov`/`sidx`/sample table，并保证所有 Range 返回的字节与索引完全一致。这不是简单 header 改造，复杂度接近自研 MP4 muxer + 持久化 segment store。

当前已把 UI 测试开关回滚为：

```json
"passthrough_seek_enabled": false,
"passthrough_seek_dlna": false,
"passthrough_seek_container": "mpegts"
```

旧 `/passthrough_live` 功能保持不受影响。

## 2. 原始目标与约束

### 2.1 用户体验目标

现有实时播放模式只能暴露 `/passthrough_live`：

- 输出容器：MPEG-TS。
- 播放体验：播放器认为这是 live/streaming resource。
- seek 方式：通过 DLNA 虚拟章节或 URL 参数 `t=` 选起点。

用户实际习惯是普通视频文件式进度条：进入播放界面后看到总时长，拖动进度条，播放器自动向服务端发 Range 或 time seek 请求。

### 2.2 技术约束

关键约束如下：

- 目标播放器并不统一支持 DLNA `TimeSeekRange.dlna.org`。
- 很多播放器实际只发 HTTP `Range: bytes=N-`。
- HTTP Range 语义要求同一 URL 表示同一份稳定字节资源。
- 实时 passthrough 的输出是运行时重新编码/重新 mux 的结果，不是一份已有文件。
- 旧 `/passthrough_live` 已经是可用 fallback，不能因为实验破坏。

因此首要设计原则是：**新功能必须独立开关、独立 endpoint、默认关闭、可随时回退到旧 live。**

## 3. 本轮实现过的能力

### 3.1 新增配置与 UI 测试开关

新增并使用过的配置：

- `PT_PASSTHROUGH_SEEK_ENABLED`
  - 控制 `/passthrough_seek/...` 路由是否可访问。
  - 关闭时手工 URL 也不可用。
- `PT_PASSTHROUGH_SEEK_DLNA`
  - 控制 DLNA Browse 是否额外暴露 seek 项。
  - 不单独启用 HTTP 路由。
- `PT_PASSTHROUGH_SEEK_ROUTE_POLICY`
  - `profile` / `all` / `off`。
  - 实机测试阶段用过 `all`。
- `PT_PASSTHROUGH_SEEK_PROFILES`
  - route 层 UA 白名单。
- `PT_PASSTHROUGH_SEEK_HEADER_BYTES`
  - 预留一个真实、稳定的前缀字节区，默认约 2MB。
- `PT_PASSTHROUGH_SEEK_CONTAINER`
  - `mpegts` 或 `mp4`。
  - `mpegts` 是默认实验容器。
  - `mp4` 后来用于 true fMP4 A/B 测试。

UI 的 `runtime_cache/ui_settings.json` 曾临时启用：

```json
"passthrough_seek_enabled": true,
"passthrough_seek_dlna": true,
"passthrough_seek_route_policy": "all",
"passthrough_seek_container": "mp4" 或 "mpegts"
```

最后已回滚为 disabled。

### 3.2 新 endpoint

新增 `/passthrough_seek/{name:path}`：

- 支持 GET/HEAD。
- 支持 `Range: bytes=...`。
- 支持 `TimeSeekRange.dlna.org`。
- 支持 query `mode=green|alpha`。
- 支持 URL suffix：
  - `.seek.ts` -> `mpegts`
  - `.seek.mp4` -> `mp4`
- route 内部会剥掉 `.seek.ts` / `.seek.mp4` 再解析真实媒体 key。

这个 endpoint 和 `/passthrough_live` 分离，避免修改旧 live 行为。

### 3.3 DLNA 暴露策略

最初设计曾考虑“seek 入口替换 live 入口”，后经 review 修正为“并存”：

- 当 `ENABLED=1` 且 `DLNA=1` 时，同一个 passthrough mode 同时展示：
  - `/passthrough_seek/...`
  - 原有 `/passthrough_live/...` 章节/直播 fallback
- 这样即使某个播放器被 route profile 拦截或 seek 失败，目录里仍有 live fallback。

这是正确经验，后续任何实验都应保留：**实验入口不能替换稳定入口。**

### 3.4 虚拟 CBR byte seek 映射

实现过的核心模型：

```text
declared_size = header_reserved + estimated_output_bps * duration / 8 + padding
ratio = (range_start - header_reserved) / (declared_size - header_reserved)
mapped_time = ratio * duration
snapped_time = snap_back_to_gop(mapped_time)
```

相关设计：

- `Content-Length` 在同一会话内保持稳定。
- EWMA 码率估计只影响后续会话，不在当前响应中改 declared size。
- 通过 `X-Passthrough-Estimated-Size`、`X-Passthrough-Estimated-Bps`、`X-Passthrough-Seek-Ratio`、`X-Passthrough-Seek-Raw-Time`、`X-Passthrough-Seek-Gop` 输出诊断。

数学本身可行，但只解决“byte -> time”的近似映射，不解决“这个 byte offset 在容器里是否真实存在”的文件语义问题。

### 3.5 probe / prefix / tail 处理

实现过多类 Range 特判：

- `bytes=0-`
  - 启动请求，按 200 全量启动处理。
- header-only range
  - 落在 `[0, header_reserved)` 内的 bounded Range 必须从真实 prefix cache 读取。
- header-crossing range
  - 例如 `bytes=589824-`，start 在 2MB header 区内但 end 到 EOF。
  - 后来改为先拼接 cached prefix slice，再从新 producer 丢弃 `header_reserved` 字节后继续输出。
- tail probe
  - nPlayer 会发接近 EOF 的小范围 Range。
  - 曾用零字节 body 快速返回，避免启动 GPU producer。

这些处理解决了一些 503、自动跳下一项、prefix 不一致问题，但最终无法解决播放器对容器索引的要求。

### 3.6 并发与清理修复

本轮顺带修复/强化了一些与 seek 高并发有关的问题：

- 同 client 新 seek/live 请求允许抢占自己旧的 stale stream，避免旧流占住 Matter/GPU slot。
- `_replace_active_slot(..., close_on_failure=stream)` 保证 Matter release 前先 close stream，避免 worker 还在用 Matter 时池子复用。
- StreamingResponse finally 里尽早 close/release active slot，避免 Starlette cancellation 后留下假 busy。
- route 被 profile 拒绝时返回 403，总开关关闭时返回 404。

这些修复对未来仍有价值，即使 seekable passthrough 本身放弃。

## 4. Review 中修过的重要问题

### 4.1 DLNA OP/FLAGS

初版 seek 广告用过：

```text
DLNA.ORG_OP=01
```

review 指出 `OP=01` / `OP=10` 的方向容易混淆。新 seek route 既接受 byte Range，又接受 `TimeSeekRange.dlna.org`，所以改成：

```text
DLNA.ORG_OP=11
```

后续又发现：

```text
DLNA.ORG_FLAGS=617000...
```

包含 `lop-npt` / `lop-bytes` limited operation 位，与 `OP=11` 的 full random access 语义矛盾。最后 seek-only flags 改成：

```text
DLNA.ORG_FLAGS=01F00000000000000000000000000000
```

这去掉了 lop 位，保留 file-like transfer bits，并加入 interactive transfer bit。

经验：**DLNA header 可以影响一些客户端 UI，但不能覆盖容器本身的文件/直播属性。**

### 4.2 probe cache O(N^2)

初版每个 chunk 都把 `probe_prefix` 转成 bytes 写入全局 `_probe_cache`，导致 2MB/16MB 前缀填充时有大量重复 memcpy 和锁内扫描。

修正：

- seek prefix cap 改为 `PT_PASSTHROUGH_SEEK_HEADER_BYTES`。
- 写入从每 chunk 改为分段或最终写入。
- legacy `/passthrough` 也做了类似降低写入频率的修复。

经验：probe cache 是全局共享资源，任何 per-chunk 全量 copy 都会放大为并发性能问题。

### 4.3 DLNA seek 项不能替换 live

一轮 review 发现：如果打开 `PT_PASSTHROUGH_SEEK_DLNA=1` 后用 seek 项替换 live 项，那么被 profile 拦截的播放器会在目录里完全失去可播入口。

修正为 seek/live 并存。

经验：实验入口必须 additive，不应覆盖稳定 fallback。

### 4.4 transferMode / Cache-Control / availableSeekRange

逐步调整过：

- `transferMode.dlna.org: Streaming` -> `Interactive`
- `Cache-Control: no-store` -> `no-cache`
- 增加标准 `availableSeekRange.dlna.org`，不只用 `X-AvailableSeekRange.dlna.org`
- 503 prefix cache not ready 增加 `Content-Type`
- `.mp4` 原始后缀返回 `video/MP2T` 容易混淆，改成 `.seek.ts` / `.seek.mp4`

这些都是正确清理，但没有改变最终结论。

## 5. 实机测试时间线与现象

### 5.1 开启 seek 入口

最初通过 UI settings 开启：

```json
"passthrough_seek_enabled": true,
"passthrough_seek_dlna": true,
"passthrough_seek_route_policy": "all"
```

用户特别确认不能用 `run_server.bat`，因为大量配置来自 UI。后续测试均以 UI 启动为准。

### 5.2 503 启动失败

现象：

- 播放器进入前就出现 503。
- 播放器尚未出现进度条。

日志分析：

- 多个 seek/live startup 流占住 active slot 或 Matter pool。
- 部分流在客户端断开后没有足够早释放 active slot。

修复：

- same-client seek/live 可以抢占 stale stream。
- finally 中先 close/release active slot。
- Matter 获取设置 timeout。

结果：

- 503 busy 类问题减少。
- 但播放器 UI/seek 语义仍失败。

### 5.3 自动跳到下一个 item

现象：

- 播放两个 seek 链接会自动跳到下一个。

日志模式：

- nPlayer 启动后发多个 tail open Range。
- 早期 tail probe 触发靠近片尾的短 producer，只生成最后约一秒内容。
- 随后 prefix cache 对 `bytes=65536-` 返回短切片，播放器视为 EOF，于是跳下一个 DLNA item。

修复：

- tail probe ceiling 提到 2MB。
- prefix cache 不再返回“不完整的短 slice”冒充完整 open Range。
- header-crossing open Range 不再只回短 prefix cache。

结果：

- 自动跳下一个问题得到缓解。
- 但没有让播放器出现正常文件时间轴。

### 5.4 MPEG-TS 无进度条

抓包示例：

```http
GET /passthrough_seek/...mp4.seek.ts?mode=green&ptv=7
User-Agent: nPlayer/3.0

HTTP/1.1 200 OK
content-type: video/MP2T
transfermode.dlna.org: Interactive
contentfeatures.dlna.org: DLNA.ORG_PN=HEVC_TS_NA_ISO;DLNA.ORG_OP=11;DLNA.ORG_CI=0;DLNA.ORG_FLAGS=01F000...
accept-ranges: bytes
content-length: ...
timeseekrange.dlna.org: npt=0.000-...
availableSeekRange.dlna.org: 1 npt=0.000-...
```

结果：

- nPlayer/Skybox 仍显示 live-like UI。
- 没有普通文件进度条。
- 可拖动时也会卡住或触发大量 Range probe。

关键判断：

- nPlayer 的播放请求是裸 GET：

```http
Accept: */*
User-Agent: nPlayer/3.0
```

它没有发送：

```http
getcontentFeatures.dlna.org: 1
transferMode.dlna.org: Interactive
TimeSeekRange.dlna.org: ...
```

这说明 nPlayer 很可能已经把 URL 当普通 HTTP resource 播放，而不是严格走 DLNA AVTransport 的 time-seek 流程。此时 DLNA headers 对 UI 的影响有限，它更看容器 MIME 和 demuxer 能否给出 seek map。

结论：

**TS 容器本身过强地暗示 live/streaming。**裸 HTTP + MPEG-TS 即使有 `Content-Length` 和 Range，在 nPlayer/Skybox 上仍可能没有普通文件时间轴。

### 5.5 fMP4 尝试

为验证“播放器是不是只认 MP4 文件体验”，新增：

```json
"passthrough_seek_container": "mp4"
```

响应变为：

```http
content-type: video/mp4
contentfeatures.dlna.org: DLNA.ORG_PN=HEVC_MP4_MAIN;DLNA.ORG_OP=11;DLNA.ORG_CI=0;...
url: /passthrough_seek/...mp4.seek.mp4?mode=green&ptv=7
```

PyNv mux 使用 true fMP4：

```text
-movflags +frag_keyframe+empty_moov+default_base_moof
-frag_duration 100000
-f mp4 -
```

测试现象：

- 第一段无 Range 请求返回 200。
- nPlayer 立即发第二段 Range，例如：

```http
Range: bytes=589824-
```

早期实现会从新 fMP4 producer 丢弃 589824 字节后返回。这违反“同一 MP4 representation 字节稳定”要求，播放器卡住。

### 5.6 fMP4 prefix crossing 修复

针对 `bytes=589824-` 这类落在前缀区内但延伸到 EOF 的请求，实现：

- 等待完整 prefix cache。
- 先返回 cached prefix slice。
- 再启动新 producer，从头丢弃 `header_reserved` 字节后继续返回。

日志确认新分支生效：

```text
prefix crossing cache hit
first chunk: source=prefix-splice
```

但 fMP4 仍无法播放，nPlayer 继续发：

```text
bytes=131072-
bytes=786432-
bytes=2097152-
tail probe
middle/tail open range
```

问题变得更明确：

- fMP4 prefix 修复只能保证头部某段一致。
- `bytes=2097152-` 刚好在 prefix 边界之后，仍然需要后续字节也属于同一份稳定 MP4。
- 新 producer 输出的是“另一个 fMP4 representation”的片段，不是原响应中 offset=2097152 的真实字节。
- nPlayer 对 MP4 是按真实文件索引和 byte offset 读取，不接受动态拼接的“近似 MP4”。

结论：

**在线 fMP4 fragment stream 不是普通 MP4 文件。**

### 5.7 回退 TS 并修正 DLNA flags

外部建议指出 `OP=11` 与 `6170...` flags 中的 lop bits 矛盾。于是：

- active UI config 从 `mp4` 回退到 `mpegts`。
- seek-only flags 改为 `01F0...`。

结果：

- header 更规范。
- 但 TS 仍被播放器认为是 live-like stream，没有普通文件进度条。

这说明问题已不在 DLNA flags 单点。

### 5.8 真实 MP4 抓包对照

用户抓到 nPlayer 播放真实源 MP4 的成功路径：

```http
GET /media/.../4k2.me%40aquga00010_1_8k.mp4

HTTP/1.1 200 OK
accept-ranges: bytes
content-type: video/mp4
content-length: 4178713040
```

随后：

```http
Range: bytes=4178706432-

HTTP/1.1 206 Partial Content
content-range: bytes 4178706432-4178713039/4178713040
content-length: 6608
content-type: video/mp4
```

又继续读靠前/中间的 Range：

```text
bytes=4178640896-
bytes=1507328-
bytes=308871168-
bytes=681705472-
```

解释：

- nPlayer 会读 MP4 尾部查 `moov` 或验证 box。
- 然后根据 `moov` sample table 请求中间位置。
- 它不是简单按 `Content-Length` 线性估算时间。

这份抓包最终推翻“只要伪装成 video/mp4 + Range 就行”的设想。

## 6. 根因分析

### 6.1 HTTP Range 的根本约束

HTTP Range 是对“同一个 selected representation”的字节切片请求。

如果同一 URL 第一次返回的第 N 字节和第二次 Range 请求第 N 字节不是同一个字节，严格播放器就有权失败。

我们的在线生成路线做不到这个不变量：

- 每次 Range 都会启动新的 PyNv/FFmpeg producer。
- 新 producer 从某个时间点重新编码。
- 重新编码后每帧大小不稳定，容器 offset 不稳定。
- 即使用 CBR，也无法保证每个 sample/chunk 的字节大小与先前响应一致。

这不是 header 层能解决的问题。

### 6.2 MPEG-TS 失败原因

MPEG-TS 的优点：

- 可从任意 PAT/PMT + IDR 附近重新开始解码。
- 对 live streaming 友好。
- 不需要全局 moov 索引。

MPEG-TS 的失败点：

- 裸 `video/MP2T` 被播放器天然视作 stream/live。
- TS 没有 MP4 那种容器级 duration/index。
- nPlayer/Skybox 的 UI 决策很可能优先看 MIME/demuxer 类型，不完全信 DLNA headers。
- 即使字节映射大致正确，播放器也不一定展示普通文件进度条。

因此 TS 更适合继续做 `/passthrough_live`，不适合作为 nPlayer 的普通文件式 seek 体验。

### 6.3 fMP4 失败原因

fMP4 的优点：

- MIME 是 `video/mp4`。
- 能流式输出 `empty_moov + moof/mdat`。
- 有机会被部分播放器当 MP4 处理。

fMP4 的失败点：

- 在线 fMP4 是 fragment stream，不是完整静态 MP4 文件。
- nPlayer 会发 tail/mid Range 查 MP4 box/index。
- `empty_moov` 不包含完整 sample table。
- 每次 Range 启动新 fMP4 producer，相当于返回另一份 MP4 的局部字节。
- prefix splice 只能修补头部，不能让整份虚拟文件字节稳定。

因此 fMP4 也不能靠简单拼接实现普通文件 seek。

### 6.4 不能复用原始 MP4 索引

曾讨论“用真实源 MP4 的信息作为返回 MP4 索引，再给处理后透视内容”。

结论：不能直接复用。

MP4 `moov` 内包含的不只是时长：

- `stsz`: 每个 sample 大小。
- `stco/co64`: 每个 chunk 在文件内的真实字节偏移。
- `stss`: 关键帧表。
- `stts/ctts`: 解码/显示时间戳。
- `hvcC` / `avcC`: codec initialization data。
- audio/video interleave 关系。

处理后 passthrough 视频会重新编码：

- 码流从源 HEVC Main10 可能变成 8-bit HEVC。
- green/alpha compositing 会改变每帧复杂度和大小。
- GOP/IDR 位置可能不同。
- 音频可能 copy、AAC transcode 或重新对齐。
- mux 后 chunk offset 完全不同。

所以原始 MP4 的 sample table 指向的是“源文件字节”，不能用来索引“处理后视频字节”。

唯一可复用的是 metadata 级信息：

- duration
- fps
- resolution
- color metadata
- audio stream 参数
- 估算码率的冷启动参考

不能复用 byte-level index。

## 7. 哪些尝试是有效资产

虽然目标失败，本轮留下了有价值的资产。

### 7.1 隔离式实验开关

`PT_PASSTHROUGH_SEEK_ENABLED` / `DLNA` / `ROUTE_POLICY` / `CONTAINER` 这套开关证明很有必要：

- 可以灰度。
- 可以按 UI 启动测试。
- 可以快速回滚。
- 不影响旧 `/passthrough_live`。

后续任何类似实验都应继续使用这套门控方式。

### 7.2 request_history 与日志诊断

`debug_output/request_history/request_history_20260528.jsonl` 对本轮判断非常关键：

- 能区分 `/thumb`、`/media`、`/passthrough_seek`。
- 能看到 UA、Range、status、content_length、elapsed_ms。
- 能确认 nPlayer 同时请求 seek item 和原始 `/media`。
- 能比较真实 MP4 与 seek TS/fMP4 的 Range 模式。

经验：播放器兼容问题必须以真实请求历史为准，不能只看 header 理论。

### 7.3 DLNA additive 暴露

seek/live 并存是正确模式。后续实验入口应始终 additive，不替换稳定入口。

### 7.4 Matter/active slot 清理

seek 高并发 probe 暴露了 active slot/Matter release 顺序问题。本轮的 close-before-release、same-client preempt、finally 早释放等修复是通用稳定性资产。

### 7.5 prefix cache 不变量

“与 `[0, header_reserved)` 相交的 Range 必须返回真实、稳定、逐字节一致的 prefix”是正确不变量。

即使本轮放弃，未来如果实现虚拟 MP4 或其他伪文件，也必须从这个原则开始。

## 8. 哪些尝试被证明无效

### 8.1 只靠 header 让 TS 变成普通文件

尝试过：

- `transferMode.dlna.org: Interactive`
- `OP=11`
- `CI=0`
- flags 从 `6170...` 改到 `01F0...`
- `Content-Length`
- `Accept-Ranges`
- `TimeSeekRange.dlna.org`
- `availableSeekRange.dlna.org`
- `Cache-Control: no-cache`

结果：nPlayer/Skybox 仍把 TS 当 live/stream，不显示普通时间轴。

结论：header 不能逆转 TS 容器在播放器 demuxer 里的 live-like 判断。

### 8.2 用 URL 后缀/MIME 伪装容器

`.mp4` URL 返回 `video/MP2T` 可能更糟，后来改成 `.seek.ts` / `.seek.mp4`。

但事实证明：

- `.seek.ts` 仍是 TS。
- `.seek.mp4` 如果 body 不是普通静态 MP4，也不会通过 nPlayer 的 Range 探测。

结论：URL 后缀只能减少混淆，不能提供文件语义。

### 8.3 fMP4 fragment stream 当普通 MP4 文件

fMP4 是真实 MP4 bytes，但不是静态 MP4 文件。

播放器如果只顺序播放，fMP4 可能可用；播放器如果按文件 Range 读 `moov`/sample table，fMP4 实时流不满足。

结论：fMP4 不等于可随机访问 MP4 文件。

### 8.4 tail probe 返回零字节

tail probe 返回 0 可以避免 GPU 被尾部探测点燃，但对 MP4 文件语义是错的：

- 真实 MP4 tail 通常有 box、moov 或 metadata。
- 返回全 0 会让播放器无法解析尾部结构。

这个策略只适合“不希望触发 producer 的 TS probe”，不适合 MP4 伪装。

### 8.5 byte->time 线性映射作为唯一 seek map

线性映射对 TS 近似跳转有用，但对 MP4 文件 seek 不够。

MP4 seek 需要 sample table 和 byte offsets，不是简单 CBR ratio。

## 9. 未来第二次尝试的可行路线

### 9.1 推荐路线 A：离线/近线真 MP4 缓存

这是最现实路线。

设计：

1. 用户在 DLNA 里看到 seekable MP4 item 之前，系统必须已经有处理后的完整 MP4 文件。
2. 该文件由现有 offline passthrough 生成链路产生：
   - `tools/offline_passthrough.py`
   - `tools/offline_alpha_passthrough.py`
   - `offline/convert.py`
3. DLNA 直接暴露生成后的 MP4，或 `/passthrough_seek` 在 `container=mp4` 时转发到缓存文件。
4. HTTP 返回直接复用 `_file_range_response()` / `/media` 语义：
   - `Content-Type: video/mp4`
   - `Accept-Ranges: bytes`
   - 真实 `Content-Length`
   - 真实 `Content-Range`
   - 真实文件 bytes
5. 没有缓存时：
   - 不暴露 seek MP4 item，或
   - 返回 404/503 明确提示需要先生成，或
   - 触发后台生成但在完成前不伪装成可播放 MP4。

优点：

- 与 nPlayer 已验证的真实 MP4 路径一致。
- 不需要欺骗播放器。
- Range、tail probe、middle probe 全部自然正确。

缺点：

- 不再是即时实时播放。
- 需要磁盘空间。
- 需要生成进度 UI、缓存清理、命名和失效策略。

### 9.2 可行路线 B：近线首段/全片生成后切换

折中方案：

- 用户点击后先展示“生成中”状态。
- 后台快速生成完整 MP4 或至少完整可索引分段集。
- 完成后 DLNA/HTTP resource 变成真实文件。

这仍然要求完成前不能冒充普通 MP4。可以返回 202/503/Retry-After 或仅在目录中暴露“生成中”状态。

### 9.3 高复杂路线 C：真正虚拟 MP4 文件系统

只有在必须“边生成边支持普通 MP4 Range”时才考虑。

最低要求：

- 预先生成或预测完整 `moov`。
- 对每个 video/audio sample 确定：
  - duration
  - size
  - chunk offset
  - keyframe
  - codec config
- Range 请求任意 offset 时能返回与 `moov` 一致的真实 bytes。
- 生成内容必须可复读、可缓存、可拼接。
- 处理音频 interleave。
- 处理 tail moov 或 faststart moov。
- 处理播放器读取尾部/中部小 Range。

可能实现形态：

- CMAF/fMP4 segment store + `sidx`，但客户端是否按普通 MP4 文件接受未知。
- 固定 GOP segment，每段落盘，生成全局 index。
- 先快速预编码一遍统计 sizes，再生成 moov，再服务。这个实际上已接近离线生成。

风险：

- 实现复杂度高。
- debug 难度高。
- 与现有实时目标冲突。

暂不建议投入。

### 9.4 不推荐路线：继续调 DLNA headers

本轮已经验证：

- DLNA header 可以修正语义矛盾。
- 但 nPlayer 的裸 GET 很可能不走严格 DLNA seek 流程。
- 容器/demuxer 判断优先级更高。

继续调 `OP`、`FLAGS`、`transferMode`、`Cache-Control` 的收益很低。

## 10. 未来测试必须收集的证据

若第二次尝试重启，应先做以下判定实验。

### 10.1 真 MP4 baseline

已观察到真实 MP4 成功模式。后续需要保存更多播放器样本：

- nPlayer
- Skybox
- HereSphere
- 4XVR
- DeoVR
- VLC/mpv

记录：

- 首次 GET/HEAD。
- tail Range。
- mid Range。
- 是否显示总时长。
- 是否拖动后继续播放。

### 10.2 离线 passthrough MP4 baseline

用现有 offline 生成 green/alpha MP4，然后通过 `/media` 播放。

如果这个文件 nPlayer 有进度条并能拖动，说明“真 MP4 缓存路线”成立。

如果离线输出 MP4 也失败，则问题转向：

- MP4 mux flags。
- codec/profile。
- 8K HEVC Main/Main10 兼容性。
- audio copy/AAC。
- VR 文件命名。

### 10.3 不要再用 fake MP4 单点测试

以下测试没有决策价值：

- `Content-Type: video/mp4` 但 body 是 TS。
- URL 后缀 `.mp4` 但 body 是 TS。
- tail Range 返回 0。
- 用原 MP4 moov 指向处理后 bytes。

这些都会污染判断。

## 11. 当前代码与配置状态

### 11.1 已回滚运行开关

当前 `runtime_cache/ui_settings.json`：

```json
"passthrough_seek_enabled": false,
"passthrough_seek_dlna": false,
"passthrough_seek_route_policy": "all",
"passthrough_seek_container": "mpegts"
```

UI 启动后不应再暴露 `/passthrough_seek`。

### 11.2 代码中保留的实验资产

代码中仍保留：

- `/passthrough_seek` route。
- seek 配置项。
- DLNA additive seek item 逻辑。
- TS/fMP4 suffix 支持。
- prefix cache / tail probe / diagnostics。
- tests。

这些资产可作为后续研究基础，但不应在生产/日常使用中开启。

### 11.3 稳定 fallback

稳定路径仍是：

- `/passthrough_live`
- live chapter container
- `/media` 原始文件静态 Range
- offline generated passthrough MP4（如已有）

## 12. 本轮验证命令

本轮最后相关验证包括：

```powershell
python -m py_compile config.py http_app/routes_media.py dlna/content_directory.py ui/settings.py
.\.venv\Scripts\python.exe -m pytest tests/test_routes_media_cache.py tests/test_content_directory_modes.py tests/test_settings.py tests/test_config_defaults.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

最后完整测试结果：

```text
336 passed, 2 skipped, 44 subtests passed
```

## 13. 一句话经验

这次最大的经验是：

**播放器的进度条不是由 `Content-Length` 和几个 DLNA header 决定的，而是由“容器索引是否真实、Range 字节是否稳定、demuxer 是否信任这是文件”共同决定的。实时生成流可以做 live，也可以做章节跳转，但不能低成本伪装成普通 MP4 文件。**

