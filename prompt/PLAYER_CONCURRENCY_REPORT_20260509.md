# 播放器播放流程并发控制报告

日期：2026-05-09

## 1. 背景

当前 passthrough 实时转码链路默认只允许 1 路生产流：

- 配置项：`PT_PASSTHROUGH_MAX_CONCURRENT`，默认 `1`。
- 等待窗口：`PT_PASSTHROUGH_BUSY_WAIT_SEC`，默认 `10` 秒。
- 超限结果：等待后仍无可用槽位时返回 `503` 和 `Retry-After`。

这样设计不是因为 HTTP 层不能并发，而是因为 passthrough 链路里有多个共享或易阻塞资源：

- 全局 `Matter/ONNX` 单例及 RVM recurrent state；
- CuPy/CUDA 临时缓冲和共享 CUDA stream；
- PyNv decoder/encoder、NVENC/NVDEC 会话；
- FFmpeg mux 管道；
- HTTP 客户端读取速度导致的 backpressure；
- 播放器启动时的 Range/probe/side request 风暴。

并发控制目标是：只让“真正播放流”占用 PyNv/GPU 生产链路，尽量把探测请求、重复请求、副请求挡在启动 GPU 工作之前。

## 2. 全局并发槽位机制

核心代码在 `http_app/routes_media.py`：

- `_active_streams` 记录当前占用 passthrough 槽位的对象。
- `_take_active_slot()` 负责申请槽位、等待、同设备预占、最终 busy 拒绝。
- `_release_active_slot()` 释放槽位并记录剩余 active 数。
- `_replace_active_slot()` 把启动占位 token 替换成真实 stream 或 `LiveSession`。

owner 设计：

- pseudo-VOD：`(path, client_ip)`。
- live：`("live", client_ip, live_profile)`。

live profile 用于区分同一设备上的不同播放器行为：

- `nplayer`：nPlayer 专用 live 双请求模型。
- `libmpv`：Skybox/libmpv 类。
- `vlc`：VLC/LibVLC/MoonVR 类。
- `lavf`：FFmpeg/Lavf 副请求。
- `default`：未知客户端，当前默认按 VLC 风格处理。

预占规则 `_can_preempt_owner()`：

- 相同 owner 可预占，主要用于同设备重试释放旧流。
- 同一客户端的新 `/passthrough_live` 真实播放请求可预占旧 live 流。
  - 当前产品按单人使用假设设计；
  - 用户切章节、切文件、回退再播放时，播放器不一定会向服务器发送明确 stop；
  - 因此不能依赖旧连接自然断开或 TTL 到期。
- 同设备上 `vlc/libmpv/default` 可预占已有 `lavf`。
- `libmpv` 新请求可较宽松地接管同设备旧 live 流。
- 不同设备之间不能互相预占。

## 3. `/passthrough_live` 按播放器分支

### 3.1 libmpv / Skybox

识别方式：

- User-Agent 包含 `libmpv` 或 `skybox`。
- profile 为 `libmpv`。

已观察行为：

- 常发 `Range: bytes=0-`。
- 可能启动后很快断开、重试、再次请求同一 URL。
- 曾出现收到初始 MPEG-TS 数据后停止读取，导致 reader queue 满、active slot 被占住。

控制策略：

- 走共享 `LiveSession` 路径。
- 启动真实 PyNv stream 后，先等待第一个非空 TS chunk，再返回 HTTP 响应。
- `LiveSession` 保存前缀缓存，重复请求可订阅同一个生产流，不启动第二个 PyNv。
- `PASSTHROUGH_LIVE_CACHE_BYTES` 控制缓存大小，默认 128MB。
- `PASSTHROUGH_LIVE_CACHE_TTL_SEC` 控制无订阅者后保留时间，默认 10 秒。
- 每个订阅者有独立队列，队列满时丢弃该订阅者，避免一个慢客户端阻塞共享生产流。
- stall watchdog 可在输出无进展时关闭流；但当前 `run_server.bat` 将 `PT_PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC=0`，用于避免 VLC pseudo-VOD 测试阶段误杀。

返回形态：

- 通常 `200 OK`。
- `contentFeatures.dlna.org` 使用 `DLNA.ORG_OP=10`。
- 不走 VLC pseudo-VOD 的 `Content-Length/Content-Range` 伪文件响应。

并发收益：

- 多个 libmpv/Skybox 重复启动请求复用同一条生产链路。
- 避免每个 probe 都启动 PyNv，减少 503 和抢占。

风险：

- 如果播放器停止读取但 TCP 未快速断开，仍可能占用槽位。
- `LiveSession` grace period 内 GPU 仍可能继续生产，需要继续验证 TTL 和 stall 策略。

### 3.2 VLC / LibVLC / MoonVR

识别方式：

- User-Agent 包含 `vlc`、`libvlc` 或 `moonvr`。
- profile 为 `vlc`。

已观察行为：

- MoonVR/LibVLC 会发真实 VLC 请求，也可能夹杂 `Lavf/58.45.100` 副请求。
- 某些请求会带非起点 Range，例如 `bytes=564-` 或巨大 tail range。
- 曾出现服务端持续输出 TS，但播放器音频有、视频无，或 loading。
- AAC、MPEG-TS 时间戳、HEVC PID 识别、初始数据量都影响启动成功率。

控制策略：

- 默认走直接 stream 路径，不使用共享 `LiveSession`。
- 非起点 Range 在启动 GPU 前返回 `416`：
  - 条件：非 `libmpv`、非 `nPlayer`、带 Range 且不是 `bytes=0-`。
  - 目的：防止 side/probe range 启动新的 PyNv 或预占真实播放流。
- 支持 VLC pseudo-VOD：
  - 配置 `PT_PASSTHROUGH_LIVE_VLC_PSEUDO_VOD=1` 时返回 `206 Partial Content`。
  - 添加合成 `Content-Length` 和 `Content-Range`。
  - Skybox/libmpv 不受此开关影响。
- 首包门控：
  - 先从 PyNv stream 取到第一个非空 TS chunk。
  - 取不到则返回 `504 first chunk timeout` 或 `503 no data`，避免 headers 已发但 body 不来。
- VLC preroll：
  - `PT_PASSTHROUGH_LIVE_VLC_PREROLL_BYTES` 当前在 `run_server.bat` 为 `1048576`。
  - 目标是在响应前攒够一段 TS 数据，提升 PAT/PMT/HEVC 初始化成功率。
- 音频策略：
  - 全局 `PT_PASSTHROUGH_AUDIO_MPEGTS=aac`。
  - VLC 可用 `PT_PASSTHROUGH_AUDIO_MPEGTS_VLC=off` 强制视频-only 作为故障隔离。
  - 当前 `run_server.bat` 使用 `pipe_ts`、AAC cache、PAT/PMT at frames、HEVC AUD。
- 同 owner 预占：
  - 后续 VLC 请求可预占同设备旧 VLC 流，减少旧连接未释放导致的 503。

返回形态：

- 当前 `run_server.bat` 设置 `PT_PASSTHROUGH_LIVE_VLC_PSEUDO_VOD=1`，因此 VLC/MoonVR live 通常返回 `206`。
- 非 pseudo-VOD 时返回 `200`，`Accept-Ranges: none`。
- pseudo-VOD 时返回 `Accept-Ranges: bytes`、合成长度和范围。

并发收益：

- 把 MoonVR 的非起点 Range 和 Lavf 副请求挡在 GPU 前。
- 真正 VLC 播放请求可以接管同设备旧请求，避免长期 503。
- 首包/preroll 减少“占了槽但播放器没收到可用初始化数据”的情况。

风险：

- pseudo-VOD + live 生成在响应完整性上更复杂，旧 stall watchdog 曾导致 ASGI 未完整响应、active slot 未及时释放。
- 当前关闭 live stall watchdog 后，如果播放器停止读取但连接不关闭，仍需要依赖断开检测和后续预占释放。

### 3.3 Lavf / FFmpeg side request

识别方式：

- User-Agent 包含 `lavf/`。
- profile 为 `lavf`。

已观察行为：

- MoonVR/LibVLC 可能在真实播放请求前后发 Lavf 请求。
- 这些请求可能是缩略图、探测或内部媒体分析，不一定是真正播放。
- 如果允许它们启动 PyNv，会抢占单槽位，导致真实 VLC 播放等待或 503。

控制策略：

- 配置项：`PT_PASSTHROUGH_LIVE_LAVF_POLICY`。
- 可选值：
  - `reject`：总是拒绝 Lavf live 请求。
  - `active_only`：只有同设备已有 VLC/default stream 时拒绝。
  - `allow`：当普通 live 客户端处理。
- 当前 `run_server.bat` 设置为 `reject`。
- 拒绝时在启动 GPU 前返回 `409` 和 `Retry-After: 1`。
- 如果允许进入并发槽位，后续真实 `vlc/libmpv/default` 请求可预占同设备 Lavf。

并发收益：

- 避免 Lavf 副请求冷建 AAC cache 或启动 PyNv。
- 降低 MoonVR “重试有时成功”的随机性。

风险：

- 如果某个播放器真正依赖 Lavf UA 承载播放流，`reject` 会阻断播放。当前策略主要针对 MoonVR 已观察行为。

### 3.4 nPlayer / OPlayer 类

识别方式：

- nPlayer：User-Agent 包含 `nplayer`。
- OPlayer 未单独识别，主要通过通用路径和历史兼容逻辑受益。

已观察行为：

- pseudo-VOD 中会发大量 open byte ranges，例如 `bytes=262144-`、`bytes=524288-`。
- 如果这些 Range 都启动新 stream，会预占或关闭主流。
- live 中也可能发重复 startup 请求。

控制策略：

live：

- nPlayer 现在使用独立 `nplayer` live profile。
- 已观察到 nPlayer 对同一条 `/passthrough_live` URL 会连续发两个完全相同的 GET：
  - 第一条是探针，请求视频基本信息和初始 TS；
  - 第二条才是实际获取直播数据流。
- nPlayer 走 managed `LiveSession` 路径，而不是 VLC/default 的直接 stream 路径：
  - 第一条请求启动 PyNv，拿到首个非空 TS chunk 后创建 `LiveSession`；
  - `LiveSession` 缓存首包和后续前缀；
  - 第二条相同请求命中 `LiveSession` 后直接订阅缓存/生产流，不再启动第二条 PyNv；
  - 这样避免第二条请求预占第一条 probe 流并触发 `503 no data`。
- nPlayer 的 live Range 不参与 `LiveSession` key，避免同一播放因为 Range 变化生成多个 key。
- `_live_starting` 记录同一 live key 的启动中状态：
  - 第二条请求如果早于 `LiveSession` 创建，会短暂等待 session 出现；
  - 等待窗口与 live 首包超时一致，避免 AAC cache 或首包生成较慢时第二条过早失败；
  - 如果等待窗口内仍未出现，才返回 `409 duplicate startup`；
  - 目的：避免重复启动 PyNv。
- nPlayer 同设备新 live 请求可以预占旧 nPlayer live session：
  - 日志显示 nPlayer 在同一设备上快速从 `t=360` 切到 `t=540` 或另一个文件；
  - 旧 `LiveSession` 即使没有订阅者，也会在 TTL 内继续占用唯一 active slot；
  - 因此同 owner 的 `nplayer` 新请求需要能关闭旧 session，而不是等待 10 秒后返回 `503 busy`。

pseudo-VOD：

- 小 Range probe 走合成或缓存响应，不启动 stream。
- tail probe 返回空填充体，不启动 stream。
- 非零 open range 会先等待 prefix cache：
  - 命中则从缓存返回。
  - 未命中且还在小前缀范围内，返回 `503 prefix-cache-not-ready`，不启动新 stream。

并发收益：

- nPlayer/OPlayer 的多 Range 探测不再轻易启动第二个 PyNv。
- 主播放流能持续填充 prefix cache，副请求尽量从缓存拿。

风险：

- 如果 prefix cache 尚未准备好，播放器收到 `503 Retry-After: 1` 后是否重试取决于客户端实现。
- nPlayer 的 live 逻辑仍需要实机验证：第二条相同 GET 应出现 `live cache hit` 或 `nPlayer duplicate startup joined cache`，不应再进入 `_take_active_slot()` 启动第二条 PyNv。
- 切换章节/文件时应出现 `passthrough preempt previous range`，旧 nPlayer session 被关闭，新请求不应再因为旧 TTL session 返回 `503 busy`。

### 3.5 unknown/default 客户端

识别方式：

- 不匹配 libmpv、VLC/MoonVR、Lavf 时使用 `PT_PASSTHROUGH_LIVE_DEFAULT_PROFILE`。
- 当前默认值为 `vlc`。

控制策略：

- 默认按 VLC 直接 stream 处理。
- 不使用 `LiveSession`，避免未知客户端被 libmpv/Skybox 的 shared cache/busy-lock 行为影响。
- 非起点 Range 仍会在启动 GPU 前返回 `416`。

并发收益：

- 未知客户端不会因共享 live session 策略而卡在缓存/订阅逻辑里。
- 更保守地避免非起点 Range 启动 GPU。

风险：

- 如果未知客户端行为更像 libmpv，可能无法从 `LiveSession` 前缀缓存受益。

## 4. `/passthrough` pseudo-VOD 的并发保护

虽然当前 DLNA catalog 主要暴露 `/passthrough_live`，旧 `/passthrough` 仍有大量并发保护：

- unsatisfiable Range：返回 `416`。
- tail probe：返回指定长度的空 body，不启动 PyNv。
- small probe：返回缓存或合成 MP4 header，不启动 PyNv。
- 非零 open range：
  - 先等 prefix cache；
  - 命中则从缓存返回；
  - 未命中且在 cache limit 内，返回 `503 prefix-cache-not-ready`，不启动新流。
- 真正需要生产时才申请 `_take_active_slot()`。
- 超限返回 `503 passthrough busy`。

这套逻辑主要针对 nPlayer/OPlayer 一类会密集发 Range 的播放器，避免每个 Range 都变成一条新实时转码链。

## 5. 当前运行配置中的关键点

`run_server.bat` 当前关键并发/播放器相关设置：

- `PT_PASSTHROUGH_MAX_FPS=30`
- `PT_PASSTHROUGH_HEVC_BITRATE=50M`
- `PT_PASSTHROUGH_AUDIO_MPEGTS=aac`
- `PT_PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE=pipe_ts`
- `PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE=1`
- `PT_PASSTHROUGH_AUDIO_MPEGTS_PAT_PMT_AT_FRAMES=1`
- `PT_PASSTHROUGH_MPEGTS_HEVC_AUD=1`
- `PT_PASSTHROUGH_LIVE_VLC_PSEUDO_VOD=1`
- `PT_PASSTHROUGH_LIVE_LAVF_POLICY=reject`
- `PT_PASSTHROUGH_MPEGTS_COLOR_RANGE=tv`
- `PT_PASSTHROUGH_LIVE_FIRST_CHUNK_TIMEOUT_SEC=30`
- `PT_PASSTHROUGH_LIVE_VLC_PREROLL_BYTES=1048576`
- `PT_PASSTHROUGH_LIVE_VLC_PREROLL_TIMEOUT_SEC=3`
- `PT_PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC=0`
- `PT_ALPHA_STRIDE=3`

注意：`PT_PASSTHROUGH_LIVE_STALL_TIMEOUT_SEC=0` 是当前重要变量。它降低了误杀 VLC pseudo-VOD 响应的风险，但也意味着 stopped reader 的自动清理更依赖断开检测和同 owner 预占。

## 6. 503 / 409 / 416 / 504 的含义

- `503 busy`：真实 passthrough 槽位被占用，等待窗口后仍不可用。
- `503 prefix-cache-not-ready`：pseudo-VOD 的 Range probe 想读未生成前缀，服务端拒绝启动第二条流。
- `503 no stream data`：启动了流但响应前没有拿到任何数据。
- `409 preempted`：请求在启动过程中被同设备后续请求接管。
- `409 Lavf rejected`：Lavf 副请求被策略拒绝。
- `409 nPlayer duplicate startup`：nPlayer 重复 live 启动被去抖拒绝。
- `416`：Range 不满足，或非 libmpv live 的非起点 Range 被挡在 GPU 前。
- `504 first chunk timeout`：响应前等待首个 TS chunk 超时。

## 7. 下一步建议重点

1. 把日志按 profile 汇总：
   - `libmpv`、`vlc`、`lavf`、`nplayer`、`default` 分开统计。
   - 统计每次 503 前 active owner、stream bytes、frames、是否有 release log。

2. 明确每类播放器的“真实播放请求”判定：
   - UA、Range、DLNA headers、transfer mode、client IP、请求顺序。
   - 形成白名单/拒绝规则，避免继续通过试错堆分支。

3. 对 VLC pseudo-VOD 做完整性审查：
   - `Content-Length` 响应下任何主动关闭都可能产生 ASGI incomplete response。
   - 需要确认 slot release 是否总是在 close 前发生。

4. 重新评估 stall watchdog：
   - 对 chunked `200` live 和 pseudo-VOD `206 Content-Length` 应该分开策略。
   - 不能让 watchdog 清理变成 active slot 泄漏来源。

5. 为 Lavf policy 保留 per-client 覆盖：
   - MoonVR 当前适合 `reject`。
   - 其他 DLNA 客户端若真实播放 UA 是 Lavf，需要可配置例外。

6. 固化单人使用模型：
   - 同一客户端的新 live 播放请求应接管旧 live 流；
   - 不应等待旧 session 的 TTL，也不应因为旧播放页未显式关闭而返回 `503 busy`；
   - Lavf/probe/非起点 Range 仍应挡在 GPU 前，不能触发接管真实播放。

7. 在考虑多路 PyNv 并发前，先完成单真实播放流稳定性验证：
   - 所有 probe/side request 不启动 GPU；
   - 真流断开后 active slot 必然释放；
   - 播放器停止读取时可控清理。
