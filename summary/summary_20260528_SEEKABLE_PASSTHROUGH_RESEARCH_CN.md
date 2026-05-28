# 可拖动 Passthrough 播放方案研究综述（2026-05-28）

## 1. 背景与问题

当前实时 Passthrough 仅以 `/passthrough_live`（MPEG-TS 直播 + 章节式起点）方式
暴露给 DLNA 客户端。用户体验上，被迫从离散章节中挑选起点而无法拖动进度条，
在大多数 VR 播放器（4XVR、Skybox、DeoVR、HereSphere、nPlayer、VLC/mpv 等）里
体感笨拙。

核心约束：

- 几乎所有目标播放器对 DLNA `TimeSeekRange.dlna.org` 支持不可靠，实际拖动一
  律走 HTTP `Range: bytes=` 请求。
- HTTP Range 语义按字节定义在「同一份资源表示」内，而我们的输出是实时生成
  的非确定性字节流，没有真实的字节-时间索引。
- 不能动 `/passthrough_live`，那是已稳定的直播兜底路径。

研究结论：**可以做，并且项目内已有完整雏形**。但应定位为「伪 VOD / 虚拟 CBR
byte seek 兼容层」，独立于直播路径，且按播放器 profile 灰度开启。

---

## 2. 现有代码资产

研究发现 `/passthrough/{name}` 已经实现了字节→时间映射的全部基础设施，目前
仅在 DLNA 目录中被刻意隐藏（[dlna/content_directory.py:414](dlna/content_directory.py#L414)）。

可复用组件：

| 资产 | 位置 | 用途 |
|---|---|---|
| `_seek_from_byte_range()` | [routes_media.py:1186](http_app/routes_media.py#L1186) | `Range.start / total * duration` 映射 |
| `_estimated_passthrough_size()` | [routes_media.py:1039](http_app/routes_media.py#L1039) | 虚拟 `Content-Length` 估算 |
| `_estimated_passthrough_bps()` | [routes_media.py:1046](http_app/routes_media.py#L1046) | 发送限速参考 |
| `_range_416()` / `_range_unsatisfiable()` | [routes_media.py:1199](http_app/routes_media.py#L1199) | 越界响应 |
| `_passthrough_content_features()` | [routes_media.py:1246](http_app/routes_media.py#L1246) | DLNA `OP=01` byte-seek 标志 |
| `estimate_for_media()` + EWMA 持久化 | [utils/bitrate_estimator.py:232](utils/bitrate_estimator.py#L232) | 学习实际码率，写入 `debug_output/bitrate_estimates.json` |
| `record_actual_bps()` | [utils/bitrate_estimator.py:237](utils/bitrate_estimator.py#L237) | 流结束时回写实测值，EWMA `0.7·old + 0.3·new` |
| `_probe_cache` (16MB×N, 64MB total) | [routes_media.py:97](http_app/routes_media.py#L97) | 吃掉重复 probe 不点燃 GPU |
| `_is_small_probe_range / _is_zero_open_range / _is_tail_probe_range` | [routes_media.py:1164-1183](http_app/routes_media.py#L1164) | 三类 probe 分类已实现 |
| 客户端 UA profile 识别 | [utils/player_compat.py](utils/player_compat.py) | nplayer / 4xvr / avpro / libmpv / vlc / lavf / default |

---

## 3. 播放器 byte seek 的真实行为

外部资料（MDN HTTP Range requests / Range header、Android Media3 `HttpUtil`、
Media3 supported formats）+ 代码侧观察得到的现实：

1. **拖动不是简单的"时间→百分比"**。播放器先用容器索引（MP4 moov、TS PAT/PMT、
   或近似 CBR seek map）把目标时间换成**容器字节位置**，再用 HTTP Range 取该
   字节。因此服务端能否撒谎成功，取决于「容器层告诉播放器它在第 N 字节会看
   到什么」与「服务端实际返回的字节」是否还能被解码器吃下。
2. **三类典型请求**（项目已识别）：
   - `bytes=0-`（open range）：取头部 probe，读容器/编码信息。
   - `bytes=<tail>-`（>95%，<=512KB）：尾部 probe，MP4 客户端读 moov 或验大小。
   - `bytes=N-`（mid-stream open range）：真实拖动 seek。
3. **MP4 vs MPEG-TS 容差差异**：
   - MP4 强依赖 `moov` sample_table，实时生成的虚拟 MP4 无法提供真实 byte→
     sample 映射；AVPro/ExoPlayer 这类严格客户端会因为索引不一致而失败。
   - MPEG-TS 没有全局索引，从任意位置找到 PAT/PMT + IDR 即可重新解码，对
     "字节并不真的连续"这件事容忍度更高。
4. **`Content-Length` 必须稳定**。播放器二次打开同一文件时如果声明大小变了，
   会重置 seek 缓存或直接报错。这要求 EWMA key 设计稳定（已具备），且单次会
   话内不能因为流实际产出与估算偏差而临时改 header。

---

## 4. 合并后的目标设计

### 4.1 端点拓扑

- 保留 `/passthrough_live/{name}`：直播 + 章节兜底，不动。
- 复活并独立暴露 `/passthrough/{name}`（或新建 `/passthrough_seek/{name}` 别名），
  作为 byte-seek 兼容层，**默认 MPEG-TS 容器**。
- DLNA 目录暴露策略：**Browse 阶段不依赖播放器 UA**（[utils/player_compat.py](utils/player_compat.py)
  是按请求 UA 分类，目录生成时拿不到这个信息）。改为：
  1. `PT_PASSTHROUGH_SEEK_ENABLED=0|1` 控制 HTTP 路由是否允许直接访问
     `/passthrough_seek/...`；关闭时手工 URL 也不可用。
  2. `PT_PASSTHROUGH_SEEK_DLNA=0|1` 只控制 DLNA Browse 是否展示 seek 项；它
     不单独启用 HTTP 路由。`ENABLED=1, DLNA=0` 用于隐藏/manual URL 灰度测试。
  3. 当 `ENABLED=1` 且 `DLNA=1` 时，DLNA 目录里的 passthrough 虚拟入口应
     **同时展示** seekable 项和旧 live 章节/直播项。seek 是实验入口，live 是
     未识别/被 profile 拦截客户端的可见兜底；`/passthrough_live` 路由始终保留。
  4. 真正的 profile 区分放在 route 层：基于请求 UA，仅对白名单 UA 返回真正
     的 byte-seek 响应；profile/policy 拦截返回 403，并提示继续使用
     `/passthrough_live`。总开关关闭时返回 404，方便把实验 URL 隐藏起来。
  - 修改入口：[dlna/content_directory.py:414](dlna/content_directory.py#L414)。

### 4.2 虚拟文件约定

- `Accept-Ranges: bytes`
- DLNA seek 项使用 `DLNA.ORG_OP=11` + `DLNA.ORG_FLAGS=617000...`，因为新路由
  同时支持 HTTP byte Range 和 `TimeSeekRange.dlna.org`。legacy live/pseudo-VOD
  的 `OP=01/10` 兼容值不因该实验改变。
- `Content-Length = header_reserved + duration * virtual_output_bps / 8`
  - `virtual_output_bps = output_video_bps + output_audio_bps + mux_overhead_bps`
  - 来源优先级：EWMA 缓存 > 配置默认 > 源码率推断（仅用于冷启动）
- mid-range 必须返回 `206 + Content-Range: bytes start-end/total`；**不使用
  chunked 编码做 206**（多数 DLNA/VR 播放器对 chunked+206 不稳定）。
- 容器选 **MPEG-TS**。fMP4 留作后续单独评估，不进入首发范围。

### 4.3 字节→时间映射

```
if range_start < header_reserved:
    # 落在真实前缀区——必须返回与首次 bytes=0- 同字节的内容
    serve_prefix_bytes(range_start, range_end)   # 走 _probe_cache
else:
    ratio = (range_start - header_reserved) / (total - header_reserved)
    t = ratio * duration
    t = snap_back_to_keyframe(t, gop_seconds = PASSTHROUGH_GOP / output_fps)
    spawn_producer(t)
```

**`header_reserved` 的语义：它不是「虚拟偏移修正项」，而是一段真实存在、
内容稳定的前缀字节区**。约束：

- 任何 Range 与 `[0, header_reserved)` 区间相交时，返回的字节必须与上一次
  `bytes=0-` 对同一虚拟资源返回的字节**逐字节一致**。否则播放器跨 Range 拼
  接 / 缓存比对会失败。
- 这要求 `_probe_cache` 至少保留 `header_reserved` 字节，且写入是「一次性
  确定」的——首次生产时记录的前缀字节就是这个虚拟资源此后一直返回的前缀。
- `header_reserved` 取值：建议设为 `_PROBE_CACHE_LIMIT`（16MB）的一个子集，
  比如 1~2MB（覆盖 PAT/PMT + 首个 IDR + 一定缓冲）。**不要**取得太小以至于
  无法容纳真实 IDR；也不要等于全部缓存以致一般 probe 都走慢路径。

源文件 size / 源 bitrate / 源 FPS 只用于：

- 冷启动估算 `virtual_output_bps`
- 自适应输出 FPS / 输出码率上限（[routes_media.py:1260](http_app/routes_media.py#L1260) 的 `_live_adaptive_max_fps` 已具备这能力）
- 输出 FPS 用于换算 GOP/IDR 时间间隔

### 4.4 Session 复用与并发

**重要前提**：`/passthrough` 当前是**每请求独立建流**，不走 `LiveSession`
共享生产者机制。`LiveSession`（[routes_media.py:115](http_app/routes_media.py#L115)
及周边）只服务 `/passthrough_live` 直播路径。所以本路径不存在「复用键」概
念，不要把 LiveSession 的量化复用思路直接套过来。

**首发阶段（必须）**：

1. **同 client + 同文件抢占**。扩展
   `_can_preempt_owner`（[routes_media.py:525](http_app/routes_media.py#L525)）
   或新增 seek-profile 专属规则：当新请求与活动流来自同一 client、同一文件
   时，允许新请求直接 preempt，避免拖动期间 503 风暴。
2. **probe cache 续用**。继续靠 `_probe_cache`（[routes_media.py:97](http_app/routes_media.py#L97)）
   吃掉 small/zero-open/tail probe，不让 GPU 管线被 probe 反复点燃。
3. **UI 端 debounce**。客户端拖动期间不可避免会有多次中间请求，UI 层应在松
   开后才真正发起最终 Range；服务端 debounce 价值有限（HTTP 无法预知用户是
   否还在拖动）。
4. **并发上限**。`PASSTHROUGH_MAX_CONCURRENT=1` 在拖动场景仍然偏紧，建议在
   多卡机上通过 `PT_PASSTHROUGH_MAX_CONCURRENT=2~3` 提高，但不在首发改默认值。

**下一阶段（推迟）**：让 `/passthrough` 接入 `LiveSession` 共享生产者，把
落在同一 GOP 桶内的拖动复用同一个流。这条改动跨模块、影响 audio cache /
matter 生命周期，**不进入首发范围**。

### 4.5 GOP / 关键帧

- FFmpeg fallback 路径使用输入侧 `-ss`（[pipeline/ffmpeg_io.py:207](pipeline/ffmpeg_io.py#L207)），
  通常会落到可解码起点附近；这不是对所有后端都成立的严格 keyframe 保证。
- PyNv 路径按 `start_sec * output_fps` 换算输出帧索引，再映射到源帧索引，并
  在第一帧编码时强制输出 IDR/SPS/PPS。因此 GOP snap 的主要目的不是让解码器
  “能解码”，而是让同一拖动区域得到稳定、可复现的起点，减少反复拖动时的
  启动差异和抢占抖动。
- 字节→时间映射结果在落到 producer 前先向前量化到 `gop_seconds`：
  `PASSTHROUGH_GOP=60` + 30fps 输出约等于 2.0s。首发阶段该量化只用于稳定
  seek 起点；不会复用同一个 `LiveSession` 或共享生产者。

### 4.6 探测与缓存

- `_probe_cache` 继续作为 probe 屏蔽层，但 seek 路径只缓存
  `header_reserved` 字节，不用全局单条 16MB 上限作为前缀目标，避免多并发
  seek 把 64MB 总池挤满。
- **不要手写 PAT/PMT，也不要复用 MP4 合成 ftyp**。当前小 probe 路径里的合成
  `ftyp` 逻辑（[routes_media.py:2337](http_app/routes_media.py#L2337) 附近）
  是 MP4 专属，**在 MPEG-TS seek 路径下必须禁用**。手写 PAT/PMT 在容器边界、
  PCR、连续性计数等细节上极易踩雷，得不偿失。
- byte-seek 路径的 probe 响应策略：
  1. **缓存命中**：从 `_probe_cache` 直接吐字节（这些字节是真实 muxer 输出的
     TS 前缀，与后续流的字节字节一致）；
  2. **缓存未命中**：返回 `503` + `Retry-After: 1~2`，强制播放器重试，等首流
     启动后填满 `_probe_cache` 再服务后续 probe。不要尝试就地合成 TS 头部。
- 缓存填充：第一次真实生产时，把输出流前 N 字节（建议 1~2MB，覆盖 PAT/PMT +
  首个 IDR）旁路写入 `_probe_cache`，键含 `(path, codec, virtual_total)`。

---

## 5. 实施步骤（建议顺序）

按开发反馈调整：**DLNA 暴露放到最后**，先把链路在 hidden 状态下打磨稳定。

1. **隐藏状态下验证 MPEG-TS byte seek**。`/passthrough` 端点保持 DLNA 隐藏，
   通过 `LIVE_REQUEST_HEADER_DUMP=1` + 手动构造 URL 在真机上跑 byte-seek
   往返，抓 4XVR / Skybox / DeoVR / nPlayer / VLC 的 Range 请求模式。**byte
   -seek 路径强制 MPEG-TS 容器**（不受全局 `PASSTHROUGH_CONTAINER` 影响），
   验证 PAT/PMT + IDR 重启行为。
2. **抽出可独立测试的 `byte→time + GOP-snap` helper**。把 §4.3 的映射逻辑
   提到 `utils/` 下（建议 `utils/byte_seek_map.py`），写单测覆盖：
   - probe 落在 `header_reserved` 区的路径
   - 越界（`ratio > 1.0`）的钳制
   - GOP snap-back 在不同 `output_fps` / `PASSTHROUGH_GOP` 下的边界
3. **修 probe prefix**：实现 §4.6 的「真实 muxer 前缀 → `_probe_cache` →
   缓存未命中 503/Retry」语义。删掉 MPEG-TS 路径上的 MP4 ftyp 合成分支。
4. **加同 client + 同文件抢占**：实现 §4.4 首发阶段两条；扩展
   `_can_preempt_owner`（[routes_media.py:525](http_app/routes_media.py#L525)）。
5. **会话常量化声明 size**：实现 §6 的「单会话内 `Content-Length` 不变」
   约束，确保 EWMA 更新不会影响进行中的会话。
6. **DLNA 暴露**（最后一步）：新增 `PT_PASSTHROUGH_SEEK_DLNA=0|1` 配置开关，
   开启时在 DLNA 中额外展示 seek 项，同时保留旧 live/chapter 虚拟入口；route
   层按 UA 白名单做最终守门。一开始建议只放开 nplayer / vlc / libmpv 三个最
   宽容的 profile，收集 1~2 周再扩到 ExoPlayer 系。

---

## 6. 风险

- **本质上不是真 byte seek**。返回的字节不是同一份不可变文件的第 N 字节。
  容忍度低的播放器（严格 MP4 索引解析、跨 Range 缓存拼接）会失败。
- **估算冷启动偏差**。第一次播放前没有 EWMA 样本；估算严重偏离时尾部不可达
  或末尾空洞。处理策略：
  - **不要下调 size 做安全裕度**——低估 `Content-Length` 会让播放器永远到不
    了真实尾部，比高估更糟糕。
  - 现有估算器（[utils/bitrate_estimator.py:232](utils/bitrate_estimator.py#L232)）
    已经通过**上浮安全系数** + `PASSTHROUGH_PAD_TO_LENGTH` 末尾填充覆盖了
    高估场景，沿用即可。
  - 真正需要保证的不变量：**单个活跃会话内声明的 `Content-Length` 必须保持
    恒定**。EWMA 更新可以发生（流结束时回写），但**不能影响正在进行中的同
    一虚拟资源的后续 Range 响应**。实现上，一个会话/客户端首次拿到的 size
    应该缓存为该会话的常量，后续 probe / range 请求都用这个常量值，不重新
    查 estimator。
- **拖动密度爆破**。首发阶段每次有效 seek 都会重新启动生产者；即使做了
  GOP snap 和同 client 抢占，启动仍可能有 300ms~数秒延迟，UI 端必须 debounce
  才能避免拖动卡顿假象。
- **MP4 路径需要单独立项**。如果未来必须支持 MP4-only 客户端，等同于需要离
  线生成/近线生成有真实索引的输出文件，与本兼容层不同方向。

---

## 7. 参考资料

- MDN — *HTTP Range requests*：byte ranges 是同一资源表示内的字节区间。
- MDN — *Range header / 206 / Content-Range*：必须用 206 + Content-Range 表达。
- Android Media3 — `HttpUtil`：`Range` header 由内部 position/length 派生。
- Android Media3 — *Supported formats*：部分格式仅按 CBR 假设近似可 seek。
- 本仓库：[HANDOVER_20260528.md](../prompt/HANDOVER_20260528.md) §Seekable
  Realtime Passthrough Research。
- 本仓库历史相关报告：
  [summary_20260509_PLAYER_CONCURRENCY_REPORT.md](summary_20260509_PLAYER_CONCURRENCY_REPORT.md)、
  [HANDOVER_20260508.md](../prompt/HANDOVER_20260508.md) §Passthrough pseudo-VOD
  byte seek integration。

---

## 8. 待决问题

- `header_reserved` 取 256KB 还是更大？需要看真机 probe 上限。
- GOP snap 粒度是按 `PASSTHROUGH_GOP / output_fps`（约 2s）还是粗化到 5~10s？
  首发阶段它影响 seek 精度、启动稳定性和抢占频率；共享 session 复用留到后续
  单独评估。
- DLNA 目录里 seek 项是否需要额外的显示标记（例如 `_seek`）？首发建议直接
  替换旧 live/chapter 虚拟入口，避免客户端同时探测两条流。
