# DLNA 播放器兼容性梳理计划

日期：2026-05-28

## 1. 背景

当前项目是本地 DLNA/UPnP 媒体服务器，面向 VR 播放器提供原始视频和实时 passthrough 视频流。实时 passthrough 会占用 GPU 解码、抠图、合成、编码等资源。对小显存用户来说，通常只有一路稳定并发能力，因此这一路能力必须优先服务用户当前真正观看的视频。

过去为了兼容不同播放器，代码里陆续加入了大量针对 User-Agent、Range 请求、重复请求、Lavf 副请求、播放器启动探测等场景的处理。这些处理有效解决了一些具体问题，但也带来了新的维护难点。现有代码里其实已经有几类隐式策略原型，例如 User-Agent 到 live profile 的映射、active owner 抢占规则、Lavf `reject/active_only/allow` 策略、nPlayer 重复启动去抖、VLC preroll 和 pseudo-VOD 响应形态。后续计划不是推翻这些经验，而是把它们回收到可观察、可测试、可配置的策略框架里。

- 播放器兼容逻辑分散在请求处理流程中，后续继续追加分支会越来越难判断影响范围。
- 单次 HTTP 请求很难判断真实意图，因为很多播放器不会请求缩略图接口，而是请求视频本身来生成截图或预览。
- 当前 `/media` 原始文件请求本身不会启动 GPU，它更多是识别播放器行为和截图/预览习惯的信号；真正需要严格防护的是 `/passthrough` 和 `/passthrough_live` 这类会进入实时 GPU 生产链路的请求。
- 有些播放器会在真实播放前后对 passthrough URL 发出探测请求、尾部 Range 请求、重复启动请求或内部分析请求，这些请求不应抢占 GPU 生产链路。
- 目前调试主要依赖 server log、临时 header dump 或外部抓包，缺少统一、可回放的请求序列记录。

## 2. 需求

本计划希望先梳理播放器兼容性处理方式，为后续重构和策略完善提供方向。当前不涉及具体代码实现。

核心需求如下：

1. 统一梳理不同播放器的兼容性差异，减少继续堆叠 User-Agent 分支。
2. 将共性处理和播放器特有处理分开，便于维护、测试和回退。
3. 支持自动识别播放器行为，同时预留用户强制指定某种播放器兼容路径的能力。
4. 建立最近 N 条请求的历史记录，不只记录视频流请求，也记录截图、原始媒体、CDS Browse、字幕等请求。
5. 基于请求历史判断请求意图，例如真实播放、截图、启动探测、尾部探测、副请求、重复启动等。
6. 将请求历史输出到 `debug_output`，方便排查真机行为，减少对 Wireshark 抓包的依赖。
7. 对小显存用户保持严格资源保护：非真实播放请求不得随意启动或中断实时 passthrough。

## 3. 目标

首要目标不是马上支持更多播放器，而是建立一套可持续的兼容性框架。

具体目标：

- 让“播放器是谁”和“这次请求想做什么”分开判断。
- 让“是否允许占用 GPU”成为明确策略，而不是隐藏在某个 User-Agent 分支中。
- 让截图、探测、副请求尽可能在启动 GPU 前被识别、复用缓存或拒绝。
- 让真实播放请求可以稳定占用或接管实时转码资源。
- 让新增播放器兼容时优先补充规则和测试，而不是直接修改主请求流程。
- 让真机兼容问题可以通过请求历史文件复盘。

## 4. 建议做法

### 4.1 分层处理

建议将当前混在一起的逻辑拆成几个概念层：

1. 请求观测层  
   负责记录所有 DLNA 相关请求，包括时间、客户端、路径、请求头、响应结果、关联媒体、耗时等。

2. 设备会话与播放器识别层  
   以客户端为单位维护一个短期 `DeviceSession`，根据 User-Agent、请求头、CDS Browse 模式、Range 行为、重复请求等信号判断播放器或播放器行为类型。

3. 请求意图层  
   判断这次请求想做什么，例如真实播放、启动探测、截图预览、尾部探测、副请求、重复启动等。

4. 决策与动作层  
   根据播放器类型、请求意图和当前活动播放状态，决定资源策略。建议明确区分：

   - Intent：请求想做什么。
   - Decision：资源策略是什么，例如 `allow_gpu`、`reuse_session`、`return_synthetic`、`reject_416`、`reject_409`、`reject_503`。
   - Action：具体 HTTP 行为和资源动作，例如返回 `StreamingResponse`、等待、抢占、拒绝、复用缓存。

这样做的好处是，后续测试可以直接验证 `profile + intent + resource_state -> decision`，不必每次都拉起完整 FastAPI 路由和真实 GPU 管线。

### 4.2 不只依赖 User-Agent

User-Agent 仍然有价值，但不应作为唯一依据。建议综合以下信号：

- User-Agent 和常见播放器标识。
- 请求路径类型：CDS、thumb、media、passthrough_live、passthrough、subs。
- 请求方法：GET、HEAD、POST。
- Range 形态：无 Range、`bytes=0-`、小范围、非零 open range、尾部范围。
- DLNA 请求头：`TimeSeekRange.dlna.org`、`transferMode.dlna.org`、`getcontentFeatures.dlna.org`。
- 同一客户端短时间内是否重复请求相同 URL。
- 请求前后是否出现 CDS Browse、缩略图请求、原始视频请求。
- 当前是否已有同客户端真实播放流。
- 请求是否持续读取数据，还是拿到少量数据后断开。

识别方式建议从简单 if-else 逐步升级为轻量评分模型。每条规则给某个 profile 或 intent 加分，并记录置信度和触发规则。例如：

- UA 包含 `nplayer`：提高 `nplayer_like` 置信度。
- UA 包含 `avpro`、`exoplayer` 或 Quest Dalvik 特征：提高 `quest_avpro_like` 置信度。
- 短时间重复请求相同 live URL：提高 `duplicate_startup` 或 `nplayer_like` 置信度。
- 非零 open Range 或尾部 Range：提高 `probe` 或 `tail_probe` 置信度。
- CDS Browse 模式、RequestedCount、重复 Browse ObjectID：作为播放器指纹信号。

当置信度不足时，应回退到一个保守 profile，而不是冒险让未知请求抢占 GPU。具体回退到 `strict_live` 还是当前默认兼容路径，需要结合真机数据讨论，见第 7 节待讨论问题。

### 4.2.1 设备会话识别

建议引入短期 `DeviceSession` 概念，以 `client_host` 或 `client_host + UA hash` 为 key，保存最近几分钟的观测信号：

- 已观察到的 UA 和请求头。
- CDS Browse 行为。
- Range 形态统计。
- 最近请求的 URL 和时间间隔。
- 当前识别出的 profile、置信度和来源规则。
- 用户是否强制绑定 profile。

优点：

- 播放器通常会先 Browse 再播放，CDS 行为可以在 GPU 请求到达前提供识别信号。
- 同一设备后续请求使用同一个 profile，避免每个请求单独识别导致 profile 摇摆。
- live session key 可以使用稳定的 profile class，避免同一设备同一 URL 因 profile 小幅变化而无法复用 LiveSession。

### 4.3 建立请求历史缓存

建议维护最近 N 条请求记录，N 可以默认 300 或 500，并允许配置。

记录范围建议包括：

- `POST /control/cds`
- `GET /description.xml`
- `GET /cds.xml`
- `GET /cm.xml`
- `GET/HEAD /media/...`
- `GET /thumb/...`
- `GET /passthrough_live/...`
- `GET/HEAD /passthrough/...`
- `GET/HEAD /subs/...`

记录内容建议包括：

- trace id，例如 `X-PT-Request-Trace-Id` 对应的请求编号。
- 请求时间和耗时。
- 客户端地址。
- 请求方法和路径类型。
- 媒体文件标识。
- User-Agent。
- Range、TimeSeekRange、transferMode、getcontentFeatures。
- SOAP Browse 的 ObjectID、BrowseFlag、RequestedCount。
- 响应状态码。
- 响应字节数或估算字节数。
- 自动识别出的播放器 profile。
- 自动识别出的请求 intent。
- 最终策略动作。
- 触发的规则列表，例如 `fired_rules`。
- shadow mode 下的新策略建议动作。

请求历史应定期或按需写入 `debug_output/request_history/`，建议使用 JSONL，便于后续筛选、比对和人工阅读。

磁盘输出建议明确滚动策略：

- 内存环形缓存控制最近 N 条，例如 300 或 500。
- 磁盘 JSONL 按天或按大小滚动，避免无限增长。
- 默认不对每条请求做同步 fsync，采用批量 flush 或关闭时 flush，避免影响播放。
- 诊断导出时包含最近 JSONL、当前 profile、规则版本和 server log 摘要。

现有 `debug_output/live_requests/` per-request header dump 可以保留为深度调试开关，但建议将 JSONL 作为默认、统一的请求历史入口。per-request txt 只在需要完整 header 细节时开启，避免两套记录长期并行造成混乱。

### 4.4 请求意图分类

建议先定义一组稳定的请求意图：

- `browse`：DLNA 浏览目录。
- `metadata`：设备描述、服务描述、ContentDirectory 元信息。
- `thumbnail_endpoint`：明确请求 `/thumb`。
- `raw_media_preview`：请求 `/media` 但行为更像截图或预览。当前仅用于识别和统计，不参与 GPU 资源决策。
- `startup_probe`：播放器启动前探测。
- `duplicate_startup`：短时间重复启动同一 live URL。
- `tail_probe`：视频尾部或大 offset Range 探测。
- `side_probe`：播放器内部副请求，例如 UA 包含 `lavf/` 的分析请求。
- `playback_primary`：真实播放请求。
- `subtitle`：字幕请求。
- `unknown`：暂无法判断。

分类可以先只写入日志和请求历史，不立即影响行为。等真机数据验证稳定后，再逐步让策略层使用这些结果。

需要特别注意：`raw_media_preview` 不等于 GPU 风险。当前 `/media` 是原始文件 Range 服务，不会启动 Matter/PyNv。它的价值主要是作为播放器行为信号，例如某个播放器在 Browse 后先请求 `/media` 小范围来做截图，再请求 `/passthrough_live` 播放。真正需要在资源层拦截的是误打到 `/passthrough` 或 `/passthrough_live` 的 probe、tail、side 和 duplicate 请求。

### 4.5 播放器兼容策略

播放器策略建议按“行为类型”组织，而不是完全按产品名组织。

初始可考虑这些 profile：

- `auto`：自动识别。
- `vlc_like`：VLC、LibVLC、MoonVR 类行为。
- `libmpv_like`：Skybox/libmpv 类行为。
- `nplayer_like`：nPlayer/OPlayer 类重复启动和 Range 行为。
- `quest_avpro_like`：4XVR、AVPro、ExoPlayer、Quest Dalvik 类行为。
- `strict_live`：严格 live streaming，不暴露 byte-range 语义。

每个 profile 应描述策略，而不是写死在主流程中：

- 是否使用 managed live session。
- 是否允许非起点 Range。
- 是否忽略 Range 参与 session key。
- live session key 使用 raw profile 还是 profile class。
- 是否启用重复启动去抖，以及去抖窗口。
- 是否允许同客户端新播放接管旧流，以及哪些 intent/profile 允许接管。
- 是否拒绝 UA/行为判定为 Lavf side_probe 的副请求。
- 是否启用 preroll。
- 是否使用 pseudo-VOD 响应形态。
- 是否覆盖音频策略。
- 哪些 intent 允许启动 GPU。
- probe/side/tail 请求的拒绝策略，例如 `416`、`409`、`503 Retry-After`、合成 `200/206`、空 body 等。
- 策略规则版本、启用状态、shadow mode 状态。

当前代码中的隐式策略应先整理成清单，再迁移到 profile 表中。至少包括：

- User-Agent 到 live profile 的映射。
- active owner 抢占规则。
- Lavf side-probe intent 识别，以及对应的 `active_only/reject/allow` 策略。
- nPlayer 重复启动去抖。
- libmpv/Skybox managed LiveSession 缓存。
- 4XVR/AVPro managed LiveSession 和同设备接管。
- VLC/LibVLC/MoonVR preroll、pseudo-VOD 和非起点 Range 拒绝。
- `/passthrough` 已有的 small probe、tail probe、prefix cache 旁路。

其中 `/passthrough` 上已经存在的探测旁路经验，应作为后续统一 `/passthrough_live` 保护策略的重要参考。

### 4.6 用户强制模式

建议预留用户可配置选项，但不建议只做全局开关。更合理的形态是：

- 全局默认：`auto`。
- 设备级绑定：某个 `client_host` 或设备指纹强制使用指定 profile。
- 临时调试覆盖：短期 A/B 时允许全局强制。

可选 profile：

`auto / vlc_like / libmpv_like / nplayer_like / quest_avpro_like / strict_live`

目的：

- 自动识别失败时，用户可以手动选择兼容路径。
- 调试播放器问题时，可以快速 A/B。
- 新播放器尚未加入自动识别时，可以先选择最接近的行为模式。
- 家里多个播放器同时存在时，不会因为全局强制 profile 影响所有设备。

强制 profile 只应影响播放器兼容策略，不应关闭请求意图判断。也就是说，即使用户强制某个 profile，明显的截图、尾部探测、副请求仍然不应随意占用 GPU。

强制 profile 也不应解除小显存资源保护原则。即使用户绑定了某个 profile，明显的 probe、side、tail 请求仍按第 5 节执行，且不允许通过强制 profile 抬高 active slot 上限。

## 5. 小显存资源保护原则

这是本计划最重要的约束。

对小显存用户来说，系统必须默认假设只有一路实时 passthrough 能力。策略上应遵守：

1. 只有 `playback_primary` 可以启动 Matter、PyNv、NVENC/NVDEC 等重资源。
2. `thumbnail_endpoint`、`raw_media_preview`、`tail_probe`、`side_probe`、`startup_probe` 应优先从缓存、合成响应或轻量拒绝中处理。
3. 副请求不能中断真实播放流。
4. 同一客户端的新真实播放请求可以按 profile 策略接管旧播放流，避免播放器未关闭旧连接导致用户无法切片或切文件。不是所有 profile 都必须允许宽松接管，接管友好型 profile 应明确列出。
5. 不同客户端之间默认不互相抢占，除非用户明确配置多用户或多并发策略。
6. 多并发能力应服从显存上限；自动多并发不应削弱单路保护规则。
7. 即使 active slot 未满，`probe`、`side_probe`、`tail_probe`、`raw_media_preview` 等非真实播放 intent 也不应启动 GPU，应按 profile 的拒绝、缓存或合成响应策略处理。
8. 当 active slot 已满时，intent 应参与优先级判断：真实播放可以接管同设备旧 probe 或无订阅 live session；probe、side、tail 不得接管真实播放。
9. `/media` 原始文件请求可以作为识别信号记录，但当前不应被视为 GPU 并发风险源。

## 6. 建议实施阶段

### Phase 0：建立基线和追踪能力

目标：先把当前隐式行为固定下来，避免后续抽离时行为漂移。

建议内容：

- 整理现存隐式策略清单。
- 为当前 `_live_response_profile`、抢占规则、Lavf 策略、nPlayer 去抖、probe 旁路等行为补 characterization tests。
- 给请求和响应建立稳定 trace id，例如响应头 `X-PT-Request-Trace-Id`，并写入日志/JSONL。
- 在日志中补齐策略审计字段的雏形：profile、intent、decision、fired_rules。

这一阶段不改变播放器行为，只让后续讨论和重构有可验证基线。Phase 0 只搭字段管道和占位值，例如日志列、响应头、JSONL 字段；Phase 1 才开始为这些字段填入真实分类结果。

### Phase 1：只做观测

目标：先看清播放器真实行为，不改变现有播放逻辑。

建议内容：

- 建立请求历史缓存。
- 输出 JSONL 到 `debug_output/request_history/`。
- 在日志中记录 profile 和 intent 的初步判断。
- 引入 DeviceSession，先只记录 profile 候选和置信度。
- 增加本地调试入口，便于查看最近请求。
- 基于收集到的 JSONL 建立 scenario replay 测试框架：能从真实设备录制的请求序列复放，断言 profile、intent、decision 序列符合预期。

这一阶段不改变任何兼容行为，风险最低。

### Phase 2：抽离现有策略

目标：把当前已有逻辑整理成清晰结构，但保持行为基本等价。

建议内容：

- 抽出播放器 profile 判断。
- 抽出现有 live 策略。
- 将 nPlayer、libmpv、VLC、Lavf side-probe、4XVR/AVPro 的现有处理整理成策略描述。
- 明确 live session key 使用 profile class 还是 raw profile，避免识别小幅变化破坏 session 复用。
- 将 Phase 1 的 scenario replay 数据纳入回归测试，作为新增 profile 或规则的标准验证手段。
- 增加回归测试，确保行为没有意外变化。

### Phase 3：引入 intent shadow mode

目标：让请求意图和新策略开始运行，但只记录，不参与拦截。

建议内容：

- 对请求历史进行意图分类。
- 并行运行新决策逻辑，记录“如果使用新策略会怎么处理”。
- 在日志和 JSONL 中记录分类结果、决策结果、规则版本和触发规则。
- 用真实播放器测试数据校正规则。
- 重点验证“截图请求”和“真实播放请求”的区分。
- 单条规则可以独立 shadow、验证和提升到生效状态，不必等整个阶段一次性切换。

### Phase 4：启用资源保护策略

目标：让 intent 影响是否允许占用 GPU。

建议内容：

- 明确禁止 probe、tail、side、preview 请求启动 GPU。
- 真实播放请求才允许 acquire passthrough 资源。
- 对重复启动请求优先复用 session。
- 对无法判断的请求采用保守策略，避免抢占当前播放。
- 拒绝策略按 profile 管理，避免某些播放器收到 `503 Retry-After` 后形成重试风暴。
- 对小显存用户，intent 优先级参与 active slot 抢占判断。

### Phase 5：用户配置与 UI

目标：让用户可以选择自动或强制兼容路径。

建议内容：

- 增加全局默认 profile 和设备级 profile 绑定。
- UI 中提供简洁的播放器兼容模式选择，例如“自动识别 / 这台设备强制按某类播放器处理”。
- 保留高级调试开关。
- 在诊断导出中包含请求历史和当前 profile。

### Phase 6：主动塑形 CDS 输出（待验证）

目标：不只被动防御请求，也可以根据已识别设备主动调整暴露给播放器的资源。

可能做法：

- 对某些 profile 只暴露更适合它的 live 资源。
- 对某些严格播放器减少容易诱发错误 Range 行为的资源。
- 对 alpha/green 支持差异明显的播放器优先暴露合适模式。

这一阶段风险较高，因为 CDS 输出会影响播放器能看到什么资源。例如不同 profile 下 ObjectID 或资源列表不一致，可能破坏播放器收藏夹、播放历史或缓存；DIDL 资源列表稳定性本身是 DLNA 交互的重要契约。建议只作为后续讨论项，且必须有明确回退开关。

## 7. 需要讨论的问题

后续与其他人一起完善计划时，建议重点讨论：

1. 请求历史默认保存多少条、保存多久、是否需要脱敏。脱敏候选字段包括客户端 IP、媒体文件名/路径、User-Agent、SOAP body 中的 ObjectID。
2. `debug_output/request_history` 是否默认开启，还是只在 debug 模式开启；per-request header dump 是否只保留为深度调试。
3. 强制 profile 的命名是按播放器名，还是按行为类型；UI 是否采用设备级绑定。
4. 未知请求应默认拒绝、等待、还是按最保守 live path 处理。
5. `/media` 原始文件请求主要作为识别信号，是否还需要针对磁盘 IO 或带宽做轻量限流。
6. 当前 live session cache 是否需要和 DeviceSession、请求历史统一关联。
7. live session key 应使用 raw profile，还是使用 profile class。
8. 多并发用户与小显存单并发用户是否需要不同默认策略。
9. 哪些播放器作为首批验证对象：Skybox、MoonVR、4XVR、nPlayer、DeoVR、HereSphere、VLC。
10. 是否以及何时启用 CDS 输出主动塑形。
11. 不同 profile 的拒绝策略应该如何选择，尤其是 `416`、`409`、`503 Retry-After`、合成 `200/206` 的取舍。

## 8. 验收标准

计划完成后的理想状态：

- 每次播放器兼容问题都有请求历史文件可查。
- 新增播放器时优先添加 profile/intent/policy，不再直接堆主流程分支。
- 小显存用户播放时，截图、探测、副请求不会抢占唯一 passthrough 能力。
- 用户切章节、切文件时，真实播放请求可以稳定接管旧流。
- 自动识别失败时，用户可以强制选择兼容路径。
- 兼容策略有明确测试覆盖，后续修改不容易引入回归。
- 每次拒绝、复用、抢占都有策略审计记录，可看到 profile、intent、decision、置信度和触发规则。
- 现有隐式策略迁移后，有 characterization tests 证明关键行为没有无意漂移。

## 9. 对评审建议的吸收与待讨论点

本次修订吸收的建议：

- 增加 Phase 0：先补当前隐式策略基线测试和 trace id。
- 明确 `/media` 当前不占 GPU，主要作为播放器识别信号。
- 将三层设计细化为请求观测、设备会话识别、intent、decision/action。
- 引入 DeviceSession，避免每个请求重复独立识别导致 profile 摇摆。
- 建议 profile 识别使用多信号评分和置信度，而不是单纯 if-else。
- 将 live session key 是否使用 profile class 纳入计划。
- 将 nPlayer debounce、Lavf side-probe intent 与策略、抢占矩阵、VLC preroll 等现有隐式策略列入回收清单。
- 将拒绝策略作为 profile 配置项，不再混用 `416/409/503/synthetic`。
- 将 JSONL 请求历史和现有 live header dump 的关系写清楚。
- 增加规则版本、shadow mode、fired_rules 和策略审计。
- 将强制 profile 从单纯全局开关调整为优先考虑设备级绑定。
- 将 active slot 抢占从只看 owner/profile，扩展为未来可参考 intent 优先级。

仍建议保持讨论状态的点：

- CDS 主动塑形很有价值，但会改变播放器看到的资源列表，风险比观测和策略抽离更高，适合放到后期。
- DeviceSession 的 key 是否只用 client_host，还是需要 UA hash、设备名、未来 MAC/UPnP 信息，需要结合真实网络环境决定。
- 对未知请求回退 `strict_live` 还是当前默认 `vlc_like`，需要通过真机数据比较。
- `/media` 是否需要 IO/带宽层面的限制，不应和 GPU 资源保护混为一谈。

## 10. 非目标

本计划用于梳理播放器兼容策略和资源保护边界，以下内容不纳入 Phase 0-5：

- 不引入基于机器学习的播放器识别。
- 不修改 SSDP/UPnP 设备通告层面的行为。
- 不在本计划周期内引入持久化 DeviceSession；DeviceSession 可以先按进程内短期状态处理，重启即丢。
- 不在 Phase 0-5 改动 DLNA PN、容器选择或 CDS 资源列表结构；资源列表主动塑形只在 Phase 6 讨论。
- 不把 `/media` 原始文件请求纳入 GPU 资源保护对象；它当前只作为行为识别和统计信号。
