# PTMediaServer 集成"混合同声传译(SI)音轨"功能 — 开发计划与可行性报告

> 日期：2026-06-16　分支：feature/si
> 参考实现：`VR_Video_Toolbox_NE/tool_dlna/si_stream.py`（已落地、已测、实时流式 v2 方案）、
> `tool_si/logic.py`（滤镜构造）、`tool_clonevoice/gui.py`（混合音轨参数 UI）。
>
> 结论先行：**可行**。参考工程已有完整的"实时 ffmpeg 转码 + seek-by-time"实现可直接移植，
> 唯一的实质性工作量在于把参考的**同步/线程**实现适配到本工程的 **async FastAPI** 架构，
> 并接入本工程已有的**热重载（runtime_settings）**与 **VR 命名（vr_naming）**体系。
> 预计 4.5～6 个工作日。

---

## 0. 需求拆解（来自用户）

1. **UI**：首页"其他配置"分组内新增一行"同声传译" toggle 滑块，行尾增加"音轨设置"按钮，点击弹出
   设置对话框。参数对齐 tool_clonevoice 混合音轨：**叠加声道(选择音轨)、原声音量、SI音量、SI延迟、降低原声(CHECKBOX)**。
   关键约束：**实时服务器运行中也必须实时生效**（含 toggle 开关本身）。
2. **DLNA 浏览**：同声传译开启时，若 DLNA 目录中存在与 MP4 同名的 `<stem>.si.wav`，则在原条目旁
   额外生成虚拟条目 `[SI]<VR视频名称>`。注意 VR 名称需走 `_SBS` 等命名处理，便于 VR 播放器选模式。
3. **播放**：客户端检索视频 range → 反算视频时间位置 → 读取原音轨与 `.si.wav` → 实时混音输出**新音轨**，
   视频流保持原样（不重编码）。本质是**伪装成正常 MP4**接管请求，分别处理视频与音频。
   必须正确处理播放器探针（probe），**多线程、不阻塞**，全程 ffmpeg 处理。

---

## 1. 现有架构关键事实（已核对源码）

| 关注点 | 本工程现状 | 文件 |
|---|---|---|
| HTTP 框架 | **async FastAPI + uvicorn**（与参考的同步 Flask/线程不同，这是主要适配点） | [http_app/server.py](http_app/server.py) |
| 进程模型 | 服务器与 UI **同进程内**运行（`create_app` 直接被 UI 拉起，非独立子进程） | [http_app/server.py:70](http_app/server.py) |
| 原始 MP4 服务 | `/media/{name:path}` GET/HEAD，标准 Range，`StreamingResponse`/`FileResponse` | [http_app/routes_media.py:1130](http_app/routes_media.py) |
| 路由挂载 | `control_router` / `dlna_router` / `media_router` | [http_app/server.py:137](http_app/server.py) |
| DLNA 浏览 | `_video_items_from_index` 为每个视频产出 原始项 + passthrough 项；`_children_for_dir` 有 **`_dir_items_cache`** | [dlna/content_directory.py:542](dlna/content_directory.py) |
| 浏览缓存键 | 含 `snapshot.signature` + 若干 config，**不含 SI 状态**（需扩展，否则切换不刷新） | [dlna/content_directory.py:766](dlna/content_directory.py) |
| 热重载范式 | `light_match`：线程安全 + **版本号** dataclass（`utils/runtime_settings.py`）+ 本地控制路由 `PUT /control/light_match` + UI 用 `urllib` 后台线程 PUT | [utils/runtime_settings.py](utils/runtime_settings.py), [http_app/routes_control.py](http_app/routes_control.py), [ui/pages/home_page.py:1002](ui/pages/home_page.py) |
| VR 命名 | `source_display_stem(stem, w, h)` → 2:1 半球源加 `_LR_180_SBS`，老 `_LR_180` → `_SBS` | [utils/vr_naming.py:64](utils/vr_naming.py) |
| 探针 | `pipeline.ffmpeg_io.probe_cached(path)` → `VideoInfo(width,height,fps,duration,...)` | [pipeline/ffmpeg_io.py:89](pipeline/ffmpeg_io.py) |
| ffprobe JSON | `utils/ffprobe_json.run_ffprobe_json(cmd)`（取分流 size / 音轨列表用） | [utils/ffprobe_json.py:19](utils/ffprobe_json.py) |
| 子进程隐藏 | `utils/subprocess_hidden.hidden_subprocess_kwargs()`（Windows 无窗口） | [utils/subprocess_hidden.py](utils/subprocess_hidden.py) |
| UI 行范式 | 每行 = label + toggle + `addStretch(1)` + 按钮；toggle 用 `_apply_switch_style`；分组 `group_layout.addWidget(row)` | [ui/pages/home_page.py:642](ui/pages/home_page.py) |
| 设置持久化 | `ui/settings.py` `Settings.data`(dict) + `DEFAULTS` + `save()/load()` → `runtime_cache/ui_settings.json` | [ui/settings.py:114](ui/settings.py) |

### 参考实现可直接复用的资产
- `tool_dlna/si_stream.py`：`SIMixConfig` / `ConfigHolder` / `LiveStreamSession` / `SIStreamService`
  （估算大小、字节↔时间换算、session 复用容差、seek 冷却、EOF 才回收、热重载）。**核心算法直接搬。**
- `tool_si/logic.build_si_mix_filter(channel, orig_vol, si_vol, delay, duck_original)`：
  滤镜纯函数，已含 `aresample/aformat/adelay/amix/alimiter/sidechaincompress(降低原声)`。
  常量：`SI_MIX_CHANNELS=("left","right","both")`、`ORIGINAL_VOLUME_CHOICES=(70,80,90,100)`、
  `SI_VOLUME_CHOICES=(50,60,70,80,90,100)`、`SI_DELAY_SECONDS_CHOICES=(0,0.3,0.5,0.7,1,1.2,1.5,2)`。
  默认：原声 100 / SI 50 / 延迟 1.0。**建议把这个纯函数小段复制进本工程**（避免对 VR_Video_Toolbox_NE 产生跨工程导入依赖）。

---

## 2. 总体方案

沿用参考工程已确认并落地的 **v2 实时流式**方案（不预混、不落盘、不预热、不阻塞、热重载）：

```
DLNA 浏览：原视频条目  +  [SI]<vr_name>  ←(仅当 toggle 开且 .si.wav 存在)
                              │ url = /media_si/<key>
                              ▼
GET /media_si/<key>  Range: bytes=N-          （伪装成普通 MP4）
   1. 读 SI 运行时配置（含 enabled、声道、音量、延迟、降低原声、音轨index）
   2. estimate_total = 视频流size + 192k 音频size，×1.05 → 作 Content-Length
   3. t = (N / estimate_total) * duration               （字节→时间）
   4. 查/起 LiveStreamSession：
        ffmpeg -ss t -i video -ss t -i si.wav
               -filter_complex <build_si_mix_filter>
               -map 0:v -c:v copy -map [si_track] -c:a aac -b:a 192k -ac 2
               -movflags +frag_keyframe+empty_moov+default_base_moof -f mp4 pipe:1
        （视频 copy 不重编码；只重编码音频；fragmented mp4 纯流式）
   5. 顺序读复用同一 ffmpeg（容差 1MB）；seek/换配置则杀旧起新（200ms 冷却）
   6. 仅 ffmpeg 真正 EOF 才回收 session（探针/短连接不回收）
```

**为什么能"伪装 MP4 且不阻塞"**：视频 `-c:v copy`，两路 `-ss` 都在 `-i` 前（demuxer fast-seek 到最近关键帧），
fragmented MP4 不需要回写 moov，因此 ffmpeg stdout 可直接 pipe 给 HTTP body 边转边发；
每个 (video, client) 维护至多一个 ffmpeg，session 复用避免每个 Range 重启。

---

## 3. 与参考实现的关键差异 / 适配点（**本计划重点**）

### 3.1 async 适配（最重要）
参考 `LiveStreamSession.read()` 是阻塞同步读 `proc.stdout.read()`。本工程是 async 事件循环，
**绝不能在协程里直接阻塞读 ffmpeg 管道**，否则卡死整个 HTTP 服务。两种落地方式（择一）：

- **方案 A（推荐，改动最小）**：保留参考的同步 `SIStreamService`/`LiveStreamSession` 不动，
  在 FastAPI 路由里用 `StreamingResponse` 包一个 **async 生成器**，每次 `chunk = await asyncio.to_thread(session.read, n)`。
  与现有 `/media` 的 `StreamingResponse(gen())` 风格一致，线程池承担阻塞读。
- 方案 B：把 read 改成 `loop.run_in_executor` 或独立 reader 线程 + `asyncio.Queue`（类似现有 `LiveSession` 的
  producer/subscriber，见 [routes_media.py:245](http_app/routes_media.py)）。更复杂，首版不必。

> 采用方案 A。`SIStreamService` 内部锁是 `threading.Lock`，由 `to_thread` 调用，安全。

### 3.2 探针/短连接处理（用户明确要求）
DLNA/VR 播放器启动时会发：HEAD、`Range: bytes=0-`、小尾部 range 探 EOF、libmpv 截图探针等。
参考实现的两条机制已覆盖，移植时务必保留：
- **EOF 才回收**：生成器 `finally` 里只有 `saw_eof` 才 `close` session（`GeneratorExit` 不杀 ffmpeg）。
  否则每个短 Range 都会重启 ffmpeg → moov 错乱、花屏。见 [si_stream.py:486](../VR_Video_Toolbox_NE/tool_dlna/si_stream.py)。
- **复用容差 + seek 冷却**：连续 Range 落在 `[cursor, cursor+1MB]` 复用；新 ffmpeg 启动加 200ms 防抖。
- **HEAD 请求**：单独实现 `HEAD /media_si/<key>`，**只回头（Content-Length=estimate、Accept-Ranges、
  Content-Type=video/mp4），绝不起 ffmpeg**。参考没有 HEAD，本工程 `/media` 有，需补。

### 3.3 "选择音轨"参数
tool_clonevoice 的"选择音轨"实为**叠加声道** `mix_channel ∈ {left,right,both}`（SI 叠到左/右/双声道）。
若用户还想选**原视频的第几条音轨**（多音轨源），则额外加 `audio_track_index`：
滤镜里 `[0:a:0]` 改为 `[0:a:{idx}]`，并用 `run_ffprobe_json` 列出音轨供下拉。
**建议**：首版 UI 至少给 `mix_channel`（对齐参考）；`audio_track_index` 作为可选项（默认 0，多音轨时才有意义）。
需在开工前与用户确认"音轨"指声道还是源音轨编号（见第 8 节待确认项）。

### 3.4 `duck_original`（降低原声）
参考 `SIMixConfig` 尚未带 `duck_original`，但 `build_si_mix_filter` 已支持。
移植时给 `SIMixConfig` 加 `duck_original: bool` 字段并透传到滤镜。

### 3.5 大小估算用本工程探针
参考用 `content_directory.probe_cached(video)` 返回 dict；本工程是 `pipeline.ffmpeg_io.probe_cached` 返回
`VideoInfo`（有 duration，无分流 size）。改造 `estimate_output_size`：
- `duration` ← `VideoInfo.duration`；
- `video_stream_size` ← 用 `run_ffprobe_json(["-show_entries","stream=codec_type,..."])` 或
  `文件总大小 - 估算原音频大小`；保守做法 `估算 = (video_size + 192k*dur/8) * 1.05`，下界取 `max(原文件size, 64KB)`。
- 估算结果**缓存**（按 mtime），避免每个 Range 都 ffprobe。

### 3.6 浏览缓存失效（易漏 bug）
`_dir_items_cache` 的 key 不含 SI 状态。必须把 **SI 运行时配置版本号**（见 3.7）加进 `cache_key` 元组
（[content_directory.py:766](dlna/content_directory.py)），否则切 toggle 后浏览结果不刷新。
另：`.si.wav` 是非视频文件，`media_index` 的目录 signature **可能不跟踪 .wav 变化** → 运行中**新增**一个
`.si.wav` 不一定触发重列。缓解：(a) SI 配置变更时清空 `_dir_items_cache`；(b) 文档注明"新增 si.wav 需重扫目录"；
(c) 进阶：让 media_index 把 `.si.wav` mtime 纳入 signature（可选，工作量稍大）。

### 3.7 热重载接入本工程范式（对齐 light_match）
- `utils/runtime_settings.py` 新增 `SIMixRuntime` dataclass（含 `version`）+ `get_si_mix()` / `set_si_mix()`
  （线程安全，`set` 时 version+1）。`SIStreamService` 读 `get_si_mix()` 拿当前配置，version 变化即 `reload_config()`
  杀旧 session。这样 **toggle 与四个参数都实时生效**，无需重启（满足用户硬约束）。
- `http_app/routes_control.py` 新增 `GET/PUT /control/si_mix`（沿用 `_is_local_request` 本地校验）。
- UI 在 toggle 切换、对话框保存时，用现有 `urllib` 后台线程 PUT（复制 `_send_light_match_live_update` 范式）。

### 3.8 VR 命名（用户明确要求）
`[SI]` 条目标题 = `"[SI]" + source_display_stem(path.stem, width, height)`，
从而 2:1 半球源得到 `_LR_180_SBS` 等后缀，VR 播放器据此进入 SBS/180 模式。
`resolution` 沿用原视频分辨率；`mime=video/mp4`、`dlna_pn=AVC_MP4_HP_HD_AAC`；外挂字幕沿用原条目逻辑。

---

## 4. 涉及文件与改动清单

| 文件 | 改动 | 估计行数 |
|---|---|---|
| `http_app/si_stream.py` *(新建)* | 移植参考 `si_stream.py`：`SIMixConfig`(+duck_original,+audio_track_index)、`LiveStreamSession`、`SIStreamService`；估算改用本工程探针；用 `hidden_subprocess_kwargs()`；滤镜函数内嵌 | +320 |
| `utils/si_filter.py` *(新建)* | 复制 `build_si_mix_filter` 纯函数 + 常量（避免跨工程导入） | +130 |
| `utils/runtime_settings.py` | 加 `SIMixRuntime` + `get_si_mix/set_si_mix`（版本号热重载） | +50 |
| `http_app/routes_control.py` | 加 `GET/PUT /control/si_mix`（本地校验） | +40 |
| `http_app/routes_media.py` | 加 `GET/HEAD /media_si/{name:path}`：Range 解析 + `StreamingResponse(async gen + to_thread.read)`；HEAD 只回头不起 ffmpeg | +120 |
| `http_app/server.py` | 启动时构造单例 `SIStreamService`（读初始配置注入 runtime_settings）；`lifespan` 关闭时 `service.shutdown()` | +20 |
| `dlna/content_directory.py` | `_video_items_from_index` 末尾按 `get_si_mix().enabled` + 同名 `.si.wav` 追加 `[SI]` 条目（`SI_ITEM_PREFIX="si_"`）；`_id_to_*`/BrowseMetadata 识别前缀；`cache_key` 加 SI version；标题走 `source_display_stem` | +80 |
| `ui/pages/home_page.py` | "其他配置"加"同声传译"行(toggle + "音轨设置"按钮)；新增 `SISettingsDialog`；保存写 `settings.data["si_*"]`；toggle/保存推 `PUT /control/si_mix` | +160 |
| `ui/settings.py` | `DEFAULTS` 增 `si_enabled/si_mix_channel/si_original_volume/si_volume/si_delay/si_duck_original/si_audio_track` | +10 |
| `ui/i18n.py` + 翻译 | 新增 SI 相关 i18n key（中/英/日） | +25 |
| `tests/test_si_stream.py` *(新建)* | 见第 6 节 | +220 |

**合计约 +1175 行（含测试）。**

---

## 5. 实施步骤（建议顺序）

1. **滤镜与配置基座**（0.5d）：`utils/si_filter.py` + `SIMixConfig` + `runtime_settings.SIMixRuntime`，单测滤镜与默认值。
2. **LiveStreamSession / SIStreamService 移植**（1d）：估算改本工程探针；真 ffmpeg 联调（短 mp4 + 短 si.wav）。
3. **控制路由 + 热重载**（0.5d）：`/control/si_mix` GET/PUT；version 变更触发 `reload_config`。
4. **媒体路由**（1d）：`GET/HEAD /media_si`，async 生成器 + `to_thread.read`；curl 验证字节、Range 206、HEAD 不起 ffmpeg。
5. **DLNA 浏览**（0.5d）：`[SI]` 平行条目 + 前缀解析 + 缓存键 + VR 命名；单测 browse 列/不列。
6. **UI**（0.5d）：同声传译行 + `SISettingsDialog` + 实时 PUT；持久化。
7. **i18n**（0.25d）。
8. **真机联调**（0.75～1.5d）：Quest/Skybox/HereSphere/电视 上验证拖动 seek、运行中切 toggle/改参数即时生效、
   多客户端、探针不卡、Content-Length 误差。

---

## 6. 测试计划（`tests/test_si_stream.py` + 浏览/控制测试）

1. `SIMixConfig` 默认值 / 边界 clamp（含 duck_original、audio_track_index）。
2. `si_filter.build_si_mix_filter` 各声道 + duck 开关输出稳定（快照断言）。
3. `has_si_source` 检测同名 `.si.wav`。
4. `estimate_output_size` 在合理范围（mock 探针）+ mtime 缓存命中。
5. `open_stream` 用 `-ss t` 起 ffmpeg（mock Popen，断言 `t≈range_start/total*duration`）。
6. Range 请求返回 206 + 正确 `Content-Range`；无 Range 返回 200。
7. 连续顺序 Range 复用同一 ffmpeg（只启一次）。
8. `GeneratorExit`（短连接）**不**回收 session；EOF 才回收。
9. `set_si_mix` version 自增 → service `reload_config` 杀 session。
10. `PUT /control/si_mix` 非本地 → 403；本地 → 下次 `current_config` 生效。
11. browse：enabled 且有 si.wav → 多一条 `si_` 条目；disabled / 无 si.wav → 不列。
12. BrowseMetadata `si_<rel>` → `[SI]` 标题 + 走 VR 命名。
13. 切换 SI version → `_dir_items_cache` 失效、重列。
14. HEAD `/media_si` 不创建 ffmpeg 进程（mock 断言）。
15. Range 解析：`bytes=1024-` → (1024,None)；畸形 → 兜底 (0,None)。

---

## 7. 边界与风险

| 场景 | 处理 |
|---|---|
| async 事件循环阻塞 | **必须** `await asyncio.to_thread(session.read, n)`，禁止协程内直读管道 |
| 播放器启动探针(HEAD/小range/尾部探) | HEAD 只回头；短连接 `GeneratorExit` 不杀 ffmpeg；EOF 才回收 |
| 频繁 seek | 200ms 冷却 + 旧 ffmpeg `terminate→wait` 后再起新 |
| Content-Length 估算偏差 | ×1.05 上浮；电视报损坏则调 ×1.10 或换 chunked |
| HEVC 源 | `-c:v copy` 透传；客户端不支持 HEVC 是源本身兼容性问题（与 /media 一致） |
| 关键帧间隔大 | `-ss` 跳最近关键帧，跟点偏几秒，与普通 MP4 拖动一致 |
| `.si.wav` 比视频短 | `amix=duration=first` 已处理：SI 结束后仅剩原声 |
| 路径含中日韩 | `Popen` stdout 保持二进制，勿 `text=True` |
| 运行中新增 si.wav | SI 配置变更清缓存可见；目录 signature 不跟踪 .wav → 文档注明需重扫（或让 index 纳入 .wav mtime） |
| 多客户端同播同一 [SI] | 按 `(video, client_ip)` 分 session，各一个 ffmpeg，互不干扰 |
| 同进程内服务器 | 热重载走内存 `runtime_settings`（非 HTTP 也可直接调 `set_si_mix`），但保留 `/control` 端点与 light_match 对齐 |
| 与现有 passthrough 并发 | `/media_si` 不走 `_active_streams`/NVENC 槽位（纯 ffmpeg copy + aac，CPU 轻量），独立计数即可；如需限流可加独立信号量 |

**回滚**：核心隔离在 `http_app/si_stream.py` 单模块；UI/浏览/配置改动与方案无关，必要时仅替换该模块或关 toggle 即停用。

---

## 8. 开工前待用户确认（默认值已选，不反对即照做）

- **A. "选择音轨"语义**：✅ **已确认 = 叠加声道 `left/right/both`**（对齐 tool_clonevoice，SI 叠到左/右/双声道）。
  **不做**原视频音轨编号选择，`SIMixConfig` 不需要 `audio_track_index` 字段，滤镜固定用 `[0:a:0]`。
  （第 3.3 节中关于 `audio_track_index` 的可选项作废。）
- **B. 降低原声默认值**：tool_clonevoice 默认勾选(True)。本工程默认建议 **开(True)** 以突出 SI 人声。
- **C. Content-Length 上浮系数**：默认 ×1.05。
- **D. `DLNA.ORG_CI`**：默认 `CI=1`（转码标记），老电视播不动再改 0。
- **E. session 维度**：默认按 `(video, client_ip)`；若同一 IP 多播放器冲突，可加细分。

---

## 9. 给开发人员的落地提示（避免踩坑）

1. 先把参考 `tool_dlna/si_stream.py` 整文件读懂——本工程 80% 逻辑照搬，重点改 `read` 的 async 包装、
   `estimate_output_size` 的探针来源、`hidden_subprocess_kwargs()`、`SIMixConfig` 加两字段。
2. 热重载严格对齐 `light_match` 全链路：`runtime_settings`(版本号) → `/control` 本地路由 → UI 后台线程 PUT。
   照抄 [home_page.py:1002 `_send_light_match_live_update`](ui/pages/home_page.py) 即可。
3. `_dir_items_cache` 的 cache_key 一定要加 SI version，否则"运行中切 toggle 不刷新"会被当成 bug 反复提。
4. HEAD 路由必须存在且不起 ffmpeg；很多 DLNA 客户端先 HEAD 后 GET。
5. 标题务必走 `source_display_stem`（不是 `path.stem`），否则 VR 播放器进不了 SBS/180 模式。
6. 真机联调以 Quest/Skybox + 一台 DLNA 电视为准；重点验证"运行中改参数即时变声"和"拖动 seek 不花屏"。
