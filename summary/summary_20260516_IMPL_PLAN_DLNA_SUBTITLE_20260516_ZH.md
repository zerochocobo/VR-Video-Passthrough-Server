# DLNA 软字幕投递实施方案（中文）

- 日期：2026-05-16
- 范围：让用户播放普通 MP4 时也能拿到字幕轨，最小化新增开销，并顺手消解 MKV 卡死路径。
- 不在范围：硬字幕烧录、字幕翻译/OCR、新版 UI。

---

## 1. 现状研判

- `routes_media.py:/media/{name}` 走 `FileResponse`/Range，**只**回吐源 MP4 字节，无任何字幕通道。
- `dlna/content_directory.py` 的 `_video_items_from_index()` 和 `_didl_for()` 输出标准 DIDL，目前只挂一个 `<res>` 元素（视频本身）+ albumArtURI（缩略图）。
- DLNA 协议本身是支持外挂字幕的：
  1. **Samsung/SEC 扩展**：`<sec:CaptionInfo>` / `<sec:CaptionInfoEx>` 元素 + HTTP 头 `CaptionInfo.sec` / `getCaptionInfo.sec`。多数智能电视、部分安卓 DLNA 播放器认它。
  2. **DLNA 标准做法**：`<item>` 内附加第二个 `<res protocolInfo="http-get:*:application/x-subrip:*">subURL</res>`。
- Quest3 端常见播放器：DeoVR / Pigasus / Skybox / SKYBOX 等。**它们对 DLNA 字幕扩展的识别度未实测**，必须放在 P0 验证。
- 用户当前为塞软字幕而切到 MKV，命中了 PyNv `SimpleDecoder` 在 Matroska 容器上的 seek+demux 卡死问题（详见 `summary/summary_20260516_MKV_PYNV_SIMPLEDECODER_STUCK_ISSUE_CN.md`）。如果有路径让用户改回 MP4，MKV 卡死问题自然消失。

---

## 2. 总体策略

按"零侵入 → 容器层重封装 → MKV 替换"三档推进：

| 阶段 | 内容 | 触发条件 | 主要代价 |
|------|------|----------|----------|
| Phase 1 | DLNA 字幕扩展元数据 + 静态 `/subs` 端点 | 立即 | 元数据 + 几 KB 文本 |
| Phase 2 | 按需 mp4 remux 缓存（视频/音频 -c copy + mov_text） | Phase 1 客户端不识别 | 首播 3–10s，磁盘 ≈ 1× 原文件 |
| Phase 3 | MKV → MP4 自动 remux（提取内嵌字幕，落到 Phase 2 缓存） | Phase 2 落地后 | 同 Phase 2 |

每阶段独立可发布，前一阶段达标可不进入下一阶段。

---

## 3. Phase 1：DLNA 字幕扩展（P0，零编解码开销）

### 3.1 目标
让 DLNA 客户端通过协议本身知道"这个视频有外挂字幕"，无需改原 MP4 字节。

### 3.2 字幕发现规则（建议）
- 与视频同目录、同 stem 的字幕文件：`name.srt`、`name.ass`、`name.vtt`、`name.ssa`。
- 多语言：`name.zh.srt`、`name.chi.srt`、`name.eng.srt`、`name.zh-CN.ass`。
- 字幕子目录约定：`name/subs/*.srt`（兼容 jellyfin/plex 习惯）。
- 命中多个时按优先级排序（中文优先 → 英文 → 其他），第一个挂为默认 caption，其余按 `xml:lang` 全部挂上。

### 3.3 新增 HTTP 端点
- `GET /subs/{rel}`：原样回吐字幕文件，支持 Range/HEAD。
  - mime：srt → `application/x-subrip`，vtt → `text/vtt`，ass → `application/x-ass`。
  - 加 `Content-Disposition: inline`、`Access-Control-Allow-Origin: *`。
- `HEAD /subs/{rel}`：仅返回头，便于客户端探测大小/存在性。

### 3.4 Content-Directory 修改点
- `dlna/content_directory.py`
  - DIDL 顶部命名空间增补 `xmlns:sec="http://www.sec.co.kr/"`（`_didl_for()` 起始 `<DIDL-Lite>` 标签）。
  - `_video_items_from_index()` 在原始 mp4 item 字典里多挂 `subtitles: list[dict]`，每条含 `url`、`lang`、`type`(srt/ass/vtt)、`mime`。
  - `_didl_for()` 渲染 item 时，如有字幕：
    - 在 `<res>视频</res>` 之后追加：
      ```
      <res protocolInfo="http-get:*:application/x-subrip:*"
           xml:lang="zh-CN">{subURL}</res>
      ```
    - 同时附加 SEC 节点（部分 LG/三星系必需）：
      ```
      <sec:CaptionInfoEx sec:type="srt">{subURL}</sec:CaptionInfoEx>
      <sec:CaptionInfo sec:type="srt">{subURL}</sec:CaptionInfo>
      ```

### 3.5 Media 路由头
- `routes_media.py:/media/{name}` 与 `/media/{name}` HEAD：
  - 若该视频探测到字幕，响应头追加：
    - `CaptionInfo.sec: <第一个字幕URL>`
    - `getCaptionInfo.sec: 1`
  - 不影响 Range/Content-Length，纯增量字段。

### 3.6 验收
- BubbleUPnP / VLC DMR / Kodi DLNA 客户端能列出字幕轨并切换。
- Quest3 上 DeoVR / Pigasus / Skybox 至少 1 个识别。识别率作为 Phase 2 决策依据。
- `/media` 与 `/subs` 的 Range 行为不退化。

### 3.7 风险与回退
- **Quest 播放器不识别**：进入 Phase 2。
- **响应头大小写敏感**：`CaptionInfo.sec` 是 SEC 规范原样大小写，必须保留。
- **字幕乱码**：srt 文件按 utf-8/gbk 嗅探，必要时实时转 utf-8 输出（仍属 Phase 1）。

### 3.8 工时估计
半天到 1 天。

---

## 4. Phase 2：按需 remux MP4 缓存（P1，兼容性兜底）

### 4.1 目标
当 Phase 1 客户端不识别外挂字幕时，把字幕"硬封软字幕"进 MP4 容器，几乎不动音视频码流。

### 4.2 缓存结构
- 目录：`cache/subbed/`（与现有 `debug_output/`、`cache/` 同级；如已有缓存基类沿用之）。
- 文件名：`<sha1(src.mtime|src.size|sub_paths.mtimes)>.mp4`。
- 索引：`cache/subbed/index.json`（path → entry meta），LRU + TTL 双策略，默认上限 50 GB / 30 天。

### 4.3 触发与生成
- `routes_media.py:/media/{name}`：
  1. 检查同名字幕。
  2. 若存在且缓存未命中 → 同步触发 ffmpeg remux（本进程一锁一文件，避免并发重复）。
  3. 命中缓存 → 走缓存文件 `FileResponse`，对外 URL 不变。
- ffmpeg 命令模板：
  ```
  ffmpeg -y -nostdin -hide_banner -loglevel error \
    -i "<src.mp4>" \
    -sub_charenc UTF-8 -i "<sub.srt>" \
    -map 0:v -map 0:a? -map 1:s \
    -c copy -c:s mov_text \
    -metadata:s:s:0 language=chi -metadata:s:s:0 title="zh-CN" \
    -movflags +faststart \
    "<cache.mp4>"
  ```
- ASS/SSA：先 `ffmpeg -i sub.ass sub.srt` 转 srt 再嵌（mov_text 不支持样式）。需要保留样式时改用 mkv 容器，但与本方案目标冲突，不做。
- 多字幕轨：循环 `-map 1:s -map 2:s ...`，每条加 `language` metadata。

### 4.4 Content-Directory 调整
- `_video_items_from_index()` 给 mp4 item 的 `url` 字段维持 `/media/{rel}`，由路由内部决定走源文件还是缓存。客户端无感。
- `bitrate`/`size` 用缓存文件实际值（首次必须计算后再回 DIDL），否则 Range/进度条会偏。

### 4.5 失效与清理
- 字幕文件 mtime/大小变化 → 缓存失效（hash 改变，旧文件留给 LRU 清理）。
- 源 mp4 mtime 变化 → 同上。
- `tools/subbed_cache_gc.py` 清理脚本（可选）。

### 4.6 性能预期
- 8K HEVC 60s mp4 + srt：remux 约 5–10s（IO-bound）。
- 4K H.264 90min mp4 + srt：remux 约 30–90s。
- CPU 几乎为零，瓶颈是磁盘吞吐。
- 后续 Range 请求与原始 mp4 完全等价。

### 4.7 风险
- **磁盘占用翻倍**：必须有 GC。
- **首播延迟**：8K 长片可能让用户等待。可通过 background 预 remux（Phase 1 落地时同步加 worker，扫库时预生成）缓解。
- **NPlayer / DeoVR 对 mov_text 渲染样式弱**：可接受，文字字幕本就无样式诉求。

### 4.8 工时估计
1.5–2 天。

---

## 5. Phase 3：MKV → MP4 自动 remux（P1，与 MKV 卡死方案合流）

### 5.1 目标
让媒体目录里的 .mkv 在 DLNA 列表中表现为 .mp4，并走 Phase 2 缓存路径。彻底绕过 PyNv `SimpleDecoder` 的 Matroska 卡死。

### 5.2 处理流程
1. 索引扫描时识别 .mkv，探测内嵌字幕轨与音视频轨。
2. 后台 remux：
   ```
   ffmpeg -y -i "<src.mkv>" \
     -map 0:v:0 -map 0:a? -map 0:s? \
     -c:v copy -c:a copy -c:s mov_text \
     -movflags +faststart \
     "<cache.mp4>"
   ```
   - 若内嵌字幕是图形字幕（PGS/VOB）→ 不能转 mov_text，需丢弃或单独 OCR（本方案默认丢弃，落后续）。
   - 若视频不是 H.264/HEVC → `-c:v copy` 仍可成功（mp4 容器支持 H.264/HEVC/AV1），不行则降级为 mkv 直出（保留原行为，标记 needs_fix）。
3. DLNA 列表用 cache 路径 + 原 stem，对外仍是单一 mp4。
4. 缓存命名键加上"src 是 mkv"标识，避免与 Phase 2 的 mp4+srt 缓存冲撞。

### 5.3 与 5/16 MKV 卡死方案的关系
- 当前 `_hide_passthrough_for_path()` 已对 `mkv_needs_fix` 做了隐藏；passthrough 的 SimpleDecoder 路径不再吃带"坏 cues"的 mkv。
- Phase 3 进一步把"不坏的 mkv"也搬走，让 PyNv 路径**完全不接触 Matroska 容器**。这等价于 MKV 卡死 summary 中的"方案 F：自动 remux"，落地点合并到字幕缓存即可。

### 5.4 风险
- 首次扫库会引发批量 remux 风暴 → 用串行 worker + 优先队列。
- MKV 文件极大时磁盘吃紧 → GC 必须跟上。

### 5.5 工时估计
0.5–1 天（在 Phase 2 之上）。

---

## 6. 优先级与里程碑

| 优先级 | 项 | 阶段 | 完成判定 |
|--------|----|------|----------|
| P0 | DLNA 字幕扩展（Phase 1） | Day 1 | Quest 实测识别率 ≥ 1 个主流播放器 |
| P0 | 字幕嗅探/编码转码 | Day 1 | utf-8 输出，srt/ass/vtt 三种类型 |
| P1 | mp4+srt remux 缓存（Phase 2） | Day 2–3 | 目标客户端默认显示字幕 |
| P1 | LRU/TTL GC | Day 3 | 缓存上限可控 |
| P1 | MKV → MP4 自动 remux（Phase 3） | Day 4 | DLNA 列表零 mkv |
| P2 | 后台预 remux worker | Day 5 | 首播延迟从秒级降到 0 |
| P2 | 多字幕轨语言切换 UI 验证 | Day 5 | 三语字幕能切换 |

---

## 7. 关键代码触点（仅列位置，不动代码）

- `dlna/content_directory.py:438-510` `_didl_for()`：DIDL 渲染入口，需扩展 res/sec 字段。
- `dlna/content_directory.py:235-330` `_video_items_from_index()`：item 字典构造点，需注入 `subtitles`。
- `http_app/routes_media.py` `/media/{name}` 处理函数：需挂 CaptionInfo.sec 头 + 选择源文件 vs 缓存。
- `http_app/routes_media.py` 顶部 router：新增 `/subs/{rel}` GET/HEAD。
- `utils/media_index.py` `IndexedChild`：可在 `child.video` 里加 `subtitles: tuple[Path, ...]`，让 DIDL 构造省一次磁盘扫描。
- `cache/subbed/`：新增缓存目录与索引文件。

---

## 8. 验收清单

- [ ] BubbleUPnP DMR 列出字幕轨可切换
- [ ] Quest3 + DeoVR/Pigasus/Skybox 至少一个能显示字幕
- [ ] `/media/{name}` Range/HEAD 行为与现状一致
- [ ] `/subs/{rel}` Range/HEAD 行为正确
- [ ] mp4+srt remux 缓存命中后 Range 进度条正确
- [ ] 缓存 GC 不误删活跃文件
- [ ] mkv 文件在 DLNA 列表中以 mp4 暴露
- [ ] PyNv passthrough 路径不再触发 Matroska 卡死
- [ ] 全量回归：现有 8K passthrough、live chapter、alpha live 行为不退化

---

## 9. 与 8K 40fps 计划的关系

- 字幕方案与 `IMPL_PLAN_8K_40FPS_20260515` 解耦，可并行推进，无共享代码热区。
- Phase 3 的 MKV→MP4 remux 与 8K 计划的 MKV 处理策略形成互补：8K 计划负责"已经在播的 mkv 不卡死"，本方案负责"从源头消除 mkv"。
- 两者都不影响 PyNv 三阶段流水线、CUDA Graph、TRT EP 等核心改造。
