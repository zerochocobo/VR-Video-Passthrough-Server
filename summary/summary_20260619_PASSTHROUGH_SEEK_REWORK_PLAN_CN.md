# passthrough_seek 改造计划

日期: 2026-06-19
状态: 设计计划

当前状态更新: 本文原始目标仍然是“虚拟 MP4 文件系统 / GOP slot cache”，也就是朝实时按需生成方向推进。2026-06-19 已完成的是 `/passthrough_seek` 的 VMP4 入口、source-size 声明、完整 cache-file 真机验证、缺失缓存自动构建、oversize 自动降码率重建和基础日志诊断。尚未完成的是真正的 GOP slot backend；当前成功路径仍依赖“完整 MP4 cache 先生成完成，再作为稳定文件服务”。

## 1. 目标

把旧的 `/passthrough_seek` 从“Range 映射到时间后重新启动实时流”的伪 VOD 方案，改造成一个更接近真实文件语义的虚拟 MP4 方案。

用户目标:

- 文件声明大小与原始视频一致。
- 文件声明时长与原始视频一致。
- VR 播放器拖动进度条时，服务端能根据播放器请求的 byte Range 定位到对应时间段并继续播放。
- 保留 `/passthrough_live` 作为稳定 fallback，不破坏现有直播播放路径。

## 2. 核心判断

这条路可以继续走，但不能只改 `Content-Length`、DLNA `size` 和 `duration`。

旧方案失败的根因是: 播放器把 MP4 当作一份 byte-stable 文件读取。它会先读 `moov`，再根据 `stts/ctts/stss/stsz/stsc/co64` 算出目标时间对应的字节 offset，然后发 HTTP Range。服务端必须让这个 Range 返回的字节与虚拟 `moov` 描述的布局一致。

可行方向不是“伪造平均码率后随便返回新流”，而是“设计一个固定 slot 的全虚拟 MP4 文件系统”:

- `moov` 描述我们自己定义的固定 GOP/chunk 布局。
- `co64` 指向每个虚拟 GOP slot 的起点。
- `stts` 描述固定 FPS 时间轴。
- `stss` 只标记每个 GOP slot 首帧为关键帧。
- `stsz/stsc` 必须与 slot 内的实际编码字节组织相匹配。
- 请求某个 byte offset 时，服务端按 slot 编号定位时间，生成或读取对应 slot 字节。

## 3. 建议架构

### 3.1 直接挂载现有 `/passthrough_seek`

不再新增 prototype URL。新的 VMP4 实验直接挂到现有:

```text
/passthrough_seek/{name:path}
```

理由:

- 旧 `/passthrough_seek` 已经证明原伪 VOD 路线不可行，继续保留为主要实验入口没有价值。
- VR 真机测试可以沿用已有 DLNA/手动 URL 入口，不需要播放器重新适配另一个 route。
- 请求历史、UA profile、active slot、Matter 管理、live fallback 等现有诊断资产都能复用。

但必须用强开关隔离，例如:

```text
PT_PASSTHROUGH_SEEK_VMP4=0|1
```

默认关闭。开启后 `/passthrough_seek` 进入新的虚拟 MP4 slot 模式；关闭时维持当前旧实验行为或直接禁用。

现有开关继续保留:

```text
PT_PASSTHROUGH_SEEK_ENABLED=0|1
PT_PASSTHROUGH_SEEK_DLNA=0|1
PT_PASSTHROUGH_SEEK_ROUTE_POLICY=profile|all|off
PT_PASSTHROUGH_SEEK_PROFILES=...
```

首轮真机测试建议:

```text
PT_PASSTHROUGH_SEEK_ENABLED=1
PT_PASSTHROUGH_SEEK_VMP4=1
PT_PASSTHROUGH_SEEK_DLNA=0
PT_PASSTHROUGH_SEEK_ROUTE_POLICY=all
PT_PASSTHROUGH_SEEK_CONTAINER=mp4
```

也就是: HTTP route 可手工访问，DLNA 暂不自动展示，避免误伤日常浏览。

### 3.2 虚拟文件布局

建议固定为:

```text
ftyp + moov + mdat
```

不要用在线 fMP4 `empty_moov + moof` 伪装普通 MP4。

初版建议采用 GOP slot:

```text
slot_duration_sec = PASSTHROUGH_GOP / output_fps
slot_count = ceil(source_duration / slot_duration_sec)
mdat_payload_size = source_file_size - len(ftyp) - len(moov) - mdat_header_size
slot_size = floor(mdat_payload_size / slot_count)
```

总虚拟大小:

```text
virtual_size = source_file_size
```

时长:

```text
virtual_duration = source_duration
```

### 3.3 Range 到 slot 的映射

播放器请求:

```http
Range: bytes=N-
```

服务端逻辑:

1. 如果 `N < len(ftyp+moov+mdat_header)`，返回稳定 init bytes。
2. 否则计算:

```text
slot_index = floor((N - mdat_payload_start) / slot_size)
slot_start_time = slot_index * slot_duration_sec
inside_slot_offset = (N - mdat_payload_start) % slot_size
```

3. 找到或生成该 slot 的 MP4 `mdat` payload 字节。
4. 从 `inside_slot_offset` 开始返回，并补齐/截断到 HTTP Range 需要的长度。

### 3.4 slot 字节生成

初版建议“按需生成 + 缓存”:

- 每个 slot 对应一个短时长 passthrough 输出片段。
- slot 生成必须从关键帧/IDR 开始。
- slot payload 小于 `slot_size` 时，用合法填充补足。
- slot payload 大于 `slot_size` 时，该 slot 失败；需要降低码率重试，或扩大 slot 策略后重建整个布局。

缓存键建议包含:

- 源文件 stat key。
- passthrough mode: `green|alpha|two_dvr`。
- 输出 FPS。
- GOP。
- 编码器参数/码率。
- slot duration。
- source-size virtual layout schema version。

## 4. 最大技术难点

### 4.1 `stsz` 不能随便平均

MP4 的 `stsz` 是 sample size 表。播放器可能按 sample 边界切包。如果 `stsz` 声称每帧固定大小，但 slot 内实际 HEVC NAL 不是这个大小，解码器会读错边界。

更稳的初版策略是:

- 不做逐帧平均 sample。
- 把每个 GOP slot 作为少量大 chunk/sample 组织。
- 尽量让播放器 seek 落到 `stss` 关键帧 chunk 起点。

这需要验证目标播放器是否接受这种粗粒度 sample/chunk 结构。

### 4.2 padding 必须合法

如果 padding 落在 sample payload 内，不能直接填 0。可选方案:

- 使用合法 HEVC filler NAL。
- 或把 padding 设计成 sample 外的 `free/skip` 区域，但 `co64/stsz` 必须保证播放器不会把它当媒体 sample 解码。

初版更建议使用合法 filler NAL 或让 slot payload 自身形成可解码样本序列。

### 4.3 音频不能忽略

如果虚拟 MP4 包含音频，音频 sample 的 `stts/stsz/co64` 也必须自洽。

降低复杂度的实验选项:

1. 首版做 video-only 虚拟 MP4，验证拖动和视频可解码。
2. 通过后再加音频 slot/interleave。

但最终生产方案必须解决音频，否则 VR 播放不可用。

### 4.4 源大小约束会反向约束码率

如果总文件大小必须等于源文件大小，那么输出总码率必须小于等于源平均码率预算。

当前 PyNv HEVC 默认目标是 `50M`，并按 `source_bps * 2.0` 封顶。这不等于源文件大小预算。新模式需要独立码率策略:

```text
target_total_bps = source_file_size * 8 / source_duration
target_video_bps = target_total_bps - audio_bps - container_overhead_bps
```

并留安全余量，避免 slot 溢出。

## 5. 实施阶段

### 阶段 0: 直接接入 `/passthrough_seek`，但不暴露 DLNA

目标:

- 新增 `PT_PASSTHROUGH_SEEK_VMP4`。
- `/passthrough_seek` 在该开关开启时进入 VMP4 backend。
- 不新增新 route。
- 不改 `/passthrough_live`。
- 不自动暴露 DLNA seek item。
- 用一个短测试片构造 `ftyp+moov+mdat` 虚拟布局。
- 支持 `HEAD`、`GET bytes=0-`、init 小范围和某个固定 slot range。

验收:

- `ffprobe` 能识别虚拟 MP4。
- `ffplay` 或 VLC 能从头播放短样片。
- 任意重复请求同一 Range 返回稳定字节。
- 关闭 `PT_PASSTHROUGH_SEEK_VMP4` 后旧行为不受影响。

### 阶段 1: video-only GOP slot

目标:

- 只处理视频。
- 固定 GOP slot。
- slot 按需生成并缓存。
- Range 到 slot 时间映射稳定。

验收:

- Range 请求落到不同 slot 时能继续解码。
- 五个 Quest3 VR 播放器至少有一个能显示时长并拖动。
- 请求历史能证明播放器按 MP4 seek map 请求中段 Range。

### 阶段 2: 加音频

目标:

- 为音频建立对应 sample/chunk 表。
- 音频和视频 slot 对齐或合理 interleave。
- 保证拖动后 A/V 同步。

验收:

- 从头播放和中段 seek 后均有声音。
- 目标 VR 播放器无明显 A/V sync 问题。

### 阶段 3: 源大小等长约束

目标:

- 总虚拟大小固定为源文件大小。
- 输出 slot 不超过预算。
- 不足部分用合法 padding 或安全 gap 补齐。

验收:

- HTTP `Content-Length == source.stat().st_size`。
- `Content-Range` total 等于源文件大小。
- DLNA DIDL `size` 等于源文件大小。
- `duration` 等于源视频时长。

### 阶段 4: 扩展完整 passthrough seek 行为

目标:

- 完善 `/passthrough_seek` 的 VMP4 backend，使其覆盖 green/alpha/2DVR 的目标测试场景。
- 旧 TS/fMP4 伪 VOD 逻辑保留为可回退实验路径，或在确认 VMP4 路线成立后移除。
- 默认仍不在 DLNA 暴露。

验收:

- 手动 URL 测试通过。
- 请求失败时不影响 `/passthrough_live`。
- 资源清理、slot cache 清理、并发限制正常。

### 阶段 5: DLNA 灰度暴露

目标:

- `PT_PASSTHROUGH_SEEK_ENABLED=1` 且 `PT_PASSTHROUGH_SEEK_DLNA=1` 时额外暴露 seek MP4 item。
- live item 继续保留。
- route 层按 profile 白名单守门。

验收:

- 五个 Quest3 VR 播放器重新 Browse 后能看到 seek item。
- seek item 可拖动。
- live fallback 仍可播放。

## 6. 测试与诊断

必须打开或继续使用:

- `debug_output/request_history/*.jsonl`
- passthrough seek 诊断头:
  - `X-Passthrough-Seek-Time`
  - `X-Passthrough-Seek-Ratio`
  - `X-Passthrough-Mode`
  - 新增建议:
    - `X-Passthrough-VMP4-Slot`
    - `X-Passthrough-VMP4-Slot-Time`
    - `X-Passthrough-VMP4-Layout-Size`

必须测试:

- `HEAD` 无 Range。
- `GET bytes=0-`。
- init 小范围 `bytes=0-65535`。
- 中段 open range。
- 尾部 range。
- 重复同一 Range，确认字节稳定。
- 播放器拖动多次，确认不会触发 stale active slot 或 Matter 泄漏。

## 7. 风险

- 目标播放器可能不接受粗粒度 GOP sample/chunk。
- HEVC filler/padding 处理不当会导致解码失败。
- 音频 interleave 可能成为比视频更难的问题。
- 如果源片本身码率偏低，保持源文件大小会迫使 passthrough 输出码率过低，画质可能明显下降。
- 如果 slot 按需生成太慢，播放器可能超时；需要预热或后台生成热点 slot。

## 8. 当前推荐结论

这条路线值得做，但它应该被定义为“虚拟 MP4 文件系统 / GOP slot cache”项目，而不是旧 `/passthrough_seek` 的小修。

最小可行实验应先做 video-only、固定 GOP slot、源大小虚拟 layout。只要能让一个目标 VR 播放器按 MP4 seek map 成功拖动，就证明路线成立；之后再补音频、码率预算、DLNA 暴露和生产级缓存。

## 9. 2026-06-19 开发切片 1: VMP4 source-size harness

已开始把新路线直接接入 `/passthrough_seek`。

本切片完成:

- 新增配置:

```text
PT_PASSTHROUGH_SEEK_VMP4=0|1
```

- 默认仍为关闭，不影响现有 `/passthrough_live` 和旧 seek 行为。
- 开启后，`/passthrough_seek` 强制解析为 `mp4` container。
- 开启后，HTTP `Content-Length` / `Content-Range total` 使用源文件大小，而不是旧的 `header_reserved + estimated_output_size`。
- 开启后，响应增加诊断头:
  - `X-Passthrough-VMP4: 1`
  - `X-Passthrough-VMP4-Phase: source-size-harness`
  - `X-Passthrough-VMP4-Layout-Size`
  - `X-Passthrough-VMP4-Source-Size`
  - `X-Passthrough-VMP4-Duration`
  - `X-Passthrough-Mode: seek-vmp4-mp4-<mode>`
- DLNA seek item 在 `PASSTHROUGH_SEEK_VMP4=1` 时强制使用:
  - `.seek.mp4`
  - `video/mp4`
  - `HEVC_MP4_MAIN`
  - `size = source file size`

重要限制:

- 当前还不是最终 byte-stable GOP slot MP4。
- 媒体字节生成仍复用当前 passthrough producer。
- 这个切片的目的是真机验证同一 `/passthrough_seek` 入口下，播放器对 source-size MP4 seek item 的请求模式和 UI 行为。
- 下一步必须实现真正 slot/cache backend，替换当前 producer 输出。

建议真机测试配置:

```powershell
$env:PT_PASSTHROUGH_SEEK_ENABLED="1"
$env:PT_PASSTHROUGH_SEEK_VMP4="1"
$env:PT_PASSTHROUGH_SEEK_ROUTE_POLICY="all"
$env:PT_PASSTHROUGH_SEEK_DLNA="0"
```

如果需要从 DLNA 目录直接看到 seek item:

```powershell
$env:PT_PASSTHROUGH_SEEK_DLNA="1"
```

2026-06-19 更新:

- 这些真机测试开关现在已经默认开启，GUI 启动服务也会显式传入:
  - `PT_PASSTHROUGH_SEEK_ENABLED=1`
  - `PT_PASSTHROUGH_SEEK_DLNA=1`
  - `PT_PASSTHROUGH_SEEK_ROUTE_POLICY=all`
  - `PT_PASSTHROUGH_SEEK_CONTAINER=mp4`
  - `PT_PASSTHROUGH_SEEK_VMP4=1`
- 旧的 `runtime_cache/ui_settings.json` 会通过一次性 migration
  `20260619_seek_vmp4_default_on` 切到上述默认值。
- 全新 settings 也会标记该 migration 已完成；后续如果需要关闭测试路线，可以在 GUI 保存关闭状态，migration 不会反复覆盖。

验证:

```text
python -m py_compile ... passed with PYTHONPYCACHEPREFIX=debug_output\pycache_compile
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_routes_media_cache.py tests\test_content_directory_modes.py tests\test_config_defaults.py -q
93 passed, 4 subtests passed
```

## 10. 2026-06-19 开发切片 2: cache-file VMP4 真机测试分支

已把 `/passthrough_seek` 的 VMP4 分支从“只改声明大小的 harness”推进到可真机测试的 cache-file 路径。

当前行为:

- `PT_PASSTHROUGH_SEEK_VMP4=1` 时，`/passthrough_seek` 不再回落到旧 realtime producer。
- route 会在源片同目录寻找已生成的 passthrough MP4:
  - green: `*_passthrough.mp4` / VR 命名后的 offline passthrough 名称。
  - alpha: alpha offline passthrough 名称。
  - two_dvr: `*_3D_LR_Screen.mp4` / 2DVR offline 输出名称。
- 找到 cache 后，HTTP/DLNA 声明大小仍等于源文件大小。
- cache 文件小于源文件时，不再补裸零字节，而是虚拟追加合法顶层 MP4 `free` box 到源文件大小。
- cache 文件大于源文件时，返回 `409 cache-oversize`。
- cache 文件时长和源片时长差异超过容差时，返回 `409 cache-duration-mismatch`，避免把 segment 输出误当全片。
- cache MP4 必须是 faststart，即顶层 `moov` 在 `mdat` 前；否则返回 `409 cache-moov-after-mdat` 或相关 moov probe 错误。
- cache 缺失时，返回 `503 cache-missing`，并通过 `X-Passthrough-VMP4-Expected-Cache` 提示期望文件名。
- `Range: bytes=0-` 保持启动兼容，返回 `200` 且不带 `Content-Range`。
- 非法 Range 在 VMP4 分支中返回带诊断头的 `416 range-unsatisfiable`。
- 诊断头中的 cache 文件名做 percent-encoding，避免中文文件名造成 HTTP header 编码错误。

新增关键诊断头:

- `X-Passthrough-VMP4-Phase`
- `X-Passthrough-VMP4-Cache`
- `X-Passthrough-VMP4-Expected-Cache`
- `X-Passthrough-VMP4-Cache-Size`
- `X-Passthrough-VMP4-Pad`
- `X-Passthrough-VMP4-Pad-Bytes`
- `X-Passthrough-VMP4-Cache-Duration`
- `X-Passthrough-VMP4-Duration-Delta`
- `X-Passthrough-VMP4-Duration-Tolerance`
- `X-Passthrough-VMP4-Moov-Offset`
- `X-Passthrough-VMP4-Mdat-Offset`

DLNA 暴露调整:

- `PT_PASSTHROUGH_SEEK_VMP4=1` 且 seek DLNA 开启时，即使同目录已有离线输出，也继续显示 seek MP4 item。
- 2DVR 可通过 `PT_PASSTHROUGH_OUTPUT_MODE=two_dvr` 暴露 `.seek.mp4` item，id 前缀为 `s3_`。
- live fallback 仍保留；2DVR 如果已有离线 cache，则只显示 seek item，避免再暴露不适用的 live 生成项。

本地真实文件 smoke:

- `videos/test_4k2d.mp4` 能自动发现 `videos/test_4k2d_S000030_E000330_3D_LR_Screen.mp4`。
- 因该 cache 是短 segment，响应级 smoke 返回:
  - `status=409`
  - `X-Passthrough-VMP4-Phase=cache-duration-mismatch`
  - 这符合预期，说明当前分支不会用 segment cache 冒充全片。

当前真机测试前置条件:

- 需要准备一个和源片同目录、全片时长匹配、大小不超过源文件大小、并且 `moov` 在 `mdat` 前的 faststart passthrough MP4 cache。
- 如果 cache 小于源片，route 会用虚拟 `free` box 补齐到源文件大小。
- 如果没有满足条件的全片 cache，真机会看到明确的 `503/409`，不是旧 realtime passthrough。

验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_routes_media_cache.py tests\test_content_directory_modes.py tests\test_config_defaults.py -q
103 passed, 4 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_settings.py tests\test_config_defaults.py tests\test_content_directory_modes.py tests\test_routes_media_cache.py -q
122 passed, 8 subtests passed
```

## 11. 2026-06-19 状态校准: 已完成项与后续修正

本节用于把原计划和当前实现重新对齐。当前可播放成功不等于实时 slot backend 已完成；它证明的是 VMP4 seek 入口、source-size 声明和 byte-stable MP4 cache 在真机上成立。

### 11.1 已完成

- 直接接入现有 `/passthrough_seek`。
  - 已新增并默认开启 `PT_PASSTHROUGH_SEEK_VMP4=1`。
  - GUI 启动服务会显式传入 VMP4 seek 相关环境变量。
  - `/passthrough_seek` 在 VMP4 模式下强制使用 MP4 container。

- HTTP/DLNA 声明层已完成。
  - HTTP `Content-Length` / `Content-Range total` 使用源文件大小。
  - DLNA seek item 使用 `.seek.mp4`、`video/mp4`、`HEVC_MP4_MAIN`。
  - DIDL `size` 使用源文件大小。
  - 响应头保留源时长诊断。

- cache-file VMP4 backend 已完成并通过初步真机验证。
  - route 会在源片同目录寻找 green/alpha/two_dvr 的完整 passthrough MP4 cache。
  - cache 必须是全片时长匹配。
  - cache 必须是 faststart，`moov` 在 `mdat` 前。
  - cache 小于源文件时，服务端虚拟追加顶层 MP4 `free` box 补齐到源文件大小。
  - cache 大于源文件时会进入 oversize 处理。
  - `Range: bytes=0-` 保持启动兼容，返回 `200`。
  - 普通中段/尾段 Range 返回 `206`。

- 缺失缓存自动构建已完成。
  - `PT_PASSTHROUGH_SEEK_VMP4_BUILD_MISSING=1` 默认开启。
  - cache 缺失时，route 返回 `503 VMP4 cache not ready`，同时后台启动构建。
  - 构建日志写入 `debug_output/vmp4_cache_build`。
  - 临时 MP4 写入源片同目录的 `.vmp4build_*` 工作目录。
  - 构建成功后通过 `os.replace(temp_target, target)` 原子移动到最终 cache 文件名。
  - 构建完成后删除临时工作目录，避免播放器读到半成品。

- oversize 自动重建已完成。
  - 已把默认构建码率从 `source` 改为 `budget`。
  - budget 基于 `source_size * 8 / source_duration` 计算，并留安全余量。
  - 发现已有 cache 大于源文件时，会强制按更低预算码率重建。
  - 构建未完成期间返回 `503 VMP4 cache rebuilding`，不再永久卡在 `409`。

- 已验证的真机行为。
  - `72463_3840p` 首次进入触发 oversize rebuild。
  - 重建完成后，第二次进入能正常播放。
  - request history 显示 nPlayer 启动请求返回 `200`，后续多次拖动请求返回 `206`。
  - 这说明目标播放器确实按 MP4 byte Range 请求中段/尾段数据，当前 cache-file 路线可作为后续 slot backend 的真机验证入口。

- 测试覆盖已同步。
  - config 默认值。
  - GUI settings/env。
  - DLNA seek item 暴露。
  - cache missing build。
  - cache-file serving。
  - oversize rebuild。
  - `mode=two_dvr` route selection。

### 11.2 部分完成

- 阶段 0 的“直接接入 `/passthrough_seek`”已完成，但原计划里的“用短测试片构造 `ftyp+moov+mdat` 虚拟布局”尚未完成。
  - 当前使用真实完整 MP4 cache 的 `moov`，不是服务端自建 slot `moov`。

- 阶段 3 的“源大小等长约束”在完整 cache-file 路径中已部分完成。
  - 总 HTTP/DLNA 声明大小等于源文件大小。
  - cache 小于源文件时可用顶层 `free` box 补齐。
  - 但 slot 级预算、slot oversize 重试、slot 内合法 padding 尚未完成。

- 阶段 5 的“DLNA 灰度暴露”已提前完成一部分。
  - 目前 seek item 已默认可见，便于真机测试。
  - 这对开发测试有价值，但进入稳定版前应重新评估默认暴露策略。

### 11.3 尚未完成

- 真正的 GOP slot backend 尚未开始。
  - 尚未生成虚拟 `ftyp+moov+mdat` layout。
  - 尚未生成 `co64/stsz/stsc/stts/stss` 对应固定 slot 的 sample/chunk 表。
  - 尚未实现 Range 到 slot 的稳定映射。
  - 尚未实现 slot 按需生成、读取、复用和失败恢复。

- video-only slot 尚未完成。
  - 尚未定义 slot 时长、slot 大小、GOP 对齐规则。
  - 尚未实现从任意 slot 时间点生成可解码 video payload。
  - 尚未验证目标 VR 播放器是否接受粗粒度 GOP sample/chunk。

- 音频 slot/interleave 尚未完成。
  - 当前完整 cache-file 路径由 offline mux 生成音视频完整 MP4。
  - 实时 slot 路径还没有音频 sample 表、音视频 interleave 和 seek 后 A/V sync 方案。

- 实时未命中策略尚未完成。
  - 当播放器请求的 slot 尚未生成时，尚未决定阻塞等待、返回 `503`、优先生成还是降级 fallback。
  - 这需要真机验证播放器对等待和重试的容忍度。

- 生产级缓存管理尚未完成。
  - 尚未有 slot manifest。
  - 尚未有磁盘 quota / 清理策略。
  - 尚未有服务重启后的 slot 恢复。
  - 尚未有 GUI 进度展示和取消/重试控制。

### 11.4 需要修正的后续路线

原计划的方向不变，但阶段顺序需要根据这次 cache-file 真机结果调整:

1. 保留当前 cache-file backend 作为稳定 fallback。
   - 建议新增明确 backend 配置:

```text
PT_PASSTHROUGH_SEEK_VMP4_BACKEND=cache_file|slot
```

   - 默认可以继续用 `cache_file`，避免 slot 实验影响已验证的真机播放路径。

2. 下一步不要直接做完整音视频实时 slot。
   - 先做 video-only slot layout，目标是验证播放器是否接受服务端自建的 `moov` 和粗粒度 GOP slot sample/chunk。
   - 如果 video-only 的虚拟 `moov` 都不能被目标 VR 播放器稳定拖动，后续音频和实时调度没有继续投入的价值。

3. 重新定义阶段 0。
   - 原阶段 0 写的是“直接接入 `/passthrough_seek`，但不暴露 DLNA”。
   - 现在接入和 DLNA 暴露已经完成，因此新的阶段 0 应改为:
     - 在 `/passthrough_seek` 内增加 `slot` backend 分支；
     - 构造最小 video-only 虚拟 MP4；
     - 支持 `HEAD`、`GET bytes=0-`、init range、一个固定 slot range；
     - 与 `cache_file` backend 并存。

4. 重新定义阶段 1。
   - 阶段 1 应聚焦“video-only 多 slot + 按需生成”:
     - 固定 slot duration；
     - 固定 slot byte budget；
     - Range 映射到 slot；
     - slot 文件缓存；
     - 同一 Range 重复请求字节稳定；
     - oversize slot 降码率重试。

5. 阶段 2 再做音频。
   - 音频不应在最小 slot 验证前加入。
   - 通过 video-only 后，再设计音频 sample/chunk 表和 A/V interleave。

6. 阶段 3 再做真实“实时体验”。
   - 包括点击后预热前几个 slot、顺序预取、拖动目标 slot 优先生成。
   - 需要根据真机行为决定未命中 slot 是等待还是返回可重试错误。

7. 阶段 4 才做默认 DLNA 暴露收敛。
   - 当前默认暴露是为了测试效率。
   - slot backend 稳定前，正式默认策略应考虑是否只暴露 cache-ready item，或同时暴露 live fallback。

### 11.5 下一步建议

短期最有价值的下一步不是继续优化完整 cache-file，而是做一个最小 slot backend 骨架:

- 新增 `PT_PASSTHROUGH_SEEK_VMP4_BACKEND`。
- 保留当前 `cache_file` 行为不变。
- 增加 `slot` 实验分支。
- 先生成 video-only、固定 2 秒或 4 秒 slot 的虚拟 MP4 layout。
- 用一个很短测试片验证:
  - `ffprobe` 能识别；
  - `GET bytes=0-` 能返回稳定 init；
  - 请求固定中段 Range 能映射到对应 slot；
  - 重复同一 Range 字节一致。

完成这个骨架后，再让 VR 真机测试 `slot` backend 是否至少能显示时长、启动播放和发出中段 Range。

## 12. 2026-06-19 开发切片 3: VMP4 backend 开关与固定 slot sample 骨架

状态: 已完成本地开发与单元测试，尚未进入真机验证。

本切片完成:

- 新增明确 backend 配置:

```text
PT_PASSTHROUGH_SEEK_VMP4_BACKEND=cache_file|slot
```

- 默认值为 `cache_file`，继续保留已验证的完整 MP4 cache-file 真机路径。
- GUI 启动服务会显式传入:

```text
PT_PASSTHROUGH_SEEK_VMP4_BACKEND=cache_file
```

- `/passthrough_seek` 的 VMP4 分支现在按 backend 分派:
  - `cache_file`: 继续使用完整 faststart passthrough MP4 cache；
  - `slot`: 使用新的 video-only slot layout 实验 backend。
- 新增 `pipeline/passthrough_vmp4_slot.py`:
  - 构造 `ftyp + moov + mdat` 虚拟 MP4；
  - 只包含 video track；
  - `co64` 按固定 GOP slot 起点生成；
  - `stsc` 使用 one sample per chunk；
  - `stsz` 现在固定为 `slot_size`，而不是源关键帧原始大小；
  - 每个 slot 当前仍用源视频关键帧作为占位 payload；
  - 源关键帧 payload 后面使用长度前缀的 H.264/H.265 filler NAL 补齐到 `slot_size`；
  - 未落入已定义 slot sample 的尾部 `mdat` 剩余字节仍按 0 填充。
- slot Range 读取已支持:
  - init 区域读取；
  - slot 内源样本读取；
  - slot 内 filler NAL 区域读取；
  - 重复同一 Range 返回稳定字节。
- 新增 slot 诊断头:

```text
X-Passthrough-VMP4-Backend: slot
X-Passthrough-VMP4-Slot-Count
X-Passthrough-VMP4-Slot-Size
X-Passthrough-VMP4-Slot-Duration
X-Passthrough-VMP4-Moov-Size
X-Passthrough-VMP4-Mdat-Payload-Start
X-Passthrough-VMP4-Mdat-Payload-Size
X-Passthrough-VMP4-Slot
X-Passthrough-VMP4-Slot-Time
X-Passthrough-VMP4-Slot-Source-Sample
X-Passthrough-VMP4-Slot-Sample-Size
X-Passthrough-VMP4-Slot-Source-Bytes
X-Passthrough-VMP4-Slot-Filler-Bytes
X-Passthrough-VMP4-Slot-Filler
```

重要说明:

- 这仍不是最终实时 passthrough GOP backend。
- 当前 slot payload 是“源视频关键帧 + filler NAL”的本地可验证骨架。
- 本切片的意义是先把虚拟 MP4 的 `stsz/co64/stsc/stts/stss` 固定 slot 语义立住，让后续真实 passthrough GOP payload 只需要满足 `payload_size <= slot_size`。

涉及文件:

- `config.py`
- `ui/settings.py`
- `http_app/routes_media.py`
- `pipeline/passthrough_vmp4_slot.py`
- `tests/test_passthrough_vmp4_slot.py`
- `tests/test_config_defaults.py`
- `tests/test_settings.py`
- `prompt/HANDOVER_20260619.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
2 passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

## 19. 2026-06-20 开发切片 10: 评审反馈修复与 ready-only slot 语义

状态: 已完成本地开发、单元测试和组合回归，尚未进入新一轮真机验证。

触发原因:

- 代码评审指出 slot backend 存在几个会影响真机稳定性的点:
  - manifest 原子写使用固定 `.manifest.json.tmp`，Windows 并发请求会竞争；
  - 0 字节 `slot_XXXXXX.bin` 会被错误标记为 `ready`；
  - `iter_vmp4_slot_range` 内部复用 `payload_size`，真实 payload/placeholder/filler 语义容易错位；
  - `moov` offset 收敛循环没有失败兜底；
  - placeholder -> real payload 会破坏同一 Range 的 byte-stability。
- 本地 `debug_output/server.log` 尾部显示当时请求主要是旧 `passthrough_seek backend=pynv_hevc` 流式路径和 ffmpeg fMP4 mux，不是当前 `VMP4 slot response` 分支；下一轮真机需要确认新进程日志里出现 `X-Passthrough-VMP4-Backend: slot` / `VMP4 slot response`。

本切片完成:

- 新增默认开启配置:

```text
PT_PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY=1
```

- GUI 启动环境同步传入:

```text
PT_PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY=1
```

- slot ready-only 行为:
  - GET 命中未 ready 的 media slot 时，先排构建，再返回 `503 VMP4 slot not ready` 和 `Retry-After: 1`；
  - HEAD 不返回 body，因此允许返回稳定声明头；
  - `bytes=0-` 这类 open range 如果前面已有稳定字节，会截到下一个未 ready slot 之前，并返回实际 `Content-Range`；
  - 响应增加诊断头:

```text
X-Passthrough-VMP4-Slot-Ready-Only
X-Passthrough-VMP4-Slot-Ready-Bounded
```

- manifest/payload 写入修复:
  - manifest 写入改为唯一临时文件 + `os.replace`；
  - 同一 manifest 目标增加 per-path 写锁，避免 Windows 同目标并发 `os.replace` 短暂 `PermissionError`；
  - Windows `os.replace` 增加短重试；
  - slot payload 写入也改为唯一临时文件原子替换，避免写入中断留下半成品；
  - placeholder build worker 不再使用固定 payload `.tmp` 文件。

- manifest 状态修复:
  - `payload_size == 0` 不再标记 `ready`，改为:

```text
state = failed
reason = payload-empty
```

  - ready payload 必须满足:

```text
0 < payload_size <= sample_size
```

- layout/manifest 性能修复:
  - route 侧新增 per-source/template layout cache，上限 64 项；
  - cache key 包含源路径/stat、duration、fps、GOP、slot sample 上限、输出模板 stsd/payload hash、输出尺寸和 codec；
  - `ensure_vmp4_slot_manifest(...)` 只有 manifest 内容实际变化时才写盘，避免每个 Range 都刷新 `updated_at`。

- slot range 读取修复:
  - `iter_vmp4_slot_range(...)` 改为显式使用 `file_payload_size` / `emitted_payload_size`；
  - filler 起点只依赖本次实际 emitted payload 长度，避免 ready/placeholder 分支复用变量导致偏移错位。

- layout 收敛修复:
  - `build_passthrough_vmp4_slot_layout(...)` 的 `moov` 迭代 4 次未收敛时明确抛出:

```text
layout-moov-offsets-not-converged
```

涉及文件:

- `config.py`
- `ui/settings.py`
- `http_app/routes_media.py`
- `pipeline/passthrough_vmp4_slot.py`
- `tests/test_passthrough_vmp4_slot.py`
- `tests/test_config_defaults.py`
- `tests/test_settings.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260620.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
12 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
138 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile config.py ui\settings.py http_app\routes_media.py pipeline\passthrough_vmp4_slot.py tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py tests\test_settings.py
passed
```

仍未完成 / 下一步:

- 当前 placeholder build 仍然只写 HEVC placeholder payload，不是真实 passthrough GOP。
- 下一步应实现真实 video-only GPU GOP slot builder:
  - slot start time -> PyNv decode/matting/HEVC Annex-B GOP；
  - 用 `write_vmp4_slot_hevc_annexb_payload(...)` 转成 MP4 length-prefixed sample；
  - 成功后 manifest 标记 ready；
  - oversize 时标记 failed 并记录原因。
- 阶段 1 真机前仍需用真实多帧 GOP payload 替换单帧 placeholder，否则播放器验证结论不能完全迁移到最终路线。

## 18. 2026-06-19 开发切片 9: HEVC Annex-B 到 MP4 sample payload 转换接口

状态: 已完成本地开发与测试。

目标:

- PyNv 实时编码路径输出的是 raw HEVC Annex-B bitstream。
- slot backend 的 `slot_XXXXXX.bin` 最终要作为 MP4 `mdat` sample payload 被 `stsz/co64` 引用。
- 因此真实 GPU GOP builder 写 payload 前，必须把 Annex-B start-code NAL 转成 MP4 length-prefixed NAL。

本切片完成:

- 在 `pipeline/passthrough_vmp4_slot.py` 新增:

```text
iter_annexb_nal_units(...)
hevc_annexb_to_length_prefixed_sample(...)
write_vmp4_slot_hevc_annexb_payload(...)
```

- 支持 3 字节和 4 字节 Annex-B start code。
- 写入时使用 layout 的 `nal_length_size`，当前 HEVC 模板为 4 字节。
- 自动去掉 NAL 后面的 trailing zero。
- 空 bitstream 返回明确错误:

```text
annexb-nal-missing
```

- payload 大于当前 slot sample size 时返回明确错误:

```text
slot-payload-oversize
```

- 这为下一步真实 GPU slot worker 提供稳定写入接口:
  - worker 从 PyNv encoder 拿 raw HEVC bytes；
  - 调用 `write_vmp4_slot_hevc_annexb_payload(...)`；
  - 成功后 manifest 标记 slot `ready`。

涉及文件:

- `pipeline/passthrough_vmp4_slot.py`
- `tests/test_passthrough_vmp4_slot.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260619.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py -q
15 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
133 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

## 16. 2026-06-19 开发切片 7: slot 统一 HEVC 输出与小样本布局

状态: 已完成本地开发、单元测试和本地 HEVC placeholder MP4 smoke，尚未进入新一轮真机验证。

触发原因:

- 真机 GPU/GUI 默认 slot 后，`server.log` 显示失败文件的 slot manifest 均为:

```text
codec_name = av01
slot_size = 2.4MB ~ 4.2MB
```

- 播放器只发了 `Range: bytes=0-`，没有继续发中段 `206` Range，说明它在解析/解码初始化阶段已放弃。
- 用户明确要求: 实时输出应统一为 HEVC，cache 模式只是测试，不要 fallback 到 cache-file。

本切片完成:

- 保持 GPU/GUI 默认 backend 仍为:

```text
PT_PASSTHROUGH_SEEK_VMP4_BACKEND=slot
```

- 不做自动 fallback 到 `cache_file`。
- slot backend 不再继承源视频 `stsd` 作为输出 codec 描述。
- 新增 HEVC 输出模板路径:
  - route 首次需要时优先用 PyAV 在进程内生成一帧黑色 HEVC MP4 模板；
  - PyAV 优先尝试 `hevc_nvenc`，再尝试 `libx265` / `hevc`；
  - 只有 PyAV 不可用或所有 PyAV 编码器失败时，才 fallback 到 ffmpeg 子进程；
  - 从模板中提取 HEVC `stsd/hvcC` 和一个合法 HEVC access unit；
  - 生成结果缓存在 `runtime_cache/vmp4_slot_template/`。
- `pipeline/passthrough_vmp4_slot.py` 现在区分:
  - `source_codec_name`: 源文件 codec，例如 `av01`；
  - `codec_name`: 虚拟 MP4 输出 codec，当前固定为 `hevc`。
- slot layout 的 `moov/stsd` 现在使用 HEVC 输出模板，不再把 AV1 源轨道暴露给播放器。
- placeholder 不再读取源 keyframe bytes。
  - 没有 ready payload 时，服务端输出 HEVC placeholder access unit；
  - placeholder build queue 写入的也是 HEVC placeholder 到 `slot_XXXXXX.bin`；
  - 后续真实 GPU GOP generator 会继续复用同一个 payload 文件接口。
- 新增可配置 sample 上限:

```text
PT_PASSTHROUGH_SEEK_VMP4_SLOT_MAX_SAMPLE_BYTES=1048576
```

- slot 文件语义从“`slot_size == slot stride`”改为:
  - `slot_stride`: slot 在源大小虚拟文件中的间隔，继续支撑 source-size byte/time 映射；
  - `slot_size`: `stsz` 里播放器实际看到的 sample size，默认最大 1MB；
  - `slot_gap_size`: slot sample 后面的未引用间隙，不进入 `stsz`，播放器不应把它当视频 sample 解码。
- 新增诊断头:

```text
X-Passthrough-VMP4-Source-Codec
X-Passthrough-VMP4-Output-Codec
X-Passthrough-VMP4-Output-Size
X-Passthrough-VMP4-Slot-Stride
X-Passthrough-VMP4-Slot-Gap
X-Passthrough-VMP4-Slot-Placeholder-Bytes
```

涉及文件:

- `config.py`
- `http_app/routes_media.py`
- `pipeline/passthrough_vmp4_slot.py`
- `tests/test_passthrough_vmp4_slot.py`
- `tests/test_config_defaults.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260619.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
5 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_config_defaults.py -q
8 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
131 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

本地 smoke:

```text
output = debug_output/vmp4_slot_hevc_placeholder_smoke.mp4
ffprobe:
  codec_name=hevc
  width=320
  height=180
  duration=6.000000
  nb_frames=3

ffmpeg -v error -i debug_output\vmp4_slot_hevc_placeholder_smoke.mp4 -frames:v 3 -f null -
passed
```

重要限制:

- 这一步修正了 “AV1 源 codec 被暴露给 slot 输出” 和 “multi-MB sample 触发播放器解码拒绝” 两个结构性问题。
- 但当前 ready payload 仍是 HEVC placeholder，不是真实 passthrough 画面 GOP。
- 下一步应把 placeholder builder 替换为真实 video-only GPU passthrough GOP 生成:
  - 以 slot start time 作为输入；
  - 输出 HEVC access unit / GOP payload；
  - 写入 `slot_XXXXXX.bin`；
  - payload 必须小于等于 `slot_size`；
  - 与当前 HEVC `stsd/hvcC` 保持一致，或在 layout rebuild 时整体替换模板。

## 17. 2026-06-19 开发切片 8: HEVC 模板生成改为 PyAV 优先

状态: 已完成本地开发与验证。

用户要求:

- 项目已有 PyAV，能不用 ffmpeg 的情况优先用 PyAV，避免子进程管理。

本切片完成:

- `http_app/routes_media.py` 中的 slot HEVC template 生成改为 PyAV 优先。
- 新增进程内 PyAV 模板生成流程:
  - 创建黑色 `numpy` frame；
  - PyAV mux 到临时 MP4；
  - 设置 `codec_tag = hvc1`；
  - 成功后 `os.replace` 原子替换到模板缓存路径。
- PyAV 编码器尝试顺序:

```text
hevc_nvenc -> libx265 -> hevc
```

- 只有 PyAV import 失败、编码器不可用或所有 PyAV 尝试失败时，才保留 ffmpeg fallback。
- `load_vmp4_slot_output_template(...)` 仍统一从生成后的 MP4 中抽取 HEVC `stsd/hvcC` 和 placeholder payload，因此 HTTP/slot 后续逻辑不需要区分模板来自 PyAV 还是 ffmpeg。

涉及文件:

- `http_app/routes_media.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260619.md`

本地验证:

```text
PyAV encoder probe:
  av 17.1.0
  hevc_nvenc ok
  libx265 ok
  hevc ok

PyAV template smoke:
  output = debug_output/pyav_slot_template_smoke.mp4
  codec_name = hevc
  stsd bytes = 256
  payload bytes = 64
  nal_length_size = 4

PyAV template full virtual MP4 smoke:
  output = debug_output/vmp4_slot_pyav_hevc_final_smoke.mp4
  codec_name = hevc
  codec_tag_string = hvc1
  width = 320
  height = 180
  duration = 6.000000
  nb_frames = 3
  ffmpeg -v error decode first 3 frames passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py -q
13 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
131 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

下一步:

- 增加 slot manifest/cache 结构。
- 让 slot backend 明确记录每个 slot 的状态: `placeholder|building|ready|failed`。
- 为后续真实 passthrough GOP 生成保留稳定的 slot payload 文件接口。

## 13. 2026-06-19 开发切片 4: slot manifest/cache 与 ready payload 接口

状态: 已完成本地开发与单元测试，尚未进入真机验证。

本切片完成:

- 在 `pipeline/passthrough_vmp4_slot.py` 中新增 slot cache manifest 结构。
- 每个 slot backend 请求会在:

```text
runtime_cache/vmp4_slot/<digest>/manifest.json
```

  写入或刷新 manifest。
- manifest 读取使用 `utf-8-sig`，写入使用 UTF-8，避免后续路径或诊断中出现中文编码问题。
- manifest 内容包括:
  - schema；
  - digest；
  - source path/stat；
  - output mode；
  - FPS/GOP；
  - source-size layout；
  - slot count/size/duration；
  - `mdat` payload 起点和大小；
  - codec / NAL length size；
  - 每个 slot 的 state、payload 文件名、payload size、sample size、source sample index、source size、start time。
- 新增标准 slot payload 文件接口:

```text
slot_000000.bin
slot_000001.bin
...
```

- 如果 `slot_XXXXXX.bin` 存在且大小不超过 `slot_size`，manifest 会把该 slot 标记为:

```text
state = ready
```

- slot Range 读取现在优先使用 ready payload 文件。
- ready payload 小于 `slot_size` 时，剩余部分继续用合法 filler NAL 补齐。
- 没有 ready payload 的 slot 保持:

```text
state = placeholder
```

  并继续使用“源关键帧 + filler NAL”的占位输出。
- 已保留 `building` / `failed` 状态的 manifest 兼容逻辑，后续后台构建队列可以直接写入这些状态。
- `/passthrough_seek` slot backend 已增加 manifest/cache 诊断头:

```text
X-Passthrough-VMP4-Slot-Cache
X-Passthrough-VMP4-Slot-Manifest
X-Passthrough-VMP4-Slot-States
X-Passthrough-VMP4-Slot-State
X-Passthrough-VMP4-Slot-Payload-Bytes
```

重要说明:

- 这一步仍不启动 GPU/offline 构建。
- 它完成的是“slot payload 文件格式和服务端读取接口”。
- 后续真实 GOP 生成器只需要把某个 slot 的 passthrough GOP payload 写入对应 `slot_XXXXXX.bin`，并确保大小不超过 `slot_size`。

涉及文件:

- `pipeline/passthrough_vmp4_slot.py`
- `http_app/routes_media.py`
- `tests/test_passthrough_vmp4_slot.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
3 passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

下一步:

- 增加 slot 后台构建队列骨架。
- 请求命中 placeholder slot 时，把对应 slot 标为 `building` 并启动构建任务。
- 初版构建任务可以先生成可验证的 payload 文件，再替换成真实 passthrough GOP 生成流程。

## 14. 2026-06-19 开发切片 5: slot placeholder 构建队列骨架

状态: 已完成本地开发与单元测试，尚未进入真机验证。

本切片完成:

- 新增开发期 slot 构建开关:

```text
PT_PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER=0|1
```

- 默认值为 `1`。
- 该开关只在显式选择 `PT_PASSTHROUGH_SEEK_VMP4_BACKEND=slot` 后才有实际意义；默认 `cache_file` backend 不会触发。
- 新增 slot placeholder build queue:
  - 每个 `<digest>:slot=<index>` 同一时间只会排一次；
  - 请求命中 placeholder slot 时，后台线程会把当前“源关键帧占位 payload”写成对应的 `slot_XXXXXX.bin`；
  - 写入使用临时文件 + `os.replace` 原子替换；
  - 构建开始时 manifest slot state 更新为 `building`；
  - 构建成功后 manifest slot state 更新为 `ready`；
  - 构建失败后 manifest slot state 更新为 `failed` 并记录 reason。
- 新增 `pipeline/passthrough_vmp4_slot.py` helper:
  - `write_vmp4_slot_source_payload(...)`
  - `update_vmp4_slot_manifest_slot_state(...)`
- `/passthrough_seek` slot backend 现在会返回:

```text
X-Passthrough-VMP4-Slot-State
X-Passthrough-VMP4-Slot-Build
X-Passthrough-VMP4-Slot-Build-Reason
```

- 当前请求仍立即返回 placeholder 字节，不会因为后台构建阻塞播放器。
- 后续请求会通过 manifest 扫描到 `ready` payload，并优先从 `slot_XXXXXX.bin` 读取。

重要说明:

- 当前 builder 仍只是 placeholder builder，不是真实 passthrough GOP generator。
- 这个骨架验证的是:
  - slot 级状态流转；
  - manifest 原子更新；
  - payload 文件原子落盘；
  - route 对 ready payload 的优先读取。
- 后续要替换的是 builder 内部 payload 生成逻辑，而不是 HTTP Range / manifest / slot 文件接口。

涉及文件:

- `config.py`
- `http_app/routes_media.py`
- `pipeline/passthrough_vmp4_slot.py`
- `tests/test_passthrough_vmp4_slot.py`
- `tests/test_config_defaults.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
129 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed

本地真实文件 smoke:
source = videos/dance_38280300662-1-192.mp4
output = debug_output/vmp4_slot_smoke_dance.mp4
ffprobe 可识别:
  codec_name=h264
  width=720
  height=1280
  nb_frames=8
  duration=18.320563
  size=3124014
ffmpeg -v error 解码前 8 帧到 null 通过
```

下一步:

- 把 placeholder builder 替换为真实 video-only passthrough GOP payload 生成。
- 初版可以调用现有 offline passthrough 的 `--start` / `--duration` / `--audio off` 能力生成短 MP4，再抽取 video sample payload。
- 需要保证抽取出的 payload:
  - 从 IDR/SPS/PPS/VPS 开始；
  - 大小不超过 `slot_size`；
  - 与虚拟 `stsd`/codec configuration 自洽。

## 15. 2026-06-19 开发切片 6: GPU/GUI 启动默认切到 slot backend

状态: 已完成本地开发与单元测试。

用户要求:

- “我用 GPU 启动，你默认切换到 slot。”

本切片完成:

- `PT_PASSTHROUGH_SEEK_VMP4_BACKEND` 默认值从:

```text
cache_file
```

  改为:

```text
slot
```

- `config.py` 中无显式环境变量时，VMP4 seek route 默认进入 slot backend。
- GUI settings 默认值同步改为:

```json
"passthrough_seek_vmp4_backend": "slot"
```

- 新增一次性 GUI settings migration:

```text
20260619_seek_vmp4_slot_default
```

- 该 migration 会把已有 `runtime_cache/ui_settings.json` 中的:

```json
"passthrough_seek_vmp4_backend": "cache_file"
```

  自动切到:

```json
"passthrough_seek_vmp4_backend": "slot"
```

- 全新 settings 会标记该 migration 已完成，避免后续保存后被反复覆盖。
- 如果之后需要回到完整 cache-file fallback，可以显式设置:

```powershell
$env:PT_PASSTHROUGH_SEEK_VMP4_BACKEND="cache_file"
```

  或在 GUI settings 中保存 `cache_file`；migration 不会反复覆盖已迁移配置。

测试调整:

- `tests/test_config_defaults.py` 改为断言默认 backend 是 `slot`。
- `tests/test_settings.py` 改为断言 GUI server env 传入 `PT_PASSTHROUGH_SEEK_VMP4_BACKEND=slot`。
- 新增测试覆盖已有 GUI settings 从 `cache_file` migration 到 `slot`。
- `tests/test_routes_media_cache.py` 是 cache-file 专用测试模块，已在模块级 setup/teardown 中显式固定 backend 为 `cache_file`，避免与运行默认 `slot` 混淆。

涉及文件:

- `config.py`
- `ui/settings.py`
- `tests/test_config_defaults.py`
- `tests/test_settings.py`
- `tests/test_routes_media_cache.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260619.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_config_defaults.py tests\test_settings.py tests\test_routes_media_cache.py tests\test_passthrough_vmp4_slot.py tests\test_content_directory_modes.py -q
130 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

## 20. 2026-06-20 当前接续状态索引

- 评审反馈修复已经完成，详见本文第 19 节“2026-06-20 开发切片 10: 评审反馈修复与 ready-only slot 语义”。
- 当前可以继续做真实 video-only GPU GOP slot builder；在此之前不建议直接用 placeholder slot 结果作为最终真机结论。

## 21. 2026-06-20 开发切片 11: Quest3 日志复核与 no-Range ready-only 修复

状态: 已完成本地开发与验证，等待下一轮真机确认。

用户反馈:

- Quest3 上 Skybox 显示无法播放。
- HereSphere 画面为空。
- 4XVR 播放器卡死。
- 要求查看 `debug_output/server.log`。

日志结论:

- 当前日志已经能看到 slot backend 生效:
  - `VMP4 slot request`
  - `VMP4 slot response`
  - `source_codec=av01 output_codec=hevc`
- 但旧逻辑对无 `Range` 的 GET 返回了局部 `206`:
  - `range=None`
  - `content_range='bytes 0-.../...'`
- 无 `Range` 请求返回 `206 Partial Content` 是错误 HTTP 语义，Skybox 这类播放器会把它当成异常文件/探测失败。
- HereSphere 有正常 Range 重试行为；画面为空仍符合旧 placeholder builder 只写黑帧/模板 payload 的现状。
- 4XVR 日志同时出现 `/passthrough_live` 和 `/passthrough_seek`，其中 seek 路径仍拿到 placeholder/旧 no-Range 行为，因此卡死不能作为最终 slot 结论。

本切片修复:

- ready-only 模式下，无 `Range` GET 不再返回局部 `206`。
- 如果请求覆盖到未 ready 的 slot:
  - 调度对应 slot 构建；
  - 返回 `503 VMP4 slot not ready`；
  - 带 `Retry-After: 1`。
- 对 `bytes=0-` 这类开放 Range:
  - 如果前缀 slot ready、后续 slot 未 ready，则响应只截断到下一个未 ready slot 之前；
  - 返回诊断头:
    - `X-Passthrough-VMP4-Next-Slot`
    - `X-Passthrough-VMP4-Next-Slot-State`
    - `X-Passthrough-VMP4-Next-Slot-Build`

涉及文件:

- `http_app/routes_media.py`
- `tests/test_passthrough_vmp4_slot.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260620.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
139 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

## 22. 2026-06-20 开发切片 12: slot builder 改为真实 PyNv HEVC GOP payload

状态: 已完成本地开发、mock 单元测试和组合回归；真实 GPU/Quest 播放效果仍需真机确认。

用户要求:

- 实时输出应统一 HEVC，不应输出 AV1。
- cache 模式只是测试，不要 fallback。
- 项目有 PyAV；能不用 ffmpeg 的情况优先用 PyAV，避免进程管理。
- 在必须真机测试前继续开发。

本切片完成:

- 新增进程内 PyNv GOP 构建函数:

```text
pipeline.pynv_stream.build_passthrough_hevc_annexb_gop(...)
```

- 该函数复用实时 PyNv 路径:
  - `PyNvSimpleDecoder` / `PyNvThreadedSerialDecoder`
  - Matter GPU green composite
  - AlphaPacker alpha 输出
  - PyNvVideoCodec NVENC HEVC encode
  - 首帧强制 IDR + SPS/PPS/VPS 输出
- slot 后台 worker 不再写 HEVC placeholder payload。
- slot worker 现在流程为:
  - slot start time；
  - 获取 Matter 实例；
  - 生成真实 HEVC Annex-B access unit；
  - 调用 `write_vmp4_slot_hevc_annexb_payload(...)` 转成 MP4 length-prefixed sample；
  - 原子写 `slot_XXXXXX.bin`；
  - manifest 标记 `ready`。
- 本切片曾验证“一个 MP4 sample 内塞多帧 GOP”会触发 ffmpeg 警告:

```text
Two slices reporting being the first in the same frame.
```

  因此当前 slot builder 明确每个 slot 只写 1 个 IDR access unit，匹配“一 slot 一个 MP4 sample”的现有 layout。流畅多帧输出留给后续 frame-level sample layout。
- 如果真实构建失败:
  - 不 fallback 到 placeholder；
  - 删除不完整 payload；
  - manifest 标记 `failed`；
  - reason 写入异常信息。
- `two_dvr` slot builder 目前明确 unsupported/failed，不混入这次 green/alpha 修复。
- 为避免真实 GOP 超过 `PT_PASSTHROUGH_SEEK_VMP4_SLOT_MAX_SAMPLE_BYTES`，slot builder 会按 sample 容量反推安全 NVENC bitrate，上限不超过实时估算码率。
- `PT_PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER` 保留为 legacy env 名称，但注释已改为“启用 slot payload build”；设置为 0 仅用于本地 layout placeholder 测试。

涉及文件:

- `config.py`
- `http_app/routes_media.py`
- `pipeline/pynv_stream.py`
- `tests/test_passthrough_vmp4_slot.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260620.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
14 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_routes_media_cache.py tests\test_config_defaults.py tests\test_settings.py tests\test_content_directory_modes.py -q
140 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile ...
passed
```

本地 GPU smoke:

- 先前直接用 `.\.venv\Scripts\python.exe` 的 inline 脚本跑 CUDA smoke 会超时；复查 `PROJECT.md` 后确认这是错误启动方式。
- 正确方式必须在任何 CUDA/CuPy/ORT import 前执行:
  - `utils.runtime_dll_paths.apply_runtime_dll_paths()`
  - `utils.gpu_runtime_cache.configure_gpu_runtime_cache()`
  - 使用 `uv run python ...`
- CUDA 自检通过:
  - `sys.executable = ...\.venv\Scripts\python.exe`
  - `nvrtc=(12, 9)`
  - `CUPY_COMPILE_WITH_PTX='0'`
  - `compiler._use_ptx=False`
  - `arch=('-arch=sm_120', 'cubin')`
  - `CUPY_CACHE_DIR=runtime_cache\cupy`
- 最小 CuPy/NVENC encode smoke 通过:
  - `CreateEncoder(...)` 成功；
  - `EndEncode()` 输出 183 bytes。
- 窄 builder smoke 通过:
  - `Matter(load_model=False)` + `PT_PASSTHROUGH_RVM_BYPASS_ALPHA=1`
  - `build_passthrough_hevc_annexb_gop(...)`
  - 2 帧输出 227146 bytes；
  - Annex-B 开头为 HEVC VPS NAL。
- 完整 Matter/CUDA builder smoke 通过:
  - `Matter()` 加载 RVM；
  - active providers 为 `['CUDAExecutionProvider', 'CPUExecutionProvider']`；
  - `rvm_iobinding=True`；
  - 1 帧输出 90888 bytes；
  - Annex-B 开头为 HEVC VPS NAL。
- 真实 PyNv slot MP4 smoke 通过:
  - 生成文件: `debug_output/vmp4_slot_real_pynv_smoke.mp4`
  - payload 文件: `debug_output/vmp4_slot_real_pynv_slot0.bin`
  - payload size: 90888 bytes
  - sample size: 390366 bytes
  - `ffprobe` 识别:
    - `codec_name=hevc`
    - `codec_tag_string=hvc1`
    - `width=720`
    - `height=1280`
    - `nb_frames=8`
  - `ffmpeg -v error -i ... -frames:v 2 -f null -` 通过，无错误输出。

当前剩余风险:

- slot MP4 模板 `hvcC` 仍来自模板 MP4；真实 slot payload 来自 PyNv/NVENC。payload 首帧带 VPS/SPS/PPS，但若 Quest 播放器严格要求 sample entry `hvcC` 与 payload 参数集完全一致，下一步需要把模板也改成 PyNv 参数集生成。
- 当前虚拟 MP4 仍是“一 slot 一个 MP4 sample”，所以每个 slot 只有一个真实 IDR 画面，播放流畅度不是最终形态。如果 Quest 能播但不流畅，下一步需要把 layout 从 slot-sample 改成 frame-sample。
- 下一轮真机测试前，建议先启动服务器让日志确认出现:
  - `slot PyNv GOP ready`
  - `passthrough_seek VMP4 slot build ready`

## 23. 2026-06-20 开发切片 13: 真机前 slot builder 护栏 (R1/R2/R3)

状态: 已完成本地开发、单元测试和组合回归；等待真机验证。

触发原因: 评审指出真实 PyNv slot builder 有三个会在真机出问题的运营风险。

- R1: slot sample 固定上限 1MB，4K/8K 单 IDR 必然 `payload-oversize` → 该 slot 永久 `failed` → ready-only 下永远 503 → 不可播。
- R2: `failed` slot 每次请求都重建；oversize 是确定性失败，叠加 `Retry-After:1` → 每秒重跑一次 GPU 编码且必失败，GPU 持续空转。
- R3: build worker `acquire_matter(blocking=True, timeout=None)` 会无限等；live 流占用 Matter 时 slot 永远卡 `building`。

本切片完成:

- R1 分辨率自适应 slot sample 上限:
  - 新增 `PT_PASSTHROUGH_SEEK_VMP4_SLOT_IDR_BPP`，默认 `2.0`。
  - 有效上限 = `max(PT_..._MAX_SAMPLE_BYTES, ceil(out_w*out_h*IDR_BPP/8))`。
  - 4K 输出上限 ~2MB、8K ~8MB，单 IDR 可装下；小分辨率仍用 1MB 下限。
  - layout build 与 layout cache key 统一使用有效上限。
  - builder 码率预算随 `max_bytes` 同步放大。
- R2 失败退避 + 永久失败不重试:
  - `_Vmp4SlotBuild` 增加 `attempts`/`permanent`/`next_retry_at`。
  - reason 含 `oversize`/`unsupported` 视为 permanent，永不自动重试。
  - 其余瞬时失败按 `min(RETRY_MAX, RETRY_BASE*2**attempts)` 退避重试。
  - 新增 `PT_PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_BASE`(默认5s)/`..._RETRY_MAX`(默认60s)。
- R3 Matter 获取超时:
  - worker 改为 `acquire_matter(blocking=True, timeout=PT_PASSTHROUGH_SEEK_VMP4_SLOT_MATTER_TIMEOUT)`，默认 20s。
  - 超时记为瞬时失败 `slot-real-builder-matter-timeout`，按退避重试，不再无限 hang。

涉及文件:

- `config.py`
- `http_app/routes_media.py`
- `tests/test_passthrough_vmp4_slot.py`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
17 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py tests\test_settings.py tests\test_routes_media_cache.py tests\test_content_directory_modes.py -q
143 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile config.py http_app\routes_media.py pipeline\passthrough_vmp4_slot.py pipeline\pynv_stream.py tests\test_passthrough_vmp4_slot.py
passed
```

真机观察点:

- 日志里 oversize 应大幅减少；若仍出现，调大 `PT_PASSTHROUGH_SEEK_VMP4_SLOT_IDR_BPP`。
- `failed` slot 不应再每秒重建；permanent 失败只构建一次。
- live + seek 并发时 slot build 不应永久卡 `building`；超时会转为可重试 `failed`。
- 仍未解决: 一 slot 一 IDR 的流畅度问题（需 frame-level sample layout），以及模板 `hvcC` 与 PyNv 参数集一致性。

## 24. 2026-06-20 开发切片 14: Skybox/4XVR slot-not-ready 启动兼容修复

状态: 已完成本地开发、单元测试和组合回归；等待 Quest 真机复测。

触发原因: 真机日志显示 Skybox/4XVR 并不是 PyNv 没有输出 HEVC，而是 HTTP 层把“slot 正在构建”暴露给播放器:

- Skybox:
  - 首次 `Range: bytes=0-` 在 slot0 未 ready 时返回 init-only `206`（几 KB）。
  - 随后播放器请求 `bytes=<mdat_start>-`，服务端返回 `503 slot-not-ready`。
  - slot0 实际随后不到 1 秒构建完成，但播放器已判定无法播放。
- 4XVR:
  - 对 seekable VMP4 发无 `Range` 请求。
  - ready-only 路径直接返回 `503 slot-not-ready`，播放器随后回退到 live/raw 探测并失败。

本切片完成:

- 新增 `PT_PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT`，默认 `8.0s`。
  - ready-only 请求命中未 ready slot 时，服务端先调度 PyNv slot build 并短等待 payload 落盘。
  - 等到真实 payload 后再响应；不输出 placeholder，不回退 cache，不改 AV1/HEVC 输出策略。
  - 超时或永久失败才继续暴露 `slot-not-ready`。
- 修复 Skybox open-range 行为:
  - `bytes=0-` 不再在 slot0 未 ready 时返回 init-only `206`。
  - 如果当前 range 前面已有 ready slot、下一个 slot 未 ready 且等待失败，仍可截断到下一个未 ready slot 前，避免读取占位样本。
  - 当 range 起点正好落在未 ready slot 时，先等待该 slot ready，再返回真实 HEVC payload。
- 修复 4XVR 无 Range 行为:
  - 无 `Range` 的 ready-only VMP4 请求不再直接 503。
  - 新增按 slot 等待的流式 iterator：读取到每个 slot 前调度/等待真实 payload，然后再输出该 slot。
  - 若后续 slot 等待超时，会中断流并写 warning 日志，不会输出 placeholder。
- 增加诊断响应头:
  - `X-Passthrough-VMP4-Slot-Wait`
  - `X-Passthrough-VMP4-Slot-Wait-Ready`
  - `X-Passthrough-VMP4-Slot-Wait-Seconds`

涉及文件:

- `config.py`
- `http_app/routes_media.py`
- `tests/test_passthrough_vmp4_slot.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260620.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py -q
18 passed

.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py tests\test_settings.py tests\test_routes_media_cache.py tests\test_content_directory_modes.py -q
144 passed, 8 subtests passed

PYTHONPYCACHEPREFIX=debug_output\pycache_compile python -m py_compile config.py http_app\routes_media.py pipeline\passthrough_vmp4_slot.py pipeline\pynv_stream.py tests\test_passthrough_vmp4_slot.py
passed
```

下一次真机重点观察:

- Skybox 日志中不应再出现首包 init-only 几 KB `206` 后紧接 `503 slot-not-ready`。
- Skybox 每次请求命中未 ready slot 时，响应头应出现 `X-Passthrough-VMP4-Slot-Wait-Ready: 1`，并返回真实 payload。
- 4XVR 无 Range seekable 请求不应再立刻 503；若还失败，再看是否是 VMP4 结构层问题（例如一 slot 一 IDR / `hvcC` 参数集一致性）。

## 25. 2026-06-20 开发切片 15: 事件循环阻塞修复 + slot_size 收紧 + 离线解码自检 (A/B/C)

状态: 已完成本地开发、单元测试、组合回归和离线 ffmpeg 解码自检；等待真机复测。

触发原因: Quest3 (Skybox/4XVR) 多轮真机仍失败，且上一切片(14)的 ready-wait 改动引入了并发隐患。评审定位三件事。

### A. ranged ready-wait 阻塞 asyncio 事件循环 (真 bug)

- 切片 14 的 ready-wait 对 ranged 请求是 inline 跑在 `async def passthrough_seek_get` 里的 `time.sleep` 轮询，最长 `READY_WAIT=8s`，会冻住整个事件循环（其它播放器、其它 StreamingResponse 分块、DLNA browse 全停），反而制造 503/超时级联。
- 修复: GET 的 slot 响应改为 `await run_in_threadpool(_seek_vmp4_slot_response, ...)`，等待挪出事件循环。no-Range 的等待本来就在 streaming generator(threadpool)里，是对的。

### B. slot_size 收紧（撤销切片 13 的 R1 反向放大）

- 日志证明真实 8K alpha IDR 只有 50–275KB，根本不会 oversize；R1 把 slot_size 从 1MB 抬到 stride 上限(~2.5MB)，使每个 access unit 里塞了 ~2.2MB filler NAL，方向反了。
- 关键语义: `slot_size` = 解码器按 stsz 读到的 access unit 大小；`slot_stride - slot_size = slot_gap` 是不被引用的间隙(解码器看不到)。**slot_size 越小，AU 越小，filler 越少。**
- 修复:
  - 删除 `PT_PASSTHROUGH_SEEK_VMP4_SLOT_IDR_BPP` 和 `_vmp4_slot_effective_max_sample_bytes`。
  - `PASSTHROUGH_SEEK_VMP4_SLOT_MAX_SAMPLE_BYTES` 默认 `1MB → 512KB`（仍可 env 覆盖）。
  - layout build / layout cache key 直接用该 cap。
  - 现在 8K 下 slot_size=512KB、gap~2.5MB（进 gap 不进 AU），AU 内 filler 从 ~2.2MB 降到 ≤512KB。
  - 超过 cap 的 payload 仍按 permanent-fail 处理（切片 13 的 R2 退避保持）。

### C. 离线解码自检 harness `tools/vmp4_slot_decode_check.py`

- 用真实源构建真实 slot layout，把前 N 个 slot 用真实 HEVC IDR access unit（ffmpeg 生成的模板，或 `--payload-file` 注入真实 `slot_XXXXXX.bin`）填充，通过生产同款 `iter_vmp4_slot_range` 吐出字节，再跑 `ffprobe` + `ffmpeg` 解码。
- 用法: `uv run python -m tools.vmp4_slot_decode_check --source videos/72456_3840p.mp4 --slots 5`

### C 的关键结论（重要，影响后续方向）

- 在 `videos/72456_3840p.mp4`、7680x3840、512KB cap 下:
  - `ffprobe`: `codec_name=hevc width=7680 height=3840 nb_frames=190 duration=189.03`
  - `ffmpeg -frames:v 5 -f null -`: **DECODE OK (rc=0)**
- 用旧的 2.5MB cap（2.6MB filler/AU）跑同样自检: **也 DECODE OK**。
- 含义:
  1. **虚拟 MP4 容器结构本身是合法的**，filler-in-access-unit 不是“坏文件”。
  2. ffmpeg 软解很宽容，**无法区分硬解的 per-AU 限制**；所以 C 是“必要非充分”的门：能挡住结构性坏文件，但不能证明 Quest 硬解一定接受。
  3. 因此 Quest 失败的根因**大概率不在容器结构**，而在以下之一: (a) 移动端硬解对巨型 filler-AU 的输入上限（B 已把 AU 从 2.5MB 降到 512KB，显著降低该风险）；(b) 一 slot 一 IDR 的 ~1fps 时间轴 cadence；(c) 启动期 HTTP 行为（A/14 已改善）。

涉及文件:

- `config.py`
- `http_app/routes_media.py`
- `tools/vmp4_slot_decode_check.py`（新增）
- `tests/test_passthrough_vmp4_slot.py`
- `tests/test_config_defaults.py`
- `summary/summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`
- `prompt/HANDOVER_20260620.md`

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py tests\test_settings.py tests\test_routes_media_cache.py tests\test_content_directory_modes.py -q
144 passed, 8 subtests passed

tools.vmp4_slot_decode_check --source videos/72456_3840p.mp4 --slots 5  -> DECODE OK
```

下一轮真机重点:

- 确认 AU 收紧后 Skybox/4XVR 是否有变化（响应头 `X-Passthrough-VMP4-Slot-Size` 应为 512KB 量级）。
- 若仍失败：因为容器已证明合法，应转向 cadence/结构 (frame-level sample layout，让每秒多帧而不是 1 IDR/slot) 或对照 `/passthrough_live`（已知可播）抓 Quest 实际拒绝点；不要再纠结 HTTP 时序。

## 26. 2026-06-20 真机日志诊断 + 帧级布局可行性验证

状态: 日志诊断完成；帧级 plain-MP4 方案已离线原型验证通过；后端尚未实现。

### 真机结果与日志诊断

- SKYBOX: 出画面但 ~1fps PPT。4XVR: 卡死。MoonVR: 绿屏、无进度条。
- 日志(`debug_output/server.log`)证据:
  - 整条链路跑通、无服务端报错；slot 构建成功(236–286KB IDR，全部 < 512KB cap)。
  - no-Range 已返回 200；ranged 返回 206 且 range 正确。
  - **每个 slot = 恰好 1 帧、stts ~1.2s → 整片 0.83fps**，这是 PPT 的根因(结构，非解码)。
  - 每个 slot GPU 构建 ~0.5–1.7s，按需逐 slot 编码追不上实时。
  - `1.mp4 mode=two_dvr`: `slot-real-builder-unsupported-mode:two_dvr permanent=True` → 该 item 永久 503(疑似 4XVR/MoonVR 卡死/无进度条来源之一)。

### 两个结构性天花板

1. MP4 结构 vs 可变帧率: 一 sample=一帧，要真 fps 必须每帧一个 sample；但帧大小可变、IDR≫P，与"单 moov、co64 提前固定、byte-stable"冲突。
2. 按需 GPU 延迟: 每单元 ~1s，逐请求编码追不上实时。

### 方向决策(用户)

- 坚持实时(cache_file 违背初衷，排除)。
- 不走 fMP4(此前真机证明播放器处理不好)。
- 在当前 plain ftyp+moov+mdat 内想办法。

### 验证通过的方法: 帧级固定预算布局

- 每个 GOP slot 内 K 帧 = K 个 MP4 sample，每帧 stts=1/fps → 真 fps。
- 每帧给固定字节预算(per-frame budget)，帧内真实 NAL + filler 补齐 → co64/stsz 可提前固定，仍单 moov、byte-stable、plain MP4。
- filler 粒度从"每 AU 2.4MB"降为"每帧几十 KB"，AU 更小、对硬解更友好。
- 放弃 `virtual_size == source_size` 硬约束: 8K 实时码率塞不进源大小(计划 §4.4)；total 改用实时码率估算(live 已用 `live_total_est`)，seek-bar 用估算映射。
- 实时性: 不再逐请求编码，改为后台连续顺序编码(同 live 的 PyNv loop)写入固定 layout、跑在播放光标前；seek 时重定位解码器再续。

### 离线原型(GPU-free)验证

新增 `tools/vmp4_frame_filler_proof.py`: 用 libx265(bframes=0，匹配 HEVC_BF=0)生成真实 60 帧 GOP，按"每帧固定预算 + filler"组装 plain ftyp+moov+mdat，结果:

```text
ffprobe: codec=hevc 1920x960 nb_frames=60 avg_frame_rate=30/1 duration=2.000000
ffmpeg decode: OK (无错误)
RESULT: decode=OK real_fps=OK frames=60 -> 帧级固定预算 plain MP4 可行
```

注: 用 libx265 默认(含 B 帧)时 ffmpeg null muxer 报 DTS 非单调警告，因为缺 ctts；真实 PyNv 编码 `HEVC_BF=0` 无 B 帧，bframes=0 后告警消失、解码干净。结论: **帧级方案在当前 MP4 格式下成立。**

### 下一步(待实现的后端)

1. 帧级 layout builder: `slot_count*K` 帧样本，per-frame budget(IDR/P 两档或统一档)，固定 co64/stsz、stts=1/fps、stss=GOP 头；total 用实时码率估算。
2. 后台顺序实时编码器: 复用 PyNv live loop，逐帧写入固定 offset、标记 ready、跑在光标前；seek 重定位。
3. 收尾: two_dvr 要么实现要么停止暴露其 seek item(当前永久 503)。

## 27. 2026-06-20 开发切片 16: 帧级实时 backend (slot_frames)

状态: 已完成本地开发 + 单元测试 + 离线 ffmpeg 解码验证；GPU 实时编码路径待真机验证。

目标: 在不改变 plain ftyp+moov+mdat / 单 moov / byte-stable 的前提下，把 1fps 幻灯片改成真 fps（用户要求：保持实时、不用 cache_file、不用 fMP4）。

新增 backend 值:

```text
PT_PASSTHROUGH_SEEK_VMP4_BACKEND=slot_frames   # 新默认
```

核心设计（已离线验证）:

- 每帧一个 MP4 sample，stts=1/fps → 真 fps；stss 只标 GOP 头（避免把 P 帧当 sync sample → 绿屏）。
- 每帧固定字节预算（IDR 档 / P 档），真实 NAL + filler 补齐 → co64/stsz 提前固定，仍单 moov、byte-stable。
- 放弃 `virtual_size == source_size`：total 用 per-frame budget 估算（类似 live 的 live_total_est）。
- patch mdhd timescale + duration，使 1/fps tick 映射为真实 fps（否则沿用源 timescale 会算错帧率）。

实时后台编码:

- 复用已验证的 `build_passthrough_hevc_annexb_gop`（每 GOP 一次 PyNv 编码）。
- 新增 `split_hevc_annexb_access_units(...)`：按 VCL NAL 把整 GOP Annex-B 切成逐帧 access unit（HEVC_BF=0 下一 VCL=一帧）。
- 每帧转 length-prefixed sample，超预算则该 GOP 失败（permanent，不 thrash）。
- 流式响应用按 GOP 等待的 generator：播放器拉流时即时构建当前 GOP 并预取后续（`PT_PASSTHROUGH_SEEK_VMP4_FRAMES_PREFETCH`，默认 3），跑在读光标前；运行在 StreamingResponse 线程池，不阻塞事件循环。
- 起始 GOP 未就绪超时返回 `503 + Retry-After`。

新增配置:

```text
PT_PASSTHROUGH_SEEK_VMP4_FRAMES_IDR_BYTES=786432   # IDR 帧预算 768KB
PT_PASSTHROUGH_SEEK_VMP4_FRAMES_P_BYTES=163840     # P 帧预算 160KB
PT_PASSTHROUGH_SEEK_VMP4_FRAMES_PREFETCH=3
```

GUI 默认与一次性 migration:

- DEFAULTS `passthrough_seek_vmp4_backend` 改为 `slot_frames`。
- 新增 migration `20260620_seek_vmp4_frames_default`：把已有的 `slot` 迁到 `slot_frames`，不动显式 `cache_file`。

two_dvr 收尾: frames backend 仅支持 green/alpha，two_dvr 立即返回 `409 frames-mode-unsupported`（不再永久 503 卡死）。DLNA 是否继续暴露 two_dvr seek item 待后续清理。

涉及文件:

- `config.py`、`ui/settings.py`
- `pipeline/passthrough_vmp4_frames.py`（新增）
- `pipeline/passthrough_vmp4_slot.py`（新增 `split_hevc_annexb_access_units`）
- `http_app/routes_media.py`（frames backend 路由/构建/等待/预取/流式）
- `tests/test_passthrough_vmp4_frames.py`（新增）、`tests/test_config_defaults.py`、`tests/test_settings.py`
- `tools/vmp4_frame_filler_proof.py`（离线原型）

本地验证:

```text
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_passthrough_vmp4_frames.py tests\test_passthrough_vmp4_slot.py tests\test_config_defaults.py tests\test_settings.py tests\test_routes_media_cache.py tests\test_content_directory_modes.py -q
150 passed, 8 subtests passed

# 离线: 真实 libx265(bframes=0) GOP -> frames layout -> ffprobe 30/1 fps, 60 frames, dur 2.0, ffmpeg decode OK
```

真机前需重点验证/调参:

- P 帧预算 160KB 可能在 8K 高动态下偏小 → `frames-oversize` 使该 GOP 失败/503。按真机 `frames-oversize` 日志调大 `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_P_BYTES`。
- 每 GOP ~1s GPU 编码 vs 1.2s 内容：靠 prefetch 维持领先；若 rebuffer 频繁，调大 prefetch。
- DLNA 声明 size（源大小）与 HTTP total（budget 估算）不一致；多数播放器用 HTTP Content-Range，但需真机确认。
- 响应头 `X-Passthrough-VMP4-Backend: slot_frames`、`X-Passthrough-VMP4-Frames-Fps`、`X-Passthrough-VMP4-Frames-Wait-Ready` 可用于诊断。

### 27.1 two_dvr DLNA 暴露收尾

- `dlna/content_directory.py` 新增 `_seek_vmp4_two_dvr_supported()`（仅 `cache_file` backend 为 True）与 `_seek_vmp4_mode_seekable(mode)`。
- two_dvr 的 `.seek.mp4` DLNA item 现在只在 `cache_file` backend 下暴露（有离线 two_dvr cache 时）；`slot`/`slot_frames` 下不再暴露，避免播放器选到永久失败入口。
- 计数函数 `_video_item_count` 与构建函数 `_video_items_from_index` 同步加同一守门，count/items 保持一致。
- live two_dvr item 不受影响（live 路径支持 two_dvr）。
- 测试: `tests/test_content_directory_modes.py` 更新（two_dvr seek 用 cache_file 断言；新增 slot_frames 下不暴露的用例）。`150 passed, 8 subtests`。

## 28. 2026-06-20 nPlayer 真机: frames-oversize 根因与 CBR/VBV 修复

真机现象: nPlayer 大部分 503，只有两个小 dance 文件能播。

日志根因（`debug_output/server.log`）: `frames-oversize` —— 个别 P 帧远超 160KB P 预算（如 `idx=54:478029`、`idx=61:447532`、`idx=16:209127`），导致整个 GOP `permanent` 失败 → 该段永久 503。dance 文件能播是因为低分辨率/低运动，P 帧都很小。IDR 预算没问题，问题是高运动/场景切换的 P 帧尖峰。

关键认识: **服务端每个 sample 永远发送 `budget` 字节（payload+filler），所以 HTTP 带宽 == FRAME_BYTES * fps，与编码器码率控制无关。** 因此：
- 加大预算 = 浪费带宽（filler 占满）；
- 减小预算 = P 尖峰溢出失败。
- 正解：**统一每帧预算 + CBR + 1 帧 VBV 上限**，让 NVENC 把每帧都填到预算附近且不超过，固定带宽用于真实数据而非 filler。之前默认 VBR 才产生 P 尖峰。

本切片完成:

- 配置从两档（IDR/P）改为统一 `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES`（默认 256KiB ≈ 50fps 下 100Mbps，对齐 live 发送率）+ `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_RATE_HEADROOM`（默认 0.85，CBR 目标留余量避免超过硬预算）。
- `build_passthrough_hevc_annexb_gop(..., per_frame_cap_bytes=)`：>0 时 `rc=cbr`、bitrate=budget*headroom*8*fps、`vbvbufsize/vbvinit=budget` 作 1 帧硬上限。
- VBV kwargs 兼容性兜底：若该 PyNvVideoCodec 版本不接受 `maxbitrate/vbvbufsize/vbvinit`，回退仍保留 `rc=cbr`（主修复），靠 per-frame oversize 守门兜底。
- frames 后端 layout/cache key/worker 全部改用统一 FRAME_BYTES；诊断头 `X-Passthrough-VMP4-Frames-Budget`。

预期: CBR 把每帧均衡到 ~218KB（256KB*0.85）、VBV 硬顶 256KB，P 尖峰被压下，`frames-oversize` 基本消失；带宽固定 ~100Mbps（与 live 同量级）。

本地验证: `132 passed, 8 subtests`。真机需确认 `frames-oversize` 是否消失；若仍偶发，调小 `RATE_HEADROOM` 或确认 VBV kwargs 是否被接受（日志看是否仍有尖峰）。带宽过高可调小 FRAME_BYTES。

## 29. 2026-06-20 修复 frames 全 503: 等待时长/预取/余量 + GPU 吞吐天花板

现象: CBR 改动后所有 seek 视频首请求 503。

日志诊断（已修好 oversize 尖峰，P 帧从 478KB 降到 ~75–150KB）但暴露三个问题:

1. **`READY_WAIT=8s` 远小于 GOP 构建耗时**: 60 帧 4K/8K-alpha GOP 编码要 ~5–15s，首请求 8s 内必 503。日志里 dance 实际第 3 次重试才 200（gop0 ~16s 才 ready）。
2. **prefetch=3 抢占 GPU**，拖慢关键的首个 GOP。
3. **CBR 仍偶发轻微 overshoot**（263324>262144，仅超 1KB）→ GOP permanent 失败，因为 headroom 0.85 余量太小（无硬 VBV 时 CBR 只均衡不硬顶）。

本切片修复:

- 新增 `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_READY_WAIT`（默认 30s），frames 路径不再复用 slot 的 8s。
- `PASSTHROUGH_SEEK_VMP4_FRAMES_RATE_HEADROOM` 默认 0.85 → 0.6（CBR 目标压到 ~154KB，overshoot 落在 256KB 预算内）。
- `PASSTHROUGH_SEEK_VMP4_FRAMES_PREFETCH` 默认 3 → 1，减少对首 GOP 的 GPU 抢占。

本地: `72 passed, 8 subtests`。

**仍存在的根本天花板（需用户决策）**: 8K-alpha 的 PyNv 编码约 ~10fps，而播放需 50fps，**按需编码追不上实时**（~0.2x）。30s 等待能让首个 GOP 就绪、首屏出画，但后续会持续 rebuffer。要真正流畅，只能降低 seek 后端的输出分辨率/帧率（如 seek 预览限到 4K）让编码接近实时，或接受 8K 卡顿。这是 GPU 吞吐限制，非参数能消除。

## 30. 2026-06-20 根因修正: 逐 GOP 重建管线 → 持久流式会话

用户关键观察: live MPEG-TS 在 TRT 下能 60fps+,为什么 slot_frames 这么慢/全 503?

日志确认根因 **不是编码速度,是生命周期**: `build_passthrough_hevc_annexb_gop` 每个 1.2s GOP 都重建整条管线——`PyNvSimpleDecoder(源)` 打开 8K 文件、seek、`nvc.CreateEncoder` 新建 NVENC session、`matter.reset_state()` RVM 重置。日志里每个 GOP 独立 `slot PyNv GOP ready`,~4–11s 才出一个 60 帧 GOP = 有效 ~6–15fps。而 live **只建一次**然后连续跑,所以 60fps+。等于把"建管线"开销对一个 187s 的片付了 156 次。顺带每 GOP reset RVM 还导致每个边界 alpha 抖动。

修复(复用 live 的持久会话思路):

- `pipeline/pynv_stream.py` 新增 `iter_pynv_passthrough_annexb_frames(...)`: 把原 setup+编码循环重构成**只 setup 一次、逐帧 yield HEVC Annex-B access unit** 的生成器。`build_passthrough_hevc_annexb_gop` 变成它的薄封装(slot 旧后端不变)。
- `http_app/routes_media.py`: frames 后端从"逐 GOP 构建"改成**单个持久 filler**(`_Vmp4FramesFiller`):一次 setup,顺序流式编码,边写 `frame_XXXXXX.bin` 边推进 cursor,跑在播放光标前。
  - seek: 请求的 GOP 远超 cursor(超过 lead ≈ 20s 帧数)或在会话起点之前 → 停旧 filler,在该 GOP 处重启一个前向 filler;已写文件保留复用。
  - oversize 帧改为 clamp 到预算(单帧轻微 glitch),不再让整个 GOP 永久 503。
- 移除每 GOP 的构建队列/重试/prefetch(持久 filler 天然一直在前向预取);`..._FRAMES_PREFETCH` 标记为 legacy/no-op。

预期: 吞吐回到 live 的 60fps+,8K 也能接近/达到实时,首屏快、不再大面积 503;alpha 边界抖动消失(整段只 reset 一次 RVM)。

本地: `150 passed, 8 subtests`(含新的持久 filler 单测,mock 生成器验证一次会话流式写出所有帧并可服务)。

真机验证点: 日志应出现 `PyNv frame stream begin`(每次会话/seek 一条,而不是每 GOP 一条),GOP 就绪节奏接近实时;`X-Passthrough-VMP4-Frames-Wait-Ready: 1`;若仍跟不上,再考虑降分辨率/帧率(上一节的开关仍在)。

## 31. 2026-06-20 修复并发 filler 抖动 + Content-Length 崩溃

持久会话上线后仍全 503。日志根因:

1. **filler 抖动**: nPlayer 同时开多个不同 offset 的 range 连接(日志里 start=7.200 与 start=32.400 交替),每个请求的 `ensure_filler` 都把单个 filler 杀掉重启到自己的位置 → 每 ~3s 重启一次,**从不产出任何帧**。
2. **`RuntimeError: Response content shorter than Content-Length`**(ASGI 崩溃): 等待式流在中途 GOP 等待失败时提前返回,但 Content-Length 已按完整长度声明 → 实际字节不足 → 崩。

修复:

- **filler 重启去抖**: 新增 `_VMP4_FRAMES_RESTART_COOLDOWN=8s`。运行中的 filler 在冷却期内不被重启;未覆盖到的并发请求只等待/503 并重试,让 filler 真正向前产出。`started_at` 记录会话起点。
- **206 只服务已构建的连续前缀**: 新增 `_vmp4_frames_served_end(...)`,把 206 的 end 收敛到"从起点开始连续 ready 的最后一帧",Content-Length 精确、全部来自磁盘文件,**不再中途等待**,从根本上消除长度不足崩溃。起点帧未就绪 → 503。播放器再请求下一段,filler 继续向前。
- **无 Range / zero-open 改为 chunked**(不带 Content-Length,和 live 一样): 用等待式流式,落后时阻塞(rebuffer)而不报错;不会再因长度声明崩溃。
- HEAD 仍声明完整大小、无 body、不触发构建。

本地: `151 passed, 8 subtests`(含 206 连续前缀边界单测)。

真机验证点:
- 日志里 `PyNv frame stream begin` 不应再每 ~3s 刷屏(去抖后一个会话至少跑 8s+ 才可能重定位)。
- 不应再出现 `Response content shorter than Content-Length`。
- 206 响应的 `Content-Range` 长度应等于已构建连续帧;播放器顺序拉流时,filler 在前面持续产出。
- 若多连接探测仍导致个别 503,属并发随机访问与按需生成的固有张力;持续播放(单点顺序)应能推进。
