# SI 混音传输层 — 开发计划（交付开发，2026-06-17 改写版）

> 配套研究存档：`summary_20260617_DLNA_TS_BYTE_SEEK_RESEARCH_CN.md`（必读，尤其 §6/§7/§8/§9）。
> 前置失败复盘：`summary_20260528_SEEKABLE_PASSTHROUGH_LESSONS_CN.md`（必读）。
>
> **本文取代旧版"实时 TS 字节 seek"方案**（那条线已被 2026-05-28 复盘 + HEVC-in-PS 实测判为高风险，详见研究存档 §6/§7）。
>
> 一句话目标：让 DLNA 里的 `[SI]` 条目在 Skybox/nPlayer/HereSphere 上**真正能播**，并尽量给出**普通文件式可拖动进度条**。
>
> **状态（2026-06-17）：Phase 0 真机已完成、门槛已过。** SKYBOX/nPlayer/4XVR 三端均能播放并 seek 静态 (f)MP4；其中 **progressive 明显最优**（SKYBOX 单条 `bytes=0-` 顺序读、4XVR/nPlayer 实发量最低），fMP4 触发 4.7x~6.3x 重复读取与 Range 风暴。**Phase 1 路线已锁定 = progressive virtual remux MP4**（fMP4 仅留作个别播放器 fallback）。基础件、字节稳定不变量测试、PyAV 打包探针均已就绪 → **可以开工 Phase 1**（先做下方"里程碑 M1"验证残留风险，再全量接入）。

---

## 0. 背景与已确认的硬事实（动手前必须接受）

1. 现有已提交的 SI 实现（`http_app/si_stream.py`，每请求现编 fragmented-MP4 + 估算巨型 Content-Length + IP 共享 session）在 SKYBOX 8K 上**整体失败**，**传输层作废重写**（研究存档 §3）。
2. 源内容是 **HEVC 8K**。**MPEG-PS(`video/mpeg`) 装不下 HEVC**（实测 `codec_name=unknown`+underflow），Windows 的 `.mpg` 把戏复刻不了（研究存档 §7）。
3. **裸 `video/MP2T`(TS) 会被 Skybox/nPlayer 当 live、不给普通进度条**（2026-05-28 实测，研究存档 §6.2）。
4. 对 HEVC，"能进度条 + 字节 seek 的文件式容器"**只剩 MP4**（progressive 或 fMP4），且要求**真实索引 + 字节稳定**（同一 URL 字节 N 多次请求恒为同一字节）。
5. **SI 是 `-c:v copy`**（只混音频、不重编码视频）→ 视频样本表可从源 moov 白拿、音频可预生成一次 → **整份布局可提前算死、字节稳定** → 这是"虚拟 remux MP4"唯一干净可行的场景（研究存档 §9.2）。
6. 不变更范围：**不**把虚拟 MP4 推广到转码模式（绿幕/alpha/2D→3D，它们重编码、fragment 大小不可预知）；**不**用 PyAV 替换 GPU 转码链。

### 复用已建好的部分（只换传输层）
以下已实现、保留不动：`utils/si_filter.py`（`SIMixParams`+`build_si_mix_filter`）、`utils/runtime_settings.py` 热重载（`SIMixRuntime`/`get_si_mix`/`set_si_mix`）、`http_app/routes_control.py` 的 `/control/si_mix`、首页"其他配置"toggle+音轨设置对话框、`ui/settings.py` 的 `si_*` key、DLNA `[SI]` 条目的 **VR 命名**（`source_display_stem` → `[SI]name_LR_180_SBS`）。
**本次只改"如何把这条 `[SI]` 流传给播放器"。**

---

## Phase 0（决定性门槛，先做，约 1 天）— 静态 (f)MP4 真机能力测试

> 目的：把"播放器认不认一个 (f)MP4 **文件**为可 seek"从所有虚拟文件复杂度里隔离出来。不过这关，后面全是白做（2026-05-28 的教训）。

### 步骤
1. 用现成 ffmpeg 生成**三个短(约 30s)静态文件**（视频 `-c:v copy`、音频混音用 `build_si_mix_filter`）：
   - `A_progressive.mp4`：`-movflags +faststart`（moov 在前、完整 stbl）。
   - `B_fmp4.mp4`：`-movflags +empty_moov+default_base_moof+frag_keyframe`。
   - `C_fmp4_dash.mp4`：`-movflags +dash`（含 `sidx`；可再 `MP4Box -frag`/`-add` 补 `mfra`）。
   - 命令模板见 §附录。
2. 把三个文件放进 DLNA 目录，用现有 **`/media` 静态 Range 路由**原样服务（真实 size、`OP=01`、`video/mp4`）。
3. 在 **Skybox/nPlayer/HereSphere/DeoVR/4XVR** 逐个播放，记录：是否显示总时长进度条、拖动 25%/50%/90% 是否 seek 成功、是否能播到结尾。
4. 用 `debug_output/request_history/*.jsonl` 抓真实 Range 模式（尾部 moov 探测？中部 Range？纯顺序？）。

### 判定门（决定 Phase 1 走哪条）
- **progressive 能 seek、fMP4 不能** → Phase 1 走 **progressive 虚拟 remux**。
- **fMP4 也能 seek，且请求模式稳定** → Phase 1 可走 **fMP4 虚拟 remux**（更易让 ffmpeg 生成 + 拦截）。
- **fMP4 能 seek 但触发高频重叠 Range / 缓冲卡顿** → 不按"能 seek"简单放行，优先走 **progressive 虚拟 remux**。
- **两者在 Skybox 都不给进度条** → **放弃 scrub**，SI 改走 **Option A（TS-live，章节 seek）**，结束。

> Phase 0 不写任何虚拟文件代码，纯 ffmpeg + 现有 `/media` + 真机抓包。**这是最高优先级。**

### SKYBOX 实测结果（2026-06-17）

抓包：`debug_output/si_proto/skybox.pcapng`；服务日志：`debug_output/server_skybox.log`。

- `A_progressive.mp4`：1 个 `bytes=0-` 顺序请求，服务端 1.75s 发送完整约 92.8MB；用户体感缓存条快速读满、播放流畅。
- `B_fmp4.mp4`：856 个 open-ended Range；先扫 `moof/mfra`，随后大量重叠 `mdat` 请求；服务端约 20s 发送 343MB，约为文件大小 3.7 倍；可拖动但体感卡顿。
- `C_fmp4_dash.mp4`：807 个 open-ended Range；先扫尾部/`sidx`，随后大量重叠 `mdat` 请求；服务端约 19s 发送 333MB，约为文件大小 3.6 倍；可拖动但体感卡顿。

**SKYBOX 结论**：fMP4 不是"不认"，而是"认但请求模式差"。Phase 1 对 SKYBOX 的首选路线改为 **progressive virtual remux**；fMP4 降为其他播放器实测更优时的备选。注意 A 样本较短且已快速整片缓存，后续 progressive `/media_si` 接入后仍需用真实长片验证未缓存状态下的拖动 Range。

### nPlayer 实测结果（2026-06-17）

抓包：`debug_output/si_proto/nPlayer.pcapng`；服务日志：`debug_output/server_nplayer.log`。

- `A_progressive.mp4`：nPlayer 使用无 Range `200` 全文件请求 + 少量 open-ended Range（尾部/拖动位置）；多次播放/拖动混在同一日志中，pcap 约 185MB 实发，其中单次完整 full-file 约 92.8MB/1.8s。
- `B_fmp4.mp4`：2 个无 Range `200` + 36 个 open-ended Range，服务端约 94MB 实发；多数 Range 起点落在 `mdat` 内部，不贴 `moof/mfra` 边界。
- `C_fmp4_dash.mp4`：2 个无 Range `200` + 34 个 open-ended Range，服务端约 99MB 实发；多数 Range 起点也落在 `mdat` 内部，不贴 `sidx/moof` 边界。

**nPlayer 结论**：三种文件都能 seek；fMP4 没有 SKYBOX 那种 800+ Range 风暴，但 `sidx`/fMP4 边界没有带来明显收益。nPlayer 不推翻 SKYBOX 的路线判断：Phase 1 仍默认先做 **progressive virtual remux**；fMP4 保留为播放器专项备选。

### 4XVR 实测结果（2026-06-17）

抓包：`debug_output/si_proto/4xvr.pcapng`；服务日志：`debug_output/server_4xvr.log`。注意：日志开头 4XVR 曾点到 `/passthrough_live/...A_progressive...` alpha 条目，该请求不纳入 Phase 0 静态 `/media` 判定。

- `A_progressive.mp4`：静态 `/media` 中 2 个无 Range `200` + 1 个闭区间 Range `bytes=45088768-92795440`；pcap 按请求时间归属统计约 116MB 实发，约为文件大小 1.25 倍。
- `B_fmp4.mp4`：1 个无 Range `200` + 45 个闭区间 Range，均为 `bytes=N-file_end`；server log 声明总 length 约 2.68GB，pcap 实发约 589MB，约为文件大小 6.35 倍。
- `C_fmp4_dash.mp4`：1 个无 Range `200` + 32 个闭区间 Range，均为 `bytes=N-file_end`；server log 声明总 length 约 2.08GB，pcap 实发约 439MB，约为文件大小 4.73 倍。
- B/C 的所有 Range 起点都落在 `mdat` 内部，没有命中 `moof`/`mfra`/`sidx` 边界；`C_fmp4_dash` 的 sidx 没看到收益。

**4XVR 结论**：三种文件都能 seek，但 fMP4 触发的闭区间到尾部重复读取明显重于 progressive。4XVR 不推翻 progressive 默认路线，反而进一步支持 Phase 1 先做 **progressive virtual remux MP4**。

---

## Phase 1（路线已锁定）— SI **progressive** virtual remux MP4

Phase 0 三端实测后**锁定 progressive**（fMP4 仅作个别播放器 fallback，本期不实现）。核心：**对外暴露一个稳定的逻辑 progressive MP4（moov 在前 + mdat 在后），视频字节直接引用源文件、音频字节来自预生成 sidecar，全片布局提前算死。** 复用已建好的 `pipeline/si_virtual_mp4.py`（VirtualRegion/Range 切片/box 捕获/样本表读取，与容器无关，progressive 同样适用）。

### ⚠️ 两个必须先正视的点（progressive 特有）
1. **progressive 是"播放器更友好但更难构造"的容器。** fMP4 的 moov 极小（无逐样本 stbl，3838B 已实测）；progressive 必须**完整构造 moov/stbl**（`stsz`/`stco→co64`/`stsc`/`stts`/`ctts`/`stss` + 视频 `hvcC` + 音频 `esds`）。这是 Phase 1 的主要工程量与风险——专家当初推 fMP4 正是为绕开它。好在 SI 是 `-c:v copy`：视频样本的 size/duration/ctts/keyframe **可直接读源 moov 平移**，不是从零发明，只需把"源视频 stbl + 预生成音频 stbl"按选定的 mdat 交错布局合并并重算偏移。**注意 faststart 的 moov-前置不能靠 `ffmpeg -movflags +faststart`（那要落整片 8GB），必须解析式构造。**
2. **SKYBOX 的中段字节 seek 在长片上尚未验证。** Phase 0 的 `A_progressive` 只有 30s/92MB，SKYBOX 用**单条 `bytes=0-` 顺序读**就整片缓存完了——我们**没观察到 SKYBOX 对 progressive 发出中段 Range**（4XVR 确实发了 `bytes=45088768-…` 中段闭区间，nPlayer 也有；但 SKYBOX 这条没出现）。50min/7.5GB 真实长片在未缓存状态下 SKYBOX 拖动是否发出正确中段 Range、并被我们的虚拟文件正确响应，是**最大残留未知数**。

### 里程碑 M1（开工第一步，把残留风险前移）
先做**最小可用的 progressive `/media_si`**：仅对单个长片、隐藏在现有 seek 门控开关后（`PASSTHROUGH_SEEK_ENABLED` 同款灰度，不进 DLNA 默认目录），实现解析式 moov + 虚拟区域服务。**用真实 50min 8K 片在 SKYBOX/4XVR/nPlayer 上验证：①未缓存下拖动中段产生正确中段 Range；②返回字节与逻辑文件一致；③播到结尾正常收尾。** M1 过了再做 §1.5 的 DLNA 默认接入与全量化。M1 不过（如 SKYBOX 长片拖动仍异常），退 Option A（TS-live）。

### 1.1 一次性元数据准备（每个 video+config 算一次，缓存）
- 解析源 MP4 的视频样本表（每 sample：size / offset(源文件内) / duration / DTS/PTS / 是否关键帧 / `ctts` / `hvcC`）——已由 `pipeline/si_virtual_mp4.read_video_sample_table` 提供雏形。
- 用 `build_si_mix_filter(...)` 把混音音频**预生成为小 sidecar**（`-map [si_track] -c:a aac`，几十 MB），取其音频样本表（含 `esds`）。
- **解析式构造 progressive `moov`**：合并"源视频 stbl（平移 size/stts/ctts/stss）+ 音频 stbl"，按选定 mdat 交错布局重算 `stco/co64` → 得到完整 moov 字节。**禁止用 `ffmpeg -movflags +faststart`（要落整片）。** 序列化用手写 box writer 或 `pymp4`（需则单独引入；非生产级，仅做 box 组装），并**对照 ffmpeg 在短片上产出的真实 progressive moov（`mp4dump`/`MP4Box -info`）逐 box 校验**。
- 由布局算出**精确 Content-Length + 稳定字节↔区域映射**。
- 缓存 key = f(源路径, 源 mtime/size, si.wav mtime, `SIMixParams`)；配置变即新布局（热重载已具备）。

### 1.2 服务（虚拟区域映射）
- 维护 `Region` 表：init 段（内存）/ 各视频 sample（指向源文件偏移）/ 各音频 sample（指向 sidecar）。
- 处理 `Range: bytes=A-B`：定位穿过的 region → 视频段从**源文件**对应偏移读、音频段从 sidecar 读、init 段从内存取 → 拼接返回 `206` + 正确 `Content-Range`。
- **字节稳定不变量**：同一逻辑文件，字节 N 任意次请求恒等；给稳定 `ETag`。**moov/init 必须定义为逻辑文件偏移 0 的一部分，不得每次响应私自前置。**

### 1.3 实现工具：PyAV（限定范围引入）
- 用 PyAV 做：muxer 接口（写 file-like）、拦截顶层 box 边界（`ftyp/moov/moof/mdat/sidx`）、进程内取精确 sample size/PTS。架构即专家的 `FragmentCollector`（研究存档 §9.4/§9.5）。
- 备选不引 PyAV：用现有 ffmpeg.exe 生成 + mp4frag 式 box 拆分（同架构，仍需进程管理）。
- **打包风险先行验证**：单独做一次带 PyAV 的 PyInstaller 构建，确认 PyAV 自带 libav 与现有 CUDA/TensorRT/ffmpeg DLL（`utils/runtime_dll_paths.py`、`utils/tensorrt_runtime_libs.py`）不冲突，再正式开发。

### 1.4 Annex-B 注意
源是 `.mp4`（length-prefixed sample），`-c:v copy` 直接可用；若将来支持 `.ts` 源（Annex-B），需 bsf 转 length-prefixed 并正确构造 `hvcC`（研究存档 §9 专家三.1）。

### 1.5 DLNA 广播（M1 通过后才接入默认目录）
`[SI]` 条目改为：`url=/media_si/<key>.mp4`、`mime=video/mp4`、`dlna_pn=HEVC_MP4_MAIN`、`OP=01`、`size`=精确 Content-Length；标题/命名不变（`[SI]source_display_stem`）。**M1 验证前先藏在 seek 门控开关后，不进默认 DLNA 目录。**

---

## Option A（Phase 0 判否时的兜底）— TS-live，能播 + 章节 seek

- SI producer = `-c:v copy -map [si_track] -c:a aac -f mpegts`，挂到现有 `/passthrough_live` 模型（`video/MP2T`、`OP=00`/章节 `t=` 选起点、`.ts` 后缀提示 Skybox）。
- 复用 `LiveSession` producer/subscriber 去重、`_active_streams` 槽位、EOF 回收。
- 代价：无平滑进度条，只能章节跳转。**优点：不重复 2026-05-28 失败（它不假装可字节 seek），低风险、数日可落地。**

---

## 删除 / 改写清单

| 文件 | 动作 |
|---|---|
| `http_app/si_stream.py` | **删/重写**：废弃 fragmented-MP4 realtime + 估算 + padding + IP 共享 session |
| `http_app/routes_media.py` `/media_si` GET/HEAD | 据 Phase 0 结果：重写为 ① 虚拟 remux 静态 Range（Phase 1）或 ② TS-live（Option A） |
| `pipeline/si_virtual_mp4.py` *(Phase 1 新建)* | 源 moov 解析 + 音频预生成 + 布局计算 + 虚拟区域映射；PyAV 封装 |
| `dlna/content_directory.py` | `[SI]` 条目 res 改为最终容器/OP（Phase 1=mp4/OP=01，Option A=MP2T/OP=00）；命名/缓存键不变 |
| `utils/si_filter.py` / `runtime_settings.py` / `routes_control.py` / `ui/*` / `config.py` | **不变**（配置/滤镜/热重载/UI 复用） |
| `tests/test_si_mix.py` | 改：删 fragmented/估算/padding 用例；按所选路径加（虚拟区域字节稳定性 / 布局确定性 / DLNA 广播 / TS-live producer） |

---

## 测试计划

- Phase 0：真机矩阵（5 个播放器 × {progressive, fMP4, fMP4+sidx}），记录进度条/seek/尾部探测模式。**这是唯一能定方向的测试。**
- Phase 1 单元：源 moov 样本表解析正确；布局确定性（同输入 → 同 Content-Length / 同字节映射）；`Range` 跨 region 切片字节精确；字节稳定不变量（同 N 两次相同）；DLNA 广播 size=真实布局。
- Phase 1 真机：进度条、任意 seek、播到结尾、A/V 同步 + SI 延迟、运行中改参数下次 seek 后变声、多客户端。
- Option A 单元/真机：能播、章节跳转、热重载变声。

---

## 风险与对策

| 风险 | 对策 |
|---|---|
| ~~Skybox 不把 (f)MP4 当可 seek 文件~~ | **已由 Phase 0 排除**：三端均可 seek，progressive 最优 |
| **SKYBOX 长片中段字节 seek 未验证**（Phase 0 A 样本仅 30s，被整片缓存掩盖） | **里程碑 M1 先验**：真实 50min 8K 片、未缓存拖动；不过则退 Option A |
| **progressive moov/stbl 解析式构造复杂**（Phase 1 主要工程量） | 视频 stbl 直接平移自源 moov（非从零）；`pymp4`/手写 box + 对照 ffmpeg 短片 moov 逐 box 校验；co64 处理 >4GB |
| 虚拟文件字节不稳定（最致命，当年病根） | 布局提前算死 + 稳定 ETag + 视频字节直引源文件；**字节稳定不变量已被 `tests/test_si_virtual_mp4.py` + `tools/si_proto/virtual_range_selftest.py` 双层钉死** |
| PyAV 打包与现有 CUDA/TRT/ffmpeg DLL 冲突 | 开发前先做带 PyAV 的 PyInstaller 验证；冲突则退回 ffmpeg.exe 生成+拦截 |
| >4GB 偏移（8K） | progressive 用 `co64`；fMP4 用 `default-base-is-moof` 相对偏移规避 |
| B 帧 ctts / AAC priming / A-V interleave | 用 PyAV/ffmpeg 生成参考样本对照（`mp4dump`/`MP4Box -info`）校验 |
| 误把虚拟 MP4 推广到转码模式 | 明确非目标：转码模式 fragment 大小不可预知（研究存档 §9.2） |

---

## 落地顺序（给开发）
1. ~~Phase 0~~ —— **已完成**（SKYBOX/nPlayer/4XVR 抓包，结论 progressive，详见上文各端实测）。
2. ~~PyAV 打包验证~~ —— **已完成**（onedir 冻结 exe 能加载 av==17.1.0 并打开 fMP4，与现有 CUDA hook 共存，见 FINDINGS）。
3. **里程碑 M1（开工第一步）**：解析式 progressive moov 构造器 + 最小 `/media_si`（单长片、藏在 seek 门控后）→ 真实 50min 8K 片在 SKYBOX/4XVR/nPlayer 上验证未缓存中段拖动 Range。
4. **M1 过** → 全量化（多文件、缓存淘汰、A/V interleave、热重载、§1.5 DLNA 默认接入、删/重写 `http_app/si_stream.py`）→ 真机回归。
5. **M1 不过** → 退 Option A（TS-live，能播+章节 seek）。
6. 全程保留旧 `/passthrough_live` 与 `/media` 不动，便于回滚。

---

## 附录：Phase 0 生成命令模板
```bash
# A. progressive faststart（完整 moov）
ffmpeg -t 30 -i SRC.mp4 -i SRC.si.wav -filter_complex "<build_si_mix_filter>" \
  -map 0:v -c:v copy -map "[si_track]" -c:a aac -b:a 192k -ar 48000 -ac 2 \
  -movflags +faststart -f mp4 A_progressive.mp4

# B. fMP4
ffmpeg -t 30 -i SRC.mp4 -i SRC.si.wav -filter_complex "<build_si_mix_filter>" \
  -map 0:v -c:v copy -map "[si_track]" -c:a aac -b:a 192k -ar 48000 -ac 2 \
  -movflags +empty_moov+default_base_moof+frag_keyframe -f mp4 B_fmp4.mp4

# C. fMP4 + sidx（如需 mfra：再用 MP4Box 处理）
ffmpeg -t 30 -i SRC.mp4 -i SRC.si.wav -filter_complex "<build_si_mix_filter>" \
  -map 0:v -c:v copy -map "[si_track]" -c:a aac -b:a 192k -ar 48000 -ac 2 \
  -movflags +dash -f mp4 C_fmp4_dash.mp4
```
（`<build_si_mix_filter>` 取 `utils/si_filter.build_si_mix_filter` 的输出字符串。）
