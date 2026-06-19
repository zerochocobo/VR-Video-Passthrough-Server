# 研究存档：Windows DLNA 处理 TS 文件的方式 / MP4 字节 seek 失败根因 / 容器选型结论

> 日期：2026-06-17　分支：feature/si
> 来源：对 `debug_output/dlna_dir.pcapng`（Windows 自带 DLNA / WMP，nPlayer 客户端，源文件 `bilibili_732.ts`）的抓包分析，
> 以及 `debug_output/server.log`（我们 SI `/media_si` 在 SKYBOX 8K 上的失败日志）的对照分析。
> 用途：研究存档。配套落地方案见 `summary_20260617_REALTIME_TS_DEV_PLAN_CN.md`。

---

## 1. 抓包事实：Windows DLNA 怎么暴露一个 .ts 文件

把 `bilibili_732.ts`（2:18，~27.7MB）放进共享目录后，WMP 在 ContentDirectory 里给**同一个视频**挂了 **20+ 个 `<res>`**，分两类：

### 1.1 原生资源（客户端实际播放的那个）
URL 带 `0_` 前缀、**无** `?formatID`：
```
protocolInfo="http-get:*:video/mpeg:DLNA.ORG_OP=01;DLNA.ORG_FLAGS=01700000000000000000000000000000"
URL: http://192.168.31.185:10243/WMPNSSv4/3595152422/0_MTVfOTE0YTk4NGItODg0NTc1.mpg
```
- `video/mpeg` —— 容器是 **MPEG-PS**（`.mpg`），由源 `.ts` **无损 remux** 而来；
- `DLNA.ORG_OP=01` —— **只开字节 seek**，关时间 seek；
- **没有 `DLNA.ORG_CI`**（即 CI=0）—— 声明"未转码/原生"；
- **没有 `DLNA.ORG_PN`** —— 不声明 DLNA profile 约束。

nPlayer 对它的真实请求与响应：
```
GET .../0_MTVfOTE0YTk4NGItODg0NTc1.mpg   Range: bytes=12582912-
→ HTTP/1.1 206 Partial Content
  Content-Type:  video/mpeg
  Content-Length: 16436204
  Content-Range:  bytes 12582912-29019115/29019116      ← 真实精确总大小
  Accept-Ranges: bytes
  TransferMode.DLNA.ORG: Streaming
```
**总大小 29019116 是精确真值，不是估算。**

### 1.2 一堆转码备选
URL 带 `?formatID=62/66/70/108/...`，全部 `DLNA.ORG_CI=1`（转码）+ `DLNA.ORG_OP=10`（时间 seek），覆盖多种 profile：
`video/mp4`(AVC_MP4_*)、`video/mpeg`(AVC_TS_*_ISO)、`video/vnd.dlna.mpeg-tts`(AVC_TS_*_T)、`video/x-ms-asf`、`video/x-ms-wmv`，外加 rtsp 变体。
这些是 WMP 的**实时转码目标**，客户端不兼容原生流时才回退。

---

## 2. 结论一：Windows 字节 seek 之所以稳，是"静态文件 + 确定性变换"

| 维度 | Windows 的 `.mpg` 原生资源 |
|---|---|
| 资源性质 | **一个真实、固定大小的字节流**，字节 N 永远对应同一份内容 |
| Content-Length | **精确真值** 29019116，不是估算 |
| TS→PS 变换 | **确定性**纯容器变换（剥 188 字节 TS 包头、加 PS pack 头，不重编码），字节偏移可复现 → 并发 range 也对齐 |
| seek 原理 | MPEG-PS/TS 是**扁平、自同步**流（PS 用 `0x000001BA` pack 起始码，TS 每 188 字节有 `0x47` 同步字节），从任意字节切进去解码器都能 resync |
| OP/CI | `OP=01` 字节 seek、CI=0 原生 |

核心：**Windows 服务的是"一个真文件"，字节↔内容映射静态且可复现**，所以多连接、并发 range、任意位置 seek 都不会错位。

### TS→PS 变换是什么（概念澄清）
- **TS（Transport Stream，.ts）**：数据切成**固定 188 字节小包**，每包 4 字节头（含 `0x47` 同步字节 / PID / 连续计数）。为广播/丢包设计 → 自同步、可任意位置切入；代价是包头开销大、PAT/PMT 反复出现。
- **PS（Program Stream，.mpg/.vob）**：数据装进**变长 pack**，为可靠存储设计，开销比 TS 小。
- **TS→PS remux** = 把视频 ES、音频 ES 从 TS 包里拆出来重新塞进 PS pack，**不重编码**，只换外壳。因为变换确定，输出大小可精确算、字节偏移可复现。

---

## 3. 结论二：我们的 `/media_si` 为什么崩（SKYBOX 8K 日志实证）

当前 SI 实现（已废弃方向）：每个 HTTP Range 请求 → `byte→time` 反算 → 现起一个 `ffmpeg -ss t ... -movflags +empty_moov+frag_keyframe -f mp4 pipe:1`，吐出一个**全新的、带各自 ftyp/moov 的独立 fragmented-MP4**，并贴上估算的巨型 Content-Length（79 亿）。

`server.log` 实证（SKYBOX/2.0.2，SI_TEST_8K.mp4）：
```
media_si[24] range='bytes=329568396-' seek=125.068s
media_si[25] range='bytes=337512699-' seek=128.083s   ← 偏移 +7.9MB
media_si[21] end ... sent=888573  content_length=7617955257   ← 只发 0.8MB 就被掐断
```
- SKYBOX 在大文件上发起**几十个并发 `Range: bytes=N-`**，偏移每次跳 ~7-8MB；
- 每个请求都新起一个 ffmpeg（seek 125s→128s→130s…），且偏移超出 1MB 复用容差 → 不复用；
- 更糟：`_session_key` 只用 IP，并发连接同 IP **共用一个 session**，互相驱逐 → 谁也流不完整。

### 根因（两条，核心是第 1 条）
1. **MP4 的"字节↔内容"映射由 `moov` 索引决定，不是物理决定。** progressive MP4 = `ftyp + moov + mdat`，`moov` 记录每个样本的大小（stsz）、绝对字节偏移（stco/co64）、时间戳（stts）。播放器**先读 moov，再照 moov 说的偏移请求 mdat**。我们每个请求现编一份**不同的 moov / fragment 布局**，播放器拿不到一份一致索引 → 字节映射不稳定 → 乱码。
2. **机械放大**：同 IP 并发请求共用 session 互踢；冷却 `time.sleep(0.2)` 还持锁执行，进一步抖动。

> 一句话：我们伪造的是"每次都不一样的流"，而非"一个真文件"。这与 Windows 的做法在本质上相反。

---

## 4. 结论三：MP4 vs TS 的可行性判断

要让字节 seek 在 SKYBOX 这类多连接播放器上稳，二选一：

- **(A) 真实静态文件**：把混音结果落盘成真 MP4 再静态服务。Windows 同款可靠，但 **8K 中间文件 ≈ 源大小（~8GB）**，用户已否决。
- **(B) 虚拟 remux MP4**：只算一份确定的 moov + 小音频 sidecar，视频字节从源文件切。中间文件小、字节 seek 完美，但需**手写迷你 MP4 muxer**（B 帧 ctts、>4GB co64、AAC priming、A/V sync），工程量大。
- **(C) 实时 TS，借"无 moov + 自同步"**：输出 MPEG-TS，`byte→time→-ss` 实时起流。**TS 没有 moov、自同步**，从任意字节切入解码器能 resync，不要求字节精确对齐 —— 这正是源 `.ts` 和 Windows `.mpg` 用 `OP=01` 就能 seek 的原因。Content-Length 可用源视频大小**近似**（输出 ≈ 源视频 ES + 新音频 + ~3% TS 开销，误差几个百分点，TS 能容忍）。中间文件可为零。

### Content-Length 的关键差异
- **索引型容器（MP4）**：播放器照 moov 偏移取字节，Content-Length 与字节布局必须**精确自洽**，否则错位 → 只能用真文件（A）或算准布局（B）。
- **自同步流（TS/PS）**：播放器靠同步字节 resync，**不依赖全局索引**，Content-Length **近似即可**（用户"用原视频真值近似"的设想在 TS 下成立）。

---

## 5. 初步选型（已被第 6~8 节推翻，保留作过程记录）

> ⚠️ 本节的"采用实时 TS 字节 seek"结论，经第 6 节（发现 2026-05-28 旧复盘）与第 7 节（HEVC-in-PS 实测）后**已被推翻**。
> 直接看第 8 节"修正后的结论"。

采用 **(C) 实时 TS**。理由：
1. 同时满足"无大中间文件 + 视频 copy + 实时音频"三个硬约束，最贴近最初的简洁设想；
2. 与抓包看到的两个**成功案例同源同理**（源 .ts、Windows .mpg 都是自同步流 + `OP=01`）；
3. 本项目已有成熟的实时 MPEG-TS 管线（`pipeline/pynv_stream.py`、`/passthrough_live`），基础设施可复用；
4. 失败的根因（moov 索引不一致）在 TS 下**天然不存在**。

代价 / 待真机验证：
- seek 精度到最近关键帧（与拖普通 .ts 一致，可接受）；
- 需确认 SKYBOX 在 **TS 资源**上按顺序流读取、不再发散并发抓取（大概率成立——它对 `.ts` 走流式管线，对 fragmented-MP4 才发散）；
- Content-Length 近似误差对个别客户端的容忍度（首发用源大小 ×1.0~1.05，真机调）。

### 延伸价值（同样受第 8 节修正约束）
本项目其余实时模式（`/passthrough_live` 绿幕、alpha、2D→3D，以及 `/passthrough_seek`）本就基于 MPEG-TS。**注意**：下文第 6 节证明"实时 TS 字节 seek 当普通文件 scrub"已被实机判死，因此这条"统一伪 VOD 字节 seek 底座"的设想**不成立**；实时 TS 只适合做 live + 章节 seek。

---

## 6. 关键反驳一：项目内已有 2026-05-28 失败复盘（`/passthrough_seek` 已被判死）

第 5 节写完后翻到 `summary/summary_20260528_SEEKABLE_PASSTHROUGH_LESSONS_CN.md` —— 本项目**早就试过**"实时流伪装成可拖动文件"，结论是**放弃**。其三条结构性根因（均有 nPlayer/Skybox/HereSphere 实机证据）：

1. **§6.1 HTTP Range 不变量被破坏**：同一 URL 的字节 N，多次请求必须是同一字节。实时流每个 Range 新起 producer 从某时间点重新生成 → 字节偏移不稳定 → 严格播放器有权失败。
2. **§6.2 / §5.4 TS 被天然当 live**：nPlayer/Skybox 对裸 `video/MP2T`，**无论怎么写 `OP=11`/`CI=0`/`Interactive`/`Content-Length`/`availableSeekRange`，都按直播流处理，不给普通文件进度条**。nPlayer 是裸 GET 播放，根本不走严格 DLNA time-seek 流程，header 对它几乎无效。
3. **§6.3 fMP4 在线流 ≠ 静态 MP4**：nPlayer 发尾部/中部 Range 找 `moov`/sample table，`empty_moov` 给不出 → 卡死 / 自动跳下一项。

复盘的一句话结论（§13）："**实时生成流可以做 live、可以做章节跳转，但不能低成本伪装成普通 MP4 文件。**"

**对照我们的第 5 节方案**：第 5 节的"实时 TS 字节 seek"正好踩中 §6.2 这颗雷 —— 想用实时 TS 骗出文件式 scrub，2026-05-28 已判死。`-c:v copy` 只能削弱 §6.1（同一 `-ss t` 重读字节稳定），**救不了 §6.2（TS=live 是容器认知，与 copy/转码无关）**。

> 当年 `/passthrough_seek` 是 **NVENC 转码**（§6.1 病根最重）。SI 是 `-c:v copy`，§6.1 大幅缓解，但 §6.2 这堵墙不动。

## 7. 关键反驳二：HEVC 装不进 MPEG-PS（"换 video/mpeg 像 Windows"对我方内容物理不可行）

针对"上次只测了 `video/MP2T`、Windows 用的是 `video/mpeg`（MPEG-PS）、是否该改用 PS"的设问，做了实测。

源文件 `videos/SI_TEST_8K.mp4` = **HEVC Main, 8192×4096, 音频 AAC**。把 HEVC `-c:v copy` 灌进两种容器：

| 容器 | 命令 | 结果 |
|---|---|---|
| **MPEG-PS**（`video/mpeg`，Windows 用的） | `-f vob` / `-f mpeg` | 3 秒内 **3089 条 `buffer underflow`**；回读探测 **`codec_name=unknown`** —— HEVC 在 PS 里**根本不被识别** |
| **MPEG-TS**（`video/MP2T`，我方用的） | `-f mpegts` | **0 警告**；回读 **`codec_name=hevc`** 正常 |

根因：
- **Windows 那个 `.ts` 是 H.264**（抓包 protocolInfo 写 `AVC_TS_*`、codec GUID `{34363248...}`=H264）。H.264 低码率能装进 MPEG-PS，所以 WMP 能 TS→PS remux 成 `.mpg`、贴 `video/mpeg`、当普通文件服务。
- **我方 VR 内容是 HEVC/H.265**。**MPEG-PS 无 HEVC 标准流类型** → `codec_name=unknown`，任何播放器都解不出；且 PS 缓冲模型为几 Mbps 的 MPEG-1/2 设计，8K 码率直接持续 underflow。
- 若把 SI 转码成 H.264 迁就 PS，则丢掉"视频 copy、廉价"的全部意义，8K H.264 编码代价巨大且 PS 仍 underflow，不可行。

**结论：用 `video/mpeg` 复刻 Windows 的做法，对 HEVC 内容物理上走不通 —— 不是 header/MIME 问题，是容器装不下这个编码。**

## 8. 修正后的结论（替代第 5 节）

把第 6、7 节合起来，对 **HEVC** 内容，"能显示进度条 + 字节 seek 的文件式容器"只剩 **MP4** 一种：

| 容器 | 装 HEVC | 被当文件(有进度条) | 字节 seek 可行 |
|---|---|---|---|
| MPEG-PS (`video/mpeg`) | ❌（codec unknown + underflow） | ✅ | — |
| MPEG-TS (`video/MP2T`) | ✅ | ❌（被当 live，§6.2 实测） | — |
| MP4 (`video/mp4`) | ✅ | ✅ | 仅当**真实 moov + 字节稳定**（真静态文件 或 虚拟 remux） |

因此 SI 的现实选择收敛为二选一（实时 TS 字节 seek 当普通文件这条线作废）：

- **A. TS-live（能播 + 章节 seek，低风险，可立即落地）**：SI 走现有 `/passthrough_live` 模型（`video/MP2T`、`OP=00`/章节 `t=` 选起点），producer = 视频 copy + 音频混音。代价：无平滑进度条，只能章节跳转。**不重复 2026-05-28 失败，因为它不假装可字节 seek。**
- **B2. 虚拟 remux MP4（唯一"实时 + 真 scrub + 不落大盘"）**：视频样本表从源 MP4 的 `moov` 直接拿、视频字节从源文件切、只为小混音音频建表 + 拼一份带 `co64` 的 `moov`。复杂（co64 / ctts / AAC priming / A-V interleave），但**因为 SI 视频是 copy 而独有可行性**（当年 `/passthrough_seek` 转码做不到，§6.4）。
- ~~B1. 真静态 MP4 落盘~~：8K ≈ 源大小（~8GB），用户已否决。

### Windows `.mpg` 给我们的真正启发（最终版）
不是"用 `video/mpeg`"，而是"**服务一份真实、字节稳定、索引完整的文件**"。对 HEVC，这件事的等价物就是 **B2 的虚拟 remux MP4**。`video/mpeg` 只是 H.264 时代恰好可用的外壳，对 HEVC 已失效。

### 配套开发方案的状态
`summary_20260617_REALTIME_TS_DEV_PLAN_CN.md`（实时 TS 字节 seek）据本节降级为**高风险/会重复 2026-05-28 失败**，不应作为"普通文件 scrub"目标的落地依据；其内容仅在"目标 A：TS-live 能播+章节 seek"范围内有参考价值。下一步若要"真 scrub"，应转向 B2 的虚拟 remux MP4 可行性验证（先对该 8K 文件拼一份 moov，真机抓包确认 nPlayer/Skybox 是否给进度条 + 能 seek）。

---

## 9. 外部专家 fMP4 方案评估 + 实测 + PyAV 判断（2026-06-17）

就 B2"虚拟 remux MP4"咨询了外部专家，得到一份以 **fMP4（分片 MP4）** 构造虚拟文件的详细方案（init 段 `ftyp+moov(含 mvex/trex)` + 持续 `moof+mdat`，配 `sidx`/`mfra`，虚拟偏移映射，HTTP Range 切片）。评估如下。

### 9.1 实测验证（在本机 `SI_TEST_8K.mp4` 上）
- `ffmpeg -c:v copy -c:a aac -movflags +empty_moov+default_base_moof+frag_keyframe -f mp4`：**干净生成、0 警告**，回读 `hevc/aac/8192x4096` 正常；init 段确为 `ftyp+moov(mvex/trex)`。
- `-movflags +dash`：能产出 `sidx`（时间→字节索引）。
- 结论：**fMP4 能正常承载 8K HEVC**（与 §7 的 MPEG-PS 装不下 HEVC 形成对比），专家关于 init/fragment/hvcC/`default-base-is-moof`/`sidx`/`mfra` 的论述技术上**全部成立**。

### 9.2 专家方案的判断：成立，但有一个决定性前提（专家无我方失败史）
专家未见过 §6 的 2026-05-28 复盘，因此不知道**我们已试过 fMP4 并失败**（复盘 §5.5-5.6）。但那次失败的原因是"每个 Range 现编一份不同 fMP4、字节不稳定"，**恰恰就是专家反复强调的生死线"偏移必须稳定"**。所以两者不矛盾：专家方案（fragment 生成一次、缓存、布局固定）正是修复当年失败的解法。

专家给的 A~E 五种"保证 fragment 大小稳定"的策略，全部卡在一个前提（他自己点了，策略 E）：**fragment 大小必须能提前预知**。这条前提把命运分成两半：

| 模式 | 视频处理 | fragment 大小能否预知 | 虚拟 fMP4 |
|---|---|---|---|
| **SI 混音** | `-c:v copy` | **能**（视频样本表从源 moov 白拿 + 音频预生成一次） | ✅ 成立（策略 E，整份布局/`sidx`/`mfra`/Content-Length 提前算死、字节稳定） |
| 绿幕 / alpha / 2D→3D | NVENC 重编码 | **不能**（编码前不知每帧大小） | ❌ 只能退固定槽位 padding（Content-Length 灌水、丑、脆） |

**关键结论：专家方案对 SI 成立（因 SI 不重编码），但不能推广到实时转码模式——"把以前的都改成这个模式"不成立。SI 的特殊性正源于 `-c:v copy`。** 这与 §8 的 B2 同一结论，只是实现从"手写大 moov"换成"fMP4 + 让 ffmpeg 生成再拦截"。

### 9.3 仍未解决、且当年栽过的决定性未知数
专家也承认成败取决于**播放器是"扫 moof 按时间 seek"还是"按 Content-Length 比例猜字节"**。这正是复盘 §10.3 的教训。已知：nPlayer 能 seek 真静态 progressive MP4（§5.8）；我方在线 fMP4 当年失败（但因不稳定）。

因此投入造虚拟文件机器**之前**，必须先做一个便宜且决定性的隔离测试（见开发文档 Phase 0）：
> 生成短的**静态** fMP4 + 静态 progressive(faststart) 文件 → 放进现有 `/media` 静态路由 → Skybox/nPlayer/HereSphere 抓包看：有无进度条、能否 seek。
> 它把"播放器认不认 (f)MP4 文件"从所有虚拟文件复杂度里隔离出来，一个下午定生死。

### 9.4 PyAV 判断（分场景，结论相反）
- **用于 SI 虚拟 MP4 / box 拦截：合理、推荐。** 纯 CPU（copy+aac），PyAV 擅长"写 file-like + 拦截 box 边界 + 进程内取精确 sample size/PTS"，即专家的 `FragmentCollector`，免 subprocess/管道管理。
- **用于替换 GPU 转码管线（NVENC+matting+TRT+CuPy）：不合理、高风险。** 现有 `pynv_stream`/`matting` 与 ffmpeg 进程 + 自研 CUDA 深度绑定；PyAV 硬件编码支持更弱，迁移是巨大重写且不降复杂度，还要趟 PyInstaller 里 PyAV 自带 libav 与现有 CUDA/TensorRT/ffmpeg DLL 冲突。
- **落地建议**：把 PyAV 作为**新增、限定范围**的依赖仅用于 SI 虚拟 MP4；**先单独验证一次带 PyAV 的 PyInstaller 打包**（主要风险点）；不动 GPU 链。

### 9.5 开源参考（专家给出，按对本项目优先级）
1. **PyAV / libavformat** — 实际 Python muxer 接口（首选实现层）。
2. **kevinGodell/mp4frag**（JS）— "让 ffmpeg 正确 mux，再从字节流拆 init/fragment"的架构范例，逻辑易移植。
3. **Eyevinn/mp4ff**（Go）— 清晰的 init/track/fragment 创建参考。
4. **kaltura/nginx-vod-module**（C，AGPLv3 慎用代码）— on-demand MP4 repackager 整体架构参考。
5. **GPAC/MP4Box、Bento4** — 生成标准样本 + 兼容性校验工具（`MP4Box -info`、`mp4dump`）。
6. **beardypig/pymp4**（Python）— box 检查/原型工具，非生产级 muxer。

### 9.6 最终方向（交付开发）
- **SI 目标若为"真 scrub"**：先过 Phase 0 真机门槛 → 据结果在"progressive 虚拟 remux"或"fMP4 虚拟 remux"中选一，用 PyAV（限定范围）实现，复用已建好的 SI 配置/滤镜/热重载/UI/DLNA 命名。
- **SI 目标若可接受"章节 seek"**：直接走 Option A（TS-live），低风险即时落地。
- **不要**把虚拟 MP4 推广到转码模式；**不要**用 PyAV 替换 GPU 链。
- 详见 `summary_20260617_REALTIME_TS_DEV_PLAN_CN.md`（已据本节改写为正式开发计划）。

---

## 10. Phase 0 SKYBOX 抓包结果（2026-06-17）

素材：`debug_output/si_proto/{A_progressive,B_fmp4,C_fmp4_dash}.mp4`。抓包：`debug_output/si_proto/skybox.pcapng`。服务日志：`debug_output/server_skybox.log`。

### 10.1 关键事实

| 文件 | 容器 | SKYBOX 体验 | Range 行为 |
|---|---|---|---|
| `A_progressive.mp4` | progressive faststart MP4 | 缓存条快速读满，播放流畅 | 仅 1 个 `bytes=0-` 顺序请求，1.75s 拉完整个 92.8MB 文件 |
| `B_fmp4.mp4` | fragmented MP4 + `mfra` | 有进度条，可拖动，但缓存条追着进度条，体感卡顿 | 856 个 open-ended Range；先扫 `moof/mfra`，随后反复抓重叠 `mdat` |
| `C_fmp4_dash.mp4` | fragmented MP4 + `sidx`/`mfra` | 有进度条，可拖动，但体感同样卡顿 | 807 个 open-ended Range；先扫尾部/`sidx`，随后反复抓重叠 `mdat` |

pcap 按 TCP stream 统计：A 实际发送约 92.8MB；B 实际发送约 343MB；C 实际发送约 333MB。B/C 的网络读取量约为文件大小的 3.6 倍，足以解释体感卡顿。

### 10.2 修正判断

Phase 0 的答案不是"SKYBOX 不认 fMP4"，而是：**SKYBOX 认 fMP4 并能 seek，但 fMP4 请求模式很差**。因此不能再按"fMP4 能 seek → 优先 fMP4"的简单判定走。

对 SKYBOX，首选路线应改为 **progressive virtual remux MP4**：
- progressive faststart 走稳定的文件式顺序缓存路径；
- fMP4 仍可作为技术备选，但不作为 SKYBOX 首选；
- A 样本较短且被 1.75s 整片缓存，后续 progressive `/media_si` 仍需用真实长片验证未缓存状态下拖动 Range。

### 10.3 对开发计划的影响

`summary_20260617_REALTIME_TS_DEV_PLAN_CN.md` 已据此更新：Phase 1 默认先做 progressive virtual remux；fMP4 只有在其他目标播放器实测更稳时再启用。

---

## 11. Phase 0 nPlayer 抓包结果（2026-06-17）

素材：`debug_output/si_proto/{A_progressive,B_fmp4,C_fmp4_dash}.mp4`。抓包：`debug_output/si_proto/nPlayer.pcapng`。服务日志：`debug_output/server_nplayer.log`。

### 11.1 关键事实

| 文件 | 容器 | nPlayer 行为 | Range 行为 |
|---|---|---|---|
| `A_progressive.mp4` | progressive faststart MP4 | 能播、能拖；多次播放/拖动混在同一日志 | 4 个无 Range `200` 全文件请求 + 8 个 open-ended Range；常见尾部和拖动位置请求 |
| `B_fmp4.mp4` | fragmented MP4 + `mfra` | 能播、能拖 | 2 个无 Range `200` + 36 个 open-ended Range；多数起点落在 `mdat` 内部 |
| `C_fmp4_dash.mp4` | fragmented MP4 + `sidx`/`mfra` | 能播、能拖 | 2 个无 Range `200` + 34 个 open-ended Range；多数起点落在 `mdat` 内部 |

pcap 按 TCP stream 统计：A 本次混合多次播放/拖动，实际发送约 185MB，其中一次完整 full-file 约 92.8MB/1.8s；B 实际发送约 94MB；C 实际发送约 99MB。B/C 没有出现 SKYBOX 那种 800+ Range / 3.6x 重叠读取风暴。

### 11.2 修正判断

nPlayer 对 progressive 和 fMP4 都能 seek；但它并未明显利用 fMP4 的 `sidx/moof` 边界，`C_fmp4_dash` 对比 `B_fmp4` 没看到收益。因此 nPlayer **不推翻** SKYBOX 给出的 progressive 优先结论。

当前跨播放器结论：
- SKYBOX：progressive 明显最好；fMP4 可 seek 但请求模式差。
- nPlayer：三者可 seek；fMP4 可用但无明显优势。

所以 Phase 1 默认路线仍是 **progressive virtual remux MP4**，fMP4 保留为后续播放器专项备选。

---

## 12. Phase 0 4XVR 抓包结果（2026-06-17）
素材：`debug_output/si_proto/{A_progressive,B_fmp4,C_fmp4_dash}.mp4`。抓包：`debug_output/si_proto/4xvr.pcapng`。服务日志：`debug_output/server_4xvr.log`。

### 12.1 关键事实

4XVR UA 为 Quest 3 上的 `Dalvik/2.1.0`。日志开头曾点到 `/passthrough_live/...A_progressive...` alpha 条目，这是现有 DLNA 模式项干扰，不纳入静态 `/media` Phase 0 判定。

| 文件 | 容器 | 4XVR 行为 | Range 行为 |
|---|---|---|---|
| `A_progressive.mp4` | progressive faststart MP4 | 能播、能拖动 | 静态 `/media` 中 2 个无 Range `200` + 1 个闭区间 Range `bytes=45088768-92795440` |
| `B_fmp4.mp4` | fragmented MP4 + `mfra` | 能播、能拖动，但请求压力大 | 1 个无 Range `200` + 45 个闭区间 Range，均为 `bytes=N-file_end` |
| `C_fmp4_dash.mp4` | fragmented MP4 + `sidx`/`mfra` | 能播、能拖动，但请求压力仍大 | 1 个无 Range `200` + 32 个闭区间 Range，均为 `bytes=N-file_end` |

pcap 按请求时间归属服务端 TCP payload：A 约 116MB（约 1.25x 文件大小）；B 约 589MB（约 6.35x）；C 约 439MB（约 4.73x）。server log 的响应声明 length 更夸张：B 约 2.68GB，C 约 2.08GB，这是因为 4XVR 请求的是闭区间到文件尾，但常会提前断开。

用 `BoxSplittingSink` 映射 B/C 的 Range 起点后确认：B 的 45 个起点、C 的 32 个起点全部落在 `mdat` 内部，没有命中 `moof`、`mfra` 或 `sidx` 边界。4XVR 的 Range 起点大量 1MiB 对齐，更像按缓存窗口/估算位置反复拉取到尾部，而不是按 fMP4 索引精确取 fragment。

### 12.2 修正判断

4XVR 同样证明：fMP4 不是“不能 seek”，而是“可 seek 但请求模式不优”。`C_fmp4_dash` 没体现出 `sidx` 收益，B/C 的实际网络读量明显高于 A。

当前跨播放器结论：
- SKYBOX：progressive 明显最好；fMP4 可 seek 但触发 800+ open-ended Range 和卡顿。
- nPlayer：三者可 seek；fMP4 无明显优势。
- 4XVR：三者可 seek；fMP4 触发大量闭区间到尾部读取，实际读量 4.7x~6.3x。

因此 Phase 1 默认路线继续锁定 **progressive virtual remux MP4**。fMP4 不删除，但只作为播放器专项 fallback，不作为通用默认。
