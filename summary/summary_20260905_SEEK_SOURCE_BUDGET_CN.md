# passthrough_seek 源预算方案（source-size budget）

日期：2026-09-05　分支：`feat/seek-source-budget`
前置：`summary_20260528_SEEKABLE_PASSTHROUGH_LESSONS_CN.md`、`summary_20260619_PASSTHROUGH_SEEK_REWORK_PLAN_CN.md`

---

## 1. 这一轮做了什么

把 `feature/passthrough_seek`（停在 2026-06-20 的 slot_frames 虚拟 MP4）合并到当前主线，
并在它之上换掉「每帧一个固定预算常数」的比特分配：**改为继承源文件自己的逐 GOP 比特分布**，
使虚拟文件大小精确等于源文件大小。

顺带修掉一个会让播放器在放出第一帧之前就放弃的缺陷（结构性 Range 被 503）。

---

## 2. 用户提出的原始想法与判定

原始想法：**moov 完全使用原视频的，程序解析 moov，把播放器请求的字节位置映射回时间点，返回实时数据。**

拆成三段判定：

| 设想 | 判定 | 依据 |
|---|---|---|
| 声明 size/duration 等于源 | 可行，2026-06-19 切片 1/2 已验证 | 播放器能出正常进度条 |
| 解析 moov 精确反查 byte→time | 可行且正确 | 比旧的线性 CBR 映射精确；解析能力现成（`si_virtual_mp4`） |
| **stsz/stco 照抄源 + 返回实时数据** | **不成立** | 见下 |

第三段不成立的原因：播放器不是「从这个 offset 顺序读流」，而是**按 moov 把字节切成 sample 逐个喂解码器**。
照抄源 `stsz` 等于要求我们实时编码的每一帧恰好等于源那一帧的字节数，而：

1. 源是 VBR，逐帧预算剧烈波动（B 帧可能只有 2–5 KiB），我们的重编码帧塞不进去。
   2026-06-20 nPlayer 真机就死在这里：当时用的还是宽松的统一 160 KiB P 预算，
   仍出现 `idx=54:478029` 这种尖峰把整个 GOP 判死（§28）。
2. 帧数对不上：`PASSTHROUGH_MAX_FPS=30`，源多为 59.94fps，输出 sample 数直接减半。
3. `stsd/hvcC` 是源编码器的参数集，而我们的码流是 NVENC 出的，分辨率也可能不同
   （取决于 `PT_DECODE_MAX_SIDE`）。换成我们自己的之后，moov 已经不是「原来的」了。
4. §4.4 早算过账：总大小锁死 = 总码率锁死。

**但想法里「源文件已经告诉了我们比特该怎么分配」这个直觉是对的**，这是本轮落地的部分。

---

## 3. 落地形态

`stsz/stco/stts/stss` 全部自己写，只从源 moov 继承**GOP 级字节预算**：

```text
gop_budget[k]  = 源文件在第 k 个输出 GOP 时间窗口内花掉的视频字节
              -> clamp 到下限, 缺口按超出量比例从富余 GOP 扣回（总量守恒）
frame_budget[i] = gop_budget[i // gop] / gop_frames      # GOP 头拿余数
sum(frame_budget) = 源文件大小 - (ftyp + moov + mdat header)
```

实现：`pipeline/source_budget_plan.py`（新增）、`pipeline/passthrough_vmp4_frames.py`
（layout 接受 `frame_budgets` 覆盖）、`http_app/routes_media.py`（接线 + 诊断头）。

### 3.1 为什么 clamp 是必须的

源的黑场/静止段会掉到 ~2 KiB/帧（SI_TEST_8K 的 p1 = 1.8 KiB，VENTA 8K 的 p1 = 2.4 KiB），
但我们抠像合成后的输出在那里不一定是静止的。下限 48 KiB/帧
（`PT_PASSTHROUGH_SEEK_VMP4_FRAMES_FLOOR_BYTES`）把这些窗口抬起来，代价从富余窗口按比例扣。

实测代价（中位数下降）：STAYC 0.3%、SI_TEST_8K 2.2%、VENTA 2.7%，其余 0%。总量守恒误差 0.0000%。

### 3.2 为什么不用重建 encoder

逐 GOP 变码率原本要担心「每 GOP 重建 encoder」——那正是 §30 里把吞吐拖到 6–15fps 的元凶。
本机 PyNvVideoCodec 支持 `PyNvEncoder.Reconfigure(structEncodeReconfigureParams)`，
可在会话中途改 `averageBitrate` / `maxBitRate` / `vbvBufferSize` / `vbvInitialDelay`。

实测（1280×720 噪声，3M/24M 交替 schedule）：

```text
GOP0 目标  3.0 Mbps -> 实际  6.60 Mbps
GOP1 目标 24.0 Mbps -> 实际 12.99 Mbps
GOP2 目标  3.0 Mbps -> 实际  4.36 Mbps
GOP3 目标 24.0 Mbps -> 实际 11.53 Mbps
```

跟随关系清楚（低目标段约为高目标段的 1/3）。绝对值超标是因为测试内容是纯随机噪声（不可压缩）。
**注意：单帧最大仍达 264 KiB/776 KiB，说明 1 帧 VBV 不是硬顶**，所以 §30 的
per-frame oversize clamp 必须保留（现状：clamp 到预算，单帧轻微 glitch，不再让整个 GOP 永久 503）。

`iter_pynv_passthrough_annexb_frames(..., per_frame_cap_schedule=...)` 在预算变化处调用 Reconfigure，
持久会话不变。

---

## 4. 判定数据（本地素材，`tools/seek_source_budget_probe.py`）

| 素材 | 分辨率 / codec | 源码率 | live 实际<br>`min(50M, 源×2)` | 源预算 | = live 的 | 每帧 KiB<br>源 → 我们 |
|---|---|---|---|---|---|---|
| SI_TEST_8K | 8192×4096 hevc | 19.6 M | 39.2 M | 19.6 M | 50% | 40 → 80 |
| VENTA X 8K | 8192×4096 hevc | 35.3 M | 50.0 M | 35.3 M | 71% | 72 → 144 |
| STAYC 180VR | 7350×3972 hevc | 42.7 M | 50.0 M | 42.7 M | 85% | 87 → 174 |
| 72456_3840p | 7680×3840 **av01** | 24.7 M | 49.4 M | 24.7 M | 50% | 50 → 100 |
| 2_2 | 8192×4096 hevc | 30.3 M | 50.0 M | 30.3 M | 61% | 62 → 123 |
| test_4k_full_vr | 3840×1920 **h264 / 29.97fps** | 12.0 M | 24.0 M | 12.0 M | 50% | 49 → 49 ⚠ |

关键：**判据是每输出帧的字节数，不是码率。**输出 fps 被 `PASSTHROUGH_MAX_FPS=30` 砍半，
所以 60fps 源给我们的每帧字节是它自己的 2.0 倍，正好抵消 NVENC 实时（P1 / ultra_low_latency）
相对源离线慢档的效率劣势。

**注意输出分辨率取决于 `PT_DECODE_MAX_SIDE`**：本机 GUI 实际传的是 `0`（不限制），
所以 8K 源的输出就是 8K，**没有像素红利，只有 fps 红利**（每帧 2 倍字节）。
早先在 config 默认值（4096）下测到的 4096×2048 输出不代表实际运行配置——
如果这个值被设成 4096，8K 源降到 4K 输出，像素只剩 1/4，预算会宽裕得多。
判定时以实际生效值为准。

**唯一不利档是等 fps 源**（最后一行）：没有 fps 红利，每帧字节与源持平，只能靠
HEVC 对 H.264 的效率优势兜底。probe 工具会对 ratio < 1.5 打 `thin` 标记。

### 4.1 layout 校验

`--layout` 用真实源建 layout：

```text
2_2.mp4       total=226187465  source=226187465  delta=0  moov=22314   frames=1775   LAYOUT OK
STAYC         total=1210443143 source=1210443143 delta=0  moov=82336   frames=6750   LAYOUT OK
SI_TEST_8K    total=7474531284 source=7474531284 delta=0  moov=1091862 frames=90210  LAYOUT OK
```

8K 50min 的 layout 构建（含解析 180240 个源 sample）约 4 秒，且有缓存。moov 1.09 MB。

---

## 5. 顺带修掉的关键缺陷：结构性 Range 被 503

用 TestClient 打真实路由发现：`bytes=0-22277`（完全落在 ftyp+moov 内，一个 mdat 字节都不碰）
返回 **503 "GOP not ready"**。原因是 frames 后端把每个 Range 都解析成 GOP 再等它就绪。

播放器打开 MP4 的第一件事就是读文件头、探测尾部（§5.8 的 nPlayer 抓包对真实文件也是这个行为）。
头都拿不到，播放器直接放弃——**这很可能是 2026-06-20 真机上 4XVR 卡死 / MoonVR 无进度条的原因之一。**

`iter_vmp4_frames_range` 本来就能服务任意偏移（init 走内存，未编码帧走 placeholder+filler），
这些 Range 根本不需要等。现在分类直接服务：

- `init`：范围完全在 ftyp+moov 内 —— 永远直接返回（不论是否 bounded）
- `tail`：bounded 且落在文件尾部区域 —— 播放器在找它以为存在的尾部 moov，我们没有，返回 filler
- `header-crossing`：bounded 且从头部跨进前几帧
- 阈值 `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_PROBE_BYTES`（1 MiB，设 0 关闭）

开放 Range 和中段 bounded 读仍按播放处理，照旧等待编码。

修复后实测：

```text
纯 init   bytes=0-22277        -> 206  22278 B   head=b'\x00\x00\x00\x1cftypisom'
尾部探测  bytes=226180857-     -> 206   6608 B   (filler)
中段探测  bytes=113093732-+64K -> 503            (仍按播放等待)
```

---

## 5.5 端到端验证（真实 GPU 编码 + 解码 + seek）

`dance_38110363874-1-192_4K.mp4`（4K，88.5s，2654 帧，输出 1216×2160 green）走完整路由跑通：

```text
GET (no Range) -> 200   frames-wait-ready: 1   frames-source-budget: 1
收到 6.03 MB（流式）
后台 filler 编完整片 2654/2654 帧
```

再用 `tools/vmp4_frames_snapshot_check.py` 把已构建的帧按 layout 拼成一个真实文件：

```text
layout total=220936795  source=220936795   frames=2654  source_budget=True
digest 与运行时一致       写出 220936795 字节  match=True

ffprobe : codec=hevc 1216x2160 nb_frames=2654 avg_frame_rate=2997/100 duration=88.555
ffmpeg  : 解码 600 帧            rc=0 无错误
ffmpeg  : -ss 40 后解码 120 帧   rc=0 无错误     <- 拖动进度条的核心行为
```

这是这个功能第一次拿到「完整、真实内容、可拖动解码」的产物。产出的快照已放到
`videos/VMP4_SNAPSHOT_dance4K.mp4`，可以直接在头显里走静态 `/media` 路径播放，
用来把「容器/内容是否被播放器接受」和「实时生成/并发」两个变量分开测（见 §8）。

### 5.5.1 顺带发现：headroom 0.6 让 40% 带宽变成 filler

实际 payload：min=18373 p50=46833 max=187808，而预算 min=74905 p50≈83234 max=316058。
中位数只用掉预算的 56%，其余是 filler。这是 `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_RATE_HEADROOM=0.6`
的直接后果——那个值是 §29 在**全局固定预算**下为了防 overshoot 定的。

现在预算跟着内容走（同一 GOP 内帧复杂度相近，CBR 更容易命中），而且实测没有一帧超预算，
所以 headroom 有明显的上调空间（0.8 大约能把浪费从 44% 降到 20%，直接变成画质）。

**本轮没有改这个默认值**：真机从未通过，先只改一个变量。等实机跑通之后再调，
调完重点看日志里 `frames oversize` 是否开始出现。

## 5.6 真机结果：静态快照通过（2026-09-06）

`videos/VMP4_SNAPSHOT_dance4K.mp4` 在 **5 种 VR 播放器上实测，拖动跳转播放均正常。**

这是 2026-05-28 以来第一次拿到的正面真机结果，它确定性地排除了一整类嫌疑：

- 帧级固定预算 + filler 的 access unit 结构，移动端硬解**能解**
  （§25 的 C 项当年担心的"giant filler AU 超过硬解 per-AU 限制"在当前预算下不成立）；
- 单 moov + co64 + 逐帧 stsz 的布局，播放器**认**，并据此正确 seek；
- `stts=1/fps` 的真 fps 时间轴、`stss` 只标 GOP 头，播放器**接受**，不再是 0.83fps 幻灯片；
- 源预算分配出的码率水平（这个片是 4K，均值 ~81 KiB/帧），画质与结构都过关。

**剩余失败面因此收窄到实时那一侧**：

| 变量 | 状态 |
|---|---|
| 容器结构 / co64 / stsz / stts | 已验证（5 播放器） |
| 内容（NVENC + 源预算 + filler） | 已验证（同上） |
| HTTP 结构性探测（init/tail） | 已修，路由级验证（§5） |
| 实时编码吞吐 | **未验证** |
| 并发多连接抖动（§31） | **未验证** |

换句话说：**如果 `.seek.mp4` 实时项在真机上仍然失败，原因一定在实时生成或 HTTP 并发，
不可能再是容器或内容。**这条排除线值得记住，能省掉未来大量的错误方向。

## 5.7 并发压测：找到并修掉片尾死循环（2026-09-06）

静态快照通过后，剩余风险只剩实时侧，于是用 `tools/vmp4_frames_concurrency_probe.py`
在本地重放播放器的并发 Range 模式（1 个顺序播放 + 头部/尾部探测 + 3 个分散 seek，
并发发出，重复 3 轮，8K 素材 `2_2.mp4`，每轮前清空帧缓存）。

### 根因：最后一个 GOP 永远不就绪 → filler 无限重启

日志里每 6 秒一次 `PyNv frame stream begin: start=58.000 frames=33`，反复几十次，
而且发生在所有请求都返回之后的空档期。

`layout.frame_count` 来自 `duration × fps` 四舍五入，比解码器实际能吐出的帧数**多 1 帧**
（日志里 `frames=33` 而该 GOP 有 34 帧）。于是：

1. filler 跑到源尾部、生成器耗尽，最后一帧的 `frame_XXXXXX.bin` 永远没写；
2. `_vmp4_frames_gop_ready` 对最后一个 GOP 永远返回 False；
3. 每个请求该 GOP 的连接都调 `ensure_filler`，而 `state="done"` 的 filler 不受冷却保护 → 立刻重启；
4. 每次重启 = 一次 8K 建管线（约 6 秒），**无限循环**，GPU 被这个循环占满，
   其他连接因此拿不到资源而 503。

**这是真机上一定会发生的 bug**：播放器打开文件就探测尾部，播放到片尾也会碰到。

修法：生成器耗尽（而非被抢占）时，把 `[cursor, frame_count)` 的空缺写成 0 字节文件。
range 迭代器本来就会把空 payload 当 filler 服务——那些尾部 sample 本来就是 filler。

### 另一处时序问题：冷却从创建计时

`_VMP4_FRAMES_RESTART_COOLDOWN` 原本从 filler **创建**时刻算 8 秒，但 8K 建管线
（开解码器、CreateEncoder、matter reset）本身就要 5–10 秒，等于整个冷却窗口都花在 setup 上，
filler 刚开始产帧就被下一个连接抢走。改为**从写出第一帧开始**计时，并给正在产帧的 run
一个更长的保护窗口（`_VMP4_FRAMES_PRODUCTIVE_HOLD = 15s`）。

### 效果

| | 修复前 | 只改冷却计时 | + 片尾修复 |
|---|---|---|---|
| GPU 会话重建（3 轮 18 请求） | **26** | 25 | **4** |
| 顺序播放连接首 4 MB（第 3 轮） | 49.7s | 49.5s | **7.3s** |
| 503 | 1 | 1 | 2（都在冷启动首轮） |
| oversize | 0 | 0 | 0 |
| 结构性探测（头部/尾部） | — | 全部 0.3–1.3s 秒回 | 同 |

冷却计时的改动单独看几乎没效果（26→25），**真正的根因是片尾死循环**；但它本身是时序错误，保留。

首轮冷启动仍有 2 个 503：8K 从零开始时 6 个连接抢一块 GPU，这是物理限制，
播放器按 `Retry-After: 1` 重试即可，第二轮起全部成功。4K 素材会快得多。

## 5.8 真机通过 + 音频与吞吐（2026-09-06）

Skybox 上 `.seek.mp4` **播放成功，可以拖动跳转**。剩下两个问题及处理：

### 问题 1：没有声音 —— 已修

frames layout 一直是 video-only（2026-06-19 的计划里明确写了「首版做 video-only，
验证拖动和视频可解码，通过后再加音频」，见 §4.3）。现在拖动验证通过，补上了。

做法：**音频原样 copy，不重编码**。sample 的 size/duration 用源的，只有它在虚拟文件里的
位置是我们的；moov 里的 audio trak 保留源 stsd/timescale，只重写 chunk offset
（复用 SI 路径的 `_rewrite_audio_trak_samples`）。布局改为**按 GOP 交织**：
`[GOP k 的视频帧][GOP k 的音频][GOP k+1 ...]`，播放器顺序拉流就能同时拿到两轨。
音频字节从预算里先扣，总大小仍精确等于源大小。

验证（`dance_38110363874-1-192_4K.mp4`）：

```text
layout total=220936795 = 源大小, 区域间空隙 0
ffprobe: hevc 2654 帧 + aac 3813 帧, duration 88.555
虚拟文件解码 30s 音频 PCM 的 md5 == 源文件同段的 md5
```

最后一条是关键：copy 出来的确实是源音频，不是错位的字节切片。

### 问题 2：卡顿 —— 根因是 Annex-B 扫描，不是算力

初测产出速率只有 dance 4K green 21.7fps（需 30fps，0.72x）、现场 8K alpha 约 0.5x，
一度以为是 GPU 算力天花板。**用户的质疑推翻了这个结论：live 模式 4K 能跑 80fps+，
同样的 matter、同样的编码器，凭什么 seek 只有 21.7？**

分解测试（`scratchpad/bench_filler.py`，300 帧，逐项叠加）：

| 配置 | 优化前 | 优化后 |
|---|---|---|
| A live 风格（无 cap/CBR/VBV） | 16.7 | **57.5** |
| B +CBR+1帧VBV（当前 seek） | 31.6 | 59.8 |
| C +逐 GOP Reconfigure | 32.0 | 59.0 |
| D +AnnexB→length-prefix 转换 | 21.7 | 57.5 |
| E +每帧原子落盘（完整 filler） | **20.8** | **52.5** |

根因是 `_annexb_start_codes` 在**纯 Python 里逐字节扫描**起始码，每个位置做一次 4 字节和
一次 3 字节切片比较——42 KiB 的帧要 ~11ms，而 frames 路径**每帧扫两遍**
（切 access unit 一次、length-prefix 一次）。live 从不付这个代价，它把 bitstream
直接交给 muxer。

修法：4 字节起始码就是 3 字节起始码前面多一个 0，所以一次 `bytes.find` 循环
（C 层扫描）就能找出两种形式。等价性对照旧实现验证：所有边界形状
（2/3/4 个前导零、相邻起始码、截断尾巴）+ 300 条零字节密集的随机流，全部一致。

```text
单帧 42 KiB 扫描      10.87ms -> 0.01ms   (1000x)
完整 filler 吞吐      20.8 -> 52.5 fps    (2.5x)
```

**52.5 fps 对 30 fps 的播放需求 = 1.75x 实时**，播放器终于能攒出缓冲。
按同样倍数推算，8K alpha 应从 ~0.5x 提到 ~1.2x。

剩余开销：每帧原子落盘约占 9%（52.5 vs 57.5），远排在后面，暂不动。

`PT_PASSTHROUGH_SEEK_VMP4_FRAMES_MAX_FPS`（只压 seek 路径帧率，降帧率同时按比例
放大每帧预算）仍然保留，作为 8K 或更重模式的后备手段；默认 0（跟随 live），
在 4K 上现在应该用不到。

另一个已知的浪费：实测 payload 中位数只占预算的 55%
（dance：p50 42 KiB vs P 预算 77 KiB），其余是 filler，源于
`_FRAMES_RATE_HEADROOM=0.6`。容器和音频都验证过之后，这个值有明显上调空间
（0.8 大约把浪费从 45% 降到 20%，直接变画质），代价是 oversize clamp 可能偶发。

## 5.9 中间时间点 seek 失败：三个叠加故障（2026-09-06）

现象：顺序播放正常、有声音，但**点进度条中间某点会先播一下就自动跳到下一个视频**，
一次都稳不住。日志（IPVR-060）里三个特征同时出现：5 次 seek 对应 5 条 `stream begin`、
两条 matter DIAG 计数序列一直交替到日志末尾、seek 后的响应只有几百 KB。

### 故障 1：Matter 池被耗尽

filler 重定位时会 `stop_event.set()` 停掉旧的，但 **worker 只在产出帧之间检查这个标志**，
而生成器的 setup（打开 8K 解码器、`matter.reset_state()`、`CreateEncoder`）要好几秒。
于是旧 run 在 setup 里攥着 Matter 不放，新 run 又去要一个——`PT_MAX_CONCURRENT=2`，
**第三次 seek 就没有可用的了**。日志里两条 matter 计数交替就是两个 run 同时活着的证据。

修法：生成器接受 `cancel` 事件并在每个 setup 阶段检查；新 filler 等前任的
`done_event`（Matter 已归还）再去 acquire。

### 故障 2：去抖窗口把真实 seek 也挡了

`_VMP4_FRAMES_PRODUCTIVE_HOLD=15s` 本意是防止并发连接互相踢 filler，
但它不区分「并发探测」和「用户真的拖了进度条」，15 秒内的 seek 一律拒绝重定位。

修法：只在**还有请求落在 filler 覆盖范围内**时才去抖。播放器 seek 的方式是断开旧连接、
在新位置开一个，所以 2 秒内没有请求命中旧位置 = 没人跟了 = 立即让位。

### 故障 3：一帧就返回，播放器起不了播

206 只等**首帧**就绪就返回（这是为了压低冷启动延迟做的）。结果刚 seek 完的播放器
拿到不到一秒的视频，播完发现后面没有了，就换下一个 item——正是用户描述的现象。

修法：等一个 GOP 的连续帧（`PT_PASSTHROUGH_SEEK_VMP4_FRAMES_PREBUFFER_FRAMES`，
默认 = `PASSTHROUGH_GOP`，设 0 恢复旧行为）。当前吞吐下约 1 秒代价。

### 效果（冷缓存重放 seek 序列）

| 动作 | 修复前 | 修复后 |
|---|---|---|
| seek 到 40% | 0.4 MB | **3.1 MB** |
| 继续播 41% | 0.0 MB | **3.1 MB** |
| seek 到 70% | 0.2 MB | **3.1 MB** |
| seek 回 20% | 0.1 MB | **3.1 MB** |

首字节 0.4–3.9s；4 个位置 4 个 filler 会话、3 次干净交接、
**0 次 matter 超时、0 次 GOP not ready**。

## 5.10 尾部 GOP 预算把编码器搞崩（2026-09-07）

现象：修完切换视频的问题后，Skybox 仍然「跳转后放一会儿就跳走」。

本地复现（seek 到未缓存的 55% 后连续拉取）第一个请求就 503、等满 30 秒，
日志里是**连续三次 `NvEncInitializeEncoder ... error 8`（INVALID_PARAM）**——
filler 建不出编码器，那段内容永远产不出来。

### 根因：不完整的尾部窗口拿了整窗的预算

`test_4k.mp4` 1802 帧、GOP 60，**最后一个 GOP 只有 2 帧**。而窗口权重取的是
「源在这 2 秒里花了多少字节」= 7.5 MB，IDR 权重再把其中 80% 给了那一帧：

```text
frames[1800].budget = 94,590,647   (90 MB 一帧)
   -> bitrate 13,621 Mbps   vbv 756 Mbit   -> NVENC error 8
```

**帧数不是 GOP 整数倍的视频都有这样一条尾巴**，也就是几乎所有视频。

修法：窗口权重按**它实际产出的帧数**缩放（`w * gop_sizes[i] / gop`），
floor 也改成逐 GOP（尾部 GOP 的 floor 是 `floor_frame × 实际帧数`，
不再是整个 GOP 的）。

| 素材 | 尾 GOP | 最大帧预算/均值 前 → 后 |
|---|---|---|
| test_4k.mp4 | 2 帧 | **47.7x → 4.1x** |
| test_4k_multi.mp4 | 3 帧 | **32.2x → 4.4x** |
| 2_2.mp4 | 34 帧 | 5.9x → 5.9x |
| dance_4K | 14 帧 | 4.0x → 4.0x |

所有素材的 `total - source_size` 仍然是 0。

另外加了 `PT_PASSTHROUGH_PYNV_MAX_RC_BPS`（500 Mbps）钳住送进编码器的
bitrate/VBV：预算来自源数据，一个坏值应该降画质，而不该让编码器初始化失败。

### 顺带：不再回极短的响应

播放器读得远比播放快——Skybox 3 秒拉了 13 秒内容，而编码器只有 ~1.2x 实时，
所以 seek 后几秒内读取位置就会追上编码位置。此后每次响应只带「上次之后新编的那几帧」，
播放器把它当成流结束就跳走了。

`PT_PASSTHROUGH_SEEK_VMP4_FRAMES_MIN_SPAN_BYTES`（4 MiB）保证最小响应量，
超出已编码范围的部分用 filler 顶（短暂定格），而不是用一个极短的包结束播放。

### 效果

seek 到未缓存的 55% 后连续拉 12 次：

```text
修复前  # 0 -> 503   等待 30.6s  (编码器建不起来)
修复后  # 0 -> 206   8.23MB  6.32s   (冷启动 setup)
        # 1..11 -> 206  7.3-8.8MB  每次约 0.95s
        过短响应 0/12
```

## 6. 开关与默认值

合并时**刻意偏离** `feature/passthrough_seek` 的默认值，保证不影响已有实时模式：

```text
PT_PASSTHROUGH_SEEK_ENABLED=0           # 分支上是 1
PT_PASSTHROUGH_SEEK_DLNA=0              # 分支上是 1
PT_PASSTHROUGH_SEEK_ROUTE_POLICY=profile  # 分支上是 all
PT_PASSTHROUGH_SEEK_CONTAINER=mp4
PT_PASSTHROUGH_SEEK_VMP4=1
PT_PASSTHROUGH_SEEK_VMP4_BACKEND=slot_frames
```

新增：

```text
PT_PASSTHROUGH_SEEK_VMP4_FRAMES_SOURCE_BUDGET=1     # 源预算总开关
PT_PASSTHROUGH_SEEK_VMP4_FRAMES_FLOOR_BYTES=49152   # 48 KiB/帧下限
PT_PASSTHROUGH_SEEK_VMP4_FRAMES_PROBE_BYTES=1048576 # 结构性探测阈值
```

同时删掉了 `20260619_seek_vmp4_default_on` / `20260619_seek_vmp4_slot_default` 两个 migration
——它们会强制打开 seek 并覆盖用户显式设置。

DLNA 层整体保留主线版本（Live 一律容器、superres seek 项、two_dvr 不出 seek 项），
未引入分支对 DLNA 的改动。`_seek_output_mode` 保留主线的 green/alpha/superres 集合，
因为 frames 后端服务不了 two_dvr。

---

## 7. 尚未验证 / 已知风险

1. ~~真机从来没走通过~~ **静态快照已在 5 种播放器上通过（§5.6）**，容器与内容不再是嫌疑。
   但 `.seek.mp4` **实时项**本身仍未在真机上验证过。
2. **并发抖动**已在本地压测下大幅缓解（§5.7：GPU 会话重建 26 → 4），
   但真机播放器的并发模式与本地重放未必一致，仍需实测确认。
3. **实时吞吐**（§5.8）：修掉 Annex-B 扫描后 4K green 到 1.75x 实时，8K alpha 待真机确认。
   教训：**在把性能问题归因于"算力天花板"之前，先跟一条已知更快的路径（live）逐项对比。**
4. **等 fps 源**没有 fps 红利，画质可能低于 live。probe 工具会标 `thin`。
5. 单帧 VBV 不是硬顶，oversize clamp 仍会偶发触发（单帧 glitch）。

---

## 8. 实机测试怎么跑

1. 关掉服务器，编辑 `runtime_cache/ui_settings.json`：

   ```json
   "passthrough_seek_enabled": true,
   "passthrough_seek_dlna": true,
   "passthrough_seek_route_policy": "all",
   "passthrough_seek_container": "mp4",
   "passthrough_seek_vmp4": true,
   "passthrough_seek_vmp4_backend": "slot_frames"
   ```

   `passthrough_seek_dlna` 打开后，DLNA 目录里每个视频会多出 `.seek.mp4` 项，
   **原有的 Live 项并存不受影响**（这是 2026-05-28 定下的规矩：实验入口只能 additive）。

2. GUI 启动服务器（不要用 `run_server.bat`，配置来自 UI）。

3. 先跑判定，确认素材适合：

   ```bash
   uv run python -m tools.seek_source_budget_probe --layout "videos/2_2.mp4"
   ```

4. **先测静态快照**：DLNA 目录里播放 `VMP4_SNAPSHOT_dance4K.mp4`（走普通 `/media` 静态 Range）。
   这是我们生成的虚拟 MP4 的一次冻结产物，内容和结构与实时项完全一致，只是不涉及实时生成。
   - 有进度条、能拖动 → 容器和内容播放器认，问题只可能在实时生成/并发那一侧。
   - 不能播/不能拖 → **不要再测实时项**，先解决容器/内容问题（这一步 ffmpeg 已确认无误，
     所以失败就说明是播放器对 8K/HEVC/分辨率/filler 的具体限制，需要按播放器逐个排查）。
   不需要时直接删掉这个文件即可。

5. 在播放器里选 `.seek.mp4` 项。**先用 4K 素材**，理由不只是编码快：
   本机 `PT_DECODE_MAX_SIDE=0`，8K 源输出仍是 8K，而源预算给的码率约是当前 live 的一半
   （SI_TEST_8K：19.6 Mbps 对 live 的 39.2 Mbps），8K 下画质压力明显；4K 素材没有这个问题。
   如果播放器目录里看不到新增的 `.seek.mp4` 项，在播放器里刷新/重连一次 DLNA 服务器
   （客户端会缓存 Browse 结果）。

6. 日志观察点（`debug_output/server.log`）：

   - `PyNv frame stream begin` —— 每次会话/seek 一条。**如果每 ~3s 刷屏，说明并发抖动没解决**（§31）。
   - `frames probe (init|tail|header-crossing)` —— 播放器的结构性探测被正确直接服务。
   - `VMP4 frames gop not ready` —— 只应出现在真实播放追不上编码时。
   - `frames oversize ... (clamped)` —— 偶发正常；持续大量出现说明预算太紧，调大
     `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_FLOOR_BYTES` 或关掉源预算回到 flat。

7. 响应头核对点：

   - `X-Passthrough-VMP4-Frames-Source-Budget: 1`
   - `X-Passthrough-VMP4-Layout-Size` == 源文件字节数
   - `X-Passthrough-VMP4-Frames-Budget-Range: min-max/mean`
   - `X-Passthrough-VMP4-Frames-Probe`（探测请求上）

8. 回退：把 `passthrough_seek_enabled` 改回 false，或只关源预算
   `PT_PASSTHROUGH_SEEK_VMP4_FRAMES_SOURCE_BUDGET=0` 回到 flat 256 KiB 预算。
   Live 路径任何时候都不受影响。
