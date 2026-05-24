# A8.2 / A8.P series — First-chunk pipe-waiting decomposition

日期: 2026-05-23
所属轨道: Track A / 阶段 A8
依据数据: A8.1 落地后首组 breakdown（场景: pipe_ts / alpha / nPlayer / loglevel=info）

---

## 0. 数据快照（开发方反馈）

```
T1_write   = 182.1ms
T2a_video  = 984.7ms     -> (T2a - T1) = 802.6ms  ← 前级 raw HEVC probe
T2b_final  = 2750.5ms    -> (T2b - T2a) = 1765.8ms ← 前级 stdout 到 final mux stderr 首行
T3a_vcodec = 2751.8ms    -> (T3a - T2b) = 1.3ms   ← final mux 解出 video codec
T3b_acodec = 2754.0ms    -> (T3b - T3a) = 2.2ms   ← final mux 解出 audio codec
T3c_output = 2758.2ms    -> (T3c - T3b) = 4.2ms   ← final mux output ready
T4_reader  = 2821.3ms    -> (T4 - T3c) = 63.1ms   ← header 写到 stdout 首字节
total      = 2821.3ms
```

无 `PPS id out of range`、无 `Could not find codec parameters`、播放链路正常。

## 1. 决策更新

旧 A8 排序假设（A8.2 = max_interleave_delta，A8.3 = AAC 预热）在 A8.1 初始数据下曾降级；P2.A 收敛后需要重新排序：

| 阶段 | 原假设 | 实测占比 | 修正 |
|---|---|---|---|
| A8.2 max_interleave_delta | "T4-T3c 较大时" | 初始 **63ms / 2821ms ≈ 2.2%**；P2.A 后 **630-730ms / ~2s** | **升级**：P2.A 后的新最大瓶颈，先做单点实验 |
| A8.3 AAC cache 预热 | "audio probe 慢时" | **2.2ms** | **取消**：已极快 |
| A8.4 fMP4 alpha 评估 | 容器格式回探 | n/a | 维持，独立轨 |
| **新 A8.P1** | 前级 raw HEVC probe 收敛 | **802.6ms / 2821ms ≈ 28.4%** | **第一优先** |
| **新 A8.P2** | 前级→后级 stdin pipe 等待 | **1765.8ms / 2821ms ≈ 62.6%** | **最高优先**，但与 P1 强耦合 |

合计 A8.P1+P2 覆盖 **91.1% 的首块延迟黑箱**。

> 注：A8.P1 和 A8.P2 物理上是一条链上前后两段。若 P1 把 T2a 拉到 ~200ms，P2 段在 T2a 上累加的 ~1.8s 也会同步前移（因为 final mux 是 stdin-driven）。因此先做 P1，重测，再判断 P2 是否还需要专门处理。

---

## 2. A8.P1 — Raw HEVC stdin probe 收敛实验

### 2.1 问题模型

`_open_pipe_ts_muxer` / `_open_slate_pipe_ts_muxer` 中 video_proc 当前配置:
- `_mux_probe_args(for_raw_video=True)` 对 raw HEVC 返回 **空**（A7 回退保留，防 nPlayer audio-only）
- 因此 video mux 用 ffmpeg 默认 `probesize=5000000 (5MB) analyzeduration=5000000us (5s)`

8192x4096 HEVC 流的 VPS+SPS+PPS NAL 在每个 IDR 前都会出现，单组通常 **<2KB**。但 raw HEVC demuxer 不知道何时收齐，必须先抓 5MB 或 5s 任一阈值。在 75fps、IDR 间隔较大、首 GOP 数据量大的情形下，**T2a-T1 ≈ 803ms** 与"等 5MB 字节积累"高度一致。

### 2.2 不可踩的雷

历史回归（已确认）：
- `MUX_PROBESIZE_OVERRIDE = "32"` 给 raw HEVC stdin → PPS id out of range → nPlayer/Quest3 自动 fallback 音频模式
- `hevc_metadata=aud=insert` bsf → 同上

> 因此本实验 **必须**:
> 1. 不再触碰 raw HEVC bsf 链（`setts=...:pts=N*tick:dts=N*tick` 保留）
> 2. probesize 不下探到 < 64KB
> 3. 每一步在两类播放器上验证（nPlayer iOS + Quest3 DeoVR）

### 2.3 渐进试探表（A/B 落地）

新增配置项（按需启用，default keep 5MB）:

| env | 默认 | 含义 |
|---|---|---|
| `MUX_RAW_VIDEO_PROBESIZE` | unset / `5000000` | 给 video mux raw HEVC stdin 的 `-probesize`，单位 byte |
| `MUX_RAW_VIDEO_ANALYZEDURATION` | unset / `5000000` | 给 video mux raw HEVC stdin 的 `-analyzeduration`，单位 us |

落地点（**只动一处**）: `pipeline/pynv_stream.py` 中 `_mux_probe_args(for_raw_video=True)` 分支，从"返回空"改为"按 `MUX_RAW_VIDEO_*` 注入 `-probesize`/`-analyzeduration`，未设置时仍返回空"。

实验序列（每个 step 完整跑一次冷启动+首播+nPlayer+Quest3）:

| step | probesize | analyzeduration | 预期 T2a-T1 | 验收 |
|---|---|---|---|---|
| baseline | (unset) | (unset) | ~800ms | 当前数据 |
| P1.A | 2000000 (2MB) | 2000000 (2s) | < 500ms? | 视频模式 + 无 PPS 错 |
| P1.B | 1000000 (1MB) | 1000000 (1s) | < 300ms? | 视频模式 + 无 PPS 错 |
| P1.C | 524288 (512KB) | 500000 (500ms) | < 200ms? | 视频模式 + 无 PPS 错 |
| P1.D | 262144 (256KB) | 200000 (200ms) | < 100ms? | 视频模式 + 无 PPS 错 |
| P1.E | 131072 (128KB) | 100000 (100ms) | < 50ms? | 视频模式 + 无 PPS 错 |
| P1.STOP | < 64KB | < 50000 | n/a | **不再试**（回归边界） |

**首次出现以下任一现象就停止并回退到上一步**:
- `PPS id ... out of range`
- `Could not find codec parameters`
- `Video: hevc ..., none` 或缺 `8192x4096`
- nPlayer 进音频模式
- Quest3 DeoVR 黑屏 / 无画面

最优值（最小且通过验收）落库为 `MUX_RAW_VIDEO_PROBESIZE` 的 default。

### 2.4 旁路保险

若 P1.A 已回归，说明 raw HEVC 头识别在该流上对 probesize 极敏感，则:
- 维持 baseline
- 把瓶颈交给 A8.P3（输出侧加速），见 §4

---

## 3. A8.P2 — 前级→后级 pipe 等待

`(T2b - T2a) = 1765.8ms` 的可能成因（与 P1 强相关，但仍有独立部分）:

1. **A**: video mux 在 T2a 出 stderr 时（自身 Input 解析完成）尚未开始大量写 stdout；FFmpeg 内部 input→output 衔接还在 buffer/flush。
2. **B**: video mux 输出 mpegts 后，OS pipe (`video_proc.stdout` → `final_proc.stdin`) 在 stdin 端积累到 32KB（`MUX_CONTAINER_PROBESIZE_OVERRIDE`）才让 final mux 解析。
3. **C**: final mux 在出第一条 stderr 之前要先完成 `avformat_find_stream_info()`，这一步要至少 1 个完整 PMT + 1 个 video access unit；首 IDR 数据量大（8K 帧可能 MB 级），积累慢。

P1 改善后预期 P2 同步下降；如果 P1 把 T2a 从 985ms 拉到 200ms，但 T2b 仍 > 2200ms，说明 P2 段独立成本 > 1.5s，需要另外做实验:

### 3.1 P2 子实验（仅在 P1 落地+重测后启用）

| 实验 | 改动 | 预期 |
|---|---|---|
| P2.A | `MUX_CONTAINER_PROBESIZE_OVERRIDE` 32768 → **8192** | T2b - T2a 下降 |
| P2.B | video mux 输出加 `-mpegts_flags +resend_headers`（每包重发 PAT/PMT） | T2b 提前（final mux 更早收到完整 PAT/PMT） |
| P2.C | video mux `-pat_period 0.1` → **0.02** | 同上 |
| P2.D | 把 video mux stdout 改用 Python 转发线程 + small chunk write 到 final mux stdin（绕 OS pipe buffer） | 仅在 P2.A-C 都不解时考虑（侵入大） |

> P2.B/P2.C 要再验 nPlayer/Quest3 长播稳定性（PAT 太频会被某些 PS3/电视拒绝，但 nPlayer/Quest3 通常容忍）。

---

## 4. A8.P3 — 输出侧加速（baseline 保险）

不论 P1/P2 是否完全成功，video mux 已有的输出侧参数都已较激进:
- `-flush_packets 1`
- `-muxdelay 0 -muxpreload 0`
- `-pat_period 0.1 -pcr_period 20`
- `-mpegts_flags <由 _mpegts_flags() 决定>`

P3 任务（**仅在 P1/P2 全部受阻时**）:
- 确认 `_mpegts_flags()` 当前实际值；如未含 `+nobuffer`/`+latm`，评估是否加
- 评估 `-fflags +flush_packets`（输入侧）是否已加（A7.3 应已生效）

P3 不涉及业务参数，只是确认 baseline 没倒退。

---

## 5. A8.2（max_interleave_delta）现状

P2.A 收敛后结论已改变。P2.A.1 `16384/0` 把 `T2b` 从约 2760ms 压到约 1275ms，但 `T4 - T3c` 从个位数/几十毫秒级暴涨到 `630-730ms`：

| 场景 | T2b | T3c | T4 | T4-T3c | total |
|---|---:|---:|---:|---:|---:|
| nPlayer / P2.A.1 | 1275.1 | 1300.4 | 1978.4 | 678.0 | 1978.4 |
| DeoVR / P2.A.1 | 1287.2 | 1306.8 | 1939.1 | 632.3 | 1939.1 |
| SkyBoxVR / P2.A.1 | 1350.2 | 1368.8 | 2028.9 | 660.1 | 2028.9 |
| nPlayer / P2.A.2 | 1300.5 | 1327.5 | 2056.1 | 728.6 | 2056.1 |
| nPlayer / A8.2 (`max_interleave_delta=0`) | 1268.2 | 1296.3 | 1959.8 | 663.5 | 1959.8 |

**结论更新**: A8.2 从低优先级升级为下一优先级。它直接针对 `Output #0 ready -> reader first stdout chunk` 的等待段。P2.B/P2.C 暂缓，因为 `+resend_headers` / `pat_period` 更可能影响 PAT/PMT probe 阶段，而 `T2b` 已经被 P2.A 打到地板。

当前实验安排:
- P2.A 最终候选保持 `MUX_INTERMEDIATE_TS_PROBESIZE=16384` / `MUX_INTERMEDIATE_TS_ANALYZEDURATION=0`。
- 临时把 `PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA` 默认设为 `0`，只跑一次 A8.2 单点。
- 若 `T4-T3c < 100ms` 且无音画同步/黑屏/audio-only 回归，则 A8.2 有效，再决定是否锁定 `0` 或保守锁 `100000`。
- 若 `T4-T3c` 不动或出现回归，恢复 `500000000`，接受约 2s first chunk 收尾。

历史提醒: 2026-05-09 曾记录 `-max_interleave_delta 0` 在旧 AAC live mux 路径存在死锁风险。本次只在 P2.A 已收敛、双段 pipe_ts 路径下单点验证，不与 P2.B/P2.C 叠加。

### 5.1 A8.2 单点结果

2026-05-23 19:09 nPlayer 验证:
- 命令已确认包含 `-probesize 16384 -analyzeduration 0` 与 `-max_interleave_delta 0`。
- `T4-T3c = 663.5ms`，仍在 P2.A.1 的 `630-730ms` 区间内，未接近 `<100ms` 目标。
- `total=1959.8ms`，相比 P2.A.1 nPlayer `1978.4ms` 仅快 `18.6ms`，属于噪声级。
- 未见 codec 参数失败、audio-only、PPS、Broken pipe、Traceback 或 reader-waiting 回归。

**判定**: A8.2 对当前 `T4-T3c` 瓶颈无实质收益，恢复 `PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA=500000000`。不把 `0` 锁为 default。

---

## 6. 推荐执行顺序

1. **A8.P1** 阶梯实验（P1.A → P1.E），找到最小可用 `MUX_RAW_VIDEO_PROBESIZE`
   - 工作量：单点改动 `_mux_probe_args`，4-5 轮压测
   - 风险：中（依赖播放器验证）
   - 预期收益：T2a-T1 降到 100-200ms，整体首块 -500~700ms
2. **重测** A8.1 breakdown，更新数据
3. 若 T2b - T2a 仍 > 500ms → **A8.P2** 子实验（先 P2.A，再 P2.B/C）
4. 若 1+3 已把总 first chunk 拉到 < 1s → **进入 A4** (JSON 日志) / **A5** (UI 进度条)
5. 最终 polish 阶段评估 **A8.2** max_interleave_delta

---

## 7. 验收（A8.P1 完工时）

- [ ] `config.py` 新增 `MUX_RAW_VIDEO_PROBESIZE` / `MUX_RAW_VIDEO_ANALYZEDURATION`
- [ ] `_mux_probe_args(for_raw_video=True)` 按新配置注入参数；未配置时维持空（不回归）
- [ ] 默认值为实验找到的最小通过点（建议保守，比最小通过点大 2x）
- [ ] nPlayer iOS 视频模式播放正常
- [ ] Quest3 DeoVR 视频模式播放正常
- [ ] 8K 高码率源 cold start first_chunk total < 1800ms（保守目标）
- [ ] 实验过程数据（每 step 的 T2a/T2b/total + 是否回归）回填到本文档附录

---

## 8. 不要做的事

- ❌ 不要再给 raw HEVC stdin 加 bsf
- ❌ 不要把 `MUX_RAW_VIDEO_PROBESIZE` 设到 < 65536
- ❌ 不要为了赶进度跳过 nPlayer/Quest3 双验（仅 web client 通过不算数）
- ❌ 不要把 final mux stdin probesize 改到 8192 之前先做 P2.A 单点对照（容易和 P1 互相干扰，无法归因）
- ❌ 不要顺手开 `-flags low_delay`：在 HEVC copy + mpegts 路径下没有实际意义，且不同 ffmpeg 版本语义不一

---

## 9. 附录 — 实测数据与决策更新（开发持续回填）

### 9.1 实测表

| step | probesize | analyzeduration | T1 | T2a | T2b | T3a | T3b | T3c | T4 | total | nPlayer | Quest3 | T2a-T1 | T2b-T2a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | (unset=5M) | (unset=5s) | 182.1 | 984.7 | 2750.5 | 2751.8 | 2754.0 | 2758.2 | 2821.3 | 2821.3 | video | (n/a) | 802.6 | 1765.8 |
| **P1.A** | 2000000 | 2000000 | 157.6 | 845.7 | 2671.1 | 2673.3 | 2675.7 | 2680.0 | 2689.9 | 2690.0 | video | video | **688.1** (-114) | **1825.4** (+59) |
| **P1.B** | 1000000 | 1000000 | 173.8 | 481.7 | 2645.0 | 2646.0 | 2649.2 | 2654.2 | 2664.4 | 2664.4 | video | video | **307.9** (-494) | **2163.3** (+397) |
| P1.C | 524288 | 500000 | | | | | | | | | | | | |
| P1.D | 262144 | 200000 | | | | | | | | | | | | |
| P1.E | 131072 | 100000 | | | | | | | | | | | | |

### 9.2 P1.A / P1.B 解读

| 指标 | baseline | P1.A | P1.B | P1.A→B Δ | baseline→B Δ |
|---|---|---|---|---|---|
| T2a-T1 | 802.6 | 688.1 | 307.9 | -380 (-55%) | **-495 (-62%)** |
| T2b 绝对值 | 2750.5 | 2671.1 | 2645.0 | -26 | -106 (-3.8%) |
| T2b-T2a | 1765.8 | 1825.4 | 2163.3 | +338 | +397 |
| total | 2821.3 | 2690.0 | 2664.4 | -26 | -157 (-5.6%) |
| T4-T3c | 63.1 | 10 (approx) | 10.2 | 0 | -53 |

关键观察:
- **T2b 绝对值几乎"地板锁定"在 ~2.6s**（baseline → P1.B 仅减 106ms）。
- P1 把 T2a 提前 495ms，但 T2b 几乎不前移；结果 T2b-T2a 从 1765 → 2163 (+397ms)。
- 这证明 **T2b 不是 "T2a + 某常数"**，而是 **"final mux stdin 积累到 32768 字节才出 stderr 首行"** 这种独立事件。
- final mux 自身解析仍极快（T3a-T2b=1ms, T3b-T3a=3.2ms, T3c-T3b=5ms, T4-T3c=10.2ms）。

**重要修正升级**: P1 段的所有收益（即使全部压到 T2a-T1=50ms）都会被 T2b 这个"地板"吞掉。继续做 P1.C/D/E 性价比下降。**P2.A 已升级为必须项**。

### 9.3 收益预估更新（基于 P1.A→P1.B 实测）

- 即使 P1.E 把 T2a-T1 压到 50ms，total 也只会从 2664 → ~2400ms 左右（**仍远 > 1800ms 验收**）
- T2b 地板的根因几乎确定是 final mux 32768 字节 mpegts probe + OS pipe buffer 等待
- **真正能砍掉 1s+ 的只有 P2.A**（probesize 下界 → 4096）

### 9.4 推荐继续动作（路线调整）

> **变更**: 暂停 P1 阶梯（P1.B 已是可用点），优先进 P2.A。

1. **P1 暂停在 P1.B**: 默认锁定 `MUX_RAW_VIDEO_PROBESIZE=1000000` / `MUX_RAW_VIDEO_ANALYZEDURATION=1000000`（P1.B 已通过 nPlayer + Quest3 双验）。
2. **进入 P2.A**: 改 `MUX_CONTAINER_PROBESIZE_OVERRIDE` 32768 → 16384 → 8192 → 4096（阶梯），观测 T2b 绝对值下降幅度。每步双验播放器。
3. P2.A 完成后再决定:
   - 若 total < 1500ms → 直接进入 A4/A5（不再做 P1.C-E）
   - 若 total ∈ [1500, 1800ms] → 视 ROI 决定是否回头补 P1.C
   - 若 total > 1800ms → 继续 P2.B (`+resend_headers`) → P2.C (`-pat_period 0.02`)
4. P1.C-E 仅在 P2 全部用尽且 total 仍 > 1500ms 时回头补充。

---

## 10. A8.P2 实验执行细节（待 P1 完成后启用）

### 10.1 P2.A — final mux mpegts stdin probesize 收敛【已升为必须项】

**改动定位**: `_open_pipe_ts_muxer` (line ~1339) 的 `_mux_intermediate_ts_probe_args()` 当前实际值返回 32768；该函数应受 `MUX_CONTAINER_PROBESIZE_OVERRIDE` 控制（请开发同学先 Read 该函数实现，确认环境变量旋钮链路通畅，否则需补一行注入）。

**做法**:
- 阶梯: 32768 (baseline) → 16384 → 8192 → 4096
- 每步观测的关键指标是 **T2b 绝对值**（而非 T2b-T2a 相对值，因 T2a 已被 P1.B 推前）
- **下界 4096**: 一个 mpegts packet = 188B，4096 ≈ 21 packets，结合 video mux `-pat_period 0.1`（100ms 一次 PAT），4KB 内 100% 包含 PAT/PMT + 至少 1 个 video AU 头。
- 不要 < 4096（PAT/PMT 节奏即使 100ms 也要 ~376B + video header）。

**预期**:
- 若每步线性下降，T2b 从 2645 → ~1800 → ~1200 → ~600ms
- 实际很可能在某一步遇到"OS pipe buffer 地板"（Windows pipe default 4-64KB），届时 T2b 不再随 probesize 下降，此时该 step 即为最优

**验收**: T2b 绝对值**实际下降** > 200ms 才算该 step 通过。否则视为撞地板，回退上一步。

**风险**: final mux 的 mpegts demuxer 拿不够数据就猜不到 stream layout，可能报 `Could not find codec parameters for stream 0`。若出现立即回退。

**双验**: 每个 step 落地前过一次 nPlayer + Quest3（防 audio-only 风险）。

### 10.2 P2.B — `+resend_headers`

**改动定位**: `_mpegts_flags()` 的返回值字符串中加 `+resend_headers`（如果未含）。

**含义**: 每个 mpegts segment 重发 PAT/PMT。代价：mpegts 流大小 +1-2%。收益：final mux 不必"等下一个 PAT 周期"。

**做法**: 单点改动 + 重测；不需要阶梯。

**验收**: T2b-T2a 下降 > 100ms。

**风险**: 部分老旧硬件解码器对频繁 PAT 敏感（nPlayer/Quest3/DeoVR 不受影响，已知）。

### 10.3 P2.C — `-pat_period 0.02`

**改动定位**: `_open_pipe_ts_muxer` video_cmd 和 final_cmd 的 `-pat_period 0.1` 改为 `0.02`（20ms）。

**含义**: 把 PAT 间隔从 100ms 缩到 20ms，让 final mux 在 stdin 收到 4-21KB 内就能命中 1-2 个完整 PAT。

**做法**: 单点 + 重测。

**验收**: 与 P2.B 类似，T2b-T2a 下降 > 100ms。

**风险**: 流总大小 +1%；DLNA 兼容性几乎无影响。

### 10.4 P2.A/B/C 互斥还是组合？

- 三者机制不同（probesize 下界 vs. header 频率 vs. PAT 频率），**可组合**，但要先**单点验证收益归属**，再叠加。
- 推荐顺序: P2.A → 看数据 → P2.B（如 A 收益小）→ 看数据 → P2.C → 看数据 → 三选最优组合落库。

### 10.5 P2 完工时验收

- [ ] T2b-T2a < 500ms（保守）/ < 300ms（理想）
- [ ] total first chunk < 1800ms（保守）/ < 1500ms（理想）
- [ ] nPlayer / Quest3 视频模式 + 长播 5 分钟不掉流
- [ ] 流量增加 < 5%（用 `bytes/s` 估算）

---

## 11. 不做的事（补充）

- ❌ 不要在 P1 未跑完前就开 P2 实验（数据归因混乱）
- ❌ 不要把 `_mpegts_flags()` 改成清空再加，应该 in-place append `+resend_headers`，保留现有 flag
- ❌ 不要把 `-pat_period` 下探到 < 0.01（10ms 内 PAT 会被某些 sniffer 当作攻击流量）
- ❌ 不要为了凑 < 1000ms 目标而硬塞 single-stage mux（已知 nPlayer 回归）
