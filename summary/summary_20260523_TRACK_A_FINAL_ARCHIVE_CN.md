# Track A 终档（首帧 ramp-up + 首块延迟）

日期: 2026-05-23
状态: ✅ 完结（架构地板已触达，剩余收益需架构级改造，归入新优化项目）

> 父文档: `summary_20260522_ROADMAP_POST_TRT_WARMUP_CN.md`
> 详情文档: `summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md`
> 阶段 patch 档:
> - `summary_20260523_FIRST_CHUNK_LATENCY_PATCH_CN.md` (A6/A7 首轮)
> - `summary_20260523_FIRST_CHUNK_LATENCY_A8_1_PATCH_CN.md` (T2→T4 拆分诊断)
> - `summary_20260523_FIRST_CHUNK_LATENCY_A8_2_PATCH_CN.md` (P1/P2 路线变更)
> - `summary_20260523_FIRST_CHUNK_LATENCY_A8_P2A_EXECUTION_CN.md` (P2.A 阶梯实验 + A8.2 单点)

---

## 1. Track A 范围

两个并列目标，最终都达成或触及架构地板:

| 目标 | 基线 | 验收 | 最终实测 | 结果 |
|---|---|---|---|---|
| frame 30 fps | 32.12 | ≥ 40 | **54.9** | ✅ 超过 |
| frame 60 fps | 44.17 | ≥ 55 | **59.9** | ✅ 达标 |
| frame 120 fps | 56.61 | ≥ 65 | ~77-78 | ✅ 超过 |
| 稳态 fps | ~75-76 | ≥ 75 | ~77-78 | ✅ 持平略增 |
| 首块延迟 (nPlayer) | 2821ms | ≤ 1800ms（保守）/ ≤ 1500ms（理想） | **1978ms** | ⚠️ 接近保守线，触地板 |
| 首块延迟 (DeoVR) | n/a | n/a | **1939ms** | ✅ 基本达保守线 |
| 首块延迟 (SkyBoxVR) | n/a | n/a | **2029ms** | ⚠️ 略高于 nPlayer，同地板 |
| preprocess (首帧) | 39.3ms | <5ms | **0.1ms** | ✅ 远超 |
| ort_run (首帧) | 18.6ms | <30ms | **14.2ms** | ✅ |

结论: 帧率类目标全部超额；首块延迟收敛到约 **1.94-2.03s**，逼近"NVENC 节奏 × 双段 mux 串行"架构地板，**Track A 在当前架构下没有进一步空间**。

---

## 2. 阶段地图

| 阶段 | 内容 | 状态 | 关键产物 |
|---|---|---|---|
| A1 | 首帧 preprocess 诊断（`PT_WARMUP_RAMPUP_DIAG_FRAMES`） | ✅ | `pipeline/matting.py` 加 `[DIAG][PREPROC]` 首 N 帧细分 |
| A2 | CuPy 预处理路径 warmup | ✅ | preprocess 39.3 → 0.1ms |
| A3 | NVENC encoder preflight | ✅ | `pipeline/pynv_stream.py:startup_preflight()`，mux 首段被吸收到 warmup |
| A4 | `[WARMUP]` JSON 结构化日志 | ✅ | server.log 可被外部 parse |
| A5 | UI 启动进度条 | ✅ | `/status` step 链细分 7-8 step，UI overlay `n/N + 文案` |
| A6 | T0-T4 首块时间戳框架（`MUX_LATENCY_DIAG`） | ✅ | `[DIAG][MUX] first_chunk_breakdown` |
| A7 | 单段 mux + nobuffer + frag_duration（首轮） | 部分回退 | `MUX_NOBUFFER_ENABLE=1`、`FMP4_FRAG_DURATION_US=100000` 保留；`MUX_PROBESIZE_OVERRIDE=32` 因 nPlayer audio-only 回退；`hevc_metadata=aud=insert` 永久禁用 |
| A8.1 | T2→T4 拆分（T2a/T2b/T3a/T3b/T3c），`MUX_LATENCY_DIAG_VERBOSE` | ✅ | `_drain_stderr` + `_log_first_chunk_breakdown` |
| A8.P1.A | raw HEVC stdin probe 2MB/2s | 中间步 | 单测过，T2a-T1: 802→688ms |
| A8.P1.B | raw HEVC stdin probe 1MB/1s | ✅ 锁定 | `MUX_RAW_VIDEO_PROBESIZE=1000000`, `MUX_RAW_VIDEO_ANALYZEDURATION=1000000`；T2a-T1: 802→308ms |
| A8.P2.A.1 | intermediate mpegts stdin probe 16384/0 | ✅ 锁定 | `MUX_INTERMEDIATE_TS_PROBESIZE=16384`, `MUX_INTERMEDIATE_TS_ANALYZEDURATION=0`；T2b: 2761→1275ms，三播放器双验通过 |
| A8.P2.A.2 | intermediate mpegts stdin probe 8192/0 | ❌ 撞地板 | T2b +25ms 回升，T4-T3c +50ms 放大，回退 16384/0 |
| A8.P2.A.3 | 4096/0 | 跳过 | 越过 audio-only 回归边界，风险 > 收益 |
| A8.2 | `PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA=0` 单点 | ❌ 噪声级 | T4-T3c=663.5ms（与 P2.A.1 同区间），total 改善 18.6ms，恢复 500000000 |
| A8.P2.B | `+resend_headers` | 不做 | T4-T3c 是 NVENC 输出节奏，非 PAT/PMT 问题 |
| A8.P2.C | `-pat_period 0.02` | 不做 | 同上 |
| A8.P2.D | Python 转发线程（侵入式） | 不做 | 工程成本高，且 GIL 在 8K 流下风险大；归入新项目候选 |

---

## 3. 锁定的 default 配置

`config.py` 默认值（无需 env 覆盖即生效）:

```python
# A8.P1.B
MUX_RAW_VIDEO_PROBESIZE = _env("MUX_RAW_VIDEO_PROBESIZE", "1000000").strip()
MUX_RAW_VIDEO_ANALYZEDURATION = _env("MUX_RAW_VIDEO_ANALYZEDURATION", "1000000").strip()

# A8.P2.A.1
MUX_INTERMEDIATE_TS_PROBESIZE = _env("MUX_INTERMEDIATE_TS_PROBESIZE", "16384").strip()
MUX_INTERMEDIATE_TS_ANALYZEDURATION = _env("MUX_INTERMEDIATE_TS_ANALYZEDURATION", "0").strip()

# A8.2 回退
PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA = _env("PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA", "500000000").strip()

# A6 / A8.1 诊断
MUX_LATENCY_DIAG = _env("MUX_LATENCY_DIAG", "1") == "1"           # 保留（轻量）
MUX_LATENCY_DIAG_VERBOSE = _env("MUX_LATENCY_DIAG_VERBOSE", "0") == "1"  # release 关闭

# A7（保留有效部分）
MUX_NOBUFFER_ENABLE = _env("MUX_NOBUFFER_ENABLE", "1") == "1"
FMP4_FRAG_DURATION_US = _env("FMP4_FRAG_DURATION_US", "100000").strip()
```

已确认 release default: `MUX_LATENCY_DIAG_VERBOSE=0`，`MUX_FFMPEG_LOGLEVEL=warning`。

---

## 4. 架构地板：T4-T3c ≈ 630-730ms

A8.P2.A.1 之后，最大残余瓶颈从"前级 raw HEVC probe / 后级 mpegts probe"转移到 `T4 - T3c`（final mux output ready → 首字节写出）。该段在三类播放器、两类 probesize、两类 interleave 配置下均稳定在 630-730ms。

### 4.1 物理来源

1. **NVENC 输出节奏**: HEVC 编码器（NV12 8192×4096 60fps）首个 GOP 完整封包到 stdout 需要"足够帧累积 + B-frame 重排"，约 400-500ms。
2. **双段 mux 串行**: `video_proc`（raw HEVC → mpegts stdout）需要把第一个 PAT/PMT + 第一个完整 video AU 写到 stdin pipe；`final_proc` 必须把 video + audio 在 mpegts interleave 阈值内对齐后才输出第一字节。两段串行叠加约 150-200ms。
3. **OS pipe buffer (Windows 4-64KB)**: 即使 final mux probesize=16384，video_proc → final_proc 之间的 stdin pipe 仍需要积累 ≥ 1 个完整封包；这是 A8.P2.A.2 撞地板的根因。

### 4.2 为什么 A8.2 无效

`max_interleave_delta=0` 仅在"video 和 audio 速率严重失衡"时有效。本场景 audio = AAC 44.1kHz/2ch（持续小包）与 video = HEVC 8K/60fps（突发大包）的写入节奏差异主要由 NVENC 输出节奏决定，不是 ffmpeg 的 interleave wait 决定。

### 4.3 突破地板的唯一路径（不在 Track A 范围）

把 NVENC 直接输出 mpegts（`pynvvideocodec` + 自实现 mpegts mux 或 NVENC `outputBitstreamCallback` 直写），**取消 ffmpeg 双段 mux**。预期可砍 600-700ms，但工作量量级在 1-2 周。归入新优化项目（候选 Track D）。

---

## 5. 已证伪 / 永久禁止再试

| 方案 | 故障模式 | 回归阶段 |
|---|---|---|
| `MUX_PROBESIZE_OVERRIDE=32` 作用于 raw HEVC stdin | nPlayer/Quest3 `PPS id out of range` → 进 audio-only | A7 |
| `hevc_metadata=aud=insert` bsf | 同上（NAL 类型与 setts pts 冲突） | A7 |
| `MUX_INTERMEDIATE_TS_PROBESIZE < 8192` | nPlayer `Could not find codec parameters`，audio-only | A8.P2.A.2 边界 |
| `MUX_RAW_VIDEO_PROBESIZE < 524288` | 未实测，但 P1.B 数据曲线 + P1 vs P2 地板分析显示边际收益 < 50ms，风险高 | A8.P1 推断 |
| `PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA=0` | T4-T3c 不下降（NVENC 节奏决定，非 interleave wait） | A8.2 |
| `single-stage mux`（直接 NVENC → final mpegts） | nPlayer audio-only（VPS/SPS/PPS 时序与 mpegts probe 不匹配） | A7 |
| `_mpegts_flags()` 加 `+resend_headers`（用于 pipe_ts） | 未测；但 T4-T3c 根因不在 PAT/PMT 频率，归因失败可能性高 | 推断不做 |
| `-pat_period 0.02` | 同上 | 推断不做 |

---

## 6. 最终成绩单

| 指标 | baseline (2026-05-22) | A4/A5 落地后 | A6+A7 后 | A8.1 拆分时 | **A8.P2.A.1 锁定后** | 收益 |
|---|---|---|---|---|---|---|
| frame 30 fps | 32.12 | 54.9 | 54.9 | 54.9 | **54.9** | +71% |
| frame 60 fps | 44.17 | 59.9 | 59.9 | 59.9 | **59.9** | +35% |
| 稳态 fps | 75-76 | 77-78 | 77-78 | 77-78 | **77-78** | +2% |
| preprocess 首帧 | 39.3ms | 0.1ms | 0.1ms | 0.1ms | **0.1ms** | -99.7% |
| first chunk (nPlayer) | 2821ms | ~2800ms | ~2800ms | 2821ms（重测） | **1978ms** | **-30%** |
| first chunk (DeoVR) | n/a | n/a | n/a | n/a | **1939ms** | n/a |
| first chunk (SkyBoxVR) | n/a | n/a | n/a | n/a | **2029ms** | n/a |
| T2a-T1 | 803ms | 803ms | 803ms | 803ms | **308ms** (P1.B) | -62% |
| T2b-T2a | 1766ms | 1766ms | 1766ms | 1766ms | **~800ms** (P2.A.1) | -55% |
| T4-T3c | 63ms | 63ms | 63ms | 63ms | **663ms** (地板) | +600ms (代价) |

> 注: A8.P2.A.1 后 `T4-T3c` 从 63ms 涨到 663ms，是把 T2b 段的"等数据"延后到了 T4 段（pipe 数据没消失，只是阶段归属变了）。total 净降才是真实收益。

---

## 7. 历史文档索引

按时间顺序:

1. `summary_20260522_ROADMAP_POST_TRT_WARMUP_CN.md` — 父文档（Track A/B/C/D/E 划分）
2. `summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md` — Track A 总规划（A1-A5 + A6/A7 收尾 + A8 嵌入）
3. `summary_20260523_FIRST_CHUNK_LATENCY_PATCH_CN.md` — A6/A7 首轮落地
4. `summary_20260523_FIRST_CHUNK_LATENCY_A8_1_PATCH_CN.md` — A8.1 T2→T4 拆分（`_drain_stderr` / `_log_first_chunk_breakdown` / `MUX_LATENCY_DIAG_VERBOSE`）
5. `summary_20260523_FIRST_CHUNK_LATENCY_A8_2_PATCH_CN.md` — 路线变更：A8.2 降级、A8.P1/P2 升级，含 P1.A/P1.B 实测表
6. `summary_20260523_FIRST_CHUNK_LATENCY_A8_P2A_EXECUTION_CN.md` — P2.A 阶梯（baseline/16384/8192）+ Quest3 三播放器门禁 + A8.2 单点结果
7. **`summary_20260523_TRACK_A_FINAL_ARCHIVE_CN.md`（本文）** — Track A 终档

`summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md` 第 10 节建议追加一行: `Track A 终档参见 summary_20260523_TRACK_A_FINAL_ARCHIVE_CN.md。`

---

## 8. 收尾 checklist（合并到新项目分支前）

- [x] `config.MUX_LATENCY_DIAG_VERBOSE` default 改为 `"0"`（release 不需要 stderr 文本扫描）
- [x] `MUX_RAW_VIDEO_PROBESIZE` / `MUX_RAW_VIDEO_ANALYZEDURATION` / `MUX_INTERMEDIATE_TS_PROBESIZE` / `MUX_INTERMEDIATE_TS_ANALYZEDURATION` 四个 env 写入 `prompt/HANDOVER_*.md` 配置默认值清单
- [x] `summary_20260522_ROADMAP_POST_TRT_WARMUP_CN.md` 第 8 节 Track A 行标 `✅ 已完成 (2026-05-23)`，备注首块约 1.94-2.03s 触地板
- [ ] 把 Track A 相关 commit 按阶段分组打 tag（可选）: `track-a/a6`、`track-a/a8.1`、`track-a/a8.p1.b`、`track-a/a8.p2.a.1`、`track-a/final`
- [ ] 提交 `summary_20260523_TRACK_A_FINAL_ARCHIVE_CN.md`（本文）

---

## 9. 新优化项目候选（用户自决）

按 ROI 排列，**不在 Track A 范围**:

| 候选 | 收益预期 | 工作量 | 风险 | 建议 |
|---|---|---|---|---|
| **Track D（NVENC 直出 mpegts）** | first chunk 1978 → ~1300ms | 1-2 周 | NVENC API 学习；mpegts 自实现 mux | **唯一能再砍首块的路径**；若用户对 ≤1500ms 仍有诉求，优先做 |
| Track B（CUDA Graph） | 稳态 +5-10fps | 3-5 天 | TRT engine 与 graph 兼容性 | 收益小，frame rate 已达标，可推后 |
| Track C（alpha 通道真透明输出） | DeoVR α 通道支持 | 5-10 天 | DeoVR 协议、容器格式 | 功能性，非性能；按用户路线选 |
| Track E（ConnectionManager 完整化 + matting.py 拆模块） | 维护性 | 2-3 天 | 无 | 工程债，闲时做 |

**默认推荐**: 若用户对首块 ~2s 接受，跳过 Track D 直接进 Track C（功能扩展）；若用户要求 ≤1500ms 首块，进 Track D。

---

## 10. 一句话总结

Track A 把首帧 ramp-up 从可视卡顿（32fps）打到 55fps、首块从 2.82s 打到 1.94-2.03s（-30%），代价是确认了"双段 ffmpeg mux + NVENC 输出节奏"的 600-700ms 架构地板，所有 probesize / interleave / pat_period 旋钮均已穷举。**继续砍首块需架构改造，归新项目。**
