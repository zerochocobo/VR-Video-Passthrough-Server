# A8.P2.A 执行计划 — final mux mpegts stdin probesize 阶梯

日期: 2026-05-23
所属轨道: Track A / 阶段 A8.P2
前置: A6 / A8.1 (诊断) + A8.P1.B (raw HEVC probe 收敛到 1MB / 1s)
状态: WAITING_FOR_IMPLEMENTATION
依赖: 无（A4/A5 已完工，本阶段独立）

---

## 0. 当前状态盘点

### 0.1 已锁定 default

| 配置 | 值 | 来源 |
|---|---|---|
| `MUX_RAW_VIDEO_PROBESIZE` | 1000000 | A8.P1.B |
| `MUX_RAW_VIDEO_ANALYZEDURATION` | 1000000 | A8.P1.B |
| `MUX_PROBESIZE_OVERRIDE` | 32 (raw HEVC 不走此路径) | A7 残留 |
| `MUX_CONTAINER_PROBESIZE_OVERRIDE` | 32768 | A7 |
| `MUX_AUDIO_PROBESIZE_OVERRIDE` | 32768 | A7 |
| `MUX_ANALYZEDURATION_US` | 0 | A7 |
| `MUX_LATENCY_DIAG` | 1 | A6 |
| `MUX_LATENCY_DIAG_VERBOSE` | 1 | A8.1 |

### 0.2 P1.B 实测

```
T1_write   = 173.8ms
T2a_video  = 481.7ms
T2b_final  = 2645.0ms    ← 当前最大瓶颈
T3a_vcodec = 2646.0ms
T3b_acodec = 2649.2ms
T3c_output = 2654.2ms
T4_reader  = 2664.4ms
total      = 2664.4ms
```

`T2b_final - T2a_video = 2163ms`，是首块延迟剩下的几乎全部黑箱（占 81%）。

### 0.3 代码现状（已确认）

`pipeline/pynv_stream.py:183-187`:

```python
def _mux_intermediate_ts_probe_args() -> list[str]:
    # The pipe_ts final mux must inspect the intermediate TS long enough to see
    # HEVC VPS/SPS/PPS. A low probesize/analyzeduration=0 makes strict players
    # classify the final stream as audio-only.
    return _mux_probe_args("", analyzeduration_us="")
```

**显式置空**（保险），所以现在 final mux mpegts stdin 用 ffmpeg 默认 5MB / 5s。注释中已经记录了 audio-only 回归的历史教训。

调用点（pipe_ts 和 slate pipe_ts 两处）:
- `pynv_stream.py:1375` `_open_pipe_ts_muxer` final_cmd
- `pynv_stream.py:1518` `_open_slate_pipe_ts_muxer` final_cmd

---

## 1. Step 1 — 代码接通（必做，独立 commit）

### 1.1 新增 config

`config.py` 在已有 mux 配置区块（`MUX_PROBESIZE_OVERRIDE` 附近）追加:

```python
# Intermediate mpegts stdin (pipe_ts final mux) probesize/analyzeduration.
# 历史教训 (2026-05-23 A7 回归):
#   过小的值（< 32KB / analyzeduration=0）会让 nPlayer/Quest3 等 strict players
#   把 final stream 当作 audio-only。下界请保守。
# Default 留空 = 沿用 ffmpeg 默认 (5MB / 5s)，与 baseline 行为一致。
# A8.P2.A 阶梯实验使用此旋钮。
MUX_INTERMEDIATE_TS_PROBESIZE = _env("MUX_INTERMEDIATE_TS_PROBESIZE", "").strip()
MUX_INTERMEDIATE_TS_ANALYZEDURATION = _env("MUX_INTERMEDIATE_TS_ANALYZEDURATION", "").strip()
```

### 1.2 改造 `_mux_intermediate_ts_probe_args`

`pipeline/pynv_stream.py:183-187` 改为读 1.1 的两个新 env:

```python
def _mux_intermediate_ts_probe_args() -> list[str]:
    """Intermediate TS probe for pipe_ts final mux stdin.

    历史: 过小的值会让 strict players 把流分类为 audio-only
    (见 summary_20260523_FIRST_CHUNK_LATENCY_PATCH_CN.md A7 回退记录)。
    """
    args: list[str] = []
    if config.MUX_INTERMEDIATE_TS_PROBESIZE:
        args.extend(["-probesize", config.MUX_INTERMEDIATE_TS_PROBESIZE])
    if config.MUX_INTERMEDIATE_TS_ANALYZEDURATION:
        args.extend(["-analyzeduration", config.MUX_INTERMEDIATE_TS_ANALYZEDURATION])
    return args
```

### 1.3 关键约束

- **独立 env**: 不复用 `MUX_CONTAINER_PROBESIZE_OVERRIDE`（后者同时作用于 audio file probe 等其他路径，复用会污染数据归因）。
- **default 留空**: 保持现行 baseline（ffmpeg 5MB / 5s），不破坏未启用实验的部署。
- **保留注释**: 旧函数体中的历史教训注释挪到新版函数 docstring。

### 1.4 验证 env 真正接通

启动一次播放（不设新 env），日志里 `pipe_ts final mux cmd` 行**不应**含 `-probesize` / `-analyzeduration` 在 `-f mpegts -i -` 之前 → 与现状一致。

启动一次播放（`MUX_INTERMEDIATE_TS_PROBESIZE=32768 MUX_INTERMEDIATE_TS_ANALYZEDURATION=0`），日志里 `pipe_ts final mux cmd` 行**应**含 `-probesize 32768 -analyzeduration 0`。**只有在这一步通过后才进入 Step 2**。

### 1.5 单元测试（可选但建议）

在 `tests/test_pynv_mux_latency.py` 增 2 个用例:
1. 不设 env → `_mux_intermediate_ts_probe_args()` 返回 `[]`
2. 设 `MUX_INTERMEDIATE_TS_PROBESIZE=8192` → 返回 `["-probesize", "8192"]`

---

## 2. Step 2 — P2.A 阶梯实验

### 2.1 阶梯

每个 step 是一次完整冷启动 + 首播 + 双验 + 截日志:

| step | MUX_INTERMEDIATE_TS_PROBESIZE | MUX_INTERMEDIATE_TS_ANALYZEDURATION | 预期 T2b |
|---|---|---|---|
| P2.A.0 | (unset, baseline 重测) | (unset) | ~2645ms |
| P2.A.1 | 16384 | 0 | < 2400ms? |
| P2.A.2 | 8192 | 0 | < 1800ms? |
| P2.A.3 | 4096 | 0 | < 1200ms? |
| P2.A.STOP | < 4096 | — | **不做**（高风险撞 audio-only 回归） |

> `analyzeduration=0` 是显式设最小值；ffmpeg 文档解释 0 = "no analyze duration"，与 probesize 单独发挥作用。

### 2.2 测试场景（每个 step 必须全部跑）

1. **场景 A**: pipe_ts / alpha / nPlayer iOS（同 P1 验收口径）
2. **场景 B**: pipe_ts / alpha / Quest3 DeoVR
3. **场景 C** (smoke)：8K HEVC 长视频从 0s 启动，确认前 30s 不掉流

只要任一场景出现以下任一现象，**立即停止并回退到上一通过 step**:
- `Could not find codec parameters for stream 0`
- final mux 日志中 `Video: hevc ..., none`（无分辨率）
- nPlayer 进音频模式（黑屏 + 仅声音）
- Quest3 DeoVR 黑屏 / 仅声音
- 长播 30s 内画面冻结或卡顿

### 2.3 关键观测指标

**主指标**: `T2b_final` 绝对值（取自 `first_chunk_breakdown` 日志）。
**辅指标**: `total`、`T4_reader`、播放器表现。

不再看 `T2b-T2a` 相对值（P1.B 后 T2a 已经被推前，相对值会误导）。

### 2.4 单点判定

每个 step 跑完后:
- `T2b` 比上一通过 step **下降 > 200ms** → 该 step 通过，继续下探
- `T2b` 下降 ≤ 200ms（撞地板，OS pipe buffer 限制）→ **该 step 即为最优**，停止下探
- 任一回归现象出现 → 回退到上一通过 step，停止

---

## 3. Step 3 — 决策树（P2.A 跑完后）

按 P2.A 最优 step 的 total 值分流:

### 3.A `total < 1500ms` （理想）

1. 锁定 default: 把最优 step 的两个 env 值写入 `.env.example` / 部署文档 / `config.py` default
2. **关闭 verbose diag**: `MUX_LATENCY_DIAG_VERBOSE=0`（保留 A6 基础 T0-T4 即可）
3. Track A 进入收尾（见 §5）
4. 不做 P2.B / P2.C / P1.C-E

### 3.B `total ∈ [1500ms, 1800ms]` （可接受）

1. 锁定 P2.A 最优 step 的 default
2. **可选**: 试一次 P2.B (`+resend_headers`) 单点，看 total 能否再降到 < 1500ms
3. 若 P2.B 收益 < 100ms → 收尾
4. 若 P2.B 收益 > 100ms → 再叠加 P2.C 一次
5. 不补 P1.C-E（边际收益小）

### 3.C `total > 1800ms` （仍未达标）

按顺序，每步一次冷启动 + 双验:

1. **P2.B**: `_mpegts_flags()` in-place append `+resend_headers`，重测
2. **P2.C**: `-pat_period 0.1` → `0.02`（video_cmd 和 final_cmd 各一处），重测
3. 三者最优组合 + P2.A 最优 step 一起锁库
4. 若组合后仍 > 1800ms → 启动 **P2.D 评估**（见 §6，侵入式）

---

## 4. 数据回填模板

### 4.1 阶梯表（开发每步完成后填）

| step | probesize | analyze | T1 | T2a | T2b | T3a | T3b | T3c | T4 | total | nPlayer | Quest3 | ΔT2b vs 上一step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (P1.B) | (unset) | (unset) | 173.8 | 481.7 | 2645.0 | 2646.0 | 2649.2 | 2654.2 | 2664.4 | 2664.4 | video | video | — |
| P2.A.0 (重测) | (unset) | (unset) | 164.0 | 478.4 | 2760.7 | 2763.7 | 2767.8 | 2768.7 | 2778.0 | 2778.0 | video | 未测 | +115.7 |
| P2.A.1 | 16384 | 0 | 171.4 | 473.9 | 1275.1 | 1291.1 | 1294.9 | 1300.4 | 1978.4 | 1978.4 | video | video | -1485.6 |
| P2.A.2 | 8192 | 0 | 132.6 | 426.8 | 1300.5 | 1318.4 | 1322.8 | 1327.5 | 2056.1 | 2056.1 | video | 未测 | +25.4 |
| P2.A.3 | 4096 | 0 |  |  |  |  |  |  |  |  |  |  |  |

P2.A.0 重测说明: 2026-05-23 16:58 的 `pipe_ts final mux cmd` 在第一路 `-f mpegts -i -` 前仍无 `-probesize/-analyzeduration`，所以该轮是 baseline 重测，不是 P2.A.1。

P2.A.1 说明: 2026-05-23 17:17 的 `pipe_ts final mux cmd` 已确认在第一路 `-f mpegts -i -` 前带上 `-probesize 16384 -analyzeduration 0`。final mux 正常识别 `Video: hevc ... 8192x4096`，未见 `Could not find codec parameters` / audio-only 相关错误。`T2b` 降幅显著，按规则继续下探 P2.A.2。

P2.A.1 二阶现象: `T2b` 相比 P2.A.0 下降 `1485.6ms`，但 `total` 只下降 `799.6ms`，差额主要转移到 `T4_reader - T3c_output`，该段从 P2.A.0 的 `9.3ms` 增至 P2.A.1 的 `678.0ms`。后续 P2.A.2 不能只看 `T2b`，必须同时观察 `T4 - T3c` 是否继续放大。

P2.A.1 Quest3 门禁: P2.A.2 前必须先在 Quest3 / DeoVR 上验证 `16384/0`。通过则 P2.A.1 可作为保底验收点；若黑屏、audio-only 或 codec 参数错误，立即回退到 P2.A.0 baseline，不进入 P2.A.2。

P2.A.1 Quest3 门禁结果: 2026-05-23 17:37-17:38 已用同一台 Quest3 验证 DeoVR 与 SkyBoxVR，均走 `-probesize 16384 -analyzeduration 0`，均无 `Could not find codec parameters` / audio-only / PPS / Broken pipe / Traceback。DeoVR UA 为 `AVProMobileVideo/15.6.3755 ... ExoPlayerLib/1.4.1`，`total=1939.1ms`，`T4-T3c=632.3ms`；SkyBoxVR UA 为 `SKYBOX/2.0.2`，`total=2028.9ms`，`T4-T3c=660.1ms`。P2.A.1 作为保底验收点通过，允许进入 P2.A.2。

P2.A.2 nPlayer 结果: 2026-05-23 17:48 已确认 `-probesize 8192 -analyzeduration 0` 生效。final mux 正常识别 `Video: hevc ... 8192x4096`，未见 `Could not find codec parameters` / audio-only / PPS / Broken pipe / Traceback。但 `T2b=1300.5ms`，相对 P2.A.1 nPlayer 的 `1275.1ms` 不降反升 `25.4ms`；`total=2056.1ms`，`T4-T3c=728.6ms` 继续放大。按 §2.4 / P2.A.2 决策，判定 8192 已撞地板且无收益，停止下探，回到 `16384/0` 作为 P2.A 最终默认候选。不做 P2.A.3 `4096/0`。

后续 A8.2: P2.A 已停止下探，剩余最大瓶颈转为 `T4-T3c=630-730ms`。下一步只单测 `PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA=0`，观察 `T4-T3c` 是否降到 `<100ms`；不叠加 P2.B/P2.C，避免污染归因。

A8.2 单点结果: 2026-05-23 19:09 nPlayer 已确认 `-max_interleave_delta 0` 生效，但 `T4-T3c=663.5ms`，仍处于 P2.A.1 的 `630-730ms` 区间，`total=1959.8ms` 仅比 P2.A.1 nPlayer 快 `18.6ms`。判定无实质收益，恢复 `PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA=500000000`，不锁定 `0`。

P2.A.2 决策:
- 若 `T2b < 800ms` 且 `total < 1500ms`，停在 `8192/0`，进入 §5 收尾，不强测 4096。
- 若 `T2b` 在 `800-1100ms` 且 `total` 在 `1500-1800ms`，结合 Quest3 双验决定 `8192/0` 或 `16384/0` 作为 default。
- 若 `T2b > 1100ms` 或相对 P2.A.1 下降小于 `200ms`，判定撞地板，停在 `16384/0` 收尾。
- 若出现 codec error、audio-only、黑屏或冻结，回退到 `16384/0` 收尾。

P2.A.3 态度: 默认不做。仅当 P2.A.2 仍线性大幅下降（`ΔT2b > 400ms`）且 `total > 1500ms` 时才考虑 `4096/0`。否则 4096 的 audio-only 回归风险高于收益。

### 4.2 关键日志行

```
[PYNV][sid] pipe_ts final mux cmd: ... -probesize XXXX -analyzeduration 0 ... -f mpegts -i - ...
[DIAG][MUX][sid] mark key=T2b_final_first_stderr delta_from_T0_ms=XXXX.X
[DIAG][MUX][sid] first_chunk_breakdown ... T2b_final=XXXX.X ... total=XXXX.X
```

### 4.3 撞地板判定

当某 step 出现 ΔT2b vs 上一step < 200ms 时，记入"撞地板判定"小节:

```
撞地板 step: P2.A.?  (probesize=????)
撞地板时 T2b: ????ms
理论分析: OS pipe buffer = 4KB-64KB (Windows default); 该步 probesize=...
最终 default 选定: P2.A.?  (上一通过 step)
```

---

## 5. Track A 收尾流程（决策树 3.A 或 3.B 触达后）

### 5.1 配置定型

`config.py` 把 P2.A 最优值写为 `_env` 的 default:

```python
MUX_INTERMEDIATE_TS_PROBESIZE = _env("MUX_INTERMEDIATE_TS_PROBESIZE", "8192").strip()  # 示例
MUX_INTERMEDIATE_TS_ANALYZEDURATION = _env("MUX_INTERMEDIATE_TS_ANALYZEDURATION", "0").strip()
```

`MUX_RAW_VIDEO_PROBESIZE` 同理 default 写 `"1000000"`、`MUX_RAW_VIDEO_ANALYZEDURATION` 写 `"1000000"`。

### 5.2 关掉诊断 verbose

```python
MUX_LATENCY_DIAG = _env("MUX_LATENCY_DIAG", "1") == "1"          # 保留
MUX_LATENCY_DIAG_VERBOSE = _env("MUX_LATENCY_DIAG_VERBOSE", "0") == "1"  # 关闭
```

T3a/T3b/T3c stderr 文本扫描在 release 不需要常驻。

### 5.3 数据回填到 Track A 主文档

`summary/summary_20260523_ROADMAP_TRACK_A_BREAKDOWN_CN.md` 第 10 节追加:

- baseline first chunk → 最终 first chunk 的总降幅
- 各阶段贡献（A1-A3 / A6 / A8.1 / A8.P1 / A8.P2.A / 可选 P2.B P2.C）
- 锁定的 default env 集合
- "Track A 完结"标注

### 5.4 提交

建议 3 个独立 commit:
1. Step 1 接通（config + `_mux_intermediate_ts_probe_args` + 单测）
2. 阶梯实验后的 default 锁定（仅改 default 值）
3. verbose diag 关闭 + summary 文档更新

---

## 6. P2.D 评估（仅在 §3.C 全部用尽时启动）

**侵入式方案**: 引入 Python 转发线程，把 video mux stdout 以小块（< 4KB）write 到 final mux stdin，绕开 OS pipe buffer。

### 6.1 风险

- Python 线程 + GIL 在 8K 高码率流下可能成为新瓶颈
- 增加一次内存 copy
- 中间字节流逻辑复杂化，未来 maintain 成本增加

### 6.2 不要在 P2.A/B/C 数据出来之前讨论 P2.D。

---

## 7. 不做的事（再次明确）

- ❌ 不要把 `MUX_INTERMEDIATE_TS_PROBESIZE` 设到 < 4096（audio-only 回归边界）
- ❌ 不要在 P2.A 跑完前改 `-pat_period` 或 `_mpegts_flags()`（污染归因）
- ❌ 不要把 `_mux_intermediate_ts_probe_args` 改成接 `MUX_CONTAINER_PROBESIZE_OVERRIDE`（复用旋钮污染其他路径）
- ❌ 不要省略 nPlayer + Quest3 双验直接跳到下一 step
- ❌ 不要在每个 step 之间忘记冷启动（残留进程会让 T0 不准）
- ❌ 不要看到 `total < 1500ms` 就继续无脑下探（更小 probesize 收益 < 边际风险）

---

## 8. 时间预算

| 阶段 | 工作量 |
|---|---|
| Step 1 接通 + 单测 + 验证 | 30 分钟 |
| Step 2 阶梯 4 step × (冷启 + 双验 + 截日志) | 60 分钟 |
| Step 3 决策 + default 锁定 | 15 分钟 |
| §5 收尾 + 提交 | 30 分钟 |
| **总计** | **~2.5 小时** |

P2.B/C 视情况追加，每个 30 分钟。

---

## 9. 验收清单

完成 P2.A 时:

- [ ] `config.MUX_INTERMEDIATE_TS_PROBESIZE` / `MUX_INTERMEDIATE_TS_ANALYZEDURATION` 已新增
- [ ] `_mux_intermediate_ts_probe_args()` 读取新 env，default 留空
- [ ] 单元测试覆盖 2 种 env 状态
- [ ] §4.1 阶梯表 4 行（baseline + 3 step）全部填完
- [ ] 撞地板 step 明确标注
- [ ] 最优 step 的 default 写入 `config.py`
- [ ] nPlayer + Quest3 双验通过
- [ ] 长播 5 分钟无回归
- [ ] `total` 实测下降记录到 Track A 主文档

---

## 10. 完工后的下一步

| total 达成区间 | Track A 状态 | 下一步 |
|---|---|---|
| < 1500ms | 完结 | 进入 Track B/C/D/E 选择 |
| [1500, 1800ms] | 视 ROI | 可选 P2.B 单点 |
| > 1800ms | 未完结 | 顺序执行 P2.B → P2.C → P2.D 评估 |

无论哪种结果，A4/A5 已经独立完工，用户感知层（UI 进度条 + 结构化日志）已具备，**P2.A 失败也不会让 Track A 在用户体验维度退步**。
