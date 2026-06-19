# SI Sidecar 构建加速 — 详细开发计划

日期: 2026-06-18
状态: 待实现（指派他人），本文件作为实现规范 + 审核基线
负责人: TBD
审核人: 外部专家（本文档作者）

---

## 0. 一句话目标

把"首次冷启动播放 `/media_si`"的准备时间从当前 ~70–80s（甚至更早的 ~150s）压到 **~25s 量级**，且**不改变播放时序/不引入接缝爆音**，对下游 virtual layout 完全透明。

---

## 1. 背景与现状

已完成（前序提交，未 commit 的工作区改动）：

- ① 视频样本表从 `moov` 解析（`_read_video_sample_table_from_moov`），不再 PyAV 全扫。
- ③ 预热单 worker + 有界优先级队列 + 播放抢占 + ffmpeg 降优先级/可取消 + Browse 每目录限 N 条。
- ② 部分：`build_source_audio_sidecar()` + `_write_source_audio_mp4()` 已用 `moov` 索引按音频字节范围抽原声，落成小 `*.source-audio.mp4`，按源视频 stat 缓存；`build_mixed_audio_sidecar()` 优先用该小缓存 + `.si.wav` 混音；首次冷路径用 `pipe:0` 让抽取与混音重叠。

仍未解决（本计划要做）：抽取冷成本 ~80s，且混音+编码是单线程 ~71s。

---

## 2. 问题定位（已实测，2026-06-18，样片 `videos/SI_TEST_8K.mp4` 8GB / 50min / 8K HEVC，机器 8 核）

| 项 | 实测 | 备注 |
|---|---|---|
| 源音频轨 | 96.2 MB / 140958 samples / **140952 chunks** | 即 ~1 sample/chunk，逐 sample 交织散布在整 8GB |
| 抽取（页缓存 **warm**，散读 96MB） | **0.80s** | 176k 随机读/s，纯 Python 迭代开销可忽略 |
| 抽取（页缓存 **cold**，开发实测） | **~80.9s** | 140952 次冷随机读，**纯磁盘 seek 受限** |
| 混音 filter（解码+ducking+amix+limiter，`-f null` 不编码） | **~11.9s** | |
| 混音 + AAC 编码（单线程，到文件） | **~70.9s** | → **AAC 编码本身 ≈ 59s** |
| **8 路并行分段混音+编码** | **~19.3s** | 3.7x（filter 12s 是并发下硬底） |
| 后续 virtual layout 构建 | ~4.3s | |

两个独立瓶颈：

1. **冷抽取 80s** = 140952 次冷随机读（seek 受限，不是带宽受限）。
2. **AAC 编码 59s** = ffmpeg 原生 `aac` 编码器单线程，占满 1 核，其余 7 核闲置。

**关键耦合（必须在计划里强调）**：当前 `pipe` 重叠把"抽取 80s"藏在"混音 71s"之下，冷路径 ~70s。一旦把混音并行化到 19s，**抽取 80s 会重新暴露成瓶颈**：冷路径 = max(80 抽取, 19 编码) = 80s。所以 **B（并行编码）和 A（加速冷抽取）必须一起做，且 A 要把冷抽取压到 ≤ ~19s，B 的收益才能在冷路径兑现。**

---

## 3. 验收指标（硬性）

在样片 `SI_TEST_8K.mp4`、删除所有相关 cache、确保 OS 页缓存 cold 的条件下：

- A1: source-audio 冷抽取 ≤ **20s**（目标 ~10s）。
- B1: 混音+编码（从已有 source-audio cache）≤ **22s**。
- C1: 冷首播总准备（抽取+混音+layout，含重叠）≤ **30s**。
- D1: 后续仅改 `volume/delay/duck` 的 remix ≤ **22s**（不再读 8GB）。
- E1: **正确性**：并行分段产物解码出的 PCM，与单线程 sidecar 解码 PCM 在每个接缝 ±50ms 容差外**逐采样一致**；接缝处无可听爆音/静音空隙。
- F1: 全部既有测试通过；新增测试见 §9。

---

## 4. 设计 A — 冷抽取加速（把随机读变顺序/并行）

根因：音频散布全 8GB，"只读 96MB"在 cold 下是 14 万次 seek。两条可选路线，**实现者需各测一次冷基准再选**：

### A 方案一（首选，低风险）：单趟顺序大块扫描

把 `_copy_file_ranges_to_outputs` 的"逐 run seek+read"改为"**按 offset 顺序的单趟前向扫描**"：

- 维护一个大读缓冲（如 8–32MB）。沿文件**单向前进**，每次 `read(buf_size)`；从缓冲里切出落在本窗口内的音频 sample 字节写出。
- 因为音频 run 已按 offset 升序，且首末音频 sample 几乎覆盖整文件，这等价于"顺序读整个 8GB 跨度，丢弃中间的视频字节"。
- 冷下从"seek 受限 80s"变成"带宽受限"：8GB / (SSD 顺序 ~0.5–1GB/s) ≈ **8–16s**。

注意：仍只**写出** 96MB；只是**读**变成顺序。内存占用 = 单个读缓冲，不缓存全文件。

### A 方案二（可选，若方案一冷测仍 >20s）：N 路并行区段顺序扫描

把文件按 offset 切 N 段（与 §5 的时间分段对齐），N 个线程各自对自己那 1/8 文件区间做"方案一"的顺序扫描，并发写各自的分段音频。NVMe 多队列下常能更接近峰值带宽，并天然与 §5 的并行管线融合（见 §5.4）。

### A 的产物与缓存

- 仍产出单一 `*.source-audio.mp4`（普通 96MB AAC MP4），键沿用 `_source_audio_cache_digest`（按 video stat）。
- `_source_audio_mp4_plan` / `_build_audio_only_moov` 不变。
- 删除/保留 `pipe:0` 重叠路径：A 做完后冷抽取已 ≤20s，**建议删除 pipe 重叠**以降低复杂度（§5 的并行混音从落地的小 cache 读，逻辑更清晰）。是否删除留作审核点 R3。

---

## 5. 设计 B — 并行分段混音+编码 + 接缝裁剪拼接（核心）

从（已 cache 的）`*.source-audio.mp4`（96MB，cold 后也已在页缓存）+ `.si.wav` 出发，把 `build_mixed_audio_sidecar` 改为并行分段。

### 5.1 分段网格（必须帧对齐）

- AAC 帧 = 1024 samples @ 48000 Hz = **0.0213333s/帧**。
- 设 `N = clamp(os.cpu_count() 或配置上限, 1, SI_MIX_PARALLEL_MAX)`（配置见 §8，默认建议 6–8）。
- `total_frames = ceil(source_audio_samples_48k / 1024)`（由 source-audio 时长推出；source-audio 已是 48k？若不是，以 48k 重采样后的样本数为准）。
- 段 i 覆盖**帧区间** `[Fi, Fi+1)`，`Fi = round(i * total_frames / N)`，`F0=0`，`FN=total_frames`。**边界天然落在帧网格上**，段间内容不重叠不缺失。

### 5.2 每段编码（带前导预热 overlap）

- 预热帧数 `L = ceil(WARMUP_SECONDS / 0.0213333)`，`WARMUP_SECONDS≈1.0` → `L≈47` 帧。预热用途有二：吃掉 AAC 编码器 priming（1 帧）+ 让 `sidechaincompress`/`alimiter` 包络在进入"保留区"前到达稳态，**消除接缝处 ducking 电平跳变**。
- 段 i（i>0）实际编码时间范围 `[(Fi - L)*fd , Fi+1*fd)`；段 0 无预热，范围 `[0, F1*fd)`。
- 命令：对输入 `[0]=source_audio.mp4`、`[1]=si.wav` 各加 `-ss <start> -to <end>`（WAV/AAC 输入 seek 都很快），filter 复用 `build_si_mix_filter(...)` **原样不变**，`-c:a aac -b:a 192k -ar 48000 -ac 2 -f mp4 seg_i.mp4`。
- N 个 ffmpeg **并发**，复用现有 `_run_ffmpeg_sidecar`（已带 cancel + 降优先级）。并发度 = N；注意与 §7 的全局 build slot 协调（见 R4）。

### 5.3 接缝裁剪与拼接（决定正确性，重点）

目标：拼出的最终 AAC 轨，**解码 PCM 等价于单线程 sidecar**，且只带一条 edit list。

- 读每个 `seg_i.mp4` 的音频 sample 表（用现有 `_read_sample_table_from_moov(path, moov, "audio")` 或 `_parse_stsz`）。
- 段 i 需**丢弃的前导存储帧数**：
  - i==0：丢弃 = 该段自身 priming（通常 1 帧，以 `seg_0` 的 `skip_samples`/`elst media_time/1024` 为准，**按实测值取，不要硬编码**）。
  - i>0：丢弃 = priming + `L`（预热帧）。
- 丢弃后，段 i 的"保留帧"正好表示内容时间 `[Fi*fd, Fi+1*fd)`；各段保留帧**按段序拼接** = 完整内容帧 `[0, total)`。
- 组最终 audio trak：`stsz` = 保留帧 size 序列；`stts` = 单条目 `(总保留帧数, 1024)`；`stco/co64` 指向新 mdat；`stsd/esds` 复用任一段（编码参数一致）。
- **edit list / priming 约定**：保留帧已不含 priming，最终轨设 `elst media_time=0`（呈现从内容 0 开始）。这与当前单线程 sidecar 的 `media_time=1024` 在"呈现 PCM"上等价，但对忽略 edit list 的播放器更干净。**此项与未解决的 A/V sync 调查相关，列为审核点 R1**：实现者需在 PR 里明确最终采用 `media_time=0`（去 priming）还是 `1024`（留 priming，匹配旧行为），并附 PCM 对比证据。
- mdat：把各段保留帧的样本字节拼接落盘（从各 `seg_i.mp4` 按其 sample offset 读出，或编码时各段已是连续 mdat 可整段拷贝减去前导）。产出与现有 `*.audio.mp4` 同名同位置，下游 `build_progressive_si_virtual_mp4` **零改动**。

### 5.4 （可选融合）每段直接从 8GB 读自己区段的音频

若 A 选方案二，可把"读该段音频字节 → 内存拼小 MP4 → 喂 ffmpeg `-i pipe:0`"做进每个 worker，彻底省掉中间 `*.source-audio.mp4`。**默认不做**（复杂度高）；先做 A方案一 + B（从落地小 cache 读），跑通达标再评估。列为审核点 R5。

---

## 6. 代码改动清单（文件 / 函数级）

`pipeline/si_virtual_mp4.py`：

- [改] `_copy_file_ranges_to_outputs`：实现 §4 方案一顺序大块扫描（或新增 `_copy_audio_ranges_sequential`）。保留逐 run 版本作为 `_sample_runs` 退化兜底。
- [新] `build_mixed_audio_sidecar_parallel(video, si_wav, params, *, cancel_event, low_priority, segments)` 或在 `build_mixed_audio_sidecar` 内按 `SI_MIX_PARALLEL_MAX>1` 分流。
- [新] `_plan_mix_segments(total_frames, n) -> list[(Fi, Fi1)]`（纯函数，易单测）。
- [新] `_encode_mix_segment(...)`：构 `-ss/-to` + filter + `_run_ffmpeg_sidecar`，产 `seg_i.mp4`。
- [新] `_stitch_aac_segments(seg_paths, drop_frames_per_seg, out_path)`：§5.3 拼接，复用 `_parse_stsz / _make_co64 / _make_stsc_one_sample_per_chunk / _build_audio_only_moov` 等现有 box 工具。
- [改] 调用处：`build_progressive_si_virtual_mp4` 内 `build_mixed_audio_sidecar(...)` 改为可走并行实现（开关控制，便于 A/B 与回退）。
- [回退] 任一并行/拼接步骤异常 → `log.warning` 后回退单线程 `build_mixed_audio_sidecar` 老路径，保证不退化为不可用。

`config.py`：见 §8。

`tests/`：见 §9。

`prompt/HANDOVER_*.md`、`summary/`：更新。

---

## 7. 集成：并发 / 取消 / 优先级（不要破坏前序 ③ 的设计）

- 全局只有一个 build slot（`_build_slot`）。并行分段的 N 个 ffmpeg 属于**同一个** layout build，应在**已持有 slot 的那次 build 内部**并发起子进程，**不要**让每段去抢全局 slot（否则与播放抢占语义冲突）。审核点 R4。
- `cancel_event`：传入每段 `_run_ffmpeg_sidecar`；取消时**全部**子进程 terminate，清理 `seg_*.mp4` 与 `*.tmp`。
- 低优先级（预热）：N 段 ffmpeg 都加 `BELOW_NORMAL_PRIORITY_CLASS`。注意预热时 N=8 可能与前台播放争 CPU——预热路径建议 `N_prewarm = max(1, cpu//2)` 或直接复用播放的 N 但靠 OS 降优先级让路。审核点 R4。
- 临时文件：放 `_ensure_cache_dir()` 下带唯一前缀（含 digest + pid），完成后清理；异常/取消路径必须清理。

---

## 8. 新增配置项（`config.py`，PT_ 前缀 env，给默认值并写进 `test_config_defaults`）

- `SI_MIX_PARALLEL_MAX`（默认 `min(8, cpu_count)` 或固定 `6`）：分段并发上限；`1` = 关闭并行，强制老单线程路径（回退开关）。
- `SI_MIX_SEGMENT_WARMUP_MS`（默认 `1000`）：每段前导预热毫秒，影响 `L`。
- `SI_AUDIO_EXTRACT_MODE`（默认 `sequential`；可选 `runs`=旧逐run、`parallel`=A方案二）：抽取策略开关，便于冷基准对比与回退。
- 复用既有 `SI_PREWARM_QUEUE_MAX` / `SI_BROWSE_PREWARM_LIMIT`，不动。

---

## 9. 测试计划

纯函数 / 结构单测（必须，CI 友好，不依赖大文件）：

- `_plan_mix_segments`：N=1/3/8、边界帧对齐、`F0=0`、`FN=total`、段不重叠不漏。
- `_stitch_aac_segments`：构造 3 个合成 AAC-ish 小段（可用真实 ffmpeg 生成 3 个 1s 正弦 AAC），验证拼接后帧数 = Σ保留帧、`stts` 单条目、`stco` 单调、`elst` 约定符合 R1 决议。
- 抽取顺序扫描：合成一个"音频 sample 散布在大 padding 中"的小 MP4，验证 `sequential` 与 `runs` 两种产物**字节一致**。
- 回退：mock 段编码抛错 → 确认回退单线程路径并产出可用 sidecar。

真机基准 + 一致性（手动，记录到 HANDOVER）：

- 删除 cache，cold 跑：记录 A1/B1/C1/D1 实测，贴日志。
- **PCM 等价校验脚本**：分别用单线程 sidecar 与并行 sidecar，`ffmpeg -i x -f f32le -ar 48000 -ac 2 out.pcm`，逐采样比对；接缝 ±50ms 容差区外必须一致；统计最大/RMS 差。产出附 PR。
- 接缝可听性：导出每个接缝前后 0.5s，人工/波形确认无空隙、无电平台阶。

回归：

- `tests\test_si_mix.py test_si_virtual_mp4.py test_config_defaults.py test_routes_media_cache.py test_content_directory_modes.py -k "not versioned_live_id_resolves"` 全过。
- `git diff --check`、`py_compile` 通过。

---

## 10. 风险 / 超出范围

- 真正"首字节即时、与文件大小无关"只有 **fMP4 边播边生成** 能做到，但有播放器兼容风险（SKYBOX/nPlayer/VR），是更大重构。**本计划不含**；A+B 达标后再评估。
- 接缝处 ducking 包络在极端素材下仍可能有微小差异（预热 1s 通常足够）；若 PCM 校验发现接缝差异超阈值，调大 `SI_MIX_SEGMENT_WARMUP_MS` 或减小 N。
- A 方案一在**机械硬盘**上顺序读 8GB 也要 ~50s+；本机为快存储。若部署到 HDD，需回看 A 方案二或接受较慢。

---

## 11. 审核检查点（交付时我按此审）

- R1: 最终 edit-list/priming 约定（`media_time=0` vs `1024`）已明确，并附 PCM 等价证据；与 A/V sync 调查不冲突。
- R2: 冷基准 A1/B1/C1/D1 全部达标且有日志佐证（不能只给 warm 数）。
- R3: `pipe:0` 重叠路径是否删除/保留，理由清晰。
- R4: 并行 ffmpeg 在单个 build slot 内并发，未破坏 ③ 的播放抢占/取消/降优先级语义；预热并发度对前台播放友好。
- R5: 是否融合 §5.4，决策有据。
- R6: 取消/异常路径无残留 `seg_*`/`*.tmp`；回退路径实测可用。
- R7: 下游 `build_progressive_si_virtual_mp4` 零改动，sidecar 产物路径/命名不变。

---

## 12. 建议实施顺序

1. A 方案一（顺序扫描）+ 冷基准（先把 80s 干到 ≤20s，独立可验证、低风险）。
2. `_plan_mix_segments` + 单测。
3. `_encode_mix_segment` + `_stitch_aac_segments` + 拼接单测。
4. 串起并行 `build_mixed_audio_sidecar`，PCM 等价校验。
5. 配置项 + 回退开关 + 取消/清理。
6. 真机 cold 全链路基准，回归，更新 HANDOVER。
