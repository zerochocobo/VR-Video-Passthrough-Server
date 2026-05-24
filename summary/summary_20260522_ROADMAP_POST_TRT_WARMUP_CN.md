# PTServer 后续工作备忘录（TRT cold-start warmup 之后）

日期：2026-05-22

## 0. 当前完结状态

已收尾的工作：

- TensorRT 静态 ONNX + IOBinding 路径打通，稳态 8K alpha ~76fps
- 离线 alpha 路径吞吐对齐（36 → 75fps，详见 `summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_RESOLUTION_CN.md`）
- TRT 冷启动 warmup 阶段 1：singleton 复用 + static TRT 预加载（详见 `summary_20260522_TENSORRT_COLD_START_WARMUP_PATCH_STAGE1_CN.md`）
- TRT 冷启动 warmup 阶段 2：`Matter.__init__` 形参隔离（详见 `summary_20260522_TENSORRT_COLD_START_WARMUP_PATCH_STAGE2_CN.md`）
- 首次播放 `alpha #1 ort_run = 18.6ms`（远低于 30ms 门槛）

后续工作分 5 个 Track 推进。本文档为 Track A-E（暂不含发版/打包 Track F）的备忘录。

## 1. Track A：首帧 ramp-up 收尾

目标：把 frame 30 fps 从 32 推到 ≥40，frame 60 从 44 推到 ≥55，把首播体验完全平滑掉。

| 步骤 | 内容 | 工作量 | 预期收益 | 依赖 |
|---|---|---|---|---|
| **A1** | **诊断首帧 `preprocess=39.3ms` 来源**。在 `Matter.composite_green()` / `composite_nv12()` 入口加细分计时，分离 CuPy kernel JIT、GPU buffer 分配、H2D 拷贝、batch=2 SBS 拆分等子段 | 0.5 天 | 不直接收益，为 A2 提供数据 | 无 |
| **A2** | **CuPy 预处理路径 warmup**。在 `utils/gpu_runtime_cache.py` 的 `_warmup_resident_matter_runtime` 内追加一段：用 zero frame 跑 `composite_green` / `composite_nv12` / `AlphaPacker.pack_uploaded` 各一次，让 CuPy kernel JIT 和 buffer 分配在启动期完成 | 1 天 | preprocess 首次 ~40ms → <5ms | A1 数据 |
| **A3** | **NVENC encoder preflight**。启动阶段创建一个目标分辨率（8192×4096 或常见 4K）/preset=P1 的 NVENC encoder，立即释放。让首段 mux 222ms 的 NVENC bitstream 初始化提前支付 | 1-1.5 天 | mux 首段 222ms → <50ms | 无 |
| **A4** | **结构化 `[WARMUP]` 日志**。当前 freeform 文本，改成 JSON 行：`{"phase":"static_trt_preload","batch":1,"shape":"1024x1024","elapsed_ms":...}`，便于 UI 与监控解析 | 0.5 天 | 可观测性 | 无 |
| **A5** | **UI 启动进度条**。`startup_status` 字段扩成 phase 进度（warmup → matter init → static TRT batch=1 → batch=2 → NVENC preflight → ready），UI 显示 | 1 天 | 体验 | A3 + A4 |

**推荐做法**：

- A1 → A2 → A3 是性能轨，按顺序做。A1 数据出来后 A2/A3 可并行。
- A4 + A5 是 UX 轨，可推后或与性能轨并行。
- 完成 A1-A3 后，再做一次冷启动观测，对照阶段 2 收尾数据看 frame 30 fps 改善幅度。

**风险**：

- A2 的 CuPy warmup 如果触发 `_g_chw / _g_frame / _g_out` 申请，会提前占用约 100-200MB 显存。可接受。
- A3 的 NVENC preflight 在不同分辨率/preset 下行为可能不同，需要选定一个「最常见 case」做 warmup；不会预测所有真实请求。
- A4 改日志格式可能影响下游解析（当前 startup_status 解析逻辑），同步更新。

## 2. Track B：RVM 稳态性能调优

目标：稳态从 ~76fps 推到 ≥85fps，或在更高 input_size 下保持 75fps。

| 步骤 | 内容 | 工作量 | 预期收益 | 依赖 |
|---|---|---|---|---|
| **B1** | **CUDA Graph 重新评估**。当前 `ONNX_TRT_CUDA_GRAPH_ENABLE=0`。RVM static shape + IOBinding + 固定输入下应可开启。关键：batch=1 / batch=2 切换时 graph 重建开销 | 1-2 天 | ort_run 5.86ms → 4-5ms（稳态 +5-10%） | 无 |
| **B2** | **Alpha 路径接入 `Matter.acquire_nv12_output_slot()` ring slot**。之前评议方案 B，因接口差异搁置；阶段 2 后 alpha throughput 已与 green 对齐，但接 ring slot 仍有 sync 路径统一化收益 | 2-3 天 | 架构整洁；性能微收益 | 无 |
| **B3** | **ThreadedDecoder 替换 SimpleDecoder**。之前评议方案 C 搁置项。线程化预取释放 NVDEC 并行度 | 2 天 | decode_avg 6.4ms → 4-5ms | 无 |
| **B4** | **input_size 上探到 1280 / 1536**。当前 1024，看更高输入分辨率在 8K 源上的画质 vs 性能 trade-off。涉及 TRT engine cache 重建 | 0.5 天 | 画质或观测点 | 无 |
| **B5** | **RVM recurrent state 显存/带宽 profile**。用 `nsight-systems` / `nvprof` 看 r1i..r4i 在 stream 上的拷贝是否成为瓶颈 | 1 天 | 为后续优化提供数据 | 无 |

**推荐做法**：

- **B1 优先做**，单 commit，风险可控，收益明确。
- **B2 / B3 暂缓**。阶段 2 后 alpha = green = 75fps 已对齐，B2/B3 的边际收益要在打开 CUDA Graph + 提升 input_size 后再评估。
- **B4** 可在 B1 完成后试一次。
- **B5** 仅在需要进一步榨性能或遇到瓶颈时再做。

**风险**：

- B1 在 batch=1/2 切换处可能引入 graph rebuild 抖动；如果观察到稳态 fps 下降，需要回退或加 hysteresis。
- B4 上探后 TRT engine cache 会无效，下次启动有一次完整 build（几十秒），UI 需要正确显示这一次 build 进度。

## 3. Track C：DeoVR Alpha 通道输出

目标：从绿幕合成扩展到真 alpha 通道，DeoVR / Quest3 客户端透明叠加。

| 步骤 | 内容 | 工作量 | 关键决策点 |
|---|---|---|---|
| **C1** | **DeoVR alpha 容器/编码格式调研**。需调查 DeoVR 官方文档、对实际示例视频抓包/MediaInfo，确认是双流（color + alpha 分轨）、RGBA 内嵌、HEVC alpha extension、还是 SEI metadata | 1-2 天 | 整条 Track 的入口；上游格式决定下游全部 |
| **C2** | **FFmpeg/NVENC alpha 编码支持验证**。NVENC 是否支持 alpha 通道（截至当前 SDK 不支持 RGBA / YUVA）；如果不支持，看 libx264/libx265 软编 alpha 在 8K 是否能做实时（大概率不行）；其它替代方案：双流分别 H.265 编码合 MP4 | 1 天 | Track 可行性总闸 |
| **C3** | **AlphaPacker 输出格式适配**。`AlphaPacker.pack_uploaded()` 当前已是 alpha+source 拼接，需按 C1 决定的格式调整（拆分两路 NVENC encoder、或拼成 RGBA 大帧给软编） | 2-3 天 | Track 主体改动 |
| **C4** | **DLNA `protocolInfo` 与 fMP4 适配**。可能要新增双 MIME 类型（color + alpha）或双 URL 路径；DLNA `dc:title` 可能要带 hint 让 DeoVR 识别为 passthrough alpha 流 | 0.5 天 | DLNA 层最小改动 |
| **C5** | **DeoVR 设备端兼容测试**。Quest3 + DeoVR 实测：起播、seek、暂停恢复、码率自动切换、长时间播放稳定性 | 1 天 | 验收关 |

**推荐做法**：

- **C1 → C2 是调研轨**，必须先做。**结论不明确前不要进 C3+**。
- 如果 C2 结论是「NVENC 不支持 + 软编不实时」，整个 Track C 转为长期研究项目（候选方向：CUDA 上做 RGBA 软编 / NVIDIA Video Codec SDK 后续版本支持 / 用单声道 1bit alpha mask 偷工）。
- C5 必须用真实头显跑，模拟器/桌面播放器不可靠。

**风险**：

- DeoVR 的 alpha 实现是社区/闭源生态，文档可能不完整，需要逆向。
- 8K alpha 实时输出对硬件压力远超绿幕，可能需要降低 input_size 或 fps 来换 alpha。

## 4. Track D：可观测性 / 监控 / 运维

目标：长时间播放下能定位问题，PyInstaller 包能现场诊断。

| 步骤 | 内容 | 工作量 | 预期收益 |
|---|---|---|---|
| **D1** | **HTTP `/metrics` 端点**。FastAPI 路由，暴露 fps、ort_run、queue depth、active workers、GPU 显存占用、TRT engine cache size。Prometheus 文本格式或简单 JSON | 1 天 | 后续监控/告警基础 |
| **D2** | **长时间压测脚本**。自动 4-8 小时 8K alpha 播放，监控 GPU 显存增长、ORT session 行为、worker 重启次数、错误日志 | 1 天写脚本 + 8 小时跑 | 发现潜在显存/资源泄漏 |
| **D3** | **结构化日志全局推进**。当前 log 混合 freeform/特定 tag，统一改成 JSON 行（或并行双格式），便于 ELK/Loki 解析 | 2 天 | 长期维护性 |
| **D4** | **PyInstaller 包诊断日志**。frozen exe 启动失败时（126/127 err code 等），DLL 加载链路日志落盘到 `%LOCALAPPDATA%\PTServer\diag\` | 0.5 天 | 现场排障 |
| **D5** | **worker 异常优雅降级**。TensorRT 在某请求中 fall back 时（`_require_tensorrt_still_active` 抛错），按请求记录并重启 worker，而不是杀整个 server | 1-2 天 | 鲁棒性 |

**推荐做法**：

- **D2 必做**，最便宜（脚本一天 + 跑就行），最可能暴露隐藏 bug。
- **D1 + D4** 紧跟其后，长期价值高。
- **D3** 可推后到 Track E 一起做（与日志格式重构对齐）。
- **D5** 在 D2 真发现 worker 异常事件后再启动。

**风险**：

- D2 跑 8 小时可能占用整张 GPU，无法同时开发；建议夜里跑。
- D3 改日志格式影响所有现有 log 解析逻辑（startup_status / debug_output），同步更新。

## 5. Track E：技术债清理（小步快走）

阶段 1+2 留下的、及历史遗留。颗粒非常小，适合插空做。

| 步骤 | 内容 | 工作量 | 备注 |
|---|---|---|---|
| **E1** | `get_matter()` 改 `*, warmup_runs` kwarg-only | 5 分钟 | 阶段 2 偏差备注里说留作下次 refactor 顺手收 |
| **E2** | `invalidate_singleton()` 落地为函数（即使无调用方），文档化预期使用方式 | 15 分钟 | 阶段 2 trip-wire 配套，留 hook |
| **E3** | tools/ 与 ui/services/ 那 10+ 处 `config.MATTING_WARMUP_RUNS = 0` 统一迁移到 `Matter(warmup_runs=0)` 或 `get_matter(warmup_runs=0)` | 1 小时 | 阶段 2 形参隔离收尾 |
| **E4** | DLNA ConnectionManager 完整实现 | 2 小时 | MEMORY.md 中标注当前只有 GetProtocolInfo |
| **E5** | `pipeline/matting.py` ~3000 行拆模块：`matting/state.py` / `matting/rvm.py` / `matting/composite.py` / `matting/iobinding.py` / `matting/singleton.py` | 1-2 天 | 风险高，**暂缓** |

**推荐做法**：

- **E1 + E2 + E3 一个 commit 收掉**，建议作为阶段 2 后的第一个 commit。
- **E4** 单独 commit。
- **E5 暂缓**到下次大版本前；改动面大、回归风险高，且 `matting.py` 当前 3000 行虽长但内部组织尚可。

**风险**：

- E5 拆模块会污染 git blame 和未来 cherry-pick；不建议无目的地做。

## 6. 推荐路径（按 ROI 排序）

### 短期（一周内）

1. **E1 + E2 + E3**（半天）—— 阶段 2 留下的技术债，干净收尾。
2. **A1 → A2**（1.5 天）—— preprocess JIT 预热，frame 30 fps 32 → ≥38。
3. **D2**（脚本半天 + 跑 4 小时）—— 长时间压测，捞潜在 bug。

### 中期（两周内）

4. **A3**（1.5 天）—— NVENC preflight，mux 首段 222ms → <50ms。
5. **B1**（2 天）—— CUDA Graph 评估，稳态可能 +5-10%。
6. **D1**（1 天）—— `/metrics` 端点。

### 长期（一个月+）

7. **C1 → C2**（调研轨）—— DeoVR alpha 通道方案确认。结论决定后续。
8. **A4 + A5**（1.5 天）—— UI 启动进度条 + 结构化日志。
9. **D4 + D5**（按 D2 压测结果决定优先级）。

### 暂缓项

- **B2 / B3**（收益边际，耦合大；等 B1/B4 后再评估）
- **B5**（无明确瓶颈前不做 profile）
- **E5**（拆模块，风险大）
- **C3+**（取决于 C2 结论）

## 7. 跨 Track 依赖与并行机会

依赖关系：

```
E1+E2+E3 ── 独立
A1 → A2 ── 性能轨基础
       \
        → A3（NVENC，独立）
       /
A4 → A5 ── UX 轨

B1 ── 独立
B4 ── 依赖 B1（CUDA Graph 开启后再调 input_size）

C1 → C2 → C3 → C4 → C5（强串行）

D1 ── 独立
D2 ── 独立
D3 ── 与日志格式（A4）配合
D5 ── 依赖 D2 数据
```

可并行：

- 性能轨（A/B）与可观测性轨（D）可同时推进
- E1-E4 可在任何空闲插空做
- Track C 调研期（C1+C2）可与 A/B/D 并行（调研工作量小）

## 8. 完成判定（每个 Track）

| Track | 完成判定 |
|---|---|
| A | ✅ 已完成 (2026-05-23)。frame 30/60 已达标，首块约 1.94-2.03s；剩余 `T4-T3c` 为双段 mux 地板。终档见 `summary_20260523_TRACK_A_FINAL_ARCHIVE_CN.md` |
| B | 稳态 ≥ 85fps（在当前 input_size=1024 下）；或 input_size=1280 下稳态 ≥ 75fps |
| C | Quest3 + DeoVR 真机播放透明 alpha，能 seek/暂停，长时间稳定 |
| D | `/metrics` 端点上线；8 小时压测无显存泄漏与 worker 异常 |
| E | 阶段 1+2 偏差清零；DLNA ConnectionManager 完整 |

## 9. 不在本备忘录内（明确排除）

- 发版/打包/部署链路完善（Track F）—— 与本文档解耦，单独排期。
- 多模型支持（MODNet/RVM/BEN2/RMBG 切换 UI）—— 当前 RVM 单模型已满足需求。
- 自适应 input_size（根据视频分辨率/源 FPS 动态选）—— 需求未明确。
- 远程管理 / 多 server 集群 —— 超出 PTServer 设计范围。

## 10. 维护说明

本备忘录每完成一个 Track 后更新对应「完成判定」段落与正文中的状态标记。Track 内部步骤完成可直接划掉或在状态列加 ✅。新增需求/发现的 bug 视情况补充到对应 Track 或新增 Track。
