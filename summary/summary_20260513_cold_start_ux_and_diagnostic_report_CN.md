# 冷启动 UX 与诊断报告功能 — 工作小结

- 日期：2026-05-13
- 范围：UI 启动遮罩、`/status` 轮询、诊断报告、`main.py` 启动状态流
- 目标用户：非技术型最终用户（VR 头显场景），失败时由用户一键复制硬件报告寻求支持

---

## 1. 背景

- 用户硬件升级到 RTX 50 系（sm_120）后，首次启动需要 onnxruntime-gpu / CuPy 现场 JIT 编译，整个过程可达 1–3 分钟。先前的 UI 在这段时间内只显示一个静态主页，没有任何提示，非技术用户很容易误判为崩溃直接关掉。
- 用户测试覆盖 GTX TITAN X（sm_5.2 / 已被现代 CuPy 14.x 与 ORT-GPU 1.19.2 抛弃），这类不被支持的硬件需要一个清晰的“启动失败 + 一键复制报告”的退出路径。
- 用户要求：**(1)** 非技术友好的引导；**(2)** 一键复制硬件诊断报告。

## 2. 新增 / 修改文件一览

| 文件 | 状态 | 作用 |
| --- | --- | --- |
| `utils/gpu_runtime_cache.py` | 修改 | 新增 `ColdStartReport` 数据类与 `predict_warmup_state()` 纯函数；新增 sm_120 已知慢组合识别 |
| `utils/startup_status.py` | 重写 | 扩展结构化字段（step / progress / eta / cold / is_known_slow / gpu_* / reason / detail）；新增 `reset_startup_progress()` |
| `main.py` | 修改 | 启动前发布预测、warmup 失败前发布 `phase=failed` 并预留 0.8 s 给轮询器读取，再关闭状态服务 |
| `ui/diagnostics.py` | 新增 | `build_diagnostic_report()`：拼接 GPU / ORT / CuPy / **FFmpeg+FFprobe+NVENC** / Warmup marker / `/status` / **server.log 尾 200 行** |
| `ui/services/startup_status_poller.py` | 新增 | `StartupStatusPoller`（500 ms 轮询 `127.0.0.1:8299/status`），终止相位 `warmed/listening/failed/shutting_down` 发出 `finished` |
| `ui/widgets/startup_overlay.py` | 新增 | 非模态 QDialog 启动遮罩：标题/消息/ETA/进度条/黄色提示/详情面板/复制报告/取消按钮；动态省略号；阻塞步骤切换为不定进度（busy）模式 |
| `ui/main_window.py` | 修改 | 启动时打开遮罩、启动轮询；多重终止路径；失败时合并而非替换 last_status |
| `ui/translations/{zh_CN,en_US,ja_JP}.json` | 修改 | 新增 19 个 `startup.*` 键 |
| `tests/test_predict_warmup_state.py` | 新增 | 9 条用例：sm_120/旧 ORT 已知慢识别、marker 缺失/存在/解析失败、`set_startup_phase` 结构化 kwargs、`reset_startup_progress()` |

## 3. 关键设计决策

### 3.1 预测先行
启动时（heavy CUDA 工作之前）调用 `predict_warmup_state()` 把预测（cold / reason / eta / gpu / is_known_slow）写入 8299 端点。UI 立刻显示“首次启动需要 1-3 分钟”而不是空白等待。

ETA 桶：
- `_ETA_CACHE_HIT_SEC = 4.0`
- `_ETA_KEY_CHANGED_SEC = 30.0`
- `_ETA_FIRST_RUN_NORMAL_SEC = 45.0`
- `_ETA_FIRST_RUN_KNOWN_SLOW_SEC = 150.0`

已知慢组合：`compute_capability` major ≥ 12 且 `onnxruntime` 版本 < (1, 22)。

### 3.2 友好遮罩 + 不定进度回退
warmup 内部是一次阻塞调用，无法发出子步骤事件。仅靠 `progress=0.1` 的一次更新会让进度条视觉上“卡住”。

`StartupOverlay.apply_status()` 在 `phase=warming && elapsed≤0.1 && progress≤0.11` 时把进度条切换到 `setRange(0, 0)` 的不定（marquee）模式，让动画继续跑；当后续 `/status` 真正给出进度时再切回确定模式。

### 3.3 一键诊断报告（单按钮策略）
非技术用户不应被“硬件报告 vs 日志”两个按钮迷惑。所有内容合并到同一份纯文本，复制到剪贴板即可粘贴给技术支持：

- 时间 / 应用版本 / 主机 / OS / Python / 是否 frozen / cwd
- `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap`
- onnxruntime / providers / cupy / cupy devices / numpy
- **ffmpeg / ffprobe 解析路径 + 首行版本 + NVENC 编码器存在性**（Windows 上最常见的支持工单根因）
- Warmup marker（含 ORT cuda dll 哈希）
- 最近一次 `/status` 完整字段
- **server.log 尾 200 行**（最有诊断价值的部分）

实现要点：
- 所有探针都是惰性导入 / 子进程 + 短超时（5 s），任一步骤异常都被吞掉并打印安全提示，绝不抛出。
- 日志读取用 `errors="replace"`，部分 UTF-8 序列损坏不会让报告失败。

### 3.4 多重终止路径（鲁棒退出）

| 路径 | 触发 | 行为 |
| --- | --- | --- |
| `/status` 轮询命中 `warmed`/`listening` | 正常完成 | 合并 last_status，闪显 100%，关闭遮罩 |
| `/status` 轮询命中 `failed` | warmup 主动抛错 | 合并 last_status，保留 detail/message，遮罩留在失败视图供复制报告 |
| `Uvicorn running on` 出现在 stdout | 8299 不可达（端口冲突/防火墙/IPv4 禁用） | `_scan_server_output_for_ready` 合成 `listening` 状态强制关闭遮罩 |
| `QProcess.finished` 且 last_status 非就绪态 | 服务进程崩溃且 8299 在轮询周期内消失 | `_server_state_changed` 合成 `failed` 状态，保留之前的 gpu_name/step/reason 等字段 |
| 用户点 Cancel | 手动放弃 | `_cancel_startup` 同时停 server、停 poller、关遮罩 |

服务端额外保险：warmup 失败发布 `phase=failed` 后 `time.sleep(0.8)` 再 `stop_startup_status_server()`，给 500 ms 轮询器一次稳妥读取窗口；UI 端的合成失败状态作为兜底。

### 3.5 失败时不擦字段
之前 `_on_startup_finished(failed)` 用 `apply_status({"phase":"failed","message":...})` 替换全字段，结果诊断报告里 `gpu_name/step/cold/reason/detail` 全空。改为：

```python
merged = dict(self.startup_overlay.last_status() or {})
merged["phase"] = "failed"
if not merged.get("message"):
    merged["message"] = self.i18n.t("startup.failed_generic")
self.startup_overlay.apply_status(merged)
```

`warmed`/`listening` 路径同步改为 merge 模式，保证成功路径的 GPU 信息也能保留进报告。

## 4. 用户测试验证 & 修复回顾

### 4.1 GTX TITAN X (sm_5.2) 首次启动失败
- 日志显示 `nvrtc: error: invalid value for --gpu-architecture` + `CUDA Provider not available`。
- 当时 UI 长时间停在 `warming/ort_session_and_runs/progress=0.1` —— 服务端只发了一次 `progress=0.1` 后阻塞数秒，最后发布 `failed` 后 ~12 ms 内就关闭了 8299。
- 修复后：服务端 0.8 s 延迟 + UI 端合成 failed 兜底，遮罩可靠地翻到失败视图；复制报告里直接看到 nvrtc/CUDA Provider 关键行。

### 4.2 warm cache 机器遮罩不关
- 现象：UI 停留在“正在连接服务器进程”，详情面板空，`server.log` 显示已 listening。
- 根因：8299 端点不可达（最常见是端口冲突或本机策略），UI 永远拿不到状态。
- 修复：监听 `ServerProcess.output` 中的 `Uvicorn running on` / `Application startup complete`，作为不依赖 8299 的就绪信号强制关闭遮罩。

### 4.3 失败态诊断字段为空
- 根因：`_on_startup_finished(failed)` 用 dict 替换而非 merge。
- 修复：merge 之后所有 GPU/step/reason 字段都保留。

## 5. 测试结果

- 新增 `tests/test_predict_warmup_state.py`：9 条用例全部通过。
- 全量回归（15 个测试模块）：61/61 通过。
- 命令：`PYTHONPATH=. .venv/Scripts/python.exe -m unittest tests.<module> ...`

## 6. 翻译键新增（19 条）

`startup.window_title / title_starting / title_first_run / title_first_run_slow / title_verifying / title_ready / title_failed / connecting / complete / gpu_label / eta_template / hint_known_slow / hint_failed / failed_generic / show_details / hide_details / copy_report / report_copied / cancel`

zh_CN / en_US / ja_JP 三份 JSON 同步，由 `tests/test_i18n.py` 保证键集一致。

## 7. 已知后续可做

- `warmup_gpu_runtime_cache()` 内部仍是单次阻塞调用；如果想从 `progress=0.1` 之后还能给出实质进度，需要在 warmup 内部分阶段回调（OrtSession 加载、首次 batch=1 run、batch=2 run、second-pass 验证），代价较大，当前以不定进度模式视觉上覆盖了。
- 已知慢识别目前仅覆盖 sm_120 × ORT < 1.22。若未来 sm_130 等出现可以扩展 `_KNOWN_SLOW_*` 常量。
- 老硬件（TITAN X / sm_5.2）已不被 CuPy 14.x 支持，仅靠 UI 无法挽救；当前策略是清晰地暴露失败 + 一键报告，不去尝试软件兼容。

## 8. 关键文件路径速查

- `D:\p\PTServer\utils\gpu_runtime_cache.py` — `predict_warmup_state`, `ColdStartReport`, ETA 桶, sm_120 检测
- `D:\p\PTServer\utils\startup_status.py` — 结构化 `_state`, `set_startup_phase(**fields)`, `reset_startup_progress`
- `D:\p\PTServer\ui\diagnostics.py` — `build_diagnostic_report(app_version, language, last_status, marker_path, log_path, log_tail_lines)`
- `D:\p\PTServer\ui\services\startup_status_poller.py` — 500 ms 轮询，终止相位集合
- `D:\p\PTServer\ui\widgets\startup_overlay.py` — 遮罩 QDialog，不定进度模式切换
- `D:\p\PTServer\ui\main_window.py` — `toggle_server` / `_on_startup_status` / `_on_startup_finished` / `_scan_server_output_for_ready` / `_server_state_changed` / `_copy_startup_report`
- `D:\p\PTServer\main.py` — warmup 阶段事件 + 失败 0.8 s 延迟
- `D:\p\PTServer\tests\test_predict_warmup_state.py` — 9 条用例
