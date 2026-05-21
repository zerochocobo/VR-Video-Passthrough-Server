# 冷启动 UI 进度反馈改造实施计划

日期：2026-05-21
分支：master
目标：解决"启动后弹窗显示『正在连接服务器进程』几分钟不动 → 然后突然消失"的体验问题。让用户在整个冷启动过程中（最长可达 150s）持续看到有意义的进度更新。

---

## 1. 问题根因（先于方案，方便审计）

### 现象
- 用户点击启动 → 弹出 `StartupOverlay` → 显示中文 "正在连接服务器进程" → 数十秒至数分钟不动 → 突然关闭进入主界面。
- 中间没有 GPU 名、ETA、阶段、JIT 提示，看起来像卡死。
- 用户日志最后一条停在 `[matting] model kind=rvm input=src shape=[...]`，之后服务端 stdout 静默直到 uvicorn 起来。

### 两个"无声窗口期"
| 窗口 | 期间 | UI 端实际状态 | 服务端实际状态 |
|---|---|---|---|
| A | UI 弹窗 → 服务子进程绑定 :8299 | 每 500ms `URLError unreachable`，但 `StartupStatusPoller.error` 信号**未在 `main_window.py` connect**，UI 完全无感知 | PyInstaller 启动 + `configure_gpu_runtime_cache` + `setup()` 等几秒 |
| B | `set_startup_phase("warming", "loading ONNX Runtime CUDA and running warmup", progress=0.1)` 之后 | `apply_status` 进入 `warming + elapsed<=0.1 + progress<=0.11` 特例分支，进度条切 indeterminate marquee，文字英文写死且**整段阻塞期完全不变** | `warmup_gpu_runtime_cache()` 单点阻塞：`build_warmup_key` → `import Matter` → `Matter()` ORT JIT → `for shape × runs: _run_rvm_iobinding_from_dev` → `_cache_stats` → `write_marker`，全程 0 次状态回写 |

代码现状的自我承认（已存在的 TODO）：
- `main.py:221-224` 注释：`"warmup_gpu_runtime_cache is a single blocking call. We can't currently emit per-substep events without restructuring it"`
- `ui/widgets/startup_overlay.py:240-246` 用 marquee 兜底，承认"feels frozen to a non-technical user"

### 文案根因
- `ui/translations/zh_CN.json:131` `"startup.connecting": "正在连接服务器进程"` 是 `StartupOverlay.reset()` 的默认底字。只要 poller 没成功响应过一次，就被 `_tick_ellipsis` 持续加点显示。
- 服务端 `set_startup_phase` 的 `message` 字段写死英文（如 `"loading ONNX Runtime CUDA and running warmup"`），UI 直接显示未 i18n 化的英文。

---

## 2. 改造总览（五个改造点 + 优先级）

| # | 改造点 | 目标 | 文件 | 估算 LOC | 优先级 |
|---|---|---|---|---|---|
| ① | UI 端补 `poller.error` 处理 | 干掉窗口 A 的"假死" | `ui/main_window.py`, `ui/widgets/startup_overlay.py`, 3 个 i18n json | ~40 | P0 |
| ② | startup_status 心跳线程 | 阻塞期间 elapsed/progress 持续推进，UI 进度条不再死 | `utils/startup_status.py`, `main.py` | ~50 | P0 |
| ③ | 拆开 `warmup_gpu_runtime_cache` + progress_cb | 把 30~150s 黑盒拆成 6 段语义化进度 | `utils/gpu_runtime_cache.py`, `main.py` | ~80 | P1 |
| ④ | predict 之前先发"探测硬件中" | 填补 `process started` → `first-time GPU initialization` 的几秒空窗 | `main.py` | ~10 | P2 |
| ⑤ | UI 端按 step 映射 i18n 翻译 | 中文界面不再显示英文 message | `ui/widgets/startup_overlay.py`, 3 个 i18n json | ~50 | P2 |

P0 = 必须做（< 100 LOC 立刻解决主要痛点）；P1 = 强烈推荐（提供真进度）；P2 = 锦上添花。

---

## 3. 详细方案

### 改造 ① — UI 端补 poller.error 处理

**目标**：UI 弹窗后 2 秒内若仍连不上 :8299，提示用户"服务进程正在启动中，首次启动可能需要 1~3 分钟"。

**修改**：
- `ui/main_window.py`：
  - `__init__` 增加 `self.status_poller.error.connect(self._on_startup_error)` 和 `self._poll_error_streak = 0`、`self._poll_first_success = False`。
  - 新增 `_on_startup_error(message: str)`：累加 `_poll_error_streak`；若 ≥ 4（≈ 2s）且 `not _poll_first_success`：调用 `self.startup_overlay.show_bootstrapping_hint()`；若 ≥ 60（≈ 30s）：调用 `show_bootstrapping_hint_long()`（更具体的"GPU 首次编译"提示）。
  - `_on_startup_status` 开头补 `self._poll_first_success = True; self._poll_error_streak = 0`。
  - `_open_startup_overlay` 重置这两个字段。
- `ui/widgets/startup_overlay.py`：
  - `reset()` 内已经有 `_base_message`，不动。
  - 新增 `show_bootstrapping_hint()` / `show_bootstrapping_hint_long()`：把 `_base_message` 替换为对应 i18n key 翻译；保留 ellipsis 动画；进度条保持 marquee。
- i18n（`zh_CN.json` / `en_US.json` / `ja_JP.json`）：
  - 新增 `startup.bootstrapping`（zh: "服务进程正在启动中，首次启动可能需要 1~3 分钟…"）
  - 新增 `startup.bootstrapping_long`（zh: "服务进程仍在启动。GPU 首次冷启动需要为本机显卡编译内核，请耐心等待。"）

**验证**：
- 模拟 :8299 不响应（如把端口改成不存在的）：UI 弹窗 2 秒内文字切换为 bootstrapping，30 秒后切换为 bootstrapping_long。
- 正常启动：第一次成功响应后文本不再被 error handler 替换。

---

### 改造 ② — startup_status 心跳线程

**目标**：在 set_startup_phase 之间补"假装在动"，让 ETA 行（"已用 X 秒，预计还需 Y 秒"）和进度条持续推进，避免 marquee 哑动画。

**修改**：
- `utils/startup_status.py`：
  - 新增模块级 `_heartbeat_thread: threading.Thread | None`、`_heartbeat_stop: threading.Event | None`。
  - 新增 `start_heartbeat(eta_sec: float, baseline_progress: float, ceiling_progress: float = 0.95) -> None`：
    - 启动 daemon 线程，每 500ms：
      - `with _lock`: `_state["elapsed_sec"] += 0.5`；`_state["updated_at"] = time.time()`。
      - 按 `min(ceiling_progress, baseline_progress + (1-baseline_progress) * elapsed/eta_sec)` 推进 `_state["progress"]`，**永远不超过 ceiling**（留给真实事件覆盖到 1.0）。
  - 新增 `stop_heartbeat()`：set event + join(1.0)。
  - `set_startup_phase(phase, message, **fields)`：若 `phase` 变化或 `step` 变化，自动 `stop_heartbeat()` + 重置 elapsed_sec=0 + 按新 eta 重新 `start_heartbeat`。（或显式由 `main.py` 控制，看实现简洁度选哪种。）
- `main.py`：
  - `predict_warmup_state()` 返回后 `set_startup_phase` 时，调 `start_heartbeat(prediction.estimate_sec, baseline_progress=0.1, ceiling_progress=0.95)`。
  - warmup 成功/失败后 `stop_heartbeat()`。

**验证**：
- 启动观察 UI：进度条不再是 marquee，是真实从 10% 缓慢爬到 ~95%；ETA 行每秒更新一次。
- 即使 warmup 阻塞 90s，UI 数字也在动。

---

### 改造 ③ — 拆开 warmup_gpu_runtime_cache + progress_cb

**目标**：把单点阻塞拆成 6 个语义化阶段，每个阶段都通过 callback 写 /status。

**修改**：
- `utils/gpu_runtime_cache.py`：
  - `warmup_gpu_runtime_cache` 签名增加 `progress_cb: Callable[[str, str, float], None] | None = None`，调用约定 `progress_cb(step, message, progress)`。
  - 插入 6 个上报点：
    ```
    progress_cb("ort_import",         "loading ONNX Runtime and CuPy",      0.05)
    progress_cb("warmup_key",         "checking GPU and model fingerprint", 0.10)
    # marker_matches → return（无需更多上报）
    progress_cb("import_matter",      "loading matting module",             0.20)
    progress_cb("rvm_session_create", "compiling RVM model for your GPU",   0.35)
    # Matter() 完成
    total_runs = sum(max(1, runs_per_shape) for _ in key.shapes)
    done = 0
    for shape in key.shapes:
        for i in range(max(1, runs_per_shape)):
            done += 1
            progress_cb("rvm_verify_run",
                       f"warming up RVM ({done}/{total_runs})",
                       0.5 + 0.4 * done / total_runs)
            matter._run_rvm_iobinding_from_dev(x)
            ...
    progress_cb("write_marker", "saving GPU cache marker", 0.95)
    write_marker(...)
    ```
- `main.py:233`：
  - 定义 `def _on_warmup_progress(step, message, progress)`：
    ```
    set_startup_phase(
        "warming", message,
        step=step, step_index=index_by_step[step], step_total=6,
        progress=progress,
        eta_sec=prediction.estimate_sec,
        elapsed_sec=time.perf_counter() - warmup_start,
        cold=prediction.cold,
        ...其他诊断字段
    )
    ```
  - 注意：这里手动写 `elapsed_sec` 会覆盖心跳线程的累加，所以每次真实事件来了 progress 都会"对齐"一下；心跳继续在事件之间填空。
  - 把 `warmup_gpu_runtime_cache(..., progress_cb=_on_warmup_progress)` 传入。
- 删除 `main.py:221-224` 关于"can't emit per-substep events"的注释，改成简短说明这些事件由 progress_cb 驱动。

**验证**：
- 用 `STARTUP_GPU_WARMUP_FORCE=1` 强制冷启动，观察 6 条 set_startup_phase 调用都有效。
- /status 抓包能看到 step 字段在 6 个值之间切换。

---

### 改造 ④ — predict_warmup_state 之前先发"探测硬件中"

**目标**：填补 `process started` 到 `first-time GPU initialization` 的 5~10 秒空窗（cupy+ort 首次 import + nvidia-smi）。

**修改**：
- `main.py:148` 在 `predict_warmup_state()` 调用之前插入：
  ```
  set_startup_phase("warming", "detecting GPU and ORT versions",
                    step="predict_probe", step_index=0, step_total=6,
                    progress=0.02)
  ```
- `prediction = predict_warmup_state()` 之后那条已有的 set_startup_phase 把 step 改为 `step="predict_done"`、`step_index=1`、`step_total=6` 与改造 ③ 的总步数对齐。

**验证**：
- 启动初期 UI 显示"正在探测显卡和 ORT 版本…"约 5~10 秒。

---

### 改造 ⑤ — UI 端按 step 映射 i18n 翻译

**目标**：中文界面不再显示英文 message。

**修改**：
- `ui/widgets/startup_overlay.py`：
  - 新增模块级映射：
    ```python
    _STEP_I18N_KEYS = {
        "predict_probe":      "startup.step.predict_probe",
        "predict":            "startup.step.predict",
        "predict_done":       "startup.step.predict_done",
        "ort_import":         "startup.step.ort_import",
        "warmup_key":         "startup.step.warmup_key",
        "import_matter":      "startup.step.import_matter",
        "rvm_session_and_runs":"startup.step.rvm_session_and_runs",
        "rvm_session_create": "startup.step.rvm_session_create",
        "rvm_verify_run":     "startup.step.rvm_verify_run",
        "write_marker":       "startup.step.write_marker",
        "firewall":           "startup.step.firewall",
        "ssdp":               "startup.step.ssdp",
        "http_starting":      "startup.step.http_starting",
    }
    ```
  - `apply_status` 中构造 `_base_message`：
    - 若 `step` 命中 `_STEP_I18N_KEYS`：用 `i18n.t(key)` 替代英文 message。
    - rvm_verify_run 携带 `done/total_runs` 情况：i18n 模板里用 `{current}/{total}`，从 message 字符串解析或新增结构化字段（推荐后者：服务端把 done/total 单独放进 `set_startup_phase(... fields)` 的扩展字段，UI 端读出来格式化）。
  - 英文 message 仅作 `details` 面板兜底显示，主体不显示。
- 三个 i18n json 新增对应 `startup.step.*` 键：
  - zh_CN 示例：
    - `startup.step.predict_probe`: "正在探测显卡和 ORT 版本…"
    - `startup.step.predict_done`: "首次启动：正在初始化显卡环境"
    - `startup.step.ort_import`: "正在加载 ONNX Runtime 和 CuPy…"
    - `startup.step.warmup_key`: "正在生成显卡缓存指纹…"
    - `startup.step.import_matter`: "正在加载抠图模块…"
    - `startup.step.rvm_session_create`: "正在为本机显卡编译 RVM 模型（首次启动需要 1~3 分钟）…"
    - `startup.step.rvm_verify_run`: "正在校验抠图模型（{current}/{total}）…"
    - `startup.step.write_marker`: "正在保存显卡缓存…"
    - `startup.step.firewall`: "正在配置防火墙规则…"
    - `startup.step.ssdp`: "正在启动 DLNA 发现服务…"
    - `startup.step.http_starting`: "正在绑定网络端口…"

**验证**：
- 中文界面冷启动全程中文文案，无英文残留。
- 进入 rvm_verify_run 时显示"正在校验抠图模型（3/6）…"等动态计数。

---

## 4. 实施顺序

1. **第一步（P0 必做，< 30 分钟）**：改造 ①
   - 修 `MainWindow` connect + 新增 `_on_startup_error`，新增 `StartupOverlay.show_bootstrapping_hint`。
   - 三个 i18n 文件加 `startup.bootstrapping` / `startup.bootstrapping_long`。
   - 立刻消除"窗口 A 假死感"。

2. **第二步（P0 必做，< 30 分钟）**：改造 ②
   - 在 `utils/startup_status.py` 加心跳线程，在 `main.py` warmup 入口/出口 start/stop。
   - 立刻让进度条/ETA 持续推进，不再 marquee 死动画。

3. **第三步（P1 推荐，~1 小时）**：改造 ③
   - 改 `warmup_gpu_runtime_cache` 签名加 callback，插入 6 个上报点。
   - 改 `main.py` 定义 `_on_warmup_progress` 并传入。
   - 进度条真正有 6 段语义化推进。

4. **第四步（P2 优化，< 10 分钟）**：改造 ④
   - main.py 在 predict 前后加两次 set_startup_phase。

5. **第五步（P2 优化，~30 分钟）**：改造 ⑤
   - UI 端 step → i18n 映射，三份 i18n 加 13 条新键。
   - 实现 rvm_verify_run 的 done/total 结构化字段。

---

## 5. 改造后体验对照表

| 时刻 | 改造前 | 改造后 |
|---|---|---|
| T0（点击启动） | "正在连接服务器进程…"  marquee | "正在启动服务器" + marquee |
| T0+2s（仍连不上 :8299） | 同上，无变化 | 切换至 "服务进程正在启动中，首次启动可能需要 1~3 分钟…" |
| Ts（:8299 起） | "process started…" 几秒 | "正在探测显卡和 ORT 版本…" 2% |
| Ts+5s | "first-time GPU initialization" + ETA | "首次启动：正在初始化显卡环境" + "RTX 2080（计算能力 7.5）" + ETA 45s, 10% |
| Ts+6s | "loading ONNX Runtime CUDA and running warmup" + marquee（之后一动不动） | "正在加载 ONNX Runtime 和 CuPy…" 5% → "正在生成显卡缓存指纹…" 10% |
| Ts+8s | 同上 | "正在加载抠图模块…" 20% |
| Ts+10s | 同上 | "正在为本机显卡编译 RVM 模型（首次启动需要 1~3 分钟）…" 35%，心跳推进进度条 |
| Ts+30s | 同上 | "正在校验抠图模型（3/6）…" 70% |
| Ts+45s | 同上 | "正在保存显卡缓存…" 95% |
| Ts+46s | overlay 关闭 | "正在配置防火墙规则…" → "正在启动 DLNA 发现服务…" → "正在绑定网络端口…" → overlay 关闭 |

---

## 6. 风险与回滚

- **改造 ②** 心跳线程要小心：必须用 `daemon=True`，停机优先级低于 set_startup_phase 的真实更新，否则会撞掉 progress=1.0 的终态。建议 stop_heartbeat 在 phase 切到 `warmed` / `failed` / `listening` 时强制调用一次。
- **改造 ③** 把 `from pipeline.matting import Matter` 拉到 progress_cb 之后的位置时，确认 Matter() 构造里 ORT session 是同步阻塞——目前确实如此，调 progress_cb 应放在 `Matter()` 之前（"compiling RVM model" 文案预告即将进入耗时阶段），而非之后。
- **改造 ⑤** 若 step 未命中映射表，回落到英文 message 显示，确保新增 step 不会让 UI 显示空字符串。
- **回滚**：每个改造点独立，可单独 revert；P0 两项即使没有 P1/P2 也是一次明显的体验提升。

---

## 7. 验收清单

- [ ] 关掉 :8299 端口（或干脆别启动 ServerProcess）观察 UI 在 2s 后切到 bootstrapping。
- [ ] `STARTUP_GPU_WARMUP_FORCE=1` 强制冷启动，逐项核对 6 段进度文案中文化、进度条单调递增。
- [ ] 正常冷启动一次（< 60s），验收 overlay 全程中文 + 进度有意义。
- [ ] 热启动（marker_matches）核对：① 不显示 rvm_session_create / rvm_verify_run；② 心跳不会让进度卡在 95%（写 marker 后真实事件覆盖至 100%）。
- [ ] 主动取消（cancel 按钮）：心跳线程必须停掉，不能继续后台写 /status。
- [ ] 三种语言（zh_CN / en_US / ja_JP）全部覆盖测试。
