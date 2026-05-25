# 冷启动 UI 进度反馈改造实施计划

日期：2026-05-21（原版）/ 2026-05-25（更新，校准当前代码现状）
分支：master
目标：解决"启动后弹窗显示『正在连接服务器进程』几分钟不动 → 然后突然消失"的体验问题。让用户在整个冷启动过程中（最长可达 150s）持续看到有意义的进度更新。

---

## 0. 与原计划的差异说明（2026-05-25 校准）

原版 5 项改造在当前 master 上的实际进度：

| # | 原改造点 | 当前进度 | 说明 |
|---|---|---|---|
| ① | UI 端补 `poller.error` 处理 | ❌ 未做 | `ui/main_window.py:88-89` 只 connect 了 `updated` / `finished`，`error` 信号悬空 |
| ② | startup_status 心跳线程 | ❌ 未做 | `utils/startup_status.py` 中无心跳相关代码 |
| ③ | 拆开 `warmup_gpu_runtime_cache` + progress_cb | ✅ 已完成（实现方式不同） | `utils/gpu_runtime_cache.py:463+` 内部直接调用 `set_startup_phase`，不再使用 progress_cb 模式。step 命名变更为 `matter_singleton` / `static_trt_preload` / `ort_iobinding_runs` / `composite_jit` / `reset_state` 共 5 段 |
| ④ | predict 之前先发"探测硬件中" | ❌ 未做 | `main.py:214-279` 仍是 `set_startup_phase("starting", "process started")` 一发完就直接进入 `predict_warmup_state()` |
| ⑤ | UI 端按 step 映射 i18n | ⚠️ 部分完成 | `startup_overlay.py:329-334` 已通过 `startup.step.{step}` 模式自动查表；三份 i18n 已含 12 个 step key；但服务端 message 仍是英文且 `apply_status` 优先把英文 message 拼进 `_base_message` 显示给中文用户看 |

剩余待办：**①、②、④** 完整保留；**③** 不再改 callback 接口，只需补 `ort_iobinding_runs` 的 done/total 细分进度；**⑤** 只需把 UI 端的 message 改成不显示英文（让 i18n 化的 step 文案承担主显，message 仅入 details 面板）。

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
| B | `set_startup_phase("warming", "starting GPU warmup", step="matter_singleton", progress=0.1)` 之后 | `apply_status` 进入 `warming + elapsed<=0.1 + progress<=0.11` 特例分支（`startup_overlay.py:265-271`），进度条切 indeterminate marquee；服务端在该 phase 内会继续发 `static_trt_preload`/`ort_iobinding_runs`/`composite_jit`/`reset_state`，但单段内（特别是 `ort_iobinding_runs` 的多 shape × 多 run）依然完全无中间反馈 | `get_matter()` 单点阻塞：ORT JIT → static TRT 引擎加载 → `for shape × runs: _run_rvm_iobinding_from_dev` → composite_jit 多几何体预热 → `reset_state`，单段内 0 次状态回写 |

代码现状的自我承认（已存在的 TODO/注释）：
- `ui/widgets/startup_overlay.py:265-271` 用 marquee 兜底，承认"feels frozen to a non-technical user"。
- `main.py` 中已不再有"can't emit per-substep events"的注释——因为 `warmup_gpu_runtime_cache` 已经拆出 5 个 step；但 step 之间的长尾仍无心跳。

### 文案根因
- `ui/translations/zh_CN.json:131` `"startup.connecting": "正在连接服务器进程"` 是 `StartupOverlay.reset()` 的默认底字。只要 poller 没成功响应过一次，就被 `_tick_ellipsis` 持续加点显示。
- 服务端 `set_startup_phase` 的 `message` 字段写死英文（如 `"loading matting runtime"`、`"running GPU inference warmup"`），`apply_status` 在 `startup_overlay.py:218-224` 直接把英文 message 拼入 `_base_message` 显示，未做 i18n 化。step 文案虽已国际化，但仍只显示在 step_label 一行，主体仍英文。

---

## 2. 改造总览（剩余四个改造点 + 优先级）

| # | 改造点 | 目标 | 文件 | 估算 LOC | 优先级 |
|---|---|---|---|---|---|
| ① | UI 端补 `poller.error` 处理 | 干掉窗口 A 的"假死" | `ui/main_window.py`, `ui/widgets/startup_overlay.py`, 3 个 i18n json | ~40 | P0 |
| ② | startup_status 心跳线程 | 5 个 step 之间的长尾阻塞也持续推进，UI 进度条不再死 | `utils/startup_status.py`, `main.py`, `utils/gpu_runtime_cache.py` | ~60 | P0 |
| ③' | `ort_iobinding_runs` 细分 done/total | 把当前 5 段中最长的一段（多 shape × 多 run）再切成 N 次心跳 | `utils/gpu_runtime_cache.py`, `ui/widgets/startup_overlay.py`, 3 个 i18n json | ~30 | P1 |
| ④ | predict 之前先发"探测硬件中" | 填补 `process started` → `first-time GPU initialization` 的几秒空窗 | `main.py` | ~10 | P2 |
| ⑤' | message 不再用英文兜底主显 | 中文界面不再显示英文 message | `ui/widgets/startup_overlay.py` | ~10 | P2 |
| ⑥ | TRT vs CUDA 路径差异化展示 | 同样 5 段在两条路径下意义不同，文案/step_total/ETA 也应不同 | `main.py`, `utils/gpu_runtime_cache.py`, `ui/widgets/startup_overlay.py`, 3 个 i18n json | ~80 | P1 |
| ⑦ | 长阻塞段分级安抚文案 | 用户开始怀疑卡死（30s/60s/120s/180s）时主动喊话"这是正常的，还在编译" | `ui/widgets/startup_overlay.py`, 3 个 i18n json | ~50 | P0 |

P0 = 必须做（< 100 LOC 立刻解决主要痛点）；P1 = 强烈推荐（消除最后一段长尾静默 + 区分 TRT/CUDA 路径）；P2 = 锦上添花。

---

## 3. 详细方案

### 改造 ① — UI 端补 poller.error 处理

**目标**：UI 弹窗后 2 秒内若仍连不上 :8299，提示用户"服务进程正在启动中，首次启动可能需要 1~3 分钟"。

**修改**：
- `ui/main_window.py`：
  - `__init__` 在 `self.status_poller.finished.connect(...)` 之后增加：
    - `self.status_poller.error.connect(self._on_startup_error)`
    - `self._poll_error_streak = 0`
    - `self._poll_first_success = False`
  - 新增 `_on_startup_error(message: str)`：累加 `_poll_error_streak`；若 ≥ 4（≈ 2s）且 `not _poll_first_success` 且 overlay 仍可见：调用 `self.startup_overlay.show_bootstrapping_hint()`；若 ≥ 60（≈ 30s）：调用 `show_bootstrapping_hint_long()`。
  - `_on_startup_status` 开头补 `self._poll_first_success = True; self._poll_error_streak = 0`。
  - `_open_startup_overlay` 重置这两个字段。
- `ui/widgets/startup_overlay.py`：
  - `reset()` 内已有 `_base_message`，不动。
  - 新增 `show_bootstrapping_hint()` / `show_bootstrapping_hint_long()`：把 `_base_message` 替换为对应 i18n key 翻译；保留 ellipsis 动画；进度条保持 marquee（`setRange(0, 0)`）。
- i18n（`zh_CN.json` / `en_US.json` / `ja_JP.json`）：
  - 新增 `startup.bootstrapping`（zh: "服务进程正在启动中，首次启动可能需要 1~3 分钟…"）
  - 新增 `startup.bootstrapping_long`（zh: "服务进程仍在启动。GPU 首次冷启动需要为本机显卡编译内核，请耐心等待。"）

**验证**：
- 模拟 :8299 不响应（如把 `STARTUP_STATUS_PORT` 改成不存在的）：UI 弹窗 2 秒内文字切换为 bootstrapping，30 秒后切换为 bootstrapping_long。
- 正常启动：第一次成功响应后文本不再被 error handler 替换。

---

### 改造 ② — startup_status 心跳线程

**目标**：在 set_startup_phase 之间补"假装在动"，让 ETA 行（"已用 X 秒，预计还需 Y 秒"）和进度条持续推进，避免 marquee 哑动画。当前最痛的两段：
1. `matter_singleton` → `static_trt_preload`：`get_matter()` 内 ORT session 创建 + JIT，可达 30~120s。
2. `ort_iobinding_runs` 内的多 shape × `runs_per_shape` 循环，单段内可能 5~60s 无回写。

**修改**：
- `utils/startup_status.py`：
  - 新增模块级 `_heartbeat_thread: threading.Thread | None`、`_heartbeat_stop: threading.Event | None`、`_heartbeat_baseline: float`、`_heartbeat_ceiling: float`、`_heartbeat_eta: float`、`_heartbeat_started_at: float`。
  - 新增 `start_heartbeat(eta_sec: float, baseline_progress: float, ceiling_progress: float = 0.95) -> None`：
    - 启动 daemon 线程，每 500ms：
      - `with _lock`: `_state["elapsed_sec"] = time.time() - heartbeat_started_at`；`_state["updated_at"] = time.time()`。
      - 按 `min(ceiling_progress, baseline_progress + (1-baseline_progress) * elapsed/eta_sec)` 推进 `_state["progress"]`，**永远不超过 ceiling**（留给真实事件覆盖到 1.0）。
  - 新增 `stop_heartbeat()`：set event + join(1.0)。
  - `set_startup_phase(phase, message, **fields)` **不**自动触发心跳（避免破坏现有行为），由 `main.py` / `warmup_gpu_runtime_cache` 显式 start/stop。
- `main.py`：
  - `predict_warmup_state()` 之后那条 set_startup_phase 调用之后，立即调 `start_heartbeat(prediction.estimate_sec, baseline_progress=0.1, ceiling_progress=0.95)`。
  - `warmup_gpu_runtime_cache(...)` 调用前 stop 一次心跳（重新校准），调用后 stop_heartbeat()。
  - `try/except` 失败分支也要 stop_heartbeat()。
- `utils/gpu_runtime_cache.py`（可选优化，若 ② 单独不够）：
  - `_warmup_resident_matter_runtime` 内每个 step 切换前先 `start_heartbeat` / 切换时 `stop_heartbeat`，把每段 eta 拆开（参考 `_ETA_*` 默认值平均分布）。

**验证**：
- 启动观察 UI：进度条不再长时间 marquee，是真实从 10% 缓慢爬到 ~95%；ETA 行每秒更新一次。
- 即使 warmup 阻塞 90s，UI 数字也在动。
- 主动取消（cancel）后，心跳线程必须停掉。

---

### 改造 ③' — `ort_iobinding_runs` 细分 done/total

**背景**：原计划改造 ③ 中的 6 段拆分已由 `gpu_runtime_cache.py` 用不同 step 名称实现（`matter_singleton` / `static_trt_preload` / `ort_iobinding_runs` / `composite_jit` / `reset_state`）。剩余的痛点只剩 `ort_iobinding_runs` 这一段：当前在 `gpu_runtime_cache.py:544-558` 是一个嵌套循环，对每个 shape 跑 `runs_per_shape` 次，全程不发任何状态。

**修改**：
- `utils/gpu_runtime_cache.py:_warmup_resident_matter_runtime`：
  - 在 `for shape in warmup_key.shapes:` 外层先算 `total_runs = sum(max(1, runs_per_shape) for s in warmup_key.shapes if s[1] == 3)`。
  - 进入 `ort_iobinding_runs` 那个外层 `set_startup_phase` 后，循环内每完成一次 `matter._run_rvm_iobinding_from_dev(x) + stream.synchronize()`：
    ```python
    done += 1
    set_startup_phase(
        "warming",
        "running GPU inference warmup",
        step="ort_iobinding_runs",
        step_index=3,
        step_total=step_total,
        progress=(3.0 / step_total) + (1.0 / step_total) * (done / total_runs),
        run_done=done,
        run_total=total_runs,
    )
    ```
  - 注意 `run_done` / `run_total` 是新加的结构化字段（`set_startup_phase` 会原样存入 `_state`，已经向前兼容）。
- `ui/widgets/startup_overlay.py`：
  - `_step_text(step)` 拿到的是基础翻译，对 `step == "ort_iobinding_runs"` 时，若 `_last_status` 中有 `run_done` / `run_total`，把它格式化拼接（如 "预热 GPU 推理（3/6）"）。
- i18n 三份文件：
  - 现有 `startup.step.ort_iobinding_runs` 文案保持不变；UI 端在拼接 `{done}/{total}` 时不必新增 key（也可新增 `startup.step.ort_iobinding_runs_progress` 模板，二选一，按口味）。

**验证**：
- `STARTUP_GPU_WARMUP_FORCE=1` 强冷，观察 UI step_label 在 `ort_iobinding_runs` 阶段显示 `3/5 预热 GPU 推理（1/6）` → `（2/6）` → ... 单调递增。
- 进度条在 60%~80% 段位平滑爬升，不再是 marquee。

---

### 改造 ④ — predict_warmup_state 之前先发"探测硬件中"

**目标**：填补 `process started` 到 `first-time GPU initialization` 的 5~10 秒空窗（cupy+ort 首次 import + nvidia-smi）。

**修改**：
- `main.py:214` 在 `if config.STARTUP_GPU_WARMUP:` 进入后、`predict_warmup_state()` 调用之前插入：
  ```python
  set_startup_phase(
      "warming",
      "detecting GPU and ORT versions",
      step="predict_probe",
      step_index=0,
      step_total=startup_step_total,
      progress=0.02,
  )
  ```
- `ui/widgets/startup_overlay.py:_STEP_KEYS` 在首位前插入 `"predict_probe"`（或单独走 i18n fallback 也行）。
- 三个 i18n json 增加 `startup.step.predict_probe`：
  - zh_CN：`"正在探测显卡和 ORT 版本…"`
  - en_US：`"Detecting GPU and ORT versions…"`
  - ja_JP：`"GPU と ORT のバージョンを検出中…"`

**验证**：
- 启动初期 UI 显示"正在探测显卡和 ORT 版本…"约 5~10 秒，然后切到 "首次启动：正在初始化显卡环境"。

---

### 改造 ⑥ — TRT vs CUDA 路径差异化展示

**背景**：当前 `_validate_tensorrt_provider()`（`main.py:89-118`）根据 `utils/trt_manifest` 的指纹比对结果，把 `config.ONNX_PROVIDERS` 切换为 `TRT_PROVIDER_CHAIN`（cache 命中）或回落到 `["CUDAExecutionProvider", "CPUExecutionProvider"]`。之后 `warmup_gpu_runtime_cache` **不感知**这个切换，5 个 step 一律照走：

- `static_trt_preload`（`gpu_runtime_cache.py:496-534`）：循环对每个 shape 调 `matter._get_trt_static_session(...)`。
  - **TRT 路径**：从 `runtime_cache/trt/*.engine` 加载预构建引擎，每 shape 1~3s，**真做事**。
  - **CUDA 路径**：`_rvm_static_trt_available=False`，函数立刻返回 None；step 也照样发 `set_startup_phase`，UI 显示 "加载 TensorRT 引擎" 1~2 秒后跳走，**完全是空跑**，对用户具误导性。
- `ort_iobinding_runs`（`gpu_runtime_cache.py:536-558`）：
  - **TRT 路径**：走 `_run_rvm_static_trt_iobinding_from_dev`，主体已被 TRT subgraph 吸收，单次推理 < 100ms，总耗时几秒。
  - **CUDA 路径**：走 CUDA EP JIT，首次运行可达 2 分钟（Blackwell sm_120 无 cubin）。
- ETA 估算（`gpu_runtime_cache._ETA_FIRST_RUN_NORMAL_SEC = 45.0` / `_ETA_FIRST_RUN_KNOWN_SLOW_SEC = 150.0`）只覆盖 CUDA 路径冷启动；TRT 路径的"引擎加载 + 几次 verify run"通常 5~15s，没有专门 bucket。

注：TRT 引擎的**构建**（首次可达 5~30 分钟）发生在 `TrtCacheDialog` + `trt_warmup_process` 子进程中，由用户主动触发，**不属于本计划范围**。本节只覆盖"引擎已就绪、服务器进程启动时如何展示"。

**目标**：让 UI 在两条路径下显示不同文案、不同 step 数、不同 ETA，让用户知道"我现在走的是 TRT 快路径还是 CUDA 慢路径"。

**修改**：
1. **`utils/gpu_runtime_cache.py`**：
   - 在 `ColdStartReport` dataclass 加 `provider_kind: str` 字段（值为 `"trt"` / `"cuda"` / `"cpu"` / `""`）。在 `predict_warmup_state()` 末尾根据 `config.ONNX_PROVIDERS` 首元素填充：`"TensorrtExecutionProvider"` → `"trt"`，`"CUDAExecutionProvider"` → `"cuda"`，否则 `""`。
   - 新增 ETA 默认值 `_ETA_TRT_ENGINE_LOAD_SEC = 12.0`；`predict_warmup_state()` 中当 `provider_kind == "trt"` 且 cache_hit / key_changed 路径都使用该值（覆盖 normal/known_slow bucket）。
   - `warmup_gpu_runtime_cache._warmup_resident_matter_runtime` 改造：
     - 计算 `step_total = (1 + has_trt + 1 + has_composite + 1) + has_nvenc`，其中 `has_trt = bool(_rvm_static_trt_available)`。当前固定为 5 + nvenc，改为动态。
     - 当 `_rvm_static_trt_available` 为 False 时，**不发** `static_trt_preload` 这条 set_startup_phase，且后续 step_index 序号相应前移。
     - 改 `static_trt_preload` 文案为更精确的 "加载 TensorRT 引擎缓存"；改 `ort_iobinding_runs` 文案分两版：TRT 路径 "校验 TensorRT 引擎"（短），CUDA 路径 "预热 GPU 推理（首次需编译内核）"（长）。可通过新加 `provider_kind` 字段透传给 UI 让 UI 选择文案；或服务端直接根据 `_rvm_static_trt_available` 写不同 message。
2. **`main.py`**：
   - `predict_warmup_state()` 后，`set_startup_phase` 调用透传 `provider_kind=prediction.provider_kind` 到 /status 字段。
   - `startup_step_total` 计算改为 `4 + (1 if trt_ready else 0) + (1 if config.USE_PYNV and config.NVENC_PREFLIGHT_ENABLE else 0)`。其中 `trt_ready` 可从 `_validate_tensorrt_provider` 返回值（需把它从 None 改为返回 bool）或直接看 `config.ONNX_PROVIDERS[0] == "TensorrtExecutionProvider"` 判断。
3. **`ui/widgets/startup_overlay.py`**：
   - `apply_status` 读取 `provider_kind` 字段。当 `provider_kind == "trt"` 且 `phase == "warming"`：
     - 标题改为 `startup.title_trt_loading`（zh: "正在加载 TensorRT 引擎缓存"），覆盖 `title_first_run` / `title_first_run_slow`。
     - 已有的 known_slow 提示 (`startup.hint_known_slow`) **抑制**（TRT 路径不应再吓唬用户"1-3 分钟"）。
   - `_STEP_KEYS` 不变；UI 端只关心"我收到 step 就展示"，跳过的 step 不会出现。
4. **3 个 i18n json**：
   - 新增 `startup.title_trt_loading`（zh: "正在加载 TensorRT 引擎缓存"）。
   - `startup.step.static_trt_preload` 文案微调，更明确："加载 TensorRT 引擎"（已有，建议保留）。
   - 新增 `startup.step.ort_iobinding_runs_trt`（zh: "校验 TensorRT 引擎"）作为 TRT 路径专用 step 文案；UI 在 `_step_text` 内当 `provider_kind=="trt" and step=="ort_iobinding_runs"` 时优先用此 key。

**验证**：
- TRT 引擎已就绪，正常启动：UI 标题显示 "正在加载 TensorRT 引擎缓存"；step_total 显示 5（含 nvenc）/ 4（不含 nvenc）；step 序列：matter_singleton → static_trt_preload → ort_iobinding_runs(校验) → composite_jit → reset_state；总耗时 < 15s。
- 删 TRT cache（或改 fingerprint）→ 回落 CUDA：UI 标题显示 "首次启动：正在初始化显卡环境"；step_total 显示 4（含 nvenc）/ 3（不含）；step 序列跳过 static_trt_preload；总耗时 30~150s。
- 切换两次：TRT 标题不被 known_slow hint 污染；CUDA 标题保留 known_slow hint（如硬件命中）。

---

### 改造 ⑦ — 长阻塞段分级安抚文案（"不是卡死，请放心等待"）

**背景**：心跳线程（改造 ②）能让进度条持续蠕动，run_done/total（改造 ③'）能让用户看到"5/6 → 6/6"在前进，但**用户的心理时间**是另一码事：
- 0~20s：进度条动 = 安心。
- 20~60s：进度条几乎不动（CUDA EP JIT 内部单段就要几十秒）→ 用户开始怀疑"是不是哪里出错了"。
- 60~150s：Blackwell sm_120 + ORT 1.21 这种 known_slow 组合下，单段就要 1~3 分钟 → 用户开始想点"取消"。
- 150s+：用户认定卡死，强杀进程，下次还是卡死，体验崩塌。

当前 `startup.hint_known_slow` 是**一次性的静态提示**，对 known_slow 硬件一开始就显示，但用户看了一遍后注意力就移开了；对**普通硬件**（如 RTX 2080）反而不显示，可这些硬件冷启动也要 30~60s。

**目标**：根据 elapsed_sec 在 hint_label 处显示**分级安抚文案**，随时间推移**升级措辞**和**佐证细节**（GPU 名、ORT 版本、缓存路径），让用户即使在 progress 蠕动很慢时也持续被安抚。

**修改**：
- `ui/widgets/startup_overlay.py`：
  - 新增模块级阈值表：
    ```python
    _REASSURANCE_TIERS = [
        # (elapsed_threshold_sec, i18n_key, severity)
        (20.0,  "startup.reassure.t20",  "info"),    # 20s+: "首次启动需要为本机显卡编译内核，请耐心等待"
        (45.0,  "startup.reassure.t45",  "info"),    # 45s+: "仍在编译。RTX 2080 通常需要 30~60s，Blackwell 显卡需要 1~3 分钟。"
        (90.0,  "startup.reassure.t90",  "warn"),    # 90s+: "继续编译中。这是首次启动的一次性开销，下次启动 < 5s。"
        (180.0, "startup.reassure.t180", "warn"),    # 180s+: "已用 X 秒，仍在进行中。如确认怀疑卡死，可点击「复制硬件报告」反馈。"
    ]
    ```
  - `apply_status` 在更新 ETA 行之后新增一段：根据 `elapsed_sec` 选出**最高**已跨过的阈值，把对应 i18n 翻译写入 `hint_label`（已有控件，复用），覆盖 known_slow hint（仅当无 known_slow，或时间超过 known_slow 文案的"信息密度"）。
  - 文案模板中可用 `{gpu}` / `{ort}` / `{cc}` / `{elapsed}` 占位符，UI 端从已有 status 字段格式化注入，比通用安抚词更有说服力。
  - 阈值越大，hint_label 边框颜色越向"暖橙"倾斜（已有 `#FFF7E0/#EBC97A` → 改为更明显的边框宽度或脉冲提示），让用户视觉感受到"系统知道你等久了"。
  - phase 切到 `warmed` / `firewall` / 之后立刻清掉 hint_label，避免安抚文案残留到"已经过 warmup 阶段"。
- `ui/services/startup_status_poller.py`：无需改动，elapsed_sec 已经在 /status 心跳里持续推进。
- 3 个 i18n json：新增 4 条键，例如 zh_CN：
  - `startup.reassure.t20`: "首次启动需要为您的显卡 {gpu} 编译 GPU 内核，请耐心等待…"
  - `startup.reassure.t45`: "仍在编译中。已用 {elapsed} 秒。RTX 2080/3000/4000 系列一般 30-60 秒，RTX 50 系列需要 1-3 分钟。"
  - `startup.reassure.t90`: "继续编译中（已用 {elapsed} 秒）。这是**首次启动的一次性开销**，编译结果会保存到缓存，下次启动只需几秒。请勿强制关闭程序，否则下次还要重新编译。"
  - `startup.reassure.t180`: "已用 {elapsed} 秒。如确认怀疑卡死，请点击下方「复制硬件报告」并联系技术支持。否则继续等待，进程仍在工作中。"
  - en_US / ja_JP 同步翻译。

**与改造 ①、⑥ 的边界**：
- 改造 ①（bootstrapping）只在 :8299 还**未通**时生效；本改造在 :8299 已通、phase=warming 时生效。两者互斥。
- 改造 ⑥ 的 TRT 路径通常 < 15s 完成，t20 阈值都触发不到，自然不会显示。只在 CUDA 路径长阻塞下生效，与设计目标完全契合。
- 已有的 known_slow hint 与本改造的 t45+ 文案**择一显示**（优先 known_slow，因为它更精确说明根因；t90+ 后再覆盖为更强语气的 t90）。

**验证**：
- 用环境变量 `STARTUP_GPU_WARMUP_FORCE=1` 强冷在 RTX 2080 上：30s 时出现 t20 文案；普通硬件能完成在 60s 内，t45 文案出现 5~15s 后被 warmed 清掉。
- 模拟 Blackwell（或人为 sleep 2 分钟）：依次看到 t20 → t45 → t90 → t180，每条文案的 GPU 名 / elapsed 占位符都正确填充。
- phase 切到 warmed/firewall 后 hint_label 立刻消失。

---

### 改造 ⑤' — message 不再用英文兜底主显

**背景**：现状 `apply_status` 在 `startup_overlay.py:217-224`：
```python
friendly: list[str] = []
if message:
    friendly.append(message)
if gpu:
    friendly.append(self.i18n.t("startup.gpu_label").format(gpu=gpu, cc=cc or "?"))
self._base_message = "\n".join(friendly) if friendly else self.i18n.t("startup.connecting")
```
直接把服务端英文 `message` 推到主显，对中文用户不友好。step_label 一行虽然有翻译，但 message_label 仍然英文。

**修改**：
- `ui/widgets/startup_overlay.py:apply_status`：
  - 把 `if message: friendly.append(message)` 改为：先调 `_step_text(step)` 得到本地化文案，若命中（即不是 raw step 名）则推到 friendly 列表；只有当 step 为空或映射失败时才 fallback 到英文 `message`。
  - 显示 step_text 后 step_label 行可改为只显示进度数字 `step_index/step_total`（避免重复）；或保持现状（重复但显式），按 UI 评审定。
- 英文 message 始终进入 details 面板 raw 行（已有，`_format_details` 已含 "message" 字段）。
- i18n 不再新增 key，所有 step.* 文案已齐备。

**验证**：
- 中文界面冷启动全程中文主显，无英文残留。
- details 面板仍能看到原始英文 message，便于反馈/排障。

---

## 4. 实施顺序

1. **第一步（P0 必做，< 30 分钟）**：改造 ①
   - 修 `MainWindow` connect + 新增 `_on_startup_error`，新增 `StartupOverlay.show_bootstrapping_hint`。
   - 三个 i18n 文件加 `startup.bootstrapping` / `startup.bootstrapping_long`。
   - 立刻消除"窗口 A 假死感"。

2. **第二步（P0 必做，< 30 分钟）**：改造 ②
   - 在 `utils/startup_status.py` 加心跳线程，在 `main.py` warmup 入口/出口 start/stop。
   - 立刻让进度条/ETA 持续推进，不再 marquee 死动画。
   - 如有余力，再在 `gpu_runtime_cache._warmup_resident_matter_runtime` 内按段 start/stop（更细的进度）。

3. **第三步（P1 推荐，~30 分钟）**：改造 ③'
   - 改 `_warmup_resident_matter_runtime` 中 `ort_iobinding_runs` 段：每次 _run_rvm_iobinding_from_dev 后写 `run_done`/`run_total` 进度。
   - 改 `startup_overlay._step_text` 在 step 命中 ort_iobinding_runs 时拼 `(done/total)`。

4. **第四步（P1 推荐，~45 分钟）**：改造 ⑥
   - `ColdStartReport` 加 `provider_kind`，`predict_warmup_state` 填充。
   - `_warmup_resident_matter_runtime` 动态 step_total + 跳过空跑的 static_trt_preload。
   - UI 端 `apply_status` 根据 provider_kind 切换标题 / hint / step 文案。
   - 3 份 i18n 加 `startup.title_trt_loading` + `startup.step.ort_iobinding_runs_trt`。

5. **第五步（P0 必做，~30 分钟）**：改造 ⑦
   - `startup_overlay.py` 加 `_REASSURANCE_TIERS` + 在 `apply_status` 末尾按 elapsed_sec 选条目写 hint_label。
   - 3 份 i18n 加 4 条 `startup.reassure.t{20,45,90,180}` 文案。
   - 解决"用户在长阻塞段开始怀疑卡死"的心理问题。

6. **第六步（P2 优化，< 10 分钟）**：改造 ④
   - main.py 在 predict 前加一次 set_startup_phase，i18n 加 `startup.step.predict_probe`。

7. **第七步（P2 优化，~15 分钟）**：改造 ⑤'
   - 改 `apply_status` 让 step 翻译承担主显，message 仅入 details。

---

## 5. 改造后体验对照表

### 5.1 CUDA 路径（TRT 缓存不可用 / 用户未启用 TRT）

| 时刻 | 改造前 | 改造后 |
|---|---|---|
| T0（点击启动） | "正在连接服务器进程…"  marquee | "正在连接服务器进程…" marquee |
| T0+2s（仍连不上 :8299） | 同上，无变化 | 切换至 "服务进程正在启动中，首次启动可能需要 1~3 分钟…" |
| Ts（:8299 起） | "process started" 几秒 | "正在探测显卡和 ORT 版本…" 2% |
| Ts+5s | "first-time GPU initialization" + ETA | "首次启动：正在初始化显卡环境" + "RTX 2080（计算能力 7.5）" + ETA 45s, 10% |
| Ts+6s | "starting GPU warmup" → "loading matting runtime"（英文，2 秒一闪）然后 marquee 长时间不动 | "加载抠像运行时" 25%（心跳推进，step 1/4） |
| Ts+25s | 仍 marquee | step 不变；**hint_label 弹出 t20 文案**："首次启动需要为您的显卡 RTX 2080 编译 GPU 内核，请耐心等待…" |
| Ts+30s | 仍 marquee | "预热 GPU 推理（3/6）" 70%（step 2/4，心跳 + run_done/total） |
| Ts+45s | 仍 marquee | "预热合成内核" 85%（step 3/4） → "收尾" 95%（step 4/4）；**hint_label 升级为 t45 文案** |
| Ts+90s（Blackwell） | 仍 marquee | step 不变；**hint_label 升级为 t90 文案**："这是首次启动的一次性开销，下次启动只需几秒，请勿强制关闭程序" |
| Ts+46s | overlay 切到 "GPU runtime warmup complete" → 几秒后突然关闭 | "检查防火墙" → "启动投屏发现服务" → "启动媒体服务" → overlay 关闭 |

### 5.2 TRT 路径（TRT 引擎已就绪）

| 时刻 | 改造前 | 改造后 |
|---|---|---|
| T0~Ts+5s | 同上，UI 不知道走 TRT 还是 CUDA | "正在加载 TensorRT 引擎缓存" + ETA 12s, 10%（标题区分） |
| Ts+6s | "loading matting runtime" → "loading TensorRT engines"（英文） | "加载抠像运行时" 20%（step 1/5） |
| Ts+8s | "loading TensorRT engines" 几秒 → 跳走 | "加载 TensorRT 引擎" 40%（step 2/5，真做事 1~3s/shape） |
| Ts+11s | "running GPU inference warmup" → marquee 几秒 | "校验 TensorRT 引擎（1/2）" → "（2/2）" 60%（step 3/5） |
| Ts+13s | "warming composite kernels" 几秒 | "预热合成内核" 80%（step 4/5） |
| Ts+14s | "resetting warmup state" → "GPU runtime warmup complete" | "收尾" 95%（step 5/5）→ "检查防火墙" → ... → overlay 关闭 |
| 提示 | 通用 known_slow 警告（误报） | 不显示 known_slow 提示（TRT 路径不应触发） |

---

## 6. 风险与回滚

- **改造 ②** 心跳线程要小心：必须用 `daemon=True`，停机优先级低于 set_startup_phase 的真实更新，否则会撞掉 progress=1.0 的终态。建议在 `main.py` 内 `warmed` / `failed` / `nvenc_preflight` / `firewall` / `ssdp` / `http_starting` 切换前都强制 `stop_heartbeat()` 一次（最稳）。
- **改造 ②** 由 `set_startup_phase` 写入的 elapsed_sec 会被心跳每 500ms 覆盖。要么完全让心跳负责 elapsed_sec（推荐），要么 `main.py` 调真实 set_startup_phase 时不传 elapsed_sec。两者择一，避免抖动。
- **改造 ③'** `run_done/run_total` 字段是新增，旧版 UI 收到时会忽略（`_state` 透传），向后兼容。
- **改造 ⑤'** 若 step 未命中映射表，回落到英文 message 显示，确保新增 step 不会让 UI 显示空字符串。
- **改造 ⑥** `_validate_tensorrt_provider` 可能在中途把 ONNX_PROVIDERS 从 TRT 回落到 CUDA（cache 损坏），`predict_warmup_state` 必须在它之后调用才能拿到正确的 `provider_kind`；`main.py` 当前顺序已正确（`_validate_tensorrt_provider` 在 if 块外、`predict_warmup_state` 在 if 块内），改造时保持此顺序。
- **改造 ⑥** step_total 动态会引入"4 vs 5"的不一致：进度条若按 step_index/step_total 推断时分母不同，UI 端要小心展示一致性（建议优先用 server 端写入的 `progress` 字段，不再自行 step_index/step_total 二次计算）。
- **改造 ⑦** 与改造 ⑥ 的 known_slow hint 互相覆盖；优先级排序：t180 > t90 > known_slow > t45 > t20 > 默认空。即"已知慢硬件 + 等久了"时显示更强语气的 t90/t180，而不是软绵绵的 known_slow。
- **改造 ⑦** elapsed_sec 来自 /status 心跳，若改造 ② 没做，则 elapsed_sec 不会推进，t20/t45 文案永远不会出现 —— 这是**依赖关系**：⑦ 必须在 ② 之后或同时实施。
- **回滚**：每个改造点独立，可单独 revert；P0 两项即使没有 P1/P2 也是一次明显的体验提升。

---

## 7. 验收清单

- [ ] 关掉 :8299 端口（或干脆别启动 ServerProcess）观察 UI 在 2s 后切到 bootstrapping，30s 后切到 bootstrapping_long。
- [ ] `STARTUP_GPU_WARMUP_FORCE=1` 强制冷启动，逐项核对 5 段（+ 探测 + 预测）进度文案中文化、进度条单调递增、心跳每 500ms 推进。
- [ ] 正常冷启动一次（< 60s），验收 overlay 全程中文 + 进度有意义。
- [ ] 热启动（marker_matches）核对：① ort_iobinding_runs 段也跑过 verify_pass，run_done/total 仍能正常显示；② 心跳不会让进度卡在 95%（写 marker 后真实事件 warmed 覆盖至 5/5）。
- [ ] 主动取消（cancel 按钮）：心跳线程必须停掉，不能继续后台写 /status。
- [ ] 三种语言（zh_CN / en_US / ja_JP）全部覆盖测试。
- [ ] 关 NVENC（`USE_PYNV=0`）时 CUDA 路径 `step_total=4`、TRT 路径 `step_total=5`；开 NVENC 时分别为 5 / 6。两种模式下进度条都能跑到 100%。
- [ ] TRT 引擎就绪状态：UI 标题为 "正在加载 TensorRT 引擎缓存"，不显示 known_slow 警告，step_label 不含 "预热 GPU 推理（首次需编译内核）" 而显示 "校验 TensorRT 引擎"。
- [ ] 删除 `runtime_cache/trt/manifest.json` 后再启动：自动回落 CUDA，UI 标题为 "首次启动：…"，step_label 跳过 static_trt_preload。
- [ ] 分级安抚验收：人为 sleep 200s 模拟极慢冷启动，逐项见到 t20 → t45 → t90 → t180 四级文案；占位符 `{gpu}` / `{elapsed}` 正确填充；phase 切到 warmed 后 hint_label 立刻清掉。
- [ ] 改造 ⑦ 单独跑（不开 ②）应能从 status.elapsed_sec 推进；改造 ⑦ 与 ② 同时开应不冲突（心跳负责推进 elapsed_sec，⑦ 只读不写）。
