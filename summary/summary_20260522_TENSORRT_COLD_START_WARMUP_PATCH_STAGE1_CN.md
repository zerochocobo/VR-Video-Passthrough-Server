# TensorRT 冷启动 Warmup —— 阶段 1 可执行 Patch 描述

日期：2026-05-22

关联文档：
- 计划原文：`summary/summary_20260522_TENSORRT_COLD_START_WARMUP_PLAN_CN.md`
- 我方回复：`summary/summary_20260522_TENSORRT_COLD_START_WARMUP_PLAN_REPLY_CN.md`
- 本文：阶段 1 的精确改动定位 + 改法描述，交付给开发人员落地。

## 0. 一句话目标

把 `warmup_gpu_runtime_cache()` 内 throwaway `Matter()` 改成全局 singleton `get_matter()`，并在 warmup 期间显式触发 batch=1 / batch=2 static TensorRT session 加载，让首次播放请求不再支付 `ort_run=1133.6ms` 的冷启动成本。

## 1. 改动范围

仅一个文件：
- `utils/gpu_runtime_cache.py`

不动：
- `pipeline/matting.py`（singleton 已存在，无需新增）
- `main.py`（启动顺序已正确）
- `ui/services/trt_warmup_process.py`（这是另一条独立的 engine cache 构建子进程链路，与首帧 session load 无关）
- `pipeline/pynv_stream.py` / `tools/offline_*`

## 2. 精确改动位置

文件：`utils/gpu_runtime_cache.py`
函数：`warmup_gpu_runtime_cache(force, timeout_sec, runs_per_shape)`
当前问题代码区间：第 490-519 行。

### 改动 A：import 由 `Matter` 改为 `get_matter`

**改前**（490 行）：
```python
        from pipeline.matting import Matter
```

**改后**：
```python
        from pipeline.matting import get_matter
```

> **⚠️ 落地补丁（2026-05-22 验证补充）**：`from pipeline.matting import get_matter` 这一行 **必须位于 `config.MATTING_WARMUP_RUNS = 0` 之后**。
>
> 原因：`Matter.__init__` 内部读取的是模块级全局 `config.MATTING_WARMUP_RUNS`，没有形参隔离。如果 `pipeline.matting` 在更早的 import 链中已被加载并触发了首个 `Matter()` 构造（典型现象：启动日志出现 `alpha #1 ... ort_run=808.9ms` + `matting warmup: runs=1 elapsed=1589.2ms`），则那次构造仍以默认 warmup_runs > 0 跑了一遍自带 warmup。
>
> 推荐顺序：
> ```python
>         old_warmup = config.MATTING_WARMUP_RUNS
>         config.MATTING_WARMUP_RUNS = 0
>         try:
>             from pipeline.matting import get_matter
>             matter = get_matter()
>         finally:
>             config.MATTING_WARMUP_RUNS = old_warmup
> ```
>
> 根治方案（阶段 2）：给 `Matter.__init__` 增加 `warmup_runs: int | None = None` 形参，warmup 路径显式传 0；本文档不在阶段 1 内落地。

### 改动 B：实例化改为复用 singleton

**改前**（492-497 行）：
```python
        old_warmup = config.MATTING_WARMUP_RUNS
        config.MATTING_WARMUP_RUNS = 0
        try:
            matter = Matter()
        finally:
            config.MATTING_WARMUP_RUNS = old_warmup
```

**改后**：
```python
        old_warmup = config.MATTING_WARMUP_RUNS
        config.MATTING_WARMUP_RUNS = 0
        try:
            matter = get_matter()
        finally:
            config.MATTING_WARMUP_RUNS = old_warmup
        log.info(
            "[WARMUP] matter singleton id=%s static_trt_available=%s providers=%s",
            id(matter),
            getattr(matter, "_rvm_static_trt_available", None),
            list(matter.sess.get_providers()),
        )
```

> 说明：`get_matter()` 内部已有 double-check + lock，幂等；首次调用构造，后续返回同一对象。

> 日志来源：文件顶部已 `log = logging.getLogger(__name__)`（已存在，无需新增 import）。如果该文件没有 `log` 对象，开发人员请在文件头部加 `import logging; log = logging.getLogger(__name__)`，落地前先确认。

### 改动 C：在 shape 循环前显式预加载 static TRT sessions

**改前**（502-509 行）：
```python
        verify_start = 0.0
        verify_elapsed = 0.0
        for shape in key.shapes:
            batch, channels, h, w = shape
            if channels != 3:
                continue
            x = cp.zeros((batch, channels, h, w), dtype=matter.input_dtype)
            matter.reset_state()
```

**改后**：
```python
        verify_start = 0.0
        verify_elapsed = 0.0

        # 显式触发 static TensorRT session 加载，避免首次真实请求时再 lazy-load。
        # _get_trt_static_session 内部有缓存，重复 key 不会重建。
        for shape in key.shapes:
            batch, channels, h, w = shape
            if channels != 3:
                continue
            t_static = time.perf_counter()
            sess = matter._get_trt_static_session(int(batch), int(h), int(w))
            log.info(
                "[WARMUP] static_trt preload batch=%d shape=%dx%d loaded=%s elapsed_ms=%.1f",
                int(batch), int(h), int(w),
                sess is not None,
                (time.perf_counter() - t_static) * 1000.0,
            )

        for shape in key.shapes:
            batch, channels, h, w = shape
            if channels != 3:
                continue
            x = cp.zeros((batch, channels, h, w), dtype=matter.input_dtype)
            matter.reset_state()
```

> 说明：`_get_trt_static_session(batch, h, w)` 是 `Matter` 私有方法，位置 `pipeline/matting.py:1103`。当 static TRT 路径不可用时返回 `None`（不抛错），日志会反映 `loaded=False`。

### 改动 D：循环结束后清理 recurrent state

**改前**（519-520 行）：
```python
                if i == max(1, runs_per_shape) - 1:
                    verify_elapsed += time.perf_counter() - t0

        count, size = _cache_stats(Path(env.cuda_cache_path))
```

**改后**：
```python
                if i == max(1, runs_per_shape) - 1:
                    verify_elapsed += time.perf_counter() - t0

        # 防止 warmup 残留的 RVM recurrent state 污染首次真实请求。
        matter.reset_state()
        log.info("[WARMUP] reset_state after warmup; first request will start from zero state.")

        count, size = _cache_stats(Path(env.cuda_cache_path))
```

> 说明：warmup 内部已经在每个 shape 进入循环时调用了 `matter.reset_state()`，但循环出口处 r1o..r4o 仍是 warmup 最后一帧的 0-input 推理结果。改为 singleton 后这些状态会被首次真实请求继承，必须显式清零。

## 3. 最终预期 diff（合并后的连续片段）

`utils/gpu_runtime_cache.py` 第 490-533 行将变为：

```python
        from pipeline.matting import get_matter

        old_warmup = config.MATTING_WARMUP_RUNS
        config.MATTING_WARMUP_RUNS = 0
        try:
            matter = get_matter()
        finally:
            config.MATTING_WARMUP_RUNS = old_warmup
        log.info(
            "[WARMUP] matter singleton id=%s static_trt_available=%s providers=%s",
            id(matter),
            getattr(matter, "_rvm_static_trt_available", None),
            list(matter.sess.get_providers()),
        )

        import cupy as cp
        import pipeline.matting as matting_mod

        verify_start = 0.0
        verify_elapsed = 0.0

        for shape in key.shapes:
            batch, channels, h, w = shape
            if channels != 3:
                continue
            t_static = time.perf_counter()
            sess = matter._get_trt_static_session(int(batch), int(h), int(w))
            log.info(
                "[WARMUP] static_trt preload batch=%d shape=%dx%d loaded=%s elapsed_ms=%.1f",
                int(batch), int(h), int(w),
                sess is not None,
                (time.perf_counter() - t_static) * 1000.0,
            )

        for shape in key.shapes:
            batch, channels, h, w = shape
            if channels != 3:
                continue
            x = cp.zeros((batch, channels, h, w), dtype=matter.input_dtype)
            matter.reset_state()
            for i in range(max(1, runs_per_shape)):
                t0 = time.perf_counter()
                matter._run_rvm_iobinding_from_dev(x)
                stream = getattr(matting_mod, "_CUDA_STREAM", None)
                if stream is not None:
                    stream.synchronize()
                else:
                    cp.cuda.Stream.null.synchronize()
                if i == max(1, runs_per_shape) - 1:
                    verify_elapsed += time.perf_counter() - t0

        matter.reset_state()
        log.info("[WARMUP] reset_state after warmup; first request will start from zero state.")

        count, size = _cache_stats(Path(env.cuda_cache_path))
```

其余代码（marker_path / WarmupLock / key 校验 / marker 写盘）保持原样。

## 4. 落地前自检清单

- [ ] 文件顶部已存在 `log = logging.getLogger(__name__)`。若无，先补一行 `import logging` + `log = logging.getLogger(__name__)`。
- [ ] 文件顶部已 `import time`。已存在，确认即可。
- [ ] `from pipeline.matting import Matter` 改为 `from pipeline.matting import get_matter`。
- [ ] 全文搜索 `Matter()` 在本文件应只剩 0 处（全部改成 `get_matter()`）。
- [ ] 不要把 `from pipeline.matting import Matter` 完全删除——如果其它地方还需要类型注解可保留 `from pipeline.matting import Matter, get_matter`；当前函数体内未用 `Matter` 类型，建议只 import `get_matter`。

## 5. 验证步骤（必须按顺序）

### 5.1 静态检查

```powershell
.\.venv\Scripts\python.exe -m compileall utils\gpu_runtime_cache.py
```

预期：无语法错误。

### 5.2 单测

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_settings.py
```

如果有 `tests\test_gpu_runtime_cache.py`，一并跑。无需新增测试（singleton 行为很难纯单测覆盖）。

### 5.3 冷启动观测

清掉 marker 触发完整 warmup：

```powershell
del /F /Q "%LOCALAPPDATA%\PTServer\gpu_warmup_marker.json"
.\.venv\Scripts\python.exe main.py
```

> marker 路径以 `utils/gpu_runtime_cache.py` 内 `env.marker_path` 实际值为准；若用户机器为另一路径，按实际删除。

**启动日志预期**：
- `[WARMUP] matter singleton id=<n> static_trt_available=True providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- `[WARMUP] static_trt preload batch=1 shape=1024x1024 loaded=True elapsed_ms=<n>`
- `[WARMUP] static_trt preload batch=2 shape=1024x1024 loaded=True elapsed_ms=<n>`
- `[WARMUP] reset_state after warmup; first request will start from zero state.`

### 5.4 首次请求观测

启动后立刻发起一次 alpha SBS 播放请求，看 `debug_output/server.log`：

| 指标 | 修复前 | 修复后预期 |
|---|---|---|
| 首次 `alpha #1 ort_run` | 1133.6 ms | < 30 ms |
| frame 30 fps | 17.60 | ≥ 35 |
| frame 60 fps | 23.22 | ≥ 38 |
| frame 120 fps | 29.22 | ≥ 39 |
| frame 600 fps | 39.03 | 同 |
| frame 1110 fps | 39.90 | 同 |

`static_trt=True` 必须在第一帧诊断行就出现，而不是在请求中段才看到 `static TensorRT RVM session loaded` 日志。

### 5.5 singleton 复用确认

若开发人员想精确确认 singleton 没被丢，可在第 5.4 步的请求 worker 启动处临时加一行日志，比较 `id(get_matter())` 与启动日志里 `[WARMUP] matter singleton id=<n>` 是否一致。验证通过后撤回临时日志。

## 6. 回滚

仅一个文件，单 commit。回滚 `git checkout HEAD~1 -- utils/gpu_runtime_cache.py` 即可。无数据迁移、无 marker 兼容性问题（marker 内容未变）。

## 7. 已知风险与处置

| 风险 | 触发条件 | 处置 |
|---|---|---|
| singleton 在 warmup 阶段抛错后保持半初始化态 | `_get_trt_static_session` 抛非预期异常 | `get_matter()` 抛错会让 startup 失败；当前实现的 `_get_trt_static_session` 在不可用时返回 `None` 不抛错，正常路径安全。仍建议用 try/except 把 `matter._get_trt_static_session(...)` 包起来，异常时只 log warning，不中断 warmup。 |
| 显存提前占用 | 始终触发，singleton 持有 ORT session + TRT engine | 用户机器 2×RTX 2080，本来稳态就持有这部分显存。无新增。 |
| warmup 期间 UI 卡顿 | warmup 是阻塞调用，启动期 UI 不可用 | 现状本就如此，本 patch 未改变阻塞性。`startup_status` 已有 phase 推进，可由 UI 显示 progress。 |
| 后续 fingerprint 变化导致 marker miss + 二次构造 | model/input_size/downsample 变更 | `get_matter()` 不感知 fingerprint 变化，旧 singleton 不会自动失效。**当前不在阶段 1 范围内**——若 UI 允许在运行期切换 input_size，必须在切换处显式调用 `pipeline.matting._singleton = None`。阶段 1 不做。 |

## 8. 不在本 patch 内（明确排除）

- 不引入后台线程做 warmup
- 不引入全局 session 池（多 stream 共享 ORT session）
- 不改 `Matter` 类签名
- 不改 `_get_trt_static_session` 可见性（仍保留 `_` 前缀，warmup 直接访问私有方法是受控范围）
- 不做 NVENC/NVDEC preflight
- 不做 UI progress UI 改动

## 9. 完成判定

满足以下全部条件即视为阶段 1 完成：

1. 启动日志包含 4 行 `[WARMUP]` 标识。
2. 首次真实播放请求 `alpha #1 ort_run < 30 ms`。
3. frame 30 fps ≥ 35。
4. 二次启动（marker hit）不进入改动 B/C/D 路径，零额外日志。
5. `pytest tests\test_settings.py` 通过。
6. PyInstaller 打包后行为一致（手动跑一次 onedir 验证）。

完成后再评估是否做阶段 2（UI 状态曝光）或 NVENC preflight。
