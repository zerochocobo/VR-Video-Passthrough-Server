# TensorRT 首帧 Warmup 优化 —— 评审与执行方案

日期：2026-05-22

关联文档：
- 计划原文：`summary/summary_20260522_TENSORRT_COLD_START_WARMUP_PLAN_CN.md`
- 同期闭环：`summary/summary_20260522_OFFLINE_RVM_THROUGHPUT_GAP_RESOLUTION_CN.md`
- 本文：评审与最终执行方案

## 1. 计划方案逐项可行性评审

### 方案 A：server 启动后台预热 TRT static sessions
- 方向正确，但**落地点错**。不需要新建后台线程 + 锁。
- 启动 warmup 当前已是阻塞同步流程，且 `WarmupLock` 已保证唯一性。

### 方案 B：按实际播放路径预热 batch=2
- `utils/gpu_runtime_cache.py:251 default_warmup_shapes()` 已含 batch=2，前提是 `MATTING_SPLIT_SBS=True and MATTING_SBS_BATCH=True`。
- 已经做了，缺的是显式日志校验 batch=2 走的是 static_trt 路径。

### 方案 C：全局 session / Matter runtime 池
- 不需要做。
- 架构上 `Matter` 已经是全局单例（`pipeline/matting.py:2961-2971 get_matter()`），`self._trt_static_sessions` 按 `(batch,h,w)` 字典缓存。
- RVM recurrent state 跨请求隔离是另一个独立议题，不属于本次冷启动范围。

### 方案 D：PyNv/NVDEC/NVENC 轻量预初始化
- 收益小于 ORT/TRT。日志显示 `worker start → first real bitstream` 约 1 秒，远小于 `ort_run=1133.6ms` 的占比。
- 留作阶段 4 评估。

## 2. 关键发现 —— 计划未覆盖的根因

启动 warmup 其实已经做了 batch=1 + batch=2，**但结果被丢弃了**。

### 2.1 现有 warmup 已经覆盖两路 batch

`utils/gpu_runtime_cache.py:495-519 warmup_gpu_runtime_cache()`：

```python
matter = Matter()                       # 本地变量，新建 Matter
for shape in key.shapes:                # 默认 [(1,3,1024,1024), (2,3,1024,1024)]
    x = cp.zeros((batch, 3, h, w), ...)
    matter.reset_state()
    for i in range(runs_per_shape):
        matter._run_rvm_iobinding_from_dev(x)
        stream.synchronize()
```

`_run_rvm_iobinding_from_dev` 第 1927 行立刻调 `_run_rvm_static_trt_iobinding_from_dev`，后者第 1982 行调 `_get_trt_static_session(batch, h, w)` —— 两路 static TRT session 在启动时确实已经加载并跑过一次。

### 2.2 但 warmup 用的 `matter` 是局部变量

请求来到时走的是 `pipeline/matting.py:2965` 的全局单例：

```python
_singleton: Matter | None = None

def get_matter() -> Matter:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Matter()     # 重新构建
    return _singleton
```

`warmup_gpu_runtime_cache` 中的 `matter = Matter()` 从未赋给 `_singleton`。函数返回后 `matter` 被 GC，`_singleton` 仍为 `None`。

请求触达 `get_matter()` 时再 `Matter()`：
- 主 ORT session 重新构建（CUDA EP）
- 两路 static TRT session **重新 lazy-load**（第一次 `_get_trt_static_session` 调用时）
- 首帧 `ort_run=1133.6ms` 就是 batch=2 session 第一次激活 + 引擎反序列化 + IOBinding 首次绑定的代价

CUDA driver 级缓存（PTX→SASS）跨进程持久化，所以部分工作省到了；但 **per-`InferenceSession` 的引擎/上下文/binding 必须每个 Matter 实例重做**。

### 2.3 结论

计划方案 A/B 描述的"工程级改造"实际上是**一处单例复用 bug**。改 ~30 行可解决，不需要新建后台线程、不需要任务编排、不需要 session 池。

## 3. 执行方案（以阶段为序）

### 阶段 1：单例复用修复（必做，高优先级）

#### 步骤 1.1：让 warmup 注入单例

**位置**：`utils/gpu_runtime_cache.py:495` `warmup_gpu_runtime_cache()` 函数内。

**改法**：将 `matter = Matter()` 改为 `matter = get_matter()`。`get_matter()` 已线程安全（`_singleton_lock`），调用即注册单例。

**辅助保留**：临时将 `config.MATTING_WARMUP_RUNS = 0` 包裹起来，防止 `Matter.__init__` 自带 warmup 与外层 for 循环重复跑（现有逻辑已这么做）。

#### 步骤 1.2：显式 force-load 两路 static TRT session

**位置**：同函数内，shape 循环之后。

**改法**：增加：
- `matter._get_trt_static_session(1, size, size)`
- `matter._get_trt_static_session(2, size, size)`

`_run_rvm_iobinding_from_dev` 已经间接调用，此处显式调用是为了诊断日志清晰；同时如果未来 inference 路径变化，显式调用保证 session 一定驻留。

#### 步骤 1.3：诊断日志

**位置**：同函数 + `Matter._get_trt_static_session`。

**输出**：
- warmup 完成时：`static_trt batch=1 loaded=True elapsed=X.Xs / batch=2 loaded=True elapsed=Y.Ys`
- 第一次请求 `get_matter()` 命中单例时：`get_matter() singleton_reused=True`
- 若未命中（fallback）：`get_matter() singleton_reused=False reason=...`

#### 步骤 1.4：状态清理

warmup 跑完所有 shape inference 后调 `matter.reset_state()`，清掉 warmup 帧造成的 RVM recurrent state 残留。

### 阶段 2：非 TRT 路径同步收益（**无需额外动作，附带收益**）

详见第 4 节。

### 阶段 3：UI 进度暴露（中优先级）

在阶段 1 数据验证有效后再做。

#### 步骤 3.1：扩展 startup_status 步骤
- step 1: `ort_session_and_runs`（现有）
- step 2: `trt_static_b1`（新）
- step 3: `trt_static_b2`（新）
- step 4: `warmed`（现有）

#### 步骤 3.2：UI 端透传
`/status` 端点已存在，UI overlay 按现有机制读取即可。

### 阶段 4：方案 D 评估（可选，低优先级）

#### 步骤 4.1：测量
单独统计 `worker start → first real bitstream` 时间。如果 < 500ms，不做。

#### 步骤 4.2：实施
在启动 warmup 末尾新建一个最小尺寸 NVENC encoder 并立即 `EndEncode()`，强制 NVENC 驱动 DLL 加载 + 上下文初始化。

不做 NVDEC preflight —— 真实 PyNv decoder 创建依赖文件元数据，启动时没有可用文件。

## 4. 非 TensorRT 路径要不要同时做

### 4.1 结论

**阶段 1 的单例复用修复同时覆盖 TRT 路径与非 TRT (CUDA EP) 路径，无需额外动作**。

### 4.2 依据

- `Matter.__init__` 在 `pipeline/matting.py:1023` 构建的主 `self.sess` 用 CUDA EP（当 static TRT 可用时），非 TRT 用户用的就是这条主 session。
- `_run_rvm_iobinding_from_dev` 在 `pipeline/matting.py:1928` 判断 static 不可用时直接回落到主 session 路径（line 1930-1979），同样走 IOBinding。
- 当前 warmup 跑的 `matter._run_rvm_iobinding_from_dev(x)` 对两种路径都走完一遍。单例复用后，两种路径的首帧成本都被吸收到启动阶段。

### 4.3 非 TRT 路径独有的小残留

- **AlphaPacker 内部 CuPy kernel JIT**：alpha 路径首帧 pack 时 CuPy `RawKernel` 第一次编译。一般 50-150ms，远小于 ORT 1133ms。
- **`Matter.composite_green_gpu_*` 路径的 CuPy kernel**：同理。

这部分 warmup 不会覆盖（warmup 只调 `_run_rvm_iobinding_from_dev`，不走完整 composite/pack 流程）。

### 4.4 是否要扩展 warmup 覆盖 composite/pack 路径

**不建议在阶段 1 一起做**，原因：

1. 收益数量级低（ORT 1133ms vs CuPy JIT ~50-150ms），阶段 1 不解决这部分也已经把"首播慢"感受消除。
2. 扩展到 composite/pack 需要构造一组与真实播放维度一致的输入（NV12 GPU buffer、SBS 拆分、alpha pack 输出尺寸），warmup 代码复杂度显著上升。
3. CuPy JIT 产物有 `CUPY_CACHE_DIR` 持久化，第二次冷启动后自然消失，本身就是一次性成本。

**建议节奏**：阶段 1 上线后跑实测，如果 alpha 路径首帧 composite/pack 还有可见抖动，再考虑在 warmup 末尾追加一次 composite + pack 的 dummy 调用。

## 5. 验证标准（阶段 1 验收）

跑同一条 `[VenusReality]Hannah02-8K.mp4`，关闭 adaptive FPS，对比修复前后：

| 指标 | 修复前 | 修复后预期 | 判据 |
|---|---|---|---|
| 首次 RVM `ort_run` | 1133.6 ms | **< 30 ms** | 必须达成 |
| frame 30 累计 fps | 17.60 | **≥ 35** | 必须达成 |
| frame 60 累计 fps | 23.22 | **≥ 37** | 必须达成 |
| frame 120 累计 fps | 29.22 | **≥ 38** | 必须达成 |
| 稳态 fps（frame 600+） | 39.03 | 39.0 左右不变 | 不应回退 |
| `get_matter() singleton_reused` | — | `True` | 必须达成 |

只要首帧 `ort_run` 大幅回落 + `singleton_reused=True`，阶段 1 验收通过。

辅助验证：
```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_settings.py tests\test_offline_convert.py
git diff --check
```

## 6. 不要做的事

1. **不要新起后台线程跑 warmup**。当前主流程同步阻塞，`WarmupLock` 已保护。新线程与 `_singleton_lock` 可能死锁。
2. **不要在阶段 1 一起做方案 C（session 池）**。涉及 RVM recurrent state 隔离的独立设计问题。
3. **不要修改 `ui/services/trt_warmup_process.py`**。它解决的是 `.engine` 文件构建（首次安装/换 GPU），与本次主进程冷启动是两个层级。
4. **不要预热 batch=4 / batch=8 等其他规格**。仅 batch=1, 2 在用。
5. **不要在阶段 1 一次做完所有阶段**。先验证数据，再决定后续。
6. **不要扩展 warmup 覆盖 composite/pack（除非阶段 1 后实测残留抖动明显）**。

## 7. 风险清单

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| 单例提前创建导致显存常驻 | 阶段 1 上线后 | 仅在 `STARTUP_GPU_WARMUP=True` 时触发，用户主动配置 |
| warmup fp16 与运行时 fp16 不一致 | 配置漂移 | `fingerprint` 已含 `trt_fp16`，marker 不一致自动重 warmup |
| `Matter()` 在 main 进程构建失败但 marker 已写 | rare | 现有 except 包装，失败不写 marker |
| 单例线程不安全 | 多请求并发 | `_singleton_lock` 已保护构造；inference 阶段 ORT 1.25 支持并发 |
| frozen exe 下 DLL 路径未应用 | 打包环境 | `main.py:182 apply_runtime_dll_paths()` 已在 warmup 之前调用 |
| warmup 残留 recurrent state 污染首请求 | rare | 步骤 1.4 末尾调 `matter.reset_state()` |

## 8. 对计划"专家审阅 7 个问题"的回答

| 问题 | 回答 |
|---|---|
| Q1 ORT session 多 stream 共享安全 | 安全。当前单例已这么做。 |
| Q2 static TRT session 已被缓存 | 已缓存。`Matter._trt_static_sessions` 按 `(batch,h,w)` 字典。 |
| Q3 warmup 是否污染 RVM recurrent state | 会，但步骤 1.4 调 `reset_state()` 即可。 |
| Q4 batch=1 是否仍要预热 | 要。单眼路径（非 SBS、缩略图、debug 工具）仍走 batch=1。 |
| Q5 TRT context per-thread 限制 | ORT TRT EP 1.25 支持多线程 inference；`run_with_iobinding` 调用本身要外部串行。当前 worker 单线程，不受影响。 |
| Q6 显存是否可接受 | 5060 Ti 16GB 余量充足。8K SBS batch=2 input_size=1024 额外占用约 500MB-1GB。 |
| Q7 exe 打包顺序 | 已 OK。`main.py:182 apply_runtime_dll_paths()` 在 warmup 前调用。 |

## 9. 最终建议

1. **先只做阶段 1（约 30 行修改 + 日志）**，跑一次 `[VenusReality]Hannah02-8K.mp4` 验证首帧 `ort_run` 回落。
2. 同步覆盖 TRT 与非 TRT 路径（阶段 1 一并解决，无需为非 TRT 单独做）。
3. 数据回来后再决定阶段 3/4 是否做。绝大概率阶段 1 上线，"首播慢"感受就基本消除。
4. 阶段 2（非 TRT composite/pack JIT）按需评估，不预先做。
