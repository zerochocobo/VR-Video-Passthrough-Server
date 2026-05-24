# TensorRT 冷启动 Warmup —— 阶段 2 可执行 Patch 描述

日期：2026-05-22

关联文档：
- 阶段 1 落地：`summary/summary_20260522_TENSORRT_COLD_START_WARMUP_PATCH_STAGE1_CN.md`
- 计划原文：`summary/summary_20260522_TENSORRT_COLD_START_WARMUP_PLAN_CN.md`

## 0. 阶段 2 要做的两件事

1. **形参隔离**：给 `Matter.__init__` 加 `warmup_runs: int | None = None` 形参，根治阶段 1 暴露的 import-binding 陷阱。
2. **singleton 失效钩子**：提供 `invalidate_singleton()` 函数，给后续 UI/运行期切换 `MATTING_INPUT_SIZE` / `RVM_DOWNSAMPLE_RATIO` / `MATTING_MODEL_KIND` 等 fingerprint 字段时显式失效用。

> 注意：阶段 2 **不强求**调用方迁移。tools/ 与 ui/services/ 现存的 `config.MATTING_WARMUP_RUNS = 0` 写法保留兼容（它们都在子进程或一次性脚本里，副作用面有限），只有 `utils/gpu_runtime_cache.py` 这条主路径切换到新形参。

## 1. 根因（为什么阶段 1 的 try/finally 实际是「歪打正着」）

`pipeline/matting.py:40`：

```python
from config import (
    ...
    MATTING_WARMUP_RUNS,
    ...
)
```

这是 `from X import Y` 形式，会在 `pipeline.matting` 模块加载时**把当时的 `config.MATTING_WARMUP_RUNS` 值快照绑定**到 `pipeline.matting.MATTING_WARMUP_RUNS`。之后即使外部代码做 `config.MATTING_WARMUP_RUNS = 0`，`pipeline.matting.MATTING_WARMUP_RUNS` 仍然是快照值（默认 1）。

`pipeline/matting.py:1080`：

```python
self.warmup(MATTING_WARMUP_RUNS)
```

这里读的是 `pipeline.matting` 命名空间的快照，不是 `config` 的现值。

阶段 1 的 `try/finally` 块写法：

```python
old_warmup = config.MATTING_WARMUP_RUNS
config.MATTING_WARMUP_RUNS = 0
try:
    from pipeline.matting import get_matter   # ← 仅在「matting 尚未被加载」时才捕获 0
    matter = get_matter()
finally:
    config.MATTING_WARMUP_RUNS = old_warmup
```

为什么阶段 1「import 移到 `= 0` 之后」修复奏效：

- 如果 `pipeline.matting` 之前**没**被任何路径加载过：`from pipeline.matting import get_matter` 触发模块首次加载，此时 `from config import MATTING_WARMUP_RUNS` 捕获到的是已被改为 0 的值，`Matter.__init__` 看到的 `MATTING_WARMUP_RUNS = 0`，不跑 warmup。
- 如果 `pipeline.matting` **已经**被加载（比如 main.py 早期 import 链）：快照已是 1，无论 `config` 怎么改都没用，`Matter.__init__` 仍然会跑一次 warmup。

这个语义对依赖关系敏感、对加载顺序敏感、对将来的 import 重排极其脆弱。阶段 2 直接断掉这条依赖。

## 2. 改动范围

| 文件 | 改动 |
|---|---|
| `pipeline/matting.py` | `Matter.__init__` 加 `warmup_runs` 形参；`get_matter` 加 `warmup_runs` kwarg；新增 `invalidate_singleton()` |
| `utils/gpu_runtime_cache.py` | 移除 `config.MATTING_WARMUP_RUNS` 临时改写，改用 `get_matter(warmup_runs=0)` |

**不动**：tools/、ui/services/、http_app/、main.py、其他子进程脚本。

## 3. 精确改动描述

### 改动 1：`pipeline/matting.py` `Matter.__init__` 签名

**位置**：第 930 行。

**改前**：
```python
class Matter:
    def __init__(self, model_path: Path = MODEL_PATH, load_model: bool = True):
```

**改后**：
```python
class Matter:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        load_model: bool = True,
        warmup_runs: int | None = None,
    ):
```

### 改动 2：`pipeline/matting.py` `__init__` 末尾 warmup 调用

**位置**：第 1080 行。

**改前**：
```python
        self.warmup(MATTING_WARMUP_RUNS)
```

**改后**：
```python
        effective_runs = MATTING_WARMUP_RUNS if warmup_runs is None else int(warmup_runs)
        self.warmup(effective_runs)
```

> 说明：保留对模块级 `MATTING_WARMUP_RUNS` 的兼容（无显式传参时行为不变），同时允许 `get_matter(warmup_runs=0)` 等调用方显式覆盖。

### 改动 3：`pipeline/matting.py` `get_matter` 加 kwarg

**位置**：第 2965-2971 行。

**改前**：
```python
def get_matter() -> Matter:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Matter()
    return _singleton
```

**改后**：
```python
def get_matter(*, warmup_runs: int | None = None) -> Matter:
    """Return the process-wide Matter singleton, constructing it on first call.

    `warmup_runs` only takes effect on the very first construction; subsequent
    calls reuse the existing instance regardless of the argument value.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Matter(warmup_runs=warmup_runs)
    return _singleton


def invalidate_singleton() -> None:
    """Drop the global Matter singleton; next get_matter() rebuilds.

    Use when fingerprint-affecting config changes at runtime
    (MATTING_INPUT_SIZE, RVM_DOWNSAMPLE_RATIO, MATTING_MODEL_KIND, providers).
    Note: the old instance is only dereferenced here; its ORT session, TRT
    engines and CuPy buffers are released by GC + ORT shutdown. Callers that
    need deterministic teardown should keep no other references to it.
    """
    global _singleton
    with _singleton_lock:
        _singleton = None
```

> **kwarg-only**：`def get_matter(*, warmup_runs=...)` 强制关键字传参，避免后续误用位置参数。

> **`invalidate_singleton` 当前调用方为零**：本阶段仅落地函数与文档，等到 UI 真要做运行期切换时调用方再接。

### 改动 4：`utils/gpu_runtime_cache.py` 去掉 `config.MATTING_WARMUP_RUNS` 改写

**位置**：第 466-477 行（含阶段 1 已经做过的 `get_matter` 注入）。

**改前**（阶段 1 落地后的现状）：
```python
        old_warmup = config.MATTING_WARMUP_RUNS
        config.MATTING_WARMUP_RUNS = 0
        try:
            from pipeline.matting import get_matter
            matter = get_matter()
        finally:
            config.MATTING_WARMUP_RUNS = old_warmup
        log.info(
            "[WARMUP] matter singleton id=%s ...",
            ...
        )
```

**改后**：
```python
        from pipeline.matting import get_matter

        matter = get_matter(warmup_runs=0)
        log.info(
            "[WARMUP] matter singleton id=%s ...",
            ...
        )
```

> `config.MATTING_WARMUP_RUNS` 的临时改写整段删除。

> import 现在不再依赖位置，可以放回函数体顶端或保留在原位，都等价。建议放在 `with WarmupLock(...)` 块内首行，与原始风格一致。

## 4. 最终预期 diff（关键片段）

`pipeline/matting.py` 第 930 行：

```python
class Matter:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        load_model: bool = True,
        warmup_runs: int | None = None,
    ):
```

`pipeline/matting.py` 第 1080 行：

```python
        effective_runs = MATTING_WARMUP_RUNS if warmup_runs is None else int(warmup_runs)
        self.warmup(effective_runs)
```

`pipeline/matting.py` 第 2965 行起：

```python
def get_matter(*, warmup_runs: int | None = None) -> Matter:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Matter(warmup_runs=warmup_runs)
    return _singleton


def invalidate_singleton() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None
```

`utils/gpu_runtime_cache.py` 第 466 行区域：

```python
        from pipeline.matting import get_matter

        matter = get_matter(warmup_runs=0)
        log.info(
            "[WARMUP] matter singleton id=%s static_trt_available=%s providers=%s",
            id(matter),
            getattr(matter, "_rvm_static_trt_available", None),
            list(matter.sess.get_providers()),
        )
```

## 5. 落地前自检清单

- [ ] `pipeline/matting.py:930` 签名追加 `warmup_runs: int | None = None`。
- [ ] `pipeline/matting.py:1080` 改成读 `effective_runs`。
- [ ] `pipeline/matting.py:2965` `get_matter` 加 `*, warmup_runs=None` kwarg-only 形参。
- [ ] `pipeline/matting.py` 末尾新增 `invalidate_singleton()`。
- [ ] `utils/gpu_runtime_cache.py` 删除 `old_warmup = config.MATTING_WARMUP_RUNS` / `config.MATTING_WARMUP_RUNS = 0` / `try/finally` 三段；改为 `matter = get_matter(warmup_runs=0)`。
- [ ] 全文搜索 `Matter(warmup_runs=` 仅 1 处命中（`get_matter` 内部）。
- [ ] 全文搜索 `get_matter(warmup_runs=` 仅 1 处命中（`utils/gpu_runtime_cache.py` 内）。
- [ ] tools/、ui/services/、main.py 不动。

## 6. 验证步骤

### 6.1 静态检查

```powershell
.\.venv\Scripts\python.exe -m compileall pipeline\matting.py utils\gpu_runtime_cache.py
```

### 6.2 单测

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_settings.py
```

### 6.3 冷启动观测

清掉 marker 并启动 server：

```powershell
del /F /Q "%LOCALAPPDATA%\PTServer\gpu_warmup_marker.json"
.\.venv\Scripts\python.exe main.py
```

**启动日志必须不出现**：
- `alpha #1 ... ort_run=808.9ms`（这是 `Matter.__init__` 自带 warmup 的残留）
- `matting warmup: runs=1 elapsed=1589.2ms`

**启动日志应该只看到**：
- `matting warmup: runs=0 elapsed=...`（来自 `Matter.__init__` 调用 `warmup(0)`，实际跳过 warmup）或不出现 warmup 行
- 后续 4 行 `[WARMUP]` 标识（阶段 1 已有）

### 6.4 二次启动（marker hit）观测

第二次启动，verify 路径下 `Matter.__init__` 仍然走 `warmup_runs=0`：

预期：`elapsed=3.4s` 左右，与阶段 1 验证一致；不应出现 `elapsed=1589.2ms` 的额外 warmup。

### 6.5 import 顺序无敏感性测试（可选回归用）

在 `main.py` 中故意在 `warmup_gpu_runtime_cache` 之前显式 import：

```python
import pipeline.matting  # 故意提前
```

然后清 marker 重启。预期：仍然不出现 `runs=1 elapsed=1589.2ms`。这一步用来证明阶段 2 已经把 import 顺序敏感性彻底消除。验证完恢复 main.py 原状。

## 7. 已知风险与处置

| 风险 | 处置 |
|---|---|
| 现有 tools/ 与 ui/services/ 仍用 `config.MATTING_WARMUP_RUNS = 0` 模式 | 不影响。这些路径都是子进程或一次性脚本，自己起 `Matter()` 时未传 `warmup_runs`，所以走 fallback `MATTING_WARMUP_RUNS`。它们要嘛在子进程顶层就改 `config`（生效），要嘛是 import 前改的（也生效）。本阶段不强迫迁移。 |
| `invalidate_singleton()` 资源释放不彻底 | 文档已说明：旧实例的 ORT/TRT/CuPy 资源由 GC + ORT shutdown 释放。若未来要做精确释放，需要在 `Matter` 上加 `close()` 方法并由 `invalidate_singleton` 调用，本阶段不做。 |
| 调用方误用 `get_matter(warmup_runs=0)` 期望第二次也无 warmup | 文档 docstring 已明确：`warmup_runs` 只在首次构造生效。后续 caller 看 docstring 即可。 |
| 多线程同时调 `get_matter(warmup_runs=...)` 两边传值不同 | double-check + lock 保证只有一次构造；先到的传参生效。后到的传参被忽略——这是 singleton 语义，可接受。 |

## 8. 回滚

两个文件、两个 commit（建议拆分：matting.py 一个 commit，gpu_runtime_cache.py 另一个）。回滚靠 `git revert` 即可。无数据迁移、无 marker 兼容性问题。

## 9. 不在本 patch 内（明确排除）

- 不动 tools/ 与 ui/services/ 任何 `config.MATTING_WARMUP_RUNS = 0` 写法。
- 不实现 `Matter.close()` 资源释放。
- 不接入 `invalidate_singleton()` 到 UI 任何具体 control（UI 实际能切 input_size 时再接）。
- 不改 `MATTING_WARMUP_RUNS` 的环境变量/默认值/含义。

## 10. Commit 前置调整（2026-05-22 代码审核补充）

阶段 2 改动落地后、`git commit` 之前，根据实际代码审核结果，作如下调整记录与硬要求。

### 10.1 实际改动盘点

工作区 `git diff --stat HEAD` 显示 4 个文件改动：

| 文件 | 是否阶段 2 范围 | 处置 |
|---|---|---|
| `pipeline/matting.py` | ✅ 是 | 与阶段 2 一起 commit |
| `utils/gpu_runtime_cache.py` | ✅ 是 | 与阶段 2 一起 commit |
| `prompt/HANDOVER_20260522.md` | ✅ 是（HANDOVER 记录） | 与阶段 2 一起 commit |
| `pipeline/pynv_stream.py` | ❌ **不是阶段 2** | **必须拆出独立 commit** |

### 10.2 `pipeline/pynv_stream.py` 独立 commit 要求

工作区中 `pynv_stream.py` 含约 20 行无关改动，内容包括：

- `_stop_proc(...)` 调用新增 `close_pipes=False` kwarg（2 处）
- slate audio cache build 在 `interrupted/stop` 与「非中断失败」两种分支的处理拆分（原来挤在一个分支里）
- 对应日志格式更新（去掉 `interrupted=%s`，独立日志行）

**这部分不属于 TRT cold-start warmup 阶段 2**，必须单独 commit。建议 commit 顺序：

```text
commit 1 (阶段 2 主体):
  pipeline/matting.py
  utils/gpu_runtime_cache.py
  prompt/HANDOVER_20260522.md
  message: "TRT cold-start warmup Stage 2: isolate Matter warmup_runs parameter"

commit 2 (独立小修复):
  pipeline/pynv_stream.py
  message: 由开发人员根据 audio cache build 改动的真实意图填写
```

回滚粒度更细，将来定位问题不会被混淆。

### 10.3 实际改动与原 patch 描述的差异

开发人员落地时与本文档第 3 节存在以下偏差，**全部接受**，仅在此存档：

| 项 | 原 patch 描述 | 实际落地 | 评价 |
|---|---|---|---|
| `Matter.__init__(warmup_runs=...)` | 位置参数允许 | 位置参数允许 | 一致 |
| `get_matter(warmup_runs=...)` | `*, warmup_runs=None` kwarg-only | `warmup_runs=None` 普通位置参数 | **偏离**。风险等级低（`get_matter` 只有一个参数，混淆空间小）。不阻塞 commit，**留作下次 refactor 顺手收掉**。 |
| `effective_runs` 中间变量 + `int()` 转换 | 提议显式中间变量与 `int()` 强转 | 内联三元 `MATTING_WARMUP_RUNS if warmup_runs is None else warmup_runs` | 语义等价，可读性持平。**不阻塞 commit**。 |
| 新增 `invalidate_singleton()` 函数 | 提议落地函数，留 hook 给未来 | **未落地** | 开发人员判断为 YAGNI（无调用方）。**接受**。**但要求在 HANDOVER 加一行 trip-wire**（见 10.4）。 |
| `_warmup_resident_matter_runtime` 内嵌函数包装 | 未涉及 | 已重构为内嵌函数 | 阶段 1 → 阶段 2 期间的隐性重构，更紧凑，接受。 |

### 10.4 HANDOVER 追加 trip-wire（commit 1 必含）

由于阶段 2 未落地 `invalidate_singleton()`，必须在 `prompt/HANDOVER_20260522.md` 当次阶段 2 段落里追加一条防踩坑记录：

```markdown
- **Trip-wire for future runtime fingerprint switching**:
  if any future code path allows in-process changes to MATTING_INPUT_SIZE,
  RVM_DOWNSAMPLE_RATIO, MATTING_MODEL_KIND, or ONNX_PROVIDERS without restarting
  the process, the caller MUST explicitly reset the singleton before the next
  get_matter() call:
      import pipeline.matting as matting_mod
      with matting_mod._singleton_lock:
          matting_mod._singleton = None
  Otherwise the old ORT/TRT session and CuPy buffers will be reused with
  inconsistent shapes/providers.
```

这是阶段 2 不落地 `invalidate_singleton()` 的代价；用文档形式留住要求，将来 UI 真做运行期切换时不会忘。

### 10.5 Commit 前必跑的主路径冷启动观测

单测覆盖不到的硬要求，**不通过则不 commit**：

```powershell
del /F /Q "%LOCALAPPDATA%\PTServer\gpu_warmup_marker.json"
.\.venv\Scripts\python.exe main.py
# 启动日志观察 30 秒
```

**判定条件**：

- ✅ 启动日志包含 4 行 `[WARMUP]` 标识（阶段 1 已有）。
- ✅ 启动日志中 `matting warmup: ...` 一行：要么 `runs=0` 要么不出现，**严禁 `runs=1 elapsed=1589.2ms` 这一类残留**。
- ✅ 首次真实播放请求 `alpha #1 ort_run < 30ms`。

如果出现 `runs=1` 残留，说明阶段 2 形参传递链路有断点，必须在 commit 前修复。

### 10.6 （可选）import 顺序无敏感性回归

发现 Stage 2 真正切断 import 顺序敏感性的强力证据：

```python
# main.py 顶端临时加：
import pipeline.matting  # noqa: F401
```

清 marker 重启。预期：仍不出现 `runs=1 elapsed=1589.2ms`。验证完撤回该 import。

**此项可选，不阻塞 commit**，但建议至少跑一次以确认 Stage 2 设计目标达成。

## 11. 完成判定

1. `compileall pipeline\matting.py utils\gpu_runtime_cache.py` 通过。
2. `pytest tests\test_settings.py tests\test_matting_runtime_policy.py tests\test_alpha_packer.py` 通过（实际已通过 16 测）。
3. **10.5 冷启动观测**：启动日志无 `matting warmup: runs=1` 残留 — **commit 硬门槛**。
4. 二次启动（marker hit）路径无额外 warmup 耗时。
5. （可选）10.6 import 顺序回归通过。
6. PyInstaller 打包后行为一致（手动跑一次 onedir 验证）。
7. `pynv_stream.py` **已拆出独立 commit**，commit message 由开发人员根据真实意图填写。
8. HANDOVER 已追加 10.4 的 trip-wire 段落。

完成后阶段 2 收尾。`invalidate_singleton()` 与 `Matter.close()` 留待 UI 真要做运行期切换时再接（届时另起阶段 3 或合并到具体 UI 特性 PR 里）。
