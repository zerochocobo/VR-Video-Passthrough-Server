# Passthrough 实时透传多并发改造小结

日期：2026-05-28

## 1. 目标与动机

把实时透传的并发能力从硬性 1 路扩展到自适应 1-3 路，显著降低播放器在以下场景遇到 `503 busy` / `409 preempted` 的概率：

- 用户在同一台机器快速切章节、切文件；
- 大显存显卡（≥20GB）完全有余量同时跑 2-3 路 8K 透传，原硬限制造成资源浪费；
- 客户端的 probe/Range 风暴有时会和真实播放请求抢同一个槽位。

`PT_PASSTHROUGH_MAX_CONCURRENT` 配置项早就存在，但默认只能填 1——**真正的阻塞在 GPU 侧的共享 `Matter` 单例**。本次工作把这层假设拆开。

## 2. 原阻塞点

`pipeline.matting.get_matter()` 返回进程级单例 `Matter`，内部持有：

- 单个 `ort.InferenceSession`；
- **每路独有的 RVM 递归状态**（`self._rvm_rec`、`self._rvm_rec_ort`、`self._rvm_io_outputs`、`self._rvm_state_slots`）；
- 共享 GPU 临时缓冲（`self._g_frame/_g_alpha/_g_out/_g_chw` 等）；
- 共享缓存的上一帧 alpha（`self._cached_alpha_small`、`self._cached_alpha_shape`）；
- `_SceneCutDetector`、`_AlphaSmoother` 等时序状态。

直接把 `MAX_CONCURRENT` 拉到 2 而不改造 Matter，会让两路播放共用同一份递归状态 → alpha mask 跳动、抠图崩坏。

## 3. 实施方案

**Matter 实例池 + 自适应并发上限**，不改 Matter 的对外 API。

### 3.1 自适应并发上限（[utils/gpu_requirements.py](../utils/gpu_requirements.py) + [config.py](../config.py)）

`PASSTHROUGH_MAX_CONCURRENT` 默认值改为 `"auto"`。`config.py` 只读取环境变量并调用 `resolve_passthrough_max_concurrent()`；实际 VRAM 探测逻辑放在 `utils/gpu_requirements.py`：

- `detect_nvidia_total_vram_gib()` 通过 `nvidia-smi --query-gpu=memory.total` 探测 VRAM；
- `resolve_passthrough_max_concurrent()` 按下表映射：

| 条件 | max_concurrent |
|---|---:|
| 显式设为整数 | 直接使用 |
| VRAM ≥ 20 GB | 3 |
| VRAM ≥ 12 GB | 2 |
| 其他 / 探测失败 | 1 |

### 3.2 Matter 实例池（[pipeline/matting.py](../pipeline/matting.py)）

替换原来的 `_singleton` 模型为池化模型：

```python
_pool_lock = threading.Lock()
_pool_cond = threading.Condition(_pool_lock)
_pool_all: list[Matter] = []          # 已创建实例
_pool_available: list[Matter] = []    # 空闲实例
_pool_max: int = 1                    # 由 configure_matter_pool() 注入
```

对外 API：

- `configure_matter_pool(max_concurrent)`：启动时设置池容量。
- `acquire_matter(blocking=True, timeout=None)`：取一个实例，必要时懒加载。
- `release_matter(instance)`：归还实例并 `notify` 等待者。
- `get_matter()`：保留旧入口给 warmup 和工具脚本，始终返回 slot 0，且 slot 0 仍在 `_pool_available` 内可被后续 acquire 复用——避免 `MAX_CONCURRENT=1` 时 warmup 创建的 slot 0 永远被 utility 占用、播放流无法 acquire 的死局。

实例**懒加载**：第一次 acquire 才创建，避免冷启动多花 N× ORT/CUDA 加载时间。

### 3.3 路由层挂接（[http_app/routes_media.py](../http_app/routes_media.py)）

引入 `_active_matter: dict[object, Matter]`，与 `_active_streams` 并行追踪每个 active slot 占用的 Matter。

关键改动：

- 两处 `matter = get_matter()` 改为 `matter = acquire_matter()`；
- 取到 matter 后立即在 `_active_lock` 下绑定到 `slot_token`，确保中途被 preempt 也不会泄漏；
- `_replace_active_slot(old, new)` 在交换 owner/started 的同时把 matter 也从 old key 迁到 new key；
- `_release_active_slot(stream)` 取出 matter 并调用 `release_matter`，使 stream/LiveSession 完整生命周期结束时自动归还。

build_stream 内的 acquire 在异常路径也会立即 `release_matter` 防泄漏。

### 3.4 启动配置（[main.py](../main.py)）

GPU warmup 后、`PIPELINE:` 日志前调用 `configure_matter_pool(config.PASSTHROUGH_MAX_CONCURRENT)`。原有的 `MAX_CONCURRENT=%d` 日志会打印 auto 解析后的实际值。

## 4. 风险与边界

- **TensorRT 静态缓存**：每个 Matter 实例独立持有 TRT 引擎，VRAM 占用线性增加，必须在 VRAM 估算里计入（VRAM 映射已留出余量）。
- **MatAnyone2 离线引擎**直接 `Matter(load_model=False)`，不走单例，**不受影响**。
- **CUDA stream 串行执行**：本次没改默认 stream 共享。要榨干 GPU 还需要 per-Matter CUDA stream，留作下一期；本期目标是消除单一会话假设带来的语义错误。
- **NVENC 会话耗尽**：30/40/50 系新驱动 ≥ 8 个会话不会成为瓶颈；老驱动 3 会话上限可在后续 `resolve_passthrough_max_concurrent` 里增加探测。
- **轻匹配版本号**（`self._light_match_version`、`LIGHT_MATCH_FLUSH_QUEUES`）：每个实例独立持有；`get_light_match().version` 全局递增，所有实例下一帧自动同步，语义一致。

## 5. 验证

- 在 RTX 5060 Ti（16GB）上 `PT_PASSTHROUGH_MAX_CONCURRENT=auto` 解析为 **2**。
- 新增 [tests/test_matting_pool.py](../tests/test_matting_pool.py) 覆盖：懒加载 + 复用、上限并发持有、`blocking=False` 满池返回 `None`、`blocking=True` 带 timeout 超时返回 `None`、`release_matter` 幂等 + 忽略陌生实例、`get_matter` slot 0 可被 `acquire_matter` 取走并归还、阻塞 acquire 在 release 后唤醒——共 **8 passed**（用 `_StubMatter` 替换 ONNX 重量级初始化）。
- [tests/test_routes_media_cache.py](../tests/test_routes_media_cache.py) `ReplaceActiveSlotMatterReleaseTests` 覆盖 `_replace_active_slot` 失败路径 Matter 归还、成功路径不释放、close-before-release 顺序、close 失败时仍释放——共 **4 个新增**。
- `tests/test_routes_media_cache.py` + `tests/test_content_directory_modes.py` **42 passed, 4 subtests passed**。合并所有相关套件跑 **54 passed**。
- `tests/test_pynv_stream_bitrate.py` 的 3 个 collection ERROR 是**预先存在**的——引用了从未在 `pipeline/pynv_stream.py` 出现过的 `_realtime_pynv_bitrate`，与本次改动无关。

### 5.1 审核后补的关键修复

**第一轮审核**：发现 `/passthrough` 同步分支在 `MAX_CONCURRENT=1` 时存在事件循环死锁。修复（[http_app/routes_media.py](../http_app/routes_media.py)）：

- `/passthrough` preempt 后调用 `await _close_preempted_stream(...)`，与 live 路径对齐；
- `acquire_matter` 一律走 `asyncio.to_thread` 并带 `PASSTHROUGH_BUSY_WAIT_SEC` 上限的 timeout；
- timeout 或返回 `None` 时回滚 slot（`_release_active_slot`）并返回 `503 Retry-After: 2`；
- live 路径的 `build_stream` 同样改为带 timeout 的 `acquire_matter`，避免 worker 线程被池耗尽长期阻塞。

**第二轮审核**：发现 `_replace_active_slot()` 失败路径 Matter 泄漏。场景：A acquire Matter 并绑定到 `slot_token` → B preempt 了 A（`_active_streams` 中 slot_token 被移走）→ A 后续 `_replace_active_slot(slot_token, stream)` 因 `owner is None` 返回 False，但原实现把 matter 重新塞回 `_active_matter[slot_token]`，三个调用点（[live slot→stream:1683](../http_app/routes_media.py)、[stream→session:2018](../http_app/routes_media.py)、[passthrough slot→stream:2347](../http_app/routes_media.py)）都是 `stream.close()` + `return 409`，**永不再调用 `_release_active_slot`**，Matter 在 dict 里成孤儿，多次累积后池耗尽。

修复 v1：把 `_replace_active_slot` 的失败分支改为内联 `release_matter(leaked_matter)`——既不依赖调用方记得清理，也不重新插入 dict。

**第三轮审核**：修复 v1 虽然解决了"永不释放"，但在 `_replace_active_slot(stream, session)` 这条路径（live `stream → session`，`old_stream` 是已经产出 `first_live_chunk`、worker 在跑的真实 stream）上引入了新的释放-过早竞态：内联 release 后到 caller `await stream.close()` 之间有 `await` 让事件循环可调度，新请求可以瞬间拿走刚释放的 Matter，而旧 stream 的 worker 还在用它做抠图——又退回到本次改造想消除的串扰风险。

修复 v2（[http_app/routes_media.py:693-735](../http_app/routes_media.py)）：在 `_replace_active_slot` 失败分支里，**先**用 `asyncio.to_thread(close)` 关闭 `old_stream`（如果它带 `close()`），**再**释放 Matter。把"关闭 → 释放"的顺序契约集中到这一个函数里，调用方失败路径不需要再 await 第二次 close（line 2018 那处冗余 close 已删除）。`close()` 的异常被吞掉记日志，确保 Matter 仍会归还池。

新增/累计回归测试（[tests/test_routes_media_cache.py](../tests/test_routes_media_cache.py) `ReplaceActiveSlotMatterReleaseTests`）：

1. failure path：`_active_matter` 有 slot_token、`_active_streams` 没有 → 返回 False + `release_matter` 被调用一次 + dict 清空；
2. success path：正常迁移 → 返回 True + `release_matter` 未被调用 + matter 出现在 new_stream key 下；
3. **close-before-release ordering**：构造带 `close()` 的 `old_stream`，patch `release_matter` 记录调用顺序，断言事件序列为 `["close", "release"]`；
4. **close 失败仍释放**：构造 `close()` 抛异常的 stream，断言 Matter 依然归还池（避免坏 close 拖垮整个池）。

运行时验证清单（待实机执行）：

1. 双客户端同时点不同 `*-passthrough-live`，确认两路都被服务且抠图无串扰。
2. `nvidia-smi dmon -s mu` 监控 VRAM 峰值 ≤ 80% 总量，否则下调 `PT_PASSTHROUGH_MAX_CONCURRENT`。
3. `debug_output/server.log` 出现 `passthrough active slot released: active=1` 等递减日志。
4. 故意 `PT_PASSTHROUGH_MAX_CONCURRENT=4` 跑超载，确认池阻塞而不是死锁。
5. 回退 `PT_PASSTHROUGH_MAX_CONCURRENT=1` 应完全保持改造前行为。

## 6. 改动文件清单

| 文件 | 改动 |
|---|---|
| [utils/gpu_requirements.py](../utils/gpu_requirements.py) | 新增 `detect_nvidia_total_vram_gib` / `resolve_passthrough_max_concurrent`，集中放置 `nvidia-smi` 探测逻辑 |
| [config.py](../config.py) | `PASSTHROUGH_MAX_CONCURRENT` 默认 `auto`，仅调用 `resolve_passthrough_max_concurrent` 得到最终整数 |
| [pipeline/matting.py](../pipeline/matting.py) | 单例 → 实例池；新增 `acquire_matter` / `release_matter` / `configure_matter_pool`；`get_matter` 保留并改为 slot 0 可复用 |
| [http_app/routes_media.py](../http_app/routes_media.py) | `_active_matter` 追踪；两处 `get_matter()` → `acquire_matter()`（均通过 `to_thread` + timeout）；`/passthrough` preempt 后主动关旧 stream；超时/失败回滚 slot 返回 503；`_replace_active_slot` / `_release_active_slot` 自动迁移并归还 |
| [main.py](../main.py) | 启动调用 `configure_matter_pool(PASSTHROUGH_MAX_CONCURRENT)` |
| [tests/test_matting_pool.py](../tests/test_matting_pool.py) | **新增**：池行为单元测试，使用 `_StubMatter` 替换 ONNX 初始化 |

## 7. 与既有报告的关系

[summary_20260509_PLAYER_CONCURRENCY_REPORT.md](summary_20260509_PLAYER_CONCURRENCY_REPORT.md) 第 7 节"下一步建议重点"最后一条已明确指出：

> 在考虑多路 PyNv 并发前，先完成单真实播放流稳定性验证。

本次工作建立在那之后单路稳定性已经验证的基础上，提供**可选**的多路能力——`MAX_CONCURRENT=1` 时行为与改造前完全一致，仅当用户/自动探测拉到 ≥2 时进入新路径。
