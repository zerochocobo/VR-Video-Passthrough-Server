# 阶段 2 小结 - NV12 输出槽所有权基础

## 范围

本阶段为 green passthrough composite 路径引入显式 GPU NV12 输出槽所有权。

目标是避免继续把 Matter 的 GPU NV12 输出当成单一隐式 scratch buffer，为后续
安全的 encode/pipeline overlap 做结构准备。

## 改动

- 新增 `PT_PASSTHROUGH_NV12_RING_SLOTS`，默认 `3`。
- 新增 Matter 侧 slot pool API：
  - `Nv12OutputSlot`；
  - `Matter.acquire_nv12_output_slot(h, w)`；
  - `Matter.release_nv12_output_slot(slot)`。
- 更新 green GPU composite 函数，让调用方可以传入显式输出 slot buffer。
- 更新 live green composite 路径：
  - composite 前 acquire slot；
  - 将 slot 传给 Matter composite；
  - `Encode()` 返回后通过 `finally` release slot。

## 验证

```powershell
python -m compileall config.py pipeline\matting.py pipeline\pynv_stream.py
```

Green smoke：

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 1 --startup-timeout 240 --client-timeout 60
```

Alpha smoke：

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer alpha --duration 1 --startup-timeout 240 --client-timeout 60
```

两个 smoke 都通过。

## 对基线的影响

本阶段不应该期待超过 Phase 1 性能基线。

原因是：当前 NVENC encode 前的 CUDA stream synchronize 仍然保留。Ring slot pool
解决的是输出 buffer 所有权和复用问题，但它不能单独证明 NVENC 会等待
`_CUDA_STREAM` 上的 composite kernel 完成。

本阶段是安全基础，不是主要 FPS 提升点。

## 风险和下一步

关键剩余风险是 NVENC 输入生命周期。

如果 `Encode()` 返回时 NVENC 已经消费或复制了输入帧，那么 slot 可以在
`Encode()` 返回后安全复用。如果 `Encode()` 异步保留输入指针，那么复用或覆盖 slot
可能污染编码输出。

本地可见的 PyNv encoder API 没有明显的 CUDA event 或 stream-wait handoff。因此
下一阶段必须先做 encode-input lifetime probe，然后才能考虑移除 per-frame sync。

建议 probe：

```text
composite 到 slot A
不 synchronize
Encode(slot A)
Encode 返回后立刻覆盖 slot A
继续编码更多帧
解码编码输出
检查 slot A 对应帧是否被污染
```

如果输出没有被污染，后续阶段可以考虑移除或缩小 sync。如果出现污染，sync 必须保留，
或者必须找到可靠的 CUDA event/stream handoff。

## 决策

进入下一阶段前必须验证 encode-input lifetime。不要盲目删除
`cuda_stream.synchronize()`。
