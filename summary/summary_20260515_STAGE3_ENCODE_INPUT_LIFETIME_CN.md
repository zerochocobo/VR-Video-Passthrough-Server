# 阶段 3 小结 - Encode 输入生命周期

## 范围

本阶段验证交给 PyNv/NVENC 的 GPU NV12 buffer 生命周期。

核心问题是：Matter 持有的 NV12 输出 slot 是否能在 `Encode()` 返回后立刻复用。结论是不能，立即复用可能污染编码输出。

## 改动

- 新增 `tools/pynv_encode_lifetime_probe.py`。
- 探针会创建确定性的 GPU NV12 帧，用 PyNvVideoCodec 编码，可选地在 `Encode()` 返回后立刻覆盖刚编码的输入 slot，再用 FFmpeg 解码输出并检查 Y 值。
- 修改 `pipeline/pynv_stream.py` 的 green 路径，不再在 `Encode()` 返回后立即释放 `nv12_slot`。
- live green 路径现在把最近编码过的 slot 保存在 pending 队列中。默认 `PASSTHROUGH_NV12_RING_SLOTS=3` 时，保留最近 2 个 slot，只释放最旧的 slot。
- pending slot 会在 `EndEncode()` 后释放，也会在 worker `finally` 中兜底释放。

## 结论

- 单 slot 在 `Encode()` 后立即覆盖是不安全的。
- `Encode()` 后做 CUDA null stream synchronize 也不能让立即覆盖变安全。
- 三 slot 延迟复用通过了当前合成探针，包括 8K HEVC。
- 本阶段证明的是延迟释放 slot 的必要性，不证明可以删除 encode 前的 `cuda_stream.synchronize()`。

## 验证

```powershell
python -m compileall config.py pipeline\matting.py pipeline\pynv_stream.py tools\pynv_encode_lifetime_probe.py
```

8K HEVC 生命周期探针：

```powershell
uv run python tools\pynv_encode_lifetime_probe.py --width 8192 --height 4096 --frames 24 --codec hevc --bitrate 50000000 --gop 60 --progress 6 --slots 3
```

结果：

- 报告：`baseline/pynv_encode_lifetime_stage3_20260515_111503_740948.md`；
- `ok=True`；
- 污染或未知帧：`0`；
- encode FPS：`36.48`。

live smoke：

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 1 --startup-timeout 240 --client-timeout 60
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer alpha --duration 1 --startup-timeout 240 --client-timeout 60
```

报告：

- green：`baseline/auto_tune_8k_phase1_20260515_111516.md`；
- alpha：`baseline/auto_tune_8k_phase1_20260515_111532.md`。

两个 smoke 都通过。

## 对基线的影响

本阶段不应期待明确超过 Phase 1 基线。它是正确性和安全性阶段。

短测数据没有明显回退，但不能作为严格对比，因为 smoke 只有 1 秒：

- green 最新 interval FPS：`36.64`；
- alpha 最新 interval FPS：`35.96`；
- Phase 1 基线：平均 interval FPS `34.32`，最新 interval FPS `35.58`。

严格对比可运行：

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer alpha --duration 60 --startup-timeout 240 --client-timeout 90
```

## 进入下一阶段前的风险

- 第四阶段不能直接删除 `cuda_stream.synchronize()`。阶段 3 证明的是输入 slot 生命周期和复用距离，没有证明 NVENC 会等待 Matter 的 `_CUDA_STREAM` composite kernel。
- 不要把 `Encode()` 返回理解成 GPU 输入内存已经可以覆盖。
- 如果 GOP、B 帧、codec、分辨率、驱动或 PyNvVideoCodec 版本变化，建议重跑生命周期探针。
- green 路径使用 Matter NV12 slot pool；alpha packer 路径不使用该 pool，因此不能把 green slot 的结论直接外推到 alpha 内部 buffer。

## 决策

保留延迟释放 slot 的前提下进入阶段 4。下一步真正有性能收益的工作是 decode / matting / encode 三段流水线，并且先通过离线 `tools/pynv_fullchain_probe.py --pipeline staged` 门禁验证。
