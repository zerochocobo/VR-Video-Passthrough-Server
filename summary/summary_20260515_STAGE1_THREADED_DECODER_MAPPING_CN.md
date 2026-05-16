# 阶段 1 小结 - ThreadedDecoder 帧映射校准

## 范围

本阶段调查 `PyNvVideoCodec.ThreadedDecoder` 是否可以安全替代或补充当前
`SimpleDecoder[index]` 路径，用于后续 8K passthrough 的 staged pipeline。

核心风险是帧身份。如果解码出的画面内容和帧序号不能稳定对应，那么它不能进入
生产 live passthrough 路径。

## 关键结论

之前看到的“帧不一致”来自错误的验证方法。

错误测试在读取返回的 GPU frame 前先调用了 `ThreadedDecoder.end()`。根据
NVIDIA ThreadedDecoder 的使用语义，返回的 frame 必须在下一次
`get_batch_frames()` 调用前、并且在 decoder 结束前被消费。

修正 frame 生命周期处理后，画面内容稳定满足：

```text
ThreadedDecoder(start_frame=N) 的第 K 个输出 == SimpleDecoder[N + K]
```

验证方式是比较 Y/UV 平面的 SHA256。

## 重要限制

在 `videos/test_8k_2.mp4` 上，同一视觉帧的 `ThreadedDecoder.getPTS()` 和
`SimpleDecoder.getPTS()` 并不完全一致。

观察到的 PTS 偏移是稳定的：

- 普通帧：`2002` ticks；
- `start_frame=0` 的第一帧：`3003` ticks。

因此后续代码不能用 `getPTS()` 作为帧身份依据，必须使用显式源帧计数器。

## 验证

新增：

- `tools/pynv_threaded_decode_probe.py`
- `tools/pynv_threaded_mapping_probe.py`

正式映射验证命令：

```powershell
uv run python tools\pynv_threaded_mapping_probe.py videos\test_8k_2.mp4 --frames-per-start 16 --repeats 3 --buffer-size 32
```

结果：

- `150/150` 个 case 通过；
- 覆盖起点：`0,1,2,3,4,10,30,58,120,300`；
- 覆盖 batch size：`1,2,4,8,16`；
- 每组重复：`3` 次；
- 所有检查帧都按 Y/UV SHA256 匹配 `SimpleDecoder[index]`。

产物：

- `baseline/pynv_threaded_mapping_phase2_20260515_102152.md`
- `baseline/pynv_threaded_mapping_phase2_20260515_102152.json`

## 决策

在已测试条件下，ThreadedDecoder 的帧映射已经校准并证明稳定。

后续 staged pipeline 可以使用 ThreadedDecoder，但必须遵守：

- 在下一次 `get_batch_frames()` 前消费当前返回的 frame；
- 不要在 `end()` 后读取已返回 frame；
- 使用 `expected_index = start_frame + local_sequence` 追踪帧身份；
- 保留 `SimpleDecoder` 作为 fallback。

本问题不需要再生成外部专家求助 summary。
