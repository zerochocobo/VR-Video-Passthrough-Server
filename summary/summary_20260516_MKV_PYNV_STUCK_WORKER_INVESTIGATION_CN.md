# MKV PyNv Worker 卡死排查小结

日期：2026-05-16

## 背景

实时 alpha 直通测试中，播放 `test_mkv_8k.mkv` 后再切换到 MP4，红色 alpha 层仍然会闪烁。MP4 本身可以继续启动和播放，但闪烁会在 MKV 播放路径触发后持续出现。

## 发现

- MKV 流在 preempt/close 后可能留下一个未退出的 `pynv-worker` 线程。
- 这个 worker 没有走到 `worker_done`。
- 捕获到的线程栈显示，它卡在 PyNvVideoCodec 的 native 取帧位置：

```text
pipeline/pynv_stream.py: frame = self._dec.frame_at(src_idx)
pipeline/pynv_io.py: frame = self._decoder[index]
PyNvVideoCodec SimpleDecoder.__getitem__
```

- 这说明 MKV 源上使用 `SimpleDecoder[index]` 随机取帧是高风险边界。
- 一旦线程卡进 native 代码，Python 进程内无法安全强杀该线程。
- 后续 MP4 alpha 播放仍然和这个残留 worker 共用同一个 CUDA/NVDEC/NVENC 进程状态，因此红色 alpha 闪烁持续存在是合理现象。

## 临时尝试

- 增加了 VRAM 诊断日志和卡死线程栈打印。
- 当 worker 无法停止时，增加了进程级 PyNv taint 标记。
- 临时尝试在 taint 后阻断或警告 alpha 播放。
- 开始探索为 MKV 使用顺序 `ThreadedDecoder`，避免 `SimpleDecoder[index]`。

这些改动对定位问题有帮助，但不作为最终修复方案保留。代码改动会回滚，之后重新设计 MKV 处理方式。

## 后续方向

下一步应集中研究更安全的 MKV 实时 alpha 解码路线：

- MKV 不再使用 `SimpleDecoder[index]` 随机取帧；
- 测试 PyNv `ThreadedDecoder` 顺序解码是否能替代；
- 对比帧正确性和关闭行为；
- 如果 native decoder 无法可靠停止，考虑把高风险 MKV PyNv 解码隔离到子进程。

## 验证依据

关键证据来自 `debug_output/server.log` 中 MKV 切 MP4 的测试日志。最强信号是 worker 卡在 `SimpleDecoder.__getitem__` 的线程栈。
