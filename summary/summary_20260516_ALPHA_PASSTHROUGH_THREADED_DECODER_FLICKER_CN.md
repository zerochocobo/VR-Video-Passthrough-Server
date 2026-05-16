# Alpha 直通 ThreadedDecoder 红色 Alpha 闪烁问题小结

## 结论摘要

Alpha 直通模式在使用 PyNv `ThreadedDecoder` / `threaded_serial` 时，部分视频会出现红色 alpha 通道持续闪烁；同一视频切回 `SimpleDecoder` 后稳定。已确认该问题不是简单的启动异常，也不是普通 decode-only 帧内容不匹配。当前生产保护策略是：**alpha 直通默认强制使用 `SimpleDecoder`，绿幕路径仍可继续使用 `ThreadedDecoder`。**

此问题仍需外部专家判断 PyNv `ThreadedDecoder`、CUDA stream 同步、RVM recurrent state 与 alpha packer 组合下的根因。

## 复现信息

- 复现模式：在线 alpha 直通播放
- 问题源示例：`videos\72456_3840p.mp4`
- 用户观察：
  - `threaded_serial`：红色 alpha 通道持续明显闪烁
  - `simple`：不闪烁，画面稳定
- 相关运行配置：
  - `PT_ALPHA_STRIDE=1`
  - RVM FP16 模型在线路径已可用
  - decoder 由 UI 显存档位或环境变量切换

## 当前相关代码

- `config.py`
  - `PASSTHROUGH_PYNV_DECODER`
  - 新增诊断开关：`PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER`
  - 默认 `0`，即 alpha 不允许 threaded decoder 进入生产路径
- `pipeline/pynv_stream.py`
  - 在线 alpha 路径中，如果 `decoder=threaded_serial` 且未显式开启诊断开关，则实际回退为 `decoder=simple`
  - 绿幕路径不受影响
- `tools/offline_alpha_passthrough.py`
  - 离线 alpha 使用同样保护策略
- `pipeline/pynv_io.py`
  - 为 `GpuNv12Frame` / `GpuP016Frame` 增加 `owned_copy()`
  - 尝试将 PyNv-owned frame plane 拷贝到 CuPy-owned GPU buffer
  - copy 前后均加入同步边界

## 已尝试但不足的修复

### 1. Owned GPU copy

尝试动机：

- PyNv `ThreadedDecoder.get_batch_frames()` 返回帧的生命周期敏感；
- 之前已经确认返回帧不能跨 batch / 跨线程长期持有；
- alpha 路径比绿幕路径复杂，会先 decode，再 RVM，再 fisheye alpha pack，再 red-channel overlay。

实现：

- `frame.owned_copy()`：
  - `cp.cuda.Device().synchronize()`
  - 用 `CudaPlane.as_cupy()` 读取 PyNv plane
  - `cp.ascontiguousarray()` 拷贝成 CuPy-owned buffer
  - `cp.cuda.get_current_stream().synchronize()`

结果：

- 解决了最初 `cp.asarray(raw_pyNv_view)` 导致的 `TypeError: Expected tuple, got list` 和 503 问题；
- 但用户复测后，alpha 红通道仍然严重闪烁；
- 因此 owned copy 不是充分修复。

### 2. Decode-only hash 复核

命令：

```powershell
.\.venv\Scripts\python.exe tools\pynv_threaded_decode_probe.py videos\72456_3840p.mp4 --frames 180 --fps 30 --batch-size 4 --buffer-size 8 --hash-frames 30
```

结果：

- Threaded selected FPS: `79.43`
- Simple baseline FPS: `79.85`
- Hash checked: `30`
- Matched: `30`
- OK: `True`
- PTS deltas: 全部 `0`

解释：

- 抽样 decode-only 内容与 SimpleDecoder 一致；
- 但 live alpha 仍闪烁；
- 说明问题可能不在简单帧内容映射，而在完整 alpha 链路中的时序、stream 可见性、RVM recurrent state 或 PyNv ThreadedDecoder 内部发布语义。

## 当前生产保护

新增配置：

```text
PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=0
```

默认行为：

- alpha 直通：强制实际 decoder 为 `simple`
- green/绿幕直通：仍使用 UI 显存档位或 `PT_PASSTHROUGH_PYNV_DECODER`
- 如需诊断，可显式设置：

```text
PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=1
```

开启后：

- alpha 允许 `threaded_serial`
- 会执行 `owned_copy()`
- 仅用于诊断，不建议生产使用

## 已知重要线索

1. `SimpleDecoder` 稳定，`ThreadedDecoder` alpha 不稳定。
2. Decode-only hash 能匹配，说明不是普通逐帧内容错位。
3. Owned copy + device synchronize 仍无法消除闪烁。
4. 问题主要出现在 alpha 直通，绿幕路径暂未报告同类红色 alpha 闪烁。
5. RVM 是 recurrent 模型，如果输入帧时序、内容发布、CUDA stream 可见性或状态推进出现轻微异常，alpha 结果可能被放大成连续闪烁。

## 需要外部专家判断的问题

1. PyNv `ThreadedDecoder.get_batch_frames()` 返回 GPU frame 后，除了“下一次 get_batch_frames 前消费完”，是否还需要官方 stream/event 等待方式？
2. `get_batch_frames()` 返回是否保证对应 CUDA memory 在调用返回时已经对默认 stream / CuPy stream 可见？
3. `cp.cuda.Device().synchronize()` 是否足以等待 PyNv 内部 NVDEC / postprocess stream？
4. ThreadedDecoder 对 AV1 7680x3840 60fps 源是否可能存在内部 frame reorder、surface reuse 或 delayed output 语义，decode-only hash 抽样无法覆盖？
5. RVM recurrent state 与 ThreadedDecoder 的 batch/prefetch 输出组合是否需要额外的 frame boundary 或 stream synchronization？
6. 是否应彻底禁止 alpha + ThreadedDecoder，还是存在官方推荐的 retain/copy/sync API 可以安全启用？

## 当前建议

在外部专家确认前：

- 生产 alpha 直通保持 `SimpleDecoder`
- `ThreadedDecoder` 继续用于 green/绿幕或 decode-only 性能路径
- 不再尝试通过简单 copy/sync 继续把 alpha 生产路径切回 threaded

