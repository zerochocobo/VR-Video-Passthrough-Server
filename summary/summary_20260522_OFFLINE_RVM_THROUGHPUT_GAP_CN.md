# Offline RVM Throughput Gap Summary

日期：2026-05-22

## 问题

同一个 `videos/test_8k.mp4`，实时 alpha 模式在 `PT_PASSTHROUGH_MAX_FPS=0` 时可以达到约 `77fps`，但离线 RVM TensorRT 生成长期停在约 `36fps`。离线日志中的主要耗时集中在 `decode_avg ~= 19-21ms`，而 RVM TensorRT 本身约 `5.8-6.0ms`。

## 当前结论

离线没有被输出 FPS 限制。日志中 `output_fps=59.940060`，且 `target=899` 对应 15 秒源片段。

TensorRT/RVM 推理是生效的。离线日志显示：

- `static_trt=True`
- `rvm_iobinding=True`
- `rvm_ort_avg ~= 5.8-6.0ms`

实时对照来自 `debug_output/server.log`，同样是 alpha、simple decoder、serial worker，不是 threaded decoder：

- `output_mode=alpha`
- `decoder=simple`
- `worker_mode=serial`
- `fps_cap=0.000`
- 稳态约 `decode=5-6ms`、`composite=6ms`、总吞吐约 `77fps`

离线当前仍是：

- green: `throughput ~= 36.4fps`
- alpha: `throughput ~= 36.3-36.4fps`
- alpha 最新验证：`decode_avg=20.722ms`、`matting_avg=6.227ms`、`decode_matting_avg=26.949ms`

因此，根因仍在离线进程中 `PyNvSimpleDecoder.frame_at()` 等待明显高于实时路径，而不是 RVM/TensorRT、alpha pack、音频、输出 FPS、索引回退或 NVENC 参数。

## 已排除

1. 输出 FPS 限制

离线日志明确显示 `output_fps=59.940060`，没有 30fps 或其他 cap。

2. TensorRT 未生效

离线 RVM 日志显示 `static_trt=True`，`rvm_ort_avg ~= 5.8ms`，和实时 `ort_run ~= 5.4-6.0ms` 同量级。

3. 音频并发读源文件

green 离线已改成先抽取 AAC sidecar，再视频-only 处理，最后 mux。用户复测后无改善。

4. green composite 与 alpha pack 差异

离线 alpha 也只有约 `36fps`，且 `rvm_alpha_pack_avg=0.063-0.064ms`，alpha pack 不是瓶颈。

5. source index 重复/回退

离线 green/alpha 主循环已加入实时路径同样的单调 `src_idx` 保护。最新日志：

- `source_index_rewinds = 0`

无改善。

6. Matter/ORT 初始化顺序

RVM 离线已改为先初始化 RVM engine/Matter/ORT/TRT，再创建 PyNv decoder，以贴近实时路径。无改善。

7. NVENC 参数差异

RVM 离线命令已追加 `--realtime-encoder-args`。最新日志确认生效：

- command includes `--realtime-encoder-args`
- `encoder_kwargs={'bitrate': '10086258'}`

吞吐仍为 `36.32fps`。

## 已改动文件

- `pipeline/matting.py`
  - 降低正常 TensorRT/ORT warning 噪声。
  - static TRT session 失败后不重复尝试。

- `offline/convert.py`
  - RVM 离线命令追加 `--realtime-encoder-args` 用于诊断。
  - 离线 TRT provider 只在 RVM fast 且 cache ready 时保留。

- `tools/offline_passthrough.py`
  - green 离线音频改为先抽 AAC、视频-only 处理、最后 mux。
  - RVM green 使用 `Matter.acquire_nv12_output_slot()` 和 pending ring，贴近实时。
  - 增加 `sync`、`dec+mat+sync`、`source_index_rewinds`、`encoder_kwargs` 日志。
  - RVM 离线先初始化 Matter/ORT/TRT，再创建 decoder。

- `tools/offline_alpha_passthrough.py`
  - 增加 `dec+mat`、`source_index_rewinds`、`encoder_kwargs` 日志。
  - 支持 `--realtime-encoder-args`。
  - RVM 离线先初始化 Matter/ORT/TRT，再创建 decoder。
  - 保留 alpha 输出像素码率缩放测试兼容。

## 验证

已通过：

```powershell
.\.venv\Scripts\python.exe -m compileall offline\convert.py tools\offline_passthrough.py tools\offline_alpha_passthrough.py
.\.venv\Scripts\python.exe -m pytest tests\test_offline_convert.py tests\test_settings.py tests\test_offline_alpha_bitrate.py
git diff --check
```

测试结果：`20 passed`。

## 关键日志对比

实时 alpha，来自 `debug_output/server.log`：

- `decoder=simple`
- `worker_mode=serial`
- `output_mode=alpha`
- `fps_cap=0.000`
- `output_fps=59.940`
- 稳态 `decode=5-6ms`
- 稳态 `composite=~6ms`
- 总吞吐约 `77fps`

离线 alpha，最新用户复测：

- command includes `--realtime-encoder-args`
- `encoder_kwargs={'bitrate': '10086258'}`
- `source_index_rewinds = 0`
- `decode_avg = 20.722 ms`
- `matting_avg = 6.227 ms`
- `decode_matting_avg = 26.949 ms`
- `throughput = 36.32 fps`

## 剩余可疑点

当前最可疑的是离线工具进程中的 `PyNvSimpleDecoder.__getitem__` / `frame_at()` 行为与实时 worker 中同一 decoder 的行为存在上下文差异。这个差异不是 decoder 类型、不是索引顺序、不是 RVM、不是 NVENC 参数、不是音频并发。

建议专家继续看：

1. PyNv SimpleDecoder 是否在离线脚本进程中受到 MP4 输出 mux/文件写入/同进程 ffmpeg pipe 反压影响，而实时 StreamingResponse/mpegts 缓存路径没有同样反压。
2. 实时路径是否有隐藏的预热、reader 订阅、cache 或 server 生命周期状态，使 SimpleDecoder 进入更快的顺序读取路径。
3. 是否需要新增一个最小 A/B probe：同一个 `base_environment()` 下，只跑 `PyNvSimpleDecoder.frame_at()` 顺序解码 899 帧，不做 RVM、不做 NVENC、不做 mux；再逐步加入 RVM、NVENC、ffmpeg pipe，定位是哪一步让下一次 `frame_at()` 从 5-6ms 变成 20ms。
4. 注意不要裸跑 GPU/ORT 工具。必须通过 UI/`offline.convert`/`main.py`，或显式使用 `ui.services.process_helpers.base_environment()`，否则 CUDA provider 可能失效并产生错误结论。
