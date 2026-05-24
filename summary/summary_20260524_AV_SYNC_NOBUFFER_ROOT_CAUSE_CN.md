# AV Sync 根因小结：raw HEVC `+nobuffer` 丢首 GOP

日期：2026-05-24

## 结论

这次音画不同步的主因不是 AAC cache、video slate、Range 探测或固定音频延迟，而是 PyNv 实时输出 raw HEVC 后交给 FFmpeg mux MPEG-TS 时使用了：

```text
-fflags +genpts+nobuffer+flush_packets
```

其中 `+nobuffer` 会让 FFmpeg 在 raw HEVC 输入上丢掉开头约一个 GOP。当前 `PT_PASSTHROUGH_GOP=60`，源视频约 59.94fps，所以视频内容实际领先音频约 1 秒。

用户听到的“音频晚 1 秒左右”，本质是“视频内容提前 1 秒左右”。

## 项目背景

`/passthrough_live/{name}` 是实时生成流：

- 视频：源视频经 PyNvVideoCodec 解码，GPU matting/composite 或 alpha packing，再由 PyNv/NVENC 编成 HEVC elementary stream。
- 音频：源 MP4 的音频由 FFmpeg 在最终 mux 阶段读取，并转 AAC 后进入 MPEG-TS。
- 最终封装：当前 PyNv 路径采用两段 mux：
  1. raw HEVC -> video-only MPEG-TS，并用 `setts` 合成 CFR 时间戳；
  2. video-only MPEG-TS + 源音频 -> 最终 MPEG-TS。

因此音频和视频确实是分开处理后再合流，必须同时验证音频内容时间和视频内容时间。

## 关键证据

测试视频：

```text
videos/urvrsp00566_1_8k.mp4
```

原始 MP4 已确认音画同步。

音频侧检查一直正常：

```text
audio_content_offset ≈ -0.021s
audio/video stream start delta ≈ -0.021s
```

这说明最终 TS 里的音频内容基本来自正确源时间。

新增视频内容匹配后发现旧输出异常：

```text
expected source frame index: 10789
matched source frame index: 10849
delta: 60 frames ≈ 0.997s
```

这与当前 GOP=60 完全吻合。

直接 probe 进一步确认：

- raw PyNv HEVC 编码 120 帧本身没有只剩 60 帧的问题。
- 使用 `+genpts+nobuffer+flush_packets` mux 后，FFmpeg 输出只剩 60 个 video packets，首帧内容匹配到 `10849`。
- 去掉 `+nobuffer` 后，输出 120 个 video packets，首帧内容匹配回 `10789`，视频内容偏移约 `-0.004s`。

## 排除项

- AAC cache：关闭后问题仍存在。
- video slate：关闭后有所改善，但仍约 1 秒错位，不是最终主因。
- Range 大请求：更像播放器尾部/元数据探测；本次决定性证据来自服务端抓包内容匹配，不依赖播放器行为。
- 固定 audio offset：不应使用。音频本身没有晚 1 秒，硬调音频会破坏正确 TS。
- 关键帧 seek：中途播放可能受关键帧影响，但本问题在 `t=0` 也出现，且最终确认是 mux 丢首 GOP。

## 修复

已修改：

- CODE: `config.py`
  - `MUX_NOBUFFER_ENABLE = _env("MUX_NOBUFFER_ENABLE", "0") == "1"`
  - `PT_MUX_NOBUFFER_ENABLE` 默认从 `1` 改为 `0`。
  - 注释说明 raw HEVC 上启用 `+nobuffer` 会导致首 GOP 丢失。
- CODE: `pipeline/pynv_stream.py`
  - `_mux_fflags()` 仍保留 `+genpts+nobuffer+flush_packets` 分支，但默认不进入。
  - `_open_pipe_ts_muxer()` 的 video-only TS mux 命令继续调用 `_mux_fflags()`，因此默认变成 `-fflags +genpts`。
- CODE: `tests/test_config_defaults.py`
  - `test_mux_latency_defaults_are_low_latency` 更新为断言 `config.MUX_NOBUFFER_ENABLE == False`。
- CODE: `tests/test_pynv_mux_latency.py`
  - raw video direct mux 和 pipe_ts video mux 的测试期望从 `+genpts+nobuffer+flush_packets` 改为 `+genpts`。
- CODE: `tools/check_live_video_alignment.py`
  - 新增绿色模式视频内容客观校验工具。
  - 工作方式：抽取抓包首帧，mask 掉绿色背景，用 PyNv 解码源视频窗口并匹配前景 luma。

修复后用户真机反馈：音画同步“终于完全正常”。

## 代码定位

- CODE CONFIG: `config.py` -> `MUX_NOBUFFER_ENABLE`
- CODE MUX FLAGS: `pipeline/pynv_stream.py` -> `_mux_fflags()`
- CODE PIPE-TS VIDEO MUX: `pipeline/pynv_stream.py` -> `_open_pipe_ts_muxer()`
- CODE AUDIO CHECK TOOL: `tools/check_live_audio_alignment.py`
- CODE VIDEO CHECK TOOL: `tools/check_live_video_alignment.py`
- CODE SLATE CHECK TOOL: `tools/check_av_sync_slate.py`
- CODE CAPTURE TOOL: `tools/capture_live_mpegts.py`
- CODE TESTS: `tests/test_config_defaults.py`, `tests/test_pynv_mux_latency.py`

## 验证命令

服务端重启后，建议用同一个视频跑：

```bat
uv run python tools\capture_live_mpegts.py --host 127.0.0.1 --port 8200 --name videos/urvrsp00566_1_8k.mp4 --t 180 --mode green --seconds 7 --out debug_output\after_fix_green.ts --check
uv run python tools\check_live_audio_alignment.py debug_output\after_fix_green.ts --source videos\urvrsp00566_1_8k.mp4 --source-start 180 --json
uv run python tools\check_live_video_alignment.py debug_output\after_fix_green.ts --source videos\urvrsp00566_1_8k.mp4 --source-start 180 --json
```

预期：

- 音频内容偏移接近 0，允许几十毫秒。
- 视频内容偏移接近 0，不应再出现约 1 秒领先。

代码测试：

```bat
uv run python -m pytest tests\test_config_defaults.py tests\test_pynv_mux_latency.py
uv run python -m py_compile tools\check_live_video_alignment.py
```

已通过：

```text
21 passed
py_compile passed
```

## 后续注意

不要在常规播放中设置：

```bat
PT_MUX_NOBUFFER_ENABLE=1
```

该开关现在只适合专项诊断。以后排查音画同步时，不要只看 `ffprobe stream start_time`，还要同时跑音频内容匹配和视频内容匹配。
