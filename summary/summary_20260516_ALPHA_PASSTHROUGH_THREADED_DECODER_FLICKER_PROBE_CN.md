# Alpha 直通 ThreadedDecoder 闪烁隔离测试记录

## 背景

外部反馈认为，alpha 红色闪烁等价于 alpha mask 帧间剧烈跳变。`owned_copy()` 和 `Device().synchronize()` 只能保证过去写入完成，不能证明 PyNv `ThreadedDecoder` 的 producer 后续不会复用 surface，也不能证明 RVM recurrent state 不会放大一次输入污染。

因此本次先落实两个低风险代码项，并执行 T3 隔离测试。

## 代码防御项

### 1. `owned_copy()` 增加 shape 自检

文件：`pipeline/pynv_io.py`

- NV12/P016 Y plane 必须匹配 `(h, w)`
- UV plane 接受两种合法形态：
  - `(h / 2, w)`
  - `(h / 2, w / 2, 2)`

第一次自检写得过严，只接受 `(h / 2, w)`，T3 暴露 PyNv 当前 UV shape 为 `(1920, 3840, 2)`。已修正。

### 2. 在线 alpha 启动日志增强

文件：`pipeline/pynv_stream.py`

alpha 路径现在打印：

```text
alpha decoder detail: effective_decoder=<...> batch=<...> buffer=<...> owned_copy=<...> allow_threaded=<...>
```

用于后续确认实际生效 decoder，而不是只看 UI 配置。

## T3 隔离测试

目的：

测试将 `ThreadedDecoder` 压到 `batch_size=1, buffer_size=2` 后，是否能缓解/消除 alpha 闪烁。如果消失，则更支持“surface 跨批回收/producer-consumer 竞争”假说。

命令：

```powershell
$env:PT_PASSTHROUGH_PYNV_DECODER='threaded_serial'
$env:PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER='1'
$env:PT_PASSTHROUGH_PYNV_THREADED_BATCH_SIZE='1'
$env:PT_PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE='2'
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
.\.venv\Scripts\python.exe tools\offline_alpha_passthrough.py videos\72456_3840p.mp4 --engine rvm --out debug_output\alpha_t3_threaded_b1_buf2.mp4 --duration 3 --fps 30 --alpha-stride 1 --input-size 1024 --sbs-batch --audio off --preset P1
```

输出文件：

```text
debug_output\alpha_t3_threaded_b1_buf2.mp4
```

日志关键行：

```text
[offline-alpha] decoder=threaded_serial batch=1 buffer=2 owned_copy=True
frames = 90
throughput = 31.86 fps
decode_avg = 2.481 ms
matting_avg = 25.255 ms
rvm_ort_avg = 24.763 ms
```

## 当前结果解释

- T3 已成功生成 90 帧 / 3 秒输出；
- 该输出尚需肉眼检查是否仍有红色 alpha 闪烁；
- 如果 T3 输出不闪，说明大 batch / surface 回收竞争嫌疑增强；
- 如果 T3 输出仍闪，说明问题更可能在 ThreadedDecoder 与 RVM recurrent/stream visibility 的组合，而不是单纯跨批 surface 回收窗口。

## 当前生产策略保持不变

- `PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=0` 仍为默认；
- alpha 直通生产路径仍使用 `SimpleDecoder`；
- `ThreadedDecoder` 只保留给绿幕路径和显式诊断。

## 追加观察：红色变灰

用户进一步观察到：暂停在闪烁帧时，红色 alpha 通道不是遮罩形状消失，而是**遮罩形状仍在、颜色从红色变成灰色**。

这改变了问题判断：

- 如果遮罩形状仍在，Y 平面很可能仍然写入了 mask；
- 颜色变灰更像是 NV12 UV 色度没有稳定写成红色；
- 因此问题不一定是 alpha mask 值本身乱跳，也可能是 alpha block 区域的 UV 写法不完整。

## Alpha packer UV 修复

文件：`pipeline/alpha_packer.py`

原逻辑：

- `overlay_alpha_packer_layout` 对 alpha block 内每个非零 mask 像素写 Y；
- 但只有当 `(x, y)` 都是偶数时才写对应 NV12 UV；
- 如果一个 2x2 色度块左上角像素 mask 为 0，而其它三个像素 mask 非 0，就会出现：
  - Y 平面有遮罩亮度；
  - UV 平面没有写红色，保留原图或中性灰；
  - 视觉上就是“遮罩形状还在，但红色变灰”。

修复：

- 抽出 `alpha_layout_source()` / `alpha_layout_mask_at()`；
- 每个 2x2 NV12 UV block 写 UV 时，读取该 2x2 内四个 alpha layout 像素的最大 mask；
- 用最大 mask 计算红色 UV，确保只要 2x2 内有遮罩，色度就写成红色。

生成的对照片段：

```text
debug_output\alpha_uv_fix_simple_3s.mp4
debug_output\alpha_uv_fix_threaded_b1_buf2_3s.mp4
```

这两个文件需要肉眼检查红/灰闪烁是否消失。
