# 离线 passthrough 与 MatAnyone2 ONNX 接入计划

日期：2026-05-10

## 目标

新增一个离线功能：指定某个 MP4，自动生成已经抠图并绿幕合成的
passthrough MP4 文件。

关键原则：

- 复用当前已经优化过的 PyNv 输入链路：
  `PyNvVideoCodec decode -> GPU NV12 -> composite -> NVENC -> FFmpeg mux`。
- 离线输出 MP4 文件，不走 DLNA live / MPEG-TS / slate / 音频缓存逻辑。
- 第一阶段先打通离线文件生成骨架；MatAnyone2 ONNX 接入作为独立模型层
  替换。

## 参考项目结论

MatAnyone2 官方仓库：

- 项目地址：https://github.com/pq-yang/MatAnyone2
- README 明确说明推理输入是“视频 + 第一帧分割 mask”。
- 官方推理入口会输出 foreground 视频和 alpha 视频。
- Python API 使用 `MatAnyone2.from_pretrained(...)` 和
  `InferenceCore(...).process_video(input_path, mask_path, output_path)`。

官方 `inference_matanyone2.py` 的核心流程：

- 读取全部视频帧。
- 读取第一帧 mask。
- 对第一帧重复 warmup。
- `processor.step(image, mask, objects=objects)` 初始化目标。
- 后续帧调用 `processor.step(image)` 做时序传播。
- 输出 alpha matte，再把前景与绿幕合成。

这说明 MatAnyone2 和 RVM 的接口本质不同：

- RVM：单帧/递归状态输入，直接输出 alpha。
- MatAnyone2：第一帧 mask + 内部记忆传播 + 后续逐帧 step。

因此不能简单把 `PT_MODEL_PATH` 换成 MatAnyone2 ONNX。

## 已实现

新增文件：

- `tools/offline_passthrough.py`

当前能力：

- 命令行指定视频：
  `python tools/offline_passthrough.py <video> --out <out.mp4>`
- 视频路径可以是绝对路径，也可以相对 `PT_VIDEO_DIR`。
- 默认输出：`<source-stem>-passthrough-offline.mp4`。
- 复用当前热路径：
  - `PyNvSimpleDecoder`
  - `Matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile`
  - `GpuNv12AppFrame`
  - NVENC
  - FFmpeg MP4 mux
- 默认 engine 是 `rvm`，用于验证离线输出链路。
- 预留 `--engine matanyone2_onnx --model ... --mask ...`，但当前会明确报错，
  避免误以为 MatAnyone2 已经完成 ONNX 接入。

示例：

```powershell
python tools\offline_passthrough.py "VR\VR108\xxxx.mp4" --duration 10 --out debug_output\offline_test.mp4
```

## MatAnyone2 ONNX 转换风险

需要先确认并拆分模型边界：

- 初始化图：
  - 输入：第一帧 RGB、第一帧 mask、objects。
  - 输出：初始 memory / hidden state / alpha。
- 逐帧 step 图：
  - 输入：当前帧 RGB、上一帧 memory / hidden state。
  - 输出：当前 alpha、新 memory / hidden state。

风险点：

- 官方 PyTorch `InferenceCore.step()` 可能包含 Python 控制流和动态对象管理。
- 直接 `torch.onnx.export(InferenceCore)` 可能失败，或导出后不适合 ONNX Runtime
  逐帧调用。
- 模型输出不是单纯 alpha，可能需要保留多路 memory tensor。
- 第一帧 mask 必须来自外部：手工 mask、SAM/SAM2、或未来 UI 交互。
- 8K SBS 离线处理时，MatAnyone2 的输入分辨率、左右眼拆分、显存占用需要单独
  评估。官方推理支持 `--max-size` 下采样，但 passthrough 输出应保持原分辨率，
  所以 alpha 需要高质量上采样回源尺寸。

## 下一步建议

1. 用当前 `rvm` engine 验证 `tools/offline_passthrough.py` 能生成可播放 MP4。
2. 拉取 MatAnyone2 到 `reference/MatAnyone2` 或外部临时目录，先跑官方
   PyTorch 推理确认 checkpoint 和 mask 工作。
3. 写 `tools/export_matanyone2_onnx.py`，目标不是一次导出完整视频流程，而是
   先导出最小 step 模型。
4. 写 `pipeline/matanyone2_onnx.py`：
   - 封装 ONNX Runtime session；
   - 管理 memory state；
   - 接收 PyNv 解码帧或从 GPU/CPU 转换后的 RGB；
   - 返回 alpha。
5. 替换 `tools/offline_passthrough.py` 中的 `MatAnyone2OnnxEngine` stub。

## 轻量 ONNX 人体检测/分割 + MatAnyone2 的推荐方案

`vr-masking-tools` 的核心不是“MatAnyone2 自己找人”，而是两阶段：

1. 对每个短 segment 的左右眼第一帧做目标识别/分割，生成第一帧 mask。
2. 把该 segment 视频和第一帧 mask 输入 MatAnyone2，让 MatAnyone2 做时序传播。

原项目用的是 finetuned SAM3，因此依赖 torch。我们的应用层只允许 ONNX，所以应
替换成 ONNX detector/segmenter：

- 第一优先：轻量实例分割 ONNX，例如 YOLOv8n-seg / YOLO11n-seg 导出的
  ONNX。
  - 输入：第一帧 RGB。
  - 输出：person boxes + mask coefficients/prototypes。
  - 后处理：NMS、选择 person 类、按 VR 规则选目标、生成二值/灰度 mask。
- 第二选择：人体解析/显著性分割 ONNX，例如 MODNet/RMBG/BEN2 类模型。
  - 优点：直接输出 alpha/mask。
  - 缺点：可能不是“指定人”，多人场景或背景人体会不稳定。
- 不建议应用层使用 SAM/SAM2/SAM3 的 PyTorch 版本。
  - 如需 SAM 类模型，应该在外部 torch 环境中导出 ONNX 后，仅把 ONNX 和
    后处理代码接入本项目。

推荐离线流程：

1. PyNv 解码源视频。
2. 按固定 segment 长度切分，例如 5-10 秒。MatAnyone2 的时序传播越长，漂移
   风险越高；短 segment 可以重新锚定第一帧 mask。
3. SBS 视频左右眼分别处理：
   - 第一帧裁出 left/right eye。
   - 可选 center crop，减少 VR 圆边和黑边干扰。
   - ONNX person segmenter 生成 left/right 第一帧 mask。
4. MatAnyone2 ONNX：
   - 对 left segment 调用 init/step。
   - 对 right segment 调用 init/step。
   - 输出左右眼 alpha。
5. 把左右眼 alpha 拼回 SBS alpha。
6. 用现有 GPU composite kernel 把源视频前景合成到透视/绿幕背景。
7. NVENC 编码为 HEVC MP4，并 copy/aac 音频。

需要新增的模块边界：

- `pipeline/person_mask_onnx.py`
  - `PersonMaskOnnx(model_path)`
  - `mask_first_frame(rgb) -> np.ndarray`
  - 支持阈值、NMS、person class、最大面积/中心优先选择。
- `pipeline/matanyone2_onnx.py`
  - `MatAnyone2Onnx(init_model, step_model)`
  - `init(frame_rgb, mask) -> alpha, state`
  - `step(frame_rgb, state) -> alpha, state`
- `tools/export_yolo_seg_onnx.md` 或脚本
  - 在外部 torch 环境中导出 ONNX。
- `tools/export_matanyone2_onnx.py`
  - 在外部 torch 环境中导出 init/step ONNX。

应用层依赖原则：

- 本项目只新增 `onnxruntime-gpu` 已有能力和 NumPy/OpenCV 后处理。
- 不在 `pyproject.toml` 引入 torch/torchvision/SAM3。
- torch 只存在于外部模型导出环境。

落地顺序：

1. 先接 YOLO-seg ONNX 第一帧 mask，验证能从左右眼首帧得到合理人体 mask。
2. 做一个 `--engine yolo_mask_debug`，只输出 mask PNG/MP4，不跑 MatAnyone2。
3. 导出并验证 MatAnyone2 ONNX init/step。
4. 把 MatAnyone2 alpha 接入 `tools/offline_passthrough.py`。
5. 最后优化 GPU/CPU 拷贝，避免把整条离线链路退回 CPU。

## 验证

- `python -m compileall tools\offline_passthrough.py` 通过。
- `git diff --check -- tools\offline_passthrough.py` 通过。
- `python tools\offline_passthrough.py --help` 正常输出 CLI 帮助。
