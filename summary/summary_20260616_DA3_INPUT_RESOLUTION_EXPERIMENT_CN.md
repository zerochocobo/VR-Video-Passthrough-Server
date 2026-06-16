# DA3 输入分辨率对抠边质量的影响实验（2026-06-16）

## 背景

2D→3D 的 DIBR 抠边残留(头发/衣服边缘)主要受 **DA3 深度边界清晰度**限制。此前用 base@518。
本实验验证:**提高输入分辨率**是否比换更大模型更能改善边界(见
`summary_20260616_DA3_MODEL_CHOICE`/对话结论:我们只需单目相对视差,METRIC/NESTED/GIANT
都不对路,分辨率才是头发的主瓶颈)。

## 导出

`examples/da3_to_onnx.py` 加了按尺寸命名(518 保持原名,其它加 `_<size>` 后缀)。用
Toolbox venv(torch 2.8 cu128)导出 base 两档(均 `--fold-preprocess`,uint8 输入):

- `models/DA3/da3_base_700.onnx`(700=14×50,393.9MB,torch↔ORT rel 6.6e-5)
- `models/DA3/da3_base_1036.onnx`(1036=14×74,393.9MB,rel 3.4e-5)

非 518 尺寸不再命中 DINOv2 pos-embed 快路径,改走 bicubic pos-embed 插值——因输入尺寸在
导出时是静态的,插值折叠成常量,导出/校验均通过。模型 gitignore(脚本可复现)。

## 结果

### 深度边界(头发左缘热力图,518 | 700 | 1036)
分辨率越高,前景/背景边界**越锐利**:
- 518:边界有明显数像素宽的**软过渡带**(就是产生 sliver 的根源)。
- 700:过渡带收窄。
- 1036:过渡带几乎消失,边界干净,翘起的发丝可分辨。

→ **确认:提高输入分辨率直接锐化深度边界、减少 matting 残留**,是头发抠边的主要杠杆。

### 渲染输出(rim=0 原始 hybrid,左眼头发)
1036 的发丝边比 518 更紧、更自然;但因 hybrid+rim 已经擦掉大部分,成品差异是**增量的**、
不如深度图差异明显。

### 速度(TRT 稳态,含 CPU letterbox 预处理)
| 尺寸 | depth ms/帧 | 相对 |
|---|---|---|
| 518 | 12.3 | 1× |
| 700 | 19.5 | ~1.6× |
| 1036 | 48.1 | ~4× |

(实时 GPU 路径用 folded uint8 + GPU letterbox kernel,纯推理比这更快;此处比值有参考意义。)
空洞占比三档接近(~1.6%,空洞量取决于视差不取决于深度分辨率)。

## 建议

- **离线高质量档**:用 **1036**(或 700)。速度非关键,换来最干净的边界。
- **实时**:**700** 是平衡点(~1.6× depth,大概率仍可 ~60fps);**1036 ~4× 大概率掉出
  60fps**,实时不建议。
- 边界变干净后,前景膨胀 `_dilate_near_fg`/toggle 的鼓边可考虑减小。

## 集成注意(尚未接线)

要把这两档接入生产,需要:
1. `Da3DepthEngine`/`MODEL_FILES` 支持按尺寸选模型(目前 size 硬编 518,MODEL_FILES 只有
   small/base);realtime/offline 加一个分辨率选项或档位。
2. **TRT 引擎缓存目录冲突**:`_trt_cache_dir(variant)` 只按 variant(`base`),三个尺寸会
   共用 `runtime_cache/da3_trt/base`。TRT 按输入 shape 哈希生成独立 `.engine`,可共存,但
   `trt_engine_cached()`/`_ensure_trt_cache()` 的"存在任一 engine 即就绪"判断需要改成
   按尺寸,否则会误判已就绪 / 首帧触发不同 shape 的重新构建(700/1036 首次构建各 ~40–60s)。
3. 预热流程(`build-trt`)需对选定尺寸各跑一次。

模型已导出可直接用 `Da3DepthEngine(variant="base", model_path=..., size=700/1036)` 测试。
