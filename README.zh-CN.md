# VR视频透视服务器

中文 | [English](README.md) | [日本語](README.ja-JP.md)

VR视频透视服务器 的目标是让所有VR视频都可以透视，实现混合现实(MR)。

![VR视频透视服务器概览](resources/intro_cn_s.png)

它是以 Windows 为主要运行平台的 VR DLNA 本地媒体服务器，兼顾桌面控制和离线生成流程。它通过 DLNA/UPnP 暴露本地视频库，并支持实时透视流输出，可在绿幕合成和 Alpha 直通之间切换，同时支持实时字幕嵌入。VR视频透视服务器 主要面向 VR180 半等柱体投影（half-equirectangular）视频源。

## 项目起源

本想做一个 VR 视频透视工具。  
有人说: 你这是重复造轮子。  
我说：那个多年前的旧轮子已经太老了，该换新的了。  
七天后，新轮子诞生了。  
这是属于 AI 时代的奇迹。  

## 功能

- DLNA 发现与 视频资源目录 浏览
- 基于 GPU 抠像和 HEVC 编码的实时透视串流
- 实时透视流内嵌字幕
- 绿幕模式与 Alpha 直通模式
- 离线透视视频生成
- 支持多个本地视频根目录
- PySide6 桌面 UI，支持中文、英文、日文
- 字幕预览与字幕样式配置
- 面向 8K 级源视频的显存与吞吐优化，尽量保持硬件可承受范围内的实时 30fps 输出

## 透视视频效果图

| Alpha Passthrough | 绿幕 Passthrough |
| --- | --- |
| ![Alpha Passthrough 效果图](resources/sample_alpha.jpg) | ![绿幕 Passthrough 效果图](resources/sample_green.jpg) |

## 运行要求

- Windows 10 / 11
- Python 3.12
- NVIDIA GPU，用于实时处理链路。粗略建议使用 RTX 20 系列及以上，具体型号请查询 NVIDIA 官方列表：<https://developer.nvidia.com/cuda/gpus>。推荐显存：实时服务器和 RVM 离线生成建议 6 GB 以上，MatAnyone2 / SAM3 离线流程建议约 15 GB 以上。
- FFmpeg / FFprobe

## 快速启动

```bash
uv run python main.py
```

启动桌面 UI：

```bash
uv run python -m ui.app
```

## 支持的 VR 视频播放器

基于 Meta Quest 3 设备测试。

| 播放器 | Alpha 直通 | 灰色绿幕 | ChromaKey 绿幕 | 网站 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Skybox VR Player 2.0.2 Preview | 支持 | - | 支持 | [官网](https://skybox.xyz) | [安装说明](https://forum.skybox.xyz/d/2920-skybox-quest-v202-preview-performance-improvements) |
| Moon Player | - | 支持 | 支持 | [官网](https://moonvrplayer.com) | - |
| 4XVR Video Player | 支持 | - | 支持 | [官网](https://www.4xvr.net/) | - |
| DeoVR player | 支持 | - | 支持 | [官网](https://deovr.com/) | - |
| HereSphere VR Video Player | 支持 | - | 支持 | [官网](https://heresphere.com/) | - |

## 配置说明

- `PT_VIDEO_DIR` 支持用 `|` 分隔的多个目录
- `PT_PASSTHROUGH_OUTPUT_MODE` 支持 `none`、`green`、`alpha`、`all`
- Alpha 模式下虚拟条目标题为 `Alpha Passthrough`
- UI 配置与后台运行配置分离保存

## 项目结构

```text
main.py        服务入口
config.py      运行时配置
dlna/          UPnP / DLNA 发现与目录
http_app/      FastAPI 路由
pipeline/      解码、抠像、编码、缩略图、字幕流水线
offline/       生产用离线转换入口
ui/            PySide6 桌面 UI、页面、国际化与进程控制
tools/         开发探针和诊断工具
models/        本地模型文件与清单
resources/     打包用 UI / 运行时资源
prompt/        交接记录与调研文档
```

## 引用的开源模型

VR视频透视服务器 本身不训练抠像模型，只使用下列上游项目提供的模型与模型文件。

| 模型 | 用途 | 上游链接 |
| --- | --- | --- |
| Robust Video Matting (RVM) | 实时主抠像路径，包括 `rvm_mobilenetv3_fp32.onnx` 和 `rvm_resnet50_fp32.onnx` | [GitHub](https://github.com/PeterL1n/RobustVideoMatting) |
| MatAnyone2 | 离线转换与实验流程中使用的更慢但通常更高质量的抠像路径 | [GitHub](https://github.com/pq-yang/MatAnyone2) |
| Segment Anything Model 3 (SAM 3) | 用于实验性 Alpha 工具和预处理流程的辅助分割模型 | [GitHub](https://github.com/facebookresearch/sam3) |

## 引用的依赖

- [PySide6](https://www.qt.io/qt-for-python)
- [FastAPI](https://github.com/fastapi/fastapi)
- [Uvicorn](https://github.com/encode/uvicorn)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [CuPy](https://github.com/cupy/cupy)
- [PyNvVideoCodec](https://github.com/NVIDIA/VideoProcessingFramework)

## 说明

- 本项目当前主要面向本地 Windows 机器运行，而不是作为托管服务部署。
- Alpha 直通在 DLNA 中显示为虚拟条目 `VR Passthrough Server`。
- 英文版请见 [README.md](README.md)，日文版请见 [README.ja-JP.md](README.ja-JP.md)。

## 许可

许可：`AGPL-3.0-or-later`。


项目许可见仓库根目录的许可证文件。上游模型仓库各自保留自己的许可与使用条款。
