# 离线 2D→VR/3D 功能开发与性能优化汇总（2026-06-15）

分支：`feature/2dvr`（基于 `master`，3 个 commit，未 push——本仓库未配置 remote）。

- `46e3e5f` Add offline 2D->VR/3D feature (DA3 depth via ONNX Runtime)
- `7efb212` Speed up 2D->VR render ~8x (cv2 resampling, default inverse_warp)
- `7f0cf63` Optimize offline 2D->VR throughput ~18x (TensorRT depth + GPU/cv2 render)

本日工作分三段：(1) DA3 深度模型转 ONNX；(2) 离线 2D→VR 页面与后端；(3) 吞吐性能优化。

---

## 一、背景与目标

`G:\GIT\debug\VR_Video_Toolbox_NE\tool_2dvr` 是一个独立的 **纯 PyTorch** 2D→VR 工具
（Depth Anything 3 单目深度 → 双目视差 → VR 投影）。本项目（PTMediaServer）整套是
**onnxruntime + CUDAExecutionProvider** 架构（rvm/birefnet/matanyone2/sam3/yolo26m 全是
`.onnx`）。目标是把 tool_2dvr 移植成与本项目一致的 ORT 实时/离线管线，入口与首页"离线
透视"按钮并排。

第一步必须先解决 **DA3 → ONNX**。HuggingFace 上只有 `lacykaltgr/DA3-BASE-ONNX`（BASE，
DINOv2 ViT-L），与 SMALL（ViT-S，维度不同）不兼容，**SMALL 必须自己转**。

---

## 二、DA3 → ONNX 转换（`examples/da3_to_onnx.py`）

### 设计

- 只导出 **depth-only 子图**：`DepthAnything3Net.forward(..., skip_camera=True,
  skip_sky=True)`。这条路绕开相机/天空/3DGS 分支里 `torch.quantile` / `randint` /
  `.item()` / 布尔掩码等数据依赖控制流（ONNX 不可导）。剩下 DINOv2 + DualDPT，纯前向。
- **固定 518 方形输入**：恰好命中 DINOv2 原生 37×37 patch 网格，跳过 pos-embed 的 bicubic
  插值；只有 batch 轴 dynamic。
- 后处理（归一化等）留在 ORT 外，图只吐 `output["depth"]`。

### 踩坑与修复

- RoPE 的 `PositionGetter` 用了 `torch.cartesian_prod` —— **没有 ONNX 算子**。monkeypatch
  成等价 `meshgrid`+`stack`（常量网格，精度无损）。`_patch_position_getter()`。
- RoPE 用实值 sin/cos + `F.embedding`（非 `view_as_complex`），可导。
- SDPA 用 opset 18 可导。S=1 单视图时 `THRESH_FOR_REF_SELECTION=3` 让参考视图选择分支静态
  跳过 → 图静态。

### `--fold-preprocess`（性能优化阶段加上，现为生产用法）

- 图输入改为 **uint8 `(B,518,518,3)`**，在图内做 `permute + /255 + (x-mean)/std`。
- TensorRT 不支持 uint8 中间张量 → ORT 自动把 uint8 前处理切给 CUDA EP、ViT 主体留在 TRT。
  净效果：**省掉每帧 ~6ms 的 CPU 归一化**（搬到 GPU）。
- `da3_depth` 通过 `session.get_inputs()[0].type` 是否含 `uint8` 自动分支，兼容折叠/未折叠
  两种模型。

### 产物与环境

- `models/DA3/da3_small.onnx`（100MB）、`da3_base.onnx`（394MB）。**已 gitignore**，用脚本重生成。
- 用 Toolbox venv `G:\GIT\debug\VR_Video_Toolbox_NE\.venv`（含 torch）导出；`onnx` 包用
  `uv pip install` 装入。
- 校验：PyTorch ↔ onnxruntime 相对误差 ~4e-5（small/base，含折叠预处理）。CUDA provider 正常。

---

## 三、离线 2D→VR 页面与后端

方向确认（与用户）：**2D→VR/3D**；后端 **ONNX Runtime**（用刚转的 onnx）；页面 **单文件+批量**
两个标签页。

### 后端（纯净，无 torch 依赖）

- **`offline/da3_depth.py`** — `Da3DepthEngine`：
  - provider 链 `trt → cuda → cpu`，默认 `trt`（fp16，引擎缓存
    `runtime_cache/da3_trt/<variant>`）。
  - letterbox 到 518；折叠模型喂 uint8 canvas，未折叠模型走 `_normalize` 出 float NCHW。
  - `predict_batch(frames, upscale=False)` 返回模型分辨率深度（热路径用），`upscale=True`
    才 resize 回原帧。
- **`offline/two_dvr_render.py`** — 从 tool_2dvr `logic.py` 移植的纯 numpy/cv2 渲染：
  - 深度→`_normalize_near`（视差≈1/Z）→视差→双目。
  - `inverse_warp`（逆采样，快，默认）/ `soft_shift`（前向映射+遮挡补洞，画质，慢）。
  - 投影 flat3d / hequirect-180 / fisheye-180。
  - `StereoRenderer` 有状态类（缓存采样网格、预分配 SBS 缓冲）。
- **`offline/two_dvr.py`** — CLI：
  - `single` / `batch`，时间 `--start/--duration/--segment`，`--out-dir/--recursive/
    --skip-existing`，2D→VR 选项 `--model/--projection/--hole-fill/--eye-distance/
    --flat-fov/--max-side/--provider/--preset/--bitrate`。
  - 管线：ffmpeg 解码(rgb24) → DA3 深度 → 立体渲染 → hevc_nvenc 编码 + 音频。
  - import 时调 `apply_runtime_dll_paths()`（否则 ORT CUDA 静默回退 CPU；`utils.ffmpeg_checker`
    是 tool_2dvr 的、本项目没有，`get_startupinfo` 改用 `hidden_subprocess_kwargs()`）。
  - 输出命名 `<stem>_2dvr_<model>_<proj>_LR_SBS[_S..E..].mp4`。

### UI

- `ui/pages/two_dvr_page.py`：单文件+批量两 tab，复用 `offline_page` 的时间范围/分段助手
  （`_resolve_time_range` / `_resolve_time_segments` 等），选项：深度模型(small/base)、投影、
  补洞、瞳距、处理精度（fast=1280 / balanced=1920 / hq=原始 → `--max-side`）、时间范围/分段。
- `ui/services/offline_process.py`：新增 `TwoDvrProcess`（`OfflineProcess` 子类，重写
  `_command()` 钩子）；`OfflineProcess` 抽出 `_command()` 便于覆盖。
- `ui/services/process_helpers.py`：`two_dvr_command()`。
- `ui/main_window.py`：注册页面、stack 导航、返回/首页按钮、`open_two_dvr`、服务器/离线
  互斥保护、closeEvent 停止进程。
- `ui/pages/home_page.py`：`two_dvr_button` 与 `offline_button` **并排**（水平 row）。
- 三语 i18n：`button.two_dvr` + `twodvr.*`（zh/en/ja），43 个页面引用键全部校验通过。

---

## 四、性能优化（重点）

⚠️ 测试文件 `videos/test_2d_4k.mp4` **实测 1920×1080（1080p）**，非 4K（HEVC 60fps，60s）。
GPU：**RTX 5060 Ti，sm_120 (Blackwell)**。

### 优化前后（SMALL，inverse_warp，flat3d）

| 处理分辨率 | 初始(纯numpy) | cv2优化后 | 最终(TRT+折叠) |
|---|---|---|---|
| 1920 (满 1080p) | 1.5 fps | ~9 fps | **27 fps** |
| 1600 | — | — | 35.6 fps |
| 1280 | — | ~12 fps | **47.6 fps** |
| 960 | — | 15 fps | ~50 fps |

**总提升约 18×。40fps 落在 ~1450px。**

### 逐项分析（1080p 单帧）

诊断：深度（GPU）不是瓶颈，**渲染（CPU numpy）才是**（初始 ~580ms）。

**深度：38ms(CUDA fp32) → ~5ms**
- **TensorRT fp16 EP**（引擎缓存）：融合整个 ViT，**消除 53 个 CUDA memcpy 节点**——单个
  最大杀手。仅 TRT run ≈ 4.7ms。
- **折叠预处理**：归一化搬到 GPU，省 ~6ms CPU。predict 端到端 14→5ms。
- 坑：CLI `--provider` 默认曾是 `cuda`，覆盖了引擎默认 `trt` → 实际跑 CUDA。改默认为 `trt`。

**渲染：89ms → ~16ms**
- flat3d 之前对**恒等投影图**做了无意义的双线性重采样（~390ms）→ 改为直接透传左右眼
  (`ProjectionMap.is_identity`)。
- `_normalize_near` 之前用全帧布尔掩码 + `np.percentile`（42ms）→ 改为**在深度模型低分辨率
  (~518×291) 上归一化**、视差上采样到原帧（深度/视差是低频平滑的）。normalize 42→1.8ms。
- `StereoRenderer` 缓存 cols/map_y、预分配 SBS 与视差/映射 scratch，`np.subtract|add(out=)`、
  `cv2.resize(dst=)` 全程零分配；`cv2.remap` 直接写进 SBS 两半（无 concat）。
- `cv2.remap` 取代 numpy 双线性；`cv2.GaussianBlur/dilate/blur` 取代 numpy 循环；
  `_shift_fill_holes_rgb` 向量化（原本按列循环最多 width 次）。

**管线**
- 3 段线程：`深度(GPU) | 渲染(CPU/cv2) | 编码(ffmpeg)`。重活（TRT run、cv2.remap、管道 IO）
  都释放 GIL，故深度(N+1) 与渲染(N) 真正重叠。
- writer 直接 `enc.stdin.write(ndarray)`（buffer 协议，无 `tobytes`）。
- 注：早期"reader+compute+writer"三线程把 depth+render 放同一线程 → GIL 抖动反而更慢
  (14fps)；改成 depth/render 分线程才有效。

### 各管线段实测（参考）

- 解码 only（1080p hevc）：126 fps（非瓶颈）。
- 编码 only（3840×1080 hevc_nvenc p5）：91 fps（非瓶颈，NVENC 争用约 -6fps）。
- 深度+渲染（无编码）：~22 fps（真实冷缓存）；加编码 ~16-18 fps；线程化后稳定 ~21-27 fps。

---

## 五、为什么满 1080p / 真 4K 到不了 40fps

满 1080p 卡 27fps 是**渲染受限**——两次全分辨率 `cv2.remap`（~9ms）在 CPU + GIL 是硬地板。
突破需要 **GPU 上做 remap**，但本机被锁死：

- **CuPy 14 装了但跑不起来**：sm_120 无 kernel image（`CUDA_ERROR_NO_BINARY_FOR_GPU`，
  NVRTC 12.x 不出 sm_120）。
- **torch 未安装**；`cv2.cuda` 0 设备。
- 唯一可用 GPU 计算是 onnxruntime（已被深度占用）。试过 ORT GridSample 做 warp，但每帧
  33MB grid 传输反而更慢（31ms）。
- 真 4K（3840×2160）渲染像素是 1080p 的 4 倍，光两次 remap ~64ms，本栈下 40fps 不现实。

---

## 六、给用户的选择 / 后续

1. **要 40fps+：处理分辨率用 ≤1280（`fast` 档），47fps**，VR 头显里画质损失基本不可见。
2. **要满分辨率/4K 高帧率**：装 torch cu128，把 warp 重写成 GPU 张量版（或等 CuPy sm_120
   轮子）。那满 1080p 上 40fps、4K 接近实时都有戏。
3. 可考虑把 UI `balanced` 默认从 1920 下调，让默认即达 40fps（取舍画质）。

---

## 七、验证清单

- ONNX：small/base 折叠导出，PyTorch↔ORT 相对误差 ~4e-5。
- 真实跑通：small/base、inverse_warp/soft_shift、flat3d/fisheye、single/batch/segments；输出
  尺寸正确（flat3d 3840×1080、fisheye 每眼方形）；TRT provider 实际生效。
- `py_compile` 全过；三语 JSON 合法；43 个 i18n 键齐全；跨模块导入校验。
- **未在 Qt 中实跑**：本沙箱裸启 python 时 PySide6 DLL 加载失败（环境问题，非代码）。页面用
  静态校验，逻辑严格对齐可正常工作的 `offline_page`。**建议在正式 app 入口点一下确认 UI。**

## 八、已知问题

- soft_shift 慢（前向映射 scatter 是 numpy/CPU，~457ms@1080p），定位为画质模式。
- ORT 警告 `ScatterND with reduction=='none'` 来自深度图的 `x[:,:,0]=cam_token` 切片赋值，
  无害（数值已校验）；不值得为它重导出。
- da3 onnx（500MB）和 TRT 引擎缓存（`runtime_cache/da3_trt`）均 gitignore。
