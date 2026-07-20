# 实时视频去马赛克（RM）功能对接 + FPS 优化小结

日期：2026-06-25
分支：`feat/demosaic`
关联提交：`5de2860` → `e459279` → `f409c1f` → `67ae7e7` → `549e205` → `7a4ce24` → `1ed81af`

---

## 1. 背景与目标

接入两个外部 ONNX 模型，实现实时视频去马赛克（Mosaic Restoration，简称 **RM**），并作为 DLNA `[RM]` 入口与离线生成功能上线：

- 检测模型 `models/demosaic/vr_mosaic_detection_model_v2_accurate.onnx`
  — YOLO11m-seg，输入 `1×3×640×640`，输出 `output0 (1,38,8400)` + `output1 (1,32,160,160)` mask 原型。
- 恢复模型 `models/demosaic/vr_mosaic_restoration_windows_model_v0.1.onnx`
  — 7 帧滑窗，输入 `1×7×3×256×256` RGB(0..1) → 输出 `1×3×256×256` 恢复后的**中心帧**（含 grid_sample，需 TRT≥8.5）。

目标：实时 60FPS（`videos/test_1080p_2d.mp4`），后扩展到 4K/8K 与 10-bit。

**关键约束**：项目不依赖 ultralytics，所有 YOLO 都用 `onnxruntime` 跑 → 检测器需自写 YOLO11-seg 后处理（解框/NMS/mask 原型重建）。

---

## 2. 功能架构（与实时 2D→3D / SI 对齐）

| 层 | 文件 | 内容 |
|---|---|---|
| 推理核心 | `pipeline/demosaic.py` | ORT 检测器(手写后处理)+恢复器；TRT fp16 引擎缓存；进程级共享引擎单例；`GpuRmProcessor`(GPU crop/resize/blend + IOBinding + 检测降频) |
| 实时管线 | `pipeline/pynv_stream.py::_worker_loop_rm` | NVDEC→cupy RGB(7帧窗)→检测→逐区域恢复→NVENC，帧常驻 GPU |
| DLNA 入口 | `dlna/content_directory.py`、`utils/vr_naming.py` | 注册 `"rm"` 到全套 mode 映射，`[RM]…_live` 入口，门控 2D 源(屏蔽 2:1 VR) |
| HTTP 流 | `http_app/routes_media.py` | 复用 `/passthrough_live?mode=rm`，`_rm_live_block_reason` |
| 运行时开关 | `utils/runtime_settings.py`、`http_app/routes_control.py` | `RmRuntime`(带 version 触发 DLNA 刷新)、`/control/rm` |
| 首页 UI | `ui/pages/home_page.py` + 三语 i18n | 「马赛克复原」开关(默认开) + 「离线生成」按钮 |
| TRT 预热 | `main.py` | server 启动阻塞预热(独立进度文案)；运行时切换后台预热 |
| 离线生成 | `offline/demosaic_offline.py`、`ui/pages/rm_page.py` | 全 GPU 离线通路；输出 `<stem>_<时间区间>_restored.mp4` |
| 基准工具 | `tools/bench_rm.py`、`tools/profile_rm.py` | FPS 基准 / 逐阶段 CPU·GPU profiler |

配置项（`config.py`）：`RM_ENABLED`(默认开)、`RM_CONF`(0.25)、`RM_DETECT_INTERVAL`(默认 2)。

---

## 3. FPS 优化历程（真实含马赛克片段，1080p，TRT fp16）

测试片段：`test_1080p_2d.mp4` 的 24:00–25:00（每帧恰好 1 个马赛克区域）。

| 阶段 | FPS | 每帧 ms | 关键改动 |
|---|---:|---:|---|
| 初版（host/numpy 路径） | 20.9 | 47.9 | 参考实现直译，7×全帧 `.get()` + cv2 缩放/掩码/混合 |
| + Opt2 GPU 化 | 34.6 | 28.9 | crop/resize/blend 全搬 cupy，仅小张量过 PCIe；自写 bilinear resize kernel(直接从全帧 crop 采样，无需连续拷贝) |
| + Opt1 检测降频 | 51.8 | 19.1 | 每 N 帧检测一次，中间复用框+GPU 掩码；恢复仍跑当前 7 帧窗 |
| + SPyNet 批处理（模型侧） | 58.8 | — | 模型侧把 6 次顺序 spynet 合成 1 次批处理(单次恢复 13.7→10.7ms) |
| **+ IOBinding/GPU 归一化** | **192.7** | **5.2** | 见下，决定性的一击 |

### 决定性优化：ORT IOBinding + GPU 归一化

用 `tools/profile_rm.py` 逐阶段拆解后发现：**真瓶颈不是模型推理，而是 CPU 上的输入归一化**（`astype(float32)/255 + transpose`）：

- 8K/2 区域下：恢复归一化 **13.95ms** + 检测归一化 **6.21ms** ≈ 20ms 纯 CPU；GPU 算子本身全部 <2ms。
- 我们一直以为的"恢复 ~13ms"其实大部分是 CPU 预处理，真正的 GPU 推理只有 ~2–3ms。

修复：归一化改在 GPU(cupy) 做 + **ORT IOBinding 直接绑定 GPU 显存指针**（检测器输入绑 GPU、输出回 host 做 NMS；恢复器输入+输出都绑 GPU，restored 全程不落地）。

效果（真实片段）：
- **1080p 1 区域：58.8 → 192.7 FPS（每帧 17→5.2ms）**
- **8K 2 区域：25.7 → 119 FPS；8K 每帧检测(interval=1) 也有 70.9 FPS**

正确性：GPU 路径 vs host 路径逐像素平均绝对差 0.025（仅缩放插值的微小舍入）。

> 模型侧原计划的「跨重叠窗特征缓存」被 profiling 推翻——feat_extract 仅 1.45ms，不值得做；真瓶颈 SPyNet 已由批处理解决。

### 检测降频甜点
有了 IOBinding 的余量后，默认 `RM_DETECT_INTERVAL` 设为 **2**（8K 仍 95FPS，框滞后 ≤1 帧，质量近每帧检测）。可用 `PT_RM_DETECT_INTERVAL` 按片源运动情况微调。

---

## 4. 离线生成功能

- 入口：首页「马赛克复原」行最右「离线生成」按钮 → 新页面 `ui/pages/rm_page.py`（仿 2D→3D 离线页，单个+批量标签页）。
- 后端 `offline/demosaic_offline.py`：**与实时同一条全 GPU 通路**（NVDEC→cupy→GpuRmProcessor→NVENC→ffmpeg 仅混音），非 ffmpeg rawvideo 主机往返。
- TRT 缓存先构建再转换（`_ensure_trt_cache` / `build-trt` 子命令），避免无缓存直接跑慢路径。
- 输出命名：`<stem>_S<起>_E<止>_restored.mp4`（区间+`_restored` 后缀），批量/全片为 `<stem>_restored.mp4`；保留源音频(按区间裁剪)。
- 进度：百分比 + 每 100 帧一次。
- 实测：1080p 单区域离线 **~114 fps**（暖管线）。

### 离线踩坑修复
- **音频混流死锁**：mux 的 stderr 未并发排空 → 缓冲满 → ffmpeg 阻塞。加 stderr 排空线程。
- **audio:0KiB**：`-shortest` + 裸 hevc 管道(无容器时长)会在音频写入前结束。去掉 `-shortest`（音频已由 `-t` 裁剪，时长与视频一致）。

---

## 5. 10-bit（P016/P010）GPU 支持

为 4K/8K HDR/10-bit 源扩展，**四条 GPU 路径全部支持 10-bit，无 ffmpeg 回退**：

| 路径 | 之前 | 现在 |
|---|---|---|
| RM 实时 / 离线 | 拒绝/跳过 >8-bit | ✅ |
| two_dvr 实时 | 拒绝 >8-bit | ✅ |
| two_dvr 离线 GPU | 按 8-bit 解码 | ✅ |

机制：NVDEC 解出 P016/P010(uint16) → 新增 cupy kernel `p016_to_rgb` / `p016_to_rgb_letterbox`（高位右移 `PASSTHROUGH_PYNV_10BIT_SHIFT=8` + BT.709 limited-range）→ 8-bit RGB 喂模型 → 8-bit 输出（与现有 experimental 10-bit passthrough 一致；测试源均为 **SDR** 10-bit/bt709，降位仅精度损失无偏色）。

涉及文件：`offline/two_dvr_pynv.py`(新增 letterbox kernel + convert_clip_pynv 探测 bit_depth 并分支)、`pipeline/pynv_stream.py`(两个 worker 分支)、`offline/demosaic_offline.py`、`http_app/routes_media.py`(移除 8-bit 限制)。

验证：
- RM 8K(8192×4096) 10-bit 离线产出 video+audio，颜色无偏。
- two_dvr 10-bit 1080p 离线 SBS 输出与同内容 8-bit **逐帧几乎一致**（帧30: 10bit [26.1,31.9,25.4] vs 8bit [26.2,32.2,25.4]）；排查发现首帧绿色是 two_dvr 固有 slate 帧，非位深问题。

---

## 6. 最终性能总账

| 场景 | FPS |
|---|---:|
| 1080p 1 区域（实时核心） | ~193 |
| 8K 2 区域 | ~119 |
| 8K 每帧检测(最佳质量) | ~71 |
| 1080p 离线生成(含编码+混音) | ~114 |

实时 60FPS 目标达成且对 4K/8K 有大量余量。

---

## 7. 后续可选项（均非必要）

1. **多区域批处理**：模型侧已提供 `--dynamic-batch` ONNX；一帧多马赛克时把多个 256 窗堆 batch 一次推理（当前逐区域循环）。
2. **检测器异步线程**：消除检测帧的瞬时尖峰（当前 interval=2 尖峰已很小）。
3. **真机端到端**：DLNA `[RM]` 入口 + NVDEC/NVENC/推流的实际观感确认。
4. 10-bit **保留输出**（当前降位 8-bit）：若遇 HDR(PQ/HLG) 源需 NVENC Main10 + tone-map，目前测试源均为 SDR 故未做。

---

## 8. 关键文件索引

- 推理核心：`pipeline/demosaic.py`
- 实时管线：`pipeline/pynv_stream.py`（`_worker_loop_rm`、`_worker_loop_two_dvr`）
- 离线后端：`offline/demosaic_offline.py`、`offline/two_dvr_pynv.py`
- GPU kernel：`offline/two_dvr_pynv.py`（`_NV12_RGB_KERNELS`：nv12/p016 ↔ rgb、letterbox、rgb→nv12、bilinear_resize）
- UI：`ui/pages/home_page.py`、`ui/pages/rm_page.py`、`ui/main_window.py`
- DLNA/HTTP：`dlna/content_directory.py`、`http_app/routes_media.py`、`http_app/routes_control.py`
- 工具：`tools/bench_rm.py`、`tools/profile_rm.py`
