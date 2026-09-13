# [DLSS5] 通道开发计划

> **状态（2026-09-13）：任务 A / B / C 已完成。** 实施结果、实测数据与三处计划外的
> 发现（CUDA 主上下文 blocking-sync 前置条件、fps × 分辨率表、参数默认值原先是错的）
> 见 [HANDOVER_20260913.md](HANDOVER_20260913.md)。本文保留为设计依据。

面向接手 `[DLSS5]` Neural Rendering 展示通道的开发者。目标：把已经跑通的
离线 GPU 推理（`DLSS5Stage`）接成一条完整的对外通道 —— **UI 配置入口 +
seek 虚拟文件通道 + live 直播通道**，形态与我们已上线的 **VSR 1X 原生增强 /
`[SUPERRES]`** 完全对齐。

## 0. 为什么可以照抄 SuperRes 原生

DLSS5 NR 是 **1x 增强，不改变分辨率**：输出尺寸 == 源尺寸。这让它落在 seek
通道里**最简单的一档**，和 VSR 原生 1x 完全同构：

- 天然可 seek（分辨率不变，源 MP4 的 moov 外壳直接可用）
- 允许和 Alpha 一样的虚拟文件播放模式
- **不需要** SuperRes 6K 那套 `_vmp4_slot_output_size` 放大分支和帧预算放大
  （那是为放大档位加的，DLSS5 用不到）

所以整条链是「把 `superres` 这个 mode 旁边再加一个 `dlss5`」，绝大多数改动是
在现有 `frozenset`/`tuple`/分派表里多加一项，加一个 `DLSS5Stage` 构造分支。

## 1. 现状盘点（已完成）

| 模块 | 状态 |
|---|---|
| `models/dlss5/neural_bridge.py` | ✅ `BRIDGE` 单例，ABI v6，`process_cuda_pointers` |
| `pipeline/dlss5_stage.py` | ✅ NV12↔RGBf32 CUDA kernel + NR 评估，GPU 驻留 |
| `utils/dlss5.py` | ✅ `DLSS5Settings` / `is_dlss5_available` / `source_block_reason_dlss5` |
| `offline/dlss5_offline.py` | ✅ 离线单文件/批量（FFmpeg 解码→GPU→NVENC） |
| `offline/convert.py` | ✅ `--engine dlss5` 已注册 |
| `utils/vr_naming.py` | ✅ `DLSS5_PREFIX="[DLSS5]"`、`dlss5_output_stem` |
| `config.py` | ✅ `DLSS5_*` 全局配置项 |
| **UI 配置入口** | ✅ `DLSS5SettingsDialog` + 仪表盘卡片（2026-09-13） |
| **seek 虚拟文件通道** | ✅ `make_frame_stage` + `SEEK_FRAME_MODES`（2026-09-13） |
| **live 直播通道** | ✅ 同上，两处构造点已合并（2026-09-13） |
| **DLNA 展示（`[DLSS5]` 前缀 + 目录/虚拟文件项）** | ✅ `pldn_` / `sdn_` 前缀（2026-09-13） |
| **测试** | ✅ `tests/test_dlss5.py` + 并列用例（2026-09-13） |

> ⚠️ 下表原写"离线推理已跑通"。实际上 `offline/dlss5_offline.py` 当时缺
> `import numpy as np`，第一帧就 NameError；另有 NVENC preset 大小写问题。
> 两个都已修（见 HANDOVER_20260913 §5）。

`DLSS5Stage` 的 NV12→NR→NV12 端到端推理已在 RTX 5060 Ti 跑通。下面是把它
接成对外通道的三块工作。

---

## 2. 任务 A：seek + live 通道（核心，照 SuperRes 改）

参照实现就是 `[SUPERRES]` 原生 1x。逐个文件对照改。

### 2.1 `pipeline/pynv_stream.py`

- **`pynv_stream.py:307`** `SEEK_FRAME_MODES = frozenset({"green", "alpha", "superres"})`
  → 加入 `"dlss5"`。这是 seek 帧生成器认可的 mode 权威集合，其余守卫全部读它。
- **`pynv_stream.py:357`** 的 `if mode not in SEEK_FRAME_MODES` 守卫自动生效。
- **`pynv_stream.py:394`（live worker）** 和 **`pynv_stream.py:2467`（seek 帧生成器）**
  两处 `SuperResStage(...)` 构造点：加一个按 mode 分派的分支，`mode == "dlss5"`
  时构造 `DLSS5Stage(width=w, height=h, settings=DLSS5Settings.from_config())`。
  - **两处都要改**，这是历史上反复踩的「保持 N 份同步」坑（SuperRes 曾因为漏了
    一处守卫返回 409）。建议抽一个 `def _make_frame_stage(mode, w, h, settings)`
    统一构造，两个调用点都走它，避免再漏。
  - `DLSS5Stage.process_nv12(...)` 的签名已经和 seek 帧生成器要的 NV12 in/out
    指针形态一致，可直接用。

### 2.2 `http_app/routes_media.py`

- **`routes_media.py:266`** `_SEEK_ROUTE_MODES = ("green", "alpha", "superres")`
  → 加 `"dlss5"`。
- **`routes_media.py:269` `_split_seek_route_name`** 自动识别 `<key>.dlss5.seek.<ext>`。
- **`routes_media.py:6010` `_seek_output_mode`** 分派表里加 `dlss5`。
- **`_vmp4_slot_output_size`**：DLSS5 是 1x，输出尺寸 == 源尺寸，走**默认分支即可**，
  **不要**加放大逻辑。确认 `superres` 的放大分支不会把 `dlss5` 也带进去（按 mode
  显式判断，别用「非 green/alpha 即放大」的写法）。
- **`_vmp4_frames_frame_budget`**：同理，DLSS5 用源分辨率的默认帧预算，**不需要**
  SuperRes 6K 那套按输出像素放大的分支。
- 四处 `mode not in SEEK_FRAME_MODES` 守卫（`:2069 :2480 :3970 :4476`）读同一个
  集合，加了 `dlss5` 后自动放行。**改完逐一确认，不要漏。**

### 2.3 `dlna/content_directory.py`

- **`content_directory.py:976`** 标题前缀映射 `"superres": "SUPERRES"` 旁边加
  `"dlss5": "DLSS5"`。展示标题固定 `[DLSS5]`，**不要**把参数（intensity 等）写进
  标题 —— Skybox 会缓存文件名，一旦变了就永远变不回来（这是明确踩过的坑）。
- DLNA 的 live/seek 对象前缀（`SUPERRES_LIVE_PREFIX="plsr_"`、
  `SUPERRES_SEEK_ITEM_PREFIX="ssr_"` 等，`:96`–`:109`）：给 DLSS5 起一套独立前缀
  （如 `pldn_` / `pldn_item_` / `sdn_`），并在 `:318`–`:424` 的 `object_id` 分派、
  `:1021`–`:1104` 的前缀返回函数里加对应分支。
- **`content_directory.py:1146` `_seek_supported_modes`** 读 `SEEK_FRAME_MODES`，
  加 `dlss5` 后自动含。
- **`_seek_supported_mode`（`:1157`）**：DLSS5 恒为 1x，恒可 seek，返回 True 即可
  （不像 SuperRes 要看 `seek_supported_target()` 是否放大）。
- **`_passthrough_resolution`（`:1209`）**：DLSS5 分辨率不变，直接用源 `width×height`。
- **可用性门控**：只有 `is_dlss5_available()`（runtime 三件套齐全）时才在目录里
  列出 `[DLSS5]` 项，否则整条通道不出现 —— 干净 clone / 未部署 runtime 的机器不该
  看到一个点了播不出的条目。

### 2.4 GPU 会话 / stream 复用（务必注意的坑）

SuperRes 曾因为把 **CUDA stream 放进 NGX 会话 key**，导致每帧 pointer 不同、
240 帧重建了 241 次 NGX feature。`DLSS5Stage` 目前每次 `process_cuda_pointers`
传入 `stream_ptr=int(stream.ptr)` —— **务必确认 `neuroframe_engine.dll` 内部不会
按 stream 重建 feature**。接 seek/live 前先做这一步验证：固定分辨率连续跑 N 帧，
确认 NGX create 只发生一次。若发现每帧重建，会话 key 应只按 CUDA context 而非
stream 建（参见 `utils/rtx_vsr.py` 的同名修复）。

---

## 3. 任务 B：UI 配置入口

项目约定：新功能必须在 UI 有配置入口。DLSS5 现在完全没有。照 SuperRes 的对话框
和卡片做。

### 3.1 参数配置对话框

- 参照 `ui/dialogs/feature_dialogs.py` 的 `SuperResSettingsDialog`，新建
  `DLSS5SettingsDialog`。
- 需暴露的参数（对应 `DLSS5Settings` / `RenderParametersV6`）：`style`、`intensity`、
  `local_tone`、`local_structure`、`skin_structure`、`auto_mask`、`color_strength`、
  `tone_preservation`、`face_skin_protection`、`grain_preservation`、`nr_passes`、
  `shimmer_suppression`、`prefer_nvof`。先挑最能体现效果的几个（`intensity`、
  `style`、`shimmer_suppression`、`nr_passes`）做主控件，其余进「高级」折叠区。
- **播放模式选择器**：DLSS5 恒为 1x，恒可虚拟文件播放，直接复用
  `PlaybackModeChooser`（virtual / live），不需要 SuperRes 那个「只在原生档才显示」
  的可见性联动。
- 保存到 `settings.data`，键名统一加 `dlss5_` 前缀。

### 3.2 卡片入口

- `ui/pages/dashboard_page.py`：加一张 `[DLSS5]` 功能卡片，齿轮按钮打开上面的对话框。
- 卡片可用性：`is_dlss5_available()` 为假时，卡片置灰或不显示，并给出提示
  （runtime 未部署）。

### 3.3 翻译

- 三语文件 `ui/translations/{zh_CN,en_US,ja_JP}.json` 加 `dlss5.*` 键。
  注意文件是 **UTF-8 BOM**，读写用 `utf-8-sig`，保持缩进和键顺序（现有工具脚本用
  `OrderedDict` 增量写，不要整体重排）。

### 3.4 驱动/硬件提示（已有基建，直接复用）

- `utils/ngx_requirements.py` 已能读 `nvngx_dlssnr.dll` 声明的最低要求
  （驱动 615.00 / 架构 Blackwell2 = RTX 50），`ui/widgets/ngx_driver_notice.py`
  已有 `NgxDriverNotice` 控件。
- 在 `DLSS5SettingsDialog` 里加一个 `NgxDriverNotice`，`runtimes` 传
  `dlss5_runtimes()`（`utils/ngx_requirements.py` 已提供）。驱动/显卡不达标时自动
  显示「需要 RTX 50 + 驱动 615.00」，达标时不出现。这块**已经做好，只需接线**。

---

## 4. 任务 C：测试

参照 `tests/test_ngx_requirements.py`、`tests/test_ui_smoke.py`、
`tests/test_content_directory_modes.py`。

- **`tests/test_dlss5.py`**：`DLSS5Settings.from_config` / `to_params` 的字段映射；
  `source_block_reason_dlss5` 的尺寸/可用性判断（runtime 缺失时的 `dlss5_runtime_missing`
  分支可 mock `is_dlss5_available`）。不依赖真实 runtime。
- **seek 路由**：在 `test_routes_media_*` / `test_content_directory_modes.py` 里加
  `dlss5` 的路由拆分、mode 分派、`[DLSS5]` 标题、可 seek 判定用例，和 `superres`
  用例并列。
- **UI**：`test_ui_smoke.py` 加 `DLSS5SettingsDialog` 构造 + 参数读回 + 驱动提示
  显隐（mock 驱动版本）用例，照 `SuperResSettingsDialog` 现有用例。
- 运行方式（重要）：`.venv/Scripts/python.exe -m pytest tests`。base python 缺
  PySide6；不带 `tests` 路径会去 collect `reference/` 和 `runtime_cache/`。当前基线
  `669 passed, 2 skipped`。

---

## 5. 需要留意的技术点

1. **实时可行性要先测**。seek/live 通道是按**播放速度**拉流的，必须先测 DLSS5 NR
   在目标分辨率下的 fps。SuperRes 原生 1x 当时是实测确认能跟上才接的。若 DLSS5 在
   4K/8K VR 分辨率下跟不上播放速度，seek 通道对高分辨率源要给回退（只提供 live
   目录形态，或限制适用分辨率），别让播放器拉不动。**先出一张 fps × 分辨率表。**
2. **离线路径当前是 host 往返**。`offline/dlss5_offline.py` 每帧 `d_out_y.get()` +
   `.get()` 回主机再喂 NVENC。离线可以接受，但 **seek/live 通道必须全程 GPU 驻留**
   并复用 NVENC ring（参照 `SuperResStage` 的编码环），不要走 host 往返。
3. **色彩空间**。`dlss5_stage.py` 的 CUDA kernel 目前**写死 BT.709 limited range**。
   SD（BT.601）、full range、10-bit 源都会偏色或丢精度。接通道前确认源色彩空间的
   处理，或与 SuperRes 的转换路径统一。
4. **10-bit / HDR**。当前解码强制 `nv12`（8-bit），10-bit 源会被静默降到 8-bit。
   若要支持 HDR 源，需要 P010 路径（`FORMAT_P010=3` 在 bridge 里已定义，但未接）。
5. **模块级副作用**。`dlss5_stage.py` 顶部直接调 `configure_gpu_runtime_cache()`。
   确认在服务进程里不会与既有初始化重复/冲突（SuperRes 的 stage 是懒构造的）。

---

## 6. 建议顺序

1. **先测 fps × 分辨率**（决定 seek 通道对哪些分辨率开放），顺便验证 NGX feature
   不会每帧重建（§2.4）。
2. **接 seek + live 通道**（任务 A），先用现有的全局 seek 开关（Alpha/绿幕那套）
   验证形态，跑通「目录里出现 `[DLSS5]` 虚拟文件、能拖动、拖到哪生成到哪」。
3. **UI 配置入口**（任务 B），把开关和参数搬进 `DLSS5SettingsDialog`。
4. **补测试**（任务 C）。
5. 打包验证：`build_exe.py` 的 `verify_dlss5_runtime()` 和两个 `.spec` 已条件化
   包含 `models/dlss5/runtime/`，clean clone（无 runtime）也能构建、只是通道不出现。

---

## 7. 关键文件索引

| 作用 | 文件 |
|---|---|
| NR bridge 单例 | `models/dlss5/neural_bridge.py` |
| GPU stage（NV12↔RGB↔NR） | `pipeline/dlss5_stage.py` |
| 参数/可用性/尺寸校验 | `utils/dlss5.py` |
| 离线转码 | `offline/dlss5_offline.py` |
| seek 帧生成 / live worker | `pipeline/pynv_stream.py`（`SEEK_FRAME_MODES` `:307`，构造点 `:394`/`:2467`） |
| HTTP 路由 | `http_app/routes_media.py`（`_SEEK_ROUTE_MODES` `:266`，`_seek_output_mode` `:6010`） |
| DLNA 展示 | `dlna/content_directory.py`（标题映射 `:976`，可 seek 判定 `:1146`/`:1157`） |
| 命名 | `utils/vr_naming.py`（`DLSS5_PREFIX` `:67`） |
| 全局配置 | `config.py`（`DLSS5_*` `:84`+） |
| 驱动/硬件提示（已就绪） | `utils/ngx_requirements.py`、`ui/widgets/ngx_driver_notice.py` |
| 参照实现（SuperRes 原生） | `pipeline/superres_stage.py`、`ui/dialogs/feature_dialogs.py::SuperResSettingsDialog` |
