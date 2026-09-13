# 离线超分 TrueHDR（HDR10）实现记录（中文）

日期：2026-09-11
分支：`DLSS`
状态：已实现并实测通过

调研背景见 [summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md](summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md)，本项是其中「建议执行顺序」的第 1 项。

---

## 1. 做了什么

离线超分的「HDR 观感」新增第四档 **TrueHDR**，走 NVIDIA RTX Video 的 NGX TrueHDR 特性，输出真正的 HDR10（HEVC Main 10 + BT.2020/PQ），而不是原有的 SDR 内调色。

关键前提：native 桥 `models/rtx_vsr/native/rtx_video_api_cuda_impl.cpp` **早就实现了 TrueHDR**（create/evaluate/VSR→THDR 串联/中间纹理都在），只是 Python 侧 `rtx_video_api_cuda_create(..., THDREnable=0, VSREnable=1)` 一直把它关着。所以本次**没有重新编译 native bridge**，只补了运行时资产和 Python/CUDA/UI 链路。

---

## 2. 改动清单

### 资产与打包

| 文件 | 改动 |
|---|---|
| `models/rtx_vsr/runtime/nvngx_truehdr.dll` | 从 `reference/RTX_Video_SDK_v1.1.0/bin/Windows/x64/rel/` 复制（与 `nvngx_vsr.dll` 同一份可再分发许可） |
| `build_exe.py` | `verify_rtx_vsr_runtime()` 的必需资产列表加入 `nvngx_truehdr.dll` |

打包方式无需改动：`build_exe.py` 与两个 `.spec` 都是以**整目录**方式带上 `models\rtx_vsr\runtime`，新 DLL 自动进入产物；新模块 `pipeline/true_hdr.py` 由 `collect_submodules('pipeline')` 收集。未引入任何新的第三方依赖。

### 核心链路

| 文件 | 改动 |
|---|---|
| `pipeline/true_hdr.py`（新增） | `abgr10_to_p010` CUDA kernel、HDR10 色彩元数据、`is_true_hdr()`、`p010_grid()` |
| `pipeline/pynv_io.py` | 新增 `GpuP010AppFrame` 与 `_RawCudaPlaneView`（PyNv 只接受 `\|u2` typestr 和它自己的 P010 plane 布局，CuPy 的描述符会被拒） |
| `utils/rtx_vsr.py` | `TrueHdrSettings` 数据类（按 SDK 范围钳制）；`initialize()/initialize_cupy()` 增加 `true_hdr`；context key 纳入该标志，切换模式时 shutdown 重建；`evaluate_deviceptr()` 传 THDR 参数并拒绝模式不匹配；`process_cupy_rgba()` 增加 `out=` 复用与 TrueHDR 输出（`uint32` HxW） |
| `pipeline/hdr_look.py` | 模式集合加入 `truehdr`，且对 SDR kernel 是 no-op |
| `offline/rtx_vsr_pynv.py` | TrueHDR 分支：packed 输出缓冲、P010 环形缓冲、`P010` 编码器、HDR10 色彩参数；左右眼输出缓冲改为会话级预分配 |
| `offline/rtx_vsr.py` | `hdr_settings` 贯通；FFmpeg rawvideo 回退路径遇 TrueHDR **明确拒绝**（rc=3），不静默输出 SDR |
| `offline/convert.py` | CLI `--rtx-vsr-hdr-look truehdr` 与四个 `--rtx-vsr-truehdr-*` 参数 |
| `pipeline/pynv_stream.py` | 实时路径遇 `truehdr` 记一条 warning 并按 SDR 处理（实时仍是 8bit NV12 交付） |
| `config.py` | `PT_RTX_VSR_TRUEHDR_CONTRAST/SATURATION/MIDDLE_GRAY/MAX_NITS` |

### UI

| 文件 | 改动 |
|---|---|
| `ui/pages/superres_page.py` | HDR 观感下拉新增 TrueHDR；新增「TrueHDR 参数」一行（对比度 0–200、饱和度 0–200、中灰 10–100、峰值亮度 400–2000），单文件与批量两个页签共享同一份值；该行仅在选中 TrueHDR 时出现，其余观感下整行隐藏并塌陷；选中后自动附加 CLI 参数 |
| `ui/settings.py` | 四个新设置项写入 `config.ini` |
| `ui/translations/*.json` | 三语新增 5 个键（`superres.hdr_look_truehdr` 与 4 个参数名） |
| `README.md` / `README.zh-CN.md` / `README.ja-JP.md`、`PROJECT.md`、`CHANGELOG.md` | 同步文档；PROJECT.md 原先写着「TrueHDR/HDR10 不是当前产品功能」，已改写 |

实时超分对话框（`ui/dialogs/feature_dialogs.py`）**没有**加 TrueHDR 选项 —— 实时链路交付 8bit NV12，接 HDR10 需要另一套 P010 传输与客户端兼容性验证。

---

## 3. 关键技术点（都是实测确认的，不是推断）

### TrueHDR 的输出格式

SDK 注释只说「10 bit ABGR10」。实测（`probe_truehdr`，纯色块输入）：

- **位序**：`bits[0:10]=R, [10:20]=G, [20:30]=B, [30:32]=A`（纯红只抬高低 10 位）。
- **范围**：纯黑输出 0 → **full range**（limited range 的黑应是 64）。
- **色域**：sRGB 纯红输出 `R=562 G=331 B=193`，绿蓝分量明显非零 → **BT.2020 primaries**（若是 BT.709 则应为纯红）。
- **传输函数**：默认 1000 nits 设置下，纯白落在 768/1023 ≈ PQ 0.751 ≈ 1000 nits → **SMPTE 2084 (PQ)**。改成 400 nits 时白点降到 665，改成 2000 nits 升到 769，参数确实生效。

因此转换 kernel 的输入假设是 full-range PQ BT.2020 RGB，输出 limited-range BT.2020 非恒定亮度 P010，容器标注 `bt2020 / smpte2084 / bt2020nc / tv`。kernel 与 NumPy 参考实现对拍**最大误差 0**。

### NVENC 侧的两个坑

- `Pixel_Format` 枚举里叫 `P016`，但 `CreateEncoder` 的格式字符串必须是 **`"P010"`**（用 `P016` 直接报 Unknown format）。
- PyNv 拒绝 CuPy 的 `__cuda_array_interface__`（`Invalid typestr: <u2`）。必须按 SDK 样例的形状喂：Y 平面 `(H, W, 1)`、UV 平面 `(H/2, W/2, 2)`，strides 都是 `(W*2, 2, 1)`，typestr `|u2`。

### 模式切换

TrueHDR 是 create 时决定的 NGX 特性，同一进程内在 SDR/HDR 之间切换必须 `shutdown` 后重建。桥的 context key 已纳入该标志，自动处理；`evaluate` 在模式不匹配时直接报错而不是产出错误格式的数据。

---

## 4. 实测结果（RTX 5060 Ti）

| 场景 | 输入 → 输出 | 质量 | 处理帧率 |
|---|---|---|---|
| 2D | 1280×720 → 3840×2160 | High(3) | **93.7 FPS** |
| SBS VR 分眼 | 3840×1920 → 8192×4096 | High(3) | **15.6 FPS** |

8K VR 的对照是纯 VSR 的 23–24 FPS，即 TrueHDR 这第二次 NGX 评估带来约 35% 的额外开销。离线可接受，也再次印证了不把它接进实时链路的判断。

输出校验（ffprobe）：`hevc / Main 10 / yuv420p10le / bt2020 / smpte2084 / bt2020nc / tv`，帧数与源一致，左右眼拼接无错位。

---

## 5. 测试

- 新增 `tests/test_true_hdr.py`（8 项）：模式识别、HDR10 元数据、grid 计算、参数钳制、桥的创建/切换/缺 DLL/模式不匹配、参数透传。
- 新增 `tests/test_ui_smoke.py::test_superres_page_exposes_true_hdr_controls_only_for_truehdr`：控件仅在 TrueHDR 下显示、两个页签共享值、CLI 参数正确附加。
- 全量：`642 passed, 2 skipped`。

---

## 6. 已知边界

- **仅离线**。实时超分选 TrueHDR 会记 warning 并回到 SDR 观感。
- **FFmpeg 回退路径不支持**，明确拒绝（rc=3）而非静默降级。
- 输出未写 mastering display / MaxCLL SEI，只有容器与 VUI 层面的 HDR10 标注。大多数播放器够用；若某些头显要求码流内 SEI，需要再加 `hevc_metadata` bitstream filter（受 ffmpeg 版本影响，暂未引入）。
- 源仍限 8bit：10bit 源在门控处就被拒，与本次改动无关。
