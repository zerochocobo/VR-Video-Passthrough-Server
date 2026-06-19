# 2DVR Soft-Shift 白色竖线问题总结

日期：2026-06-17

## 背景

2D->3D 的 `soft_shift` 路径不是简单平移图像，而是：

1. DA3 预测深度；
2. 深度归一化为 near map；
3. 根据 near 做 forward warp，把源像素写入左右眼；
4. 用 `zbuf` 记录每个输出像素由哪个 near 优先级写入；
5. 对 `zbuf==0` 的空洞做 hybrid 填充：用 `inverse_warp` 的同位置像素填；
6. 对空洞接缝做轻微 blend。

之前的“头发横向黑/暗色 smear”主要是 `zbuf==0` 空洞的 row-copy 填充造成，已经通过 hybrid hole fill 解决。当前这次白色竖线是另一个问题。

## 现象

用户重新生成：

`videos/test_4k2d_S000030_E000130_3D_LR_Screen.mp4`

在输出 3-11 秒之间，头发附近仍有明显白色竖线：

- 左眼：白线出现在右侧头发；
- 右眼：白线出现在左侧头发；
- 源视频对应 33-41 秒。

这个左右眼位置相反的现象很重要：它说明白线不是源图自带，也不是简单抠图残留。它跟左右眼 forward warp 方向有关，是 2D->3D 生成过程里的伪影。

## 调试材料

主要调试输出：

- `debug_output/2dvr_hair_white_lines_20260617/hair_source_left_right_t03_t11.png`
- `debug_output/2dvr_hair_white_lines_20260617/t07_zbuf_analysis/left_eye_right_hair_pipeline.png`
- `debug_output/2dvr_hair_white_lines_20260617/t07_zbuf_analysis/right_eye_left_hair_pipeline.png`
- `debug_output/2dvr_hair_white_lines_20260617/fg_bad_current_sim/Lhair_win8_compare.png`
- `debug_output/2dvr_hair_white_lines_20260617/fg_bad_current_sim/Rhair_win8_compare.png`

关键复现帧：

- 输出 t=7s；
- 源视频 t=37s；
- 用已有 `frame_out_t07_src_t37_repro/buffers.npz` 分析 `raw`、`zbuf`、`inv`、`blend`。

## 根因分类

白色竖线不是单一原因，已经遇到过三类：

### 1. 背景侧 disocclusion rim / hole seam

这类问题出现在真实 `zbuf==0` disocclusion hole 与真实背景的交界处，常见于左眼头发左侧、
右眼头发右侧这类“人物被视差推开后露出的背景洞”。它不是人体内部白竖线，而是 hole
边界外侧仍有一圈 `zbuf!=0` 的低 near 背景侧 seam 像素；纯 hybrid 只替换 `zbuf==0`，
不会动这圈非 hole 像素。

这类问题应由 `PT_TWO_DVR_RIM` / `fw_hole_from_inv` 的 rim 分支处理：

- 只处理每只眼的背景侧方向；
- 只处理 low-near 像素；
- 前景/high-near 剪影保留。

2026-06-17 晚间 BASE518 复查确认：左眼头发左侧 hole/background 接缝在
`PT_TWO_DVR_RIM=0/8/16` 下随数值增大明显变干净；`PT_TWO_DVR_FG_BAD=0/8`
对这条接缝影响很小。因此这类 seam 不应再归因到 `fg_bad`，也不应简单归因成
“RIM 产生的白竖线”。

### 2. 人体内部 foreground crack

soft_shift forward warp 后，人体内部可能留下细小 `zbuf==0` 裂缝。hybrid hole fill 会把这些 hole 当成真实 disocclusion hole，用 `inverse_warp` 填；如果 inverse 同位置采到墙面或背景，就会变成白色短竖线。

这类问题需要把“人体内部裂缝”和“真实背景空洞”区分开。

### 3. 低 near 背景像素写进头发/人体区域

这次头发白竖线主要属于这一类。

在 t=7/source=37 的头发区域，白线像素有这些特征：

- 多数不是 `zbuf==0`；
- 它们是 `zbuf!=0` 的 written pixel；
- near 很低，常见约 `0.014-0.031`；
- RGB 接近墙面亮色；
- 在 `raw_forward` 阶段已经存在；
- `inverse_warp` 和 `blend` 基本只是保留它，不是生成它。

也就是说：背景/墙面低 near 像素被 forward warp 写进了头发边缘附近，成为白色竖线。因为它们不是 hole，所以只修 `zbuf==0` 空洞不够。

## 为什么左右眼位置相反

左右眼的 forward warp 方向相反：

- 左眼前景向一侧偏移；
- 右眼前景向另一侧偏移。

低 near 背景污染会出现在各自 shift 后的相反轮廓边，因此用户看到：

- 左眼在右侧头发；
- 右眼在左侧头发。

这正好证明它不是源图或分割单边问题，而是 stereo warp 后的方向性伪影。

## 当前修复算法

新增 GPU kernel：

`offline/two_dvr_gpu.py::fw_fg_bad_local`

它折进已有 `_SOFT_SHIFT_KERNELS` RawModule，不新建独立 CuPy module，避免额外 cold compile 风险。

核心逻辑：

1. 对 SBS 输出中的每个像素检查 `zbuf`；
2. 如果它是 background-ish：
   - `zbuf==0`，或
   - `zbuf!=0` 但 near < `0.5`
3. 在同一眼内，向左右各搜索 `win` 像素；
4. 如果左右两侧都能找到 high-near foreground（near >= `0.5`），说明它是嵌在前景内或发丝边缘内部的污染；
5. 用最近一侧的前景颜色替换它；
6. 如果只有一侧有前景，认为它是真实轮廓外背景，不动。

运行位置：

`fw_zbuf -> fw_color -> fw_hole_from_inv / fw_fill -> fw_blend -> fw_fg_bad_local -> project_flat_lr`

它跑在 `fw_blend` 之后，所以修掉的点不会再被后续 blend 重新变白。

## 参数：PT_TWO_DVR_FG_BAD

`PT_TWO_DVR_FG_BAD` 是 `fw_fg_bad_local` 的左右搜索窗口，单位是源视频全分辨率像素。

当前默认：

- 源宽 1920：默认 `8`；
- 源宽 3840：默认 `16`；
- 源宽 720：默认 `3`；
- 可用 `PT_TWO_DVR_FG_BAD=0` 显式关闭。

为什么按源宽缩放：

- 这个窗口是在全分辨率图上按像素搜索；
- 同样视觉宽度的边缘污染，在更高分辨率源图中会占更多像素；
- 因此用 `round(src_w / 240)` 做近似比例缩放。

注意：

- 这是经验值，不是严格物理公式；
- 头发白线更直接受 depth edge、shift 强度、发丝细节影响；
- 如果默认值误伤复杂发丝或细节，可以把 `PT_TWO_DVR_FG_BAD=0` 关闭，或设成更小值如 `4`；
- 如果白线仍残留，可尝试 `12`、`16`。

## 参数：PT_TWO_DVR_RIM

`PT_TWO_DVR_RIM` 是背景侧 disocclusion seam 清理窗口，单位也是源视频全分辨率像素。

当前默认：

- 源宽 1920：默认 `16`；
- 源宽 3840：默认 `32`；
- 源宽 720：默认 `6`；
- 可用 `PT_TWO_DVR_RIM=0` 显式关闭。

它和 `PT_TWO_DVR_FG_BAD` 处理的是两类问题：

- `RIM`：真实空洞外侧、背景侧接缝，通常贴着大块 `zbuf==0` hole 边界；
- `FG_BAD`：人体/头发内部的低 near 背景线或 foreground crack，要求左右两侧都被前景包住。

## 如何判断以后看到的白竖线是哪一类

建议按下面顺序判断。

### A. 看是否启用了新清理

如果没有设置环境变量，并且代码已是当前版本：

- 1920 源宽默认应自动启用 `8`；
- 1920 源宽 `PT_TWO_DVR_RIM` 默认应自动启用 `16`；
- 如要确认关闭/开启影响，可分别跑：

```powershell
$env:PT_TWO_DVR_RIM="0"
$env:PT_TWO_DVR_FG_BAD="0"
# regenerate

$env:PT_TWO_DVR_RIM="16"
$env:PT_TWO_DVR_FG_BAD="8"
# regenerate
```

如果 `RIM=16` 明显改善，说明更像真实空洞背景侧 seam；如果 `FG_BAD=8`
明显改善，说明属于 foreground-embedded background 污染。

### B. 看白线是否左右眼位置相反

如果左眼白线在一侧头发、右眼在相反侧头发，通常是 forward warp 方向性污染，不是源视频自带。

### C. 看 raw_forward / zbuf

如果有 debug buffers：

- 白线在 `raw_forward` 已经存在：forward warp 写入阶段产生；
- 白线只在 `blend` 后出现：blend 或 hole fill 后处理问题；
- 白线像素 `zbuf==0`：foreground crack / hole 填充问题；
- 白线像素 `zbuf!=0` 且 near 很低：低 near 背景 written pixel；
- 白线像素 `zbuf!=0` 且 near 很高：可能是高 near winner/edge tie 问题，当前 `fw_fg_bad_local` 不会动它。

### D. 看颜色来源

- 如果白线颜色接近墙面/背景，多半是低 near 背景污染；
- 如果白线颜色接近肤色/衣服，可能是前景边界残留或 near dilation/sharpen 相关；
- 如果是大面积横向条带，则更像旧 row-copy 空洞填充问题。

## 已验证现象

在 t=7/source=37 的头发区域做 CPU 等价模拟：

- `win=8` 可以明显移除左眼右侧头发、右眼左侧头发的白色竖线；
- mask 主要覆盖头发边缘和人体内部低 near 细线；
- 这说明用户重新生成后仍看到白线，主要是因为当时该清理还是默认关闭。

## 代码改动

主要文件：

- `offline/two_dvr_gpu.py`
  - 新增 `_two_dvr_fg_bad_width(src_w)`；
  - 新增 `fw_fg_bad_local`；
  - 在 soft_shift path 中加载 `_k_fg_bad`；
  - 在 `fw_blend` 后调用；
  - 默认窗口改为 `max(2, round(src_w / 240))`；
  - `PT_TWO_DVR_FG_BAD=0` 可关闭。

- `tests/test_two_dvr_hybrid_hole_fill.py`
  - 增加/更新 `PT_TWO_DVR_FG_BAD` 默认值和 override 测试。

- `utils/gpu_runtime_cache.py`
  - `CUPY_COMPILE_WITH_PTX` 改为硬置 `0`，避免 stale shell env 让 CuPy 走 PTX driver-JIT 慢路径。

- `PROJECT.md`
  - 记录 sm_120 / NVRTC / CuPy RawModule 编译卡住排查方式；
  - 强调要看 `cupy.cuda.compiler._use_ptx`，不能只看 `os.environ`。

## 验证

已跑：

```powershell
uv run python -m pytest tests/test_two_dvr_hybrid_hole_fill.py -q
```

结果：

```text
7 passed
```

额外已跑：

```powershell
uv run python -m pytest tests/test_da3_download_and_trt.py tests/test_two_dvr_ui_hole_fill.py tests/test_ui_smoke.py -q
```

结果：

```text
11 passed
```

## 后续建议

1. 用当前默认值重新生成 `test_4k2d_S000030_E000130_3D_LR_Screen.mp4`，重点检查输出 3-11 秒头发边缘。
2. 如果白线仍残留：
   - 先试 `PT_TWO_DVR_FG_BAD=12`；
   - 再试 `PT_TWO_DVR_FG_BAD=16`；
   - 对比是否误伤发丝或扩大头发轮廓。
3. 如果出现误伤：
   - 试 `PT_TWO_DVR_FG_BAD=4`；
   - 或 `PT_TWO_DVR_FG_BAD=0` 关闭；
   - 重新查看 `raw_forward` / `zbuf`，确认是否属于高 near written 或其它类别。
4. 如果以后又看到类似白线，优先判断它是：
   - hole；
   - 低 near written；
   - 高 near written；
   - blend 后产生；
   - source 自带。

这五类的修法不同，不应再统一归因到“空洞填充”。
