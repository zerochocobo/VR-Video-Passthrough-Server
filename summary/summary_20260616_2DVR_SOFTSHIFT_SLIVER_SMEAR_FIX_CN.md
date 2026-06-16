# 2D→3D soft_shift 前景涂抹空洞（深度软边碎片）根治（2026-06-16）

接前两份 summary（补洞方向、去预平滑）。用户在**实时模式**实测真实人物帧
（`videos/test_4k2d.mp4` 的 `00:48:00`）仍不对：左眼人物左侧空洞被皮肤涂抹拉长。

## 用真实帧 + 真实 DA3 深度在 uv 环境定位

把该帧走真实 DA3 深度 + 实时 GPU 路径（`render_into_gpu`）逐像素查看：

- 人物左侧遮挡空洞带宽达 ~33px（65mm 视距 → ±33px 视差），**带内被皮肤色碎片填满**。
- 原因：DA3 深度在人体轮廓是**软边**（near 在数像素内 0→1 渐变），前向映射把这些中间
  视差像素散射成遍布空洞带的前景碎片；1D 方向补洞就近抓到碎片 → 皮肤涂抹。
- 这是 soft_shift 前向映射的**固有问题**，CPU、GPU、甚至原版 `tool_2dvr` 都一样。
  前两次的「补洞方向」「去预平滑」修复对硬边有效，但 DA3 深度本身软，无法靠它们根治。

A/B 验证（同帧左眼，人物左缘裁剪）：从左填 / 从右填都涂抹皮肤——因为碎片在两侧都有。
inverse_warp（无空洞）则完全干净。用户选择继续用 soft_shift，故根治其补洞。

## 根治方案：near 边缘 toggle 锐化（就地，不增宽）

对 near 图做 **toggle（形态学对比）锐化**：每个像素吸附到其局部水平 min/max 中更近的
一个，把软的深度轮廓**就地**变成硬边（不整体外扩、不增胖前景）。硬边让前向映射不再散射
碎片，空洞带塌缩成一个由实前景/实背景夹住的干净空洞，从真背景填充。

关键：必须在**全分辨率**（kernel 上采样之后）锐化——在低分辨率 near 上锐化会被双线性
上采样重新糊掉。

- GPU（实时 + 离线 pynv + `GpuStereoRenderer`）：在 `offline/two_dvr_gpu.py` 的
  `_near_at` 设备函数内做 toggle——双线性采样后，取 ±约6 全分辨率像素窗口的低分辨率
  min/max，吸附。无需额外 buffer / 上传 / CPU 开销；仅 soft_shift 的 `fw_zbuf`/
  `fw_color` 走 `_near_at`，inverse_warp / VR 投影核不受影响（它们无空洞）。
- CPU（离线 ffmpeg 回退路径）：`offline/two_dvr_render.py` 新增 `_sharpen_near_edges`
  （cv2 erode/dilate 1×k + where），在 `_make_soft_shift_pair` 里对全分辨率 near 应用。

GPU 与 CPU 是各自独立路径，不会重复锐化。

## 验证（uv 环境，真实 00:48:00 人物帧）

- 实时 GPU `render_into_gpu`：左右眼人物轮廓贴墙锐利，**皮肤涂抹消失**、两眼对称。
- CPU `StereoRenderer` soft_shift：同样干净，与 GPU 一致。
- `py_compile` 全过；`tests/test_offline_outputs.py` 6 passed。

仍建议在真实 app 实时模式复测（重启 server 加载新代码）。

## 备注 / 代价

- `_near_at` 每次多 ~5 次低分辨率 near 读取（窗口 ±2 低分辨率像素），GPU 开销很小。
- toggle 在极细前景结构（发丝、手指）处可能轻微移动边缘位置（≤几像素），实测人物上不可见。
- 这条与前两条修复叠加：去预平滑（不模糊深度）+ 边缘 toggle（锐化深度）+ 按眼别补洞方向。
- inverse_warp 仍是最干净/最快的选项，可作为对画质不满意时的备选。
