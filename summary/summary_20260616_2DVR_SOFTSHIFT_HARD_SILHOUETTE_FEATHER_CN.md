# 2D→3D soft_shift 羽化改为「硬剪影 + 仅软化背景接缝」（2026-06-16）

接 sliver-smear 修复。与用户对齐算法后确认需求：

1. 空洞只能用**背景**填，绝不能用移动的人体前景。
2. 人体/背景之间是**硬剪影，不羽化**。
3. 只羽化「横向复制的背景补丁」与「真背景」之间的接缝（真实视频背景复杂，需软化拉丝）。

极致推演（纯色背景）：硬分割前景/背景 → 前景平移 → 空洞=纯背景色 → 任何地方都不需要
羽化。可见羽化不是填洞的必需步骤，只为掩盖纹理背景被横向拉伸的瑕疵。

## 之前错在哪

旧 `fw_blend` / `_soft_blend_holes_rgb` 对**1px 膨胀后的整个空洞边界**做对称模糊——这把
人体剪影侧也一起糊了，导致剪影发虚。

## 改法：只羽化空洞像素，模糊核排除前景

核心：模糊只发生在已填的空洞像素上，且模糊核**排除前景（遮挡者）**邻居。于是：

- 前景剪影侧：核里不含前景像素 → 剪影**保持绝对硬**；前景像素本身直接透传不参与混合。
- 背景侧：只在背景/填充背景之间做轻模糊 → 软化横向拉伸接缝。

前景判定用 z-buffer 里编码的 near（priority = near*1e6+1）：当模糊窗口跨越一个深度台阶
（窗口内 near 极差 > 0.30）才以 `0.5*(near_min+near_max)` 为阈值丢掉近的一半（前景）；
否则不门控（纯背景区只是轻微平滑）。混合比例仍 35%。

实现：

- GPU `offline/two_dvr_gpu.py:fw_blend`：只处理 `zbuf==0` 的填充洞；K=3、V=2 窗口，按
  上述阈值排除前景；其余像素透传（保持锐利）。仅 soft_shift 路径走它。
- CPU `offline/two_dvr_render.py`：
  - `_forward_warp_eye_rgb` 现在多返回 warped `near_buffer`（输出坐标系，z-buffer near）。
  - `_soft_blend_holes_rgb(image, holes, near_buffer)` 改为**带掩码的盒模糊**：
    `blur = box(image*bg) / box(bg)`，`bg = 未写入(洞) | near<=thr`，前景被排除；
    只在 `holes` 上以 0.35 混合。
  - `_make_soft_shift_pair` 传 warped near + 调用更新。
- `_dilate_mask_np`/`_box_blur_rgb` 不再被使用（保留未删，无副作用）。

## 验证（uv 环境，真实 00:48:00 人物帧）

- 实时 GPU `render_into_gpu`：人体轮廓贴墙**锐利**，背景填充区被轻微软化、无硬拉丝条带；
  剪影不再发虚。
- CPU `StereoRenderer` soft_shift 烟雾测试通过。
- `py_compile` 全过；`tests/test_offline_outputs.py` 6 passed。

需在真实 app 实时模式复测（重启 server 加载新代码）。

## soft_shift 完整算法现状（四步）

1. `near_from_depth`：soft_shift 用**未平滑**深度。
2. `_sharpen_near_edges` / 核内 `_near_at` toggle：把深度软边**锐化成硬边**，消除前景碎片散射。
3. 前向映射 + z-buffer（硬剪影），按眼别从**背景侧**方向填洞。
4. 羽化：只软化填充洞，模糊核**排除前景** → 硬剪影 + 软背景接缝。
