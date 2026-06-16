# 2D→3D soft_shift 残留皮肤涂抹根治（前景 near 膨胀）（2026-06-16）

接「toggle 锐化」「硬剪影羽化」。用户用**当前代码离线生成** test_4k2d 30–60s 实测,
34–37s 人物左侧空洞仍有竖向皮肤涂抹带。我自己离线生成同片段、从编码输出抽帧确认:
确实还在。

## 根因(比深度软边更深一层)

深度边界**永远不会和颜色剪影精确对齐**。人体边缘有一圈像素:颜色是皮肤,但 DA3 给的
near 偏低/中间。前向映射时它们**不随人体平移**(因为 near 低),原地留下,变成空洞**背景侧
紧贴的皮肤色碎片**。方向补洞(左眼从左取最近有效像素)就抓到这个皮肤碎片 → 整条空洞被
横向涂成皮肤色。

关键:这些碎片 near 偏低,所以「按 near 排除前景」抓不到它们(toggle 锐化、排除前景羽化
都无效)——它们在 near 上看起来就是背景,只是颜色是皮肤。

## 修法:前景 near 膨胀(`_dilate_near_fg`)

对 near 图做**前景膨胀(max 滤波)**:让边界那一圈像素的 near 被周围前景"撑成"前景值,
于是它们**随人体一起平移**、不再原地留碎片。膨胀后落在"中间 near"的换成了更外侧的**墙体
颜色**像素——即使仍有残留碎片也是墙色,填进空洞与背景**无缝、不可见**。

- 半径按 near 图宽度自适应:`r = round(width/512)`(518 低分辨率→1;1920→~4;4K→~8),
  与 warp 分辨率匹配。
- toggle/snap 仍保留,把膨胀后的边再锐化硬。

应用位置(soft_shift 专用,inverse_warp 不动):

- `offline/two_dvr_render.py:_dilate_near_fg`(新);`near_from_depth` soft 分支先膨胀
  (覆盖**实时** `pynv_stream` + **离线 pynv** `two_dvr_pynv`,二者都走 near_from_depth)。
- `_sharpen_near_edges`(CPU offline)改为先膨胀再 toggle。
- `offline/two_dvr_gpu.py:GpuStereoRenderer.render()` soft 分支同样先膨胀。
- GPU 核 `_near_at` 的 toggle 不变(它对已膨胀的 near 做硬化);GPU 路径 = near_from_depth
  膨胀 → 核 toggle,无重复。

## 验证(uv 环境,真实离线编码输出)

- 用当前代码 `python -m offline.two_dvr single ... --start 30 --duration 30
  --model base --hole-fill soft_shift` 生成 1798 帧 @ ~83fps,depth=TensorRT,
  pipeline=pynv-gpu。
- 从**编码后的 mp4** 抽 34/35/36/37s(= 输出 4/5/6/7s)左眼裁剪:人体手臂/腿剪影贴墙
  **干净锐利,皮肤涂抹带消失**,跨多帧一致。
- `tests/test_offline_outputs.py` 6 passed;`py_compile` 全过。
- 生成的对比视频已放到 `videos/test_4k2d_S000030_E000100_3D_LR_Screen.mp4`。

## soft_shift 完整链(五步)

1. `near_from_depth`:soft_shift 用未平滑深度。
2. `_dilate_near_fg`:前景 near 膨胀(把边界混合区推到墙色,消除皮肤碎片)。**本次新增**。
3. toggle / `_near_at`:把(膨胀后的)深度边锐化成硬边。
4. 前向映射 + z-buffer(硬剪影),按眼别从背景侧填洞。
5. 羽化:只软化填充洞、模糊核排除前景 → 硬剪影 + 软背景接缝。

inverse_warp 仍是无空洞的备选。
