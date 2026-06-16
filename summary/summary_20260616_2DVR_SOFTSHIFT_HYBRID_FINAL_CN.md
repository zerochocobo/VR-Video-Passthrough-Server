# 2D→3D soft_shift 头发横纹根治：hybrid 补洞 + 背景侧残留清理（最终方案，2026-06-16）

用户确认"效果不错"。本文记录最终落地的方案与全链路。提交：
`b277d29`(hybrid)、`02b6972`(rim 清理),分支 `feature/2dvr`。

## 问题回顾

soft_shift(前向映射 DIBR)在头发等处,左眼人物左侧出现**暗色横向条纹**,左右眼不一致
→ 闪烁。多轮单点修复(补洞方向 / 去预平滑 / toggle 锐化 / 硬剪影羽化 / 前景膨胀)只能救
实体边缘,救不了飘逸头发。

**真正根因(开发人员原型诊断,已证实)**:不是"抠图脏",而是 soft_shift 的**按行补洞
(row-copy)**——把每行那**一个干净背景像素**横向复制满整块空洞,墙/砖每行亮度不同 → 形成
横向条带。对头发还会复制到留下的发丝。这与用户"空洞左侧其实是干净背景"的观察一致:空洞
干净,是**填充**把它抹成了条带。

## 最终方案

### 1. hybrid 混合补洞(`b277d29`)

soft_shift 负责所有 z-buffer 写入像素(正确遮挡 + 硬剪影);只把 `zbuf==0` 的**空洞像素**
换成 **inverse_warp** 子渲染的结果。inverse_warp 是反向采样、**无空洞**,所以空洞处没有
按行条带;又因为**只在空洞用**,inverse 单用时的"砍人像/边缘撕裂"不会出现(人体仍是
soft_shift,逐字节保留)。

GPU 实现(关键:**折进现有 kernel,不新建独立 RawKernel** → 规避开发人员遇到的 sm_120
新 kernel 编译卡死):
- 在已有 `_SOFT_SHIFT_KERNELS` 模块里加一个极小的 `fw_hole_from_inv`;
- 在 soft 分支实例化已有的 `_SBS_INV_WARP_KERNEL` + `_inv_g` 缓冲;
- `_launch` soft+flat3d:fw_zbuf → fw_color → inverse 子渲染 → fw_hole_from_inv →
  fw_blend(排除前景的接缝羽化)。VR/project 保持旧 row-copy。

### 2. 背景侧残留(matting 抠图残留)清理(`02b6972`)

人物边缘有一圈像素**颜色是头发/衣服、但深度被判低 → 不随前景平移、留在原位**,形成空洞
**背景侧边缘**的污染边(它们有有效颜色,不是空洞,hybrid 不动)。`fw_hole_from_inv` 扩展为
也把这些边缘像素换成 inverse,带两个安全约束(用户要求):
- **方向性**:只处理该眼的**背景侧**(左眼=右侧紧邻空洞的像素,右眼=左侧),**绝不碰**
  空洞另一侧的真实人物剪影;
- **深度门控**:只清理 `near < 0.5`(背景/残留);`near ≥ 0.5`(前景)保留 → 既护剪影,
  又护**窄空洞间的细前景条**不被误吃(对应"空洞很小别误处理")。

范围 `rim` 随宽度自适应(1080p→8px,4K→16px),env `PT_TWO_DVR_RIM` 可调、0 关。

## soft_shift 完整链(现状)

1. `near_from_depth`:soft_shift 用未平滑深度。
2. `_dilate_near_fg`:前景 near 膨胀(把边界混合区推到背景色)。
3. `_near_at` toggle:把深度软边锐化成硬边。
4. 前向映射 + z-buffer(硬剪影)。
5. **hybrid**:空洞像素 ← inverse_warp(去掉按行条带)。
6. **rim 清理**:背景侧 matting 残留边 ← inverse(方向 + 深度门控)。
7. `fw_blend`:只软化接缝、排除前景。

> 注:步骤 2/3 在 hybrid 落地后**或可简化**(空洞已由 inverse 干净填充,前景膨胀的鼓边
> 副作用可减小),列为后续可选优化。

## 性能与验证(uv 环境,RTX 5060 Ti)

- 每帧渲染(含 hybrid 合并)≈ **0.59ms**;整段 30–60s 离线 **~83fps**,depth=TensorRT、
  pipeline=pynv-gpu。
- 编码输出头发帧 t=5/7/10 左眼:**横纹消失、发丝边自然、人像完整**,跨帧一致。
- rim 清理量化:rim0→16 每帧改 ~0.7–3% 像素,其中几百个大改动(真清掉残留 sliver),其余
  为背景边缘换成几乎相同的 inverse(无害,降低该边的逐帧闪烁)。深度门控确认未侵蚀前景。
- 测试:`test_two_dvr_hybrid_hole_fill`(5)+ `test_offline_outputs`(6)全过。
- 调试开关 `PT_TWO_DVR_DEBUG_HOLES=1`(洋红空洞)、`PT_TWO_DVR_RIM`(rim 宽度)。

## 实现备注 / 后续

- 开发人员卡的"新 CuPy kernel 编译卡死"在本机未复现(合并进现有模块 + 复用现有 inverse
  kernel)。若他机复现,把这两个 kernel 纳入启动预热/缓存。
- inverse 子渲染目前复用膨胀后的 near,质量已够;需要时可单独传平滑 near。
- 待真实播放器确认运动中双眼无闪烁。
- 工作区另有别的 session 未提交的 UI 改动(隐藏 hole-fill 选项),与本方案无关。

相关:`summary_20260616_2DVR_SOFTSHIFT_HAIR_HOLE_FILL_REVIEW_CN.md`(原型)、
`summary_20260616_2DVR_SOFTSHIFT_HAIR_SMEAR_RESEARCH_BRIEF.md`(排查史)。
