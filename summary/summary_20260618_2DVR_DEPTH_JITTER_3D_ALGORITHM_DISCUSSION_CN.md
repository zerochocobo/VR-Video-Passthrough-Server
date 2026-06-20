# 2DVR 深度/视差抖动消除算法讨论稿

日期：2026-06-18

## 1. 问题背景

当前 2D 转 3D/VR 的核心链路是：

```text
RGB 视频帧 -> DA3 单帧深度 -> near/disparity -> 左右眼平移 -> SBS/VR 输出
```

用户实际观察到的问题不是单纯的边缘瑕疵，而是更影响舒适度的 **左右眼视差在轻微持续变化**：

```text
同一个对象在画面中的 2D 宽度/轮廓几乎不变，
但左右眼间距不断轻微变化，
视觉上像对象在前后抖动，导致眩晕。
```

这说明 DA3 的逐帧深度存在时间漂移。即使归一化分位带已经做 EMA，仍然会有对象级或区域级的视差闪烁。

## 2. 核心判断

真实 3D 运动应该有 2D 图像证据支撑。

如果对象靠近摄像机，通常会出现：

- 像素宽度增大；
- 局部纹理尺度变大；
- 轮廓外扩；
- 透视关系变化。

如果对象远离摄像机，通常会出现：

- 像素宽度减小；
- 局部纹理尺度变小；
- 轮廓内收。

因此可以建立一个约束：

```text
如果图像中没有足够的尺度变化证据，
就不应允许 near/disparity 出现明显变化。
```

当前 DA3 给出的深度变化是“模型估计变化”，不是“几何证据变化”。消抖算法应当优先相信图像证据。

## 3. 不推荐继续默认使用 Dense Flow

已经验证 OpenCV Farneback dense flow 对稳定有帮助，但代价过高：

- 离线 FPS 大幅下降；
- 实时播放会周期性卡顿；
- 当前 `opencv-python` 环境没有可用 CUDA optical flow；
- 自编译 CUDA OpenCV 或接 NVIDIA Optical Flow SDK 都会显著增加打包和兼容复杂度。

结论：

```text
Dense flow 可以保留为离线实验选项，
不适合作为默认或实时路径。
```

## 4. 建议基础算法：图像证据门控的视差稳定

建议算法名称：

```text
Evidence-Gated Disparity Stabilization
图像证据门控视差稳定
```

目标：

```text
只在 RGB 图像存在足够变化证据时，才允许 near/disparity 明显变化；
静态或近似静态区域强力锁定视差，消除对象前后抖动。
```

输入：

```text
RGB_t            当前 RGB 帧
near_raw_t       当前 DA3 归一化 near map
RGB_{t-1}        上一 RGB 帧
near_stable_{t-1} 上一稳定 near map
```

输出：

```text
near_stable_t
```

## 5. 基础算法 V0

### 5.1 near map 分解为 base/detail

把当前 near map 分成低频主体和高频细节：

```text
base_t   = blur(near_raw_t, large_radius)
detail_t = near_raw_t - base_t
```

含义：

- `base_t` 控制对象整体前后距离，也就是最容易造成眩晕的视差主体；
- `detail_t` 保留边缘、局部结构和深度层次。

稳定重点放在 `base_t`，而不是全图无差别平滑。

### 5.2 估计图像稳定证据

先做一个低成本稳定 mask：

```text
luma_diff = abs(gray_t - gray_{t-1})
stable_mask = luma_diff < threshold
```

这一步不需要 dense flow，成本低。

含义：

- 像素亮度变化小，说明该区域很可能是同一表面或近似静态背景；
- 亮度变化大，可能是运动、遮挡、切换、强光变化，应减少历史约束。

### 5.3 全局或分块 near affine 对齐

在 `stable_mask` 内，估计当前 `base_t` 到上一帧 `base_stable_{t-1}` 的全局 scale/bias：

```text
base_aligned = scale * base_t + bias
```

约束：

```text
scale 限制在 [0.8, 1.2]
bias  限制在 [-0.12, 0.12]
```

作用：

```text
消除 DA3 逐帧整体深度尺度漂移，
专门压制“整个人/整个物体左右眼距轻微变化”。
```

后续可以从全局 affine 升级到 tile affine：

```text
画面划分为 16x9 或 32x18 tiles，
每个 tile 独立估计 scale/bias，
再做空间平滑。
```

### 5.4 死区和最大变化速度限制

对低频 base 做时间限制：

```text
delta = base_aligned - base_stable_prev

if abs(delta) < deadband:
    base_stable = base_stable_prev
else:
    base_stable = base_stable_prev + clamp(delta, -max_step, max_step)
```

建议初始参数：

```text
deadband = 0.004 ~ 0.008
max_step = 0.010 ~ 0.020 / frame
```

含义：

- 小幅深度闪烁直接吞掉；
- 真实深度变化也不能每帧突跳；
- 镜头切换或大幅场景变化时 reset。

### 5.5 根据图像证据调节融合强度

在稳定区域强力使用历史，在变化区域更多使用当前帧：

```text
if stable_mask:
    alpha = 0.10 ~ 0.20
else:
    alpha = 0.50 ~ 1.00

base_stable = (1 - alpha) * base_stable_prev + alpha * base_limited
```

含义：

- 静态物体：视差几乎锁住；
- 运动/遮挡区域：允许当前深度更快跟随；
- 避免直接 EMA 造成明显拖影。

### 5.6 细节回填

最终 near：

```text
near_stable_t = base_stable_t + detail_t * detail_weight
near_stable_t = clip(near_stable_t, 0, 1)
```

建议：

```text
detail_weight = 0.7 ~ 1.0
```

含义：

- 整体距离稳定；
- 当前帧边缘和局部层次仍然保留；
- 不把人物边缘糊成慢动作残影。

## 6. 可选增强：Sparse KLT 尺度证据

如果 V0 仍不够，可以加稀疏特征跟踪，不做 dense flow。

流程：

```text
1. goodFeaturesToTrack 找角点；
2. calcOpticalFlowPyrLK 跟踪到当前帧；
3. 在局部 tile 或对象区域内估计尺度变化；
4. 用尺度变化约束 near/disparity 变化。
```

尺度关系：

```text
图像尺度 scale ≈ 当前特征点间距 / 上一帧特征点间距
视差/near 的允许变化 ≈ scale 的变化
```

例子：

```text
如果局部 scale ≈ 1.00，
但 DA3 near 想变化 8%，
则判定为深度抖动，强力抑制。
```

优点：

- 计算量远低于 dense flow；
- 更符合“对象宽度不变则深度不应变”的几何直觉；
- 可作为 tile 级 gate，而不是逐像素 warp。

## 7. 场景切换和异常处理

必须保留 reset 条件：

```text
1. RGB 全局差异过大；
2. near percentile band 大幅跳变；
3. stable_mask 覆盖率过低；
4. 视频 segment / seek / clip 起点；
5. 解码帧不连续。
```

触发 reset 时：

```text
near_stable_t = near_raw_t
清空历史状态
```

否则会在镜头切换时产生拖尾。

## 8. 推荐开发顺序

> 注意：以下为 **2026-06-18 评审后修订版**（原始顺序见各 Phase 内的“原始版本”说明）。最大改动是把“全局运动补偿”从隐含缺失提到 Phase 1 第一位——没有它，后面 affine / deadband / EMA 在任何运镜片源上都建立在错误的像素对应关系上。详见 8.5。

### Phase 1：全局运动补偿 + warp-then-filter + Base/Detail + Deadband

实现：

```text
phaseCorrelate / KLT 估全局运动 -> warp near_stable_{t-1} 到当前帧
near -> base/detail（base 用 guided filter，避免边界 halo）
base 做 deadband + max_step + EMA（在 warped_prev 基础上）
detail 当前帧回填
```

优点：

- 先堵上“运镜下失效”这个最大缺口，这一步就能解决大部分晕；
- 全局 warp 非 dense flow，几乎不影响 FPS；
- 直接针对整体视差抖动。

> 原始版本：仅 Base/Detail + Deadband，无运动补偿——只在静止机位成立。

### Phase 2：Tile Affine（鲁棒回归）+ KLT 局部尺度门控

实现：

```text
画面分 tile（如 16x9），每 tile 用鲁棒回归（Huber/RANSAC）估 scale/bias；
KLT 局部特征间距变化 -> 局部尺度证据 -> 有据放行真实运动、压制无证据抖动。
```

优点：

- 针对 DA3 全局/区域深度尺度漂移；
- 对“对象宽度不变但视差在变”更有效；
- KLT 同时提供运动补偿与尺度证据（见 8.5.3 A）。

> 原始版本：Phase 2 = luma stable_mask + 全局 affine；Phase 3 = Sparse KLT。评审后将 KLT 提前并入此处，luma mask 改为运动补偿后再用。

### Phase 3：离线专属 — 对称时间窗 + 边缘感知

实现：

```text
离线可看未来帧：对 base（运动补偿后）做 5~7 帧对称时间中值/高斯；
按深度梯度加权：平坦区强锁，边缘区放松，避免深度边界游泳/光晕。
```

优点：

- 对称窗口消抖且无拖影、无延迟（离线独有红利，见 8.5.3 B）；
- 边缘感知避免过平滑产生的边界游泳；
- 比 dense flow 轻很多。

## 8.5 评审与修订（2026-06-18 复盘）

总判断：方向认可，核心原则（**阻止没有图像证据支撑的视差变化**）正确。但 V0 方案有三个隐藏假设会在真实片源上失效，必须先修；另有若干增强能显著提升上限。

### 8.5.1 定位：治标 vs 治本

- VDA / NVDS 等是“治本”——把时序一致性训进模型，产出真正正确的时序深度，但工程量大 + Base 非商用许可证 + 导出风险。
- 本自写算法是“治标”——不试图估对每像素深度，而是阻止无证据的视差变化。它无法凭空造出正确的时序深度，但用户的核心症状（**静止对象左右眼距持续微变 → 眩晕**）正是它最擅长压制的。
- 结论：针对当前痛点，自写算法是性价比最高、且大概率够用的近期方案。

### 8.5.2 三个致命假设（必须修）

**问题 1：`luma_diff` 当 stable_mask，遇到运镜直接崩（最严重）。**

`stable_mask = |gray_t - gray_{t-1}| < threshold` 隐含假设摄像机不动。一旦镜头平移/手持/缓推，整帧亮度都变 → `stable_mask` 覆盖率塌到接近 0 → 稳定化自动关闭；而缓慢平移扫过静态场景恰恰是抖动最难看、最晕的工况。第 7 节还把“stable_mask 覆盖率过低”列为 reset 条件，运镜时会被误判成场景切换反复 reset，雪上加霜。

修法：**先做全局运动补偿，再比对。** 不用 dense flow，用便宜的全局配准：`cv2.phaseCorrelate`（相位相关，亚像素平移，~0.3ms）或 KLT 角点 + `estimateAffinePartial2D`（全局 2~4 DOF 运动）。拿到全局运动后把 `near_stable_{t-1}` warp 到当前帧坐标系，再做 luma 比对和 EMA，运镜下 `luma_diff≈0`，mask 恢复正常。

**问题 2：filter-in-place 假设了像素对应关系。**

5.3~5.5 全部在同一像素位置比较 `base_t` 与 `base_stable_{t-1}`，只有静止机位成立。任何运镜下同一屏幕像素在 t-1 和 t 是不同物理表面 → 拿背景历史深度去锁前景当前深度，要么拖影要么乱锁。

修法：把架构从“原地滤波”改成 **warp-then-filter**：

```text
Stage 0  全局运动补偿：warp near_stable_{t-1} → 当前帧 (warped_prev)
Stage 1  在 warped_prev 上做 base/detail、affine、deadband、EMA
Stage 2  detail 用当前帧回填
```

只加一个全局 warp（非 dense），即可覆盖约 80% 的手持/平移片源——这是 V0 目前最大的缺口。

**问题 3：每帧对齐到上一帧 → 长期漂移累积。**

5.3 的 `base_aligned = scale·base_t + bias` 对齐目标是 `base_stable_{t-1}`，逐帧链式相乘。scale 哪怕每帧偏 0.999，几百帧后整段深度尺度漂没，长镜头里“整个世界慢慢变远/变近”。

修法：① affine 估计用鲁棒回归（Huber / RANSAC），普通最小二乘会被运动物体 outlier 带偏全局 scale；② 对齐参考不要只用前一帧，维护一个缓慢更新的锚（或周期性把累积 scale 软约束回 1.0 附近），抑制漂移。

### 8.5.3 增强项（提升上限）

- **A. 把 Sparse KLT 从 Phase 3 提为核心。** KLT（`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`，~500 角点 1~2ms，远低于 Farneback dense）一次给两样：① 全局运动补偿（修问题 1/2）；② 局部尺度证据（第 6 节“对象宽度不变则深度不应变”的几何约束）。一个组件解决三个问题，应放进 Phase 1/2。
- **B. 离线路径用“对称时间窗”，别用因果 EMA。** EMA/deadband 是因果的，必然在“稳定 vs 拖影”二选一。离线转换管线可看未来帧 → 对 `base`（运动补偿后）做 5~7 帧时间中值/高斯，既消抖又无拖影、无延迟，因为它对称。这是离线独有的免费红利。明确分档：实时=warp+证据门控因果滤波；离线=warp+对称时间窗中值。
- **C. 边缘/置信度感知。** DA3 抖动最重的是深度不连续边缘和低纹理天空。按深度梯度加权：平坦区强锁，边缘区放松（边缘强行平滑会产生“深度边界游泳/光晕”，比抖动更难看）。
- **D. base/detail 分解换 edge-aware。** 大半径高斯 blur 求 base 会在深度边界产生 halo（base 越界泄漏）。用 guided filter（以 RGB 为引导）或 bilateral 代替高斯，base 不糊穿物体边界，detail 回填更干净。

### 8.5.4 对第 9 节开问的回答

1. **稳定粒度**：选 tile（如 16×9）affine + 全局运动补偿。全局太粗，语义对象/深度层太重且引入分割抖动，tile 是甜点。
2. **是否牺牲前后运动响应换舒适度**：值得，VR 里时序稳定 > 每帧跟手。但要靠 KLT 尺度证据有据地放行真实运动，而非一刀切大 deadband（否则真·推近镜头会“卡住不动”，同样出戏）。
3. **默认偏稳还是偏跟手**：默认偏稳，但门控驱动而非常数 deadband——静态区 deadband 大，KLT 检到尺度变化时自动放大 max_step。
4. **离线/实时分档**：要分，按增强项 B。

## 8.6 运镜下的核心逻辑：区分“视差证据”与“抖动”

这是证据门控思想的试金石，也是 8.5 目前缺失的最关键一块。**运镜时不是“允许抖动”，而是“允许有视差证据支撑的深度变化”。两者几何上可分。**

### 8.6.1 运镜拆成两种，正确行为相反

**① 纯旋转（pan / tilt 摇镜）——没有视差，深度关系不该变。**
摄像机只转不平移时，所有物体（无论远近）在 2D 上以相同规律移动，物体间深度关系完全不变。这时 DA3 让对象间深度产生差异 → 是抖动，应压制。一个全局 homography 能把纯旋转完全解释掉，warp 后残差≈0 → 全局锁定 = 正确。

**② 平移（dolly / truck 推轨）——有运动视差，深度关系确实该变。**
摄像机平移时近物体在 2D 上移动得比远物体快（motion parallax）。这是真实几何证据，近景对象相对背景的视差本来就该变——不是抖动，是正确的 3D，必须放行。

> 所以“原对象和其他对象深度发生显著差异时允许抖动吗”——若差异来自平移视差，它不是抖动，是对的，必须放行；若来自纯旋转或静止却仍在变，才是抖动，要压。

### 8.6.2 warp-then-filter 架构天然能区分

全局 warp（单 homography）只能对齐一个深度层（通常是主背景）：

```text
pan：   所有层都被同一 homography 解释干净，残差≈0  → 全图锁 = 对（无真实深度变化）
dolly： 背景被解释，前景因视差留下残差            → 残差 = 真实视差信号
```

**全局 warp 之后的残差，本身就是“该不该变”的判据：**

- 残差≈0 的 tile（运动被全局变换解释干净）→ 无几何证据 → DA3 还在变就是抖动 → 锁。
- 残差显著且空间连贯的 tile（近景没跟上背景 warp）→ 视差证据 → 放行深度变化。

### 8.6.3 门控判据必须是 per-tile 局部证据，不能是全局 luma

门控变量应是每个 tile 的局部 2D 尺度/运动残差，而非全局亮度差（再次印证 8.5.3 A 把 KLT 提为核心）：

- tile 的局部 scale≈1、运动被全局 warp 解释干净 → 无证据 → 锁深度；
- tile 的局部 scale 在变、或 warp 后仍有连贯位移 → 有证据 → 按证据放行对应幅度的深度变化。

例：推轨经过前景人物，人物 2D 宽度/特征间距在变 → KLT 检测到 → 允许其视差跟着变；同帧远处墙面 scale≈1 → 锁住，不让 DA3 噪声把墙面深度抖出来。

### 8.6.4 必须单独处理的边界：disocclusion（遮挡揭露）

平移运镜会露出之前被挡住的新背景。这些像素没有历史（warp 来的 prev 在此无效/被前景覆盖）→ 不能锁也无从锁 → 必须回退到当前帧 DA3。靠前后向一致性检查 / warp 残差 mask 标出这些区域，单独放行。

### 8.6.5 小结

> 运镜下不允许抖动，只允许有视差证据的深度变化。判据 = 全局运动补偿后的局部残差 + KLT 局部尺度：
> - 纯摇镜：全局变换全解释 → 残差≈0 → 全锁；
> - 推轨平移：前景留残差 = 真实视差 → 按 tile 放行；远景无残差 → 锁；
> - 遮挡新露出区：无历史 → 回退当前帧。
>
> 这证明 warp-then-filter + per-tile 证据门控架构是对的：不是“运镜就放弃稳定”，而是用残差结构把“该变的”和“不该变的”在几何上分开。

## 9. 需要讨论的问题

1. 稳定对象应按什么粒度？

```text
全局 / tile / 语义对象 / 深度层
```

2. 是否需要牺牲一点真实前后运动响应，换取更强舒适度？

```text
VR 观看中，稳定通常比“每帧响应 DA3 变化”更重要。
```

3. 默认参数应偏稳还是偏跟手？

建议默认偏稳：

```text
deadband 较大
max_step 较小
alpha 较小
场景变化时 reset
```

4. 是否需要为离线和实时设置不同档位？

建议：

```text
实时：Base/Detail + Deadband + Global Affine
离线：可选 Tile Affine / Sparse KLT
实验：Dense Flow / GMFlow
```

## 10. 初步结论

当前最值得讨论和推进的基础算法不是 dense flow，而是：

```text
图像证据门控 + 低频视差稳定 + 全局/分块 affine 校正 + 变化速度限制
```

这个方向更符合 2D 转 3D 的实际问题：

```text
不是要精准估计每个像素的运动，
而是要阻止没有图像证据支撑的深度/视差变化。
```

如果对象在 2D 画面中尺寸和位置基本稳定，那么它的 3D 视差也应该稳定。这条约束可以作为后续 2DVR 消抖算法的核心原则。

## 11. 评审后修订要点（一句话清单）

1. 核心原则正确、方向认可，自写算法是当前痛点的性价比最优解（治标但够用，见 8.5.1）。
2. **必修三处致命假设**（V0 默认机位静止）：
   - luma stable_mask 遇运镜失效 → 先做全局运动补偿（phaseCorrelate / KLT）再比对（8.5.2 问题 1）；
   - filter-in-place 假设像素对应 → 架构改为 warp-then-filter（8.5.2 问题 2）；
   - 每帧对齐上一帧 → 漂移累积 → 鲁棒回归 + 缓更新锚（8.5.2 问题 3）。
3. **KLT 从可选提为核心**：一举解决运动补偿 + 局部尺度证据（8.5.3 A）。
4. **离线吃免费红利**：对称时间窗中值替代因果 EMA，消抖且无拖影（8.5.3 B）。
5. **边缘感知**：平坦区强锁、边缘区放松；base/detail 用 guided filter 防边界 halo（8.5.3 C/D）。
6. 开发顺序据此修订（见第 8 节修订版）：Phase 1 首位加入全局运动补偿。
7. **运镜不等于放弃稳定**（见 8.6）：纯摇镜全锁、推轨平移按 per-tile 视差证据放行、遮挡新露出区回退当前帧——用 warp 后残差区分”该变”与”抖动”。

## 12. 实现进度（VVPS 稳定器，自研）

命名：实时模式稳定器 = **VVPS（自研）**；离线模式将另提供 **NVDS** 选项（下阶段接入，不替换 VVPS）。

- ✅ **核心 base/detail（mode=ema）已落地并默认开启**，CPU + GPU 双路（commit `15b10f9`）。替换了会锁前景 + 撕 soft_shift 透明块的旧逐像素稳定器。用户实测”比之前稳定多了”。
  - CPU：`TemporalDepthStabilizer._stabilize_base_detail`（`offline/two_dvr_render.py`）。
  - GPU：`box_blur_h/v` + `base_detail_combine` 自写 RawKernel（`offline/two_dvr_gpu.py`），规避 `cupyx.scipy.ndimage.uniform_filter` 在 CUDA 12.9/sm_120 上的 cccl 编译 bug。
- ✅ **8.6 全局运动补偿（warp-then-filter）— CPU + GPU 双路已实现，默认开**：
  - CPU：`_estimate_global_translation`（phaseCorrelate，下采样 longest≤512）+ `_motion_compensate_base`（warpAffine）。
  - GPU：`_estimate_translation_gpu`（用 `rgb_to_gray_resize` kernel 在 GPU 下采样到小图，只把 ≤512px 小灰图下载到 host 跑 cv2.phaseCorrelate，host 传输极小）+ `translate_bilinear` RawKernel（把 base_prev warp 到当前帧）+ `base_detail_combine` 拆分 prev 读/写缓冲。CPU/GPU 估计符号幅度一致（实测 roll+5/+8/+12）。
  - **GUI 两条路（离线-pynv / 实时-pynv）都经 `prepare_near_gpu(depth, canvas_crop)` 传帧 → 运动补偿在 GUI 直接生效。** 关闭 `PT_TWO_DVR_TEMPORAL_MOTION_COMP=0`。
  - 小注：GPU 路每帧有一次 ≤512px 小灰图的 D2H 同步（phaseCorrelate 在 host），开销小；若实时吞吐吃紧可后续异步化。
- ✅ **8.6.3 per-tile 证据门控 — CPU + GPU 双路已实现，默认开**：全局运动补偿后算"补偿后局部残差"`|cur_gray - warp(prev_gray)|`，tile 平滑后驱动 base-EMA 的**逐像素 alpha**：残差≤`_EVID_R_LO`（静止/平移已解释）→ alpha=`a_lock`(=`depth_alpha×lock_scale`，锁更狠)；≥`_EVID_R_HI`（动对象/视差/遮挡揭露）→ alpha→1（跟当前帧）。自动覆盖 8.6.4 遮挡揭露（高残差→跟当前，不锁垃圾）。
  - CPU：`_evidence_alpha`（全分辨率残差）。GPU：在小图(≤512)上算残差+alpha→上传→`upsample_bilinear` kernel 上采样到全分辨率；`base_detail_combine` 加 alpha-map 支持。实测静止区 alpha=0.1、运动区 0.83。
  - 开关 `PT_TWO_DVR_TEMPORAL_EVIDENCE_GATE=0`；锁强度 `PT_TWO_DVR_TEMPORAL_EVIDENCE_LOCK`（默认 0.5，越小静止区锁越狠）。
- ✅ **8.5.3 B 离线对称时间窗 — GPU 已实现并接入 pynv（GUI 离线路），默认 radius=6**：
  - 组件 `SymmetricBaseWindow`(numpy) / `GpuSymmetricWindow`(cupy)：对 base 取以当前帧为中心的**高斯加权对称均值**（无拖影无延迟），邻帧先全局运动补偿对齐，只在"补偿后残差低=静止"处平滑、运动/遮挡区保留中心帧 base；延迟线 + flush。
  - **静止段反而更抖的修复**：最初用 5 帧中值（radius=2），对静止场景的高斯型 DA3 抖动平滑能力（std 0.53σ）**远不如实时因果 EMA（0.23σ）**——静止段离线反而比实时更抖。改为高斯加权均值 + radius 默认 6（13 帧，0.29σ ≈ 实时），均值比中值更适合连续抖动；可调到 10 追平/超过实时。新增 `weighted_mean_stack` kernel。
  - 接入 `convert_clip_pynv`：window>0 时关掉因果稳定器，prepare 出 near_raw → `window.push` → 延迟 `render_into_gpu` → 收尾 flush。开关 `PT_TWO_DVR_TEMPORAL_WINDOW` / CLI `--temporal-window`（默认 2，0=关，回退因果稳定器）。
  - **修了一个关键 bug**：`cv2.phaseCorrelate` 就地加窗污染输入 → 之前对称窗缓冲灰图被污染、运动补偿/证据门控 GPU 路也一直降级（commit `43c62c2`）。修后 CPU/GPU 对称窗逐像素一致、静止抖动方差降 48%。
  - ⚠️ 未端到端验证：NVENC 编码循环（需真实视频+硬件编码）未在开发环境跑过；render_into_gpu 之前的全链路已冒烟验证。CPU/ffmpeg-pump 路的窗接入为后续。
- ✅ **场景切换 reset（参考 nunif 思路，自研实现，默认开）**：硬切时旧镜头深度会污染新镜头（band EMA / base EMA / 运动补偿混入），仅靠 `temporal_norm_reset` 的 depth-band 跳变启发式不够（深度域相近的切换漏检、域内大变化误触发）。改用 HSV 直方图 `utils.scene_detection.SceneCutDetector`：
  - CPU 因果路：`TemporalDepthStabilizer.begin_frame(frame)` 在归一化前检测+`reset()`，接入 `near_from_depth`。
  - GPU 离线 pynv / GPU 实时 pynv_stream：复用每帧已下载的 `canvas` 跑检测，命中即 `renderer.reset()` 重置 band（对称窗的残差掩码已处理 base 跨切污染）。
  - 开关 `PT_TWO_DVR_SCENE_CUT` / `config.TWO_DVR_SCENE_CUT`（默认开），阈值 `_THRESHOLD`（默认 0.4）。
- ✅ **归一化 band 的离线 lookahead（nunif `--ema-buffer` 思路，自研，默认开）**：之前对称窗稳的是 near base，5/95 band 仍是 causal EMA（有 ~9 帧滞后）。现：离线 window 模式关掉 causal band EMA（`temporal_norm` off，`band_g` 存逐帧 raw band），symmetric window 对窗口内 raw band 做**居中高斯平滑**得对称 band，再用**仿射重归一化**（`near*(g_s/g_c)+(lo_c-lo_s)*g_s`）把各帧 base/detail 校正到对称 band → 零相位、无滞后。新增 `weighted_affine_mean_stack` kernel + reband_a/b。CPU/GPU 逐像素一致、band flicker 降 56%。开关 `PT_TWO_DVR_BAND_LOOKAHEAD`。注：仿射重归一化对 clip 像素是近似（raw band 较稳→修正小→误差小）。
- ⬜ 待办：① CPU/ffmpeg-pump 路统一接入 `--temporal-window`；② 不采用 NVDS 作默认（参考分析与 nunif 一致：算法型 EMA/range-reset 优于重模型，契合本项目 ONNX-only/FPS 敏感目标）。

## 12. 补充定位：算法版 NVDS 稳定器

可以把本方案明确定位为：

```text
DA3 depth predictor + algorithmic stabilizer + stereo renderer
```

也就是一个“算法版 NVDS”：

- NVDS 用神经网络学习如何把 flickering disparity 稳定成 temporally-consistent disparity；
- 本方案不训练模型，而是用图像证据、运动补偿、尺度约束、死区、速度限制和鲁棒统计来做稳定；
- 它应作为 DA3 后处理模块存在，不侵入 DA3 推理，也不改变后面的左右眼渲染器。

建议接口抽象：

```text
DepthStabilizer.update(
    rgb_t,
    near_raw_t,
    frame_index,
    timestamp,
    reset_hint=False
) -> near_stable_t, diagnostics
```

其中 `diagnostics` 至少应包含：

```text
scene_reset          是否触发 reset
global_motion        全局运动估计结果
stable_ratio         稳定区域占比
parallax_ratio       有视差证据的 tile 占比
occlusion_ratio      遮挡/新露出区域占比
affine_scale/bias    本帧全局或 tile scale/bias
mean_disparity_delta 稳定前后视差变化量
```

这样后续可以替换实现：

```text
V0: norm EMA
V1: base/detail + deadband
V2: global warp + affine
V3: tile + KLT scale gate
V4: offline symmetric window
Vx: NVDS / GMFlow / learned stabilizer
```

渲染端只需要消费 `near_stable_t`，不关心稳定器内部实现。

## 13. 3D/VR 特有约束：稳定目标应以“输出视差像素”为单位

当前 near 是 0..1 的抽象深度量，但用户真正感受到的是左右眼之间的像素视差：

```text
disparity_px = near * max_shift_px
```

因此稳定器的关键阈值最好最终换算到 `disparity_px`：

```text
static_deadband_px    静态区域低于多少 px 的视差变化直接吞掉
static_max_step_px    静态区域每帧最多允许变化多少 px
motion_max_step_px    有证据运动区域每帧最多允许变化多少 px
```

建议初始讨论值：

```text
static_deadband_px = 0.15 ~ 0.35 px
static_max_step_px = 0.25 ~ 0.75 px/frame
motion_max_step_px = 1.0 ~ 3.0 px/frame
```

原因：

- near 的同样变化在不同输出宽度、不同 3D strength 下体感不同；
- 直接用 px 可以把算法阈值和用户眩晕来源对齐；
- 也方便做 ROI 量化：统计静态区域 `disparity_px` 的帧间标准差。

## 14. 需要补充的特殊场景与处理建议

### 14.1 纯 zoom / 变焦

变焦会让全图尺度变化，但它不等同于对象真实靠近。若算法把 zoom 误判成所有对象深度变化，会造成整屏 3D 呼吸。

处理建议：

```text
KLT / affine 先判断是否为全局一致 scale；
全局一致 scale 更可能是镜头 zoom，应优先保持相对深度稳定；
只有局部 scale 相对背景变化时，才放行局部深度变化。
```

### 14.2 数字裁切 / 视频稳定 / 后期防抖

很多视频本身经过电子防抖，会产生全局平移、缩放、边缘裁切和黑边变化。它会污染 luma mask 和 KLT。

处理建议：

```text
忽略黑边/letterbox 区域；
运动估计只在有效画面 mask 内做；
检测到边界裁切变化时，不把边缘 tile 作为 affine 参考。
```

### 14.3 淡入淡出 / 闪白 / 曝光变化

这类变化不是几何运动，但 luma_diff 会全局变大，容易误 reset。

处理建议：

```text
除 luma absolute diff 外，增加 normalized correlation / gradient correlation；
如果梯度结构稳定但亮度整体变化，降低 luma gate 权重，不立即 reset。
```

### 14.4 低纹理区域：天空、墙面、虚化背景

DA3 在低纹理区域容易深度漂移；KLT 又没有足够特征。

处理建议：

```text
低纹理 + 无局部尺度证据 = 强锁 base；
这类区域宁可稳定，也不要让深度自由漂。
```

### 14.5 深度边界：人脸/头发/手臂/细物体边缘

边界处最容易发生 DA3 边界游泳。过强时间平滑会造成边缘拖尾或 halo。

处理建议：

```text
按 depth gradient / RGB edge 降低历史权重；
base 用 guided filter / bilateral，避免高斯 blur 穿过边界；
detail 尽量来自当前帧，只对低频 base 强稳定。
```

### 14.6 非刚体：人物、衣服、头发

人物局部会变形，不能完全按刚体尺度约束。

处理建议：

```text
tile 级证据比对象级整体证据更稳；
人脸/躯干这种大平面可以强锁；
手、头发、衣摆等高运动边缘放松。
```

### 14.7 透明/反射/水面/烟雾/火焰/屏幕内容

这些区域的 RGB 变化不对应稳定几何表面，深度估计天然不可靠。

处理建议：

```text
检测高频闪烁或低 KLT 可跟踪率；
不做强 affine 参考；
输出上可偏向背景深度或降低 3D 强度，避免产生强烈错误视差。
```

### 14.8 遮挡揭露和对象进出画

新出现区域没有历史，强行稳定会拖出上一帧物体深度。

处理建议：

```text
warp 后无有效来源、KLT forward/backward 不一致、或 luma 残差连片过大 → 标记 disocclusion；
disocclusion 区域直接用当前 near，等待数帧建立历史。
```

### 14.9 深度排序翻转

DA3 有时会让前后景排序突然翻转。简单 affine/EMA 无法完全修复这种语义级错误。

处理建议：

```text
维护局部 depth ordering inertia；
如果两个大区域的相对排序突然翻转，但 RGB/KLT 无明显遮挡证据，则延迟或拒绝翻转；
只有出现明确遮挡关系变化时，允许排序改变。
```

### 14.10 动画/二次元/低帧率视频

动画线条清晰但真实纹理和运动模糊少，DA3 深度可能更跳；低帧率视频帧间差异大，KLT 可靠性下降。

处理建议：

```text
动画：更依赖边缘/区域级稳定，减少纹理尺度推断权重；
低帧率：按 timestamp 而不是 frame count 调整 max_step，避免 24fps/60fps 参数不一致。
```

## 15. 建议的鲁棒性优先级

为了避免算法越来越复杂但不可控，建议按以下优先级设计：

```text
1. 绝不制造新的强错误视差；
2. 静态区域视差必须稳；
3. 运镜下能区分纯旋转和真实平移视差；
4. 边界不拖影、不 halo；
5. 真实前后运动允许稍慢响应，但不能完全锁死；
6. 实时默认必须轻量，离线可以开更强选项。
```

这也意味着默认策略应保守：

```text
无证据 -> 锁 / 慢变
弱证据 -> 限速变化
强证据 -> 放行
无历史 -> 当前帧
```

## 16. 建议新增验证指标

除肉眼看 VR 外，应补充几个自动指标：

### 16.1 静态 ROI 视差抖动

人工或自动选静态 ROI，统计：

```text
std(disparity_px[t] - disparity_px[t-1])
mean(abs(disparity_px[t] - disparity_px[t-1]))
```

目标是稳定后显著下降。

### 16.2 图像证据与视差变化一致性

统计 tile 级：

```text
image_scale_delta vs disparity_delta
```

如果 `image_scale_delta≈0` 但 `disparity_delta` 很大，应被算法压掉。

### 16.3 Reset 频率

记录每分钟 reset 次数。频繁 reset 通常说明把运镜或曝光变化误判成场景切换。

### 16.4 舒适度指标

按输出视差像素统计：

```text
静态区每秒累计视差抖动量
全帧 p95 视差变化速度
边界区视差变化速度
```

这些指标比 near 的数值变化更贴近用户体感。

## 17. 补充：稳定器需要“历史置信度”，不能只有上一帧

如果稳定器只有一个 `near_stable_prev`，很容易在真实片源里出现两类问题：

```text
证据稍微变弱 -> 立刻放弃历史 -> 抖动回来
证据稍微变强 -> 立刻强锁历史 -> 边界拖影
```

建议稳定器维护一个与 near 同分辨率或 tile 同分辨率的 `history_confidence`：

```text
history_confidence = 0.0 ~ 1.0
```

它表示“warp 过来的历史在当前帧这个位置是否可信”，由以下信息共同决定：

- 全局 warp 是否成功；
- warp 后 RGB/梯度残差是否小；
- KLT forward/backward 是否一致；
- 当前区域是否为 disocclusion / 新露出区域；
- 当前区域是否有足够纹理或稳定边缘；
- 当前区域是否接近深度边界或遮挡边界。

使用方式：

```text
高置信度：强 deadband、低 alpha、低 max_step，强力稳定；
中置信度：允许限速变化，避免拖影；
低置信度但有历史：弱稳定，快速重新收敛；
无历史/新露出：直接使用当前 near，进入 warm-up。
```

同时应使用迟滞，避免 lock/unlock 每帧跳变：

```text
lock_threshold   > unlock_threshold
reset_threshold  单独设置，不能和 unlock 混用
```

这会让算法更像一个鲁棒稳定器，而不是单纯滤波器。

## 18. 补充：短期历史 + 长期锚，抑制慢漂移

只对齐上一帧会产生长镜头慢漂移。建议把状态分成两层：

```text
short_state:
    上一帧 RGB
    上一帧 near_stable / base_stable
    上一帧 history_confidence

anchor_state:
    缓慢更新的参考 base/disparity
    缓慢更新的全局 scale/bias
    每个 tile 的稳定类别与可信度
```

短期状态负责帧间连续，长期锚负责防止“世界慢慢变近/变远”。

更新策略：

```text
只有在高置信静态区域，才允许 anchor 缓慢更新；
发生 reset / crossfade / 大遮挡时，不更新 anchor；
全局 scale/bias 不允许长期偏离 1/0，应有软约束回归。
```

这样可以解决一个很隐蔽的问题：单帧看很稳，但 30 秒长镜头后整体 3D 强度慢慢变了。

## 19. 建议 V1 可执行流程

V1 不追求一步到 NVDS 级别，而是先实现一个足够鲁棒、可诊断、性能安全的版本：

```text
0. 输入准备
   RGB crop、near_raw、valid_mask、letterbox/黑边 mask
   将关键阈值换算到 disparity_px

1. Reset/soft-reset 检测
   seek、clip 起点、timestamp 跳变、硬切、严重坏帧 -> hard reset
   crossfade、闪白、曝光突变 -> soft reset / 降低历史权重

2. 全局运动估计
   phaseCorrelate 或 KLT + estimateAffinePartial2D
   输出 global_transform、global_motion_quality

3. 历史 warp 到当前坐标
   warp near_stable_prev/base_prev/history_confidence_prev
   得到 warped_prev、warped_confidence

4. 计算 per-tile 证据
   warp 后 RGB/gradient residual
   KLT 局部 scale / residual
   feature_count / texture_score
   disocclusion / occlusion mask
   edge_score / depth_gradient

5. 当前 near 分解
   用 guided filter / bilateral 得到 base_raw + detail_raw
   避免高斯 base 穿过深度边界

6. 鲁棒 scale/bias 校正
   只在高置信静态 tile 上拟合
   排除 near 饱和区、边界区、运动物体区
   约束 scale/bias 范围，并参考长期 anchor

7. 证据门控参数
   静态可信 tile：大 deadband、小 max_step、小 alpha
   有局部尺度/视差证据 tile：小 deadband、大 max_step、中高 alpha
   disocclusion tile：alpha=1，直接当前帧
   低纹理但历史可信 tile：强锁 base

8. 稳定 base
   在 disparity_px 单位做 deadband/max_step
   再转回 near/base 表示

9. 回填 detail
   detail 主要取当前帧
   对极高频闪烁可以轻微限幅，但不要拖边界

10. 空间清理与输出
    edge-aware 平滑门控图和 base
    clip near 到合法范围
    输出 near_stable_t + diagnostics

11. 更新状态
    更新 short_state
    仅用高置信静态区域缓慢更新 anchor_state
```

优先在 DA3 输出分辨率上做这些操作，不在最终 4K/8K 输出图上做。这样性能可控，且与现有左右眼渲染器解耦。

## 20. 重要边界：这是深度/视差稳定器，不是 RGB 防抖器

全局运动补偿只用于找到“上一帧历史在当前帧对应哪里”，不应该改变输出 RGB 画面。

正确关系是：

```text
当前 RGB 原样进入 stereo renderer
稳定器只输出与当前 RGB 对齐的 near_stable
renderer 用当前 RGB + near_stable 生成左右眼
```

不能把算法做成普通视频防抖，否则会引入裁切、黑边、画面漂移和额外重采样。这里的目标不是稳定画面位置，而是稳定当前画面坐标下的输出视差。

同理，稳定器也不应该隐藏或替代 hole filling / occlusion 处理：

```text
disocclusion 没有可信历史 -> 当前 near
stereo 渲染产生的洞/边界 -> 由后续 hole fill 处理
```

## 21. 额外特殊场景补充

### 21.1 rolling shutter / 手机防抖残留

手机或运动相机素材可能有 rolling shutter，全局 affine/homography 无法解释整帧。

处理建议：

```text
全局 motion quality 低但局部 KLT 连贯时，降低全局锁定权重；
允许 tile 级 transform 或更保守的 per-tile evidence；
不要因为全局配准失败就直接全帧 reset。
```

### 21.2 运动模糊

快速运动或低快门会让 KLT 特征质量下降，但这不等于深度应该自由漂移。

处理建议：

```text
检测 blur/gradient 能量下降；
若历史仍可信，短时间保持稳定；
若连续多帧 blur 且 residual 大，逐步放松而不是突然 reset。
```

### 21.3 字幕、台标、UI 叠加层

字幕/台标通常贴在屏幕平面，且不属于真实场景几何。它们会干扰 KLT、luma residual 和 DA3 深度。

处理建议：

```text
检测长期固定在画面边缘/底部的高对比文字区域；
不把这些区域用于全局 motion / affine 参考；
输出 depth 可偏向屏幕平面或低 3D 强度，避免字幕左右眼错位晕眩。
```

### 21.4 前景占满画面 / 人脸特写

当人物或物体占据大部分画面时，KLT 估计到的“全局运动”可能其实是前景运动，不是相机运动。

处理建议：

```text
全局参考应优先选稳定背景或远景 tile；
如果没有可靠背景，就把当前主对象作为局部稳定对象处理；
不要用一个运动前景的 scale 去校正整帧 depth。
```

### 21.5 画中画、镜子、屏幕内视频

画面中可能有独立运动的小屏幕、镜子、电视、监控画面。它们的运动不服从主场景几何。

处理建议：

```text
这类区域通常表现为矩形边界内 residual 与外部不一致；
不要把它们用于全局 motion；
内部按独立 tile 证据处理，必要时降低 3D 强度。
```

### 21.6 自动对焦 / 景深呼吸

自动对焦会改变清晰度和局部对比，但不一定有真实距离变化。DA3 可能把虚化变化误判为深度变化。

处理建议：

```text
blur 变化不能单独作为深度变化证据；
尺度/轮廓没有变化时，仍应稳定视差；
焦点切换区域可短时降低 alpha，等待图像结构稳定。
```

### 21.7 near 饱和和深度压扁

DA3 归一化后可能出现大片 near 接近 0 或 1。饱和区域参与 affine 拟合会把 scale/bias 拉坏。

处理建议：

```text
affine 拟合排除 near < 0.03 或 near > 0.97 的区域；
对饱和区单独限速；
避免让少数极近/极远像素决定整帧 3D 强度。
```

### 21.8 帧率变化 / 丢帧 / VFR

离线和实时都可能遇到 VFR、解码丢帧或 timestamp 不均匀。按“每帧固定 max_step”会导致不同 fps 下体感不一致。

处理建议：

```text
max_step 应按 dt 缩放，而不是只按 frame count；
dt 异常大时触发 soft reset 或放宽跟随；
diagnostics 记录 timestamp gap，便于定位卡顿和抖动。
```

## 22. 实时/离线分档建议

实时默认必须把最坏情况耗时控制住：

```text
Realtime V1-lite:
    phaseCorrelate 或少量 KLT
    global warp
    base/detail
    disparity-px deadband/max-step
    全局 affine + 简单 tile evidence
    不启用 dense flow

Offline V1-HQ:
    更多 KLT 点
    tile affine / tile scale evidence
    5~7 帧对称窗口
    更完整的 disocclusion 检测
    更详细 diagnostics 输出

Experimental:
    Dense flow / NVIDIA Optical Flow / GMFlow / learned stabilizer
    只作为对比或高质量慢速选项
```

默认策略建议：

```text
有历史且无证据变化 -> 稳定
有历史且有强证据变化 -> 限速放行
无历史或新露出 -> 当前帧
证据互相矛盾 -> 保守降低 3D 强度或降低历史权重，不做激进校正
```

这样即使算法判断错，也更不容易制造强烈的错误视差。

## 23. 建议留给专家继续讨论的决策点

后续评审可以重点决定以下问题：

1. 稳定变量最终选 `near`、`base near`、还是 `disparity_px` 作为主状态？
2. V1 实时默认用 `phaseCorrelate`，还是直接用少量 KLT + affine？
3. tile 网格默认多大，是否需要随 DA3 输出分辨率自适应？
4. 低纹理区域应强锁到历史，还是偏向当前帧并降低 3D 强度？
5. 字幕/台标是否进入专门 mask，还是先用通用异常区域处理？
6. 离线对称窗口默认 5 帧、7 帧，还是按 fps 换算到固定毫秒？
7. 当舒适度和真实前后运动响应冲突时，默认参数应偏稳到什么程度？

这些点决定的是产品默认手感，不只是算法细节。

## 24. 产品化要求：必须有硬开关和可回退路径

稳定器不能成为不可绕过的核心路径。它是 DA3 后面的算法稳定层，效果不好、误判、性能不达标或遇到特殊片源时，用户必须能直接关闭。

建议开关层级：

```text
总开关:
    depth_stabilizer_enabled = true/false

模式:
    off          完全关闭，near_raw 直接进入 renderer
    lite         实时轻量稳定，默认候选
    hq_offline   离线高质量稳定
    experimental Dense flow / learned / 外部算法实验
```

关闭语义必须非常明确：

```text
off = bypass
near_stable_t = near_raw_t
diagnostics.mode = off
不执行 motion/KLT/affine/window/filter
不增加额外耗时
不改变渲染器后续行为
```

切换状态也要处理干净：

```text
从 on -> off：
    清空稳定器历史或标记 inactive；
    后续帧完全不读旧历史。

从 off -> on：
    第一帧 hard reset；
    用当前 near_raw 初始化历史；
    warm-up 数帧内逐步增加历史权重。
```

用户侧入口建议同时覆盖：

```text
实时播放 UI 开关
离线转换 UI 开关
CLI 参数：--depth-stabilizer / --no-depth-stabilizer
环境变量或配置项：PT_TWO_DVR_DEPTH_STABILIZER=1/0
```

如果稳定器默认开启，也应保留一个清晰的“原始 DA3 / 无稳定”对照模式，方便定位问题：

```text
Raw DA3
Realtime Lite
Offline HQ
Experimental
```

开发和测试也应把 bypass 当作正式路径覆盖：

```text
关闭稳定器时输出必须与旧路径一致；
关闭稳定器时 FPS 不应下降；
打开/关闭切换不应继承旧状态导致画面突跳；
diagnostics/log 必须能证明当前是否真的 bypass。
```

这条约束很重要：稳定器的目标是提高舒适度，不应在某些片源上把用户锁死在更差的效果里。
