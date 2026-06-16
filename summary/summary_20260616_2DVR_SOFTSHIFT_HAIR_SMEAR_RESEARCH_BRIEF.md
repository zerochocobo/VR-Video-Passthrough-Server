# 2D→3D soft_shift 头发横纹 / 左右眼不一致 —— 研究简报（交接给下一位专家）

> 目的：把"实时/离线 2D→3D 的 soft_shift 补洞,在头发等飘逸细节处产生横向涂抹/横纹,
> 且左右眼不一致导致闪烁"这个问题的全部上下文、已做尝试、已排除的假设、当前的**未解矛盾**
> 一次讲清,便于换人接手。日期 2026-06-16,分支 `feature/2dvr`(无 remote,未 push)。

---

## 1. 功能与目标

把普通 2D 视频实时/离线转成 **flat3d 左右眼 SBS(3D)**。深度来自单目 **Depth Anything 3
(DA3) ONNX**(固定 518×518 输入,输出"距离型"深度,值越大越远)。两种补洞模式:

- `soft_shift`(质量模式,**当前默认**,本简报的主角):前向映射(forward warp)+ z-buffer +
  遮挡补洞。能给出"正确的遮挡关系 + 露出真实背景",但在细节处会涂抹。
- `inverse_warp`(快速模式):反向采样(backward warp),**无空洞**。但在深度突变处会
  拉伸/撕裂,**用户实测"像被刀砍、人像不全,不可接受"**,已暂时否决,下阶段单独调优。

约束:实时需 ~60–85fps@1080p,GPU 常驻(CuPy,**无 torch、sm_120 上无 cv2.cuda**),
CPU inpainting 不可用于实时。深度是低分辨率(518)单目,质量有限。

---

## 2. 问题现象

- 左眼画面中,**人物头发左侧**出现**暗色横向条纹(横纹/拉丝)**;实体边缘(手臂/腿/身体)
  在最近的修复后已干净。
- 左右眼不一致(同一遮挡两眼填相反侧、内容不同)→ **双眼竞争 + 逐帧变化 → 闪烁**,
  用户反馈"好几秒都在闪"。
- 用户反复强调:**脏的总是左眼**(他用概率论质疑这不该是随机抠图误差)。
- 复现片段:`videos/test_4k2d.mp4`(1920×1080 HEVC 60fps),源 **33–41s**(= 输出片段
  `_S000030_E000100` 的第 3–11s)是一个女生坐在凳子上、有头发。

---

## 3. 管线与关键代码位置

实时与离线 GPU 都走 **同一套 CuPy kernel**(`offline/two_dvr_gpu.py` 的
`GpuStereoRenderer`)。CPU 路径(离线 ffmpeg 回退)在 `offline/two_dvr_render.py`,逻辑等价。

near 的准备(soft_shift,`offline/two_dvr_render.py`):
1. `_normalize_near(depth)`:深度→反深度→5/95 百分位裁剪→归一化 0..1(1=最近)。
2. `near_from_depth(depth, "soft_shift")`:**不**做高斯预平滑,调用 `_dilate_near_fg`。
3. `_dilate_near_fg(near)`:前景 near **膨胀(max 滤波)**,半径 `round(width/512)`(518→1px,
   1920→~4px,4K→~8px),把"深度边界外、颜色是前景"的像素并进前景,让它随前景平移。

GPU kernel(`_SOFT_SHIFT_KERNELS` in `offline/two_dvr_gpu.py`):
- `_near_at`:把低分辨率 near **双线性上采样 + toggle(形态学对比)锐化**——吸附到局部
  水平 min/max 更近者,把软深度边变硬边。
- `fw_zbuf`:前向散射 + z-buffer。**眼别符号**:`sign = eye==0 ? +1 : -1`
  (eye0=左眼,前景右移;eye1=右眼,前景左移)。
- `fw_color`:写入 z-buffer 胜者颜色。空洞处 zbuf==0、颜色=0。
- `fw_fill`:**按眼别从背景侧补洞**——左眼取最近左侧有效像素(li),右眼取最近右侧(ri);
  单侧无效时回退另一侧。
- `fw_blend`:**只羽化已填空洞像素**,模糊核**排除前景**(near 高的邻居),保证人体剪影硬、
  只软化背景接缝。
- `PT_TWO_DVR_DEBUG_HOLES=1`:跳过 fill/blend,把空洞涂**洋红**,看原始 warp(flat3d soft 限定)。

CPU 对应:`_make_soft_shift_pair` / `_forward_warp_eye_rgb`(返回 out, holes, warped near_buffer)/
`_shift_fill_holes_rgb` / `_soft_blend_holes_rgb` / `_sharpen_near_edges`。

其它文件:`offline/two_dvr_pynv.py`(离线 GPU `convert_clip_pynv`)、
`pipeline/pynv_stream.py`(实时 worker `_worker_loop_two_dvr`)、`offline/two_dvr.py`(CLI)、
`offline/da3_depth.py`(DA3 引擎)。配置:`config.py` 的 `TWO_DVR_HOLE_FILL`、
`ui/settings.py` 的 `two_dvr_live_hole_fill`(默认 soft_shift)。

---

## 4. 已做的修复(均在 `feature/2dvr`,按时间)

| commit | 内容 | 效果 |
|---|---|---|
| `f2be654` | 补洞方向按眼别(左眼从左、右眼从右),对齐 CPU 参考 | 必要,未根治 |
| `12d0656` | soft_shift 跳过深度高斯预平滑(原会使前景"变胖") | 减轻 |
| `b9e8135` | `_near_at` 内 toggle 锐化深度软边 | 实体边缘改善 |
| `d1179d4` | 羽化改为"硬剪影 + 仅背景侧、排除前景" | 剪影不再发虚 |
| `48e9baf` | `_dilate_near_fg` 前景膨胀 | **实体边缘(手臂/腿)已干净**;头发仍横纹 |
| `8e184f5` | `PT_TWO_DVR_DEBUG_HOLES` 调试开关 | 工具 |

净结论:**实体边缘已修好;飘逸头发仍有横纹,且左右眼不一致。**

---

## 5. 已做的诊断与已排除的假设

1. **左右眼对称性(已用数值证明对称,排除"左眼偏向 bug")**
   做"镜像测试":把同一帧(及其深度)水平镜像后渲染。若渲染器左右对称,则
   "镜像左眼翻回来" 应逐像素等于"原始右眼"。结果:
   - 渲染器对称性差异 mean=**0.12**、>25 像素占比 **0.1%**(≈0,仅取整噪声)。
   - 真实左右眼差异 mean=**6.77**、>25 占比 **8.6%**(是前者的 ~55 倍)。
   → **算法严格左右对称**。"总是左眼脏"是**这条素材内容不对称**(左侧墙有砖纹理/打光,
     右侧纯墙;镜像后脏会跑到右眼)。但"左右眼不一致→闪烁"本身是真实存在的(8.6% 像素不同)。

2. **深度热力图**:头发**大体被判为前景**(红),墙为背景(蓝),但边界是 **~8px 软过渡**
   (518 深度上采样到 1080p 的固有结果)。

3. **膨胀半径扫描**(在头发帧实测):当前 ~±4px 盖不住飘散发丝;需 ~±15px(low-res 9×9)
   才能让头发横纹消失,但那会让整体轮廓**鼓一圈背景光晕**,不可接受。

4. **inverse_warp 对照**:同帧/同段**完全干净**(无空洞=无涂抹),但用户认为它"砍人像"
   (深度突变处撕裂),否决。

---

## 6. ⚠️ 当前未解的核心矛盾(交给下一位专家的关键)

我(上一位)给出的诊断是:**头发处空洞被"没随前景平移的头发像素"污染(抠图/深度问题)**,
依据是 `PT_TWO_DVR_DEBUG_HOLES` 输出里头发区是"洋红空洞 + 深色头发像素"的椒盐状混杂。

但**用户仔细看了调试视频后明确反驳**:
> "发丝问题完全看不到,**空洞左侧是干净的背景**。"

也就是说:**用户认为遮挡空洞是被干净背景包围的**(左眼空洞左侧=干净墙)。

这与"填充后却出现头发横纹"直接矛盾:
- 若空洞左侧确为干净背景,则左眼 `fw_fill` 取 li(左)= 干净背景 → 填充应当干净;
- 但成品确有头发横纹。

**因此必须先解决这个矛盾,定位横纹的真正来源。** 候选假设(尚未逐一证伪):
- (a) 我把"洋红空洞里的深色像素"误读成了污染,其实那是**发丝之间露出的真实背景孔洞**
  (椒盐),填充时这些细孔被相邻发丝填上→视觉上头发"变实/拉丝"。即横纹来自**发丝间细孔的
  填充**,而非主空洞左侧。
- (b) `fw_fill` 在某些行 `li<0` 回退到 `ri`(=前景)——需检查这些行的占比与分布。
- (c) `fw_blend` 羽化在头发处把前景带入(虽已"排除前景",但 near 阈值在头发软边可能失效)。
- (d) 横纹其实是 `_dilate_near_fg` 膨胀后,**背景侧**多出的"随前景平移的墙色/发色环"
  造成的,而非补洞。
- (e) 我之前"左侧暗带=横纹"的截图,可能把**她真实垂落的头发**当成了 artifact。

**建议下一步**:把 `DEBUGHOLES`(洋红空洞)与 `filled`(成品)**同帧同裁剪逐像素叠加**,
精确标出"成品里横纹的像素"在调试图里对应的是 空洞 / 前景 / 背景 / 发丝细孔,
一锤定音地确定横纹来源,再决定改 fill / 改 blend / 改膨胀 / 还是确属深度。

---

## 7. 复现方法(uv 环境)

环境:`G:\GIT\debug\PTMediaServer\.venv`(uv),cupy 14.0.1 + cv2 + RTX 5060 Ti(sm_120)。

- 生成 soft_shift 成品(用 CLI,会 `apply_runtime_dll_paths()` → TensorRT 深度,快):
  ```
  uv run python -m offline.two_dvr single videos/test_4k2d.mp4 \
    --start 30 --duration 30 --model base --hole-fill soft_shift \
    --projection flat3d --eye-distance 65 --out-dir <dir>
  ```
- 调试空洞图:同命令前加 `PT_TWO_DVR_DEBUG_HOLES=1`。
- inverse_warp 对照:`--hole-fill inverse_warp`。
- ⚠️ 用裸 `uv run python -c "import ..."` 直接 import `Da3DepthEngine` 会**因 DLL 路径**
  退回 CPU 深度(很慢);务必走上面的 CLI 或先 `apply_runtime_dll_paths()`。
- SBS 输出左半=左眼、右半=右眼。本素材 1920×1080→SBS 3840×1080。

已生成可直接对比的三条视频(在 `videos/`):
- `test_4k2d_S000030_E000100_3D_LR_Screen.mp4` —— soft_shift 成品(有头发横纹)
- `test_4k2d_S000030_E000100_3D_LR_Screen_DEBUGHOLES.mp4` —— 洋红空洞(未填充)
- `test_4k2d_S000030_E000100_3D_LR_Screen_INVWARP.mp4` —— inverse_warp(干净但砍人像)
- 关注源 33–41s(= 视频 3–11s)左眼头发左侧。

---

## 8. 其它待办 / 备注

- `_dilate_near_fg` 自适应半径 `round(w/512)`;在 4K(native pynv 不降分辨率)半径更大,
  需复测是否过度鼓边。
- `_dilate_mask_np`/`_box_blur_rgb`(two_dvr_render.py)已无人调用,可清理。
- 实时与离线共用 `GpuStereoRenderer`,改 kernel 两处都生效。
- 若结论确为"soft_shift 对头发结构性不行",产品决策可能是:把 inverse_warp 的撕裂调优好
  后切默认,或对人像/发区做混合(soft_shift 实体 + inverse_warp 细节),或上更高分辨率/
  matting 深度。这些都属"下阶段"。

相关历史详记:同目录
`summary_20260616_2DVR_SOFTSHIFT_*`(方向 / 去平滑 / 锐化 / 硬剪影羽化 / 前景膨胀)。
