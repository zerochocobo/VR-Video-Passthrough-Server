# 2D→3D soft_shift 前景变胖（深度预平滑）修复（2026-06-16）

接 `summary_20260616_2DVR_SOFTSHIFT_HOLEFILL_FIX_CN.md`。上一版把 GPU 补洞方向改成
按眼别对齐 CPU 后，实时实测人像仍不对：左眼对象左侧被“拉长羽化”（用前景填了空洞），
右眼对象更细，左右眼脸/胳膊粗细不一致。

## 用 uv 环境实测定位

在 `uv` 环境（cupy 14.0.1 + cv2 + RTX 5060 Ti）用合成图（居中红色前景竖条，宽 48px，
背景为渐变）逐像素打印前向映射 + 补洞结果，确认了真正的根因。

左眼（前景右移）前向映射后中间行（`b`=背景 `F`=前景 `.`=空洞）：

```
cols:  …135 136 137 138-142 143…
pre :  bbbb  .   F   .....   FFFF…
```

**深度在前向映射前被 `_smooth_depth`（3×3 高斯）模糊**，使前景边界像素拿到一个中间视差，
被散射到背景与遮挡空洞之间，留下一个**孤立的 1px 前景碎片（col 137）**。随后方向补洞
（左眼从左取最近有效像素）就近抓到了这个前景碎片，而不是真正的背景（col ≤135），把前景
**向空洞方向摊开**，红条从 48px 变胖到 52px。两眼的“变胖”发生在各自的遮挡侧（左眼向左、
右眼向右），真实深度下左右不对称，即用户看到的“左眼脸大胳膊粗、右眼更细”。

关键对比（合成图，正确立体应保持 48）：

| 深度 | soft_shift 补洞后前景宽 |
|---|---|
| 预平滑（旧） | **52**（变胖） |
| 不平滑 | **48**（正确） |

`_normalize_near` 的 5/95 百分位裁剪已足够稳，前向映射不需要、也不应该预平滑——锐利的
深度边界才能让遮挡空洞干净地从真正背景填充。`inverse_warp` 是连续重采样、无空洞，平滑能
抑制视差边缘的锯齿，**保留**。

## 修复

新增集中式 `offline/two_dvr_render.py:near_from_depth(depth, hole_fill_mode)`：

- `soft_shift` → `_normalize_near(depth)`（不平滑）
- 其它（`inverse_warp`）→ `_normalize_near(_smooth_depth(depth))`（保留平滑）

各路径改走该函数 / 对应分支：

- `offline/two_dvr_render.py`：`_make_soft_shift_pair` 改用未平滑 near；
  `StereoRenderer` 通用（soft_shift）路径经此函数；inverse_warp 快路径仍平滑。
- `offline/two_dvr_gpu.py`：`GpuStereoRenderer.render()` 按 `self._soft` 分支
  （`render_into_gpu` 由调用方算 near）。
- `pipeline/pynv_stream.py` 实时 `_worker_loop_two_dvr`：
  `near = near_from_depth(depth_crop, hole_fill)`。
- `offline/two_dvr_pynv.py` 离线 GPU：`near = near_from_depth(depth_crop, args.hole_fill)`。

补洞方向仍是上一版的按眼别（左眼左、右眼右）——本修复让它拿到无碎片的干净空洞即可正确工作。

## 验证（uv 环境）

合成图，修复后：

```
soft_shift    CPU [48, 48]  GPU(render_into_gpu) [48, 48]   ← 不再变胖
inverse_warp  CPU [42, 42]  GPU [42, 42]                    ← 无空洞，不受影响
```

CPU 与 GPU 一致。`py_compile` 全过；`tests/test_offline_outputs.py` 6 passed。

**仍需在真实 app 实时模式复测**（test_4k2d.mp4，hole_fill=soft_shift）：确认人脸/肢体两眼
粗细一致、遮挡侧锐利边缘正常（那是正确遮挡，不是 bug）。

## 备注

- 用户观察到的“右眼人左侧像刀切”是**遮挡侧的正确锐利边界**（前景盖住背景），非缺陷。
- 去掉 soft_shift 的预平滑对噪声的影响可忽略：DA3 深度本身较平滑，且百分位归一化已裁剪极值；
  换来的是无前景摊开。inverse_warp 不变。
