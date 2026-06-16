# 2D→3D soft_shift 补洞左右眼方向修复（2026-06-16）

## 现象

实时 2D→3D（以及离线 GPU soft_shift 路径）输出中，人脸等前景物体出现**左右眼不对称畸变**：
一只眼睛（左眼）边缘被拉长，另一只眼睛（右眼）边缘被切短。VR 头显里表现为人脸/前景
轮廓鬼影、立体不舒适。

## 根因

soft_shift（前向映射 + z-buffer + 补洞）的 GPU 实现 `offline/two_dvr_gpu.py` 的补洞核
`fw_fill`，对**左右眼用了同一套补洞规则**，而正确做法是两眼方向相反。

立体前向映射时，两眼平移方向相反：

- 左眼（`fw_zbuf`/`fw_color` 中 `eye==0`，sign=+1）把前景**向右**推；
- 右眼（`eye==1`，sign=-1）把前景**向左**推。

因此遮挡空洞（disocclusion，被前景让开后露出的背景）在两眼里出现在**相反的一侧**，
补洞也必须从相反的一侧取背景像素填：

- 左眼：从**左侧**邻居（背景）填；
- 右眼：从**右侧**邻居（背景）填。

这正是 CPU 参考实现 `offline/two_dvr_render.py` 的语义（已验证、视觉正确）：

```python
left  = _shift_fill_holes_rgb(left_raw,  left_holes,  -1)  # 左眼，方向 -1（左）
right = _shift_fill_holes_rgb(right_raw, right_holes,  1)  # 右眼，方向 +1（右）
```

但移植到 GPU 时，`fw_fill` 改用了一套**与眼无关**的“优先更远（near 更小=背景）一侧”启发式：

```c
// 旧（错误）：两眼相同，且 nr==nl 平局时恒取右邻居
float nl=_near_at(...,li-lo), nr=_near_at(...,ri-lo);
pick = (nr <= nl) ? ri : li;
```

平局（含平坦区/噪声）恒偏向右邻居，对左眼相当于把前景拉进空洞 → 该眼边缘被拉长；
右眼则相对被切短。两眼方向不一致即造成上述不对称畸变。

## 修复

`offline/two_dvr_gpu.py` 的 `fw_fill`：当两侧都有有效像素时，按眼别选取背景侧，
与 CPU 的固定方向严格对齐。

```c
else pick = (eye == 0) ? li : ri;  // 左眼取左侧，右眼取右侧
```

- 单侧有效时仍回退到有效的一侧（行为不变）。
- `nearmap/h/w` 形参保留（不再使用，但 `_launch` 调用签名不变，避免牵动其它核）；
  `_near_at` 仍被 `fw_zbuf`/`fw_color`/`fw_blend` 使用，保留。
- inverse_warp 路径无空洞（逆采样），不涉及；本修复同时覆盖 flat3d 与 VR
  （fisheye/hequirect）的 soft_shift（VR 是 `fw_fill` 之后再 `project_flat_lr`）。

## 影响范围

- 实时 2D→3D live（`pipeline/pynv_stream.py` 的 `_worker_loop_two_dvr`，默认 soft_shift）。
- 离线 GPU 路径 `offline/two_dvr_pynv.py`（soft_shift）。
- CPU 路径本就正确，无改动。

## 验证

- `python -m py_compile offline/two_dvr_gpu.py` 通过。
- CUDA 核为仅改一个三元表达式 + 删两行的最小改动，语法安全。
- **未能在本机 JIT/实跑核**：沙箱 bare python 缺 `cv2`（环境问题，HANDOVER 已记），
  GPU 不可用。需在真实 app 跑一段含明显前景（人脸/近物）的 2D 片源，确认两眼边缘
  对称、无一侧拉长一侧切短。

## 后续可选项

- `fw_fill` 残留的 `nearmap/h/w` 形参可在后续清理时连同 `_launch` 签名一起删。
- 若日后想在“固定方向”之外再做背景择优，应**在各自眼别允许的方向内**择优，
  而非跨眼共用一条规则。
