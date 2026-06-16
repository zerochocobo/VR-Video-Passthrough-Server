# 2DVR soft_shift 头发横向条带研究与审核说明

## 背景

接手的问题来自 `summary/summary_20260616_2DVR_SOFTSHIFT_HAIR_SMEAR_RESEARCH_BRIEF.md`：2D 转 3D 的 `soft_shift` 在头发附近仍有横向涂抹/横纹，尤其是 `videos/test_4k2d_S000030_E000100_3D_LR_Screen.mp4` 输出片段 3-12s 的左眼。

用户后来确认重新生成了该文件，之前看的文件并非真正 `soft_shift`。因此本轮重新基于 2026-06-16 09:54 生成的 soft_shift 成品做判断。

## 现象复核

输出 t=5/7/10/12s 左眼头部左侧有明显宽横向条带。它不是单纯的“头发被填进空洞”，而是 `soft_shift` 当前按行填洞策略导致的背景纹理拖拽：

- forward warp 后 `zbuf==0` 形成大块 disocclusion hole；
- `fw_fill` 对左眼按行从左侧最近有效像素复制；
- 墙面/砖纹每行亮度不同，被横向复制进整块空洞；
- 结果在头发左侧形成稳定的横向 row-copy band。

相关调试图：

- `debug_output/2dvr_hair/regen/soft_regen_left_eye_sheet_t03_t12.png`
- `debug_output/2dvr_hair/regen/direct_t37_hybrid_no_new_kernel_visual.png`
- `debug_output/2dvr_hair/regen/direct_t37_softshift_buffers.npz`

## 尝试过的方案

1. 调大现有 `fw_blend` 的窗口和 alpha。

结果：只能轻微缓解，无法消除宽横向条带。原因是问题源头在 `fw_fill` 已经把背景按行复制成横带，后续小范围 feather 不足以恢复合理背景。

2. 对填洞来源列做纵向平滑。

结果：CPU 原型有效，V12/V16/V24 都能降低条带。但 GPU 实现需要新增 kernel 或大改现有 kernel，遇到了 CuPy 新 kernel 编译卡住问题，暂未落地。

3. `soft_shift` 非空洞保留，只把 `zbuf==0` 空洞像素替换为 `inverse_warp` 输出。

结果：当前最优视觉方向。`inverse_warp` 单独使用会有“砍人像/边缘撕裂”问题，但仅用于 soft_shift 的空洞像素时，可以去掉宽横向条带，同时保留 soft_shift 的主体遮挡和轮廓。

`tests/test_two_dvr_hybrid_hole_fill.py` 固化了该语义：非空洞像素必须 byte-for-byte 保留 soft_shift，只有 `zbuf==0` 的位置来自 inverse_warp。

## 性能与实现状态

host 侧验证单帧 1920x1080：

- soft_shift GPU render + download: 约 13.5ms
- inverse_warp GPU render + download: 约 7.7ms
- CPU merge: 约 17.3ms
- 空洞占 SBS 像素约 2.12%

这证明算法方向有效，但 CPU merge 不适合实时 GPU-resident 路径。生产实现需要 GPU merge：每个像素判断 `zbuf==0`，从 inverse 或 soft 结果拷贝 RGB。

## 编译卡住问题

本轮没有把 GPU merge kernel 写入生产代码，原因是当前环境中新 CuPy kernel 编译不可控：

- 合法的一行 noop RawKernel / RawModule 都会在首次 launch 或 `get_function()` 阶段超过 60-120 秒；
- 设置 `CUPY_CACHE_DIR=runtime_cache/cupy` 无效；
- `CUPY_COMPILE_WITH_PTX=0/1` 都无效；
- 没有发现 CuPy lock 文件；
- 挂起的测试 python 进程已清理；
- 已缓存的现有 kernel 很快，例如 `_SOFT_SHIFT_KERNELS`、`_SBS_INV_WARP_KERNEL`；
- GPU 环境：RTX 5060 Ti, sm_120, CuPy 14.0.1, CUDA runtime 12.9, driver 13.0, `CUDA_PATH` 指向 CUDA 12.6。

初步判断：不是网络、DA3、uv 下载或视频处理卡住，而是 sm_120 + 当前 CuPy/NVRTC 新 kernel 编译路径异常慢或卡住。审核实现前需要先解决这一点，或者把 merge kernel 纳入稳定的预编译/预热缓存流程。

## 建议审核方向

推荐审核的算法方案：

1. soft_shift 仍负责 forward warp、z-buffer、主体遮挡关系和硬轮廓；
2. inverse_warp 额外生成一张 SBS；
3. 对 soft_shift 的 `zbuf==0` hole 像素，用 inverse_warp 对应像素替换；
4. 再使用现有 foreground-excluded feather 只处理 hole seam。

推荐审核的工程方案：

- 不要用 CuPy boolean indexing 或 `cp.where` 做实时 merge；测试中这些路径也触发了长时间卡住/慢路径。
- 应使用一个简单 merge kernel，但必须先解决新 CuPy kernel 编译卡住，或将该 kernel 放入可靠 warmup/cache 流程。
- 在生产代码落地前，先做 3-12s 左眼局部视频对比，并确认右眼没有引入新的 stereo 不一致。

## 当前代码状态

本轮没有修改生产渲染代码。新增的是审核辅助：

- `tests/test_two_dvr_hybrid_hole_fill.py`
- `summary/summary_20260616_2DVR_SOFTSHIFT_HAIR_HOLE_FILL_REVIEW_CN.md`

验证：

- `uv run pytest tests/test_two_dvr_hybrid_hole_fill.py`
- `uv run pytest tests/test_offline_outputs.py`
