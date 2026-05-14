# CuPy/CUDA 打包依赖修复小结

日期：2026-05-14

## 范围

完成 Windows 冻结包中 CuPy、CUDA/NVRTC、PyNvVideoCodec 以及 CUDA 路径隔离相关的打包依赖修复。

## 发现的问题

- CuPy 在 Windows 上会探测 `CUDA_PATH\bin`，打包目录缺少该目录时会在 import 阶段失败。
- PyInstaller 漏收了 `cupy._core._carray` 等 CuPy Cython 扩展模块。
- CuPy/NVRTC 查找头文件时会通过 `cuda.pathfinder` 启动子进程探测，但冻结后的 `pt_core.exe` 一开始把这些模块调用误当成业务 CLI 参数。
- `cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess` 未被稳定纳入 hidden import。
- RawKernel JIT 编译需要 `nvrtc-builtins*.dll` 和 CUDA headers。
- RTX 5060 Ti 等 Blackwell GPU（`sm_120`）上，CUDA 12.6 NVRTC 直接生成 cubin 时可能不适配目标 GPU，导致 `CUDA_ERROR_NO_BINARY_FOR_GPU`。
- PyNvVideoCodec 会按驱动能力动态选择 `PyNvVideoCodec_121` 或 `PyNvVideoCodec_130`，这些动态 `.pyd` 文件之前未稳定进入最终包。
- 最终用户电脑可能存在旧版、损坏或不匹配的系统 `CUDA_PATH`，会污染打包程序运行环境。

## 已实现修复

- 新增 `packaging/runtime_hook_cuda_dlls.py`，并接入 UI 和 server 两个 PyInstaller 构建。
- runtime hook 在冻结程序中将 `CUDA_PATH` 和 `CUDA_HOME` 指向发行目录根路径，并把原始值保存到 `PT_ORIGINAL_CUDA_PATH` 和 `PT_ORIGINAL_CUDA_HOME`。
- runtime hook 将随包 CUDA DLL 路径优先加入 DLL 搜索路径，避免优先使用用户系统 CUDA。
- 默认设置 `CUPY_COMPILE_WITH_PTX=1`，提高跨代 GPU 兼容性。
- 新增 `packaging/hooks/hook-cupy.py`，显式收集 CuPy 和 CuPy backend 的 `.pyd` 文件。
- 显式加入 `cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess` hidden import。
- 修改 `main.py`，在正常 CLI 解析前转发内部模块式子进程调用到 `runpy.run_module()`。
- 更新 `build_exe.bat`，创建 `bin\.keep`，复制 `nvrtc-builtins*.dll`，复制 CUDA headers 到 `include`，并增加关键文件校验。
- 增加 CuPy `_carray`、PyNvVideoCodec 动态扩展、CUDA headers、NVRTC builtins、冻结版 `cuda.pathfinder` 子进程 probe 的构建后校验。

## 兼容策略

- 发布包优先使用随包 CUDA/NVRTC 组件。
- CuPy RawKernel 默认使用 PTX，由最终用户机器上的 NVIDIA 驱动 JIT 到实际 GPU 架构。
- 未来新显卡依赖足够新的 NVIDIA 驱动。
- 老显卡仍受 CUDA、ONNX Runtime、NVENC/HEVC 支持和显存容量限制。

## 验证情况

- 修复后主 server 已能进入 FastAPI/Uvicorn 启动阶段。
- CuPy import、ONNX Runtime CUDA Provider 初始化、RVM warmup 已越过早期的 import 和路径错误。
- 后续 GPU 兼容性主要取决于用户驱动版本和硬件能力，而不是缺少打包的 Python 模块。

## 涉及文件

- `build_exe.bat`
- `main.py`
- `packaging/runtime_hook_cuda_dlls.py`
- `packaging/hooks/hook-cupy.py`
- `utils/gpu_runtime_cache.py`

