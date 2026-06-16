# CuPy sm_120 (Blackwell) NVRTC 升级与副作用研究（2026-06-15）

目标：让 CuPy 在 RTX 50 系（sm_120）上正常工作并消除冷启动 JIT 编译，且 `uv sync` 可复现。
仅改 `pyproject.toml`（加一条 pip NVRTC 依赖），不改 CuPy 版本。

## 结论（TL;DR）

- 加一条依赖：`nvidia-cuda-nvrtc-cu12>=12.8,<12.9`。
- 这给 CuPy 一个 **sm_120 原生的 NVRTC（12.8）**，RawKernel/ElementwiseKernel 直接编译成
  sm_120 cubin，**不再走 PTX→驱动 JIT 冷启动**（用户抱怨的 "cuda_compute_cache 冷启动编译"）。
- **不需要升级系统 CUDA 到 12.8**（pip NVRTC 自带，`uv sync` 可复现，与系统 CUDA 解耦）；
  系统装 12.8 也无害（冗余）。
- **不动 CuPy 版本**（保持 14.0.1）：项目管线只用 RawKernel/ElementwiseKernel，不用 cccl 算子，
  所以 CuPy 14 的 cccl 头文件 bug 不影响本项目。
- ONNX Runtime / TensorRT / 离线 2D→VR 管线均无回归（已验证）。

## 背景：为什么 12.6 不行

CuPy 的 RawKernel/ElementwiseKernel 在运行时用 **NVRTC** 把核函数源码编译成机器码：

- **直接 cubin 模式（默认）**：NVRTC 直接生成目标架构 cubin。CUDA 12.6 的 NVRTC 不认识
  sm_120 → 生成的 cubin 架构不匹配 → 加载时 `CUDA_ERROR_NO_BINARY_FOR_GPU`。
- **PTX 模式（`CUPY_COMPILE_WITH_PTX=1`）**：NVRTC 生成架构无关的 PTX，由 NVIDIA 驱动在首次
  使用时 JIT 成 sm_120 cubin 并缓存（驱动 compute cache）。这是项目 2026-05-14 既有的兼容策略
  （见 `summary_20260514_cupy_cuda_packaging_dependencies_CN.md`，frozen 构建里由
  `packaging/runtime_hook_cuda_dlls.py` 设置）。它**能用但有冷启动 JIT**。

用户要消除冷启动 → 需要 NVRTC 能直接出 sm_120 cubin → **NVRTC ≥ 12.8**。

实测（当前 cupy 14.0.1）：
- 系统 NVRTC 12.6：`cp.arange` 直接 NO_BINARY；PTX 模式可用（验证 sum 正确）。
- pip NVRTC 12.8：`cp.arange` / RawKernel / ElementwiseKernel **直接 cubin 全部 OK**（无 PTX 环境变量）。

## 为什么不升级 / 降级 CuPy

测试矩阵（sm_120，map_coordinates/percentile 等 cccl 算子）：

| CuPy | NVRTC | 基础核(RawKernel/EW) | cccl 核(percentile/ndimage) |
|---|---|---|---|
| 14.0.1 | 系统 12.6 | NO_BINARY | NO_BINARY |
| 14.0.1 | pip 12.9 | **OK** | FAIL：`__nv_fp8_e8m0` 头文件不完整 |
| 14.0.1 | pip 12.8 | **OK** | FAIL：同上 fp8 头文件 bug |
| 13.6.0 | （未用 pip）系统 12.6 | FAIL：找不到 nvrtc-builtins64_126 | — |

- **CuPy 14.0.1 的 cccl bug**：其 vendored `libcudacxx` 头在用 NVRTC 编译 thrust/cccl 核
  （percentile、sort、`cupyx.scipy.ndimage.map_coordinates`、`gaussian_filter`）时报
  `incomplete type "__nv_fp8_e8m0"`，**与 NVRTC 12.8/12.9 都冲突**（fp8 是 CUDA 12.8 引入的类型）。
- **关键**：`grep` 全项目 `pipeline/ offline/ utils/ tools/`，**没有任何 cccl 算子使用**
  （无 `cp.percentile/sort/argsort/median`，无 `cupyx.scipy.ndimage`）。管线只用
  RawKernel/ElementwiseKernel/基础算术/fancy-index —— 这些在 NVRTC 12.8 下全部正常。所以
  CuPy 14 的 bug 对**本项目无影响**，不必降级。
- CuPy 13.6 没有该 bug，但它的 NVRTC 加载器没采用 pip 包、回退到系统 12.6 并触发
  builtins 版本错配 —— 在不升级系统 CUDA 的前提下不可用。故**保持 CuPy 14 + pip NVRTC 12.8** 最稳。

- **为什么 pin 12.8 而非 12.9**：两者对项目用到的基础核都可用；但 12.9 会让 CuPy 14 的 cccl 核
  100% 编译失败（fp8），万一以后有人用到 cccl 算子会踩雷。12.8 是安全下限，也正是用户提到的版本。

## 副作用 / 需要注意

1. **ONNX Runtime / TensorRT**：无影响。它们用各自自带的 CUDA 组件；pip NVRTC 由 CuPy 的
   `cuda-pathfinder` 进程内 `add_dll_directory` 定位，不进全局 PATH。已验证 DA3 深度仍走
   `TensorrtExecutionProvider`，输出正确，无回归。
2. **`apply_runtime_dll_paths` 把系统 CUDA_PATH\bin（12.6）前置到 PATH**：不影响 CuPy ——
   `cuda-pathfinder` 优先选 pip NVRTC（实测加载到 12.8 而非系统 12.6）。
3. **numpy**：CuPy 14 要求 `numpy>=2.0`；项目 pin `<2.1` → 解析到 2.0.x。本次未变。
4. **冻结打包（PyInstaller）⚠️ 需后续处理**：`build_exe.bat` 目前从**系统 CUDA** 拷贝
   `nvrtc-builtins*.dll` 并设 `CUPY_COMPILE_WITH_PTX=1`。引入 pip NVRTC 后，更优做法是把
   `site-packages/nvidia/cuda_nvrtc/bin` 的 `nvrtc64_120_0.dll` + `nvrtc-builtins64_128.dll`
   打进发行包，并可去掉 PTX 模式以获得 sm_120 原生无冷启动。**`build_exe.bat` /
   `packaging/hooks/hook-cupy.py` / `packaging/runtime_hook_cuda_dlls.py` 待更新**（本次只改了
   pyproject，未动打包脚本）。
5. **冷启动的残留**：直接 cubin 省掉了**驱动 PTX JIT**；CuPy 首次仍会用 NVRTC 编译一次核并落
   `~/.cupy/kernel_cache`（更快，且非驱动 compute cache）。用户抱怨的驱动级冷启动已消除。
6. **离线 2D→VR 工具本身不用 CuPy**（纯 cv2/numpy + ORT），无论如何都不受影响。
7. 磁盘：pip NVRTC wheel +~30MB。

## 验证

- `uv sync` 干净（移除了 cupy13 实验残留的 `fastrlock`，69 包匹配 lock）。
- CuPy sm_120 直接 cubin（无 PTX 环境变量）：`cp.arange`、RawKernel、ElementwiseKernel 全 OK，
  `nvrtc=(12,8)`，`cc=120`。
- DA3 深度：`provider=TensorrtExecutionProvider`，shape 正确，无回归。
- 离线 2D→VR 端到端：1280 档 42fps，正常出片。

## 改动文件

- `pyproject.toml`：新增 `nvidia-cuda-nvrtc-cu12>=12.8,<12.9`（含说明注释）。
- `uv.lock`：`uv add` + `uv sync` 自动更新。

## 加固：venv 自包含 CUDA（取代单条 NVRTC 方案）

用户确认后，把方案升级为**整套 pip CUDA 12.x 运行时自包含**（不再只加 NVRTC）：

`pyproject.toml`：
- `cupy-cuda12x` → **`cupy-cuda12x[ctk]`**：`[ctk]` extra 拉入 pip CUDA 工具链
  （`cuda-toolkit==12.9.2.0` → nvrtc/cublas/cudart/cufft/curand/cusolver/cusparse + 头文件）。
- 新增 **`nvidia-cudnn-cu12>=9.7,<10`**（ONNX Runtime CUDA EP 需要 cuDNN 9；不在 ctk 里）。
- 移除之前单独的 `nvidia-cuda-nvrtc-cu12>=12.8,<12.9`（ctk 已带 nvrtc 12.9；12.8/12.9 对 cupy14
  cccl 都坏，对项目都无影响，故不必卡 12.8）。

`uv sync` 拉入：cuda-toolkit 12.9.2.0、nvidia-cublas/cudart/cufft/curand/cusolver/cusparse-cu12
（12.9.x）、nvidia-cudnn-cu12 9.23、nvjitlink、nvrtc 12.9.86。约 +1.5GB。

DLL 与头文件落点（供打包）：
- `site-packages/nvidia/<comp>/bin/*.dll`（24 个：cublas/cublasLt/cudart/cufft/curand/cusolver/
  cusparse/nvrtc/nvrtc-builtins64_129/nvJitLink/cudnn*）。
- `site-packages/nvidia/cuda_runtime/include/`（84 个头，含 cuda_fp16/bf16/runtime/device_*/
  vector_types —— cupy RawKernel JIT 所需齐全；系统 12.6 的 133 个里多出来的是 cublas/cufft 等
  库头，项目用不到）。

验证（dev）：cupy 14 + nvrtc 12.9 的 **fp16 RawKernel 在 sm_120 编译运行 OK**（找到 CUDA 头），
DA3 深度仍走 TensorRT 无回归。

## 打包脚本改动（`build_exe.py` + runtime hook）

目标：frozen 包用 **pip 的 sm_120 原生 CUDA 12.9**，不依赖构建机/用户机的系统 CUDA，且去掉冷启动。

- `build_exe.py`：
  - 新增 `copy_pip_cuda_runtime_dlls()`：把 `site-packages/nvidia/*/bin/*.dll` 复制进 `_internal`，
    **在所有 CUDA 拷贝之后执行**，覆盖 PyInstaller 从系统工具链收来的同名 DLL（如 cublasLt64_12.dll）。
    带必需项校验（cudart64_12 / cublasLt64_12 / nvrtc64_120_0）。
  - `copy_cuda_headers()`：CUDA 头优先取 pip `nvidia/cuda_runtime/include`，无则回退系统 CUDA_PATH。
  - 在 `main()` 的 EP 依赖拷贝之后调用新函数。
  - 既有 `verify_cuda_auxiliary_runtime`（nvrtc-builtins + cuda_fp16.h）/`verify_ort_cuda_ep_runtime`
    （cudnn64_9.dll）均由 pip 满足，无需改。
- `packaging/runtime_hook_cuda_dlls.py`：`CUPY_COMPILE_WITH_PTX` 默认 `1` → **`0`**。捆绑了
  sm_120 原生 nvrtc 后走直接 cubin，**消除 PTX→驱动 JIT 冷启动**；仍可用环境变量覆盖回 1。
  （runtime hook 早已把 `_internal/nvidia/**/bin` 加入 DLL 搜索路径，pip DLL 能被找到。）

⚠️ **未在本会话跑完整 `python build_exe.py`**（耗时长）。改动已 `py_compile` 通过、pip DLL/头清单已核
对齐全。**请实际跑一次构建**，它末尾的 `run_frozen_probe`（cuda.pathfinder canary）+ 建议再启动
一次 `pt_core.exe` 冒烟（CuPy import + ORT CUDA + 一次离线任务）确认无回归。

## 后续/未决

- 若将来 2D→VR 想做 CuPy GPU 渲染并用到 `map_coordinates` 等 cccl 算子：需等 CuPy 14.x 修复 fp8 头
  bug，或换用不依赖 cccl 的自写 RawKernel/ElementwiseKernel（flat3d inverse_warp 可纯基础算子实现）。
- 关于系统 CUDA 13.0：本项目整套是 cu12 ABI，**升级系统到 13 无收益且有风险**（SONAME 变 _13、13 砍了
  老架构）；保留一个 12.x 工具链或如本次走 pip 自包含即可。
