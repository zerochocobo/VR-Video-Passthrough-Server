# CuPy/CUDA Packaging Dependencies Summary

Date: 2026-05-14

## Scope

Completed the packaging dependency fixes required for CuPy, CUDA/NVRTC, PyNvVideoCodec, and CUDA path isolation in the frozen Windows distribution.

## Problems Found

- CuPy failed during import because the packaged app did not provide the `CUDA_PATH\bin` directory CuPy probes on Windows.
- CuPy Cython extension modules such as `cupy._core._carray` were missed by PyInstaller.
- CuPy/NVRTC header lookup launched `cuda.pathfinder` subprocess probes, but the frozen `pt_core.exe` initially treated those module invocations as application CLI arguments.
- `cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess` was not reliably included as a hidden import.
- RawKernel JIT compilation required `nvrtc-builtins*.dll` and CUDA headers.
- On Blackwell GPUs such as RTX 5060 Ti (`sm_120`), CUDA 12.6 NVRTC could emit incompatible cubin output, causing `CUDA_ERROR_NO_BINARY_FOR_GPU`.
- PyNvVideoCodec dynamically selects driver capability extensions such as `PyNvVideoCodec_121` or `PyNvVideoCodec_130`, but those dynamic `.pyd` files were not reliably present in the final package.
- User machines may have stale or broken system `CUDA_PATH`, which can interfere with the bundled runtime.

## Implemented Fixes

- Added `packaging/runtime_hook_cuda_dlls.py` and wired it into both PyInstaller builds.
- The runtime hook now sets packaged-process `CUDA_PATH` and `CUDA_HOME` to the distribution root, preserving original values as `PT_ORIGINAL_CUDA_PATH` and `PT_ORIGINAL_CUDA_HOME`.
- The runtime hook prepends packaged CUDA DLL directories before user/system CUDA paths.
- Added default `CUPY_COMPILE_WITH_PTX=1` to improve cross-generation GPU compatibility.
- Added `packaging/hooks/hook-cupy.py` to explicitly collect CuPy and CuPy backend `.pyd` files.
- Added hidden import coverage for `cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess`.
- Updated `main.py` to forward internal module-style subprocess invocations through `runpy.run_module()` before normal CLI parsing.
- Updated `build_exe.bat` to create `bin\.keep`, copy `nvrtc-builtins*.dll`, copy CUDA headers into `include`, and validate critical packaged files.
- Updated packaging validation for CuPy `_carray`, PyNvVideoCodec dynamic extensions, CUDA headers, NVRTC builtins, and the frozen `cuda.pathfinder` subprocess probe.

## Compatibility Strategy

- Packaged releases use bundled CUDA/NVRTC components first.
- CuPy RawKernel uses PTX by default so the end user’s NVIDIA driver can JIT for the actual GPU architecture.
- Future GPUs require sufficiently new NVIDIA drivers.
- Older GPUs still depend on CUDA, ONNX Runtime, NVENC/HEVC support, and enough VRAM.

## Verification Notes

- The main server startup reached FastAPI/Uvicorn successfully after the packaging fixes.
- CuPy import, ONNX Runtime CUDA provider initialization, and RVM warmup progressed past earlier import and path failures.
- Remaining GPU compatibility is expected to depend primarily on driver support and hardware capability rather than missing packaged Python modules.

## Files Changed

- `build_exe.bat`
- `main.py`
- `packaging/runtime_hook_cuda_dlls.py`
- `packaging/hooks/hook-cupy.py`
- `utils/gpu_runtime_cache.py`

