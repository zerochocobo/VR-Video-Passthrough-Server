# TensorRT IO Binding Performance and Hang Summary

Date: 2026-05-21

## Conclusion

I cannot currently resolve this to the desired state where TensorRT is stable and faster than the existing CUDA realtime path.

TensorRT cache build, runtime activation, and realtime playback now work. The remaining problem is the core realtime RVM execution path:

- `TensorRTExecutionProvider + RVM IOBinding`: hangs inside `onnxruntime.InferenceSession.run_with_iobinding()`, causing first-chunk timeout and HTTP 504.
- `TensorRTExecutionProvider + normal sess.run()`: stable, but significantly slower than CUDA + IOBinding for 4K alpha realtime playback.
- `CUDAExecutionProvider + RVM IOBinding`: currently the faster production path.

This looks like an ONNX Runtime TensorRT EP / IOBinding / RVM recurrent-state / dynamic-shape interaction issue rather than a normal application bug. It likely needs someone with ONNX Runtime TensorRT EP and TensorRT profile/partitioning expertise.

## Environment

- Windows
- GPU: NVIDIA GeForce RTX 5060 Ti
- Driver: 581.57
- ONNX Runtime: 1.25.1
- Available providers:
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`
- TensorRT Python package: 10.16.1.11
- Model: `models/rvm_mobilenetv3_fp32.onnx`
- TensorRT runtime model:
  - `runtime_cache/trt_engines/rvm_mobilenetv3_fp32_shape_inferred.onnx`
- TensorRT cache:
  - `runtime_cache/trt_engines/manifest.json`
  - usable engine around 10.6 MB

## TensorRT Work Completed

- Added TensorRT UI configuration, build flow, status display, and backend switch.
- Added TensorRT cache manifest handling:
  - fingerprint includes GPU, driver, CUDA runtime, TensorRT, ORT, model sha256, input size, downsample ratio, FP16, CUDA graph.
  - cache states: `missing`, `ready`, `stale`, `failed`.
- Added warmup/build process:
  - creates `rvm_mobilenetv3_fp32_shape_inferred.onnx`.
  - forces TensorRT/CUDA/CPU provider chain.
  - detects ORT provider fallback to prevent false-ready manifests.
  - ready cache requires both shape-inferred ONNX and a real `.engine` larger than 1 MiB.
- Runtime integration:
  - `main.py` switches `config.MODEL_PATH` to the shape-inferred ONNX when TensorRT cache is ready.
  - TensorRT/CUDA DLL path injection works for both development Python and frozen exe.
- RVM dynamic symbol fix:
  - original ONNX reused `height` / `width` symbols for `src` and recurrent state tensors.
  - TensorRT treated recurrent states as having the same H/W as the source frame, causing optimization profile conflicts.
  - warmup now renames recurrent state input/output symbolic dims before shape inference.

## Issue 1: TensorRT + IOBinding Hangs

Observed in `debug_output/server.log`.

During 8K green realtime playback, TensorRT was active:

```text
Matting model loaded ... active=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] ... rvm_iobinding=True
```

The worker then blocked at:

```text
pipeline/matting.py", line 1823, in _run_rvm_iobinding_from_dev
    self.sess.run_with_iobinding(binding)
```

Result:

```text
return 504 first chunk timeout after 30.0s
PyNv runtime marked tainted because worker did not stop
```

Because `run_with_iobinding()` does not return, the Python exception fallback path cannot run.

## Current Workaround

To avoid realtime server hangs:

- CUDA-only runs keep RVM IOBinding enabled.
- TensorRT-active runs disable RVM IOBinding.

Relevant code:

- `pipeline/matting.py`
  - `_should_enable_rvm_iobinding(active_providers)`

Current TensorRT path logs:

```text
active=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
rvm_iobinding=False
```

This workaround is stable but slower.

## Issue 2: TensorRT Normal sess.run Is Slower Than CUDA IOBinding

4K alpha test logs confirm TensorRT was active:

```text
ONNX providers requested=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
trt cache ready; ONNX providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
runtime_model=...\runtime_cache\trt_engines\rvm_mobilenetv3_fp32_shape_inferred.onnx
Matting model loaded ... active=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] ... rvm_iobinding=False
```

4K alpha video:

```text
size=4096x2048
output_mode=alpha
input_shape=(2,3,1024,1024)
```

Performance logs:

```text
frame 300 ... fps=26.53 ... mat_avg_ms pre=4.61 ort=24.76
frame 600 ... fps=26.94 ... mat_avg_ms pre=5.04 ort=27.64
frame 840 ... fps=26.50 ... mat_avg_ms pre=5.22 ort=27.47
```

Interpretation:

- decode / encode / mux are not the main bottleneck.
- RVM matting inference is the bottleneck.
- TensorRT normal `sess.run()` has `ort_run` around 24-27ms.
- With preprocessing, matting is near 30ms per frame, limiting realtime throughput to about 26-27 FPS.

Historical CUDA + IOBinding logs show an 8K green path with:

```text
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
rvm_iobinding=True
frame=8192x4096 input_shape=(2,3,1024,1024)
ort_run=18-19ms
```

This is not a perfectly controlled alpha-vs-alpha A/B run, but it is enough to show that the current TensorRT normal `sess.run()` path is not faster.

## Recommended Expert Focus

1. Whether ORT TensorRT EP supports the current RVM multi-input, multi-output, recurrent-state CUDA IOBinding pattern.
2. Whether the `run_with_iobinding()` hang is caused by:
   - TensorRT EP bug;
   - output binding shape/type mismatch;
   - recurrent OrtValue lifetime/ownership issue;
   - CUDA stream/synchronization issue;
   - mixed TensorRT/CUDA EP execution after graph partition fallback.
3. Whether ORT TensorRT provider options need changes.
4. Whether RVM recurrent state management should be rewritten to be more TensorRT-friendly.
5. Whether a more static-shape RVM ONNX export can avoid problematic Resize/dynamic-shape partitions.
6. Whether direct TensorRT engine inference should replace ORT TensorRT EP for this path.

## Current Recommendation

Until `TensorRT + IOBinding` is solved, TensorRT should not be the default realtime backend.

Recommended behavior:

- Default realtime playback: CUDA + RVM IOBinding.
- TensorRT: keep as experimental/manual backend.
- UI should indicate that TensorRT may not improve realtime FPS yet.

## Relevant Files

- `pipeline/matting.py`
- `ui/services/trt_warmup_process.py`
- `utils/trt_manifest.py`
- `utils/runtime_dll_paths.py`
- `main.py`
- `ui/settings.py`
- `ui/pages/home_page.py`
- `tests/test_matting_runtime_policy.py`
- `tests/test_trt_warmup_process.py`
- `tests/test_runtime_dll_paths.py`

## Tests Passed

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_exe.py tests/test_runtime_dll_paths.py tests/test_matting_runtime_policy.py tests/test_trt_manifest.py tests/test_trt_warmup_process.py tests/test_settings.py tests/test_i18n.py tests/test_ui_smoke.py tests/test_main_args.py -q
```

Result:

```text
31 passed, 7 subtests passed
```
