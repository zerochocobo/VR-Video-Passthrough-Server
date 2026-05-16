> ⚠️ SUPERSEDED 2026-05-16
> FPS, sync, and bottleneck-attribution conclusions in this report were produced under the old default PT_PASSTHROUGH_MAX_FPS=30 and/or non-target PT_ALPHA_STRIDE=3 conditions. Later review invalidated them or limited them to those diagnostic conditions only.
> Use summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_EN.md as the corrected baseline entry point.
> This file is retained only as a research-process archive; implementation conclusions unrelated to the FPS cap must be read with their stated scope.
# Stage 4R/P0 Summary - Profiling Findings and TensorRT Blocker

## Background

This stage followed the external review and prioritized three open questions:

1. How much of the `composite` time is RVM inference versus the green-screen composite kernel.
2. What the `cuda_stream.synchronize()` wait is actually waiting for.
3. Whether TensorRT EP can be promoted directly to the next P1 optimization.

No additional Python pipeline staging was added, and the crash-prone ThreadedDecoder staged design remains disabled.

## Code Changes

- `pipeline/pynv_stream.py`
  - Added `mat_avg_ms pre/ort/kernel` and `mat_max_ms pre/ort/kernel` to production diagnostics.

- `tools/auto_tune_8k.py`
  - Parser now accepts the new matting sub-stage fields.
  - Reports now include `Latest mat avg ms`.
  - Added `--server-env KEY=VALUE` so experiments can pass explicit environment overrides to the spawned server process and record them in the report.

- `config.py`
  - Added TensorRT EP settings:
    - `PT_ONNX_TRT_ENGINE_CACHE_ENABLE`
    - `PT_ONNX_TRT_ENGINE_CACHE_PATH`
    - `PT_ONNX_TRT_FP16_ENABLE`
    - `PT_ONNX_TRT_CUDA_GRAPH_ENABLE`
  - Added diagnostic switch `PT_PASSTHROUGH_PYNV_SYNC_PROBE`.

- `pipeline/matting.py`
  - Supplies TensorRT FP16, engine cache, and CUDA Graph provider options when `TensorrtExecutionProvider` is selected.
  - Logs ONNX providers as wanted / available / selected.
  - With `PT_PASSTHROUGH_PYNV_SYNC_PROBE=1`, splits PyNv green GPU synchronization into upload, RVM/alpha, and composite segments.

## P0 Tests and Findings

### 1. 60s / Full-file Green Baseline

Reports:

- `baseline/auto_tune_8k_phase1_20260516_154731.md`
- `baseline/auto_tune_8k_phase1_20260516_154731.json`

The requested duration was 60s, but `videos/test_8k_2.mp4` is about 26s, so the run completed the full 785-frame file.

Results:

- HTTP status: `200`
- first byte: `2.765 s`
- latest interval FPS: `36.40`
- average interval FPS: `36.13`
- stage avg:
  - decode: `0.05 ms`
  - composite: `15.98 ms`
  - sync: `11.00 ms`
  - encode: `0.40 ms`
  - mux: `0.02 ms`
- mat avg:
  - preprocess: `0.11 ms`
  - ORT/RVM: `15.17 ms`
  - kernel: `0.62 ms`

Conclusion:

- The main `composite` cost is RVM/ORT inference, not the custom green composite kernel.
- The green composite kernel is about `0.5-0.6 ms`, so a custom composite rewrite is not a near-term priority.

### 2. Nsight Systems Attempt

Nsight Systems CLI exists at:

- `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe`

Generated files:

- `baseline/nsys_stage4r_green_20260516_154929.nsys-rep`
- `baseline/nsys_stage4r_green_20260516_154929.sqlite`

But this trace is not useful for CUDA timeline analysis:

- It profiled the `auto_tune_8k.py` parent process, while CUDA work happens in the spawned server process.
- `nsys stats` only showed minimal parent-process CUDA API activity.
- No CUDA kernel, GPU memory, NVTX, or NVVIDEO data was captured from the real worker.

Conclusion:

- The next Nsight attempt must profile or attach to the actual server process.

### 3. `PT_PASSTHROUGH_PYNV_SYNC_PROBE=1` Short Run

Command:

```powershell
$env:PT_PASSTHROUGH_PYNV_SYNC_PROBE='1'
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 6 --startup-timeout 240 --client-timeout 120
```

Reports:

- `baseline/auto_tune_8k_phase1_20260516_155457.md`
- `baseline/auto_tune_8k_phase1_20260516_155457.json`

Results:

- latest interval FPS: `36.46`
- stage avg:
  - decode: `0.07 ms`
  - composite: `26.35 ms`
  - sync: `0.01 ms`
  - encode: `0.45 ms`
  - mux: `0.54 ms`
- mat avg:
  - preprocess: `0.05 ms`
  - ORT/RVM: `7.22 ms`
  - kernel: `18.98 ms`

Example diagnostic:

```text
[DIAG] pynv sync probe nv12 frame=240 upload_sync=20.74ms alpha_call=0.03ms alpha_tail_sync=0.02ms composite_sync=1.64ms
```

Interpretation:

- The outer `sync` drops from about `10-11 ms` to `0.01 ms`, proving the normal outer sync is waiting for previously enqueued GPU work.
- This diagnostic mode changes attribution by moving waits into the inner upload/composite regions.
- The large `mat_kernel=18.98 ms` in this run must not be read as real kernel compute cost. It is synchronization wait moved into that timing bucket.
- Non-inference frames also showed `upload_sync` waits around `20-25 ms`, suggesting PyNv/ThreadedDecoder GPU production or cross-stream visibility is part of the wait.

Conclusion:

- Moving the synchronize point alone will not increase single-session FPS.
- A correct Nsight trace of the server process is still required to prove the exact GPU overlap and wait source.

## TensorRT EP P1 Pre-check

Local provider check:

```powershell
@'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
'@ | .venv\Scripts\python.exe -
```

Result:

- ORT: `1.25.1`
- available providers:
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`

TRT auto_tune command:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 `
  --video videos\test_8k_2.mp4 `
  --profile quest `
  --prefer green `
  --duration 4 `
  --startup-timeout 900 `
  --client-timeout 240 `
  --server-env PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider `
  --server-env PT_ONNX_TRT_ENGINE_CACHE_ENABLE=1 `
  --server-env PT_ONNX_TRT_FP16_ENABLE=1 `
  --server-env PT_ONNX_TRT_CUDA_GRAPH_ENABLE=1
```

Reports:

- `baseline/auto_tune_8k_phase1_20260516_155825.md`
- `baseline/auto_tune_8k_phase1_20260516_155825.json`

Results:

- latest interval FPS: `36.53`
- mat ORT/RVM: `15.21 ms`
- active providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

Key log:

```text
[DIAG] ONNX providers wanted=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] available=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] selected=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
*************** EP Error ***************
Please install TensorRT libraries as mentioned in the GPU requirements page, make sure they're in the PATH or LD_LIBRARY_PATH, and that your GPU is supported.
```

Conclusion:

- The ORT package exposes TensorRT EP.
- The server process receives the TRT provider override correctly.
- ORT fails to initialize TensorRT EP because TensorRT native libraries are missing from PATH or not installed.
- The session falls back to CUDA EP, so the current TRT run is not a real TRT performance result.
- No TensorRT engine cache was generated.

## Judgement Against the Expert Advice

- P0 profiling: partially complete.
  - RVM/ORT is now proven to dominate `composite`.
  - The green composite kernel is not a near-term target.
  - Nsight still needs a correct server-process capture.

- P1 TensorRT EP: correct direction, currently environment-blocked.
  - Runtime switches and provider options are ready.
  - Native TensorRT libraries must be installed or added to PATH first.

- P2 CUDA Graph: deferred.
  - TRT is not active yet.
  - CUDA EP graph capture would require fixed IOBinding pointers and carries more implementation risk.

- P3 custom composite kernel: not recommended now.
  - Kernel cost is only about `0.5-0.6 ms` in steady CUDA EP runs.

- P4 encode/composite event chain: lower priority.
  - It may reduce CPU blocking, but current single-session FPS remains GPU-wait bound.

## Risks and Blockers

- TensorRT runtime DLLs are missing or not on PATH.
- Nsight must capture the real server process, not the auto_tune parent process.
- `PT_PASSTHROUGH_PYNV_SYNC_PROBE=1` is diagnostic only and changes timing attribution.
- `DEBUG_LOGS=1` emits very verbose composite logs in long runs.
- `tools/pynv_fullchain_probe.py --decoder threaded` must remain disabled because the staged ThreadedDecoder design has known native crash risk.

## Recommended Next Stage

1. Fix the TensorRT runtime environment:
   - install a TensorRT version compatible with the current ORT/CUDA stack;
   - or add the existing TensorRT runtime DLL directory to the spawned server PATH.

2. Re-run the TRT command above and require active providers to include `TensorrtExecutionProvider`.

3. If TRT becomes active and engine cache files appear, run a full-file green baseline and compare against `baseline/auto_tune_8k_phase1_20260516_154731.md`.

4. Prepare a direct Nsight profile of the server process to answer the remaining overlap/wait-source question.

5. Do not add more Python staged pipeline work until TRT or correct Nsight evidence changes the bottleneck picture.

## Simple Verification Commands

Compile:

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\matting.py pipeline\pynv_stream.py tools\auto_tune_8k.py
```

Check ORT providers:

```powershell
@'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
'@ | .venv\Scripts\python.exe -
```

CUDA EP green baseline:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 10 --startup-timeout 240 --client-timeout 120
```

TRT environment retest:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 4 --startup-timeout 900 --client-timeout 240 --server-env PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider --server-env PT_ONNX_TRT_ENGINE_CACHE_ENABLE=1 --server-env PT_ONNX_TRT_FP16_ENABLE=1 --server-env PT_ONNX_TRT_CUDA_GRAPH_ENABLE=1
```
