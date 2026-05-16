> ⚠️ SUPERSEDED 2026-05-16
> FPS, sync, and bottleneck-attribution conclusions in this report were produced under the old default PT_PASSTHROUGH_MAX_FPS=30 and/or non-target PT_ALPHA_STRIDE=3 conditions. Later review invalidated them or limited them to those diagnostic conditions only.
> Use summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_EN.md as the corrected baseline entry point.
> This file is retained only as a research-process archive; implementation conclusions unrelated to the FPS cap must be read with their stated scope.
# Stage 4R Summary - RVM FP16 Model Baseline

## Background

Following the external review, this stage did not use DirectML and did not promote TensorRT as the default path. The goal was to measure the existing RVM FP16 ONNX model on the current ORT CUDA EP + IOBinding pipeline.

## Scope Correction

The original FP16 baselines in this file used the project default `PT_ALPHA_STRIDE=3` unless the command explicitly set `PT_ALPHA_STRIDE=1`.

Therefore the previous `36-37fps` conclusions only apply to stride=3 diagnostics. They do not validate the original target of 8K 40fps with `ALPHA_STRIDE=1`.

Model tested:

- `models/rvm_mobilenetv3_fp16.onnx`

Confirmed model types:

- `src`: `tensor(float16)`
- `r1i/r2i/r3i/r4i`: `tensor(float16)`
- `downsample_ratio`: `tensor(float)`
- `pha/fgr/r1o-r4o`: `tensor(float16)`

## Commands

Short sanity run:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 8 --startup-timeout 240 --client-timeout 120 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx
```

60s baseline:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 180 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx
```

Notes:

- `videos/test_8k_2.mp4` is about 26s and is useful for quick sanity checks.
- `videos/test_8k.mp4` is 8192x4096, about 60.06s, 59.94fps, and is the better steady-state baseline sample.

## Results

### FP16 Short Run: test_8k_2.mp4

Reports:

- `baseline/auto_tune_8k_phase1_20260516_162123.md`
- `baseline/auto_tune_8k_phase1_20260516_162123.json`

Results:

- active providers: `CUDAExecutionProvider`, `CPUExecutionProvider`
- model input type: `tensor(float16)`
- latest interval FPS: `36.90`
- average interval FPS: `35.60`
- stage avg:
  - decode: `0.07 ms`
  - composite: `14.02 ms`
  - sync: `11.97 ms`
  - encode: `0.43 ms`
  - mux: `0.60 ms`
- mat avg:
  - preprocess: `0.03 ms`
  - ORT/RVM: `13.48 ms`
  - kernel: `0.39 ms`

### FP16 Full-file Run: test_8k_2.mp4

Reports:

- `baseline/auto_tune_8k_phase1_20260516_162213.md`
- `baseline/auto_tune_8k_phase1_20260516_162213.json`

Results:

- latest interval FPS: `36.93`
- average interval FPS: `36.63`
- stage avg:
  - decode: `0.05 ms`
  - composite: `13.99 ms`
  - sync: `12.67 ms`
  - encode: `0.34 ms`
  - mux: `0.02 ms`
- mat avg:
  - preprocess: `0.08 ms`
  - ORT/RVM: `13.23 ms`
  - kernel: `0.59 ms`

### FP16 60s Baseline: test_8k.mp4

Reports:

- `baseline/auto_tune_8k_phase1_20260516_162343.md`
- `baseline/auto_tune_8k_phase1_20260516_162343.json`

Results:

- HTTP status: `200`
- first byte: `3.913 s`
- bytes read: `156943716`
- average client bitrate: `24.29 Mbps`
- latest interval FPS: `36.94`
- average interval FPS: `36.89`
- slow mux warnings: `0`
- pacing/stall/timeout lines: `0`
- stage avg:
  - decode: `0.06 ms`
  - composite: `14.25 ms`
  - sync: `12.34 ms`
  - encode: `0.34 ms`
  - mux: `0.07 ms`
- mat avg:
  - preprocess: `0.04 ms`
  - ORT/RVM: `13.68 ms`
  - kernel: `0.43 ms`

### FP16 60s Target Baseline: test_8k.mp4 + ALPHA_STRIDE=1

Command:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 240 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx --server-env PT_ALPHA_STRIDE=1
```

Reports:

- `baseline/auto_tune_8k_phase1_20260516_170451.md`
- `baseline/auto_tune_8k_phase1_20260516_170451.json`

Results:

- HTTP status: `200`
- first byte: `2.858 s`
- latest interval FPS: `35.56`
- average interval FPS: `35.03`
- slow mux warnings: `0`
- pacing/stall/timeout lines: `0`
- stage avg:
  - decode: `0.08 ms`
  - composite: `25.65 ms`
  - sync: `1.84 ms`
  - encode: `0.50 ms`
  - mux: `0.05 ms`
- mat avg:
  - preprocess: `0.17 ms`
  - ORT/RVM: `24.60 ms`
  - kernel: `0.55 ms`

Conclusion:

- Stride=1 and stride=3 do differ materially.
- With stride=1, the bottleneck clearly shifts back to ORT/RVM/composite instead of the sync wait observed in stride=3 diagnostics.
- Current FP16 + ThreadedDecoder + HEVC reaches about `35.03fps` on `videos/test_8k.mp4`, about `5fps` short of 40fps.

## FP32 Comparison

Reference FP32 baseline:

- `baseline/auto_tune_8k_phase1_20260516_154731.md`
- video: `videos/test_8k_2.mp4`
- latest interval FPS: `36.40`
- average interval FPS: `36.13`
- ORT/RVM: `15.17 ms`
- composite: `15.98 ms`
- sync: `11.00 ms`

FP16 on the same video:

- `baseline/auto_tune_8k_phase1_20260516_162213.md`
- latest interval FPS: `36.93`
- average interval FPS: `36.63`
- ORT/RVM: `13.23 ms`
- composite: `13.99 ms`
- sync: `12.67 ms`

Stride=3 conclusion:

- The FP16 model is active and valid.
- In stride=3, ORT/RVM dropped from about `15.17 ms` to about `13.23 ms`, a reduction of about `1.94 ms`.
- In stride=3, end-to-end FPS only increased from about `36.13` to about `36.63`, about `0.5 fps`.
- The 60s `test_8k.mp4` stride=3 baseline is stable at about `36.89 fps`, still below the 40fps target.
- Sync wait increased to about `12-13 ms`, so the saved ORT time is absorbed by GPU wait / cross-stream synchronization in stride=3.

## Risks and Notes

- This stage only validated performance and stability. It did not perform subjective frame/alpha edge quality comparison.
- FP16 may slightly alter alpha edges and should be checked in real VR playback.
- `test_8k.mp4` and `test_8k_2.mp4` have different content and bitrate, so client bitrate and first-byte time should not be compared directly between them.
- FP16 alone is not enough to reach 40fps.
- `PT_ALPHA_STRIDE=1` and the default `PT_ALPHA_STRIDE=3` must be treated as separate performance targets.

## Recommended Next Step

FP16 can remain as an optional or default-candidate model, but it does not close the stride=1 40fps gap by itself. Recommended next steps:

1. Capture a correct Nsight profile of the actual server process to explain the `sync/upload_sync` wait.
2. If continuing on ORT/RVM, evaluate CUDA Graph after confirming fixed IOBinding pointers and fixed input shapes.
3. Do not resume DirectML, default TensorRT, custom composite kernel work, or Python staged pipeline work for now.

## Verification

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\matting.py pipeline\pynv_stream.py tools\auto_tune_8k.py
```
