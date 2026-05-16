> ⚠️ SUPERSEDED 2026-05-16
> FPS, sync, and bottleneck-attribution conclusions in this report were produced under the old default PT_PASSTHROUGH_MAX_FPS=30 and/or non-target PT_ALPHA_STRIDE=3 conditions. Later review invalidated them or limited them to those diagnostic conditions only.
> Use summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_EN.md as the corrected baseline entry point.
> This file is retained only as a research-process archive; implementation conclusions unrelated to the FPS cap must be read with their stated scope.
# Stage 4R Summary - Downstream Barrier Probes

## Background

The external review observed that FP16 saved ORT/RVM time but the saving was mostly absorbed by `sync`. It suggested a structural ~27ms serial barrier and proposed low-cost checks:

1. Confirm whether the production path is still using H264 NVENC.
2. Increase the NV12 output slot pool.
3. Bypass RVM to isolate decode -> composite -> encode/mux throughput.

This stage ran those checks.

## Scope Correction

The slot=8 and RVM bypass isolation tests in this file were run under the default `PT_ALPHA_STRIDE=3` or under bypass conditions. They only show a fixed wait/throughput ceiling in the stride=3 diagnostic path. They do not represent the original `PT_ALPHA_STRIDE=1` target.

A later explicit `PT_ALPHA_STRIDE=1` run produced the correct target baseline:

- `baseline/auto_tune_8k_phase1_20260516_170451.md`
- average interval FPS: `35.03`
- ORT/RVM: `24.60 ms`
- sync: `1.84 ms`

This means stride=1 is still mainly ORT/RVM/composite bound, unlike the sync-bound behavior seen in this stride=3 diagnostic file.

## Corrected Premise: Production PyNv Already Uses HEVC

Code evidence:

- `pipeline/pynv_stream.py` has `PYNV_OUTPUT_CODEC = "hevc"`.
- `PYNV_BACKEND_LABEL = "pynv_hevc"`.
- Live cache keys in logs contain `/hevc/30.000/...`.

Therefore the proposed “switch from h264_nvenc to hevc_nvenc” test is already satisfied on the current production PyNv live path. The current ~37fps ceiling cannot be explained by accidentally using H264 NVENC.

Note:

- `config.PASSTHROUGH_VCODEC=hevc_nvenc` controls the FFmpeg fallback path.
- The PyNv production live path does not use that setting.

## Test 1: NV12 Slot Pool 3 -> 8

Command:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 180 --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx --server-env PT_PASSTHROUGH_NV12_RING_SLOTS=8
```

Reports:

- `baseline/auto_tune_8k_phase1_20260516_164945.md`
- `baseline/auto_tune_8k_phase1_20260516_164945.json`

Comparison:

| Config | average FPS | latest FPS | composite | sync | ORT/RVM |
|---|---:|---:|---:|---:|---:|
| FP16 slot=3 | 36.89 | 36.94 | 14.25ms | 12.34ms | 13.68ms |
| FP16 slot=8 | 36.87 | 36.97 | 15.36ms | 11.08ms | 14.55ms |

Conclusion:

- Increasing the slot pool from 3 to 8 did not improve end-to-end FPS.
- The `composite` and `sync` buckets moved slightly, but throughput stayed the same.
- A shallow NV12 output slot pool is not the main cause of the ~37fps ceiling.

## Test 2: RVM Bypass / Downstream Isolation

Added diagnostic switch:

- `PT_PASSTHROUGH_RVM_BYPASS_ALPHA=1`

Files:

- `config.py`
- `pipeline/matting.py`

Behavior:

- Diagnostic only.
- Skips RVM inference in the PyNv/CuPy green path.
- Uses an all-foreground alpha mask.
- Keeps decode, NV12 composite kernel, sync, HEVC encode, and mux.
- Disabled by default and not a production visual mode.

Command:

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 --startup-timeout 240 --client-timeout 180 --server-env PT_PASSTHROUGH_RVM_BYPASS_ALPHA=1
```

Reports:

- `baseline/auto_tune_8k_phase1_20260516_165209.md`
- `baseline/auto_tune_8k_phase1_20260516_165209.json`

Results:

- latest interval FPS: `37.11`
- average interval FPS: `37.02`
- stage avg:
  - decode: `7.02 ms`
  - composite: `0.69 ms`
  - sync: `18.42 ms`
  - encode: `0.49 ms`
  - mux: `0.29 ms`
- mat avg:
  - preprocess: `0.00 ms`
  - ORT/RVM: `0.00 ms`
  - kernel: `0.63 ms`

Key log:

```text
[DIAG] alpha #1800 bypass: frame=8192x4096 alpha_shape=(1024, 2048) use_nv12=True
```

Conclusion:

- With RVM fully bypassed, FPS is still only about `37.02`.
- This proves the current ~37fps ceiling is not an RVM/ORT compute bottleneck.
- The wait moves into `decode/sync`, pointing to a downstream GPU synchronization, decode visibility, or encode-tail barrier.

## Overall Conclusion

Now ruled out:

- H264 encoding assumption: production is already HEVC.
- Shallow slot pool: slot=8 equals slot=3 throughput.
- RVM/ORT as the primary bottleneck: RVM bypass still caps at ~37fps.
- Custom composite kernel: kernel is about 0.4-0.7ms.

Stride=3 diagnostic state:

- There is a structural ~27ms GPU/encode/decode synchronization barrier.
- In stride=3 or RVM bypass diagnostics, further ORT-only work is likely to be absorbed by `sync` or decode wait.
- This conclusion must not be directly applied to stride=1. The explicit stride=1 run shows ORT/RVM is still the main bottleneck.
- If investigating the stride=3 fixed barrier further, the next step needs a correct Nsight trace of the actual server process.

## Recommended Next Step

1. Profile the server process directly, not the auto_tune parent process.
2. Nsight should answer:
   - delay from NVENC submit to bitstream availability;
   - whether ThreadedDecoder/NVDEC GPU work contends with ORT/CuPy;
   - whether CuPy, ORT, and PyNv streams serialize;
   - what kernels/memcopies are pending before `cuda_stream.synchronize()`.
3. Defer CUDA Graph for stride=3 barrier work. For stride=1, CUDA Graph / ORT optimization may still matter because ORT/RVM is `24.60 ms`.
4. Keep `PT_PASSTHROUGH_RVM_BYPASS_ALPHA` as a diagnostic switch only.

## Verification

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\matting.py pipeline\pynv_stream.py tools\auto_tune_8k.py
```
