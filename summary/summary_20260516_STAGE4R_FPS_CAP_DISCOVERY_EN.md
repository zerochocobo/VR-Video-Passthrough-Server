# Stage 4R Summary - FPS Cap Discovery and Superseded Performance Conclusions

## Conclusion

On 2026-05-16, review confirmed that most previous Stage 4R performance runs were polluted by the old default `PT_PASSTHROUGH_MAX_FPS=30`. Some runs also used the non-target condition `PT_ALPHA_STRIDE=3`. Therefore, earlier conclusions about a `~37fps` physical ceiling, sync attribution, RVM bypass showing no gain, and slot/decoder/codec changes being ineffective must not be used as evidence for the 8K 40fps / `ALPHA_STRIDE=1` target.

The old summaries are not deleted. They now carry a `SUPERSEDED 2026-05-16` banner and remain only as research-process archives. Implementation conclusions unrelated to the FPS cap remain valid within their own stated scope.

## Root Cause

The old default in `config.py` was:

```python
PASSTHROUGH_MAX_FPS = float(_env("PASSTHROUGH_MAX_FPS", 30))
```

This value flows through `utils/video_metadata.py::effective_fps()` and caps the output media FPS to 30. `videos/test_8k.mp4` is about 59.94fps, but previous live cache keys contained `/hevc/30.000/`, proving that the output stream was generated as a 30fps media stream.

The producer could generate that 30fps output stream faster than realtime, so logged `36-37fps` values were not the true uncapped 8K 59.94fps throughput ceiling.

Additional clarification: `PT_PASSTHROUGH_MAX_FPS=30` caps output media timestamps and selected output frames; it is not a producer wall-clock rate limiter. Whether the output media FPS is 30 or 59.94, the producer still generates the target frames as fast as it can. Therefore, the old observation that "cap=30 still logged 36-37fps" is not contradictory; it happened to be close to the PyNv 8K HEVC encode throughput discovered later.

## Fixes Applied

- `config.py` now defaults `PT_PASSTHROUGH_MAX_FPS` to `0`, meaning no output FPS cap.
- The config knob remains available: set `PT_PASSTHROUGH_MAX_FPS=30` explicitly for client-compatibility diagnostics.
- `pipeline/pynv_stream.py` now logs runtime config values, including actual `alpha_stride`, `max_fps`, `output_fps`, decoder, worker mode, model, and RVM bypass state.
- Superseded banners were added to the Stage 4R threaded/two-stage, profiling/TRT blocker, FP16 baseline, and downstream barrier probe summaries in both languages.

## Old Conclusion Triage

Still valid:

- ThreadedDecoder frames must not be queued across threads; old batch GPU pointers may become invalid after the next `get_batch_frames()`, so the native crash risk remains valid.
- The serial-consumption/lifetime design of `PyNvThreadedSerialDecoder` remains valid.
- The MKV unsafe-Cues block policy remains valid.
- TensorRT EP fallback caused by missing TRT runtime DLLs remains a deployment finding.
- DirectML remains architecturally incompatible with the current CUDA-resident pipeline.
- The FP16 model can load and run under ORT CUDA EP, but its performance benefit must be remeasured under uncapped target-stride conditions.

Invalidated or requiring retest:

- `~37fps` as a physical ceiling: invalidated.
- sync wait as proof of a GPU/NVENC bottleneck: invalidated; in old data it may have been cap-induced waiting migration.
- RVM bypass still being around 37fps therefore proving a downstream bottleneck: invalidated under cap conditions.
- slot=3 vs slot=8 having no effect: retest required uncapped.
- FP16, decoder, and worker-mode FPS benefits: retest required uncapped with explicit stride.

## First Clean Fact

One clean run has been completed on `videos/test_8k.mp4` with FP16, `PT_ALPHA_STRIDE=1`, and `PT_PASSTHROUGH_MAX_FPS=0`:

- Report: `baseline/auto_tune_8k_phase1_20260516_171415.md/json`
- Target frames: `3600`, no longer the roughly `1801` frames from the 30fps cap
- Latest interval FPS: `36.32`
- Average interval FPS: `36.50`
- Stage avg: decode `0.06ms`, composite `24.95ms`, sync `2.06ms`, encode `0.43ms`, mux `0.03ms`
- Mat avg: pre `0.13ms`, ORT/RVM `24.13ms`, kernel `0.44ms`

This confirms that under uncapped `stride=1`, RVM/ORT is back on the critical path. The old stride=3/bypass evidence must not be used to infer the target bottleneck.

## Retest Plan

1. Baseline A: `stride=1`, FP32, simple decoder, slot=3, `PT_PASSTHROUGH_MAX_FPS=0`.
2. Comparison B: from A, run `stride=1/3/6` to quantify stride sensitivity.
3. Comparison C: compare FP32 vs FP16 under `stride=1`.
4. Comparison D: compare simple vs threaded_serial under the winning config.
5. Comparison E: winning config + RVM bypass to measure the true downstream ceiling.

Every report must include runtime config evidence proving `alpha_stride`, `max_fps=0`, and `output_fps≈59.94`; otherwise it must not be used for optimization decisions.
## Uncapped Retest Addendum

All tests below used `videos/test_8k.mp4` with explicit `PT_PASSTHROUGH_MAX_FPS=0`. The target frame count was `3600`, and the live cache key used `/hevc/0.000/`, not the old `/hevc/30.000/`.

| Test | Report | Conditions | Avg FPS | Latest FPS | Key stage timings |
|---|---|---|---:|---:|---|
| A | `baseline/auto_tune_8k_phase1_20260516_172105.md` | FP32, simple, stride=1 | 34.37 | 34.30 | decode 3.27, composite 23.48, sync 1.98, ORT 22.71 |
| B1 | `baseline/auto_tune_8k_phase1_20260516_172245.md` | FP32, simple, stride=3 | 36.68 | 36.95 | decode 17.38, composite 7.60, sync 1.67, ORT 7.04 |
| B2 | `baseline/auto_tune_8k_phase1_20260516_172408.md` | FP32, simple, stride=6 | 37.05 | 37.01 | decode 21.07, composite 4.03, sync 1.53, ORT 3.49 |
| C | `baseline/auto_tune_8k_phase1_20260516_172542.md` | FP16, simple, stride=1 | 36.59 | 36.34 | decode 7.00, composite 18.58, sync 1.45, ORT 17.70 |
| D | `baseline/auto_tune_8k_phase1_20260516_171415.md` | FP16, threaded_serial, stride=1 | 36.50 | 36.32 | decode 0.06, composite 24.95, sync 2.06, ORT 24.13 |
| E1 | `baseline/auto_tune_8k_phase1_20260516_172723.md` | FP16, simple, stride=1, RVM bypass | 37.30 | 37.14 | decode 23.94, composite 0.58, sync 1.81, ORT 0.00 |

## Final Correction: PyNv Uppercase P1 Preset Removes the 8K Encode Bottleneck

The previous judgement that "PyNv 8K HEVC is capped around 37fps" was still incomplete. After strictly following the NVIDIA PyNvVideoCodec API format, the missing detail was `preset` case: PyNv 2.1.0 accepts `preset="P1"` (uppercase P plus digit), but rejects `preset="p1"`.

Correct conclusion:

- The old `PT_PASSTHROUGH_MAX_FPS=30` did pollute early FPS attribution.
- The uncapped `~37fps` result was not a hardware limit and not an unavoidable PyNv limit.
- It was caused by the PyNv encoder default/wrong-preset configuration.
- With `preset="P1"` + `tuning_info="ultra_low_latency"` + `rc="cbr"`, both PyNv 8K HEVC encode and the production path exceed 40fps.

## Official Format Check

External review pointed out that the NVIDIA PyNvVideoCodec API Programming Guide uses this style:

```python
nvc.CreateEncoder(
    width=1920,
    height=1080,
    format="NV12",
    codec="hevc",
    preset="P2",
    tuning_info="low_latency",
)
```

Local validation confirms that PyNv 2.1.0 treats `preset` as case-sensitive:

- `preset="p1"`: initialization fails.
- `preset="P1"`: initializes and is much faster.
- `preset="P2"`: initializes but stays close to the old slow path.
- Legacy `LOW_LATENCY_HQ` initializes but is slower.

## P1 Pure Encode Validation

All tests used `tools/pynv_encode_probe.py --reuse-gpu-frame` to avoid CPU frame generation and per-frame upload noise.

| Conditions | Result |
|---|---:|
| `preset=P1` | `72.83fps` |
| `preset=P1, tuning_info=ultra_low_latency` | `76.84fps` |
| `preset=P1, tuning_info=ultra_low_latency, rc=cbr` | `76.90fps` |
| `preset=P1, tuning_info=ultra_low_latency, rc=cbr, gop=30, idrperiod=30` | `76.11fps` |
| `preset=P2, tuning_info=ultra_low_latency, rc=cbr` | `38.68fps` |
| `preset=P3, tuning_info=ultra_low_latency, rc=cbr` | `35.54fps` |

Conclusion: `P1` is the decisive option; P2/P3/legacy presets remain in the 35-39fps range.

## P1 Transcode Validation

`tools/pynv_transcode_probe.py` now supports `--preset`, `--tuning-info`, `--rc`, `--idrperiod`, and `--enc-opt` so probe and production can pass the same PyNv encoder parameters.

Command:

```powershell
.venv\Scripts\python.exe tools\pynv_transcode_probe.py test_8k.mp4 `
  --duration 20 --fps 0 --codec hevc --bitrate 60000000 --gop 60 `
  --preset P1 --tuning-info ultra_low_latency --rc cbr `
  --progress 300 --out debug_output\probe_transcode_8k_hevc_p1_ull_cbr.mp4
```

Result:

- `throughput=79.46fps`.
- `avg_decode=11.267ms`.
- `avg_encode=0.331ms`.
- Output remains `8192x4096 HEVC`.

This matches the independent FFmpeg CUDA decode + `hevc_nvenc -preset p1 -tune ull` control at about `80fps`, proving that PyNv can reach the same class of performance when configured correctly.

## Production Path Change

`config.py` now exposes production knobs:

- `PT_PASSTHROUGH_PYNV_PRESET`, default `P1`.
- `PT_PASSTHROUGH_PYNV_TUNING_INFO`, default `ultra_low_latency`.
- `PT_PASSTHROUGH_PYNV_RC`, default `cbr`.
- `PT_PASSTHROUGH_PYNV_IDR_PERIOD`, default empty.

`pipeline/pynv_stream.py` now uses one shared production/preflight encoder kwargs builder:

- `codec=hevc`;
- `bitrate=<effective_default_bitrate>`;
- `fps=<effective_fps>`;
- `gop=<PT_PASSTHROUGH_GOP>`;
- `bf=<PT_PASSTHROUGH_HEVC_BF>`;
- `preset=P1`;
- `tuning_info=ultra_low_latency`;
- `rc=cbr`.

## Production 8K / Stride=1 Validation

The validation used `videos/test_8k.mp4`, `PT_ALPHA_STRIDE=1`, FP16 RVM, `PT_PASSTHROUGH_MAX_FPS=0`, and `threaded_serial` decoder.

| Report | Duration | Avg FPS | Latest FPS | Key stage timings |
|---|---:|---:|---:|---|
| `baseline/auto_tune_8k_phase1_20260516_185207.md` | 20s | `55.62` | `56.41` | decode 0.05, composite 16.23, sync 1.10, encode 0.32, mux 0.02, ORT 15.61 |
| `baseline/auto_tune_8k_phase1_20260516_185313.md` | 60s | `56.16` | `56.57` | decode 0.04, composite 16.05, sync 1.24, encode 0.31, mux 0.02, ORT 15.47 |

Conclusion: the original target of 8K, `ALPHA_STRIDE=1`, and above 40fps is met. The current 60-second steady state is about `56fps`.

## Updated Valid Judgement

1. `PT_PASSTHROUGH_MAX_FPS=30` caps output media timestamps / selected frames, not producer wall-clock rate; old cap=30 logs around 36-37fps were not contradictory.
2. Uppercase `preset="P1"` is the key PyNv 2.1.0 8K HEVC performance option; lowercase `p1` fails, and P2/P3 remain slow.
3. The old "PyNv 8K HEVC is capped around 37fps" and "move to FFmpeg/PyAV" conclusions are invalidated.
4. The active bottleneck is now RVM/ORT again: the 60-second production run reports `ORT?15.47ms`, with total throughput around `56fps`.
5. Future optimization should use the P1 configuration as the baseline before revisiting FP16/TRT/CUDA Graph/RVM work; no further decisions should rely on the old 37fps data.

## Verification Commands

```powershell
.venv\Scripts\python.exe -m py_compile config.py pipeline\pynv_stream.py tools\pynv_transcode_probe.py tools\pynv_encode_probe.py
```

```powershell
.venv\Scripts\python.exe tools\auto_tune_8k.py phase1 `
  --video videos\test_8k.mp4 --profile quest --prefer green --duration 60 `
  --startup-timeout 240 --client-timeout 240 `
  --server-env PT_MODEL_PATH=G:\GIT\debug\PTMediaServer\models\rvm_mobilenetv3_fp16.onnx `
  --server-env PT_ALPHA_STRIDE=1 `
  --server-env PT_PASSTHROUGH_MAX_FPS=0 `
  --server-env PT_PASSTHROUGH_PYNV_DECODER=threaded_serial
```
