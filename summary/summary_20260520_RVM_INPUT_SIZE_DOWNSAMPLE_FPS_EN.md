# RVM Input Size and Downsample FPS Summary

Date: 2026-05-20

## Scope

This summary records the realtime RVM FPS experiments for `videos/test_8k.mp4` on the production DLNA path. The comparison covered:

- RVM fp32 vs fp16.
- SBS batch=2 vs two separate batch=1 eye inferences.
- RVM input sizes `1024`, `2048`, and original per-eye `4096`.
- Several `PT_RVM_DOWNSAMPLE_RATIO` values.

## Final Default Decision

The production defaults are:

- `PT_MODEL_PATH=models/rvm_mobilenetv3_fp16.onnx`
- `PT_MATTING_INPUT_SIZE=1024`
- `PT_RVM_DOWNSAMPLE_RATIO=0.5`
- `PT_MATTING_SBS_BATCH=1`

`1024 + 0.5` is the final balanced default. `2048 + 0.125` was faster in producer FPS diagnostics, but follow-up visual matte validation showed it could barely extract the person. `1024 + 1.0` was slightly cleaner in some frames but produced more unrelated matte regions, and `2048 + 0.5` was much slower. Across quality, false positives, and performance, `1024 + 0.5` was the best-balanced choice.

## Key Results

All numbers below are warmup-filtered server producer diagnostics from the same uncapped realtime DLNA setup: `PT_PASSTHROUGH_MAX_FPS=0`, `PT_ALPHA_STRIDE=1`, `PT_DECODE_MAX_SIDE=0`, fp16 model unless noted.

| Case | Actual RVM input | Downsample | Mean interval FPS | Median interval FPS | ORT mean |
|---|---:|---:|---:|---:|---:|
| Final balanced default | `(2,3,1024,1024)` | `0.5` | `43.13` | `42.29` | `18.81 ms` |
| Faster but rejected after matte QA | `(2,3,2048,2048)` | `0.125` | `53.20` | `49.73` | `14.64 ms` |
| 2048 faster variant | `(2,3,2048,2048)` | `0.0625` | `54.62` | `51.94` | `13.91 ms` |
| 2048 quality-heavy | `(2,3,2048,2048)` | `0.25` | `34.83` | `34.65` | `24.08 ms` |
| Original per-eye size | `(2,3,4096,4096)` | `0.125` | `20.25` | `20.43` | `43.73 ms` |
| Original per-eye size | `(2,3,4096,4096)` | `0.0625` | `25.26` | `25.27` | `34.30 ms` |

## Other Findings

- fp16 was about 16-17% faster than fp32 under the tested RVM paths.
- SBS batch=2 remained faster than two separate batch=1 eye inferences, so `PT_MATTING_SBS_BATCH=1` stays as the default.
- Full per-eye `4096x4096` RVM input is not viable for realtime on this machine, even with low downsample ratios.
- `2048 + 0.125` outperformed the previous `1024 + 0.5` path in FPS, but was rejected after visual matte QA because the person mask quality was unacceptable.
- Final visual review selected `1024 + 0.5` because it has the best overall balance: acceptable matte quality, fewer unrelated matte regions than `1024 + 1.0`, and much better speed than `2048 + 0.5`.

## SBS Batch 1 vs Batch 2

`PT_MATTING_SPLIT_SBS=1` was fixed for these tests. Batch=2 means both eyes are sent to RVM in one ORT call as `(2,3,H,W)`. Batch=1 means the left and right eyes are inferred separately as `2x(1,3,H,W)` with independent RVM recurrent state slots.

| Model | Batch mode | Actual RVM input | Mean interval FPS | Median interval FPS | ORT mean |
|---|---:|---:|---:|---:|---:|
| fp32 | batch=2 | `(2,3,1024,1024)` | `37.17` | `36.87` | `22.68 ms` |
| fp32 | two batch=1 eyes | `2x(1,3,1024,1024)` | `35.88` | `35.17` | `23.74 ms` |
| fp16 | batch=2 | `(2,3,1024,1024)` | `43.13` | `42.29` | `18.81 ms` |
| fp16 | two batch=1 eyes | `2x(1,3,1024,1024)` | `41.93` | `40.38` | `19.70 ms` |

Findings:

- fp32 batch=2 was about `3.6%` faster than two separate batch=1 eye inferences.
- fp16 batch=2 was about `2.9%` faster than two separate batch=1 eye inferences.
- The gain is modest but consistent, so `PT_MATTING_SBS_BATCH=1` remains the default.
- RVM now respects `PT_MATTING_SBS_BATCH`; before the fix, RVM always took the batch=2 path even when the setting was `0`.

## Resize Behavior

The matting resize path does not force non-1:1 videos into a square when `PT_MATTING_SQUARE=0`.

- For SBS 8K `8192x4096`, each eye becomes `4096x4096`, then the default RVM reference path produces `1024x1024`.
- For normal 2D non-square videos, aspect ratio is preserved as much as possible and dimensions are rounded down to multiples of 32.
- Small 2D inputs are no longer upscaled to the RVM reference size; they keep their source scale except for the multiple-of-32 alignment.

## Artifacts

- `baseline/auto_tune_8k_phase1_20260520_114353.*`: `1024 + 0.5`
- `baseline/auto_tune_8k_phase1_20260520_130418.*`: `2048 + 0.125`
- `baseline/auto_tune_8k_phase1_20260520_130454.*`: `2048 + 0.0625`
- `baseline/auto_tune_8k_phase1_20260520_123117.*`: `4096 + 0.125`
- `baseline/auto_tune_8k_phase1_20260520_123243.*`: `4096 + 0.0625`

## Matte Screenshot Follow-Ups

Visual review found that `1024 + 1.0` was slightly cleaner in some frames than `1024 + 0.5`, but also produced more unrelated matte regions. A second visual check compared `1024 + 0.5` against `2048 + 0.5`; the existing `1024 + 0.5` outputs were reused.

Only the left half was exported for single-variant and comparison images.

| Screenshot comparison | ORT mean, sample 0 excluded | Composite mean, sample 0 excluded | Output |
|---|---:|---:|---|
| `1024 + 1.0` | `48.16 ms` | `26.56 ms` | `debug_output/rvm_matte_compare_1024_dr1_vs_dr0p5` |
| `1024 + 0.5` | `15.55 ms` | `26.11 ms` | `debug_output/rvm_matte_compare_1024_dr1_vs_dr0p5` |
| `2048 + 0.5` | `45.25 ms` | `24.52 ms` | `debug_output/rvm_matte_compare_left_1024_dr0p5_vs_2048_dr0p5` |

## Full SBS vs Split-SBS Matte Check

With the final `1024 + 0.5` parameters fixed, another screenshot experiment compared full SBS input against split-SBS batch2:

| Mode | Settings | Actual RVM shape | ORT mean, sample 0 excluded | Composite mean, sample 0 excluded |
|---|---|---:|---:|---:|
| Full SBS batch1 | `PT_MATTING_SPLIT_SBS=0`, `PT_MATTING_SBS_BATCH=0` | `(1,3,1024,2048)` | `17.25 ms` | `37.27 ms` |
| Split SBS batch2 | `PT_MATTING_SPLIT_SBS=1`, `PT_MATTING_SBS_BATCH=1` | `(2,3,1024,1024)` | `14.60 ms` | `31.48 ms` |

Artifacts:

- `debug_output/rvm_matte_compare_full_sbs_vs_split_batch2`
- Videos: `test_4k.mp4`, `test_8k.mp4`, `72456_3840p.mp4`
- 10 sampled frames per video
- Comparison sheets are stacked vertically; for the 8K sample, single matte images are `1280x640` and comparison sheets are `1280x1280`.
