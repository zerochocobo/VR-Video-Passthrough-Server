# MatAnyone2 Offline V2 Execution Review Summary

**Date**: 2026-05-26  
**Scope**: MatAnyone2 offline quality/stability/performance V2 implementation  
**Status**: Mainline implementation completed, with guided refine disabled by default after halo investigation.

## 1. Executive Summary

The MatAnyone2 offline V2 mainline has been implemented:

- G0 shared engine extraction
- G1 batch-1 `step_update` IOBinding
- G2 guided alpha refinement, now experimental/off by default
- G4 scene-cut prepass segment merge
- G5 per-eye alpha EMA smoothing
- G3-A 1024 ROI crop/letterbox quality mode, off by default

The current conservative default path is:

- `PT_MATANYONE2_IOBINDING=1`
- `PT_MATANYONE2_ALPHA_SMOOTH=1`
- `PT_MATANYONE2_SCENE_RESET=1`
- `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0`
- `PT_MATANYONE2_ROI_CROP=0`

This keeps the safe V1-like matte boundary while retaining the new performance and stability changes.

## 2. Implemented Work

### G0 Shared Engine

- Added `offline/matanyone2_engine.py`.
- Green and alpha offline tools now share one MatAnyone2 engine.
- Output behavior is selected by `output_mode="green"` or `"alpha"`.

### G1 IOBinding

- Added batch-1 GPU IOBinding for the hot `matanyone2_step_update.onnx` path.
- Bootstrap graphs still use normal `session.run()`.
- Added automatic fallback on IOBinding failure.
- Added alternating output slots to avoid recurrent-state input/output aliasing.

### G2 Guided Alpha Refine

- Added `pipeline/alpha_guided_filter.py`.
- Uses the uploaded NV12 Y plane as the guide.
- Added support clamps:
  - `PT_MATANYONE2_GUIDED_SUPPORT_FLOOR=0.02`
  - `PT_MATANYONE2_GUIDED_MAX_DELTA=0.08`
- After visual ablation, guided refine is default off because it can create visible background halos.

### G4/G5 Stability

- Added `utils.scene_detection.SceneCutDetector`.
- SAM3 and YOLOWorld+EfficientSAM prepasses merge active scene cuts into MatAnyone2 segment starts.
- Added per-eye alpha EMA smoothing with segment reset cleanup.

### G3-A ROI Quality Mode

- Added `pipeline/matanyone2_roi.py`.
- Added GPU NV12 ROI crop/letterbox preprocess and ROI alpha unwarp/feather kernels.
- ROI is derived from segment bootstrap masks and cached per segment.
- ROI requires valid ROIs for both eyes; otherwise the engine falls back to full-eye processing.
- Fixed right-eye SBS source offset and ROI bootstrap-mask coordinate alignment.
- ROI remains default off because it is quality-only with the fixed 1024 model.

## 3. Verification

Static and unit verification:

- `py_compile` passed for the MatAnyone2 engine, matting, guided filter, ROI helper, tools, and tests.
- `pytest tests/test_alpha_guided_filter.py tests/test_matanyone2_engine.py tests/test_scene_detection.py tests/test_offline_convert.py tests/test_matanyone2_trt_runtime_paths.py`
- Result: `25 passed, 2 skipped`.
- `git diff --check` passed.

Runtime smoke verification on `videos/test_8k.mp4`, duration `0.2s`:

- ROI green TRT+IOBinding smoke passed.
- ROI alpha TRT+IOBinding smoke passed.
- Default post-halo-fix smoke passed and logged `alpha_refine=off`.
- Guided ablation outputs were generated under `debug_output/matanyone2_ablation_*`.

## 4. Halo Investigation

Observed issue:

- V2 default output showed a visible dirty matte/background ring around the person.
- Pre-V2/V1-like output was clean.

Ablation:

- `v2_default`: guided on + smoother on -> halo visible.
- `smooth_off`: guided on + smoother off -> halo still visible.
- `guided_off`: guided off + smoother on -> clean.
- `v1_like`: guided off + smoother off + scene reset off -> clean.

Conclusion:

- The main source was G2 guided alpha refinement, not EMA smoothing or ROI.
- The luma-guided filter amplified weak low-confidence alpha near background-adjacent regions.

Mitigation:

- Added support floor and max-delta clamps to guided refine.
- Changed `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE` default from `1` to `0`.
- Guided refine remains available as an experimental feature for future tuning.

## 5. Remaining Review Items

- The full 15s five-case matrix is still pending:
  - V1 baseline
  - IOBinding only
  - IOBinding + guided
  - Full V2 without ROI
  - Full V2 + ROI-A
- SAM3 ROI smoke has not yet been run; the current ROI smoke used YOLOWorld+EfficientSAM.
- ROI-B 512/768 speed mode is intentionally not implemented and remains a conditional follow-up.
- Guided refine needs a new quality strategy before it can be safely default-on.

## 6. Recommended Reviewer Focus

- Confirm default output quality with `PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0`.
- Review whether G2 should remain experimental or be redesigned.
- Review ROI-A quality value on far-subject footage.
- Decide whether to start the separate G3-B 512/768 ROI speed-mode effort.

## 7. Safe Rollback Switches

```bat
set PT_MATANYONE2_IOBINDING=0
set PT_MATANYONE2_EDGE_AWARE_UPSAMPLE=0
set PT_MATANYONE2_ROI_CROP=0
set PT_MATANYONE2_SCENE_RESET=0
set PT_MATANYONE2_ALPHA_SMOOTH=0
```
