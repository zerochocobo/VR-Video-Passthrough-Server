# MatAnyone2 Drag / Afterimage Handling Summary

Date: 2026-05-27

## Background

This investigation focused on the MatAnyone2 drag / afterimage issue in `videos/72456_3840p.mp4`. The visible symptom is a semi-transparent old body, arm, or hair silhouette left behind after fast subject motion. The issue existed before the V2 optimization work, so it is not a new regression introduced by V2 or G2 guided refine.

Existing outputs inspected:

- `videos/72456_3840p_matanyone2_S000000_30S_LR_180_SBS_passthrough.mp4`
- `videos/72456_3840p_matanyone2m_S000000_15S_LR_180_SBS_passthrough.mp4`
- `videos/72456_3840p_rvm1_S000000_30S_LR_180_SBS_passthrough.mp4`

Diagnostic contact sheets:

- `debug_output/72456_drag_existing/m2_30s.contact.png`
- `debug_output/72456_drag_existing/rvm_30s.contact.png`
- `debug_output/72456_drag_compare/contact_crop_left_variants.png`
- `debug_output/72456_drag_compare/contact_crop_reset120_vs_reset60.png`
- `debug_output/72456_drag_compare/contact_default_fix.png`

## Root Cause Assessment

The main cause is MatAnyone2 propagation-state residue. During sequential propagation, historical foreground shape can leak into later alpha frames. With fast motion, this appears as old arm, body, or hair outlines.

`PT_MATANYONE2_ALPHA_SMOOTH` EMA smoothing makes the drag more visible, but it is not the root cause. Disabling smoothing slightly improves the result and improves throughput, but old silhouettes remain because the propagation state itself is still stale.

G2 guided refine mainly handles edge upsampling and local refinement. It does not own the MatAnyone2 temporal propagation state. The previously observed light edge expansion around frames 360/540/720 is a separate edge-refinement side effect, not the same problem as this drag issue.

The rug / floor region under the feet is a foreground-selection or bootstrap-mask pollution issue, not purely temporal drag. It should be handled separately with better bootstrap filtering, ROI constraints, or person-only constraints.

## Experiments

Baseline default path:

- Output: `debug_output/72456_drag_baseline_10s.mp4`
- 600 frames, return code 0, throughput 16.70 fps, `matting_avg=56.924 ms`
- Visual result: severe old arm/body silhouettes after fast motion.

Disable alpha EMA smoothing only:

- Output: `debug_output/72456_drag_nosmooth_10s.mp4`
- 600 frames, return code 0, throughput 19.30 fps, `matting_avg=49.222 ms`
- Visual result: slightly better and faster, but visible afterimages remain.

Re-bootstrap every 120 frames, smoothing disabled:

- Output: `debug_output/72456_drag_reset120_nosmooth_10s.mp4`
- 600 frames, return code 0, throughput 19.10 fps, `matting_avg=49.808 ms`
- Visual result: clear improvement near reset frames, but visible silhouettes remain between resets.

Re-bootstrap every 60 frames, smoothing disabled:

- Output: `debug_output/72456_drag_reset60_nosmooth_10s.mp4`
- 600 frames, return code 0, throughput 17.94 fps, `matting_avg=53.179 ms`
- Visual result: best tested option; old arm/body silhouettes are substantially reduced.

Default path after the fix:

- Output: `debug_output/72456_drag_default_after_fix_10s.mp4`
- Log showed `segment_frames=60`, `alpha_smooth=0`, `alpha_refine=off`, `iobinding=1`
- `matanyone2_step_update.onnx` used `TensorrtExecutionProvider`
- Segment plan: `[0, 60, 120, 180, 240, 300, 360, 420, 480, 540]`
- 600 frames, return code 0, throughput 18.01 fps, `matting_avg=52.992 ms`
- `matanyone2_step_update_avg=21.938 ms n=1180`
- `matanyone2_first_refine_avg=112.094 ms n=20`
- Visual comparison in `debug_output/72456_drag_compare/contact_default_fix.png` matches the earlier best `reset60_nosmooth` result.

## Adopted Strategy

Disable MatAnyone2 alpha EMA smoothing by default:

- `PT_MATANYONE2_ALPHA_SMOOTH` default changed from `1` to `0`

Add fixed-interval propagation reset:

- Added `PT_MATANYONE2_SEGMENT_FRAMES`
- Initial default value: `60`; after Phase 1 and the left/right buffer fix, this was changed to `240`
- Value `0` disables fixed-interval reset

The setting is wired into:

- `tools/offline_passthrough.py`
- `tools/offline_alpha_passthrough.py`

`PROJECT.md` was updated with the new key configuration.

## Verification

Syntax check:

```powershell
.\.venv\Scripts\python.exe -m py_compile config.py tools\offline_passthrough.py tools\offline_alpha_passthrough.py offline\matanyone2_engine.py
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alpha_guided_filter.py tests\test_matanyone2_engine.py tests\test_scene_detection.py tests\test_offline_convert.py tests\test_matanyone2_trt_runtime_paths.py
```

Result: `25 passed, 2 skipped`.

## Practical Lessons

For MatAnyone2-style propagation matting, fast-motion drag should be separated into three categories:

- Propagation-state residue: old pose or limb silhouettes leak into later frames. This was the main issue here.
- Post-process smoothing residue: EMA or temporal filters make old alpha decay too slowly. Disabling smoothing helps, but does not replace state reset.
- Initial mask pollution: floor, rug, or objects are included as foreground. This requires better bootstrap mask quality, not only temporal tuning.

The early practical mitigation was short-segment MatAnyone2 re-bootstrap with alpha EMA smoothing disabled. A 60-frame reset interval provided an acceptable quality/performance balance before the root-cause work; after Phase 1 and the left/right buffer fix, the default was relaxed to 240 frames.

## Remaining Risks

Frequent re-bootstrap increases first-refine calls and can raise p99 latency at reset frames. The 240-frame default reduces that overhead, but longer videos and higher-concurrency jobs still need tail-latency monitoring.

If the prepass mask is unstable at reset points, the output may show small per-second mask consistency changes. More video coverage is needed.

The rug / floor inclusion under the feet is still not fully solved. It should be treated as a separate bootstrap-mask quality issue, possibly with person-only constraints, ROI limiting, or stronger initial-mask filtering.

## Follow-Up Execution Update

After Phase 1 root-cause fixes and the SBS left/right buffer fix, the `videos/72456_3840p.mp4` SAM3 prepass masks were reused for a 10-second ablation:

- `segment_frames=60`: 14.31 fps, `matting_avg=67.123 ms`
- `segment_frames=120`: 17.11 fps, `matting_avg=55.881 ms`
- `segment_frames=240`: 17.60 fps, `matting_avg=54.159 ms`

Visual contact sheet: `debug_output/72456_phase1_ablation_compare/seg60_120_240_mid_end_contact.png`.

Decision: after Phase 1 and the left/right buffer fix, `segment_frames=240` did not show worse segment-tail drag than 60 frames on this clip and significantly reduced bootstrap overhead. The default was changed from `60` to `240`.

After user retesting, Phase 1 quality was accepted as sufficient. Phase 2 optical-flow compensation and Phase 3 bidirectional propagation are deferred and should not be implemented unless a new release-quality requirement reopens the work.
