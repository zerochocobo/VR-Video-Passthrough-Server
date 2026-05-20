# Light Matching Phase 1 Implementation Plan (English)

- Date: 2026-05-19
- Scope: Add a runtime "ambient light matching" filter applied to the foreground (matted person) before NV12 composite, so the video subject blends more naturally into the user's real-world home lighting seen through Quest3 passthrough.
- Out of scope: Neural relighting (IC-Light / SwitchLight), directional shadows, edge bounce light, color transfer from reference photo. Those are Phase 2+.

---

## 1. Background

When wearing Quest3 with passthrough on, the user sees:
- Their real home environment (whatever color temperature their lamps are — often warm 3000K incandescent or 4000K LED).
- A green-keyed or alpha-keyed video subject composited on top by the VR player.

If the video was shot under studio lighting (cool 5500K) while the room is warm, the subject looks pasted in — the classic "cutout" effect. The cheapest fix that meaningfully reduces this is a per-frame color/luma correction applied to the foreground only, before it leaves the server.

## 2. Goal

- Apply user-configurable color temperature, tint, exposure, contrast, gamma, and saturation to the foreground portion of every passthrough frame (green and alpha modes both).
- Keep the cost below 0.5 ms per 8K frame on a RTX 20-series class GPU so the 8K target FPS is unaffected.
- Provide three friendly presets plus full manual control through the desktop UI.
- Persist settings per-session via `ui_settings.json` and propagate via `PT_*` environment variables to the server process.

## 3. Non-Goals

- No background modification (alpha mode has no real background; green mode background is replaced by the player anyway).
- No directional shadow casting.
- No automatic detection of the user's home lighting (Quest3 does not expose passthrough camera frames to third-party apps).
- No 3D LUT (`.cube`) import — deferred to Phase 2.

## 4. Current Pipeline Reference

- `pipeline/matting.py` already defines GPU composite kernels:
  - `composite_green_nv12_to_nv12` (`_COMPOSITE_NV12_TO_NV12_KERNEL_SRC`, line ~520)
  - `composite_green_nv12_upsample` (line ~446)
  - `composite_green_upsample` (line ~123)
- `pipeline/alpha_packer.py` handles alpha-mode foreground packing in NV12.
- `config.py:COMPOSITE_BG_RGB_HEX` is the only existing color-related runtime knob.
- Live passthrough goes through `pipeline/pynv_stream.py`, which calls Matter's composite methods every frame.
- UI settings flow: `ui/settings.py::server_env()` → child server process env → `config.py` reads `PT_*`.

## 5. Design

### 5.1 Color math in YUV (NV12) domain

The composite kernels already operate in NV12 limited-range BT.709. To keep cost minimal we apply the correction **directly in YUV** before mixing the foreground into the output NV12:

- **Color temperature / tint**: a 2x2 affine on (U, V) plus a small Y bias. Approximated from a precomputed RGB-domain RGB-gain matrix (computed once when settings change, uploaded as a 12-float `__constant__` buffer), reduced to a YUV-domain matrix using BT.709 conversion constants:

  ```
  Y' = Y * y_gain + y_bias
  U' = (U - 128) * uu + (V - 128) * uv + u_bias + 128
  V' = (U - 128) * vu + (V - 128) * vv + v_bias + 128
  ```

  The host computes these 9 coefficients (`y_gain, y_bias, uu, uv, u_bias, vu, vv, v_bias`) from the user-facing sliders.

- **Exposure (EV)**: multiply Y by `2 ** ev`, folded into `y_gain`.
- **Contrast**: `Y' = (Y - 128) * contrast + 128`, folded into `y_gain, y_bias`.
- **Gamma**: piecewise. Y plane only. Implemented via a 256-entry uint8 LUT uploaded each settings change.
- **Saturation**: `U' = (U - 128) * sat + 128; V' = (V - 128) * sat + 128`, folded into `uu, vv`.

All linear ops collapse into the same 9-coefficient affine. Only gamma needs the LUT branch.

### 5.2 Kernel changes

- Add a new shared device function `apply_light_match(y, u, v, coeffs, gamma_lut)` in the kernel source string.
- Call it inside `composite_green_nv12_to_nv12` and `composite_green_nv12_upsample` immediately **after** sampling the source Y/UV and **before** the alpha blending step. This way the correction is applied to the foreground but not to the green background fill, so the green key still keys cleanly.
- For `alpha_packer.py` NV12 path, apply the same function on the foreground tile only (the alpha-packed corners stay untouched since they store mask data, not color).
- When `light_matching.enabled == false`, host passes the identity coefficient buffer and an identity LUT. We also gate the device-side call behind a single `if (!identity)` check using a uniform argument to avoid even the load cost when disabled.

### 5.3 Coefficient computation (host side)

```
def build_light_match_coeffs(temp_k, tint, exposure_ev, contrast, gamma, saturation):
    # 1. temp_k + tint -> RGB gains (Bradford CAT or simplified Planckian -> sRGB)
    # 2. exposure -> uniform RGB gain *= 2 ** ev
    # 3. RGB gains -> 3x3 RGB transform
    # 4. RGB transform -> 2x2 UV affine + Y gain (BT.709 derivation)
    # 5. contrast/saturation fold into Y/UV affine
    # 6. gamma -> 256-entry uint8 LUT on Y
    return Coeffs(...), gamma_lut_u8
```

Coefficient build is pure Python+numpy, < 1 ms. Triggered only when UI changes a slider, not per-frame.

### 5.4 Configuration surface

| Env var | Default | Range | Purpose |
|---|---:|---|---|
| `PT_LIGHT_MATCH_ENABLED` | `0` | 0/1 | Master switch. |
| `PT_LIGHT_MATCH_TEMP_K` | `5500` | 2700-9000 | Target color temperature in Kelvin. |
| `PT_LIGHT_MATCH_TINT` | `0` | -50..+50 | Green-magenta shift. |
| `PT_LIGHT_MATCH_EXPOSURE_EV` | `0.0` | -2.0..+2.0 | Stops of exposure. |
| `PT_LIGHT_MATCH_CONTRAST` | `1.0` | 0.5..1.5 | Y-domain contrast. |
| `PT_LIGHT_MATCH_GAMMA` | `1.0` | 0.7..1.4 | Y-domain gamma. |
| `PT_LIGHT_MATCH_SATURATION` | `1.0` | 0.0..2.0 | UV-domain saturation. |
| `PT_LIGHT_MATCH_PRESET` | `custom` | `home_warm`/`daylight`/`night_cool`/`custom` | UI preset key, controls which sliders are locked. |

Default behavior: disabled. Existing users see no change.

### 5.5 UI changes (`ui/pages/home_page.py`)

- New collapsible section "Light Matching / 光照匹配" below the existing performance section.
- One toggle (Enabled) + three preset buttons (Home Warm 3000K, Daylight 5500K, Night Cool 6500K) + six sliders.
- "Reset" button restores defaults.
- A small "?" tooltip explaining: "Adjusts the video subject's color to better blend with your room lighting. Has no effect on the room you see through passthrough."
- Translations into `zh_CN`, `en_US`, `ja_JP` keys under `light_match.*`. Files are UTF-8 with BOM (project convention).

### 5.6 Backend wiring

- `config.py`: add 8 new constants read via `_env`/`_env_any`, plus a `LIGHT_MATCH_DICT` snapshot helper.
- `pipeline/matting.py`:
  - `Matter.__init__` builds initial coefficients + gamma LUT, uploads to a CuPy `device_array` of 12 floats and a 256-byte LUT.
  - Add `Matter.update_light_match(...)` for live re-upload (no allocation reuse).
  - Pass the two device pointers into the kernel launch args. Keep them as the last two kernel arguments to minimize source-diff in the existing kernel bodies.
- `pipeline/alpha_packer.py`: same coefficient pointer pattern, applied on the foreground sampling step.
- `pipeline/pynv_stream.py`: read config once at stream start, log effective values, pass to Matter via constructor. No per-frame Python work.

### 5.7 Live reload behavior

Phase 1 ships **without** live reload. Settings are read at stream start. Changing sliders in UI takes effect on the next session. This avoids cross-thread sync work; live reload is Phase 2.

## 6. Performance Budget

- Coefficient prep: < 1 ms on settings change (host CPU).
- Kernel cost per 8K frame:
  - 9 fused multiply-add ops per Y pixel + 4 mads per UV pixel + 1 LUT lookup per Y pixel.
  - Expected delta vs current composite kernel: < 0.4 ms on RTX 2080 at 8K.
- Memory: 48 bytes (12 floats) + 256 bytes (gamma LUT) per stream. Negligible.
- Disabled path: one extra uniform-branch test per kernel launch, no measurable cost.

## 7. Validation Plan

1. **Numerical unit tests** (`tests/test_light_match_coeffs.py`):
   - Identity input (`temp_k=6500, tint=0, ev=0, contrast=1, gamma=1, sat=1`) produces identity coefficients (within float epsilon).
   - Temp 3000K shifts pure white toward warm (R > G > B after applying gains).
   - Saturation 0 collapses U,V to 128.
   - Gamma 0.5 boosts midtones (lookup of 64 > 64).

2. **Kernel reference tests** (`tests/test_light_match_kernel.py`):
   - Run kernel on a 4x4 NV12 fixture with identity coeffs; assert byte-equal output.
   - Run with temp_k=3000; assert output U/V drift in expected direction.

3. **Integration smoke**:
   - Start passthrough with `PT_LIGHT_MATCH_ENABLED=1 PT_LIGHT_MATCH_TEMP_K=3000` on `videos/test_8k.mp4`.
   - Compare 10-second baseline FPS against current (`baseline/auto_tune_8k_phase1_*`).
   - Acceptance: average interval FPS within 1 fps of baseline.

4. **Manual VR check**:
   - On Quest3, side-by-side three sessions (disabled / Home Warm 3000K / Daylight 5500K) on a warm-lit room.
   - User confirms subjective improvement on "cutout feel".

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| YUV-domain affine mis-derived → wrong colors | Low | Unit tests cover identity + 3 known points before kernel ships. |
| Limited-range underflow/overflow at extremes | Medium | Clamp Y to [16,235], UV to [16,240] in kernel. |
| Per-frame perf regression on slower GPUs | Low | Kernel cost is dominated by composite already; new ops are fused. Bench gate at 1 fps tolerance. |
| User confusion (no effect on background) | Medium | Tooltip + preset descriptions explicitly say "applies to video subject only". |
| Default-on regret | N/A | Default off. Existing user has zero behavior change. |

## 9. Out-of-Scope but Worth Logging for Phase 2

- Edge bounce light (foreground edge tinted by configured ambient color) — biggest perceived gain after Phase 1.
- Drop shadow (planar offset alpha blur with user-set direction).
- 3D LUT (`.cube`) import.
- Live reload while a stream is active.
- Color transfer from a user-uploaded reference photo (Reinhard).
- Per-source preset (remember setting per-file or per-directory).

## 10. File-Level Change Summary

| File | Change |
|---|---|
| `config.py` | Add 8 `PT_LIGHT_MATCH_*` constants + helper. |
| `pipeline/light_match.py` (new) | Pure-Python coefficient + LUT builder. |
| `pipeline/matting.py` | Extend NV12 composite kernels with `apply_light_match`; new uploader and updater. |
| `pipeline/alpha_packer.py` | Same `apply_light_match` on foreground tile. |
| `pipeline/pynv_stream.py` | Pass config snapshot to Matter at stream start; log values. |
| `ui/settings.py` | Persist 8 fields; export to `server_env()`. |
| `ui/pages/home_page.py` | New "Light Matching" panel + preset buttons. |
| `ui/translations/{zh_CN,en_US,ja_JP}.json` | New `light_match.*` keys (UTF-8 BOM). |
| `tests/test_light_match_coeffs.py` (new) | Coefficient unit tests. |
| `tests/test_light_match_kernel.py` (new) | Kernel reference tests behind CUDA-availability gate. |
| `summary/summary_20260519_IMPL_PLAN_LIGHT_MATCHING_PHASE1_EN.md` | This document. |
| `summary/summary_20260519_IMPL_PLAN_LIGHT_MATCHING_PHASE1_CN.md` | Chinese counterpart. |
| `prompt/HANDOVER_20260519.md` | Post-implementation handover entry. |

## 11. Acceptance Criteria

- All 4 unit tests pass.
- 60-second 8K green and alpha runs both produce FPS within 1 fps of the pre-change baseline.
- UI presets visibly change the output frame in player.
- Settings persist across UI restart.
- Disabled state (default) is byte-equivalent to pre-change output (verified by hashing first 256 KB of `/passthrough_live` response).

## 12. Estimated Effort

- Coefficient + LUT builder + unit tests: 0.5 day.
- Kernel extension + alpha_packer + pynv_stream wiring: 1 day.
- UI panel + translations + persistence: 0.5 day.
- Manual VR validation + tweak defaults: 0.5 day.
- Total: ~2.5 working days for one developer.
