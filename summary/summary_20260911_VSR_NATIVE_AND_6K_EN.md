# VSR native 1x enhancement + 6K middle target — implementation notes

Date: 2026-09-11
Branch: `DLSS`
Status: implemented and measured on hardware

Items 2 and 3 of the [research report](summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md) (Chinese). Item 1, TrueHDR, is in [summary_20260911_TRUEHDR_OFFLINE_SUPERRES_EN.md](summary_20260911_TRUEHDR_OFFLINE_SUPERRES_EN.md).

---

## 1. The finding that matters: NGX VSR cost follows the input, not the output

This measurement overturns the research report's assumption that, being compute-bound, frame rate would fall roughly with output pixels and a 6K target would reach ~40 FPS.

RTX 5060 Ti, quality 3, same footage, per-eye NGX time from `PT_RTX_VSR_STAGE_TIMING=1`:

| Input (per eye) | Output | NGX ms/eye | End-to-end FPS |
|---|---|---|---|
| 1920×1920 | 8192×4096 | 17.2 | 25.3 |
| 1920×1920 | 6144×3072 | 16.1 | 27.2 |
| 1920×1920 | 3840×1920 (native 1x) | 15.1 | 30.4 |
| **960×960** | 8192×4096 | **6.2** | 34.9 |

Dropping the output from 8K to 6K is −44% pixels but only −7% time; quartering the input cuts the time to about a third. **The input decides the cost.**

Consequences:

- A 4K VR source (1920×1920 per eye) costs roughly 17 ms × 2 eyes ≈ 34 ms per frame no matter how large the output is — an **upper bound near 29 FPS**. Realtime 60 FPS is only reachable by lowering the input.
- The 6K target is not a performance step. Its value is that the output is **56% of 8K's pixels**, which cuts bitrate and headset decode load — useful for devices that struggle with 8K HEVC.
- Native 1x saves little NGX time on VR (15.1 vs 17.2 ms); what it saves is output size. On 2D it is genuinely fast (below).

---

## 2. What was built

### Native 1x enhancement (target = 0)

VSR evaluates at the source resolution without enlarging. It is a real effect: on low-bitrate 720p footage, hair, edges and contours come out visibly cleaner. Note that PSNR against the clean master *drops* (40.5 → 39.9 dB), which is what sharpening does — so this must not be described as fidelity-preserving artefact removal.

Gating change: the upscale input ceiling (`PT_RTX_VSR_INPUT_MAX_HEIGHT`, default 1440) does **not** apply at 1x, because 1x costs what the source costs. Native uses its own ceiling, `PT_RTX_VSR_NATIVE_MAX_HEIGHT` (default 4096), plus NVENC's 8192-pixel side limit. So 4K 2D and 2160p vertical footage — previously rejected — can now be processed.

### 6K middle target (target = 3072)

VR sources render 6144×3072 (3072×3072 per eye); ordinary 2D converges to 3840×2160 exactly as the 4096 target does.

---

## 3. Changes

| File | Change |
|---|---|
| `utils/rtx_vsr.py` | `NATIVE_TARGET_HEIGHT=0`, `ENCODER_MAX_SIDE=8192`, `resolve_target_height()`, `is_native_target()`; 6K and native branches in `target_resolution()`; native support in `target_dimensions()`, `source_exceeds_target_resolution()`, `effective_offline_target_height()` and `source_block_reason()` |
| `config.py` | `PT_RTX_VSR_TARGET_HEIGHT` now accepts 0 (the old `max(2, …)` turned 0 into 2); new `PT_RTX_VSR_NATIVE_MAX_HEIGHT` |
| `utils/vr_naming.py` | `_1X` and `_6K` output suffixes, including the strip-and-replace regex |
| `offline/convert.py` | Target parsing via `resolve_target_height()` (the `or` idiom swallowed 0); batch discovery skips `_1x`/`_6k` |
| `offline/rtx_vsr.py` | Same; 6K joins 8K in disabling the whole-frame FFmpeg fallback, since VR targets need the split-eye GPU path |
| `offline/rtx_vsr_pynv.py` | VR sources are split per eye at 1x as well |
| `pipeline/pynv_stream.py` | Same for realtime |
| `dlna/content_directory.py` | Directory visibility uses the native ceiling when the target is 1x |
| `ui/superres_targets.py` (new) | Single source of truth for the choices and their label keys, shared by three UI sites |
| `ui/pages/superres_page.py`, `ui/dialogs/feature_dialogs.py`, `ui/pages/dashboard_page.py` | Target combo grows from 3 to 5 entries (native / 2K / 4K / 6K VR / 8K VR); the dashboard summary uses the same mapping |
| `ui/translations/*.json` | `superres.target_native` and `superres.target_6k_vr` in all three languages |

### One trap worth naming

`0` is a legitimate target height (native), but the codebase was full of `int(target_height or config.RTX_VSR_TARGET_HEIGHT)`, which silently replaces it with the default. Every such site now calls `resolve_target_height()`, and the constraint is documented at the top of `utils/rtx_vsr.py`. The UI's `int_setting()` / `_setting_value()` test `is None or == ""`, so they were already safe for 0.

---

## 4. Measurements

RTX 5060 Ti, quality 3, 8-second clips, full chain (NVDEC → VSR → NVENC → mux):

| Case | Input → output | FPS |
|---|---|---|
| 2D native 1x | 1216×2160 → same | **75.3** |
| VR native 1x | 3840×1920 → same (1920×1920 per eye) | **30.4** |
| VR 6K | 3840×1920 → 6144×3072 | **27.2** |
| VR 8K (control) | 3840×1920 → 8192×4096 | **25.3** |

Frame checks: native VR output rejoins the eyes without offset; 2D 1x is visibly sharper than the source.

Naming: `[SuperRes]<name>_1X.mp4` / `_6K.mp4`; re-running replaces an existing marker and batch discovery skips both.

---

## 5. Tests

- `tests/test_rtx_vsr.py`, six new cases: native survives the `or` idiom, native keeps the source size, 6K is VR-only with 2D falling back to 4K, the native gate replaces the upscale gate (including the encoder side limit), `_1X`/`_6K` naming, and UI choices mapping one-to-one onto label keys.
- `tests/test_offline_convert.py`: batch discovery skips `_1X`/`_2K`/`_4K`/`_6K`/`_8K` outputs.
- Full suite: `648 passed, 2 skipped`.

---

## 6. Limits and follow-ups

- Realtime VR is bounded by roughly 17 ms per eye for a 4K VR source, so **the 6K target will not turn realtime VR SuperRes into 60 FPS**. The only lever for realtime frame rate is a smaller NGX input (e.g. upscaling from a 2K VR source), or accepting 30 FPS.
- 8K VR sources stay hidden from realtime (the DLNA `width > 4096` rule is unchanged); offline 1x allows up to the 8192 side limit.
- Native 1x is sharpening, not fidelity restoration; clean sources gain little.


---

## 7. Follow-up: SuperRes on the virtual-file (seekable) route

### What was actually in the way

Not just a wrong resolution in the shell. The frame generator opened with:

```python
if mode not in {"green", "alpha"}:
    raise RuntimeError(f"slot PyNv GOP builder does not support output_mode={mode!r}")
```

so SuperRes over the virtual-file route **could not run at all** — it raised on the first frame. Separately, `_vmp4_slot_output_size()` had branches for `alpha` and `two_dvr` but none for `superres`, so the shell declared the source resolution. Both needed fixing.

The byte budget was not the obstacle: that layer already special-cased superres (dropping the source-inherited budget for the flat one), and 8K overshoot was solved earlier for alpha's 8K VR with an adaptive headroom loop. Measured over 240 frames, **no frame exceeded its budget**.

### Approach

1. The VSR work moved into `pipeline/superres_stage.py`, shared by the live worker and the virtual-file frame generator — one implementation of eye splitting, buffer reuse, the NVENC input ring and the SDR look.
2. The frame generator accepts `superres` and takes its output size from the stage.
3. `_vmp4_slot_output_size()` gained the superres branch and asks the same function the stage does, so the shell always declares what is encoded.
4. The seek route calls `seek_target_height()` instead of the configured target: native 1x by default.

### Why it falls back to 1x

This route is **pulled at playback speed**, so the stage has to keep up with the player. NGX cost follows the input, so enlarging barely changes it — a 4K VR source runs 25–27 FPS whether the output is 6K or 8K, short of a 60 fps title. Native 1x at least removes the output size and encoding pressure.

The escape hatch `PT_RTX_VSR_SEEK_ALLOW_UPSCALE=1` keeps the configured target but **has not been verified against a real player**.

### Measured (RTX 5060 Ti, virtual-file route, native 1x, default quality)

| Source | Output | FPS | Over-budget frames |
|---|---|---|---|
| 1216×2160 2D | same | 37.4 | 0 / 240 |
| 3840×1920 SBS VR | same | 27.2 | 0 / 240 |

Slower than offline on the same footage (75 FPS for 2D, 30 FPS for VR) because this route yields frame by frame and splits Annex-B access units, without the offline pipeline's overlap.

**Meaning**: for 30 fps titles 2D is comfortable and VR is tight (27 < 30, covered by the background pre-generation buffer); for 60 fps titles neither keeps up. This comes from throughput numbers — **real player seeking is still unverified**.

### Not done

- No new UI switch: virtual file vs live stream is still the single global choice shared with Alpha and green screen.
- Enlarging over the virtual-file route is unverified (unreachable by default).
- Real player scrubbing, jumping and remaining-time display are unverified.

### Performance hunt: one real bug fixed, the main cause still unidentified

The virtual-file route runs about 1.9x slower than offline on the same footage and settings (75.7 vs 39.9 FPS). What was found:

1. **Per-frame NGX rebuild (fixed).** The bridge keyed its session on the CUDA stream as well as the context; callers running inside a CuPy stream context hand it a different stream pointer every frame, so it tore NGX down and rebuilt it per frame — **241 native creates** across a 240-frame run. Keying on the CUDA context alone brings that to **1**. Offline is unchanged (75.3 → 75.7); the seek route went 37.4 → 39.9 FPS (**+7%**). The same bug affected the live path, which should benefit too (not separately measured).
2. **The CBR per-frame budget is not the cause.** Disabling it (back to VBR) moved 38.5 → 41.2 FPS, also about 7%.
3. **The profile's headline reading is an artefact.** Both cProfile and perf_counter attribute ~3.9 s to `initialize`, but an isolated benchmark measures it at **0.002 ms/call** — that time is scheduling/wait charged to the function, not its own work.

**The remaining ~1.9x is not explained.** The next step is to break the route down with CUDA events rather than CPU timers, to establish whether the GPU side is genuinely slower or the per-frame yield and access-unit splitting serialize the pipeline. No cause should be claimed before that measurement exists.

### What quality levels cost (measured, offline route, 4K VR source at native 1x)

| Quality | NGX ms/eye | End-to-end FPS |
|---|---|---|
| 2 Medium | 6.5 | **62.2** |
| 3 High | 15.2 | 30.3 |
| 4 Ultra | 20.8 | 22.5 |

**Medium to High doubles the cost; High to Ultra adds another 35%.** Offline batch work that does not need the last bit of quality can double its throughput by choosing Medium.

The same setting barely moves the realtime and virtual-file routes: on a 4K VR source quality 1 gives 29.8 FPS and quality 2 gives 27.6 FPS, 7% apart, where offline at quality 2 reaches 62 FPS. So NGX is not what bounds those two routes. Ruled out so far: the decoder (simple vs threaded_serial, 7%) and the CBR per-frame budget (disabling it, 7%). **The bottleneck is still unidentified**; the next step is a CUDA-event breakdown rather than more CPU-timer guesses.

### Measuring the virtual-file route with enlargement (and one real bug)

With `PT_RTX_VSR_SEEK_ALLOW_UPSCALE=1` at 6K, the first attempt **would not play**:

```
Invalid NAL unit size (267026 > 262140)
server log: VMP4 frames oversize idx=3 267030>262144 (clamped)
```

The 256 KiB per-frame budget is tuned for Green/Alpha, whose output is either the source geometry or a matte-packed picture — both cheap to code. Enlarged SuperRes is far denser; frame 3 overran and was truncated, and the adaptive headroom loop reacts per GOP, too late for the opening frames.

Fix: scale the budget with output pixels above the 4K reference, **for SuperRes only**, capped at 4x. Measured after the change:

| Case | Per-frame budget | Bandwidth @30fps | Result |
|---|---|---|---|
| Green/Alpha, any size | 256 KiB | 63 Mbps | unchanged |
| SuperRes native 1x (3840×1920) | 256 KiB | 63 Mbps | 240 frames decode clean |
| SuperRes 6K (6144×3072) | 704 KiB | 173 Mbps | 240 frames decode clean, no oversize |

The 6K output rejoins the eyes correctly and mid-file ranges land on the right GOP. But it costs 2.7x the bandwidth and the server still only produces about 27 FPS, so `PT_RTX_VSR_SEEK_ALLOW_UPSCALE` stays off by default — working is not the same as advisable.

### Final design: the target decides the playback shape

The first implementation rendered native 1x on the virtual-file route whatever target was configured, so choosing 6K still played 1x. That is a silent downgrade, and labelling the entry `[SUPERRES 1X]` does not fix it either: players such as Skybox cache item titles, so a name that changes later never reaches the user.

The shape now follows the target:

| SuperRes target | DLNA entry shape | Playback mode selectable |
|---|---|---|
| Native 1x | virtual file or live directory | yes, sharing Alpha's global choice |
| 2K / 4K / 6K / 8K | live chapter directory only | no, the dialog explains why |

- `seek_target_height()` no longer substitutes anything; `seek_supported_target()` decides whether the seek shape is offered at all.
- The browse side publishes a seek entry only for the native target, and the seek route answers 409 for a target it does not offer rather than half-serving something else.
- The realtime SuperRes dialog gained Alpha's `PlaybackModeChooser`, shown only for the native target.
- Entry titles stay `[SUPERRES]`.

Measured against a running server with a 4K SBS VR source: at the 6K target the seek URL returns `409 SuperRes virtual-file playback requires the native 1x target`; at 1x it returns 200, output 3840×1920, and all 240 frames decode clean.

### Defaults and the in-dialog rule

- The realtime SuperRes target now defaults to **native 1x** (`superres_target_height` in `ui/settings.py` and `PT_RTX_VSR_TARGET_HEIGHT` in `config.py`), since it is the only target that can also be offered as a virtual file.
- Offline keeps the 8K VR default: it has no playback-speed constraint, and enlarging is the point of it. The two defaults are now separate constants in `ui/superres_targets.py` (`REALTIME_DEFAULT_TARGET` / `OFFLINE_DEFAULT_TARGET`).
- The note "only native enhance can be played as a virtual file" is **permanent at every target**, not shown only once the chooser disappears: it is the rule that decides whether the chooser exists, so it should be readable before the target is chosen.
- Existing user settings are untouched: only the default changed, so anyone who already picked a target keeps it.
