# Offline SuperRes TrueHDR (HDR10) implementation notes

Date: 2026-09-11
Branch: `DLSS`
Status: implemented and verified on hardware

Background research (Chinese): [summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md](summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md). This is item 1 of that report's recommended order of work.

---

## 1. What was built

Offline SuperRes gains a fourth "HDR look": **TrueHDR**, which runs NVIDIA RTX Video's NGX TrueHDR feature and produces real HDR10 (HEVC Main 10, BT.2020/PQ) instead of the existing 8-bit SDR grade.

The key precondition: the native bridge `models/rtx_vsr/native/rtx_video_api_cuda_impl.cpp` **already implemented TrueHDR** (create, evaluate, VSR→THDR chaining, the intermediate texture). Python simply called `rtx_video_api_cuda_create(..., THDREnable=0, VSREnable=1)` and kept it switched off. So the native bridge was **not rebuilt**; only the runtime asset and the Python/CUDA/UI path were added.

---

## 2. Changes

### Assets and packaging

| File | Change |
|---|---|
| `models/rtx_vsr/runtime/nvngx_truehdr.dll` | Copied from `reference/RTX_Video_SDK_v1.1.0/bin/Windows/x64/rel/` (same redistribution licence as `nvngx_vsr.dll`) |
| `build_exe.py` | `verify_rtx_vsr_runtime()` now requires `nvngx_truehdr.dll` |

No packaging changes were needed otherwise: `build_exe.py` and both `.spec` files ship `models\rtx_vsr\runtime` as a **whole directory**, so the new DLL is picked up automatically, and the new `pipeline/true_hdr.py` is covered by `collect_submodules('pipeline')`. No new third-party dependency was introduced.

### Core path

| File | Change |
|---|---|
| `pipeline/true_hdr.py` (new) | `abgr10_to_p010` CUDA kernel, HDR10 colour metadata, `is_true_hdr()`, `p010_grid()` |
| `pipeline/pynv_io.py` | New `GpuP010AppFrame` and `_RawCudaPlaneView` (PyNv only accepts the `\|u2` typestr and its own P010 plane layout; CuPy's descriptor is rejected) |
| `utils/rtx_vsr.py` | `TrueHdrSettings` dataclass clamped to the SDK ranges; `true_hdr` on `initialize()`/`initialize_cupy()`; the flag is part of the context key so switching modes tears the native session down and rebuilds it; `evaluate_deviceptr()` passes the THDR controls and refuses a mismatched mode; `process_cupy_rgba()` gained `out=` reuse and the packed `uint32` HxW TrueHDR output |
| `pipeline/hdr_look.py` | `truehdr` added to the mode set and is a no-op for the SDR kernel |
| `offline/rtx_vsr_pynv.py` | TrueHDR branch: packed output buffers, P010 ring, `P010` encoder, HDR10 colour args; per-eye output buffers are now allocated once per session |
| `offline/rtx_vsr.py` | `hdr_settings` threaded through; the FFmpeg rawvideo fallback **refuses** TrueHDR (rc=3) instead of silently writing SDR |
| `offline/convert.py` | CLI `--rtx-vsr-hdr-look truehdr` plus four `--rtx-vsr-truehdr-*` controls |
| `pipeline/pynv_stream.py` | Realtime logs a warning for `truehdr` and keeps the SDR look (realtime still delivers 8-bit NV12) |
| `config.py` | `PT_RTX_VSR_TRUEHDR_CONTRAST/SATURATION/MIDDLE_GRAY/MAX_NITS` |

### UI

| File | Change |
|---|---|
| `ui/pages/superres_page.py` | TrueHDR in the HDR-look combo; a new "TrueHDR controls" row (contrast 0–200, saturation 0–200, middle gray 10–100, peak nits 400–2000) shared by the single and batch tabs, shown only when TrueHDR is selected (the row is hidden and collapses otherwise), and appended to the CLI arguments |
| `ui/settings.py` | Four new settings persisted in `config.ini` |
| `ui/translations/*.json` | Five new keys in all three languages |
| `README*.md`, `PROJECT.md`, `CHANGELOG.md` | Documentation synced; PROJECT.md previously stated TrueHDR/HDR10 "is not a current product feature" and was rewritten |

The realtime SuperRes dialog (`ui/dialogs/feature_dialogs.py`) deliberately does **not** offer TrueHDR: the realtime chain delivers 8-bit NV12, and HDR10 there would need a separate P010 delivery path and client compatibility work.

---

## 3. Findings that came from measurement, not assumption

### TrueHDR output format

The SDK only says "10 bit ABGR10". Measured with solid-colour patches:

- **Bit order**: `bits[0:10]=R, [10:20]=G, [20:30]=B, [30:32]=A`.
- **Range**: pure black comes out as 0, so the output is **full range** (limited range would be 64).
- **Primaries**: sRGB pure red produces `R=562 G=331 B=193` — clearly non-zero green and blue, so **BT.2020**, not BT.709.
- **Transfer**: at the default 1000-nit setting pure white lands at 768/1023 ≈ PQ 0.751 ≈ 1000 nits, so **SMPTE 2084**. Setting 400 nits drops white to 665 and 2000 nits raises it to 769, confirming the controls reach NGX.

The kernel therefore treats the input as full-range PQ BT.2020 RGB and writes limited-range BT.2020 non-constant-luminance P010, tagged `bt2020 / smpte2084 / bt2020nc / tv`. Cross-checked against a NumPy reference: **max error 0**.

### Two NVENC traps

- The `Pixel_Format` enum spells it `P016`, but `CreateEncoder` needs the string **`"P010"`** (`P016` fails with "Unknown format").
- PyNv rejects CuPy's `__cuda_array_interface__` (`Invalid typestr: <u2`). Planes must match the SDK sample: Y `(H, W, 1)`, UV `(H/2, W/2, 2)`, strides `(W*2, 2, 1)`, typestr `|u2`.

### Mode switching

TrueHDR is chosen when the NGX feature is created, so switching between SDR and HDR in one process requires a shutdown and rebuild. The bridge's context key covers this automatically, and `evaluate` raises on a mismatch rather than emitting data in the wrong format.

---

## 4. Measured performance (RTX 5060 Ti)

| Case | Input → output | Quality | Processing FPS |
|---|---|---|---|
| 2D | 1280×720 → 3840×2160 | High (3) | **93.7** |
| Split-eye SBS VR | 3840×1920 → 8192×4096 | High (3) | **15.6** |

The 8K VR baseline for VSR alone is 23–24 FPS, so the second NGX evaluation costs roughly 35%. That is fine offline and reinforces keeping it out of the realtime chain.

Output verified with ffprobe: `hevc / Main 10 / yuv420p10le / bt2020 / smpte2084 / bt2020nc / tv`, frame count matching the source, eyes rejoined without offset.

---

## 5. Tests

- New `tests/test_true_hdr.py` (8 cases): mode recognition, HDR10 metadata, grid maths, control clamping, bridge create/switch/missing-DLL/mismatched-mode, control pass-through.
- New `tests/test_ui_smoke.py::test_superres_page_exposes_true_hdr_controls_only_for_truehdr`: controls shown only for TrueHDR, values shared across tabs, CLI arguments appended correctly.
- Full suite: `642 passed, 2 skipped`.

---

## 6. Known limits

- **Offline only.** Realtime SuperRes logs a warning and keeps the SDR look.
- **Not supported on the FFmpeg fallback**, which refuses (rc=3) rather than degrading silently.
- No mastering-display / MaxCLL SEI is written; HDR10 is signalled at the container and VUI level only. That is enough for most players; a `hevc_metadata` bitstream filter would be needed for in-stream SEI and was left out because of ffmpeg version variance.
- Sources are still 8-bit only; 10-bit input is rejected by the existing gate, unchanged by this work.
