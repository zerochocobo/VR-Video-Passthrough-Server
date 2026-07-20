# RTX Video SDK / VSR Official Scope Review

> Note: The 360p–1440p wording in this earlier review is superseded by `summary_20260718_RTX_VSR_PROGRAMMING_GUIDE_REVIEW_EN.md`; it is an initial project policy, not an explicit hard limit in the Programming Guide.

Date: 2026-07-18

## Conclusion

For this project, RTX VSR should be officially supported only for ordinary 2D video frames whose input resolution is within 360p–1440p. Inputs below 360p or above 1440p do not enter VSR; this excludes 4K 2D, 4K VR full frames, and high-resolution per-eye crops.

This is not because NGX exposes a literal “2D-only” flag. The API accepts resources, source/destination rectangles, and a quality level, but no VR layout, eye, or projection metadata. The SDK samples process conventional YUV/RGB video frames and contain no SBS/OU, per-eye, equirectangular, or spherical processing.

Therefore the accurate product statement is: **RTX VSR is for ordinary 2D realtime/offline enhancement in the 360p–1440p input range; VR and out-of-range inputs remain on the existing path. Outputs use `[SuperRes]`.**

## Evidence from the bundled SDK

- `include/nvsdk_ngx_defs_vsr.h`: quality levels and availability/driver/feature parameters only; no VR parameters.
- `include/nvsdk_ngx_helpers_vsr.h`: DX11/DX12/CUDA evaluation consists of input/output resources, rectangles, and quality; this is a generic frame interface, not a VR support declaration.
- `samples/ReadMe.md` and the VSR samples: YUV/RGB data files, normal display size/FPS/quality controls, no VR geometry or eye handling.
- DX11/DX12 samples convert video input with a video processor and run VSR on ordinary source/destination rectangles.

The Programming Guide remains the authority for exact resolution and driver limits. PDF text extraction is unavailable in the current environment, so numeric limits must be confirmed during the POC from the guide and runtime `VSR.Available` / `VSR.NeedsUpdatedDriver` checks rather than inferred from sample defaults.

## Plan corrections

- Remove realtime 4K VR VSR from the scope and configuration matrix.
- Restrict `PT_RTX_VSR_SOURCE_POLICY` to `2d`/`auto`; `auto` must reject VR markers, 2:1/equirectangular sources, and VR modes with an `unsupported_vr_source` diagnostic.
- Remove `PT_RTX_VSR_4K_VR_TARGET`.
- Keep a 2D target setting, but constrain it by the Programming Guide and runtime capability; do not hard-code 4K/8K before the POC.
- Add 360p minimum and 1440p maximum input gating, with `unsupported_input_resolution` diagnostics.
- Use `[SuperRes]` only on in-range ordinary 2D RTX VSR outputs.
