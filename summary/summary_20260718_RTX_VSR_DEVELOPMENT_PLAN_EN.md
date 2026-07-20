# NVIDIA RTX VSR High-Resolution Feature Development Plan

Date: 2026-07-18  
Status: Pre-development plan; no feature code implemented in this pass.

## Objective

Add NVIDIA RTX Video Super Resolution for ordinary 2D video, using a conservative initial project policy of 360p–1440p for realtime and offline enhancement. The Programming Guide does not state this as a hard SDK limit. Outputs carry `[SuperRes]`; out-of-policy and VR inputs remain on the existing path pending POC validation.

## Findings and architecture

- The SDK is under `reference/RTX_Video_SDK_v1.1.0`; production code must not depend on that path.
- VSR quality levels are Bicubic(0), Low(1), Medium(2), High(3), and Ultra(4). Expose the complete 0–4 enum, with 0 explicitly representing the Bicubic baseline.
- Initially gate input resolution to 360p–1440p as a project policy, then validate actual SDK behavior from 360p through 2160p and on non-16:9 inputs.
- The current Python realtime path is PyNvVideoCodec/CuPy/NVENC and the offline path is FFmpeg subprocess based. There is no existing Python NGX binding, so a Windows x64 native bridge is required (CUDA interop first, DX11 fallback if resource sharing requires it).
- VSR is spatial upscaling, not encoding. Existing NVENC/FFmpeg mux, audio, timestamps, and player compatibility remain responsible for output packaging.

## Proposed configuration

Add documented `PT_RTX_VSR_*` settings for global/realtime/offline enablement, quality, 360p minimum and 1440p maximum input resolution, source policy (`auto` or `2d`), capability-constrained 2D target, maximum output pixels, realtime queue depth and latency budget, fallback (`passthrough`/`error`), `[SuperRes]` prefix, and packaged SDK runtime directory.

## Implementation phases

1. Keep all RTX VSR assets under `models/rtx_vsr`: native sources/headers/libs in `native`, and distributable DLL/license/version assets in `runtime`. Prebuild the bridge on the developer/release machine; end users do not compile C++ when installing the packaged application.
2. Add a realtime VSR stage after decode and before existing composition/encode, keeping frames on GPU. Hard-reject VR markers, equirectangular/2:1 sources, and VR modes; gate ordinary 2D by target size, VRAM, FPS, and latency, then fall back according to configuration.
3. Add an offline converter/engine with single, batch, and segment modes. Use GPU decode/bridge/VSR/NVENC and existing audio/mux rules. Integrate `[SuperRes]` naming and skip-existing behavior.
4. Extend UI/runtime controls, diagnostics, and translations for enablement, target, quality, fallback, capability, and progress.
5. Update `build_exe.py`, both PyInstaller specs, and runtime path resolution so the bridge and SDK runtime are copied into the final package. Do not ship samples, debug/dev DLLs, ARM64 assets, or rely on `reference`.

## Tests and acceptance

Cover configuration/classification/naming, 360p–1440p gating, VR and out-of-range rejection, bridge failure modes and quality 1–4, NV12/P010/RGBA formats, realtime 2D FPS/latency/VRAM/audio sync, offline long/batch/segment/no-audio/10-bit cases, and clean-machine packaged execution. Enabled output must reach the selected target and carry `[SuperRes]`.

## Risks and prerequisites

The main risk is NGX/CUDA or DX11 resource interop from the existing Python GPU pipeline. First milestone is therefore a native proof-of-concept that loads the release DLL, reports capability, and processes one ordinary 2D GPU frame. Confirm exact input/output limits, minimum driver, NGX initialization requirements, and SDK redistribution terms before production integration.
