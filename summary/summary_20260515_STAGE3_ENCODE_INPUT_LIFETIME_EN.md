# Stage 3 Summary - Encode Input Lifetime

## Scope

This stage validated the lifetime of GPU NV12 buffers handed to PyNv/NVENC.

The key question was whether a Matter-owned NV12 output slot can be reused as soon as `Encode()` returns. The answer is no: immediate reuse can corrupt the encoded output.

## Changes

- Added `tools/pynv_encode_lifetime_probe.py`.
- The probe creates deterministic GPU NV12 frames, encodes them with PyNvVideoCodec, optionally overwrites the just-encoded input slot immediately after `Encode()`, decodes the output with FFmpeg, and checks decoded Y values.
- Updated `pipeline/pynv_stream.py` green path so `nv12_slot` is not released immediately after `Encode()`.
- The live green path now keeps recently encoded slots in a pending queue. With default `PASSTHROUGH_NV12_RING_SLOTS=3`, it retains the latest two slots and releases only the oldest slot.
- Pending slots are released after `EndEncode()` and also in the worker `finally` block.

## Findings

- One-slot immediate overwrite after `Encode()` is unsafe.
- CUDA null-stream synchronization after `Encode()` does not make immediate overwrite safe.
- Three-slot delayed reuse passed the current synthetic probes, including 8K HEVC.
- This validates delayed slot release, not removal of the pre-encode `cuda_stream.synchronize()`.

## Validation

```powershell
python -m compileall config.py pipeline\matting.py pipeline\pynv_stream.py tools\pynv_encode_lifetime_probe.py
```

8K HEVC lifetime probe:

```powershell
uv run python tools\pynv_encode_lifetime_probe.py --width 8192 --height 4096 --frames 24 --codec hevc --bitrate 50000000 --gop 60 --progress 6 --slots 3
```

Result:

- report: `baseline/pynv_encode_lifetime_stage3_20260515_111503_740948.md`;
- `ok=True`;
- corrupt/unknown frames: `0`;
- encode FPS: `36.48`.

Live smoke tests:

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 1 --startup-timeout 240 --client-timeout 60
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer alpha --duration 1 --startup-timeout 240 --client-timeout 60
```

Reports:

- green: `baseline/auto_tune_8k_phase1_20260515_111516.md`;
- alpha: `baseline/auto_tune_8k_phase1_20260515_111532.md`.

Both smoke tests passed.

## Baseline Impact

This stage should not be expected to clearly exceed the Phase 1 baseline. It is a correctness and safety stage.

Short smoke numbers were not worse than baseline, but they are not a strict comparison because the smoke duration was only one second:

- green latest interval FPS: `36.64`;
- alpha latest interval FPS: `35.96`;
- Phase 1 baseline: average interval FPS `34.32`, latest interval FPS `35.58`.

For a strict comparison, run:

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer alpha --duration 60 --startup-timeout 240 --client-timeout 90
```

## Risks Before The Next Stage

- Do not delete `cuda_stream.synchronize()` directly in Phase 4. Stage 3 proved input-slot lifetime and reuse distance; it did not prove that NVENC waits for Matter's `_CUDA_STREAM` composite kernels.
- Treat `Encode()` return as insufficient proof that the GPU input memory can be overwritten.
- Re-run the lifetime probe if GOP, B-frame settings, codec, resolution, driver, or PyNvVideoCodec version changes.
- The green path uses the Matter NV12 slot pool. The alpha packer path does not use that pool, so green-slot conclusions must not be blindly applied to alpha-internal buffers.

## Decision

Proceed to Stage 4 only with delayed slot release preserved. The next meaningful performance work is the staged decode / matting / encode pipeline, first behind the offline `tools/pynv_fullchain_probe.py --pipeline staged` gate.
