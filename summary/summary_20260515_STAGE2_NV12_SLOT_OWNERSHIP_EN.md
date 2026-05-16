# Stage 2 Summary - NV12 Slot Ownership Foundation

## Scope

This stage introduced explicit GPU NV12 output slot ownership for the green
passthrough composite path.

The goal was to stop treating Matter's GPU NV12 output as a single implicit
scratch buffer and prepare the codebase for later safe encode/pipeline overlap.

## Changes

- Added `PT_PASSTHROUGH_NV12_RING_SLOTS`, default `3`.
- Added Matter-side slot pool APIs:
  - `Nv12OutputSlot`;
  - `Matter.acquire_nv12_output_slot(h, w)`;
  - `Matter.release_nv12_output_slot(slot)`.
- Updated green GPU composite functions so callers can pass an explicit output
  slot buffer.
- Updated the live green composite path to:
  - acquire a slot before composite;
  - pass the slot into Matter composite;
  - release it after `Encode()` returns via `finally`.

## Validation

```powershell
python -m compileall config.py pipeline\matting.py pipeline\pynv_stream.py
```

Green smoke:

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 1 --startup-timeout 240 --client-timeout 60
```

Alpha smoke:

```powershell
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer alpha --duration 1 --startup-timeout 240 --client-timeout 60
```

Both smoke tests passed.

## Baseline Impact

This stage should not be expected to exceed the Phase 1 performance baseline.

Reason: the existing CUDA stream synchronization before NVENC encode is still
present. The ring slot pool solves output-buffer ownership and reuse, but it
does not by itself prove that NVENC waits for composite kernels on
`_CUDA_STREAM`.

This stage is a safety foundation, not the main FPS improvement.

## Risks And Required Next Step

The key remaining risk is NVENC input lifetime.

If `Encode()` returns only after NVENC has consumed/copied the input frame, the
slot can be safely reused after `Encode()` returns. If `Encode()` keeps a pointer
to the input asynchronously, reusing or overwriting the slot can corrupt encoded
output.

The PyNv encoder API available locally does not expose an obvious CUDA event or
stream-wait handoff. Therefore the next stage must run an encode-input lifetime
probe before removing the per-frame sync.

Recommended probe:

```text
composite into slot A
do not synchronize
Encode(slot A)
immediately overwrite slot A after Encode returns
encode more frames
decode the encoded output
check whether the frame encoded from slot A was corrupted
```

If the output remains clean, later stages can consider removing or reducing the
sync. If corruption appears, the sync must remain or a proper CUDA event/stream
handoff must be found.

## Decision

Proceed to the next stage only after validating encode-input lifetime. Do not
blindly remove `cuda_stream.synchronize()`.
