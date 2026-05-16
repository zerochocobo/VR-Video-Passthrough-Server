> ⚠️ SUPERSEDED 2026-05-16
> FPS, sync, and bottleneck-attribution conclusions in this report were produced under the old default PT_PASSTHROUGH_MAX_FPS=30 and/or non-target PT_ALPHA_STRIDE=3 conditions. Later review invalidated them or limited them to those diagnostic conditions only.
> Use summary/summary_20260516_STAGE4R_FPS_CAP_DISCOVERY_EN.md as the corrected baseline entry point.
> This file is retained only as a research-process archive; implementation conclusions unrelated to the FPS cap must be read with their stated scope.
# Stage 4R Summary - ThreadedDecoder Serial Integration and Two-stage Validation

## Background

This stage continued from the expert advice in `summary/summary_20260516_STAGE4_PYNV_THREADED_STAGED_CRASH_ADVICE_EN.md`. The original three-stage ThreadedDecoder staged pipeline remains unsafe because PyNv ThreadedDecoder frames must not be queued across worker threads while the decode worker continues calling `get_batch_frames()`.

The replacement plan was lower risk: validate sequential ThreadedDecoder CFR decimation, integrate it into the existing production serial worker, then try a green-only two-stage design that passes only Matter-owned NV12 slots across threads.

## Completed Work

- Added `PyNvThreadedSerialDecoder` in `pipeline/pynv_io.py`.
- Added decoder selection:
  - `PT_PASSTHROUGH_PYNV_DECODER=threaded_serial` by default;
  - `PT_PASSTHROUGH_PYNV_DECODER=simple` as rollback.
- Fixed `tools/auto_tune_8k.py` so the spawned server inherits development CUDA/cuDNN DLL paths through `base_environment()`.
- Added timeout-aware Matter NV12 slot acquisition in `pipeline/matting.py`.
- Added experimental green-only two-stage worker behind `PT_PASSTHROUGH_PYNV_WORKER_MODE=two_stage`.

## Results

Threaded decode-only probe:

- Command: `uv run python tools\pynv_threaded_decode_probe.py videos\test_8k_2.mp4 --frames 300 --fps 30 --batch-size 8 --buffer-size 32 --hash-frames 20`
- Report: `baseline/pynv_threaded_decode_phase2_20260516_150052.md`
- Result: `selected_fps=86.24`, `source_fetch_fps=171.91`, `hash_compare.ok=True`.

Threaded serial green 10s:

- Report: `baseline/auto_tune_8k_phase1_20260516_150735.md`
- Latest interval FPS: `36.43`
- Average interval FPS: `36.02`
- Decode: `0.07 ms`
- Composite: `15.08 ms`
- Sync: `11.43 ms`
- Slow mux warnings: `0`

SimpleDecoder control under the same harness:

- Report: `baseline/auto_tune_8k_phase1_20260516_150912.md`
- Latest interval FPS: `36.55`
- Average interval FPS: `35.94`
- Decode: `18.26 ms`
- Composite: `6.90 ms`
- Sync: `1.29 ms`

Alpha smoke:

- Report: `baseline/auto_tune_8k_phase1_20260516_150832.md`
- Latest interval FPS: `34.99`
- Decode: `0.06 ms`
- No worker exception.

Two-stage green 10s:

- Report: `baseline/auto_tune_8k_phase1_20260516_151600.md`
- Latest interval FPS: `36.32`
- Average interval FPS: `35.95`
- Decode: `0.13 ms`
- Composite: `18.31 ms`
- Sync: `9.03 ms`
- Encode: `0.75 ms`
- No slot timeout or worker exception.

## Conclusions

- ThreadedDecoder serial integration is safe and frame mapping is stable.
- Decode cost was effectively removed from the production loop.
- Overall FPS did not pass 40 because the bottleneck moved to composite/sync/RVM CUDA execution.
- A simple Python two-stage split does not improve throughput; the critical GPU work is still serialized by stream synchronization and shared CUDA execution.
- The experimental two-stage path is safe enough to keep behind a flag, but it should not become the default.

## Risks

- `PyNvThreadedSerialDecoder` supports monotonic source indices only; backward seek requires a new decoder instance.
- ThreadedDecoder decoded frames must still never cross worker boundaries.
- The two-stage path currently applies only to green output. Alpha needs a separate ownership/lifetime design.
- Further FPS gains require CUDA event/stream work, CUDA Graph, TensorRT EP, or RVM/composite restructuring.

## Recommended Next Step

Stop adding Python pipeline stages for now. The next stage should inspect CUDA synchronization: replace per-frame full stream synchronization with a safer event handoff if possible, or move to CUDA Graph / TensorRT EP optimization if event handoff is not exposed cleanly.

## Verification

```powershell
python -m py_compile config.py pipeline\matting.py pipeline\pynv_io.py pipeline\pynv_stream.py tools\auto_tune_8k.py
python -m unittest tests.test_content_directory_modes tests.test_subtitles
uv run python tools\auto_tune_8k.py phase1 --video videos\test_8k_2.mp4 --profile quest --prefer green --duration 10 --startup-timeout 240 --client-timeout 120
```