# Stage 4 Advice - Reply to STAGE4_PYNV_THREADED_STAGED_CRASH_EN

This document replies to `summary/summary_20260516_STAGE4_PYNV_THREADED_STAGED_CRASH_EN.md`.

It judges the crash cause, validates the underlying assumptions against current code, and recommends a concrete next path. All claims here were verified against the production tree before writing.

## Verification Pass

Cited file/line evidence used by this advice:

- `pipeline/pynv_io.py:172-215` — `PyNvSimpleDecoder.frame_at(index)` is a thin wrapper over `SimpleDecoder.__getitem__`. Random indexed access only.
- `pipeline/pynv_stream.py:1783` — live worker calls `self._dec.frame_at(src_idx)` inside the per-frame serial loop. This is the production decode call.
- `pipeline/pynv_stream.py:1778-1781` — CFR source-index selection (`cfr_source_index(...)` then monotonic clamp) is already done here; the worker already knows the next monotonic source index.
- `pipeline/pynv_stream.py:1853-1858` — Stage 3 delayed slot release: pending slots up to `PASSTHROUGH_NV12_RING_SLOTS - 1` are held after `Encode()`.
- `pipeline/matting.py:1206-1223` — `Matter.acquire_nv12_output_slot()` raises `RuntimeError` on empty pool. No blocking variant.
- `tools/pynv_threaded_mapping_probe.py:141-176` — safe ThreadedDecoder pattern: frames are consumed (hashed) inside the `for raw in batch` loop, before the next `get_batch_frames()` call and before `end()`. Already proven by Stage 1.
- `tools/pynv_threaded_decode_probe.py:166-229` — sequential ThreadedDecoder with CFR decimation (selected source-frame skipping) already exists as an offline probe. This is the missing measurement.

The original Stage 4 report's findings are consistent with all of the above.

## Root Cause Judgement (high confidence)

The Stage 4 staged design crashed because it violated the explicit ThreadedDecoder frame-lifetime contract that Stage 1 had already written down.

Stage 1 stated:

> ThreadedDecoder frames must be consumed before the next `get_batch_frames()` call and before `end()`.

The failed staged design did:

```
decode worker:    get_batch_frames()
                  for raw in batch: queue.put(wrapped_raw)
                  get_batch_frames()    <-- previous batch now invalidated
matting worker:   wrapped_raw = queue.get()
                  read frame.cuda() planes   <-- use-after-free
```

A second `get_batch_frames()` invalidates the previous batch's underlying CUDA storage. A separate matting thread that holds `frame.cuda()` views from the previous batch then reads invalidated device memory. The Windows native crash dialog (`0x00007FFFCCC27880` reading `0x0000000B69200000`) is the expected symptom of this use-after-free.

This is not a PyNvVideoCodec bug. It is a design that broke a documented contract.

## Secondary Findings

### Slot backpressure is a latent design issue

`Matter.acquire_nv12_output_slot()` throws on empty (`pipeline/matting.py:1223`). In a staged pipeline, matting can outrun encode + delayed-release, requesting an N+1 slot and immediately exploding. The current pool is correct for the serial worker only. Any future staged or threaded design must change this API to support blocking or `try_acquire + wait`.

### SimpleDecoder staged did not move the needle

The Stage 4 SimpleDecoder staged run produced `31.76 fps` because the decode path remained `SimpleDecoder[index]` random access at ~27 ms per frame. Even with perfect three-stage overlap, the theoretical ceiling is `1000 / 27 ≈ 37 fps`. SimpleDecoder + staging cannot reach 40 fps.

## Overlooked Key Insight

**ThreadedDecoder is already an internally pipelined decoder. Wrapping it inside a staged pipeline is mostly redundant.**

Reference numbers we already have:

| Decode pattern | 8K avg per frame | Source |
|---|---|---|
| `SimpleDecoder[index]` (random) | ~18-27 ms | `baseline/baseline_20260508_pynv_8k.txt` and Stage 4 staged report |
| ThreadedDecoder sequential, decode-only | ~7 ms | `baseline/baseline_20260508_pynv_8k.txt:51` reports 141.73 fps |

If the live serial worker simply swapped `frame_at(src_idx)` for ThreadedDecoder sequential pull + CFR drop, a rough budget is:

```
decode (sequential, drop half)   ~7 ms
matting (stride=3)               ~6 ms
encode                           ~0.3-6 ms
mux                              ~2 ms
-----------------------------------
total                            ~15-22 ms  =>  45-65 fps
```

40 fps is plausibly reachable inside the current serial worker shape, without any staged-pipeline crash surface. This must be measured before any further staged work.

## Verification Probe That Should Be Run First

`tools/pynv_threaded_decode_probe.py` already implements exactly the measurement we need:

- sequential ThreadedDecoder pull from `start_frame`
- CFR decimation using the same `cfr_source_index` logic the live worker uses
- selected-frame FPS, source-fetch FPS, fetch latency
- hash comparison against `SimpleDecoder[index]` for the selected indices

Recommended run (8K, 30 fps output target, realistic batch):

```powershell
uv run python tools\pynv_threaded_decode_probe.py videos\test_8k_2.mp4 --frames 300 --fps 30 --batch-size 8 --buffer-size 32 --hash-frames 20
```

Decision gates:

- `threaded.selected_fps >= 45` and `hash_compare.ok == true` -> path A is green-lit.
- `threaded.selected_fps` between 35 and 45 -> path A still preferred, but path B is likely needed.
- `threaded.selected_fps < 35` or `hash_compare.ok == false` -> escalate; the bottleneck is not decode.

This single probe answers the path-selection question and costs almost nothing.

## Recommended Path (in priority order)

### Path A - Serial worker + ThreadedDecoder (recommended first)

Replace `self._dec.frame_at(src_idx)` in the live serial loop with a ThreadedDecoder sequential pull + CFR drop, keeping everything else in the same worker thread.

Implementation outline (no code in this doc):

- Create a small `PyNvThreadedSerialReader` wrapper that:
  - opens `ThreadedDecoder(start_frame=initial_src_idx, buffer_size=32)`;
  - exposes `next_selected_frame(target_source_idx) -> GpuNv12Frame | GpuP016Frame`;
  - internally pulls batches of size N (suggested 4-8) and skips frames whose `source_idx` is not selected;
  - returns the selected frame immediately, **and consumes/composites it before the next `get_batch_frames()` call** (so we keep Stage 1's lifetime contract).
- In `pipeline/pynv_stream.py` worker (around lines 1767-1858):
  - Replace `self._dec.frame_at(src_idx)` with the wrapper call.
  - Move the matting/composite work and the `cuda_stream.synchronize()` so they all complete on this iteration before the next batch is fetched (which is naturally true in serial code; only document the invariant).
- Keep Stage 3 delayed slot release exactly as-is. Serial path never holds more than 1 in-flight composite slot, so `count=3` is more than enough.
- Keep `cuda_stream.synchronize()` before `Encode()`. Stage 3 explicitly warned against removing it without an event-handoff replacement.

Constraints to preserve:

- Within one wrapper call, every returned frame is fully consumed (composite -> sync -> encode appended to pending) before the wrapper triggers the next `get_batch_frames()`.
- `cfr_source_index` already returns monotonic indices when the caller clamps with `last_src_idx + 1` (lines 1778-1781); the wrapper must respect this.
- For seek/restart (start_sec != 0), open the ThreadedDecoder with `start_frame=initial_src_idx` so identity stays `start + local_sequence`.

Expected outcome: 8K stride=3 lifts from `~31-35` to `~45-55` fps. If that hits 40 fps + sufficient headroom, Stage 4 staged work can be abandoned.

Effort: 1 day. Risk: low. Crash surface: none new.

### Path B - Two-stage pipeline (decode+matting / encode+mux)

Only if path A measures below 40 fps with reasonable headroom.

Two workers:

- Worker 1: ThreadedDecoder pull + matting/composite into a Matter NV12 slot.
  - **ThreadedDecoder frames never leave this worker.** They are consumed inside Worker 1's loop iteration.
  - The cross-thread handoff is the Matter NV12 slot, whose lifetime is owned by `Matter.acquire/release_nv12_output_slot`. Stage 2 already validated this.
- Worker 2: `Encode(slot)`, mux write, slot release.

Required changes outside the worker:

- `Matter.acquire_nv12_output_slot()` must support blocking with timeout, or a `try_acquire` + condition-variable wait must be added. Current throw-on-empty breaks Worker 1 under backpressure.
- `PASSTHROUGH_NV12_RING_SLOTS` should rise from 3 to 4 or 5 to absorb encode jitter. Stage 3 lifetime probe was only validated at 3; re-run the probe at the new size:

```powershell
uv run python tools\pynv_encode_lifetime_probe.py --width 8192 --height 4096 --frames 24 --codec hevc --bitrate 50000000 --gop 60 --progress 6 --slots 5
```

Expected outcome: another `+5-10` fps over path A, comfortable margin over 40 fps.

Effort: 1.5 days. Risk: medium (slot pool API change, encode jitter handling).

### Path C - Three-stage pipeline behind an owned-decode ring

Only if both path A and path B fail to reach the target. Implements Option A from the original Stage 4 report.

Before any staged work resumes, add a lifetime probe (the original report's "ownership probe"):

```text
tools/pynv_threaded_lifetime_probe.py
  ThreadedDecoder.get_batch_frames(N)
  cudaMemcpy2DAsync Y plane and UV plane into user-owned CuPy NV12 ring slot
  stream.synchronize()
  get_batch_frames(N) again   <-- old batch invalidated by design
  hash owned slot contents and encode them
  decode encoded output, compare against ground-truth hashes
  repeat across many batches and start_frame values
```

Only after this passes for thousands of iterations may a 3-stage design be implemented. Without owned-ring validation, three-stage with ThreadedDecoder is the exact failure mode Stage 4 already hit.

Effort: probe 0.5 day + 3-stage 1 day. Risk: high.

### Path D - Required regardless of path choice

Make `Matter.acquire_nv12_output_slot()` capable of blocking (timeout-aware) or expose `try_acquire`. Current throw-on-empty is a latent landmine for any future overlap work, and is also a small DX cost for the offline probe.

Effort: 0.5 day. Risk: trivial.

## Answers to the Report's 10 Expert Questions

| # | Question | Answer (best available from code + Stage 1 evidence) |
|---|----------|------|
| 1 | Correct ownership/lifetime model | Frames are valid from return of `get_batch_frames()` until the **next** `get_batch_frames()` call (and until `end()`). Stage 1 already validated this. |
| 2 | Retain/copy API | No retain. Device-to-device copy to a user-owned buffer is the only safe carry. |
| 3 | Device-to-device copy safety | Safe if completed and stream-synchronised **before** the next `get_batch_frames()`. CuPy slice assignment or `cudaMemcpy2DAsync` both work. |
| 4 | Copy primitive | CuPy slice assignment with explicit stream is simplest; `cudaMemcpy2DAsync` if pitched plane copies are needed. |
| 5 | Which CUDA stream owns ThreadedDecoder output | Internal PyNv stream, not exposed. User must `stream.synchronize()` after fetching and before letting the next batch invalidate the storage. |
| 6 | `frame.cuda()` cross-thread validity | Within the batch's validity window, planes are device pointers; any thread that reads before the next batch is technically safe, but practically very hard to guarantee. Do not rely on this. |
| 7 | Decode and matting in same thread? | Yes for path A/B. This is the safest pattern and what the existing probes already do. |
| 8 | 3-stage without owned ring? | No. The owned-ring copy is mandatory for cross-stage decode. |
| 9 | Correct shutdown sequence | Drain or discard the last batch's frames, then call `end()`. Calling `end()` with held frames is undefined. |
| 10 | CUDA events / refcount APIs? | None known in PyNvVideoCodec 2.1.0 Python API. Encode side already lacks an event-handoff API; the project mitigates this with Stage 3 delayed-release. |

## Relation to Existing Plans

- `prompt/IMPL_PLAN_8K_40FPS_20260515.md` Phase 2-4: Path A here compresses Phase 2 + most of Phase 4 into a single low-risk change. If path A reaches 40 fps, Phase 4 staged work can be dropped entirely.
- Phase 5 (CUDA Graph for RVM) and Phase 6 (TRT EP) remain as **headroom** improvements rather than gating items. They should still be pursued but are decoupled from the 40 fps target.
- The MKV stall mitigation in `summary/summary_20260516_MKV_PYNV_SIMPLEDECODER_STUCK_ISSUE_CN.md` is orthogonal; nothing in this advice affects that path.

## Recommended Next Actions (concrete, ordered)

1. Run `tools/pynv_threaded_decode_probe.py` on `videos/test_8k_2.mp4` at `--fps 30 --frames 300 --batch-size 8`. Read `threaded.selected_fps` and `hash_compare.ok`.
2. If selected_fps >= 45 and hash ok -> implement path A (`PyNvThreadedSerialReader` + worker swap).
3. Re-run `tools/auto_tune_8k.py phase1 ... --duration 60 --prefer green/alpha` to confirm live FPS.
4. If short of target, implement path D (blocking slot acquire) then path B (two-stage).
5. Do not start path C until the owned-ring probe has passed.
6. Keep `tools/pynv_fullchain_probe.py --decoder threaded` disabled until path A or path C is shipped.

## Status

Stage 4 (as originally scoped) is correctly suspended. The replacement plan above (path A first, paths B/C/D conditional) carries no new native-crash surface and reuses validation tooling that already exists in the tree. Decision gate is the single ThreadedDecoder decode probe; it has not yet been run with realistic CFR decimation parameters and should be the first action taken.
