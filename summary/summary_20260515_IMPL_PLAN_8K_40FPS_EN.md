# 8K Realtime Passthrough 40fps Implementation Plan — 2026-05-15

## Goal

Lift production passthrough from "barely 30fps with `ALPHA_STRIDE=3`" on 8K to a steady **40fps with `ALPHA_STRIDE=1`** (no alpha skipping), without requiring users to upgrade GPU.

This document is the engineering follow-up to:
- `prompt/HANDOVER_20260515.md` (feasibility research and reviewer notes)
- `baseline/baseline_20260508_pynv_8k.txt` (the only existing 8K full-chain measurement)

The plan is broken into one automation-preparation phase plus six implementation phases that mirror the reviewer's priority order. Each implementation phase has explicit acceptance gates; do not start phase N+1 until phase N's gate passes.

---

## Phase 0 — Automation harness prerequisites

Before any performance work starts, remove the dependency on manual playback from a physical DLNA client. No acceptance gate in this plan should require the user to manually open a player, select a DLNA item, and observe playback. Physical-device tests remain optional compatibility soaks after automated gates pass.

### Existing test code to reuse
- `tools/pynv_fullchain_probe.py`: base offline full-chain probe for decode → matting → encode → mux timings.
- `tools/pynv_decode_probe.py`: decode-only baseline and SimpleDecoder comparison point.
- `tools/bench.py`: auxiliary local decode / pipeline / playback checks. It is useful for quick smoke tests, but it is not sufficient as the main 8K automation gate because it does not exercise DLNA Browse or the current live item selection path.
- `tools/dlna_client_probe.py`: automated DLNA SOAP Browse + HTTP stream-pull simulator. It should prefer alpha-live by default and allow passthrough-live as an explicit option.

### Required harness work before Phase 1
1. Add a top-level orchestrator, proposed as `tools/auto_tune_8k.py`, that can run one phase at a time and write machine-readable results.
2. Let the orchestrator start the PTMediaServer process in a subprocess with `PT_DEBUG_LOGS=1`, wait until `/description.xml` and `/control/cds` are reachable, then stop the server at the end of the run.
3. Use `tools/dlna_client_probe.py` from the orchestrator to Browse the library, select the requested video, prefer the alpha-live item, GET the returned DIDL `<res>` URL, and read the stream for a fixed duration.
4. Add a log parser for `debug_output/server.log` that extracts `[PYNV][sid] ... interval_fps=... stage_avg_ms decode=... composite=... sync=... encode=... mux=...`, `mux stdin write slow`, HTTP pacing warnings, selected mode, and stream session id.
5. Extend `tools/pynv_fullchain_probe.py` with a `--pipeline=serial|staged` option and JSON output so Phase 4 can be gated offline before the live route is tested.
6. Add or schedule the missing standalone probes: `tools/pynv_threaded_decode_probe.py` for ThreadedDecoder validation and `tools/trt_rvm_probe.py` for TensorRT EP validation.
7. Write all outputs under `baseline/`, for example:
   - `baseline/auto_tune_8k_phase1_YYYYMMDD_HHMMSS.json`
   - `baseline/auto_tune_8k_phase1_YYYYMMDD_HHMMSS.md`
   - captured server log excerpt and client probe JSON.

### Phase 0 acceptance gate
One command can run the Phase 1 measurement without user interaction:

```powershell
uv run python tools/auto_tune_8k.py phase1 --video <8k-file> --profile quest --prefer alpha --duration 60
```

The command must start the server, run the automated DLNA Browse and live HTTP pull, parse producer/client metrics, stop the server, and write a baseline report. If this gate is not ready, do not start Phase 1 measurements because they would still depend on manual testing.

---

## Background — measured single-frame budget at 8K

From `baseline_20260508_pynv_8k.txt` (steady state, RTX-class GPU, MATTING_INPUT_SIZE=512, SBS batch=2, RVM_DOWNSAMPLE_RATIO=0.5):

| Stage | stride=3 | stride=1 | Notes |
|---|---:|---:|---|
| `frame_at(src_idx)` decode | 18.8 ms | 5.0 ms | random-access via `SimpleDecoder[index]` is the cause of the stride=3 inflation |
| matting (preprocess+ORT, averaged) | 5.8 ms | 20.7 ms | one ORT call costs ~17 ms regardless of stride |
| composite | 0.16 ms | 0.16 ms | CuPy RawKernel, GPU |
| encode (NVENC HEVC) | 0.33 ms | 0.40 ms | |
| mux_write (FFmpeg stdin) | 1.92 ms | 3.61 ms | |
| serial sum | ~27 ms | ~30 ms | matches probe FPS 35.9 / 31.9 |

Production target: ≤ 25 ms per frame at stride=1 → 40 fps.

---

## Phase 1 — Diagnose the live HTTP gap (no code changes)

### Why first
Probe shows 35.9 fps but the user reports ~30 fps in production. Before we restructure the pipeline, confirm whether the bottleneck is the producer or the HTTP delivery path.

### Steps
1. Run `tools/auto_tune_8k.py phase1 --video <8k-file> --profile quest --prefer alpha --duration 60`.
2. The orchestrator starts the server with `PT_DEBUG_LOGS=1`, waits for DLNA readiness, runs `tools/dlna_client_probe.py`, and saves both client JSON and server log excerpts under `baseline/`.
3. From `debug_output/server.log`, automatically extract the periodic `[PYNV][sid] frame N/M ... interval_fps=... stage_avg_ms decode=... composite=... sync=... encode=... mux=...` lines emitted at `_DIAG_INTERVAL`.
4. Check three signals:
   - `interval_fps` — actual producer throughput.
   - `mux_write` average and any `mux stdin write slow` warning (`pipeline/pynv_stream.py:1779-1787`).
   - Compare to client-side first-byte time, bytes read, average bitrate, stalls/timeouts, and HTTP status from `dlna_client_probe.py`.

### Decision matrix
- **interval_fps ≈ 35 and client ≈ 26** → HTTP send pacing or queue is the bottleneck. Mitigations:
  - Raise `PT_PASSTHROUGH_SEND_PACING_MULTIPLIER` from `2.0` to `3.0`–`4.0`.
  - Temporarily set `PT_PASSTHROUGH_SEND_REALTIME_PACING=0` and re-measure.
  - Inspect `_audio_cache` lock acquisition latency.
- **interval_fps ≈ 26** with healthy mux_write → producer is the bottleneck; proceed to Phase 2+.
- **mux_write spikes > 100 ms** → FFmpeg mux subprocess is back-pressured; investigate downstream (TS muxer, slate/audio path) before touching encode.

### Acceptance gate
A clear written attribution of the missing 5–10 fps (HTTP pacing, mux back-pressure, or genuine producer cap), recorded as a generated probe note under `baseline/`. The report must be produced by the automation harness, not by manual player observation.

### Risk
None — this phase is read-only.

---

## Phase 2 — `ThreadedDecoder` sequential probe (isolated)

### Why
`pipeline/pynv_io.py:201` (`PyNvSimpleDecoder.frame_at`) wraps `SimpleDecoder[index]`. NVIDIA documents `ThreadedDecoder` as the API designed for inference/high-throughput sequential workloads with background prefetch. The 8K, source 60fps → output 30fps case is exactly 1:2 sequential decimation; the random-access cost in `SimpleDecoder` (~12 ms over the steady decode floor) should disappear.

### Steps
1. Add `tools/pynv_threaded_decode_probe.py` that:
   - Opens the 8K test file with `nvc.ThreadedDecoder(filename, gpu_id=..., use_device_memory=True)` (or the closest API name in the installed PyNvVideoCodec 2.1.0).
   - Calls `seek_to_frame(start_idx)` if needed, then loops `get_next_frame()` (or equivalent) and discards intermediate frames to honor the CFR `cfr_source_index` mapping currently in `pynv_stream.py:1704`.
   - Reports per-frame `t_decode` and steady-state fps over 300 frames.
2. Compare against `tools/pynv_decode_probe.py` (current SimpleDecoder, 141 fps decode-only).
3. Verify pixel-identical output for the first 10 decoded frames against SimpleDecoder (NV12 plane SHA256).

### Decision matrix
- ThreadedDecoder steady decode time ≤ 8 ms per output frame (1:2 decimation): **proceed to Phase 3**.
- ThreadedDecoder API not exposed in PyNv 2.1.0, or returns equal/worse than SimpleDecoder: keep SimpleDecoder, proceed to Phase 3 anyway — the ring buffer + sync removal will still help.
- Frame mismatch vs SimpleDecoder: stop, investigate; the live route depends on deterministic CFR mapping.

### Acceptance gate
A standalone probe script reproducing decode time ≤ 8 ms and a written CFR-skip pattern that matches `cfr_source_index`. **No changes to the live route in this phase.**

### Risk
PyNv 2.1.0 `ThreadedDecoder` API surface may differ from current docs. If the constructor signature, frame-pull method, or seek behavior is incompatible with seek-then-stream playback, the probe must explicitly document it; in that case Phase 4's improvements still apply but decode stays at SimpleDecoder.

---

## Phase 3 — GPU NV12 ring buffer (2–3 slots)

### Why
`pipeline/matting.py:_ensure_dev_nv12_out` returns a single `_g_out` buffer. The next frame's `composite` overwrites the same memory that NVENC may still be reading, so the live loop currently inserts a hard `cuda_stream.synchronize()` at `pipeline/pynv_stream.py:1751`. Without per-slot ownership, no overlap is safe.

### Concrete code touch points
- `pipeline/matting.py`:
  - `Matter._ensure_dev_nv12_out(h, w)` → grow into `Matter._acquire_nv12_slot(h, w)` returning an indexed slot from a fixed-size pool (size from new `config.PASSTHROUGH_NV12_RING_SLOTS`, default `3`).
  - Add `Matter._release_nv12_slot(idx)` and a slot state mask (`free` / `compositing` / `encoding`).
- `pipeline/pynv_stream.py`:
  - Replace `out_nv12, _ = self.matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(frame)` site to acquire a free slot before composite.
  - Track which slot is in NVENC's hands; release after `self._enc.Encode(app_frame, flags)` returns the bitstream (or after the next `Encode` call returns, depending on PyNv reentrancy).

### Sync strategy (safest first)
1. **Delayed slot reuse** (no event API needed): with N=3 slots, by the time the producer wants slot 0 again, the prior `Encode` has returned and consumed its input. This is correct as long as `self._enc.Encode()` blocks until input is fully read into NVENC's internal queue (the standard PyNv documented behavior).
2. **CUDA event handoff** (only if PyNv exposes it): record an event after composite, pass to `Encode(input, wait_event=...)`. **Verify in Phase 2 probe whether this signature exists in the installed 2.1.0 wheel** before committing to it.

### Acceptance gate
- Producer no longer calls `cp.cuda.get_current_stream().synchronize()` per frame for the NV12 slot handoff.
- Visual A/B against current build on `videos/test_8k.mp4`: zero artifacts in the first 200 frames (corrupted NV12 reuse would show as horizontal tearing or color blocks).
- `interval_fps` rises by at least the equivalent of removed sync time (currently `sync` is measured in the diagnostic line as `sum_sync` — should fall to ~0).

### Risk
- If PyNv `Encode()` is asynchronous (returns before reading input), delayed reuse is unsafe. **Phase 2 probe must measure**: time `Encode()` call, then immediately overwrite input buffer with garbage, encode N more frames, and inspect output for corruption. If unsafe, fall back to one explicit `cuda_stream.synchronize()` per ring rotation (not per frame).

---

## Phase 4 — Three-stage pipeline (decode / ORT+composite / encode+mux)

### Why
With Phases 2 and 3 complete, the three hardware engines (NVDEC, CUDA SM/Tensor, NVENC) can run concurrently. Theoretical wall-clock per output frame becomes `max(decode, ORT+composite, encode+mux)`. Using stride=1 numbers from baseline:
- decode (with ThreadedDecoder): ~7 ms
- ORT + composite: ~20 ms
- encode + mux: ~4 ms
- max ≈ 20 ms → ~50 fps (conservative, assumes some contention)

### Architecture
- One Python thread per stage, communicating via bounded `queue.Queue(maxsize=ring_slots)`.
- Stage A (decode): pulls frames from `ThreadedDecoder` (or SimpleDecoder fallback), pushes a `(slot_idx, GpuNv12Frame)` token.
- Stage B (matting): consumes A's token, runs ORT IOBinding + composite into the slot's NV12 buffer, pushes `(slot_idx, app_frame)` token.
- Stage C (encode+mux): consumes B's token, calls `self._enc.Encode(app_frame)`, writes bitstream to mux stdin, releases the slot.
- The current `_worker_loop` (`pipeline/pynv_stream.py:1489`) is the supervisor that starts/joins the three threads, propagates `_stop`, and aggregates the diagnostic stats.

### Cross-stage GPU sync
- Stage A → Stage B: NVDEC writes into device memory; CuPy import is zero-copy (`PyNvSimpleDecoder.frame_at` already does this). The CUDA stream issuing the decode is internal to PyNv; the matting stream is `_CUDA_STREAM`. Insert one `cudaEventRecord` on PyNv's stream and `cudaStreamWaitEvent` on `_CUDA_STREAM` per frame token. If PyNv does not expose its decode stream, fall back to `cp.cuda.runtime.deviceSynchronize()` once on the producer thread before queueing — coarse but safe.
- Stage B → Stage C: per-slot event recorded on `_CUDA_STREAM` after composite finishes; Stage C waits on it before `Encode()`. Or, given delayed slot reuse from Phase 3, omit the event and rely on slot rotation.

### Concrete code touch points
- `pipeline/pynv_stream.py:_worker_loop` is the only function that needs to be split. The rest of the class (preflight, mux open, AAC cache, slate, subtitles) stays identical.
- New helper module `pipeline/pynv_pipeline.py` holding the three thread bodies and a shared `Slot` dataclass keeps `pynv_stream.py` from ballooning.

### Acceptance gate
- New 8K full-chain probe (`tools/pynv_fullchain_probe.py --pipeline=staged`) reports steady fps ≥ 40 with `ALPHA_STRIDE=1`.
- Automated live HTTP probe (`tools/auto_tune_8k.py phase4 --profile quest --prefer alpha --duration 60`) shows `interval_fps ≥ 38` for ≥ 60 s with no `mux stdin write slow` warnings, and the client-side pull reports sustained bytes/bitrate with no timeout or premature disconnect.
- Subtitle overlay path (`_apply_subtitle_overlay`) still works; alpha-pack path still works.

### Risk
- **Concurrency in `Matter`**: `_g_alpha`, `_g_chw`, `_rvm_io_outputs`, `_rvm_rec_ort`, `_cached_alpha_small` are all process-singleton state on the `Matter` instance. Stage B is the only thread that touches them, so they remain single-threaded internally — but make sure Stage A does not also touch the matter instance for any preflight side effects.
- **Order preservation**: queues must preserve frame order; do not use a parallel pool for Stage B unless a per-slot sequence number is propagated and Stage C reorders. Keep Stage B single-threaded for v1.

---

## Phase 5 — ORT CUDA Graph for RVM

### Why
At this point ORT is the new bottleneck (~17 ms). ORT supports CUDA Graph capture when input/output addresses are stable (`enable_cuda_graph` provider option). Expected reduction: 20–35% of kernel-launch overhead → ORT down to ~12 ms.

### Prerequisite: stable recurrent state addresses
`pipeline/matting.py:1416` does `self._rvm_rec_ort = rec_outs[:4]`, which **swaps the OrtValue objects**. CUDA Graph captures pointer values at capture time; swapping breaks replay. Required change:

- Allocate four persistent `OrtValue` buffers for `r1i..r4i` and four for `r1o..r4o` once per resolution, sized from `_rvm_output_shape_for`.
- After each `run_with_iobinding`, copy `r*o` device buffers back into `r*i` device buffers in place (`cudaMemcpyAsync` on `_CUDA_STREAM`, or expose an `OrtValue` device-to-device copy helper).
- Bind `r*i` as inputs and `r*o` as outputs via `bind_ortvalue_input` / `bind_ortvalue_output` always to the same OrtValue objects.

### Enable
Once addresses are stable, add `"enable_cuda_graph": "1"` to `_provider_config` in `pipeline/matting.py:828`.

### Validation
- First inference in a session: regular CUDA EP run (graph capture).
- Subsequent inferences: graph replay; verify with `nvidia-smi dmon` that kernel launches drop sharply.
- Bit-exact alpha output vs non-graph build for 100 frames at the same shape.

### Acceptance gate
- ORT average drops from ~17 ms to ≤ 13 ms on the same hardware.
- No accuracy regression (PSNR of composited output ≥ 50 dB vs non-graph reference).

### Risk
- Graph capture invalidated if any input shape (including SBS active vs not, batch=1 vs 2) changes mid-stream. **Force a fixed shape per session**; if SBS flips, fall back to non-graph mode and log it.
- ORT versions vary in CUDA Graph stability for recurrent graphs; `1.19.2` is current — verify with a 5-minute soak test.

---

## Phase 6 — TensorRT EP probe (optional, highest variance)

### Why
TensorRT EP can deliver another 1.5–2.5× over CUDA EP fp32, especially on Turing (RTX 2080) where Tensor Cores are well-utilized for fp16. Reviewer confirmed `TensorrtExecutionProvider` is available in installed `onnxruntime==1.19.2`.

### Steps
1. Build a standalone probe `tools/trt_rvm_probe.py`:
   - Loads the same RVM ONNX with `providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": "1", "trt_engine_cache_enable": "1", "trt_engine_cache_path": "runtime_cache/trt_engines"})]`.
   - Forces the RVM input shapes used in production (8K SBS halves at 4096×4096 downsampled to 512×512, batch=2 plus rec inputs).
   - Times 100 inferences after one warmup.
2. Compare against CUDA EP IOBinding numbers from existing baseline.
3. Validate output bit-for-bit (or PSNR ≥ 50 dB) against fp32 CUDA EP.

### Decision matrix
- TRT inference ≤ 8 ms with no quality regression: schedule a follow-up integration phase.
- TRT inference between 8 and 13 ms: still worth integrating because of headroom.
- TRT engine build fails on RVM recurrent inputs / inference quality drops noticeably: drop TRT, accept Phase 5's CUDA Graph as the ORT-side ceiling.

### Acceptance gate
A standalone probe report under `baseline/`, including: build time, engine cache size, steady fps, output PSNR, and notes on FP16 dynamic range issues if any.

### Risk
- RVM recurrent loops sometimes confuse TRT shape inference; may need to pin all `r*i` input shapes explicitly.
- Engine build may take 1–3 minutes the first time; must be hidden inside startup warmup (`utils/gpu_runtime_cache.py`).

---

## Configuration knobs introduced

To be added to `config.py` with `PT_*` env overrides (defaults shown):

| Variable | Default | Purpose |
|---|---:|---|
| `PT_PASSTHROUGH_NV12_RING_SLOTS` | `3` | NV12 output slot count for Phase 3 |
| `PT_PASSTHROUGH_PIPELINE_MODE` | `staged` | `serial` (current) or `staged` (Phase 4) |
| `PT_RVM_CUDA_GRAPH` | `0` | Enable CUDA Graph after Phase 5 stabilizes |
| `PT_RVM_TENSORRT_EP` | `0` | Enable TRT EP after Phase 6 PoC succeeds |

Default migration plan: ship Phase 3 + Phase 4 with `PT_PASSTHROUGH_PIPELINE_MODE=staged` as default after the gate passes; keep Phases 5 and 6 behind their flags pending soak tests.

---

## Cumulative expected gain (revised, conservative)

| After phase | ORT path | Expected steady 8K fps (stride=1) |
|---|---|---:|
| 0 (current) | CUDA EP IOBinding, serial loop | 31.9 |
| 1 (HTTP fix) | same | 32–35 (HTTP-only) |
| 3 (ring buffer) | same, sync removed | 35–38 |
| 4 (3-stage pipeline) | same | 40–48 |
| 5 (CUDA Graph) | CUDA Graph | 45–55 |
| 6 (TRT EP) | TRT fp16 | 55–70+ |

40 fps target is reached after Phase 4. Phases 5 and 6 buy headroom for new features.

---

## What could still defeat this plan

1. PyNv 2.1.0 `Encode()` is internally synchronous on input read AND on output emit, with no event handoff exposed. Then Phase 4's encode stage cannot overlap with NVENC silicon work — but NVENC silicon is anyway sub-ms here, so this is a small cost.
2. NVDEC on the user's GPU is gen-1 (Maxwell). Then 8K HEVC decode itself caps below 30 fps and no amount of pipelining helps. Detect this in Phase 2 probe; for those users, the only realistic option is downscale-on-decode or precomputed alpha (offline path already exists).
3. ORT 1.19.2 CUDA Graph misbehaves with the specific RVM recurrent topology. Fall back to plain IOBinding and accept Phase 4's number.
4. The existing live route's audio cache (`_lock_for_audio_cache`, `pipeline/pynv_stream.py:125`) holds locks long enough that Phase 4's three threads still serialize through it. Audit lock scopes during Phase 4 implementation.

---

## Suggested order of work for the assigned developer

1. **Day 0** — Phase 0 automation harness. Wire `auto_tune_8k.py`, server subprocess lifecycle, `dlna_client_probe.py`, log parsing, and baseline report generation.
2. **Day 1** — Phase 1 automated diagnosis. Run the harness, try one possible env tweak if indicated, write generated findings to `baseline/`.
3. **Day 1–2** — Phase 2 probe. Standalone, no risk to production.
4. **Day 2–3** — Phase 3. Ring buffer + delayed reuse. Heaviest correctness review needed here.
5. **Day 3–5** — Phase 4. Pipeline split. Bulk of the win lives here.
6. **Day 5+** — Phase 5 (graph) and Phase 6 (TRT) only after Phase 4 ships and is stable for at least a few sessions.

After each phase, append a results section to this document so the project keeps a single record of what worked.
