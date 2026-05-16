# Stage 4 Issue Report - PyNv Threaded Staged Pipeline Crash

## Background

This report documents a failed Stage 4 optimization attempt for the 8K realtime passthrough pipeline.

The intended Stage 4 goal was to split the current serial live processing path into three stages:

- decode;
- RVM matting and GPU NV12 composite;
- NVENC encode and mux.

The expected benefit was overlap between NVDEC, CUDA/ORT work, and NVENC/mux, with the long-term target of reaching 8K realtime `40fps` with `ALPHA_STRIDE=1`.

This work did not reach a usable result and caused native `python.exe` crash dialogs on Windows. The work should be treated as unsafe until reviewed by an expert.

## Environment And Test Asset

- Repository: `G:\GIT\debug\PTMediaServer`
- Test video: `videos/test_8k_2.mp4`
- Platform: Windows
- Relevant native stack: PyNvVideoCodec, CUDA, CuPy, ONNX Runtime CUDA EP, NVENC
- Current stable production route before Stage 4:
  - PyNv SimpleDecoder;
  - Matter GPU matting/composite;
  - PyNv NVENC;
  - FFmpeg mux;
  - serial worker loop in `pipeline/pynv_stream.py`.

## Previous Safety Findings Used By Stage 4

Stage 1 proved ThreadedDecoder content mapping can be correct only when used with strict lifetime rules:

- `ThreadedDecoder(start_frame=N)` content maps to `SimpleDecoder[N + local_sequence]`;
- `getPTS()` must not be used as frame identity;
- returned frames must be consumed before the next `get_batch_frames()` call and before `end()`.

Stage 3 proved NVENC input lifetime is also not immediate:

- reusing or overwriting the same GPU NV12 input slot immediately after `Encode()` returns is unsafe;
- a 3-slot delayed reuse ring passed synthetic 8K HEVC probes;
- therefore live green path was changed to delayed release of NV12 output slots.

## What Was Changed During Stage 4 Attempt

The live route was not intentionally converted to the staged pipeline.

Most Stage 4 work happened in the offline probe:

- `tools/pynv_fullchain_probe.py`
  - implemented `--pipeline staged`;
  - added decode / matting / encode worker threads;
  - added bounded queues;
  - aligned runtime cache initialization with production by calling `configure_gpu_runtime_cache()`;
  - disabled the old fixed `tempfile.TemporaryDirectory` monkey patch by default;
  - added `--decoder simple|threaded`;
  - later disabled `--decoder threaded` explicitly after native crashes.

Additional diagnostic logging was temporarily added in:

- `pipeline/matting.py`
  - guarded by `DEBUG_LOGS`;
  - logs around GPU NV12-to-NV12 composite kernel boundaries.

## Important Offline Probe Fix Found En Route

`tools/pynv_fullchain_probe.py` previously patched `tempfile.TemporaryDirectory` to a single fixed directory.

That caused CuPy / RawKernel paths to hang around the NV12 composite stage. The probe recovered after:

- using `configure_gpu_runtime_cache()`;
- leaving normal temp directory behavior enabled;
- making the old fixed-tempdir behavior opt-in with `PT_PYNV_FULLCHAIN_FIXED_TEMPDIR=1`.

This issue appears separate from the later native crash.

## Commands That Passed

SimpleDecoder staged smoke passed:

```powershell
uv run python tools\pynv_fullchain_probe.py videos\test_8k_2.mp4 --pipeline staged --frames 30 --discard 3 --fps 30 --codec hevc --bitrate 50000000 --alpha-stride 3 --input-size 1024 --raw-video-out debug_output\stage4_staged_30.hevc --out debug_output\stage4_staged_30.mp4 --json-out baseline\pynv_fullchain_stage4_staged_smoke_20260515.json --progress 10
```

Longer SimpleDecoder staged run also passed:

```powershell
uv run python tools\pynv_fullchain_probe.py videos\test_8k_2.mp4 --pipeline staged --frames 120 --discard 10 --fps 30 --codec hevc --bitrate 50000000 --alpha-stride 3 --input-size 1024 --raw-video-out debug_output\stage4_staged_120.hevc --out debug_output\stage4_staged_120.mp4 --json-out baseline\pynv_fullchain_stage4_staged_120_20260515.json --progress 30
```

Reports:

- `baseline/pynv_fullchain_stage4_staged_smoke_20260515.json`
- `baseline/pynv_fullchain_stage4_staged_120_20260515.json`

Result summary for the 120-frame SimpleDecoder staged run:

- completed successfully;
- steady FPS around `31.76`;
- still not enough for the 40fps target;
- decode remained expensive because it still used indexed/random `SimpleDecoder` access.

## Why SimpleDecoder Staged Was Not Enough

SimpleDecoder staged still used the current indexed decode pattern. For 8K 59.94fps source to 30fps output, the code repeatedly selected CFR source indices and called indexed decode.

That means the staged pipeline could overlap some work, but decode still dominated:

- decode average remained around `27 ms`;
- matting steady average around `8.9 ms`;
- encode steady average around `6.4 ms`.

The result was not a meaningful Stage 4 win.

## The Failed ThreadedDecoder Staged Attempt

To reduce decode cost, the offline staged probe attempted to use `PyNvVideoCodec.ThreadedDecoder`.

The attempted command pattern was:

```powershell
uv run python tools\pynv_fullchain_probe.py videos\test_8k_2.mp4 --pipeline staged --decoder threaded --frames 120 --discard 10 --fps 30 --codec hevc --bitrate 50000000 --alpha-stride 3 --input-size 1024 --raw-video-out debug_output\stage4_staged_threaded_120.hevc --out debug_output\stage4_staged_threaded_120.mp4 --json-out baseline\pynv_fullchain_stage4_staged_threaded_120_20260515.json --progress 30
```

Observed behavior:

- progress reached around `90/120` frames in the console;
- then the process failed before writing JSON;
- Windows repeatedly showed native `python.exe` crash dialogs.

User-reported crash text:

```text
python.exe - application error:
0x00007FFFCCC27880 referenced memory at 0x0000000B69200000.
The memory could not be read.
```

This is a native access violation. It is not a normal Python exception and cannot be handled safely with `try/except`.

## Most Likely Cause

The most likely cause is invalid lifetime handling of PyNv ThreadedDecoder frames.

The Stage 1 mapping work already established:

- ThreadedDecoder returned frames remain valid only until the next `get_batch_frames()` call;
- data must be consumed before fetching the next batch;
- calling `end()` before consuming returned frames is invalid.

The failed staged design violated that model:

- decode worker called `get_batch_frames()`;
- selected frames were wrapped and put into a cross-thread queue;
- decode worker continued fetching later batches;
- matting worker consumed earlier frame objects later.

That means matting may have read GPU pointers after PyNv had invalidated or reused the underlying decoded frame storage.

Given the crash address and native access violation, this is consistent with a CUDA/PyNv use-after-free or invalid device pointer read.

## Additional Risk Found: Slot Backpressure

The first SimpleDecoder staged implementation also hit:

```text
RuntimeError: no free NV12 output slot: count=3 shape=(6144, 8192)
```

Cause:

- Stage 3 delayed slot release keeps recent NV12 output slots pending to protect NVENC input lifetime;
- staged matting can outrun encode and request a fourth slot;
- the current Matter slot API throws immediately instead of blocking.

Temporary offline-probe workaround:

- staged probe now waits and retries slot acquisition.

Expert question:

- should `Matter.acquire_nv12_output_slot()` become a blocking API for staged pipelines, or should staged pipeline own slot scheduling outside Matter?

## Safety Action Already Taken

`tools/pynv_fullchain_probe.py --decoder threaded` is now disabled with an explicit `SystemExit`.

Current message:

```text
ThreadedDecoder is temporarily disabled in the staged full-chain probe.
PyNv ThreadedDecoder frames are only valid until the next get_batch_frames() call;
passing them across worker threads caused native Python/PyNv crashes during Phase 4 testing.
```

No further Python/GPU probe should be run until the crash path is reviewed.

## Recommended Expert Review Questions

1. What is the correct ownership and lifetime model for PyNvVideoCodec `ThreadedDecoder.get_batch_frames()` returned frames?
2. Is there an official way to retain/copy a decoded frame for use after the next `get_batch_frames()` call?
3. Can decoded frames be safely copied device-to-device into a user-owned CuPy NV12 buffer before fetching the next batch?
4. If yes, what copy primitive should be used: CuPy assignment, `cudaMemcpy2DAsync`, PyNv API, or another CUDA interop path?
5. Which CUDA stream owns ThreadedDecoder output, and how should a user stream wait for decode completion?
6. Is `frame.cuda()` memory valid across Python threads if the PyNv frame object remains referenced, or is validity still batch-scoped?
7. Should decode and matting stay in the same worker thread so ThreadedDecoder frames are consumed before the next batch?
8. Is a 3-stage pipeline possible with ThreadedDecoder without an owned intermediate decoded-frame ring?
9. What is the correct shutdown sequence for ThreadedDecoder when worker threads and queued frames are involved?
10. Does PyNvVideoCodec expose CUDA events, stream handles, or reference-counted frame ownership APIs that should be used here?

## Possible Safer Designs

### Option A - Owned Decode Ring

Decode thread:

1. calls `get_batch_frames()`;
2. for each selected frame, immediately copies Y and UV planes into a user-owned GPU NV12 ring buffer;
3. only the owned buffer is queued to matting;
4. next `get_batch_frames()` is called only after all selected frames in the current batch are copied.

This preserves ThreadedDecoder frame lifetime but adds a GPU copy.

### Option B - Decode And Matting In One Stage

Combine ThreadedDecoder and matting in one worker:

1. pull batch;
2. run matting/composite for selected frames before the next batch;
3. queue only Matter-owned NV12 encode slots to the encode worker.

This may reduce concurrency but avoids passing ThreadedDecoder frames across stages.

### Option C - Keep SimpleDecoder For Stage 4

Use SimpleDecoder staged only and seek performance elsewhere.

This is safer but did not reach the target in the 120-frame probe.

## Current Recommendation

Do not continue ThreadedDecoder staged implementation without external guidance.

The next attempt should first implement a small isolated ownership probe:

```text
ThreadedDecoder get_batch_frames()
copy selected frame to owned GPU NV12 buffer
fetch next batch
use copied buffer after original batch has been invalidated
hash/encode/compare copied content
repeat across many batches
```

Only if that passes should Stage 4 resume.

## Files Of Interest

- `tools/pynv_fullchain_probe.py`
- `tools/pynv_threaded_mapping_probe.py`
- `tools/pynv_threaded_decode_probe.py`
- `tools/pynv_encode_lifetime_probe.py`
- `pipeline/pynv_io.py`
- `pipeline/matting.py`
- `pipeline/pynv_stream.py`

## Status

Stage 4 is blocked.

The work produced useful diagnostic findings, but it did not deliver the intended performance improvement and introduced a native crash risk. ThreadedDecoder staged mode must remain disabled until the ownership/lifetime model is redesigned and validated in isolation.
