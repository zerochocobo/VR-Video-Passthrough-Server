# Alpha Passthrough ThreadedDecoder Red-Alpha Flicker Summary

## Executive Summary

In alpha passthrough mode, PyNv `ThreadedDecoder` / `threaded_serial` causes continuous red alpha-channel flicker on some videos. The same source is stable when using `SimpleDecoder`. This is not a simple startup failure or an obvious decode-only frame-content mismatch. The current production safeguard is: **alpha passthrough uses `SimpleDecoder` by default; green-screen passthrough can still use `ThreadedDecoder`.**

The root cause still needs external review, especially around PyNv `ThreadedDecoder`, CUDA stream synchronization, RVM recurrent state, and the alpha packer pipeline.

## Reproduction

- Mode: online alpha passthrough playback
- Example source: `videos\72456_3840p.mp4`
- User observation:
  - `threaded_serial`: strong continuous red alpha-channel flicker
  - `simple`: stable output, no flicker
- Relevant runtime setup:
  - `PT_ALPHA_STRIDE=1`
  - online path uses an RVM FP16 model
  - decoder mode is selected by the UI memory profile or environment variables

## Relevant Code

- `config.py`
  - `PASSTHROUGH_PYNV_DECODER`
  - new diagnostic flag: `PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER`
  - default `0`, meaning alpha production does not allow threaded decoder
- `pipeline/pynv_stream.py`
  - online alpha path falls back to effective `decoder=simple` when configured decoder is `threaded_serial`, unless the diagnostic flag is explicitly enabled
  - green-screen path is not affected
- `tools/offline_alpha_passthrough.py`
  - offline alpha uses the same safeguard
- `pipeline/pynv_io.py`
  - added `owned_copy()` for `GpuNv12Frame` / `GpuP016Frame`
  - attempts to copy PyNv-owned frame planes into CuPy-owned GPU buffers
  - synchronization is applied before and after the copy

## Attempted But Insufficient Fixes

### 1. Owned GPU Copy

Motivation:

- PyNv `ThreadedDecoder.get_batch_frames()` returned frames have a sensitive lifetime contract.
- Earlier work established that returned frames must not be retained across batches or across worker stages.
- The alpha path is more complex than green passthrough: decode, RVM, fisheye alpha packing, and red-channel overlay.

Implementation:

- `frame.owned_copy()`:
  - `cp.cuda.Device().synchronize()`
  - read PyNv planes through `CudaPlane.as_cupy()`
  - copy via `cp.ascontiguousarray()` into CuPy-owned buffers
  - `cp.cuda.get_current_stream().synchronize()`

Result:

- This fixed the initial `TypeError: Expected tuple, got list` / HTTP 503 failure caused by direct `cp.asarray(raw_pyNv_view)`.
- User retesting still showed severe alpha flicker.
- Therefore owned copy is not a sufficient production fix.

### 2. Decode-only Hash Verification

Command:

```powershell
.\.venv\Scripts\python.exe tools\pynv_threaded_decode_probe.py videos\72456_3840p.mp4 --frames 180 --fps 30 --batch-size 4 --buffer-size 8 --hash-frames 30
```

Result:

- Threaded selected FPS: `79.43`
- Simple baseline FPS: `79.85`
- Hash checked: `30`
- Matched: `30`
- OK: `True`
- PTS deltas: all `0`

Interpretation:

- Sampled decode-only content matches `SimpleDecoder`.
- Live alpha still flickers.
- The issue is likely not a simple frame mapping mismatch. It may involve full-pipeline timing, CUDA stream visibility, RVM recurrent state, or PyNv ThreadedDecoder output publication semantics.

## Current Production Safeguard

New configuration:

```text
PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=0
```

Default behavior:

- alpha passthrough: effective decoder is forced to `simple`
- green-screen passthrough: still follows the UI memory profile or `PT_PASSTHROUGH_PYNV_DECODER`
- diagnostics can explicitly enable threaded alpha:

```text
PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=1
```

When enabled:

- alpha allows `threaded_serial`
- `owned_copy()` is still applied
- this is diagnostic only and not recommended for production

## Important Clues

1. `SimpleDecoder` is stable; `ThreadedDecoder` alpha is not.
2. Decode-only hashes match, so this is not an obvious per-frame content mismatch.
3. Owned copy plus device synchronization still does not remove the flicker.
4. The issue is reported in alpha passthrough; green-screen passthrough has not shown the same red-alpha flicker.
5. RVM is recurrent. Small problems in frame timing, memory publication, CUDA stream visibility, or state progression may be amplified into visible alpha flicker.

## Questions For External Review

1. After PyNv `ThreadedDecoder.get_batch_frames()` returns GPU frames, is the “consume before next get_batch_frames” rule sufficient, or is there an official stream/event wait requirement?
2. Does `get_batch_frames()` guarantee returned CUDA memory is visible to the default stream / CuPy stream at function return?
3. Is `cp.cuda.Device().synchronize()` sufficient to wait for PyNv internal NVDEC / postprocess streams?
4. For AV1 7680x3840 60fps sources, can ThreadedDecoder have internal frame reordering, surface reuse, or delayed-output semantics that decode-only hash sampling may miss?
5. Does RVM recurrent state require additional frame-boundary or stream synchronization when fed from ThreadedDecoder batch/prefetch output?
6. Should alpha + ThreadedDecoder be permanently disabled, or is there an official retain/copy/sync API that would make it safe?

## Current Recommendation

Until external review resolves the root cause:

- keep production alpha passthrough on `SimpleDecoder`
- keep `ThreadedDecoder` available for green-screen passthrough and decode-only performance paths
- do not re-enable threaded alpha production based only on simple copy/sync changes

