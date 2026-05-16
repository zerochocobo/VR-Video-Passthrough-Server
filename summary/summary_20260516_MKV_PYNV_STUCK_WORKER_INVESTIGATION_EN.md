# MKV PyNv Stuck Worker Investigation Summary

Date: 2026-05-16

## Background

During live alpha passthrough testing, red alpha-layer flicker appeared after playing `test_mkv_8k.mkv` and then switching to MP4 videos. The MP4 videos could still run, but the flicker persisted after the MKV playback path had been exercised.

## Findings

- The MKV stream could leave a `pynv-worker` thread alive after stream preemption/close.
- The stuck worker did not reach `worker_done`.
- The captured stack showed the worker blocked inside PyNvVideoCodec native frame retrieval:

```text
pipeline/pynv_stream.py: frame = self._dec.frame_at(src_idx)
pipeline/pynv_io.py: frame = self._decoder[index]
PyNvVideoCodec SimpleDecoder.__getitem__
```

- This points to `SimpleDecoder[index]` random frame access on the MKV source as the risky boundary.
- Once the worker is stuck in native code, Python cannot safely kill that thread inside the same process.
- Continuing MP4 alpha playback in the same process may still share contaminated CUDA/NVDEC/NVENC state, which explains why the red alpha flicker can persist after the MKV test.

## Temporary Attempts

- Added diagnostic VRAM logging and stuck-thread stack logging.
- Added a process-level PyNv taint marker when a worker failed to stop.
- Temporarily tried to block or warn on alpha playback after taint.
- Started exploring a sequential `ThreadedDecoder` route for MKV to avoid `SimpleDecoder[index]`.

These changes were useful for diagnosis, but they were not accepted as the final fix. The code changes are being rolled back so MKV handling can be redesigned cleanly.

## Current Direction

The next investigation should focus on replacing MKV live alpha decoding with a safer route:

- avoid `SimpleDecoder[index]` random frame access for MKV;
- test PyNv `ThreadedDecoder` sequential decoding as a possible replacement;
- compare frame correctness and shutdown behavior;
- consider isolating risky MKV PyNv decode in a subprocess if native decoder shutdown cannot be made reliable.

## Verification Data

The key evidence was collected from `debug_output/server.log` around the MKV-to-MP4 test. The strongest signal is the stuck stack in `SimpleDecoder.__getitem__`.
