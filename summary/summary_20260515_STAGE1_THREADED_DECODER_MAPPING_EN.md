# Stage 1 Summary - ThreadedDecoder Frame Mapping Calibration

## Scope

This stage investigated whether `PyNvVideoCodec.ThreadedDecoder` can safely
replace or complement the current `SimpleDecoder[index]` path for future staged
8K passthrough work.

The key concern was frame identity. A mismatch between decoded frame content and
frame indices would be a production blocker for live passthrough.

## Key Finding

The apparent frame mismatch was caused by an invalid validation method.

The incorrect test called `ThreadedDecoder.end()` before reading the returned
GPU frames. NVIDIA's ThreadedDecoder frames must be consumed before the next
`get_batch_frames()` call and before decoder shutdown.

After correcting frame lifetime handling, decoded frame content maps stably as:

```text
ThreadedDecoder(start_frame=N) sequence K == SimpleDecoder[N + K]
```

Frame content was validated by Y/UV SHA256 hashes.

## Important Constraint

`ThreadedDecoder.getPTS()` does not match `SimpleDecoder.getPTS()` exactly for
the same visual frame on `videos/test_8k_2.mp4`.

Observed PTS deltas were stable:

- normal frames: `2002` ticks;
- first frame from `start_frame=0`: `3003` ticks.

Therefore future code must not use `getPTS()` as the frame identity. It must use
an explicit source-frame counter.

## Validation

Added:

- `tools/pynv_threaded_decode_probe.py`
- `tools/pynv_threaded_mapping_probe.py`

Formal mapping validation:

```powershell
uv run python tools\pynv_threaded_mapping_probe.py videos\test_8k_2.mp4 --frames-per-start 16 --repeats 3 --buffer-size 32
```

Result:

- `150/150` cases passed;
- starts: `0,1,2,3,4,10,30,58,120,300`;
- batch sizes: `1,2,4,8,16`;
- repeats per case: `3`;
- all checked frames matched `SimpleDecoder[index]` by Y/UV SHA256.

Artifacts:

- `baseline/pynv_threaded_mapping_phase2_20260515_102152.md`
- `baseline/pynv_threaded_mapping_phase2_20260515_102152.json`

## Decision

ThreadedDecoder frame mapping is considered calibrated and stable under the
tested conditions.

Future staged pipeline work may use ThreadedDecoder if it follows these rules:

- consume returned frames before the next `get_batch_frames()` call;
- do not read returned frames after `end()`;
- track identity with `expected_index = start_frame + local_sequence`;
- keep `SimpleDecoder` as a fallback.

No external expert summary is needed for this issue.
