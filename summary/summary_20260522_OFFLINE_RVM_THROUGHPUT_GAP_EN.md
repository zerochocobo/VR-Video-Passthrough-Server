# Offline RVM Throughput Gap Summary

Date: 2026-05-22

## Problem

For the same `videos/test_8k.mp4`, realtime alpha mode with `PT_PASSTHROUGH_MAX_FPS=0` reaches about `77fps`, while offline RVM TensorRT generation stays around `36fps`. Offline timing is dominated by `decode_avg ~= 19-21ms`; RVM TensorRT itself is only about `5.8-6.0ms`.

## Current Conclusion

Offline output FPS is not capped. Logs show `output_fps=59.940060`, and `target=899` matches a 15-second source segment.

TensorRT/RVM inference is active:

- `static_trt=True`
- `rvm_iobinding=True`
- `rvm_ort_avg ~= 5.8-6.0ms`

The realtime reference in `debug_output/server.log` is also alpha, simple decoder, serial worker, not threaded decoder:

- `output_mode=alpha`
- `decoder=simple`
- `worker_mode=serial`
- `fps_cap=0.000`
- steady-state roughly `decode=5-6ms`, `composite=6ms`, total throughput around `77fps`

Offline remains:

- green: `throughput ~= 36.4fps`
- alpha: `throughput ~= 36.3-36.4fps`
- latest alpha probe: `decode_avg=20.722ms`, `matting_avg=6.227ms`, `decode_matting_avg=26.949ms`

So the remaining gap is that `PyNvSimpleDecoder.frame_at()` waits much longer in the offline process than in the realtime path. The root cause is not RVM/TensorRT, alpha packing, audio, output FPS, source-index rewind, or NVENC argument differences.

## Ruled Out

1. Output FPS cap

Offline logs show `output_fps=59.940060`; there is no 30fps or other cap.

2. TensorRT not active

Offline RVM logs show `static_trt=True`, `rvm_ort_avg ~= 5.8ms`, matching realtime `ort_run ~= 5.4-6.0ms`.

3. Concurrent audio read from the source file

Green offline was changed to extract an AAC sidecar before video processing, run video-only processing, then mux at the end. User retested; no improvement.

4. Green composite vs alpha packing

Offline alpha also remains around `36fps`, and `rvm_alpha_pack_avg=0.063-0.064ms`; alpha packing is not the bottleneck.

5. Source-index duplicate/rewind

Offline green/alpha loops now use the same monotonic `src_idx` guard as realtime. Latest logs show:

- `source_index_rewinds = 0`

No improvement.

6. Matter/ORT initialization order

RVM offline now initializes the RVM engine/Matter/ORT/TRT before creating the PyNv decoder, closer to realtime. No improvement.

7. NVENC argument differences

RVM offline now appends `--realtime-encoder-args`. Latest logs confirm:

- command includes `--realtime-encoder-args`
- `encoder_kwargs={'bitrate': '10086258'}`

Throughput remains `36.32fps`.

## Changed Files

- `pipeline/matting.py`
  - Reduces normal TensorRT/ORT warning noise.
  - Stops retrying static TRT sessions after activation failure.

- `offline/convert.py`
  - Appends `--realtime-encoder-args` for RVM offline diagnostics.
  - Keeps TensorRT providers only for RVM fast when cache is ready.

- `tools/offline_passthrough.py`
  - Green offline audio now uses extract-audio, video-only processing, final mux.
  - RVM green uses `Matter.acquire_nv12_output_slot()` plus a pending ring, closer to realtime.
  - Adds `sync`, `dec+mat+sync`, `source_index_rewinds`, and `encoder_kwargs` logs.
  - Initializes Matter/ORT/TRT before creating the decoder for RVM.

- `tools/offline_alpha_passthrough.py`
  - Adds `dec+mat`, `source_index_rewinds`, and `encoder_kwargs` logs.
  - Supports `--realtime-encoder-args`.
  - Initializes Matter/ORT/TRT before creating the decoder for RVM.
  - Keeps compatibility with alpha output-pixel bitrate scaling tests.

## Verification

Passed:

```powershell
.\.venv\Scripts\python.exe -m compileall offline\convert.py tools\offline_passthrough.py tools\offline_alpha_passthrough.py
.\.venv\Scripts\python.exe -m pytest tests\test_offline_convert.py tests\test_settings.py tests\test_offline_alpha_bitrate.py
git diff --check
```

Test result: `20 passed`.

## Key Log Comparison

Realtime alpha from `debug_output/server.log`:

- `decoder=simple`
- `worker_mode=serial`
- `output_mode=alpha`
- `fps_cap=0.000`
- `output_fps=59.940`
- steady-state `decode=5-6ms`
- steady-state `composite=~6ms`
- total throughput around `77fps`

Latest offline alpha user retest:

- command includes `--realtime-encoder-args`
- `encoder_kwargs={'bitrate': '10086258'}`
- `source_index_rewinds = 0`
- `decode_avg = 20.722 ms`
- `matting_avg = 6.227 ms`
- `decode_matting_avg = 26.949 ms`
- `throughput = 36.32 fps`

## Remaining Suspects

The most likely remaining area is contextual behavior of `PyNvSimpleDecoder.__getitem__` / `frame_at()` in the offline tool process versus the realtime worker. This is not explained by decoder type, index order, RVM, NVENC arguments, or audio concurrency.

Suggested expert next steps:

1. Check whether the offline script's MP4 output mux, file writes, or same-process ffmpeg pipe backpressure affects the next `frame_at()` call, while realtime StreamingResponse/mpegts/cache behavior does not.
2. Check whether the realtime path has hidden warmup, reader subscription, cache, or server lifetime state that makes SimpleDecoder use a faster sequential path.
3. Add a minimal A/B probe under the same `base_environment()`: sequentially call `PyNvSimpleDecoder.frame_at()` for 899 frames with no RVM, no NVENC, no mux; then add RVM, NVENC, and ffmpeg pipe step by step to find what changes `frame_at()` from 5-6ms to ~20ms.
4. Do not run GPU/ORT tools naked. Use UI/`offline.convert`/`main.py`, or explicitly wrap with `ui.services.process_helpers.base_environment()`, otherwise CUDA provider loading can fail and produce invalid CPU-fallback results.
