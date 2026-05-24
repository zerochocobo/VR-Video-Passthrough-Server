# TensorRT Static RVM Runtime Summary

Date: 2026-05-21

## Background

This stage continued the TensorRT work for realtime RVM matting. The earlier dynamic ONNX Runtime TensorRT EP path had two confirmed problems:

- `TensorRTExecutionProvider + RVM IOBinding` can block inside `run_with_iobinding()`, causing first-chunk timeout.
- `TensorRTExecutionProvider + sess.run()` is stable, but ORT partitions RVM into fallback-heavy subgraphs, making inference seconds per frame and slower than CUDA + IOBinding.

Expert feedback pointed to ORT TensorRT EP dynamic graph partition fragmentation and recommended a static-shape ONNX path.

## Completed

1. Added static RVM ONNX generation:
   - New file: `utils/rvm_static_onnx.py`.
   - Generates TensorRT-specific fixed-shape ONNX files for batch 1 and batch 2.
   - Freezes `src`, `r1i..r4i`, `fgr`, `pha`, and `r1o..r4o` shapes.
   - Removes runtime graph input `downsample_ratio`.
   - Replaces the dynamic Resize scale tensor `388` with constant `[1, 1, downsample, downsample]`.

2. Added the static TensorRT runtime fast path:
   - `Matter` detects cached static batch1/batch2 TRT ONNX files.
   - When static cache exists, the main RVM session uses CUDA/CPU only for metadata and fallback.
   - Realtime inference uses separate static TensorRT sessions with CUDA IOBinding.
   - This avoids the dynamic TensorRT `Resize_3` parser failure in the main runtime session.

3. Fixed RVM recurrent state handling:
   - Added robust channel fallback for `r1/r2/r3/r4` state tensors.
   - Static TensorRT recurrent state and output OrtValue caches now participate in the existing SBS slot mechanism, preventing left/right-eye state contamination.

4. Updated TensorRT warmup/cache build:
   - `ui/services/trt_warmup_process.py` now generates static batch1 and batch2 ONNX files.
   - Static engines are built directly via ORT TensorRT sessions.
   - Removed the old dynamic stage-3 build/solidification slow path.
   - Formal build result: `DONE:total_seconds=186`; stage 3 is now `0s`.

5. Improved diagnostics:
   - Matting logs now show a provider diagnostic like:
     `providers={'main': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'static_trt': True, 'iobinding': True}`
   - This prevents misreading CUDA/CPU main-session providers as “TensorRT is disabled”.

6. Disabled per-frame debug logs that limited FPS:
   - Commented out these `pipeline/matting.py` logs:
     - `nv12->nv12 y kernel returned ...`
     - `nv12->nv12 uv kernel returned ...`
     - `nv12->nv12 mono kernel returned ...`
   - The reported log line:
     `nv12->nv12 mono kernel returned in 0.02xms`
     was an info-level per-frame diagnostic. It polluted `server.log` and could affect realtime FPS when debug logging was enabled.
   - These kernel-return logs are now off by default and should only be restored locally for focused CUDA kernel profiling.

## Local Verification

- TensorRT EP is available in the `uv` environment:
  - `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- Static TRT runtime probe:
  - main session providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`
  - `static_trt=True`
  - `iobinding=True`
  - later `(2,3,1024,1024)` calls are around `5-10ms`
- Formal warmup:
  - command: `.\.venv\Scripts\python.exe main.py trt_warmup --input-size 1024 --downsample 0.5 --fp16 1 --cuda-graph 0 --progress-stdout`
  - result: `DONE:total_seconds=186`
  - cache status: `ready`
- Tests:
  - `tests/test_rvm_static_onnx.py`
  - `tests/test_trt_warmup_process.py`
  - `tests/test_trt_manifest.py`
  - `tests/test_matting_runtime_policy.py`
  - result: `13 passed`
- Compile checks passed for:
  - `pipeline/matting.py`
  - `ui/services/trt_warmup_process.py`
  - `utils/rvm_static_onnx.py`

## Remaining Issues

1. First static TRT inference still has lazy setup overhead:
   - The first probe call can take hundreds of milliseconds.
   - Later calls are already down to `5-10ms`.
   - A follow-up can preload static batch1/batch2 sessions during server startup or warmup to reduce first-frame latency.

2. End-to-end 4K/8K playback FPS still needs real playback validation:
   - RVM static TRT inference itself is fast.
   - Full-path FPS still depends on decode, preprocess, matting, NV12 composite, encode, mux, and DLNA client backpressure.

3. Static TensorRT sessions still print some warnings:
   - Mainly unused/empty initializer warnings.
   - They do not currently affect cache readiness or inference performance.
   - If logs remain too noisy, the next cleanup is removing unused initializers from generated static ONNX files.

4. TensorRT cache build is still long:
   - Current formal build is about 186 seconds.
   - This is much shorter than the old dynamic path, but initial build still requires waiting.
   - The UI can later show clearer stage timing and user guidance.

5. Dynamic ORT TensorRT EP should not be the realtime primary path:
   - Dynamic RVM still has the `Resize_3` parser/partition issue.
   - Product runtime should prefer static TRT + IOBinding, with CUDA + IOBinding retained as a stable fallback.

