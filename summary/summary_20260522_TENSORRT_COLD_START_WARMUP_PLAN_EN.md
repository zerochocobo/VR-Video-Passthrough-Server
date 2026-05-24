# TensorRT Cold-Start Warmup Optimization Plan

Date: 2026-05-22

## Background

The static TensorRT RVM path is now close to the target steady-state performance. For `[VenusReality]Hannah02-8K.mp4`, after disabling live adaptive FPS, output recovered from the 24 FPS cap to nearly 40 FPS:

- `source_fps=59.940 output_fps=40.000 fps_cap=40.000`
- `static_trt=True, iobinding=True`
- steady `ort_run` is commonly `5.5-8.7ms`
- later cumulative FPS reaches `39.9FPS`

However, first-frame / first-chunk startup still has visible latency. Players may include this startup phase in their average FPS display, or users may experience it as slow startup.

## Log Evidence

From the same request in `debug_output/server.log`:

- request start: `12:18:36.144`
- first RVM diagnostic: `alpha #1 ... ort_run=1133.6ms`
- worker start: `12:18:38.856`
- source meta: `output_fps=40.000`
- batch=2 static TRT session loads during the request
- first real video bitstream: `12:18:39.867`
- first stdout chunk: around `12:18:41`

The first intervals show low cumulative FPS:

- frame 30: `fps=17.60`
- frame 60: `fps=23.22`
- frame 120: `fps=29.22`

Steady state recovers:

- frame 600: `fps=39.03`
- frame 900: `fps=39.62`
- frame 1110: `fps=39.90`

## Current Interpretation

The first-frame delay is not primarily a steady-state TensorRT performance issue. It is a stack of cold-start costs:

1. First ONNX Runtime / static TensorRT session load.
2. Current warmup does not fully match the real batch=2 SBS/alpha path.
3. First CUDA / ORT / TensorRT calls trigger context, buffer, kernel, and engine runtime initialization.
4. PyNv / NVDEC / NVENC / FFmpeg mux pipeline creation also has fixed startup cost.

## Proposed Options

### Option A: Background Warmup of Static TensorRT Sessions on Server Startup

After server startup, launch a background warmup thread that creates `Matter` and loads static TensorRT RVM sessions.

Recommended warmup scope:

- batch=1 static RVM session
- batch=2 static RVM session
- current `MATTING_INPUT_SIZE=1024`
- current `RVM_DOWNSAMPLE_RATIO=0.5`
- current `ONNX_TRT_FP16_ENABLE=1`
- keep CUDA Graph disabled as currently configured

Goal: move session load / warmup cost from first playback to server startup or idle background time.

Pros:

- Relatively small change surface.
- Does not alter the main playback path.
- User-visible wait shifts to server startup/background time.

Risks:

- GPU memory is occupied earlier.
- If UI/config changes model, input size, downsample, or provider settings, warmup must be invalidated and rerun.
- Need locking/state guards so warmup and a real request do not build the same TensorRT session concurrently.

### Option B: Warm Up the Actual Batch=2 Playback Path

Logs show an initial slow batch=1 call, then batch=2 static session loading during the request. SBS alpha steady state uses batch=2, so warmup should cover the actual path.

Recommendations:

- warmup API should explicitly support a batch list, e.g. `[1, 2]`
- alpha/SBS mode should at least warm batch=2
- warmup logs should report batch, shape, provider, and elapsed time

### Option C: Global Session / Matter Runtime Pool

Create a runtime cache keyed by:

- model path
- providers
- input size
- downsample ratio
- fp16 / cuda graph / trt cache path

Potentially shareable:

- ORT InferenceSession
- static TRT sessions
- fixed model metadata

Must remain per-stream or carefully isolated:

- RVM recurrent state OrtValue / CuPy buffers
- per-stream input/output binding buffers
- per-stream CUDA buffer lifecycle

This may produce the largest benefit, but state isolation risk should be reviewed carefully.

### Option D: Lightweight PyNv / NVDEC / NVENC Preinitialization

Run a lightweight background preflight:

- initialize CUDA/PyNv runtime
- create and release a small or target-size NVENC encoder
- optionally create a decoder preflight

Expected benefit is smaller than TensorRT session warmup, but it may reduce first-chunk jitter.

## Recommended Implementation Order

1. Implement background warmup for batch=1 + batch=2 static TRT sessions.
2. Add locking/status guards to avoid concurrent warmup and real-request builds.
3. Expose warmup state in logs or UI: `idle / warming / ready / failed`.
4. Re-evaluate whether a global session pool is needed.
5. Evaluate PyNv/NVENC/NVDEC preflight last.

## Questions for Expert Review

1. Can ORT InferenceSession be safely shared across streams? If yes, is it enough to isolate RVM recurrent state and IOBinding buffers?
2. Are static TensorRT sessions already cached inside `Matter`? If so, what is the cache granularity?
3. Can warmup pollute RVM recurrent state? Should it use throwaway state only?
4. Is batch=1 still necessary, or can alpha/SBS focus on batch=2 warmup?
5. Are there TensorRT engine/context restrictions per thread or per CUDA stream?
6. Is the extra GPU memory footprint acceptable? Should warmup be cancelable or delayed until TensorRT mode is selected?
7. Does the packaged exe path require special DLL/path initialization ordering before background warmup?

## Current Recommendation

Start with Options A+B: background warmup of batch=1 and batch=2 static TensorRT RVM sessions after server startup. Do not implement a global session pool yet; keep RVM state isolation separate from the cold-start optimization.
