# ORT CUDA cold-start report - 2026-05-08

## Executive summary

We observe an abnormal first-inference latency of about 145-175 seconds when running an ONNX Runtime CUDA Execution Provider inference for an RVM matting model.

The delay is isolated to the first CUDA EP execution of the ONNX graph. Model/session construction is fast, video decode is fast, preprocessing is fast, and later steady-state inference is fast.

This is not currently believed to be caused by PyNvVideoCodec, TensorRT EP selection, ONNX Runtime IOBinding, or a custom/shared CUDA stream. Those were tested separately.

Update after a targeted cache experiment: the strongest current explanation is NVIDIA driver PTX JIT compilation for Blackwell / sm_120. With a fresh `CUDA_CACHE_PATH`, the first new process took about 154.6 seconds in ORT, then a second new process using the same cache took only about 0.8 seconds. The cache directory contained 25 files totaling about 212 MB after the first run.

## Application context

The application is a local VR/DLNA media server. It decodes VR video frames, runs a video matting model to generate an alpha mask, composites the original frame with a green background, and then encodes/muxes the result for DLNA playback.

The relevant model is Robust Video Matting (RVM), `rvm_mobilenetv3_fp32.onnx`, default input size 512. For side-by-side VR content the code often runs batch size 2 with input shape similar to:

```text
(2, 3, 512, 512)
```

The RVM graph has recurrent state inputs and outputs:

```text
inputs:
  src
  r1i
  r2i
  r3i
  r4i
  downsample_ratio

outputs:
  fgr
  pha
  r1o
  r2o
  r3o
  r4o
```

## Observed environment

```text
OS: Windows, WDDM GPU mode
GPU: NVIDIA GeForce RTX 5060 Ti 16GB
NVIDIA driver: 581.57
nvidia-smi CUDA version: 13.0
Python: 3.12.12
onnxruntime: 1.25.1
CuPy: 14.0.1
CuPy CUDA runtime: 12090
NumPy: 2.0.2
OpenCV: 4.13.0
PyNvVideoCodec: 2.1.0
ORT available providers:
  TensorrtExecutionProvider
  CUDAExecutionProvider
  CPUExecutionProvider
Active providers in the failing/slow runs:
  CUDAExecutionProvider
  CPUExecutionProvider
```

GPU compute capability from CuPy:

```text
name=NVIDIA GeForce RTX 5060 Ti
compute_capability=12.0
```

Project dependency declaration:

```toml
requires-python = ">=3.12,<3.13"
dependencies = [
    "onnxruntime-gpu>=1.19",
    "numpy>=1.26,<2.1",
    "opencv-python>=4.10",
    "pillow>=10",
    "cupy-cuda12x",
    "pynvvideocodec>=2.1.0",
]
```

Relevant model files available locally:

```text
rvm_mobilenetv3_fp32.onnx  14,975,696 bytes
rvm_mobilenetv3_fp16.onnx   7,503,483 bytes
rvm_resnet50_fp32.onnx    107,479,165 bytes
rvm_resnet50_fp16.onnx     53,752,431 bytes
```

The reported cold-start issue is with the RVM fp32 MobilenetV3 model.

## Symptom details

### Key timing observations

From a PyNv decode + matting probe:

```text
matting import: 0.728s
decoder init: 0.177s
Matter init: 0.524s
first decode: 0.065s
first mat total: 159.409s
first preprocess: 0.052s
first ort: 159.350s
first composite: 0.006s
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Forced providers:

```text
--providers CUDAExecutionProvider,CPUExecutionProvider
first ort: 159.481s
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Forced CUDA providers plus cuDNN heuristic search:

```text
--cuda-cudnn-search HEURISTIC
first ort: 173.687s
```

RVM IOBinding disabled:

```text
PT_RVM_IOBINDING=0
first ort: 174.629s
rvm_iobinding=False
```

Shared CUDA stream disabled:

```text
PT_CUDA_SHARED_STREAM=0
first ort: 174.984s
```

Pure ORT cold probe, no PyNvVideoCodec:

```text
uv run python tools\ort_cold_probe.py --split-sbs --alpha-stride 3 --input-size 512 --no-warmup --providers CUDAExecutionProvider,CPUExecutionProvider --cuda-cudnn-search HEURISTIC

matting import: 0.607s
Matter init: 0.511s
first mat total: 145.080s
first preprocess: 0.048s
first ort: 145.022s
first composite: 0.009s
```

### Steady-state is fast

After the first ORT CUDA execution, the system reaches normal throughput. Examples:

4K PyNv fullchain with matting:

```text
stride=1 steady_fps ~= 50.65 fps
steady_ort_avg ~= 17.749 ms
```

8K PyNv fullchain with matting:

```text
stride=3 steady_fps ~= 34.86 fps
steady_ort_avg ~= 5.563 ms

stride=1 steady_fps ~= 31.90 fps
steady_ort_avg ~= 20.263 ms
```

This strongly suggests the problem is a one-time lazy initialization/compilation/cache/security-scan style cost, not normal inference speed.

### ComputeCache experiment

To test whether this happens every process start or only when NVIDIA's driver compute cache is cold, a new empty cache directory under the project workspace was used:

```bat
set CUDA_CACHE_PATH=G:\GIT\debug\PTMediaServer\debug_output\cuda_compute_cache_probe_20260508_092334
set CUDA_CACHE_MAXSIZE=2147483648
```

Then the same pure ORT cold probe was run twice as two separate Python processes with the same cache path.

Run 1:

```text
Matter init: 3.227s
first mat total: 155.668s
first preprocess: 0.952s
first ort: 154.559s
first composite: 0.087s
```

Cache after run 1:

```text
files=25
bytes=212,023,684
```

Run 2:

```text
Matter init: 0.505s
first mat total: 0.825s
first preprocess: 0.036s
first ort: 0.777s
first composite: 0.010s
```

Interpretation:

- The delay is not expected on every program start if the NVIDIA ComputeCache remains valid and readable.
- It is expected after cache deletion, cache invalidation, driver/runtime changes, GPU architecture changes, or when using a fresh cache path.
- This result strongly supports the PTX JIT / driver ComputeCache hypothesis.

## Reproduction probe

The minimal local probe used to reproduce the issue without PyNvVideoCodec is:

```python
"""
Measure ONNX Runtime cold-start cost without PyNvVideoCodec.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Matter/ORT cold-start without video decode.")
    parser.add_argument("--providers", default="", help="override providers")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--split-sbs", action="store_true")
    parser.add_argument("--alpha-stride", type=int, default=3)
    parser.add_argument("--cuda-cudnn-search", default="")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    config.MATTING_INPUT_SIZE = int(args.input_size)
    if args.providers:
        config.ONNX_PROVIDERS = [p.strip() for p in args.providers.split(",") if p.strip()]
        os.environ["PT_ONNX_PROVIDERS"] = ",".join(config.ONNX_PROVIDERS)
    if args.split_sbs:
        os.environ["PT_MATTING_SPLIT_SBS"] = "1"
        os.environ["PT_MATTING_SBS_BATCH"] = "1"
    if args.cuda_cudnn_search:
        os.environ["PT_CUDA_CUDNN_CONV_ALGO_SEARCH"] = args.cuda_cudnn_search
    if args.no_warmup:
        config.MATTING_WARMUP_RUNS = 0
    os.environ["PT_ALPHA_STRIDE"] = str(max(1, args.alpha_stride))

    t0 = time.perf_counter()
    print("[ort-cold] import matting", flush=True)
    from pipeline.matting import Matter

    print(f"[timeline] matting import: {time.perf_counter() - t0:.3f}s", flush=True)
    t0 = time.perf_counter()
    print("[ort-cold] Matter init", flush=True)
    matter = Matter()
    print(f"[timeline] Matter init: {time.perf_counter() - t0:.3f}s", flush=True)

    import numpy as np

    h, w = (2048, 4096) if args.split_sbs else (512, 512)
    nv12 = np.zeros((h * 3 // 2, w), dtype=np.uint8)
    t0 = time.perf_counter()
    print("[ort-cold] first mat", flush=True)
    _, timing = matter.composite_green_nv12_to_nv12_profile(nv12.reshape(-1), h, w)
    elapsed = time.perf_counter() - t0
    print(f"[timeline] first mat total: {elapsed:.3f}s")
    print(f"[timeline] first preprocess: {timing.preprocess_ms / 1000.0:.3f}s")
    print(f"[timeline] first ort: {timing.ort_ms / 1000.0:.3f}s")
    print(f"[timeline] first composite: {timing.composite_ms / 1000.0:.3f}s")
    return 0
```

Representative command:

```bat
set UV_CACHE_DIR=G:\GIT\debug\PTMediaServer\.uv-cache
uv run python tools\ort_cold_probe.py ^
  --split-sbs ^
  --alpha-stride 3 ^
  --input-size 512 ^
  --no-warmup ^
  --providers CUDAExecutionProvider,CPUExecutionProvider ^
  --cuda-cudnn-search HEURISTIC
```

Representative output:

```text
matting import: 0.607s
Matter init: 0.511s
first mat total: 145.080s
first preprocess: 0.048s
first ort: 145.022s
first composite: 0.009s
```

## Relevant ONNX Runtime setup code

Session creation:

```python
providers = _filter_available_providers(ONNX_PROVIDERS)
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if "DmlExecutionProvider" in providers:
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
provider_config = _provider_config(providers)
self.sess = ort.InferenceSession(
    str(model_path), sess_options=sess_opts, providers=provider_config
)
```

CUDA provider options:

```python
def _provider_config(providers: list[str]):
    configured = []
    for provider in providers:
        if (
            provider == "CUDAExecutionProvider"
            and _CUDA_STREAM is not None
            and CUDA_SHARED_STREAM
        ):
            cuda_options = {
                "user_compute_stream": str(int(_CUDA_STREAM.ptr)),
                "do_copy_in_default_stream": "0",
            }
            cudnn_search = os.environ.get("PT_CUDA_CUDNN_CONV_ALGO_SEARCH", "").strip()
            if cudnn_search:
                cuda_options["cudnn_conv_algo_search"] = cudnn_search
            configured.append((provider, cuda_options))
        else:
            configured.append(provider)
    return configured
```

Important note: even when shared stream and IOBinding are disabled, the first ORT run still costs about 175 seconds. So the custom stream and IOBinding path are unlikely to be the root cause.

## Relevant inference code

The fast path uses ONNX Runtime IOBinding with GPU-resident CuPy input:

```python
def _run_rvm_iobinding_from_dev(self, x_dev):
    batch, _, h, w = (int(v) for v in x_dev.shape)
    self._reset_rvm_rec_ort_if_needed(batch, h, w)

    binding = self.sess.io_binding()
    binding.bind_input(
        self.input_name,
        "cuda",
        0,
        self.input_dtype,
        tuple(x_dev.shape),
        int(x_dev.data.ptr),
    )
    for name, state in zip(self.input_names[1:5], self._rvm_rec_ort):
        binding.bind_ortvalue_input(name, state)
    if len(self.input_names) >= 6 and self._rvm_io_downsample is not None:
        binding.bind_ortvalue_input(self.input_names[5], self._rvm_io_downsample)

    for meta in self.output_metas:
        binding.bind_output(meta.name, "cuda", 0)

    self.sess.run_with_iobinding(binding)
    outputs = binding.get_outputs()
```

The fallback path uses normal `sess.run()`:

```python
def _run_rvm(self, x: np.ndarray) -> np.ndarray:
    if x.dtype != self.input_dtype:
        x = x.astype(self.input_dtype)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    batch, _, h, w = x.shape
    self._reset_rvm_rec_if_needed(int(batch), int(h), int(w))

    feed: dict[str, np.ndarray] = {self.input_name: x}
    rec_inputs = self.input_names[1:5]
    for name, rec in zip(rec_inputs, self._rvm_rec or []):
        feed[name] = rec
    if len(self.input_names) >= 6:
        feed[self.input_names[5]] = np.asarray([RVM_DOWNSAMPLE_RATIO], dtype=self.rvm_downsample_dtype)

    outputs = self.sess.run(self.output_names, feed)
```

Disabling IOBinding and using this fallback path did not remove the cold-start delay.

## Relevant model/session metadata observed at runtime

```text
model kind=rvm
input=src
shape=['batch_size', 3, 'height', 'width']
type=tensor(float)
batch2=True
model_batch2=True
rvm_iobinding=True
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
inputs=['src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio']
outputs=['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']
```

ONNX Runtime warning observed:

```text
1 Memcpy nodes are added to the graph torch-jit-export for CUDAExecutionProvider.
It might have negative impact on performance (including unable to run CUDA graph).
Set session_options.log_severity_level=1 to see the detail logs before this message.
```

This warning appears both in normal probes and 8K fullchain probes. It may or may not be related to cold-start.

## What has been ruled out so far

### TensorRT EP accidentally building an engine

Available providers include TensorRT:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

But the cold-start reproductions forced active providers to:

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

The delay remained around 145-175 seconds.

### PyNvVideoCodec

Pure ORT probe without video decode or PyNvVideoCodec still reproduced the delay:

```text
first ort: 145.022s
```

### ONNX Runtime IOBinding

Disabling RVM IOBinding still reproduced the delay:

```text
PT_RVM_IOBINDING=0
first ort: 174.629s
```

### Custom/shared CUDA stream

Disabling shared CUDA stream still reproduced the delay:

```text
PT_CUDA_SHARED_STREAM=0
first ort: 174.984s
```

### cuDNN exhaustive convolution algorithm search

Setting CUDA EP provider option:

```text
cudnn_conv_algo_search=HEURISTIC
```

did not improve cold-start in this environment. One measured run was worse:

```text
first ort: 173.687s
```

## Hypotheses still open

These are not proven. They are listed for external review.

1. ONNX Runtime CUDA EP 1.25.1 + CUDA 12.x runtime + RTX 50-series first execution path is doing unusually expensive lazy initialization, JIT, or library loading.

2. cuDNN 9 / CUDA libraries are doing first-use kernel selection or compilation not affected by `cudnn_conv_algo_search=HEURISTIC`.

3. NVIDIA driver compute cache is missing, disabled, redirected, or repeatedly invalidated.

4. Windows Defender or another security scanner is scanning CUDA/ORT/cuDNN DLLs or generated cache artifacts on first use.

5. The ONNX graph contains an operator that falls into a slow CUDA EP first-run path, possibly related to the warning about inserted Memcpy nodes.

6. The dynamic shapes in the RVM model (`batch_size`, `height`, `width`) cause a first-run specialization/compilation cost.

7. There may be an ORT 1.25.1 regression with this model/runtime combination; older ORT GPU versions may behave differently.

8. Current high-confidence hypothesis after the cache experiment: the ORT CUDA wheel likely does not contain precompiled sm_120 cubins for all kernels used by this graph, so the NVIDIA driver compiles embedded PTX to sm_120 SASS on first use and stores the result in ComputeCache.

## Questions for experts

1. Is a 145-175 second first `InferenceSession.run()` / `run_with_iobinding()` on CUDA EP known for ONNX Runtime 1.25.x on Windows with CUDA 12/cuDNN 9?

2. What ORT CUDA EP logs should be enabled to pinpoint the exact phase? Suggested settings might include ORT verbose logging, provider-specific logs, CUDA module loading logs, or cuDNN frontend logs.

3. Can ORT CUDA EP persist or precompile any first-run artifacts for this kind of graph?

4. Could the inserted Memcpy node warning indicate a CPU fallback or layout transfer that triggers extremely slow first-run initialization?

5. Is there a recommended set of CUDA EP provider options for minimizing first-run latency on Windows?

6. Should we test ORT versions such as 1.19, 1.20, 1.22, 1.24, or nightly to identify a regression?

7. Is there a known interaction between RTX 50-series drivers, CUDA 12.x runtime wheels, and ORT 1.25.1 that would explain this?

8. How can we confirm or rule out Windows Defender / antivirus scanning of CUDA generated code or DLL loading?

9. Which cache directories should be inspected or excluded?
   Examples:
   - `%USERPROFILE%\.nv\ComputeCache`
   - CuPy cache directory
   - `%TEMP%`
   - project virtual environment DLL directories

10. Would exporting a static-shape ONNX model or simplifying the RVM recurrent inputs likely reduce first-run time?

## Suggested next diagnostics

These have not all been run yet.

1. Run the pure ORT cold probe twice in the same Windows session and compare first-run latency. This has now been done with a dedicated `CUDA_CACHE_PATH`; the first run took about 154.6s in ORT, while the second run took about 0.8s.

2. Check NVIDIA ComputeCache before and after a cold run:

```bat
dir "%USERPROFILE%\AppData\Roaming\NVIDIA\ComputeCache"
```

3. Temporarily add antivirus exclusions for:

```text
project directory
project .venv directory
project .uv-cache directory
%USERPROFILE%\AppData\Roaming\NVIDIA\ComputeCache
%TEMP%
```

Then rerun the pure ORT cold probe.

4. Enable ORT verbose logs:

```python
sess_opts.log_severity_level = 0
sess_opts.log_verbosity_level = 4
```

5. Try disabling graph optimizations for comparison:

```python
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
```

6. Try CPU EP only to separate model graph cost from CUDA provider cost.

7. Try the fp16 RVM model and smaller static input shapes, not for production quality but to see whether cold-start scales with graph/type.

8. Try older `onnxruntime-gpu` versions in a separate virtual environment.

9. Use Windows Process Monitor to identify file activity during the 145-175 second stall.

10. Use NVIDIA/CUDA profiling or Nsight Systems to see whether the process is CPU-blocked, loading DLLs, compiling kernels, or executing GPU kernels during the stall.

## Product impact

The current product can hide the delay by doing startup warmup before any DLNA playback request. However, the delay is too large to leave unexplained:

```text
145-175 seconds at first ORT CUDA execution
```

If this occurs on user machines, first launch experience will be poor unless a startup progress UI or prewarm process is implemented.

The steady-state performance is good enough for 4K and barely enough for 8K, so solving or hiding cold-start is the primary blocker before productionizing the PyNv backend.
