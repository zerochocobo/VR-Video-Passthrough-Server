# TensorRT Acceleration + Engine Cache UI Implementation Plan (English)

- Date: 2026-05-20
- Scope: Add a TensorRT acceleration toggle and engine cache management to the "Performance" panel on the desktop UI home page; provide first-time build progress feedback and detection of driver/version changes; ensure that disabling the toggle falls back losslessly to the CUDA path at any time.
- Out of scope: dynamic provider switching without restarting the service; cross-machine distribution of pre-built engines; TRT caching for offline tools (`offline/sam3_matanyone2.py`, `offline/yoloworld_efficientsam.py`).

---

## 1. Background

After v0.1.0-beta.1, 8K real-time passthrough dropped from 40-50 FPS to ~30 FPS. Root cause: `ALPHA_STRIDE` was changed from 3 to 1 for quality. Reverting causes alpha black crosses and loss of distant figures, so the quality decision stays. **The performance loss is recovered by accelerating RVM inference with TensorRT.**

This iteration uses `rvm_mobilenetv3_fp32.onnx` as the only inference model (the fp16 variant was verified to noticeably degrade matting quality in VR scenes and has been dropped). **All the speedup comes from the TensorRT FP16 backend compiling directly from the fp32 ONNX** — TRT builds the engine in FP16 precision (`trt_fp16_enable=1` is already on by default) while taking the fp32 ONNX weights as compile-time input. On RTX 2080 this typically yields 1.5-2x over CUDA EP for RVM. The first launch must compile engines (5-15 minutes); any change to driver / CUDA / TRT / ORT / model invalidates the cache. The task here is not "turn on TRT" — `ONNX_TRT_FP16_ENABLE` already defaults to 1, just `ONNX_PROVIDERS` does not include `TensorrtExecutionProvider`. The task is to give the user a **visible, controllable, reversible** UI flow.

**TensorRT acceleration targets only `rvm_mobilenetv3_fp32.onnx`.** No other models are in scope for this iteration.

**Input strategy**: fixed square `MATTING_INPUT_SIZE=1024` + `RVM_DOWNSAMPLE_RATIO=0.5`. A 2048 long-side + 0.125 downsample variant was tried but verified to lose subjects almost entirely in VR scenes, so we stay on the 1024 square layout — which also happens to be ideal for TRT: only two fixed shapes (`1×3×1024×1024` and `2×3×1024×1024`), each gets its own static engine with the most stable cache, best kernel selection, and no profile management to worry about.

## 2. Goals

- The user sees a "TensorRT acceleration" toggle in the home Performance panel with clear state semantics.
- Engines must be cached before first enable; the build shows staged progress with live counters and is cancellable.
- The toggle only becomes effective after caching is ready; after a service restart, load is sub-5s.
- Driver / CUDA / TRT / ORT / model changes are auto-detected as stale; the UI shows "needs rebuild".
- When the toggle is off or the cache is invalid, the service automatically falls back to CUDA EP without affecting playback.
- The cache exists independently of the toggle: turning off the toggle does not delete the cache; turning it back on uses the cache instantly.

## 3. Non-goals

- Do not implement ORT runtime provider hot-swap (ORT does not support it; forcing it introduces many corner cases).
- Do not bundle pre-built engines with the installer (engines are bound to a specific GPU/driver and not portable).
- Do not expose fine-grained engines (batch=1 / batch=2 / different recurrent state shapes) to the user; expose at model granularity only.
- Do not add any new model beyond RVM in this iteration.

## 4. Existing Pipeline Reference Points

- `pipeline/matting.py:880-882` already has provider-options injection for `TensorrtExecutionProvider`, honoring `ONNX_TRT_FP16_ENABLE` and `ONNX_TRT_CUDA_GRAPH_ENABLE`.
- `pipeline/matting.py:977-1003` creates `InferenceSession` via `_filter_available_providers(ONNX_PROVIDERS)`. `ONNX_PROVIDERS` default in `config.py:394-397` is `CUDAExecutionProvider,CPUExecutionProvider` — no TRT.
- `config.py:403-408`: `ONNX_TRT_ENGINE_CACHE_ENABLE=1`, `ONNX_TRT_ENGINE_CACHE_PATH=ROOT/runtime_cache/trt_engines`, `ONNX_TRT_FP16_ENABLE=1`, `ONNX_TRT_CUDA_GRAPH_ENABLE=1` are in place.
- `ui/services/server_process.py` already has start/stop/restart primitives for the service process.
- `ui/settings.py::server_env()` passes UI config to the service subprocess as `PT_*` env vars; this is the natural channel for the TRT toggle.
- `pipeline/matting.py:_supports_batch2` (around line 1003) decides whether to also build the batch=2 SBS engine.

## 5. Design

### 5.1 Three-Party Data Flow

```
UI Performance Panel (toggle / cache button / progress)
    │
    ├── Writes ui_settings.json (persists trt_enabled, etc.)
    ├── Reads runtime_cache/trt_engines/manifest.json (cache state)
    └── Spawns a standalone "warmup subprocess" (one-shot)
            │
            └── Invokes onnxruntime InferenceSession; TRT compiles and saves.
                On success writes manifest.json.
                On error cleans up half-finished engine files.

Main service process ptserver-server
    │
    └── At startup, based on trt_enabled + manifest validity,
        decides whether providers includes TensorrtExecutionProvider.
```

**Hard constraint**: the warmup runs in a **standalone subprocess**, isolated from the running service. Reasons:
- ORT InferenceSession is a blocking call and cannot be cancelled mid-compile.
- Compilation consumes large VRAM and competes with the service.
- Subprocess crash does not affect the service; the UI can kill the subprocess to "cancel".

### 5.2 Engine Cache Manifest (manifest.json)

Path: `runtime_cache/trt_engines/manifest.json`

```json
{
  "version": 1,
  "fingerprint": {
    "gpu_uuid": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "gpu_name": "NVIDIA GeForce RTX 2080",
    "driver_version": "560.94",
    "cuda_runtime": "12.4",
    "trt_version": "10.0.1.6",
    "ort_version": "1.20.0",
    "model_sha256": "ab12...",
    "matting_input_size": 1024,
    "rvm_downsample_ratio": 0.5,
    "trt_fp16": true,
    "trt_cuda_graph": true
  },
  "models": [
    {
      "key": "rvm_mobilenetv3",
      "label": "Robust Video Matting",
      "engines": [
        {"shape": "1x3x1024x1024", "size_mb": 38, "built_at": "2026-05-20T14:33:00Z"},
        {"shape": "2x3x1024x1024", "size_mb": 68, "built_at": "2026-05-20T14:39:30Z"}
      ],
      "total_build_seconds": 452,
      "status": "ready"
    }
  ],
  "built_at": "2026-05-20T14:39:30Z"
}
```

Fields:
- `fingerprint`: environment fingerprint; any change moves the cache into `stale`.
- `fingerprint.matting_input_size` / `rvm_downsample_ratio`: input-strategy fingerprint; changing the square side or the downsample ratio invalidates.
- `models[].status`: `ready` / `stale` / `failed`.
- `models[].total_build_seconds`: used to refine the next-time estimate.

### 5.3 UI Design

**Add a single row at the bottom of the Performance panel:**

```
TensorRT Acceleration    [○○]    [Configure]    Not cached
```

Layout (left to right):
1. Label: `TensorRT Acceleration`
2. Toggle (`QCheckBox` or a custom toggle widget)
3. Configure button: `[Configure]`
4. State text (short copy, no icons)

Four state strings:

| Cache state | Text |
|---|---|
| Missing | `Not cached` |
| Building | `Building…` |
| Ready (fresh) | `Cached` |
| Stale | `Rebuild needed` |

**All other information (last build duration, driver version, engine size, stage progress, etc.) goes into the popup dialog opened by `[Configure]`.** This single row in the main panel only carries two semantic bits: "enabled or not" and "ready or not".

Toggle interaction rules:
- When the cache state is `Cached`, the toggle is freely switchable; toggling it triggers the existing Performance panel `Save` → service-restart flow.
- In the other three states, the toggle is forced off and disabled; clicking it shows a tooltip: "Please complete the cache in Configure first."

Clicking `[Configure]` opens the dialog described in 5.4, which is the sole entry point for all details, build trigger, rebuild, delete, and progress display.

### 5.4 Configure Dialog

Clicking `[Configure]` opens a modal dialog. This dialog is the only entry point for all TensorRT details and operations.

#### 5.4.1 Missing / Ready (fresh) / Stale states

```
┌─ TensorRT Acceleration ───────────────────────┐
│                                                │
│  Model:    rvm_mobilenetv3_fp32.onnx           │
│  TRT prec: FP16 (compiled from fp32 ONNX)       │
│  GPU:     NVIDIA GeForce RTX 2080              │
│  Driver:  560.94                                │
│  TensorRT: 10.0.1.6                            │
│                                                │
│  Cache status: Cached                           │
│  Last build:   7m 32s                           │
│  Engine size:  106 MB                           │
│  Cache path:   runtime_cache/trt_engines/       │
│                                                │
│  ⓘ Enabling TensorRT runs inference via the    │
│    TensorRT FP16 backend, compiled directly    │
│    from the fp32 ONNX model.                    │
│                                                │
│  ⚠ The GPU is busy during the build. Do not   │
│    play video while building.                   │
│                                                │
│      [Delete cache]  [Close]  [Start build]     │
└────────────────────────────────────────────────┘
```

The primary bottom button label switches with the current state:

| State | Primary button | Secondary |
|---|---|---|
| Missing | `Start build` | — |
| Ready (fresh) | `Rebuild` | `Delete cache` |
| Stale | `Rebuild` (highlighted, with "driver upgraded" hint) | `Delete cache` |

#### 5.4.2 Building state

After clicking `Start build` / `Rebuild`, **the same dialog switches to a progress view** (no second-level modal):

```
┌─ TensorRT Acceleration ───────────────────────┐
│                                                │
│  Building rvm_mobilenetv3_fp32.onnx (TRT FP16)  │
│                                                │
│  [████████████░░░░░░░░░░░░] Stage 2 / 3        │
│                                                │
│  Current stage: Building SBS dual-eye engine    │
│  Elapsed:       04:21 / ETA 08-12 min           │
│  Engines built: 1                                │
│                                                │
│  ⚠ Closing this dialog cancels the build and   │
│    leaves the cache unusable.                   │
│                                                │
│                          [Cancel]               │
└────────────────────────────────────────────────┘
```

Progress composition (**no percentage promised, only stages**):
1. Stage 1/3: build single-eye engine (batch=1, 1024×1024)
2. Stage 2/3: build SBS dual-eye engine (batch=2, 1024×1024)
3. Stage 3/3: solidify runtime cache (warmup runs)

"Engines built" is driven by a `QFileSystemWatcher` on `runtime_cache/trt_engines/` counting `.engine` files.

When the build finishes, the dialog flips back to the 5.4.1 view with state `Cached`, and the main panel state text updates in sync.

### 5.5 Backend Implementation

#### 5.5.1 Warmup Subprocess

Add `ui/services/trt_warmup_process.py`. CLI entry:

```bash
ptserver-trt-warmup --model rvm --input-size 1024 --downsample 0.5 \
                    --fp16 1 --cuda-graph 1 \
                    --cache-dir runtime_cache/trt_engines \
                    --progress-stdout
```

Subprocess behavior:
1. Print `STAGE:1:start:Building single-eye engine`
2. Create the batch=1 session with fixed shape `1×3×1024×1024` (`r1i~r4i` derived via strides 4/8/16/32 are also fixed), run one inference to trigger TRT compilation
3. Print `STAGE:1:done`
4. Create the batch=2 session (if `_supports_batch2`) with fixed shape `2×3×1024×1024`, run one inference
5. Print `STAGE:2:done`
6. Run `MATTING_WARMUP_RUNS` warmups
7. Print `STAGE:3:done`
8. Write manifest.json
9. Exit code 0 = success, non-zero = failure

The UI uses `QProcess`, parses stdout line by line.

#### 5.5.2 Manifest Module

Add `utils/trt_manifest.py`:

```python
def collect_fingerprint() -> dict
def manifest_path() -> Path
def load_manifest() -> dict | None
def save_manifest(manifest: dict) -> None
def cache_status() -> Literal["missing", "ready", "stale", "failed"]
def stale_reasons(saved_fp: dict, actual_fp: dict) -> list[str]
def clear_cache() -> None  # delete all of runtime_cache/trt_engines/
```

`collect_fingerprint()` uses `pynvml` for GPU UUID / name / driver; `onnxruntime.__version__`; TRT version is read from the nvinfer DLL via ctypes file-version-info (the same approach already used in `prompt/PYINSTALLER_PACKAGING_REPORT`).

#### 5.5.3 Startup Validation

In `main.py`, before starting the service:
1. Read `ui_settings.trt_enabled`.
2. If true, call `cache_status()`:
   - `ready` → insert `TensorrtExecutionProvider` at the head of the providers list.
   - Anything else → log `trt cache invalid (reason=...), falling back to CUDA EP`, do not pass TRT, continue normal startup.
3. The UI main process performs the same check in parallel and updates the cache-state badge.

The service startup path **never triggers TRT compilation in the main process**. Caching only happens in the UI-initiated subprocess.

#### 5.5.4 settings.py Additions

Add to `ui/settings.py`:
- `inference_backend`: `"cuda" | "tensorrt"`, default `"cuda"`.
- `server_env()` when `inference_backend == "tensorrt"` and `cache_status() == "ready"`, injects `PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`; otherwise leaves the value untouched.

Note: `config.py` defaults need no change; the new env var simply overrides them.

### 5.6 State Machine

```
                     ┌──────────────┐
                     │  CUDA (default) │
                     └──────┬───────┘
                            │ Switch to TRT, no cache
                            ▼
                     ┌──────────────┐
                     │ Blocked: build first │
                     └──────┬───────┘
                            │ User clicks Build
                            ▼
                     ┌──────────────┐
                     │   Building   │ ←──── Cancel (kill subprocess)
                     └──────┬───────┘
                            │ Success
                            ▼
                     ┌──────────────┐
                     │ Ready (cached) │
                     └──────┬───────┘
                            │ Apply and restart service
                            ▼
                     ┌──────────────┐
                     │  TRT service running │
                     └──────┬───────┘
                            │ Driver/version change detected
                            ▼
                     ┌──────────────┐
                     │  Stale (yellow) │
                     └──────────────┘
```

### 5.7 Error Handling

| Scenario | Handling |
|---|---|
| GPU OOM during build | Subprocess exits non-zero; UI shows the specific error and a hint (close other VRAM-hungry processes) |
| User clicks Cancel | UI kills subprocess; sweeps `runtime_cache/trt_engines/` for size=0 engines and orphan files (engine without matching .profile) |
| manifest exists but engine files missing | Detected at startup; cache_status returns `failed`; clear directory and prompt user |
| TRT EP load error (missing DLL) | Service process catches it, logs `trt provider load failed`, falls back to CUDA EP and notifies UI to set state `failed` |
| User closes the main app during build | Subprocess becomes orphan; let it finish; next launch checks manifest timestamp; if incomplete, ask the user whether to continue |
| Post-build verify fails | Delete the entire cache + clear manifest; report and ask the user to retry |

### 5.8 PyInstaller Packaging

New DLLs must land in `_internal/`:
- `nvinfer.dll` / `nvinfer_10.dll` (TRT core)
- `nvonnxparser.dll`
- `nvinfer_plugin.dll` (if any plugins are used)
- `onnxruntime_providers_tensorrt.dll`
- `onnxruntime_providers_shared.dll`
- `cudnn*.dll` (already used; verify version compatibility with TRT)

Mandatory smoke test after packaging: install on a clean Windows machine, enable TRT in the UI, run the full build flow. Any missing DLL causes the TRT EP to silently fail to load and fall back to CUDA EP — invisible to the user.

The subprocess `ptserver-trt-warmup` is a standalone entry point and needs a PyInstaller entry spec in `build_exe.bat`.

### 5.9 Subprocess stdout Protocol

Use a line-based text protocol between UI and subprocess (simple, stable):

```
STAGE:1:start:Building single-eye engine
STAGE:1:elapsed:01:23
STAGE:1:done:88
STAGE:2:start:Building SBS dual-eye engine
STAGE:2:elapsed:03:01
STAGE:2:done:262
STAGE:3:start:Solidifying runtime cache
STAGE:3:done:8
DONE:total_seconds=452
```

Failure:
```
STAGE:1:start:Building single-eye engine
ERROR:GPU OOM
EXIT:1
```

Schema: `STAGE:<n>:<event>[:value]`. The UI consumes `STAGE:` prefixed lines for state; all other output is appended verbatim to `runtime_cache/trt_engines/build.log` for diagnostics.

### 5.10 On Driver Change Detection

No need to check on every 8K frame — the cost is unacceptable. **Detect only at service startup**:
- `main.py` calls `cache_status()` at startup; on `stale`, log `trt cache stale due to driver_version change: 560.94 -> 561.09, falling back to CUDA`.
- The UI process performs the same check whenever the Performance panel is shown, to decide the badge color.

The UI does not poll. The check happens when the user navigates to the panel; the cost is negligible.

## 6. Implementation Breakdown (Suggested Order)

1. **utils/trt_manifest.py + tests**
   Pure functions, unit-testable. `collect_fingerprint` is mocked in CI without a GPU.

2. **ui/services/trt_warmup_process.py subprocess**
   Standalone CLI, runs without the UI. First validate the flow with CUDA EP, then switch to TRT EP for real compilation. Manual smoke tests.

3. **ui/settings.py: add inference_backend field + server_env() wiring**
   Pure string handling, unit-testable.

4. **main.py / config.py startup detection**
   Read manifest, decide whether to inject the TRT provider. Clean logs.

5. **UI Performance panel new section**
   Static layout first; then wire manifest reads to display state; finally hook up QProcess for the subprocess + progress updates.

6. **PyInstaller spec update + clean-machine smoke**

7. **CHANGELOG + user documentation**
   Document first-build time, cache directory location, and behavior on driver changes.

## 7. Acceptance Criteria

- Fresh machine (no manifest), launch UI: TRT radio visible but disabled; cache button shows "Not cached".
- Click Build → progress dialog → after 5-15 min, the manifest is written and the state turns green.
- Switch to TRT and click "Apply and Restart Service": service restarts; logs show `providers=[TensorrtExecutionProvider, ...]`.
- Switch back to CUDA and click "Apply and Restart Service": service restarts; logs show `providers=[CUDAExecutionProvider, ...]`; TRT is not loaded; cache directory is preserved.
- Manually edit manifest.json's driver_version to simulate an upgrade → restart UI → state auto-turns yellow with a rebuild prompt.
- Close the UI main window during a build → the subprocess either finishes or is reaped; no half-finished artifacts remain.
- 8K SBS measurement: with TRT enabled, FPS is at least 50% higher than CUDA (on RTX 20 series, single SBS video, ALPHA_STRIDE=1, input=1024, downsample=0.5).

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Build time exceeds expectations, high cancel rate | Staged progress + early feedback (stage 1 visible within minutes) |
| Driver auto-upgrade (Windows Update) invalidates the cache without the user noticing | Active check at startup with a notification badge in the UI |
| Concurrent VRAM-heavy processes cause OOM | Subprocess error surfaces specifics + a retry button |
| Antivirus sweeps the cache directory | When manifest references missing engines, state becomes `failed`, UI prompts rebuild |
| PyInstaller misses a DLL; TRT silently does not load | Startup log checks that providers match settings; show a one-time toast on mismatch |
| Multi-user / multi-GPU systems | manifest pins GPU UUID; swapping GPU auto-invalidates; multi-GPU stays on device 0 (matches current server behavior) |

## 9. Optional Future Work (Out of This Iteration)

- Support multi-model caching (added when Phase 2 introduces a new model).
- Cache directory management (purge old engines, enforce size cap).
- "Hot switch" of the service that does not drop in-flight streams (requires server-side request migration to a new session).
- One-click cache export/import for same-machine reinstall (still cross-machine invalid).

---

Companion Chinese document: `summary_20260520_IMPL_PLAN_TENSORRT_CACHE_UI_CN.md`
