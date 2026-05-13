# Cold-Start UX and Diagnostic Report — Work Summary

- Date: 2026-05-13
- Scope: UI startup overlay, `/status` poller, diagnostic report, `main.py` startup-phase flow
- Target users: non-technical end users (VR headset scenario); on failure they copy a single hardware report to share with support

---

## 1. Background

- After upgrading to RTX 50 series GPUs (sm_120), first-time startup needs onnxruntime-gpu / CuPy to JIT-compile on the fly, which can take 1–3 minutes. The previous UI showed a static home page during that window with no feedback at all — non-technical users easily misread the silence as a crash and closed the program.
- Testing also covered GTX TITAN X (sm_5.2 — already dropped by modern CuPy 14.x and ORT-GPU 1.19.2). For unsupported hardware like this we need a clean "startup failed + one-click report" exit path.
- User requirements: **(1)** friendly guidance for non-technical users; **(2)** one-click copy of a hardware diagnostic report.

## 2. Files Added / Modified

| File | Status | Purpose |
| --- | --- | --- |
| `utils/gpu_runtime_cache.py` | modified | New `ColdStartReport` dataclass and `predict_warmup_state()` pure function; sm_120 known-slow detection |
| `utils/startup_status.py` | rewritten | Extended structured fields (step / progress / eta / cold / is_known_slow / gpu_* / reason / detail); new `reset_startup_progress()` |
| `main.py` | modified | Publishes prediction up-front; on warmup failure publishes `phase=failed`, sleeps 0.8 s for the poller, then stops the status server |
| `ui/diagnostics.py` | new | `build_diagnostic_report()`: composes GPU / ORT / CuPy / **FFmpeg+FFprobe+NVENC** / warmup marker / `/status` / **last 200 lines of server.log** |
| `ui/services/startup_status_poller.py` | new | `StartupStatusPoller` (500 ms polling of `127.0.0.1:8299/status`); emits `finished` on terminal phases `warmed/listening/failed/shutting_down` |
| `ui/widgets/startup_overlay.py` | new | Non-modal QDialog overlay: title / message / ETA / progress bar / yellow advisory / details panel / Copy Report / Cancel; animated ellipsis; switches the bar to indeterminate (busy) mode during the blocking step |
| `ui/main_window.py` | modified | Opens overlay and starts the poller on Start; multiple termination paths; merges (rather than replaces) `last_status` on failure |
| `ui/translations/{zh_CN,en_US,ja_JP}.json` | modified | 19 new `startup.*` keys |
| `tests/test_predict_warmup_state.py` | new | 9 cases: sm_120/old-ORT known-slow detection, marker missing/present/inspect-failed, structured-kwargs in `set_startup_phase`, `reset_startup_progress()` |

## 3. Key Design Decisions

### 3.1 Predict-First
Before any heavy CUDA work begins the server calls `predict_warmup_state()` and publishes (cold / reason / eta / gpu / is_known_slow) on port 8299. The UI immediately shows "First-time startup needs 1–3 minutes" instead of an opaque wait.

ETA buckets:
- `_ETA_CACHE_HIT_SEC = 4.0`
- `_ETA_KEY_CHANGED_SEC = 30.0`
- `_ETA_FIRST_RUN_NORMAL_SEC = 45.0`
- `_ETA_FIRST_RUN_KNOWN_SLOW_SEC = 150.0`

Known-slow detection: `compute_capability` major ≥ 12 **and** `onnxruntime` version < (1, 22).

### 3.2 Friendly Overlay + Indeterminate Fallback
Warmup is a single blocking call internally — no sub-step events are possible without invasive refactoring. With only one `progress=0.1` update the bar visually appears frozen.

`StartupOverlay.apply_status()` switches the bar to `setRange(0, 0)` (indeterminate / marquee) when `phase=warming && elapsed≤0.1 && progress≤0.11`, so the animation keeps running. Once a later `/status` carries real progress, it flips back to the determinate mode.

### 3.3 One-Click Diagnostic Report (single-button strategy)
Non-technical users shouldn't have to choose between "hardware report" vs "log". Everything is merged into one plain-text blob that copies to the clipboard:

- Timestamp / app version / host / OS / Python / frozen / cwd
- `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap`
- onnxruntime / providers / cupy / cupy devices / numpy
- **ffmpeg / ffprobe resolved path + first version line + NVENC encoder presence** (most common Windows support root cause)
- Warmup marker (including ORT CUDA DLL hash)
- The last `/status` snapshot, all fields
- **Last 200 lines of `server.log`** (the single most diagnostic-rich piece)

Implementation notes:
- Every probe uses lazy imports or subprocess with a short (5 s) timeout; any failure is swallowed and replaced with a safe placeholder — the report never raises.
- Log tail reads with `errors="replace"` so a partially-corrupt UTF-8 sequence cannot break the report.

### 3.4 Multiple Termination Paths (robust exit)

| Path | Trigger | Behavior |
| --- | --- | --- |
| `/status` poll sees `warmed` / `listening` | Normal completion | Merge `last_status`, flash 100 %, close overlay |
| `/status` poll sees `failed` | Warmup raised | Merge `last_status`, preserve `detail`/`message`, leave overlay on failure view so the user can copy the report |
| `Uvicorn running on` appears in stdout | 8299 unreachable (port conflict / firewall / IPv4 disabled) | `_scan_server_output_for_ready` synthesizes `listening`, force-closes overlay |
| `QProcess.finished` with non-ready last phase | Server died and 8299 went away inside one poll interval | `_server_state_changed` synthesizes `failed`, preserves earlier `gpu_name`/`step`/`reason` |
| User clicks Cancel | Manual abort | `_cancel_startup` stops server + poller, closes overlay |

Server-side belt-and-braces: after publishing `phase=failed`, `main.py` does `time.sleep(0.8)` before `stop_startup_status_server()` so the 500 ms poller has a deterministic chance to read the precise failure message; the UI-side synthesized failed state is the fallback.

### 3.5 Don't Wipe Fields on Failure
Previously `_on_startup_finished(failed)` called `apply_status({"phase":"failed","message":...})`, which **replaced** the whole dict — the diagnostic report ended up with empty `gpu_name/step/cold/reason/detail`. Fixed by merging:

```python
merged = dict(self.startup_overlay.last_status() or {})
merged["phase"] = "failed"
if not merged.get("message"):
    merged["message"] = self.i18n.t("startup.failed_generic")
self.startup_overlay.apply_status(merged)
```

The `warmed`/`listening` path was changed to merge too, so GPU info from the success path also survives into the report.

## 4. User-Test Verification & Fixes

### 4.1 GTX TITAN X (sm_5.2) first-run failure
- Log showed `nvrtc: error: invalid value for --gpu-architecture` and `CUDA Provider not available`.
- UI sat on `warming/ort_session_and_runs/progress=0.1` — the server published `progress=0.1` once, blocked for several seconds, then tore down port 8299 within ~12 ms of publishing `failed`.
- After the fix: server-side 0.8 s grace period plus UI-side synthesized `failed` state reliably flip the overlay to the failure view; the copy-report shows the nvrtc / CUDA-Provider lines directly.

### 4.2 Warm-cache machine: overlay never closes
- Symptom: UI stuck on "Connecting to server process", details panel empty, yet `server.log` says it is already listening.
- Root cause: 8299 unreachable (most often a port conflict or a host-local policy), so the UI never gets any status update.
- Fix: watch `ServerProcess.output` for `Uvicorn running on` / `Application startup complete` as a readiness signal independent of 8299, and force-close the overlay.

### 4.3 Failed-state report missing structured fields
- Root cause: `_on_startup_finished(failed)` was replacing the dict.
- Fix: merge — all GPU/step/reason fields are preserved.

## 5. Test Results

- New `tests/test_predict_warmup_state.py`: 9 cases, all passing.
- Full regression (15 test modules): 61/61 passing.
- Command: `PYTHONPATH=. .venv/Scripts/python.exe -m unittest tests.<module> ...`

## 6. Translation Keys Added (19)

`startup.window_title / title_starting / title_first_run / title_first_run_slow / title_verifying / title_ready / title_failed / connecting / complete / gpu_label / eta_template / hint_known_slow / hint_failed / failed_generic / show_details / hide_details / copy_report / report_copied / cancel`

zh_CN / en_US / ja_JP kept in sync; `tests/test_i18n.py` enforces key-set parity.

## 7. Known Follow-Ups

- `warmup_gpu_runtime_cache()` is still a single blocking call. Reporting real sub-step progress (OrtSession load, first batch=1 run, batch=2 run, second-pass verify) would require restructuring that function. For now indeterminate-bar mode covers the visual gap.
- Known-slow detection only covers sm_120 × ORT < 1.22 today. Future architectures (sm_130+) can be added by extending the `_KNOWN_SLOW_*` constants.
- Legacy hardware (TITAN X / sm_5.2) is unsupported by current CuPy 14.x — the UI can't paper over this. Current policy is to surface the failure clearly and provide a one-click report rather than attempt software compatibility.

## 8. Quick File Reference

- `D:\p\PTServer\utils\gpu_runtime_cache.py` — `predict_warmup_state`, `ColdStartReport`, ETA buckets, sm_120 detection
- `D:\p\PTServer\utils\startup_status.py` — structured `_state`, `set_startup_phase(**fields)`, `reset_startup_progress`
- `D:\p\PTServer\ui\diagnostics.py` — `build_diagnostic_report(app_version, language, last_status, marker_path, log_path, log_tail_lines)`
- `D:\p\PTServer\ui\services\startup_status_poller.py` — 500 ms poll, terminal-phase set
- `D:\p\PTServer\ui\widgets\startup_overlay.py` — overlay QDialog, indeterminate-mode switching
- `D:\p\PTServer\ui\main_window.py` — `toggle_server` / `_on_startup_status` / `_on_startup_finished` / `_scan_server_output_for_ready` / `_server_state_changed` / `_copy_startup_report`
- `D:\p\PTServer\main.py` — warmup-phase events + 0.8 s failure grace
- `D:\p\PTServer\tests\test_predict_warmup_state.py` — 9 cases
