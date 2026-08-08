"""Small local health endpoint available while the main server is warming up.

run_server.bat can expose this on localhost so test automation can tell whether
the process is alive before uvicorn starts listening on the DLNA HTTP port.

The structured fields (step, step_index, step_total, progress, eta_sec,
cold, is_known_slow, gpu_*) are read by the UI startup overlay to show a
human-friendly "first GPU initialization" experience instead of a blank wait.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from utils.logger import get

_lock = threading.Lock()
_started_at = time.time()
_state: dict[str, Any] = {
    "phase": "starting",
    "message": "process starting",
    "started_at": _started_at,
    "updated_at": _started_at,
    # Structured fields (optional; populated as startup progresses).
    "step": "",
    "step_index": 0,
    "step_total": 0,
    "progress": 0.0,           # 0.0..1.0
    "eta_sec": 0.0,            # estimated remaining seconds for current phase
    "elapsed_sec": 0.0,        # elapsed seconds inside current phase
    "cold": False,             # True when warmup is a cache miss
    "is_known_slow": False,    # True for sm_120 without bundled cubin etc.
    "gpu_name": "",
    "compute_capability": "",
    "driver_version": "",
    "onnxruntime_version": "",
    "reason": "",              # cache_hit | marker_missing | key_changed | ...
    "detail": "",
}
_server: ThreadingHTTPServer | None = None
_thread: threading.Thread | None = None
_heartbeat_thread: threading.Thread | None = None
_heartbeat_stop: threading.Event | None = None
_plan_steps: list[dict[str, Any]] = []
_plan_started_at = 0.0
_plan_current_step = ""
_plan_step_started_at = 0.0
_plan_history_path: Path | None = None
_plan_history: dict[str, float] = {}
_plan_seen: set[str] = set()


def configure_startup_plan(
    steps: list[tuple[str, float]],
    *,
    history_path: Path | None = None,
    estimate_profiles: dict[str, str] | None = None,
) -> None:
    """Install the ordered, weighted plan used by the startup overlay.

    Estimates are seconds, optionally refined with an EWMA from previous
    successful starts. Callers may keep publishing their existing local
    progress; this module maps the active step into one monotonic global bar.
    """
    global _plan_steps, _plan_started_at, _plan_current_step, _plan_step_started_at
    global _plan_history_path, _plan_history, _plan_seen
    stop_heartbeat()
    history: dict[str, float] = {}
    if history_path is not None:
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                history = {
                    str(key): max(0.1, float(value))
                    for key, value in loaded.items()
                    if isinstance(value, (int, float))
                }
        except (OSError, ValueError, TypeError):
            history = {}
    profiles = estimate_profiles or {}
    normalized = []
    for key, default_estimate in steps:
        step_key = str(key)
        profile = str(profiles.get(step_key) or "").strip()
        history_key = f"{step_key}@{profile}" if profile else step_key
        if profile == "pending":
            # Pending is a conservative pre-validation budget, not a runtime
            # provider whose duration should be learned. Ignore and remove
            # values written by older builds with the profile-switch bug.
            history.pop(history_key, None)
            estimate = max(0.1, float(default_estimate))
        else:
            estimate = history.get(history_key, max(0.1, float(default_estimate)))
        normalized.append({"key": step_key, "history_key": history_key, "estimate": estimate})
    now = time.time()
    with _lock:
        _plan_steps = normalized
        _plan_started_at = now
        _plan_current_step = ""
        _plan_step_started_at = 0.0
        _plan_history_path = history_path
        _plan_history = history
        _plan_seen = set()
        _state["plan_active"] = bool(normalized)
        _state["plan_steps"] = [item["key"] for item in normalized]
        _state["plan_estimate_sec"] = round(sum(float(item["estimate"]) for item in normalized), 1)
        _state["step_total"] = len(normalized)
        _state["step_index"] = 0
        _state["progress"] = 0.0
        _state["eta_sec"] = _state["plan_estimate_sec"]
        _state["elapsed_sec"] = 0.0
        _state["total_elapsed_sec"] = 0.0
        _state["step_elapsed_sec"] = 0.0


def reconfigure_startup_plan(
    steps: list[tuple[str, float]],
    *,
    estimate_profiles: dict[str, str] | None = None,
) -> None:
    """Replace future plan weights without resetting elapsed/progress state."""
    global _plan_steps
    now = time.time()
    with _lock:
        previous_steps = list(_plan_steps)
        profiles = estimate_profiles or {}
        normalized = []
        for key, default_estimate in steps:
            step_key = str(key)
            profile = str(profiles.get(step_key) or "").strip()
            history_key = f"{step_key}@{profile}" if profile else step_key
            estimate = _plan_history.get(history_key, max(0.1, float(default_estimate)))
            normalized.append({"key": step_key, "history_key": history_key, "estimate": estimate})
        retained_keys = {str(item["key"]) for item in normalized}
        for item in previous_steps:
            if str(item["key"]) not in _plan_seen and str(item["key"]) not in retained_keys:
                _mark_plan_step_skipped_locked(str(item["history_key"]))
        _plan_steps = normalized
        _state["plan_active"] = bool(normalized)
        _state["plan_steps"] = [item["key"] for item in normalized]
        _state["plan_estimate_sec"] = round(sum(float(item["estimate"]) for item in normalized), 1)
        _state["step_total"] = len(normalized)
        if _plan_current_step:
            current = next((item for item in normalized if item["key"] == _plan_current_step), None)
            if current is not None:
                elapsed = max(0.0, now - _plan_step_started_at)
                local = min(0.95, elapsed / max(0.1, float(current["estimate"])))
                _apply_plan_locked(_plan_current_step, now, local)


def _plan_index(step: str) -> int:
    for index, item in enumerate(_plan_steps):
        if item["key"] == step:
            return index
    return -1


def _complete_plan_step_locked(now: float) -> None:
    global _plan_current_step, _plan_step_started_at
    if not _plan_current_step or _plan_step_started_at <= 0:
        return
    elapsed = max(0.01, now - _plan_step_started_at)
    item = next((candidate for candidate in _plan_steps if candidate["key"] == _plan_current_step), None)
    history_key = str(item["history_key"]) if item is not None else _plan_current_step
    _update_plan_history_locked(history_key, elapsed)
    _plan_current_step = ""
    _plan_step_started_at = 0.0


def _update_plan_history_locked(step: str, elapsed: float) -> None:
    previous = _plan_history.get(step)
    _plan_history[step] = elapsed if previous is None else previous * 0.7 + elapsed * 0.3


def _mark_plan_step_skipped_locked(step: str) -> None:
    # A skipped step is a deterministic zero-duration observation, not a noisy
    # timing sample. Collapse it immediately so the next start is not burdened
    # by a stale duration from an earlier configuration/cache state.
    _plan_history[step] = 0.01


def _save_plan_history_locked() -> None:
    for item in _plan_steps:
        key = str(item["key"])
        if key not in _plan_seen:
            _mark_plan_step_skipped_locked(str(item["history_key"]))
    if _plan_history_path is None or not _plan_history:
        return
    try:
        _plan_history_path.parent.mkdir(parents=True, exist_ok=True)
        _plan_history_path.write_text(
            json.dumps(_plan_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _apply_plan_locked(step: str, now: float, local_progress: float = 0.0) -> None:
    global _plan_current_step, _plan_step_started_at
    index = _plan_index(step)
    if index < 0:
        if _plan_current_step and step != _plan_current_step:
            _complete_plan_step_locked(now)
        return
    if step != _plan_current_step:
        _complete_plan_step_locked(now)
        _plan_current_step = step
        _plan_step_started_at = now
    _plan_seen.add(step)
    local = max(0.0, min(0.99, float(local_progress)))
    total_estimate = sum(float(item["estimate"]) for item in _plan_steps) or 1.0
    completed_estimate = sum(float(item["estimate"]) for item in _plan_steps[:index])
    current_estimate = float(_plan_steps[index]["estimate"])
    step_elapsed = max(0.0, now - _plan_step_started_at)
    progress = (completed_estimate + current_estimate * local) / total_estimate
    remaining = max(0.0, current_estimate * (1.0 - local)) + sum(
        float(item["estimate"]) for item in _plan_steps[index + 1 :]
    )
    _state["step_index"] = index + 1
    _state["step_total"] = len(_plan_steps)
    _state["progress"] = max(float(_state.get("progress") or 0.0), min(0.99, progress))
    _state["eta_sec"] = round(remaining, 1)
    _state["elapsed_sec"] = round(max(0.0, now - _plan_started_at), 3)
    _state["total_elapsed_sec"] = _state["elapsed_sec"]
    _state["step_elapsed_sec"] = round(step_elapsed, 3)
    _state["step_estimate_sec"] = round(current_estimate, 1)


def set_startup_phase(phase: str, message: str = "", **fields: Any) -> None:
    """Update the structured startup state.

    Backwards compatible: existing callers pass (phase, message). New callers
    can additionally pass any subset of step, step_index, step_total, progress,
    eta_sec, elapsed_sec, cold, is_known_slow, gpu_name, compute_capability,
    driver_version, onnxruntime_version, reason, detail.

    Unknown keys are stored verbatim so the endpoint stays forward-compatible.
    """
    now = time.time()
    monotonic_progress = bool(fields.pop("monotonic_progress", False))
    local_progress = fields.pop("step_progress", None)
    minimum_step_estimate = fields.pop("minimum_step_estimate_sec", None)
    with _lock:
        previous_phase = str(_state.get("phase") or "")
        previous_step = str(_state.get("step") or "")
        _state["phase"] = phase
        _state["message"] = message
        _state["updated_at"] = now
        step = str(fields.get("step") or "")
        if step != previous_step:
            if "trt_building" not in fields:
                _state["trt_building"] = False
            if "trt_build_model" not in fields:
                _state["trt_build_model"] = ""
        if _plan_steps and phase != "failed":
            if minimum_step_estimate is not None:
                item = next((candidate for candidate in _plan_steps if candidate["key"] == step), None)
                if item is not None:
                    item["estimate"] = max(float(item["estimate"]), float(minimum_step_estimate))
                    _state["plan_estimate_sec"] = round(
                        sum(float(candidate["estimate"]) for candidate in _plan_steps),
                        1,
                    )
            if local_progress is None:
                done = fields.get("run_done")
                total = fields.get("run_total")
                try:
                    local_progress = float(done) / float(total) if float(total) > 0 else 0.0
                except (TypeError, ValueError, ZeroDivisionError):
                    local_progress = 0.0
            _apply_plan_locked(step, now, float(local_progress))
        for key, value in fields.items():
            if _plan_steps and phase != "failed" and key in {"step_index", "step_total", "progress", "eta_sec", "elapsed_sec"}:
                continue
            if key == "progress" and monotonic_progress and previous_phase == phase:
                try:
                    value = max(float(value), float(_state.get("progress") or 0.0))
                except (TypeError, ValueError):
                    pass
            _state[key] = value
        if phase == "listening":
            _complete_plan_step_locked(now)
            _state["progress"] = 1.0
            _state["eta_sec"] = 0.0
            _state["elapsed_sec"] = round(max(0.0, now - _plan_started_at), 3) if _plan_started_at else 0.0
            _state["total_elapsed_sec"] = _state["elapsed_sec"]
            _save_plan_history_locked()


def start_heartbeat(eta_sec: float, baseline_progress: float, ceiling_progress: float = 0.95) -> None:
    """Advance elapsed/progress while startup is inside a long blocking call."""
    global _heartbeat_thread, _heartbeat_stop
    stop_heartbeat()

    eta = max(0.1, float(eta_sec or 0.1))
    with _lock:
        if _plan_steps and _plan_current_step:
            current = next((item for item in _plan_steps if item["key"] == _plan_current_step), None)
            if current is not None:
                eta = max(0.1, float(current["estimate"]))
    baseline = max(0.0, min(1.0, float(baseline_progress)))
    ceiling = max(baseline, min(1.0, float(ceiling_progress)))
    started_at = time.time()
    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.wait(0.5):
            now = time.time()
            elapsed = max(0.0, now - started_at)
            progress = min(ceiling, baseline + (ceiling - baseline) * min(1.0, elapsed / eta))
            with _lock:
                _state["updated_at"] = now
                if _plan_steps and _plan_current_step:
                    _apply_plan_locked(_plan_current_step, now, min(0.95, elapsed / eta))
                else:
                    _state["elapsed_sec"] = round(elapsed, 3)
                    if progress > float(_state.get("progress") or 0.0):
                        _state["progress"] = progress

    _heartbeat_stop = stop_event
    _heartbeat_thread = threading.Thread(target=_run, name="startup-status-heartbeat", daemon=True)
    _heartbeat_thread.start()


def stop_heartbeat() -> None:
    """Stop the startup heartbeat thread if one is active."""
    global _heartbeat_thread, _heartbeat_stop
    stop_event = _heartbeat_stop
    thread = _heartbeat_thread
    _heartbeat_stop = None
    _heartbeat_thread = None
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)


def reset_startup_progress() -> None:
    """Clear structured per-step fields between phases.

    Keeps phase/message/timestamps in place; only zeros the moving values.
    """
    with _lock:
        _state["step"] = ""
        _state["step_index"] = 0
        _state["step_total"] = 0
        _state["progress"] = 0.0
        _state["eta_sec"] = 0.0
        _state["elapsed_sec"] = 0.0


def get_startup_state() -> dict[str, Any]:
    now = time.time()
    with _lock:
        state = dict(_state)
    state["uptime_sec"] = round(now - float(state["started_at"]), 3)
    state["age_sec"] = round(now - float(state["updated_at"]), 3)
    return state


class _StatusHandler(BaseHTTPRequestHandler):
    server_version = "PTStartupStatus/1.0"

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as e:
            get("startup_status").debug("startup status client disconnected: %s", e)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] not in {"/", "/status", "/health"}:
            try:
                self.send_error(404)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as e:
                get("startup_status").debug("startup status client disconnected during 404: %s", e)
            return
        body = json.dumps(get_startup_state(), sort_keys=True).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as e:
            get("startup_status").debug("startup status client disconnected during response: %s", e)

    def log_message(self, fmt: str, *args: Any) -> None:
        get("startup_status").debug("127.0.0.1 status: " + fmt, *args)


def start_startup_status_server(port: int) -> None:
    global _server, _thread
    if port <= 0:
        return
    if _server is not None:
        return
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _StatusHandler)
    except OSError as e:
        get("startup_status").warning("startup status port unavailable: 127.0.0.1:%d (%s)", port, e)
        return
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="startup-status", daemon=True)
    _server = server
    _thread = thread
    thread.start()
    get("startup_status").info("startup status listening on http://127.0.0.1:%d/status", port)


def stop_startup_status_server() -> None:
    global _server, _thread
    stop_heartbeat()
    server = _server
    thread = _thread
    _server = None
    _thread = None
    if server is None:
        return
    def _shutdown() -> None:
        server.shutdown()

    shutdown_thread = threading.Thread(target=_shutdown, name="startup-status-shutdown", daemon=True)
    shutdown_thread.start()
    shutdown_thread.join(timeout=1.0)
    if shutdown_thread.is_alive():
        get("startup_status").warning("startup status shutdown timed out; closing socket")
    server.server_close()
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
