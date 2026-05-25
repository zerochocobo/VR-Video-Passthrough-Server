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
    with _lock:
        previous_phase = str(_state.get("phase") or "")
        _state["phase"] = phase
        _state["message"] = message
        _state["updated_at"] = now
        for key, value in fields.items():
            if key == "progress" and monotonic_progress and previous_phase == phase:
                try:
                    value = max(float(value), float(_state.get("progress") or 0.0))
                except (TypeError, ValueError):
                    pass
            _state[key] = value


def start_heartbeat(eta_sec: float, baseline_progress: float, ceiling_progress: float = 0.95) -> None:
    """Advance elapsed/progress while startup is inside a long blocking call."""
    global _heartbeat_thread, _heartbeat_stop
    stop_heartbeat()

    eta = max(0.1, float(eta_sec or 0.1))
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
                _state["elapsed_sec"] = round(elapsed, 3)
                _state["updated_at"] = now
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
