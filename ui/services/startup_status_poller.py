"""Poll the server process's local /status endpoint.

The server publishes structured startup state (phase, step, progress, eta_sec,
cold, is_known_slow, gpu_*, ...) on http://127.0.0.1:STARTUP_STATUS_PORT/status
while it is initializing the GPU runtime. This service polls the endpoint with
a short interval and emits a Qt signal so the UI can display friendly progress
without blocking the UI thread.

Polling stops automatically once phase is "listening" (server has bound the
DLNA port and is truly available) or "failed" (warmup failed). Intermediate
terminal-ish phases such as "warmed" are forwarded through ``updated`` so the
overlay can reflect them, but they do NOT close the overlay — the server still
runs firewall / SSDP / uvicorn between ``warmed`` and ``listening`` and the
window must stay up so the user does not start clicking before the DLNA port
is actually ready.

Implementation note: the HTTP request runs in a daemon worker thread. Calling
``urlopen()`` directly inside the ``QTimer.timeout`` callback would block the
Qt event loop for up to ``timeout_sec`` seconds whenever the status endpoint
is slow or unreachable, defeating the whole point of a "non-blocking startup
overlay". The worker thread emits a private Qt signal which Qt auto-marshals
back to the main thread via a queued connection.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from PySide6.QtCore import QObject, QTimer, Signal

from ui.services.startup_diagnostics import log_startup_event


DEFAULT_PORT = 8299
# Phases that cause the overlay to close. ``warmed`` is intentionally NOT in
# this set: GPU warmup is finished at that point but the server still has to
# install firewall rules, start SSDP, and bind the DLNA HTTP port before the
# product is actually usable. The overlay must wait for ``listening``.
TERMINAL_PHASES = frozenset({"listening", "failed", "shutting_down"})
_DIRECT_OPENER = build_opener(ProxyHandler({}))


class StartupStatusPoller(QObject):
    """Poll /status periodically and emit updates."""

    updated = Signal(dict)            # Latest status dict from the server.
    finished = Signal(str)            # Terminal phase that stopped polling.
    error = Signal(str)               # Transport error string (non-fatal).

    # Private signals used to marshal worker-thread results back to the Qt
    # main thread. Queued by Qt automatically because emitter and receiver
    # live in different threads.
    _gotResponse = Signal(int, bytes)
    _gotError = Signal(int, str)

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        interval_ms: int = 500,
        timeout_sec: float = 1.0,
        max_duration_sec: float = 300.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.max_duration_sec = float(max_duration_sec)
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._tick)
        self._running = False
        self._last_phase = ""
        self._generation = 0
        self._started_at = 0.0
        # True while a worker thread is in flight. Skips re-issuing a request
        # if the previous one has not returned yet, so a hanging endpoint
        # cannot pile up dozens of pending threads.
        self._inflight = False
        self._gotResponse.connect(self._handle_response)
        self._gotError.connect(self._handle_error)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_phase = ""
        self._inflight = False
        self._generation += 1
        self._started_at = time.monotonic()
        log_startup_event(
            "poller_start",
            port=self.port,
            interval_ms=self._timer.interval(),
            timeout_sec=self.timeout_sec,
            max_duration_sec=self.max_duration_sec,
            generation=self._generation,
        )
        # First tick immediately so the UI does not wait one interval to react.
        self._tick()
        self._timer.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._timer.stop()
        self._generation += 1
        log_startup_event("poller_stop", generation=self._generation)

    def is_running(self) -> bool:
        return self._running

    def _tick(self) -> None:
        if not self._running:
            return
        if self.max_duration_sec > 0 and self._started_at > 0:
            elapsed = time.monotonic() - self._started_at
            if elapsed >= self.max_duration_sec:
                self._finish_timeout(elapsed)
                return
        if self._inflight:
            # Previous request has not returned yet. Skip this slot to avoid
            # piling up worker threads against a hanging endpoint.
            log_startup_event("poller_tick_skipped_inflight", generation=self._generation)
            return
        self._inflight = True
        url = f"http://127.0.0.1:{self.port}/status"
        generation = self._generation
        worker = threading.Thread(
            target=self._fetch_worker,
            name="startup-status-poll",
            args=(generation, url, self.timeout_sec),
            daemon=True,
        )
        worker.start()

    def _fetch_worker(self, generation: int, url: str, timeout: float) -> None:
        """Run in a daemon thread. Must never touch Qt widgets directly."""
        try:
            request = Request(url, headers={"Cache-Control": "no-cache"})
            with _DIRECT_OPENER.open(request, timeout=timeout) as resp:
                raw = resp.read()
        except URLError as e:
            log_startup_event("poller_fetch_error", generation=generation, url=url, error=f"unreachable: {e.reason}")
            self._gotError.emit(generation, f"unreachable: {e.reason}")
            return
        except Exception as e:  # pragma: no cover - defensive
            log_startup_event("poller_fetch_error", generation=generation, url=url, error=f"poll failed: {e}")
            self._gotError.emit(generation, f"poll failed: {e}")
            return
        log_startup_event("poller_fetch_ok", generation=generation, url=url, bytes=len(raw))
        self._gotResponse.emit(generation, raw)

    def _handle_error(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._inflight = False
        if not self._running:
            return
        log_startup_event("poller_error", generation=generation, message=message)
        self.error.emit(message)

    def _handle_response(self, generation: int, raw: bytes) -> None:
        if generation != self._generation:
            return
        self._inflight = False
        if not self._running:
            return
        try:
            data: dict[str, Any] = json.loads(raw.decode("utf-8", "replace"))
        except Exception as e:
            log_startup_event("poller_decode_error", generation=generation, error=str(e), bytes=len(raw))
            self.error.emit(f"decode failed: {e}")
            return

        log_startup_event(
            "poller_update",
            generation=generation,
            phase=data.get("phase"),
            step=data.get("step"),
            progress=data.get("progress"),
            elapsed_sec=data.get("elapsed_sec"),
            provider_kind=data.get("provider_kind"),
        )
        self.updated.emit(data)
        phase = str(data.get("phase") or "")
        if phase and phase in TERMINAL_PHASES and phase != self._last_phase:
            self._last_phase = phase
            self.stop()
            log_startup_event("poller_finished", generation=generation, phase=phase)
            self.finished.emit(phase)

    def _finish_timeout(self, elapsed: float) -> None:
        if not self._running:
            return
        data = {
            "phase": "failed",
            "message": "startup status polling timed out",
            "detail": f"no terminal startup status after {elapsed:.1f}s",
            "reason": "startup_status_timeout",
            "elapsed_sec": elapsed,
        }
        log_startup_event("poller_timeout", elapsed_sec=elapsed)
        self.updated.emit(data)
        self.stop()
        self.finished.emit("failed")
