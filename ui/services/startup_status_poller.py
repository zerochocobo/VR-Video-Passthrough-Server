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
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from PySide6.QtCore import QObject, QTimer, Signal


DEFAULT_PORT = 8299
# Phases that cause the overlay to close. ``warmed`` is intentionally NOT in
# this set: GPU warmup is finished at that point but the server still has to
# install firewall rules, start SSDP, and bind the DLNA HTTP port before the
# product is actually usable. The overlay must wait for ``listening``.
TERMINAL_PHASES = frozenset({"listening", "failed", "shutting_down"})


class StartupStatusPoller(QObject):
    """Poll /status periodically and emit updates."""

    updated = Signal(dict)            # Latest status dict from the server.
    finished = Signal(str)            # Terminal phase that stopped polling.
    error = Signal(str)               # Transport error string (non-fatal).

    # Private signals used to marshal worker-thread results back to the Qt
    # main thread. Queued by Qt automatically because emitter and receiver
    # live in different threads.
    _gotResponse = Signal(bytes)
    _gotError = Signal(str)

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        interval_ms: int = 500,
        timeout_sec: float = 1.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._tick)
        self._running = False
        self._last_phase = ""
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
        # First tick immediately so the UI does not wait one interval to react.
        self._tick()
        self._timer.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._timer.stop()

    def is_running(self) -> bool:
        return self._running

    def _tick(self) -> None:
        if not self._running:
            return
        if self._inflight:
            # Previous request has not returned yet. Skip this slot to avoid
            # piling up worker threads against a hanging endpoint.
            return
        self._inflight = True
        url = f"http://127.0.0.1:{self.port}/status"
        worker = threading.Thread(
            target=self._fetch_worker,
            name="startup-status-poll",
            args=(url, self.timeout_sec),
            daemon=True,
        )
        worker.start()

    def _fetch_worker(self, url: str, timeout: float) -> None:
        """Run in a daemon thread. Must never touch Qt widgets directly."""
        try:
            with urlopen(url, timeout=timeout) as resp:
                raw = resp.read()
        except URLError as e:
            self._gotError.emit(f"unreachable: {e.reason}")
            return
        except Exception as e:  # pragma: no cover - defensive
            self._gotError.emit(f"poll failed: {e}")
            return
        self._gotResponse.emit(raw)

    def _handle_error(self, message: str) -> None:
        self._inflight = False
        if not self._running:
            return
        self.error.emit(message)

    def _handle_response(self, raw: bytes) -> None:
        self._inflight = False
        if not self._running:
            return
        try:
            data: dict[str, Any] = json.loads(raw.decode("utf-8", "replace"))
        except Exception as e:
            self.error.emit(f"decode failed: {e}")
            return

        self.updated.emit(data)
        phase = str(data.get("phase") or "")
        if phase and phase in TERMINAL_PHASES and phase != self._last_phase:
            self._last_phase = phase
            self.stop()
            self.finished.emit(phase)
