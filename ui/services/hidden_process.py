from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from utils.subprocess_hidden import hidden_subprocess_kwargs

_INT32_MAX = 2**31 - 1
_UINT32_MOD = 2**32


def _qt_exit_code(code: int) -> int:
    """Normalize Windows DWORD process exit codes for Qt's signed int signal."""
    value = int(code)
    if value > _INT32_MAX:
        value -= _UINT32_MOD
    return value


class HiddenProcess(QObject):
    """Small QObject wrapper around subprocess.Popen with no console window."""

    stdout = Signal(str)
    stderr = Signal(str)
    error = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._generation = 0
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            proc = self._proc
            return proc is not None and proc.poll() is None

    def process_id(self) -> int:
        with self._lock:
            proc = self._proc
        return int(proc.pid) if proc is not None and proc.pid else 0

    def start(
        self,
        program: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> bool:
        cmd = [str(program), *(str(arg) for arg in (args or []))]
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False
            self._generation += 1
            generation = self._generation
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd) if cwd is not None else None,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    **hidden_subprocess_kwargs(),
                )
            except Exception as exc:
                self._proc = None
                error_text = f"{type(exc).__name__}: {exc}"
            else:
                error_text = ""
                self._proc = proc
        if error_text:
            self.error.emit(error_text)
            self.finished.emit(-1)
            return False
        assert proc is not None
        threading.Thread(target=self._reader_loop, args=(proc, "stdout"), name="hidden-process-stdout", daemon=True).start()
        threading.Thread(target=self._reader_loop, args=(proc, "stderr"), name="hidden-process-stderr", daemon=True).start()
        threading.Thread(target=self._wait_loop, args=(proc, generation), name="hidden-process-wait", daemon=True).start()
        return True

    def terminate(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def kill(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def wait_for_finished(self, timeout_ms: int) -> bool:
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        while time.monotonic() <= deadline:
            if not self.is_running():
                return True
            time.sleep(0.02)
        return not self.is_running()

    def _reader_loop(self, proc: subprocess.Popen, stream_name: str) -> None:
        stream = proc.stdout if stream_name == "stdout" else proc.stderr
        signal = self.stdout if stream_name == "stdout" else self.stderr
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                signal.emit(bytes(chunk).decode("utf-8", "replace"))
        except Exception:
            pass

    def _wait_loop(self, proc: subprocess.Popen, generation: int) -> None:
        try:
            rc = _qt_exit_code(int(proc.wait()))
        except Exception:
            rc = -1
        should_emit = False
        with self._lock:
            if self._proc is proc and self._generation == generation:
                self._proc = None
                should_emit = True
        if should_emit:
            self.finished.emit(rc)
