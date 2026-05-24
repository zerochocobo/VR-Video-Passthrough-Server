from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import QObject, Signal

from ui.log_sanitizer import clean_log_text
from ui.services.hidden_process import HiddenProcess
from ui.services.process_helpers import base_environment, server_command
from utils.subprocess_hidden import hidden_subprocess_kwargs

_TERMINATE_WAIT_MS = 3000
_TASKKILL_WAIT_MS = 5000
_KILL_WAIT_MS = 3000


class ServerProcess(QObject):
    output = Signal(str)
    state_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.process = HiddenProcess(self)
        self.process.stdout.connect(self._read_stdout)
        self.process.stderr.connect(self._read_stderr)
        self.process.error.connect(self._error_occurred)
        self.process.finished.connect(self._finished)

    def is_running(self) -> bool:
        return self.process.is_running()

    def start(self, env: dict[str, str]) -> None:
        if self.is_running():
            return
        exe, args = server_command()
        if self.process.start(exe, args, env=base_environment(env)):
            self.state_changed.emit(True)

    def stop(self) -> None:
        if not self.is_running():
            return
        pid = self.process.process_id()
        self.process.terminate()
        if self.process.wait_for_finished(_TERMINATE_WAIT_MS):
            return
        taskkill_ok = False
        if sys.platform.startswith("win") and pid > 0:
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                    **hidden_subprocess_kwargs(),
                )
                taskkill_ok = result.returncode == 0
            except Exception as exc:
                self.output.emit(f"Server process taskkill failed: {type(exc).__name__}: {exc}\n")
        if taskkill_ok and self.process.wait_for_finished(_TASKKILL_WAIT_MS):
            return
        self.process.kill()
        self.process.wait_for_finished(_KILL_WAIT_MS)

    def _read_stdout(self, text: str) -> None:
        data = clean_log_text(text)
        if data:
            self.output.emit(data)

    def _read_stderr(self, text: str) -> None:
        data = clean_log_text(text)
        if data:
            self.output.emit(data)

    def _error_occurred(self, error: str) -> None:
        self.output.emit(f"Server process error: {error}\n")
        if not self.is_running():
            self.state_changed.emit(False)

    def _finished(self, _exit_code: int = 0) -> None:
        self.state_changed.emit(False)
