from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ui.log_sanitizer import clean_log_text
from ui.services.process_helpers import base_environment, server_command


class ServerProcess(QObject):
    output = Signal(str)
    state_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._error_occurred)
        self.process.finished.connect(self._finished)

    def is_running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def start(self, env: dict[str, str]) -> None:
        if self.is_running():
            return
        exe, args = server_command()
        qenv = QProcessEnvironment.systemEnvironment()
        for key, value in base_environment(env).items():
            qenv.insert(key, value)
        self.process.setProcessEnvironment(qenv)
        self.process.start(exe, args)
        self.state_changed.emit(True)

    def stop(self) -> None:
        if not self.is_running():
            return
        pid = int(self.process.processId())
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            if sys.platform.startswith("win") and pid > 0:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                    )
                except Exception:
                    self.process.kill()
            else:
                self.process.kill()

    def _read_stdout(self) -> None:
        data = clean_log_text(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))
        if data:
            self.output.emit(data)

    def _read_stderr(self) -> None:
        data = clean_log_text(bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))
        if data:
            self.output.emit(data)

    def _error_occurred(self, error) -> None:
        self.output.emit(f"Server process error: {error}\n")
        if not self.is_running():
            self.state_changed.emit(False)

    def _finished(self) -> None:
        self.state_changed.emit(False)
