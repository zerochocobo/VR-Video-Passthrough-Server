from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import QObject, Signal

from ui.log_sanitizer import clean_log_text
from ui.services.hidden_process import HiddenProcess
from ui.services.process_helpers import ROOT, base_environment, offline_command, two_dvr_command
from utils.subprocess_hidden import hidden_subprocess_kwargs

_TERMINATE_WAIT_MS = 3000
_TASKKILL_WAIT_MS = 5000
_KILL_WAIT_MS = 3000


class OfflineProcess(QObject):
    output = Signal(str)
    state_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.process = HiddenProcess(self)
        self._stop_requested = False
        self.process.stdout.connect(self._read_stdout)
        self.process.stderr.connect(self._read_stderr)
        self.process.error.connect(self._error_occurred)
        self.process.finished.connect(self._finished)

    def _command(self) -> tuple[str, list[str]]:
        return offline_command()

    def is_running(self) -> bool:
        return self.process.is_running()

    def start(self, args: list[str], extra_env: dict[str, str] | None = None) -> None:
        if self.is_running():
            return
        self._stop_requested = False
        merged_env = base_environment(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8:replace",
                **(extra_env or {}),
            }
        )
        program, base_args = self._command()
        self.output.emit(
            "[offline] GPU runtime initialization may take a while on first use. "
            "If this is the first offline run after installation or a driver/model update, "
            "the window can stay quiet for 1-3 minutes while CUDA/ONNX Runtime caches are built.\n"
        )
        if self.process.start(program, [*base_args, *args], cwd=ROOT, env=merged_env):
            self.state_changed.emit(True)

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stop_requested = True
        self.output.emit("[offline] stop requested; terminating process\n")
        pid = self.process.process_id()
        self.process.terminate()
        if self.process.wait_for_finished(_TERMINATE_WAIT_MS):
            return
        taskkill_ok = False
        if sys.platform.startswith("win") and pid > 0:
            self.output.emit("[offline] process did not exit after terminate; killing process tree\n")
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
                self.output.emit(f"[offline] process tree taskkill failed: {type(exc).__name__}: {exc}\n")
        if taskkill_ok and self.process.wait_for_finished(_TASKKILL_WAIT_MS):
            return
        self.output.emit("[offline] process did not exit after tree kill; killing parent process\n")
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
        self.output.emit(f"[offline] process start error: {error}\n")
        if not self.is_running():
            self.state_changed.emit(False)

    def _finished(self, exit_code: int = 0) -> None:
        if self._stop_requested:
            self.output.emit(f"[offline] process stopped rc={exit_code}\n")
        elif exit_code != 0:
            self.output.emit(f"[offline] process exited with error rc={exit_code}\n")
        else:
            self.output.emit("[offline] process completed rc=0\n")
        self._stop_requested = False
        self.state_changed.emit(False)


class TwoDvrProcess(OfflineProcess):
    """Offline 2D->VR/3D converter process (offline/two_dvr.py)."""

    def _command(self) -> tuple[str, list[str]]:
        return two_dvr_command()
