from __future__ import annotations

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ui.log_sanitizer import clean_log_text
from ui.services.process_helpers import ROOT, base_environment, offline_command


class OfflineProcess(QObject):
    output = Signal(str)
    state_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)

    def is_running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def start(self, args: list[str], extra_env: dict[str, str] | None = None) -> None:
        if self.is_running():
            return
        self.process.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        merged_env = base_environment({"PYTHONUNBUFFERED": "1", **(extra_env or {})})
        for key, value in merged_env.items():
            env.insert(str(key), str(value))
        self.process.setProcessEnvironment(env)
        program, base_args = offline_command()
        self.output.emit(
            "[offline] GPU runtime initialization may take a while on first use. "
            "If this is the first offline run after installation or a driver/model update, "
            "the window can stay quiet for 1-3 minutes while CUDA/ONNX Runtime caches are built.\n"
        )
        self.process.start(program, [*base_args, *args])
        self.state_changed.emit(True)

    def stop(self) -> None:
        if not self.is_running():
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()

    def _read_stdout(self) -> None:
        data = clean_log_text(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))
        if data:
            self.output.emit(data)

    def _read_stderr(self) -> None:
        data = clean_log_text(bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))
        if data:
            self.output.emit(data)

    def _finished(self) -> None:
        self.state_changed.emit(False)
