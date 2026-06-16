from __future__ import annotations

import subprocess
import os
import unittest
from unittest.mock import patch

from ui.qt_runtime import configure_qt_runtime_paths

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
configure_qt_runtime_paths()

from PySide6.QtWidgets import QApplication

from ui.services import offline_process, server_process
from ui.services.hidden_process import HiddenProcess
from ui.services.offline_process import OfflineProcess
from ui.services.server_process import ServerProcess


def _app() -> QApplication:
    app = QApplication.instance()
    if app is not None:
        return app
    return QApplication([])


class _FakeProc:
    pid = 123

    def __init__(self, rc: int = 0) -> None:
        self.rc = rc

    def wait(self) -> int:
        return self.rc


class HiddenProcessTests(unittest.TestCase):
    def test_old_wait_loop_does_not_emit_finished_for_new_generation(self) -> None:
        _app()
        process = HiddenProcess()
        old_proc = _FakeProc(0)
        new_proc = _FakeProc(0)
        finished: list[int] = []
        process.finished.connect(lambda rc: finished.append(int(rc)))
        with process._lock:
            process._generation = 2
            process._proc = new_proc  # type: ignore[assignment]

        process._wait_loop(old_proc, 1)  # type: ignore[arg-type]

        self.assertEqual(finished, [])
        self.assertIs(process._proc, new_proc)

        process._wait_loop(new_proc, 2)  # type: ignore[arg-type]

        self.assertEqual(finished, [0])
        self.assertIsNone(process._proc)

    def test_wait_loop_normalizes_unsigned_windows_exit_code(self) -> None:
        _app()
        process = HiddenProcess()
        proc = _FakeProc(0xFFFFFFFF)
        finished: list[int] = []
        process.finished.connect(lambda rc: finished.append(int(rc)))
        with process._lock:
            process._generation = 1
            process._proc = proc  # type: ignore[assignment]

        process._wait_loop(proc, 1)  # type: ignore[arg-type]

        self.assertEqual(finished, [-1])
        self.assertIsNone(process._proc)


class _FakeHiddenProcess:
    def __init__(self, waits: list[bool], pid: int = 123) -> None:
        self.waits = list(waits)
        self.wait_ms: list[int] = []
        self.pid = pid
        self.terminated = False
        self.killed = False

    def is_running(self) -> bool:
        return True

    def process_id(self) -> int:
        return self.pid

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait_for_finished(self, timeout_ms: int) -> bool:
        self.wait_ms.append(timeout_ms)
        return self.waits.pop(0) if self.waits else False


class ServerProcessStopTests(unittest.TestCase):
    def test_stop_waits_after_successful_taskkill(self) -> None:
        _app()
        service = ServerProcess()
        fake_process = _FakeHiddenProcess([False, True])
        service.process = fake_process  # type: ignore[assignment]
        result = subprocess.CompletedProcess(["taskkill"], 0)

        with (
            patch.object(server_process.sys, "platform", "win32"),
            patch.object(server_process.subprocess, "run", return_value=result) as run,
        ):
            service.stop()

        run.assert_called_once()
        self.assertTrue(fake_process.terminated)
        self.assertFalse(fake_process.killed)
        self.assertEqual(fake_process.wait_ms, [3000, 5000])

    def test_stop_kills_when_taskkill_fails(self) -> None:
        _app()
        service = ServerProcess()
        fake_process = _FakeHiddenProcess([False, False])
        service.process = fake_process  # type: ignore[assignment]
        result = subprocess.CompletedProcess(["taskkill"], 1)

        with (
            patch.object(server_process.sys, "platform", "win32"),
            patch.object(server_process.subprocess, "run", return_value=result),
        ):
            service.stop()

        self.assertTrue(fake_process.terminated)
        self.assertTrue(fake_process.killed)
        self.assertEqual(fake_process.wait_ms, [3000, 3000])


class OfflineProcessStopTests(unittest.TestCase):
    def test_stop_waits_after_successful_taskkill(self) -> None:
        _app()
        service = OfflineProcess()
        fake_process = _FakeHiddenProcess([False, True])
        service.process = fake_process  # type: ignore[assignment]
        result = subprocess.CompletedProcess(["taskkill"], 0)
        output: list[str] = []
        service.output.connect(lambda text: output.append(str(text)))

        with (
            patch.object(offline_process.sys, "platform", "win32"),
            patch.object(offline_process.subprocess, "run", return_value=result) as run,
        ):
            service.stop()

        run.assert_called_once()
        self.assertTrue(fake_process.terminated)
        self.assertFalse(fake_process.killed)
        self.assertEqual(fake_process.wait_ms, [3000, 5000])
        self.assertTrue(any("killing process tree" in line for line in output))

    def test_stop_kills_when_taskkill_fails(self) -> None:
        _app()
        service = OfflineProcess()
        fake_process = _FakeHiddenProcess([False, False])
        service.process = fake_process  # type: ignore[assignment]
        result = subprocess.CompletedProcess(["taskkill"], 1)

        with (
            patch.object(offline_process.sys, "platform", "win32"),
            patch.object(offline_process.subprocess, "run", return_value=result),
        ):
            service.stop()

        self.assertTrue(fake_process.terminated)
        self.assertTrue(fake_process.killed)
        self.assertEqual(fake_process.wait_ms, [3000, 3000])


if __name__ == "__main__":
    unittest.main()
