from __future__ import annotations

import os
import site
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_DLL_HANDLES = []
if hasattr(os, "add_dll_directory"):
    for site_dir in site.getsitepackages():
        base = Path(site_dir)
        for dll_dir in (base / "PySide6", base / "shiboken6"):
            if dll_dir.exists():
                _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))

from PySide6.QtCore import QCoreApplication

from ui.services import startup_status_poller
from ui.services.startup_status_poller import StartupStatusPoller


def _app() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication([])


class StartupStatusPollerTests(unittest.TestCase):
    def test_stale_generation_response_is_ignored(self) -> None:
        _app()
        poller = StartupStatusPoller(max_duration_sec=0)
        updates: list[dict] = []
        finished: list[str] = []
        poller.updated.connect(lambda data: updates.append(dict(data)))
        poller.finished.connect(lambda phase: finished.append(str(phase)))
        poller._running = True
        poller._inflight = True
        poller._generation = 2

        poller._handle_response(1, b'{"phase":"listening"}')

        self.assertEqual(updates, [])
        self.assertEqual(finished, [])
        self.assertTrue(poller._inflight)

        poller._handle_response(2, b'{"phase":"listening"}')

        self.assertEqual(updates[-1]["phase"], "listening")
        self.assertEqual(finished, ["listening"])
        self.assertFalse(poller.is_running())

    def test_watchdog_finishes_failed_when_startup_never_reaches_terminal_phase(self) -> None:
        _app()
        poller = StartupStatusPoller(max_duration_sec=1)
        updates: list[dict] = []
        finished: list[str] = []
        poller.updated.connect(lambda data: updates.append(dict(data)))
        poller.finished.connect(lambda phase: finished.append(str(phase)))
        poller._running = True
        poller._generation = 1
        poller._started_at = 100.0

        with patch.object(startup_status_poller.time, "monotonic", return_value=102.0):
            poller._tick()

        self.assertEqual(updates[-1]["phase"], "failed")
        self.assertEqual(updates[-1]["reason"], "startup_status_timeout")
        self.assertEqual(finished, ["failed"])
        self.assertFalse(poller.is_running())


if __name__ == "__main__":
    unittest.main()
