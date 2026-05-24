from __future__ import annotations

import shutil
import unittest
import logging
from pathlib import Path

from utils.logger import _LoggerNameAliasFilter, _UvicornSocketSendNoiseFilter, _rotate_server_logs


class LoggerRotationTests(unittest.TestCase):
    def test_rotate_server_logs_keeps_five_archives(self) -> None:
        root = Path("runtime_cache/test_logger_rotation")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        (root / "server.log").write_text("current", encoding="utf-8")
        for idx in range(1, 6):
            (root / f"server_{idx}.log").write_text(str(idx), encoding="utf-8")
        _rotate_server_logs(root, keep=5)
        self.assertFalse((root / "server.log").exists())
        self.assertEqual((root / "server_1.log").read_text(encoding="utf-8"), "current")
        self.assertEqual((root / "server_2.log").read_text(encoding="utf-8"), "1")
        self.assertEqual((root / "server_5.log").read_text(encoding="utf-8"), "4")

    def test_uvicorn_error_logger_is_displayed_as_server(self) -> None:
        record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, "ok", (), None)
        filt = _LoggerNameAliasFilter({"uvicorn.error": "uvicorn.server"})
        self.assertTrue(filt.filter(record))
        self.assertEqual(record.name, "uvicorn.server")

    def test_uvicorn_socket_send_noise_is_dropped(self) -> None:
        record = logging.LogRecord(
            "uvicorn.error",
            logging.WARNING,
            __file__,
            1,
            "socket.send() raised exception",
            (),
            None,
        )
        filt = _UvicornSocketSendNoiseFilter()
        self.assertFalse(filt.filter(record))

    def test_uvicorn_other_messages_are_kept(self) -> None:
        record = logging.LogRecord("uvicorn.error", logging.WARNING, __file__, 1, "real warning", (), None)
        filt = _UvicornSocketSendNoiseFilter()
        self.assertTrue(filt.filter(record))


if __name__ == "__main__":
    unittest.main()
