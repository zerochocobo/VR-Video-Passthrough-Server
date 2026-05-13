from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import config
import main


class MainArgsTests(unittest.TestCase):
    def test_debug_positional_enables_verbose_logs(self) -> None:
        original_env = os.environ.get("PT_DEBUG_LOGS")
        original_config = config.DEBUG_LOGS
        try:
            os.environ.pop("PT_DEBUG_LOGS", None)
            config.DEBUG_LOGS = False
            args = main._parse_args(["DEBUG"])
            main._apply_debug_arg(args)
            self.assertEqual(os.environ["PT_DEBUG_LOGS"], "1")
            self.assertTrue(config.DEBUG_LOGS)
        finally:
            if original_env is None:
                os.environ.pop("PT_DEBUG_LOGS", None)
            else:
                os.environ["PT_DEBUG_LOGS"] = original_env
            config.DEBUG_LOGS = original_config

    def test_debug_flag_enables_verbose_logs(self) -> None:
        original_env = os.environ.get("PT_DEBUG_LOGS")
        original_config = config.DEBUG_LOGS
        try:
            os.environ.pop("PT_DEBUG_LOGS", None)
            config.DEBUG_LOGS = False
            args = main._parse_args(["--debug"])
            main._apply_debug_arg(args)
            self.assertEqual(os.environ["PT_DEBUG_LOGS"], "1")
            self.assertTrue(config.DEBUG_LOGS)
        finally:
            if original_env is None:
                os.environ.pop("PT_DEBUG_LOGS", None)
            else:
                os.environ["PT_DEBUG_LOGS"] = original_env
            config.DEBUG_LOGS = original_config

    def test_main_accepts_argv_before_starting_runtime(self) -> None:
        with patch.object(main, "_apply_debug_arg") as apply_debug, patch.object(main, "configure_gpu_runtime_cache", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                main.main(["DEBUG"])
        apply_debug.assert_called_once()

    def test_main_tool_dispatches_without_starting_server(self) -> None:
        seen: dict[str, list[str]] = {}

        def fake_tool_main() -> int:
            seen["argv"] = sys.argv[:]
            return 7

        original_argv = sys.argv[:]
        with patch("tools.offline_passthrough.main", side_effect=fake_tool_main) as tool_main:
            self.assertEqual(main.main(["tool", "offline_passthrough", "--help"]), 7)
        tool_main.assert_called_once_with()
        self.assertEqual(seen["argv"], ["offline_passthrough", "--help"])
        self.assertEqual(sys.argv, original_argv)

    def test_main_tool_forces_line_buffered_output(self) -> None:
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if stdout_reconfigure is None or stderr_reconfigure is None:
            self.skipTest("stdio reconfigure is unavailable")
        with patch.object(sys.stdout, "reconfigure") as stdout_configure, patch.object(sys.stderr, "reconfigure") as stderr_configure:
            with patch("tools.offline_passthrough.main", return_value=0):
                self.assertEqual(main.main(["tool", "offline_passthrough", "--help"]), 0)
        stdout_configure.assert_called_with(line_buffering=True, write_through=True)
        stderr_configure.assert_called_with(line_buffering=True, write_through=True)


if __name__ == "__main__":
    unittest.main()
