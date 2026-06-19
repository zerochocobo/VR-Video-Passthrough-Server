from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

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
        fake_tool = types.ModuleType("tools.offline_passthrough")
        fake_tool.main = Mock(side_effect=fake_tool_main)
        with patch.dict(sys.modules, {"tools.offline_passthrough": fake_tool}):
            self.assertEqual(main.main(["tool", "offline_passthrough", "--help"]), 7)
        fake_tool.main.assert_called_once_with()
        self.assertEqual(seen["argv"], ["offline_passthrough", "--help"])
        self.assertEqual(sys.argv, original_argv)

    def test_main_trt_warmup_dispatches_without_starting_server(self) -> None:
        with patch("ui.services.trt_warmup_process.main", return_value=9) as warmup_main:
            self.assertEqual(main.main(["trt_warmup", "--progress-stdout"]), 9)
        warmup_main.assert_called_once_with(["--progress-stdout"])

    def test_main_two_dvr_dispatches_without_starting_server(self) -> None:
        seen: dict[str, list[str]] = {}
        fake_two_dvr = types.ModuleType("offline.two_dvr")

        def fake_two_dvr_main(argv: list[str]) -> int:
            seen["argv"] = argv
            return 11

        fake_two_dvr.main = fake_two_dvr_main
        with patch.dict(sys.modules, {"offline.two_dvr": fake_two_dvr}):
            self.assertEqual(main.main(["two_dvr", "single", "video.mp4"]), 11)
        self.assertEqual(seen["argv"], ["single", "video.mp4"])

    def test_main_tool_forces_line_buffered_output(self) -> None:
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if stdout_reconfigure is None or stderr_reconfigure is None:
            self.skipTest("stdio reconfigure is unavailable")
        with patch.object(sys.stdout, "reconfigure") as stdout_configure, patch.object(sys.stderr, "reconfigure") as stderr_configure:
            fake_tool = types.ModuleType("tools.offline_passthrough")
            fake_tool.main = Mock(return_value=0)
            with patch.dict(sys.modules, {"tools.offline_passthrough": fake_tool}):
                self.assertEqual(main.main(["tool", "offline_passthrough", "--help"]), 0)
        fake_tool.main.assert_called_once_with()
        stdout_configure.assert_called_with(line_buffering=True, write_through=True)
        stderr_configure.assert_called_with(line_buffering=True, write_through=True)


if __name__ == "__main__":
    unittest.main()
