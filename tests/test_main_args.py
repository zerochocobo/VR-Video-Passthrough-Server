from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
