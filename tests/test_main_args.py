from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

import config
import main


class MainArgsTests(unittest.TestCase):
    def test_startup_plan_includes_configured_long_running_steps(self) -> None:
        with (
            patch.object(config, "DLNA_ALL_VIDEOS_ENABLED", True),
            patch.object(config, "GPU_REPAIR_REBUILD_TRT", True),
            patch.object(config, "ONNX_PROVIDERS", ["TensorrtExecutionProvider", "CUDAExecutionProvider"]),
            patch.object(config, "RTX_VSR_REALTIME_ENABLED", True),
            patch.object(config, "PASSTHROUGH_OUTPUT_MODE", "superres,two_dvr"),
            patch.object(config, "STARTUP_GPU_WARMUP", True),
            patch.object(config, "WARMUP_COMPOSITE_ENABLE", True),
            patch.object(config, "USE_PYNV", True),
            patch.object(config, "NVENC_PREFLIGHT_ENABLE", True),
            patch.object(main, "_passthrough_mode_enabled", return_value=True),
            patch("utils.runtime_settings.get_rm", return_value=types.SimpleNamespace(enabled=True)),
        ):
            keys = [key for key, _estimate in main._startup_plan_steps(active_provider_kind="trt")]

        for expected in (
            "media_index",
            "trt_rebuild",
            "trt_validate",
            "vsr_preflight",
            "static_trt_preload",
            "da3_trt_warmup",
            "rm_trt_warmup",
            "nvenc_preflight",
            "runtime_pool",
            "listening",
        ):
            self.assertIn(expected, keys)
        self.assertEqual(keys, list(dict.fromkeys(keys)))

    def test_startup_plan_uses_final_provider_for_runtime_steps(self) -> None:
        with (
            patch.object(config, "ONNX_PROVIDERS", ["TensorrtExecutionProvider", "CUDAExecutionProvider"]),
            patch.object(config, "STARTUP_GPU_WARMUP", True),
            patch.object(config, "WARMUP_COMPOSITE_ENABLE", False),
            patch.object(main, "_passthrough_mode_enabled", return_value=False),
            patch("utils.runtime_settings.get_rm", return_value=types.SimpleNamespace(enabled=False)),
        ):
            pending_steps = dict(main._startup_plan_steps(requested_trt=True))
            trt_steps = dict(main._startup_plan_steps(requested_trt=True, active_provider_kind="trt"))
            cuda_steps = dict(main._startup_plan_steps(requested_trt=True, active_provider_kind="cuda"))

        self.assertIn("trt_validate", cuda_steps)
        self.assertNotIn("static_trt_preload", pending_steps)
        self.assertEqual(pending_steps["ort_iobinding_runs"], 90.0)
        self.assertIn("static_trt_preload", trt_steps)
        self.assertNotIn("static_trt_preload", cuda_steps)
        self.assertEqual(trt_steps["ort_iobinding_runs"], 12.0)
        self.assertEqual(cuda_steps["ort_iobinding_runs"], 90.0)

    def test_gpu_repair_rebuilds_realtime_trt_in_isolated_process(self) -> None:
        log = Mock()
        completed = subprocess.CompletedProcess(["python"], 0, stdout="DONE")
        with (
            patch("ui.services.process_helpers.trt_warmup_command", return_value=("python", ["-m", "warmup"])),
            patch("ui.services.process_helpers.base_environment", return_value={"PT_GPU_REPAIR_REBUILD_TRT": "1"}),
            patch.object(main.subprocess, "run", return_value=completed) as run,
            patch.object(main, "set_startup_phase") as set_status,
            patch.object(main, "start_heartbeat"),
            patch.object(main, "stop_heartbeat"),
        ):
            self.assertTrue(main._rebuild_realtime_trt_cache_isolated(log))

        command = run.call_args.args[0]
        self.assertIn("--model", command)
        self.assertIn("rvm", command)
        self.assertNotIn("PT_GPU_REPAIR_REBUILD_TRT", run.call_args.kwargs["env"])
        first_status = set_status.call_args_list[0]
        self.assertTrue(first_status.kwargs["trt_building"])
        self.assertEqual(first_status.kwargs["trt_build_model"], "rvm")
        self.assertEqual(first_status.kwargs["minimum_step_estimate_sec"], 300.0)

    def test_gpu_repair_trt_rebuild_failure_allows_cuda_fallback(self) -> None:
        log = Mock()
        completed = subprocess.CompletedProcess(["python"], 9, stdout="ERROR:boom")
        with (
            patch("ui.services.process_helpers.trt_warmup_command", return_value=("python", [])),
            patch("ui.services.process_helpers.base_environment", return_value={}),
            patch.object(main.subprocess, "run", return_value=completed),
            patch.object(main, "set_startup_phase"),
            patch.object(main, "start_heartbeat"),
            patch.object(main, "stop_heartbeat"),
        ):
            self.assertFalse(main._rebuild_realtime_trt_cache_isolated(log))

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
