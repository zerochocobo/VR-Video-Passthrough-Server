from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import offline.convert as convert


class OfflineConvertTests(unittest.TestCase):
    def test_default_output_names(self) -> None:
        src = Path("sample.mp4")
        self.assertEqual(convert._default_out(src, "green", 1920, 1080), Path("sample_passthrough.mp4"))
        self.assertEqual(convert._default_out(src, "green", 3840, 1920), Path("sample_LR_180_SBS_passthrough.mp4"))
        self.assertEqual(convert._default_out(src, "alpha"), Path("sample_LR_180_FISHEYE_F180_alpha.mp4"))

    def test_single_output_name_includes_mode_engine_start_end_and_duration(self) -> None:
        src = Path("sample.mp4")
        green_args = SimpleNamespace(command="single", mode="green", engine="rvm_fast", start=300.0, duration=15.0)
        alpha_args = SimpleNamespace(command="single", mode="alpha", engine="matanyone2", start=5.0, duration=300.0)
        all_args = SimpleNamespace(command="single", mode="green", engine="rvm_fast", start=0.0, duration=0.0)

        self.assertEqual(
            convert._single_out(src, green_args, 3840, 1920),
            Path("sample_rvm1_S000500_E000515_15S_LR_180_SBS_passthrough.mp4"),
        )
        self.assertEqual(convert._single_out(src, alpha_args), Path("sample_matanyone2_S000005_E000505_5M_LR_180_FISHEYE_F180_alpha.mp4"))
        self.assertEqual(convert._single_out(src, all_args, 1920, 1080), Path("sample_rvm1_S000000_ALL_passthrough.mp4"))

    def test_single_segments_output_name_includes_segment_count_and_range(self) -> None:
        src = Path("sample.mp4")
        args = SimpleNamespace(command="single", mode="green", engine="rvm_fast")
        self.assertEqual(
            convert._single_segments_out(src, args, [(0.0, 15.0), (60.0, 90.0)], 3840, 1920),
            Path("sample_rvm1_SEG2_S000000_E000130_LR_180_SBS_passthrough.mp4"),
        )

    def test_segment_arg_parses_hhmmss_range(self) -> None:
        self.assertEqual(convert._segment_arg("00:01:00-00:01:30"), (60.0, 90.0))
        with self.assertRaises(Exception):
            convert._segment_arg("00:01:00-00:00:30")

    def test_batch_video_files_skip_passthrough(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pt_offline_convert_"))
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        (root / "a.mp4").write_text("a", encoding="utf-8")
        (root / "b_ALPHA_passthrough.mp4").write_text("b", encoding="utf-8")
        (root / "c.txt").write_text("c", encoding="utf-8")
        files = convert._video_files(root, recursive=False)
        self.assertEqual([p.name for p in files], ["a.mp4"])

    def test_command_uses_formal_arguments(self) -> None:
        args = SimpleNamespace(
            mode="green",
            engine="rvm_fast",
            start=12.5,
            duration=30.0,
            fps=0.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=2,
            bitrate="source",
            preset="P5",
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertIn("--start", cmd)
        self.assertIn("12.5", cmd)
        self.assertIn("--duration", cmd)
        self.assertIn("30.0", cmd)
        self.assertIn("--model", cmd)
        self.assertIn(str(convert.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx"), cmd)
        self.assertNotIn("--fps", cmd)
        self.assertIn("--input-size", cmd)
        self.assertIn("1024", cmd)
        self.assertIn("--rvm-downsample-ratio", cmd)
        self.assertIn("0.5", cmd)
        self.assertIn("--alpha-stride", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--sbs-batch", cmd)
        self.assertIn("--bitrate", cmd)
        self.assertIn("source", cmd)
        self.assertIn("--preset", cmd)
        self.assertIn("P5", cmd)
        self.assertIn("--cq", cmd)
        self.assertIn("-1", cmd)
        self.assertIn("--audio", cmd)
        self.assertIn("copy", cmd)

    def test_rvm_fast_offline_env_keeps_tensorrt_when_cache_ready(self) -> None:
        args = SimpleNamespace(engine="rvm_fast")
        base = {"PT_ONNX_PROVIDERS": "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"}
        with patch.object(convert, "cache_status", return_value="ready"):
            env = convert._offline_child_env(args, base)
        self.assertEqual(env["PT_ONNX_PROVIDERS"], convert.TRT_PROVIDER_CHAIN)
        self.assertEqual(env["PT_OFFLINE_RVM_TRT"], "1")

    def test_matanyone2_offline_env_keeps_tensorrt_when_cache_ready(self) -> None:
        args = SimpleNamespace(engine="matanyone2")
        base = {"PT_ONNX_PROVIDERS": "CUDAExecutionProvider,CPUExecutionProvider"}
        with patch.object(convert, "cache_status", return_value="ready"):
            env = convert._offline_child_env(args, base)
        self.assertEqual(env["PT_ONNX_PROVIDERS"], convert.TRT_PROVIDER_CHAIN)
        self.assertEqual(env["PT_OFFLINE_MATANYONE2_TRT"], "1")

    def test_offline_env_honors_model_tensorrt_disable_flags(self) -> None:
        base = {
            "PT_ONNX_PROVIDERS": "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
            "PT_OFFLINE_RVM_TRT_ENABLE": "0",
            "PT_OFFLINE_MATANYONE2_TRT_ENABLE": "0",
        }
        with patch.object(convert, "cache_status", return_value="ready"):
            rvm_env = convert._offline_child_env(SimpleNamespace(engine="rvm_fast"), base)
            mat_env = convert._offline_child_env(SimpleNamespace(engine="matanyone2"), base)
        self.assertEqual(rvm_env["PT_ONNX_PROVIDERS"], "CUDAExecutionProvider,CPUExecutionProvider")
        self.assertEqual(mat_env["PT_ONNX_PROVIDERS"], "CUDAExecutionProvider,CPUExecutionProvider")
        self.assertNotIn("PT_OFFLINE_RVM_TRT", rvm_env)
        self.assertNotIn("PT_OFFLINE_MATANYONE2_TRT", mat_env)

    def test_matanyone2_offline_env_strips_tensorrt_when_cache_missing(self) -> None:
        base = {"PT_ONNX_PROVIDERS": "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"}
        for engine in ("matanyone2",):
            with self.subTest(engine=engine):
                with patch.object(convert, "cache_status", return_value="missing"):
                    env = convert._offline_child_env(SimpleNamespace(engine=engine), base)
                self.assertEqual(env["PT_ONNX_PROVIDERS"], "CUDAExecutionProvider,CPUExecutionProvider")
                self.assertNotIn("PT_OFFLINE_RVM_TRT", env)
                self.assertNotIn("PT_OFFLINE_MATANYONE2_TRT", env)

    def test_frozen_command_uses_internal_tool_subcommand(self) -> None:
        args = SimpleNamespace(
            mode="green",
            engine="rvm_fast",
            start=0.0,
            duration=0.0,
            fps=30.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=0,
            bitrate="source",
            preset="P5",
            cq=-1,
        )
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", r"C:\App\pt_core.exe"):
            cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertEqual(cmd[:3], [r"C:\App\pt_core.exe", "tool", "offline_passthrough"])
        self.assertNotIn("offline_passthrough.py", cmd[1:3])

    def test_matanyone2_command_does_not_receive_rvm_speed_args(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="matanyone2",
            start=0.0,
            duration=0.0,
            fps=30.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=2,
            bitrate="source",
            preset="P4",
            matanyone2_size=1024,
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertIn("--fps", cmd)
        self.assertIn("--bitrate", cmd)
        self.assertIn("source", cmd)
        self.assertIn("--preset", cmd)
        self.assertIn("P4", cmd)
        self.assertIn("--cq", cmd)
        self.assertIn("-1", cmd)
        self.assertNotIn("--input-size", cmd)
        self.assertNotIn("--rvm-downsample-ratio", cmd)
        self.assertIn("--matanyone2-size", cmd)
        self.assertIn("1024", cmd)
        self.assertIn("--alpha-stride", cmd)
        self.assertIn("1", cmd)

    def test_matanyone2_medium_defaults_to_yolo26m_prepass(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="matanyone2_medium",
            start=0.0,
            duration=0.0,
            fps=30.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=0,
            bitrate="source",
            preset="P4",
            matanyone2_size=1024,
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        index = cmd.index("--matanyone2-prepass")
        self.assertEqual(cmd[index + 1], "yolo26m_efficientsam")

    def test_matanyone2_medium_accepts_birefnet_prepass(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="matanyone2_medium",
            matanyone2_prepass="yolo26m_birefnet",
            start=0.0,
            duration=0.0,
            fps=30.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=0,
            bitrate="source",
            preset="P4",
            matanyone2_size=1024,
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        index = cmd.index("--matanyone2-prepass")
        self.assertEqual(cmd[index + 1], "yolo26m_birefnet")

    def test_matanyone2_command_accepts_512_size(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="matanyone2",
            start=0.0,
            duration=0.0,
            fps=0.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=0,
            bitrate="source",
            preset="P4",
            matanyone2_size=512,
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        index = cmd.index("--matanyone2-size")
        self.assertEqual(cmd[index + 1], "512")

    def test_matanyone2_command_rejects_unsupported_size(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="matanyone2",
            start=0.0,
            duration=0.0,
            fps=0.0,
            input_size=1024,
            rvm_downsample_ratio=0.5,
            skip_frames=0,
            bitrate="source",
            preset="P4",
            matanyone2_size=2048,
        )
        with self.assertRaises(ValueError):
            convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))

    def test_run_hidden_streaming_forwards_output_and_return_code(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        script = "import sys; print('child-out'); print('child-err', file=sys.stderr); sys.exit(5)"
        env = dict(os.environ)

        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            rc = convert.run_hidden_streaming([sys.executable, "-c", script], env=env, exit_label="offline")

        self.assertEqual(rc, 5)
        self.assertIn("child-out", out.getvalue())
        self.assertIn("[offline] child process exited rc=5", out.getvalue())
        self.assertIn("child-err", err.getvalue())

    def test_single_out_dir_uses_default_passthrough_name(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pt_offline_out_dir_"))
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "out").mkdir(parents=True, exist_ok=True)
        src = root / "src" / "demo.mp4"
        src.write_text("video", encoding="utf-8")
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        seen: dict[str, Path] = {}
        original_base_cmd = convert._base_cmd

        def fake_base_cmd(args, src_path, out_path):
            seen["out"] = out_path
            return ["python", "-c", "import sys; sys.exit(0)"]

        fake_meta = SimpleNamespace(
            timing=SimpleNamespace(),
            codec=SimpleNamespace(codec_name="hevc", profile="Main", pix_fmt="yuv420p", width=3840, height=1920),
            color=SimpleNamespace(),
        )
        fake_decision = SimpleNamespace(verdict="pynv_hevc", reason="test")
        convert._base_cmd = fake_base_cmd
        try:
            args = SimpleNamespace(
                command="single",
                out_dir=str(root / "out"),
                out="",
                mode="alpha",
                engine="rvm_fast",
                start=300.0,
                duration=15.0,
                fps=0.0,
                input_size=1024,
                rvm_downsample_ratio=0.5,
                skip_frames=0,
                bitrate="source",
                preset="P4",
                skip_existing=False,
                cq=-1,
            )
            with patch.object(convert, "probe_video_metadata", return_value=fake_meta), patch.object(
                convert, "select_backend", return_value=fake_decision
            ):
                self.assertEqual(convert._run_one(args, src), 0)
        finally:
            convert._base_cmd = original_base_cmd

        self.assertEqual(seen["out"], (root / "out" / "demo_rvm1_S000500_E000515_15S_LR_180_FISHEYE_F180_alpha.mp4").resolve())

    def test_run_segments_generates_each_part_and_concats(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="pt_offline_segments_"))
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "out").mkdir(parents=True, exist_ok=True)
        src = root / "src" / "demo.mp4"
        src.write_text("video", encoding="utf-8")
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        fake_meta = SimpleNamespace(
            timing=SimpleNamespace(duration=600.0),
            codec=SimpleNamespace(width=3840, height=1920),
        )
        args = SimpleNamespace(
            command="single",
            out_dir=str(root / "out"),
            out="",
            mode="green",
            engine="rvm_fast",
            segments=[(0.0, 15.0), (60.0, 90.0)],
            skip_existing=False,
        )
        seen: dict[str, object] = {"parts": []}

        def fake_run_one(segment_args, src_path):
            seen["parts"].append((segment_args.start, segment_args.duration, Path(segment_args.out).name))
            Path(segment_args.out).write_text("part", encoding="utf-8")
            return 0

        def fake_concat(paths, out, work_dir):
            seen["concat_paths"] = [Path(path).name for path in paths]
            seen["out"] = out
            out.write_text("final", encoding="utf-8")
            return 0

        gpu_requirement = SimpleNamespace(detected=False, supported=True)
        with patch.object(convert, "detect_nvidia_gpu_requirement", return_value=gpu_requirement), patch.object(
            convert, "probe_video_metadata", return_value=fake_meta
        ), patch.object(convert, "_run_one", side_effect=fake_run_one), patch.object(convert, "_concat_segments", side_effect=fake_concat):
            self.assertEqual(convert._run_segments(args, src), 0)

        self.assertEqual(seen["parts"], [(0.0, 15.0, "part_001.mp4"), (60.0, 30.0, "part_002.mp4")])
        self.assertEqual(seen["concat_paths"], ["part_001.mp4", "part_002.mp4"])
        self.assertEqual(seen["out"], (root / "out" / "demo_rvm1_SEG2_S000000_E000130_LR_180_SBS_passthrough.mp4").resolve())


if __name__ == "__main__":
    unittest.main()
