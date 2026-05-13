from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import offline.convert as convert


class OfflineConvertTests(unittest.TestCase):
    def test_default_output_names(self) -> None:
        src = Path("sample.mp4")
        self.assertEqual(convert._default_out(src, "green"), Path("sample_passthrough.mp4"))
        self.assertEqual(convert._default_out(src, "alpha"), Path("sample_FISHEYE180_alpha.mp4"))

    def test_single_output_name_includes_mode_engine_start_and_duration(self) -> None:
        src = Path("sample.mp4")
        green_args = SimpleNamespace(command="single", mode="green", engine="rvm_fast", start=300.0, duration=15.0)
        alpha_args = SimpleNamespace(command="single", mode="alpha", engine="matanyone2", start=5.0, duration=300.0)
        all_args = SimpleNamespace(command="single", mode="green", engine="rvm_balanced", start=0.0, duration=0.0)

        self.assertEqual(convert._single_out(src, green_args), Path("sample_rvm1_S000500_15S_passthrough.mp4"))
        self.assertEqual(convert._single_out(src, alpha_args), Path("sample_matanyone2_S000005_5M_FISHEYE180_alpha.mp4"))
        self.assertEqual(convert._single_out(src, all_args), Path("sample_rvm2_S000000_ALL_passthrough.mp4"))

    def test_batch_video_files_skip_passthrough(self) -> None:
        root = Path("runtime_cache/test_offline_convert")
        shutil.rmtree(root, ignore_errors=True)
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
            skip_frames=2,
            bitrate="live",
            preset="P5",
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertIn("--start", cmd)
        self.assertIn("12.5", cmd)
        self.assertIn("--duration", cmd)
        self.assertIn("30.0", cmd)
        self.assertIn("--model", cmd)
        self.assertIn(str(convert.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx"), cmd)
        self.assertNotIn(str(convert.ROOT / "models" / "rvm_resnet50_fp32.onnx"), cmd)
        self.assertNotIn("--fps", cmd)
        self.assertIn("--input-size", cmd)
        self.assertIn("1024", cmd)
        self.assertIn("--alpha-stride", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--sbs-batch", cmd)
        self.assertIn("--bitrate", cmd)
        self.assertIn("live", cmd)
        self.assertIn("--preset", cmd)
        self.assertIn("P5", cmd)
        self.assertIn("--cq", cmd)
        self.assertIn("-1", cmd)
        self.assertIn("--audio", cmd)
        self.assertIn("copy", cmd)

    def test_frozen_command_uses_internal_tool_subcommand(self) -> None:
        args = SimpleNamespace(
            mode="green",
            engine="rvm_fast",
            start=0.0,
            duration=0.0,
            fps=30.0,
            input_size=1024,
            skip_frames=0,
            bitrate="live",
            preset="P5",
            cq=-1,
        )
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", r"C:\App\pt_core.exe"):
            cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertEqual(cmd[:3], [r"C:\App\pt_core.exe", "tool", "offline_passthrough"])
        self.assertNotIn("offline_passthrough.py", cmd[1:3])

    def test_balanced_engine_uses_resnet_model(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="rvm_balanced",
            start=0.0,
            duration=0.0,
            fps=0.0,
            input_size=1024,
            skip_frames=0,
            bitrate="live",
            preset="P4",
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertIn(str(convert.ROOT / "models" / "rvm_resnet50_fp32.onnx"), cmd)

    def test_matanyone2_command_does_not_receive_rvm_speed_args(self) -> None:
        args = SimpleNamespace(
            mode="alpha",
            engine="matanyone2",
            start=0.0,
            duration=0.0,
            fps=30.0,
            input_size=1024,
            skip_frames=2,
            bitrate="live",
            preset="P4",
        )
        cmd = convert._base_cmd(args, Path("input.mp4"), Path("out.mp4"))
        self.assertIn("--fps", cmd)
        self.assertNotIn("--input-size", cmd)
        self.assertNotIn("--alpha-stride", cmd)

    def test_single_out_dir_uses_default_passthrough_name(self) -> None:
        root = Path("runtime_cache/test_offline_out_dir")
        shutil.rmtree(root, ignore_errors=True)
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

        convert._base_cmd = fake_base_cmd
        try:
            args = SimpleNamespace(
                command="single",
                out_dir=str(root / "out"),
                out="",
                mode="alpha",
                engine="rvm_balanced",
                start=300.0,
                duration=15.0,
                fps=0.0,
                input_size=1024,
                skip_frames=0,
                bitrate="live",
                preset="P4",
                skip_existing=False,
                cq=-1,
            )
            self.assertEqual(convert._run_one(args, src), 0)
        finally:
            convert._base_cmd = original_base_cmd

        self.assertEqual(seen["out"], (root / "out" / "demo_rvm2_S000500_15S_FISHEYE180_alpha.mp4").resolve())


if __name__ == "__main__":
    unittest.main()
