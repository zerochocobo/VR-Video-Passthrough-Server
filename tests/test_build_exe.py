from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import build_exe


class BuildExeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("runtime_cache/test_build_exe")
        self.site = self.root / "site-packages"
        self.dist = self.root / "dist" / build_exe.APP_NAME
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        (self.site / "tensorrt_libs").mkdir(parents=True, exist_ok=True)
        for name in (
            "nvinfer_10.dll",
            "nvinfer_plugin_10.dll",
            "nvonnxparser_10.dll",
            "nvinfer_builder_resource_sm75_10.dll",
        ):
            (self.site / "tensorrt_libs" / name).write_bytes(b"dll")
        patchers = [
            patch.object(build_exe, "SITE_PACKAGES", self.site),
            patch.object(build_exe, "DIST_DIR", self.dist),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_copy_ort_tensorrt_ep_dependencies(self) -> None:
        build_exe.copy_ort_tensorrt_ep_dependencies()
        copied = self.dist / "_internal" / "tensorrt_libs"
        self.assertFalse((copied / "nvinfer_10.dll").exists())
        self.assertTrue((copied / "nvinfer_plugin_10.dll").exists())
        self.assertTrue((copied / "nvonnxparser_10.dll").exists())
        self.assertFalse((copied / "nvinfer_builder_resource_sm75_10.dll").exists())

    def test_verify_ort_tensorrt_ep_runtime(self) -> None:
        capi = self.dist / "_internal" / "onnxruntime" / "capi"
        trt = self.dist / "_internal" / "tensorrt_libs"
        capi.mkdir(parents=True, exist_ok=True)
        trt.mkdir(parents=True, exist_ok=True)
        (capi / "onnxruntime_providers_tensorrt.dll").write_bytes(b"dll")
        for name in ("nvinfer_plugin_10.dll", "nvonnxparser_10.dll"):
            (trt / name).write_bytes(b"dll")
        build_exe.verify_ort_tensorrt_ep_runtime()

    def test_verify_ort_tensorrt_ep_runtime_fails_when_provider_missing(self) -> None:
        with self.assertRaises(build_exe.BuildError):
            build_exe.verify_ort_tensorrt_ep_runtime()

    def test_build_server_embeds_internal_python_modules_without_source_data(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, env: dict[str, str]) -> None:
            calls.append(cmd)

        with patch.object(build_exe, "run", side_effect=fake_run):
            build_exe.build_server(["pyinstaller"], {})

        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        add_data = [cmd[index + 1] for index, token in enumerate(cmd) if token == "--add-data"]
        hidden_imports = [cmd[index + 1] for index, token in enumerate(cmd) if token == "--hidden-import"]
        collected_submodules = [cmd[index + 1] for index, token in enumerate(cmd) if token == "--collect-submodules"]
        self.assertIn("resources;resources", add_data)
        self.assertNotIn("tools;tools", add_data)
        self.assertNotIn("offline;offline", add_data)
        self.assertIn("offline.convert", hidden_imports)
        self.assertIn("offline.two_dvr", hidden_imports)
        self.assertIn("tools.offline_passthrough", hidden_imports)
        self.assertIn("tools.offline_alpha_passthrough", hidden_imports)
        self.assertIn("tools.warmup_offline_trt", hidden_imports)
        self.assertIn("tools.generate_yoloworld_person_txt_feats", hidden_imports)
        self.assertIn("offline", collected_submodules)


if __name__ == "__main__":
    unittest.main()
