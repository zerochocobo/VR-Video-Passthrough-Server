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


if __name__ == "__main__":
    unittest.main()
