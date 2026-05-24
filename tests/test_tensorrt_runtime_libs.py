from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from utils import tensorrt_runtime_libs as trt_libs
from utils.gpu_requirements import GpuRequirementResult


class TensorRTRuntimeLibTests(unittest.TestCase):
    def test_sm_resource_name(self) -> None:
        self.assertEqual(trt_libs.sm_resource_name("12.0"), "nvinfer_builder_resource_sm120_10.dll")
        self.assertEqual(trt_libs.sm_resource_name("8.9"), "nvinfer_builder_resource_sm89_10.dll")

    def test_check_distinguishes_standard_and_sm_dlls(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in trt_libs.STANDARD_TRT_DLLS:
                (root / name).write_bytes(b"dll")
            gpu = GpuRequirementResult(True, True, name="GPU", compute_capability="12.0")
            with patch.object(trt_libs, "detect_nvidia_gpu_requirement", return_value=gpu):
                status = trt_libs.check_tensorrt_runtime_libs(root)
            self.assertFalse(status.ready)
            self.assertEqual(status.missing_standard, ())
            self.assertEqual(status.missing_sm, ("nvinfer_builder_resource_sm120_10.dll",))

    def test_extract_required_libs_from_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            whl = root / "trt.whl"
            with zipfile.ZipFile(whl, "w") as archive:
                for name in (*trt_libs.STANDARD_TRT_DLLS, "nvinfer_builder_resource_sm120_10.dll"):
                    archive.writestr(f"tensorrt_libs/{name}", b"dll")
                archive.writestr("tensorrt_libs/nvinfer_builder_resource_sm75_10.dll", b"skip")
            gpu = GpuRequirementResult(True, True, name="GPU", compute_capability="12.0")
            target = root / "installed"
            with patch.object(trt_libs, "detect_nvidia_gpu_requirement", return_value=gpu):
                extracted = trt_libs.extract_required_tensorrt_libs(whl, target)
            self.assertEqual({path.name for path in extracted}, {*trt_libs.STANDARD_TRT_DLLS, "nvinfer_builder_resource_sm120_10.dll"})
            self.assertFalse((target / "nvinfer_builder_resource_sm75_10.dll").exists())

    def test_extract_only_installs_missing_libs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            whl = root / "trt.whl"
            target = root / "installed"
            target.mkdir()
            for name in trt_libs.STANDARD_TRT_DLLS:
                (target / name).write_bytes(f"existing:{name}".encode("utf-8"))
            with zipfile.ZipFile(whl, "w") as archive:
                for name in (*trt_libs.STANDARD_TRT_DLLS, "nvinfer_builder_resource_sm120_10.dll"):
                    archive.writestr(f"tensorrt_libs/{name}", f"wheel:{name}".encode("utf-8"))

            gpu = GpuRequirementResult(True, True, name="GPU", compute_capability="12.0")
            with patch.object(trt_libs, "detect_nvidia_gpu_requirement", return_value=gpu):
                extracted = trt_libs.extract_required_tensorrt_libs(whl, target)

            self.assertEqual([path.name for path in extracted], ["nvinfer_builder_resource_sm120_10.dll"])
            for name in trt_libs.STANDARD_TRT_DLLS:
                self.assertEqual((target / name).read_bytes(), f"existing:{name}".encode("utf-8"))

    def test_frozen_lib_dir_uses_internal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            exe = Path(raw) / "app.exe"
            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(exe)):
                self.assertEqual(trt_libs.tensorrt_lib_dir(), exe.parent / "_internal" / "tensorrt_libs")

    def test_download_uses_known_size_when_content_length_missing(self) -> None:
        class FakeResponse:
            headers = {}

            def __init__(self) -> None:
                self._chunks = [b"abc", b""]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

        progress: list[tuple[int, int]] = []
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "trt.whl"
            with (
                patch.object(trt_libs.urllib.request, "urlopen", return_value=FakeResponse()),
                patch.object(trt_libs, "_sha256", return_value=trt_libs.TENSORRT_CU12_LIBS_WHL_SHA256),
            ):
                trt_libs.download_tensorrt_wheel(target, progress=lambda done, total: progress.append((done, total)))

        self.assertEqual(progress[0], (0, trt_libs.TENSORRT_CU12_LIBS_WHL_SIZE_BYTES))
        self.assertEqual(progress[-1], (3, trt_libs.TENSORRT_CU12_LIBS_WHL_SIZE_BYTES))


if __name__ == "__main__":
    unittest.main()
