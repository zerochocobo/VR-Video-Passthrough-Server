from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui.services import process_helpers
from utils import runtime_dll_paths


class RuntimeDllPathsTests(unittest.TestCase):
    def test_dev_runtime_paths_prepend_tensorrt_and_cuda_bins(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            trt_libs = root / ".venv" / "Lib" / "site-packages" / "tensorrt_libs"
            cuda_bin = root / "cuda" / "bin"
            trt_libs.mkdir(parents=True)
            cuda_bin.mkdir(parents=True)
            env = {"PATH": str(root / "existing"), "CUDA_PATH": str(root / "cuda")}

            with patch.object(runtime_dll_paths.config, "ROOT", root):
                runtime_dll_paths.apply_runtime_dll_paths(env)

            parts = env["PATH"].split(os.pathsep)
            self.assertEqual(parts[:2], [str(trt_libs), str(cuda_bin)])
            self.assertEqual(parts[2], str(root / "existing"))

    def test_frozen_runtime_paths_prepend_internal_tensorrt_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            exe = root / "app.exe"
            internal = root / "_internal"
            trt_libs = internal / "tensorrt_libs"
            trt_libs.mkdir(parents=True)
            env = {"PATH": str(root / "existing")}

            with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(exe)):
                runtime_dll_paths.apply_runtime_dll_paths(env)

            parts = env["PATH"].split(os.pathsep)
            self.assertEqual(parts[:2], [str(trt_libs), str(internal)])
            self.assertIn(str(root / "existing"), parts)

    def test_base_environment_merges_extra_before_runtime_path_injection(self) -> None:
        seen: dict[str, str] = {}

        def fake_apply(env: dict[str, str]) -> dict[str, str]:
            seen.update(env)
            env["PATH"] = "patched"
            return env

        with patch.object(process_helpers, "apply_runtime_dll_paths", side_effect=fake_apply):
            env = process_helpers.base_environment({"PT_CUDNN_BIN": "custom-cudnn"})

        self.assertEqual(seen["PT_CUDNN_BIN"], "custom-cudnn")
        self.assertEqual(env["PATH"], "patched")


if __name__ == "__main__":
    unittest.main()
