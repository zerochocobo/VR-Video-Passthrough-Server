from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline import matting
from pipeline.matting import _should_enable_rvm_iobinding


class MattingRuntimePolicyTests(unittest.TestCase):
    def test_rvm_iobinding_enabled_for_cuda_only(self) -> None:
        self.assertTrue(_should_enable_rvm_iobinding(["CUDAExecutionProvider", "CPUExecutionProvider"]))

    def test_rvm_iobinding_disabled_when_tensorrt_is_active(self) -> None:
        self.assertFalse(
            _should_enable_rvm_iobinding(
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        )

    def test_rvm_iobinding_can_be_enabled_for_tensorrt_experiment(self) -> None:
        with patch.object(matting, "TRT_RVM_IOBINDING", True):
            self.assertTrue(
                _should_enable_rvm_iobinding(
                    ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            )

    def test_rvm_iobinding_disabled_without_cuda(self) -> None:
        self.assertFalse(_should_enable_rvm_iobinding(["CPUExecutionProvider"]))


if __name__ == "__main__":
    unittest.main()
