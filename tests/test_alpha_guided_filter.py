import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("PT_RUN_CUDA_TESTS") == "1", "set PT_RUN_CUDA_TESTS=1 to run CUDA tests")
class AlphaGuidedFilterTests(unittest.TestCase):
    def test_constant_alpha_stays_constant(self):
        import cupy as cp

        from pipeline.alpha_guided_filter import fast_guided_filter_upsample

        alpha = cp.full((4, 8), 0.35, dtype=cp.float32)
        guide = cp.arange(16 * 32, dtype=cp.uint8).reshape(16, 32)

        out = fast_guided_filter_upsample(alpha, guide, radius=1, eps=0.0025)
        arr = cp.asnumpy(out)

        self.assertEqual(arr.shape, (16, 32))
        np.testing.assert_allclose(arr, np.full((16, 32), 0.35, dtype=np.float32), atol=2e-3)

    def test_half_scale_output_shape(self):
        import cupy as cp

        from pipeline.alpha_guided_filter import fast_guided_filter_upsample

        alpha = cp.linspace(0.0, 1.0, 4 * 8, dtype=cp.float32).reshape(4, 8)
        guide = cp.full((16, 32), 128, dtype=cp.uint8)

        out = fast_guided_filter_upsample(alpha, guide, radius=1, eps=0.0025, fullres_scale=0.5)

        self.assertEqual(tuple(out.shape), (8, 16))


if __name__ == "__main__":
    unittest.main()
