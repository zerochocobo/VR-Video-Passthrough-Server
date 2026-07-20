from __future__ import annotations

import unittest

from pipeline import demosaic


class DemosaicChunkModelTests(unittest.TestCase):
    def test_restoration_model_points_to_chunk_onnx(self) -> None:
        self.assertEqual(demosaic.WINDOW, 8)
        self.assertEqual(
            demosaic.restoration_model_path().name,
            "vr_mosaic_restoration_chunk_model_v0.1.onnx",
        )


if __name__ == "__main__":
    unittest.main()
