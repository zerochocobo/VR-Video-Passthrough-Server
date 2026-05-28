from __future__ import annotations

import unittest
from unittest.mock import patch

from utils import gpu_requirements


class PassthroughConcurrencyResolveTests(unittest.TestCase):
    def test_explicit_integer_bypasses_vram_probe(self) -> None:
        with patch.object(gpu_requirements, "detect_nvidia_total_vram_gib") as probe:
            self.assertEqual(gpu_requirements.resolve_passthrough_max_concurrent("2"), 2)
            probe.assert_not_called()

    def test_explicit_integer_is_clamped_to_one(self) -> None:
        with patch.object(gpu_requirements, "detect_nvidia_total_vram_gib") as probe:
            self.assertEqual(gpu_requirements.resolve_passthrough_max_concurrent("0"), 1)
            probe.assert_not_called()

    def test_auto_maps_vram_to_concurrency(self) -> None:
        cases = [
            (None, 1),
            (8.0, 1),
            (12.0, 2),
            (19.9, 2),
            (20.0, 3),
            (24.0, 3),
        ]
        for vram_gib, expected in cases:
            with self.subTest(vram_gib=vram_gib), patch.object(
                gpu_requirements,
                "detect_nvidia_total_vram_gib",
                return_value=vram_gib,
            ):
                self.assertEqual(gpu_requirements.resolve_passthrough_max_concurrent("auto"), expected)


if __name__ == "__main__":
    unittest.main()
