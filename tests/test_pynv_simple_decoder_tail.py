from __future__ import annotations

import unittest

from pipeline.pynv_io import PyNvSimpleDecoder


class _FakeDecoder:
    def __init__(self, n: int):
        self.n = n
        self.requests: list[int] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        self.requests.append(index)
        return index


def _make(n: int) -> tuple[PyNvSimpleDecoder, _FakeDecoder]:
    dec = PyNvSimpleDecoder.__new__(PyNvSimpleDecoder)
    fake = _FakeDecoder(n)
    dec._decoder = fake
    dec.bit_depth = 8
    dec._last_index = -1
    dec._skipped_forward = False
    return dec, fake


class PyNvSimpleDecoderTailTest(unittest.TestCase):
    def _run(self, dec: PyNvSimpleDecoder, indices: list[int]) -> None:
        from unittest.mock import patch

        with patch("pipeline.pynv_io.GpuNv12Frame.from_decoded_frame", side_effect=lambda f, w, h: f), \
                patch.object(PyNvSimpleDecoder, "info", create=True, new=type("I", (), {"width": 1, "height": 1})()):
            for i in indices:
                dec.frame_at(i)

    def test_skipping_run_never_requests_last_frame(self):
        # 59.94 -> 40fps stride: ..., n-3, n-1. PyNv crashes natively on n-1.
        dec, fake = _make(100)
        self._run(dec, [90, 91, 93, 94, 96, 97, 99])
        self.assertNotIn(99, fake.requests)
        self.assertEqual(fake.requests[-1], 98)

    def test_stays_clamped_after_an_earlier_skip(self):
        dec, fake = _make(100)
        self._run(dec, [10, 12, 13, 97, 98, 99])
        self.assertNotIn(99, fake.requests)

    def test_one_by_one_run_still_gets_last_frame(self):
        dec, fake = _make(100)
        self._run(dec, [95, 96, 97, 98, 99])
        self.assertEqual(fake.requests[-1], 99)

    def test_random_access_to_last_frame_is_untouched(self):
        dec, fake = _make(100)
        self._run(dec, [99])
        self.assertEqual(fake.requests, [99])


if __name__ == "__main__":
    unittest.main()
