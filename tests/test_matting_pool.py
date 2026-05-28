"""Unit tests for the Matter instance pool.

Matter instantiation pulls in CUDA/ONNX which is far too heavy for unit tests,
so we monkeypatch the constructor used inside ``pipeline.matting`` to return
lightweight stub instances. This keeps the tests focused on the pool plumbing
(lazy create, cap, blocking, release, idempotency, get_matter slot 0 reuse).
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from pipeline import matting


class _StubMatter:
    """Cheap stand-in for pipeline.matting.Matter used in pool tests."""

    _counter = 0

    def __init__(self, *_, **__):
        type(self)._counter += 1
        self.id = type(self)._counter


def _reset_pool() -> None:
    """Clear pool state so each test starts from a known baseline."""
    with matting._pool_lock:
        matting._pool_all.clear()
        matting._pool_available.clear()
        matting._pool_max = 1
        matting._pool_warmup_runs = None


class MatterPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_pool()
        _StubMatter._counter = 0
        self._matter_patch = patch.object(matting, "Matter", _StubMatter)
        self._matter_patch.start()

    def tearDown(self) -> None:
        self._matter_patch.stop()
        _reset_pool()

    # ------------------------------------------------------------------
    # 1. Lazy create + reuse
    # ------------------------------------------------------------------
    def test_acquire_creates_lazily_then_reuses(self) -> None:
        matting.configure_matter_pool(2)
        self.assertEqual(len(matting._pool_all), 0)

        a = matting.acquire_matter()
        self.assertIsNotNone(a)
        self.assertEqual(len(matting._pool_all), 1)

        matting.release_matter(a)
        b = matting.acquire_matter()
        self.assertIs(b, a, "released instance must be reused")
        self.assertEqual(len(matting._pool_all), 1)
        matting.release_matter(b)

    # ------------------------------------------------------------------
    # 2. Cap = N concurrent holders
    # ------------------------------------------------------------------
    def test_pool_grows_up_to_cap(self) -> None:
        matting.configure_matter_pool(3)
        held = [matting.acquire_matter() for _ in range(3)]
        self.assertEqual(len({id(m) for m in held}), 3, "each acquire must give a distinct instance")
        self.assertEqual(len(matting._pool_all), 3)
        for m in held:
            matting.release_matter(m)

    # ------------------------------------------------------------------
    # 3. Non-blocking returns None when exhausted
    # ------------------------------------------------------------------
    def test_acquire_nonblocking_returns_none_when_full(self) -> None:
        matting.configure_matter_pool(1)
        held = matting.acquire_matter()
        try:
            self.assertIsNone(matting.acquire_matter(blocking=False))
        finally:
            matting.release_matter(held)

    # ------------------------------------------------------------------
    # 4. Blocking timeout returns None
    # ------------------------------------------------------------------
    def test_acquire_blocking_timeout_returns_none(self) -> None:
        matting.configure_matter_pool(1)
        held = matting.acquire_matter()
        try:
            t0 = time.monotonic()
            result = matting.acquire_matter(timeout=0.2)
            elapsed = time.monotonic() - t0
            self.assertIsNone(result)
            self.assertGreaterEqual(elapsed, 0.15)
            self.assertLess(elapsed, 1.0)
        finally:
            matting.release_matter(held)

    # ------------------------------------------------------------------
    # 5. Release is idempotent
    # ------------------------------------------------------------------
    def test_release_idempotent(self) -> None:
        matting.configure_matter_pool(1)
        m = matting.acquire_matter()
        matting.release_matter(m)
        # Second release must not duplicate the instance in _pool_available.
        matting.release_matter(m)
        with matting._pool_lock:
            self.assertEqual(matting._pool_available.count(m), 1)

    def test_release_ignores_unknown_instance(self) -> None:
        matting.configure_matter_pool(1)
        stranger = _StubMatter()
        matting.release_matter(stranger)  # must not raise
        with matting._pool_lock:
            self.assertNotIn(stranger, matting._pool_available)

    # ------------------------------------------------------------------
    # 6. get_matter slot 0 is acquirable by realtime path
    # ------------------------------------------------------------------
    def test_get_matter_slot_zero_is_acquirable_and_returnable(self) -> None:
        matting.configure_matter_pool(1)
        slot0 = matting.get_matter()
        self.assertIs(slot0, matting._pool_all[0])
        # Realtime acquire must be able to take slot 0 since it is parked in
        # _pool_available rather than being held by the utility caller.
        acquired = matting.acquire_matter()
        self.assertIs(acquired, slot0)
        matting.release_matter(acquired)
        # get_matter on a populated pool must keep returning slot 0.
        self.assertIs(matting.get_matter(), slot0)

    # ------------------------------------------------------------------
    # 7. Blocking acquire wakes up when a holder releases
    # ------------------------------------------------------------------
    def test_acquire_unblocks_on_release(self) -> None:
        matting.configure_matter_pool(1)
        held = matting.acquire_matter()
        result: dict = {}

        def worker():
            result["m"] = matting.acquire_matter(timeout=2.0)

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.1)
        matting.release_matter(held)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "blocked acquire should unblock after release")
        self.assertIs(result["m"], held)
        matting.release_matter(result["m"])


if __name__ == "__main__":
    unittest.main()
