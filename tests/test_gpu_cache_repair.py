from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.gpu_cache_repair import (
    GPU_CACHE_TARGETS,
    cleanup_old_quarantines,
    discard_quarantine,
    quarantine_gpu_caches,
)


class GpuCacheRepairTests(unittest.TestCase):
    def test_quarantines_only_allowlisted_gpu_caches(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in GPU_CACHE_TARGETS:
                path = root / name
                if path.suffix or name.startswith("."):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("generated", encoding="utf-8")
                else:
                    path.mkdir(parents=True)
                    (path / "artifact.bin").write_bytes(b"gpu")
            preserved = {
                "ui_settings.json": b"settings",
                "library_index.sqlite": b"library",
                "clip_text_onnx/vocab.bin": b"resource",
                "thumbs/thumb.jpg": b"thumb",
            }
            for name, content in preserved.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            result = quarantine_gpu_caches(root, stamp="test")

            self.assertFalse(result.errors)
            self.assertEqual(set(result.moved), set(GPU_CACHE_TARGETS))
            self.assertIsNotNone(result.quarantine_dir)
            assert result.quarantine_dir is not None
            for name in GPU_CACHE_TARGETS:
                self.assertFalse((root / name).exists(), name)
                self.assertTrue((result.quarantine_dir / name).exists(), name)
            for name, content in preserved.items():
                self.assertEqual((root / name).read_bytes(), content)

            discard_quarantine(result.quarantine_dir)
            self.assertFalse((root / "gpu_cache_quarantine").exists())

    def test_empty_cache_still_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = quarantine_gpu_caches(Path(raw), stamp="empty")
        self.assertEqual(result.moved, ())
        self.assertEqual(result.errors, ())
        self.assertIsNone(result.quarantine_dir)

    def test_cleanup_removes_only_expired_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old = root / "gpu_cache_quarantine" / "old"
            recent = root / "gpu_cache_quarantine" / "recent"
            old.mkdir(parents=True)
            recent.mkdir(parents=True)
            (old / "cache.bin").write_bytes(b"old")
            (recent / "cache.bin").write_bytes(b"recent")
            old.touch()
            recent.touch()
            import os

            os.utime(old, (100.0, 100.0))
            os.utime(recent, (900.0, 900.0))

            removed = cleanup_old_quarantines(root, max_age_days=500.0 / 86400.0, now=1000.0)

            self.assertEqual(removed, ("old",))
            self.assertFalse(old.exists())
            self.assertTrue(recent.exists())


if __name__ == "__main__":
    unittest.main()
