from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.offline_passthrough import _cleanup_audio_sidecar


class OfflineAudioSidecarCleanupTests(unittest.TestCase):
    def test_cleanup_audio_sidecar_deletes_existing_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "movie._audio.aac"
            sidecar.write_bytes(b"partial aac")

            _cleanup_audio_sidecar(sidecar)

            self.assertFalse(sidecar.exists())

    def test_cleanup_audio_sidecar_ignores_missing_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _cleanup_audio_sidecar(Path(tmp) / "missing._audio.aac")
        _cleanup_audio_sidecar(None)


if __name__ == "__main__":
    unittest.main()
