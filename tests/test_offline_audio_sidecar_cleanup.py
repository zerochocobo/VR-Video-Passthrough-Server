from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

from tools.offline_passthrough import (
    _audio_sidecar_path,
    _cleanup_audio_sidecar,
    _is_prepass_child,
)
from utils.vr_naming import offline_passthrough_stem


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


class OfflineAudioSidecarOwnershipTests(unittest.TestCase):
    def test_prepass_child_is_detected_for_every_prepass_kind(self) -> None:
        for field in ("sam3_prepass_out", "ywes_prepass_out", "y26es_prepass_out", "y26br_prepass_out"):
            args = argparse.Namespace(
                sam3_prepass_out="",
                ywes_prepass_out="",
                y26es_prepass_out="",
                y26br_prepass_out="",
            )
            setattr(args, field, "masks.npz")
            self.assertTrue(_is_prepass_child(args), field)

    def test_full_run_is_not_a_prepass_child(self) -> None:
        args = argparse.Namespace(
            sam3_prepass_out="",
            ywes_prepass_out="",
            y26es_prepass_out="",
            y26br_prepass_out="",
        )
        self.assertFalse(_is_prepass_child(args))

    def test_batch_parent_and_prepass_child_do_not_share_a_sidecar(self) -> None:
        # Batch generation names the output exactly like the tool's own default,
        # so the prepass child used to overwrite and then delete the sidecar the
        # parent needs for its final audio mux.
        src = Path(tempfile.gettempdir()) / "movie.mp4"
        out = src.with_name(f"{offline_passthrough_stem(src.stem, 'green', 7680, 3840)}.mp4")
        sidecar = _audio_sidecar_path(out)

        self.assertIn("._audio", sidecar.name)
        self.assertIn(str(os.getpid()), sidecar.name)
        self.assertEqual(sidecar.suffix, ".aac")


if __name__ == "__main__":
    unittest.main()
