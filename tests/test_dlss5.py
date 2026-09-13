"""DLSS 5 Neural Rendering: settings, gating and the [DLSS5] channel wiring.

None of this touches the runtime DLLs or a GPU: availability and the prepared
CUDA context are the two facts the rest is built on, so both are patched and the
logic around them is what gets exercised.
"""
from __future__ import annotations

import os
import site
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config
import dlna.content_directory as cds
import utils.dlss5 as dlss5
from http_app import routes_media
from pipeline.pynv_stream import SEEK_FRAME_MODES
from utils.dlss5 import DLSS5_RANGES, DLSS5Settings, source_block_reason_dlss5
from utils.vr_naming import DLSS5_PREFIX, dlss5_output_stem, live_passthrough_title


# Qt bootstrap for the dialog tests below, as in tests/test_ui_smoke.py:
# PySide6 ships its DLLs beside the package and offscreen keeps this headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_DLL_HANDLES = []
if hasattr(os, "add_dll_directory"):
    for _site_dir in site.getsitepackages():
        _base = Path(_site_dir)
        for _dll_dir in (_base / "PySide6", _base / "shiboken6"):
            if _dll_dir.exists():
                _DLL_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
        _plugins = _base / "PySide6" / "plugins"
        if (_plugins / "platforms").exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_plugins / "platforms"))
        if _plugins.exists():
            os.environ.setdefault("QT_PLUGIN_PATH", str(_plugins))


def _available(value: bool = True):
    return patch.object(dlss5, "is_dlss5_available", return_value=value)


def _context_ready(value: bool = True):
    return patch.object(dlss5, "cuda_context_ready", return_value=value)


class DLSS5SettingsTests(unittest.TestCase):
    def test_from_config_reads_every_field(self) -> None:
        settings = DLSS5Settings.from_config()
        self.assertEqual(settings.style, config.DLSS5_STYLE)
        self.assertAlmostEqual(settings.intensity, config.DLSS5_INTENSITY)
        self.assertAlmostEqual(settings.local_tone, config.DLSS5_LOCAL_TONE)
        self.assertAlmostEqual(settings.local_structure, config.DLSS5_LOCAL_STRUCTURE)
        self.assertAlmostEqual(settings.skin_structure, config.DLSS5_SKIN_STRUCTURE)
        self.assertAlmostEqual(settings.color_strength, config.DLSS5_COLOR_STRENGTH)
        self.assertAlmostEqual(settings.tone_preservation, config.DLSS5_TONE_PRESERVATION)
        self.assertAlmostEqual(settings.face_skin_protection, config.DLSS5_FACE_SKIN_PROTECTION)
        self.assertAlmostEqual(settings.grain_preservation, config.DLSS5_GRAIN_PRESERVATION)
        self.assertEqual(settings.nr_passes, config.DLSS5_NR_PASSES)
        self.assertAlmostEqual(settings.shimmer_suppression, config.DLSS5_SHIMMER_SUPPRESSION)
        self.assertEqual(settings.prefer_nvof, config.DLSS5_PREFER_NVOF)

    def test_to_params_maps_onto_the_abi_field_names(self) -> None:
        """The struct renames three controls; a silent mismatch here is a wrong
        picture nobody can trace back to a settings dialog."""
        settings = DLSS5Settings(
            style=2, intensity=1.25, local_tone=0.5, local_structure=1.75,
            skin_structure=-1.0, auto_mask=True, color_strength=0.25,
            tone_preservation=0.75, face_skin_protection=0.4,
            grain_preservation=0.1, nr_passes=3, shimmer_suppression=0.7,
            prefer_nvof=True,
        )
        params = settings.to_params(reset=True)
        self.assertEqual(params.style, 2)
        self.assertAlmostEqual(params.intensity, 1.25, places=5)
        self.assertAlmostEqual(params.tone, 0.5, places=5)            # local_tone
        self.assertAlmostEqual(params.structure, 1.75, places=5)      # local_structure
        self.assertAlmostEqual(params.skin, -1.0, places=5)           # skin_structure
        self.assertEqual(params.automask, 1)
        self.assertEqual(params.reset, 1)
        self.assertAlmostEqual(params.color_strength, 0.25, places=5)
        self.assertAlmostEqual(params.tone_preservation, 0.75, places=5)
        self.assertAlmostEqual(params.face_skin_protection, 0.4, places=5)
        self.assertAlmostEqual(params.grain_preservation, 0.1, places=5)
        self.assertEqual(params.nr_passes, 3)
        self.assertAlmostEqual(params.shimmer_suppression, 0.7, places=5)
        self.assertEqual(params.prefer_nvof, 1)
        self.assertEqual(settings.to_params(reset=False).reset, 0)

    def test_defaults_are_the_runtime_neutral_values_not_zero(self) -> None:
        """Most neutral values are 1.0, and skin structure's is -1.0.

        A zeroed block reads as "no colour work at all" plus a deliberate skin
        adjustment, which is not what an untouched install should ask for.
        """
        settings = DLSS5Settings()
        self.assertAlmostEqual(settings.intensity, 1.0)
        self.assertAlmostEqual(settings.local_tone, 1.0)
        self.assertAlmostEqual(settings.local_structure, 1.0)
        self.assertAlmostEqual(settings.skin_structure, -1.0)
        self.assertAlmostEqual(settings.color_strength, 1.0)
        for key, (low, high) in DLSS5_RANGES.items():
            with self.subTest(control=key):
                self.assertLessEqual(low, getattr(settings, key))
                self.assertLessEqual(getattr(settings, key), high)


class DLSS5SourceGateTests(unittest.TestCase):
    def test_offline_only_checks_geometry_and_the_runtime(self) -> None:
        with _available(True):
            self.assertIsNone(source_block_reason_dlss5(3840, 2160))
            self.assertEqual(source_block_reason_dlss5(0, 1080), "invalid_dimensions")
            self.assertEqual(source_block_reason_dlss5(1921, 1080), "odd_dimensions")
            # Offline has no resolution ceiling: 8K is slow, not refused.
            self.assertIsNone(source_block_reason_dlss5(7680, 3840))
        with _available(False):
            self.assertEqual(source_block_reason_dlss5(3840, 2160), "dlss5_runtime_missing")

    def test_realtime_adds_the_playback_speed_limits(self) -> None:
        with _available(True), _context_ready(True), patch.object(config, "DLSS5_REALTIME_ENABLED", True):
            self.assertIsNone(source_block_reason_dlss5(3840, 1920, realtime=True))
            self.assertIsNone(source_block_reason_dlss5(3840, 2160, realtime=True))
            # 6K and 8K run at 14 and 8 fps before decode and NVENC take their
            # share, so they are offline-only.
            self.assertEqual(
                source_block_reason_dlss5(5760, 2880, realtime=True), "dlss5_source_too_large"
            )
            self.assertEqual(
                source_block_reason_dlss5(7680, 3840, realtime=True), "dlss5_source_too_large"
            )
            self.assertEqual(
                source_block_reason_dlss5(320, 180, realtime=True), "dlss5_source_too_small"
            )
            # The kernel writes 8-bit BT.709 NV12; a 10-bit source would be
            # truncated rather than enhanced.
            self.assertEqual(
                source_block_reason_dlss5(3840, 1920, is_10bit=True, realtime=True),
                "dlss5_10bit_unsupported",
            )

    def test_the_source_frame_rate_decides_inside_the_ceiling(self) -> None:
        """4K at 24fps plays and the same 4K at 60fps does not, because the
        channel is pulled at playback speed. Measured end to end (decode + NR +
        NVENC) the chain holds ~230 Mpixel/s: 31.9fps at 3840x1920 and 28.3fps
        at 3840x2160, so a 60fps 4K source cannot be offered."""
        with _available(True), _context_ready(True), patch.object(config, "DLSS5_REALTIME_ENABLED", True):
            self.assertIsNone(source_block_reason_dlss5(3840, 2160, fps=23.976, realtime=True))
            self.assertEqual(
                source_block_reason_dlss5(3840, 2160, fps=59.94, realtime=True),
                "dlss5_source_too_fast",
            )
            self.assertEqual(
                source_block_reason_dlss5(3840, 1920, fps=59.94, realtime=True),
                "dlss5_source_too_fast",
            )
            # 1216x2160 measured 78fps end to end, so 60fps has room there.
            self.assertIsNone(source_block_reason_dlss5(1216, 2160, fps=59.94, realtime=True))
            # 30fps 4K VR is where the budget bites, and it is the shape the
            # library holds: 4096x2048 at 30fps asks 252 Mpixel/s, just over
            # the measured rate, and is offered anyway - see the note on
            # DLSS5_REALTIME_PIXEL_RATE. At 200 Mpixel/s no 4K VR size passed
            # at 30fps at all, which is what left the channel empty.
            self.assertIsNone(source_block_reason_dlss5(4096, 2048, fps=30.0, realtime=True))
            self.assertIsNone(source_block_reason_dlss5(3840, 1920, fps=30.0, realtime=True))
            self.assertIsNone(source_block_reason_dlss5(3840, 2160, fps=30.0, realtime=True))
            # Half speed is not a few percent short: 60fps at that size stays out.
            self.assertEqual(
                source_block_reason_dlss5(4096, 2048, fps=59.94, realtime=True),
                "dlss5_source_too_fast",
            )
            # 4096 is the width ceiling itself, so the limit must not be
            # exclusive - the library's own titles sit exactly on it.
            self.assertIsNone(source_block_reason_dlss5(4096, 2048, fps=24.0, realtime=True))
            # An unprobed frame rate is not a refusal: the resolution ceiling
            # still applies and the route reads its own metadata anyway.
            self.assertIsNone(source_block_reason_dlss5(3840, 2160, fps=0.0, realtime=True))
        with _available(True), _context_ready(True), \
             patch.object(config, "DLSS5_REALTIME_ENABLED", True), \
             patch.object(config, "DLSS5_REALTIME_PIXEL_RATE", 0):
            # The budget is one env var, and zero turns the check off entirely
            # for a GPU faster than the one it was measured on.
            self.assertIsNone(source_block_reason_dlss5(3840, 2160, fps=59.94, realtime=True))

    def test_realtime_needs_the_switch_and_the_prepared_context(self) -> None:
        with _available(True), _context_ready(True), patch.object(config, "DLSS5_REALTIME_ENABLED", False):
            self.assertEqual(
                source_block_reason_dlss5(3840, 1920, realtime=True), "dlss5_realtime_disabled"
            )
        with _available(True), _context_ready(False), patch.object(config, "DLSS5_REALTIME_ENABLED", True):
            self.assertEqual(
                source_block_reason_dlss5(3840, 1920, realtime=True), "dlss5_cuda_context_unprepared"
            )


class DLSS5CudaContextTests(unittest.TestCase):
    """The one process-wide precondition the whole channel rests on.

    NR reaches the picture through CUDA interop and refuses unless the device's
    primary context carries the blocking-sync flag. That flag can only be
    claimed while the context is inactive, so it is claimed at startup - once
    cupy or PyNvVideoCodec has created the context, the answer for the rest of
    the process is no.
    """

    def setUp(self) -> None:
        self._saved = dlss5._cuda_context_prepared
        dlss5._cuda_context_prepared = None

    def tearDown(self) -> None:
        dlss5._cuda_context_prepared = self._saved

    def _fake_cuda(self, *, get_state=0, set_flags=0, flags=0):
        cuda = SimpleNamespace()
        cuda.cuInit = lambda _: 0

        def _get_state(_device, flags_ref, active_ref):
            flags_ref._obj.value = flags
            active_ref._obj.value = 1
            return get_state

        cuda.cuDevicePrimaryCtxGetState = _get_state
        cuda.cuDevicePrimaryCtxSetFlags = lambda _device, _flags: set_flags
        return cuda

    def test_claims_the_flag_once_and_remembers_the_answer(self) -> None:
        calls: list[int] = []
        cuda = self._fake_cuda()
        cuda.cuDevicePrimaryCtxSetFlags = lambda device, flags: (calls.append(flags), 0)[1]
        with patch("ctypes.WinDLL", return_value=cuda):
            self.assertTrue(dlss5.prepare_cuda_context_for_dlss5())
            self.assertTrue(dlss5.prepare_cuda_context_for_dlss5())
        self.assertEqual(calls, [dlss5._CU_CTX_SCHED_BLOCKING_SYNC])
        self.assertTrue(dlss5.cuda_context_ready())

    def test_an_already_flagged_context_needs_no_call(self) -> None:
        cuda = self._fake_cuda(flags=dlss5._CU_CTX_SCHED_BLOCKING_SYNC)
        cuda.cuDevicePrimaryCtxSetFlags = lambda device, flags: 1 / 0  # must not run
        with patch("ctypes.WinDLL", return_value=cuda):
            self.assertTrue(dlss5.prepare_cuda_context_for_dlss5())

    def test_a_live_context_reports_failure_instead_of_raising(self) -> None:
        # 708 is CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: something got there first.
        cuda = self._fake_cuda(set_flags=708)
        with patch("ctypes.WinDLL", return_value=cuda):
            self.assertFalse(dlss5.prepare_cuda_context_for_dlss5())
        self.assertFalse(dlss5.cuda_context_ready())

    def test_a_machine_without_a_driver_reports_failure(self) -> None:
        with patch("ctypes.WinDLL", side_effect=OSError("nvcuda.dll not found")):
            self.assertFalse(dlss5.prepare_cuda_context_for_dlss5())


class DLSS5NamingTests(unittest.TestCase):
    def test_output_stem_carries_the_marker_once(self) -> None:
        self.assertEqual(dlss5_output_stem("clip"), f"{DLSS5_PREFIX}clip")
        self.assertEqual(dlss5_output_stem(f"{DLSS5_PREFIX}clip"), f"{DLSS5_PREFIX}clip")

    def test_live_title_keeps_the_source_geometry_markers(self) -> None:
        """NR is 1x, so the output is still a 2:1 SBS VR picture and the title
        has to keep saying so or the player will not enter VR180."""
        title = live_passthrough_title("clip", "dlss5", 3840, 1920)
        self.assertTrue(title.startswith(DLSS5_PREFIX))
        self.assertTrue(title.endswith("_live"))
        self.assertIn("180", title)


class DLSS5SeekRouteTests(unittest.TestCase):
    def test_the_mode_is_in_the_one_authoritative_set(self) -> None:
        self.assertIn("dlss5", SEEK_FRAME_MODES)
        self.assertIn("dlss5", routes_media._SEEK_ROUTE_MODES)

    def test_the_route_name_carries_the_mode(self) -> None:
        """A player re-issuing the URL for its range requests may drop the
        query, and a dropped mode used to fall back and play another picture."""
        self.assertEqual(
            routes_media._split_seek_route_name("clip.dlss5.seek.mp4"),
            ("clip", "mp4", "dlss5"),
        )
        self.assertEqual(
            routes_media._split_seek_route_name("clip.dlss5.seek.ts"),
            ("clip", "mpegts", "dlss5"),
        )

    def test_the_path_borne_mode_beats_the_query(self) -> None:
        with patch.object(routes_media, "PASSTHROUGH_OUTPUT_MODE", "green,dlss5"):
            self.assertEqual(routes_media._seek_output_mode("green", "dlss5"), "dlss5")
            self.assertEqual(routes_media._seek_output_mode("dlss5", None), "dlss5")
            # Nothing asked for: the cheap mode answers, not the GPU stage.
            self.assertEqual(routes_media._seek_output_mode(None, None), "green")
        with patch.object(routes_media, "PASSTHROUGH_OUTPUT_MODE", "dlss5"):
            self.assertEqual(routes_media._seek_output_mode(None, None), "dlss5")

    def test_the_virtual_file_keeps_the_source_geometry_and_budget(self) -> None:
        """NR does not resize, so unlike SuperRes nothing about the MP4 shell
        or the per-frame budget may be scaled for it."""
        info = SimpleNamespace(width=3840, height=1920)
        template = SimpleNamespace(width=3840, height=1920)
        with patch.object(routes_media, "DECODE_MAX_SIDE", 0):
            self.assertEqual(routes_media._vmp4_slot_output_size(info, "dlss5"), (3840, 1920))
        # The stage works from the source size and never sees DECODE_MAX_SIDE,
        # so the shell must not inherit the decode cap either - a shell that
        # declares 2048 for a 3840 encode is a stream no player can read.
        with patch.object(routes_media, "DECODE_MAX_SIDE", 2048):
            self.assertEqual(routes_media._vmp4_slot_output_size(info, "dlss5"), (3840, 1920))
            self.assertEqual(routes_media._vmp4_slot_output_size(info, "green"), (2048, 1024))
        self.assertEqual(
            routes_media._vmp4_frames_frame_budget(template, "dlss5"),
            routes_media._vmp4_frames_frame_budget(template, "green"),
        )


class DLSS5ContentDirectoryTests(unittest.TestCase):
    def _enabled(self, **overrides):
        defaults = {"DLSS5_ENABLED": True}
        defaults.update(overrides)
        return [patch.object(cds, key, value) for key, value in defaults.items()]

    def test_the_title_is_the_bare_marker(self) -> None:
        """No parameters in the name: Skybox caches filenames, and a title that
        moves with a slider never comes back."""
        title = cds._passthrough_seek_title(Path("clip.mp4"), "dlss5", 3840, 1920)
        self.assertTrue(title.startswith("[DLSS5]"))
        for token in ("intensity", "0.5", "1.00", "passes"):
            self.assertNotIn(token, title.lower())

    def test_object_id_prefixes_do_not_collide(self) -> None:
        prefixes = {
            cds._passthrough_live_prefix("dlss5"),
            cds._passthrough_live_item_prefix("dlss5"),
            cds._passthrough_seek_item_prefix("dlss5"),
            cds._passthrough_live_prefix("superres"),
            cds._passthrough_live_item_prefix("superres"),
            cds._passthrough_seek_item_prefix("superres"),
            cds._passthrough_live_prefix("green"),
            cds._passthrough_seek_item_prefix("green"),
        }
        self.assertEqual(len(prefixes), 8)

    def test_ids_round_trip_back_to_the_dlss5_mode(self) -> None:
        """The item prefix starts with the directory prefix, so the dispatch
        order matters: pldn_item_ must be tested before pldn_."""
        rel = "clip.mp4"
        path = Path("clip.mp4")
        with patch.object(cds.MEDIA_LIBRARY, "key_to_path", return_value=path), \
             patch.object(cds.MEDIA_LIBRARY, "contains", return_value=True), \
             patch.object(Path, "is_file", return_value=True):
            self.assertEqual(cds._id_to_live(f"{cds.DLSS5_LIVE_PREFIX}{rel}"), (path, "dlss5"))
            self.assertEqual(cds._id_to_live(f"{cds.DLSS5_LIVE_ITEM_PREFIX}{rel}"), (path, "dlss5"))
            self.assertEqual(cds._id_to_seek(f"{cds.DLSS5_SEEK_ITEM_PREFIX}{rel}"), (path, "dlss5"))
            # And the superres ids still answer superres.
            self.assertEqual(cds._id_to_live(f"{cds.SUPERRES_LIVE_ITEM_PREFIX}{rel}"), (path, "superres"))
            self.assertEqual(cds._id_to_seek(f"{cds.SUPERRES_SEEK_ITEM_PREFIX}{rel}"), (path, "superres"))

    def test_dlss5_is_always_seekable(self) -> None:
        """1x output shares the source's geometry, so the source MP4 shell
        describes it; there is no enlarging target to disqualify it."""
        self.assertTrue(cds._seek_supported_mode("dlss5"))
        self.assertIn("dlss5", cds._seek_supported_modes())

    def test_the_advertised_resolution_is_the_source_resolution(self) -> None:
        self.assertEqual(cds._passthrough_resolution(3840, 1920, "dlss5"), "3840x1920")
        self.assertEqual(cds._passthrough_resolution(3840, 1920, "dlss5", seekable=True), "3840x1920")

    def test_the_route_suffix_carries_the_mode(self) -> None:
        self.assertEqual(
            cds._seek_passthrough_route_suffix("mp4", "dlss5"), ".dlss5.seek.mp4"
        )
        self.assertIn("mode=dlss5", cds._passthrough_seek_query("dlss5"))

    def test_a_build_without_the_runtime_lists_nothing(self) -> None:
        """A clean clone has no runtime/ at all. An entry that cannot play is
        worse than no entry, so the whole channel disappears."""
        with patch.object(cds, "DLSS5_ENABLED", True), _available(False):
            self.assertFalse(cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 1920))
        with patch.object(cds, "DLSS5_ENABLED", False), _available(True), _context_ready(True), \
             patch.object(config, "DLSS5_REALTIME_ENABLED", True):
            self.assertFalse(cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 1920))

    def test_only_sources_the_channel_can_sustain_are_listed(self) -> None:
        with patch.object(cds, "DLSS5_ENABLED", True), _available(True), _context_ready(True), \
             patch.object(config, "DLSS5_REALTIME_ENABLED", True):
            self.assertTrue(cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 1920))
            self.assertTrue(cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 2160))
            self.assertFalse(cds._dlss5_dlna_enabled(Path("clip.mp4"), 7680, 3840))
            # The indexed frame rate is read too, so a 60fps 4K title that the
            # channel cannot sustain is never listed in the first place.
            def child(fps: float, pix_fmt: str = "yuv420p"):
                return SimpleNamespace(video=SimpleNamespace(fps=fps, pix_fmt=pix_fmt))

            self.assertTrue(cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 2160, child(23.976)))
            self.assertFalse(cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 2160, child(59.94)))
            # A 10-bit source would be truncated to 8-bit rather than enhanced.
            self.assertFalse(
                cds._dlss5_dlna_enabled(Path("clip.mp4"), 3840, 1920, child(24.0, "yuv420p10le"))
            )

    def test_the_mode_survives_the_configured_mode_list(self) -> None:
        with patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "green,dlss5"):
            self.assertEqual(cds._passthrough_modes(), ("green", "dlss5"))
        with patch.object(routes_media, "PASSTHROUGH_OUTPUT_MODE", "green,dlss5"):
            self.assertEqual(routes_media._configured_passthrough_modes(), ("green", "dlss5"))


class DLSS5StageSynchronisationTests(unittest.TestCase):
    """NR runs on D3D12 and reaches these buffers only through CUDA interop, so
    it carries no ordering against the stream the conversion kernels use. Until
    the evaluate was bracketed by a synchronise on each side - the same pair
    RTX VSR uses around its own evaluate - the encoder picked up half-written
    surfaces: grey and cyan blocks, a bright line along the top edge, and a
    picture that flickered frame to frame.
    """

    def _fake_stage(self, calls: list[str]):
        from pipeline.dlss5_stage import DLSS5Stage

        class _Stream:
            ptr = 0xBEEF

            def synchronize(self) -> None:
                calls.append("sync")

        stream = _Stream()

        class _Cuda:
            @staticmethod
            def get_current_stream():
                return stream

        class _Cp:
            cuda = _Cuda
            uint64 = staticmethod(lambda value: value)
            int32 = staticmethod(lambda value: value)

        def _buffer(ptr: int):
            return SimpleNamespace(data=SimpleNamespace(ptr=ptr))

        stage = object.__new__(DLSS5Stage)
        stage.width, stage.height = 64, 32
        stage.settings = DLSS5Settings.from_config()
        stage._cp = _Cp
        stage._rgb_in = _buffer(0x1000)
        stage._rgb_out = _buffer(0x2000)
        stage._k_to_rgb = lambda *args, **kwargs: calls.append("to_rgb")
        stage._k_to_nv12 = lambda *args, **kwargs: calls.append("to_nv12")
        return stage

    def test_the_evaluate_is_bracketed_by_a_stream_synchronise(self) -> None:
        import pipeline.dlss5_stage as stage_module

        calls: list[str] = []
        stage = self._fake_stage(calls)
        with patch.object(stage_module.BRIDGE, "process_cuda_pointers") as evaluate:
            evaluate.side_effect = lambda *args, **kwargs: calls.append("nr")
            stage.process_nv12(1, 2, 64, 64, 3, 4, 64, 64, reset=True)
        self.assertEqual(calls, ["to_rgb", "sync", "nr", "sync", "to_nv12"])
        # The evaluate is handed the same stream it is synchronised against.
        self.assertEqual(evaluate.call_args.kwargs["stream_ptr"], 0xBEEF)

    def test_reset_reaches_the_render_parameters(self) -> None:
        """reset tells NR its temporal history no longer describes the picture;
        a stage is built per session, so its first frame is always a cut."""
        import pipeline.dlss5_stage as stage_module

        calls: list[str] = []
        stage = self._fake_stage(calls)
        with patch.object(stage_module.BRIDGE, "process_cuda_pointers") as evaluate:
            stage.process_nv12(1, 2, 64, 64, 3, 4, 64, 64, reset=True)
            self.assertEqual(evaluate.call_args.args[4].reset, 1)
            stage.process_nv12(1, 2, 64, 64, 3, 4, 64, 64, reset=False)
            self.assertEqual(evaluate.call_args.args[4].reset, 0)


class DLSS5ProbeMetricTests(unittest.TestCase):
    """tools/dlss5_probe.py answers "is the enhancement even there" with
    numbers, so the numbers have to mean what the legend says they mean."""

    def _pair(self):
        import numpy as np

        rng = np.random.default_rng(0)
        base = (rng.random((48, 48, 3)).astype("float32") * 0.5) + 0.25
        blurred = base.copy()
        blurred[1:-1, 1:-1] = (
            base[:-2, 1:-1] + base[2:, 1:-1] + base[1:-1, :-2] + base[1:-1, 2:] + base[1:-1, 1:-1]
        ) / 5.0
        return base, blurred

    def test_an_unchanged_frame_reads_as_no_change(self) -> None:
        from tools.dlss5_probe import _measure

        base, _ = self._pair()
        metrics = _measure(base, base.copy())
        self.assertEqual(metrics["mad"], 0.0)
        self.assertAlmostEqual(metrics["sharpness"], 1.0, places=6)
        self.assertAlmostEqual(metrics["luma"], 0.0, places=6)

    def test_softening_and_sharpening_move_sharpness_the_right_way(self) -> None:
        import numpy as np

        from tools.dlss5_probe import _measure

        base, blurred = self._pair()
        self.assertLess(_measure(base, blurred)["sharpness"], 1.0)
        sharpened = np.clip(base + (base - blurred) * 1.5, 0.0, 1.0)
        self.assertGreater(_measure(base, sharpened)["sharpness"], 1.0)

    def test_luma_reports_the_direction_a_picture_moved(self) -> None:
        import numpy as np

        from tools.dlss5_probe import _measure

        base, _ = self._pair()
        darker = _measure(base, np.clip(base - 0.04, 0.0, 1.0))
        self.assertAlmostEqual(darker["luma"], -0.04 * 255.0, places=3)
        self.assertAlmostEqual(darker["mad"], 0.04 * 255.0, places=3)
        self.assertGreater(_measure(base, np.clip(base + 0.04, 0.0, 1.0))["luma"], 0.0)

    def test_the_sweep_covers_the_controls_a_flat_result_would_blame(self) -> None:
        from tools.dlss5_probe import SWEEP

        labels = " ".join(label for label, _ in SWEEP)
        for expected in ("current", "shimmer", "detail-only", "intensity", "passes", "structure"):
            self.assertIn(expected, labels)
        # The first row must be the configured settings, so every run states
        # what the server would actually do alongside the alternatives.
        self.assertEqual(SWEEP[0][1], {})


class DLSS5ColourKernelTests(unittest.TestCase):
    """DLSS5 writes its own NV12<->RGB kernels because NR wants float32 RGB, but
    the colour maths must be the maths every other stage uses.

    It was not: the decode direction carried the BT.601 integer set under a
    comment that said BT.709, while the encode direction was genuinely 709. A
    709 source - every HD and 4K title - was therefore decoded on the wrong
    matrix, enhanced in those colours, and written back on the right one.
    """

    # BT.709 limited range, as the kernels apply it: YUV on (Y-16, U-128, V-128),
    # RGB before the +16/+128 offsets.
    YUV_TO_RGB = (
        (1.16438356, 0.0, 1.79274107),
        (1.16438356, -0.21324861, -0.53290933),
        (1.16438356, 2.11240179, 0.0),
    )
    RGB_TO_YUV = (
        (0.182586, 0.614231, 0.062007),
        (-0.100644, -0.338572, 0.439216),
        (0.439216, -0.398942, -0.040274),
    )
    # The set that was there instead, on the decode side only.
    BT601_COEFFICIENTS = ("298.082f", "408.583f", "100.291f", "208.120f", "516.412f")

    def test_the_two_directions_are_inverses(self) -> None:
        """A round trip through NR must not move the colour by itself."""
        import numpy as np

        product = np.array(self.YUV_TO_RGB) @ np.array(self.RGB_TO_YUV)
        np.testing.assert_allclose(product, np.eye(3), atol=2e-5)

    def test_a_601_decode_against_a_709_encode_is_not_a_round_trip(self) -> None:
        """What the bug cost, in numbers: mid-grey survives (the luma row is the
        same either way) and saturated colour does not."""
        import numpy as np

        bt601_to_rgb = np.array([
            [1.16438356, 0.0, 1.59602679],
            [1.16438356, -0.39176229, -0.81296765],
            [1.16438356, 2.01723214, 0.0],
        ])
        drift = bt601_to_rgb @ np.array(self.RGB_TO_YUV)
        self.assertGreater(np.abs(drift - np.eye(3)).max(), 0.1)

    def test_both_kernels_carry_the_shared_coefficients(self) -> None:
        from offline.two_dvr_pynv import _NV12_RGB_KERNELS
        from pipeline.dlss5_stage import _DLSS5_CUDA_KERNELS

        shared = (
            "1.16438356f", "1.79274107f", "0.21324861f", "0.53290933f", "2.11240179f",
            "0.182586f", "0.614231f", "0.062007f",
            "0.100644f", "0.338572f", "0.439216f", "0.398942f", "0.040274f",
        )
        for coefficient in shared:
            self.assertIn(coefficient, _NV12_RGB_KERNELS, coefficient)
            self.assertIn(coefficient, _DLSS5_CUDA_KERNELS, coefficient)
        for coefficient in self.BT601_COEFFICIENTS:
            self.assertNotIn(coefficient, _DLSS5_CUDA_KERNELS, coefficient)

    def test_chroma_is_averaged_not_point_sampled(self) -> None:
        """4:2:0 takes one chroma sample per 2x2 block. Reading only the corner
        pixel aliases every colour edge - and NR has just sharpened them."""
        from pipeline.dlss5_stage import _DLSS5_CUDA_KERNELS

        _head, _, tail = _DLSS5_CUDA_KERNELS.partition("rgb_f32_to_nv12")
        self.assertIn("for (int dy = 0; dy < 2; dy++)", tail)
        self.assertIn("for (int dx = 0; dx < 2; dx++)", tail)
        self.assertIn("255.0f / 4.0f", tail)


class DLSS5SeekBudgetTests(unittest.TestCase):
    """The virtual file's byte budget, which is also its bitrate.

    The shared multiplier says how much more than its source an output mode
    needs, and DLSS5 is the one mode whose output is 1:1 with the source, so it
    is the one mode that does not need more.
    """

    def test_only_dlss5_drops_the_passthrough_multiplier(self) -> None:
        self.assertEqual(config.seek_source_multiplier("dlss5"), config.DLSS5_SEEK_SOURCE_MULTIPLIER)
        for mode in ("green", "alpha", "superres", "", "DLSS5X"):
            self.assertEqual(
                config.seek_source_multiplier(mode), config.PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER
            )
        # Case and padding come from the route and the DIDL layer alike.
        self.assertEqual(config.seek_source_multiplier(" DLSS5 "), config.DLSS5_SEEK_SOURCE_MULTIPLIER)

    def test_a_4k_vr_source_is_not_laid_out_at_twice_its_rate(self) -> None:
        duration, width, height, fps = 3600.0, 4096, 2048, 30.0
        source_size = int(20_000_000 * duration / 8)      # a 20 Mbps 4K VR title
        green = config.seek_budget_scale(source_size, duration, width, height, fps, "green")
        dlss5 = config.seek_budget_scale(source_size, duration, width, height, fps, "dlss5")
        self.assertGreater(green, dlss5)
        # 1/0.75: the VMP4 filler is all that is left above the source's own
        # level, and the filler bytes are really sent.
        self.assertAlmostEqual(dlss5, 1.333, places=2)
        self.assertAlmostEqual(green, 2.667, places=2)

    def test_the_bits_per_pixel_floor_still_applies(self) -> None:
        """A cheaply encoded source is still raised to the floor that keeps the
        picture out of blocking - the multiplier is a ceiling, not a target."""
        duration, width, height, fps = 3600.0, 4096, 2048, 30.0
        source_size = int(8_000_000 * duration / 8)
        floor_bps = config.PASSTHROUGH_SEEK_VMP4_FRAMES_MIN_BITS_PER_PIXEL * width * height * fps
        scale = config.seek_budget_scale(source_size, duration, width, height, fps, "dlss5")
        self.assertAlmostEqual(
            source_size * 8 / duration * scale,
            floor_bps / config.PASSTHROUGH_SEEK_VMP4_FRAMES_RATE_HEADROOM,
            delta=100_000,
        )

    def test_didl_and_http_are_handed_the_same_mode(self) -> None:
        """They compute the declared size independently and must land on the
        same byte count; a mode that reaches one side only would split them."""
        import inspect

        from http_app import routes_media

        didl = inspect.getsource(cds._seek_declared_size)
        self.assertIn("mode,", didl)
        http = inspect.getsource(routes_media._vmp4_frames_source_budget)
        self.assertIn("output_mode,", http)
        layout_key = inspect.getsource(routes_media._vmp4_frames_layout_key)
        self.assertIn("output_mode,", layout_key)

    def test_the_declared_size_follows_the_mode(self) -> None:
        duration, width, height, fps = 3600.0, 4096, 2048, 30.0
        source_size = int(20_000_000 * duration / 8)
        green = config.seek_declared_total_bytes(source_size, duration, width, height, fps, "green")
        dlss5 = config.seek_declared_total_bytes(source_size, duration, width, height, fps, "dlss5")
        self.assertGreater(green, dlss5)
        self.assertGreaterEqual(dlss5, source_size)
        # No mode is the passthrough behaviour, unchanged.
        self.assertEqual(
            config.seek_declared_total_bytes(source_size, duration, width, height, fps), green
        )


class DLSS5WarmupTests(unittest.TestCase):
    def test_warmup_reports_the_runtime_and_never_raises(self) -> None:
        with patch.object(dlss5.BRIDGE, "initialize", return_value=True),              patch.object(dlss5.BRIDGE, "process_host") as process_host:
            report = dlss5.warmup_dlss5_runtime(32, 16)
        self.assertTrue(report["ok"])
        self.assertIn("seconds", report)
        frame = process_host.call_args.args[0]
        self.assertEqual(frame.shape, (16, 32, 3))
        # A warmup frame is a cut: there is no history for it to continue.
        self.assertEqual(process_host.call_args.args[2].reset, 1)

    def test_a_failing_warmup_is_reported_rather_than_raised(self) -> None:
        with patch.object(dlss5.BRIDGE, "initialize", side_effect=RuntimeError("no runtime")):
            report = dlss5.warmup_dlss5_runtime(32, 16)
        self.assertFalse(report["ok"])
        self.assertIn("no runtime", report["error"])


if __name__ == "__main__":
    unittest.main()
