from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import config
from pipeline.pynv_stream import (
    PyNvPassthroughStream,
    _hevc_nal_summary,
    _mux_intermediate_ts_probe_args,
    _slate_frame_pace_delay,
)


class PyNvMuxLatencyTests(unittest.TestCase):
    def _stream(self, *, container: str = "mpegts") -> PyNvPassthroughStream:
        stream = PyNvPassthroughStream(
            Path("sample.mp4"),
            0.0,
            MagicMock(),
            metadata=SimpleNamespace(color=SimpleNamespace(ffmpeg_args=lambda: [])),
            container=container,
        )
        stream.sid = 77
        return stream

    def _args_before_first_input(self, cmd: list[str]) -> list[str]:
        return cmd[: cmd.index("-i")]

    def test_direct_mpegts_raw_video_uses_configured_raw_probe(self) -> None:
        stream = self._stream()
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.subprocess.Popen", return_value=proc) as popen,
            patch.object(stream, "_audio_mode", return_value="off"),
        ):
            stream._open_muxer(59.94, 10.0)

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-fflags") + 1], "+genpts")
        raw_input_args = self._args_before_first_input(cmd)
        self.assertIn("-probesize", raw_input_args)
        self.assertEqual(raw_input_args[raw_input_args.index("-probesize") + 1], "1000000")
        self.assertIn("-analyzeduration", raw_input_args)
        self.assertEqual(raw_input_args[raw_input_args.index("-analyzeduration") + 1], "1000000")

    def test_pipe_ts_muxers_scope_probe_to_raw_video_and_audio_inputs(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen:
            stream._open_pipe_ts_muxer(59.94, 10.0, Path("audio.aac"))

        video_cmd = popen.call_args_list[0].args[0]
        final_cmd = popen.call_args_list[1].args[0]
        self.assertEqual(video_cmd[video_cmd.index("-fflags") + 1], "+genpts")
        self.assertEqual(final_cmd[final_cmd.index("-fflags") + 1], "+genpts")
        self.assertEqual(video_cmd.count("-probesize"), 1)
        self.assertEqual(video_cmd[video_cmd.index("-probesize") + 1], "1000000")
        self.assertEqual(video_cmd.count("-analyzeduration"), 1)
        self.assertEqual(video_cmd[video_cmd.index("-analyzeduration") + 1], "1000000")
        final_probe_values = [
            final_cmd[index + 1]
            for index, item in enumerate(final_cmd)
            if item == "-probesize"
        ]
        self.assertEqual(final_probe_values, ["16384", "32768"])
        final_analyze_values = [
            final_cmd[index + 1]
            for index, item in enumerate(final_cmd)
            if item == "-analyzeduration"
        ]
        self.assertEqual(final_analyze_values, ["0", "0"])
        ts_input = final_cmd.index("-f", final_cmd.index("-thread_queue_size"))
        self.assertEqual(final_cmd[ts_input + 1], "mpegts")
        ts_input_args = final_cmd[:ts_input]
        self.assertIn("-probesize", ts_input_args)
        self.assertEqual(ts_input_args[ts_input_args.index("-probesize") + 1], "16384")
        self.assertIn("-analyzeduration", ts_input_args)
        self.assertEqual(ts_input_args[ts_input_args.index("-analyzeduration") + 1], "0")
        self.assertEqual(final_cmd[final_cmd.index("-fflags") + 1], "+genpts")
        interleave_index = final_cmd.index("-max_interleave_delta")
        self.assertEqual(final_cmd[interleave_index + 1], "500000000")

    def test_raw_video_probe_override_is_scoped_to_raw_video_input(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.config.MUX_RAW_VIDEO_PROBESIZE", "524288"),
            patch("pipeline.pynv_stream.config.MUX_RAW_VIDEO_ANALYZEDURATION", "500000"),
            patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen,
        ):
            stream._open_pipe_ts_muxer(59.94, 10.0, Path("audio.aac"))

        video_cmd = popen.call_args_list[0].args[0]
        final_cmd = popen.call_args_list[1].args[0]
        video_input_args = self._args_before_first_input(video_cmd)
        self.assertIn("-probesize", video_input_args)
        self.assertEqual(video_input_args[video_input_args.index("-probesize") + 1], "524288")
        self.assertIn("-analyzeduration", video_input_args)
        self.assertEqual(video_input_args[video_input_args.index("-analyzeduration") + 1], "500000")

        ts_input = final_cmd.index("-f", final_cmd.index("-thread_queue_size"))
        self.assertEqual(final_cmd[ts_input + 1], "mpegts")
        ts_input_args = final_cmd[:ts_input]
        self.assertIn("-probesize", ts_input_args)
        self.assertEqual(ts_input_args[ts_input_args.index("-probesize") + 1], "16384")
        self.assertIn("-analyzeduration", ts_input_args)
        self.assertEqual(ts_input_args[ts_input_args.index("-analyzeduration") + 1], "0")
        self.assertEqual(final_cmd[final_cmd.index("-fflags") + 1], "+genpts")

    def test_intermediate_ts_probe_args_use_current_defaults(self) -> None:
        self.assertEqual(
            _mux_intermediate_ts_probe_args(),
            ["-probesize", "16384", "-analyzeduration", "0"],
        )

    def test_intermediate_ts_probe_args_use_explicit_config(self) -> None:
        with (
            patch("pipeline.pynv_stream.config.MUX_INTERMEDIATE_TS_PROBESIZE", "4096"),
            patch("pipeline.pynv_stream.config.MUX_INTERMEDIATE_TS_ANALYZEDURATION", "0"),
        ):
            self.assertEqual(
                _mux_intermediate_ts_probe_args(),
                ["-probesize", "4096", "-analyzeduration", "0"],
            )

    def test_pipe_ts_final_mux_uses_configured_intermediate_ts_probe(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.config.MUX_INTERMEDIATE_TS_PROBESIZE", "16384"),
            patch("pipeline.pynv_stream.config.MUX_INTERMEDIATE_TS_ANALYZEDURATION", "0"),
            patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen,
        ):
            stream._open_pipe_ts_muxer(59.94, 10.0, Path("audio.aac"))

        final_cmd = popen.call_args_list[1].args[0]
        ts_input = final_cmd.index("-f", final_cmd.index("-thread_queue_size"))
        ts_input_args = final_cmd[:ts_input]
        self.assertIn("-probesize", ts_input_args)
        self.assertEqual(ts_input_args[ts_input_args.index("-probesize") + 1], "16384")
        self.assertIn("-analyzeduration", ts_input_args)
        self.assertEqual(ts_input_args[ts_input_args.index("-analyzeduration") + 1], "0")

    def test_slate_pipe_ts_final_mux_uses_genpts_without_nobuffer(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen:
            stream._open_slate_pipe_ts_muxer(59.94, 10.0, ("127.0.0.1", 12345))

        final_cmd = popen.call_args_list[1].args[0]
        self.assertEqual(final_cmd[final_cmd.index("-fflags") + 1], "+genpts")
        self.assertEqual(final_cmd[final_cmd.index("-map") + 1], "1:v:0")
        self.assertIn("0:a:0?", final_cmd)

    def test_fmp4_fragment_duration_is_configured_default(self) -> None:
        stream = self._stream(container="mp4")
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.subprocess.Popen", return_value=proc) as popen,
            patch.object(stream, "_audio_mode", return_value="off"),
        ):
            stream._open_muxer(59.94, 10.0)

        cmd = popen.call_args.args[0]
        frag_index = cmd.index("-frag_duration")
        self.assertEqual(cmd[frag_index + 1], str(config.PASSTHROUGH_FMP4_FRAG_DURATION_US))
        self.assertEqual(config.PASSTHROUGH_FMP4_FRAG_DURATION_US, 100000)

    def test_single_stage_setts_uses_actual_fps_ticks(self) -> None:
        stream = self._stream()
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.subprocess.Popen", return_value=proc) as popen,
            patch.object(stream, "_audio_mode", return_value="aac"),
            patch.object(stream, "_cached_aac_input", return_value=Path("audio.aac")),
            patch("pipeline.pynv_stream.config.PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE", "setts"),
        ):
            stream._open_muxer(59.940060, 10.0)

        cmd = popen.call_args.args[0]
        raw_input_args = self._args_before_first_input(cmd)
        self.assertIn("-probesize", raw_input_args)
        self.assertEqual(raw_input_args[raw_input_args.index("-probesize") + 1], "1000000")
        self.assertIn("-analyzeduration", raw_input_args)
        self.assertEqual(raw_input_args[raw_input_args.index("-analyzeduration") + 1], "1000000")
        bsf_index = cmd.index("-bsf:v")
        self.assertIn("pts=N*1502:dts=N*1502", cmd[bsf_index + 1])
        self.assertNotIn("pts=N*3000:dts=N*3000", cmd[bsf_index + 1])

    def test_pipe_ts_video_mux_does_not_insert_hevc_aud_before_setts(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen:
            stream._open_pipe_ts_muxer(59.94, 10.0, Path("audio.aac"))

        video_cmd = popen.call_args_list[0].args[0]
        bsf_index = video_cmd.index("-bsf:v")
        self.assertIn("setts=time_base=1/90000", video_cmd[bsf_index + 1])
        self.assertNotIn("hevc_metadata=aud=insert", video_cmd[bsf_index + 1])

    def test_pipe_ts_without_audio_cache_uses_source_audio_muxer(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.config.PASSTHROUGH_AUDIO_MPEGTS_CACHE", False),
            patch("pipeline.pynv_stream.config.PASSTHROUGH_MPEGTS_VIDEO_SLATE", True),
            patch.object(stream, "_audio_mode", return_value="aac"),
            patch.object(stream, "_aac_cache_path", return_value=("cachekey", Path("cache.aac"))),
            patch.object(stream, "_start_slate_audio_server", return_value=("127.0.0.1", 12345)) as start_slate,
            patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen,
            patch("pipeline.pynv_stream.config.PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE", "pipe_ts"),
        ):
            stream._open_muxer(59.94, 10.0)

        start_slate.assert_not_called()
        self.assertEqual(popen.call_count, 2)
        final_cmd = popen.call_args_list[1].args[0]
        self.assertIn(str(stream.src), final_cmd)
        self.assertNotIn("tcp://127.0.0.1:12345", final_cmd)
        self.assertNotIn("-f aac", " ".join(final_cmd))
        self.assertEqual(final_cmd[final_cmd.index("-fflags") + 1], "+genpts")
        self.assertEqual(final_cmd[final_cmd.index("-map") + 1], "0:v:0")
        self.assertIn("1:a:0?", final_cmd)

    def test_pipe_ts_cache_miss_with_video_slate_disabled_does_not_start_slate(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.config.PASSTHROUGH_AUDIO_MPEGTS_CACHE", True),
            patch("pipeline.pynv_stream.config.PASSTHROUGH_MPEGTS_VIDEO_SLATE", False),
            patch.object(stream, "_audio_mode", return_value="aac"),
            patch.object(stream, "_cached_aac_input", return_value=Path("audio.aac")) as cached_audio,
            patch.object(stream, "_aac_cache_path", return_value=("cachekey", Path("cache.aac"))) as cache_path,
            patch.object(stream, "_start_slate_audio_server", return_value=("127.0.0.1", 12345)) as start_slate,
            patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen,
            patch("pipeline.pynv_stream.config.PASSTHROUGH_AUDIO_MPEGTS_TIMESTAMP_MODE", "pipe_ts"),
        ):
            stream._open_muxer(59.94, 10.0)

        start_slate.assert_not_called()
        cache_path.assert_not_called()
        cached_audio.assert_called_once_with("aac")
        self.assertEqual(popen.call_count, 2)
        final_cmd = popen.call_args_list[1].args[0]
        self.assertIn("audio.aac", " ".join(str(x) for x in final_cmd))
        self.assertNotIn("tcp://127.0.0.1:12345", final_cmd)

    def test_mux_loglevel_can_be_raised_for_diagnostics(self) -> None:
        stream = self._stream()
        video_proc = MagicMock()
        video_proc.stdin = MagicMock()
        video_proc.stdout = MagicMock()
        video_proc.stderr = MagicMock()
        final_proc = MagicMock()
        final_proc.stdin = MagicMock()
        final_proc.stdout = MagicMock()
        final_proc.stderr = MagicMock()

        with (
            patch("pipeline.pynv_stream.config.MUX_FFMPEG_LOGLEVEL", "info"),
            patch("pipeline.pynv_stream.subprocess.Popen", side_effect=[video_proc, final_proc]) as popen,
        ):
            stream._open_pipe_ts_muxer(59.94, 10.0, Path("audio.aac"))

        video_cmd = popen.call_args_list[0].args[0]
        final_cmd = popen.call_args_list[1].args[0]
        self.assertEqual(video_cmd[video_cmd.index("-loglevel") + 1], "info")
        self.assertEqual(final_cmd[final_cmd.index("-loglevel") + 1], "info")

    def test_hevc_nal_summary_reports_parameter_sets(self) -> None:
        data = (
            b"\x00\x00\x00\x01\x40\x01abc"
            b"\x00\x00\x00\x01\x42\x01defg"
            b"\x00\x00\x00\x01\x44\x01h"
            b"\x00\x00\x00\x01\x26\x01ijk"
        )
        self.assertEqual(_hevc_nal_summary(data), "32:5,33:6,34:3,19:5")

    def test_slate_pacing_sends_only_configured_burst_immediately(self) -> None:
        fps = 30.0
        start = 100.0
        self.assertEqual(
            _slate_frame_pace_delay(
                fps=fps,
                sent_frames=0,
                burst_frames=1,
                pace_start=start,
                now=start,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _slate_frame_pace_delay(
                fps=fps,
                sent_frames=1,
                burst_frames=1,
                pace_start=start,
                now=start,
            ),
            1.0 / fps,
            places=6,
        )
        self.assertAlmostEqual(
            _slate_frame_pace_delay(
                fps=fps,
                sent_frames=90,
                burst_frames=90,
                pace_start=start,
                now=start + 0.10,
            ),
            2.90,
            places=6,
        )

    def test_stderr_drain_marks_split_final_mux_stages(self) -> None:
        stream = self._stream()
        proc = SimpleNamespace(
            stderr=BytesIO(
                b"ffmpeg version test\r\n"
                b"  Stream #0:0: Video: hevc (Main), 8192x4096\r\n"
                b"  Stream #1:0: Audio: aac, 48000 Hz, stereo\r\n"
                b"Output #0, mpegts, to 'pipe:'\r\n"
            )
        )

        with patch("pipeline.pynv_stream.config.MUX_LATENCY_DIAG_VERBOSE", True):
            stream._drain_stderr(proc, "ffmpeg")

        for key in (
            "T2b_final_first_stderr",
            "T2_first_stderr",
            "T3a_final_video_codec",
            "T3b_final_audio_codec",
            "T3c_final_output_ready",
        ):
            self.assertIn(key, stream._first_chunk_marks)
        self.assertNotIn("T2a_video_first_stderr", stream._first_chunk_marks)

    def test_stderr_drain_marks_video_proc_and_honors_verbose_off(self) -> None:
        stream = self._stream()
        proc = SimpleNamespace(
            stderr=BytesIO(
                b"  Stream #0:0: Video: hevc (Main), 8192x4096\r\n"
                b"Output #0, mpegts, to 'pipe:'\r\n"
            )
        )

        with patch("pipeline.pynv_stream.config.MUX_LATENCY_DIAG_VERBOSE", False):
            stream._drain_stderr(proc, "ffmpeg-video")

        self.assertIn("T2a_video_first_stderr", stream._first_chunk_marks)
        self.assertIn("T2_first_stderr", stream._first_chunk_marks)
        self.assertNotIn("T2b_final_first_stderr", stream._first_chunk_marks)
        self.assertNotIn("T3a_final_video_codec", stream._first_chunk_marks)
        self.assertNotIn("T3c_final_output_ready", stream._first_chunk_marks)


if __name__ == "__main__":
    unittest.main()
