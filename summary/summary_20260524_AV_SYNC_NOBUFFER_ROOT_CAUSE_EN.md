# AV Sync Root Cause Summary: raw HEVC `+nobuffer` Dropped the First GOP

Date: 2026-05-24

## Conclusion

The main A/V sync issue was not AAC cache, video slate, Range probing, or a fixed audio delay. The root cause was FFmpeg muxing raw HEVC into MPEG-TS with:

```text
-fflags +genpts+nobuffer+flush_packets
```

On the raw HEVC input, `+nobuffer` caused FFmpeg to drop about one initial GOP. With `PT_PASSTHROUGH_GOP=60` and a 59.94fps source, this made video content lead audio by about one second.

The perceived “audio is late by about one second” was actually “video content is early by about one second”.

## Project Context

`/passthrough_live/{name}` is a generated live stream:

- Video: source video is decoded by PyNvVideoCodec, processed by GPU matting/composite or alpha packing, then encoded by PyNv/NVENC as HEVC elementary stream.
- Audio: source audio is read by FFmpeg during the final mux stage and transcoded to AAC.
- Final container: the current PyNv path uses two mux stages:
  1. raw HEVC -> video-only MPEG-TS with CFR timestamps synthesized by `setts`;
  2. video-only MPEG-TS + source audio -> final MPEG-TS.

Audio and video are therefore processed separately and merged later, so both audio-content time and video-content time must be verified.

## Key Evidence

Test video:

```text
videos/urvrsp00566_1_8k.mp4
```

The original MP4 was confirmed to be in sync.

Audio checks were consistently normal:

```text
audio_content_offset ≈ -0.021s
audio/video stream start delta ≈ -0.021s
```

This proved the final TS audio content was near the requested source time.

After adding video content matching, the old output showed:

```text
expected source frame index: 10789
matched source frame index: 10849
delta: 60 frames ≈ 0.997s
```

That exactly matches the current GOP length.

A direct probe confirmed the mechanism:

- PyNv raw HEVC encoding of 120 frames was intact.
- Muxing with `+genpts+nobuffer+flush_packets` produced only 60 video packets, and the first decoded frame matched source index `10849`.
- Muxing without `+nobuffer` produced 120 video packets, and the first decoded frame matched source index `10789`, with about `-0.004s` video content offset.

## Ruled Out

- AAC cache: disabling it did not fix the issue.
- Video slate: disabling it improved startup behavior but left about one second of offset.
- Large Range requests: likely metadata/tail probing; the decisive evidence came from server-side content matching.
- Fixed audio offset: not appropriate because audio itself was not one second late.
- Keyframe seek: mid-stream starts can have keyframe effects, but this issue also appeared at `t=0`, and the final root cause was mux-time first-GOP loss.

## Fix

Changed:

- CODE: `config.py`
  - `MUX_NOBUFFER_ENABLE = _env("MUX_NOBUFFER_ENABLE", "0") == "1"`
  - Default `PT_MUX_NOBUFFER_ENABLE` changed from `1` to `0`.
  - Comments now document the raw HEVC first-GOP loss risk.
- CODE: `pipeline/pynv_stream.py`
  - `_mux_fflags()` still supports `+genpts+nobuffer+flush_packets`, but the default path now returns `+genpts`.
  - `_open_pipe_ts_muxer()` still calls `_mux_fflags()` for the video-only TS mux, so the default FFmpeg input flags become `-fflags +genpts`.
- CODE: `tests/test_config_defaults.py`
  - `test_mux_latency_defaults_are_low_latency` now asserts `config.MUX_NOBUFFER_ENABLE == False`.
- CODE: `tests/test_pynv_mux_latency.py`
  - Raw direct mux and pipe_ts video mux expectations changed from `+genpts+nobuffer+flush_packets` to `+genpts`.
- CODE: `tools/check_live_video_alignment.py`
  - Added objective green-mode video-content alignment checker.
  - It extracts a captured frame, masks out green background, decodes nearby source frames with PyNv, and matches foreground luma.

After the fix and server restart, the user confirmed real-device playback is fully in sync.

## Code Map

- CODE CONFIG: `config.py` -> `MUX_NOBUFFER_ENABLE`
- CODE MUX FLAGS: `pipeline/pynv_stream.py` -> `_mux_fflags()`
- CODE PIPE-TS VIDEO MUX: `pipeline/pynv_stream.py` -> `_open_pipe_ts_muxer()`
- CODE AUDIO CHECK TOOL: `tools/check_live_audio_alignment.py`
- CODE VIDEO CHECK TOOL: `tools/check_live_video_alignment.py`
- CODE SLATE CHECK TOOL: `tools/check_av_sync_slate.py`
- CODE CAPTURE TOOL: `tools/capture_live_mpegts.py`
- CODE TESTS: `tests/test_config_defaults.py`, `tests/test_pynv_mux_latency.py`

## Verification Commands

After restarting the server:

```bat
uv run python tools\capture_live_mpegts.py --host 127.0.0.1 --port 8200 --name videos/urvrsp00566_1_8k.mp4 --t 180 --mode green --seconds 7 --out debug_output\after_fix_green.ts --check
uv run python tools\check_live_audio_alignment.py debug_output\after_fix_green.ts --source videos\urvrsp00566_1_8k.mp4 --source-start 180 --json
uv run python tools\check_live_video_alignment.py debug_output\after_fix_green.ts --source videos\urvrsp00566_1_8k.mp4 --source-start 180 --json
```

Expected:

- Audio content offset near zero, with small tens-of-milliseconds tolerance.
- Video content offset near zero, no one-second lead.

Code validation:

```bat
uv run python -m pytest tests\test_config_defaults.py tests\test_pynv_mux_latency.py
uv run python -m py_compile tools\check_live_video_alignment.py
```

Passed:

```text
21 passed
py_compile passed
```

## Follow-up Notes

Do not set this for normal playback:

```bat
PT_MUX_NOBUFFER_ENABLE=1
```

It should remain a diagnostic-only switch. Future A/V sync investigations should not rely only on `ffprobe stream start_time`; they should run both audio-content matching and video-content matching.
