# AAC MPEG-TS live audio mux report - 2026-05-09

## Executive summary

The application is a local DLNA/HTTP VR passthrough server. It decodes source VR videos, runs GPU matting, composites the foreground onto a configurable background color, encodes the result with NVENC HEVC, and streams the result to DLNA clients as live MPEG-TS.

Video-only live MPEG-TS is stable across the main tested players. The current blocker is adding audio to the live passthrough stream.

When AAC audio is enabled in the production live path, both MoonVR/LibVLC and SkyboxVR/libmpv fail on the same sample. The server starts normally and PyNv continues producing video frames at about 30 fps, but FFmpeg only emits about 11-13 KB of MPEG-TS output and then stops emitting stdout. The HTTP stream stalls after the first tiny prefix. This happens even though the encoder keeps writing video packets into FFmpeg stdin.

Earlier, adding `-use_wallclock_as_timestamps 1` made FFmpeg emit continuous MPEG-TS bytes, but it also produced many `Non-monotonic DTS` warnings on the video stream. That made the stream unsuitable for MoonVR/VLC. Removing `-use_wallclock_as_timestamps 1` avoids the DTS regression in a controlled probe, but the production live mux stalls after the initial TS prefix.

The current leading hypothesis is that the production path is feeding FFmpeg a live raw HEVC elementary stream without sufficiently reliable packet timestamps for audio/video interleaving. Video-only MPEG-TS tolerates this. AAC muxing does not.

## Application context

The relevant production route is:

```text
GET /passthrough_live/{path}?t=<start_seconds>
```

The current production pipeline is:

```text
source video
  -> PyNvVideoCodec decoder
  -> RVM matting on GPU
  -> GPU NV12 green/background composite
  -> PyNvVideoCodec NVENC HEVC encoder
  -> FFmpeg subprocess muxer
  -> HTTP StreamingResponse as video/MP2T
```

The live route uses coarse chapter entries for seeking. A chapter entry starts a new live stream at `?t=<seconds>`, so true byte-range VOD seeking is intentionally not used for `/passthrough_live`.

## Tested clients

### MoonVR / LibVLC

Observed request headers include:

```http
User-Agent: VLC/3.0.18 LibVLC/3.0.18
Range: bytes=0-
Accept: */*
```

MoonVR also sends background `Lavf/58.45.100` requests through the same video endpoint, likely for thumbnails or internal probes.

### SkyboxVR / libmpv

Observed request headers include:

```http
User-Agent: libmpv
Range: bytes=0-
Accept: */*
Icy-MetaData: 1
```

Skybox/libmpv probes, disconnects, and then retries the same URL. The server already has a live-prefix cache to handle that behavior.

## Current server response strategy

The live route returns live stream semantics:

```http
HTTP/1.1 200 OK
Content-Type: video/MP2T
Cache-Control: no-store
transferMode.dlna.org: Streaming
X-Passthrough-Mode: live-mpegts
X-Passthrough-Seek-Time: <seconds>
X-Passthrough-Backend: pynv_hevc
X-Passthrough-Backend-Verdict: pynv_hevc
TimeSeekRange.dlna.org: npt=<start>-<duration>/<duration>
X-AvailableSeekRange.dlna.org: 1 npt=0.000-<duration>
X-Passthrough-FrameRate: 30
contentFeatures.dlna.org: DLNA.ORG_PN=HEVC_TS_NA_ISO;DLNA.ORG_OP=10;DLNA.ORG_CI=1;DLNA.ORG_FLAGS=41700000000000000000000000000000
Transfer-Encoding: chunked
```

For live streams, the server intentionally does not return fake `Content-Length` or `Content-Range` for libmpv, because earlier testing showed that libmpv may try to byte-seek near the estimated end of the MPEG-TS stream and then fail.

## Important history

### Video-only MPEG-TS is stable

With audio disabled:

```text
audio=off
```

MoonVR/VLC can receive sustained MPEG-TS output:

```text
progress: sent=52,442,144 stream_bytes=52,442,144 frames=147
progress: sent=104,873,168 stream_bytes=104,873,168 frames=298
...
progress: sent=823,481,736 stream_bytes=823,481,736 frames=2310
```

This suggests that the decode/matting/encode path and HTTP streaming path are healthy when audio muxing is not involved.

### `-use_wallclock_as_timestamps 1` was previously added to unstick AAC mux output

The project previously added:

```text
-use_wallclock_as_timestamps 1
```

before the raw HEVC stdin input. That made the AAC-enabled MPEG-TS mux emit bytes continuously, but a targeted probe later reproduced many FFmpeg warnings:

```text
Non-monotonic DTS; previous: 33664, current: 33664; changing to 33665.
Non-monotonic DTS; previous: 48714, current: 48714; changing to 48715.
Non-monotonic DTS; previous: 112382, current: 112381; changing to 112383.
...
```

In practice, MoonVR/VLC would receive traffic but fail to render reliably. Therefore wall-clock timestamps are not considered an acceptable final solution unless they are combined with another fix that guarantees monotonic DTS/PTS.

### Removing wallclock timestamps removes DTS regressions in a controlled probe

A tool-level probe was added:

```text
tools/mpegts_audio_pts_probe.py
```

It feeds an HEVC Annex-B elementary stream into FFmpeg stdin while muxing source AAC into MPEG-TS.

Using a generated 8-second HEVC elementary sample from the failing real video:

```text
debug_output/freak_8s_30fps.hevc
```

The no-wallclock probe produced continuous stdout bytes and no `Non-monotonic DTS` warnings:

```json
{
  "out": "<stdout>",
  "out_bytes": 8057304,
  "stdout_first_at": 0.4049,
  "stdout_last_at": 14.1257,
  "non_monotonic": false,
  "stderr": "[mpegts] Timestamps are unset in a packet for stream 0..."
}
```

The same probe with `-use_wallclock_as_timestamps 1` produced continuous bytes but `non_monotonic: true`.

This means the no-wallclock command can work in a file-derived controlled probe, but production live output still stalls.

## Failing sample

The current most useful failing sample is:

```text
videos/[8K S3D] VENTA X -'it's Live'_雨琪 (G)I-DLE - FREAK.mp4
```

The server sees it as:

```text
codec=hevc
profile=Main
pix_fmt=yuv420p
bit_depth=8
level=183
size=8192x4096
source_fps=59.940
output_fps=30.000
fps_cap=30.000
duration=235.435
frames=14112
color=tv/bt709/bt709/bt709
container=mpegts
```

The source contains AAC audio and is side-by-side 8K VR content.

## Current production FFmpeg command

For the failing sample, the production mux command is:

```text
C:\WINDOWS\ffmpeg.EXE
  -hide_banner
  -loglevel warning
  -fflags +genpts
  -f hevc
  -framerate 30.000000
  -i -
  -t 235.435
  -i G:\Downloads\VR\VRKorea\1\[8K S3D] VENTA X -'it's Live'_雨琪 (G)I-DLE - FREAK.mp4
  -map 0:v:0
  -map 1:a:0?
  -c:a aac
  -t 235.435
  -c:v copy
  -color_range tv
  -color_primaries bt709
  -color_trc bt709
  -colorspace bt709
  -max_interleave_delta 0
  -flush_packets 1
  -muxdelay 0
  -muxpreload 0
  -mpegts_flags +resend_headers
  -pat_period 0.1
  -sdt_period 0.5
  -pcr_period 20
  -f mpegts
  -
```

The production code writes PyNv NVENC HEVC access units to FFmpeg stdin and now explicitly flushes after each write.

## Production failure evidence

### MoonVR / VLC failure

After MoonVR requests the failing sample:

```text
2026-05-09 12:34:12,644 [INFO] pynv_stream: [PYNV][1] mux cmd: ... -f hevc -framerate 30.000000 -i - -t 235.435 -i ... -c:a aac ... -f mpegts -
2026-05-09 12:34:14,437 [INFO] pynv_stream: [PYNV][1] reader first stdout chunk len=8648
2026-05-09 12:34:14,438 [INFO] pynv_stream: [PYNV][1] iter first chunk len=8648 bytes=8648
2026-05-09 12:34:14,440 [INFO] media: passthrough_live[1] first chunk: len=8648 sent=8648 stream_bytes=8648
```

Then PyNv continues producing frames, but stream bytes do not advance:

```text
2026-05-09 12:34:14,623 [INFO] pynv_stream: [PYNV][1] frame 60/7063 fps=30.40 ... bytes=11468
2026-05-09 12:34:15,637 [INFO] pynv_stream: [PYNV][1] frame 90/7063 fps=30.12 ... bytes=11468
2026-05-09 12:34:16,624 [INFO] pynv_stream: [PYNV][1] frame 120/7063 fps=30.19 ... bytes=11468
2026-05-09 12:34:17,639 [INFO] pynv_stream: [PYNV][1] frame 150/7063 fps=30.06 ... bytes=11468
2026-05-09 12:34:18,629 [INFO] pynv_stream: [PYNV][1] frame 180/7063 fps=30.10 ... bytes=11468
2026-05-09 12:34:19,622 [INFO] pynv_stream: [PYNV][1] frame 210/7063 fps=30.12 ... bytes=11468
2026-05-09 12:34:20,634 [INFO] pynv_stream: [PYNV][1] frame 240/7063 fps=30.05 ... bytes=11468
2026-05-09 12:34:21,627 [INFO] pynv_stream: [PYNV][1] frame 270/7063 fps=30.07 ... bytes=11468
```

The live stall watchdog then closes the stream:

```text
2026-05-09 12:34:22,630 [INFO] media: passthrough_live[1] send stall watchdog closing stream: sent=11468 stream_bytes=11468 frames=299 idle=8.2s
2026-05-09 12:34:22,633 [INFO] pynv_stream: [PYNV][1] close begin bytes=11468 frames=299
2026-05-09 12:34:22,635 [INFO] pynv_stream: [PYNV][1] reader done chunks=2 bytes=11468 stop=True
2026-05-09 12:34:22,637 [INFO] pynv_stream: [PYNV][1] worker done frames=299 bytes_emitted=11468 reader_started=True
2026-05-09 12:34:22,640 [INFO] media: passthrough active slot released: active=0 owner=('live', '192.168.31.112', 'vlc')
```

### Skybox / libmpv failure on the same sample

After the MoonVR test, the same sample was tested in SkyboxVR/libmpv. Skybox also failed. It waited, briefly showed a green frame, then advanced to the next video.

Server-side evidence:

```text
2026-05-09 12:34:53,715 [INFO] media: passthrough_live[2] request headers: ua='libmpv' accept='*/*' range='bytes=0-' ...
2026-05-09 12:34:53,880 [INFO] pynv_stream: [PYNV][2] mux cmd: ... -f hevc -framerate 30.000000 -i - -t 235.435 -i ... -c:a aac ... -f mpegts -
2026-05-09 12:34:55,678 [INFO] pynv_stream: [PYNV][2] reader first stdout chunk len=8648
2026-05-09 12:34:55,681 [INFO] media: passthrough_live[2] first chunk: len=12784 sent=12784 stream_bytes=12784
```

Skybox/libmpv issued duplicate probe/play requests and hit the live prefix cache:

```text
2026-05-09 12:34:55,695 [INFO] media: passthrough_live[3] live cache hit: key=... bytes=12784 frames=54
2026-05-09 12:34:55,710 [INFO] media: passthrough_live[4] live cache hit: key=... bytes=12784 frames=54
```

But the underlying producer again stalled at about 13 KB:

```text
2026-05-09 12:34:55,863 [INFO] pynv_stream: [PYNV][2] frame 60/7063 ... bytes=12972
2026-05-09 12:34:56,874 [INFO] pynv_stream: [PYNV][2] frame 90/7063 ... bytes=12972
...
2026-05-09 12:35:03,761 [INFO] media: passthrough_live[2] send stall watchdog closing stream: sent=12972 stream_bytes=12972 frames=296 idle=8.0s
2026-05-09 12:35:14,210 [INFO] media: live session closed: key=... bytes=12972 stream_bytes=12972 frames=609 reason=ttl expired
```

This strongly suggests the issue is not player-specific. Both clients received only the tiny TS prefix because the FFmpeg muxer stopped emitting stdout.

## What has already been tried

### 1. AAC in production MPEG-TS

Configuration:

```bat
set PT_PASSTHROUGH_AUDIO_MPEGTS=aac
```

Result:

```text
Video stream starts, FFmpeg emits only about 11-13 KB, then stdout stalls.
MoonVR black loading screen.
Skybox waits, briefly shows green, then skips to next video.
```

### 2. AAC with `-use_wallclock_as_timestamps 1`

Result:

```text
FFmpeg emits continuous MPEG-TS bytes.
But FFmpeg reports many video Non-monotonic DTS warnings.
MoonVR/VLC receives traffic but decoding/playback is unreliable.
```

Conclusion:

```text
This option fixes mux liveness but breaks timestamp monotonicity.
```

### 3. AAC without `-use_wallclock_as_timestamps 1`

Result in controlled probe:

```text
No DTS regression and stdout continues.
```

Result in production:

```text
Mux stdout stalls after only about 11-13 KB while PyNv continues producing frames.
```

Conclusion:

```text
The controlled probe is not yet equivalent to the production live access-unit stream.
```

### 4. `copy` source audio

Earlier tests indicated copied source audio can delay or block the first muxed bytes long enough that clients show a black loading screen. AAC transcoding is currently preferred over copy for live MPEG-TS.

### 5. Client-specific UA handling

UA-specific response headers are already used for Skybox/libmpv versus MoonVR/VLC behavior, but the current AAC stall occurs below the HTTP compatibility layer. The muxer is not producing enough data for either client.

### 6. Live stall cleanup

The HTTP live stall watchdog now cancels stalled responses and releases the active slot, preventing permanent `503 busy` after the mux stalls. This improves recovery but does not solve audio playback.

## Current hypothesis

The raw HEVC stream produced by PyNvVideoCodec is being passed to FFmpeg as:

```text
-f hevc -framerate 30.000000 -i -
```

and then stream-copied:

```text
-c:v copy
```

When audio is absent, FFmpeg can mux this into MPEG-TS well enough for clients to decode.

When AAC audio is present, FFmpeg must interleave video and audio based on packet timestamps. The raw HEVC demuxer plus generated timestamps may not be producing a stable live timeline for the exact access-unit stream produced by PyNv/NVENC. FFmpeg emits an initial PAT/PMT/headers/small prefix, then appears to buffer or wait indefinitely instead of writing more stdout.

`-use_wallclock_as_timestamps 1` gives FFmpeg a live timeline and makes it emit packets, but because the raw HEVC packetization/timing is not aligned with the intended 30 fps CFR timeline, DTS can repeat or regress.

## Specific questions for expert review

1. For a live raw HEVC elementary stream written to FFmpeg stdin, with one access unit per write, what is the correct way to synthesize monotonic PTS/DTS suitable for MPEG-TS muxing with AAC?

2. Is `-f hevc -framerate 30 -i - -c:v copy` expected to provide reliable packet timestamps for live audio/video interleaving, or is it only reliable after stdin EOF / file-style input?

3. Is FFmpeg's `setts` bitstream filter appropriate here, for example:

   ```text
   -bsf:v setts=pts=N/(30*TB):dts=N/(30*TB):duration=1/(30*TB)
   ```

   If yes, what exact expression and time base should be used for a 30 fps raw HEVC input stream copied into MPEG-TS?

4. Would a timestamped intermediate container be the correct architecture?

   Possible example:

   ```text
   PyNv HEVC -> FFmpeg video-only NUT/Matroska/MPEG-TS with generated CFR timestamps -> second FFmpeg mux with source AAC
   ```

   The concern is added latency and process complexity.

5. Is there an FFmpeg option combination that avoids both failure modes?

   Known failure modes:

   ```text
   no wallclock: mux stdout stalls after about 11-13 KB in production
   wallclock: mux stdout flows but video DTS becomes non-monotonic
   ```

   Candidate areas:

   ```text
   -fflags +genpts placement
   -copyts / -start_at_zero
   -avoid_negative_ts make_zero
   -muxdelay / -max_interleave_delta values
   -fps_mode cfr / -vsync cfr
   -bsf:v setts
   audio input ordering and -ss/-t placement
   ```

6. If the final transport is DLNA live MPEG-TS, is AAC inside MPEG-TS the best audio choice for MoonVR/LibVLC and Skybox/libmpv, or should another codec/container strategy be considered?

## Current workaround

For stable video-only playback:

```bat
set PT_PASSTHROUGH_AUDIO_MPEGTS=off
```

or for MoonVR-specific fallback:

```bat
set PT_PASSTHROUGH_AUDIO_MPEGTS_VLC=off
```

This is not acceptable for product readiness because passthrough live output needs audio.

## Relevant files

```text
pipeline/pynv_stream.py
  Production PyNv decode/matting/encode plus FFmpeg mux subprocess.

http_app/routes_media.py
  /passthrough_live route, DLNA/HTTP response headers, live prefix cache, stall cleanup.

tools/mpegts_audio_pts_probe.py
  Tool for testing raw HEVC stdin + source AAC MPEG-TS mux variants outside production.

tools/audio_mux_probe.py
  Earlier standalone audio mux validation using generated HEVC video.

prompt/HANDOVER_20260509.md
  Chronological implementation notes and prior findings.
```

## Minimal reproduction data to share

Sample:

```text
[8K S3D] VENTA X -'it's Live'_雨琪 (G)I-DLE - FREAK.mp4
```

Production command:

```text
ffmpeg -hide_banner -loglevel warning -fflags +genpts ^
  -f hevc -framerate 30.000000 -i - ^
  -t 235.435 -i "<source>.mp4" ^
  -map 0:v:0 -map 1:a:0? ^
  -c:a aac -t 235.435 ^
  -c:v copy ^
  -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 ^
  -max_interleave_delta 0 -flush_packets 1 -muxdelay 0 -muxpreload 0 ^
  -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.5 -pcr_period 20 ^
  -f mpegts -
```

Observed production behavior:

```text
first stdout chunk: 8648 bytes
total emitted before stall: 11468 to 12972 bytes
video frames produced before watchdog: about 296 to 302
encoder remains near 30 fps
FFmpeg stdout stops advancing
no useful player output
```

Controlled probe contradiction:

```text
Same sample's generated 8s HEVC elementary file + AAC can emit about 8 MB via stdout without wallclock timestamps and without Non-monotonic DTS.
Production live PyNv-generated access units still stall.
```
