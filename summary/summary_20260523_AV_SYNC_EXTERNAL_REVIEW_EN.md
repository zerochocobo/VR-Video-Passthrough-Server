# A/V Sync External Review Brief

Date: 2026-05-23
Target: PTMediaServer / VR passthrough live MPEG-TS
Audience: external reviewer with no source-code access. This document describes observed behavior, experiments, objective results, and remaining hypotheses.

---

## 1. Executive Summary

Observed problem:

- During realtime passthrough playback on Quest/VR players, audio/video sync is subjectively wrong.
- Initially TensorRT mode sounded much worse than CUDA EP mode; the user described audio as significantly late, roughly 1-2 seconds.
- Green-screen mode and alpha mode showed almost the same offset at the same timestamp.
- A second player showed the same issue, so this is not obviously one player setting.
- Disabling the video slate improved the issue, but the user still hears audio about 1 second late.

Current contradiction:

- Server-side live MPEG-TS captures show audio/video stream start times aligned within about 21 ms.
- Audio-content correlation also shows the captured live audio matches the requested source timestamp within about 21 ms.
- Real-device playback still sounds late.
- Therefore a fixed hard-coded audio offset is not justified yet.

Current interpretation:

- The earlier "video slate + silent AAC + AAC cache" path had a real and explainable desync risk, especially when TensorRT first-frame latency after seek was high.
- The current default path has AAC cache disabled and video slate disabled. It reads source MP4 audio directly into the final MPEG-TS mux.
- On that path, server-side captures appear nearly synchronized.
- Remaining suspects are player/client handling of live MPEG-TS, HTTP Range, DLNA live seek headers, HEVC-in-TS buffering, or a missing objective measurement of the output video content timeline.

---

## 2. System Model

The application is a local DLNA/UPnP media server. It exposes generated passthrough streams at URLs like:

```text
GET /passthrough_live/{name}?t=<seconds>&mode=green|alpha
```

Realtime pipeline:

1. Source file: MP4 with HEVC video and AAC audio.
2. Video path: PyNvVideoCodec/NVDEC decode, RVM matting, CUDA or TensorRT inference, NVENC HEVC encode.
3. Audio path: AAC output, currently 48 kHz stereo.
4. Container: FFmpeg muxes MPEG-TS and streams it to the DLNA/VR player.

The most important architectural detail:

- Audio and video are not processed as one inseparable transcode pipeline.
- Video is extracted and processed independently: seek to the requested source time, NVDEC decode, RVM matting, alpha/green compositing, then NVENC re-encode into a new HEVC bitstream.
- Audio does not enter the GPU matting/compositing path and is not processed frame-by-frame together with video. It is read/seeked separately from the source file, converted to AAC, and only rejoined with the generated video at the final MPEG-TS mux stage.
- Therefore A/V sync depends on three boundaries agreeing on `t=<seconds>`: the video decode start, the audio read start, and the final MPEG-TS PTS/DTS/PCR timeline. A mismatch at any boundary can become a visible/audible offset.

Current MPEG-TS mux strategy:

- Two-stage mux:
  - Stage 1 muxes NVENC raw HEVC into an intermediate video-only MPEG-TS and synthesizes CFR PTS/DTS with `setts`.
  - Stage 2 muxes that intermediate TS video with source MP4 audio into the final MPEG-TS.
- When AAC disk cache is disabled, Stage 2 reads the source MP4 audio directly.

Useful search terms:

- HEVC in MPEG-TS
- AAC in MPEG-TS
- MPEG-TS PTS/DTS/PCR
- FFmpeg `setts` bitstream filter
- FFmpeg `-fflags +genpts`
- FFmpeg `-muxdelay 0`, `-muxpreload 0`, `-max_interleave_delta`
- DLNA `TimeSeekRange.dlna.org`
- DLNA `transferMode.dlna.org: Streaming`
- HTTP Range requests on generated live streams
- Quest 3 / Android / VR player live MPEG-TS buffering
- 4XVR / SKYBOX / nPlayer / LibVLC HEVC TS compatibility
- NVENC low-latency HEVC, B-frames disabled
- keyframe seeking / GOP-aligned seek / non-keyframe accurate seek
- ONNX Runtime CUDAExecutionProvider vs TensorRTExecutionProvider first-run latency
- RVM recurrent state reset after seek

---

## 3. Test Asset

User-specified file:

```text
videos/urvrsp00566_1_8k.mp4
```

ffprobe summary:

```json
{
  "video": {
    "codec": "hevc",
    "resolution": "8192x4096",
    "avg_frame_rate": "60000/1001",
    "start_time": "0.000000",
    "duration": "1280.729450"
  },
  "audio": {
    "codec": "aac",
    "sample_rate": 48000,
    "channels": 2,
    "start_time": "0.000000",
    "duration": "1280.768167"
  },
  "format_duration": "1280.768167"
}
```

The source file does not show an obvious stream start-time offset: both audio and video start at 0.

---

## 4. Experiments And Results

| # | Experiment / Change | Observation | Current Interpretation |
|---:|---|---|---|
| 1 | Subjective TensorRT vs CUDA EP comparison | TensorRT was much worse; CUDA was also off but less severe | Matched the early model where TensorRT first-frame latency after seek made the slate phase longer |
| 2 | Green mode vs alpha mode | Same timestamp sounded almost equally offset | Not likely caused only by alpha packing or green compositing |
| 3 | `check_av_sync_slate.py` | Sometimes no slate records because cached AAC bypassed slate; later TensorRT/CUDA records showed `estimated_video_lead_sec=0.0` with burst=1 | After reducing slate burst to 1, the log-level slate video lead was not significant |
| 4 | Disable AAC cache: `PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE=0` | Initially playback broke; after fixing the cache-off path, playback worked but still sounded off | The cache-off switch now uses direct source MP4 audio in the final mux |
| 5 | Output at source FPS: `PT_PASSTHROUGH_MAX_FPS=0` | Logs showed `source_fps=59.940`, `output_fps=59.940`; subjective issue remained | Old 30 fps cap was not the sole cause |
| 6 | Disable video slate: `PT_PASSTHROUGH_MPEGTS_VIDEO_SLATE=0` | User reported improvement, but audio still about 1 second late | Slate likely amplified the issue, but is not the only current explanation |
| 7 | Try another player | Issue remained | Not obviously one player setting; could still be common Quest/Android live TS behavior |
| 8 | Capture live TS and check stream start times | Alpha/green at `t=180` both had about `audio_minus_video=-0.021333s` | Captured server output does not show 1-2 s audio lateness |
| 9 | Correlate live audio content against source audio | Matched source time `179.9786875s`, offset `-0.0213125s`, correlation `0.994846` | Captured live audio content is not late |
| 10 | PyNv decoder frame PTS sanity check | `frame_at(10789).pts / 60000 = 179.963117s`, close to requested 180s | Source video indexing does not appear to jump 1-2 seconds into the future |

---

## 5. Important Log Pattern

Expected current path:

```text
source_fps=59.940
output_fps=59.940
audio cache disabled; using source audio in pipe_ts final mux
pipe_ts final mux cmd: ... -i <source.mp4> -map 0:v:0 -map 1:a:0? -c:a aac ...
```

These lines should not appear in the current no-slate path:

```text
slate audio server listening
slate video begin
```

Real-device request logs repeatedly show non-zero HTTP Range requests:

```text
request headers: range='bytes=1072693248-1073741823' ...
live cache hit: ... snapshot=<bytes> ...
live cache subscribe: ... primary=True/False ...
```

Interpretation:

- Some VR players send multiple non-zero Range requests for this generated/live stream.
- The server shares a short-lived live producer and cached prefix/snapshot between duplicate requests.
- For managed live profiles, the server treats the response more like streaming than exact byte-range VOD.
- If the player interprets these responses as successful byte-range VOD responses, it may build an incorrect timeline or buffer model. This is a remaining suspect.

---

## 6. Objective Validation Tools

### 6.1 Slate Risk Check

Command:

```bat
uv run python tools\check_av_sync_slate.py --json --max-lead 0.10
```

Representative result:

```json
{
  "sid": 1,
  "frames": 86,
  "fps": 59.94,
  "burst_frames": 1,
  "elapsed_sec": 1.45,
  "video_pts_sec": 1.4347681014347682,
  "estimated_video_lead_sec": 0.0
}
```

This only detects slate-stage video PTS lead. It does not prove real-device playback sync.

### 6.2 MPEG-TS Stream Start-Time Check

Command:

```bat
uv run python tools\check_mpegts_sync.py debug_output\urvrsp_current_alpha_t180.ts --json --max-delta 0.10
```

Result:

```text
audio_minus_video_sec = -0.021333
```

Negative means audio starts slightly before video, around one AAC frame.

### 6.3 Audio-Content Correlation

Command:

```bat
uv run python tools\check_live_audio_alignment.py ^
  debug_output\urvrsp_current_alpha_t180.ts ^
  --source videos\urvrsp00566_1_8k.mp4 ^
  --source-start 180 ^
  --duration 5 ^
  --json
```

Result:

```json
{
  "matched_source_sec": 179.9786875,
  "relative_to_source_start_sec": -0.0213125,
  "correlation": 0.994846
}
```

The captured live TS audio matches the requested source timeline closely.

---

## 7. Current Runtime State

Important current behavior:

- AAC cache default is disabled: `PT_PASSTHROUGH_AUDIO_MPEGTS_CACHE=0`
- Video slate default is disabled: `PT_PASSTHROUGH_MPEGTS_VIDEO_SLATE=0`
- Legacy `PT_PASSTHROUGH_AUDIO_MPEGTS_SLATE` remains as an alias for the video slate switch.
- Live output defaults to source cadence: `PT_PASSTHROUGH_MAX_FPS=0`
- Producer pacing is enabled.
- NVENC HEVC uses no B-frames and low-latency-oriented settings.
- MPEG-TS repeats PAT/PMT and inserts HEVC AUD for compatibility.

Local validation already passed:

```text
uv run python -m pytest tests\test_config_defaults.py tests\test_pynv_mux_latency.py
21 passed

uv run python -m pytest tests\test_settings.py tests\test_pynv_mux_latency.py tests\test_config_defaults.py
30 passed
```

---

## 8. Relationship To The Earlier Slate Root Cause

Earlier path:

- On AAC cache miss, the server could send video slate while the audio side emitted silent AAC.
- Video slate PTS was synthesized by frame index using `setts`.
- Silent AAC was paced by wall-clock time.
- If slate video frames were written in a burst, video PTS could advance faster than audio wall-clock.
- TensorRT first-frame latency after seek made the slate phase longer and therefore more likely to hit the maximum offset.

Mitigations already applied:

- Slate burst reduced to 1.
- Cache-off path no longer uses slate/TCP audio.
- Video slate has an explicit master switch and is disabled by default.

Current implication:

- The early slate mechanism was a real risk, and disabling it improved user perception.
- But on the current path, logs show direct source MP4 audio in the final mux and no slate activity.
- The remaining issue should be investigated as player/live-stream semantics, MPEG-TS PCR/packet details, or missing objective video-content measurement.

---

## 9. What Not To Do

Do not hard-code a fixed audio offset such as 1.024 seconds yet.

Reasons:

- TensorRT and CUDA subjective offsets are not identical.
- Different files, seek points, or players may differ.
- Server-side captures show audio content already near the requested source time.
- If the root cause is client Range/live-TS interpretation, an offset only masks some cases and may break correct captures.

---

## 10. Remaining Hypotheses

### H1: Player handling of live MPEG-TS plus non-zero HTTP Range is wrong

Evidence:

- Quest/Android clients repeatedly send large non-zero byte ranges.
- The server shares live producer/cache snapshots rather than providing exact VOD byte-range semantics.

Questions:

- Should generated live MPEG-TS reject non-zero Range requests with 416 or clearly advertise `Accept-Ranges: none`?
- Do DLNA `OP=10`, `TimeSeekRange`, and `X-AvailableSeekRange` induce byte/time seek behavior in these players?
- For Quest VR players, is 200 chunked streaming, 206 pseudo-VOD, or strict no-Range live streaming preferred?

### H2: Audio is verified, but video content timeline is not fully verified

Evidence:

- The audio content has been correlated to the source timeline.
- Video has only had decoder-index/PTS sanity checks, not full visual-content correlation.
- User says "audio is late", which could also mean "video content is early".

Needed:

- Use a synthetic sync clip with visible flashes/frame numbers and clicks/beeps.
- Detect flash and beep offsets in the captured TS.
- If server-side flash/beep are aligned but device playback is not, the issue is almost certainly client/device-side.

### H3: MPEG-TS PCR/packet-level behavior is not captured by stream start_time

Evidence:

- Current `check_mpegts_sync.py` only compares stream start times.
- It does not check PCR jitter, PCR-vs-PTS lead, long-term drift, packet interleave, or initial audio packet byte position.

Needed:

- Inspect the first 10 seconds with `ffprobe -show_packets` or a dedicated TS analyzer.
- Compare audio/video packet order, PCR lead, PTS/DTS sequence, and physical byte position of early audio packets.

### H4: Quest/Android HEVC-in-TS buffering behavior

Evidence:

- The source/output is 8192x4096 59.94fps HEVC in a live MPEG-TS stream.
- The device/player may apply buffering rules based on TS demux, HEVC decoder startup, or live stream modeling.

Questions:

- Do these players require CBR muxrate, stricter PCR interval, different PAT/PMT cadence, AUD, IDR/GOP constraints, or different interleave?

### H5: Mid-stream seek may be aligned to video keyframes/GOPs differently than audio

Background:

- Compressed video normally cannot start decoding from an arbitrary P/B frame.
- A decoder often has to roll back to the nearest preceding keyframe/IDR frame, decode forward, and then reach the requested frame.
- If the video path actually outputs frames from around that earlier keyframe while the audio path starts exactly at requested time `t`, the result looks like video is early and audio is late.
- Conversely, if the video path decodes preroll frames but drops them before output while the final timestamp baseline still assumes a different start, another start-time mismatch can appear.

Why this is not a complete explanation yet:

- This hypothesis can explain desync when playback starts from a middle chapter or seek point.
- However, the user also reports a similar issue when starting from the beginning. At `t=0`, the source should already start at a keyframe, so there should be no rollback-to-earlier-keyframe effect.
- Therefore keyframe/GOP alignment is a real mid-stream risk, but it cannot by itself explain all observations.

Needed validation:

- Log and capture requested `t`, actual first video source PTS, and actual first audio source PTS.
- Test the same file at `t=0`, `t=180`, `t=360`, etc., and see whether the offset correlates with GOP/keyframe distance.
- Use a burned-in frame-number/timecode test clip to objectively detect which source timestamp the output first video frame came from.

---

## 11. Recommended Next Experiments

1. Build an automatic sync calibration clip:
   - video flash or visible frame counter every second;
   - audio click/beep at exactly the same timestamps;
   - send it through the same passthrough live path.

2. Add video-content correlation:
   - extract frames from captured live TS;
   - extract source frames around the requested timestamp;
   - match via flash detection, OCR frame number, perceptual hash, or SSIM.

3. Add seek/keyframe start-point auditing:
   - log actual first video source PTS minus requested `t`;
   - log actual first audio source PTS minus requested `t`;
   - test both `t=0` and multiple mid-stream timestamps to check correlation with GOP/keyframe distance.

4. A/B test Range behavior:
   - reject non-zero Range with 416 and `Accept-Ranges: none`;
   - disable live snapshot cache;
   - alternatively implement true pseudo-VOD 206 semantics.

5. Analyze TS packets/PCR:
   - check first 10 seconds of audio/video PTS/DTS/PCR;
   - inspect packet order and byte location of early audio packets.

6. Baseline the original MP4 in the same players:
   - if the original file is also late, the issue may be source/player decoding;
   - if original is synchronized but live is not, focus on generated stream/container/transport.

---

## 12. Questions For The Expert

1. For Quest/Android VR players, how should a generated HEVC+AAC live MPEG-TS stream handle HTTP Range?
2. What commonly causes ffprobe and audio-content checks to show near-zero offset while real playback sounds about 1 second late?
3. Can PCR/packet arrival order cause player sync behavior that is invisible in simple stream start-time checks?
4. Are there known compatibility issues with synthesizing raw HEVC CFR timestamps via `setts`, then muxing with source MP4 AAC into MPEG-TS?
5. Can DLNA `TimeSeekRange`/`X-AvailableSeekRange` headers on a live stream trigger incorrect seeking or buffering behavior in VR players?
6. In realtime transcoding from a non-keyframe timestamp, where do video keyframe rollback/preroll, audio accurate seek, and final timestamp rebasing most commonly create 0.5-2 second offsets? If `t=0` also shows a similar offset, which non-keyframe causes should be prioritized?
