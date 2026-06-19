# SI Progressive Virtual MP4 Original-Audio A/V Desync Research Summary, Corrected

Date: 2026-06-18

## 0. Correction

The previous summary made an incorrect inference: it attributed the observed “obvious A/V desync” too strongly to `DEFAULT_SI_DELAY_SECONDS=1.0`. The user has clarified that the actual test issue is **the original video sound being out of sync with the picture**, not the translated SI voice being offset relative to the original sound.

Therefore the conclusion must be corrected:

- `si_delay_seconds` only affects the SI WAV / translated voice relative to the original source audio.
- It cannot explain “the original source audio is out of sync with the picture”.
- Changing `DEFAULT_SI_DELAY_SECONDS` from `1.0` to `0.0` is still a reasonable default change, but it does not close the original-audio-vs-video desync bug.
- Further investigation should focus on whether the source-audio component is shifted in the sidecar, whether the imported sidecar audio track is interpreted differently after virtual MP4 assembly, and whether the player/client is caching an old moov or using non-standard byte-seek heuristics.

This file supersedes the earlier conclusion.

## 1. Current Implementation Context

`/media_si` currently uses a progressive virtual MP4 remux path:

- Video is not re-encoded. Source MP4 video samples are reused by byte slicing.
- Source audio `[0:a:0]` and SI WAV `[1:a:0]` are mixed by ffmpeg into a small AAC sidecar MP4.
- The logical virtual MP4 is served as:
  - in-memory `ftyp + rewritten moov + mdat header`
  - source video sample file regions
  - AAC sidecar audio sample file regions
- `moov` handling:
  - source video trak timing/description boxes are preserved.
  - sidecar audio trak is imported.
  - only `stsc` is rewritten to one-sample-per-chunk and `stco/co64` is rewritten to absolute offsets in the virtual file.

Relevant code:

- `pipeline/si_virtual_mp4.py`
- `http_app/routes_media.py`
- `dlna/content_directory.py`
- `utils/si_filter.py`

## 2. Actual Meaning Of `si_delay_seconds`

`si_delay_seconds` is the UI setting for delaying the translated SI voice relative to the original video sound.

Code location: `utils/si_filter.py::build_si_mix_filter()`

Input mapping:

- `[0:a:0]`: original audio from the source video.
- `[1:a:0]`: `.si.wav` translation / simultaneous interpretation audio.

`si_delay_seconds` is applied only to `[1:a:0]`:

```text
[1:a:0] ... adelay={si_delay_ms} ...
```

Meaning:

- Positive value: translated SI voice is later.
- `0.0`: no artificial SI delay.
- It does not move source audio `[0:a:0]`.

Therefore:

- If the SI voice is late relative to the original sound, the old `1.0` default can explain that.
- If the original source sound is late/early relative to the picture, the old `1.0` default cannot explain that.

## 3. Known Sidecars And Metadata

Test video:

- `videos\SI_TEST_8K.mp4`
- `videos\SI_TEST_8K.si.wav`

Old sidecar:

- `runtime_cache\si_virtual_mp4\37be0fbab73c053fa0708d1ffc8a04afaf05931006c4894421482301c859ec62.audio.mp4`
- Parameters: `si_delay_seconds=1.0`
- File size: `72,596,626` bytes

New sidecar:

- `runtime_cache\si_virtual_mp4\b6a4e5fc94f1f98a15f57e2abeddbd39a12f1711a82dbb42bd04ac9debd7b0ab.audio.mp4`
- Parameters: `si_delay_seconds=0.0`
- File size: `72,568,392` bytes

Both sidecars report identical audio stream-level metadata through `ffprobe`:

```text
codec_name=aac
time_base=1/48000
start_time=0.000000
duration=3007.104000
nb_frames=140959
```

## 4. External Expert Box-Structure Findings

The external expert inspected the actual MP4 box structures and reported:

| Check | Source Video | 0.0s Sidecar | Finding |
|---|---:|---:|---|
| movie timescale (`mvhd`) | 1000 | 1000 | matched |
| video edit list | `media_time=528 @16000 ~=33ms` | - | CTS compensation |
| audio edit list | - | `media_time=1024 @48000 ~=21.3ms` | AAC priming trim |
| video `stsz/stts/co64` count | 180240 | - | matched |
| audio `stsz/stts/co64` count | - | 140959 | matched |

These findings support:

- movie timescale mismatch is not the likely cause.
- sidecar audio edit-list/AAC-priming information exists and is preserved.
- sample-table counts match the PyAV-demuxed offset counts.
- even if a player ignored edit lists, the video/audio residual difference would be about 12 ms, not an obvious desync.

These are useful structural findings, but they do not prove that `si_delay=1.0` caused the user’s reported original-audio-vs-picture desync. The user clarified that the original audio is the part that appears out of sync.

## 5. New Evidence: The Source-Audio Component Is Not Shifted In The Sidecar

To test whether the ffmpeg mix/filter stage shifts the original source audio, a PCM cross-correlation test was run:

Method:

- Decode source MP4 `[0:a:0]` to 48 kHz mono f32 PCM.
- Decode the sidecar audio track to the same PCM format.
- Run FFT cross-correlation over 20-second windows.
- Positive lag means the sidecar audio is later than the source audio.
- Test at the beginning, middle, and later parts of the file.

Results:

```text
delay1_37be
  start=  0.000s lag_ms=    4.958 score=0.1242
  start=600.000s lag_ms=    4.479 score=0.2690
  start=1500.000s lag_ms=   4.479 score=0.3473

delay0_b6a4
  start=  0.000s lag_ms=    4.979 score=0.1085
  start=600.000s lag_ms=    4.479 score=0.1795
  start=1500.000s lag_ms=   4.479 score=0.2944
```

Interpretation:

- In both the old `delay=1.0` sidecar and the new `delay=0.0` sidecar, the original-audio component is only about `4.5-5 ms` later than the original source audio.
- This is far below an obvious perceptual desync.
- The lag remains stable at 600 s and 1500 s, so there is no evidence of drift.
- The ffmpeg mix/filter stage did not shift the original source audio in any meaningful way.
- Changing `si_delay_seconds` did not affect original-audio alignment, which matches the code semantics: it only affects the SI WAV path.

This is the most important new evidence.

## 6. Directions That Are Now Lower Priority Or Ruled Out

Based on the external box inspection and PCM correlation:

1. **SI delay causing original-audio desync**
   - Ruled out.
   - It only affects the translated SI voice.

2. **ffmpeg mix/filter shifting the original audio**
   - Very unlikely.
   - Source audio vs sidecar source-audio component differs by only about `4.5-5 ms`.

3. **Sidecar original-audio drift**
   - Very unlikely.
   - Correlation lag is stable at 600 s and 1500 s.

4. **Obvious sample-table count mismatch**
   - The external expert’s `stsz/stts/co64` count check rules this out.

5. **Movie timescale mismatch**
   - The external expert found both source and sidecar `mvhd` timescale are 1000.

## 7. Remaining Valid Suspects

### 7.1 Which Track/Moov The Real Device Actually Used

After the next device test, inspect `server.log`:

```powershell
rg -n "SI mixed AAC sidecar|built SI progressive virtual MP4|media_si" debug_output\server.log
```

The log should include:

```text
digest=... delay=...
```

For original-audio-vs-picture desync, `delay=1.0/0.0` is not itself the explanation. The log is still useful to confirm the selected sidecar, runtime parameters, and request path.

### 7.2 Player Interpretation Of The Assembled Virtual MP4

The sidecar itself keeps the source-audio component aligned with the source audio. But after virtual MP4 assembly, the player sees a new movie:

- source video trak
- imported sidecar audio trak
- rewritten `stsc/co64`
- copied timing boxes/edit lists

If the original audio is still obviously out of sync with video, the issue is more likely one of:

- the actual served virtual `moov` differs from the expected structure
- the target player has a compatibility issue with this source-video-trak + imported-audio-trak combination
- the player does not fully follow sample tables for progressive MP4 seeking and uses byte-position heuristics
- the client cached an old moov or old DLNA item

### 7.3 Whether The Original File Is Already Out Of Sync On The Target Player

Required control test:

- Play the original `SI_TEST_8K.mp4` directly in the same player.
- Play the normal non-SI DLNA item or `/media` route.
- Check whether only `/media_si` desyncs or the original file also desyncs.

If the original file is already desynced on the target player, `/media_si` is not the root cause.

## 8. Recommended Next Steps

### 8.1 Correct The Device-Test Observation

For the next test, record:

- Is the issue “original source sound vs picture”, or “SI translated voice vs original sound”?
- Is original audio early or late?
- Approximate offset: 100 ms, 500 ms, 1 s, or larger?
- Is the offset the same at the beginning and after mid-file seek?
- Does the same player play the original `SI_TEST_8K.mp4` in sync?

### 8.2 Probe The HTTP Virtual Output Directly

If the server is running, probe/decode the `/media_si` URL itself, not just the sidecar:

```powershell
ffprobe -v error -show_streams -show_format "http://127.0.0.1:8200/media_si/SI_TEST_8K.mp4"
```

Decode audio from the HTTP virtual MP4 and compare it with source audio:

```powershell
ffmpeg -v error -ss 600 -i "http://127.0.0.1:8200/media_si/SI_TEST_8K.mp4" -t 20 -map 0:a:0 -ac 1 -ar 48000 -f f32le virtual_audio_600.f32
```

If the virtual URL’s decoded audio also correlates with source audio within about 5 ms, the service-side audio payload/timing is likely correct, and the issue shifts toward player compatibility or video timeline interpretation.

### 8.3 Dump The Actual Virtual Moov

The actual rewritten `moov` served to the client should be dumped and inspected:

- Does the audio trak come from the expected sidecar digest?
- Are `mvhd/tkhd/mdhd/elst/stts/ctts/stsz/co64` consistent with the expert’s findings?
- Are `co64` entries monotonic and do counts match sample counts?

A useful temporary debug output would be:

```text
debug_output/si_virtual_mp4_last_moov.mp4box.bin
```

or an init segment:

```text
ftyp + moov + mdat header
```

for independent Bento4/MP4Box/pymp4 inspection.

### 8.4 Do Not Change Audio Timing Filters Yet

Do not add these just because they were mentioned in the earlier summary:

- `asetpts=PTS-STARTPTS`
- `aresample=async=1:first_pts=0`
- `-avoid_negative_ts make_zero`

Reason:

- PCM correlation shows the sidecar’s source-audio component is not meaningfully shifted.
- Forcing new PTS/priming behavior may break the current AAC priming/edit-list handling.

## 9. Current Corrected Conclusion

More accurate current conclusion:

1. `DEFAULT_SI_DELAY_SECONDS=1.0` is a default-value issue for the SI translated voice relative to original audio. Changing it to `0.0` is reasonable.
2. After the user clarification, the reported issue is original audio vs picture desync, so it cannot be attributed to SI delay.
3. The source-audio component inside both sidecars is aligned with the source audio within about `5 ms`, with no meaningful drift.
4. If `/media_si` still shows obvious original-audio-vs-picture desync, the next target is actual HTTP virtual MP4 output and target-player interpretation, not SI delay or audio filter timing.

In one sentence:

**The SI delay fix only addresses translated-voice offset. The original-audio A/V desync remains open. Current evidence suggests sidecar/filter timing is clean, so the next investigation should focus on the actual virtual MP4 output and player interpretation.**

## 10. 2026-06-18 Additional Experiment: Audio Edit-List A/B

Based on the latest expert feedback, the most useful device-side hypothesis is:

> The player handles source-video edit-list/negative DTS/ctts and sidecar-audio AAC priming edit-list asymmetrically, causing a fixed original-audio-vs-picture offset after remux.

To make this testable, a narrow A/B switch was added:

```text
PT_SI_AUDIO_EDIT_MODE=remove|preserve
```

Current default:

```text
PT_SI_AUDIO_EDIT_MODE=remove
```

Mode semantics:

- `preserve`: old behavior; keep the imported sidecar audio trak `edts/elst`.
- `remove`: remove only the imported audio trak `edts/elst` in the virtual moov. It does not modify the sidecar file, video trak, sample payload, or SI delay.

Implementation:

- `config.py`
  - Added `SI_AUDIO_EDIT_MODE`, default `remove`.
- `pipeline/si_virtual_mp4.py`
  - `_rewrite_trak(..., drop_edts=True)` can skip `edts`.
  - `_build_moov(..., audio_edit_mode=...)` applies this only to the imported audio trak.
  - Layout cache digest / ETag now include `audio_edit_mode`, so switching A/B modes cannot reuse the wrong in-process moov.
  - `ProgressiveSIVirtualMp4.audio_edit_mode` records the current mode.
- `http_app/routes_media.py`
  - Added response header:
    - `X-SI-Audio-Edit: remove|preserve`

Real `SI_TEST_8K.mp4` verification:

```text
audio=b6a4e5fc94f1f98a15f57e2abeddbd39a12f1711a82dbb42bd04ac9debd7b0ab.audio.mp4
audio_edit=remove
size=7449678144
moov=6173403
samples=180240+140959
regions=321200
etag=be9eb7fa1da278feee64a2ee1ec2aa00
```

Actual init segment inspection:

```text
audio_edit remove
video_edts True
audio_edts False
moov_size 6173403
```

This confirms the experiment removes only the imported audio trak `edts`; the source video trak `edts` remains intact.

Tests:

```text
tests\test_si_mix.py tests\test_si_virtual_mp4.py tests\test_config_defaults.py tests\test_routes_media_cache.py
74 passed

tests\test_content_directory_modes.py -k "not versioned_live_id_resolves"
34 passed, 1 deselected, 4 subtests passed

git diff --check
passed, only CRLF warnings
```

Next device-test checks:

- Default is now `remove`; restart the server and test `/media_si`.
- Confirm the response/log shows:
  - `X-SI-Audio-Edit: remove`
  - `built SI progressive virtual MP4 ... audio_edit=remove ...`
- If original-audio-vs-picture sync is fixed under `remove`, the player likely mishandles audio priming edit-list differently from video edit/ctts.
- To restore the old behavior for A/B testing:

```powershell
$env:PT_SI_AUDIO_EDIT_MODE="preserve"
```

Then restart the server, re-Browse the `[SI]` item, and test again.
