# Alpha Passthrough ThreadedDecoder Flicker Probe Notes

## Background

External review pointed out that red-alpha flicker means the alpha mask is jumping strongly between frames. `owned_copy()` plus `Device().synchronize()` can only prove previous writes are complete. It cannot prove PyNv `ThreadedDecoder` will not reuse the same surface later, and it cannot prove RVM recurrent state will not amplify one corrupted input frame.

This round added two low-risk code safeguards and ran the T3 isolation test.

## Code Safeguards

### 1. Shape validation in `owned_copy()`

File: `pipeline/pynv_io.py`

- NV12/P016 Y plane must match `(h, w)`.
- UV plane accepts both valid layouts:
  - `(h / 2, w)`
  - `(h / 2, w / 2, 2)`

The first validation attempt was too strict and only accepted `(h / 2, w)`. T3 showed the current PyNv UV shape is `(1920, 3840, 2)`. The guard has been corrected.

### 2. More explicit online alpha decoder logging

File: `pipeline/pynv_stream.py`

The alpha path now logs:

```text
alpha decoder detail: effective_decoder=<...> batch=<...> buffer=<...> owned_copy=<...> allow_threaded=<...>
```

This makes future regressions easier to inspect from server logs.

## T3 Isolation Test

Purpose:

Run alpha + `ThreadedDecoder` with `batch_size=1, buffer_size=2` to see whether reducing the surface reuse window mitigates or removes the flicker. If flicker disappears, the surface recycle / producer-consumer race hypothesis becomes stronger.

Command:

```powershell
$env:PT_PASSTHROUGH_PYNV_DECODER='threaded_serial'
$env:PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER='1'
$env:PT_PASSTHROUGH_PYNV_THREADED_BATCH_SIZE='1'
$env:PT_PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE='2'
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
.\.venv\Scripts\python.exe tools\offline_alpha_passthrough.py videos\72456_3840p.mp4 --engine rvm --out debug_output\alpha_t3_threaded_b1_buf2.mp4 --duration 3 --fps 30 --alpha-stride 1 --input-size 1024 --sbs-batch --audio off --preset P1
```

Output file:

```text
debug_output\alpha_t3_threaded_b1_buf2.mp4
```

Key log lines:

```text
[offline-alpha] decoder=threaded_serial batch=1 buffer=2 owned_copy=True
frames = 90
throughput = 31.86 fps
decode_avg = 2.481 ms
matting_avg = 25.255 ms
rvm_ort_avg = 24.763 ms
```

## Interpretation So Far

- T3 successfully generated 90 frames / 3 seconds of output.
- The generated output still needs visual inspection for red-alpha flicker.
- If T3 does not flicker, large-batch surface reuse / producer-consumer competition becomes the stronger hypothesis.
- If T3 still flickers, the issue is more likely in the interaction between ThreadedDecoder and RVM recurrent state / CUDA stream visibility, not just cross-batch surface recycling.

## Production Policy After UV Fix

- After the UV coverage fix, `PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER` now defaults to `1`.
- Alpha passthrough follows the UI memory profile / `PT_PASSTHROUGH_PYNV_DECODER`, including `threaded_serial`.
- If real-device testing still shows red/gray alpha flicker, set `PT_PASSTHROUGH_ALPHA_ALLOW_THREADED_DECODER=0` to force alpha back to `SimpleDecoder`.

## Additional Observation: Red Turns Gray

The user later observed that, when paused on a flickering frame, the red alpha channel does not lose the mask shape. The mask shape is still present, but the color turns from red to gray.

This changes the diagnosis:

- If the mask shape remains visible, the Y plane is probably still carrying the mask.
- Red turning gray points more directly at unstable or incomplete NV12 UV chroma writes.
- Therefore the issue may not be alpha mask value instability. It may be incomplete UV coverage in the alpha block overlay.

## Alpha Packer UV Fix

File: `pipeline/alpha_packer.py`

Previous logic:

- `overlay_alpha_packer_layout` wrote Y for every non-zero mask pixel inside the alpha blocks.
- It wrote NV12 UV only when both `x` and `y` were even.
- If the top-left pixel of a 2x2 chroma block had mask 0 while another pixel in the same 2x2 block had non-zero mask, the result was:
  - Y plane contains the mask brightness;
  - UV plane was not forced to red and could remain source chroma or neutral gray;
  - visually, the mask shape remains but red turns gray.

Fix:

- Added `alpha_layout_source()` / `alpha_layout_mask_at()`.
- For each 2x2 NV12 UV block, the overlay kernel now reads the maximum mask value across the four covered alpha-layout pixels.
- UV is computed from that maximum mask, ensuring any covered 2x2 block is chroma-colored red.

Generated comparison clips:

```text
debug_output\alpha_uv_fix_simple_3s.mp4
debug_output\alpha_uv_fix_threaded_b1_buf2_3s.mp4
debug_output\alpha_uv_fix_simple_10s.mp4
debug_output\alpha_uv_fix_threaded_b1_buf2_10s.mp4
```

The 10s clips looked close to the SimpleDecoder baseline, so threaded alpha was enabled for real-device testing.

## Follow-up: Black Cross Lines

The user inspected exported frames and found that most bad frames are not random alpha-mask disappearance. The red mask is cut by vertical, horizontal, or cross-shaped black lines, and some cut regions turn gray.

This points to another alpha-packer overlay issue:

- The previous kernel returned early when `mask == 0`.
- That meant alpha-layout pixels with zero mask did not explicitly overwrite the underlying projected image.
- In the six alpha blocks, this leaves source image / old chroma in places that should be black, producing visible black/cross-like cuts and gray regions.

Fix:

- Removed the `if (mask == 0) return;` early return in `overlay_alpha_packer_layout`.
- The alpha layout now writes every alpha-block pixel:
  - zero mask becomes explicit black YUV;
  - non-zero mask becomes red YUV;
  - each 2x2 UV block still uses the maximum mask over covered pixels.

Generated v2 threaded diagnostic package:

```text
debug_output\alpha_flicker_5s_threaded_b4_buf8_v2.mp4
debug_output\alpha_flicker_5s_frames_v2\threaded_full\frame_0001.png ... frame_0150.png
debug_output\alpha_flicker_5s_frames_v2\threaded_scaled\frame_0001.png ... frame_0150.png
debug_output\alpha_flicker_5s_frames_v2\sheets\sheet_0001_0030.png ... sheet_0121_0150.png
```

These v2 sheets should be compared against the original `debug_output\alpha_flicker_5s_frames\sheets`.
