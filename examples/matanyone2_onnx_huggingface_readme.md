# MatAnyone2 ONNX

This repository contains fixed-shape ONNX exports of MatAnyone2 inference subgraphs.

The models are intended for projects that want to run MatAnyone2 with ONNX Runtime without importing PyTorch at runtime. They are especially useful for offline video matting, VR180/SBS workflows, and first-frame-mask propagation.

Upstream model: https://github.com/pq-yang/MatAnyone2

## Model Folders

Each folder is self-contained and includes `manifest.json`.

```text
matanyone2_onnx_1024_bs1
matanyone2_onnx_1024_bs2
```

PTMediaServer currently enables only the 1024 precision tier for MatAnyone2. The 2048 export is not used because it is too slow for the offline VR workflow.

Use `bs1` for normal single-view video or sequential left/right-eye processing.

Use `bs2` when you want to process side-by-side VR180 left/right eyes as a batch of 2.

## Files

Required files:

```text
manifest.json
matanyone2_image_key.onnx
matanyone2_mask_memory.onnx
matanyone2_first_frame_refine.onnx
matanyone2_propagate.onnx
```

Optional acceleration/convenience files:

```text
matanyone2_propagate_update.onnx
matanyone2_step_update.onnx
```

`step_update` fuses image feature extraction, propagation, and memory update for following frames. If it is missing, callers can run `image_key`, `propagate`, and `mask_memory` separately.

## Input Preprocessing

Video frame input to `matanyone2_image_key.onnx`:

```text
shape: [batch, 3, height, width]
layout: RGB, CHW
dtype: float32 unless the ONNX file input says float16
range: 0.0 to 1.0
```

First-frame mask input to `matanyone2_mask_memory.onnx`:

```text
shape: [batch, 1, height, width]
dtype: same as image input
range: 0.0 background, 1.0 foreground
```

The exported height and width are fixed. Read them from `manifest.json`.

## Runtime Dependencies

GPU:

```bash
pip install onnxruntime-gpu opencv-python numpy
```

CPU smoke tests:

```bash
pip install onnxruntime opencv-python numpy
```

## Standalone Video Example

This repository includes `matanyone2_onnx_video_infer.py`, a complete example that loads all exported ONNX graphs, propagates a first-frame mask through a video, and writes a preview video.

Single-view alpha-mask output:

```bash
python matanyone2_onnx_video_infer.py \
  --model-dir matanyone2_onnx_1024_bs1 \
  --video input.mp4 \
  --mask first_frame_mask.png \
  --output-mode alpha \
  --out alpha_preview.mp4
```

Side-by-side VR180 green-background preview:

```bash
python matanyone2_onnx_video_infer.py \
  --model-dir matanyone2_onnx_1024_bs1 \
  --video sbs_vr180.mp4 \
  --mask first_frame_mask_sbs.png \
  --sbs \
  --output-mode green \
  --out green_preview.mp4
```

Use the optional batch-2 1024 model:

```bash
python matanyone2_onnx_video_infer.py \
  --model-dir matanyone2_onnx_1024_bs2 \
  --video input.mp4 \
  --mask first_frame_mask.png \
  --out alpha_1024.mp4
```

## ONNX Graph Call Order

For the first frame:

```text
image_key(image)
mask_memory(image, first_frame_mask, zero_sensory, pix_feat)
first_frame_refine(features, msk_value, obj_memory, sensory, first_frame_mask)
mask_memory(image, refined_alpha, sensory, pix_feat)
```

For following frames, prefer:

```text
step_update(image, memory_key, memory_shrinkage, msk_value, obj_memory, sensory, last_mask, last_pix_feat, last_pred_mask, last_msk_value)
```

Fallback when `step_update` is unavailable:

```text
image_key(image)
propagate(features, memory_key, memory_shrinkage, msk_value, obj_memory, sensory, last_mask, last_pix_feat, last_pred_mask, last_msk_value)
mask_memory(image, alpha, sensory, pix_feat)
```

The example script implements both paths.

## Notes

- The current exports target one foreground object.
- The models do not generate the first mask. You must provide a first-frame mask from SAM, manual annotation, chroma keying, or another detector.
- For long videos or scene changes, reset the state periodically and provide a new bootstrap mask.
- `bs2` is useful for left/right eye batching, but it also increases memory pressure.
- These ONNX files are exported from the upstream MatAnyone2 model and should be used under the upstream model's license and terms.
