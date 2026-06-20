# NVDS Temporal Depth Stabilizer ONNX

This document describes the NVDS (Neural Video Depth Stabilizer) ONNX export used
by PTMediaServer as an optional temporal depth stabilizer in the offline
2D-to-3D/VR pipeline.

The exporter is [examples/export_nvds_onnx.py](export_nvds_onnx.py). It converts
the NVDS PyTorch checkpoint into fixed-resolution ONNX Runtime graphs so
inference does not need PyTorch at runtime. NVDS smooths the per-frame depth that
DA3 produces, removing the temporal "depth swimming" that causes discomfort when
the stereo output is viewed.

## Upstream

- Official project: <https://github.com/RaymondWang987/NVDS>
- This exporter targets the original `NVDS_Stabilizer.pth` checkpoint (a SegFormer
  MiT-B5 backbone plus a focal cross-attention stabilizer head).
- These ONNX files are derived from NVDS weights and should be used according to
  the upstream license and citation requirements.

## Files And Paths

```text
examples/export_nvds_onnx.py
examples/nvds_README.md
models/NVDS/NVDS_Stabilizer.pth            (input checkpoint, ~354 MB)
models/NVDS/NVDS_Stabilizer_672x384.onnx   (monolithic export)
models/NVDS/NVDS_Backbone_512x288.onnx     (split: per-frame backbone)
models/NVDS/NVDS_Head_512x288.onnx         (split: cross-frame head)
models/NVDS/NVDS_Backbone_672x384.onnx
models/NVDS/NVDS_Head_672x384.onnx
```

The conversion script defaults are currently:

| Purpose | Default |
| --- | --- |
| NVDS source tree | `reference/NVDS` in this repository |
| PyTorch checkpoint | `models/NVDS/NVDS_Stabilizer.pth` |
| ONNX output | `models/NVDS` in this repository |

Resolution tiers PTMediaServer uses (both 16:9):

| Tier | Width x Height | Notes |
| --- | --- | --- |
| Fast (default) | `512 x 288` | ~1.76x faster than 672x384; the UI default. |
| High quality | `672 x 384` | Sharper depth boundaries; reachable via `--nvds-res 672x384`. |

## Export Modes

The exporter has two modes.

### Monolithic (`--width/--height`, no `--split`)

A single graph that takes the whole 4-frame window and runs the backbone on all
4 frames internally. Output name `NVDS_Stabilizer_{w}x{h}.onnx`.

### Split (`--split`, recommended)

Two graphs that let the runtime run the heavy backbone once per frame and cache
the last 4 results, instead of recomputing 3/4 of it for every sliding window:

| File | Role |
| --- | --- |
| `NVDS_Backbone_{w}x{h}.onnx` | Single RGBD frame -> 4 multi-scale feature maps. Run once per frame and cached. |
| `NVDS_Head_{w}x{h}.onnx` | The 4 window frames' features (stacked) + last-frame RGB -> stabilized depth. |

Both modes are numerically identical (the backbone has no cross-frame ops, so its
features are batch-independent). Measured `max_abs_diff` between split and
monolith is about `3e-05`.

Note on performance: the focal cross-attention head, not the backbone, dominates
NVDS cost, so the split's main practical benefit is enabling the lower-resolution
tier (the head cost scales with input pixel count). See
`summary/summary_20260620_NVDS_INTEGRATION_EXTERNAL_REVIEW_CN.md` for the
measurements.

## Fixed Resolution

Each ONNX file is exported at a fixed input resolution.

Rules:

- Height and width are fixed in each ONNX file.
- `--width` and `--height` must each be a multiple of `32`.
- The intended tiers are `512 x 288` and `672 x 384` (16:9). Re-export to add
  another size.

## Input And Output

### Monolithic graph

```text
input name : rgbd_seq
input dtype: float32
input shape: [1, 4, 4, height, width]   (batch, time=4, channels=RGB+near, H, W)

output name : stabilized_depth
output dtype: float32
output shape: [1, 1, height, width]
```

### Split graphs

Backbone:

```text
input name : frame_rgbd
input shape: [1, 4, height, width]       (one RGBD frame: RGB + near)

output names: feat0, feat1, feat2, feat3 (multi-scale features, batch 1)
```

Head:

```text
input names: feat0..feat3                (each [4, C, h, w] = the 4 window frames stacked)
             last_rgb                     ([1, 3, height, width], the current frame's RGB)

output name : stabilized_depth
output shape: [1, 1, height, width]
```

The output is a stabilized normalized near/disparity map (larger = nearer),
already temporally smoothed. Unlike raw DA3 depth, it must NOT be reciprocated or
percentile-normalized again at render time; PTMediaServer feeds it through the
dedicated `render_near(...)` path.

## Preprocessing

For every frame, the runtime wrapper builds a 4-channel RGBD frame at the export
resolution:

1. Load the frame as RGB and resize to `[width, height]`.
2. Convert to float32 in `0.0 .. 1.0` and apply ImageNet normalization with mean
   `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`; transpose to CHW.
3. Convert the DA3 depth to a normalized near/disparity map (reciprocal +
   percentile normalization) and resize it to `[width, height]`.
4. Concatenate RGB (3 ch) + near (1 ch) into a `[4, height, width]` frame.

For the monolithic graph the wrapper keeps the last 4 such frames, padding the
first frames by repetition, and stacks them into `[1, 4, 4, H, W]`. For the split
graphs it runs the backbone on each frame, caches the last 4 feature tuples,
stacks them on the batch axis, and passes the current frame's RGB as `last_rgb`.

PTMediaServer's `offline.nvds_stabilizer.NvdsDepthStabilizer` performs all of
this and auto-selects the split graphs when they are present.

## Runtime Dependencies

CPU:

```bash
pip install onnxruntime numpy opencv-python
```

GPU:

```bash
pip install onnxruntime-gpu numpy opencv-python
```

TensorRT is not usable for NVDS (see Notes); the runtime uses the CUDA execution
provider with a bounded GPU memory arena.

## Quick Inference Example

This example runs the split graphs over a 4-frame window. `frame_rgbd_t` is a
preprocessed `[1, 4, H, W]` float32 frame as described in Preprocessing.

```python
import numpy as np
import onnxruntime as ort

W, H = 512, 288
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
backbone = ort.InferenceSession("models/NVDS/NVDS_Backbone_512x288.onnx", providers=providers)
head = ort.InferenceSession("models/NVDS/NVDS_Head_512x288.onnx", providers=providers)

bb_out = [o.name for o in backbone.get_outputs()]          # feat0..feat3
window = []                                                # last 4 feature tuples
for frame_rgbd_t in stream:                                # each [1, 4, H, W] float32
    feats = backbone.run(bb_out, {"frame_rgbd": frame_rgbd_t})
    window.append(feats)
    window = window[-4:]
    pad = [window[0]] * (4 - len(window)) + window         # causal padding
    feeds = {f"feat{s}": np.concatenate([pad[t][s] for t in range(4)], axis=0)
             for s in range(4)}
    feeds["last_rgb"] = np.ascontiguousarray(frame_rgbd_t[:, 0:3])
    stable_near = head.run(["stabilized_depth"], feeds)[0][0, 0]   # [H, W]
```

## Convert From PyTorch Weights

Run the exporter from this repository root. The VR_Video_Toolbox_NE virtual
environment already contains the PyTorch dependencies (the runtime venv is
ONNX-only and has no torch):

```bash
G:/GIT/debug/VR_Video_Toolbox_NE/.venv/Scripts/python.exe \
  examples/export_nvds_onnx.py --split --width 512 --height 288 --device cuda
```

Export the high-quality tier:

```bash
python examples/export_nvds_onnx.py --split --width 672 --height 384 --device cuda
```

Export the monolithic graph instead of the split pair:

```bash
python examples/export_nvds_onnx.py --width 672 --height 384 --device cuda
```

Useful options:

```text
--split               Export backbone + head graphs instead of the monolith.
--width 672           Input width. Must be a multiple of 32.
--height 384          Input height. Must be a multiple of 32.
--source-root PATH    Vendored NVDS source root (contains full_model.py).
--checkpoint PATH     NVDS_Stabilizer.pth.
--output PATH         Output path for the monolithic export.
--opset 17            ONNX opset version.
--device cpu|cuda     Device used for tracing.
--dynamic-batch       Mark only the batch axis dynamic (monolithic only).
--skip-ort-check      Skip the ONNX Runtime comparison.
```

Expected checkpoint layout:

```text
NVDS/
  NVDS_Stabilizer.pth
```

## Validation

The exporter compares outputs with ONNX Runtime and reports a max/mean absolute
difference. In `--split` mode it runs the per-frame backbone plus head over a
4-frame window and compares against the monolithic PyTorch forward, so a single
run verifies that the split is equivalent to the original model.

## Notes

- TensorRT cannot run this graph: the optimized model exceeds the 2 GB protobuf
  limit and the graph contains many `ScatterND` / dynamic ops. The runtime treats
  a `trt` request for NVDS as CUDA and bounds the CUDA arena with `gpu_mem_limit`
  (DA3 still uses TensorRT normally).
- NVDS is VRAM-heavy. Sharing the GPU with DA3 and the renderer pushes the
  combined working set close to 16 GB; the bounded arena keeps it from spilling
  into Windows shared memory.
- NVDS is limited to 16:9 input in this project.
- The output is a stabilized near/disparity map and must be rendered through the
  `render_near(...)` path, not the raw-depth path.
- The split graphs are preferred and auto-detected at runtime; the monolith is a
  fallback when the split files are absent.

## Citation

If you use NVDS or ONNX exports derived from it, cite the upstream work (verify
against the upstream repository for the authoritative entry):

```bibtex
@InProceedings{Wang_2023_ICCV,
  author    = {Wang, Yiran and Pan, Zhiyu and Li, Xingyi and Cao, Zhiguo and Xian, Ke and Zhang, Jianming},
  title     = {Neural Video Depth Stabilizer},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```
