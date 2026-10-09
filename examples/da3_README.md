# Depth Anything 3 ONNX

This document describes the Depth Anything 3 (DA3) ONNX export used by
PTMediaServer for image, video, and 2D-to-3D/VR depth estimation.

The exporter is [examples/da3_to_onnx.py](da3_to_onnx.py). It converts the
ONNX-friendly depth branch of DA3 Small/Base/Large into fixed-resolution ONNX
Runtime graphs so inference does not need PyTorch at runtime.

## Upstream

- Official project: <https://github.com/ByteDance-Seed/depth-anything-3>
- This exporter targets the DA3 Main Series `DA3-SMALL`, `DA3-BASE`, and
  `DA3-LARGE` checkpoints.
- The upstream model cards list `DA3-SMALL`, `DA3-BASE`, and `DA3-LARGE` as
  Apache 2.0.
- These ONNX files are derived from Depth Anything 3 weights and should be used
  according to the upstream license, model cards, and citation requirements.

## Files And Paths

```text
examples/da3_to_onnx.py
examples/da3_README.md
models/DA3/da3_small.onnx
models/DA3/da3_base.onnx
models/DA3/da3_large_1036.onnx
```

The conversion script defaults are currently:

| Purpose | Default |
| --- | --- |
| DA3 source tree | `reference/da3` if present; otherwise `G:/GIT/debug/VR_Video_Toolbox_NE/tool_2dvr/_vendor/da3` |
| PyTorch weights | `models/DA3/Small`, `Base`, and `Large` in this repository |
| ONNX output | `models/DA3` in this repository |

Use `--download` to fetch `config.json` and `model.safetensors` from the official
Hugging Face repository for each selected variant. Original checkpoints and
generated ONNX files are ignored by Git.

Canonical output names at the default size:

| File | Variant | Notes |
| --- | --- | --- |
| `da3_small.onnx` | Depth Anything 3 Small | Faster and lighter. |
| `da3_base.onnx` | Depth Anything 3 Base | Larger model, usually better depth quality. |

For non-default export sizes, the script appends the side length:

```text
models/DA3/da3_small_700.onnx
models/DA3/da3_base_1036.onnx
models/DA3/da3_large_1036.onnx
```

`Large` is shipped only at the high-detail `1036` size (`da3_large_1036.onnx`),
which is the highest-quality depth tier PTMediaServer exposes.

## Exported Graph

The converter adds a singleton view dimension and directly calls the upstream
backbone and depth head:

```python
feats, _ = net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="middle")
output = net.head(feats, height, width, patch_start_idx=0)
```

Only the depth-only sub-graph is exported:

- DINOv2 encoder, ViT-S for Small, ViT-B for Base, and ViT-L for Large.
- DualDPT depth head.
- Single-view input, `S=1`, one independent view per frame.
- Dynamic batch axis only.

This works with unmodified upstream DA3. The old `skip_camera` / `skip_sky`
arguments were specific to the former vendored copy. The loader constructs the
same model preset as the upstream API and strictly loads all safetensors weights,
without importing optional scene-export and pose-alignment dependencies.

Camera, sky, and Gaussian-splatting branches are intentionally excluded because
their `torch.quantile`, random sampling, `.item()`, and boolean-mask control
flow are not suitable for this static ONNX trace.

The script also patches DA3 RoPE's `PositionGetter` during export so
`torch.cartesian_prod` is replaced with an ONNX-exportable `meshgrid` equivalent.

## Fixed Square Input

The default input size is `518 x 518`, matching DA3's native `37 x 37` patch
grid with patch size `14`. At this size the learned positional embedding uses
the clean `npatch == N and w == h` path, avoiding bicubic position interpolation
in the exported graph.

Rules:

- Height and width are fixed in each ONNX file.
- Only the batch dimension is dynamic.
- `--size` must be a multiple of `14`.
- Re-export when you need another input side length.

## Input And Output

Default export:

```text
input name : image
input dtype: float32
input shape: [batch, 3, size, size]
layout     : RGB, CHW
range      : ImageNet-normalized float32

output name : depth
output dtype: float32
output shape: [batch, height, width]
```

Folded preprocessing export (`--fold-preprocess`):

```text
input name : image
input dtype: uint8
input shape: [batch, size, size, 3]
layout     : RGB, HWC
range      : 0..255

output name : depth
output dtype: float32
output shape: [batch, height, width]
```

The output is raw DA3 depth, which is distance-like in this project. The
2D-to-3D path treats smaller values as nearer, then resizes, inverts, clips, and
normalizes depth at runtime.

## Preprocessing

For the default float32 ONNX export:

1. Load the image or video frame as RGB.
2. Resize or letterbox to the fixed square export size.
3. Convert to float32 in the `0.0 .. 1.0` range.
4. Apply ImageNet normalization with mean `[0.485, 0.456, 0.406]` and std
   `[0.229, 0.224, 0.225]`.
5. Transpose from HWC to CHW.
6. Add the batch dimension.

For `--fold-preprocess`, the ONNX graph performs steps 3 to 5 internally. The
application still needs to resize or letterbox to `[size, size]` and pass RGB
`uint8` input.

PTMediaServer's `offline.da3_depth.Da3DepthEngine` detects the ONNX input dtype
and supports both contracts. The PyNv hot path requires the folded-preprocess
model because it uploads a `uint8` letterbox canvas.

## Runtime Dependencies

CPU:

```bash
pip install onnxruntime numpy opencv-python
```

GPU:

```bash
pip install onnxruntime-gpu numpy opencv-python
```

## Quick Inference Example

This example is for the default float32 export.

```python
import cv2
import numpy as np
import onnxruntime as ort

model = "models/DA3/da3_small.onnx"
size = 518

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

image_bgr = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_CUBIC)

x = resized.astype(np.float32) / 255.0
x = (x - mean) / std
x = np.transpose(x, (2, 0, 1))[None, ...]

session = ort.InferenceSession(
    model,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
depth = session.run(["depth"], {"image": x})[0][0]

depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]))
depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("depth.png", depth_u8)
```

For a folded-preprocess model, feed `resized[None, ...]` directly as `uint8`
RGB `[batch, size, size, 3]`.

## Convert From PyTorch Weights

Run the exporter from this repository root. The VR_Video_Toolbox_NE virtual
environment on this workstation contains the required PyTorch dependencies.
Elsewhere, use a Python environment with `torch`, `torchvision`, `numpy`, `onnx`,
`onnxruntime` (or `onnxruntime-gpu`), `safetensors`, `omegaconf`, `einops`, and
`addict`; `--download` additionally needs `huggingface_hub`.

The DA3 source is also required. `--vendor` accepts an official checkout (with
`src/depth_anything_3/`) or a source directory containing `depth_anything_3/`.
If there is no local checkout, obtain it with:

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3 reference/da3
```

Download and export the official [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL)
for PTMediaServer, including preprocessing and validation (PowerShell):

```powershell
& G:/GIT/debug/VR_Video_Toolbox_NE/.venv/Scripts/python.exe examples/da3_to_onnx.py --variant small --download --vendor reference/da3 --validate --fold-preprocess
```

The result is `models/DA3/da3_small.onnx`, with RGB uint8 input
`[batch, 518, 518, 3]` and float32 depth output `[batch, 518, 518]`.
For reproducible downloads, add `--revision e08cab65ca0ec38e7826075418411ab90cab4da3`
(the DA3-SMALL revision verified on 2026-10-08). Omit `--download` when weights
are already available locally. Omit `--fold-preprocess` for normalized float32
NCHW input.

Export both Small and Base from already downloaded weights:

```bash
G:/GIT/debug/VR_Video_Toolbox_NE/.venv/Scripts/python.exe \
  examples/da3_to_onnx.py --variant both --validate
```

Export one variant:

```bash
python examples/da3_to_onnx.py --variant small --validate
python examples/da3_to_onnx.py --variant base --validate
```

Download and export the Large high-detail model into this repo's `models/DA3`:

```bash
G:/GIT/debug/VR_Video_Toolbox_NE/.venv/Scripts/python.exe \
  examples/da3_to_onnx.py --variant large --download --size 1036 \
  --src-root models/DA3 --out-dir models/DA3 --device cuda --validate
```

Export folded-preprocess models for PTMediaServer's fast video paths:

```bash
python examples/da3_to_onnx.py --variant both --validate --fold-preprocess
```

Export another fixed input size:

```bash
python examples/da3_to_onnx.py --variant base --size 700 --validate --fold-preprocess
```

Useful options:

```text
--variant small|base|large|both
--src-root PATH       Folder containing Small/, Base/, and Large/ weight directories.
--vendor PATH         DA3 checkout or source root containing depth_anything_3/.
--download            Fetch official config.json and model.safetensors into --src-root.
--revision REVISION   Hugging Face revision for --download (default main).
--out-dir PATH        Output folder for da3_*.onnx.
--size 518            Fixed square input side. Must be a multiple of 14.
--opset 18            ONNX opset version.
--device cpu|cuda     Device used for tracing. CPU is the default.
--validate            Check ONNX and compare with PyTorch (enabled by default).
--no-validate         Skip ONNX Runtime validation.
--fold-preprocess     Export uint8 NHWC input with ImageNet normalize inside ONNX.
```

Expected weight layout:

```text
DA3/
  Small/
    config.json
    model.safetensors
  Base/
    config.json
    model.safetensors
  Large/
    config.json
    model.safetensors
```

Note: `--variant both` only exports Small and Base. Export `large` explicitly.

## Validation

When validation is enabled, the converter compares PyTorch output with ONNX
Runtime output and reports:

- output shape
- max absolute error
- mean absolute error
- relative mean error
- active ONNX Runtime providers

The script also runs `onnx.checker` and fails if the output shape differs,
either output contains non-finite values, or relative mean error exceeds `1e-2`.
Validation uses batch size 2 to exercise the dynamic batch axis.

## Notes

- The ONNX graph is depth-only; it does not export pose, camera, confidence,
  sky segmentation, or Gaussian outputs.
- These exports are fixed-shape models. Re-export for another input side.
- Use input sizes that are multiples of `14`; `518` is the safest default.
- `--fold-preprocess` changes the ONNX input contract and overwrites the same
  output filename unless you also change `--out-dir` or `--size`.
- The output is raw model depth, not metric depth.
- Smaller depth values are treated as nearer in PTMediaServer's 2D-to-3D path.

## Citation

If you use Depth Anything 3 or ONNX exports derived from it in research or a
published project, cite the upstream work:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
