MatAnyone2 ONNX models are not bundled in the release package.

Download the 512 and/or 1024 ONNX model folders from:

https://huggingface.co/zerochocobo/matanyone2_onnx/tree/main

Place the downloaded folders directly under this project's models/ directory.

Recommended for PTMediaServer:

- matanyone2_onnx_512_bs2
  Default choice for normal VR180 side-by-side offline conversion.
- matanyone2_onnx_512_bs1
  Fallback for non-SBS or lower-memory processing.
- matanyone2_onnx_1024_bs1 / matanyone2_onnx_1024_bs2
  Higher-resolution masks, slower, higher VRAM usage.

Minimum files expected in each model folder:

- manifest.json
- matanyone2_image_key.onnx
- matanyone2_mask_memory.onnx
- matanyone2_first_frame_refine.onnx
- matanyone2_propagate.onnx

Optional files used when available:

- matanyone2_propagate_update.onnx
- matanyone2_step_update.onnx

For people using the ONNX models outside PTMediaServer, see:

- examples/matanyone2_onnx_huggingface_readme.md
- examples/matanyone2_onnx_video_infer.py

Chinese:

MatAnyone2 ONNX 模型不会随主程序内置。

请从以下地址下载：

https://huggingface.co/zerochocobo/matanyone2_onnx/tree/main

下载后，将模型目录直接放到项目的 models/ 目录下。

PTMediaServer 推荐普通用户至少下载：

- matanyone2_onnx_512_bs2
- matanyone2_onnx_512_bs1

1024 版本质量更高但更慢、显存压力更大，仅建议高显存用户使用。
