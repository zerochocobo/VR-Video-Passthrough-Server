MatAnyone2 ONNX models are not bundled in the release package.

Download the ONNX model folders from:

https://huggingface.co/zerochocobo/matanyone2_onnx/tree/main

Place the downloaded folders directly under this project's models/ directory.

Recommended for PTMediaServer:

- matanyone2_onnx_512_bs1
- matanyone2_onnx_1024_bs1
  Enabled offline MatAnyone2 processing precisions. 1024 remains the default.

Deprecated:

- matanyone2_onnx_512_bs2
- matanyone2_onnx_2048_bs1

Minimum files expected in each model folder:

- manifest.json
- matanyone2_image_key.onnx
- matanyone2_mask_memory.onnx
- matanyone2_first_frame_refine.onnx
- matanyone2_propagate.onnx

Optional files used when available (matanyone2_step_update.onnx is required for TensorRT cache build):

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

PTMediaServer 离线 MatAnyone2 需要下载：

- matanyone2_onnx_512_bs1
- matanyone2_onnx_1024_bs1

512 与 1024 是当前启用的 MatAnyone2 离线精度，默认仍选择 1024。2048 版本已停用。
