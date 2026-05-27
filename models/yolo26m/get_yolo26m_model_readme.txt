YOLO26m ONNX model download
===========================

This folder is used by PTMediaServer's MatAnyone2 Medium mode:

    YOLO26m -> EfficientSAM -> MatAnyone2

YOLO26m detects the person box in each sampled SBS eye frame. EfficientSAM
turns the selected box into a mask, and MatAnyone2 propagates that mask through
the segment.


Required file
-------------

Download the FP32 ONNX model and place it here with exactly this path:

- models/yolo26m/yolo26m_model.onnx

If the downloaded file has a different name, rename it to:

- yolo26m_model.onnx


Download source
---------------

ONNX model mirror:

https://huggingface.co/onnx-community/yolo26m-ONNX/tree/main/onnx

Project site:

https://github.com/ultralytics/ultralytics


Expected folder after setup
---------------------------

models/yolo26m/
  yolo26m_model.onnx
  get_yolo26m_model_readme.txt


Related required model
----------------------

MatAnyone2 Medium also needs EfficientSAM:

- models/efficientsam/efficientsam_s.onnx

See:

- models/efficientsam/get_efficientsam_model_readme.txt


Runtime notes
-------------

- Use the FP32 model as `yolo26m_model.onnx`.
- Do not use `yolo26m_model_fp16.onnx` with CUDAExecutionProvider. In local
  verification, the FP16 export silently produced near-zero person scores on
  CUDA and caused empty or unreliable masks.
- The exported ONNX graph is used at a fixed 640 letterbox input size.
- The default person score threshold in the offline tools is 0.35.
- The prepass normally runs in a child process so its CUDA context is released
  before MatAnyone2 starts.


Useful debug probes
-------------------

Inspect YOLO26m output on one real frame:

    python tools/probe_yolo26m_real.py videos/example.mp4 --time 0 --top 10

Inspect the full YOLO26m + EfficientSAM prepass decisions:

    python tools/probe_y26es_prepass.py videos/example.mp4 --duration 30 --debug-dir debug_output/y26es_real_prepass


Citation
--------

@software{yolo26_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO26},
  version = {26.0.0},
  year = {2026},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}


中文说明
========

本目录用于 PTMediaServer 的 MatAnyone2 中速模式：

    YOLO26m -> EfficientSAM -> MatAnyone2

YOLO26m 负责在采样到的 SBS 左右眼画面中检测人物框。EfficientSAM 会把选中的
人物框转换成遮罩，随后 MatAnyone2 在后续片段中传播该遮罩。


必须下载的文件
--------------

请下载 FP32 ONNX 模型，并放到下面这个位置，文件名必须保持一致：

- models/yolo26m/yolo26m_model.onnx

如果下载下来的文件名不同，请重命名为：

- yolo26m_model.onnx


下载地址
--------

ONNX 模型镜像：

https://huggingface.co/onnx-community/yolo26m-ONNX/tree/main/onnx

项目主页：

https://github.com/ultralytics/ultralytics


配置完成后的目录结构
--------------------

models/yolo26m/
  yolo26m_model.onnx
  get_yolo26m_model_readme.txt


相关必需模型
------------

MatAnyone2 中速模式还需要 EfficientSAM：

- models/efficientsam/efficientsam_s.onnx

请参考：

- models/efficientsam/get_efficientsam_model_readme.txt


运行说明
--------

- 请使用 FP32 模型，并命名为 `yolo26m_model.onnx`。
- 不要在 CUDAExecutionProvider 下使用 `yolo26m_model_fp16.onnx`。本项目本地验证中，
  FP16 导出会静默产生接近 0 的人物分数，导致遮罩为空或不稳定。
- 当前 ONNX 图按固定 640 letterbox 输入尺寸使用。
- 离线工具默认的人物分数阈值是 0.35。
- 预处理通常会在子进程中运行，以便在 MatAnyone2 启动前释放 YOLO26m +
  EfficientSAM 使用过的 CUDA 上下文。


调试工具
--------

查看 YOLO26m 在真实帧上的输出：

    python tools/probe_yolo26m_real.py videos/example.mp4 --time 0 --top 10

查看完整 YOLO26m + EfficientSAM 预处理决策：

    python tools/probe_y26es_prepass.py videos/example.mp4 --duration 30 --debug-dir debug_output/y26es_real_prepass
