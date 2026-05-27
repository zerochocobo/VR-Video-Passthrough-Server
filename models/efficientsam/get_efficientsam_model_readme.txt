EfficientSAM ONNX model download
================================

This folder is used by PTMediaServer's MatAnyone2 Medium mode:

    YOLO26m -> EfficientSAM -> MatAnyone2

YOLO26m detects the person box. EfficientSAM turns that box into a mask, and
MatAnyone2 then propagates the mask through the segment.


Required file
-------------

Download this ONNX file:

- efficientsam_s.onnx

Place it here with exactly this path:

- models/efficientsam/efficientsam_s.onnx


Download source
---------------

ONNX model mirror:

https://huggingface.co/yunyangx/EfficientSAM/tree/main

Official project:

https://github.com/yformer/EfficientSAM


Expected folder after setup
---------------------------

models/efficientsam/
  efficientsam_s.onnx
  get_efficientsam_model_readme.txt


Related required model
----------------------

MatAnyone2 Medium also needs the YOLO26m ONNX model:

- models/yolo26m/yolo26m_model.onnx

See:

- models/yolo26m/get_yolo26m_model_readme.txt


Runtime notes
-------------

- The current prepass is YOLO26m + EfficientSAM.
- EfficientSAM runs only during the prepass that creates bootstrap masks for
  MatAnyone2. MatAnyone2 performs the later mask propagation.


Citation
--------

@article{xiong2023efficientsam,
  title={EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything},
  author={Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra},
  journal={arXiv:2312.00863},
  year={2023}
}


中文说明
========

本目录用于 PTMediaServer 的 MatAnyone2 中速模式：

    YOLO26m -> EfficientSAM -> MatAnyone2

YOLO26m 负责检测人物框。EfficientSAM 会把人物框转换成遮罩，随后交给
MatAnyone2 在后续片段中传播遮罩。


必须下载的文件
--------------

请下载以下 ONNX 文件：

- efficientsam_s.onnx

并放到下面这个位置，文件名必须保持一致：

- models/efficientsam/efficientsam_s.onnx


下载地址
--------

ONNX 模型镜像：

https://huggingface.co/yunyangx/EfficientSAM/tree/main

官方项目：

https://github.com/yformer/EfficientSAM


配置完成后的目录结构
--------------------

models/efficientsam/
  efficientsam_s.onnx
  get_efficientsam_model_readme.txt


相关必需模型
------------

MatAnyone2 中速模式还需要 YOLO26m ONNX 模型：

- models/yolo26m/yolo26m_model.onnx

请参考：

- models/yolo26m/get_yolo26m_model_readme.txt


运行说明
--------

- 当前预处理链路是 YOLO26m + EfficientSAM。
- EfficientSAM 只负责在预处理阶段生成 MatAnyone2 的引导遮罩；后续遮罩传播由
  MatAnyone2 完成。
