Face beautification ONNX models
===============================

The offline face-beauty tool (offline/face_beauty.py, UI: Tools -> Offline Face
Beauty) downloads these into this directory on first use. Nothing needs to be
fetched by hand -- the UI shows a progress dialog -- but the direct links are
listed here so the files can be staged on an offline machine.

Every model below is an ONNX re-export of a permissively licensed upstream
project. They are hosted on Hugging Face; the mirror (hf-mirror.com) is used
automatically when the UI language is Chinese, and HF_ENDPOINT overrides both.

  file                          role         upstream / license
  ----------------------------  -----------  ------------------------------
  yunet_2023_mar.onnx           detector     YuNet, OpenCV -- MIT
  2dfan4.onnx                   landmarker   2DFAN4, breadbread1984 -- MIT
  bisenet_resnet_34.onnx        parser       BiSeNet, yakhyo -- MIT
  gfpgan_1.4.onnx               restoration  GFPGAN, TencentARC -- Apache-2.0
  gfpgan_1.3.onnx               restoration  GFPGAN, TencentARC -- Apache-2.0
  gfpgan_1.2.onnx               restoration  GFPGAN, TencentARC -- Apache-2.0
  restoreformer_plus_plus.onnx  restoration  RestoreFormer++, wzhouxiff -- Apache-2.0

Download URLs (yunet lives in the 3.4.0 bundle, everything else in 3.0.0):

  https://huggingface.co/facefusion/models-3.4.0/resolve/main/yunet_2023_mar.onnx
  https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx
  https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx
  https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
  https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.3.onnx
  https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.2.onnx
  https://huggingface.co/facefusion/models-3.0.0/resolve/main/restoreformer_plus_plus.onnx

Mirror form: replace huggingface.co with hf-mirror.com.

Or from the command line:

  python offline/face_beauty.py download --enhancer gfpgan_1.4

Note on the host repo: the ONNX conversions are distributed by the FaceFusion
project. Only these permissively licensed graphs are used -- the same repos also
host non-commercial models (inswapper, GPEN, CodeFormer, arcface, ...) which
this tool deliberately does not reference.

TensorRT engines are built from the detector / landmarker / parser into
runtime_cache/face_beauty_trt/<model>/ and are not redistributable; they are
rebuilt per machine/driver via the TensorRT dialog on the page.

The restoration models (gfpgan / restoreformer) deliberately have no TensorRT
engine: their StyleGAN-style decoder overflows in fp16 and silently returns a
darkened, washed-out face, while an fp32 engine measured slower than the CUDA
provider. They always run on CUDA.
