# NVIDIA RTX Video SDK v1.1.0 Programming Guide Review

Date: 2026-07-18

The bundled Programming Guide does **not** state that VSR input is restricted to 360p–1440p, and it does not state that only 2D video is supported.

Pages 46–51 define VSR as taking an RGB SDR GPU surface and upscaling, sharpening, and deblocking it into an RGB output surface. Evaluation parameters are input/output resources, source/destination rectangles, and quality level 0–4. There is no resolution enum, 360p/1440p boundary, VR layout, projection, eye metadata, or fixed scale factor.

The guide shows 480p-to-1920×1080 output and a 1080p VSR+TrueHDR input. Its suggestion to downscale high-resolution video appears in sample raw-file preparation instructions, not as an SDK evaluate limit.

The guide explicitly states Turing RTX 20xx+ GPUs and driver 550.58+; the CUDA build path requires CUDA Toolkit 12.8+.

Therefore 360p–1440p may be used as PTMediaServer's conservative initial product policy, but it must not be described as a hard limit stated by this SDK PDF. VR and 4K VR remain outside the first release because the guide does not cover or guarantee them, not because it explicitly forbids them. The native POC must test 360p through 2160p and non-16:9 inputs. Output naming remains `[SuperRes]`.
