"""Realtime mosaic-restoration ("RM") inference core.

Two ONNX models, both run through onnxruntime (the project does not depend on
ultralytics, so YOLO post-processing is implemented here):

  - detector    models/demosaic/vr_mosaic_detection_model_v2_accurate.onnx
                YOLO11m-seg: input 'images' (1,3,640,640) ->
                output0 (1, 4+nc+32, 8400), output1 (1, 32, 160, 160) mask protos
  - restoration models/demosaic/vr_mosaic_restoration_windows_model_v0.1.onnx
                sliding window: input 'frames' (1,7,3,256,256) RGB 0..1 ->
                output 'restored' (1,3,256,256) restored CENTER frame

TensorRT engine caching mirrors the DA3 depth engine (see offline/da3_depth.py):
the TensorRT EP builds an fp16 engine on first use and caches it under
runtime_cache/demosaic_trt/{detector,restoration}. The restoration graph contains
grid_sample (optical-flow warp) which needs TensorRT >= 8.5; if that EP fails to
build it transparently falls back to CUDA then CPU.

The pipeline (detect on the CENTER frame, crop each region across the temporal
window, restore, feather-blend back) follows the reference at
G:\\GIT\\lada\\realtime_inference_reference.py.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

import config

WINDOW = 7              # temporal window of the restoration model
CENTER = WINDOW // 2    # index of the restored frame within the window (== 3)
REST_SIZE = 256         # restoration works on 256x256 region crops
DET_SIZE = 640          # detector input side (overridden from the ONNX shape)
PROTO_DIM = 32          # mask prototype channels

_DETECTOR_FILE = "vr_mosaic_detection_model_v2_accurate.onnx"
_RESTORATION_FILE = "vr_mosaic_restoration_windows_model_v0.1.onnx"


def _models_dir() -> Path:
    return config.ROOT / "models" / "demosaic"


def detector_model_path() -> Path:
    return _models_dir() / _DETECTOR_FILE


def restoration_model_path() -> Path:
    return _models_dir() / _RESTORATION_FILE


def _trt_cache_dir(name: str) -> Path:
    path = config.ROOT / "runtime_cache" / "demosaic_trt" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def rm_trt_cached() -> bool:
    """True if TensorRT engines have been built+cached for both RM models."""
    base = config.ROOT / "runtime_cache" / "demosaic_trt"
    for name in ("detector", "restoration"):
        cache = base / name
        if not (cache.is_dir() and any(cache.glob("*.engine"))):
            return False
    return True


def _onnx_providers(provider: str, cache_name: str) -> list:
    """Provider chain. ``trt`` puts the TensorRT EP (fp16 + cached engine) in
    front of CUDA, then CPU. Mirrors offline.da3_depth.onnx_providers."""
    available = set(ort.get_available_providers())
    chain: list = []
    if provider == "trt" and "TensorrtExecutionProvider" in available:
        chain.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(_trt_cache_dir(cache_name)),
                "trt_timing_cache_enable": True,
            },
        ))
    if provider in ("trt", "cuda") and "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


def _make_session(model_path: Path, provider: str, cache_name: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=_onnx_providers(provider, cache_name))


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to size x size (centered, gray 114).
    Returns (canvas, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


class Detection:
    __slots__ = ("box", "score", "coeffs")

    def __init__(self, box: tuple[int, int, int, int], score: float, coeffs: np.ndarray):
        self.box = box          # (x1, y1, x2, y2) in original frame coords
        self.score = score
        self.coeffs = coeffs    # (32,) mask coefficients


class DemosaicDetector:
    """YOLO11m-seg mosaic detector running on onnxruntime."""

    def __init__(self, provider: str = "trt"):
        self.sess = _make_session(detector_model_path(), provider, "detector")
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        try:
            self.size = int(shape[2]) if isinstance(shape[2], int) else DET_SIZE
        except Exception:
            self.size = DET_SIZE
        self.output_names = [o.name for o in self.sess.get_outputs()]

    @property
    def providers(self) -> list:
        return self.sess.get_providers()

    def detect(self, frame_rgb: np.ndarray, conf: float = 0.25,
               iou: float = 0.5) -> tuple[list[Detection], np.ndarray]:
        """Detect mosaic regions on a single RGB frame.

        Returns (detections, protos) where protos is (32,160,160). Boxes are in
        original-frame pixel coords; mask coeffs are paired with protos so a
        per-region mask can be built lazily (see segmentation_mask)."""
        H, W = frame_rgb.shape[:2]
        canvas, scale, pad_x, pad_y = _letterbox(frame_rgb, self.size)
        return self.detect_blob(canvas, scale, pad_x, pad_y, (H, W), conf, iou)

    def detect_blob(self, canvas_u8: np.ndarray, scale: float, pad_x: int, pad_y: int,
                    frame_hw: tuple[int, int], conf: float = 0.25,
                    iou: float = 0.5) -> tuple[list[Detection], np.ndarray]:
        """Run the detector on a pre-letterboxed (size x size x 3) uint8 canvas.

        Lets the caller do the letterbox on the GPU and hand over just the small
        square canvas (the GPU path), skipping the host-side cv2 resize of the
        full frame."""
        blob = canvas_u8.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out0, out1 = self.sess.run(self.output_names,
                                   {self.input_name: np.ascontiguousarray(blob)})
        return _postprocess_detections(out0, out1, scale, pad_x, pad_y, frame_hw, conf, iou)


def _postprocess_detections(out0: np.ndarray, out1: np.ndarray, scale: float,
                            pad_x: int, pad_y: int, frame_hw: tuple[int, int],
                            conf: float, iou: float) -> tuple[list[Detection], np.ndarray]:
    """Decode YOLO11-seg raw outputs to (detections, protos). Shared by the CPU
    and GPU (IOBinding) detector paths."""
    H, W = frame_hw
    preds = out0[0].T                      # (8400, 4+nc+32)
    protos = out1[0]                       # (32, 160, 160)
    nc = preds.shape[1] - 4 - PROTO_DIM
    boxes_xywh = preds[:, 0:4]
    cls_scores = preds[:, 4:4 + nc]
    coeffs = preds[:, 4 + nc:4 + nc + PROTO_DIM]
    scores = cls_scores.max(axis=1)
    keep = scores >= conf
    if not np.any(keep):
        return [], protos
    boxes_xywh, scores, coeffs = boxes_xywh[keep], scores[keep], coeffs[keep]
    # xywh (letterbox space) -> xyxy (original frame space)
    cx, cy, bw, bh = boxes_xywh.T
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    idx = _nms(boxes_xyxy, scores, iou)
    dets: list[Detection] = []
    for i in idx:
        bx1 = max(0, int(round(boxes_xyxy[i, 0])))
        by1 = max(0, int(round(boxes_xyxy[i, 1])))
        bx2 = min(W, int(round(boxes_xyxy[i, 2])))
        by2 = min(H, int(round(boxes_xyxy[i, 3])))
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        dets.append(Detection((bx1, by1, bx2, by2), float(scores[i]), coeffs[i]))
    return dets, protos


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Class-agnostic greedy NMS (matches the reference, which ignores class)."""
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[ovr <= iou_thr]
    return keep


def segmentation_mask(det: Detection, protos: np.ndarray, frame_hw: tuple[int, int],
                      det_size: int) -> np.ndarray:
    """Feathered float mask (H_box, W_box, 1) for one detection's box region."""
    H, W = frame_hw
    x1, y1, x2, y2 = det.box
    scale = min(det_size / W, det_size / H)
    pad_x = (det_size - int(round(W * scale))) // 2
    pad_y = (det_size - int(round(H * scale))) // 2
    ph, pw = protos.shape[1], protos.shape[2]            # 160, 160
    s = ph / det_size                                    # mask-space per letterbox px
    # box -> mask(160) space
    mx1 = int(np.floor((x1 * scale + pad_x) * s))
    my1 = int(np.floor((y1 * scale + pad_y) * s))
    mx2 = int(np.ceil((x2 * scale + pad_x) * s))
    my2 = int(np.ceil((y2 * scale + pad_y) * s))
    mx1, my1 = max(0, mx1), max(0, my1)
    mx2, my2 = min(pw, max(mx1 + 1, mx2)), min(ph, max(my1 + 1, my2))
    proto_crop = protos[:, my1:my2, mx1:mx2].reshape(PROTO_DIM, -1)
    m = 1.0 / (1.0 + np.exp(-(det.coeffs @ proto_crop)))
    m = m.reshape(my2 - my1, mx2 - mx1).astype(np.float32)
    m = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur((m > 0.5).astype(np.float32), (5, 5), 0)
    return m[..., None]


class DemosaicRestorer:
    """Sliding-window restoration generator running on onnxruntime."""

    def __init__(self, provider: str = "trt"):
        self.sess = _make_session(restoration_model_path(), provider, "restoration")
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    @property
    def providers(self) -> list:
        return self.sess.get_providers()

    def restore(self, window_crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """window_crops_rgb: WINDOW RGB uint8 crops (any HxW). Returns the
        restored CENTER crop as RGB uint8 at REST_SIZE x REST_SIZE."""
        arr = np.stack([
            cv2.resize(c, (REST_SIZE, REST_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            for c in window_crops_rgb
        ])                                          # (7, 256, 256, 3)
        arr = arr.transpose(0, 3, 1, 2)[None]       # (1, 7, 3, 256, 256)
        return self.restore_stack(arr)

    def restore_stack(self, frames_nchw: np.ndarray) -> np.ndarray:
        """frames_nchw: (1,7,3,256,256) float32 0..1 (already resized). Returns the
        restored CENTER crop as RGB uint8 256x256. Lets the GPU path resize the
        crops itself and skip the host-side cv2.resize loop."""
        out = self.sess.run([self.output_name],
                            {self.input_name: np.ascontiguousarray(frames_nchw, dtype=np.float32)})[0]
        out = np.clip(out[0], 0.0, 1.0).transpose(1, 2, 0)
        return (out * 255.0).astype(np.uint8)


def restore_center_frame(window_frames_rgb: Sequence[np.ndarray],
                         detector: DemosaicDetector, restorer: DemosaicRestorer,
                         conf: float = 0.25) -> np.ndarray:
    """Detect mosaics on the CENTER frame and restore each region using the
    temporal window. window_frames_rgb: WINDOW consecutive RGB uint8 frames.
    Returns the restored CENTER frame (RGB uint8). Mirrors the reference."""
    center = window_frames_rgb[CENTER]
    H, W = center.shape[:2]
    out = center.copy()
    dets, protos = detector.detect(center, conf=conf)
    if not dets:
        return out
    for det in dets:
        x1, y1, x2, y2 = det.box
        crops = [f[y1:y2, x1:x2] for f in window_frames_rgb]
        restored = restorer.restore(crops)
        restored = cv2.resize(restored, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        m = segmentation_mask(det, protos, (H, W), detector.size)
        region = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = (region * (1 - m) + restored.astype(np.float32) * m).astype(np.uint8)
    return out


class DemosaicEngines:
    """Bundle of detector + restorer sharing one provider preference."""

    def __init__(self, provider: str = "trt"):
        self.detector = DemosaicDetector(provider)
        self.restorer = DemosaicRestorer(provider)

    @property
    def providers(self) -> list:
        return self.detector.providers


# --- GPU (cupy) pipeline -----------------------------------------------------
# Bilinear resize sampling an arbitrary crop region of a contiguous HWC uint8
# source (offset + row stride), writing a contiguous (dh,dw,3) destination. One
# kernel serves both the full-frame detector letterbox and per-region crops, so
# crops never need a contiguous copy first.
_RESIZE_KERNEL_SRC = r"""
extern "C" __global__ void bilinear_resize_u8(
    const unsigned char* src, unsigned char* dst,
    int src_stride_w, int src_x0, int src_y0, int crop_w, int crop_h,
    int dh, int dw) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float fx = (x + 0.5f) * crop_w / dw - 0.5f;
  float fy = (y + 0.5f) * crop_h / dh - 0.5f;
  int x0 = (int)floorf(fx); int y0 = (int)floorf(fy);
  float ax = fx - x0; float ay = fy - y0;
  int x1 = x0 + 1; int y1 = y0 + 1;
  if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
  if (x1 > crop_w - 1) x1 = crop_w - 1; if (y1 > crop_h - 1) y1 = crop_h - 1;
  if (x0 > crop_w - 1) x0 = crop_w - 1; if (y0 > crop_h - 1) y0 = crop_h - 1;
  const unsigned char* s00 = src + ((src_y0 + y0) * src_stride_w + (src_x0 + x0)) * 3;
  const unsigned char* s01 = src + ((src_y0 + y0) * src_stride_w + (src_x0 + x1)) * 3;
  const unsigned char* s10 = src + ((src_y0 + y1) * src_stride_w + (src_x0 + x0)) * 3;
  const unsigned char* s11 = src + ((src_y0 + y1) * src_stride_w + (src_x0 + x1)) * 3;
  unsigned char* d = dst + (y * dw + x) * 3;
  for (int c = 0; c < 3; c++) {
    float v = s00[c]*(1-ax)*(1-ay) + s01[c]*ax*(1-ay)
            + s10[c]*(1-ax)*ay     + s11[c]*ax*ay;
    d[c] = (unsigned char)(v + 0.5f);
  }
}
"""


class GpuRmProcessor:
    """GPU-resident RM pipeline: takes a temporal window of cupy RGB frames and
    returns the restored CENTER frame as a cupy RGB array.

    Crop, resize and blend all run on the GPU; only the small detector blob
    (size x size) and the restoration crop window (7 x 256 x 256) cross PCIe to
    onnxruntime, instead of downloading every full frame to host as the numpy
    path does. The detector mask is still built on the host (box sized, cheap)
    and uploaded for blending."""

    def __init__(self, engines: DemosaicEngines, detect_interval: int = 1):
        import cupy as cp

        self.cp = cp
        self.engines = engines
        self.det_size = engines.detector.size
        self._resize = cp.RawModule(code=_RESIZE_KERNEL_SRC).get_function("bilinear_resize_u8")
        self._block = (16, 16, 1)
        # Detector frequency reduction: run detection every ``detect_interval``
        # frames and reuse the cached (box, gpu-mask) pairs in between. Restoration
        # always runs on the fresh temporal window, so only the box location is
        # held stale for up to detect_interval-1 frames.
        self.detect_interval = max(1, int(detect_interval))
        self._frame_count = 0
        self._cached_regions: list[tuple[tuple[int, int, int, int], "cp.ndarray"]] | None = None
        # ORT IOBinding lets both models read their input straight from GPU memory,
        # so the float32 /255 + transpose runs on the GPU (cupy) and nothing is
        # downloaded to host for preprocessing. The detector outputs still come to
        # host for NMS (small); the restorer output stays on the GPU.
        self._det_io = engines.detector.sess.io_binding()
        self._rest_io = engines.restorer.sess.io_binding()
        self._rest_out_g = cp.empty((1, 3, REST_SIZE, REST_SIZE), cp.float32)
        self._stream = cp.cuda.get_current_stream()

    def _run_detector_gpu(self, canvas_g, scale, pad_x, pad_y, frame_hw, conf):
        """Normalize the letterbox canvas on the GPU, run the detector via
        IOBinding (input on device), return decoded detections + protos."""
        cp = self.cp
        det = self.engines.detector
        blob_g = cp.ascontiguousarray(
            (canvas_g.astype(cp.float32) / 255.0).transpose(2, 0, 1)[None])  # (1,3,S,S)
        self._stream.synchronize()
        io = self._det_io
        io.bind_input(det.input_name, "cuda", 0, np.float32, blob_g.shape, int(blob_g.data.ptr))
        for name in det.output_names:
            io.bind_output(name)                       # host outputs (small, for NMS)
        det.sess.run_with_iobinding(io)
        out0, out1 = io.copy_outputs_to_cpu()
        return _postprocess_detections(out0, out1, scale, pad_x, pad_y, frame_hw, conf, 0.5)

    def _run_restore_gpu(self, stack_g):
        """stack_g: (WINDOW,256,256,3) uint8 cupy. Normalize on GPU, run via
        IOBinding (input + output on device), return restored (256,256,3) uint8 cupy."""
        cp = self.cp
        res = self.engines.restorer
        x = cp.ascontiguousarray(
            (stack_g.astype(cp.float32) / 255.0).transpose(0, 3, 1, 2)[None])  # (1,7,3,256,256)
        self._stream.synchronize()
        io = self._rest_io
        io.bind_input(res.input_name, "cuda", 0, np.float32, x.shape, int(x.data.ptr))
        io.bind_output(res.output_name, "cuda", 0, np.float32,
                       self._rest_out_g.shape, int(self._rest_out_g.data.ptr))
        res.sess.run_with_iobinding(io)
        self._stream.synchronize()
        out = cp.clip(self._rest_out_g[0], 0.0, 1.0).transpose(1, 2, 0)        # (256,256,3) f32
        return (out * 255.0).astype(cp.uint8)

    def _resize_into(self, src_g, dst_g, src_x0: int, src_y0: int, crop_w: int, crop_h: int) -> None:
        cp = self.cp
        dh, dw = int(dst_g.shape[0]), int(dst_g.shape[1])
        src_stride_w = int(src_g.shape[1])
        grid = ((dw + 15) // 16, (dh + 15) // 16, 1)
        self._resize(
            grid, self._block,
            (src_g, dst_g, np.int32(src_stride_w), np.int32(src_x0), np.int32(src_y0),
             np.int32(crop_w), np.int32(crop_h), np.int32(dh), np.int32(dw)),
        )

    def _detect_regions(self, center_g, conf: float):
        """Run the detector on the center frame and cache (box, gpu-mask) pairs."""
        cp = self.cp
        H, W = int(center_g.shape[0]), int(center_g.shape[1])
        size = self.det_size
        scale = min(size / W, size / H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2

        # letterbox the center frame to size x size on the GPU (no host download:
        # normalization + inference input stay on device via IOBinding)
        canvas_g = cp.full((size, size, 3), 114, cp.uint8)
        sub = canvas_g[pad_y:pad_y + nh, pad_x:pad_x + nw]
        tmp = cp.empty((nh, nw, 3), cp.uint8)
        self._resize_into(center_g, tmp, 0, 0, W, H)
        sub[...] = tmp

        dets, protos = self._run_detector_gpu(canvas_g, scale, pad_x, pad_y, (H, W), conf)
        regions: list[tuple[tuple[int, int, int, int], "cp.ndarray"]] = []
        for det in dets:
            mask = segmentation_mask(det, protos, (H, W), size)       # (bh,bw,1) float host
            regions.append((det.box, cp.asarray(mask)))
        return regions

    def process(self, window_g, conf: float = 0.25):
        cp = self.cp
        center_g = window_g[CENTER]

        if self._cached_regions is None or self._frame_count % self.detect_interval == 0:
            self._cached_regions = self._detect_regions(center_g, conf)
        self._frame_count += 1

        out_g = center_g.copy()
        for box, m_g in self._cached_regions:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            # crop+resize each window frame to 256 on the GPU; normalize + restore
            # via IOBinding so nothing leaves the GPU (restored stays on device)
            stack_g = cp.empty((WINDOW, REST_SIZE, REST_SIZE, 3), cp.uint8)
            for k in range(WINDOW):
                self._resize_into(window_g[k], stack_g[k], x1, y1, bw, bh)
            restored_g = self._run_restore_gpu(stack_g)               # (256,256,3) uint8 cupy

            # resize restored crop back to box on GPU, blend with the cached mask
            resized_g = cp.empty((bh, bw, 3), cp.uint8)
            self._resize_into(restored_g, resized_g, 0, 0, REST_SIZE, REST_SIZE)
            region = out_g[y1:y2, x1:x2].astype(cp.float32)
            out_g[y1:y2, x1:x2] = (region * (1 - m_g) + resized_g.astype(cp.float32) * m_g).astype(cp.uint8)
        return out_g


def models_available() -> bool:
    return detector_model_path().is_file() and restoration_model_path().is_file()


def warmup_rm_engines(provider: str = "trt", log=print) -> DemosaicEngines:
    """Build/load both engines and run one dummy inference each so the (slow)
    first-call TensorRT build happens here. Mirrors da3_depth.warmup_depth_engine."""
    engines = DemosaicEngines(provider)
    log(f"[rm] detector providers={engines.detector.providers}")
    log(f"[rm] restorer providers={engines.restorer.providers}")
    dummy = np.zeros((engines.detector.size, engines.detector.size, 3), np.uint8)
    engines.detector.detect(dummy, conf=0.99)
    engines.restorer.restore([np.zeros((REST_SIZE, REST_SIZE, 3), np.uint8)] * WINDOW)
    return engines


# --- shared engine singleton -------------------------------------------------
# One DemosaicEngines instance is shared process-wide. ORT InferenceSession.run
# is thread-safe, so concurrent [RM] streams reuse the same sessions instead of
# each rebuilding/loading their own. The lock is held for the whole (slow) build
# so a background prewarm and a playback request never build the TRT engine
# twice into the same cache dir at once -- the second caller waits, then reuses.
_engines_lock = threading.Lock()
_engines: "DemosaicEngines | None" = None
_engines_provider = "trt"


def get_shared_engines(provider: str | None = None, log=print) -> "DemosaicEngines":
    """Return the process-wide engine singleton, building it on first call.

    Blocks if another thread is mid-build (e.g. a startup/toggle prewarm), then
    returns the already-built instance. The first build performs the (minutes
    long) TensorRT engine compilation; subsequent calls are immediate."""
    global _engines
    with _engines_lock:
        if _engines is None:
            _engines = warmup_rm_engines(provider=provider or _engines_provider, log=log)
        return _engines


def prewarm_rm_async(provider: str = "trt", log=print) -> None:
    """Build the shared engines in a daemon thread (idempotent, non-blocking).

    Used at server startup and when RM is toggled on at runtime so the slow
    first-time TensorRT build happens off both the server-boot path and the first
    [RM] playback path. Safe to call repeatedly."""
    global _engines_provider
    _engines_provider = provider
    if not models_available() or _engines is not None:
        return

    def _run() -> None:
        try:
            get_shared_engines(provider=provider, log=log)
        except Exception as exc:  # non-fatal: live worker can still build lazily
            log(f"[rm] background prewarm failed: {exc}")

    threading.Thread(target=_run, name="rm-warmup", daemon=True).start()
