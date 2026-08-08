"""GPU-resident face-beautification processor.

Mirrors :class:`pipeline.demosaic.GpuRmProcessor`: frames stay in device memory
from NVDEC to NVENC, the ONNX graphs read and write GPU buffers through ORT
IOBinding, and every resample is a CuPy kernel. The CPU only sees the detector's
small output tensors (for NMS) and the 512x512 crop while it is retouched.

Why this matters at 8K: the ffmpeg-pipe path moves an 88 MB frame across the
process boundary twice per frame and runs the 7680x3840 -> 640 detector
downscale on the CPU. Both disappear here. The per-face inference cost does not
change -- see the module docstring of :mod:`offline.face_beauty_engine` for the
measured floor.

Frames are RGB uint8 (H,W,3) on device, matching the NV12->RGB kernels the other
offline GPU pipelines use. The models disagree about channel order (YuNet and
2DFAN4 want BGR, BiSeNet and GFPGAN want RGB), so the preprocessing kernels take
a ``swap_rb`` flag rather than materialising converted copies.
"""
from __future__ import annotations

import math

import numpy as np

from offline import face_beauty_engine as fb
from offline.face_beauty_retouch_gpu import GpuRetouch

# Flat working view for the VR path. Large enough that the aligned 512 crop is
# not upscaled from it for a typical face.
FLAT_SIZE = 640

# --- kernels ----------------------------------------------------------------

_KERNEL_SRC = r"""
extern "C" {

__device__ __forceinline__ void sample_rgb(
    const unsigned char* src, int stride_w, int W, int H,
    float fx, float fy, float* out) {
  int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
  float ax = fx - x0, ay = fy - y0;
  int x1 = x0 + 1, y1 = y0 + 1;
  if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
  if (x1 > W - 1) x1 = W - 1; if (y1 > H - 1) y1 = H - 1;
  if (x0 > W - 1) x0 = W - 1; if (y0 > H - 1) y0 = H - 1;
  const unsigned char* s00 = src + (y0 * stride_w + x0) * 3;
  const unsigned char* s01 = src + (y0 * stride_w + x1) * 3;
  const unsigned char* s10 = src + (y1 * stride_w + x0) * 3;
  const unsigned char* s11 = src + (y1 * stride_w + x1) * 3;
  for (int c = 0; c < 3; c++)
    out[c] = s00[c]*(1-ax)*(1-ay) + s01[c]*ax*(1-ay)
           + s10[c]*(1-ax)*ay     + s11[c]*ax*ay;
}

// Detector input: letterbox the frame into the top-left of a zeroed SxS canvas
// and emit CHW float32 in one pass. Matches FaceDetector.detect's CPU path
// (fit-inside scale, top-left placement, raw 0..255 values).
__global__ void letterbox_chw(
    const unsigned char* src, float* dst, int W, int H, int S,
    float inv_scale, int new_w, int new_h, int swap_rb,
    int win_x, int win_y, int win_w, int win_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  float rgb[3] = {0.f, 0.f, 0.f};
  if (x < new_w && y < new_h) {
    // Sample inside the source window (one detection tile), clamped to it.
    float sx = fminf((x + 0.5f) * inv_scale - 0.5f, win_w - 1.f);
    float sy = fminf((y + 0.5f) * inv_scale - 0.5f, win_h - 1.f);
    sample_rgb(src + (win_y * W + win_x) * 3, W, win_w, win_h, sx, sy, rgb);
  }
  int plane = S * S;
  for (int c = 0; c < 3; c++) {
    int sc = swap_rb ? (2 - c) : c;
    dst[c * plane + y * S + x] = rgb[sc];
  }
}

// dst(x,y) = src(Minv * (x,y)) -- ``minv`` is the 2x3 inverse of the forward
// src->dst affine, matching cv2.warpAffine's convention.
__global__ void warp_affine_u8(
    const unsigned char* src, unsigned char* dst, const float* minv,
    int W, int H, int dw, int dh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float fx = minv[0] * x + minv[1] * y + minv[2];
  float fy = minv[3] * x + minv[4] * y + minv[5];
  float rgb[3];
  sample_rgb(src, W, W, H, fx, fy, rgb);
  unsigned char* d = dst + (y * dw + x) * 3;
  for (int c = 0; c < 3; c++) d[c] = (unsigned char)(rgb[c] + 0.5f);
}

// HWC uint8 crop -> NCHW float32 model input, with optional R/B swap and a
// per-tensor affine (v * mul + add) covering /255, (v/255-0.5)/0.5 and the
// ImageNet normalisation the parser wants.
__global__ void crop_to_chw(
    const unsigned char* crop, float* dst, int S, int swap_rb,
    float mul0, float add0, float mul1, float add1, float mul2, float add2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  const unsigned char* s = crop + (y * S + x) * 3;
  float mul[3] = {mul0, mul1, mul2};
  float add[3] = {add0, add1, add2};
  int plane = S * S;
  for (int c = 0; c < 3; c++) {
    int sc = swap_rb ? (2 - c) : c;
    dst[c * plane + y * S + x] = s[sc] * mul[c] + add[c];
  }
}

// NCHW float32 model output -> HWC uint8 crop (v * mul + add, then clamp).
__global__ void chw_to_crop(
    const float* src, unsigned char* crop, int S, int swap_rb,
    float mul, float add) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int plane = S * S;
  unsigned char* d = crop + (y * S + x) * 3;
  for (int c = 0; c < 3; c++) {
    int sc = swap_rb ? (2 - c) : c;
    float v = src[sc * plane + y * S + x] * mul + add;
    d[c] = (unsigned char)(fminf(255.f, fmaxf(0.f, v)) + 0.5f);
  }
}

// Parser logits (C,S,S) -> uint8 label map, then straight through a 256-entry
// LUT so a class-set mask is one pass instead of argmax + isin.
__global__ void argmax_to_mask(
    const float* logits, float* mask, const float* lut, int C, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int plane = S * S, idx = y * S + x;
  float best = logits[idx];
  int arg = 0;
  for (int c = 1; c < C; c++) {
    float v = logits[c * plane + idx];
    if (v > best) { best = v; arg = c; }
  }
  mask[idx] = lut[arg];
}

__global__ void argmax_labels(const float* logits, unsigned char* labels, int C, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  int plane = S * S, idx = y * S + x;
  float best = logits[idx];
  int arg = 0;
  for (int c = 1; c < C; c++) {
    float v = logits[c * plane + idx];
    if (v > best) { best = v; arg = c; }
  }
  labels[idx] = (unsigned char)arg;
}

__global__ void labels_to_mask(
    const unsigned char* labels, float* mask, const float* lut, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  mask[y * S + x] = lut[labels[y * S + x]];
}

// Separable Gaussian over a single-channel float plane (cupyx.scipy.ndimage
// does not JIT on this install, so the feathering blur is hand-rolled).
__global__ void blur_h(const float* src, float* dst, const float* k, int r, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  float acc = 0.f;
  for (int i = -r; i <= r; i++) {
    int sx = min(S - 1, max(0, x + i));
    acc += src[y * S + sx] * k[i + r];
  }
  dst[y * S + x] = acc;
}

__global__ void blur_v(const float* src, float* dst, const float* k, int r, int S) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;
  float acc = 0.f;
  for (int i = -r; i <= r; i++) {
    int sy = min(S - 1, max(0, y + i));
    acc += src[sy * S + x] * k[i + r];
  }
  dst[y * S + x] = acc;
}

// (clip(v,0.5,1) - 0.5) * 2 -- the parse-mask feather tail.
__global__ void feather_tail(float* m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = m[i];
  v = fminf(1.f, fmaxf(0.5f, v));
  m[i] = (v - 0.5f) * 2.f;
}

// Composite the finished crop back through the forward affine: for every frame
// pixel in the paste box, map to crop space and blend. Writes the frame in
// place, so nothing is allocated per face.
__global__ void paste_back_blend(
    unsigned char* frame, const unsigned char* crop, const float* mask,
    const float* m, int W, int H, int S, int x0, int y0, int bw, int bh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bw || y >= bh) return;
  int fx = x0 + x, fy = y0 + y;
  if (fx < 0 || fy < 0 || fx >= W || fy >= H) return;
  float cx = m[0] * fx + m[1] * fy + m[2];
  float cy = m[3] * fx + m[4] * fy + m[5];
  if (cx < 0.f || cy < 0.f || cx > S - 1.f || cy > S - 1.f) return;
  int ix0 = (int)floorf(cx), iy0 = (int)floorf(cy);
  float ax = cx - ix0, ay = cy - iy0;
  int ix1 = min(S - 1, ix0 + 1), iy1 = min(S - 1, iy0 + 1);
  float a = mask[iy0 * S + ix0] * (1-ax) * (1-ay) + mask[iy0 * S + ix1] * ax * (1-ay)
          + mask[iy1 * S + ix0] * (1-ax) * ay     + mask[iy1 * S + ix1] * ax * ay;
  if (a <= 0.f) return;
  if (a > 1.f) a = 1.f;
  unsigned char* d = frame + (fy * W + fx) * 3;
  const unsigned char* s00 = crop + (iy0 * S + ix0) * 3;
  const unsigned char* s01 = crop + (iy0 * S + ix1) * 3;
  const unsigned char* s10 = crop + (iy1 * S + ix0) * 3;
  const unsigned char* s11 = crop + (iy1 * S + ix1) * 3;
  for (int c = 0; c < 3; c++) {
    float v = s00[c]*(1-ax)*(1-ay) + s01[c]*ax*(1-ay)
            + s10[c]*(1-ax)*ay     + s11[c]*ax*ay;
    d[c] = (unsigned char)(v * a + d[c] * (1.f - a) + 0.5f);
  }
}

// Same mapping as paste_back_blend, but records the alpha it would have used
// into a flat-space plane instead of blending. Lets the VR path carry the face
// mask through the reprojection so only masked pixels are written back.
__global__ void paste_mask_into(
    float* alpha_plane, const float* mask, const float* m,
    int W, int H, int S, int x0, int y0, int bw, int bh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bw || y >= bh) return;
  int fx = x0 + x, fy = y0 + y;
  if (fx < 0 || fy < 0 || fx >= W || fy >= H) return;
  float cx = m[0] * fx + m[1] * fy + m[2];
  float cy = m[3] * fx + m[4] * fy + m[5];
  if (cx < 0.f || cy < 0.f || cx > S - 1.f || cy > S - 1.f) return;
  int ix0 = (int)floorf(cx), iy0 = (int)floorf(cy);
  float ax = cx - ix0, ay = cy - iy0;
  int ix1 = min(S - 1, ix0 + 1), iy1 = min(S - 1, iy0 + 1);
  float a = mask[iy0 * S + ix0] * (1-ax) * (1-ay) + mask[iy0 * S + ix1] * ax * (1-ay)
          + mask[iy1 * S + ix0] * (1-ax) * ay     + mask[iy1 * S + ix1] * ax * ay;
  alpha_plane[fy * W + fx] = fminf(1.f, fmaxf(0.f, a));
}

// Blend two crops (used for the enhancer strength knob).
__global__ void blend_crops(
    unsigned char* dst, const unsigned char* other, float w, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = (unsigned char)(dst[i] * w + other[i] * (1.f - w) + 0.5f);
}

}
"""

_BLOCK = (16, 16, 1)


def _grid(w: int, h: int) -> tuple[int, int, int]:
    return ((w + 15) // 16, (h + 15) // 16, 1)


def _gauss_kernel(sigma: float) -> tuple[np.ndarray, int]:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32), radius


class GpuFaceBeautyProcessor:
    """Detect, restore and retouch every face in a device-resident RGB frame.

    ``process(frame_g)`` edits the frame in place and returns it, so a caller can
    hand NVDEC output straight to NVENC.
    """

    def __init__(self, options: fb.BeautyOptions, log=print) -> None:
        import cupy as cp

        self.cp = cp
        self.options = options
        self.log = log
        provider = str(options.provider or "trt").lower()

        # ORT is handed this stream, so every later process() must run on it too:
        # the sessions no longer synchronise, and a caller on a different stream
        # (a different thread, e.g. the realtime worker) would race. process()
        # re-enters it explicitly rather than trusting the caller's current one.
        self._stream = cp.cuda.get_current_stream()
        stream_ptr = int(self._stream.ptr)
        self.detector = fb.FaceDetector(provider=provider, stream_ptr=stream_ptr)
        self.landmarker = fb.FaceLandmarker(provider=provider, stream_ptr=stream_ptr) if options.use_landmarker else None
        self.parser = fb.FaceParser(provider=provider, stream_ptr=stream_ptr) if options.needs_parser() else None
        self.enhancer = (
            fb.FaceEnhancer(options.enhancer, provider=provider, stream_ptr=stream_ptr)
            if fb.normalize_enhancer(options.enhancer) != fb.ENHANCER_NONE else None
        )
        self.crop_size = fb.CROP_SIZE
        self.template = self.enhancer.template if self.enhancer else fb.DEFAULT_TEMPLATE
        enhancer_size = self.enhancer.size if self.enhancer else fb.CROP_SIZE

        module = cp.RawModule(code=_KERNEL_SRC)
        self._k_letterbox = module.get_function("letterbox_chw")
        self._k_warp = module.get_function("warp_affine_u8")
        self._k_to_chw = module.get_function("crop_to_chw")
        self._k_from_chw = module.get_function("chw_to_crop")
        self._k_labels = module.get_function("argmax_labels")
        self._k_mask = module.get_function("labels_to_mask")
        self._k_blur_h = module.get_function("blur_h")
        self._k_blur_v = module.get_function("blur_v")
        self._k_feather = module.get_function("feather_tail")
        self._k_paste = module.get_function("paste_back_blend")
        self._k_paste_mask = module.get_function("paste_mask_into")
        self._k_blend = module.get_function("blend_crops")

        self._det_io = self.detector.session.io_binding()
        self._enh_io = self.enhancer.session.io_binding() if self.enhancer else None
        self._par_io = self.parser.session.io_binding() if self.parser else None
        self._lm_io = self.landmarker.session.io_binding() if self.landmarker else None

        # Persistent buffers: one face at a time, so these are allocated once.
        S = self.crop_size
        self._det_in = cp.empty((1, 3, self.detector.size, self.detector.size), cp.float32)
        self._crop = cp.empty((S, S, 3), cp.uint8)
        self._crop_orig = cp.empty((S, S, 3), cp.uint8)
        # The enhancer runs at its own resolution; everything else stays at S.
        E = enhancer_size
        self._enh_size = E
        self._enh_crop = cp.empty((E, E, 3), cp.uint8) if E != S else None
        self._enh_in = cp.empty((1, 3, E, E), cp.float32)
        self._enh_out = cp.empty((1, 3, E, E), cp.float32)
        self._par_in = cp.empty((1, 3, S, S), cp.float32)
        self._labels = cp.empty((S, S), cp.uint8)
        self._mask = cp.empty((S, S), cp.float32)
        self._mask_tmp = cp.empty((S, S), cp.float32)
        self._lm_in = cp.empty((1, 3, 256, 256), cp.float32)
        self._lm_crop = cp.empty((256, 256, 3), cp.uint8)
        # 2DFAN4 outputs: 68 points and their 64x64 heatmaps, both kept on device.
        self._lm_points = cp.empty((1, 68, 3), cp.float32)
        self._lm_heat = cp.empty((1, 68, 64, 64), cp.float32)
        self._box_mask = cp.asarray(
            fb.create_box_mask(S, options.mask_blur, options.mask_padding))
        self._luts = {
            name: cp.asarray(fb._class_lut(classes))
            for name, classes in (("region", fb.REGION_MASK_CLASSES), ("skin", fb.SKIN_CLASSES),
                                  ("eye", fb.EYE_CLASSES), ("lip", fb.LIP_CLASSES),
                                  ("teeth", fb.TEETH_CLASSES))
        }
        self._retouch = GpuRetouch(cp, S)
        k5, r5 = _gauss_kernel(5.0)
        self._k5 = cp.asarray(k5)
        self._r5 = r5
        self.tracker = fb._FaceTracker(options.temporal_smooth)
        self._tiles: list[tuple[int, int, int, int]] | None = None
        self._cached_faces: list[fb.DetectedFace] = []
        self._frame_index = 0
        self._detects_since_sweep = 0
        self._reproj = None
        self._flat = None
        self._flat_alpha = None

    # -- helpers -------------------------------------------------------------

    def provider_summary(self) -> str:
        parts = [f"detector={self.detector.providers[0] if self.detector.providers else 'unknown'}"]
        if self.enhancer:
            parts.append(f"enhancer={self.enhancer.providers[0] if self.enhancer.providers else 'unknown'}")
        return " ".join(parts)

    def reset(self) -> None:
        self.tracker.reset()
        self._cached_faces = []
        self._frame_index = 0
        self._detects_since_sweep = 0

    def _run(self, session, io, feeds: dict, outputs: dict) -> None:
        """Bind GPU pointers and run. ``feeds``/``outputs`` map name -> cupy array;
        an output mapped to ``None`` comes back on the host."""
        for name, arr in feeds.items():
            io.bind_input(name, "cuda", 0, np.float32, arr.shape, int(arr.data.ptr))
        for name, arr in outputs.items():
            if arr is None:
                io.bind_output(name)
            else:
                io.bind_output(name, "cuda", 0, np.float32, arr.shape, int(arr.data.ptr))
        # No synchronise: ORT shares this CuPy stream, so the surrounding
        # kernels and this inference are already ordered on it.
        session.run_with_iobinding(io)


    # -- VR flat view --------------------------------------------------------

    def _reprojector(self):
        if self._reproj is None:
            from pipeline.vr_reproject import VrReprojector

            self._reproj = VrReprojector(self.cp)
            self._flat = self.cp.empty((FLAT_SIZE, FLAT_SIZE, 3), self.cp.uint8)
            self._flat_alpha = self.cp.empty((FLAT_SIZE, FLAT_SIZE), self.cp.float32)
        return self._reproj

    def _vr_enabled(self, W: int, H: int) -> bool:
        mode = str(self.options.vr_reproject or "auto").lower()
        if mode == "off":
            return False
        return W >= 2 * H if mode == "auto" else True

    def _face_view(self, box, W: int, H: int):
        """Map an equirect face box to its eye plus the view angles that centre
        it in a flat view. Same derivation as pipeline.demosaic._region_view."""
        eye_w = W // 2 if W >= 2 * H else W
        eye_h = H
        x1, y1, x2, y2 = [float(v) for v in box]
        eye_x0 = eye_w if (W >= 2 * H and (x1 + x2) * 0.5 >= eye_w) else 0
        ex1, ey1, ex2, ey2 = x1 - eye_x0, y1, x2 - eye_x0, y2
        ecx, ecy = (ex1 + ex2) * 0.5, (ey1 + ey2) * 0.5
        yaw = (ecx / eye_w) * 180.0 - 90.0
        pitch = 90.0 - (ecy / eye_h) * 180.0
        # Longitude is compressed by cos(lat): near the poles a physically small
        # box spans many pixels horizontally, so a linear px->deg reading would
        # wildly overestimate the fov. Use the edge nearest the equator.
        lat_near = min(abs(90.0 - (ey1 / eye_h) * 180.0), abs(90.0 - (ey2 / eye_h) * 180.0))
        cos_lat = max(0.15, math.cos(math.radians(lat_near)))
        fov_x = ((ex2 - ex1) / eye_w) * 180.0 * cos_lat
        fov_y = ((ey2 - ey1) / eye_h) * 180.0
        d_fov = min(self.options.vr_max_fov, max(fov_x, fov_y) * self.options.vr_fov_margin)
        return eye_x0, eye_w, eye_h, (ex1, ey1, ex2, ey2), yaw, pitch, d_fov

    @staticmethod
    def _eye_crop(elocal, eye_w: int, eye_h: int):
        """Compositing window (eye-local x,y,w,h): the box grown so the whole
        reprojected patch lands inside it."""
        ex1, ey1, ex2, ey2 = elocal
        mx, my = 0.5 * (ex2 - ex1), 0.5 * (ey2 - ey1)
        cx0 = max(0, int(math.floor(ex1 - mx)))
        cy0 = max(0, int(math.floor(ey1 - my)))
        cx1 = min(eye_w, int(math.ceil(ex2 + mx)))
        cy1 = min(eye_h, int(math.ceil(ey2 + my)))
        return cx0, cy0, cx1 - cx0, cy1 - cy0

    def _process_face_vr(self, frame_g, face, W: int, H: int,
                         refine_landmarks: bool = True) -> bool:
        """Restore one face through a flat view. Returns False to fall back."""
        from pipeline.vr_reproject import project_points_to_flat

        cp = self.cp
        view = self._face_view(face.bounding_box, W, H)
        eye_x0, eye_w, eye_h, elocal, yaw, pitch, d_fov = view
        if d_fov >= self.options.vr_max_fov:
            return False          # too wide to be worth the gnomonic stretch
        if abs(pitch) < self.options.vr_min_pitch:
            # Near the horizon the direct warp is already right; reprojecting
            # would cost ~4.4 ms for a sub-pixel gain. See vr_min_pitch.
            return False
        reproj = self._reprojector()
        reproj.to_flat(frame_g, yaw, pitch, d_fov, FLAT_SIZE, FLAT_SIZE,
                       eye_origin=(eye_x0, 0), eye_size=(eye_w, eye_h), out=self._flat)

        # The detector's keypoints are already known in eye-local equirect space,
        # so project them rather than re-running the detector inside the view.
        eye_landmarks = face.landmark_5 - np.array([eye_x0, 0], np.float32)
        flat_landmarks, valid = project_points_to_flat(
            eye_landmarks, eye_w, eye_h, yaw, pitch, d_fov, FLAT_SIZE, FLAT_SIZE)
        if not valid.all():
            return False
        flat_face = fb.DetectedFace(_points_bbox(flat_landmarks), face.score,
                                    flat_landmarks.astype(np.float32))
        if self.landmarker is not None:
            cached = getattr(face, "_flat_landmark_5", None)
            if refine_landmarks or cached is None:
                self._refine_landmarks(self._flat, flat_face)
                face._flat_landmark_5 = flat_face.landmark_5.copy()
            else:
                flat_face.landmark_5 = cached

        self._flat_alpha.fill(0)
        mask = self._restore_into(self._flat, flat_face, FLAT_SIZE, FLAT_SIZE,
                                  alpha_plane=self._flat_alpha)
        if mask is None:
            return False
        crop_box = self._eye_crop(elocal, eye_w, eye_h)
        reproj.blend_into_masked(self._flat, self._flat_alpha, frame_g, yaw, pitch, d_fov,
                                 eye_w, eye_h, crop=crop_box, eye_origin=(eye_x0, 0),
                                 feather=self.options.vr_feather)
        return True

    # -- stages --------------------------------------------------------------

    def _detect_window(self, frame_g, score: float,
                       window: tuple[int, int, int, int]) -> list[fb.DetectedFace]:
        """Detect inside one source window; boxes come back in frame coordinates."""
        S = self.detector.size
        W, H = int(frame_g.shape[1]), int(frame_g.shape[0])
        wx, wy, ww, wh = window
        scale = min(S / max(1, ww), S / max(1, wh), 1.0)
        new_w, new_h = max(1, int(ww * scale)), max(1, int(wh * scale))
        self._k_letterbox(_grid(S, S), _BLOCK,
                          (frame_g, self._det_in, np.int32(W), np.int32(H), np.int32(S),
                           np.float32(1.0 / scale), np.int32(new_w), np.int32(new_h),
                           np.int32(1), np.int32(wx), np.int32(wy), np.int32(ww), np.int32(wh)))
        io = self._det_io
        io.bind_input(self.detector.input_name, "cuda", 0, np.float32,
                      self._det_in.shape, int(self._det_in.data.ptr))
        for name in [o.name for o in self.detector.session.get_outputs()]:
            io.bind_output(name)              # small tensors, NMS runs on the host
        self.detector.session.run_with_iobinding(io)
        faces = fb.decode_yunet(io.copy_outputs_to_cpu(), S,
                                ww / float(new_w), wh / float(new_h), score)
        if wx or wy:
            offset_box = np.array([wx, wy, wx, wy], np.float32)
            offset_pt = np.array([wx, wy], np.float32)
            for face in faces:
                face.bounding_box = face.bounding_box + offset_box
                face.landmark_5 = face.landmark_5 + offset_pt
        return faces

    def _detect(self, frame_g, score: float) -> list[fb.DetectedFace]:
        H, W = int(frame_g.shape[0]), int(frame_g.shape[1])
        if self._tiles is None:
            self._tiles = fb.plan_detection_tiles(W, H, self.options.detect_mode)
            if len(self._tiles) > 1:
                self.log(f"detection: {len(self._tiles)} windows "
                         f"({self.options.detect_mode}), "
                         f"every {self.options.detect_interval} frame(s)")
        windows = self._tiles
        roi = False
        if (self.options.detect_roi and len(self._tiles) > 1 and self._cached_faces
                and self._detects_since_sweep < max(1, int(self.options.roi_sweep_interval))):
            candidate = fb.roi_windows(self._cached_faces, W, H)
            if candidate:
                windows, roi = candidate, True

        faces: list[fb.DetectedFace] = []
        for window in windows:
            faces.extend(self._detect_window(frame_g, score, window))
        if roi and not faces:
            # A tracked face left or was lost -- fall back to the full grid now
            # rather than waiting for the next scheduled sweep.
            windows, roi = self._tiles, False
            for window in windows:
                faces.extend(self._detect_window(frame_g, score, window))
        self._detects_since_sweep = self._detects_since_sweep + 1 if roi else 0
        return fb.merge_detections(faces, score) if len(windows) > 1 else faces

    def _landmark(self, frame_g, bounding_box) -> tuple[np.ndarray, float] | None:
        cp = self.cp
        if self.landmarker is None:
            return None
        H, W = int(frame_g.shape[0]), int(frame_g.shape[1])
        size = self.landmarker.size
        scale = 195.0 / max(1.0, float(np.subtract(bounding_box[2:], bounding_box[:2]).max()))
        translation = (size - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]], dtype=np.float32)
        inv = np.asarray(_invert_affine(matrix), dtype=np.float32)
        self._k_warp(_grid(size, size), _BLOCK,
                     (frame_g, self._lm_crop, cp.asarray(inv.ravel()),
                      np.int32(W), np.int32(H), np.int32(size), np.int32(size)))
        # BGR /255 -- the CPU path's conditional CLAHE is skipped here; it only
        # triggers on very dark crops and would cost a device->host round trip.
        self._k_to_chw(_grid(size, size), _BLOCK,
                       (self._lm_crop, self._lm_in, np.int32(size), np.int32(1),
                        np.float32(1 / 255.0), np.float32(0.0), np.float32(1 / 255.0),
                        np.float32(0.0), np.float32(1 / 255.0), np.float32(0.0)))
        io = self._lm_io
        io.bind_input(self.landmarker.input_name, "cuda", 0, np.float32,
                      self._lm_in.shape, int(self._lm_in.data.ptr))
        # Bind both outputs to device buffers. The heatmap is (1,68,64,64) --
        # 1.1 MB per face per frame -- and the only thing taken from it is one
        # scalar, so letting ORT copy it to the host was the single largest cost
        # in this stage. Reduce it on the GPU and bring back the 68 points alone.
        outputs = [o.name for o in self.landmarker.session.get_outputs()]
        io.bind_output(outputs[0], "cuda", 0, np.float32,
                       self._lm_points.shape, int(self._lm_points.data.ptr))
        io.bind_output(outputs[1], "cuda", 0, np.float32,
                       self._lm_heat.shape, int(self._lm_heat.data.ptr))
        self.landmarker.session.run_with_iobinding(io)
        # Host read below, so this one sync stays.
        self._stream.synchronize()
        peak = float(self._lm_heat.max(axis=(2, 3)).mean())
        landmark_68 = cp.asnumpy(self._lm_points)[:, :, :2][0] / 64.0 * size
        landmark_68 = fb._transform_points(landmark_68, _invert_affine(matrix))
        score = float(np.interp(peak, [0, 0.9], [0, 1]))
        return landmark_68.astype(np.float32), score

    def _refine_landmarks(self, target_g, face) -> None:
        """Replace a face's detector keypoints with 68-point landmarks."""
        try:
            result = self._landmark(target_g, face.bounding_box)
            if result is not None:
                face.landmark_score = result[1]
                if result[1] > 0.1:
                    face.landmark_5 = fb.landmark_68_to_5(result[0])
        except Exception as exc:
            self.log(f"landmarker failed, using detector keypoints: {type(exc).__name__}: {exc}")

    def _parse_mask(self, lut_name: str, sigma: float = 5.0):
        """Feathered mask for a class set, from the already-computed label map.

        Returns a fresh buffer: several masks are alive at once during the
        retouch, so they must not share the scratch plane."""
        cp = self.cp
        S = self.crop_size
        self._k_mask(_grid(S, S), _BLOCK,
                     (self._labels, self._mask_tmp, self._luts[lut_name], np.int32(S)))
        return self._blur_feather(self._mask_tmp, sigma)

    def _blur_feather(self, plane, sigma: float):
        cp = self.cp
        S = int(plane.shape[0])
        if abs(sigma - 5.0) < 1e-6:
            k, r = self._k5, self._r5
        else:
            kn, r = _gauss_kernel(sigma)
            k = cp.asarray(kn)
        tmp = cp.empty_like(plane)
        out = cp.empty_like(plane)
        self._k_blur_h(_grid(S, S), _BLOCK, (plane, tmp, k, np.int32(r), np.int32(S)))
        self._k_blur_v(_grid(S, S), _BLOCK, (tmp, out, k, np.int32(r), np.int32(S)))
        n = S * S
        self._k_feather(((n + 255) // 256, 1, 1), (256, 1, 1), (out, np.int32(n)))
        return out

    # -- main ----------------------------------------------------------------

    def process(self, frame_g):
        with self._stream:
            return self._process(frame_g)

    def _process(self, frame_g):
        cp = self.cp
        options = self.options
        H, W = int(frame_g.shape[0]), int(frame_g.shape[1])
        interval = max(1, int(options.detect_interval))
        frame_index = self._frame_index
        if frame_index % interval == 0 or not self._cached_faces:
            self._cached_faces = self._detect(frame_g, options.detector_score)
        faces = self._cached_faces
        self._frame_index += 1
        min_face_px = fb.resolve_min_face_px(options.min_face_mode, H)
        faces = [f for f in faces if min(f.bounding_box[2] - f.bounding_box[0],
                                         f.bounding_box[3] - f.bounding_box[1]) >= min_face_px]
        if not faces:
            self.tracker.smooth(faces)
            return frame_g, fb.FrameStats()
        faces.sort(key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
                                 (f.bounding_box[3] - f.bounding_box[1]), reverse=True)
        if options.max_faces > 0:
            faces = faces[:options.max_faces]

        vr = self._vr_enabled(W, H)
        landmark_interval = max(1, int(options.landmark_interval))
        refine_landmarks = frame_index % landmark_interval == 0
        # On the VR path _process_face_vr runs the landmarker inside the flat
        # view and overwrites whatever this pass produced, so doing it here too
        # just pays for the most expensive per-face stage twice. The view angles
        # come from the bounding box, not from these points.
        if self.landmarker is not None and not vr and refine_landmarks:
            for face in faces:
                self._refine_landmarks(frame_g, face)
        self.tracker.smooth(faces)

        stats = fb.FrameStats(faces=len(faces))
        for face in faces:
            done = False
            if vr:
                try:
                    done = self._process_face_vr(
                        frame_g, face, W, H, refine_landmarks=refine_landmarks)
                except Exception as exc:
                    self.log(f"VR reprojection failed, using the direct warp: "
                             f"{type(exc).__name__}: {exc}")
            if not done:
                if vr and self.landmarker is not None and refine_landmarks:
                    # VR fell back to the direct warp for this face, so it still
                    # needs frame-space landmarks.
                    self._refine_landmarks(frame_g, face)
                self._process_face(frame_g, face, W, H)
            stats.processed += 1
        return frame_g, stats

    def _restore_into(self, target_g, face, W: int, H: int, alpha_plane=None):
        """Warp, restore, retouch and composite one face into ``target_g``.

        ``target_g`` is the frame for the direct path and the flat working view
        for the VR path. When ``alpha_plane`` is given the same affine also
        records the face mask there, so the VR path can carry it back through
        the reprojection. Returns the crop-space mask, or None if the face
        falls outside the target."""
        cp = self.cp
        options = self.options
        S = self.crop_size
        matrix = fb.estimate_affine(face.landmark_5, S, self.template)
        inv = _invert_affine(matrix)
        inv_g = cp.asarray(np.asarray(inv, np.float32).ravel())
        self._k_warp(_grid(S, S), _BLOCK,
                     (target_g, self._crop, inv_g, np.int32(W), np.int32(H),
                      np.int32(S), np.int32(S)))

        # Parse the untouched source crop first.  If parsing happens after the
        # blind restorer, a face hallucinated from hair/back-of-head becomes its
        # own apparently valid region mask and is pasted into the frame.
        mask = self._box_mask
        if self.parser is not None:
            muls = (1.0 / 255.0) / fb.IMAGENET_STD
            adds = -fb.IMAGENET_MEAN / fb.IMAGENET_STD
            self._k_to_chw(_grid(S, S), _BLOCK,
                           (self._crop, self._par_in, np.int32(S), np.int32(0),
                            np.float32(muls[0]), np.float32(adds[0]),
                            np.float32(muls[1]), np.float32(adds[1]),
                            np.float32(muls[2]), np.float32(adds[2])))
            logits_name = self.parser.session.get_outputs()[0].name
            logits = cp.empty((1, 19, S, S), cp.float32)
            self._run(self.parser.session, self._par_io,
                      {self.parser.input_name: self._par_in}, {logits_name: logits})
            self._k_labels(_grid(S, S), _BLOCK,
                           (logits, self._labels, np.int32(19), np.int32(S)))
            source_region_mask = self._parse_mask("region")
            if options.use_region_mask:
                mask = cp.minimum(self._box_mask, source_region_mask)

        # Do not download/reduce the parser mask just to make this branch: that
        # device synchronization costs several FPS.  The source-derived mask
        # already prevents a generated face from painting over hair, while the
        # host-resident landmark score cheaply rejects the weakest detections.
        if self.enhancer is not None and fb.enhancer_is_safe(face, None):
            self._crop_orig[...] = self._crop
            E = self._enh_size
            source = self._crop
            if E != S:
                # Resample into the model's own resolution with the same affine
                # kernel (a pure scale), so a 256 restorer really runs at 256.
                scale = np.asarray([[S / E, 0, 0], [0, S / E, 0]], np.float32)
                self._k_warp(_grid(E, E), _BLOCK,
                             (self._crop, self._enh_crop, cp.asarray(scale.ravel()),
                              np.int32(S), np.int32(S), np.int32(E), np.int32(E)))
                source = self._enh_crop
            # RGB, (v/255 - 0.5) / 0.5
            mul, add = np.float32(2.0 / 255.0), np.float32(-1.0)
            self._k_to_chw(_grid(E, E), _BLOCK,
                           (source, self._enh_in, np.int32(E), np.int32(0),
                            mul, add, mul, add, mul, add))
            self._run(self.enhancer.session, self._enh_io,
                      {self.enhancer.input_names[0]: self._enh_in},
                      {self.enhancer.session.get_outputs()[0].name: self._enh_out})
            target = self._crop if E == S else self._enh_crop
            self._k_from_chw(_grid(E, E), _BLOCK,
                             (self._enh_out, target, np.int32(E), np.int32(0),
                              np.float32(127.5), np.float32(127.5)))
            if E != S:
                back = np.asarray([[E / S, 0, 0], [0, E / S, 0]], np.float32)
                self._k_warp(_grid(S, S), _BLOCK,
                             (self._enh_crop, self._crop, cp.asarray(back.ravel()),
                              np.int32(E), np.int32(E), np.int32(S), np.int32(S)))
            blend = float(np.clip(options.enhancer_blend, 0.0, 1.0))
            if blend < 1.0:
                n = S * S * 3
                self._k_blend(((n + 255) // 256, 1, 1), (256, 1, 1),
                              (self._crop, self._crop_orig, np.float32(blend), np.int32(n)))

        # Retouch on the GPU. This used to round-trip to the host, which cost
        # 74 ms of a 116 ms frame at 720p -- the transfer was never the issue,
        # the arithmetic was. Mirrors face_beauty_engine.retouch_crop exactly.
        if _needs_retouch(options):
            skin = self._parse_mask("skin") if self.parser is not None else self._box_mask
            self._retouch.skin_smooth(self._crop, skin, options.skin_smooth)
            adjustments = [
                {"mask": skin, "strength": options.skin_even, "even_chroma": True},
                {"mask": skin, "strength": options.skin_brighten, "luma_gain": 28.0},
            ]
            if self.parser is not None:
                if options.eye_brighten > 0:
                    adjustments.append({"mask": self._parse_mask("eye", sigma=2.0),
                                        "strength": options.eye_brighten, "luma_gain": 25.0})
                if options.teeth_white > 0:
                    adjustments.append({"mask": self._parse_mask("teeth", sigma=2.0),
                                        "strength": options.teeth_white,
                                        "luma_gain": 20.0, "blue_shift": -12.0})
                if options.lip_vivid > 0:
                    adjustments.append({"mask": self._parse_mask("lip", sigma=2.0),
                                        "strength": options.lip_vivid, "chroma_gain": 1.5})
            self._retouch.lab_adjustments(self._crop, adjustments)
            self._retouch.sharpen(self._crop, options.sharpen)

        x1, y1, x2, y2 = _paste_box(matrix, S, W, H)
        if x2 <= x1 or y2 <= y1:
            return None
        mask_g = cp.ascontiguousarray(mask)
        matrix_g = cp.asarray(np.asarray(matrix, np.float32).ravel())
        grid = _grid(x2 - x1, y2 - y1)
        args = (np.int32(W), np.int32(H), np.int32(S),
                np.int32(x1), np.int32(y1), np.int32(x2 - x1), np.int32(y2 - y1))
        self._k_paste(grid, _BLOCK, (target_g, self._crop, mask_g, matrix_g) + args)
        if alpha_plane is not None:
            self._k_paste_mask(grid, _BLOCK, (alpha_plane, mask_g, matrix_g) + args)
        return mask

    def _process_face(self, frame_g, face, W: int, H: int) -> None:
        self._restore_into(frame_g, face, W, H)

def _needs_retouch(options: fb.BeautyOptions) -> bool:
    return any((options.skin_smooth, options.skin_brighten, options.skin_even,
                options.eye_brighten, options.teeth_white, options.lip_vivid, options.sharpen))


def _invert_affine(matrix) -> np.ndarray:
    import cv2

    return cv2.invertAffineTransform(np.asarray(matrix, np.float32))


def _paste_box(matrix, size: int, W: int, H: int) -> tuple[int, int, int, int]:
    """Frame-space bounding box of the warped crop (the paste target)."""
    import cv2

    inv = cv2.invertAffineTransform(np.asarray(matrix, np.float32))
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    pts = fb._transform_points(corners, inv)
    x1, y1 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [W, H])
    x2, y2 = np.clip(np.ceil(pts.max(axis=0)).astype(int), 0, [W, H])
    return int(x1), int(y1), int(x2), int(y2)


def _points_bbox(points) -> np.ndarray:
    """Axis-aligned box around projected keypoints, padded to a face-sized box.

    The 5 keypoints span roughly the inner face, so the landmarker (which wants
    a detector-style box) gets them grown by half again in each direction."""
    pts = np.asarray(points, np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    mx, my = (x2 - x1) * 0.5, (y2 - y1) * 0.5
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], np.float32)
