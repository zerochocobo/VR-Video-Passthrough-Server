"""Offline face-beautification ("FB") inference core.

Four ONNX graphs, all run through onnxruntime (same stack as the DA3 depth and
RM mosaic engines), all with permissive upstream licenses:

  - detector    YuNet 2023mar (OpenCV, MIT)
                input 'input' (1,3,H,W) BGR 0..255 -> 12 tensors
                (cls / obj / bbox / kps for strides 8,16,32)
  - landmarker  2DFAN4 (MIT), input (1,3,256,256) BGR 0..1
                -> (1,68,3) heatmap-argmax coords in 64-space + heatmap
  - parser      BiSeNet ResNet-34 (MIT), input (1,3,512,512) RGB ImageNet-norm
                -> (1,19,512,512) face-part logits, used for the region mask
                and for gating the skin/eye/teeth/lip retouch passes
  - enhancer    GFPGAN 1.2/1.3/1.4 (Apache-2.0) or RestoreFormer++ (Apache-2.0)
                input 'input' (1,3,512,512) RGB -1..1 -> same shape

Pipeline per face: detect -> (optional) 68-point landmarks -> 5-point affine
warp to the ffhq_512 template -> enhancer -> skin/eye/teeth/lip retouch ->
box (+ region) mask -> paste back -> global blend.

Everything runs on the frame as-is, so SBS/VR sources work without special
handling: each eye's face is detected and processed independently. Faces very
close to an equirect pole are stretched by the projection and the affine warp
cannot undo that -- those are best left to the `--min-face` filter.

TensorRT engine caching mirrors the DA3 engine (see offline/da3_depth.py): the
TensorRT EP builds an fp16 engine on first use and caches it under
runtime_cache/face_beauty_trt/<key>. GFPGAN is a large StyleGAN-style decoder
and its first build can run for several minutes, which is why the UI drives
`offline/face_beauty.py build-trt` from a progress dialog instead of paying that
cost inside the first conversion.

The detector/landmarker/mask geometry follows the reference implementation in
FaceFusion (github.com/facefusion/facefusion) -- template points, anchor
decoding and paste-back math are equivalent so the ONNX graphs behave the same.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import config

# --- model registry ---------------------------------------------------------
# Every entry is a public ONNX re-export hosted on Hugging Face. ``repo`` is the
# HF repo id; ``file`` the artifact. Engine caches are keyed by the dict key, so
# two presets with different input shapes never share a TRT cache dir.

MODELS_DIR_NAME = "FaceBeauty"
TRT_CACHE_NAME = "face_beauty_trt"

DETECTOR_MODEL = {
    "repo": "facefusion/models-3.4.0",
    "file": "yunet_2023_mar.onnx",
    "license": "MIT (OpenCV)",
}
LANDMARKER_MODEL = {
    "repo": "facefusion/models-3.0.0",
    "file": "2dfan4.onnx",
    "license": "MIT (breadbread1984)",
    "size": 256,
}
PARSER_MODEL = {
    "repo": "facefusion/models-3.0.0",
    "file": "bisenet_resnet_34.onnx",
    "license": "MIT (yakhyo)",
    "size": 512,
}
ENHANCER_MODELS: dict[str, dict] = {
    "gfpgan_1.4": {
        "repo": "facefusion/models-3.0.0",
        "file": "gfpgan_1.4.onnx",
        "size": 512,
        "template": "ffhq_512",
        "license": "Apache-2.0 (TencentARC)",
    },
    "gfpgan_1.3": {
        "repo": "facefusion/models-3.0.0",
        "file": "gfpgan_1.3.onnx",
        "size": 512,
        "template": "ffhq_512",
        "license": "Apache-2.0 (TencentARC)",
    },
    "gfpgan_1.2": {
        "repo": "facefusion/models-3.0.0",
        "file": "gfpgan_1.2.onnx",
        "size": 512,
        "template": "ffhq_512",
        "license": "Apache-2.0 (TencentARC)",
    },
    "restoreformer_plus_plus": {
        "repo": "facefusion/models-3.0.0",
        "file": "restoreformer_plus_plus.onnx",
        "size": 512,
        "template": "ffhq_512",
        "license": "Apache-2.0 (wzhouxiff)",
    },
    # GPEN. Much cheaper than GFPGAN at the same visual quality -- gpen_bfr_256
    # measured 6.1 ms against gfpgan_1.4's 38.0 ms -- which is what makes a
    # realtime budget plausible at all.
    #
    # Licence warning: the upstream repo (github.com/yangxy/GPEN) ships **no
    # LICENSE file**, so no rights are granted by default, and its README states
    # the best weights were withheld "due to commercial issues". Usable for a
    # non-commercial, open-source build that downloads the weights at runtime;
    # do not bundle these files into a redistributed package.
    "gpen_bfr_256": {
        "repo": "facefusion/models-3.0.0",
        "file": "gpen_bfr_256.onnx",
        "size": 256,
        "template": "arcface_128",
        "license": "Non-Commercial / unlicensed (yangxy)",
    },
    "gpen_bfr_512": {
        "repo": "facefusion/models-3.0.0",
        "file": "gpen_bfr_512.onnx",
        "size": 512,
        "template": "ffhq_512",
        "license": "Non-Commercial / unlicensed (yangxy)",
    },
    "gpen_bfr_1024": {
        "repo": "facefusion/models-3.0.0",
        "file": "gpen_bfr_1024.onnx",
        "size": 1024,
        "template": "ffhq_512",
        "license": "Non-Commercial / unlicensed (yangxy)",
    },
}
DEFAULT_ENHANCER = "gfpgan_1.4"
ENHANCER_NONE = "none"

# YuNet's ONNX has a fixed 640x640 input (its three stride heads emit exactly
# 6400/1600/400 anchors), so there is no detector-size choice to make. Finding
# small faces in an 8K frame needs tiled/ROI detection, not a bigger input.
DETECTOR_SIZE = 640

# Detection tiling. YuNet's input is a fixed 640, so on a large frame a
# whole-frame letterbox destroys small faces: a real 119 px face in a 7680x3840
# VR frame lands at ~10 px and is missed at every threshold. Splitting the frame
# into overlapping windows that are each only ~2x the detector input keeps faces
# resolvable. Tiles never straddle the eye boundary of an SBS frame -- half a
# face per eye is not a face.
DETECT_MODES = ("auto", "full", "tiled")
DEFAULT_DETECT_MODE = "auto"
TILE_TARGET = 1600          # source pixels per tile edge before the 640 downscale
TILE_OVERLAP = 0.25         # so a face on a tile seam is whole in a neighbour
AUTO_TILE_MIN_EDGE = 2200   # below this the whole frame already resolves faces


def plan_detection_tiles(width: int, height: int, mode: str = DEFAULT_DETECT_MODE) -> list[tuple[int, int, int, int]]:
    """Source windows (x, y, w, h) to run the detector over.

    A single window covering the frame means "no tiling". SBS frames are split
    per eye first, then each eye is gridded."""
    mode = str(mode or "").lower()
    if mode not in DETECT_MODES:
        mode = DEFAULT_DETECT_MODE
    if mode == "full":
        return [(0, 0, width, height)]
    # Splitting an SBS frame per eye is free resolution even when no grid is
    # needed: one letterbox of the whole frame would squash both eyes into the
    # same 640 and halve the pixels a face gets.
    eyes = ([(0, width // 2), (width // 2, width - width // 2)]
            if width >= 2 * height else [(0, width)])
    eye_w_max = max(eye_w for _x0, eye_w in eyes)
    grid = mode == "tiled" or max(eye_w_max, height) >= AUTO_TILE_MIN_EDGE
    if not grid and len(eyes) == 1:
        return [(0, 0, width, height)]

    windows: list[tuple[int, int, int, int]] = []
    for eye_x0, eye_w in eyes:
        if not grid:
            windows.append((eye_x0, 0, eye_w, height))
            continue
        cols = max(1, int(math.ceil(eye_w / TILE_TARGET)))
        rows = max(1, int(math.ceil(height / TILE_TARGET)))
        if cols == 1 and rows == 1:
            windows.append((eye_x0, 0, eye_w, height))
            continue
        step_x = eye_w / cols
        step_y = height / rows
        pad_x = step_x * TILE_OVERLAP
        pad_y = step_y * TILE_OVERLAP
        for row in range(rows):
            for col in range(cols):
                x0 = int(max(0, math.floor(col * step_x - pad_x)))
                y0 = int(max(0, math.floor(row * step_y - pad_y)))
                x1 = int(min(eye_w, math.ceil((col + 1) * step_x + pad_x)))
                y1 = int(min(height, math.ceil((row + 1) * step_y + pad_y)))
                windows.append((eye_x0 + x0, y0, x1 - x0, y1 - y0))
    return windows


def roi_windows(faces: list["DetectedFace"], width: int, height: int,
                margin: float = 1.5) -> list[tuple[int, int, int, int]]:
    """One detection window per known face, grown by ``margin`` around its box.

    Windows are clamped to the face's own eye on an SBS frame, so a window can
    never span the seam and turn two half-faces into one detection."""
    is_sbs = width >= 2 * height
    eye_w = width // 2 if is_sbs else width
    windows: list[tuple[int, int, int, int]] = []
    for face in faces:
        x1, y1, x2, y2 = [float(v) for v in face.bounding_box]
        eye_x0 = eye_w if (is_sbs and (x1 + x2) * 0.5 >= eye_w) else 0
        eye_x1 = eye_x0 + eye_w
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half = max(bw, bh) * (0.5 + margin)
        wx0 = int(max(eye_x0, math.floor(cx - half)))
        wy0 = int(max(0, math.floor(cy - half)))
        wx1 = int(min(eye_x1, math.ceil(cx + half)))
        wy1 = int(min(height, math.ceil(cy + half)))
        if wx1 - wx0 >= 16 and wy1 - wy0 >= 16:
            windows.append((wx0, wy0, wx1 - wx0, wy1 - wy0))
    return windows


def merge_detections(faces: list["DetectedFace"], score_threshold: float,
                     nms_threshold: float = 0.4) -> list["DetectedFace"]:
    """NMS across tiles, so a face seen in two overlapping windows appears once."""
    if len(faces) < 2:
        return faces
    xywh = [(float(f.bounding_box[0]), float(f.bounding_box[1]),
             float(f.bounding_box[2] - f.bounding_box[0]),
             float(f.bounding_box[3] - f.bounding_box[1])) for f in faces]
    scores = [float(f.score) for f in faces]
    keep = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=score_threshold,
                            nms_threshold=nms_threshold)
    return [faces[int(i)] for i in np.array(keep).reshape(-1)]


# Minimum-face policy. Expressing this as a share of frame height rather than a
# pixel count is what makes one setting work for both a 720p clip and an 8K VR
# source, where the same person occupies wildly different pixel counts.
# (ratio of frame height, absolute floor in pixels)
MIN_FACE_MODES: dict[str, tuple[float, int]] = {
    "auto": (0.015, 24),      # skip faces too small to restore usefully
    "loose": (0.0, 20),       # process everything the detector reports
    "strict": (0.05, 48),     # foreground subject only
}
DEFAULT_MIN_FACE_MODE = "auto"


def resolve_min_face_px(mode: str, frame_height: int) -> int:
    ratio, floor = MIN_FACE_MODES.get(str(mode or "").lower(), MIN_FACE_MODES[DEFAULT_MIN_FACE_MODE])
    return max(floor, int(frame_height * ratio))


# BiSeNet face-parsing class ids (19-class CelebAMask-HQ ordering).
PARSE_SKIN = 1
PARSE_LEFT_EYEBROW = 2
PARSE_RIGHT_EYEBROW = 3
PARSE_LEFT_EYE = 4
PARSE_RIGHT_EYE = 5
PARSE_GLASSES = 6
PARSE_NOSE = 10
PARSE_MOUTH = 11
PARSE_UPPER_LIP = 12
PARSE_LOWER_LIP = 13
# Region mask: the parts the enhancer is allowed to write. Glasses and hair stay
# out so spectacle frames are not repainted.
REGION_MASK_CLASSES = (
    PARSE_SKIN, PARSE_LEFT_EYEBROW, PARSE_RIGHT_EYEBROW, PARSE_LEFT_EYE,
    PARSE_RIGHT_EYE, PARSE_NOSE, PARSE_MOUTH, PARSE_UPPER_LIP, PARSE_LOWER_LIP,
)
SKIN_CLASSES = (PARSE_SKIN, PARSE_NOSE)
EYE_CLASSES = (PARSE_LEFT_EYE, PARSE_RIGHT_EYE)
LIP_CLASSES = (PARSE_UPPER_LIP, PARSE_LOWER_LIP)
TEETH_CLASSES = (PARSE_MOUTH,)

# Blind restorers are generative: given a detector false-positive on hair or a
# heavily occluded/profile head they can invent a complete frontal face.  The
# parser is discriminative and sees the unmodified aligned crop, so its facial
# region coverage is a useful safety signal.  Below this coverage restoration
# is disabled; non-generative retouch may still run over whatever real facial
# pixels the parser found.
MIN_ENHANCER_FACE_COVERAGE = 0.08
MIN_ENHANCER_LANDMARK_SCORE = 0.20

# Five-point alignment templates (normalized). Which one a restorer wants is a
# property of how it was trained, so it travels with the model entry.
FFHQ_512_TEMPLATE = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32)
ARCFACE_128_TEMPLATE = np.array([
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
], dtype=np.float32)
WARP_TEMPLATES = {
    "ffhq_512": FFHQ_512_TEMPLATE,
    "arcface_128": ARCFACE_128_TEMPLATE,
}
DEFAULT_TEMPLATE = "ffhq_512"

# The crop every stage after the warp works in. Fixed at 512 because the parser's
# input is fixed at 512; a restorer with a different native size gets its own
# tensor resampled from this crop.
CROP_SIZE = 512

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ENGINE_LOCK = threading.Lock()


def _models_dir() -> Path:
    return config.ROOT / "models" / MODELS_DIR_NAME


def model_path(entry: dict) -> Path:
    return _models_dir() / entry["file"]


def _trt_cache_dir(key: str) -> Path:
    path = config.ROOT / "runtime_cache" / TRT_CACHE_NAME / key
    path.mkdir(parents=True, exist_ok=True)
    return path


# Written by build_trt_stage only after the engine has been built *and* run
# once. A crashed build leaves a partial .engine behind, so the presence of the
# file alone is not evidence that it is loadable -- that mistake sends a corrupt
# engine into the runtime, where it segfaults.
TRT_READY_MARKER = "build.ok"


def trt_engine_cached(key: str) -> bool:
    cache = config.ROOT / "runtime_cache" / TRT_CACHE_NAME / key
    return (cache / TRT_READY_MARKER).is_file() and any(cache.glob("*.engine"))


def mark_trt_ready(key: str) -> None:
    (_trt_cache_dir(key) / TRT_READY_MARKER).write_text("ok", encoding="utf-8")


def runtime_provider(provider: str, cache_key: str, log=None, force: bool = False) -> str:
    """Downgrade ``trt`` to ``cuda`` when that engine is not cached yet.

    Engine builds must only ever happen inside :func:`build_trt`'s isolated
    subprocess -- an inline build crashes the process when other sessions are
    around, and on a loaded GPU it can crash on its own. Loading a cached engine
    is always safe, so the runtime rule is simply: cached or CUDA."""
    if force:
        return provider          # build_trt_stage: building is the point
    if provider == "trt" and not trt_engine_cached(cache_key):
        if log:
            log(f"TensorRT engine for {cache_key} is not built; using CUDA for it")
        return "cuda"
    return provider


def onnx_providers(provider: str, cache_key: str, fp16: bool = True,
                   stream_ptr: int | None = None) -> list:
    """Provider chain. ``trt`` puts the TensorRT EP (cached engine) in front of
    CUDA; falls back to CUDA then CPU when unavailable.

    ``fp16=False`` builds an fp32 engine -- see :class:`FaceEnhancer`."""
    available = set(ort.get_available_providers())
    chain: list = []
    if provider == "trt" and "TensorrtExecutionProvider" in available:
        chain.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": bool(fp16),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(_trt_cache_dir(cache_key)),
                "trt_timing_cache_enable": True,
                # Pin the workspace so the engine hash does not depend on free
                # VRAM at build time (see config.ONNX_TRT_MAX_WORKSPACE_BYTES).
                "trt_max_workspace_size": config.ONNX_TRT_MAX_WORKSPACE_BYTES,
                # Run on the caller's CUDA stream so its kernels and this
                # inference are ordered without a device-wide synchronise.
                **({"has_user_compute_stream": "1",
                    "user_compute_stream": str(stream_ptr)} if stream_ptr else {}),
            },
        ))
    if provider in ("trt", "cuda") and "CUDAExecutionProvider" in available:
        if stream_ptr:
            chain.append(("CUDAExecutionProvider", {
                "has_user_compute_stream": "1",
                "user_compute_stream": str(stream_ptr),
            }))
        else:
            chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


# --- model download ---------------------------------------------------------


def enhancer_entry(name: str) -> dict:
    return ENHANCER_MODELS.get(str(name or "").strip().lower(), ENHANCER_MODELS[DEFAULT_ENHANCER])


def normalize_enhancer(name: str) -> str:
    value = str(name or "").strip().lower()
    if value == ENHANCER_NONE:
        return ENHANCER_NONE
    return value if value in ENHANCER_MODELS else DEFAULT_ENHANCER


def required_models(enhancer: str, use_landmarker: bool, use_parser: bool) -> list[tuple[str, dict]]:
    """``(label, entry)`` for everything a run with these options will load."""
    items: list[tuple[str, dict]] = [("detector", DETECTOR_MODEL)]
    if use_landmarker:
        items.append(("landmarker", LANDMARKER_MODEL))
    if use_parser:
        items.append(("parser", PARSER_MODEL))
    enhancer = normalize_enhancer(enhancer)
    if enhancer != ENHANCER_NONE:
        items.append((enhancer, enhancer_entry(enhancer)))
    return items


def model_available(entry: dict) -> bool:
    return model_path(entry).exists()


def download_target(entry: dict, language: str | None = None) -> tuple[str, Path, list[str]]:
    """``(filename, dest_path, urls)`` for the UI download dialog."""
    from utils import hf_download

    return entry["file"], model_path(entry), hf_download.hf_resolve_urls(entry["repo"], entry["file"], language)


def download_model(entry: dict, log=print, progress=None) -> Path:
    from utils import hf_download

    dest = model_path(entry)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = hf_download.hf_resolve_urls(entry["repo"], entry["file"])
    hf_download.download_file(urls, dest, progress=progress, log=log)
    return dest


def ensure_models_available(enhancer: str, use_landmarker: bool, use_parser: bool, log=print) -> bool:
    """Download any missing ONNX files. Returns False (and logs) on failure."""
    for label, entry in required_models(enhancer, use_landmarker, use_parser):
        if model_available(entry):
            continue
        try:
            log(f"downloading {label} model: {entry['file']}")
            download_model(entry, log=log)
        except Exception as exc:
            log(f"download failed for {entry['file']}: {type(exc).__name__}: {exc}")
            return False
    return True


# --- geometry helpers (equivalent to FaceFusion's face_helper) ---------------


@lru_cache()
def _static_anchors(feature_stride: int, stride_height: int, stride_width: int) -> np.ndarray:
    x, y = np.mgrid[:stride_width, :stride_height]
    anchors = np.stack((y, x), axis=-1)
    return (anchors * feature_stride).reshape((-1, 2)).astype(np.float32)


def estimate_affine(landmark_5: np.ndarray, crop_size: int,
                    template_name: str = DEFAULT_TEMPLATE) -> np.ndarray:
    template = WARP_TEMPLATES.get(template_name, FFHQ_512_TEMPLATE) * float(crop_size)
    matrix = cv2.estimateAffinePartial2D(
        landmark_5.astype(np.float32), template, method=cv2.RANSAC, ransacReprojThreshold=100
    )[0]
    return matrix


def warp_face(frame: np.ndarray, landmark_5: np.ndarray, crop_size: int,
              template_name: str = DEFAULT_TEMPLATE) -> tuple[np.ndarray, np.ndarray]:
    matrix = estimate_affine(landmark_5, crop_size, template_name)
    crop = cv2.warpAffine(frame, matrix, (crop_size, crop_size),
                          borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, matrix


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(1, -1, 2).astype(np.float64), matrix).reshape(-1, 2)


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Alpha-composite an enhanced crop back through the inverse affine.

    Only the axis-aligned box covered by the inverse-warped crop is touched, so
    the cost stays proportional to the face, not to the (8K) frame."""
    height, width = frame.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0], [crop.shape[1], 0], [crop.shape[1], crop.shape[0]], [0, crop.shape[0]]])
    region = _transform_points(corners, inverse)
    x1, y1 = np.clip(np.floor(region.min(axis=0)).astype(int), 0, [width, height])
    x2, y2 = np.clip(np.ceil(region.max(axis=0)).astype(int), 0, [width, height])
    if x2 <= x1 or y2 <= y1:
        return frame
    paste_matrix = inverse.copy()
    paste_matrix[0, 2] -= x1
    paste_matrix[1, 2] -= y1
    size = (int(x2 - x1), int(y2 - y1))
    inverse_mask = cv2.warpAffine(mask, paste_matrix, size).clip(0, 1)[:, :, None]
    inverse_crop = cv2.warpAffine(crop, paste_matrix, size, borderMode=cv2.BORDER_REPLICATE)
    target = frame[y1:y2, x1:x2].astype(np.float32)
    blended = target * (1.0 - inverse_mask) + inverse_crop.astype(np.float32) * inverse_mask
    frame[y1:y2, x1:x2] = blended.astype(frame.dtype)
    return frame


def landmark_68_to_5(landmark_68: np.ndarray) -> np.ndarray:
    return np.array([
        np.mean(landmark_68[36:42], axis=0),
        np.mean(landmark_68[42:48], axis=0),
        landmark_68[30],
        landmark_68[48],
        landmark_68[54],
    ], dtype=np.float32)


def create_box_mask(crop_size: int, blur: float, padding: tuple[float, float, float, float]) -> np.ndarray:
    """Feathered rectangle in crop space; ``padding`` is (top, right, bottom, left) percent."""
    blur_amount = int(crop_size * 0.5 * float(blur))
    blur_area = max(blur_amount // 2, 1)
    mask = np.ones((crop_size, crop_size), dtype=np.float32)
    mask[:max(blur_area, int(crop_size * padding[0] / 100)), :] = 0
    mask[-max(blur_area, int(crop_size * padding[2] / 100)):, :] = 0
    mask[:, :max(blur_area, int(crop_size * padding[3] / 100))] = 0
    mask[:, -max(blur_area, int(crop_size * padding[1] / 100)):] = 0
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    return mask


def _feather(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Soften a hard 0/1 parse mask without eroding it away."""
    return (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), sigma).clip(0.5, 1) - 0.5) * 2


@lru_cache()
def _class_lut(classes: tuple[int, ...]) -> np.ndarray:
    """256-entry float lookup selecting a set of face-parse class ids."""
    lut = np.zeros(256, dtype=np.float32)
    lut[list(classes)] = 1.0
    return lut


# --- detection --------------------------------------------------------------


@dataclass
class DetectedFace:
    bounding_box: np.ndarray          # (4,) x1,y1,x2,y2 in frame space
    score: float
    landmark_5: np.ndarray            # (5,2) in frame space
    landmark_score: float | None = None


def enhancer_is_safe(face: DetectedFace, source_face_coverage: float | None) -> bool:
    """Whether a blind restorer may synthesize details for this detection.

    Prefer the parser signal because it rejects back-of-head/hair detections
    even when YuNet keypoints look geometrically plausible.  The landmark score
    is the fallback for configurations that deliberately disable the parser.
    """
    if source_face_coverage is not None:
        return source_face_coverage >= MIN_ENHANCER_FACE_COVERAGE
    if face.landmark_score is not None:
        return face.landmark_score >= MIN_ENHANCER_LANDMARK_SCORE
    return True


class FaceDetector:
    """YuNet 2023mar. Fixed square input; the frame is letterboxed top-left."""

    def __init__(self, provider: str = "trt", log=None, force_trt: bool = False,
                 stream_ptr: int | None = None):
        self.size = DETECTOR_SIZE
        provider = runtime_provider(provider, "detector", log, force_trt)
        path = model_path(DETECTOR_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"face detector ONNX not found: {path}")
        self.session = ort.InferenceSession(
            str(path), sess_options=_session_options(provider),
            providers=onnx_providers(provider, "detector", stream_ptr=stream_ptr),
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

    def detect_tiled(self, frame_bgr: np.ndarray, score_threshold: float,
                     tiles: list[tuple[int, int, int, int]],
                     nms_threshold: float = 0.4) -> list[DetectedFace]:
        """Detect over each source window and merge into frame coordinates."""
        faces: list[DetectedFace] = []
        for x0, y0, tw, th in tiles:
            window = frame_bgr[y0:y0 + th, x0:x0 + tw]
            for face in self.detect(window, score_threshold, nms_threshold):
                face.bounding_box = face.bounding_box + np.array([x0, y0, x0, y0], np.float32)
                face.landmark_5 = face.landmark_5 + np.array([x0, y0], np.float32)
                faces.append(face)
        return merge_detections(faces, score_threshold, nms_threshold)

    def detect(self, frame_bgr: np.ndarray, score_threshold: float, nms_threshold: float = 0.4) -> list[DetectedFace]:
        height, width = frame_bgr.shape[:2]
        scale = min(self.size / max(1, width), self.size / max(1, height))
        if scale < 1.0:
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_bgr
            new_w, new_h = width, height
        ratio_w = width / float(new_w)
        ratio_h = height / float(new_h)

        canvas = np.zeros((self.size, self.size, 3), dtype=np.float32)
        canvas[:new_h, :new_w, :] = small
        tensor = np.expand_dims(canvas.transpose(2, 0, 1), axis=0)
        detection = self.session.run(None, {self.input_name: tensor})
        return decode_yunet(detection, self.size, ratio_w, ratio_h,
                            score_threshold, nms_threshold)


def decode_yunet(detection, size: int, ratio_w: float, ratio_h: float,
                 score_threshold: float, nms_threshold: float = 0.4) -> list["DetectedFace"]:
    """Decode YuNet's 12 output tensors into faces in source-frame coordinates.

    Shared by the CPU path and :mod:`offline.face_beauty_gpu`, which produces the
    same tensors from a GPU letterbox -- the decode itself is cheap and stays on
    the host either way, since NMS does."""
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    landmarks: list[np.ndarray] = []
    for index, feature_stride in enumerate((8, 16, 32)):
        raw_scores = (detection[index] * detection[index + 3]).reshape(-1)
        keep = np.where(raw_scores >= score_threshold)[0]
        if not keep.size:
            continue
        stride_h = size // feature_stride
        stride_w = size // feature_stride
        # Decode only the anchors that passed the score gate. The three strides
        # carry 8400 anchors between them and typically one survives, so
        # decoding all of them (an exp, a stack and a concatenate over the full
        # set) cost more than the inference itself.
        anchors = _static_anchors(feature_stride, stride_h, stride_w)[keep]
        raw_boxes = detection[index + 6].squeeze(0)[keep]
        centers = raw_boxes[:, :2] * feature_stride + anchors
        sizes = np.exp(raw_boxes[:, 2:4]) * feature_stride
        corners = np.stack([
            (centers[:, 0] - sizes[:, 0] / 2) * ratio_w,
            (centers[:, 1] - sizes[:, 1] / 2) * ratio_h,
            (centers[:, 0] + sizes[:, 0] / 2) * ratio_w,
            (centers[:, 1] + sizes[:, 1] / 2) * ratio_h,
        ], axis=-1)
        raw_kps = detection[index + 9].squeeze(0)[keep]
        kps = np.concatenate([raw_kps[:, i:i + 2] * feature_stride + anchors for i in range(0, 10, 2)], axis=-1)
        kps = kps.reshape(-1, 5, 2) * [ratio_w, ratio_h]
        for slot, i in enumerate(keep):
            boxes.append(corners[slot])
            scores.append(float(raw_scores[i]))
            landmarks.append(kps[slot].astype(np.float32))

    if not boxes:
        return []
    xywh = [(float(x1), float(y1), float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in boxes]
    keep_indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    faces = []
    for index in np.array(keep_indices).reshape(-1):
        faces.append(DetectedFace(boxes[int(index)], scores[int(index)], landmarks[int(index)]))
    return faces


class FaceLandmarker:
    """2DFAN4: 68 landmarks from a bounding box, used for a steadier 5-point
    alignment than the detector's own keypoints."""

    def __init__(self, provider: str = "trt", log=None, force_trt: bool = False,
                 stream_ptr: int | None = None):
        self.size = int(LANDMARKER_MODEL["size"])
        provider = runtime_provider(provider, "landmarker", log, force_trt)
        path = model_path(LANDMARKER_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"face landmarker ONNX not found: {path}")
        self.session = ort.InferenceSession(
            str(path), sess_options=_session_options(provider),
            providers=onnx_providers(provider, "landmarker", stream_ptr=stream_ptr),
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray, bounding_box: np.ndarray) -> tuple[np.ndarray, float]:
        scale = 195.0 / max(1.0, float(np.subtract(bounding_box[2:], bounding_box[:2]).max()))
        translation = (self.size - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]], dtype=np.float32)
        crop = cv2.warpAffine(frame_bgr, matrix, (self.size, self.size))
        crop = _optimize_contrast(crop)
        tensor = np.expand_dims(crop.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
        landmark, heatmap = self.session.run(None, {self.input_name: tensor})[:2]
        landmark_68 = landmark[:, :, :2][0] / 64.0 * self.size
        landmark_68 = _transform_points(landmark_68, cv2.invertAffineTransform(matrix))
        score = float(np.interp(np.mean(np.amax(heatmap, axis=(2, 3))), [0, 0.9], [0, 1]))
        return landmark_68.astype(np.float32), score


def _optimize_contrast(crop: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    if float(np.mean(lab[:, :, 0])) < 30:
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return crop


class FaceParser:
    """BiSeNet ResNet-34 face parsing; supplies the region mask and the
    per-part gates the retouch passes need."""

    def __init__(self, provider: str = "trt", log=None, force_trt: bool = False,
                 stream_ptr: int | None = None):
        self.size = int(PARSER_MODEL["size"])
        provider = runtime_provider(provider, "parser", log, force_trt)
        path = model_path(PARSER_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"face parser ONNX not found: {path}")
        self.session = ort.InferenceSession(
            str(path), sess_options=_session_options(provider),
            providers=onnx_providers(provider, "parser", stream_ptr=stream_ptr),
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

    def parse(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a per-pixel class-id map at the crop's own resolution."""
        resized = cv2.resize(crop_bgr, (self.size, self.size))
        tensor = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        labels = logits.argmax(0).astype(np.uint8)
        if labels.shape[0] != crop_bgr.shape[0]:
            labels = cv2.resize(labels, crop_bgr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        return labels

    @staticmethod
    def class_mask(labels: np.ndarray, classes: tuple[int, ...], sigma: float = 5.0) -> np.ndarray:
        # LUT indexing rather than numpy.isin: the label map is uint8, so a
        # 256-entry table turns the membership test into a single gather.
        return _feather(_class_lut(classes)[labels], sigma)


class FaceEnhancer:
    """GFPGAN / RestoreFormer++ blind face restoration at 512x512.

    Runs on TensorRT in **fp32**, not fp16: the StyleGAN-style decoder overflows
    in half precision and the engine returns a washed-out, darkened face
    (measured mean 83 / std 16 against 180 / 43 on the same crop) instead of
    failing loudly. The fp32 engine matches CUDA numerically and is ~2x faster
    (30 ms vs 55 ms), which makes it the largest single win in the pipeline.

    Its engine must be built in isolation. Building *any* TensorRT engine while
    another TensorRT session is alive segfaults this ORT/TRT build; measured:

      fp32 alone                        -> ok, 30.3 ms
      fp32 after a CUDA-EP session      -> ok, 36.4 ms
      fp32 after a TensorRT-EP session  -> segfault during the build
      cached fp32 after a TRT session   -> ok, 29.5 ms

    So only the *build* is unsafe, not the load. :func:`build_trt` creates and
    frees one session at a time, and the CLI pre-builds before the engine is
    assembled, so the runtime path only ever deserialises. This matches the
    multi-session GPU thread-safety reports in onnxruntime#26610 / #16790."""

    def __init__(self, name: str = DEFAULT_ENHANCER, provider: str = "trt", log=None,
                 force_trt: bool = False, stream_ptr: int | None = None):
        self.name = normalize_enhancer(name)
        entry = enhancer_entry(self.name)
        self.size = int(entry["size"])
        self.template = str(entry.get("template", DEFAULT_TEMPLATE))
        # Distinct from any fp16 engine an older build may have left behind.
        self.trt_cache_key = f"{self.name}_fp32"
        provider = runtime_provider(provider, self.trt_cache_key, log, force_trt)
        path = model_path(entry)
        if not path.exists():
            raise FileNotFoundError(f"face enhancer ONNX not found: {path}")
        self.session = ort.InferenceSession(
            str(path), sess_options=_session_options(provider),
            providers=onnx_providers(provider, self.trt_cache_key, fp16=False,
                                     stream_ptr=stream_ptr),
        )
        self.providers = self.session.get_providers()
        self.input_names = [i.name for i in self.session.get_inputs()]

    def enhance(self, crop_bgr: np.ndarray) -> np.ndarray:
        """``crop_bgr`` is the shared CROP_SIZE crop; it is resampled to the
        model's own resolution and back, so a 256 restorer really does run at
        256 while every other stage keeps working at 512."""
        crop_size = crop_bgr.shape[0]
        if crop_size != self.size:
            crop_bgr = cv2.resize(crop_bgr, (self.size, self.size),
                                  interpolation=cv2.INTER_AREA if self.size < crop_size else cv2.INTER_LINEAR)
        tensor = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
        feeds = {self.input_names[0]: tensor}
        # CodeFormer-style graphs take a fidelity 'weight' scalar; the
        # Apache-2.0 models we ship do not, but keep the branch cheap and safe.
        if "weight" in self.input_names:
            feeds["weight"] = np.array([1.0], dtype=np.double)
        output = self.session.run(None, feeds)[0][0]
        output = np.clip(output, -1, 1)
        output = ((output + 1) / 2).transpose(1, 2, 0)
        result = (output * 255.0).round().astype(np.uint8)[:, :, ::-1]
        if crop_size != self.size:
            result = cv2.resize(result, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        return result


def _session_options(provider: str) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "trt":
        # The TensorRT parser logs benign errors for nodes it hands back to
        # CUDA; suppress that noise but keep our own failures visible.
        opts.log_severity_level = 4
    return opts


# --- retouch passes ---------------------------------------------------------
# All operate on the aligned 512 crop and are gated by a parse mask (or the box
# mask when the parser is off). Strengths are 0..1.


SMOOTH_SCALE = 0.5      # bilateral runs at half res; skin smoothing is low-frequency anyway


def apply_skin_smooth(crop: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Edge-preserving skin smoothing.

    The bilateral runs at half resolution with a fixed kernel: at full res with
    a derived kernel (d=0) this single pass costs ~150 ms per face and dominates
    the whole pipeline, while the visible result is indistinguishable -- what it
    removes is exactly the detail that does not survive the downscale."""
    if strength <= 0:
        return crop
    height, width = crop.shape[:2]
    small = cv2.resize(crop, (int(width * SMOOTH_SCALE), int(height * SMOOTH_SCALE)),
                       interpolation=cv2.INTER_AREA)
    sigma_color = 20.0 + 60.0 * strength
    smoothed = cv2.bilateralFilter(small, 7, sigma_color, 7)
    smoothed = cv2.resize(smoothed, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = (mask * strength)[:, :, None]
    return (crop.astype(np.float32) * (1 - alpha) + smoothed.astype(np.float32) * alpha).astype(np.uint8)


@dataclass
class LabAdjustment:
    """One masked tonal tweak, applied inside a shared LAB round-trip."""

    mask: np.ndarray
    strength: float
    luma_gain: float = 0.0
    blue_shift: float = 0.0
    chroma_gain: float = 1.0
    even_chroma: bool = False       # pull chroma toward the masked mean


def apply_lab_adjustments(crop: np.ndarray, adjustments: list[LabAdjustment]) -> np.ndarray:
    """Apply every tonal pass in one BGR->LAB->BGR round-trip.

    Each conversion pair costs ~14 ms at 512x512, so running brighten / even /
    eye / teeth / lip separately would spend ~70 ms on colour-space churn alone.
    Every pass is additive or affine per channel, so they accumulate in place on
    the one or two channels they actually touch -- no per-pass 3-channel copy."""
    active = [a for a in adjustments if a.strength > 0]
    if not active:
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    for adjustment in active:
        mask = adjustment.mask
        if adjustment.even_chroma:
            total = float(mask.sum())
            if total >= 1.0:
                for channel in (1, 2):
                    plane = lab[:, :, channel]
                    target = float((plane * mask).sum() / total)
                    plane += (target - plane) * (adjustment.strength * mask)
        if adjustment.luma_gain:
            lab[:, :, 0] += (adjustment.luma_gain * adjustment.strength) * mask
        if adjustment.blue_shift:
            lab[:, :, 2] += (adjustment.blue_shift * adjustment.strength) * mask
        if adjustment.chroma_gain != 1.0:
            factor = (adjustment.chroma_gain - 1.0) * adjustment.strength
            for channel in (1, 2):
                plane = lab[:, :, channel]
                plane += (plane - 128.0) * (factor * mask)
    np.clip(lab, 0, 255, out=lab)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)


def apply_sharpen(crop: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return crop
    blurred = cv2.GaussianBlur(crop, (0, 0), 1.6)
    return cv2.addWeighted(crop, 1.0 + strength, blurred, -strength, 0)


def retouch_crop(crop: np.ndarray, labels: np.ndarray | None,
                 options: "BeautyOptions", box_mask: np.ndarray) -> np.ndarray:
    """Run every retouch pass over an aligned BGR crop.

    Shared by the CPU engine and the GPU processor. ``labels`` is the parser's
    class map (or ``None``, in which case the box mask stands in for the skin
    mask and the per-part passes are skipped). Kept on the host in both paths:
    the crop is a fixed 512x512 regardless of frame size, so the transfer is
    ~0.3 ms while a GPU equivalent needs a hand-written LAB round trip."""
    skin_mask = FaceParser.class_mask(labels, SKIN_CLASSES) if labels is not None else box_mask
    crop = apply_skin_smooth(crop, skin_mask, options.skin_smooth)
    adjustments = [
        LabAdjustment(skin_mask, options.skin_even, even_chroma=True),
        LabAdjustment(skin_mask, options.skin_brighten, luma_gain=28.0),
    ]
    if labels is not None:
        if options.eye_brighten > 0:
            adjustments.append(LabAdjustment(FaceParser.class_mask(labels, EYE_CLASSES, sigma=2.0),
                                             options.eye_brighten, luma_gain=25.0))
        if options.teeth_white > 0:
            adjustments.append(LabAdjustment(FaceParser.class_mask(labels, TEETH_CLASSES, sigma=2.0),
                                             options.teeth_white, luma_gain=20.0, blue_shift=-12.0))
        if options.lip_vivid > 0:
            adjustments.append(LabAdjustment(FaceParser.class_mask(labels, LIP_CLASSES, sigma=2.0),
                                             options.lip_vivid, chroma_gain=1.5))
    crop = apply_lab_adjustments(crop, adjustments)
    return apply_sharpen(crop, options.sharpen)


# --- temporal tracking ------------------------------------------------------


class _FaceTracker:
    """Match faces frame-to-frame by bounding-box overlap and EMA-smooth their
    5-point landmarks. Without this, per-frame alignment jitter of a pixel or
    two turns into visible face wobble in the output video."""

    def __init__(self, strength: float) -> None:
        self.strength = float(np.clip(strength, 0.0, 0.95))
        self._tracks: list[tuple[np.ndarray, np.ndarray]] = []   # (box, landmark_5)

    def reset(self) -> None:
        self._tracks = []

    def smooth(self, faces: list[DetectedFace]) -> None:
        if self.strength <= 0:
            return
        next_tracks: list[tuple[np.ndarray, np.ndarray]] = []
        for face in faces:
            previous = self._match(face.bounding_box)
            if previous is not None:
                face.landmark_5 = (previous * self.strength + face.landmark_5 * (1.0 - self.strength)).astype(np.float32)
            next_tracks.append((face.bounding_box.copy(), face.landmark_5.copy()))
        self._tracks = next_tracks

    def _match(self, box: np.ndarray) -> np.ndarray | None:
        best_iou = 0.3
        best: np.ndarray | None = None
        for previous_box, previous_landmark in self._tracks:
            iou = _iou(box, previous_box)
            if iou > best_iou:
                best_iou = iou
                best = previous_landmark
        return best


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


# --- options ----------------------------------------------------------------


@dataclass
class BeautyOptions:
    """Everything the UI exposes. Percent-style knobs are 0..100 on the CLI and
    normalized to 0..1 here."""

    enhancer: str = DEFAULT_ENHANCER
    enhancer_blend: float = 0.8            # global blend of the restored face
    skin_smooth: float = 0.0
    skin_brighten: float = 0.0
    skin_even: float = 0.0
    eye_brighten: float = 0.0
    teeth_white: float = 0.0
    lip_vivid: float = 0.0
    sharpen: float = 0.0
    mask_blur: float = 0.3
    mask_padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    use_region_mask: bool = True
    use_landmarker: bool = True
    detector_score: float = 0.5
    # VR: process each face through a gnomonic flat view. The affine face
    # warp assumes a perspective camera, so on an equirect eye it is only
    # correct near the centre; reprojecting first makes it correct anywhere.
    vr_reproject: str = "auto"          # auto (SBS sources) / off / on
    vr_fov_margin: float = 1.6          # flat view fov vs the face box
    vr_max_fov: float = 120.0           # past this, gnomonic stretch costs more than it gains
    vr_feather: float = 0.08            # flat-view edge feather, fraction of half-extent
    # Equirect distortion is a function of latitude, not longitude: a face
    # anywhere along the horizon is already almost rectilinear. Measured
    # alignment error the affine warp cannot undo, in pixels of the 512 crop:
    # pitch 0-15 deg ~0.04, 20 deg 3.2, 40 deg 18.4, 60 deg 58. Below this
    # threshold reprojection costs ~4.4 ms/face and buys under 1% of the crop.
    vr_min_pitch: float = 20.0
    detect_mode: str = DEFAULT_DETECT_MODE
    detect_interval: int = 1               # reuse boxes for N-1 frames
    landmark_interval: int = 1             # reuse refined 5-point landmarks for N-1 frames
    # Between full sweeps, detect only around faces found last time. Only
    # engages when the frame needs more than one detection window.
    detect_roi: bool = True
    roi_sweep_interval: int = 15           # detections between full sweeps
    min_face_mode: str = DEFAULT_MIN_FACE_MODE
    max_faces: int = 0                     # 0 = every detected face
    temporal_smooth: float = 0.5
    provider: str = "trt"

    def needs_parser(self) -> bool:
        return bool(self.use_region_mask) or any((
            self.skin_smooth, self.skin_brighten, self.skin_even,
            self.eye_brighten, self.teeth_white, self.lip_vivid,
        ))

    def retouch_summary(self) -> str:
        parts = [f"enhancer={self.enhancer}"]
        if self.enhancer != ENHANCER_NONE:
            parts.append(f"blend={self.enhancer_blend:.2f}")
        for label, value in (
            ("smooth", self.skin_smooth), ("brighten", self.skin_brighten), ("even", self.skin_even),
            ("eye", self.eye_brighten), ("teeth", self.teeth_white), ("lip", self.lip_vivid),
            ("sharpen", self.sharpen),
        ):
            if value > 0:
                parts.append(f"{label}={value:.2f}")
        parts.append(f"region_mask={'on' if self.use_region_mask else 'off'}")
        parts.append(f"landmarker={'on' if self.use_landmarker else 'off'}")
        parts.append(f"det={DETECTOR_SIZE}@{self.detector_score:.2f}")
        parts.append(f"detect={self.detect_mode}/{self.detect_interval}")
        parts.append(f"landmark_interval={self.landmark_interval}")
        parts.append(f"vr={self.vr_reproject}")
        parts.append(f"min_face={self.min_face_mode}")
        parts.append(f"temporal={self.temporal_smooth:.2f}")
        return " ".join(parts)


# --- presets ----------------------------------------------------------------
# Only the beauty knobs are preset; detection/mask/stability settings are
# orthogonal to "how strong should this look" and keep one good default across
# all presets (see BeautyOptions).
#
# The restoration blend follows the values the FaceFusion community converged
# on -- 80-90 reads natural, 100 is the maximum-detail setting, and dropping
# toward 50 keeps more of the original face. The retouch values have no such
# reference (FaceFusion has no skin retouch at all) and are set so that
# "standard" is visible but not plastic.
PRESET_NATURAL = "natural"
PRESET_STANDARD = "standard"
PRESET_STRONG = "strong"
PRESET_RESTORE = "restore"
PRESET_CUSTOM = "custom"
DEFAULT_PRESET = PRESET_STANDARD

# (enhancer_blend, skin_smooth, skin_brighten, skin_even, eye_brighten,
#  teeth_white, lip_vivid, sharpen), all as 0..100 percent.
PRESETS: dict[str, tuple[int, ...]] = {
    PRESET_RESTORE:  (85,  0,  0,  0,  0,  0,  0,  0),
    PRESET_NATURAL:  (60, 20,  0, 10, 10, 10,  0, 10),
    PRESET_STANDARD: (80, 40, 15, 20, 20, 20, 10, 15),
    PRESET_STRONG:  (100, 60, 25, 35, 35, 35, 25, 20),
}
PRESET_FIELDS = (
    "enhancer_blend", "skin_smooth", "skin_brighten", "skin_even",
    "eye_brighten", "teeth_white", "lip_vivid", "sharpen",
)


def normalize_preset(name: str) -> str:
    value = str(name or "").strip().lower()
    return value if value in PRESETS else DEFAULT_PRESET


def preset_percentages(name: str) -> dict[str, int]:
    """The preset's knob values as 0..100 percentages, keyed by option name."""
    return dict(zip(PRESET_FIELDS, PRESETS[normalize_preset(name)]))


def preset_options(name: str, **overrides) -> BeautyOptions:
    values = {field: percent / 100.0 for field, percent in preset_percentages(name).items()}
    values.update(overrides)
    return BeautyOptions(**values)


def match_preset(percentages: dict[str, int]) -> str:
    """Reverse lookup: which preset a set of knob values corresponds to, or
    ``custom``. Lets the UI keep the preset dropdown honest after hand-tuning."""
    for name in PRESETS:
        if all(int(percentages.get(field, -1)) == value
               for field, value in preset_percentages(name).items()):
            return name
    return PRESET_CUSTOM


# --- engine -----------------------------------------------------------------


@dataclass
class FrameStats:
    faces: int = 0
    processed: int = 0


class FaceBeautyEngine:
    """Detect, restore and retouch every face in a frame."""

    def __init__(self, options: BeautyOptions, log=print) -> None:
        self.options = options
        self.log = log
        provider = str(options.provider or "trt").lower()
        with _ENGINE_LOCK:
            self.detector = FaceDetector(provider=provider)
            self.landmarker = FaceLandmarker(provider=provider) if options.use_landmarker else None
            self.parser = FaceParser(provider=provider) if options.needs_parser() else None
            self.enhancer = (
                FaceEnhancer(options.enhancer, provider=provider)
                if normalize_enhancer(options.enhancer) != ENHANCER_NONE else None
            )
        self.crop_size = CROP_SIZE
        self.template = self.enhancer.template if self.enhancer else DEFAULT_TEMPLATE
        self.tracker = _FaceTracker(options.temporal_smooth)
        self._box_mask = create_box_mask(self.crop_size, options.mask_blur, options.mask_padding)
        self._tiles: list[tuple[int, int, int, int]] | None = None
        self._cached_faces: list[DetectedFace] = []
        self._frame_index = 0

    def provider_summary(self) -> str:
        parts = [f"detector={self.detector.providers[0] if self.detector.providers else 'unknown'}"]
        if self.enhancer:
            parts.append(f"enhancer={self.enhancer.providers[0] if self.enhancer.providers else 'unknown'}")
        return " ".join(parts)

    def reset(self) -> None:
        self.tracker.reset()
        self._cached_faces = []
        self._frame_index = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, FrameStats]:
        options = self.options
        stats = FrameStats()
        if self._tiles is None:
            self._tiles = plan_detection_tiles(
                frame_bgr.shape[1], frame_bgr.shape[0], options.detect_mode)
            if len(self._tiles) > 1:
                self.log(f"detection: {len(self._tiles)} windows "
                         f"({options.detect_mode}), every {options.detect_interval} frame(s)")
        interval = max(1, int(options.detect_interval))
        if self._frame_index % interval == 0 or not self._cached_faces:
            if len(self._tiles) == 1 and self._tiles[0][2] == frame_bgr.shape[1]:
                self._cached_faces = self.detector.detect(frame_bgr, options.detector_score)
            else:
                self._cached_faces = self.detector.detect_tiled(
                    frame_bgr, options.detector_score, self._tiles)
        faces = self._cached_faces
        self._frame_index += 1
        # Resolved per frame so the same option works for 720p and 8K sources.
        min_face_px = resolve_min_face_px(options.min_face_mode, frame_bgr.shape[0])
        faces = [f for f in faces if min(f.bounding_box[2] - f.bounding_box[0],
                                         f.bounding_box[3] - f.bounding_box[1]) >= min_face_px]
        stats.faces = len(faces)
        if not faces:
            self.tracker.smooth(faces)
            return frame_bgr, stats
        faces.sort(key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) *
                                 (f.bounding_box[3] - f.bounding_box[1]), reverse=True)
        if options.max_faces > 0:
            faces = faces[:options.max_faces]

        if self.landmarker is not None:
            for face in faces:
                try:
                    landmark_68, score = self.landmarker.detect(frame_bgr, face.bounding_box)
                    face.landmark_score = score
                    if score > 0.1:
                        face.landmark_5 = landmark_68_to_5(landmark_68)
                except Exception as exc:
                    self.log(f"landmarker failed, using detector keypoints: {type(exc).__name__}: {exc}")
        self.tracker.smooth(faces)

        for face in faces:
            frame_bgr = self._process_face(frame_bgr, face)
            stats.processed += 1
        return frame_bgr, stats

    def _process_face(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        options = self.options
        crop, matrix = warp_face(frame_bgr, face.landmark_5, self.crop_size, self.template)

        # Parse the source crop, not the generated result.  Parsing afterwards
        # lets the enhancer validate its own hallucination: a fake frontal face
        # generated from hair is then classified as skin/eyes and pasted back.
        labels = self.parser.parse(crop) if self.parser is not None else None
        source_region_mask = (
            FaceParser.class_mask(labels, REGION_MASK_CLASSES) if labels is not None else None
        )
        source_face_coverage = (
            float(source_region_mask.mean()) if source_region_mask is not None else None
        )

        if self.enhancer is not None and enhancer_is_safe(face, source_face_coverage):
            # Blend the restoration against the original crop straight away, so
            # the strength knob scales only the restorer -- the retouch passes
            # below then run at the strength the user actually dialled in.
            enhanced = self.enhancer.enhance(crop)
            blend = float(np.clip(options.enhancer_blend, 0.0, 1.0))
            crop = enhanced if blend >= 1.0 else cv2.addWeighted(enhanced, blend, crop, 1.0 - blend, 0)

        crop = retouch_crop(crop, labels, options, self._box_mask)

        mask = self._box_mask
        if source_region_mask is not None and options.use_region_mask:
            mask = np.minimum(mask, source_region_mask)
        return paste_back(frame_bgr, crop, mask.clip(0, 1), matrix)


# --- TensorRT prebuild ------------------------------------------------------


def trt_cache_keys(options: BeautyOptions) -> list[str]:
    """Engine-cache keys this configuration needs."""
    keys = ["detector"]
    if options.use_landmarker:
        keys.append("landmarker")
    if options.needs_parser():
        keys.append("parser")
    enhancer = normalize_enhancer(options.enhancer)
    if enhancer != ENHANCER_NONE:
        keys.append(f"{enhancer}_fp32")
    return keys


def trt_cache_ready(options: BeautyOptions) -> bool:
    return all(trt_engine_cached(key) for key in trt_cache_keys(options))


def build_trt_stage(key: str, options: BeautyOptions, log=print) -> int:
    """Build exactly one engine. Must be the only TensorRT work in the process --
    see :func:`build_trt`."""
    import time

    dummy_frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    dummy_crop = (np.random.rand(CROP_SIZE, CROP_SIZE, 3) * 255).astype(np.uint8)
    dummy_box = np.array([540.0, 260.0, 740.0, 460.0], dtype=np.float32)
    enhancer = normalize_enhancer(options.enhancer)

    # These construct with provider="trt" *before* the engine exists, which is
    # exactly what runtime_provider forbids elsewhere -- building is this
    # function's whole job, and it runs alone in its own process.
    if key == "detector":
        run = lambda: FaceDetector(provider="trt", force_trt=True).detect(dummy_frame, 0.5)
    elif key == "landmarker":
        run = lambda: FaceLandmarker(provider="trt", force_trt=True).detect(dummy_frame, dummy_box)
    elif key == "parser":
        run = lambda: FaceParser(provider="trt", force_trt=True).parse(dummy_crop)
    elif key == f"{enhancer}_fp32":
        run = lambda: FaceEnhancer(enhancer, provider="trt", force_trt=True).enhance(dummy_crop)
    else:
        log(f"build-trt: unknown stage {key}")
        return 2

    log(f"build-trt: building TensorRT engine for {key} (first time is slow)...")
    started = time.time()
    try:
        run()
    except Exception as exc:
        log(f"build-trt: {key} failed: {type(exc).__name__}: {exc}")
        return 1
    mark_trt_ready(key)
    log(f"build-trt: {key} ready in {time.time() - started:.1f}s")
    return 0


def _build_trt_in_process(options: BeautyOptions, log=print) -> int:
    """Sequential in-process build. Only safe when nothing else has created a
    TensorRT session; the CLI prefers the subprocess-per-stage path."""
    import gc
    import time

    rc = 0
    for key in trt_cache_keys(options):
        rc |= build_trt_stage(key, options, log=log)
        gc.collect()
    return rc
