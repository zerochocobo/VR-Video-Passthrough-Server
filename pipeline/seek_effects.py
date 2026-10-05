"""Processing settings and GPU stages for seekable video effects.

Imports stay light until a stage is constructed. A layout owns a settings
snapshot so its frame cache never mixes different live control values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import config


def effect_settings(mode: str, path: Path) -> dict:
    from utils.runtime_settings import get_face_beauty, get_light_match
    from pipeline.subtitles import find_subtitle_for_video

    prefixes = {
        "green": ("RVM_", "COMPOSITE_"),
        "alpha": ("RVM_", "ALPHA_"),
        "superres": ("RTX_VSR_",),
        "dlss5": ("DLSS5_",),
        "face_beauty": ("FACE_BEAUTY_",),
        "two_dvr": ("TWO_DVR_",),
        "rm": ("RM_",),
    }.get(mode, ()) + ("SUBTITLE_", "PASSTHROUGH_PYNV_", "PASSTHROUGH_HEVC_")
    values = {k: v for k, v in vars(config).items()
              if k.startswith(prefixes) and (v is None or isinstance(v, (str, int, float, bool, tuple)))}
    values["COMPOSITE_BG_RGB"] = config.COMPOSITE_BG_RGB
    snapshot = {"config": values}
    if mode in {"green", "alpha"}:
        # Keep the version in the snapshot for the GPU table cache; the digest
        # below ignores it so returning to identical settings reuses frames.
        snapshot["light_match"] = get_light_match().to_dict()
    if mode == "face_beauty":
        snapshot["face_beauty"] = get_face_beauty().to_dict()
    subtitle = find_subtitle_for_video(path)
    if subtitle is not None:
        st = subtitle.stat()
        snapshot["subtitle"] = (str(subtitle), st.st_size, st.st_mtime_ns)
    return snapshot


def stage_config(settings: dict | None):
    return SimpleNamespace(**(vars(config) | (settings or {}).get("config", {})))


def settings_digest(settings: dict | None) -> str:
    clean = {k: ({n: v for n, v in value.items() if n != "version"}
                 if isinstance(value, dict) else value)
             for k, value in (settings or {}).items()}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def iter_stage_frames(stage, frames):
    """Preserve presentation indices for both single-frame and chunk stages."""
    if hasattr(stage, "iter_process"):
        yield from stage.iter_process(frames)
    else:
        for index, frame in frames:
            yield index, stage.process(frame, index)


class _RgbStage:
    def __init__(self, width: int, height: int, *, output_width: int | None = None, settings=None):
        import cupy as cp
        from offline.two_dvr_pynv import _NV12_RGB_KERNELS

        self.cp = cp
        self.config = stage_config(settings)
        self.width, self.height = width, height
        self.out_w, self.out_h = output_width or width, height
        module = cp.RawModule(code=_NV12_RGB_KERNELS)
        self.module = module
        self.to_rgb = module.get_function("nv12_to_rgb")
        self.p016_to_rgb = module.get_function("p016_to_rgb")
        self.to_nv12 = module.get_function("rgb_to_nv12")
        self.block = (16, 16, 1)
        self.grid = ((width + 15) // 16, (height + 15) // 16, 1)
        self.out_grid = ((self.out_w + 15) // 16, (height + 15) // 16, 1)
        self.ring = [cp.empty((height * 3 // 2, self.out_w), cp.uint8)
                     for _ in range(max(2, int(config.PASSTHROUGH_NV12_RING_SLOTS)))]

    @property
    def output_size(self):
        return self.out_w, self.out_h

    def rgb(self, frame):
        import numpy as np
        from pipeline.pynv_io import GpuP016Frame

        cp = self.cp
        rgb = cp.empty((self.height, self.width, 3), cp.uint8)
        p016 = isinstance(frame, GpuP016Frame)
        dtype = cp.uint16 if p016 else cp.uint8
        y = frame.y.as_cupy(dtype).reshape(self.height, self.width)
        uv = frame.uv.as_cupy(dtype).reshape(self.height // 2, self.width)
        args = (y, uv, rgb, np.int32(self.width), np.int32(self.height))
        if p016:
            args += (np.int32(self.config.PASSTHROUGH_PYNV_10BIT_SHIFT),)
        (self.p016_to_rgb if p016 else self.to_rgb)(self.grid, self.block, args)
        return rgb

    def nv12(self, rgb, index):
        import numpy as np

        out = self.ring[index % len(self.ring)]
        self.to_nv12(self.out_grid, self.block,
                     (rgb, out, np.int32(self.out_w), np.int32(self.out_h)))
        self.cp.cuda.get_current_stream().synchronize()
        return out


class FaceBeautyStage(_RgbStage):
    def __init__(self, width, height, *, settings=None, logger=None):
        from offline import face_beauty_engine as fb
        from offline.face_beauty_gpu import GpuFaceBeautyProcessor
        from utils.runtime_settings import get_face_beauty, FACE_BEAUTY_STRENGTH_KEYS

        super().__init__(width, height, settings=settings)
        live = (settings or {}).get("face_beauty") or get_face_beauty().to_dict()
        options = fb.preset_options(live["preset"] if live["preset"] in fb.PRESETS else fb.DEFAULT_PRESET,
                                   provider="trt", enhancer=str(self.config.FACE_BEAUTY_MODEL),
                                   detect_interval=self.config.FACE_BEAUTY_DETECT_INTERVAL)
        options.landmark_interval = self.config.FACE_BEAUTY_LANDMARK_INTERVAL
        options.max_faces = self.config.FACE_BEAUTY_MAX_FACES_VR if width >= 2 * height else self.config.FACE_BEAUTY_MAX_FACES_2D
        for key in FACE_BEAUTY_STRENGTH_KEYS:
            setattr(options, key, max(0.0, min(1.0, live[key] / 100.0)))
        self.processor = GpuFaceBeautyProcessor(options, log=logger.info if logger else print)
        self.processor.reset()

    def process(self, frame, index):
        rgb = self.rgb(frame)
        self.processor.process(rgb)
        return self.nv12(rgb, index)


class TwoDvrStage(_RgbStage):
    def __init__(self, width, height, *, settings=None, logger=None):
        from offline.da3_depth import ensure_model_available, warmup_depth_engine
        from offline.two_dvr import _ensure_trt_cache
        from offline.two_dvr_gpu import GpuStereoRenderer
        from offline.two_dvr_pynv import _letterbox_box
        from offline.two_dvr_render import DEFAULT_FLAT_FOV_DEG, PROJECTION_FLAT_3D, effective_eye_distance_mm
        from utils.scene_detection import SceneCutDetector

        if width > 4096:
            raise RuntimeError("2D->3D source width exceeds 4096px")
        super().__init__(width, height, output_width=width * 2, settings=settings)
        emit = logger.info if logger else print
        model = self.config.TWO_DVR_MODEL
        if not ensure_model_available(model, log=emit):
            raise RuntimeError(f"DA3 model {model} unavailable")
        _ensure_trt_cache(model, "trt")
        self.engine = warmup_depth_engine(variant=model, provider="trt", log=emit)
        if not self.engine.folded:
            raise RuntimeError("2D->3D requires folded-preprocess DA3 model")
        names = ("temporal_norm", "temporal_norm_alpha", "temporal_norm_reset", "temporal_depth",
                 "temporal_depth_mode", "temporal_depth_alpha", "temporal_flow_diff", "temporal_flow_consistency",
                 "temporal_flow_motion_gate", "temporal_affine", "temporal_affine_max_scale", "temporal_affine_max_bias",
                 "temporal_static_deadband_px", "temporal_static_max_step_px", "temporal_motion_max_step_px")
        self.renderer = GpuStereoRenderer(width, height, PROJECTION_FLAT_3D,
                                         effective_eye_distance_mm(self.config.TWO_DVR_EYE_DISTANCE_MM, self.config.TWO_DVR_STRENGTH),
                                         self.config.TWO_DVR_HOLE_FILL, DEFAULT_FLAT_FOV_DEG,
                                         **{n: getattr(self.config, "TWO_DVR_" + n.upper()) for n in names})
        self.renderer.reset()
        self.cut_detector = SceneCutDetector(threshold=self.config.TWO_DVR_SCENE_CUT_THRESHOLD) if self.config.TWO_DVR_SCENE_CUT else None
        self.box = _letterbox_box(width, height, self.engine.size)
        self.canvas = self.cp.zeros((self.engine.size, self.engine.size, 3), self.cp.uint8)
        self.letterbox = self.module.get_function("nv12_to_rgb_letterbox")
        self.p016_letterbox = self.module.get_function("p016_to_rgb_letterbox")

    def process(self, frame, index):
        import numpy as np

        # Use the same GPU letterbox kernel as live; only the small DA3 input
        # crosses PCIe. The full decoded frame remains on the GPU.
        rgb = self.rgb(frame)
        x0, y0, nw, nh = self.box
        canvas = self.canvas
        from pipeline.pynv_io import GpuP016Frame
        p016 = isinstance(frame, GpuP016Frame)
        dtype = self.cp.uint16 if p016 else self.cp.uint8
        kernel = self.p016_letterbox if p016 else self.letterbox
        args = (frame.y.as_cupy(dtype).reshape(self.height, self.width),
                frame.uv.as_cupy(dtype).reshape(self.height // 2, self.width), canvas,
                np.int32(self.width), np.int32(self.height), np.int32(self.engine.size),
                np.int32(x0), np.int32(y0), np.int32(nw), np.int32(nh))
        if p016:
            args += (np.int32(self.config.PASSTHROUGH_PYNV_10BIT_SHIFT),)
        kernel(((self.engine.size + 15) // 16, (self.engine.size + 15) // 16, 1), self.block, args)
        host = canvas.get()
        if self.cut_detector is not None and self.cut_detector.step(host):
            self.renderer.reset()
        depth = self.engine.session.run([self.engine.output_name], {self.engine.input_name: host[None]})[0][0]
        near = self.renderer.prepare_near_gpu(depth[y0:y0 + nh, x0:x0 + nw], canvas[y0:y0 + nh, x0:x0 + nw])
        return self.nv12(self.renderer.render_into_gpu(rgb, near), index)


class RmStage(_RgbStage):
    def __init__(self, width, height, *, settings=None, logger=None):
        from pipeline.demosaic import GpuRmProcessor, get_shared_engines, WINDOW

        super().__init__(width, height, settings=settings)
        self.window = WINDOW
        self.processor = GpuRmProcessor(get_shared_engines(provider="trt", log=logger.info if logger else print),
                                        detect_interval=self.config.RM_DETECT_INTERVAL)

    def iter_process(self, frames):
        chunk = []
        for index, frame in frames:
            chunk.append((index, self.rgb(frame)))
            if len(chunk) == self.window:
                yield from self._chunk(chunk)
                chunk = []
        if chunk:
            yield from self._chunk(chunk)

    def _chunk(self, chunk):
        rgb = [item[1] for item in chunk]
        rgb += [rgb[-1]] * (self.window - len(rgb))
        output = self.processor.process_chunk(rgb, float(self.config.RM_CONF))
        for (index, _), result in zip(chunk, output):
            yield index, self.nv12(result, index)
