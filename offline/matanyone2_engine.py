"""Shared MatAnyone2 ONNX offline engine."""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import numpy as np

import config
from pipeline.matanyone2_roi import RoiMeta, roi_from_mask


SessionProviderFactory = Callable[[str, Path], list]
AlphaPackerFactory = Callable[[object], object]
OutputMode = Literal["green", "alpha"]


_STATE_NAMES = (
    "memory_key",
    "memory_shrinkage",
    "msk_value",
    "obj_memory",
    "sensory",
    "last_mask",
    "last_pix_feat",
    "last_pred_mask",
    "last_uncert",
)


def _dtype_from_onnx_type(type_name: str):
    return np.float16 if type_name == "tensor(float16)" else np.float32


class MatAnyone2OnnxEngine:
    """Shared MatAnyone2 propagation engine for green and alpha offline output."""

    class _EyeState:
        def __init__(self):
            self.sensory = None
            self.memory_key = None
            self.memory_shrinkage = None
            self.memory_msk_value = None
            self.obj_memory = None
            self.last_pix_feat = None
            self.last_mask = None
            self.last_msk_value = None
            self.last_uncert = None
            self.initialized = False

        def reset(self):
            self.sensory = None
            self.memory_key = None
            self.memory_shrinkage = None
            self.memory_msk_value = None
            self.obj_memory = None
            self.last_pix_feat = None
            self.last_mask = None
            self.last_msk_value = None
            self.last_uncert = None
            self.initialized = False

    def __init__(
        self,
        model_dir: Path,
        mask: Path | None,
        sam3_dir: Path,
        session_providers: SessionProviderFactory,
        sam3_prompt: str = "person",
        bootstrap_threshold: float = 0.55,
        bootstrap_erode: int = 1,
        bootstrap_dilate: int = 0,
        bootstrap_soft: bool = False,
        segment_frames: int = 300,
        use_fused_update: bool = False,
        use_step_update: bool = True,
        output_mode: OutputMode = "green",
        alpha_packer_factory: AlphaPackerFactory | None = None,
        log_prefix: str = "[offline]",
    ):
        import cv2
        import onnxruntime as ort

        from pipeline.matting import Matter, _AlphaSmoother

        self.cv2 = cv2
        self.np = np
        self.ort = ort
        self.model_dir = model_dir
        self.mask_path = mask
        self.sam3_dir = sam3_dir
        self.sam3_prompt = sam3_prompt
        self.output_mode = output_mode
        self.bootstrap_threshold = min(1.0, max(0.0, float(bootstrap_threshold)))
        self.bootstrap_erode = max(0, int(bootstrap_erode))
        self.bootstrap_dilate = max(0, int(bootstrap_dilate))
        self.bootstrap_soft = bool(bootstrap_soft)
        self.segment_frames = max(0, int(segment_frames))
        self.bootstrap_refine_iters = max(1, int(config.MATANYONE2_BOOTSTRAP_REFINE_ITERS))
        self.alpha_stride = max(1, int(config.ALPHA_STRIDE))
        self._frame_index = 0
        self._source_frame_index = -1
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"MatAnyone2 manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        self.in_h = int(manifest.get("height") or 512)
        self.in_w = int(manifest.get("width") or 512)
        self.batch_size = int(manifest.get("batch_size") or 1)
        if int(manifest.get("objects") or 1) != 1:
            raise RuntimeError("MatAnyone2 offline engine currently supports one object only")
        provider_report: dict[str, list[str]] = {}
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def sess(name: str):
            path = model_dir / name
            if not path.exists():
                raise RuntimeError(f"MatAnyone2 ONNX file not found: {path}")
            session = ort.InferenceSession(str(path), sess_options=sess_opts, providers=session_providers(name, model_dir))
            provider_report[name] = list(session.get_providers())
            return session

        self.image_key = sess("matanyone2_image_key.onnx")
        self.mask_memory = sess("matanyone2_mask_memory.onnx")
        self.first_refine = sess("matanyone2_first_frame_refine.onnx")
        self.propagate = sess("matanyone2_propagate.onnx")
        self.propagate_update = None
        if use_fused_update and (model_dir / "matanyone2_propagate_update.onnx").exists():
            self.propagate_update = sess("matanyone2_propagate_update.onnx")
        self.step_update = None
        if use_step_update and (model_dir / "matanyone2_step_update.onnx").exists():
            self.step_update = sess("matanyone2_step_update.onnx")
        self._first_refine_inputs = {i.name for i in self.first_refine.get_inputs()}
        self._propagate_inputs = {i.name for i in self.propagate.get_inputs()}
        self._propagate_update_inputs = {i.name for i in self.propagate_update.get_inputs()} if self.propagate_update else set()
        self._step_update_inputs = {i.name for i in self.step_update.get_inputs()} if self.step_update else set()
        self.tensor_dtype = self.np.float16 if self.image_key.get_inputs()[0].type == "tensor(float16)" else self.np.float32
        self._step_output_names = [o.name for o in self.step_update.get_outputs()] if self.step_update else []
        self._step_output_dtypes = {
            o.name: _dtype_from_onnx_type(o.type)
            for o in self.step_update.get_outputs()
        } if self.step_update else {}
        image_batch = self.image_key.get_inputs()[0].shape[0]
        self.batch2_enabled = self.batch_size >= 2 or image_batch == 2
        sensory_meta = next(i for i in self.mask_memory.get_inputs() if i.name == "sensory")
        sensory_shape = [int(v) for v in sensory_meta.shape]
        self.sensory_shape = tuple(sensory_shape)
        self.sensory_single_shape = tuple([1, *sensory_shape[1:]])
        self.matter = Matter(config.ROOT / "models" / "rvm_mobilenetv3_fp32.onnx", load_model=False)
        self.matter.reset_state()
        self.packer = alpha_packer_factory(self.matter) if alpha_packer_factory is not None else None
        if self.output_mode == "alpha" and self.packer is None:
            raise RuntimeError("MatAnyone2 alpha output requires an alpha_packer_factory")
        self.eyes = [self._EyeState(), self._EyeState()]
        self._eye_smoothers = (
            [_AlphaSmoother(config.MATANYONE2_ALPHA_SMOOTH_WEIGHT), _AlphaSmoother(config.MATANYONE2_ALPHA_SMOOTH_WEIGHT)]
            if config.MATANYONE2_ALPHA_SMOOTH
            else []
        )
        self._mask_cache: list[np.ndarray] | None = None
        self.segment_masks: dict[int, list[np.ndarray]] = {}
        self._active_segment_start = -1
        self._cached_alpha_sbs = None
        self._iobinding_enabled = False
        self._iobinding_failed = False
        self._step_io_outputs: dict[int, dict[str, object]] = {0: {}, 1: {}}
        self._step_io_slots = [0, 0]
        self._eye_smoother_gpu_prev = [None, None] if self._eye_smoothers else []
        self._guided_upsample_enabled = bool(config.MATANYONE2_EDGE_AWARE_UPSAMPLE)
        self._guided_upsample_failed = False
        self._roi_enabled = bool(config.MATANYONE2_ROI_CROP)
        self._roi_failed = False
        self._segment_rois: dict[int, list[RoiMeta | None]] = {}
        self._last_mask_gate_failed = False
        self._last_sensory_decay_frame = -1
        if config.MATANYONE2_IOBINDING and self.step_update is not None:
            active_providers = set(self.step_update.get_providers())
            self._iobinding_enabled = bool(active_providers & {"CUDAExecutionProvider", "TensorrtExecutionProvider"})
        self.profile = defaultdict(list)
        print(
            f"{log_prefix} MatAnyone2 ONNX loaded dir={model_dir} input={self.in_w}x{self.in_h} "
            f"sbs=per-eye bootstrap={'mask' if mask else 'sam3'} sam3_prompt={sam3_prompt!r} "
            f"bootstrap_erode={self.bootstrap_erode} bootstrap_dilate={self.bootstrap_dilate} "
            f"bootstrap_soft={self.bootstrap_soft} segment_frames={self.segment_frames} "
            f"bootstrap_refine_iters={self.bootstrap_refine_iters} alpha_stride={self.alpha_stride} "
            f"batch2={self.batch2_enabled} dtype={self.tensor_dtype.__name__} "
            f"alpha_smooth={int(bool(self._eye_smoothers))} output={self.output_mode} "
            f"alpha_refine={'guided' if self._guided_upsample_enabled else 'off'} "
            f"drag_gate={config.MATANYONE2_LAST_MASK_UNCERT_GATE:.2f} "
            f"sensory_decay={config.MATANYONE2_SENSORY_DECAY_INTERVAL}/{config.MATANYONE2_SENSORY_DECAY_FACTOR:.2f} "
            f"last_pred_bin={int(config.MATANYONE2_LAST_PRED_BINARIZE)} "
            f"roi={int(self._roi_enabled)} "
            f"iobinding={int(self._iobinding_enabled)} "
            f"fused_update={self.propagate_update is not None} step_update={self.step_update is not None} "
            f"providers={provider_report}"
        )

    @staticmethod
    def _as_numpy(x):
        try:
            import cupy as cp

            if isinstance(x, cp.ndarray):
                return cp.asnumpy(x)
        except Exception:
            pass
        return x

    def _tensor(self, x):
        return x.astype(self.tensor_dtype, copy=False)

    def _to_numpy(self, x):
        if hasattr(x, "data_ptr") and hasattr(x, "numpy"):
            return x.numpy()
        return self._as_numpy(x)

    @staticmethod
    def _is_cupy_array(value) -> bool:
        return hasattr(value, "data") and hasattr(value.data, "ptr")

    @staticmethod
    def _array_shape(value) -> tuple[int, ...]:
        if hasattr(value, "shape"):
            shape = value.shape() if callable(value.shape) else value.shape
            return tuple(int(v) for v in shape)
        return tuple()

    def _as_numpy4(self, value) -> np.ndarray:
        arr = self._to_numpy(value)
        arr = self.np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]
        return arr

    def _as_cupy4(self, value):
        import cupy as cp

        if hasattr(value, "data_ptr"):
            dtype = self._step_output_dtypes.get("uncert_prob", self.tensor_dtype)
            arr = self._ortvalue_to_cupy(value, dtype)
        elif self._is_cupy_array(value):
            arr = value
        else:
            arr = cp.asarray(value)
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]
        return arr

    def _resize_numpy_batch(self, arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if int(arr.shape[-2]) == int(out_h) and int(arr.shape[-1]) == int(out_w):
            return arr
        b, c = int(arr.shape[0]), int(arr.shape[1])
        out = self.np.empty((b, c, int(out_h), int(out_w)), dtype=self.np.float32)
        for bi in range(b):
            for ci in range(c):
                out[bi, ci] = self.cv2.resize(
                    arr[bi, ci].astype(self.np.float32, copy=False),
                    (int(out_w), int(out_h)),
                    interpolation=self.cv2.INTER_LINEAR,
                )
        return out

    def _resize_cupy_batch(self, arr, out_h: int, out_w: int):
        if int(arr.shape[-2]) == int(out_h) and int(arr.shape[-1]) == int(out_w):
            return arr
        import cupy as cp
        from pipeline.alpha_guided_filter import _resize_float

        planes = []
        for bi in range(int(arr.shape[0])):
            row = []
            for ci in range(int(arr.shape[1])):
                row.append(_resize_float(arr[bi, ci], int(out_h), int(out_w)))
            planes.append(cp.stack(row, axis=0))
        return cp.stack(planes, axis=0)

    def _align_uncert_to_mask(self, uncert, mask):
        if uncert is None:
            return None
        target_shape = self._array_shape(mask)
        if len(target_shape) != 4:
            return None
        target_b, target_c, target_h, target_w = target_shape
        use_cupy = self._is_cupy_array(mask) or hasattr(uncert, "data_ptr")
        if use_cupy:
            import cupy as cp

            arr = self._as_cupy4(uncert)
            if arr.ndim != 4:
                return None
            if int(arr.shape[0]) not in (1, target_b) or int(arr.shape[1]) not in (1, target_c):
                return None
            arr = self._resize_cupy_batch(arr, target_h, target_w)
            return cp.clip(arr.astype(cp.float32, copy=False), 0.0, 1.0)
        arr = self._as_numpy4(uncert)
        if arr.ndim != 4:
            return None
        if int(arr.shape[0]) not in (1, target_b) or int(arr.shape[1]) not in (1, target_c):
            return None
        arr = self._resize_numpy_batch(arr, target_h, target_w)
        return self.np.clip(arr.astype(self.np.float32, copy=False), 0.0, 1.0)

    def _last_mask_input(self, state):
        last_mask = state.last_mask
        strength = float(config.MATANYONE2_LAST_MASK_UNCERT_GATE)
        if strength <= 0.0 or getattr(state, "last_uncert", None) is None or last_mask is None:
            return last_mask
        try:
            uncert = self._align_uncert_to_mask(state.last_uncert, last_mask)
            if uncert is None:
                return last_mask
            if self._is_cupy_array(last_mask):
                import cupy as cp

                gate = cp.maximum(0.0, 1.0 - strength * uncert)
                return (last_mask * gate).astype(last_mask.dtype, copy=False)
            gate = self.np.maximum(0.0, 1.0 - strength * uncert)
            return (last_mask * gate).astype(last_mask.dtype, copy=False)
        except Exception as exc:
            if not self._last_mask_gate_failed:
                self._last_mask_gate_failed = True
                print(
                    f"[offline] MatAnyone2 last_mask uncertainty gate disabled after failure "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
            return last_mask

    def _last_pred_mask_input(self, last_mask):
        if not config.MATANYONE2_LAST_PRED_BINARIZE:
            return last_mask
        threshold = float(config.MATANYONE2_LAST_PRED_BIN_THRESHOLD)
        if self._is_cupy_array(last_mask):
            import cupy as cp

            return cp.where(last_mask > threshold, 1.0, 0.0).astype(last_mask.dtype, copy=False)
        return (last_mask > threshold).astype(last_mask.dtype, copy=False)

    def _image_key(self, image: np.ndarray) -> dict[str, np.ndarray]:
        names = ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"]
        t0 = time.perf_counter()
        outs = self.image_key.run(names, {"image": image})
        self.profile["image_key"].append((time.perf_counter() - t0) * 1000)
        return dict(zip(names, outs))

    def _mask_memory(self, image, mask, sensory, pix_feat):
        t0 = time.perf_counter()
        msk_value, new_sensory, obj_memory = self.mask_memory.run(
            ["msk_value", "new_sensory", "obj_memory"],
            {
                "image": image,
                "mask": mask,
                "sensory": sensory,
                "pix_feat": pix_feat,
            },
        )
        self.profile["mask_memory"].append((time.perf_counter() - t0) * 1000)
        return msk_value, new_sensory, obj_memory

    def _first_frame_refine(self, feats, last_msk_value, obj_memory, sensory, last_mask):
        feed = {
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "last_msk_value": last_msk_value,
            "obj_memory": obj_memory,
            "sensory": sensory,
            "last_mask": last_mask,
        }
        t0 = time.perf_counter()
        prob, new_sensory, _logits = self.first_refine.run(
            ["prob", "new_sensory", "logits"],
            {k: v for k, v in feed.items() if k in self._first_refine_inputs},
        )
        self.profile["first_refine"].append((time.perf_counter() - t0) * 1000)
        return prob, new_sensory

    def _propagate(self, feats, state):
        assert state.memory_key is not None
        assert state.memory_shrinkage is not None
        assert state.memory_msk_value is not None
        assert state.obj_memory is not None
        assert state.sensory is not None
        assert state.last_mask is not None
        assert state.last_pix_feat is not None
        assert state.last_msk_value is not None
        last_mask = self._last_mask_input(state)
        last_pred_mask = self._last_pred_mask_input(state.last_mask)
        feed = {
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "key": feats["key"],
            "selection": feats["selection"],
            "memory_key": self._to_numpy(state.memory_key),
            "memory_shrinkage": self._to_numpy(state.memory_shrinkage),
            "msk_value": self._to_numpy(state.memory_msk_value),
            "obj_memory": self._to_numpy(state.obj_memory),
            "sensory": self._to_numpy(state.sensory),
            "last_mask": self._to_numpy(last_mask),
            "last_pix_feat": self._to_numpy(state.last_pix_feat),
            "last_pred_mask": self._to_numpy(last_pred_mask),
            "last_msk_value": self._to_numpy(state.last_msk_value),
        }
        t0 = time.perf_counter()
        prob, new_sensory, _logits, uncert_prob = self.propagate.run(
            ["prob", "new_sensory", "logits", "uncert_prob"],
            {k: v for k, v in feed.items() if k in self._propagate_inputs},
        )
        self.profile["propagate"].append((time.perf_counter() - t0) * 1000)
        return prob, new_sensory, uncert_prob

    def _propagate_update(self, image, feats, state):
        if self.propagate_update is None:
            prob, sensory, uncert_prob = self._propagate(feats, state)
            alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.np.float32, copy=False)
            msk_value, sensory, obj_memory = self._mask_memory(image, alpha, sensory, feats["pix_feat"])
            return prob, sensory, msk_value, obj_memory, uncert_prob
        last_mask = self._last_mask_input(state)
        last_pred_mask = self._last_pred_mask_input(state.last_mask)
        feed = {
            "image": image,
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "key": feats["key"],
            "selection": feats["selection"],
            "memory_key": self._to_numpy(state.memory_key),
            "memory_shrinkage": self._to_numpy(state.memory_shrinkage),
            "msk_value": self._to_numpy(state.memory_msk_value),
            "obj_memory": self._to_numpy(state.obj_memory),
            "sensory": self._to_numpy(state.sensory),
            "last_mask": self._to_numpy(last_mask),
            "last_pix_feat": self._to_numpy(state.last_pix_feat),
            "last_pred_mask": self._to_numpy(last_pred_mask),
            "last_msk_value": self._to_numpy(state.last_msk_value),
        }
        t0 = time.perf_counter()
        prob, sensory, msk_value, obj_memory, _logits, uncert_prob = self.propagate_update.run(
            ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "logits", "uncert_prob"],
            {k: v for k, v in feed.items() if k in self._propagate_update_inputs},
        )
        self.profile["propagate_update"].append((time.perf_counter() - t0) * 1000)
        return prob, sensory, msk_value, obj_memory, uncert_prob

    def _step_update(self, image, state, eye_idx: int | None = None):
        if eye_idx is not None and self._should_use_iobinding(image, state):
            try:
                return self._step_update_iobinding(image, state, eye_idx)
            except Exception as exc:
                self._iobinding_failed = True
                print(f"[offline] MatAnyone2 IOBinding failed; falling back to NumPy step_update ({type(exc).__name__}: {exc})", flush=True)
        if self.step_update is None:
            feats = self._image_key(image)
            prob, sensory, msk_value, obj_memory, uncert_prob = self._propagate_update(image, feats, state)
            return prob, sensory, msk_value, obj_memory, feats["pix_feat"], uncert_prob
        last_mask = self._last_mask_input(state)
        last_pred_mask = self._last_pred_mask_input(state.last_mask)
        feed = {
            "image": self._to_numpy(image),
            "memory_key": self._to_numpy(state.memory_key),
            "memory_shrinkage": self._to_numpy(state.memory_shrinkage),
            "msk_value": self._to_numpy(state.memory_msk_value),
            "obj_memory": self._to_numpy(state.obj_memory),
            "sensory": self._to_numpy(state.sensory),
            "last_mask": self._to_numpy(last_mask),
            "last_pix_feat": self._to_numpy(state.last_pix_feat),
            "last_pred_mask": self._to_numpy(last_pred_mask),
            "last_msk_value": self._to_numpy(state.last_msk_value),
        }
        t0 = time.perf_counter()
        prob, sensory, msk_value, obj_memory, pix_feat, _logits, uncert_prob = self.step_update.run(
            ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "pix_feat", "logits", "uncert_prob"],
            {k: v for k, v in feed.items() if k in self._step_update_inputs},
        )
        self.profile["step_update"].append((time.perf_counter() - t0) * 1000)
        return prob, sensory, msk_value, obj_memory, pix_feat, uncert_prob

    def _should_use_iobinding(self, image, state) -> bool:
        if not self._iobinding_enabled or self._iobinding_failed or self.step_update is None:
            return False
        if not hasattr(image, "data") or not hasattr(image.data, "ptr"):
            return False
        if int(image.shape[0]) != 1:
            return False
        return all(getattr(state, attr, None) is not None for attr in (
            "memory_key",
            "memory_shrinkage",
            "memory_msk_value",
            "obj_memory",
            "sensory",
            "last_mask",
            "last_pix_feat",
            "last_msk_value",
        ))

    def _ortvalue_to_cupy(self, value, dtype=None):
        import cupy as cp

        shape = tuple(int(v) for v in value.shape())
        dtype = cp.dtype(dtype or self.tensor_dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        mem = cp.cuda.UnownedMemory(int(value.data_ptr()), nbytes, value)
        ptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray(shape, dtype=dtype, memptr=ptr)

    def _cuda_ortvalue_from_numpy(self, value):
        return self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), "cuda", 0)

    def _state_ortvalue(self, value):
        if hasattr(value, "data_ptr"):
            return value
        return self._cuda_ortvalue_from_numpy(value)

    def _bind_cuda_or_ortvalue_input(self, binding, name: str, value) -> None:
        if hasattr(value, "data") and hasattr(value.data, "ptr"):
            binding.bind_input(name, "cuda", 0, self.tensor_dtype, tuple(int(v) for v in value.shape), int(value.data.ptr))
            return
        binding.bind_ortvalue_input(name, self._state_ortvalue(value))

    def _step_output_ortvalue(self, name: str, shape: tuple[int, ...], slot: int | None = None, eye_idx: int = 0):
        dtype = self._step_output_dtypes.get(name, self.tensor_dtype)
        eye_idx = int(eye_idx)
        if eye_idx < 0 or eye_idx >= len(self._step_io_slots):
            eye_idx = 0
        slot = self._step_io_slots[eye_idx] if slot is None else int(slot) & 1
        bucket = self._step_io_outputs.get(eye_idx)
        if not isinstance(bucket, dict):
            bucket = {}
            self._step_io_outputs[eye_idx] = bucket
        slots = bucket.get(name)
        if not isinstance(slots, list) or len(slots) != 2:
            slots = [None, None]
            bucket[name] = slots
        value = slots[slot]
        if value is None or tuple(int(v) for v in value.shape()) != tuple(shape):
            value = self.ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, "cuda", 0)
            slots[slot] = value
        return value

    def _step_update_iobinding(self, image, state, eye_idx: int):
        eye_idx = int(eye_idx)
        if eye_idx < 0 or eye_idx >= len(self._step_io_slots):
            eye_idx = 0
        binding = self.step_update.io_binding()
        binding.bind_input("image", "cuda", 0, self.tensor_dtype, tuple(int(v) for v in image.shape), int(image.data.ptr))
        state.memory_key = self._state_ortvalue(state.memory_key)
        state.memory_shrinkage = self._state_ortvalue(state.memory_shrinkage)
        state.memory_msk_value = self._state_ortvalue(state.memory_msk_value)
        state.obj_memory = self._state_ortvalue(state.obj_memory)
        state.sensory = self._state_ortvalue(state.sensory)
        state.last_pix_feat = self._state_ortvalue(state.last_pix_feat)
        state.last_msk_value = self._state_ortvalue(state.last_msk_value)
        last_mask = self._last_mask_input(state)
        last_pred_mask = self._last_pred_mask_input(state.last_mask)
        bindings = {
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": last_pred_mask,
            "last_msk_value": state.last_msk_value,
        }
        for name, value in bindings.items():
            if name in self._step_update_inputs:
                self._bind_cuda_or_ortvalue_input(binding, name, value)

        output_slot = self._step_io_slots[eye_idx]
        for meta in self.step_update.get_outputs():
            shape = tuple(int(v) for v in meta.shape)
            binding.bind_ortvalue_output(meta.name, self._step_output_ortvalue(meta.name, shape, output_slot, eye_idx))

        t0 = time.perf_counter()
        self.step_update.run_with_iobinding(binding)
        self.profile["step_update"].append((time.perf_counter() - t0) * 1000)
        outputs = dict(zip(self._step_output_names, binding.get_outputs()))
        self._step_io_slots[eye_idx] = 1 - output_slot
        self.profile["iobinding_copy"].append(0.0)
        return (
            outputs["prob"],
            outputs["new_sensory"],
            outputs["new_msk_value"],
            outputs["new_obj_memory"],
            outputs["pix_feat"],
            outputs.get("uncert_prob"),
        )

    def _uncert_output_value(self, value):
        if value is None or hasattr(value, "data_ptr"):
            return value
        return value.astype(self.np.float32, copy=False)

    def _decay_sensory(self, value, factor: float):
        factor = float(factor)
        if value is None or factor >= 1.0:
            return value
        if hasattr(value, "data_ptr"):
            arr = self._ortvalue_to_cupy(value, self.tensor_dtype)
            arr *= factor
            return value
        if self._is_cupy_array(value):
            value *= factor
            return value.astype(value.dtype, copy=False)
        return (value * factor).astype(value.dtype, copy=False)

    def _maybe_decay_sensory(self):
        interval = int(config.MATANYONE2_SENSORY_DECAY_INTERVAL)
        if interval <= 0 or self._frame_index <= 0 or self._frame_index % interval != 0:
            return
        if self._last_sensory_decay_frame == self._frame_index:
            return
        factor = float(config.MATANYONE2_SENSORY_DECAY_FACTOR)
        if factor >= 1.0:
            return
        t0 = time.perf_counter()
        changed = False
        for eye in self.eyes:
            if eye.sensory is not None:
                eye.sensory = self._decay_sensory(eye.sensory, factor)
                changed = True
        self._last_sensory_decay_frame = self._frame_index
        if changed:
            self.profile["sensory_decay"].append((time.perf_counter() - t0) * 1000)

    def _load_masks(self) -> list[np.ndarray] | None:
        if self.mask_path is None:
            return None
        if self._mask_cache is not None:
            return self._mask_cache
        mask = self.cv2.imread(str(self.mask_path), self.cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read MatAnyone2 mask: {self.mask_path}")
        mh, mw = mask.shape[:2]
        if mw >= 2 * mh:
            half = mw // 2
            masks = [mask[:, :half], mask[:, half:half * 2]]
        else:
            masks = [mask, mask]
        self._mask_cache = []
        for eye_mask in masks:
            resized = self.cv2.resize(eye_mask, (self.in_w, self.in_h), interpolation=self.cv2.INTER_NEAREST)
            self._mask_cache.append((resized.astype(self.np.float32) / 255.0)[None, None, :, :])
        return self._mask_cache

    def _reset_segment(self):
        for eye in self.eyes:
            eye.reset()
        for smoother in self._eye_smoothers:
            smoother.reset()
        for idx in range(len(self._eye_smoother_gpu_prev)):
            self._eye_smoother_gpu_prev[idx] = None
        self._active_segment_start = -1
        self._cached_alpha_sbs = None
        self._step_io_outputs = {0: {}, 1: {}}
        self._step_io_slots = [0, 0]
        self._segment_rois = {}

    def set_segment_masks(self, segment_start: int, masks: list[np.ndarray]):
        self.segment_masks[int(segment_start)] = masks
        self._segment_rois.pop(int(segment_start), None)

    def set_segment_plan(self, starts: list[int]):
        self.segment_starts = sorted(set(int(x) for x in starts))

    def set_source_frame_index(self, src_idx: int):
        self._source_frame_index = int(src_idx)

    def _current_segment_start(self) -> int:
        starts = getattr(self, "segment_starts", None)
        if starts:
            current = starts[0]
            for start in starts:
                if start > self._frame_index:
                    break
                current = start
            return current
        if self.segment_frames <= 0:
            return 0
        return (self._frame_index // self.segment_frames) * self.segment_frames

    def is_active_frame(self) -> bool:
        if self.mask_path is not None:
            return True
        return self._current_segment_start() in self.segment_masks

    def composite_green_nv12(self, frame):
        h, w = self._upload(frame)
        alpha = self.np.zeros((self.in_h, self.in_w * 2), dtype=self.np.float32)
        self._frame_index += 1
        return self._emit_output(alpha, h, w), None

    def _roi_bootstrap_alpha(self, alpha: np.ndarray, roi: RoiMeta) -> np.ndarray:
        mask = self.np.asarray(alpha, dtype=self.np.float32)
        out = self.np.zeros((roi.model_h_total, roi.model_w_total), dtype=self.np.float32)
        if mask.ndim != 2:
            return out
        mh, mw = (int(v) for v in mask.shape[:2])
        sx = float(mw) / max(1.0, float(roi.eye_w))
        sy = float(mh) / max(1.0, float(roi.eye_h))
        x0 = max(0, min(mw, int(self.np.floor(float(roi.x0) * sx))))
        y0 = max(0, min(mh, int(self.np.floor(float(roi.y0) * sy))))
        x1 = max(0, min(mw, int(self.np.ceil(float(roi.x1) * sx))))
        y1 = max(0, min(mh, int(self.np.ceil(float(roi.y1) * sy))))
        if x1 <= x0 or y1 <= y0 or roi.model_w <= 0 or roi.model_h <= 0:
            return out
        crop = mask[y0:y1, x0:x1]
        resized = self.cv2.resize(crop, (roi.model_w, roi.model_h), interpolation=self.cv2.INTER_LINEAR)
        yy0 = max(0, min(roi.model_h_total, roi.model_y0))
        xx0 = max(0, min(roi.model_w_total, roi.model_x0))
        yy1 = max(yy0, min(roi.model_h_total, yy0 + roi.model_h))
        xx1 = max(xx0, min(roi.model_w_total, xx0 + roi.model_w))
        out[yy0:yy1, xx0:xx1] = resized[: yy1 - yy0, : xx1 - xx0]
        return out

    def _bootstrap_mask(self, h: int, w: int, eye_idx: int, roi: RoiMeta | None = None) -> np.ndarray:
        masks = self._load_masks()
        if masks is not None:
            return masks[eye_idx]
        segment_start = self._current_segment_start()
        sam3_masks = self.segment_masks.get(segment_start)
        if sam3_masks is None:
            raise RuntimeError(
                f"Missing precomputed MatAnyone2 bootstrap mask for segment_start={segment_start}. "
                "Run the configured prepass or pass --mask."
            )
        alpha = sam3_masks[eye_idx][0, 0]
        if roi is not None:
            alpha = self._roi_bootstrap_alpha(alpha, roi)
        hard = (alpha >= self.bootstrap_threshold).astype(self.np.uint8)
        kernel = self.np.ones((3, 3), self.np.uint8)
        if self.bootstrap_erode > 0:
            hard = self.cv2.erode(hard, kernel, iterations=self.bootstrap_erode)
        if self.bootstrap_dilate > 0:
            hard = self.cv2.dilate(hard, kernel, iterations=self.bootstrap_dilate)
        if self.bootstrap_soft:
            alpha = self.np.minimum(alpha, hard.astype(self.np.float32))
        else:
            alpha = hard.astype(self.np.float32)
        return alpha[None, None, :, :].astype(self.tensor_dtype, copy=False)

    def _preprocess_eye(self, x0: int, eye_w: int, eye_idx: int | None = None):
        t0 = time.perf_counter()
        batch = 2 if eye_idx is not None else 1
        batch_idx = int(eye_idx) if eye_idx is not None else 0
        use_iobinding = (
            self._iobinding_enabled
            and not self._iobinding_failed
            and eye_idx is not None
            and self.eyes[eye_idx].initialized
        )
        image = self.matter._gpu_preprocess_nv12_one(
            x0,
            eye_w,
            self.in_w,
            self.in_h,
            batch=batch,
            batch_idx=batch_idx,
            copy_to_host=not use_iobinding,
        )
        self.profile["preprocess_eye"].append((time.perf_counter() - t0) * 1000)
        return image.astype(self.tensor_dtype, copy=False)

    def _segment_roi(self, segment_start: int, eye_idx: int, eye_w: int, h: int) -> RoiMeta | None:
        if not self._roi_enabled or self._roi_failed or self.mask_path is not None:
            return None
        rois = self._segment_rois.get(segment_start)
        if rois is None:
            masks = self.segment_masks.get(segment_start)
            rois = [None, None]
            if masks is not None and len(masks) >= 2:
                for idx in (0, 1):
                    try:
                        rois[idx] = roi_from_mask(
                            masks[idx][0, 0],
                            eye_w=eye_w,
                            eye_h=h,
                            model_w=self.in_w,
                            model_h=self.in_h,
                            expand=config.MATANYONE2_ROI_EXPAND,
                            max_eye_fraction=config.MATANYONE2_ROI_MAX_EYE_FRACTION,
                        )
                    except Exception:
                        rois[idx] = None
            self._segment_rois[segment_start] = rois
        return rois[eye_idx] if 0 <= eye_idx < len(rois) else None

    def _segment_roi_pair(self, segment_start: int, eye_w: int, h: int) -> tuple[RoiMeta | None, RoiMeta | None]:
        left_roi = self._segment_roi(segment_start, 0, eye_w, h)
        right_roi = self._segment_roi(segment_start, 1, eye_w, h)
        if left_roi is None or right_roi is None:
            return None, None
        return left_roi, right_roi

    def _preprocess_eye_roi(self, roi: RoiMeta, eye_idx: int | None = None):
        t0 = time.perf_counter()
        batch = 2 if eye_idx is not None else 1
        batch_idx = int(eye_idx) if eye_idx is not None else 0
        use_iobinding = (
            self._iobinding_enabled
            and not self._iobinding_failed
            and eye_idx is not None
            and self.eyes[eye_idx].initialized
        )
        image = self.matter._gpu_preprocess_nv12_roi_one(
            roi,
            self.in_w,
            self.in_h,
            batch=batch,
            batch_idx=batch_idx,
            source_x0=(int(eye_idx) * roi.eye_w) if eye_idx is not None else 0,
            copy_to_host=not use_iobinding,
        )
        self.profile["preprocess_eye"].append((time.perf_counter() - t0) * 1000)
        return image.astype(self.tensor_dtype, copy=False)

    def _preprocess_eyes_batch2(self, eye_w: int) -> np.ndarray:
        t0 = time.perf_counter()
        left = self.matter._gpu_preprocess_nv12_one(0, eye_w, self.in_w, self.in_h, batch=2, batch_idx=0, copy_to_host=True)
        right = self.matter._gpu_preprocess_nv12_one(eye_w, eye_w, self.in_w, self.in_h, batch=2, batch_idx=1, copy_to_host=True)
        self.profile["preprocess_eye"].append((time.perf_counter() - t0) * 1000)
        return self.np.ascontiguousarray(self.np.concatenate([left, right], axis=0)).astype(self.tensor_dtype, copy=False)

    def _smooth_eye_alpha(self, alpha_2d: np.ndarray, eye_idx: int) -> np.ndarray:
        if not self._eye_smoothers:
            return alpha_2d
        if hasattr(alpha_2d, "data") and hasattr(alpha_2d.data, "ptr"):
            t0 = time.perf_counter()
            prev = self._eye_smoother_gpu_prev[eye_idx]
            if prev is None or prev.shape != alpha_2d.shape:
                self._eye_smoother_gpu_prev[eye_idx] = alpha_2d.copy()
                self.profile["alpha_smooth"].append((time.perf_counter() - t0) * 1000)
                return alpha_2d
            out = prev * config.MATANYONE2_ALPHA_SMOOTH_WEIGHT + alpha_2d * (1.0 - config.MATANYONE2_ALPHA_SMOOTH_WEIGHT)
            self._eye_smoother_gpu_prev[eye_idx] = out
            self.profile["alpha_smooth"].append((time.perf_counter() - t0) * 1000)
            return out.astype(alpha_2d.dtype, copy=False)
        t0 = time.perf_counter()
        batch = alpha_2d[None, ...]
        smoothed = self._eye_smoothers[eye_idx].step(batch)[0]
        self.profile["alpha_smooth"].append((time.perf_counter() - t0) * 1000)
        return smoothed.astype(alpha_2d.dtype, copy=False)

    def _upload(self, frame) -> tuple[int, int]:
        from pipeline.pynv_io import GpuP016Frame

        h, w = int(frame.height), int(frame.width)
        t0 = time.perf_counter()
        if isinstance(frame, GpuP016Frame):
            self.matter.upload_p016_planes_as_nv12_gpu(
                frame.y.as_cupy(),
                frame.uv.as_cupy(),
                h,
                w,
                shift_bits=int(config.PASSTHROUGH_PYNV_10BIT_SHIFT),
            )
        else:
            self.matter.upload_nv12_planes_gpu(frame.y.as_cupy(), frame.uv.as_cupy(), h, w)
        self.profile["upload_nv12"].append((time.perf_counter() - t0) * 1000)
        return h, w

    def _run_eye(self, image, h: int, w: int, eye_idx: int, roi: RoiMeta | None = None) -> np.ndarray:
        state = self.eyes[eye_idx]
        if not state.initialized:
            feats = self._image_key(image)
            sensory = self.np.zeros(self.sensory_single_shape, dtype=self.tensor_dtype)
            mask = self._bootstrap_mask(h, w, eye_idx, roi=roi)
            msk_value, sensory, obj_memory = self._mask_memory(image, mask, sensory, feats["pix_feat"])
            alpha = mask
            for _ in range(self.bootstrap_refine_iters):
                prob, sensory = self._first_frame_refine(feats, msk_value, obj_memory[:, :, None, :, :], sensory, alpha)
                alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
                msk_value, sensory, obj_memory = self._mask_memory(image, alpha, sensory, feats["pix_feat"])
            state.memory_key = feats["key"][:, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.memory_shrinkage = feats["shrinkage"][:, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.memory_msk_value = msk_value[:, :, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.obj_memory = obj_memory[:, :, None, :, :].astype(self.tensor_dtype, copy=False)
            state.sensory = sensory.astype(self.tensor_dtype, copy=False)
            state.last_mask = alpha
            state.last_pix_feat = feats["pix_feat"].astype(self.tensor_dtype, copy=False)
            state.last_msk_value = msk_value.astype(self.tensor_dtype, copy=False)
            state.last_uncert = None
            state.initialized = True
        else:
            prob, sensory, msk_value, _obj_memory, pix_feat, uncert_prob = self._step_update(image, state, eye_idx=eye_idx)
            alpha = self._prob_to_alpha(prob)
            state.sensory = self._state_output_value(sensory)
            state.last_mask = alpha
            state.last_pix_feat = self._state_output_value(pix_feat)
            state.last_msk_value = self._state_output_value(msk_value)
            state.last_uncert = self._uncert_output_value(uncert_prob)
        alpha_2d = self._smooth_eye_alpha(alpha[0, 0], eye_idx)
        if roi is not None:
            t0 = time.perf_counter()
            alpha_2d = self.matter._gpu_unwarp_roi_alpha_to_eye(
                alpha_2d,
                roi,
                roi.eye_w,
                roi.eye_h,
                feather=config.MATANYONE2_ROI_FEATHER,
            )
            self.profile["roi_unwarp"].append((time.perf_counter() - t0) * 1000)
        return alpha_2d

    def _run_eyes_batch2(self, images, h: int, w: int) -> np.ndarray:
        states = self.eyes
        if images.shape[0] != 2:
            left = self._run_eye(images[0:1], h, w, 0)
            right = self._run_eye(images[1:2], h, w, 1)
            return self._concat_alpha_sbs(left, right)

        if any(not state.initialized for state in states):
            feats = self._image_key(images)
            sensory = self.np.zeros(self.sensory_shape, dtype=self.tensor_dtype)
            masks = self.np.concatenate([self._bootstrap_mask(h, w, 0), self._bootstrap_mask(h, w, 1)], axis=0)
            msk_value, sensory, obj_memory = self._mask_memory(images, masks, sensory, feats["pix_feat"])
            alpha = masks
            for _ in range(self.bootstrap_refine_iters):
                prob, sensory = self._first_frame_refine(
                    feats,
                    msk_value,
                    obj_memory[:, :, None, :, :],
                    sensory,
                    alpha,
                )
                alpha = self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)
                msk_value, sensory, obj_memory = self._mask_memory(images, alpha, sensory, feats["pix_feat"])
            for idx, state in enumerate(states):
                state.memory_key = feats["key"][idx:idx + 1, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.memory_shrinkage = feats["shrinkage"][idx:idx + 1, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.memory_msk_value = msk_value[idx:idx + 1, :, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.obj_memory = obj_memory[idx:idx + 1, :, None, :, :].astype(self.tensor_dtype, copy=False)
                state.sensory = sensory[idx:idx + 1].astype(self.tensor_dtype, copy=False)
                state.last_mask = alpha[idx:idx + 1]
                state.last_pix_feat = feats["pix_feat"][idx:idx + 1].astype(self.tensor_dtype, copy=False)
                state.last_msk_value = msk_value[idx:idx + 1].astype(self.tensor_dtype, copy=False)
                state.last_uncert = None
                state.initialized = True
            left = self._smooth_eye_alpha(alpha[0, 0], 0)
            right = self._smooth_eye_alpha(alpha[1, 0], 1)
            return self._concat_alpha_sbs(left, right)

        for other in states[1:]:
            assert other.memory_key is not None
        batched_state = self._EyeState()
        batched_state.memory_key = self.np.concatenate([s.memory_key for s in states], axis=0)
        batched_state.memory_shrinkage = self.np.concatenate([s.memory_shrinkage for s in states], axis=0)
        batched_state.memory_msk_value = self.np.concatenate([s.memory_msk_value for s in states], axis=0)
        batched_state.obj_memory = self.np.concatenate([s.obj_memory for s in states], axis=0)
        batched_state.sensory = self.np.concatenate([s.sensory for s in states], axis=0)
        batched_state.last_mask = self.np.concatenate([s.last_mask for s in states], axis=0)
        batched_state.last_pix_feat = self.np.concatenate([s.last_pix_feat for s in states], axis=0)
        batched_state.last_msk_value = self.np.concatenate([s.last_msk_value for s in states], axis=0)
        if all(s.last_uncert is not None for s in states):
            batched_state.last_uncert = self.np.concatenate([self._as_numpy4(s.last_uncert) for s in states], axis=0)
        batched_state.initialized = True
        prob, sensory, msk_value, _obj_memory, pix_feat, uncert_prob = self._step_update(images, batched_state)
        alpha = self._prob_to_alpha(prob)
        uncert_prob = self._uncert_output_value(uncert_prob)
        for idx, state in enumerate(states):
            state.sensory = sensory[idx:idx + 1].astype(self.tensor_dtype, copy=False)
            state.last_mask = alpha[idx:idx + 1]
            state.last_pix_feat = pix_feat[idx:idx + 1].astype(self.tensor_dtype, copy=False)
            state.last_msk_value = msk_value[idx:idx + 1].astype(self.tensor_dtype, copy=False)
            state.last_uncert = uncert_prob[idx:idx + 1] if uncert_prob is not None else None
        left = self._smooth_eye_alpha(alpha[0, 0], 0)
        right = self._smooth_eye_alpha(alpha[1, 0], 1)
        return self._concat_alpha_sbs(left, right)

    def _prob_to_alpha(self, prob):
        if hasattr(prob, "data_ptr"):
            import cupy as cp

            prob_cp = self._ortvalue_to_cupy(prob, self._step_output_dtypes.get("prob", self.tensor_dtype))
            return cp.clip(prob_cp[:, 1:2], 0.0, 1.0).astype(cp.dtype(self.tensor_dtype), copy=False)
        return self.np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.tensor_dtype, copy=False)

    def _state_output_value(self, value):
        if hasattr(value, "data_ptr"):
            return value
        return value.astype(self.tensor_dtype, copy=False)

    def _concat_alpha_sbs(self, left, right):
        if hasattr(left, "data") and hasattr(left.data, "ptr"):
            import cupy as cp

            return cp.ascontiguousarray(cp.concatenate([left, right], axis=1))
        return self.np.ascontiguousarray(self.np.concatenate([left, right], axis=1))

    def composite_nv12(self, frame):
        segment_start = self._current_segment_start()
        if self._active_segment_start != segment_start:
            print(
                f"[offline] MatAnyone2 segment reset at frame={self._frame_index} "
                f"src_idx={self._source_frame_index} segment_start={segment_start}"
            )
            self._reset_segment()
            self._active_segment_start = segment_start
        h, w = self._upload(frame)
        eye_w = w // 2
        if eye_w <= 0 or w < 2 * h:
            raise RuntimeError(f"MatAnyone2 ONNX offline engine expects SBS input, got {w}x{h}")
        should_update = (
            self._cached_alpha_sbs is None
            or self.alpha_stride <= 1
            or self._frame_index % self.alpha_stride == 0
        )
        if should_update and self.batch2_enabled:
            self._maybe_decay_sensory()
            alpha_sbs = self._run_eyes_batch2(self._preprocess_eyes_batch2(eye_w), h, w)
            self._cached_alpha_sbs = self._maybe_refine_alpha(alpha_sbs, h, w)
        elif should_update:
            self._maybe_decay_sensory()
            left_roi, right_roi = self._segment_roi_pair(segment_start, eye_w, h)
            roi_active = left_roi is not None and right_roi is not None
            left_image = self._preprocess_eye_roi(left_roi, 0) if left_roi is not None else self._preprocess_eye(0, eye_w, 0)
            right_image = self._preprocess_eye_roi(right_roi, 1) if right_roi is not None else self._preprocess_eye(eye_w, eye_w, 1)
            left = self._run_eye(left_image, h, w, 0, left_roi)
            right = self._run_eye(right_image, h, w, 1, right_roi)
            t0 = time.perf_counter()
            alpha_sbs = self._concat_alpha_sbs(left, right)
            self.profile["alpha_concat"].append((time.perf_counter() - t0) * 1000)
            self._cached_alpha_sbs = alpha_sbs if roi_active else self._maybe_refine_alpha(alpha_sbs, h, w)
        else:
            self.profile["alpha_reuse"].append(0.0)
        alpha_sbs = self._cached_alpha_sbs
        self._frame_index += 1
        return self._emit_output(alpha_sbs, h, w), None

    def _maybe_refine_alpha(self, alpha_sbs, h: int, w: int):
        if not self._guided_upsample_enabled or self._guided_upsample_failed:
            return alpha_sbs
        if tuple(int(v) for v in alpha_sbs.shape[:2]) == (int(h), int(w)):
            return alpha_sbs
        try:
            from pipeline.alpha_guided_filter import fast_guided_filter_upsample

            t0 = time.perf_counter()
            guide_y = self.matter._g_frame[: int(h), : int(w)]
            refined = fast_guided_filter_upsample(
                alpha_sbs,
                guide_y,
                radius=config.MATANYONE2_GUIDED_RADIUS,
                eps=config.MATANYONE2_GUIDED_EPS,
                fullres_scale=config.MATANYONE2_GUIDED_FULLRES_SCALE,
                support_floor=config.MATANYONE2_GUIDED_SUPPORT_FLOOR,
                max_delta=config.MATANYONE2_GUIDED_MAX_DELTA,
                band_lo=config.MATANYONE2_GUIDED_BAND_LO,
                band_hi=config.MATANYONE2_GUIDED_BAND_HI,
            )
            self.profile["guided_upsample"].append((time.perf_counter() - t0) * 1000)
            return refined
        except Exception as exc:
            self._guided_upsample_failed = True
            print(
                f"[offline] MatAnyone2 guided alpha upsample failed; falling back to bilinear composite "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            return alpha_sbs

    def _emit_output(self, alpha_sbs, h: int, w: int):
        t0 = time.perf_counter()
        if self.output_mode == "alpha":
            out = self.packer.pack_uploaded(alpha_sbs, h, w)
            self.profile["alpha_pack"].append((time.perf_counter() - t0) * 1000)
            return out
        out = self.matter._composite_nv12_to_nv12_gpu_using_uploaded_frame(alpha_sbs, h, w)
        self.profile["composite"].append((time.perf_counter() - t0) * 1000)
        return out

    def profile_lines(self) -> list[str]:
        order = [
            "upload_nv12",
            "preprocess_eye",
            "step_update",
            "image_key",
            "propagate_update",
            "propagate",
            "mask_memory",
            "first_refine",
            "sensory_decay",
            "alpha_concat",
            "alpha_smooth",
            "roi_unwarp",
            "alpha_reuse",
            "iobinding_copy",
            "guided_upsample",
            "alpha_pack",
            "composite",
        ]
        lines = []
        for key in order:
            values = self.profile.get(key)
            if values:
                lines.append(f"matanyone2_{key}_avg = {statistics.fmean(values):.3f} ms n={len(values)}")
        return lines
