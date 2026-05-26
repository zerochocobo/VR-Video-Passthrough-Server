"""Standalone MatAnyone2 ONNX video inference example.

This file is intended for users of the exported MatAnyone2 ONNX folders:

    matanyone2_onnx_1024_bs1
    matanyone2_onnx_1024_bs2

It demonstrates every exported ONNX graph:

    matanyone2_image_key.onnx
    matanyone2_mask_memory.onnx
    matanyone2_first_frame_refine.onnx
    matanyone2_propagate.onnx
    matanyone2_propagate_update.onnx   optional
    matanyone2_step_update.onnx        optional

The example takes a video and a first-frame foreground mask, propagates the
mask through the video, and writes either a grayscale alpha video or a green
background preview video.

Dependencies:

    pip install onnxruntime-gpu opencv-python numpy

CPU also works for small tests:

    pip install onnxruntime opencv-python numpy

Usage examples:

    python examples/matanyone2_onnx_video_infer.py ^
      --model-dir models/matanyone2_onnx_1024_bs1 ^
      --video input.mp4 ^
      --mask first_frame_mask.png ^
      --out alpha_preview.mp4

    python examples/matanyone2_onnx_video_infer.py ^
      --model-dir models/matanyone2_onnx_1024_bs1 ^
      --video sbs_vr180.mp4 ^
      --mask first_frame_mask_sbs.png ^
      --sbs ^
      --output-mode green ^
      --out green_preview.mp4
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def _providers(use_cuda: bool) -> list[str]:
    available = set(ort.get_available_providers())
    out: list[str] = []
    if use_cuda and "CUDAExecutionProvider" in available:
        out.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        out.append("CPUExecutionProvider")
    if not out:
        out = ort.get_available_providers()
    return out


def _load_session(model_dir: Path, name: str, providers: list[str]) -> ort.InferenceSession | None:
    path = model_dir / name
    if not path.exists():
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _input_names(session: ort.InferenceSession | None) -> set[str]:
    if session is None:
        return set()
    return {meta.name for meta in session.get_inputs()}


def _feed_supported(session: ort.InferenceSession, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = _input_names(session)
    return {name: value for name, value in feed.items() if name in names}


def _preprocess_bgr(frame_bgr: np.ndarray, width: int, height: int, dtype: np.dtype) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw.astype(dtype, copy=False))


def _load_mask(path: Path, width: int, height: int, dtype: np.dtype) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    arr = resized.astype(np.float32) / 255.0
    return arr[None, None, :, :].astype(dtype, copy=False)


def _split_sbs_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame_bgr.shape[:2]
    half = w // 2
    if half <= 0 or w < 2 * h:
        raise RuntimeError(f"--sbs expects side-by-side video, got {w}x{h}")
    return frame_bgr[:, :half], frame_bgr[:, half:half * 2]


def _split_sbs_mask(mask_path: Path, width: int, height: int, dtype: np.dtype) -> list[np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    mh, mw = mask.shape[:2]
    if mw >= 2 * mh:
        half = mw // 2
        parts = [mask[:, :half], mask[:, half:half * 2]]
    else:
        parts = [mask, mask]
    out = []
    for part in parts:
        resized = cv2.resize(part, (width, height), interpolation=cv2.INTER_NEAREST)
        out.append((resized.astype(np.float32) / 255.0)[None, None, :, :].astype(dtype, copy=False))
    return out


def _alpha_to_output_frame(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    output_mode: str,
    green_bgr: tuple[int, int, int],
) -> np.ndarray:
    alpha_resized = cv2.resize(alpha, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha_resized = np.clip(alpha_resized, 0.0, 1.0)
    if output_mode == "alpha":
        gray = (alpha_resized * 255.0).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bg = np.empty_like(frame_bgr)
    bg[:, :] = np.array(green_bgr, dtype=np.uint8)
    a3 = alpha_resized[:, :, None].astype(np.float32)
    return np.clip(frame_bgr.astype(np.float32) * a3 + bg.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)


@dataclass
class StreamState:
    sensory: np.ndarray | None = None
    memory_key: np.ndarray | None = None
    memory_shrinkage: np.ndarray | None = None
    memory_msk_value: np.ndarray | None = None
    obj_memory: np.ndarray | None = None
    last_pix_feat: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_msk_value: np.ndarray | None = None
    initialized: bool = False


class MatAnyone2OnnxRunner:
    """Small ONNX Runtime wrapper around the exported MatAnyone2 subgraphs."""

    def __init__(self, model_dir: Path, use_cuda: bool = True):
        self.model_dir = model_dir
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"manifest.json not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        self.height = int(manifest.get("height") or 512)
        self.width = int(manifest.get("width") or 512)
        self.batch_size = int(manifest.get("batch_size") or 1)
        self.providers = _providers(use_cuda)

        self.image_key_sess = _load_session(model_dir, "matanyone2_image_key.onnx", self.providers)
        self.mask_memory_sess = _load_session(model_dir, "matanyone2_mask_memory.onnx", self.providers)
        self.first_refine_sess = _load_session(model_dir, "matanyone2_first_frame_refine.onnx", self.providers)
        self.propagate_sess = _load_session(model_dir, "matanyone2_propagate.onnx", self.providers)
        self.propagate_update_sess = _load_session(model_dir, "matanyone2_propagate_update.onnx", self.providers)
        self.step_update_sess = _load_session(model_dir, "matanyone2_step_update.onnx", self.providers)

        required = {
            "matanyone2_image_key.onnx": self.image_key_sess,
            "matanyone2_mask_memory.onnx": self.mask_memory_sess,
            "matanyone2_first_frame_refine.onnx": self.first_refine_sess,
            "matanyone2_propagate.onnx": self.propagate_sess,
        }
        missing = [name for name, sess in required.items() if sess is None]
        if missing:
            raise RuntimeError(f"missing required MatAnyone2 ONNX files: {missing}")

        input_type = self.image_key_sess.get_inputs()[0].type
        self.dtype = np.float16 if input_type == "tensor(float16)" else np.float32
        sensory_meta = next(meta for meta in self.mask_memory_sess.get_inputs() if meta.name == "sensory")
        self.sensory_shape = tuple(int(v) for v in sensory_meta.shape)
        self.single_sensory_shape = (1, *self.sensory_shape[1:])

    def image_key(self, image: np.ndarray) -> dict[str, np.ndarray]:
        names = ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"]
        outs = self.image_key_sess.run(names, {"image": image})
        return dict(zip(names, outs))

    def mask_memory(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        sensory: np.ndarray,
        pix_feat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(self.mask_memory_sess.run(
            ["msk_value", "new_sensory", "obj_memory"],
            {
                "image": image,
                "mask": mask,
                "sensory": sensory,
                "pix_feat": pix_feat,
            },
        ))

    def first_frame_refine(
        self,
        feats: dict[str, np.ndarray],
        last_msk_value: np.ndarray,
        obj_memory: np.ndarray,
        sensory: np.ndarray,
        last_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        prob, sensory, _logits = self.first_refine_sess.run(
            ["prob", "new_sensory", "logits"],
            _feed_supported(self.first_refine_sess, feed),
        )
        return prob, sensory

    def propagate(self, feats: dict[str, np.ndarray], state: StreamState) -> tuple[np.ndarray, np.ndarray]:
        feed = {
            "f16": feats["f16"],
            "f8": feats["f8"],
            "f4": feats["f4"],
            "f2": feats["f2"],
            "f1": feats["f1"],
            "pix_feat": feats["pix_feat"],
            "key": feats["key"],
            "selection": feats["selection"],
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": state.last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": state.last_mask,
            "last_msk_value": state.last_msk_value,
        }
        prob, sensory, _logits, _uncert_prob = self.propagate_sess.run(
            ["prob", "new_sensory", "logits", "uncert_prob"],
            _feed_supported(self.propagate_sess, feed),
        )
        return prob, sensory

    def propagate_update(
        self,
        image: np.ndarray,
        feats: dict[str, np.ndarray],
        state: StreamState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.propagate_update_sess is None:
            prob, sensory = self.propagate(feats, state)
            alpha = np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.dtype, copy=False)
            msk_value, sensory, obj_memory = self.mask_memory(image, alpha, sensory, feats["pix_feat"])
            return prob, sensory, msk_value, obj_memory
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
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": state.last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": state.last_mask,
            "last_msk_value": state.last_msk_value,
        }
        prob, sensory, msk_value, obj_memory, _logits, _uncert_prob = self.propagate_update_sess.run(
            ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "logits", "uncert_prob"],
            _feed_supported(self.propagate_update_sess, feed),
        )
        return prob, sensory, msk_value, obj_memory

    def step_update(
        self,
        image: np.ndarray,
        state: StreamState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.step_update_sess is None:
            feats = self.image_key(image)
            prob, sensory, msk_value, obj_memory = self.propagate_update(image, feats, state)
            return prob, sensory, msk_value, obj_memory, feats["pix_feat"]
        feed = {
            "image": image,
            "memory_key": state.memory_key,
            "memory_shrinkage": state.memory_shrinkage,
            "msk_value": state.memory_msk_value,
            "obj_memory": state.obj_memory,
            "sensory": state.sensory,
            "last_mask": state.last_mask,
            "last_pix_feat": state.last_pix_feat,
            "last_pred_mask": state.last_mask,
            "last_msk_value": state.last_msk_value,
        }
        prob, sensory, msk_value, obj_memory, pix_feat, _logits, _uncert_prob = self.step_update_sess.run(
            ["prob", "new_sensory", "new_msk_value", "new_obj_memory", "pix_feat", "logits", "uncert_prob"],
            _feed_supported(self.step_update_sess, feed),
        )
        return prob, sensory, msk_value, obj_memory, pix_feat

    def run_frame(self, image: np.ndarray, bootstrap_mask: np.ndarray, state: StreamState) -> np.ndarray:
        if not state.initialized:
            feats = self.image_key(image)
            sensory = np.zeros((image.shape[0], *self.sensory_shape[1:]), dtype=self.dtype)
            msk_value, sensory, obj_memory = self.mask_memory(image, bootstrap_mask, sensory, feats["pix_feat"])
            prob, sensory = self.first_frame_refine(
                feats,
                msk_value,
                obj_memory[:, :, None, :, :],
                sensory,
                bootstrap_mask,
            )
            alpha = np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.dtype, copy=False)
            msk_value, sensory, obj_memory = self.mask_memory(image, alpha, sensory, feats["pix_feat"])
            state.memory_key = feats["key"][:, :, None, :, :].astype(self.dtype, copy=False)
            state.memory_shrinkage = feats["shrinkage"][:, :, None, :, :].astype(self.dtype, copy=False)
            state.memory_msk_value = msk_value[:, :, :, None, :, :].astype(self.dtype, copy=False)
            state.obj_memory = obj_memory[:, :, None, :, :].astype(self.dtype, copy=False)
            state.sensory = sensory.astype(self.dtype, copy=False)
            state.last_mask = alpha
            state.last_pix_feat = feats["pix_feat"].astype(self.dtype, copy=False)
            state.last_msk_value = msk_value.astype(self.dtype, copy=False)
            state.initialized = True
            return alpha[:, 0]

        prob, sensory, msk_value, _obj_memory, pix_feat = self.step_update(image, state)
        alpha = np.clip(prob[:, 1:2], 0.0, 1.0).astype(self.dtype, copy=False)
        state.sensory = sensory.astype(self.dtype, copy=False)
        state.last_mask = alpha
        state.last_pix_feat = pix_feat.astype(self.dtype, copy=False)
        state.last_msk_value = msk_value.astype(self.dtype, copy=False)
        return alpha[:, 0]


def run_video(args: argparse.Namespace) -> None:
    runner = MatAnyone2OnnxRunner(Path(args.model_dir).resolve(), use_cuda=not args.cpu)
    if args.sbs and runner.batch_size < 2:
        print("[info] SBS input with bs1 model: eyes will run sequentially.")
    if not args.sbs and runner.batch_size >= 2:
        raise RuntimeError("Use a bs1 model for non-SBS video, or pass --sbs for side-by-side video.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("video has no readable frames")

    states = [StreamState(), StreamState()] if args.sbs else [StreamState()]
    if args.sbs:
        masks = _split_sbs_mask(Path(args.mask), runner.width, runner.height, runner.dtype)
        left, right = _split_sbs_frame(first)
        if runner.batch_size >= 2:
            first_images = np.concatenate([
                _preprocess_bgr(left, runner.width, runner.height, runner.dtype),
                _preprocess_bgr(right, runner.width, runner.height, runner.dtype),
            ], axis=0)
            first_masks = np.concatenate(masks, axis=0)
            first_alpha = runner.run_frame(first_images, first_masks, states[0])
            # Split the batched state into two independent states for clarity.
            batched_state = states[0]
            states = [StreamState(), StreamState()]
            for idx, state in enumerate(states):
                state.memory_key = batched_state.memory_key[idx:idx + 1]
                state.memory_shrinkage = batched_state.memory_shrinkage[idx:idx + 1]
                state.memory_msk_value = batched_state.memory_msk_value[idx:idx + 1]
                state.obj_memory = batched_state.obj_memory[idx:idx + 1]
                state.sensory = batched_state.sensory[idx:idx + 1]
                state.last_mask = batched_state.last_mask[idx:idx + 1]
                state.last_pix_feat = batched_state.last_pix_feat[idx:idx + 1]
                state.last_msk_value = batched_state.last_msk_value[idx:idx + 1]
                state.initialized = True
            alpha_parts = [first_alpha[0], first_alpha[1]]
        else:
            alpha_parts = [
                runner.run_frame(_preprocess_bgr(left, runner.width, runner.height, runner.dtype), masks[0], states[0])[0],
                runner.run_frame(_preprocess_bgr(right, runner.width, runner.height, runner.dtype), masks[1], states[1])[0],
            ]
        output_first = np.concatenate([
            _alpha_to_output_frame(left, alpha_parts[0], args.output_mode, tuple(args.green_bgr)),
            _alpha_to_output_frame(right, alpha_parts[1], args.output_mode, tuple(args.green_bgr)),
        ], axis=1)
    else:
        mask = _load_mask(Path(args.mask), runner.width, runner.height, runner.dtype)
        image = _preprocess_bgr(first, runner.width, runner.height, runner.dtype)
        alpha = runner.run_frame(image, mask, states[0])[0]
        output_first = _alpha_to_output_frame(first, alpha, args.output_mode, tuple(args.green_bgr))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = output_first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output writer: {out_path}")
    writer.write(output_first)

    frame_index = 1
    while True:
        if args.max_frames and frame_index >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if args.sbs:
            left, right = _split_sbs_frame(frame)
            if runner.batch_size >= 2:
                images = np.concatenate([
                    _preprocess_bgr(left, runner.width, runner.height, runner.dtype),
                    _preprocess_bgr(right, runner.width, runner.height, runner.dtype),
                ], axis=0)
                batched_state = StreamState(
                    sensory=np.concatenate([s.sensory for s in states], axis=0),
                    memory_key=np.concatenate([s.memory_key for s in states], axis=0),
                    memory_shrinkage=np.concatenate([s.memory_shrinkage for s in states], axis=0),
                    memory_msk_value=np.concatenate([s.memory_msk_value for s in states], axis=0),
                    obj_memory=np.concatenate([s.obj_memory for s in states], axis=0),
                    last_pix_feat=np.concatenate([s.last_pix_feat for s in states], axis=0),
                    last_mask=np.concatenate([s.last_mask for s in states], axis=0),
                    last_msk_value=np.concatenate([s.last_msk_value for s in states], axis=0),
                    initialized=True,
                )
                alpha_batch = runner.run_frame(images, np.zeros((2, 1, runner.height, runner.width), dtype=runner.dtype), batched_state)
                for idx, state in enumerate(states):
                    state.sensory = batched_state.sensory[idx:idx + 1]
                    state.last_mask = batched_state.last_mask[idx:idx + 1]
                    state.last_pix_feat = batched_state.last_pix_feat[idx:idx + 1]
                    state.last_msk_value = batched_state.last_msk_value[idx:idx + 1]
                alpha_parts = [alpha_batch[0], alpha_batch[1]]
            else:
                alpha_parts = [
                    runner.run_frame(_preprocess_bgr(left, runner.width, runner.height, runner.dtype), masks[0], states[0])[0],
                    runner.run_frame(_preprocess_bgr(right, runner.width, runner.height, runner.dtype), masks[1], states[1])[0],
                ]
            out_frame = np.concatenate([
                _alpha_to_output_frame(left, alpha_parts[0], args.output_mode, tuple(args.green_bgr)),
                _alpha_to_output_frame(right, alpha_parts[1], args.output_mode, tuple(args.green_bgr)),
            ], axis=1)
        else:
            image = _preprocess_bgr(frame, runner.width, runner.height, runner.dtype)
            alpha = runner.run_frame(image, mask, states[0])[0]
            out_frame = _alpha_to_output_frame(frame, alpha, args.output_mode, tuple(args.green_bgr))
        writer.write(out_frame)
        frame_index += 1
        if args.progress and frame_index % args.progress == 0:
            print(f"[progress] {frame_index}/{total or '?'}")

    writer.release()
    cap.release()
    print(f"[done] wrote {frame_index} frames to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MatAnyone2 ONNX on a video with a first-frame mask.")
    parser.add_argument("--model-dir", required=True, help="directory containing MatAnyone2 ONNX files and manifest.json")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--mask", required=True, help="first-frame foreground mask; grayscale or SBS grayscale")
    parser.add_argument("--out", required=True, help="output video path")
    parser.add_argument("--sbs", action="store_true", help="treat the video as side-by-side left/right eyes")
    parser.add_argument("--output-mode", choices=["alpha", "green"], default="alpha")
    parser.add_argument("--green-bgr", type=int, nargs=3, default=(0, 255, 0), help="B G R background for green mode")
    parser.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider")
    parser.add_argument("--max-frames", type=int, default=0, help="optional frame limit for testing")
    parser.add_argument("--progress", type=int, default=30)
    parser.add_argument("--fourcc", default="mp4v", help="OpenCV VideoWriter fourcc, e.g. mp4v or avc1")
    args = parser.parse_args()
    run_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
