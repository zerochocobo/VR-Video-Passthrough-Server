"""YOLO-World + EfficientSAM prepass helpers for MatAnyone2 offline mode."""
from __future__ import annotations

import json
import sys
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort

import config
from offline.decoded_frames import decoded_frame_to_bgr
from utils.scene_detection import SceneCutDetector
from utils.subprocess_hidden import hidden_subprocess_kwargs, run_hidden_streaming


@dataclass
class Detection:
    box_xyxy: np.ndarray
    score: float
    class_id: int = 0


def onnx_providers(provider: str = "cuda") -> list:
    available = set(ort.get_available_providers())
    if provider == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    positive = arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _letterbox_rgb(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    h, w = image_rgb.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0 = int(round(pad_x))
    y0 = int(round(pad_y))
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, float(x0), float(y0)


def _preprocess_yolo(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    canvas, scale, pad_x, pad_y = _letterbox_rgb(image_rgb, size)
    chw = canvas.astype(np.float32) / 255.0
    chw = np.transpose(chw, (2, 0, 1))[None]
    return np.ascontiguousarray(chw), scale, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1.0e-6)
        order = rest[iou <= iou_threshold]
    return keep


class YoloWorldEfficientSamMasker:
    def __init__(
        self,
        model_dir: Path,
        txt_feats_path: Path,
        provider: str = "cuda",
        yolo_model: str = "yolov8l-worldv2.onnx",
        sam_model: str = "efficientsam_s.onnx",
        yolo_size: int = 1280,
        score_threshold: float = 0.03,
        nms_threshold: float = 0.6,
        box_expand: float = 0.08,
        top_k: int = 1,
    ) -> None:
        self.model_dir = model_dir
        self.yolo_size = int(yolo_size)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.box_expand = float(box_expand)
        self.top_k = max(1, int(top_k))
        providers = onnx_providers(provider)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.yolo = ort.InferenceSession(str(model_dir / yolo_model), sess_options=opts, providers=providers)
        self.sam = ort.InferenceSession(str(model_dir / sam_model), sess_options=opts, providers=providers)
        self.txt_feats = np.load(txt_feats_path).astype(np.float32, copy=False)
        if self.txt_feats.ndim != 3 or self.txt_feats.shape[0] != 1 or self.txt_feats.shape[2] != 512:
            raise RuntimeError(f"unexpected YOLO-World txt_feats shape: {self.txt_feats.shape}")
        self.num_classes = int(self.txt_feats.shape[1])
        print(
            f"[offline] YOLO-World+EfficientSAM loaded dir={model_dir} yolo={yolo_model} sam={sam_model} "
            f"provider={providers} yolo_size={self.yolo_size} score={self.score_threshold:g} "
            f"nms={self.nms_threshold:g} expand={self.box_expand:g} top_k={self.top_k} "
            f"txt_classes={self.num_classes}"
        )

    def detect(self, image_rgb: np.ndarray, top_k: int | None = None) -> list[Detection]:
        h, w = image_rgb.shape[:2]
        inp, scale, pad_x, pad_y = _preprocess_yolo(image_rgb, self.yolo_size)
        out = self.yolo.run(None, {"images": inp, "txt_feats": self.txt_feats})[0][0]
        pred = out.T.astype(np.float32, copy=False)
        if pred.shape[1] < 4 + self.num_classes:
            raise RuntimeError(
                f"unexpected YOLO-World output shape {out.shape}; "
                f"expected at least {4 + self.num_classes} channels for {self.num_classes} text classes"
            )
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:4 + self.num_classes]
        if np.nanmin(class_scores) < 0.0:
            class_scores = _sigmoid(class_scores)
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = scores >= self.score_threshold
        if not np.any(keep):
            return []
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - bw / 2.0 - pad_x) / max(scale, 1.0e-6)
        y1 = (cy - bh / 2.0 - pad_y) / max(scale, 1.0e-6)
        x2 = (cx + bw / 2.0 - pad_x) / max(scale, 1.0e-6)
        y2 = (cy + bh / 2.0 - pad_y) / max(scale, 1.0e-6)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        valid = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        keep_indices = _nms(boxes, scores, self.nms_threshold)[: (top_k or self.top_k)]
        return [Detection(boxes[i], float(scores[i]), int(class_ids[i])) for i in keep_indices]

    @staticmethod
    def _box_stats(det: Detection, image_shape: tuple[int, int, int]) -> dict:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = det.box_xyxy.astype(np.float32)
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        return {
            "area": float((bw * bh) / float(max(1, w * h))),
            "aspect": float(bw / bh),
            "cx": float(((x1 + x2) * 0.5) / max(1, w)),
            "cy": float(((y1 + y2) * 0.5) / max(1, h)),
            "height": float(bh / max(1, h)),
            "width": float(bw / max(1, w)),
        }

    def _is_plausible_person_box(self, det: Detection, image_shape: tuple[int, int, int]) -> bool:
        stats = self._box_stats(det, image_shape)
        return (
            0.02 <= stats["area"] <= 0.24
            and 0.12 <= stats["aspect"] <= 1.05
            and stats["height"] >= 0.25
        )

    def _project_detection(self, det: Detection, from_shape: tuple[int, int, int], to_shape: tuple[int, int, int]) -> Detection:
        from_h, from_w = from_shape[:2]
        to_h, to_w = to_shape[:2]
        x1, y1, x2, y2 = det.box_xyxy.astype(np.float32)
        projected = np.array(
            [
                x1 / max(1, from_w) * to_w,
                y1 / max(1, from_h) * to_h,
                x2 / max(1, from_w) * to_w,
                y2 / max(1, from_h) * to_h,
            ],
            dtype=np.float32,
        )
        projected[[0, 2]] = np.clip(projected[[0, 2]], 0, to_w - 1)
        projected[[1, 3]] = np.clip(projected[[1, 3]], 0, to_h - 1)
        return Detection(projected, float(det.score), int(det.class_id))

    def select_stereo_detections(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        candidate_k: int = 8,
    ) -> tuple[list[Detection], list[Detection], dict]:
        left_all = self.detect(left_rgb, top_k=max(candidate_k, self.top_k))
        right_all = self.detect(right_rgb, top_k=max(candidate_k, self.top_k))
        left = [det for det in left_all if self._is_plausible_person_box(det, left_rgb.shape)]
        right = [det for det in right_all if self._is_plausible_person_box(det, right_rgb.shape)]
        mode = "independent"
        if left and right:
            best_pair = None
            best_cost = float("inf")
            for ldet in left:
                ls = self._box_stats(ldet, left_rgb.shape)
                for rdet in right:
                    rs = self._box_stats(rdet, right_rgb.shape)
                    cost = (
                        abs(ls["cy"] - rs["cy"]) * 4.0
                        + abs(ls["height"] - rs["height"]) * 2.0
                        + abs(ls["aspect"] - rs["aspect"])
                        + abs(ls["area"] - rs["area"]) * 2.0
                        - (ldet.score + rdet.score)
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (ldet, rdet)
            assert best_pair is not None
            left_sel = [best_pair[0]]
            right_sel = [best_pair[1]]
            mode = "paired"
        elif left:
            left_sel = [left[0]]
            right_sel = [self._project_detection(left[0], left_rgb.shape, right_rgb.shape)]
            mode = "project_left_to_right"
        elif right:
            left_sel = [self._project_detection(right[0], right_rgb.shape, left_rgb.shape)]
            right_sel = [right[0]]
            mode = "project_right_to_left"
        else:
            left_sel = left_all[:1]
            right_sel = right_all[:1]
            mode = "fallback_unfiltered"
        return left_sel, right_sel, {
            "stereo_mode": mode,
            "left_candidates": len(left_all),
            "right_candidates": len(right_all),
            "left_plausible": len(left),
            "right_plausible": len(right),
        }

    def _sam_mask_for_box(self, image_rgb: np.ndarray, box_xyxy: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        out_w, out_h = out_size
        src_h, src_w = image_rgb.shape[:2]
        image_small = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = box_xyxy.astype(np.float32)
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * self.box_expand
        x2 += bw * self.box_expand
        y1 -= bh * self.box_expand
        y2 += bh * self.box_expand
        sx = out_w / max(1, src_w)
        sy = out_h / max(1, src_h)
        coords = np.array([[[[x1 * sx, y1 * sy], [x2 * sx, y2 * sy]]]], dtype=np.float32)
        coords[..., 0] = np.clip(coords[..., 0], 0, out_w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, out_h - 1)
        labels = np.array([[[2.0, 3.0]]], dtype=np.float32)
        batched_images = np.transpose(image_small.astype(np.float32) / 255.0, (2, 0, 1))[None]
        masks, ious, *_ = self.sam.run(
            None,
            {
                "batched_images": np.ascontiguousarray(batched_images),
                "batched_point_coords": coords,
                "batched_point_labels": labels,
            },
        )
        masks = np.asarray(masks)
        ious = np.asarray(ious)
        flat_masks = masks.reshape(-1, out_h, out_w)
        flat_ious = ious.reshape(-1)
        idx = int(np.argmax(flat_ious)) if flat_ious.size else 0
        mask = flat_masks[idx].astype(np.float32, copy=False)
        if mask.min() < 0.0 or mask.max() > 1.0:
            mask = _sigmoid(mask)
        return np.clip(mask, 0.0, 1.0)

    def mask(self, image_rgb: np.ndarray, out_size: tuple[int, int], detections: list[Detection] | None = None) -> tuple[np.ndarray, dict]:
        detections = self.detect(image_rgb) if detections is None else detections
        out_w, out_h = out_size
        if not detections:
            return np.zeros((out_h, out_w), dtype=np.float32), {
                "count": 0,
                "selected": [],
                "scores": [],
                "top_score": 0.0,
                "area_ratios": [],
                "union_area_ratio": 0.0,
            }
        masks = [self._sam_mask_for_box(image_rgb, det.box_xyxy, out_size) for det in detections]
        union = np.maximum.reduce(masks).astype(np.float32, copy=False)
        area_ratio = float((union >= 0.5).sum() / float(out_h * out_w))
        return union, {
            "count": len(detections),
            "selected": list(range(len(detections))),
            "scores": [float(det.score) for det in detections],
            "top_score": float(max(det.score for det in detections)),
            "boxes_xyxy": [det.box_xyxy.astype(float).tolist() for det in detections],
            "class_ids": [int(getattr(det, "class_id", 0)) for det in detections],
            "area_ratios": [area_ratio],
            "union_area_ratio": area_ratio,
        }


def _probe_keyframe_indices(path: Path, source_fps: float, target: int, output_fps: float) -> list[int]:
    ffprobe = "ffprobe"
    cmd = [
        ffprobe, "-hide_banner", "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey", "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,coded_picture_number",
        "-of", "json", str(path),
    ]
    try:
        data = json.loads(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs()))
    except Exception:
        return []
    indices = []
    for frame in data.get("frames") or []:
        idx = None
        ts = frame.get("best_effort_timestamp_time") or frame.get("pkt_pts_time")
        if ts is not None:
            try:
                idx = int(round(float(ts) * output_fps))
            except Exception:
                idx = None
        if idx is None:
            try:
                coded = int(frame.get("coded_picture_number"))
                idx = int(round(coded * output_fps / source_fps)) if source_fps > 0 else coded
            except Exception:
                idx = None
        if idx is not None and 0 <= idx < target:
            indices.append(idx)
    return sorted(set(indices))


def _planned_starts(
    records: list[dict],
    target: int,
    max_frames: int,
    min_frames: int,
    cut_on_count_change: bool,
    cut_every_active_sample: bool,
    scene_min_frames: int = 0,
) -> list[int]:
    if not records:
        return [0]
    starts = [int(records[0]["frame"])]
    last = starts[0]
    last_active = bool(records[0]["active"])
    last_count = int(records[0]["object_count"])
    for record in records[1:]:
        idx = int(record["frame"])
        active = bool(record["active"])
        count = int(record["object_count"])
        force_cut = active != last_active
        if active and cut_on_count_change and count != last_count:
            force_cut = True
        if active and cut_every_active_sample:
            force_cut = True
        if active and bool(record.get("scene_cut")) and (scene_min_frames <= 0 or idx - last >= scene_min_frames):
            force_cut = True
        timed_cut = (min_frames > 0 and idx - last >= min_frames) or (max_frames > 0 and idx - last >= max_frames)
        if force_cut or timed_cut:
            starts.append(idx)
            last = idx
        last_active = active
        last_count = max(last_count, count) if active else count
    while max_frames > 0 and target - last > max_frames:
        last += max_frames
        starts.append(last)
    return sorted(set(x for x in starts if 0 <= x < target))


def _fill_short_inactive_gaps(
    records: list[dict],
    masks_by_start: dict[int, list[np.ndarray]],
    max_gap_frames: int,
) -> list[int]:
    if max_gap_frames <= 0 or len(records) < 3:
        return []
    filled: list[int] = []
    i = 0
    while i < len(records):
        if records[i].get("active"):
            i += 1
            continue
        run_start = i
        while i < len(records) and not records[i].get("active"):
            i += 1
        run_end = i
        prev_idx = run_start - 1
        next_idx = run_end
        if prev_idx < 0 or next_idx >= len(records):
            continue
        prev = records[prev_idx]
        nxt = records[next_idx]
        if not prev.get("active") or not nxt.get("active"):
            continue
        prev_frame = int(prev["frame"])
        next_frame = int(nxt["frame"])
        if next_frame - prev_frame > max_gap_frames:
            continue
        prev_masks = masks_by_start.get(prev_frame)
        next_masks = masks_by_start.get(next_frame)
        if prev_masks is None and next_masks is None:
            continue
        object_count = max(int(prev.get("object_count") or 0), int(nxt.get("object_count") or 0), 1)
        for j in range(run_start, run_end):
            frame = int(records[j]["frame"])
            source_masks = prev_masks if (next_masks is None or frame - prev_frame <= next_frame - frame) else next_masks
            if source_masks is None:
                continue
            masks_by_start[frame] = [mask.copy() for mask in source_masks]
            records[j]["active"] = True
            records[j]["object_count"] = object_count
            records[j]["gap_filled"] = True
            filled.append(frame)
    return filled


def sample_points(args, src: Path, source_fps: float, fps: float, target: int) -> list[int]:
    scan = str(getattr(args, "ywes_scan", "hybrid"))
    interval = float(getattr(args, "ywes_scan_interval_sec", 1.0))
    if scan in {"keyframe", "hybrid"}:
        candidates = _probe_keyframe_indices(src, source_fps, target, fps)
        if scan == "hybrid":
            step = max(1, int(round(max(0.1, interval) * fps)))
            candidates = sorted(set(candidates) | set(range(0, target, step)))
        if not candidates:
            step = max(1, int(round(max(0.1, interval) * fps)))
            candidates = list(range(0, target, step))
    else:
        step = max(1, int(round(max(0.1, interval) * fps)))
        candidates = list(range(0, target, step))
    points = sorted(set(x for x in candidates if 0 <= x < target))
    if 0 not in points:
        points.insert(0, 0)
    return points


def precompute_segment_masks(args, src: Path, dec, source_fps: float, fps: float, target: int, cfr_source_index: Callable[[int, float, float], int]):
    if getattr(args, "engine", "") != "matanyone2_onnx" or getattr(args, "mask", ""):
        return {}, [0]
    if bool(getattr(args, "ywes_subprocess", True)) and not bool(getattr(args, "_ywes_child", False)):
        return precompute_segment_masks_subprocess(args, src, source_fps, fps, target)
    scan_points = sample_points(args, src, source_fps, fps, target)
    max_segment_frames = max(1, int(getattr(args, "matanyone2_segment_frames", 300) or target))
    min_segment_frames = max(1, int(round(max(0.0, getattr(args, "matanyone2_min_segment_sec", 3.0)) * fps)))
    scene_detector = (
        SceneCutDetector(
            threshold=config.MATANYONE2_SCENE_THRESHOLD,
            cooldown_frames=config.MATANYONE2_SCENE_COOLDOWN,
            ref_ema_alpha=config.MATANYONE2_SCENE_REF_EMA,
        )
        if config.MATANYONE2_SCENE_RESET
        else None
    )
    scene_min_frames = (
        max(1, int(round(max(0.0, config.MATANYONE2_SCENE_MIN_SEGMENT_SEC) * fps)))
        if scene_detector is not None
        else 0
    )
    masker = YoloWorldEfficientSamMasker(
        Path(getattr(args, "ywes_model_dir", config.ROOT / "models" / "yoloworld_efficientsam")).resolve(),
        Path(getattr(args, "ywes_txt_feats", config.ROOT / "models" / "person_txt_feats.npy")).resolve(),
        provider=str(getattr(args, "ywes_provider", "cuda")),
        yolo_model=str(getattr(args, "ywes_yolo_model", "yolov8l-worldv2.onnx")),
        sam_model=str(getattr(args, "ywes_sam_model", "efficientsam_s.onnx")),
        yolo_size=int(getattr(args, "ywes_yolo_size", 1280)),
        score_threshold=float(getattr(args, "ywes_score_threshold", 0.03)),
        nms_threshold=float(getattr(args, "ywes_nms_threshold", 0.6)),
        box_expand=float(getattr(args, "ywes_box_expand", 0.08)),
        top_k=int(getattr(args, "ywes_top_k", 1)),
    )
    debug_dir = Path(args.ywes_debug_dir).resolve() if getattr(args, "ywes_debug_dir", "") else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[offline] YOLO-World+EfficientSAM prepass samples={len(scan_points)} "
        f"scan={getattr(args, 'ywes_scan', 'hybrid')} max_segment_frames={max_segment_frames} "
        f"scene_reset={int(scene_detector is not None)}"
    )
    masks_by_start: dict[int, list[np.ndarray]] = {}
    records: list[dict] = []
    for n, start in enumerate(scan_points, 1):
        src_idx = min(len(dec) - 1, cfr_source_index(start, source_fps, fps))
        frame = dec.frame_at(src_idx)
        bgr = decoded_frame_to_bgr(frame)
        half = frame.width // 2
        scene_cut = False
        scene_distance = 0.0
        if scene_detector is not None:
            scene_cut = scene_detector.step(bgr[:, :half] if half > 0 else bgr)
            scene_distance = scene_detector.last_distance
        eye_images = [
            cv2.cvtColor(bgr[:, :half], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bgr[:, half:half * 2], cv2.COLOR_BGR2RGB),
        ]
        masks = []
        infos = []
        t0 = time.perf_counter()
        stereo_detections = None
        stereo_info = {}
        if len(eye_images) == 2:
            left_dets, right_dets, stereo_info = masker.select_stereo_detections(eye_images[0], eye_images[1])
            stereo_detections = [left_dets, right_dets]
        for eye_idx, image_rgb in enumerate(eye_images):
            dets = stereo_detections[eye_idx] if stereo_detections is not None else None
            mask, info = masker.mask(image_rgb, (int(args._matanyone2_in_w), int(args._matanyone2_in_h)), dets)
            if stereo_info:
                info.update(stereo_info)
            infos.append(info)
            masks.append(mask[None, None, :, :].astype(np.float32, copy=False))
            if debug_dir is not None:
                eye_name = "left" if eye_idx == 0 else "right"
                frame_small = cv2.resize(image_rgb, (int(args._matanyone2_in_w), int(args._matanyone2_in_h)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(debug_dir / f"seg_{start:06d}_{eye_name}_frame.png"), cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(debug_dir / f"seg_{start:06d}_{eye_name}_mask.png"), (mask * 255).astype(np.uint8))
        active = (
            infos[0]["union_area_ratio"] >= float(getattr(args, "ywes_active_min_area_ratio", 0.001))
            or infos[1]["union_area_ratio"] >= float(getattr(args, "ywes_active_min_area_ratio", 0.001))
        )
        object_count = max((len(info.get("selected") or []) for info in infos), default=0) if active else 0
        if active:
            masks_by_start[start] = masks
        records.append(
            {
                "frame": int(start),
                "src_idx": int(src_idx),
                "active": bool(active),
                "object_count": int(object_count),
                "scene_cut": bool(scene_cut),
            }
        )
        if debug_dir is not None:
            (debug_dir / f"seg_{start:06d}_info.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
        print(
            f"[offline] YW+ES prepass {n}/{len(scan_points)} frame={start} src_idx={src_idx} "
            f"ms={(time.perf_counter() - t0) * 1000:.1f} active={active} objects={object_count} "
            f"L={infos[0]['selected']} score={float(infos[0].get('top_score') or 0.0):.4f} area={infos[0]['union_area_ratio']:.4f} "
            f"R={infos[1]['selected']} score={float(infos[1].get('top_score') or 0.0):.4f} area={infos[1]['union_area_ratio']:.4f} "
            f"stereo={infos[0].get('stereo_mode', '-')} "
            f"cand=L{infos[0].get('left_candidates', '?')}/R{infos[0].get('right_candidates', '?')} "
            f"plaus=L{infos[0].get('left_plausible', '?')}/R{infos[0].get('right_plausible', '?')}"
            + (f" scene_cut=1 dist={scene_distance:.3f}" if scene_cut else ""),
            flush=True,
        )
    gap_fill_frames = int(getattr(args, "ywes_gap_fill_frames", max_segment_frames) or 0)
    filled_gaps = _fill_short_inactive_gaps(records, masks_by_start, gap_fill_frames)
    if filled_gaps:
        print(
            f"[offline] YOLO-World+EfficientSAM filled short inactive gaps frames={filled_gaps} "
            f"max_gap_frames={gap_fill_frames}",
            flush=True,
        )
    starts = _planned_starts(
        records,
        target,
        max_segment_frames,
        min_segment_frames,
        bool(getattr(args, "ywes_cut_on_count_change", True)),
        bool(getattr(args, "ywes_cut_every_active_sample", False)),
        scene_min_frames,
    )
    scene_cut_frames = [int(record["frame"]) for record in records if record.get("scene_cut")]
    if scene_cut_frames:
        scene_with_masks = [frame for frame in scene_cut_frames if frame in masks_by_start]
        scene_included = [frame for frame in scene_with_masks if frame in starts]
        scene_ignored = [frame for frame in scene_cut_frames if frame not in scene_included]
        print(
            f"[offline] MatAnyone2 scene cuts detected={scene_cut_frames} "
            f"included={scene_included} ignored={scene_ignored}"
        )
    masks_by_start = {start: masks_by_start[start] for start in starts if start in masks_by_start}
    print(f"[offline] MatAnyone2 segment plan starts={starts} active={sorted(masks_by_start)}")
    if not masks_by_start and bool(getattr(args, "ywes_fail_on_empty", True)):
        raise RuntimeError(
            "YOLO-World + EfficientSAM prepass found no active person masks. "
            "Try lowering --ywes-score-threshold, using --ywes-debug-dir, or falling back to MatAnyone2 slow/SAM3."
        )
    return masks_by_start, starts


def write_prepass_result(path: Path, masks_by_start: dict[int, list[np.ndarray]], starts: list[int]) -> None:
    payload = {
        "segment_starts": np.asarray(starts, dtype=np.int64),
        "active_starts": np.asarray(sorted(masks_by_start), dtype=np.int64),
    }
    for idx, start in enumerate(sorted(masks_by_start)):
        payload[f"mask_{idx}_left"] = masks_by_start[start][0].astype(np.float32, copy=False)
        payload[f"mask_{idx}_right"] = masks_by_start[start][1].astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def read_prepass_result(path: Path) -> tuple[dict[int, list[np.ndarray]], list[int]]:
    data = np.load(path, allow_pickle=False)
    starts = [int(x) for x in data["segment_starts"].tolist()]
    active_starts = [int(x) for x in data["active_starts"].tolist()]
    masks_by_start: dict[int, list[np.ndarray]] = {}
    for idx, start in enumerate(active_starts):
        masks_by_start[start] = [
            data[f"mask_{idx}_left"].astype(np.float32, copy=False),
            data[f"mask_{idx}_right"].astype(np.float32, copy=False),
        ]
    return masks_by_start, starts


def precompute_segment_masks_subprocess(args, src: Path, source_fps: float, fps: float, target: int):
    tmp_dir = config.ROOT / "debug_output" / "_ywes_prepass"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result_path = tmp_dir / f"ywes_prepass_{int(time.time() * 1000)}_{id(args)}.npz"
    tool_name = str(getattr(args, "_tool_name", "offline_alpha_passthrough"))
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "tool", tool_name]
    else:
        cmd = [sys.executable, str(config.ROOT / "tools" / f"{tool_name}.py")]
    cmd += [
        str(src),
        "--engine", "matanyone2_onnx",
        "--model", str(Path(args._matanyone2_model_dir).resolve()),
        "--matanyone2-size", str(args.matanyone2_size),
        "--matanyone2-batch", str(args.matanyone2_batch),
        "--matanyone2-prepass", "yoloworld_efficientsam",
        "--ywes-model-dir", str(Path(args.ywes_model_dir).resolve()),
        "--ywes-txt-feats", str(Path(args.ywes_txt_feats).resolve()),
        "--ywes-provider", str(args.ywes_provider),
        "--ywes-yolo-model", str(args.ywes_yolo_model),
        "--ywes-sam-model", str(args.ywes_sam_model),
        "--ywes-yolo-size", str(args.ywes_yolo_size),
        "--ywes-score-threshold", str(args.ywes_score_threshold),
        "--ywes-nms-threshold", str(args.ywes_nms_threshold),
        "--ywes-box-expand", str(args.ywes_box_expand),
        "--ywes-top-k", str(args.ywes_top_k),
        "--ywes-scan", str(args.ywes_scan),
        "--ywes-scan-interval-sec", str(args.ywes_scan_interval_sec),
        "--ywes-active-min-area-ratio", str(args.ywes_active_min_area_ratio),
        "--ywes-gap-fill-frames", str(args.ywes_gap_fill_frames),
        "--matanyone2-segment-frames", str(args.matanyone2_segment_frames),
        "--matanyone2-min-segment-sec", str(args.matanyone2_min_segment_sec),
        "--frames", str(target),
        "--fps", str(fps),
        "--ywes-prepass-out", str(result_path),
    ]
    if getattr(args, "ywes_debug_dir", ""):
        cmd += ["--ywes-debug-dir", str(Path(args.ywes_debug_dir).resolve())]
    if not bool(getattr(args, "ywes_cut_on_count_change", True)):
        cmd += ["--no-ywes-cut-on-count-change"]
    if bool(getattr(args, "ywes_cut_every_active_sample", False)):
        cmd += ["--ywes-cut-every-active-sample"]
    if not bool(getattr(args, "ywes_fail_on_empty", True)):
        cmd += ["--no-ywes-fail-on-empty"]
    print("[offline] YOLO-World+EfficientSAM prepass subprocess=" + subprocess.list2cmdline(cmd))
    run_hidden_streaming(cmd, check=True, exit_label="offline-ywes")
    masks_by_start, starts = read_prepass_result(result_path)
    try:
        result_path.unlink(missing_ok=True)
    except Exception:
        pass
    return masks_by_start, starts
