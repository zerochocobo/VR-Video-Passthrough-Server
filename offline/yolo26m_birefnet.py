"""YOLO26m + BiRefNet prepass helpers for MatAnyone2 offline mode."""
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

BirefnetMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
BirefnetStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


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


class Yolo26mBiRefNetMasker:
    def __init__(
        self,
        model_dir: Path,
        birefnet_model_dir: Path,
        provider: str = "cuda",
        yolo_model: str = "yolo26m_model.onnx",
        birefnet_model: str = "model_fp16.onnx",
        yolo_size: int = 640,
        birefnet_input_size: int = 1024,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.6,
        box_expand: float = 0.08,
        top_k: int = 0,
        person_class_id: int = 0,
        binarize_mask: bool = True,
        mask_erode_px: int = 1,
        max_box_area: float = 0.50,
        cross_eye_area_ratio: float = 1.5,
    ) -> None:
        self.model_dir = model_dir
        self.birefnet_model_dir = birefnet_model_dir
        self.yolo_size = int(yolo_size)
        self.birefnet_input_size = int(birefnet_input_size)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.box_expand = float(box_expand)
        # top_k = 0 (or negative) means unlimited: keep all detections that pass
        # plausibility/score filters. Treated as a sentinel throughout the class.
        self.top_k = max(0, int(top_k))
        self.person_class_id = int(person_class_id)
        self.binarize_mask = bool(binarize_mask)
        self.mask_erode_px = max(0, int(mask_erode_px))
        self.max_box_area = float(max_box_area)
        self.cross_eye_area_ratio = float(cross_eye_area_ratio)
        providers = onnx_providers(provider)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.yolo = ort.InferenceSession(str(model_dir / yolo_model), sess_options=opts, providers=providers)
        self.birefnet = ort.InferenceSession(str(birefnet_model_dir / birefnet_model), sess_options=opts, providers=providers)
        self.birefnet_input_name = self.birefnet.get_inputs()[0].name
        print(
            f"[offline] YOLO26m+BiRefNet loaded dir={model_dir} birefnet_dir={birefnet_model_dir} "
            f"yolo={yolo_model} birefnet={birefnet_model} provider={providers} "
            f"yolo_size={self.yolo_size} birefnet_size={self.birefnet_input_size} score={self.score_threshold:g} "
            f"nms={self.nms_threshold:g} expand={self.box_expand:g} top_k={self.top_k} "
            f"person_class={self.person_class_id} binarize={int(self.binarize_mask)} erode={self.mask_erode_px} "
            f"max_area={self.max_box_area:g} cross_eye_ratio={self.cross_eye_area_ratio:g}"
        )
        # YOLO26m fp16 ONNX silently produces near-zero person scores on ORT
        # CUDAExecutionProvider (verified 2026-05-27 against yolo26m_model_fp16.onnx).
        # Run a quick synthetic-image probe on init so we fail loudly rather
        # than silently emitting empty masks for the whole video.
        active_providers = self.yolo.get_providers() if hasattr(self.yolo, "get_providers") else providers
        if "CUDAExecutionProvider" in active_providers and "fp16" in yolo_model.lower():
            self._sanity_check_yolo_outputs(yolo_model)

    def _sanity_check_yolo_outputs(self, yolo_model: str) -> None:
        # Feed a synthetic mid-gray image and check that the top sigmoid score
        # across all queries/classes is not pathologically low. A working
        # YOLO26m export sees max-class sigmoid > ~0.3 even on noise; a broken
        # fp16+CUDA load collapses to < 0.05 across the board.
        probe = np.full((self.yolo_size, self.yolo_size, 3), 128, dtype=np.uint8)
        chw = (probe.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        try:
            logits, _ = self.yolo.run(None, {"pixel_values": np.ascontiguousarray(chw)})[:2]
            max_sig = float(_sigmoid(np.asarray(logits)[0]).max())
        except Exception as exc:
            print(f"[offline] WARNING YOLO26m sanity probe failed: {exc}", flush=True)
            return
        if max_sig < 0.05:
            print(
                f"[offline] WARNING YOLO26m fp16 export appears broken on CUDA "
                f"(max sigmoid={max_sig:.4f} on synthetic input; healthy >= 0.3). "
                f"Switch --y26br-yolo-model to yolo26m_model.onnx (fp32) or run on CPU.",
                flush=True,
            )

    def detect(self, image_rgb: np.ndarray, top_k: int | None = None) -> list[Detection]:
        h, w = image_rgb.shape[:2]
        inp, scale, pad_x, pad_y = _preprocess_yolo(image_rgb, self.yolo_size)
        logits, pred_boxes = self.yolo.run(None, {"pixel_values": inp})[:2]
        logits = np.asarray(logits)[0].astype(np.float32, copy=False)
        pred_boxes = np.asarray(pred_boxes)[0].astype(np.float32, copy=False)
        if logits.ndim != 2 or pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
            raise RuntimeError(f"unexpected YOLO26m output shapes logits={logits.shape} pred_boxes={pred_boxes.shape}")
        if self.person_class_id < 0 or self.person_class_id >= logits.shape[1]:
            raise RuntimeError(f"person_class_id={self.person_class_id} out of range for logits shape {logits.shape}")
        person_scores = _sigmoid(logits[:, self.person_class_id])
        keep = person_scores >= self.score_threshold
        if not np.any(keep):
            return []
        boxes_norm = pred_boxes[keep]
        scores = person_scores[keep]
        cx = boxes_norm[:, 0] * self.yolo_size
        cy = boxes_norm[:, 1] * self.yolo_size
        bw = boxes_norm[:, 2] * self.yolo_size
        bh = boxes_norm[:, 3] * self.yolo_size
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
        # top_k semantics: explicit positive int = cap; None = use class default;
        # 0 or negative = unlimited (return everything that survives NMS).
        effective = self.top_k if top_k is None else int(top_k)
        nms_keep = _nms(boxes, scores, self.nms_threshold)
        keep_indices = nms_keep if effective <= 0 else nms_keep[:effective]
        return [Detection(boxes[i], float(scores[i]), self.person_class_id) for i in keep_indices]

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
            0.005 <= stats["area"] <= self.max_box_area
            and 0.10 <= stats["aspect"] <= 2.5
            and stats["height"] >= 0.10
            and det.score >= 0.45
        )

    def _is_within_max_box_area(self, det: Detection, image_shape: tuple[int, int, int]) -> bool:
        stats = self._box_stats(det, image_shape)
        return stats["area"] <= self.max_box_area

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

    def _pair_cost(
        self,
        ldet: "Detection",
        left_shape: tuple[int, int, int],
        rdet: "Detection",
        right_shape: tuple[int, int, int],
    ) -> float:
        ls = self._box_stats(ldet, left_shape)
        rs = self._box_stats(rdet, right_shape)
        geom_cost = (
            abs(ls["cy"] - rs["cy"]) * 4.0
            + abs(ls["height"] - rs["height"]) * 2.0
            + abs(ls["aspect"] - rs["aspect"]) * 1.0
            + abs(ls["area"] - rs["area"]) * 2.0
        )
        score_bonus = (ldet.score + rdet.score) * 4.0
        return geom_cost - score_bonus

    def select_stereo_detections(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        candidate_k: int = 8,
    ) -> tuple[list[Detection], list[Detection], dict]:
        # When top_k == 0 we treat the call as "unlimited": pull every NMS
        # survivor through detect(), and let plausibility / cross-eye logic
        # decide how many pairs to keep. Otherwise we keep the old behavior of
        # asking detect() for at least `candidate_k` (8 by default) candidates
        # so the pairing inner loop has options to choose from.
        detect_limit = 0 if self.top_k <= 0 else max(candidate_k, self.top_k)
        left_all = self.detect(left_rgb, top_k=detect_limit)
        right_all = self.detect(right_rgb, top_k=detect_limit)
        left = [det for det in left_all if self._is_plausible_person_box(det, left_rgb.shape)]
        right = [det for det in right_all if self._is_plausible_person_box(det, right_rgb.shape)]

        left_sel: list[Detection] = []
        right_sel: list[Detection] = []
        projection_dirs: list[str] = []  # "l2r" / "r2l" per pair, "none" if untouched
        mode = "no_detection"

        # Compute how many pairs we are willing to keep. In unlimited mode this
        # is min(len(left), len(right)) — at most one pair per person visible in
        # both eyes. In capped mode it is self.top_k as before.
        pair_limit = self.top_k if self.top_k > 0 else min(len(left), len(right))

        if left and right:
            # Greedy multi-pair selection: at each step pick the (l, r) pair
            # with the lowest cost among unused candidates. Each candidate can
            # only be used once. Stops at pair_limit pairs or when no candidates
            # remain on either side.
            used_l: set[int] = set()
            used_r: set[int] = set()
            for _ in range(pair_limit):
                best_li = best_ri = -1
                best_cost = float("inf")
                for li, ldet in enumerate(left):
                    if li in used_l:
                        continue
                    for ri, rdet in enumerate(right):
                        if ri in used_r:
                            continue
                        cost = self._pair_cost(ldet, left_rgb.shape, rdet, right_rgb.shape)
                        if cost < best_cost:
                            best_cost = cost
                            best_li, best_ri = li, ri
                if best_li < 0:
                    break
                used_l.add(best_li)
                used_r.add(best_ri)
                ldet = left[best_li]
                rdet = right[best_ri]
                # Per-pair cross-eye size sanity: when L/R box areas diverge by
                # more than `cross_eye_area_ratio`, project the higher-score
                # side's box onto the other eye. Keeps left/right masks
                # symmetric even if YOLO26m@640 produced uneven boxes for
                # this individual.
                ls = self._box_stats(ldet, left_rgb.shape)
                rs = self._box_stats(rdet, right_rgb.shape)
                min_area = max(min(ls["area"], rs["area"]), 1e-6)
                area_ratio = max(ls["area"], rs["area"]) / min_area
                if area_ratio > self.cross_eye_area_ratio:
                    if ldet.score >= rdet.score:
                        rdet = self._project_detection(ldet, left_rgb.shape, right_rgb.shape)
                        projection_dirs.append("l2r")
                    else:
                        ldet = self._project_detection(rdet, right_rgb.shape, left_rgb.shape)
                        projection_dirs.append("r2l")
                else:
                    projection_dirs.append("none")
                left_sel.append(ldet)
                right_sel.append(rdet)
            # Asymmetric leftover projection (unlimited mode only): if one eye
            # had more plausible candidates than the other (e.g., a person at
            # the edge is occluded in the other eye), project the leftover
            # candidates onto the missing side so they still contribute to the
            # bootstrap mask. Capped mode keeps the prior "drop leftovers"
            # behavior so top_k=N still returns exactly N pairs.
            if self.top_k <= 0:
                for li, ldet in enumerate(left):
                    if li in used_l:
                        continue
                    left_sel.append(ldet)
                    right_sel.append(self._project_detection(ldet, left_rgb.shape, right_rgb.shape))
                    projection_dirs.append("l2r")
                for ri, rdet in enumerate(right):
                    if ri in used_r:
                        continue
                    left_sel.append(self._project_detection(rdet, right_rgb.shape, left_rgb.shape))
                    right_sel.append(rdet)
                    projection_dirs.append("r2l")
            mode = self._stereo_paired_mode(projection_dirs)
        elif left:
            # Right eye has no plausible candidate; project every plausible L
            # to R (or up to top_k in capped mode).
            single_limit = len(left) if self.top_k <= 0 else self.top_k
            for ldet in left[:single_limit]:
                left_sel.append(ldet)
                right_sel.append(self._project_detection(ldet, left_rgb.shape, right_rgb.shape))
            mode = "project_left_to_right"
        elif right:
            single_limit = len(right) if self.top_k <= 0 else self.top_k
            for rdet in right[:single_limit]:
                right_sel.append(rdet)
                left_sel.append(self._project_detection(rdet, right_rgb.shape, left_rgb.shape))
            mode = "project_right_to_left"
        else:
            # Fallback: no eye produced a plausibility-passing detection. Still
            # guard against "fill the frame" false positives (YOLO26m sometimes
            # returns area>0.55 boxes that pass score threshold) by enforcing
            # the same max_box_area cap as plausibility.
            fallback_min_score = max(self.score_threshold * 1.5, 0.45)
            left_fallback = [
                d for d in left_all
                if d.score >= fallback_min_score
                and self._is_within_max_box_area(d, left_rgb.shape)
            ]
            right_fallback = [
                d for d in right_all
                if d.score >= fallback_min_score
                and self._is_within_max_box_area(d, right_rgb.shape)
            ]
            fb_limit_l = len(left_fallback) if self.top_k <= 0 else self.top_k
            fb_limit_r = len(right_fallback) if self.top_k <= 0 else self.top_k
            left_sel = left_fallback[:fb_limit_l] if left_fallback else []
            right_sel = right_fallback[:fb_limit_r] if right_fallback else []
            mode = "fallback_score_gate" if (left_sel or right_sel) else "no_detection"

        return left_sel, right_sel, {
            "stereo_mode": mode,
            "left_candidates": len(left_all),
            "right_candidates": len(right_all),
            "left_plausible": len(left),
            "right_plausible": len(right),
            "pairs": max(len(left_sel), len(right_sel)),
            "projection_dirs": projection_dirs,
        }

    @staticmethod
    def _stereo_paired_mode(projection_dirs: list[str]) -> str:
        """Map per-pair projection directions to a single mode label.

        - All "none" -> "paired"
        - Single pair with projection -> "paired_project_l2r" / "paired_project_r2l"
          (preserves the labels used by the previous single-pair pipeline)
        - Multiple pairs with at least one projection -> "paired_project_mixed"
        """
        non_none = [d for d in projection_dirs if d != "none"]
        if not non_none:
            return "paired"
        if len(projection_dirs) == 1:
            return f"paired_project_{non_none[0]}"
        if len(set(non_none)) == 1 and len(non_none) == len(projection_dirs):
            return f"paired_project_{non_none[0]}"
        return "paired_project_mixed"

    def _birefnet_mask_for_box(self, image_rgb: np.ndarray, box_xyxy: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        out_w, out_h = out_size
        src_h, src_w = image_rgb.shape[:2]
        x1, y1, x2, y2 = box_xyxy.astype(np.float32)
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * self.box_expand
        x2 += bw * self.box_expand
        y1 -= bh * self.box_expand
        y2 += bh * self.box_expand
        src_x1 = int(np.floor(np.clip(x1, 0, max(0, src_w - 1))))
        src_y1 = int(np.floor(np.clip(y1, 0, max(0, src_h - 1))))
        src_x2 = int(np.ceil(np.clip(x2, src_x1 + 1, src_w)))
        src_y2 = int(np.ceil(np.clip(y2, src_y1 + 1, src_h)))
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return np.zeros((out_h, out_w), dtype=np.float32)

        crop = image_rgb[src_y1:src_y2, src_x1:src_x2]
        model_size = max(1, self.birefnet_input_size)
        resized = cv2.resize(crop, (model_size, model_size), interpolation=cv2.INTER_AREA)
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        chw = (chw - BirefnetMean) / BirefnetStd
        logits, *_ = self.birefnet.run(None, {self.birefnet_input_name: np.ascontiguousarray(chw[None])})
        mask = np.asarray(logits).reshape(-1, model_size, model_size)[0].astype(np.float32, copy=False)
        if mask.min() < 0.0 or mask.max() > 1.0:
            mask = _sigmoid(mask)
        mask = np.clip(mask, 0.0, 1.0)

        out_x1 = int(np.floor(np.clip(src_x1 * out_w / max(1, src_w), 0, max(0, out_w - 1))))
        out_y1 = int(np.floor(np.clip(src_y1 * out_h / max(1, src_h), 0, max(0, out_h - 1))))
        out_x2 = int(np.ceil(np.clip(src_x2 * out_w / max(1, src_w), out_x1 + 1, out_w)))
        out_y2 = int(np.ceil(np.clip(src_y2 * out_h / max(1, src_h), out_y1 + 1, out_h)))
        if out_x2 <= out_x1 or out_y2 <= out_y1:
            return np.zeros((out_h, out_w), dtype=np.float32)

        mask = cv2.resize(mask, (out_x2 - out_x1, out_y2 - out_y1), interpolation=cv2.INTER_LINEAR)
        if self.binarize_mask:
            bin_mask = (mask >= 0.5).astype(np.uint8)
            if self.mask_erode_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                bin_mask = cv2.erode(bin_mask, kernel, iterations=self.mask_erode_px)
            mask = bin_mask.astype(np.float32)
        full = np.zeros((out_h, out_w), dtype=np.float32)
        full[out_y1:out_y2, out_x1:out_x2] = mask.astype(np.float32, copy=False)
        return full

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
        masks = [self._birefnet_mask_for_box(image_rgb, det.box_xyxy, out_size) for det in detections]
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
    fill_boundaries: bool = True,
    respect_scene_cuts: bool = True,
) -> list[int]:
    """Fill inactive scan points with neighboring masks.

    Three operations, in this order:

    1. Middle gaps: for an inactive run between two active samples, copy from
       the closer side. If `respect_scene_cuts`, inactive frames whose record
       carries `scene_cut=True` (or anything after the first scene cut inside
       the run) get the post-cut neighbor instead of the pre-cut one.
    2. Start boundary: if the first active sample is preceded by inactive
       samples, copy its mask backward to frame 0. Skipped if the first active
       record carries `scene_cut=True` (the pre-cut scene is genuinely
       different) or if the boundary distance exceeds `max_gap_frames`.
    3. End boundary: symmetric to (2), copy the last active sample's mask
       forward to the tail.

    Returns the list of frames whose masks were newly populated.
    """
    if len(records) == 0:
        return []
    filled: list[int] = []

    # --- 1. Middle inactive runs --------------------------------------------
    if max_gap_frames > 0 and len(records) >= 3:
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
            # Index of the first scene_cut record inside this inactive run.
            scene_cut_idx: int | None = None
            if respect_scene_cuts:
                for k in range(run_start, run_end):
                    if records[k].get("scene_cut"):
                        scene_cut_idx = k
                        break
            for j in range(run_start, run_end):
                frame = int(records[j]["frame"])
                if scene_cut_idx is not None and j >= scene_cut_idx:
                    source_masks = next_masks if next_masks is not None else prev_masks
                elif scene_cut_idx is not None and j < scene_cut_idx:
                    source_masks = prev_masks if prev_masks is not None else next_masks
                else:
                    source_masks = prev_masks if (next_masks is None or frame - prev_frame <= next_frame - frame) else next_masks
                if source_masks is None:
                    continue
                masks_by_start[frame] = [mask.copy() for mask in source_masks]
                records[j]["active"] = True
                records[j]["object_count"] = object_count
                records[j]["gap_filled"] = True
                filled.append(frame)

    # --- 2 + 3. Boundary fills ----------------------------------------------
    if fill_boundaries and max_gap_frames > 0:
        active_indices = [k for k, r in enumerate(records) if r.get("active")]
        if active_indices:
            # Start boundary: from frame 0 to first active.
            first_active = active_indices[0]
            if first_active > 0:
                anchor = records[first_active]
                anchor_frame = int(anchor["frame"])
                # Respect scene cut: if the first active sample IS a scene
                # cut, the pre-cut content is different. Skip backward fill.
                blocked_by_scene = bool(respect_scene_cuts and anchor.get("scene_cut"))
                # Cap how far back we fill so we don't paper over a long intro.
                within_window = anchor_frame <= max_gap_frames
                anchor_masks = masks_by_start.get(anchor_frame)
                if anchor_masks is not None and not blocked_by_scene and within_window:
                    object_count = max(int(anchor.get("object_count") or 0), 1)
                    for j in range(0, first_active):
                        if records[j].get("active"):
                            continue
                        frame = int(records[j]["frame"])
                        masks_by_start[frame] = [mask.copy() for mask in anchor_masks]
                        records[j]["active"] = True
                        records[j]["object_count"] = object_count
                        records[j]["gap_filled"] = True
                        records[j]["boundary_filled"] = "start"
                        filled.append(frame)
            # End boundary: from last active to last sample.
            last_active = active_indices[-1]
            if last_active < len(records) - 1:
                anchor = records[last_active]
                anchor_frame = int(anchor["frame"])
                blocked_by_scene = False
                if respect_scene_cuts:
                    for k in range(last_active + 1, len(records)):
                        if records[k].get("scene_cut"):
                            blocked_by_scene = True
                            break
                tail_frame = int(records[-1]["frame"])
                within_window = (tail_frame - anchor_frame) <= max_gap_frames
                anchor_masks = masks_by_start.get(anchor_frame)
                if anchor_masks is not None and not blocked_by_scene and within_window:
                    object_count = max(int(anchor.get("object_count") or 0), 1)
                    for j in range(last_active + 1, len(records)):
                        if records[j].get("active"):
                            continue
                        frame = int(records[j]["frame"])
                        masks_by_start[frame] = [mask.copy() for mask in anchor_masks]
                        records[j]["active"] = True
                        records[j]["object_count"] = object_count
                        records[j]["gap_filled"] = True
                        records[j]["boundary_filled"] = "end"
                        filled.append(frame)

    return filled


def sample_points(args, src: Path, source_fps: float, fps: float, target: int) -> list[int]:
    scan = str(getattr(args, "y26br_scan", "hybrid"))
    interval = float(getattr(args, "y26br_scan_interval_sec", 1.0))
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
    if bool(getattr(args, "y26br_subprocess", True)) and not bool(getattr(args, "_y26br_child", False)):
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
    masker = Yolo26mBiRefNetMasker(
        Path(getattr(args, "y26br_model_dir", config.ROOT / "models" / "yolo26m")).resolve(),
        Path(getattr(args, "y26br_birefnet_model_dir", config.ROOT / "models" / "BiRefNet")).resolve(),
        provider=str(getattr(args, "y26br_provider", "cuda")),
        yolo_model=str(getattr(args, "y26br_yolo_model", "yolo26m_model.onnx")),
        birefnet_model=str(getattr(args, "y26br_birefnet_model", "model_fp16.onnx")),
        yolo_size=int(getattr(args, "y26br_yolo_size", 640)),
        birefnet_input_size=int(getattr(args, "y26br_birefnet_input_size", 1024)),
        score_threshold=float(getattr(args, "y26br_score_threshold", 0.35)),
        nms_threshold=float(getattr(args, "y26br_nms_threshold", 0.6)),
        box_expand=float(getattr(args, "y26br_box_expand", 0.08)),
        top_k=int(getattr(args, "y26br_top_k", 0)),
        binarize_mask=bool(getattr(args, "y26br_binarize_mask", True)),
        mask_erode_px=int(getattr(args, "y26br_mask_erode_px", 1)),
        max_box_area=float(getattr(args, "y26br_max_box_area", 0.50)),
        cross_eye_area_ratio=float(getattr(args, "y26br_cross_eye_area_ratio", 1.5)),
    )
    debug_dir = Path(args.y26br_debug_dir).resolve() if getattr(args, "y26br_debug_dir", "") else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[offline] YOLO26m+BiRefNet prepass samples={len(scan_points)} "
        f"scan={getattr(args, 'y26br_scan', 'hybrid')} max_segment_frames={max_segment_frames} "
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
            infos[0]["union_area_ratio"] >= float(getattr(args, "y26br_active_min_area_ratio", 0.001))
            or infos[1]["union_area_ratio"] >= float(getattr(args, "y26br_active_min_area_ratio", 0.001))
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
            f"[offline] Y26+BR prepass {n}/{len(scan_points)} frame={start} src_idx={src_idx} "
            f"ms={(time.perf_counter() - t0) * 1000:.1f} active={active} objects={object_count} "
            f"L={infos[0]['selected']} score={float(infos[0].get('top_score') or 0.0):.4f} area={infos[0]['union_area_ratio']:.4f} "
            f"R={infos[1]['selected']} score={float(infos[1].get('top_score') or 0.0):.4f} area={infos[1]['union_area_ratio']:.4f} "
            f"stereo={infos[0].get('stereo_mode', '-')} "
            f"cand=L{infos[0].get('left_candidates', '?')}/R{infos[0].get('right_candidates', '?')} "
            f"plaus=L{infos[0].get('left_plausible', '?')}/R{infos[0].get('right_plausible', '?')}"
            + (f" scene_cut=1 dist={scene_distance:.3f}" if scene_cut else ""),
            flush=True,
        )
    gap_fill_frames = int(getattr(args, "y26br_gap_fill_frames", max_segment_frames) or 0)
    fill_boundaries = bool(getattr(args, "y26br_fill_boundaries", True))
    scene_aware_fill = bool(getattr(args, "y26br_scene_aware_fill", True))
    filled_gaps = _fill_short_inactive_gaps(
        records,
        masks_by_start,
        gap_fill_frames,
        fill_boundaries=fill_boundaries,
        respect_scene_cuts=scene_aware_fill,
    )
    if filled_gaps:
        print(
            f"[offline] YOLO26m+BiRefNet filled short inactive gaps frames={filled_gaps} "
            f"max_gap_frames={gap_fill_frames} boundary={int(fill_boundaries)} "
            f"scene_aware={int(scene_aware_fill)}",
            flush=True,
        )
    starts = _planned_starts(
        records,
        target,
        max_segment_frames,
        min_segment_frames,
        bool(getattr(args, "y26br_cut_on_count_change", True)),
        bool(getattr(args, "y26br_cut_every_active_sample", False)),
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
    if not masks_by_start and bool(getattr(args, "y26br_fail_on_empty", True)):
        raise RuntimeError(
            "YOLO26m + BiRefNet prepass found no active person masks. "
            "Try lowering --y26br-score-threshold, using --y26br-debug-dir, or falling back to MatAnyone2 slow/SAM3."
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
    tmp_dir = config.ROOT / "debug_output" / "_y26br_prepass"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result_path = tmp_dir / f"y26br_prepass_{int(time.time() * 1000)}_{id(args)}.npz"
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
        "--matanyone2-prepass", "yolo26m_birefnet",
        "--y26br-model-dir", str(Path(args.y26br_model_dir).resolve()),
        "--y26br-birefnet-model-dir", str(Path(args.y26br_birefnet_model_dir).resolve()),
        "--y26br-provider", str(args.y26br_provider),
        "--y26br-yolo-model", str(args.y26br_yolo_model),
        "--y26br-birefnet-model", str(args.y26br_birefnet_model),
        "--y26br-yolo-size", str(args.y26br_yolo_size),
        "--y26br-birefnet-input-size", str(args.y26br_birefnet_input_size),
        "--y26br-score-threshold", str(args.y26br_score_threshold),
        "--y26br-nms-threshold", str(args.y26br_nms_threshold),
        "--y26br-box-expand", str(args.y26br_box_expand),
        "--y26br-top-k", str(args.y26br_top_k),
        "--y26br-mask-erode-px", str(args.y26br_mask_erode_px),
        "--y26br-max-box-area", str(args.y26br_max_box_area),
        "--y26br-cross-eye-area-ratio", str(args.y26br_cross_eye_area_ratio),
        "--y26br-scan", str(args.y26br_scan),
        "--y26br-scan-interval-sec", str(args.y26br_scan_interval_sec),
        "--y26br-active-min-area-ratio", str(args.y26br_active_min_area_ratio),
        "--y26br-gap-fill-frames", str(args.y26br_gap_fill_frames),
        "--y26br-fill-boundaries" if bool(getattr(args, "y26br_fill_boundaries", True)) else "--no-y26br-fill-boundaries",
        "--y26br-scene-aware-fill" if bool(getattr(args, "y26br_scene_aware_fill", True)) else "--no-y26br-scene-aware-fill",
        "--matanyone2-segment-frames", str(args.matanyone2_segment_frames),
        "--matanyone2-min-segment-sec", str(args.matanyone2_min_segment_sec),
        "--frames", str(target),
        "--fps", str(fps),
        "--y26br-prepass-out", str(result_path),
    ]
    if not bool(getattr(args, "y26br_binarize_mask", True)):
        cmd += ["--no-y26br-binarize-mask"]
    if getattr(args, "y26br_debug_dir", ""):
        cmd += ["--y26br-debug-dir", str(Path(args.y26br_debug_dir).resolve())]
    if not bool(getattr(args, "y26br_cut_on_count_change", True)):
        cmd += ["--no-y26br-cut-on-count-change"]
    if bool(getattr(args, "y26br_cut_every_active_sample", False)):
        cmd += ["--y26br-cut-every-active-sample"]
    if not bool(getattr(args, "y26br_fail_on_empty", True)):
        cmd += ["--no-y26br-fail-on-empty"]
    print("[offline] YOLO26m+BiRefNet prepass subprocess=" + subprocess.list2cmdline(cmd))
    run_hidden_streaming(cmd, check=True, exit_label="offline-y26br")
    masks_by_start, starts = read_prepass_result(result_path)
    try:
        result_path.unlink(missing_ok=True)
    except Exception:
        pass
    return masks_by_start, starts
