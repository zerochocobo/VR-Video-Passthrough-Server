"""Shared SAM3 prepass helpers for MatAnyone2 offline tools."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np


def clear_gpu_memory_pools() -> None:
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def fill_short_inactive_gaps(
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


class Sam3TextMasker:
    _KNOWN_CLIP_TOKENS = {
        "person": [2533],
    }
    _TOKENIZER = None

    def __init__(
        self,
        model_dir: Path,
        prompt: str,
        providers: list,
        decoder_providers: list | None = None,
        score_threshold: float = 0.5,
        min_area_ratio: float = 0.0005,
        max_area_ratio: float = 0.95,
        top_k: int = 0,
        low_memory: bool = True,
    ):
        import onnxruntime as ort

        self.model_dir = model_dir
        self.prompt = prompt.strip() or "person"
        self.providers = providers
        self.decoder_providers = decoder_providers or providers
        self.score_threshold = float(score_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.top_k = max(0, int(top_k))
        self.low_memory = bool(low_memory)
        self.image_encoder = None
        self.decoder = None
        if not self.low_memory:
            self.image_encoder = ort.InferenceSession(
                str(model_dir / "sam3_image_encoder.onnx"),
                providers=providers,
            )
            self.decoder = ort.InferenceSession(
                str(model_dir / "sam3_decoder.onnx"),
                providers=self.decoder_providers,
            )
        # Keep SAM3 text encoding on CPU: for short prompts it is faster in
        # local tests and avoids inflating CUDA memory before image decoding.
        language_encoder = ort.InferenceSession(
            str(model_dir / "sam3_language_encoder.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.language_mask, self.language_features, _ = language_encoder.run(
            None,
            {"tokens": self._tokenize(self.prompt)},
        )
        del language_encoder

    @staticmethod
    def _is_ort_cuda_arena_oom(exc: Exception) -> bool:
        text = str(exc)
        return (
            "BFCArena::AllocateRawInternal" in text
            or "Available memory" in text and "requested bytes" in text
            or "Failed to allocate memory" in text
        )

    def _reset_image_encoder(self) -> None:
        if self.low_memory:
            return
        del self.image_encoder
        clear_gpu_memory_pools()
        self.image_encoder = self._image_encoder_session()

    def _reset_decoder(self) -> None:
        if self.low_memory:
            return
        del self.decoder
        clear_gpu_memory_pools()
        self.decoder = self._decoder_session()

    def _image_encoder_session(self):
        import onnxruntime as ort

        return ort.InferenceSession(
            str(self.model_dir / "sam3_image_encoder.onnx"),
            providers=self.providers,
        )

    def _decoder_session(self):
        import onnxruntime as ort

        return ort.InferenceSession(
            str(self.model_dir / "sam3_decoder.onnx"),
            providers=self.decoder_providers,
        )

    def _run_image_encoder(self, sam_image):
        if self.low_memory:
            session = self._image_encoder_session()
            try:
                return session.run(None, {"image": sam_image})
            finally:
                del session
                clear_gpu_memory_pools()
        try:
            return self.image_encoder.run(None, {"image": sam_image})
        except Exception as exc:
            if not self._is_ort_cuda_arena_oom(exc):
                raise
            print("[offline] SAM3 image encoder CUDA arena exhausted; recreating encoder session and retrying once")
            self._reset_image_encoder()
            return self.image_encoder.run(None, {"image": sam_image})

    def _run_decoder(self, feed):
        if self.low_memory:
            session = self._decoder_session()
            try:
                return session.run(None, feed)
            finally:
                del session
                clear_gpu_memory_pools()
        try:
            return self.decoder.run(None, feed)
        except Exception as exc:
            if not self._is_ort_cuda_arena_oom(exc):
                raise
            print("[offline] SAM3 decoder CUDA arena exhausted; recreating decoder session and retrying once")
            self._reset_decoder()
            return self.decoder.run(None, feed)

    def prepare_image(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        import cv2

        h, w = image_rgb.shape[:2]
        sam_image = cv2.resize(image_rgb, (1008, 1008), interpolation=cv2.INTER_AREA)
        sam_image = np.ascontiguousarray(sam_image.transpose(2, 0, 1).astype(np.uint8, copy=False))
        return sam_image, (w, h)

    def encode_prepared(self, image_encoder, sam_image):
        if image_encoder is None:
            return self._run_image_encoder(sam_image)
        return image_encoder.run(None, {"image": sam_image})

    def decode_encoded(
        self,
        decoder,
        image_out,
        source_size: tuple[int, int],
        out_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict]:
        source_w, source_h = source_size
        out_w, out_h = out_size or source_size
        run_decoder = self._run_decoder if decoder is None else lambda feed: decoder.run(None, feed)
        _boxes, scores, masks = run_decoder({
            "original_height": np.array(out_h, dtype=np.int64),
            "original_width": np.array(out_w, dtype=np.int64),
            "vision_pos_enc_2": image_out[2],
            "backbone_fpn_0": image_out[3],
            "backbone_fpn_1": image_out[4],
            "backbone_fpn_2": image_out[5],
            "language_mask": self.language_mask,
            "language_features": self.language_features,
            "box_coords": np.zeros((1, 1, 4), dtype=np.float32),
            "box_labels": np.array([[1]], dtype=np.int64),
            "box_masks": np.array([[True]], dtype=np.bool_),
        })
        del image_out
        if masks.size == 0:
            raise RuntimeError("SAM3 returned no masks for text prompt")
        masks = masks[:, 0].astype(np.bool_, copy=False)
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
        if np.max(areas) <= 0:
            raise RuntimeError("SAM3 returned empty masks for text prompt")
        scores = scores.astype(np.float32, copy=False)
        area_ratios = areas / float(out_h * out_w)
        keep = (
            (scores >= self.score_threshold)
            & (area_ratios >= self.min_area_ratio)
            & (area_ratios <= self.max_area_ratio)
        )
        if not np.any(keep):
            area_weight = np.sqrt(np.maximum(areas, 1.0) / float(out_h * out_w))
            keep[int(np.argmax(scores * area_weight))] = True
        selected = np.where(keep)[0]
        if self.top_k > 0 and selected.size > self.top_k:
            order = selected[np.argsort(scores[selected])[::-1]]
            selected = order[: self.top_k]
        union = np.any(masks[selected], axis=0)
        union_area_ratio = float(union.sum() / float(out_h * out_w))
        ys, xs = np.where(union)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if xs.size else []
        info = {
            "count": int(masks.shape[0]),
            "selected": selected.astype(int).tolist(),
            "scores": [float(x) for x in scores.tolist()],
            "area_ratios": [float(x) for x in area_ratios.tolist()],
            "union_area_ratio": union_area_ratio,
            "union_bbox_xyxy": bbox,
            "source_size": [int(source_w), int(source_h)],
            "mask_size": [int(out_w), int(out_h)],
        }
        return union.astype(np.float32), info

    @classmethod
    def _tokenize(cls, text: str) -> np.ndarray:
        normalized = " ".join(text.lower().strip().split())
        try:
            from osam._models.yoloworld.clip import tokenize

            return tokenize(texts=[normalized or text], context_length=32).astype(np.int64)
        except Exception as exc:
            token_ids = cls._KNOWN_CLIP_TOKENS.get(normalized)
            if token_ids is None:
                token_ids = cls._clip_token_ids(normalized, exc)
        tokens = np.zeros((1, 32), dtype=np.int64)
        seq = [49406, *token_ids, 49407]
        if len(seq) > tokens.shape[1]:
            seq = seq[: tokens.shape[1]]
            seq[-1] = 49407
        tokens[0, :len(seq)] = seq
        return tokens

    @classmethod
    def _clip_token_ids(cls, text: str, original_exc: Exception | None = None) -> list[int]:
        if cls._TOKENIZER is None:
            try:
                from tools.generate_yoloworld_person_txt_feats import (
                    DEFAULT_BPE_PATH,
                    DEFAULT_MERGES_PATH,
                    DEFAULT_VOCAB_PATH,
                    ClipBpeTokenizer,
                )

                vocab_path, merges_path, bpe_path = cls._clip_tokenizer_paths(
                    DEFAULT_VOCAB_PATH,
                    DEFAULT_MERGES_PATH,
                    DEFAULT_BPE_PATH,
                )
                if vocab_path is not None and merges_path is not None:
                    cls._TOKENIZER = ClipBpeTokenizer.from_vocab_merges(vocab_path, merges_path)
                elif bpe_path is not None:
                    cls._TOKENIZER = ClipBpeTokenizer.from_openai_bpe(bpe_path)
                else:
                    raise FileNotFoundError(
                        f"CLIP tokenizer files not found: {DEFAULT_VOCAB_PATH}, {DEFAULT_MERGES_PATH}, or {DEFAULT_BPE_PATH}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    "SAM3 text prompt tokenization needs either osam-yoloworld or local CLIP tokenizer cache. "
                    "Run tools/generate_yoloworld_person_txt_feats.py once to prepare the tokenizer cache, "
                    f"or use the built-in prompt 'person'. prompt={text!r}"
                ) from (original_exc or exc)
        return cls._TOKENIZER.encode(text)

    @staticmethod
    def _clip_tokenizer_paths(
        default_vocab_path: Path,
        default_merges_path: Path,
        default_bpe_path: Path,
    ) -> tuple[Path | None, Path | None, Path | None]:
        roots = [default_bpe_path.parents[2]]
        if getattr(sys, "frozen", False):
            exe_root = Path(sys.executable).resolve().parent
            roots.extend([exe_root, exe_root / "_internal"])
        roots.append(Path.cwd())

        seen: set[Path] = set()
        for root in roots:
            root = root.resolve()
            if root in seen:
                continue
            seen.add(root)
            cache_dir = root / "runtime_cache" / "clip_text_onnx"
            vocab_path = cache_dir / default_vocab_path.name
            merges_path = cache_dir / default_merges_path.name
            if vocab_path.exists() and merges_path.exists():
                return vocab_path, merges_path, None
            bpe_path = cache_dir / default_bpe_path.name
            if bpe_path.exists():
                return None, None, bpe_path
        return None, None, None

    def mask(self, image_rgb: np.ndarray, out_size: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
        sam_image, source_size = self.prepare_image(image_rgb)
        image_out = self.encode_prepared(None, sam_image)
        return self.decode_encoded(None, image_out, source_size, out_size)


def empty_sam3_mask(width: int, height: int, reason: str = ""):
    return np.zeros((height, width), dtype=np.bool_), {
        "count": 0,
        "selected": [],
        "scores": [],
        "area_ratios": [],
        "union_area_ratio": 0.0,
        "union_bbox_xyxy": [],
        "empty_reason": reason,
    }


def sam3_mask_stats(mask) -> dict:
    h, w = mask.shape[:2]
    ys, xs = np.where(mask >= 0.5)
    if xs.size == 0:
        return {"active": False, "area": 0.0, "aspect": 0.0, "height": 0.0, "width": 0.0, "cy": 0.0, "bbox": []}
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return {
        "active": True,
        "area": float(xs.size / float(max(1, w * h))),
        "aspect": float(bw / float(bh)),
        "height": float(bh / float(max(1, h))),
        "width": float(bw / float(max(1, w))),
        "cy": float(((y1 + y2) * 0.5) / float(max(1, h))),
        "bbox": [x1, y1, x2, y2],
    }


def sam3_mask_plausible(stats: dict) -> bool:
    return bool(
        stats.get("active")
        and 0.001 <= float(stats["area"]) <= 0.45
        and 0.08 <= float(stats["aspect"]) <= 2.2
        and float(stats["height"]) >= 0.12
    )


def apply_sam3_stereo_guard(masks: list, infos: list[dict]) -> tuple[list, list[dict], str]:
    if len(masks) != 2 or len(infos) != 2:
        return masks, infos, "single"
    stats = [sam3_mask_stats(mask) for mask in masks]
    plausible = [sam3_mask_plausible(item) for item in stats]
    mode = "paired"
    if plausible[0] and plausible[1]:
        cy_gap = abs(float(stats[0]["cy"]) - float(stats[1]["cy"]))
        height_gap = abs(float(stats[0]["height"]) - float(stats[1]["height"]))
        area_ratio = max(float(stats[0]["area"]), float(stats[1]["area"])) / max(
            min(float(stats[0]["area"]), float(stats[1]["area"])),
            1.0e-6,
        )
        if cy_gap > 0.22 or height_gap > 0.35 or area_ratio > 3.5:
            mode = "diverged"
    elif plausible[0] and not plausible[1]:
        masks[1] = masks[0].astype(np.float32, copy=True)
        infos[1] = dict(infos[1])
        infos[1]["stereo_projected_from"] = "left"
        infos[1]["union_area_ratio"] = float((masks[1] >= 0.5).sum() / float(masks[1].size))
        infos[1]["union_bbox_xyxy"] = stats[0]["bbox"]
        mode = "project_left_to_right"
    elif plausible[1] and not plausible[0]:
        masks[0] = masks[1].astype(np.float32, copy=True)
        infos[0] = dict(infos[0])
        infos[0]["stereo_projected_from"] = "right"
        infos[0]["union_area_ratio"] = float((masks[0] >= 0.5).sum() / float(masks[0].size))
        infos[0]["union_bbox_xyxy"] = stats[1]["bbox"]
        mode = "project_right_to_left"
    else:
        mode = "fallback_unfiltered"
    for idx in range(2):
        infos[idx] = dict(infos[idx])
        infos[idx]["stereo_mode"] = mode
        infos[idx]["stereo_mask_stats"] = stats[idx]
    return masks, infos, mode
