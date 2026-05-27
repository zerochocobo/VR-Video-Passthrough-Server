"""Quick probe of YOLO26m ONNX behavior on a real frame.

Decode one frame from a video, split SBS L/R, run YOLO26m with different
preprocessing assumptions, and dump the top scores and boxes so we can verify
which assumption matches the model.

Usage:
    uv run python tools/probe_yolo26m_real.py videos/72456_3840p.mp4 \
        --time 0.0 --top 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent


def _decode_frame_at(path: Path, time_sec: float) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    raw = subprocess.check_output(cmd)
    if len(raw) != w * h * 3:
        raise RuntimeError(f"unexpected frame bytes {len(raw)} for {w}x{h}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)


def _letterbox(image_rgb: np.ndarray, size: int):
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _preprocess(image_rgb: np.ndarray, size: int, mode: str):
    """mode: 'div255', 'imagenet', 'div255_bgr'"""
    canvas, scale, pad_x, pad_y = _letterbox(image_rgb, size)
    if mode == "div255":
        arr = canvas.astype(np.float32) / 255.0
    elif mode == "imagenet":
        arr = canvas.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    elif mode == "div255_bgr":
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        arr = canvas_bgr.astype(np.float32) / 255.0
    else:
        raise ValueError(mode)
    chw = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(chw), scale, pad_x, pad_y


def _run_and_summarize(session, inp_arr, person_class_id, top_n: int):
    logits, pred_boxes = session.run(None, {"pixel_values": inp_arr})[:2]
    logits = logits[0]      # [300, 80]
    pred_boxes = pred_boxes[0]  # [300, 4]
    # Raw person score (logit)
    person_logits = logits[:, person_class_id]
    person_sig = _sigmoid(person_logits)
    # Also try max over all classes (in case class 0 is not person)
    max_per_query_logit = logits.max(axis=1)
    max_per_query_class = logits.argmax(axis=1)
    max_per_query_sig = _sigmoid(max_per_query_logit)

    summary = {
        "person_class_id_used": int(person_class_id),
        "person_logit_min": float(person_logits.min()),
        "person_logit_max": float(person_logits.max()),
        "person_logit_mean": float(person_logits.mean()),
        "person_sig_max": float(person_sig.max()),
        "person_sig_p99": float(np.percentile(person_sig, 99)),
        "person_sig_top": [float(x) for x in np.sort(person_sig)[::-1][:top_n]],
        "max_logit_top": [float(x) for x in np.sort(max_per_query_logit)[::-1][:top_n]],
        "max_sig_top": [float(x) for x in np.sort(max_per_query_sig)[::-1][:top_n]],
    }
    # top queries by max-any-class score
    order = np.argsort(max_per_query_sig)[::-1][:top_n]
    summary["top_queries"] = [
        {
            "query_idx": int(q),
            "class_id": int(max_per_query_class[q]),
            "max_sig": float(max_per_query_sig[q]),
            "person_sig": float(person_sig[q]),
            "box_raw": [float(v) for v in pred_boxes[q]],
        }
        for q in order
    ]
    # box value distribution sanity
    summary["pred_boxes_min"] = [float(pred_boxes[:, i].min()) for i in range(4)]
    summary["pred_boxes_max"] = [float(pred_boxes[:, i].max()) for i in range(4)]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--model", default=str(ROOT / "models" / "yolo26m" / "yolo26m_model.onnx"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--save-dir", default="")
    args = parser.parse_args()

    src = Path(args.video).resolve()
    if not src.exists():
        print(f"video not found: {src}", file=sys.stderr)
        return 1
    print(f"[probe] decode time={args.time}s from {src}")
    sbs = _decode_frame_at(src, args.time)
    h, w = sbs.shape[:2]
    half = w // 2
    left = sbs[:, :half]
    right = sbs[:, half:half * 2]
    print(f"[probe] frame {w}x{h} -> left {left.shape[1]}x{left.shape[0]} / right {right.shape[1]}x{right.shape[0]}")

    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "left.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / "right.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / "sbs.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in set(ort.get_available_providers()) else ["CPUExecutionProvider"]
    print(f"[probe] providers={providers}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.model, sess_options=opts, providers=providers)
    in_meta = session.get_inputs()[0]
    print(f"[probe] input name={in_meta.name} shape={in_meta.shape} type={in_meta.type}")
    for o in session.get_outputs():
        print(f"[probe] output name={o.name} shape={o.shape} type={o.type}")

    for eye_name, eye_img in (("left", left), ("right", right)):
        print(f"\n========== {eye_name} eye ==========")
        for preproc in ("div255", "imagenet", "div255_bgr"):
            inp_arr, scale, pad_x, pad_y = _preprocess(eye_img, 640, preproc)
            print(f"\n--- preproc={preproc} input min={inp_arr.min():.3f} max={inp_arr.max():.3f} mean={inp_arr.mean():.3f} ---")
            summary = _run_and_summarize(session, inp_arr, args.person_class_id, args.top)
            for k, v in summary.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"  {k}:")
                    for entry in v[:5]:
                        print(f"    {entry}")
                else:
                    print(f"  {k}: {v}")
            if save_dir is not None:
                payload_path = save_dir / f"{eye_name}_{preproc}.json"
                payload_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
