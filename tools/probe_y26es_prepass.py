"""Drive the YOLO26m+EfficientSAM prepass directly without MatAnyone2, to surface
real detect/plausibility/stereo decisions on actual decoded frames.

Usage:
    uv run python tools/probe_y26es_prepass.py videos/72456_3840p.mp4 \
        --duration 30 --debug-dir debug_output/y26es_real_prepass
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure runtime cache like offline tools do
import utils.gpu_runtime_cache as gpu_runtime_cache
gpu_runtime_cache.configure_gpu_runtime_cache()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from offline.yolo26m_efficientsam import Yolo26mEfficientSamMasker  # noqa: E402
from offline.decoded_frames import decoded_frame_to_bgr  # noqa: E402


def _decode_pynv_frames(src: Path, count: int):
    import PyNvVideoCodec as nvc  # noqa: F401
    from offline.decoded_frames import DecodedFrames
    dec = DecodedFrames(src)
    return dec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--debug-dir", default="debug_output/y26es_real_prepass")
    parser.add_argument("--out-size", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.video).resolve()
    debug_dir = Path(args.debug_dir).resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    masker = Yolo26mEfficientSamMasker(
        Path(config.ROOT) / "models" / "yolo26m",
        Path(config.ROOT) / "models" / "efficientsam",
        provider="cuda",
        # Use module defaults; this should now point at the fp32 model after the fix.
        sam_model="efficientsam_s.onnx",
        yolo_size=640,
        score_threshold=args.score_threshold,
        nms_threshold=0.6,
        box_expand=0.08,
        top_k=args.top_k,
        binarize_mask=True,
        mask_erode_px=1,
    )

    # Sample N evenly-spaced timestamps across [0, duration]
    sample_times = np.linspace(0.0, args.duration, args.frames)
    out_size = (int(args.out_size), int(args.out_size))

    import subprocess
    import json
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(src)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    sw, sh = int(stream["width"]), int(stream["height"])
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    fps = num / den if den else 25.0
    print(f"[probe] source {sw}x{sh} fps={fps:.3f}")

    for t in sample_times:
        # Decode frame via ffmpeg (matches what the SBS path would see)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{float(t):.3f}", "-i", str(src),
            "-frames:v", "1",
            "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-",
        ]
        raw = subprocess.check_output(cmd)
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(sh, sw, 3)
        half = sw // 2
        eyes = [rgb[:, :half], rgb[:, half:half * 2]]

        t0 = time.perf_counter()
        left_dets, right_dets, stereo_info = masker.select_stereo_detections(eyes[0], eyes[1])
        sel_dt = (time.perf_counter() - t0) * 1000.0

        # All raw detections (top-8) regardless of plausibility, for diagnosis
        raw_left = masker.detect(eyes[0], top_k=8)
        raw_right = masker.detect(eyes[1], top_k=8)

        print(f"\n=== t={t:6.2f}s ===")
        print(f"  raw detect L={len(raw_left)} R={len(raw_right)}")
        for tag, dets in (("L", raw_left), ("R", raw_right)):
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d.box_xyxy.tolist()
                bw, bh = x2 - x1, y2 - y1
                area = (bw * bh) / (half * sh)
                aspect = bw / max(1.0, bh)
                ht = bh / sh
                print(f"    {tag}[{i}] score={d.score:.4f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                      f"area={area:.4f} aspect={aspect:.3f} h={ht:.3f}")
        print(f"  stereo: mode={stereo_info['stereo_mode']} cand=L{stereo_info['left_candidates']}/R{stereo_info['right_candidates']} "
              f"plaus=L{stereo_info['left_plausible']}/R{stereo_info['right_plausible']} took={sel_dt:.1f}ms")
        # Compute masks
        for eye_idx, eye_rgb in enumerate(eyes):
            dets = (left_dets, right_dets)[eye_idx]
            mask, info = masker.mask(eye_rgb, out_size, dets)
            eye_name = "left" if eye_idx == 0 else "right"
            area_ratio = info["union_area_ratio"]
            print(f"  {eye_name} count={info['count']} top_score={info['top_score']:.4f} area_ratio={area_ratio:.4f}")
            # Save debug images
            small = cv2.resize(eye_rgb, out_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(debug_dir / f"t{t:06.2f}_{eye_name}_frame.png"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(debug_dir / f"t{t:06.2f}_{eye_name}_mask.png"), (mask * 255).astype(np.uint8))
            # Overlay
            overlay = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            mask_color = np.zeros_like(overlay); mask_color[..., 1] = (mask * 255).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.5, 0)
            cv2.imwrite(str(debug_dir / f"t{t:06.2f}_{eye_name}_overlay.png"), overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
