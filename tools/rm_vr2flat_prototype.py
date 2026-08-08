"""Offline prototype: "VR-to-flat before restoration" (VR转平面后解码).

Goal: validate the *image-quality* hypothesis that de-mosaicing a VR180 region
works better if we first gnomonic-project the equirectangular region to a flat
(rectilinear) view, restore it there, then reproject + alpha-overlay back --
compared to the current path that restores directly on the stretched
equirectangular pixels.

This is a QUALITY prototype, not the production path:
  * single eye only (left half of a SBS frame), no L/R recombine
  * one representative yaw/pitch/fov for the whole clip (from the top-scoring
    mosaic box on a sample frame), no per-frame / multi-region tracking
  * reuses ffmpeg v360 (same math/rorder/alpha as VR_Video_Toolbox area_selection
    _vr2flat/logic.py) instead of a GPU kernel -- the GPU per-frame kernel is
    stage 2, only worth building once quality is confirmed.

Projection is hequirect (half-equirect, 180x180, pixel<->angle linear), matching
the reference tool. yaw/pitch/fov are derived analytically from the detected
equirect box (same formula as the reference GUI's click-to-select):
    yaw   = (ecx/eye_w)*180 - 90
    pitch = 90 - (ecy/eye_h)*180
    d_fov = max(bw/eye_w, bh/eye_h)*180 * FOV_MARGIN

Outputs (next to input, or --out-dir):
    <stem>_L_clip.mp4                 cropped left-eye clip (equirect)
    <stem>_L_direct_restored.mp4      current path: RM directly on equirect
    <stem>_L_flat.mp4                 gnomonic flat view (equirect->flat)
    <stem>_L_flat_restored.mp4        RM on the flat view
    <stem>_L_vr2flat_restored.mp4     flat restored -> reprojected + overlaid

Compare *_L_direct_restored.mp4 (baseline) vs *_L_vr2flat_restored.mp4 (new).

    uv run python tools/rm_vr2flat_prototype.py <vr180_sbs_input>.mp4 \
        --start 240 --duration 20
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from offline import demosaic_offline

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

FOV_MARGIN = 1.4          # enlarge fov so the mosaic sits inside the flat view w/ context
FLAT_OVERSAMPLE = 1.5     # flat resolution vs box pixel size (rectilinear stretches edges)
V360_YPR = "rorder=ypr"   # forward: equirect -> flat
V360_RPY = "rorder=rpy"   # inverse: flat -> equirect (reversed order + negated angles)


def log(m: str) -> None:
    print(m, flush=True)


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def probe_wh(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        text=True).strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def crop_left_eye_clip(src: Path, dst: Path, start: float, duration: float) -> None:
    """Cut [start, start+duration) and keep the left half (one hequirect eye)."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-stats", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src),
            "-vf", "crop=iw/2:ih:0:0",
            "-c:a", "copy",
            "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
            "-pix_fmt", "p010le", "-color_range", "tv",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            str(dst)]
    run(cmd)


def sample_frame_rgb(clip: Path, at_sec: float) -> np.ndarray:
    tmp = clip.with_suffix(".sample.png")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{at_sec:.3f}", "-i", str(clip), "-frames:v", "1", str(tmp)])
    bgr = cv2.imread(str(tmp))
    tmp.unlink(missing_ok=True)
    if bgr is None:
        raise RuntimeError(f"failed to sample a frame from {clip}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def auto_view_from_detection(rgb: np.ndarray) -> tuple[float, float, float, int, tuple]:
    """Return (yaw, pitch, d_fov, flat_side, box) from the top-scoring mosaic box.

    rgb is a single hequirect eye (eye_w x eye_h). hequirect is 180x180 so pixel
    coords map linearly to angles."""
    from pipeline.demosaic import DemosaicDetector

    eye_h, eye_w = rgb.shape[:2]
    det = DemosaicDetector("cuda")
    dets, _protos = det.detect(rgb, conf=config.RM_CONF)
    if not dets:
        raise RuntimeError("no mosaic region detected on the sample frame")
    best = max(dets, key=lambda d: d.score)
    x1, y1, x2, y2 = best.box
    bw, bh = x2 - x1, y2 - y1
    ecx, ecy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    yaw = (ecx / eye_w) * 180.0 - 90.0
    pitch = 90.0 - (ecy / eye_h) * 180.0
    d_fov = min(160.0, max(bw / eye_w, bh / eye_h) * 180.0 * FOV_MARGIN)
    flat_side = int(round(max(bw, bh) * FLAT_OVERSAMPLE / 10.0)) * 10
    flat_side = max(320, flat_side)
    log(f"[view] box={best.box} score={best.score:.3f} bw={bw} bh={bh} "
        f"-> yaw={yaw:.2f} pitch={pitch:.2f} d_fov={d_fov:.2f} flat={flat_side}")
    return yaw, pitch, d_fov, flat_side, best.box


def equirect_to_flat(src_eye: Path, dst_flat: Path, yaw: float, pitch: float,
                     d_fov: float, flat_side: int) -> None:
    vf = (f"scale=in_color_matrix=bt709,format=yuv420p10le,"
          f"v360=hequirect:flat:d_fov={d_fov}:yaw={yaw}:pitch={pitch}"
          f":w={flat_side}:h={flat_side}:{V360_YPR},"
          f"scale=out_color_matrix=bt709:out_range=limited,format=yuv420p10le")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-stats", "-y",
         "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", str(src_eye),
         "-vf", vf, "-c:a", "copy",
         "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
         "-pix_fmt", "p010le", "-color_range", "tv",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         str(dst_flat)])


def flat_to_equirect_overlay(background_eye: Path, flat_restored: Path, dst: Path,
                             yaw: float, pitch: float, d_fov: float,
                             eye_w: int, eye_h: int) -> None:
    """Reproject the restored flat patch back to hequirect and alpha-overlay it
    onto the original eye. Inverse uses negated angles + reversed rorder (rpy)."""
    fc = (f"[1:v]scale=in_color_matrix=bt709,format=yuv420p10le,"
          f"v360=input=flat:output=hequirect:w={eye_w}:h={eye_h}:id_fov={d_fov}"
          f":yaw={-yaw}:pitch={-pitch}:{V360_RPY}:alpha_mask=1,"
          f"scale=out_color_matrix=bt709:out_range=limited,format=yuva420p10le[patch];"
          f"[0:v][patch]overlay=eof_action=pass:format=auto:alpha=straight[outv]")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-stats", "-y",
         "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", str(background_eye),
         "-i", str(flat_restored),
         "-filter_complex", fc, "-map", "[outv]", "-map", "0:a?",
         "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
         "-pix_fmt", "p010le", "-color_range", "tv",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-c:a", "copy", str(dst)])


def rm_restore(src: Path) -> Path:
    """Run the PTMediaServer offline RM on a whole clip; return the output path."""
    rc = demosaic_offline.main(["single", str(src), "--out-dir", str(src.parent)])
    if rc != 0:
        raise RuntimeError(f"demosaic_offline failed rc={rc} on {src}")
    out = src.with_name(f"{src.stem}{demosaic_offline.OUTPUT_SUFFIX}.mp4")
    if not out.is_file():
        raise RuntimeError(f"expected RM output missing: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VR-to-flat before restoration prototype")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out-dir", dest="out_dir", default="")
    args = ap.parse_args(argv)

    src = Path(args.video)
    if not src.is_file():
        log(f"input not found: {src}")
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    # 1) crop the left eye for the clip window
    eye_clip = out_dir / f"{stem}_L_clip.mp4"
    crop_left_eye_clip(src, eye_clip, args.start, args.duration)
    eye_w, eye_h = probe_wh(eye_clip)
    log(f"[eye] {eye_w}x{eye_h}")

    # 2) auto yaw/pitch/fov from a representative frame (mid clip)
    rgb = sample_frame_rgb(eye_clip, max(0.0, args.duration / 2.0))
    yaw, pitch, d_fov, flat_side, _box = auto_view_from_detection(rgb)

    # 3a) baseline: RM directly on the equirect eye
    log("=== baseline: RM directly on equirect ===")
    direct = rm_restore(eye_clip)
    direct_named = out_dir / f"{stem}_L_direct_restored.mp4"
    shutil.move(str(direct), str(direct_named))

    # 3b) new path: equirect -> flat -> RM -> reproject + overlay
    log("=== vr2flat: equirect->flat->RM->reproject ===")
    flat = out_dir / f"{stem}_L_flat.mp4"
    equirect_to_flat(eye_clip, flat, yaw, pitch, d_fov, flat_side)
    flat_restored = rm_restore(flat)
    flat_restored_named = out_dir / f"{stem}_L_flat_restored.mp4"
    shutil.move(str(flat_restored), str(flat_restored_named))
    vr2flat = out_dir / f"{stem}_L_vr2flat_restored.mp4"
    flat_to_equirect_overlay(eye_clip, flat_restored_named, vr2flat,
                             yaw, pitch, d_fov, eye_w, eye_h)

    log("")
    log("=== DONE — compare these two ===")
    log(f"  baseline (current): {direct_named}")
    log(f"  vr2flat  (new)    : {vr2flat}")
    log(f"  (intermediates: {flat.name}, {flat_restored_named.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
