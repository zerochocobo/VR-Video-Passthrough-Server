from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from pipeline.subtitles import SubtitleRenderer
from ui.settings import SETTINGS_PATH


DEFAULT_VIDEO = config.ROOT / "videos" / "sub_35s.mp4"
DEFAULT_OUT_DIR = config.ROOT / "debug_output" / "subtitle_screenshots"


def _decode_rgb_frame(video_path: Path, seconds: float, max_width: int = 2048) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(seconds)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to decode frame at {seconds:.3f}s from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        return _resize_preview(image, max_width)
    finally:
        cap.release()


def _resize_preview(image: Image.Image, max_width: int) -> Image.Image:
    max_width = int(max_width or 0)
    if max_width <= 0 or image.width <= max_width:
        return image
    width = max_width if max_width % 2 == 0 else max_width - 1
    height = max(2, int(round(image.height * (width / image.width))))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _rgb_image_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_array_to_rgb_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _rgb_image_to_nv12(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    yuv_i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    h, w = rgb.shape[:2]
    y = yuv_i420[:h, :]
    u_flat = yuv_i420[h:h + h // 4, :].reshape(-1)
    v_flat = yuv_i420[h + h // 4:, :].reshape(-1)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv.reshape(-1)[0::2] = u_flat
    uv.reshape(-1)[1::2] = v_flat
    return np.vstack([y, uv])


def _nv12_to_rgb_image(nv12: np.ndarray, width: int, height: int) -> Image.Image:
    rgb = cv2.cvtColor(nv12.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2RGB_NV12)
    return Image.fromarray(rgb)


def _overlay_with_alpha(base: Image.Image, rgba: np.ndarray, left: int, top: int) -> None:
    if rgba.size <= 0:
        return
    overlay = Image.fromarray(rgba, "RGBA")
    base.alpha_composite(overlay, (int(left), int(top)))


def _positioned_overlays(renderer: SubtitleRenderer, overlay, frame_width: int):
    rgba, left, top = overlay
    if rgba.size <= 0:
        return []
    eye_w = renderer.eye_width
    mode = config.SUBTITLE_MODE
    if mode == "auto":
        mode = "dual" if frame_width >= 3000 else "mono"
    if mode == "left":
        positions = [(left, top)]
    elif mode == "right":
        positions = [(eye_w + left, top)]
    elif mode == "dual":
        positions = [(left, top), (eye_w + left + renderer.parallax_px(), top)]
    else:
        positions = [(max(0, (frame_width - int(rgba.shape[1])) // 2), top)]
    return [(rgba, x, y) for x, y in positions]


def _is_sbs_frame(renderer: SubtitleRenderer, frame_width: int) -> bool:
    return renderer.eye_width > 0 and frame_width >= renderer.eye_width * 2


def _positioned_overlay_for_eye(renderer: SubtitleRenderer, overlay, eye: int):
    rgba, left, top = overlay
    if rgba.size <= 0:
        return None
    mode = config.SUBTITLE_MODE
    if mode == "auto":
        mode = "dual"
    if mode == "left" and eye != 0:
        return None
    if mode == "right" and eye != 1:
        return None
    eye_left = left
    if mode == "dual" and eye == 1:
        eye_left += renderer.parallax_px()
    return rgba, eye_left, top


def _render_overlay(renderer: SubtitleRenderer, seconds: float, timeout_sec: float = 5.0):
    deadline = time.monotonic() + max(0.1, timeout_sec)
    while time.monotonic() < deadline:
        overlay = renderer.overlay_for_time(seconds)
        if overlay is not None:
            return overlay
        time.sleep(0.02)
    cue = renderer.cue_at(seconds)
    if cue is None:
        return None
    key = renderer._overlay_key(cue)
    return renderer._render_and_store(key)


def _subtitle_sample_times(renderer: SubtitleRenderer, max_cues: int) -> list[tuple[int, float, float, float, str]]:
    samples = []
    for index, cue in enumerate(renderer.cues, start=1):
        if not cue.text.strip():
            continue
        start = float(cue.start)
        end = float(cue.end)
        middle = start + (end - start) * 0.5
        samples.append((index, start, end, middle, cue.text))
        if len(samples) >= max_cues:
            break
    return samples


def _background_for_scenario(frame: Image.Image, scenario: str) -> Image.Image:
    if scenario == "green":
        bg = Image.new("RGB", frame.size, tuple(int(v) for v in config.COMPOSITE_BG_RGB))
        # Keep the source frame faintly visible so the screenshot still shows the sampled moment.
        return Image.blend(bg, frame, 0.35)
    if scenario == "alpha":
        bg = Image.new("RGB", frame.size, (18, 18, 18))
        return Image.blend(bg, frame, 0.25)
    return frame.copy()


def _matte_green_frame(frame: Image.Image, use_matting: bool) -> Image.Image:
    if not use_matting:
        bg = Image.new("RGB", frame.size, tuple(int(v) for v in config.COMPOSITE_BG_RGB))
        return Image.blend(bg, frame.convert("RGB"), 0.35)
    from pipeline.matting import get_matter

    matter = get_matter()
    composited_bgr = matter.composite_green(_rgb_image_to_bgr_array(frame))
    return _bgr_array_to_rgb_image(composited_bgr)


def _fisheye_eye(eye: Image.Image) -> Image.Image:
    arr = np.asarray(eye.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    out = np.zeros_like(arr)
    cx = w * 0.5
    cy = h * 0.5
    radius = min(w, h) * 0.5
    yy, xx = np.indices((h, w), dtype=np.float32)
    nx = (xx + 0.5 - cx) / radius
    ny = (yy + 0.5 - cy) / radius
    rr = np.sqrt(nx * nx + ny * ny)
    valid = rr <= 1.0
    theta = np.arctan2(nx, -ny)
    phi = rr * (np.pi * 0.5)
    lon = theta
    lat = (np.pi * 0.5) - phi
    src_x = ((lon / np.pi) + 0.5) * (w - 1)
    src_y = (0.5 - lat / np.pi) * (h - 1)
    sx = np.clip(np.rint(src_x).astype(np.int32), 0, w - 1)
    sy = np.clip(np.rint(src_y).astype(np.int32), 0, h - 1)
    out[valid] = arr[sy[valid], sx[valid]]
    return Image.fromarray(out)


def _alpha_layout_frame(frame: Image.Image, renderer: SubtitleRenderer, overlay, use_matting: bool) -> Image.Image:
    if use_matting:
        from pipeline.matting import get_matter
        from pipeline.alpha_packer import AlphaPacker

        matter = get_matter()
        bgr = _rgb_image_to_bgr_array(frame)
        alpha = matter.alpha(bgr)
        nv12 = _rgb_image_to_nv12(frame)
        matter._upload_nv12_gpu(nv12, frame.height, frame.width)
        packer = AlphaPacker(matter)
        subtitle_overlays = _positioned_overlays(renderer, overlay, frame.width) if overlay is not None else None
        out_nv12 = packer.pack_uploaded(alpha, frame.height, frame.width, subtitle_overlay=subtitle_overlays)
        import cupy as cp

        return _nv12_to_rgb_image(cp.asnumpy(out_nv12), frame.width, frame.height)
    else:
        gray = np.asarray(frame.convert("L"), dtype=np.float32) / 255.0
        alpha_full = np.clip(gray * 0.8 + 0.1, 0.0, 1.0)
    alpha_img = Image.fromarray(np.clip(alpha_full * 255.0, 0, 255).astype(np.uint8), "L").convert("RGB")

    w, h = frame.size
    eye_w = renderer.eye_width if _is_sbs_frame(renderer, w) else w
    left_eye = frame.crop((0, 0, eye_w, h))
    right_eye = frame.crop((eye_w, 0, eye_w * 2, h)) if _is_sbs_frame(renderer, w) else frame
    left_alpha = alpha_img.crop((0, 0, eye_w, h))
    right_alpha = alpha_img.crop((eye_w, 0, eye_w * 2, h)) if _is_sbs_frame(renderer, w) else alpha_img

    eyes = [left_eye.convert("RGBA"), right_eye.convert("RGBA")]
    for eye_index, eye_img in enumerate(eyes):
        if overlay is None:
            continue
        positioned = _positioned_overlay_for_eye(renderer, overlay, eye_index)
        if positioned is None:
            continue
        rgba, left, top = positioned
        _overlay_with_alpha(eye_img, rgba, left, top)

    fisheye_left = _fisheye_eye(eyes[0].convert("RGB"))
    fisheye_right = _fisheye_eye(eyes[1].convert("RGB"))
    alpha_left = _fisheye_eye(left_alpha)
    alpha_right = _fisheye_eye(right_alpha)
    alpha_w = max(4, int(round(eye_w * 0.4)) & ~3)
    alpha_h = max(2, int(round(h * 0.4)) & ~1)
    alpha_left = alpha_left.resize((alpha_w, alpha_h), Image.Resampling.LANCZOS)
    alpha_right = alpha_right.resize((alpha_w, alpha_h), Image.Resampling.LANCZOS)

    out = Image.new("RGB", (eye_w * 2 if _is_sbs_frame(renderer, w) else w, h), (0, 0, 0))
    out.paste(fisheye_left, (0, 0))
    out.paste(fisheye_right, (eye_w, 0))
    out.paste(alpha_left, (0, h - alpha_h))
    out.paste(alpha_right, (eye_w, h - alpha_h))
    return out


def _write_case(
    video_path: Path,
    subtitle_path: Path,
    scenario: str,
    out_dir: Path,
    max_cues: int,
    overlay_timeout_sec: float,
    preview_width: int,
    use_matting: bool,
) -> list[dict]:
    render_width, render_height = _preview_dimensions(video_path, preview_width)
    renderer = SubtitleRenderer(subtitle_path, render_width, render_height)
    samples = _subtitle_sample_times(renderer, max_cues)
    case_dir = out_dir / f"{scenario}_{subtitle_path.suffix.lower().lstrip('.')}"
    case_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cue_index, start, end, middle, text in samples:
        frame = _decode_rgb_frame(video_path, middle, preview_width)
        overlay = _render_overlay(renderer, middle, overlay_timeout_sec)
        if scenario == "green":
            canvas = _matte_green_frame(frame, use_matting).convert("RGBA")
        elif scenario == "alpha":
            canvas = _alpha_layout_frame(frame, renderer, overlay, use_matting).convert("RGBA")
        else:
            canvas = _background_for_scenario(frame, scenario).convert("RGBA")
        if overlay is not None and scenario == "green":
            for rgba, left, top in _positioned_overlays(renderer, overlay, canvas.width):
                _overlay_with_alpha(canvas, rgba, left, top)
        filename = f"cue{cue_index:02d}_{middle:06.2f}s.png"
        output = case_dir / filename
        canvas.convert("RGB").save(output)
        records.append(
            {
                "scenario": scenario,
                "subtitle": str(subtitle_path),
                "cue_index": cue_index,
                "start": start,
                "end": end,
                "sample_time": middle,
                "width": canvas.width,
                "height": canvas.height,
                "matting": use_matting,
                "image": str(output),
                "text": text,
            }
        )
    return records


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Unable to read video dimensions: {video_path}")
        return width, height
    finally:
        cap.release()


def _preview_dimensions(video_path: Path, max_width: int) -> tuple[int, int]:
    width, height = _video_dimensions(video_path)
    max_width = int(max_width or 0)
    if max_width <= 0 or width <= max_width:
        return width, height
    out_w = max_width if max_width % 2 == 0 else max_width - 1
    out_h = max(2, int(round(height * (out_w / width))))
    return out_w, out_h


def _setting_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_ui_settings(settings_path: Path) -> dict:
    if not settings_path.is_file():
        return {}
    data = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    config.COMPOSITE_BG_RGB_HEX = str(data.get("background_color") or config.COMPOSITE_BG_RGB_HEX)
    config.COMPOSITE_BG_RGB = config._rgb_hex(config.COMPOSITE_BG_RGB_HEX, "808080")
    config.GREEN_BGR = (config.COMPOSITE_BG_RGB[2], config.COMPOSITE_BG_RGB[1], config.COMPOSITE_BG_RGB[0])
    config.SUBTITLE_MODE = str(data.get("subtitle_mode") or config.SUBTITLE_MODE)
    config.SUBTITLE_DIRECTION = str(data.get("subtitle_direction") or config.SUBTITLE_DIRECTION)
    config.SUBTITLE_DISTANCE_M = float(data.get("subtitle_distance_m") or config.SUBTITLE_DISTANCE_M)
    config.SUBTITLE_FOV = float(data.get("subtitle_fov") or config.SUBTITLE_FOV)
    config.SUBTITLE_YAW = float(data.get("subtitle_yaw") or config.SUBTITLE_YAW)
    config.SUBTITLE_PITCH = float(data.get("subtitle_pitch") or config.SUBTITLE_PITCH)
    config.SUBTITLE_FONT_SCALE = float(data.get("subtitle_font_scale") or config.SUBTITLE_FONT_SCALE)
    config.SUBTITLE_OUTLINE_SCALE = float(data.get("subtitle_outline_scale") or config.SUBTITLE_OUTLINE_SCALE)
    config.SUBTITLE_MARGIN_V_SCALE = float(data.get("subtitle_margin_v_scale") or config.SUBTITLE_MARGIN_V_SCALE)
    config.SUBTITLE_ALPHA = float(data.get("subtitle_alpha") or config.SUBTITLE_ALPHA)
    raw = str(data.get("subtitle_color") or "").strip()
    config.SUBTITLE_COLOR_RAW = raw
    config.SUBTITLE_COLOR = config._rgb_hex(raw, "FFFFFF") if raw else None
    config.SUBTITLE_OUTLINE_COLOR = config._rgb_hex(str(data.get("subtitle_outline_color") or "000000"), "000000")
    config.SUBTITLE_V360 = _setting_bool(data.get("subtitle_v360", config.SUBTITLE_V360))
    return data


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_name in ("green_srt", "green_ass", "alpha_srt", "alpha_ass"):
        shutil.rmtree(out_dir / case_name, ignore_errors=True)
    if args.matting:
        from utils.gpu_runtime_cache import configure_gpu_runtime_cache

        configure_gpu_runtime_cache()
    settings = {}
    if not args.no_ui_settings:
        settings = _apply_ui_settings(Path(args.settings).resolve())

    subtitle_paths = {
        "srt": Path(args.srt).resolve() if args.srt else video_path.with_suffix(".srt"),
        "ass": Path(args.ass).resolve() if args.ass else video_path.with_suffix(".ass"),
    }
    missing = [str(path) for path in subtitle_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing subtitle file(s): " + ", ".join(missing))

    if args.direction:
        config.SUBTITLE_DIRECTION = args.direction
    if args.mode:
        config.SUBTITLE_MODE = args.mode
    if args.subtitle_color is not None:
        raw = args.subtitle_color.strip()
        config.SUBTITLE_COLOR_RAW = raw
        config.SUBTITLE_COLOR = config._rgb_hex(raw, "FFFFFF") if raw else None

    records = []
    for scenario in ("green", "alpha"):
        for subtitle_path in (subtitle_paths["srt"], subtitle_paths["ass"]):
            records.extend(
                _write_case(
                    video_path,
                    subtitle_path,
                    scenario,
                    out_dir,
                    max(1, int(args.max_cues)),
                    float(args.overlay_timeout_sec),
                    int(args.preview_width),
                    bool(args.matting),
                )
            )

    index_path = out_dir / "index.json"
    payload = {
        "video": str(video_path),
        "settings_path": "" if args.no_ui_settings else str(Path(args.settings).resolve()),
        "applied_settings": {
            "subtitle_mode": config.SUBTITLE_MODE,
            "subtitle_direction": config.SUBTITLE_DIRECTION,
            "subtitle_distance_m": config.SUBTITLE_DISTANCE_M,
            "subtitle_fov": config.SUBTITLE_FOV,
            "subtitle_yaw": config.SUBTITLE_YAW,
            "subtitle_pitch": config.SUBTITLE_PITCH,
            "subtitle_alpha": config.SUBTITLE_ALPHA,
            "subtitle_color": config.SUBTITLE_COLOR_RAW,
            "background_color": config.COMPOSITE_BG_RGB_HEX,
            "loaded_ui_settings": bool(settings),
            "preview_width": int(args.preview_width),
            "matting": bool(args.matting),
        },
        "records": records,
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} screenshots under {out_dir}")
    print(f"Index: {index_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate local subtitle debug screenshots without a DLNA player."
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--srt", default="")
    parser.add_argument("--ass", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--settings", default=str(SETTINGS_PATH), help="UI settings JSON to apply before rendering.")
    parser.add_argument("--no-ui-settings", action="store_true", help="Ignore saved UI settings and use config.py defaults.")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=2048,
        help="Resize screenshots before matting/rendering. Use 0 for source size.",
    )
    parser.add_argument(
        "--matting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the RVM matting model for foreground/alpha. Use --no-matting for fast layout-only checks.",
    )
    parser.add_argument("--max-cues", type=int, default=3)
    parser.add_argument("--direction", default="", help="Override PT_SUBTITLE_DIRECTION for this run.")
    parser.add_argument("--mode", default="", help="Override PT_SUBTITLE_MODE for this run.")
    parser.add_argument(
        "--subtitle-color",
        default=None,
        help="Override SRT subtitle color as RGB hex. Empty string means inverse background.",
    )
    parser.add_argument("--overlay-timeout-sec", type=float, default=5.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
