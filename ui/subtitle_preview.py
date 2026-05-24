from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from utils.ffprobe_json import run_ffprobe_json
from utils.subprocess_hidden import hidden_subprocess_kwargs

IPD_METERS = 0.063
MAX_PREVIEW_WIDTH = 4096
DEFAULT_PREVIEW_TEXT_LINES = ["Test Subtitle Test Subtitle"]


def run_hidden(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"command failed with code {result.returncode}")


def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        video_path,
    ]
    data = run_ffprobe_json(cmd)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(fmt.get("duration") or 0.0),
        "codec": str(stream.get("codec_name") or ""),
    }


def extract_left_eye_frame(video_path: str, seconds: int) -> Image.Image:
    out = str(Path(tempfile.gettempdir()) / "ptserver_subtitle_frame.jpg")
    vf = f"crop=iw/2:ih:0:0,scale=w='min({MAX_PREVIEW_WIDTH},iw)':h=-1:flags=lanczos"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0, int(seconds))),
        "-i",
        video_path,
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-y",
        out,
    ]
    run_hidden(cmd)
    return Image.open(out).convert("RGB")


def sec_to_ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    centis = int(round((seconds - total) * 100))
    if centis >= 100:
        total += 1
        centis = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}.{centis:02d}"


def _preview_dimensions(width: int, height: int) -> tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    if width <= MAX_PREVIEW_WIDTH:
        return width, height
    scale = MAX_PREVIEW_WIDTH / width
    return MAX_PREVIEW_WIDTH, max(1, int(round(height * scale)))


def _filter_path(path: str) -> str:
    normalized = os.path.abspath(path).replace("\\", "/")
    return normalized.replace(":", "\\:").replace("'", "\\'")


def _decoder_options(codec: str) -> list[str]:
    if codec == "h264":
        return ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    if codec == "hevc":
        return ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    return []


def _write_temp_ass(lines: list[str]) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(lines) + "\n")
        return tmp.name
    finally:
        tmp.close()


def _rgb_hex(value: str, default: str = "FFFFFF") -> tuple[int, int, int]:
    text = str(value or default).strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        text = default
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return _rgb_hex(default, "FFFFFF")


def _ass_color_from_rgb(value: str) -> str:
    r, g, b = _rgb_hex(value, "FFFFFF")
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _srt_preview_color(subtitle_color: str, background_color: str) -> str:
    if str(subtitle_color or "").strip():
        return _ass_color_from_rgb(subtitle_color)
    r, g, b = _rgb_hex(background_color, "808080")
    return f"&H00{255 - b:02X}{255 - g:02X}{255 - r:02X}"


def _read_text(path: str | Path) -> str:
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp932", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _verticalize_ass_text(text: str) -> str:
    tokens: list[str] = []
    pending_tags = ""
    index = 0
    while index < len(text):
        if text[index] == "{":
            end = text.find("}", index + 1)
            if end >= 0:
                pending_tags += text[index:end + 1]
                index = end + 1
                continue
        if text.startswith("\\N", index) or text.startswith("\\n", index):
            index += 2
            continue
        if text.startswith("\\h", index):
            char = " "
            index += 2
        else:
            char = text[index]
            index += 1
        if char in "\r\n":
            continue
        tokens.append(pending_tags + char)
        pending_tags = ""
    if pending_tags:
        tokens.append(pending_tags)
    return "{\\fsp0\\q2}" + "\\N".join(tokens) if tokens else text


def _style_indexes(format_fields: list[str]) -> dict[str, int]:
    indexes = {"name": 0, "fontsize": 2, "italic": 8, "spacing": 13, "alignment": 18, "marginl": 19, "marginr": 20, "marginv": 21}
    for key in list(indexes):
        if key in format_fields:
            indexes[key] = format_fields.index(key)
    return indexes


def _safe_int(value: str, fallback: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def _vertical_style_margins(style_info: dict) -> tuple[int, int]:
    default = style_info.get("default", {})
    default_font_size = max(1, default.get("fontsize", 100))
    default_margin = max(default.get("marginl", 0), round(default_font_size * 0.56))
    gap = max(24, round(default_font_size * 0.4))
    return default_margin, default_margin + default_font_size + gap


def _rewrite_style_alignment(line: str, indexes: dict[str, int], alignment: int) -> str:
    prefix, payload = line.split(":", 1)
    parts = payload.split(",")
    if len(parts) <= indexes["alignment"]:
        return line
    parts[indexes["alignment"]] = str(int(alignment))
    return prefix + ":" + ",".join(parts)


def _stabilize_vertical_style_parts(parts: list[str], indexes: dict[str, int]) -> None:
    if len(parts) > indexes["spacing"]:
        parts[indexes["spacing"]] = "0"
    if len(parts) > indexes["italic"]:
        parts[indexes["italic"]] = "0"
    if parts:
        last = parts[-1].strip().lower()
        if last in ("\\q0", "\\q1", "\\q2", "\\q3"):
            parts[-1] = "\\q2"
        else:
            parts.append("\\q2")


def _rewrite_vertical_style_line(line: str, indexes: dict[str, int], side: str, default_margin: int, secondary_margin: int, style_name: str) -> str:
    prefix, payload = line.split(":", 1)
    parts = payload.split(",")
    required_index = max(indexes["alignment"], indexes["marginl"], indexes["marginr"])
    if len(parts) <= required_index:
        return line
    _stabilize_vertical_style_parts(parts, indexes)
    is_secondary = style_name == "secondary"
    if side == "right":
        parts[indexes["alignment"]] = "6"
        parts[indexes["marginl"]] = "10"
        parts[indexes["marginr"]] = str(int(secondary_margin if is_secondary else default_margin))
    elif side == "middle":
        parts[indexes["alignment"]] = "5"
        if is_secondary:
            parts[indexes["marginl"]] = str(int(secondary_margin))
            parts[indexes["marginr"]] = str(int(default_margin))
        else:
            parts[indexes["marginl"]] = str(int(default_margin))
            parts[indexes["marginr"]] = str(int(secondary_margin))
    else:
        parts[indexes["alignment"]] = "4"
        parts[indexes["marginl"]] = str(int(secondary_margin if is_secondary else default_margin))
        parts[indexes["marginr"]] = "10"
    return prefix + ":" + ",".join(parts)


def apply_subtitle_direction(source_ass: str, subtitle_direction: str = "horizontal_middle") -> str:
    alignment_by_direction = {
        "horizontal_top": 8,
        "horizontal_middle": 5,
        "horizontal_bottom": 2,
        "vertical_left": 4,
        "vertical_middle": 5,
        "vertical_right": 6,
    }
    direction = subtitle_direction if subtitle_direction in alignment_by_direction else "horizontal_middle"
    is_vertical = direction.startswith("vertical_")
    vertical_side = direction.rsplit("_", 1)[-1] if is_vertical else ""
    lines = _read_text(source_ass).splitlines()
    style_format_fields: list[str] = []
    style_indexes = _style_indexes(style_format_fields)
    style_info: dict[str, dict[str, int]] = {}
    in_styles = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "[v4+ styles]":
            in_styles = True
            continue
        if stripped.startswith("[") and lower != "[v4+ styles]":
            in_styles = False
            continue
        if in_styles and lower.startswith("format:"):
            style_format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            style_indexes = _style_indexes(style_format_fields)
            continue
        if in_styles and lower.startswith("style:"):
            parts = line.split(":", 1)[1].split(",")
            if len(parts) <= max(style_indexes["name"], style_indexes["fontsize"], style_indexes["marginl"]):
                continue
            style_name = parts[style_indexes["name"]].strip().lower()
            if style_name in {"default", "secondary"}:
                style_info[style_name] = {
                    "fontsize": _safe_int(parts[style_indexes["fontsize"]], 100),
                    "marginl": _safe_int(parts[style_indexes["marginl"]], 0),
                }
    default_margin, secondary_margin = _vertical_style_margins(style_info)
    alignment = alignment_by_direction[direction]
    out: list[str] = []
    in_styles = False
    in_events = False
    format_fields: list[str] = []
    text_index = 9
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "[v4+ styles]":
            in_styles = True
            in_events = False
            out.append(line)
            continue
        if lower == "[events]":
            in_styles = False
            in_events = True
            out.append(line)
            continue
        if stripped.startswith("[") and lower not in {"[v4+ styles]", "[events]"}:
            in_styles = False
            in_events = False
            out.append(line)
            continue
        if in_styles and lower.startswith("format:"):
            style_format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            style_indexes = _style_indexes(style_format_fields)
            out.append(line)
            continue
        if in_styles and lower.startswith("style:"):
            parts = line.split(":", 1)[1].split(",")
            style_name = parts[style_indexes["name"]].strip().lower() if len(parts) > style_indexes["name"] else ""
            if is_vertical:
                out.append(_rewrite_vertical_style_line(line, style_indexes, vertical_side, default_margin, secondary_margin, style_name))
            else:
                out.append(_rewrite_style_alignment(line, style_indexes, alignment))
            continue
        if in_events and lower.startswith("format:"):
            format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            text_index = format_fields.index("text") if "text" in format_fields else 9
            out.append(line)
            continue
        if in_events and lower.startswith("dialogue:"):
            prefix, payload = line.split(":", 1)
            max_splits = max(len(format_fields) - 1, 9) if format_fields else 9
            parts = payload.split(",", max_splits)
            if is_vertical and len(parts) > text_index:
                parts[text_index] = _verticalize_ass_text(parts[text_index])
                out.append(prefix + ":" + ",".join(parts))
                continue
        out.append(line)
    return _write_temp_ass(out)


def create_preview_ass(
    video_info: dict,
    template_path: Path,
    text: str,
    *,
    subtitle_color: str = "",
    background_color: str = "808080",
) -> str:
    preview_width, preview_height = _preview_dimensions(int(video_info["width"]), int(video_info["height"]))
    width = max(1, preview_width // 2)
    height = max(1, preview_height)
    scale = ((width * height) / (1280 * 720)) ** 0.5
    template = template_path.read_text(encoding="utf-8-sig")
    header = template.format(
        width=width,
        height=height,
        cn_size=round(42 * scale),
        jp_size=round(30 * scale),
        marginv=round(32 * scale),
        alignment=5,
        DefaultPrimaryColour=_srt_preview_color(subtitle_color, background_color),
        DefaultOutlineColour="&H00000000",
        SecondaryPrimaryColour="&H00FFFFFF",
        SecondaryOutlineColour="&H00000000",
    ).rstrip()
    duration = max(10.0, float(video_info.get("duration", 10.0) or 10.0))
    end = sec_to_ass_time(duration)
    safe_text = text.replace(",", " ")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8")
    try:
        tmp.write(f"{header}\nDialogue: 0,0:00:00.00,{end},Default,,0,0,0,,{safe_text}\n")
        return tmp.name
    finally:
        tmp.close()


def _parallax_px(video_width: int, distance_m: float) -> int:
    eye_width = int(video_width // 2)
    distance_m = max(0.1, float(distance_m))
    angle_rad = 2 * math.atan(IPD_METERS / (2 * distance_m))
    return -int(round((angle_rad / math.pi) * eye_width))


def _alpha_factor(transparency_percent: float) -> float:
    return 1.0 - max(0.0, min(70.0, float(transparency_percent))) / 100.0


def build_filter_complex(
    ass_file: str,
    width: int,
    height: int,
    fov: float,
    yaw: float,
    pitch: float,
    transparency_percent: float,
    mode: str,
    distance_m: float,
    *,
    scale_main: bool = False,
) -> tuple[str, str]:
    eye_w = int(width // 2)
    eye_h = int(height)
    patch_size = max(512, int(round(min(eye_w, eye_h) / 2)))
    ass_path = _filter_path(ass_file)
    alpha = _alpha_factor(transparency_percent)
    ffmpeg_yaw = -float(yaw)
    ffmpeg_pitch = -float(pitch)
    base = (
        f"[1:v]ass='{ass_path}':alpha=1,split[rgb_src][alpha_src];"
        f"[alpha_src]alphaextract,lutyuv=y='val*{alpha:.3f}',"
        f"v360=input=flat:output=hequirect:w={eye_w}:h={eye_h}:"
        f"id_fov={float(fov):.3f}:yaw={ffmpeg_yaw:.3f}:pitch={ffmpeg_pitch:.3f}:rorder=rpy[alpha_proj];"
        f"[rgb_src]v360=input=flat:output=hequirect:w={eye_w}:h={eye_h}:"
        f"id_fov={float(fov):.3f}:yaw={ffmpeg_yaw:.3f}:pitch={ffmpeg_pitch:.3f}:rorder=rpy[rgb_proj];"
    )
    main_filter = f"[0:v]scale={width}:{height}:flags=lanczos,format=yuv420p[main];" if scale_main else "[0:v]format=yuv420p[main];"
    if mode == "left":
        overlay = (
            "[rgb_proj][alpha_proj]alphamerge,format=yuva420p[patch];"
            f"{main_filter}"
            "[main][patch]overlay=x=0:y=0:eof_action=pass:format=yuv420:alpha=straight[final]"
        )
    elif mode == "right":
        overlay = (
            "[rgb_proj][alpha_proj]alphamerge,format=yuva420p[patch];"
            f"{main_filter}"
            f"[main][patch]overlay=x={eye_w}:y=0:eof_action=pass:format=yuv420:alpha=straight[final]"
        )
    else:
        right_x = eye_w + _parallax_px(width, distance_m)
        overlay = (
            "[rgb_proj][alpha_proj]alphamerge,format=yuva420p,split[patch_l][patch_r];"
            f"{main_filter}"
            "[main][patch_l]overlay=x=0:y=0:eof_action=pass:format=yuv420:alpha=straight[left_done];"
            f"[left_done][patch_r]overlay=x={right_x}:y=0:eof_action=pass:format=yuv420:alpha=straight[final]"
        )
    color_src = f"color=c=0x00000000:s={patch_size}x{patch_size}:r=30,format=yuva420p"
    return base + overlay, color_src


def generate_preview_image(
    video_path: str,
    ass_file: str,
    seconds: int,
    out: str,
    *,
    fov: float,
    yaw: float,
    pitch: float,
    transparency_percent: float,
    mode: str,
    distance_m: float,
    subtitle_direction: str,
) -> None:
    info = get_video_info(video_path)
    source_width = int(info["width"])
    source_height = int(info["height"])
    preview_width, preview_height = _preview_dimensions(source_width, source_height)
    directed_ass = apply_subtitle_direction(ass_file, subtitle_direction)
    try:
        filter_complex, color_src = build_filter_complex(
            directed_ass,
            preview_width,
            preview_height,
            fov,
            yaw,
            pitch,
            transparency_percent,
            mode,
            distance_m,
            scale_main=(preview_width != source_width or preview_height != source_height),
        )
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(max(0, int(seconds)))]
        cmd.extend(_decoder_options(info.get("codec", "")))
        cmd.extend([
            "-i",
            video_path,
            "-f",
            "lavfi",
            "-i",
            color_src,
            "-filter_complex",
            filter_complex,
            "-map",
            "[final]",
            "-frames:v",
            "1",
            "-y",
            out,
        ])
        run_hidden(cmd)
    finally:
        try:
            os.unlink(directed_ass)
        except OSError:
            pass
