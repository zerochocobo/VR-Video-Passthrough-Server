"""Runtime subtitle parsing/rendering for burned-in PyNv output."""

from __future__ import annotations

import bisect
import concurrent.futures
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from utils.logger import get
from utils.subtitles import subtitle_output_enabled

log = get("subtitles")

IPD_METERS = 0.063
_TAG_RE = re.compile(r"\{[^}]*\}")
_SRT_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d+)\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d+)"
)
_PROJECT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="subtitle-v360")


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str
    lines: tuple["SubtitleLine", ...] = ()


@dataclass(frozen=True)
class SubtitleLine:
    text: str
    primary_color: tuple[int, int, int] | None = None
    outline_color: tuple[int, int, int] | None = None
    font_size: int | None = None


def find_subtitle_for_video(video_path: Path) -> Path | None:
    if not subtitle_output_enabled():
        return None
    seen: set[Path] = set()
    tried: list[Path] = []
    for ext in config.SUBTITLE_EXTS:
        candidate = video_path.parent / f"{video_path.stem}{ext}"
        tried.append(candidate)
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            log.info("subtitle matched: video=%s subtitle=%s", video_path, candidate)
            return candidate
    log.info(
        "subtitle not found: video=%s tried=%s",
        video_path,
        [str(path) for path in tried],
    )
    return None


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp932", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _srt_timestamp_to_sec(match: re.Match[str], offset: int) -> float:
    hours = int(match.group(offset))
    minutes = int(match.group(offset + 1))
    seconds = int(match.group(offset + 2))
    frac = match.group(offset + 3)
    millis = int((frac + "000")[:3])
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


def _ass_time_to_sec(value: str) -> float:
    main, _, frac = value.strip().partition(".")
    parts = [int(p) for p in main.split(":")]
    if len(parts) != 3:
        raise ValueError(f"invalid ASS time {value!r}")
    centis = int((frac + "00")[:2]) if frac else 0
    return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2] + centis / 100.0


def _clean_ass_text(text: str) -> str:
    text = text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    text = _TAG_RE.sub("", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _parse_ass_color(value: str) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    match = re.search(r"&H([0-9A-Fa-f]{6,8})", text)
    if not match:
        return None
    hex_text = match.group(1)[-6:]
    try:
        b = int(hex_text[0:2], 16)
        g = int(hex_text[2:4], 16)
        r = int(hex_text[4:6], 16)
    except ValueError:
        return None
    return r, g, b


def _subtitle_color() -> tuple[int, int, int]:
    if config.SUBTITLE_COLOR is not None:
        return config.SUBTITLE_COLOR
    r, g, b = config.COMPOSITE_BG_RGB
    return 255 - int(r), 255 - int(g), 255 - int(b)


def _parse_srt(path: Path) -> list[SubtitleCue]:
    text = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    cues: list[SubtitleCue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = 0
        if "-->" not in lines[0] and len(lines) > 1:
            time_index = 1
        if time_index >= len(lines):
            continue
        match = _SRT_TIME_RE.search(lines[time_index])
        if not match:
            continue
        body = "\n".join(lines[time_index + 1 :]).strip()
        if not body:
            continue
        clean_lines = [
            _TAG_RE.sub("", line).strip()
            for line in body.splitlines()
            if _TAG_RE.sub("", line).strip()
        ]
        clean_text = "\n".join(clean_lines).strip()
        if not clean_text:
            continue
        cues.append(
            SubtitleCue(
                _srt_timestamp_to_sec(match, 1),
                _srt_timestamp_to_sec(match, 5),
                clean_text,
                tuple(SubtitleLine(line) for line in clean_lines),
            )
        )
    return sorted(cues, key=lambda cue: cue.start)


def _parse_ass(path: Path) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    in_events = False
    in_styles = False
    fields: list[str] = []
    text_index = 9
    start_index = 1
    end_index = 2
    style_index = 3
    style_fields: list[str] = []
    style_name_index = 0
    style_fontsize_index = 2
    style_primary_index = 3
    style_outline_index = 5
    play_res_y = 0
    style_info: dict[str, tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, int | None]] = {}
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("playresy:"):
            try:
                play_res_y = int(float(stripped.split(":", 1)[1].strip()))
            except ValueError:
                play_res_y = 0
            continue
        if lower == "[v4+ styles]":
            in_styles = True
            in_events = False
            continue
        if lower == "[events]":
            in_events = True
            in_styles = False
            continue
        if stripped.startswith("[") and lower not in {"[events]", "[v4+ styles]"}:
            in_events = False
            in_styles = False
            continue
        if in_styles and lower.startswith("format:"):
            style_fields = [part.strip().lower() for part in stripped.split(":", 1)[1].split(",")]
            style_name_index = style_fields.index("name") if "name" in style_fields else 0
            style_fontsize_index = style_fields.index("fontsize") if "fontsize" in style_fields else 2
            style_primary_index = style_fields.index("primarycolour") if "primarycolour" in style_fields else 3
            style_outline_index = style_fields.index("outlinecolour") if "outlinecolour" in style_fields else 5
            continue
        if in_styles and lower.startswith("style:"):
            parts = line.split(":", 1)[1].split(",")
            if len(parts) <= max(style_name_index, style_fontsize_index, style_primary_index, style_outline_index):
                continue
            try:
                font_size = int(round(float(parts[style_fontsize_index])))
            except ValueError:
                font_size = None
            style_info[parts[style_name_index].strip().lower()] = (
                _parse_ass_color(parts[style_primary_index]),
                _parse_ass_color(parts[style_outline_index]),
                font_size,
            )
            continue
        if not in_events:
            continue
        if lower.startswith("format:"):
            fields = [part.strip().lower() for part in stripped.split(":", 1)[1].split(",")]
            text_index = fields.index("text") if "text" in fields else 9
            start_index = fields.index("start") if "start" in fields else 1
            end_index = fields.index("end") if "end" in fields else 2
            style_index = fields.index("style") if "style" in fields else 3
            continue
        if not lower.startswith("dialogue:"):
            continue
        payload = line.split(":", 1)[1]
        max_splits = max(len(fields) - 1, text_index, 9)
        parts = payload.split(",", max_splits)
        if len(parts) <= max(text_index, start_index, end_index, style_index):
            continue
        text = _clean_ass_text(parts[text_index])
        if not text:
            continue
        try:
            style_name = parts[style_index].strip().lower()
            primary, outline, font_size = style_info.get(style_name, (None, None, None))
            scaled_font_size = None
            if font_size is not None and play_res_y > 0:
                scaled_font_size = -max(1, font_size) * 1000000 - play_res_y
            cues.append(
                SubtitleCue(
                    _ass_time_to_sec(parts[start_index]),
                    _ass_time_to_sec(parts[end_index]),
                    text,
                    (SubtitleLine(text, primary, outline, scaled_font_size),),
                )
            )
        except ValueError:
            continue
    return _merge_same_time_cues(cues)


def _merge_same_time_cues(cues: list[SubtitleCue]) -> list[SubtitleCue]:
    grouped: dict[tuple[int, int], list[SubtitleLine]] = {}
    order: list[tuple[int, int]] = []
    times: dict[tuple[int, int], tuple[float, float]] = {}
    for cue in sorted(cues, key=lambda item: (item.start, item.end)):
        key = (int(round(cue.start * 1000)), int(round(cue.end * 1000)))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
            times[key] = (cue.start, cue.end)
        lines = cue.lines or (SubtitleLine(cue.text),)
        existing_texts = {line.text for line in grouped[key]}
        for line in lines:
            if line.text not in existing_texts:
                grouped[key].append(line)
                existing_texts.add(line.text)
    return [
        SubtitleCue(times[key][0], times[key][1], "\n".join(line.text for line in grouped[key]), tuple(grouped[key]))
        for key in order
    ]


class SubtitleRenderer:
    def __init__(self, path: Path, video_width: int, video_height: int):
        self.path = path
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        mode = str(config.SUBTITLE_MODE or "auto").lower()
        stereo_hint = mode in {"dual", "left", "right"} or (mode == "auto" and self.video_width >= 3000)
        self.eye_width = self.video_width // 2 if stereo_hint else self.video_width
        self.eye_height = self.video_height
        suffix = path.suffix.lower()
        self.cues = _parse_ass(path) if suffix == ".ass" else _parse_srt(path)
        self.starts = [cue.start for cue in self.cues]
        self._cache: dict[tuple, tuple[np.ndarray, int, int]] = {}
        self._pending: dict[tuple, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._font = self._load_font()
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {
            int(getattr(self._font, "size", 24)): self._font
        }
        log.info("subtitle loaded: %s cues=%d mode=%s direction=%s", path.name, len(self.cues), config.SUBTITLE_MODE, config.SUBTITLE_DIRECTION)

    @property
    def enabled(self) -> bool:
        return bool(self.cues)

    def _load_font(self):
        size = max(16, int(round(self.eye_height * float(config.SUBTITLE_FONT_SCALE))))
        candidates = []
        if config.SUBTITLE_FONT:
            candidates.append(config.SUBTITLE_FONT)
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\meiryo.ttc",
                r"C:\Windows\Fonts\arial.ttf",
            ]
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _font_for_line(self, line: SubtitleLine):
        size = int(getattr(self._font, "size", 24))
        if line.font_size is not None:
            if line.font_size < -1000000:
                encoded = -int(line.font_size)
                ass_size = encoded // 1000000
                play_res_y = encoded % 1000000
                if play_res_y > 0:
                    size = max(1, int(round(ass_size * self.eye_height / play_res_y)))
            else:
                size = max(1, int(line.font_size))
        cached = self._font_cache.get(size)
        if cached is not None:
            return cached
        candidates = []
        if config.SUBTITLE_FONT:
            candidates.append(config.SUBTITLE_FONT)
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\meiryo.ttc",
                r"C:\Windows\Fonts\arial.ttf",
            ]
        )
        for candidate in candidates:
            try:
                font = ImageFont.truetype(candidate, size=size)
                self._font_cache[size] = font
                return font
            except OSError:
                continue
        self._font_cache[size] = self._font
        return self._font

    def cue_at(self, seconds: float) -> SubtitleCue | None:
        if not self.cues:
            return None
        idx = bisect.bisect_right(self.starts, seconds) - 1
        if idx < 0:
            return None
        cue = self.cues[idx]
        if cue.start <= seconds < cue.end:
            return cue
        return None

    def text_at(self, seconds: float) -> str:
        cue = self.cue_at(seconds)
        return cue.text if cue is not None else ""

    def overlay_for_time(self, seconds: float) -> tuple[np.ndarray, int, int] | None:
        cue = self.cue_at(seconds)
        if cue is None or not cue.text:
            self._prewarm_around(seconds)
            return None
        key = self._overlay_key(cue)
        cached_overlay: tuple[np.ndarray, int, int] | None = None
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                cached_overlay = cached
                future = None
            else:
                future = self._pending.get(key)
                if future is None:
                    future = _PROJECT_EXECUTOR.submit(self._render_and_store, key)
                    self._pending[key] = future
                    return None
                if not future.done():
                    return None
                self._pending.pop(key, None)
        if cached_overlay is not None:
            self._prewarm_around(seconds)
            return cached_overlay
        try:
            overlay = future.result()
        except Exception as e:
            log.warning("subtitle render failed: %s", e)
            return None
        self._prewarm_around(seconds)
        return overlay

    def _overlay_key(self, cue: SubtitleCue) -> tuple:
        line_key = tuple(
            (
                line.text,
                line.primary_color,
                line.outline_color,
                line.font_size,
            )
            for line in (cue.lines or (SubtitleLine(cue.text),))
        )
        return (
            cue.text,
            line_key,
            self.eye_width,
            self.eye_height,
            config.SUBTITLE_DIRECTION,
            str(_subtitle_color()),
            str(config.SUBTITLE_V360),
            f"{config.SUBTITLE_FOV:.3f}",
            f"{config.SUBTITLE_YAW:.3f}",
            f"{config.SUBTITLE_PITCH:.3f}",
        )

    def _prewarm_around(self, seconds: float) -> None:
        idx = max(0, bisect.bisect_left(self.starts, seconds))
        for cue in self.cues[idx : idx + 4]:
            self._submit_render(self._overlay_key(cue))

    def _submit_render(self, key: tuple) -> None:
        with self._lock:
            if key in self._cache or key in self._pending:
                return
            future = _PROJECT_EXECUTOR.submit(self._render_and_store, key)
            self._pending[key] = future

    def _render_and_store(self, key: tuple) -> tuple[np.ndarray, int, int]:
        started = time.perf_counter()
        lines = tuple(
            SubtitleLine(line_text, primary, outline, font_size)
            for line_text, primary, outline, font_size in key[1]
        )
        overlay = self._render_text_overlay(lines)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._cache[key] = overlay
            self._pending.pop(key, None)
        if elapsed_ms >= 100.0:
            log.info("subtitle overlay rendered: text_len=%d elapsed=%.1fms cache=%d pending=%d", len(key[0]), elapsed_ms, len(self._cache), len(self._pending))
        return overlay

    def _render_text_overlay(self, styled_lines: tuple[SubtitleLine, ...]) -> tuple[np.ndarray, int, int]:
        direction = config.SUBTITLE_DIRECTION
        max_width = max(64, int(self.eye_width * 0.86))
        margin = max(4, int(round(self.eye_height * float(config.SUBTITLE_MARGIN_V_SCALE))))
        stroke = max(0, int(round(getattr(self._font, "size", 24) * float(config.SUBTITLE_OUTLINE_SCALE))))
        if direction.startswith("vertical"):
            image = self._render_vertical_image(styled_lines, stroke)
        else:
            image = self._render_horizontal_image(styled_lines, max_width, stroke)
        if direction.startswith("vertical"):
            top = max(0, (self.eye_height - image.height) // 2)
            if direction.endswith("left"):
                left = margin
            elif direction.endswith("right"):
                left = max(0, self.eye_width - image.width - margin)
            else:
                left = max(0, (self.eye_width - image.width) // 2)
        else:
            if direction.endswith("top"):
                top = margin
            elif direction.endswith("middle"):
                top = max(0, (self.eye_height - image.height) // 2)
            else:
                top = max(0, self.eye_height - image.height - margin)
            left = max(0, (self.eye_width - image.width) // 2)
        if config.SUBTITLE_V360:
            return self._project_flat_to_eye(image, left, top)
        return np.asarray(image, dtype=np.uint8), left, top

    def _render_horizontal_image(self, styled_lines: tuple[SubtitleLine, ...], max_width: int, stroke: int) -> Image.Image:
        draw_lines = self._layout_horizontal_lines(styled_lines, max_width)
        dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        boxes = [draw.textbbox((0, 0), line.text, font=self._font_for_line(line), stroke_width=stroke) for line in draw_lines]
        width = min(max_width, max((box[2] - box[0] for box in boxes), default=1) + stroke * 2)
        line_heights = [box[3] - box[1] for box in boxes] or [1]
        line_gap = max(2, int(round(getattr(self._font, "size", 24) * 0.18)))
        height = sum(line_heights) + line_gap * max(0, len(draw_lines) - 1) + stroke * 2
        image = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        alpha = int(round(255 * float(config.SUBTITLE_ALPHA)))
        y = stroke
        for line, box, line_h in zip(draw_lines, boxes, line_heights):
            line_w = box[2] - box[0]
            x = max(0, (width - line_w) // 2)
            fill_rgb = line.primary_color or _subtitle_color()
            outline_rgb = line.outline_color or config.SUBTITLE_OUTLINE_COLOR
            font = self._font_for_line(line)
            draw.text(
                (x, y),
                line.text,
                font=font,
                fill=(*fill_rgb, alpha),
                stroke_width=stroke,
                stroke_fill=(*outline_rgb, alpha),
            )
            y += line_h + line_gap
        return image

    def _render_vertical_image(self, styled_lines: tuple[SubtitleLine, ...], stroke: int) -> Image.Image:
        columns = [
            (line, [char for char in line.text.replace("\r", "").replace("\n", "") if not char.isspace()])
            for line in styled_lines
            if line.text.strip()
        ]
        if not columns:
            columns = [(SubtitleLine(""), [""])]
        dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        font_size = max(1, int(getattr(self._font, "size", 24)))
        col_gap = max(6, int(round(font_size * 0.28)))
        measured: list[tuple[SubtitleLine, list[str], int, int, list[tuple[int, int, int, int]]]] = []
        for source, chars in columns:
            font = self._font_for_line(source)
            boxes = [draw.textbbox((0, 0), char, font=font, stroke_width=stroke) for char in chars]
            size = max(1, int(getattr(font, "size", font_size)))
            col_width = max(size, max((box[2] - box[0] for box in boxes), default=size)) + stroke * 2
            cell_height = max(size, max((box[3] - box[1] for box in boxes), default=size)) + max(0, stroke * 2)
            measured.append((source, chars, col_width, cell_height, boxes))
        width = sum(item[2] for item in measured) + col_gap * max(0, len(measured) - 1)
        height = max(
            (item[3] * len(item[1]) + stroke * 2 for item in measured),
            default=font_size,
        )
        image = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        alpha = int(round(255 * float(config.SUBTITLE_ALPHA)))
        x = 0
        for source, chars, col_width, cell_height, boxes in measured:
            fill_rgb = source.primary_color or _subtitle_color()
            outline_rgb = source.outline_color or config.SUBTITLE_OUTLINE_COLOR
            font = self._font_for_line(source)
            col_height = cell_height * len(chars) + stroke * 2
            y = max(stroke, (height - col_height) // 2 + stroke)
            for char, box in zip(chars, boxes):
                char_w = box[2] - box[0]
                char_h = box[3] - box[1]
                cell_x = x + max(0, (col_width - char_w) // 2) - box[0]
                cell_y = y + max(0, (cell_height - char_h) // 2) - box[1]
                draw.text(
                    (cell_x, cell_y),
                    char,
                    font=font,
                    fill=(*fill_rgb, alpha),
                    stroke_width=stroke,
                    stroke_fill=(*outline_rgb, alpha),
                )
                y += cell_height
            x += col_width + col_gap
        return image

    def _project_flat_to_eye(self, image: Image.Image, left: int, top: int) -> tuple[np.ndarray, int, int]:
        src = np.asarray(image, dtype=np.uint8)
        alpha = src[:, :, 3]
        local_ys, local_xs = np.nonzero(alpha)
        if len(local_xs) == 0 or len(local_ys) == 0:
            return src[:1, :1, :], 0, 0

        src_rgba = src[local_ys, local_xs, :]
        flat_x = local_xs.astype(np.float32) + float(left) + 0.5
        flat_y = local_ys.astype(np.float32) + float(top) + 0.5
        cx = self.eye_width * 0.5
        cy = self.eye_height * 0.5
        fov = math.radians(max(1.0, min(179.0, float(config.SUBTITLE_FOV))))
        focal = (self.eye_width * 0.5) / math.tan(fov * 0.5)

        x = (flat_x - cx) / focal
        y = -(flat_y - cy) / focal
        z = np.ones_like(x)

        yaw = math.radians(-float(config.SUBTITLE_YAW))
        pitch = math.radians(-float(config.SUBTITLE_PITCH))
        cyaw = math.cos(yaw)
        syaw = math.sin(yaw)
        cpitch = math.cos(pitch)
        spitch = math.sin(pitch)

        x_yaw = cyaw * x + syaw * z
        z_yaw = -syaw * x + cyaw * z
        y_pitch = cpitch * y - spitch * z_yaw
        z_pitch = spitch * y + cpitch * z_yaw

        norm = np.sqrt(x_yaw * x_yaw + y_pitch * y_pitch + z_pitch * z_pitch)
        lon = np.arctan2(x_yaw, z_pitch)
        lat = np.arcsin(np.clip(y_pitch / np.maximum(norm, 1.0e-6), -1.0, 1.0))
        out_x = np.rint((lon / math.pi + 0.5) * self.eye_width - 0.5).astype(np.int32)
        out_y = np.rint((0.5 - lat / math.pi) * self.eye_height - 0.5).astype(np.int32)
        valid = (out_x >= 0) & (out_x < self.eye_width) & (out_y >= 0) & (out_y < self.eye_height)
        if not np.any(valid):
            return src[:1, :1, :], 0, 0

        out_x = out_x[valid]
        out_y = out_y[valid]
        src_rgba = src_rgba[valid]

        x0 = max(0, int(out_x.min()) - 1)
        x1 = min(self.eye_width, int(out_x.max()) + 2)
        y0 = max(0, int(out_y.min()) - 1)
        y1 = min(self.eye_height, int(out_y.max()) + 2)
        arr = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        rel_x = out_x - x0
        rel_y = out_y - y0
        for dy in (0, 1):
            yy = rel_y + dy
            y_valid = yy < arr.shape[0]
            if not np.any(y_valid):
                continue
            for dx in (0, 1):
                xx = rel_x + dx
                mask = y_valid & (xx < arr.shape[1])
                if not np.any(mask):
                    continue
                arr[yy[mask], xx[mask], :] = src_rgba[mask]

        alpha = arr[:, :, 3]
        ys, xs = np.nonzero(alpha)
        if len(xs) == 0 or len(ys) == 0:
            return arr[:1, :1, :], 0, 0
        crop_x0, crop_x1 = int(xs.min()), int(xs.max()) + 1
        crop_y0, crop_y1 = int(ys.min()), int(ys.max()) + 1
        return np.ascontiguousarray(arr[crop_y0:crop_y1, crop_x0:crop_x1, :]), x0 + crop_x0, y0 + crop_y0

    def _wrap_text(self, text: str, max_width: int) -> list[str]:
        dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        out: list[str] = []
        for raw_line in text.splitlines() or [text]:
            chars = list(raw_line.strip())
            line = ""
            for char in chars:
                candidate = line + char
                box = draw.textbbox((0, 0), candidate, font=self._font)
                if line and box[2] - box[0] > max_width:
                    out.append(line)
                    line = char
                else:
                    line = candidate
            if line:
                out.append(line)
        return out or [text]

    def _layout_horizontal_lines(self, styled_lines: tuple[SubtitleLine, ...], max_width: int) -> list[SubtitleLine]:
        out: list[SubtitleLine] = []
        for source in styled_lines or (SubtitleLine(""),):
            for wrapped in self._wrap_text(source.text, max_width):
                out.append(SubtitleLine(wrapped, source.primary_color, source.outline_color, source.font_size))
        return [line for line in out if line.text] or [SubtitleLine("")]

    def parallax_px(self) -> int:
        distance_m = max(0.1, float(config.SUBTITLE_DISTANCE_M))
        angle_rad = 2 * math.atan(IPD_METERS / (2 * distance_m))
        return -int(round((angle_rad / math.pi) * self.eye_width))
