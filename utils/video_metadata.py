"""ffprobe metadata and backend routing policy for passthrough streams.

The server uses this module to decide whether a video can use the PyNv all-HEVC
path or must fall back to the older FFmpeg pipeline. The decision is based on
timing stability, codec, pixel format, bit depth, HDR markers, and resolution.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Literal

import config
from utils.cache_key import stat_key
from utils.gpu_requirements import detect_nvidia_gpu_requirement, parse_compute_capability
from utils.subprocess_hidden import hidden_subprocess_kwargs

BackendVerdict = Literal["pynv_hevc", "ffmpeg_fallback", "block"]


@dataclass(frozen=True)
class VideoColorMetadata:
    color_range: str = ""
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""

    def ffmpeg_args(self) -> list[str]:
        args: list[str] = []
        if self.color_range:
            args += ["-color_range", self.color_range]
        if self.color_primaries:
            args += ["-color_primaries", self.color_primaries]
        if self.color_transfer:
            args += ["-color_trc", self.color_transfer]
        if self.color_space:
            args += ["-colorspace", self.color_space]
        return args


@dataclass(frozen=True)
class VideoTimingMetadata:
    r_frame_rate: str = ""
    avg_frame_rate: str = ""
    duration: float = 0.0
    nb_frames: int = 0
    time_base: str = ""
    source_fps: float = 0.0
    is_cfr: bool = False
    fps_diff_ratio: float = 0.0

    def effective_fps(self, cap_fps: float | None) -> float:
        fps = self.source_fps if self.source_fps > 0 else 30.0
        if cap_fps and cap_fps > 0:
            return min(fps, float(cap_fps))
        return fps


@dataclass(frozen=True)
class VideoCodecMetadata:
    codec_name: str = ""
    profile: str = ""
    level: int = 0
    pix_fmt: str = ""
    width: int = 0
    height: int = 0
    bit_depth: int = 0
    audio_codec: str = ""
    audio_profile: str = ""

    @property
    def max_side(self) -> int:
        return max(self.width, self.height)

    @property
    def pixels(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def resolution_bucket(self) -> str:
        if self.width <= 0 or self.height <= 0:
            return "unknown"
        if self.max_side >= 7680 or self.pixels >= 7680 * 4320:
            return "8k_plus"
        if self.max_side >= 3840 or self.pixels >= 3840 * 2160:
            return "4k"
        if self.max_side >= 1920 or self.pixels >= 1920 * 1080:
            return "1080p"
        return "sub_1080p"


@dataclass(frozen=True)
class VideoProbeMetadata:
    codec: VideoCodecMetadata
    color: VideoColorMetadata
    timing: VideoTimingMetadata


@dataclass(frozen=True)
class BackendDecision:
    verdict: BackendVerdict
    reason: str


def parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    text = str(value).strip()
    if not text or text == "0/0":
        return 0.0
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except Exception:
            return 0.0


def probe_color_metadata(path: Path) -> VideoColorMetadata:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_range,color_space,color_transfer,color_primaries",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs())
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
    except Exception:
        stream = {}
    return VideoColorMetadata(
        color_range=str(stream.get("color_range") or ""),
        color_space=str(stream.get("color_space") or ""),
        color_transfer=str(stream.get("color_transfer") or ""),
        color_primaries=str(stream.get("color_primaries") or ""),
    )


def probe_timing_metadata(path: Path) -> VideoTimingMetadata:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,time_base:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs())
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
    except Exception:
        stream = {}
        fmt = {}

    r_rate = str(stream.get("r_frame_rate") or "")
    avg_rate = str(stream.get("avg_frame_rate") or "")
    r_fps = parse_rate(r_rate)
    avg_fps = parse_rate(avg_rate)
    source_fps = avg_fps or r_fps or 30.0
    if r_fps > 0 and avg_fps > 0:
        diff_ratio = abs(r_fps - avg_fps) / max(r_fps, avg_fps)
    else:
        diff_ratio = 0.0
    try:
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    except Exception:
        duration = 0.0
    try:
        nb_frames = int(stream.get("nb_frames") or 0)
    except Exception:
        nb_frames = 0

    return VideoTimingMetadata(
        r_frame_rate=r_rate,
        avg_frame_rate=avg_rate,
        duration=duration,
        nb_frames=nb_frames,
        time_base=str(stream.get("time_base") or ""),
        source_fps=source_fps,
        is_cfr=(r_fps > 0 and avg_fps > 0 and diff_ratio < 0.01),
        fps_diff_ratio=diff_ratio,
    )


def probe_video_metadata(path: Path) -> VideoProbeMetadata:
    key = stat_key(path)
    cached = _video_probe_cache.get(key)
    if cached is not None:
        return cached
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,level,pix_fmt,width,height,bits_per_raw_sample,"
            "color_range,color_space,color_transfer,color_primaries,"
            "avg_frame_rate,r_frame_rate,nb_frames,duration,time_base:"
            "format=duration:"
            "stream_tags=rotate"
        ),
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs())
        data = json.loads(out)
        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), streams[0] if streams else {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        fmt = data.get("format") or {}
    except Exception:
        video = {}
        audio = {}
        fmt = {}

    r_rate = str(video.get("r_frame_rate") or "")
    avg_rate = str(video.get("avg_frame_rate") or "")
    r_fps = parse_rate(r_rate)
    avg_fps = parse_rate(avg_rate)
    source_fps = avg_fps or r_fps or 30.0
    diff_ratio = abs(r_fps - avg_fps) / max(r_fps, avg_fps) if r_fps > 0 and avg_fps > 0 else 0.0
    try:
        duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    except Exception:
        duration = 0.0
    try:
        nb_frames = int(video.get("nb_frames") or 0)
    except Exception:
        nb_frames = 0
    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        level = int(video.get("level") or 0)
    except Exception:
        width = height = level = 0
    try:
        bit_depth = int(video.get("bits_per_raw_sample") or 0)
    except Exception:
        bit_depth = 0
    pix_fmt = str(video.get("pix_fmt") or "")
    if bit_depth <= 0 and "10" in pix_fmt:
        bit_depth = 10
    elif bit_depth <= 0 and pix_fmt:
        bit_depth = 8

    meta = VideoProbeMetadata(
        codec=VideoCodecMetadata(
            codec_name=str(video.get("codec_name") or ""),
            profile=str(video.get("profile") or ""),
            level=level,
            pix_fmt=pix_fmt,
            width=width,
            height=height,
            bit_depth=bit_depth,
            audio_codec=str(audio.get("codec_name") or ""),
            audio_profile=str(audio.get("profile") or ""),
        ),
        color=VideoColorMetadata(
            color_range=str(video.get("color_range") or ""),
            color_space=str(video.get("color_space") or ""),
            color_transfer=str(video.get("color_transfer") or ""),
            color_primaries=str(video.get("color_primaries") or ""),
        ),
        timing=VideoTimingMetadata(
            r_frame_rate=r_rate,
            avg_frame_rate=avg_rate,
            duration=duration,
            nb_frames=nb_frames,
            time_base=str(video.get("time_base") or ""),
            source_fps=source_fps,
            is_cfr=(r_fps > 0 and avg_fps > 0 and diff_ratio < 0.01),
            fps_diff_ratio=diff_ratio,
        ),
    )
    _video_probe_cache[key] = meta
    return meta


def probe_codec_metadata(path: Path) -> VideoCodecMetadata:
    return probe_video_metadata(path).codec


def select_backend(
    timing: VideoTimingMetadata,
    codec_meta: VideoCodecMetadata,
    color: VideoColorMetadata | None = None,
) -> BackendDecision:
    codec = codec_meta.codec_name.lower()
    pix_fmt = codec_meta.pix_fmt.lower()
    profile = codec_meta.profile.lower()
    transfer = (color.color_transfer if color else "").lower()
    primaries = (color.color_primaries if color else "").lower()
    is_hdr = transfer in {"smpte2084", "arib-std-b67"}

    if codec not in {"h264", "hevc", "vp9", "av1"}:
        if codec in {"mpeg4", "msmpeg4v3"}:
            return BackendDecision("ffmpeg_fallback", f"codec {codec or 'unknown'} is not enabled for PyNv route")
        return BackendDecision("block", f"unsupported or unknown video codec: {codec or 'unknown'}")
    if codec == "av1":
        gpu, cc_text, cc = _av1_decode_gpu_capability()
        if cc is None or cc < 8.6:
            return BackendDecision(
                "ffmpeg_fallback",
                f"AV1 NVDEC is not available on {gpu} (compute capability {cc_text}); using FFmpeg decode fallback",
            )
    if not timing.is_cfr:
        return BackendDecision("ffmpeg_fallback", "VFR or weak-CFR source needs timestamp-preserving path")
    if pix_fmt not in {"yuv420p", "yuvj420p", "nv12", "p010le", "yuv420p10le"}:
        return BackendDecision("ffmpeg_fallback", f"pixel format {pix_fmt or 'unknown'} is not safe for PyNv NV12/P010 route")
    if is_hdr or "10" in pix_fmt or codec_meta.bit_depth > 8 or "main 10" in profile or "main10" in profile:
        if (
            config.PASSTHROUGH_PYNV_10BIT
            and not is_hdr
            and codec == "hevc"
            and pix_fmt in {"p010le", "yuv420p10le"}
            and (not primaries or primaries in {"bt709", "unknown", "unspecified"})
            and (not transfer or transfer in {"bt709", "unknown", "unspecified"})
        ):
            return BackendDecision("pynv_hevc", "experimental SDR Main10/P016 route enabled")
        return BackendDecision("ffmpeg_fallback", "HDR/Main10/P010 needs a separate color/10-bit policy")
    if codec_meta.width <= 0 or codec_meta.height <= 0:
        return BackendDecision("block", "missing dimensions")
    if codec_meta.resolution_bucket == "8k_plus":
        return BackendDecision("pynv_hevc", "8K+ output should use HEVC; H.264 NVENC failed in probe")
    return BackendDecision("pynv_hevc", "CFR SDR 8-bit source is eligible for all-HEVC PyNv passthrough")


@lru_cache(maxsize=1)
def _av1_decode_gpu_capability() -> tuple[str, str, float | None]:
    requirement = detect_nvidia_gpu_requirement()
    gpu = requirement.name or "current NVIDIA GPU"
    cc_text = requirement.compute_capability or "unknown"
    return gpu, cc_text, parse_compute_capability(cc_text)


def cfr_source_index(out_index: int, source_fps: float, output_fps: float) -> int:
    if source_fps <= 0 or output_fps <= 0:
        return int(out_index)
    return int(round(out_index * source_fps / output_fps))


_video_probe_cache: dict[tuple[str, int, int], VideoProbeMetadata] = {}
