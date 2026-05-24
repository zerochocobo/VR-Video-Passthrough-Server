"""FFmpeg subprocess wrappers for the legacy passthrough pipeline.

DecoderProcess seeks and decodes source video to raw BGR24 or NV12 frames on
stdout. EncoderProcess accepts composited raw frames on stdin and produces
fragmented MP4 or MPEG-TS bytes on stdout. ffprobe helpers provide lightweight
metadata for routing, sizing, and DLNA responses.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from config import (
    DECODE_MAX_SIDE,
    DECODE_PIX_FMT,
    FFMPEG_HWACCEL,
    FFMPEG_HWACCEL_OUTPUT,
    PASSTHROUGH_BITRATE,
    PASSTHROUGH_AUD,
    PASSTHROUGH_BF,
    PASSTHROUGH_CONTAINER,
    PASSTHROUGH_DELAY,
    PASSTHROUGH_GOP,
    PASSTHROUGH_MULTIPASS,
    PASSTHROUGH_NO_SCENECUT,
    PASSTHROUGH_RC,
    PASSTHROUGH_RC_LOOKAHEAD,
    PASSTHROUGH_MAX_FPS,
    PASSTHROUGH_PRESET,
    PASSTHROUGH_SPATIAL_AQ,
    PASSTHROUGH_STRICT_GOP,
    PASSTHROUGH_SURFACES,
    PASSTHROUGH_TEMPORAL_AQ,
    PASSTHROUGH_TUNE,
    PASSTHROUGH_VCODEC,
    PASSTHROUGH_ZERO_LATENCY,
)
from utils.cache_key import stat_key
from utils.logger import get
from utils.subprocess_hidden import hidden_subprocess_kwargs

log = get("ffmpeg")

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float  # seconds
    codec_name: str = ""
    pix_fmt: str = ""


def probe(path: Path) -> VideoInfo:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **hidden_subprocess_kwargs())
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    dur = float(j.get("format", {}).get("duration", 0.0))
    return VideoInfo(
        int(s["width"]),
        int(s["height"]),
        fps,
        dur,
        str(s.get("codec_name") or ""),
        str(s.get("pix_fmt") or ""),
    )


# (absolute path, file size, mtime_ns) -> VideoInfo
_probe_cache: dict[tuple[str, int, int], VideoInfo] = {}


def probe_cached(path: Path) -> VideoInfo:
    key = stat_key(path)
    info = _probe_cache.get(key)
    if info is None:
        info = probe(path)
        _probe_cache[key] = info
    return info


class DecoderProcess:
    """Seek and decode one source file into raw frames for matting."""

    def __init__(self, src: Path, start_sec: float, info: VideoInfo, max_fps: float | None = None):
        self.info = info
        self.max_fps = max_fps
        strategy = self._choose_strategy(src, start_sec)
        cmd = self._build_decode_cmd(src, start_sec, strategy.hwaccel_args, strategy.vf_chain)
        self.strategy_name = strategy.name
        self.cmd = cmd
        self.failed_strategies = list(getattr(self, "_failed_strategies", []))
        log.info("[DIAG] decoder plan: requested_hw=%s selected=%s hwaccel_output=%s decode_max_side=%d", (FFMPEG_HWACCEL or "auto").strip().lower(), strategy.name, FFMPEG_HWACCEL_OUTPUT, DECODE_MAX_SIDE)
        log.info("[DIAG] decoder cmd: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
        # Probe the post-filter output size because scaling/fps filters can
        # change the byte layout consumed by the caller.
        self.out_info = self._probe_output_info(src, start_sec)
        self.frame_size = self._frame_size(self.out_info.width, self.out_info.height)
        log.info("[DIAG] decoder out: %dx%d @ %.2ffps", self.out_info.width, self.out_info.height, self.out_info.fps)

    class _DecodeStrategy(NamedTuple):
        name: str
        hwaccel_args: list[str]
        vf_chain: list[str]

    def _build_scale_filter(self) -> str | None:
        if DECODE_MAX_SIDE <= 0:
            return None
        return (
            f"scale='if(gt(iw,ih),min(iw,{DECODE_MAX_SIDE}),-2)':"
            f"'if(gt(iw,ih),-2,min(ih,{DECODE_MAX_SIDE}))'"
        )

    def _output_fps(self) -> float:
        fps_cap = PASSTHROUGH_MAX_FPS if self.max_fps is None else float(self.max_fps)
        if fps_cap > 0:
            return min(self.info.fps, fps_cap)
        return self.info.fps

    def _build_fps_filter(self) -> str | None:
        out_fps = self._output_fps()
        if out_fps >= self.info.fps * 0.999:
            return None
        return f"fps=fps={out_fps:.6f}:round=near"

    @staticmethod
    def _bytes_per_pixel_num_den() -> tuple[int, int]:
        if DECODE_PIX_FMT == "nv12":
            return 3, 2
        return 3, 1

    def _frame_size(self, w: int, h: int) -> int:
        num, den = self._bytes_per_pixel_num_den()
        return w * h * num // den

    def _scaled_size(self) -> tuple[int, int]:
        w, h = self.info.width, self.info.height
        if DECODE_MAX_SIDE <= 0 or max(w, h) <= DECODE_MAX_SIDE:
            return w, h
        if w >= h:
            out_w = DECODE_MAX_SIDE
            out_h = int(round(h * DECODE_MAX_SIDE / w))
        else:
            out_h = DECODE_MAX_SIDE
            out_w = int(round(w * DECODE_MAX_SIDE / h))
        out_w -= out_w % 2
        out_h -= out_h % 2
        return max(out_w, 2), max(out_h, 2)

    def _cuvid_decoder(self) -> str | None:
        return {
            "h264": "h264_cuvid",
            "hevc": "hevc_cuvid",
            "av1": "av1_cuvid",
            "vp8": "vp8_cuvid",
            "vp9": "vp9_cuvid",
            "mpeg1video": "mpeg1_cuvid",
            "mpeg2video": "mpeg2_cuvid",
            "mpeg4": "mpeg4_cuvid",
        }.get(self.info.codec_name)

    def _cuvid_args(self) -> list[str] | None:
        decoder = self._cuvid_decoder()
        if not decoder:
            return None
        args = ["-c:v", decoder]
        out_w, out_h = self._scaled_size()
        if (out_w, out_h) != (self.info.width, self.info.height):
            args += ["-resize", f"{out_w}x{out_h}"]
        return args

    def _hwdownload_filter(self) -> list[str]:
        sw_format = "p010le" if "10" in self.info.pix_fmt else "nv12"
        vf = ["hwdownload", f"format={sw_format}"]
        scale_filter = self._build_scale_filter()
        if scale_filter:
            vf.append(scale_filter)
        fps_filter = self._build_fps_filter()
        if fps_filter:
            vf.append(fps_filter)
        vf.append(f"format={DECODE_PIX_FMT}")
        return vf

    def _build_decode_cmd(self, src: Path, start_sec: float, hwaccel_args: list[str], vf_chain: list[str]) -> list[str]:
        vf_args: list[str] = ["-vf", ",".join(vf_chain)] if vf_chain else []
        return [
            FFMPEG, "-hide_banner", "-loglevel", "warning",
            *hwaccel_args,
            "-threads", "0",
            "-ss", f"{max(0.0, start_sec):.3f}",
            "-i", str(src),
            "-an", "-sn",
            "-fps_mode", "passthrough",
            *vf_args,
            "-f", "rawvideo",
            "-pix_fmt", DECODE_PIX_FMT,
            "-",
        ]

    def _build_probe_cmd(self, src: Path, start_sec: float, hwaccel_args: list[str], vf_chain: list[str]) -> list[str]:
        vf_args: list[str] = ["-vf", ",".join(vf_chain)] if vf_chain else []
        return [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            *hwaccel_args,
            "-threads", "0",
            "-ss", f"{max(0.0, start_sec):.3f}",
            "-i", str(src),
            "-an", "-sn",
            *vf_args,
            "-frames:v", "1",
            "-f", "null",
            "-",
        ]

    def _strategy_candidates(self) -> list[_DecodeStrategy]:
        hw = (FFMPEG_HWACCEL or "auto").strip().lower()
        scale_filter = self._build_scale_filter()
        fps_filter = self._build_fps_filter()

        def with_scale(vf: list[str]) -> list[str]:
            out = list(vf)
            if scale_filter:
                out.append(scale_filter)
            if fps_filter:
                out.append(fps_filter)
            return out

        def with_fps_only(vf: list[str]) -> list[str]:
            out = list(vf)
            if fps_filter:
                out.append(fps_filter)
            return out

        strategies: list[DecoderProcess._DecodeStrategy] = []
        if hw == "cuda":
            cuvid_args = self._cuvid_args()
            if cuvid_args:
                strategies.append(self._DecodeStrategy(
                    name=f"{self._cuvid_decoder()}+resize",
                    hwaccel_args=cuvid_args,
                    vf_chain=with_fps_only([]),
                ))
            if FFMPEG_HWACCEL_OUTPUT:
                strategies.append(self._DecodeStrategy(
                    name="cuda+hwdownload",
                    hwaccel_args=["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
                    vf_chain=self._hwdownload_filter(),
                ))
            strategies.append(self._DecodeStrategy(
                name="cuda",
                hwaccel_args=["-hwaccel", "cuda"],
                vf_chain=with_scale([]),
            ))
            strategies.append(self._DecodeStrategy(
                name="auto",
                hwaccel_args=["-hwaccel", "auto"],
                vf_chain=with_scale([]),
            ))
            strategies.append(self._DecodeStrategy(
                name="software",
                hwaccel_args=[],
                vf_chain=with_scale([]),
            ))
            return strategies

        if hw == "auto":
            strategies.append(self._DecodeStrategy("auto", ["-hwaccel", "auto"], with_scale([])))
            strategies.append(self._DecodeStrategy("software", [], with_scale([])))
            return strategies

        if hw == "none":
            return [self._DecodeStrategy("software", [], with_scale([]))]

        strategies.append(self._DecodeStrategy(hw, ["-hwaccel", hw], with_scale([])))
        strategies.append(self._DecodeStrategy("software", [], with_scale([])))
        return strategies

    def _choose_strategy(self, src: Path, start_sec: float) -> _DecodeStrategy:
        """ 1 EOF """
        last_err = ""
        self._failed_strategies: list[tuple[str, int, str]] = []
        for s in self._strategy_candidates():
            probe_cmd = self._build_probe_cmd(src, start_sec, s.hwaccel_args, s.vf_chain)
            r = subprocess.run(
                probe_cmd,
                capture_output=True,
                **hidden_subprocess_kwargs(),
            )
            if r.returncode == 0:
                if s.name != (FFMPEG_HWACCEL or "auto").strip().lower():
                    log.warning("[DIAG] decoder fallback: selected=%s", s.name)
                return s
            err = (r.stderr or b"").decode("utf-8", "ignore")[-400:]
            last_err = err
            self._failed_strategies.append((s.name, int(r.returncode), err))
            log.warning("[DIAG] decoder strategy failed: %s rc=%s err=%s", s.name, r.returncode, err.replace("\n", " | "))

        raise RuntimeError(f"No decoder strategy available. Last ffmpeg error: {last_err}")

    def _probe_output_info(self, src: Path, start_sec: float) -> VideoInfo:
        """ ffprobe + lavfi """
        if DECODE_MAX_SIDE <= 0:
            return VideoInfo(self.info.width, self.info.height, self._output_fps(), self.info.duration, self.info.codec_name, DECODE_PIX_FMT)
        out_w, out_h = self._scaled_size()
        if (out_w, out_h) == (self.info.width, self.info.height):
            return VideoInfo(self.info.width, self.info.height, self._output_fps(), self.info.duration, self.info.codec_name, DECODE_PIX_FMT)
        return VideoInfo(out_w, out_h, self._output_fps(), self.info.duration, self.info.codec_name, DECODE_PIX_FMT)

    def read_frame(self) -> bytes | bytearray | None:
        buf = bytearray(self.frame_size)
        view = memoryview(buf)
        got = 0
        try:
            while got < self.frame_size:
                n = self.proc.stdout.readinto(view[got:])
                if not n:
                    return None
                got += n
        except (ValueError, OSError):
            # stdout EOF
            return None
        return buf

    @staticmethod
    def _close_pipe(pipe):
        try:
            if pipe:
                pipe.close()
        except Exception:
            pass

    def close(self):
        self._close_pipe(self.proc.stdout)
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1)
        except Exception:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                    self.proc.wait(timeout=1)
            except Exception:
                pass
        self._close_pipe(self.proc.stderr)

    def read_stderr_nonblock(self) -> str:
        """ stderr """
        try:
            if self.proc.poll() is not None:
                data = self.proc.stderr.read()
                return data.decode("utf-8", "ignore") if data else ""
        except Exception:
            pass
        return ""


class EncoderProcess:
    """
 stdin BGR24 -> stdout fMP4/MPEGTS
 fMP4 frag_keyframe+empty_moov+default_base_moof /DLNA
 """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        input_pix_fmt: str = "bgr24",
        container: str | None = None,
        audio_src: Path | None = None,
        audio_start_sec: float = 0.0,
    ):
        self.input_pix_fmt = input_pix_fmt
        container = (container or PASSTHROUGH_CONTAINER).lower()
        if container == "mpegts":
            container_args = ["-muxdelay", "0", "-muxpreload", "0", "-f", "mpegts"]
        else:
            container_args = [
                "-movflags", "frag_keyframe+empty_moov+default_base_moof+faststart",
                "-frag_duration", "250000",
                "-f", "mp4",
            ]
        # Internal note.
        codec_args = [
            "-c:v", PASSTHROUGH_VCODEC,
            "-preset", PASSTHROUGH_PRESET,
            "-b:v", PASSTHROUGH_BITRATE,
            "-g", str(PASSTHROUGH_GOP),
            "-pix_fmt", "yuv420p",
        ]
        if PASSTHROUGH_TUNE:
            codec_args += ["-tune", PASSTHROUGH_TUNE]
        if PASSTHROUGH_RC:
            codec_args += ["-rc", PASSTHROUGH_RC]
        nvenc_options = (
            ("-rc-lookahead", PASSTHROUGH_RC_LOOKAHEAD),
            ("-bf", PASSTHROUGH_BF),
            ("-multipass", PASSTHROUGH_MULTIPASS),
            ("-no-scenecut", PASSTHROUGH_NO_SCENECUT),
            ("-spatial-aq", PASSTHROUGH_SPATIAL_AQ),
            ("-temporal-aq", PASSTHROUGH_TEMPORAL_AQ),
            ("-surfaces", PASSTHROUGH_SURFACES),
            ("-delay", PASSTHROUGH_DELAY),
            ("-zerolatency", PASSTHROUGH_ZERO_LATENCY),
            ("-strict_gop", PASSTHROUGH_STRICT_GOP),
            ("-aud", PASSTHROUGH_AUD),
        )
        for opt, value in nvenc_options:
            if value != "":
                codec_args += [opt, str(value)]
        audio_args = ["-an"]
        if audio_src is not None:
            audio_args = [
                "-ss", f"{max(0.0, audio_start_sec):.3f}",
                "-i", str(audio_src),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac",
                "-b:a", "192k",
            ]
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "warning",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", input_pix_fmt,
            "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}",
            "-i", "-",
            *audio_args,
            *codec_args,
            *container_args,
            "-",
        ]
        self.cmd = cmd
        log.info("[DIAG] encoder cmd: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )

    def write_frame(self, frame_bgr) -> bool:
        chunk_size = 1024 * 1024
        try:
            view = memoryview(frame_bgr)
            for offset in range(0, len(view), chunk_size):
                if self.proc.poll() is not None:
                    return False
                self.proc.stdin.write(view[offset:offset + chunk_size])
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def read_output(self, n: int = 65536) -> bytes:
        return self.proc.stdout.read(n)

    def read_stderr_nonblock(self) -> str:
        """ stderr """
        try:
            if self.proc.poll() is not None:
                data = self.proc.stderr.read()
                return data.decode("utf-8", "ignore") if data else ""
        except Exception:
            pass
        return ""

    @staticmethod
    def _close_pipe(pipe):
        try:
            if pipe:
                pipe.close()
        except Exception:
            pass

    def close(self):
        # stdout Ctrl+C/ executor read()
        self._close_pipe(self.proc.stdin)
        self._close_pipe(self.proc.stdout)
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1)
        except Exception:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                    self.proc.wait(timeout=1)
            except Exception:
                pass
        self._close_pipe(self.proc.stderr)
