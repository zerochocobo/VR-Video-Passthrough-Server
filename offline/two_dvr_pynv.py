"""GPU-resident offline 2D->3D/VR pipeline (PyNvVideoCodec + CuPy).

Keeps every frame on the GPU end to end:

    NVDEC decode -> GpuNv12Frame
      -> NV12->RGB (CuPy kernel)
      -> DA3 depth (small 518 letterbox is the only host round-trip)
      -> stereo warp (CuPy RawKernel -> SBS RGB on device)
      -> RGB->NV12 (CuPy kernel)
      -> NVENC encode -> ffmpeg mux (+ source audio)

This removes the rawvideo CPU pipe and the per-frame frame-upload / SBS-download
of the ffmpeg pipeline (offline/two_dvr.py). The SBS frame is 2x the source, so
that download was the dominant cost at >=1080p.

flat3d only for now (the 3D / SBS target). VR projections fall back to the
ffmpeg pipeline. Requires CuPy (sm_120-native build) + PyNvVideoCodec; the caller
should fall back to offline/two_dvr.py when unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import config
from offline.da3_depth import Da3DepthEngine
from offline.two_dvr_gpu import GpuStereoRenderer, gpu_available
from offline.two_dvr_render import (
    DEFAULT_TEMPORAL_DEPTH,
    DEFAULT_TEMPORAL_DEPTH_ALPHA,
    DEFAULT_TEMPORAL_DEPTH_MODE,
    DEFAULT_TEMPORAL_AFFINE,
    DEFAULT_TEMPORAL_AFFINE_MAX_BIAS,
    DEFAULT_TEMPORAL_AFFINE_MAX_SCALE,
    DEFAULT_TEMPORAL_FLOW_CONSISTENCY,
    DEFAULT_TEMPORAL_FLOW_DIFF,
    DEFAULT_TEMPORAL_FLOW_MOTION_GATE,
    DEFAULT_TEMPORAL_MOTION_MAX_STEP_PX,
    DEFAULT_TEMPORAL_WINDOW,
    DEFAULT_TEMPORAL_NORM,
    DEFAULT_TEMPORAL_NORM_ALPHA,
    DEFAULT_TEMPORAL_NORM_RESET,
    DEFAULT_TEMPORAL_STATIC_DEADBAND_PX,
    DEFAULT_TEMPORAL_STATIC_MAX_STEP_PX,
    HOLE_FILL_INVERSE_WARP,
    HOLE_FILL_SOFT_SHIFT,
    PROJECTION_FLAT_3D,
    effective_eye_distance_mm,
    strength_multiplier,
)
from utils.subprocess_hidden import hidden_subprocess_kwargs

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
DA3_SIZE = 518


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if not np.isfinite(value):
        return "--:--"
    total = max(0, int(round(value)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _progress_message(done: int, total: int, started: float) -> str:
    elapsed = max(0.0, time.time() - started)
    fps = float(done) / max(1e-6, elapsed)
    if total > 0:
        percent = min(100.0, max(0.0, float(done) * 100.0 / float(total)))
        remaining = max(0, int(total) - int(done))
        eta = (float(remaining) / fps) if fps > 1e-6 and remaining > 0 else 0.0
        return (
            f"{done}/{total} frames ({percent:5.1f}%) "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} {fps:.1f} fps"
        )
    return f"{done} frames elapsed={_format_duration(elapsed)} eta=--:-- {fps:.1f} fps"


# BT.709 limited-range conversion (1080p+). Decode and encode use the same
# matrix so the RGB round-trip is consistent.
_NV12_RGB_KERNELS = r'''
extern "C" __global__ void nv12_to_rgb(
    const unsigned char* Y, const unsigned char* UV, unsigned char* rgb, int W, int H)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;
    float c = (float)Y[(long)y*W+x] - 16.0f;
    int cy=y>>1, cx=(x>>1)<<1;
    float du = (float)UV[(long)cy*W+cx]   - 128.0f;
    float dv = (float)UV[(long)cy*W+cx+1] - 128.0f;
    float R = 1.16438356f*c + 1.79274107f*dv;
    float G = 1.16438356f*c - 0.21324861f*du - 0.53290933f*dv;
    float B = 1.16438356f*c + 2.11240179f*du;
    long o=((long)y*W+x)*3;
    rgb[o+0]=(unsigned char)fminf(fmaxf(R,0.f),255.f);
    rgb[o+1]=(unsigned char)fminf(fmaxf(G,0.f),255.f);
    rgb[o+2]=(unsigned char)fminf(fmaxf(B,0.f),255.f);
}

// Letterbox the source into a square (size x size) RGB canvas for DA3 depth.
extern "C" __global__ void nv12_to_rgb_letterbox(
    const unsigned char* Y, const unsigned char* UV, unsigned char* rgb,
    int W, int H, int size, int x0, int y0, int nw, int nh)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox>=size || oy>=size) return;
    long o=((long)oy*size+ox)*3;
    if (ox<x0 || ox>=x0+nw || oy<y0 || oy>=y0+nh){ rgb[o]=0; rgb[o+1]=0; rgb[o+2]=0; return; }
    float sx = (float)(ox-x0)*(float)W/(float)nw;
    float sy = (float)(oy-y0)*(float)H/(float)nh;
    int xi=min((int)sx,W-1), yi=min((int)sy,H-1);
    float c = (float)Y[(long)yi*W+xi] - 16.0f;
    int cy=yi>>1, cx=(xi>>1)<<1;
    float du=(float)UV[(long)cy*W+cx]-128.0f, dv=(float)UV[(long)cy*W+cx+1]-128.0f;
    float R=1.16438356f*c+1.79274107f*dv;
    float G=1.16438356f*c-0.21324861f*du-0.53290933f*dv;
    float B=1.16438356f*c+2.11240179f*du;
    rgb[o+0]=(unsigned char)fminf(fmaxf(R,0.f),255.f);
    rgb[o+1]=(unsigned char)fminf(fmaxf(G,0.f),255.f);
    rgb[o+2]=(unsigned char)fminf(fmaxf(B,0.f),255.f);
}

// Pack RGB (H,W,3) into NV12 (Y on top H rows, interleaved UV on H/2 rows).
extern "C" __global__ void rgb_to_nv12(
    const unsigned char* rgb, unsigned char* nv12, int W, int H)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;
    long i=((long)y*W+x)*3;
    float R=rgb[i], G=rgb[i+1], B=rgb[i+2];
    float Yv=16.f + 0.182586f*R + 0.614231f*G + 0.062007f*B;
    nv12[(long)y*W+x]=(unsigned char)fminf(fmaxf(Yv,0.f),255.f);
    if ((x&1)==0 && (y&1)==0){
        // average the 2x2 block for chroma
        float r=0,g=0,b=0; int n=0;
        for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
            int xx=min(x+dx,W-1), yy=min(y+dy,H-1); long j=((long)yy*W+xx)*3;
            r+=rgb[j]; g+=rgb[j+1]; b+=rgb[j+2]; n++;
        }
        r/=n; g/=n; b/=n;
        float U=128.f -0.100644f*r -0.338572f*g +0.439216f*b;
        float V=128.f +0.439216f*r -0.398942f*g -0.040274f*b;
        long uvbase=(long)W*H + (long)(y>>1)*W + (x);
        nv12[uvbase]  =(unsigned char)fminf(fmaxf(U,0.f),255.f);
        nv12[uvbase+1]=(unsigned char)fminf(fmaxf(V,0.f),255.f);
    }
}
'''


def supported(projection: str, hole_fill: str) -> bool:
    # flat3d (3D/SBS) + fisheye/hequirect 180 (VR), both inverse_warp and
    # soft_shift hole-fill, run on the GPU-resident path.
    return hole_fill in (HOLE_FILL_INVERSE_WARP, HOLE_FILL_SOFT_SHIFT) and gpu_available()


def _letterbox_box(W: int, H: int, size: int = DA3_SIZE) -> tuple[int, int, int, int]:
    scale = min(size / max(1, W), size / max(1, H))
    nw = max(1, int(round(W * scale)))
    nh = max(1, int(round(H * scale)))
    return (size - nw) // 2, (size - nh) // 2, nw, nh


def _encoder_kwargs(bitrate: str, fps: float) -> dict:
    kwargs = {"codec": "hevc", "bitrate": str(bitrate), "fps": f"{fps:.6f}",
              "gop": str(config.PASSTHROUGH_GOP),
              "bf": str(config.PASSTHROUGH_HEVC_BF),
              "repeatspspps": "1"}
    if config.PASSTHROUGH_PYNV_PRESET:
        kwargs["preset"] = str(config.PASSTHROUGH_PYNV_PRESET)
    if config.PASSTHROUGH_PYNV_RC:
        kwargs["rc"] = str(config.PASSTHROUGH_PYNV_RC)
    return kwargs


def _open_muxer(out: Path, fps: float, src: Path, start: float, duration: float, with_audio: bool):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-fflags", "+genpts",
           "-f", "hevc", "-framerate", f"{fps:.6f}", "-i", "-"]
    if with_audio:
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        if duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "copy", "-tag:v", "hvc1", "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            **hidden_subprocess_kwargs())


def convert_clip_pynv(src: Path, out: Path, engine: Da3DepthEngine, args,
                      start: float, duration: float, log=print) -> int:
    import cupy as cp
    import PyNvVideoCodec as nvc
    from pipeline.pynv_io import GpuNv12AppFrame, PyNvThreadedSerialDecoder

    if not engine.folded:
        raise RuntimeError("pynv pipeline requires the folded-preprocess DA3 model")

    mod = cp.RawModule(code=_NV12_RGB_KERNELS)
    k_to_rgb = mod.get_function("nv12_to_rgb")
    k_lb = mod.get_function("nv12_to_rgb_letterbox")
    k_to_nv12 = mod.get_function("rgb_to_nv12")

    from pipeline.pynv_io import PyNvSimpleDecoder

    sp = PyNvSimpleDecoder(src)
    info_obj = sp.info
    W, H, fps, total_frames = sp.info.width, sp.info.height, sp.info.fps, len(sp)
    sp.stop()
    start_frame = int(round(start * fps)) if start > 0 else 0
    n_frames = int(round(duration * fps)) if duration > 0 else (total_frames - start_frame)
    n_frames = max(0, min(n_frames, total_frames - start_frame))

    strength = strength_multiplier(getattr(args, "strength", 1.0))
    eye_distance = effective_eye_distance_mm(args.eye_distance, strength)
    temporal_kwargs = {
        "temporal_norm": bool(getattr(args, "temporal_norm", DEFAULT_TEMPORAL_NORM)),
        "temporal_norm_alpha": float(getattr(args, "temporal_norm_alpha", DEFAULT_TEMPORAL_NORM_ALPHA)),
        "temporal_norm_reset": float(getattr(args, "temporal_norm_reset", DEFAULT_TEMPORAL_NORM_RESET)),
        "temporal_depth": bool(getattr(args, "temporal_depth", DEFAULT_TEMPORAL_DEPTH)),
        "temporal_depth_mode": str(getattr(args, "temporal_depth_mode", DEFAULT_TEMPORAL_DEPTH_MODE)),
        "temporal_depth_alpha": float(getattr(args, "temporal_depth_alpha", DEFAULT_TEMPORAL_DEPTH_ALPHA)),
        "temporal_flow_diff": float(getattr(args, "temporal_flow_diff", DEFAULT_TEMPORAL_FLOW_DIFF)),
        "temporal_flow_consistency": float(getattr(args, "temporal_flow_consistency", DEFAULT_TEMPORAL_FLOW_CONSISTENCY)),
        "temporal_flow_motion_gate": float(getattr(args, "temporal_flow_motion_gate", DEFAULT_TEMPORAL_FLOW_MOTION_GATE)),
        "temporal_affine": bool(getattr(args, "temporal_affine", DEFAULT_TEMPORAL_AFFINE)),
        "temporal_affine_max_scale": float(getattr(args, "temporal_affine_max_scale", DEFAULT_TEMPORAL_AFFINE_MAX_SCALE)),
        "temporal_affine_max_bias": float(getattr(args, "temporal_affine_max_bias", DEFAULT_TEMPORAL_AFFINE_MAX_BIAS)),
        "temporal_static_deadband_px": float(
            getattr(args, "temporal_static_deadband_px", DEFAULT_TEMPORAL_STATIC_DEADBAND_PX)
        ),
        "temporal_static_max_step_px": float(
            getattr(args, "temporal_static_max_step_px", DEFAULT_TEMPORAL_STATIC_MAX_STEP_PX)
        ),
        "temporal_motion_max_step_px": float(
            getattr(args, "temporal_motion_max_step_px", DEFAULT_TEMPORAL_MOTION_MAX_STEP_PX)
        ),
    }
    # Offline symmetric temporal window (8.5.3 B): replaces the causal base/detail
    # stabilizer with a lag-free symmetric base median. When on, disable the causal
    # depth stabilizer so prepare_near_gpu returns the un-smoothed near.
    window_radius = max(0, min(10, int(getattr(args, "temporal_window", DEFAULT_TEMPORAL_WINDOW))))
    band_lookahead = window_radius > 0 and config.TWO_DVR_BAND_LOOKAHEAD
    if window_radius > 0:
        temporal_kwargs["temporal_depth"] = False
        temporal_kwargs["temporal_depth_mode"] = "off"
        if band_lookahead:
            # raw per-frame band; the symmetric window does the (lookahead) smoothing.
            temporal_kwargs["temporal_norm"] = False
    renderer = GpuStereoRenderer(W, H, args.projection, eye_distance, args.hole_fill, args.flat_fov, **temporal_kwargs)
    renderer.reset()
    window = None
    if window_radius > 0:
        from offline.two_dvr_gpu import GpuSymmetricWindow
        window = GpuSymmetricWindow(renderer._near_pre, radius=window_radius)
    cut_detector = None
    if config.TWO_DVR_SCENE_CUT:
        from utils.scene_detection import SceneCutDetector
        cut_detector = SceneCutDetector(threshold=config.TWO_DVR_SCENE_CUT_THRESHOLD)
    out_w, out_h = renderer.out_w, renderer.out_h
    da3_size = int(engine.size)   # 518 (base/small) or 1036 (hd presets)
    x0, y0, nw, nh = _letterbox_box(W, H, da3_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporal_depth_on = temporal_kwargs["temporal_depth"] and temporal_kwargs["temporal_depth_mode"] != "off"
    log(f"{src.name}: {W}x{H}@{fps:.3f} -> SBS {out_w}x{out_h} proj={args.projection} "
        f"fill={args.hole_fill} strength={strength:.2f} "
        f"temporal_norm={'on' if temporal_kwargs['temporal_norm'] else 'off'} "
        f"temporal_depth={'on' if temporal_depth_on else 'off'} "
        f"sym_window={window_radius if window_radius > 0 else 'off'} "
        f"depth_mode={temporal_kwargs['temporal_depth_mode']} "
        f"affine={'on' if temporal_kwargs['temporal_affine'] else 'off'} "
        f"deadband_px={temporal_kwargs['temporal_static_deadband_px']:.2f} "
        f"static_step_px={temporal_kwargs['temporal_static_max_step_px']:.2f} "
        f"motion_step_px={temporal_kwargs['temporal_motion_max_step_px']:.2f} model={args.model} "
        f"depth={engine.providers[0]} pipeline=pynv-gpu")

    dec = PyNvThreadedSerialDecoder(src, start_frame=start_frame, info=info_obj, num_frames=total_frames)
    from utils.bitrate_estimator import parse_bitrate, projection_capped_bitrate
    eff_bitrate = projection_capped_bitrate(
        args.bitrate, src, args.projection,
        config.TWO_DVR_BITRATE_MULT_3D, config.TWO_DVR_BITRATE_MULT_VR,
    )
    if str(eff_bitrate) != str(parse_bitrate(args.bitrate)):
        log(f"{src.name}: bitrate capped {args.bitrate} -> {eff_bitrate/1e6:.1f}M "
            f"(<= {'4x' if str(args.projection).lower() in ('fisheye','hequirect') else '3x'} source)")
    enc = nvc.CreateEncoder(out_w, out_h, "NV12", False, **_encoder_kwargs(eff_bitrate, fps))
    has_audio = _has_audio(src)
    mux = _open_muxer(out, fps, src, start, duration, has_audio)

    # Preallocated device buffers.
    rgb_g = cp.empty((H, W, 3), cp.uint8)
    canvas_g = cp.empty((da3_size, da3_size, 3), cp.uint8)
    out_nv12 = cp.empty((out_h * 3 // 2, out_w), cp.uint8)
    bx = (16, 16, 1)
    grid = ((W + 15) // 16, (H + 15) // 16, 1)
    grid_lb = ((da3_size + 15) // 16, (da3_size + 15) // 16, 1)
    grid_out = ((out_w + 15) // 16, (out_h + 15) // 16, 1)

    started = time.time()
    count = 0

    def _encode_pair(near_c, rgb_c) -> None:
        nonlocal count
        sbs_rgb = renderer.render_into_gpu(rgb_c, near_c)
        k_to_nv12(grid_out, bx, (sbs_rgb, out_nv12, np.int32(out_w), np.int32(out_h)))
        flags = (int(nvc.NV_ENC_PIC_FLAGS.FORCEIDR) | int(nvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)) if count == 0 else 0
        bitstream = enc.Encode(GpuNv12AppFrame(out_nv12, out_w, out_h), flags) if flags else enc.Encode(GpuNv12AppFrame(out_nv12, out_w, out_h))
        if bitstream:
            mux.stdin.write(bitstream)
        count += 1
        if count % 64 == 0:
            log(f"  {_progress_message(count, n_frames, started)}")

    try:
        for idx in range(start_frame, start_frame + n_frames):
            frame = dec.frame_at(idx).owned_copy()
            y_g = frame.y.as_cupy(cp.uint8).reshape(H, W)
            uv_g = frame.uv.as_cupy(cp.uint8).reshape(H // 2, W)
            # depth: NV12 -> 518 letterbox RGB -> ORT (small host round-trip)
            k_lb(grid_lb, bx, (y_g, uv_g, canvas_g, np.int32(W), np.int32(H), np.int32(da3_size),
                               np.int32(x0), np.int32(y0), np.int32(nw), np.int32(nh)))
            canvas = canvas_g.get()[None]  # (1,518,518,3) uint8
            # scene-cut reset before normalization so the new shot re-seeds the
            # depth band (the symmetric window's residual mask handles the base).
            if cut_detector is not None and cut_detector.step(canvas[0]):
                renderer.reset()
            depth = engine.session.run([engine.output_name], {engine.input_name: canvas})[0][0]
            near_g = renderer.prepare_near_gpu(
                depth[y0:y0 + nh, x0:x0 + nw],
                canvas_g[y0:y0 + nh, x0:x0 + nw],
            )
            # full-res NV12 -> RGB, then GPU stereo warp -> SBS RGB
            k_to_rgb(grid, bx, (y_g, uv_g, rgb_g, np.int32(W), np.int32(H)))
            if window is None:
                _encode_pair(near_g, rgb_g)
            else:
                # symmetric window: align by the DA3-input crop, render the buffered
                # full-res rgb. Output trails input by `window_radius` frames.
                band = None
                if band_lookahead:
                    lo_raw, hi_raw = (float(v) for v in renderer._near_pre.band_g[:2].get())
                    band = (lo_raw, hi_raw, lo_raw, hi_raw)
                for near_c, rgb_c, _p in window.push(near_g, canvas_g[y0:y0 + nh, x0:x0 + nw], rgb_g, idx, band=band):
                    _encode_pair(near_c, rgb_c)
        if window is not None:
            for near_c, rgb_c, _p in window.flush():
                _encode_pair(near_c, rgb_c)
        tail = enc.EndEncode()
        if tail:
            mux.stdin.write(tail)
    finally:
        dec.stop()
        if mux.stdin:
            try:
                mux.stdin.close()
            except Exception:
                pass
        mux_err = (mux.stderr.read() or b"").decode("utf-8", "replace") if mux.stderr else ""
        mux.wait()
    if mux.returncode not in (0, None):
        log(f"mux failed rc={mux.returncode}: {mux_err.strip()[:400]}")
        return 1
    elapsed = time.time() - started
    log(f"done {out.name}: {count} frames in {elapsed:.1f}s ({count / max(1e-6, elapsed):.1f} fps)")
    return 0 if count > 0 else 1


def _has_audio(path: Path) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
           "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace",
                                      **hidden_subprocess_kwargs())
        return bool(out.strip())
    except Exception:
        return False
