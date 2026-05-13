"""Application entry point.

Startup order:
1. Configure runtime caches and optional GPU warmup.
2. Start SSDP so DLNA clients can discover the server.
3. Start FastAPI/uvicorn for device descriptions, SOAP, and media streams.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import uvicorn

import config
from dlna.ssdp import SSDPServer
from http_app.server import create_app
from utils.firewall import ensure_rules
from utils.gpu_runtime_cache import (
    configure_gpu_runtime_cache,
    predict_warmup_state,
    warmup_gpu_runtime_cache,
)
from utils.logger import get, setup
from utils.startup_status import (
    reset_startup_progress,
    set_startup_phase,
    start_startup_status_server,
    stop_startup_status_server,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start VR Video Passthrough Server.")
    parser.add_argument("mode", nargs="?", choices=["DEBUG", "debug"], default=None, help="use DEBUG to enable verbose diagnostics")
    parser.add_argument("--debug", action="store_true", help="enable verbose diagnostic logs")
    return parser.parse_args(argv)


def _apply_debug_arg(args: argparse.Namespace) -> None:
    enabled = bool(args.debug or str(args.mode).lower() == "debug")
    if enabled:
        os.environ["PT_DEBUG_LOGS"] = "1"
        config.DEBUG_LOGS = True


def main(argv: list[str] | None = None) -> int:
    """Start the DLNA media server process."""

    if argv and argv[0] == "offline":
        from offline.convert import main as offline_main

        return offline_main(argv[1:])

    args = _parse_args(argv)
    _apply_debug_arg(args)
    cache_env = configure_gpu_runtime_cache()
    setup()
    log = get("main")
    start_startup_status_server(config.STARTUP_STATUS_PORT)
    set_startup_phase("starting", "process started")
    log.info("LAN_IP=%s HTTP_PORT=%d UUID=%s", config.LAN_IP, config.HTTP_PORT, config.DEVICE_UUID)
    log.info("VIDEO_DIRS=%s", "|".join(str(path) for path in config.VIDEO_DIRS))
    log.info("MODEL_PATH=%s (exists=%s)", config.MODEL_PATH, config.MODEL_PATH.exists())
    log.info("GPU_RUNTIME_CACHE=%s", cache_env)
    if config.STARTUP_GPU_WARMUP:
        # Publish a prediction first so the UI can show the expected duration
        # before any heavy CUDA work begins. Failure to predict is non-fatal.
        try:
            prediction = predict_warmup_state()
            set_startup_phase(
                "warming",
                ("first-time GPU initialization" if prediction.cold else "verifying GPU cache"),
                step="predict",
                step_index=0,
                step_total=4,
                progress=0.0,
                eta_sec=prediction.estimate_sec,
                elapsed_sec=0.0,
                cold=prediction.cold,
                is_known_slow=prediction.is_known_slow,
                gpu_name=prediction.gpu_name,
                compute_capability=prediction.compute_capability,
                driver_version=prediction.driver_version,
                onnxruntime_version=prediction.onnxruntime_version,
                reason=prediction.reason,
                detail=prediction.detail,
            )
            log.info(
                "warmup prediction: cold=%s reason=%s known_slow=%s eta=%.1fs gpu=%s cc=%s ort=%s",
                prediction.cold,
                prediction.reason,
                prediction.is_known_slow,
                prediction.estimate_sec,
                prediction.gpu_name,
                prediction.compute_capability,
                prediction.onnxruntime_version,
            )
        except Exception as e:
            log.warning("warmup prediction failed (non-fatal): %s", e)
            set_startup_phase("warming", "GPU runtime warmup")

        log.info(
            "startup GPU warmup begin: force=%s timeout=%.1fs runs_per_shape=%d",
            config.STARTUP_GPU_WARMUP_FORCE,
            config.STARTUP_GPU_WARMUP_TIMEOUT,
            config.STARTUP_GPU_WARMUP_RUNS_PER_SHAPE,
        )
        warmup_start = time.perf_counter()
        try:
            # Note: warmup_gpu_runtime_cache is a single blocking call. We can't
            # currently emit per-substep events without restructuring it, but we
            # do report the "running" sub-step so the UI knows the ORT session
            # is loading and JIT compilation is in progress.
            set_startup_phase(
                "warming",
                "loading ONNX Runtime CUDA and running warmup",
                step="ort_session_and_runs",
                step_index=1,
                step_total=4,
                progress=0.1,
            )
            marker = warmup_gpu_runtime_cache(
                force=config.STARTUP_GPU_WARMUP_FORCE,
                timeout_sec=max(1.0, config.STARTUP_GPU_WARMUP_TIMEOUT),
                runs_per_shape=max(1, config.STARTUP_GPU_WARMUP_RUNS_PER_SHAPE),
            )
        except Exception as e:
            set_startup_phase(
                "failed",
                f"startup GPU warmup failed: {e}",
                step="failed",
                progress=0.0,
                detail=str(e),
            )
            log.exception("startup GPU warmup failed; server will not start: %s", e)
            # Give the UI poller (500 ms interval) one more chance to read the
            # "failed" status before we tear down the local /status endpoint.
            # Without this delay the UI sometimes sees only the prior "warming"
            # snapshot and falls back to the synthesized failed state, which
            # loses the precise detail/message published by this branch.
            time.sleep(0.8)
            stop_startup_status_server()
            return 1
        warmup_elapsed = time.perf_counter() - warmup_start
        set_startup_phase(
            "warmed",
            "GPU runtime warmup complete",
            step="warmed",
            step_index=4,
            step_total=4,
            progress=1.0,
            eta_sec=0.0,
            elapsed_sec=warmup_elapsed,
        )
        log.info(
            "startup GPU warmup done: elapsed=%.3fs marker_elapsed=%.3fs verified_second_pass=%.3fs cache_files=%d cache_size=%d",
            warmup_elapsed,
            marker.elapsed_sec,
            marker.verified_second_pass_sec,
            marker.cache_file_count_after_warmup,
            marker.cache_size_after_warmup,
        )
    else:
        set_startup_phase(
            "warmed",
            "startup GPU warmup disabled",
            step="warmed",
            progress=1.0,
            eta_sec=0.0,
        )
        log.info("startup GPU warmup disabled")
    reset_startup_progress()
    from pipeline.matting import matter_device
    log.info(
        "PIPELINE: HWACCEL=%s DECODE_MAX_SIDE=%d DECODE_PIX_FMT=%s PASSTHROUGH_MAX_FPS=%.2f "
        "ALPHA_STRIDE=%d "
        "MATTING_INPUT_SIZE=%d WARMUP_RUNS=%d "
        "USE_PYNV=%s VCODEC=%s HEVC_BITRATE=%s HEVC_BF=%s CONTAINER=%s SEEK_MODE=%s OUTPUT_MODE=%s "
        "MAX_CONCURRENT=%d PAD_TO_LENGTH=%s COMPOSITE_DEVICE=%s PYNV_10BIT=%s PYNV_10BIT_SHIFT=%d",
        config.FFMPEG_HWACCEL,
        config.DECODE_MAX_SIDE,
        config.DECODE_PIX_FMT,
        config.PASSTHROUGH_MAX_FPS,
        config.ALPHA_STRIDE,
        config.MATTING_INPUT_SIZE,
        config.MATTING_WARMUP_RUNS,
        config.USE_PYNV,
        config.PASSTHROUGH_VCODEC,
        config.PASSTHROUGH_HEVC_BITRATE,
        config.PASSTHROUGH_HEVC_BF,
        config.PASSTHROUGH_CONTAINER,
        config.PASSTHROUGH_SEEK_MODE,
        config.PASSTHROUGH_OUTPUT_MODE,
        config.PASSTHROUGH_MAX_CONCURRENT,
        config.PASSTHROUGH_PAD_TO_LENGTH,
        matter_device(),
        config.PASSTHROUGH_PYNV_10BIT,
        config.PASSTHROUGH_PYNV_10BIT_SHIFT,
    )

    set_startup_phase("firewall", "ensuring firewall rules")
    ensure_rules()

    set_startup_phase("ssdp", "starting SSDP")
    ssdp = SSDPServer()
    ssdp.start()

    app = create_app()
    set_startup_phase("http_starting", f"uvicorn starting on 0.0.0.0:{config.HTTP_PORT}")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=config.HTTP_PORT,
            log_level="info",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=3,
        )
    except KeyboardInterrupt:
        log.info("keyboard interrupt received")
    finally:
        set_startup_phase("shutting_down", "uvicorn stopped")
        log.info("shutting down...")
        ssdp.stop()
        stop_startup_status_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
