"""Application entry point.

Startup order:
1. Configure runtime caches and optional GPU warmup.
2. Start SSDP so DLNA clients can discover the server.
3. Start FastAPI/uvicorn for device descriptions, SOAP, and media streams.
"""
from __future__ import annotations

import argparse
import os
import runpy
import threading
import sys
import time

import config
from utils.trt_manifest import (
    TRT_PROVIDER_CHAIN,
    cache_status,
    collect_fingerprint,
    load_manifest,
    stale_reasons,
    trt_runtime_model_path,
)
from utils.gpu_runtime_cache import (
    configure_gpu_runtime_cache,
    predict_warmup_state,
    provider_kind_from_config,
    startup_warmup_step_total,
    warmup_gpu_runtime_cache,
)
from utils.runtime_dll_paths import apply_runtime_dll_paths
from utils.gpu_requirements import (
    MIN_NVIDIA_COMPUTE_CAPABILITY,
    parse_compute_capability,
    unsupported_gpu_message,
    GpuRequirementResult,
)
from utils.logger import get, setup
from utils.startup_status import (
    get_startup_state,
    set_startup_phase,
    start_heartbeat,
    stop_heartbeat,
    start_startup_status_server,
    stop_startup_status_server,
)


class _StartupReadySignal:
    def __init__(self, step_total: int) -> None:
        self.done = threading.Event()
        self.step_total = step_total

    async def __call__(self) -> None:
        set_startup_phase(
            "listening",
            "server ready",
            step="listening",
            step_index=self.step_total,
            step_total=self.step_total,
            progress=1.0,
            eta_sec=0.0,
        )
        self.done.set()


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


def _force_line_buffered_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def _validate_tensorrt_provider(log) -> bool:
    providers = [p.strip() for p in config.ONNX_PROVIDERS if p.strip()]
    if "TensorrtExecutionProvider" not in providers:
        return False
    try:
        actual_fp = collect_fingerprint()
        manifest = load_manifest()
        status = cache_status(actual_fp=actual_fp, manifest=manifest)
    except Exception as exc:
        status = "failed"
        manifest = None
        actual_fp = {}
        log.warning("trt cache validation failed: %s; falling back to CUDA EP", exc)
    if status == "ready":
        config.ONNX_PROVIDERS = [p.strip() for p in TRT_PROVIDER_CHAIN.split(",") if p.strip()]
        runtime_model = trt_runtime_model_path()
        if runtime_model.exists() and runtime_model != config.MODEL_PATH:
            config.MODEL_PATH = runtime_model.resolve()
        log.info("trt cache ready; ONNX providers=%s runtime_model=%s", config.ONNX_PROVIDERS, config.MODEL_PATH)
        return True
    reasons: list[str] = []
    if status == "stale" and isinstance(manifest, dict):
        saved_fp = manifest.get("fingerprint")
        if isinstance(saved_fp, dict) and actual_fp:
            reasons = stale_reasons(saved_fp, actual_fp)
    config.ONNX_PROVIDERS = [p for p in providers if p != "TensorrtExecutionProvider"]
    if not config.ONNX_PROVIDERS:
        config.ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    reason_text = "; ".join(reasons) if reasons else status
    log.warning("trt cache invalid (%s); falling back to ONNX providers=%s", reason_text, config.ONNX_PROVIDERS)
    return False


def _passthrough_mode_enabled(mode: str) -> bool:
    raw = str(config.PASSTHROUGH_OUTPUT_MODE or "none").strip().lower()
    if raw == "all":
        return mode in {"green", "alpha"}
    return mode in {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}


def _warmup_da3_trt_if_needed(log, *, step_total: int, provider_kind: str) -> None:
    if not _passthrough_mode_enabled("two_dvr"):
        return
    try:
        import onnxruntime as ort
    except Exception as exc:
        log.warning("DA3 TensorRT startup warmup skipped: onnxruntime unavailable: %s", exc)
        return
    if "TensorrtExecutionProvider" not in set(ort.get_available_providers()):
        log.warning("DA3 TensorRT startup warmup skipped: TensorRT EP unavailable")
        return
    from offline.da3_depth import default_model_path, normalize_model, trt_engine_cached, warmup_depth_engine

    variant = normalize_model(config.TWO_DVR_MODEL)
    cache_present = trt_engine_cached(variant)
    message = "warming DA3 realtime engine" if cache_present else "building DA3 TensorRT cache"
    set_startup_phase(
        "warming",
        message,
        step="da3_trt_warmup",
        step_index=max(0, step_total - 1),
        step_total=step_total,
        progress=0.92,
        provider_kind=provider_kind,
        detail=variant,
    )
    start_heartbeat(
        eta_sec=30.0 if cache_present else 120.0,
        baseline_progress=0.92,
        ceiling_progress=0.96,
    )
    try:
        path = default_model_path(variant)
        if not path.is_file():
            log.warning("DA3 TensorRT startup warmup skipped missing %s model: %s", variant, path)
            return
        t0 = time.perf_counter()
        set_startup_phase(
            "warming",
            f"warming DA3 {variant} realtime engine",
            step="da3_trt_warmup",
            step_index=max(0, step_total - 1),
            step_total=step_total,
            progress=0.96,
            provider_kind=provider_kind,
            detail=variant,
            monotonic_progress=True,
        )
        log.info(
            "DA3 TensorRT startup warmup begin: variant=%s model=%s cache_present=%s",
            variant,
            path,
            cache_present,
        )
        engine = warmup_depth_engine(
            variant=variant,
            provider="trt",
            log=lambda msg: log.info("%s", msg),
        )
        log.info(
            "DA3 TensorRT startup warmup ready: variant=%s provider=%s elapsed=%.1fs cached=%s retained=True",
            variant,
            engine.providers[0] if engine.providers else "unknown",
            time.perf_counter() - t0,
            trt_engine_cached(variant),
        )
    finally:
        stop_heartbeat()


def _warmup_da3_trt_nonfatal(log, *, step_total: int, provider_kind: str) -> None:
    try:
        _warmup_da3_trt_if_needed(log, step_total=step_total, provider_kind=provider_kind)
    except Exception as e:
        set_startup_phase(
            "warmed",
            "DA3 realtime warmup failed; continuing without preloaded 2D->3D engine",
            step="da3_trt_warning",
            step_index=max(0, step_total - 1),
            step_total=step_total,
            progress=0.94,
            detail=str(e),
            provider_kind=provider_kind,
            monotonic_progress=True,
        )
        log.warning(
            "DA3 TensorRT startup warmup failed; continuing without prewarmed 2D->3D engine: %s",
            e,
            exc_info=True,
        )


def _run_legacy_tool(tool_main, tool_name: str, tool_args: list[str]) -> int:
    _force_line_buffered_stdio()
    original_argv = sys.argv[:]
    try:
        sys.argv = [tool_name, *tool_args]
        return tool_main()
    finally:
        sys.argv = original_argv


def _run_internal_python_module(argv: list[str] | None) -> int | None:
    """Handle subprocess probes that call the frozen exe like python -m."""
    if not argv:
        return None

    args = list(argv)
    if args[0] == "-m":
        if len(args) < 2:
            return 2
        module_name = args[1]
        module_args = args[2:]
    else:
        module_name = args[0]
        module_args = args[1:]

    allowed_prefixes = ("cuda.pathfinder.",)
    if not module_name.startswith(allowed_prefixes):
        return None

    _force_line_buffered_stdio()
    original_argv = sys.argv[:]
    try:
        sys.argv = [module_name, *module_args]
        try:
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            return 1
        return 0
    finally:
        sys.argv = original_argv


def main(argv: list[str] | None = None) -> int:
    """Start the DLNA media server process."""

    internal_module_result = _run_internal_python_module(argv)
    if internal_module_result is not None:
        return internal_module_result

    if argv and argv[0] == "offline":
        _force_line_buffered_stdio()
        from offline.convert import main as offline_main

        return offline_main(argv[1:])
    if argv and argv[0] == "two_dvr":
        _force_line_buffered_stdio()
        from offline.two_dvr import main as two_dvr_main

        return two_dvr_main(argv[1:])
    if argv and argv[0] == "trt_warmup":
        _force_line_buffered_stdio()
        from ui.services.trt_warmup_process import main as trt_warmup_main

        return trt_warmup_main(argv[1:])
    if argv and argv[0] == "tool":
        if len(argv) < 2:
            raise SystemExit("tool name required")
        tool_name = argv[1]
        tool_args = argv[2:]
        if tool_name == "offline_passthrough":
            from tools.offline_passthrough import main as tool_main
        elif tool_name == "offline_alpha_passthrough":
            from tools.offline_alpha_passthrough import main as tool_main
        elif tool_name == "warmup_offline_trt":
            from tools.warmup_offline_trt import main as tool_main
        else:
            raise SystemExit(f"unknown tool: {tool_name}")
        return _run_legacy_tool(tool_main, tool_name, tool_args)

    args = _parse_args(argv)
    _apply_debug_arg(args)
    apply_runtime_dll_paths()
    cache_env = configure_gpu_runtime_cache()
    setup()
    log = get("main")
    start_startup_status_server(config.STARTUP_STATUS_PORT)
    set_startup_phase("starting", "process started")
    log.info("LAN_IP=%s HTTP_PORT=%d UUID=%s", config.LAN_IP, config.HTTP_PORT, config.DEVICE_UUID)
    log.info("VIDEO_DIRS=%s", "|".join(str(path) for path in config.VIDEO_DIRS))
    log.info("MODEL_PATH=%s (exists=%s)", config.MODEL_PATH, config.MODEL_PATH.exists())
    log.info("ONNX providers requested=%s env=%s", config.ONNX_PROVIDERS, os.environ.get("PT_ONNX_PROVIDERS"))
    log.info("GPU_RUNTIME_CACHE=%s", cache_env)
    _validate_tensorrt_provider(log)
    log.info("ONNX providers active_after_validation=%s MODEL_PATH=%s", config.ONNX_PROVIDERS, config.MODEL_PATH)
    provider_kind = provider_kind_from_config()
    startup_step_total = startup_warmup_step_total(provider_kind)
    nvenc_step_enabled = bool(config.USE_PYNV and config.NVENC_PREFLIGHT_ENABLE)
    if config.STARTUP_GPU_WARMUP:
        # Publish a prediction first so the UI can show the expected duration
        # before any heavy CUDA work begins. Failure to predict is non-fatal.
        try:
            set_startup_phase(
                "warming",
                "detecting GPU and ORT versions",
                step="predict_probe",
                step_index=0,
                step_total=0,
                progress=0.02,
                provider_kind=provider_kind,
                monotonic_progress=True,
            )
            prediction = predict_warmup_state()
            provider_kind = prediction.provider_kind or provider_kind_from_config()
            startup_step_total = startup_warmup_step_total(provider_kind)
            nvenc_step_enabled = bool(config.USE_PYNV and config.NVENC_PREFLIGHT_ENABLE)
            set_startup_phase(
                "warming",
                ("first-time GPU initialization" if prediction.cold else "verifying GPU cache"),
                step="predict",
                step_index=0,
                step_total=startup_step_total,
                progress=0.0,
                eta_sec=prediction.estimate_sec,
                elapsed_sec=0.0,
                cold=prediction.cold,
                is_known_slow=prediction.is_known_slow,
                gpu_name=prediction.gpu_name,
                compute_capability=prediction.compute_capability,
                driver_version=prediction.driver_version,
                onnxruntime_version=prediction.onnxruntime_version,
                provider_kind=provider_kind,
                reason=prediction.reason,
                detail=prediction.detail,
            )
            start_heartbeat(
                prediction.estimate_sec,
                baseline_progress=0.1,
                ceiling_progress=0.95,
            )
            log.info(
                "warmup prediction: cold=%s reason=%s known_slow=%s provider=%s eta=%.1fs gpu=%s cc=%s ort=%s",
                prediction.cold,
                prediction.reason,
                prediction.is_known_slow,
                provider_kind,
                prediction.estimate_sec,
                prediction.gpu_name,
                prediction.compute_capability,
                prediction.onnxruntime_version,
            )
            cc = parse_compute_capability(prediction.compute_capability)
            if cc is not None and cc < MIN_NVIDIA_COMPUTE_CAPABILITY:
                stop_heartbeat()
                message = unsupported_gpu_message(
                    GpuRequirementResult(
                        detected=True,
                        supported=False,
                        name=prediction.gpu_name,
                        compute_capability=prediction.compute_capability,
                    )
                )
                set_startup_phase(
                    "failed",
                    message,
                    step="gpu_requirement",
                    step_index=0,
                    step_total=startup_step_total,
                    progress=0.0,
                    eta_sec=0.0,
                    cold=prediction.cold,
                    is_known_slow=prediction.is_known_slow,
                    gpu_name=prediction.gpu_name,
                    compute_capability=prediction.compute_capability,
                    driver_version=prediction.driver_version,
                    onnxruntime_version=prediction.onnxruntime_version,
                    provider_kind=provider_kind,
                    reason="unsupported_gpu",
                    detail=message,
                )
                log.error(message)
                time.sleep(0.8)
                stop_startup_status_server()
                return 1
        except Exception as e:
            log.warning("warmup prediction failed (non-fatal): %s", e)
            set_startup_phase(
                "warming",
                "GPU runtime warmup",
                step="predict",
                step_index=0,
                step_total=startup_step_total,
                progress=0.0,
                provider_kind=provider_kind,
                detail=str(e),
            )

        log.info(
            "startup GPU warmup begin: force=%s timeout=%.1fs runs_per_shape=%d",
            config.STARTUP_GPU_WARMUP_FORCE,
            config.STARTUP_GPU_WARMUP_TIMEOUT,
            config.STARTUP_GPU_WARMUP_RUNS_PER_SHAPE,
        )
        warmup_start = time.perf_counter()
        try:
            stop_heartbeat()
            set_startup_phase(
                "warming",
                "starting GPU warmup",
                step="warmup_start",
                step_index=0,
                step_total=0,
                progress=0.1,
                provider_kind=provider_kind,
                monotonic_progress=True,
            )
            marker = warmup_gpu_runtime_cache(
                force=config.STARTUP_GPU_WARMUP_FORCE,
                timeout_sec=max(1.0, config.STARTUP_GPU_WARMUP_TIMEOUT),
                runs_per_shape=max(1, config.STARTUP_GPU_WARMUP_RUNS_PER_SHAPE),
            )
        except Exception as e:
            stop_heartbeat()
            set_startup_phase(
                "failed",
                f"startup GPU warmup failed: {e}",
                step="failed",
                progress=0.0,
                detail=str(e),
                provider_kind=provider_kind,
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
        stop_heartbeat()
        warmup_elapsed = time.perf_counter() - warmup_start
        warmup_status = get_startup_state()
        provider_kind = str(warmup_status.get("provider_kind") or provider_kind)
        try:
            startup_step_total = int(warmup_status.get("step_total") or startup_step_total)
        except (TypeError, ValueError):
            pass
        warmup_done_step = startup_step_total - (1 if nvenc_step_enabled else 0)
        set_startup_phase(
            "warmed",
            "GPU runtime warmup complete",
            step="warmed",
            step_index=warmup_done_step,
            step_total=startup_step_total,
            progress=warmup_done_step / startup_step_total,
            eta_sec=0.0,
            elapsed_sec=warmup_elapsed,
            provider_kind=provider_kind,
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
        stop_heartbeat()
        warmup_done_step = startup_step_total - (1 if nvenc_step_enabled else 0)
        set_startup_phase(
            "warmed",
            "startup GPU warmup disabled",
            step="warmed",
            step_index=warmup_done_step,
            step_total=startup_step_total,
            progress=warmup_done_step / startup_step_total,
            eta_sec=0.0,
            provider_kind=provider_kind,
        )
        log.info("startup GPU warmup disabled")
    _warmup_da3_trt_nonfatal(log, step_total=startup_step_total, provider_kind=provider_kind)
    if nvenc_step_enabled:
        set_startup_phase(
            "warming",
            "warming NVENC encoder",
            step="nvenc_preflight",
            step_index=startup_step_total,
            step_total=startup_step_total,
            progress=0.95,
            provider_kind=provider_kind,
        )
        try:
            from pipeline.pynv_stream import PyNvPassthroughStream

            PyNvPassthroughStream.startup_preflight()
        except Exception as e:
            log.warning("nvenc startup preflight failed; first request will pay it lazily: %s", e, exc_info=True)
    from pipeline.matting import configure_matter_pool, matter_device

    configure_matter_pool(config.PASSTHROUGH_MAX_CONCURRENT)
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

    set_startup_phase(
        "firewall",
        "ensuring firewall rules",
        step="firewall",
        step_index=startup_step_total,
        step_total=startup_step_total,
        progress=0.97,
        provider_kind=provider_kind,
    )
    from utils.firewall import ensure_rules

    ensure_rules()

    set_startup_phase(
        "ssdp",
        "starting SSDP",
        step="ssdp",
        step_index=startup_step_total,
        step_total=startup_step_total,
        progress=0.98,
        provider_kind=provider_kind,
    )
    from dlna.ssdp import SSDPServer

    ssdp = SSDPServer()
    ssdp.start()

    from http_app.server import create_app

    ready_signal = _StartupReadySignal(startup_step_total)
    app = create_app(startup_hook=ready_signal)
    set_startup_phase(
        "http_starting",
        f"uvicorn starting on 0.0.0.0:{config.HTTP_PORT}",
        step="http_starting",
        step_index=startup_step_total,
        step_total=startup_step_total,
        progress=0.99,
        provider_kind=provider_kind,
    )
    try:
        import uvicorn

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
        if ready_signal.done.is_set():
            set_startup_phase("shutting_down", "uvicorn stopped")
        log.info("shutting down...")
        ssdp.stop()
        stop_startup_status_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
