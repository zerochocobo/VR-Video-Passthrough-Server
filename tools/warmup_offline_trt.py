from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    start = time.perf_counter()
    os.environ["PT_MATTING_MODEL_KIND"] = "rvm"
    os.environ["PT_MATTING_WARMUP_RUNS"] = "0"
    os.environ["PT_ONNX_PROVIDERS"] = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
    os.environ["PT_ONNX_TRT_FP16_ENABLE"] = "0"
    os.environ["PT_ONNX_TRT_CUDA_GRAPH_ENABLE"] = "0"

    import config

    config.MATTING_MODEL_KIND = "rvm"
    config.MATTING_WARMUP_RUNS = 0
    config.ONNX_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    config.ONNX_TRT_FP16_ENABLE = False
    config.ONNX_TRT_CUDA_GRAPH_ENABLE = False

    from ui.services.process_helpers import base_environment
    from ui.services.trt_warmup_process import (
        _clean_cache_dir,
        _copy_rvm_shared_1024_artifacts,
        _engine_entries,
        _prune_unusable_engine_artifacts,
        _rvm_shared_1024_artifacts_available,
        _run_static_shape,
        _shape_inferred_model_path,
        _static_model_path,
    )
    from utils.gpu_runtime_cache import configure_gpu_runtime_cache
    from utils.trt_manifest import (
        RVM_OFFLINE_TRT_SHAPES,
        TRT_MODEL_RVM,
        build_manifest,
        cache_dir_for_model,
        cache_status,
        collect_fingerprint,
        original_rvm_model_path,
        save_manifest,
    )

    os.environ.update(base_environment())
    configure_gpu_runtime_cache()

    cache_dir = cache_dir_for_model(TRT_MODEL_RVM, scope="offline")
    config.ONNX_TRT_ENGINE_CACHE_PATH = cache_dir
    os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(cache_dir)
    source_model = original_rvm_model_path()
    _clean_cache_dir(cache_dir)
    realtime_cache_dir = cache_dir.parent if cache_dir.name == "offline" else cache_dir_for_model(TRT_MODEL_RVM)
    previous_input_size = int(getattr(config, "MATTING_INPUT_SIZE", 1024))
    previous_downsample = float(getattr(config, "RVM_DOWNSAMPLE_RATIO", 0.5))
    config.MATTING_INPUT_SIZE = 1024
    config.RVM_DOWNSAMPLE_RATIO = 0.5
    try:
        realtime_ready = cache_status(model_key=TRT_MODEL_RVM, cache_dir=realtime_cache_dir) == "ready"
    finally:
        config.MATTING_INPUT_SIZE = previous_input_size
        config.RVM_DOWNSAMPLE_RATIO = previous_downsample
    if realtime_ready:
        copied = _copy_rvm_shared_1024_artifacts(realtime_cache_dir, cache_dir, source_model)
        if copied:
            print(f"INFO:Reused {copied} realtime TensorRT cache artifacts for offline 1024", flush=True)
    _shape_inferred_model_path(source_model, cache_dir)

    built_shapes: list[dict[str, object]] = []
    for index, (input_size, batch, downsample) in enumerate(RVM_OFFLINE_TRT_SHAPES, 1):
        shared_1024 = input_size == 1024 and downsample == 0.5
        print(
            f"STAGE:{index}:start:Building offline RVM TensorRT input={input_size} batch={batch} downsample={downsample}",
            flush=True,
        )
        config.MATTING_INPUT_SIZE = int(input_size)
        config.RVM_DOWNSAMPLE_RATIO = float(downsample)
        static_path = _static_model_path(source_model, cache_dir, batch, input_size, downsample)
        shape_start = time.perf_counter()
        if shared_1024 and _rvm_shared_1024_artifacts_available(cache_dir, source_model):
            print("INFO:Offline RVM 1024 cache already available; skipping TensorRT build", flush=True)
        else:
            _run_static_shape(static_path, batch, input_size, downsample)
        built_shapes.append(
            {
                "input_size": input_size,
                "batch": batch,
                "downsample_ratio": downsample,
                "static_model": static_path.name,
                "seconds": round(time.perf_counter() - shape_start, 1),
            }
        )
        print(f"STAGE:{index}:done:{int(round(time.perf_counter() - shape_start))}", flush=True)

    _prune_unusable_engine_artifacts(cache_dir)
    engines = _engine_entries(cache_dir)
    if not engines:
        raise RuntimeError("offline TensorRT warmup did not produce engine files")

    config.MATTING_INPUT_SIZE = 2048
    config.RVM_DOWNSAMPLE_RATIO = 0.25
    fingerprint = collect_fingerprint(TRT_MODEL_RVM, source_model)
    fingerprint["offline_precision_tiers"] = True
    fingerprint["offline_shapes"] = built_shapes
    marker = {
        "version": 1,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shapes": built_shapes,
        "engines": engines,
    }
    (cache_dir / "offline_trt_engines.marker.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = build_manifest(fingerprint, engines, time.perf_counter() - start, model_key=TRT_MODEL_RVM)
    save_manifest(manifest, model_key=TRT_MODEL_RVM, cache_dir=cache_dir)
    print(f"DONE:total_seconds={int(round(time.perf_counter() - start))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
