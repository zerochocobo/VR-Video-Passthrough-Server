from __future__ import annotations

import argparse
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np


def _print_event(line: str) -> None:
    print(line, flush=True)


def _engine_entries(cache_dir: Path, recursive: bool = False) -> list[dict]:
    from utils.trt_manifest import engine_artifact_paths

    entries: list[dict] = []
    cache_files = engine_artifact_paths(cache_dir, recursive=recursive)
    for path in sorted(cache_files):
        try:
            size_mb = round(path.stat().st_size / (1024 * 1024), 1)
        except OSError:
            size_mb = 0.0
        shape = path.stem
        if recursive:
            try:
                shape = str(path.relative_to(cache_dir).with_suffix(""))
            except ValueError:
                pass
        entries.append({"shape": shape, "size_mb": size_mb, "built_at": _utc_now()})
    return entries


def _prune_unusable_engine_artifacts(cache_dir: Path) -> None:
    from utils.trt_manifest import is_engine_artifact

    if not cache_dir.exists():
        return
    for path in list(cache_dir.iterdir()):
        if path.suffix.lower() != ".engine" or is_engine_artifact(path):
            continue
        for stale in (path, path.with_suffix(".profile")):
            try:
                if stale.exists():
                    stale.unlink()
            except OSError:
                pass


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_cache_dir(cache_dir: Path, preserve_names: set[str] | None = None) -> None:
    preserve_names = preserve_names or set()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in cache_dir.iterdir():
        if path.name == "build.log" or path.name in preserve_names:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass


def _copy_file_if_usable(source: Path, target: Path) -> bool:
    try:
        if not source.is_file() or source.stat().st_size <= 0:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == source.stat().st_size:
            return False
        shutil.copy2(source, target)
        return True
    except OSError:
        return False


def _copy_rvm_shared_1024_artifacts(source_cache: Path, target_cache: Path, source_model: Path) -> int:
    from utils.rvm_static_onnx import static_rvm_model_path
    from utils.trt_manifest import is_engine_artifact, shape_inferred_model_path

    if not source_cache.exists():
        return 0
    required = [
        shape_inferred_model_path(source_model, source_cache),
        static_rvm_model_path(source_model, source_cache, 1, 1024, 0.5),
        static_rvm_model_path(source_model, source_cache, 2, 1024, 0.5),
    ]
    engines = [path for path in source_cache.iterdir() if is_engine_artifact(path)]
    if not all(path.is_file() for path in required) or not engines:
        return 0

    copied = 0
    for source in required:
        if _copy_file_if_usable(source, target_cache / source.name):
            copied += 1
    for engine in engines:
        if _copy_file_if_usable(engine, target_cache / engine.name):
            copied += 1
        profile = engine.with_suffix(".profile")
        if profile.exists() and _copy_file_if_usable(profile, target_cache / profile.name):
            copied += 1
    return copied


def _rvm_shared_1024_artifacts_available(cache_dir: Path, source_model: Path) -> bool:
    from utils.rvm_static_onnx import static_rvm_model_path
    from utils.trt_manifest import is_engine_artifact, shape_inferred_model_path

    if not cache_dir.exists():
        return False
    required = [
        shape_inferred_model_path(source_model, cache_dir),
        static_rvm_model_path(source_model, cache_dir, 1, 1024, 0.5),
        static_rvm_model_path(source_model, cache_dir, 2, 1024, 0.5),
    ]
    return all(path.is_file() for path in required) and any(is_engine_artifact(path) for path in cache_dir.iterdir())


def _shape_inferred_model_path(model_path: Path, cache_dir: Path) -> Path:
    import onnx
    from onnx import shape_inference
    from utils.trt_manifest import shape_inferred_model_path

    target = shape_inferred_model_path(model_path, cache_dir)
    if target.exists():
        try:
            src_stat = model_path.stat()
            dst_stat = target.stat()
            if dst_stat.st_mtime >= src_stat.st_mtime and dst_stat.st_size > 0:
                return target
        except OSError:
            pass
    _print_event("INFO:Running ONNX shape inference for TensorRT")
    model = onnx.load(str(model_path))
    _make_rvm_state_dims_unique(model)
    inferred = shape_inference.infer_shapes(model)
    onnx.save(inferred, str(target))
    return target


def _static_model_path(model_path: Path, cache_dir: Path, batch: int, input_size: int, downsample: float) -> Path:
    from utils.rvm_static_onnx import make_static_rvm_model, static_rvm_model_path

    target = static_rvm_model_path(model_path, cache_dir, batch, input_size, downsample)
    if target.exists():
        try:
            src_stat = model_path.stat()
            dst_stat = target.stat()
            if dst_stat.st_mtime >= src_stat.st_mtime and dst_stat.st_size > 0:
                return target
        except OSError:
            pass
    _print_event(f"INFO:Generating static TensorRT RVM ONNX batch={batch}")
    return make_static_rvm_model(model_path, target, batch, input_size, input_size, downsample)


def _make_rvm_state_dims_unique(model) -> None:
    """Avoid TensorRT treating RVM state tensors as the same H/W as src."""

    def _rename(value_info, prefix: str) -> None:
        dims = value_info.type.tensor_type.shape.dim
        names = ("batch", "channels", "height", "width")
        for idx, dim in enumerate(dims):
            if dim.dim_param:
                dim.dim_param = f"{prefix}_{names[idx] if idx < len(names) else idx}"

    for value_info in model.graph.input:
        if value_info.name in {"r1i", "r2i", "r3i", "r4i"}:
            _rename(value_info, value_info.name)
    for value_info in model.graph.output:
        if value_info.name in {"r1o", "r2o", "r3o", "r4o"}:
            _rename(value_info, value_info.name)


def _run_shape(matter, batch: int, input_size: int) -> None:
    matter.reset_state()
    frame = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    if batch <= 1:
        matter.alpha(frame)
        _require_tensorrt_still_active(matter)
        return
    if not getattr(matter, "_supports_batch2", False):
        return
    sbs = np.zeros((input_size, input_size * 2, 3), dtype=np.uint8)
    matter.alpha(sbs)
    _require_tensorrt_still_active(matter)


def _run_static_shape(model_path: Path, batch: int, input_size: int, downsample: float) -> None:
    import onnxruntime as ort
    from pipeline.matting import _filter_available_providers, _provider_config

    providers = _filter_available_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=_provider_config(providers))
    providers = list(session.get_providers())
    if "TensorrtExecutionProvider" not in providers:
        raise RuntimeError(f"TensorRT provider did not activate; active providers={providers}")
    state_shapes = {
        "r1i": (batch, 16, max(1, int(round(input_size * downsample / 2))), max(1, int(round(input_size * downsample / 2)))),
        "r2i": (batch, 20, max(1, int(round(input_size * downsample / 4))), max(1, int(round(input_size * downsample / 4)))),
        "r3i": (batch, 40, max(1, int(round(input_size * downsample / 8))), max(1, int(round(input_size * downsample / 8)))),
        "r4i": (batch, 64, max(1, int(round(input_size * downsample / 16))), max(1, int(round(input_size * downsample / 16)))),
    }
    feed = {"src": np.zeros((batch, 3, input_size, input_size), dtype=np.float32)}
    for name in [meta.name for meta in session.get_inputs()[1:5]]:
        feed[name] = np.zeros(state_shapes[name], dtype=np.float32)
    session.run([meta.name for meta in session.get_outputs()], feed)


def _run_matanyone2_step_update(model_path: Path) -> None:
    import onnxruntime as ort
    from pipeline.matting import _filter_available_providers, _provider_config

    providers = _filter_available_providers(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=_provider_config(providers))
    active = list(session.get_providers())
    if "TensorrtExecutionProvider" not in active:
        raise RuntimeError(f"TensorRT provider did not activate; active providers={active}")
    feed = {}
    for meta in session.get_inputs():
        dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
        shape = [int(dim) for dim in meta.shape]
        feed[meta.name] = np.zeros(shape, dtype=dtype)
    session.run([meta.name for meta in session.get_outputs()], feed)
    active = list(session.get_providers())
    if "TensorrtExecutionProvider" not in active:
        raise RuntimeError(f"TensorRT provider fell back during warmup; active providers={active}")


def _run_matanyone2_step_update_isolated(model_key: str, model_path: Path, cache_dir: Path) -> None:
    from ui.services.process_helpers import base_environment, trt_warmup_command
    from utils.subprocess_hidden import run_hidden_streaming

    exe, base_args = trt_warmup_command()
    env = base_environment(
        {
            "PT_MATTING_MODEL_KIND": "matanyone2",
            "PT_MATTING_WARMUP_RUNS": "0",
            "PT_ONNX_PROVIDERS": "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
            "PT_ONNX_TRT_FP16_ENABLE": "0",
            "PT_ONNX_TRT_CUDA_GRAPH_ENABLE": "0",
            "PT_ONNX_TRT_ENGINE_CACHE_PATH": str(cache_dir.resolve()),
        }
    )
    rc = run_hidden_streaming(
        [
            exe,
            *base_args,
            "--model",
            "matanyone2",
            "--matanyone2-model-key",
            model_key,
            "--cache-dir",
            str(cache_dir.resolve()),
            "--fp16",
            "0",
            "--cuda-graph",
            "0",
        ],
        env=env,
        exit_label=f"matanyone2-trt-{model_key}",
    )
    if rc != 0:
        raise RuntimeError(f"MatAnyone2 TensorRT subprocess failed for {model_key}: exit code {rc}")


def _require_tensorrt_active(matter) -> None:
    providers = list(getattr(matter.sess, "get_providers")())
    if "TensorrtExecutionProvider" not in providers:
        raise RuntimeError(f"TensorRT provider did not activate; active providers={providers}")


def _require_tensorrt_still_active(matter) -> None:
    providers = list(getattr(matter.sess, "get_providers")())
    if "TensorrtExecutionProvider" not in providers:
        raise RuntimeError(f"TensorRT provider fell back during warmup; active providers={providers}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT engine cache.")
    parser.add_argument("--model", default="rvm", choices=["rvm", "matanyone2"])
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--downsample", type=float, default=0.5)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--cuda-graph", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--progress-stdout", action="store_true")
    parser.add_argument("--matanyone2-model-key", default="", choices=["", "matanyone2_onnx_512_bs1", "matanyone2_onnx_1024_bs1"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start = time.perf_counter()
    try:
        os.environ["PT_MATTING_MODEL_KIND"] = "rvm" if args.model == "rvm" else "matanyone2"
        os.environ["PT_MATTING_INPUT_SIZE"] = str(int(args.input_size))
        os.environ["PT_RVM_DOWNSAMPLE_RATIO"] = str(float(args.downsample))
        os.environ["PT_MATTING_WARMUP_RUNS"] = "0"
        os.environ["PT_ONNX_PROVIDERS"] = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
        os.environ["PT_ONNX_TRT_FP16_ENABLE"] = "1" if args.fp16 else "0"
        os.environ["PT_ONNX_TRT_CUDA_GRAPH_ENABLE"] = "1" if args.cuda_graph else "0"
        if args.cache_dir is not None:
            os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(args.cache_dir.resolve())

        import config

        config.MATTING_MODEL_KIND = "rvm" if args.model == "rvm" else "matanyone2"
        config.MATTING_INPUT_SIZE = int(args.input_size)
        config.RVM_DOWNSAMPLE_RATIO = float(args.downsample)
        config.MATTING_WARMUP_RUNS = 0
        config.ONNX_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        config.ONNX_TRT_FP16_ENABLE = bool(args.fp16)
        config.ONNX_TRT_CUDA_GRAPH_ENABLE = bool(args.cuda_graph)
        if args.cache_dir is not None:
            config.ONNX_TRT_ENGINE_CACHE_PATH = args.cache_dir.resolve()

        from ui.services.process_helpers import base_environment
        from utils.gpu_runtime_cache import configure_gpu_runtime_cache

        os.environ.update(base_environment())
        configure_gpu_runtime_cache()
        from utils.trt_manifest import (
            MATANYONE2_CACHE_KEY,
            MATANYONE2_MODEL_KEYS,
            TRT_MODEL_MATANYONE2,
            TRT_MODEL_RVM,
            build_manifest,
            cache_dir_for_model,
            collect_fingerprint,
            matanyone2_trt_cache_dir_for_key,
            matanyone2_trt_source_model_paths,
            original_rvm_model_path,
            save_manifest,
        )

        model_key = TRT_MODEL_MATANYONE2 if args.model == "matanyone2" else TRT_MODEL_RVM
        cache_dir = Path(config.ONNX_TRT_ENGINE_CACHE_PATH)
        if args.cache_dir is None:
            cache_dir = cache_dir_for_model(model_key)
            config.ONNX_TRT_ENGINE_CACHE_PATH = cache_dir
            os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(cache_dir)
        if model_key == TRT_MODEL_MATANYONE2 and args.matanyone2_model_key:
            source_model_path = matanyone2_trt_source_model_paths()[args.matanyone2_model_key]
            if not source_model_path.is_file():
                raise RuntimeError(f"MatAnyone2 TensorRT source model not found: {source_model_path}")
            _clean_cache_dir(cache_dir)
            _run_matanyone2_step_update(source_model_path)
            _prune_unusable_engine_artifacts(cache_dir)
            if not _engine_entries(cache_dir):
                raise RuntimeError(f"TensorRT warmup did not produce a usable MatAnyone2 engine cache for {args.matanyone2_model_key}")
            return 0
        preserve_names = {"offline", MATANYONE2_CACHE_KEY} if model_key == TRT_MODEL_RVM and cache_dir.name != "offline" else set()
        _clean_cache_dir(cache_dir, preserve_names=preserve_names)
        if model_key == TRT_MODEL_MATANYONE2:
            source_model_paths = matanyone2_trt_source_model_paths()
            missing = [str(path) for path in source_model_paths.values() if not path.is_file()]
            if missing:
                raise RuntimeError(f"MatAnyone2 TensorRT source model not found: {', '.join(missing)}")

            for index, model_name in enumerate(MATANYONE2_MODEL_KEYS, 1):
                source_model_path = source_model_paths[model_name]
                model_cache_dir = matanyone2_trt_cache_dir_for_key(model_name, cache_dir)
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                _print_event(f"STAGE:{index}:start:Building MatAnyone2 step-update engine {model_name}")
                stage_start = time.perf_counter()
                _run_matanyone2_step_update_isolated(model_name, source_model_path, model_cache_dir)
                _print_event(f"STAGE:{index}:done:{int(round(time.perf_counter() - stage_start))}")

            verify_stage = len(MATANYONE2_MODEL_KEYS) + 1
            _print_event(f"STAGE:{verify_stage}:start:Verifying MatAnyone2 TensorRT cache")
            stage_start = time.perf_counter()
            engines = _engine_entries(cache_dir, recursive=True)
            if not engines:
                raise RuntimeError("TensorRT warmup did not produce a usable MatAnyone2 engine cache")
            _print_event(f"STAGE:{verify_stage}:done:{int(round(time.perf_counter() - stage_start))}")

            manifest_stage = verify_stage + 1
            _print_event(f"STAGE:{manifest_stage}:start:Solidifying runtime cache")
            stage_start = time.perf_counter()
            manifest = build_manifest(collect_fingerprint(model_key), engines, time.perf_counter() - start, model_key=model_key)
            save_manifest(manifest, model_key=model_key, cache_dir=cache_dir)
            _print_event(f"STAGE:{manifest_stage}:done:{int(round(time.perf_counter() - stage_start))}")
            _print_event(f"DONE:total_seconds={int(round(time.perf_counter() - start))}")
            return 0

        source_model_path = original_rvm_model_path()
        if int(args.input_size) == 1024 and float(args.downsample) == 0.5:
            offline_cache_dir = cache_dir_for_model(TRT_MODEL_RVM, scope="offline")
            from utils.trt_manifest import cache_status

            if cache_status(model_key=TRT_MODEL_RVM, scope="offline") == "ready":
                copied = _copy_rvm_shared_1024_artifacts(offline_cache_dir, cache_dir, source_model_path)
                if copied:
                    _print_event(f"INFO:Reused {copied} offline TensorRT cache artifacts for realtime 1024")
        trt_model_path = _shape_inferred_model_path(source_model_path, cache_dir)
        static_b1_path = _static_model_path(source_model_path, cache_dir, 1, int(args.input_size), float(args.downsample))
        static_b2_path = _static_model_path(source_model_path, cache_dir, 2, int(args.input_size), float(args.downsample))

        _print_event("STAGE:1:start:Building single-eye engine")
        stage_start = time.perf_counter()
        reused_shared_1024 = int(args.input_size) == 1024 and float(args.downsample) == 0.5 and _rvm_shared_1024_artifacts_available(cache_dir, source_model_path)
        if reused_shared_1024:
            _print_event("INFO:Realtime RVM 1024 cache already available; skipping single-eye TensorRT build")
        else:
            _run_static_shape(static_b1_path, 1, int(args.input_size), float(args.downsample))
        _print_event(f"STAGE:1:done:{int(round(time.perf_counter() - stage_start))}")

        _print_event("STAGE:2:start:Building SBS dual-eye engine")
        stage_start = time.perf_counter()
        if reused_shared_1024:
            _print_event("INFO:Realtime RVM 1024 cache already available; skipping SBS TensorRT build")
        else:
            _run_static_shape(static_b2_path, 2, int(args.input_size), float(args.downsample))
        _print_event(f"STAGE:2:done:{int(round(time.perf_counter() - stage_start))}")

        _print_event("STAGE:3:start:Solidifying runtime cache")
        stage_start = time.perf_counter()
        if not trt_model_path.is_file() or not static_b1_path.is_file() or not static_b2_path.is_file():
            raise RuntimeError("TensorRT warmup did not produce required ONNX cache files")
        _prune_unusable_engine_artifacts(cache_dir)
        engines = _engine_entries(cache_dir)
        if not engines:
            raise RuntimeError("TensorRT warmup did not produce a usable engine cache")
        manifest = build_manifest(collect_fingerprint(model_key, source_model_path), engines, time.perf_counter() - start, model_key=model_key)
        save_manifest(manifest, model_key=model_key, cache_dir=cache_dir)
        _print_event(f"STAGE:3:done:{int(round(time.perf_counter() - stage_start))}")
        _print_event(f"DONE:total_seconds={int(round(time.perf_counter() - start))}")
        return 0
    except Exception as exc:
        _print_event(f"ERROR:{exc}")
        _print_event("EXIT:1")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
