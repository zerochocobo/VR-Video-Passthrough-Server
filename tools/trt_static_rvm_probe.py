from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from utils.rvm_static_onnx import make_static_rvm_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_env(cache_dir: Path) -> None:
    os.environ["PT_MATTING_MODEL_KIND"] = "rvm"
    os.environ["PT_MATTING_INPUT_SIZE"] = "1024"
    os.environ["PT_RVM_DOWNSAMPLE_RATIO"] = "0.5"
    os.environ["PT_MATTING_WARMUP_RUNS"] = "0"
    os.environ["PT_ONNX_PROVIDERS"] = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
    os.environ["PT_ONNX_TRT_ENGINE_CACHE_PATH"] = str(cache_dir.resolve())
    os.environ["PT_ONNX_TRT_ENGINE_CACHE_ENABLE"] = "1"
    os.environ["PT_ONNX_TRT_FP16_ENABLE"] = "0"
    os.environ["PT_ONNX_TRT_CUDA_GRAPH_ENABLE"] = "0"


def _clean_cache(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _run_probe(model_path: Path, batch: int, h: int, w: int) -> None:
    from utils.runtime_dll_paths import apply_runtime_dll_paths

    apply_runtime_dll_paths()
    from pipeline.matting import Matter

    matter = Matter(model_path=model_path)
    print(f"providers={matter.sess.get_providers()} model={model_path}", flush=True)
    feed = {
        matter.input_name: np.zeros((batch, 3, h, w), dtype=matter.input_dtype),
    }
    for name in matter.input_names[1:5]:
        feed[name] = np.zeros(matter._rvm_initial_state_shape(name, batch, h, w), dtype=matter.input_dtype)
    if len(matter.input_names) >= 6:
        feed[matter.input_names[5]] = np.asarray([0.5], dtype=matter.rvm_downsample_dtype)
    for idx in range(5):
        t0 = time.perf_counter()
        outputs = matter.sess.run(matter.output_names, feed)
        print(f"run{idx + 1}_elapsed_ms={(time.perf_counter() - t0) * 1000.0:.1f}", flush=True)
    print([getattr(output, "shape", None) for output in outputs], flush=True)

    binding = matter.sess.io_binding()
    import onnxruntime as ort

    for name, value in feed.items():
        binding.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), "cuda", 0))
    for meta in matter.output_metas:
        binding.bind_output(meta.name, "cuda", 0)
    for idx in range(5):
        t0 = time.perf_counter()
        matter.sess.run_with_iobinding(binding)
        print(f"iobinding{idx + 1}_elapsed_ms={(time.perf_counter() - t0) * 1000.0:.1f}", flush=True)
    print([(output.device_name(), output.shape()) for output in binding.get_outputs()], flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "models" / "rvm_mobilenetv3_fp32.onnx")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runtime_cache" / "trt_static_rvm_probe")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--h", type=int, default=1024)
    parser.add_argument("--w", type=int, default=1024)
    parser.add_argument("--downsample", type=float, default=0.5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    _clean_cache(args.cache_dir)
    target = args.cache_dir / f"{args.source.stem}_static_b{args.batch}_{args.h}x{args.w}_ds{args.downsample:g}.onnx"
    make_static_rvm_model(args.source, target, args.batch, args.h, args.w, args.downsample)
    _configure_env(args.cache_dir)
    _run_probe(target, args.batch, args.h, args.w)
    print(f"cache_dir={args.cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
