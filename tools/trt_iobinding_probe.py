from __future__ import annotations

import argparse
import faulthandler
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_env() -> None:
    os.environ["PT_ONNX_PROVIDERS"] = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
    os.environ.setdefault("PT_MATTING_WARMUP_RUNS", "0")


def _make_session():
    from utils.runtime_dll_paths import apply_runtime_dll_paths

    apply_runtime_dll_paths()
    import main

    main._validate_tensorrt_provider(logging.getLogger("trt-iobinding-probe"))

    from pipeline.matting import Matter

    return Matter()


def _real_state_shape(m, name: str, batch: int, h: int, w: int) -> tuple[int, ...]:
    return m._rvm_initial_state_shape(name, batch, h, w)


def _host_feed(m, batch: int, h: int, w: int, real_state: bool) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {
        m.input_name: np.zeros((batch, 3, h, w), dtype=m.input_dtype)
    }
    for name in m.input_names[1:5]:
        shape = _real_state_shape(m, name, batch, h, w) if real_state else (batch, 1, 1, 1)
        feed[name] = np.zeros(shape, dtype=m.input_dtype)
    if len(m.input_names) >= 6:
        feed[m.input_names[5]] = np.asarray([0.5], dtype=m.rvm_downsample_dtype)
    return feed


def _cuda_ortvalue(value: np.ndarray) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), "cuda", 0)


def _bind_inputs(binding, m, feed: dict[str, np.ndarray], mode: str) -> None:
    if mode == "cpu":
        for name, value in feed.items():
            binding.bind_cpu_input(name, value)
        return
    if mode == "cuda_ort":
        for name, value in feed.items():
            binding.bind_ortvalue_input(name, _cuda_ortvalue(value))
        return
    if mode == "cuda_ptr":
        cp = __import__("cupy")
        src = cp.asarray(feed[m.input_name])
        binding.bind_input(m.input_name, "cuda", 0, m.input_dtype, tuple(src.shape), int(src.data.ptr))
        for name in m.input_names[1:]:
            binding.bind_ortvalue_input(name, _cuda_ortvalue(feed[name]))
        return
    raise ValueError(f"unknown input mode: {mode}")


def _bind_outputs(binding, m, batch: int, h: int, w: int, mode: str) -> None:
    if mode == "cpu":
        for meta in m.output_metas:
            binding.bind_output(meta.name, "cpu", 0)
        return
    if mode == "cuda_auto":
        for meta in m.output_metas:
            binding.bind_output(meta.name, "cuda", 0)
        return
    if mode == "cuda_prealloc":
        for meta in m.output_metas:
            shape = m._rvm_output_shape_for(meta.name, batch, h, w)
            if shape is None:
                binding.bind_output(meta.name, "cuda", 0)
            else:
                binding.bind_ortvalue_output(
                    meta.name,
                    ort.OrtValue.ortvalue_from_shape_and_type(shape, m.input_dtype, "cuda", 0),
                )
        return
    raise ValueError(f"unknown output mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["run", "iobinding", "matrix"], required=True)
    parser.add_argument("--input", choices=["cpu", "cuda_ort", "cuda_ptr"], default="cpu")
    parser.add_argument("--output", choices=["cpu", "cuda_auto", "cuda_prealloc"], default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--h", type=int, default=1024)
    parser.add_argument("--w", type=int, default=1024)
    parser.add_argument("--real-state", action="store_true")
    parser.add_argument("--dump-after", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--matrix-dir", type=Path, default=ROOT / "runtime_cache" / "trt_iobinding_matrix")
    args = parser.parse_args(argv)

    if args.case == "matrix":
        args.matrix_dir.mkdir(parents=True, exist_ok=True)
        cases = [
            ("run", "cpu", "cpu"),
            ("iobinding", "cpu", "cpu"),
            ("iobinding", "cpu", "cuda_auto"),
            ("iobinding", "cpu", "cuda_prealloc"),
            ("iobinding", "cuda_ort", "cpu"),
            ("iobinding", "cuda_ort", "cuda_auto"),
            ("iobinding", "cuda_ort", "cuda_prealloc"),
            ("iobinding", "cuda_ptr", "cpu"),
            ("iobinding", "cuda_ptr", "cuda_auto"),
            ("iobinding", "cuda_ptr", "cuda_prealloc"),
        ]
        for case, input_mode, output_mode in cases:
            label = f"{case}_{input_mode}_{output_mode}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                case,
                "--input",
                input_mode,
                "--output",
                output_mode,
                "--batch",
                str(args.batch),
                "--h",
                str(args.h),
                "--w",
                str(args.w),
                "--dump-after",
                str(args.dump_after),
            ]
            if args.real_state:
                cmd.append("--real-state")
            out_path = args.matrix_dir / f"{label}.out.log"
            err_path = args.matrix_dir / f"{label}.err.log"
            print(f"matrix start {label}", flush=True)
            t0 = time.perf_counter()
            with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
                try:
                    proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, timeout=args.timeout)
                    status = f"exit={proc.returncode}"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(f"matrix done {label} {status} elapsed_ms={elapsed:.1f}", flush=True)
        print(f"matrix logs={args.matrix_dir}", flush=True)
        return 0

    _configure_env()
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, repeat=True, file=sys.stderr)

    m = _make_session()
    print(f"providers={m.sess.get_providers()} model={m.model_kind}", flush=True)
    print(
        f"case={args.case} input={args.input} output={args.output} "
        f"shape=({args.batch},3,{args.h},{args.w}) real_state={args.real_state}",
        flush=True,
    )
    feed = _host_feed(m, args.batch, args.h, args.w, args.real_state)
    t0 = time.perf_counter()
    if args.case == "run":
        outputs = m.sess.run(m.output_names, feed)
        print(f"done run elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)
        print([getattr(o, "shape", None) for o in outputs], flush=True)
        return 0

    binding = m.sess.io_binding()
    print("binding inputs", flush=True)
    _bind_inputs(binding, m, feed, args.input)
    print("binding outputs", flush=True)
    _bind_outputs(binding, m, args.batch, args.h, args.w, args.output)
    print("running", flush=True)
    m.sess.run_with_iobinding(binding)
    print(f"done iobinding elapsed_ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)
    outputs = binding.get_outputs()
    print([(o.device_name(), o.shape()) for o in outputs], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
