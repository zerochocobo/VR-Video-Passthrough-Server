"""Offline passthrough conversion CLI used by the desktop UI.

The implementation delegates to the currently validated offline conversion
scripts while keeping UI-facing entry points outside tools/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import config
from utils.gpu_requirements import detect_nvidia_gpu_requirement, unsupported_gpu_message
from utils.subprocess_hidden import hidden_subprocess_kwargs, run_hidden_streaming
from utils.trt_manifest import TRT_MODEL_MATANYONE2, TRT_MODEL_RVM, TRT_PROVIDER_CHAIN, cache_dir_for_model, cache_status
from utils.video_metadata import probe_video_metadata, select_backend
from utils.vr_naming import offline_passthrough_stem

ROOT = config.ROOT
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}
_WARMUP_NOTICE_PRINTED = False

ENGINES = {
    "rvm_fast": ("rvm", ROOT / "models" / "rvm_mobilenetv3_fp32.onnx"),
    "matanyone2_medium": ("matanyone2_onnx", None),
    "matanyone2": ("matanyone2_onnx", None),
}

ENGINE_TAGS = {
    "rvm_fast": "rvm1",
    "matanyone2_medium": "matanyone2m",
    "matanyone2": "matanyone2",
}

RVM_DEFAULT_ARGS = {
    "input_size": 2048,
    "downsample_ratio": 0.25,
    "skip_frames": 0,
    "fps": 0.0,
    "bitrate": "source",
    "preset": config.PASSTHROUGH_PYNV_PRESET,
    "cq": -1,
}
MATANYONE2_DEFAULT_SIZE = 1024
MATANYONE2_SIZE_CHOICES = (512, 1024)


def _strip_tensorrt_provider(provider_text: str) -> str:
    providers = [p.strip() for p in str(provider_text or "").split(",") if p.strip()]
    providers = [p for p in providers if p != "TensorrtExecutionProvider"]
    return ",".join(providers)


def _env_flag_enabled(env: dict[str, str], key: str, default: bool = True) -> bool:
    raw = str(env.get(key, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _offline_child_env(args: argparse.Namespace, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    provider_text = env.get("PT_ONNX_PROVIDERS", "")
    wants_trt = "TensorrtExecutionProvider" in [p.strip() for p in provider_text.split(",") if p.strip()]
    if getattr(args, "engine", "") == "rvm_fast" and _env_flag_enabled(env, "PT_OFFLINE_RVM_TRT_ENABLE"):
        try:
            rvm_cache_dir = cache_dir_for_model(TRT_MODEL_RVM, scope="offline")
            if cache_status(model_key=TRT_MODEL_RVM, scope="offline") == "ready":
                env["PT_ONNX_PROVIDERS"] = TRT_PROVIDER_CHAIN
                env["PT_OFFLINE_RVM_TRT"] = "1"
                return env
        except Exception:
            pass
    if getattr(args, "engine", "") in {"matanyone2", "matanyone2_medium"} and _env_flag_enabled(env, "PT_OFFLINE_MATANYONE2_TRT_ENABLE"):
        try:
            if cache_status(model_key=TRT_MODEL_MATANYONE2) == "ready":
                env["PT_ONNX_PROVIDERS"] = TRT_PROVIDER_CHAIN
                env["PT_OFFLINE_MATANYONE2_TRT"] = "1"
                return env
        except Exception:
            pass
    if not wants_trt:
        return env
    stripped = _strip_tensorrt_provider(provider_text)
    if stripped:
        env["PT_ONNX_PROVIDERS"] = stripped
    else:
        env.pop("PT_ONNX_PROVIDERS", None)
    env.pop("PT_OFFLINE_RVM_TRT", None)
    env.pop("PT_OFFLINE_MATANYONE2_TRT", None)
    return env


def _script_for(mode: str) -> Path:
    if mode == "alpha":
        return ROOT / "tools" / "offline_alpha_passthrough.py"
    if mode == "green":
        return ROOT / "tools" / "offline_passthrough.py"
    raise ValueError(f"unsupported mode: {mode}")


def _tool_command(mode: str) -> list[str]:
    if getattr(sys, "frozen", False):
        tool = "offline_alpha_passthrough" if mode == "alpha" else "offline_passthrough"
        return [sys.executable, "tool", tool]
    return [sys.executable, str(_script_for(mode))]


def _default_out(src: Path, mode: str, width: int = 0, height: int = 0) -> Path:
    return src.with_name(f"{offline_passthrough_stem(src.stem, mode, width, height)}.mp4")


def _time_tag(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}{m:02d}{s:02d}"


def _duration_tag(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total <= 0:
        return "ALL"
    if total % 60 == 0:
        return f"{total // 60}M"
    return f"{total}S"


def _single_out(src: Path, args: argparse.Namespace, width: int = 0, height: int = 0) -> Path:
    engine_tag = ENGINE_TAGS[args.engine]
    start_tag = f"S{_time_tag(args.start)}"
    duration_tag = _duration_tag(args.duration)
    base = f"{src.stem}_{engine_tag}_{start_tag}_{duration_tag}"
    return src.with_name(f"{offline_passthrough_stem(base, args.mode, width, height)}.mp4")


def _video_files(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    out: list[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if "passthrough" in path.name.lower():
            continue
        out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


def _base_cmd(args: argparse.Namespace, src: Path, out: Path) -> list[str]:
    engine, model = ENGINES[args.engine]
    cmd = [
        *_tool_command(args.mode),
        str(src),
        "--engine",
        engine,
        "--out",
        str(out),
        "--audio",
        "copy",
    ]
    if model is not None:
        cmd.extend(["--model", str(model)])
    if args.start > 0:
        cmd.extend(["--start", str(args.start)])
    if args.duration > 0:
        cmd.extend(["--duration", str(args.duration)])
    fps = float(args.fps or 0.0)
    if fps > 0:
        cmd.extend(["--fps", str(fps)])
    cmd.extend(["--bitrate", str(args.bitrate)])
    cmd.extend(["--alpha-stride", "1"])
    if args.engine in ("matanyone2", "matanyone2_medium"):
        matanyone2_size = int(getattr(args, "matanyone2_size", MATANYONE2_DEFAULT_SIZE) or MATANYONE2_DEFAULT_SIZE)
        if matanyone2_size not in MATANYONE2_SIZE_CHOICES:
            supported = ", ".join(str(size) for size in MATANYONE2_SIZE_CHOICES)
            raise ValueError(f"unsupported MatAnyone2 size: {matanyone2_size}; supported: {supported}")
        cmd.extend(["--matanyone2-size", str(matanyone2_size)])
        cmd.extend(["--matanyone2-batch", "1"])
        cmd.append("--no-sbs-batch")
    if args.engine == "matanyone2":
        cmd.extend(["--sam3-prompt", str(getattr(args, "sam3_prompt", "person") or "person")])
    if args.engine == "matanyone2_medium":
        cmd.extend(["--matanyone2-prepass", "yolo26m_efficientsam"])
    cmd.extend(["--preset", str(args.preset)])
    cmd.extend(["--cq", str(getattr(args, "cq", RVM_DEFAULT_ARGS["cq"]))])
    if engine == "rvm":
        cmd.extend(["--input-size", str(args.input_size)])
        cmd.extend(["--rvm-downsample-ratio", str(args.rvm_downsample_ratio)])
        cmd.append("--sbs-batch")
    return cmd


def _print_warmup_notice() -> None:
    global _WARMUP_NOTICE_PRINTED
    if _WARMUP_NOTICE_PRINTED:
        return
    _WARMUP_NOTICE_PRINTED = True
    try:
        from utils.gpu_runtime_cache import predict_warmup_state

        prediction = predict_warmup_state()
        state = "cold" if prediction.cold else "cache-hit"
        print(
            "[offline] gpu warmup: "
            f"state={state} reason={prediction.reason} eta={prediction.estimate_sec:.0f}s "
            f"gpu={prediction.gpu_name or 'unknown'} cc={prediction.compute_capability or 'unknown'} "
            f"ort={prediction.onnxruntime_version or 'unknown'}",
            flush=True,
        )
        if prediction.cold:
            print(
                "[offline] gpu warmup: first offline run can appear idle while CUDA/CuPy/ONNX Runtime "
                "load libraries and build caches. Please wait for the first [matting] or [offline-*] "
                "progress line.",
                flush=True,
            )
    except Exception as exc:
        print(
            "[offline] gpu warmup: checking cache state failed; first offline run may still take "
            f"1-3 minutes before progress appears ({type(exc).__name__}: {exc})",
            flush=True,
        )


def _run_one(args: argparse.Namespace, src: Path) -> int:
    gpu_requirement = detect_nvidia_gpu_requirement()
    if gpu_requirement.detected and not gpu_requirement.supported:
        print(f"[offline] ERROR: {unsupported_gpu_message(gpu_requirement)}", flush=True)
        return 3
    meta = probe_video_metadata(src)
    decision = select_backend(meta.timing, meta.codec, meta.color)
    if decision.verdict == "block":
        print(
            "[offline] ERROR: unsupported source video. "
            f"input codec={meta.codec.codec_name or 'unknown'} "
            f"profile={meta.codec.profile or 'unknown'} "
            f"pix_fmt={meta.codec.pix_fmt or 'unknown'} "
            f"size={meta.codec.width}x{meta.codec.height}; "
            f"reason={decision.reason}",
            flush=True,
        )
        return 4
    width = int(getattr(meta.codec, "width", 0) or 0)
    height = int(getattr(meta.codec, "height", 0) or 0)
    default_out = (
        _single_out(src, args, width, height)
        if getattr(args, "command", "") == "single"
        else _default_out(src, args.mode, width, height)
    )
    if getattr(args, "out_dir", ""):
        out = Path(args.out_dir).resolve() / default_out.name
    else:
        out = Path(args.out).resolve() if args.out else default_out
    if out.exists() and args.skip_existing:
        print(f"[offline] skip existing: {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _base_cmd(args, src, out)
    print("[offline] run: " + subprocess.list2cmdline(cmd), flush=True)
    _print_warmup_notice()
    env = _offline_child_env(args)
    print(
        "[offline] env: "
        f"decoder={env.get('PT_PASSTHROUGH_PYNV_DECODER', '')} "
        f"batch={env.get('PT_PASSTHROUGH_PYNV_THREADED_BATCH_SIZE', '')} "
        f"buffer={env.get('PT_PASSTHROUGH_PYNV_THREADED_BUFFER_SIZE', '')} "
        f"preset={env.get('PT_PASSTHROUGH_PYNV_PRESET', '')} "
        f"model={env.get('PT_MODEL_PATH', '')} "
        f"matanyone2_size={getattr(args, 'matanyone2_size', '') if args.engine in {'matanyone2', 'matanyone2_medium'} else ''} "
        f"matanyone2_model={'matanyone2_onnx_' + str(getattr(args, 'matanyone2_size', MATANYONE2_DEFAULT_SIZE)) + '_bs1' if args.engine in {'matanyone2', 'matanyone2_medium'} else ''} "
        f"providers={env.get('PT_ONNX_PROVIDERS', '')} "
        f"offline_rvm_trt_enable={env.get('PT_OFFLINE_RVM_TRT_ENABLE', '1')} "
        f"offline_matanyone2_trt_enable={env.get('PT_OFFLINE_MATANYONE2_TRT_ENABLE', '1')} "
        f"offline_rvm_trt={env.get('PT_OFFLINE_RVM_TRT', '0')} "
        f"offline_matanyone2_trt={env.get('PT_OFFLINE_MATANYONE2_TRT', '0')}",
        flush=True,
    )
    return run_hidden_streaming(cmd, cwd=ROOT, env=env, exit_label="offline")


def _gpu_vram_gb() -> float:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0.0
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            **hidden_subprocess_kwargs(),
        )
        first = out.strip().splitlines()[0]
        return float(first.strip()) / 1024.0
    except Exception:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline passthrough converter")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="convert one video")
    single.add_argument("video")
    single.add_argument("--out", default="")
    single.add_argument("--out-dir", default="")
    single.add_argument("--mode", choices=["green", "alpha"], default="green")
    single.add_argument("--engine", choices=sorted(ENGINES), default="rvm_fast")
    single.add_argument("--start", type=float, default=0.0)
    single.add_argument("--duration", type=float, default=0.0)
    single.add_argument("--fps", type=float, default=RVM_DEFAULT_ARGS["fps"])
    single.add_argument("--input-size", type=int, default=RVM_DEFAULT_ARGS["input_size"])
    single.add_argument("--rvm-downsample-ratio", type=float, default=RVM_DEFAULT_ARGS["downsample_ratio"])
    single.add_argument("--skip-frames", type=int, choices=[0, 1, 2], default=RVM_DEFAULT_ARGS["skip_frames"])
    single.add_argument("--bitrate", default=RVM_DEFAULT_ARGS["bitrate"])
    single.add_argument("--preset", default=RVM_DEFAULT_ARGS["preset"])
    single.add_argument("--cq", type=int, default=RVM_DEFAULT_ARGS["cq"])
    single.add_argument("--matanyone2-size", type=int, choices=MATANYONE2_SIZE_CHOICES, default=MATANYONE2_DEFAULT_SIZE)
    single.add_argument("--sam3-prompt", default="person")
    single.add_argument("--skip-existing", action="store_true")

    batch = sub.add_parser("batch", help="convert all videos under a directory")
    batch.add_argument("directory")
    batch.add_argument("--mode", choices=["green", "alpha"], default="green")
    batch.add_argument("--engine", choices=sorted(ENGINES), default="rvm_fast")
    batch.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    batch.add_argument("--skip-existing", action="store_true")
    batch.add_argument("--fps", type=float, default=RVM_DEFAULT_ARGS["fps"])
    batch.add_argument("--input-size", type=int, default=RVM_DEFAULT_ARGS["input_size"])
    batch.add_argument("--rvm-downsample-ratio", type=float, default=RVM_DEFAULT_ARGS["downsample_ratio"])
    batch.add_argument("--skip-frames", type=int, choices=[0, 1, 2], default=RVM_DEFAULT_ARGS["skip_frames"])
    batch.add_argument("--bitrate", default=RVM_DEFAULT_ARGS["bitrate"])
    batch.add_argument("--preset", default=RVM_DEFAULT_ARGS["preset"])
    batch.add_argument("--cq", type=int, default=RVM_DEFAULT_ARGS["cq"])
    batch.add_argument("--matanyone2-size", type=int, choices=MATANYONE2_SIZE_CHOICES, default=MATANYONE2_DEFAULT_SIZE)
    batch.add_argument("--sam3-prompt", default="person")
    batch.set_defaults(out="", start=0.0, duration=0.0)

    args = parser.parse_args(argv)
    if args.engine == "matanyone2":
        print("[offline] MatAnyone2 requires an NVIDIA GPU with at least 16GB VRAM.", flush=True)
        vram = _gpu_vram_gb()
        if vram < 15.5:
            print(f"[offline] insufficient GPU VRAM: detected={vram:.1f}GB required=16GB", flush=True)
            return 2
    if args.command == "single":
        return _run_one(args, Path(args.video).resolve())

    root = Path(args.directory).resolve()
    failures = 0
    files = _video_files(root, args.recursive)
    print(f"[offline] batch files={len(files)} root={root}", flush=True)
    for src in files:
        rc = _run_one(args, src)
        if rc != 0:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
