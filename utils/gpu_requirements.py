from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from utils.subprocess_hidden import hidden_subprocess_kwargs


MIN_NVIDIA_COMPUTE_CAPABILITY = 7.5


@dataclass(frozen=True)
class GpuRequirementResult:
    detected: bool
    supported: bool
    name: str = ""
    compute_capability: str = ""
    detail: str = ""


def parse_compute_capability(value: str) -> float | None:
    text = str(value or "").strip().lower().replace("sm_", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _nvidia_smi_path() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for root in (os.environ.get("SystemRoot"), os.environ.get("WINDIR")):
        if not root:
            continue
        system32_smi = Path(root) / "System32" / "nvidia-smi.exe"
        if system32_smi.exists():
            return str(system32_smi)
    return None


def detect_nvidia_gpu_requirement() -> GpuRequirementResult:
    exe = _nvidia_smi_path()
    if not exe:
        return GpuRequirementResult(False, False, detail="nvidia-smi not found")
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            **hidden_subprocess_kwargs(),
        )
    except Exception as exc:
        return GpuRequirementResult(False, False, detail=f"nvidia-smi failed: {type(exc).__name__}: {exc}")

    candidates: list[tuple[float, str, str]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        cc = parse_compute_capability(parts[-1])
        if cc is None:
            continue
        candidates.append((cc, parts[0], parts[-1]))
    if not candidates:
        return GpuRequirementResult(False, False, detail="no NVIDIA compute capability reported")
    cc, name, cc_text = max(candidates, key=lambda item: item[0])
    return GpuRequirementResult(
        detected=True,
        supported=cc >= MIN_NVIDIA_COMPUTE_CAPABILITY,
        name=name,
        compute_capability=cc_text,
    )


def detect_nvidia_total_vram_gib() -> float | None:
    """Return the largest detected NVIDIA GPU VRAM size in GiB."""
    exe = _nvidia_smi_path()
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            **hidden_subprocess_kwargs(),
        )
    except Exception:
        return None

    best_mib = 0.0
    for line in out.splitlines():
        try:
            best_mib = max(best_mib, float(line.strip()))
        except ValueError:
            continue
    return best_mib / 1024.0 if best_mib > 0 else None


def resolve_passthrough_max_concurrent(raw) -> int:
    text = str(raw).strip().lower() if raw is not None else "auto"
    if text and text not in {"auto", ""}:
        try:
            return max(1, int(text))
        except ValueError:
            pass

    vram_gib = detect_nvidia_total_vram_gib()
    if vram_gib is None:
        return 1
    if vram_gib >= 20.0:
        return 3
    if vram_gib >= 12.0:
        return 2
    return 1


def unsupported_gpu_message(result: GpuRequirementResult) -> str:
    gpu = result.name or "unknown NVIDIA GPU"
    cc = result.compute_capability or "unknown"
    return (
        f"Unsupported GPU: {gpu} (compute capability {cc}). "
        "NVIDIA RTX 20 series or newer is required "
        f"(compute capability >= {MIN_NVIDIA_COMPUTE_CAPABILITY:.1f})."
    )
