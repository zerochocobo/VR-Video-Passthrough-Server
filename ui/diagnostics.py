"""Diagnostic report generator for non-technical users.

Produces a single multi-line text string that the user can copy to their
clipboard with one click and paste into chat/forum/email when they need
technical help. Covers GPU, drivers, ONNX Runtime, CuPy, warmup history,
operating system, and recent startup status so a remote technician can
quickly understand the environment without screen sharing.

The generator does NOT import config.py or any GPU library at module import
time; it imports lazily so it can be safely called even when CUDA/CuPy
are broken or missing.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.subprocess_hidden import hidden_subprocess_kwargs


def _safe(text: Any) -> str:
    if text is None:
        return ""
    try:
        return str(text)
    except Exception:
        return ""


def _read_marker(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _nvidia_smi_summary() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            **hidden_subprocess_kwargs(),
        )
        return out.strip()
    except FileNotFoundError:
        return "nvidia-smi: not on PATH"
    except subprocess.CalledProcessError as e:
        return f"nvidia-smi: exit {e.returncode}"
    except Exception as e:
        return f"nvidia-smi: {e}"


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as e:
        return f"(not installed: {e})"
    return _safe(getattr(module, "__version__", "unknown"))


def _ort_providers() -> str:
    try:
        import onnxruntime as ort  # type: ignore

        return ",".join(ort.get_available_providers())
    except Exception as e:
        return f"(unavailable: {e})"


def _ffbinary_summary(name: str) -> str:
    """Return path + first version line for ffmpeg / ffprobe.

    Common failure modes a remote technician needs to see at a glance:
      - binary not on PATH at all (the most frequent support case)
      - binary present but the wrong build (no NVENC, wrong libc, etc.)
      - binary too old to support a flag the pipeline relies on
    """
    binary = shutil.which(name)
    if not binary:
        return f"  {name}: (not on PATH)"
    try:
        out = subprocess.check_output(
            [binary, "-version"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            **hidden_subprocess_kwargs(),
        )
    except FileNotFoundError:
        return f"  {name}: (resolved {binary} but exec failed: FileNotFoundError)"
    except subprocess.TimeoutExpired:
        return f"  {name}: {binary}\n             (timed out reading -version)"
    except subprocess.CalledProcessError as e:
        return f"  {name}: {binary}\n             (exit {e.returncode})"
    except Exception as e:  # pragma: no cover - defensive
        return f"  {name}: {binary}\n             (error: {e})"
    first_line = out.strip().splitlines()[0] if out.strip() else "(no output)"
    return f"  {name}: {binary}\n             {first_line}"


def _ffmpeg_nvenc_summary() -> str:
    """Return whether ffmpeg ships h264_nvenc / hevc_nvenc encoders."""
    binary = shutil.which("ffmpeg")
    if not binary:
        return "  nvenc_encoders: (ffmpeg not on PATH)"
    try:
        out = subprocess.check_output(
            [binary, "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            **hidden_subprocess_kwargs(),
        )
    except Exception as e:
        return f"  nvenc_encoders: (probe failed: {e})"
    found = [name for name in ("h264_nvenc", "hevc_nvenc") if name in out]
    return "  nvenc_encoders: " + (", ".join(found) if found else "(none — software encode only)")


def _cupy_devices() -> str:
    try:
        import cupy as cp  # type: ignore

        count = int(cp.cuda.runtime.getDeviceCount())
        names = []
        for i in range(count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            raw = props.get("name", "")
            label = raw.decode() if isinstance(raw, bytes) else str(raw)
            names.append(f"{i}:{label} sm_{props.get('major')}{props.get('minor')}")
        return "; ".join(names) if names else "(no devices)"
    except Exception as e:
        return f"(unavailable: {e})"


def _format_marker(marker: dict | None) -> str:
    if marker is None:
        return "  (no warmup marker — first run not completed)"
    key = marker.get("key", {}) or {}
    lines = [
        f"  created_at:                  {marker.get('created_at', '')}",
        f"  elapsed_sec:                 {marker.get('elapsed_sec', '')}",
        f"  verified_second_pass_sec:    {marker.get('verified_second_pass_sec', '')}",
        f"  cache_file_count:            {marker.get('cache_file_count_after_warmup', '')}",
        f"  cache_size:                  {marker.get('cache_size_after_warmup', '')}",
        f"  gpu_name:                    {key.get('gpu_name', '')}",
        f"  compute_capability:          {key.get('compute_capability', '')}",
        f"  driver_version:              {key.get('driver_version', '')}",
        f"  onnxruntime_version:         {key.get('onnxruntime_version', '')}",
        f"  ort_cuda_dll_hash:           {key.get('onnxruntime_providers_cuda_dll_hash', '')}",
        f"  cupy_version:                {key.get('cupy_version', '')}",
        f"  cupy_cuda_runtime:           {key.get('cupy_cuda_runtime', '')}",
        f"  model_name:                  {key.get('model_name', '')}",
        f"  model_sha256_16:             {key.get('model_sha256_16', '')}",
        f"  input_size:                  {key.get('input_size', '')}",
        f"  providers:                   {key.get('providers', '')}",
        f"  shapes:                      {key.get('shapes', '')}",
    ]
    return "\n".join(lines)


def _tail_server_log(log_path: Path | None, max_lines: int = 200) -> str:
    """Return the last ``max_lines`` lines of the server log file.

    Returns a human-readable note if the file is missing or unreadable so the
    diagnostic report stays useful regardless.
    """
    if log_path is None:
        return "  (log path not provided)"
    path = Path(log_path)
    if not path.exists():
        return f"  (log file not found: {path})"
    try:
        # Read with replace so a partial UTF-8 sequence never crashes the report.
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            buffer: list[str] = []
            for line in handle:
                buffer.append(line.rstrip("\n"))
                if len(buffer) > max_lines:
                    buffer.pop(0)
        if not buffer:
            return "  (log file is empty)"
        return "\n".join(buffer)
    except OSError as e:
        return f"  (failed to read log: {e})"


def _format_status(status: dict | None) -> str:
    if status is None:
        return "  (status endpoint not reachable)"
    keys = (
        "phase",
        "step",
        "step_index",
        "step_total",
        "progress",
        "eta_sec",
        "elapsed_sec",
        "cold",
        "is_known_slow",
        "gpu_name",
        "compute_capability",
        "driver_version",
        "onnxruntime_version",
        "reason",
        "message",
        "detail",
        "uptime_sec",
        "age_sec",
    )
    return "\n".join(f"  {k:24s}: {status.get(k, '')}" for k in keys)


def build_diagnostic_report(
    *,
    app_version: str = "",
    language: str = "",
    last_status: dict | None = None,
    marker_path: Path | None = None,
    log_path: Path | None = None,
    log_tail_lines: int = 200,
) -> str:
    """Compose a copy-pasteable multi-line diagnostic report.

    Parameters are all optional; missing pieces are filled with safe defaults
    and labelled clearly so the report stays useful even if some probes fail.
    """
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marker: dict | None = None
    marker_path_str = ""
    if marker_path is not None:
        marker = _read_marker(Path(marker_path))
        marker_path_str = str(marker_path)

    section_header = "=== PTServer Diagnostic Report ==="
    lines: list[str] = [
        section_header,
        f"generated:        {now}",
        f"app_version:      {app_version or '(unknown)'}",
        f"ui_language:      {language or '(unknown)'}",
        f"host:             {socket.gethostname()}",
        f"os:               {platform.platform()}",
        f"python:           {sys.version.split()[0]}",
        f"executable:       {sys.executable}",
        f"frozen:           {bool(getattr(sys, 'frozen', False))}",
        f"cwd:              {os.getcwd()}",
        "",
        "--- GPU (nvidia-smi) ---",
        f"  {_nvidia_smi_summary()}",
        "",
        "--- Runtime libraries ---",
        f"  onnxruntime:     {_module_version('onnxruntime')}",
        f"  ort_providers:   {_ort_providers()}",
        f"  cupy:            {_module_version('cupy')}",
        f"  cupy_devices:    {_cupy_devices()}",
        f"  numpy:           {_module_version('numpy')}",
        "",
        "--- FFmpeg / FFprobe (PATH lookup) ---",
        _ffbinary_summary("ffmpeg"),
        _ffbinary_summary("ffprobe"),
        _ffmpeg_nvenc_summary(),
        "",
        "--- Warmup marker ---",
        f"  path:            {marker_path_str or '(not provided)'}",
        _format_marker(marker),
        "",
        "--- Last startup status (from /status) ---",
        _format_status(last_status),
        "",
        f"--- Recent server.log (last {log_tail_lines} lines) ---",
        f"  path:            {str(log_path) if log_path is not None else '(not provided)'}",
        _tail_server_log(log_path, max_lines=log_tail_lines),
        "",
        "=== End of report ===",
    ]
    return "\n".join(lines)
