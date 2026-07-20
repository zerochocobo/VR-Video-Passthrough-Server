"""NVIDIA RTX Video SDK VSR CUDA bridge.

The bridge consumes/produces contiguous RGBA8 CUDA device buffers. Video
pipelines remain responsible for NV12/P010 <-> RGB conversion and encoding.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from utils.subprocess_hidden import hidden_subprocess_kwargs
from utils.vr_naming import is_half_equirectangular_source


class _Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_uint32), ("top", ctypes.c_uint32), ("right", ctypes.c_uint32), ("bottom", ctypes.c_uint32)]


class _VsrSetting(ctypes.Structure):
    _fields_ = [("quality_level", ctypes.c_uint32)]


class _ThdrSetting(ctypes.Structure):
    _fields_ = [
        ("contrast", ctypes.c_uint32),
        ("saturation", ctypes.c_uint32),
        ("middle_gray", ctypes.c_uint32),
        ("max_luminance", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class RtxVsrCapability:
    available: bool
    reason: str
    runtime_dir: Path | None = None


def runtime_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("PT_RTX_VSR_SDK_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
        candidates.extend((base / "models" / "rtx_vsr" / "runtime", base / "_internal" / "models" / "rtx_vsr" / "runtime"))
    else:
        candidates.append(config.ROOT / "models" / "rtx_vsr" / "runtime")
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            path = path.resolve()
        except OSError:
            continue
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def target_resolution(target_height: int | None = None, width: int = 0, height: int = 0) -> tuple[int, int]:
    """Return target bounds, including experimental 8K SBS VR output."""
    out_h = int(target_height or config.RTX_VSR_TARGET_HEIGHT)
    if out_h >= 2160 and int(width or 0) > 0 and int(height or 0) > 0 and is_half_equirectangular_source(width, height):
        if out_h >= 4096:
            return 8192, 4096
        return 4096, 2048
    if out_h >= 4096:
        return 3840, 2160
    out_w = int(round(out_h * 16.0 / 9.0))
    return out_w + (out_w & 1), out_h + (out_h & 1)


def effective_offline_target_height(target_height: int | None, width: int, height: int) -> int:
    """Limit experimental 8K output to 2:1 SBS VR sources; otherwise use 4K."""
    requested = int(target_height or config.RTX_VSR_TARGET_HEIGHT)
    if requested >= 4096 and not is_half_equirectangular_source(width, height):
        return 2160
    return requested


def source_exceeds_target_resolution(width: int, height: int, target_height: int | None = None) -> bool:
    """Compare landscape or portrait sources against the configured target bounds."""
    target_w, target_h = target_resolution(target_height, width, height)
    source_long, source_short = max(int(width), int(height)), min(int(width), int(height))
    target_long, target_short = max(target_w, target_h), min(target_w, target_h)
    return source_long > target_long or source_short > target_short


def source_block_reason(width: int, height: int, *, is_vr: bool = False, is_10bit: bool = False, target_height: int | None = None, allow_vr: bool = False) -> str | None:
    if not config.RTX_VSR_ENABLED:
        return "disabled"
    if is_vr and not allow_vr:
        return "unsupported_vr_source"
    if is_10bit:
        return "unsupported_10bit_source"
    h = int(height or 0)
    if h < config.RTX_VSR_INPUT_MIN_HEIGHT or (h > config.RTX_VSR_INPUT_MAX_HEIGHT and not (is_vr and allow_vr)):
        return "project_resolution_policy"
    if int(width or 0) <= 0:
        return "invalid_source_size"
    if source_exceeds_target_resolution(width, height, target_height):
        return "source_exceeds_target_resolution"
    return None


def target_dimensions(width: int, height: int, target_height: int | None = None) -> tuple[int, int]:
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    target_w, target_h = target_resolution(target_height, width, height)
    if height > width:
        target_w, target_h = target_h, target_w
    scale = min(target_w / width, target_h / height)
    if scale <= 1.0:
        return width + (width & 1), height + (height & 1)
    out_w = round(width * scale)
    out_h = round(height * scale)
    return out_w + (out_w & 1), out_h + (out_h & 1)


class RtxVsrBridge:
    def __init__(self, runtime_dir: Path, dll: ctypes.WinDLL):
        self.runtime_dir = runtime_dir
        self._dll = dll
        self._initialized = False
        self._context_key: tuple[int, int, int] | None = None
        self._lock = threading.RLock()
        self._bind()

    def _bind(self) -> None:
        self._dll.pt_rtx_vsr_set_app_path.argtypes = [ctypes.c_wchar_p]
        self._dll.pt_rtx_vsr_set_app_path.restype = None
        self._dll.pt_rtx_vsr_set_paths.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        self._dll.pt_rtx_vsr_set_paths.restype = None
        self._dll.rtx_video_api_cuda_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
        self._dll.rtx_video_api_cuda_create.restype = ctypes.c_uint
        self._dll.rtx_video_api_cuda_evaluate_deviceptr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            _Rect,
            _Rect,
            ctypes.POINTER(_VsrSetting),
            ctypes.POINTER(_ThdrSetting),
        ]
        self._dll.rtx_video_api_cuda_evaluate_deviceptr.restype = ctypes.c_uint
        self._dll.rtx_video_api_cuda_evaluate_hostptr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            _Rect,
            _Rect,
            ctypes.POINTER(_VsrSetting),
            ctypes.POINTER(_ThdrSetting),
        ]
        self._dll.rtx_video_api_cuda_evaluate_hostptr.restype = ctypes.c_uint
        self._dll.rtx_video_api_cuda_shutdown.argtypes = []
        self._dll.rtx_video_api_cuda_shutdown.restype = None

    def initialize(self, gpu_index: int = 0, *, cu_context: int = 0, cu_stream: int = 0) -> bool:
        with self._lock:
            requested_key = (int(gpu_index), int(cu_context or 0), int(cu_stream or 0))
            if self._initialized and self._context_key == requested_key:
                return True
            if self._initialized:
                self._dll.rtx_video_api_cuda_shutdown()
                self._initialized = False
                self._context_key = None
            data_dir = Path(tempfile.gettempdir()) / "PTMediaServer" / "rtx_vsr"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._dll.pt_rtx_vsr_set_paths(str(self.runtime_dir), str(data_dir))
            self._initialized = bool(
                self._dll.rtx_video_api_cuda_create(
                    ctypes.c_void_p(int(cu_context or 0)),
                    ctypes.c_void_p(int(cu_stream or 0)),
                    int(gpu_index),
                    0,
                    1,
                )
            )
            if self._initialized:
                self._context_key = requested_key
            return self._initialized

    def initialize_cupy(self, cp: Any, gpu_index: int = 0) -> bool:
        cp.cuda.Device(int(gpu_index)).use()
        # Force creation of the runtime/primary context before asking for its
        # handle.  NGX must be initialized with the same context that owns the
        # CuPy device pointers passed to EvaluateFeature.
        cp.cuda.runtime.free(0)
        context = int(cp.cuda.driver.ctxGetCurrent())
        stream = int(cp.cuda.get_current_stream().ptr)
        if not context:
            raise RuntimeError("CuPy did not provide a current CUDA context")
        return self.initialize(gpu_index, cu_context=context, cu_stream=stream)

    def evaluate_deviceptr(
        self,
        input_ptr: int,
        output_ptr: int,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        quality: int | None = None,
    ) -> None:
        with self._lock:
            if not self._initialized and not self.initialize():
                raise RuntimeError("RTX VSR feature initialization failed")
            in_w, in_h = map(int, input_size)
            out_w, out_h = map(int, output_size)
            requested_quality = config.RTX_VSR_QUALITY if quality is None else quality
            setting = _VsrSetting(max(0, min(4, int(requested_quality))))
            thdr = _ThdrSetting()
            ok = self._dll.rtx_video_api_cuda_evaluate_deviceptr(
                ctypes.c_void_p(int(input_ptr)),
                ctypes.c_void_p(int(output_ptr)),
                _Rect(0, 0, in_w, in_h),
                _Rect(0, 0, out_w, out_h),
                ctypes.byref(setting),
                ctypes.byref(thdr),
            )
            if not ok:
                raise RuntimeError(f"RTX VSR evaluate failed input={in_w}x{in_h} output={out_w}x{out_h}")

    def process_cupy_rgba(self, rgba: Any, output_size: tuple[int, int], quality: int | None = None):
        import cupy as cp

        if rgba.dtype != cp.uint8 or rgba.ndim != 3 or int(rgba.shape[2]) != 4 or not rgba.flags.c_contiguous:
            raise ValueError("RTX VSR input must be contiguous uint8 HxWx4 RGBA")
        if not self.initialize_cupy(cp, int(rgba.device.id)):
            raise RuntimeError("RTX VSR feature initialization failed for CuPy context")
        out_w, out_h = map(int, output_size)
        out = cp.empty((out_h, out_w, 4), dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        self.evaluate_deviceptr(
            int(rgba.data.ptr),
            int(out.data.ptr),
            (int(rgba.shape[1]), int(rgba.shape[0])),
            (out_w, out_h),
            quality,
        )
        cp.cuda.get_current_stream().synchronize()
        return out

    def process_host_rgba(self, rgba: bytes | bytearray, input_size: tuple[int, int], output_size: tuple[int, int], quality: int | None = None) -> bytes:
        """Diagnostic host-pointer path that does not require CuPy.

        Production realtime/offline paths use CUDA device buffers.  This
        method exists for capability/evaluate probes on machines where the
        Python environment does not yet have CuPy installed.
        """
        in_w, in_h = map(int, input_size)
        out_w, out_h = map(int, output_size)
        expected_in = in_w * in_h * 4
        if len(rgba) != expected_in:
            raise ValueError(f"RGBA host input length {len(rgba)} != {expected_in}")
        source = (ctypes.c_ubyte * expected_in).from_buffer_copy(rgba)
        target = (ctypes.c_ubyte * (out_w * out_h * 4))()
        requested_quality = config.RTX_VSR_QUALITY if quality is None else quality
        setting = _VsrSetting(max(0, min(4, int(requested_quality))))
        thdr = _ThdrSetting()
        with self._lock:
            if not self._initialized and not self.initialize():
                raise RuntimeError("RTX VSR feature initialization failed")
            ok = self._dll.rtx_video_api_cuda_evaluate_hostptr(
                ctypes.cast(source, ctypes.c_void_p),
                ctypes.cast(target, ctypes.c_void_p),
                _Rect(0, 0, in_w, in_h),
                _Rect(0, 0, out_w, out_h),
                ctypes.byref(setting),
                ctypes.byref(thdr),
            )
        if not ok:
            raise RuntimeError(f"RTX VSR host evaluate failed input={in_w}x{in_h} output={out_w}x{out_h}")
        return bytes(target)

    def process_driver_rgba(self, rgba: bytes | bytearray, input_size: tuple[int, int], output_size: tuple[int, int], quality: int | None = None) -> bytes:
        """Exercise the CUDA device-pointer bridge without requiring CuPy.

        This is a diagnostic fallback for the probe only.  The application
        still uses CuPy for frame conversion and buffer reuse.
        """
        if not sys.platform.startswith("win"):
            raise RuntimeError("CUDA driver diagnostic is Windows-only")
        in_w, in_h = map(int, input_size)
        out_w, out_h = map(int, output_size)
        in_size = in_w * in_h * 4
        out_size = out_w * out_h * 4
        if len(rgba) != in_size:
            raise ValueError(f"RGBA host input length {len(rgba)} != {in_size}")
        driver = ctypes.WinDLL("nvcuda.dll")
        cu_mem_alloc = getattr(driver, "cuMemAlloc_v2", None) or getattr(driver, "cuMemAlloc")
        cu_mem_free = getattr(driver, "cuMemFree_v2", None) or getattr(driver, "cuMemFree")
        cu_h2d = getattr(driver, "cuMemcpyHtoD_v2", None) or getattr(driver, "cuMemcpyHtoD")
        cu_d2h = getattr(driver, "cuMemcpyDtoH_v2", None) or getattr(driver, "cuMemcpyDtoH")
        cu_mem_alloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu_mem_alloc.restype = ctypes.c_int
        cu_mem_free.argtypes = [ctypes.c_uint64]
        cu_mem_free.restype = ctypes.c_int
        cu_h2d.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cu_h2d.restype = ctypes.c_int
        cu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu_d2h.restype = ctypes.c_int
        source = (ctypes.c_ubyte * in_size).from_buffer_copy(rgba)
        target = (ctypes.c_ubyte * out_size)()
        d_in, d_out = ctypes.c_uint64(), ctypes.c_uint64()
        if cu_mem_alloc(ctypes.byref(d_in), in_size) != 0 or cu_mem_alloc(ctypes.byref(d_out), out_size) != 0:
            if d_in.value:
                cu_mem_free(d_in)
            raise RuntimeError("CUDA driver allocation failed")
        try:
            if cu_h2d(d_in, ctypes.cast(source, ctypes.c_void_p), in_size) != 0:
                raise RuntimeError("CUDA driver HtoD copy failed")
            self.evaluate_deviceptr(d_in.value, d_out.value, (in_w, in_h), (out_w, out_h), quality)
            if cu_d2h(ctypes.cast(target, ctypes.c_void_p), d_out, out_size) != 0:
                raise RuntimeError("CUDA driver DtoH copy failed")
            return bytes(target)
        finally:
            cu_mem_free(d_in)
            cu_mem_free(d_out)

    def shutdown(self) -> None:
        with self._lock:
            if self._initialized:
                self._dll.rtx_video_api_cuda_shutdown()
                self._initialized = False
                self._context_key = None


_bridge: RtxVsrBridge | None = None
_bridge_lock = threading.Lock()
_preflight_lock = threading.Lock()
_preflight_result: dict[str, Any] | None = None
_preflight_failed_at: float = 0.0
# A failed preflight is retried, but not on every request: each attempt costs a
# child process plus a full CUDA/NGX init, and DLNA players retry 409s on their
# own.  Serve the cached failure inside this window instead.
_PREFLIGHT_FAILURE_RETRY_SEC = 60.0


def load_bridge() -> RtxVsrBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is not None:
            return _bridge
        if not sys.platform.startswith("win"):
            raise RuntimeError("RTX Video SDK is supported only on Windows")
        errors: list[str] = []
        for runtime_dir in runtime_candidates():
            bridge_dll = runtime_dir / "pt_rtx_vsr_bridge.dll"
            feature_dll = runtime_dir / "nvngx_vsr.dll"
            if not bridge_dll.exists() or not feature_dll.exists():
                errors.append(f"missing runtime files under {runtime_dir}")
                continue
            try:
                os.add_dll_directory(str(runtime_dir))
                dll = ctypes.WinDLL(str(bridge_dll))
                _bridge = RtxVsrBridge(runtime_dir, dll)
                return _bridge
            except (AttributeError, OSError) as exc:
                errors.append(f"{runtime_dir}: {exc}")
        raise RuntimeError("RTX VSR runtime unavailable: " + "; ".join(errors))


def probe_capability(gpu_index: int = 0) -> RtxVsrCapability:
    if not config.RTX_VSR_ENABLED:
        return RtxVsrCapability(False, "disabled")
    try:
        bridge = load_bridge()
        if not bridge.initialize(gpu_index):
            return RtxVsrCapability(False, "feature_unavailable", bridge.runtime_dir)
        return RtxVsrCapability(True, "available", bridge.runtime_dir)
    except Exception as exc:
        return RtxVsrCapability(False, str(exc))


def run_evaluation_preflight(*, timeout_sec: float | None = None, gpu_index: int = 0) -> dict[str, Any]:
    """Run one real VSR evaluation outside the server process.

    NGX may perform first-run model installation/initialization inside the
    evaluate call.  Keeping this probe in a child process means a driver or
    SDK hang cannot permanently consume a realtime media worker.
    """
    global _preflight_result, _preflight_failed_at
    with _preflight_lock:
        # Successful capability/evaluate checks are reusable. Failed checks are
        # retried after a cooldown: a first-run NGX model install can exceed the
        # timeout once and succeed on the next attempt, but a persistently
        # broken driver must not spawn a probe per request.
        if _preflight_result is not None:
            if _preflight_result.get("ok"):
                return dict(_preflight_result)
            if time.monotonic() - _preflight_failed_at < _PREFLIGHT_FAILURE_RETRY_SEC:
                return dict(_preflight_result)
        timeout = float(timeout_sec or config.RTX_VSR_EVALUATE_TIMEOUT_SEC)
        if getattr(sys, "frozen", False):
            command = [sys.executable, "tool", "rtx_vsr_probe", "--evaluate", "--gpu", str(int(gpu_index))]
            cwd = str(Path(sys.executable).resolve().parent)
        else:
            command = [sys.executable, "-m", "tools.rtx_vsr_probe", "--evaluate", "--gpu", str(int(gpu_index))]
            cwd = str(config.ROOT)
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=os.environ.copy(),
                **hidden_subprocess_kwargs(),
            )
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            payload: dict[str, Any] = {}
            for line in lines:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    payload.update(value)
            if completed.returncode == 0 and payload.get("checksum") is not None:
                result = {"ok": True, "reason": "available", **payload}
            else:
                detail = (completed.stderr or "").strip()[-1000:]
                result = {"ok": False, "reason": payload.get("reason") or "evaluate_failed", "detail": detail}
        except subprocess.TimeoutExpired:
            result = {"ok": False, "reason": "evaluate_timeout", "detail": f"timeout after {timeout:.1f}s"}
        except Exception as exc:
            result = {"ok": False, "reason": "preflight_error", "detail": str(exc)}
        _preflight_result = result
        _preflight_failed_at = 0.0 if result.get("ok") else time.monotonic()
        return dict(result)


def evaluation_preflight_status() -> dict[str, Any] | None:
    with _preflight_lock:
        return dict(_preflight_result) if _preflight_result is not None else None
