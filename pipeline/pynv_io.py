"""PyNvVideoCodec adapters for GPU-resident NV12 decode and encode.

The wrappers expose decoded CUDA Array Interface planes to CuPy and wrap the
contiguous composited NV12 buffer as the AppFrame shape expected by the PyNv
encoder. Keeping this boundary small limits PyNv-specific assumptions in the
rest of the server.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cuda_device_summary(gpu_id: int = 0) -> str:
    """Return a compact CUDA device summary for diagnostics."""
    try:
        import cupy as cp

        gpu = int(gpu_id)
        props = cp.cuda.runtime.getDeviceProperties(gpu)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        major = int(props.get("major", 0))
        minor = int(props.get("minor", 0))
        with cp.cuda.Device(gpu):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        driver = cp.cuda.runtime.driverGetVersion()
        runtime = cp.cuda.runtime.runtimeGetVersion()
        return (
            f"gpu_id={gpu} name={name} cc={major}.{minor} "
            f"vram={total_bytes / (1024 ** 3):.1f}GB free={free_bytes / (1024 ** 3):.1f}GB "
            f"driver={driver} runtime={runtime}"
        )
    except Exception as exc:
        return f"gpu_id={gpu_id} unavailable: {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class CudaPlane:
    """A CUDA Array Interface plane owned by a PyNvVideoCodec DecodedFrame."""

    view: Any
    owner: Any
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    ptr: int
    readonly: bool

    @classmethod
    def from_view(cls, view: Any, owner: Any) -> "CudaPlane":
        cai = getattr(view, "__cuda_array_interface__", None)
        if not cai:
            raise TypeError(f"object does not expose CUDA Array Interface: {type(view)!r}")
        data = cai.get("data")
        if not data or not isinstance(data, tuple):
            raise TypeError(f"invalid CUDA Array Interface data field: {data!r}")
        return cls(
            view=view,
            owner=owner,
            shape=tuple(cai["shape"]),
            strides=tuple(cai.get("strides") or ()),
            dtype=str(cai["typestr"]),
            ptr=int(data[0]),
            readonly=bool(data[1]),
        )

    @classmethod
    def from_cupy_array(cls, arr: Any) -> "CudaPlane":
        return cls.from_view(arr, arr)

    @property
    def nbytes(self) -> int:
        if not self.shape:
            return 0
        if self.strides:
            return max(1, self.shape[0] * self.row_stride_bytes)
        n = 1
        for dim in self.shape:
            n *= dim
        return n * self.itemsize

    @property
    def itemsize(self) -> int:
        if self.dtype in {"|u1", "uint8", "u1"}:
            return 1
        if self.dtype in {"<u2", ">u2", "|u2", "uint16", "u2"}:
            return 2
        if self.dtype and self.dtype[-1:].isdigit():
            try:
                return int(self.dtype[-1])
            except Exception:
                pass
        return 1

    @property
    def row_stride_bytes(self) -> int:
        if self.strides:
            stride = int(self.strides[0])
            # PyNvVideoCodec reports P016/P010 plane strides as element counts
            # even though CUDA Array Interface normally uses byte strides.
            if self._strides_are_elements():
                return stride * self.itemsize
            return stride
        width = int(self.shape[1]) if len(self.shape) > 1 else int(self.shape[0] if self.shape else 0)
        return width * self.itemsize

    def _strides_are_elements(self) -> bool:
        return bool(self.itemsize > 1 and self.strides and any(int(s) % self.itemsize for s in self.strides))

    @property
    def cupy_strides(self) -> tuple[int, ...] | None:
        if not self.strides:
            return None
        if self.itemsize <= 1:
            return self.strides
        if self._strides_are_elements():
            return tuple(int(s) * self.itemsize for s in self.strides)
        return self.strides

    def as_cupy(self, dtype=None):
        """Return a zero-copy CuPy ndarray view over the plane."""
        import cupy as cp

        cp_dtype = dtype or (cp.uint16 if self.itemsize == 2 else cp.uint8)
        mem = cp.cuda.UnownedMemory(self.ptr, self.nbytes, self.owner)
        mp = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray(self.shape, dtype=cp_dtype, memptr=mp, strides=self.cupy_strides)


@dataclass(frozen=True)
class GpuNv12Frame:
    """GPU-resident NV12 frame decoded by PyNvVideoCodec."""

    owner: Any
    y: CudaPlane
    uv: CudaPlane
    width: int
    height: int
    pts: int

    @classmethod
    def from_decoded_frame(cls, frame: Any, width: int, height: int) -> "GpuNv12Frame":
        planes = frame.cuda()
        if len(planes) < 2:
            raise RuntimeError(f"expected at least 2 NV12 planes, got {len(planes)}")
        return cls(
            owner=frame,
            y=CudaPlane.from_view(planes[0], frame),
            uv=CudaPlane.from_view(planes[1], frame),
            width=width,
            height=height,
            pts=int(frame.getPTS()),
        )

    def owned_copy(self) -> "GpuNv12Frame":
        """Copy PyNv-owned planes into CuPy-owned GPU memory.

        ThreadedDecoder batch frames are only valid for a narrow lifetime. The
        alpha path runs several CUDA/ORT steps from one decoded frame, so it
        must not keep reading PyNv-managed batch memory after the handoff.
        """
        import cupy as cp

        h, w = int(self.height), int(self.width)
        if tuple(self.y.shape[:2]) != (h, w):
            raise RuntimeError(f"unexpected NV12 Y plane shape: frame={w}x{h} y_shape={self.y.shape}")
        uv_shape = tuple(self.uv.shape)
        if uv_shape not in {(h // 2, w), (h // 2, w // 2, 2)}:
            raise RuntimeError(f"unexpected NV12 UV plane shape: frame={w}x{h} uv_shape={self.uv.shape}")
        # ThreadedDecoder writes on an internal PyNv/NVDEC stream that is not
        # exposed to CuPy. Make the decoded planes visible before copying them
        # into memory owned by this process.
        cp.cuda.Device().synchronize()
        y_src = self.y.as_cupy(cp.uint8).reshape(h, w)
        uv_src = self.uv.as_cupy(cp.uint8).reshape(h // 2, w)
        y = cp.ascontiguousarray(y_src)
        uv = cp.ascontiguousarray(uv_src)
        cp.cuda.get_current_stream().synchronize()
        owner = (y, uv)
        return GpuNv12Frame(
            owner=owner,
            y=CudaPlane.from_cupy_array(y),
            uv=CudaPlane.from_cupy_array(uv),
            width=w,
            height=h,
            pts=self.pts,
        )


@dataclass(frozen=True)
class GpuP016Frame:
    """GPU-resident 10-bit 4:2:0 frame decoded into P016/P010-like planes."""

    owner: Any
    y: CudaPlane
    uv: CudaPlane
    width: int
    height: int
    pts: int

    @classmethod
    def from_decoded_frame(cls, frame: Any, width: int, height: int) -> "GpuP016Frame":
        planes = frame.cuda()
        if len(planes) < 2:
            raise RuntimeError(f"expected at least 2 P016 planes, got {len(planes)}")
        y = CudaPlane.from_view(planes[0], frame)
        uv = CudaPlane.from_view(planes[1], frame)
        if y.itemsize < 2 or uv.itemsize < 2:
            raise RuntimeError(f"expected uint16 P016 planes, got y={y.dtype} uv={uv.dtype}")
        return cls(
            owner=frame,
            y=y,
            uv=uv,
            width=width,
            height=height,
            pts=int(frame.getPTS()),
        )

    def owned_copy(self) -> "GpuP016Frame":
        """Copy PyNv-owned 16-bit planes into CuPy-owned GPU memory."""
        import cupy as cp

        h, w = int(self.height), int(self.width)
        if tuple(self.y.shape[:2]) != (h, w):
            raise RuntimeError(f"unexpected P016 Y plane shape: frame={w}x{h} y_shape={self.y.shape}")
        uv_shape = tuple(self.uv.shape)
        if uv_shape not in {(h // 2, w), (h // 2, w // 2, 2)}:
            raise RuntimeError(f"unexpected P016 UV plane shape: frame={w}x{h} uv_shape={self.uv.shape}")
        # See GpuNv12Frame.owned_copy(): PyNv's decode stream is not exposed.
        cp.cuda.Device().synchronize()
        y_src = self.y.as_cupy(cp.uint16).reshape(h, w)
        uv_src = self.uv.as_cupy(cp.uint16).reshape(h // 2, w)
        y = cp.ascontiguousarray(y_src)
        uv = cp.ascontiguousarray(uv_src)
        cp.cuda.get_current_stream().synchronize()
        owner = (y, uv)
        return GpuP016Frame(
            owner=owner,
            y=CudaPlane.from_cupy_array(y),
            uv=CudaPlane.from_cupy_array(uv),
            width=w,
            height=h,
            pts=self.pts,
        )


@dataclass(frozen=True)
class PyNvVideoInfo:
    """Basic video metadata reported by PyNvVideoCodec.SimpleDecoder."""
    width: int
    height: int
    fps: float
    duration: float
    codec_name: str
    bitrate: float
    num_frames: int


class PyNvSimpleDecoder:
    """Thin wrapper around PyNvVideoCodec.SimpleDecoder."""

    def __init__(self, src: Path, gpu_id: int = 0, bit_depth: int = 8):
        import PyNvVideoCodec as nvc

        self.src = Path(src).resolve()
        self.gpu_id = gpu_id
        self.bit_depth = int(bit_depth or 8)
        self._decoder = nvc.SimpleDecoder(
            str(self.src),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
        )
        meta = self._decoder.get_stream_metadata()
        self.info = PyNvVideoInfo(
            width=int(meta.width),
            height=int(meta.height),
            fps=float(meta.average_fps),
            duration=float(meta.duration),
            codec_name=str(meta.codec_name),
            bitrate=float(meta.bitrate),
            num_frames=int(meta.num_frames),
        )

    def __len__(self) -> int:
        return len(self._decoder)

    def frame_at(self, index: int) -> GpuNv12Frame | GpuP016Frame:
        frame = self._decoder[index]
        if self.bit_depth > 8:
            return GpuP016Frame.from_decoded_frame(frame, self.info.width, self.info.height)
        return GpuNv12Frame.from_decoded_frame(frame, self.info.width, self.info.height)

    def stop(self) -> None:
        stop = getattr(self._decoder, "stop", None)
        if callable(stop):
            try:
                stop()
            except AttributeError:
                # PyNvVideoCodec 2.1.0 exposes SimpleDecoder.stop() in Python,
                # but the wrapped native object does not implement stop().
                pass


class PyNvThreadedSerialDecoder:
    """Sequential ThreadedDecoder wrapper for monotonic source-frame access.

    PyNv ThreadedDecoder frames are only valid until the next get_batch_frames()
    call. This wrapper never queues decoded frames across threads; callers must
    finish consuming the returned frame before calling frame_at() again.
    """

    def __init__(
        self,
        src: Path,
        gpu_id: int = 0,
        bit_depth: int = 8,
        start_frame: int = 0,
        batch_size: int = 8,
        buffer_size: int = 32,
        info: PyNvVideoInfo | None = None,
        num_frames: int | None = None,
    ):
        import PyNvVideoCodec as nvc

        self.src = Path(src).resolve()
        self.gpu_id = int(gpu_id)
        self.bit_depth = int(bit_depth or 8)
        self.batch_size = max(1, int(batch_size))
        self.buffer_size = max(1, int(buffer_size))
        self.start_frame = max(0, int(start_frame))
        self.info = info
        self._len = int(num_frames) if num_frames is not None else 0
        if self.info is None or self._len <= 0:
            probe = PyNvSimpleDecoder(self.src, gpu_id=self.gpu_id, bit_depth=self.bit_depth)
            try:
                self.info = probe.info
                self._len = len(probe)
            finally:
                probe.stop()
        self._decoder = nvc.ThreadedDecoder(
            str(self.src),
            self.buffer_size,
            gpu_id=self.gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.NATIVE,
            start_frame=self.start_frame,
        )
        self._batch: list[Any] = []
        self._batch_pos = 0
        self._batch_start_idx = self.start_frame
        self._next_source_idx = self.start_frame
        self._ended = False

    def __len__(self) -> int:
        return self._len

    def frame_at(self, index: int) -> GpuNv12Frame | GpuP016Frame:
        if self._ended:
            raise RuntimeError("ThreadedDecoder has already ended")
        target = int(index)
        if target < self._next_source_idx:
            raise ValueError(
                f"Threaded serial decoder only supports monotonic access: "
                f"target={target} next_source_idx={self._next_source_idx}"
            )
        while True:
            if self._batch_pos >= len(self._batch):
                self._batch = []
                self._batch_pos = 0
                self._batch_start_idx = self._next_source_idx
                batch = self._decoder.get_batch_frames(self.batch_size)
                if not batch:
                    raise RuntimeError(f"ThreadedDecoder returned no frames at source_idx={self._next_source_idx}")
                self._batch = list(batch)
            current = self._batch_start_idx + self._batch_pos
            raw = self._batch[self._batch_pos]
            self._batch_pos += 1
            self._next_source_idx = current + 1
            if current < target:
                continue
            if current > target:
                raise RuntimeError(f"ThreadedDecoder skipped target frame: target={target} current={current}")
            assert self.info is not None
            if self.bit_depth > 8:
                return GpuP016Frame.from_decoded_frame(raw, self.info.width, self.info.height)
            return GpuNv12Frame.from_decoded_frame(raw, self.info.width, self.info.height)

    def stop(self) -> None:
        if self._ended:
            return
        self._batch = []
        self._batch_pos = 0
        end = getattr(self._decoder, "end", None)
        if callable(end):
            end()
        self._ended = True


class FfmpegNv12SequentialDecoder:
    """Sequential FFmpeg raw-NV12 decoder that uploads frames to GPU memory.

    This is a compatibility fallback for source codecs that PyNv/NVDEC rejects
    at high resolution, such as MPEG-4 Visual. It preserves the downstream
    GPU matting/composite/PyNv encode path, but decode itself goes through
    FFmpeg and a host-to-device upload.
    """

    def __init__(
        self,
        src: Path,
        start_sec: float = 0.0,
        max_fps: float | None = None,
    ):
        import config

        config.DECODE_MAX_SIDE = 0
        config.DECODE_PIX_FMT = "nv12"
        from pipeline.ffmpeg_io import DecoderProcess, probe

        self.src = Path(src).resolve()
        self.start_sec = max(0.0, float(start_sec or 0.0))
        self._probe_info = probe(self.src)
        self._decoder = DecoderProcess(self.src, self.start_sec, self._probe_info, max_fps=max_fps)
        self.info = PyNvVideoInfo(
            width=int(self._decoder.out_info.width),
            height=int(self._decoder.out_info.height),
            fps=float(self._decoder.out_info.fps),
            duration=max(0.0, float(self._decoder.out_info.duration) - self.start_sec),
            codec_name=str(self._probe_info.codec_name),
            bitrate=0.0,
            num_frames=int(max(1, round(max(0.0, float(self._probe_info.duration) - self.start_sec) * self._decoder.out_info.fps))),
        )
        self._next_index = 0
        self._ended = False

    def __len__(self) -> int:
        return self.info.num_frames

    def frame_at(self, index: int) -> GpuNv12Frame:
        import numpy as np
        import cupy as cp

        if self._ended:
            raise RuntimeError("FFmpeg decoder has already ended")
        target = int(index)
        if target < self._next_index:
            raise ValueError(
                f"FFmpeg sequential decoder only supports monotonic access: "
                f"target={target} next_index={self._next_index}"
            )
        raw = None
        while self._next_index <= target:
            raw = self._decoder.read_frame()
            if raw is None:
                raise RuntimeError(f"FFmpeg decoder ended before frame {target}")
            self._next_index += 1
        assert raw is not None
        h, w = int(self.info.height), int(self.info.width)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h + h // 2, w)
        nv12 = cp.asarray(arr)
        y = nv12[:h, :]
        uv = nv12[h:, :]
        owner = nv12
        return GpuNv12Frame(
            owner=owner,
            y=CudaPlane.from_cupy_array(y),
            uv=CudaPlane.from_cupy_array(uv),
            width=w,
            height=h,
            pts=target,
        )

    def stop(self) -> None:
        if self._ended:
            return
        close = getattr(self._decoder, "close", None)
        if callable(close):
            close()
        self._ended = True


class CudaArrayView:
    """Expose a CuPy array slice through CUDA Array Interface for PyNv."""
    def __init__(self, arr: Any):
        self.arr = arr

    @property
    def __cuda_array_interface__(self):
        cai = dict(self.arr.__cuda_array_interface__)
        cai["shape"] = tuple(cai["shape"])
        if cai.get("strides") is not None:
            cai["strides"] = tuple(cai["strides"])
        return cai


class GpuNv12AppFrame:
    """AppFrame wrapper accepted by PyNvVideoCodec encoder for GPU NV12 input."""

    def __init__(self, nv12_dev: Any, width: int, height: int):
        self.nv12_dev = nv12_dev
        self.width = int(width)
        self.height = int(height)
        self.y = CudaArrayView(nv12_dev[: self.height, :].reshape(self.height, self.width, 1))
        self.uv = CudaArrayView(nv12_dev[self.height :, :].reshape(self.height // 2, self.width // 2, 2))

    def cuda(self):
        return [self.y, self.uv]
