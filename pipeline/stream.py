"""Legacy FFmpeg subprocess passthrough stream.

Pipeline:
- source video -> FFmpeg seek/decode -> Matter matting/composite -> FFmpeg encode
- decoder, producer, writer, and reader run in separate threads
- the async iterator receives encoded bytes through an asyncio.Queue
- close() tears down subprocesses first so blocked reads/writes unblock quickly
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from pipeline.ffmpeg_io import DecoderProcess, EncoderProcess, VideoInfo, probe
from pipeline.matting import Matter
from utils.logger import get

log = get("stream")

# Emit a timing summary every N frames.
_DIAG_INTERVAL = 30
# Queue used by the reader thread to feed the async response.
_QUEUE_MAX = 8
# Bytes read from encoder stdout per read call.
_READ_CHUNK = 256 * 1024
# Maximum time to wait for daemon worker threads during close().
_THREAD_JOIN_TIMEOUT = 0.1
# Decoder prefetch queue. max=2 lets matting usually have one frame ready.
_DEC_QUEUE_MAX = 2
# Encoded input frame queue. The writer owns blocking writes to encoder.stdin.
_ENC_QUEUE_MAX = 4


class PassthroughStream:
    """Manage one legacy FFmpeg passthrough request lifecycle."""

    def __init__(
        self,
        src: Path,
        start_sec: float,
        matter: Matter,
        container: str | None = None,
        max_fps: float | None = None,
        audio_mode: str | None = None,
    ):
        self.src = src
        self.start_sec = start_sec
        self.matter = matter
        self.container = container
        self.max_fps = max_fps
        self.audio_mode = (audio_mode or "off").lower()
        self.bytes_emitted = 0
        self.frames_produced = 0
        self.output_fps = 0.0
        self.info: VideoInfo | None = None
        self.dec: DecoderProcess | None = None
        self.enc: EncoderProcess | None = None
        self._producer: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._decoder: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False
        self._queue: asyncio.Queue | None = None
        self._dec_queue: queue.Queue | None = None
        self._enc_queue: queue.Queue | None = None
        self._enc_free_queue: queue.Queue | None = None
        self._enc_pool_refs: list[object] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------- Decoder thread: read dec.read_frame() into _dec_queue -------------
    def _decoder_loop(self):
        assert self.dec and self._dec_queue is not None
        n = 0
        t_total = 0.0
        t_start = time.perf_counter()
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    raw = self.dec.read_frame()
                except Exception as e:
                    log.warning("[DIAG] decoder thread read exception: %s", e)
                    break
                t1 = time.perf_counter()
                t_total += t1 - t0
                if raw is None:
                    log.info("[DIAG] decoder thread EOF after %d frames", n)
                    break
                # Keep reading from FFmpeg even when the producer is briefly
                # behind, otherwise the decoder stdout pipe can fill and stall.
                while not self._stop.is_set():
                    try:
                        self._dec_queue.put(raw, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return
                n += 1
        except Exception as e:
            log.error("[DIAG] decoder thread exception: %s\n%s", e, traceback.format_exc(limit=4))
        finally:
            elapsed = time.perf_counter() - t_start
            if n > 0:
                log.info(
                    "[DIAG] decoder thread summary: %d frames in %.2fs (avg read=%.1fms)",
                    n, elapsed, t_total / n * 1000,
                )
            try:
                self._dec_queue.put_nowait(None)  # Sentinel for producer shutdown.
            except Exception:
                pass

    # ------------- Writer thread: drain _enc_queue into encoder.stdin -------------
    def _writer_loop(self):
        assert self.enc and self._enc_queue is not None
        n = 0
        t_total = 0.0
        t_start = time.perf_counter()
        try:
            while not self._stop.is_set():
                item = self._enc_queue.get()
                if item is None:
                    break
                frame, release_to_pool = item
                t0 = time.perf_counter()
                try:
                    ok = self.enc.write_frame(memoryview(np.ascontiguousarray(frame)).cast("B"))
                finally:
                    if release_to_pool and self._enc_free_queue is not None:
                        try:
                            self._enc_free_queue.put_nowait(frame)
                        except queue.Full:
                            pass
                t1 = time.perf_counter()
                t_total += t1 - t0
                n += 1
                if not ok:
                    log.warning("[DIAG] writer enc.write_frame returned False at frame %d", n)
                    break
        except Exception as e:
            log.error("[DIAG] writer exception: %s\n%s", e, traceback.format_exc(limit=4))
        finally:
            try:
                if self.enc and self.enc.proc.stdin:
                    self.enc.proc.stdin.close()
            except Exception:
                pass
            elapsed = time.perf_counter() - t_start
            if n > 0:
                log.info(
                    "[DIAG] writer summary: %d frames in %.2fs (avg enc_write=%.1fms)",
                    n, elapsed, t_total / n * 1000,
                )

    # ------------- _dec_queue + + _enc_queue -------------
    def _producer_loop(self):
        assert self.info and self.dec and self.enc and self._dec_queue is not None and self._enc_queue is not None
        w, h = self.dec.out_info.width, self.dec.out_info.height
        log.info(
            "[DIAG] producer start: %s  src=%dx%d out=%dx%d fps=%.2f duration=%.1fs",
            self.src.name, self.info.width, self.info.height, w, h, self.info.fps, self.info.duration,
        )

        n = 0
        t_total_decode = 0.0  # queue.get
        t_total_mat = 0.0
        t_total_enc_queue = 0.0
        t_loop_start = time.perf_counter()

        try:
            while not self._stop.is_set():
                # ---- stop ----
                t0 = time.perf_counter()
                raw = None
                while not self._stop.is_set():
                    try:
                        raw = self._dec_queue.get(timeout=0.5)
                        break
                    except queue.Empty:
                        continue
                else:
                    break
                t1 = time.perf_counter()
                if raw is None:
                    log.info("[DIAG] producer got EOF sentinel after %d frames", n)
                    break

                # ---- ----
                release_to_pool = False
                composed = None
                out_buf = None
                try:
                    if self.dec.out_info.pix_fmt == "nv12":
                        frame = np.frombuffer(raw, dtype=np.uint8)
                        if self._enc_free_queue is not None:
                            while not self._stop.is_set():
                                try:
                                    out_buf = self._enc_free_queue.get(timeout=0.5)
                                    break
                                except queue.Empty:
                                    continue
                        composed, _ = self.matter.composite_green_nv12_to_nv12_profile(frame, h, w, out=out_buf)
                        release_to_pool = out_buf is not None
                    else:
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                        composed = self.matter.composite_green(frame)
                except Exception as e:
                    if out_buf is not None and self._enc_free_queue is not None:
                        try:
                            self._enc_free_queue.put_nowait(out_buf)
                        except queue.Full:
                            pass
                    log.warning("matting failed, fallback raw: %s", e)
                    composed = np.frombuffer(raw, dtype=np.uint8)
                    release_to_pool = False
                t2 = time.perf_counter()

                # ---- ----
                if release_to_pool:
                    composed_for_writer = np.ascontiguousarray(composed)
                else:
                    # composed pinned buffer writer
                    composed_for_writer = np.ascontiguousarray(composed).copy()
                queued_for_writer = False
                while not self._stop.is_set():
                    try:
                        self._enc_queue.put((composed_for_writer, release_to_pool), timeout=0.5)
                        queued_for_writer = True
                        break
                    except queue.Full:
                        continue
                if release_to_pool and not queued_for_writer and self._enc_free_queue is not None:
                    try:
                        self._enc_free_queue.put_nowait(composed_for_writer)
                    except queue.Full:
                        pass
                t3 = time.perf_counter()

                t_total_decode += t1 - t0
                t_total_mat += t2 - t1
                t_total_enc_queue += t3 - t2
                n += 1
                self.frames_produced = n

                if n == 1:
                    log.info(
                        "[DIAG] frame #1: dec_wait=%.3fs  matting=%.3fs  enc_queue=%.3fs",
                        t1 - t0, t2 - t1, t3 - t2,
                    )

                if n % _DIAG_INTERVAL == 0:
                    elapsed = time.perf_counter() - t_loop_start
                    fps_actual = n / elapsed if elapsed > 0 else 0
                    avg_dec = t_total_decode / n * 1000
                    avg_mat = t_total_mat / n * 1000
                    avg_enc = t_total_enc_queue / n * 1000
                    log.info(
                        "[DIAG] frame %d | fps=%.2f (src=%.2f) | "
                        "avg dec_wait=%.1fms  matting=%.1fms  enc_queue=%.1fms  enc_q=%d",
                        n, fps_actual, self.info.fps,
                        avg_dec, avg_mat, avg_enc, self._enc_queue.qsize(),
                    )
        except Exception as e:
            log.error("[DIAG] producer exception: %s\n%s", e, traceback.format_exc(limit=8))
        finally:
            try:
                self._enc_queue.put_nowait(None)
            except Exception:
                pass
            elapsed = time.perf_counter() - t_loop_start
            fps_final = n / elapsed if elapsed > 0 else 0
            if n > 0:
                log.info(
                    "[DIAG] producer summary: %d frames in %.2fs %.2f fps "
                    "(avg: dec_wait=%.1fms  matting=%.1fms  enc_queue=%.1fms)",
                    n, elapsed, fps_final,
                    t_total_decode / n * 1000,
                    t_total_mat / n * 1000,
                    t_total_enc_queue / n * 1000,
                )
            log.info("producer done: %s @%.2fs", self.src.name, self.start_sec)

    # ------------- Reader thread: bridge encoder.stdout into asyncio.Queue -------------
    def _reader_loop(self):
        assert self.enc and self._queue is not None and self._loop is not None
        stdout = self.enc.proc.stdout
        n_chunks = 0
        t_start = time.perf_counter()
        try:
            while not self._stop.is_set():
                try:
                    data = stdout.read(_READ_CHUNK)
                except (ValueError, OSError):
                    # close() may close stdout while this thread is blocked in read().
                    break
                if not data:
                    log.info("[DIAG] reader: encoder stdout EOF after %d chunks", n_chunks)
                    break

                # The queue belongs to the event loop, so the reader thread
                # submits the put coroutine and waits with a timeout.
                try:
                    fut = asyncio.run_coroutine_threadsafe(self._queue.put(data), self._loop)
                except RuntimeError:
                    # The loop is already closing.
                    break
                while not self._stop.is_set():
                    try:
                        fut.result(timeout=0.5)
                        break
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        log.warning("[DIAG] reader: queue.put failed: %s", e)
                        return
                else:
                    fut.cancel()
                    return

                n_chunks += 1
                if n_chunks % 200 == 0:
                    elapsed = time.perf_counter() - t_start
                    mbps = (n_chunks * _READ_CHUNK) / elapsed / 1_000_000 if elapsed > 0 else 0
                    log.info("[DIAG] reader: %d chunks (%.1f MB/s)", n_chunks, mbps)
        except Exception as e:
            log.error("[DIAG] reader exception: %s\n%s", e, traceback.format_exc(limit=4))
        finally:
            # Wake the async iterator even if the encoder exits without bytes.
            self._post_sentinel()

    def _post_sentinel(self):
        if self._queue is None or self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)
        except RuntimeError:
            pass

    # ------------- Queue -------------
    async def iter_bytes(self) -> AsyncIterator[bytes]:
        self.matter.reset_state()
        self.info = probe(self.src)
        self.dec = DecoderProcess(self.src, self.start_sec, self.info, max_fps=self.max_fps)
        self.output_fps = float(self.dec.out_info.fps or self.info.fps or 0.0)
        enc_pix_fmt = "nv12" if self.dec.out_info.pix_fmt == "nv12" else "bgr24"
        audio_src = self.src if self.container == "mpegts" and self.audio_mode in {"aac", "copy"} else None
        self.enc = EncoderProcess(
            self.dec.out_info.width,
            self.dec.out_info.height,
            self.dec.out_info.fps,
            input_pix_fmt=enc_pix_fmt,
            container=self.container,
            audio_src=audio_src,
            audio_start_sec=self.start_sec,
        )
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._dec_queue = queue.Queue(maxsize=_DEC_QUEUE_MAX)
        self._enc_queue = queue.Queue(maxsize=_ENC_QUEUE_MAX)
        if self.dec.out_info.pix_fmt == "nv12":
            pool = self.matter.make_pinned_nv12_output_pool(
                self.dec.out_info.height,
                self.dec.out_info.width,
                _ENC_QUEUE_MAX + 1,
            )
            self._enc_pool_refs = [mem for mem, _ in pool]
            self._enc_free_queue = queue.Queue(maxsize=len(pool))
            for _, arr in pool:
                self._enc_free_queue.put_nowait(arr)
            log.info("[DIAG] encoder handoff pool: %d pinned NV12 frames", len(pool))
        self._decoder = threading.Thread(target=self._decoder_loop, name="pt-decoder", daemon=True)
        self._producer = threading.Thread(target=self._producer_loop, name="pt-producer", daemon=True)
        self._writer = threading.Thread(target=self._writer_loop, name="pt-writer", daemon=True)
        self._reader = threading.Thread(target=self._reader_loop, name="pt-reader", daemon=True)
        self._decoder.start()
        self._producer.start()
        self._writer.start()
        self._reader.start()

        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                self.bytes_emitted += len(chunk)
                yield chunk
        finally:
            # finally CancelledError
            self.close()

    # ------------- IO -------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()

        # 1) kill encoder reader.stdout.read() 0
        # producer.write_frame() BrokenPipe
        if self.enc:
            try:
                rc_before = self.enc.proc.poll()
                log.info("[DIAG] decoder/encoder close start (encoder rc=%s)", rc_before)
                self.enc.close()
                log.info("[DIAG] encoder returncode(after-close)=%s", self.enc.proc.poll())
            except Exception as e:
                log.warning("encoder close error: %s", e)

        # 2) kill decoder producer read_frame EOF
        if self.dec:
            try:
                self.dec.close()
                log.info("[DIAG] decoder returncode(after-close)=%s", self.dec.proc.poll())
            except Exception as e:
                log.warning("decoder close error: %s", e)

        # 3) await queue.get
        self._post_sentinel()
        if self._enc_queue is not None:
            try:
                self._enc_queue.put_nowait(None)
            except Exception:
                pass

        # 4) join daemon=True
        for thread in (self._reader, self._producer, self._writer, self._decoder):
            if thread and thread.is_alive():
                thread.join(timeout=_THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    log.warning("[DIAG] %s thread still alive after close: %s", thread.name, self.src.name)
