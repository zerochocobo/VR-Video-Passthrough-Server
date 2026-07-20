"""Interactive SAM3 keyword preview dialog for the offline passthrough page."""
from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)

import config

PREVIEW_MAX_WIDTH = 1280
PREVIEW_MIN_IMAGE_SIZE = (320, 240)
# Frames at least this wide relative to height are treated as side-by-side VR
# and only the left eye is previewed, matching the offline per-eye prepass.
_SBS_ASPECT_THRESHOLD = 1.98
_SLIDER_DEBOUNCE_MS = 250


def _sam3_model_dir() -> Path:
    return config.ROOT / "models" / "sam3_onnx"


def _sam3_preview_providers(cuda_memory_limit_mb: int) -> list:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]
    options = {}
    if cuda_memory_limit_mb > 0:
        options["gpu_mem_limit"] = str(int(cuda_memory_limit_mb) * 1024 * 1024)
        options["arena_extend_strategy"] = "kSameAsRequested"
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]


class _Sam3PreviewWorker(QThread):
    """Owns the SAM3 sessions; loads once, then serves coalesced inference requests."""

    load_progress = Signal(int, str)  # percent, stage key suffix
    load_finished = Signal(bool, str)  # ok, error text
    infer_finished = Signal(int, object, object)  # request_id, overlay RGB ndarray or None, info dict or error text

    def __init__(self, prompt: str, parent=None) -> None:
        super().__init__(parent)
        self._prompt = prompt
        self._commands: queue.Queue = queue.Queue()

    def set_prompt(self, prompt: str) -> None:
        self._commands.put(("prompt", prompt))

    def request_infer(self, request_id: int, frame_rgb: np.ndarray) -> None:
        self._commands.put(("infer", request_id, frame_rgb))

    def stop(self) -> None:
        self._commands.put(("stop",))

    def run(self) -> None:  # noqa: C901
        from offline.sam3_matanyone2 import Sam3TextMasker, clear_gpu_memory_pools
        from utils.runtime_dll_paths import apply_runtime_dll_paths

        # The UI process does not set up CUDA DLL paths on its own (the server
        # and offline tools do it in their entry points), so ORT would silently
        # fall back to CPU without this.
        apply_runtime_dll_paths()
        masker = None
        try:
            model_dir = _sam3_model_dir()
            self.load_progress.emit(5, "language")
            masker = Sam3TextMasker(
                model_dir,
                self._prompt,
                _sam3_preview_providers(8192),
                decoder_providers=_sam3_preview_providers(4096),
                low_memory=True,
            )
            self.load_progress.emit(25, "encoder")
            masker.image_encoder = masker._image_encoder_session()
            self.load_progress.emit(80, "decoder")
            masker.decoder = masker._decoder_session()
            masker.low_memory = False
            self.load_progress.emit(100, "ready")
            self.load_finished.emit(True, "")
        except Exception as exc:
            self.load_finished.emit(False, f"{type(exc).__name__}: {exc}")
            masker = None
            clear_gpu_memory_pools()
            return

        pending_prompt: str | None = None
        pending_infer: tuple[int, np.ndarray] | None = None
        stopping = False
        try:
            while not stopping:
                command = self._commands.get()
                # Coalesce queued commands so only the latest prompt/frame is processed.
                while True:
                    if command[0] == "stop":
                        stopping = True
                        break
                    if command[0] == "prompt":
                        pending_prompt = str(command[1])
                    elif command[0] == "infer":
                        pending_infer = (int(command[1]), command[2])
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                if stopping:
                    break
                if pending_prompt is not None:
                    prompt = pending_prompt
                    pending_prompt = None
                    if prompt != masker.prompt:
                        try:
                            masker.set_prompt(prompt)
                        except Exception as exc:
                            if pending_infer is not None:
                                request_id, _ = pending_infer
                                pending_infer = None
                                self.infer_finished.emit(request_id, None, f"{type(exc).__name__}: {exc}")
                            continue
                if pending_infer is None:
                    continue
                request_id, frame_rgb = pending_infer
                pending_infer = None
                try:
                    mask, info = masker.mask(frame_rgb)
                    overlay = self._compose_overlay(frame_rgb, mask, info)
                    self.infer_finished.emit(request_id, overlay, info)
                except RuntimeError as exc:
                    if "masks for text prompt" in str(exc):
                        self.infer_finished.emit(request_id, frame_rgb, {"selected": [], "union_area_ratio": 0.0})
                    else:
                        self.infer_finished.emit(request_id, None, f"{type(exc).__name__}: {exc}")
                except Exception as exc:
                    self.infer_finished.emit(request_id, None, f"{type(exc).__name__}: {exc}")
        finally:
            if masker is not None:
                masker.image_encoder = None
                masker.decoder = None
                del masker
            clear_gpu_memory_pools()

    @staticmethod
    def _compose_overlay(frame_rgb: np.ndarray, mask: np.ndarray, info: dict) -> np.ndarray:
        import cv2

        overlay = frame_rgb.copy()
        selected = mask >= 0.5
        if np.any(selected):
            tint = overlay[selected].astype(np.float32)
            tint[:, 0] = tint[:, 0] * 0.45
            tint[:, 1] = np.minimum(255.0, tint[:, 1] * 0.55 + 140.0)
            tint[:, 2] = tint[:, 2] * 0.45
            overlay[selected] = tint.astype(np.uint8)
            contours, _ = cv2.findContours(
                selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 80), 2)
        bbox = info.get("union_bbox_xyxy") or []
        if len(bbox) == 4:
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 220, 0), 2)
        return overlay


class Sam3PreviewDialog(QDialog):
    def __init__(self, i18n, settings, parent=None, video_path: str = "", on_prompt_saved=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.settings = settings
        self._on_prompt_saved = on_prompt_saved
        self._capture = None
        self._frame_count = 0
        self._fps = 0.0
        self._current_frame: np.ndarray | None = None
        self._model_ready = False
        self._request_id = 0
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("offline.sam3_preview_title"))
        self.resize(480, 480)

        layout = QVBoxLayout(self)

        video_row = QHBoxLayout()
        video_label = QLabel(self.i18n.t("offline.video"))
        self.video_edit = QLineEdit(video_path)
        browse = QPushButton("...")
        browse.clicked.connect(self._browse_video)
        self.video_edit.editingFinished.connect(self._open_video)
        video_row.addWidget(video_label)
        video_row.addWidget(self.video_edit, 1)
        video_row.addWidget(browse)
        layout.addLayout(video_row)

        prompt_row = QHBoxLayout()
        prompt_label = QLabel(self.i18n.t("offline.sam3_prompt_label"))
        prompt = str(self.settings.data.get("offline_sam3_prompt") or "").strip() or "person"
        self.prompt_edit = QLineEdit(prompt)
        self.prompt_edit.editingFinished.connect(self._apply_prompt)
        self.refresh_button = QPushButton(self.i18n.t("offline.sam3_preview_refresh"))
        self.refresh_button.clicked.connect(self._show_current_frame)
        self.save_button = QPushButton(self.i18n.t("button.save"))
        self.save_button.clicked.connect(self._save_prompt)
        prompt_row.addWidget(prompt_label)
        prompt_row.addWidget(self.prompt_edit, 1)
        prompt_row.addWidget(self.refresh_button)
        prompt_row.addWidget(self.save_button)
        layout.addLayout(prompt_row)

        hint = QLabel(self.i18n.t("offline.sam3_prompt_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5f6368;")
        layout.addWidget(hint)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(*PREVIEW_MIN_IMAGE_SIZE)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #202124;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.time_label)
        layout.addLayout(slider_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        close_button = QPushButton(self.i18n.t("button.close"))
        close_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(_SLIDER_DEBOUNCE_MS)
        self._seek_timer.timeout.connect(self._show_current_frame)

        self.worker = _Sam3PreviewWorker(prompt, self)
        self.worker.load_progress.connect(self._on_load_progress)
        self.worker.load_finished.connect(self._on_load_finished)
        self.worker.infer_finished.connect(self._on_infer_finished)
        self.worker.start()

        if video_path:
            self._open_video()

    # ------------------------------------------------------------------ video
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self.i18n.t("file.select_video"), "", "Videos (*.mp4 *.mkv *.mov *.m4v)"
        )
        if path:
            self.video_edit.setText(path)
            self._open_video()

    def _open_video(self) -> None:
        import cv2

        path = self.video_edit.text().strip()
        if not path:
            return
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self.slider.setEnabled(False)
            self.status_label.setText(self.i18n.t("offline.sam3_preview_video_failed").format(path=path))
            return
        self._capture = capture
        self._frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._frame_count - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        self._show_current_frame()

    def _format_seconds(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_time_label(self) -> None:
        current = self.slider.value() / self._fps if self._fps > 0 else 0.0
        total = self._frame_count / self._fps if self._fps > 0 else 0.0
        self.time_label.setText(f"{self._format_seconds(current)} / {self._format_seconds(total)}")

    def _on_slider_changed(self, _value: int) -> None:
        self._update_time_label()
        self._seek_timer.start()

    def _show_current_frame(self) -> None:
        import cv2

        if self._capture is None:
            return
        index = self.slider.value()
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        if height > 0 and width / height >= _SBS_ASPECT_THRESHOLD:
            frame_rgb = frame_rgb[:, : width // 2]
        if frame_rgb.shape[1] > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / frame_rgb.shape[1]
            frame_rgb = cv2.resize(
                frame_rgb,
                (PREVIEW_MAX_WIDTH, max(1, int(round(frame_rgb.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        self._current_frame = np.ascontiguousarray(frame_rgb)
        self._display_image(self._current_frame)
        self._update_time_label()
        self._request_inference()

    # ------------------------------------------------------------------ model
    def _on_load_progress(self, percent: int, stage: str) -> None:
        self.progress.setValue(percent)
        if stage != "ready":
            self.status_label.setText(
                self.i18n.t("offline.sam3_preview_loading").format(
                    stage=self.i18n.t(f"offline.sam3_preview_stage_{stage}")
                )
            )

    def _on_load_finished(self, ok: bool, error: str) -> None:
        if not ok:
            self.progress.setVisible(False)
            self.status_label.setText(self.i18n.t("offline.sam3_preview_load_failed").format(error=error))
            return
        self._model_ready = True
        self.progress.setVisible(False)
        self.status_label.setText(self.i18n.t("offline.sam3_preview_ready"))
        self._request_inference()

    # -------------------------------------------------------------- inference
    def _apply_prompt(self) -> None:
        if not self._model_ready:
            return
        self.worker.set_prompt(self.prompt_edit.text())
        self._request_inference()

    def _save_prompt(self) -> None:
        prompt = self.prompt_edit.text().strip() or "person"
        self.prompt_edit.setText(prompt)
        self.settings.data["offline_sam3_prompt"] = prompt
        self.settings.save()
        if callable(self._on_prompt_saved):
            self._on_prompt_saved()
        self.status_label.setText(self.i18n.t("offline.sam3_preview_saved"))
        self._apply_prompt()

    def _request_inference(self) -> None:
        if not self._model_ready or self._current_frame is None:
            return
        self._request_id += 1
        self.status_label.setText(self.i18n.t("offline.sam3_preview_running"))
        self.worker.set_prompt(self.prompt_edit.text())
        self.worker.request_infer(self._request_id, self._current_frame)

    def _on_infer_finished(self, request_id: int, overlay, info) -> None:
        if request_id != self._request_id:
            return
        if overlay is None:
            self.status_label.setText(self.i18n.t("offline.sam3_preview_no_result") + f" ({info})")
            if self._current_frame is not None:
                self._display_image(self._current_frame)
            return
        self._display_image(overlay)
        selected = info.get("selected") or []
        ratio = float(info.get("union_area_ratio") or 0.0) * 100.0
        if selected:
            self.status_label.setText(
                self.i18n.t("offline.sam3_preview_result").format(count=len(selected), ratio=f"{ratio:.1f}")
            )
        else:
            self.status_label.setText(self.i18n.t("offline.sam3_preview_no_result"))

    # ---------------------------------------------------------------- display
    def _display_image(self, frame_rgb: np.ndarray) -> None:
        height, width = frame_rgb.shape[:2]
        image = QImage(frame_rgb.data, width, height, frame_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._last_pixmap = pixmap

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        pixmap = getattr(self, "_last_pixmap", None)
        if pixmap is not None:
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    # ---------------------------------------------------------------- cleanup
    def closeEvent(self, event) -> None:
        self._shutdown()
        super().closeEvent(event)

    def reject(self) -> None:
        self._shutdown()
        super().reject()

    def _shutdown(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(15000)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
