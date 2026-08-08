"""TensorRT config dialog for the offline face-beautification models.

Mirrors ui.widgets.da3_trt_dialog.Da3TrtConfigDialog: a modal dialog that drives
``offline/face_beauty.py build-trt`` and reports per-stage progress. Face beauty
loads up to four graphs (detector / landmarker / parser / enhancer) and the
GFPGAN-class enhancer alone can take several minutes to compile, which is why
the page routes the first build through this dialog instead of stalling silently
inside the conversion run.

The build args must match the ones the conversion will use -- the engine cache
is keyed per model, and the detector cache is keyed per input size.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ui.log_sanitizer import clean_log_text
from ui.services.hidden_process import HiddenProcess
from ui.services.process_helpers import base_environment, face_beauty_command
from ui.settings import ROOT as UI_ROOT

TRT_CACHE_NAME = "face_beauty_trt"


def face_beauty_trt_cache_root() -> Path:
    return UI_ROOT / "runtime_cache" / TRT_CACHE_NAME


def face_beauty_trt_cached(key: str) -> bool:
    cache = face_beauty_trt_cache_root() / key
    return cache.is_dir() and any(cache.glob("*.engine"))


def face_beauty_trt_status(keys: list[str]) -> str:
    """'ready' if every required engine is cached, 'missing' if none, else 'stale'."""
    cached = [face_beauty_trt_cached(key) for key in keys]
    if cached and all(cached):
        return "ready"
    if not any(cached):
        return "missing"
    return "stale"


class FaceBeautyTrtConfigDialog(QDialog):
    """``keys`` are the engine-cache keys this configuration needs; ``build_args``
    the ``face_beauty.py`` flags that select exactly those models."""

    def __init__(self, i18n, keys: list[str], build_args: list[str], parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.keys = list(keys)
        self.build_args = list(build_args)
        self.process: HiddenProcess | None = None
        self._stages_done = 0
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("trt.title"))

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: 700;")
        self.warning_label = QLabel(self.i18n.t("beauty.trt_warning"))
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #1677c7; font-weight: 600;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.stage_label = QLabel("")
        self.stage_label.setWordWrap(True)
        self.stage_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.delete_button = QPushButton()
        self.close_button = QPushButton()
        self.build_button = QPushButton()
        self.delete_button.clicked.connect(self._delete_cache)
        self.close_button.clicked.connect(self.close)
        self.build_button.clicked.connect(self._start_build)

        buttons = QHBoxLayout()
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.build_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self.info)
        layout.addWidget(self.status_label)
        layout.addWidget(self.warning_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.stage_label)
        layout.addLayout(buttons)
        self.setMinimumWidth(540)
        self.resize(540, 300)
        self._refresh()

    # -- info / status -------------------------------------------------------

    def _gpu_name(self) -> str:
        try:
            from utils.tensorrt_runtime_libs import check_tensorrt_runtime_libs

            return check_tensorrt_runtime_libs().gpu_name or "-"
        except Exception:
            return "-"

    def _refresh(self) -> None:
        status = face_beauty_trt_status(self.keys)
        per = "\n".join(
            f"  {key}: {self.i18n.t('trt.status_' + ('ready' if face_beauty_trt_cached(key) else 'missing'))}"
            for key in self.keys
        )
        self.info.setText("\n".join([
            f"{self.i18n.t('trt.model')}: {', '.join(self.keys)}",
            f"{self.i18n.t('trt.precision')}: FP16",
            f"{self.i18n.t('trt.gpu')}: {self._gpu_name()}",
            f"{self.i18n.t('trt.cache_path')}: {face_beauty_trt_cache_root()}",
            per,
        ]))
        self.status_label.setText(f"{self.i18n.t('trt.cache_status')}: {self.i18n.t('trt.status_' + status)}")
        self.delete_button.setText(self.i18n.t("trt.delete_cache"))
        self.close_button.setText(self.i18n.t("button.close"))
        self.build_button.setText(
            self.i18n.t("trt.rebuild") if status in {"ready", "stale"} else self.i18n.t("trt.start_build")
        )
        self.delete_button.setVisible(status in {"ready", "stale"})

    def is_ready(self) -> bool:
        return face_beauty_trt_status(self.keys) == "ready"

    # -- build ---------------------------------------------------------------

    def _start_build(self) -> None:
        if self.process is not None:
            return
        self._stages_done = 0
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.stage_label.setText(
            self.i18n.t("trt.building_model").format(model=", ".join(self.keys), precision="FP16")
        )
        self.build_button.setVisible(False)
        self.delete_button.setVisible(False)
        self.close_button.setEnabled(False)

        process = HiddenProcess(self)
        self.process = process
        process.stdout.connect(self._read_output)
        process.stderr.connect(self._read_output)
        process.finished.connect(self._build_finished)
        program, base_args = face_beauty_command()
        process.start(program, [*base_args, "build-trt", *self.build_args], env=base_environment())

    def _read_output(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        total = max(1, len(self.keys))
        for line in text.splitlines():
            if "build-trt:" not in line:
                continue
            if " ready in " in line:
                self._stages_done = min(total, self._stages_done + 1)
                self.progress.setValue(int(self._stages_done * 99 / total))
            self.stage_label.setText(line.split("build-trt:", 1)[-1].strip())

    def _build_finished(self, exit_code: int) -> None:
        self.process = None
        self.close_button.setEnabled(True)
        if exit_code == 0:
            self.progress.setValue(100)
        self._refresh()
        if exit_code != 0:
            self.progress.setVisible(True)
            self.stage_label.setText(self.i18n.t("trt.build_failed").format(error=f"exit code {exit_code}"))

    def _delete_cache(self) -> None:
        import shutil

        for key in self.keys:
            cache = face_beauty_trt_cache_root() / key
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
        self._refresh()

    def closeEvent(self, event) -> None:
        if self.process is not None:
            self.process.kill()
            self.process = None
        super().closeEvent(event)
