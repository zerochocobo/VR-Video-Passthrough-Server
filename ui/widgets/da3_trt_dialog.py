"""TensorRT config dialog for the DA3 depth models (2D->3D/VR page).

Mirrors ui.widgets.trt_cache_dialog.TensorRTConfigDialog's look and interaction
(configure button -> dialog -> build/rebuild/delete + status), but targets the
DA3 Small/Base ONNX models. DA3's TensorRT path is ONNX Runtime's TensorRT EP
with an engine cache under runtime_cache/da3_trt/<variant>, built by
`offline/two_dvr.py build-trt`, so this dialog drives that command instead of the
RVM/MatAnyone2 trt_manifest warmup flow.
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
from ui.services.process_helpers import base_environment, two_dvr_command
from ui.settings import ROOT as UI_ROOT

DA3_VARIANTS = ("small", "base")


def da3_trt_cache_dir(variant: str) -> Path:
    return UI_ROOT / "runtime_cache" / "da3_trt" / variant


def da3_trt_cached(variant: str) -> bool:
    cache = da3_trt_cache_dir(variant)
    return cache.is_dir() and any(cache.glob("*.engine"))


def da3_trt_status() -> str:
    """'ready' if both variants are cached, 'missing' if neither, else 'stale'."""
    cached = [da3_trt_cached(v) for v in DA3_VARIANTS]
    if all(cached):
        return "ready"
    if not any(cached):
        return "missing"
    return "stale"


class Da3TrtConfigDialog(QDialog):
    def __init__(self, i18n, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
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
        self.fps_hint_label = QLabel(self.i18n.t("trt.fps_hint"))
        self.fps_hint_label.setWordWrap(True)
        self.fps_hint_label.setStyleSheet("color: #1677c7; font-weight: 600;")
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
        layout.addWidget(self.fps_hint_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.stage_label)
        layout.addLayout(buttons)
        self.setMinimumWidth(520)
        self.resize(520, 260)
        self._refresh()

    # -- info / status -------------------------------------------------------

    def _gpu_name(self) -> str:
        try:
            from utils.tensorrt_runtime_libs import check_tensorrt_runtime_libs

            return check_tensorrt_runtime_libs().gpu_name or "-"
        except Exception:
            return "-"

    def _refresh(self) -> None:
        status = da3_trt_status()
        per = "  ".join(
            f"{v.capitalize()}: {self.i18n.t('trt.status_' + ('ready' if da3_trt_cached(v) else 'missing'))}"
            for v in DA3_VARIANTS
        )
        self.info.setText("\n".join([
            f"{self.i18n.t('trt.model')}: DA3 Small + Base",
            f"{self.i18n.t('trt.precision')}: FP16",
            f"{self.i18n.t('trt.gpu')}: {self._gpu_name()}",
            f"{self.i18n.t('trt.cache_path')}: {da3_trt_cache_dir('small').parent}",
            per,
        ]))
        self.status_label.setText(f"{self.i18n.t('trt.cache_status')}: {self.i18n.t('trt.status_' + status)}")
        self.delete_button.setText(self.i18n.t("trt.delete_cache"))
        self.close_button.setText(self.i18n.t("button.close"))
        self.build_button.setText(self.i18n.t("trt.rebuild") if status in {"ready", "stale"} else self.i18n.t("trt.start_build"))
        self.delete_button.setVisible(status in {"ready", "stale"})

    # -- build ---------------------------------------------------------------

    def _start_build(self) -> None:
        if self.process is not None:
            return
        self._stages_done = 0
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.stage_label.setText(self.i18n.t("trt.building_model").format(model="DA3 Small + Base", precision="FP16"))
        self.build_button.setVisible(False)
        self.delete_button.setVisible(False)
        self.close_button.setEnabled(False)

        process = HiddenProcess(self)
        self.process = process
        process.stdout.connect(self._read_output)
        process.stderr.connect(self._read_output)
        process.finished.connect(self._build_finished)
        program, base_args = two_dvr_command()
        process.start(program, [*base_args, "build-trt", "--model", "both"], env=base_environment())

    def _read_output(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        for line in text.splitlines():
            if "build-trt:" in line and "ready" in line:
                self._stages_done = min(2, self._stages_done + 1)
                self.progress.setValue(int(self._stages_done * 99 / 2))
            if "build-trt:" in line:
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

        for v in DA3_VARIANTS:
            cache = da3_trt_cache_dir(v)
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
        self._refresh()

    def closeEvent(self, event) -> None:
        if self.process is not None:
            self.process.kill()
            self.process = None
        super().closeEvent(event)
