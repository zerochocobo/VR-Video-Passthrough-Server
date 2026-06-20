"""Confirmation + progress dialog for fetching missing ONNX models.

Given a list of missing files, it probes each remote size (mirror-aware), shows a
confirmation listing every file and the total, and on confirm downloads them with
per-file and overall progress. ``exec()`` returns ``Accepted`` only when every
file downloaded successfully, so callers can gate a run on the result.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from utils import hf_download


@dataclass
class DownloadItem:
    label: str
    dest: Path
    urls: list[str]
    size: int = 0  # bytes; filled in by the size probe


def _mb(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


class _DownloadSignals(QObject):
    sizes_ready = Signal(object)          # list[DownloadItem]
    progress = Signal(int, int, str)      # overall_done, overall_total, current label
    finished = Signal(bool, str)          # ok, error


class ModelDownloadDialog(QDialog):
    def __init__(self, i18n, items: list[DownloadItem], parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.items = items
        self.signals = _DownloadSignals(self)
        self.downloading = False
        self.total_bytes = 0
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("modeldl.title"))

        self.intro = QLabel(self.i18n.t("modeldl.intro"))
        self.intro.setWordWrap(True)
        self.intro.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.files_label = QLabel(self.i18n.t("modeldl.checking"))
        self.files_label.setWordWrap(True)
        self.files_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.download_button = QPushButton(self.i18n.t("modeldl.download"))
        self.cancel_button = QPushButton(self.i18n.t("button.cancel"))
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._start_download)
        self.cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.download_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self.intro)
        layout.addWidget(self.files_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        self.setMinimumWidth(560)

        self.signals.sizes_ready.connect(self._on_sizes_ready)
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self._probe_sizes()

    # -- size probe ----------------------------------------------------------

    def _probe_sizes(self) -> None:
        def worker() -> None:
            for item in self.items:
                try:
                    item.size = hf_download.remote_size(item.urls)
                except Exception:
                    item.size = 0
            self.signals.sizes_ready.emit(self.items)

        threading.Thread(target=worker, name="model-size-probe", daemon=True).start()

    def _on_sizes_ready(self, items: list[DownloadItem]) -> None:
        self.total_bytes = sum(max(0, i.size) for i in items)
        lines = []
        for item in items:
            if item.size > 0:
                lines.append(f"• {item.label}  —  {_mb(item.size):.1f} MB")
            else:
                lines.append(f"• {item.label}  —  {self.i18n.t('modeldl.unknown_size')}")
        if self.total_bytes > 0:
            lines.append("")
            lines.append(self.i18n.t("modeldl.total").format(size=f"{_mb(self.total_bytes):.1f} MB"))
        self.files_label.setText("\n".join(lines))
        self.download_button.setEnabled(True)

    # -- download ------------------------------------------------------------

    def _start_download(self) -> None:
        if self.downloading:
            return
        self.downloading = True
        self.download_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("")

        items = self.items
        total = self.total_bytes

        def worker() -> None:
            completed = 0  # bytes finished in earlier files
            try:
                for item in items:
                    def on_progress(done: int, _file_total: int, _label=item.label, _base=completed) -> None:
                        self.signals.progress.emit(_base + done, total, _label)

                    hf_download.download_file(item.urls, item.dest, progress=on_progress)
                    completed += max(item.size, item.dest.stat().st_size if item.dest.exists() else 0)
            except Exception as exc:
                self.signals.finished.emit(False, f"{type(exc).__name__}: {exc}")
            else:
                self.signals.finished.emit(True, "")

        threading.Thread(target=worker, name="model-download", daemon=True).start()

    def _on_progress(self, overall_done: int, overall_total: int, label: str) -> None:
        if overall_total > 0:
            percent = max(0, min(100, int(overall_done * 100 / overall_total)))
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            self.progress.setFormat(f"{percent}%")
            self.status.setText(
                self.i18n.t("modeldl.downloading").format(
                    name=label, done=f"{_mb(overall_done):.1f}", total=f"{_mb(overall_total):.1f}"
                )
            )
        else:
            self.progress.setRange(0, 0)  # indeterminate
            self.status.setText(
                self.i18n.t("modeldl.downloading_unknown").format(name=label, done=f"{_mb(overall_done):.1f}")
            )

    def _on_finished(self, ok: bool, error: str) -> None:
        self.downloading = False
        self.cancel_button.setEnabled(True)
        if ok:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("100%")
            self.status.setText(self.i18n.t("modeldl.done"))
            self.accept()
        else:
            self.progress.setVisible(False)
            self.download_button.setEnabled(True)
            self.status.setText(self.i18n.t("modeldl.failed").format(error=error))

    def closeEvent(self, event) -> None:
        if self.downloading:
            event.ignore()
            return
        super().closeEvent(event)
