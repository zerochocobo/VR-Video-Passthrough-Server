from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from media_library import is_unc_path


class VideoDirsDialog(QDialog):
    def __init__(self, i18n, directories: list[str], parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: #5f6368; font-size: 8.5pt;")
        self.list_widget = QListWidget()
        for directory in directories:
            text = directory.strip()
            if text and not is_unc_path(text):
                self.list_widget.addItem(text)

        self.add_button = QPushButton()
        self.remove_button = QPushButton()
        self.save_button = QPushButton()
        self.cancel_button = QPushButton()

        self.add_button.clicked.connect(self.add_directory)
        self.remove_button.clicked.connect(self.remove_selected)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        action_row = QHBoxLayout()
        action_row.addWidget(self.add_button)
        action_row.addWidget(self.remove_button)
        action_row.addStretch(1)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.save_button)
        bottom_row.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        layout.addLayout(action_row)
        layout.addWidget(self.note_label)
        layout.addLayout(bottom_row)
        self.resize(520, 360)
        self.retranslate()

    def directories(self) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for index in range(self.list_widget.count()):
            text = self.list_widget.item(index).text().strip()
            if not text or is_unc_path(text):
                continue
            key = str(Path(text).expanduser()).casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(text)
        return values

    def add_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("file.select_directory"))
        if not path:
            return
        if is_unc_path(path):
            self._show_unc_warning(path)
            return
        self.list_widget.addItem(path)

    def _show_unc_warning(self, path: str) -> None:
        QMessageBox.warning(
            self,
            self.i18n.t("dialog.warning"),
            self.i18n.t("video_dirs.unc_blocked_message").format(path=path),
        )

    def remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

    def retranslate(self) -> None:
        self.setWindowTitle(self.i18n.t("video_dirs.dialog_title"))
        self.note_label.setText(self.i18n.t("video_dirs.mount_timeout_note"))
        self.add_button.setText(self.i18n.t("button.add"))
        self.remove_button.setText(self.i18n.t("button.remove"))
        self.save_button.setText(self.i18n.t("button.save"))
        self.cancel_button.setText(self.i18n.t("button.cancel"))
