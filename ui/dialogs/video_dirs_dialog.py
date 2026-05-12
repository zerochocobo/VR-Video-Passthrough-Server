from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)


class VideoDirsDialog(QDialog):
    def __init__(self, i18n, directories: list[str], parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.list_widget = QListWidget()
        for directory in directories:
            if directory.strip():
                self.list_widget.addItem(directory.strip())

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
        layout.addLayout(bottom_row)
        self.resize(520, 320)
        self.retranslate()

    def directories(self) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for index in range(self.list_widget.count()):
            text = self.list_widget.item(index).text().strip()
            if not text:
                continue
            key = str(Path(text).expanduser()).casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(text)
        return values

    def add_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("file.select_directory"))
        if path:
            self.list_widget.addItem(path)

    def remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

    def retranslate(self) -> None:
        self.setWindowTitle(self.i18n.t("video_dirs.dialog_title"))
        self.add_button.setText(self.i18n.t("button.add"))
        self.remove_button.setText(self.i18n.t("button.remove"))
        self.save_button.setText(self.i18n.t("button.save"))
        self.cancel_button.setText(self.i18n.t("button.cancel"))
