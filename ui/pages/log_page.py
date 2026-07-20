"""Log page: server output view plus debug toggle and problem help."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.resources import SWITCH_OFF_IMAGE_PATH, SWITCH_ON_IMAGE_PATH


class LogPage(QWidget):
    def __init__(self, i18n) -> None:
        super().__init__()
        self.i18n = i18n

        self.title = QLabel()
        self.title.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {theme.TEXT_PRIMARY}; background: transparent;"
        )

        self.debug_toggle = QCheckBox()
        self.debug_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.debug_toggle.setStyleSheet(
            "QCheckBox { spacing: 8px; background: transparent; }"
            "QCheckBox::indicator { width: 38px; height: 20px; }"
            f"QCheckBox::indicator:unchecked {{ image: url({SWITCH_OFF_IMAGE_PATH.as_posix()}); }}"
            f"QCheckBox::indicator:checked {{ image: url({SWITCH_ON_IMAGE_PATH.as_posix()}); }}"
        )
        self.debug_label = QLabel()
        self.debug_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")
        self.clear_button = QPushButton()
        self.problem_help_button = QPushButton()
        self.problem_help_button.clicked.connect(self._show_problem_help)
        self.clear_button.clicked.connect(lambda: self.log.clear())

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(self.debug_label)
        toolbar.addWidget(self.debug_toggle)
        toolbar.addStretch(1)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.problem_help_button)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(UI_LOG_MAX_BLOCKS)
        log_font = QFont()
        log_font.setPointSize(8)
        self.log.setFont(log_font)
        self.log.setStyleSheet("QTextEdit { font-size: 8pt; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(self.title)
        layout.addLayout(toolbar)
        layout.addWidget(self.log, 1)

        self.retranslate()

    def append_log(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def clear_log(self) -> None:
        self.log.clear()

    def _show_problem_help(self) -> None:
        QMessageBox.information(
            self,
            self.i18n.t("problem_help.title"),
            self.i18n.t("problem_help.message"),
        )

    def retranslate(self) -> None:
        self.title.setText(self.i18n.t("nav.log"))
        self.debug_label.setText(self.i18n.t("log.debug"))
        self.clear_button.setText(self.i18n.t("log.clear"))
        self.problem_help_button.setText(self.i18n.t("problem_help.button"))
