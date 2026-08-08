"""Dashboard feature card: icon + title + switch, summary line + config gear."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ui import theme
from ui.icons import line_icon, line_pixmap
from ui.resources import SWITCH_OFF_IMAGE_PATH, SWITCH_ON_IMAGE_PATH

# Content floor is ~72px: 18 margins + 22 top row (icon/switch) + 6 spacing +
# 26 bottom row (the 26px tool buttons). 76 keeps a little slack while letting
# the dashboard's 2D row wrap without growing the window much.
CARD_HEIGHT = 76
CARD_ICON_SIZE = 20
CARD_TOOL_ICON = 17
CARD_LOCK_ICON = 14


def _tool_button(icon: QIcon) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(CARD_TOOL_ICON, CARD_TOOL_ICON))
    button.setFixedSize(26, 26)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(
        "QPushButton { border: none; border-radius: 6px; background: transparent; padding: 0; }"
        "QPushButton:hover { background: #eef2f6; }"
    )
    return button


class FeatureCard(QFrame):
    toggled = Signal(bool)
    configure_requested = Signal()
    help_requested = Signal()

    def __init__(self, icon_name: str, configurable: bool = True, with_help: bool = False) -> None:
        super().__init__()
        self._icon_name = icon_name
        self.setObjectName("FeatureCard")
        self.setFixedHeight(CARD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(CARD_ICON_SIZE + 2, CARD_ICON_SIZE + 2)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {theme.TEXT_PRIMARY}; background: transparent;"
        )
        self.switch = QCheckBox()
        self.switch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.switch.setStyleSheet(
            "QCheckBox { spacing: 0; background: transparent; }"
            "QCheckBox::indicator { width: 38px; height: 20px; }"
            f"QCheckBox::indicator:unchecked {{ image: url({SWITCH_OFF_IMAGE_PATH.as_posix()}); }}"
            f"QCheckBox::indicator:checked {{ image: url({SWITCH_ON_IMAGE_PATH.as_posix()}); }}"
        )
        self._switch_opacity = QGraphicsOpacityEffect(self.switch)
        self._switch_opacity.setOpacity(1.0)
        self.switch.setGraphicsEffect(self._switch_opacity)
        self.lock_label = QLabel()
        self.lock_label.setFixedSize(CARD_LOCK_ICON + 2, CARD_LOCK_ICON + 2)
        self.lock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lock_label.setPixmap(line_pixmap("lock", theme.TEXT_FAINT, CARD_LOCK_ICON))
        self.lock_label.setVisible(False)
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet(
            f"font-size: 8.5pt; color: {theme.TEXT_MUTED}; background: transparent;"
        )
        self.summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.help_button = _tool_button(line_icon("question", theme.TEXT_FAINT, CARD_TOOL_ICON))
        self.help_button.setVisible(with_help)
        self.config_button = _tool_button(line_icon("settings", theme.TEXT_FAINT, CARD_TOOL_ICON))
        self.config_button.setVisible(configurable)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(7)
        top_row.addWidget(self.icon_label)
        top_row.addWidget(self.title_label, 1)
        top_row.addWidget(self.lock_label)
        top_row.addWidget(self.switch)
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(4)
        bottom_row.addWidget(self.summary_label, 1)
        bottom_row.addWidget(self.help_button)
        bottom_row.addWidget(self.config_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(top_row)
        layout.addStretch(1)
        layout.addLayout(bottom_row)

        self.switch.toggled.connect(self._on_toggled)
        self.config_button.clicked.connect(self.configure_requested)
        self.help_button.clicked.connect(self.help_requested)
        self._apply_state()

    def set_title(self, text: str) -> None:
        self.title_label.setText(text)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)
        self.summary_label.setToolTip(text)

    def set_checked(self, checked: bool) -> None:
        self.switch.blockSignals(True)
        self.switch.setChecked(checked)
        self.switch.blockSignals(False)
        self._apply_state()

    def is_checked(self) -> bool:
        return self.switch.isChecked()

    def set_toggle_enabled(self, enabled: bool, disabled_tooltip: str = "") -> None:
        """Enable/disable only the card's on/off switch."""
        enabled = bool(enabled)
        self.switch.setEnabled(enabled)
        self._switch_opacity.setOpacity(1.0 if enabled else 0.38)
        self.lock_label.setVisible(not enabled)
        cursor = Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ForbiddenCursor
        self.switch.setCursor(cursor)
        tooltip = "" if enabled else str(disabled_tooltip or "")
        self.switch.setToolTip(tooltip)
        self.lock_label.setToolTip(tooltip)

    def _on_toggled(self, checked: bool) -> None:
        self._apply_state()
        self.toggled.emit(checked)

    def _apply_state(self) -> None:
        checked = self.switch.isChecked()
        icon_color = theme.BLUE_DARK if checked else theme.TEXT_FAINT
        self.icon_label.setPixmap(line_pixmap(self._icon_name, icon_color, CARD_ICON_SIZE))
        border = theme.CARD_BORDER_ACTIVE if checked else theme.CARD_BORDER
        self.setStyleSheet(
            "QFrame#FeatureCard {"
            f"background: {theme.CARD_BG}; border: 1px solid {border}; border-radius: 10px;"
            "}"
        )
