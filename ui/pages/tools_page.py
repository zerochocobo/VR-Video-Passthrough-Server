"""Offline tools launcher page: one card per offline workflow."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.icons import line_pixmap

TOOL_ICON_SIZE = 30


class ToolCard(QFrame):
    open_requested = Signal()

    def __init__(self, icon_name: str) -> None:
        super().__init__()
        self.setObjectName("ToolCard")
        self.setStyleSheet(
            "QFrame#ToolCard {"
            f"background: {theme.CARD_BG}; border: 1px solid {theme.CARD_BORDER}; border-radius: 12px;"
            "}"
        )
        self.icon_label = QLabel()
        self.icon_label.setPixmap(line_pixmap(icon_name, theme.BLUE_DARK, TOOL_ICON_SIZE))
        self.icon_label.setFixedSize(TOOL_ICON_SIZE + 4, TOOL_ICON_SIZE + 4)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("background: transparent;")
        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            f"font-size: 11pt; font-weight: 600; color: {theme.TEXT_PRIMARY}; background: transparent;"
        )
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(
            f"font-size: 9pt; color: {theme.TEXT_MUTED}; background: transparent;"
        )
        self.open_button = QPushButton()
        self.open_button.setFixedWidth(96)
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.open_requested)

        text_box = QVBoxLayout()
        text_box.setSpacing(3)
        text_box.addWidget(self.title_label)
        text_box.addWidget(self.desc_label)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)
        layout.addWidget(self.icon_label)
        layout.addLayout(text_box, 1)
        layout.addWidget(self.open_button)


class ToolsPage(QWidget):
    open_offline = Signal()
    open_two_dvr = Signal()
    open_rm = Signal()
    open_superres = Signal()

    def __init__(self, i18n, settings=None) -> None:
        super().__init__()
        self.i18n = i18n
        self.settings = settings

        self.title = QLabel()
        self.title.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {theme.TEXT_PRIMARY}; background: transparent;"
        )
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(
            f"font-size: 9pt; color: {theme.TEXT_FAINT}; background: transparent;"
        )

        self.offline_card = ToolCard("green_screen")
        self.two_dvr_card = ToolCard("two_dvr")
        self.rm_card = ToolCard("rm")
        self.superres_card = ToolCard("rm")
        self.offline_card.open_requested.connect(self.open_offline)
        self.two_dvr_card.open_requested.connect(self.open_two_dvr)
        self.rm_card.open_requested.connect(self.open_rm)
        self.superres_card.open_requested.connect(self.open_superres)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(self.title)
        layout.addWidget(self.hint)
        layout.addSpacing(4)
        layout.addWidget(self.offline_card)
        layout.addWidget(self.two_dvr_card)
        layout.addWidget(self.rm_card)
        layout.addWidget(self.superres_card)
        layout.addStretch(1)

        self.set_rm_card_visible(bool(settings and settings.data.get("rm_card_visible")))
        self.retranslate()

    def set_rm_card_visible(self, visible: bool) -> None:
        """Show or hide the offline Remove Mosaic entry from the tools page."""
        self.rm_card.setVisible(bool(visible))

    def retranslate(self) -> None:
        self.title.setText(self.i18n.t("nav.tools"))
        self.hint.setText(self.i18n.t("tools.hint"))
        self.offline_card.title_label.setText(self.i18n.t("button.offline"))
        self.offline_card.desc_label.setText(self.i18n.t("tools.offline_desc"))
        self.two_dvr_card.title_label.setText(self.i18n.t("button.two_dvr"))
        self.two_dvr_card.desc_label.setText(self.i18n.t("tools.two_dvr_desc"))
        self.rm_card.title_label.setText(self.i18n.t("tools.rm_title"))
        self.rm_card.desc_label.setText(self.i18n.t("tools.rm_desc"))
        self.superres_card.title_label.setText(self.i18n.t("superres.offline_title"))
        self.superres_card.desc_label.setText(self.i18n.t("tools.superres_desc"))
        for card in (self.offline_card, self.two_dvr_card, self.rm_card, self.superres_card):
            card.open_button.setText(self.i18n.t("tools.open"))
