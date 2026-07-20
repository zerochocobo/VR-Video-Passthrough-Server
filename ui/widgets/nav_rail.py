"""Left navigation rail: logo header plus icon+text entries on brand navy."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ui import theme
from ui.icons import line_pixmap
from ui.resources import LOGO_PATH

NAV_WIDTH = 92
NAV_ICON_SIZE = 24
NAV_ITEM_HEIGHT = 66
LOGO_SIZE = 52


class NavItem(QPushButton):
    def __init__(self, key: str, icon_name: str) -> None:
        super().__init__()
        self.key = key
        self.icon_name = icon_name
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(NAV_ITEM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._icon_label = QLabel(self)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._icon_label.setStyleSheet("background: transparent;")
        self._text_label = QLabel(self)
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(4)
        layout.addWidget(self._icon_label)
        layout.addWidget(self._text_label)
        self._apply_state()
        self.toggled.connect(lambda _checked: self._apply_state())

    def set_text(self, text: str) -> None:
        self._text_label.setText(text)

    def _apply_state(self) -> None:
        color = theme.NAV_TEXT_ACTIVE if self.isChecked() else theme.NAV_TEXT
        self._icon_label.setPixmap(line_pixmap(self.icon_name, color, NAV_ICON_SIZE))
        self._text_label.setStyleSheet(
            f"background: transparent; color: {color}; font-size: 9pt;"
            + ("font-weight: 600;" if self.isChecked() else "")
        )


class NavRail(QWidget):
    page_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("NavRail")
        self.setFixedWidth(NAV_WIDTH)
        # Custom QWidget subclasses ignore stylesheet backgrounds without this.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#NavRail {{ background: {theme.NAVY}; }}"
            "QWidget#NavRail QPushButton {"
            "border: none; border-radius: 10px; background: transparent; padding: 0;"
            "}"
            f"QWidget#NavRail QPushButton:hover {{ background: {theme.NAVY_HOVER}; }}"
            f"QWidget#NavRail QPushButton:checked {{ background: {theme.NAVY_ACTIVE}; }}"
        )
        self._items: dict[str, NavItem] = {}

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("background: transparent;")
        pixmap = QPixmap(str(LOGO_PATH))
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(
                QSize(LOGO_SIZE, LOGO_SIZE),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 14, 8, 10)
        self._layout.setSpacing(4)
        self._layout.addWidget(logo)
        self._layout.addSpacing(8)
        self._stretch_added = False

    def add_item(self, key: str, icon_name: str, bottom: bool = False) -> None:
        if bottom and not self._stretch_added:
            self._layout.addStretch(1)
            self._stretch_added = True
        item = NavItem(key, icon_name)
        item.clicked.connect(lambda _checked=False, k=key: self.select(k, emit=True))
        self._items[key] = item
        self._layout.addWidget(item)

    def finish(self) -> None:
        if not self._stretch_added:
            self._layout.addStretch(1)
            self._stretch_added = True

    def set_item_text(self, key: str, text: str) -> None:
        item = self._items.get(key)
        if item is not None:
            item.set_text(text)

    def select(self, key: str, emit: bool = False) -> None:
        for item_key, item in self._items.items():
            item.blockSignals(True)
            item.setChecked(item_key == key)
            item.blockSignals(False)
            item._apply_state()
        if emit:
            self.page_selected.emit(key)

    def current(self) -> str | None:
        for key, item in self._items.items():
            if item.isChecked():
                return key
        return None
