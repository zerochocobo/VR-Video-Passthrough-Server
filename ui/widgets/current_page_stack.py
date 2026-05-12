from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QStackedWidget


class CurrentPageStackedWidget(QStackedWidget):
    def sizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is not None:
            return current.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is not None:
            return current.minimumSizeHint()
        return super().minimumSizeHint()
