from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


BACK_ICON_SIZE = 18


def back_icon() -> QIcon:
    pixmap = QPixmap(BACK_ICON_SIZE, BACK_ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor("#2F6FDD"), 2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline([QPoint(10, 4), QPoint(5, 9), QPoint(10, 14)])
    painter.drawLine(6, 9, 14, 9)
    painter.end()
    return QIcon(pixmap)
