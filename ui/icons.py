"""Shared QPainter line icons for the nav rail and feature cards."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _painter(pixmap: QPixmap, color: str, width: float) -> QPainter:
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    return painter


def _draw_home(p: QPainter, s: float) -> None:
    p.drawPolyline([QPointF(s * 0.16, s * 0.5), QPointF(s * 0.5, s * 0.2), QPointF(s * 0.84, s * 0.5)])
    p.drawPolyline([
        QPointF(s * 0.26, s * 0.48), QPointF(s * 0.26, s * 0.8),
        QPointF(s * 0.74, s * 0.8), QPointF(s * 0.74, s * 0.48),
    ])
    p.drawLine(QPointF(s * 0.44, s * 0.8), QPointF(s * 0.44, s * 0.6))
    p.drawLine(QPointF(s * 0.44, s * 0.6), QPointF(s * 0.56, s * 0.6))
    p.drawLine(QPointF(s * 0.56, s * 0.6), QPointF(s * 0.56, s * 0.8))


def _draw_tools(p: QPainter, s: float) -> None:
    for frac, knob in ((0.3, 0.62), (0.5, 0.34), (0.7, 0.7)):
        y = s * frac
        p.drawLine(QPointF(s * 0.18, y), QPointF(s * 0.82, y))
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(s * knob, y), s * 0.07, s * 0.07)
        p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_subtitle(p: QPainter, s: float) -> None:
    p.drawRoundedRect(QRectF(s * 0.14, s * 0.2, s * 0.72, s * 0.6), s * 0.08, s * 0.08)
    p.drawLine(QPointF(s * 0.24, s * 0.56), QPointF(s * 0.52, s * 0.56))
    p.drawLine(QPointF(s * 0.58, s * 0.56), QPointF(s * 0.76, s * 0.56))
    p.drawLine(QPointF(s * 0.24, s * 0.68), QPointF(s * 0.4, s * 0.68))
    p.drawLine(QPointF(s * 0.46, s * 0.68), QPointF(s * 0.76, s * 0.68))


def _draw_log(p: QPainter, s: float) -> None:
    p.drawRoundedRect(QRectF(s * 0.22, s * 0.14, s * 0.56, s * 0.72), s * 0.06, s * 0.06)
    for frac in (0.34, 0.5, 0.66):
        p.drawLine(QPointF(s * 0.34, s * frac), QPointF(s * 0.66, s * frac))


def _draw_settings(p: QPainter, s: float) -> None:
    center = s / 2
    p.save()
    p.translate(center, center)
    for _ in range(8):
        p.drawLine(QPointF(0, -s * 0.4), QPointF(0, -s * 0.31))
        p.rotate(45)
    p.restore()
    p.drawEllipse(QPointF(center, center), s * 0.26, s * 0.26)
    p.drawEllipse(QPointF(center, center), s * 0.09, s * 0.09)


def _draw_green_screen(p: QPainter, s: float) -> None:
    p.drawRoundedRect(QRectF(s * 0.12, s * 0.18, s * 0.76, s * 0.64), s * 0.07, s * 0.07)
    p.drawEllipse(QPointF(s * 0.5, s * 0.42), s * 0.1, s * 0.1)
    path = QPainterPath()
    path.moveTo(s * 0.3, s * 0.82)
    path.quadTo(s * 0.5, s * 0.5, s * 0.7, s * 0.82)
    p.drawPath(path)


def _draw_alpha(p: QPainter, s: float) -> None:
    path = QPainterPath()
    path.moveTo(s * 0.5, s * 0.14)
    path.cubicTo(s * 0.78, s * 0.48, s * 0.76, s * 0.66, s * 0.5, s * 0.84)
    path.cubicTo(s * 0.24, s * 0.66, s * 0.22, s * 0.48, s * 0.5, s * 0.14)
    p.drawPath(path)
    p.drawLine(QPointF(s * 0.4, s * 0.6), QPointF(s * 0.6, s * 0.6))


def _draw_two_dvr(p: QPainter, s: float) -> None:
    path = QPainterPath()
    path.addRoundedRect(QRectF(s * 0.1, s * 0.28, s * 0.8, s * 0.44), s * 0.1, s * 0.1)
    p.drawPath(path)
    notch = QPainterPath()
    notch.moveTo(s * 0.38, s * 0.72)
    notch.quadTo(s * 0.5, s * 0.56, s * 0.62, s * 0.72)
    p.drawPath(notch)
    p.drawEllipse(QPointF(s * 0.32, s * 0.48), s * 0.06, s * 0.06)
    p.drawEllipse(QPointF(s * 0.68, s * 0.48), s * 0.06, s * 0.06)


def _draw_rm(p: QPainter, s: float) -> None:
    p.setBrush(p.pen().color())
    for ix in range(3):
        for iy in range(3):
            if (ix, iy) == (2, 0):
                continue
            p.drawEllipse(QPointF(s * (0.26 + ix * 0.24), s * (0.26 + iy * 0.24)), s * 0.045, s * 0.045)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(s * 0.64, s * 0.36), QPointF(s * 0.84, s * 0.16))
    p.drawLine(QPointF(s * 0.66, s * 0.16), QPointF(s * 0.84, s * 0.16))
    p.drawLine(QPointF(s * 0.84, s * 0.16), QPointF(s * 0.84, s * 0.34))


def _draw_translate(p: QPainter, s: float) -> None:
    font = QFont()
    font.setPointSizeF(max(6.0, s * 0.34))
    font.setBold(True)
    p.setFont(font)
    p.drawRoundedRect(QRectF(s * 0.1, s * 0.12, s * 0.5, s * 0.5), s * 0.07, s * 0.07)
    p.drawText(QRectF(s * 0.1, s * 0.12, s * 0.5, s * 0.5), Qt.AlignmentFlag.AlignCenter, "A")
    p.drawRoundedRect(QRectF(s * 0.42, s * 0.42, s * 0.5, s * 0.5), s * 0.07, s * 0.07)
    p.drawText(QRectF(s * 0.42, s * 0.42, s * 0.5, s * 0.5), Qt.AlignmentFlag.AlignCenter, "文")


def _draw_light(p: QPainter, s: float) -> None:
    center = s / 2
    p.drawEllipse(QPointF(center, center), s * 0.18, s * 0.18)
    p.save()
    p.translate(center, center)
    for _ in range(8):
        p.drawLine(QPointF(0, -s * 0.42), QPointF(0, -s * 0.3))
        p.rotate(45)
    p.restore()


def _draw_folder(p: QPainter, s: float) -> None:
    path = QPainterPath()
    path.moveTo(s * 0.14, s * 0.3)
    path.lineTo(s * 0.14, s * 0.76)
    path.lineTo(s * 0.86, s * 0.76)
    path.lineTo(s * 0.86, s * 0.36)
    path.lineTo(s * 0.48, s * 0.36)
    path.lineTo(s * 0.4, s * 0.24)
    path.lineTo(s * 0.2, s * 0.24)
    path.closeSubpath()
    p.drawPath(path)


def _draw_convert(p: QPainter, s: float) -> None:
    p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.44, s * 0.36), s * 0.06, s * 0.06)
    p.drawRoundedRect(QRectF(s * 0.44, s * 0.48, s * 0.44, s * 0.36), s * 0.06, s * 0.06)
    p.drawLine(QPointF(s * 0.34, s * 0.6), QPointF(s * 0.34, s * 0.72))
    p.drawLine(QPointF(s * 0.34, s * 0.72), QPointF(s * 0.42, s * 0.72))
    p.drawLine(QPointF(s * 0.66, s * 0.4), QPointF(s * 0.66, s * 0.28))
    p.drawLine(QPointF(s * 0.66, s * 0.28), QPointF(s * 0.58, s * 0.28))


def _draw_question(p: QPainter, s: float) -> None:
    p.drawEllipse(QRectF(s * 0.1, s * 0.1, s * 0.8, s * 0.8))
    font = QFont()
    font.setBold(True)
    font.setPointSizeF(max(6.0, s * 0.42))
    p.setFont(font)
    p.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "?")


def _draw_lock(p: QPainter, s: float) -> None:
    p.drawRoundedRect(QRectF(s * 0.22, s * 0.42, s * 0.56, s * 0.42), s * 0.07, s * 0.07)
    shackle = QPainterPath()
    shackle.moveTo(s * 0.34, s * 0.42)
    shackle.lineTo(s * 0.34, s * 0.32)
    shackle.cubicTo(s * 0.34, s * 0.12, s * 0.66, s * 0.12, s * 0.66, s * 0.32)
    shackle.lineTo(s * 0.66, s * 0.42)
    p.drawPath(shackle)


_DRAWERS = {
    "home": _draw_home,
    "tools": _draw_tools,
    "subtitle": _draw_subtitle,
    "log": _draw_log,
    "settings": _draw_settings,
    "green_screen": _draw_green_screen,
    "alpha": _draw_alpha,
    "two_dvr": _draw_two_dvr,
    "rm": _draw_rm,
    "translate": _draw_translate,
    "light": _draw_light,
    "folder": _draw_folder,
    "convert": _draw_convert,
    "question": _draw_question,
    "lock": _draw_lock,
}


def line_pixmap(name: str, color: str, size: int = 22, pen_width: float | None = None) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = _painter(pixmap, color, pen_width if pen_width is not None else max(1.6, size / 12.0))
    _DRAWERS[name](painter, float(size))
    painter.end()
    return pixmap


def line_icon(name: str, color: str, size: int = 22, pen_width: float | None = None) -> QIcon:
    return QIcon(line_pixmap(name, color, size, pen_width))


def gear_icon(color: str = "#4f5965") -> QIcon:
    return line_icon("settings", color, 22)


def question_icon(color: str = "#4f5965") -> QIcon:
    return line_icon("question", color, 22)
