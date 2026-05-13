from __future__ import annotations

import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageQt
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.page_icons import BACK_ICON_SIZE, back_icon
from ui.settings import DEFAULTS
from ui.subtitle_preview import create_preview_ass, extract_left_eye_frame, generate_preview_image, get_video_info


ROOT = Path(sys.executable).resolve().parent / "_internal" if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
SUBTITLE_TEMPLATE_PATH = ROOT / "resources" / "subtitle_ass_templates.txt"
SUBTITLE_PAGE_WIDTH = 1120
SUBTITLE_PAGE_HEIGHT = 600
SUBTITLE_SETTING_KEYS = (
    "subtitle_mode",
    "subtitle_direction",
    "subtitle_distance_m",
    "subtitle_fov",
    "subtitle_yaw",
    "subtitle_pitch",
    "subtitle_font_scale",
    "subtitle_outline_scale",
    "subtitle_margin_v_scale",
    "subtitle_alpha",
    "subtitle_color",
    "subtitle_outline_color",
    "subtitle_v360",
)


def _save_icon() -> QIcon:
    size = 20
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#2F6FDD"))
    painter.drawRoundedRect(2, 2, 16, 16, 2, 2)
    painter.setBrush(QColor("#FFFFFF"))
    painter.drawRect(5, 4, 9, 5)
    painter.setBrush(QColor("#DCEBFF"))
    painter.drawRoundedRect(5, 12, 10, 4, 1, 1)
    painter.setBrush(QColor("#1F4E9D"))
    painter.drawRect(13, 4, 2, 4)
    painter.end()
    return QIcon(pixmap)


def _help_icon() -> QIcon:
    size = 20
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#409EFF"))
    painter.drawEllipse(2, 2, 16, 16)
    painter.setPen(QColor("#FFFFFF"))
    font = painter.font()
    font.setBold(True)
    font.setPointSize(12)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")
    painter.end()
    return QIcon(pixmap)


class PreviewCanvas(QWidget):
    region_selected = Signal(float, float, float)
    image_double_clicked = Signal()

    def __init__(self, selectable: bool = False) -> None:
        super().__init__()
        self.selectable = selectable
        self.pixmap: QPixmap | None = None
        self.image_rect = QRect()
        self.start: QPoint | None = None
        self.selection = QRect()
        self.setMinimumSize(360, 260)
        self.setStyleSheet("background: #050505; border: 1px solid #30343a;")

    def set_pil_image(self, image: Image.Image) -> None:
        qimage = ImageQt.ImageQt(image.convert("RGB"))
        self.pixmap = QPixmap.fromImage(qimage)
        self.selection = QRect()
        self.update()

    def set_message(self, text: str) -> None:
        self.pixmap = None
        self.selection = QRect()
        self._message = text
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.pixmap is None:
            painter.setPen(Qt.lightGray)
            painter.drawText(self.rect(), Qt.AlignCenter, getattr(self, "_message", ""))
            return
        scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self.image_rect = QRect(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(self.image_rect, scaled)
        if not self.selection.isNull():
            painter.setPen(QPen(Qt.red, 2))
            painter.drawRect(self.selection.normalized())

    def mousePressEvent(self, event) -> None:
        if self.selectable and self.pixmap is not None and self.image_rect.contains(event.position().toPoint()):
            self.start = event.position().toPoint()
            self.selection = QRect(self.start, self.start)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.start is not None:
            self.selection = QRect(self.start, event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.start is None or self.pixmap is None or self.image_rect.isNull():
            return
        end = event.position().toPoint()
        rect = QRect(self.start, end).normalized()
        self.start = None
        x1 = max(0, min(rect.left() - self.image_rect.left(), self.image_rect.width()))
        x2 = max(0, min(rect.right() - self.image_rect.left(), self.image_rect.width()))
        y1 = max(0, min(rect.top() - self.image_rect.top(), self.image_rect.height()))
        y2 = max(0, min(rect.bottom() - self.image_rect.top(), self.image_rect.height()))
        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            return
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        yaw = (center_x / self.image_rect.width()) * 180.0 - 90.0
        pitch = 90.0 - (center_y / self.image_rect.height()) * 180.0
        fov = max(10.0, min(130.0, (abs(x2 - x1) / self.image_rect.width()) * 180.0))
        self.region_selected.emit(yaw, pitch, fov)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.pixmap is not None and self.image_rect.contains(event.position().toPoint()):
            self.image_double_clicked.emit()


class ImageZoomCanvas(QWidget):
    def __init__(self, pixmap: QPixmap) -> None:
        super().__init__()
        self.source = pixmap
        self.scale = 0.5
        self.offset = QPoint(0, 0)
        self.drag_start: QPoint | None = None
        self.offset_start = QPoint(0, 0)
        self.setMinimumSize(720, 460)
        self.setStyleSheet("background: #000000;")

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.source.isNull():
            return
        width = max(1, int(self.source.width() * self.scale))
        height = max(1, int(self.source.height() * self.scale))
        scaled = self.source.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self.offset.isNull() and (scaled.width() < self.width() or scaled.height() < self.height()):
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.drawPixmap(self.offset, scaled)

    def wheelEvent(self, event) -> None:
        old_scale = self.scale
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale = max(0.1, min(8.0, self.scale * factor))
        pos = event.position().toPoint()
        if old_scale > 0:
            ratio = self.scale / old_scale
            delta = pos - self.offset
            self.offset = QPoint(pos.x() - int(delta.x() * ratio), pos.y() - int(delta.y() * ratio))
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.drag_start = event.position().toPoint()
            self.offset_start = QPoint(self.offset)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self.drag_start is not None:
            self.offset = self.offset_start + event.position().toPoint() - self.drag_start
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.drag_start = None
            self.unsetCursor()


class ImageZoomDialog(QDialog):
    def __init__(self, parent: QWidget, pixmap: QPixmap, hint: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(hint)
        self.resize(1100, 760)
        layout = QVBoxLayout(self)
        layout.addWidget(ImageZoomCanvas(pixmap), 1)
        hint_label = QLabel(hint)
        hint_label.setStyleSheet("color: #626975; padding: 4px 6px;")
        layout.addWidget(hint_label)


class SubtitlePage(QWidget):
    frame_loaded = Signal(object)
    preview_loaded = Signal(object)
    worker_log = Signal(str)
    worker_error = Signal(str)
    frame_done = Signal()
    preview_done = Signal()

    def __init__(self, i18n, settings) -> None:
        super().__init__()
        self.setObjectName("SubtitlePage")
        self.setStyleSheet(
            "QWidget#SubtitlePage, QWidget#SubtitlePage QLabel, QWidget#SubtitlePage QCheckBox, "
            "QWidget#SubtitlePage QRadioButton { font-size: 9pt; }"
            "QWidget#SubtitlePage QPushButton, QWidget#SubtitlePage QLineEdit, QWidget#SubtitlePage QComboBox, "
            "QWidget#SubtitlePage QTextEdit { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#SubtitlePage QLabel#SubtitlePageTitle { font-size: 14pt; font-weight: 700; }"
            "QWidget#SubtitlePage QLabel#SaveStatus { color: #18A058; font-size: 9pt; }"
        )
        self.i18n = i18n
        self.settings = settings
        self.draft = {key: settings.data.get(key, DEFAULTS.get(key)) for key in SUBTITLE_SETTING_KEYS}
        self.video_info: dict | None = None
        self.frame_ready = False

        self.title_label = QLabel()
        self.subtitle_note_label = QLabel()
        self.back_button = QPushButton()
        self.back_button.setIcon(back_icon())
        self.back_button.setIconSize(QPixmap(BACK_ICON_SIZE, BACK_ICON_SIZE).size())
        self.restore_button = QPushButton()
        self.video_path = QLineEdit()
        self.browse_video_button = QPushButton("...")
        self.load_frame_button = QPushButton()
        self.preview_button = QPushButton()
        self.preview_button.setEnabled(False)
        self.save_button = QPushButton()
        self.save_button.setIcon(_save_icon())
        self.save_button.setIconSize(QPixmap(20, 20).size())
        self.save_status_label = QLabel()
        self.save_status_label.setObjectName("SaveStatus")
        self.mode_help_button = QPushButton()
        self.mode_help_button.setIcon(_help_icon())
        self.mode_help_button.setIconSize(QPixmap(20, 20).size())
        self.mode_help_button.setFixedWidth(32)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_label = QLabel("00:00:00")
        self.region_label = QLabel()
        self.mode_group = QButtonGroup(self)
        self.mode_dual = QRadioButton()
        self.mode_left = QRadioButton()
        self.mode_right = QRadioButton()
        self.distance = QComboBox()
        self.alpha = QComboBox()
        self.direction = QComboBox()
        self.subtitle_color = QComboBox()
        self.original_canvas = PreviewCanvas(selectable=True)
        self.preview_canvas = PreviewCanvas(selectable=False)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(UI_LOG_MAX_BLOCKS)
        self.log.setMaximumHeight(90)

        for value, button in (("dual", self.mode_dual), ("left", self.mode_left), ("right", self.mode_right)):
            self.mode_group.addButton(button)
            button.setProperty("value", value)
        self.distance.addItems([f"{i}m" for i in range(1, 11)])
        self.alpha.addItems([f"{i}%" for i in range(0, 71, 10)])
        self.direction.addItems([
            "horizontal_middle",
            "horizontal_top",
            "horizontal_bottom",
            "vertical_left",
            "vertical_middle",
            "vertical_right",
        ])
        self.time_slider.setRange(0, 0)

        self._build_layout()
        self.load_values()
        self.retranslate()
        self._bind()

    def _build_layout(self) -> None:
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        self.title_label.setObjectName("SubtitlePageTitle")
        self.subtitle_note_label.setStyleSheet("color: #626975;")
        title_col.addWidget(self.title_label)
        title_col.addWidget(self.subtitle_note_label)
        header.addLayout(title_col)
        header.addStretch(1)
        header.addWidget(self.back_button)

        top = QGridLayout()
        top.addWidget(QLabel(""), 0, 0)
        top.addWidget(self.video_path, 0, 1)
        top.addWidget(self.browse_video_button, 0, 2)
        top.addWidget(QLabel(""), 1, 0)
        top.addWidget(self.time_slider, 1, 1)
        top.addWidget(self.time_label, 1, 2)
        self.form_labels = [top.itemAtPosition(row, 0).widget() for row in range(2)]

        region_row = QHBoxLayout()
        region_row.addWidget(self.region_label)
        region_row.addStretch(1)

        action_row = QHBoxLayout()
        action_row.addWidget(self.load_frame_button)
        action_row.addWidget(self.preview_button)
        action_row.addWidget(self.save_button)
        action_row.addWidget(self.save_status_label)
        action_row.addStretch(1)
        action_row.addWidget(self.restore_button)

        option_row = QHBoxLayout()
        option_row.addWidget(QLabel())
        self.mode_label = option_row.itemAt(0).widget()
        option_row.addWidget(self.mode_dual)
        option_row.addWidget(self.mode_left)
        option_row.addWidget(self.mode_right)
        self.distance_label = QLabel()
        option_row.addWidget(self.distance_label)
        option_row.addWidget(self.distance)
        option_row.addWidget(self.mode_help_button)
        self.alpha_label = QLabel()
        option_row.addWidget(self.alpha_label)
        option_row.addWidget(self.alpha)
        self.direction_label = QLabel()
        option_row.addWidget(self.direction_label)
        option_row.addWidget(self.direction)
        self.subtitle_color_label = QLabel()
        option_row.addWidget(self.subtitle_color_label)
        option_row.addWidget(self.subtitle_color)
        option_row.addStretch(1)

        canvases = QHBoxLayout()
        original_box = QGroupBox()
        preview_box = QGroupBox()
        self.original_box = original_box
        self.preview_box = preview_box
        original_layout = QVBoxLayout(original_box)
        preview_layout = QVBoxLayout(preview_box)
        original_layout.addWidget(self.original_canvas)
        preview_layout.addWidget(self.preview_canvas)
        canvases.addWidget(original_box, 1)
        canvases.addWidget(preview_box, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(top)
        layout.addLayout(region_row)
        layout.addLayout(option_row)
        layout.addLayout(action_row)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)
        layout.addLayout(canvases, 1)
        layout.addWidget(self.log)

    def _bind(self) -> None:
        self.browse_video_button.clicked.connect(self.browse_video)
        self.video_path.textChanged.connect(lambda _text: self.invalidate_frame())
        self.load_frame_button.clicked.connect(self.load_frame)
        self.preview_button.clicked.connect(self.generate_preview)
        self.save_button.clicked.connect(self.save_settings)
        self.mode_help_button.clicked.connect(self.show_mode_help)
        self.restore_button.clicked.connect(self.restore_defaults)
        self.time_slider.valueChanged.connect(self.update_time_label)
        self.time_slider.valueChanged.connect(self.invalidate_frame)
        self.original_canvas.region_selected.connect(self.apply_region)
        self.frame_loaded.connect(self.on_frame_loaded)
        self.preview_loaded.connect(self.preview_canvas.set_pil_image)
        self.worker_log.connect(self.append_log)
        self.worker_error.connect(lambda text: self.append_log(f"Error: {text}"))
        self.frame_done.connect(lambda: self.load_frame_button.setEnabled(True))
        self.preview_done.connect(lambda: self.preview_button.setEnabled(self.frame_ready))
        self.preview_canvas.image_double_clicked.connect(self.open_preview_viewer)
        for button in (self.mode_dual, self.mode_left, self.mode_right):
            button.toggled.connect(self.save_values)
            button.toggled.connect(self.update_distance_visibility)
        for combo in (self.distance, self.alpha, self.direction, self.subtitle_color):
            combo.currentIndexChanged.connect(self.save_values)

    def browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.i18n.t("file.select_video"), "", "Videos (*.mp4 *.mkv *.webm *.mov *.m4v)")
        if not path:
            return
        self.video_path.setText(path)
        self.invalidate_frame()
        self.load_video_info()

    def load_video_info(self) -> None:
        try:
            info = get_video_info(self.video_path.text())
            self.video_info = info
            duration = int(float(info.get("duration", 0))) if info else 0
            self.time_slider.setRange(0, max(0, duration))
            self.append_log(self.i18n.t("subtitle.loaded_video").format(info.get("width"), info.get("height"), info.get("duration", 0)))
        except Exception as e:
            self.append_log(f"Error: {e}")

    def load_frame(self) -> None:
        if not self.video_path.text().strip():
            return
        self.load_frame_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.frame_ready = False
        self.original_canvas.set_message(self.i18n.t("subtitle.loading_frame"))

        def worker():
            try:
                image = extract_left_eye_frame(self.video_path.text(), self.time_slider.value())
                self.frame_loaded.emit(image)
            except Exception as e:
                self.worker_error.emit(str(e))
            finally:
                self.frame_done.emit()

        threading.Thread(target=worker, daemon=True).start()

    def generate_preview(self) -> None:
        if not self.frame_ready:
            self.append_log(self.i18n.t("subtitle.frame_required"))
            return
        if not self.video_path.text().strip():
            self.append_log(self.i18n.t("subtitle.select_video"))
            return
        self.preview_button.setEnabled(False)
        self.preview_canvas.set_message(self.i18n.t("subtitle.generating_preview"))

        def worker():
            ass_file = ""
            try:
                out = str(Path(tempfile.gettempdir()) / "ptserver_subtitle_preview.jpg")
                ass_file = self._create_preview_ass()
                generate_preview_image(
                    self.video_path.text(),
                    ass_file,
                    self.time_slider.value(),
                    out,
                    fov=float(self.draft.get("subtitle_fov", 60.0)),
                    yaw=float(self.draft.get("subtitle_yaw", 0.0)),
                    pitch=float(self.draft.get("subtitle_pitch", 0.0)),
                    transparency_percent=float(self._alpha_percent()),
                    mode=self._mode_value(),
                    distance_m=float(self._distance_m()),
                    subtitle_direction=str(self.direction.currentData() or self.direction.currentText() or "horizontal_bottom"),
                )
                image = Image.open(out).copy()
                self.preview_loaded.emit(image)
                self.worker_log.emit(self.i18n.t("subtitle.preview_done"))
            except Exception as e:
                self.worker_error.emit(str(e))
            finally:
                if ass_file:
                    try:
                        Path(ass_file).unlink()
                    except OSError:
                        pass
                self.preview_done.emit()

        threading.Thread(target=worker, daemon=True).start()

    def invalidate_frame(self) -> None:
        self.frame_ready = False
        self.preview_button.setEnabled(False)

    def on_frame_loaded(self, image: Image.Image) -> None:
        self.original_canvas.set_pil_image(image)
        self.frame_ready = True
        self.preview_button.setEnabled(True)

    def _create_preview_ass(self) -> str:
        info = self.video_info or get_video_info(self.video_path.text())
        if not info:
            raise RuntimeError("Unable to read video info")
        return create_preview_ass(
            info,
            SUBTITLE_TEMPLATE_PATH,
            self.i18n.t("subtitle.preview_test_text"),
            subtitle_color=str(self.draft.get("subtitle_color") or ""),
            background_color=str(self.settings.data.get("background_color") or "808080"),
        )

    def open_preview_viewer(self) -> None:
        if self.preview_canvas.pixmap is None:
            self.append_log(self.i18n.t("subtitle.preview_required"))
            return
        dialog = ImageZoomDialog(self, self.preview_canvas.pixmap, self.i18n.t("subtitle.zoom_hint"))
        dialog.exec()

    def apply_region(self, yaw: float, pitch: float, fov: float) -> None:
        self.draft["subtitle_yaw"] = round(yaw, 2)
        self.draft["subtitle_pitch"] = round(pitch, 2)
        self.draft["subtitle_fov"] = round(fov, 2)
        self.update_region_label()
        self.append_log(self.i18n.t("subtitle.selected_region").format(yaw, pitch, fov))

    def _mode_value(self) -> str:
        button = self.mode_group.checkedButton()
        return str(button.property("value") if button is not None else "dual")

    def _distance_m(self) -> int:
        return max(1, self.distance.currentIndex() + 1)

    def _alpha_percent(self) -> int:
        return max(0, self.alpha.currentIndex() * 10)

    def save_values(self) -> None:
        mode = self._mode_value()
        direction = self.direction.currentData() or self.direction.currentText() or "horizontal_bottom"
        self.draft.update(
            {
                "subtitle_mode": mode,
                "subtitle_direction": direction,
                "subtitle_color": str(self.subtitle_color.currentData() or ""),
                "subtitle_distance_m": float(self._distance_m()),
                "subtitle_alpha": max(0.0, min(1.0, 1.0 - self._alpha_percent() / 100.0)),
                "subtitle_v360": True,
            }
        )

    def save_settings(self) -> None:
        self.save_values()
        for key in SUBTITLE_SETTING_KEYS:
            self.settings.data[key] = self.draft.get(key, DEFAULTS.get(key))
        self.settings.save()
        self.save_status_label.setText(f"{self.i18n.t('subtitle.save_done')} {datetime.now().strftime('%H:%M:%S')}")

    def show_mode_help(self) -> None:
        QMessageBox.information(self, self.i18n.t("subtitle.mode_help_title"), self.i18n.t("subtitle.mode_help_msg"))

    def load_values(self) -> None:
        controls = [
            self.mode_dual,
            self.mode_left,
            self.mode_right,
            self.distance,
            self.alpha,
            self.direction,
            self.subtitle_color,
        ]
        for control in controls:
            control.blockSignals(True)
        try:
            data = self.draft
            mode = str(data.get("subtitle_mode", "dual"))
            {"auto": self.mode_dual, "dual": self.mode_dual, "left": self.mode_left, "right": self.mode_right}.get(mode, self.mode_dual).setChecked(True)
            distance = int(float(data.get("subtitle_distance_m", 4.0)))
            self.distance.setCurrentIndex(max(0, min(9, distance - 1)))
            alpha_percent = int(round((1.0 - float(data.get("subtitle_alpha", 1.0))) * 100))
            self.alpha.setCurrentIndex(max(0, min(7, alpha_percent // 10)))
            direction = str(data.get("subtitle_direction", "horizontal_bottom"))
            index = self.direction.findData(direction)
            if index >= 0:
                self.direction.setCurrentIndex(index)
            color = str(data.get("subtitle_color", "") or "")
            color_index = self.subtitle_color.findData(color)
            if color_index >= 0:
                self.subtitle_color.setCurrentIndex(color_index)
        finally:
            for control in controls:
                control.blockSignals(False)
        self.update_region_label()
        self.update_distance_visibility()

    def restore_defaults(self) -> None:
        for key in SUBTITLE_SETTING_KEYS:
            self.draft[key] = DEFAULTS[key]
        self.draft["subtitle_mode"] = "dual"
        self.load_values()

    def update_distance_visibility(self) -> None:
        visible = self._mode_value() == "dual"
        self.distance_label.setVisible(visible)
        self.distance.setVisible(visible)

    def update_time_label(self, seconds: int) -> None:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        self.time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def update_region_label(self) -> None:
        self.region_label.setText(
            f"{self.i18n.t('subtitle.region')}: "
            f"yaw={float(self.draft.get('subtitle_yaw', 0.0)):.2f}, "
            f"pitch={float(self.draft.get('subtitle_pitch', 0.0)):.2f}, "
            f"fov={float(self.draft.get('subtitle_fov', 60.0)):.2f}"
        )

    def append_log(self, text: str) -> None:
        text = clean_log_text(str(text))
        if text:
            self.log.append(text)

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("subtitle.page_title"))
        self.subtitle_note_label.setText(self.i18n.t("subtitle.auto_note"))
        self.back_button.setText(self.i18n.t("button.back"))
        self.restore_button.setText(self.i18n.t("subtitle.restore_default"))
        self.load_frame_button.setText(self.i18n.t("subtitle.load_frame"))
        self.preview_button.setText(self.i18n.t("subtitle.generate_preview"))
        self.save_button.setText(self.i18n.t("subtitle.save_settings"))
        self.form_labels[0].setText(self.i18n.t("offline.video"))
        self.form_labels[1].setText(self.i18n.t("subtitle.time"))
        self.mode_label.setText(self.i18n.t("subtitle.mode"))
        self.mode_dual.setText(self.i18n.t("subtitle.mode_dual"))
        self.mode_left.setText(self.i18n.t("subtitle.mode_left"))
        self.mode_right.setText(self.i18n.t("subtitle.mode_right"))
        self.mode_help_button.setToolTip(self.i18n.t("subtitle.mode_help"))
        self.distance_label.setText(self.i18n.t("subtitle.distance"))
        self.alpha_label.setText(self.i18n.t("subtitle.alpha"))
        self.direction_label.setText(self.i18n.t("subtitle.direction"))
        self.subtitle_color_label.setText(self.i18n.t("subtitle.srt_color"))
        self.original_box.setTitle(self.i18n.t("subtitle.original"))
        self.preview_box.setTitle(self.i18n.t("subtitle.preview"))
        self.direction.blockSignals(True)
        self.subtitle_color.blockSignals(True)
        try:
            self.direction.clear()
            for key, value in (
                ("subtitle.horizontal_middle", "horizontal_middle"),
                ("subtitle.horizontal_top", "horizontal_top"),
                ("subtitle.horizontal_bottom", "horizontal_bottom"),
                ("subtitle.vertical_left", "vertical_left"),
                ("subtitle.vertical_middle", "vertical_middle"),
                ("subtitle.vertical_right", "vertical_right"),
            ):
                self.direction.addItem(self.i18n.t(key), value)
            self.subtitle_color.clear()
            for key, value in (
                ("subtitle.color_inverse_bg", ""),
                ("subtitle.color_white_black", "FFFFFF"),
                ("subtitle.color_black_white", "000000"),
                ("subtitle.color_green_black", "5AFF65"),
                ("subtitle.color_yellow_black", "FFFF00"),
                ("subtitle.color_red_black", "FF0000"),
            ):
                self.subtitle_color.addItem(self.i18n.t(key), value)
        finally:
            self.direction.blockSignals(False)
            self.subtitle_color.blockSignals(False)
        self.load_values()
