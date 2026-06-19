from __future__ import annotations

import json
import os
import threading
import urllib.request

from PySide6.QtCore import QFileSystemWatcher, QPoint, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QAbstractItemView,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QButtonGroup,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from media_library import build_media_roots, parse_video_dirs
from ui.dialogs.video_dirs_dialog import VideoDirsDialog
from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.player_support import load_player_support
from ui.resources import SWITCH_OFF_IMAGE_PATH, SWITCH_ON_IMAGE_PATH
from ui.settings import DEFAULTS, LIGHT_MATCH_PRESETS, ROOT as UI_ROOT, quality_speed_value
from ui.widgets.trt_cache_dialog import TensorRTConfigDialog
from utils.si_filter import ORIGINAL_VOLUME_CHOICES, SI_DELAY_SECONDS_CHOICES, SI_MIX_CHANNELS, SI_VOLUME_CHOICES
from utils.trt_manifest import cache_status, manifest_path


SWITCH_OFF_IMAGE = SWITCH_OFF_IMAGE_PATH.as_posix()
SWITCH_ON_IMAGE = SWITCH_ON_IMAGE_PATH.as_posix()
HOME_COMPACT_WIDTH = 560
HOME_LOG_WIDTH = 380
HOME_HEIGHT = 560
CONFIG_ROW_HEIGHT = 34
SERVER_ICON_SIZE = 22
PROJECT_URL = "https://wapok.com"
SI_TOOLBOX_URL = "https://github.com/zerochocobo/VR-Video-Toolbox-CE"
PROJECT_LINK_HEIGHT = 28
ICON_BUTTON_SIZE = 30
LIGHT_MATCH_DEFAULT_PRESET = str(DEFAULTS["light_match_preset"])
_UI_TWO_DVR_HOLE_FILL = "soft_shift"
_TWO_DVR_STRENGTH_OPTIONS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def _retain_size_when_hidden(widget: QWidget) -> None:
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


def _int_setting(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_setting(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_switch_style(widget: QCheckBox) -> None:
    widget.setObjectName("Switch")
    widget.setStyleSheet(
        "QCheckBox#Switch { spacing: 8px; }"
        "QCheckBox#Switch::indicator {"
        "width: 38px; height: 20px;"
        "}"
        "QCheckBox#Switch::indicator:unchecked {"
        f"image: url({SWITCH_OFF_IMAGE});"
        "}"
        "QCheckBox#Switch::indicator:checked {"
        f"image: url({SWITCH_ON_IMAGE});"
        "}"
    )


def _server_button_icon(running: bool) -> QIcon:
    size = SERVER_ICON_SIZE
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = QColor("#D93025" if running else "#18A058")
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    if running:
        side = int(size * 0.52)
        offset = (size - side) // 2
        painter.drawRoundedRect(offset, offset, side, side, 2, 2)
    else:
        left = int(size * 0.34)
        top = int(size * 0.25)
        bottom = int(size * 0.75)
        right = int(size * 0.76)
        painter.drawPolygon(
            [
                QPoint(left, top),
                QPoint(left, bottom),
                QPoint(right, size // 2),
            ]
        )
    painter.end()
    return QIcon(pixmap)


def _gear_icon() -> QIcon:
    size = 22
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor("#4f5965"), 2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    center = size // 2
    for angle in range(0, 360, 45):
        painter.save()
        painter.translate(center, center)
        painter.rotate(angle)
        painter.drawLine(0, -9, 0, -7)
        painter.restore()
    painter.drawEllipse(center - 6, center - 6, 12, 12)
    painter.drawEllipse(center - 2, center - 2, 4, 4)
    painter.end()
    return QIcon(pixmap)


def _question_icon() -> QIcon:
    size = 22
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor("#4f5965"), 2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(2, 2, size - 4, size - 4)
    font = QFont()
    font.setBold(True)
    font.setPointSize(12)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "?")
    painter.end()
    return QIcon(pixmap)


def _icon_button(icon: QIcon) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(ICON_BUTTON_SIZE, ICON_BUTTON_SIZE)
    button.setText("")
    return button


def _link_label(text: str, url: str) -> QLabel:
    label = QLabel()
    label.setOpenExternalLinks(False)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    if url:
        label.setText(f'<a href="{url}">{text}</a>')
        label.linkActivated.connect(lambda value: QDesktopServices.openUrl(QUrl(value)))
    else:
        label.setText("-")
    return label


class PlayerSupportDialog(QDialog):
    def __init__(self, i18n, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("player_support.window_title"))

        title = QLabel(self.i18n.t("player_support.table_title"))
        title.setObjectName("PlayerSupportTitle")
        title.setStyleSheet("QLabel#PlayerSupportTitle { font-size: 12pt; font-weight: 700; }")

        rows = load_player_support()
        headers = [
            self.i18n.t("player_support.player"),
            self.i18n.t("player_support.alpha"),
            self.i18n.t("player_support.gray_green"),
            self.i18n.t("player_support.chroma_key"),
            self.i18n.t("player_support.website"),
            self.i18n.t("player_support.notes"),
        ]
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setWordWrap(True)
        for row_index, row in enumerate(rows):
            values = [
                row.player,
                self._support_text(row.alpha),
                self._support_text(row.gray_green),
                self._support_text(row.chroma_key),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if column else Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_index, column, item)
            table.setCellWidget(row_index, 4, _link_label(self.i18n.t("player_support.official_site"), row.website_url))
            table.setCellWidget(
                row_index,
                5,
                _link_label(self.i18n.t("player_support.install_notes"), row.notes_url),
            )
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(headers)):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        table.resizeRowsToContents()

        close_button = QPushButton(self.i18n.t("button.close"))
        close_button.clicked.connect(self.accept)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(table)
        layout.addLayout(button_row)
        self.resize(760, 330)

    def _support_text(self, supported: bool) -> str:
        return self.i18n.t("player_support.supported") if supported else "-"


class Alpha2DSettingsDialog(QDialog):
    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.settings = settings
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("alpha2d.dialog_title"))

        projection = str(settings.data.get("alpha_2d_projection") or "fisheye").lower()
        if projection not in {"fisheye", "flat3d"}:
            projection = "fisheye"

        self.fisheye_radio = QRadioButton(self.i18n.t("alpha2d.projection_fisheye"))
        self.flat3d_radio = QRadioButton(self.i18n.t("alpha2d.projection_flat3d"))
        self.projection_group = QButtonGroup(self)
        self.projection_group.addButton(self.fisheye_radio)
        self.projection_group.addButton(self.flat3d_radio)
        self.fisheye_radio.setChecked(projection == "fisheye")
        self.flat3d_radio.setChecked(projection == "flat3d")

        projection_label = QLabel(self.i18n.t("alpha2d.projection"))
        projection_row = QHBoxLayout()
        projection_row.addWidget(projection_label)
        projection_row.addWidget(self.fisheye_radio)
        projection_row.addWidget(self.flat3d_radio)
        projection_row.addStretch(1)

        try:
            distance = int(round(float(settings.data.get("alpha_2d_distance_m") or 4.0)))
        except (TypeError, ValueError):
            distance = 4
        distance = max(1, min(10, distance))
        self.distance_value = QLabel()
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(1, 10)
        self.distance_slider.setSingleStep(1)
        self.distance_slider.setPageStep(1)
        self.distance_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.distance_slider.setTickInterval(1)
        self.distance_slider.setValue(distance)
        self.distance_slider.valueChanged.connect(self._update_distance_label)

        distance_label = QLabel(self.i18n.t("alpha2d.distance"))
        distance_row = QHBoxLayout()
        distance_row.addWidget(distance_label)
        distance_row.addWidget(self.distance_slider, 1)
        distance_row.addWidget(self.distance_value)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(projection_row)
        layout.addLayout(distance_row)
        layout.addWidget(buttons)
        self._update_distance_label(distance)
        self.resize(360, 150)

    def _update_distance_label(self, value: int) -> None:
        self.distance_value.setText(f"{int(value)}m")

    def selected_projection(self) -> str:
        return "flat3d" if self.flat3d_radio.isChecked() else "fisheye"

    def selected_distance_m(self) -> float:
        return float(self.distance_slider.value())


class SISettingsDialog(QDialog):
    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.settings = settings
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("si.dialog_title"))

        self.channel = QComboBox()
        for value in SI_MIX_CHANNELS:
            self.channel.addItem(self.i18n.t(f"si.channel_{value}"), value)
        idx = self.channel.findData(str(settings.data.get("si_mix_channel") or DEFAULTS["si_mix_channel"]))
        self.channel.setCurrentIndex(max(0, idx))

        self.original_volume = QComboBox()
        for value in ORIGINAL_VOLUME_CHOICES:
            self.original_volume.addItem(f"{value}%", value)
        idx = self.original_volume.findData(_int_setting(
            settings.data.get("si_original_volume_percent"),
            DEFAULTS["si_original_volume_percent"],
        ))
        self.original_volume.setCurrentIndex(max(0, idx))

        self.si_volume = QComboBox()
        for value in SI_VOLUME_CHOICES:
            self.si_volume.addItem(f"{value}%", value)
        idx = self.si_volume.findData(_int_setting(settings.data.get("si_volume_percent"), DEFAULTS["si_volume_percent"]))
        self.si_volume.setCurrentIndex(max(0, idx))

        self.delay = QComboBox()
        for value in SI_DELAY_SECONDS_CHOICES:
            self.delay.addItem(f"{value:g}s", value)
        delay = round(_float_setting(settings.data.get("si_delay_seconds"), DEFAULTS["si_delay_seconds"]), 1)
        idx = self.delay.findData(delay)
        self.delay.setCurrentIndex(max(0, idx))

        self.duck_original = QCheckBox(self.i18n.t("si.duck_original"))
        self.duck_original.setToolTip(self.i18n.t("si.duck_original_tooltip"))
        self.duck_original.setChecked(bool(settings.data.get("si_duck_original", DEFAULTS["si_duck_original"])))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        for label_key, widget in (
            ("si.mix_channel", self.channel),
            ("si.original_volume", self.original_volume),
            ("si.si_volume", self.si_volume),
            ("si.delay", self.delay),
        ):
            row = QHBoxLayout()
            label = QLabel(self.i18n.t(label_key))
            label.setFixedWidth(110)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label)
            row.addWidget(widget, 1)
            layout.addLayout(row)
        duck_row = QHBoxLayout()
        duck_row.addSpacing(110)
        duck_row.addWidget(self.duck_original)
        duck_row.addStretch(1)
        layout.addLayout(duck_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(360, 220)

    def payload(self) -> dict:
        return {
            "mix_channel": str(self.channel.currentData() or DEFAULTS["si_mix_channel"]),
            "original_volume_percent": int(self.original_volume.currentData() or DEFAULTS["si_original_volume_percent"]),
            "si_volume_percent": int(self.si_volume.currentData() or DEFAULTS["si_volume_percent"]),
            "si_delay_seconds": float(self.delay.currentData() or DEFAULTS["si_delay_seconds"]),
            "duck_original": self.duck_original.isChecked(),
        }


class SIHelpDialog(QDialog):
    def __init__(self, i18n, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("si.help_title"))

        message = self.i18n.t("si.help_message").format(
            link=f'<a href="{SI_TOOLBOX_URL}">{SI_TOOLBOX_URL}</a>'
        )
        label = QLabel(message)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(False)
        label.linkActivated.connect(lambda value: QDesktopServices.openUrl(QUrl(value)))
        label.setMinimumWidth(560)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(self.i18n.t("button.close"))
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addWidget(label)
        layout.addWidget(buttons)
        self.resize(640, 220)


class LightMatchAdvancedDialog(QDialog):
    def __init__(self, i18n, payload: dict, live_callback, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self._live_callback = live_callback
        self._updating = False
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("light_match.custom_title"))

        self.temp = self._slider(2700, 9000, _int_setting(payload.get("temp_k"), DEFAULTS["light_match_temp_k"]))
        self.tint = self._slider(-50, 50, _int_setting(payload.get("tint"), 0))
        self.exposure = self._slider(-100, 100, int(round(_float_setting(payload.get("exposure_ev"), 0.0) * 100)))
        self.contrast = self._slider(80, 120, int(round(_float_setting(payload.get("contrast"), 1.0) * 100)))
        self.gamma = self._slider(85, 115, int(round(_float_setting(payload.get("gamma"), 1.0) * 100)))
        self.saturation = self._slider(50, 150, int(round(_float_setting(payload.get("saturation"), 1.0) * 100)))
        self._value_labels: dict[QSlider, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        for label_key, slider, left_text, right_text in (
            ("light_match.temp_k", self.temp, "2700K", "9000K"),
            ("light_match.tint", self.tint, "-50", "+50"),
            ("light_match.exposure", self.exposure, "-1EV", "+1EV"),
            ("light_match.contrast", self.contrast, "80%", "120%"),
            ("light_match.gamma", self.gamma, "0.85", "1.15"),
            ("light_match.saturation", self.saturation, "50%", "150%"),
        ):
            row = QHBoxLayout()
            row.setSpacing(6)
            label = QLabel(self.i18n.t(label_key))
            label.setFixedWidth(92)
            label.setStyleSheet("color: #5f6368;")
            value_label = QLabel()
            value_label.setFixedWidth(72)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setStyleSheet("color: #1677c7; font-weight: 600;")
            self._value_labels[slider] = value_label
            row.addWidget(label)
            left_value = QLabel(left_text)
            left_value.setStyleSheet("color: #8a8f98;")
            row.addWidget(left_value)
            row.addWidget(slider, 1)
            right_value = QLabel(right_text)
            right_value.setStyleSheet("color: #8a8f98;")
            row.addWidget(right_value)
            row.addSpacing(8)
            row.addWidget(value_label)
            layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.restore_defaults_button = QPushButton(self.i18n.t("light_match.restore_defaults"))
        bottom = QHBoxLayout()
        bottom.addWidget(self.restore_defaults_button)
        bottom.addStretch(1)
        bottom.addWidget(buttons)
        layout.addLayout(bottom)

        for slider in (self.temp, self.tint, self.exposure, self.contrast, self.gamma, self.saturation):
            slider.valueChanged.connect(self._manual_changed)
        self.restore_defaults_button.clicked.connect(self._restore_defaults)
        self._update_value_labels()
        self.resize(520, 340)

    def _slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(max(minimum, min(maximum, value)))
        return slider

    def payload(self) -> dict:
        return {
            "enabled": True,
            "temp_k": int(self.temp.value()),
            "tint": float(self.tint.value()),
            "exposure_ev": float(self.exposure.value()) / 100.0,
            "contrast": float(self.contrast.value()) / 100.0,
            "gamma": float(self.gamma.value()) / 100.0,
            "saturation": float(self.saturation.value()) / 100.0,
            "preset": "custom",
        }

    def _emit_live(self) -> None:
        if not self._updating:
            self._live_callback(self.payload())

    def _manual_changed(self) -> None:
        self._update_value_labels()
        self._emit_live()

    def _restore_defaults(self) -> None:
        self._updating = True
        for slider, value in (
            (self.temp, DEFAULTS["light_match_temp_k"]),
            (self.tint, 0),
            (self.exposure, 0),
            (self.contrast, 100),
            (self.gamma, 100),
            (self.saturation, 100),
        ):
            slider.setValue(value)
        self._updating = False
        self._update_value_labels()
        self._emit_live()

    def _update_value_labels(self) -> None:
        values = {
            self.temp: f"{self.temp.value()}K",
            self.tint: f"{self.tint.value():+d}",
            self.exposure: f"{self.exposure.value() / 100.0:+.2f}EV",
            self.contrast: f"{self.contrast.value()}%",
            self.gamma: f"{self.gamma.value() / 100.0:.2f}",
            self.saturation: f"{self.saturation.value()}%",
        }
        for slider, text in values.items():
            label = self._value_labels.get(slider)
            if label is not None:
                label.setText(text)


class HomePage(QWidget):
    def __init__(self, i18n, settings, display_version: str = "") -> None:
        super().__init__()
        self.i18n = i18n
        self.settings = settings
        self.display_version = display_version

        self.title = QLabel()
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignCenter)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("Subtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.apply_heading_fonts()
        self.title.setStyleSheet("QLabel#Title { font-size: 19pt; font-weight: 900; }")
        self.subtitle.setStyleSheet("QLabel#Subtitle { font-size: 9pt; font-weight: 400; color: #606266; }")
        self.language = QComboBox()
        self.language.addItems(["中文", "English", "日本語"])
        self.language.setFixedWidth(120)
        self.project_link = QLabel()
        self.project_link.setObjectName("ProjectLink")
        self.project_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.project_link.setFixedHeight(PROJECT_LINK_HEIGHT)
        self.project_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.project_link.setOpenExternalLinks(False)
        self.project_link.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.project_link.setStyleSheet(
            "QLabel#ProjectLink { font-size: 8.5pt; color: #606266; padding: 2px 0; }"
            "QLabel#ProjectLink a { color: #1677c7; text-decoration: underline; }"
        )
        self.video_dirs_label = QLabel()
        self.video_dirs_label.setObjectName("VideoDirsSummary")
        self.video_dirs_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.video_dirs_label.setMinimumWidth(220)
        self.video_dirs_manage_button = _icon_button(_gear_icon())

        self.server_button = QPushButton()
        self.offline_button = QPushButton()
        self.two_dvr_button = QPushButton()
        self.server_button.setMinimumHeight(58)
        self.server_button.setIconSize(QSize(SERVER_ICON_SIZE, SERVER_ICON_SIZE))
        self.offline_button.setMinimumHeight(58)
        self.two_dvr_button.setMinimumHeight(58)

        self.green_mode = QCheckBox()
        self.alpha_mode = QCheckBox()
        _apply_switch_style(self.green_mode)
        _apply_switch_style(self.alpha_mode)
        self.green_mode_label = QLabel()
        self.alpha_mode_label = QLabel()
        self.alpha_2d_button = QPushButton()
        self.alpha_2d_button.setFixedWidth(94)
        _retain_size_when_hidden(self.alpha_2d_button)
        self.video_dirs_title = QLabel()
        self.green_mode.setChecked(bool(settings.data.get("mode_green")))
        self.alpha_mode.setChecked(bool(settings.data.get("mode_alpha")))
        self.bg_color = QComboBox()
        self.bg_color.addItem("", "808080")
        self.bg_color.addItem("", "C8C8C8")
        self.bg_color.addItem("", "00FF00")
        self.bg_color.addItem("", "0000FF")
        self.bg_color.setFixedWidth(170)
        _retain_size_when_hidden(self.bg_color)
        idx = self.bg_color.findData(settings.data.get("background_color", "00FF00"))
        self.bg_color.setCurrentIndex(max(0, idx))

        self.subtitle_enable = QCheckBox()
        self.subtitle_enable.setChecked(bool(settings.data.get("subtitle_enable")))
        _apply_switch_style(self.subtitle_enable)
        self.subtitle_enable_label = QLabel()
        self.subtitle_style_button = QPushButton()
        self.subtitle_style_button.setFixedWidth(94)
        _retain_size_when_hidden(self.subtitle_style_button)
        self.player_support_button = _icon_button(_question_icon())
        self.log_toggle = QCheckBox()
        self.log_toggle.setChecked(False)
        _apply_switch_style(self.log_toggle)
        self.log_toggle_label = QLabel()
        self.debug_toggle = QCheckBox()
        self.debug_toggle.setChecked(False)
        _apply_switch_style(self.debug_toggle)
        self.debug_toggle_label = QLabel()
        _retain_size_when_hidden(self.debug_toggle)
        _retain_size_when_hidden(self.debug_toggle_label)
        self.problem_help_button = QPushButton()
        self.problem_help_button.setFixedWidth(104)
        self.home_two_dvr_label = QLabel()
        self.home_two_dvr_toggle = QCheckBox()
        self.home_two_dvr_toggle.setChecked(bool(settings.data.get("mode_two_dvr")))
        _apply_switch_style(self.home_two_dvr_toggle)
        self.home_two_dvr_strength_label = QLabel()
        self.home_two_dvr_strength = QComboBox()
        for value in _TWO_DVR_STRENGTH_OPTIONS:
            self.home_two_dvr_strength.addItem("", value)
        strength = max(0.5, min(2.0, _float_setting(
            settings.data.get("two_dvr_live_strength"),
            DEFAULTS["two_dvr_live_strength"],
        )))
        self.home_two_dvr_strength.setCurrentIndex(min(
            range(len(_TWO_DVR_STRENGTH_OPTIONS)),
            key=lambda i: abs(_TWO_DVR_STRENGTH_OPTIONS[i] - strength),
        ))
        self.home_two_dvr_strength.setFixedWidth(82)
        _retain_size_when_hidden(self.home_two_dvr_strength_label)
        _retain_size_when_hidden(self.home_two_dvr_strength)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(UI_LOG_MAX_BLOCKS)
        self.log.setFixedWidth(HOME_LOG_WIDTH)
        log_font = QFont()
        log_font.setPointSize(8)
        self.log.setFont(log_font)
        self.log.setStyleSheet("QTextEdit { font-size: 8pt; }")
        self.log.setVisible(False)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)

        buttons = QVBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.server_button)
        offline_row = QHBoxLayout()
        offline_row.setSpacing(8)
        offline_row.addWidget(self.offline_button, 1)
        offline_row.addWidget(self.two_dvr_button, 1)
        buttons.addLayout(offline_row)

        quick_config = QWidget()
        quick_config.setObjectName("QuickConfig")
        quick_config.setStyleSheet(
            "QWidget#QuickConfig {"
            "border: 1px solid #a9b0ba; border-radius: 6px; background: #fbfbfc;"
            "}"
            "QWidget#QuickConfigContent {"
            "border-top: 1px solid #d6dbe1; background: #ffffff;"
            "border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;"
            "}"
            "QWidget#QuickConfig QCheckBox, QWidget#QuickConfig QComboBox, QWidget#QuickConfig QPushButton {"
            "font-size: 9pt; padding: 2px 6px;"
            "}"
        )
        quick_config_layout = QVBoxLayout(quick_config)
        quick_config_layout.setContentsMargins(0, 0, 0, 0)
        quick_config_layout.setSpacing(0)
        self.config_header = QToolButton()
        self.config_header.setObjectName("QuickConfigHeader")
        self.config_header.setCheckable(True)
        self.config_header.setChecked(True)
        self.config_header.setArrowType(Qt.ArrowType.DownArrow)
        self.config_header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.config_header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.config_header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.config_header.setStyleSheet(
            "QToolButton#QuickConfigHeader {"
            "border: 0; padding: 7px 10px;"
            "font-size: 10pt; font-weight: 600; background: transparent;"
            "}"
            "QToolButton#QuickConfigHeader:hover { background: #f1f3f5; }"
        )
        self.config_content = QWidget()
        self.config_content.setObjectName("QuickConfigContent")
        group_layout = QVBoxLayout(self.config_content)
        group_layout.setContentsMargins(10, 8, 10, 8)
        group_layout.setSpacing(4)
        dirs_row_widget = QWidget()
        dirs_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        dirs_row = QHBoxLayout(dirs_row_widget)
        dirs_row.setContentsMargins(0, 0, 0, 0)
        dirs_row.addWidget(self.video_dirs_title)
        dirs_row.addWidget(self.video_dirs_label, 1)
        dirs_row.addStretch(1)
        dirs_row.addWidget(self.video_dirs_manage_button)
        green_row_widget = QWidget()
        green_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        green_row = QHBoxLayout(green_row_widget)
        green_row.setContentsMargins(0, 0, 0, 0)
        green_row.addWidget(self.green_mode_label)
        green_row.addWidget(self.green_mode)
        green_row.addWidget(self.bg_color)
        green_row.addStretch(1)
        green_row.addWidget(self.player_support_button)
        alpha_row_widget = QWidget()
        alpha_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        alpha_row = QHBoxLayout(alpha_row_widget)
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.addWidget(self.alpha_mode_label)
        alpha_row.addWidget(self.alpha_mode)
        alpha_row.addStretch(1)
        alpha_row.addWidget(self.alpha_2d_button)
        subtitle_row_widget = QWidget()
        subtitle_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        subtitle_row = QHBoxLayout(subtitle_row_widget)
        subtitle_row.setContentsMargins(0, 0, 0, 0)
        subtitle_row.addWidget(self.subtitle_enable_label)
        subtitle_row.addWidget(self.subtitle_enable)
        subtitle_row.addStretch(1)
        subtitle_row.addWidget(self.subtitle_style_button)
        home_two_dvr_row_widget = QWidget()
        home_two_dvr_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        home_two_dvr_row = QHBoxLayout(home_two_dvr_row_widget)
        home_two_dvr_row.setContentsMargins(0, 0, 0, 0)
        home_two_dvr_row.addWidget(self.home_two_dvr_label)
        home_two_dvr_row.addWidget(self.home_two_dvr_toggle)
        home_two_dvr_row.addSpacing(10)
        home_two_dvr_row.addWidget(self.home_two_dvr_strength_label)
        home_two_dvr_row.addWidget(self.home_two_dvr_strength)
        home_two_dvr_row.addStretch(1)
        group_layout.addWidget(dirs_row_widget)
        group_layout.addWidget(green_row_widget)
        group_layout.addWidget(alpha_row_widget)
        group_layout.addWidget(subtitle_row_widget)
        group_layout.addWidget(home_two_dvr_row_widget)
        quick_config_layout.addWidget(self.config_header)
        quick_config_layout.addWidget(self.config_content)
        performance_config = QWidget()
        performance_config.setObjectName("QuickConfig")
        performance_config.setStyleSheet(quick_config.styleSheet())
        performance_layout = QVBoxLayout(performance_config)
        performance_layout.setContentsMargins(0, 0, 0, 0)
        performance_layout.setSpacing(0)
        self.performance_header = QToolButton()
        self.performance_header.setObjectName("QuickConfigHeader")
        self.performance_header.setCheckable(True)
        self.performance_header.setChecked(False)
        self.performance_header.setArrowType(Qt.ArrowType.RightArrow)
        self.performance_header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.performance_header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.performance_header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.performance_header.setStyleSheet(self.config_header.styleSheet())
        self.performance_content = QWidget()
        self.performance_content.setObjectName("QuickConfigContent")
        performance_content_layout = QVBoxLayout(self.performance_content)
        performance_content_layout.setContentsMargins(10, 8, 10, 8)
        performance_content_layout.setSpacing(4)
        self.performance_quality_label = QLabel()
        self.performance_fps_label = QLabel()
        self.performance_output_size_label = QLabel()
        self.performance_quality = QComboBox()
        for value in ("ultrafast", "medium"):
            self.performance_quality.addItem("", value)
        self.performance_quality.setFixedWidth(140)
        idx = self.performance_quality.findData(quality_speed_value(settings.data.get("quality_speed")))
        self.performance_quality.setCurrentIndex(max(0, idx))
        self.performance_fps = QComboBox()
        self.performance_fps.addItem("", 0)
        for value in (20, 30, 40, 50, 60):
            self.performance_fps.addItem(str(value), value)
        self.performance_fps.setFixedWidth(120)
        idx = self.performance_fps.findData(_int_setting(settings.data.get("passthrough_max_fps"), 0))
        self.performance_fps.setCurrentIndex(max(0, idx))
        self.performance_fps_help = _icon_button(_question_icon())
        self.performance_output_size = QComboBox()
        self.performance_output_size.addItem("", 0)
        self.performance_output_size.addItem("", 4096)
        self.performance_output_size.addItem("", 8192)
        self.performance_output_size.setFixedWidth(150)
        idx = self.performance_output_size.findData(_int_setting(settings.data.get("decode_max_side"), 4096))
        self.performance_output_size.setCurrentIndex(max(0, idx))
        self.trt_enabled_label = QLabel()
        self.trt_enabled = QCheckBox()
        self.trt_enabled.setChecked(str(settings.data.get("inference_backend") or "cuda").lower() == "tensorrt")
        _apply_switch_style(self.trt_enabled)
        self.trt_configure_button = QPushButton()
        self.trt_status_label = QLabel()
        self.trt_status_label.setStyleSheet("color: #5f6368;")
        self.trt_cache_watcher = QFileSystemWatcher(self)
        self.trt_cache_watcher.directoryChanged.connect(lambda _path: self._update_trt_state())
        self.trt_cache_watcher.fileChanged.connect(lambda _path: self._update_trt_state())
        self._refresh_trt_watcher()
        self.light_match_enabled = QCheckBox()
        self.light_match_enabled.setChecked(bool(settings.data.get("light_match_enabled")))
        _apply_switch_style(self.light_match_enabled)
        self.light_match_enabled_label = QLabel()
        self.light_match_help = _icon_button(_question_icon())
        self.light_match_advanced_button = QPushButton()
        self.si_mix_enabled = QCheckBox()
        self.si_mix_enabled.setChecked(bool(settings.data.get("si_enabled")))
        _apply_switch_style(self.si_mix_enabled)
        self.si_mix_enabled_label = QLabel()
        self.si_mix_settings_button = QPushButton()
        self.si_mix_settings_button.setFixedWidth(94)
        self.si_mix_help = _icon_button(_question_icon())
        self.light_match_preset = QComboBox()
        for key in ("home_warm", "daylight", "night_cool", "custom"):
            self.light_match_preset.addItem("", key)
        self.light_match_preset.setFixedWidth(150)
        idx = self.light_match_preset.findData(settings.data.get("light_match_preset") or LIGHT_MATCH_DEFAULT_PRESET)
        self.light_match_preset.setCurrentIndex(max(0, idx))
        self._light_match_live_timer = QTimer(self)
        self._light_match_live_timer.setSingleShot(True)
        self._light_match_live_timer.setInterval(80)
        self._light_match_live_timer.timeout.connect(self._send_light_match_live_update)
        memory_row_widget = QWidget()
        memory_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        memory_row = QHBoxLayout(memory_row_widget)
        memory_row.setContentsMargins(0, 0, 0, 0)
        memory_row.addWidget(self.performance_quality_label)
        memory_row.addWidget(self.performance_quality)
        memory_row.addStretch(1)
        fps_row_widget = QWidget()
        fps_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        fps_row = QHBoxLayout(fps_row_widget)
        fps_row.setContentsMargins(0, 0, 0, 0)
        fps_row.addWidget(self.performance_fps_label)
        fps_row.addWidget(self.performance_fps)
        fps_row.addStretch(1)
        fps_row.addWidget(self.performance_fps_help)
        output_size_row_widget = QWidget()
        output_size_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        output_size_row = QHBoxLayout(output_size_row_widget)
        output_size_row.setContentsMargins(0, 0, 0, 0)
        output_size_row.addWidget(self.performance_output_size_label)
        output_size_row.addWidget(self.performance_output_size)
        output_size_row.addStretch(1)
        trt_row_widget = QWidget()
        trt_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        trt_row = QHBoxLayout(trt_row_widget)
        trt_row.setContentsMargins(0, 0, 0, 0)
        trt_row.addWidget(self.trt_enabled_label)
        trt_row.addWidget(self.trt_enabled)
        trt_row.addWidget(self.trt_configure_button)
        trt_row.addWidget(self.trt_status_label)
        trt_row.addStretch(1)
        performance_content_layout.addWidget(memory_row_widget)
        performance_content_layout.addWidget(fps_row_widget)
        performance_content_layout.addWidget(output_size_row_widget)
        performance_content_layout.addWidget(trt_row_widget)
        performance_layout.addWidget(self.performance_header)
        performance_layout.addWidget(self.performance_content)
        self.performance_content.setVisible(False)

        light_match_config = QWidget()
        light_match_config.setObjectName("QuickConfig")
        light_match_config.setStyleSheet(quick_config.styleSheet())
        light_match_layout = QVBoxLayout(light_match_config)
        light_match_layout.setContentsMargins(0, 0, 0, 0)
        light_match_layout.setSpacing(0)
        self.light_match_header = QToolButton()
        self.light_match_header.setObjectName("QuickConfigHeader")
        self.light_match_header.setCheckable(True)
        self.light_match_header.setChecked(False)
        self.light_match_header.setArrowType(Qt.ArrowType.RightArrow)
        self.light_match_header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.light_match_header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.light_match_header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.light_match_header.setStyleSheet(self.config_header.styleSheet())
        self.light_match_header.setMinimumHeight(42)
        self.light_match_content = QWidget()
        self.light_match_content.setObjectName("QuickConfigContent")
        light_content_layout = QVBoxLayout(self.light_match_content)
        light_content_layout.setContentsMargins(10, 8, 10, 8)
        light_content_layout.setSpacing(4)
        light_enable_row_widget = QWidget()
        light_enable_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        light_enable_row = QHBoxLayout(light_enable_row_widget)
        light_enable_row.setContentsMargins(0, 0, 0, 0)
        light_enable_row.addWidget(self.light_match_enabled_label)
        light_enable_row.addWidget(self.light_match_enabled)
        light_enable_row.addWidget(self.light_match_preset)
        light_enable_row.addWidget(self.light_match_advanced_button)
        light_enable_row.addStretch(1)
        light_enable_row.addWidget(self.light_match_help)
        si_row_widget = QWidget()
        si_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        si_row = QHBoxLayout(si_row_widget)
        si_row.setContentsMargins(0, 0, 0, 0)
        si_row.addWidget(self.si_mix_enabled_label)
        si_row.addWidget(self.si_mix_enabled)
        si_row.addStretch(1)
        si_row.addWidget(self.si_mix_settings_button)
        si_row.addWidget(self.si_mix_help)
        log_row_widget = QWidget()
        log_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        log_row = QHBoxLayout(log_row_widget)
        log_row.setContentsMargins(0, 0, 0, 0)
        log_row.addWidget(self.log_toggle_label)
        log_row.addWidget(self.log_toggle)
        log_row.addSpacing(16)
        log_row.addWidget(self.debug_toggle_label)
        log_row.addWidget(self.debug_toggle)
        log_row.addStretch(1)
        log_row.addWidget(self.problem_help_button)
        light_content_layout.addWidget(light_enable_row_widget)
        light_content_layout.addWidget(si_row_widget)
        light_content_layout.addWidget(log_row_widget)
        light_match_layout.addWidget(self.light_match_header)
        light_match_layout.addWidget(self.light_match_content)
        self.light_match_content.setVisible(False)
        for label in (
            self.video_dirs_title,
            self.green_mode_label,
            self.alpha_mode_label,
            self.subtitle_enable_label,
            self.home_two_dvr_label,
            self.log_toggle_label,
            self.performance_quality_label,
            self.performance_fps_label,
            self.performance_output_size_label,
            self.trt_enabled_label,
            self.light_match_enabled_label,
            self.si_mix_enabled_label,
        ):
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        left_panel = QWidget()
        left_panel.setFixedWidth(HOME_COMPACT_WIDTH)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 16, 20, 16)
        left_layout.setSpacing(0)
        left_layout.addLayout(title_box)
        left_layout.addSpacing(12)
        left_layout.addLayout(buttons)
        left_layout.addSpacing(12)
        left_layout.addWidget(quick_config)
        left_layout.addWidget(performance_config)
        left_layout.addWidget(light_match_config)
        left_layout.addStretch(1)
        left_layout.addWidget(self.project_link)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(left_panel)
        layout.addWidget(self.log)
        self.config_group = quick_config
        self.performance_group = performance_config
        self.light_match_group = light_match_config
        self.retranslate()
        self._bind_settings()

    def apply_heading_fonts(self) -> None:
        title_font = QFont()
        title_font.setPointSize(19)
        title_font.setBold(True)
        title_font.setWeight(QFont.Weight.Black)
        self.title.setFont(title_font)
        subtitle_font = QFont()
        subtitle_font.setPointSize(9)
        self.subtitle.setFont(subtitle_font)

    def sizeHint(self) -> QSize:
        width = HOME_COMPACT_WIDTH + (HOME_LOG_WIDTH if self.log_toggle.isChecked() else 0)
        return QSize(width, self._current_home_height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _bind_settings(self) -> None:
        self.green_mode.toggled.connect(self._save)
        self.alpha_mode.toggled.connect(self._save)
        self.bg_color.currentIndexChanged.connect(self._save)
        self.subtitle_enable.toggled.connect(self._save)
        self.home_two_dvr_toggle.toggled.connect(self._save)
        self.home_two_dvr_strength.currentIndexChanged.connect(self._save)
        self.performance_quality.currentIndexChanged.connect(self._save)
        self.performance_fps.currentIndexChanged.connect(self._save)
        self.performance_output_size.currentIndexChanged.connect(self._save)
        self.trt_enabled.toggled.connect(self._save)
        self.trt_configure_button.clicked.connect(self.show_trt_config)
        self.light_match_enabled.toggled.connect(self._save_light_match)
        self.light_match_preset.currentIndexChanged.connect(self._preset_light_match)
        self.light_match_advanced_button.clicked.connect(self.show_light_match_advanced)
        self.si_mix_enabled.toggled.connect(self._save_si_mix)
        self.si_mix_settings_button.clicked.connect(self.show_si_mix_settings)
        self.si_mix_help.clicked.connect(self.show_si_mix_help)
        self.performance_fps_help.clicked.connect(self._show_fps_help)
        self.config_header.toggled.connect(self._toggle_quick_config)
        self.performance_header.toggled.connect(self._toggle_performance_config)
        self.light_match_header.toggled.connect(self._toggle_light_match_config)
        self.green_mode.toggled.connect(self._update_enabled)
        self.alpha_mode.toggled.connect(self._update_enabled)
        self.home_two_dvr_toggle.toggled.connect(self._update_enabled)
        self.subtitle_enable.toggled.connect(self._update_enabled)
        self.log_toggle.toggled.connect(self._update_enabled)
        self.video_dirs_manage_button.clicked.connect(self.manage_video_dirs)
        self.player_support_button.clicked.connect(self.show_player_support)
        self.alpha_2d_button.clicked.connect(self.show_alpha_2d_settings)
        self.light_match_help.clicked.connect(self._show_light_match_help)
        self.problem_help_button.clicked.connect(self.show_problem_help)
        self._update_enabled()
        self._update_trt_state()
        self._update_light_match_visibility()
        self.update_video_dirs_summary()

    def _save(self) -> None:
        self.settings.data["mode_green"] = self.green_mode.isChecked()
        self.settings.data["mode_alpha"] = self.alpha_mode.isChecked()
        self.settings.data["mode_two_dvr"] = self.home_two_dvr_toggle.isChecked()
        self.settings.data["two_dvr_live_model"] = DEFAULTS["two_dvr_live_model"]
        self.settings.data["two_dvr_live_hole_fill"] = _UI_TWO_DVR_HOLE_FILL
        self.settings.data["two_dvr_live_eye_distance"] = DEFAULTS["two_dvr_live_eye_distance"]
        self.settings.data["two_dvr_live_strength"] = float(
            self.home_two_dvr_strength.currentData() or DEFAULTS["two_dvr_live_strength"]
        )
        self.settings.data["background_color"] = self.bg_color.currentData()
        self.settings.data["quality_speed"] = self.performance_quality.currentData()
        self.settings.data["alpha_stride"] = 1
        self.settings.data["passthrough_max_fps"] = self.performance_fps.currentData()
        self.settings.data["decode_max_side"] = self.performance_output_size.currentData()
        self.settings.data["inference_backend"] = "tensorrt" if self.trt_enabled.isChecked() else "cuda"
        self.settings.data["subtitle_enable"] = self.subtitle_enable.isChecked()
        self.settings.save()

    def _refresh_trt_watcher(self) -> None:
        for path in self.trt_cache_watcher.files():
            self.trt_cache_watcher.removePath(path)
        for path in self.trt_cache_watcher.directories():
            self.trt_cache_watcher.removePath(path)
        cache_dir = manifest_path(scope="realtime").parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.trt_cache_watcher.addPath(str(cache_dir))
        manifest = manifest_path(scope="realtime")
        if manifest.exists():
            self.trt_cache_watcher.addPath(str(manifest))

    def _trt_status(self) -> str:
        try:
            return cache_status(scope="realtime")
        except Exception:
            return "failed"

    def _update_trt_state(self) -> None:
        self._refresh_trt_watcher()
        status = self._trt_status()
        ready = status == "ready"
        self.trt_enabled.blockSignals(True)
        self.trt_enabled.setEnabled(ready)
        if not ready:
            self.trt_enabled.setChecked(False)
            if str(self.settings.data.get("inference_backend") or "cuda").lower() == "tensorrt":
                self.settings.data["inference_backend"] = "cuda"
                self.settings.save()
        else:
            self.trt_enabled.setChecked(str(self.settings.data.get("inference_backend") or "cuda").lower() == "tensorrt")
        self.trt_enabled.blockSignals(False)
        self.trt_status_label.setText(self.i18n.t("trt.status_" + status))
        self.trt_enabled.setToolTip("" if ready else self.i18n.t("trt.build_first_tooltip"))

    def _light_match_payload(self) -> dict:
        return {
            "enabled": self.light_match_enabled.isChecked(),
            "temp_k": _int_setting(self.settings.data.get("light_match_temp_k"), DEFAULTS["light_match_temp_k"]),
            "tint": _float_setting(self.settings.data.get("light_match_tint"), 0.0),
            "exposure_ev": _float_setting(self.settings.data.get("light_match_exposure_ev"), 0.0),
            "contrast": _float_setting(self.settings.data.get("light_match_contrast"), 1.0),
            "gamma": _float_setting(self.settings.data.get("light_match_gamma"), 1.0),
            "saturation": _float_setting(self.settings.data.get("light_match_saturation"), 1.0),
            "preset": str(self.light_match_preset.currentData() or LIGHT_MATCH_DEFAULT_PRESET),
        }

    def _apply_light_match_payload(self, payload: dict, save: bool) -> None:
        for key, value in payload.items():
            self.settings.data[f"light_match_{key}"] = value
        idx = self.light_match_preset.findData(payload.get("preset", LIGHT_MATCH_DEFAULT_PRESET))
        if idx >= 0 and self.light_match_preset.currentIndex() != idx:
            self.light_match_preset.blockSignals(True)
            self.light_match_preset.setCurrentIndex(idx)
            self.light_match_preset.blockSignals(False)
        if self.light_match_enabled.isChecked() != bool(payload.get("enabled")):
            self.light_match_enabled.blockSignals(True)
            self.light_match_enabled.setChecked(bool(payload.get("enabled")))
            self.light_match_enabled.blockSignals(False)
        self._update_light_match_visibility()
        if save:
            self.settings.save()
        self._send_light_match_live_update(payload)

    def _save_light_match(self) -> None:
        self._update_light_match_visibility()
        self._apply_light_match_payload(self._light_match_payload(), save=True)

    def _send_light_match_live_update(self, payload: dict | None = None) -> None:
        payload = payload or self._light_match_payload()
        port = str(os.environ.get("PT_HTTP_PORT") or "8200").strip() or "8200"
        url = f"http://127.0.0.1:{port}/control/light_match"

        def worker() -> None:
            try:
                data = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=data,
                    method="PUT",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=0.35) as response:
                    response.read(2048)
            except Exception:
                pass

        threading.Thread(target=worker, name="light-match-live-update", daemon=True).start()

    def _si_mix_payload(self) -> dict:
        return {
            "enabled": self.si_mix_enabled.isChecked(),
            "mix_channel": str(self.settings.data.get("si_mix_channel") or DEFAULTS["si_mix_channel"]),
            "original_volume_percent": _int_setting(
                self.settings.data.get("si_original_volume_percent"),
                DEFAULTS["si_original_volume_percent"],
            ),
            "si_volume_percent": _int_setting(self.settings.data.get("si_volume_percent"), DEFAULTS["si_volume_percent"]),
            "si_delay_seconds": _float_setting(self.settings.data.get("si_delay_seconds"), DEFAULTS["si_delay_seconds"]),
            "duck_original": bool(self.settings.data.get("si_duck_original", DEFAULTS["si_duck_original"])),
        }

    def _save_si_mix(self) -> None:
        self.settings.data["si_enabled"] = self.si_mix_enabled.isChecked()
        self.settings.save()
        self._send_si_mix_live_update()

    def _send_si_mix_live_update(self, payload: dict | None = None) -> None:
        payload = payload or self._si_mix_payload()
        port = str(os.environ.get("PT_HTTP_PORT") or "8200").strip() or "8200"
        url = f"http://127.0.0.1:{port}/control/si_mix"

        def worker() -> None:
            try:
                data = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=data,
                    method="PUT",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=0.35) as response:
                    response.read(2048)
            except Exception:
                pass

        threading.Thread(target=worker, name="si-mix-live-update", daemon=True).start()

    def _preset_light_match(self) -> None:
        preset = str(self.light_match_preset.currentData() or LIGHT_MATCH_DEFAULT_PRESET)
        self._update_light_match_visibility()
        payload = self._light_match_payload()
        values = LIGHT_MATCH_PRESETS.get(preset, {})
        payload.update(values)
        payload["preset"] = preset
        self._apply_light_match_payload(payload, save=True)

    def show_light_match_advanced(self) -> None:
        before = self._light_match_payload()
        dialog = LightMatchAdvancedDialog(self.i18n, before, lambda payload: self._send_light_match_live_update(payload), self)
        if dialog.exec() == LightMatchAdvancedDialog.DialogCode.Accepted:
            self._apply_light_match_payload(dialog.payload(), save=True)
            return
        self._apply_light_match_payload(before, save=False)

    def show_si_mix_settings(self) -> None:
        dialog = SISettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != SISettingsDialog.DialogCode.Accepted:
            return
        payload = dialog.payload()
        self.settings.data["si_mix_channel"] = payload["mix_channel"]
        self.settings.data["si_original_volume_percent"] = payload["original_volume_percent"]
        self.settings.data["si_volume_percent"] = payload["si_volume_percent"]
        self.settings.data["si_delay_seconds"] = payload["si_delay_seconds"]
        self.settings.data["si_duck_original"] = payload["duck_original"]
        self.settings.save()
        live_payload = self._si_mix_payload()
        live_payload.update(payload)
        self._send_si_mix_live_update(live_payload)

    def show_si_mix_help(self) -> None:
        SIHelpDialog(self.i18n, self).exec()

    def _update_light_match_visibility(self) -> None:
        enabled = self.light_match_enabled.isChecked()
        preset = str(self.light_match_preset.currentData() or LIGHT_MATCH_DEFAULT_PRESET)
        self.light_match_preset.setVisible(enabled)
        self.light_match_advanced_button.setVisible(enabled and preset == "custom")

    def _show_light_match_help(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(self, self.i18n.t("light_match.title"), self.i18n.t("light_match.help"))

    def _show_fps_help(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(self, self.i18n.t("performance.output_fps"), self.i18n.t("performance.output_fps_help"))

    def sync_from_settings(self) -> None:
        value = quality_speed_value(self.settings.data.get("quality_speed"))
        idx = self.performance_quality.findData(value)
        if idx >= 0 and self.performance_quality.currentIndex() != idx:
            self.performance_quality.blockSignals(True)
            self.performance_quality.setCurrentIndex(idx)
            self.performance_quality.blockSignals(False)

    def manage_video_dirs(self) -> None:
        dialog = VideoDirsDialog(self.i18n, self.settings.video_dirs(), self)
        if dialog.exec() != VideoDirsDialog.DialogCode.Accepted:
            return
        self.settings.set_video_dirs(dialog.directories())
        self.settings.save()
        self.update_video_dirs_summary()

    def show_player_support(self) -> None:
        dialog = PlayerSupportDialog(self.i18n, self)
        dialog.exec()

    def show_alpha_2d_settings(self) -> None:
        dialog = Alpha2DSettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != Alpha2DSettingsDialog.DialogCode.Accepted:
            return
        self.settings.data["alpha_2d_projection"] = dialog.selected_projection()
        self.settings.data["alpha_2d_distance_m"] = dialog.selected_distance_m()
        self.settings.save()

    def show_trt_config(self) -> None:
        dialog = TensorRTConfigDialog(self.i18n, self, scope="realtime")
        dialog.exec()
        if self._trt_status() == "ready":
            self.settings.data["inference_backend"] = "tensorrt"
            self.settings.save()
        self._update_trt_state()

    def show_problem_help(self) -> None:
        QMessageBox.information(
            self,
            self.i18n.t("problem_help.title"),
            self.i18n.t("problem_help.message"),
        )

    def update_video_dirs_summary(self) -> None:
        roots = build_media_roots(parse_video_dirs("|".join(self.settings.video_dirs()), UI_ROOT / "videos"))
        names = [root.label for root in roots]
        text = ", ".join(names) if names else self.i18n.t("video_dirs.none")
        self.video_dirs_label.setText(text)
        self.video_dirs_label.setToolTip("|".join(str(root.path) for root in roots))

    def _update_enabled(self) -> None:
        self.bg_color.setVisible(self.green_mode.isChecked())
        self.alpha_2d_button.setVisible(self.alpha_mode.isChecked())
        two_dvr_enabled = self.home_two_dvr_toggle.isChecked()
        self.home_two_dvr_strength_label.setVisible(two_dvr_enabled)
        self.home_two_dvr_strength.setVisible(two_dvr_enabled)
        self.subtitle_style_button.setVisible(self.subtitle_enable.isChecked())
        self.log.setVisible(self.log_toggle.isChecked())
        debug_visible = self.log_toggle.isChecked()
        self.debug_toggle_label.setVisible(debug_visible)
        self.debug_toggle.setVisible(debug_visible)
        if not debug_visible:
            self.debug_toggle.setChecked(False)
        self._adjust_window()

    def _toggle_quick_config(self, expanded: bool) -> None:
        if expanded and self.performance_header.isChecked():
            self.performance_header.blockSignals(True)
            self.performance_header.setChecked(False)
            self.performance_header.blockSignals(False)
            self.performance_content.setVisible(False)
            self.performance_header.setArrowType(Qt.ArrowType.RightArrow)
            self._update_performance_config_title()
        if expanded and self.light_match_header.isChecked():
            self.light_match_header.blockSignals(True)
            self.light_match_header.setChecked(False)
            self.light_match_header.blockSignals(False)
            self.light_match_content.setVisible(False)
            self.light_match_header.setArrowType(Qt.ArrowType.RightArrow)
            self._update_light_match_config_title()
        self.config_content.setVisible(expanded)
        self.config_header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._update_quick_config_title()
        self.config_group.updateGeometry()
        self._adjust_window()

    def _toggle_performance_config(self, expanded: bool) -> None:
        if expanded and self.config_header.isChecked():
            self.config_header.blockSignals(True)
            self.config_header.setChecked(False)
            self.config_header.blockSignals(False)
            self.config_content.setVisible(False)
            self.config_header.setArrowType(Qt.ArrowType.RightArrow)
            self._update_quick_config_title()
        if expanded and self.light_match_header.isChecked():
            self.light_match_header.blockSignals(True)
            self.light_match_header.setChecked(False)
            self.light_match_header.blockSignals(False)
            self.light_match_content.setVisible(False)
            self.light_match_header.setArrowType(Qt.ArrowType.RightArrow)
            self._update_light_match_config_title()
        self.performance_content.setVisible(expanded)
        self.performance_header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._update_performance_config_title()
        self.performance_group.updateGeometry()
        self._adjust_window()

    def _toggle_light_match_config(self, expanded: bool) -> None:
        if expanded and self.config_header.isChecked():
            self.config_header.blockSignals(True)
            self.config_header.setChecked(False)
            self.config_header.blockSignals(False)
            self.config_content.setVisible(False)
            self.config_header.setArrowType(Qt.ArrowType.RightArrow)
            self._update_quick_config_title()
        if expanded and self.performance_header.isChecked():
            self.performance_header.blockSignals(True)
            self.performance_header.setChecked(False)
            self.performance_header.blockSignals(False)
            self.performance_content.setVisible(False)
            self.performance_header.setArrowType(Qt.ArrowType.RightArrow)
            self._update_performance_config_title()
        self.light_match_content.setVisible(expanded)
        self.light_match_header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._update_light_match_config_title()
        self.light_match_group.updateGeometry()
        self._adjust_window()

    def _update_quick_config_title(self) -> None:
        key = "group.quick_config" if self.config_header.isChecked() else "group.quick_config_short"
        self.config_header.setText(self.i18n.t(key))

    def _update_performance_config_title(self) -> None:
        key = "group.performance_config" if self.performance_header.isChecked() else "group.performance_config_short"
        self.performance_header.setText(self.i18n.t(key))

    def _update_light_match_config_title(self) -> None:
        key = "group.light_match_config" if self.light_match_header.isChecked() else "group.light_match_config_short"
        self.light_match_header.setText(self.i18n.t(key))

    def _current_home_height(self) -> int:
        return HOME_HEIGHT

    def _adjust_window(self) -> None:
        window = self.window()
        if window is not None:
            width = HOME_COMPACT_WIDTH + (HOME_LOG_WIDTH if self.log_toggle.isChecked() else 0)
            height = self._current_home_height()
            self.setMinimumWidth(width)
            self.setMaximumWidth(width)
            self.resize(width, height)
            self.updateGeometry()
            self.layout().activate()
            window.setMinimumWidth(width)
            window.setMaximumWidth(width)
            window.setMinimumHeight(height)
            window.setMaximumHeight(height)
            window.resize(width, height)

    def set_server_running(self, running: bool) -> None:
        self.server_button.setText(self.i18n.t("button.stop_server") if running else self.i18n.t("button.start_server"))
        self.server_button.setIcon(_server_button_icon(running))

    def append_log(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def clear_log(self) -> None:
        self.log.clear()

    def _sync_quick_label_widths(self) -> None:
        labels = (
            self.video_dirs_title,
            self.green_mode_label,
            self.alpha_mode_label,
            self.subtitle_enable_label,
            self.home_two_dvr_label,
            self.log_toggle_label,
            self.performance_quality_label,
            self.performance_fps_label,
            self.performance_output_size_label,
            self.trt_enabled_label,
            self.light_match_enabled_label,
            self.si_mix_enabled_label,
        )
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setFixedWidth(width)

    def retranslate(self) -> None:
        self.title.setText(self.i18n.t("app.title"))
        self.subtitle.setText(self.i18n.t("app.subtitle"))
        self.project_link.setText(
            f'{self.i18n.t("project.url_label")}：<a href="{PROJECT_URL}">{PROJECT_URL}</a>'
        )
        self.server_button.setText(self.i18n.t("button.start_server"))
        self.offline_button.setText(self.i18n.t("button.offline"))
        self.two_dvr_button.setText(self.i18n.t("button.two_dvr"))
        self._update_quick_config_title()
        self._update_performance_config_title()
        self._update_light_match_config_title()
        self.video_dirs_manage_button.setToolTip(self.i18n.t("button.manage"))
        self.video_dirs_title.setText(self.i18n.t("video_dirs.label"))
        self.green_mode.setText("")
        self.green_mode_label.setText(self.i18n.t("mode.green"))
        self.alpha_mode.setText("")
        self.alpha_mode_label.setText(self.i18n.t("mode.alpha"))
        self.alpha_2d_button.setText(self.i18n.t("alpha2d.button"))
        self.subtitle_enable.setText("")
        self.subtitle_enable_label.setText(self.i18n.t("subtitle.enable"))
        self.subtitle_style_button.setText(self.i18n.t("subtitle.style_config"))
        self.player_support_button.setToolTip(self.i18n.t("player_support.window_title"))
        self.home_two_dvr_toggle.setText("")
        self.home_two_dvr_label.setText(self.i18n.t("home.two_dvr_toggle"))
        self.home_two_dvr_strength_label.setText(self.i18n.t("twodvr.strength"))
        for i, value in enumerate(_TWO_DVR_STRENGTH_OPTIONS):
            self.home_two_dvr_strength.setItemText(i, f"{int(round(value * 100))}%")
        self.log_toggle.setText("")
        self.log_toggle_label.setText(self.i18n.t("log.show"))
        self.problem_help_button.setText(self.i18n.t("problem_help.button"))
        self.debug_toggle.setText("")
        self.debug_toggle_label.setText(self.i18n.t("log.debug"))
        self.performance_quality_label.setText(self.i18n.t("performance.quality_speed"))
        self.performance_fps_label.setText(self.i18n.t("performance.output_fps"))
        self.performance_fps_help.setToolTip(self.i18n.t("performance.output_fps_help"))
        self.performance_output_size_label.setText(self.i18n.t("performance.output_size"))
        self.trt_enabled.setText("")
        self.trt_enabled_label.setText(self.i18n.t("trt.row_label"))
        self.trt_configure_button.setText(self.i18n.t("trt.configure"))
        self.light_match_enabled.setText("")
        self.light_match_enabled_label.setText(self.i18n.t("light_match.enabled"))
        self.light_match_help.setToolTip(self.i18n.t("light_match.help"))
        self.light_match_advanced_button.setText(self.i18n.t("light_match.advanced"))
        self.si_mix_enabled.setText("")
        self.si_mix_enabled_label.setText(self.i18n.t("si.enabled"))
        self.si_mix_settings_button.setText(self.i18n.t("si.audio_settings"))
        self.si_mix_help.setToolTip(self.i18n.t("si.help_title"))
        self.performance_fps.setItemText(0, self.i18n.t("performance.output_fps_unlimited"))
        for i, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium")):
            self.performance_quality.setItemText(i, self.i18n.t(key))
        self.performance_output_size.setItemText(0, self.i18n.t("performance.output_size_original"))
        self.performance_output_size.setItemText(1, self.i18n.t("performance.output_size_4k"))
        self.performance_output_size.setItemText(2, self.i18n.t("performance.output_size_8k"))
        for i, key in enumerate((
            "light_match.preset_home_warm",
            "light_match.preset_daylight",
            "light_match.preset_night_cool",
            "light_match.preset_custom",
        )):
            self.light_match_preset.setItemText(i, self.i18n.t(key))
        self.update_video_dirs_summary()
        for i, key in enumerate(("bg.neutral_gray", "bg.light_gray", "bg.soft_green", "bg.soft_blue")):
            self.bg_color.setItemText(i, self.i18n.t(key))
        self._sync_quick_label_widths()
        self._update_trt_state()
