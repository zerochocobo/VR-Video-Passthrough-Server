from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QAbstractItemView,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
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
from ui.settings import ROOT as UI_ROOT, quality_speed_value


SWITCH_OFF_IMAGE = SWITCH_OFF_IMAGE_PATH.as_posix()
SWITCH_ON_IMAGE = SWITCH_ON_IMAGE_PATH.as_posix()
HOME_COMPACT_WIDTH = 560
HOME_LOG_WIDTH = 380
HOME_HEIGHT = 508
CONFIG_ROW_HEIGHT = 34
SERVER_ICON_SIZE = 22
PROJECT_URL = "https://github.com/zerochocobo/VR-Video-Passthrough-Server"
PROJECT_LINK_HEIGHT = 28
ICON_BUTTON_SIZE = 30


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
        self.server_button.setMinimumHeight(58)
        self.server_button.setIconSize(QSize(SERVER_ICON_SIZE, SERVER_ICON_SIZE))
        self.offline_button.setMinimumHeight(58)

        self.green_mode = QCheckBox()
        self.alpha_mode = QCheckBox()
        _apply_switch_style(self.green_mode)
        _apply_switch_style(self.alpha_mode)
        self.green_mode_label = QLabel()
        self.alpha_mode_label = QLabel()
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
        buttons.addWidget(self.offline_button)

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
        subtitle_row_widget = QWidget()
        subtitle_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        subtitle_row = QHBoxLayout(subtitle_row_widget)
        subtitle_row.setContentsMargins(0, 0, 0, 0)
        subtitle_row.addWidget(self.subtitle_enable_label)
        subtitle_row.addWidget(self.subtitle_enable)
        subtitle_row.addStretch(1)
        subtitle_row.addWidget(self.subtitle_style_button)
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
        group_layout.addWidget(dirs_row_widget)
        group_layout.addWidget(green_row_widget)
        group_layout.addWidget(alpha_row_widget)
        group_layout.addWidget(subtitle_row_widget)
        group_layout.addWidget(log_row_widget)
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
        for value in (20, 30, 40, 60):
            self.performance_fps.addItem(str(value), value)
        self.performance_fps.setFixedWidth(120)
        idx = self.performance_fps.findData(_int_setting(settings.data.get("passthrough_max_fps"), 0))
        self.performance_fps.setCurrentIndex(max(0, idx))
        self.performance_output_size = QComboBox()
        self.performance_output_size.addItem("", 0)
        self.performance_output_size.addItem("", 4096)
        self.performance_output_size.addItem("", 8192)
        self.performance_output_size.setFixedWidth(150)
        idx = self.performance_output_size.findData(_int_setting(settings.data.get("decode_max_side"), 4096))
        self.performance_output_size.setCurrentIndex(max(0, idx))
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
        output_size_row_widget = QWidget()
        output_size_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        output_size_row = QHBoxLayout(output_size_row_widget)
        output_size_row.setContentsMargins(0, 0, 0, 0)
        output_size_row.addWidget(self.performance_output_size_label)
        output_size_row.addWidget(self.performance_output_size)
        output_size_row.addStretch(1)
        performance_content_layout.addWidget(memory_row_widget)
        performance_content_layout.addWidget(fps_row_widget)
        performance_content_layout.addWidget(output_size_row_widget)
        performance_layout.addWidget(self.performance_header)
        performance_layout.addWidget(self.performance_content)
        self.performance_content.setVisible(False)
        for label in (
            self.video_dirs_title,
            self.green_mode_label,
            self.alpha_mode_label,
            self.subtitle_enable_label,
            self.log_toggle_label,
            self.performance_quality_label,
            self.performance_fps_label,
            self.performance_output_size_label,
        ):
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        left_panel = QWidget()
        left_panel.setFixedWidth(HOME_COMPACT_WIDTH)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 16, 20, 16)
        left_layout.setSpacing(12)
        left_layout.addLayout(title_box)
        left_layout.addLayout(buttons)
        left_layout.addWidget(quick_config)
        left_layout.addWidget(performance_config)
        left_layout.addStretch(1)
        left_layout.addWidget(self.project_link)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(left_panel)
        layout.addWidget(self.log)
        self.config_group = quick_config
        self.performance_group = performance_config
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
        self.performance_quality.currentIndexChanged.connect(self._save)
        self.performance_fps.currentIndexChanged.connect(self._save)
        self.performance_output_size.currentIndexChanged.connect(self._save)
        self.config_header.toggled.connect(self._toggle_quick_config)
        self.performance_header.toggled.connect(self._toggle_performance_config)
        self.green_mode.toggled.connect(self._update_enabled)
        self.subtitle_enable.toggled.connect(self._update_enabled)
        self.log_toggle.toggled.connect(self._update_enabled)
        self.video_dirs_manage_button.clicked.connect(self.manage_video_dirs)
        self.player_support_button.clicked.connect(self.show_player_support)
        self._update_enabled()
        self.update_video_dirs_summary()

    def _save(self) -> None:
        self.settings.data["mode_green"] = self.green_mode.isChecked()
        self.settings.data["mode_alpha"] = self.alpha_mode.isChecked()
        self.settings.data["background_color"] = self.bg_color.currentData()
        self.settings.data["quality_speed"] = self.performance_quality.currentData()
        self.settings.data["alpha_stride"] = 1
        self.settings.data["passthrough_max_fps"] = self.performance_fps.currentData()
        self.settings.data["decode_max_side"] = self.performance_output_size.currentData()
        self.settings.data["subtitle_enable"] = self.subtitle_enable.isChecked()
        self.settings.save()

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

    def update_video_dirs_summary(self) -> None:
        roots = build_media_roots(parse_video_dirs("|".join(self.settings.video_dirs()), UI_ROOT / "videos"))
        names = [root.label for root in roots]
        text = ", ".join(names) if names else self.i18n.t("video_dirs.none")
        self.video_dirs_label.setText(text)
        self.video_dirs_label.setToolTip("|".join(str(root.path) for root in roots))

    def _update_enabled(self) -> None:
        self.bg_color.setVisible(self.green_mode.isChecked())
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
        self.performance_content.setVisible(expanded)
        self.performance_header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._update_performance_config_title()
        self.performance_group.updateGeometry()
        self._adjust_window()

    def _update_quick_config_title(self) -> None:
        key = "group.quick_config" if self.config_header.isChecked() else "group.quick_config_short"
        self.config_header.setText(self.i18n.t(key))

    def _update_performance_config_title(self) -> None:
        key = "group.performance_config" if self.performance_header.isChecked() else "group.performance_config_short"
        self.performance_header.setText(self.i18n.t(key))

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
            self.log_toggle_label,
            self.performance_quality_label,
            self.performance_fps_label,
            self.performance_output_size_label,
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
        self._update_quick_config_title()
        self._update_performance_config_title()
        self.video_dirs_manage_button.setToolTip(self.i18n.t("button.manage"))
        self.video_dirs_title.setText(self.i18n.t("video_dirs.label"))
        self.green_mode.setText("")
        self.green_mode_label.setText(self.i18n.t("mode.green"))
        self.alpha_mode.setText("")
        self.alpha_mode_label.setText(self.i18n.t("mode.alpha"))
        self.subtitle_enable.setText("")
        self.subtitle_enable_label.setText(self.i18n.t("subtitle.enable"))
        self.subtitle_style_button.setText(self.i18n.t("subtitle.style_config"))
        self.player_support_button.setToolTip(self.i18n.t("player_support.window_title"))
        self.log_toggle.setText("")
        self.log_toggle_label.setText(self.i18n.t("log.show"))
        self.debug_toggle.setText("")
        self.debug_toggle_label.setText(self.i18n.t("log.debug"))
        self.performance_quality_label.setText(self.i18n.t("performance.quality_speed"))
        self.performance_fps_label.setText(self.i18n.t("performance.output_fps"))
        self.performance_output_size_label.setText(self.i18n.t("performance.output_size"))
        self.performance_fps.setItemText(0, self.i18n.t("performance.output_fps_unlimited"))
        for i, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium")):
            self.performance_quality.setItemText(i, self.i18n.t(key))
        self.performance_output_size.setItemText(0, self.i18n.t("performance.output_size_original"))
        self.performance_output_size.setItemText(1, self.i18n.t("performance.output_size_4k"))
        self.performance_output_size.setItemText(2, self.i18n.t("performance.output_size_8k"))
        self.update_video_dirs_summary()
        for i, key in enumerate(("bg.neutral_gray", "bg.light_gray", "bg.soft_green", "bg.soft_blue")):
            self.bg_color.setItemText(i, self.i18n.t(key))
        self._sync_quick_label_widths()
