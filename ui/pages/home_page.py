from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from media_library import build_media_roots, parse_video_dirs
from ui.dialogs.video_dirs_dialog import VideoDirsDialog
from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.resources import SWITCH_OFF_IMAGE_PATH, SWITCH_ON_IMAGE_PATH
from ui.settings import ROOT as UI_ROOT


SWITCH_OFF_IMAGE = SWITCH_OFF_IMAGE_PATH.as_posix()
SWITCH_ON_IMAGE = SWITCH_ON_IMAGE_PATH.as_posix()
HOME_COMPACT_WIDTH = 560
HOME_LOG_WIDTH = 380
HOME_HEIGHT = 508
CONFIG_ROW_HEIGHT = 34
SERVER_ICON_SIZE = 22
PROJECT_URL = "https://github.com/zerochocobo/VR-Video-Passthrough-Server"
PROJECT_LINK_HEIGHT = 28


def _retain_size_when_hidden(widget: QWidget) -> None:
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


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
        self.video_dirs_manage_button = QPushButton()
        self.video_dirs_manage_button.setFixedWidth(86)

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
        idx = self.bg_color.findData(settings.data.get("background_color", "808080"))
        self.bg_color.setCurrentIndex(max(0, idx))

        self.subtitle_enable = QCheckBox()
        self.subtitle_enable.setChecked(bool(settings.data.get("subtitle_enable")))
        _apply_switch_style(self.subtitle_enable)
        self.subtitle_enable_label = QLabel()
        self.subtitle_style_button = QPushButton()
        self.subtitle_style_button.setFixedWidth(170)
        _retain_size_when_hidden(self.subtitle_style_button)
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

        group = QGroupBox()
        group.setObjectName("QuickConfig")
        group.setStyleSheet(
            "QGroupBox#QuickConfig {"
            "border: 1px solid #a9b0ba; border-radius: 6px; margin-top: 10px; padding: 8px 8px 6px 8px;"
            "font-size: 10pt; font-weight: 600;"
            "}"
            "QGroupBox#QuickConfig::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QGroupBox#QuickConfig QCheckBox, QGroupBox#QuickConfig QComboBox, QGroupBox#QuickConfig QPushButton {"
            "font-size: 9pt; padding: 2px 6px;"
            "}"
        )
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 12, 10, 8)
        group_layout.setSpacing(4)
        dirs_row_widget = QWidget()
        dirs_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        dirs_row = QHBoxLayout(dirs_row_widget)
        dirs_row.setContentsMargins(0, 0, 0, 0)
        dirs_row.addWidget(self.video_dirs_title)
        dirs_row.addWidget(self.video_dirs_label, 1)
        dirs_row.addWidget(self.video_dirs_manage_button)
        green_row_widget = QWidget()
        green_row_widget.setFixedHeight(CONFIG_ROW_HEIGHT)
        green_row = QHBoxLayout(green_row_widget)
        green_row.setContentsMargins(0, 0, 0, 0)
        green_row.addWidget(self.green_mode_label)
        green_row.addWidget(self.green_mode)
        green_row.addWidget(self.bg_color)
        green_row.addStretch(1)
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
        subtitle_row.addWidget(self.subtitle_style_button)
        subtitle_row.addStretch(1)
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
        for label in (
            self.video_dirs_title,
            self.green_mode_label,
            self.alpha_mode_label,
            self.subtitle_enable_label,
            self.log_toggle_label,
        ):
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        left_panel = QWidget()
        left_panel.setFixedWidth(HOME_COMPACT_WIDTH)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 16, 20, 16)
        left_layout.setSpacing(12)
        left_layout.addLayout(title_box)
        left_layout.addLayout(buttons)
        left_layout.addWidget(group)
        left_layout.addStretch(1)
        left_layout.addWidget(self.project_link)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(left_panel)
        layout.addWidget(self.log)
        self.config_group = group
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
        return QSize(width, HOME_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _bind_settings(self) -> None:
        self.green_mode.toggled.connect(self._save)
        self.alpha_mode.toggled.connect(self._save)
        self.bg_color.currentIndexChanged.connect(self._save)
        self.subtitle_enable.toggled.connect(self._save)
        self.green_mode.toggled.connect(self._update_enabled)
        self.subtitle_enable.toggled.connect(self._update_enabled)
        self.log_toggle.toggled.connect(self._update_enabled)
        self.video_dirs_manage_button.clicked.connect(self.manage_video_dirs)
        self._update_enabled()
        self.update_video_dirs_summary()

    def _save(self) -> None:
        self.settings.data["mode_green"] = self.green_mode.isChecked()
        self.settings.data["mode_alpha"] = self.alpha_mode.isChecked()
        self.settings.data["background_color"] = self.bg_color.currentData()
        self.settings.data["subtitle_enable"] = self.subtitle_enable.isChecked()
        self.settings.save()

    def manage_video_dirs(self) -> None:
        dialog = VideoDirsDialog(self.i18n, self.settings.video_dirs(), self)
        if dialog.exec() != VideoDirsDialog.DialogCode.Accepted:
            return
        self.settings.set_video_dirs(dialog.directories())
        self.settings.save()
        self.update_video_dirs_summary()

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

    def _adjust_window(self) -> None:
        window = self.window()
        if window is not None:
            width = HOME_COMPACT_WIDTH + (HOME_LOG_WIDTH if self.log_toggle.isChecked() else 0)
            self.setMinimumWidth(width)
            self.setMaximumWidth(width)
            self.resize(width, self.height())
            self.updateGeometry()
            self.layout().activate()
            window.setMinimumWidth(width)
            window.setMaximumWidth(width)
            window.setMinimumHeight(HOME_HEIGHT)
            window.setMaximumHeight(HOME_HEIGHT)
            window.resize(width, HOME_HEIGHT)

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
        self.config_group.setTitle(self.i18n.t("group.quick_config"))
        self.video_dirs_manage_button.setText(self.i18n.t("button.manage"))
        self.video_dirs_title.setText(self.i18n.t("video_dirs.label"))
        self.green_mode.setText("")
        self.green_mode_label.setText(self.i18n.t("mode.green"))
        self.alpha_mode.setText("")
        self.alpha_mode_label.setText(self.i18n.t("mode.alpha"))
        self.subtitle_enable.setText("")
        self.subtitle_enable_label.setText(self.i18n.t("subtitle.enable"))
        self.subtitle_style_button.setText(self.i18n.t("subtitle.style_page"))
        self.log_toggle.setText("")
        self.log_toggle_label.setText(self.i18n.t("log.show"))
        self.debug_toggle.setText("")
        self.debug_toggle_label.setText(self.i18n.t("log.debug"))
        self.update_video_dirs_summary()
        for i, key in enumerate(("bg.neutral_gray", "bg.light_gray", "bg.soft_green", "bg.soft_blue")):
            self.bg_color.setItemText(i, self.i18n.t(key))
        self._sync_quick_label_widths()
