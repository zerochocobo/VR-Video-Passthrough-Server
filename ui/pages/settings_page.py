"""Settings page: media library, language, performance/TRT, feature debug gates."""
from __future__ import annotations

from PySide6.QtCore import QFileSystemWatcher, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from media_library import build_media_roots, parse_video_dirs
from ui import theme
from ui.dialogs.video_dirs_dialog import VideoDirsDialog
from ui.icons import line_icon, question_icon
from ui.resources import SWITCH_OFF_IMAGE_PATH, SWITCH_ON_IMAGE_PATH
from ui.settings import (
    DEFAULT_HTTP_PORT,
    DEFAULT_SERVER_NAME,
    ROOT as UI_ROOT,
    quality_speed_value,
)
from ui.widgets.trt_cache_dialog import TensorRTConfigDialog
from utils.trt_manifest import cache_artifact_status as cache_status, manifest_path

ROW_HEIGHT = 34


def _int_setting(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _switch(checked: bool) -> QCheckBox:
    box = QCheckBox()
    box.setChecked(checked)
    box.setCursor(Qt.CursorShape.PointingHandCursor)
    box.setStyleSheet(
        "QCheckBox { spacing: 8px; background: transparent; }"
        "QCheckBox::indicator { width: 38px; height: 20px; }"
        f"QCheckBox::indicator:unchecked {{ image: url({SWITCH_OFF_IMAGE_PATH.as_posix()}); }}"
        f"QCheckBox::indicator:checked {{ image: url({SWITCH_ON_IMAGE_PATH.as_posix()}); }}"
    )
    return box


def _icon_button(icon) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(30, 30)
    return button


class SettingsGroup(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("SettingsGroup")
        self.setStyleSheet(
            "QFrame#SettingsGroup {"
            f"background: {theme.CARD_BG}; border: 1px solid {theme.CARD_BORDER}; border-radius: 12px;"
            "}"
        )
        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {theme.TEXT_PRIMARY}; background: transparent;"
        )
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 12)
        self.body.setSpacing(4)
        self.body.addWidget(self.title_label)

    def add_row(self, *widgets, stretch_index: int | None = None) -> QHBoxLayout:
        row_widget = QWidget()
        row_widget.setFixedHeight(ROW_HEIGHT)
        row_widget.setStyleSheet("background: transparent;")
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        for index, widget in enumerate(widgets):
            if widget is None:
                row.addStretch(1)
            else:
                row.addWidget(widget, 1 if index == stretch_index else 0)
        self.body.addWidget(row_widget)
        return row


class SettingsPage(QWidget):
    rm_card_visibility_changed = Signal(bool)
    face_beauty_card_visibility_changed = Signal(bool)
    gpu_cache_repair_requested = Signal()

    def __init__(self, i18n, settings) -> None:
        super().__init__()
        self.i18n = i18n
        self.settings = settings

        self.title = QLabel()
        self.title.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {theme.TEXT_PRIMARY}; background: transparent;"
        )

        # Media library group.
        self.library_group = SettingsGroup()
        self.video_dirs_label = QLabel()
        self.video_dirs_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.video_dirs_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent; font-size: 9pt;")
        self.video_dirs_manage_button = _icon_button(line_icon("folder", "#4f5965"))
        self.video_dirs_manage_button.clicked.connect(self._manage_video_dirs)
        self.library_group.add_row(self.video_dirs_label, None, self.video_dirs_manage_button, stretch_index=0)

        # DLNA server group: friendly name + HTTP port.
        self.dlna_group = SettingsGroup()
        self.dlna_note = QLabel()
        self.dlna_note.setWordWrap(True)
        self.dlna_note.setStyleSheet(f"color: {theme.TEXT_FAINT}; background: transparent; font-size: 8.5pt;")
        self.server_name_label = QLabel()
        self.server_name = QLineEdit()
        self.server_name.setPlaceholderText(DEFAULT_SERVER_NAME)
        self.server_name.setMaxLength(64)
        self.server_name.setFixedWidth(260)
        self.server_name.setText(settings.server_name())
        self.http_port_label = QLabel()
        self.http_port = QSpinBox()
        self.http_port.setRange(1024, 65535)
        self.http_port.setValue(settings.http_port())
        self.http_port.setFixedWidth(110)
        self.dlna_all_videos_label = QLabel()
        self.dlna_all_videos = _switch(bool(settings.data.get("dlna_all_videos_enabled")))
        self.dlna_save_button = QPushButton()
        self.dlna_save_button.setEnabled(False)
        self.dlna_saved_label = QLabel()
        self.dlna_saved_label.setStyleSheet(f"color: {theme.GREEN}; background: transparent; font-size: 9pt;")
        self._dlna_saved_timer = QTimer(self)
        self._dlna_saved_timer.setSingleShot(True)
        self._dlna_saved_timer.setInterval(4000)
        self._dlna_saved_timer.timeout.connect(self.dlna_saved_label.clear)
        self.dlna_group.body.addWidget(self.dlna_note)
        self.dlna_group.add_row(self.server_name_label, self.server_name, None)
        self.dlna_group.add_row(self.http_port_label, self.http_port, None)
        self.dlna_group.add_row(self.dlna_all_videos_label, self.dlna_all_videos, None)
        self.dlna_group.add_row(self.dlna_save_button, self.dlna_saved_label, None)

        # General group: language.
        self.general_group = SettingsGroup()
        self.language_label = QLabel()
        self.language = QComboBox()
        self.language.addItems(["中文", "English", "日本語"])
        self.language.setFixedWidth(150)
        self.general_group.add_row(self.language_label, self.language, None)

        # Performance group.
        self.performance_group = SettingsGroup()
        self.performance_quality_label = QLabel()
        self.performance_quality = QComboBox()
        for value in ("ultrafast", "medium"):
            self.performance_quality.addItem("", value)
        self.performance_quality.setFixedWidth(150)
        idx = self.performance_quality.findData(quality_speed_value(settings.data.get("quality_speed")))
        self.performance_quality.setCurrentIndex(max(0, idx))

        self.performance_fps_label = QLabel()
        self.performance_fps = QComboBox()
        self.performance_fps.addItem("", 0)
        for value in (20, 30, 40, 50, 60):
            self.performance_fps.addItem(str(value), value)
        self.performance_fps.setFixedWidth(120)
        idx = self.performance_fps.findData(_int_setting(settings.data.get("passthrough_max_fps"), 0))
        self.performance_fps.setCurrentIndex(max(0, idx))
        self.performance_fps_help = _icon_button(question_icon())
        self.performance_fps_help.clicked.connect(self._show_fps_help)

        self.performance_output_size_label = QLabel()
        self.performance_output_size = QComboBox()
        self.performance_output_size.addItem("", 0)
        self.performance_output_size.addItem("", 4096)
        self.performance_output_size.addItem("", 8192)
        self.performance_output_size.setFixedWidth(150)
        idx = self.performance_output_size.findData(_int_setting(settings.data.get("decode_max_side"), 4096))
        self.performance_output_size.setCurrentIndex(max(0, idx))

        self.trt_enabled_label = QLabel()
        self.trt_enabled = _switch(str(settings.data.get("inference_backend") or "cuda").lower() == "tensorrt")
        self.trt_configure_button = QPushButton()
        self.trt_status_label = QLabel()
        self.trt_status_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")
        self.trt_cache_watcher = QFileSystemWatcher(self)
        self._runtime_provider_kind = ""
        self.trt_cache_watcher.directoryChanged.connect(lambda _path: self._update_trt_state())
        self.trt_cache_watcher.fileChanged.connect(lambda _path: self._update_trt_state())

        self.performance_group.add_row(self.performance_quality_label, self.performance_quality, None)
        self.performance_group.add_row(self.performance_fps_label, self.performance_fps, None, self.performance_fps_help)
        self.performance_group.add_row(self.performance_output_size_label, self.performance_output_size, None)
        self.performance_group.add_row(
            self.trt_enabled_label, self.trt_enabled, self.trt_configure_button, self.trt_status_label, None
        )

        # Troubleshooting group: filesystem-only GPU cache repair.  MainWindow
        # owns process coordination and the confirmation/restart flow.
        self.repair_group = SettingsGroup()
        self.gpu_cache_repair_note = QLabel()
        self.gpu_cache_repair_note.setWordWrap(True)
        self.gpu_cache_repair_note.setStyleSheet(
            f"color: {theme.TEXT_FAINT}; background: transparent; font-size: 8.5pt;"
        )
        self.gpu_cache_repair_button = QPushButton()
        self.repair_group.body.addWidget(self.gpu_cache_repair_note)
        self.repair_group.add_row(self.gpu_cache_repair_button, None)

        # Feature debug group.
        self.debug_group = SettingsGroup()
        self.debug_note = QLabel()
        self.debug_note.setWordWrap(True)
        self.debug_note.setStyleSheet(f"color: {theme.TEXT_FAINT}; background: transparent; font-size: 8.5pt;")
        self.rm_card_label = QLabel()
        self.rm_card_switch = _switch(bool(settings.data.get("rm_card_visible")))
        self.face_beauty_card_label = QLabel()
        self.face_beauty_card_switch = _switch(bool(settings.data.get("face_beauty_card_visible")))
        self.debug_group.body.addWidget(self.debug_note)
        self.debug_group.add_row(self.rm_card_label, self.rm_card_switch, None)
        self.debug_group.add_row(self.face_beauty_card_label, self.face_beauty_card_switch, None)

        scroll_body = QWidget()
        scroll_body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(scroll_body)
        body_layout.setContentsMargins(0, 0, 6, 0)
        body_layout.setSpacing(10)
        body_layout.addWidget(self.library_group)
        body_layout.addWidget(self.dlna_group)
        body_layout.addWidget(self.general_group)
        body_layout.addWidget(self.performance_group)
        body_layout.addWidget(self.repair_group)
        body_layout.addWidget(self.debug_group)
        body_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.setWidget(scroll_body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 12, 10)
        layout.setSpacing(10)
        layout.addWidget(self.title)
        layout.addWidget(scroll, 1)

        self.server_name.textChanged.connect(self._update_dlna_dirty)
        self.http_port.valueChanged.connect(self._update_dlna_dirty)
        self.dlna_all_videos.toggled.connect(self._save_dlna_all_videos)
        self.server_name.editingFinished.connect(self._save_dlna)
        self.http_port.editingFinished.connect(self._save_dlna)
        self.dlna_save_button.clicked.connect(self._save_dlna)
        self.performance_quality.currentIndexChanged.connect(self._save)
        self.performance_fps.currentIndexChanged.connect(self._save)
        self.performance_output_size.currentIndexChanged.connect(self._save)
        self.trt_enabled.toggled.connect(self._save)
        self.trt_configure_button.clicked.connect(self._show_trt_config)
        self.gpu_cache_repair_button.clicked.connect(lambda _checked=False: self.gpu_cache_repair_requested.emit())
        self.rm_card_switch.toggled.connect(self._toggle_rm_card)
        self.face_beauty_card_switch.toggled.connect(self._toggle_face_beauty_card)

        self._update_trt_state()
        self.retranslate()

    # ---- persistence ----

    def _save(self) -> None:
        self.settings.data["quality_speed"] = self.performance_quality.currentData()
        self.settings.data["passthrough_max_fps"] = self.performance_fps.currentData()
        self.settings.data["decode_max_side"] = self.performance_output_size.currentData()
        if self.trt_enabled.isEnabled():
            self.settings.data["inference_backend"] = "tensorrt" if self.trt_enabled.isChecked() else "cuda"
        self.settings.save()

    def _dlna_dirty(self) -> bool:
        return (
            self.server_name.text().strip() != self.settings.server_name()
            or int(self.http_port.value()) != self.settings.http_port()
        )

    def _update_dlna_dirty(self) -> None:
        dirty = self._dlna_dirty()
        self.dlna_save_button.setEnabled(dirty)
        if dirty:
            self._dlna_saved_timer.stop()
            self.dlna_saved_label.clear()

    def _save_dlna(self) -> None:
        if not self._dlna_dirty():
            return
        self.settings.data["server_name"] = self.server_name.text().strip()
        self.settings.data["http_port"] = int(self.http_port.value())
        self.settings.save()
        self.dlna_save_button.setEnabled(False)
        self.dlna_saved_label.setText(self.i18n.t("settings.dlna_saved"))
        self._dlna_saved_timer.start()

    def _save_dlna_all_videos(self, checked: bool) -> None:
        self.settings.data["dlna_all_videos_enabled"] = bool(checked)
        self.settings.save()

    def _toggle_rm_card(self, checked: bool) -> None:
        self.settings.data["rm_card_visible"] = bool(checked)
        self.settings.save()
        self.rm_card_visibility_changed.emit(bool(checked))

    def _toggle_face_beauty_card(self, checked: bool) -> None:
        self.settings.data["face_beauty_card_visible"] = bool(checked)
        if not checked:
            self.settings.data["face_beauty_enabled"] = False
        self.settings.save()
        self.face_beauty_card_visibility_changed.emit(bool(checked))

    def sync_from_settings(self) -> None:
        value = quality_speed_value(self.settings.data.get("quality_speed"))
        idx = self.performance_quality.findData(value)
        if idx >= 0 and self.performance_quality.currentIndex() != idx:
            self.performance_quality.blockSignals(True)
            self.performance_quality.setCurrentIndex(idx)
            self.performance_quality.blockSignals(False)
        self.update_video_dirs_summary()

    # ---- dialogs ----

    def _manage_video_dirs(self) -> None:
        dialog = VideoDirsDialog(self.i18n, self.settings.video_dirs(), self)
        if dialog.exec() != VideoDirsDialog.DialogCode.Accepted:
            return
        self.settings.set_video_dirs(dialog.directories())
        self.settings.save()
        self.update_video_dirs_summary()

    def _show_fps_help(self) -> None:
        QMessageBox.information(self, self.i18n.t("performance.output_fps"), self.i18n.t("performance.output_fps_help"))

    def _show_trt_config(self) -> None:
        dialog = TensorRTConfigDialog(self.i18n, self, scope="realtime")
        dialog.exec()
        if self._trt_status() == "ready":
            self.settings.data["inference_backend"] = "tensorrt"
            self.settings.save()
        self._update_trt_state()

    # ---- TRT state ----

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
        else:
            self.trt_enabled.setChecked(
                str(self.settings.data.get("inference_backend") or "cuda").lower() == "tensorrt"
            )
        self.trt_enabled.blockSignals(False)
        status_text = self.i18n.t("trt.status_" + status)
        if self._runtime_provider_kind:
            runtime_key = {
                "trt": "trt.runtime_trt",
                "cuda": "trt.runtime_cuda",
                "cpu": "trt.runtime_cpu",
            }.get(self._runtime_provider_kind, "trt.runtime_cpu")
            status_text = f"{status_text} · {self.i18n.t(runtime_key)}"
        self.trt_status_label.setText(status_text)
        self.trt_enabled.setToolTip("" if ready else self.i18n.t("trt.build_first_tooltip"))

    def set_runtime_provider_kind(self, provider_kind: str) -> None:
        value = str(provider_kind or "").strip().lower()
        if value == self._runtime_provider_kind:
            return
        self._runtime_provider_kind = value
        self._update_trt_state()

    # ---- misc ----

    def update_video_dirs_summary(self) -> None:
        roots = build_media_roots(parse_video_dirs("|".join(self.settings.video_dirs()), UI_ROOT / "videos"))
        names = [root.label for root in roots]
        text = ", ".join(names) if names else self.i18n.t("video_dirs.none")
        self.video_dirs_label.setText(text)
        self.video_dirs_label.setToolTip("|".join(str(root.path) for root in roots))

    def retranslate(self) -> None:
        self.title.setText(self.i18n.t("nav.settings"))
        self.library_group.title_label.setText(self.i18n.t("video_dirs.label"))
        self.dlna_group.title_label.setText(self.i18n.t("settings.dlna"))
        self.dlna_note.setText(self.i18n.t("settings.dlna_note"))
        self.server_name_label.setText(self.i18n.t("settings.server_name"))
        self.http_port_label.setText(self.i18n.t("settings.http_port"))
        self.dlna_all_videos_label.setText(self.i18n.t("settings.dlna_all_videos"))
        self.dlna_save_button.setText(self.i18n.t("button.save"))
        self.general_group.title_label.setText(self.i18n.t("settings.general"))
        self.language_label.setText(self.i18n.t("settings.language"))
        self.performance_group.title_label.setText(self.i18n.t("group.performance_config_short"))
        self.performance_quality_label.setText(self.i18n.t("performance.quality_speed"))
        self.performance_fps_label.setText(self.i18n.t("performance.output_fps"))
        self.performance_fps_help.setToolTip(self.i18n.t("performance.output_fps_help"))
        self.performance_output_size_label.setText(self.i18n.t("performance.output_size"))
        self.trt_enabled_label.setText(self.i18n.t("trt.row_label"))
        self.trt_configure_button.setText(self.i18n.t("trt.configure"))
        self.repair_group.title_label.setText(self.i18n.t("gpu_repair.group"))
        self.gpu_cache_repair_note.setText(self.i18n.t("gpu_repair.settings_note"))
        self.gpu_cache_repair_button.setText(self.i18n.t("gpu_repair.settings_button"))
        self.debug_group.title_label.setText(self.i18n.t("settings.feature_debug"))
        self.debug_note.setText(self.i18n.t("settings.feature_debug_note"))
        self.rm_card_label.setText(self.i18n.t("rm.enabled"))
        self.face_beauty_card_label.setText(self.i18n.t("beauty.entry_visible"))
        self.performance_fps.setItemText(0, self.i18n.t("performance.output_fps_unlimited"))
        for i, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium")):
            self.performance_quality.setItemText(i, self.i18n.t(key))
        self.performance_output_size.setItemText(0, self.i18n.t("performance.output_size_original"))
        self.performance_output_size.setItemText(1, self.i18n.t("performance.output_size_4k"))
        self.performance_output_size.setItemText(2, self.i18n.t("performance.output_size_8k"))
        self.video_dirs_manage_button.setToolTip(self.i18n.t("button.manage"))
        self.update_video_dirs_summary()
        self._update_trt_state()
