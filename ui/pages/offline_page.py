from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.page_icons import BACK_ICON_SIZE, back_icon
from ui.resources import SWITCH_OFF_IMAGE_PATH, SWITCH_ON_IMAGE_PATH
from ui.settings import quality_speed_preset, quality_speed_value
from ui.widgets.trt_cache_dialog import TensorRTConfigDialog
from utils.trt_manifest import TRT_MODEL_MATANYONE2, TRT_MODEL_RVM, cache_status, manifest_path


OFFLINE_LABEL_WIDTH = 132
ACTION_ICON_SIZE = 20
HELP_ICON_SIZE = 20
SWITCH_OFF_IMAGE = SWITCH_OFF_IMAGE_PATH.as_posix()
SWITCH_ON_IMAGE = SWITCH_ON_IMAGE_PATH.as_posix()


def _action_icon(kind: str) -> QIcon:
    pixmap = QPixmap(ACTION_ICON_SIZE, ACTION_ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    if kind == "stop":
        painter.setBrush(QColor("#D93025"))
        painter.drawRoundedRect(4, 4, 12, 12, 2, 2)
    else:
        painter.setBrush(QColor("#18A058"))
        painter.drawPolygon([QPoint(6, 4), QPoint(6, 16), QPoint(16, 10)])
    painter.end()
    return QIcon(pixmap)


def _help_icon() -> QIcon:
    pixmap = QPixmap(HELP_ICON_SIZE, HELP_ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor("#4f5965"), 2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(2, 2, HELP_ICON_SIZE - 4, HELP_ICON_SIZE - 4)
    font = QFont()
    font.setBold(True)
    font.setPointSize(11)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")
    painter.end()
    return QIcon(pixmap)


def _help_button() -> QPushButton:
    button = QPushButton()
    button.setIcon(_help_icon())
    button.setIconSize(QSize(HELP_ICON_SIZE, HELP_ICON_SIZE))
    button.setFixedWidth(32)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return button


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


def _label() -> QLabel:
    label = QLabel()
    label.setFixedWidth(OFFLINE_LABEL_WIDTH)
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return label


def _fit_combo(combo: QComboBox) -> QComboBox:
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return combo


class OfflinePage(QWidget):
    def __init__(self, i18n, settings, process) -> None:
        super().__init__()
        self.setObjectName("OfflinePage")
        self.setStyleSheet(
            "QWidget#OfflinePage, QWidget#OfflinePage QLabel, QWidget#OfflinePage QCheckBox { font-size: 9pt; }"
            "QWidget#OfflinePage QPushButton, QWidget#OfflinePage QLineEdit, QWidget#OfflinePage QComboBox, "
            "QWidget#OfflinePage QTextEdit, QWidget#OfflinePage QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#OfflinePage QLabel#OfflinePageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n = i18n
        self.settings = settings
        self.process = process
        self.title_label = QLabel()
        self.title_label.setObjectName("OfflinePageTitle")
        self.back_button = QPushButton()
        self.back_button.setIcon(back_icon())
        self.back_button.setIconSize(QSize(BACK_ICON_SIZE, BACK_ICON_SIZE))
        self.back_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.tabs = QTabWidget()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(UI_LOG_MAX_BLOCKS)
        self.start_single = self._action_button("start")
        self.stop_single = self._action_button("stop")
        self.start_batch = self._action_button("start")
        self.stop_batch = self._action_button("stop")
        self.stop_single.clicked.connect(process.stop)
        self.stop_batch.clicked.connect(process.stop)
        process.output.connect(self.append_log)
        process.state_changed.connect(self.set_running)
        self._single_tab()
        self._batch_tab()
        header = QHBoxLayout()
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.back_button)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.tabs)
        layout.addWidget(self.log, 1)
        self.retranslate()
        self.set_running(False)

    def _action_button(self, kind: str) -> QPushButton:
        button = QPushButton()
        button.setIcon(_action_icon(kind))
        button.setIconSize(QSize(ACTION_ICON_SIZE, ACTION_ICON_SIZE))
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return button

    def _engine_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", "rvm_fast")
        combo.addItem("", "matanyone2")
        return combo

    def _recognition_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", "yoloworld_efficientsam")
        combo.addItem("", "sam3")
        return combo

    def _mode_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", "green")
        combo.addItem("", "alpha")
        return combo

    def _duration_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", 15.0)
        combo.addItem("", 30.0)
        combo.addItem("", 60.0)
        combo.addItem("", "custom")
        combo.addItem("", 0.0)
        return combo

    def _quality_speed_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in ("ultrafast", "medium", "veryslow"):
            combo.addItem("", value)
        idx = combo.findData(quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium"))
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_quality_speed)
        return combo

    def _performance_row(self, combo: QComboBox) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(combo)
        row.addStretch(1)
        return row

    def _trt_cache_row(self, scope: str) -> QHBoxLayout:
        enabled = QCheckBox()
        enabled.setText("")
        _apply_switch_style(enabled)
        button = QPushButton()
        status = QLabel()
        status.setStyleSheet("color: #5f6368;")
        watcher = QFileSystemWatcher(self)
        watcher.directoryChanged.connect(lambda _path: self._update_trt_cache_rows())
        watcher.fileChanged.connect(lambda _path: self._update_trt_cache_rows())
        enabled.toggled.connect(lambda checked, item=scope: self._save_trt_enabled(item, checked))
        button.clicked.connect(lambda: self.show_trt_config(scope))
        setattr(self, f"{scope}_trt_enabled", enabled)
        setattr(self, f"{scope}_trt_configure_button", button)
        setattr(self, f"{scope}_trt_status_label", status)
        setattr(self, f"{scope}_trt_cache_watcher", watcher)
        row = QHBoxLayout()
        row.addWidget(enabled)
        row.addWidget(button)
        row.addWidget(status)
        row.addStretch(1)
        return row

    def _time_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.single_start = QLineEdit("00:00:00")
        self.single_start.setFixedWidth(90)
        self.single_duration = self._duration_combo()
        self.single_custom_minutes_label = QLabel()
        self.single_custom_minutes = QLineEdit("5")
        self.single_custom_minutes.setFixedWidth(48)
        self.single_duration.currentIndexChanged.connect(self._update_custom_duration_visibility)
        row.addWidget(self.single_start)
        row.addSpacing(12)
        row.addWidget(self.single_duration)
        row.addWidget(self.single_custom_minutes_label)
        row.addWidget(self.single_custom_minutes)
        row.addStretch(1)
        return row

    def _single_tab(self) -> None:
        page = QWidget()
        self.single_video = QLineEdit()
        browse_video = QPushButton("...")
        browse_video.clicked.connect(lambda: self._browse_file(self.single_video))
        self.single_out_dir = QLineEdit()
        browse_out = QPushButton("...")
        browse_out.clicked.connect(lambda: self._browse_dir(self.single_out_dir))
        self.single_mode = self._mode_combo()
        self.single_engine = self._engine_combo()
        self.single_recognition = self._recognition_combo()
        self.single_matanyone_help = _help_button()
        self.single_sam3_prompt_button = QPushButton()
        self.single_sam3_prompt_label = QLabel()
        self.single_quality_speed = self._quality_speed_combo()
        self.single_skip = QCheckBox()
        self.single_skip.setChecked(True)
        self.start_single.clicked.connect(self.run_single)
        self.single_engine.currentIndexChanged.connect(self._update_matanyone_help_visibility)
        self.single_engine.currentIndexChanged.connect(self._update_recognition_visibility)
        self.single_engine.currentIndexChanged.connect(self._update_trt_cache_rows)
        self.single_recognition.currentIndexChanged.connect(self._update_recognition_visibility)
        self.single_matanyone_help.clicked.connect(self.show_matanyone_help)
        self.single_sam3_prompt_button.clicked.connect(self.show_sam3_prompt_dialog)
        row_video = QHBoxLayout()
        row_video.addWidget(self.single_video)
        row_video.addWidget(browse_video)
        row_out = QHBoxLayout()
        row_out.addWidget(self.single_out_dir)
        row_out.addWidget(browse_out)
        actions = QHBoxLayout()
        actions.addWidget(self.start_single)
        actions.addWidget(self.stop_single)
        actions.addStretch(1)
        grid = QGridLayout(page)
        grid.setColumnMinimumWidth(0, OFFLINE_LABEL_WIDTH)
        grid.setColumnStretch(1, 1)
        self.single_labels = {key: _label() for key in ("video", "output", "mode", "engine", "recognition", "trt", "performance", "time")}
        grid.addWidget(self.single_labels["video"], 0, 0)
        grid.addLayout(row_video, 0, 1)
        grid.addWidget(self.single_labels["output"], 1, 0)
        grid.addLayout(row_out, 1, 1)
        grid.addWidget(self.single_labels["mode"], 2, 0)
        grid.addWidget(self.single_mode, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["engine"], 3, 0)
        single_engine_row = QHBoxLayout()
        single_engine_row.addWidget(self.single_engine)
        single_engine_row.addStretch(1)
        grid.addLayout(single_engine_row, 3, 1)
        grid.addWidget(self.single_labels["recognition"], 4, 0)
        single_recognition_row = QHBoxLayout()
        single_recognition_row.addWidget(self.single_recognition)
        single_recognition_row.addWidget(self.single_matanyone_help)
        single_recognition_row.addWidget(self.single_sam3_prompt_button)
        single_recognition_row.addWidget(self.single_sam3_prompt_label)
        single_recognition_row.addStretch(1)
        grid.addLayout(single_recognition_row, 4, 1)
        grid.addWidget(self.single_labels["trt"], 5, 0)
        grid.addLayout(self._trt_cache_row("single"), 5, 1)
        grid.addWidget(self.single_labels["performance"], 6, 0)
        grid.addLayout(self._performance_row(self.single_quality_speed), 6, 1)
        grid.addWidget(self.single_labels["time"], 7, 0)
        grid.addLayout(self._time_row(), 7, 1)
        grid.addWidget(self.single_skip, 8, 1)
        grid.addLayout(actions, 9, 1)
        self.tabs.addTab(page, "")
        self._update_custom_duration_visibility()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit()
        browse_dir = QPushButton("...")
        browse_dir.clicked.connect(lambda: self._browse_dir(self.batch_dir))
        self.batch_mode = self._mode_combo()
        self.batch_engine = self._engine_combo()
        self.batch_recognition = self._recognition_combo()
        self.batch_matanyone_help = _help_button()
        self.batch_sam3_prompt_button = QPushButton()
        self.batch_sam3_prompt_label = QLabel()
        self.batch_quality_speed = self._quality_speed_combo()
        self.batch_recursive = QCheckBox()
        self.batch_recursive.setChecked(True)
        self.batch_skip = QCheckBox()
        self.batch_skip.setChecked(True)
        self.start_batch.clicked.connect(self.run_batch)
        self.batch_engine.currentIndexChanged.connect(self._update_matanyone_help_visibility)
        self.batch_engine.currentIndexChanged.connect(self._update_recognition_visibility)
        self.batch_engine.currentIndexChanged.connect(self._update_trt_cache_rows)
        self.batch_recognition.currentIndexChanged.connect(self._update_recognition_visibility)
        self.batch_matanyone_help.clicked.connect(self.show_matanyone_help)
        self.batch_sam3_prompt_button.clicked.connect(self.show_sam3_prompt_dialog)
        row_dir = QHBoxLayout()
        row_dir.addWidget(self.batch_dir)
        row_dir.addWidget(browse_dir)
        actions = QHBoxLayout()
        actions.addWidget(self.start_batch)
        actions.addWidget(self.stop_batch)
        actions.addStretch(1)
        grid = QGridLayout(page)
        grid.setColumnMinimumWidth(0, OFFLINE_LABEL_WIDTH)
        grid.setColumnStretch(1, 1)
        self.batch_labels = {key: _label() for key in ("directory", "mode", "engine", "recognition", "trt", "performance")}
        grid.addWidget(self.batch_labels["directory"], 0, 0)
        grid.addLayout(row_dir, 0, 1)
        grid.addWidget(self.batch_labels["mode"], 1, 0)
        grid.addWidget(self.batch_mode, 1, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["engine"], 2, 0)
        batch_engine_row = QHBoxLayout()
        batch_engine_row.addWidget(self.batch_engine)
        batch_engine_row.addStretch(1)
        grid.addLayout(batch_engine_row, 2, 1)
        grid.addWidget(self.batch_labels["recognition"], 3, 0)
        batch_recognition_row = QHBoxLayout()
        batch_recognition_row.addWidget(self.batch_recognition)
        batch_recognition_row.addWidget(self.batch_matanyone_help)
        batch_recognition_row.addWidget(self.batch_sam3_prompt_button)
        batch_recognition_row.addWidget(self.batch_sam3_prompt_label)
        batch_recognition_row.addStretch(1)
        grid.addLayout(batch_recognition_row, 3, 1)
        grid.addWidget(self.batch_labels["trt"], 4, 0)
        grid.addLayout(self._trt_cache_row("batch"), 4, 1)
        grid.addWidget(self.batch_labels["performance"], 5, 0)
        grid.addLayout(self._performance_row(self.batch_quality_speed), 5, 1)
        grid.addWidget(self.batch_recursive, 6, 1)
        grid.addWidget(self.batch_skip, 7, 1)
        grid.addLayout(actions, 8, 1)
        self.tabs.addTab(page, "")

    def _browse_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.i18n.t("file.select_video"), "", "Videos (*.mp4 *.mkv *.mov *.m4v)")
        if path:
            target.setText(path)
            self.single_out_dir.setText(str(Path(path).parent))

    def _browse_save(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getSaveFileName(self, self.i18n.t("file.output_video"), "", "MP4 (*.mp4)")
        if path:
            target.setText(path)

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("file.select_directory"))
        if path:
            target.setText(path)

    def _start_seconds(self) -> float:
        text = self.single_start.text().strip()
        try:
            if ":" not in text:
                return max(0.0, float(text or 0))
            parts = [float(part or 0) for part in text.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            h, m, s = parts[-3:]
            return max(0.0, h * 3600 + m * 60 + s)
        except ValueError:
            return 0.0

    def _duration_seconds(self) -> float:
        value = self.single_duration.currentData()
        if value == "custom":
            try:
                return max(1.0, float(self.single_custom_minutes.text() or 1) * 60.0)
            except ValueError:
                return 60.0
        return float(value or 0.0)

    def _update_custom_duration_visibility(self) -> None:
        visible = self.single_duration.currentData() == "custom"
        self.single_custom_minutes_label.setVisible(visible)
        self.single_custom_minutes.setVisible(visible)

    def _update_matanyone_help_visibility(self) -> None:
        self.single_matanyone_help.setVisible(str(self.single_engine.currentData()) == "matanyone2")
        self.batch_matanyone_help.setVisible(str(self.batch_engine.currentData()) == "matanyone2")

    def _update_recognition_visibility(self) -> None:
        single_visible = str(self.single_engine.currentData()) == "matanyone2"
        batch_visible = str(self.batch_engine.currentData()) == "matanyone2"
        single_sam3_visible = single_visible and str(self.single_recognition.currentData()) == "sam3"
        batch_sam3_visible = batch_visible and str(self.batch_recognition.currentData()) == "sam3"
        self.single_labels["recognition"].setVisible(single_visible)
        self.single_recognition.setVisible(single_visible)
        self.single_sam3_prompt_button.setVisible(single_sam3_visible)
        self.single_sam3_prompt_label.setVisible(single_sam3_visible)
        self.batch_labels["recognition"].setVisible(batch_visible)
        self.batch_recognition.setVisible(batch_visible)
        self.batch_sam3_prompt_button.setVisible(batch_sam3_visible)
        self.batch_sam3_prompt_label.setVisible(batch_sam3_visible)
        self._update_sam3_prompt_labels()
        self._update_matanyone_help_visibility()

    def _trt_model_key_for_scope(self, scope: str) -> str:
        combo = self.single_engine if scope == "single" else self.batch_engine
        return TRT_MODEL_MATANYONE2 if str(combo.currentData()) == "matanyone2" else TRT_MODEL_RVM

    def _trt_setting_key(self, scope: str, model_key: str) -> str:
        model_part = "matanyone2" if model_key == TRT_MODEL_MATANYONE2 else "rvm"
        return f"offline_{scope}_trt_{model_part}_enabled"

    def _offline_trt_enabled(self, scope: str, model_key: str) -> bool:
        return bool(self.settings.data.get(self._trt_setting_key(scope, model_key), True))

    def _save_trt_enabled(self, scope: str, checked: bool) -> None:
        model_key = self._trt_model_key_for_scope(scope)
        self.settings.data[self._trt_setting_key(scope, model_key)] = bool(checked)
        self.settings.save()

    def _trt_status(self, model_key: str) -> str:
        try:
            return cache_status(model_key=model_key)
        except Exception:
            return "failed"

    def _refresh_trt_cache_watcher(self, scope: str, model_key: str) -> None:
        watcher = getattr(self, f"{scope}_trt_cache_watcher", None)
        if not isinstance(watcher, QFileSystemWatcher):
            return
        for path in watcher.files():
            watcher.removePath(path)
        for path in watcher.directories():
            watcher.removePath(path)
        cache_dir = manifest_path(model_key).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        watcher.addPath(str(cache_dir))
        manifest = manifest_path(model_key)
        if manifest.exists():
            watcher.addPath(str(manifest))

    def _update_trt_cache_rows(self, *_args) -> None:
        for scope in ("single", "batch"):
            if not hasattr(self, f"{scope}_trt_status_label"):
                continue
            model_key = self._trt_model_key_for_scope(scope)
            self._refresh_trt_cache_watcher(scope, model_key)
            status = self._trt_status(model_key)
            model_text = self.i18n.t("trt.model_matanyone2" if model_key == TRT_MODEL_MATANYONE2 else "trt.model_rvm")
            switch = getattr(self, f"{scope}_trt_enabled")
            switch.blockSignals(True)
            switch.setEnabled(status == "ready")
            switch.setChecked(status == "ready" and self._offline_trt_enabled(scope, model_key))
            switch.blockSignals(False)
            switch.setToolTip("" if status == "ready" else self.i18n.t("trt.build_first_tooltip"))
            getattr(self, f"{scope}_trt_status_label").setText(f"{model_text}: {self.i18n.t('trt.status_' + status)}")

    def show_trt_config(self, scope: str) -> None:
        model_key = self._trt_model_key_for_scope(scope)
        dialog = TensorRTConfigDialog(self.i18n, self, model_key=model_key)
        dialog.exec()
        self._update_trt_cache_rows()

    def _effective_engine(self, engine_combo: QComboBox, recognition_combo: QComboBox) -> str:
        engine = str(engine_combo.currentData())
        if engine != "matanyone2":
            return engine
        recognition = str(recognition_combo.currentData())
        return "matanyone2_medium" if recognition == "yoloworld_efficientsam" else "matanyone2"

    def _sam3_prompt(self) -> str:
        prompt = str(self.settings.data.get("offline_sam3_prompt") or "").strip()
        return prompt or "person"

    def _update_sam3_prompt_labels(self) -> None:
        prompt = self._sam3_prompt()
        self.single_sam3_prompt_label.setText(prompt)
        self.batch_sam3_prompt_label.setText(prompt)

    def show_matanyone_help(self) -> None:
        QMessageBox.information(
            self,
            self.i18n.t("offline.matanyone_help_title"),
            self.i18n.t("offline.matanyone_help_msg"),
        )

    def show_sam3_prompt_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self.i18n.t("offline.sam3_prompt_title"))
        layout = QVBoxLayout(dialog)
        hint = QLabel(self.i18n.t("offline.sam3_prompt_hint"))
        hint.setWordWrap(True)
        prompt_row = QHBoxLayout()
        prompt_label = QLabel(self.i18n.t("offline.sam3_prompt_label"))
        prompt_edit = QLineEdit(self._sam3_prompt())
        prompt_edit.setMinimumWidth(260)
        prompt_row.addWidget(prompt_label)
        prompt_row.addWidget(prompt_edit, 1)
        buttons = QHBoxLayout()
        save_button = QPushButton(self.i18n.t("button.save"))
        close_button = QPushButton(self.i18n.t("button.close"))
        buttons.addStretch(1)
        buttons.addWidget(save_button)
        buttons.addWidget(close_button)

        def save_prompt() -> None:
            prompt = prompt_edit.text().strip() or "person"
            self.settings.data["offline_sam3_prompt"] = prompt
            self.settings.save()
            self._update_sam3_prompt_labels()
            dialog.accept()

        save_button.clicked.connect(save_prompt)
        close_button.clicked.connect(dialog.reject)
        layout.addWidget(hint)
        layout.addLayout(prompt_row)
        layout.addLayout(buttons)
        dialog.exec()

    def _save_quality_speed(self) -> None:
        sender = self.sender()
        if isinstance(sender, QComboBox):
            value = quality_speed_value(sender.currentData(), "medium")
            self.settings.data["offline_quality_speed"] = value
            for combo in (getattr(self, "single_quality_speed", None), getattr(self, "batch_quality_speed", None)):
                if isinstance(combo, QComboBox) and combo is not sender:
                    idx = combo.findData(value)
                    if idx >= 0 and combo.currentIndex() != idx:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(idx)
                        combo.blockSignals(False)
            self.settings.save()

    def sync_from_settings(self) -> None:
        value = quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium")
        for combo in (getattr(self, "single_quality_speed", None), getattr(self, "batch_quality_speed", None)):
            if isinstance(combo, QComboBox):
                idx = combo.findData(value)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)

    def run_single(self) -> None:
        engine = self._effective_engine(self.single_engine, self.single_recognition)
        args = [
            "single",
            self.single_video.text(),
            "--mode",
            self.single_mode.currentData(),
            "--engine",
            engine,
            "--start",
            str(self._start_seconds()),
            "--duration",
            str(self._duration_seconds()),
            "--skip-frames",
            "0",
        ]
        if self.single_out_dir.text().strip():
            args.extend(["--out-dir", self.single_out_dir.text().strip()])
        if self.single_skip.isChecked():
            args.append("--skip-existing")
        if engine == "matanyone2":
            args.extend(["--sam3-prompt", self._sam3_prompt()])
        self.settings.save()
        env = self.settings.server_env()
        env["PT_DECODE_MAX_SIDE"] = "0"
        env["PT_PASSTHROUGH_PYNV_PRESET"] = quality_speed_preset(self.settings.data.get("offline_quality_speed"), "medium")
        self._apply_offline_trt_env(env, "single", engine)
        self.process.start(args, env)

    def run_batch(self) -> None:
        engine = self._effective_engine(self.batch_engine, self.batch_recognition)
        args = [
            "batch",
            self.batch_dir.text(),
            "--mode",
            self.batch_mode.currentData(),
            "--engine",
            engine,
            "--skip-frames",
            "0",
        ]
        args.append("--recursive" if self.batch_recursive.isChecked() else "--no-recursive")
        if self.batch_skip.isChecked():
            args.append("--skip-existing")
        if engine == "matanyone2":
            args.extend(["--sam3-prompt", self._sam3_prompt()])
        self.settings.save()
        env = self.settings.server_env()
        env["PT_DECODE_MAX_SIDE"] = "0"
        env["PT_PASSTHROUGH_PYNV_PRESET"] = quality_speed_preset(self.settings.data.get("offline_quality_speed"), "medium")
        self._apply_offline_trt_env(env, "batch", engine)
        self.process.start(args, env)

    def _apply_offline_trt_env(self, env: dict[str, str], scope: str, engine: str) -> None:
        model_key = TRT_MODEL_MATANYONE2 if engine in {"matanyone2", "matanyone2_medium"} else TRT_MODEL_RVM
        enabled = self._offline_trt_enabled(scope, model_key)
        env["PT_OFFLINE_RVM_TRT_ENABLE"] = "1" if model_key == TRT_MODEL_RVM and enabled else "0"
        env["PT_OFFLINE_MATANYONE2_TRT_ENABLE"] = "1" if model_key == TRT_MODEL_MATANYONE2 and enabled else "0"

    def set_running(self, running: bool) -> None:
        self.start_single.setEnabled(not running)
        self.start_batch.setEnabled(not running)
        self.stop_single.setEnabled(running)
        self.stop_batch.setEnabled(running)
        self.single_trt_configure_button.setEnabled(not running)
        self.batch_trt_configure_button.setEnabled(not running)
        if not running:
            self._update_trt_cache_rows()
        else:
            self.single_trt_enabled.setEnabled(False)
            self.batch_trt_enabled.setEnabled(False)

    def append_log(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("button.offline"))
        self.back_button.setText(self.i18n.t("button.back"))
        self.tabs.setTabText(0, self.i18n.t("offline.single_tab"))
        self.tabs.setTabText(1, self.i18n.t("offline.batch_tab"))
        for button in (self.start_single, self.start_batch):
            button.setText(self.i18n.t("button.start"))
        for button in (self.stop_single, self.stop_batch):
            button.setText(self.i18n.t("button.stop"))
        self.single_skip.setText(self.i18n.t("offline.skip_existing"))
        self.batch_recursive.setText(self.i18n.t("offline.recursive"))
        self.batch_skip.setText(self.i18n.t("offline.skip_existing"))
        self.single_labels["video"].setText(self.i18n.t("offline.video"))
        self.single_labels["output"].setText(self.i18n.t("offline.output"))
        self.single_labels["mode"].setText(self.i18n.t("offline.mode"))
        self.single_labels["engine"].setText(self.i18n.t("offline.engine"))
        self.single_labels["recognition"].setText(self.i18n.t("offline.recognition_model"))
        self.single_labels["trt"].setText(self.i18n.t("trt.row_label"))
        self.single_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        self.single_labels["time"].setText(self.i18n.t("offline.time_range"))
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory"))
        self.batch_labels["mode"].setText(self.i18n.t("offline.mode"))
        self.batch_labels["engine"].setText(self.i18n.t("offline.engine"))
        self.batch_labels["recognition"].setText(self.i18n.t("offline.recognition_model"))
        self.batch_labels["trt"].setText(self.i18n.t("trt.row_label"))
        self.batch_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        for combo in (self.single_mode, self.batch_mode):
            combo.setItemText(0, self.i18n.t("mode.green"))
            combo.setItemText(1, self.i18n.t("mode.alpha"))
        for combo in (self.single_engine, self.batch_engine):
            combo.setItemText(0, self.i18n.t("engine.rvm_fast"))
            combo.setItemText(1, self.i18n.t("engine.matanyone2"))
        for combo in (self.single_recognition, self.batch_recognition):
            combo.setItemText(0, self.i18n.t("recognition.yoloworld_efficientsam"))
            combo.setItemText(1, self.i18n.t("recognition.sam3"))
        self.single_sam3_prompt_button.setText(self.i18n.t("offline.sam3_prompt_button"))
        self.batch_sam3_prompt_button.setText(self.i18n.t("offline.sam3_prompt_button"))
        self.single_matanyone_help.setToolTip(self.i18n.t("offline.matanyone_help_title"))
        self.batch_matanyone_help.setToolTip(self.i18n.t("offline.matanyone_help_title"))
        self._update_recognition_visibility()
        for index, key in enumerate(("offline.duration_15s", "offline.duration_30s", "offline.duration_1m", "offline.duration_custom", "offline.duration_full")):
            self.single_duration.setItemText(index, self.i18n.t(key))
        self.single_custom_minutes_label.setText(self.i18n.t("offline.minutes"))
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            for index, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium", "quality_speed.veryslow")):
                combo.setItemText(index, self.i18n.t(key))
        self.single_trt_configure_button.setText(self.i18n.t("trt.configure"))
        self.batch_trt_configure_button.setText(self.i18n.t("trt.configure"))
        self._update_trt_cache_rows()
