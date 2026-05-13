from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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


OFFLINE_LABEL_WIDTH = 132
ACTION_ICON_SIZE = 20
HELP_ICON_SIZE = 20


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
    def __init__(self, i18n, process) -> None:
        super().__init__()
        self.setObjectName("OfflinePage")
        self.setStyleSheet(
            "QWidget#OfflinePage, QWidget#OfflinePage QLabel, QWidget#OfflinePage QCheckBox { font-size: 9pt; }"
            "QWidget#OfflinePage QPushButton, QWidget#OfflinePage QLineEdit, QWidget#OfflinePage QComboBox, "
            "QWidget#OfflinePage QTextEdit, QWidget#OfflinePage QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#OfflinePage QLabel#OfflinePageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n = i18n
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
        combo.addItem("", "rvm_balanced")
        combo.addItem("", "matanyone2")
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

    def _skip_frames_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", 0)
        combo.addItem("", 1)
        combo.addItem("", 2)
        return combo

    def _performance_row(self, combo: QComboBox) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel()
        row.addWidget(label)
        row.addWidget(combo)
        row.addStretch(1)
        if not hasattr(self, "performance_skip_labels"):
            self.performance_skip_labels = []
        self.performance_skip_labels.append(label)
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
        self.single_matanyone_help = _help_button()
        self.single_skip_frames = self._skip_frames_combo()
        self.single_skip = QCheckBox()
        self.single_skip.setChecked(True)
        self.start_single.clicked.connect(self.run_single)
        self.single_engine.currentIndexChanged.connect(self._update_matanyone_help_visibility)
        self.single_matanyone_help.clicked.connect(self.show_matanyone_help)
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
        self.single_labels = {key: _label() for key in ("video", "output", "mode", "engine", "performance", "time")}
        grid.addWidget(self.single_labels["video"], 0, 0)
        grid.addLayout(row_video, 0, 1)
        grid.addWidget(self.single_labels["output"], 1, 0)
        grid.addLayout(row_out, 1, 1)
        grid.addWidget(self.single_labels["mode"], 2, 0)
        grid.addWidget(self.single_mode, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["engine"], 3, 0)
        single_engine_row = QHBoxLayout()
        single_engine_row.addWidget(self.single_engine)
        single_engine_row.addWidget(self.single_matanyone_help)
        single_engine_row.addStretch(1)
        grid.addLayout(single_engine_row, 3, 1)
        grid.addWidget(self.single_labels["performance"], 4, 0)
        grid.addLayout(self._performance_row(self.single_skip_frames), 4, 1)
        grid.addWidget(self.single_labels["time"], 5, 0)
        grid.addLayout(self._time_row(), 5, 1)
        grid.addWidget(self.single_skip, 6, 1)
        grid.addLayout(actions, 7, 1)
        self.tabs.addTab(page, "")
        self._update_custom_duration_visibility()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit()
        browse_dir = QPushButton("...")
        browse_dir.clicked.connect(lambda: self._browse_dir(self.batch_dir))
        self.batch_mode = self._mode_combo()
        self.batch_engine = self._engine_combo()
        self.batch_matanyone_help = _help_button()
        self.batch_skip_frames = self._skip_frames_combo()
        self.batch_recursive = QCheckBox()
        self.batch_recursive.setChecked(True)
        self.batch_skip = QCheckBox()
        self.batch_skip.setChecked(True)
        self.start_batch.clicked.connect(self.run_batch)
        self.batch_engine.currentIndexChanged.connect(self._update_matanyone_help_visibility)
        self.batch_matanyone_help.clicked.connect(self.show_matanyone_help)
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
        self.batch_labels = {key: _label() for key in ("directory", "mode", "engine", "performance")}
        grid.addWidget(self.batch_labels["directory"], 0, 0)
        grid.addLayout(row_dir, 0, 1)
        grid.addWidget(self.batch_labels["mode"], 1, 0)
        grid.addWidget(self.batch_mode, 1, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["engine"], 2, 0)
        batch_engine_row = QHBoxLayout()
        batch_engine_row.addWidget(self.batch_engine)
        batch_engine_row.addWidget(self.batch_matanyone_help)
        batch_engine_row.addStretch(1)
        grid.addLayout(batch_engine_row, 2, 1)
        grid.addWidget(self.batch_labels["performance"], 3, 0)
        grid.addLayout(self._performance_row(self.batch_skip_frames), 3, 1)
        grid.addWidget(self.batch_recursive, 4, 1)
        grid.addWidget(self.batch_skip, 5, 1)
        grid.addLayout(actions, 6, 1)
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
        self.single_matanyone_help.setVisible(self.single_engine.currentData() == "matanyone2")
        self.batch_matanyone_help.setVisible(self.batch_engine.currentData() == "matanyone2")

    def show_matanyone_help(self) -> None:
        QMessageBox.information(
            self,
            self.i18n.t("offline.matanyone_help_title"),
            self.i18n.t("offline.matanyone_help_msg"),
        )

    def run_single(self) -> None:
        args = [
            "single",
            self.single_video.text(),
            "--mode",
            self.single_mode.currentData(),
            "--engine",
            self.single_engine.currentData(),
            "--start",
            str(self._start_seconds()),
            "--duration",
            str(self._duration_seconds()),
            "--skip-frames",
            str(self.single_skip_frames.currentData()),
        ]
        if self.single_out_dir.text().strip():
            args.extend(["--out-dir", self.single_out_dir.text().strip()])
        if self.single_skip.isChecked():
            args.append("--skip-existing")
        self.process.start(args)

    def run_batch(self) -> None:
        args = [
            "batch",
            self.batch_dir.text(),
            "--mode",
            self.batch_mode.currentData(),
            "--engine",
            self.batch_engine.currentData(),
            "--skip-frames",
            str(self.batch_skip_frames.currentData()),
        ]
        args.append("--recursive" if self.batch_recursive.isChecked() else "--no-recursive")
        if self.batch_skip.isChecked():
            args.append("--skip-existing")
        self.process.start(args)

    def set_running(self, running: bool) -> None:
        self.start_single.setEnabled(not running)
        self.start_batch.setEnabled(not running)
        self.stop_single.setEnabled(running)
        self.stop_batch.setEnabled(running)

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
        self.single_labels["performance"].setText(self.i18n.t("offline.performance"))
        self.single_labels["time"].setText(self.i18n.t("offline.time_range"))
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory"))
        self.batch_labels["mode"].setText(self.i18n.t("offline.mode"))
        self.batch_labels["engine"].setText(self.i18n.t("offline.engine"))
        self.batch_labels["performance"].setText(self.i18n.t("offline.performance"))
        for combo in (self.single_mode, self.batch_mode):
            combo.setItemText(0, self.i18n.t("mode.green"))
            combo.setItemText(1, self.i18n.t("mode.alpha"))
        for combo in (self.single_engine, self.batch_engine):
            combo.setItemText(0, self.i18n.t("engine.rvm_fast"))
            combo.setItemText(1, self.i18n.t("engine.rvm_balanced"))
            combo.setItemText(2, self.i18n.t("engine.matanyone2"))
        self.single_matanyone_help.setToolTip(self.i18n.t("offline.matanyone_help_title"))
        self.batch_matanyone_help.setToolTip(self.i18n.t("offline.matanyone_help_title"))
        self._update_matanyone_help_visibility()
        for index, key in enumerate(("offline.duration_15s", "offline.duration_30s", "offline.duration_1m", "offline.duration_custom", "offline.duration_full")):
            self.single_duration.setItemText(index, self.i18n.t(key))
        self.single_custom_minutes_label.setText(self.i18n.t("offline.minutes"))
        for label in self.performance_skip_labels:
            label.setText(self.i18n.t("offline.frame_skip"))
        for combo in (self.single_skip_frames, self.batch_skip_frames):
            for index, key in enumerate(("offline.skip_none", "offline.skip_1", "offline.skip_2")):
                combo.setItemText(index, self.i18n.t(key))
