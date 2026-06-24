"""Offline mosaic-restoration (RM) page.

Sibling of :class:`ui.pages.two_dvr_page.TwoDvrPage`. Drives
``offline/demosaic_offline.py`` (via :class:`ui.services.offline_process.RmProcess`)
to run the same detector + restoration models used by the realtime [RM] DLNA path
on the full GPU pipeline, writing ``<stem>_<time>_restored.mp4`` so the result can
be reviewed without a DLNA player. Reuses the offline page time-range helpers.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
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
from ui.pages.offline_page import (
    ACTION_ICON_SIZE,
    OFFLINE_LABEL_WIDTH,
    _action_icon,
    _fit_combo,
    _label,
    _resolve_time_range,
    _format_time_seconds,
)
from ui.settings import quality_speed_preset, quality_speed_value
from utils.video_metadata import probe_video_metadata


class RmPage(QWidget):
    def __init__(self, i18n, settings, process) -> None:
        super().__init__()
        self.setObjectName("RmPage")
        self.setStyleSheet(
            "QWidget#RmPage, QWidget#RmPage QLabel, QWidget#RmPage QCheckBox { font-size: 9pt; }"
            "QWidget#RmPage QPushButton, QWidget#RmPage QLineEdit, QWidget#RmPage QComboBox, "
            "QWidget#RmPage QTextEdit, QWidget#RmPage QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#RmPage QLabel#RmPageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n = i18n
        self.settings = settings
        self.process = process

        self.title_label = QLabel()
        self.title_label.setObjectName("RmPageTitle")
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

    # -- shared widgets ------------------------------------------------------

    def _action_button(self, kind: str) -> QPushButton:
        button = QPushButton()
        button.setIcon(_action_icon(kind))
        button.setIconSize(QSize(ACTION_ICON_SIZE, ACTION_ICON_SIZE))
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return button

    def _quality_speed_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in ("ultrafast", "medium", "veryslow"):
            combo.addItem("", value)
        idx = combo.findData(quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium"))
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_quality_speed)
        return combo

    def _duration_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", 15.0)
        combo.addItem("", 30.0)
        combo.addItem("", 60.0)
        combo.addItem("", "custom")
        combo.addItem("", "custom_end")
        combo.addItem("", 0.0)
        return combo

    # -- single tab ----------------------------------------------------------

    def _time_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.single_start = QLineEdit("00:00:00")
        self.single_start.setFixedWidth(90)
        self.single_duration = self._duration_combo()
        self.single_custom_minutes_label = QLabel()
        self.single_custom_minutes = QLineEdit("5")
        self.single_custom_minutes.setFixedWidth(48)
        self.single_custom_end_label = QLabel()
        self.single_custom_end = QLineEdit("00:05:00")
        self.single_custom_end.setFixedWidth(90)
        self.single_duration.currentIndexChanged.connect(self._update_custom_duration_visibility)
        row.addWidget(self.single_start)
        row.addSpacing(12)
        row.addWidget(self.single_duration)
        row.addWidget(self.single_custom_minutes_label)
        row.addWidget(self.single_custom_minutes)
        row.addWidget(self.single_custom_end_label)
        row.addWidget(self.single_custom_end)
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
        self.single_quality_speed = self._quality_speed_combo()
        self.single_skip = QCheckBox()
        self.single_skip.setChecked(True)
        self.start_single.clicked.connect(self.run_single)

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
        self.single_labels = {key: _label() for key in ("video", "output", "performance", "time")}
        grid.addWidget(self.single_labels["video"], 0, 0)
        grid.addLayout(row_video, 0, 1)
        grid.addWidget(self.single_labels["output"], 1, 0)
        grid.addLayout(row_out, 1, 1)
        grid.addWidget(self.single_labels["performance"], 2, 0)
        grid.addWidget(self.single_quality_speed, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["time"], 3, 0, alignment=Qt.AlignRight)
        grid.addLayout(self._time_row(), 3, 1)
        grid.addWidget(self.single_skip, 4, 1)
        grid.addLayout(actions, 5, 1)
        self.tabs.addTab(page, "")
        self._update_custom_duration_visibility()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit()
        browse_dir = QPushButton("...")
        browse_dir.clicked.connect(lambda: self._browse_dir(self.batch_dir))
        self.batch_quality_speed = self._quality_speed_combo()
        self.batch_recursive = QCheckBox()
        self.batch_recursive.setChecked(True)
        self.batch_skip = QCheckBox()
        self.batch_skip.setChecked(True)
        self.start_batch.clicked.connect(self.run_batch)

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
        self.batch_labels = {key: _label() for key in ("directory", "performance")}
        grid.addWidget(self.batch_labels["directory"], 0, 0)
        grid.addLayout(row_dir, 0, 1)
        grid.addWidget(self.batch_labels["performance"], 1, 0)
        grid.addWidget(self.batch_quality_speed, 1, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_recursive, 2, 1)
        grid.addWidget(self.batch_skip, 3, 1)
        grid.addLayout(actions, 4, 1)
        self.tabs.addTab(page, "")

    # -- browsing ------------------------------------------------------------

    def _browse_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.i18n.t("file.select_video"), "", "Videos (*.mp4 *.mkv *.mov *.m4v)")
        if path:
            target.setText(path)
            self.single_out_dir.setText(str(Path(path).parent))

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("file.select_directory"))
        if path:
            target.setText(path)

    # -- time mode -----------------------------------------------------------

    def _update_custom_duration_visibility(self) -> None:
        value = self.single_duration.currentData()
        minutes_visible = value == "custom"
        end_visible = value == "custom_end"
        self.single_custom_minutes_label.setVisible(minutes_visible)
        self.single_custom_minutes.setVisible(minutes_visible)
        self.single_custom_end_label.setVisible(end_visible)
        self.single_custom_end.setVisible(end_visible)

    # -- validation ----------------------------------------------------------

    def _show_time_error(self, key: str, video_duration: float = 0.0) -> None:
        message = self.i18n.t(key).format(duration=_format_time_seconds(video_duration), row=0)
        QMessageBox.warning(self, self.i18n.t("offline.time_error_title"), message)

    def _validated_single_time_range(self) -> tuple[float, float] | None:
        video_text = self.single_video.text().strip()
        if not video_text or not Path(video_text).is_file():
            self._show_time_error("offline.time_error_video_missing")
            return None
        try:
            video_duration = float(probe_video_metadata(Path(video_text)).timing.duration or 0.0)
        except Exception:
            video_duration = 0.0
        if video_duration <= 0:
            self._show_time_error("offline.time_error_video_duration")
            return None
        start, duration, error_key = _resolve_time_range(
            self.single_start.text(),
            self.single_duration.currentData(),
            self.single_custom_minutes.text(),
            self.single_custom_end.text(),
            video_duration,
        )
        if error_key:
            self._show_time_error(error_key, video_duration)
            return None
        return start, duration

    # -- run -----------------------------------------------------------------

    def _quality_env(self) -> dict[str, str]:
        env = self.settings.server_env()
        env["PT_PASSTHROUGH_PYNV_PRESET"] = quality_speed_preset(
            self.settings.data.get("offline_quality_speed"), "medium"
        )
        return env

    def run_single(self) -> None:
        video = self.single_video.text().strip()
        if not video or not Path(video).is_file():
            self._show_time_error("offline.time_error_video_missing")
            return
        time_range = self._validated_single_time_range()
        if time_range is None:
            return
        start_seconds, duration_seconds = time_range
        args = ["single", video, "--start", str(start_seconds), "--duration", str(duration_seconds)]
        if self.single_out_dir.text().strip():
            args.extend(["--out-dir", self.single_out_dir.text().strip()])
        if self.single_skip.isChecked():
            args.append("--skip-existing")
        self.settings.save()
        self.process.start(args, self._quality_env())

    def run_batch(self) -> None:
        directory = self.batch_dir.text().strip()
        if not directory or not Path(directory).is_dir():
            self._show_time_error("offline.time_error_video_missing")
            return
        args = ["batch", directory]
        args.append("--recursive" if self.batch_recursive.isChecked() else "--no-recursive")
        if self.batch_skip.isChecked():
            args.append("--skip-existing")
        self.settings.save()
        self.process.start(args, self._quality_env())

    # -- state / logging -----------------------------------------------------

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

    def sync_from_settings(self) -> None:
        value = quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium")
        for combo in (getattr(self, "single_quality_speed", None), getattr(self, "batch_quality_speed", None)):
            if isinstance(combo, QComboBox):
                idx = combo.findData(value)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)

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

    # -- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("rm.offline_title"))
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
        self.single_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        self.single_labels["time"].setText(self.i18n.t("offline.time_mode_range"))
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory"))
        self.batch_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            for index, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium", "quality_speed.veryslow")):
                combo.setItemText(index, self.i18n.t(key))
        for index, key in enumerate((
            "offline.duration_15s",
            "offline.duration_30s",
            "offline.duration_1m",
            "offline.duration_custom",
            "offline.duration_custom_end",
            "offline.duration_full",
        )):
            self.single_duration.setItemText(index, self.i18n.t(key))
        self.single_custom_minutes_label.setText(self.i18n.t("offline.minutes"))
        self.single_custom_end_label.setText(self.i18n.t("offline.end_time"))
        self._update_custom_duration_visibility()
