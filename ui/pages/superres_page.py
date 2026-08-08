"""Standalone offline NVIDIA RTX VSR SuperRes page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSizePolicy, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.page_icons import BACK_ICON_SIZE, back_icon
from ui.pages.offline_page import (
    ACTION_ICON_SIZE, OFFLINE_LABEL_WIDTH, _action_icon, _fit_combo, _format_time_seconds,
    _label, _resolve_time_range,
)
from utils.video_metadata import probe_video_metadata
from ui.settings import quality_speed_preset, quality_speed_value


class SuperResPage(QWidget):
    def __init__(self, i18n, settings, process) -> None:
        super().__init__()
        self.setObjectName("SuperResPage")
        self.setStyleSheet(
            "QWidget#SuperResPage, QWidget#SuperResPage QLabel, QWidget#SuperResPage QCheckBox { font-size: 9pt; }"
            "QWidget#SuperResPage QPushButton, QWidget#SuperResPage QLineEdit, QWidget#SuperResPage QComboBox, "
            "QWidget#SuperResPage QTextEdit, QWidget#SuperResPage QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#SuperResPage QLabel#SuperResPageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n, self.settings, self.process = i18n, settings, process
        self.title_label = QLabel(objectName="SuperResPageTitle")
        self.back_button = QPushButton()
        self.back_button.setIcon(back_icon())
        self.back_button.setIconSize(QSize(BACK_ICON_SIZE, BACK_ICON_SIZE))
        self.back_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.tabs = QTabWidget()
        self.log = QTextEdit(readOnly=True)
        self.log.document().setMaximumBlockCount(UI_LOG_MAX_BLOCKS)
        self.start_single, self.stop_single = self._action_button("start"), self._action_button("stop")
        self.start_batch, self.stop_batch = self._action_button("start"), self._action_button("stop")
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

    @staticmethod
    def _target_combo() -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", 1440)
        combo.addItem("", 2160)
        combo.addItem("", 4096)
        combo.setCurrentIndex(2)
        return combo

    @staticmethod
    def _quality_combo() -> QComboBox:
        combo = _fit_combo(QComboBox())
        for quality in (2, 3, 4):
            combo.addItem("", quality)
        combo.setCurrentIndex(2)
        return combo

    def _quality_speed_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in ("ultrafast", "medium", "veryslow"):
            combo.addItem("", value)
        idx = combo.findData(quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium"))
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_quality_speed)
        return combo

    def _hdr_look_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for mode in ("off", "natural", "vivid"):
            combo.addItem("", mode)
        current = str(self.settings.data.get("superres_offline_hdr_look") or "natural").strip().lower()
        idx = combo.findData(current if current in {"off", "natural", "vivid"} else "natural")
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_hdr_look)
        return combo

    def _bitrate_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for mode in ("auto", "1.2", "1.5", "2", "3"):
            combo.addItem("", mode)
        current = str(self.settings.data.get("superres_offline_bitrate_mode") or "auto")
        idx = combo.findData(current if current in {"auto", "1.2", "1.5", "2", "3"} else "auto")
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_bitrate_mode)
        return combo

    @staticmethod
    def _duration_combo() -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in (15.0, 30.0, 60.0, "custom", "custom_end", 0.0):
            combo.addItem("", value)
        return combo

    def _single_tab(self) -> None:
        page = QWidget()
        self.single_video, self.single_out_dir = QLineEdit(), QLineEdit()
        browse_video, browse_out = QPushButton("..."), QPushButton("...")
        browse_video.clicked.connect(self._browse_video)
        browse_out.clicked.connect(lambda: self._browse_dir(self.single_out_dir))
        self.single_target, self.single_quality = self._target_combo(), self._quality_combo()
        self.single_quality_speed = self._quality_speed_combo()
        self.single_hdr_look = self._hdr_look_combo()
        self.single_bitrate = self._bitrate_combo()
        self.single_start = QLineEdit("00:00:00")
        self.single_start.setFixedWidth(90)
        self.single_duration = self._duration_combo()
        self.single_custom_minutes = QLineEdit("5")
        self.single_custom_minutes.setFixedWidth(48)
        self.single_custom_end = QLineEdit("00:05:00")
        self.single_custom_end.setFixedWidth(90)
        self.single_custom_minutes_label, self.single_custom_end_label = QLabel(), QLabel()
        self.single_duration.currentIndexChanged.connect(self._update_custom_visibility)
        self.single_skip = QCheckBox(); self.single_skip.setChecked(True)
        self.start_single.clicked.connect(self.run_single)
        video_row, out_row, time_row, actions = QHBoxLayout(), QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        video_row.addWidget(self.single_video); video_row.addWidget(browse_video)
        out_row.addWidget(self.single_out_dir); out_row.addWidget(browse_out)
        for widget in (self.single_start, self.single_duration, self.single_custom_minutes_label,
                       self.single_custom_minutes, self.single_custom_end_label, self.single_custom_end):
            time_row.addWidget(widget)
        time_row.addStretch(1)
        actions.addWidget(self.start_single); actions.addWidget(self.stop_single); actions.addStretch(1)
        grid = QGridLayout(page); grid.setColumnMinimumWidth(0, OFFLINE_LABEL_WIDTH); grid.setColumnStretch(1, 1)
        self.single_labels = {k: _label() for k in ("video", "output", "target", "quality", "hdr_look", "bitrate", "performance", "time")}
        rows = [("video", video_row), ("output", out_row)]
        for row, (key, value) in enumerate(rows): grid.addWidget(self.single_labels[key], row, 0); grid.addLayout(value, row, 1)
        grid.addWidget(self.single_labels["target"], 2, 0); grid.addWidget(self.single_target, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["quality"], 3, 0); grid.addWidget(self.single_quality, 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["hdr_look"], 4, 0); grid.addWidget(self.single_hdr_look, 4, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["bitrate"], 5, 0); grid.addWidget(self.single_bitrate, 5, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["performance"], 6, 0); grid.addWidget(self.single_quality_speed, 6, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["time"], 7, 0); grid.addLayout(time_row, 7, 1)
        grid.addWidget(self.single_skip, 8, 1); grid.addLayout(actions, 9, 1)
        self.tabs.addTab(page, "")
        self._update_custom_visibility()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit(); browse = QPushButton("...")
        browse.clicked.connect(lambda: self._browse_dir(self.batch_dir))
        self.batch_target, self.batch_quality = self._target_combo(), self._quality_combo()
        self.batch_quality_speed = self._quality_speed_combo()
        self.batch_hdr_look = self._hdr_look_combo()
        self.batch_bitrate = self._bitrate_combo()
        self.batch_recursive = QCheckBox(); self.batch_recursive.setChecked(True)
        self.batch_skip = QCheckBox(); self.batch_skip.setChecked(True)
        self.start_batch.clicked.connect(self.run_batch)
        dir_row, actions = QHBoxLayout(), QHBoxLayout()
        dir_row.addWidget(self.batch_dir); dir_row.addWidget(browse)
        actions.addWidget(self.start_batch); actions.addWidget(self.stop_batch); actions.addStretch(1)
        grid = QGridLayout(page); grid.setColumnMinimumWidth(0, OFFLINE_LABEL_WIDTH); grid.setColumnStretch(1, 1)
        self.batch_labels = {k: _label() for k in ("directory", "target", "quality", "hdr_look", "bitrate", "performance")}
        grid.addWidget(self.batch_labels["directory"], 0, 0); grid.addLayout(dir_row, 0, 1)
        grid.addWidget(self.batch_labels["target"], 1, 0); grid.addWidget(self.batch_target, 1, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["quality"], 2, 0); grid.addWidget(self.batch_quality, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["hdr_look"], 3, 0); grid.addWidget(self.batch_hdr_look, 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["bitrate"], 4, 0); grid.addWidget(self.batch_bitrate, 4, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["performance"], 5, 0); grid.addWidget(self.batch_quality_speed, 5, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_recursive, 6, 1); grid.addWidget(self.batch_skip, 7, 1); grid.addLayout(actions, 8, 1)
        self.tabs.addTab(page, "")

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.i18n.t("file.select_video"), "", "Videos (*.mp4 *.mkv *.mov *.m4v)")
        if path:
            self.single_video.setText(path); self.single_out_dir.setText(str(Path(path).parent))

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("file.select_directory"))
        if path: target.setText(path)

    def _update_custom_visibility(self) -> None:
        value = self.single_duration.currentData()
        self.single_custom_minutes_label.setVisible(value == "custom")
        self.single_custom_minutes.setVisible(value == "custom")
        self.single_custom_end_label.setVisible(value == "custom_end")
        self.single_custom_end.setVisible(value == "custom_end")

    def _time_range(self) -> tuple[float, float] | None:
        path = Path(self.single_video.text().strip())
        if not path.is_file():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("offline.time_error_video_missing")); return None
        try: duration = float(probe_video_metadata(path).timing.duration or 0.0)
        except Exception: duration = 0.0
        start, length, error = _resolve_time_range(self.single_start.text(), self.single_duration.currentData(),
                                                   self.single_custom_minutes.text(), self.single_custom_end.text(), duration)
        if error:
            QMessageBox.warning(self, self.i18n.t("offline.time_error_title"), self.i18n.t(error).format(duration=_format_time_seconds(duration), row=0)); return None
        return start, length

    def _common_args(self, target: QComboBox, quality: QComboBox, hdr_look: QComboBox, bitrate: QComboBox) -> list[str]:
        return ["--rtx-vsr-target-height", str(target.currentData()), "--rtx-vsr-quality", str(quality.currentData()), "--rtx-vsr-hdr-look", str(hdr_look.currentData()), "--rtx-vsr-bitrate-mode", str(bitrate.currentData()), "--preset", quality_speed_preset(self.settings.data.get("offline_quality_speed"), "medium").lower()]

    def run_single(self) -> None:
        time_range = self._time_range()
        if time_range is None: return
        start, duration = time_range
        args = ["single", self.single_video.text().strip(), "--start", str(start), "--duration", str(duration), *self._common_args(self.single_target, self.single_quality, self.single_hdr_look, self.single_bitrate)]
        if self.single_out_dir.text().strip(): args += ["--out-dir", self.single_out_dir.text().strip()]
        if self.single_skip.isChecked(): args.append("--skip-existing")
        self.process.start(args, self.settings.server_env())

    def run_batch(self) -> None:
        directory = self.batch_dir.text().strip()
        if not Path(directory).is_dir():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("offline.time_error_video_missing")); return
        args = ["batch", directory, *self._common_args(self.batch_target, self.batch_quality, self.batch_hdr_look, self.batch_bitrate), "--recursive" if self.batch_recursive.isChecked() else "--no-recursive"]
        if self.batch_skip.isChecked(): args.append("--skip-existing")
        self.process.start(args, self.settings.server_env())

    def set_running(self, running: bool) -> None:
        self.start_single.setEnabled(not running); self.start_batch.setEnabled(not running)
        self.stop_single.setEnabled(running); self.stop_batch.setEnabled(running)

    def append_log(self, text: str) -> None:
        text = clean_log_text(text)
        if text: self.log.moveCursor(self.log.textCursor().MoveOperation.End); self.log.insertPlainText(text); self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def sync_from_settings(self) -> None:
        value = quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium")
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            idx = combo.findData(value)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
        hdr_mode = str(self.settings.data.get("superres_offline_hdr_look") or "natural").strip().lower()
        if hdr_mode not in {"off", "natural", "vivid"}:
            hdr_mode = "natural"
        for combo in (self.single_hdr_look, self.batch_hdr_look):
            idx = combo.findData(hdr_mode)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
        bitrate_mode = str(self.settings.data.get("superres_offline_bitrate_mode") or "auto")
        if bitrate_mode not in {"auto", "1.2", "1.5", "2", "3"}:
            bitrate_mode = "auto"
        for combo in (self.single_bitrate, self.batch_bitrate):
            idx = combo.findData(bitrate_mode)
            if idx >= 0 and combo.currentIndex() != idx:
                combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)

    def _save_quality_speed(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QComboBox):
            return
        value = quality_speed_value(sender.currentData(), "medium")
        self.settings.data["offline_quality_speed"] = value
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            if combo is not sender:
                idx = combo.findData(value)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
        self.settings.save()

    def _save_hdr_look(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QComboBox):
            return
        value = str(sender.currentData() or "natural")
        if value not in {"off", "natural", "vivid"}:
            value = "natural"
        self.settings.data["superres_offline_hdr_look"] = value
        for combo in (self.single_hdr_look, self.batch_hdr_look):
            if combo is not sender:
                idx = combo.findData(value)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
        self.settings.save()

    def _save_bitrate_mode(self) -> None:
        sender = self.sender()
        if not isinstance(sender, QComboBox):
            return
        value = str(sender.currentData() or "auto")
        if value not in {"auto", "1.2", "1.5", "2", "3"}:
            value = "auto"
        self.settings.data["superres_offline_bitrate_mode"] = value
        for combo in (self.single_bitrate, self.batch_bitrate):
            if combo is not sender:
                idx = combo.findData(value)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
        self.settings.save()

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("superres.offline_title")); self.back_button.setText(self.i18n.t("button.back"))
        self.tabs.setTabText(0, self.i18n.t("offline.single_tab")); self.tabs.setTabText(1, self.i18n.t("offline.batch_tab"))
        for b in (self.start_single, self.start_batch): b.setText(self.i18n.t("button.start"))
        for b in (self.stop_single, self.stop_batch): b.setText(self.i18n.t("button.stop"))
        self.single_labels["video"].setText(self.i18n.t("offline.video")); self.single_labels["output"].setText(self.i18n.t("offline.output"))
        self.single_labels["target"].setText(self.i18n.t("superres.target")); self.single_labels["quality"].setText(self.i18n.t("superres.quality")); self.single_labels["hdr_look"].setText(self.i18n.t("superres.hdr_look")); self.single_labels["bitrate"].setText(self.i18n.t("superres.output_bitrate")); self.single_labels["performance"].setText(self.i18n.t("performance.quality_speed")); self.single_labels["time"].setText(self.i18n.t("offline.time_mode_range"))
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory")); self.batch_labels["target"].setText(self.i18n.t("superres.target")); self.batch_labels["quality"].setText(self.i18n.t("superres.quality")); self.batch_labels["hdr_look"].setText(self.i18n.t("superres.hdr_look")); self.batch_labels["bitrate"].setText(self.i18n.t("superres.output_bitrate")); self.batch_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        for combo in (self.single_hdr_look, self.batch_hdr_look):
            for i, key in enumerate(("superres.hdr_look_off", "superres.hdr_look_natural", "superres.hdr_look_vivid")): combo.setItemText(i, self.i18n.t(key))
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            for i, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium", "quality_speed.veryslow")): combo.setItemText(i, self.i18n.t(key))
        bitrate_keys = ("superres.bitrate_auto", "superres.bitrate_1_2", "superres.bitrate_1_5", "superres.bitrate_2", "superres.bitrate_3")
        for combo in (self.single_bitrate, self.batch_bitrate):
            for i, key in enumerate(bitrate_keys): combo.setItemText(i, self.i18n.t(key))
        target_keys = ("superres.target_2k", "superres.target_4k", "superres.target_8k_vr")
        for combo in (self.single_target, self.batch_target):
            for i, key in enumerate(target_keys): combo.setItemText(i, self.i18n.t(key))
        quality_keys = ("superres.quality_2", "superres.quality_3", "superres.quality_4")
        for combo in (self.single_quality, self.batch_quality):
            for i, key in enumerate(quality_keys): combo.setItemText(i, self.i18n.t(key))
        duration_keys = ("offline.duration_15s", "offline.duration_30s", "offline.duration_1m", "offline.duration_custom", "offline.duration_custom_end", "offline.duration_full")
        for i, key in enumerate(duration_keys): self.single_duration.setItemText(i, self.i18n.t(key))
        self.single_skip.setText(self.i18n.t("offline.skip_existing")); self.batch_recursive.setText(self.i18n.t("offline.recursive")); self.batch_skip.setText(self.i18n.t("offline.skip_existing"))
        self.single_custom_minutes_label.setText(self.i18n.t("offline.minutes")); self.single_custom_end_label.setText(self.i18n.t("offline.end_time")); self._update_custom_visibility()
