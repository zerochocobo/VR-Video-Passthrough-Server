"""Standalone offline DLSS 5 Neural Rendering page.

The realtime [DLSS5] card stops at 4K because the decoder, the NR stage and
NVENC all want the same GPU at playback speed. Offline has no such deadline, so
this page is where a 6K or 8K source is enhanced - same engine, same settings,
no resolution ceiling.

The enhancement parameters are the ones the dashboard card already owns, so the
page shows a summary line and opens that very dialog (in its offline shape,
without the playback chooser) rather than growing a second copy of thirteen
sliders that could drift from the first.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSizePolicy, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from ui import theme
from ui.dialogs.feature_dialogs import DLSS5SettingsDialog, float_setting, int_setting
from ui.log_limits import UI_LOG_MAX_BLOCKS
from ui.log_sanitizer import clean_log_text
from ui.page_icons import BACK_ICON_SIZE, back_icon
from ui.pages.offline_page import (
    ACTION_ICON_SIZE, OFFLINE_LABEL_WIDTH, _action_icon, _fit_combo, _format_time_seconds,
    _label, _resolve_time_range,
)
from ui.settings import DEFAULTS, quality_speed_preset, quality_speed_value
from utils.video_metadata import probe_video_metadata

# (settings key, CLI flag) for the values offline.convert reads as floats.
DLSS5_FLOAT_ARGS = (
    ("dlss5_intensity", "--dlss5-intensity"),
    ("dlss5_shimmer_suppression", "--dlss5-shimmer-suppression"),
    ("dlss5_local_tone", "--dlss5-local-tone"),
    ("dlss5_local_structure", "--dlss5-local-structure"),
    ("dlss5_skin_structure", "--dlss5-skin-structure"),
    ("dlss5_color_strength", "--dlss5-color-strength"),
    ("dlss5_tone_preservation", "--dlss5-tone-preservation"),
    ("dlss5_face_skin_protection", "--dlss5-face-skin-protection"),
    ("dlss5_grain_preservation", "--dlss5-grain-preservation"),
)
DLSS5_INT_ARGS = (
    ("dlss5_style", "--dlss5-style"),
    ("dlss5_nr_passes", "--dlss5-nr-passes"),
)


class Dlss5Page(QWidget):
    def __init__(self, i18n, settings, process) -> None:
        super().__init__()
        self.setObjectName("Dlss5Page")
        self.setStyleSheet(
            "QWidget#Dlss5Page, QWidget#Dlss5Page QLabel, QWidget#Dlss5Page QCheckBox { font-size: 9pt; }"
            "QWidget#Dlss5Page QPushButton, QWidget#Dlss5Page QLineEdit, QWidget#Dlss5Page QComboBox, "
            "QWidget#Dlss5Page QTextEdit, QWidget#Dlss5Page QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#Dlss5Page QLabel#Dlss5PageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n, self.settings, self.process = i18n, settings, process
        self.title_label = QLabel(objectName="Dlss5PageTitle")
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
        # Both tabs show the same settings, so the summary labels and the button
        # that edits them are collected and updated together.
        self._params_summaries: list[QLabel] = []
        self._params_buttons: list[QPushButton] = []
        self._single_tab()
        self._batch_tab()
        self.offline_note = QLabel()
        self.offline_note.setWordWrap(True)
        self.offline_note.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")
        header = QHBoxLayout()
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.back_button)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.offline_note)
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

    def _quality_speed_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in ("ultrafast", "medium", "veryslow"):
            combo.addItem("", value)
        idx = combo.findData(quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium"))
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_quality_speed)
        return combo

    def _params_row(self) -> QHBoxLayout:
        """Current enhancement settings, plus the button that edits them."""
        summary = QLabel()
        summary.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")
        button = QPushButton()
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.clicked.connect(self._configure_params)
        self._params_summaries.append(summary)
        self._params_buttons.append(button)
        row = QHBoxLayout()
        row.addWidget(summary, 1)
        row.addWidget(button)
        return row

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
        params_row = self._params_row()
        self.single_quality_speed = self._quality_speed_combo()
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
        self.single_labels = {k: _label() for k in ("video", "output", "params", "performance", "time")}
        grid.addWidget(self.single_labels["video"], 0, 0); grid.addLayout(video_row, 0, 1)
        grid.addWidget(self.single_labels["output"], 1, 0); grid.addLayout(out_row, 1, 1)
        grid.addWidget(self.single_labels["params"], 2, 0); grid.addLayout(params_row, 2, 1)
        grid.addWidget(self.single_labels["performance"], 3, 0); grid.addWidget(self.single_quality_speed, 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["time"], 4, 0); grid.addLayout(time_row, 4, 1)
        grid.addWidget(self.single_skip, 5, 1); grid.addLayout(actions, 6, 1)
        self.tabs.addTab(page, "")
        self._update_custom_visibility()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit(); browse = QPushButton("...")
        browse.clicked.connect(lambda: self._browse_dir(self.batch_dir))
        params_row = self._params_row()
        self.batch_quality_speed = self._quality_speed_combo()
        self.batch_recursive = QCheckBox(); self.batch_recursive.setChecked(True)
        self.batch_skip = QCheckBox(); self.batch_skip.setChecked(True)
        self.start_batch.clicked.connect(self.run_batch)
        dir_row, actions = QHBoxLayout(), QHBoxLayout()
        dir_row.addWidget(self.batch_dir); dir_row.addWidget(browse)
        actions.addWidget(self.start_batch); actions.addWidget(self.stop_batch); actions.addStretch(1)
        grid = QGridLayout(page); grid.setColumnMinimumWidth(0, OFFLINE_LABEL_WIDTH); grid.setColumnStretch(1, 1)
        self.batch_labels = {k: _label() for k in ("directory", "params", "performance")}
        grid.addWidget(self.batch_labels["directory"], 0, 0); grid.addLayout(dir_row, 0, 1)
        grid.addWidget(self.batch_labels["params"], 1, 0); grid.addLayout(params_row, 1, 1)
        grid.addWidget(self.batch_labels["performance"], 2, 0); grid.addWidget(self.batch_quality_speed, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_recursive, 3, 1); grid.addWidget(self.batch_skip, 4, 1); grid.addLayout(actions, 5, 1)
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

    def _configure_params(self) -> None:
        dialog = DLSS5SettingsDialog(self.i18n, self.settings, self, offline=True)
        if dialog.exec() != DLSS5SettingsDialog.DialogCode.Accepted:
            return
        self.settings.data.update(dialog.payload())
        self.settings.save()
        self._update_params_summary()

    def _update_params_summary(self) -> None:
        data = self.settings.data
        style_key = DLSS5SettingsDialog.STYLE_KEYS[
            max(0, min(len(DLSS5SettingsDialog.STYLE_KEYS) - 1,
                       int_setting(data.get("dlss5_style"), DEFAULTS["dlss5_style"])))
        ]
        intensity = float_setting(data.get("dlss5_intensity"), DEFAULTS["dlss5_intensity"])
        passes = int_setting(data.get("dlss5_nr_passes"), DEFAULTS["dlss5_nr_passes"])
        text = (
            f"{self.i18n.t(style_key)} · {self.i18n.t('dlss5.intensity')} {intensity:.2f}"
            f" · {self.i18n.t('dlss5.nr_passes')} {passes}"
        )
        for label in self._params_summaries:
            label.setText(text)

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

    def _common_args(self) -> list[str]:
        """The NR settings, spelled out so a run does not depend on the env."""
        data = self.settings.data
        args = ["--preset", quality_speed_preset(data.get("offline_quality_speed"), "medium").lower()]
        for key, flag in DLSS5_INT_ARGS:
            args += [flag, str(int_setting(data.get(key), DEFAULTS[key]))]
        for key, flag in DLSS5_FLOAT_ARGS:
            args += [flag, f"{float_setting(data.get(key), DEFAULTS[key]):.2f}"]
        args.append(
            "--dlss5-auto-mask" if data.get("dlss5_auto_mask", DEFAULTS["dlss5_auto_mask"])
            else "--no-dlss5-auto-mask"
        )
        return args

    def run_single(self) -> None:
        time_range = self._time_range()
        if time_range is None: return
        start, duration = time_range
        args = ["single", self.single_video.text().strip(), "--start", str(start), "--duration", str(duration), *self._common_args()]
        if self.single_out_dir.text().strip(): args += ["--out-dir", self.single_out_dir.text().strip()]
        if self.single_skip.isChecked(): args.append("--skip-existing")
        self.process.start(args, self.settings.server_env())

    def run_batch(self) -> None:
        directory = self.batch_dir.text().strip()
        if not Path(directory).is_dir():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("offline.time_error_video_missing")); return
        args = ["batch", directory, *self._common_args(), "--recursive" if self.batch_recursive.isChecked() else "--no-recursive"]
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
        self._update_params_summary()

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

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("dlss5.offline_title")); self.back_button.setText(self.i18n.t("button.back"))
        self.offline_note.setText(self.i18n.t("dlss5.offline_note"))
        self.tabs.setTabText(0, self.i18n.t("offline.single_tab")); self.tabs.setTabText(1, self.i18n.t("offline.batch_tab"))
        for b in (self.start_single, self.start_batch): b.setText(self.i18n.t("button.start"))
        for b in (self.stop_single, self.stop_batch): b.setText(self.i18n.t("button.stop"))
        for button in self._params_buttons: button.setText(self.i18n.t("dlss5.configure"))
        self.single_labels["video"].setText(self.i18n.t("offline.video")); self.single_labels["output"].setText(self.i18n.t("offline.output"))
        self.single_labels["params"].setText(self.i18n.t("dlss5.params")); self.single_labels["performance"].setText(self.i18n.t("performance.quality_speed")); self.single_labels["time"].setText(self.i18n.t("offline.time_mode_range"))
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory")); self.batch_labels["params"].setText(self.i18n.t("dlss5.params")); self.batch_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            for i, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium", "quality_speed.veryslow")): combo.setItemText(i, self.i18n.t(key))
        duration_keys = ("offline.duration_15s", "offline.duration_30s", "offline.duration_1m", "offline.duration_custom", "offline.duration_custom_end", "offline.duration_full")
        for i, key in enumerate(duration_keys): self.single_duration.setItemText(i, self.i18n.t(key))
        self.single_skip.setText(self.i18n.t("offline.skip_existing")); self.batch_recursive.setText(self.i18n.t("offline.recursive")); self.batch_skip.setText(self.i18n.t("offline.skip_existing"))
        self.single_custom_minutes_label.setText(self.i18n.t("offline.minutes")); self.single_custom_end_label.setText(self.i18n.t("offline.end_time")); self._update_custom_visibility()
        self._update_params_summary()
