"""Offline face-beautification page.

Sibling of :class:`ui.pages.two_dvr_page.TwoDvrPage`: same single/batch tab
layout, same time-range / segment helpers, same log pane.  Drives
``offline/face_beauty.py`` via :class:`ui.services.offline_process.FaceBeautyProcess`.

The page shows one choice -- the beauty preset -- plus a "fine tune" button that
opens :class:`ui.widgets.face_beauty_tuning_dialog.FaceBeautyTuningDialog` with
the full option set.  Presets and their values live in
:mod:`offline.face_beauty_engine` so the CLI and the UI cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
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
    _coerce_time_segments,
    _fit_combo,
    _format_time_seconds,
    _label,
    _parse_hhmmss_text,
    _resolve_time_range,
    _resolve_time_segments,
    _serialize_time_segments,
)
from utils.video_metadata import probe_video_metadata

_SETTINGS_TIME_SEGMENTS_KEY = "face_beauty_single_time_segments"
_SETTINGS_OPTIONS_KEY = "face_beauty_options"

# Restoration models offered on the page, fastest first. Measured on one 512
# crop with a TensorRT fp32 engine: gpen_bfr_256 6.1 ms, gfpgan_1.4 38.0 ms,
# restoreformer_plus_plus 130.1 ms. The labels say fast/slow rather than
# quoting these numbers, which mean nothing to most users.
_ENHANCERS = ("gpen_bfr_256", "gfpgan_1.4", "restoreformer_plus_plus", "none")
_ENHANCER_LABEL_KEYS = {
    "gpen_bfr_256": "beauty.model_gpen256",
    "gfpgan_1.4": "beauty.model_gfpgan",
    "restoreformer_plus_plus": "beauty.model_restoreformer",
    "none": "beauty.model_none",
}

_PRESETS = ("restore", "natural", "standard", "strong")
_PRESET_CUSTOM = "custom"
_PRESET_LABEL_KEYS = {
    "restore": "beauty.preset_restore",
    "natural": "beauty.preset_natural",
    "standard": "beauty.preset_standard",
    "strong": "beauty.preset_strong",
    _PRESET_CUSTOM: "beauty.preset_custom",
}

# Retouch sliders that need the BiSeNet parser: any of them being non-zero pulls
# that graph (and its TensorRT engine) into the run.
_PARSER_GATED_KEYS = ("skin_smooth", "skin_brighten", "skin_even",
                      "eye_brighten", "teeth_white", "lip_vivid")
# Percent sliders -> CLI flags.
_SLIDER_FLAGS = {
    "enhancer_blend": "--enhancer-blend",
    "skin_smooth": "--skin-smooth",
    "skin_brighten": "--skin-brighten",
    "skin_even": "--skin-even",
    "eye_brighten": "--eye-brighten",
    "teeth_white": "--teeth-white",
    "lip_vivid": "--lip-vivid",
    "sharpen": "--sharpen",
    "mask_blur": "--mask-blur",
    "temporal_smooth": "--temporal-smooth",
}
# Everything the tuning dialog can change, beyond the preset sliders.
_ADVANCED_DEFAULTS = {
    "enhancer": "gpen_bfr_256",
    "mask_blur": 30,
    "temporal_smooth": 50,
    "mask_padding": 0,
    "region_mask": True,
    "max_side": 0,
    "min_face_mode": "auto",
    "detect_mode": "auto",
    "vr_reproject": "auto",
    "detect_interval": 1,
    "detect_roi": True,
    "detector_score": 50,
    "max_faces": 0,
    "landmarker": True,
}


def _default_values(preset: str = "standard") -> dict:
    from offline.face_beauty_engine import preset_percentages

    values = dict(_ADVANCED_DEFAULTS)
    values.update(preset_percentages(preset))
    return values


class FaceBeautyPage(QWidget):
    def __init__(self, i18n, settings, process) -> None:
        super().__init__()
        self.setObjectName("FaceBeautyPage")
        self.setStyleSheet(
            "QWidget#FaceBeautyPage, QWidget#FaceBeautyPage QLabel, QWidget#FaceBeautyPage QCheckBox { font-size: 9pt; }"
            "QWidget#FaceBeautyPage QPushButton, QWidget#FaceBeautyPage QLineEdit, QWidget#FaceBeautyPage QComboBox, "
            "QWidget#FaceBeautyPage QTextEdit, QWidget#FaceBeautyPage QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#FaceBeautyPage QLabel#FaceBeautyPageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n = i18n
        self.settings = settings
        self.process = process
        # Execution provider for the next run; _preflight_trt drops it to "cuda"
        # when the user skips the TensorRT build.
        self._provider = "trt"
        self.values = self._load_values()
        self.single_time_segments = _coerce_time_segments(self.settings.data.get(_SETTINGS_TIME_SEGMENTS_KEY))

        self.title_label = QLabel()
        self.title_label.setObjectName("FaceBeautyPageTitle")
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

    # -- persisted option state ----------------------------------------------

    def _load_values(self) -> dict:
        values = _default_values()
        stored = self.settings.data.get(_SETTINGS_OPTIONS_KEY)
        if isinstance(stored, dict):
            # Only adopt keys we still know about, so an older settings file
            # cannot resurrect a removed option.
            values.update({k: v for k, v in stored.items() if k in values})
        return values

    def _save_values(self) -> None:
        self.settings.data[_SETTINGS_OPTIONS_KEY] = dict(self.values)
        self.settings.save()

    def current_preset(self) -> str:
        from offline.face_beauty_engine import match_preset

        return match_preset(self.values)

    # -- shared widgets ------------------------------------------------------

    def _action_button(self, kind: str) -> QPushButton:
        button = QPushButton()
        button.setIcon(_action_icon(kind))
        button.setIconSize(QSize(ACTION_ICON_SIZE, ACTION_ICON_SIZE))
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return button

    def _beauty_settings_cell(self, scope: str) -> QVBoxLayout:
        """Preset dropdown + fine-tune button + effect summary, one instance per
        tab (``scope`` is ``single`` or ``batch``).

        Both tabs edit the same :attr:`values`, so the two instances are kept in
        lockstep by :meth:`_refresh_settings_rows` -- picking a preset on one tab
        is immediately reflected on the other."""
        combo = _fit_combo(QComboBox())
        for name in (*_PRESETS, _PRESET_CUSTOM):
            combo.addItem("", name)
        combo.currentIndexChanged.connect(lambda _index, item=scope: self._preset_selected(item))
        button = QPushButton()
        button.clicked.connect(self.show_tuning_dialog)
        summary = QLabel()
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #5f6368;")
        setattr(self, f"{scope}_preset_combo", combo)
        setattr(self, f"{scope}_tune_button", button)
        setattr(self, f"{scope}_preset_summary", summary)

        row = QHBoxLayout()
        row.addWidget(combo)
        row.addWidget(button)
        row.addStretch(1)
        cell = QVBoxLayout()
        cell.setSpacing(2)
        cell.addLayout(row)
        cell.addWidget(summary)
        return cell

    def _model_row(self, scope: str) -> QHBoxLayout:
        """Restoration-model picker plus its download state.

        Kept on the page rather than in the tuning dialog: it is the setting that
        decides both speed and whether anything needs downloading at all."""
        combo = _fit_combo(QComboBox())
        for name in _ENHANCERS:
            combo.addItem("", name)
        combo.currentIndexChanged.connect(lambda _i, item=scope: self._enhancer_selected(item))
        status = QLabel()
        status.setStyleSheet("color: #5f6368;")
        button = QPushButton()
        button.clicked.connect(self.download_selected_model)
        setattr(self, f"{scope}_model_combo", combo)
        setattr(self, f"{scope}_model_status", status)
        setattr(self, f"{scope}_model_download", button)
        row = QHBoxLayout()
        row.addWidget(combo)
        row.addWidget(button)
        row.addWidget(status)
        row.addStretch(1)
        return row

    def _enhancer_selected(self, scope: str) -> None:
        combo = getattr(self, f"{scope}_model_combo")
        self.values["enhancer"] = str(combo.currentData())
        self._save_values()
        self._refresh_settings_rows()

    def _model_entry(self):
        """``(entry, available)`` for the selected model; entry is None for 'none'."""
        from offline import face_beauty_engine as engine

        name = str(self.values.get("enhancer", "gpen_bfr_256"))
        if engine.normalize_enhancer(name) == engine.ENHANCER_NONE:
            return None, True
        entry = engine.ENHANCER_MODELS.get(name)
        if entry is None:
            return None, True
        return entry, engine.model_available(entry)

    def download_selected_model(self) -> None:
        """Fetch just the selected restoration model (mirror-aware)."""
        from offline import face_beauty_engine as engine
        from ui.widgets.model_download_dialog import DownloadItem, ModelDownloadDialog

        entry, available = self._model_entry()
        if entry is None or available:
            return
        name, dest, urls = engine.download_target(entry, getattr(self.i18n, "language", None))
        ModelDownloadDialog(self.i18n, [DownloadItem(label=name, dest=dest, urls=urls)], self).exec()
        self._refresh_settings_rows()

    def _trt_row(self, scope: str) -> QHBoxLayout:
        """Configure button + cache status, mirroring the TensorRT row on the
        offline passthrough page (ui/pages/offline_page.py:_trt_cache_row)."""
        button = QPushButton()
        button.clicked.connect(self.show_trt_dialog)
        status = QLabel()
        status.setStyleSheet("color: #5f6368;")
        setattr(self, f"{scope}_trt_configure_button", button)
        setattr(self, f"{scope}_trt_status_label", status)
        row = QHBoxLayout()
        row.addWidget(button)
        row.addWidget(status)
        row.addStretch(1)
        return row

    def _preset_selected(self, scope: str) -> None:
        """Applying a preset overwrites only its own sliders; detection and
        quality settings the user tuned are theirs and stay put."""
        from offline.face_beauty_engine import preset_percentages

        combo = getattr(self, f"{scope}_preset_combo")
        name = str(combo.currentData() or "")
        if name == _PRESET_CUSTOM:
            # "Custom" is a state, not a choice: offer the knobs that produce it.
            self.show_tuning_dialog()
            return
        self.values.update(preset_percentages(name))
        self._save_values()
        self._refresh_settings_rows()

    def _refresh_settings_rows(self) -> None:
        """Push :attr:`values` into both tabs' preset combo, summary and
        TensorRT status."""
        from ui.widgets.face_beauty_trt_dialog import face_beauty_trt_status

        preset = self.current_preset()
        enhancer = str(self.values.get("enhancer", "gfpgan_1.4"))
        enhancer_text = "-" if enhancer == "none" else f"{enhancer} {self.values.get('enhancer_blend', 0)}%"
        summary_text = self.i18n.t("beauty.summary").format(
            enhancer=enhancer_text,
            smooth=self.values.get("skin_smooth", 0),
            brighten=self.values.get("skin_brighten", 0),
            sharpen=self.values.get("sharpen", 0),
        )
        # The engine set depends on the options, so the status has to follow them.
        status_text = self.i18n.t("trt.status_" + face_beauty_trt_status(self._trt_cache_keys()))
        entry, available = self._model_entry()
        needs_download = entry is not None and not available
        model_status = self.i18n.t("beauty.model_ready" if not needs_download else "beauty.model_missing")

        for scope in ("single", "batch"):
            combo = getattr(self, f"{scope}_preset_combo", None)
            if combo is None:
                continue
            index = combo.findData(preset)
            if index >= 0 and combo.currentIndex() != index:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
            getattr(self, f"{scope}_preset_summary").setText(summary_text)
            model_combo = getattr(self, f"{scope}_model_combo", None)
            if model_combo is not None:
                index = model_combo.findData(enhancer)
                if index >= 0 and model_combo.currentIndex() != index:
                    model_combo.blockSignals(True)
                    model_combo.setCurrentIndex(index)
                    model_combo.blockSignals(False)
                getattr(self, f"{scope}_model_status").setText(model_status)
                getattr(self, f"{scope}_model_download").setVisible(needs_download)
            label = getattr(self, f"{scope}_trt_status_label", None)
            if label is not None:
                label.setText(f"{self.i18n.t('trt.cache_status')}: {status_text}")

    def show_tuning_dialog(self) -> None:
        from offline.face_beauty_engine import preset_percentages
        from ui.widgets.face_beauty_tuning_dialog import FaceBeautyTuningDialog

        # "Reset to preset" restores the last named preset, or standard when the
        # current values match none of them.
        preset = self.current_preset()
        defaults = preset_percentages("standard" if preset == _PRESET_CUSTOM else preset)
        dialog = FaceBeautyTuningDialog(self.i18n, self.values, defaults, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.values = dialog.values
            self._save_values()
        self._refresh_settings_rows()

    def _duration_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", 15.0)
        combo.addItem("", 30.0)
        combo.addItem("", 60.0)
        combo.addItem("", "custom")
        combo.addItem("", "custom_end")
        combo.addItem("", 0.0)
        return combo

    def _time_mode_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", "range")
        combo.addItem("", "segments")
        combo.currentIndexChanged.connect(self._update_time_mode_visibility)
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
        self.single_segments_config_button = QPushButton()
        self.single_segments_label = QLabel()
        self.single_segments_label.setStyleSheet("color: #5f6368;")
        self.single_segments_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.single_duration.currentIndexChanged.connect(self._update_custom_duration_visibility)
        self.single_segments_config_button.clicked.connect(self.show_time_segments_dialog)
        row.addWidget(self.single_start)
        row.addSpacing(12)
        row.addWidget(self.single_duration)
        row.addWidget(self.single_custom_minutes_label)
        row.addWidget(self.single_custom_minutes)
        row.addWidget(self.single_custom_end_label)
        row.addWidget(self.single_custom_end)
        row.addWidget(self.single_segments_config_button)
        row.addWidget(self.single_segments_label, 1)
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
        self.single_time_mode = self._time_mode_combo()
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
        self.single_labels = {key: _label() for key in ("video", "output", "model", "beauty", "trt")}
        grid.addWidget(self.single_labels["video"], 0, 0)
        grid.addLayout(row_video, 0, 1)
        grid.addWidget(self.single_labels["output"], 1, 0)
        grid.addLayout(row_out, 1, 1)
        grid.addWidget(self.single_labels["model"], 2, 0)
        grid.addLayout(self._model_row("single"), 2, 1)
        grid.addWidget(self.single_labels["beauty"], 3, 0, alignment=Qt.AlignTop)
        grid.addLayout(self._beauty_settings_cell("single"), 3, 1)
        grid.addWidget(self.single_labels["trt"], 4, 0)
        grid.addLayout(self._trt_row("single"), 4, 1)
        grid.addWidget(self.single_time_mode, 5, 0, alignment=Qt.AlignRight)
        grid.addLayout(self._time_row(), 5, 1)
        grid.addWidget(self.single_skip, 6, 1)
        grid.addLayout(actions, 7, 1)
        grid.setRowStretch(8, 1)
        self.tabs.addTab(page, "")
        self._update_custom_duration_visibility()
        self._update_time_mode_visibility()
        self._update_time_segments_label()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit()
        browse_dir = QPushButton("...")
        browse_dir.clicked.connect(lambda: self._browse_dir(self.batch_dir))
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
        self.batch_labels = {key: _label() for key in ("directory", "model", "beauty", "trt")}
        grid.addWidget(self.batch_labels["directory"], 0, 0)
        grid.addLayout(row_dir, 0, 1)
        grid.addWidget(self.batch_labels["model"], 1, 0)
        grid.addLayout(self._model_row("batch"), 1, 1)
        grid.addWidget(self.batch_labels["beauty"], 2, 0, alignment=Qt.AlignTop)
        grid.addLayout(self._beauty_settings_cell("batch"), 2, 1)
        grid.addWidget(self.batch_labels["trt"], 3, 0)
        grid.addLayout(self._trt_row("batch"), 3, 1)
        grid.addWidget(self.batch_recursive, 4, 1)
        grid.addWidget(self.batch_skip, 5, 1)
        grid.addLayout(actions, 6, 1)
        grid.setRowStretch(7, 1)
        self.tabs.addTab(page, "")

    # -- browsing ------------------------------------------------------------

    def _browse_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.i18n.t("file.select_video"), "",
                                              "Videos (*.mp4 *.mkv *.mov *.m4v)")
        if path:
            target.setText(path)
            self.single_out_dir.setText(str(Path(path).parent))

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, self.i18n.t("file.select_directory"))
        if path:
            target.setText(path)

    # -- time mode -----------------------------------------------------------

    def _update_custom_duration_visibility(self) -> None:
        if str(self.single_time_mode.currentData()) == "segments":
            return
        value = self.single_duration.currentData()
        minutes_visible = value == "custom"
        end_visible = value == "custom_end"
        self.single_custom_minutes_label.setVisible(minutes_visible)
        self.single_custom_minutes.setVisible(minutes_visible)
        self.single_custom_end_label.setVisible(end_visible)
        self.single_custom_end.setVisible(end_visible)

    def _update_time_mode_visibility(self) -> None:
        segments_mode = str(self.single_time_mode.currentData()) == "segments"
        for widget in (
            self.single_start,
            self.single_duration,
            self.single_custom_minutes_label,
            self.single_custom_minutes,
            self.single_custom_end_label,
            self.single_custom_end,
        ):
            widget.setVisible(not segments_mode)
        self.single_segments_config_button.setVisible(segments_mode)
        self.single_segments_label.setVisible(segments_mode)
        if not segments_mode:
            self._update_custom_duration_visibility()

    def _time_segments_label_text(self) -> str:
        if not self.single_time_segments:
            return self.i18n.t("offline.time_segments_none")
        ranges = ", ".join(
            f"{_format_time_seconds(start)}-{_format_time_seconds(end)}"
            for start, end in self.single_time_segments[:2]
        )
        if len(self.single_time_segments) > 2:
            ranges = f"{ranges}, +{len(self.single_time_segments) - 2}"
        return self.i18n.t("offline.time_segments_summary").format(
            count=len(self.single_time_segments), ranges=ranges
        )

    def _update_time_segments_label(self) -> None:
        text = self._time_segments_label_text()
        tooltip = "\n".join(
            f"{_format_time_seconds(start)} - {_format_time_seconds(end)}"
            for start, end in self.single_time_segments
        )
        self.single_segments_label.setText(text)
        self.single_segments_label.setToolTip(tooltip)

    # -- validation ----------------------------------------------------------

    def _show_time_error(self, key: str, video_duration: float = 0.0, row: int = 0) -> None:
        message = self.i18n.t(key).format(duration=_format_time_seconds(video_duration), row=row)
        QMessageBox.warning(self, self.i18n.t("offline.time_error_title"), message)

    def _validated_single_video_duration(self) -> float | None:
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
        return video_duration

    def _validated_single_time_range(self) -> tuple[float, float] | None:
        video_duration = self._validated_single_video_duration()
        if video_duration is None:
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

    def _validated_single_time_segments(self) -> list[tuple[float, float]] | None:
        video_duration = self._validated_single_video_duration()
        if video_duration is None:
            return None
        segments, error_key, row = _resolve_time_segments(self.single_time_segments, video_duration)
        if error_key:
            self._show_time_error(error_key, video_duration, row)
            return None
        return segments

    def show_time_segments_dialog(self) -> None:
        video_duration = self._validated_single_video_duration()
        if video_duration is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.i18n.t("offline.time_segments_dialog_title"))
        layout = QVBoxLayout(dialog)
        table = QTableWidget(dialog)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(
            [self.i18n.t("offline.time_segments_start"), self.i18n.t("offline.time_segments_end")]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setMinimumWidth(320)

        def append_row(start: float = 0.0, end: float = 0.0) -> None:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(_format_time_seconds(start)))
            table.setItem(row, 1, QTableWidgetItem(_format_time_seconds(end)))

        for start, end in (self.single_time_segments or [(0.0, min(300.0, video_duration))]):
            append_row(start, end)

        buttons = QHBoxLayout()
        add_button = QPushButton(self.i18n.t("button.add"))
        remove_button = QPushButton(self.i18n.t("button.remove"))
        save_button = QPushButton(self.i18n.t("button.save"))
        close_button = QPushButton(self.i18n.t("button.close"))
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        buttons.addWidget(save_button)
        buttons.addWidget(close_button)

        def add_segment() -> None:
            if table.rowCount() > 0:
                prev_item = table.item(table.rowCount() - 1, 1)
                previous_end = _parse_hhmmss_text(prev_item.text() if prev_item else "") or 0.0
            else:
                previous_end = 0.0
            start = min(previous_end, video_duration)
            append_row(start, min(start + 300.0, video_duration))

        def remove_segment() -> None:
            row = table.currentRow()
            if row < 0:
                row = table.rowCount() - 1
            if row >= 0:
                table.removeRow(row)

        def save_segments() -> None:
            raw: list[tuple[float | None, float | None]] = []
            for row in range(table.rowCount()):
                s_item = table.item(row, 0)
                e_item = table.item(row, 1)
                raw.append((
                    _parse_hhmmss_text(s_item.text() if s_item else ""),
                    _parse_hhmmss_text(e_item.text() if e_item else ""),
                ))
            segments, error_key, row = _resolve_time_segments(raw, video_duration)
            if error_key:
                self._show_time_error(error_key, video_duration, row)
                return
            self.single_time_segments = segments
            self.settings.data[_SETTINGS_TIME_SEGMENTS_KEY] = _serialize_time_segments(segments)
            self.settings.save()
            self._update_time_segments_label()
            dialog.accept()

        add_button.clicked.connect(add_segment)
        remove_button.clicked.connect(remove_segment)
        save_button.clicked.connect(save_segments)
        close_button.clicked.connect(dialog.reject)
        layout.addWidget(table)
        layout.addLayout(buttons)
        dialog.exec()

    # -- common args ---------------------------------------------------------

    def _model_args(self) -> list[str]:
        """The flags that select which graphs load -- shared by the run and the
        TensorRT prebuild so both hit the same engine-cache keys.  The parser
        also loads when any parse-gated retouch slider is non-zero, so those
        sliders are part of the model selection too."""
        args = ["--enhancer", str(self.values["enhancer"])]
        args.append("--region-mask" if self.values["region_mask"] else "--no-region-mask")
        args.append("--landmarker" if self.values["landmarker"] else "--no-landmarker")
        for key in _PARSER_GATED_KEYS:
            args += [_SLIDER_FLAGS[key], str(int(self.values[key]))]
        return args

    def _render_args(self) -> list[str]:
        args = self._model_args()
        for key, flag in _SLIDER_FLAGS.items():
            if key in _PARSER_GATED_KEYS:
                continue      # already emitted by _model_args
            args += [flag, str(int(self.values[key]))]
        padding = str(int(self.values["mask_padding"]))
        args += [
            "--mask-padding-top", padding,
            "--mask-padding-right", padding,
            "--mask-padding-bottom", padding,
            "--mask-padding-left", padding,
            "--min-face-mode", str(self.values["min_face_mode"]),
            "--detect-mode", str(self.values["detect_mode"]),
            "--vr-reproject", str(self.values["vr_reproject"]),
            "--detect-interval", str(int(self.values["detect_interval"])),
            "--detect-roi" if self.values["detect_roi"] else "--no-detect-roi",
            "--detector-score", f"{int(self.values['detector_score']) / 100.0:.2f}",
            "--max-faces", str(int(self.values["max_faces"])),
            "--max-side", str(int(self.values["max_side"])),
            "--provider", self._provider,
        ]
        return args

    # -- preflight -----------------------------------------------------------

    def _needs_parser(self) -> bool:
        return bool(self.values["region_mask"]) or any(
            int(self.values[key]) > 0 for key in _PARSER_GATED_KEYS
        )

    def _trt_cache_keys(self) -> list[str]:
        from offline.face_beauty_engine import BeautyOptions, trt_cache_keys

        options = BeautyOptions(
            enhancer=str(self.values["enhancer"]),
            use_region_mask=bool(self.values["region_mask"]),
            use_landmarker=bool(self.values["landmarker"]),
            **{key: int(self.values[key]) / 100.0 for key in _PARSER_GATED_KEYS},
        )
        return trt_cache_keys(options)

    def _preflight_models(self) -> bool:
        """Download any missing ONNX files through the shared progress dialog."""
        from offline import face_beauty_engine as engine
        from ui.widgets.model_download_dialog import DownloadItem, ModelDownloadDialog

        language = getattr(self.i18n, "language", None)
        items: list[DownloadItem] = []
        for _label_name, entry in engine.required_models(
            str(self.values["enhancer"]), bool(self.values["landmarker"]), self._needs_parser()
        ):
            if engine.model_available(entry):
                continue
            name, dest, urls = engine.download_target(entry, language)
            items.append(DownloadItem(label=name, dest=dest, urls=urls))
        if not items:
            return True
        dialog = ModelDownloadDialog(self.i18n, items, self)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _preflight_trt(self) -> bool:
        """The TensorRT engines take minutes to compile the first time, so the
        build gets its own progress dialog instead of happening invisibly inside
        the conversion.

        A cancelled or failed build is not fatal: the run can still go ahead on
        the CUDA provider, just slower."""
        from ui.widgets.face_beauty_trt_dialog import FaceBeautyTrtConfigDialog, face_beauty_trt_status

        self._provider = "trt"
        keys = self._trt_cache_keys()
        if face_beauty_trt_status(keys) == "ready":
            return True
        dialog = FaceBeautyTrtConfigDialog(self.i18n, keys, self._model_args(), self)
        dialog.exec()
        if dialog.is_ready():
            return True
        answer = QMessageBox.question(
            self,
            self.i18n.t("trt.title"),
            self.i18n.t("beauty.trt_skipped"),
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._provider = "cuda"
        return True

    def show_trt_dialog(self) -> None:
        from ui.widgets.face_beauty_trt_dialog import FaceBeautyTrtConfigDialog

        FaceBeautyTrtConfigDialog(self.i18n, self._trt_cache_keys(), self._model_args(), self).exec()

    def _preflight(self) -> bool:
        return self._preflight_models() and self._preflight_trt()

    # -- run -----------------------------------------------------------------

    def run_single(self) -> None:
        video = self.single_video.text().strip()
        if not video or not Path(video).is_file():
            self._show_time_error("offline.time_error_video_missing")
            return
        if not self._preflight():
            return
        args = ["single", video, *self._render_args()]
        if str(self.single_time_mode.currentData()) == "segments":
            segments = self._validated_single_time_segments()
            if segments is None:
                return
            for start_seconds, end_seconds in segments:
                args.extend(["--segment",
                             f"{_format_time_seconds(start_seconds)}-{_format_time_seconds(end_seconds)}"])
        else:
            time_range = self._validated_single_time_range()
            if time_range is None:
                return
            start_seconds, duration_seconds = time_range
            args.extend(["--start", str(start_seconds), "--duration", str(duration_seconds)])
        if self.single_out_dir.text().strip():
            args.extend(["--out-dir", self.single_out_dir.text().strip()])
        if self.single_skip.isChecked():
            args.append("--skip-existing")
        self.settings.save()
        self.process.start(args, self.settings.server_env())

    def run_batch(self) -> None:
        directory = self.batch_dir.text().strip()
        if not directory or not Path(directory).is_dir():
            self._show_time_error("offline.time_error_video_missing")
            return
        if not self._preflight():
            return
        args = ["batch", directory, *self._render_args()]
        args.append("--recursive" if self.batch_recursive.isChecked() else "--no-recursive")
        if self.batch_skip.isChecked():
            args.append("--skip-existing")
        self.settings.save()
        self.process.start(args, self.settings.server_env())

    # -- state / logging -----------------------------------------------------

    def set_running(self, running: bool) -> None:
        self.start_single.setEnabled(not running)
        self.start_batch.setEnabled(not running)
        self.stop_single.setEnabled(running)
        self.stop_batch.setEnabled(running)
        self.single_time_mode.setEnabled(not running)
        self.single_segments_config_button.setEnabled(not running)
        for scope in ("single", "batch"):
            getattr(self, f"{scope}_preset_combo").setEnabled(not running)
            getattr(self, f"{scope}_tune_button").setEnabled(not running)
            getattr(self, f"{scope}_trt_configure_button").setEnabled(not running)
            getattr(self, f"{scope}_model_combo").setEnabled(not running)
            getattr(self, f"{scope}_model_download").setEnabled(not running)

    def append_log(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def sync_from_settings(self) -> None:
        self.values = self._load_values()
        self._refresh_settings_rows()

    # -- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("beauty.title"))
        self.back_button.setText(self.i18n.t("button.back"))
        for scope in ("single", "batch"):
            getattr(self, f"{scope}_tune_button").setText(self.i18n.t("beauty.tune"))
            getattr(self, f"{scope}_trt_configure_button").setText(self.i18n.t("trt.configure"))
            getattr(self, f"{scope}_model_download").setText(self.i18n.t("beauty.model_download"))
            combo = getattr(self, f"{scope}_model_combo")
            for index, name in enumerate(_ENHANCERS):
                combo.setItemText(index, self.i18n.t(_ENHANCER_LABEL_KEYS[name]))
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
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory"))
        for labels in (self.single_labels, self.batch_labels):
            labels["model"].setText(self.i18n.t("beauty.model"))
            labels["beauty"].setText(self.i18n.t("beauty.settings"))
            labels["trt"].setText(self.i18n.t("trt.row_label"))
        for scope in ("single", "batch"):
            combo = getattr(self, f"{scope}_preset_combo")
            for index, name in enumerate((*_PRESETS, _PRESET_CUSTOM)):
                combo.setItemText(index, self.i18n.t(_PRESET_LABEL_KEYS[name]))
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
        self.single_time_mode.setItemText(0, self.i18n.t("offline.time_mode_range"))
        self.single_time_mode.setItemText(1, self.i18n.t("offline.time_mode_segments"))
        self.single_segments_config_button.setText(self.i18n.t("offline.time_segments_configure"))
        self._refresh_settings_rows()
        self._update_time_segments_label()
        self._update_time_mode_visibility()
