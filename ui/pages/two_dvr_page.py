"""Offline 2D -> VR/3D page (DA3 depth based).

Sibling of :class:`ui.pages.offline_page.OfflinePage`. Drives ``offline/two_dvr.py``
(via :class:`ui.services.offline_process.TwoDvrProcess`) to turn flat 2D video into
SBS left/right VR using DA3 monocular depth. Reuses the offline page's time-range /
segment helpers so the single-clip timing UI matches.
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
    _parse_time_text,
    _resolve_time_range,
    _resolve_time_segments,
    _serialize_time_segments,
)
from ui.settings import quality_speed_preset, quality_speed_value
from utils.video_metadata import probe_video_metadata

_SETTINGS_TIME_SEGMENTS_KEY = "two_dvr_single_time_segments"
_UI_TWO_DVR_HOLE_FILL = "soft_shift"
_TWO_DVR_STRENGTH_OPTIONS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
# UI temporal-stability tiers. The single "nvds" entry uses the default NVDS
# resolution (512x288); the higher-quality 672x384 tier is reachable via the
# --nvds-res CLI flag but is intentionally not surfaced in the UI.
_TWO_DVR_DEPTH_STABILIZERS = ("default", "nvds")


class TwoDvrPage(QWidget):
    def __init__(self, i18n, settings, process) -> None:
        super().__init__()
        self.setObjectName("TwoDvrPage")
        self.setStyleSheet(
            "QWidget#TwoDvrPage, QWidget#TwoDvrPage QLabel, QWidget#TwoDvrPage QCheckBox { font-size: 9pt; }"
            "QWidget#TwoDvrPage QPushButton, QWidget#TwoDvrPage QLineEdit, QWidget#TwoDvrPage QComboBox, "
            "QWidget#TwoDvrPage QTextEdit, QWidget#TwoDvrPage QTabBar::tab { font-size: 9pt; padding: 3px 7px; }"
            "QWidget#TwoDvrPage QLabel#TwoDvrPageTitle { font-size: 14pt; font-weight: 700; }"
        )
        self.i18n = i18n
        self.settings = settings
        self.process = process
        self.single_time_segments = _coerce_time_segments(self.settings.data.get(_SETTINGS_TIME_SEGMENTS_KEY))

        self.title_label = QLabel()
        self.title_label.setObjectName("TwoDvrPageTitle")
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

    def _model_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", "base")
        combo.addItem("", "base_hd")
        combo.addItem("", "large_hd")
        combo.setCurrentIndex(0)
        return combo

    def _projection_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        combo.addItem("", "flat3d")
        combo.addItem("", "hequirect")
        combo.addItem("", "fisheye")
        combo.setCurrentIndex(0)
        return combo

    def _quality_speed_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in ("ultrafast", "medium", "veryslow"):
            combo.addItem("", value)
        idx = combo.findData(quality_speed_value(self.settings.data.get("offline_quality_speed"), "medium"))
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_quality_speed)
        return combo

    def _temporal_stability_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in _TWO_DVR_DEPTH_STABILIZERS:
            combo.addItem("", value)
        current = str(self.settings.data.get("two_dvr_depth_stabilizer") or "default").strip().lower()
        idx = combo.findData(current if current in _TWO_DVR_DEPTH_STABILIZERS else "default")
        combo.setCurrentIndex(max(0, idx))
        combo.currentIndexChanged.connect(self._save_depth_stabilizer)
        return combo

    def _strength_combo(self) -> QComboBox:
        combo = _fit_combo(QComboBox())
        for value in _TWO_DVR_STRENGTH_OPTIONS:
            combo.addItem("", value)
        combo.setCurrentIndex(2)
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
        self.single_model = self._model_combo()
        self.single_projection = self._projection_combo()
        self.single_strength = self._strength_combo()
        self.single_quality_speed = self._quality_speed_combo()
        self.single_depth_stabilizer = self._temporal_stability_combo()
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
        self.single_labels = {
            key: _label() for key in ("video", "output", "model", "projection", "strength", "performance", "stability")
        }
        grid.addWidget(self.single_labels["video"], 0, 0)
        grid.addLayout(row_video, 0, 1)
        grid.addWidget(self.single_labels["output"], 1, 0)
        grid.addLayout(row_out, 1, 1)
        grid.addWidget(self.single_labels["model"], 2, 0)
        grid.addWidget(self.single_model, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["projection"], 3, 0)
        grid.addWidget(self.single_projection, 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["strength"], 4, 0)
        grid.addWidget(self.single_strength, 4, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["performance"], 5, 0)
        grid.addWidget(self.single_quality_speed, 5, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_labels["stability"], 6, 0)
        grid.addWidget(self.single_depth_stabilizer, 6, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.single_time_mode, 7, 0, alignment=Qt.AlignRight)
        grid.addLayout(self._time_row(), 7, 1)
        grid.addWidget(self.single_skip, 8, 1)
        grid.addLayout(actions, 9, 1)
        self.tabs.addTab(page, "")
        self._update_custom_duration_visibility()
        self._update_time_mode_visibility()
        self._update_time_segments_label()

    def _batch_tab(self) -> None:
        page = QWidget()
        self.batch_dir = QLineEdit()
        browse_dir = QPushButton("...")
        browse_dir.clicked.connect(lambda: self._browse_dir(self.batch_dir))
        self.batch_model = self._model_combo()
        self.batch_projection = self._projection_combo()
        self.batch_strength = self._strength_combo()
        self.batch_quality_speed = self._quality_speed_combo()
        self.batch_depth_stabilizer = self._temporal_stability_combo()
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
        self.batch_labels = {
            key: _label() for key in ("directory", "model", "projection", "strength", "performance", "stability")
        }
        grid.addWidget(self.batch_labels["directory"], 0, 0)
        grid.addLayout(row_dir, 0, 1)
        grid.addWidget(self.batch_labels["model"], 1, 0)
        grid.addWidget(self.batch_model, 1, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["projection"], 2, 0)
        grid.addWidget(self.batch_projection, 2, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["strength"], 3, 0)
        grid.addWidget(self.batch_strength, 3, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["performance"], 4, 0)
        grid.addWidget(self.batch_quality_speed, 4, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_labels["stability"], 5, 0)
        grid.addWidget(self.batch_depth_stabilizer, 5, 1, alignment=Qt.AlignLeft)
        grid.addWidget(self.batch_recursive, 6, 1)
        grid.addWidget(self.batch_skip, 7, 1)
        grid.addLayout(actions, 8, 1)
        self.tabs.addTab(page, "")

    # -- TensorRT acceleration ----------------------------------------------

    @staticmethod
    def _trt_provider() -> str:
        return "trt"

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
        return self.i18n.t("offline.time_segments_summary").format(count=len(self.single_time_segments), ranges=ranges)

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

    def _strength_value(self, combo: QComboBox) -> str:
        try:
            value = float(combo.currentData())
        except (TypeError, ValueError):
            value = 1.0
        return f"{value:.2f}"

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

    def _render_args(
        self,
        model: QComboBox,
        projection: QComboBox,
        strength: QComboBox,
        depth_stabilizer: QComboBox,
    ) -> list[str]:
        preset = quality_speed_preset(self.settings.data.get("offline_quality_speed"), "medium")
        args = [
            "--model", str(model.currentData()),
            "--projection", str(projection.currentData()),
            "--hole-fill", _UI_TWO_DVR_HOLE_FILL,
            "--strength", self._strength_value(strength),
            "--max-side", "0",
            "--preset", preset.lower(),
            "--provider", self._trt_provider(),
        ]
        stabilizer = "nvds" if str(depth_stabilizer.currentData() or "default") == "nvds" else "default"
        args += ["--depth-stabilizer", stabilizer]
        return args

    def _preflight_models(self, model_combo: QComboBox, stabilizer_combo: QComboBox) -> bool:
        """Ensure the selected DA3 model (and NVDS files, if chosen) are present.

        Missing files trigger a confirm + progress download dialog. Returns False
        when the user cancels or a download fails, so the run is aborted.
        """
        from offline import da3_depth
        from offline import nvds_stabilizer as nvds
        from ui.widgets.model_download_dialog import DownloadItem, ModelDownloadDialog

        language = getattr(self.i18n, "language", None)
        items: list[DownloadItem] = []

        model_key = str(model_combo.currentData() or "base")
        if not da3_depth.model_available(model_key):
            name, dest, urls = da3_depth.download_target(model_key, language)
            items.append(DownloadItem(label=name, dest=dest, urls=urls))

        if str(stabilizer_combo.currentData() or "default") == "nvds":
            width, height = nvds.resolve_resolution(nvds.NVDS_DEFAULT_RES)
            for name, dest, urls in nvds.download_targets(width, height, language):
                items.append(DownloadItem(label=name, dest=dest, urls=urls))

        if not items:
            return True
        dialog = ModelDownloadDialog(self.i18n, items, self)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def run_single(self) -> None:
        video = self.single_video.text().strip()
        if not video or not Path(video).is_file():
            self._show_time_error("offline.time_error_video_missing")
            return
        if not self._preflight_models(self.single_model, self.single_depth_stabilizer):
            return
        args = ["single", video]
        args.extend(self._render_args(
            self.single_model, self.single_projection, self.single_strength, self.single_depth_stabilizer,
        ))
        if str(self.single_time_mode.currentData()) == "segments":
            segments = self._validated_single_time_segments()
            if segments is None:
                return
            for start_seconds, end_seconds in segments:
                args.extend(["--segment", f"{_format_time_seconds(start_seconds)}-{_format_time_seconds(end_seconds)}"])
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
        env = self.settings.server_env()
        env["PT_PASSTHROUGH_PYNV_PRESET"] = quality_speed_preset(
            self.settings.data.get("offline_quality_speed"), "medium"
        )
        self.process.start(args, env)

    def run_batch(self) -> None:
        directory = self.batch_dir.text().strip()
        if not directory or not Path(directory).is_dir():
            self._show_time_error("offline.time_error_video_missing")
            return
        if not self._preflight_models(self.batch_model, self.batch_depth_stabilizer):
            return
        args = ["batch", directory]
        args.extend(self._render_args(
            self.batch_model, self.batch_projection, self.batch_strength, self.batch_depth_stabilizer,
        ))
        args.append("--recursive" if self.batch_recursive.isChecked() else "--no-recursive")
        if self.batch_skip.isChecked():
            args.append("--skip-existing")
        self.settings.save()
        env = self.settings.server_env()
        env["PT_PASSTHROUGH_PYNV_PRESET"] = quality_speed_preset(
            self.settings.data.get("offline_quality_speed"), "medium"
        )
        self.process.start(args, env)

    # -- state / logging -----------------------------------------------------

    def set_running(self, running: bool) -> None:
        self.start_single.setEnabled(not running)
        self.start_batch.setEnabled(not running)
        self.stop_single.setEnabled(running)
        self.stop_batch.setEnabled(running)
        self.single_time_mode.setEnabled(not running)
        self.single_segments_config_button.setEnabled(not running)

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
        stabilizer = str(self.settings.data.get("two_dvr_depth_stabilizer") or "default").strip().lower()
        if stabilizer not in _TWO_DVR_DEPTH_STABILIZERS:
            stabilizer = "default"
        for combo in (getattr(self, "single_depth_stabilizer", None), getattr(self, "batch_depth_stabilizer", None)):
            if isinstance(combo, QComboBox):
                idx = combo.findData(stabilizer)
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

    def _save_depth_stabilizer(self) -> None:
        sender = self.sender()
        if isinstance(sender, QComboBox):
            value = str(sender.currentData() or "default").strip().lower()
            if value not in _TWO_DVR_DEPTH_STABILIZERS:
                value = "default"
            self.settings.data["two_dvr_depth_stabilizer"] = value
            for combo in (getattr(self, "single_depth_stabilizer", None), getattr(self, "batch_depth_stabilizer", None)):
                if isinstance(combo, QComboBox) and combo is not sender:
                    idx = combo.findData(value)
                    if idx >= 0 and combo.currentIndex() != idx:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(idx)
                        combo.blockSignals(False)
            self.settings.save()

    # -- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        self.title_label.setText(self.i18n.t("button.two_dvr"))
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
        self.single_labels["model"].setText(self.i18n.t("twodvr.model"))
        self.single_labels["projection"].setText(self.i18n.t("twodvr.projection"))
        self.single_labels["strength"].setText(self.i18n.t("twodvr.strength"))
        self.single_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        self.single_labels["stability"].setText(self.i18n.t("twodvr.temporal_stability"))
        self.batch_labels["directory"].setText(self.i18n.t("offline.directory"))
        self.batch_labels["model"].setText(self.i18n.t("twodvr.model"))
        self.batch_labels["projection"].setText(self.i18n.t("twodvr.projection"))
        self.batch_labels["strength"].setText(self.i18n.t("twodvr.strength"))
        self.batch_labels["performance"].setText(self.i18n.t("performance.quality_speed"))
        self.batch_labels["stability"].setText(self.i18n.t("twodvr.temporal_stability"))
        for combo in (self.single_model, self.batch_model):
            combo.setItemText(0, self.i18n.t("twodvr.model_base"))
            combo.setItemText(1, self.i18n.t("twodvr.model_base_hd"))
            combo.setItemText(2, self.i18n.t("twodvr.model_large_hd"))
        for combo in (self.single_projection, self.batch_projection):
            combo.setItemText(0, self.i18n.t("twodvr.projection_flat3d"))
            combo.setItemText(1, self.i18n.t("twodvr.projection_hequirect"))
            combo.setItemText(2, self.i18n.t("twodvr.projection_fisheye"))
        for combo in (self.single_quality_speed, self.batch_quality_speed):
            for index, key in enumerate(("quality_speed.ultrafast", "quality_speed.medium", "quality_speed.veryslow")):
                combo.setItemText(index, self.i18n.t(key))
        for combo in (self.single_depth_stabilizer, self.batch_depth_stabilizer):
            combo.setItemText(0, self.i18n.t("twodvr.temporal_default"))
            combo.setItemText(1, self.i18n.t("twodvr.temporal_nvds"))
        for combo in (self.single_strength, self.batch_strength):
            for index, value in enumerate(_TWO_DVR_STRENGTH_OPTIONS):
                combo.setItemText(index, f"{int(round(value * 100))}%")
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
        self._update_time_segments_label()
        self._update_time_mode_visibility()
