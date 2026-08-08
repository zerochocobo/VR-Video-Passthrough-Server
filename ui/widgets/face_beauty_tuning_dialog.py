"""Fine-tuning dialog for the offline face-beauty page.

The page itself only offers a preset dropdown -- everything else lives here, so
the common case stays a single choice and the twenty-odd knobs are one click
away for people who want them.

Works on a plain ``dict`` of values (percent ints, booleans, enum strings) so
the page can persist it verbatim in the UI settings file and hand the same shape
to :func:`ui.pages.face_beauty_page.FaceBeautyPage._render_args`.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

LABEL_WIDTH = 132

VR_MODES = ("auto", "on", "off")
VR_MODE_LABEL_KEYS = ("beauty.vr_auto", "beauty.vr_on", "beauty.vr_off")
DETECT_MODES = ("auto", "full", "tiled")
DETECT_MODE_LABEL_KEYS = ("beauty.detect_auto", "beauty.detect_full", "beauty.detect_tiled")
MIN_FACE_MODES = ("auto", "loose", "strict")
MIN_FACE_LABEL_KEYS = ("beauty.min_face_auto", "beauty.min_face_loose", "beauty.min_face_strict")
MAX_SIDE_OPTIONS = (0, 3840, 1920)
MAX_SIDE_LABEL_KEYS = ("beauty.max_side_original", "beauty.max_side_4k", "beauty.max_side_1080")

# Sliders driven by the preset; the dialog's "reset to preset" button restores
# exactly these. Order is the display order.
PRESET_SLIDERS = (
    ("enhancer_blend", "beauty.enhancer_blend"),
    ("skin_smooth", "beauty.skin_smooth"),
    ("skin_brighten", "beauty.skin_brighten"),
    ("skin_even", "beauty.skin_even"),
    ("eye_brighten", "beauty.eye_brighten"),
    ("teeth_white", "beauty.teeth_white"),
    ("lip_vivid", "beauty.lip_vivid"),
    ("sharpen", "beauty.sharpen"),
)
# Quality/stability sliders that are deliberately not part of any preset.
QUALITY_SLIDERS = (
    ("mask_blur", "beauty.mask_blur"),
    ("temporal_smooth", "beauty.temporal_smooth"),
)


class PercentSlider(QWidget):
    """0-100 slider with a live value label; the CLI takes the same units."""

    def __init__(self, default: int) -> None:
        super().__init__()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(default))
        self.slider.setMinimumWidth(200)
        self.value_label = QLabel(f"{int(default)}%")
        self.value_label.setFixedWidth(42)
        self.value_label.setStyleSheet("color: #5f6368;")
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(f"{v}%"))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

    def value(self) -> int:
        return int(self.slider.value())

    def set_value(self, value: int) -> None:
        self.slider.setValue(int(value))


class FaceBeautyTuningDialog(QDialog):
    """Edits a copy of ``values``; read :attr:`values` after ``exec()`` returns
    ``Accepted``."""

    def __init__(self, i18n, values: dict, preset_defaults: dict[str, int], parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.values = dict(values)
        self._preset_defaults = dict(preset_defaults)
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("beauty.tuning_title"))

        self.sliders: dict[str, PercentSlider] = {}
        self.labels: dict[str, QLabel] = {}

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(4, 4, 4, 4)
        body_layout.setSpacing(10)
        body_layout.addWidget(self._beauty_group())
        body_layout.addWidget(self._quality_group())
        body_layout.addWidget(self._detection_group())
        body_layout.addStretch(1)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(body)

        self.reset_button = QPushButton(self.i18n.t("beauty.reset_to_preset"))
        self.reset_button.clicked.connect(self._reset_to_preset)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self.i18n.t("button.save"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.close"))
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)

        footer = QHBoxLayout()
        footer.addWidget(self.reset_button)
        footer.addStretch(1)
        footer.addWidget(self.buttons)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.addWidget(area, 1)
        layout.addLayout(footer)
        self.setMinimumWidth(560)
        self.resize(560, 640)

    # -- groups --------------------------------------------------------------

    def _grid(self, box: QGroupBox) -> QGridLayout:
        grid = QGridLayout(box)
        grid.setColumnMinimumWidth(0, LABEL_WIDTH)
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(6)
        return grid

    def _add_row(self, grid: QGridLayout, row: int, key: str, widget: QWidget, label_key: str) -> int:
        label = QLabel(self.i18n.t(label_key))
        label.setWordWrap(True)
        self.labels[key] = label
        grid.addWidget(label, row, 0)
        grid.addWidget(widget, row, 1)
        return row + 1

    def _beauty_group(self) -> QGroupBox:
        box = QGroupBox(self.i18n.t("beauty.group_beauty"))
        grid = self._grid(box)
        row = 0
        for key, label_key in PRESET_SLIDERS:
            slider = PercentSlider(int(self.values.get(key, 0)))
            self.sliders[key] = slider
            row = self._add_row(grid, row, key, slider, label_key)
        return box

    def _quality_group(self) -> QGroupBox:
        box = QGroupBox(self.i18n.t("beauty.group_quality"))
        grid = self._grid(box)
        row = 0
        for key, label_key in QUALITY_SLIDERS:
            slider = PercentSlider(int(self.values.get(key, 0)))
            self.sliders[key] = slider
            row = self._add_row(grid, row, key, slider, label_key)
        self.mask_padding = QSpinBox()
        self.mask_padding.setRange(0, 40)
        self.mask_padding.setValue(int(self.values.get("mask_padding", 0)))
        self.mask_padding.setSuffix("%")
        self.mask_padding.setFixedWidth(90)
        row = self._add_row(grid, row, "mask_padding", self.mask_padding, "beauty.mask_padding")
        self.region_mask = QCheckBox(self.i18n.t("beauty.region_mask_hint"))
        self.region_mask.setChecked(bool(self.values.get("region_mask", True)))
        row = self._add_row(grid, row, "region_mask", self.region_mask, "beauty.region_mask")
        self.max_side = QComboBox()
        for index, value in enumerate(MAX_SIDE_OPTIONS):
            self.max_side.addItem(self.i18n.t(MAX_SIDE_LABEL_KEYS[index]), value)
        index = self.max_side.findData(int(self.values.get("max_side", 0)))
        self.max_side.setCurrentIndex(max(0, index))
        row = self._add_row(grid, row, "max_side", self.max_side, "beauty.max_side")
        return box

    def _detection_group(self) -> QGroupBox:
        box = QGroupBox(self.i18n.t("beauty.group_detection"))
        grid = self._grid(box)
        row = 0
        self.min_face_mode = QComboBox()
        for index, mode in enumerate(MIN_FACE_MODES):
            self.min_face_mode.addItem(self.i18n.t(MIN_FACE_LABEL_KEYS[index]), mode)
        index = self.min_face_mode.findData(str(self.values.get("min_face_mode", "auto")))
        self.min_face_mode.setCurrentIndex(max(0, index))
        row = self._add_row(grid, row, "min_face_mode", self.min_face_mode, "beauty.min_face")

        self.vr_reproject = QComboBox()
        for index, mode in enumerate(VR_MODES):
            self.vr_reproject.addItem(self.i18n.t(VR_MODE_LABEL_KEYS[index]), mode)
        index = self.vr_reproject.findData(str(self.values.get("vr_reproject", "auto")))
        self.vr_reproject.setCurrentIndex(max(0, index))
        row = self._add_row(grid, row, "vr_reproject", self.vr_reproject, "beauty.vr_reproject")

        self.detect_mode = QComboBox()
        for index, mode in enumerate(DETECT_MODES):
            self.detect_mode.addItem(self.i18n.t(DETECT_MODE_LABEL_KEYS[index]), mode)
        index = self.detect_mode.findData(str(self.values.get("detect_mode", "auto")))
        self.detect_mode.setCurrentIndex(max(0, index))
        row = self._add_row(grid, row, "detect_mode", self.detect_mode, "beauty.detect_mode")

        self.detect_roi = QCheckBox(self.i18n.t("beauty.detect_roi_hint"))
        self.detect_roi.setChecked(bool(self.values.get("detect_roi", True)))
        row = self._add_row(grid, row, "detect_roi", self.detect_roi, "beauty.detect_roi")

        self.detect_interval = QSpinBox()
        self.detect_interval.setRange(1, 30)
        self.detect_interval.setValue(int(self.values.get("detect_interval", 1)))
        self.detect_interval.setSuffix(self.i18n.t("beauty.frames_suffix"))
        self.detect_interval.setFixedWidth(110)
        row = self._add_row(grid, row, "detect_interval", self.detect_interval, "beauty.detect_interval")

        self.detector_score = QSpinBox()
        self.detector_score.setRange(10, 95)
        self.detector_score.setValue(int(self.values.get("detector_score", 50)))
        self.detector_score.setSuffix("%")
        self.detector_score.setFixedWidth(90)
        row = self._add_row(grid, row, "detector_score", self.detector_score, "beauty.detector_score")

        self.max_faces = QSpinBox()
        self.max_faces.setRange(0, 16)
        self.max_faces.setValue(int(self.values.get("max_faces", 0)))
        self.max_faces.setSpecialValueText(self.i18n.t("beauty.max_faces_all"))
        self.max_faces.setFixedWidth(90)
        row = self._add_row(grid, row, "max_faces", self.max_faces, "beauty.max_faces")

        self.landmarker = QCheckBox(self.i18n.t("beauty.landmarker_hint"))
        self.landmarker.setChecked(bool(self.values.get("landmarker", True)))
        row = self._add_row(grid, row, "landmarker", self.landmarker, "beauty.landmarker")
        return box

    # -- actions -------------------------------------------------------------

    def _reset_to_preset(self) -> None:
        """Restore only the preset-driven sliders; detection and quality
        settings are the user's own and survive."""
        for key, value in self._preset_defaults.items():
            if key in self.sliders:
                self.sliders[key].set_value(value)

    def _accept(self) -> None:
        for key, slider in self.sliders.items():
            self.values[key] = slider.value()
        self.values.update({
            "mask_padding": self.mask_padding.value(),
            "region_mask": self.region_mask.isChecked(),
            "max_side": int(self.max_side.currentData()),
            "min_face_mode": str(self.min_face_mode.currentData()),
            "detector_score": self.detector_score.value(),
            "detect_mode": str(self.detect_mode.currentData()),
            "vr_reproject": str(self.vr_reproject.currentData()),
            "detect_interval": self.detect_interval.value(),
            "detect_roi": self.detect_roi.isChecked(),
            "max_faces": self.max_faces.value(),
            "landmarker": self.landmarker.isChecked(),
        })
        self.accept()
