"""Feature configuration dialogs opened from dashboard cards.

Moved from the legacy home_page module; the card grid opens these via the
gear buttons, so the dialogs stay independent from page layout code.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QGridLayout,
    QWidget,
    QVBoxLayout,
)

from ui import theme
from ui.icons import question_icon
from ui.player_support import load_player_support
from ui.settings import DEFAULTS, LIGHT_MATCH_PRESETS
from ui.superres_targets import REALTIME_DEFAULT_TARGET, TARGET_CHOICES, target_i18n_key
from utils.dlss5 import DLSS5_NR_PASSES_RANGE, DLSS5_RANGES
from utils.rtx_vsr import NATIVE_TARGET_HEIGHT
from utils.si_filter import (
    ORIGINAL_VOLUME_CHOICES,
    SI_DELAY_SECONDS_CHOICES,
    SI_DUCK_PRESET_CHOICES,
    SI_MIX_CHANNELS,
    SI_VOLUME_CHOICES,
)

SI_TOOLBOX_URL = "https://github.com/zerochocobo/VR-Video-Toolbox-CE"
ICON_BUTTON_SIZE = 30
LIGHT_MATCH_DEFAULT_PRESET = str(DEFAULTS["light_match_preset"])
TWO_DVR_STRENGTH_OPTIONS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
BG_COLOR_CHOICES = (
    ("bg.neutral_gray", "808080"),
    ("bg.light_gray", "C8C8C8"),
    ("bg.soft_green", "00FF00"),
    ("bg.soft_blue", "0000FF"),
)


def int_setting(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_setting(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _icon_button(icon: QIcon) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(ICON_BUTTON_SIZE, ICON_BUTTON_SIZE)
    button.setText("")
    return button


def _link_label(text: str, url: str) -> QLabel:
    label = QLabel()
    label.setOpenExternalLinks(False)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    if url:
        label.setText(f'<a href="{url}">{text}</a>')
        label.linkActivated.connect(lambda value: QDesktopServices.openUrl(QUrl(value)))
    else:
        label.setText("-")
    return label


class PlayerSupportDialog(QDialog):
    def __init__(self, i18n, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("player_support.window_title"))

        title = QLabel(self.i18n.t("player_support.table_title"))
        title.setObjectName("PlayerSupportTitle")
        title.setStyleSheet("QLabel#PlayerSupportTitle { font-size: 12pt; font-weight: 700; }")

        rows = load_player_support()
        headers = [
            self.i18n.t("player_support.player"),
            self.i18n.t("player_support.alpha"),
            self.i18n.t("player_support.gray_green"),
            self.i18n.t("player_support.chroma_key"),
            self.i18n.t("player_support.website"),
            self.i18n.t("player_support.notes"),
        ]
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setWordWrap(True)
        for row_index, row in enumerate(rows):
            values = [
                row.player,
                self._support_text(row.alpha),
                self._support_text(row.gray_green),
                self._support_text(row.chroma_key),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if column else Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_index, column, item)
            table.setCellWidget(row_index, 4, _link_label(self.i18n.t("player_support.official_site"), row.website_url))
            table.setCellWidget(
                row_index,
                5,
                _link_label(self.i18n.t("player_support.install_notes"), row.notes_url),
            )
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(headers)):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        table.resizeRowsToContents()

        close_button = QPushButton(self.i18n.t("button.close"))
        close_button.clicked.connect(self.accept)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(table)
        layout.addLayout(button_row)
        self.resize(760, 330)

    def _support_text(self, supported: bool) -> str:
        return self.i18n.t("player_support.supported") if supported else "-"


class PlaybackModeChooser(QWidget):
    """How a passthrough mode is handed to the player: virtual file or live stream.

    A two-way, mutually exclusive choice where the consequence of each option is
    not obvious from its name, so it reads as two radio rows with a line of plain
    explanation under each rather than a combo box (which hides the alternative)
    or a switch (which cannot label both sides). The closing note points at the
    fallback, because the virtual file is the one that can fail on an unusual
    player and the user needs to know there is somewhere to go.
    """

    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        mode = str(
            settings.data.get("passthrough_playback_mode")
            or DEFAULTS["passthrough_playback_mode"]
        ).strip().lower()

        self.virtual_radio = QRadioButton(self.i18n.t("playback.virtual"))
        self.live_radio = QRadioButton(self.i18n.t("playback.live"))
        self.group = QButtonGroup(self)
        self.group.addButton(self.virtual_radio)
        self.group.addButton(self.live_radio)
        self.live_radio.setChecked(mode == "live")
        self.virtual_radio.setChecked(mode != "live")

        title = QLabel(self.i18n.t("playback.title"))
        title.setStyleSheet("font-weight: 600; background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(title)
        for radio, hint_key in (
            (self.virtual_radio, "playback.virtual_hint"),
            (self.live_radio, "playback.live_hint"),
        ):
            radio.setCursor(Qt.CursorShape.PointingHandCursor)
            hint = QLabel(self.i18n.t(hint_key))
            hint.setWordWrap(True)
            hint.setStyleSheet(
                f"font-size: 8.5pt; color: {theme.TEXT_MUTED}; background: transparent;"
            )
            hint.setContentsMargins(22, 0, 0, 0)
            layout.addWidget(radio)
            layout.addWidget(hint)
            layout.addSpacing(2)

        note = QLabel(self.i18n.t("playback.note"))
        note.setWordWrap(True)
        note.setStyleSheet(
            f"font-size: 8.5pt; color: {theme.TEXT_MUTED}; background: {theme.BLUE_SOFT};"
            " border-radius: 6px; padding: 7px 9px;"
        )
        layout.addWidget(note)

    def selected_mode(self) -> str:
        return "live" if self.live_radio.isChecked() else "virtual"


class GreenScreenSettingsDialog(QDialog):
    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("mode.green"))

        self.bg_color = QComboBox()
        for label_key, value in BG_COLOR_CHOICES:
            self.bg_color.addItem(self.i18n.t(label_key), value)
        idx = self.bg_color.findData(str(settings.data.get("background_color") or DEFAULTS["background_color"]))
        self.bg_color.setCurrentIndex(max(0, idx))

        color_label = QLabel(self.i18n.t("dashboard.green_bg_color"))
        color_row = QHBoxLayout()
        color_row.addWidget(color_label)
        color_row.addWidget(self.bg_color, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.playback = PlaybackModeChooser(self.i18n, settings, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(color_row)
        layout.addWidget(self.playback)
        layout.addWidget(buttons)
        self.resize(420, 300)

    def selected_color(self) -> str:
        return str(self.bg_color.currentData() or DEFAULTS["background_color"])

    def selected_playback_mode(self) -> str:
        return self.playback.selected_mode()


class AlphaPassthroughSettingsDialog(QDialog):
    """Alpha passthrough options. Only the playback mode for now."""

    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("mode.alpha"))

        self.playback = PlaybackModeChooser(self.i18n, settings, self)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addWidget(self.playback)
        layout.addWidget(buttons)
        self.resize(420, 260)

    def selected_playback_mode(self) -> str:
        return self.playback.selected_mode()


class Alpha2DSettingsDialog(QDialog):
    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.settings = settings
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("alpha2d.dialog_title"))

        projection = str(settings.data.get("alpha_2d_projection") or "fisheye").lower()
        if projection not in {"fisheye", "flat3d"}:
            projection = "fisheye"

        self.fisheye_radio = QRadioButton(self.i18n.t("alpha2d.projection_fisheye"))
        self.flat3d_radio = QRadioButton(self.i18n.t("alpha2d.projection_flat3d"))
        self.projection_group = QButtonGroup(self)
        self.projection_group.addButton(self.fisheye_radio)
        self.projection_group.addButton(self.flat3d_radio)
        self.fisheye_radio.setChecked(projection == "fisheye")
        self.flat3d_radio.setChecked(projection == "flat3d")

        projection_label = QLabel(self.i18n.t("alpha2d.projection"))
        projection_row = QHBoxLayout()
        projection_row.addWidget(projection_label)
        projection_row.addWidget(self.fisheye_radio)
        projection_row.addWidget(self.flat3d_radio)
        projection_row.addStretch(1)

        try:
            distance = int(round(float(settings.data.get("alpha_2d_distance_m") or 4.0)))
        except (TypeError, ValueError):
            distance = 4
        distance = max(1, min(10, distance))
        self.distance_value = QLabel()
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(1, 10)
        self.distance_slider.setSingleStep(1)
        self.distance_slider.setPageStep(1)
        self.distance_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.distance_slider.setTickInterval(1)
        self.distance_slider.setValue(distance)
        self.distance_slider.valueChanged.connect(self._update_distance_label)

        distance_label = QLabel(self.i18n.t("alpha2d.distance"))
        distance_row = QHBoxLayout()
        distance_row.addWidget(distance_label)
        distance_row.addWidget(self.distance_slider, 1)
        distance_row.addWidget(self.distance_value)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(projection_row)
        layout.addLayout(distance_row)
        layout.addWidget(buttons)
        self._update_distance_label(distance)
        self.resize(360, 150)

    def _update_distance_label(self, value: int) -> None:
        self.distance_value.setText(f"{int(value)}m")

    def selected_projection(self) -> str:
        return "flat3d" if self.flat3d_radio.isChecked() else "fisheye"

    def selected_distance_m(self) -> float:
        return float(self.distance_slider.value())


class SuperResSettingsDialog(QDialog):
    """Realtime RTX VSR settings exposed from the dashboard card."""

    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("superres.realtime_title"))
        self.target = QComboBox()
        for target in TARGET_CHOICES:
            self.target.addItem(self.i18n.t(target_i18n_key(target)), target)
        current_target = int_setting(settings.data.get("superres_target_height"), REALTIME_DEFAULT_TARGET)
        self.target.setCurrentIndex(max(0, self.target.findData(current_target)))
        self.quality = QComboBox()
        for quality in (1, 2, 3, 4):
            self.quality.addItem("", quality)
        current_quality = max(1, min(4, int_setting(settings.data.get("superres_quality"), 4)))
        self.quality.setCurrentIndex(current_quality - 1)
        self.hdr_look = QComboBox()
        for mode in ("off", "natural", "vivid"):
            self.hdr_look.addItem(self.i18n.t(f"superres.hdr_look_{mode}"), mode)
        current_hdr = str(settings.data.get("superres_hdr_look") or "natural").strip().lower()
        hdr_index = self.hdr_look.findData(current_hdr if current_hdr in {"off", "natural", "vivid"} else "natural")
        self.hdr_look.setCurrentIndex(max(0, hdr_index))

        target_row = QHBoxLayout(); target_row.addWidget(QLabel(self.i18n.t("superres.target"))); target_row.addWidget(self.target); target_row.addStretch(1)
        quality_row = QHBoxLayout(); quality_row.addWidget(QLabel(self.i18n.t("superres.quality"))); quality_row.addWidget(self.quality); quality_row.addStretch(1)
        hdr_row = QHBoxLayout(); hdr_row.addWidget(QLabel(self.i18n.t("superres.hdr_look"))); hdr_row.addWidget(self.hdr_look); hdr_row.addStretch(1)
        self.performance_note = QLabel(self.i18n.t("superres.performance_note"))
        self.performance_note.setWordWrap(True)
        self.performance_note.setStyleSheet("color: #626975; background: transparent;")
        # Only the native 1x target can be served as a virtual file: enlarging
        # runs at 25-27 FPS and needs about 173 Mbps, which a player pulling at
        # playback speed cannot ride. So the choice appears exactly when it
        # applies, and the note says why it is missing otherwise.
        self.playback = PlaybackModeChooser(self.i18n, settings, self)
        self.playback_note = QLabel(self.i18n.t("superres.playback_native_only"))
        self.playback_note.setWordWrap(True)
        self.playback_note.setStyleSheet("color: #626975; background: transparent;")
        self.target.currentIndexChanged.connect(self._update_playback_visibility)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.setContentsMargins(16, 14, 16, 14); layout.setSpacing(12)
        layout.addLayout(target_row); layout.addLayout(quality_row); layout.addLayout(hdr_row)
        layout.addWidget(self.performance_note); layout.addWidget(self.playback); layout.addWidget(self.playback_note)
        layout.addWidget(buttons)
        self.quality_keys = ("superres.quality_1", "superres.quality_2", "superres.quality_3", "superres.quality_4")
        for i, key in enumerate(self.quality_keys): self.quality.setItemText(i, self.i18n.t(key))
        self.setMinimumWidth(520)
        self.setMaximumWidth(520)
        self._update_playback_visibility()
        self.adjustSize()

    def _update_playback_visibility(self) -> None:
        # The note stays visible at every target: it is the rule that decides
        # whether the chooser above it exists, so the user should read it
        # before picking a target, not only after losing the choice.
        native = int(self.target.currentData() or 0) <= NATIVE_TARGET_HEIGHT
        self.playback.setVisible(native)
        self.adjustSize()

    def selected_playback_mode(self) -> str:
        """The shared passthrough playback mode, unchanged unless 1x is chosen."""
        if int(self.target.currentData() or 0) > NATIVE_TARGET_HEIGHT:
            return ""
        return self.playback.selected_mode()

    def selected_target_height(self) -> int:
        return int(self.target.currentData())

    def selected_quality(self) -> int:
        return int(self.quality.currentData())

    def selected_hdr_look(self) -> str:
        return str(self.hdr_look.currentData() or "natural")


class DLSS5SettingsDialog(QDialog):
    """Realtime DLSS 5 Neural Rendering settings, from the [DLSS5] card.

    Neural Rendering has thirteen controls and most of them refine one another.
    The ones that change what a viewer notices lead; the rest sit behind a
    disclosure, because a dialog that opens onto thirteen sliders is a dialog
    nobody adjusts.

    Every slider's range comes from utils.dlss5.DLSS5_RANGES, so the UI cannot
    offer a value the runtime then rejects. They are integer sliders over
    hundredths - Qt has no float slider - and the label beside each one shows
    the real value.
    """

    #: (settings key, label key, range key). Order is the order on screen.
    MAIN_SLIDERS = (
        ("dlss5_intensity", "dlss5.intensity", "intensity"),
        ("dlss5_shimmer_suppression", "dlss5.shimmer_suppression", "shimmer_suppression"),
    )
    ADVANCED_SLIDERS = (
        ("dlss5_local_tone", "dlss5.local_tone", "local_tone"),
        ("dlss5_local_structure", "dlss5.local_structure", "local_structure"),
        ("dlss5_skin_structure", "dlss5.skin_structure", "skin_structure"),
        ("dlss5_color_strength", "dlss5.color_strength", "color_strength"),
        ("dlss5_tone_preservation", "dlss5.tone_preservation", "tone_preservation"),
        ("dlss5_face_skin_protection", "dlss5.face_skin_protection", "face_skin_protection"),
        ("dlss5_grain_preservation", "dlss5.grain_preservation", "grain_preservation"),
    )
    STYLE_KEYS = ("dlss5.style_default", "dlss5.style_natural", "dlss5.style_cinematic")

    def __init__(self, i18n, settings, parent=None, *, offline: bool = False) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.offline = bool(offline)
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("dlss5.offline_title" if offline else "dlss5.realtime_title"))

        self.style = QComboBox()
        for index, key in enumerate(self.STYLE_KEYS):
            self.style.addItem(self.i18n.t(key), index)
        self.style.setCurrentIndex(
            max(0, min(
                len(self.STYLE_KEYS) - 1,
                int_setting(settings.data.get("dlss5_style"), DEFAULTS["dlss5_style"]),
            ))
        )
        self.nr_passes = QComboBox()
        for passes in range(DLSS5_NR_PASSES_RANGE[0], DLSS5_NR_PASSES_RANGE[1] + 1):
            self.nr_passes.addItem(str(passes), passes)
        self.nr_passes.setCurrentIndex(
            max(0, self.nr_passes.findData(
                int_setting(settings.data.get("dlss5_nr_passes"), DEFAULTS["dlss5_nr_passes"])
            ))
        )

        self.sliders: dict[str, QSlider] = {}
        self.value_labels: dict[str, QLabel] = {}

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel(self.i18n.t("dlss5.style")))
        style_row.addWidget(self.style)
        style_row.addStretch(1)
        style_row.addWidget(QLabel(self.i18n.t("dlss5.nr_passes")))
        style_row.addWidget(self.nr_passes)

        main_form = self._slider_form(settings, self.MAIN_SLIDERS)
        self.advanced = self._slider_form(settings, self.ADVANCED_SLIDERS)
        self.auto_mask = QCheckBox(self.i18n.t("dlss5.auto_mask"))
        self.auto_mask.setChecked(bool(settings.data.get("dlss5_auto_mask", DEFAULTS["dlss5_auto_mask"])))
        advanced_box = QVBoxLayout()
        advanced_box.setContentsMargins(0, 0, 0, 0)
        advanced_box.setSpacing(8)
        advanced_box.addWidget(self.advanced)
        advanced_box.addWidget(self.auto_mask)
        self.advanced_panel = QFrame()
        self.advanced_panel.setLayout(advanced_box)
        self.advanced_panel.setVisible(False)
        self.advanced_toggle = QPushButton()
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setFlat(True)
        self.advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.advanced_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; background: transparent;"
            f" color: {theme.BLUE_DARK}; }}"
        )
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self._toggle_advanced(False)

        # NR is 1x, so there is no target that cannot be served as a virtual
        # file: the chooser is always meaningful and always shown, unlike the
        # SuperRes dialog where it appears only at the native target. Offline
        # has no playback channel at all, so there it is absent.
        self.playback = None if self.offline else PlaybackModeChooser(self.i18n, settings, self)
        self.performance_note = QLabel(
            self.i18n.t("dlss5.offline_note" if self.offline else "dlss5.performance_note")
        )
        self.performance_note.setWordWrap(True)
        self.performance_note.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(style_row)
        layout.addWidget(main_form)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_panel)
        layout.addWidget(self.performance_note)
        if self.playback is not None:
            layout.addWidget(self.playback)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
        self.setMaximumWidth(520)
        self.adjustSize()

    def _slider_form(self, settings, rows) -> QWidget:
        form = QWidget()
        grid = QGridLayout(form)
        grid.setContentsMargins(0, 0, 0, 0)
        for row, (key, label_key, range_key) in enumerate(rows):
            low, high = DLSS5_RANGES[range_key]
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(round(low * 100)), int(round(high * 100)))
            value = float_setting(settings.data.get(key), DEFAULTS[key])
            slider.setValue(int(round(max(low, min(high, value)) * 100)))
            value_label = QLabel(f"{slider.value() / 100:.2f}")
            value_label.setFixedWidth(42)
            value_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: transparent;")
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v / 100:.2f}"))
            self.sliders[key] = slider
            self.value_labels[key] = value_label
            grid.addWidget(QLabel(self.i18n.t(label_key)), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)
        grid.setColumnStretch(1, 1)
        return form

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_panel.setVisible(bool(checked))
        arrow = "\u25be" if checked else "\u25b8"
        self.advanced_toggle.setText(f"{arrow} {self.i18n.t('dlss5.advanced')}")
        self.adjustSize()

    def selected_playback_mode(self) -> str:
        """The shared passthrough choice, or "" when this dialog is offline."""
        return "" if self.playback is None else self.playback.selected_mode()

    def payload(self) -> dict:
        """Every DLSS5 setting this dialog owns, ready for settings.data."""
        values: dict = {
            "dlss5_style": int(self.style.currentData() or 0),
            "dlss5_nr_passes": int(self.nr_passes.currentData() or 1),
            "dlss5_auto_mask": self.auto_mask.isChecked(),
        }
        for key, slider in self.sliders.items():
            values[key] = round(slider.value() / 100.0, 2)
        return values


class TwoDvrSettingsDialog(QDialog):
    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("home.two_dvr_toggle"))

        self.strength = QComboBox()
        for value in TWO_DVR_STRENGTH_OPTIONS:
            self.strength.addItem(f"{int(round(value * 100))}%", value)
        current = max(0.5, min(2.0, float_setting(
            settings.data.get("two_dvr_live_strength"),
            DEFAULTS["two_dvr_live_strength"],
        )))
        self.strength.setCurrentIndex(min(
            range(len(TWO_DVR_STRENGTH_OPTIONS)),
            key=lambda i: abs(TWO_DVR_STRENGTH_OPTIONS[i] - current),
        ))

        strength_label = QLabel(self.i18n.t("twodvr.strength"))
        row = QHBoxLayout()
        row.addWidget(strength_label)
        row.addWidget(self.strength, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(row)
        layout.addWidget(buttons)
        self.resize(320, 120)

    def selected_strength(self) -> float:
        return float(self.strength.currentData() or DEFAULTS["two_dvr_live_strength"])


class RmSettingsDialog(QDialog):
    """Mosaic-restoration settings opened from the [RM] card's gear button."""

    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("rm.dialog_title"))

        self.vr2flat = QCheckBox(self.i18n.t("rm.vr2flat_decode"))
        self.vr2flat.setChecked(bool(settings.data.get(
            "rm_vr2flat_decode", DEFAULTS["rm_vr2flat_decode"])))
        self.vr2flat.setToolTip(self.i18n.t("rm.vr2flat_decode_hint"))
        hint = QLabel(self.i18n.t("rm.vr2flat_decode_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6b7280; font-size: 8.5pt;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self.vr2flat)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.resize(380, 150)

    def payload(self) -> dict:
        return {"vr2flat_decode": self.vr2flat.isChecked()}


class FaceBeautySettingsDialog(QDialog):
    """Realtime beautification settings, opened from the [FaceBeauty] card.

    Only strength is configurable here: the realtime path pins the cheapest
    restorer and its detection/mask settings so the frame budget is predictable,
    while the offline page keeps the full option set. The presets and their
    values come from offline.face_beauty_engine, so the two cannot drift."""

    STRENGTH_KEYS = (
        ("enhancer_blend", "beauty.enhancer_blend"),
        ("skin_smooth", "beauty.skin_smooth"),
        ("skin_brighten", "beauty.skin_brighten"),
        ("skin_even", "beauty.skin_even"),
        ("eye_brighten", "beauty.eye_brighten"),
        ("teeth_white", "beauty.teeth_white"),
        ("lip_vivid", "beauty.lip_vivid"),
        ("sharpen", "beauty.sharpen"),
    )
    PRESETS = ("restore", "natural", "standard", "strong")
    PRESET_LABEL_KEYS = {
        "restore": "beauty.preset_restore",
        "natural": "beauty.preset_natural",
        "standard": "beauty.preset_standard",
        "strong": "beauty.preset_strong",
        "custom": "beauty.preset_custom",
    }

    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.settings = settings
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("home.face_beauty_toggle"))

        stored = settings.data.get("face_beauty_live") or {}
        if not isinstance(stored, dict):
            stored = {}

        self.preset = QComboBox()
        for name in (*self.PRESETS, "custom"):
            self.preset.addItem(self.i18n.t(self.PRESET_LABEL_KEYS[name]), name)
        self.preset.currentIndexChanged.connect(self._preset_changed)

        self.sliders: dict[str, QSlider] = {}
        self.value_labels: dict[str, QLabel] = {}
        form = QWidget()
        grid = QGridLayout(form)
        grid.setContentsMargins(0, 0, 0, 0)
        for row, (key, label_key) in enumerate(self.STRENGTH_KEYS):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(stored.get(key, self._preset_values("standard").get(key, 0))))
            value_label = QLabel(f"{slider.value()}%")
            value_label.setFixedWidth(42)
            value_label.setStyleSheet("color: #5f6368;")
            slider.valueChanged.connect(
                lambda v, lbl=value_label: lbl.setText(f"{v}%"))
            slider.valueChanged.connect(self._mark_custom)
            self.sliders[key] = slider
            self.value_labels[key] = value_label
            grid.addWidget(QLabel(self.i18n.t(label_key)), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)
        self.form = form

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(self.i18n.t("beauty.settings")))
        preset_row.addWidget(self.preset, 1)

        note = QLabel(self.i18n.t("beauty.live_note"))
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280; font-size: 8.5pt;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(preset_row)
        layout.addWidget(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.resize(430, 380)

        self._syncing = False
        stored_preset = str(stored.get("preset") or "standard")
        index = self.preset.findData(stored_preset if stored_preset in
                                     (*self.PRESETS, "custom") else "standard")
        self.preset.setCurrentIndex(max(0, index))
        self._update_form_visibility()

    # -- preset plumbing -----------------------------------------------------

    @staticmethod
    def _preset_values(name: str) -> dict:
        try:
            from offline.face_beauty_engine import preset_percentages

            return preset_percentages(name)
        except Exception:
            return {}

    def _preset_changed(self) -> None:
        name = str(self.preset.currentData() or "standard")
        if name != "custom":
            values = self._preset_values(name)
            self._syncing = True
            for key, slider in self.sliders.items():
                if key in values:
                    slider.setValue(int(values[key]))
            self._syncing = False
        self._update_form_visibility()

    def _mark_custom(self) -> None:
        """Touching a slider is what makes the configuration custom."""
        if self._syncing:
            return
        index = self.preset.findData("custom")
        if index >= 0 and self.preset.currentIndex() != index:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(index)
            self.preset.blockSignals(False)
            self._update_form_visibility()

    def _update_form_visibility(self) -> None:
        self.form.setVisible(str(self.preset.currentData() or "") == "custom")
        self.adjustSize()

    def payload(self) -> dict:
        data = {"preset": str(self.preset.currentData() or "standard")}
        data.update({key: int(slider.value()) for key, slider in self.sliders.items()})
        return data


class SISettingsDialog(QDialog):
    def __init__(self, i18n, settings, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.settings = settings
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("si.dialog_title"))

        self.dub_mode = QCheckBox(self.i18n.t("si.dub_mode"))
        self.dub_mode.setChecked(bool(settings.data.get("si_dub_mode", DEFAULTS["si_dub_mode"])))
        self.dub_mode_help = _icon_button(question_icon())
        self.dub_mode_help.setToolTip(self.i18n.t("si.dub_mode_help_title"))
        self.dub_mode_help.clicked.connect(self._show_dub_mode_help)

        self.channel = QComboBox()
        for value in SI_MIX_CHANNELS:
            self.channel.addItem(self.i18n.t(f"si.channel_{value}"), value)
        idx = self.channel.findData(str(settings.data.get("si_mix_channel") or DEFAULTS["si_mix_channel"]))
        self.channel.setCurrentIndex(max(0, idx))

        self.original_volume = QComboBox()
        for value in ORIGINAL_VOLUME_CHOICES:
            self.original_volume.addItem(f"{value}%", value)
        idx = self.original_volume.findData(int_setting(
            settings.data.get("si_original_volume_percent"),
            DEFAULTS["si_original_volume_percent"],
        ))
        self.original_volume.setCurrentIndex(max(0, idx))

        self.si_volume = QComboBox()
        for value in SI_VOLUME_CHOICES:
            self.si_volume.addItem(f"{value}%", value)
        idx = self.si_volume.findData(int_setting(settings.data.get("si_volume_percent"), DEFAULTS["si_volume_percent"]))
        self.si_volume.setCurrentIndex(max(0, idx))

        self.delay = QComboBox()
        for value in SI_DELAY_SECONDS_CHOICES:
            self.delay.addItem(f"{value:g}s", value)
        delay = round(float_setting(settings.data.get("si_delay_seconds"), DEFAULTS["si_delay_seconds"]), 1)
        idx = self.delay.findData(delay)
        self.delay.setCurrentIndex(max(0, idx))

        self.duck_original = QCheckBox(self.i18n.t("si.duck_original"))
        self.duck_original.setToolTip(self.i18n.t("si.duck_original_tooltip"))
        self.duck_original.setChecked(bool(settings.data.get("si_duck_original", DEFAULTS["si_duck_original"])))

        self.duck_preset_label = QLabel(self.i18n.t("si.duck_preset"))
        self.duck_preset = QComboBox()
        for value in SI_DUCK_PRESET_CHOICES:
            self.duck_preset.addItem(self.i18n.t(f"si.duck_preset_{value}"), value)
        idx = self.duck_preset.findData(str(settings.data.get("si_duck_preset") or DEFAULTS["si_duck_preset"]))
        self.duck_preset.setCurrentIndex(max(0, idx))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        for label_key, widget in (
            ("si.mix_channel", self.channel),
            ("si.original_volume", self.original_volume),
            ("si.si_volume", self.si_volume),
            ("si.delay", self.delay),
        ):
            row = QHBoxLayout()
            label = QLabel(self.i18n.t(label_key))
            label.setFixedWidth(110)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label)
            row.addWidget(widget, 1)
            layout.addLayout(row)
        duck_row = QHBoxLayout()
        duck_row.addSpacing(110)
        duck_row.addWidget(self.duck_original)
        duck_row.addStretch(1)
        layout.addLayout(duck_row)

        duck_preset_row = QHBoxLayout()
        self.duck_preset_label.setFixedWidth(110)
        self.duck_preset_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        duck_preset_row.addWidget(self.duck_preset_label)
        duck_preset_row.addWidget(self.duck_preset, 1)
        layout.addLayout(duck_preset_row)

        dub_row = QHBoxLayout()
        dub_row.addSpacing(110)
        dub_row.addWidget(self.dub_mode)
        dub_row.addWidget(self.dub_mode_help)
        dub_row.addStretch(1)
        layout.addLayout(dub_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.duck_original.toggled.connect(self._update_duck_preset_enabled)
        self._update_duck_preset_enabled(self.duck_original.isChecked())
        self.resize(360, 290)

    def _update_duck_preset_enabled(self, enabled: bool) -> None:
        self.duck_preset_label.setEnabled(bool(enabled))
        self.duck_preset.setEnabled(bool(enabled))

    def _show_dub_mode_help(self) -> None:
        QMessageBox.information(
            self,
            self.i18n.t("si.dub_mode_help_title"),
            self.i18n.t("si.dub_mode_help"),
        )

    def payload(self) -> dict:
        return {
            "mix_channel": str(self.channel.currentData() or DEFAULTS["si_mix_channel"]),
            "original_volume_percent": int(self.original_volume.currentData() or DEFAULTS["si_original_volume_percent"]),
            "si_volume_percent": int(self.si_volume.currentData() or DEFAULTS["si_volume_percent"]),
            "si_delay_seconds": float(self.delay.currentData() or DEFAULTS["si_delay_seconds"]),
            "duck_original": self.duck_original.isChecked(),
            "duck_preset": str(self.duck_preset.currentData() or DEFAULTS["si_duck_preset"]),
            "dub_mode_enabled": self.dub_mode.isChecked(),
        }


class SIHelpDialog(QDialog):
    def __init__(self, i18n, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("si.help_title"))

        message = self.i18n.t("si.help_message").format(
            link=f'<a href="{SI_TOOLBOX_URL}">{SI_TOOLBOX_URL}</a>'
        )
        label = QLabel(message)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(False)
        label.linkActivated.connect(lambda value: QDesktopServices.openUrl(QUrl(value)))
        label.setMinimumWidth(560)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(self.i18n.t("button.close"))
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addWidget(label)
        layout.addWidget(buttons)
        self.resize(640, 220)


class LightMatchAdvancedDialog(QDialog):
    def __init__(self, i18n, payload: dict, live_callback, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self._live_callback = live_callback
        self._updating = False
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("light_match.custom_title"))

        self.temp = self._slider(2700, 9000, int_setting(payload.get("temp_k"), DEFAULTS["light_match_temp_k"]))
        self.tint = self._slider(-50, 50, int_setting(payload.get("tint"), 0))
        self.exposure = self._slider(-100, 100, int(round(float_setting(payload.get("exposure_ev"), 0.0) * 100)))
        self.contrast = self._slider(80, 120, int(round(float_setting(payload.get("contrast"), 1.0) * 100)))
        self.gamma = self._slider(85, 115, int(round(float_setting(payload.get("gamma"), 1.0) * 100)))
        self.saturation = self._slider(50, 150, int(round(float_setting(payload.get("saturation"), 1.0) * 100)))
        self._value_labels: dict[QSlider, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        for label_key, slider, left_text, right_text in (
            ("light_match.temp_k", self.temp, "2700K", "9000K"),
            ("light_match.tint", self.tint, "-50", "+50"),
            ("light_match.exposure", self.exposure, "-1EV", "+1EV"),
            ("light_match.contrast", self.contrast, "80%", "120%"),
            ("light_match.gamma", self.gamma, "0.85", "1.15"),
            ("light_match.saturation", self.saturation, "50%", "150%"),
        ):
            row = QHBoxLayout()
            row.setSpacing(6)
            label = QLabel(self.i18n.t(label_key))
            label.setFixedWidth(92)
            label.setStyleSheet("color: #5f6368;")
            value_label = QLabel()
            value_label.setFixedWidth(72)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setStyleSheet("color: #1677c7; font-weight: 600;")
            self._value_labels[slider] = value_label
            row.addWidget(label)
            left_value = QLabel(left_text)
            left_value.setStyleSheet("color: #8a8f98;")
            row.addWidget(left_value)
            row.addWidget(slider, 1)
            right_value = QLabel(right_text)
            right_value.setStyleSheet("color: #8a8f98;")
            row.addWidget(right_value)
            row.addSpacing(8)
            row.addWidget(value_label)
            layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.restore_defaults_button = QPushButton(self.i18n.t("light_match.restore_defaults"))
        bottom = QHBoxLayout()
        bottom.addWidget(self.restore_defaults_button)
        bottom.addStretch(1)
        bottom.addWidget(buttons)
        layout.addLayout(bottom)

        for slider in (self.temp, self.tint, self.exposure, self.contrast, self.gamma, self.saturation):
            slider.valueChanged.connect(self._manual_changed)
        self.restore_defaults_button.clicked.connect(self._restore_defaults)
        self._update_value_labels()
        self.resize(520, 340)

    def _slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(max(minimum, min(maximum, value)))
        return slider

    def payload(self) -> dict:
        return {
            "enabled": True,
            "temp_k": int(self.temp.value()),
            "tint": float(self.tint.value()),
            "exposure_ev": float(self.exposure.value()) / 100.0,
            "contrast": float(self.contrast.value()) / 100.0,
            "gamma": float(self.gamma.value()) / 100.0,
            "saturation": float(self.saturation.value()) / 100.0,
            "preset": "custom",
        }

    def _emit_live(self) -> None:
        if not self._updating:
            self._live_callback(self.payload())

    def _manual_changed(self) -> None:
        self._update_value_labels()
        self._emit_live()

    def _restore_defaults(self) -> None:
        self._updating = True
        for slider, value in (
            (self.temp, DEFAULTS["light_match_temp_k"]),
            (self.tint, 0),
            (self.exposure, 0),
            (self.contrast, 100),
            (self.gamma, 100),
            (self.saturation, 100),
        ):
            slider.setValue(value)
        self._updating = False
        self._update_value_labels()
        self._emit_live()

    def _update_value_labels(self) -> None:
        values = {
            self.temp: f"{self.temp.value()}K",
            self.tint: f"{self.tint.value():+d}",
            self.exposure: f"{self.exposure.value() / 100.0:+.2f}EV",
            self.contrast: f"{self.contrast.value()}%",
            self.gamma: f"{self.gamma.value() / 100.0:.2f}",
            self.saturation: f"{self.saturation.value()}%",
        }
        for slider, text in values.items():
            label = self._value_labels.get(slider)
            if label is not None:
                label.setText(text)


class LightMatchSettingsDialog(QDialog):
    """Preset picker with an advanced editor for the custom preset."""

    def __init__(self, i18n, payload: dict, live_callback, parent=None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self._live_callback = live_callback
        self._payload = dict(payload)
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("light_match.title"))

        self.preset = QComboBox()
        for key in ("home_warm", "daylight", "night_cool", "custom"):
            self.preset.addItem(self.i18n.t(f"light_match.preset_{key}"), key)
        idx = self.preset.findData(str(payload.get("preset") or LIGHT_MATCH_DEFAULT_PRESET))
        self.preset.setCurrentIndex(max(0, idx))

        preset_label = QLabel(self.i18n.t("light_match.enabled"))
        preset_row = QHBoxLayout()
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.preset, 1)

        self.advanced_button = QPushButton(self.i18n.t("light_match.advanced"))
        self.advanced_button.clicked.connect(self._open_advanced)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(self.i18n.t("button.save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.i18n.t("button.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.addLayout(preset_row)
        layout.addWidget(self.advanced_button)
        layout.addWidget(buttons)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self._update_advanced_visible()
        self.resize(360, 150)

    def _preset_changed(self) -> None:
        preset = str(self.preset.currentData() or LIGHT_MATCH_DEFAULT_PRESET)
        values = LIGHT_MATCH_PRESETS.get(preset)
        if values is not None:
            self._payload.update(values)
        self._payload["preset"] = preset
        self._update_advanced_visible()
        self._live_callback(self.payload())

    def _update_advanced_visible(self) -> None:
        self.advanced_button.setVisible(str(self.preset.currentData()) == "custom")

    def _open_advanced(self) -> None:
        dialog = LightMatchAdvancedDialog(self.i18n, self.payload(), self._live_callback, self)
        if dialog.exec() == LightMatchAdvancedDialog.DialogCode.Accepted:
            self._payload.update(dialog.payload())
        self._live_callback(self.payload())

    def payload(self) -> dict:
        payload = dict(self._payload)
        payload["preset"] = str(self.preset.currentData() or LIGHT_MATCH_DEFAULT_PRESET)
        payload["enabled"] = True
        return payload
