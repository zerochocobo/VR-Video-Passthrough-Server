"""Dashboard home page: server control bar plus the feature card grid."""
from __future__ import annotations

import os
import socket

from PySide6.QtCore import QPoint, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.dialogs.feature_dialogs import (
    Alpha2DSettingsDialog,
    BG_COLOR_CHOICES,
    GreenScreenSettingsDialog,
    LightMatchSettingsDialog,
    PlayerSupportDialog,
    SIHelpDialog,
    SISettingsDialog,
    SuperResSettingsDialog,
    TwoDvrSettingsDialog,
    float_setting,
    int_setting,
)
from ui.services.live_control import send_control
from ui.settings import DEFAULTS
from ui.widgets.feature_card import FeatureCard

DASHBOARD_WIDTH = 700
DASHBOARD_HEIGHT = 560
SERVER_ICON_SIZE = 22
GRID_COLUMNS = 3
PROJECT_URL = "https://wapok.com"


def _server_button_icon(running: bool) -> QIcon:
    size = SERVER_ICON_SIZE
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#ffffff"))
    if running:
        side = int(size * 0.5)
        offset = (size - side) // 2
        painter.drawRoundedRect(offset, offset, side, side, 2, 2)
    else:
        left = int(size * 0.34)
        painter.drawPolygon([
            QPoint(left, int(size * 0.25)),
            QPoint(left, int(size * 0.75)),
            QPoint(int(size * 0.76), size // 2),
        ])
    painter.end()
    return QIcon(pixmap)


def _http_port() -> str:
    return str(os.environ.get("PT_HTTP_PORT") or "8200").strip() or "8200"


def _detect_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return str(probe.getsockname()[0])
    except Exception:
        return "127.0.0.1"


class DashboardPage(QWidget):
    open_subtitle_page = Signal()

    def __init__(self, i18n, settings) -> None:
        super().__init__()
        self.i18n = i18n
        self.settings = settings
        self._server_running = False

        self.server_button = QPushButton()
        self.server_button.setMinimumHeight(52)
        self.server_button.setMinimumWidth(190)
        self.server_button.setIconSize(QSize(SERVER_ICON_SIZE, SERVER_ICON_SIZE))
        self.server_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.server_state_label = QLabel()
        self.server_state_label.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {theme.TEXT_MUTED}; background: transparent;"
        )
        self.server_detail_label = QLabel()
        self.server_detail_label.setStyleSheet(
            f"font-size: 9pt; color: {theme.TEXT_FAINT}; background: transparent;"
        )
        server_text = QVBoxLayout()
        server_text.setSpacing(2)
        server_text.addWidget(self.server_state_label)
        server_text.addWidget(self.server_detail_label)

        server_bar = QFrame()
        server_bar.setObjectName("ServerBar")
        server_bar.setStyleSheet(
            "QFrame#ServerBar {"
            f"background: {theme.CARD_BG}; border: 1px solid {theme.CARD_BORDER}; border-radius: 12px;"
            "}"
        )
        server_bar_layout = QHBoxLayout(server_bar)
        server_bar_layout.setContentsMargins(14, 12, 14, 12)
        server_bar_layout.setSpacing(14)
        server_bar_layout.addWidget(self.server_button)
        server_bar_layout.addLayout(server_text, 1)

        self.switch_lock_notice = QLabel()
        self.switch_lock_notice.setWordWrap(True)
        self.switch_lock_notice.setStyleSheet(
            f"font-size: 8.5pt; color: {theme.TEXT_MUTED}; background: transparent; padding-left: 2px;"
        )
        self.switch_lock_notice.setVisible(False)

        self.cards: dict[str, FeatureCard] = {
            "green": FeatureCard("green_screen"),
            "alpha": FeatureCard("alpha", configurable=False, with_help=True),
            "alpha2d": FeatureCard("alpha"),
            "two_dvr": FeatureCard("two_dvr"),
            "superres": FeatureCard("rm", configurable=True),
            "rm": FeatureCard("rm", configurable=False),
            "subtitle": FeatureCard("subtitle"),
            "si": FeatureCard("translate", with_help=True),
            "light": FeatureCard("light"),
        }
        self._realtime_keys = ["alpha", "green", "superres"]
        self._2d_keys = ["two_dvr", "si", "rm"]
        self._audio_keys = ["subtitle", "light", "alpha2d"]

        self.realtime_group_label = QLabel()
        self.audio_group_label = QLabel()
        for label in (self.realtime_group_label, self.audio_group_label):
            label.setStyleSheet(
                f"font-size: 9pt; font-weight: 600; color: {theme.TEXT_FAINT}; background: transparent;"
            )

        self._realtime_grid = QGridLayout()
        self._realtime_grid.setSpacing(8)
        self._2d_grid = QGridLayout()
        self._2d_grid.setSpacing(8)
        self._audio_grid = QGridLayout()
        self._audio_grid.setSpacing(8)

        self.project_link = QLabel()
        self.project_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.project_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.project_link.setOpenExternalLinks(False)
        self.project_link.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.project_link.setStyleSheet(
            "QLabel { font-size: 8.5pt; color: #606266; background: transparent; }"
            f"QLabel a {{ color: {theme.BLUE_DARK}; text-decoration: underline; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 10)
        layout.setSpacing(10)
        layout.addWidget(server_bar)
        layout.addWidget(self.switch_lock_notice)
        layout.addSpacing(2)
        layout.addWidget(self.realtime_group_label)
        layout.addLayout(self._realtime_grid)
        layout.addSpacing(4)
        layout.addLayout(self._2d_grid)
        layout.addSpacing(4)
        layout.addWidget(self.audio_group_label)
        layout.addLayout(self._audio_grid)
        layout.addStretch(1)
        layout.addWidget(self.project_link)

        self._rebuild_grids()
        self._bind_cards()
        self.sync_from_settings()
        self.retranslate()
        self.set_server_running(False)

    # ---- card wiring ----

    def _bind_cards(self) -> None:
        self.cards["green"].toggled.connect(lambda checked: self._save_flag("mode_green", checked))
        self.cards["alpha"].toggled.connect(lambda checked: self._save_flag("mode_alpha", checked))
        self.cards["alpha2d"].toggled.connect(lambda checked: self._save_flag("mode_2d", checked))
        self.cards["two_dvr"].toggled.connect(lambda checked: self._save_flag("mode_two_dvr", checked))
        self.cards["superres"].toggled.connect(lambda checked: self._save_flag("mode_superres", checked))
        self.cards["subtitle"].toggled.connect(lambda checked: self._save_flag("subtitle_enable", checked))
        self.cards["rm"].toggled.connect(self._toggle_rm)
        self.cards["si"].toggled.connect(self._toggle_si)
        self.cards["light"].toggled.connect(self._toggle_light_match)
        self.cards["green"].configure_requested.connect(self._configure_green)
        self.cards["alpha"].help_requested.connect(lambda: PlayerSupportDialog(self.i18n, self).exec())
        self.cards["alpha2d"].configure_requested.connect(self._configure_alpha)
        self.cards["superres"].configure_requested.connect(self._configure_superres)
        self.cards["two_dvr"].configure_requested.connect(self._configure_two_dvr)
        self.cards["subtitle"].configure_requested.connect(self.open_subtitle_page)
        self.cards["si"].configure_requested.connect(self._configure_si)
        self.cards["si"].help_requested.connect(lambda: SIHelpDialog(self.i18n, self).exec())
        self.cards["light"].configure_requested.connect(self._configure_light_match)

    def _rebuild_grids(self) -> None:
        for grid, keys in (
            (self._realtime_grid, self._realtime_keys),
            (self._2d_grid, self._2d_keys),
            (self._audio_grid, self._audio_keys),
        ):
            while grid.count():
                item = grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            visible_keys = [key for key in keys if key != "rm" or self._rm_card_enabled()]
            for index, key in enumerate(visible_keys):
                card = self.cards[key]
                card.setVisible(True)
                grid.addWidget(card, index // GRID_COLUMNS, index % GRID_COLUMNS)
            for column in range(GRID_COLUMNS):
                grid.setColumnStretch(column, 1)
            rows = max(1, (len(visible_keys) + GRID_COLUMNS - 1) // GRID_COLUMNS)
            filled = rows * GRID_COLUMNS
            for pad_index in range(len(visible_keys), filled):
                spacer = QWidget()
                spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                grid.addWidget(spacer, pad_index // GRID_COLUMNS, pad_index % GRID_COLUMNS)
        self.cards["rm"].setVisible(self._rm_card_enabled())

    def _rm_card_enabled(self) -> bool:
        return bool(self.settings.data.get("rm_card_visible"))

    def set_rm_card_visible(self, visible: bool) -> None:
        self._rebuild_grids()
        if not visible and self.cards["rm"].is_checked():
            self.cards["rm"].set_checked(False)
            self._toggle_rm(False)

    # ---- toggle handlers ----

    def _save_flag(self, key: str, checked: bool) -> None:
        self.settings.data[key] = bool(checked)
        self.settings.save()
        self._update_summaries()

    def _toggle_rm(self, checked: bool) -> None:
        self.settings.data["rm_enabled"] = bool(checked)
        self.settings.save()
        send_control("rm", {"enabled": bool(checked)})
        self._update_summaries()

    def _toggle_si(self, checked: bool) -> None:
        self.settings.data["si_enabled"] = bool(checked)
        self.settings.save()
        send_control("si_mix", self._si_payload())
        self._update_summaries()

    def _toggle_light_match(self, checked: bool) -> None:
        self.settings.data["light_match_enabled"] = bool(checked)
        self.settings.save()
        send_control("light_match", self._light_match_payload())
        self._update_summaries()

    # ---- configure handlers ----

    def _configure_green(self) -> None:
        dialog = GreenScreenSettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != GreenScreenSettingsDialog.DialogCode.Accepted:
            return
        self.settings.data["background_color"] = dialog.selected_color()
        self.settings.save()
        self._update_summaries()

    def _configure_alpha(self) -> None:
        dialog = Alpha2DSettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != Alpha2DSettingsDialog.DialogCode.Accepted:
            return
        self.settings.data["alpha_2d_projection"] = dialog.selected_projection()
        self.settings.data["alpha_2d_distance_m"] = dialog.selected_distance_m()
        self.settings.save()
        self._update_summaries()

    def _configure_two_dvr(self) -> None:
        dialog = TwoDvrSettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != TwoDvrSettingsDialog.DialogCode.Accepted:
            return
        self.apply_two_dvr_strength(dialog.selected_strength())

    def _configure_superres(self) -> None:
        dialog = SuperResSettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != SuperResSettingsDialog.DialogCode.Accepted:
            return
        self.settings.data["superres_target_height"] = dialog.selected_target_height()
        self.settings.data["superres_quality"] = dialog.selected_quality()
        self.settings.data["superres_hdr_look"] = dialog.selected_hdr_look()
        self.settings.save()
        self._update_summaries()

    def apply_two_dvr_strength(self, strength: float) -> None:
        # The realtime 2D->VR path only exposes strength; model/hole-fill/eye
        # distance stay pinned to the supported defaults.
        self.settings.data["two_dvr_live_model"] = DEFAULTS["two_dvr_live_model"]
        self.settings.data["two_dvr_live_hole_fill"] = DEFAULTS["two_dvr_live_hole_fill"]
        self.settings.data["two_dvr_live_eye_distance"] = DEFAULTS["two_dvr_live_eye_distance"]
        self.settings.data["two_dvr_live_strength"] = float(strength)
        self.settings.save()
        self._update_summaries()

    def _configure_si(self) -> None:
        dialog = SISettingsDialog(self.i18n, self.settings, self)
        if dialog.exec() != SISettingsDialog.DialogCode.Accepted:
            return
        payload = dialog.payload()
        self.settings.data["si_mix_channel"] = payload["mix_channel"]
        self.settings.data["si_original_volume_percent"] = payload["original_volume_percent"]
        self.settings.data["si_volume_percent"] = payload["si_volume_percent"]
        self.settings.data["si_delay_seconds"] = payload["si_delay_seconds"]
        self.settings.data["si_duck_original"] = payload["duck_original"]
        self.settings.data["si_duck_preset"] = payload["duck_preset"]
        self.settings.data["si_dub_mode"] = payload["dub_mode_enabled"]
        self.settings.save()
        send_control("si_mix", self._si_payload())
        self._update_summaries()

    def _configure_light_match(self) -> None:
        before = self._light_match_payload()
        dialog = LightMatchSettingsDialog(
            self.i18n,
            before,
            lambda payload: send_control("light_match", payload),
            self,
        )
        if dialog.exec() == LightMatchSettingsDialog.DialogCode.Accepted:
            payload = dialog.payload()
            for key, value in payload.items():
                if key != "enabled":
                    self.settings.data[f"light_match_{key}"] = value
            self.settings.save()
            payload["enabled"] = self.cards["light"].is_checked()
            send_control("light_match", payload)
        else:
            send_control("light_match", before)
        self._update_summaries()

    # ---- payload builders ----

    def _si_payload(self) -> dict:
        data = self.settings.data
        return {
            "enabled": bool(data.get("si_enabled")),
            "mix_channel": str(data.get("si_mix_channel") or DEFAULTS["si_mix_channel"]),
            "original_volume_percent": int_setting(
                data.get("si_original_volume_percent"), DEFAULTS["si_original_volume_percent"]
            ),
            "si_volume_percent": int_setting(data.get("si_volume_percent"), DEFAULTS["si_volume_percent"]),
            "si_delay_seconds": float_setting(data.get("si_delay_seconds"), DEFAULTS["si_delay_seconds"]),
            "duck_original": bool(data.get("si_duck_original", DEFAULTS["si_duck_original"])),
            "duck_preset": str(data.get("si_duck_preset") or DEFAULTS["si_duck_preset"]),
            "dub_mode_enabled": bool(data.get("si_dub_mode", DEFAULTS["si_dub_mode"])),
        }

    def _light_match_payload(self) -> dict:
        data = self.settings.data
        return {
            "enabled": bool(data.get("light_match_enabled")),
            "temp_k": int_setting(data.get("light_match_temp_k"), DEFAULTS["light_match_temp_k"]),
            "tint": float_setting(data.get("light_match_tint"), 0.0),
            "exposure_ev": float_setting(data.get("light_match_exposure_ev"), 0.0),
            "contrast": float_setting(data.get("light_match_contrast"), 1.0),
            "gamma": float_setting(data.get("light_match_gamma"), 1.0),
            "saturation": float_setting(data.get("light_match_saturation"), 1.0),
            "preset": str(data.get("light_match_preset") or DEFAULTS["light_match_preset"]),
        }

    # ---- state sync ----

    def sync_from_settings(self) -> None:
        data = self.settings.data
        self.cards["green"].set_checked(bool(data.get("mode_green")))
        self.cards["alpha"].set_checked(bool(data.get("mode_alpha")))
        self.cards["alpha2d"].set_checked(bool(data.get("mode_2d", DEFAULTS["mode_2d"])))
        self.cards["two_dvr"].set_checked(bool(data.get("mode_two_dvr")))
        self.cards["superres"].set_checked(bool(data.get("mode_superres")))
        self.cards["rm"].set_checked(bool(data.get("rm_enabled", DEFAULTS["rm_enabled"])))
        self.cards["subtitle"].set_checked(bool(data.get("subtitle_enable")))
        self.cards["si"].set_checked(bool(data.get("si_enabled")))
        self.cards["light"].set_checked(bool(data.get("light_match_enabled")))
        self._rebuild_grids()
        self._update_summaries()

    def _update_summaries(self) -> None:
        data = self.settings.data
        bg_value = str(data.get("background_color") or DEFAULTS["background_color"])
        bg_key = next((key for key, value in BG_COLOR_CHOICES if value == bg_value), None)
        bg_name = self.i18n.t(bg_key) if bg_key else f"#{bg_value}"
        self.cards["green"].set_summary(f"[GREEN] · {self.i18n.t('dashboard.green_bg_color')} {bg_name}")

        projection = str(data.get("alpha_2d_projection") or "fisheye").lower()
        projection_key = "alpha2d.projection_flat3d" if projection == "flat3d" else "alpha2d.projection_fisheye"
        distance = int(round(float_setting(data.get("alpha_2d_distance_m"), 4.0)))
        self.cards["alpha"].set_summary("[ALPHA]最好的透视效果")
        self.cards["alpha2d"].set_summary(f"{self.i18n.t(projection_key)} · {distance}m")

        strength = float_setting(data.get("two_dvr_live_strength"), DEFAULTS["two_dvr_live_strength"])
        self.cards["two_dvr"].set_summary(f"[2D>3D] · {self.i18n.t('twodvr.strength')} {int(round(strength * 100))}%")
        target = int_setting(data.get("superres_target_height"), DEFAULTS["superres_target_height"])
        quality = max(1, min(4, int_setting(data.get("superres_quality"), DEFAULTS["superres_quality"])))
        quality_key = f"superres.quality_{quality}"
        target_key = "superres.target_2k" if target <= 1440 else ("superres.target_8k_vr" if target >= 4096 else "superres.target_4k")
        hdr_mode = str(data.get("superres_hdr_look") or DEFAULTS["superres_hdr_look"])
        if hdr_mode not in {"off", "natural", "vivid"}:
            hdr_mode = "natural"
        self.cards["superres"].set_summary(f"[SuperRes] · {self.i18n.t(target_key)} · {self.i18n.t(quality_key)} · {self.i18n.t(f'superres.hdr_look_{hdr_mode}')}")

        self.cards["rm"].set_summary(f"[RM] · {self.i18n.t('dashboard.rm_summary')}")
        self.cards["subtitle"].set_summary(self.i18n.t("dashboard.subtitle_summary"))

        if bool(data.get("si_dub_mode", DEFAULTS["si_dub_mode"])):
            self.cards["si"].set_summary(f"[SI] · {self.i18n.t('si.dub_mode')}")
        else:
            original = int_setting(data.get("si_original_volume_percent"), DEFAULTS["si_original_volume_percent"])
            si_volume = int_setting(data.get("si_volume_percent"), DEFAULTS["si_volume_percent"])
            self.cards["si"].set_summary(f"[SI] · {original}% / {si_volume}%")

        preset = str(data.get("light_match_preset") or DEFAULTS["light_match_preset"])
        self.cards["light"].set_summary(self.i18n.t(f"light_match.preset_{preset}"))

    def set_server_running(self, running: bool) -> None:
        self._server_running = running
        locked_tip = self.i18n.t("dashboard.switch_locked_tooltip")
        for key, card in self.cards.items():
            card.set_toggle_enabled(running is False or key == "light", locked_tip)
        self.switch_lock_notice.setVisible(running)
        color = theme.RED if running else theme.GREEN
        hover = "#B3271E" if running else "#268A47"
        self.server_button.setStyleSheet(
            "QPushButton {"
            f"background: {color}; color: #ffffff; border: none; border-radius: 10px;"
            "font-size: 12pt; font-weight: 600; padding: 8px 16px;"
            "}"
            f"QPushButton:hover {{ background: {hover}; }}"
            "QPushButton:disabled { background: #a8abb2; color: #f0f0f0; }"
        )
        self.server_button.setIcon(_server_button_icon(running))
        self.server_button.setText(
            self.i18n.t("button.stop_server") if running else self.i18n.t("button.start_server")
        )
        if running:
            port = _http_port()
            self.server_state_label.setText(self.i18n.t("dashboard.server_running"))
            self.server_state_label.setStyleSheet(
                f"font-size: 10pt; font-weight: 600; color: {theme.GREEN}; background: transparent;"
            )
            self.server_detail_label.setText(f"http://{_detect_lan_ip()}:{port}")
        else:
            self.server_state_label.setText(self.i18n.t("dashboard.server_stopped"))
            self.server_state_label.setStyleSheet(
                f"font-size: 10pt; font-weight: 600; color: {theme.TEXT_MUTED}; background: transparent;"
            )
            self.server_detail_label.setText(self.i18n.t("dashboard.server_hint"))

    def set_runtime_status(self, text: str) -> None:
        if not self._server_running:
            return
        address = f"http://{_detect_lan_ip()}:{_http_port()}"
        self.server_detail_label.setText(f"{address}  ·  {text}" if text else address)

    def retranslate(self) -> None:
        self.switch_lock_notice.setText(self.i18n.t("dashboard.switches_locked"))
        self.realtime_group_label.setText(self.i18n.t("dashboard.group_realtime"))
        self.audio_group_label.setText(self.i18n.t("dashboard.group_audio"))
        self.cards["green"].set_title(self.i18n.t("mode.green"))
        self.cards["alpha"].set_title(self.i18n.t("mode.alpha"))
        self.cards["alpha2d"].set_title(self.i18n.t("alpha2d.button"))
        self.cards["two_dvr"].set_title(self.i18n.t("home.two_dvr_toggle"))
        self.cards["superres"].set_title(self.i18n.t("superres.realtime_title"))
        self.cards["rm"].set_title(self.i18n.t("rm.enabled"))
        self.cards["subtitle"].set_title(self.i18n.t("subtitle.enable"))
        self.cards["si"].set_title(self.i18n.t("si.enabled"))
        self.cards["light"].set_title(self.i18n.t("light_match.enabled"))
        self.project_link.setText(
            f'{self.i18n.t("project.url_label")}：<a href="{PROJECT_URL}">{PROJECT_URL}</a>'
        )
        self.set_server_running(self._server_running)
        self._update_summaries()
