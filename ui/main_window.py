from __future__ import annotations

import json
import os
import threading
import urllib.request
from pathlib import Path

from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QMessageBox, QWidget

from ui.diagnostics import build_diagnostic_report
from ui.i18n import I18n, system_language
from ui.metadata import load_app_metadata
from ui.pages.dashboard_page import DASHBOARD_HEIGHT, DASHBOARD_WIDTH, DashboardPage
from ui.pages.face_beauty_page import FaceBeautyPage
from ui.pages.log_page import LogPage
from ui.pages.offline_page import OfflinePage
from ui.pages.rm_page import RmPage
from ui.pages.superres_page import SuperResPage
from ui.pages.settings_page import SettingsPage
from ui.pages.subtitle_page import SUBTITLE_PAGE_HEIGHT, SUBTITLE_PAGE_WIDTH, SubtitlePage
from ui.pages.tools_page import ToolsPage
from ui.pages.two_dvr_page import TwoDvrPage
from ui.resources import app_icon
from ui.services.offline_process import FaceBeautyProcess, OfflineProcess, RmProcess, SuperResProcess, TwoDvrProcess
from ui.services.server_process import ServerProcess
from ui.services.startup_diagnostics import LOG_PATH as UI_STARTUP_LOG_PATH, log_startup_event
from ui.services.startup_status_poller import DEFAULT_PORT as STATUS_DEFAULT_PORT, StartupStatusPoller
from ui.settings import ROOT as UI_ROOT, Settings
from ui.styles import font_for_language
from ui.widgets.current_page_stack import CurrentPageStackedWidget
from ui.widgets.nav_rail import NAV_WIDTH, NavRail
from ui.widgets.startup_overlay import StartupOverlay
from utils.gpu_cache_repair import cleanup_old_quarantines, discard_quarantine, quarantine_gpu_caches


SUPPORTED_LANGUAGES = ("zh_CN", "en_US", "ja_JP")
OFFLINE_PAGE_WIDTH = 600
OFFLINE_PAGE_HEIGHT = 600
STARTUP_BOOTSTRAP_HINT_ERRORS = 4      # 500 ms poll interval * 4 ~= 2 seconds.
STARTUP_BOOTSTRAP_LONG_ERRORS = 60     # 500 ms poll interval * 60 ~= 30 seconds.


class MainWindow(QMainWindow):
    runtime_status_received = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        os.environ["PT_HTTP_PORT"] = str(self.settings.http_port())
        self.metadata = load_app_metadata()
        self.setWindowIcon(app_icon())
        saved_language = self._configured_language()
        self.i18n = I18n(saved_language)
        self.server = ServerProcess()
        self.offline_process = OfflineProcess()
        self.two_dvr_process = TwoDvrProcess()
        self.rm_process = RmProcess()
        self.superres_process = SuperResProcess()
        self.face_beauty_process = FaceBeautyProcess()

        self.stack = CurrentPageStackedWidget()
        self.dashboard = DashboardPage(self.i18n, self.settings)
        self.tools = ToolsPage(self.i18n, self.settings)
        self.offline = OfflinePage(self.i18n, self.settings, self.offline_process)
        self.two_dvr = TwoDvrPage(self.i18n, self.settings, self.two_dvr_process)
        self.rm = RmPage(self.i18n, self.settings, self.rm_process)
        self.superres = SuperResPage(self.i18n, self.settings, self.superres_process)
        self.face_beauty = FaceBeautyPage(self.i18n, self.settings, self.face_beauty_process)
        self.subtitle = SubtitlePage(self.i18n, self.settings)
        self.log_page = LogPage(self.i18n)
        self.settings_page = SettingsPage(self.i18n, self.settings)
        for page in (
            self.dashboard,
            self.tools,
            self.subtitle,
            self.log_page,
            self.settings_page,
            self.offline,
            self.two_dvr,
            self.rm,
            self.superres,
            self.face_beauty,
        ):
            self.stack.addWidget(page)

        self.nav = NavRail()
        self.nav.add_item("home", "home")
        self.nav.add_item("tools", "tools")
        self.nav.add_item("subtitle", "subtitle")
        self.nav.add_item("log", "log")
        self.nav.add_item("settings", "settings", bottom=True)
        self.nav.page_selected.connect(self._nav_selected)
        self._nav_pages = {
            "home": self.dashboard,
            "tools": self.tools,
            "subtitle": self.subtitle,
            "log": self.log_page,
            "settings": self.settings_page,
        }

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.nav)
        central_layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.version_label = QLabel(self.metadata.display_version)
        self.version_label.setObjectName("VersionStatus")
        self.version_label.setStyleSheet("QLabel#VersionStatus { font-size: 9pt; }")
        self.statusBar().addPermanentWidget(self.version_label)

        self.dashboard.server_button.clicked.connect(self.toggle_server)
        self.dashboard.open_subtitle_page.connect(lambda: self._show_page("subtitle"))
        self.tools.open_offline.connect(self.open_offline)
        self.tools.open_two_dvr.connect(self.open_two_dvr)
        self.tools.open_rm.connect(self.open_rm)
        self.tools.open_superres.connect(self.open_superres)
        self.tools.open_face_beauty.connect(self.open_face_beauty)
        self.settings_page.language.currentIndexChanged.connect(self.change_language)
        self.settings_page.rm_card_visibility_changed.connect(self.dashboard.set_rm_card_visible)
        self.settings_page.rm_card_visibility_changed.connect(self.tools.set_rm_card_visible)
        self.settings_page.face_beauty_card_visibility_changed.connect(self.dashboard.set_face_beauty_card_visible)
        self.settings_page.face_beauty_card_visibility_changed.connect(self.tools.set_face_beauty_card_visible)
        self.settings_page.gpu_cache_repair_requested.connect(self._repair_gpu_cache_and_restart)
        self.offline.back_button.clicked.connect(lambda: self._show_page("tools"))
        self.two_dvr.back_button.clicked.connect(lambda: self._show_page("tools"))
        self.rm.back_button.clicked.connect(lambda: self._show_page("tools"))
        self.superres.back_button.clicked.connect(lambda: self._show_page("home"))
        self.face_beauty.back_button.clicked.connect(lambda: self._show_page("tools"))
        self.subtitle.back_button.clicked.connect(lambda: self._show_page("home"))

        self.server.output.connect(self.log_page.append_log)
        self.server.output.connect(self._scan_server_output_for_ready)
        self.server.state_changed.connect(self.dashboard.set_server_running)
        self.server.state_changed.connect(self._server_state_changed)
        self.offline_process.state_changed.connect(self._offline_state_changed)
        self.two_dvr_process.state_changed.connect(self._two_dvr_state_changed)
        self.rm_process.state_changed.connect(self._rm_state_changed)
        self.superres_process.state_changed.connect(self._superres_state_changed)
        self.face_beauty_process.state_changed.connect(self._face_beauty_state_changed)

        # Startup overlay + status poller (lazy: created when first needed).
        self.startup_overlay: StartupOverlay | None = None
        self.status_poller = StartupStatusPoller(port=STATUS_DEFAULT_PORT, parent=self)
        self.status_poller.updated.connect(self._on_startup_status)
        self.status_poller.finished.connect(self._on_startup_finished)
        self.status_poller.error.connect(self._on_startup_error)
        self._poll_error_streak = 0
        self._poll_first_success = False
        log_startup_event("main_window_init", status_port=STATUS_DEFAULT_PORT)
        self.runtime_status_timer = QTimer(self)
        self.runtime_status_timer.setInterval(1500)
        self.runtime_status_timer.timeout.connect(self._poll_runtime_status)
        self.runtime_status_received.connect(self._apply_runtime_status)
        self._runtime_status_pending = False
        self._server_action_pending: str | None = None
        self._gpu_repair_rebuild_trt = False
        self._gpu_repair_trt_requested = False
        self._gpu_repair_quarantine: Path | None = None
        self._gpu_repair_pending = False
        self._last_startup_provider_kind = ""
        removed_quarantines = cleanup_old_quarantines(UI_ROOT / "runtime_cache")
        if removed_quarantines:
            log_startup_event("gpu_cache_old_quarantines_removed", names=list(removed_quarantines))
        self._sync_language_combo()
        self.retranslate()
        app_font = QFont()
        app_font.setPointSize(11)
        self.setFont(app_font)
        self.setStyleSheet(
            "QWidget { font-size: 11pt; } "
            "QPushButton { font-size: 12pt; padding: 8px 12px; } "
            "QGroupBox { font-weight: 600; } "
            "QComboBox, QLineEdit, QDoubleSpinBox { padding: 4px; }"
        )
        self.setMinimumSize(NAV_WIDTH + 560, 540)
        self.nav.select("home")
        self._show_page("home")

    def _sync_language_combo(self) -> None:
        saved_language = self._configured_language()
        mapping = {"zh_CN": 0, "en_US": 1, "ja_JP": 2}
        self.settings_page.language.blockSignals(True)
        self.settings_page.language.setCurrentIndex(mapping.get(saved_language, 0))
        self.settings_page.language.blockSignals(False)

    def change_language(self, index: int) -> None:
        lang = SUPPORTED_LANGUAGES[index]
        self.i18n.load(lang)
        self.settings.data["language"] = lang
        self.settings.save()
        self.setFont(font_for_language(lang))
        self.retranslate()

    def _configured_language(self) -> str:
        saved_language = str(self.settings.data.get("language") or "")
        if saved_language not in SUPPORTED_LANGUAGES:
            saved_language = system_language()
            self.settings.data["language"] = saved_language
            self.settings.save()
        return saved_language

    def retranslate(self) -> None:
        self.i18n.load(self._configured_language())
        self.setFont(font_for_language(self.i18n.language))
        self.setWindowTitle(f"{self.i18n.t('app.title')} ({self.metadata.display_version})")
        self.version_label.setText(self.metadata.display_version)
        self.nav.set_item_text("home", self.i18n.t("nav.home"))
        self.nav.set_item_text("tools", self.i18n.t("nav.tools"))
        self.nav.set_item_text("subtitle", self.i18n.t("nav.subtitle"))
        self.nav.set_item_text("log", self.i18n.t("nav.log"))
        self.nav.set_item_text("settings", self.i18n.t("nav.settings"))
        self.dashboard.retranslate()
        self.tools.retranslate()
        self.log_page.retranslate()
        self.settings_page.retranslate()
        self.offline.retranslate()
        self.two_dvr.retranslate()
        self.rm.retranslate()
        self.superres.retranslate()
        self.face_beauty.retranslate()
        self.subtitle.retranslate()
        self.dashboard.set_server_running(self.server.is_running())

    # ---- navigation ----

    def _nav_selected(self, key: str) -> None:
        self._show_page(key)

    def _show_page(self, key: str) -> None:
        page = self._nav_pages.get(key)
        if page is None:
            return
        self.nav.select(key)
        self.stack.setCurrentWidget(page)
        if page is self.dashboard:
            self.dashboard.sync_from_settings()
        elif page is self.settings_page:
            self.settings_page.sync_from_settings()
        self._resize_for(page)

    def _show_sub_page(self, page: QWidget) -> None:
        self.nav.select("tools")
        self.stack.setCurrentWidget(page)
        self._resize_for(page)

    def _resize_for(self, page: QWidget) -> None:
        if page is self.subtitle:
            content = QSize(SUBTITLE_PAGE_WIDTH, SUBTITLE_PAGE_HEIGHT)
        elif page in (self.offline, self.two_dvr, self.rm, self.superres, self.face_beauty):
            content = QSize(OFFLINE_PAGE_WIDTH, OFFLINE_PAGE_HEIGHT)
        else:
            content = QSize(DASHBOARD_WIDTH, self.dashboard.preferred_height())
        self.resize(NAV_WIDTH + content.width(), max(content.height(), 560))

    def toggle_server(self) -> None:
        if self._server_action_pending is not None:
            return
        if self.server.is_running():
            self._set_server_action_pending("stopping")
            self.server.stop()
            return
        if (
            self.offline_process.is_running()
            or self.two_dvr_process.is_running()
            or self.rm_process.is_running()
            or self.superres_process.is_running()
            or self.face_beauty_process.is_running()
        ):
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_offline_first"))
            return
        self.settings.save()
        env = self.settings.server_env()
        if self._gpu_repair_rebuild_trt:
            env["PT_ONNX_PROVIDERS"] = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
            env["PT_GPU_REPAIR_REBUILD_TRT"] = "1"
            self._gpu_repair_rebuild_trt = False
        env["PT_DEBUG_LOGS"] = "1" if self.log_page.debug_toggle.isChecked() else "0"
        # Keep the UI process's own port view (dashboard address, runtime
        # status poller, live control PUTs) aligned with the server we launch.
        os.environ["PT_HTTP_PORT"] = env["PT_HTTP_PORT"]
        log_startup_event(
            "server_start_requested",
            status_port=STATUS_DEFAULT_PORT,
            pt_startup_status_port=env.get("PT_STARTUP_STATUS_PORT"),
            providers=env.get("PT_ONNX_PROVIDERS"),
            debug_logs=env.get("PT_DEBUG_LOGS"),
            ui_startup_log=str(UI_STARTUP_LOG_PATH),
        )
        self.log_page.clear_log()
        self._set_server_action_pending("starting")
        self.server.start(env)
        log_startup_event("server_start_called", server_running=self.server.is_running(), pid=self.server.process.process_id())
        # Show the startup overlay so non-technical users see a friendly
        # progress dialog while the server warms up the GPU (potentially
        # 2+ minutes on first run with sm_120 GPUs).
        self._open_startup_overlay()
        self.status_poller.start()

    def open_offline(self) -> None:
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.offline.sync_from_settings()
        self._show_sub_page(self.offline)

    def open_two_dvr(self) -> None:
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.two_dvr.sync_from_settings()
        self._show_sub_page(self.two_dvr)

    def open_rm(self) -> None:
        if not bool(self.settings.data.get("rm_card_visible")):
            return
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.rm.sync_from_settings()
        self._show_sub_page(self.rm)

    def open_superres(self) -> None:
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.superres.sync_from_settings()
        self._show_sub_page(self.superres)

    def open_face_beauty(self) -> None:
        if not bool(self.settings.data.get("face_beauty_card_visible")):
            return
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.face_beauty.sync_from_settings()
        self._show_sub_page(self.face_beauty)

    def _offline_state_changed(self, running: bool) -> None:
        if running:
            self._show_sub_page(self.offline)

    def _two_dvr_state_changed(self, running: bool) -> None:
        if running:
            self._show_sub_page(self.two_dvr)

    def _rm_state_changed(self, running: bool) -> None:
        if running:
            self._show_sub_page(self.rm)

    def _superres_state_changed(self, running: bool) -> None:
        if running:
            self._show_sub_page(self.superres)

    def _face_beauty_state_changed(self, running: bool) -> None:
        if running:
            self._show_sub_page(self.face_beauty)

    def closeEvent(self, event) -> None:
        self.status_poller.stop()
        if self.startup_overlay is not None:
            self.startup_overlay.close()
        self.server.stop()
        self.offline_process.stop()
        self.two_dvr_process.stop()
        self.rm_process.stop()
        self.superres_process.stop()
        self.face_beauty_process.stop()
        super().closeEvent(event)

    # ---- Startup overlay glue ----

    def _open_startup_overlay(self) -> None:
        if self.startup_overlay is None:
            self.startup_overlay = StartupOverlay(self.i18n, self)
            self.startup_overlay.cancelRequested.connect(self._cancel_startup)
            self.startup_overlay.copyReportRequested.connect(self._copy_startup_report)
            self.startup_overlay.repairGpuCacheRequested.connect(self._repair_gpu_cache_and_restart)
        overlay = self.startup_overlay
        self._poll_error_streak = 0
        self._poll_first_success = False
        self._last_startup_provider_kind = ""
        log_startup_event("overlay_open")
        overlay.reset()
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

    def _on_startup_status(self, status: dict) -> None:
        self._poll_first_success = True
        self._poll_error_streak = 0
        provider_kind = str(status.get("provider_kind") or "")
        if provider_kind:
            self._last_startup_provider_kind = provider_kind
        log_startup_event(
            "ui_status_received",
            phase=status.get("phase"),
            step=status.get("step"),
            progress=status.get("progress"),
            elapsed_sec=status.get("elapsed_sec"),
            provider_kind=status.get("provider_kind"),
            visible=bool(self.startup_overlay is not None and self.startup_overlay.isVisible()),
        )
        if self.startup_overlay is None or not self.startup_overlay.isVisible():
            return
        self.startup_overlay.apply_status(status)

    def _on_startup_error(self, message: str) -> None:
        log_startup_event(
            "ui_status_error",
            message=message,
            streak=self._poll_error_streak + 1,
            first_success=self._poll_first_success,
            visible=bool(self.startup_overlay is not None and self.startup_overlay.isVisible()),
        )
        if self._poll_first_success:
            return
        if self.startup_overlay is None or not self.startup_overlay.isVisible():
            return
        self._poll_error_streak += 1
        if self._poll_error_streak >= STARTUP_BOOTSTRAP_LONG_ERRORS:
            log_startup_event("overlay_bootstrapping_long", streak=self._poll_error_streak)
            self.startup_overlay.show_bootstrapping_hint_long()
        elif self._poll_error_streak >= STARTUP_BOOTSTRAP_HINT_ERRORS:
            log_startup_event("overlay_bootstrapping", streak=self._poll_error_streak)
            self.startup_overlay.show_bootstrapping_hint()

    def _on_startup_finished(self, phase: str) -> None:
        log_startup_event("ui_startup_finished", phase=phase)
        if self.startup_overlay is None:
            return
        if phase == "listening":
            self._set_server_action_pending(None)
            # The server has bound the DLNA HTTP port and is truly ready to
            # serve clients. Brief "done" flash so the user sees a confirmation
            # before the overlay disappears.
            merged = dict(self.startup_overlay.last_status() or {})
            merged["phase"] = phase
            merged["progress"] = 1.0
            self.startup_overlay.apply_status(merged)
            log_startup_event("overlay_close_finished", phase=phase, via="poller")
            self.startup_overlay.close()
            self._finish_gpu_cache_repair()
        elif phase == "failed":
            self._set_server_action_pending(None)
            # Keep overlay visible so the user can read the failure and copy a
            # report. The server process exits on its own. Merge into the
            # last seen status so that step/cold/gpu_name/reason emitted by
            # the server before the crash are preserved in the report.
            merged = dict(self.startup_overlay.last_status() or {})
            merged["phase"] = "failed"
            if not merged.get("message"):
                merged["message"] = self.i18n.t("startup.failed_generic")
            self.startup_overlay.apply_status(merged)
            self._mark_gpu_cache_repair_failed(merged)

    def _cancel_startup(self) -> None:
        # User explicitly aborted the long wait. Stop both the poller and the
        # server process so resources are released cleanly.
        self._mark_gpu_cache_repair_failed()
        self.status_poller.stop()
        self._set_server_action_pending("stopping")
        log_startup_event("startup_cancel_requested")
        self.server.stop()
        if self.startup_overlay is not None:
            self.startup_overlay.close()

    def _server_log_path(self) -> Path:
        """Return where the server writes ``server.log`` (rotated).

        Mirrors :mod:`utils.logger` so the diagnostic report can include the
        most recent crash output without depending on the server module being
        importable from the UI process.
        """
        return UI_ROOT / "debug_output" / "server.log"

    def _copy_startup_report(self) -> None:
        marker_path = UI_ROOT / "runtime_cache" / "gpu_warmup_marker.json"
        log_path = self._server_log_path()
        status = self.startup_overlay.last_status() if self.startup_overlay is not None else None
        report = build_diagnostic_report(
            app_version=self.metadata.display_version,
            language=self.i18n.language,
            last_status=status,
            marker_path=marker_path if marker_path.exists() else marker_path,
            log_path=log_path,
        )
        StartupOverlay.copy_to_clipboard(report)
        if self.startup_overlay is not None:
            self.startup_overlay.update_diagnostic_text(report)
            self.startup_overlay.show_copy_confirmation()

    def _scan_server_output_for_ready(self, text: str) -> None:
        """Fallback close: watch server stdout for uvicorn's ready banner.

        The primary path is the /status poller on 127.0.0.1:8299. On a machine
        where 8299 is blocked (firewall, port conflict, IPv4 disabled, etc.)
        we still need the overlay to disappear once the server is actually
        listening. uvicorn always prints "Uvicorn running on http://0.0.0.0:..."
        when ready, and we capture stdout via ServerProcess.output regardless
        of the status endpoint's health.
        """
        if self.startup_overlay is None or not self.startup_overlay.isVisible():
            return
        if "Uvicorn running on" not in text and "Application startup complete" not in text:
            return
        last = self.startup_overlay.last_status()
        if last is not None and str(last.get("phase") or "") == "listening":
            return  # Normal path already closed (or is about to close) the overlay.
        merged = dict(last or {})
        merged["phase"] = "listening"
        merged["progress"] = 1.0
        merged["message"] = (merged.get("message") or "").strip() or self.i18n.t("startup.complete")
        log_startup_event(
            "stdout_ready_fallback",
            last_phase=(last or {}).get("phase") if isinstance(last, dict) else None,
            text=text.strip()[:200],
        )
        self.status_poller.stop()
        self.startup_overlay.apply_status(merged)
        log_startup_event("overlay_close_finished", phase="listening", via="stdout_fallback")
        self.startup_overlay.close()
        self._set_server_action_pending(None)
        self._finish_gpu_cache_repair()

    def _server_state_changed(self, running: bool) -> None:
        log_startup_event(
            "server_state_changed",
            running=running,
            pid=self.server.process.process_id(),
            exit_code=self.server.last_exit_code if not running else None,
        )
        if running:
            self.runtime_status_timer.start()
            self._poll_runtime_status()
        else:
            self.runtime_status_timer.stop()
            self._runtime_status_pending = False
            self.settings_page.set_runtime_provider_kind("")
        if not running:
            self._set_server_action_pending(None)
            self.status_poller.stop()
            if self.startup_overlay is not None and self.startup_overlay.isVisible():
                last = self.startup_overlay.last_status() or {}
                phase = str(last.get("phase") or "")
                if phase == "listening":
                    # Clean stop after the server became ready — just hide.
                    # NOTE: ``warmed`` is intentionally NOT treated as success
                    # here. After warmed the server still has to install
                    # firewall rules, start SSDP, and bind the DLNA HTTP port.
                    # A process that died at ``warmed`` is a failure.
                    log_startup_event("overlay_close_server_stopped", last_phase=phase)
                    self.startup_overlay.close()
                else:
                    # Server process exited before becoming ready. The 8299
                    # /status endpoint is normally shut down within milliseconds
                    # of the failure being published, so the 500 ms poller can
                    # easily miss the "failed" transition. Synthesize the
                    # terminal state here so the overlay flips to the failed
                    # view (with the "Copy hardware report" button still
                    # available) instead of sitting on a stale "warming"
                    # snapshot with the indeterminate bar spinning forever.
                    merged = dict(last)
                    merged["phase"] = "failed"
                    if not merged.get("message"):
                        merged["message"] = self.i18n.t("startup.failed_generic")
                    if not merged.get("detail"):
                        exit_code = self.server.last_exit_code
                        exit_text = ""
                        if exit_code is not None:
                            exit_text = f" (exit code {exit_code} / 0x{exit_code & 0xFFFFFFFF:08X})"
                        merged["detail"] = f"server process exited before warmup completed{exit_text}"
                    log_startup_event("overlay_failed_server_stopped", last_phase=phase, detail=merged.get("detail"))
                    self.startup_overlay.apply_status(merged)
                    self._mark_gpu_cache_repair_failed(merged)

    def _repair_gpu_cache_and_restart(self) -> None:
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            if self.startup_overlay is not None:
                self.startup_overlay.repair_gpu_cache_btn.setEnabled(True)
            return
        if any(
            process.is_running()
            for process in (self.offline_process, self.two_dvr_process, self.rm_process,
                            self.superres_process, self.face_beauty_process)
        ):
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_offline_first"))
            if self.startup_overlay is not None:
                self.startup_overlay.repair_gpu_cache_btn.setEnabled(True)
            return
        answer = QMessageBox.question(
            self,
            self.i18n.t("gpu_repair.confirm_title"),
            self.i18n.t("gpu_repair.confirm_text"),
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            if self.startup_overlay is not None:
                self.startup_overlay.repair_gpu_cache_btn.setEnabled(True)
            return

        rebuild_trt = str(self.settings.data.get("inference_backend") or "cuda").lower() == "tensorrt"
        # The user-facing repair targets the application's standard cache root.
        # Diagnostic PT_* cache path overrides are intentionally left untouched.
        result = quarantine_gpu_caches(UI_ROOT / "runtime_cache")
        log_startup_event(
            "gpu_cache_repair",
            moved=list(result.moved),
            errors=list(result.errors),
            quarantine=str(result.quarantine_dir or ""),
        )
        if result.errors:
            QMessageBox.critical(
                self,
                self.i18n.t("gpu_repair.failed_title"),
                self.i18n.t("gpu_repair.failed_text").format(error="\n".join(result.errors)),
            )
            if self.startup_overlay is not None:
                self.startup_overlay.repair_gpu_cache_btn.setEnabled(True)
            return

        self._gpu_repair_quarantine = result.quarantine_dir
        self._gpu_repair_pending = True
        self._gpu_repair_trt_requested = rebuild_trt
        self._gpu_repair_rebuild_trt = rebuild_trt
        self._set_server_action_pending(None)
        self.toggle_server()

    def _finish_gpu_cache_repair(self) -> None:
        if not self._gpu_repair_pending:
            return
        self._gpu_repair_pending = False
        trt_requested = self._gpu_repair_trt_requested
        self._gpu_repair_trt_requested = False
        provider_kind = self._last_startup_provider_kind
        if self.startup_overlay is not None:
            provider_kind = str((self.startup_overlay.last_status() or {}).get("provider_kind") or provider_kind)
        quarantine = self._gpu_repair_quarantine
        self._gpu_repair_quarantine = None
        if quarantine is not None:
            try:
                discard_quarantine(quarantine)
            except Exception as exc:
                log_startup_event("gpu_cache_quarantine_cleanup_failed", path=str(quarantine), error=str(exc))
            else:
                log_startup_event("gpu_cache_repair_succeeded", quarantine=str(quarantine))
        QMessageBox.information(
            self,
            self.i18n.t("gpu_repair.success_title"),
            self.i18n.t(
                "gpu_repair.success_text_trt"
                if trt_requested and provider_kind == "trt"
                else "gpu_repair.success_text_cuda_fallback"
                if trt_requested and provider_kind == "cuda"
                else "gpu_repair.success_text"
            ),
        )

    def _mark_gpu_cache_repair_failed(self, status: dict | None = None) -> None:
        if not self._gpu_repair_pending:
            return
        quarantine = self._gpu_repair_quarantine
        self._gpu_repair_pending = False
        self._gpu_repair_quarantine = None
        self._gpu_repair_rebuild_trt = False
        self._gpu_repair_trt_requested = False
        log_startup_event(
            "gpu_cache_repair_start_failed",
            quarantine=str(quarantine or ""),
            detail=str((status or {}).get("detail") or ""),
        )

    def _set_server_action_pending(self, action: str | None) -> None:
        self._server_action_pending = action
        self.dashboard.server_button.setEnabled(action is None)

    def _runtime_status_url(self) -> str:
        port = str(os.environ.get("PT_HTTP_PORT") or "8200").strip() or "8200"
        return f"http://127.0.0.1:{port}/runtime_status"

    def _poll_runtime_status(self) -> None:
        if self._runtime_status_pending or not self.server.is_running():
            return
        self._runtime_status_pending = True
        url = self._runtime_status_url()

        def worker() -> None:
            payload: dict
            try:
                with urllib.request.urlopen(url, timeout=0.7) as response:
                    raw = response.read(8192)
                payload = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                payload = {"ok": False}
            self.runtime_status_received.emit(payload)

        threading.Thread(target=worker, name="runtime-status-poll", daemon=True).start()

    def _apply_runtime_status(self, status: dict) -> None:
        self._runtime_status_pending = False
        if not self.server.is_running() or not status.get("ok"):
            self.dashboard.set_runtime_status("")
            self.settings_page.set_runtime_provider_kind("")
            return
        self.settings_page.set_runtime_provider_kind(str(status.get("provider_kind") or ""))
        used = status.get("vram_used_mib")
        total = status.get("vram_total_mib")
        active = bool(status.get("active"))
        parts: list[str] = []
        if active:
            produced_fps = float(status.get("produced_fps") or 0.0)
            output_fps = float(status.get("output_fps") or 0.0)
            fps = produced_fps if produced_fps > 0 else output_fps
            if fps > 0:
                parts.append(f"FPS {fps:.1f}")
            else:
                parts.append("FPS --")
        if used is not None and total is not None:
            try:
                parts.append(f"{self.i18n.t('status.vram')} {float(used):.0f}/{float(total):.0f} MB")
            except (TypeError, ValueError):
                pass
        elif active:
            parts.append(f"{self.i18n.t('status.vram')} --")
        self.dashboard.set_runtime_status(" | ".join(parts))
