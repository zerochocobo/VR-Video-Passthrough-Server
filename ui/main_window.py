from __future__ import annotations

import json
import os
import subprocess
import shutil
import threading
import urllib.request
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QWidget

from ui.diagnostics import build_diagnostic_report
from ui.i18n import I18n, system_language
from ui.metadata import load_app_metadata
from ui.pages.home_page import HOME_COMPACT_WIDTH, HOME_HEIGHT, HomePage
from ui.pages.offline_page import OfflinePage
from ui.pages.subtitle_page import SUBTITLE_PAGE_HEIGHT, SUBTITLE_PAGE_WIDTH, SubtitlePage
from ui.resources import app_icon
from ui.services.offline_process import OfflineProcess
from ui.services.server_process import ServerProcess
from ui.services.startup_status_poller import DEFAULT_PORT as STATUS_DEFAULT_PORT, StartupStatusPoller
from ui.settings import ROOT as UI_ROOT, Settings
from ui.styles import font_for_language
from ui.widgets.current_page_stack import CurrentPageStackedWidget
from ui.widgets.startup_overlay import StartupOverlay


SUPPORTED_LANGUAGES = ("zh_CN", "en_US", "ja_JP")
QT_MAX_WIDGET_SIZE = 16777215
OFFLINE_PAGE_WIDTH = 600
OFFLINE_PAGE_HEIGHT = 600
MIN_NVIDIA_COMPUTE_CAPABILITY = 7.5


def _parse_compute_capability(value: str) -> float | None:
    text = str(value or "").strip().lower().replace("sm_", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _detect_nvidia_gpu_support() -> tuple[bool, str, str] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        for root in (os.environ.get("SystemRoot"), os.environ.get("WINDIR")):
            if not root:
                continue
            system32_smi = Path(root) / "System32" / "nvidia-smi.exe"
            if system32_smi.exists():
                nvidia_smi = str(system32_smi)
                break
    if not nvidia_smi:
        return None
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    candidates: list[tuple[float, str, str]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        cc = _parse_compute_capability(parts[-1])
        if cc is None:
            continue
        candidates.append((cc, parts[0], parts[-1]))
    if not candidates:
        return None
    cc, name, cc_text = max(candidates, key=lambda item: item[0])
    return cc >= MIN_NVIDIA_COMPUTE_CAPABILITY, name, cc_text


class MainWindow(QMainWindow):
    runtime_status_received = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.metadata = load_app_metadata()
        self.setWindowIcon(app_icon())
        saved_language = self._configured_language()
        self.i18n = I18n(saved_language)
        self.server = ServerProcess()
        self.offline_process = OfflineProcess()
        self.stack = CurrentPageStackedWidget()
        self.home = HomePage(self.i18n, self.settings, self.metadata.display_version)
        self.offline = OfflinePage(self.i18n, self.settings, self.offline_process)
        self.subtitle = SubtitlePage(self.i18n, self.settings)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.offline)
        self.stack.addWidget(self.subtitle)
        self.setCentralWidget(self.stack)
        self.version_label = QLabel(self.metadata.display_version)
        self.version_label.setObjectName("VersionStatus")
        self.version_label.setStyleSheet("QLabel#VersionStatus { font-size: 9pt; }")
        self.runtime_status_label = QLabel("")
        self.runtime_status_label.setObjectName("RuntimeStatus")
        self.runtime_status_label.setStyleSheet("QLabel#RuntimeStatus { font-size: 9pt; }")
        self.runtime_status_label.setAlignment(Qt.AlignCenter)
        self.home.language.setFixedHeight(22)
        self.home.language.setStyleSheet("QComboBox { font-size: 9pt; padding: 0 6px; }")
        self.status_left_spacer = QWidget()
        self.status_left_spacer.setFixedWidth(20)
        self.statusBar().addWidget(self.status_left_spacer)
        self.statusBar().addWidget(self.home.language)
        self.statusBar().addWidget(self.runtime_status_label, 1)
        self.statusBar().addPermanentWidget(self.version_label)
        self.home.server_button.clicked.connect(self.toggle_server)
        self.home.offline_button.clicked.connect(self.open_offline)
        self.home.subtitle_style_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.subtitle))
        self.home.language.currentIndexChanged.connect(self.change_language)
        self.offline.back_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.subtitle.back_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.stack.currentChanged.connect(self._page_changed)
        self.server.output.connect(self.home.append_log)
        self.server.output.connect(self._scan_server_output_for_ready)
        self.server.state_changed.connect(self.home.set_server_running)
        self.server.state_changed.connect(self._server_state_changed)
        self.offline_process.state_changed.connect(self._offline_state_changed)
        # Startup overlay + status poller (lazy: created when first needed).
        self.startup_overlay: StartupOverlay | None = None
        self.status_poller = StartupStatusPoller(port=STATUS_DEFAULT_PORT, parent=self)
        self.status_poller.updated.connect(self._on_startup_status)
        self.status_poller.finished.connect(self._on_startup_finished)
        self.runtime_status_timer = QTimer(self)
        self.runtime_status_timer.setInterval(1500)
        self.runtime_status_timer.timeout.connect(self._poll_runtime_status)
        self.runtime_status_received.connect(self._apply_runtime_status)
        self._runtime_status_pending = False
        self._server_action_pending: str | None = None
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
        self.home.apply_heading_fonts()
        self.resize(HOME_COMPACT_WIDTH, HOME_HEIGHT)
        self._page_changed(self.stack.currentIndex())

    def _sync_language_combo(self) -> None:
        saved_language = self._configured_language()
        mapping = {"zh_CN": 0, "en_US": 1, "ja_JP": 2}
        self.home.language.blockSignals(True)
        self.home.language.setCurrentIndex(mapping.get(saved_language, 0))
        self.home.language.blockSignals(False)

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
        self.home.retranslate()
        self.home.set_server_running(self.server.is_running())
        self.offline.retranslate()
        self.subtitle.retranslate()

    def toggle_server(self) -> None:
        if self._server_action_pending is not None:
            return
        if self.server.is_running():
            self._set_server_action_pending("stopping")
            self.server.stop()
            return
        if self.offline_process.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_offline_first"))
            return
        self.settings.save()
        env = self.settings.server_env()
        env["PT_DEBUG_LOGS"] = "1" if self.home.debug_toggle.isChecked() else "0"
        self.home.clear_log()
        self._set_server_action_pending("starting")
        self.server.start(env)
        # Show the startup overlay so non-technical users see a friendly
        # progress dialog while the server warms up the GPU (potentially
        # 2+ minutes on first run with sm_120 GPUs).
        self._open_startup_overlay()
        self.status_poller.start()

    def open_offline(self) -> None:
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.stack.setCurrentWidget(self.offline)

    def _page_changed(self, index: int) -> None:
        if self.stack.widget(index) is self.home:
            self.home.sync_from_settings()
            self.home._adjust_window()
            return
        self.setMinimumWidth(0)
        self.setMaximumWidth(QT_MAX_WIDGET_SIZE)
        self.setMinimumHeight(0)
        self.setMaximumHeight(QT_MAX_WIDGET_SIZE)
        if self.stack.widget(index) is self.subtitle:
            self.resize(SUBTITLE_PAGE_WIDTH, SUBTITLE_PAGE_HEIGHT)
        elif self.stack.widget(index) is self.offline:
            self.offline.sync_from_settings()
            self.resize(OFFLINE_PAGE_WIDTH, OFFLINE_PAGE_HEIGHT)

    def _offline_state_changed(self, running: bool) -> None:
        if running:
            self.stack.setCurrentWidget(self.offline)

    def closeEvent(self, event) -> None:
        self.status_poller.stop()
        if self.startup_overlay is not None:
            self.startup_overlay.close()
        self.server.stop()
        self.offline_process.stop()
        super().closeEvent(event)

    # ---- Startup overlay glue ----

    def _open_startup_overlay(self) -> None:
        if self.startup_overlay is None:
            self.startup_overlay = StartupOverlay(self.i18n, self)
            self.startup_overlay.cancelRequested.connect(self._cancel_startup)
            self.startup_overlay.copyReportRequested.connect(self._copy_startup_report)
        overlay = self.startup_overlay
        overlay.reset()
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

    def _on_startup_status(self, status: dict) -> None:
        if self.startup_overlay is None or not self.startup_overlay.isVisible():
            return
        self.startup_overlay.apply_status(status)

    def _on_startup_finished(self, phase: str) -> None:
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
            self.startup_overlay.close()
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

    def _cancel_startup(self) -> None:
        # User explicitly aborted the long wait. Stop both the poller and the
        # server process so resources are released cleanly.
        self.status_poller.stop()
        self._set_server_action_pending("stopping")
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
        self.status_poller.stop()
        self.startup_overlay.apply_status(merged)
        self.startup_overlay.close()
        self._set_server_action_pending(None)

    def _server_state_changed(self, running: bool) -> None:
        if running:
            self.runtime_status_timer.start()
            self._poll_runtime_status()
        else:
            self.runtime_status_timer.stop()
            self._runtime_status_pending = False
            self.runtime_status_label.clear()
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
                        merged["detail"] = "server process exited before warmup completed"
                    self.startup_overlay.apply_status(merged)

    def _set_server_action_pending(self, action: str | None) -> None:
        self._server_action_pending = action
        self.home.server_button.setEnabled(action is None)

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
        if not self.server.is_running():
            self.runtime_status_label.clear()
            return
        if not status.get("ok"):
            self.runtime_status_label.clear()
            return
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
        if used is not None and total is not None:
            try:
                parts.append(f"{self.i18n.t('status.vram')} {float(used):.0f}/{float(total):.0f} MB")
            except (TypeError, ValueError):
                pass
        self.runtime_status_label.setText(" | ".join(parts))
