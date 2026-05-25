"""Non-modal startup overlay shown while the server is warming the GPU.

Goals (driven by the "面向非技术型用户" UX requirement):
  - Friendly language. Avoid words like "JIT", "PTX", "cubin"; use plain
    phrases such as "First-time GPU initialization" so non-technical users
    are not scared away.
  - Set expectations BEFORE the long wait. The overlay shows the predicted
    duration as soon as the server starts (from predict_warmup_state in the
    server process). The user knows ahead of time whether to expect 5s or
    150s, and why.
  - Visible animation. A determinate progress bar and an animated ellipsis
    in the status label reassure the user the program is alive.
  - Easy escape hatch. One-click "Copy hardware report" puts a detailed
    text report on the clipboard so the user can paste it into chat/forum
    to get help.

This widget is intentionally implemented as a non-modal QDialog rather than
an embedded widget so it can be raised over any current page without
restructuring HomePage's layout. It can be cancelled by the user (which
should be wired to stop the server process).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Heuristic step ordering used for the labelled checklist on the left side.
# Server-side `step` strings map into these indices; unknown values fall
# back to the current step_index from /status.
_STEP_KEYS = (
    "predict_probe",
    "predict",
    "warmup_start",
    "matter_singleton",
    "static_trt_preload",
    "ort_iobinding_runs",
    "composite_jit",
    "reset_state",
    "nvenc_preflight",
    "firewall",
    "ssdp",
    "http_starting",
    "listening",
    "warmed",
)

_REASSURANCE_TIERS = (
    (20.0, "startup.reassure.t20", "info"),
    (45.0, "startup.reassure.t45", "info"),
    (90.0, "startup.reassure.t90", "warn"),
    (180.0, "startup.reassure.t180", "warn"),
)

_HINT_STYLE_INFO = (
    "QLabel { color: #444; font-size: 9pt; background: #FFF7E0;"
    " border: 1px solid #EBC97A; border-radius: 6px; padding: 8px; }"
)
_HINT_STYLE_WARN = (
    "QLabel { color: #3D2B00; font-size: 9pt; background: #FFF0D6;"
    " border: 2px solid #E59A2F; border-radius: 6px; padding: 8px; }"
)


class StartupOverlay(QDialog):
    """Friendly progress overlay shown while the server initializes the GPU."""

    cancelRequested = Signal()
    copyReportRequested = Signal()
    showDetailsToggled = Signal(bool)

    def __init__(self, i18n, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.setObjectName("StartupOverlay")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, False)
        self.setModal(False)
        self._ellipsis_phase = 0
        self._ellipsis_timer = QTimer(self)
        self._ellipsis_timer.setInterval(450)
        self._ellipsis_timer.timeout.connect(self._tick_ellipsis)
        self._base_message = ""
        self._last_status: dict | None = None

        self.title_label = QLabel()
        title_font = self.title_label.font()
        title_font.setPointSize(13)
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setMinimumHeight(48)

        self.step_label = QLabel()
        self.step_label.setStyleSheet("QLabel { color: #303133; font-weight: 600; }")

        self.eta_label = QLabel()
        self.eta_label.setStyleSheet("QLabel { color: #606266; }")

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(14)

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(_HINT_STYLE_INFO)
        self.hint_label.setVisible(False)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setVisible(False)
        self.details_text.setFixedHeight(140)
        self.details_text.setStyleSheet(
            "QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; }"
        )

        self.toggle_details_btn = QPushButton()
        self.toggle_details_btn.setCheckable(True)
        self.toggle_details_btn.toggled.connect(self._on_toggle_details)

        self.copy_report_btn = QPushButton()
        self.copy_report_btn.clicked.connect(self._on_copy_report_clicked)

        self.cancel_btn = QPushButton()
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)

        button_row = QHBoxLayout()
        button_row.addWidget(self.toggle_details_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.copy_report_btn)
        button_row.addWidget(self.cancel_btn)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(self.title_label)
        layout.addWidget(self.message_label)
        layout.addWidget(self.step_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.eta_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.details_text)
        layout.addWidget(divider)
        layout.addLayout(button_row)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(460)
        self.retranslate()

    # ---- Public API ----

    def reset(self) -> None:
        self._last_status = None
        self._ellipsis_phase = 0
        self._base_message = self.i18n.t("startup.connecting")
        self.title_label.setText(self.i18n.t("startup.title_starting"))
        self.message_label.setText(self._base_message)
        self.step_label.setText("")
        self.eta_label.setText("")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.hint_label.setText("")
        self.hint_label.setStyleSheet(_HINT_STYLE_INFO)
        self.hint_label.setVisible(False)
        self.details_text.setVisible(False)
        self.details_text.clear()
        self.toggle_details_btn.setText(self.i18n.t("startup.show_details"))
        self.copy_report_btn.setText(self.i18n.t("startup.copy_report"))
        self.toggle_details_btn.setChecked(False)
        self._ellipsis_timer.start()

    def apply_status(self, status: dict) -> None:
        """Update the overlay from a /status response dict."""
        self._last_status = status
        phase = str(status.get("phase") or "")
        step = str(status.get("step") or "")
        progress_value = float(status.get("progress") or 0.0)
        step_index = int(status.get("step_index") or 0)
        step_total = int(status.get("step_total") or 0)
        eta = float(status.get("eta_sec") or 0.0)
        elapsed = float(status.get("elapsed_sec") or 0.0)
        cold = bool(status.get("cold"))
        known_slow = bool(status.get("is_known_slow"))
        gpu = str(status.get("gpu_name") or "")
        cc = str(status.get("compute_capability") or "")
        reason = str(status.get("reason") or "")
        message = str(status.get("message") or "")
        provider_kind = str(status.get("provider_kind") or "")
        step_text = self._step_text(step, status)
        step_translated = self._step_has_translation(step, status)

        # ----- Title (phase friendly name) -----
        if phase == "listening":
            self.title_label.setText(self.i18n.t("startup.title_ready"))
        elif phase == "warmed":
            # GPU warmup finished but the server still has to install firewall
            # rules, start SSDP, and bind the DLNA HTTP port. Show a
            # transitional title rather than "ready" so the user doesn't think
            # the overlay is stuck after it's actually progressing.
            self.title_label.setText(self.i18n.t("startup.title_verifying"))
        elif phase == "failed":
            self.title_label.setText(self.i18n.t("startup.title_failed"))
        elif phase == "starting":
            self.title_label.setText(self.i18n.t("startup.title_starting"))
        elif phase in {"firewall", "ssdp", "http_starting"}:
            self.title_label.setText(self.i18n.t("startup.title_starting"))
        elif phase == "warming":
            if provider_kind == "trt":
                self.title_label.setText(self.i18n.t("startup.title_trt_loading"))
            elif cold and known_slow:
                self.title_label.setText(self.i18n.t("startup.title_first_run_slow"))
            elif cold:
                self.title_label.setText(self.i18n.t("startup.title_first_run"))
            else:
                self.title_label.setText(self.i18n.t("startup.title_verifying"))
        else:
            self.title_label.setText(self.i18n.t("startup.title_starting"))

        # ----- Friendly message body -----
        friendly: list[str] = []
        if step_text and step_translated:
            friendly.append(step_text)
        elif message:
            friendly.append(message)
        if gpu:
            friendly.append(self.i18n.t("startup.gpu_label").format(gpu=gpu, cc=cc or "?"))
        self._base_message = "\n".join(friendly) if friendly else self.i18n.t("startup.connecting")
        self.message_label.setText(self._base_message)

        if step_text and step_total > 0:
            shown_index = max(1, min(step_total, step_index or self._step_index_for(step)))
            self.step_label.setText(f"{shown_index}/{step_total}  {step_text}")
        elif step_text:
            self.step_label.setText(step_text)
        else:
            self.step_label.setText("")

        # ----- ETA / elapsed line -----
        if phase == "listening":
            self.eta_label.setText(self.i18n.t("startup.complete"))
            self._ellipsis_timer.stop()
        elif phase == "warmed":
            # Warmup is done, but the server isn't listening yet. Keep the
            # ellipsis animating so the user knows we're still progressing.
            self.eta_label.setText("")
        elif eta > 0 or elapsed > 0:
            self.eta_label.setText(
                self.i18n.t("startup.eta_template").format(
                    elapsed=int(elapsed),
                    eta=max(0, int(eta - elapsed)) if eta > elapsed else int(eta),
                )
            )
        else:
            self.eta_label.setText("")

        # ----- Progress bar -----
        if phase == "listening":
            # Pin to 100% and switch out of busy mode if we were in it.
            if self.progress.minimum() == 0 and self.progress.maximum() == 0:
                self.progress.setRange(0, 1000)
            self.progress.setValue(self.progress.maximum() or 1000)
        elif phase in {"warmed", "firewall", "ssdp", "http_starting"}:
            # Warmup is finished but uvicorn hasn't bound the port yet. Show
            # near-full progress with an indeterminate marquee so the user
            # knows the program is still doing something between the GPU
            # readiness signal and the network being live.
            self.progress.setRange(0, 0)
        elif phase == "warming" and elapsed <= 0.1 and progress_value <= 0.11:
            # The server emits a single update right before the long blocking
            # ORT session load + warmup runs, then goes silent for tens of
            # seconds. A static 30% bar feels frozen to a non-technical user;
            # an indeterminate (marquee) bar keeps animating so they know the
            # program is alive while we wait for the next /status update.
            self.progress.setRange(0, 0)
        else:
            if self.progress.minimum() == 0 and self.progress.maximum() == 0:
                self.progress.setRange(0, 1000)
            # If progress is reported, use it; otherwise infer from elapsed/eta.
            if progress_value > 0:
                value = int(max(0.0, min(1.0, progress_value)) * self.progress.maximum())
            elif step_total > 0 and step_index > 0:
                value = int(max(0.0, min(0.99, step_index / step_total)) * self.progress.maximum())
            elif eta > 0:
                value = int(max(0.0, min(0.99, elapsed / eta)) * self.progress.maximum())
            else:
                value = 0
            # Never let the bar look stuck near 0 once we've heard from the server.
            value = max(value, 30 if phase == "warming" else 0)
            self.progress.setValue(value)

        # ----- Advisory / reassurance -----
        self._apply_hint(status, phase, cold, known_slow, provider_kind, elapsed)

        # ----- Details panel content (raw status) -----
        if self.details_text.isVisible():
            self.details_text.setPlainText(self._format_details(status, reason))

    def update_diagnostic_text(self, text: str) -> None:
        """Show a pre-built diagnostic report inside the details panel."""
        self.details_text.setPlainText(text)
        if not self.toggle_details_btn.isChecked():
            self.toggle_details_btn.setChecked(True)

    def show_copy_confirmation(self) -> None:
        original = self.copy_report_btn.text()
        self.copy_report_btn.setText(self.i18n.t("startup.report_copied"))
        QTimer.singleShot(
            1500,
            lambda: self.copy_report_btn.setText(original),
        )

    def last_status(self) -> dict | None:
        return self._last_status

    def show_bootstrapping_hint(self) -> None:
        self._show_bootstrapping_text("startup.bootstrapping")

    def show_bootstrapping_hint_long(self) -> None:
        self._show_bootstrapping_text("startup.bootstrapping_long")

    # ---- i18n ----

    def retranslate(self) -> None:
        self.setWindowTitle(self.i18n.t("startup.window_title"))
        self.title_label.setText(self.i18n.t("startup.title_starting"))
        self.message_label.setText(self.i18n.t("startup.connecting"))
        self.toggle_details_btn.setText(self.i18n.t("startup.show_details"))
        self.copy_report_btn.setText(self.i18n.t("startup.copy_report"))
        self.cancel_btn.setText(self.i18n.t("startup.cancel"))

    def _step_key(self, step: str, status: dict | None = None) -> str:
        provider_kind = str((status or {}).get("provider_kind") or "")
        if step == "ort_iobinding_runs" and provider_kind == "trt":
            return "startup.step.ort_iobinding_runs_trt"
        return f"startup.step.{step}"

    def _step_has_translation(self, step: str, status: dict | None = None) -> bool:
        if not step:
            return False
        key = self._step_key(step, status)
        return self.i18n.t(key) != key

    def _step_text(self, step: str, status: dict | None = None) -> str:
        if not step:
            return ""
        key = self._step_key(step, status)
        translated = self.i18n.t(key)
        text = translated if translated != key else step.replace("_", " ")
        if step == "ort_iobinding_runs" and status is not None:
            try:
                done = int(status.get("run_done") or 0)
                total = int(status.get("run_total") or 0)
            except (TypeError, ValueError):
                done = 0
                total = 0
            if done > 0 and total > 0:
                text = f"{text} ({min(done, total)}/{total})"
        return text

    def _step_index_for(self, step: str) -> int:
        try:
            return _STEP_KEYS.index(step) + 1
        except ValueError:
            return 0

    # ---- Internal handlers ----

    def _format_details(self, status: dict, reason: str) -> str:
        rows = []
        for key in (
            "phase",
            "step",
            "step_index",
            "step_total",
            "progress",
            "eta_sec",
            "elapsed_sec",
            "cold",
            "is_known_slow",
            "provider_kind",
            "run_done",
            "run_total",
            "gpu_name",
            "compute_capability",
            "driver_version",
            "onnxruntime_version",
            "reason",
            "message",
            "detail",
        ):
            rows.append(f"{key:24s}: {status.get(key, '')}")
        return "\n".join(rows)

    def _apply_hint(
        self,
        status: dict,
        phase: str,
        cold: bool,
        known_slow: bool,
        provider_kind: str,
        elapsed: float,
    ) -> None:
        if phase == "failed":
            self._set_hint(self.i18n.t("startup.hint_failed"), "warn")
            return
        if phase != "warming":
            self.hint_label.setVisible(False)
            return

        tier: tuple[float, str, str] | None = None
        for candidate in _REASSURANCE_TIERS:
            if elapsed >= candidate[0]:
                tier = candidate
        if provider_kind == "trt" and (tier is None or tier[0] < 90.0):
            self.hint_label.setVisible(False)
        elif tier is not None and tier[0] >= 90.0:
            self._set_hint(self._format_hint(tier[1], status, elapsed), tier[2])
        elif known_slow and cold:
            self._set_hint(self.i18n.t("startup.hint_known_slow"), "info")
        elif tier is not None:
            self._set_hint(self._format_hint(tier[1], status, elapsed), tier[2])
        else:
            self.hint_label.setVisible(False)

    def _format_hint(self, key: str, status: dict, elapsed: float) -> str:
        template = self.i18n.t(key)
        values = {
            "gpu": str(status.get("gpu_name") or "GPU"),
            "cc": str(status.get("compute_capability") or "?"),
            "ort": str(status.get("onnxruntime_version") or "?"),
            "elapsed": int(elapsed),
        }
        try:
            return template.format(**values)
        except Exception:
            return template

    def _set_hint(self, text: str, severity: str) -> None:
        self.hint_label.setStyleSheet(_HINT_STYLE_WARN if severity == "warn" else _HINT_STYLE_INFO)
        self.hint_label.setText(text)
        self.hint_label.setVisible(bool(text))

    def _show_bootstrapping_text(self, key: str) -> None:
        text = self.i18n.t(key)
        self._base_message = text
        self.message_label.setText(text)
        self.progress.setRange(0, 0)
        if not self._ellipsis_timer.isActive():
            self._ellipsis_timer.start()

    def _tick_ellipsis(self) -> None:
        self._ellipsis_phase = (self._ellipsis_phase + 1) % 4
        suffix = "." * self._ellipsis_phase
        self.message_label.setText(self._base_message + ("  " + suffix if suffix else ""))

    def _on_toggle_details(self, checked: bool) -> None:
        self.details_text.setVisible(checked)
        self.toggle_details_btn.setText(
            self.i18n.t("startup.hide_details" if checked else "startup.show_details")
        )
        self.showDetailsToggled.emit(checked)
        if checked and self._last_status is not None:
            self.details_text.setPlainText(
                self._format_details(self._last_status, str(self._last_status.get("reason") or ""))
            )

    def _on_copy_report_clicked(self) -> None:
        self.copyReportRequested.emit()

    def _on_cancel_clicked(self) -> None:
        self.cancelRequested.emit()

    # ---- Qt close-event hooks ----

    def closeEvent(self, event) -> None:
        self._ellipsis_timer.stop()
        super().closeEvent(event)

    @staticmethod
    def copy_to_clipboard(text: str) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
