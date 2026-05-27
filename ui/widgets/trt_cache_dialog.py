from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout

from ui.services.hidden_process import HiddenProcess
from ui.log_sanitizer import clean_log_text
from ui.services.process_helpers import base_environment, offline_trt_warmup_command, trt_warmup_command
from utils.tensorrt_runtime_libs import (
    TENSORRT_CU12_LIBS_WHL_URL,
    TENSORRT_CU12_LIBS_WHL_SIZE_BYTES,
    check_tensorrt_runtime_libs,
    download_and_install_tensorrt_libs,
)
from utils.trt_manifest import (
    MATANYONE2_MODEL_KEYS,
    TRT_MODEL_MATANYONE2,
    TRT_MODEL_RVM,
    cache_status,
    clear_cache,
    collect_fingerprint,
    engine_artifact_paths,
    load_manifest_for_model,
    matanyone2_trt_source_model_paths,
    manifest_path,
    model_label,
    normalized_model_key,
    source_model_path,
    stale_reasons,
)


class _TensorRTDownloadSignals(QObject):
    progress = Signal(object, object)
    finished = Signal(bool, str)


class TensorRTConfigDialog(QDialog):
    def __init__(self, i18n, parent=None, model_key: str | None = None, scope: str | None = None) -> None:
        super().__init__(parent)
        self.i18n = i18n
        self.model_key = normalized_model_key(model_key)
        self.scope = "offline" if self.model_key == TRT_MODEL_RVM and str(scope or "").lower() == "offline" else None
        self.process: HiddenProcess | None = None
        self.download_signals = _TensorRTDownloadSignals(self)
        self.downloading = False
        self.build_error_text = ""
        self.stage = 0
        self.setModal(True)
        self.setWindowTitle(self.i18n.t("trt.title"))

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setMinimumWidth(0)
        self.info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: 700;")
        self.fps_hint_label = QLabel(self.i18n.t("trt.fps_hint"))
        self.fps_hint_label.setWordWrap(True)
        self.fps_hint_label.setStyleSheet("color: #1677c7; font-weight: 600;")
        self.progress = QProgressBar()
        self.progress.setRange(0, self._build_stage_count())
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.stage_label = QLabel("")
        self.stage_label.setWordWrap(True)
        self.stage_label.setMinimumWidth(0)
        self.stage_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.engines_label = QLabel("")
        self.engines_label.setWordWrap(True)
        self.engines_label.setMinimumWidth(0)

        self.download_button = QPushButton()
        self.manual_download_button = QPushButton()
        self.delete_button = QPushButton()
        self.close_button = QPushButton()
        self.build_button = QPushButton()
        self.cancel_button = QPushButton()
        self.download_button.clicked.connect(self._start_download)
        self.manual_download_button.clicked.connect(self._manual_download)
        self.delete_button.clicked.connect(self._delete_cache)
        self.close_button.clicked.connect(self.close)
        self.build_button.clicked.connect(self._start_build)
        self.cancel_button.clicked.connect(self._cancel_build)
        self.download_signals.progress.connect(self._download_progress)
        self.download_signals.finished.connect(self._download_finished)

        buttons = QHBoxLayout()
        buttons.addWidget(self.download_button)
        buttons.addWidget(self.manual_download_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.build_button)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self.info)
        layout.addWidget(self.status_label)
        layout.addWidget(self.fps_hint_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.engines_label)
        layout.addLayout(buttons)
        self.setMinimumWidth(560)
        self.resize(560, 340)
        self._refresh()

    def _status(self) -> str:
        try:
            return cache_status(model_key=self.model_key, scope=self.scope)
        except Exception:
            return "failed"

    def _build_stage_count(self) -> int:
        if self.scope == "offline":
            return 6
        if self.model_key == TRT_MODEL_MATANYONE2:
            return len(MATANYONE2_MODEL_KEYS)
        return 3

    def _set_build_progress(self, completed_stages: int) -> None:
        total = max(1, self._build_stage_count())
        completed = max(0, min(total, int(completed_stages)))
        percent = 99 if completed >= total else int(round(completed * 99 / total))
        self.progress.setRange(0, 100)
        self.progress.setValue(percent)
        self.progress.setFormat(f"{percent}%")

    def _model_display_name(self) -> str:
        if self.model_key == TRT_MODEL_MATANYONE2:
            return model_label(self.model_key)
        path = source_model_path(self.model_key)
        return path.name if path.name else str(path)

    def _description_text(self) -> str:
        if self.model_key == TRT_MODEL_MATANYONE2:
            return self.i18n.t("trt.description_matanyone2")
        return self.i18n.t("trt.description")

    def _manifest_path(self):
        if self.scope is None:
            return manifest_path(self.model_key)
        return manifest_path(self.model_key, scope=self.scope)

    @staticmethod
    def _compact_path(path: Path, max_chars: int = 76) -> str:
        text = str(path)
        if len(text) <= max_chars:
            return text
        parts = path.parts
        if len(parts) >= 3:
            suffix = str(Path(*parts[-3:]))
            drive = parts[0]
            compact = f"{drive}\\...\\{suffix}" if drive.endswith("\\") or drive.endswith(":\\") else f"...\\{suffix}"
            if len(compact) <= max_chars:
                return compact
        return "..." + text[-max(0, max_chars - 3):]

    def _refresh(self, reset_progress: bool = True) -> None:
        libs = check_tensorrt_runtime_libs()
        status = self._status()
        manifest = load_manifest_for_model(self.model_key, scope=self.scope) or {}
        saved_fp = manifest.get("fingerprint") if isinstance(manifest.get("fingerprint"), dict) else {}
        try:
            actual_fp = collect_fingerprint(self.model_key)
        except Exception:
            actual_fp = {}
        fp = saved_fp if status == "ready" and saved_fp else actual_fp or saved_fp
        models = manifest.get("models") if isinstance(manifest.get("models"), list) else []
        total_seconds = 0
        engine_size = 0.0
        for model in models:
            if not isinstance(model, dict):
                continue
            total_seconds = max(total_seconds, int(model.get("total_build_seconds") or 0))
            engines = model.get("engines")
            if isinstance(engines, list):
                for engine in engines:
                    if isinstance(engine, dict):
                        try:
                            engine_size += float(engine.get("size_mb") or 0)
                        except (TypeError, ValueError):
                            pass
        details = [
            f"{self.i18n.t('trt.model')}: {self._model_display_name()}",
            f"{self.i18n.t('trt.precision')}: FP32",
            f"{self.i18n.t('trt.gpu')}: {fp.get('gpu_name') or libs.gpu_name or '-'}",
            f"{self.i18n.t('trt.driver')}: {fp.get('driver_version') or '-'}",
            f"{self.i18n.t('trt.tensorrt')}: {fp.get('trt_version') or '-'}",
            f"{self.i18n.t('trt.cache_path')}: {self._compact_path(self._manifest_path().parent)}",
        ]
        if libs.frozen:
            details.append(f"{self.i18n.t('trt.runtime_lib_path')}: {self._compact_path(Path(libs.lib_dir))}")
            if libs.compute_capability:
                details.append(f"{self.i18n.t('trt.sm_library')}: {libs.sm_dll or '-'}")
            if not libs.ready:
                details.append(f"{self.i18n.t('trt.missing_libraries')}: {', '.join(libs.missing)}")
        if total_seconds:
            details.append(f"{self.i18n.t('trt.last_build')}: {total_seconds // 60}m {total_seconds % 60}s")
        if engine_size:
            details.append(f"{self.i18n.t('trt.engine_size')}: {engine_size:.1f} MB")
        if status == "stale":
            reasons = stale_reasons(saved_fp, actual_fp) if saved_fp and actual_fp else []
            if reasons:
                details.append(f"{self.i18n.t('trt.stale_reason')}: {reasons[0]}")
        self.info.setText("\n".join(details))
        if libs.frozen and not libs.ready:
            self.status_label.setText(self.i18n.t("trt.runtime_missing"))
        else:
            self.status_label.setText(f"{self.i18n.t('trt.cache_status')}: {self.i18n.t('trt.status_' + status)}")
        if reset_progress:
            self.progress.setRange(0, 100)
            self.progress.setFormat("%p%")
            self.progress.setVisible(False)
            self.stage_label.setText(self.i18n.t("trt.download_hint") if libs.frozen and not libs.ready else self._description_text())
        self.engines_label.setText(self.i18n.t("trt.warning"))
        self.download_button.setText(self.i18n.t("trt.auto_download"))
        self.manual_download_button.setText(self.i18n.t("trt.manual_download"))
        self.delete_button.setText(self.i18n.t("trt.delete_cache"))
        self.close_button.setText(self.i18n.t("button.close"))
        self.build_button.setText(self.i18n.t("trt.rebuild") if status in {"ready", "stale"} else self.i18n.t("trt.start_build"))
        self.cancel_button.setText(self.i18n.t("button.cancel"))
        missing_runtime = libs.frozen and not libs.ready
        self.download_button.setVisible(missing_runtime)
        self.manual_download_button.setVisible(missing_runtime)
        self.delete_button.setVisible(not missing_runtime and status in {"ready", "stale", "failed"})
        self.cancel_button.setVisible(False)
        self.build_button.setVisible(not missing_runtime)
        self.close_button.setVisible(True)

    def _start_build(self) -> None:
        if self.process is not None:
            return
        libs = check_tensorrt_runtime_libs()
        if libs.frozen and not libs.ready:
            self._refresh()
            return
        if self.model_key == TRT_MODEL_MATANYONE2:
            missing = [path for path in matanyone2_trt_source_model_paths().values() if not path.is_file()]
            missing_text = ", ".join(str(path) for path in missing)
        else:
            model_path = source_model_path(self.model_key)
            missing_text = str(model_path) if not model_path.is_file() else ""
        if missing_text:
            self.build_error_text = "ERROR: " + self.i18n.t("trt.source_model_missing").format(path=missing_text)
            self.progress.setVisible(False)
            self.stage_label.setText(self.build_error_text)
            self.engines_label.setText("")
            return
        self.build_error_text = ""
        self.stage = 0
        self.progress.setRange(0, 100)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self.stage_label.setText(
            self.i18n.t("trt.building_model").format(
                model=self._model_display_name(),
                precision="FP32",
            )
        )
        self.engines_label.setText("")
        self.build_button.setVisible(False)
        self.close_button.setVisible(False)
        self.delete_button.setVisible(False)
        self.cancel_button.setVisible(True)

        process = HiddenProcess(self)
        self.process = process
        process.stdout.connect(self._read_process_output)
        process.stderr.connect(self._read_process_output)
        process.finished.connect(self._build_finished)
        if self.scope == "offline":
            exe, args = offline_trt_warmup_command()
        else:
            exe, base_args = trt_warmup_command()
            is_matanyone2 = self.model_key == TRT_MODEL_MATANYONE2
            args = [
                *base_args,
                "--model",
                TRT_MODEL_MATANYONE2 if is_matanyone2 else TRT_MODEL_RVM,
                "--input-size",
                "1024",
                "--downsample",
                "0.5",
                "--fp16",
                "0",
                "--cuda-graph",
                "0",
                "--cache-dir",
                str(self._manifest_path().parent),
                "--progress-stdout",
            ]
        process.start(exe, args, env=base_environment())

    def _manual_download(self) -> None:
        QDesktopServices.openUrl(QUrl(TENSORRT_CU12_LIBS_WHL_URL))

    def _start_download(self) -> None:
        if self.downloading:
            return
        self.downloading = True
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self.progress.setVisible(True)
        self.stage_label.setText(self.i18n.t("trt.downloading"))
        self.engines_label.setText("")
        self.download_button.setEnabled(False)
        self.manual_download_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self._download_progress(0, TENSORRT_CU12_LIBS_WHL_SIZE_BYTES)

        def worker() -> None:
            try:
                download_and_install_tensorrt_libs(
                    progress=lambda received, total: self.download_signals.progress.emit(received, total)
                )
            except Exception as exc:
                self.download_signals.finished.emit(False, str(exc))
            else:
                self.download_signals.finished.emit(True, "")

        threading.Thread(target=worker, name="trt-runtime-download", daemon=True).start()

    def _download_progress(self, received: int, total: int) -> None:
        received = max(0, int(received or 0))
        total = int(total or 0)
        if total > 0:
            percent = max(0, min(100, int(received * 100 / total)))
            self.progress.setValue(percent)
            self.progress.setFormat(f"{percent}%")
            self.stage_label.setText(
                self.i18n.t("trt.downloading_progress").format(done=received / (1024 * 1024), total=total / (1024 * 1024))
            )
        else:
            self.progress.setFormat("")
            self.stage_label.setText(self.i18n.t("trt.downloading_progress_unknown").format(done=received / (1024 * 1024)))

    def _download_finished(self, ok: bool, message: str) -> None:
        self.downloading = False
        self.download_button.setEnabled(True)
        self.manual_download_button.setEnabled(True)
        self.close_button.setEnabled(True)
        if ok:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("100%")
            self.stage_label.setText(self.i18n.t("trt.download_done"))
        else:
            self.stage_label.setText(self.i18n.t("trt.download_failed").format(error=message))
        self._refresh(reset_progress=False)

    def _append_build_log(self, text: str) -> None:
        try:
            text = clean_log_text(text)
            if not text:
                return
            path = self._manifest_path().parent / "build.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception:
            pass

    def _read_process_output(self, text: str) -> None:
        text = clean_log_text(text)
        if not text:
            return
        self._append_build_log(text)
        for line in text.splitlines():
            if line.startswith("STAGE:"):
                parts = line.split(":", 3)
                if len(parts) >= 3:
                    stage_number = 0
                    try:
                        stage_number = int(parts[1])
                        self.stage = max(self.stage, stage_number)
                    except ValueError:
                        pass
                    completed = stage_number if parts[2] == "done" else stage_number - 1
                    self._set_build_progress(completed)
                    if len(parts) == 4 and parts[2] == "start":
                        self.stage_label.setText(parts[3])
            elif line.startswith("ERROR:"):
                self.build_error_text = line
                self.stage_label.setText(line)
        cache_dir = self._manifest_path().parent
        count = len(engine_artifact_paths(cache_dir, recursive=self.model_key == TRT_MODEL_MATANYONE2)) if cache_dir.exists() else 0
        self.engines_label.setText(self.i18n.t("trt.engines_built").format(count=count))

    def _build_finished(self, exit_code: int) -> None:
        self.process = None
        if exit_code == 0:
            self.build_error_text = ""
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("100%")
            self._refresh()
            return
        if not self.build_error_text:
            self.build_error_text = self.i18n.t("trt.build_failed").format(error=f"exit code {exit_code}")
        self.progress.setVisible(True)
        self._refresh(reset_progress=False)
        self.stage_label.setText(self.build_error_text)

    def _cancel_build(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process = None
        self.build_error_text = ""
        self._refresh()

    def _delete_cache(self) -> None:
        clear_cache(self.model_key, scope=self.scope)
        self._refresh()

    def closeEvent(self, event) -> None:
        if self.process is not None:
            self._cancel_build()
        super().closeEvent(event)
