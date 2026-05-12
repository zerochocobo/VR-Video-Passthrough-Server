from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QWidget

from ui.i18n import I18n, system_language
from ui.metadata import load_app_metadata
from ui.pages.home_page import HOME_COMPACT_WIDTH, HOME_HEIGHT, HomePage
from ui.pages.offline_page import OfflinePage
from ui.pages.subtitle_page import SUBTITLE_PAGE_HEIGHT, SUBTITLE_PAGE_WIDTH, SubtitlePage
from ui.resources import app_icon
from ui.services.offline_process import OfflineProcess
from ui.services.server_process import ServerProcess
from ui.settings import Settings
from ui.styles import font_for_language
from ui.widgets.current_page_stack import CurrentPageStackedWidget


SUPPORTED_LANGUAGES = ("zh_CN", "en_US", "ja_JP")
QT_MAX_WIDGET_SIZE = 16777215
OFFLINE_PAGE_WIDTH = 600
OFFLINE_PAGE_HEIGHT = 600


class MainWindow(QMainWindow):
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
        self.offline = OfflinePage(self.i18n, self.offline_process)
        self.subtitle = SubtitlePage(self.i18n, self.settings)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.offline)
        self.stack.addWidget(self.subtitle)
        self.setCentralWidget(self.stack)
        self.version_label = QLabel(self.metadata.display_version)
        self.version_label.setObjectName("VersionStatus")
        self.version_label.setStyleSheet("QLabel#VersionStatus { font-size: 9pt; }")
        self.home.language.setFixedHeight(22)
        self.home.language.setStyleSheet("QComboBox { font-size: 9pt; padding: 0 6px; }")
        self.status_left_spacer = QWidget()
        self.status_left_spacer.setFixedWidth(20)
        self.statusBar().addWidget(self.status_left_spacer)
        self.statusBar().addWidget(self.home.language)
        self.statusBar().addPermanentWidget(self.version_label)
        self.home.server_button.clicked.connect(self.toggle_server)
        self.home.offline_button.clicked.connect(self.open_offline)
        self.home.subtitle_style_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.subtitle))
        self.home.language.currentIndexChanged.connect(self.change_language)
        self.offline.back_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.subtitle.back_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.stack.currentChanged.connect(self._page_changed)
        self.server.output.connect(self.home.append_log)
        self.server.state_changed.connect(self.home.set_server_running)
        self.offline_process.state_changed.connect(self._offline_state_changed)
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
        if self.server.is_running():
            self.server.stop()
            return
        if self.offline_process.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_offline_first"))
            return
        self.settings.save()
        env = self.settings.server_env()
        env["PT_DEBUG_LOGS"] = "1" if self.home.debug_toggle.isChecked() else "0"
        self.home.clear_log()
        self.server.start(env)

    def open_offline(self) -> None:
        if self.server.is_running():
            QMessageBox.warning(self, self.i18n.t("dialog.warning"), self.i18n.t("dialog.stop_server_first"))
            return
        self.stack.setCurrentWidget(self.offline)

    def _page_changed(self, index: int) -> None:
        if self.stack.widget(index) is self.home:
            self.home._adjust_window()
            return
        self.setMinimumWidth(0)
        self.setMaximumWidth(QT_MAX_WIDGET_SIZE)
        self.setMinimumHeight(0)
        self.setMaximumHeight(QT_MAX_WIDGET_SIZE)
        if self.stack.widget(index) is self.subtitle:
            self.resize(SUBTITLE_PAGE_WIDTH, SUBTITLE_PAGE_HEIGHT)
        elif self.stack.widget(index) is self.offline:
            self.resize(OFFLINE_PAGE_WIDTH, OFFLINE_PAGE_HEIGHT)

    def _offline_state_changed(self, running: bool) -> None:
        if running:
            self.stack.setCurrentWidget(self.offline)

    def closeEvent(self, event) -> None:
        self.server.stop()
        self.offline_process.stop()
        super().closeEvent(event)
