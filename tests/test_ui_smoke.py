from __future__ import annotations

import contextlib
import os
import shutil
import site
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_DLL_HANDLES = []
if hasattr(os, "add_dll_directory"):
    for site_dir in site.getsitepackages():
        base = Path(site_dir)
        for dll_dir in (base / "PySide6", base / "shiboken6"):
            if dll_dir.exists():
                _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
        plugins = base / "PySide6" / "plugins"
        platforms = plugins / "platforms"
        if platforms.exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms))
        if plugins.exists():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))


class UiSmokeTests(unittest.TestCase):
    def test_superres_dashboard_dialog_includes_low_quality_and_performance_note(self) -> None:
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.dialogs.feature_dialogs import SuperResSettingsDialog
        from ui.i18n import I18n

        app = QApplication.instance() or QApplication([])
        i18n = I18n("zh_CN")
        settings = SimpleNamespace(data={
            "superres_target_height": 4096,
            "superres_quality": 1,
            "superres_hdr_look": "natural",
        })
        dialog = SuperResSettingsDialog(i18n, settings)
        try:
            self.assertEqual(dialog.quality.count(), 4)
            self.assertEqual(dialog.quality.currentData(), 1)
            self.assertEqual(dialog.quality.itemText(0), i18n.t("superres.quality_1"))
            self.assertEqual(dialog.performance_note.text(), i18n.t("superres.performance_note"))
            self.assertEqual(dialog.performance_note.font().pointSizeF(), dialog.font().pointSizeF())
            self.assertEqual(dialog.minimumWidth(), 520)
            self.assertEqual(dialog.maximumWidth(), 520)
            self.assertGreaterEqual(dialog.height(), dialog.sizeHint().height())
        finally:
            dialog.close()
            app.processEvents()

    def test_video_dirs_dialog_has_mount_timeout_note(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs.video_dirs_dialog import VideoDirsDialog
        from ui.i18n import I18n

        app = QApplication.instance() or QApplication([])
        i18n = I18n("zh_CN")
        dialog = VideoDirsDialog(i18n, ["Y:\\"])
        try:
            dialog.show()
            app.processEvents()
            self.assertEqual(dialog.note_label.text(), i18n.t("video_dirs.mount_timeout_note"))
            self.assertTrue(dialog.note_label.wordWrap())
            self.assertIn("font-size: 8.5pt", dialog.note_label.styleSheet())
            self.assertTrue(dialog.note_label.text().startswith("提示："))
            self.assertIn("网盘挂载", dialog.note_label.text())
            self.assertGreater(dialog.note_label.y(), dialog.add_button.y())
            self.assertLess(dialog.note_label.y(), dialog.save_button.y())
        finally:
            dialog.deleteLater()

    def test_video_dirs_dialog_filters_unc_entries(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.dialogs.video_dirs_dialog import VideoDirsDialog
        from ui.i18n import I18n

        app = QApplication.instance() or QApplication([])
        i18n = I18n("zh_CN")
        dialog = VideoDirsDialog(i18n, [r"\\nas\VR", r"Y:\VR", "//nas/Movies"])
        try:
            dialog.show()
            app.processEvents()
            self.assertEqual(dialog.list_widget.count(), 1)
            self.assertEqual(dialog.directories(), [r"Y:\VR"])
        finally:
            dialog.deleteLater()

    def test_main_window_constructs(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui import settings as settings_module
        from ui.main_window import MainWindow
        from ui.log_limits import UI_LOG_MAX_BLOCKS
        from ui.widgets.nav_rail import NAV_WIDTH

        app = QApplication.instance() or QApplication([])
        settings_root = Path(tempfile.mkdtemp(prefix="pt_ui_smoke_"))
        self.addCleanup(lambda: shutil.rmtree(settings_root, ignore_errors=True))
        patch_stack = contextlib.ExitStack()
        patch_stack.enter_context(patch.object(settings_module, "SETTINGS_PATH", settings_root / "ui_settings.json"))
        patch_stack.enter_context(patch.object(settings_module, "SETTINGS_META_PATH", settings_root / "ui_settings_meta.json"))
        patch_stack.enter_context(patch("ui.pages.settings_page.cache_status", return_value="missing"))
        patch_stack.enter_context(patch("ui.pages.offline_page.cache_status", return_value="missing"))
        window = MainWindow()
        try:
            self.assertTrue(window.windowTitle())
            self.assertFalse(window.windowIcon().isNull())
            self.assertIn(f"({window.metadata.display_version})", window.windowTitle())
            self.assertEqual(window.version_label.text(), window.metadata.display_version)
            self.assertIn("font-size: 9pt", window.version_label.styleSheet())

            # Nav rail: five entries, home selected at start.
            self.assertEqual(window.nav.width(), NAV_WIDTH)
            self.assertEqual(set(window.nav._items), {"home", "tools", "subtitle", "log", "settings"})
            self.assertEqual(window.nav.current(), "home")
            for key in ("home", "tools", "subtitle", "log", "settings"):
                self.assertTrue(window.nav._items[key]._text_label.text())
            self.assertEqual(window.stack.count(), 9)
            self.assertIs(window.stack.currentWidget(), window.dashboard)

            # Dashboard: server bar and feature cards.
            self.assertFalse(window.dashboard.server_button.icon().isNull())
            self.assertEqual(window.dashboard.server_button.iconSize().width(), 22)
            self.assertIn("https://wapok.com", window.dashboard.project_link.text())
            self.assertIn(window.i18n.t("project.url_label"), window.dashboard.project_link.text())
            self.assertFalse(window.dashboard.project_link.openExternalLinks())
            self.assertEqual(
                set(window.dashboard.cards),
                {"green", "alpha", "alpha2d", "two_dvr", "superres", "rm", "subtitle", "si", "light"},
            )
            for key, card in window.dashboard.cards.items():
                self.assertTrue(card.title_label.text(), key)
            self.assertTrue(window.dashboard.cards["green"].is_checked())
            self.assertTrue(window.dashboard.cards["alpha"].is_checked())
            self.assertTrue(window.dashboard.cards["alpha2d"].is_checked())
            self.assertTrue(window.dashboard.cards["green"].summary_label.text().startswith("[GREEN]"))
            self.assertEqual(window.dashboard.cards["alpha"].summary_label.text(), "[ALPHA]最好的透视效果")
            self.assertFalse(window.dashboard.cards["alpha"].help_button.isHidden())
            self.assertTrue(window.dashboard.cards["superres"].summary_label.text().startswith("[SuperRes]"))
            self.assertTrue(window.dashboard.cards["two_dvr"].summary_label.text().startswith("[2D>3D]"))
            self.assertTrue(window.dashboard.cards["si"].summary_label.text().startswith("[SI]"))
            self.assertFalse(window.dashboard.cards["superres"].config_button.isHidden())
            self.assertEqual(window.dashboard._realtime_keys[:3], ["alpha", "green", "superres"])
            self.assertEqual(window.dashboard._2d_keys, ["two_dvr", "si", "rm"])
            self.assertEqual(window.dashboard._audio_keys, ["subtitle", "light", "alpha2d"])
            self.assertFalse(hasattr(window.dashboard, "two_d_group_label"))
            self.assertFalse(window.dashboard.cards["light"].is_checked())
            self.assertFalse(window.settings.data["rm_enabled"])
            # RM card hidden until the settings debug gate enables it.
            self.assertFalse(window.dashboard.cards["rm"].isVisible())
            window.dashboard.set_server_running(True)
            self.assertFalse(window.dashboard.switch_lock_notice.isHidden())
            for key, card in window.dashboard.cards.items():
                self.assertEqual(card.switch.isEnabled(), key == "light", key)
                self.assertEqual(card.lock_label.isHidden(), key == "light", key)
            window.dashboard.set_server_running(False)
            self.assertTrue(window.dashboard.switch_lock_notice.isHidden())
            for card in window.dashboard.cards.values():
                self.assertTrue(card.switch.isEnabled())
                self.assertTrue(card.lock_label.isHidden())

            window.show()
            app.processEvents()
            base_size = window.size()
            self.assertEqual(base_size.width(), NAV_WIDTH + 700)

            # Settings page: performance combos + TRT (cache missing => disabled).
            window._show_page("settings")
            app.processEvents()
            self.assertIs(window.stack.currentWidget(), window.settings_page)
            self.assertEqual(window.nav.current(), "settings")
            page = window.settings_page
            self.assertEqual(page.language.count(), 3)
            self.assertNotIn("Follow system", [page.language.itemText(i) for i in range(3)])
            self.assertEqual(page.performance_quality.itemData(0), "ultrafast")
            self.assertEqual(page.performance_fps.itemData(2), 30)
            self.assertGreaterEqual(page.performance_fps.findData(50), 0)
            self.assertEqual(page.performance_output_size.itemData(0), 0)
            self.assertEqual(page.performance_output_size.itemData(1), 4096)
            self.assertTrue(page.trt_enabled_label.text())
            self.assertTrue(page.trt_configure_button.text())
            self.assertFalse(page.trt_enabled.isEnabled())
            page.performance_output_size.setCurrentIndex(0)
            app.processEvents()
            self.assertEqual(window.settings.data["decode_max_side"], 0)

            # Release UI hides the feature-debug section while retaining its
            # internal switch wiring for saved settings and diagnostics.
            self.assertTrue(page.debug_group.isHidden())
            self.assertTrue(page.rm_card_label.text())
            page.rm_card_switch.setChecked(True)
            app.processEvents()
            self.assertTrue(window.settings.data["rm_card_visible"])
            window._show_page("home")
            app.processEvents()
            self.assertTrue(window.dashboard.cards["rm"].isVisible())
            window._show_page("tools")
            app.processEvents()
            self.assertTrue(window.tools.rm_card.isVisible())
            page.rm_card_switch.setChecked(False)
            app.processEvents()
            self.assertFalse(window.dashboard.cards["rm"].isVisible())
            self.assertFalse(window.tools.rm_card.isVisible())

            # Log page basics.
            window._show_page("log")
            app.processEvents()
            self.assertIs(window.stack.currentWidget(), window.log_page)
            self.assertEqual(window.log_page.log.document().maximumBlockCount(), UI_LOG_MAX_BLOCKS)
            self.assertFalse(window.log_page.debug_toggle.isChecked())

            window._show_page("subtitle")
            app.processEvents()
            self.assertGreaterEqual(window.width(), NAV_WIDTH + 1100)
            self.assertTrue(hasattr(window.subtitle, "original_canvas"))
            self.assertTrue(hasattr(window.subtitle, "preview_canvas"))
            self.assertTrue(window.subtitle.title_label.text())
            self.assertFalse(hasattr(window.subtitle, "subtitle_path"))
            self.assertLess(window.subtitle.load_frame_button.geometry().x(), window.subtitle.preview_button.geometry().x())
            self.assertLess(window.subtitle.preview_button.geometry().x(), window.subtitle.save_button.geometry().x())
            self.assertFalse(window.subtitle.save_button.icon().isNull())
            self.assertEqual(window.subtitle.save_status_label.text(), "")
            self.assertEqual(window.subtitle.log.document().maximumBlockCount(), UI_LOG_MAX_BLOCKS)
            direction_index = window.subtitle.direction.findData("vertical_left")
            self.assertGreaterEqual(direction_index, 0)
            window.subtitle.direction.setCurrentIndex(direction_index)
            window.subtitle.retranslate()
            self.assertEqual(window.subtitle.direction.currentData(), "vertical_left")
            window.subtitle.save_settings()
            self.assertEqual(window.settings.data["subtitle_direction"], "vertical_left")
            self.assertIn(window.i18n.t("subtitle.save_done"), window.subtitle.save_status_label.text())
            self.assertLess(window.subtitle.save_button.geometry().x(), window.subtitle.restore_button.geometry().x())
            window._show_sub_page(window.offline)
            app.processEvents()
            self.assertEqual(window.width(), NAV_WIDTH + 600)
            self.assertEqual(window.nav.current(), "tools")
            self.assertTrue(window.offline.title_label.text())
            self.assertGreaterEqual(window.offline.back_button.width(), window.offline.back_button.sizeHint().width())
            self.assertNotIn("转换", window.offline.tabs.tabText(0))
            self.assertNotIn("转换", window.offline.tabs.tabText(1))
            self.assertEqual(window.offline.single_time_mode.itemText(0), window.i18n.t("offline.time_mode_range"))
            self.assertEqual(window.offline.single_time_mode.itemText(1), window.i18n.t("offline.time_mode_segments"))
            self.assertTrue(window.offline.single_segments_config_button.isHidden())
            self.assertEqual(window.offline.single_labels["output"].text(), window.i18n.t("offline.output"))
            self.assertEqual(window.offline.single_labels["performance"].text(), window.i18n.t("performance.quality_speed"))
            self.assertEqual(window.offline.single_labels["trt"].text(), window.i18n.t("trt.row_label"))
            self.assertEqual(window.offline.single_quality_speed.count(), 3)
            self.assertIn(window.offline.single_quality_speed.currentData(), {"ultrafast", "medium", "veryslow"})
            self.assertEqual(window.offline.single_duration.findData("custom_end"), 4)
            self.assertTrue(hasattr(window.offline, "single_custom_end"))
            window.offline.single_time_mode.setCurrentIndex(window.offline.single_time_mode.findData("segments"))
            app.processEvents()
            self.assertFalse(window.offline.single_segments_config_button.isHidden())
            self.assertTrue(window.offline.single_start.isHidden())
            self.assertEqual(window.offline.batch_quality_speed.count(), 3)
            self.assertEqual(window.offline.single_engine.count(), 2)
            self.assertEqual(window.offline.single_engine.itemData(0), "rvm_fast")
            self.assertEqual(window.offline.single_engine.itemData(1), "matanyone2")
            self.assertEqual(window.offline.single_recognition.count(), 3)
            self.assertEqual(window.offline.single_recognition.itemData(0), "yolo26m_efficientsam")
            self.assertEqual(window.offline.single_recognition.itemData(1), "yolo26m_birefnet")
            self.assertEqual(window.offline.single_recognition.itemData(2), "sam3")
            self.assertTrue(window.offline.single_trt_configure_button.text())
            self.assertEqual(window.offline.single_trt_enabled.text(), "")
            self.assertFalse(window.offline.single_trt_enabled.isEnabled())
            self.assertTrue(window.offline.single_matanyone_help.isHidden())
            self.assertTrue(window.offline.batch_matanyone_help.isHidden())
            window.offline.single_engine.setCurrentIndex(1)
            app.processEvents()
            self.assertFalse(window.offline.single_matanyone_help.isHidden())
            self.assertEqual(window.offline.single_precision.count(), 2)
            self.assertEqual(window.offline.single_precision.itemData(0), ("matanyone2", 512))
            self.assertEqual(window.offline.single_precision.currentData(), ("matanyone2", 1024))
            self.assertTrue(window.offline.single_precision.isEnabled())
            self.assertEqual(window.offline.log.document().maximumBlockCount(), UI_LOG_MAX_BLOCKS)
            self.assertTrue(hasattr(window.offline, "single_out_dir"))
            self.assertFalse(window.offline.start_single.icon().isNull())
            self.assertFalse(window.offline.stop_single.icon().isNull())
            self.assertTrue(window.offline.start_single.isEnabled())
            self.assertFalse(window.offline.stop_single.isEnabled())
            self.assertTrue(window.offline.batch_recursive.isChecked())
            self.assertTrue(window.offline.batch_recursive.text())
            window._show_sub_page(window.superres)
            app.processEvents()
            self.assertTrue(window.superres.title_label.text())
            self.assertEqual(window.superres.single_quality.count(), 3)
            self.assertEqual(window.superres.single_quality.currentData(), 4)
            self.assertEqual(window.superres.single_target.currentData(), 4096)
            self.assertEqual(window.superres.single_hdr_look.currentData(), "natural")
            self.assertEqual(window.superres.batch_hdr_look.currentData(), "natural")
            self.assertEqual(window.superres.single_quality_speed.currentData(), "medium")
            self.assertEqual(window.superres.batch_quality_speed.currentData(), "medium")
            common_args = window.superres._common_args(window.superres.single_target, window.superres.single_quality, window.superres.single_hdr_look)
            self.assertIn("p4", common_args)
            self.assertIn("natural", common_args)
            window._show_page("home")
            app.processEvents()
            self.assertEqual(window.width(), base_size.width())
            self.assertEqual(window.nav.current(), "home")
        finally:
            window.close()
            app.processEvents()
            patch_stack.close()


if __name__ == "__main__":
    unittest.main()
