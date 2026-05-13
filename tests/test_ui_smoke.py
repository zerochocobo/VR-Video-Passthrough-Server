from __future__ import annotations

import os
import site
import unittest
from pathlib import Path

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
    def test_main_window_constructs(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        from ui.log_limits import UI_LOG_MAX_BLOCKS

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            self.assertTrue(window.windowTitle())
            self.assertFalse(window.windowIcon().isNull())
            self.assertNotIn("(v0.1)", window.home.title.text())
            self.assertIn(f"({window.metadata.display_version})", window.windowTitle())
            self.assertEqual(window.home.title.font().pointSize(), 19)
            self.assertGreaterEqual(window.home.title.font().weight(), 800)
            self.assertLessEqual(window.home.subtitle.font().pointSize(), 9)
            self.assertFalse(window.home.server_button.icon().isNull())
            self.assertEqual(window.home.server_button.iconSize().width(), 22)
            self.assertEqual(window.version_label.text(), window.metadata.display_version)
            self.assertEqual(window.home.language.count(), 3)
            self.assertNotIn("Follow system", [window.home.language.itemText(i) for i in range(3)])
            self.assertIs(window.home.language.parentWidget(), window.statusBar())
            self.assertEqual(window.status_left_spacer.width(), 20)
            self.assertIn("font-size: 9pt", window.home.language.styleSheet())
            self.assertIn("font-size: 9pt", window.version_label.styleSheet())
            self.assertIn("font-size: 8pt", window.home.log.styleSheet())
            self.assertEqual(window.home.log.document().maximumBlockCount(), UI_LOG_MAX_BLOCKS)
            self.assertIn("github.com/zerochocobo/VR-Video-Passthrough-Server", window.home.project_link.text())
            self.assertIn(window.i18n.t("project.url_label"), window.home.project_link.text())
            self.assertTrue(window.home.project_link.openExternalLinks() is False)
            self.assertEqual(window.home.project_link.height(), 28)
            self.assertEqual(window.home.subtitle_enable.text(), "")
            self.assertTrue(window.home.subtitle_enable_label.text())
            self.assertEqual(window.home.log_toggle.text(), "")
            self.assertTrue(window.home.log_toggle_label.text())
            self.assertEqual(window.home.debug_toggle.text(), "")
            self.assertTrue(window.home.debug_toggle_label.text())
            self.assertTrue(window.home.debug_toggle.isHidden())
            self.assertTrue(window.home.debug_toggle_label.isHidden())
            self.assertEqual(window.home.green_mode.text(), "")
            self.assertEqual(window.home.alpha_mode.text(), "")
            self.assertTrue(window.home.green_mode_label.text())
            self.assertTrue(window.home.alpha_mode_label.text())
            self.assertEqual(window.home.bg_color.itemData(2), "00FF00")
            self.assertEqual(window.home.bg_color.itemData(3), "0000FF")
            self.assertEqual(window.home.bg_color.itemText(2), window.i18n.t("bg.soft_green"))
            self.assertEqual(window.home.bg_color.itemText(3), window.i18n.t("bg.soft_blue"))
            self.assertFalse(hasattr(window.home, "bg_color_note"))
            self.assertEqual(window.home.green_mode_label.width(), window.home.alpha_mode_label.width())
            quick_label_widths = {
                window.home.video_dirs_title.width(),
                window.home.green_mode_label.width(),
                window.home.alpha_mode_label.width(),
                window.home.subtitle_enable_label.width(),
                window.home.log_toggle_label.width(),
                window.home.performance_mask_skip_label.width(),
                window.home.performance_fps_label.width(),
                window.home.performance_output_size_label.width(),
            }
            self.assertEqual(len(quick_label_widths), 1)
            self.assertTrue(window.home.config_header.isChecked())
            self.assertFalse(window.home.performance_header.isChecked())
            self.assertTrue(window.home.performance_content.isHidden())
            self.assertEqual(window.home.performance_mask_skip.itemData(0), 1)
            self.assertEqual(window.home.performance_mask_skip.itemData(2), 3)
            self.assertEqual(window.home.performance_fps.itemData(1), 30)
            self.assertEqual(window.home.performance_output_size.itemData(0), 0)
            self.assertEqual(window.home.performance_output_size.itemData(1), 4096)
            window.home.performance_output_size.setCurrentIndex(0)
            app.processEvents()
            self.assertEqual(window.settings.data["decode_max_side"], 0)
            window.home.performance_header.setChecked(True)
            app.processEvents()
            self.assertFalse(window.home.config_header.isChecked())
            self.assertTrue(window.home.config_content.isHidden())
            self.assertFalse(window.home.performance_content.isHidden())
            self.assertEqual(window.stack.count(), 3)
            base_size = window.size()
            self.assertEqual(base_size.height(), 508)
            window.home.log_toggle.setChecked(True)
            app.processEvents()
            self.assertEqual(window.height(), base_size.height())
            self.assertGreater(window.width(), base_size.width())
            self.assertEqual(window.home.width(), window.width())
            self.assertEqual(window.home.log.x(), 560)
            self.assertFalse(window.home.debug_toggle.isHidden())
            self.assertFalse(window.home.debug_toggle_label.isHidden())
            window.home.debug_toggle.setChecked(True)
            window.home.log_toggle.setChecked(False)
            app.processEvents()
            self.assertTrue(window.home.debug_toggle.isHidden())
            self.assertFalse(window.home.debug_toggle.isChecked())
            self.assertEqual(window.height(), base_size.height())
            self.assertEqual(window.width(), base_size.width())
            window.stack.setCurrentWidget(window.subtitle)
            app.processEvents()
            self.assertGreaterEqual(window.width(), 1100)
            self.assertEqual(window.height(), 600)
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
            window.stack.setCurrentWidget(window.offline)
            app.processEvents()
            self.assertEqual(window.width(), 600)
            self.assertEqual(window.height(), 600)
            self.assertTrue(window.offline.title_label.text())
            self.assertGreaterEqual(window.offline.back_button.width(), window.offline.back_button.sizeHint().width())
            self.assertNotIn("转换", window.offline.tabs.tabText(0))
            self.assertNotIn("转换", window.offline.tabs.tabText(1))
            self.assertEqual(window.offline.single_labels["time"].text(), window.i18n.t("offline.time_range"))
            self.assertEqual(window.offline.single_labels["output"].text(), window.i18n.t("offline.output"))
            self.assertEqual(window.offline.single_labels["performance"].text(), window.i18n.t("offline.performance"))
            self.assertEqual(window.offline.single_skip_frames.count(), 3)
            self.assertEqual(window.offline.single_skip_frames.currentData(), 0)
            self.assertEqual(window.offline.batch_skip_frames.count(), 3)
            self.assertTrue(window.offline.single_matanyone_help.isHidden())
            self.assertTrue(window.offline.batch_matanyone_help.isHidden())
            window.offline.single_engine.setCurrentIndex(2)
            app.processEvents()
            self.assertFalse(window.offline.single_matanyone_help.isHidden())
            self.assertEqual(window.offline.log.document().maximumBlockCount(), UI_LOG_MAX_BLOCKS)
            self.assertTrue(hasattr(window.offline, "single_out_dir"))
            self.assertFalse(window.offline.start_single.icon().isNull())
            self.assertFalse(window.offline.stop_single.icon().isNull())
            self.assertTrue(window.offline.start_single.isEnabled())
            self.assertFalse(window.offline.stop_single.isEnabled())
            self.assertTrue(window.offline.batch_recursive.isChecked())
            self.assertTrue(window.offline.batch_recursive.text())
            window.stack.setCurrentWidget(window.home)
            app.processEvents()
            self.assertEqual(window.width(), base_size.width())
            self.assertEqual(window.height(), base_size.height())
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
