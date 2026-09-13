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
    def test_startup_overlay_offers_gpu_repair_for_warmup_failure_only(self) -> None:
        from PySide6.QtWidgets import QApplication
        from ui.i18n import I18n
        from ui.widgets.startup_overlay import StartupOverlay

        app = QApplication.instance() or QApplication([])
        overlay = StartupOverlay(I18n("zh_CN"))
        try:
            overlay.apply_status({"phase": "failed", "step": "ort_iobinding_runs"})
            self.assertFalse(overlay.repair_gpu_cache_btn.isHidden())
            overlay.apply_status({"phase": "failed", "step": "firewall"})
            self.assertTrue(overlay.repair_gpu_cache_btn.isHidden())
            overlay.apply_status({
                "phase": "starting",
                "step": "media_index",
                "step_index": 2,
                "step_total": 12,
                "progress": 0.2,
                "eta_sec": 30.0,
                "elapsed_sec": 5.0,
                "plan_active": True,
            })
            self.assertEqual(overlay.progress.maximum(), 1000)
            self.assertEqual(overlay.progress.value(), 200)
            self.assertIn("2/12", overlay.step_label.text())
            self.assertIn("30", overlay.eta_label.text())
            overlay.apply_status({
                "phase": "warming",
                "step": "ort_iobinding_runs",
                "progress": 0.4,
                "eta_sec": 60.0,
                "elapsed_sec": 20.0,
                "plan_active": False,
            })
            self.assertIn("40", overlay.eta_label.text())
            overlay.apply_status({
                "phase": "warming",
                "step": "ort_iobinding_runs",
                "progress": 0.4,
                "eta_sec": 60.0,
                "elapsed_sec": 100.0,
                "step_elapsed_sec": 5.0,
                "plan_active": True,
                "provider_kind": "cuda",
            })
            self.assertTrue(overlay.hint_label.isHidden())
            overlay.apply_status({
                "phase": "warming",
                "step": "da3_trt_warmup",
                "progress": 0.4,
                "eta_sec": 150.0,
                "elapsed_sec": 10.0,
                "step_elapsed_sec": 10.0,
                "step_estimate_sec": 120.0,
                "plan_active": True,
                "provider_kind": "trt",
                "trt_building": True,
                "trt_build_model": "da3",
            })
            self.assertFalse(overlay.hint_label.isHidden())
            self.assertIn("2D 转 3D", overlay.hint_label.text())
            self.assertIn("2 分钟", overlay.hint_label.text())
            self.assertIn("之后启动会明显更快", overlay.hint_label.text())
        finally:
            overlay.close()
            app.processEvents()

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

    def test_superres_dialog_offers_playback_mode_only_for_the_native_target(self) -> None:
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.dialogs.feature_dialogs import SuperResSettingsDialog
        from ui.i18n import I18n
        from ui.settings import DEFAULTS

        app = QApplication.instance() or QApplication([])
        i18n = I18n("zh_CN")
        # Realtime now defaults to native 1x, the only seekable target.
        self.assertEqual(DEFAULTS["superres_target_height"], 0)
        data = dict(DEFAULTS)
        dialog = SuperResSettingsDialog(i18n, SimpleNamespace(data=data))
        try:
            self.assertEqual(dialog.target.currentData(), 0)
            self.assertFalse(dialog.playback.isHidden())
            # The rule stays on screen at every target, not only when the
            # chooser disappears.
            self.assertFalse(dialog.playback_note.isHidden())
            self.assertEqual(dialog.playback_note.text(), i18n.t("superres.playback_native_only"))
            self.assertIn(dialog.selected_playback_mode(), {"virtual", "live"})

            dialog.target.setCurrentIndex(dialog.target.findData(3072))
            app.processEvents()
            self.assertTrue(dialog.playback.isHidden())
            self.assertFalse(dialog.playback_note.isHidden())
            # An enlarging target must not rewrite the shared playback setting.
            self.assertEqual(dialog.selected_playback_mode(), "")
        finally:
            dialog.close()
            app.processEvents()

    def test_dlss5_dialog_reads_back_every_control_it_writes(self) -> None:
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.dialogs.feature_dialogs import DLSS5SettingsDialog
        from ui.i18n import I18n
        from ui.settings import DEFAULTS
        from utils.dlss5 import DLSS5_RANGES

        app = QApplication.instance() or QApplication([])
        i18n = I18n("zh_CN")
        data = dict(DEFAULTS)
        data.update({
            "dlss5_style": 2,
            "dlss5_nr_passes": 3,
            "dlss5_intensity": 1.5,
            "dlss5_skin_structure": -1.0,
            "dlss5_auto_mask": True,
        })
        dialog = DLSS5SettingsDialog(i18n, SimpleNamespace(data=data))
        try:
            payload = dialog.payload()
            self.assertEqual(payload["dlss5_style"], 2)
            self.assertEqual(payload["dlss5_nr_passes"], 3)
            self.assertAlmostEqual(payload["dlss5_intensity"], 1.5)
            self.assertAlmostEqual(payload["dlss5_skin_structure"], -1.0)
            self.assertTrue(payload["dlss5_auto_mask"])
            # Every control the dialog saves has a default to fall back on.
            for key in payload:
                self.assertIn(key, DEFAULTS)
            # A slider must not be able to ask for a value the runtime rejects.
            for key, _label, range_key in (
                DLSS5SettingsDialog.MAIN_SLIDERS + DLSS5SettingsDialog.ADVANCED_SLIDERS
            ):
                low, high = DLSS5_RANGES[range_key]
                slider = dialog.sliders[key]
                self.assertEqual(slider.minimum(), int(round(low * 100)))
                self.assertEqual(slider.maximum(), int(round(high * 100)))
            self.assertEqual(dialog.performance_note.text(), i18n.t("dlss5.performance_note"))
            self.assertIsNotNone(dialog.playback)
        finally:
            dialog.close()
            app.processEvents()

    def test_dlss5_dialog_drops_the_playback_chooser_when_it_is_offline(self) -> None:
        """The offline page reuses this dialog, and offline has no playback
        channel to choose - so the chooser is absent and the note is the one
        that says the realtime 4K ceiling does not apply."""
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.dialogs.feature_dialogs import DLSS5SettingsDialog
        from ui.i18n import I18n
        from ui.settings import DEFAULTS

        app = QApplication.instance() or QApplication([])
        i18n = I18n("zh_CN")
        dialog = DLSS5SettingsDialog(i18n, SimpleNamespace(data=dict(DEFAULTS)), offline=True)
        try:
            self.assertIsNone(dialog.playback)
            self.assertEqual(dialog.selected_playback_mode(), "")
            self.assertEqual(dialog.performance_note.text(), i18n.t("dlss5.offline_note"))
            self.assertEqual(dialog.windowTitle(), i18n.t("dlss5.offline_title"))
            # Same settings either way: the payload keeps every NR control.
            self.assertIn("dlss5_intensity", dialog.payload())
        finally:
            dialog.close()
            app.processEvents()

    def test_dlss5_dialog_hides_the_advanced_block_until_it_is_asked_for(self) -> None:
        """Thirteen sliders at once is a dialog nobody adjusts; four is one they
        do. The rest stay one click away rather than gone."""
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.dialogs.feature_dialogs import DLSS5SettingsDialog
        from ui.i18n import I18n
        from ui.settings import DEFAULTS

        app = QApplication.instance() or QApplication([])
        dialog = DLSS5SettingsDialog(I18n("zh_CN"), SimpleNamespace(data=dict(DEFAULTS)))
        try:
            dialog.show()
            app.processEvents()
            self.assertFalse(dialog.advanced_panel.isVisible())
            dialog.advanced_toggle.setChecked(True)
            app.processEvents()
            self.assertTrue(dialog.advanced_panel.isVisible())
            # NR is 1x, so unlike SuperRes the playback chooser always applies.
            self.assertFalse(dialog.playback.isHidden())
            self.assertIn(dialog.selected_playback_mode(), {"virtual", "live"})
        finally:
            dialog.close()
            app.processEvents()

    def test_superres_page_exposes_true_hdr_controls_only_for_truehdr(self) -> None:
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.i18n import I18n
        from ui.pages.superres_page import SuperResPage
        from ui.services.offline_process import SuperResProcess
        from ui.settings import DEFAULTS

        app = QApplication.instance() or QApplication([])
        data = dict(DEFAULTS)
        data["superres_offline_hdr_look"] = "natural"
        settings = SimpleNamespace(data=data, save=lambda: None)
        page = SuperResPage(I18n("zh_CN"), settings, SuperResProcess())
        try:
            self.assertEqual(page.single_hdr_look.count(), 4)
            self.assertGreaterEqual(page.single_hdr_look.findData("truehdr"), 0)
            # The NGX controls stay hidden while an SDR look is selected.
            self.assertTrue(page.single_true_hdr_box.isHidden())
            self.assertTrue(page.single_labels["truehdr"].isHidden())
            args = page._common_args(page.single_target, page.single_quality, page.single_hdr_look, page.single_bitrate)
            self.assertNotIn("--rtx-vsr-truehdr-contrast", args)

            page.single_hdr_look.setCurrentIndex(page.single_hdr_look.findData("truehdr"))
            app.processEvents()
            self.assertEqual(settings.data["superres_offline_hdr_look"], "truehdr")
            self.assertFalse(page.single_true_hdr_box.isHidden())
            self.assertFalse(page.single_labels["truehdr"].isHidden())
            self.assertFalse(page.batch_true_hdr_box.isHidden())
            # Both tabs share one value.
            page.single_true_hdr["max_nits"].setValue(1400)
            app.processEvents()
            self.assertEqual(page.batch_true_hdr["max_nits"].value(), 1400)
            self.assertEqual(settings.data["superres_truehdr_max_nits"], 1400)
            args = page._common_args(page.single_target, page.single_quality, page.single_hdr_look, page.single_bitrate)
            self.assertIn("--rtx-vsr-truehdr-max-nits", args)
            self.assertEqual(args[args.index("--rtx-vsr-truehdr-max-nits") + 1], "1400")
            self.assertIn("--rtx-vsr-truehdr-middle-gray", args)
        finally:
            page.close()
            app.processEvents()

    def test_playback_mode_chooser_appears_on_both_passthrough_dialogs(self) -> None:
        """Green and Alpha both switch between the virtual file and the live stream."""
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.dialogs.feature_dialogs import (
            AlphaPassthroughSettingsDialog,
            GreenScreenSettingsDialog,
        )
        from ui.i18n import I18n
        from ui.settings import DEFAULTS

        app = QApplication.instance() or QApplication([])
        self.assertEqual(DEFAULTS["passthrough_playback_mode"], "virtual")
        for lang in ("zh_CN", "en_US", "ja_JP"):
            i18n = I18n(lang)
            for key in (
                "playback.title", "playback.virtual", "playback.virtual_hint",
                "playback.live", "playback.live_hint", "playback.note",
            ):
                self.assertTrue(i18n.t(key).strip(), f"{lang}:{key}")

        i18n = I18n("zh_CN")
        for cls in (GreenScreenSettingsDialog, AlphaPassthroughSettingsDialog):
            settings = SimpleNamespace(data={"background_color": "00FF00"})
            dialog = cls(i18n, settings)
            try:
                # Nothing stored yet -> the recommended virtual file.
                self.assertEqual(dialog.selected_playback_mode(), "virtual", cls.__name__)
                self.assertTrue(dialog.playback.virtual_radio.isChecked())
                dialog.playback.live_radio.setChecked(True)
                self.assertEqual(dialog.selected_playback_mode(), "live", cls.__name__)
                # Mutually exclusive: picking one clears the other.
                self.assertFalse(dialog.playback.virtual_radio.isChecked())
            finally:
                dialog.close()
                app.processEvents()

        settings = SimpleNamespace(data={"passthrough_playback_mode": "live"})
        dialog = AlphaPassthroughSettingsDialog(i18n, settings)
        try:
            self.assertEqual(dialog.selected_playback_mode(), "live")
            self.assertTrue(dialog.playback.live_radio.isChecked())
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
        patch_stack.enter_context(patch("ui.main_window.cleanup_old_quarantines", return_value=()))
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
            self.assertEqual(window.stack.count(), 11)
            self.assertIs(window.stack.currentWidget(), window.dashboard)

            # Dashboard: server bar and feature cards.
            self.assertFalse(window.dashboard.server_button.icon().isNull())
            self.assertEqual(window.dashboard.server_button.iconSize().width(), 22)
            self.assertIn("https://wapok.com", window.dashboard.project_link.text())
            self.assertIn(window.i18n.t("project.url_label"), window.dashboard.project_link.text())
            self.assertFalse(window.dashboard.project_link.openExternalLinks())
            self.assertEqual(
                set(window.dashboard.cards),
                {"green", "alpha", "alpha2d", "face_beauty", "two_dvr", "superres", "dlss5",
                 "rm", "subtitle", "si", "light"},
            )
            for key, card in window.dashboard.cards.items():
                self.assertTrue(card.title_label.text(), key)
            self.assertTrue(window.dashboard.cards["green"].is_checked())
            self.assertTrue(window.dashboard.cards["alpha"].is_checked())
            self.assertTrue(window.dashboard.cards["alpha2d"].is_checked())
            self.assertTrue(window.dashboard.cards["green"].summary_label.text().startswith("[GREEN]"))
            # The card also states which playback mode the entry is offered in.
            self.assertTrue(
                window.dashboard.cards["alpha"].summary_label.text().startswith("[ALPHA]最好的透视效果")
            )
            self.assertIn(
                window.i18n.t("playback.virtual"),
                window.dashboard.cards["alpha"].summary_label.text(),
            )
            self.assertFalse(window.dashboard.cards["alpha"].help_button.isHidden())
            self.assertTrue(window.dashboard.cards["superres"].summary_label.text().startswith("[SuperRes]"))
            self.assertTrue(window.dashboard.cards["two_dvr"].summary_label.text().startswith("[2D>3D]"))
            self.assertTrue(window.dashboard.cards["si"].summary_label.text().startswith("[SI]"))
            self.assertFalse(window.dashboard.cards["superres"].config_button.isHidden())
            self.assertEqual(window.dashboard._realtime_keys[:3], ["alpha", "green", "superres"])
            self.assertEqual(window.dashboard._2d_keys, ["face_beauty", "two_dvr", "si", "rm"])
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
            # DLSS5, RM and face beauty are all hidden by default, so no group
            # holds a fourth card and the page is three columns wide.
            self.assertEqual(window.dashboard.grid_columns, 3)
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
            self.assertTrue(page.gpu_cache_repair_button.text())
            self.assertTrue(page.gpu_cache_repair_note.text())
            window.settings.data["inference_backend"] = "tensorrt"
            page._update_trt_state()
            self.assertEqual(window.settings.data["inference_backend"], "tensorrt")
            page.set_runtime_provider_kind("cuda")
            self.assertIn(window.i18n.t("trt.runtime_cuda"), page.trt_status_label.text())
            page.performance_output_size.setCurrentIndex(0)
            app.processEvents()
            self.assertEqual(window.settings.data["decode_max_side"], 0)
            self.assertEqual(window.settings.data["inference_backend"], "tensorrt")
            page.set_runtime_provider_kind("cpu")
            self.assertIn(window.i18n.t("trt.runtime_cpu"), page.trt_status_label.text())

            # The feature-debug section is exposed, so the mosaic-restoration
            # card can be shown or hidden from the settings page.
            self.assertFalse(page.debug_group.isHidden())
            self.assertTrue(page.debug_group.title_label.text())
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
            self.assertEqual(window.superres.single_bitrate.currentData(), "auto")
            self.assertEqual(window.superres.batch_bitrate.currentData(), "auto")
            self.assertEqual(window.superres.single_quality_speed.currentData(), "medium")
            self.assertEqual(window.superres.batch_quality_speed.currentData(), "medium")
            common_args = window.superres._common_args(window.superres.single_target, window.superres.single_quality, window.superres.single_hdr_look, window.superres.single_bitrate)
            self.assertIn("p4", common_args)
            self.assertIn("natural", common_args)
            self.assertIn("--rtx-vsr-bitrate-mode", common_args)
            self.assertIn("auto", common_args)
            window._show_page("home")
            app.processEvents()
            self.assertEqual(window.width(), base_size.width())
            self.assertEqual(window.nav.current(), "home")

            window._gpu_repair_pending = True
            window._gpu_repair_rebuild_trt = True
            window._gpu_repair_trt_requested = True
            window._cancel_startup()
            self.assertFalse(window._gpu_repair_pending)
            self.assertFalse(window._gpu_repair_rebuild_trt)
            self.assertFalse(window._gpu_repair_trt_requested)
        finally:
            window.close()
            app.processEvents()
            patch_stack.close()


if __name__ == "__main__":
    unittest.main()
