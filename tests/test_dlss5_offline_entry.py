"""The offline DLSS5 entry: CLI wrapper, process command, page and tools card.

The realtime channel stops at 4K, so a 6K or 8K source has to reach the engine
some other way. These cover that path end to end at the seams the UI depends on:
the wrapper pins the engine, main dispatches the frozen subcommand, the page
spells out every NR setting on the command line, and the tools card is the entry
that opens it.
"""

from __future__ import annotations

import os
import site
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import main

# Same Qt bootstrap as tests/test_ui_smoke.py: PySide6 ships its DLLs beside the
# package, and an offscreen platform keeps the widgets headless.
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


class Dlss5OfflineCliTests(unittest.TestCase):
    def test_wrapper_pins_the_dlss5_engine(self) -> None:
        from offline import dlss5_convert

        seen: dict[str, list[str]] = {}

        def fake_convert_main(argv: list[str]) -> int:
            seen["argv"] = argv
            return 0

        with patch.object(dlss5_convert, "convert_main", fake_convert_main):
            self.assertEqual(dlss5_convert.main(["single", "a.mp4", "--dlss5-nr-passes", "2"]), 0)
        self.assertEqual(
            seen["argv"], ["single", "--engine", "dlss5", "a.mp4", "--dlss5-nr-passes", "2"]
        )

    def test_wrapper_requires_a_subcommand(self) -> None:
        from offline import dlss5_convert

        with self.assertRaises(SystemExit):
            dlss5_convert.main([])

    def test_main_dispatches_dlss5_offline_without_starting_server(self) -> None:
        seen: dict[str, list[str]] = {}
        fake = types.ModuleType("offline.dlss5_convert")

        def fake_main(argv: list[str]) -> int:
            seen["argv"] = argv
            return 13

        fake.main = fake_main
        with patch.dict(sys.modules, {"offline.dlss5_convert": fake}):
            self.assertEqual(main.main(["dlss5_offline", "batch", "D:/videos"]), 13)
        self.assertEqual(seen["argv"], ["batch", "D:/videos"])

    def test_process_command_points_at_the_wrapper(self) -> None:
        from ui.services.process_helpers import dlss5_offline_command

        program, args = dlss5_offline_command()
        self.assertTrue(program)
        self.assertEqual(Path(args[-1]).name, "dlss5_convert.py")

    def test_auto_mask_can_be_turned_off_from_the_command_line(self) -> None:
        """The page always states the flag, so the stored "off" has to survive a
        config default of on - which a bare store_true could not express."""
        import config
        from offline.convert import main as convert_main

        with patch.object(config, "DLSS5_AUTO_MASK", True), patch("offline.convert._run_one", return_value=0):
            with patch("sys.argv", ["convert"]):
                self.assertEqual(
                    convert_main(["single", "a.mp4", "--engine", "dlss5", "--no-dlss5-auto-mask"]), 0
                )


class Dlss5OfflinePageTests(unittest.TestCase):
    def setUp(self) -> None:
        from PySide6.QtWidgets import QApplication

        self.app = QApplication.instance() or QApplication([])

    def _page(self, overrides: dict | None = None):
        from types import SimpleNamespace

        from ui.i18n import I18n
        from ui.pages.dlss5_page import Dlss5Page
        from ui.settings import DEFAULTS

        data = dict(DEFAULTS)
        data.update(overrides or {})
        settings = SimpleNamespace(
            data=data,
            save=lambda: None,
            server_env=lambda: {},
        )
        process = SimpleNamespace(
            output=_Signal(),
            state_changed=_Signal(),
            stop=lambda: None,
            started=[],
        )
        process.start = lambda args, env=None: process.started.append((args, env))
        return Dlss5Page(I18n("zh_CN"), settings, process), process

    def test_common_args_spell_out_every_nr_setting(self) -> None:
        from ui.pages.dlss5_page import DLSS5_FLOAT_ARGS, DLSS5_INT_ARGS

        page, _process = self._page({"dlss5_style": 2, "dlss5_nr_passes": 3, "dlss5_intensity": 1.5})
        try:
            args = page._common_args()
            for _key, flag in DLSS5_INT_ARGS + DLSS5_FLOAT_ARGS:
                self.assertIn(flag, args)
            self.assertEqual(args[args.index("--dlss5-style") + 1], "2")
            self.assertEqual(args[args.index("--dlss5-nr-passes") + 1], "3")
            self.assertEqual(args[args.index("--dlss5-intensity") + 1], "1.50")
            # Off is stated, not left to the engine's own default.
            self.assertIn("--no-dlss5-auto-mask", args)
            self.assertIn("--preset", args)
        finally:
            page.deleteLater()

    def test_batch_run_passes_the_directory_and_the_recursion_choice(self) -> None:
        page, process = self._page()
        try:
            page.batch_dir.setText(str(Path(__file__).resolve().parent))
            page.batch_recursive.setChecked(False)
            page.run_batch()
            args, _env = process.started[-1]
            self.assertEqual(args[0], "batch")
            self.assertIn("--no-recursive", args)
            self.assertIn("--skip-existing", args)
        finally:
            page.deleteLater()

    def test_page_summarises_the_settings_the_dialog_owns(self) -> None:
        page, _process = self._page({"dlss5_nr_passes": 4})
        try:
            for label in page._params_summaries:
                self.assertIn("4", label.text())
            self.assertTrue(page._params_buttons)
            for button in page._params_buttons:
                self.assertTrue(button.text())
        finally:
            page.deleteLater()

    def test_tools_page_offers_the_offline_dlss5_card(self) -> None:
        from types import SimpleNamespace

        from ui.i18n import I18n
        from ui.pages.tools_page import ToolsPage

        i18n = I18n("zh_CN")
        settings = SimpleNamespace(data={})
        with patch("ui.pages.tools_page._dlss5_available", return_value=True):
            page = ToolsPage(i18n, settings)
        try:
            self.assertEqual(page.dlss5_card.title_label.text(), i18n.t("dlss5.offline_title"))
            self.assertEqual(page.dlss5_card.desc_label.text(), i18n.t("tools.dlss5_desc"))
            seen: list[bool] = []
            page.open_dlss5.connect(lambda: seen.append(True))
            page.dlss5_card.open_button.click()
            self.assertEqual(seen, [True])
        finally:
            page.deleteLater()

    def test_tools_page_hides_the_card_without_the_runtime(self) -> None:
        from types import SimpleNamespace

        from ui.i18n import I18n
        from ui.pages.tools_page import ToolsPage

        settings = SimpleNamespace(data={})
        with patch("ui.pages.tools_page._dlss5_available", return_value=False):
            page = ToolsPage(I18n("zh_CN"), settings)
        try:
            self.assertTrue(page.dlss5_card.isHidden())
        finally:
            page.deleteLater()


class _Signal:
    """Minimal stand-in for a Qt signal the page only connects to."""

    def connect(self, _slot) -> None:
        return None


if __name__ == "__main__":
    unittest.main()


class Dlss5HiddenTests(unittest.TestCase):
    """DLSS5 is parked: everything stays built, nothing of it is shown."""

    def _settings(self, **overrides):
        from ui.settings import DEFAULTS, Settings

        settings = object.__new__(Settings)
        settings.data = {**DEFAULTS, "mode_green": False, "mode_alpha": False, **overrides}
        return settings

    def test_hidden_by_default(self) -> None:
        from ui.settings import DEFAULTS

        self.assertFalse(DEFAULTS["dlss5_card_visible"])

    def test_a_leftover_switch_does_not_reach_dlna(self) -> None:
        """mode_dlss5 saved as on from before must not list [DLSS5] entries."""
        settings = self._settings(mode_dlss5=True)
        self.assertFalse(settings.dlss5_enabled())
        self.assertNotIn("dlss5", settings.passthrough_mode())

    def test_revealing_it_brings_the_mode_back(self) -> None:
        settings = self._settings(mode_dlss5=True, dlss5_card_visible=True)
        self.assertTrue(settings.dlss5_enabled())
        self.assertIn("dlss5", settings.passthrough_mode())

    def test_the_tools_card_stays_hidden_even_with_the_runtime(self) -> None:
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.i18n import I18n
        from ui.pages.tools_page import ToolsPage

        QApplication.instance() or QApplication([])
        with patch("ui.pages.tools_page._dlss5_available", return_value=True):
            hidden = ToolsPage(I18n("zh_CN"), SimpleNamespace(data={}))
            shown = ToolsPage(I18n("zh_CN"), SimpleNamespace(data={"dlss5_card_visible": True}))
        try:
            self.assertTrue(hidden.dlss5_card.isHidden())
            self.assertFalse(shown.dlss5_card.isHidden())
        finally:
            hidden.deleteLater()
            shown.deleteLater()


class DashboardColumnTests(unittest.TestCase):
    """The grid is as wide as the fullest visible group, never wider."""

    def _page(self, **flags):
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication
        from ui.i18n import I18n
        from ui.pages.dashboard_page import DashboardPage
        from ui.settings import DEFAULTS

        QApplication.instance() or QApplication([])
        settings = SimpleNamespace(data={**DEFAULTS, **flags}, save=lambda: None)
        with patch("utils.dlss5.is_dlss5_available", return_value=True):
            return DashboardPage(I18n("zh_CN"), settings)

    def test_a_default_install_is_three_columns(self) -> None:
        from ui.pages.dashboard_page import DASHBOARD_WIDTHS

        page = self._page()
        try:
            self.assertEqual(page.grid_columns, 3)
            self.assertEqual(page.preferred_width(), DASHBOARD_WIDTHS[3])
            # Every group fits on one row, so no height is added for wrapping.
            self.assertTrue(all(rows == 1 for rows in page._grid_rows.values()))
            self.assertTrue(page.cards["dlss5"].isHidden())
        finally:
            page.deleteLater()

    def test_a_fourth_card_widens_the_grid_instead_of_wrapping(self) -> None:
        from ui.pages.dashboard_page import DASHBOARD_WIDTHS

        page = self._page(rm_card_visible=True, face_beauty_card_visible=True)
        try:
            self.assertEqual(page.grid_columns, 4)
            self.assertEqual(page.preferred_width(), DASHBOARD_WIDTHS[4])
            self.assertTrue(all(rows == 1 for rows in page._grid_rows.values()))
        finally:
            page.deleteLater()

    def test_hiding_the_fourth_card_narrows_it_again_without_a_hole(self) -> None:
        page = self._page(rm_card_visible=True, face_beauty_card_visible=True)
        try:
            page.settings.data["rm_card_visible"] = False
            page.set_rm_card_visible(False)
            self.assertEqual(page.grid_columns, 3)
            # The column that was the fourth must not keep its stretch.
            for grid in (page._realtime_grid, page._2d_grid, page._audio_grid):
                self.assertEqual(grid.columnStretch(3), 0)
        finally:
            page.deleteLater()

    def test_revealing_dlss5_makes_the_realtime_row_four_wide(self) -> None:
        page = self._page(dlss5_card_visible=True)
        try:
            self.assertEqual(page.grid_columns, 4)
            self.assertFalse(page.cards["dlss5"].isHidden())
        finally:
            page.deleteLater()
