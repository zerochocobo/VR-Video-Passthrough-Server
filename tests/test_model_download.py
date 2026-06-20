from __future__ import annotations

import os
import site
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
        if (plugins / "platforms").exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins / "platforms"))
        if plugins.exists():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))


class FakeI18n:
    language = "en_US"

    def t(self, key: str) -> str:
        values = {
            "modeldl.title": "Download required models",
            "modeldl.intro": "intro",
            "modeldl.checking": "Checking…",
            "modeldl.unknown_size": "size unknown",
            "modeldl.total": "Total: {size}",
            "modeldl.download": "Download",
            "button.cancel": "Cancel",
        }
        return values.get(key, key)


class HfEndpointTests(unittest.TestCase):
    def test_chinese_prefers_mirror_first(self) -> None:
        from utils.hf_download import hf_endpoints

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                hf_endpoints("zh_CN"),
                ["https://hf-mirror.com", "https://huggingface.co"],
            )

    def test_other_language_prefers_hf_first(self) -> None:
        from utils.hf_download import hf_endpoints

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                hf_endpoints("en_US"),
                ["https://huggingface.co", "https://hf-mirror.com"],
            )

    def test_override_wins(self) -> None:
        from utils.hf_download import hf_endpoints

        with patch.dict(os.environ, {"HF_ENDPOINT": "https://example.test/"}, clear=True):
            self.assertEqual(hf_endpoints("zh_CN"), ["https://example.test"])


class NvdsDownloadTests(unittest.TestCase):
    def test_required_filenames_are_split_graphs(self) -> None:
        from offline import nvds_stabilizer as nvds

        self.assertEqual(
            nvds.required_filenames(512, 288),
            ["NVDS_Backbone_512x288.onnx", "NVDS_Head_512x288.onnx"],
        )

    def test_nvds_urls_use_mirror_for_chinese(self) -> None:
        from offline import nvds_stabilizer as nvds

        urls = nvds.download_urls("NVDS_Head_512x288.onnx", "zh_CN")
        self.assertEqual(urls[0], "https://hf-mirror.com/zerochocobo/NVDS_onnx/resolve/main/NVDS_Head_512x288.onnx")


class ModelDownloadDialogTests(unittest.TestCase):
    def _app(self):
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    def test_lists_files_and_total_after_size_probe(self) -> None:
        from ui.widgets.model_download_dialog import DownloadItem, ModelDownloadDialog

        self._app()
        items = [
            DownloadItem(label="a.onnx", dest=Path("a.onnx"), urls=["http://x/a"]),
            DownloadItem(label="b.onnx", dest=Path("b.onnx"), urls=["http://x/b"]),
        ]
        # Avoid real network: the probe thread is irrelevant; drive the slot directly.
        with patch("ui.widgets.model_download_dialog.hf_download.remote_size", return_value=0):
            dialog = ModelDownloadDialog(FakeI18n(), items, None)
        try:
            items[0].size = 10 * 1024 * 1024
            items[1].size = 5 * 1024 * 1024
            dialog._on_sizes_ready(items)
            text = dialog.files_label.text()
            self.assertIn("a.onnx", text)
            self.assertIn("b.onnx", text)
            self.assertIn("15.0 MB", text)  # total
            self.assertTrue(dialog.download_button.isEnabled())
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
