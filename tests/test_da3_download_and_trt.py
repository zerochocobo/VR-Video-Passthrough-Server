from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class Da3DownloadAndTrtTests(unittest.TestCase):
    def test_chinese_ui_download_uses_hf_mirror_only(self) -> None:
        from offline import da3_depth

        with patch.dict(os.environ, {"PT_UI_LANGUAGE": "zh_CN"}, clear=True):
            urls = da3_depth.model_download_urls("base_hd")

        self.assertEqual(urls, [
            "https://hf-mirror.com/zerochocobo/DepthAnything3_ONNX/resolve/main/da3_base_1036.onnx"
        ])

    def test_hf_endpoint_override_is_respected(self) -> None:
        from offline import da3_depth

        with patch.dict(os.environ, {"HF_ENDPOINT": "https://example.test/"}, clear=True):
            urls = da3_depth.model_download_urls("base")

        self.assertEqual(urls, [
            "https://example.test/zerochocobo/DepthAnything3_ONNX/resolve/main/da3_base.onnx"
        ])

    def test_trt_cache_is_per_selected_preset(self) -> None:
        from offline import da3_depth

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "runtime_cache" / "da3_trt" / "base"
            cache.mkdir(parents=True)
            (cache / "base.engine").write_bytes(b"engine")

            with patch.object(da3_depth.config, "ROOT", root):
                self.assertTrue(da3_depth.trt_engine_cached("base"))
                self.assertFalse(da3_depth.trt_engine_cached("base_hd"))

    def test_build_trt_both_does_not_download_all_presets_first(self) -> None:
        from offline import two_dvr

        with (
            patch.object(two_dvr, "ensure_model_available", side_effect=AssertionError("should not be called")),
            patch.object(two_dvr, "_build_trt", return_value=0) as build_trt,
        ):
            rc = two_dvr.main(["build-trt", "--model", "both"])

        self.assertEqual(rc, 0)
        build_trt.assert_called_once_with("both")

    def test_startup_warmup_checks_only_selected_preset(self) -> None:
        import onnxruntime as ort
        import main
        from offline import da3_depth

        class FakeLog:
            def info(self, *_args, **_kwargs) -> None:
                pass

            def warning(self, *_args, **_kwargs) -> None:
                pass

        with tempfile.TemporaryDirectory() as raw:
            model_path = Path(raw) / "da3_base.onnx"
            model_path.write_bytes(b"onnx")
            cached_calls = []
            session = SimpleNamespace(run=MagicMock(return_value=[None]))
            fake_engine = SimpleNamespace(
                providers=["TensorrtExecutionProvider"],
                session=session,
                input_name="input",
                output_name="output",
                folded=True,
                size=518,
            )

            def fake_cached(model: str) -> bool:
                self.assertEqual(model, "base")
                cached_calls.append(model)
                return True

            def fake_model_path(model: str) -> Path:
                self.assertEqual(model, "base")
                return model_path

            with (
                patch.object(main, "_passthrough_mode_enabled", return_value=True),
                patch.object(main.config, "TWO_DVR_MODEL", "base"),
                patch.object(ort, "get_available_providers", return_value=["TensorrtExecutionProvider"]),
                patch.object(da3_depth, "trt_engine_cached", side_effect=fake_cached),
                patch.object(da3_depth, "default_model_path", side_effect=fake_model_path),
                patch.object(da3_depth, "_ENGINE_CACHE", {}),
                patch.object(da3_depth, "Da3DepthEngine", return_value=fake_engine) as engine,
                patch.object(main, "set_startup_phase"),
                patch.object(main, "start_heartbeat"),
                patch.object(main, "stop_heartbeat"),
            ):
                main._warmup_da3_trt_if_needed(FakeLog(), step_total=3, provider_kind="trt")

        engine.assert_called_once_with(variant="base", provider="trt")
        session.run.assert_called_once()

    def test_startup_da3_warmup_failure_is_nonfatal(self) -> None:
        import main

        fake_log = SimpleNamespace(warning=MagicMock())

        with (
            patch.object(main, "_warmup_da3_trt_if_needed", side_effect=RuntimeError("boom")) as warmup,
            patch.object(main, "set_startup_phase") as set_phase,
        ):
            main._warmup_da3_trt_nonfatal(fake_log, step_total=4, provider_kind="trt")

        warmup.assert_called_once()
        fake_log.warning.assert_called_once()
        set_phase.assert_called_once()
        self.assertEqual(set_phase.call_args.args[0], "warmed")
        self.assertEqual(set_phase.call_args.kwargs["step"], "da3_trt_warning")


if __name__ == "__main__":
    unittest.main()
