from __future__ import annotations

from pathlib import Path
import re

from utils.rtx_vsr import runtime_candidates


def test_development_runtime_is_owned_by_models_tree():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "models" / "rtx_vsr" / "runtime"
    assert (runtime / "nvngx_vsr.dll").exists()
    assert (runtime / "pt_rtx_vsr_bridge.dll").exists()
    assert (runtime / "NVIDIA_RTX_Video_SDK_License.pdf").exists()
    assert all("reference" not in str(path).lower() for path in runtime_candidates())


def test_pyinstaller_specs_copy_owned_runtime_tree():
    root = Path(__file__).resolve().parents[1]
    for name in ("pt_core.spec", "VR_Video_Passthrough_Server.spec"):
        text = (root / name).read_text(encoding="utf-8-sig").replace("/", "\\").lower()
        assert re.search(r"models\\+rtx_vsr\\+runtime", text)
