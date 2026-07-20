"""Standalone offline NVIDIA RTX VSR entry point."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offline.convert import main as convert_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("single or batch command required")
    return convert_main([args[0], "--engine", "rtx_vsr", *args[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
